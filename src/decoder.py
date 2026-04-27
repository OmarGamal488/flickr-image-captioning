"""LSTM / GRU decoders for the captioning pipeline.

- ``DecoderLSTM`` — Phase 3 Show-and-Tell baseline (image as first input to LSTM)
- ``DecoderAttention`` — Phase 4 Bahdanau attention decoder, supports both
  LSTMCell and GRUCell via ``rnn_type``. Consumes spatial features
  ``(B, num_pixels, encoder_dim)`` from ``EncoderCNN_Attention``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import BahdanauAttention
from src.vocabulary import Vocabulary


class DecoderLSTM(nn.Module):
    """Show-and-Tell baseline decoder.

    Forward pass (training, teacher forcing):
        features : (B, embed_size)       — projected CNN output
        captions : (B, T)                — full caption incl. <start>...<end>

    At step 0 the LSTM sees the image features; at step t>=1 it sees the
    embedding of the ground-truth previous token ``captions[:, t-1]``. The
    output at step t is then trained to predict ``captions[:, t]``, so:

        logits  : (B, T, vocab_size)
        targets : captions  (same shape; <pad>=0 ignored in CE loss)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=Vocabulary.PAD_IDX)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        # Drop the last caption token so inputs and targets have the same length
        # after we prepend ``features`` as step 0.
        # inputs[t=0] = features
        # inputs[t>=1] = embedding(captions[:, t-1])
        embeddings = self.embedding(captions[:, :-1])                    # (B, T-1, E)
        inputs = torch.cat([features.unsqueeze(1), embeddings], dim=1)   # (B, T, E)
        hiddens, _ = self.lstm(inputs)                                   # (B, T, H)
        logits = self.fc(self.dropout(hiddens))                          # (B, T, V)
        return logits

    @torch.no_grad()
    def generate_greedy(
        self,
        features: torch.Tensor,
        max_len: int = 20,
        end_idx: int = Vocabulary.END_IDX,
    ) -> list[list[int]]:
        """Greedy decoding. ``features`` shape: (B, embed_size). Returns per-sample id lists."""
        self.eval()
        B = features.size(0)
        device = features.device
        states: tuple[torch.Tensor, torch.Tensor] | None = None
        inputs = features.unsqueeze(1)                                   # (B, 1, E)

        generated = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            out, states = self.lstm(inputs, states)                      # (B, 1, H)
            logits = self.fc(out.squeeze(1))                             # (B, V)
            tokens = logits.argmax(dim=-1)                               # (B,)

            for i in range(B):
                if not finished[i]:
                    tok = int(tokens[i].item())
                    generated[i].append(tok)
                    if tok == end_idx:
                        finished[i] = True

            if bool(finished.all()):
                break
            inputs = self.embedding(tokens).unsqueeze(1)                 # (B, 1, E)

        return generated


class DecoderAttention(nn.Module):
    """Bahdanau-attention decoder.

    Accepts spatial encoder features ``(B, num_pixels, encoder_dim)`` from
    ``EncoderCNN_Attention``. At each decoder step it:

        1. Computes attention over pixels conditioned on the previous hidden state
        2. Forms a context vector (weighted sum of pixel features)
        3. Concatenates the context with the current word embedding
        4. Feeds into an ``LSTMCell`` or ``GRUCell`` (per-step, so attention can
           be recomputed every timestep)
        5. Projects the output to vocabulary logits

    Forward returns both ``logits`` (for CrossEntropyLoss) and ``alphas``
    (for doubly-stochastic regularization + heatmap visualization).
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 2048,
        embed_size: int = 256,
        hidden_size: int = 512,
        attention_dim: int = 256,
        dropout: float = 0.5,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        assert rnn_type in {"lstm", "gru"}, f"rnn_type must be lstm or gru, got {rnn_type}"
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.rnn_type = rnn_type

        self.attention = BahdanauAttention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=Vocabulary.PAD_IDX)

        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size) if rnn_type == "lstm" else None

        rnn_input = embed_size + encoder_dim
        self.rnn_cell: nn.Module = (
            nn.LSTMCell(rnn_input, hidden_size)
            if rnn_type == "lstm"
            else nn.GRUCell(rnn_input, hidden_size)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_states(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean_feat = features.mean(dim=1)                                 # (B, enc_dim)
        h = torch.tanh(self.init_h(mean_feat))
        c = torch.tanh(self.init_c(mean_feat)) if self.rnn_type == "lstm" else None
        return h, c

    def forward(
        self, features: torch.Tensor, captions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features : (B, P, enc_dim)
        # captions : (B, T) with <start>...<end>
        # At step t: feed embedding of captions[:, t] (previous token), predict captions[:, t+1].
        # Loop runs T-1 steps; logits[:, t] is trained against captions[:, t+1] in the loss.
        B, T = captions.shape
        T_out = T - 1

        h, c = self.init_states(features)
        embeddings = self.embedding(captions[:, :-1])                    # (B, T-1, E)

        logits = features.new_zeros(B, T_out, self.vocab_size)
        alphas = features.new_zeros(B, T_out, features.size(1))

        for t in range(T_out):
            context, alpha = self.attention(features, h)                 # (B, enc_dim), (B, P)
            inp = torch.cat([embeddings[:, t], context], dim=1)           # (B, E+enc_dim)
            if self.rnn_type == "lstm":
                h, c = self.rnn_cell(inp, (h, c))
            else:
                h = self.rnn_cell(inp, h)
            logits[:, t] = self.fc(self.dropout(h))
            alphas[:, t] = alpha

        return logits, alphas

    @torch.no_grad()
    def generate_greedy(
        self,
        features: torch.Tensor,
        max_len: int = 20,
        start_idx: int = Vocabulary.START_IDX,
        end_idx: int = Vocabulary.END_IDX,
        return_alphas: bool = False,
    ) -> list[list[int]] | tuple[list[list[int]], list[list[list[float]]]]:
        """Greedy attention decoding. ``features`` shape: (B, P, enc_dim)."""
        self.eval()
        B = features.size(0)
        device = features.device

        h, c = self.init_states(features)
        tokens = torch.full((B,), start_idx, dtype=torch.long, device=device)

        generated: list[list[int]] = [[] for _ in range(B)]
        per_step_alphas: list[list[list[float]]] = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.embedding(tokens)                                 # (B, E)
            context, alpha = self.attention(features, h)                 # (B, enc_dim), (B, P)
            inp = torch.cat([emb, context], dim=1)
            if self.rnn_type == "lstm":
                h, c = self.rnn_cell(inp, (h, c))
            else:
                h = self.rnn_cell(inp, h)
            logits = self.fc(h)                                          # (B, V)
            tokens = logits.argmax(dim=-1)                               # (B,)

            for i in range(B):
                if finished[i]:
                    continue
                tok = int(tokens[i].item())
                generated[i].append(tok)
                if return_alphas:
                    per_step_alphas[i].append(alpha[i].detach().cpu().tolist())
                if tok == end_idx:
                    finished[i] = True

            if bool(finished.all()):
                break

        if return_alphas:
            return generated, per_step_alphas
        return generated

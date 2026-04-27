"""Inference helpers for the attention captioning model.

- ``load_attention_model``       : load an EncoderCNN_Attention + DecoderAttention
  checkpoint and return ``(encoder, decoder, config)``.
- ``encode_image``               : load a raw PIL/path image and apply the eval transform.
- ``generate_greedy``            : greedy decode → caption string.
- ``generate_beam``              : beam search with length normalization + repetition
  penalty → ``(best_caption, top_k_alternatives)``.
- ``caption_image``              : one-shot API used by the FastAPI / Gradio demo.

Beam search returns the single best caption *and* the full top-k list so the
FastAPI endpoint can expose "alternative captions" at no extra cost.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
from PIL import Image
from torch import nn

from src.dataset import build_eval_transform
from src.decoder import DecoderAttention
from src.encoder import EncoderCNN_Attention
from src.utils import AttentionConfig
from src.vocabulary import Vocabulary


@dataclass
class BeamResult:
    caption: str
    score: float
    token_ids: list[int]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_attention_model(
    ckpt_path: str,
    vocab_size: int,
    device: torch.device,
) -> tuple[nn.Module, DecoderAttention, AttentionConfig]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = AttentionConfig(**ckpt["config"])

    encoder = EncoderCNN_Attention(pretrained=False, freeze=True).to(device)
    decoder = DecoderAttention(
        vocab_size=vocab_size,
        encoder_dim=cfg.encoder_dim,
        embed_size=cfg.embed_size,
        hidden_size=cfg.hidden_size,
        attention_dim=cfg.attention_dim,
        dropout=cfg.dropout,
        rnn_type=cfg.rnn_type,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, cfg


def encode_image(
    image: str | Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """PIL or path → (1, 3, 224, 224) ImageNet-normalized tensor on ``device``."""
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")
    return build_eval_transform()(img).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Greedy decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_greedy(
    encoder: nn.Module,
    decoder: DecoderAttention,
    image_tensor: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = 20,
) -> str:
    encoder.eval()
    decoder.eval()
    features = encoder(image_tensor)                                # (1, P, enc_dim)
    ids = decoder.generate_greedy(features, max_len=max_len)[0]     # list[int]
    return vocab.denumericalize(ids)


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_beam(
    encoder: nn.Module,
    decoder: DecoderAttention,
    image_tensor: torch.Tensor,
    vocab: Vocabulary,
    beam_width: int = 5,
    max_len: int = 20,
    length_alpha: float = 0.7,
    repetition_penalty: float = 1.2,
) -> list[BeamResult]:
    """Beam search with length normalization + repetition penalty.

    Standard shrinking-beam implementation:
      * Start with a single alive candidate: the ``<start>`` token.
      * Each step, expand every alive candidate to its top-``beam_width`` next
        tokens, then keep the overall top-``k`` by cumulative log-prob where
        ``k = beam_width - len(completed)``.
      * Candidates that emit ``<end>`` move to the ``completed`` pool with a
        length-normalized score and no longer occupy a beam slot.
      * Repetition penalty divides the log-prob of any token already in the
        candidate's sequence by ``repetition_penalty``.
    """
    encoder.eval()
    decoder.eval()
    device = image_tensor.device
    assert image_tensor.size(0) == 1, "beam search is per-image"

    features = encoder(image_tensor)                                 # (1, P, enc_dim)
    h0, c0 = decoder.init_states(features)                            # (1, H), optional

    # Each candidate: (cum_log_prob, token_ids, h, c)
    # ``token_ids`` is a plain python list to avoid building new tensors each step.
    alive = [
        {
            "score": 0.0,
            "tokens": [Vocabulary.START_IDX],
            "h": h0,
            "c": c0,  # None for GRU
        }
    ]
    completed: list[BeamResult] = []

    def _finalize(cand: dict) -> BeamResult:
        token_ids = cand["tokens"][1:]  # strip <start>
        length = max(len(token_ids), 1)
        norm = cand["score"] / (length ** length_alpha)
        return BeamResult(
            caption=vocab.denumericalize(token_ids),
            score=norm,
            token_ids=token_ids,
        )

    for step in range(max_len):
        if not alive:
            break

        # Batch all alive candidates through one decoder step.
        last_tokens = torch.tensor(
            [cand["tokens"][-1] for cand in alive], device=device, dtype=torch.long
        )                                                             # (n_alive,)
        h_batch = torch.cat([cand["h"] for cand in alive], dim=0)     # (n_alive, H)
        if decoder.rnn_type == "lstm":
            c_batch = torch.cat([cand["c"] for cand in alive], dim=0)  # (n_alive, H)

        emb = decoder.embedding(last_tokens)                          # (n_alive, E)
        features_batch = features.expand(len(alive), -1, -1)          # (n_alive, P, enc_dim)
        context, _ = decoder.attention(features_batch, h_batch)
        inp = torch.cat([emb, context], dim=1)
        if decoder.rnn_type == "lstm":
            h_new, c_new = decoder.rnn_cell(inp, (h_batch, c_batch))
        else:
            h_new = decoder.rnn_cell(inp, h_batch)
            c_new = None
        logits = decoder.fc(h_new)                                    # (n_alive, V)
        log_probs = torch.log_softmax(logits, dim=-1)                 # (n_alive, V)

        # Repetition penalty (per alive candidate).
        if repetition_penalty != 1.0:
            for i, cand in enumerate(alive):
                for tok in cand["tokens"]:
                    if tok >= 4:  # skip special tokens
                        log_probs[i, tok] /= repetition_penalty

        # For each alive candidate, collect its top-k children.
        k = beam_width
        topk_vals, topk_idx = log_probs.topk(k, dim=-1)               # (n_alive, k)

        children: list[dict] = []
        for i, cand in enumerate(alive):
            cand_h = h_new[i : i + 1]
            cand_c = c_new[i : i + 1] if c_new is not None else None
            for j in range(k):
                tok = int(topk_idx[i, j].item())
                logp = float(topk_vals[i, j].item())
                children.append(
                    {
                        "score": cand["score"] + logp,
                        "tokens": cand["tokens"] + [tok],
                        "h": cand_h,
                        "c": cand_c,
                    }
                )

        # Keep only the overall best (beam_width - len(completed)) children.
        remaining_slots = beam_width - len(completed)
        if remaining_slots <= 0:
            break
        children.sort(key=lambda c: -c["score"])
        children = children[:remaining_slots]

        # Separate finished from still-running.
        new_alive: list[dict] = []
        for cand in children:
            if cand["tokens"][-1] == Vocabulary.END_IDX:
                completed.append(_finalize(cand))
            else:
                new_alive.append(cand)
        alive = new_alive

    # Finalize any candidates still alive at the step limit.
    for cand in alive:
        completed.append(_finalize(cand))

    if not completed:
        return [BeamResult("", -float("inf"), [])]

    completed.sort(key=lambda r: -r.score)
    return completed[:beam_width]


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


@torch.no_grad()
def caption_image(
    encoder: nn.Module,
    decoder: DecoderAttention,
    image: str | Image.Image,
    vocab: Vocabulary,
    device: torch.device,
    method: str = "beam",
    beam_width: int = 5,
    max_len: int = 20,
) -> tuple[str, list[BeamResult] | None]:
    """Convenience wrapper. Returns (best_caption, beam_results_or_None)."""
    tensor = encode_image(image, device)
    if method == "greedy":
        return generate_greedy(encoder, decoder, tensor, vocab, max_len=max_len), None
    if method == "beam":
        beams = generate_beam(
            encoder, decoder, tensor, vocab, beam_width=beam_width, max_len=max_len
        )
        return beams[0].caption, beams
    raise ValueError(f"method must be 'greedy' or 'beam', got {method!r}")


@torch.no_grad()
def batch_generate(
    encoder: nn.Module,
    decoder: DecoderAttention,
    image_ids: Iterable[str],
    images_dir: str,
    vocab: Vocabulary,
    device: torch.device,
    method: str = "greedy",
    beam_width: int = 5,
    max_len: int = 20,
    batch_size: int = 32,
) -> dict[str, str]:
    """Generate captions for a list of image filenames.

    Greedy uses real batching (fast). Beam falls back to per-image loop.
    Returns {image_id: caption_string}.
    """
    image_ids = list(image_ids)
    out: dict[str, str] = {}
    transform = build_eval_transform()

    if method == "greedy":
        for i in range(0, len(image_ids), batch_size):
            chunk = image_ids[i : i + batch_size]
            tensors = []
            for img_id in chunk:
                with Image.open(os.path.join(images_dir, img_id)) as img:
                    tensors.append(transform(img.convert("RGB")))
            batch = torch.stack(tensors).to(device, non_blocking=True)
            features = encoder(batch)
            gen = decoder.generate_greedy(features, max_len=max_len)
            for img_id, ids in zip(chunk, gen):
                out[img_id] = vocab.denumericalize(ids)
        return out

    # beam search, per-image
    for img_id in image_ids:
        with Image.open(os.path.join(images_dir, img_id)) as img:
            tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
        beams = generate_beam(
            encoder, decoder, tensor, vocab, beam_width=beam_width, max_len=max_len
        )
        out[img_id] = beams[0].caption
    return out

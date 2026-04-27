"""Bahdanau (additive) attention over a spatial CNN feature map.

Given encoder features of shape (B, num_pixels, encoder_dim) and a decoder
hidden state of shape (B, decoder_dim), produces:

  context  — (B, encoder_dim)  weighted sum of features
  alpha    — (B, num_pixels)   attention weights (sum to 1 across pixels)

This is the single biggest architectural improvement over the Show-and-Tell
baseline: it lets the decoder look at different spatial regions for each word.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, attention_dim)
        self.W_dec = nn.Linear(decoder_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)

    def forward(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features : (B, P, enc_dim)
        # hidden   : (B, dec_dim)
        att_enc = self.W_enc(features)                       # (B, P, A)
        att_dec = self.W_dec(hidden).unsqueeze(1)            # (B, 1, A)
        scores = self.V(torch.tanh(att_enc + att_dec)).squeeze(-1)  # (B, P)
        alpha = F.softmax(scores, dim=1)                     # (B, P)
        context = (features * alpha.unsqueeze(-1)).sum(dim=1)  # (B, enc_dim)
        return context, alpha

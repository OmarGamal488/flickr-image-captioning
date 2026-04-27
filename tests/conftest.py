"""Shared pytest fixtures.

Tests run on CPU with synthetic inputs — no GPU, no trained checkpoint, no
images on disk required. The fixtures here build a tiny in-memory vocab
and a 4×4 image batch so every test takes well under a second.
"""

from __future__ import annotations

import pytest
import torch

from src.vocabulary import Vocabulary


@pytest.fixture(scope="session")
def tiny_vocab() -> Vocabulary:
    v = Vocabulary(freq_threshold=1)
    v.build_from_captions([
        "a dog runs on the grass",
        "a man with a dog",
        "two children playing in a park",
        "a red car on the road",
    ])
    return v


@pytest.fixture
def fake_images() -> torch.Tensor:
    """(B=2, 3, 224, 224) zero-mean, unit-variance synthetic input."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def fake_caption_ids(tiny_vocab: Vocabulary) -> torch.Tensor:
    """A small (B=2, T=8) caption-id batch with <pad> in the trailing slots."""
    seqs = [
        tiny_vocab.numericalize("a dog runs on the grass"),
        tiny_vocab.numericalize("a man with a dog"),
    ]
    max_len = max(len(s) for s in seqs)
    padded = [s + [Vocabulary.PAD_IDX] * (max_len - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long)

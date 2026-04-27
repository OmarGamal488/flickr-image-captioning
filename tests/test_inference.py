"""Greedy + beam decoding correctness on a freshly-built tiny model.

These tests don't need a trained checkpoint — they wire up a small randomly
initialized encoder/decoder pair and just verify the decoding *plumbing* (output
types, ordering, termination, repetition penalty) so they run on CPU in <2s.
"""

from __future__ import annotations

import torch
from PIL import Image

from src.decoder import DecoderAttention
from src.encoder import EncoderCNN_Attention
from src.inference import (
    BeamResult,
    caption_image,
    encode_image,
    generate_beam,
    generate_greedy,
)
from src.vocabulary import Vocabulary


def _build_tiny_pipeline(vocab: Vocabulary) -> tuple[EncoderCNN_Attention, DecoderAttention]:
    encoder = EncoderCNN_Attention(pretrained=False, freeze=True).eval()
    decoder = DecoderAttention(
        vocab_size=len(vocab),
        encoder_dim=2048,
        embed_size=64,
        hidden_size=128,
        attention_dim=64,
        dropout=0.0,
        rnn_type="lstm",
    ).eval()
    return encoder, decoder


def test_encode_image_returns_normalized_batch():
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    tensor = encode_image(img, device=torch.device("cpu"))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_generate_greedy_returns_string(tiny_vocab):
    encoder, decoder = _build_tiny_pipeline(tiny_vocab)
    image_tensor = torch.randn(1, 3, 224, 224)
    caption = generate_greedy(encoder, decoder, image_tensor, tiny_vocab, max_len=10)
    assert isinstance(caption, str)
    # No special tokens leak into the rendered string
    assert Vocabulary.START_TOKEN not in caption
    assert Vocabulary.END_TOKEN not in caption
    assert Vocabulary.PAD_TOKEN not in caption


def test_generate_beam_returns_sorted_results(tiny_vocab):
    encoder, decoder = _build_tiny_pipeline(tiny_vocab)
    image_tensor = torch.randn(1, 3, 224, 224)
    beams = generate_beam(
        encoder, decoder, image_tensor, tiny_vocab,
        beam_width=3, max_len=10, repetition_penalty=1.2,
    )
    assert 1 <= len(beams) <= 3
    for b in beams:
        assert isinstance(b, BeamResult)
        assert isinstance(b.caption, str)
        assert isinstance(b.token_ids, list)
    # Best-first ordering
    scores = [b.score for b in beams]
    assert scores == sorted(scores, reverse=True)


def test_caption_image_dispatches_methods(tiny_vocab):
    encoder, decoder = _build_tiny_pipeline(tiny_vocab)
    img = Image.new("RGB", (320, 240), color=(50, 50, 50))
    device = torch.device("cpu")

    cap_g, beams_g = caption_image(encoder, decoder, img, tiny_vocab, device, method="greedy")
    assert isinstance(cap_g, str)
    assert beams_g is None

    cap_b, beams_b = caption_image(encoder, decoder, img, tiny_vocab, device, method="beam", beam_width=2)
    assert isinstance(cap_b, str)
    assert beams_b is not None and len(beams_b) >= 1
    assert beams_b[0].caption == cap_b


def test_caption_image_rejects_unknown_method(tiny_vocab):
    encoder, decoder = _build_tiny_pipeline(tiny_vocab)
    img = Image.new("RGB", (32, 32))
    try:
        caption_image(encoder, decoder, img, tiny_vocab, torch.device("cpu"), method="nucleus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown decoding method")

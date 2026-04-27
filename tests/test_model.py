"""Encoder + decoder forward-pass shape checks (CPU, randomly-initialized)."""

from __future__ import annotations

import torch

from src.attention import BahdanauAttention
from src.decoder import DecoderAttention, DecoderLSTM
from src.encoder import EncoderCNN, EncoderCNN_Attention


# ----- Encoders -------------------------------------------------------------


def test_encoder_cnn_baseline_output_shape(fake_images):
    enc = EncoderCNN(embed_size=128, pretrained=False, freeze=True).eval()
    out = enc(fake_images)
    assert out.shape == (fake_images.size(0), 128)


def test_encoder_cnn_attention_returns_spatial_map(fake_images):
    enc = EncoderCNN_Attention(pretrained=False, freeze=True).eval()
    feats = enc(fake_images)
    # ResNet50 + 224 input → 7×7 spatial × 2048 channels
    assert feats.shape == (fake_images.size(0), 49, 2048)
    assert enc.encoder_dim == 2048


# ----- Bahdanau attention ---------------------------------------------------


def test_attention_alpha_is_a_probability_distribution():
    attn = BahdanauAttention(encoder_dim=2048, decoder_dim=512, attention_dim=256)
    feats = torch.randn(2, 49, 2048)
    h_prev = torch.randn(2, 512)
    context, alpha = attn(feats, h_prev)
    assert context.shape == (2, 2048)
    assert alpha.shape == (2, 49)
    # Each row of alpha should sum to ~1 (softmax over 49 locations)
    sums = alpha.sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)


# ----- Decoders -------------------------------------------------------------


def test_decoder_lstm_baseline_output_shape(tiny_vocab, fake_caption_ids):
    dec = DecoderLSTM(
        vocab_size=len(tiny_vocab),
        embed_size=128,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
    )
    feats = torch.randn(fake_caption_ids.size(0), 128)
    logits = dec(feats, fake_caption_ids)
    # DecoderLSTM prepends features as step 0, so logits has length T (matches caption).
    B, T = fake_caption_ids.shape
    assert logits.shape == (B, T, len(tiny_vocab))


def test_decoder_attention_forward_returns_logits_and_alphas(tiny_vocab, fake_caption_ids):
    dec = DecoderAttention(
        vocab_size=len(tiny_vocab),
        encoder_dim=2048,
        embed_size=128,
        hidden_size=256,
        attention_dim=128,
        dropout=0.0,
        rnn_type="lstm",
    )
    feats = torch.randn(fake_caption_ids.size(0), 49, 2048)
    logits, alphas = dec(feats, fake_caption_ids)
    B, T = fake_caption_ids.shape
    assert logits.shape == (B, T - 1, len(tiny_vocab))
    assert alphas.shape == (B, T - 1, 49)
    # Each step's alpha is a softmax over the 49 locations
    assert torch.allclose(alphas.sum(dim=2), torch.ones(B, T - 1), atol=1e-5)


def test_decoder_attention_gru_variant_runs(tiny_vocab, fake_caption_ids):
    dec = DecoderAttention(
        vocab_size=len(tiny_vocab),
        encoder_dim=2048,
        embed_size=128,
        hidden_size=256,
        attention_dim=128,
        dropout=0.0,
        rnn_type="gru",
    )
    feats = torch.randn(fake_caption_ids.size(0), 49, 2048)
    logits, alphas = dec(feats, fake_caption_ids)
    B, T = fake_caption_ids.shape
    assert logits.shape == (B, T - 1, len(tiny_vocab))


def test_decoder_attention_gradient_flows(tiny_vocab, fake_caption_ids):
    """One backward pass must update at least the embedding and the output linear."""
    dec = DecoderAttention(
        vocab_size=len(tiny_vocab),
        encoder_dim=2048,
        embed_size=128,
        hidden_size=256,
        attention_dim=128,
        dropout=0.0,
        rnn_type="lstm",
    )
    feats = torch.randn(fake_caption_ids.size(0), 49, 2048)
    logits, _ = dec(feats, fake_caption_ids)
    targets = fake_caption_ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=0,
    )
    loss.backward()
    assert dec.embedding.weight.grad is not None
    assert dec.fc.weight.grad is not None
    assert dec.embedding.weight.grad.abs().sum().item() > 0

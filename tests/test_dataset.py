"""Vocabulary + dataset + collate_fn correctness."""

from __future__ import annotations

import torch

from src.dataset import (
    CaptionItem,
    build_eval_transform,
    build_train_transform,
    collate_fn,
)
from src.vocabulary import Vocabulary


# ----- Vocabulary -----------------------------------------------------------


def test_vocabulary_specials_have_fixed_indices():
    v = Vocabulary(freq_threshold=1)
    v.build_from_captions(["a dog"])
    assert v.stoi[Vocabulary.PAD_TOKEN] == 0
    assert v.stoi[Vocabulary.START_TOKEN] == 1
    assert v.stoi[Vocabulary.END_TOKEN] == 2
    assert v.stoi[Vocabulary.UNK_TOKEN] == 3


def test_vocabulary_round_trip(tiny_vocab):
    text = "a dog runs"
    ids = tiny_vocab.numericalize(text)
    # numericalize wraps with <start> ... <end>
    assert ids[0] == Vocabulary.START_IDX
    assert ids[-1] == Vocabulary.END_IDX
    decoded = tiny_vocab.denumericalize(ids)
    assert decoded == text


def test_vocabulary_unk_for_oov(tiny_vocab):
    ids = tiny_vocab.numericalize("a quokka jumps")  # 'quokka' is OOV
    decoded_tokens = tiny_vocab.denumericalize(ids).split()
    assert Vocabulary.UNK_TOKEN in decoded_tokens or "quokka" not in decoded_tokens


def test_vocabulary_frequency_threshold_drops_rare_words():
    v = Vocabulary(freq_threshold=2)
    v.build_from_captions(["the cat naps", "the cat sleeps"])
    # 'the' and 'cat' appear twice; 'naps' and 'sleeps' once → dropped
    assert "the" in v.stoi
    assert "cat" in v.stoi
    assert "naps" not in v.stoi
    assert "sleeps" not in v.stoi


# ----- Transforms -----------------------------------------------------------


def test_eval_transform_produces_normalized_3x224x224():
    from PIL import Image
    img = Image.new("RGB", (640, 480), color=(120, 200, 80))
    out = build_eval_transform()(img)
    assert out.shape == (3, 224, 224)
    # ImageNet normalization can produce negative values; check it's not in [0, 1]
    assert (out < 0).any().item()


def test_train_transform_outputs_match_eval_shape():
    from PIL import Image
    img = Image.new("RGB", (640, 480), color=(120, 200, 80))
    out = build_train_transform()(img)
    assert out.shape == (3, 224, 224)


# ----- Dataset + collate_fn -------------------------------------------------


def test_collate_fn_pads_to_longest_caption():
    a = (torch.zeros(3, 224, 224), torch.tensor([1, 5, 6, 2], dtype=torch.long))
    b = (torch.zeros(3, 224, 224), torch.tensor([1, 7, 2], dtype=torch.long))
    images, captions, lengths = collate_fn([a, b])
    assert images.shape == (2, 3, 224, 224)
    assert captions.shape == (2, 4)
    # Second caption is shorter → trailing column is <pad>
    assert captions[1, 3].item() == Vocabulary.PAD_IDX
    assert lengths.tolist() == [4, 3]


def test_caption_item_dataclass_round_trip():
    item = CaptionItem(image_file="img.jpg", caption_ids=[1, 2, 3])
    assert item.image_file == "img.jpg"
    assert item.caption_ids == [1, 2, 3]

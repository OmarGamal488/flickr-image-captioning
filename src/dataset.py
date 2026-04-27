"""Flickr8k dataset + DataLoader helpers.

Captions are pre-numericalized at construction time so __getitem__ is cheap.
Batches use dynamic padding via ``collate_fn`` — no global max_length needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

from src.vocabulary import Vocabulary

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@dataclass
class CaptionItem:
    image_file: str
    caption_ids: list[int]


class FlickrDataset(Dataset):
    """One row = one (image, caption) pair.

    Since each image has 5 captions, the dataset length is 5 * num_images for
    the train split (all 5 are seen each epoch). For val/test, by default only
    the first caption is used during the per-step BLEU probe; full evaluation
    compares against all 5 references separately in ``evaluate.py``.
    """

    def __init__(
        self,
        items: list[CaptionItem],
        image_dir: str,
        transform: transforms.Compose,
    ) -> None:
        self.items = items
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        with Image.open(os.path.join(self.image_dir, item.image_file)) as img:
            image = self.transform(img.convert("RGB"))
        caption = torch.tensor(item.caption_ids, dtype=torch.long)
        return image, caption


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Pad captions to the longest in the batch. Returns (images, captions, lengths)."""
    images, captions = zip(*batch)
    images_tensor = torch.stack(images, dim=0)
    lengths = torch.tensor([c.size(0) for c in captions], dtype=torch.long)
    captions_padded = pad_sequence(
        list(captions), batch_first=True, padding_value=Vocabulary.PAD_IDX
    )
    return images_tensor, captions_padded, lengths


def build_items(
    image_ids: list[str], captions_by_image: dict[str, list[str]], vocab: Vocabulary
) -> list[CaptionItem]:
    """Flatten (image, [5 captions]) → list of (image, caption_ids) rows."""
    out: list[CaptionItem] = []
    for img_id in image_ids:
        for cap in captions_by_image.get(img_id, []):
            out.append(CaptionItem(image_file=img_id, caption_ids=vocab.numericalize(cap)))
    return out

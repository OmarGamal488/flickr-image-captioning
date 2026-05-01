"""Flickr dataset + DataLoader helpers.

Supports Flickr8k and Flickr30k — auto-detected from the raw data directory.
Captions are pre-numericalized at construction time so __getitem__ is cheap.
Batches use dynamic padding via ``collate_fn`` — no global max_length needed.
"""

from __future__ import annotations

import os
import pickle
import random
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


# ---------------------------------------------------------------------------
# Auto-detection + preprocessing (Flickr8k and Flickr30k)
# ---------------------------------------------------------------------------


def detect_dataset(raw_dir: str) -> tuple[str, str]:
    """Return ``(dataset_type, images_dir)`` by inspecting ``raw_dir``.

    Flickr8k:  raw_dir/captions.txt  + raw_dir/Images/
    Flickr30k: raw_dir/results.csv   + raw_dir/flickr30k_images/
    """
    if os.path.isfile(os.path.join(raw_dir, "captions.txt")):
        return "flickr8k", os.path.join(raw_dir, "Images")
    if os.path.isfile(os.path.join(raw_dir, "results.csv")):
        return "flickr30k", os.path.join(raw_dir, "flickr30k_images")
    raise FileNotFoundError(
        f"Could not detect dataset in {raw_dir!r}. "
        "Expected captions.txt (Flickr8k) or results.csv (Flickr30k)."
    )


def _read_captions_flickr8k(raw_dir: str) -> dict[str, list[str]]:
    captions: dict[str, list[str]] = {}
    with open(os.path.join(raw_dir, "captions.txt"), encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            img, cap = line.split(",", 1)
            captions.setdefault(img.strip(), []).append(cap.strip())
    return captions


def _read_captions_flickr30k(raw_dir: str) -> dict[str, list[str]]:
    captions: dict[str, list[str]] = {}
    with open(os.path.join(raw_dir, "results.csv"), encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            captions.setdefault(parts[0].strip(), []).append(parts[2].strip())
    return captions


def build_merged_raw(out_dir: str = "data/raw_merged") -> None:
    """Combine Flickr8k + Flickr30k into a single raw directory.

    Downloads both datasets via kagglehub if not already cached, then creates:
        out_dir/Images/      — symlinks to every image from both datasets
        out_dir/captions.txt — merged captions in Flickr8k CSV format

    Safe to call multiple times — skips if out_dir already exists.
    """
    if os.path.isfile(os.path.join(out_dir, "captions.txt")):
        print(f"[merge] {out_dir!r} already exists — skipping")
        return

    import kagglehub
    print("[merge] resolving dataset paths via kagglehub ...")
    flickr8k_dir  = kagglehub.dataset_download("adityajn105/flickr8k")
    flickr30k_dir = os.path.join(
        kagglehub.dataset_download("hsankesara/flickr-image-dataset"),
        "flickr30k_images",
    )

    images_out = os.path.join(out_dir, "Images")
    os.makedirs(images_out, exist_ok=True)

    # Symlink Flickr8k images
    flickr8k_images = os.path.join(flickr8k_dir, "Images")
    for img in os.listdir(flickr8k_images):
        dst = os.path.join(images_out, img)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(flickr8k_images, img)), dst)

    # Symlink Flickr30k images
    flickr30k_images = os.path.join(flickr30k_dir, "flickr30k_images")
    for img in os.listdir(flickr30k_images):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            dst = os.path.join(images_out, img)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(os.path.join(flickr30k_images, img)), dst)

    # Read both caption sets
    caps8k  = _read_captions_flickr8k(flickr8k_dir)
    caps30k = _read_captions_flickr30k(flickr30k_dir)
    merged  = {**caps8k, **caps30k}

    # Write merged captions.txt in Flickr8k format
    with open(os.path.join(out_dir, "captions.txt"), "w", encoding="utf-8") as f:
        f.write("image,caption\n")
        for img, caps in merged.items():
            for cap in caps:
                f.write(f"{img},{cap}\n")

    print(
        f"[merge] {len(caps8k)} Flickr8k + {len(caps30k)} Flickr30k "
        f"= {len(merged)} total images → {out_dir!r}"
    )


def build_processed_data(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    freq_threshold: int = 5,
    seed: int = 42,
) -> str:
    """Read raw captions, build splits + vocabulary, save pkl files.

    Returns the detected ``images_dir`` so the caller can update its config.
    Safe to call multiple times — skips if all pkl files already exist.
    """
    needed = [
        os.path.join(out_dir, f)
        for f in ("vocab.pkl", "train_data.pkl", "val_data.pkl", "test_data.pkl")
    ]
    if all(os.path.isfile(p) for p in needed):
        dataset_type, images_dir = detect_dataset(raw_dir)
        print(f"[preprocess] processed data already exists in {out_dir!r} — skipping build")
        return images_dir

    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    dataset_type, images_dir = detect_dataset(raw_dir)
    print(f"[preprocess] detected {dataset_type} — images_dir={images_dir!r}")

    if dataset_type == "flickr8k":
        captions = _read_captions_flickr8k(raw_dir)
        n_train, n_val = 6000, 1000
    else:
        captions = _read_captions_flickr30k(raw_dir)
        n_train, n_val = 29783, 1000

    # Keep only images with at least 5 captions
    all_images = sorted(img for img, caps in captions.items() if len(caps) >= 5)
    for img in all_images:
        captions[img] = captions[img][:5]
    print(f"[preprocess] {len(all_images)} images with >=5 captions")

    random.shuffle(all_images)
    train_ids = all_images[:n_train]
    val_ids   = all_images[n_train : n_train + n_val]
    test_ids  = all_images[n_train + n_val :]
    print(f"[preprocess] splits — train:{len(train_ids)}  val:{len(val_ids)}  test:{len(test_ids)}")

    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_from_captions(cap for img in train_ids for cap in captions[img])
    print(f"[preprocess] vocab size: {len(vocab)}")

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split = {"image_ids": ids, "captions_text": {img: captions[img] for img in ids}}
        with open(os.path.join(out_dir, f"{name}_data.pkl"), "wb") as f:
            pickle.dump(split, f)

    with open(os.path.join(out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    print(f"[preprocess] saved all pkl files to {out_dir!r}")
    return images_dir

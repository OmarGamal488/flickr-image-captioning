"""Phase 3 training loop — baseline ResNet50 + LSTM.

Usage:
    uv run python -m src.train --config configs/baseline.json
    uv run python -m src.train --config configs/baseline.json --smoke-test

The smoke-test mode trains on 128 samples for 1 epoch — it's meant to catch
shape/logic bugs without spending GPU time.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    FlickrDataset,
    build_eval_transform,
    build_items,
    build_processed_data,
    build_train_transform,
    collate_fn,
)
from src.decoder import DecoderGRU, DecoderLSTM
from src.encoder import EncoderCNN
from src.utils import (
    AverageMeter,
    TrainConfig,
    get_device,
    load_config,
    save_checkpoint,
    seed_everything,
)
from src.vocabulary import Vocabulary


def _load_split(processed_dir: str, name: str) -> dict:
    with open(os.path.join(processed_dir, f"{name}_data.pkl"), "rb") as f:
        return pickle.load(f)


def _build_loaders(
    cfg: TrainConfig, vocab: Vocabulary, smoke: bool
) -> tuple[DataLoader, DataLoader, dict, dict]:
    train = _load_split(cfg.processed_dir, "train")
    val = _load_split(cfg.processed_dir, "val")

    train_items = build_items(train["image_ids"], train["captions_text"], vocab)
    val_items = build_items(val["image_ids"], val["captions_text"], vocab)

    if smoke:
        train_items = train_items[:128]
        val_items = val_items[:32]

    train_ds = FlickrDataset(train_items, cfg.images_dir, build_train_transform())
    val_ds = FlickrDataset(val_items, cfg.images_dir, build_eval_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader, train, val


def train_one_epoch(
    encoder: EncoderCNN,
    decoder: DecoderLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float,
    log_interval: int,
    wandb_run=None,
    global_step: int = 0,
) -> tuple[float, int]:
    encoder.train()
    decoder.train()
    meter = AverageMeter()
    pbar = tqdm(loader, desc="train", leave=False)

    for step, (images, captions, lengths) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        features = encoder(images)                         # (B, E)
        logits = decoder(features, captions)               # (B, T, V)

        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            captions.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in decoder.parameters() if p.requires_grad]
            + [p for p in encoder.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )
        optimizer.step()

        meter.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=f"{meter.avg:.4f}")
        global_step += 1

        if wandb_run is not None and (step + 1) % log_interval == 0:
            wandb_run.log(
                {"train/loss_step": loss.item(), "train/running_avg": meter.avg},
                step=global_step,
            )

    return meter.avg, global_step


@torch.no_grad()
def validate(
    encoder: EncoderCNN,
    decoder: DecoderLSTM,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    encoder.eval()
    decoder.eval()
    meter = AverageMeter()
    for images, captions, _ in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        features = encoder(images)
        logits = decoder(features, captions)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            captions.reshape(-1),
        )
        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def validate_bleu(
    encoder: EncoderCNN,
    decoder: DecoderLSTM,
    val_split: dict,
    vocab: Vocabulary,
    cfg: TrainConfig,
    device: torch.device,
    max_samples: int,
) -> dict[str, float]:
    """Generate greedy captions for up to ``max_samples`` val images and compute BLEU-1..4.

    Uses all 5 ground-truth captions per image as references.
    """
    encoder.eval()
    decoder.eval()

    image_ids = val_split["image_ids"][:max_samples]
    transform = build_eval_transform()
    smoothing = SmoothingFunction().method1

    from PIL import Image as PILImage  # local import — not needed elsewhere

    refs_all: list[list[list[str]]] = []
    hyps_all: list[list[str]] = []

    batch_size = cfg.batch_size
    for i in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[i : i + batch_size]
        batch_tensors = []
        for img_id in batch_ids:
            with PILImage.open(os.path.join(cfg.images_dir, img_id)) as img:
                batch_tensors.append(transform(img.convert("RGB")))
        images = torch.stack(batch_tensors).to(device, non_blocking=True)
        features = encoder(images)
        generated = decoder.generate_greedy(features, max_len=20)

        for img_id, ids in zip(batch_ids, generated):
            hyp = vocab.denumericalize(ids).split()
            refs = [Vocabulary.tokenize(c) for c in val_split["captions_text"][img_id]]
            hyps_all.append(hyp)
            refs_all.append(refs)

    bleu = {
        "BLEU-1": corpus_bleu(refs_all, hyps_all, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        "BLEU-2": corpus_bleu(refs_all, hyps_all, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
        "BLEU-3": corpus_bleu(refs_all, hyps_all, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoothing),
        "BLEU-4": corpus_bleu(refs_all, hyps_all, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing),
    }
    return bleu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.json")
    parser.add_argument("--smoke-test", action="store_true", help="1-epoch run on 128 samples")
    parser.add_argument("--no-wandb", action="store_true", help="Force-disable W&B logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke_test:
        cfg.num_epochs = 1
        cfg.num_workers = 0
        cfg.run_name = cfg.run_name + "-smoke"
        cfg.val_bleu_subset = 16

    seed_everything(cfg.seed)
    device = get_device()
    print(f"device: {device}")
    print(f"config: {json.dumps(asdict(cfg), indent=2)}")

    # Build processed data if missing (auto-detects Flickr8k / Flickr30k)
    detected_images_dir = build_processed_data(
        raw_dir="data/raw", out_dir=cfg.processed_dir, seed=cfg.seed
    )
    cfg.images_dir = detected_images_dir

    # Vocabulary
    with open(os.path.join(cfg.processed_dir, "vocab.pkl"), "rb") as f:
        vocab: Vocabulary = pickle.load(f)
    print(f"vocab size: {len(vocab)}")

    # Data
    train_loader, val_loader, _, val_split = _build_loaders(cfg, vocab, smoke=args.smoke_test)
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # Model
    encoder = EncoderCNN(embed_size=cfg.embed_size, pretrained=True, freeze=True).to(device)
    decoder_cls = DecoderGRU if getattr(cfg, "rnn_type", "lstm") == "gru" else DecoderLSTM
    decoder = decoder_cls(
        vocab_size=len(vocab),
        embed_size=cfg.embed_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    trainable = [p for p in decoder.parameters() if p.requires_grad]
    trainable += [p for p in encoder.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_params/1e6:.2f}M")

    loss_fn = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_IDX, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
    )

    # W&B (optional — off by default in smoke mode)
    wandb_run = None
    if not args.no_wandb and not args.smoke_test and cfg.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name,
                mode=cfg.wandb_mode,
                config=asdict(cfg),
            )
        except Exception as e:
            print(f"wandb init failed, continuing without: {e}")
            wandb_run = None

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_bleu4 = 0.0
    global_step = 0
    history: list[dict] = []

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.perf_counter()
        train_loss, global_step = train_one_epoch(
            encoder, decoder, train_loader, optimizer, loss_fn, device,
            cfg.grad_clip, cfg.log_interval, wandb_run, global_step,
        )
        val_loss = validate(encoder, decoder, val_loader, loss_fn, device)
        bleu = validate_bleu(
            encoder, decoder, val_split, vocab, cfg, device, cfg.val_bleu_subset
        )
        dt = time.perf_counter() - t0

        scheduler.step(bleu["BLEU-4"])
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": lr_now,
            "time_s": round(dt, 1),
            **{k: round(v, 4) for k, v in bleu.items()},
        }
        history.append(row)
        print(json.dumps(row))

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/BLEU-1": bleu["BLEU-1"],
                    "val/BLEU-2": bleu["BLEU-2"],
                    "val/BLEU-3": bleu["BLEU-3"],
                    "val/BLEU-4": bleu["BLEU-4"],
                    "lr": lr_now,
                    "epoch_time_s": dt,
                },
                step=global_step,
            )

        if bleu["BLEU-4"] > best_bleu4:
            best_bleu4 = bleu["BLEU-4"]
            ckpt_path = os.path.join(cfg.save_dir, f"{cfg.run_name}.pth")
            save_checkpoint(ckpt_path, encoder, decoder, optimizer, epoch, row, cfg)
            print(f"  ✓ new best BLEU-4 = {best_bleu4:.4f} saved to {ckpt_path}")

    history_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"history saved to {history_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

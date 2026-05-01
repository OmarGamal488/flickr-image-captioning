"""Phase 4 training loop — ResNet50 + Bahdanau attention + LSTM/GRU decoder.

Supports both LSTMCell and GRUCell decoders via ``rnn_type`` in the config,
and two-phase training: Phase A freezes the CNN, Phase B unfreezes the last
two ResNet residual blocks and fine-tunes with a smaller encoder learning
rate. Doubly-stochastic regularization on the attention weights is added to
the CE loss at training time.

Usage:
    uv run python -m src.train_attention --config configs/attention_lstm.json
    uv run python -m src.train_attention --config configs/attention_lstm.json --smoke-test
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
from src.decoder import DecoderAttention
from src.encoder import EncoderCNN_Attention
from src.utils import (
    AttentionConfig,
    AverageMeter,
    get_device,
    load_att_config,
    seed_everything,
)
from src.vocabulary import Vocabulary


def _load_split(processed_dir: str, name: str) -> dict:
    with open(os.path.join(processed_dir, f"{name}_data.pkl"), "rb") as f:
        return pickle.load(f)


def _build_loaders(cfg: AttentionConfig, vocab: Vocabulary, smoke: bool):
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
    return train_loader, val_loader, val


def _build_optimizer(
    encoder: EncoderCNN_Attention,
    decoder: DecoderAttention,
    decoder_lr: float,
    encoder_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build optimizer with separate LR for encoder-trainable params (if any)."""
    param_groups: list[dict] = [
        {
            "params": [p for p in decoder.parameters() if p.requires_grad],
            "lr": decoder_lr,
        }
    ]
    enc_trainable = [p for p in encoder.parameters() if p.requires_grad]
    if enc_trainable:
        param_groups.append({"params": enc_trainable, "lr": encoder_lr})
    return torch.optim.Adam(param_groups, weight_decay=weight_decay)


def train_one_epoch(
    encoder: EncoderCNN_Attention,
    decoder: DecoderAttention,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    alpha_c: float,
    device: torch.device,
    grad_clip: float,
    log_interval: int,
    wandb_run=None,
    global_step: int = 0,
    ss_prob: float = 0.0,
) -> tuple[float, float, int]:
    encoder.train()
    decoder.train()
    ce_meter = AverageMeter()
    att_meter = AverageMeter()
    pbar = tqdm(loader, desc="train", leave=False)

    for step, (images, captions, lengths) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        features = encoder(images)                              # (B, P, enc_dim)
        logits, alphas = decoder(features, captions, ss_prob)   # (B, T-1, V), (B, T-1, P)

        # Shifted targets: at step t predict captions[:, t+1] given input captions[:, t]
        targets = captions[:, 1:]                               # (B, T-1)
        ce = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        # Doubly-stochastic regularization (Show, Attend and Tell §4.2.1)
        double_stoch = ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        loss = ce + alpha_c * double_stoch

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in decoder.parameters() if p.requires_grad]
            + [p for p in encoder.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )
        optimizer.step()

        ce_meter.update(ce.item(), images.size(0))
        att_meter.update(double_stoch.item(), images.size(0))
        pbar.set_postfix(ce=f"{ce_meter.avg:.4f}", att=f"{att_meter.avg:.4f}")
        global_step += 1

        if wandb_run is not None and (step + 1) % log_interval == 0:
            wandb_run.log(
                {
                    "train/ce_step": ce.item(),
                    "train/att_step": double_stoch.item(),
                    "train/ce_avg": ce_meter.avg,
                },
                step=global_step,
            )

    return ce_meter.avg, att_meter.avg, global_step


@torch.no_grad()
def validate(
    encoder: EncoderCNN_Attention,
    decoder: DecoderAttention,
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
        logits, _ = decoder(features, captions)
        targets = captions[:, 1:]
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def validate_bleu(
    encoder: EncoderCNN_Attention,
    decoder: DecoderAttention,
    val_split: dict,
    vocab: Vocabulary,
    cfg: AttentionConfig,
    device: torch.device,
    max_samples: int,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()

    image_ids = val_split["image_ids"][:max_samples]
    transform = build_eval_transform()
    smoothing = SmoothingFunction().method1

    from PIL import Image as PILImage

    refs_all: list[list[list[str]]] = []
    hyps_all: list[list[str]] = []

    for i in range(0, len(image_ids), cfg.batch_size):
        batch_ids = image_ids[i : i + cfg.batch_size]
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

    return {
        "BLEU-1": corpus_bleu(refs_all, hyps_all, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        "BLEU-2": corpus_bleu(refs_all, hyps_all, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
        "BLEU-3": corpus_bleu(refs_all, hyps_all, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoothing),
        "BLEU-4": corpus_bleu(refs_all, hyps_all, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing),
    }


def save_att_checkpoint(
    path: str,
    encoder: EncoderCNN_Attention,
    decoder: DecoderAttention,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    cfg: AttentionConfig,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": asdict(cfg),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_att_config(args.config)
    if args.smoke_test:
        cfg.num_epochs = 1
        cfg.num_workers = 0
        cfg.run_name = cfg.run_name + "-smoke"
        cfg.val_bleu_subset = 16
        cfg.fine_tune_start_epoch = 999  # effectively disabled

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
    train_loader, val_loader, val_split = _build_loaders(cfg, vocab, args.smoke_test)
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # Model
    encoder = EncoderCNN_Attention(pretrained=True, freeze=True).to(device)
    decoder = DecoderAttention(
        vocab_size=len(vocab),
        encoder_dim=cfg.encoder_dim,
        embed_size=cfg.embed_size,
        hidden_size=cfg.hidden_size,
        attention_dim=cfg.attention_dim,
        dropout=cfg.dropout,
        rnn_type=cfg.rnn_type,
    ).to(device)

    if cfg.glove_path:
        from src.utils import load_glove_embeddings
        glove_weights = load_glove_embeddings(cfg.glove_path, vocab, cfg.embed_size)
        decoder.embedding.weight.data.copy_(glove_weights)

    n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"trainable params: decoder {n_dec/1e6:.2f}M, encoder {n_enc/1e6:.2f}M")

    loss_fn = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_IDX, label_smoothing=cfg.label_smoothing)
    optimizer = _build_optimizer(
        encoder, decoder, cfg.decoder_lr, cfg.encoder_lr, cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
    )

    # W&B
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
            print(f"wandb init failed: {e}")
            wandb_run = None

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_bleu4 = 0.0
    global_step = 0
    history: list[dict] = []

    for epoch in range(1, cfg.num_epochs + 1):
        # Phase B: unfreeze last N ResNet blocks at the configured epoch
        if epoch == cfg.fine_tune_start_epoch:
            encoder.fine_tune_last_blocks(cfg.fine_tune_blocks)
            optimizer = _build_optimizer(
                encoder, decoder, cfg.decoder_lr, cfg.encoder_lr, cfg.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
            )
            n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            print(f"[epoch {epoch}] unfroze last {cfg.fine_tune_blocks} ResNet blocks — trainable encoder params {n_enc/1e6:.2f}M")

        # Linear scheduled sampling ramp: 0 → scheduled_sampling_max over all epochs
        ss_prob = cfg.scheduled_sampling_max * (epoch - 1) / max(cfg.num_epochs - 1, 1)

        t0 = time.perf_counter()
        train_ce, train_att, global_step = train_one_epoch(
            encoder, decoder, train_loader, optimizer, loss_fn, cfg.alpha_c,
            device, cfg.grad_clip, cfg.log_interval, wandb_run, global_step,
            ss_prob=ss_prob,
        )
        val_loss = validate(encoder, decoder, val_loader, loss_fn, device)
        bleu = validate_bleu(encoder, decoder, val_split, vocab, cfg, device, cfg.val_bleu_subset)
        dt = time.perf_counter() - t0

        scheduler.step(bleu["BLEU-4"])
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_ce": round(train_ce, 4),
            "train_att": round(train_att, 4),
            "val_loss": round(val_loss, 4),
            "decoder_lr": lr_now,
            "time_s": round(dt, 1),
            **{k: round(v, 4) for k, v in bleu.items()},
        }
        history.append(row)
        print(json.dumps(row))

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/ce": train_ce,
                    "train/att": train_att,
                    "val/loss": val_loss,
                    "val/BLEU-1": bleu["BLEU-1"],
                    "val/BLEU-2": bleu["BLEU-2"],
                    "val/BLEU-3": bleu["BLEU-3"],
                    "val/BLEU-4": bleu["BLEU-4"],
                    "decoder_lr": lr_now,
                    "epoch_time_s": dt,
                },
                step=global_step,
            )

        if bleu["BLEU-4"] > best_bleu4:
            best_bleu4 = bleu["BLEU-4"]
            ckpt_path = os.path.join(cfg.save_dir, f"{cfg.run_name}.pth")
            save_att_checkpoint(ckpt_path, encoder, decoder, optimizer, epoch, row, cfg)
            print(f"  ✓ new best BLEU-4 = {best_bleu4:.4f} saved to {ckpt_path}")

    hist_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"history saved to {hist_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

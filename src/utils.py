"""Shared utilities: seeding, device, checkpointing, moving averages."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    """Running average over the scalars you feed in."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


@dataclass
class AttentionConfig:
    """Config for the attention-based captioner used by train_attention.py + hpo.py.

    Kept in utils (lightweight module) so ``src.inference`` can import it without
    pulling in nltk/tqdm/optuna via train_attention.py — the FastAPI and Gradio
    runtimes only need torch + numpy and this class.
    """

    # Data
    images_dir: str = "data/raw/Images"
    processed_dir: str = "data/processed"

    # Model
    encoder_dim: int = 2048
    embed_size: int = 256
    hidden_size: int = 512
    attention_dim: int = 256
    dropout: float = 0.5
    rnn_type: str = "lstm"          # lstm / gru

    # Loss
    alpha_c: float = 1.0            # doubly-stochastic regularization weight

    # Training
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 25
    decoder_lr: float = 4e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    # Two-phase fine-tuning
    fine_tune_start_epoch: int = 16
    fine_tune_blocks: int = 2

    # Logistics
    seed: int = 42
    save_dir: str = "models"
    run_name: str = "attention_lstm"
    log_interval: int = 50
    val_bleu_subset: int = 200
    wandb_project: str = "flickr8k-captioning"
    wandb_mode: str = "online"


def load_att_config(path: str) -> "AttentionConfig":
    with open(path) as f:
        data = json.load(f)
    return AttentionConfig(**data)


@dataclass
class TrainConfig:
    # Data
    images_dir: str = "data/raw/Images"
    processed_dir: str = "data/processed"

    # Model
    embed_size: int = 256
    hidden_size: int = 512
    num_layers: int = 1
    dropout: float = 0.5
    backbone: str = "resnet50"

    # Training
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    # Logistics
    seed: int = 42
    save_dir: str = "models"
    run_name: str = "baseline"
    log_interval: int = 50
    wandb_project: str = "flickr8k-captioning"
    wandb_mode: str = "online"  # online / offline / disabled
    val_bleu_subset: int = 200  # cap per-epoch val BLEU probe for speed


def save_config(cfg: TrainConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        data = json.load(f)
    return TrainConfig(**data)


def save_checkpoint(
    path: str,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: TrainConfig,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    path: str, encoder: torch.nn.Module, decoder: torch.nn.Module, map_location: Any = None
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    return ckpt

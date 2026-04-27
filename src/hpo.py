"""Optuna HPO for the Bahdanau-attention captioning model.

Each trial samples hyperparameters, builds an encoder+decoder, trains for a few
epochs on a **reduced** train subset, and reports the best val BLEU-4 achieved.
The goal is to rank configurations, not to train each one to convergence — the
winning config will be re-trained at full scale afterwards.

Search space:
  - embed_size      : {128, 256, 512}
  - hidden_size     : {256, 512, 1024}
  - attention_dim   : {128, 256, 512}
  - dropout         : float [0.3, 0.6]
  - decoder_lr      : float [1e-4, 1e-3] log-uniform
  - rnn_type        : {lstm, gru}
  - alpha_c         : float [0.5, 2.0]         (doubly-stochastic reg weight)

Usage:
    uv run python -m src.hpo --n-trials 20
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (
    FlickrDataset,
    build_eval_transform,
    build_items,
    build_train_transform,
    collate_fn,
)
from src.decoder import DecoderAttention
from src.encoder import EncoderCNN_Attention
from src.train_attention import (
    _load_split,
    train_one_epoch,
    validate,
    validate_bleu,
)
from src.utils import AttentionConfig, get_device, seed_everything
from src.vocabulary import Vocabulary


# ----- HPO-specific constants ------------------------------------------------

TRAIN_SUBSET_SIZE = 3000   # caption rows (~600 images × 5 captions)
VAL_LOSS_SUBSET = 500      # caption rows for the CE val loss
VAL_BLEU_SUBSET = 100      # images for per-epoch BLEU-4 probe
NUM_EPOCHS_PER_TRIAL = 5
BATCH_SIZE = 32
NUM_WORKERS = 2            # lower than full training to avoid forking between trials
WANDB_PROJECT = "flickr8k-captioning-hpo"
STUDY_NAME = "flickr8k_attention_hpo"
STORAGE_URL = "sqlite:///experiments/optuna_study.db"

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/Images"


def _build_trial_loaders(
    vocab: Vocabulary,
) -> tuple[DataLoader, DataLoader, dict]:
    """Same dataloaders across every trial — ensures comparisons are apples-to-apples."""
    train = _load_split(PROCESSED_DIR, "train")
    val = _load_split(PROCESSED_DIR, "val")

    train_items = build_items(train["image_ids"], train["captions_text"], vocab)[:TRAIN_SUBSET_SIZE]
    val_items = build_items(val["image_ids"], val["captions_text"], vocab)[:VAL_LOSS_SUBSET]

    train_ds = FlickrDataset(train_items, IMAGES_DIR, build_train_transform())
    val_ds = FlickrDataset(val_items, IMAGES_DIR, build_eval_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return train_loader, val_loader, val


def _build_objective(vocab: Vocabulary):
    device = get_device()
    # Load data loaders once; they're identical across trials.
    train_loader, val_loader, val_split = _build_trial_loaders(vocab)

    def objective(trial: optuna.Trial) -> float:
        # --- sample hyperparameters -----------------------------------------
        embed_size = trial.suggest_categorical("embed_size", [128, 256, 512])
        hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024])
        attention_dim = trial.suggest_categorical("attention_dim", [128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.3, 0.6)
        decoder_lr = trial.suggest_float("decoder_lr", 1e-4, 1e-3, log=True)
        rnn_type = trial.suggest_categorical("rnn_type", ["lstm", "gru"])
        alpha_c = trial.suggest_float("alpha_c", 0.5, 2.0)

        cfg = AttentionConfig(
            images_dir=IMAGES_DIR,
            processed_dir=PROCESSED_DIR,
            encoder_dim=2048,
            embed_size=embed_size,
            hidden_size=hidden_size,
            attention_dim=attention_dim,
            dropout=dropout,
            rnn_type=rnn_type,
            alpha_c=alpha_c,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            num_epochs=NUM_EPOCHS_PER_TRIAL,
            decoder_lr=decoder_lr,
            encoder_lr=1e-5,
            weight_decay=0.0,
            grad_clip=5.0,
            scheduler_patience=3,
            scheduler_factor=0.5,
            fine_tune_start_epoch=999,
            fine_tune_blocks=2,
            seed=42,
            save_dir="experiments/hpo",
            run_name=f"hpo_trial_{trial.number:02d}",
            log_interval=25,
            val_bleu_subset=VAL_BLEU_SUBSET,
            wandb_project=WANDB_PROJECT,
            wandb_mode="online",
        )

        print(
            f"\n=== trial {trial.number:02d} "
            f"| embed={embed_size} hidden={hidden_size} att={attention_dim} "
            f"dropout={dropout:.2f} lr={decoder_lr:.1e} rnn={rnn_type} alpha_c={alpha_c:.2f} ==="
        )

        seed_everything(cfg.seed)

        # --- build fresh model -------------------------------------------------
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

        loss_fn = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_IDX)
        trainable = [p for p in decoder.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=cfg.decoder_lr, weight_decay=cfg.weight_decay)

        # --- optional W&B --------------------------------------------------
        wandb_run = None
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name,
                config=asdict(cfg),
                reinit=True,
            )
        except Exception as e:
            print(f"  wandb init failed: {e}")

        # --- train a few epochs --------------------------------------------
        best_bleu4 = 0.0
        try:
            global_step = 0
            for epoch in range(1, cfg.num_epochs + 1):
                t0 = time.perf_counter()
                train_ce, train_att, global_step = train_one_epoch(
                    encoder, decoder, train_loader, optimizer, loss_fn, cfg.alpha_c,
                    device, cfg.grad_clip, cfg.log_interval, wandb_run, global_step,
                )
                val_loss = validate(encoder, decoder, val_loader, loss_fn, device)
                bleu = validate_bleu(
                    encoder, decoder, val_split, vocab, cfg, device, cfg.val_bleu_subset
                )
                dt = time.perf_counter() - t0
                bleu4 = bleu["BLEU-4"]
                best_bleu4 = max(best_bleu4, bleu4)

                print(
                    f"  epoch {epoch}: ce={train_ce:.4f} val_loss={val_loss:.4f} "
                    f"BLEU-4={bleu4:.4f} ({dt:.1f}s)"
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch": epoch,
                            "train/ce": train_ce,
                            "train/att": train_att,
                            "val/loss": val_loss,
                            "val/BLEU-4": bleu4,
                            "best/BLEU-4": best_bleu4,
                            "epoch_time_s": dt,
                        }
                    )

                # Optuna pruning: report intermediate and optionally stop early
                trial.report(bleu4, epoch)
                if trial.should_prune():
                    print("  pruned by optuna")
                    if wandb_run is not None:
                        wandb_run.finish()
                    raise optuna.TrialPruned()

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM — trial {trial.number} marked as failed")
            if wandb_run is not None:
                wandb_run.finish()
            del encoder, decoder, optimizer
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        # --- cleanup -------------------------------------------------------
        if wandb_run is not None:
            wandb_run.summary["final_best_BLEU-4"] = best_bleu4
            wandb_run.finish()

        del encoder, decoder, optimizer
        torch.cuda.empty_cache()
        return best_bleu4

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--seed", type=int, default=42, help="TPE sampler seed for reproducibility"
    )
    args = parser.parse_args()

    os.makedirs("experiments", exist_ok=True)
    os.makedirs("experiments/hpo", exist_ok=True)

    with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "rb") as f:
        vocab: Vocabulary = pickle.load(f)
    print(f"vocab size: {len(vocab)}")
    print(f"device: {get_device()}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = _build_objective(vocab)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # --- summary ----------------------------------------------------------
    print("\n=== HPO finished ===")
    print(f"best BLEU-4: {study.best_value:.4f}")
    print("best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Persist the summary for the notebook + report
    out = {
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/hpo_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("summary → experiments/hpo_summary.json")


if __name__ == "__main__":
    main()

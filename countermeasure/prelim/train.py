"""
Phase 2 — AASIST Training Script
==================================
Train AASIST / AASIST-L on ASVspoof 2019 LA for deepfake audio detection.

Features:
  - Weighted cross-entropy loss (handles class imbalance)
  - EER (Equal Error Rate) evaluation on dev set
  - Cosine annealing LR scheduler with warmup
  - Gradient clipping
  - Model checkpointing (best EER)
  - TensorBoard logging (optional)
  - Mixed precision training (optional, for GPU)
  - Early stopping

Usage:
    python train.py                                  # Train AASIST-L with defaults
    python train.py --variant AASIST                 # Train full AASIST
    python train.py --epochs 50 --batch_size 24      # Custom settings
    python train.py --use_preprocessed               # Use .npy files (faster)
    python train.py --subset 5000                    # Quick experiment
    python train.py --resume checkpoints/best.pt     # Resume training
"""

import argparse
import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    TRAIN_PROTOCOL, TRAIN_FLAC_DIR, DEV_PROTOCOL, DEV_FLAC_DIR,
    EVAL_PROTOCOL, EVAL_FLAC_DIR,
    MAX_AUDIO_LENGTH, METADATA_OUTPUT_DIR,
    TRAIN_OUTPUT_DIR, DEV_OUTPUT_DIR, EVAL_OUTPUT_DIR,
)
from dataset import ASVspoofRawDataset, ASVspoofDataset
from aasist_model import build_model


# ─────────────────────────────────────────────
# EER Computation
# ─────────────────────────────────────────────
def compute_eer(scores, labels):
    """
    Compute Equal Error Rate (EER).

    Parameters
    ----------
    scores : np.ndarray
        Predicted scores (higher = more likely bonafide).
    labels : np.ndarray
        Ground truth labels (1 = bonafide, 0 = spoof).

    Returns
    -------
    eer : float
        Equal Error Rate.
    threshold : float
        Threshold at which FRR ≈ FAR.
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find EER where FPR = FNR
    try:
        eer = brentq(lambda x: interp1d(fpr, fpr)(x) - interp1d(fpr, fnr)(x), 0.0, 1.0)
        thresh = interp1d(fpr, thresholds)(eer)
    except ValueError:
        # Fallback: find closest point
        abs_diff = np.abs(fpr - fnr)
        idx = np.argmin(abs_diff)
        eer = (fpr[idx] + fnr[idx]) / 2
        thresh = thresholds[idx]

    return float(eer), float(thresh)


def compute_t_dcf(scores, labels, Pspoof=0.05, Cmiss=1, Cfa=10):
    """
    Compute a simplified min t-DCF (tandem detection cost function).

    Parameters
    ----------
    scores : np.ndarray
    labels : np.ndarray
    Pspoof : float
        Prior probability of spoof.
    Cmiss : float
        Cost of missing a bonafide.
    Cfa : float
        Cost of accepting a spoof.

    Returns
    -------
    min_tdcf : float
    """
    from sklearn.metrics import roc_curve

    bonafide_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]

    n_bona = len(bonafide_scores)
    n_spoof = len(spoof_scores)

    if n_bona == 0 or n_spoof == 0:
        return float('inf')

    # Combine and sort thresholds
    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    thresholds = np.sort(all_scores)

    min_tdcf = float('inf')

    for thresh in thresholds:
        # False rejection: bonafide classified as spoof
        frr = np.mean(bonafide_scores < thresh)
        # False acceptance: spoof classified as bonafide
        far = np.mean(spoof_scores >= thresh)

        tdcf = Cmiss * frr * (1 - Pspoof) + Cfa * far * Pspoof

        if tdcf < min_tdcf:
            min_tdcf = tdcf

    return float(min_tdcf)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """
    Evaluate model on a dataset.

    Returns
    -------
    dict with keys: eer, threshold, min_tdcf, loss, accuracy,
                    mean_score_bonafide, mean_score_spoof
    """
    model.eval()

    all_scores = []
    all_labels = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y, utt_ids in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

        # Score = P(bonafide)
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().numpy()

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

        all_scores.extend(scores)
        all_labels.extend(y.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Compute metrics
    eer, threshold = compute_eer(all_scores, all_labels)
    min_tdcf = compute_t_dcf(all_scores, all_labels)

    results = {
        "eer":          eer,
        "eer_pct":      eer * 100,
        "threshold":    threshold,
        "min_tdcf":     min_tdcf,
        "accuracy":     total_correct / total_samples if total_samples > 0 else 0,
        "loss":         total_loss / total_samples if total_samples > 0 and criterion else None,
        "num_samples":  total_samples,
        "mean_score_bonafide": float(all_scores[all_labels == 1].mean()) if (all_labels == 1).any() else 0,
        "mean_score_spoof":    float(all_scores[all_labels == 0].mean()) if (all_labels == 0).any() else 0,
    }

    return results


# ─────────────────────────────────────────────
# Warmup + Cosine Annealing Scheduler
# ─────────────────────────────────────────────
class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * alpha
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ─────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────
def plot_training_curves(history, output_dir, timestamp):
    """
    Generate and save all training plots at the end of training.

    Produces:
      1. Loss curves (train + dev)
      2. EER over epochs with best marked
      3. Accuracy curves (train + dev)
      4. min t-DCF over epochs
      5. Learning rate schedule
      6. Score separation (bonafide vs spoof mean scores)

    All saved as PNGs in output_dir.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots.")
        print("  Install with: pip install matplotlib")
        return

    if not history:
        print("  No training history to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    dev_eer = [h["dev_eer_pct"] for h in history]
    dev_tdcf = [h["dev_min_tdcf"] for h in history]
    dev_loss = [h["dev_loss"] for h in history]
    dev_acc = [h["dev_acc"] for h in history]
    lrs = [h["lr"] for h in history]

    # Best epoch indices
    best_eer_idx = int(np.argmin(dev_eer))
    best_loss_idx = int(np.argmin(dev_loss))
    best_acc_idx = int(np.argmax(dev_acc))
    best_tdcf_idx = int(np.argmin(dev_tdcf))

    # ── Combined overview (2x3 grid) ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("AASIST Training Summary", fontsize=16, fontweight="bold", y=0.98)

    # 1) Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=1.5)
    ax.plot(epochs, dev_loss, "r-", label="Dev Loss", linewidth=1.5)
    ax.axvline(epochs[best_loss_idx], color="red", linestyle=":", alpha=0.4)
    ax.scatter([epochs[best_loss_idx]], [dev_loss[best_loss_idx]],
               color="red", s=80, zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(f"Best: {dev_loss[best_loss_idx]:.4f}\n(epoch {epochs[best_loss_idx]})",
                xy=(epochs[best_loss_idx], dev_loss[best_loss_idx]),
                xytext=(15, 15), textcoords="offset points",
                fontsize=9, fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Dev Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) EER curve
    ax = axes[0, 1]
    ax.plot(epochs, dev_eer, "g-o", markersize=3, linewidth=1.5, label="Dev EER")
    ax.scatter([epochs[best_eer_idx]], [dev_eer[best_eer_idx]],
               color="red", s=120, zorder=5, marker="*", edgecolors="black", linewidths=0.5)
    ax.axhline(dev_eer[best_eer_idx], color="green", linestyle="--", alpha=0.3)
    ax.annotate(f"Best: {dev_eer[best_eer_idx]:.2f}%\n(epoch {epochs[best_eer_idx]})",
                xy=(epochs[best_eer_idx], dev_eer[best_eer_idx]),
                xytext=(15, 15), textcoords="offset points",
                fontsize=10, fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EER (%)")
    ax.set_title("Dev Equal Error Rate")
    ax.grid(True, alpha=0.3)

    # 3) Accuracy curves
    ax = axes[0, 2]
    ax.plot(epochs, [a * 100 for a in train_acc], "b-", label="Train Acc", linewidth=1.5)
    ax.plot(epochs, [a * 100 for a in dev_acc], "r-", label="Dev Acc", linewidth=1.5)
    ax.scatter([epochs[best_acc_idx]], [dev_acc[best_acc_idx] * 100],
               color="red", s=80, zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(f"Best: {dev_acc[best_acc_idx]*100:.2f}%\n(epoch {epochs[best_acc_idx]})",
                xy=(epochs[best_acc_idx], dev_acc[best_acc_idx] * 100),
                xytext=(-15, -25), textcoords="offset points",
                fontsize=9, fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Dev Accuracy")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # 4) min t-DCF
    ax = axes[1, 0]
    ax.plot(epochs, dev_tdcf, "m-o", markersize=3, linewidth=1.5)
    ax.scatter([epochs[best_tdcf_idx]], [dev_tdcf[best_tdcf_idx]],
               color="red", s=120, zorder=5, marker="*", edgecolors="black", linewidths=0.5)
    ax.annotate(f"Best: {dev_tdcf[best_tdcf_idx]:.4f}\n(epoch {epochs[best_tdcf_idx]})",
                xy=(epochs[best_tdcf_idx], dev_tdcf[best_tdcf_idx]),
                xytext=(15, 15), textcoords="offset points",
                fontsize=9, fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("min t-DCF")
    ax.set_title("Dev min t-DCF")
    ax.grid(True, alpha=0.3)

    # 5) Learning rate schedule
    ax = axes[1, 1]
    ax.plot(epochs, lrs, "k-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))
    ax.grid(True, alpha=0.3)

    # 6) Score separation
    has_scores = "mean_score_bonafide" in history[0]
    if has_scores:
        bona_scores = [h["mean_score_bonafide"] for h in history]
        spoof_scores = [h["mean_score_spoof"] for h in history]
        ax = axes[1, 2]
        ax.plot(epochs, bona_scores, "g-", label="Bonafide (mean)", linewidth=1.5)
        ax.plot(epochs, spoof_scores, "r-", label="Spoof (mean)", linewidth=1.5)
        ax.fill_between(epochs, spoof_scores, bona_scores, alpha=0.15, color="blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score P(bonafide)")
        ax.set_title("Score Separation Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 2].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    overview_path = output_dir / f"training_overview_{timestamp}.png"
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Training overview plot saved: {overview_path}")

    # ── Individual high-res plots ──
    # EER standalone
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, dev_eer, "g-o", markersize=4, linewidth=2, label="Dev EER (%)")
    ax.scatter([epochs[best_eer_idx]], [dev_eer[best_eer_idx]],
               color="red", s=150, zorder=5, marker="*", edgecolors="black", linewidths=1)
    ax.axhline(dev_eer[best_eer_idx], color="green", linestyle="--", alpha=0.3,
               label=f"Best = {dev_eer[best_eer_idx]:.2f}% (epoch {epochs[best_eer_idx]})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_title("Dev EER Over Training", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    eer_path = output_dir / f"eer_curve_{timestamp}.png"
    plt.savefig(eer_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  EER plot saved: {eer_path}")

    # Loss standalone
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, "b-", linewidth=2, label="Train Loss")
    ax.plot(epochs, dev_loss, "r-", linewidth=2, label="Dev Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    loss_path = output_dir / f"loss_curve_{timestamp}.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Loss plot saved: {loss_path}")


def print_best_metrics_summary(history):
    """
    Print a table of best metrics and the epoch at which each was achieved.
    """
    if not history:
        return

    dev_eer = [h["dev_eer_pct"] for h in history]
    dev_loss = [h["dev_loss"] for h in history]
    dev_acc = [h["dev_acc"] for h in history]
    dev_tdcf = [h["dev_min_tdcf"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]

    best = {
        "Dev EER (%)":      (min(dev_eer),      history[int(np.argmin(dev_eer))]["epoch"]),
        "Dev min t-DCF":    (min(dev_tdcf),     history[int(np.argmin(dev_tdcf))]["epoch"]),
        "Dev Loss":         (min(dev_loss),     history[int(np.argmin(dev_loss))]["epoch"]),
        "Dev Accuracy (%)": (max(dev_acc)*100,  history[int(np.argmax(dev_acc))]["epoch"]),
        "Train Loss":       (min(train_loss),   history[int(np.argmin(train_loss))]["epoch"]),
        "Train Accuracy (%)": (max(train_acc)*100, history[int(np.argmax(train_acc))]["epoch"]),
    }

    has_scores = "mean_score_bonafide" in history[0]
    if has_scores:
        bona_scores = [h["mean_score_bonafide"] for h in history]
        spoof_scores = [h["mean_score_spoof"] for h in history]
        separations = [b - s for b, s in zip(bona_scores, spoof_scores)]
        best_sep_idx = int(np.argmax(separations))
        best["Score Separation"] = (
            separations[best_sep_idx],
            history[best_sep_idx]["epoch"],
        )

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │           Best Metrics Summary                      │")
    print("  ├───────────────────────┬──────────────┬──────────────┤")
    print("  │ Metric                │    Value     │  Best Epoch  │")
    print("  ├───────────────────────┼──────────────┼──────────────┤")
    for metric, (value, ep) in best.items():
        if "(%)" in metric or "EER" in metric:
            val_str = f"{value:>10.2f}%"
        elif "Separation" in metric:
            val_str = f"{value:>11.4f}"
        else:
            val_str = f"{value:>11.4f}"
        print(f"  │ {metric:<21s} │ {val_str:>12s} │ {ep:>12d} │")
    print("  └───────────────────────┴──────────────┴──────────────┘")


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    max_grad_norm=5.0, use_amp=False):
    """
    Train for one epoch.

    Returns
    -------
    dict with keys: loss, accuracy, grad_norm
    """
    model.train()

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    grad_norms = []

    for batch_idx, (x, y, _) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # Track metrics
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)
        grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else gn)

    results = {
        "loss":      total_loss / total_samples,
        "accuracy":  total_correct / total_samples,
        "grad_norm": np.mean(grad_norms),
    }
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train AASIST")
    parser.add_argument("--variant", type=str, default="AASIST-L",
                        choices=["AASIST", "AASIST-L"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--spoof_weight", type=float, default=1.0,
                        help="CE weight for spoof class")
    parser.add_argument("--bonafide_weight", type=float, default=9.0,
                        help="CE weight for bonafide class")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit samples per split (for quick experiments)")
    parser.add_argument("--use_preprocessed", action="store_true",
                        help="Use .npy files instead of raw .flac")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Evaluate on dev set every N epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs without EER improvement)")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use mixed precision training (CUDA only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Setup ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("╔" + "═" * 58 + "╗")
    print("║   Phase 2: AASIST Training — ASVspoof 2019 LA            ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Model:        {args.variant}")
    print(f"  Device:       {device}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate:{args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup:       {args.warmup_epochs} epochs")
    print(f"  CE weights:   spoof={args.spoof_weight}, bonafide={args.bonafide_weight}")
    print(f"  Mixed prec:   {args.use_amp}")
    print(f"  Checkpoints:  {checkpoint_dir}")
    if args.subset:
        print(f"  Subset:       {args.subset} samples per split")
    print()

    # ── Data ──
    print("Loading datasets...")
    if args.use_preprocessed:
        train_dataset = ASVspoofDataset(
            metadata_csv=METADATA_OUTPUT_DIR / "train_metadata.csv",
            npy_dir=TRAIN_OUTPUT_DIR,
            subset_size=args.subset,
        )
        dev_dataset = ASVspoofDataset(
            metadata_csv=METADATA_OUTPUT_DIR / "dev_metadata.csv",
            npy_dir=DEV_OUTPUT_DIR,
            subset_size=args.subset,
        )
    else:
        train_dataset = ASVspoofRawDataset(
            protocol_file=TRAIN_PROTOCOL,
            flac_dir=TRAIN_FLAC_DIR,
            subset_size=args.subset,
        )
        dev_dataset = ASVspoofRawDataset(
            protocol_file=DEV_PROTOCOL,
            flac_dir=DEV_FLAC_DIR,
            subset_size=args.subset,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"  Train: {len(train_dataset):,} utterances, {len(train_loader)} batches")
    print(f"  Dev:   {len(dev_dataset):,} utterances, {len(dev_loader)} batches")

    # ── Model ──
    model = build_model(args.variant).to(device)

    # ── Loss, optimizer, scheduler ──
    class_weights = torch.FloatTensor([args.spoof_weight, args.bonafide_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs
    )

    # ── Resume ──
    start_epoch = 0
    best_eer = float('inf')
    history = []

    if args.resume:
        print(f"\n  Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer = ckpt.get("best_eer", float('inf'))
        history = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch}, best EER = {best_eer*100:.2f}%")

    # ── TensorBoard (optional) ──
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = checkpoint_dir / f"logs_{timestamp}"
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"  TensorBoard: {log_dir}")
    except ImportError:
        print("  TensorBoard not available (install tensorboard to enable)")

    # ── Training loop ──
    print("\n" + "─" * 60)
    print("  Starting training...")
    print("─" * 60)

    patience_counter = 0
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()[0]

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            max_grad_norm=args.max_grad_norm, use_amp=args.use_amp,
        )

        epoch_time = time.time() - epoch_start

        # Log training
        print(f"\n  Epoch {epoch+1:3d}/{args.epochs} "
              f"({epoch_time:.1f}s) │ "
              f"lr={current_lr:.2e} │ "
              f"loss={train_metrics['loss']:.4f} │ "
              f"acc={train_metrics['accuracy']:.4f} │ "
              f"grad={train_metrics['grad_norm']:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            dev_metrics = evaluate(model, dev_loader, device, criterion)

            print(f"         Dev   │ "
                  f"EER={dev_metrics['eer_pct']:6.2f}% │ "
                  f"min-tDCF={dev_metrics['min_tdcf']:.4f} │ "
                  f"loss={dev_metrics['loss']:.4f} │ "
                  f"acc={dev_metrics['accuracy']:.4f}")
            print(f"               │ "
                  f"score(bona)={dev_metrics['mean_score_bonafide']:.4f} │ "
                  f"score(spoof)={dev_metrics['mean_score_spoof']:.4f}")

            # Save history
            record = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_grad_norm": train_metrics["grad_norm"],
                "dev_eer": dev_metrics["eer"],
                "dev_eer_pct": dev_metrics["eer_pct"],
                "dev_min_tdcf": dev_metrics["min_tdcf"],
                "dev_loss": dev_metrics["loss"],
                "dev_acc": dev_metrics["accuracy"],
                "mean_score_bonafide": dev_metrics["mean_score_bonafide"],
                "mean_score_spoof": dev_metrics["mean_score_spoof"],
            }
            history.append(record)

            # TensorBoard
            if writer:
                writer.add_scalar("Train/Loss", train_metrics["loss"], epoch)
                writer.add_scalar("Train/Accuracy", train_metrics["accuracy"], epoch)
                writer.add_scalar("Train/GradNorm", train_metrics["grad_norm"], epoch)
                writer.add_scalar("Dev/EER", dev_metrics["eer_pct"], epoch)
                writer.add_scalar("Dev/MinTDCF", dev_metrics["min_tdcf"], epoch)
                writer.add_scalar("Dev/Loss", dev_metrics["loss"], epoch)
                writer.add_scalar("Dev/Accuracy", dev_metrics["accuracy"], epoch)
                writer.add_scalar("LR", current_lr, epoch)

            # Checkpoint if best EER
            if dev_metrics["eer"] < best_eer:
                best_eer = dev_metrics["eer"]
                patience_counter = 0

                ckpt_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_eer": best_eer,
                    "dev_metrics": dev_metrics,
                    "args": vars(args),
                    "history": history,
                }, ckpt_path)
                print(f"         ★ New best EER={best_eer*100:.2f}% → saved {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n  ⚠ Early stopping: no EER improvement for {args.patience} epochs")
                    break

    # ── Save final model ──
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eer": best_eer,
        "args": vars(args),
        "history": history,
    }, final_path)

    # ── Save training history ──
    history_path = checkpoint_dir / f"history_{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - training_start

    if writer:
        writer.close()

    # ── Summary ──
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   Training Complete                                      ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Total time:    {total_time/60:.1f} minutes")
    print(f"  Best dev EER:  {best_eer*100:.2f}%")
    print(f"  Best model:    {checkpoint_dir / 'best_model.pt'}")
    print(f"  Final model:   {final_path}")
    print(f"  History:       {history_path}")

    # ── Best metrics summary ──
    print_best_metrics_summary(history)

    # ── Generate and save training plots ──
    print("\n  Generating training plots...")
    plot_training_curves(history, checkpoint_dir, timestamp)

    print(f"\n  Next steps:")
    print(f"  1. Evaluate on eval set:")
    print(f"     python evaluate.py --checkpoint {checkpoint_dir / 'best_model.pt'}")
    print(f"  2. Analyze per-attack performance")
    print(f"  3. Generate DET curves")


if __name__ == "__main__":
    main()
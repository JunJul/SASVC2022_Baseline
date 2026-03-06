"""
Phase 3 — Evaluation Script
=============================
Evaluate a trained AASIST model on the ASVspoof 2019 LA eval set.

Produces:
  - Overall EER and min t-DCF
  - Per-attack-type EER breakdown (A07–A19)
  - Score distributions (bonafide vs spoof)
  - DET curve plot
  - Per-attack EER bar chart
  - Evaluation summary overview (multi-panel)
  - Score file for official ASVspoof evaluation tools

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt
    python evaluate.py --checkpoint checkpoints/best_model.pt --plot
    python evaluate.py --checkpoint checkpoints/best_model.pt --split dev
    python evaluate.py --checkpoint checkpoints/best_model.pt --split eval --plot
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime

from config import (
    DEV_PROTOCOL, DEV_FLAC_DIR, EVAL_PROTOCOL, EVAL_FLAC_DIR,
    METADATA_OUTPUT_DIR, DEV_OUTPUT_DIR, EVAL_OUTPUT_DIR,
    ATTACK_TYPES, LABEL_MAP, MAX_AUDIO_LENGTH, TARGET_SAMPLE_RATE,
)
from dataset import ASVspoofRawDataset, ASVspoofDataset
from aasist_model import build_model
from train import compute_eer, compute_t_dcf


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model and config from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    variant = args.get("variant", "AASIST-L")

    model = build_model(variant).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Loaded {variant} from epoch {ckpt.get('epoch', -1) + 1}")
    print(f"  Training best EER: {ckpt.get('best_eer', 0)*100:.2f}%")

    return model, args, ckpt


@torch.no_grad()
def run_inference(model, loader, device):
    """Run model inference, returning scores, labels, and utterance IDs."""
    model.eval()

    all_scores = []
    all_labels = []
    all_utt_ids = []

    for x, y, utt_ids in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().numpy()  # P(bonafide)

        all_scores.extend(scores)
        all_labels.extend(y.numpy())
        all_utt_ids.extend(utt_ids)

    return np.array(all_scores), np.array(all_labels), all_utt_ids


def get_attack_types_from_protocol(protocol_path):
    """Parse protocol to get attack type per utterance."""
    utt_to_attack = {}
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                utt_id = parts[1]
                attack_type = parts[3]
                utt_to_attack[utt_id] = attack_type
    return utt_to_attack


def per_attack_analysis(scores, labels, utt_ids, utt_to_attack):
    """Compute EER for each attack type separately."""
    attacks = np.array([utt_to_attack.get(uid, "-") for uid in utt_ids])

    # Bonafide scores (positive class for each attack comparison)
    bona_mask = labels == 1
    bona_scores = scores[bona_mask]

    if len(bona_scores) == 0:
        print(" No bonafide samples found, skipping per-attack analysis.")
        return {}

    results = {}
    unique_attacks = sorted(set(attacks[labels == 0]))  # spoof attack types only

    for attack in unique_attacks:
        attack_mask = (attacks == attack) & (labels == 0)
        attack_scores = scores[attack_mask]

        if len(attack_scores) == 0:
            continue

        # Combine bonafide + this attack's spoof scores
        combined_scores = np.concatenate([bona_scores, attack_scores])
        combined_labels = np.concatenate([
            np.ones(len(bona_scores)),
            np.zeros(len(attack_scores))
        ])

        eer, thresh = compute_eer(combined_scores, combined_labels)
        desc = ATTACK_TYPES.get(attack, "unknown")

        results[attack] = {
            "eer": eer,
            "eer_pct": eer * 100,
            "threshold": thresh,
            "num_samples": int(attack_mask.sum()),
            "mean_score": float(attack_scores.mean()),
            "std_score": float(attack_scores.std()),
            "description": desc,
        }

    return results


def save_score_file(scores, utt_ids, output_path):
    """Save scores in ASVspoof format for official evaluation tools."""
    with open(output_path, "w") as f:
        for uid, score in zip(utt_ids, scores):
            f.write(f"{uid} {score:.6f}\n")
    print(f"  Score file saved: {output_path}")


# ─────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────
def _setup_matplotlib():
    """Import and configure matplotlib. Returns (matplotlib, plt) or (None, None)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError:
        print("  matplotlib not installed, skipping plots.")
        print("    Install with: pip install matplotlib")
        return None, None


def plot_det_curve(scores, labels, output_path):
    """Plot DET (Detection Error Tradeoff) curve."""
    _, plt = _setup_matplotlib()
    if plt is None:
        return

    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        print("  sklearn not installed, skipping DET plot.")
        return

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr * 100, fnr * 100, "b-", linewidth=2, label="AASIST-L")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Random")

    # Mark EER point
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_val = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
    ax.scatter([fpr[eer_idx] * 100], [fnr[eer_idx] * 100],
               color="red", s=120, zorder=5, marker="*", edgecolors="black",
               linewidths=1, label=f"EER = {eer_val:.2f}%")

    ax.set_xlabel("False Acceptance Rate (%)", fontsize=12)
    ax.set_ylabel("False Rejection Rate (%)", fontsize=12)
    ax.set_title("DET Curve — ASVspoof 2019 LA", fontsize=14, fontweight="bold")
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  DET curve saved: {output_path}")


def plot_score_distributions(scores, labels, eer_threshold, output_path):
    """Plot bonafide vs spoof score distributions with EER threshold."""
    _, plt = _setup_matplotlib()
    if plt is None:
        return

    bona_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(bona_scores, bins=80, alpha=0.6, label=f"Bonafide (n={len(bona_scores):,})",
            color="green", density=True)
    ax.hist(spoof_scores, bins=80, alpha=0.6, label=f"Spoof (n={len(spoof_scores):,})",
            color="red", density=True)
    ax.axvline(eer_threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"EER threshold = {eer_threshold:.4f}")

    ax.set_xlabel("Score P(bonafide)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distributions — Eval Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Score distributions saved: {output_path}")


def plot_per_attack_eer(attack_results, output_path):
    """Bar chart of EER per attack type."""
    _, plt = _setup_matplotlib()
    if plt is None or not attack_results:
        return

    attacks = sorted(attack_results.keys())
    eers = [attack_results[a]["eer_pct"] for a in attacks]
    descriptions = [attack_results[a]["description"] for a in attacks]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(attacks)), eers, color="steelblue", edgecolor="black",
                  linewidth=0.5, alpha=0.85)

    # Color worst attacks in red
    if eers:
        max_eer = max(eers)
        for bar, eer in zip(bars, eers):
            if eer > max_eer * 0.8:
                bar.set_color("indianred")

    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([f"{a}\n{d}" for a, d in zip(attacks, descriptions)],
                       fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_title("Per-Attack EER Breakdown", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, eer in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{eer:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add overall mean line
    mean_eer = np.mean(eers)
    ax.axhline(mean_eer, color="red", linestyle="--", alpha=0.5,
               label=f"Mean EER = {mean_eer:.2f}%")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-attack EER chart saved: {output_path}")


def plot_eval_overview(scores, labels, attack_results, eer, min_tdcf, threshold,
                       split_name, output_path):
    """
    Generate a multi-panel evaluation overview plot (2x2 grid).
    Panels: DET curve, Score distributions, Per-attack EER, Summary text.
    """
    _, plt = _setup_matplotlib()
    if plt is None:
        return

    try:
        from sklearn.metrics import roc_curve, confusion_matrix
    except ImportError:
        print(" sklearn not installed, skipping overview plot.")
        return

    bona_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(f"AASIST Evaluation Summary — {split_name.upper()} Set",
                 fontsize=16, fontweight="bold", y=0.98)

    # ── Panel 1: DET Curve ──
    ax = axes[0, 0]
    ax.plot(fpr * 100, fnr * 100, "b-", linewidth=2, label="AASIST-L")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3)
    eer_idx = np.argmin(np.abs(fpr - fnr))
    ax.scatter([fpr[eer_idx] * 100], [fnr[eer_idx] * 100],
               color="red", s=120, zorder=5, marker="*", edgecolors="black")
    ax.set_xlabel("False Acceptance Rate (%)")
    ax.set_ylabel("False Rejection Rate (%)")
    ax.set_title("DET Curve")
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.legend([f"EER = {eer*100:.2f}%"], fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Score Distributions ──
    ax = axes[0, 1]
    ax.hist(bona_scores, bins=80, alpha=0.6, label=f"Bonafide (n={len(bona_scores):,})",
            color="green", density=True)
    ax.hist(spoof_scores, bins=80, alpha=0.6, label=f"Spoof (n={len(spoof_scores):,})",
            color="red", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Score P(bonafide)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Per-Attack EER ──
    ax = axes[1, 0]
    if attack_results:
        attacks = sorted(attack_results.keys())
        eers = [attack_results[a]["eer_pct"] for a in attacks]
        colors = ["indianred" if e > np.mean(eers) else "steelblue" for e in eers]
        bars = ax.bar(range(len(attacks)), eers, color=colors,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels(attacks, fontsize=8, rotation=45, ha="right")
        ax.axhline(np.mean(eers), color="red", linestyle="--", alpha=0.5,
                    label=f"Mean = {np.mean(eers):.2f}%")
        for bar, e in zip(bars, eers):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{e:.1f}", ha="center", va="bottom", fontsize=7)
        ax.legend(fontsize=9)
    ax.set_ylabel("EER (%)")
    ax.set_title("Per-Attack EER")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Summary Metrics ──
    ax = axes[1, 1]
    ax.axis("off")

    # Compute additional metrics
    preds = (scores >= threshold).astype(int)
    accuracy = (preds == labels).mean() * 100
    bona_correct = (preds[labels == 1] == 1).mean() * 100 if (labels == 1).any() else 0
    spoof_correct = (preds[labels == 0] == 0).mean() * 100 if (labels == 0).any() else 0

    summary_lines = [
        ("EER",                     f"{eer*100:.2f}%"),
        ("min t-DCF",               f"{min_tdcf:.4f}"),
        ("EER Threshold",           f"{threshold:.4f}"),
        ("",                        ""),
        ("Total Samples",           f"{len(scores):,}"),
        ("Bonafide Samples",        f"{int((labels==1).sum()):,}"),
        ("Spoof Samples",           f"{int((labels==0).sum()):,}"),
        ("",                        ""),
        ("Accuracy @ EER thresh",   f"{accuracy:.2f}%"),
        ("Bonafide Recall",         f"{bona_correct:.2f}%"),
        ("Spoof Detection Rate",    f"{spoof_correct:.2f}%"),
        ("",                        ""),
        ("Mean Score (bonafide)",   f"{bona_scores.mean():.4f} ± {bona_scores.std():.4f}"),
        ("Mean Score (spoof)",      f"{spoof_scores.mean():.4f} ± {spoof_scores.std():.4f}"),
        ("Score Separation",        f"{bona_scores.mean() - spoof_scores.mean():.4f}"),
    ]

    if attack_results:
        eers_list = [attack_results[a]["eer_pct"] for a in attack_results]
        easiest = min(attack_results, key=lambda a: attack_results[a]["eer_pct"])
        hardest = max(attack_results, key=lambda a: attack_results[a]["eer_pct"])
        summary_lines.extend([
            ("",                        ""),
            ("Easiest Attack",          f"{easiest} ({attack_results[easiest]['eer_pct']:.2f}%)"),
            ("Hardest Attack",          f"{hardest} ({attack_results[hardest]['eer_pct']:.2f}%)"),
            ("Attack EER Range",        f"{min(eers_list):.2f}% – {max(eers_list):.2f}%"),
        ])

    y_pos = 0.95
    ax.text(0.5, 1.0, "Evaluation Metrics Summary",
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            ha="center", va="top")
    for label, value in summary_lines:
        if label == "" and value == "":
            y_pos -= 0.02
            continue
        ax.text(0.08, y_pos, label, transform=ax.transAxes,
                fontsize=10, fontfamily="monospace", va="top")
        ax.text(0.65, y_pos, value, transform=ax.transAxes,
                fontsize=10, fontfamily="monospace", fontweight="bold", va="top")
        y_pos -= 0.045

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Evaluation overview saved: {output_path}")


def plot_training_history(history_path, output_path):
    """Plot training curves from history JSON (if available)."""
    _, plt = _setup_matplotlib()
    if plt is None:
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    dev_eer = [h["dev_eer_pct"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "b-", label="Train Loss")
    if history[0].get("dev_loss") is not None:
        dev_loss = [h["dev_loss"] for h in history]
        ax1.plot(epochs, dev_loss, "r-", label="Dev Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, dev_eer, "g-o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("EER (%)")
    ax2.set_title("Dev EER")
    ax2.grid(True, alpha=0.3)

    best_idx = np.argmin(dev_eer)
    ax2.annotate(f"Best: {dev_eer[best_idx]:.2f}%",
                 xy=(epochs[best_idx], dev_eer[best_idx]),
                 fontsize=10, fontweight="bold", color="red")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {output_path}")


# ─────────────────────────────────────────────
# Metrics Summary
# ─────────────────────────────────────────────
def print_eval_summary(eer, min_tdcf, threshold, scores, labels,
                       attack_results, split_name, train_best_eer=None):
    """Print a comprehensive evaluation metrics summary table."""

    bona_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]
    preds = (scores >= threshold).astype(int)
    accuracy = (preds == labels).mean() * 100

    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │         Evaluation Results — {split_name.upper():>4s} Set                     │")
    print(f"  ├────────────────────────────┬───────────────────────────────┤")
    print(f"  │ Metric                     │ Value                         │")
    print(f"  ├────────────────────────────┼───────────────────────────────┤")
    print(f"  │ EER                        │ {eer*100:>10.2f}%                    │")
    print(f"  │ min t-DCF                  │ {min_tdcf:>10.4f}                     │")
    print(f"  │ EER Threshold              │ {threshold:>10.4f}                     │")
    print(f"  │ Accuracy @ EER thresh      │ {accuracy:>10.2f}%                    │")
    print(f"  ├────────────────────────────┼───────────────────────────────┤")
    print(f"  │ Total Samples              │ {len(scores):>10,}                     │")
    print(f"  │ Bonafide Samples           │ {int((labels==1).sum()):>10,}                     │")
    print(f"  │ Spoof Samples              │ {int((labels==0).sum()):>10,}                     │")
    print(f"  ├────────────────────────────┼───────────────────────────────┤")
    print(f"  │ Mean Score (bonafide)      │ {bona_scores.mean():>7.4f} ± {bona_scores.std():.4f}            │")
    print(f"  │ Mean Score (spoof)         │ {spoof_scores.mean():>7.4f} ± {spoof_scores.std():.4f}            │")
    print(f"  │ Score Separation           │ {bona_scores.mean()-spoof_scores.mean():>10.4f}                     │")

    if train_best_eer is not None:
        delta = eer * 100 - train_best_eer * 100
        direction = "↑" if delta > 0 else "↓"
        print(f"  ├────────────────────────────┼───────────────────────────────┤")
        print(f"  │ Training Best Dev EER       │ {train_best_eer*100:>10.2f}%                    │")
        print(f"  │ Eval vs Dev EER delta       │ {direction} {abs(delta):>8.2f}%                    │")

    print(f"  └────────────────────────────┴───────────────────────────────┘")

    # Per-attack summary
    if attack_results:
        eers_list = [attack_results[a]["eer_pct"] for a in attack_results]
        easiest = min(attack_results, key=lambda a: attack_results[a]["eer_pct"])
        hardest = max(attack_results, key=lambda a: attack_results[a]["eer_pct"])

        print(f"\n  ┌──────────────────────────────────────────────────────────┐")
        print(f"  │         Per-Attack Analysis                              │")
        print(f"  ├────────────────────────────┬───────────────────────────────┤")
        print(f"  │ Mean Attack EER            │ {np.mean(eers_list):>10.2f}%                    │")
        print(f"  │ Median Attack EER          │ {np.median(eers_list):>10.2f}%                    │")
        print(f"  │ Std Attack EER             │ {np.std(eers_list):>10.2f}%                    │")
        print(f"  │ Easiest Attack             │ {easiest:>5s} ({attack_results[easiest]['eer_pct']:.2f}%)               │")
        print(f"  │ Hardest Attack             │ {hardest:>5s} ({attack_results[hardest]['eer_pct']:.2f}%)               │")
        print(f"  └────────────────────────────┴───────────────────────────────┘")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AASIST model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="eval",
                        choices=["dev", "eval"],
                        help="Which split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_preprocessed", action="store_true")
    parser.add_argument("--plot", action="store_true",
                        help="Generate DET curve, score distribution, and overview plots")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("╔" + "═" * 58 + "╗")
    print("║   AASIST Evaluation — ASVspoof 2019 LA                   ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Device: {device}")
    print(f"  Split:  {args.split}")
    print()

    # ── Load model ──
    model, train_args, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    train_best_eer = ckpt.get("best_eer", None)

    # ── Load data ──
    if args.split == "eval":
        protocol, flac_dir = EVAL_PROTOCOL, EVAL_FLAC_DIR
        npy_dir = EVAL_OUTPUT_DIR
        meta_csv = METADATA_OUTPUT_DIR / "eval_metadata.csv"
    else:
        protocol, flac_dir = DEV_PROTOCOL, DEV_FLAC_DIR
        npy_dir = DEV_OUTPUT_DIR
        meta_csv = METADATA_OUTPUT_DIR / "dev_metadata.csv"

    if args.use_preprocessed:
        # Check if preprocessed files actually exist before trying
        if not npy_dir.exists() or not any(npy_dir.glob("*.npy")):
            print(f"\n  ⚠ No preprocessed .npy files found in: {npy_dir}")
            print(f"    Falling back to raw .flac loading...")
            print(f"    (To use preprocessed data, run: python preprocess_audio.py --split {args.split})")
            print()
            args.use_preprocessed = False

    if args.use_preprocessed:
        dataset = ASVspoofDataset(metadata_csv=meta_csv, npy_dir=npy_dir)
    else:
        dataset = ASVspoofRawDataset(protocol_file=protocol, flac_dir=flac_dir)

    # ── Guard: check dataset is not empty ──
    if len(dataset) == 0:
        print(f"\n  ✗ ERROR: Dataset is empty (0 utterances loaded).")
        print(f"    Possible causes:")
        print(f"      1. Preprocessed .npy files not found in: {npy_dir}")
        print(f"         Run: python preprocess_audio.py --split {args.split}")
        print(f"      2. Raw .flac files not found in: {flac_dir}")
        print(f"         Check that EVAL_FLAC_DIR in config.py is correct")
        print(f"      3. Protocol file missing or empty: {protocol}")
        print(f"\n    Try running without --use_preprocessed to load from raw .flac files:")
        print(f"      python evaluate.py --checkpoint {args.checkpoint} --split {args.split} --plot")
        return

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"  Evaluating on {len(dataset):,} utterances...")

    # ── Run inference ──
    scores, labels, utt_ids = run_inference(model, loader, device)

    # ── Guard: check we have both classes ──
    n_bona = (labels == 1).sum()
    n_spoof = (labels == 0).sum()
    if n_bona == 0 or n_spoof == 0:
        print(f"\n  ERROR: Need both bonafide and spoof samples to compute metrics.")
        print(f"    Found: {n_bona} bonafide, {n_spoof} spoof")
        return

    # ── Overall metrics ──
    eer, threshold = compute_eer(scores, labels)
    min_tdcf = compute_t_dcf(scores, labels)

    # ── Per-attack analysis ──
    utt_to_attack = get_attack_types_from_protocol(protocol)
    attack_results = per_attack_analysis(scores, labels, utt_ids, utt_to_attack)

    # ── Print detailed per-attack table ──
    print(f"\n  {'─'*60}")
    print(f"  Per-Attack EER Breakdown")
    print(f"  {'─'*60}")
    if attack_results:
        print(f"  {'Attack':>6s}  {'EER%':>7s}  {'Count':>6s}  {'Mean Score':>10s}  {'Std':>6s}  Description")
        print(f"  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*10}  {'─'*6}  {'─'*25}")

        for attack in sorted(attack_results.keys()):
            r = attack_results[attack]
            print(f"  {attack:>6s}  {r['eer_pct']:6.2f}%  {r['num_samples']:6d}  "
                  f"{r['mean_score']:10.4f}  {r['std_score']:6.4f}  {r['description']}")

    # ── Print summary table ──
    print_eval_summary(eer, min_tdcf, threshold, scores, labels,
                       attack_results, args.split, train_best_eer)

    # ── Save score file ──
    score_path = output_dir / f"scores_{args.split}.txt"
    save_score_file(scores, utt_ids, score_path)

    # ── Save results JSON ──
    results = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "timestamp": timestamp,
        "num_samples": len(scores),
        "num_bonafide": int(n_bona),
        "num_spoof": int(n_spoof),
        "eer": eer,
        "eer_pct": eer * 100,
        "min_tdcf": min_tdcf,
        "threshold": threshold,
        "accuracy_at_threshold": float(((scores >= threshold).astype(int) == labels).mean()),
        "mean_score_bonafide": float(scores[labels == 1].mean()),
        "mean_score_spoof": float(scores[labels == 0].mean()),
        "score_separation": float(scores[labels == 1].mean() - scores[labels == 0].mean()),
        "per_attack": attack_results,
    }
    results_path = output_dir / f"results_{args.split}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")

    # ── Generate plots ──
    if args.plot:
        print(f"\n  Generating plots...")

        # DET curve
        det_path = output_dir / f"det_curve_{args.split}_{timestamp}.png"
        plot_det_curve(scores, labels, det_path)

        # Score distributions
        dist_path = output_dir / f"score_distributions_{args.split}_{timestamp}.png"
        plot_score_distributions(scores, labels, threshold, dist_path)

        # Per-attack EER bar chart
        if attack_results:
            attack_path = output_dir / f"per_attack_eer_{args.split}_{timestamp}.png"
            plot_per_attack_eer(attack_results, attack_path)

        # Full evaluation overview
        overview_path = output_dir / f"eval_overview_{args.split}_{timestamp}.png"
        plot_eval_overview(scores, labels, attack_results, eer, min_tdcf,
                           threshold, args.split, overview_path)

        # Training history (if available near checkpoint)
        ckpt_dir = Path(args.checkpoint).parent
        for hist_file in sorted(ckpt_dir.glob("history_*.json")):
            hist_plot_path = output_dir / "training_curves.png"
            plot_training_history(hist_file, hist_plot_path)
            break

    print(f"\n  Evaluation complete!")


if __name__ == "__main__":
    main()
"""
PA (Replay Attack) Evaluation Script
======================================
Evaluate a trained AASIST model on ASVspoof 2019 PA dev/eval sets.

Produces:
  - Overall EER and min t-DCF
  - Per-attack-configuration EER breakdown (AA-CC and unseen eval configs)
  - DET curve plot
  - Score distribution plot
  - Evaluation overview (multi-panel)

Usage:
    python evaluate_pa.py --checkpoint checkpoints_pa/best_model.pt
    python evaluate_pa.py --checkpoint checkpoints_pa/best_model.pt --split dev --plot
    python evaluate_pa.py --checkpoint checkpoints_pa/best_model.pt --split eval --plot
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime

from config_pa import (
    DEV_PROTOCOL, DEV_FLAC_DIR,
    EVAL_PROTOCOL, EVAL_FLAC_DIR,
    METADATA_OUTPUT_DIR as PA_METADATA_DIR,
    DEV_OUTPUT_DIR as PA_DEV_OUTPUT_DIR,
    EVAL_OUTPUT_DIR as PA_EVAL_OUTPUT_DIR,
    PA_ATTACK_TYPES,
)
from dataset import ASVspoofRawDataset, ASVspoofDataset
from aasist_model import build_model
from train import compute_eer, compute_t_dcf


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model and config from a PA training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    variant = args.get("variant", "AASIST-L")

    model = build_model(variant).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Loaded {variant} from epoch {ckpt.get('epoch', -1) + 1}")
    print(f"  Training best EER: {ckpt.get('best_eer', 0)*100:.2f}%")
    print(f"  Task: {ckpt.get('task', 'unknown')}")

    return model, args, ckpt


@torch.no_grad()
def run_inference(model, loader, device):
    """Run model inference, returning scores, labels, and utterance IDs."""
    model.eval()
    all_scores, all_labels, all_utt_ids = [], [], []

    for x, y, utt_ids in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().numpy()

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
                utt_to_attack[parts[1]] = parts[3]
    return utt_to_attack


def per_attack_analysis(scores, labels, utt_ids, utt_to_attack):
    """Compute EER for each PA attack configuration separately."""
    attacks = np.array([utt_to_attack.get(uid, "-") for uid in utt_ids])

    bona_mask = labels == 1
    bona_scores = scores[bona_mask]

    if len(bona_scores) == 0:
        print("  No bonafide samples found.")
        return {}

    results = {}
    unique_attacks = sorted(set(attacks[labels == 0]))

    for attack in unique_attacks:
        attack_mask = (attacks == attack) & (labels == 0)
        attack_scores = scores[attack_mask]

        if len(attack_scores) == 0:
            continue

        combined_scores = np.concatenate([bona_scores, attack_scores])
        combined_labels = np.concatenate([
            np.ones(len(bona_scores)),
            np.zeros(len(attack_scores))
        ])

        eer, thresh = compute_eer(combined_scores, combined_labels)
        desc = PA_ATTACK_TYPES.get(attack, f"replay config {attack}")

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


def plot_det_curve(scores, labels, output_path):
    """Plot DET curve for PA evaluation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        print("  matplotlib or sklearn not installed, skipping plot.")
        return

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr * 100, fnr * 100, "b-", linewidth=2, label="AASIST (PA)")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Random")

    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_val = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
    ax.scatter([fpr[eer_idx] * 100], [fnr[eer_idx] * 100],
               color="red", s=120, zorder=5, marker="*", edgecolors="black",
               linewidths=1, label=f"EER = {eer_val:.2f}%")

    ax.set_xlabel("False Acceptance Rate (%)", fontsize=12)
    ax.set_ylabel("False Rejection Rate (%)", fontsize=12)
    ax.set_title("DET Curve -- ASVspoof 2019 PA (Replay)", fontsize=14, fontweight="bold")
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  DET curve saved: {output_path}")


def plot_score_distributions(scores, labels, threshold, output_path):
    """Plot bonafide vs replay score distributions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    bona_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(bona_scores, bins=80, alpha=0.6,
            label=f"Bonafide (n={len(bona_scores):,})", color="green", density=True)
    ax.hist(spoof_scores, bins=80, alpha=0.6,
            label=f"Replay (n={len(spoof_scores):,})", color="red", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"EER threshold = {threshold:.4f}")

    ax.set_xlabel("Score P(bonafide)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distributions -- PA (Replay)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Score distributions saved: {output_path}")


def plot_per_attack_eer(attack_results, output_path):
    """Bar chart of EER per replay configuration."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not attack_results:
        return

    attacks = sorted(attack_results.keys())
    eers = [attack_results[a]["eer_pct"] for a in attacks]
    descriptions = [attack_results[a]["description"] for a in attacks]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(attacks)), eers, color="steelblue",
                  edgecolor="black", linewidth=0.5, alpha=0.85)

    if eers:
        max_eer = max(eers)
        for bar, eer in zip(bars, eers):
            if eer > max_eer * 0.8:
                bar.set_color("indianred")

    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([f"{a}\n{d}" for a, d in zip(attacks, descriptions)],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_title("Per-Configuration EER -- PA Replay Attacks",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, eer in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{eer:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    mean_eer = np.mean(eers)
    ax.axhline(mean_eer, color="red", linestyle="--", alpha=0.5,
               label=f"Mean EER = {mean_eer:.2f}%")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-attack EER chart saved: {output_path}")


def save_score_file(scores, utt_ids, output_path):
    """Save scores in ASVspoof format."""
    with open(output_path, "w") as f:
        for uid, score in zip(utt_ids, scores):
            f.write(f"{uid} {score:.6f}\n")
    print(f"  Score file saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AASIST on PA (replay) data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PA model checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="eval",
                        choices=["dev", "eval"])
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_preprocessed", action="store_true")
    parser.add_argument("--plot", action="store_true",
                        help="Generate DET, score distribution, and per-attack plots")
    parser.add_argument("--output_dir", type=str, default="results_pa")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  AASIST Evaluation -- ASVspoof 2019 PA (Replay)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Split:  {args.split}")
    print()

    # ── Load model ──
    model, train_args, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    train_best_eer = ckpt.get("best_eer", None)

    # ── Load data ──
    if args.split == "eval":
        protocol, flac_dir = EVAL_PROTOCOL, EVAL_FLAC_DIR
        npy_dir = PA_EVAL_OUTPUT_DIR
        meta_csv = PA_METADATA_DIR / "eval_metadata.csv"
    else:
        protocol, flac_dir = DEV_PROTOCOL, DEV_FLAC_DIR
        npy_dir = PA_DEV_OUTPUT_DIR
        meta_csv = PA_METADATA_DIR / "dev_metadata.csv"

    if args.use_preprocessed and npy_dir.exists() and any(npy_dir.glob("*.npy")):
        dataset = ASVspoofDataset(metadata_csv=meta_csv, npy_dir=npy_dir)
    else:
        if args.use_preprocessed:
            print(f"  No preprocessed files in {npy_dir}, falling back to raw .flac")
        dataset = ASVspoofRawDataset(protocol_file=protocol, flac_dir=flac_dir)

    if len(dataset) == 0:
        print(f"  ERROR: Dataset is empty. Check paths in config_pa.py.")
        return

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"  Evaluating on {len(dataset):,} utterances...")

    # ── Run inference ──
    scores, labels, utt_ids = run_inference(model, loader, device)

    n_bona = (labels == 1).sum()
    n_spoof = (labels == 0).sum()
    if n_bona == 0 or n_spoof == 0:
        print(f"  ERROR: Need both classes. Found: {n_bona} bonafide, {n_spoof} spoof")
        return

    # ── Overall metrics ──
    eer, threshold = compute_eer(scores, labels)
    min_tdcf = compute_t_dcf(scores, labels)

    # ── Per-attack analysis ──
    utt_to_attack = get_attack_types_from_protocol(protocol)
    attack_results = per_attack_analysis(scores, labels, utt_ids, utt_to_attack)

    # ── Print per-attack table ──
    print(f"\n  {'-'*60}")
    print(f"  Per-Configuration EER Breakdown")
    print(f"  {'-'*60}")
    if attack_results:
        print(f"  {'Config':>6s}  {'EER%':>7s}  {'Count':>6s}  "
              f"{'Mean Score':>10s}  {'Std':>6s}  Description")
        print(f"  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*10}  {'-'*6}  {'-'*30}")

        for attack in sorted(attack_results.keys()):
            r = attack_results[attack]
            print(f"  {attack:>6s}  {r['eer_pct']:6.2f}%  {r['num_samples']:6d}  "
                  f"{r['mean_score']:10.4f}  {r['std_score']:6.4f}  {r['description']}")

    # ── Print summary ──
    bona_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]
    preds = (scores >= threshold).astype(int)
    accuracy = (preds == labels).mean() * 100

    print(f"\n  {'='*60}")
    print(f"  PA Evaluation Results -- {args.split.upper()} Set")
    print(f"  {'='*60}")
    print(f"  EER:                    {eer*100:.2f}%")
    print(f"  min t-DCF:              {min_tdcf:.4f}")
    print(f"  EER Threshold:          {threshold:.4f}")
    print(f"  Accuracy @ EER thresh:  {accuracy:.2f}%")
    print(f"  Total Samples:          {len(scores):,}")
    print(f"  Bonafide:               {int(n_bona):,}")
    print(f"  Replay (spoof):         {int(n_spoof):,}")
    print(f"  Mean Score (bonafide):  {bona_scores.mean():.4f} +/- {bona_scores.std():.4f}")
    print(f"  Mean Score (replay):    {spoof_scores.mean():.4f} +/- {spoof_scores.std():.4f}")
    print(f"  Score Separation:       {bona_scores.mean() - spoof_scores.mean():.4f}")

    if attack_results:
        eers_list = [attack_results[a]["eer_pct"] for a in attack_results]
        easiest = min(attack_results, key=lambda a: attack_results[a]["eer_pct"])
        hardest = max(attack_results, key=lambda a: attack_results[a]["eer_pct"])
        print(f"  Mean Attack EER:        {np.mean(eers_list):.2f}%")
        print(f"  Easiest Config:         {easiest} ({attack_results[easiest]['eer_pct']:.2f}%)")
        print(f"  Hardest Config:         {hardest} ({attack_results[hardest]['eer_pct']:.2f}%)")

    if train_best_eer is not None:
        delta = eer * 100 - train_best_eer * 100
        direction = "higher" if delta > 0 else "lower"
        print(f"  Training Best Dev EER:  {train_best_eer*100:.2f}%")
        print(f"  Eval vs Dev delta:      {abs(delta):.2f}% {direction}")

    print(f"  {'='*60}")

    # ── Save score file ──
    score_path = output_dir / f"scores_pa_{args.split}.txt"
    save_score_file(scores, utt_ids, score_path)

    # ── Save results JSON ──
    results = {
        "task": "PA",
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
        "accuracy_at_threshold": float(accuracy),
        "mean_score_bonafide": float(bona_scores.mean()),
        "mean_score_spoof": float(spoof_scores.mean()),
        "per_attack": attack_results,
    }
    results_path = output_dir / f"results_pa_{args.split}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")

    # ── Plots ──
    if args.plot:
        print(f"\n  Generating plots...")
        det_path = output_dir / f"det_curve_pa_{args.split}_{timestamp}.png"
        plot_det_curve(scores, labels, det_path)

        dist_path = output_dir / f"score_dist_pa_{args.split}_{timestamp}.png"
        plot_score_distributions(scores, labels, threshold, dist_path)

        if attack_results:
            attack_path = output_dir / f"per_attack_eer_pa_{args.split}_{timestamp}.png"
            plot_per_attack_eer(attack_results, attack_path)

    print(f"\n  PA evaluation complete.")


if __name__ == "__main__":
    main()

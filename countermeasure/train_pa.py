"""
PA (Replay Attack) Training Script
====================================
Trains AASIST on ASVspoof 2019 PA data for replay/physical-access detection.

Reuses the same model architecture and training loop from the LA pipeline.
Only the data paths and checkpoint directory differ.

Usage:
    python train_pa.py                                  # Train AASIST-L
    python train_pa.py --variant AASIST                 # Full AASIST
    python train_pa.py --epochs 50 --batch_size 24
    python train_pa.py --use_preprocessed               # Use .npy (run preprocess first)
    python train_pa.py --subset 5000                    # Quick experiment

Before running:
    1. Verify paths:   python config_pa.py
    2. (Optional) Preprocess for speed:
       python preprocess_audio_pa.py
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# PA-specific paths
from config_pa import (
    TRAIN_PROTOCOL, TRAIN_FLAC_DIR,
    DEV_PROTOCOL, DEV_FLAC_DIR,
    METADATA_OUTPUT_DIR as PA_METADATA_DIR,
    TRAIN_OUTPUT_DIR as PA_TRAIN_OUTPUT_DIR,
    DEV_OUTPUT_DIR as PA_DEV_OUTPUT_DIR,
)

# Reuse everything else from existing codebase
from dataset import ASVspoofRawDataset, ASVspoofDataset
from aasist_model import build_model
from train import (
    compute_eer,
    compute_t_dcf,
    evaluate,
    train_one_epoch,
    WarmupCosineScheduler,
    plot_training_curves,
    print_best_metrics_summary,
)


def main():
    parser = argparse.ArgumentParser(description="Train AASIST on PA (replay) data")
    parser.add_argument("--variant", type=str, default="AASIST-L",
                        choices=["AASIST", "AASIST-L"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--spoof_weight", type=float, default=1.0,
                        help="CE weight for spoof (replay) class")
    parser.add_argument("--bonafide_weight", type=float, default=9.0,
                        help="CE weight for bonafide class")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit samples per split (quick experiments)")
    parser.add_argument("--use_preprocessed", action="store_true",
                        help="Use .npy files instead of raw .flac")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_pa")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--use_amp", action="store_true")
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

    print("=" * 60)
    print("  PA Replay Detection -- AASIST Training")
    print("=" * 60)
    print(f"  Model:        {args.variant}")
    print(f"  Device:       {device}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate:{args.lr}")
    print(f"  CE weights:   spoof={args.spoof_weight}, bonafide={args.bonafide_weight}")
    print(f"  Checkpoints:  {checkpoint_dir}")
    if args.subset:
        print(f"  Subset:       {args.subset} samples per split")
    print()

    # ── Data ──
    print("Loading PA datasets...")
    if args.use_preprocessed:
        train_dataset = ASVspoofDataset(
            metadata_csv=PA_METADATA_DIR / "train_metadata.csv",
            npy_dir=PA_TRAIN_OUTPUT_DIR,
            subset_size=args.subset,
        )
        dev_dataset = ASVspoofDataset(
            metadata_csv=PA_METADATA_DIR / "dev_metadata.csv",
            npy_dir=PA_DEV_OUTPUT_DIR,
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
    best_eer = float("inf")
    history = []

    if args.resume:
        print(f"  Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer = ckpt.get("best_eer", float("inf"))
        history = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch}, best EER = {best_eer*100:.2f}%")

    # ── TensorBoard ──
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = checkpoint_dir / f"logs_{timestamp}"
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"  TensorBoard: {log_dir}")
    except ImportError:
        pass

    # ── Training loop ──
    print("\n" + "-" * 60)
    print("  Starting PA training...")
    print("-" * 60)

    patience_counter = 0
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()[0]

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            max_grad_norm=args.max_grad_norm, use_amp=args.use_amp,
        )
        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch {epoch+1:3d}/{args.epochs} "
              f"({epoch_time:.1f}s) | "
              f"lr={current_lr:.2e} | "
              f"loss={train_metrics['loss']:.4f} | "
              f"acc={train_metrics['accuracy']:.4f} | "
              f"grad={train_metrics['grad_norm']:.4f}")

        if (epoch + 1) % args.eval_every == 0:
            dev_metrics = evaluate(model, dev_loader, device, criterion)

            print(f"         Dev   | "
                  f"EER={dev_metrics['eer_pct']:6.2f}% | "
                  f"min-tDCF={dev_metrics['min_tdcf']:.4f} | "
                  f"loss={dev_metrics['loss']:.4f} | "
                  f"acc={dev_metrics['accuracy']:.4f}")
            print(f"               | "
                  f"score(bona)={dev_metrics['mean_score_bonafide']:.4f} | "
                  f"score(spoof)={dev_metrics['mean_score_spoof']:.4f}")

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

            if writer:
                writer.add_scalar("Train/Loss", train_metrics["loss"], epoch)
                writer.add_scalar("Train/Accuracy", train_metrics["accuracy"], epoch)
                writer.add_scalar("Dev/EER", dev_metrics["eer_pct"], epoch)
                writer.add_scalar("Dev/MinTDCF", dev_metrics["min_tdcf"], epoch)
                writer.add_scalar("Dev/Loss", dev_metrics["loss"], epoch)
                writer.add_scalar("LR", current_lr, epoch)

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
                    "task": "PA",  # tag so pipeline knows this is a PA model
                }, ckpt_path)
                print(f"         * New best EER={best_eer*100:.2f}% -> saved {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n  Early stopping after {args.patience} epochs without improvement")
                    break

    # ── Save final ──
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eer": best_eer,
        "args": vars(args),
        "history": history,
        "task": "PA",
    }, final_path)

    history_path = checkpoint_dir / f"history_{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - training_start

    if writer:
        writer.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  PA Training Complete")
    print("=" * 60)
    print(f"  Total time:    {total_time/60:.1f} minutes")
    print(f"  Best dev EER:  {best_eer*100:.2f}%")
    print(f"  Best model:    {checkpoint_dir / 'best_model.pt'}")
    print(f"  Final model:   {final_path}")
    print(f"  History:       {history_path}")

    print_best_metrics_summary(history)

    print("\n  Generating training plots...")
    plot_training_curves(history, checkpoint_dir, timestamp)

    print(f"\n  Next steps:")
    print(f"  1. Evaluate:")
    print(f"     python evaluate_pa.py --checkpoint {checkpoint_dir / 'best_model.pt'} --plot")
    print(f"  2. Use in pipeline:")
    print(f"     python pipeline.py --checkpoint checkpoints/best_model.pt \\")
    print(f"                        --pa_checkpoint {checkpoint_dir / 'best_model.pt'} \\")
    print(f"                        --test_audio <file> --ref_audio <file>")


if __name__ == "__main__":
    main()

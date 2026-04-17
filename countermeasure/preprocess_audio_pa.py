"""
Audio Preprocessor for ASVspoof 2019 PA (Replay)
==================================================
Preprocesses PA partition audio files (resample, pad/truncate, save as .npy).
Optional but speeds up training significantly.

Usage:
    python preprocess_audio_pa.py
    python preprocess_audio_pa.py --subset 500          # Quick test
    python preprocess_audio_pa.py --split train         # One split only
    python preprocess_audio_pa.py --workers 8           # More parallelism
"""

import argparse
import pandas as pd
from pathlib import Path

from config_pa import (
    TRAIN_PROTOCOL, TRAIN_FLAC_DIR,
    DEV_PROTOCOL, DEV_FLAC_DIR,
    EVAL_PROTOCOL, EVAL_FLAC_DIR,
    TRAIN_OUTPUT_DIR, DEV_OUTPUT_DIR, EVAL_OUTPUT_DIR,
    METADATA_OUTPUT_DIR,
    TARGET_SAMPLE_RATE, MAX_AUDIO_LENGTH, PAD_MODE,
    LABEL_MAP,
    create_output_dirs, validate_paths,
)

# Reuse the actual preprocessing functions from the LA pipeline.
# These are generic (no LA-specific logic).
from preprocess_audio import preprocess_split


def parse_pa_protocol(protocol_path, flac_dir):
    """Parse a PA protocol file into a DataFrame."""
    records = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            records.append({
                "speaker_id":  parts[0],
                "utt_id":      parts[1],
                "attack_type": parts[3],
                "label_str":   parts[4],
                "label":       LABEL_MAP[parts[4]],
                "flac_path":   str(flac_dir / f"{parts[1]}.flac"),
            })
    return pd.DataFrame(records)


def save_metadata(protocols):
    """Save parsed protocol DataFrames as CSV."""
    METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, df in protocols.items():
        out = METADATA_OUTPUT_DIR / f"{split}_metadata.csv"
        df.to_csv(out, index=False)
        print(f"  Saved {split} metadata -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ASVspoof 2019 PA audio")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "dev", "eval", "all"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("  ASVspoof 2019 PA -- Audio Preprocessing")
    print("=" * 60)

    if not validate_paths():
        return

    create_output_dirs()

    # Parse protocols
    print("\nParsing PA protocol files...")
    protocols = {}
    for split, proto, flac_dir in [
        ("train", TRAIN_PROTOCOL, TRAIN_FLAC_DIR),
        ("dev",   DEV_PROTOCOL,   DEV_FLAC_DIR),
        ("eval",  EVAL_PROTOCOL,  EVAL_FLAC_DIR),
    ]:
        df = parse_pa_protocol(proto, flac_dir)
        protocols[split] = df
        n_bona = (df["label_str"] == "bonafide").sum()
        n_spoof = (df["label_str"] == "spoof").sum()
        print(f"  {split:>5s}: {len(df):>7,} total | "
              f"bonafide: {n_bona:,} | spoof: {n_spoof:,}")

    save_metadata(protocols)

    # Preprocess
    split_configs = {
        "train": (protocols["train"], TRAIN_OUTPUT_DIR),
        "dev":   (protocols["dev"],   DEV_OUTPUT_DIR),
        "eval":  (protocols["eval"],  EVAL_OUTPUT_DIR),
    }

    splits = [args.split] if args.split != "all" else ["train", "dev", "eval"]

    for split_name in splits:
        df, out_dir = split_configs[split_name]
        preprocess_split(
            df, out_dir, split_name,
            subset_size=args.subset,
            num_workers=args.workers,
        )

    print("\n" + "=" * 60)
    print("  PA preprocessing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Protocol Parser for ASVspoof 2019 LA
=====================================
Reads the CM (countermeasure) protocol files and produces structured metadata.

Protocol file format (space-separated):
    SPEAKER_ID  AUDIO_FILENAME  -  ATTACK_TYPE  LABEL
    
Example lines:
    LA_0079  LA_T_1138215  -  A06  spoof
    LA_0079  LA_T_1139806  -  -    bonafide
"""

import pandas as pd
from pathlib import Path
from collections import Counter
from config import (
    TRAIN_PROTOCOL, DEV_PROTOCOL, EVAL_PROTOCOL,
    LABEL_MAP, ATTACK_TYPES, METADATA_OUTPUT_DIR,
    TRAIN_FLAC_DIR, DEV_FLAC_DIR, EVAL_FLAC_DIR,
)


def parse_protocol(protocol_path: Path, flac_dir: Path) -> pd.DataFrame:
    """
    Parse an ASVspoof 2019 LA protocol file into a DataFrame.
    
    Parameters
    ----------
    protocol_path : Path
        Path to the protocol .txt file.
    flac_dir : Path
        Path to the directory containing the .flac audio files.
    
    Returns
    -------
    pd.DataFrame with columns:
        speaker_id, utt_id, attack_type, label_str, label, flac_path
    """
    records = []
    
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            speaker_id  = parts[0]
            utt_id      = parts[1]
            # parts[2] is always "-" (separator)
            attack_type = parts[3]
            label_str   = parts[4]
            
            flac_path = flac_dir / f"{utt_id}.flac"
            
            records.append({
                "speaker_id":  speaker_id,
                "utt_id":      utt_id,
                "attack_type": attack_type,
                "label_str":   label_str,
                "label":       LABEL_MAP[label_str],
                "flac_path":   str(flac_path),
            })
    
    df = pd.DataFrame(records)
    return df


def print_dataset_stats(df: pd.DataFrame, name: str):
    """Print summary statistics for a parsed protocol DataFrame."""
    
    print(f"\n{'='*60}")
    print(f"  {name} Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Total utterances:  {len(df):,}")
    print(f"  Unique speakers:   {df['speaker_id'].nunique()}")
    
    # Label distribution
    label_counts = df["label_str"].value_counts()
    print(f"\n  Label distribution:")
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"    {label:>10s}: {count:>7,}  ({pct:.1f}%)")
    
    # Attack type distribution
    attack_counts = df["attack_type"].value_counts().sort_index()
    print(f"\n  Attack types:")
    for atype, count in attack_counts.items():
        desc = ATTACK_TYPES.get(atype, "unknown")
        print(f"    {atype:>4s}: {count:>7,}  — {desc}")
    
    # Check for missing files
    missing = df[~df["flac_path"].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        print(f"\n  Missing audio files: {len(missing):,}")
        print(f"    First few: {missing['utt_id'].head(3).tolist()}")
    else:
        print(f"\n  All {len(df):,} audio files found on disk.")
    
    print(f"{'='*60}")


def load_all_protocols(verbose: bool = True):
    """
    Load train, dev, and eval protocol files.
    
    Returns
    -------
    dict with keys 'train', 'dev', 'eval', each containing a DataFrame.
    """
    protocols = {}
    
    for split, proto_path, flac_dir in [
        ("train", TRAIN_PROTOCOL, TRAIN_FLAC_DIR),
        ("dev",   DEV_PROTOCOL,   DEV_FLAC_DIR),
        ("eval",  EVAL_PROTOCOL,  EVAL_FLAC_DIR),
    ]:
        df = parse_protocol(proto_path, flac_dir)
        protocols[split] = df
        
        if verbose:
            print_dataset_stats(df, split.upper())
    
    return protocols


def save_metadata(protocols: dict):
    """Save parsed protocol DataFrames as CSV for later use."""
    METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for split, df in protocols.items():
        out_path = METADATA_OUTPUT_DIR / f"{split}_metadata.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {split} metadata → {out_path}")


if __name__ == "__main__":
    protocols = load_all_protocols(verbose=True)
    save_metadata(protocols)
    
    
    print("\n\nSample rows from TRAIN:")
    print(protocols["train"].head(10).to_string(index=False))

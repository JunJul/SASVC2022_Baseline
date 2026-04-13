"""
PyTorch Dataset for ASVspoof 2019 LA
=====================================
Provides two dataset variants:
  1. ASVspoofDataset       — loads preprocessed .npy files (fast)
  2. ASVspoofRawDataset    — loads from raw .flac on the fly (no preprocessing needed)

Both are compatible with AASIST's expected input format.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
from pathlib import Path

from config import (
    TARGET_SAMPLE_RATE, MAX_AUDIO_LENGTH, PAD_MODE, LABEL_MAP,
    TRAIN_OUTPUT_DIR, DEV_OUTPUT_DIR, EVAL_OUTPUT_DIR,
    METADATA_OUTPUT_DIR,
)


class ASVspoofDataset(Dataset):
    """
    Dataset that loads preprocessed .npy waveforms.
    
    Use this AFTER running preprocess_audio.py.
    
    Parameters
    ----------
    metadata_csv : str or Path
        Path to the metadata CSV (from protocol_parser.py).
    npy_dir : str or Path
        Directory containing preprocessed .npy files.
    subset_size : int or None
        If set, use only this many samples (stratified).
    """
    
    def __init__(self, metadata_csv, npy_dir, subset_size=None):
        self.npy_dir = Path(npy_dir)
        self.df = pd.read_csv(metadata_csv)
        
        # Filter to only files that exist in npy_dir
        self.df["npy_path"] = self.df["utt_id"].apply(
            lambda uid: str(self.npy_dir / f"{uid}.npy")
        )
        self.df = self.df[
            self.df["npy_path"].apply(lambda p: Path(p).exists())
        ].reset_index(drop=True)
        
        if subset_size is not None and subset_size < len(self.df):
            self.df = self.df.groupby("label_str", group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), subset_size // 2), random_state=42)
            ).reset_index(drop=True)
        
        print(f"  ASVspoofDataset: {len(self.df):,} utterances from {self.npy_dir}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load preprocessed audio
        audio = np.load(row["npy_path"]).astype(np.float32)
        
        # Convert to tensor
        x = torch.FloatTensor(audio)
        y = torch.LongTensor([row["label"]])[0]  # scalar tensor
        
        return x, y, row["utt_id"]


class ASVspoofRawDataset(Dataset):
    """
    Dataset that loads and preprocesses .flac files on the fly.
    
    Use this for quick experiments WITHOUT running preprocess_audio.py first.
    Slower than ASVspoofDataset but requires no disk preprocessing.
    
    Parameters
    ----------
    protocol_file : str or Path
        Path to the ASVspoof protocol file.
    flac_dir : str or Path
        Directory containing .flac audio files.
    max_len : int
        Maximum audio length in samples.
    target_sr : int
        Target sample rate.
    subset_size : int or None
        If set, use only this many samples.
    """
    
    def __init__(self, protocol_file, flac_dir, max_len=MAX_AUDIO_LENGTH,
                 target_sr=TARGET_SAMPLE_RATE, subset_size=None):
        self.flac_dir = Path(flac_dir)
        self.max_len = max_len
        self.target_sr = target_sr
        
        # Parse protocol file directly
        records = []
        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    records.append({
                        "speaker_id":  parts[0],
                        "utt_id":      parts[1],
                        "attack_type": parts[3],
                        "label_str":   parts[4],
                        "label":       LABEL_MAP[parts[4]],
                    })
        
        self.df = pd.DataFrame(records)
        
        if subset_size is not None and subset_size < len(self.df):
            self.df = self.df.groupby("label_str", group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), subset_size // 2), random_state=42)
            ).reset_index(drop=True)
        
        print(f"  ASVspoofRawDataset: {len(self.df):,} utterances from {self.flac_dir}")
    
    def __len__(self):
        return len(self.df)
    
    def _pad_or_truncate(self, audio):
        length = len(audio)
        if length >= self.max_len:
            return audio[:self.max_len]
        repeats = (self.max_len // length) + 1
        return np.tile(audio, repeats)[:self.max_len]
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        flac_path = self.flac_dir / f"{row['utt_id']}.flac"
        
        # Load and resample
        audio, _ = librosa.load(str(flac_path), sr=self.target_sr, mono=True)
        audio = self._pad_or_truncate(audio).astype(np.float32)
        
        x = torch.FloatTensor(audio)
        y = torch.LongTensor([row["label"]])[0]
        
        return x, y, row["utt_id"]


def get_dataloaders(batch_size=16, subset_size=None, use_preprocessed=True,
                    num_workers=2):
    """
    Convenience function to create train/dev/eval DataLoaders.
    
    Parameters
    ----------
    batch_size : int
    subset_size : int or None
        Limit samples per split.
    use_preprocessed : bool
        If True, load from .npy files (must run preprocess_audio.py first).
        If False, load from raw .flac files on the fly.
    num_workers : int
        DataLoader workers.
    
    Returns
    -------
    dict with keys 'train', 'dev', 'eval', each a DataLoader.
    """
    from config import (
        TRAIN_PROTOCOL, DEV_PROTOCOL, EVAL_PROTOCOL,
        TRAIN_FLAC_DIR, DEV_FLAC_DIR, EVAL_FLAC_DIR,
    )
    
    loaders = {}
    
    splits = {
        "train": {
            "meta_csv":   METADATA_OUTPUT_DIR / "train_metadata.csv",
            "npy_dir":    TRAIN_OUTPUT_DIR,
            "protocol":   TRAIN_PROTOCOL,
            "flac_dir":   TRAIN_FLAC_DIR,
            "shuffle":    True,
        },
        "dev": {
            "meta_csv":   METADATA_OUTPUT_DIR / "dev_metadata.csv",
            "npy_dir":    DEV_OUTPUT_DIR,
            "protocol":   DEV_PROTOCOL,
            "flac_dir":   DEV_FLAC_DIR,
            "shuffle":    False,
        },
        "eval": {
            "meta_csv":   METADATA_OUTPUT_DIR / "eval_metadata.csv",
            "npy_dir":    EVAL_OUTPUT_DIR,
            "protocol":   EVAL_PROTOCOL,
            "flac_dir":   EVAL_FLAC_DIR,
            "shuffle":    False,
        },
    }
    
    for split_name, cfg in splits.items():
        if use_preprocessed:
            dataset = ASVspoofDataset(
                metadata_csv=cfg["meta_csv"],
                npy_dir=cfg["npy_dir"],
                subset_size=subset_size,
            )
        else:
            dataset = ASVspoofRawDataset(
                protocol_file=cfg["protocol"],
                flac_dir=cfg["flac_dir"],
                subset_size=subset_size,
            )
        
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=cfg["shuffle"],
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == "train"),
        )
    
    return loaders


if __name__ == "__main__":
    """Quick test: load a small subset and verify shapes."""
    
    print("Testing ASVspoofRawDataset (no preprocessing needed)...")
    print("=" * 50)
    
    # Try raw dataset on a tiny subset
    from config import TRAIN_PROTOCOL, TRAIN_FLAC_DIR
    
    dataset = ASVspoofRawDataset(
        protocol_file=TRAIN_PROTOCOL,
        flac_dir=TRAIN_FLAC_DIR,
        subset_size=20,
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_idx, (x, y, utt_ids) in enumerate(loader):
        print(f"\n  Batch {batch_idx}:")
        print(f"    Audio shape: {x.shape}")  # Expected: (batch, 64600)
        print(f"    Labels:      {y.tolist()}")
        print(f"    Utt IDs:     {list(utt_ids)}")
        print(f"    Audio range: [{x.min():.4f}, {x.max():.4f}]")
        
        if batch_idx >= 2:
            break
    
    print("\nDataset test passed")

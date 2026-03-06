"""
In-Memory Dataset — Preloads all .npy files into RAM for fast training.
Drop this file into your project folder alongside the other files.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import LABEL_MAP


class ASVspoofMemoryDataset(Dataset):
    """
    Loads ALL preprocessed .npy files into RAM at init.
    Eliminates per-batch disk I/O entirely.
    """

    def __init__(self, metadata_csv, npy_dir, subset_size=None):
        self.npy_dir = Path(npy_dir)
        df = pd.read_csv(metadata_csv)

        # Filter to existing files
        df["npy_path"] = df["utt_id"].apply(lambda uid: self.npy_dir / f"{uid}.npy")
        df = df[df["npy_path"].apply(lambda p: p.exists())].reset_index(drop=True)

        if subset_size is not None and subset_size < len(df):
            df = df.groupby("label_str", group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), subset_size // 2), random_state=42)
            ).reset_index(drop=True)

        # Preload everything into RAM
        print(f"  Loading {len(df):,} files into RAM from {npy_dir}...")
        audios = []
        labels = []
        utt_ids = []
        for row in tqdm(df.itertuples(), total=len(df), desc="  Caching", unit="utt"):
            audios.append(np.load(row.npy_path))
            labels.append(row.label)
            utt_ids.append(row.utt_id)

        self.audio = torch.FloatTensor(np.stack(audios))
        self.labels = torch.LongTensor(labels)
        self.utt_ids = utt_ids

        size_gb = self.audio.nbytes / 1e9
        print(f"  Cached: {len(self.audio):,} utterances ({size_gb:.1f} GB RAM)")

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return self.audio[idx], self.labels[idx], self.utt_ids[idx]

import pickle
import numpy as np
from pathlib import Path
import sys

# Add project root to path so config imports work
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TRAIN_PROTOCOL, DEV_PROTOCOL, EVAL_PROTOCOL,
    LABEL_MAP,
)

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent

EMBEDDING_PATHS = {
    "asv_trn":  BASE / "embeddings" / "asv_embd_trn.pk",
    "asv_dev":  BASE / "embeddings" / "asv_embd_dev.pk",
    "asv_eval": BASE / "embeddings" / "asv_embd_eval.pk",
    "cm_trn":   BASE / "embeddings" / "cm_embd_trn.pk",
    "cm_dev":   BASE / "embeddings" / "cm_embd_dev.pk",
    "cm_eval":  BASE / "embeddings" / "cm_embd_eval.pk",
}

# Use the CM protocol files from config.py for all splits.
# The original code referenced SASV-specific ASV trial lists
# (ASVspoof2019.LA.asv.dev.gi.trl.txt) which are from SASV 2022,
# not the standard ASVspoof 2019 download. CM protocols work for
# all three splits and give bonafide/spoof labels directly.
PROTOCOL_PATHS = {
    "trn":  TRAIN_PROTOCOL,
    "dev":  DEV_PROTOCOL,
    "eval": EVAL_PROTOCOL,
}


# ── helpers ──────────────────────────────────────────────────────────────────
def load_pickle(path):
    """Load a .pk file and return the dictionary inside."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_labels(split):
    """
    Read the CM protocol file for a given split (trn / dev / eval).

    CM protocol format: speaker | trial | - | attack_type | bonafide/spoof

    Returns:
        dict { trial_id -> label }
        1 = bonafide (accept)
        0 = spoof (reject)
    """
    labels = {}
    with open(PROTOCOL_PATHS[split]) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            trial_id = parts[1]
            label_str = parts[4]
            labels[trial_id] = 1 if label_str == "bonafide" else 0

    return labels


# ── main function ─────────────────────────────────────────────────────────────
def load_data(split):
    """
    Load and combine ASV + CM embeddings with labels for a given split.

    The combined dimension is detected automatically from the embedding
    files (ASV is typically 192-dim from ECAPA-TDNN; CM dimension depends
    on the AASIST variant: 16 for AASIST-L, 64 for full AASIST).

    Args:
        split : one of 'trn', 'dev', 'eval'

    Returns:
        X          : numpy array of shape (N, asv_dim + cm_dim)
        y          : numpy array of shape (N,)  -- labels (1=accept, 0=reject)
        trial_ids  : list of trial IDs in the same order as X and y
    """
    print(f"[utils] Loading '{split}' split...")

    # 1. Load embeddings
    asv_embeddings = load_pickle(EMBEDDING_PATHS[f"asv_{split}"])
    cm_embeddings  = load_pickle(EMBEDDING_PATHS[f"cm_{split}"])
    print(f"  ASV embeddings loaded: {len(asv_embeddings)} entries")
    print(f"  CM  embeddings loaded: {len(cm_embeddings)} entries")

    # Detect dimensions from actual data
    sample_asv = next(iter(asv_embeddings.values()))
    sample_cm  = next(iter(cm_embeddings.values()))
    asv_dim = sample_asv.shape[0]
    cm_dim  = sample_cm.shape[0]
    combined_dim = asv_dim + cm_dim
    print(f"  ASV dim: {asv_dim}, CM dim: {cm_dim}, Combined: {combined_dim}")

    # 2. Load labels
    labels = load_labels(split)
    print(f"  Labels loaded: {len(labels)} entries")

    # 3. Find trial IDs present in ALL three sources
    valid_ids = sorted(
        set(asv_embeddings.keys()) & set(cm_embeddings.keys()) & set(labels.keys())
    )
    print(f"  Valid (matched) trial IDs: {len(valid_ids)}")

    # 4. Build X and y
    X, y, trial_ids = [], [], []
    for trial_id in valid_ids:
        asv_vec  = asv_embeddings[trial_id]
        cm_vec   = cm_embeddings[trial_id]
        combined = np.concatenate([asv_vec, cm_vec])
        X.append(combined)
        y.append(labels[trial_id])
        trial_ids.append(trial_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"  X shape : {X.shape}")
    print(f"  Bonafide (accept): {y.sum()}  |  Reject: {(y == 0).sum()}")
    return X, y, trial_ids

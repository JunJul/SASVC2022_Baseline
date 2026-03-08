import pickle
import numpy as np
from pathlib import Path

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

PROTOCOL_PATHS = {
    "trn":  BASE / "protocols" / "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":  BASE / "protocols" / "ASVspoof2019.LA.asv.dev.gi.trl.txt",
    "eval": BASE / "protocols" / "ASVspoof2019.LA.asv.eval.gi.trl.txt",
}


# ── helpers ──────────────────────────────────────────────────────────────────
def load_pickle(path):
    """Load a .pk file and return the dictionary inside."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_labels(split):
    """
    Read the protocol file for a given split (trn / dev / eval).

    CM protocol (trn)  : speaker | trial | - | - | bonafide/spoof
    ASV protocol (dev/eval): speaker | trial | bonafide/A0x | target/nontarget/spoof

    Returns:
        dict { trial_id -> label }
        1 = bonafide target  (accept)
        0 = nontarget or spoof (reject)
    """
    labels = {}
    with open(PROTOCOL_PATHS[split]) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # skip malformed lines

            trial_id = parts[1]

            if split == "trn":
                # fifth column: bonafide or spoof
                label_str = parts[4]
                labels[trial_id] = 1 if label_str == "bonafide" else 0
            else:
                # fourth column: target / nontarget / spoof
                sasv_label = parts[3]
                labels[trial_id] = 1 if sasv_label == "target" else 0

    return labels


# ── main function ─────────────────────────────────────────────────────────────
def load_data(split):
    """
    Load and combine ASV + CM embeddings with labels for a given split.

    Args:
        split : one of 'trn', 'dev', 'eval'

    Returns:
        X          : numpy array of shape (N, 352)  — concatenated ASV+CM embeddings
        y          : numpy array of shape (N,)      — labels (1=accept, 0=reject)
        trial_ids  : list of trial IDs in the same order as X and y
    """
    print(f"[utils] Loading '{split}' split...")

    # 1. Load embeddings
    asv_embeddings = load_pickle(EMBEDDING_PATHS[f"asv_{split}"])
    cm_embeddings  = load_pickle(EMBEDDING_PATHS[f"cm_{split}"])
    print(f"  ASV embeddings loaded: {len(asv_embeddings)} entries")
    print(f"  CM  embeddings loaded: {len(cm_embeddings)} entries")

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
        asv_vec  = asv_embeddings[trial_id]          # shape (192,)
        cm_vec   = cm_embeddings[trial_id]           # shape (160,)
        combined = np.concatenate([asv_vec, cm_vec]) # shape (352,)
        X.append(combined)
        y.append(labels[trial_id])
        trial_ids.append(trial_id)

    X = np.array(X, dtype=np.float32)  # (N, 352)
    y = np.array(y, dtype=np.int32)    # (N,)

    print(f"  X shape : {X.shape}")
    print(f"  Bonafide (accept): {y.sum()}  |  Reject: {(y == 0).sum()}")
    return X, y, trial_ids

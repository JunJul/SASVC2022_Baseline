"""
Embedding Extraction for SASV Fusion
======================================
Extracts two types of embeddings for the fusion stage:

  1. CM embeddings (countermeasure) — from the AASIST penultimate layer.
     Shape: (N, 2*gat_dims[-1]) = (N, 16) for AASIST-L or (N, 64) for AASIST.

  2. ASV embeddings (speaker verification) — from a pretrained ECAPA-TDNN
     via SpeechBrain. Shape: (N, 192).

Outputs are saved as pickle files compatible with the fusion module:
    embeddings/cm_embd_trn.pk
    embeddings/cm_embd_dev.pk
    embeddings/cm_embd_eval.pk
    embeddings/asv_embd_trn.pk
    embeddings/asv_embd_dev.pk
    embeddings/asv_embd_eval.pk

Each pickle file maps utt_id -> np.ndarray (1D embedding vector).

Usage:
    python extract_embeddings.py --checkpoint checkpoints/best_model.pt
    python extract_embeddings.py --checkpoint checkpoints/best_model.pt --split eval
    python extract_embeddings.py --checkpoint checkpoints/best_model.pt --cm_only
    python extract_embeddings.py --checkpoint checkpoints/best_model.pt --asv_only
"""

import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from config import (
    TRAIN_PROTOCOL, TRAIN_FLAC_DIR,
    DEV_PROTOCOL, DEV_FLAC_DIR,
    EVAL_PROTOCOL, EVAL_FLAC_DIR,
    METADATA_OUTPUT_DIR, TRAIN_OUTPUT_DIR, DEV_OUTPUT_DIR, EVAL_OUTPUT_DIR,
    TARGET_SAMPLE_RATE, MAX_AUDIO_LENGTH,
)
from dataset import ASVspoofRawDataset, ASVspoofDataset
from aasist_model import build_model


# ─────────────────────────────────────────────
# CM Embedding Extraction (AASIST)
# ─────────────────────────────────────────────

class AASISTEmbeddingExtractor:
    """
    Wraps a trained AASIST model to extract penultimate-layer embeddings.

    The embedding is the concatenated graph attention output (before the
    final classification head). For AASIST-L this is 16-dim, for full
    AASIST it's 64-dim.
    """

    def __init__(self, checkpoint_path, device):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        args = ckpt["args"]
        variant = args.get("variant", "AASIST-L")

        self.model = build_model(variant).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Register a forward hook on the classifier to grab input embeddings.
        self._embedding = None

        def hook_fn(module, input, output):
            # input is a tuple; input[0] is the (B, D) tensor fed to the classifier
            self._embedding = input[0].detach().cpu()

        self.model.classifier.register_forward_hook(hook_fn)

        print(f"  Loaded {variant} from {checkpoint_path}")
        print(f"  Training epoch: {ckpt.get('epoch', -1) + 1}")

    @torch.no_grad()
    def extract(self, loader):
        """
        Run inference and capture embeddings.

        Returns
        -------
        dict mapping utt_id (str) -> np.ndarray of shape (embed_dim,)
        """
        embeddings = {}
        for x, y, utt_ids in tqdm(loader, desc="  CM embeddings", unit="batch"):
            x = x.to(self.device)
            _ = self.model(x)  # triggers hook
            emb = self._embedding.numpy()
            for i, uid in enumerate(utt_ids):
                embeddings[uid] = emb[i]
        return embeddings


# ─────────────────────────────────────────────
# ASV Embedding Extraction (ECAPA-TDNN)
# ─────────────────────────────────────────────

class ECAPAEmbeddingExtractor:
    """
    Uses SpeechBrain's pretrained ECAPA-TDNN to extract 192-dim speaker
    embeddings from raw waveforms.

    Requires: pip install speechbrain
    The model downloads automatically on first use (~80 MB).
    """

    def __init__(self, device):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError(
                "SpeechBrain is required for ASV embeddings.\n"
                "Install it: pip install speechbrain"
            )

        self.device = device
        print("  Loading ECAPA-TDNN from SpeechBrain (downloads on first run)...")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa_tdnn",
            run_opts={"device": str(device)},
        )
        print("  ECAPA-TDNN loaded. Output dim: 192")

    @torch.no_grad()
    def extract(self, loader):
        """
        Extract 192-dim speaker embeddings.

        The ECAPA-TDNN expects waveforms at 16 kHz, which matches our
        preprocessing. We pass each batch through the encoder and collect
        the embeddings.

        Returns
        -------
        dict mapping utt_id (str) -> np.ndarray of shape (192,)
        """
        embeddings = {}
        for x, y, utt_ids in tqdm(loader, desc="  ASV embeddings", unit="batch"):
            # SpeechBrain expects (batch, time) at 16 kHz
            emb = self.model.encode_batch(x.to(self.device))
            # emb shape: (batch, 1, 192) -> squeeze
            emb = emb.squeeze(1).cpu().numpy()
            for i, uid in enumerate(utt_ids):
                embeddings[uid] = emb[i]
        return embeddings


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_loader(split, use_preprocessed=False, batch_size=32, num_workers=4):
    """Create a DataLoader for the given split."""
    configs = {
        "train": (TRAIN_PROTOCOL, TRAIN_FLAC_DIR, TRAIN_OUTPUT_DIR, "train"),
        "dev":   (DEV_PROTOCOL,   DEV_FLAC_DIR,   DEV_OUTPUT_DIR,   "dev"),
        "eval":  (EVAL_PROTOCOL,  EVAL_FLAC_DIR,  EVAL_OUTPUT_DIR,  "eval"),
    }
    protocol, flac_dir, npy_dir, name = configs[split]
    meta_csv = METADATA_OUTPUT_DIR / f"{name}_metadata.csv"

    if use_preprocessed and npy_dir.exists() and any(npy_dir.glob("*.npy")):
        dataset = ASVspoofDataset(metadata_csv=meta_csv, npy_dir=npy_dir)
    else:
        dataset = ASVspoofRawDataset(protocol_file=protocol, flac_dir=flac_dir)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader


def save_embeddings(embeddings, path):
    """Save embedding dict as pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"  Saved {len(embeddings):,} embeddings -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings for SASV fusion")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained AASIST checkpoint")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "dev", "eval", "all"])
    parser.add_argument("--output_dir", type=str, default="embeddings")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_preprocessed", action="store_true")
    parser.add_argument("--cm_only", action="store_true",
                        help="Only extract CM (AASIST) embeddings")
    parser.add_argument("--asv_only", action="store_true",
                        help="Only extract ASV (ECAPA-TDNN) embeddings")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [args.split] if args.split != "all" else ["train", "dev", "eval"]
    split_name_map = {"train": "trn", "dev": "dev", "eval": "eval"}

    extract_cm = not args.asv_only
    extract_asv = not args.cm_only

    print("=" * 60)
    print("  Embedding Extraction for SASV Fusion")
    print("=" * 60)
    print(f"  Device:    {device}")
    print(f"  Splits:    {splits}")
    print(f"  Extract:   {'CM' if extract_cm else ''} {'ASV' if extract_asv else ''}")
    print()

    # Initialize extractors
    cm_extractor = None
    asv_extractor = None

    if extract_cm:
        cm_extractor = AASISTEmbeddingExtractor(args.checkpoint, device)

    if extract_asv:
        asv_extractor = ECAPAEmbeddingExtractor(device)

    # Extract per split
    for split in splits:
        print(f"\n{'─'*50}")
        print(f"  Processing {split.upper()} split")
        print(f"{'─'*50}")

        loader = make_loader(
            split, use_preprocessed=args.use_preprocessed,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        suffix = split_name_map[split]

        if cm_extractor:
            cm_emb = cm_extractor.extract(loader)
            save_embeddings(cm_emb, output_dir / f"cm_embd_{suffix}.pk")

            # Print a quick sanity check
            sample_key = next(iter(cm_emb))
            print(f"    CM embedding dim: {cm_emb[sample_key].shape[0]}")

        if asv_extractor:
            asv_emb = asv_extractor.extract(loader)
            save_embeddings(asv_emb, output_dir / f"asv_embd_{suffix}.pk")

            sample_key = next(iter(asv_emb))
            print(f"    ASV embedding dim: {asv_emb[sample_key].shape[0]}")

    print("\n" + "=" * 60)
    print("  Embedding extraction complete.")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

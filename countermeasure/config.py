"""
Configuration for ASVspoof 2019 LA - Deepfake Audio Detection Pipeline
======================================================================
Phase 1: Data preprocessing and AASIST integration
"""

import os
from pathlib import Path

# Dataset paths (UPDATE THESE TO MATCH YOUR SYSTEM)
BASE_DIR = Path(r"C:\Users\kingr\Downloads\biometrics_project_prelim\LA")

# Raw audio directories
TRAIN_FLAC_DIR = BASE_DIR / "ASVspoof2019_LA_train"
DEV_FLAC_DIR   = BASE_DIR / "ASVspoof2019_LA_dev"
EVAL_FLAC_DIR  = BASE_DIR / "ASVspoof2019_LA_eval"

# Protocol files
PROTOCOL_DIR = BASE_DIR / "ASVspoof2019_LA_cm_protocols"
TRAIN_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL   = PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
EVAL_PROTOCOL  = PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"

# Preprocessed output directories
OUTPUT_DIR          = BASE_DIR / "preprocessed"
TRAIN_OUTPUT_DIR    = OUTPUT_DIR / "train"
DEV_OUTPUT_DIR      = OUTPUT_DIR / "dev"
EVAL_OUTPUT_DIR     = OUTPUT_DIR / "eval"
METADATA_OUTPUT_DIR = OUTPUT_DIR / "metadata"

# Audio preprocessing parameters
TARGET_SAMPLE_RATE = 16000   # Resample all audio to 16 kHz
MAX_AUDIO_LENGTH   = 64600   # ~4.04 seconds at 16 kHz (AASIST default)
PAD_MODE           = "wrap"  


# Subset sizes for quick experiments
SMALL_SUBSET_SIZE  = 500     
MEDIUM_SUBSET_SIZE = 5000    

# Label encoding
LABEL_MAP = {
    "bonafide": 1,
    "spoof":    0,
}

# Attack type mapping
ATTACK_TYPES = {
    "-":    "bonafide",
    "A01":  "TTS (neural waveform)",
    "A02":  "TTS (vocoder)",
    "A03":  "TTS (vocoder)",
    "A04":  "TTS (waveform concat)",
    "A05":  "VC (voice conversion)",
    "A06":  "VC (voice conversion)",
    "A07":  "TTS (vocoder-based)",
    "A08":  "TTS (neural)",
    "A09":  "TTS (vocoder)",
    "A10":  "TTS (neural)",
    "A11":  "TTS (griffin-lim)",
    "A12":  "TTS (neural)",
    "A13":  "TTS/VC (hybrid)",
    "A14":  "TTS (neural)",
    "A15":  "TTS (neural)",
    "A16":  "TTS (waveform)",
    "A17":  "VC (voice conversion)",
    "A18":  "TTS (vocoder)",
    "A19":  "TTS (neural)",
}

# AASIST model configuration 
AASIST_CONFIG = {
    "architecture": "AASIST",
    "nb_samp":      MAX_AUDIO_LENGTH,
    "first_conv":   128,
    "filts":        [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims":     [64, 32],
    "pool_ratios":  [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

# For resource-constrained training (AASIST-L)
AASIST_L_CONFIG = {
    "architecture": "AASIST-L",
    "nb_samp":      MAX_AUDIO_LENGTH,
    "first_conv":   16,
    "filts":        [10, [1, 4], [4, 8], [8, 16], [16, 16]],
    "gat_dims":     [16, 8],
    "pool_ratios":  [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def create_output_dirs():
    """Create all necessary output directories."""
    for d in [TRAIN_OUTPUT_DIR, DEV_OUTPUT_DIR, EVAL_OUTPUT_DIR, METADATA_OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Output directories created under: {OUTPUT_DIR}")


def validate_paths():
    """Check that all expected input paths exist."""
    issues = []
    for name, path in [
        ("Train audio",    TRAIN_FLAC_DIR),
        ("Dev audio",      DEV_FLAC_DIR),
        ("Eval audio",     EVAL_FLAC_DIR),
        ("Protocol dir",   PROTOCOL_DIR),
        ("Train protocol", TRAIN_PROTOCOL),
        ("Dev protocol",   DEV_PROTOCOL),
        ("Eval protocol",  EVAL_PROTOCOL),
    ]:
        if not path.exists():
            issues.append(f"  ✗ {name}: {path}")

    if issues:
        print("Missing paths detected:")
        print("\n".join(issues))
        print("\nPlease verify BASE_DIR in config.py and your folder structure.")
        print("Expected structure:")
        print("  LA/")
        print("  ├── ASVspoof2019_LA_train/flac/  (LA_T_*.flac)")
        print("  ├── ASVspoof2019_LA_dev/flac/    (LA_D_*.flac)")
        print("  ├── ASVspoof2019_LA_eval/flac/   (LA_E_*.flac)")
        print("  └── ASVspoof2019_LA_cm_protocols/")
        print("      ├── ASVspoof2019.LA.cm.train.trn.txt")
        print("      ├── ASVspoof2019.LA.cm.dev.trl.txt")
        print("      └── ASVspoof2019.LA.cm.eval.trl.txt")
        return False
    else:
        print("All input paths verified.")
        return True


if __name__ == "__main__":
    validate_paths()

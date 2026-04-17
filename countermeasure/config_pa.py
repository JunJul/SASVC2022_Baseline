"""
Configuration for ASVspoof 2019 PA - Replay Attack Detection
=============================================================
Physical Access (PA) partition: detects audio replayed through
loudspeakers and re-recorded by a microphone.

Attack configurations encode three variables:
    Letter 1 (room):     A=small, B=medium, C=large
    Letter 2 (attacker): A=close+high-quality, B=mid, C=far+low-quality

If your PA directory structure differs, update the paths below.
Run 'python config_pa.py' to verify all paths exist.
"""

from pathlib import Path

# ── Dataset paths ─────────────────────────────────────────────────────────────
# UPDATE THIS to match your system
BASE_DIR = Path(r"C:\Users\kingr\Downloads\biometrics_project_prelim\PA")

# Raw audio directories
TRAIN_FLAC_DIR = BASE_DIR / "ASVspoof2019_PA_train"
DEV_FLAC_DIR   = BASE_DIR / "ASVspoof2019_PA_dev"
EVAL_FLAC_DIR  = BASE_DIR / "ASVspoof2019_PA_eval"

# Protocol files
PROTOCOL_DIR   = BASE_DIR / "ASVspoof2019_PA_cm_protocols"
TRAIN_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.PA.cm.train.trn.txt"
DEV_PROTOCOL   = PROTOCOL_DIR / "ASVspoof2019.PA.cm.dev.trl.txt"
EVAL_PROTOCOL  = PROTOCOL_DIR / "ASVspoof2019.PA.cm.eval.trl.txt"

# Preprocessed output directories
OUTPUT_DIR          = BASE_DIR / "preprocessed"
TRAIN_OUTPUT_DIR    = OUTPUT_DIR / "train"
DEV_OUTPUT_DIR      = OUTPUT_DIR / "dev"
EVAL_OUTPUT_DIR     = OUTPUT_DIR / "eval"
METADATA_OUTPUT_DIR = OUTPUT_DIR / "metadata"


# ── Audio parameters (same as LA -- AASIST expects identical input) ──────────
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH   = 64600   # ~4.04 s at 16 kHz
PAD_MODE           = "wrap"

SMALL_SUBSET_SIZE  = 500
MEDIUM_SUBSET_SIZE = 5000

LABEL_MAP = {
    "bonafide": 1,
    "spoof":    0,
}


# ── PA attack types ──────────────────────────────────────────────────────────
# The two-letter codes describe the replay configuration.
# First letter  = acoustic environment (room size / reverberation)
# Second letter = attacker-to-speaker distance and device quality
#
# Train/dev use 9 known configurations (AA-CC).
# Eval adds unseen combinations to test generalization.
PA_ATTACK_TYPES = {
    "-":   "bonafide",
    "AA":  "replay (small room, close+HQ)",
    "AB":  "replay (small room, mid)",
    "AC":  "replay (small room, far+LQ)",
    "BA":  "replay (medium room, close+HQ)",
    "BB":  "replay (medium room, mid)",
    "BC":  "replay (medium room, far+LQ)",
    "CA":  "replay (large room, close+HQ)",
    "CB":  "replay (large room, mid)",
    "CC":  "replay (large room, far+LQ)",
}

# Reuse AASIST model configs from the LA config -- same architecture
# handles both LA and PA tasks, just trained on different data.
from config import AASIST_CONFIG, AASIST_L_CONFIG


# ── Helpers ──────────────────────────────────────────────────────────────────

def create_output_dirs():
    """Create all necessary output directories."""
    for d in [TRAIN_OUTPUT_DIR, DEV_OUTPUT_DIR, EVAL_OUTPUT_DIR, METADATA_OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Output directories created under: {OUTPUT_DIR}")


def validate_paths():
    """Check that all expected input paths exist."""
    issues = []
    for name, path in [
        ("PA Train audio",    TRAIN_FLAC_DIR),
        ("PA Dev audio",      DEV_FLAC_DIR),
        ("PA Eval audio",     EVAL_FLAC_DIR),
        ("PA Protocol dir",   PROTOCOL_DIR),
        ("PA Train protocol", TRAIN_PROTOCOL),
        ("PA Dev protocol",   DEV_PROTOCOL),
        ("PA Eval protocol",  EVAL_PROTOCOL),
    ]:
        if not path.exists():
            issues.append(f"  MISSING: {name}: {path}")

    if issues:
        print("PA path problems:")
        print("\n".join(issues))
        print()
        print("Expected structure:")
        print("  PA/")
        print("  +-- ASVspoof2019_PA_train/   (PA_T_*.flac)")
        print("  +-- ASVspoof2019_PA_dev/     (PA_D_*.flac)")
        print("  +-- ASVspoof2019_PA_eval/    (PA_E_*.flac)")
        print("  +-- ASVspoof2019_PA_cm_protocols/")
        print("      +-- ASVspoof2019.PA.cm.train.trn.txt")
        print("      +-- ASVspoof2019.PA.cm.dev.trl.txt")
        print("      +-- ASVspoof2019.PA.cm.eval.trl.txt")
        print()
        print("If your .flac files are in a 'flac/' subdirectory,")
        print("update the TRAIN/DEV/EVAL_FLAC_DIR paths above.")
        return False
    else:
        print("All PA input paths verified.")
        return True


if __name__ == "__main__":
    validate_paths()

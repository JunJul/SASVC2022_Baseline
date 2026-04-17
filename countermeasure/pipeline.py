"""
End-to-End SASV Pipeline with Replay Detection
=================================================
Takes a test audio file and an enrolled speaker's reference audio,
runs four checks, and outputs one of four decisions:

    ACCEPT         -- genuine speaker, real audio, not replayed
    REJECT-SPOOF   -- deepfake / synthetic audio (TTS or voice conversion)
    REJECT-REPLAY  -- physical replay attack detected
    REJECT-SPEAKER -- wrong speaker

The pipeline chains up to four models:
    1. ECAPA-TDNN    (speaker embedding, 192-dim)
    2. AASIST-LA     (countermeasure for TTS/VC attacks)
    3. AASIST-PA     (countermeasure for replay attacks) [optional]
    4. Fusion model   (logistic / MLP / CatBoost on LA embeddings)

If --pa_checkpoint is not provided, the pipeline runs without replay
detection (same three-way behavior as before).

Usage:
    # Full four-way pipeline (LA + PA)
    python pipeline.py --checkpoint checkpoints/best_model.pt \\
                       --pa_checkpoint checkpoints_pa/best_model.pt \\
                       --test_audio test.flac \\
                       --ref_audio enrolled_speaker.flac

    # LA-only (three-way, no replay detection)
    python pipeline.py --checkpoint checkpoints/best_model.pt \\
                       --test_audio test.flac \\
                       --ref_audio enrolled_speaker.flac

    # Adjust all thresholds
    python pipeline.py --checkpoint checkpoints/best_model.pt \\
                       --pa_checkpoint checkpoints_pa/best_model.pt \\
                       --test_audio test.flac \\
                       --ref_audio enrolled_speaker.flac \\
                       --spoof_threshold 0.5 \\
                       --replay_threshold 0.5 \\
                       --speaker_threshold 0.25

    # Test a directory against one reference
    python pipeline.py --checkpoint checkpoints/best_model.pt \\
                       --pa_checkpoint checkpoints_pa/best_model.pt \\
                       --test_dir test_audio/ \\
                       --ref_audio enrolled_speaker.flac

    # Multiple enrollment utterances for stronger speaker template
    python pipeline.py --checkpoint checkpoints/best_model.pt \\
                       --pa_checkpoint checkpoints_pa/best_model.pt \\
                       --test_audio test.flac \\
                       --ref_dir enrollment_audio/
"""

import argparse
import sys
import pickle
import numpy as np
import torch
import soundfile as sf
from math import gcd
from pathlib import Path

from config import TARGET_SAMPLE_RATE, MAX_AUDIO_LENGTH
from aasist_model import build_model


# ── Audio Loading ─────────────────────────────────────────────────────────────

def load_audio(audio_path, target_sr=TARGET_SAMPLE_RATE,
               max_len=MAX_AUDIO_LENGTH):
    """
    Load an audio file, resample to target_sr, pad/truncate to max_len.
    Uses soundfile to avoid librosa/lazy_loader issues on Python 3.14.
    """
    audio, sr = sf.read(str(audio_path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)

    original_duration = len(audio) / target_sr

    if len(audio) >= max_len:
        audio = audio[:max_len]
    else:
        repeats = (max_len // len(audio)) + 1
        audio = np.tile(audio, repeats)[:max_len]

    return audio, original_duration


# ── CM Embedding Extractor (AASIST) ─────────────────────────────────────────

class CMExtractor:
    """
    Loads a trained AASIST checkpoint and extracts penultimate-layer
    embeddings via a forward hook on the classifier head.
    Works for both LA (spoof) and PA (replay) checkpoints.
    """

    def __init__(self, checkpoint_path, device, label="CM"):
        self.device = device
        self.label = label
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        args = ckpt["args"]
        self.variant = args.get("variant", "AASIST-L")
        self.task = ckpt.get("task", "LA")

        self.model = build_model(self.variant).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self._embedding = None

        def hook_fn(module, inp, out):
            self._embedding = inp[0].detach().cpu()

        self.model.classifier.register_forward_hook(hook_fn)

    @torch.no_grad()
    def extract(self, waveform):
        """Extract penultimate-layer embedding from a waveform."""
        x = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
        _ = self.model(x)
        return self._embedding[0].numpy()

    @torch.no_grad()
    def score(self, waveform):
        """Return P(bonafide) from the classifier head."""
        x = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        return probs[0, 1].item()


# ── ASV Embedding Extractor (ECAPA-TDNN) ────────────────────────────────────

class ASVExtractor:
    """192-dim speaker embeddings from SpeechBrain's pretrained ECAPA-TDNN."""

    def __init__(self, device):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError(
                "SpeechBrain is required for ASV embeddings.\n"
                "Install: pip install speechbrain"
            )
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa_tdnn",
            run_opts={"device": str(device)},
        )

    @torch.no_grad()
    def extract(self, waveform):
        x = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
        emb = self.model.encode_batch(x)
        return emb.squeeze(0).squeeze(0).cpu().numpy()

    def extract_from_file(self, audio_path):
        waveform, _ = load_audio(audio_path)
        return self.extract(waveform)


# ── Speaker Comparison ────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def enroll_speaker(asv_extractor, ref_paths):
    """Average ASV embeddings from reference utterances, L2-normalize."""
    embeddings = []
    for p in ref_paths:
        emb = asv_extractor.extract_from_file(p)
        embeddings.append(emb)
    avg = np.mean(embeddings, axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-8)
    return avg


# ── Fusion Model ─────────────────────────────────────────────────────────────

def load_fusion_model(model_name):
    model_path = Path("fusion") / "saved_models" / f"{model_name}.pk"
    if not model_path.exists():
        print(f"Fusion model not found at {model_path}")
        print("Run 'python -m fusion.train' first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ── Decision Logic ────────────────────────────────────────────────────────────

def make_decision(fusion_score, speaker_sim, replay_score=None,
                  spoof_threshold=0.5, replay_threshold=0.5,
                  speaker_threshold=0.25):
    """
    Four-way decision (or three-way if replay_score is None).

    Priority order:
        1. Spoof check   -- synthetic audio is always rejected first
        2. Replay check   -- replayed audio rejected second
        3. Speaker check  -- wrong speaker rejected third
        4. Accept         -- passes all checks
    """
    details = {
        "fusion_score": round(fusion_score, 4),
        "spoof_threshold": spoof_threshold,
        "speaker_similarity": round(speaker_sim, 4),
        "speaker_threshold": speaker_threshold,
    }

    if replay_score is not None:
        details["replay_score"] = round(replay_score, 4)
        details["replay_threshold"] = replay_threshold

    if fusion_score < spoof_threshold:
        return "REJECT-SPOOF", details

    if replay_score is not None and replay_score < replay_threshold:
        return "REJECT-REPLAY", details

    if speaker_sim < speaker_threshold:
        return "REJECT-SPEAKER", details

    return "ACCEPT", details


# ── Single-File Pipeline ─────────────────────────────────────────────────────

def run_pipeline(test_path, enrollment_emb,
                 la_extractor, asv_extractor, fusion_model,
                 pa_extractor=None,
                 spoof_threshold=0.5, replay_threshold=0.5,
                 speaker_threshold=0.25):
    """Run the full SASV pipeline on one test audio file."""

    waveform, duration = load_audio(test_path)

    # Speaker verification
    test_asv = asv_extractor.extract(waveform)
    test_asv_norm = test_asv / (np.linalg.norm(test_asv) + 1e-8)
    speaker_sim = cosine_similarity(test_asv_norm, enrollment_emb)

    # LA countermeasure (TTS/VC)
    test_cm = la_extractor.extract(waveform)
    combined = np.concatenate([test_asv, test_cm]).reshape(1, -1)
    fusion_score = fusion_model.predict_scores(combined)[0]
    la_score = la_extractor.score(waveform)

    # PA countermeasure (replay) -- optional
    replay_score = None
    pa_score_raw = None
    if pa_extractor is not None:
        replay_score = pa_extractor.score(waveform)
        pa_score_raw = replay_score

    # Decision
    decision, details = make_decision(
        fusion_score, speaker_sim, replay_score,
        spoof_threshold=spoof_threshold,
        replay_threshold=replay_threshold,
        speaker_threshold=speaker_threshold,
    )

    result = {
        "file": Path(test_path).name,
        "decision": decision,
        "fusion_score": details["fusion_score"],
        "la_score": round(la_score, 4),
        "speaker_similarity": details["speaker_similarity"],
        "duration_sec": round(duration, 2),
        "spoof_threshold": spoof_threshold,
        "speaker_threshold": speaker_threshold,
    }

    if pa_score_raw is not None:
        result["replay_score"] = round(pa_score_raw, 4)
        result["replay_threshold"] = replay_threshold

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SASV pipeline: audio in, four-way decision out",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Models
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained LA AASIST checkpoint (.pt)")
    parser.add_argument("--pa_checkpoint", type=str, default=None,
                        help="Path to trained PA AASIST checkpoint (.pt). "
                             "Omit to skip replay detection.")
    parser.add_argument("--fusion_model", type=str, default="mlp",
                        choices=["logistic", "mlp", "catboost"])

    # Test audio
    parser.add_argument("--test_audio", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)

    # Reference / enrollment audio
    parser.add_argument("--ref_audio", type=str, default=None)
    parser.add_argument("--ref_dir", type=str, default=None)

    # Thresholds
    parser.add_argument("--spoof_threshold", type=float, default=0.5,
                        help="Fusion score below this = TTS/VC spoof (default: 0.5)")
    parser.add_argument("--replay_threshold", type=float, default=0.5,
                        help="PA score below this = replay attack (default: 0.5)")
    parser.add_argument("--speaker_threshold", type=float, default=0.25,
                        help="Cosine sim below this = wrong speaker (default: 0.25)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to this JSON file")

    args = parser.parse_args()

    if args.test_audio is None and args.test_dir is None:
        print("Provide --test_audio or --test_dir.")
        sys.exit(1)
    if args.ref_audio is None and args.ref_dir is None:
        print("Provide --ref_audio or --ref_dir for speaker enrollment.")
        sys.exit(1)

    # ── Collect files ──
    audio_exts = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}

    test_files = []
    if args.test_audio:
        test_files.append(Path(args.test_audio))
    if args.test_dir:
        for f in sorted(Path(args.test_dir).iterdir()):
            if f.suffix.lower() in audio_exts:
                test_files.append(f)

    ref_files = []
    if args.ref_audio:
        ref_files.append(Path(args.ref_audio))
    if args.ref_dir:
        for f in sorted(Path(args.ref_dir).iterdir()):
            if f.suffix.lower() in audio_exts:
                ref_files.append(f)

    if not test_files:
        print("No test audio files found.")
        sys.exit(1)
    if not ref_files:
        print("No reference audio files found.")
        sys.exit(1)

    # ── Initialize models ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pa_enabled = args.pa_checkpoint is not None

    print("=" * 65)
    print("  SASV End-to-End Pipeline")
    if pa_enabled:
        print("  Mode: Full (LA + PA + Speaker Verification)")
    else:
        print("  Mode: LA only (no replay detection)")
    print("=" * 65)
    print(f"  Device:            {device}")
    print(f"  Fusion model:      {args.fusion_model}")
    print(f"  Spoof threshold:   {args.spoof_threshold}")
    if pa_enabled:
        print(f"  Replay threshold:  {args.replay_threshold}")
    print(f"  Speaker threshold: {args.speaker_threshold}")
    print(f"  Test files:        {len(test_files)}")
    print(f"  Reference files:   {len(ref_files)}")
    print()

    print("  Loading LA model (TTS/VC detection)...")
    la_extractor = CMExtractor(args.checkpoint, device, label="LA")

    pa_extractor = None
    if pa_enabled:
        print("  Loading PA model (replay detection)...")
        pa_extractor = CMExtractor(args.pa_checkpoint, device, label="PA")

    print("  Loading ECAPA-TDNN (speaker verification)...")
    asv_extractor = ASVExtractor(device)

    print(f"  Loading fusion model ({args.fusion_model})...")
    fusion_model = load_fusion_model(args.fusion_model)

    # ── Enroll speaker ──
    print(f"\n  Enrolling speaker from {len(ref_files)} reference file(s)...")
    enrollment_emb = enroll_speaker(asv_extractor, ref_files)
    print(f"  Enrollment embedding shape: {enrollment_emb.shape}")

    # ── Run pipeline ──
    print(f"\n{'-'*65}")

    results = []
    for fpath in test_files:
        try:
            result = run_pipeline(
                fpath, enrollment_emb,
                la_extractor, asv_extractor, fusion_model,
                pa_extractor=pa_extractor,
                spoof_threshold=args.spoof_threshold,
                replay_threshold=args.replay_threshold,
                speaker_threshold=args.speaker_threshold,
            )
            results.append(result)

            tag = {
                "ACCEPT":         "  OK   ",
                "REJECT-SPOOF":   " SPOOF ",
                "REJECT-REPLAY":  " REPLAY",
                "REJECT-SPEAKER": "SPEAKER",
            }[result["decision"]]

            if pa_enabled:
                print(f"  [{tag}]  {result['file']:<30s}  "
                      f"fusion={result['fusion_score']:.4f}  "
                      f"replay={result.get('replay_score', 0):.4f}  "
                      f"spkr={result['speaker_similarity']:.4f}  "
                      f"la={result['la_score']:.4f}")
            else:
                print(f"  [{tag}]  {result['file']:<30s}  "
                      f"fusion={result['fusion_score']:.4f}  "
                      f"spkr={result['speaker_similarity']:.4f}  "
                      f"la={result['la_score']:.4f}")

        except Exception as e:
            print(f"  [ ERROR ]  {fpath.name}: {e}")

    # ── Summary ──
    print(f"\n{'-'*65}")
    n_accept  = sum(1 for r in results if r["decision"] == "ACCEPT")
    n_spoof   = sum(1 for r in results if r["decision"] == "REJECT-SPOOF")
    n_replay  = sum(1 for r in results if r["decision"] == "REJECT-REPLAY")
    n_speaker = sum(1 for r in results if r["decision"] == "REJECT-SPEAKER")

    print(f"  Accepted:           {n_accept}")
    print(f"  Rejected (spoof):   {n_spoof}")
    if pa_enabled:
        print(f"  Rejected (replay):  {n_replay}")
    print(f"  Rejected (speaker): {n_speaker}")
    print(f"  Total:              {len(results)}")

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved results to {args.output}")

    print()


if __name__ == "__main__":
    main()

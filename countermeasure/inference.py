"""
End-to-End Inference Pipeline
==============================
Takes a raw audio file and outputs a bonafide/spoof decision with confidence.

Supports single files, directories, and batch processing.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt --audio sample.flac
    python inference.py --checkpoint checkpoints/best_model.pt --audio_dir test_audio/
    python inference.py --checkpoint checkpoints/best_model.pt --audio sample.wav --threshold 0.73
"""

import argparse
import sys
import numpy as np
import torch
import librosa
from pathlib import Path

from config import TARGET_SAMPLE_RATE, MAX_AUDIO_LENGTH
from aasist_model import build_model


class DeepfakeDetector:
    """
    Wraps a trained AASIST checkpoint into a callable detector.

    Usage:
        detector = DeepfakeDetector("checkpoints/best_model.pt")
        result = detector.predict("audio.flac")
        print(result)
        # {'decision': 'bonafide', 'score': 0.987, 'confidence': 'high'}
    """

    def __init__(self, checkpoint_path, device=None, threshold=None):
        """
        Parameters
        ----------
        checkpoint_path : str
            Path to a trained AASIST .pt checkpoint.
        device : str or None
            Force device. If None, auto-detects CUDA.
        threshold : float or None
            Decision threshold on P(bonafide). If None, uses the EER
            threshold from training (stored in the checkpoint).
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        args = ckpt["args"]
        variant = args.get("variant", "AASIST-L")

        self.model = build_model(variant).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Threshold: prefer explicit arg, fall back to checkpoint's dev EER threshold
        if threshold is not None:
            self.threshold = threshold
        else:
            dev_metrics = ckpt.get("dev_metrics", {})
            self.threshold = dev_metrics.get("threshold", 0.5)

        self.variant = variant
        self.epoch = ckpt.get("epoch", -1) + 1

    def load_audio(self, audio_path):
        """
        Load an audio file, resample to 16 kHz, pad/truncate to
        MAX_AUDIO_LENGTH samples.

        Supports any format librosa can read: .flac, .wav, .mp3, .ogg, etc.
        """
        audio, sr = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        original_duration = len(audio) / TARGET_SAMPLE_RATE

        # Pad or truncate
        if len(audio) >= MAX_AUDIO_LENGTH:
            audio = audio[:MAX_AUDIO_LENGTH]
        else:
            repeats = (MAX_AUDIO_LENGTH // len(audio)) + 1
            audio = np.tile(audio, repeats)[:MAX_AUDIO_LENGTH]

        return audio, original_duration

    @torch.no_grad()
    def predict(self, audio_path):
        """
        Run detection on a single audio file.

        Returns
        -------
        dict with keys:
            file:       filename
            decision:   'bonafide' or 'spoof'
            score:      float P(bonafide) in [0, 1]
            threshold:  the decision threshold used
            confidence: 'high', 'medium', or 'low'
            duration:   original audio duration in seconds
        """
        audio, duration = self.load_audio(audio_path)
        x = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        score = probs[0, 1].item()  # P(bonafide)

        decision = "bonafide" if score >= self.threshold else "spoof"

        # Confidence based on distance from threshold
        margin = abs(score - self.threshold)
        if margin > 0.3:
            confidence = "high"
        elif margin > 0.1:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "file": Path(audio_path).name,
            "decision": decision,
            "score": round(score, 6),
            "threshold": round(self.threshold, 6),
            "confidence": confidence,
            "duration": round(duration, 2),
        }

    def predict_batch(self, audio_paths):
        """Run detection on multiple files. Returns list of result dicts."""
        return [self.predict(p) for p in audio_paths]


def main():
    parser = argparse.ArgumentParser(description="Deepfake audio detection inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to a single audio file")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Path to directory of audio files")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: use EER threshold from training)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to this JSON file")
    args = parser.parse_args()

    if args.audio is None and args.audio_dir is None:
        print("Provide --audio (single file) or --audio_dir (directory).")
        sys.exit(1)

    # Collect files
    audio_files = []
    if args.audio:
        audio_files.append(Path(args.audio))
    if args.audio_dir:
        exts = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}
        for f in sorted(Path(args.audio_dir).iterdir()):
            if f.suffix.lower() in exts:
                audio_files.append(f)

    if not audio_files:
        print("No audio files found.")
        sys.exit(1)

    # Initialize
    detector = DeepfakeDetector(
        args.checkpoint,
        threshold=args.threshold,
    )

    print(f"\n  Model:     {detector.variant} (epoch {detector.epoch})")
    print(f"  Device:    {detector.device}")
    print(f"  Threshold: {detector.threshold:.4f}")
    print(f"  Files:     {len(audio_files)}")
    print()

    # Run
    results = []
    for fpath in audio_files:
        try:
            result = detector.predict(fpath)
            results.append(result)

            marker = "OK" if result["decision"] == "bonafide" else "SPOOF"
            print(f"  [{marker:>5s}]  {result['file']:<40s}  "
                  f"score={result['score']:.4f}  "
                  f"conf={result['confidence']}")
        except Exception as e:
            print(f"  [ERROR]  {fpath.name}: {e}")

    # Summary
    n_bona = sum(1 for r in results if r["decision"] == "bonafide")
    n_spoof = sum(1 for r in results if r["decision"] == "spoof")
    print(f"\n  Results: {n_bona} bonafide, {n_spoof} spoof out of {len(results)} files")

    # Optionally save
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results to {args.output}")


if __name__ == "__main__":
    main()

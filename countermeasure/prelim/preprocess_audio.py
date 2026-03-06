"""
Audio Preprocessor for ASVspoof 2019 LA
========================================
Uses librosa to:
  1. Load .flac files
  2. Resample to 16 kHz
  3. Truncate or pad to a fixed length (for AASIST compatibility)
  4. Save as .npy for fast loading during training

Usage:
    python preprocess_audio.py                    # Full preprocessing
    python preprocess_audio.py --subset 500       # Quick test with 500 samples
    python preprocess_audio.py --split train      # Only preprocess train split
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

from config import (
    TARGET_SAMPLE_RATE, MAX_AUDIO_LENGTH, PAD_MODE,
    TRAIN_OUTPUT_DIR, DEV_OUTPUT_DIR, EVAL_OUTPUT_DIR,
    METADATA_OUTPUT_DIR, SMALL_SUBSET_SIZE,
    create_output_dirs,
)
from protocol_parser import load_all_protocols


def load_and_resample(flac_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Load a .flac file and resample to the target sample rate.
    
    Parameters
    ----------
    flac_path : str
        Path to the .flac audio file.
    target_sr : int
        Target sample rate in Hz.
    
    Returns
    -------
    np.ndarray
        Audio waveform as a 1D float32 array, resampled to target_sr.
    """
    # librosa.load automatically resamples to sr
    audio, sr = librosa.load(flac_path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def pad_or_truncate(audio: np.ndarray, max_len: int = MAX_AUDIO_LENGTH,
                    mode: str = PAD_MODE) -> np.ndarray:
    """
    Ensure audio is exactly max_len samples.
    
    - If longer:  truncate to max_len
    - If shorter: pad according to mode
        - "wrap": tile the audio (repeat from beginning)
        - "zero": zero-pad at the end
    
    Parameters
    ----------
    audio : np.ndarray
        1D audio waveform.
    max_len : int
        Desired output length in samples.
    mode : str
        Padding strategy: "wrap" or "zero".
    
    Returns
    -------
    np.ndarray of shape (max_len,)
    """
    length = len(audio)
    
    if length >= max_len:
        # Truncate
        return audio[:max_len]
    
    if mode == "wrap":
        # Tile the audio to fill max_len
        repeats = (max_len // length) + 1
        audio_tiled = np.tile(audio, repeats)
        return audio_tiled[:max_len]
    else:
        # Zero-pad
        padded = np.zeros(max_len, dtype=np.float32)
        padded[:length] = audio
        return padded


def preprocess_single(args: tuple) -> dict:
    """
    Preprocess a single utterance. Designed for multiprocessing.
    
    Parameters
    ----------
    args : tuple of (utt_id, flac_path, output_dir, target_sr, max_len, pad_mode)
    
    Returns
    -------
    dict with preprocessing stats (duration, original_sr, etc.)
    """
    utt_id, flac_path, output_dir, target_sr, max_len, pad_mode = args
    
    try:
        # Get original sample rate without loading
        info = sf.info(flac_path)
        original_sr = info.samplerate
        original_duration = info.duration
        
        # Load and resample
        audio = load_and_resample(flac_path, target_sr=target_sr)
        
        # Record pre-pad length
        resampled_length = len(audio)
        
        # Pad or truncate
        audio = pad_or_truncate(audio, max_len=max_len, mode=pad_mode)
        
        # Save as .npy for fast loading
        output_path = Path(output_dir) / f"{utt_id}.npy"
        np.save(output_path, audio)
        
        return {
            "utt_id":            utt_id,
            "original_sr":       original_sr,
            "original_duration": round(original_duration, 4),
            "resampled_length":  resampled_length,
            "final_length":      max_len,
            "was_padded":        resampled_length < max_len,
            "was_truncated":     resampled_length > max_len,
            "success":           True,
            "error":             None,
        }
    
    except Exception as e:
        return {
            "utt_id":  utt_id,
            "success": False,
            "error":   str(e),
        }


def preprocess_split(df, output_dir: Path, split_name: str,
                     subset_size: int = None, num_workers: int = 4):
    """
    Preprocess all utterances in a dataset split.
    
    Parameters
    ----------
    df : pd.DataFrame
        Protocol DataFrame with columns: utt_id, flac_path, label, etc.
    output_dir : Path
        Directory to save preprocessed .npy files.
    split_name : str
        Name of the split (for logging).
    subset_size : int or None
        If set, only process this many utterances.
    num_workers : int
        Number of parallel workers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if subset_size is not None:
        # Stratified subset: maintain bonafide/spoof ratio
        df_subset = df.groupby("label_str", group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), subset_size // 2),
                random_state=42
            )
        ).reset_index(drop=True)
        df = df_subset
        print(f"  Using subset of {len(df):,} utterances (stratified)")
    
    print(f"\n{'─'*50}")
    print(f"  Preprocessing {split_name.upper()} split")
    print(f"  Utterances: {len(df):,}")
    print(f"  Output dir: {output_dir}")
    print(f"  Target SR:  {TARGET_SAMPLE_RATE} Hz")
    print(f"  Max length: {MAX_AUDIO_LENGTH} samples ({MAX_AUDIO_LENGTH/TARGET_SAMPLE_RATE:.2f}s)")
    print(f"  Pad mode:   {PAD_MODE}")
    print(f"  Workers:    {num_workers}")
    print(f"{'─'*50}")
    
    # Prepare arguments for multiprocessing
    tasks = [
        (row.utt_id, row.flac_path, str(output_dir),
         TARGET_SAMPLE_RATE, MAX_AUDIO_LENGTH, PAD_MODE)
        for row in df.itertuples()
    ]
    
    results = []
    start_time = time.time()
    
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(preprocess_single, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  {split_name}", unit="utt"):
                results.append(future.result())
    else:
        # Single-threaded (easier to debug)
        for t in tqdm(tasks, desc=f"  {split_name}", unit="utt"):
            results.append(preprocess_single(t))
    
    elapsed = time.time() - start_time
    
    # Report results
    successes = [r for r in results if r["success"]]
    failures  = [r for r in results if not r["success"]]
    
    print(f"\n  Results for {split_name.upper()}:")
    print(f"    Processed: {len(successes):,} / {len(results):,}")
    print(f"    Failed:    {len(failures):,}")
    print(f"    Time:      {elapsed:.1f}s ({len(results)/elapsed:.1f} utt/s)")
    
    if successes:
        durations = [r["original_duration"] for r in successes]
        padded    = sum(1 for r in successes if r.get("was_padded", False))
        truncated = sum(1 for r in successes if r.get("was_truncated", False))
        orig_srs  = set(r["original_sr"] for r in successes)
        
        print(f"    Original SRs:  {orig_srs}")
        print(f"    Duration range: {min(durations):.2f}s – {max(durations):.2f}s")
        print(f"    Mean duration:  {np.mean(durations):.2f}s")
        print(f"    Padded:    {padded:,} ({padded/len(successes)*100:.1f}%)")
        print(f"    Truncated: {truncated:,} ({truncated/len(successes)*100:.1f}%)")
    
    if failures:
        print(f"\n Failed utterances:")
        for f in failures[:10]:
            print(f"    {f['utt_id']}: {f['error']}")
        if len(failures) > 10:
            print(f"    ... and {len(failures)-10} more")
    
    # Save preprocessing stats
    stats_path = METADATA_OUTPUT_DIR / f"{split_name}_preprocess_stats.json"
    METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({
            "split": split_name,
            "total": len(results),
            "success": len(successes),
            "failed": len(failures),
            "elapsed_seconds": round(elapsed, 2),
            "target_sr": TARGET_SAMPLE_RATE,
            "max_length": MAX_AUDIO_LENGTH,
            "pad_mode": PAD_MODE,
        }, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Preprocess ASVspoof 2019 LA audio")
    parser.add_argument("--subset", type=int, default=None,
                        help=f"Process only N utterances per split (default: all)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "dev", "eval", "all"],
                        help="Which split to preprocess")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  ASVspoof 2019 LA — Audio Preprocessing Pipeline")
    print("=" * 60)
    
    # Create output directories
    create_output_dirs()
    
    # Load protocol files
    print("\nLoading protocol files...")
    protocols = load_all_protocols(verbose=True)
    
    # Preprocess requested splits
    split_configs = {
        "train": (protocols["train"], TRAIN_OUTPUT_DIR),
        "dev":   (protocols["dev"],   DEV_OUTPUT_DIR),
        "eval":  (protocols["eval"],  EVAL_OUTPUT_DIR),
    }
    
    splits_to_process = (
        [args.split] if args.split != "all"
        else ["train", "dev", "eval"]
    )
    
    for split_name in splits_to_process:
        df, out_dir = split_configs[split_name]
        preprocess_split(
            df, out_dir, split_name,
            subset_size=args.subset,
            num_workers=args.workers,
        )
    
    print("\n" + "=" * 60)
    print("  Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

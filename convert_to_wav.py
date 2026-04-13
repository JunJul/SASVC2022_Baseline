"""
Convert VoxCeleb2 .m4a files to .wav (16kHz mono) for faster training.
Runs in parallel using multiple ffmpeg processes.

Output goes to ECAPATDNN/Voxceleb2_dev_wav/parta/ mirroring the source structure.
After conversion, update train_campplus_sv.py --train_path to point to the wav dir.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def find_ffmpeg():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "tools" / "ffmpeg.exe",
        script_dir / "tools" / "ffmpeg",
        "ffmpeg",
    ]
    for c in candidates:
        if str(c) == "ffmpeg":
            # Check if ffmpeg is in PATH
            try:
                subprocess.run([str(c), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return str(c)
            except FileNotFoundError:
                continue
        elif Path(c).is_file():
            return str(c)
    raise RuntimeError("ffmpeg not found. Place it in tools/ or add to PATH.")


def convert_one(ffmpeg_bin, src, dst):
    """Convert a single .m4a to .wav (16kHz mono)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "skip"
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(src),
             "-ac", "1", "-ar", "16000", "-v", "error", str(dst)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        return "ok"
    except subprocess.CalledProcessError as e:
        return f"fail: {e.stderr.decode().strip()}"


def main():
    parser = argparse.ArgumentParser(description="Convert VoxCeleb2 m4a to wav")
    parser.add_argument("--src", type=str,
                        default="ECAPATDNN/Voxceleb2_dev_aac/parta",
                        help="Source directory with .m4a files")
    parser.add_argument("--dst", type=str,
                        default="ECAPATDNN/Voxceleb2_dev_wav/parta",
                        help="Output directory for .wav files")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel ffmpeg processes")
    args = parser.parse_args()

    ffmpeg_bin = find_ffmpeg()
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    # Collect all .m4a files
    m4a_files = sorted(src_dir.rglob("*.m4a"))
    print(f"Found {len(m4a_files)} .m4a files in {src_dir}")

    # Build conversion pairs
    pairs = []
    for m4a in m4a_files:
        rel = m4a.relative_to(src_dir)
        wav = dst_dir / rel.with_suffix(".wav")
        pairs.append((m4a, wav))

    # Check how many already exist
    existing = sum(1 for _, wav in pairs if wav.exists())
    remaining = len(pairs) - existing
    print(f"Already converted: {existing}, remaining: {remaining}")

    if remaining == 0:
        print("All files already converted!")
        # Generate train list for wav
        generate_wav_train_list(dst_dir)
        return

    done = existing
    failed = 0
    total = len(pairs)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for src, dst in pairs:
            f = pool.submit(convert_one, ffmpeg_bin, src, dst)
            futures[f] = (src, dst)

        for f in as_completed(futures):
            result = f.result()
            if result == "ok":
                done += 1
            elif result == "skip":
                done += 1
            else:
                failed += 1
                src, _ = futures[f]
                print(f"  FAIL {src}: {result}", file=sys.stderr)

            if done % 5000 == 0 or done == total:
                print(f"  Progress: {done}/{total} ({done*100//total}%), failed: {failed}")

    print(f"\nDone! Converted: {done - existing}, skipped: {existing}, failed: {failed}")
    generate_wav_train_list(dst_dir)


def generate_wav_train_list(wav_dir):
    """Generate a train list pointing to .wav files."""
    wav_dir = Path(wav_dir)
    lines = []
    for wav in sorted(wav_dir.rglob("*.wav")):
        rel = wav.relative_to(wav_dir)
        speaker_id = rel.parts[0]
        lines.append(f"{speaker_id} {rel.as_posix()}")

    out = "data/train_list_campplus_wav.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_spk = len(set(l.split()[0] for l in lines))
    print(f"\nGenerated {out}: {len(lines)} utterances, {n_spk} speakers")
    print(f"\nTo train with wav files, run:")
    print(f"  python train_campplus_sv.py --train_list {out} --train_path {wav_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Count total audio duration (hours) under a directory.

Defaults to `ECAPATDNN/Voxceleb2_dev_aac` in the workspace root.
Tries `ffprobe` first, falls back to `mutagen` if available.
"""
import os
import sys
import argparse
import subprocess
import shutil


def get_duration_ffprobe(path):
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        out = out.decode().strip()
        if out:
            return float(out)
    except Exception:
        return None


def get_duration_mutagen(path):
    try:
        from mutagen import File
    except Exception:
        return None
    try:
        f = File(path)
        if f is None:
            return None
        info = getattr(f, "info", None)
        if info is None or not hasattr(info, "length"):
            return None
        return float(info.length)
    except Exception:
        return None


def format_hms(total_seconds):
    secs = int(total_seconds + 0.5)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Compute total audio duration under a directory (prints hours)."
    )
    parser.add_argument(
        "--dir",
        "-d",
        default=os.path.join("ECAPATDNN", "Voxceleb2_dev_aac"),
        help="Root directory containing the Voxceleb2 aac dataset",
    )
    parser.add_argument(
        "--exts",
        default="wav,flac,mp3,m4a,aac,ogg",
        help="Comma-separated audio extensions to include (no dots)",
    )
    args = parser.parse_args()

    root = args.dir
    if not os.path.exists(root):
        print(f"Directory not found: {root}")
        sys.exit(1)

    exts = tuple(
        ["." + e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip()]
    )

    total_seconds = 0.0
    file_count = 0

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            path = os.path.join(dirpath, fn)
            dur = get_duration_ffprobe(path)
            if dur is None:
                dur = get_duration_mutagen(path)
            if dur is None:
                print(f"Warning: could not read duration for {path}", file=sys.stderr)
                continue
            total_seconds += dur
            file_count += 1

    hours = total_seconds / 3600.0

    print(f"Files processed: {file_count}")
    print(f"Total seconds: {total_seconds:.2f}")
    print(f"Total hours: {hours:.4f}")
    print(f"Total duration (HH:MM:SS): {format_hms(total_seconds)}")


if __name__ == "__main__":
    main()

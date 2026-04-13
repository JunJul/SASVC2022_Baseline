#!/usr/bin/env python3
"""Check accessibility of audio files under a directory.

By default checks `ECAPATDNN/Voxceleb2_dev_aac` in the workspace root.
Per-file checks (fast): existence, read permission, open-read test.
Optional: use `ffprobe` to probe durations when available.

Example:
  python scripts/check_voxceleb2_access.py --dir ECAPATDNN/Voxceleb2_dev_aac --sample 50
"""
import os
import sys
import argparse
import shutil
import subprocess


def has_ffprobe():
    return shutil.which("ffprobe") is not None


def probe_duration(path):
    if not has_ffprobe():
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
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        out = out.decode().strip()
        if out:
            return float(out)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Check access to Voxceleb2 audio files")
    parser.add_argument("--dir", "-d", default=os.path.join("ECAPATDNN", "Voxceleb2_dev_aac"), help="Root dir")
    parser.add_argument("--exts", default="wav,flac,mp3,m4a,aac,ogg", help="Comma-separated extensions")
    parser.add_argument("--sample", "-n", type=int, default=0, help="Number of files to run the open/ffprobe checks on (0 = none)")
    parser.add_argument("--max-list", type=int, default=20, help="Max problematic files to list")
    args = parser.parse_args()

    root = args.dir
    if not os.path.exists(root):
        print(f"Directory not found: {root}")
        sys.exit(2)

    exts = tuple("." + e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip())

    all_files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                all_files.append(os.path.join(dp, fn))

    total = len(all_files)
    print(f"Found {total} audio files under: {root}")

    unreadable = []
    open_failed = []
    probe_ok = 0
    probe_failed = 0

    sample_n = args.sample if args.sample > 0 else 0
    to_check = all_files[:sample_n] if sample_n else []

    for path in to_check:
        if not os.path.exists(path):
            unreadable.append((path, "missing"))
            continue
        if not os.access(path, os.R_OK):
            unreadable.append((path, "no-read-permission"))
            continue
        try:
            with open(path, "rb") as fh:
                fh.read(4)
        except Exception as e:
            open_failed.append((path, str(e)))
            continue

        dur = probe_duration(path)
        if dur is None:
            probe_failed += 1
        else:
            probe_ok += 1

    print()
    print(f"Sample checks performed: {len(to_check)}")
    print(f"Readable files in sample: {len(to_check) - len(unreadable) - len(open_failed)}")
    print(f"Files unreadable (missing/no-perm): {len(unreadable)}")
    print(f"Files open-failed: {len(open_failed)}")
    if sample_n:
        if has_ffprobe():
            print(f"ffprobe succeeded: {probe_ok}, failed: {probe_failed}")
        else:
            print("ffprobe not available — skip duration probes")

    if unreadable:
        print() 
        print(f"Listing up to {args.max_list} unreadable files:")
        for p, reason in unreadable[: args.max_list]:
            print(f" - {p} -> {reason}")

    if open_failed:
        print()
        print(f"Listing up to {args.max_list} open-failed files:")
        for p, err in open_failed[: args.max_list]:
            print(f" - {p} -> {err}")

    if total == 0:
        print("No audio files found. Check the directory path and extensions.")


if __name__ == "__main__":
    main()

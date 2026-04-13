"""
Prepare VoxCeleb2 training list for CAM++ speaker verification.

Generates a train_list.txt that points directly to .m4a files — no conversion needed.
The CAM++ dataloader reads .m4a on-the-fly via ffmpeg.
"""

import argparse
import os
from pathlib import Path


def create_train_list(src_dir, output_file):
    """Generate train_list.txt from .m4a files in src_dir.

    Handles both flat layout (speaker/video/file.m4a) and
    partitioned layout (parta/speaker/video/file.m4a).
    """
    src = Path(src_dir)
    lines = []

    # Check if src has parta/partb/etc. subdirs
    has_parts = any((src / f"part{c}").is_dir() for c in "abcdefgh")

    if has_parts:
        for part_dir in sorted(src.iterdir()):
            if part_dir.is_dir() and part_dir.name.startswith("part"):
                for m4a in part_dir.rglob("*.m4a"):
                    rel = m4a.relative_to(part_dir)
                    speaker_id = rel.parts[0]
                    # train_path will be set to part_dir, so use path relative to part_dir
                    lines.append(f"{speaker_id} {rel.as_posix()}")
    else:
        for m4a in sorted(src.rglob("*.m4a")):
            rel = m4a.relative_to(src)
            speaker_id = rel.parts[0]
            lines.append(f"{speaker_id} {rel.as_posix()}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_speakers = len(set(l.split()[0] for l in lines))
    print(f"Created {output_file}: {len(lines)} utterances, {n_speakers} speakers")
    return n_speakers


def main():
    parser = argparse.ArgumentParser(description="Generate VoxCeleb2 training list for CAM++")
    parser.add_argument(
        "--src_dir", type=str,
        default="ECAPATDNN/Voxceleb2_dev_aac/parta",
        help="Directory with .m4a files (speaker/video/file.m4a)",
    )
    parser.add_argument(
        "--train_list", type=str,
        default="data/train_list_campplus.txt",
        help="Output training list file",
    )
    args = parser.parse_args()

    create_train_list(args.src_dir, args.train_list)
    print(f"\nTo train, run:")
    print(f"  python train_campplus_sv.py --train_list {args.train_list} --train_path {args.src_dir}")


if __name__ == "__main__":
    main()

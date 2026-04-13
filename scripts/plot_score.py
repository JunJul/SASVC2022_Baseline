#!/usr/bin/env python3
"""Plot ACC and EER from a training `score.txt` file.

Usage:
  python scripts/plot_score.py ECAPATDNN/exps/exp1/score.txt --out ECAPATDNN/exps/exp1/score_plot.png
"""
import re
import sys
import argparse
from pathlib import Path


def parse_score_file(path):
    epoch_re = re.compile(r"(\d+)\s*epoch")
    acc_re = re.compile(r"ACC\s*([0-9]+\.?[0-9]*)%")
    eer_re = re.compile(r"EER\s*([0-9]+\.?[0-9]*)%")

    epochs = []
    accs = []
    eers = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e_match = epoch_re.search(line)
            a_match = acc_re.search(line)
            r_match = eer_re.search(line)
            if e_match and a_match and r_match:
                epochs.append(int(e_match.group(1)))
                accs.append(float(a_match.group(1)))
                eers.append(float(r_match.group(1)))

    return epochs, accs, eers


def write_plot(epochs, accs, eers, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required to generate the plot. Install with: pip install matplotlib", file=sys.stderr)
        raise

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, accs,  label="ACC (%)", color="#2ca02c")
    ax.plot(epochs, eers, label="EER (%)", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Training ACC and EER by Epoch")
    ax.legend(loc="best")
    ax.set_xlim(min(epochs) - 1, max(epochs) + 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ACC and EER from score.txt")
    parser.add_argument("score", help="Path to score.txt file")
    parser.add_argument("--out", "-o", default=None, help="Output PNG path")
    args = parser.parse_args()

    score = Path(args.score)
    if not score.exists():
        print(f"score file not found: {score}")
        sys.exit(1)

    epochs, accs, eers = parse_score_file(score)
    if not epochs:
        print("No valid epoch/ACC/EER lines parsed from the file.")
        sys.exit(2)

    out = args.out
    if out is None:
        out = score.parent / "score_plot.png"

    write_plot(epochs, accs, eers, out)


if __name__ == "__main__":
    main()

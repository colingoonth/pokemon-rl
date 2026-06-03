"""Plot training curves from a runs/<name>/metrics.csv file.

Usage:
  uv run python -m pokerl.scripts.plot_metrics runs/elsa_brock/metrics.csv
  uv run python -m pokerl.scripts.plot_metrics runs/elsa_brock/metrics.csv --out curves.png

Renders a 2x3 panel: return, episodes, tiles, entropy, policy_loss, value_loss.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output png path. Defaults to <csv-dir>/curves.png.")
    p.add_argument("--smooth", type=int, default=1,
                   help="Rolling-mean window (in iters) for smoothing.")
    return p.parse_args()


def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    rows: list[dict[str, str]] = []
    with args.csv.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("CSV is empty")

    iters = np.array([int(r["iteration"]) for r in rows])
    steps = np.array([int(r["global_step"]) for r in rows])
    rets = np.array([float(r["mean_return_last20"]) for r in rows])
    eps = np.array([int(r["episodes_completed"]) for r in rows])
    tiles = np.array([int(r["unique_tiles_total"]) for r in rows])
    entropy = np.array([float(r["entropy"]) for r in rows])
    pg = np.array([float(r["policy_loss"]) for r in rows])
    vl = np.array([float(r["value_loss"]) for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    panels = [
        (axes[0, 0], rets,    "Mean return (last 20 eps)", "return"),
        (axes[0, 1], eps,     "Episodes completed (cumulative)", "count"),
        (axes[0, 2], tiles,   "Unique tiles visited (all envs)", "count"),
        (axes[1, 0], entropy, "Policy entropy", "H"),
        (axes[1, 1], pg,      "Policy loss",  "loss"),
        (axes[1, 2], vl,      "Value loss",   "loss"),
    ]

    for ax, y, title, ylabel in panels:
        x = steps[: len(y)]
        if args.smooth > 1 and len(y) >= args.smooth:
            y_s = smooth(y, args.smooth)
            x_s = x[len(x) - len(y_s):]
            ax.plot(x, y, color="lightgray", linewidth=0.8, label="raw")
            ax.plot(x_s, y_s, color="C0", linewidth=1.5, label=f"smooth({args.smooth})")
            ax.legend(loc="best", fontsize=8)
        else:
            ax.plot(x, y, color="C0")
        ax.set_title(title)
        ax.set_xlabel("global step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    fig.suptitle(args.csv.parent.name, fontsize=12)
    fig.tight_layout()

    out = args.out if args.out is not None else args.csv.parent / "curves.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

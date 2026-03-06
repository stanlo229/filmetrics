"""
plot_fitnk.py — Plot n(λ) and k(λ) from a Filmetrics .fitnk file.

Usage:
    python _code/plot_fitnk.py <path/to/file.fitnk> [--output <path.png>]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_fitnk(path):
    """Parse a Filmetrics v3 .fitnk file and return (wavelength_nm, n, k) arrays."""
    wl, n, k = [], [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            try:
                wl.append(float(parts[0]))
                n.append(float(parts[1]))
                k.append(float(parts[2]))
            except ValueError:
                continue
    return np.array(wl), np.array(n), np.array(k)


def plot_fitnk(fitnk_path, output_path=None):
    path = Path(fitnk_path)
    wl, n, k = load_fitnk(path)

    has_k = np.any(k != 0)
    nrows = 2 if has_k else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 4 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    fig.suptitle(path.name, fontsize=10)

    axes[0].plot(wl, n, color="royalblue", lw=1.5)
    axes[0].set_ylabel("n  (refractive index)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    if has_k:
        axes[1].plot(wl, k, color="firebrick", lw=1.5)
        axes[1].set_ylabel("k  (extinction coefficient)")
        axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Wavelength (nm)")
    fig.tight_layout()

    if output_path is None:
        output_path = path.with_suffix(".png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot n/k from a .fitnk file")
    parser.add_argument("fitnk", help="Path to the .fitnk file")
    parser.add_argument("--output", default=None, help="Output PNG path (default: same folder as input)")
    args = parser.parse_args()
    plot_fitnk(args.fitnk, args.output)


if __name__ == "__main__":
    main()

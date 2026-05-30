"""
paper_figure_6_sic.py — Figure 6: SIC curves for Model A (2M), A (10M), B2, C.
SIC = epsilon_S / sqrt(epsilon_B) at each threshold.
Two panels: 2-prong (left), 3-prong (right).
No GPU required. Output: data/results/paper_figures/fig6_sic.{pdf,png}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUT = Path("data/results/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    with open(path) as f:
        return json.load(f)


def sic_from_roc(fpr, tpr):
    """SIC = tpr / sqrt(fpr), defined where fpr > 0."""
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    mask = fpr > 0
    sic = np.zeros_like(tpr)
    sic[mask] = tpr[mask] / np.sqrt(fpr[mask])
    return tpr, sic  # x = signal efficiency, y = SIC


def find_model_key(data, substring):
    for k in data:
        if substring in k:
            return k
    return None


def main():
    d2m  = load("data/results/ablation_2M_final/ablation_results.json")
    d10m = load("data/results/ablation_10M_final/ablation_results.json")

    key_A   = find_model_key(d2m, "Model A")
    key_B2  = find_model_key(d2m, "B2")
    key_C   = find_model_key(d2m, "autoencoder")
    key_A10 = find_model_key(d10m, "Model A")

    models = [
        ("Real CMS (2M, ep40)",  d2m[key_A],   "#2166ac", "-",  "o"),
        ("Real CMS (10M, ep75)", d10m[key_A10], "#4dac26", "-",  "s"),
        ("Simulation (1M, ep75)",d2m[key_B2],   "#d6604d", "--", "^"),
        ("Autoencoder (2M)",     d2m[key_C],    "#888888", ":",  "D"),
    ]

    signals = ["2-prong", "3-prong"]
    titles  = ["2-prong signal", "3-prong signal"]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2), sharey=False)
    fig.subplots_adjust(wspace=0.32)

    for ax, sig, title in zip(axes, signals, titles):
        for label, mdata, color, ls, marker in models:
            tpr, sic = sic_from_roc(mdata[sig]["fpr"], mdata[sig]["tpr"])
            n = max(1, len(tpr) // 400)
            ax.plot(tpr[::n], sic[::n],
                    color=color, ls=ls, lw=1.6, label=label)

            # Mark max SIC with annotation
            max_idx = np.argmax(sic)
            ax.plot(tpr[max_idx], sic[max_idx],
                    marker=marker, color=color, ms=5, zorder=5)

        ax.set_xlim(0.0, 0.80)
        ax.set_ylim(0.85, 1.12)   # zoom to region of interest
        ax.set_xlabel("Signal efficiency $\\varepsilon_S$")
        ax.set_ylabel("SIC = $\\varepsilon_S / \\sqrt{\\varepsilon_B}$")
        ax.set_title(title, fontsize=9, pad=4)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7, label="Random baseline")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    axes[0].legend(loc="upper right", frameon=False, fontsize=7.5)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig6_sic.{ext}")
    plt.close(fig)
    print(f"fig6_sic saved to {OUT}/")

    # Print max SIC values for reporting
    print("\nMax SIC summary:")
    print(f"{'Model':<28} {'2-prong max SIC':>16} {'3-prong max SIC':>16}")
    print("-" * 62)
    for label, mdata, *_ in models:
        for sig in signals:
            _, sic = sic_from_roc(mdata[sig]["fpr"], mdata[sig]["tpr"])
            print(f"  {label:<26} {sig}: {np.max(sic):.2f}")


if __name__ == "__main__":
    main()

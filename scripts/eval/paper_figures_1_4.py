"""
paper_figures_1_4.py — Generate paper Figures 1–4 from saved JSON results.
No GPU required. Outputs to data/results/paper_figures/.

Figures:
  fig1_convergence.pdf/.png   — Convergence dynamics (A-2M, A-10M, B2)
  fig2_scaling.pdf/.png       — Scaling curve with error bars
  fig3_aug_ablation.pdf/.png  — Augmentation ablation bar chart (2 panels)
  fig4_signal_injection.pdf/.png — Signal injection AUC vs S/B
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

# ── Style ────────────────────────────────────────────────────────────────────
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

W_SINGLE = 5.5   # NeurIPS single-column width in inches
W_DOUBLE = 5.5   # still single-col for 2-panel (each panel ~2.5 in)


def load(path):
    with open(path) as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Convergence Dynamics
# ─────────────────────────────────────────────────────────────────────────────
def fig1_convergence():
    # Model A 2M — epochs 10-50 from epoch_auc
    d2m = load("data/results/epoch_auc/epoch_auc_results.json")
    a2m_ep  = d2m["Model A"]["epochs"]           # [10,20,30,40,50]
    a2m_auc = d2m["Model A"]["2-prong"]

    # Model A 10M — epochs 10-75 from epoch_auc_10M_v4; filter to 10,20,...,70,75
    d10m = load("data/results/epoch_auc_10M_v4/epoch_auc_results.json")
    keep = {10, 20, 30, 40, 50, 60, 70, 75}
    a10m_ep, a10m_auc = zip(*[
        (ep, auc)
        for ep, auc in zip(d10m["Model A"]["epochs"], d10m["Model A"]["2-prong"])
        if ep in keep
    ])

    # B2 (sim, 75ep) — tracked as "Model A" in epoch_auc_B2_75ep
    db2 = load("data/results/epoch_auc_B2_75ep/epoch_auc_results.json")
    b2_ep_all  = db2["Model A"]["epochs"]        # [10,20,30,40,50,55,60,65,70,75]
    b2_auc_all = db2["Model A"]["2-prong"]
    keep_b2 = {10, 20, 30, 40, 50, 60, 70, 75}
    b2_ep, b2_auc = zip(*[
        (ep, auc)
        for ep, auc in zip(b2_ep_all, b2_auc_all)
        if ep in keep_b2
    ])

    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.4))

    ax.plot(a2m_ep, a2m_auc, color="#2166ac", ls="-",  marker="o",
            ms=5, lw=1.5, label="Real CMS (2M)")
    ax.plot(a10m_ep, a10m_auc, color="#4dac26", ls="-", marker="s",
            ms=5, lw=1.5, label="Real CMS (10M)")
    ax.plot(b2_ep, b2_auc,   color="#d6604d", ls="--", marker="^",
            ms=5, lw=1.5, label="Simulation (1M)")

    # Optional: shade the ep10 gap
    a2m_ep10 = a2m_auc[0]   # 0.6020
    b2_ep10  = b2_auc[0]    # 0.5112
    ax.axvspan(8, 12, alpha=0.07, color="gray", lw=0)
    ax.annotate(
        "", xy=(11, b2_ep10 + 0.003), xytext=(11, a2m_ep10 - 0.003),
        arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8),
    )
    ax.text(12.5, (a2m_ep10 + b2_ep10) / 2, "0.065\ngap",
            fontsize=7, color="gray", va="center")

    ax.set_xlim(5, 80)
    ax.set_ylim(0.50, 0.65)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("2-prong AUC")
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 75])
    ax.legend(loc="lower right", frameon=False)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_convergence.{ext}")
    plt.close(fig)
    print("fig1_convergence saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Scaling Curve
# ─────────────────────────────────────────────────────────────────────────────
def fig2_scaling():
    sizes = [2e6, 4e6, 6e6, 10e6]
    paths = [
        "data/results/ablation_2M_final/ablation_results.json",
        "data/results/ablation_4M_75ep/ablation_results.json",
        "data/results/ablation_6M_final/ablation_results.json",
        "data/results/ablation_10M_final/ablation_results.json",
    ]

    auc_2p, std_2p, auc_3p, std_3p = [], [], [], []
    for path in paths:
        d = load(path)
        for k, v in d.items():
            if "Model A" in k and "contrastive" in k.lower():
                auc_2p.append(v["2-prong"]["mean_auc"])
                std_2p.append(v["2-prong"]["std_auc"])
                auc_3p.append(v["3-prong"]["mean_auc"])
                std_3p.append(v["3-prong"]["std_auc"])

    # B2 reference
    d_b2 = load("data/results/ablation_2M_final/ablation_results.json")
    for k, v in d_b2.items():
        if "B2" in k:
            b2_2p = v["2-prong"]["mean_auc"]  # 0.6286

    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.4))

    ax.errorbar(sizes, auc_2p, yerr=std_2p,
                color="#2166ac", ls="-", marker="o", ms=5, lw=1.5,
                capsize=3, label="2-prong")
    ax.errorbar(sizes, auc_3p, yerr=std_3p,
                color="#d6604d", ls="--", marker="s", ms=5, lw=1.5,
                capsize=3, label="3-prong")

    # B2 reference line
    ax.axhline(b2_2p, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.text(10.5e6, b2_2p + 0.002, "B2 (sim, 1M)",
            fontsize=7.5, color="gray", va="bottom", ha="right")

    ax.set_xscale("log")
    ax.set_xlim(1.5e6, 12e6)
    ax.set_ylim(0.50, 0.66)
    ax.set_xlabel("Training jets")
    ax.set_ylabel("AUC")
    # Force exactly the 4 data ticks; suppress all auto minor/major log ticks
    ax.set_xticks(sizes)
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x/1e6)}M")
    )
    ax.legend(loc="lower right", frameon=False)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_scaling.{ext}")
    plt.close(fig)
    print("fig2_scaling saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Augmentation Ablation
# ─────────────────────────────────────────────────────────────────────────────
def fig3_aug_ablation():
    baseline_2p = 0.6065
    baseline_3p = 0.5165

    rows = [
        ("No soft-drop",    "no_softdrop"),
        ("No rotation",     "no_rotate"),
        ("No collinear",    "no_collinear"),   # shortened to avoid crowding
        ("No translation",  "no_translate"),
        ("No pT smearing",  "no_smear"),
    ]

    labels, d2p, s2p, d3p, s3p = [], [], [], [], []
    for label, key in rows:
        d = load(f"data/results/aug_ablation/{key}/ablation_results.json")
        for k, v in d.items():
            if "Model A" in k and "contrastive" in k.lower():
                d2p.append(v["2-prong"]["mean_auc"] - baseline_2p)
                s2p.append(v["2-prong"]["std_auc"])
                d3p.append(v["3-prong"]["mean_auc"] - baseline_3p)
                s3p.append(v["3-prong"]["std_auc"])
        labels.append(label)

    y = np.arange(len(labels))
    colors_2p = ["#d6604d" if v < 0 else "#4dac26" for v in d2p]
    colors_3p = ["#d6604d" if v < 0 else "#4dac26" for v in d3p]

    # Wider figure, more height per bar, constrained layout
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.8, 3.0), sharey=True,
        constrained_layout=True,
    )

    XMIN, XMAX = -0.075, 0.030   # extra room so text never clips axis edge

    def _bars(ax, deltas, stds, colors, title):
        ax.barh(y, deltas, xerr=stds, color=colors,
                height=0.55, capsize=3,
                error_kw={"lw": 0.8, "capthick": 0.8},
                zorder=3)
        ax.axvline(0, color="black", lw=0.8, zorder=4)
        ax.set_xlim(XMIN, XMAX)
        ax.set_xlabel("Δ AUC", labelpad=3)
        ax.set_title(title, fontsize=9, pad=5)
        ax.tick_params(axis="y", left=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([-0.06, -0.04, -0.02, 0.00, 0.02])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        # Place value text PAST the error bar cap so it never overlaps the bar
        for i, (val, std) in enumerate(zip(deltas, stds)):
            if val < 0:
                xpos = val - std - 0.005   # left of error cap
                ha   = "right"
            else:
                xpos = val + std + 0.005   # right of error cap
                ha   = "left"
            # Clamp so text stays inside axes
            xpos = max(XMIN + 0.002, min(xpos, XMAX - 0.002))
            ax.text(xpos, i, f"{val:+.3f}", va="center", ha=ha,
                    fontsize=7, color="black")

    _bars(ax1, d2p, s2p, colors_2p, "2-prong")
    _bars(ax2, d3p, s3p, colors_3p, "3-prong")

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.invert_yaxis()

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_aug_ablation.{ext}")
    plt.close(fig)
    print("fig3_aug_ablation saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Signal Injection
# ─────────────────────────────────────────────────────────────────────────────
def fig4_signal_injection():
    d = load("data/results/signal_injection/signal_injection.json")

    fracs, aucs, stds = [], [], []
    for r in d["results"]:
        fracs.append(r["fraction"] * 100)   # to percent
        aucs.append(r["mean_auc"])
        stds.append(r["std_auc"])

    fracs = np.array(fracs)
    aucs  = np.array(aucs)
    stds  = np.array(stds)

    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.4))

    ax.fill_between(fracs, aucs - stds, aucs + stds,
                    color="#2166ac", alpha=0.15)
    ax.plot(fracs, aucs, color="#2166ac", ls="-", marker="o",
            ms=5, lw=1.5, label="Model A (10M, ep75)")

    ax.axhline(0.50, color="gray", ls="--", lw=0.9, alpha=0.7,
               label="Random (AUC = 0.50)")

    ax.set_xscale("log")
    ax.set_xlim(0.008, 12)
    ax.set_ylim(0.45, 0.70)
    ax.set_xlabel("Signal fraction S/B (%)")
    ax.set_ylabel("2-prong AUC")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:g}")
    )
    ax.legend(loc="upper right", frameon=False)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_signal_injection.{ext}")
    plt.close(fig)
    print("fig4_signal_injection saved")


if __name__ == "__main__":
    fig1_convergence()
    fig2_scaling()
    fig3_aug_ablation()
    fig4_signal_injection()
    print(f"\nAll figures saved to {OUT}/")

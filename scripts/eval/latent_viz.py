"""
latent_viz.py — t-SNE and UMAP visualisation of Model A and B2 latent spaces.

Encodes LHCO background + 2-prong + 3-prong signal jets through both models,
then projects to 2-D with t-SNE and UMAP.  Produces:
  - 2×2 panel: (Model A | Model B2) × (t-SNE | UMAP), coloured by jet class
  - k-NN anomaly score overlay on Model A t-SNE embedding

Usage:
    python latent_viz.py \
        --ckpt_A  data/checkpoints/phase3/model_A_10M_v4/encoder_epoch075.pt \
        --ckpt_B2 data/checkpoints/phase2/model_B2/encoder_epoch050.pt \
        --out_dir data/results/latent_viz
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))


import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import umap

from src.data.lhco_loader import LHCOJetDataset
from src.data.preprocessed_loader import PreprocessedJetDatasetCached
from src.models.encoder import JetEncoder
from src.anomaly.knn_scorer import KNNAnomalyScorer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_contrastive(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    state = ckpt["model"]
    input_dim = state["input_proj.weight"].shape[1]
    model = JetEncoder(
        input_dim=input_dim,
        d_model=a.get("d_model", 128),
        nhead=a.get("nhead", 8),
        num_layers=a.get("num_layers", 4),
        latent_dim=a.get("latent_dim", 256),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def encode_backbone(model, xs, masks, device, batch_size=1024):
    """Pooled transformer backbone output (d_model dim), not L2-normalised."""
    zs = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            xb = xs[i:i + batch_size].to(device)
            mb = masks[i:i + batch_size].to(device)
            zs.append(model.backbone(xb, mb).cpu())
    return torch.cat(zs, dim=0).numpy()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bg(path, n, rng):
    ds = PreprocessedJetDatasetCached(path)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    xs    = torch.from_numpy(ds._x[idx])
    masks = torch.from_numpy(ds._mask[idx])
    return xs, masks


def load_signal(path, n, rng):
    ds = LHCOJetDataset(path, is_signal=True)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    xs    = torch.stack([ds[i][0] for i in idx])
    masks = torch.stack([ds[i][1] for i in idx])
    return xs, masks


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def run_tsne(z, perplexity=40, seed=42):
    z_scaled = StandardScaler().fit_transform(z)
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                n_jobs=-1).fit_transform(z_scaled)


def run_umap(z, n_neighbors=30, min_dist=0.1, seed=42):
    z_scaled = StandardScaler().fit_transform(z)
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                     min_dist=min_dist, random_state=seed).fit_transform(z_scaled)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

CLASSES = {
    "Background": {"color": "#aaaaaa", "alpha": 0.25, "zorder": 1, "s": 4},
    "2-prong signal": {"color": "#e6504a", "alpha": 0.70, "zorder": 3, "s": 10},
    "3-prong signal": {"color": "#4a90e6", "alpha": 0.70, "zorder": 2, "s": 10},
}


def scatter_classes(ax, proj, labels, title):
    names = ["Background", "2-prong signal", "3-prong signal"]
    for i, name in enumerate(names):
        mask = labels == i
        kw = CLASSES[name]
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=kw["color"], alpha=kw["alpha"],
                   s=kw["s"], zorder=kw["zorder"],
                   label=f"{name} (n={mask.sum():,})", rasterized=True)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def scatter_score(ax, proj, scores, title):
    vmin, vmax = np.percentile(scores, [2, 98])
    sc = ax.scatter(proj[:, 0], proj[:, 1],
                    c=scores, cmap="plasma",
                    vmin=vmin, vmax=vmax,
                    s=4, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="k-NN anomaly score", shrink=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rng = np.random.default_rng(args.seed)

    # ---- load data ---------------------------------------------------------
    print(f"\nLoading background ({args.n_bg:,} jets)...")
    bg_xs, bg_masks = load_bg(args.lhco_bg, args.n_bg, rng)

    print(f"Loading 2-prong signal ({args.n_sig:,} jets)...")
    p2_xs, p2_masks = load_signal(args.lhco_2p, args.n_sig, rng)

    print(f"Loading 3-prong signal ({args.n_sig:,} jets)...")
    p3_xs, p3_masks = load_signal(args.lhco_3p, args.n_sig, rng)

    all_xs    = torch.cat([bg_xs,    p2_xs,    p3_xs],    dim=0)
    all_masks = torch.cat([bg_masks, p2_masks, p3_masks], dim=0)
    labels    = np.array([0] * len(bg_xs) +
                         [1] * len(p2_xs) +
                         [2] * len(p3_xs))
    print(f"Total jets: {len(all_xs):,}  (bg={len(bg_xs):,}, "
          f"2p={len(p2_xs):,}, 3p={len(p3_xs):,})")

    # ---- encode with both models -------------------------------------------
    print("\nLoading Model A...")
    model_A = load_contrastive(args.ckpt_A, device)
    print("Encoding with Model A...")
    z_A = encode_backbone(model_A, all_xs, all_masks, device)
    del model_A
    torch.cuda.empty_cache()

    print("\nLoading Model B2...")
    model_B2 = load_contrastive(args.ckpt_B2, device)
    print("Encoding with Model B2...")
    z_B2 = encode_backbone(model_B2, all_xs, all_masks, device)
    del model_B2
    torch.cuda.empty_cache()

    # ---- k-NN anomaly scores on Model A (using bg as reference) ------------
    print("\nComputing k-NN anomaly scores (Model A)...")
    scorer = KNNAnomalyScorer(k=10)
    scorer.fit(z_A[:len(bg_xs)])          # reference = background
    knn_scores = scorer.score(z_A)

    # ---- dimensionality reduction ------------------------------------------
    print("\nRunning t-SNE on Model A...")
    tsne_A = run_tsne(z_A, perplexity=args.perplexity)
    print("Running t-SNE on Model B2...")
    tsne_B2 = run_tsne(z_B2, perplexity=args.perplexity)
    print("Running UMAP on Model A...")
    umap_A = run_umap(z_A, n_neighbors=args.umap_neighbors)
    print("Running UMAP on Model B2...")
    umap_B2 = run_umap(z_B2, n_neighbors=args.umap_neighbors)

    # ---- Figure 1: 2×2 class-coloured panel --------------------------------
    print("\nPlotting class panels...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(
        "Latent space structure: real-data (Model A) vs sim-data (Model B2) contrastive encoder\n"
        f"LHCO test set — {len(bg_xs):,} background, {len(p2_xs):,} 2-prong, {len(p3_xs):,} 3-prong",
        fontsize=12, y=1.01
    )

    scatter_classes(axes[0, 0], tsne_A,   labels, "Model A (real, 10M)  —  t-SNE")
    scatter_classes(axes[0, 1], tsne_B2,  labels, "Model B2 (sim)  —  t-SNE")
    scatter_classes(axes[1, 0], umap_A,   labels, "Model A (real, 10M)  —  UMAP")
    scatter_classes(axes[1, 1], umap_B2,  labels, "Model B2 (sim)  —  UMAP")

    # shared legend
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    out1 = out_dir / "latent_class_panel.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # ---- Figure 2: k-NN score overlays (t-SNE + UMAP for Model A) ---------
    print("Plotting k-NN score overlays...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Model A (real, 10M ep75) — k-NN anomaly score in latent space\n"
        "Higher score = more anomalous",
        fontsize=12
    )

    scatter_score(axes[0], tsne_A, knn_scores, "t-SNE")
    scatter_score(axes[1], umap_A, knn_scores, "UMAP")

    # overlay signal contours on score panels
    for ax, proj in zip(axes, [tsne_A, umap_A]):
        for sig_cls, color, label in [
            (1, "#ff3333", "2-prong"), (2, "#3399ff", "3-prong")
        ]:
            mask = labels == sig_cls
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       facecolors="none", edgecolors=color,
                       s=16, linewidths=0.6, alpha=0.6,
                       zorder=5, label=label)
        ax.legend(fontsize=9, title="Signal", framealpha=0.8)

    plt.tight_layout()
    out2 = out_dir / "latent_score_overlay.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")

    # ---- Figure 3: Model A t-SNE, large single panel for paper ------------
    print("Plotting paper-ready single panel...")
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, tsne_A, labels,
                    "Model A (real, 10M jets, ep75) — t-SNE of latent space")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, lbls, fontsize=10, framealpha=0.9,
              loc="upper right")
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    plt.tight_layout()
    out3 = out_dir / "latent_tsne_modelA_paper.png"
    fig.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out3}")

    print(f"\nAll outputs in {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_A",  default="data/checkpoints/phase3/model_A_10M_v4/encoder_epoch075.pt")
    p.add_argument("--ckpt_B2", default="data/checkpoints/phase2/model_B2/encoder_epoch050.pt")
    p.add_argument("--lhco_bg", default="data/lhco/lhco_bg_jets.h5")
    p.add_argument("--lhco_2p", default="data/lhco/events_anomalydetection_v2.h5")
    p.add_argument("--lhco_3p", default="data/lhco/events_anomalydetection_Z_XY_qqq.h5")
    p.add_argument("--n_bg",    type=int, default=8000,
                   help="Background jets to visualise")
    p.add_argument("--n_sig",   type=int, default=2000,
                   help="Signal jets per class to visualise")
    p.add_argument("--perplexity",     type=int,   default=40)
    p.add_argument("--umap_neighbors", type=int,   default=30)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--out_dir", default="data/results/latent_viz")
    args = p.parse_args()
    main(args)

"""
paper_figure_5_tsne.py — Figure 5: 2-panel t-SNE of Model A vs B2 latent spaces.
Requires GPU. Encodes ~20K bg + ~2K 2-prong signal jets, runs t-SNE, plots.
Output: data/results/paper_figures/fig5_tsne.{pdf,png}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

from src.data.preprocessed_loader import PreprocessedJetDatasetCached
from src.data.lhco_loader import LHCOJetDataset
from src.models.encoder import JetEncoder

# ── Config ───────────────────────────────────────────────────────────────────
CKPT_A  = "data/checkpoints/phase3/model_A_10M_v4/encoder_epoch075.pt"
CKPT_B2 = "data/checkpoints/phase2/model_B2_75ep/encoder_epoch075.pt"
LHCO_BG  = "data/lhco/lhco_bg_jets.h5"
LHCO_SIG = "data/lhco/events_anomalydetection_v2.h5"

N_BG  = 20_000
N_SIG = 2_000
TSNE_PERP = 30
ENCODE_BS  = 1024
SEED = 42

OUT = Path("data/results/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_model(ckpt_path, device):
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


@torch.no_grad()
def encode(model, xs, masks, device):
    reps = []
    for i in range(0, len(xs), ENCODE_BS):
        xb = xs[i:i+ENCODE_BS].to(device)
        mb = masks[i:i+ENCODE_BS].to(device)
        reps.append(model.backbone(xb, mb).cpu())
    return torch.cat(reps, dim=0).numpy()


def load_lhco_jets(path, n, signal_label, rng):
    """Load jets from LHCO file, return (xs, masks)."""
    ds = LHCOJetDataset(path, n_jets=n * 4, signal=signal_label)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    xs    = torch.stack([ds[i][0] for i in idx])
    masks = torch.stack([ds[i][1] for i in idx])
    return xs, masks


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rng = np.random.default_rng(SEED)

    # ── Load jets ────────────────────────────────────────────────────────────
    print(f"Loading {N_BG} background jets...")
    # Background: use preprocessed LHCO bg (same as eval pipeline)
    bg_ds = PreprocessedJetDatasetCached(LHCO_BG, max_jets=N_BG * 2)
    bg_idx = rng.choice(len(bg_ds), size=min(N_BG, len(bg_ds)), replace=False)
    bg_xs    = torch.stack([bg_ds[i][0] for i in bg_idx])
    bg_masks = torch.stack([bg_ds[i][1] for i in bg_idx])

    print(f"Loading {N_SIG} 2-prong signal jets...")
    sig_ds = LHCOJetDataset(LHCO_SIG, is_signal=True, max_jets=N_SIG * 3)
    sig_idx = rng.choice(len(sig_ds), size=min(N_SIG, len(sig_ds)), replace=False)
    sig_xs    = torch.stack([sig_ds[i][0] for i in sig_idx])
    sig_masks = torch.stack([sig_ds[i][1] for i in sig_idx])

    # Combine: bg first, then signal
    all_xs    = torch.cat([bg_xs, sig_xs], dim=0)
    all_masks = torch.cat([bg_masks, sig_masks], dim=0)
    labels = np.array([0] * len(bg_idx) + [1] * len(sig_idx))

    # Slice to 4 features if model was trained on 4 features only
    def maybe_slice(model, xs):
        input_dim = model.input_proj.weight.shape[1]
        return xs[:, :, :input_dim]

    # ── Encode and t-SNE for each model ──────────────────────────────────────
    results = {}
    for name, ckpt in [("Model A\n(Real CMS, 10M)", CKPT_A),
                        ("Model B2\n(Simulation, 1M)", CKPT_B2)]:
        print(f"\nEncoding {name.replace(chr(10), ' ')}...")
        model = load_model(ckpt, device)
        xs_in = maybe_slice(model, all_xs)
        emb = encode(model, xs_in, all_masks, device)
        print(f"  Embedding shape: {emb.shape}  — running t-SNE (perplexity={TSNE_PERP})...")
        tsne = TSNE(n_components=2, perplexity=TSNE_PERP, random_state=SEED,
                    max_iter=1000, init="pca", learning_rate="auto")
        proj = tsne.fit_transform(emb)
        results[name] = proj
        del model

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6))
    fig.subplots_adjust(wspace=0.08)

    for ax, (name, proj) in zip(axes, results.items()):
        # Background — gray, very transparent, small points
        ax.scatter(proj[labels == 0, 0], proj[labels == 0, 1],
                   c="gray", alpha=0.06, s=1.5, linewidths=0, rasterized=True)
        # Signal — red, semi-transparent
        ax.scatter(proj[labels == 1, 0], proj[labels == 1, 1],
                   c="#d6604d", alpha=0.40, s=4, linewidths=0, rasterized=True,
                   label="2-prong signal")

        ax.set_title(name, fontsize=8.5, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=5, label="Background", alpha=0.5),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d6604d",
               markersize=5, label="2-prong signal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5_tsne.{ext}")
    plt.close(fig)
    print(f"\nfig5_tsne saved to {OUT}/")


if __name__ == "__main__":
    main()

"""
substructure_overlay.py — Overlay jet substructure variables on the latent space.

Encodes LHCO background + signal jets through Model A, projects to 2D with t-SNE,
then colours each point by a substructure variable instead of jet class.

Substructure variables computed without FastJet (from constituent (pT, eta, phi)):
  - jet_mass:    invariant mass of all constituents (massless approx)
  - jet_pt:      scalar sum of constituent pT
  - multiplicity: number of valid constituents
  - girth:       pT-weighted mean ΔR from jet axis
  - pt_frac_lead: leading constituent pT / total pT  (hard-core fraction)
  - pt_dispersion: std(pT_i) / mean(pT_i)  (how spread the pT distribution is)

These are physically meaningful proxies for:
  - jet_mass → invariant mass (distinguishes W/top from QCD)
  - girth → jet width / angularity
  - pt_frac_lead → 1-prong-ness (large = one hard core, small = diffuse)
  - multiplicity → hadron activity / jet complexity

Usage:
    python scripts/eval/substructure_overlay.py \\
        --ckpt_A  data/checkpoints/phase3/model_A_10M_v4/encoder_epoch100.pt \\
        --out_dir data/results/substructure_overlay
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

from src.data.lhco_loader import LHCOJetDataset, _extract_leading_jet, _to_relative
from src.data.preprocessed_loader import PreprocessedJetDatasetCached
from src.models.encoder import JetEncoder
from src.data.aspen_loader import N_FEATURES


# ---------------------------------------------------------------------------
# Substructure variables from (pT, eta, phi) constituents
# ---------------------------------------------------------------------------

def compute_substructure(consts_ptetaphi):
    """
    consts_ptetaphi: (N, 3) array of valid constituents.
    Returns dict of substructure variables.
    """
    pT  = consts_ptetaphi[:, 0]
    eta = consts_ptetaphi[:, 1]
    phi = consts_ptetaphi[:, 2]
    N = len(pT)

    # Jet axis (pT-weighted centroid)
    jet_pT = pT.sum() + 1e-8
    jet_eta = (pT * eta).sum() / jet_pT
    jet_phi_s = np.sum(pT * np.sin(phi)) / jet_pT
    jet_phi_c = np.sum(pT * np.cos(phi)) / jet_pT
    jet_phi = np.arctan2(jet_phi_s, jet_phi_c)

    # ΔR from jet axis per constituent
    deta = eta - jet_eta
    dphi = ((phi - jet_phi) + np.pi) % (2 * np.pi) - np.pi
    dr = np.sqrt(deta**2 + dphi**2)

    # Jet mass (massless 4-momenta)
    E  = pT * np.cosh(eta)
    px = pT * np.cos(phi)
    py = pT * np.sin(phi)
    pz = pT * np.sinh(eta)
    m2 = (E.sum())**2 - (px.sum())**2 - (py.sum())**2 - (pz.sum())**2
    mass = float(np.sqrt(max(0.0, m2)))

    return {
        "jet_mass":       mass,
        "jet_pt":         float(jet_pT),
        "multiplicity":   int(N),
        "girth":          float((pT * dr).sum() / jet_pT),
        "pt_frac_lead":   float(pT.max() / jet_pT),
        "pt_dispersion":  float(pT.std() / (pT.mean() + 1e-8)),
    }


# ---------------------------------------------------------------------------
# Data loading (with substructure)
# ---------------------------------------------------------------------------

def load_with_substructure(hdf5_path, is_signal, n, seed=42, jet_radius=0.8):
    import pandas as pd
    df = pd.read_hdf(hdf5_path, key="df")
    arr = df.values
    labels = arr[:, -1].astype(int)
    particles = arr[:, :-1].reshape(-1, 700, 3).astype(np.float32)
    sel = particles[labels == int(is_signal)]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sel), size=min(n, len(sel)), replace=False)

    xs_list, masks_list, structs = [], [], []
    max_const = 50

    for i in idx:
        consts = _extract_leading_jet(sel[i], jet_radius)
        structs.append(compute_substructure(consts))

        feats = _to_relative(consts)
        N = min(len(feats), max_const)
        out = np.zeros((max_const, N_FEATURES), dtype=np.float32)
        out[:N] = feats[:N]
        mask = np.ones(max_const, dtype=bool)
        mask[:N] = False
        xs_list.append(out)
        masks_list.append(mask)

    xs    = torch.from_numpy(np.stack(xs_list))
    masks = torch.from_numpy(np.stack(masks_list))
    return xs, masks, structs


def load_bg_with_substructure(hdf5_path, n, seed=42):
    ds = PreprocessedJetDatasetCached(hdf5_path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    xs    = torch.from_numpy(ds._x[idx])
    masks = torch.from_numpy(ds._mask[idx])
    # For preprocessed bg, load raw to compute substructure
    return xs, masks, None   # substructure computed separately if needed


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    a = ckpt["args"]
    model = JetEncoder(
        input_dim=state["input_proj.weight"].shape[1],
        d_model=a.get("d_model", 128), nhead=a.get("nhead", 8),
        num_layers=a.get("num_layers", 4), latent_dim=a.get("latent_dim", 256),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def encode(model, xs, masks, device, batch_size=1024, feature_cols=None):
    zs = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            xb = xs[i:i+batch_size].to(device)
            mb = masks[i:i+batch_size].to(device)
            if feature_cols is not None:
                xb = xb[:, :, feature_cols]
            zs.append(model.backbone(xb, mb).cpu())
    return torch.cat(zs).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    print(f"Device: {device}")

    # Load all jets WITH substructure (must use raw LHCO for substructure)
    print(f"\nLoading {args.n_bg} background jets (raw LHCO for substructure)...")
    bg_xs, bg_masks, bg_structs = load_with_substructure(
        args.lhco_bg_raw, is_signal=False, n=args.n_bg, seed=args.seed)

    print(f"Loading {args.n_sig} 2-prong signal jets...")
    p2_xs, p2_masks, p2_structs = load_with_substructure(
        args.lhco_2p, is_signal=True, n=args.n_sig, seed=args.seed)

    print(f"Loading {args.n_sig} 3-prong signal jets...")
    p3_xs, p3_masks, p3_structs = load_with_substructure(
        args.lhco_3p, is_signal=True, n=args.n_sig, seed=args.seed)

    # Concatenate
    all_xs    = torch.cat([bg_xs,    p2_xs,    p3_xs])
    all_masks = torch.cat([bg_masks, p2_masks, p3_masks])
    all_structs = bg_structs + p2_structs + p3_structs
    jet_class = np.array([0]*len(bg_xs) + [1]*len(p2_xs) + [2]*len(p3_xs))
    print(f"Total jets: {len(all_xs):,}")

    # Extract substructure arrays
    var_names = ["jet_mass", "jet_pt", "multiplicity", "girth", "pt_frac_lead", "pt_dispersion"]
    var_arrays = {v: np.array([s[v] for s in all_structs]) for v in var_names}

    # Load model and encode
    print(f"\nLoading model from {args.ckpt_A}...")
    model = load_model(args.ckpt_A, device)
    feat = args.feature_cols

    print("Encoding all jets...")
    z = encode(model, all_xs, all_masks, device, feature_cols=feat)

    # t-SNE
    print("Running t-SNE...")
    z_scaled = StandardScaler().fit_transform(z)
    proj = TSNE(n_components=2, perplexity=args.perplexity,
                random_state=args.seed, n_jobs=-1).fit_transform(z_scaled)

    # --- Plotting ---
    CLASS_COLORS = {0: "#aaaaaa", 1: "#e6504a", 2: "#4a90e6"}
    CLASS_LABELS = {0: "Background", 1: "2-prong signal", 2: "3-prong signal"}

    # Panel 1: class-coloured t-SNE
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in [0, 1, 2]:
        m = jet_class == cls
        ax.scatter(proj[m, 0], proj[m, 1], c=CLASS_COLORS[cls],
                   s=4 if cls == 0 else 10, alpha=0.25 if cls == 0 else 0.7,
                   label=f"{CLASS_LABELS[cls]} (n={m.sum():,})", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9); ax.set_title("t-SNE — jet class", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "tsne_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Panel 2: substructure variable overlays (2×3 grid)
    var_labels = {
        "jet_mass":       "Jet mass [GeV]",
        "jet_pt":         "Jet pT [GeV]",
        "multiplicity":   "Constituent multiplicity",
        "girth":          "Jet girth (pT-weighted mean ΔR)",
        "pt_frac_lead":   "Leading constituent pT fraction",
        "pt_dispersion":  "pT dispersion (std/mean)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Substructure variables in latent space\n{Path(args.ckpt_A).parent.name}",
                 fontsize=12)

    for ax, var in zip(axes.flat, var_names):
        vals = var_arrays[var]
        vmin, vmax = np.percentile(vals, [2, 98])
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=vals,
                        cmap="plasma", vmin=vmin, vmax=vmax,
                        s=3, alpha=0.35, rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.85)
        ax.set_title(var_labels[var], fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_dir / "tsne_substructure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")
    print("  tsne_class.png")
    print("  tsne_substructure.png")

    # Summary stats per class
    print("\nSubstructure summary (mean ± std):")
    print(f"  {'Variable':<22}  {'Background':>20}  {'2-prong signal':>20}  {'3-prong signal':>20}")
    print("  " + "-"*88)
    for var in var_names:
        vals = var_arrays[var]
        row = f"  {var:<22}"
        for cls in [0, 1, 2]:
            m = jet_class == cls
            row += f"  {vals[m].mean():>8.2f} ± {vals[m].std():>6.2f}"
        print(row)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_A",   default="data/checkpoints/phase3/model_A_10M_v4/encoder_epoch075.pt")
    p.add_argument("--lhco_bg_raw", default="data/lhco/events_anomalydetection_v2.h5",
                   help="Raw LHCO HDF5 needed for jet mass/substructure computation")
    p.add_argument("--lhco_2p",  default="data/lhco/events_anomalydetection_v2.h5")
    p.add_argument("--lhco_3p",  default="data/lhco/events_anomalydetection_Z_XY_qqq.h5")
    p.add_argument("--n_bg",     type=int, default=6000)
    p.add_argument("--n_sig",    type=int, default=2000)
    p.add_argument("--perplexity", type=int, default=40)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--feature_cols", type=int, nargs="+", default=None)
    p.add_argument("--out_dir",  default="data/results/substructure_overlay")
    main(p.parse_args())

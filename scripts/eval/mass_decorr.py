"""
mass_decorr.py — Mass decorrelation and background sculpting check.

A well-behaved anomaly score should be approximately independent of jet mass.
If the score correlates with mass, applying a threshold sculpts the background
mass distribution, creating a fake bump — useless (or harmful) for bump hunts.

Tests:
  1. Spearman correlation between anomaly score and jet mass (should be ~0)
  2. Mean anomaly score vs jet mass bin (should be flat)
  3. Background mass distribution at three score thresholds (should not shift)
  4. Same checks for jet pT (score should not trivially track energy scale)

Jet mass is computed from LHCO constituent (pT, eta, phi) via massless 4-momenta:
  E_i = pT_i * cosh(eta_i),  pz_i = pT_i * sinh(eta_i)
  m_jet = sqrt(max(0, (ΣE)^2 - (Σpx)^2 - (Σpy)^2 - (Σpz)^2))

Usage:
    python scripts/eval/mass_decorr.py \\
        --ckpt  data/checkpoints/phase3/model_A_10M_v4/encoder_epoch100.pt \\
        --out_dir data/results/mass_decorr
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
from scipy.stats import spearmanr
import torch

from src.data.lhco_loader import LHCOJetDataset, _extract_leading_jet
from src.models.encoder import JetEncoder
from src.anomaly.knn_scorer import KNNAnomalyScorer


# ---------------------------------------------------------------------------
# Jet mass from (pT, eta, phi) constituents — massless particle approximation
# ---------------------------------------------------------------------------

def jet_mass(consts_ptetaphi):
    """
    consts_ptetaphi: (N, 3) array of (pT, eta, phi) for valid constituents.
    Returns scalar jet invariant mass.
    """
    pT  = consts_ptetaphi[:, 0]
    eta = consts_ptetaphi[:, 1]
    phi = consts_ptetaphi[:, 2]

    E   = pT * np.cosh(eta)
    px  = pT * np.cos(phi)
    py  = pT * np.sin(phi)
    pz  = pT * np.sinh(eta)

    sE, spx, spy, spz = E.sum(), px.sum(), py.sum(), pz.sum()
    m2 = sE**2 - spx**2 - spy**2 - spz**2
    return float(np.sqrt(max(0.0, m2)))


def jet_pt(consts_ptetaphi):
    return float(consts_ptetaphi[:, 0].sum())


# ---------------------------------------------------------------------------
# Load LHCO jets with raw kinematics (before relativizing)
# ---------------------------------------------------------------------------

def load_lhco_raw(hdf5_path, is_signal, max_jets, jet_radius=0.8):
    """
    Returns (xs, masks, masses, pts) — xs/masks for encoding, masses/pts for decorr.
    """
    import pandas as pd
    from src.data.aspen_loader import N_FEATURES

    df = pd.read_hdf(hdf5_path, key="df")
    arr = df.values
    labels = arr[:, -1].astype(int)
    particles = arr[:, :-1].reshape(-1, 700, 3).astype(np.float32)

    sel = particles[labels == int(is_signal)]
    if max_jets:
        sel = sel[:max_jets]

    xs_list, masks_list, masses, pts = [], [], [], []
    from src.data.lhco_loader import _to_relative
    max_const = 50

    for i in range(len(sel)):
        consts = _extract_leading_jet(sel[i], jet_radius)
        masses.append(jet_mass(consts))
        pts.append(jet_pt(consts))

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
    return xs, masks, np.array(masses), np.array(pts)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device, feature_cols=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    input_dim = state["input_proj.weight"].shape[1]
    a = ckpt["args"]
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
    print(f"Device: {device}")

    feat = args.feature_cols

    print(f"\nLoading {args.n_jets:,} background jets from {args.lhco_bg}...")
    bg_xs, bg_masks, bg_masses, bg_pts = load_lhco_raw(
        args.lhco_bg, is_signal=False, max_jets=args.n_jets)
    print(f"  Jet mass: mean={bg_masses.mean():.1f} GeV  std={bg_masses.std():.1f} GeV")
    print(f"  Jet pT:   mean={bg_pts.mean():.1f} GeV    std={bg_pts.std():.1f} GeV")

    print(f"\nLoading model from {args.ckpt}...")
    model = load_model(args.ckpt, device, feat)

    print("Encoding background...")
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(bg_xs))
    n_ref = args.n_ref
    ref_z   = encode(model, bg_xs[perm[:n_ref]],      bg_masks[perm[:n_ref]],      device, feature_cols=feat)
    test_z  = encode(model, bg_xs[perm[n_ref:]],      bg_masks[perm[n_ref:]],      device, feature_cols=feat)
    test_masses = bg_masses[perm[n_ref:]]
    test_pts    = bg_pts[perm[n_ref:]]

    print("Computing k-NN anomaly scores...")
    scorer = KNNAnomalyScorer(k=10)
    scorer.fit(ref_z)
    scores = scorer.score(test_z)

    # --- 1. Spearman correlations ---
    rho_mass, p_mass = spearmanr(scores, test_masses)
    rho_pt,   p_pt   = spearmanr(scores, test_pts)
    print(f"\nSpearman correlation:")
    print(f"  score vs jet mass: rho={rho_mass:+.4f}  p={p_mass:.2e}")
    print(f"  score vs jet pT:   rho={rho_pt:+.4f}  p={p_pt:.2e}")

    if abs(rho_mass) < 0.1:
        print("  PASS: mass correlation is negligible (<0.10)")
    else:
        print(f"  WARNING: mass correlation {rho_mass:.3f} — may cause sculpting")

    # --- 2. Mean score vs mass bins ---
    mass_bins = np.percentile(test_masses, np.linspace(0, 100, args.n_bins + 1))
    bin_idx = np.digitize(test_masses, mass_bins[1:-1])
    bin_means = [scores[bin_idx == b].mean() for b in range(args.n_bins)]
    bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    # --- 3. Mass distributions at score thresholds ---
    q50 = np.percentile(scores, 50)
    q90 = np.percentile(scores, 90)
    q99 = np.percentile(scores, 99)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Mass Decorrelation Check\n{Path(args.ckpt).parent.name}", fontsize=12)

    # Panel 1: score vs mass scatter
    ax = axes[0, 0]
    ax.hexbin(test_masses, scores, gridsize=60, cmap="Blues", mincnt=1)
    ax.set_xlabel("Jet mass [GeV]")
    ax.set_ylabel("k-NN anomaly score")
    ax.set_title(f"Score vs mass  (ρ={rho_mass:+.3f}, p={p_mass:.1e})")
    ax.grid(True, alpha=0.3)

    # Panel 2: mean score per mass bin
    ax = axes[0, 1]
    ax.plot(bin_centers, bin_means, "o-", color="#1f77b4", lw=2)
    ax.axhline(np.mean(scores), color="gray", ls="--", label="Overall mean")
    ax.set_xlabel("Jet mass [GeV]")
    ax.set_ylabel("Mean k-NN score")
    ax.set_title("Mean anomaly score per mass bin")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: mass distributions at thresholds
    ax = axes[1, 0]
    bins = np.linspace(0, np.percentile(test_masses, 99), 60)
    ax.hist(test_masses,                        bins=bins, density=True, alpha=0.4, label="All background", color="gray")
    ax.hist(test_masses[scores > q50],          bins=bins, density=True, alpha=0.5, label="Top 50% score", color="#ff7f0e")
    ax.hist(test_masses[scores > q90],          bins=bins, density=True, alpha=0.6, label="Top 10% score", color="#d62728")
    ax.hist(test_masses[scores > q99],          bins=bins, density=True, alpha=0.8, label="Top 1% score",  color="#9467bd")
    ax.set_xlabel("Jet mass [GeV]")
    ax.set_ylabel("Normalised density")
    ax.set_title("Background sculpting check")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: score vs pT scatter
    ax = axes[1, 1]
    ax.hexbin(test_pts, scores, gridsize=60, cmap="Greens", mincnt=1)
    ax.set_xlabel("Jet pT [GeV]")
    ax.set_ylabel("k-NN anomaly score")
    ax.set_title(f"Score vs pT  (ρ={rho_pt:+.3f}, p={p_pt:.1e})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "mass_decorr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out_dir}/mass_decorr.png")

    # --- Save results ---
    import json
    results = {
        "ckpt": str(args.ckpt),
        "n_bg": len(test_masses),
        "n_ref": n_ref,
        "spearman_mass": {"rho": float(rho_mass), "p": float(p_mass)},
        "spearman_pt":   {"rho": float(rho_pt),   "p": float(p_pt)},
        "mass_bins": {"centers": bin_centers.tolist(), "mean_scores": bin_means},
        "thresholds": {"q50": float(q50), "q90": float(q90), "q99": float(q99)},
    }
    with open(out_dir / "mass_decorr.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir}/mass_decorr.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     default="data/checkpoints/phase3/model_A_10M_v4/encoder_epoch075.pt")
    p.add_argument("--lhco_bg", default="data/lhco/events_anomalydetection_v2.h5",
                   help="Raw LHCO HDF5 (not preprocessed) — needed for jet mass computation")
    p.add_argument("--n_jets",  type=int, default=50000)
    p.add_argument("--n_ref",   type=int, default=20000)
    p.add_argument("--n_bins",  type=int, default=20)
    p.add_argument("--feature_cols", type=int, nargs="+", default=None)
    p.add_argument("--out_dir", default="data/results/mass_decorr")
    main(p.parse_args())

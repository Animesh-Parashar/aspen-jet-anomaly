"""
signal_injection.py — AUC vs signal fraction (S/B) curve.

Real LHC anomaly searches operate at S/B << 1%. The default LHCO eval
uses ~9% signal fraction — unrealistically high. This script sweeps S/B
from 0.01% to 10% and reports AUC at each point with error bars.

If AUC drops to ~0.5 below 1% signal fraction, the model is not useful
for realistic searches. If AUC holds, that is a strong result to report.

Usage:
    python scripts/eval/signal_injection.py \\
        --ckpt data/checkpoints/phase3/model_A_10M_v4/encoder_epoch100.pt \\
        --out_dir data/results/signal_injection
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score

from src.data.lhco_loader import LHCOJetDataset
from src.data.preprocessed_loader import PreprocessedJetDatasetCached
from src.models.encoder import JetEncoder
from src.anomaly.knn_scorer import KNNAnomalyScorer


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    a = ckpt["args"]
    model = JetEncoder(
        input_dim=state["input_proj.weight"].shape[1],
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


def auc_at_fraction(ref_z, bg_z, sig_z, sig_frac, n_test_bg, seed):
    """Compute AUC with signal fraction sig_frac in the test set."""
    rng = np.random.default_rng(seed)

    n_sig = max(1, int(n_test_bg * sig_frac / (1.0 - sig_frac)))
    n_sig = min(n_sig, len(sig_z))

    sig_idx = rng.choice(len(sig_z), size=n_sig, replace=False)
    bg_idx  = rng.choice(len(bg_z),  size=n_test_bg, replace=False)

    test_z  = np.concatenate([bg_z[bg_idx], sig_z[sig_idx]])
    labels  = np.array([0] * n_test_bg + [1] * n_sig)

    scorer = KNNAnomalyScorer(k=10)
    scorer.fit(ref_z)
    scores = scorer.score(test_z)
    return float(roc_auc_score(labels, scores)), n_sig, n_test_bg


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feat = args.feature_cols

    # --- Load data ---
    print("\nLoading LHCO background (preprocessed)...")
    ds_bg = PreprocessedJetDatasetCached(args.lhco_bg)
    bg_xs    = torch.from_numpy(ds_bg._x)
    bg_masks = torch.from_numpy(ds_bg._mask)
    print(f"  {len(bg_xs):,} background jets")

    print("Loading 2-prong signal...")
    ds_sig = LHCOJetDataset(args.lhco_2p, is_signal=True)
    sig_xs    = torch.stack([ds_sig[i][0] for i in range(len(ds_sig))])
    sig_masks = torch.stack([ds_sig[i][1] for i in range(len(ds_sig))])
    print(f"  {len(sig_xs):,} signal jets")

    # --- Load model and encode everything ---
    print(f"\nLoading model from {args.ckpt}...")
    model = load_model(args.ckpt, device)

    print("Encoding background (reference set)...")
    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(len(bg_xs))
    ref_xs    = bg_xs[perm[:args.n_ref]]
    ref_masks = bg_masks[perm[:args.n_ref]]
    test_bg_xs    = bg_xs[perm[args.n_ref:args.n_ref + args.n_test_bg]]
    test_bg_masks = bg_masks[perm[args.n_ref:args.n_ref + args.n_test_bg]]

    ref_z    = encode(model, ref_xs,      ref_masks,      device, feature_cols=feat)
    test_bg_z = encode(model, test_bg_xs, test_bg_masks,  device, feature_cols=feat)
    sig_z    = encode(model, sig_xs,      sig_masks,      device, feature_cols=feat)
    print("  Encoding done.")

    # --- Signal fraction sweep ---
    fractions = args.fractions  # e.g. [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10]
    results = []

    print(f"\nSignal fraction sweep ({args.n_seeds} seeds each):")
    print(f"  {'S/B fraction':>14}  {'n_sig':>7}  {'AUC':>8}  {'std':>8}")
    print("  " + "-"*46)

    for frac in fractions:
        aucs = []
        for seed in range(args.n_seeds):
            auc, n_sig, n_bg = auc_at_fraction(
                ref_z, test_bg_z, sig_z, frac, args.n_test_bg, seed)
            aucs.append(auc)
        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)
        results.append({
            "fraction": frac,
            "n_sig": n_sig,
            "n_bg": n_bg,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "aucs": aucs,
        })
        pct = frac * 100
        print(f"  {pct:>13.3f}%  {n_sig:>7d}  {mean_auc:.4f}  ±{std_auc:.4f}")

    # --- Plot ---
    fracs   = [r["fraction"] for r in results]
    aucs    = [r["mean_auc"] for r in results]
    stds    = [r["std_auc"]  for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(fracs, aucs, yerr=stds, fmt="o-", capsize=5,
                lw=2, ms=7, color="#1f77b4", label="Model A (real contrastive)")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random (AUC=0.5)")
    ax.set_xscale("log")
    ax.set_xlabel("Signal fraction S/(S+B)", fontsize=12)
    ax.set_ylabel("2-prong AUC (k-NN)", fontsize=12)
    ax.set_title("Anomaly detection AUC vs signal injection rate\n"
                 f"{Path(args.ckpt).parent.name}", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    # Annotate realistic search regime
    ax.axvspan(1e-4, 1e-2, alpha=0.05, color="red", label="Realistic LHC S/B range")
    ax.text(3e-4, 0.502, "Realistic LHC\nS/B range", fontsize=8, color="red", va="bottom")

    ax.set_xlim(min(fracs) * 0.5, max(fracs) * 2)
    ax.set_ylim(0.48, max(aucs) + max(stds) + 0.03)

    plt.tight_layout()
    fig.savefig(out_dir / "signal_injection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out_dir}/signal_injection.png")

    with open(out_dir / "signal_injection.json", "w") as f:
        json.dump({"ckpt": str(args.ckpt), "results": results}, f, indent=2)
    print(f"Results saved to {out_dir}/signal_injection.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     default="data/checkpoints/phase3/model_A_10M_v4/encoder_epoch075.pt")
    p.add_argument("--lhco_bg", default="data/lhco/lhco_bg_jets.h5")
    p.add_argument("--lhco_2p", default="data/lhco/events_anomalydetection_v2.h5")
    p.add_argument("--n_ref",       type=int, default=20000)
    p.add_argument("--n_test_bg",   type=int, default=50000,
                   help="Fixed background size in test set")
    p.add_argument("--n_seeds",     type=int, default=5)
    p.add_argument("--fractions",   type=float, nargs="+",
                   default=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10],
                   help="Signal fractions to test (S/(S+B))")
    p.add_argument("--feature_cols", type=int, nargs="+", default=None)
    p.add_argument("--out_dir",  default="data/results/signal_injection")
    main(p.parse_args())

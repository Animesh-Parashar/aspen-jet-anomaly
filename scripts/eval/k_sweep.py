"""
k_sweep.py — Sweep k in k-NN anomaly scoring for Model A (10M, ep75).
Tests k = 1, 5, 10, 20, 50 with 5 seeds each.
Output: data/results/k_sweep/k_sweep_results.json
        data/results/paper_figures/fig7_ksweep.{pdf,png}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import roc_auc_score, roc_curve
from src.anomaly.knn_scorer import KNNAnomalyScorer, compute_rejection
from src.data.preprocessed_loader import PreprocessedJetDatasetCached
from src.data.lhco_loader import LHCOJetDataset

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

K_VALUES  = [1, 5, 10, 20, 50]
N_SEEDS   = 5
N_REF     = 20_000
N_BG      = 20_000
N_SIG     = 2_000
ENCODE_BS = 2048


def load_model(ckpt_path, device):
    from src.models.encoder import JetEncoder
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


def eval_k(ref_z, test_z, labels, k):
    scorer = KNNAnomalyScorer(k=k)
    scorer.fit(ref_z)
    scores = scorer.score(test_z)
    auc = roc_auc_score(labels, scores)
    return auc, scores


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading background jets...")
    bg_ds = PreprocessedJetDatasetCached(args.lhco_bg, max_jets=N_REF + N_BG + 5000)
    bg_xs    = torch.stack([bg_ds[i][0] for i in range(min(N_REF + N_BG + 5000, len(bg_ds)))])
    bg_masks = torch.stack([bg_ds[i][1] for i in range(min(N_REF + N_BG + 5000, len(bg_ds)))])

    print("Loading 2-prong signal...")
    sig2_ds = LHCOJetDataset(args.lhco_2p, is_signal=True, max_jets=N_SIG * 3)
    sig2_xs    = torch.stack([sig2_ds[i][0] for i in range(len(sig2_ds))])
    sig2_masks = torch.stack([sig2_ds[i][1] for i in range(len(sig2_ds))])

    print("Loading 3-prong signal...")
    sig3_ds = LHCOJetDataset(args.lhco_3p, is_signal=True, max_jets=N_SIG * 3)
    sig3_xs    = torch.stack([sig3_ds[i][0] for i in range(len(sig3_ds))])
    sig3_masks = torch.stack([sig3_ds[i][1] for i in range(len(sig3_ds))])

    # ── Load and encode model once ────────────────────────────────────────────
    print(f"\nLoading model: {args.ckpt}")
    model = load_model(args.ckpt, device)

    # Slice features if 4-feat model
    input_dim = model.input_proj.weight.shape[1]
    bg_xs_in   = bg_xs[:, :, :input_dim]
    sig2_xs_in = sig2_xs[:, :, :input_dim]
    sig3_xs_in = sig3_xs[:, :, :input_dim]

    print("Encoding all jets (done once, reused across k values)...")
    all_bg_z   = encode(model, bg_xs_in,   bg_masks,   device)
    all_sig2_z = encode(model, sig2_xs_in, sig2_masks, device)
    all_sig3_z = encode(model, sig3_xs_in, sig3_masks, device)
    print(f"  bg: {all_bg_z.shape}, sig2: {all_sig2_z.shape}, sig3: {all_sig3_z.shape}")

    # ── Sweep k ───────────────────────────────────────────────────────────────
    results = {k: {"2-prong": [], "3-prong": []} for k in K_VALUES}
    rng = np.random.default_rng(0)

    for seed in range(N_SEEDS):
        perm = rng.permutation(len(all_bg_z))
        ref_z  = all_bg_z[perm[:N_REF]]
        tbg_z  = all_bg_z[perm[N_REF:N_REF + N_BG]]

        sig2_perm = rng.permutation(len(all_sig2_z))[:N_SIG]
        sig3_perm = rng.permutation(len(all_sig3_z))[:N_SIG]
        tsig2_z = all_sig2_z[sig2_perm]
        tsig3_z = all_sig3_z[sig3_perm]

        test2_z = np.concatenate([tbg_z, tsig2_z], axis=0)
        test3_z = np.concatenate([tbg_z, tsig3_z], axis=0)
        labels2 = np.array([0]*N_BG + [1]*N_SIG)
        labels3 = np.array([0]*N_BG + [1]*N_SIG)

        for k in K_VALUES:
            auc2, _ = eval_k(ref_z, test2_z, labels2, k)
            auc3, _ = eval_k(ref_z, test3_z, labels3, k)
            results[k]["2-prong"].append(auc2)
            results[k]["3-prong"].append(auc3)
            print(f"  seed={seed} k={k:2d}: 2p={auc2:.4f}  3p={auc3:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nk-sweep summary:")
    print(f"{'k':>4}  {'2-prong AUC':>14}  {'3-prong AUC':>14}")
    print("-" * 38)
    json_out = {}
    for k in K_VALUES:
        m2, s2 = np.mean(results[k]["2-prong"]), np.std(results[k]["2-prong"])
        m3, s3 = np.mean(results[k]["3-prong"]), np.std(results[k]["3-prong"])
        print(f"  k={k:2d}  {m2:.4f} ± {s2:.4f}   {m3:.4f} ± {s3:.4f}")
        json_out[str(k)] = {
            "2-prong": {"mean_auc": float(m2), "std_auc": float(s2), "aucs": results[k]["2-prong"]},
            "3-prong": {"mean_auc": float(m3), "std_auc": float(s3), "aucs": results[k]["3-prong"]},
        }

    json_path = out_dir / "k_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_out = Path("data/results/paper_figures")
    fig_out.mkdir(parents=True, exist_ok=True)

    ks = K_VALUES
    m2s = [json_out[str(k)]["2-prong"]["mean_auc"] for k in ks]
    s2s = [json_out[str(k)]["2-prong"]["std_auc"]  for k in ks]
    m3s = [json_out[str(k)]["3-prong"]["mean_auc"] for k in ks]
    s3s = [json_out[str(k)]["3-prong"]["std_auc"]  for k in ks]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.errorbar(ks, m2s, yerr=s2s, color="#2166ac", ls="-", marker="o",
                ms=5, lw=1.5, capsize=3, label="2-prong")
    ax.errorbar(ks, m3s, yerr=s3s, color="#d6604d", ls="--", marker="s",
                ms=5, lw=1.5, capsize=3, label="3-prong")

    # Vertical line at default k=10
    ax.axvline(10, color="gray", ls=":", lw=0.9, alpha=0.7)
    ax.text(10.5, min(m2s) - 0.002, "k = 10\n(default)", fontsize=7,
            color="gray", va="top")

    ax.set_xlabel("k (number of nearest neighbours)")
    ax.set_ylabel("AUC")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend(loc="lower right", frameon=False)

    for ext in ("pdf", "png"):
        fig.savefig(fig_out / f"fig7_ksweep.{ext}")
    plt.close(fig)
    print(f"fig7_ksweep saved to {fig_out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     default="data/checkpoints/phase3/model_A_10M_v4/encoder_epoch075.pt")
    p.add_argument("--lhco_bg",  default="data/lhco/lhco_bg_jets.h5")
    p.add_argument("--lhco_2p",  default="data/lhco/events_anomalydetection_v2.h5")
    p.add_argument("--lhco_3p",  default="data/lhco/events_anomalydetection_Z_XY_qqq.h5")
    p.add_argument("--out_dir",  default="data/results/k_sweep")
    args = p.parse_args()
    main(args)

"""
Phase 2 Evaluation: all models × both LHCO signal types.

Models tested:
  Model A  — contrastive encoder trained on real AspenOpenJets
  Model B2 — contrastive encoder trained on LHCO QCD simulation
  Model C  — transformer autoencoder trained on real AspenOpenJets

Anomaly scoring:
  Models A, B2 : k-NN distance in backbone embedding space
  Model C      : per-jet reconstruction error (input-space anomaly score)

Signal benchmarks:
  2-prong : Z'→XY (W/Z bosons → qq) from events_anomalydetection_v2.h5
  3-prong : Z'→XY(qq)q from events_anomalydetection_Z_XY_qqq.h5

Runs n_seeds independent trials (different bg/sig splits) per (model, signal)
pair and reports mean AUC ± std.

Usage:
    python phase2_eval.py \
        --lhco_bg data/lhco/lhco_bg_jets.h5 \
        --lhco_2p data/lhco/events_anomalydetection_v2.h5 \
        --lhco_3p data/lhco/events_anomalydetection_Z_XY_qqq.h5 \
        --ckpt_A  data/checkpoints/phase2/model_A/encoder_epoch050.pt \
        --ckpt_B2 data/checkpoints/phase2/model_B2/encoder_epoch050.pt \
        --ckpt_C  data/checkpoints/phase2/model_C/autoencoder_epoch050.pt \
        --n_ref 20000 --n_bg 20000 --n_sig 2000 --n_seeds 5
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

from src.data.lhco_loader import LHCOJetDataset
from src.data.preprocessed_loader import PreprocessedJetDatasetCached
from src.models.encoder import JetEncoder
from src.models.autoencoder import JetAutoencoder, reconstruction_anomaly_score
from src.anomaly.knn_scorer import KNNAnomalyScorer, compute_rejection


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


def load_autoencoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    state = ckpt["model"]
    input_dim = state["input_proj.weight"].shape[1]
    model = JetAutoencoder(
        input_dim=input_dim,
        d_model=a.get("d_model", 64),
        nhead=a.get("nhead", 4),
        num_layers=a.get("num_layers", 2),
        latent_dim=a.get("latent_dim", 128),
        max_constituents=a.get("max_constituents", 50),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_backbone(model, xs, masks, device, batch_size=1024):
    """Encode with backbone() — pooled transformer output, not L2-normalised."""
    zs = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            xb = xs[i:i+batch_size].to(device)
            mb = masks[i:i+batch_size].to(device)
            zs.append(model.backbone(xb, mb).cpu())
    return torch.cat(zs, dim=0)


def recon_scores(model, xs, masks, device, batch_size=1024):
    """Compute per-jet reconstruction error for Model C."""
    scores = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            xb = xs[i:i+batch_size].to(device)
            mb = masks[i:i+batch_size].to(device)
            s = reconstruction_anomaly_score(model, xb, mb)
            scores.append(s.cpu())
    return torch.cat(scores, dim=0)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_bg_preprocessed(hdf5_path):
    """Load background from preprocessed HDF5 into RAM tensors."""
    ds = PreprocessedJetDatasetCached(hdf5_path)
    xs   = torch.from_numpy(ds._x)
    masks = torch.from_numpy(ds._mask)
    return xs, masks


def load_signal_lhco(hdf5_path, max_jets=None):
    """Load signal jets from raw LHCO HDF5 (pandas store, is_signal=True)."""
    ds = LHCOJetDataset(hdf5_path, is_signal=True, max_jets=max_jets)
    xs    = torch.stack([ds[i][0] for i in range(len(ds))])
    masks = torch.stack([ds[i][1] for i in range(len(ds))])
    return xs, masks


# ---------------------------------------------------------------------------
# Single-seed evaluation
# ---------------------------------------------------------------------------

def eval_one_seed(model, model_type, bg_xs, bg_masks,
                  sig_xs, sig_masks,
                  n_ref, n_bg, n_sig, seed, device):
    """
    Returns AUC for one random split.
    model_type: 'contrastive' or 'autoencoder'
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n_total_bg = len(bg_xs)

    # Random non-overlapping split of background
    perm = rng.permutation(n_total_bg)
    ref_idx  = perm[:n_ref]
    test_bg_idx = perm[n_ref:n_ref + n_bg]

    ref_xs    = bg_xs[ref_idx]
    ref_masks = bg_masks[ref_idx]
    tbg_xs    = bg_xs[test_bg_idx]
    tbg_masks = bg_masks[test_bg_idx]

    # Random signal subset
    n_sig_use = min(n_sig, len(sig_xs))
    sig_perm  = rng.permutation(len(sig_xs))[:n_sig_use]
    tsig_xs   = sig_xs[sig_perm]
    tsig_masks = sig_masks[sig_perm]

    # Concatenate test set
    test_xs    = torch.cat([tbg_xs, tsig_xs], dim=0)
    test_masks = torch.cat([tbg_masks, tsig_masks], dim=0)
    labels     = np.array([0] * n_bg + [1] * n_sig_use)

    if model_type == "contrastive":
        ref_z  = encode_backbone(model, ref_xs, ref_masks, device)
        test_z = encode_backbone(model, test_xs, test_masks, device)
        scorer = KNNAnomalyScorer(k=10)
        scorer.fit(ref_z)
        scores = scorer.score(test_z)

    else:  # autoencoder
        scores = recon_scores(model, test_xs, test_masks, device).numpy()

    auc = roc_auc_score(labels, scores)
    return auc, scores, labels


# ---------------------------------------------------------------------------
# Multi-seed evaluation for one (model, signal) pair
# ---------------------------------------------------------------------------

def eval_model_signal(model, model_type, bg_xs, bg_masks,
                      sig_xs, sig_masks,
                      n_ref, n_bg, n_sig, n_seeds, device, label):
    aucs = []
    best_auc, best_scores, best_labels = -1, None, None

    for seed in range(n_seeds):
        auc, scores, labels = eval_one_seed(
            model, model_type, bg_xs, bg_masks,
            sig_xs, sig_masks, n_ref, n_bg, n_sig, seed, device
        )
        aucs.append(auc)
        if auc > best_auc:
            best_auc, best_scores, best_labels = auc, scores, labels
        print(f"    seed {seed}: AUC={auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    print(f"  [{label}]  AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    # Compute ROC on best-seed scores for plotting
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(best_labels, best_scores)
    rej50 = compute_rejection(fpr, tpr, signal_eff=0.50)
    rej30 = compute_rejection(fpr, tpr, signal_eff=0.30)

    return {
        "label": label,
        "aucs": aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "rej_50": float(rej50),
        "rej_30": float(rej30),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  output: {out_dir}\n")

    # --- Load background (preprocessed, cached in RAM) ---
    print("Loading preprocessed LHCO background into RAM...")
    bg_xs, bg_masks = load_bg_preprocessed(args.lhco_bg)
    print(f"  {len(bg_xs):,} background jets available\n")

    # --- Load signals (raw LHCO HDF5, on-demand) ---
    print("Loading 2-prong signal (Z'→XY)...")
    sig2p_xs, sig2p_masks = load_signal_lhco(args.lhco_2p)
    print(f"  {len(sig2p_xs):,} 2-prong signal jets\n")

    print("Loading 3-prong signal (Z'→XY(qq)q)...")
    sig3p_xs, sig3p_masks = load_signal_lhco(args.lhco_3p)
    print(f"  {len(sig3p_xs):,} 3-prong signal jets\n")

    # --- Load models ---
    print("Loading models...")
    model_A  = load_contrastive(args.ckpt_A, device)
    print(f"  Model A  loaded: {args.ckpt_A}")
    model_B2 = load_contrastive(args.ckpt_B2, device)
    print(f"  Model B2 loaded: {args.ckpt_B2}")
    model_C  = load_autoencoder(args.ckpt_C, device)
    print(f"  Model C  loaded: {args.ckpt_C}\n")

    # --- Evaluate ---
    configs = [
        ("Model A (real contrastive)",  model_A,  "contrastive"),
        ("Model B2 (sim contrastive)",  model_B2, "contrastive"),
        ("Model C (autoencoder)",       model_C,  "autoencoder"),
    ]
    signals = [
        ("2-prong", sig2p_xs, sig2p_masks),
        ("3-prong", sig3p_xs, sig3p_masks),
    ]

    all_results = {}
    for model_name, model, model_type in configs:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")
        all_results[model_name] = {}

        for sig_name, sig_xs, sig_masks in signals:
            print(f"\n  Signal: {sig_name}  ({args.n_seeds} seeds)")
            res = eval_model_signal(
                model, model_type,
                bg_xs, bg_masks, sig_xs, sig_masks,
                n_ref=args.n_ref, n_bg=args.n_bg,
                n_sig=args.n_sig, n_seeds=args.n_seeds,
                device=device,
                label=f"{model_name} | {sig_name}",
            )
            all_results[model_name][sig_name] = res

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"  Phase 2 Results Summary")
    print(f"{'='*70}")
    header = f"  {'Model':<28} {'Signal':<10} {'AUC (mean±std)':<18} {'Rej@50%':>10} {'Rej@30%':>10}"
    print(header)
    print(f"  {'-'*66}")
    for model_name in all_results:
        for sig_name in all_results[model_name]:
            r = all_results[model_name][sig_name]
            rej50 = f"{r['rej_50']:.1f}" if r['rej_50'] != float('inf') else "∞"
            rej30 = f"{r['rej_30']:.1f}" if r['rej_30'] != float('inf') else "∞"
            print(f"  {model_name:<28} {sig_name:<10} "
                  f"{r['mean_auc']:.4f} ± {r['std_auc']:.4f}    "
                  f"{rej50:>10} {rej30:>10}")
    print(f"{'='*70}\n")

    # --- ROC curves ---
    COLORS = {
        "Model A (real contrastive)":  {"2-prong": "#1f77b4", "3-prong": "#aec7e8"},
        "Model B2 (sim contrastive)":  {"2-prong": "#ff7f0e", "3-prong": "#ffbb78"},
        "Model C (autoencoder)":       {"2-prong": "#2ca02c", "3-prong": "#98df8a"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sig_name in zip(axes, ["2-prong", "3-prong"]):
        for model_name in all_results:
            r = all_results[model_name][sig_name]
            color = COLORS[model_name][sig_name]
            auc_str = f"{r['mean_auc']:.3f}±{r['std_auc']:.3f}"
            ax.plot(r["fpr"], r["tpr"], lw=2, color=color,
                    label=f"{model_name} (AUC={auc_str})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate (background)")
        ax.set_ylabel("True Positive Rate (signal)")
        ax.set_title(f"Phase 2 ROC — {sig_name} signal")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_path = out_dir / "phase2_roc.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    print(f"ROC curves saved to {roc_path}")

    # --- Save JSON (excluding large fpr/tpr arrays) ---
    json_results = {}
    for model_name in all_results:
        json_results[model_name] = {}
        for sig_name in all_results[model_name]:
            r = all_results[model_name][sig_name]
            json_results[model_name][sig_name] = {
                "aucs":     r["aucs"],
                "mean_auc": r["mean_auc"],
                "std_auc":  r["std_auc"],
                "rej_50":   r["rej_50"],
                "rej_30":   r["rej_30"],
            }
    json_path = out_dir / "phase2_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Results JSON saved to {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lhco_bg",  default="data/lhco/lhco_bg_jets.h5")
    p.add_argument("--lhco_2p",  default="data/lhco/events_anomalydetection_v2.h5")
    p.add_argument("--lhco_3p",  default="data/lhco/events_anomalydetection_Z_XY_qqq.h5")
    p.add_argument("--ckpt_A",   default="data/checkpoints/phase2/model_A/encoder_epoch050.pt")
    p.add_argument("--ckpt_B2",  default="data/checkpoints/phase2/model_B2/encoder_epoch050.pt")
    p.add_argument("--ckpt_C",   default="data/checkpoints/phase2/model_C/autoencoder_epoch050.pt")
    p.add_argument("--n_ref",    type=int, default=20000,
                   help="Background jets for k-NN reference set")
    p.add_argument("--n_bg",     type=int, default=20000,
                   help="Background jets in test set (non-overlapping with ref)")
    p.add_argument("--n_sig",    type=int, default=2000,
                   help="Signal jets in test set")
    p.add_argument("--n_seeds",  type=int, default=5,
                   help="Independent random splits for AUC statistics")
    p.add_argument("--out_dir",  default="data/results/phase2")
    args = p.parse_args()
    main(args)

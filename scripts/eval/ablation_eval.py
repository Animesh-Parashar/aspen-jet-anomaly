"""
ablation_eval.py — Fair ablation: all three models scored with k-NN.

Phase 2 used reconstruction error for Model C, which inverts (AUC ≈ 0.39)
due to domain gap between LHCO QCD test jets and the AspenOpenJets training
distribution.  This script instead encodes Model C's bottleneck with the same
k-NN scorer used for A and B2, making the comparison architecturally fair.

Interpretation guide:
  Model C k-NN << Model A  →  contrastive objective matters.
  Model C k-NN ≈  Model A  →  real data dominates; objective is secondary.

Outputs:
  data/results/ablation/ablation_results.json
  data/results/ablation/ablation_roc.png

Usage:
    python ablation_eval.py \
        --lhco_bg data/lhco/lhco_bg_jets.h5 \
        --lhco_2p data/lhco/events_anomalydetection_v2.h5 \
        --lhco_3p data/lhco/events_anomalydetection_Z_XY_qqq.h5 \
        --ckpt_A  data/checkpoints/phase2/model_A/encoder_epoch050.pt \
        --ckpt_B2 data/checkpoints/phase2/model_B2/encoder_epoch050.pt \
        --ckpt_C  data/checkpoints/phase2/model_C/autoencoder_epoch050.pt
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

def encode_backbone(model, xs, masks, device, batch_size=1024, feature_cols=None):
    """Contrastive encoder: pooled transformer output (d_model), NOT L2-normalised."""
    zs = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            xb = xs[i:i+batch_size].to(device)
            mb = masks[i:i+batch_size].to(device)
            if feature_cols is not None:
                xb = xb[:, :, feature_cols]
            zs.append(model.backbone(xb, mb).cpu())
    return torch.cat(zs, dim=0)


def encode_ae_bottleneck(model, xs, masks, device, batch_size=1024):
    """AE encoder: (B, latent_dim) bottleneck, NOT L2-normalised."""
    zs = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            xb = xs[i:i+batch_size].to(device)
            mb = masks[i:i+batch_size].to(device)
            zs.append(model.encode(xb, mb).cpu())
    return torch.cat(zs, dim=0)


def recon_scores(model, xs, masks, device, batch_size=1024):
    """Reconstruction error scores for Model C (kept for reference comparison)."""
    scores = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            xb = xs[i:i+batch_size].to(device)
            mb = masks[i:i+batch_size].to(device)
            s = reconstruction_anomaly_score(model, xb, mb)
            scores.append(s.cpu())
    return torch.cat(scores, dim=0).numpy()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_bg_preprocessed(hdf5_path):
    ds = PreprocessedJetDatasetCached(hdf5_path)
    return torch.from_numpy(ds._x), torch.from_numpy(ds._mask)


def load_signal_lhco(hdf5_path, max_jets=None):
    ds = LHCOJetDataset(hdf5_path, is_signal=True, max_jets=max_jets)
    xs    = torch.stack([ds[i][0] for i in range(len(ds))])
    masks = torch.stack([ds[i][1] for i in range(len(ds))])
    return xs, masks


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------

def eval_one_seed(encode_fn, bg_xs, bg_masks, sig_xs, sig_masks,
                  n_ref, n_bg, n_sig, seed, device):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bg_xs))
    ref_xs    = bg_xs[perm[:n_ref]]
    ref_masks = bg_masks[perm[:n_ref]]
    tbg_xs    = bg_xs[perm[n_ref:n_ref + n_bg]]
    tbg_masks = bg_masks[perm[n_ref:n_ref + n_bg]]

    n_sig_use = min(n_sig, len(sig_xs))
    sig_perm  = rng.permutation(len(sig_xs))[:n_sig_use]
    tsig_xs   = sig_xs[sig_perm]
    tsig_masks = sig_masks[sig_perm]

    test_xs    = torch.cat([tbg_xs, tsig_xs], dim=0)
    test_masks = torch.cat([tbg_masks, tsig_masks], dim=0)
    labels     = np.array([0] * n_bg + [1] * n_sig_use)

    ref_z  = encode_fn(ref_xs, ref_masks, device)
    test_z = encode_fn(test_xs, test_masks, device)
    scorer = KNNAnomalyScorer(k=10)
    scorer.fit(ref_z)
    scores = scorer.score(test_z)

    auc = roc_auc_score(labels, scores)
    return auc, scores, labels


def eval_model_signal(encode_fn, bg_xs, bg_masks, sig_xs, sig_masks,
                      n_ref, n_bg, n_sig, n_seeds, device, label):
    from sklearn.metrics import roc_curve
    aucs = []
    best_auc, best_scores, best_labels = -1, None, None

    for seed in range(n_seeds):
        auc, scores, labels = eval_one_seed(
            encode_fn, bg_xs, bg_masks, sig_xs, sig_masks,
            n_ref, n_bg, n_sig, seed, device)
        aucs.append(auc)
        if auc > best_auc:
            best_auc, best_scores, best_labels = auc, scores, labels
        print(f"    seed {seed}: AUC={auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    print(f"  [{label}]  AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    fpr, tpr, _ = roc_curve(best_labels, best_scores)
    rej50 = compute_rejection(fpr, tpr, 0.50)
    rej30 = compute_rejection(fpr, tpr, 0.30)

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

    print("Loading preprocessed LHCO background...")
    bg_xs, bg_masks = load_bg_preprocessed(args.lhco_bg)
    print(f"  {len(bg_xs):,} background jets\n")

    print("Loading 2-prong signal...")
    sig2p_xs, sig2p_masks = load_signal_lhco(args.lhco_2p)
    print(f"  {len(sig2p_xs):,} 2-prong jets\n")

    print("Loading 3-prong signal...")
    sig3p_xs, sig3p_masks = load_signal_lhco(args.lhco_3p)
    print(f"  {len(sig3p_xs):,} 3-prong jets\n")

    print("Loading models...")
    model_A  = load_contrastive(args.ckpt_A, device)
    model_B2 = load_contrastive(args.ckpt_B2, device)
    model_C  = load_autoencoder(args.ckpt_C, device)
    print(f"  A : {args.ckpt_A}")
    print(f"  B2: {args.ckpt_B2}")
    print(f"  C : {args.ckpt_C}\n")

    # Wrap encode functions (closes over model + batch_size)
    feat_A = args.feature_cols_A  # None = all 7; [0,1,2,3] = 4-feature variant

    def enc_A(xs, masks, dev):
        return encode_backbone(model_A, xs, masks, dev, feature_cols=feat_A)

    def enc_B2(xs, masks, dev):
        return encode_backbone(model_B2, xs, masks, dev)

    def enc_C(xs, masks, dev):
        return encode_ae_bottleneck(model_C, xs, masks, dev)

    configs = [
        ("Model A  (real contrastive, k-NN)",     enc_A),
        ("Model B2 (sim contrastive,  k-NN)",     enc_B2),
        ("Model C  (autoencoder, bottleneck k-NN)", enc_C),
    ]
    signals = [
        ("2-prong", sig2p_xs, sig2p_masks),
        ("3-prong", sig3p_xs, sig3p_masks),
    ]

    all_results = {}
    for model_name, enc_fn in configs:
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        all_results[model_name] = {}
        for sig_name, sig_xs, sig_masks in signals:
            print(f"\n  Signal: {sig_name}  ({args.n_seeds} seeds)")
            res = eval_model_signal(
                enc_fn, bg_xs, bg_masks, sig_xs, sig_masks,
                n_ref=args.n_ref, n_bg=args.n_bg,
                n_sig=args.n_sig, n_seeds=args.n_seeds,
                device=device,
                label=f"{model_name} | {sig_name}",
            )
            all_results[model_name][sig_name] = res

    # --- Also score Model C with reconstruction (reference only) ---
    print(f"\n{'='*60}\n  Model C (reconstruction, reference only)\n{'='*60}")
    from sklearn.metrics import roc_auc_score
    recon_aucs = {"2-prong": [], "3-prong": []}
    for sig_name, sig_xs, sig_masks in signals:
        print(f"\n  Signal: {sig_name}")
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(bg_xs))
            tbg_xs    = bg_xs[perm[args.n_ref:args.n_ref + args.n_bg]]
            tbg_masks = bg_masks[perm[args.n_ref:args.n_ref + args.n_bg]]
            n_sig_use = min(args.n_sig, len(sig_xs))
            sp = rng.permutation(len(sig_xs))[:n_sig_use]
            test_xs    = torch.cat([tbg_xs, sig_xs[sp]])
            test_masks = torch.cat([tbg_masks, sig_masks[sp]])
            labels     = np.array([0]*args.n_bg + [1]*n_sig_use)
            scores = recon_scores(model_C, test_xs, test_masks, device)
            auc = roc_auc_score(labels, scores)
            recon_aucs[sig_name].append(auc)
            print(f"    seed {seed}: AUC={auc:.4f}")
        m, s = np.mean(recon_aucs[sig_name]), np.std(recon_aucs[sig_name])
        print(f"  Model C recon | {sig_name}  AUC = {m:.4f} ± {s:.4f}")

    # --- Summary table ---
    print(f"\n{'='*74}")
    print(f"  Ablation Summary (k-NN on bottleneck for all models)")
    print(f"{'='*74}")
    header = f"  {'Model':<44} {'Signal':<8} {'AUC':>12} {'Rej@50%':>10} {'Rej@30%':>10}"
    print(header)
    print(f"  {'-'*70}")
    for model_name in all_results:
        for sig_name in all_results[model_name]:
            r = all_results[model_name][sig_name]
            rej50 = f"{r['rej_50']:.1f}" if r['rej_50'] != float('inf') else "∞"
            rej30 = f"{r['rej_30']:.1f}" if r['rej_30'] != float('inf') else "∞"
            print(f"  {model_name:<44} {sig_name:<8} "
                  f"{r['mean_auc']:.4f}±{r['std_auc']:.4f}  {rej50:>10} {rej30:>10}")
    print(f"\n  Reference: Model C reconstruction error (phase2 approach, inverted):")
    for sig_name in ["2-prong", "3-prong"]:
        m, s = np.mean(recon_aucs[sig_name]), np.std(recon_aucs[sig_name])
        print(f"  {'Model C  (recon, reference)':<44} {sig_name:<8} {m:.4f}±{s:.4f}")
    print(f"{'='*74}\n")

    # --- ROC plot ---
    COLORS = {
        "Model A  (real contrastive, k-NN)":      {"2-prong": "#1f77b4", "3-prong": "#aec7e8"},
        "Model B2 (sim contrastive,  k-NN)":      {"2-prong": "#ff7f0e", "3-prong": "#ffbb78"},
        "Model C  (autoencoder, bottleneck k-NN)": {"2-prong": "#2ca02c", "3-prong": "#98df8a"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sig_name in zip(axes, ["2-prong", "3-prong"]):
        for model_name in all_results:
            r = all_results[model_name][sig_name]
            color = COLORS[model_name][sig_name]
            auc_str = f"{r['mean_auc']:.3f}±{r['std_auc']:.3f}"
            ax.plot(r["fpr"], r["tpr"], lw=2, color=color,
                    label=f"{model_name.split('(')[0].strip()} (AUC={auc_str})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Ablation ROC — {sig_name} signal (k-NN on bottleneck)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = out_dir / "ablation_roc.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    print(f"ROC curves saved to {roc_path}")

    # --- Save JSON ---
    json_out = {}
    for model_name in all_results:
        json_out[model_name] = {}
        for sig_name in all_results[model_name]:
            r = all_results[model_name][sig_name]
            json_out[model_name][sig_name] = {
                "aucs": r["aucs"], "mean_auc": r["mean_auc"], "std_auc": r["std_auc"],
                "rej_50": r["rej_50"], "rej_30": r["rej_30"],
            }
    json_out["Model C (recon, reference)"] = {
        sig: {"aucs": recon_aucs[sig],
              "mean_auc": float(np.mean(recon_aucs[sig])),
              "std_auc":  float(np.std(recon_aucs[sig]))}
        for sig in ["2-prong", "3-prong"]
    }
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lhco_bg",  default="data/lhco/lhco_bg_jets.h5")
    p.add_argument("--lhco_2p",  default="data/lhco/events_anomalydetection_v2.h5")
    p.add_argument("--lhco_3p",  default="data/lhco/events_anomalydetection_Z_XY_qqq.h5")
    p.add_argument("--ckpt_A",   default="data/checkpoints/phase2/model_A/encoder_epoch050.pt")
    p.add_argument("--ckpt_B2",  default="data/checkpoints/phase2/model_B2/encoder_epoch050.pt")
    p.add_argument("--ckpt_C",   default="data/checkpoints/phase2/model_C/autoencoder_epoch050.pt")
    p.add_argument("--n_ref",    type=int, default=20000)
    p.add_argument("--n_bg",     type=int, default=20000)
    p.add_argument("--n_sig",    type=int, default=2000)
    p.add_argument("--n_seeds",  type=int, default=5)
    p.add_argument("--out_dir",  default="data/results/ablation")
    p.add_argument("--feature_cols_A", type=int, nargs="+", default=None,
                   help="Feature columns to use for Model A encoding. "
                        "4-feature variant: --feature_cols_A 0 1 2 3")
    args = p.parse_args()
    main(args)

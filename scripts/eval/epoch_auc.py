"""
epoch_auc.py — AUC learning curves across all saved checkpoints.

For each model (A, B2, C) at each saved epoch (10, 20, 30, 40, 50), computes
2-prong and 3-prong AUC using k-NN scoring.  Model C uses bottleneck k-NN
(same fix as ablation_eval.py, not reconstruction error).

Why this matters:
  - Shows whether AUC grows monotonically or peaks at an earlier epoch.
  - Finding: if 3-prong peaks at epoch 20-30 while 2-prong keeps improving,
    that justifies epoch-specific checkpoints in the paper.
  - Cheap — all data stays in RAM; only encoder weights change.

Outputs:
  data/results/epoch_auc/epoch_auc_results.json
  data/results/epoch_auc/epoch_auc_curves.png

Usage:
    python epoch_auc.py \
        --lhco_bg data/lhco/lhco_bg_jets.h5 \
        --lhco_2p data/lhco/events_anomalydetection_v2.h5 \
        --lhco_3p data/lhco/events_anomalydetection_Z_XY_qqq.h5 \
        --n_seeds 3
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
from src.models.autoencoder import JetAutoencoder
from src.anomaly.knn_scorer import KNNAnomalyScorer


# ---------------------------------------------------------------------------
# Checkpoint catalogue
# ---------------------------------------------------------------------------

CHECKPOINTS = {
    "Model A": {
        "type": "contrastive",
        "epochs": [10, 20, 30, 40, 50],
        "fmt": "data/checkpoints/phase2/model_A/encoder_epoch{:03d}.pt",
    },
    "Model B2": {
        "type": "contrastive",
        "epochs": [10, 20, 30, 40, 50],
        "fmt": "data/checkpoints/phase2/model_B2/encoder_epoch{:03d}.pt",
    },
    "Model C": {
        "type": "autoencoder",
        "epochs": [10, 20, 30, 40, 50],
        "fmt": "data/checkpoints/phase2/model_C/autoencoder_epoch{:03d}.pt",
    },
}


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

def encode_contrastive(model, xs, masks, device, batch_size=1024):
    zs = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            zs.append(model.backbone(xs[i:i+batch_size].to(device),
                                      masks[i:i+batch_size].to(device)).cpu())
    return torch.cat(zs)


def encode_ae_bottleneck(model, xs, masks, device, batch_size=1024):
    zs = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            zs.append(model.encode(xs[i:i+batch_size].to(device),
                                    masks[i:i+batch_size].to(device)).cpu())
    return torch.cat(zs)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_bg(hdf5_path):
    ds = PreprocessedJetDatasetCached(hdf5_path)
    return torch.from_numpy(ds._x), torch.from_numpy(ds._mask)


def load_signal(hdf5_path):
    ds = LHCOJetDataset(hdf5_path, is_signal=True)
    xs    = torch.stack([ds[i][0] for i in range(len(ds))])
    masks = torch.stack([ds[i][1] for i in range(len(ds))])
    return xs, masks


# ---------------------------------------------------------------------------
# Single-seed AUC
# ---------------------------------------------------------------------------

def auc_one_seed(encode_fn, bg_xs, bg_masks, sig_xs, sig_masks,
                 n_ref, n_bg, n_sig, seed, device):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bg_xs))
    ref_xs    = bg_xs[perm[:n_ref]]
    ref_masks = bg_masks[perm[:n_ref]]
    tbg_xs    = bg_xs[perm[n_ref:n_ref + n_bg]]
    tbg_masks = bg_masks[perm[n_ref:n_ref + n_bg]]

    n_sig_use = min(n_sig, len(sig_xs))
    sp = rng.permutation(len(sig_xs))[:n_sig_use]
    test_xs    = torch.cat([tbg_xs, sig_xs[sp]])
    test_masks = torch.cat([tbg_masks, sig_masks[sp]])
    labels     = np.array([0]*n_bg + [1]*n_sig_use)

    ref_z  = encode_fn(ref_xs, ref_masks, device)
    test_z = encode_fn(test_xs, test_masks, device)
    scorer = KNNAnomalyScorer(k=10)
    scorer.fit(ref_z)
    scores = scorer.score(test_z)
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  n_seeds: {args.n_seeds}\n")

    if args.ckpt_dir_A is not None:
        CHECKPOINTS["Model A"]["fmt"] = str(Path(args.ckpt_dir_A) / "encoder_epoch{:03d}.pt")
        print(f"Model A checkpoints overridden → {args.ckpt_dir_A}\n")
    if args.epochs_A is not None:
        CHECKPOINTS["Model A"]["epochs"] = args.epochs_A
        print(f"Model A epochs overridden → {args.epochs_A}\n")

    print("Loading background into RAM...")
    bg_xs, bg_masks = load_bg(args.lhco_bg)
    print(f"  {len(bg_xs):,} jets\n")

    print("Loading 2-prong signal...")
    sig2_xs, sig2_masks = load_signal(args.lhco_2p)
    print(f"  {len(sig2_xs):,} jets\n")

    print("Loading 3-prong signal...")
    sig3_xs, sig3_masks = load_signal(args.lhco_3p)
    print(f"  {len(sig3_xs):,} jets\n")

    results = {}  # results[model_name][sig_name] = list of mean_auc per epoch

    for model_name, cfg in CHECKPOINTS.items():
        results[model_name] = {"2-prong": [], "3-prong": [], "epochs": cfg["epochs"]}
        print(f"\n{'='*60}\n  {model_name} ({cfg['type']})\n{'='*60}")

        for epoch in cfg["epochs"]:
            ckpt_path = cfg["fmt"].format(epoch)
            if not Path(ckpt_path).exists():
                print(f"  epoch {epoch:3d}: MISSING {ckpt_path}")
                results[model_name]["2-prong"].append(None)
                results[model_name]["3-prong"].append(None)
                continue

            if cfg["type"] == "contrastive":
                model = load_contrastive(ckpt_path, device)
                def enc(xs, masks, dev, _m=model):
                    return encode_contrastive(_m, xs, masks, dev)
            else:
                model = load_autoencoder(ckpt_path, device)
                def enc(xs, masks, dev, _m=model):
                    return encode_ae_bottleneck(_m, xs, masks, dev)

            row = {}
            for sig_name, sig_xs, sig_masks in [("2-prong", sig2_xs, sig2_masks),
                                                  ("3-prong", sig3_xs, sig3_masks)]:
                aucs = [auc_one_seed(enc, bg_xs, bg_masks, sig_xs, sig_masks,
                                     args.n_ref, args.n_bg, args.n_sig, s, device)
                        for s in range(args.n_seeds)]
                row[sig_name] = float(np.mean(aucs))
                results[model_name][sig_name].append(row[sig_name])

            print(f"  epoch {epoch:3d}:  2-prong AUC={row['2-prong']:.4f}  "
                  f"3-prong AUC={row['3-prong']:.4f}")
            del model   # free GPU memory before loading next checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"  Epoch-by-Epoch AUC Summary")
    print(f"{'='*70}")
    all_epochs = results[next(iter(results))]["epochs"]
    print(f"  {'Model':<12} {'Signal':<10}", end="")
    for ep in all_epochs:
        print(f"  ep{ep:02d}", end="")
    print()
    print(f"  {'-'*66}")
    for model_name in results:
        for sig_name in ["2-prong", "3-prong"]:
            print(f"  {model_name:<12} {sig_name:<10}", end="")
            for v in results[model_name][sig_name]:
                print(f"  {v:.3f}" if v is not None else "   N/A", end="")
            print()
    print(f"{'='*70}\n")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    STYLE = {
        "Model A":  {"color": "#1f77b4", "ls": "-",  "marker": "o"},
        "Model B2": {"color": "#ff7f0e", "ls": "--", "marker": "s"},
        "Model C":  {"color": "#2ca02c", "ls": ":",  "marker": "^"},
    }
    for ax, sig_name in zip(axes, ["2-prong", "3-prong"]):
        for model_name, sty in STYLE.items():
            epochs = results[model_name]["epochs"]
            aucs   = results[model_name][sig_name]
            valid_ep  = [e for e, a in zip(epochs, aucs) if a is not None]
            valid_auc = [a for a in aucs if a is not None]
            ax.plot(valid_ep, valid_auc,
                    color=sty["color"], ls=sty["ls"], marker=sty["marker"],
                    lw=2, ms=7, label=model_name)
        ax.axhline(0.5, color="k", lw=1, ls="--", label="Random")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("AUC (k-NN, mean over seeds)")
        ax.set_title(f"AUC learning curve — {sig_name} signal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([10, 20, 30, 40, 50])

    plt.tight_layout()
    fig_path = out_dir / "epoch_auc_curves.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Learning curves saved to {fig_path}")

    # --- Save JSON (strip None for serialisability) ---
    json_out = {}
    for model_name in results:
        json_out[model_name] = {
            "epochs": results[model_name]["epochs"],
            "2-prong": results[model_name]["2-prong"],
            "3-prong": results[model_name]["3-prong"],
        }
    json_path = out_dir / "epoch_auc_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lhco_bg",  default="data/lhco/lhco_bg_jets.h5")
    p.add_argument("--lhco_2p",  default="data/lhco/events_anomalydetection_v2.h5")
    p.add_argument("--lhco_3p",  default="data/lhco/events_anomalydetection_Z_XY_qqq.h5")
    p.add_argument("--n_ref",    type=int, default=20000)
    p.add_argument("--n_bg",     type=int, default=20000)
    p.add_argument("--n_sig",    type=int, default=2000)
    p.add_argument("--n_seeds",  type=int, default=3,
                   help="Seeds per (model, epoch, signal) — 3 is sufficient for curves")
    p.add_argument("--out_dir",  default="data/results/epoch_auc")
    p.add_argument("--ckpt_dir_A", default=None,
                   help="Override Model A checkpoint dir, e.g. data/checkpoints/phase3/model_A_6M")
    p.add_argument("--epochs_A", type=int, nargs="+", default=None,
                   help="Override epoch checkpoints to evaluate for Model A, e.g. --epochs_A 10 15 20")
    args = p.parse_args()
    main(args)

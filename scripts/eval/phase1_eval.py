"""
Phase 1 go/no-go evaluation.

Encodes LHCO background + signal-injected jets, computes k-NN anomaly scores
using LHCO background as the reference distribution, plots ROC curve, and
prints AUC. AUC > 0.55 = proceed to Phase 2.

Key design: the encoder is trained on AspenOpenJets (real data), but the
k-NN reference distribution is LHCO background (same domain as the test set).
This correctly measures whether the learned representation separates signal
from background within the LHCO benchmark domain.

Usage:
    python phase1_eval.py \
        --checkpoint data/checkpoints/phase1/encoder_epoch010.pt \
        --lhco data/lhco/events_anomalydetection_v2.h5 \
        --n_bg 10000 --n_sig 1000
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))


import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data.lhco_loader import LHCOJetDataset, make_injection_dataset
from src.models.encoder import JetEncoder
from src.anomaly.knn_scorer import KNNAnomalyScorer, compute_rejection


def encode_dataset(model, xs, masks, device, batch_size=512):
    model.eval()
    zs = []
    for i in range(0, len(xs), batch_size):
        xb = xs[i:i+batch_size].to(device)
        mb = masks[i:i+batch_size].to(device)
        with torch.no_grad():
            z = model(xb, mb)
        zs.append(z.cpu())
    return torch.cat(zs, dim=0)


def load_encoder(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    state = ckpt["model"]
    input_dim = state["input_proj.weight"].shape[1]
    model = JetEncoder(
        input_dim=input_dim,
        d_model=saved_args.get("d_model", 64),
        nhead=saved_args.get("nhead", 4),
        num_layers=saved_args.get("num_layers", 2),
        latent_dim=saved_args.get("latent_dim", 128),
    ).to(device)
    model.load_state_dict(state)
    return model, saved_args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, saved_args = load_encoder(args.checkpoint, device)
    max_const = saved_args.get("max_constituents", 50)
    print(f"Loaded checkpoint: {args.checkpoint}  (input_dim={saved_args.get('d_model',64)}, latent={saved_args.get('latent_dim',128)})")

    # LHCO background: split into reference set and test background
    print("Loading LHCO background...")
    lhco_bg = LHCOJetDataset(args.lhco, is_signal=False,
                              max_jets=args.n_bg * 2,
                              max_constituents=max_const)
    print("Loading LHCO signal...")
    lhco_sig = LHCOJetDataset(args.lhco, is_signal=True,
                               max_jets=args.n_sig,
                               max_constituents=max_const)

    # Reference: first n_bg background jets (held out from test)
    # Test: next n_bg background jets + n_sig signal jets
    n_ref = min(args.n_bg, len(lhco_bg) // 2)
    n_test_bg = min(args.n_bg, len(lhco_bg) - n_ref)

    ref_jets  = torch.stack([lhco_bg[i][0] for i in range(n_ref)])
    ref_masks = torch.stack([lhco_bg[i][1] for i in range(n_ref)])

    # Build test set from non-overlapping background + signal
    class _SlicedDataset:
        def __init__(self, ds, start, end):
            self.ds = ds; self.start = start; self.end = end
        def __len__(self): return self.end - self.start
        def __getitem__(self, i): return self.ds[self.start + i]

    test_bg_ds  = _SlicedDataset(lhco_bg, n_ref, n_ref + n_test_bg)
    test_jets, test_masks, test_labels = make_injection_dataset(
        test_bg_ds, lhco_sig, n_bg=n_test_bg, n_sig=args.n_sig
    )

    print(f"Reference set: {n_ref:,} background jets")
    print(f"Test set: {n_test_bg:,} background + {min(args.n_sig, len(lhco_sig)):,} signal")

    print("Encoding reference set...")
    ref_z  = encode_dataset(model, ref_jets, ref_masks, device)
    print("Encoding test set...")
    test_z = encode_dataset(model, test_jets, test_masks, device)

    # k-NN scoring: anomaly score = distance to k-th NN in LHCO background
    scorer = KNNAnomalyScorer(k=10)
    scorer.fit(ref_z)
    auc, fpr, tpr, _ = scorer.evaluate(test_z, test_labels.numpy())

    rej_50 = compute_rejection(fpr, tpr, signal_eff=0.50)
    rej_30 = compute_rejection(fpr, tpr, signal_eff=0.30)

    print(f"\n{'='*40}")
    print(f"  Phase 1 Go/No-Go Results")
    print(f"{'='*40}")
    print(f"  AUC                   : {auc:.4f}")
    print(f"  Bkg rej @ 50% sig eff : {rej_50:.1f}")
    print(f"  Bkg rej @ 30% sig eff : {rej_30:.1f}")
    print(f"{'='*40}")

    if auc > 0.55:
        print("  VERDICT: GO  (AUC > 0.55 — proceed to Phase 2)")
    elif auc > 0.50:
        print("  VERDICT: MARGINAL (0.50 < AUC <= 0.55 — try more epochs or more jets)")
    else:
        print("  VERDICT: NO-GO (AUC < 0.50 — inverted; check preprocessing)")

    # ROC plot
    out_dir = Path(args.checkpoint).parent
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"Contrastive encoder (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate (background)")
    ax.set_ylabel("True Positive Rate (signal)")
    ax.set_title("Phase 1 Go/No-Go ROC\n(k-NN on LHCO background reference)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    roc_path = out_dir / "phase1_roc.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    print(f"\n  ROC curve saved to {roc_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lhco", required=True, help="LHCO R&D HDF5 (events_anomalydetection_v2.h5)")
    p.add_argument("--n_bg", type=int, default=10_000)
    p.add_argument("--n_sig", type=int, default=1_000)
    args = p.parse_args()
    main(args)

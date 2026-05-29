"""
scaling_curve.py — Plot Model A AUC vs training data size.

Reads ablation JSON files produced by ablation_eval.py at each scale point.
Skips any entry whose JSON file does not yet exist (so you can run
this incrementally as each eval finishes).

Usage:
    python scaling_curve.py
    python scaling_curve.py --out_dir data/results/scaling_curve
    python scaling_curve.py --add 8M data/results/ablation_8M/ablation_results.json
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

MODEL_KEY = "Model A  (real contrastive, k-NN)"

DEFAULT_POINTS = [
    ("2M",  "data/results/ablation_2M_final/ablation_results.json"),
    ("4M",  "data/results/ablation_4M_75ep/ablation_results.json"),
    ("6M",  "data/results/ablation_6M_final/ablation_results.json"),
    ("10M", "data/results/ablation_10M_final/ablation_results.json"),
]

SIZE_TO_FLOAT = {
    "2M": 2.0, "4M": 4.0, "6M": 6.0, "8M": 8.0, "10M": 10.0,
}


def load_point(label, json_path):
    p = Path(json_path)
    if not p.exists():
        print(f"  [{label}] {json_path} not found — skipping")
        return None
    with open(p) as f:
        data = json.load(f)
    if MODEL_KEY not in data:
        print(f"  [{label}] key '{MODEL_KEY}' not in JSON — skipping")
        return None
    r = data[MODEL_KEY]
    return {
        "label": label,
        "size":  SIZE_TO_FLOAT.get(label, float(label.rstrip("M"))),
        "2p_mean": r["2-prong"]["mean_auc"],
        "2p_std":  r["2-prong"]["std_auc"],
        "3p_mean": r["3-prong"]["mean_auc"],
        "3p_std":  r["3-prong"]["std_auc"],
    }


def main(args):
    points_spec = list(DEFAULT_POINTS)
    for label, path in args.add:
        points_spec.append((label, path))
    points_spec.sort(key=lambda x: SIZE_TO_FLOAT.get(x[0], float(x[0].rstrip("M"))))

    points = []
    for label, path in points_spec:
        pt = load_point(label, path)
        if pt is not None:
            points.append(pt)

    if not points:
        print("No data points found. Run ablation_eval.py first.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes      = [p["size"] for p in points]
    labels     = [p["label"] for p in points]
    auc_2p     = [p["2p_mean"] for p in points]
    err_2p     = [p["2p_std"]  for p in points]
    auc_3p     = [p["3p_mean"] for p in points]
    err_3p     = [p["3p_std"]  for p in points]

    print(f"\n{'='*52}")
    print(f"  Scaling curve — Model A (real contrastive, k-NN)")
    print(f"{'='*52}")
    print(f"  {'Size':<6} {'2-prong AUC':>14} {'3-prong AUC':>14}")
    print(f"  {'-'*46}")
    for p in points:
        print(f"  {p['label']:<6} {p['2p_mean']:.4f}±{p['2p_std']:.4f}"
              f"   {p['3p_mean']:.4f}±{p['3p_std']:.4f}")
    print(f"{'='*52}\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, auc, err, sig in zip(axes,
                                  [auc_2p, auc_3p],
                                  [err_2p, err_3p],
                                  ["2-prong", "3-prong"]):
        ax.errorbar(sizes, auc, yerr=err, fmt="o-", capsize=5,
                    linewidth=2, markersize=7, color="#1f77b4")
        ax.set_xticks(sizes)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Training data size")
        ax.set_ylabel("AUC")
        ax.set_title(f"Scaling curve — {sig} signal")
        ax.set_ylim(0.48, 0.68)
        ax.grid(True, alpha=0.3)
        for x, y, e, lbl in zip(sizes, auc, err, labels):
            ax.annotate(f"{y:.3f}", (x, y + e + 0.003),
                        ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model A (real contrastive) — AUC vs. training data size",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig_path = out_dir / "scaling_curve.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {fig_path}")

    json_out = {p["label"]: {
        "size_M": p["size"],
        "2p_mean": p["2p_mean"], "2p_std": p["2p_std"],
        "3p_mean": p["3p_mean"], "3p_std": p["3p_std"],
    } for p in points}
    json_path = out_dir / "scaling_curve.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Data saved to {json_path}")


class AddPointAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, []) or []
        items.append((values[0], values[1]))
        setattr(namespace, self.dest, items)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/results/scaling_curve")
    p.add_argument("--add", nargs=2, metavar=("LABEL", "JSON_PATH"),
                   action=AddPointAction, default=[],
                   help="Extra scale point, e.g. --add 8M data/results/ablation_8M/ablation_results.json")
    main(p.parse_args())

"""
One-time preprocessing: extract leading jets from LHCO QCD background,
save as a flat HDF5 with the same (N, max_constituents, N_FEATURES) layout
as AspenOpenJets so the same training script can load either.

Usage:
    python preprocess_lhco_bg.py \
        --lhco data/lhco/events_anomalydetection_v2.h5 \
        --out  data/lhco/lhco_bg_jets.h5 \
        --max_jets 1000000
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))


import argparse
import numpy as np
import h5py
from tqdm import tqdm

from src.data.lhco_loader import LHCOJetDataset
from src.data.aspen_loader import N_FEATURES


def main(args):
    print(f"Loading LHCO background from {args.lhco} ...")
    ds = LHCOJetDataset(args.lhco, is_signal=False, max_jets=args.max_jets,
                        max_constituents=args.max_constituents)
    N = len(ds)
    print(f"Preprocessing {N:,} jets → {args.out}")

    with h5py.File(args.out, "w") as f:
        dset = f.create_dataset(
            "constituents",
            shape=(N, args.max_constituents, N_FEATURES),
            dtype="float32",
            chunks=(1024, args.max_constituents, N_FEATURES),
            compression="lzf",
        )
        mask_dset = f.create_dataset(
            "mask",
            shape=(N, args.max_constituents),
            dtype="bool",
            chunks=(1024, args.max_constituents),
            compression="lzf",
        )

        batch = 4096
        for start in tqdm(range(0, N, batch)):
            end = min(start + batch, N)
            xs    = np.stack([ds[i][0].numpy() for i in range(start, end)])
            masks = np.stack([ds[i][1].numpy() for i in range(start, end)])
            dset[start:end]      = xs
            mask_dset[start:end] = masks

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lhco", default="data/lhco/events_anomalydetection_v2.h5")
    p.add_argument("--out",  default="data/lhco/lhco_bg_jets.h5")
    p.add_argument("--max_jets", type=int, default=1_000_000)
    p.add_argument("--max_constituents", type=int, default=50)
    args = p.parse_args()
    main(args)

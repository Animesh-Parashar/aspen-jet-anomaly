"""
One-time preprocessing: process an AspenOpenJets HDF5 batch into a flat
(N, max_constituents, N_FEATURES) float16 array for fast cached training.

Stores as float16 (halves RAM/disk vs float32), converted back at load time.
Run once per batch file before training.

Usage:
    python preprocess_aspen.py \
        --input  data/aspen/RunG_batch0.h5 \
        --output data/aspen/RunG_batch0_processed.h5
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))


import argparse
import numpy as np
import h5py
from tqdm import tqdm

from src.data.aspen_loader import _to_relative, N_FEATURES, _PX, _PY, _JET_ETA, _JET_PHI


def process_batch(pf_batch, jk_batch, max_constituents):
    """Process a chunk of jets. pf_batch: (B, 150, 11), jk_batch: (B, 4)."""
    B = len(pf_batch)
    out_x    = np.zeros((B, max_constituents, N_FEATURES), dtype=np.float16)
    out_mask = np.ones((B, max_constituents), dtype=bool)

    for i in range(B):
        consts = pf_batch[i].astype(np.float32)
        jet_eta = float(jk_batch[i, _JET_ETA])
        jet_phi = float(jk_batch[i, _JET_PHI])

        pT = np.sqrt(consts[:, _PX]**2 + consts[:, _PY]**2)
        valid = pT > 0
        consts = consts[valid]

        if len(consts) == 0:
            consts = np.zeros((1, 11), dtype=np.float32)

        feats = _to_relative(consts, jet_eta, jet_phi)
        N = min(len(feats), max_constituents)
        out_x[i, :N] = feats[:N].astype(np.float16)
        out_mask[i, :N] = False

    return out_x, out_mask


def main(args):
    with h5py.File(args.input, "r") as f:
        n_total = f["jet_kinematics"].shape[0]
    n = n_total if args.max_jets is None else min(args.max_jets, n_total)
    print(f"Processing {n:,} jets from {args.input}")

    chunk = args.chunk_size
    with h5py.File(args.output, "w") as out:
        x_ds = out.create_dataset(
            "constituents", shape=(n, args.max_constituents, N_FEATURES),
            dtype="float16",
            chunks=(min(4096, n), args.max_constituents, N_FEATURES),
            compression="lzf",
        )
        m_ds = out.create_dataset(
            "mask", shape=(n, args.max_constituents),
            dtype="bool",
            chunks=(min(4096, n), args.max_constituents),
            compression="lzf",
        )
        out.attrs["n_jets"]           = n
        out.attrs["max_constituents"] = args.max_constituents
        out.attrs["n_features"]       = N_FEATURES
        out.attrs["source"]           = args.input

        with h5py.File(args.input, "r") as src:
            for start in tqdm(range(0, n, chunk), desc="Preprocessing"):
                end = min(start + chunk, n)
                pf  = src["PFCands"][start:end]
                jk  = src["jet_kinematics"][start:end]
                xs, masks = process_batch(pf, jk, args.max_constituents)
                x_ds[start:end] = xs
                m_ds[start:end] = masks

    size_gb = h5py.File(args.output, "r")["constituents"].id.get_storage_size() / 1e9
    print(f"Saved {args.output}  ({size_gb:.2f} GB compressed)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/aspen/RunG_batch0.h5")
    p.add_argument("--output", default="data/aspen/RunG_batch0_processed.h5")
    p.add_argument("--max_jets", type=int, default=None)
    p.add_argument("--max_constituents", type=int, default=50)
    p.add_argument("--chunk_size", type=int, default=8192)
    args = p.parse_args()
    main(args)



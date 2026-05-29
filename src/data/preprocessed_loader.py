"""
Fast loader for preprocessed jet HDF5 files (output of preprocess_lhco_bg.py
or any file with 'constituents' and 'mask' datasets).
Used for Model B2 (LHCO sim baseline) and JetClass (Model B1).
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class PreprocessedJetDataset(Dataset):
    def __init__(self, hdf5_path, max_jets=None):
        self.path = Path(hdf5_path)
        with h5py.File(self.path, "r") as f:
            n_total = f["constituents"].shape[0]
        self.n_jets = n_total if max_jets is None else min(max_jets, n_total)

    def __len__(self):
        return self.n_jets

    def __getitem__(self, idx):
        with h5py.File(self.path, "r") as f:
            x    = torch.from_numpy(f["constituents"][idx])   # (N, F)
            mask = torch.from_numpy(f["mask"][idx].astype(bool))  # (N,)
        return x, mask


class PreprocessedJetDatasetCached(Dataset):
    """
    Loads entire dataset into RAM at startup — zero I/O during training.
    Supports float16 storage (converted to float32 on __getitem__).
    Safe for large num_workers: data is shared copy-on-write across worker forks.
    """
    def __init__(self, hdf5_path, max_jets=None):
        with h5py.File(hdf5_path, "r") as f:
            n_total = f["constituents"].shape[0]
            n = n_total if max_jets is None else min(max_jets, n_total)
            print(f"[PreprocessedCached] Loading {n:,} jets into RAM...", flush=True)
            # Store as numpy (shared across DataLoader workers via fork COW)
            raw = f["constituents"][:n]   # float16 or float32
            self._x    = raw.astype(np.float32)   # always float32 in RAM
            self._mask = f["mask"][:n].astype(bool)
        ram_gb = (self._x.nbytes + self._mask.nbytes) / 1e9
        print(f"[PreprocessedCached] Ready. RAM used: {ram_gb:.2f} GB", flush=True)

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        return torch.from_numpy(self._x[idx]), torch.from_numpy(self._mask[idx])


def make_dataloader(hdf5_path, batch_size=512, max_jets=None,
                    num_workers=4, shuffle=True, cached=False):
    cls = PreprocessedJetDatasetCached if cached else PreprocessedJetDataset
    ds  = cls(hdf5_path, max_jets=max_jets)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers if not cached else 0,
                      pin_memory=True)

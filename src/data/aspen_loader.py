"""
AspenOpenJets data loader.
Dataset: arXiv:2412.10504, doi:10.25592/uhhfdm.16505
~178M CMS 2016 jets, up to 150 constituents each.

Confirmed HDF5 schema (RunG_batch0.h5):
  jet_kinematics  (N, 4)      — [pT, eta, phi, softdrop_mass]
  PFCands         (N, 150, 11)— [px, py, pz, E, d0, dz, d0_err, dz_err, charge, pdgId, flag]
  jet_tagging     (N, 13)     — n_constituents + tagger scores (not used here)
  event_info      (N, 3)      — [run, lumi, event] (not used)

Padding rows in PFCands are all-zero.
Neutral particles (photons, K0L) have d0=dz=d0_err=dz_err=-1.0 (no track).
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# PFCands column indices
_PX, _PY, _PZ, _E = 0, 1, 2, 3
_D0, _DZ, _D0ERR, _DZERR = 4, 5, 6, 7
_CHARGE, _PDGID, _FLAG = 8, 9, 10

# jet_kinematics column indices
_JET_PT, _JET_ETA, _JET_PHI, _JET_MASS = 0, 1, 2, 3


def _to_relative(consts, jet_eta, jet_phi):
    """
    Build the feature vector for each constituent:
      [eta_rel, phi_rel, log_pT_rel, log_E, d0, dz, charge]   → 7 features

    d0/dz are zeroed for neutral particles (where stored as -1 sentinel).
    """
    px = consts[:, _PX]
    py = consts[:, _PY]
    pz = consts[:, _PZ]
    E  = consts[:, _E]

    pT = np.sqrt(px**2 + py**2) + 1e-8
    eta = np.arcsinh(pz / pT)
    phi = np.arctan2(py, px)

    eta_rel = eta - jet_eta
    dphi = phi - jet_phi
    phi_rel = (dphi + np.pi) % (2 * np.pi) - np.pi

    jet_pT_sum = pT.sum() + 1e-8
    log_pT_rel = np.log(pT / jet_pT_sum + 1e-8)
    log_E      = np.log(E + 1e-8)

    # Sanitize impact parameters:
    # - Neutrals stored with sentinel -1.0 → zero (photons, K_L have no track)
    # - Inf values (rare HDF5 artifacts) → zero
    # - Clip to ±5 cm (captures secondary vertices, discards pileup outliers)
    # NOTE: use exact sentinel check (abs(d0+1)<0.05), not d0<-0.5, which would
    # incorrectly zero real displaced tracks from B/D-meson decays at large |d0|.
    d0 = consts[:, _D0].copy()
    dz = consts[:, _DZ].copy()
    d0[~np.isfinite(d0)] = 0.0
    dz[~np.isfinite(dz)] = 0.0
    d0[np.abs(d0 + 1.0) < 0.05] = 0.0   # sentinel: -1.0 = no track (neutral)
    dz[np.abs(dz + 1.0) < 0.05] = 0.0
    d0 = np.clip(d0, -5.0, 5.0)
    dz = np.clip(dz, -5.0, 5.0)

    charge = consts[:, _CHARGE]

    return np.column_stack([eta_rel, phi_rel, log_pT_rel, log_E, d0, dz, charge]).astype(np.float32)


# Number of features output by _to_relative
N_FEATURES = 7


class AspenJetDataset(Dataset):
    """
    Loads jets from a single AspenOpenJets HDF5 file.
    Each item is (x, key_padding_mask):
      x:    (max_constituents, N_FEATURES) float32
      mask: (max_constituents,) bool — True = padding token (ignored by attention)
    """

    def __init__(self, hdf5_path, max_jets=None, max_constituents=50):
        self.path = Path(hdf5_path)
        self.max_constituents = max_constituents

        with h5py.File(self.path, "r") as f:
            n_total = f["jet_kinematics"].shape[0]
        self.n_jets = n_total if max_jets is None else min(max_jets, n_total)

    def __len__(self):
        return self.n_jets

    def __getitem__(self, idx):
        with h5py.File(self.path, "r") as f:
            jk     = f["jet_kinematics"][idx]   # (4,)
            consts = f["PFCands"][idx]           # (150, 11)

        jet_eta = float(jk[_JET_ETA])
        jet_phi = float(jk[_JET_PHI])

        consts = np.array(consts, dtype=np.float32)

        # Padding rows are all-zero; detect by pT
        pT = np.sqrt(consts[:, _PX]**2 + consts[:, _PY]**2)
        valid = pT > 0
        consts = consts[valid]

        if len(consts) == 0:
            consts = np.zeros((1, 11), dtype=np.float32)

        feats = _to_relative(consts, jet_eta, jet_phi)  # (N_valid, N_FEATURES)

        N = min(len(feats), self.max_constituents)
        out = np.zeros((self.max_constituents, N_FEATURES), dtype=np.float32)
        out[:N] = feats[:N]

        key_padding_mask = np.ones(self.max_constituents, dtype=bool)
        key_padding_mask[:N] = False  # False = real token

        return torch.from_numpy(out), torch.from_numpy(key_padding_mask)


class AspenJetDatasetCached(Dataset):
    """
    Memory-mapped version: loads the full PFCands and jet_kinematics arrays
    into RAM for fast random access (requires ~30 GB RAM per full batch).
    Use for training where repeated random access is needed.
    """

    def __init__(self, hdf5_path, max_jets=None, max_constituents=50):
        self.max_constituents = max_constituents

        with h5py.File(hdf5_path, "r") as f:
            n_total = f["jet_kinematics"].shape[0]
            n = n_total if max_jets is None else min(max_jets, n_total)
            print(f"[AspenCached] Loading {n:,} jets into RAM...")
            self.jk     = f["jet_kinematics"][:n]   # (n, 4)
            self.consts = f["PFCands"][:n]           # (n, 150, 11)
        self.n_jets = n
        print(f"[AspenCached] Done. RAM usage: ~{self.consts.nbytes / 1e9:.1f} GB")

    def __len__(self):
        return self.n_jets

    def __getitem__(self, idx):
        jk     = self.jk[idx]
        consts = self.consts[idx].astype(np.float32)

        jet_eta = float(jk[_JET_ETA])
        jet_phi = float(jk[_JET_PHI])

        pT = np.sqrt(consts[:, _PX]**2 + consts[:, _PY]**2)
        valid = pT > 0
        consts = consts[valid]

        if len(consts) == 0:
            consts = np.zeros((1, 11), dtype=np.float32)

        feats = _to_relative(consts, jet_eta, jet_phi)

        N = min(len(feats), self.max_constituents)
        out = np.zeros((self.max_constituents, N_FEATURES), dtype=np.float32)
        out[:N] = feats[:N]

        key_padding_mask = np.ones(self.max_constituents, dtype=bool)
        key_padding_mask[:N] = False

        return torch.from_numpy(out), torch.from_numpy(key_padding_mask)


def make_dataloader(hdf5_path, batch_size=256, max_jets=None, max_constituents=50,
                    num_workers=4, shuffle=True, cached=False):
    cls = AspenJetDatasetCached if cached else AspenJetDataset
    ds = cls(hdf5_path, max_jets=max_jets, max_constituents=max_constituents)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)

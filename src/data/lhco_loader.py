"""
LHC Olympics 2020 R&D dataset loader.
Source: arXiv:2101.08320, zenodo.org/records/6466204

HDF5 format (pandas store, key='df'):
  (N_events, 2101) — 700 particles × 3 features (pT, eta, phi) + 1 label column
  label: 0 = QCD background, 1 = Z'→XY signal

Particles are NOT sorted by pT; zero-pT rows are padding.
The main events_anomalydetection_v2.h5 has a zero-pT divider particle
separating jet1 and jet2. The qqq signal file has no divider.

Strategy: extract the leading jet via ΔR clustering around the highest-pT particle.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.data.aspen_loader import N_FEATURES  # must match encoder input_dim=7


def _delta_r(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((eta1 - eta2)**2 + dphi**2)


def _extract_leading_jet(particles, jet_radius=0.8):
    """
    From a (700, 3) array of (pT, eta, phi), extract the constituents
    of the leading jet (highest-sum-pT cluster within jet_radius).
    Returns (N_const, 3) array.
    """
    valid = particles[:, 0] > 0
    pts = particles[valid]          # (N_valid, 3)
    if len(pts) == 0:
        return np.zeros((1, 3), dtype=np.float32)

    # Seed: highest-pT particle
    seed = pts[np.argmax(pts[:, 0])]
    seed_eta, seed_phi = seed[1], seed[2]

    dr = _delta_r(pts[:, 1], pts[:, 2], seed_eta, seed_phi)
    jet_mask = dr < jet_radius
    jet_consts = pts[jet_mask]

    if len(jet_consts) == 0:
        jet_consts = pts[:1]

    return jet_consts.astype(np.float32)


def _to_relative(consts_ptetaphi):
    """
    Build relative feature vector matching Aspen loader output (N_FEATURES=7).
    Input cols: (pT, eta, phi). No impact parameter info in LHCO.
    Output: (eta_rel, phi_rel, log_pT_rel, log_E≈log_pT, d0=0, dz=0, charge=0)
    """
    pT  = consts_ptetaphi[:, 0]
    eta = consts_ptetaphi[:, 1]
    phi = consts_ptetaphi[:, 2]

    jet_pT = pT.sum() + 1e-8
    jet_eta = (pT * eta).sum() / jet_pT
    jet_phi_sin = np.sum(pT * np.sin(phi)) / jet_pT
    jet_phi_cos = np.sum(pT * np.cos(phi)) / jet_pT
    jet_phi = np.arctan2(jet_phi_sin, jet_phi_cos)

    eta_rel = eta - jet_eta
    dphi = phi - jet_phi
    phi_rel = (dphi + np.pi) % (2 * np.pi) - np.pi
    log_pT_rel = np.log(pT / jet_pT + 1e-8)
    # log_E uses pT as proxy for energy (ignores cosh(η) factor). This is deliberate:
    # including cosh(η) introduces η-redundancy with feature 0 (η_rel) and makes background
    # heterogeneous (QCD jets span wider |η| than boosted signal), which inverts the
    # anomaly score on the LHCO benchmark. log(pT) is the right choice for this setup.
    log_E = np.log(pT + 1e-8)

    n = len(pT)
    zeros = np.zeros(n, dtype=np.float32)

    return np.column_stack([
        eta_rel.astype(np.float32),
        phi_rel.astype(np.float32),
        log_pT_rel.astype(np.float32),
        log_E.astype(np.float32),
        zeros,   # d0 (not available)
        zeros,   # dz (not available)
        zeros,   # charge (not available)
    ])


class LHCOJetDataset(Dataset):
    """
    Loads jets from LHCO R&D HDF5 (pandas store format).

    is_signal=True  → label==1 events (Z' signal)
    is_signal=False → label==0 events (QCD background)
    """

    def __init__(self, hdf5_path, is_signal=False, max_jets=None,
                 max_constituents=50, jet_radius=0.8):
        self.max_constituents = max_constituents
        self.jet_radius = jet_radius

        df = pd.read_hdf(hdf5_path, key="df")
        arr = df.values                             # (N, 2101)
        labels = arr[:, -1].astype(int)
        particles = arr[:, :-1].reshape(-1, 700, 3) # (N, 700, 3)

        sel = labels == int(is_signal)
        self.particles = particles[sel].astype(np.float32)

        if max_jets is not None:
            self.particles = self.particles[:max_jets]

        tag = "signal" if is_signal else "background"
        print(f"[LHCOLoader] {hdf5_path}: loaded {len(self.particles):,} {tag} events")

    def __len__(self):
        return len(self.particles)

    def __getitem__(self, idx):
        jet_consts = _extract_leading_jet(self.particles[idx], self.jet_radius)
        feats = _to_relative(jet_consts)            # (N_const, N_FEATURES)

        N = min(len(feats), self.max_constituents)
        out = np.zeros((self.max_constituents, N_FEATURES), dtype=np.float32)
        out[:N] = feats[:N]

        key_padding_mask = np.ones(self.max_constituents, dtype=bool)
        key_padding_mask[:N] = False

        return torch.from_numpy(out), torch.from_numpy(key_padding_mask)


def make_injection_dataset(bg_dataset, sig_dataset, n_bg=10000, n_sig=1000, seed=42):
    """
    Mix background and signal jets. Returns (jets, masks, labels).
    labels: 0=background, 1=signal.
    """
    rng = np.random.default_rng(seed)

    n_bg = min(n_bg, len(bg_dataset))
    n_sig = min(n_sig, len(sig_dataset))
    bg_idx = rng.choice(len(bg_dataset), size=n_bg, replace=False)
    sig_idx = rng.choice(len(sig_dataset), size=n_sig, replace=False)

    bg_jets  = torch.stack([bg_dataset[i][0] for i in bg_idx])
    sig_jets = torch.stack([sig_dataset[i][0] for i in sig_idx])
    bg_masks  = torch.stack([bg_dataset[i][1] for i in bg_idx])
    sig_masks = torch.stack([sig_dataset[i][1] for i in sig_idx])

    jets   = torch.cat([bg_jets, sig_jets], dim=0)
    masks  = torch.cat([bg_masks, sig_masks], dim=0)
    labels = torch.cat([torch.zeros(n_bg), torch.ones(n_sig)]).long()

    return jets, masks, labels

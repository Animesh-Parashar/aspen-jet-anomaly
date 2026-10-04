"""Explicit v2 representation. Historical loaders remain unchanged.

Both domains use unweighted constituent momenta and a summed-momentum axis,
computed BEFORE truncation. Aspen starts from the producer's stored AK8
constituents (at most 150), not from a re-clustered full collision event.
LHCO starts from massless event particles and uses FastJet anti-kt, R=0.8.
This matches feature definitions, not detector response or event selection.
"""
import hashlib
import struct

import numpy as np

FEATURE_NAMES = ['eta_rel', 'phi_rel', 'log_pt_fraction', 'log_pt_GeV']
SCHEMA = 'jet-preprocessing-v2.0'


def event_bucket(namespace, event_id, seed=20260925):
    payload = namespace.encode() + b'\0' + struct.pack('<Q', seed)
    payload += b''.join(struct.pack('<q', int(value)) for value in event_id)
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, 'little') % 10000


def partition(namespace, event_id, label=-1, seed=20260925, development=False):
    """Membership depends on event identity, never on file order or chunk size.

    Aspen labels are unknown (-1). LHCO signal is evaluation-only. Previously inspected events
    are forced into validation, including every Aspen jet sharing their IDs.
    """
    if label not in (-1, 0, 1):
        raise ValueError('Unexpected label')
    if development:
        return 'validation'
    bucket = event_bucket(namespace, event_id, seed)
    if label == 1:
        return 'validation' if bucket < 5000 else 'test'
    if bucket < 6000:
        return 'train'
    if bucket < 7500:
        return 'validation'
    if bucket < 8500:
        return 'reference'
    return 'test'


def leading_lhco_constituents(particles, radius=0.8):
    """Returns descending-pT (px,py,pz,E) constituents of the leading AK jet."""
    import fastjet
    particles = np.asarray(particles, dtype=np.float64)
    if not np.isfinite(particles).all() or (particles[:, 0] < 0).any():
        raise ValueError('invalid_lhco_particles')
    particles = particles[particles[:, 0] > 0]
    if not len(particles):
        raise ValueError('empty_event')
    inputs = []
    for pt, eta, phi in particles:
        inputs.append(fastjet.PseudoJet(float(pt * np.cos(phi)), float(pt * np.sin(phi)),
                                       float(pt * np.sinh(eta)), float(pt * np.cosh(eta))))
    sequence = fastjet.ClusterSequence(inputs, fastjet.JetDefinition(fastjet.antikt_algorithm, radius))
    leading = fastjet.sorted_by_pt(sequence.inclusive_jets())[0]
    return np.array([[c.px(), c.py(), c.pz(), c.E()]
                     for c in fastjet.sorted_by_pt(leading.constituents())])


def represent(constituents, max_constituents=50, detector=False):
    """Construct shared features, masks and diagnostics without using labels."""
    if max_constituents < 1:
        raise ValueError('max_constituents must be positive')
    c = np.asarray(constituents, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] < (11 if detector else 4):
        raise ValueError('invalid_constituent_shape')
    if not np.isfinite(c[:, :4]).all() or (c[:, 3] < 0).any():
        raise ValueError('invalid_four_vectors')
    pt = np.hypot(c[:, 0], c[:, 1])
    c = c[pt > 0]
    pt = pt[pt > 0]
    if not len(c):
        raise ValueError('empty_jet')
    order = np.argsort(-pt, kind='stable')
    c, pt = c[order], pt[order]
    px, py, pz, energy = c[:, :4].sum(axis=0)
    jet_pt = np.hypot(px, py)
    if jet_pt <= 0:
        raise ValueError('undefined_jet_axis')
    jet_eta, jet_phi = np.arcsinh(pz / jet_pt), np.arctan2(py, px)
    eta = np.arcsinh(c[:, 2] / pt)
    phi = np.arctan2(c[:, 1], c[:, 0])
    features = np.column_stack((eta - jet_eta, (phi - jet_phi + np.pi) % (2*np.pi) - np.pi,
                                np.log(pt / pt.sum()), np.log(pt)))
    n = min(len(c), max_constituents)
    x = np.zeros((max_constituents, 4), dtype=np.float32)
    x[:n] = features[:n]
    mask = np.arange(max_constituents) >= n
    if not np.isfinite(x).all():
        raise ValueError('nonfinite_features')
    result = dict(constituents=x, mask=mask, n_constituents=np.int32(len(c)),
                  retained_pt_fraction=np.float32(pt[:n].sum() / pt.sum()),
                  jet_kinematics=np.array([jet_pt, jet_eta, jet_phi,
                       np.sqrt(max(0., energy**2-px**2-py**2-pz**2))], dtype=np.float32))
    if detector:
        # Producer schema: d0, d0Err, dz, dzErr at columns 4,5,6,7.
        # Missing tracks are marked via charge/uncertainty, not by deleting a
        # neighborhood around a physical impact parameter of -1.
        aux = np.zeros((max_constituents, 3), dtype=np.float32)
        observed = np.zeros((max_constituents, 3), dtype=bool)
        for out_col, value_col, error_col in [(0, 4, 5), (1, 6, 7)]:
            valid = ((c[:n, 8] != 0) & np.isfinite(c[:n, 8]) &
                     np.isfinite(c[:n, value_col]) & np.isfinite(c[:n, error_col]) &
                     (c[:n, error_col] >= 0))
            aux[:n, out_col] = np.where(valid, np.clip(c[:n, value_col], -5, 5), 0)
            observed[:n, out_col] = valid
        valid_charge = np.isfinite(c[:n, 8])
        aux[:n, 2] = np.where(valid_charge, c[:n, 8], 0)
        observed[:n, 2] = valid_charge
        result.update(detector_features=aux, detector_observed=observed)
    return result

"""Bounded, read-only raw-data audit; no training or preprocessing mutations.

Requires numpy, h5py, hdf5plugin and fastjet. Samples reproducible contiguous
blocks across each file; sample statistics are not full-dataset measurements.
"""
import argparse
import json
import sys
from pathlib import Path

import hdf5plugin  # Registers the Blosc filter used by LHCO.
import h5py
import numpy as np
import fastjet


def summary(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return None if not len(x) else dict(
        mean=float(x.mean()), p05=float(np.quantile(x, .05)),
        median=float(np.median(x)), p95=float(np.quantile(x, .95)))


def block_indices(n, blocks, width, seed):
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n, blocks + 1, dtype=int)
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        count = min(width, hi - lo)
        if count:
            start = int(rng.integers(lo, hi - count + 1))
            result.extend(range(start, start + count))
    return np.array(result, dtype=int)


def read_blocks(dataset, indices):
    # Contiguous reads avoid expensive HDF5 point selection on compressed files.
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return np.concatenate([dataset[g[0]:g[-1] + 1] for g in groups if len(g)])


def constituent_stats(pt):
    counts = (pt > 0).sum(axis=1)
    total = pt.sum(axis=1)
    sorted_pt = np.sort(pt, axis=1)[:, ::-1]
    return dict(
        jets=int(len(pt)), empty_jets=int((counts == 0).sum()),
        fraction_above_50=float((counts > 50).mean()),
        fraction_pt_ordered=float((np.diff(pt, axis=1) <= 1e-5).all(axis=1).mean()),
        multiplicity=summary(counts),
        first50_pt_fraction=summary(pt[:, :50].sum(axis=1) / np.maximum(total, 1e-12)),
        top50_pt_fraction=summary(sorted_pt[:, :50].sum(axis=1) / np.maximum(total, 1e-12)))


def audit_aspen(path, args):
    with h5py.File(path, 'r') as f:
        idx = block_indices(len(f['PFCands']), args.blocks, args.width, args.seed)
        c = read_blocks(f['PFCands'], idx)
        ids = read_blocks(f['event_info'], idx)
        pt = np.hypot(c[:, :, 0], c[:, :, 1])
        valid = pt > 0
        stats = constituent_stats(pt)
        _, event_counts = np.unique(ids, axis=0, return_counts=True)
        stats.update(total_jets=len(f['PFCands']), sampled_indices=idx.tolist(),
                     nonfinite_values=int((~np.isfinite(c)).sum()),
                     unique_sample_event_ids=int(len(event_counts)),
                     sample_rows_beyond_first_per_event=int(len(ids) - len(event_counts)),
                     sample_rows_in_repeated_events=int(event_counts[event_counts > 1].sum()))
        stats['columns_4_to_7'] = {
            str(i): summary(c[:, :, i][valid]) for i in range(4, 8)}
        energy = c[:, :, 3][valid]
        momentum = pt[valid]
        good = (energy > 0) & (momentum > 0)
        stats['log_energy_minus_log_pt'] = summary(np.log(energy[good] / momentum[good]))
        return stats


def audit_lhco(path, args):
    with h5py.File(path, 'r') as f:
        ds = f['df/block0_values']
        if not np.array_equal(f['df/block0_items'][:], np.arange(2101)):
            raise ValueError('Unexpected LHCO column order')
        idx = block_indices(len(ds), args.blocks, args.width, args.seed)
        raw = read_blocks(ds, idx)
        total = len(ds)
    labels = raw[:, -1]
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Unexpected LHCO labels')
    particles = raw[:, :-1].reshape(-1, 700, 3).astype(np.float32)
    output = dict(total_events=total, sampled_indices=idx.tolist(),
                  nonfinite_values=int((~np.isfinite(raw)).sum()), classes={})
    definition = fastjet.JetDefinition(fastjet.antikt_algorithm, .8)
    for label in (0, 1):
        selected = particles[labels == label]
        if not len(selected):
            output['classes'][str(label)] = {'sample_events': 0}
            continue
        cones, clustered, overlap, ratios = [], [], [], []
        for event in selected:
            p = event[event[:, 0] > 0]
            if not len(p):
                continue
            seed = p[p[:, 0].argmax()]
            dphi = (p[:, 2] - seed[2] + np.pi) % (2 * np.pi) - np.pi
            cone_ids = set(np.flatnonzero(np.hypot(p[:, 1] - seed[1], dphi) < .8).tolist())
            inputs = []
            for i, (pt, eta, phi) in enumerate(p):
                pt, eta, phi = float(pt), float(eta), float(phi)
                jet = fastjet.PseudoJet(pt * np.cos(phi), pt * np.sin(phi),
                                       pt * np.sinh(eta), pt * np.cosh(eta))
                jet.set_user_index(i)
                inputs.append(jet)
            sequence = fastjet.ClusterSequence(inputs, definition)
            leading = fastjet.sorted_by_pt(sequence.inclusive_jets())[0]
            jet_ids = {c.user_index() for c in leading.constituents()}
            overlap.append(len(cone_ids & jet_ids) / len(cone_ids | jet_ids))
            ratios.append(p[sorted(cone_ids), 0].sum() / p[sorted(jet_ids), 0].sum())
            a = np.zeros(700); a[:len(cone_ids)] = p[sorted(cone_ids), 0]
            b = np.zeros(700); b[:len(jet_ids)] = np.sort(p[sorted(jet_ids), 0])[::-1]
            cones.append(a); clustered.append(b)
        if not overlap:
            output['classes'][str(label)] = dict(sample_events=len(selected), compared_events=0)
            continue
        output['classes'][str(label)] = dict(
            sample_events=len(selected), compared_events=len(overlap),
            fraction_identical_membership=float((np.array(overlap) == 1).mean()),
            fraction_disjoint_membership=float((np.array(overlap) == 0).mean()),
            constituent_jaccard=summary(overlap), cone_over_antikt_scalar_pt=summary(ratios),
            legacy_cone=constituent_stats(np.asarray(cones)),
            antikt_sorted=constituent_stats(np.asarray(clustered)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=20)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20260925)
    args = parser.parse_args()
    if args.blocks < 1 or args.width < 1:
        parser.error('blocks and width must be positive')
    result = dict(scope='Sample audit, not full-dataset certification', seed=args.seed,
                  blocks=args.blocks, width=args.width,
                  versions=dict(python=sys.version, numpy=np.__version__,
                                h5py=h5py.__version__, fastjet=fastjet.__version__))
    result['input_verification'] = {}
    for name in ('aspen/RunG_batch0.h5', 'lhco/events_anomalydetection_v2.h5'):
        manifest = Path(str(args.data / name) + '.download.json')
        if manifest.exists():
            result['input_verification'][name] = json.loads(manifest.read_text())
    result['aspen'] = audit_aspen(args.data / 'aspen/RunG_batch0.h5', args)
    print('Aspen sample audited', flush=True)
    result['lhco'] = audit_lhco(args.data / 'lhco/events_anomalydetection_v2.h5', args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('Audit saved:', args.out, flush=True)


if __name__ == '__main__':
    main()

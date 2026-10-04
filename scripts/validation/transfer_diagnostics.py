"""Post-hoc support/representation diagnostics; no training or score selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.validation.representation_diagnostics import geometry
from scripts.validation.scientific_value_gate import observables
from scripts.validation.frozen_topology_eval import folder

PROTOCOL = Path('docs/transfer_diagnostics_protocol_20261003.md')
CACHE = Path('data/results/scorer_robustness_features_20261003')
PAPER = Path('manuscript/revision_20261002')
SEEDS = (17, 29, 43)
DOMAINS = ('aspen', 'lhco', 'random')
OBS = ('mass', 'pt', 'multiplicity', 'girth')
EDGES = dict(pt=np.geomspace(50, 10000, 65), mass=np.linspace(0, 2000, 65),
             multiplicity=np.linspace(0, 1000, 101), log_pt=np.linspace(-12, 10, 89))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def histogram(values, edges, weights=None):
    x, e = np.asarray(values, float), np.asarray(edges, float)
    w = np.ones(x.shape) if weights is None else np.asarray(weights, float)
    if (x.ndim != 1 or x.shape != w.shape or not len(x) or
            not np.isfinite(x).all() or not np.isfinite(w).all() or
            (w < 0).any() or w.sum() <= 0 or not np.all(np.diff(e) > 0)):
        raise ValueError('Invalid histogram inputs')
    counts = np.histogram(x, e, weights=w)[0]
    under, over = w[x < e[0]].sum(), w[x > e[-1]].sum()
    np.testing.assert_allclose(counts.sum()+under+over, w.sum(), rtol=1e-9)
    return dict(edges=e.tolist(), probability=(counts/w.sum()).tolist(),
                underflow=float(under/w.sum()), overflow=float(over/w.sum()))


def constituent_values(x, mask):
    x, mask = np.asarray(x), np.asarray(mask)
    if (x.ndim != 3 or x.shape[-1] != 4 or mask.shape != x.shape[:2] or
            mask.dtype != np.bool_ or mask.all(1).any() or not np.isfinite(x).all()):
        raise ValueError('Invalid core4 tensors')
    weights = np.broadcast_to(1/(~mask).sum(1)[:, None], mask.shape)
    return x[..., 3][~mask].astype(float), weights[~mask]


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2 or not np.isfinite([a, b]).all():
        raise ValueError('Invalid correlation inputs')
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def rotation_check(x, mask):
    constituent_values(x, mask)
    radius = np.where(mask, 0, np.hypot(x[..., 0].astype(float), x[..., 1]))
    max_radius = radius.max(1)
    a = np.random.default_rng(20261009).uniform(-np.pi, np.pi, len(x))
    phi = np.sin(a)[:, None]*x[..., 0] + np.cos(a)[:, None]*x[..., 1]
    rejected = ((abs(phi) > np.pi) & ~mask).any(1)
    return dict(max_local_radius=float(max_radius.max()),
                jets_potentially_rejecting=int((max_radius > np.pi).sum()),
                rejection_impossible_for_all_angles=bool((max_radius <= np.pi).all()),
                one_angle_mc_rejection_fraction=float(rejected.mean()), mc_seed=20261009)


def selected_data(root, domain, split, count, seed):
    root = Path(root)
    meta = json.loads((root/'COMPLETE.json').read_text())['metadata']
    if (json.loads((root/'VERIFIED.json').read_text())['status'] != 'passed' or
            meta['domain'] != domain or meta['schema'] != 'jet-preprocessing-v2.0' or
            meta['features'] != ['eta_rel', 'phi_rel', 'log_pt_fraction', 'log_pt_GeV']):
        raise ValueError('Unverified data')
    path = root/(split+'.h5' if domain == 'aspen' else 'background/'+split+'.h5')
    keys = ('constituents', 'mask', 'jet_kinematics', 'n_constituents', 'event_id', 'label')
    pieces = {k: [] for k in keys}
    with h5py.File(path) as f:
        n = len(f['event_id'])
        ix = np.sort(np.random.default_rng(seed).choice(n, count, replace=False))
        for start in range(0, n, 8192):
            sel = ix[(ix >= start) & (ix < start+8192)]-start
            if len(sel):
                for k in keys:
                    pieces[k].append(f[k][start:start+8192][sel])
    data = {k: np.concatenate(v) for k, v in pieces.items()}
    if not np.all(data['label'] == (-1 if domain == 'aspen' else 0)):
        raise ValueError('Wrong labels')
    return data, dict(path=str(path), n_available=n, n_selected=count, sample_seed=seed,
        selection_indices_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
        selected_event_ids_sha256=hashlib.sha256(data['event_id'].tobytes()).hexdigest(),
        preprocessing_manifest_sha256=digest(root/'COMPLETE.json'))


def support():
    samples, result, sources = {}, {}, {}
    for domain in ('aspen', 'lhco'):
        manifests = [json.loads((folder(s, domain)/'COMPLETE.json').read_text()) for s in SEEDS]
        train = manifests[0]['train']
        for s, m in zip(SEEDS, manifests):
            sources[str(folder(s, domain)/'COMPLETE.json')] = digest(folder(s, domain)/'COMPLETE.json')
            if (m['smoke'] or m['domain'] != domain or m['train'] != train or
                    m['config']['sample_seed'] != 20260925 or m['config']['train_jets'] != 100000):
                raise ValueError('Training selection mismatch')
        cfg = manifests[0]['config']
        a, provenance = selected_data(cfg[domain+'_root'], domain, 'train', 100000, cfg['sample_seed'])
        if provenance['n_available'] != train['n_available']:
            raise ValueError('Training population changed')
        name = domain+'_train'
        samples[name] = a
        result[name] = dict(provenance=provenance,
            histories={str(s): m['history'] for s, m in zip(SEEDS, manifests)})
    a, provenance = selected_data(cfg['lhco_root'], 'lhco', 'reference', 10000, 20261004)
    old = read_npz('data/results/frozen_final_two_26302.rachel.npz')
    np.testing.assert_array_equal(a['event_id'], old['neighbor_event_ids'])
    np.testing.assert_array_equal(observables(a), old['reference_observables'])
    samples['lhco_reference'] = a
    result['lhco_reference'] = dict(provenance=provenance)
    for name, a in samples.items():
        obs = observables(a)
        logs, weights = constituent_values(a['constituents'], a['mask'])
        jets = dict(pt=obs[:, 1], mass=obs[:, 0], multiplicity=obs[:, 2])
        hists = {k: histogram(v, EDGES[k]) for k, v in jets.items()}
        hists['log_pt'] = histogram(logs, EDGES['log_pt'], weights)
        hists['log_pt_particle_weighted'] = histogram(logs, EDGES['log_pt'])
        result[name].update(histograms=hists, n_jets=len(obs),
            quantiles={k: np.quantile(v, [0, .01, .05, .5, .95, .99, 1]).tolist() for k, v in jets.items()},
            rotation=rotation_check(a['constituents'], a['mask']),
            available_constituents_ge150_fraction=float((obs[:, 2] >= 150).mean()),
            retained_constituents_mean=float((~a['mask']).sum(1).mean()))
        print('SUPPORT', name, result[name]['quantiles']['pt'], flush=True)
    rpt = samples['lhco_reference']['jet_kinematics'][:, 0]
    lo, hi = np.quantile(rpt, [.05, .95])
    support_metrics = dict(reference_central90_pt_GeV=[float(lo), float(hi)], domains={})
    for name in ('aspen_train', 'lhco_train'):
        pt = samples[name]['jet_kinematics'][:, 0]
        low, high = np.quantile(pt, [.01, .99])
        support_metrics['domains'][name] = dict(
            training_fraction_in_reference_central90=float(((pt >= lo) & (pt <= hi)).mean()),
            training_count_in_reference_central90=int(((pt >= lo) & (pt <= hi)).sum()),
            training_central98_pt_GeV=[float(low), float(high)],
            reference_fraction_outside_training_central98=float(((rpt < low) | (rpt > high)).mean()))
    signal, balanced = {}, {}
    for t in ('two', 'three'):
        path = Path(f'data/results/frozen_final_{t}_26302.rachel.npz')
        sources[str(path)] = digest(path)
        a = read_npz(path)
        signal[t] = histogram(a['query_observables'][a['y'] == 1, 0], EDGES['mass'])
        jpath = path.with_suffix('.json'); sources[str(jpath)] = digest(jpath)
        m = json.loads(jpath.read_text())
        balanced[t] = [dict(b, low_pt_GeV=m['pt_edges'][b['bin']],
                              high_pt_GeV=m['pt_edges'][b['bin']+1])
                       for b in m['bin_counts'] if b['supported']]
    return dict(samples=result, support=support_metrics, signal_mass=signal,
                balanced_bins=balanced, source_sha256=sources,
                quantile_order=[0, .01, .05, .5, .95, .99, 1])


def representations():
    sources = {}
    for name in ('final_evidence.json', 'scorer_evidence.json'):
        for p, expected in json.loads((PAPER/name).read_text()).items():
            if digest(p) != expected:
                raise ValueError('Changed upstream evidence: '+p)
            sources[p] = expected
    manifest = json.loads((CACHE/'COMPLETE.json').read_text())
    path = Path('data/results/scorer_robustness_26312.rachel.npz')
    scores = read_npz(path); sources[str(path)] = digest(path)
    sources[str(CACHE/'COMPLETE.json')] = digest(CACHE/'COMPLETE.json')
    prior = {t: read_npz(f'data/results/frozen_final_{t}_26302.rachel.npz') for t in ('two', 'three')}
    for t, a in prior.items():
        np.testing.assert_array_equal(scores[t+'_query_event_ids'], a['query_event_ids'])
        np.testing.assert_array_equal(scores[t+'_neighbor_event_ids'], a['neighbor_event_ids'])
        np.testing.assert_array_equal(scores[t+'_y'], a['y'])
        np.testing.assert_array_equal(a['y'], np.r_[np.zeros(10000), np.ones(2000)])
    np.testing.assert_array_equal(prior['two']['query_event_ids'][:10000], prior['three']['query_event_ids'][:10000])
    np.testing.assert_array_equal(prior['two']['query_observables'][:10000], prior['three']['query_observables'][:10000])
    obs = prior['two']['query_observables'][:10000]
    result = {}
    for d in DOMAINS:
        for seed in SEEDS:
            name = f'{d}_seed{seed}'; p = CACHE/(name+'.npz')
            if digest(p) != manifest['files'][str(p)]:
                raise ValueError('Changed embeddings')
            sources[str(p)] = manifest['files'][str(p)]
            z = read_npz(p)
            if any(z[k].shape != (n, 128) or not np.isfinite(z[k]).all()
                   for k, n in [('reference', 10000), ('background', 10000), ('two', 2000), ('three', 2000)]):
                raise ValueError('Invalid cached shapes/values')
            g = {k: geometry(z[k]) for k in ('reference', 'background')}
            centered = z['reference'].astype(float)-z['reference'].mean(0, dtype=float)
            svd = np.linalg.svd(centered, compute_uv=False)**2/(len(centered)-1)
            np.testing.assert_allclose(svd, g['reference']['eigenvalues_descending'], atol=1e-11, rtol=1e-6)
            ix = np.arange(0, 10000, 625)
            direct = np.partition(cdist(z['background'][ix], z['reference']), 9, axis=1)[:, 9]
            np.testing.assert_allclose(direct, scores['two_'+name+'_raw'][ix], atol=1e-8, rtol=1e-8)
            rows = {}
            for scorer in ('raw', 'mahalanobis', 'white_knn'):
                a = scores['two_'+name+'_'+scorer]
                b = scores['three_'+name+'_'+scorer]
                np.testing.assert_array_equal(a[:10000], b[:10000])
                rows[scorer] = dict(background_spearman={k: correlation(a[:10000], obs[:, j]) for j, k in enumerate(OBS)},
                    score_quantiles={k: np.quantile(v, [.05, .5, .95]).tolist()
                                     for k, v in [('background', a[:10000]), ('two', a[10000:]), ('three', b[10000:])]})
            result[name] = dict(geometry=g, scorers=rows,
                direct_distance_max_error=float(abs(direct-scores['two_'+name+'_raw'][ix]).max()),
                svd_max_error=float(abs(svd-np.array(g['reference']['eigenvalues_descending'])).max()))
            print('REPRESENTATION', name, 'rank', g['reference']['entropy_rank'], flush=True)
    return dict(models=result, source_sha256=sources, background_queries=10000,
                reference_count=10000, score_quantile_order=[.05, .5, .95])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['support', 'representations'])
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite diagnostic evidence')
    result = support() if args.mode == 'support' else representations()
    result.update(stage='post_hoc_transfer_diagnostics', mode=args.mode,
        protocol_sha256=digest(PROTOCOL), script_sha256=digest(__file__),
        limitations=['Descriptive diagnostics specified after evaluation; no independent confirmation.',
                    'Associations and spectra do not establish causal mechanisms or physical densities.',
                    'Kinematic concentration is not strict support; detector and composition differences remain.'])
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print('JOB_COMPLETE', args.out, flush=True)

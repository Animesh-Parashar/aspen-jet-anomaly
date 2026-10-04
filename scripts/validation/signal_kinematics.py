"""Descriptive class-kinematics comparison from the frozen score arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROTOCOL = Path('docs/signal_kinematics_protocol_20261003.md')
EVIDENCE = Path('manuscript/revision_20261002/final_evidence.json')
FEATURES = {'pt': (1, np.arange(0., 5001., 100.)),
            'multiplicity': (2, np.arange(-.5, 300., 5.))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_vector(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    if (x.ndim != 1 or x.shape != w.shape or not len(x) or
            not np.isfinite(x).all() or not np.isfinite(w).all() or
            (w < 0).any() or w.sum() <= 0):
        raise ValueError('Invalid observations or weights')
    return x, w


def weighted_quantile(x, w, q=(.1, .5, .9)):
    x, w = validate_vector(x, w)
    keep = w > 0
    x, w = x[keep], w[keep]
    order = np.argsort(x, kind='stable'); x, w = x[order], w[order]
    q = np.asarray(q, float)
    if q.ndim != 1 or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError('Invalid quantiles')
    cumulative = np.cumsum(w)
    # Multiplication by the same accumulated total protects the q=1 endpoint.
    ix = np.searchsorted(cumulative, q*cumulative[-1], side='left')
    return x[np.minimum(ix, len(x)-1)]


def pair_auc(y, x, w):
    x, w = validate_vector(x, w); y = np.asarray(y)
    if y.shape != x.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected both binary classes')
    _, inverse = np.unique(x, return_inverse=True)
    bg = np.bincount(inverse[y == 0], weights=w[y == 0], minlength=inverse.max()+1)
    sig = np.bincount(inverse[y == 1], weights=w[y == 1], minlength=inverse.max()+1)
    if bg.sum() <= 0 or sig.sum() <= 0:
        raise ValueError('Empty weighted class')
    below = np.cumsum(bg)-bg
    return float(np.sum(sig*(below+.5*bg))/(sig.sum()*bg.sum()))


def distribution(x, w, edges):
    x, w = validate_vector(x, w)
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError('Invalid histogram edges')
    hist = np.histogram(x, edges, weights=w)[0]/w.sum()
    under, over = float(w[x < edges[0]].sum()/w.sum()), float(w[x > edges[-1]].sum()/w.sum())
    np.testing.assert_allclose(hist.sum()+under+over, 1, atol=1e-12, rtol=0)
    quantiles = weighted_quantile(x, w)
    for p, v in zip((.1, .5, .9), quantiles):
        assert w[x < v].sum()/w.sum() <= p+1e-12
        assert w[x <= v].sum()/w.sum() >= p-1e-12
    return dict(count=len(x), positive_weight_count=int((w > 0).sum()),
                effective_n=float(w.sum()**2/(w@w)), mean=float(np.average(x, weights=w)),
                quantiles_10_50_90=quantiles.tolist(), minimum=float(x[w > 0].min()),
                maximum=float(x[w > 0].max()), edges=edges.tolist(), probability=hist.tolist(),
                underflow=under, overflow=over)


def run(out):
    if out.exists():
        raise ValueError('Refusing to overwrite results')
    manifest = json.loads(EVIDENCE.read_text()); sources = {}; results = {}; backgrounds = []
    max_error = 0.
    for topology in ('two', 'three'):
        p = Path(f'data/results/frozen_final_{topology}_26302.rachel.npz')
        j = p.with_suffix('.json')
        for f in (p, j):
            if sha(f) != manifest[str(f)]:
                raise ValueError('Changed final evidence: '+str(f))
            sources[str(f)] = sha(f)
        with np.load(p, allow_pickle=False) as f:
            y = f['y']; obs = f['query_observables']; w = f['common_pt_weights']
            ids = f['query_event_ids']
        np.testing.assert_array_equal(y, np.r_[np.zeros(10000), np.ones(2000)])
        if obs.shape != (12000, 4) or not np.isfinite(obs).all() or (obs[:, [1, 2]] < 0).any():
            raise ValueError('Invalid observables')
        np.testing.assert_array_equal(obs[:, 2], np.floor(obs[:, 2]))
        backgrounds.append((ids[y == 0], obs[y == 0]))
        result = {}
        for population, weights in [('original', np.ones(len(y))), ('common_pt', w)]:
            per = {}
            for name, (column, edges) in FEATURES.items():
                x = obs[:, column]
                auc = roc_auc_score(y, x, sample_weight=weights)
                error = abs(auc-pair_auc(y, x, weights)); max_error = max(error, max_error)
                np.testing.assert_allclose(error, 0, atol=1e-12)
                per[name] = dict(background=distribution(x[y == 0], weights[y == 0], edges),
                                 signal=distribution(x[y == 1], weights[y == 1], edges),
                                 increasing_observable_auc=float(auc))
            result[population] = per
        m = json.loads(j.read_text())
        result['raw_auc_by_seed'] = {d: {str(s): {k: m['results'][f'{d}_seed{s}'][k]
            for k in ('original_auc', 'common_pt_auc')} for s in (17,29,43)}
            for d in ('aspen','lhco','random')}
        results[topology] = result
        print(topology, {k: {'bg': v['background']['quantiles_10_50_90'],
                             'signal': v['signal']['quantiles_10_50_90']}
                         for k, v in result['original'].items()}, flush=True)
    for a, b in zip(backgrounds[0], backgrounds[1]): np.testing.assert_array_equal(a, b)
    output = dict(stage='post_evaluation_descriptive_signal_kinematics', results=results,
        source_sha256=sources, protocol_sha256=sha(PROTOCOL), script_sha256=sha(__file__),
        checks=dict(shared_background=True, weighted_histograms_and_quantiles='passed',
                    independent_auc_max_error=max_error),
        limitations=['Marginal associations do not establish a causal score decomposition.',
                     'Original and weighted samples are the existing evaluated populations.',
                     'Multiplicity counts all available constituents before the 50-particle cap.'])
    with out.open('x') as f: json.dump(output, f, indent=2, allow_nan=False)
    print('JOB_COMPLETE', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    run(p.parse_args().out)

"""Frozen three-seed two-prong DEVELOPMENT evaluation; no final partitions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.validation.scientific_value_gate import load_validation, observables, encode, metrics
from scripts.validation.gate_crosscheck import common_weights, rank_auc
from src.models.controlled_encoder import JetBackbone, ContrastiveJetModel

SEEDS = (17, 29, 43)
CACHE = Path('data/results/scientific_value_gate_26199.rachel.npz')
ROOT = Path('data/processed_v2/full_24632.rachel/lhco')
PROTOCOL = Path('docs/revision_replication_evaluation.md')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def folder(seed, domain):
    if seed == 17:
        return Path('data/results/convergence_extensions_26172.rachel') / (domain+'_contrastive')
    base = 'revision_replication_26277.rachel' if seed == 29 else 'revision_seed43_retry_26292.rachel'
    return Path('data/results') / base / f'seed{seed}_{domain}'


def weighted_points(y, s, w):
    y, s, w = np.asarray(y), np.asarray(s), np.asarray(w)
    if not (y.shape == s.shape == w.shape) or not np.isfinite(s).all() or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Invalid scores/weights')
    if set(np.unique(y)) != {0, 1} or any(w[y == c].sum() <= 0 for c in (0, 1)):
        raise ValueError('Missing weighted class')
    si = np.flatnonzero((y == 1) & (w > 0))
    si = si[np.argsort(-s[si], kind='stable')]
    cumulative = np.cumsum(w[si]) / w[si].sum()
    rows = []
    for target in (.3, .5):
        threshold = s[si[np.searchsorted(cumulative, target, side='left')]]
        passed = s >= threshold
        eff = [float(w[(y == c) & passed].sum() / w[y == c].sum()) for c in (0, 1)]
        rows.append(dict(target_signal_efficiency=target, threshold=float(threshold),
                         achieved_signal_efficiency=eff[1], background_efficiency=eff[0],
                         rejection=1/eff[0] if eff[0] else None))
    return rows


def summarize(values):
    a = np.asarray(values, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('Expected three finite seed metrics')
    return dict(per_seed=a.tolist(), mean=float(a.mean()), sample_sd=float(a.std(ddof=1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix('.npz').exists():
        raise ValueError('Refusing overwrite')
    if not torch.cuda.is_available():
        raise RuntimeError('Scheduler GPU required')
    torch.set_num_threads(1)
    with np.load(CACHE, allow_pickle=False) as f:
        old = {k: f[k] for k in f.files}
    bg = load_validation(ROOT, 'background', 20000)
    sig = load_validation(ROOT, 'signal', 2000)
    order = np.random.default_rng(1701).permutation(20000)
    ref = {k: v[order[:10000]] for k, v in bg.items()}
    query = {k: np.concatenate([v[order[10000:]], sig[k]]) for k, v in bg.items()}
    y = query['label'].astype(int)
    for actual, expected in [(y, old['y']), (ref['event_id'], old['neighbor_event_ids']),
                             (query['event_id'], old['query_event_ids']),
                             (observables(ref), old['reference_observables']),
                             (observables(query), old['query_observables'])]:
        np.testing.assert_array_equal(actual, expected)
    rid, qid = set(map(tuple, ref['event_id'])), set(map(tuple, query['event_id']))
    if rid & qid or len(rid) != 10000 or len(qid) != 12000:
        raise ValueError('Duplicate/overlapping event identities')
    edges = np.unique(np.quantile(old['reference_observables'][:, 1], np.linspace(0, 1, 21)))
    w, bins = common_weights(y, old['query_observables'][:, 1], edges)
    scores = {k: old[k] for k in ['mass_high', 'pt_high', 'multiplicity_high', 'multiplicity_low', 'girth_high', 'observable_density']}
    provenance, audits = {}, {}
    base = None
    data_provenance = {}
    for seed in SEEDS:
        initial = None
        cfg = None
        for domain in ('aspen', 'lhco'):
            p = folder(seed, domain)
            m = json.loads((p/'COMPLETE.json').read_text())
            cfg = m['config']
            reduced = {k: v for k, v in cfg.items() if k != 'training_seed'}
            if base is not None and reduced != base:
                raise ValueError('Unmatched training settings')
            base = reduced
            if (cfg['training_seed'] != seed or cfg['epochs'] != 100 or m['smoke'] or
                m['domain'] != domain or m['objective'] != 'contrastive' or
                [h['epoch'] for h in m['history']] != list(range(1, 101))):
                raise ValueError('Wrong training endpoint/task')
            if not all(np.isfinite(h[k]) for h in m['history'] for k in ('train_loss', 'validation_loss')):
                raise ValueError('Nonfinite training history')
            for src, sha in m['source_sha256'].items():
                if digest(src) != sha:
                    raise ValueError('Training source changed: '+src)
            prov = [m['train'], m['validation']]
            if domain in data_provenance and prov != data_provenance[domain]:
                raise ValueError('Training sample provenance differs across seeds')
            data_provenance[domain] = prov
            state = torch.load(p/'initial_backbone.pt', map_location='cpu', weights_only=True)
            if initial is not None and (state.keys() != initial.keys() or any(not torch.equal(v, initial[k]) for k, v in state.items())):
                raise ValueError('Unpaired initial backbone')
            initial = state
            c = torch.load(p/'last.pt', map_location='cpu', weights_only=True)
            if any(c[k] != m[k] for k in ('config', 'domain', 'objective', 'smoke')):
                raise ValueError('Checkpoint metadata mismatch')
            if not all(torch.isfinite(v).all().item() for v in c['model'].values()):
                raise ValueError('Nonfinite checkpoint')
            model = ContrastiveJetModel(JetBackbone(**cfg['backbone']))
            model.load_state_dict(c['model'])
            name = f'{domain}_seed{seed}'
            scores[name], audits[name] = score(model, ref, query)
            provenance[name] = dict(folder=str(p), checkpoint_sha256=digest(p/'last.pt'), manifest_sha256=digest(p/'COMPLETE.json'), initial_sha256=digest(p/'initial_backbone.pt'))
            print('SCORED', name, flush=True)
        model = ContrastiveJetModel(JetBackbone(**cfg['backbone']))
        model.backbone.load_state_dict(initial)
        name = f'random_seed{seed}'
        scores[name], audits[name] = score(model, ref, query)
        print('SCORED', name, flush=True)
    for domain, old_name in [('aspen', 'aspen_contrastive'), ('lhco', 'lhco_contrastive'), ('random', 'random_backbone')]:
        # Tolerate small GPU/platform float32 differences, but never silently change a historical result.
        s = scores[f'{domain}_seed17']
        np.testing.assert_allclose(s, old[old_name], rtol=1e-4, atol=1e-5)
        if abs(roc_auc_score(y, s)-roc_auc_score(y, old[old_name])) > 2e-5:
            raise ValueError('Historical seed-17 AUC did not reproduce')
    result = {}
    for name, s in scores.items():
        v = metrics(y, s)
        if abs(v['auc'] - rank_auc(y, s)) > 1e-12:
            raise ValueError('Independent rank AUC mismatch')
        result[name] = dict(original_auc=v['auc'], original_working_points=v['working_points'],
                            common_pt_auc=float(roc_auc_score(y, s, sample_weight=w)),
                            common_pt_working_points=weighted_points(y, s, w))
    summary = {}
    for metric in ('original_auc', 'common_pt_auc'):
        summary[metric] = {d: summarize([result[f'{d}_seed{s}'][metric] for s in SEEDS]) for d in ('aspen', 'lhco', 'random')}
        summary[metric]['paired_aspen_minus_lhco'] = summarize([result[f'aspen_seed{s}'][metric]-result[f'lhco_seed{s}'][metric] for s in SEEDS])
    # Shared stratified event resamples; seed means are means of individual AUCs, not score ensembles.
    rng = np.random.default_rng(20261003)
    classes = [np.flatnonzero(y == c) for c in (0, 1)]
    boot = {metric: {d: [] for d in ('aspen', 'lhco', 'random', 'paired_aspen_minus_lhco')} for metric in summary}
    for _ in range(1000):
        ix = np.concatenate([rng.choice(a, len(a), replace=True) for a in classes])
        ww, _ = common_weights(y[ix], old['query_observables'][ix, 1], edges)
        for metric, weights in [('original_auc', None), ('common_pt_auc', ww)]:
            means = {d: float(np.mean([roc_auc_score(y[ix], scores[f'{d}_seed{s}'][ix], sample_weight=weights) for s in SEEDS])) for d in ('aspen', 'lhco', 'random')}
            means['paired_aspen_minus_lhco'] = means['aspen'] - means['lhco']
            for d, value in means.items():
                boot[metric][d].append(value)
    for metric in summary:
        for d in boot[metric]:
            summary[metric][d]['conditional_event_bootstrap_95'] = np.quantile(boot[metric][d], [.025, .975]).tolist()
    output = dict(stage='two_prong_development_not_final', seeds=list(SEEDS), results=result, summary=summary,
                  pt_edges=edges.tolist(), bin_counts=bins, provenance=provenance, distance_audits=audits,
                  old_cache_sha256=digest(CACHE), protocol_sha256=digest(PROTOCOL),
                  source_sha256={str(p): digest(p) for p in [Path(__file__), Path('scripts/validation/scientific_value_gate.py'), Path('scripts/validation/gate_crosscheck.py'), Path('src/models/controlled_encoder.py')]},
                  bootstrap=dict(replicates=1000, seed=20261003, conditional_on='all fitted models, fixed reference events and bin edges; weights recomputed per resample'),
                  limitations=['Adaptively explored development sample, not final evaluation.', 'Three training seeds share fixed samples; seed SD is not a confidence interval.', 'Event intervals do not include training or reference-pool uncertainty.', 'Working points are empirical descriptive thresholds, not calibrated selections.', 'Coarse pT balancing leaves within-bin and other kinematic differences.'])
    with args.out.with_suffix('.npz').open('xb') as f:
        np.savez_compressed(f, y=y, query_event_ids=query['event_id'], neighbor_event_ids=ref['event_id'], common_pt_weights=w, **scores)
    with args.out.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print('JOB_COMPLETE', args.out, flush=True)


def score(model, ref, query):
    model.eval().cuda()
    zr, _ = encode(model, ref)
    zq, _ = encode(model, query)
    if not np.isfinite(zr).all() or not np.isfinite(zq).all():
        raise ValueError('Nonfinite embeddings')
    s = NearestNeighbors(n_neighbors=10, algorithm='brute', n_jobs=1).fit(zr).kneighbors(zq)[0][:, -1]
    subset = np.r_[np.arange(16), 10000+np.arange(16)]
    independent = np.partition(cdist(zq[subset], zr), 9, axis=1)[:, 9]
    np.testing.assert_allclose(s[subset], independent, rtol=1e-4, atol=1e-5)
    model.cpu()
    return s, dict(independent_distance_max_error=float(np.max(abs(s[subset]-independent))))


if __name__ == '__main__':
    main()

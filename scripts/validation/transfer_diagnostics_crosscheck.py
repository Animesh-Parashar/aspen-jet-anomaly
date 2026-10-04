"""Independent saved-result checks for the transfer diagnostics; CPU PBS job."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import spearmanr

ROOT = Path('.')
result = json.loads(Path('data/results/transfer_support_20261003.json').read_text())
repr_result = json.loads(Path('data/results/transfer_representations_20261003.json').read_text())
checks = []
for name, r in result['samples'].items():
    provenance = r['provenance']
    with h5py.File(provenance['path']) as f:
        n = len(f['event_id'])
        # Independent access: full small observable/identity columns, rather
        # than the diagnostic's chunked constituent reader.
        ix = np.sort(np.random.default_rng(provenance['sample_seed']).choice(n, provenance['n_selected'], replace=False))
        ids = f['event_id'][:][ix]
        kin = f['jet_kinematics'][:][ix]
        mult = f['n_constituents'][:][ix]
        assert hashlib.sha256(ids.tobytes()).hexdigest() == provenance['selected_event_ids_sha256']
        for key, v in [('pt', kin[:,0]), ('mass', kin[:,3]), ('multiplicity', mult)]:
            np.testing.assert_allclose(np.quantile(v, result['quantile_order']), r['quantiles'][key], atol=0, rtol=0)
            h = r['histograms'][key]; edges = h['edges']
            manual = [np.count_nonzero((v >= lo) & ((v < hi) if j < len(edges)-2 else (v <= hi)))/len(v)
                      for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
            np.testing.assert_allclose(manual, h['probability'], atol=0, rtol=0)
        if name in result['support']['domains']:
            lo, hi = result['support']['reference_central90_pt_GeV']
            count = sum(lo <= value <= hi for value in kin[:,0])
            assert count == result['support']['domains'][name]['training_count_in_reference_central90']
    for h in r['histograms'].values():
        np.testing.assert_allclose(sum(h['probability'])+h['underflow']+h['overflow'], 1, atol=1e-9, rtol=0)
    checks.append(name)

with np.load('data/results/scorer_robustness_26312.rachel.npz', allow_pickle=False) as scores, np.load('data/results/frozen_final_two_26302.rachel.npz', allow_pickle=False) as a:
    obs = a['query_observables'][:10000]
    largest = 0.
    for name, r in repr_result['models'].items():
        for scorer, v in r['scorers'].items():
            for j, k in enumerate(('mass','pt','multiplicity','girth')):
                direct = spearmanr(scores['two_'+name+'_'+scorer][:10000], obs[:,j]).statistic
                largest = max(largest, abs(direct-v['background_spearman'][k]))
                np.testing.assert_allclose(direct, v['background_spearman'][k], atol=1e-14, rtol=0)
        for g in r['geometry'].values():
            e = np.array(g['eigenvalues_descending']); assert np.all(e >= 0)
            np.testing.assert_allclose(e.sum(), g['total_variance'], atol=1e-12)
            assert 0 <= g['participation_rank'] <= g['entropy_rank']+1e-10 <= 128+1e-10
    checks.append('108 independent Spearman correlations and 18 rank inequalities')
out = dict(status='passed', checks=checks, max_spearman_error=largest,
           diagnostic_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path('data/results/transfer_support_20261003.json'),Path('data/results/transfer_representations_20261003.json')]},
           script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with Path('data/results/transfer_diagnostics_crosscheck_20261003.json').open('x') as f: json.dump(out, f, indent=2)
print('INDEPENDENT_CHECKS_PASSED', checks, flush=True)

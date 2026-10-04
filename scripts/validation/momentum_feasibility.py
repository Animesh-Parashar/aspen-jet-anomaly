"""Count training/validation support only; no sampling, training or evaluation."""
import hashlib
import json
from pathlib import Path
import argparse
import h5py
import numpy as np

PROTOCOL = Path('docs/momentum_feasibility_protocol_20261003.md')
SUPPORT = Path('data/results/transfer_support_20261003.json')
ROOTS = {'aspen': Path('data/processed_v2/full_24624.rachel/aspen'),
         'lhco': Path('data/processed_v2/full_24632.rachel/lhco')}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def counts(pt, edges):
    """Half-open bins, with the final upper edge included (NumPy convention)."""
    h = np.histogram(pt, edges)[0]
    independent = np.array([np.count_nonzero((pt >= lo) & ((pt < hi) if i < len(edges)-2 else (pt <= hi)))
                            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))])
    np.testing.assert_array_equal(h, independent)
    assert int(h.sum())+int((pt < edges[0]).sum())+int((pt > edges[-1]).sum()) == len(pt)
    return h

def representatives(pt, ids):
    """One highest-pT available jet per event; original row breaks ties."""
    _, inv = np.unique(ids, axis=0, return_inverse=True)
    order = np.lexsort((np.arange(len(pt)), -pt, inv))
    return order[np.r_[True, inv[order][1:] != inv[order][:-1]]]

def run(out):
    if out.exists(): raise ValueError('Refusing to overwrite results')
    support = json.loads(SUPPORT.read_text())
    lo, hi = support['support']['reference_central90_pt_GeV']
    grids = {'reference_central90': np.linspace(lo, hi, 11),
             'broad_100GeV': np.arange(0., 5000.1, 100.)}
    result, inputs, identities = {}, {}, {}
    for domain, root in ROOTS.items():
        complete = json.loads((root/'COMPLETE.json').read_text())
        verified = json.loads((root/'VERIFIED.json').read_text())
        assert verified['status'] == 'passed' and verified['event_overlap_between_partitions'] == 0
        assert complete['metadata']['domain'] == domain
        result[domain] = {}
        for split in ('train', 'validation'):
            relative = split if domain == 'aspen' else 'background/'+split
            path = root/(relative+'.h5')
            with h5py.File(path, 'r') as f:
                assert f.attrs['partition'] == relative and f.attrs['schema'] == 'jet-preprocessing-v2.0'
                pt = f['jet_kinematics'][:, 0].astype(np.float64)
                ids = f['event_id'][:]
                src = f['source_index'][:]
                labels = f['label'][:]
            assert len(pt) == complete['output_counts'][relative] == verified['output_counts'][relative]
            assert np.isfinite(pt).all() and (pt >= 0).all()
            assert np.all(labels == (-1 if domain == 'aspen' else 0))
            assert len(np.unique(src, axis=0)) == len(src)
            ix = representatives(pt, ids)
            identities[domain, split] = ids[ix]
            selected = (pt >= lo) & (pt <= hi)
            entry = dict(rows=len(pt), unique_events=len(ix),
                         central90_rows=int(selected.sum()),
                         central90_unique_events=len(np.unique(ids[selected], axis=0)),
                         central90_highest_pt_event_representatives=int(((pt[ix] >= lo) & (pt[ix] <= hi)).sum()),
                         grids={})
            for name, edges in grids.items():
                entry['grids'][name] = dict(edges_GeV=edges.tolist(),
                    jet_rows=counts(pt, edges).tolist(),
                    highest_pt_event_representatives=counts(pt[ix], edges).tolist())
            result[domain][split] = entry
            inputs[str(path)] = dict(size_bytes=path.stat().st_size,
                pt_float64_sha256=hashlib.sha256(pt.tobytes()).hexdigest(),
                event_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                source_indices_sha256=hashlib.sha256(src.tobytes()).hexdigest(),
                complete_sha256=sha(root/'COMPLETE.json'),verified_sha256=sha(root/'VERIFIED.json'))
            print(domain, split, {k:v for k,v in entry.items() if k!='grids'}, flush=True)
        combined = np.concatenate([identities[domain,s] for s in ('train','validation')])
        assert len(np.unique(combined, axis=0)) == len(combined), 'Event leakage'
    matches = {}
    for split in ('train','validation'):
        matches[split] = {}
        for name in grids:
            values = {}
            for kind in ('jet_rows','highest_pt_event_representatives'):
                a, b = [np.array(result[d][split]['grids'][name][kind]) for d in ROOTS]
                m = np.minimum(a,b)
                values[kind] = dict(per_domain_total=int(m.sum()), per_bin=m.tolist())
            matches[split][name] = values
    output = dict(stage='momentum_matching_feasibility_only',results=result,matching=matches,
        source_columns=inputs,protocol_sha256=sha(PROTOCOL),script_sha256=sha(__file__),
        reference_summary_sha256=sha(SUPPORT),checks='passed',
        interpretation=['Equal bin counts match coarse distributions, not within-bin momenta.',
          'Jet-row capacity can include correlated jets from one event.',
          'Highest-pT-per-event capacity is feasible for that fixed policy, not a maximum over all event assignments.',
          'No test/reference HDF5 or signal data read; bounds copied from previously published diagnostic.',
          'Broad-grid overlap outside the reference regime is not a substitute for in-regime training.'])
    with out.open('x') as f: json.dump(output,f,indent=2,allow_nan=False)
    print('JOB_COMPLETE',out,flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)

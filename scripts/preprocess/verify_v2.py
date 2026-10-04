"""Validate processed tensors, source-row uniqueness, and event separation."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    manifest = json.loads((root / 'COMPLETE.json').read_text())
    domain = manifest['metadata']['domain']
    all_ids, all_rows, unique_per_partition = [], [], 0
    for key, count in manifest['output_counts'].items():
        with h5py.File(root / (key + '.h5'), 'r') as f:
            check(f.attrs['partition'] == key, 'Partition metadata mismatch')
            check(all(len(ds) == count for ds in f.values()), 'Inconsistent dataset lengths')
            ids = f['event_id'][:]
            all_ids.append(ids)
            unique_per_partition += len(np.unique(ids, axis=0))
            all_rows.append(f['source_index'][:])
            for start in range(0, count, 2048):
                end = min(count, start + 2048)
                x = f['constituents'][start:end]
                mask = f['mask'][start:end]
                check(x.shape[-1] == 4 and np.isfinite(x).all(), 'Invalid core features')
                check(np.all(x[mask] == 0), 'Nonzero padded features')
                n = np.minimum(f['n_constituents'][start:end], x.shape[1])
                check(np.all((~mask).sum(axis=1) == n) and np.all(n > 0), 'Invalid multiplicity')
                check(np.array_equal(mask, np.arange(x.shape[1])[None, :] >= n[:, None]), 'Nontrailing padding')
                for row, valid in zip(x, ~mask):
                    check(np.all(np.diff(row[valid, 3]) <= 1e-5), 'Constituents not pT sorted')
                fraction = f['retained_pt_fraction'][start:end]
                check(np.all((fraction > 0) & (fraction <= 1.000001)), 'Invalid retained momentum')
                reconstructed_fraction = np.where(mask, 0, np.exp(x[:, :, 2])).sum(axis=1)
                check(np.allclose(reconstructed_fraction, fraction, atol=2e-6), 'Normalization mismatch')
                check(np.isfinite(f['jet_kinematics'][start:end]).all(), 'Invalid jet kinematics')
                labels = f['label'][start:end]
                expected = -1 if domain == 'aspen' else int(key.startswith('signal/'))
                check(np.all(labels == expected), 'Label contamination')
                check(not (expected == 1 and key.split('/')[-1] in ('train', 'reference')), 'Signal leakage')
                if domain == 'aspen':
                    aux = f['detector_features'][start:end]
                    observed = f['detector_observed'][start:end]
                    check(np.isfinite(aux).all(), 'Invalid detector features')
                    check(np.all(aux[~observed] == 0), 'Unobserved detector values must be zero')
                    check(not observed[mask].any(), 'Padding marked observed')
    ids = np.concatenate(all_ids)
    rows = np.concatenate(all_rows)
    check(len(np.unique(ids, axis=0)) == unique_per_partition, 'Events shared across partitions')
    check(len(np.unique(rows)) == len(rows), 'Source rows duplicated')
    check(len(rows) + sum(manifest['rejected'].values()) == manifest['source_rows_examined'], 'Row accounting mismatch')
    result = dict(status='passed', domain=domain, rows=int(len(rows)),
                  event_overlap_between_partitions=0, duplicate_source_rows=0,
                  output_counts=manifest['output_counts'])
    (root / 'VERIFIED.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    verify(parser.parse_args().root)

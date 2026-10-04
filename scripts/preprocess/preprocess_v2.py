"""Create versioned, physically separated train/validation/reference/test HDF5s.

Never overwrites a prior output directory. Raw data and historical scripts are
untouched. Signal outputs only have validation/test partitions. A COMPLETE.json
manifest is written only after all outputs close successfully.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hdf5plugin
import h5py
import numpy as np
import fastjet
from src.data import corrected_preprocessing as core


def sample_ranges(n, limit, chunk_size):
    if limit is None or limit >= n:
        return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    # Uniformly spread blocks; a prefix would omit signals in ordered LHCO.
    blocks = min(40, limit)
    edges = np.linspace(0, n, blocks + 1, dtype=int)
    sizes = np.full(blocks, limit // blocks)
    sizes[:limit % blocks] += 1
    return [(int((lo + hi - size) // 2), int((lo + hi - size) // 2 + size))
            for lo, hi, size in zip(edges[:-1], edges[1:], sizes)]


class Writer:
    def __init__(self, out_dir, metadata):
        self.out_dir, self.metadata, self.handles = out_dir, metadata, {}
        self.counts = Counter()

    def append(self, key, records):
        if not records:
            return
        arrays = {name: np.stack([r[name] for r in records]) for name in records[0]}
        if key not in self.handles:
            path = self.out_dir / (key + '.h5')
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = h5py.File(path, 'x')
            self.handles[key] = handle
            handle.attrs['metadata'] = json.dumps(self.metadata, sort_keys=True)
            handle.attrs['partition'] = key
            handle.attrs['schema'] = core.SCHEMA
            for name, array in arrays.items():
                handle.create_dataset(name, shape=(0,) + array.shape[1:],
                    maxshape=(None,) + array.shape[1:], dtype=array.dtype,
                    chunks=(min(256, len(array)),) + array.shape[1:], compression='lzf')
        handle = self.handles[key]
        offset = self.counts[key]
        for name, array in arrays.items():
            handle[name].resize(offset + len(records), axis=0)
            handle[name][offset:] = array
        self.counts[key] += len(records)

    def close(self):
        for key, handle in self.handles.items():
            if handle.id.valid:
                handle.attrs['n_jets'] = self.counts[key]
                handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', choices=['aspen', 'lhco'], required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True, help='Prior sample audit: reserve its events for validation')
    p.add_argument('--sample-rows', type=int)
    p.add_argument('--chunk-size', type=int, default=1024)
    p.add_argument('--max-constituents', type=int, default=50)
    p.add_argument('--seed', type=int, default=20260925)
    args = p.parse_args()
    if args.chunk_size < 1 or args.max_constituents < 1 or (args.sample_rows is not None and args.sample_rows < 1):
        p.error('Sizes must be positive')
    if not 0 <= args.seed < 2**64:
        p.error('Seed must fit uint64')
    audit = json.loads(args.audit.read_text())
    audit_indices = audit[args.domain]['sampled_indices']
    source = args.input.resolve()
    verification_path = Path(str(source) + '.download.json')
    verification = json.loads(verification_path.read_text())
    if verification['state'] != 'verified' or source.stat().st_size != verification['expected_bytes']:
        raise ValueError('Input must have a successful download verification manifest')
    t0 = time.perf_counter()
    namespace = 'CMS2016' if args.domain == 'aspen' else 'LHCO_RD_v2'
    metadata = dict(schema=core.SCHEMA, domain=args.domain, source=str(source),
        recorded_verified_md5=verification['md5'], features=core.FEATURE_NAMES,
        max_constituents=args.max_constituents, seed=args.seed, namespace=namespace,
        axis='summed unweighted constituent three-momentum before truncation',
        normalization='scalar sum of available constituent pT before truncation',
        units='input momentum in GeV; natural log; angles in radians',
        raw_aspen_limit=150, use_puppi_weights=False,
        lhco_clustering='anti-kt R=0.8, massless inputs, E-scheme, leading by vector pT',
        partitions='background/unlabeled 60/15/10/15; signal validation/test 50/50; audit events forced validation',
        detector_auxiliary='Aspen only: d0 column4, dz column6, charge column8; observed mask; impact clip [-5,5]',
        sample_rows=args.sample_rows, audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=dict(numpy=np.__version__, h5py=h5py.__version__, fastjet=fastjet.__version__))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'RUNNING.json').write_text(json.dumps(metadata, indent=2) + '\n')
    writer = Writer(args.out, metadata)
    rejected, processed = Counter(), 0
    try:
        with h5py.File(source, 'r') as raw:
            if args.domain == 'aspen':
                n = len(raw['PFCands'])
                dev_ids = {tuple(row) for row in raw['event_info'][sorted(set(audit_indices))]}
            else:
                if not np.array_equal(raw['df/block0_items'][:], np.arange(2101)):
                    raise ValueError('Unexpected LHCO column ordering')
                n = len(raw['df/block0_values'])
                dev_ids = {(int(i),) for i in audit_indices}
            ranges = sample_ranges(n, args.sample_rows, args.chunk_size)
            for lo, hi in ranges:
                if args.domain == 'aspen':
                    block = raw['PFCands'][lo:hi]
                    ids = raw['event_info'][lo:hi]
                    original_kinematics = raw['jet_kinematics'][lo:hi]
                    labels = np.full(hi-lo, -1, dtype=np.int8)
                else:
                    values = raw['df/block0_values'][lo:hi]
                    if not np.isin(values[:, -1], [0, 1]).all():
                        raise ValueError('Invalid LHCO labels')
                    labels = values[:, -1].astype(np.int8)
                    block = values[:, :-1].reshape(-1, 700, 3)
                    ids = np.arange(lo, hi, dtype=np.int64)[:, None]
                batches = {}
                for j, particles in enumerate(block):
                    label = int(labels[j])
                    event_id = tuple(int(i) for i in ids[j])
                    split = core.partition(namespace, event_id, label, args.seed, event_id in dev_ids)
                    key = split if args.domain == 'aspen' else ('signal/' if label else 'background/') + split
                    try:
                        c = particles if args.domain == 'aspen' else core.leading_lhco_constituents(particles)
                        record = core.represent(c, args.max_constituents, detector=args.domain == 'aspen')
                    except ValueError as exc:
                        rejected[str(exc)] += 1
                        continue
                    record.update(source_index=np.int64(lo+j), event_id=np.asarray(event_id, dtype=np.int64),
                                  label=np.int8(label))
                    if args.domain == 'aspen':
                        record['source_jet_kinematics'] = original_kinematics[j]
                    batches.setdefault(key, []).append(record)
                for key, records in batches.items():
                    writer.append(key, records)
                processed += hi-lo
                if processed % (10 * args.chunk_size) == 0:
                    print(f'{args.domain}: {processed:,} source rows examined', flush=True)
        writer.close()
        if not sum(writer.counts.values()):
            raise ValueError('No usable jets')
        result = dict(metadata=metadata, source_rows_examined=processed,
                      output_counts=dict(writer.counts), rejected=dict(rejected),
                      elapsed_seconds=time.perf_counter()-t0)
        (args.out / 'COMPLETE.json').write_text(json.dumps(result, indent=2) + '\n')
        (args.out / 'RUNNING.json').unlink()
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        writer.close()
        raise


if __name__ == '__main__':
    main()

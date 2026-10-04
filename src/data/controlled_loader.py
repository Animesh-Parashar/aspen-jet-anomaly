"""Verified core4 data access. Training API cannot open reference/test/signal files."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ControlledDataset(Dataset):
    def __init__(self, root, domain, split, limit=None, seed=0):
        if split not in ('train', 'validation'):
            raise ValueError('Training loader only permits train/validation')
        if domain not in ('aspen', 'lhco'):
            raise ValueError('Unknown domain')
        root = Path(root)
        complete = json.loads((root / 'COMPLETE.json').read_text())
        verified = json.loads((root / 'VERIFIED.json').read_text())
        meta = complete['metadata']
        if (verified['status'] != 'passed' or meta['domain'] != domain or
                meta['schema'] != 'jet-preprocessing-v2.0' or
                meta['features'] != ['eta_rel', 'phi_rel', 'log_pt_fraction', 'log_pt_GeV']):
            raise ValueError('Requires verified v2 core4 dataset')
        path = root / (split+'.h5' if domain == 'aspen' else 'background/'+split+'.h5')
        with h5py.File(path, 'r') as f:
            n = len(f['constituents'])
            if limit is not None and not 0 < limit <= n:
                raise ValueError('Requested sample exceeds available split or is empty')
            indices = np.arange(n) if limit is None else np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))
            # Read contiguous chunks once; avoid very slow HDF5 point selection.
            xs, masks, ids = [], [], []
            for start in range(0, n, 8192):
                selected = indices[(indices >= start) & (indices < start+8192)] - start
                if not len(selected):
                    continue
                block = slice(start, min(n, start+8192))
                expected_label = -1 if domain == 'aspen' else 0
                if not np.all(f['label'][block][selected] == expected_label):
                    raise ValueError('Unexpected training labels')
                xs.append(f['constituents'][block][selected])
                masks.append(f['mask'][block][selected])
                ids.append(f['event_id'][block][selected])
            self.x = torch.from_numpy(np.concatenate(xs))
            self.mask = torch.from_numpy(np.concatenate(masks))
            self.event_ids = np.concatenate(ids)
        if self.mask.all(1).any() or not torch.isfinite(self.x).all():
            raise ValueError('Invalid training inputs')
        self.provenance = dict(path=str(path.resolve()), n_available=n, n_selected=len(self.x),
                               sample_seed=seed, preprocessing=meta)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.mask[index]

"""
Quick script to inspect the HDF5 structure of downloaded datasets.
Run this FIRST after downloading to understand the schema.

Usage:
    python inspect_data.py data/aspen/your_file.h5
    python inspect_data.py data/lhco/events_anomalydetection_v2.h5
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))


import sys
import h5py
import numpy as np


def inspect(path):
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")

    with h5py.File(path, "r") as f:
        def _walk(name, obj):
            if hasattr(obj, "shape"):
                print(f"  [{name}]  shape={obj.shape}  dtype={obj.dtype}")
                # Print a tiny sample
                try:
                    sample = obj.flat[:5]
                    print(f"           sample: {list(sample)}")
                except Exception:
                    pass
            else:
                print(f"  [{name}]  (group)")
        f.visititems(_walk)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        inspect(path)

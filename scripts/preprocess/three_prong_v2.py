"""Signal-only LHCO qqq adapter; preserves corrected core and historical drivers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import hdf5plugin
import h5py
import numpy as np
from src.data import corrected_preprocessing as core
from scripts.preprocess.preprocess_v2 import Writer
from scripts.preprocess.verify_v2 import verify

EXPECTED_MD5='54e123a86143b668f9cb76905152a124'
NAMESPACE='LHCO_RD_qqq_v1'

def particles_from_values(values):
    values=np.asarray(values)
    if values.ndim!=2 or values.shape[1] not in (2100,2101):
        raise ValueError('Unsupported qqq layout')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite input')
    if values.shape[1]==2101:
        if not np.all(values[:,-1]==1):
            raise ValueError('Signal-only file has non-signal label')
        values=values[:,:-1]
    particles=values.reshape(-1,700,3)
    if (particles[:,:,0]<0).any():
        raise ValueError('Negative particle pT')
    return particles

def event_identity(index):
    # Existing LHCO v2 IDs are nonnegative source rows; negative qqq IDs make
    # cross-file identity separation explicit even in single-column consumers.
    if index<0: raise ValueError('Invalid source index')
    return np.array([-1-index],dtype=np.int64)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Refusing overwrite')
    md5=hashlib.md5()
    with a.input.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):md5.update(b)
    if md5.hexdigest()!=EXPECTED_MD5:raise ValueError('Official checksum mismatch')
    with h5py.File(a.input,'r') as raw:
        ds=raw['df/block0_values'];shape=ds.shape
        if shape[0]!=100000 or shape[1] not in (2100,2101):raise ValueError('Unexpected official shape')
        if not np.array_equal(raw['df/block0_items'][:],np.arange(shape[1])):raise ValueError('Unexpected column ordering')
    verification=dict(state='verified',md5=md5.hexdigest(),expected_bytes=a.input.stat().st_size,source='https://zenodo.org/records/6466204',shape=list(shape))
    Path(str(a.input)+'.download.json').write_text(json.dumps(verification,indent=2)+'\n')
    meta=dict(schema=core.SCHEMA,domain='lhco',topology='three_prong',namespace=NAMESPACE,features=core.FEATURE_NAMES,source=str(a.input.resolve()),recorded_verified_md5=EXPECTED_MD5,max_constituents=50,seed=20260925,event_id_encoding='negative one minus raw row; namespace LHCO_RD_qqq_v1',clustering='anti-kt R=0.8 massless inputs E scheme leading vector pT',axis='full constituent sum before truncation',normalization='full scalar constituent pT before truncation',partitions='signal validation/test 50/50 event hash; no training/reference',exposure='Historical full-sample evaluation possible; new partitions are not claimed historically unseen',source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [str(Path(__file__)),str(Path(core.__file__))]})
    a.out.mkdir(parents=True);(a.out/'RUNNING.json').write_text(json.dumps(meta,indent=2)+'\n');writer=Writer(a.out,meta);rejected=Counter();start=time.perf_counter()
    try:
        with h5py.File(a.input,'r') as f:
            ds=f['df/block0_values']
            for lo in range(0,len(ds),512):
                block=particles_from_values(ds[lo:lo+512]);batches={}
                for j,particles in enumerate(block):
                    index=lo+j;eid=event_identity(index)
                    split=core.partition(NAMESPACE,tuple(eid),1,20260925)
                    try:r=core.represent(core.leading_lhco_constituents(particles),50)
                    except ValueError as exc:rejected[str(exc)]+=1;continue
                    r.update(source_index=np.int64(index),event_id=eid,label=np.int8(1));batches.setdefault('signal/'+split,[]).append(r)
                for key,records in batches.items():writer.append(key,records)
                if lo%5120==0:print('PROCESSED',lo+len(block),flush=True)
        writer.close()
        result=dict(metadata=meta,source_rows_examined=100000,output_counts=dict(writer.counts),rejected=dict(rejected),elapsed_seconds=time.perf_counter()-start)
        (a.out/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');verify(a.out);(a.out/'RUNNING.json').unlink()
        print('JOB_COMPLETE',a.out,flush=True)
    finally:writer.close()
if __name__=='__main__':main()

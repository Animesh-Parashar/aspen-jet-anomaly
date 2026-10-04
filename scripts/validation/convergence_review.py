"""Descriptive fixed-window training review; no checkpoint selection."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 paths=list(Path('data/results/controlled_convergence_24711.rachel').glob('*/COMPLETE.json'))+[Path('data/results/reconstruction_extension_26160.rachel/COMPLETE.json'),Path('data/results/decoder_ablation_26167.rachel/COMPLETE.json')]
 results={}
 for path in paths:
  d=json.loads(path.read_text());h=d['history']
  if d['smoke'] or len(h)!=d['config']['epochs']:raise ValueError('Incomplete run')
  rows=[]
  for width in [5,10,20]:
   if len(h)<2*width:continue
   prev=np.array([r['validation_loss'] for r in h[-2*width:-width]]);last=np.array([r['validation_loss'] for r in h[-width:]])
   rows.append(dict(window=width,previous_mean=float(prev.mean()),last_mean=float(last.mean()),relative_improvement_percent=float(100*(prev.mean()-last.mean())/abs(prev.mean())),last_range=[float(last.min()),float(last.max())]))
  results[str(path.parent)]=dict(epochs=len(h),windows=rows,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 out=dict(results=results,limitations=['Epoch losses are serially dependent; no IID confidence intervals or convergence proof.','Dropout and fixed validation views differ from training; raw train-validation gaps are not directly comparable.','No signal/test outcomes accessed; no final budget automatically selected.'])
 with args.out.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
 print('JOB_COMPLETE',args.out)
if __name__=='__main__':main()

"""Read completed training manifests only; no evaluation data or checkpoint selection."""
import argparse,json,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);args=p.parse_args()
 results={}
 for name,epochs in [('aspen_contrastive',100),('lhco_contrastive',100),('aspen_reconstruction',150)]:
  path=args.runs/name/'COMPLETE.json';d=json.loads(path.read_text());h=d['history']
  if d['smoke'] or len(h)!=epochs or d['config']['epochs']!=epochs:raise ValueError('Incomplete planned run')
  windows=[]
  for width in [5,10,20]:
   prev=sum(x['validation_loss'] for x in h[-2*width:-width])/width
   last=sum(x['validation_loss'] for x in h[-width:])/width
   windows.append(dict(width=width,previous_mean=prev,last_mean=last,improvement_percent=100*(prev-last)/abs(prev)))
  results[name]=dict(epochs=epochs,continuation=d['continuation'],elapsed_seconds=d['elapsed_seconds'],windows=windows,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 with (args.runs/'review.json').open('x') as f:json.dump(dict(results=results,limitation='Descriptive serially dependent loss windows; not proof of convergence or anomaly sensitivity.'),f,indent=2,allow_nan=False)
 print('REVIEW_COMPLETE',args.runs,flush=True)
if __name__=='__main__':main()

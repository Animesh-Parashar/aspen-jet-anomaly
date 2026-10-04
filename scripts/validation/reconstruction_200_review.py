"""Fixed-window validation-loss review; no dataset access or model selection."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    path=args.run/'COMPLETE.json';d=json.loads(path.read_text());h=d['history']
    if d['smoke'] or len(h)!=200 or d['config']['epochs']!=200:raise ValueError('Require completed 200-epoch run')
    rows=[]
    for width in [5,10,20]:
        previous=sum(x['validation_loss'] for x in h[-2*width:-width])/width
        last=sum(x['validation_loss'] for x in h[-width:])/width
        rows.append(dict(width=width,previous_mean=previous,last_mean=last,improvement_percent=100*(previous-last)/abs(previous)))
    output=dict(epochs=200,elapsed_seconds=d['elapsed_seconds'],windows=rows,
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                limitation='Serially dependent validation-loss summaries; not proof of convergence or anomaly sensitivity. No signal/test access.')
    with (args.run/'review.json').open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps(output),flush=True)
if __name__=='__main__':main()

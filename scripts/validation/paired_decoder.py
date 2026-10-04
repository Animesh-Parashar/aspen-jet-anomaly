"""Paired query-event bootstrap, conditional on fixed models and neighbor pool."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel,delayed


def interval(a,b,x,ids):
    _,g=np.unique(ids,axis=0,return_inverse=True)
    blocks=[np.flatnonzero(g==i) for i in range(g.max()+1)]
    def statistic(ix):
        ra=float(spearmanr(a[ix],x[ix]).statistic);rb=float(spearmanr(b[ix],x[ix]).statistic)
        return [rb-ra,abs(rb)-abs(ra)]
    rng=np.random.default_rng(1702);values=[]
    for _ in range(1000):
        ix=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]);values.append(statistic(ix))
    return dict(point=statistic(np.arange(len(a))),interval_95=np.quantile(values,[.025,.975],axis=0).tolist(),
                columns=['rho_unmasked_minus_masked','absolute_rho_unmasked_minus_masked'])


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 cache=Path('data/results/decoder_ablation_26167.rachel_scores.npz')
 with np.load(cache,allow_pickle=False) as a:
  tasks=[(d,f,a[d+'_score_aspen_reconstruction_100'],a[d+'_score_aspen_reconstruction_unmasked_100'],a[d+'_'+f],a[d+'_event_ids']) for d in ['aspen','lhco'] for f in ['full_stored_multiplicity','retained_multiplicity','constituent_sum_mass','retained_pt_fraction']]
 results=Parallel(n_jobs=4)(delayed(interval)(a,b,x,ids) for d,f,a,b,x,ids in tasks)
 out=dict(results={d:{f:r for (dd,f,*_),r in zip(tasks,results) if dd==d} for d in ['aspen','lhco']},cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),limitations=['Single training seed; fixed neighbor pool; pointwise intervals, no multiplicity correction.','Difference in association is not an anomaly-performance difference.'])
 with args.out.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
 print('JOB_COMPLETE',args.out)
if __name__=='__main__':main()

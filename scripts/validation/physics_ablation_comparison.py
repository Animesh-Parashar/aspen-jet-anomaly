"""Paired conditional bootstrap for the prospectively fixed development screen."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from gate_crosscheck import common_weights
from scientific_value_gate import metrics


def weighted_auc_sorted(y,score):
    order=np.argsort(score,kind='stable');ys=y[order]
    starts=np.r_[0,np.flatnonzero(np.diff(score[order]))+1]
    def evaluate(weights):
        w=weights[order]
        bg=np.add.reduceat(w*(ys==0),starts);sg=np.add.reduceat(w*(ys==1),starts)
        return float((sg*(np.cumsum(bg)-.5*bg)).sum()/(bg.sum()*sg.sum()))
    return evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Refusing overwrite')
    with np.load(a.cache,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    arms=['rotation_split','rotation_soft'];controls=['aspen_contrastive','random_backbone','girth_high','observable_density'];names=arms+controls
    y=d['y'];pt=d['query_observables'][:,1];edges=np.unique(np.quantile(d['reference_observables'][:,1],np.linspace(0,1,21)))
    w,rows=common_weights(y,pt,edges)
    evaluators={n:weighted_auc_sorted(y,d[n]) for n in names}
    result={n:dict(metrics(y,d[n]),common_pt_auc=evaluators[n](w)) for n in names}
    for n in names:
        result[n]['eligibility_strata']={}
        for eligible in [False,True]:
            keep=d['query_split_eligible']==eligible
            counts=[int(((y==c)&keep).sum()) for c in [0,1]]
            result[n]['eligibility_strata'][str(eligible)]={'background':counts[0],'signal':counts[1],'original_auc':float(roc_auc_score(y[keep],d[n][keep])) if min(counts)>=100 else None}
        np.testing.assert_allclose(result[n]['common_pt_auc'],roc_auc_score(y,d[n],sample_weight=w),rtol=0,atol=1e-12)
    bg=np.flatnonzero(y==0);sg=np.flatnonzero(y==1);rng=np.random.default_rng(1801)
    boot=np.empty((10000,len(names)))
    for b in range(len(boot)):
        ix=np.r_[rng.choice(bg,len(bg)),rng.choice(sg,len(sg))]
        wb,_=common_weights(y[ix],pt[ix],edges)
        # Aggregate resampled weights on original event IDs, retaining duplicates.
        weights=np.bincount(ix,weights=wb,minlength=len(y))
        for j,n in enumerate(names):boot[b,j]=evaluators[n](weights)
        if b%1000==0:print('BOOTSTRAP',b,flush=True)
    for j,n in enumerate(names):result[n]['common_pt_auc_95']=np.quantile(boot[:,j],[.025,.975]).tolist()
    decisions={}
    for i,n in enumerate(arms):
        differences={}
        for j,c in enumerate(controls,start=2):
            differences[c]={'point':result[n]['common_pt_auc']-result[c]['common_pt_auc'],'simultaneous_interval':np.quantile(boot[:,i]-boot[:,j],[.003125,.996875]).tolist()}
        decisions[n]={'paired_differences':differences,'advance':all(v['point']>0 and v['simultaneous_interval'][0]>0 for v in differences.values())}
    out={'stage':'adaptive_signal_validation_development_not_final','results':result,'decisions':decisions,'replicates':10000,'bin_counts':rows,'pt_edges':edges.tolist(),'cache_sha256':hashlib.sha256(a.cache.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':['Single training seed and fixed reference pool.','Bonferroni intervals cover eight planned comparisons conditionally, not adaptive development or training variability.','No final test/reference evaluation.']}
    with a.out.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print('COMPARISON_COMPLETE',a.out,flush=True)
if __name__=='__main__':main()

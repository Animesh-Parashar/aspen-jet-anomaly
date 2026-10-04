"""Three prespecified CPU-only analyses of cached validation query scores."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from joblib import Parallel, delayed


def correlation(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(spearmanr(x,y).statistic)


def event_blocks(ids):
    _,inverse=np.unique(ids,axis=0,return_inverse=True)
    return [np.flatnonzero(inverse==i) for i in range(inverse.max()+1)]


def analyze(mode,domain,name,a):
    y=a[domain+'_score_'+name]; ids=a[domain+'_event_ids']
    features=['full_stored_multiplicity','retained_pt_fraction','constituent_sum_mass','vector_jet_pt']
    x=np.column_stack([a[domain+'_'+f] for f in features]);n=x[:,0]
    if mode=='bootstrap':
        blocks=event_blocks(ids);rng=np.random.default_rng(1701);samples=[]
        for _ in range(500):
            ix=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])
            yy=y[ix];xx=x[ix];selected=yy>=np.quantile(yy,.9);low=xx[:,0]<=50
            samples.append([correlation(yy,xx[:,j]) for j in range(4)]+[float(selected[low].mean()-selected[~low].mean()) if low.any() and (~low).any() else None])
        labels=features+['untruncated_minus_truncated_selection_fraction']
        intervals={}
        for j,label in enumerate(labels):
            valid=[row[j] for row in samples if row[j] is not None]
            intervals[label]=dict(valid_replicates=len(valid),percentile_95=np.quantile(valid,[.025,.975]).tolist() if valid else None)
        return dict(intervals=intervals,events=len(blocks),replicates=500)
    if mode=='strata':
        edges=[np.unique(np.quantile(x[:,j],[0,.25,.5,.75,1])) for j in [2,3]]
        groups=[np.searchsorted(e[1:-1],x[:,j],side='right') for e,j in zip(edges,[2,3])]
        selected=y>=np.quantile(y,.9);rows=[]
        for i in range(len(edges[0])-1):
            for j in range(len(edges[1])-1):
                mask=(groups[0]==i)&(groups[1]==j);low=mask&(n<=50);high=mask&(n>50)
                rows.append(dict(mass_bin=i,pt_bin=j,n=int(mask.sum()),untruncated_n=int(low.sum()),truncated_n=int(high.sum()),
                    multiplicity_rho=correlation(y[mask],n[mask]),
                    retained_fraction_rho=correlation(y[mask],x[mask,1]),
                    untruncated_acceptance=float(selected[low].mean()) if low.sum()>=20 else None,
                    truncated_acceptance=float(selected[high].mean()) if high.sum()>=20 else None))
        return dict(mass_edges=edges[0].tolist(),pt_edges=edges[1].tolist(),cells=rows,minimum_group_size=20)
    _,group=np.unique(ids,axis=0,return_inverse=True);rows=[]
    for seed in [17,29,43]:
        order=np.random.default_rng(seed).permutation(group.max()+1);folds=np.empty(len(order),dtype=int);folds[order]=np.arange(len(order))%5
        for leaf in [15,30,60]:
            for feature_set,columns in [('multiplicity',[0]),('combined',[0,1,2,3])]:
                prediction=np.empty(len(y))
                for fold in range(5):
                    test=folds[group]==fold;train=~test
                    assert not set(group[test]) & set(group[train])
                    model=RandomForestRegressor(n_estimators=100,max_depth=8,min_samples_leaf=leaf,random_state=seed,n_jobs=1)
                    model.fit(x[train][:,columns],y[train]);prediction[test]=model.predict(x[test][:,columns])
                rows.append(dict(seed=seed,minimum_leaf=leaf,features=feature_set,r2=float(r2_score(y,prediction)),spearman=correlation(y,prediction)))
    return dict(variants=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['bootstrap','strata','probe'],required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.out.exists() or not 1<=args.workers<=4:raise ValueError('Fresh output and 1–4 workers required')
    with np.load(args.cache,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    tasks=[(d,k[len(d+'_score_'):]) for d in ['aspen','lhco'] for k in a if k.startswith(d+'_score_')]
    results=Parallel(n_jobs=args.workers)(delayed(analyze)(args.mode,d,n,a) for d,n in tasks)
    output=dict(stage='validation_only_'+args.mode,results={d:{n:r for (dd,n),r in zip(tasks,results) if dd==d} for d in ['aspen','lhco']},
        cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['No signal sensitivity or causal attribution. No model selection from these scores.',
        'Intervals condition on trained model and fixed validation neighbor pool; exclude training/reference uncertainty.',
        'Bootstrap resamples query events and recomputes the top-decile threshold; intervals are pointwise, not simultaneous.',
        'Mass/pT strata are coarse validation quantiles, not exact matching; sparse group rates are withheld.',
        'All prespecified probe variants are reported; no best variant is selected.'])
    with args.out.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print('JOB_COMPLETE',args.out,flush=True)
if __name__=='__main__':main()

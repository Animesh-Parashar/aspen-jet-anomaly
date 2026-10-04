"""Event-cross-fitted prediction of validation scores from simple observables.

This measures score dependence, not signal sensitivity or causal attribution.
"""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score


def crossfit(x,y,ids):
    _,groups=np.unique(ids,axis=0,return_inverse=True)
    prediction=np.empty(len(y))
    for train,test in GroupKFold(n_splits=5).split(x,y,groups):
        assert not set(groups[train]) & set(groups[test])
        model=RandomForestRegressor(n_estimators=100,min_samples_leaf=30,max_depth=8,random_state=17,n_jobs=1)
        model.fit(x[train],y[train]);prediction[test]=model.predict(x[test])
    return dict(r2=float(r2_score(y,prediction)),spearman=float(spearmanr(y,prediction).statistic))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists(): raise ValueError('Refusing overwrite')
    sets=dict(multiplicity=['full_stored_multiplicity'],truncation=['full_stored_multiplicity','retained_pt_fraction'],
              kinematics=['constituent_sum_mass','vector_jet_pt'],combined=['full_stored_multiplicity','retained_pt_fraction','constituent_sum_mass','vector_jet_pt'])
    results={}
    with np.load(args.cache,allow_pickle=False) as a:
        for domain in ['aspen','lhco']:
            results[domain]={}
            for key in a.files:
                prefix=domain+'_score_'
                if not key.startswith(prefix): continue
                y=a[key]; rows={}
                for name,features in sets.items():
                    x=np.column_stack([a[domain+'_'+f] for f in features])
                    rows[name]=crossfit(x,y,a[domain+'_event_ids'])
                groups={}
                for label,mask in [('not_truncated',a[domain+'_full_stored_multiplicity']<=50),('truncated',a[domain+'_full_stored_multiplicity']>50)]:
                    groups[label]=dict(n=int(mask.sum()),median_score=float(np.median(y[mask])) if mask.any() else None,
                        top_decile_fraction=float((y[mask]>=np.quantile(y,.9)).mean()) if mask.any() else None)
                results[domain][key[len(prefix):]]=dict(crossfit=rows,truncation_groups=groups)
    output=dict(stage='validation_only_nuisance_controls',feature_sets=sets,results=results,
        cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Fixed regression probe, not tuned; poor prediction does not prove independence.',
        'R2 is out-of-fold score predictability, not anomaly performance.',
        'Dependent features prevent unique causal attribution; no uncertainty intervals.',
        'Validation query scores share a fixed validation neighbor pool.'])
    with args.out.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print('COMPLETE',args.out,flush=True)
if __name__=='__main__':main()

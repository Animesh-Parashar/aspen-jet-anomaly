"""Fixed cross-fitted linear diagnostics; not an unsupervised anomaly benchmark."""
import argparse,hashlib,json,warnings
from pathlib import Path
import numpy as np
import scipy
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import roc_auc_score,r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from gate_crosscheck import common_weights


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def event_folds(y,ids):
    if ids.ndim!=2 or len(ids)!=len(y) or len(np.unique(ids,axis=0))!=len(ids):
        raise ValueError('Need one unique event per row')
    if set(np.unique(y))!={0,1} or min(np.bincount(y.astype(int)))<5:
        raise ValueError('Insufficient binary classes')
    folds=np.full(len(y),-1,dtype=int)
    for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261002).split(np.zeros(len(y)),y)):
        if np.intersect1d(train,test).size or np.any(folds[test]!=-1):raise ValueError('Fold overlap')
        folds[test]=fold
    if (folds<0).any():raise ValueError('Unassigned event')
    return folds


def logistic_fit_predict(x,y,train,test,weights):
    if not np.isfinite(x).all() or not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Invalid probe inputs')
    if len(weights)!=len(train) or np.intersect1d(train,test).size:raise ValueError('Invalid fold')
    keep=weights>0; tr=train[keep]; w=weights[keep].copy()
    if set(np.unique(y[tr]))!={0,1}:raise ValueError('Unsupported training classes')
    w/=w.mean()
    scaler=StandardScaler().fit(x[tr],sample_weight=w)
    xx=scaler.transform(x[tr])
    model=LogisticRegression(C=1.,solver='lbfgs',tol=1e-8,max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(xx,y[tr],sample_weight=w)
    # Independent gradient of the declared summed weighted objective.
    residual=w*(expit(xx@model.coef_[0]+model.intercept_[0])-y[tr])
    gradient=np.r_[xx.T@residual+model.coef_[0],residual.sum()]/w.sum()
    error=float(np.max(np.abs(gradient)))
    if not np.isfinite(error) or error>1e-5:raise ArithmeticError('Logistic stationarity check failed: '+str(error))
    prob=model.predict_proba(scaler.transform(x[test]))[:,1]
    info={'n_supported':len(tr),'n_excluded':int((~keep).sum()),'effective_n':float(w.sum()**2/(w@w)),'class_weight_sums':[float(w[y[tr]==c].sum()) for c in [0,1]],'iterations':int(model.n_iter_[0]),'normalized_gradient_max':error}
    state={'mean':scaler.mean_,'scale':scaler.scale_,'coef':model.coef_[0],'intercept':model.intercept_}
    return prob,info,state


def ridge_fit_predict(x,target,train,test):
    if np.intersect1d(train,test).size:raise ValueError('Overlapping ridge fold')
    sx=StandardScaler().fit(x[train]);sy=StandardScaler().fit(target[train])
    if (sy.var_<=0).any():raise ValueError('Constant target')
    xx=sx.transform(x[train]);yy=sy.transform(target[train])
    model=Ridge(alpha=1.,solver='svd').fit(xx,yy)
    residual=xx@model.coef_.T+model.intercept_-yy
    gradient=xx.T@residual+model.coef_.T
    error=float(max(np.max(abs(gradient)),np.max(abs(residual.sum(0))))/len(train))
    if not np.isfinite(error) or error>1e-8:raise ArithmeticError('Ridge normal equations failed')
    pred=sy.inverse_transform(model.predict(sx.transform(x[test])))
    state={'feature_mean':sx.mean_,'feature_scale':sx.scale_,'target_mean':sy.mean_,'target_scale':sy.scale_,'coef':model.coef_,'intercept':model.intercept_}
    return pred,np.broadcast_to(sy.mean_,pred.shape),error,state


def auc_summary(y,s,w,folds):
    rows=[]
    for fold in range(5):
        k=folds==fold
        if any(w[k&(y==c)].sum()<=0 for c in [0,1]):raise ValueError('Unsupported evaluation class')
        rows.append({'fold':fold,'original_auc':float(roc_auc_score(y[k],s[k])),'common_pt_auc':float(roc_auc_score(y[k],s[k],sample_weight=w[k]))})
    return {'folds':rows,'mean_fold_original_auc':float(np.mean([r['original_auc'] for r in rows])),'mean_fold_common_pt_auc':float(np.mean([r['common_pt_auc'] for r in rows])),'pooled_original_auc':float(roc_auc_score(y,s)),'pooled_common_pt_auc':float(roc_auc_score(y,s,sample_weight=w))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.with_suffix('.npz').exists():raise ValueError('Refusing overwrite')
    meta=json.loads(a.cache.with_suffix('.json').read_text())
    if meta['cache_sha256']!=digest(a.cache) or meta['protocol_sha256']!=digest('docs/frozen_probe_protocol.md'):raise ValueError('Cache/protocol mismatch')
    for src,h in meta['source_sha256'].items():
        if digest(src)!=h:raise ValueError('Changed extraction dependency')
    with np.load(a.cache,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    y=d['y'].astype(int);ids=d['query_event_ids'];q=d['query_observables']
    if set(map(tuple,ids))&set(map(tuple,d['neighbor_event_ids'])):raise ValueError('Query/reference overlap')
    folds=event_folds(y,ids);edges=np.unique(np.quantile(d['reference_observables'][:,1],np.linspace(0,1,21)))
    ew,bins=common_weights(y,q[:,1],edges)
    models=['aspen_contrastive','lhco_contrastive','random_backbone']
    inputs={n:d['z_'+n].astype(np.float64) for n in models}
    inputs.update(girth_linear=q[:,3:4].astype(np.float64),observables_linear=q.astype(np.float64))
    arrays={'event_ids':ids,'fold':folds,'y':y,'evaluation_weights':ew,'observables':q}
    classification={};regression={}
    for name,x in inputs.items():
        predictions=np.full(len(y),np.nan);fits=[]
        for fold in range(5):
            train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
            w,_=common_weights(y[train],q[train,1],edges)
            pred,info,state=logistic_fit_predict(x,y,train,test,w)
            predictions[test]=pred;fits.append(dict(fold=fold,**info))
            for k,v in state.items():arrays[f'{name}_fold{fold}_{k}']=v
        if not np.isfinite(predictions).all():raise ValueError('Missing OOF predictions')
        arrays[name+'_probability']=predictions
        classification[name]=dict(auc_summary(y,predictions,ew,folds),fits=fits)
        print('CLASSIFIER_COMPLETE',name,flush=True)
    for name in models:
        x=inputs[name];bg=y==0;predictions=np.full_like(q,np.nan,dtype=float);baseline=np.full_like(predictions,np.nan);errors=[]
        for fold in range(5):
            train=np.flatnonzero(bg&(folds!=fold));test=np.flatnonzero(bg&(folds==fold))
            pred,base,error,state=ridge_fit_predict(x,q,train,test)
            predictions[test]=pred;baseline[test]=base;errors.append(error)
            for k,v in state.items():arrays[f'{name}_ridge_fold{fold}_{k}']=v
        rows={}
        for j,label in enumerate(['mass','pt','full_multiplicity','retained_girth']):
            truth=q[bg,j];estimated=predictions[bg,j]
            sse=float(np.sum((truth-estimated)**2));sse_base=float(np.sum((truth-baseline[bg,j])**2))
            if sse_base<=0:raise ValueError('Degenerate regression baseline')
            rows[label]={'rmse':float(np.sqrt(sse/len(truth))),'r2':float(r2_score(truth,estimated)),'skill_vs_training_mean':1-sse/sse_base}
        regression[name]={'targets':rows,'normalized_normal_equation_max':max(errors)}
        arrays[name+'_observable_predictions_background']=predictions[bg]
        print('RIDGE_COMPLETE',name,flush=True)
    controls={n:auc_summary(y,d[n],ew,folds) for n in models+['girth_high','observable_density']}
    output={'stage':'supervised_frozen_development_diagnostic','classification':classification,'regression_background':regression,'fixed_unsupervised_controls':controls,'bin_counts':bins,'fold_seed':20261002,'protocol_sha256':meta['protocol_sha256'],'cache_sha256':digest(a.cache),'source_sha256':{str(s):digest(s) for s in [Path(__file__),Path('scripts/validation/gate_crosscheck.py')]},'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'limitations':['Signal-label supervised probes; not unsupervised anomaly performance.','Adaptive development sample, single encoder seed, fixed folds and regularization.','Fold training sets overlap; no independent-seed confidence interval.','Pooled OOF ranking includes cross-fold calibration differences.','Failure of a linear readout does not establish absence of nonlinear information.']}
    with a.out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    output['predictions_sha256']=digest(a.out.with_suffix('.npz'))
    with a.out.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print('FROZEN_PROBE_COMPLETE',a.out,flush=True)
if __name__=='__main__':main()

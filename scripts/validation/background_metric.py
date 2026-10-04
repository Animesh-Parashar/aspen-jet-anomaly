"""Background-fitted metric screen; no signal labels enter transformation fitting."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from gate_crosscheck import common_weights
from scientific_value_gate import metrics
from physics_ablation_comparison import weighted_auc_sorted


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fit_metric(reference):
 r=np.asarray(reference,dtype=np.float64)
 if r.ndim!=2 or len(r)<2 or r.shape[1]<1 or not np.isfinite(r).all():raise ValueError('Invalid reference')
 mu=r.mean(0);centered=r-mu;cov=centered.T@centered/len(r);scale=np.trace(cov)/r.shape[1]
 if not np.isfinite(scale) or scale<=0:raise ValueError('Zero/nonfinite total variance')
 regularized=.9*cov+.1*scale*np.eye(r.shape[1]);factor=np.linalg.cholesky(regularized)
 # L^-1 C_reg L^-T = I, independently of empirical transformed covariance.
 left=solve_triangular(factor,regularized,lower=True)
 identity=solve_triangular(factor,left.T,lower=True).T
 error=float(np.max(abs(identity-np.eye(r.shape[1]))))
 if error>1e-10:raise ArithmeticError('Whitening identity failed')
 return {'mean':mu,'factor':factor,'covariance':cov,'regularized_covariance':regularized,'identity_error':error}


def transform(x,state):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=2 or x.shape[1]!=len(state['mean']) or not np.isfinite(x).all():raise ValueError('Invalid query')
 z=solve_triangular(state['factor'],(x-state['mean']).T,lower=True).T
 if not np.isfinite(z).all():raise ArithmeticError('Nonfinite transform')
 return z


def knn(reference,query):
 return NearestNeighbors(n_neighbors=10,algorithm='brute',n_jobs=1).fit(reference).kneighbors(query)[0][:,-1]


def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists() or a.out.with_suffix('.npz').exists():raise ValueError('Refusing overwrite')
 meta=json.loads(a.cache.with_suffix('.json').read_text())
 if meta['cache_sha256']!=digest(a.cache) or meta['protocol_sha256']!=digest('docs/background_scoring_protocol.md'):raise ValueError('Input/protocol changed')
 for src,h in meta['source_sha256'].items():
  if digest(src)!=h:raise ValueError('Changed extraction source')
 with np.load(a.cache,allow_pickle=False) as f:d={k:f[k] for k in f.files}
 if len(np.unique(d['query_event_ids'],axis=0))!=len(d['y']) or set(map(tuple,d['neighbor_event_ids']))&set(map(tuple,d['query_event_ids'])):raise ValueError('Event overlap/duplicates')
 models=['aspen_contrastive','lhco_contrastive','random_backbone'];scores={n:d[n] for n in models+['girth_high','observable_density']}
 arrays={'y':d['y'],'query_event_ids':d['query_event_ids'],'neighbor_event_ids':d['neighbor_event_ids']};audit={}
 subset=np.r_[np.arange(16),10000+np.arange(16)]
 for n in models:
  r=d['r_'+n].astype(np.float64);q=d['z_'+n].astype(np.float64)
  state=fit_metric(r);zr=transform(r,state);zq=transform(q,state)
  scores[n+'_white_knn']=knn(zr,zq);scores[n+'_mahalanobis']=np.einsum('ij,ij->i',zq,zq)
  manual=np.partition(cdist(zq[subset],zr),9,axis=1)[:,9]
  np.testing.assert_allclose(manual,scores[n+'_white_knn'][subset],rtol=1e-8,atol=1e-8)
  centered=q[subset]-state['mean'];direct=np.einsum('ij,ji->i',centered,np.linalg.solve(state['regularized_covariance'],centered.T))
  np.testing.assert_allclose(direct,scores[n+'_mahalanobis'][subset],rtol=1e-10,atol=1e-10)
  raw=knn(r,q);np.testing.assert_allclose(raw,d[n],rtol=1e-5,atol=1e-6)
  audit[n]={'whitening_identity_error':state['identity_error'],'knn_direct_error':float(np.max(abs(manual-scores[n+'_white_knn'][subset]))),'raw_cache_error':float(np.max(abs(raw-d[n]))),'mahalanobis_direct_error':float(np.max(abs(direct-scores[n+'_mahalanobis'][subset]))),'regularized_condition_number':float(np.linalg.cond(state['regularized_covariance']))}
  arrays[n+'_mean']=state['mean'];arrays[n+'_factor']=state['factor'];arrays[n+'_covariance']=state['covariance']
  print('SCORES_COMPLETE',n,flush=True)
 y=d['y'];pt=d['query_observables'][:,1];edges=np.unique(np.quantile(d['reference_observables'][:,1],np.linspace(0,1,21)))
 w,rows=common_weights(y,pt,edges);names=list(scores)
 funcs={n:weighted_auc_sorted(y,scores[n]) for n in names}
 results={n:dict(metrics(y,scores[n]),common_pt_auc=funcs[n](w)) for n in names}
 for n in names:np.testing.assert_allclose(results[n]['common_pt_auc'],roc_auc_score(y,scores[n],sample_weight=w),rtol=0,atol=1e-12)
 rng=np.random.default_rng(1801);bg=np.flatnonzero(y==0);sg=np.flatnonzero(y==1);boot=np.empty((10000,len(names)))
 for b in range(len(boot)):
  ix=np.r_[rng.choice(bg,len(bg)),rng.choice(sg,len(sg))];wb,_=common_weights(y[ix],pt[ix],edges);weights=np.bincount(ix,weights=wb,minlength=len(y))
  for j,n in enumerate(names):boot[b,j]=funcs[n](weights)
  if b%1000==0:print('BOOTSTRAP',b,flush=True)
 for j,n in enumerate(names):results[n]['common_pt_auc_95']=np.quantile(boot[:,j],[.025,.975]).tolist()
 decisions={}
 for suffix in ['white_knn','mahalanobis']:
  arm='aspen_contrastive_'+suffix;i=names.index(arm);comparators=['aspen_contrastive','random_backbone_'+suffix,'girth_high','observable_density'];differences={}
  for c in comparators:
   differences[c]={'point':results[arm]['common_pt_auc']-results[c]['common_pt_auc'],'adjusted_interval':np.quantile(boot[:,i]-boot[:,names.index(c)],[.003125,.996875]).tolist()}
  decisions[arm]={'differences':differences,'advance':all(v['point']>0 and v['adjusted_interval'][0]>0 for v in differences.values())}
 arrays.update(scores);arrays['evaluation_weights']=w
 with a.out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
 result={'stage':'background_fitted_adaptive_development','results':results,'decisions':decisions,'audit':audit,'bin_counts':rows,'bootstrap_replicates':10000,'cache_sha256':digest(a.cache),'scores_sha256':digest(a.out.with_suffix('.npz')),'protocol_sha256':meta['protocol_sha256'],'source_sha256':{str(s):digest(s) for s in [Path(__file__),Path('scripts/validation/gate_crosscheck.py'),Path('scripts/validation/scientific_value_gate.py'),Path('scripts/validation/physics_ablation_comparison.py')]},'limitations':['Single encoder seed; reference fit and neighbor pool fixed in bootstrap.','Signal labels used for evaluation only; prior diagnostic outcomes informed this fixed screen.','Approximate multiplicity-adjusted bootstrap intervals are development triage, not confirmatory discovery.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print('BACKGROUND_METRIC_COMPLETE',a.out,flush=True)
if __name__=='__main__':main()

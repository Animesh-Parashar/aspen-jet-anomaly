"""Post-evaluation scorer robustness; all scores/settings fixed in protocol."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from background_metric import fit_metric,transform,knn
from physics_ablation_comparison import weighted_auc_sorted
from scripts.validation.frozen_topology_eval import load_partition,ROOT,THREE,SEEDS,folder,digest,weighted_points,summarize
from scripts.validation.scientific_value_gate import encode,observables,metrics
from scripts.validation.gate_crosscheck import common_weights
from src.models.controlled_encoder import JetBackbone,ContrastiveJetModel

PROTOCOL=Path('docs/scorer_robustness_protocol_20261003.md')
CACHE=Path('data/results/scorer_robustness_features_20261003')
LOCK=Path('data/results/scorer_robustness_lock_20261003.json')
SCORERS=('raw','mahalanobis','white_knn')
DOMAINS=('aspen','lhco','random')

def verify_lock():
    for p,h in json.loads(LOCK.read_text())['sha256'].items():
        if digest(p)!=h:raise ValueError('Frozen source changed: '+p)

def prior(topology):
    p=Path(f'data/results/frozen_final_{topology}_26302.rachel.npz')
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def extract():
    if CACHE.exists():raise ValueError('Refusing overwrite')
    if not torch.cuda.is_available():raise RuntimeError('Scheduler GPU required')
    torch.set_num_threads(1)
    ref=load_partition(ROOT,'background','reference',10000,20261004)
    bg=load_partition(ROOT,'background','test',10000,20261004)
    sig={t:load_partition(ROOT if t=='two' else THREE,'signal','test',2000,20261004) for t in ('two','three')}
    for t in sig:
        a=prior(t);q={k:np.concatenate([bg[k],sig[t][k]]) for k in bg}
        for actual,expected in [(ref['event_id'],a['neighbor_event_ids']),(q['event_id'],a['query_event_ids']),(q['label'],a['y']),(observables(ref),a['reference_observables']),(observables(q),a['query_observables'])]:
            np.testing.assert_array_equal(actual,expected)
    CACHE.mkdir();files={};provenance={}
    for seed in SEEDS:
        initial=None
        for domain in ('aspen','lhco','random'):
            if domain!='random':
                p=folder(seed,domain);m=json.loads((p/'COMPLETE.json').read_text());cfg=m['config']
                c=torch.load(p/'last.pt',map_location='cpu',weights_only=True)
                if any(c[k]!=m[k] for k in ('config','domain','objective','smoke')):raise ValueError('Checkpoint metadata')
                if cfg['training_seed']!=seed or cfg['epochs']!=100 or m['smoke']:raise ValueError('Training endpoint')
                state=torch.load(p/'initial_backbone.pt',map_location='cpu',weights_only=True)
                if initial is not None and (state.keys()!=initial.keys() or any(not torch.equal(v,initial[k]) for k,v in state.items())):raise ValueError('Initialization mismatch')
                initial=state
                model=ContrastiveJetModel(JetBackbone(**cfg['backbone']));model.load_state_dict(c['model'])
                provenance[f'{domain}_seed{seed}']=dict(checkpoint_sha256=digest(p/'last.pt'),manifest_sha256=digest(p/'COMPLETE.json'))
            else:
                model=ContrastiveJetModel(JetBackbone(**cfg['backbone']));model.backbone.load_state_dict(initial)
            model.eval().cuda();arrays={}
            for k,data in [('reference',ref),('background',bg),('two',sig['two']),('three',sig['three'])]:
                z,_=encode(model,data)
                if not np.isfinite(z).all():raise ValueError('Nonfinite embedding')
                arrays[k]=z
            model.cpu();name=f'{domain}_seed{seed}';p=CACHE/(name+'.npz')
            with p.open('xb') as f:np.savez_compressed(f,**arrays)
            files[str(p)]=digest(p);print('EXTRACTED',name,flush=True)
    result=dict(stage='post_evaluation_robustness',files=files,provenance=provenance,lock_sha256=digest(LOCK),protocol_sha256=digest(PROTOCOL))
    (CACHE/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
    print('EXTRACTION_COMPLETE',flush=True)

def all_contrasts():
    """Fixed coefficient maps over scores; preserve seed pairing."""
    out={}
    def add(name,left,right):
        coeff={}
        for n in left:coeff[n]=coeff.get(n,0)+1/len(left)
        for n in right:coeff[n]=coeff.get(n,0)-1/len(right)
        out[name]=coeff
    names=lambda d,s:[f'{d}_seed{k}_{s}' for k in SEEDS]
    for s in SCORERS:
        add(f'aspen_minus_lhco_{s}',names('aspen',s),names('lhco',s))
        for d in DOMAINS:
            if s!='raw':add(f'{d}_{s}_minus_raw',names(d,s),names(d,'raw'))
            if d!='random':
                add(f'{d}_{s}_minus_random',names(d,s),names('random',s))
                for control in ['girth_high','observable_density']:add(f'{d}_{s}_minus_{control}',names(d,s),[control])
    return out

def evaluate(out):
    if out.exists() or out.with_suffix('.npz').exists():raise ValueError('Refusing overwrite')
    meta=json.loads((CACHE/'COMPLETE.json').read_text())
    if meta['lock_sha256']!=digest(LOCK) or meta['protocol_sha256']!=digest(PROTOCOL):raise ValueError('Changed extraction provenance')
    for p,h in meta['files'].items():
        if digest(p)!=h:raise ValueError('Changed embeddings')
    data={t:prior(t) for t in ('two','three')}
    scores={t:{n:data[t][n] for n in ['girth_high','observable_density']} for t in data}
    saved={};audits={}
    for seed in SEEDS:
        for d in DOMAINS:
            name=f'{d}_seed{seed}'
            with np.load(CACHE/(name+'.npz'),allow_pickle=False) as f:a={k:f[k].astype(np.float64) for k in f.files}
            state=fit_metric(a['reference']);zr=transform(a['reference'],state)
            for k in ('mean','factor','covariance'):saved[name+'_'+k]=state[k]
            for t in data:
                q=np.concatenate([a['background'],a[t]]);zq=transform(q,state)
                raw=knn(a['reference'],q);white=knn(zr,zq);mahal=np.sum(zq*zq,axis=1)
                np.testing.assert_allclose(raw,data[t][name],rtol=1e-4,atol=1e-5)
                if abs(roc_auc_score(data[t]['y'],raw)-roc_auc_score(data[t]['y'],data[t][name]))>2e-5:raise ValueError('Raw AUC drift')
                ix=np.r_[np.arange(16),10000+np.arange(16)]
                direct=np.partition(cdist(zq[ix],zr),9,axis=1)[:,9]
                np.testing.assert_allclose(white[ix],direct,rtol=1e-8,atol=1e-8)
                delta=q[ix]-state['mean'];quadratic=np.einsum('ij,ji->i',delta,np.linalg.solve(state['regularized_covariance'],delta.T))
                np.testing.assert_allclose(mahal[ix],quadratic,rtol=1e-10,atol=1e-10)
                audits[t+'_'+name]=dict(raw_cache_error=float(np.max(abs(raw-data[t][name]))),knn_error=float(np.max(abs(direct-white[ix]))),quadratic_error=float(np.max(abs(mahal[ix]-quadratic))),whitening_identity_error=state['identity_error'])
                for s,v in [('raw',raw),('white_knn',white),('mahalanobis',mahal)]:scores[t][name+'_'+s]=v
            print('SCORED',name,flush=True)
    results={};contrasts=all_contrasts()
    for t,a in data.items():
        y=a['y'];pt=a['query_observables'][:,1];edges=np.unique(np.quantile(a['reference_observables'][:,1],np.linspace(0,1,21)))
        w,bins=common_weights(y,pt,edges);np.testing.assert_array_equal(w,a['common_pt_weights'])
        names=list(scores[t]);funcs={n:weighted_auc_sorted(y,scores[t][n]) for n in names};per={}
        for n,s in scores[t].items():
            v=metrics(y,s);balanced=float(roc_auc_score(y,s,sample_weight=w))
            np.testing.assert_allclose([funcs[n](np.ones(len(y))),funcs[n](w)],[v['auc'],balanced],rtol=0,atol=1e-12)
            per[n]=dict(original_auc=v['auc'],common_pt_auc=balanced,original_working_points=v['working_points'],common_pt_working_points=weighted_points(y,s,w))
        rng=np.random.default_rng(20261005);classes=[np.flatnonzero(y==c) for c in (0,1)];boot=np.empty((1000,2,len(names)))
        for b in range(1000):
            ix=np.concatenate([rng.choice(c,len(c),replace=True) for c in classes]);ww,_=common_weights(y[ix],pt[ix],edges)
            counts=np.bincount(ix,minlength=len(y));weighted=np.bincount(ix,weights=ww,minlength=len(y))
            for j,n in enumerate(names):boot[b,0,j]=funcs[n](counts);boot[b,1,j]=funcs[n](weighted)
        agg={};comparison={}
        for mi,metric in enumerate(['original_auc','common_pt_auc']):
            agg[metric]={}
            for d in DOMAINS:
                for s in SCORERS:
                    ns=[f'{d}_seed{k}_{s}' for k in SEEDS];v=summarize([per[n][metric] for n in ns]);v['conditional_event_bootstrap_95']=np.quantile(boot[:,mi,[names.index(n) for n in ns]].mean(1),[.025,.975]).tolist();agg[metric][d+'_'+s]=v
            comparison[metric]={}
            for name,coeff in contrasts.items():
                draws=sum(c*boot[:,mi,names.index(n)] for n,c in coeff.items())
                per_seed=[]
                for seed in SEEDS:
                    # Each coefficient is a mean over 3 paired seeds; controls are fixed.
                    per_seed.append(sum(c*per[n][metric]*(3 if '_seed' in n else 1) for n,c in coeff.items() if '_seed' not in n or f'_seed{seed}_' in n))
                v=summarize(per_seed);v['conditional_event_bootstrap_95']=np.quantile(draws,[.025,.975]).tolist();comparison[metric][name]=v
        results[t]=dict(per_score=per,summary=agg,paired_differences=comparison,bin_counts=bins,excluded_by_class={str(c):int(((y==c)&(w==0)).sum()) for c in (0,1)},effective_sample_size={str(c):float(w[y==c].sum()**2/(w[y==c]**2).sum()) for c in (0,1)},bootstrap_score_order=names)
        saved[t+'_bootstrap_auc']=boot
        for k in ['y','query_event_ids','neighbor_event_ids','common_pt_weights']:saved[t+'_'+k]=a[k]
        for n,s in scores[t].items():saved[t+'_'+n]=s
        print('BOOTSTRAP_COMPLETE',t,flush=True)
    with out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**saved)
    result=dict(stage='post_evaluation_scorer_robustness_not_independent_test',results=results,audits=audits,protocol_sha256=digest(PROTOCOL),lock_sha256=digest(LOCK),extraction_manifest_sha256=digest(CACHE/'COMPLETE.json'),scores_sha256=digest(out.with_suffix('.npz')),limitations=['Added after final-score inspection; descriptive robustness only.','Pointwise event intervals exclude training/reference fitting uncertainty and multiplicity.','No algorithmic novelty or selected-best score claim.'])
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('JOB_COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','evaluate']);p.add_argument('--out',type=Path);args=p.parse_args();verify_lock()
    if args.mode=='extract':extract()
    else:
        if args.out is None:p.error('evaluate requires --out')
        evaluate(args.out)

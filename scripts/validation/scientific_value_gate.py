"""Prospectively specified signal-validation benchmark; never final-test access."""
import argparse,hashlib,json,sys
from pathlib import Path
import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.models.controlled_encoder import JetBackbone,ContrastiveJetModel,ReconstructionJetModel


def load_validation(root,kind,count):
    if kind not in ('background','signal'):raise ValueError('Invalid validation kind')
    meta=json.loads((root/'COMPLETE.json').read_text())['metadata']
    verified=json.loads((root/'VERIFIED.json').read_text())
    if verified['status']!='passed' or meta['domain']!='lhco' or meta['schema']!='jet-preprocessing-v2.0' or meta['features']!=['eta_rel','phi_rel','log_pt_fraction','log_pt_GeV']:raise ValueError('Dataset schema mismatch')
    path=root/kind/'validation.h5'
    keys=['constituents','mask','event_id','label','jet_kinematics','n_constituents']
    values={k:[] for k in keys}
    with h5py.File(path,'r') as f:
        n=len(f['event_id']);ix=np.sort(np.random.default_rng(20261001).choice(n,count,replace=False))
        for start in range(0,n,8192):
            sel=ix[(ix>=start)&(ix<start+8192)]-start
            if len(sel):
                for k in keys:values[k].append(f[k][start:start+8192][sel])
    a={k:np.concatenate(v) for k,v in values.items()}
    if not np.all(a['label']==(kind=='signal')):raise ValueError('Wrong label')
    if len(np.unique(a['event_id'],axis=0))!=count:raise ValueError('Expected one jet per event; grouped metrics required otherwise')
    if not np.isfinite(a['constituents']).all() or a['mask'].dtype!=np.bool_ or a['mask'].all(1).any():raise ValueError('Invalid tensors')
    return a


def observables(a):
    x=a['constituents'];valid=~a['mask']
    pt=np.exp(x[:,:,3].astype(np.float64))*valid
    girth=(pt*np.hypot(x[:,:,0],x[:,:,1])).sum(1)/pt.sum(1)
    result=np.column_stack([a['jet_kinematics'][:,3],a['jet_kinematics'][:,0],a['n_constituents'],girth])
    if not np.isfinite(result).all():raise ValueError('Invalid observables')
    return result


def metrics(y,s):
    if not np.isfinite(s).all() or set(np.unique(y))!={0,1}:raise ValueError('Invalid scores/classes')
    rows=[]
    for efficiency in [.3,.5]:
        signal=np.sort(s[y==1])[::-1];threshold=signal[int(np.ceil(efficiency*len(signal)))-1]
        bg=s[y==0]>=threshold;sig=s[y==1]>=threshold
        rows.append(dict(target_signal_efficiency=efficiency,threshold=float(threshold),achieved_signal_efficiency=float(sig.mean()),background_efficiency=float(bg.mean()),background_passing=int(bg.sum()),rejection=float(1/bg.mean()) if bg.any() else None))
    return dict(auc=float(roc_auc_score(y,s)),working_points=rows)


@torch.inference_mode()
def encode(model,a,reconstruct=False):
    zs=[];errors=[]
    for i in range(0,len(a['mask']),128):
        x=torch.from_numpy(a['constituents'][i:i+128]).cuda();m=torch.from_numpy(a['mask'][i:i+128]).cuda()
        if reconstruct:
            pred,z=model(x,m);errors.append(model.reconstruction_scores(pred,x,m).cpu().numpy())
        else:z=model.encode(x,m)
        zs.append(z.cpu().numpy())
    return np.concatenate(zs),np.concatenate(errors) if errors else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists() or args.out.with_suffix('.npz').exists():raise ValueError('Refusing overwrite')
    if not torch.cuda.is_available():raise RuntimeError('Scheduler GPU required')
    torch.set_num_threads(1)
    root=Path('data/processed_v2/full_24632.rachel/lhco')
    bg=load_validation(root,'background',20000);sig=load_validation(root,'signal',2000)
    if set(map(tuple,bg['event_id']))&set(map(tuple,sig['event_id'])):raise ValueError('Background/signal event overlap')
    order=np.random.default_rng(1701).permutation(20000);ref={k:v[order[:10000]] for k,v in bg.items()};query={k:np.concatenate([v[order[10000:]],sig[k]]) for k,v in bg.items()}
    assert not set(map(tuple,ref['event_id']))&set(map(tuple,query['event_id']))
    y=np.r_[np.zeros(10000,dtype=int),np.ones(2000,dtype=int)]
    r=observables(ref);q=observables(query)
    scores={'mass_high':q[:,0],'pt_high':q[:,1],'multiplicity_high':q[:,2],'multiplicity_low':-q[:,2],'girth_high':q[:,3]}
    scale=r.std(0)
    if (scale<=0).any():raise ValueError('Degenerate reference observable')
    scores['observable_density']=NearestNeighbors(n_neighbors=10,n_jobs=1,algorithm='brute').fit((r-r.mean(0))/scale).kneighbors((q-r.mean(0))/scale)[0][:,-1]
    folders={'aspen_contrastive':Path('data/results/convergence_extensions_26172.rachel/aspen_contrastive'),'lhco_contrastive':Path('data/results/convergence_extensions_26172.rachel/lhco_contrastive'),'aspen_reconstruction':Path('data/results/reconstruction_200_26176.rachel')}
    hashes={};initial=None;config=None
    for name,folder in folders.items():
        manifest=json.loads((folder/'COMPLETE.json').read_text());cfg=manifest['config']
        if manifest['smoke'] or len(manifest['history'])!=cfg['epochs']:raise ValueError('Incomplete checkpoint')
        if config is not None and {k:v for k,v in cfg.items() if k!='epochs'}!={k:v for k,v in config.items() if k!='epochs'}:raise ValueError('Unmatched config')
        config=cfg
        expected=200 if name=='aspen_reconstruction' else 100
        if cfg['epochs']!=expected:raise ValueError('Wrong endpoint')
        for source,digest in manifest['source_sha256'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:raise ValueError('Changed training source')
        c=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        if any(c[k]!=manifest[k] for k in ['config','domain','objective','smoke']):raise ValueError('Checkpoint mismatch')
        ae=manifest['objective']=='reconstruction';model=(ReconstructionJetModel if ae else ContrastiveJetModel)(JetBackbone(**cfg['backbone']))
        model.load_state_dict(c['model']);model.eval().cuda()
        zr,_=encode(model,ref);zq,error=encode(model,query,ae)
        scores[name]=NearestNeighbors(n_neighbors=10,algorithm='brute',n_jobs=1).fit(zr).kneighbors(zq)[0][:,-1]
        if ae:scores['reconstruction_error']=error
        state=torch.load(folder/'initial_backbone.pt',map_location='cpu',weights_only=True)
        if initial is not None and any(not torch.equal(v,initial[k]) for k,v in state.items()):raise ValueError('Unmatched initialization')
        initial=state;hashes[name]=hashlib.sha256((folder/'last.pt').read_bytes()).hexdigest()
    model=ContrastiveJetModel(JetBackbone(**config['backbone']));model.backbone.load_state_dict(initial);model.eval().cuda()
    zr,_=encode(model,ref);zq,_=encode(model,query)
    scores['random_backbone']=NearestNeighbors(n_neighbors=10,algorithm='brute',n_jobs=1).fit(zr).kneighbors(zq)[0][:,-1]
    result={k:metrics(y,v) for k,v in scores.items()}
    rng=np.random.default_rng(1703);boot={k:[] for k in scores}
    for _ in range(300):
        ix=np.r_[rng.integers(10000,size=10000),10000+rng.integers(2000,size=2000)]
        for k,v in scores.items():boot[k].append(float(roc_auc_score(y[ix],v[ix])))
    edges=np.quantile(r[:,1],[0,.25,.5,.75,1]);bins=np.searchsorted(edges[1:-1],q[:,1],side='right')
    for k,v in result.items():
        v['auc_percentile_95']=np.quantile(boot[k],[.025,.975]).tolist()
        v['paired_auc_difference']={b:dict(point=v['auc']-result[b]['auc'],percentile_95=np.quantile(np.array(boot[k])-np.array(boot[b]),[.025,.975]).tolist()) for b in ['observable_density','random_backbone']}
        v['pt_bins']=[]
        for i in range(4):
            ix=bins==i;ns=int(y[ix].sum());nb=int(ix.sum()-ns)
            v['pt_bins'].append(dict(bin=i,signal=ns,background=nb,auc=float(roc_auc_score(y[ix],scores[k][ix])) if min(ns,nb)>=100 else None))
    output=dict(stage='signal_validation_development_not_final',results=result,checkpoint_sha256=hashes,pt_edges=edges.tolist(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(Path('docs/scientific_value_gate.md').read_bytes()).hexdigest(),
        limitations=['Single training seed; bootstrap conditional on fixed neighbor pool.','Signal validation explicitly used for development; final test/reference partitions excluded.','One R&D signal sample; no multiple-topology or discovery claim.','Zero-passing-background rejection is null; working points are empirical, not calibrated selections.'])
    arrays=dict(y=y,query_event_ids=query['event_id'],neighbor_event_ids=ref['event_id'],query_observables=q,reference_observables=r,**scores)
    with args.out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    tmp=args.out.with_suffix('.tmp');tmp.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');tmp.replace(args.out)
    print('JOB_COMPLETE',args.out,flush=True)
if __name__=='__main__':main()

"""Development-only kNN dependence on validation data; no signal/test access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.controlled_loader import ControlledDataset
from src.models.controlled_encoder import JetBackbone
from scripts.validation.representation_diagnostics import encode, rho


def event_halves(ids):
    unique, inverse = np.unique(ids, axis=0, return_inverse=True)
    perm = np.random.default_rng(20260930).permutation(len(unique))
    ref_events = np.zeros(len(unique), dtype=bool)
    ref_events[perm[:len(unique)//2]] = True
    return ref_events[inverse]


def binned(values, scores, selected):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 6)))
    if len(edges)==1:
        return [dict(low=float(edges[0]),high=float(edges[0]),n=len(values),selected_fraction=float(selected.mean()),median_score=float(np.median(scores)))]
    bins = np.searchsorted(edges[1:-1], values, side='right')
    return [dict(low=float(edges[i]), high=float(edges[i+1]),
                 n=int((bins==i).sum()),
                 selected_fraction=float(selected[bins==i].mean()),
                 median_score=float(np.median(scores[bins==i])))
            for i in range(len(edges)-1) if (bins==i).any()]


def observables(data):
    p=data.provenance
    indices=np.sort(np.random.default_rng(p['sample_seed']).choice(p['n_available'],p['n_selected'],replace=False))
    keys=['jet_kinematics','n_constituents','retained_pt_fraction','event_id']
    values={k:[] for k in keys}
    with h5py.File(p['path'],'r') as f:
        for start in range(0,p['n_available'],8192):
            sub=indices[(indices>=start)&(indices<start+8192)]-start
            if len(sub):
                for k in keys: values[k].append(f[k][start:start+8192][sub])
    values={k:np.concatenate(v) for k,v in values.items()}
    if not np.array_equal(values['event_id'],data.event_ids):
        raise ValueError('Observable/event alignment failed')
    kin=values['jet_kinematics']
    out=dict(constituent_sum_mass=kin[:,3],vector_jet_pt=kin[:,0],
             full_stored_multiplicity=values['n_constituents'],
             retained_multiplicity=(~data.mask).sum(1).numpy(),
             retained_pt_fraction=values['retained_pt_fraction'])
    if any(not np.isfinite(v).all() for v in out.values()): raise ValueError('Nonfinite observable')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--extended',type=Path)
    args=p.parse_args()
    if args.out.exists(): raise ValueError('Refusing overwrite')
    if not torch.cuda.is_available(): raise RuntimeError('Scheduler GPU required')
    torch.set_num_threads(1)
    names=['aspen_contrastive','aspen_reconstruction','lhco_contrastive']
    folders={n:args.runs/n for n in names}
    folders['aspen_reconstruction_100']=Path('data/results/reconstruction_extension_26160.rachel')
    if args.extended: folders['aspen_reconstruction_unmasked_100']=args.extended
    models={}; hashes={}; cfg=None; initial=None
    for name,folder in folders.items():
        m=json.loads((folder/'COMPLETE.json').read_text())
        if m['smoke'] or len(m['history'])!=m['config']['epochs']: raise ValueError('Incomplete run')
        if cfg is None: cfg=m['config']
        if {k:v for k,v in cfg.items() if k not in ('epochs','decoder_attention_mask')} != {k:v for k,v in m['config'].items() if k not in ('epochs','decoder_attention_mask')}: raise ValueError('Config mismatch')
        for source,h in m['source_sha256'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=h: raise ValueError('Changed source '+source)
        c=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        if c['config']!=m['config'] or c['domain']!=m['domain'] or c['objective']!=m['objective'] or c['smoke']: raise ValueError('Checkpoint mismatch')
        if name == 'aspen_reconstruction_unmasked_100' and (m['config'].get('decoder_attention_mask') != 'none' or m['config']['epochs'] != 100): raise ValueError('Wrong ablation checkpoint')
        model=JetBackbone(**cfg['backbone'])
        model.load_state_dict({k.removeprefix('backbone.'):v for k,v in c['model'].items() if k.startswith('backbone.')})
        models[name]=model.eval().cuda()
        state=torch.load(folder/'initial_backbone.pt',map_location='cpu',weights_only=True)
        if initial is not None and any(not torch.equal(v,initial[k]) for k,v in state.items()): raise ValueError('Initialization mismatch')
        initial=state
        hashes[name]=hashlib.sha256((folder/'last.pt').read_bytes()).hexdigest()
    model=JetBackbone(**cfg['backbone']); model.load_state_dict(initial)
    models['random']=model.eval().cuda()
    results={}
    cache={}
    for domain in ['aspen','lhco']:
        data=ControlledDataset(cfg[domain+'_root'],domain,'validation',8192,cfg['sample_seed'])
        obs=observables(data); ref=event_halves(data.event_ids); query=~ref
        if set(map(tuple,data.event_ids[ref])) & set(map(tuple,data.event_ids[query])): raise ValueError('Event overlap')
        result=dict(provenance=data.provenance,neighbor_jets=int(ref.sum()),query_jets=int(query.sum()),
                    event_ids_sha256=hashlib.sha256(data.event_ids.tobytes()).hexdigest(),models={})
        cache[domain+'_event_ids']=data.event_ids[query]
        for key,value in obs.items(): cache[domain+'_'+key]=value[query]
        for name,model in models.items():
            z=encode(model,data.x,data.mask)
            scores=NearestNeighbors(n_neighbors=10,algorithm='brute',n_jobs=1).fit(z[ref]).kneighbors(z[query])[0][:,-1]
            cache[domain+'_score_'+name]=scores
            threshold=np.quantile(scores,.9); selected=scores>=threshold
            r=dict(threshold=float(threshold),selected_fraction=float(selected.mean()),observables={})
            for key,v in obs.items():
                v=v[query]
                r['observables'][key]=dict(score_spearman=rho(scores,v),
                    norm_spearman=rho(np.linalg.norm(z[query],axis=1),v),
                    bins=binned(v,scores,selected))
            # Conditional descriptive correlations in fixed validation pT quantile bins.
            pt=obs['vector_jet_pt'][query]; edges=np.unique(np.quantile(pt,[0,.25,.5,.75,1])); group=np.searchsorted(edges[1:-1],pt,side='right')
            r['within_pt_bins']=[dict(low=float(edges[i]),high=float(edges[i+1]),n=int((group==i).sum()),
                mass_rho=rho(scores[group==i],obs['constituent_sum_mass'][query][group==i]),
                multiplicity_rho=rho(scores[group==i],obs['full_stored_multiplicity'][query][group==i])) for i in range(len(edges)-1)]
            result['models'][name]=r
        results[domain]=result
    output=dict(stage='validation_only_physics_dependence',k=10,selection='top 10 percent of each model query score; descriptive only',
        mass_definition='Unweighted stored-constituent four-vector sum BEFORE truncation; not CMS soft-drop mass',
        limitations=['No signal sensitivity, uncertainty intervals, background closure or discovery inference.',
        'Jet-weighted; event disjoint neighbor/query samples are both validation subsets.',
        'Within-pT correlations are descriptive, not a causal adjustment.',
        'Aspen stored constituents and LHCO clustered particles differ in response and selection.'],
        checkpoint_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,numpy=np.__version__,results=results)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.with_suffix('.npz').open('xb') as f: np.savez_compressed(f,**cache)
    with args.out.open('x') as f: json.dump(output,f,indent=2,allow_nan=False)
    print('COMPLETE',args.out,flush=True)

if __name__=='__main__': main()

"""Cache fixed development-query representations, without fitting probe models."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scientific_value_gate import load_validation,encode,observables
from src.models.controlled_encoder import JetBackbone,ContrastiveJetModel


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.with_suffix('.npz').exists():raise ValueError('Refusing overwrite')
    if not torch.cuda.is_available():raise RuntimeError('Scheduled GPU required')
    torch.set_num_threads(1)
    parent=Path('data/results/scientific_value_gate_26199.rachel.npz')
    with np.load(parent,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    root=Path('data/processed_v2/full_24632.rachel/lhco')
    bg=load_validation(root,'background',20000);sig=load_validation(root,'signal',2000)
    order=np.random.default_rng(1701).permutation(20000)
    ref={k:v[order[:10000]] for k,v in bg.items()}
    query={k:np.concatenate([v[order[10000:]],sig[k]]) for k,v in bg.items()}
    for value,expected in [(query['event_id'],d['query_event_ids']),(ref['event_id'],d['neighbor_event_ids']),(query['label'],d['y']),(observables(query),d['query_observables']),(observables(ref),d['reference_observables'])]:
        np.testing.assert_array_equal(value,expected)
    if len(np.unique(query['event_id'],axis=0))!=len(query['event_id']):raise ValueError('Duplicate event IDs')
    if set(map(tuple,query['event_id']))&set(map(tuple,ref['event_id'])):raise ValueError('Reference overlap')
    provenance={};initial=None;cfg=None
    for name in ['aspen_contrastive','lhco_contrastive']:
        folder=Path('data/results/convergence_extensions_26172.rachel')/name
        manifest=json.loads((folder/'COMPLETE.json').read_text());cfg=manifest['config']
        if manifest['smoke'] or cfg['epochs']!=100 or len(manifest['history'])!=100:raise ValueError('Wrong checkpoint endpoint')
        for source,h in manifest['source_sha256'].items():
            if digest(source)!=h:raise ValueError('Training source changed: '+source)
        c=torch.load(folder/'last.pt',weights_only=True,map_location='cpu')
        for k in ['config','domain','objective','smoke']:
            if c[k]!=manifest[k]:raise ValueError('Checkpoint/manifest mismatch')
        model=ContrastiveJetModel(JetBackbone(**cfg['backbone']));model.load_state_dict(c['model']);model.eval().cuda()
        z,_=encode(model,query)
        if z.shape!=(12000,128) or not np.isfinite(z).all():raise ValueError('Invalid embeddings')
        # Repeat a batch in eval mode to catch accidental stochastic extraction.
        repeated,_=encode(model,{k:v[:128] for k,v in query.items()})
        np.testing.assert_allclose(z[:128],repeated,rtol=0,atol=0)
        d['z_'+name]=z
        state=torch.load(folder/'initial_backbone.pt',weights_only=True,map_location='cpu')
        if initial is not None and (state.keys()!=initial.keys() or any(not torch.equal(v,initial[k]) for k,v in state.items())):raise ValueError('Different random controls')
        initial=state
        provenance[name]={'checkpoint_sha256':digest(folder/'last.pt'),'manifest_sha256':digest(folder/'COMPLETE.json'),'initial_sha256':digest(folder/'initial_backbone.pt')}
    model=ContrastiveJetModel(JetBackbone(**cfg['backbone']));model.backbone.load_state_dict(initial);model.eval().cuda()
    d['z_random_backbone'],_=encode(model,query)
    if not np.isfinite(d['z_random_backbone']).all():raise ValueError('Invalid random embeddings')
    with a.out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**d)
    sources=[Path(__file__),Path('scripts/validation/scientific_value_gate.py'),Path('src/models/controlled_encoder.py')]
    result={'status':'complete','stage':'frozen_supervised_development_diagnostic','models':provenance,'parent_cache_sha256':digest(parent),'cache_sha256':digest(a.out.with_suffix('.npz')),'protocol_sha256':digest('docs/frozen_probe_protocol.md'),'source_sha256':{str(s):digest(s) for s in sources},'torch':torch.__version__,'numpy':np.__version__}
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print('EXTRACTION_COMPLETE',a.out,flush=True)
if __name__=='__main__':main()

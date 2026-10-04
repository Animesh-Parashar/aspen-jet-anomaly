"""Fixed-event development scoring of the two prospective augmentation arms."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scientific_value_gate import load_validation,encode,observables
from src.models.controlled_encoder import JetBackbone,ContrastiveJetModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.with_suffix('.npz').exists():raise ValueError('Refusing overwrite')
    if not torch.cuda.is_available():raise RuntimeError('Scheduled GPU required')
    torch.set_num_threads(1)
    cache=Path('data/results/scientific_value_gate_26199.rachel.npz')
    with np.load(cache,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    root=Path('data/processed_v2/full_24632.rachel/lhco')
    bg=load_validation(root,'background',20000);sig=load_validation(root,'signal',2000)
    order=np.random.default_rng(1701).permutation(20000)
    ref={k:v[order[:10000]] for k,v in bg.items()}
    query={k:np.concatenate([v[order[10000:]],sig[k]]) for k,v in bg.items()}
    np.testing.assert_array_equal(query['event_id'],arrays['query_event_ids'])
    np.testing.assert_array_equal(ref['event_id'],arrays['neighbor_event_ids'])
    np.testing.assert_array_equal(observables(query),arrays['query_observables'])
    initial=torch.load('data/results/convergence_extensions_26172.rachel/aspen_contrastive/initial_backbone.pt',weights_only=True,map_location='cpu')
    hashes={};checks={};ranks={};sample=None
    arrays['query_split_eligible']=query['mask'].any(1)
    for variant in ['rotation_split','rotation_soft']:
        folder=a.training/variant;manifest=json.loads((folder/'COMPLETE.json').read_text());cfg=manifest['config']
        assert not manifest['smoke'] and cfg['epochs']==100 and len(manifest['history'])==100
        assert cfg['variant']==variant and manifest['domain']=='aspen' and manifest['objective']=='contrastive'
        for src,digest in manifest['source_sha256'].items():
            assert hashlib.sha256(Path(src).read_bytes()).hexdigest()==digest
        state=torch.load(folder/'initial_backbone.pt',weights_only=True,map_location='cpu')
        assert all(torch.equal(v,state[k]) for k,v in initial.items())
        with np.load(folder/'sample_event_ids.npz',allow_pickle=False) as z:
            current={k:z[k] for k in z.files}
        if sample is not None:
            for k in sample:np.testing.assert_array_equal(sample[k],current[k])
        sample=current
        c=torch.load(folder/'last.pt',weights_only=True,map_location='cpu')
        assert c['config']==cfg and not c['smoke']
        model=ContrastiveJetModel(JetBackbone(**cfg['backbone']));model.load_state_dict(c['model']);model.eval().cuda()
        zr,_=encode(model,ref);zq,_=encode(model,query)
        eigen=np.linalg.eigvalsh(np.cov(zr.astype(np.float64),rowvar=False)).clip(0)
        positive=eigen[eigen>0]/eigen.sum()
        ranks[variant]=float(np.exp(-(positive*np.log(positive)).sum()))
        score=NearestNeighbors(n_neighbors=10,algorithm='brute',n_jobs=1).fit(zr).kneighbors(zq)[0][:,-1]
        subset=np.r_[np.arange(16),10000+np.arange(16)]
        manual=np.partition(cdist(zq[subset],zr),9,axis=1)[:,9]
        np.testing.assert_allclose(score[subset],manual,rtol=1e-5,atol=1e-6)
        arrays[variant]=score;checks[variant]=float(np.max(abs(score[subset]-manual)))
        hashes[variant]=hashlib.sha256((folder/'last.pt').read_bytes()).hexdigest()
    with a.out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    with a.out.open('x') as f:json.dump({'stage':'signal_validation_development','checkpoint_sha256':hashes,'direct_distance_max_error':checks,'reference_covariance_effective_rank':ranks,'parent_cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},f,indent=2)
    print('SCORING_COMPLETE',a.out,flush=True)
if __name__=='__main__':main()

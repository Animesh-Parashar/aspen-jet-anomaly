"""Reference-only extraction for the fixed background metric screen."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scientific_value_gate import load_validation,encode,observables
from src.models.controlled_encoder import JetBackbone,ContrastiveJetModel

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists() or a.out.with_suffix('.npz').exists():raise ValueError('Refusing overwrite')
 if not torch.cuda.is_available():raise RuntimeError('Scheduled GPU required')
 torch.set_num_threads(1)
 parent=Path('data/results/frozen_probe_features_20261001.npz');meta=json.loads(parent.with_suffix('.json').read_text())
 if digest(parent)!=meta['cache_sha256']:raise ValueError('Parent hash mismatch')
 with np.load(parent,allow_pickle=False) as f:d={k:f[k] for k in f.files}
 bg=load_validation(Path('data/processed_v2/full_24632.rachel/lhco'),'background',20000)
 order=np.random.default_rng(1701).permutation(20000);ref={k:v[order[:10000]] for k,v in bg.items()}
 np.testing.assert_array_equal(ref['event_id'],d['neighbor_event_ids']);np.testing.assert_array_equal(observables(ref),d['reference_observables'])
 if set(map(tuple,ref['event_id']))&set(map(tuple,d['query_event_ids'])):raise ValueError('Overlap')
 if len(np.unique(ref['event_id'],axis=0))!=10000:raise ValueError('Duplicate neighbors')
 initial=None;provenance={}
 for name in ['aspen_contrastive','lhco_contrastive']:
  folder=Path('data/results/convergence_extensions_26172.rachel')/name
  if digest(folder/'last.pt')!=meta['models'][name]['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
  manifest=json.loads((folder/'COMPLETE.json').read_text());cfg=manifest['config']
  for src,h in manifest['source_sha256'].items():
   if digest(src)!=h:raise ValueError('Changed training source')
  c=torch.load(folder/'last.pt',weights_only=True,map_location='cpu')
  model=ContrastiveJetModel(JetBackbone(**cfg['backbone']));model.load_state_dict(c['model']);model.eval().cuda()
  d['r_'+name],_=encode(model,ref)
  # Same first query batch as the existing cache, testing extraction consistency.
  check={k:v[order[10000:10128]] for k,v in bg.items()};z,_=encode(model,check)
  np.testing.assert_allclose(z,d['z_'+name][:128],rtol=1e-6,atol=1e-6)
  state=torch.load(folder/'initial_backbone.pt',weights_only=True,map_location='cpu')
  if digest(folder/'initial_backbone.pt')!=meta['models'][name]['initial_sha256']:raise ValueError('Initial state mismatch')
  if initial is not None and any(not torch.equal(v,initial[k]) for k,v in state.items()):raise ValueError('Random mismatch')
  initial=state;provenance[name]=meta['models'][name]
 model=ContrastiveJetModel(JetBackbone(**cfg['backbone']));model.backbone.load_state_dict(initial);model.eval().cuda()
 d['r_random_backbone'],_=encode(model,ref)
 z,_=encode(model,check);np.testing.assert_allclose(z,d['z_random_backbone'][:128],rtol=1e-6,atol=1e-6)
 for n in ['aspen_contrastive','lhco_contrastive','random_backbone']:
  if d['r_'+n].shape!=(10000,128) or not np.isfinite(d['r_'+n]).all():raise ValueError('Invalid representation')
 with a.out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**d)
 result={'status':'complete','parent_sha256':digest(parent),'cache_sha256':digest(a.out.with_suffix('.npz')),'models':provenance,'protocol_sha256':digest('docs/background_scoring_protocol.md'),'source_sha256':{str(p):digest(p) for p in [Path(__file__),Path('scripts/validation/scientific_value_gate.py'),Path('src/models/controlled_encoder.py')]}}
 with a.out.open('x') as f:json.dump(result,f,indent=2)
 print('BACKGROUND_EXTRACTION_COMPLETE',flush=True)
if __name__=='__main__':main()

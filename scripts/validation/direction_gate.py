"""Background-only paired geometry audit for deciding whether a pilot is justified."""
import argparse,json,hashlib,sys
from pathlib import Path
import h5py
import numpy as np
from scipy.special import logsumexp
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.data.controlled_loader import ControlledDataset
from src.models.controlled_encoder import JetBackbone,ContrastiveJetModel
from src.augmentations.controlled_augmentations import CoreAugmentation


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scale_pt(x,mask,factor):
 if not np.isfinite(factor) or factor<=0:raise ValueError('Invalid scale')
 out=x.clone();out[...,3]+=np.log(factor)
 return out.masked_fill(mask[...,None],0)


def negative_indices(pt,seed):
 pt=np.asarray(pt)
 if pt.ndim!=1 or not np.isfinite(pt).all() or (pt<=0).any():raise ValueError('Invalid pT')
 rng=np.random.default_rng(seed);edges=np.quantile(pt,np.linspace(0,1,9))
 if len(np.unique(edges))!=9:raise ValueError('Tied quantile boundaries')
 bins=np.searchsorted(edges[1:-1],pt,side='right')
 # Fixed independent placebo permutation across repeats.
 placebo=bins[np.random.default_rng(20261009).permutation(len(pt))]
 indices={};allrows=np.arange(len(pt))
 for name,b in [('global',None),('matched',bins),('placebo',placebo)]:
  rows=[]
  for i in allrows:
   candidates=allrows[allrows!=i] if b is None else allrows[(b==b[i])&(allrows!=i)]
   if len(candidates)<127:raise ValueError('Insufficient negative bank')
   rows.append(rng.choice(candidates,127,replace=False))
  ix=np.asarray(rows)
  if (ix==allrows[:,None]).any() or any(len(np.unique(row))!=127 for row in ix):raise ValueError('Negative contamination')
  indices[name]=ix
 return indices,edges,np.bincount(bins,minlength=8)


def bank_loss(z1,z2,indices):
 z1=np.asarray(z1,dtype=np.float64);z2=np.asarray(z2,dtype=np.float64)
 for z in [z1,z2]:
  if not np.isfinite(z).all():raise ValueError('Nonfinite embeddings')
  np.testing.assert_allclose(np.linalg.norm(z,axis=1),1,rtol=2e-6,atol=2e-6)
 positive=np.einsum('ij,ij->i',z1,z2);loss=np.zeros(len(z1));negative=np.zeros(len(z1))
 for start in range(0,len(z1),64):
  sl=slice(start,start+64);ix=indices[sl];neg=np.concatenate([z1[ix],z2[ix]],axis=1)
  for anchor in [z1[sl],z2[sl]]:
   sim=np.einsum('ij,ikj->ik',anchor,neg)
   logits=np.column_stack([positive[sl],sim])/.1
   loss[sl]+=(logsumexp(logits,axis=1)-positive[sl]/.1)/2
   negative[sl]+=sim.mean(1)/2
 return loss,negative,positive


@torch.inference_mode()
def projections(model,x,mask):
 return np.concatenate([model(x[i:i+128].cuda(),mask[i:i+128].cuda()).cpu().numpy() for i in range(0,len(x),128)])


def main():
 p=argparse.ArgumentParser();p.add_argument('--domain',choices=['aspen','lhco'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists() or a.out.with_suffix('.npz').exists():raise ValueError('Refusing overwrite')
 if not torch.cuda.is_available():raise RuntimeError('Scheduled GPU required')
 torch.set_num_threads(1);domain=a.domain
 folder=Path('data/results/convergence_extensions_26172.rachel')/(domain+'_contrastive')
 manifest=json.loads((folder/'COMPLETE.json').read_text());cfg=manifest['config']
 if manifest['smoke'] or cfg['epochs']!=100:raise ValueError('Wrong checkpoint')
 for src,h in manifest['source_sha256'].items():
  if digest(src)!=h:raise ValueError('Changed training source')
 d=ControlledDataset(cfg[domain+'_root'],domain,'validation',8192,cfg['sample_seed'])
 _,first=np.unique(d.event_ids,axis=0,return_index=True);selected=np.sort(first)[:4096]
 if len(selected)!=4096:raise ValueError('Too few unique events')
 x=d.x[selected];mask=d.mask[selected];ids=d.event_ids[selected]
 # Reproduce loader's exact sampled raw-row positions, then enforce ID equality.
 with h5py.File(d.provenance['path'],'r') as f:
  rows=np.sort(np.random.default_rng(cfg['sample_seed']).choice(len(f['event_id']),8192,replace=False))[selected]
  np.testing.assert_array_equal(f['event_id'][rows],ids)
  pt=f['jet_kinematics'][rows,0].astype(np.float64)
  if not np.all(f['label'][rows]==(-1 if domain=='aspen' else 0)):raise ValueError('Unexpected labels')
 torch.manual_seed(20261004);(x1,m1),(x2,m2)=CoreAugmentation()(x,mask)
 arrays={'event_ids':ids,'pt':pt};zs={};scaling={};sources={}
 for name in ['trained','random']:
  torch.manual_seed(17);model=ContrastiveJetModel(JetBackbone(**cfg['backbone']))
  initial=torch.load(folder/'initial_backbone.pt',weights_only=True,map_location='cpu')
  if any(not torch.equal(v,model.backbone.state_dict()[k]) for k,v in initial.items()):raise ValueError('Random initialization mismatch')
  if name=='trained':
   c=torch.load(folder/'last.pt',weights_only=True,map_location='cpu');model.load_state_dict(c['model'])
  model.eval().cuda();z1=projections(model,x1,m1);z2=projections(model,x2,m2);zs[name]=(z1,z2)
  base=projections(model,x,mask);scaling[name]={}
  for factor in [.9,1.1]:
   z=projections(model,scale_pt(x,mask,factor),mask);disp=1-np.einsum('ij,ij->i',base,z)
   arrays[f'{name}_scale_{factor}']=disp;scaling[name][str(factor)]={'mean_cosine_displacement':float(disp.mean()),'quantiles':np.quantile(disp,[.1,.5,.9,.99]).tolist()}
  scaling[name]['rotation_mean_displacement']=float(np.mean(1-np.einsum('ij,ij->i',z1,z2)))
 results={};effects={n:[] for n in zs}
 for seed in [20261005,20261006,20261007]:
  indices,edges,counts=negative_indices(pt,seed);results[str(seed)]={}
  for name,(z1,z2) in zs.items():
   values={k:bank_loss(z1,z2,ix) for k,ix in indices.items()}
   effect=values['matched'][0]-values['placebo'][0];placebo=values['placebo'][0]-values['global'][0]
   effects[name].append(float(effect.mean()))
   summary={k:{'mean_loss':float(v[0].mean()),'mean_negative_cosine':float(v[1].mean()),'mean_positive_cosine':float(v[2].mean())} for k,v in values.items()}
   summary.update(matched_minus_placebo=float(effect.mean()),placebo_minus_global=float(placebo.mean()),matched_minus_global=float((values['matched'][0]-values['global'][0]).mean()))
   if seed==20261005:
    rng=np.random.default_rng(20261008);boot=[effect[rng.integers(len(effect),size=len(effect))].mean() for _ in range(2000)]
    summary['conditional_anchor_bootstrap_95']=np.quantile(boot,[.025,.975]).tolist()
   for k,v in values.items():arrays[f'{name}_{seed}_{k}_loss']=v[0]
   results[str(seed)][name]=summary
  print('REPEAT_COMPLETE',seed,flush=True)
 eligible=all(results[str(s)]['trained']['matched_minus_placebo']>=.05 and results[str(s)]['trained']['matched_minus_placebo']-results[str(s)]['random']['matched_minus_placebo']>=.02 and abs(results[str(s)]['trained']['placebo_minus_global'])<=.02 for s in [20261005,20261006,20261007])
 with a.out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
 result={'domain':domain,'stage':'background_only_direction_gate','pilot_geometry_eligible':eligible,'results':results,'scaling':scaling,'pt_octile_edges':edges.tolist(),'counts':counts.tolist(),'checkpoint_sha256':digest(folder/'last.pt'),'sample':d.provenance,'protocol_sha256':digest('docs/direction_gate_protocol.md'),'script_sha256':digest(__file__),'arrays_sha256':digest(a.out.with_suffix('.npz')),'limitations':['No anomaly performance or causal training intervention measured.','Shared fixed negative banks: bootstrap intervals are conditional descriptive summaries.','Geometry eligibility does not establish novelty or authorize a training campaign.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print('DIRECTION_GATE_COMPLETE',a.out,flush=True)
if __name__=='__main__':main()

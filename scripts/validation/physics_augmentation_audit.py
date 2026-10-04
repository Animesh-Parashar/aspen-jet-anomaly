"""Background-only audit before the prospective augmentation screen."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.data.controlled_loader import ControlledDataset
from src.augmentations.physics_ablation import PhysicsAugmentation
from src.augmentations.controlled_augmentations import split_collinear


def observables(x,m):
    x=x.double(); pt=x[...,3].exp().masked_fill(m,0)
    eta,phi=x[...,0],x[...,1]
    p4=torch.stack([pt*eta.cosh(),pt*phi.cos(),pt*phi.sin(),pt*eta.sinh()],-1).sum(1)
    mass=(p4[:,0]**2-p4[:,1:].square().sum(1)).clamp_min(0).sqrt()
    return torch.stack([pt.sum(1),p4[:,1:3].square().sum(1).sqrt(),mass,(pt*(eta.square()+phi.square()).sqrt()).sum(1)/pt.sum(1)],1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists(): raise ValueError('Refusing overwrite')
    torch.set_num_threads(1);torch.manual_seed(88173)
    cfg=json.loads(Path('configs/controlled_convergence.json').read_text())
    d=ControlledDataset(cfg['aspen_root'],'aspen','validation',8192,cfg['sample_seed'])
    x,m=d.x,d.mask
    fraction=m.any(1).float().mean().item()
    before=observables(x,m)
    report={'status':'passed','sample':d.provenance,'event_ids_sha256':hashlib.sha256(d.event_ids.tobytes()).hexdigest(),'split_eligible_fraction':fraction,'eligible_arms':['rotation_soft'],'arms':{},'observable_order':['scalar_pt','vector_pt','mass','girth'],'scope':'retained massless constituents in original axis coordinates; background validation only'}
    if fraction>=.1: report['eligible_arms'].insert(0,'rotation_split')
    for variant in ['rotation_split','rotation_soft']:
        torch.manual_seed(88173);y,n=PhysicsAugmentation(variant).view(x,m)
        assert torch.isfinite(y).all() and (y[n]==0).all()
        if variant=='rotation_soft':
            assert torch.equal(n,m) and torch.equal(y[...,2:],x[...,2:])
        else:
            assert torch.equal((~n).sum(1),(~m).sum(1)+m.any(1).long())
        after=observables(y,n)
        delta=after-before
        report['arms'][variant]={'attempted_fraction':1.,'split_applied_fraction':fraction if variant=='rotation_split' else 0.,'delta_quantiles':torch.quantile(delta,torch.tensor([0.,.01,.5,.99,1.],dtype=torch.float64),dim=0).tolist(),'quantiles':[0,.01,.5,.99,1]}
    # Isolate splitting from the approximate rotation for conservation tests.
    y,n=split_collinear(x,m,1.)
    torch.testing.assert_close(observables(y,n)[:,:2],before[:,:2],rtol=2e-6,atol=2e-4)
    # Same angular draws as each variant: quantify rotation chart rejection directly.
    torch.manual_seed(88173);angles=torch.empty(len(x)).uniform_(-torch.pi,torch.pi)
    phi=angles.sin()[:,None]*x[...,0]+angles.cos()[:,None]*x[...,1]
    report['rotation_chart_rejected_fraction']=((phi.abs()>torch.pi)&~m).any(1).float().mean().item()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out.with_suffix('.npz'),event_ids=d.event_ids)
    a.out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
if __name__=='__main__': main()

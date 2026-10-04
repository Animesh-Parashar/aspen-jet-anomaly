"""Paired validation intervention; changes token support, not physical jet identity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.data.controlled_loader import ControlledDataset
from src.models.controlled_encoder import JetBackbone, ReconstructionJetModel, ContrastiveJetModel
from scripts.validation.physics_dependence import event_halves


def truncate(x,mask,keep):
    if not 1<=keep<=x.shape[1]:raise ValueError('Invalid cutoff')
    outmask=mask | (torch.arange(x.shape[1],device=x.device)[None,:]>=keep)
    return x.masked_fill(outmask[...,None],0),outmask


def feature_errors(model,pred,target,mask):
    residual=pred*model.backbone.feature_scale-target
    residual=residual.clone()
    residual[...,1]=torch.atan2(residual[...,1].sin(),residual[...,1].cos())
    squared=(residual/model.backbone.feature_scale).square().masked_fill(mask[...,None],0)
    return squared.sum(1)/(~mask).sum(1)[:,None]


def decode(model,z,mask):
    h=model.expand(z).reshape(-1,model.max_constituents,model.decoder_width)
    for layer in model.decoder:h=layer(h,src_key_padding_mask=mask)
    return model.output(h)


def summary(v):
    if not np.isfinite(v).all():raise ValueError('Nonfinite metric')
    return dict(n=len(v),mean=float(np.mean(v)),quantiles=np.quantile(v,[.05,.5,.95]).tolist())


@torch.inference_mode()
def run(model,data):
    x,mask=data.x,data.mask
    outputs={}
    for keep in [50,40,30]:
        xx,mm=truncate(x,mask,keep)
        zs=[]; losses=[]; frozen=[]; delta_padding=[]
        for i in range(0,len(x),128):
            a=xx[i:i+128].cuda();m=mm[i:i+128].cuda()
            original=x[i:i+128].cuda();oldmask=mask[i:i+128].cuda()
            z=model.encode(a,m);zs.append(z.cpu().numpy())
            if isinstance(model,ReconstructionJetModel):
                pred,_=model(a,m)
                # Compare on the SAME retained targets and denominator for both predictions.
                baseline,z0=model(original,oldmask)
                errors=feature_errors(model,pred,original,m)
                base_errors=feature_errors(model,baseline,original,m)
                torch.testing.assert_close(errors.mean(1),model.reconstruction_scores(pred,original,m))
                losses.append(torch.stack([base_errors,errors],1).cpu().numpy())
                # Hold latent fixed: isolate decoder mask dependence, not a physically realizable jet.
                fixed=decode(model,z0,m)
                frozen.append((feature_errors(model,fixed,original,m)-base_errors).cpu().numpy())
            if keep==50:
                dirty=a.masked_fill(m[...,None],12345.)
                delta_padding.append((model.encode(dirty,m)-z).abs().max().item())
        outputs[keep]=dict(z=np.concatenate(zs),loss=np.concatenate(losses) if losses else None,
            frozen=np.concatenate(frozen) if frozen else None,padding=max(delta_padding) if delta_padding else None)
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists() or args.out.with_suffix('.npz').exists():raise ValueError('Refusing overwrite')
    if not torch.cuda.is_available():raise RuntimeError('Scheduler GPU required')
    torch.set_num_threads(1)
    root=Path('data/results/controlled_convergence_24711.rachel')
    folders={'aspen_contrastive':root/'aspen_contrastive','lhco_contrastive':root/'lhco_contrastive',
             'reconstruction_50':root/'aspen_reconstruction','reconstruction_100':Path('data/results/reconstruction_extension_26160.rachel')}
    models={};hashes={};cfg=None;initial=None
    for name,folder in folders.items():
        manifest=json.loads((folder/'COMPLETE.json').read_text());c=manifest['config']
        if manifest['smoke'] or len(manifest['history'])!=c['epochs']:raise ValueError('Incomplete run')
        if cfg is not None and {k:v for k,v in c.items() if k!='epochs'}!={k:v for k,v in cfg.items() if k!='epochs'}:raise ValueError('Configuration mismatch')
        cfg=c
        for source,digest in manifest['source_sha256'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:raise ValueError('Changed training source')
        checkpoint=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        if any(checkpoint[k]!=manifest[k] for k in ['config','domain','objective','smoke']):raise ValueError('Checkpoint mismatch')
        model=(ReconstructionJetModel if manifest['objective']=='reconstruction' else ContrastiveJetModel)(JetBackbone(**c['backbone']))
        model.load_state_dict(checkpoint['model']);models[name]=model.eval().cuda()
        state=torch.load(folder/'initial_backbone.pt',map_location='cpu',weights_only=True)
        if initial is not None and any(not torch.equal(v,initial[k]) for k,v in state.items()):raise ValueError('Initialization mismatch')
        initial=state;hashes[name]=hashlib.sha256((folder/'last.pt').read_bytes()).hexdigest()
    random=ContrastiveJetModel(JetBackbone(**cfg['backbone']));random.backbone.load_state_dict(initial);models['random']=random.eval().cuda()
    results={};cache={}
    for domain in ['aspen','lhco']:
        data=ControlledDataset(cfg[domain+'_root'],domain,'validation',2048,cfg['sample_seed'])
        ref=event_halves(data.event_ids);query=~ref
        assert not set(map(tuple,data.event_ids[ref])) & set(map(tuple,data.event_ids[query]))
        # Truncation relies on valid tokens forming a descending-pT prefix.
        if (data.mask[:,:-1]&~data.mask[:,1:]).any():raise ValueError('Non-prefix mask')
        for row,m in zip(data.x,data.mask):
            if (torch.diff(row[~m,3])>1e-5).any():raise ValueError('Unsorted tokens')
        results[domain]=dict(provenance=data.provenance,models={})
        cache[domain+'_query_ids']=data.event_ids[query]
        counts=(~data.mask).sum(1).numpy();cache[domain+'_counts']=counts[query]
        for name,model in models.items():
            out=run(model,data);z0=out[50]['z'];knn=NearestNeighbors(n_neighbors=10,n_jobs=1,algorithm='brute').fit(z0[ref])
            base=knn.kneighbors(z0[query])[0][:,-1];rows={}
            if out[50]['padding']>1e-5:raise ValueError('Padding contamination')
            for keep in [40,30]:
                changed=counts[query]>keep;v=out[keep];score=knn.kneighbors(v['z'][query])[0][:,-1]
                displacement=np.linalg.norm(v['z'][query]-z0[query],axis=1)
                if not np.allclose(v['z'][query][~changed],z0[query][~changed],atol=1e-5,rtol=1e-5):raise ValueError('No-op intervention changed embedding')
                record=dict(affected_queries=int(changed.sum()),embedding_displacement=summary(displacement[changed]),
                    score_change_fixed_neighbors=summary((score-base)[changed]))
                prefix=f'{domain}_{name}_{keep}'
                cache[prefix+'_score_delta']=(score-base);cache[prefix+'_embedding_displacement']=displacement
                if v['loss'] is not None:
                    delta=v['loss'][query,1]-v['loss'][query,0]
                    record['feature_loss_delta_same_support']=[summary(delta[changed,j]) for j in range(4)]
                    record['decoder_mask_only_delta']=[summary(v['frozen'][query][changed,j]) for j in range(4)]
                    cache[prefix+'_feature_delta']=delta;cache[prefix+'_decoder_mask_delta']=v['frozen'][query]
                rows[str(keep)]=record
            results[domain]['models'][name]=dict(padding_max_difference=out[50]['padding'],interventions=rows)
    output=dict(stage='validation_paired_truncation_audit',checkpoint_sha256=hashes,results=results,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features=['eta_rel','phi_rel','log_pt_fraction','log_pt_GeV'],
        limitations=['50 to 40/30 token removal changes physical information and multiplicity simultaneously; not a causal separation.',
        'Original axes and scalar-pT denominator retained; no renormalization or reclustering.',
        'Neighbors fixed at original 50-token input: sensitivity experiment, not alternate final scoring.',
        'Loss comparison uses same retained target support for baseline and intervention.',
        'Decoder mask-only intervention holds latent fixed and is a mechanistic diagnostic, not a physical sample.',
        'Single-seed validation, no signal/test access or confidence intervals.'])
    with args.out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**cache)
    with args.out.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print('JOB_COMPLETE',args.out,flush=True)
if __name__=='__main__':main()

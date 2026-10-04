"""Background-calibrated efficiency and jet-mass distortion; descriptive only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import rel_entr

PROTOCOL=Path('docs/physics_operating_points_protocol_20261003.md')
LOCK=Path('data/results/physics_operating_points_lock_20261003.json')
SEEDS=(17,29,43)
TARGETS=(.1,.05)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def wilson(k,n):
    if n==0:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,mid-half)),float(min(1,mid+half))]

def threshold(scores,acceptance,counts=None):
    s=np.asarray(scores,dtype=float)
    if s.ndim!=1 or not len(s) or not np.isfinite(s).all() or not 0<acceptance<1:raise ValueError('Invalid calibration')
    c=np.ones(len(s),dtype=int) if counts is None else np.asarray(counts)
    if c.shape!=s.shape or not np.isfinite(c).all() or (c<0).any() or not np.equal(c,np.floor(c)).all() or c.sum()<=0:raise ValueError('Invalid counts')
    order=np.argsort(-s,kind='stable');ix=np.searchsorted(np.cumsum(c[order]),np.ceil(acceptance*c.sum()),side='left')
    return float(s[order[ix]])

def quantile_edges(x):
    x=np.asarray(x)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid bin data')
    internal=np.unique(np.quantile(x,np.linspace(0,1,11)[1:-1]))
    return np.r_[-np.inf,internal,np.inf]

class Shape:
    def __init__(self,mass,edges):
        self.mass=np.asarray(mass);self.edges=edges
        if self.mass.ndim!=1 or not np.isfinite(self.mass).all():raise ValueError('Invalid mass')
        _,self.group=np.unique(self.mass,return_inverse=True)
        self.bin=np.searchsorted(edges[1:-1],self.mass,side='right')
        self.ng=int(self.group.max())+1;self.nb=len(edges)-1
    def evaluate(self,selected,counts=None):
        keep=np.asarray(selected,dtype=bool);c=np.ones(len(keep)) if counts is None else np.asarray(counts,dtype=float)
        if keep.shape!=self.mass.shape or c.shape!=keep.shape or not np.isfinite(c).all() or (c<0).any() or c.sum()<=0:raise ValueError('Invalid assessment counts')
        selected_counts=c*keep
        if selected_counts.sum()==0:return np.array([np.nan,np.nan])
        inclusive=np.bincount(self.group,weights=c,minlength=self.ng)/c.sum()
        passed=np.bincount(self.group,weights=selected_counts,minlength=self.ng)/selected_counts.sum()
        ks=np.max(abs(np.cumsum(inclusive)-np.cumsum(passed)))
        p=np.bincount(self.bin,weights=c,minlength=self.nb)/c.sum()
        q=np.bincount(self.bin,weights=selected_counts,minlength=self.nb)/selected_counts.sum();m=(p+q)/2
        js=.5*(rel_entr(p,m).sum()+rel_entr(q,m).sum())
        return np.array([ks,js])

def interval(draws):
    d=np.asarray(draws);valid=np.isfinite(d)
    return dict(percentile_95=np.quantile(d[valid],[.025,.975]).tolist() if valid.any() else None,valid_replicates=int(valid.sum()))

def scalar(x):return float(x) if np.isfinite(x) else None

def bin_rates(values,edges,selected):
    bins=np.searchsorted(edges[1:-1],values,side='right');rows=[]
    for b in range(len(edges)-1):
        n=int((bins==b).sum());k=int(((bins==b)&selected).sum())
        rows.append(dict(bin=b,total=n,selected=k,acceptance=k/n if n else None,conditional_wilson_95=wilson(k,n)))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists() or args.out.with_suffix('.npz').exists():raise ValueError('Refusing overwrite')
    for f,h in json.loads(LOCK.read_text())['sha256'].items():
        if digest(f)!=h:raise ValueError('Frozen artifact changed: '+f)
    path=Path('data/results/scorer_robustness_26312.rachel.npz');metadata=json.loads(path.with_suffix('.json').read_text())
    with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    physical={}
    for t in ('two','three'):
        with np.load(f'data/results/frozen_final_{t}_26302.rachel.npz',allow_pickle=False) as f:
            physical[t]={k:f[k] for k in ['y','query_event_ids','neighbor_event_ids','query_observables']}
        for k in ['y','query_event_ids','neighbor_event_ids']:np.testing.assert_array_equal(a[t+'_'+k],physical[t][k])
        np.testing.assert_array_equal(physical[t]['y'],np.r_[np.zeros(10000),np.ones(2000)])
        ids=physical[t]['query_event_ids'];ref=physical[t]['neighbor_event_ids']
        if len(np.unique(ids,axis=0))!=12000 or set(map(tuple,ids))&set(map(tuple,ref)):raise ValueError('Event overlap')
    np.testing.assert_array_equal(physical['two']['query_event_ids'][:10000],physical['three']['query_event_ids'][:10000])
    np.testing.assert_array_equal(physical['two']['query_observables'][:10000],physical['three']['query_observables'][:10000])
    names=metadata['results']['two']['bootstrap_score_order']
    if len(names)!=29 or set(names)!=set(metadata['results']['three']['bootstrap_score_order']):raise ValueError('Wrong arms')
    for n in names:np.testing.assert_allclose(a['two_'+n][:10000],a['three_'+n][:10000],rtol=0,atol=1e-12)
    perm=np.random.default_rng(20261006).permutation(10000);cal,assess=perm[:5000],perm[5000:]
    obs=physical['two']['query_observables'];mass_edges=quantile_edges(obs[cal,0]);pt_edges=quantile_edges(obs[cal,1]);shape=Shape(obs[assess,0],mass_edges)
    # rows: arm x operating point; columns: bg efficiency, two/three signal efficiency, KS, JSD.
    cases=[(n,target) for n in names for target in TARGETS];point=np.empty((len(cases),5));rows={}
    scorecal={n:a['two_'+n][cal] for n in names};scorebg={n:a['two_'+n][assess] for n in names}
    sig={t:{n:a[t+'_'+n][10000:] for n in names} for t in ('two','three')}
    nulls={}
    for i,(n,target) in enumerate(cases):
        cut=threshold(scorecal[n],target);keep=scorebg[n]>=cut;sh=shape.evaluate(keep)
        point[i]=[keep.mean(),(sig['two'][n]>=cut).mean(),(sig['three'][n]>=cut).mean(),*sh]
        row=dict(score=n,target_background_acceptance=target,threshold=cut,calibration_acceptance=float((scorecal[n]>=cut).mean()),assessment_background_selected=int(keep.sum()),assessment_background_total=5000,background_acceptance=float(keep.mean()),background_conditional_wilson_95=wilson(int(keep.sum()),5000),mass_ks=scalar(sh[0]),mass_js_nats=scalar(sh[1]),mass_bins=bin_rates(obs[assess,0],mass_edges,keep),pt_bins=bin_rates(obs[assess,1],pt_edges,keep),signals={})
        for t in ('two','three'):
            k=int((sig[t][n]>=cut).sum());row['signals'][t]=dict(selected=k,total=2000,efficiency=k/2000,conditional_wilson_95=wilson(k,2000))
        k=int(keep.sum())
        if k not in nulls:
            rng=np.random.default_rng(20261008);draws=[]
            for _ in range(500):
                selected=np.zeros(5000,dtype=bool);selected[rng.choice(5000,k,replace=False)]=True;draws.append(shape.evaluate(selected))
            draws=np.array(draws);nulls[k]=dict(selected_count=k,ks_median=scalar(np.nanmedian(draws[:,0])) if k else None,js_median=scalar(np.nanmedian(draws[:,1])) if k else None,ks_range=interval(draws[:,0]),js_range=interval(draws[:,1]))
        row['count_matched_random_selection']=nulls[k]
        nb=np.bincount(shape.bin,minlength=shape.nb);ns=np.bincount(shape.bin,weights=keep,minlength=shape.nb)
        row['selected_to_inclusive_mass_density_ratio']=[float((ns[b]/k)/(nb[b]/5000)) if k and nb[b] else None for b in range(shape.nb)]
        rows[str(i)]=row
    print('POINT_ESTIMATES_COMPLETE',flush=True)
    rng=np.random.default_rng(20261007);boot=np.empty((1000,len(cases),5))
    for b in range(1000):
        cc=np.bincount(rng.integers(5000,size=5000),minlength=5000);bc=np.bincount(rng.integers(5000,size=5000),minlength=5000)
        sc=[np.bincount(rng.integers(2000,size=2000),minlength=2000) for _ in range(2)]
        for i,(n,target) in enumerate(cases):
            cut=threshold(scorecal[n],target,cc);keep=scorebg[n]>=cut
            boot[b,i]=[np.dot(bc,keep)/5000,np.dot(sc[0],sig['two'][n]>=cut)/2000,np.dot(sc[1],sig['three'][n]>=cut)/2000,*shape.evaluate(keep,bc)]
        if b%100==0:print('BOOTSTRAP',b,flush=True)
    metrics=['background_acceptance','two_signal_efficiency','three_signal_efficiency','mass_ks','mass_js_nats']
    for i in range(len(cases)):rows[str(i)]['refitted_threshold_bootstrap']={m:interval(boot[:,i,j]) for j,m in enumerate(metrics)}
    aggregate={};comparisons={}
    def summarize_indices(ix,other=None):
        v=point[ix].copy();d=boot[:,ix,:].copy()
        if other is not None:v-=point[other];d-=boot[:,other,:]
        return {m:dict(per_seed=[scalar(x) for x in v[:,j]],mean=scalar(v[:,j].mean()),sample_sd=scalar(v[:,j].std(ddof=1)),event_interval=interval(d[:,:,j].mean(1))) for j,m in enumerate(metrics)}
    for target in TARGETS:
        for domain in ('aspen','lhco','random'):
            for scorer in ('raw','mahalanobis','white_knn'):
                name=f'{domain}_{scorer}_{target}';ix=[cases.index((f'{domain}_seed{s}_{scorer}',target)) for s in SEEDS]
                aggregate[name]=summarize_indices(ix)
                if domain!='random':
                    ri=[cases.index((f'random_seed{s}_{scorer}',target)) for s in SEEDS]
                    comparisons[name+'_minus_random']=summarize_indices(ix,ri)
                    gi=[cases.index(('girth_high',target))]*3
                    comparisons[name+'_minus_girth']=summarize_indices(ix,gi)
    arrays=dict(point_estimates=point,bootstrap=boot,calibration_event_ids=physical['two']['query_event_ids'][cal],assessment_event_ids=physical['two']['query_event_ids'][assess],mass_edges=mass_edges,pt_edges=pt_edges)
    for t in ('two','three'):arrays[t+'_signal_event_ids']=physical[t]['query_event_ids'][10000:]
    with args.out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    output=dict(stage='post_evaluation_physics_diagnostic',rows=rows,seed_summary=aggregate,paired_differences=comparisons,metric_order=metrics,mass_internal_edges=mass_edges[1:-1].tolist(),pt_internal_edges=pt_edges[1:-1].tolist(),outer_edges='negative and positive infinity; no assessment overflow exclusions',bootstrap_replicates=1000,random_selection_replicates=500,protocol_sha256=digest(PROTOCOL),lock_sha256=digest(LOCK),arrays_sha256=digest(args.out.with_suffix('.npz')),limitations=['Post-evaluation public-sample reuse, not independent confirmation.','Jet mass, not dijet mass or search significance.','Coarse marginal mass/pT acceptance is not causal or conditional decorrelation.','Pointwise bootstrap refits thresholds but conditions on models, reference pool, bins and split.','Actual background acceptance may differ between methods and nominal targets.'])
    with args.out.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print('JOB_COMPLETE',args.out,flush=True)
if __name__=='__main__':main()

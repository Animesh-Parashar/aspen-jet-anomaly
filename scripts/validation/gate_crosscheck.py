"""Independent cached-score audit plus prespecified common-pT development comparison."""
import argparse,hashlib,json
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score


def rank_auc(y,s):
    n=int(y.sum());m=len(y)-n
    return float((rankdata(s,method='average')[y==1].sum()-n*(n+1)/2)/(n*m))


def common_weights(y,pt,edges):
    bins=np.searchsorted(edges[1:-1],pt,side='right');w=np.zeros(len(y));rows=[]
    # Exclude extrapolation outside the reference-defined range.
    inside=(pt>=edges[0])&(pt<=edges[-1])
    for b in range(len(edges)-1):
        a=inside&(bins==b)&(y==0);s=inside&(bins==b)&(y==1);nb=int(a.sum());ns=int(s.sum())
        supported=min(nb,ns)>=30
        if supported:
            target=min(nb,ns);w[a]=target/nb;w[s]=target/ns
        rows.append(dict(bin=b,background=nb,signal=ns,supported=supported))
    if not w.any():raise ValueError('No common support')
    return w,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    cache=Path('data/results/scientific_value_gate_26199.rachel.npz');original=json.loads(cache.with_suffix('.json').read_text())
    with np.load(cache,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    y=a['y'];ids=a['query_event_ids'];neighbors=a['neighbor_event_ids']
    if len(np.unique(ids,axis=0))!=len(ids) or set(map(tuple,ids))&set(map(tuple,neighbors)):raise ValueError('Event overlap')
    # Independent direct HDF5 label and physical-quantity lookup, no shared loader.
    root=Path('data/processed_v2/full_24632.rachel/lhco');checked=0
    for kind,label in [('background',0),('signal',1)]:
        with h5py.File(root/kind/'validation.h5','r') as f:
            source_ids=f['event_id'][:];lookup={tuple(e):i for i,e in enumerate(source_ids)}
            selected=np.flatnonzero(y==label)
            rows=np.array([lookup[tuple(e)] for e in ids[selected]])
            labels=f['label'][:];kin=f['jet_kinematics'][:];multiplicity=f['n_constituents'][:]
            assert np.all(labels[rows]==label)
            np.testing.assert_allclose(kin[rows,0],a['query_observables'][selected,1],rtol=0,atol=0)
            np.testing.assert_allclose(kin[rows,3],a['query_observables'][selected,0],rtol=0,atol=0)
            np.testing.assert_array_equal(multiplicity[rows],a['query_observables'][selected,2]);checked+=len(rows)
            if kind=='background':assert all(tuple(e) in lookup for e in neighbors)
    names=list(original['results']);auc_errors={n:abs(rank_auc(y,a[n])-original['results'][n]['auc']) for n in names}
    if max(auc_errors.values())>1e-12:raise ValueError('AUC mismatch')
    r=a['reference_observables'];q=a['query_observables'];subset=np.r_[np.arange(16),10000+np.arange(16)]
    manual=np.partition(cdist((q[subset]-r.mean(0))/r.std(0),(r-r.mean(0))/r.std(0)),9,axis=1)[:,9]
    np.testing.assert_allclose(manual,a['observable_density'][subset],rtol=1e-5,atol=1e-6)
    # New declared development population: equal class mass within supported pT bins.
    edges=np.unique(np.quantile(r[:,1],np.linspace(0,1,21)));w,rows=common_weights(y,q[:,1],edges)
    results={n:dict(original_auc=original['results'][n]['auc'],common_pt_auc=float(roc_auc_score(y,a[n],sample_weight=w))) for n in names}
    boot={n:[] for n in names};rng=np.random.default_rng(1801);bg=np.flatnonzero(y==0);sg=np.flatnonzero(y==1)
    for _ in range(300):
        ix=np.r_[rng.choice(bg,len(bg)),rng.choice(sg,len(sg))];ww,_=common_weights(y[ix],q[ix,1],edges)
        for n in names:boot[n].append(float(roc_auc_score(y[ix],a[n][ix],sample_weight=ww)))
    for n,v in results.items():
        v['common_pt_auc_95']=np.quantile(boot[n],[.025,.975]).tolist()
        v['difference_vs_density_95']=np.quantile(np.array(boot[n])-np.array(boot['observable_density']),[.025,.975]).tolist()
    output=dict(audit=dict(events_checked=checked,auc_max_error=max(auc_errors.values()),density_max_error=float(np.max(abs(manual-a['observable_density'][subset])))),
        results=results,pt_edges=edges.tolist(),bin_counts=rows,effective_sample_size={str(c):float(w[y==c].sum()**2/(w[y==c]**2).sum()) for c in [0,1]},
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),limitations=['Post-gate development diagnostic, not a replacement final benchmark.','Fixed original scores/neighbors; pT-only coarse balancing leaves other differences and within-bin pT structure.','Bootstrap recomputes class weights; fixed bin edges/model/reference; pointwise uncertainty only.'])
    with args.out.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print('JOB_COMPLETE',args.out,flush=True)
if __name__=='__main__':main()

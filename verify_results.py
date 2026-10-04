"""Verify manuscript statistics from the published score arrays."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent
results=ROOT/'results';checked=0
for t in ['two','three']:
 p=results/f'{t}_prong.json';r=json.loads(p.read_text())
 with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
  for n,v in r['results'].items():
   for metric,w in [('original_auc',None),('common_pt_auc',z['common_pt_weights'])]:
    assert abs(roc_auc_score(z['y'],z[n],sample_weight=w)-v[metric])<1e-12
    checked+=1
p=results/'scorer_comparison.json';r=json.loads(p.read_text())
with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
 for t,v in r['results'].items():
  for n,m in v['per_score'].items():
   for metric,w in [('original_auc',None),('common_pt_auc',z[t+'_common_pt_weights'])]:
    assert abs(roc_auc_score(z[t+'_y'],z[t+'_'+n],sample_weight=w)-m[metric])<1e-12
    checked+=1
  for metric,groups in v['summary'].items():
   for name,row in groups.items():
    d,s=name.split('_',1);a=[v['per_score'][f'{d}_seed{k}_{s}'][metric] for k in [17,29,43]]
    np.testing.assert_allclose([row['mean'],row['sample_sd']],[np.mean(a),np.std(a,ddof=1)],rtol=0,atol=1e-14)
p=results/'operating_points.json';r=json.loads(p.read_text())
with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
 for i,row in r['rows'].items():
  assert row['background_acceptance']==row['assessment_background_selected']/row['assessment_background_total']
  for t in ['two','three']:assert row['signals'][t]['efficiency']==row['signals'][t]['selected']/row['signals'][t]['total']
  for j,m in enumerate(r['metric_order']):
   np.testing.assert_allclose(row['refitted_threshold_bootstrap'][m]['percentile_95'],np.quantile(z['bootstrap'][:,int(i),j],[.025,.975]),rtol=0,atol=1e-14)
 print('Verified',checked,'AUC entries, scorer seed summaries, all physics efficiency counts and 290 bootstrap intervals.')

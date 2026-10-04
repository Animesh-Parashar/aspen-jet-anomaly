"""Regenerate paper figures, LaTeX tables and portable numerical supplements."""
from pathlib import Path
import csv,hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent;OUT=HERE/'submission'
load=lambda p:json.loads((ROOT/p).read_text())
final={t:load(f'data/results/frozen_final_{t}_26302.rachel.json') for t in ['two','three']}
scorer=load('data/results/scorer_robustness_26312.rachel.json')['results']
physics=load('data/results/physics_operating_points_26331.rachel.json')
# Verify all evidence manifests before converting results into paper assets.
for fname in ['final_evidence.json','scorer_evidence.json','physics_evidence.json','replication_evidence.json','diagnostic_evidence.json','signal_kinematics_evidence.json']:
 for p,h in json.loads((HERE/fname).read_text()).items():
  assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p

def table(path,header,rows,alignment):
 text=['\\begin{tabular}{'+alignment+'}','\\toprule',header+r' \\',r'\midrule']
 text += [' & '.join(row)+r' \\' for row in rows]
 text += [r'\bottomrule',r'\end{tabular}']
 (OUT/'tables'/path).write_text('\n'.join(text)+'\n')
def pm(v):return f"${v['mean']:.4f}\\pm{v['sample_sd']:.4f}$"
rows=[]
for n,label in [('aspen','Aspen'),('lhco','Simulation'),('random','Random'),
                ('mass_high','High mass'),('pt_high','High $p_T$'),
                ('multiplicity_high','High multiplicity'),('multiplicity_low','Low multiplicity'),
                ('girth_high','Girth'),('observable_density','Observable density')]:
 vals=[]
 for t in ['two','three']:
  for m in ['original_auc','common_pt_auc']:
   vals.append(pm(final[t]['summary'][m][n]) if n in ['aspen','lhco','random'] else f"{final[t]['results'][n][m]:.4f}")
 rows.append([label]+vals)
table('final.tex','Score & 2p original & 2p balanced & 3p original & 3p balanced',rows,'lrrrr')
rows=[];csvrows=[]
for d,label in [('aspen','Aspen'),('lhco','Simulation'),('random','Random')]:
 for s,sl in [('raw','Raw kNN'),('mahalanobis','Mahalanobis'),('white_knn','Whitened kNN')]:
  rows.append([label,sl]+[pm(scorer[t]['summary'][m][d+'_'+s]) for t in ['two','three'] for m in ['original_auc','common_pt_auc']])
  for seed in [17,29,43]:
   csvrows.append([d,s,seed]+[scorer[t]['per_score'][f'{d}_seed{seed}_{s}'][m] for t in ['two','three'] for m in ['original_auc','common_pt_auc']])
table('scorers.tex','Backbone & Scorer & 2p original & 2p balanced & 3p original & 3p balanced',rows,'llrrrr')
with (OUT/'auc_per_seed.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['backbone','scorer','seed','two_original','two_balanced','three_original','three_balanced']);w.writerows(csvrows)
domain_labels={'aspen':'Aspen','lhco':'Simulation','random':'Random'}
score_labels={'raw':'Raw kNN','mahalanobis':'Mahalanobis','white_knn':'Whitened kNN'}
table('seeds.tex','Backbone & Scorer & Seed & 2p original & 2p balanced & 3p original & 3p balanced',[[domain_labels[v[0]],score_labels[v[1]],str(v[2])]+[f'{x:.4f}' for x in v[3:]] for v in csvrows],'llrrrrr')
for target in [.1,.05]:
 rows=[]
 for d,label in [('aspen','Aspen'),('lhco','Simulation'),('random','Random')]:
  for s,sl in [('raw','Raw kNN'),('mahalanobis','Mahalanobis'),('white_knn','Whitened kNN')]:
   v=physics['seed_summary'][f'{d}_{s}_{target}'];vals=[]
   for m in physics['metric_order']:
    scale=100 if m in ['background_acceptance','two_signal_efficiency','three_signal_efficiency'] else 1
    vals.append(f"${scale*v[m]['mean']:.3f}\\pm{scale*v[m]['sample_sd']:.3f}$")
   rows.append([label,sl]+vals)
 for name,label in [('girth_high','Girth'),('observable_density','Observable density')]:
  v=next(v for v in physics['rows'].values() if v['score']==name and v['target_background_acceptance']==target)
  vals=[100*v['background_acceptance'],100*v['signals']['two']['efficiency'],100*v['signals']['three']['efficiency'],v['mass_ks'],v['mass_js_nats']]
  rows.append([label,'--']+[f'{x:.3f}' for x in vals])
 table(f'physics_{int(100*target)}.tex','Backbone & Scorer & BG (\\%) & 2p (\\%) & 3p (\\%) & KS & JSD',rows,'llrrrrr')
# Full machine-readable uncertainty and binned profiles accompany the package.
for fname in ['scorer_robustness_26312.rachel.json','physics_operating_points_26331.rachel.json']:
 data=load('data/results/'+fname)
 # Keep public ancillary terminology focused on numerical validation.
 if 'audits' in data:data['numerical_checks']=data.pop('audits')
 (OUT/fname).write_text(json.dumps(data,indent=2)+'\n')
# AUC: every seed, no best-seed selection or confidence-band substitution.
colors={'aspen':'#0072B2','lhco':'#D55E00','random':'#009E73'}
fig,axes=plt.subplots(2,2,figsize=(9,6),sharey=True)
for i,t in enumerate(['two','three']):
 for j,m in enumerate(['original_auc','common_pt_auc']):
  ax=axes[i,j]
  for k,d in enumerate(['aspen','lhco','random']):
   vals=final[t]['summary'][m][d]['per_seed'];ax.scatter(k+np.array([-.09,0,.09]),vals,color=colors[d],s=30);ax.plot([k-.16,k+.16],[np.mean(vals)]*2,color=colors[d],lw=2)
  ax.axhline(final[t]['results']['girth_high'][m],color='black',ls='--',label='Girth')
  ax.axhline(final[t]['results']['observable_density'][m],color='#CC79A7',ls=':',label='Observable density')
  ax.axhline(.5,color='gray',lw=.6,label='Random ranking');ax.set_xticks(range(3),['Aspen','Simulation','Random']);ax.set_ylim(.25,.85);ax.set_title(t.capitalize()+'-prong; '+('original' if j==0 else 'common-pT'));ax.grid(axis='y',alpha=.15)
  if j==0:ax.set_ylabel('AUC')
axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'figures/auc_seeds.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
for ax,t in zip(axes,['two','three']):
 for di,d in enumerate(['aspen','lhco','random']):
  means=[scorer[t]['summary']['common_pt_auc'][d+'_'+s]['mean'] for s in ['raw','mahalanobis','white_knn']]
  positions=np.arange(3)+(di-1)*.22
  ax.scatter(positions,means,marker='_',color=colors[d],s=180,label=domain_labels[d])
  for x,s in zip(positions,['raw','mahalanobis','white_knn']):ax.scatter(x+np.array([-.045,0,.045]),scorer[t]['summary']['common_pt_auc'][d+'_'+s]['per_seed'],color=colors[d],s=16,alpha=.7)
 ax.axhline(final[t]['results']['girth_high']['common_pt_auc'],color='black',ls='--',label='Girth');ax.set_xticks(range(3),['Raw kNN','Mahalanobis','Whitened kNN']);ax.set_title(t.capitalize()+'-prong');ax.set_ylim(.35,.76);ax.grid(alpha=.15)
axes[0].set_ylabel('Common-pT AUC');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'figures/scorer_sensitivity.pdf');plt.close(fig)
shutil.copyfile(HERE/'figures/physics_tradeoff.pdf',OUT/'figures/physics_tradeoff.pdf')
# Seed range is descriptive, not a confidence interval; controls use Wilson CIs.
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
for ax,field,title in zip(axes,['mass_bins','pt_bins'],['Jet-mass quantile bin','Jet-pT quantile bin']):
 for d in ['aspen','lhco','random']:
  profiles=[]
  for seed in [17,29,43]:
   v=next(v for v in physics['rows'].values() if v['score']==f'{d}_seed{seed}_raw' and v['target_background_acceptance']==.05)
   profiles.append([x['acceptance'] for x in v[field]])
  a=np.asarray(profiles)
  ax.plot(np.arange(1,11),a.mean(0),color=colors[d],lw=1.5,label=domain_labels[d])
  ax.fill_between(np.arange(1,11),a.min(0),a.max(0),color=colors[d],alpha=.13)
 for n,c,l in [('girth_high','black','Girth'),('observable_density','#CC79A7','Observable density')]:
  v=next(v for v in physics['rows'].values() if v['score']==n and v['target_background_acceptance']==.05)
  y=np.array([x['acceptance'] for x in v[field]]);ci=np.array([x['conditional_wilson_95'] for x in v[field]])
  ax.errorbar(np.arange(1,11),y,yerr=np.maximum(0,np.array([y-ci[:,0],ci[:,1]-y])),color=c,ls='--',label=l,capsize=2,lw=1)
 ax.axhline(.05,color='gray',lw=.7);ax.set_xlabel(title);ax.grid(alpha=.15);ax.set_ylim(0,1)
axes[0].set_ylabel('Assessment-background acceptance');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'figures/acceptance_profiles.pdf');plt.close(fig)
paths=list((OUT/'tables').glob('*.tex'))+list((OUT/'figures').glob('*.pdf'))+[OUT/'auc_per_seed.csv']
(OUT/'generated_assets_sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
print('Generated',len(paths),'paper assets from verified artifacts')

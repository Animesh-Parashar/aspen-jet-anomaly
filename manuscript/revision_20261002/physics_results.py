"""Regenerate operating-point report and figure from verified saved results."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/results/physics_operating_points_26331.rachel.json'
r=json.loads(P.read_text())
audit=json.loads((ROOT/'data/results/physics_operating_points_26331.audit.json').read_text())
assert audit['status']=='passed'
fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
colors={'aspen':'#0072B2','lhco':'#D55E00','random':'#009E73'}
markers={'raw':'o','mahalanobis':'s','white_knn':'^'}
for row,target in enumerate([.1,.05]):
 for col,t in enumerate(['two','three']):
  ax=axes[row,col]
  for v in r['rows'].values():
   if v['target_background_acceptance']!=target:continue
   n=v['score'];x=v['mass_ks'];y=v['signals'][t]['efficiency']
   # Each null interval uses this arm's selected background count. Its y value
   # is the expected signal acceptance of an independent random selection.
   null=v['count_matched_random_selection']['ks_range']['percentile_95']
   ax.hlines(v['background_acceptance'],*null,color='gray',alpha=.20,lw=2,zorder=1)
   if '_seed' in n:
    d=n.split('_seed')[0];sc=n.split('_',2)[2];ax.scatter(x,y,c=colors[d],marker=markers[sc],s=14,alpha=.25)
   else:
    c='black' if n=='girth_high' else '#CC79A7'
    ci=v['refitted_threshold_bootstrap']
    ax.hlines(y,*ci['mass_ks']['percentile_95'],color=c,lw=1)
    ax.vlines(x,*ci[t+'_signal_efficiency']['percentile_95'],color=c,lw=1)
    ax.scatter(x,y,c=c,marker='*' if n=='girth_high' else 'D',s=100,zorder=5)
  for d in colors:
   for sc in markers:
    v=r['seed_summary'][f'{d}_{sc}_{target}'];x=v['mass_ks']['mean'];y=v[t+'_signal_efficiency']['mean']
    ax.hlines(y,*v['mass_ks']['event_interval']['percentile_95'],color=colors[d],lw=1)
    ax.vlines(x,*v[t+'_signal_efficiency']['event_interval']['percentile_95'],color=colors[d],lw=1)
    ax.scatter(x,y,c=colors[d],marker=markers[sc],s=45,zorder=4)
  ax.set_title(f'{t.capitalize()}-prong; nominal background acceptance {target:.0%}')
  ax.grid(alpha=.15);ax.set_xlim(0,1);ax.set_ylim(0,.5)
  if col==0:ax.set_ylabel('Signal efficiency')
  if row==1:ax.set_xlabel('Background jet-mass KS distance (lower is less distortion)')
handles=[Line2D([],[],marker='o',color=c,ls='',label=d) for d,c in [('Aspen',colors['aspen']),('Simulation',colors['lhco']),('Random backbone',colors['random'])]]
handles += [Line2D([],[],marker=m,color='gray',ls='',label=s) for s,m in [('Raw kNN','o'),('Mahalanobis','s'),('Whitened kNN','^')]]
handles += [Line2D([],[],marker='*',color='black',ls='',label='Girth'),Line2D([],[],marker='D',color='#CC79A7',ls='',label='Observable density')]
handles += [Line2D([],[],color='gray',lw=3,label='Count-matched random-selection reference')]
fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=8)
fig.suptitle('Signal retention versus background jet-mass distortion',fontsize=14)
fig.tight_layout(rect=(0,.13,1,.95))
out=Path(__file__).parent/'figures';fig.savefig(out/'physics_tradeoff.pdf');fig.savefig(out/'physics_tradeoff.png',dpi=160);plt.close(fig)
lines=['# Physics operating-point results — 3 October 2026','','Job 26331.rachel completed. All 58 arm/working-point cases passed independent threshold/count/KS/JSD checks and bootstrap-interval recomputation. Maximum independent KS/JSD discrepancy: %.3g.'%audit['independent_ks_js_max_error'],'','## All-arm results','','Efficiencies and background acceptance are percentages. Model entries show mean ± sample SD across three seeds; controls are fixed. KS and JSD compare selected versus inclusive assessment-background jet mass. These are original unweighted populations, not the common-pT diagnostic.','','| Target BG | Method | Actual BG (%) | Two-prong efficiency (%) | Three-prong efficiency (%) | Mass KS | Mass JSD (nats) |','|---|---|---:|---:|---:|---:|---:|']
for target in [.1,.05]:
 for d in ['aspen','lhco','random']:
  for s in ['raw','mahalanobis','white_knn']:
   v=r['seed_summary'][f'{d}_{s}_{target}'];vals=[]
   for m in r['metric_order']:
    scale=100 if 'efficiency' in m or m=='background_acceptance' else 1
    vals.append(f"{scale*v[m]['mean']:.3f} ± {scale*v[m]['sample_sd']:.3f}")
   lines.append(f'| {target:.0%} | {d} {s} | '+' | '.join(vals)+' |')
 for n in ['girth_high','observable_density']:
  v=next(v for v in r['rows'].values() if v['score']==n and v['target_background_acceptance']==target)
  vals=[100*v['background_acceptance'],100*v['signals']['two']['efficiency'],100*v['signals']['three']['efficiency'],v['mass_ks'],v['mass_js_nats']]
  lines.append(f'| {target:.0%} | {n} | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
lines+=['','## Interpretation','','- Girth and observable density retain much more signal, but their selections strongly reshape the background jet-mass distribution. At nominal 5%, girth has KS 0.893 and density KS 0.586, compared with raw Aspen seed-mean KS 0.135.','- Raw Aspen retains only 3.17% of two-prong and 0.92% of three-prong signals at actual background acceptance 5.53%. Reduced distortion is accompanied by weak signal retention.','- For every Aspen scorer and both working points, mean signal efficiency on each topology is lower than its mean background acceptance. A mass-independent random selection has equal signal/background acceptance in expectation. Thus reduced sculpting alone does not establish a useful anomaly tagger; do not interpret this expectation as a separately run signal-efficiency experiment.','- At nominal 5%, raw simulation retains 9.75%/9.68% signal with KS 0.318, but training-seed variability is substantial. This is a measured retention/distortion trade-off, not a demonstrated best search strategy.','- Finite-count random-selection reference bands show the scale of spurious shape differences. For girth at 5%, the matched-count KS band is approximately [0.029,0.093], far below the observed 0.893. These are descriptive bands, not global p-values.','- Nominal background acceptance is calibrated separately; actual assessment acceptance differs by method. All comparisons must show these rates rather than claim identical achieved background rejection.','- The mass is the full leading anti-kt R=0.8 jet mass. It is not dijet resonance mass. These data do not establish a fake bump, sideband-fit bias, discovery significance or experimental search validity.','- This extension was motivated after prior result inspection. Bootstrap intervals refit thresholds but condition on model/reference pool, bins and split, and are pointwise. No model was retrained or selected.','','## Figure','','[Signal-efficiency/mass-distortion figure](../manuscript/revision_20261002/figures/physics_tradeoff.png)','','Each model marker is one seed. Marker shape identifies scorer; color identifies training domain/random control. Control points have no seed spread. This overview displays individual point estimates, not uncertainty bands; full conditional/refitted-threshold intervals are in the result JSON. Nominal acceptance is shown in panel titles, while achieved rates appear in the table.','','## Evidence','','- `data/results/physics_operating_points_26331.rachel.json`: full thresholds, selected counts, uncertainty, binned mass/pT acceptance, random-selection bands, paired comparisons.','- NPZ companion: exact identities, histogram edges, all 1,000 paired bootstrap draws.','- `data/results/physics_operating_points_26331.audit.json`: independent post-run checks.','- `docs/physics_operating_points_protocol_20261003.md`: frozen protocol.','','Contribution: operational signal-retention versus jet-mass-distortion characterization in this fixed transfer setup. It adds physics context to the AUC/scorer study without establishing a new algorithm or a practical real-data advantage.','']
lines = [('Large markers show seed means; faint markers retain every seed. '
          'Horizontal/vertical segments show marginal 95% threshold-refitting event-bootstrap intervals, '
          'not joint regions or training uncertainty. Gray count-matched KS reference segments are placed '
          'at each arm\'s achieved background acceptance (the expected signal acceptance of independent '
          'random selection). They are not measured random-cut signal efficiencies. '
          'Shape identifies scorer and color identifies domain/control.')
         if line.startswith('Each model marker is one seed.') else line for line in lines]
(ROOT/'docs/physics_operating_points_results.md').write_text('\n'.join(lines))
paths=[P,P.with_suffix('.npz'),ROOT/'data/results/physics_operating_points_26331.audit.json',ROOT/'docs/physics_operating_points_protocol_20261003.md',Path(__file__),out/'physics_tradeoff.pdf']
(Path(__file__).parent/'physics_evidence.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
print('Report, figure and provenance regenerated')

"""Render post-hoc transfer diagnostics from immutable saved summaries."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE/'submission'
paths = [ROOT/f'data/results/transfer_{mode}_20261003.json' for mode in ('support', 'representations')]
support, representations = [json.loads(p.read_text()) for p in paths]
check_path = ROOT/'data/results/transfer_diagnostics_crosscheck_20261003.json'
check = json.loads(check_path.read_text())
assert check['status'] == 'passed'
for p in paths:
    assert check['diagnostic_hashes'][str(p.relative_to(ROOT))] == hashlib.sha256(p.read_bytes()).hexdigest()
for p, r in zip(paths, (support, representations)):
    assert r['stage'] == 'post_hoc_transfer_diagnostics'
    assert r['script_sha256'] == hashlib.sha256((ROOT/'scripts/validation/transfer_diagnostics.py').read_bytes()).hexdigest()
    assert r['protocol_sha256'] == hashlib.sha256((ROOT/'docs/transfer_diagnostics_protocol_20261003.md').read_bytes()).hexdigest()
    shutil.copyfile(p, OUT/p.name)
colors = {'aspen': '#0072B2', 'lhco': '#D55E00', 'random': '#009E73'}
labels = {'aspen': 'Aspen', 'lhco': 'Simulation', 'random': 'Random'}

fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for ax, key, label in zip(axes.flat, ['pt', 'mass', 'multiplicity', 'log_pt'],
        ['Available-constituent jet pT [GeV]', 'Available-constituent jet mass [GeV]',
         'Available constituent multiplicity', 'Retained constituent ln(pT / GeV)']):
    for name, color, style, title in [('aspen_train', colors['aspen'], '-', 'Aspen training (100k)'),
            ('lhco_train', colors['lhco'], '-', 'Simulation training (100k)'),
            ('lhco_reference', 'black', '--', 'Simulation references (10k)')]:
        h = support['samples'][name]['histograms'][key]
        edges, probability = np.array(h['edges']), np.array(h['probability'])
        # Density is per displayed coordinate; pT uses log-spaced x bins.
        ax.stairs(probability/np.diff(edges), edges, color=color, ls=style, label=title)
    ax.set_xlabel(label); ax.set_ylabel('Probability density'); ax.grid(alpha=.15)
    if key == 'pt': ax.set_xscale('log'); ax.set_yscale('log')
    elif key in ('mass', 'multiplicity'): ax.set_yscale('log')
axes[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'figures/training_support.pdf'); plt.close(fig)

names = [f'{d}_seed{s}' for d in colors for s in (17, 29, 43)]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), gridspec_kw={'width_ratios':[1, 1.2]})
rho = np.array([[representations['models'][n]['scorers']['raw']['background_spearman'][k]
                 for k in ('mass','pt','multiplicity','girth')] for n in names])
im = axes[0].imshow(rho, vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')
axes[0].set_yticks(range(9), [labels[n.split('_')[0]]+' '+n.split('seed')[1] for n in names])
axes[0].set_xticks(range(4), ['Mass','pT','Multiplicity','Girth'], rotation=20)
axes[0].set_title('Raw-kNN background Spearman correlation')
for i in range(9):
    for j in range(4): axes[0].text(j, i, f'{rho[i,j]:.2f}', ha='center', va='center', fontsize=8)
fig.colorbar(im, ax=axes[0], fraction=.046, pad=.04)
for d in colors:
    for seed, style in zip((17,29,43), ('-','--',':')):
        g = representations['models'][f'{d}_seed{seed}']['geometry']['reference']
        a = np.array(g['eigenvalues_descending'])
        axes[1].plot(np.arange(1,129), a/a.sum(), color=colors[d], ls=style,
                     label=labels[d] if seed == 17 else None, lw=1)
axes[1].set_yscale('log'); axes[1].set_ylim(1e-8, 1)
axes[1].set_xlabel('Covariance eigenvalue index'); axes[1].set_ylabel('Fraction of variance')
axes[1].set_title('Background-reference spectra'); axes[1].legend(fontsize=8); axes[1].grid(alpha=.15)
fig.tight_layout(); fig.savefig(OUT/'figures/representation_diagnostics.pdf'); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for t, color in [('two', '#CC79A7'), ('three', '#009E73')]:
    h = support['signal_mass'][t]
    axes[0].stairs(np.array(h['probability'])/np.diff(h['edges']), h['edges'], label=t.capitalize()+'-prong', color=color)
axes[0].set_xlabel('Leading-jet mass [GeV]'); axes[0].set_ylabel('Probability density'); axes[0].legend(fontsize=8)
for ax, d in zip(axes[1:], ('aspen','lhco')):
    for seed, style in zip((17,29,43), ('-','--',':')):
        h = support['samples'][d+'_train']['histories'][str(seed)]
        ax.plot([v['epoch'] for v in h], [v['validation_loss'] for v in h], ls=style, label=f'Seed {seed}')
    ax.set_title(labels[d]); ax.set_xlabel('Epoch'); ax.set_ylabel('Fixed-view validation loss'); ax.set_yscale('log'); ax.legend(fontsize=8)
for ax in axes: ax.grid(alpha=.15)
fig.tight_layout(); fig.savefig(OUT/'figures/signal_mass_losses.pdf'); plt.close(fig)

rows = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
        r'Backbone & Seed & $\rho_m$ & $\rho_{p_T}$ & $\rho_n$ & $\rho_g$ & $r_{\mathrm{eff}}$ \\', r'\midrule']
for name in names:
    m = representations['models'][name]; r = m['scorers']['raw']['background_spearman']
    rows.append(' & '.join([labels[name.split('_')[0]], name.split('seed')[1]]+
        [f'{r[k]:.3f}' for k in ('mass','pt','multiplicity','girth')]+
        [f"{m['geometry']['reference']['entropy_rank']:.2f}"])+r' \\')
rows += [r'\bottomrule', r'\end{tabular}']
(OUT/'tables/diagnostics.tex').write_text('\n'.join(rows)+'\n')

counts = json.loads((ROOT/'data/results/physics_operating_points_26331.rachel.json').read_text())
rows = [r'\begin{tabular}{lrr}', r'\toprule', r'Raw-score backbone & Selected background counts (seeds 17, 29, 43) & Total \\', r'\midrule']
for d in colors:
    values = [next(v['assessment_background_selected'] for v in counts['rows'].values()
                  if v['score'] == f'{d}_seed{s}_raw' and v['target_background_acceptance'] == .05) for s in (17,29,43)]
    rows.append(labels[d]+' & '+', '.join(str(v) for v in values)+r' & 5000 \\')
rows += [r'\bottomrule', r'\end{tabular}']
(OUT/'tables/selected_counts.tex').write_text('\n'.join(rows)+'\n')
evidence = paths+[check_path, ROOT/'scripts/validation/transfer_diagnostics_crosscheck.py', ROOT/'scripts/validation/transfer_diagnostics.py', ROOT/'docs/transfer_diagnostics_protocol_20261003.md',
                  ROOT/'tests/test_transfer_diagnostics.py', Path(__file__)]
(HERE/'diagnostic_evidence.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence},indent=2)+'\n')
print('Diagnostic figures, tables and evidence generated')

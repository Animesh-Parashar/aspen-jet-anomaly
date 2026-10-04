"""Render frozen signal kinematics, checking provenance and normalization."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE/'submission'
source = ROOT/'data/results/signal_kinematics_20261003.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
data = json.loads(source.read_text())
assert data['stage'] == 'post_evaluation_descriptive_signal_kinematics'
paths = [source, Path(__file__), ROOT/'tests/test_signal_kinematics.py']
for key, name in [('script_sha256', 'scripts/validation/signal_kinematics.py'),
                  ('protocol_sha256', 'docs/signal_kinematics_protocol_20261003.md')]:
    assert sha(ROOT/name) == data[key], name
    paths.append(ROOT/name)
for name, expected in data['source_sha256'].items():
    assert sha(ROOT/name) == expected, name
    paths.append(ROOT/name)
assert data['checks']['shared_background']
assert data['checks']['weighted_histograms_and_quantiles'] == 'passed'
assert data['checks']['independent_auc_max_error'] < 1e-12

fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for row, population in enumerate(('original', 'common_pt')):
    for col, observable in enumerate(('pt', 'multiplicity')):
        ax = axes[row, col]
        for topology, color, style, label in [('two', '#CC79A7', '-', 'Two-prong'),
                                              ('three', '#009E73', '--', 'Three-prong')]:
            for kind in ('background', 'signal'):
                h = data['results'][topology][population][observable][kind]
                edges, prob = np.array(h['edges']), np.array(h['probability'])
                np.testing.assert_allclose(prob.sum()+h['underflow']+h['overflow'], 1, atol=1e-12)
                if population == 'original' and kind == 'background' and topology == 'three':
                    continue
                bglabel = 'Background' if population == 'original' else label+' background'
                ax.stairs(prob/np.diff(edges), edges,
                          color=color if kind == 'signal' else ('black' if topology == 'two' else '#777777'),
                          ls=style, lw=1.4, label=label+' signal' if kind == 'signal' else bglabel)
        ax.set_xlim((600, 2600) if observable == 'pt' else (0, 150))
        ax.set_xlabel('Available-constituent jet pT [GeV]' if observable == 'pt' else 'Available constituent multiplicity')
        ax.set_ylabel('Probability density')
        ax.set_title('Original population' if population == 'original' else 'Common-pT weighted population')
        ax.grid(alpha=.15)
        if col == 0: ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/'figures/signal_kinematics.pdf')
plt.close(fig)
shutil.copyfile(source, OUT/source.name)
(HERE/'signal_kinematics_evidence.json').write_text(json.dumps(
    {str(p.relative_to(ROOT)): sha(p) for p in paths}, indent=2)+'\n')
print('Signal kinematics figure and evidence generated')

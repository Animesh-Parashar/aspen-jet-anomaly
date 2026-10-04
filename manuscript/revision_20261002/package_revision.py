"""Build local source and numerical-reproduction archives; never upload."""
from pathlib import Path
import hashlib,json,zipfile,re
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent;SRC=HERE/'submission'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source_files=[SRC/'main.tex',SRC/'references.tex']+sorted((SRC/'tables').glob('*.tex'))+sorted((SRC/'figures').glob('*.pdf'))
if not (SRC/'main.pdf').exists():raise ValueError('Compile PDF before packaging')
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', (SRC/'main.tex').read_text(), re.S).group(1)
abstract = ' '.join(abstract.split()).replace(r'\%', '%').replace('--', '-')
assert abstract.isascii() and len(abstract) <= 1920
(HERE/'abstract.txt').write_text(abstract+'\n')
with zipfile.ZipFile(HERE/'arxiv_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in source_files:z.write(p,str(p.relative_to(SRC)))
 for p in [SRC/'auc_per_seed.csv',SRC/'scorer_robustness_26312.rachel.json',SRC/'physics_operating_points_26331.rachel.json',SRC/'transfer_support_20261003.json',SRC/'transfer_representations_20261003.json',SRC/'signal_kinematics_20261003.json']:
  z.write(p,'anc/'+p.name)
files=set()
files.add(ROOT/'scripts/hpc/signal_kinematics.pbs')
for name in ['physics_results.py','diagnostic_assets.py','signal_kinematics_assets.py','build_assets.py','verify_paper_results.py','package_revision.py','REPRODUCIBILITY.md','abstract.txt','final_evidence.json','scorer_evidence.json','physics_evidence.json','replication_evidence.json','diagnostic_evidence.json','signal_kinematics_evidence.json']:
 files.add(HERE/name)
files.update(source_files);files.add(SRC/'main.pdf')
# Include every artifact directly identified by the paper asset evidence manifests.
for name in ['final_evidence.json','scorer_evidence.json','physics_evidence.json','replication_evidence.json','diagnostic_evidence.json','signal_kinematics_evidence.json']:
 for p,h in json.loads((HERE/name).read_text()).items():
  f=ROOT/p
  if sha(f)!=h:raise ValueError('Changed upstream evidence: '+p)
  files.add(f)
for name in ['data/results/scientific_value_gate_26199.rachel.json','data/results/scientific_value_gate_26199.rachel.npz','data/results/revision_replication_eval_26300.rachel.json','data/results/revision_replication_eval_26300.rachel.npz','data/results/physics_operating_points_lock_20261003.json','data/results/scorer_robustness_lock_20261003.json','data/results/raw_validation/sample_audit.json','requirements-preprocess.txt']:
 files.add(ROOT/name)
for pattern in ['configs/revision_seed*.json','configs/controlled_convergence.json','scripts/preprocess/*.py','scripts/validation/*.py','scripts/hpc/transfer*.pbs','tests/test_*comparison*.py','tests/test_*operating*.py','tests/test_background_metric.py','tests/test_preprocessing_v2.py','tests/test_scorer_robustness.py','tests/test_three_prong_v2.py','tests/test_frozen_topology_eval.py','tests/test_transfer_diagnostics.py','tests/test_representation_diagnostics.py']:
 files.update(ROOT.glob(pattern))
for name in ['src/data/corrected_preprocessing.py','src/data/controlled_loader.py','src/models/controlled_encoder.py','src/models/encoder.py','src/augmentations/controlled_augmentations.py','src/augmentations/physics_ablation.py','scripts/train/train_controlled.py','scripts/train/continue_controlled.py']:
 files.add(ROOT/name)
for pattern in ['docs/*protocol*.md','docs/*replication*.md','docs/*scorer*.md','docs/*operating*.md','docs/*transfer*.md','docs/frozen_evaluation_results.md']:
 files.update(ROOT.glob(pattern))
files.add(ROOT/'docs/experiment_ledger.md')
# Preserve exact snapshot paths; scripts importing their siblings remain functional.
manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(files) if p.is_file()}
(HERE/'reproduction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(HERE/'reproducibility.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(files):
  if p.is_file():z.write(p,str(p.relative_to(ROOT)))
 z.write(HERE/'reproduction_manifest.json','reproduction_manifest.json')
print('Created local arxiv_source.zip and reproducibility.zip;',len(manifest),'reproduction files')

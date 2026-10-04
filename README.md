# Contrastive Pretraining on Real LHC Jets

Code and saved numerical results for **Contrastive Pretraining on Real LHC Jets:
Transfer, Scorer Dependence and Mass Distortion**. The [manuscript PDF](manuscript/revision_20261002/submission/main.pdf)
and [LaTeX source](manuscript/revision_20261002/submission/main.tex) are included.
The manuscript is prepared as a preprint; no arXiv identifier has been assigned.

## What the study finds

We compare contrastive encoders trained on 100,000 real AspenOpenJets or simulated
LHC Olympics jets, using three matched seeds, random backbones and simple jet
observables as controls. The real and simulated training samples have a large
momentum mismatch: only 24 selected Aspen training jets lie in the simulated
reference sample's central 90% momentum range. The comparison therefore does not
identify a causal effect of real versus simulated pretraining.

On the momentum-balanced two-prong queries, raw nearest-neighbor mean AUCs are
0.541 (Aspen), 0.564 (simulation) and 0.517 (random). Three-prong values are
0.419, 0.553 and 0.426. High jet momentum alone gives AUC 0.833/0.835 in the
original two-/three-prong populations; after the specified balancing, its AUC
is about 0.494/0.495. The paper also measures scorer sensitivity, signal
retention and background mass distortion. These are results for the specified
pipeline and public benchmarks, not a validated experimental search.

## Reproduce the reported results

The repository includes the saved score arrays and numerical summaries needed
to verify tables and regenerate figures. Use Python 3.11 in an isolated
environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-reproduce.txt
python -B manuscript/revision_20261002/verify_paper_results.py
python -B manuscript/revision_20261002/physics_results.py
python -B manuscript/revision_20261002/diagnostic_assets.py
python -B manuscript/revision_20261002/signal_kinematics_assets.py
python -B manuscript/revision_20261002/build_assets.py
```

Run these commands from the repository root. The manuscript can be compiled
from `manuscript/revision_20261002/submission/main.tex` with a standard LaTeX
installation. [Detailed reproduction notes](manuscript/revision_20261002/REPRODUCIBILITY.md)
explain the recorded inputs and numerical checks.

The raw Aspen and LHCO datasets, processed HDF5 arrays, trained checkpoints and
full embedding caches are not included. Regenerating the model scores requires
those inputs and a training rerun; the saved scores support exact numerical
verification of the reported tables and figures. The research scripts contain
run-specific provenance guards and HPC settings, which must be adapted for a
new run without relabeling it as the original experiment.

## Repository contents

- `manuscript/revision_20261002/`: PDF, LaTeX, figures, tables and asset builders.
- `data/results/`: saved scores and numerical summaries used in the paper.
- `src/`, `scripts/`, `configs/`: preprocessing, model, training and evaluation code.
- `tests/`: focused physics and numerical checks.
- `docs/`: protocols and result summaries needed to interpret the reported study.

The [AspenOpenJets](https://arxiv.org/abs/2412.10504) and
[LHC Olympics R&D](https://doi.org/10.5281/zenodo.6466204) source datasets are
available from their providers. The source code is MIT licensed; see [LICENSE](LICENSE).

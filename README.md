# Contrastive Pretraining on Real LHC Jets

Code and numerical results accompanying **Contrastive Pretraining on Real LHC Jets: Transfer, Scorer Dependence and Mass Distortion**, by Animesh Parashar and Aditya Parashar.

[Paper (Zenodo)](https://doi.org/10.5281/zenodo.20827791) · [Reproduction guide](REPRODUCIBILITY.md)

The study compares representations pretrained on real AspenOpenJets and simulated LHC Olympics background. It evaluates two- and three-prong anomaly-ranking benchmarks, alternative scorers, and signal efficiency versus background jet-mass distortion. The training samples have very little transverse-momentum overlap, so the comparison does not isolate a causal effect of the pretraining domain. See the paper for full results and limitations.

## Verify the reported numbers

The `results/` directory contains the saved score arrays, numerical summaries, and diagnostic supplements used for the paper. From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-reproduce.txt
python -B verify_results.py
```

The verifier recomputes AUCs from saved scores, checks three-seed summaries, and checks signal efficiencies and bootstrap intervals at the reported operating points. It requires no raw events or GPU. The [reproduction guide](REPRODUCIBILITY.md) explains the scope of these checks and how to compile the manuscript.

## Repository layout

- `results/`: saved score arrays, quantitative summaries, and the per-seed AUC table (`auc_per_seed.csv`).
- `src/`: model, augmentation, and data-processing implementations.
- `scripts/train/` and `configs/`: training implementation and published hyperparameters.
- `verify_results.py`: independent numerical checks from saved results.

The [AspenOpenJets](https://arxiv.org/abs/2412.10504) and [LHC Olympics R&D](https://doi.org/10.5281/zenodo.6466204) datasets are distributed by their providers. Raw events, processed arrays, checkpoints, and embedding caches are not included here. Source code is available under the [MIT License](LICENSE).

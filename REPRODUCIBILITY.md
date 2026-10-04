# Reproducing the reported results

This release supports direct verification of the paper's numerical results from saved scores. A full rerun from raw events additionally needs the external datasets, preprocessing, trained checkpoints, and embedding caches; those large inputs are not part of this repository.

## Numerical verification

Install `requirements-reproduce.txt` in a Python 3.11 environment and run `python -B verify_results.py` from the repository root. The script recomputes all 176 reported AUC entries from the stored score arrays, checks the three-seed means and sample standard deviations, and checks operating-point efficiency counts and 290 stored bootstrap intervals. It uses NumPy and scikit-learn and does not retrain or select a model.

The `results/` JSON files retain quantitative fields needed to interpret the NPZ arrays, plus the training-support, representation, and signal-kinematics diagnostics cited in the paper. The NPZ files are unchanged copies of the saved numerical arrays; only their public filenames were simplified. Machine-specific paths and run metadata were removed from the public JSON summaries without changing measured quantities. The paper's `paper/auc_per_seed.csv` exposes the per-seed scorer results in a small, readable format.

## Manuscript source

Compile `paper/main.tex` from within `paper/` using two runs of `pdflatex -interaction=nonstopmode -halt-on-error main.tex`. The included `paper/main.pdf` is the released preprint. No external bibliography tool or shell escape is required.

## Training code and scope

`src/` contains the model, augmentation, loading, and corrected preprocessing implementations. `scripts/train/` contains the fixed-epoch training and checkpoint-continuation implementations. The three files in `configs/` publish the numerical hyperparameters used for the final 100-epoch encoders. Their `data/aspen` and `data/lhco` entries are portable placeholders for processed input directories; they are not the original machine paths. Seed 17 reached 100 epochs through checkpoint continuation, while seeds 29 and 43 used fresh 100-epoch runs.

The training code expects processed event arrays with the schema used by `src/data/controlled_loader.py`. This repository does not provide a turnkey raw-data-to-paper rerun. To reproduce scores independently, obtain the source datasets, implement the paper's event processing and partitioning, retain the specified event identities and seeds, and record new provenance for the rerun. Do not interpret a fresh run's scores as the saved scores supplied here.

The paper's final partitions were withheld from development of this corrected pipeline, but the public benchmarks had been inspected in earlier work. Later scorer and physics studies are descriptive extensions. The reported event-bootstrap intervals condition on the trained models and chosen reference samples; variation across training seeds is reported separately.

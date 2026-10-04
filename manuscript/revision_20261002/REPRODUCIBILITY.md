# Reproduction package

The canonical typeset study is `submission/main.tex`, compiled to
`submission/main.pdf`. No arXiv identifier has been assigned yet.

## Reproduce tables/figures from saved evidence

From the root of the project or extracted reproducibility archive:

```bash
python -B manuscript/revision_20261002/verify_paper_results.py
python -B manuscript/revision_20261002/physics_results.py
python -B manuscript/revision_20261002/diagnostic_assets.py
python -B manuscript/revision_20261002/signal_kinematics_assets.py
python -B manuscript/revision_20261002/build_assets.py
```

These commands read saved scores/results and perform no new training or final
model selection. Requires NumPy, SciPy, scikit-learn and Matplotlib; Python 3.11
was used. `physics_results.py` updates the derived physics evidence manifest.
`build_assets.py` first checks its upstream evidence manifests. Numerical values
are deterministic; PDF metadata may change when figures are regenerated.
`diagnostic_assets.py` verifies the independent numerical checks and renders saved
training-support histograms, correlation/spectrum diagnostics and loss histories.
Its source JSON records exact sampling provenance and full upstream hashes.
`signal_kinematics_assets.py` verifies frozen inputs and renders signal/background
momentum and multiplicity distributions for original and common-momentum samples.

## Compile source

In `manuscript/revision_20261002/submission`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively `tectonic --keep-logs main.tex` resolves references automatically.
Only standard packages are used. There is no shell-escape requirement or external
bibliography process: verified references are in `references.tex`. The local
build uses verified official Tectonic 0.17.0; its standalone archive SHA256 is
8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7.

## Full computation, beyond saved-score reproduction

The source archive includes the exact preprocessing, model, augmentation,
training and analysis implementations, JSON configurations, PBS wrappers, tests,
protocols and result manifests. It does not include raw datasets, preprocessed
HDF5 tensors, encoder checkpoints or full embedding caches. Thus it reproduces
the reported numerical tables/figures from scores directly, but does not alone
suffice to regenerate scores from raw data without those inputs.

For a full rerun obtain the Aspen RunG_batch0 source and official LHCO files,
verify published checksums, then follow the versioned preprocessing drivers and
protocols. The initial data-inspection manifest identifies exposed events assigned
into validation. Background and signal partitions must stay separate. Train the
fixed configurations for seeds 17/29/43; seed17 uses `controlled_convergence.json`
with the documented extension to 100 epochs. Existing encoder continuation retains
optimizer and RNG states. The fresh seed29/43 configs already specify 100 epochs.

The archive retains exact original scripts, including run-ID paths and strict
source/provenance checks. These are an execution record, not a turnkey installer:
paths must be explicitly mapped to a new local run and new provenance recorded.
Do not bypass guards or relabel new runs as the original. Portable table/figure
reproduction above needs no such mapping. Do not launch HPC wrappers without
adapting resource directives to the local scheduler.

Primary final scores: `frozen_topology_eval.py`; later robustness extension:
`scorer_robustness.py`; physics operating points: `physics_operating_points.py`.
The transfer post-hoc diagnostics use `transfer_diagnostics.py` in `support`
and `representations` modes. Recomputing them requires the exact processed HDF5
partitions and all nine cached embedding files; these large inputs are not in
the portable archive. `transfer_diagnostics_crosscheck.py` independently checks observable
histograms/quantiles and all 108 score correlations. These CPU analyses use
scheduler wrappers; figure reproduction from their saved summaries does not.
No new hyperparameter or score-direction choices are part of reproduction.
The fixed-point tests in `tests/` cover physics representation, metric fitting,
ties, sample separation, covariance identities and bootstrap duplicates.

Observed environment: Python 3.11; NumPy 1.26.4; SciPy 1.13.1; scikit-learn 1.9.0;
PyTorch 2.4.1+cu121; h5py 3.11.0; FastJet 3.5.1.5; Matplotlib 3.10.8.
Preprocessing dependencies are also recorded in `requirements-preprocess.txt`.
Cross-platform neural training need not be bitwise identical; retain source and
checkpoint hashes for exact score reproduction.

## Scope and release status

The final partitions were withheld from corrected development, not historically
unseen public data. Scorer, physics and transfer diagnostics were added after final inspection.
Event intervals are conditional and pointwise; seed SD is separate. No evidence
supports a new discovery, a universal real-data advantage or an optimized search.
This repository contains the revised source and saved score artifacts.
`abstract.txt` contains ASCII submission metadata derived from the manuscript
(under 1920 characters). Author names and affiliations should be verified by
the authors before submission. Contact addresses are omitted from the public
copy. Machine-specific paths in published metadata are normalized to relative
paths; numerical arrays and reported measurements are unchanged.

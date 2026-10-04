# Frozen transfer and final evaluation — 3 October 2026

This protocol is written before these evaluations. No changes to models, training,
score signs, k, features or endpoints are allowed based on their results.

## Exposure and scope

Two-prong test/reference partitions have been withheld from corrected-pipeline
model development and scoring. Integrity checks during preprocessing are not
performance evaluation. The same public dataset appears in the historical
preprint, so these are not claimed to be historically untouched data.

The historical three-prong evaluator loads the full signal sample. Its exact
past event exposure is not recoverable from that code alone. New qqq validation
and test partitions are disjoint, but neither is claimed to be historically
unseen. Three-prong results are frozen-model transfer evidence, not a new
independent discovery dataset. No signal labels enter encoder training.

## Fixed inputs and deterministic sample selection

Existing two-prong root: data/processed_v2/full_24632.rachel/lhco.
New three-prong root: data/processed_v2/three_prong_20261003.
Official qqq checksum: 54e123a86143b668f9cb76905152a124. Use the corrected
anti-kt R=0.8 leading vector-pT jet, full-axis/full-scalar-pT normalization and
hardest 50 constituents. Signal-only source namespace LHCO_RD_qqq_v1; negative
one-minus-source-row event IDs distinguish it from existing nonnegative IDs.
Event hash seed 20260925, 50/50 signal validation/test. No outcome-based exclusions.
Structural/physics preprocessing rejects are counted and disclosed.

- Development three-prong: exactly the original 10,000 background reference and
  10,000 background query events in scientific_value_gate_26199.rachel.npz.
  Select 2,000 qqq validation rows with sorted, without-replacement NumPy
  default_rng(20261001) choices from the verified partition.
- Final two-prong: select 10,000 background/reference rows, 10,000 background/test
  rows and 2,000 signal/test rows. Each selection uses sorted, without-replacement
  default_rng(20261004) choices from its verified partition independently.
- Final three-prong: reuse exactly that final background reference/query sample;
  select 2,000 qqq signal/test rows using default_rng(20261004) likewise.

If a partition has fewer required rows, abort and disclose; do not shrink it or
substitute another partition silently. Save selected event identities and labels.
Check namespace-qualified uniqueness/disjointness; recompute event-hash membership
for selected final events and exclude historical preprocessing-audit overrides.

## Models, scores and controls

All six frozen 100-epoch models (seeds 17,29,43 in Aspen and LHCO), with three
matched random initial backbones. Paths and provenance as in
scripts/validation/revision_replication_eval.py. No further training.
Raw 128D backbone Euclidean tenth-neighbor distance, high score anomalous.
Reference fitting is background only. Controls: high mass, high pT, high and
low multiplicity (both reported), high girth, and tenth-neighbor density in
reference-standardized [mass,pT,multiplicity,girth]. Zero reference variance
aborts. No chosen-best control is advertised as a prospective algorithm.

## Metrics and inference limits

For each population report per-seed original and common-pT AUC, mean and sample
SD, and paired Aspen-minus-LHCO differences. Common-pT uses 20 reference quantile
bins, reference-range support, at least 30 queries per class per retained bin,
and equal class mass min(n_bg,n_sig) in each supported bin. Report bin counts,
excluded events and effective sample size; no supported bins means abort.
This label-dependent reweighting is a diagnostic population, not a deployable
selection. It does not eliminate within-bin or other confounding.

Empirical working points at signal efficiencies .3 and .5 use all ties and
report achieved signal/background efficiencies. They are descriptive, not
independently calibrated cuts. Zero observed background passes implies null
rejection, not infinite rejection or a discovery claim.

Use 1,000 common class-stratified event bootstrap resamples, seed 20261003,
recompute pT weights, hold reference events/edges/models fixed. Report pointwise
percentile intervals for mean individual-model AUCs and their paired differences,
not pooled-score ensembles. These exclude training and reference-pool uncertainty;
three seed SDs remain separate. No confirmatory p-value or multiplicity-adjusted
family claim is planned. Report every topology/population/seed regardless of sign.

## Execution integrity and stopping

Reuse audited scorer/statistics; verify fixed model configs, source hashes,
initialization pairing and sample provenance. Direct float64 kth-distance spot
checks and independent rank-AUC checks must pass. Save source/protocol/checkpoint
hashes, event IDs and all score arrays. Freeze a launch manifest of these files
before submitting performance evaluation. Failures stop analysis; bug corrections
require a recorded protocol/code version and explanation before rerunning.
No further tuning on final results. Update manuscript with separate development
and held-out tables, limitations and the full historical-exposure disclosure.

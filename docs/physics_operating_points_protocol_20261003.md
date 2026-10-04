# Physics operating-point analysis — 3 October 2026

Post-evaluation diagnostic specified after inspecting AUC/scorer results. It is
not independent confirmation, a new tagger, a calibrated discovery test or a
claim that background mass is intrinsically decorrelated.

## Frozen sample and methods

Use the cached final two-/three-prong scores from scorer_robustness_26312.rachel
and physical observables from frozen_final_{two,three}_26302.rachel. Verify exact
event identities, labels and shared background scores. All 27 model/scorer arms
(3 seeds × Aspen/simulation/random × raw-kNN/Mahalanobis/white-kNN), plus girth
and four-observable density, are mandatory. No score reversal or best-arm choice.
The scorer reference-fit events remain fixed and disjoint from all queries.

The 10,000 shared background queries are permuted once by NumPy default_rng
seed 20261006: first 5,000 calibrate thresholds; remaining 5,000 assess acceptance
and shape. All 2,000 signals per topology assess efficiency. Save exact identities.
No signal event, label, mass or performance enters threshold calibration.
Use the original unweighted populations; do not apply signal-dependent common-pT
weights. Two signals share one background sample and are not independent tests.

For target background acceptance a in {0.10,0.05}, threshold is the ceil(a*N)-th
largest calibration score. Select score >= threshold, retaining all ties. Report
achieved calibration and assessment acceptance, not a fictitious exact rate.
No random tie breaking. If a tied control cannot realize the target, disclose it.
The nominal working points may differ in realized assessment acceptance; compare
signal efficiency alongside that acceptance, never silently equate the two.

## Physics observables and shape diagnostics

Jet mass and vector pT are those of the full constituent-sum leading anti-kt
R=0.8 jet before the hardest-50 truncation. They are not dijet invariant mass.
Girth uses retained constituents, as in the frozen benchmark. No re-clustering,
new feature choice, event weighting or fitting to signal shapes.

Compare background mass distribution before versus after selection using:
1. Empirical CDF maximum absolute difference (KS distance). No independent-sample
   KS p-value: the selected sample is a subset of the inclusive assessment set.
2. Jensen-Shannon divergence in natural-log units between normalized inclusive
   and selected mass histograms, using 10 calibration-background quantile bins.
   Remove duplicate internal edges and extend outer edges to +/-infinity so no
   assessment event is dropped. No pseudocounts: 0 log 0 is zero. JSD is bounded
   by ln(2). Empty selected samples yield null shape metrics, not zero distortion.
3. Background acceptance in those mass bins, and separately 10 analogous pT bins,
   with selected/total counts and Wilson 95% intervals conditional on the fitted
   threshold. These are coarse marginal diagnostics, not mass dependence at fixed
   pT or causal attribution. Also retain normalized selected/inclusive mass ratios.

Finite selected samples produce nonzero apparent distortion even for mass-neutral
selection. Calibrate that noise floor descriptively: for each arm/working point,
500 uniform without-replacement subsets of the assessment background with the
same observed selected count (seed 20261008 reset per case). Report median and
95% range of KS/JSD under this count-matched random selection. These are a
reference noise band, not discovery/multiplicity-adjusted p-values or a baseline
tagger. Exact identical subset sizes share the same simulated reference band.

## Uncertainty and reporting

At fixed thresholds report assessment background and signal efficiencies with
Wilson 95% intervals (conditional on calibration). Primary uncertainty additionally
uses 1,000 paired bootstrap replicates, seed 20261007: resample calibration
background, assessment background and each signal class independently, each to
its original size. Refit all thresholds in every replicate; reuse each resample
across all arms/working points and both topologies' shared background. Recompute
KS/JSD and efficiencies on the resampled assessment set, preserving the nested
selected/inclusive relationship and score ties. Histogram edges, encoders,
reference metric fits and event split remain fixed.

Report per-seed values, mean and sample SD across 3 seeds; paired differences to
same-scorer random controls and girth for signal efficiency, realized background
acceptance and mass distortion. Event percentile intervals are pointwise and
conditional on these models/reference pool/split. They include threshold sampling
variation but exclude training/reference fitting variability. Preserve all draws.
No new discovery significance, extrapolated tail rejection, global Pareto
optimality, general mass-decorrelation or dijet bump-hunt claim.

Output: machine-readable counts, thresholds, shape metrics, bins, intervals,
identities, seed summaries, paired comparisons, figure/table source and hashes.
Present efficiency and distortion together, with uncertainty and realized
acceptance. Null or uncompetitive outcomes are reported. This ends the bounded
physics extension; no automatic tuning or further model/scorer search follows.

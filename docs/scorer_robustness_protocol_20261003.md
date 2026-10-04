# Scorer robustness extension — 3 October 2026

This analysis was proposed after inspecting the frozen final results. It is a
post-evaluation robustness extension, not another independent final test or a
new-method claim. No new encoder training or signal-supervised score fitting.

Reuse exactly both final query samples and the shared background reference pool
from frozen_final_two_26302.rachel and frozen_final_three_26302.rachel. All six
100-epoch encoders (Aspen/LHCO, seeds 17/29/43) and three corresponding initial
backbones remain fixed. Extract and cache their raw 128D embeddings once. Verify
sample identities, observables, model/source hashes and initialization pairing.

Three higher-is-anomalous scores for every backbone:
1. Raw Euclidean tenth-neighbor distance (historical frozen score, reproduced).
2. Squared regularized Mahalanobis distance to the reference mean.
3. Tenth-neighbor distance after regularized covariance whitening.

Fit only on the 10,000 background reference embeddings. Use maximum-likelihood
C=(R-mu)^T(R-mu)/N and Creg=.9*C+.1*trace(C)/128*I, exactly the earlier single-seed
metric screen. Cholesky Creg=L L^T; transform rows by (x-mu)L^-T. No search over
shrinkage, k, normalization, coordinate subsets, checkpoint, seed or score sign.
Whitened empirical covariance need not equal I. Mahalanobis distance is not a
calibrated tail probability and requires no claim of Gaussian background.

Report all 27 model/scorer combinations per topology and retain girth and
four-observable density controls. Report original and common-pT AUC, all seed
values and mean/sample SD, with unchanged reference-defined pT bins/support and
empirical working points at signal efficiencies .3 and .5. The support exclusions
and effective sample sizes remain explicit. No best-score headline selection.

For each topology/population report seed-paired differences: each scorer versus
raw for the same backbone; Aspen versus LHCO for each scorer; trained versus
same-seed random for the corresponding scorer; each learned/scorer arm versus
girth and observable density. Differences of seed means are means of per-seed
AUC differences, not ensembles of uncalibrated scores. Cross-topology comparisons
are descriptive; no independent-topology sampling assumption.

Use 1,000 shared class-stratified event bootstrap resamples per topology, seed
20261005, recomputing common-pT weights on fixed edges. Percentile intervals are
pointwise, conditional on the fitted models/reference pool. They exclude training
and reference-fit uncertainty and are not multiplicity-adjusted confirmatory
claims. No pass/fail novelty thresholds, selective reporting, extra scorer search
or automated model training follows these results.

Validate raw-cache agreement (rtol 1e-4, atol 1e-5 and AUC change <=2e-5), direct
float64 kth distances, quadratic form via independent linear solve, covariance
whitening identity, analytic examples, rank deficiency, orthogonal invariance,
reference-only fit, and tie/duplicate-aware weighted AUC. Abort on failed checks.
Save embeddings, fitted means/covariance/factors, all scores and bootstrap draws,
source/protocol hashes and machine-readable paired summaries.

Resources: one GPU/one CPU extraction job, then one CPU scoring job. No need to
reserve a GPU for covariance fitting or bootstrap. This finite extension ends
with a complete result table and bounded interpretation, regardless of outcome.

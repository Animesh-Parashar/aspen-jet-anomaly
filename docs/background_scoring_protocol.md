# Background-only frozen scoring screen — 2026-10-01

Written before computing the new scores. This is adaptive development following
negative raw-kNN results and supervised probes, not a final confirmation study.
No signal labels or supervised probe parameters enter any fitted transformation.
Final test/reference partitions are not opened.

Use exactly the original gate's 10,000 background validation neighbor events and
12,000 disjoint validation query events. Reuse the three frozen raw 128D query
representations from the probe cache; extract matching reference representations
for Aspen contrastive epoch 100, LHCO contrastive epoch 100, and the shared random
backbone. Verify identities and checkpoint/source/cache hashes. Use eval mode.

For each representation fit only on the background neighbor pool:

- mu = mean(reference embeddings).
- C = (R-mu)^T (R-mu)/N (maximum-likelihood covariance).
- C_reg = 0.9 C + 0.1 trace(C)/d I, d=128.
- Cholesky C_reg = L L^T; whiten row vectors by z=(x-mu)L^(-T).

The shrinkage coefficient 0.1 is a fixed regularization choice, not a fitted
optimum. It makes the metric positive definite when total reference variance is
positive. Reject nonfinite inputs, zero total variance, or failed factorization.
No coordinate-wise standardization, component truncation or parameter search.
Whitening makes C_reg map to I; it does not make the empirical covariance exactly
I. Verify that distinction numerically. Original latent coordinates are not
physical momentum components and this transformation asserts no physical symmetry.

Two new higher-is-anomalous scores per backbone:
1. Distance to the 10th background neighbor in regularized whitened coordinates.
2. Squared regularized Mahalanobis distance to the background mean, ||z||^2.

The second score measures ellipsoidal distance; it is not a calibrated p-value
or an assertion that the background is Gaussian. The first changes the metric
of the existing kNN score while fixing k and event identities. Fit and neighbor
pools are the same background-only set; query events are disjoint.

Primary screen: the two Aspen scores versus (a) original Aspen raw kNN,
(b) the corresponding new score on random features, (c) girth, and (d) original
four-observable density. Eight comparisons in total. Report all six new scores,
including LHCO as a contextual transfer control. Reuse reference-defined 20-bin
common-pT evaluation weights and 10,000 paired class-stratified query-event
bootstrap replicates, seed 1801, recomputing weights per replicate. Original AUC
and working points at signal efficiencies 0.3/0.5 remain mandatory secondary
results. No score reversal or choice of best signal checkpoint.

Advance only if an Aspen arm has positive point differences and positive lower
Bonferroni two-sided 99.375% percentile bounds against all four comparators.
This is a development triage rule; the approximate bootstrap bounds condition
on the reference pool, transformation and encoder seed, and do not account for
adaptive reuse, training variance or reference fitting uncertainty. No automatic
publication claim or encoder-training campaign follows a pass. A failure stops
this fixed scoring screen without adding shrinkage values or k choices.

Required checks: analytic diagonal covariance example; Cholesky metric versus
independent linear solve; regularized covariance whitening identity; orthogonal
coordinate-change invariance; constant/singular input behavior; reference-only
fit unaffected by query changes; direct cdist kNN check on actual embeddings;
recomputed raw scores agree with cached controls; existing tie-aware weighted AUC
and bootstrap duplicate-event tests. Save fitted means/factors, scores and hashes.

Resources: one brief one-GPU reference extraction, then a one-core CPU scoring
and bootstrap job. No new encoder training or signal-supervised fitting.

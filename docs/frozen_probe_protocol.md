# Frozen representation diagnostic — fixed before fitting, 2026-10-01

Question: can a linear readout recover signal discrimination or physical
observables from frozen representations whose fixed kNN anomaly score failed?
This is supervised, signal-guided development. It is not an unsupervised result,
a novel method claim, or a final evaluation. No encoder training is permitted.

## Fixed sample and representations

Reuse the 12,000 query events from gate 26199 (10,000 background, 2,000 signal).
Keep its 10,000 background neighbor events out of all probe fits. Use only the
neighbor pT values to define the existing 20 quantile bins. Verify exact query
identities, observables and labels, disjoint neighbor identities, and historical
checkpoint/source hashes. The final test/reference files are not opened.

Extract raw 128-dimensional backbones at the existing fixed endpoints: Aspen
contrastive 100 epochs, LHCO contrastive 100 epochs, and their shared initial
random backbone. The two failed augmentation arms are not refitted or selected.
Controls for signal probing are girth alone and the four fixed observables
(mass, pT, full multiplicity, retained girth). Record all outcomes.

## Signal-information probe

Five shared, class-stratified, event-disjoint folds, shuffled with seed 20261002.
Enforce one query row per event; duplicates are an error, not separate fold units.
For each fold, compute the existing common-pT weights using only training labels
and training pT: reference-defined bins, at least 30 events of each class per bin,
class mass equal to min(background count, signal count). Zero unsupported rows.
Normalize positive weights to mean one. Fit feature mean/variance on training
rows using those weights only; no validation-fitted transformations. Fit an L2
logistic regression with intercept, C=1, lbfgs, tolerance 1e-8, max_iter=2000.
Treat convergence warnings as failures. No parameter grid or feature selection.

Objective: sum_i w_i [log(1+exp(t_i)) - y_i t_i] + ||beta||^2/(2 C),
t_i = beta dot standardized(x_i) + intercept; intercept is unpenalized.
Check the numerical first-order residual independently with a stable sigmoid.
Predictions on a held-out fold come only from its training-fold fit.

Report each fold's original and common-pT AUC, their descriptive arithmetic mean,
and pooled out-of-fold AUC as a secondary quantity. Evaluation common-pT weights
are computed on the full evaluation query population, then restricted per fold;
these use labels for metrics only and never enter training or standardization.
Use precisely the same fold metrics for the existing kNN scores and fixed girth
and observable-density scores. Also report the weight mass excluded from each
training fit and effective sample sizes. The five folds share training events:
the fold spread is not an independent-seed error estimate. Cross-fold prediction
calibration can affect pooled AUC, so it is not the sole conclusion.

## Observable-information probe

On background queries only, use the same outer fold assignments. Fit one
multi-output ridge regression per representation: alpha=1, intercept, SVD solver.
Feature and target standardization use training-background rows only. Predict
physical mass, pT, full multiplicity and retained girth in original units.
Report out-of-fold RMSE and R-squared for each observable, plus predictive skill
1 - SSE_model / SSE_training-fold-mean. Negative R-squared is retained. Full
mass/multiplicity include information beyond the 50 retained constituents and
therefore are not guaranteed to be recoverable from these inputs.
Independently check ridge normal equations in standardized coordinates.

## Interpretation and limits

Strong held-out linear discrimination despite poor kNN performance supports a
mismatch between information accessible to a linear readout and the fixed kNN
score on this development sample. It does not prove unsupervised recoverability.
Strong random-feature performance would weaken a claim of benefit from learned
features. Poor linear probes do not prove information is absent: nonlinear
readouts, regularization and domain shift remain possible explanations. No
post-hoc sign reversal of the unsupervised scores is allowed.

This is a bounded diagnostic: 25 small logistic fits (five inputs times five
folds), 15 multi-output ridge fits (three backbones times five folds), one GPU
feature-extraction job, and one one-core CPU job. No encoder training, replication
campaign, significance test, or confidence interval is triggered automatically.
No bootstrap with fixed fitted predictions will be presented as training uncertainty.

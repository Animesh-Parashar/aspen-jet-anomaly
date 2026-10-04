# Scorer robustness results — 3 October 2026

Jobs 26311.rachel and 26312.rachel completed. This extension was specified after final-score inspection and is descriptive robustness evidence, not independent confirmation. It uses no new encoder training and no scorer hyperparameter search.

## Full mean-AUC comparison

Values are mean ± sample SD across three training/initialization seeds. SD is not a confidence interval. Both populations and all scorers are reported.

| Backbone | Scorer | Two-prong original | Two-prong common-pT | Three-prong original | Three-prong common-pT |
|---|---|---:|---:|---:|---:|
| Aspen | Raw kNN | 0.5114 ± 0.0626 | 0.5407 ± 0.0391 | 0.3968 ± 0.0534 | 0.4188 ± 0.0317 |
| Aspen | Mahalanobis | 0.4664 ± 0.0353 | 0.5182 ± 0.0097 | 0.3678 ± 0.0330 | 0.4162 ± 0.0137 |
| Aspen | Whitened kNN | 0.5078 ± 0.0532 | 0.5426 ± 0.0225 | 0.3843 ± 0.0527 | 0.4101 ± 0.0233 |
| Simulation | Raw kNN | 0.5504 ± 0.2170 | 0.5639 ± 0.0672 | 0.5446 ± 0.2219 | 0.5528 ± 0.0848 |
| Simulation | Mahalanobis | 0.5555 ± 0.1886 | 0.5591 ± 0.0706 | 0.5518 ± 0.2017 | 0.5596 ± 0.0908 |
| Simulation | Whitened kNN | 0.5653 ± 0.1962 | 0.5849 ± 0.0711 | 0.5538 ± 0.2054 | 0.5698 ± 0.0904 |
| Random | Raw kNN | 0.6361 ± 0.0164 | 0.5172 ± 0.0174 | 0.5580 ± 0.0168 | 0.4255 ± 0.0160 |
| Random | Mahalanobis | 0.6366 ± 0.0086 | 0.5463 ± 0.0106 | 0.5598 ± 0.0070 | 0.4623 ± 0.0123 |
| Random | Whitened kNN | 0.6296 ± 0.0141 | 0.5642 ± 0.0132 | 0.5420 ± 0.0094 | 0.4733 ± 0.0100 |
| Physics control | Girth | 0.6858 | 0.7131 | 0.6838 | 0.7118 |
| Physics control | Observable density | 0.8095 | 0.6452 | 0.7691 | 0.6060 |

## Interpretation

1. **Failure is not restricted to raw Euclidean kNN.** Neither fixed covariance-based alternative makes Aspen competitive with girth on these signals. This bounds three tested scorers, not every possible readout.
2. **Metric improvements need matched random controls.** Whitened kNN changes common-pT mean AUC by +0.0019 (two-prong) and −0.0087 (three-prong) for Aspen, versus +0.0470 and +0.0478 for random backbones. Such gains do not by themselves demonstrate learned anomaly information.
3. **A learned-versus-random conclusion depends on scorer choice.** On two-prong common-pT queries Aspen exceeds random by +0.0235 with raw kNN, but trails it by −0.0217 with whitened kNN and −0.0281 with Mahalanobis. Report these as descriptive comparisons, not a claim of a universal ranking.
4. **Simulation improves modestly under whitening, with substantial seed variation.** Common-pT means rise from 0.5639 to 0.5849 (two-prong), and 0.5528 to 0.5698 (three-prong). Girth remains higher than every trained seed/scorer in each common-pT population.
5. **Three-prong Aspen ranking remains inverted.** All three fixed scorers on all three Aspen seeds have common-pT AUC below 0.5; score signs were not reversed after viewing signals.
6. **Contribution remains empirical.** The extension supports a domain × initialization × scorer comparison with random and physics controls; covariance scoring and the general need for such controls are established ideas, not algorithmic novelty.

## Scope and numerical validation

- Same final reference/query identities and pT support as the preceding evaluation. Over half the background queries are excluded from common-pT support; original AUC remains essential.
- Covariance fitting uses background references only, with fixed shrinkage 0.1; no signal labels enter the metric fit.
- 1,000 shared event resamples per topology give pointwise intervals conditional on the fitted models/reference pool, not training uncertainty or multiplicity-adjusted confirmation. Three-seed SDs remain separate.
- All frozen hashes, 58 cached score AUCs (54 model/scorer/topology cases plus four physics-control cases), 36 seed summaries and their bootstrap intervals passed post-run checks.
- Maximum raw-score reproduction error 6.51e-6; direct whitened-distance error 1.07e-14; independent quadratic-form error 1.28e-13; regularized-covariance whitening error 1.97e-14.
- Full per-seed scores, empirical working points and paired differences are in `data/results/scorer_robustness_26312.rachel.json`; fitted metrics, score arrays and bootstrap draws are in its NPZ companion.

This completes the authorized scorer extension. No further scoring or model searches are part of this revision.

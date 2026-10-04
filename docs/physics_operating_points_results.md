# Physics operating-point results — 3 October 2026

Job 26331.rachel completed. All 58 arm/working-point cases passed independent threshold/count/KS/JSD checks and bootstrap-interval recomputation. Maximum independent KS/JSD discrepancy: 6.75e-14.

## All-arm results

Efficiencies and background acceptance are percentages. Model entries show mean ± sample SD across three seeds; controls are fixed. KS and JSD compare selected versus inclusive assessment-background jet mass. These are original unweighted populations, not the common-pT diagnostic.

| Target BG | Method | Actual BG (%) | Two-prong efficiency (%) | Three-prong efficiency (%) | Mass KS | Mass JSD (nats) |
|---|---|---:|---:|---:|---:|---:|
| 10% | aspen raw | 10.247 ± 0.121 | 7.167 ± 2.816 | 2.450 ± 1.344 | 0.129 ± 0.077 | 0.015 ± 0.013 |
| 10% | aspen mahalanobis | 10.093 ± 0.551 | 6.167 ± 1.533 | 2.067 ± 1.020 | 0.135 ± 0.040 | 0.018 ± 0.007 |
| 10% | aspen white_knn | 9.973 ± 0.050 | 6.150 ± 1.621 | 1.867 ± 0.909 | 0.127 ± 0.073 | 0.017 ± 0.013 |
| 10% | lhco raw | 9.920 ± 0.352 | 17.683 ± 13.365 | 18.000 ± 13.141 | 0.283 ± 0.058 | 0.055 ± 0.020 |
| 10% | lhco mahalanobis | 10.053 ± 0.281 | 14.633 ± 9.540 | 15.450 ± 11.616 | 0.257 ± 0.030 | 0.048 ± 0.015 |
| 10% | lhco white_knn | 9.707 ± 0.167 | 16.850 ± 11.394 | 16.300 ± 12.229 | 0.263 ± 0.039 | 0.050 ± 0.015 |
| 10% | random raw | 10.167 ± 0.031 | 16.567 ± 1.492 | 7.950 ± 1.276 | 0.251 ± 0.026 | 0.045 ± 0.009 |
| 10% | random mahalanobis | 10.393 ± 0.160 | 16.450 ± 1.117 | 7.333 ± 1.397 | 0.293 ± 0.037 | 0.069 ± 0.015 |
| 10% | random white_knn | 10.353 ± 0.196 | 22.200 ± 2.693 | 11.733 ± 2.675 | 0.327 ± 0.040 | 0.076 ± 0.016 |
| 10% | girth_high | 10.260 | 35.750 | 38.550 | 0.804 | 0.428 |
| 10% | observable_density | 10.300 | 42.300 | 40.800 | 0.564 | 0.216 |
| 5% | aspen raw | 5.533 ± 0.150 | 3.167 ± 1.329 | 0.917 ± 0.473 | 0.135 ± 0.082 | 0.018 ± 0.017 |
| 5% | aspen mahalanobis | 5.560 ± 0.223 | 3.100 ± 0.577 | 0.733 ± 0.388 | 0.158 ± 0.063 | 0.023 ± 0.013 |
| 5% | aspen white_knn | 5.507 ± 0.064 | 2.867 ± 0.679 | 0.600 ± 0.304 | 0.145 ± 0.063 | 0.019 ± 0.013 |
| 5% | lhco raw | 4.947 ± 0.201 | 9.750 ± 9.148 | 9.683 ± 8.590 | 0.318 ± 0.062 | 0.068 ± 0.022 |
| 5% | lhco mahalanobis | 4.880 ± 0.347 | 6.417 ± 4.060 | 7.017 ± 5.312 | 0.279 ± 0.043 | 0.058 ± 0.020 |
| 5% | lhco white_knn | 4.993 ± 0.081 | 8.567 ± 6.698 | 8.133 ± 6.860 | 0.283 ± 0.053 | 0.058 ± 0.025 |
| 5% | random raw | 5.267 ± 0.081 | 8.083 ± 0.451 | 3.283 ± 0.076 | 0.271 ± 0.016 | 0.055 ± 0.009 |
| 5% | random mahalanobis | 5.507 ± 0.076 | 8.233 ± 0.551 | 2.950 ± 0.229 | 0.348 ± 0.067 | 0.090 ± 0.027 |
| 5% | random white_knn | 5.693 ± 0.150 | 11.633 ± 1.286 | 4.367 ± 0.679 | 0.370 ± 0.038 | 0.090 ± 0.016 |
| 5% | girth_high | 4.820 | 24.100 | 22.050 | 0.893 | 0.504 |
| 5% | observable_density | 4.740 | 31.850 | 30.300 | 0.586 | 0.244 |

## Interpretation

- Girth and observable density retain much more signal, but their selections strongly reshape the background jet-mass distribution. At nominal 5%, girth has KS 0.893 and density KS 0.586, compared with raw Aspen seed-mean KS 0.135.
- Raw Aspen retains only 3.17% of two-prong and 0.92% of three-prong signals at actual background acceptance 5.53%. Reduced distortion is accompanied by weak signal retention.
- For every Aspen scorer and both working points, mean signal efficiency on each topology is lower than its mean background acceptance. A mass-independent random selection has equal signal/background acceptance in expectation. Thus reduced sculpting alone does not establish a useful anomaly tagger; do not interpret this expectation as a separately run signal-efficiency experiment.
- At nominal 5%, raw simulation retains 9.75%/9.68% signal with KS 0.318, but training-seed variability is substantial. This is a measured retention/distortion trade-off, not a demonstrated best search strategy.
- Finite-count random-selection reference bands show the scale of spurious shape differences. For girth at 5%, the matched-count KS band is approximately [0.029,0.093], far below the observed 0.893. These are descriptive bands, not global p-values.
- Nominal background acceptance is calibrated separately; actual assessment acceptance differs by method. All comparisons must show these rates rather than claim identical achieved background rejection.
- The mass is the full leading anti-kt R=0.8 jet mass. It is not dijet resonance mass. These data do not establish a fake bump, sideband-fit bias, discovery significance or experimental search validity.
- This extension was motivated after prior result inspection. Bootstrap intervals refit thresholds but condition on model/reference pool, bins and split, and are pointwise. No model was retrained or selected.

## Figure

[Signal-efficiency/mass-distortion figure](../manuscript/revision_20261002/figures/physics_tradeoff.png)

Large markers show seed means; faint markers retain every seed. Horizontal/vertical segments show marginal 95% threshold-refitting event-bootstrap intervals, not joint regions or training uncertainty. Gray count-matched KS reference segments are placed at each arm's achieved background acceptance (the expected signal acceptance of independent random selection). They are not measured random-cut signal efficiencies. Shape identifies scorer and color identifies domain/control.

## Evidence

- `data/results/physics_operating_points_26331.rachel.json`: full thresholds, selected counts, uncertainty, binned mass/pT acceptance, random-selection bands, paired comparisons.
- NPZ companion: exact identities, histogram edges, all 1,000 paired bootstrap draws.
- `data/results/physics_operating_points_26331.audit.json`: independent post-run checks.
- `docs/physics_operating_points_protocol_20261003.md`: frozen protocol.

Contribution: operational signal-retention versus jet-mass-distortion characterization in this fixed transfer setup. It adds physics context to the AUC/scorer study without establishing a new algorithm or a practical real-data advantage.

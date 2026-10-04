# Transfer diagnostics — 3 October 2026

Jobs 26339.rachel (training support), 26340.rachel (cached representations) and
26341.rachel (independent checks) completed successfully. Each used one CPU,
4 GB requested memory and no GPU. These were post-hoc checks, with no retraining,
new score selection or modification of previous numerical results.

## Hypotheses, results and interpretation

| Question | Measured result | Interpretation |
|---|---|---|
| Do the actual selected training populations occupy comparable momentum regimes? | Aspen training median pT 336.84 GeV; simulation training 1310.94 GeV; fixed simulation references 1307.89 GeV. Only 24/100000 Aspen versus 89592/100000 simulation training jets lie in the reference central 90% range [1198.57,1740.95] GeV. | A substantial kinematic mismatch is established. Evaluation class-balancing did not remove this training mismatch. Its causal contribution to the AUC deficit remains unmeasured. |
| Is the reference population concentrated inside Aspen's main training range? | Every reference jet is outside the Aspen training central 98% pT range [257.47,706.87] GeV; only 1.92% are outside simulation's corresponding interval. | Central quantile ranges describe concentration, not strict mathematical support. Do not say there is zero overlap: rare Aspen training jets reach the evaluation range. |
| Are raw scores associated with simple jet properties? | Aspen score–multiplicity Spearman correlations are −0.532, −0.578, −0.557. Simulation score–pT correlations are −0.114, −0.436, +0.289 for seeds 17/29/43. | Associations support sensitivity to jet properties and seed-dependent geometry; they do not demonstrate a causal explanation of signal AUCs. All 108 domain/seed/scorer/observable correlations are reported. |
| Does training simply collapse the representation relative to initialization? | Reference entropy effective ranks: Aspen 7.54/6.11/7.65; simulation 17.10/15.25/16.01; random 3.20/3.99/3.33. | Both trained domains have higher effective rank than matched random backbones. Low effective rank alone neither diagnoses pathological collapse nor explains whitening gains. |
| Does the phi-chart rejection create different training augmentations? | Maximum local radius 1.192 Aspen and 1.153 simulation, both < pi. | On these selected training inputs, no rotation angle can trigger rejection. The approximate planar rotation still need not preserve exact physical jet mass. |
| Is the stored Aspen cap heavily saturated? | 3/100000 selected training jets have 150 nonzero stored constituents. | This is slot saturation (0.003%), not proof of how many particles were discarded before storage. |
| Are resumed/restarted training endpoints visibly different in loss? | All six epoch-100 fixed-view validation losses lie between 0.02079 and 0.02143. All 100 epochs are plotted. | Similar losses provide an endpoint check, not proof of useful or equally good representations. |

The constituent log-pT plots use 1/n_retained particle weights so each jet
contributes equally. Particle-weighted alternatives and overflow probabilities
are saved. Jet quantities use unweighted available constituents, not the producer's
corrected/PUPPI jet momenta. Leading signal mass distributions describe the
already-scored jets; no truth X/Y assignment is inferred.

## Validation

- Eight focused unit tests passed: histogram tails/endpoints, equal-jet weights,
  padding exclusion, tied Spearman ranks/constants, rotation bounds and spectrum
  rank invariants.
- Exact training selection provenance agrees across all three seeds per domain.
- Independent full-column HDF5 reads reproduce selected identities, jet histogram
  counts/quantiles and reference-range counts for all three populations.
- All 108 correlations agree with SciPy Spearman calculations to numerical
  precision. Covariance spectra agree with direct SVD, and raw kNN spot checks
  agree with direct distances; no cached score was changed.
- Detailed numerical checks: `data/results/transfer_diagnostics_crosscheck_20261003.json`.

## Next scientific experiment, not yet launched

The most direct bounded follow-up is a matched explicit-scale-feature ablation
in both training domains, retaining the same samples, updates, seeds and random
controls. A four-slot input with the absolute-log-pT slot fixed to zero could
preserve architecture and initialization pairing; implementation and endpoints
must be specified before execution. This tests dependence on the explicit scale
channel, not a causal realness effect or removal of every scale correlation.

Training 100000 pT-matched Aspen jets is not justified by the 24 overlapping jets
in the present selected sample. It requires a separate feasibility count over
the available training pool or additional data. One million unique simulation
training jets also exceed the current fixed training partition of 597261 jets.
Do not duplicate jets and present them as one million independent training examples.

No additional training, public code release or arXiv submission was performed.

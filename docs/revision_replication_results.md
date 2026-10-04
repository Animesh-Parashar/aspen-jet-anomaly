# Three-seed two-prong development results — 3 October 2026

Job `26300.rachel` completed. All nine score computations, identity/provenance checks, historical seed-17 reproduction, independent rank AUC and direct distance checks passed. Saved score-cache AUCs and paired differences were independently recomputed after completion. Maximum direct-distance discrepancy: 3.38e-07.

## Fixed comparison

Values are mean ± sample SD across three training/initialization seeds; these are not confidence intervals. Physics baselines have no encoder seeds.

| Score | Original AUC | Common-pT AUC |
|---|---:|---:|
| Aspen contrastive | 0.4955 ± 0.0624 | 0.5217 ± 0.0388 |
| Simulation contrastive | 0.5484 ± 0.2195 | 0.5562 ± 0.0737 |
| Random backbone | 0.6246 ± 0.0172 | 0.5139 ± 0.0162 |
| Girth | 0.6894 | 0.7340 |
| Four-observable density | 0.8027 | 0.6517 |

## Seed-by-seed common-pT AUC

| Seed | Aspen | Simulation | Random | Aspen − simulation |
|---|---:|---:|---:|---:|
| 17 | 0.4904 | 0.5824 | 0.5041 | -0.0919 |
| 29 | 0.5651 | 0.4729 | 0.5326 | 0.0921 |
| 43 | 0.5096 | 0.6132 | 0.5051 | -0.1037 |

## Interpretation

- Real-data pretraining has no demonstrated stable advantage in this fixed configuration. The domain difference changes sign across training seeds; neither stable simulation superiority nor equivalence is established.
- Common-pT Aspen mean AUC is only 0.0078 above the random-control mean. This point estimate alone is not a robust learned advantage.
- Girth outperforms every trained seed after common-pT balancing; observable density also exceeds every trained seed in that population.
- The original simulation AUC ranges from 0.3058 to 0.7333. Training variability is scientifically material; no score sign was flipped and no seed was discarded.
- Narrow event-bootstrap intervals condition on the fitted models and reference pool. They must not be used to claim that a domain ranking generalizes over training seeds.
- These are adaptively developed two-prong results. Corrected three-prong transfer and the locked final evaluation remain outstanding.

## Conditional event intervals

These exclude training and reference-pool uncertainty. They describe mean individual-model AUC, not an ensemble of averaged scores.

| Population | Mean paired Aspen − simulation | Seed SD | Conditional event-bootstrap 95% interval |
|---|---:|---:|---|
| original_auc | -0.0529 | 0.2749 | [-0.0656, -0.0389] |
| common_pt_auc | -0.0345 | 0.1098 | [-0.0541, -0.0146] |

Full per-seed empirical working points at signal efficiencies 0.3 and 0.5 are in the result JSON. Thresholds are descriptive, not independently calibrated selections.

Source: `data/results/revision_replication_eval_26300.rachel.json`. Protocol: `docs/revision_replication_evaluation.md`.

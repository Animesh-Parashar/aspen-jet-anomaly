# Frozen transfer and final evaluation results — 3 October 2026

Jobs 26301.rachel and 26302.rachel completed. All 100,000 official qqq events were processed with no rejected rows: 49,945 validation and 50,055 test. Preprocessing verification found no duplicate source rows or cross-partition event overlap. All frozen launch/source/protocol hashes match. Saved score-cache AUCs were recomputed successfully; final topologies share identical background references and queries, and selected qqq development/final signal IDs are disjoint.

Final means withheld from corrected-pipeline development, not historically unseen public data. See the historical exposure disclosure in the frozen protocol.

## Final AUCs

Means ± sample SD across seeds 17,29,43; SD is not a confidence interval. Controls have no training seeds.

| Score | Two-prong original | Two-prong common-pT | Three-prong original | Three-prong common-pT |
|---|---:|---:|---:|---:|
| Aspen contrastive | 0.5114 ± 0.0626 | 0.5407 ± 0.0391 | 0.3968 ± 0.0534 | 0.4188 ± 0.0317 |
| Simulation contrastive | 0.5504 ± 0.2170 | 0.5639 ± 0.0672 | 0.5446 ± 0.2219 | 0.5528 ± 0.0848 |
| Random backbone | 0.6361 ± 0.0164 | 0.5172 ± 0.0174 | 0.5580 ± 0.0168 | 0.4255 ± 0.0160 |
| Girth | 0.6858 | 0.7131 | 0.6838 | 0.7118 |
| Four-observable density | 0.8095 | 0.6452 | 0.7691 | 0.6060 |

## All-seed common-pT comparisons

| Population | Seed | Aspen | Simulation | Random | Aspen − simulation |
|---|---:|---:|---:|---:|---:|
| development_three | 17 | 0.3853 | 0.5932 | 0.4119 | -0.2080 |
| development_three | 29 | 0.4349 | 0.4514 | 0.4394 | -0.0165 |
| development_three | 43 | 0.3820 | 0.6089 | 0.4291 | -0.2269 |
| final_two | 17 | 0.5092 | 0.5881 | 0.5090 | -0.0789 |
| final_two | 29 | 0.5844 | 0.4879 | 0.5372 | 0.0965 |
| final_two | 43 | 0.5284 | 0.6156 | 0.5054 | -0.0872 |
| final_three | 17 | 0.4005 | 0.5874 | 0.4092 | -0.1869 |
| final_three | 29 | 0.4554 | 0.4561 | 0.4413 | -0.0007 |
| final_three | 43 | 0.4005 | 0.6147 | 0.4260 | -0.2142 |

## Support and uncertainty

| Population | Excluded background / 10000 | Excluded signal / 2000 | Background effective N | Signal effective N |
|---|---:|---:|---:|---:|
| development_three | 4877 | 147 | 2618.8 | 1825.0 |
| final_two | 5468 | 186 | 2328.9 | 1789.5 |
| final_three | 5468 | 180 | 2371.2 | 1774.5 |

Common-pT changes the target population substantially: final comparisons retain 4,532 background queries, 1,814 two-prong signals or 1,820 three-prong signals. Coarse pT bins do not eliminate within-bin or other confounding. Original AUCs must be reported alongside balanced AUCs.

The 1,000-resample event-bootstrap intervals in the JSON condition on the fixed fitted models, bin edges and reference pool. They exclude training/reference uncertainty and do not justify a stable domain-ranking claim across future training runs. Working points at signal efficiencies .3/.5 are empirical, not independently calibrated selections.

## Scientific interpretation

- No demonstrated robust real-data advantage in this fixed configuration. Two-prong domain differences change sign by seed; large simulation seed variability persists.
- Three-prong Aspen common-pT AUC is below 0.5 in all three seeds. With the prespecified high-distance-anomalous direction, this means inverted ranking for this signal; no post-hoc score flip is made. It is not a universal impossibility result for real-data pretraining.
- Girth exceeds every trained common-pT score on both final topologies. Learned scores are not competitive with that control here.
- Three-prong performance cannot be inferred universally from two-prong performance. Observed degradation is strongest for Aspen and random backbones in this comparison.
- The revised paper should present a bounded transfer/reproducibility study, not a new high-performance anomaly detector or experimental discovery.
- Planned replication, corrected topology transfer and final scoring are complete. Manuscript methods, full references, reproducible figures/tables and checked submission source/PDF remain to be finished.

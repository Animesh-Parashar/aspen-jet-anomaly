# Frozen replication evaluation — 3 October 2026

Extends the fixed two-prong development comparison to training seeds 17, 29, 43.
No final partitions are accessed. Inputs reproduce the historical cache identities,
labels and observables exactly: 10,000 background neighbors, 10,000 background
queries, 2,000 signal queries. All six 100-epoch models and three paired initial
backbones use raw 128D embeddings and Euclidean tenth-neighbor distance.
No threshold, sign, checkpoint, scorer or seed selection is performed.

Report every seed's original and common-pT AUC and empirical working points at
signal efficiencies 0.3 and 0.5. The common-pT population retains the established
20 reference quantile bins and at least 30 events of each class per supported
bin, with equal class mass per bin. Weighted working points include all score
ties and report achieved efficiencies. Zero observed background efficiency has
null rejection, not an infinite-sensitivity claim.

Aggregate individual AUCs with mean and sample SD, including seed-paired Aspen
minus LHCO differences. Do not average uncalibrated model scores. Use 1,000 shared
class-stratified event bootstraps, seed 20261003, recomputing balancing weights
with fixed bin edges. Percentile intervals describe conditional uncertainty of
mean-seed AUCs and their paired difference; they exclude training/reference-pool
uncertainty. Three seeds are not three independent datasets.

Checks: completion/provenance/source hashes, unchanged configuration except seed,
matching initial backbone within each pair, fixed data provenance across seeds,
finite histories/weights/embeddings, unique disjoint reference/query events,
independent rank AUC and direct float64 kth-distance spot checks. Recompute seed17
and compare with the existing cache (score rtol 1e-4/atol 1e-5, AUC tolerance
2e-5 for GPU float32 differences). Any failure stops reporting, not relaxed
thresholds chosen after viewing outcomes. The original trainers remain unchanged.

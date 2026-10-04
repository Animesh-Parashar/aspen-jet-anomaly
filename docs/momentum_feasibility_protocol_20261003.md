# Momentum matching feasibility — 3 October 2026

Read only verified training and validation jet kinematics, event IDs, source
indices and labels from the existing processed Aspen shard and LHCO background.
No test/reference HDF5, signal files, model scores or checkpoints are read.
Use the already reported reference central 90% bounds, 1198.5738708496094 to
1740.9484130859366 GeV, with ten equal-width bins. Also report a fixed 0–5000 GeV
grid in 100 GeV bins to describe broader overlap. Report overflow separately
through total row accounting; do not equate broad overlap with reference support.

Count all jet rows and unique event identities. For an explicitly feasible
event-independent selection policy, choose the highest-pT available jet per event
before applying the range; ties use source array order. In each bin, the smaller
domain count gives the per-domain subsampling capacity. This is not a proof of
optimal matching over every possible event assignment. Report unique events with
any jet in the central range as a separate upper bound.

Cross-check histogram bins with direct Boolean counts, validate labels, unique
source rows, preprocessing counts and training/validation event separation.
Hash all used columns, metadata, protocol and analysis source. No new training,
data download, sample selection or modification to the paper is authorized by
this feasibility count. Never infer full-collection yield from one shard as fact.

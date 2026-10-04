# Signal and background kinematics

This descriptive extension uses only the already-scored final-partition arrays.
It is specified after the primary evaluation and representation diagnostics.
No new models, samples, reference pools, score directions or cuts are selected.

For both topologies, verify saved-array hashes against the final evidence manifest,
binary labels, shared background identities and observables, nonnegative finite
weights and integer available multiplicity. Report jet pT and available constituent
multiplicity distributions on the original population and on the existing common-pT
population, using exactly the saved class weights. A single figure shows both
populations. Because class balancing differs by topology, retain both balanced
background distributions rather than silently pooling their weights.

Save weighted means and 10/50/90% inverse-ECDF quantiles, fixed-bin histograms with
explicit underflow/overflow probabilities, and descriptive AUCs treating each
observable as an increasing score. Check weighted histogram normalization and
quantiles against the empirical CDF and check AUC against a separate tied-rank
weighted-pair calculation. Preserve all seeds' existing raw-score AUCs for context.
Do not refit the encoders or claim that marginal distributions and background
correlations prove a causal explanation of multidimensional scores.

Fixed bins: pT 0--5000 GeV in 100-GeV bins; multiplicity edges -0.5--299.5 in
steps of 5, with explicit overflow. Display limits may be narrower but all
numerical summaries use the full saved samples. Computation uses one scheduler
CPU, one numerical-library thread, 2 GB RAM, no GPU. Numerical checks and source
hashes accompany the saved summaries.

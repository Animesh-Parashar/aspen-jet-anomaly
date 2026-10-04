# Transfer descriptive diagnostics — 3 October 2026

Specified after the existing evaluation results were inspected.
These analyses are post-hoc diagnostics, not a new independent test. No training,
score sign, checkpoint, reference pool, operating point or evaluation sample changes.

## Training support

Reconstruct exactly the 100,000 training-row selections using the completed run
manifests and the original sorted NumPy sampling rule (seed 20260925). Check all
six manifests agree on each domain's selection. Read verified core4 HDF5 inputs
in contiguous blocks; use the previously fixed 10,000 LHCO reference jets (seed
20261004), verifying their event identities and observables against saved evidence.

Report available-constituent jet vector pT, mass and multiplicity; retained
constituent log(pT/GeV) uses equal total weight per jet (1/n_retained per particle).
Also save ordinary particle-weighted distributions. Fixed plotting bins have
explicit underflow/overflow counts; do not silently normalize away tails.
Report quantiles, training fraction in the central 90% of reference pT, and
reference fraction outside the central 98% of each training pT distribution.
These central-range summaries describe concentration, not strict support or a
causal attribution. Values use unweighted available constituent sums, not the
producer's corrected/PUPPI jet pT. The 150-slot saturation fraction is not the
fraction proved to have lost particles at production.

Check the maximum valid local radius: if <= pi then *all* planar rotation angles
stay in the phi chart. Otherwise report one fixed uniform-angle Monte Carlo
estimate per selected jet (seed 20261009), not the historical realized rejection
rate. Include leading signal mass distributions from already-scored queries;
do not infer truth X/Y labels from reconstructed mass. Save all training/validation
loss histories without selecting epochs or comparing domain losses as quality scores.

## Cached representations

Verify extraction-file hashes, scorer evidence hashes, shared background IDs,
labels and observables. For all nine frozen backbones, compute background-reference
and background-query covariance spectra, entropy effective rank and participation
rank. Reuse the established geometry calculation; check spectra against SVD and
raw-kNN distances against independent direct distances on a fixed small subset.

For every domain/seed/scorer, report tie-aware Spearman correlations between score
and background jet mass, vector pT, available multiplicity and retained girth on
the original 10,000 background queries. Constant vectors yield undefined/null,
not zero. No correlation p-values or multiplicity-selected discovery claims.
Save signal/background score quantiles for both topologies. Correlation is not
causation; low effective rank is not by itself pathological collapse; a small
kNN distance describes proximity in the chosen embedding metric, not a calibrated
physical density or likelihood.

## Presentation and validation

Show primary correlations and reference spectra; provide all results in JSON.
Add existing threshold-refitting bootstrap intervals to the operating-point plot,
with separate faint per-seed points. Show count-matched random-selection KS bands
at each method's achieved background acceptance; these are reference expectations,
not an independently measured random signal selection or joint confidence region.
Acceptance profiles show seed means and seed ranges (not confidence bands), plus
fixed-threshold Wilson intervals for controls. Keep original and balanced AUCs.

CPU jobs: workq, one CPU and 4 GB each, at most two concurrent jobs, no GPUs.
Use one numerical-library thread. Tests cover histogram normalization/tails,
padding exclusion, tied-rank/constant handling, spectra, and rotation bounds.
Preserve existing evidence; add separate source/result hashes and update local
manuscript packages only after completion and numerical checks. No public upload.

# Controlled core4 study: development protocol

This is a scientific experiment specification, not a claim of recovered results.
Historical scripts/results are not rerun or relabeled as corrected results.

## Research question and scope

Compare real-data and simulation contrastive pretraining, reconstruction training,
and an untrained representation under matched feature definitions and scoring.
Aspen and LHCO have different detector response and event selection: equal feature
names do not isolate the causal effect of training on real data. The initial
comparison is a domain-transfer benchmark. No discovery or first-use claim follows.

## Architecture v1

All scored representations come from `JetBackbone` in
`src/models/controlled_encoder.py`: core4 input, fixed division by [1,1,5,5],
linear embedding to 128, four independently initialized pre-LN transformer
blocks, eight heads, FFN width 512 with ReLU, dropout 0.1, final LayerNorm,
masked mean pooling. There are no positional embeddings. The representation
has 128 components and is not L2-normalized. Sum pooling is a development-only
ablation; its choice must be fixed before test evaluation.

Contrastive training adds a 128 -> 128 -> 256 GELU projection MLP and L2
normalization for symmetric NT-Xent. Reconstruction uses the exact same scored
backbone, without an extra encoder MLP. Its decoder expands to 50 ordered slots,
then applies two width-64 transformer blocks and predicts four scaled features.
Targets use descending-pT ordering; this is ordered-slot reconstruction, not a
permutation-invariant set loss. The reconstruction objective averages scaled
feature errors per jet, then across jets, with wrapped phi residuals. Both the
loss scaling and decoder are experimental choices, not copied paper defaults.

The final LayerNorm and independent initialization differ from the historical
implementation. Revised checkpoints are intentionally incompatible with it.
For a given training seed, backbone initialization is identical across objectives.

This is a compact JetCLR-inspired adaptation, not an exact reproduction of
JetCLR, AnomalyCLR or DarkCLR. Relevant primary sources:
- JetCLR: https://arxiv.org/abs/2108.04253 (sum pooling and different dimensions).
- SimCLR: https://proceedings.mlr.press/v119/chen20j.html (projection and loss).
- AnomalyCLR: https://arxiv.org/abs/2301.04660 (different augmentations/scoring).
- DarkCLR: https://arxiv.org/abs/2312.03067 (anomalous removal, different score).

## Augmentations

The initial pilot uses only random rotations in the relative eta-phi plane,
applied before fixed feature scaling. This is an approximate local jet-shape
symmetry, not an exact detector symmetry or Lorentz transformation. Rotations
leaving the valid phi coordinate chart are rejected per jet. Padding remains zero.
This reduced augmentation set is a development starting point, not a claim of
optimality or exact JetCLR reproduction.

Optional collinear splitting updates log(pT fraction) and log(pT/GeV) together,
keeps daughter directions equal, conserves represented momentum, re-sorts tokens,
and skips full jets. It supports core4 only; charge/PID splitting is undefined.
Its availability depends on padding/multiplicity, which must be measured in an
ablation. It is disabled in the pilot. Random removal, coordinate translation,
and independent smearing of redundant momentum features are excluded. The
network is not claimed to be infrared/collinear safe by construction.

## Data and model selection

Use only verified v2 outputs. The training API rejects reference/test/signal
partitions. Previously audited events stay in validation. Use the same seeded
subset selection for all objectives; match training sample counts across domains.
Development pilot: 20,000 training and 4,096 background/unlabeled validation jets
per domain, three epochs, batch 128, seed 17, FP32 AdamW, fixed learning rate 3e-4,
weight decay 1e-4, gradient clip 1, temperature 0.1. No signal labels are used
for the pilot or checkpoint selection. Save the final fixed-epoch checkpoint.
Validation views are fixed across epochs and RNG state is restored afterward.

GPU timing and convergence studies must precede a final run configuration.
The pilot's epoch count is not a final scientific training budget. Final budgets
must permit adequate convergence for both objectives, with equal tuning effort;
identical epochs alone do not guarantee a fair comparison. Record examples seen,
training time, total parameters, backbone parameters and memory. Plan five
independent training seeds for the central comparison, subject to measured cost.
Freeze all settings and the exact selection rule before accessing test outcomes.

## Final evaluation specification (not executed)

Primary comparator: same Euclidean k=10 distance to the kth nearest reference jet,
on the same 128D backbone extraction for real contrastive, simulated contrastive,
real reconstruction and random models. Fix the same reference and test identities
across models. Do not optimize k or embeddings on test outcomes. Do not fit a
normalizer to test data. The different objectives still shape geometry differently;
matching the architecture does not prove a unique causal explanation for a gap.
Report AE reconstruction error separately, not as the same scoring protocol.

Report ROC AUC and background efficiency/rejection at prespecified signal
 efficiencies (0.3 and 0.5); include finite-sample intervals and independent-training
variation separately. If no background events pass, report a bound, not infinite
rejection. Use paired comparisons on shared test events. Never select the best
training seed or optimize the reported working point on the test set. Signal
labels may be used for final metric estimation, not model selection. Any later
signal-guided development must be disclosed and use validation only.

Compare against prespecified simple mass/multiplicity/substructure baselines;
quantify score dependence on mass and pT and performance within kinematic bins.
Additional observables must have consistent definitions within each benchmark.
Reference-contamination studies modify the reference sample, not merely the test
signal fraction; use disjoint signal events and retain a clean comparison.
Treat event identity as the unit of splitting and statistical resampling where
multiple jets share an event. Signal-topology claims require verified dataset
provenance; do not infer multiple topologies from binary labels.

## Real-data physics gate

The initial analysis is development-only characterization, not a resonance result.
Use stored CMS soft-drop mass separately from derived unweighted constituent mass.
ParticleNet scores are predictions, never ground-truth signal labels. Mass-window
occupancy is not signal yield. Audit trigger decisions, efficiency controls,
luminosity certification, event completeness, pileup and calibration uncertainties
before choosing a real-data search. A blind signal region, background-model closure,
and uncertainty model are required before interpreting an enrichment as a signal.

The AOJ paper explicitly flags trigger-selection biases:
https://arxiv.org/html/2412.10504v2
Real-data AD already exists; compare precisely against:
https://arxiv.org/html/2510.24066v1
https://arxiv.org/html/2603.23593v1

## Convergence development run

After the three-epoch pilot completed, the next fixed-budget study uses
100,000 training jets and 8,192 background/unlabeled validation jets per domain,
50 epochs, training seed 17, and otherwise identical pilot settings (including
batch size 128). Three runs execute sequentially on one requested GPU:
Aspen contrastive, Aspen reconstruction, LHCO contrastive. The PBS walltime
limit is two hours, below the admin's three-day GPU limit. More GPU memory or
host CPUs are not requested merely to exhaust an allowance.

Each epoch writes a live progress file and atomically replaces a checkpoint
containing model, optimizer and RNG state. A recovery checkpoint is not a
completed-run marker; downstream scientific evaluation must require COMPLETE.json.
There is no automatic resume CLI yet. Recovery would explicitly restore all
saved state and continue the saved epoch count, not silently restart training.

Assess each objective's validation curve separately. A relative change below
1% between the previous five-epoch mean and the last five-epoch mean is only a
plateau diagnostic, not proof of convergence. Inspect overfitting, variation
and representation collapse before choosing the final budget. No loss comparison
between different objectives and no test AUC-based selection are permitted.
The study is single-seed development, not a five-seed performance result.

## Validation-driven reconstruction extension (2026-09-30)

After the 50-epoch geometry audit, extend only Aspen reconstruction to a fixed
100 total epochs, keeping every other configuration field unchanged. This is
single-seed development, not the final objective comparison. Restore the model,
AdamW state, loader generator and Python/NumPy/Torch/CUDA RNG states from the
completed epoch-50 recovery checkpoint. Preserve the parent run; write all
continuation outputs into a fresh directory. The continuation script verifies
parent source hashes, configuration, data provenance and checkpoint agreement.
A synthetic dropout/AdamW continuation test must reproduce an uninterrupted step.
No early stopping or signal/test metrics are used to select epoch 100.

Physics diagnostics use 8,192 validation jets per domain, split by event identity
into disjoint neighbor/query pools. All models use the same split and Euclidean
k=10 score. A top-10%-of-query selection is a descriptive mass-sculpting diagnostic,
not a deployable calibrated threshold or physics significance. Report score and
embedding-norm Spearman correlations with constituent-sum mass, vector jet pT,
full stored and retained multiplicity, and retained scalar-pT fraction; include
quintile selection fractions and correlations within pT quartiles. Quantile edges
are validation-only, with ties retained. No confidence intervals or causal
interpretation are attached to this exploratory check.

Mass is the unweighted sum of all stored constituent four-vectors before the
50-token truncation, not CMS soft-drop mass. Full stored multiplicity remains
limited by producer storage for Aspen. Cross-domain definitions do not remove
detector-response, event-selection or constituent-storage differences. Both
before- and after-extension diagnostics remain in validation partitions; final
reference, signal and test pools are unopened by these scripts.

## Bounded convergence extensions (2026-10-01)

Extend Aspen and LHCO contrastive from epoch 50 to a fixed epoch 100, and masked
Aspen reconstruction from epoch 100 to a fixed epoch 150. Restore optimizer and all
saved random states with the tested continuation script. Keep data counts, seed,
batch size, learning rate and all architecture/objective choices unchanged. Preserve
parents and write new run directories. Run sequentially on one GPU with a two-hour
allocation; summarize fixed 5/10/20-epoch validation windows after completion.
No best-epoch or signal/test selection. These remain seed-17 development runs.
The final schedule decision is pending this evidence, not automatic upon completion.

## Fixed reconstruction 200-epoch endpoint

Continue masked Aspen reconstruction from the completed epoch-150 run to epoch
200, holding every other setting fixed. Preserve the parent; restore the optimizer
and saved RNG states. Review fixed 5/10/20-epoch validation windows after completion.
This is the bounded development endpoint proposed after the previous extension;
no automatic further extension or final checkpoint selection is authorized by a
small loss improvement alone. Test and signal data remain excluded.

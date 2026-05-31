# Aspen Jet Anomaly Detection [![DOI](https://zenodo.org/badge/1253588950.svg)](https://doi.org/10.5281/zenodo.20472552)

Self-supervised contrastive learning for jet anomaly detection on real CMS open data (AspenOpenJets).

**Paper:** [coming soon]  
**Dataset:** [AspenOpenJets](https://www.fdr.uni-hamburg.de/record/16505) | [LHCO R&D](https://zenodo.org/records/6466204)

---

## What is this?

The LHC produces jets (collimated sprays of particles from quark/gluon collisions)
at an enormous rate. Most jets are ordinary QCD background. A tiny fraction may
come from undiscovered particles. Since we do not know what new physics looks like,
we need model-agnostic anomaly detection: find unusual jets without knowing in
advance what to search for.

We train a transformer encoder with NT-Xent contrastive loss on up to 10M real
CMS jets from AspenOpenJets (CMS 2016 Open Data, 13 TeV). After training, jets
from new-physics decays should appear far from the background cluster in latent
space. The distance to the k-th nearest background neighbor is the anomaly score.

**This is the first constituent-level contrastive anomaly detection on real Run-2
CMS data.** Prior work (JetCLR, AnomalyCLR, DarkCLR) used Monte Carlo simulation
for training.

---

## Key Results

Evaluated on the LHCO R&D benchmark (Z' signal, 5 seeds, k-NN scoring).

| Model | Training data | Features | Best epoch | 2-prong AUC |
|---|---|---|---|---|
| A (contrastive) | Real CMS, 2M jets | 7 | ep40 | 0.6065 +/- 0.0072 |
| A (contrastive) | Real CMS, 10M jets | 7 | ep75 | 0.6270 +/- 0.0069 |
| A (contrastive) | Real CMS, 10M jets | 4 (sim-compatible) | ep75 | 0.6444 +/- 0.0050 |
| B2 (contrastive) | Simulation, 1M jets | 7 | ep75 | 0.6286 +/- 0.0043 |
| C (autoencoder) | Real CMS, 2M jets | 7 | ep50 | 0.5467 +/- 0.0053 |

Contrastive learning outperforms the autoencoder baseline by ~0.08-0.12 AUC.
With matched training budgets, real-data and simulation-trained models are
competitive on this simulation benchmark.

**Augmentation importance (2-prong AUC drop vs full baseline):**
soft-drop −0.048 | rotation −0.040 | collinear split −0.017 | translation −0.017 | pT smearing +0.005 (redundant on real data)

See `docs/results.md` for full tables.

---

## Repository Structure

```
src/
  models/         JetEncoder (transformer + NT-Xent), JetAutoencoder
  data/           AspenOpenJets loader, LHCO loader, preprocessed loader
  augmentations/  JetAugmentation: rotate, translate, pT smear, soft drop, collinear split
  anomaly/        KNNAnomalyScorer
scripts/
  preprocess/     preprocess_aspen.py, preprocess_lhco_bg.py, inspect_data.py
  train/          train_contrastive.py, train_autoencoder.py, augmentation_ablation.py
  eval/           ablation_eval.py, epoch_auc.py, scaling_curve.py,
                  latent_viz.py, mass_decorr.py, signal_injection.py,
                  substructure_overlay.py
docs/
  architecture.md   Model architecture and training configuration
  results.md        Final results tables and diagnostic evaluations
```

Data, checkpoints, and logs are not tracked in git. See the Data section below
for download instructions.

---

## Setup

```bash
git clone git@github.com:Animesh-Parashar/aspen-jet-anomaly.git
cd aspen-jet-anomaly
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data

### Download

```bash
# AspenOpenJets (one batch ~8 GB, we use batches 0-4)
wget -c "https://www.fdr.uni-hamburg.de/record/16505/files/RunG_batch0.h5" \
     -O data/aspen/RunG_batch0.h5

# LHCO background (2.6 GB)
wget -c "https://zenodo.org/records/6466204/files/events_anomalydetection_v2.h5" \
     -O data/lhco/events_anomalydetection_v2.h5

# LHCO 3-prong signal
wget -c "https://zenodo.org/records/6466204/files/events_anomalydetection_Z_XY_qqq.h5" \
     -O data/lhco/events_anomalydetection_Z_XY_qqq.h5
```

### Preprocess

```bash
# Convert raw Aspen HDF5 to float16 preprocessed format (run once per batch)
python scripts/preprocess/preprocess_aspen.py \
  --input  data/aspen/RunG_batch0.h5 \
  --output data/aspen/RunG_batch0_processed.h5

# Preprocess LHCO background (run once)
python scripts/preprocess/preprocess_lhco_bg.py \
  --lhco data/lhco/events_anomalydetection_v2.h5 \
  --out  data/lhco/lhco_bg_jets.h5
```

---

## Training

Always run inside `screen` to survive SSH disconnects:

```bash
screen -dmS <name> bash -c \
  'source venv/bin/activate && python scripts/train/... 2>&1 | tee logs/<name>.log'
```

```bash
# Model A - real data, 2M jets (baseline)
python scripts/train/train_contrastive.py \
  --data data/aspen/RunG_batch0_processed.h5 --data_type aspen \
  --epochs 50 --batch_size 4096 --lr 3e-4 \
  --save_dir data/checkpoints/model_A

# Model A - real data, 10M jets
python scripts/train/train_contrastive.py \
  --data data/aspen/RunG_batch{0..4}_processed.h5 --data_type aspen \
  --epochs 75 --batch_size 8192 --lr 1e-4 --grad_clip 0.5 \
  --save_dir data/checkpoints/model_A_10M

# Model B2 - simulation baseline (LHCO QCD)
python scripts/train/train_contrastive.py \
  --data data/lhco/lhco_bg_jets.h5 --data_type sim \
  --epochs 75 --batch_size 4096 --lr 3e-4 \
  --save_dir data/checkpoints/model_B2

# Model C - autoencoder baseline
python scripts/train/train_autoencoder.py \
  --data data/aspen/RunG_batch0_processed.h5 \
  --epochs 50 --save_dir data/checkpoints/model_C
```

---

## Evaluation

```bash
# Ablation: fair k-NN comparison across all models (5 seeds)
python scripts/eval/ablation_eval.py \
  --ckpt_A  data/checkpoints/model_A/encoder_epoch040.pt \
  --ckpt_B2 data/checkpoints/model_B2/encoder_epoch075.pt \
  --ckpt_C  data/checkpoints/model_C/autoencoder_epoch050.pt \
  --out_dir data/results/ablation

# Epoch-AUC curve (find best checkpoint for a model)
python scripts/eval/epoch_auc.py \
  --ckpt_dir_A data/checkpoints/model_A \
  --epochs_A 10 20 30 40 50 \
  --out_dir data/results/epoch_auc

# Scaling curve (reads ablation JSONs)
python scripts/eval/scaling_curve.py

# Mass decorrelation check
python scripts/eval/mass_decorr.py \
  --ckpt data/checkpoints/model_A_10M/encoder_epoch075.pt \
  --out_dir data/results/mass_decorr

# Signal injection curve (AUC vs S/B ratio)
python scripts/eval/signal_injection.py \
  --ckpt data/checkpoints/model_A_10M/encoder_epoch075.pt \
  --out_dir data/results/signal_injection
```

---

## Hardware

NVIDIA RTX 6000 Ada (49 GB VRAM), 128-core CPU, 251 GB RAM.
Training times: 2M model ~20 min (50 ep), 10M model ~6 hours (75 ep).

---

## Citation

```bibtex
@article{parashar2026jet,
  title   = {Self-Supervised Jet Anomaly Detection on Real LHC Data},
  author  = {Parashar, Animesh},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

MIT License. See LICENSE file.

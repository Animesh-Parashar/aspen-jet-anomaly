# Architecture

---

## Input Representation

Each jet is represented as a variable-length sequence of up to 50 constituents
(zero-padded to a fixed length). Each constituent has 7 features:

| Index | Feature | Description |
|---|---|---|
| 0 | eta_rel | Pseudorapidity relative to jet axis |
| 1 | phi_rel | Azimuthal angle relative to jet axis, wrapped to [-pi, pi] |
| 2 | log_pT_rel | log(pT_constituent / sum_pT_jet) |
| 3 | log_E | log(constituent energy) |
| 4 | d0 | Transverse impact parameter [cm], clipped to [-5, 5] |
| 5 | dz | Longitudinal impact parameter [cm], clipped to [-5, 5] |
| 6 | charge | Particle charge (-1, 0, +1) |

Features 4-6 are only available in real CMS data. In LHCO simulation they are
set to zero. A 4-feature variant (indices 0-3 only) is used to enable a
domain-mismatch-free comparison between real and simulated data.

Padding slots have all features set to 0 and are identified by a boolean mask
(True = padding, False = real constituent). Masked positions are excluded from
attention and pooling.

---

## Model A / B2 - Contrastive Encoder (JetEncoder)

`src/models/encoder.py`

```
Input: (B, 50, 7)  +  mask: (B, 50)

1. Input projection
   Linear(7 -> 128)                         -> (B, 50, 128)

2. Transformer Encoder x4 layers
   Each layer (Pre-LN):
     LayerNorm(128)
     MultiHeadAttention(d_model=128, nhead=8, head_dim=16)
       src_key_padding_mask applied (padding tokens ignored)
     Dropout(0.1) + residual
     LayerNorm(128)
     FFN: Linear(128->512) -> GELU -> Linear(512->128)
     Dropout(0.1) + residual
                                             -> (B, 50, 128)

3. Masked mean pooling
   Average over real (non-padding) tokens    -> (B, 128)   [backbone output]

4. Projection head  [training only, discarded at evaluation]
   Linear(128->128) -> GELU -> Linear(128->256)
   L2 normalize                              -> (B, 256)   [contrastive embeddings]
```

**Parameters:** 843,648 total (input proj: 1,024 | transformer: ~794K | projection: ~49K)

**Design notes:**

- Pre-LN (norm_first=True): more stable training than post-LN for deep transformers
- Masked mean pooling: permutation-invariant aggregation that correctly ignores
  padding. Formula: sum(h * valid) / clamp(sum(valid), min=1)
- Backbone output for anomaly scoring, not projection head: the projection head
  collapses distance geometry toward a unit hypersphere (useful for NT-Xent loss);
  the 128-dim backbone output preserves metric structure needed for k-NN
- enable_nested_tensor=False: required for compatibility with manual padding masks

---

## Contrastive Training - NT-Xent Loss

`src/augmentations/jet_augmentations.py`, `src/models/encoder.py`

### Data Augmentation

Two independent views of each batch are created by stochastically applying:

| Augmentation | Probability | Effect |
|---|---|---|
| rotate_batch | 1.0 | Random rotation in (eta, phi) plane, angle ~ Uniform[-pi, pi] |
| translate_batch | 1.0 | Random shift of all constituents by Uniform[-0.2, 0.2] per axis |
| smear_pt_batch | 0.5 | Gaussian noise on log_pT_rel, sigma=0.1, valid constituents only |
| soft_drop_batch | 0.5 | Drop each valid constituent with probability 0.1; keep at least one |
| collinear_split_batch | 0.3 | Split one constituent into two collinear daughters, z ~ Uniform[0.2, 0.8] |

All augmentations are fully vectorized on GPU with no Python loops over jets.

### Loss Function

```
z = concat([z1, z2])                      # (2B, 256), L2-normalized
sim[i,j] = dot(z_i, z_j) / temperature   # temperature = 0.07
            (self-pairs masked to -inf)

positive pair for row i:  row i+B (and vice versa)
loss = cross_entropy(sim, positive_labels)
```

With batch size 8192: 16,382 negatives per positive pair.

### Training Configuration

| Hyperparameter | Small-scale (2M, 4M, 6M) | Large-scale (10M) |
|---|---|---|
| Batch size | 4,096 | 8,192 |
| Learning rate | 3e-4 | 1e-4 |
| LR schedule | Linear warmup (5 ep) + cosine decay | Same |
| Gradient clipping | 1.0 | 0.5 |
| Optimizer | AdamW (weight_decay=1e-4) | Same |
| Mixed precision | bfloat16 AMP | Same |
| Compilation | torch.compile (Triton kernels) | Same |

---

## Model C - Transformer Autoencoder (JetAutoencoder)

`src/models/autoencoder.py`

Smaller architecture used as a non-contrastive baseline to isolate the effect
of the training objective.

```
Input: (B, 50, 7)  +  mask: (B, 50)

ENCODER
  Linear(7 -> 64)                            -> (B, 50, 64)
  TransformerEncoder x2 layers (d_model=64, nhead=4, FFN=128)
  Masked mean pool                            -> (B, 64)
  Linear(64->64) -> ReLU -> Linear(64->128)  -> (B, 128)   [latent z]

DECODER
  Linear(128 -> 50*64), reshape              -> (B, 50, 64)
  TransformerEncoder x2 layers (same config) -> (B, 50, 64)
  Linear(64 -> 7)                            -> (B, 50, 7)  [reconstruction]
```

**Parameters:** ~400K

**Training loss:** masked MSE over valid constituents only:
`loss = sum((x_hat - x)^2 * valid_mask) / n_valid_elements`

**Anomaly scoring:** k-NN distance on the 128-dim bottleneck embedding z,
identical protocol to Models A and B2. The reconstruction error is not used
for anomaly scoring (it produces below-random AUC due to domain gap between
real Aspen training and simulated LHCO evaluation).

---

## Anomaly Scoring - k-NN

`src/anomaly/knn_scorer.py`

```
1. fit(reference_embeddings)
   Build NearestNeighbors index (sklearn, Euclidean, k=10, n_jobs=-1)
   on 20,000 LHCO background jets

2. score(test_embeddings)
   For each jet: find 10 nearest neighbors in the reference index
   anomaly_score = distance to the 10th nearest neighbor
   (higher = more anomalous = farther from background cluster)
```

The k=10 distance (rather than k=1) provides a smoother estimate of local
density, reducing sensitivity to individual outliers in the reference set.

### Evaluation Protocol

| Parameter | Value |
|---|---|
| Reference background | 20,000 jets (random sample) |
| Test background | 20,000 jets (disjoint from reference) |
| Test signal | 2,000 jets |
| Seeds | 5 independent random splits |
| Reported metric | ROC-AUC mean +/- std across seeds |
| Additional metrics | Background rejection at 50% and 30% signal efficiency |

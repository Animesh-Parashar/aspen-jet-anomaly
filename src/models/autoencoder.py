"""
Constituent-level Transformer Autoencoder — Model C baseline.

Architecture:
  Encoder: same transformer as JetEncoder → (B, latent_dim)
  Decoder: linear expand → transformer decoder → per-constituent reconstruction
  Loss:    masked MSE on constituent features (padding ignored)

Anomaly score at inference = mean reconstruction error over valid constituents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JetAutoencoder(nn.Module):
    def __init__(self, input_dim=7, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, latent_dim=128, max_constituents=50,
                 dropout=0.1):
        super().__init__()
        self.max_constituents = max_constituents
        self.d_model = d_model

        # --- Encoder (identical to JetEncoder backbone) ---
        self.input_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.enc_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, latent_dim),
        )

        # --- Bottleneck → token sequence ---
        self.latent_to_seq = nn.Linear(latent_dim, max_constituents * d_model)

        # --- Decoder transformer ---
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            dec_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_proj = nn.Linear(d_model, input_dim)

    def encode(self, x, key_padding_mask=None):
        """Returns (B, latent_dim) embedding (NOT L2-normalized)."""
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            pooled = h.mean(1)
        return self.enc_proj(pooled)

    def decode(self, z, key_padding_mask=None):
        """z: (B, latent_dim) → reconstructed constituents (B, N, input_dim)."""
        B = z.size(0)
        seq = self.latent_to_seq(z).view(B, self.max_constituents, self.d_model)
        seq = self.decoder(seq, src_key_padding_mask=key_padding_mask)
        return self.output_proj(seq)

    def forward(self, x, key_padding_mask=None):
        z = self.encode(x, key_padding_mask)
        x_hat = self.decode(z, key_padding_mask)
        return x_hat, z


def masked_mse_loss(x_hat, x, mask):
    """
    MSE between reconstructed and original constituents, ignoring padding.
    x_hat, x: (B, N, F); mask: (B, N) True=padding.
    """
    valid = (~mask).float().unsqueeze(-1)          # (B, N, 1)
    err   = ((x_hat - x) ** 2 * valid).sum()
    n     = valid.sum() * x.size(-1) + 1e-8
    return err / n


def reconstruction_anomaly_score(model, x, mask):
    """
    Per-jet reconstruction error (anomaly score for inference).
    Returns (B,) tensor.
    """
    model.eval()
    with torch.no_grad():
        x_hat, _ = model(x, mask)
    valid = (~mask).float().unsqueeze(-1)           # (B, N, 1)
    per_const_err = ((x_hat - x) ** 2).mean(-1)    # (B, N)
    score = (per_const_err * (~mask).float()).sum(1) / (~mask).float().sum(1).clamp(min=1)
    return score

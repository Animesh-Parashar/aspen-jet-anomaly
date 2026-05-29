"""
Transformer-based contrastive jet encoder (JetCLR-style).
Architecture: input proj → Transformer encoder → masked mean pool → projection MLP → L2-norm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JetEncoder(nn.Module):
    """
    Permutation-invariant encoder for variable-length jet constituents.

    input_dim:       number of constituent features
    d_model:         transformer hidden dimension
    nhead:           attention heads  (must divide d_model)
    num_layers:      transformer encoder layers
    dim_feedforward: FFN inner dimension (default 4×d_model)
    latent_dim:      projection-head output dimension (used for contrastive loss)
    dropout:         attention + FFN dropout
    """

    def __init__(self, input_dim=7, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=None, latent_dim=256, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Projection head (2-layer MLP) — used only during contrastive training
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

    def backbone(self, x, key_padding_mask=None):
        """
        Returns (B, d_model) pooled transformer output — used for anomaly scoring.
        NOT L2-normalized, preserving distance geometry for k-NN.
        """
        h = self.input_proj(x)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            return (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        return h.mean(1)

    def forward(self, x, key_padding_mask=None):
        """
        Returns (B, latent_dim) L2-normalized projection embeddings.
        Used only for NT-Xent loss during training.
        """
        h = self.backbone(x, key_padding_mask)
        z = self.proj(h)
        return F.normalize(z, dim=-1)


class NTXentLoss(nn.Module):
    """
    Normalized temperature-scaled cross-entropy (SimCLR NT-Xent).
    Input: two L2-normalized batches z1, z2 of shape (B, D).
    With large batches the 2B−2 negatives per positive give strong gradient signal.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z1, z2):
        B = z1.size(0)
        z   = torch.cat([z1, z2], dim=0)          # (2B, D)
        sim = torch.mm(z, z.T) / self.T            # (2B, 2B)

        # Exclude self-similarity
        eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(eye, float("-inf"))

        # Positive pairs: i↔i+B
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device),
        ])
        return F.cross_entropy(sim, labels)

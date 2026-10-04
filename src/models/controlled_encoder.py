"""Versioned compact JetCLR-inspired models; historical architectures unchanged."""
import torch
from torch import nn
from torch.nn import functional as F


class JetBackbone(nn.Module):
    def __init__(self, width=128, heads=8, layers=4, feedforward=512,
                 dropout=0.1, pooling='mean'):
        super().__init__()
        if pooling not in ('mean', 'sum'):
            raise ValueError('pooling must be mean or sum')
        self.pooling = pooling
        # Fixed unit scaling shared by every domain/objective, no test fitting.
        self.register_buffer('feature_scale', torch.tensor([1., 1., 5., 5.]))
        self.input_proj = nn.Linear(4, width)
        # Construct each layer independently, rather than clone identical weights.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, feedforward, dropout,
                                       activation='relu', batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        self.width = width

    def forward(self, x, mask):
        if x.ndim != 3 or x.shape[-1] != 4 or mask.shape != x.shape[:2]:
            raise ValueError('Expected core4 inputs and matching padding mask')
        if mask.dtype != torch.bool or mask.all(dim=1).any():
            raise ValueError('Boolean padding mask must retain at least one token')
        h = self.input_proj(x.masked_fill(mask[..., None], 0) / self.feature_scale)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=mask)
        h = self.final_norm(h).masked_fill(mask[..., None], 0)
        pooled = h.sum(dim=1)
        if self.pooling == 'mean':
            pooled = pooled / (~mask).sum(dim=1, keepdim=True)
        return pooled


class ContrastiveJetModel(nn.Module):
    def __init__(self, backbone=None, projection_dim=256):
        super().__init__()
        self.backbone = JetBackbone() if backbone is None else backbone
        d = self.backbone.width
        self.projection = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, projection_dim))

    def encode(self, x, mask):
        return self.backbone(x, mask)

    def forward(self, x, mask):
        return F.normalize(self.projection(self.encode(x, mask)), dim=-1)


class ReconstructionJetModel(nn.Module):
    """Same scored backbone as contrastive; ordered pT-slot decoder is auxiliary.

    Input/target tokens must be sorted by descending pT. This loss is not a
    permutation-invariant set reconstruction loss. No extra encoder MLP.
    """
    def __init__(self, backbone=None, max_constituents=50, decoder_width=64):
        super().__init__()
        self.backbone = JetBackbone() if backbone is None else backbone
        self.max_constituents = max_constituents
        self.decoder_width = decoder_width
        self.expand = nn.Linear(self.backbone.width, max_constituents * decoder_width)
        self.decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(decoder_width, 4, 128, 0.1,
                                       activation='relu', batch_first=True, norm_first=True)
            for _ in range(2)])
        self.output = nn.Sequential(nn.LayerNorm(decoder_width), nn.Linear(decoder_width, 4))

    def encode(self, x, mask):
        return self.backbone(x, mask)

    def forward(self, x, mask):
        if x.shape[1] != self.max_constituents:
            raise ValueError('Decoder requires configured number of slots')
        z = self.encode(x, mask)
        h = self.expand(z).reshape(-1, self.max_constituents, self.decoder_width)
        for layer in self.decoder:
            h = layer(h, src_key_padding_mask=mask)
        return self.output(h), z

    def reconstruction_scores(self, prediction, target, mask):
        # Equal weight per jet; phi residual is periodic. Decode scaled features.
        residual = prediction * self.backbone.feature_scale - target
        residual = torch.stack([residual[..., 0],
            torch.atan2(residual[..., 1].sin(), residual[..., 1].cos()),
            residual[..., 2], residual[..., 3]], dim=-1)
        error = (residual / self.backbone.feature_scale).square().mean(dim=-1)
        return error.masked_fill(mask, 0).sum(dim=1) / (~mask).sum(dim=1)

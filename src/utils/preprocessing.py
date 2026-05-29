"""Shared preprocessing utilities."""

import numpy as np
import torch


def normalize_features(x, mean=None, std=None, eps=1e-6):
    """
    Standardize constituent features. If mean/std are None, compute from x.
    Returns (x_norm, mean, std).
    """
    if mean is None:
        mean = x.mean(axis=0)
        std = x.std(axis=0) + eps
    return (x - mean) / std, mean, std


def pad_collate(batch):
    """Custom collate for variable-constituent jets — pads to max in batch."""
    xs, masks = zip(*batch)
    return torch.stack(xs), torch.stack(masks)

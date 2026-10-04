"""Prospective positive-view ablations; finite robustness priors, not exact IRC safety."""
import math
import torch
from src.augmentations.controlled_augmentations import CoreAugmentation


def soft_perturb(x, mask):
    if x.ndim != 3 or x.shape[-1] != 4 or mask.shape != x.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Expected core4 and matching boolean padding mask")
    out = x.clone()
    sigma = 0.1 / x[..., 3].exp().clamp_min(2.)
    noise = torch.randn_like(x[..., :2]) * sigma[..., None]
    out[..., :2] += noise
    out[..., 1] = (out[..., 1] + math.pi).remainder(2*math.pi) - math.pi
    return out.masked_fill(mask[..., None], 0), mask.clone()


class PhysicsAugmentation(CoreAugmentation):
    def __init__(self, variant):
        if variant not in ("rotation_split", "rotation_soft"):
            raise ValueError("Unknown ablation")
        super().__init__(1. if variant == "rotation_split" else 0.)
        self.variant = variant

    def view(self, x, mask):
        out, out_mask = super().view(x, mask)
        if self.variant == "rotation_soft":
            out, out_mask = soft_perturb(out, out_mask)
        return out, out_mask

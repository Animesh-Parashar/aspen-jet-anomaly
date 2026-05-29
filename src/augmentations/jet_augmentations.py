"""
Physics-motivated jet augmentations following JetCLR (Dillon et al., 2021).

All augmentations operate on full batches (B, N_const, F) for GPU efficiency.
Single-jet forms (N_const, F) are also supported via the _augment_one path.

Feature indices matching _to_relative() in aspen_loader.py:
  [eta_rel, phi_rel, log_pT_rel, log_E, d0, dz, charge]
"""

import torch
import numpy as np

ETA_IDX    = 0
PHI_IDX    = 1
LOG_PT_IDX = 2


# ---------------------------------------------------------------------------
# Batch-level augmentations — input/output (B, N, F) or (N, F)
# ---------------------------------------------------------------------------

def rotate_batch(x, mask, max_angle=np.pi):
    """Independent random rotation per jet in the batch."""
    is_single = x.dim() == 2
    if is_single:
        x, mask = x.unsqueeze(0), mask.unsqueeze(0)

    B = x.size(0)
    angles = torch.empty(B, device=x.device).uniform_(-max_angle, max_angle)
    cos_a  = angles.cos().view(B, 1)   # (B, 1)
    sin_a  = angles.sin().view(B, 1)

    eta = x[:, :, ETA_IDX]
    phi = x[:, :, PHI_IDX]
    out = x.clone()
    out[:, :, ETA_IDX] = cos_a * eta - sin_a * phi
    out[:, :, PHI_IDX] = sin_a * eta + cos_a * phi

    return (out.squeeze(0), mask.squeeze(0)) if is_single else (out, mask)


def translate_batch(x, mask, max_shift=0.2):
    """Independent random translation per jet."""
    is_single = x.dim() == 2
    if is_single:
        x, mask = x.unsqueeze(0), mask.unsqueeze(0)

    B = x.size(0)
    d_eta = torch.empty(B, 1, device=x.device).uniform_(-max_shift, max_shift)
    d_phi = torch.empty(B, 1, device=x.device).uniform_(-max_shift, max_shift)

    out = x.clone()
    out[:, :, ETA_IDX] = x[:, :, ETA_IDX] + d_eta
    out[:, :, PHI_IDX] = x[:, :, PHI_IDX] + d_phi

    return (out.squeeze(0), mask.squeeze(0)) if is_single else (out, mask)


def smear_pt_batch(x, mask, sigma=0.1):
    """Gaussian noise on log_pT_rel for each valid constituent."""
    out = x.clone()
    noise = torch.randn_like(out[:, :, LOG_PT_IDX]) * sigma
    # Don't smear padding tokens
    valid = ~mask  # (B, N) or (N,)
    out[..., LOG_PT_IDX] = out[..., LOG_PT_IDX] + noise * valid.float()
    return out, mask


def soft_drop_batch(x, mask, drop_prob=0.1):
    """
    Independently drop each valid constituent with probability drop_prob.
    Always keeps at least one constituent per jet.
    """
    out_x    = x.clone()
    out_mask = mask.clone()
    valid    = ~mask                               # (B, N) True = real token

    drop     = torch.rand_like(x[..., 0]) < drop_prob   # (B, N)
    to_drop  = drop & valid                        # only drop real tokens

    # Safety: if all real tokens would be dropped, keep the first one
    # Find first valid token per jet
    first_valid = valid.float().argmax(dim=-1, keepdim=True)  # (B, 1)
    safety = torch.zeros_like(to_drop)
    safety.scatter_(1, first_valid, True)
    to_drop = to_drop & ~safety

    out_mask = out_mask | to_drop
    out_x[to_drop] = 0.0
    return out_x, out_mask


def collinear_split_batch(x, mask, split_prob=0.3):
    """
    Vectorized collinear split: one split per jet, fully batch-parallel.
    With probability split_prob, splits a random valid constituent into two
    collinear daughters by filling the first available padding slot.
    No Python loop over jets — O(1) regardless of batch size.
    """
    is_single = x.dim() == 2
    if is_single:
        x, mask = x.unsqueeze(0), mask.unsqueeze(0)

    B, N, _ = x.shape
    valid = ~mask                                              # (B, N) True=real

    # Jets that are eligible AND randomly selected to split
    has_pad   = mask.any(dim=1)                               # (B,)
    has_valid = valid.any(dim=1)                              # (B,)
    do_split  = (torch.rand(B, device=x.device) < split_prob) & has_valid & has_pad

    if not do_split.any():
        return (x.squeeze(0), mask.squeeze(0)) if is_single else (x, mask)

    # Random valid source constituent per jet (noise=-1 at padding → never picked)
    noise = torch.rand(B, N, device=x.device)
    noise.masked_fill_(mask, -1.0)
    src_idx = noise.argmax(dim=1)                             # (B,) — random valid

    # First free padding slot per jet
    dst_idx = mask.float().argmax(dim=1)                      # (B,) — first pad slot

    # Split fractions
    z       = torch.empty(B, device=x.device).uniform_(0.2, 0.8)
    log_z   = torch.log(z + 1e-8)
    log_1mz = torch.log(1.0 - z + 1e-8)

    # Restrict to jets that actually split
    j  = torch.where(do_split)[0]                            # (S,)
    si = src_idx[j]                                          # (S,) source positions
    di = dst_idx[j]                                          # (S,) dest positions

    out_x    = x.clone()
    out_mask = mask.clone()

    # Copy source features to destination slot
    out_x[j, di] = out_x[j, si]

    # Apply momentum split additively in log space
    out_x[j, si, LOG_PT_IDX] = out_x[j, si, LOG_PT_IDX] + log_z[j]
    out_x[j, di, LOG_PT_IDX] = out_x[j, di, LOG_PT_IDX] + log_1mz[j]

    # Small angular displacement for daughter
    out_x[j, di, ETA_IDX] = out_x[j, di, ETA_IDX] + 0.05
    out_x[j, di, PHI_IDX] = out_x[j, di, PHI_IDX] - 0.05

    # Unmask destination
    out_mask[j, di] = False

    if is_single:
        return out_x.squeeze(0), out_mask.squeeze(0)
    return out_x, out_mask


# ---------------------------------------------------------------------------
# Composed augmentation class
# ---------------------------------------------------------------------------

class JetAugmentation:
    """
    Applies stochastic augmentations to a batch and returns two independent views.
    Operates on (B, N_const, F) tensors; also accepts single (N_const, F).
    """

    def __init__(self, rotate_p=1.0, translate_p=1.0, smear_pt_p=0.5,
                 soft_drop_p=0.5, collinear_split_p=0.3):
        self.aug_fns = []
        if rotate_p         > 0: self.aug_fns.append((rotate_p,         rotate_batch))
        if translate_p      > 0: self.aug_fns.append((translate_p,      translate_batch))
        if smear_pt_p       > 0: self.aug_fns.append((smear_pt_p,       smear_pt_batch))
        if soft_drop_p      > 0: self.aug_fns.append((soft_drop_p,      soft_drop_batch))
        if collinear_split_p> 0: self.aug_fns.append((collinear_split_p,collinear_split_batch))

    def _augment(self, x, mask):
        for prob, fn in self.aug_fns:
            if torch.rand(1).item() < prob:
                x, mask = fn(x, mask)
        return x, mask

    def __call__(self, x, mask):
        """Return two independently augmented views of the same batch."""
        x1, m1 = self._augment(x.clone(), mask.clone())
        x2, m2 = self._augment(x.clone(), mask.clone())
        return (x1, m1), (x2, m2)

"""Core4 augmentations only; approximate jet-plane symmetry, not detector symmetry.

No random removal, artificial charge splitting, translation of a centered axis,
 or independent noise on redundant momentum features. Splitting is optional:
exactly collinear in these coordinates, conserving represented scalar pT and
momentum fractions; it is not a model of detector particle fragmentation.
"""
import math
import torch


def rotate(x, mask, angles):
    out = x.clone()
    c, s = angles.cos()[:, None], angles.sin()[:, None]
    out[..., 0] = c*x[..., 0] - s*x[..., 1]
    out[..., 1] = s*x[..., 0] + c*x[..., 1]
    # Wrapping would break planar distance preservation: reject outside chart.
    valid = (~mask) & (out[..., 1].abs() > math.pi)
    out = torch.where(valid.any(dim=1)[:, None, None], x, out)
    return out.masked_fill(mask[..., None], 0), mask.clone()


def split_collinear(x, mask, probability=0.3):
    if x.shape[-1] != 4:
        raise ValueError('Splitting supports core4 only, never detector features')
    out, out_mask = x.clone(), mask.clone()
    b, n, _ = x.shape
    eligible = mask.any(1) & (~mask).any(1) & (torch.rand(b, device=x.device) < probability)
    rows = torch.where(eligible)[0]
    if not len(rows):
        return out, out_mask
    src = torch.rand(b, n, device=x.device).masked_fill(mask, -1).argmax(1)[rows]
    dst = mask.long().argmax(1)[rows]
    z = torch.empty(len(rows), device=x.device).uniform_(0.2, 0.8)
    parent = x[rows, src].clone()
    out[rows, dst] = parent
    out[rows, src, 2:4] = parent[:, 2:4] + z.log()[:, None]
    out[rows, dst, 2:4] = parent[:, 2:4] + (1-z).log()[:, None]
    out_mask[rows, dst] = False
    # Restore pT ordering and trailing padding, required by downstream consumers.
    order = out[..., 3].masked_fill(out_mask, -torch.inf).argsort(dim=1, descending=True)
    out = out.gather(1, order[..., None].expand_as(out))
    out_mask = out_mask.gather(1, order)
    return out.masked_fill(out_mask[..., None], 0), out_mask


class CoreAugmentation:
    def __init__(self, split_probability=0.):
        if not 0 <= split_probability <= 1:
            raise ValueError('Invalid split probability')
        self.split_probability = split_probability

    def view(self, x, mask):
        if x.ndim != 3 or x.shape[-1] != 4 or mask.dtype != torch.bool:
            raise ValueError('Expected core4 batch and boolean mask')
        angles = torch.empty(len(x), device=x.device).uniform_(-math.pi, math.pi)
        out, out_mask = rotate(x, mask, angles)
        if self.split_probability:
            out, out_mask = split_collinear(out, out_mask, self.split_probability)
        return out, out_mask

    def __call__(self, x, mask):
        return self.view(x, mask), self.view(x, mask)

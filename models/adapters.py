"""Per-disease adaptation surface: FiLM affine + quantile head, and the pinball loss."""
from __future__ import annotations

import torch
import torch.nn as nn

from bundles import HORIZONS
from .config import D_HIDDEN, QUANTILES


class Adapter(nn.Module):
    """FiLM (gamma,beta) + quantile head. 1,428 params at d=64, |H|=4, |Q|=5.

    With the trunk frozen this is EXACTLY an affine map of the frozen features (gamma folds into the
    head weight, verified to 0.0 deviation), so FiLM adds no expressive power over a plain linear
    head and the inner loop is ANIL in the exact sense."""
    def __init__(self, d=D_HIDDEN, horizons=HORIZONS, quantiles=QUANTILES):
        super().__init__()
        self.nH, self.nQ = len(horizons), len(quantiles)
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.head = nn.Linear(d, self.nH * self.nQ)           # direct multi-horizon, one shot (C5)
        # ponytail: plain Linear can emit crossing quantiles (q05>q95). The median (point forecast)
        # is unaffected; only intervals are. Decision #6: sort the 5 quantiles post-hoc at Week-5
        # inference, before PICP/CRPS -- do NOT reparametrise the trunk. Upgrade path if post-hoc
        # sort proves insufficient: monotone cumulative-softplus head (forces a full dev re-run).

    def forward(self, h):
        h = self.gamma * h + self.beta
        return self.head(h).view(h.shape[0], self.nH, self.nQ)   # [N, |H|, |Q|]


def pinball_loss(pred, target, mask, quantiles=QUANTILES, w=None):
    """pred [N,H,Q], target [N,H], mask [N,H] (1=observed & in phase), MODEL space.

    Optional w [N] is the joint trainer's per-node weight; w=None is the plain mean, so
    single-disease callers stay bit-identical."""
    q = torch.tensor(quantiles, device=pred.device).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * err, (q - 1) * err).mean(-1)     # avg over quantiles -> [N,H]
    m = mask if w is None else mask * w.view(-1, 1)           # fold node weight into the mask
    return (loss * m).sum() / m.sum().clamp(min=1)

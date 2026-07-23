"""Per-disease adaptation surface (plan §3.5, P5): FiLM affine + quantile head, plus the pinball
loss and the shared/adaptation parameter split Week-4's MAML inner loop depends on."""
from __future__ import annotations

import torch
import torch.nn as nn

from bundles import HORIZONS
from .config import D_HIDDEN, QUANTILES


class Adapter(nn.Module):
    """FiLM (gamma,beta) + quantile head -- the ONLY thing Week-4's MAML inner loop touches; the
    trunk is the outer loop. ~388 params, sized against Ebola's 27 support cells."""
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
    """pred [N,H,Q], target [N,H], mask [N,H] (1=observed & in phase). MODEL space (plan §5).

    Optional w [N]: per-node weight for the joint trainer's within-dataset balance (e.g. dengue
    per-cell country balance). w=None is the plain mean, so single-disease callers are bit-identical.
    """
    q = torch.tensor(quantiles, device=pred.device).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * err, (q - 1) * err).mean(-1)     # avg over quantiles -> [N,H]
    m = mask if w is None else mask * w.view(-1, 1)           # fold node weight into the mask
    return (loss * m).sum() / m.sum().clamp(min=1)


def shared_params(enc):
    return list(enc.parameters())


def adaptation_params(adapter):
    return list(adapter.parameters())

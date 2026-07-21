"""Windowing (plan §6): an origin t -> input slice + per-horizon targets/masks. No instance norm
(plan §3.1 dropped per §0.9 -- the released scaler is already per-node z-score)."""
from __future__ import annotations

import torch

from bundles import HORIZONS, W


def window_slice(Z, t):
    """Input window for origin t: Z[:, t-w+1 : t+1, :] -> [N, w, F]."""
    return Z[:, t - (W - 1):t + 1, :]


def targets_and_mask(ymod, Mt, phase_mask, t, device=None):
    """For origin t: targets [N,H] (model space) and mask [N,H] (observed AND in-phase at t+h). An
    example's phase is decided by where its TARGET lands, never by the input window (Task 11.3)."""
    tgt = torch.stack([ymod[:, t + h] for h in HORIZONS], dim=1)
    msk = torch.stack([(Mt[:, t + h] * phase_mask[:, t + h]) for h in HORIZONS], dim=1)
    if device is not None:
        tgt, msk = tgt.to(device), msk.to(device)
    return tgt, msk

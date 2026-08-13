"""Windowing: an origin t -> input slice + per-horizon targets/masks. No instance norm, since the
released scaler is already a per-node z-score."""
from __future__ import annotations

import torch

from bundles import HORIZONS, W


def window_slice(Z, t):
    """Input window for origin t: Z[:, t-w+1 : t+1, :] -> [N, w, F].

    For t < W-1 the window is ZERO LEFT-PADDED to length W, needed by the Ebola few-shot arms. The
    short case is handled explicitly because a negative start index would slice from the TAIL, which
    is a silently wrong window. Zero is right for the incidence channel: the bundles are per-node
    z-scores encoding an unobserved cell as 0, so a padded step reads as "nothing known".

    ponytail: sin_doy/cos_doy are zero-padded too, and (0,0) is not on the unit circle, so a padded
    step carries an impossible calendar date. At h10 on the 12-week arm a window is ~89% pad.
    Upgrade path: back-rotate sin/cos by the weekly angle instead of zeroing them."""
    assert t >= 0, f"origin {t} < 0"
    if t >= W - 1:
        return Z[:, t - (W - 1):t + 1, :]
    pad = Z.new_zeros((Z.shape[0], W - 1 - t, Z.shape[2]))
    return torch.cat([pad, Z[:, :t + 1, :]], dim=1)


def targets_and_mask(ymod, Mt, phase_mask, t, device=None):
    """For origin t: targets [N,H] (model space) and mask [N,H] (observed AND in-phase at t+h). An
    example's phase is decided by where its TARGET lands, never by the input window (Task 11.3)."""
    tgt = torch.stack([ymod[:, t + h] for h in HORIZONS], dim=1)
    msk = torch.stack([(Mt[:, t + h] * phase_mask[:, t + h]) for h in HORIZONS], dim=1)
    if device is not None:
        tgt, msk = tgt.to(device), msk.to(device)
    return tgt, msk

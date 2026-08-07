"""Windowing (plan §6): an origin t -> input slice + per-horizon targets/masks. No instance norm
(plan §3.1 dropped per §0.9 -- the released scaler is already per-node z-score)."""
from __future__ import annotations

import torch

from bundles import HORIZONS, W


def window_slice(Z, t):
    """Input window for origin t: Z[:, t-w+1 : t+1, :] -> [N, w, F].

    For t < W-1 the window is ZERO LEFT-PADDED to length W (P7 short-window path, needed by the
    Ebola few-shot arms: their support sets sit at columns 0..12 and 0..20, so most adaptation
    origins have no full history). A negative start index would slice from the TAIL, which is a
    silently wrong window, so the short case is handled explicitly rather than by arithmetic.

    Zero is the right pad value for the incidence channel: the bundles normalise to per-node
    z-scores and encode an unobserved cell as 0, i.e. the per-node mean, so a padded step reads
    as "nothing known" on exactly the convention the encoder was trained with.

    ponytail: sin_doy/cos_doy are padded with zeros too, which is (0,0) -- not a point on the unit
    circle, so a padded step carries an impossible calendar date. That is what the Day-13 protocol
    recorded ("left-pad short windows with zeros") and obs_mask flags the step either way. The
    ceiling: at h10 on the 12-week arm a window is ~89% pad, so the encoder is reading mostly
    impossible dates. Upgrade path if that turns out to matter: back-rotate sin/cos by the weekly
    angle instead of zeroing them, which keeps the pad in-distribution and needs no extra inputs.
    """
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

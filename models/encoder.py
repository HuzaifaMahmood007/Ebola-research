"""SharedEncoder: TCN + LTR degree feature, then gated inductive spatial mixing.

forward(Z, A, M_t) -> h [N, d]. No disease name and no parameter dimension sized by N (C1/C2)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import D_HIDDEN, NODE_COUNTS
from .spatial import LTR, SpatialMixer, mask_aware_adj, normalise_adj
from .temporal import DilatedTCN


class Gate(nn.Module):
    """h_out = (1-g)*h + g*h_spatial, g = sigmoid(MLP(h)) per node: data-dependent, N-independent.

    mode 'off' forces g=0, the graph-free model used as the {g==0} ablation cell."""
    def __init__(self, d=D_HIDDEN, mode="learned"):
        super().__init__()
        assert mode in {"learned", "off"}
        self.mode = mode
        if mode == "learned":
            self.mlp = nn.Sequential(nn.Linear(d, d // 4), nn.GELU(), nn.Linear(d // 4, 1))

    def g(self, h):
        if self.mode == "off":
            return torch.zeros(h.shape[0], 1, device=h.device)
        return torch.sigmoid(self.mlp(h))

    def forward(self, h, h_spatial):
        g = self.g(h)
        return (1 - g) * h + g * h_spatial, g


def spatial_contribution(g, h, h_s):
    """The reportable gate quantity: scale-free, unlike raw g, which stays debug-only."""
    gs, hn, hsn = g.squeeze(-1), h.norm(dim=-1), h_s.norm(dim=-1)
    return (gs * hsn) / ((1 - gs) * hn + gs * hsn + 1e-8)


def node_indexed_params(module: nn.Module):
    """C2 audit: params whose shape touches a graph size. Empty list = clean."""
    return [(n, tuple(p.shape)) for n, p in module.named_parameters()
            if any(d in NODE_COUNTS for d in p.shape)]


class SharedEncoder(nn.Module):
    def __init__(self, d=D_HIDDEN, gate_mode="learned"):
        super().__init__()
        self.tcn = DilatedTCN(in_ch=4, d=d)
        self.ltr = LTR(d)
        self.spatial = SpatialMixer(d)
        self.gate = Gate(d, mode=gate_mode)

    def forward(self, Z, A, M_t=None):
        assert Z.shape[-1] == 4, f"shared trunk takes exactly 4 core channels, got {Z.shape[-1]} (C1)"
        A_hat, deg = normalise_adj(A)                         # one normalisation; deg reused by LTR
        assert (deg > 0).all(), "a node has degree 0 -- identity not added before normalisation (C3)"
        assert torch.isfinite(A_hat.values()).all(), "A_hat has NaN/Inf (C3)"
        h = self.tcn(Z) + self.ltr(deg)                       # temporal + degree feature
        A_mix = mask_aware_adj(A, M_t) if M_t is not None else A_hat
        h_s = self.spatial(h, A_mix)
        h_out, self.last_g = self.gate(h, h_s)
        self.last_h, self.last_h_s = h, h_s                   # kept for spatial_contribution() logging
        return h_out

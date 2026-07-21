"""Temporal encoder (plan §3.2): dilated causal TCN, WaveNet block with the adaptive-adjacency
machinery removed. Nodes go into the batch dimension, so it is node-count-agnostic for free (C2).
The receptive field is asserted at construction (Correction B) -- a §8 gate, not a comment."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bundles import W
from .config import D_HIDDEN, DILATIONS, DROPOUT, KERNEL


class DilatedTCN(nn.Module):
    def __init__(self, in_ch=4, d=D_HIDDEN, dilations=DILATIONS, kernel=KERNEL, dropout=DROPOUT):
        super().__init__()
        rf = 1 + sum((kernel - 1) * dl for dl in dilations)
        assert rf >= W, f"TCN receptive field {rf} < lookback w={W} (Correction B)"
        self.rf, self.kernel, self.dilations = rf, kernel, dilations
        self.inp = nn.Conv1d(in_ch, d, 1)
        self.filt = nn.ModuleList(nn.Conv1d(d, d, kernel, dilation=dl) for dl in dilations)
        self.gate = nn.ModuleList(nn.Conv1d(d, d, kernel, dilation=dl) for dl in dilations)
        self.skip = nn.ModuleList(nn.Conv1d(d, d, 1) for _ in dilations)
        self.res = nn.ModuleList(nn.Conv1d(d, d, 1) for _ in dilations)
        self.drop = nn.Dropout(dropout)

    def forward(self, Z):                                    # Z [N, T=20, 4]
        x = self.inp(Z.transpose(1, 2))                       # [N, d, T]
        skip_sum = 0
        for filt, gate, skip, res, dl in zip(self.filt, self.gate, self.skip, self.res, self.dilations):
            xp = F.pad(x, ((self.kernel - 1) * dl, 0))         # left-pad only => causal
            z = self.drop(torch.tanh(filt(xp)) * torch.sigmoid(gate(xp)))
            skip_sum = skip_sum + skip(z)
            x = x + res(z)                                     # residual
        return F.gelu(skip_sum)[:, :, -1]                      # last timestep [N, d]


# [OPEN] plan §3.2: a 1-layer GRU (d=64) is the drop-in ablation alternative (--temporal gru).
# Not built -- YAGNI until the ablation runs (Week 6); the interface is DilatedTCN.forward(Z)->[N,d].

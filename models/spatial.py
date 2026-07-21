"""Spatial channel (plan §3.3): graph normalisation (identity-first, C3), mask-aware adjacency,
inductive GraphSAGE-style message passing, and the LTR degree feature (guide Task 12.1a). Sparse
throughout -- dengue's 7,165^2 adjacency is never densified (§0.7)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import D_HIDDEN, SPATIAL_LAYERS


def sparse_from_dense_np(A_np: np.ndarray) -> torch.Tensor:
    """Dense numpy adjacency -> torch sparse COO, without ever building a dense torch [N,N]."""
    ii, jj = np.nonzero(A_np)
    idx = torch.tensor(np.stack([ii, jj]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(A_np[ii, jj], dtype=torch.float32), A_np.shape).coalesce()


def normalise_adj(A: torch.Tensor, add_self_loops: bool = True):
    """A_hat = D~^{-1/2}(A + I)D~^{-1/2}, identity FIRST (C3). Returns (A_hat sparse, deg[N]); deg is
    the self-loop-inclusive D~ that LTR reuses. add_self_loops=False is the negative control:
    a degree-0 node then has D~=0 (dense: inf*0 = NaN)."""
    N = A.shape[0]
    A = A.coalesce() if A.is_sparse else A.to_sparse().coalesce()
    idx, val = A.indices(), A.values().float()
    if add_self_loops:
        sl = torch.arange(N, device=A.device)
        idx = torch.cat([idx, torch.stack([sl, sl])], dim=1)
        val = torch.cat([val, torch.ones(N, device=A.device)])
    deg = torch.zeros(N, device=A.device).scatter_add_(0, idx[0], val)
    dinv = deg.pow(-0.5)
    A_hat = torch.sparse_coo_tensor(idx, dinv[idx[0]] * val * dinv[idx[1]], (N, N)).coalesce()
    return A_hat, deg


def mask_aware_adj(A: torch.Tensor, M_t: torch.Tensor):
    """Adjacency for one origin, smoothing over OBSERVED neighbours only (plan §3.3): a real edge
    (i,j) is weighted by M_t[j]; the self-loop is always kept so a node with no observed neighbour
    falls back to itself. Then D~^{-1/2}-renormalised."""
    N = A.shape[0]
    A = A.coalesce() if A.is_sparse else A.to_sparse().coalesce()
    idx = A.indices()
    val = A.values().float() * M_t[idx[1]].float()          # weight each neighbour by its own obs mask
    sl = torch.arange(N, device=A.device)
    idx = torch.cat([idx, torch.stack([sl, sl])], dim=1)
    val = torch.cat([val, torch.ones(N, device=A.device)])   # self-loop always present
    deg = torch.zeros(N, device=A.device).scatter_add_(0, idx[0], val)
    dinv = deg.pow(-0.5)
    return torch.sparse_coo_tensor(idx, dinv[idx[0]] * val * dinv[idx[1]], (N, N)).coalesce()


class LTR(nn.Module):
    """Node degree as a learned feature (guide Task 12.1a). forward takes the D~ from normalise_adj
    and never sees A -- so it structurally cannot recompute the degree (the 'reuse' guarantee)."""
    def __init__(self, d=D_HIDDEN):
        super().__init__()
        self.lin = nn.Linear(1, d)

    def forward(self, deg):                                  # deg [N]  (self-loop-inclusive D~)
        return self.lin(torch.log1p(deg).unsqueeze(-1))      # log1p keeps degree ~unit vs the temporal features


class SpatialMixer(nn.Module):
    """GraphSAGE-style inductive message passing; separate self/neighbour weights, both d x d (no N)."""
    def __init__(self, d=D_HIDDEN, layers=SPATIAL_LAYERS):
        super().__init__()
        self.self_w = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.neigh_w = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))

    def forward(self, h, A_hat):                             # h [N,d], A_hat sparse [N,N]
        for sw, nw in zip(self.self_w, self.neigh_w):
            h = F.gelu(sw(h) + nw(torch.sparse.mm(A_hat, h))) + h    # observed-neighbour aggregate + residual
        return h

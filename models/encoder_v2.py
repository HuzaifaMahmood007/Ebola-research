"""V2 encoder: v1's path unchanged, plus a DEVIATION branch. A mechanism test, not a replacement.

Protocol: progress/decisions/V2_Deviation_Protocol.md. Runner: ablation/run_v2_deviation.py.

    h = v1(Z, A, M_t) + P(mix_dev(TCN_dev([dev, obs])))

v1(Z, A, M_t) is SharedEncoder.forward exactly as released: TCN + LTR(deg), its SpatialMixer, its gate.
The deviation branch is ADDED on top:
  dev[i,k]  = (x[i,k] - mean of x[j,k] over OBSERVED j in i's country, same week k) * obs[i,k]
  TCN_dev   = DilatedTCN(in_ch=2, d=D_DEV, dropout=0), inputs [dev, obs_mask], convs computed as
              matmuls rather than cuDNN so the branch is deterministic (see _MatmulConv1d)
  mix_dev   = its OWN SpatialMixer(D_DEV) over one of three adjacencies, which is the arm:
                real      the same adjacency v1 mixes over             (v2-graph)
                none      the identity, so a district sees only itself  (v2-nograph)
                shuffled  a degree-preserving relabel, bound per panel  (v2-shuffled)
  P         = Linear(D_DEV, D_HIDDEN), ZERO-initialised

Why add through a zero-initialised projection: v2 starts exactly at v1 and can only move away from it
if the deviation branch lowers the training loss, so v2 is a strict superset of v1 by construction
(checked in ablation/run_v2_deviation.py --selfcheck, forward and training).

Pairing with v1 at the same seed. The v1 submodules are built first, drawing the global RNG exactly as
SharedEncoder does; the branch is built under a forked CPU RNG, so the Adapter built after the encoder
gets v1's init too. TCN_dev runs with dropout 0, which draws no random numbers, so v1's dropout masks
are unchanged. Result: v2 at the same seed starts from v1's weights, sees the same origin order and the
same dropout masks, and differs only by what the branch learns.

C1: Z still has exactly 4 channels (asserted by v1's forward); dev is derived from channels 0 and 3.
C2: no parameter is sized by N (asserted at construction). The country index is an INPUT, bound per
    panel or passed to forward, never a parameter or a buffer.
C3: every adjacency the branch mixes over carries a self-loop (mask_aware_adj / normalise_adj / I).
No future leak: dev at week k reads only week k, and the window ends at the origin (checked).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import D_HIDDEN
from .encoder import SharedEncoder, node_indexed_params
from .spatial import SpatialMixer, mask_aware_adj, normalise_adj
from .temporal import DilatedTCN

D_DEV = 16                                   # protocol-fixed branch width, not tuned
DEV_GRAPHS = ("real", "none", "shuffled")
DEV_INIT_OFFSET = 1_000_003                  # branch init seed = torch.initial_seed() + this
INCIDENCE, OBS = 0, 3                        # core channel indices (bundles meta core_feature_idx)


def group_index(node_ids, group_map):
    """Per-node country index [N] (ints) and its labels, from bundle.group_of(). An input, not a param."""
    labels = sorted({group_map[n] for n in node_ids})
    pos = {g: i for i, g in enumerate(labels)}
    return [pos[group_map[n]] for n in node_ids], labels


def deviation(Z, group):
    """dev [N, W]: each district minus the mask-aware mean of OBSERVED districts in its country, SAME week.

    Reductions run over the node axis only, one week column at a time, so week k never reads another
    week. Unobserved cells are set to 0, the z-space convention for "nothing known". A country with no
    observed district in a week has mean 0 and every dev in it is 0.

    The country sums are a one-hot matrix product, not index_add_: on CUDA index_add_ adds floats with
    atomics in a varying order, which made two runs at the same seed differ. The matmul is deterministic."""
    x, obs = Z[..., INCIDENCE], Z[..., OBS]
    G = int(group.max()) + 1
    onehot = x.new_zeros(G, x.shape[0]).scatter_(0, group.view(1, -1), 1.0)   # [G, N], an input, not a param
    tot, cnt = onehot @ (x * obs), onehot @ obs
    return (x - (tot / cnt.clamp(min=1))[group]) * obs


class _MatmulConv1d(nn.Conv1d):
    """Conv1d (stride 1, no padding, groups 1) computed as sliced matmuls instead of through cuDNN.

    Same parameters, same maths. Why: at the branch's 16-channel shapes cuDNN's default weight-gradient
    kernels are nondeterministic (measured: the dilated filt/gate weight gradients varied across
    repeats), so two runs at one seed drifted apart. Turning cudnn.deterministic on globally would fix
    the branch but also moves v1's own convolutions (measured: 0.15 percent after 3 epochs), breaking the
    pairing with the released v1 records. So only the branch leaves cuDNN."""
    def forward(self, x):                                    # x [N, C, T]
        k, dl = self.kernel_size[0], self.dilation[0]
        T = x.shape[-1] - dl * (k - 1)
        y = sum(torch.einsum("oc,nct->not", self.weight[:, :, j], x[:, :, j * dl:j * dl + T]) for j in range(k))
        return y + self.bias.view(1, -1, 1)


def _without_cudnn(module):
    """Swap every nn.Conv1d in `module` to _MatmulConv1d in place. Parameters, names and init unchanged."""
    for m in module.modules():
        if type(m) is nn.Conv1d:
            assert m.stride == (1,) and m.padding == (0,) and m.groups == 1, "only plain causal convs"
            m.__class__ = _MatmulConv1d
    return module


def _identity(N, device):
    i = torch.arange(N, device=device)
    return torch.sparse_coo_tensor(torch.stack([i, i]), torch.ones(N, device=device), (N, N)).coalesce()


class DeviationEncoder(SharedEncoder):
    def __init__(self, d=D_HIDDEN, gate_mode="learned", d_dev=D_DEV, dev_graph="real", dev_gain=1.0):
        super().__init__(d=d, gate_mode=gate_mode)           # v1 modules, same RNG draws as SharedEncoder
        assert dev_graph in DEV_GRAPHS, f"dev_graph must be one of {DEV_GRAPHS}, got {dev_graph!r}"
        self.dev_graph, self.d_dev, self.dev_gain = dev_graph, d_dev, float(dev_gain)
        # CPU generator only. torch.manual_seed would also reseed CUDA, which fork_rng(devices=[]) does
        # not restore, and that would change v1's dropout masks on the GPU.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(torch.initial_seed() + DEV_INIT_OFFSET)
            self.dev_tcn = _without_cudnn(DilatedTCN(in_ch=2, d=d_dev, dropout=0.0))
            self.dev_mix = SpatialMixer(d_dev)
            self.dev_proj = nn.Linear(d_dev, d)
        nn.init.zeros_(self.dev_proj.weight)
        nn.init.zeros_(self.dev_proj.bias)
        bad = node_indexed_params(self)
        assert not bad, f"C2: v2 has parameters sized by a graph size: {bad}"
        self.group, self.A_dev = None, None

    def bind(self, group=None, A_dev=None):
        """Attach the per-panel INPUTS: country index [N] and, for the shuffled arm, the relabelled
        adjacency. Plain attributes, never parameters or buffers, so nothing here is saved or trained."""
        self.group, self.A_dev = group, A_dev
        return self

    def branch_params(self):
        return [p for n, p in self.named_parameters() if n.startswith("dev_")]

    def forward(self, Z, A, M_t=None, group=None, A_dev=None):
        h_v1 = super().forward(Z, A, M_t)                     # v1 path, untouched; asserts C1 and C3
        N = Z.shape[0]
        group = self.group if group is None else group
        group = torch.zeros(N, dtype=torch.long, device=Z.device) if group is None else group
        assert group.shape == (N,) and group.dtype == torch.long, "country index must be long [N]"
        dev = deviation(Z, group)
        hd = self.dev_tcn(torch.stack([dev, Z[..., OBS]], dim=-1))          # [N, d_dev]
        if self.dev_graph == "none":
            A_mix = _identity(N, Z.device)
        else:
            A_src = A_dev if A_dev is not None else self.A_dev
            if self.dev_graph == "real":
                assert A_src is None, "the real arm mixes over v1's own adjacency; do not bind another"
                A_src = A
            assert A_src is not None, "the shuffled arm needs its relabelled adjacency bound first"
            A_mix = mask_aware_adj(A_src, M_t) if M_t is not None else normalise_adj(A_src)[0]
        assert torch.isfinite(A_mix.values()).all(), "deviation-branch adjacency has NaN/Inf (C3)"
        hd_s = self.dev_mix(hd, A_mix)
        self.last_v1, self.last_dev_out = h_v1, self.dev_proj(hd_s) * self.dev_gain
        return h_v1 + self.last_dev_out

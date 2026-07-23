"""train/joint.py -- multi-disease joint training over one block-diagonal supergraph (Day 14, G2).

Replaces sequential per-dataset gradient accumulation with a single forward over a block-diagonal
supergraph: the 4 dev datasets are stacked along the node axis ([SigmaN=7271, 20, 4]) with a
block-diagonal adjacency that has NO cross-dataset edges. One forward, one backward, one step.

Verified precondition (why this is safe): nothing in the encoder reduces over the node dimension --
the TCN treats nodes as the conv batch dim, spatial message passing follows edges only, and there is
no BatchNorm. So each block's math is identical whether run alone or inside the supergraph
(_equiv_check asserts this: block == solo). No cross-block edges => no inter-dataset leakage.

Samplers as loss weights (not sampling), two orthogonal vectors:
  * ACROSS datasets  w_i : uniform (P2 primary) | proportional | sqrt, from TRAIN n_obs. uniform
    hands each dataset 1/4 (dengue's 98.5% node share -> 25%); proportional = the dengue-dominated
    pooled mean; sqrt = the Week-6 middle ground (dengue ~81%).
  * WITHIN dengue  v_node : uniform | country -- per-CELL country balance (v proportional to
    1/train_obs_c) so each of dengue's 12 countries contributes equal mass. Per-cell not per-node
    because obs density spans 97x across countries; per-node would NOT actually balance them.
The two compose cleanly: v balances inside a dataset (L_i stays a proper mean), w balances across.
uniform+uniform is bit-identical to the pre-weighting loss (mean of the 4 per-dataset means).

Training is in STEPS, not epochs -- "epoch" is ill-defined when 4 datasets contribute at a fixed
ratio every step (one pass over dengue is ~24 passes over japan). Val every val_every steps,
patience counted in val-checks.

Architecture: ONE shared encoder (the transferable trunk) + one small Adapter per dataset (FiLM +
head, ~388 params each, P5). A new disease = one new adapter few-shot-fit with the trunk frozen.

Run from the repo root as a module:
  python -m train.joint --all                              # 5 seeds, uniform-uniform (primary)
  python -m train.joint --all --sampler sqrt --dengue-balance country   # a Week-6 ablation cell
  python -m train.joint --seed 42                          # one joint run, all 4 datasets scored
  python -m train.joint --smoke                            # 3 small datasets, few steps (CI-cheap)
  python -m train.joint --equiv                            # gates: block==solo, routing, balance
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from bundles import DEV_BUNDLE_NAMES, HORIZONS, W, load
from models import (MEDIAN_IDX, Adapter, SharedEncoder, pinball_loss, sparse_from_dense_np,
                    targets_and_mask, window_slice)
from to_schema import invert_scaler
from train.loop import (DEVICE, RESULTS, SEEDS, _round_trip_ok, gate_spatial_readout,
                        score_predictions, write_gate, write_per_node, write_per_origin,
                        write_records)


def block_diag_sparse(A_list, device):
    """Block-diagonal sparse COO from a list of dense numpy adjacencies -- each block offset by the
    running node count, so there is never a nonzero off any block. Never builds a dense [SigmaN,N]."""
    ii, jj, vv, off = [], [], [], 0
    for A in A_list:
        r, c = np.nonzero(A)
        ii.append(r + off); jj.append(c + off); vv.append(A[r, c]); off += A.shape[0]
    idx = torch.tensor(np.stack([np.concatenate(ii), np.concatenate(jj)]), dtype=torch.long)
    val = torch.tensor(np.concatenate(vv), dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (off, off)).coalesce().to(device)


def _prepare(names, device):
    """Load each dataset once, put its tensors on device, and record its block [start:end)."""
    ds, A_list, off = [], [], 0
    for name in names:
        b = load(name)
        assert _round_trip_ok(b), f"{name}: scaler round-trip failed -- transforms mis-ordered"
        N = b.A_geo.shape[0]
        m = b.masks()
        ds.append(SimpleNamespace(
            name=name, b=b, N=N, start=off, end=off + N,
            Z=torch.tensor(b.transfer_view(), dtype=torch.float32, device=device),   # [N,T,4]
            ymod=torch.tensor(b.y, dtype=torch.float32, device=device),               # [N,T]
            Mt=torch.tensor(b.M, dtype=torch.float32, device=device),                 # [N,T]
            A_solo=sparse_from_dense_np(b.A_geo).to(device),                          # val/test graph
            tr=b.origins(phase="train"), va=b.origins(phase="val"), te=b.origins(phase="test"),
            mtr=torch.tensor(m["train"], dtype=torch.float32, device=device),
            mva=torch.tensor(m["val"], dtype=torch.float32, device=device),
        ))
        A_list.append(b.A_geo); off += N
    return ds, block_diag_sparse(A_list, device)


# --------------------------------------------------------------------------- #
# The two weight vectors.
# --------------------------------------------------------------------------- #
def dataset_weights(ds, scheme):
    """Across-dataset weights w_i (sum 1) from TRAIN n_obs. uniform=P2 primary; proportional=the
    dengue-dominated pooled-mean endpoint; sqrt=middle ground (Week-6 ablation)."""
    assert scheme in {"uniform", "proportional", "sqrt"}
    if scheme == "uniform":
        w = np.ones(len(ds))
    else:
        size = np.array([float(d.mtr.sum().item()) for d in ds])          # train observed cells
        w = size ** (1.0 if scheme == "proportional" else 0.5)
    return (w / w.sum()).tolist()


def per_cell_country_weight(b):
    """dengue within-dataset country balance: node weight v proportional to 1/(train obs cells in
    its country), so every country contributes equal mass to the weighted mean. Per-CELL (not
    per-node) because obs density spans 97x across the 12 countries -- per-node would leave that
    97x imbalance. Train cells only (the weight must not peek at val/test observation density)."""
    g, ids = b.group_of(), b.meta["node_ids"]
    countries = np.array([g[i] for i in ids])
    obs_per_node = b.masks()["train"].astype(bool).sum(1).astype(np.float64)   # [N]
    v = np.zeros(len(ids))
    for c in np.unique(countries):
        m = countries == c
        v[m] = 1.0 / max(obs_per_node[m].sum(), 1.0)
    return (v / v.mean()).astype(np.float32)     # mean-1 scale (scale-free; keeps float32 comfy)


def _node_weight(name, b, mode, device):
    """Within-dataset per-node weight. Only dengue differs; the 3 influenza sets are single-country
    so 'country' balance == uniform for them (returns None -> plain mean)."""
    assert mode in {"uniform", "country"}
    if name != "dengue" or mode == "uniform":
        return None
    return torch.tensor(per_cell_country_weight(b), dtype=torch.float32, device=device)


def _origin_stream(origins, seed):
    """Infinite origin stream, reshuffled each pass -- even coverage, no with-replacement variance.
    Own RNG so it doesn't consume the global torch/numpy stream (model init, dropout)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(origins)
    while True:
        for t in rng.permutation(arr):
            yield int(t)


@torch.no_grad()
def _val_pinball(enc, adapters, ds, node_w, w_base):
    """Weighted val objective mirroring training (within-dataset node weights + across-dataset w_i),
    per dataset on its SOLO graph (== its block, by _equiv_check). Selection optimises what we train."""
    enc.eval()
    tot = 0.0
    for wi, ad, d, nw in zip(w_base, adapters, ds, node_w):
        ad.eval()
        s, n = 0.0, 0
        for t in d.va:
            tgt, msk = targets_and_mask(d.ymod, d.Mt, d.mva, t, DEVICE)
            if msk.sum() == 0:
                continue
            s += float(pinball_loss(ad(enc(window_slice(d.Z, t), d.A_solo, d.Mt[:, t])), tgt, msk, w=nw))
            n += 1
        tot += wi * (s / max(n, 1))
    return tot


@torch.no_grad()
def _test_dataset(enc, ad, d, seed, model_name, run_meta):
    """Score one dataset's test fold through the shared encoder + its own adapter (median = point
    forecast), inverted to count space. Metrics are UNWEIGHTED -- weights are a training device only.
    Returns (records, pernode, perorigin, gate) -- full artifact parity with the single trainer's
    train_one, so analysis.py can bootstrap origins and read the gate for joint runs too."""
    enc.eval(); ad.eval()
    T = d.b.X.shape[1]
    pred_by_h = {h: np.zeros((d.N, T), dtype=np.float64) for h in HORIZONS}
    for t in d.te:
        med = ad(enc(window_slice(d.Z, t), d.A_solo, d.Mt[:, t]))[:, :, MEDIAN_IDX].cpu().numpy()  # [N,H]
        for j, h in enumerate(HORIZONS):
            pred_by_h[h][:, t + h] = invert_scaler(med[:, j:j + 1], d.b.scaler)[:, 0]
    recs, pernode, perorigin = score_predictions(model_name, d.name, seed, pred_by_h, d.b, d.te,
                                                 run_meta=run_meta)
    gate = gate_spatial_readout(enc, ad, d.Z, d.A_solo, d.Mt, d.va, DEVICE)   # items 7-8, per dataset
    return recs, pernode, perorigin, gate


def train_joint(seed, steps=91000, val_every=1000, patience=12, lr=1e-3, wd=1e-4,
                sampler="uniform", dengue_balance="uniform", gate_mode="learned", device=DEVICE,
                verbose=True, names=DEV_BUNDLE_NAMES):
    """Joint-train the shared encoder + per-dataset adapters over the block-diagonal supergraph,
    select on the pooled weighted val, test per dataset. Returns
    {dataset_name: (records, pernode, perorigin, gate)}."""
    assert "ebola" not in names, \
        "ebola must never enter joint trunk training/selection (§0.5, C8); Week-5 few-shot is separate"
    torch.manual_seed(seed); np.random.seed(seed)
    ds, A_block = _prepare(names, device)

    enc = SharedEncoder(gate_mode=gate_mode).to(device)
    adapters = nn.ModuleList([Adapter().to(device) for _ in names])
    params = list(enc.parameters()) + list(adapters.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    w_base = dataset_weights(ds, sampler)                                  # sum 1
    node_w = [_node_weight(d.name, d.b, dengue_balance, device) for d in ds]
    model_name = f"encoder_joint:{sampler}-{dengue_balance}"
    run_meta = dict(training_regime="joint", sampler=f"{sampler}-{dengue_balance}",
                    gate_mode=gate_mode, topo_aug="none")   # schema parity with the single trainer

    # loss-space guard (Task 13.2): model-space targets are ~unit scale.
    tgt0, _ = targets_and_mask(ds[0].ymod, ds[0].Mt, ds[0].mtr, ds[0].tr[0], device)
    assert float(tgt0.abs().median()) < 10, "targets not in model space -- loss space is wrong"

    streams = [_origin_stream(d.tr, seed + i) for i, d in enumerate(ds)]
    best_val, best_state, bad = float("inf"), None, 0
    enc.train(); [ad.train() for ad in adapters]
    for step in range(1, steps + 1):
        ts = [next(s) for s in streams]                                    # one origin per dataset
        Zc = torch.cat([window_slice(d.Z, ts[i]) for i, d in enumerate(ds)], dim=0)      # [SigmaN,20,4]
        Mcol = torch.cat([ds[i].Mt[:, ts[i]] for i in range(len(ds))], dim=0)             # [SigmaN]
        h = enc(Zc, A_block, Mcol)                                                         # [SigmaN,d]
        losses, wts = [], []
        for i, (ad, d) in enumerate(zip(adapters, ds)):
            tgt, msk = targets_and_mask(d.ymod, d.Mt, d.mtr, ts[i], device)
            if msk.sum() == 0:
                continue
            losses.append(pinball_loss(ad(h[d.start:d.end]), tgt, msk, w=node_w[i]))
            wts.append(w_base[i])
        if not losses:
            continue
        wt = torch.tensor(wts, device=device); wt = wt / wt.sum()          # renorm over contributors
        loss = sum(a * L for a, L in zip(wt, losses))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if step % val_every == 0:
            val = _val_pinball(enc, adapters, ds, node_w, w_base)
            if val < best_val - 1e-5:
                best_val, bad = val, 0
                best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                              {k: v.detach().clone() for k, v in adapters.state_dict().items()})
            else:
                bad += 1
            if verbose:
                print(f"    joint seed{seed} step{step:6d} val={val:.4f} best={best_val:.4f}")
            enc.train(); [ad.train() for ad in adapters]
            if bad >= patience:
                break
    if best_state:
        enc.load_state_dict(best_state[0]); adapters.load_state_dict(best_state[1])

    return {d.name: _test_dataset(enc, ad, d, seed, model_name, run_meta)
            for ad, d in zip(adapters, ds)}


def run_joint(seeds=SEEDS, sampler="uniform", dengue_balance="uniform", **kw):
    _equiv_check()                     # assert the block-diagonal premise BEFORE producing the numbers
    tag = f"{sampler}-{dengue_balance}"
    for s in seeds:
        res = train_joint(s, sampler=sampler, dengue_balance=dengue_balance, **kw)
        for name, (recs, pernode, perorigin, gate) in res.items():
            write_records(recs, f"encoder_joint__{tag}__{name}__seed{s}.json")
            write_per_node(pernode, f"encoder_joint__{tag}__{name}__seed{s}__pernode.npz")
            write_per_origin(perorigin, f"encoder_joint__{tag}__{name}__seed{s}__perorigin.npz")
            write_gate(gate, f"encoder_joint__{tag}__{name}__seed{s}__gate.npz")
            print(f"  joint [{tag}] scored ({name} seed{s})")


# --------------------------------------------------------------------------- #
# Central-claim gates -- run under --equiv (and the cheap two under --smoke).
# --------------------------------------------------------------------------- #
def _equiv_check(device="cpu"):
    """Gate #1: block-diagonal forward == per-dataset solo forward, node-for-node. If this fails,
    something couples across nodes that shouldn't (the block-diagonal premise is void). Uses the 3
    small datasets so it stays CI-cheap (dengue would just be 7165 more identical rows)."""
    torch.manual_seed(0); np.random.seed(0)
    names = ["influenza_japan", "influenza_us-regions", "influenza_us-states"]
    enc = SharedEncoder().to(device).eval()
    ds, A_block = _prepare(names, device)
    ts = [d.tr[0] for d in ds]
    Zc = torch.cat([window_slice(d.Z, ts[i]) for i, d in enumerate(ds)], dim=0)
    Mcol = torch.cat([ds[i].Mt[:, ts[i]] for i in range(len(ds))], dim=0)
    with torch.no_grad():
        h_blk = enc(Zc, A_block, Mcol)
        for i, d in enumerate(ds):
            h_solo = enc(window_slice(d.Z, ts[i]), d.A_solo, d.Mt[:, ts[i]])
            assert torch.allclose(h_blk[d.start:d.end], h_solo, atol=1e-5), \
                f"block != solo for {d.name} -- cross-node coupling detected"
    print("ok  block-diagonal forward == per-dataset solo forward (no cross-node coupling)")


def _routing_check(device="cpu"):
    """Gate #2: a loss from ONE dataset must give every OTHER adapter exactly zero gradient. This is
    the strict isolation Week-4's MAML inner loop depends on -- adapting on one task cannot perturb
    another task's adapter. Holds because each adapter is applied only to its own block's rows."""
    torch.manual_seed(0); np.random.seed(0)
    names = ["influenza_japan", "influenza_us-regions", "influenza_us-states"]
    ds, A_block = _prepare(names, device)
    enc = SharedEncoder().to(device)
    adapters = nn.ModuleList([Adapter().to(device) for _ in names])
    ts = [d.tr[0] for d in ds]
    Zc = torch.cat([window_slice(d.Z, ts[i]) for i, d in enumerate(ds)], dim=0)
    Mcol = torch.cat([ds[i].Mt[:, ts[i]] for i in range(len(ds))], dim=0)
    h = enc(Zc, A_block, Mcol)

    B = 1                                                   # loss from dataset B only
    tgt, msk = targets_and_mask(ds[B].ymod, ds[B].Mt, ds[B].mtr, ts[B], device)
    pinball_loss(adapters[B](h[ds[B].start:ds[B].end]), tgt, msk).backward()

    for i, d in enumerate(ds):
        if i == B:
            assert any(p.grad is not None for p in adapters[i].parameters()), \
                f"{d.name}: adapter in the batch got NO gradient"
        else:
            assert all(p.grad is None for p in adapters[i].parameters()), \
                f"{d.name}: adapter LEAKED gradient from a B-only loss (routing broken)"
    assert any(p.grad is not None for p in enc.parameters()), "shared trunk got no gradient"
    print("ok  one-dataset loss -> every other adapter grad is exactly None (routing isolated)")


def _balance_check():
    """Gate #3: per-cell country balance equalises each dengue country's mass in the weighted mean
    (robust to the 97x obs-density spread); and w=None leaves pinball_loss bit-identical, so the
    single-disease trainer is unaffected."""
    b = load("dengue")
    v = per_cell_country_weight(b)
    g, ids = b.group_of(), b.meta["node_ids"]
    countries = np.array([g[i] for i in ids])
    opn = b.masks()["train"].astype(bool).sum(1)
    mass = {c: float((opn[countries == c] * v[countries == c]).sum()) for c in np.unique(countries)}
    lo, hi = min(mass.values()), max(mass.values())
    assert hi / lo < 1.01, f"country mass not balanced (spread {hi / lo:.3f}x): {mass}"
    pred, tgt, msk = torch.randn(6, 4, 5), torch.randn(6, 4), torch.ones(6, 4)
    assert torch.allclose(pinball_loss(pred, tgt, msk), pinball_loss(pred, tgt, msk, w=torch.ones(6))), \
        "w=ones != w=None -- the unweighted (single-disease) path changed"
    print(f"ok  per-cell country balance: all {len(mass)} dengue countries equal mass "
          f"(spread {hi / lo:.4f}x); w=None unchanged")


def _selection_check():
    """Gate #4 / C8 (§0.5): ebola must never enter joint trunk training/selection. Data-layer
    exclusion + a runtime guard in train_joint that fires if ebola is ever passed in `names`."""
    assert "ebola" not in DEV_BUNDLE_NAMES, "ebola leaked into DEV_BUNDLE_NAMES"
    try:
        train_joint(0, names=list(DEV_BUNDLE_NAMES) + ["ebola"], steps=1)
        raise SystemExit("C8 control did not fire: train_joint accepted ebola in names")
    except AssertionError:
        pass
    print("ok  C8: ebola excluded from DEV_BUNDLE_NAMES; train_joint guard fires on ebola in names")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--equiv", action="store_true")
    ap.add_argument("--steps", type=int, default=91000)
    ap.add_argument("--sampler", choices=["uniform", "proportional", "sqrt"], default="uniform")
    ap.add_argument("--dengue-balance", choices=["uniform", "country"], default="uniform")
    ap.add_argument("--gate-mode", choices=["learned", "off"], default="learned")
    a = ap.parse_args()

    t0 = time.time()
    if a.equiv:
        _equiv_check(); _routing_check(); _balance_check(); _selection_check()
    elif a.smoke:
        _equiv_check(); _routing_check()
        small = ["influenza_japan", "influenza_us-regions", "influenza_us-states"]
        train_joint(42, steps=300, val_every=100, patience=99, names=small, gate_mode=a.gate_mode)
        print(f"smoke done in {time.time()-t0:.0f}s")
    elif a.all:
        run_joint(steps=a.steps, sampler=a.sampler, dengue_balance=a.dengue_balance, gate_mode=a.gate_mode)
        print(f"all joint seeds done in {(time.time()-t0)/60:.1f} min")
    elif a.seed:
        run_joint(seeds=(a.seed,), steps=a.steps, sampler=a.sampler,
                  dengue_balance=a.dengue_balance, gate_mode=a.gate_mode)
        print(f"done in {time.time()-t0:.0f}s")
    else:
        ap.error("give --seed, or --all, or --smoke, or --equiv")


if __name__ == "__main__":
    main()

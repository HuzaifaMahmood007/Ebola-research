"""train/joint.py -- multi-disease joint training over one block-diagonal supergraph (Day 14, G2).

Replaces sequential per-dataset gradient accumulation with a single forward over a block-diagonal
supergraph: the 4 dev datasets are stacked along the node axis ([SigmaN=7271, 20, 4]) with a
block-diagonal adjacency that has NO cross-dataset edges. One forward, one backward, one step.

Verified precondition (why this is safe): nothing in the encoder reduces over the node dimension --
the TCN treats nodes as the conv batch dim, spatial message passing follows edges only, and there is
no BatchNorm. So each block's math is identical whether run alone or inside the supergraph
(_equiv_check asserts this: block == solo). No cross-block edges => no inter-dataset leakage.

Loss weighting, not sampling: a naive mean over 7271 nodes hands dengue 98.5% of the gradient (it is
7165 of the nodes). Instead each dataset's masked pinball is weighted by 1/(4*n_obs_i) -- which is
exactly the MEAN of the four per-dataset pinball means (pinball_loss already divides by n_obs_i), so
`pinball_loss` is reused unchanged. Same objective as uniform per-step sampling, lower variance.

Architecture: ONE shared encoder (the transferable trunk) + one small Adapter per dataset (FiLM +
head, ~388 params each, P5). A new disease = one new adapter few-shot-fit with the trunk frozen.

Run from the repo root as a module:
  python -m train.joint --all                    # 5 seeds x (joint train -> per-dataset test)
  python -m train.joint --seed 42                # one joint run, all 4 datasets scored
  python -m train.joint --smoke                  # 3 small datasets, few epochs (CI-cheap)
  python -m train.joint --equiv                  # the block==solo central-claim gate only
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
from train.loop import (DEVICE, RESULTS, SEEDS, _round_trip_ok, score_predictions, write_per_node,
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


def _train_epoch(enc, adapters, ds, A_block, opt, params, K):
    """One epoch = K block-diagonal steps. Each step samples one origin per dataset, stacks the
    windows into [SigmaN,20,4], runs one forward/backward/step. Loss = mean of the per-dataset
    pinball means (= 1/(4*n_obs_i) weighting). Small datasets cycle their origins (K=max len)."""
    enc.train(); [ad.train() for ad in adapters]
    orders = [np.random.permutation(d.tr) for d in ds]
    for k in range(K):
        ts = [orders[i][k % len(orders[i])] for i in range(len(ds))]
        Zc = torch.cat([window_slice(d.Z, ts[i]) for i, d in enumerate(ds)], dim=0)      # [SigmaN,20,4]
        Mcol = torch.cat([ds[i].Mt[:, ts[i]] for i in range(len(ds))], dim=0)             # [SigmaN]
        h = enc(Zc, A_block, Mcol)                                                         # [SigmaN,d]
        losses = []
        for i, (ad, d) in enumerate(zip(adapters, ds)):
            tgt, msk = targets_and_mask(d.ymod, d.Mt, d.mtr, ts[i], DEVICE)
            if msk.sum() == 0:
                continue
            losses.append(pinball_loss(ad(h[d.start:d.end]), tgt, msk))
        if not losses:
            continue
        opt.zero_grad()
        (sum(losses) / len(losses)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()


@torch.no_grad()
def _val_pinball(enc, adapters, ds):
    """Full val pass, per dataset on its SOLO graph (== its block, by _equiv_check), averaged to a
    dataset mean, then averaged across datasets -- the same 1/4-each weighting used in training."""
    enc.eval()
    tot, k = 0.0, 0
    for ad, d in zip(adapters, ds):
        ad.eval()
        s, n = 0.0, 0
        for t in d.va:
            tgt, msk = targets_and_mask(d.ymod, d.Mt, d.mva, t, DEVICE)
            if msk.sum() == 0:
                continue
            s += float(pinball_loss(ad(enc(window_slice(d.Z, t), d.A_solo, d.Mt[:, t])), tgt, msk))
            n += 1
        tot += s / max(n, 1); k += 1
    return tot / max(k, 1)


@torch.no_grad()
def _test_dataset(enc, ad, d, seed):
    """Score one dataset's test fold through the shared encoder + its own adapter (median = point
    forecast), inverted to count space. Reuses train.loop.score_predictions unchanged."""
    enc.eval(); ad.eval()
    T = d.b.X.shape[1]
    pred_by_h = {h: np.zeros((d.N, T), dtype=np.float64) for h in HORIZONS}
    for t in d.te:
        med = ad(enc(window_slice(d.Z, t), d.A_solo, d.Mt[:, t]))[:, :, MEDIAN_IDX].cpu().numpy()  # [N,H]
        for j, h in enumerate(HORIZONS):
            pred_by_h[h][:, t + h] = invert_scaler(med[:, j:j + 1], d.b.scaler)[:, 0]
    return score_predictions("encoder_joint", d.name, seed, pred_by_h, d.b, d.te)


def train_joint(seed, epochs=80, lr=1e-3, wd=1e-4, patience=15, device=DEVICE, verbose=True,
                names=DEV_BUNDLE_NAMES):
    """Joint train the shared encoder + per-dataset adapters, select on pooled val, test per dataset.
    Returns {dataset_name: (records, pernode)}."""
    torch.manual_seed(seed); np.random.seed(seed)
    ds, A_block = _prepare(names, device)

    enc = SharedEncoder().to(device)
    adapters = nn.ModuleList([Adapter().to(device) for _ in names])
    params = list(enc.parameters()) + list(adapters.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    K = max(len(d.tr) for d in ds)                          # dengue's origin count; small sets cycle

    # loss-space guard (Task 13.2): model-space targets are ~unit scale.
    tgt0, _ = targets_and_mask(ds[0].ymod, ds[0].Mt, ds[0].mtr, ds[0].tr[0], device)
    assert float(tgt0.abs().median()) < 10, "targets not in model space -- loss space is wrong"

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        _train_epoch(enc, adapters, ds, A_block, opt, params, K)
        val = _val_pinball(enc, adapters, ds)
        sched.step()
        if val < best_val - 1e-5:
            best_val, bad = val, 0
            best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                          {k: v.detach().clone() for k, v in adapters.state_dict().items()})
        else:
            bad += 1
        if verbose:
            print(f"    joint seed{seed} ep{ep:02d} val_pinball={val:.4f} best={best_val:.4f}")
        if bad >= patience:
            break
    if best_state:
        enc.load_state_dict(best_state[0]); adapters.load_state_dict(best_state[1])

    return {d.name: _test_dataset(enc, ad, d, seed) for ad, d in zip(adapters, ds)}


def run_joint(seeds=SEEDS, epochs=80, **kw):
    for s in seeds:
        for name, (recs, pernode) in train_joint(s, epochs=epochs, **kw).items():
            write_records(recs, f"encoder_joint__{name}__seed{s}.json")
            write_per_node(pernode, f"encoder_joint__{name}__seed{s}__pernode.npz")
            print(f"  joint encoder scored ({name} seed{s})")


def _equiv_check(device="cpu"):
    """Central-claim gate: block-diagonal forward == per-dataset solo forward, node-for-node. If this
    fails, something couples across nodes that shouldn't (the block-diagonal premise is void). Uses
    the 3 small datasets so it stays CI-cheap (dengue would just be 7165 more identical rows)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--equiv", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()

    t0 = time.time()
    if a.equiv:
        _equiv_check()
    elif a.smoke:
        _equiv_check()
        small = ["influenza_japan", "influenza_us-regions", "influenza_us-states"]
        train_joint(42, epochs=3, names=small)
        print(f"smoke done in {time.time()-t0:.0f}s")
    elif a.all:
        run_joint(epochs=a.epochs)
        print(f"all joint seeds done in {(time.time()-t0)/60:.1f} min")
    elif a.seed:
        run_joint(seeds=(a.seed,), epochs=a.epochs)
        print(f"done in {time.time()-t0:.0f}s")
    else:
        ap.error("give --seed, or --all, or --smoke, or --equiv")


if __name__ == "__main__":
    main()

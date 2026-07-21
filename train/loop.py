"""train/loop.py -- single-disease training + naive floors (Week-3 Day 13, G6 prerequisite).

One forward pass emits all four horizons (direct multi-horizon, C5), so a run is one model per
(dataset, seed), not per horizon: 4 datasets x 5 seeds = 20 runs. Selection is on the val fold
only; the test fold is scored once at the end. Everything lands in results/*.json as one record per
(model, dataset, horizon, seed, metric) -- Week 6's tables are generated from these, never typed.

Two silent-bug guards from the guide (Task 13.2), asserted at run start:
  * pinball loss is computed in MODEL space (targets ~ unit scale): assert targets.abs().median()<10;
  * scaler round-trip: invert_scaler(apply_scaler(raw)) == raw on observed cells, to 1e-4.

Naive floors (Task 13.3), scored on the identical cells: persistence, seasonal-naive (with a
recorded persistence fallback where y[t+h-52] is missing/unobserved), and the per-node train mean.

Run from the repo root as a module:
  python -m train.loop --dataset influenza_japan --seed 42     # one run
  python -m train.loop --all                                   # the 20-run matrix + naive floors
  python -m train.loop --smoke                                 # japan, 1 seed, few epochs (CI-cheap)
"""
from __future__ import annotations

import os
# torch ships libomp.dll and pandas' MKL ships libiomp5md.dll; both init OpenMP in one process on
# Windows conda. Set the documented allow-duplicate flag BEFORE numpy/torch/pandas import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import bundles
import score
from models import (MEDIAN_IDX, Adapter, SharedEncoder, pinball_loss, sparse_from_dense_np,
                    targets_and_mask, window_slice)
from to_schema import apply_scaler, invert_scaler

RESULTS = Path("results")
SEEDS = (42, 52, 62, 72, 82)
STEPS_PER_YEAR = 52                                    # weekly data; seasonal-naive lag
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _round_trip_ok(b):
    """Task 13.2: invert_scaler(apply_scaler(raw)) == raw on observed cells (mis-ordering the two
    transforms is the easiest silent bug in the trainer, and invisible in the loss curve)."""
    rt = invert_scaler(apply_scaler(b.raw, b.scaler), b.scaler)
    obs = b.M.astype(bool)
    return np.allclose(rt[obs], b.raw[obs], atol=1e-4)


def train_one(name, seed, epochs=80, lr=1e-3, wd=1e-4, batch_origins=8, patience=15,
              device=DEVICE, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    b = bundles.load(name)
    assert _round_trip_ok(b), f"{name}: scaler round-trip failed -- transforms mis-ordered"

    Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=device)     # [N,T,4]
    ymod = torch.tensor(b.y, dtype=torch.float32, device=device)                 # [N,T] model space
    Mt = torch.tensor(b.M, dtype=torch.float32, device=device)                  # [N,T]
    A = sparse_from_dense_np(b.A_geo).to(device)
    masks = {p: torch.tensor(m, dtype=torch.float32, device=device) for p, m in b.masks().items()}
    tr, va, te = (b.origins(phase="train"), b.origins(phase="val"), b.origins(phase="test"))

    enc, ad = SharedEncoder().to(device), Adapter().to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(ad.parameters()), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # loss-space guard: model-space targets are ~unit scale on the first batch (Task 13.2).
    tgt0, _ = targets_and_mask(ymod, Mt, masks["train"], tr[0], device)
    assert float(tgt0.abs().median()) < 10, "targets not in model space -- loss space is wrong"

    def run_phase(origins, train_mode):
        enc.train(train_mode); ad.train(train_mode)
        total, n = 0.0, 0
        order = np.random.permutation(origins) if train_mode else origins
        if train_mode:
            opt.zero_grad()
        for k, t in enumerate(order):
            tgt, msk = targets_and_mask(ymod, Mt, masks["train" if train_mode else "val"], t, device)
            if msk.sum() == 0:
                continue
            ctx = torch.enable_grad() if train_mode else torch.no_grad()
            with ctx:
                pred = ad(enc(window_slice(Z, t), A, Mt[:, t]))                  # [N,H,Q]
                loss = pinball_loss(pred, tgt, msk)
            if train_mode:
                loss.backward()
                if (k + 1) % batch_origins == 0:
                    torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(ad.parameters()), 1.0)
                    opt.step(); opt.zero_grad()
            total += float(loss) * int(msk.sum()); n += int(msk.sum())
        if train_mode:
            opt.step(); opt.zero_grad()
        return total / max(n, 1)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        run_phase(tr, True)
        val = run_phase(va, False)
        sched.step()
        if val < best_val - 1e-5:
            best_val, bad = val, 0
            best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                          {k: v.detach().clone() for k, v in ad.state_dict().items()})
        else:
            bad += 1
        if verbose:
            print(f"    {name} seed{seed} ep{ep:02d} val_pinball={val:.4f} best={best_val:.4f}")
        if bad >= patience:
            break
    if best_state:
        enc.load_state_dict(best_state[0]); ad.load_state_dict(best_state[1])

    # test predictions -> count space, per horizon, assembled to [N,T] aligned by target time.
    enc.eval(); ad.eval()
    T = b.X.shape[1]
    pred_by_h = {h: np.zeros((b.X.shape[0], T), dtype=np.float64) for h in bundles.HORIZONS}
    with torch.no_grad():
        for t in te:
            med = ad(enc(window_slice(Z, t), A, Mt[:, t]))[:, :, MEDIAN_IDX].cpu().numpy()   # [N,H]
            for j, h in enumerate(bundles.HORIZONS):
                pred_by_h[h][:, t + h] = invert_scaler(med[:, j:j + 1], b.scaler)[:, 0]
    return score_predictions("encoder", name, seed, pred_by_h, b, te)


# --------------------------------------------------------------------------- #
# Naive floors (Task 13.3), all in count space, scored on the same test cells.
# --------------------------------------------------------------------------- #
def naive_predictions(b, te):
    """persistence, seasonal (+ fallback count), per-node train mean -- as {model: {h: [N,T]}}."""
    N, T = b.raw.shape
    raw, Mobs = b.raw.astype(np.float64), b.M.astype(bool)
    train_mask = b.masks()["train"].astype(bool)
    train_mean = np.array([raw[i, train_mask[i]].mean() if train_mask[i].any() else 0.0
                           for i in range(N)])
    out = {m: {h: np.zeros((N, T)) for h in bundles.HORIZONS}
           for m in ("persistence", "seasonal", "train_mean")}
    fallback, total = 0, 0
    for t in te:
        for h in bundles.HORIZONS:
            tt = t + h
            out["persistence"][h][:, tt] = raw[:, t]                       # y_t
            out["train_mean"][h][:, tt] = train_mean
            src = tt - STEPS_PER_YEAR                                        # y_{t+h-52}
            if src >= 0:
                use = Mobs[:, src]
                out["seasonal"][h][:, tt] = np.where(use, raw[:, src], raw[:, t])
                fallback += int((~use).sum()); total += N
            else:
                out["seasonal"][h][:, tt] = raw[:, t]                        # whole column falls back
                fallback += N; total += N
    return out, (fallback / max(total, 1))


def score_predictions(model_name, dataset, seed, pred_by_h, b, origins, phase="test"):
    """Score {h: [N,T] count preds} through score.py per horizon; return one record per metric."""
    raw = b.raw.astype(np.float64)
    phase_mask = b.masks()[phase].astype(np.uint8)
    ids, ncmap = b.meta["node_ids"], b.group_of()
    records = []
    for h in bundles.HORIZONS:
        mask_h = np.zeros_like(phase_mask)
        for t in origins:
            mask_h[:, t + h] = phase_mask[:, t + h]                          # observed folded into the phase mask
        agg, _ = score.score_bundle(pred_by_h[h], raw, mask_h, ids, ncmap)
        for metric, a in agg.items():
            records.append(dict(model=model_name, dataset=dataset, horizon=h, seed=seed,
                                metric=metric, country_macro=a["country_macro"],
                                node_mean=a["node_mean"], n_countries=a["n_countries"],
                                n_nodes=a["n_nodes"]))
    return records


def write_records(records, fname):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / fname).write_text(json.dumps(records, indent=2))


def run_dataset(name, seeds=SEEDS, **kw):
    b = bundles.load(name)
    te = b.origins(phase="test")
    naive, fb_rate = naive_predictions(b, te)                              # deterministic; seed is null
    naive_recs = []
    for mname, preds in naive.items():
        recs = score_predictions(mname, name, None, preds, b, te)
        for r in recs:
            r["seasonal_fallback_rate"] = round(fb_rate, 4) if mname == "seasonal" else None
        naive_recs += recs
    write_records(naive_recs, f"naive__{name}.json")
    print(f"  naive floors scored ({name}); seasonal fallback rate {fb_rate:.1%}")
    for s in seeds:
        write_records(train_one(name, s, **kw), f"encoder__{name}__seed{s}.json")
        print(f"  encoder scored ({name} seed{s})")


def _selfcheck():
    """Seasonal-naive fallback accounting is correct on a tiny synthetic case (Task 13.3 note)."""
    class B:                                                                # T=70 so origin 52 + h=15 fits
        raw = np.arange(70.0).reshape(1, 70)
        M = np.ones((1, 70), np.uint8); M[0, 5] = 0                          # week 5 unobserved
        def masks(self): return {"train": np.ones((1, 70), np.uint8)}
    b = B(); b.masks = B.masks.__get__(b)
    out, rate = naive_predictions(b, [52])                                  # origin 52: h=5 -> target 57, src=5 unobs
    assert out["seasonal"][5][0, 57] == b.raw[0, 52], "unobserved seasonal source did not fall back"
    assert out["seasonal"][3][0, 55] == b.raw[0, 3], "observed seasonal source should be used"
    assert rate > 0, "fallback rate should be > 0 when a source is unobserved"
    print(f"ok  seasonal-naive fallback accounting correct (rate {rate:.3f} on the synthetic case)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=bundles.DEV_BUNDLE_NAMES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        _selfcheck(); return
    t0 = time.time()
    if a.smoke:
        _selfcheck()
        recs = train_one("influenza_japan", 42, epochs=5)
        write_records(recs, "encoder__influenza_japan__seed42__smoke.json")
        print(f"smoke done in {time.time()-t0:.0f}s; wrote {len(recs)} records")
    elif a.all:
        for name in bundles.DEV_BUNDLE_NAMES:
            run_dataset(name, epochs=a.epochs)
        print(f"all datasets done in {(time.time()-t0)/60:.1f} min")
    elif a.dataset and a.seed:
        write_records(train_one(a.dataset, a.seed, epochs=a.epochs), f"encoder__{a.dataset}__seed{a.seed}.json")
        print(f"done in {time.time()-t0:.0f}s")
    else:
        ap.error("give --dataset+--seed, or --all, or --smoke, or --selfcheck")


if __name__ == "__main__":
    main()

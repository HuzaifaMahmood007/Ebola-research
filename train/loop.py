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
              device=DEVICE, verbose=True, zero_channels=None,
              training_regime="single", sampler=None, gate_mode="learned", topo_aug="none",
              gate_read=True):
    assert name != "ebola", \
        "ebola must never enter trunk training/selection (§0.5, C8); Week-5 few-shot is a separate path"
    torch.manual_seed(seed)
    np.random.seed(seed)
    b = bundles.load(name)
    assert _round_trip_ok(b), f"{name}: scaler round-trip failed -- transforms mis-ordered"

    Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=device)     # [N,T,4]
    if zero_channels:                       # ablation hook: zero given core channels (still 4-ch, C1 holds)
        Z[:, :, list(zero_channels)] = 0.0  # e.g. (1,2) = drop sin_doy/cos_doy seasonal phase
    ymod = torch.tensor(b.y, dtype=torch.float32, device=device)                 # [N,T] model space
    Mt = torch.tensor(b.M, dtype=torch.float32, device=device)                  # [N,T]
    A = sparse_from_dense_np(b.A_geo).to(device)
    masks = {p: torch.tensor(m, dtype=torch.float32, device=device) for p, m in b.masks().items()}
    tr, va, te = (b.origins(phase="train"), b.origins(phase="val"), b.origins(phase="test"))

    enc, ad = SharedEncoder(gate_mode=gate_mode).to(device), Adapter().to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(ad.parameters()), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # loss-space guard: model-space targets are ~unit scale on the first batch (Task 13.2).
    tgt0, _ = targets_and_mask(ymod, Mt, masks["train"], tr[0], device)
    assert float(tgt0.abs().median()) < 10, "targets not in model space -- loss space is wrong"

    params = list(enc.parameters()) + list(ad.parameters())

    def run_phase(origins, train_mode):
        enc.train(train_mode); ad.train(train_mode)
        total, n = 0.0, 0
        order = np.random.permutation(origins) if train_mode else origins
        if train_mode:
            opt.zero_grad()
        pending = 0                     # backward()s since the last opt.step(); NOT the origin index
        for t in order:
            tgt, msk = targets_and_mask(ymod, Mt, masks["train" if train_mode else "val"], t, device)
            if msk.sum() == 0:
                continue
            ctx = torch.enable_grad() if train_mode else torch.no_grad()
            with ctx:
                pred = ad(enc(window_slice(Z, t), A, Mt[:, t]))                  # [N,H,Q]
                loss = pinball_loss(pred, tgt, msk)
            if train_mode:
                loss.backward()
                pending += 1
                if pending == batch_origins:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); opt.zero_grad(); pending = 0
            total += float(loss) * int(msk.sum()); n += int(msk.sum())
        if train_mode and pending:                     # clip the trailing partial batch too; skip if empty
            torch.nn.utils.clip_grad_norm_(params, 1.0)
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
    run_meta = dict(training_regime=training_regime, sampler=sampler,
                    gate_mode=gate_mode, topo_aug=topo_aug)
    gate = gate_spatial_readout(enc, ad, Z, A, Mt, va, device) if gate_read else None
    recs, pernode, perorigin = score_predictions("encoder", name, seed, pred_by_h, b, te,
                                                 run_meta=run_meta)
    return recs, pernode, perorigin, gate


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


def _per_origin_country(pred_h, raw, mask_h, origins, h, scored_idx, ccol, n_countries):
    """Per (origin, country) error sufficient stats at horizon h, count space, over scored nodes.
    Returns sae, sse [K,C] float and n [K,C] int (K=len(origins), C=n_countries). Vectorised: no
    per-node python loop. These are the bootstrap-over-origins material (item 6) -- resample origins,
    per-country MAE = sum(sae)/sum(n), RMSE = sqrt(sum(sse)/sum(n)), then macro over countries."""
    K, C = len(origins), n_countries
    sae = np.zeros((K, C)); sse = np.zeros((K, C)); nn = np.zeros((K, C), dtype=np.int64)
    if scored_idx.size == 0:
        return sae, sse, nn
    cols = np.array([t + h for t in origins])
    m = mask_h[np.ix_(scored_idx, cols)].astype(bool)                    # [S,K] observed & in-phase
    e = (pred_h[np.ix_(scored_idx, cols)] - raw[np.ix_(scored_idx, cols)]) * m
    abse, sqe = np.abs(e), e * e
    for j in range(C):
        rows = ccol == j
        if rows.any():
            sae[:, j] = abse[rows].sum(0); sse[:, j] = sqe[rows].sum(0); nn[:, j] = m[rows].sum(0)
    return sae, sse, nn


def score_predictions(model_name, dataset, seed, pred_by_h, b, origins, phase="test", run_meta=None):
    """Score {h: [N,T] count preds} through score.py per horizon. Returns (records, pernode, perorigin):
    the aggregate record per metric; the per-node score dict per horizon (paired Wilcoxon / per-node
    bootstrap, review #7); and per (origin, country) error sufficient stats for bootstrap-CI OVER
    ORIGINS, incl. the country-macro CI (item 6) -- none reconstructable from the aggregates alone.
    run_meta (training_regime/sampler/gate_mode/topo_aug) is merged into every record."""
    raw = b.raw.astype(np.float64)
    phase_mask = b.masks()[phase].astype(np.uint8)
    ids, ncmap = b.meta["node_ids"], b.group_of()
    countries = sorted({ncmap[n] for n in ids})
    cidx = {c: j for j, c in enumerate(countries)}
    records, pernode = [], {}
    perorigin = {"countries": np.array(countries), "origins": np.array(origins, dtype=np.int32)}
    for h in bundles.HORIZONS:
        mask_h = np.zeros_like(phase_mask)
        for t in origins:
            mask_h[:, t + h] = phase_mask[:, t + h]                          # observed folded into the phase mask
        agg, ns = score.score_bundle(pred_by_h[h], raw, mask_h, ids, ncmap)
        pernode[h] = ns
        scored_idx = np.array(sorted(ns), dtype=np.int64)                    # non-constant scored nodes
        ccol = np.array([cidx[ncmap[ids[i]]] for i in scored_idx], dtype=np.int64)
        sae, sse, nn = _per_origin_country(pred_by_h[h], raw, mask_h, origins, h, scored_idx, ccol,
                                           len(countries))
        perorigin[f"h{h}__sae"] = sae.astype(np.float32)
        perorigin[f"h{h}__sse"] = sse.astype(np.float32)
        perorigin[f"h{h}__n"] = nn.astype(np.int32)
        for metric, a in agg.items():
            rec = dict(model=model_name, dataset=dataset, horizon=h, seed=seed,
                       metric=metric, country_macro=a["country_macro"],
                       node_mean=a["node_mean"], n_countries=a["n_countries"], n_nodes=a["n_nodes"])
            if run_meta:
                rec.update(run_meta)
            records.append(rec)
    return records, pernode, perorigin


def write_records(records, fname):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / fname).write_text(json.dumps(records, indent=2))


def write_per_node(pernode, fname):
    """Per-node scores as compact npz (review #7). npz not JSON: dengue is 7165 nodes x 4 horizons,
    so this is a few hundred KB of arrays, not a megabyte of dict text. One node_idx array per
    horizon (the scored set differs by horizon -- constant-window nodes are dropped per horizon)."""
    RESULTS.mkdir(exist_ok=True)
    arrays = {}
    for h, ns in pernode.items():
        idx = sorted(ns)
        arrays[f"h{h}__node_idx"] = np.array(idx, dtype=np.int32)
        arrays[f"h{h}__country"] = np.array([ns[i]["country"] for i in idx])
        arrays[f"h{h}__n_cells"] = np.array([ns[i]["n_cells"] for i in idx], dtype=np.int32)
        for metric in score.METRICS:
            arrays[f"h{h}__{metric}"] = np.array([ns[i][metric] for i in idx], dtype=np.float32)
    np.savez_compressed(RESULTS / fname, **arrays)


def write_per_origin(perorigin, fname):
    """Per (origin, country) error sufficient stats (item 6). npz: countries[C], origins[K], and
    per horizon sae/sse [K,C] + n [K,C]. Node-pooled within country, count space, scored (non-
    constant) nodes -- the same population as the headline metric. Reconstructs a country-macro CI
    by resampling origins: per-country MAE=sum(sae)/sum(n), RMSE=sqrt(sum(sse)/sum(n)), macro over C.

    NOTE: this is a CELL-POOLED country-macro (sum over the country's cells), whereas the headline
    aggregate() is NODE-AVERAGED (mean of per-node metrics). They coincide on the dense influenza
    panels (near-equal cells/node) and diverge on dengue. The item-6 CI is therefore on the
    cell-pooled metric -- documented in analysis.py."""
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / fname, **perorigin)


def gate_spatial_readout(enc, ad, Z, A, Mt, origins, device):
    """Free reads 7-8: per-node gate g and normalised spatial contribution, meaned over the val
    origins. Both are PRE-HEAD (computed once per forward, before the multi-horizon head), so they
    are horizon-independent -- read once per origin, not per horizon. Returns {mean_g[N], mean_sc[N]}
    (float32); the per-dataset distribution/IQR/frac(g<0.05) is assembled offline in analysis.py."""
    from models.encoder import spatial_contribution
    enc.eval(); ad.eval()
    N = Z.shape[0]
    g_sum, sc_sum, k = np.zeros(N), np.zeros(N), 0
    with torch.no_grad():
        for t in origins:
            enc(window_slice(Z, t), A, Mt[:, t])                     # populates enc.last_g/last_h/last_h_s
            g_sum += enc.last_g.squeeze(-1).cpu().numpy()
            sc_sum += spatial_contribution(enc.last_g, enc.last_h, enc.last_h_s).cpu().numpy()
            k += 1
    return {"mean_g": (g_sum / max(k, 1)).astype(np.float32),
            "mean_sc": (sc_sum / max(k, 1)).astype(np.float32)}


def write_gate(gate, fname):
    """Per-node gate/spatial readout (items 7-8) as compact npz."""
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / fname, **gate)


NAIVE_META = dict(training_regime="single", sampler=None, gate_mode=None, topo_aug=None)


def run_dataset(name, seeds=SEEDS, **kw):
    b = bundles.load(name)
    te = b.origins(phase="test")
    naive, fb_rate = naive_predictions(b, te)                              # deterministic; seed is null
    naive_recs = []
    for mname, preds in naive.items():
        recs, pernode, perorigin = score_predictions(mname, name, None, preds, b, te, run_meta=NAIVE_META)
        for r in recs:
            r["seasonal_fallback_rate"] = round(fb_rate, 4) if mname == "seasonal" else None
        naive_recs += recs
        write_per_node(pernode, f"naive__{name}__{mname}__pernode.npz")
        write_per_origin(perorigin, f"naive__{name}__{mname}__perorigin.npz")
    write_records(naive_recs, f"naive__{name}.json")
    print(f"  naive floors scored ({name}); seasonal fallback rate {fb_rate:.1%}")
    for s in seeds:
        recs, pernode, perorigin, gate = train_one(name, s, **kw)
        write_records(recs, f"encoder__{name}__seed{s}.json")
        write_per_node(pernode, f"encoder__{name}__seed{s}__pernode.npz")
        write_per_origin(perorigin, f"encoder__{name}__seed{s}__perorigin.npz")
        if gate is not None:
            write_gate(gate, f"encoder__{name}__seed{s}__gate.npz")
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


def _perorigin_selfcheck():
    """Per-(origin,country) sufficient stats reconstruct the pooled MAE/RMSE and group by country
    correctly (item 6 artifact). 3 nodes: countries A={0,1}, B={2}; 2 origins; known errors."""
    raw = np.zeros((3, 30)); pred_h = np.zeros((3, 30)); mask_h = np.zeros((3, 30), np.uint8)
    origins, h = [10, 12], 5                                        # target cols 15, 17
    truth = {(0, 15): 10, (1, 15): 20, (2, 15): 30, (0, 17): 4, (1, 17): 5, (2, 17): 6}
    err = {(0, 15): 2, (1, 15): -4, (2, 15): 3, (0, 17): 1, (1, 17): 0, (2, 17): -1}
    for (i, c), v in truth.items():
        raw[i, c] = v; mask_h[i, c] = 1; pred_h[i, c] = v + err[(i, c)]
    scored_idx, ccol = np.array([0, 1, 2]), np.array([0, 0, 1])     # A,A,B
    sae, sse, nn = _per_origin_country(pred_h, raw, mask_h, origins, h, scored_idx, ccol, 2)
    assert (sae[0] == [6, 3]).all() and (sse[0] == [20, 9]).all() and (nn[0] == [2, 1]).all()   # col15
    assert (sae[1] == [1, 1]).all() and (sse[1] == [1, 1]).all() and (nn[1] == [2, 1]).all()   # col17
    macA_mae = sae[:, 0].sum() / nn[:, 0].sum()                     # pooled country-A MAE over origins
    assert abs(macA_mae - (6 + 1) / (2 + 2)) < 1e-9, "per-origin sae/n does not reconstruct pooled MAE"
    # an unobserved cell must contribute nothing (mask gates it out)
    mask_h[0, 15] = 0
    sae2, _, nn2 = _per_origin_country(pred_h, raw, mask_h, origins, h, scored_idx, ccol, 2)
    assert nn2[0, 0] == 1 and sae2[0, 0] == 4, "masked-out cell still contributed"
    print("ok  per-origin x country sufficient stats reconstruct pooled MAE/RMSE; mask gates cells")


def _ebola_isolation_check():
    """C8 (§0.5): ebola must never enter a trunk training/selection loop. Two-part guarantee -- the
    data-layer exclusion (ebola exists but is not in the dev set) AND a runtime guard in train_one
    that fires if ANY caller passes 'ebola', regardless of how the dataset list was built."""
    assert "ebola" in bundles.BUNDLE_NAMES and "ebola" not in bundles.DEV_BUNDLE_NAMES, \
        "ebola must exist as a bundle but be excluded from DEV_BUNDLE_NAMES"
    try:
        train_one("ebola", 42); raise SystemExit("C8 control did not fire: train_one accepted ebola")
    except AssertionError:
        pass
    print("ok  C8: ebola excluded from DEV_BUNDLE_NAMES; train_one('ebola') guard fires")


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
        _selfcheck(); _perorigin_selfcheck(); _ebola_isolation_check(); return
    t0 = time.time()
    if a.smoke:
        _selfcheck(); _perorigin_selfcheck()
        recs, pernode, perorigin, gate = train_one("influenza_japan", 42, epochs=5)
        write_records(recs, "encoder__influenza_japan__seed42__smoke.json")
        write_per_node(pernode, "encoder__influenza_japan__seed42__smoke__pernode.npz")
        write_per_origin(perorigin, "encoder__influenza_japan__seed42__smoke__perorigin.npz")
        write_gate(gate, "encoder__influenza_japan__seed42__smoke__gate.npz")
        print(f"smoke done in {time.time()-t0:.0f}s; wrote {len(recs)} records")
    elif a.all:
        for name in bundles.DEV_BUNDLE_NAMES:
            run_dataset(name, epochs=a.epochs)
        print(f"all datasets done in {(time.time()-t0)/60:.1f} min")
    elif a.dataset and a.seed:
        recs, pernode, perorigin, gate = train_one(a.dataset, a.seed, epochs=a.epochs)
        write_records(recs, f"encoder__{a.dataset}__seed{a.seed}.json")
        write_per_node(pernode, f"encoder__{a.dataset}__seed{a.seed}__pernode.npz")
        write_per_origin(perorigin, f"encoder__{a.dataset}__seed{a.seed}__perorigin.npz")
        write_gate(gate, f"encoder__{a.dataset}__seed{a.seed}__gate.npz")
        print(f"done in {time.time()-t0:.0f}s")
    else:
        ap.error("give --dataset+--seed, or --all, or --smoke, or --selfcheck")


if __name__ == "__main__":
    main()

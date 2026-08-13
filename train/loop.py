"""Single-disease training + naive floors.

One forward pass emits all four horizons (C5), so a run is one model per (dataset, seed), not per
horizon. Selection is on the val fold only; the test fold is scored once at the end. Records land in
results/*.json, one per (model, dataset, horizon, seed, metric), and every table is generated from
them, never typed.

Two silent-bug guards asserted at run start: the pinball loss is computed in MODEL space
(targets.abs().median() < 10), and the scaler round-trips to 1e-4 on observed cells. Naive floors
are scored on the identical cells: persistence, seasonal-naive (with a recorded persistence fallback
where y[t+h-52] is missing), and the per-node train mean.

  python -m train.loop --dataset influenza_japan --seed 42     # one run
  python -m train.loop --all                                   # the full matrix + naive floors
  python -m train.loop --smoke                                 # japan, 1 seed, few epochs
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
from results_paths import rpath

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


# TWO-SIDED. A [0, 3] grid can only raise a forecast, so a model that OVER-predicts has its optimum
# clamped to the 0.0 boundary -- and a clamped 0.0 is indistinguishable from a genuine "no correction
# needed", including to _bias_selfcheck, which asserts exactly c==0.0 for the no-gap case. The
# median<mean argument says the correction is usually positive; it does not license forbidding the
# other sign. _best_offset warns when the winner sits on either boundary.
BIAS_GRID = np.linspace(-1.5, 3.0, 91)         # log-space offsets searched on the val fold


def _best_offset(med, tru, obs, scaler, grid=None):
    """The scalar search behind the bias correction, kept model-free so it is self-checkable.

    med/tru/obs are [N,K] over the K fitting origins (model space, counts, observed-flag). Returns
    the grid value of c minimising count-space RMSE. c=0 is in the grid, so the fitted correction
    can never be WORSE than uncorrected on the fold it was fitted on.

    `grid` resolves at CALL time, not def time: a `grid=BIAS_GRID` default would freeze the constant
    at import and silently ignore anyone who reassigns it -- including the mutation test that is
    supposed to prove _bias_selfcheck can fail."""
    grid = BIAS_GRID if grid is None else grid
    y = tru[obs]
    best_c, best = float(grid[0]), float("inf")
    for c in grid:
        e = float(np.sqrt(np.mean((invert_scaler(med + c, scaler)[obs] - y) ** 2)))
        if e < best:
            best, best_c = e, float(c)
    if len(grid) > 1 and best_c in (float(grid[0]), float(grid[-1])):
        print(f"    WARNING bias correction hit a grid boundary (c={best_c:+.2f}); the true optimum "
              f"is outside [{grid[0]:+.2f}, {grid[-1]:+.2f}] and the value is clamped, not fitted")
    return best_c


def _fit_bias_correction(enc, ad, Z, A, Mt, b, origins, val_mask, grid=None):
    """Doubt.md §3.1: the pinball head predicts the LOG-space median, which inverts to the COUNT-space
    median, but RMSE is minimised by the count-space MEAN. Per node, mean/expm1(mean(log1p)) runs to
    19.2x on influenza_japan and 4.2x on dengue, so the point forecast is systematically low for the
    metric it is scored by -- visible as an RMSE/MAE ratio above every naive floor's on every panel.

    Fit ONE scalar offset c per horizon, in model space, on the VAL fold only (already used for early
    stopping, so no test information enters). Applied as invert_scaler(median + c): the per-node
    scaler is affine in log space, so c is a per-node MULTIPLICATIVE correction exp(c*sigma_i) on the
    counts, not a global additive fudge -- a node with a wide training spread gets a proportionally
    larger correction, which is the right shape for a lognormal gap.

    Grid search rather than a solver: one bounded scalar per horizon, cheap objective, cannot diverge.
    Returns {h: c}. Ebola never reaches here (C8) and has no val fold; its c must be inherited from
    the dev diseases and pre-registered before scoring."""
    enc.eval(); ad.eval()
    N, K = b.X.shape[0], len(origins)
    raw = b.raw.astype(np.float64)
    vm = val_mask.cpu().numpy() if torch.is_tensor(val_mask) else val_mask
    vm = vm.astype(bool)                                   # b.masks() already folds in observedness
    med = {h: np.zeros((N, K)) for h in bundles.HORIZONS}
    tru = {h: np.zeros((N, K)) for h in bundles.HORIZONS}
    obs = {h: np.zeros((N, K), dtype=bool) for h in bundles.HORIZONS}
    with torch.no_grad():
        for k, t in enumerate(origins):
            m = ad(enc(window_slice(Z, t), A, Mt[:, t]))[:, :, MEDIAN_IDX].cpu().numpy()   # [N,H]
            for j, h in enumerate(bundles.HORIZONS):
                med[h][:, k] = m[:, j]
                tru[h][:, k] = raw[:, t + h]
                obs[h][:, k] = vm[:, t + h]
    return {h: (_best_offset(med[h], tru[h], obs[h], b.scaler, grid) if obs[h].any() else 0.0)
            for h in bundles.HORIZONS}


def train_one(name, seed, epochs=80, lr=1e-3, wd=1e-4, batch_origins=8, patience=15,
              device=DEVICE, verbose=True, zero_channels=None,
              training_regime="single", sampler=None, gate_mode="learned", topo_aug="none",
              gate_read=True, quant_out=None, run_out=None, epi=None):
    assert not name.startswith("ebola"), \
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

    # Epi-informed ablation (Task 14.3 / CONFIRM-P3), off unless a caller asks for it. Imported
    # lazily so train/ never depends on ablation/ in the default path. The penalty is added to the
    # TRAIN objective only: model selection stays on val pinball in both arms, or the ablation would
    # be comparing two different early-stopping criteria as well as two different losses.
    sd_t = None
    if epi:
        from ablation.epi_penalty import growth_penalty
        sd_t = torch.tensor(np.asarray(b.scaler["std"], dtype=np.float32), device=device)

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
                if epi and train_mode:
                    pen = growth_penalty(pred, msk, ymod[:, t], Mt[:, t], sd_t, epi["r_max"])
                    epi.setdefault("_seen", [0.0, 0])
                    epi["_seen"][0] += float(pen); epi["_seen"][1] += 1
                    loss = loss + epi["lam"] * pen
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
    # `quant_out` / `run_out` are out-parameters (the idiom lodo._score already uses) purely so the
    # existing 4-tuple call sites keep working; a caller that passes neither pays nothing.
    enc.eval(); ad.eval()
    cbias = _fit_bias_correction(enc, ad, Z, A, Mt, b, va, masks["val"])
    T, N, nQ = b.X.shape[1], b.X.shape[0], len(score.QUANTILE_LEVELS)
    pred_by_h = {h: np.zeros((N, T), dtype=np.float64) for h in bundles.HORIZONS}
    predmc_by_h = {h: np.zeros((N, T), dtype=np.float64) for h in bundles.HORIZONS}
    if quant_out is not None:
        for h in bundles.HORIZONS:
            quant_out[h] = np.zeros((N, len(te), nQ), dtype=np.float32)
    with torch.no_grad():
        for k, t in enumerate(te):
            out = ad(enc(window_slice(Z, t), A, Mt[:, t])).cpu().numpy()          # [N,H,Q]
            med = out[:, :, MEDIAN_IDX]
            for j, h in enumerate(bundles.HORIZONS):
                pred_by_h[h][:, t + h] = invert_scaler(med[:, j:j + 1], b.scaler)[:, 0]
                predmc_by_h[h][:, t + h] = invert_scaler(med[:, j:j + 1] + cbias[h], b.scaler)[:, 0]
                if quant_out is not None:
                    # per-node scaler is monotone, so inverting each level independently is exact
                    for qi in range(nQ):
                        quant_out[h][:, k, qi] = invert_scaler(out[:, j, qi:qi + 1], b.scaler)[:, 0]
    run_meta = dict(training_regime=training_regime, sampler=sampler,
                    gate_mode=gate_mode, topo_aug=topo_aug)
    gate = gate_spatial_readout(enc, ad, Z, A, Mt, va, device) if gate_read else None
    recs, pernode, perorigin = score_predictions("encoder", name, seed, pred_by_h, b, te,
                                                 run_meta=run_meta)
    # The mean-corrected arm ships ALONGSIDE the median arm under a distinct model name, never in
    # place of it: the median is the calibrated forecast (and the right point for MAE), the corrected
    # one is the RMSE/peak-intensity point. Two rows lets the paper show both; overwriting would
    # silently restate every number already reported.
    mcrecs, _, _ = score_predictions("encoder_mc", name, seed, predmc_by_h, b, te, run_meta=run_meta)
    for r in mcrecs:
        r["bias_c"] = round(cbias[r["horizon"]], 4)
    if run_out is not None:
        run_out.update(enc=enc, ad=ad, te=te, bias_c=cbias)
    return recs + mcrecs, pernode, perorigin, gate


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
    rpath(fname, root=RESULTS, make=True).write_text(json.dumps(records, indent=2))


def write_per_node(pernode, fname):
    """Per-node scores as compact npz (review #7). npz not JSON: dengue is 7165 nodes x 4 horizons,
    so this is a few hundred KB of arrays, not a megabyte of dict text. One node_idx array per
    horizon (the scored set differs by horizon -- constant-window nodes are dropped per horizon)."""
    arrays = {}
    for h, ns in pernode.items():
        idx = sorted(ns)
        arrays[f"h{h}__node_idx"] = np.array(idx, dtype=np.int32)
        arrays[f"h{h}__country"] = np.array([ns[i]["country"] for i in idx])
        arrays[f"h{h}__n_cells"] = np.array([ns[i]["n_cells"] for i in idx], dtype=np.int32)
        for metric in score.METRICS:
            arrays[f"h{h}__{metric}"] = np.array([ns[i][metric] for i in idx], dtype=np.float32)
    np.savez_compressed(rpath(fname, root=RESULTS, make=True), **arrays)


def write_quantiles(quant_by_h, origins, fname):
    """Archive the FULL quantile forecast, not just the median (Review Doc para. 7 / G4).

    Every scoring path takes `[:, :, MEDIAN_IDX]` to get a point forecast and drops the other four
    quantiles at the moment they are produced. WIS, CRPS, empirical coverage, interval width and PIT
    are all implemented and self-checked in score.py, and all of them are UNUSABLE without these
    arrays -- so the calibrated-uncertainty claim has no evidence behind it until a run writes this.
    It is cheap on a run and cannot be reconstructed afterwards without retraining, which is why it
    goes in before any further trunk run rather than after.

    Stored per horizon as [N, K, Q] over the K scored ORIGINS (not the full T timeline): the dense
    [N, T, Q] form is ~20x larger and all but the origin columns are zeros. `origins` and `quantile
    levels` are stored beside the data so a reader never has to guess the axis order. COUNT space,
    matching the point forecasts -- quantiles are equivariant under the monotone per-node scaler, so
    inverting each level independently is exact.
    """
    arrays = {"origins": np.asarray(origins, dtype=np.int32),
              "quantiles": np.asarray(score.QUANTILE_LEVELS, dtype=np.float32)}
    for h, q in quant_by_h.items():
        arrays[f"h{h}__quantiles"] = np.asarray(q, dtype=np.float32)      # [N, K, Q]
    np.savez_compressed(rpath(fname, root=RESULTS, make=True), **arrays)


def write_checkpoint(enc, ad, fname, extra=None):
    """Save the trunk + adapter weights. Without this every result rests on a model that no longer
    exists: `_fit_trunk` keeps the best state in memory and never writes it, so a search for *.pt
    outside baselines/ returned ZERO files before 2026-07-31.

    That blocks more than reproducibility. The Ebola protocol is "freeze the trunk, fit an adapter on
    the support set" -- with no saved trunk there is nothing to freeze, and any re-probe, capacity
    sweep or re-adaptation costs a full retrain. torch.save of ~144k parameters is a fraction of a
    second and a few hundred KB.
    """
    payload = {"encoder": enc.state_dict(),
               "adapter": (ad.state_dict() if ad is not None else None),
               "meta": dict(extra or {})}
    torch.save(payload, rpath(fname, root=RESULTS, make=True))


def write_per_origin(perorigin, fname):
    """Per (origin, country) error sufficient stats (item 6). npz: countries[C], origins[K], and
    per horizon sae/sse [K,C] + n [K,C]. Node-pooled within country, count space, scored (non-
    constant) nodes -- the same population as the headline metric. Reconstructs a country-macro CI
    by resampling origins: per-country MAE=sum(sae)/sum(n), RMSE=sqrt(sum(sse)/sum(n)), macro over C.

    NOTE: this is a CELL-POOLED country-macro (sum over the country's cells), whereas the headline
    aggregate() is NODE-AVERAGED (mean of per-node metrics). They coincide on the dense influenza
    panels (near-equal cells/node) and diverge on dengue. The item-6 CI is therefore on the
    cell-pooled metric -- documented in analysis.py."""
    np.savez_compressed(rpath(fname, root=RESULTS, make=True), **perorigin)


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
    np.savez_compressed(rpath(fname, root=RESULTS, make=True), **gate)


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
        quant, ro = {}, {}
        recs, pernode, perorigin, gate = train_one(name, s, quant_out=quant, run_out=ro, **kw)
        write_records(recs, f"encoder__{name}__seed{s}.json")
        write_per_node(pernode, f"encoder__{name}__seed{s}__pernode.npz")
        write_per_origin(perorigin, f"encoder__{name}__seed{s}__perorigin.npz")
        if gate is not None:
            write_gate(gate, f"encoder__{name}__seed{s}__gate.npz")
        write_quantiles(quant, ro["te"], f"encoder__{name}__seed{s}__quantiles.npz")
        write_checkpoint(ro["enc"], ro["ad"], f"encoder__{name}__seed{s}__ckpt.pt",
                         extra=dict(dataset=name, seed=s, bias_c=ro["bias_c"]))
        print(f"  encoder scored ({name} seed{s}); bias_c={ {h: round(c, 2) for h, c in ro['bias_c'].items()} }")


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


def _bias_selfcheck():
    """The median->mean correction actually corrects, and stays put when there is nothing to correct.

    Model-free: a perfect log-space MEDIAN forecast (z=0) of a right-skewed target must need a
    POSITIVE offset, and applying it must lower count-space RMSE. With sigma=1.2 the RMSE-optimal
    constant is the mean, E[exp(1.2z)]-1 = exp(0.72)-1, i.e. c = 0.72/1.2 = 0.60 -- so this pins the
    fitted value, not just its sign. The degenerate arm is the guard that matters: on a target with
    no skew the fit must return exactly 0, or the correction is a free parameter inflating forecasts
    whether or not a gap exists."""
    rng = np.random.default_rng(0)
    N, K, sigma = 8, 400, 1.2
    scaler = {"mean": np.zeros(N), "std": np.full(N, sigma)}
    tru = np.expm1(rng.normal(0.0, 1.0, size=(N, K)) * sigma)     # counts; log-median is 0
    med, obs = np.zeros((N, K)), np.ones((N, K), dtype=bool)
    c = _best_offset(med, tru, obs, scaler)
    assert 0.4 < c < 0.9, f"expected c near 0.60 for sigma=1.2, got {c}"
    r0 = float(np.sqrt(np.mean((invert_scaler(med, scaler)[obs] - tru[obs]) ** 2)))
    r1 = float(np.sqrt(np.mean((invert_scaler(med + c, scaler)[obs] - tru[obs]) ** 2)))
    assert r1 < r0, f"correction did not reduce RMSE ({r1:.3f} vs {r0:.3f})"
    assert _best_offset(med, np.zeros((N, K)), obs, scaler) == 0.0, \
        "no mean/median gap must fit c=0 -- c=0 is in the grid, so the fit can never be worse"
    print(f"ok  median->mean bias correction: c={c:.2f} (expected 0.60), val RMSE {r0:.2f} -> {r1:.2f}")


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
        _selfcheck(); _perorigin_selfcheck(); _bias_selfcheck(); _ebola_isolation_check(); return
    t0 = time.time()
    if a.smoke:
        _selfcheck(); _perorigin_selfcheck(); _bias_selfcheck()
        quant, ro = {}, {}
        recs, pernode, perorigin, gate = train_one("influenza_japan", 42, epochs=5,
                                                   quant_out=quant, run_out=ro)
        write_records(recs, "encoder__influenza_japan__seed42__smoke.json")
        write_per_node(pernode, "encoder__influenza_japan__seed42__smoke__pernode.npz")
        write_per_origin(perorigin, "encoder__influenza_japan__seed42__smoke__perorigin.npz")
        write_gate(gate, "encoder__influenza_japan__seed42__smoke__gate.npz")
        write_quantiles(quant, ro["te"], "encoder__influenza_japan__seed42__smoke__quantiles.npz")
        write_checkpoint(ro["enc"], ro["ad"], "encoder__influenza_japan__seed42__smoke__ckpt.pt",
                         extra=dict(dataset="influenza_japan", seed=42, bias_c=ro["bias_c"]))
        print(f"smoke done in {time.time()-t0:.0f}s; wrote {len(recs)} records; bias_c={ro['bias_c']}")
    elif a.all:
        for name in bundles.DEV_BUNDLE_NAMES:
            run_dataset(name, epochs=a.epochs)
        print(f"all datasets done in {(time.time()-t0)/60:.1f} min")
    elif a.dataset and not a.seed:
        # one dataset, all 5 seeds + its naive floors. Needed because --all would rerun (and
        # overwrite) every other dataset just to add a new one's ceiling and floor.
        run_dataset(a.dataset, epochs=a.epochs)
        print(f"{a.dataset}: ceiling ({len(SEEDS)} seeds) + naive floors done in "
              f"{(time.time()-t0)/60:.1f} min")
    elif a.dataset and a.seed:
        quant, ro = {}, {}
        recs, pernode, perorigin, gate = train_one(a.dataset, a.seed, epochs=a.epochs,
                                                   quant_out=quant, run_out=ro)
        write_records(recs, f"encoder__{a.dataset}__seed{a.seed}.json")
        write_per_node(pernode, f"encoder__{a.dataset}__seed{a.seed}__pernode.npz")
        write_per_origin(perorigin, f"encoder__{a.dataset}__seed{a.seed}__perorigin.npz")
        write_gate(gate, f"encoder__{a.dataset}__seed{a.seed}__gate.npz")
        write_quantiles(quant, ro["te"], f"encoder__{a.dataset}__seed{a.seed}__quantiles.npz")
        write_checkpoint(ro["enc"], ro["ad"], f"encoder__{a.dataset}__seed{a.seed}__ckpt.pt",
                         extra=dict(dataset=a.dataset, seed=a.seed, bias_c=ro["bias_c"]))
        print(f"done in {time.time()-t0:.0f}s; bias_c={ro['bias_c']}")
    else:
        ap.error("give --dataset+--seed, or --all, or --smoke, or --selfcheck")


if __name__ == "__main__":
    main()

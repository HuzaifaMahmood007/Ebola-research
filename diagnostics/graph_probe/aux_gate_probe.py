"""AUX-GATE PROBE (Milestone 6 run 5, TASK 2): is the district information the encoder discards
FORECAST-RELEVANT at all?

Context. t5_input_energy showed the encoder compresses 14-94% district-specific input energy down
to 3.8-22.4% in h. The aux-district-identity objective bets that forcing h to keep more of it would
help forecasts. That bet is only worth an overnight retrain if the discarded district structure is
forecast-USEFUL, not noise. D2 (Shuffled_Adjacency_2026-09-23) found even the REAL adjacency earns
only 2-4% and only on dengue, which bounds the upside.

The probe, inference + a cheap linear fit, no retrain. Give the FROZEN model district identity
post-hoc by fitting a per-node correction on its own residuals, then see if test error drops. To
separate real district signal from the known lognormal median-vs-mean gap (loop._fit_bias_correction
already fixes that with ONE global scalar per horizon), the per-node correction is always contrasted
against a district-AGNOSTIC (pooled) correction of the SAME form:

  baseline      : model-space median, no correction
  bias_global   : median + c        (one scalar per horizon, pooled over all nodes)   <- lognormal gap
  bias_node     : median + c_i      (one scalar per NODE per horizon)                 <- + district id
  affine_global : a*median + b      (one slope/intercept per horizon, pooled)
  affine_node   : a_i*median + b_i  (per NODE per horizon)

The DISTRICT-IDENTITY signal is (global error - node error): how much a per-node correction beats a
district-agnostic one of equal expressiveness. bias_global vs baseline is the lognormal gap and is
NOT district identity; it is reported only so the reader can see the node gain is on top of it.

Leakage rule. Every correction is fit ONLY on train-phase origins and train-phase observed target
cells. Test-phase origins are never touched during fitting. The checkpoint is frozen: no retrain,
no gradient. The model was trained on the train origins, so a train-fit per-node correction that
transfers to the held-out test origins is the honest generalisation test: stable district structure
transfers, overfit noise does not.

Verdict rule (same as gate-off / shufadj). Per (panel, horizon, metric) there are 5 seeds. For each
seed, district_delta = global_error - node_error (positive = district identity helps). A cell is a
real effect only when mean district_delta over the 5 seeds exceeds its own sd over those seeds AND is
positive. dengue is one origin subsample and its seed count is whatever checkpoints exist.

Metrics: country-macro RMSE and MAE (score.score_bundle, the headline aggregation). Scored on the
same non-constant node population the headline uses.

Scope limits, stated with the finding:
 - results/single/ checkpoints only, not the transfer trunk.
 - Corrections are fit in MODEL space by least squares; the metric is count space (expm1). A
   model-space fit is not count-optimal, but per-node and global are fit identically, so the CONTRAST
   is fair. This measures whether district identity is EXTRACTABLE post-hoc, an upper-ish bound on
   what an aux head fed the same residuals could add; it is not the aux objective itself.
 - dengue uses origins[::4] (t_dengue precedent, 7165 nodes) for both fit and score.
 - A node needs >= MIN_CELLS observed train target cells at a horizon to earn a correction; below
   that it keeps the baseline median (identity). Stated so no node is corrected on 1-2 points.

Run:  conda run -n ebola-train python diagnostics/graph_probe/aux_gate_probe.py
Resumable: writes results/misc/aux_gate_probe.json incrementally; done (panel,seed) cells are skipped.
"""
import os, sys, json, time, datetime
os.chdir(r"F:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import numpy as np, torch
import bundles, score
from models import Adapter, SharedEncoder, MEDIAN_IDX, sparse_from_dense_np
from models.windows import window_slice
from to_schema import invert_scaler

PANELS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states",
          "dengue"]
SEEDS = [42, 52, 62, 72, 82]
MIN_CELLS = 8            # a node needs this many observed train target cells to earn a correction
OUT = "results/misc/aux_gate_probe.json"


def median_on_origins(enc, ad, Z, A, Mt, origins):
    """Model-space median forecast per node per origin per horizon. [N, K, H]."""
    N = Z.shape[0]; K = len(origins); H = len(bundles.HORIZONS)
    med = np.zeros((N, K, H), dtype=np.float64)
    with torch.no_grad():
        for k, t in enumerate(origins):
            out = ad(enc(window_slice(Z, t), A, Mt[:, t])).cpu().numpy()      # [N,H,Q]
            med[:, k, :] = out[:, :, MEDIAN_IDX]
    return med


def fit_and_apply(med_tr, tgt_tr, obs_tr, med_te):
    """Fit five correction variants on TRAIN, return corrected TEST medians (all [N, Kte]).

    med_tr/med_te : [N, Ktr]/[N, Kte] model-space medians at ONE horizon.
    tgt_tr        : [N, Ktr] model-space targets at that horizon (b.y at origin+h).
    obs_tr        : [N, Ktr] bool, observed & in train phase.
    Returns dict variant -> corrected test median [N, Kte]. A node with < MIN_CELLS train cells
    keeps the baseline median for every correction (identity)."""
    N = med_tr.shape[0]
    corr = {"baseline": med_te.copy()}
    # ----- pooled (district-agnostic) fits over all observed train cells of all nodes -----
    m = med_tr[obs_tr]; y = tgt_tr[obs_tr]
    c_glob = float((y - m).mean()) if m.size else 0.0
    if m.size >= 2 and np.ptp(m) > 1e-9:
        a_glob, b_glob = np.polyfit(m, y, 1)
    else:
        a_glob, b_glob = 1.0, c_glob
    corr["bias_global"] = med_te + c_glob
    corr["affine_global"] = a_glob * med_te + b_glob
    # ----- per-node fits -----
    bias_node = med_te.copy()
    affine_node = med_te.copy()
    for i in range(N):
        oi = obs_tr[i]
        if int(oi.sum()) < MIN_CELLS:                       # too few points: keep baseline
            continue
        mi = med_tr[i, oi]; yi = tgt_tr[i, oi]
        ci = float((yi - mi).mean())
        bias_node[i] = med_te[i] + ci
        if np.ptp(mi) > 1e-9:
            ai, bi = np.polyfit(mi, yi, 1)
        else:
            ai, bi = 1.0, ci
        affine_node[i] = ai * med_te[i] + bi
    corr["bias_node"] = bias_node
    corr["affine_node"] = affine_node
    return corr


def score_variant(med_h, h, b, te, ids, ncmap):
    """Invert one horizon's model-space median to counts, place on [N,T], score country-macro."""
    N, T = b.raw.shape
    pred = np.zeros((N, T), dtype=np.float64)
    counts = invert_scaler(med_h, b.scaler)                 # [N, Kte]
    phase_mask = b.masks()["test"].astype(np.uint8)
    mask_h = np.zeros_like(phase_mask)
    for k, t in enumerate(te):
        pred[:, t + h] = counts[:, k]
        mask_h[:, t + h] = phase_mask[:, t + h]
    agg, _ = score.score_bundle(pred, b.raw.astype(np.float64), mask_h, ids, ncmap)
    return {"rmse": agg["rmse"]["country_macro"], "mae": agg["mae"]["country_macro"]}


def load_out():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"script": "diagnostics/graph_probe/aux_gate_probe.py",
            "date": datetime.date.today().isoformat(), "min_train_cells": MIN_CELLS,
            "leakage_rule": "corrections fit on train-phase origins/cells only; frozen checkpoint; "
                            "test origins never seen in fitting",
            "cells": {}}          # key "panel|seed" -> {horizon -> {variant -> {rmse,mae}}}


def main():
    out = load_out()
    for name in PANELS:
        b = bundles.load(name)
        Z = torch.tensor(b.transfer_view(), dtype=torch.float32)
        A = sparse_from_dense_np(b.A_geo)
        Mt = torch.tensor(b.M, dtype=torch.float32)
        y = b.y.astype(np.float64)                          # model-space targets [N,T]
        train_mask = b.masks()["train"].astype(bool)
        ids, ncmap = b.meta["node_ids"], b.group_of()
        tr = list(b.origins(phase="train")); te = list(b.origins(phase="test"))
        sub = 4 if name == "dengue" else 1
        tr = tr[::sub]; te = te[::sub]
        for seed in SEEDS:
            key = f"{name}|{seed}"
            ckpt = f"results/single/encoder__{name}__seed{seed}__ckpt.pt"
            if key in out["cells"] or not os.path.exists(ckpt):
                if not os.path.exists(ckpt):
                    print(f"skip {key}: no checkpoint")
                continue
            t0 = time.time()
            enc = SharedEncoder(gate_mode="learned"); ad = Adapter()
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            enc.load_state_dict(ck["encoder"]); ad.load_state_dict(ck["adapter"])
            enc.eval(); ad.eval()
            med_tr = median_on_origins(enc, ad, Z, A, Mt, tr)   # [N,Ktr,H]
            med_te = median_on_origins(enc, ad, Z, A, Mt, te)   # [N,Kte,H]
            rec = {}
            for j, h in enumerate(bundles.HORIZONS):
                tcol_tr = np.array([t + h for t in tr]); tcol_te = np.array([t + h for t in te])
                tgt_tr = y[:, tcol_tr]
                obs_tr = train_mask[:, tcol_tr]
                corr = fit_and_apply(med_tr[:, :, j], tgt_tr, obs_tr, med_te[:, :, j])
                rec[str(h)] = {v: score_variant(mh, h, b, te, ids, ncmap) for v, mh in corr.items()}
            out["cells"][key] = rec
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2)
            print(f"{key:34s} done in {time.time()-t0:5.1f}s  "
                  f"(rmse h3 base={rec['3']['baseline']['rmse']:.3f} "
                  f"node={rec['3']['bias_node']['rmse']:.3f})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

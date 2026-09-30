"""Rescore the archived encoder forecasts in MODEL SPACE, next to count space. Read-only.

    conda run -n ebola-train python -m diagnostics.model_space_rescore --selfcheck   # seconds
    conda run -n ebola-train python -m diagnostics.model_space_rescore               # about 1-3 min

WHY. Every headline in the project is scored in counts (score.py), but the model trains on
log1p-then-z-score targets (to_schema.py:183-186). Counts weight a district by its size, so a
count-space win can be two big districts, and a count-space loss can be one outbreak. Model space
weights every district's relative error equally. This asks whether the headline verdicts depend on
which space you score in. Only the space changes: same archives, same cells, same node set, same
country-macro aggregation (per node, mean within country, mean over countries).

THE TRANSFORM. forward(c) = (log1p(c) - mean) / std with the run's own bundle scaler (b.scaler,
which is what every runner inverted with: train/loop.py:275, train/lodo.py:357). The inverse
(to_schema.py:189-192) is expm1 with NO clip, so predicted counts can sit in (-1, 0) and about a
third of dengue medians do. So predictions are mapped back WITHOUT a clip. apply_scaler and
verify_v2_protocol._model_mse clip at 0 and would silently move those cells; do not swap them in.
Truth is the bundle's y, asserted equal to forward(clip(raw, 0)) on every scored cell.

THE GATE. Before any model-space number is kept, the same archive is rescored in COUNT space here
and must reproduce the scored JSON's country_macro RMSE and MAE (and n_nodes) to 1e-6 relative.
That is what makes "same cells, same aggregation" a checked fact. A family that fails is dropped
and reported, never published. The naive floors are gated against results/naive/ the same way.

THE INTERVAL IS NOT THE PROJECT'S. Each comparison is a 5-seed mean with a two-sided 95% t interval
(t = 2.776, df 4) on the seed-paired differences. It prices seed variance only. It is NOT:
  * the LDO3 headline (1 / 10 / 25 over 36): that is an origin bootstrap on the cell-pooled macro of
    5-seed ENSEMBLES (diagnostics/seed_ensemble.py compare_transfer). Different instrument.
  * the Ebola interval (ebola_ci.py): a stratified district bootstrap. Different instrument.
Nothing here re-scores the frozen Ebola pre-registration. It reads the archives it wrote once.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import bundles
from results_paths import rpath

SEEDS = (42, 52, 62, 72, 82)
HORIZONS = bundles.HORIZONS
DEV = ("dengue", "influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states")
ARMS = ("ebola_L12", "ebola_L20")
LEVELS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
MED = 2
T_CRIT = 2.776                                 # two-sided 95%, df = 4 (n = 5 seeds)
TOL = 1e-6
METRICS = ("rmse", "mae", "pinball")
GATED = ("rmse", "mae")                        # the JSON carries these; pinball has no scored record
# (family, archive prefix, panels, eval phase)
FAMILIES = (("single", "encoder", DEV, "test"),
            ("ldo3", "encoder_ldo3", DEV, "test"),
            ("ebola_fewshot", "encoder_ebola", ARMS, "query"),
            ("ebola_zeroshot", "encoder_ebola_zeroshot", ARMS, "query"))
# LDO3_Results.md:246: COVID h10/h15 are "not attributable (regime break)" and out of the 36.
NOT_ATTRIBUTABLE = {("covid_us-states", 10), ("covid_us-states", 15)}
# float32 precision near -1. An archived count c in (-1, 1) is rounded by at most ULP = 6e-8, so the
# model-space value it carries is off by at most ULP / ((1 + c) * std). Refuse an archive whose worst
# cell exceeds MAX_Z_ERR, a thousandth of a z unit (the metrics live at 0.1 to 1 z). A fixed cutoff on
# 1 + c was tried first and tripped on real counts down to -0.9965 whose actual error is ~1e-5 z.
ULP = 6e-8
MAX_Z_ERR = 1e-3


class GateFail(Exception):
    pass


# --------------------------------------------------------------------------- transforms
def forward(c, sc):
    """Counts -> model space, NO clip. Defined for c > -1, which every archived value is."""
    return (np.log1p(np.asarray(c, np.float64)) - sc["mean"][:, None]) / sc["std"][:, None]


def inverse32(z, sc):
    """to_schema.invert_scaler exactly as the runners called it (float32 in, float32 out)."""
    z = np.asarray(z, np.float32)
    return np.expm1(z * sc["std"][:, None].astype(np.float32) + sc["mean"][:, None].astype(np.float32))


# --------------------------------------------------------------------------- scoring
def node_stats(P, Y, W, Q=None):
    """Per-node rmse, mae (of point P) and mean pinball (of quantiles Q) over cells W. [N] each.

    Vectorised score.per_node_scores: mean over the node's scored cells. The gate below is what
    proves this equals score.py on real data."""
    n = W.sum(1)
    nn = np.maximum(n, 1)
    e = np.where(W, P - Y, 0.0)
    out = {"rmse": np.sqrt((e * e).sum(1) / nn), "mae": np.abs(e).sum(1) / nn}
    if Q is not None:
        err = Y[..., None] - np.sort(Q, axis=-1)                 # score.py sorts crossed quantiles
        pin = np.maximum(LEVELS * err, (LEVELS - 1.0) * err).mean(-1)
    else:                                                        # a point forecast as a degenerate
        pin = 0.5 * np.abs(P - Y)                                # quantile set: mean pinball = |e|/2
    out["pinball"] = np.where(W, pin, 0.0).sum(1) / nn
    return out


def scored_nodes(Yc, W):
    """score.py's node set: at least one cell, and truth not constant (std >= 1e-8) in COUNTS."""
    n = W.sum(1)
    nn = np.maximum(n, 1)
    mu = np.where(W, Yc, 0.0).sum(1) / nn
    sd = np.sqrt(np.where(W, (Yc - mu[:, None]) ** 2, 0.0).sum(1) / nn)
    return (n > 0) & (sd >= 1e-8)


def macro(v, keep, country):
    """score.aggregate country_macro: mean within country, then mean over countries."""
    per = [v[keep & (country == c)].mean() for c in np.unique(country[keep])]
    return float(np.mean(per)) if per else float("nan")


class Panel:
    """One bundle's scored cells at each horizon, truth in both spaces, and the floors."""

    def __init__(self, name, phase, origins):
        b = bundles.load(name)
        self.name, self.origins = name, np.asarray(origins, np.int64)
        self.sc = {k: np.asarray(v, np.float64) for k, v in b.scaler.items()}
        grp = b.group_of()
        self.country = np.array([grp[n] for n in b.meta["node_ids"]])
        raw, y = b.raw.astype(np.float64), b.y.astype(np.float64)
        pm = b.masks()[phase].astype(bool)
        # floors: persistence = raw at the origin; the mean floor is train_mean (dev) or support_mean
        # (Ebola), 0 for a node with no history, as in train/loop.py:303 and train/ebola.py:256.
        hist = b.masks()["train" if phase == "test" else "support"].astype(bool)
        hmean = np.array([raw[i, hist[i]].mean() if hist[i].any() else 0.0 for i in range(len(raw))])
        self.mean_floor = "train_mean" if phase == "test" else "support_mean"
        self.pers, self.hmean = raw[:, self.origins], hmean    # raw >= 0 on every bundle: no clip needed
        self.h = {}
        for h in HORIZONS:
            cols = self.origins + h
            W, Yc, Ym = pm[:, cols], raw[:, cols], y[:, cols]
            fwd = forward(np.clip(Yc, 0, None), self.sc)
            gap = float(np.abs(fwd - Ym)[W].max()) if W.any() else 0.0
            if gap > 1e-4:
                raise GateFail(f"{name} h{h}: bundle y != forward(raw) on scored cells, max gap {gap:.2e}")
            self.h[h] = dict(W=W, Yc=Yc, Ym=Ym, keep=scored_nodes(Yc, W), y_gap=gap)

    def floor(self, fl):
        return self.pers if fl == "persistence" else np.repeat(self.hmean[:, None], len(self.origins), 1)

    def score(self, h, Pc, Qc=None):
        """{space: {metric: macro}} for a count-space point Pc [N,K] (and quantiles Qc [N,K,5])."""
        d = self.h[h]
        out = {}
        for space, P, Q, Y in (("count", Pc, Qc, d["Yc"]),
                               ("model", forward(Pc, self.sc),
                                None if Qc is None else forward(Qc.reshape(len(Qc), -1), self.sc)
                                .reshape(Qc.shape), d["Ym"])):
            ns = node_stats(P, Y, d["W"], Q)
            out[space] = {m: macro(ns[m], d["keep"], self.country) for m in METRICS}
        out["n_nodes"] = int(d["keep"].sum())
        return out


# --------------------------------------------------------------------------- records and gate
def json_macro(fname, model_pred):
    """{(h, metric): (country_macro, n_nodes)} from one scored record file."""
    recs = [r for r in json.load(open(rpath(fname), encoding="utf-8")) if model_pred(r["model"])]
    names = {r["model"] for r in recs}
    if len(names) != 1:
        raise GateFail(f"{fname}: expected one model name, got {sorted(names)}")
    return {(r["horizon"], r["metric"]): (float(r["country_macro"]), int(r["n_nodes"])) for r in recs}


def gate(sc, want, label):
    """Count-space rebuild must equal the scored JSON. Raises GateFail on the first mismatch."""
    for h, s in sc.items():
        for m in GATED:
            wv, wn = want[(h, m)]
            gv = s["count"][m]
            if abs(gv - wv) > TOL * max(1.0, abs(wv)) or s["n_nodes"] != wn:
                raise GateFail(f"{label} h{h} {m}: rebuilt {gv:.9g} (n={s['n_nodes']}) != "
                               f"scored {wv:.9g} (n={wn})")


def load_archive(prefix, ds, seed):
    z = np.load(rpath(f"{prefix}__{ds}__seed{seed}__quantiles.npz"))
    if not np.allclose(z["quantiles"], LEVELS):
        raise GateFail(f"{prefix} {ds} seed{seed}: quantile levels {z['quantiles']}")
    return z


def run_family(prefix, ds, phase, panels, info):
    """Score all five seeds of one (family, panel). Returns {seed: {h: score}} or raises GateFail."""
    out = {}
    for s in SEEDS:
        z = load_archive(prefix, ds, s)
        o = z["origins"]
        if ds not in panels:
            panels[ds] = Panel(ds, phase, o)
        pan = panels[ds]
        if not np.array_equal(o, pan.origins):
            raise GateFail(f"{prefix} {ds} seed{s}: origins differ from the panel's")
        res = {}
        for h in HORIZONS:
            Q = z[f"h{h}__quantiles"].astype(np.float64)
            info["min_1pq"] = min(info["min_1pq"], float(1.0 + Q.min()))
            zerr = float((ULP / ((1.0 + Q.min(-1)) * pan.sc["std"][:, None])).max())
            info["max_z_err"] = max(info["max_z_err"], zerr)
            if zerr > MAX_Z_ERR:
                raise GateFail(f"{prefix} {ds} seed{s} h{h}: float32 error bound {zerr:.1e} z > "
                               f"{MAX_Z_ERR:g}; count {Q.min():.6f} too close to -1 to carry model space")
            info["neg_median_cells"] += int((Q[..., MED][pan.h[h]["W"]] < 0).sum())
            info["median_cells"] += int(pan.h[h]["W"].sum())
            res[h] = pan.score(h, Q[..., MED], Q)
            if phase == "query":                                 # Ebola is tiny: keep for the zero split
                info["ebola_med"][(prefix, ds, s, h)] = Q[..., MED]
        want = json_macro(f"{prefix}__{ds}__seed{s}.json", lambda m: not m.endswith("_mc"))
        gate(res, want, f"{prefix} {ds} seed{s}")
        out[s] = res
    return out


def floor_scores(pan, ds):
    """{floor: {h: score}}, gated against results/naive/naive__<ds>.json."""
    out = {}
    for fl in ("persistence", pan.mean_floor):
        res = {h: pan.score(h, pan.floor(fl)) for h in HORIZONS}
        gate(res, json_macro(f"naive__{ds}.json", lambda m, fl=fl: m == fl), f"naive {ds} {fl}")
        out[fl] = res
    return out


# --------------------------------------------------------------------------- comparisons
def paired(a, b):
    """a, b: 5 values each (b may be one constant). Negative diff = a better (lower)."""
    a, b = np.asarray(a, float), np.broadcast_to(np.asarray(b, float), (len(a),))
    d = a - b
    if len(d) != 5:
        raise GateFail(f"paired t needs 5 seeds, got {len(d)}")
    m, half = float(d.mean()), T_CRIT * float(d.std(ddof=1)) / np.sqrt(5)
    ref = float(b.mean())
    v = "win" if m + half < 0 else ("loss" if m - half > 0 else "noise")
    return dict(diff=m, lo=m - half, hi=m + half, pct=100 * m / ref, lo_pct=100 * (m - half) / ref,
                hi_pct=100 * (m + half) / ref, a=float(a.mean()), b=ref, verdict=v)


def compare(A, B, cells, label):
    """A, B: {ds: {seed: {h: score}}} (B may be a floor {ds: {h: score}}). Both spaces per cell."""
    rows = []
    for ds, h, m in cells:
        row = dict(panel=ds, h=h, metric=m, label=label)
        seeded = set(B[ds]) == set(SEEDS)                        # else a deterministic floor {h: ...}
        for space in ("count", "model"):
            av = [A[ds][s][h][space][m] for s in SEEDS]
            bv = [B[ds][s][h][space][m] for s in SEEDS] if seeded else B[ds][h][space][m]
            row[space] = paired(av, bv)
        rows.append(row)
    return rows


def print_rows(title, rows, win="win", loss="loss", note=None):
    print(f"\n{'=' * 108}\n{title}\n{'=' * 108}")
    if note:
        print(f"  {note}")
    print(f"  {'panel':21s} {'h':>3} {'metric':7s} | {'COUNT d%':>8} {'95% t CI %':>19} {'verdict':7s}"
          f" | {'MODEL d%':>8} {'95% t CI %':>19} {'verdict':7s} | agree")
    name = {"win": win, "loss": loss, "noise": "noise"}
    for r in rows:
        c, mo = r["count"], r["model"]
        ag = "yes" if c["verdict"] == mo["verdict"] else "NO"
        print(f"  {r['panel']:21s} {r['h']:>3} {r['metric']:7s} | {c['pct']:+8.1f} "
              f"[{c['lo_pct']:+7.1f}, {c['hi_pct']:+7.1f}] {name[c['verdict']]:7s} | {mo['pct']:+8.1f} "
              f"[{mo['lo_pct']:+7.1f}, {mo['hi_pct']:+7.1f}] {name[mo['verdict']]:7s} | {ag}")
    for m in sorted({r["metric"] for r in rows}):
        for sp in ("count", "model"):
            t = {k: sum(1 for r in rows if r["metric"] == m and r[sp]["verdict"] == k)
                 for k in ("win", "noise", "loss")}
            print(f"  tally {m:7s} {sp:5s}: {name['win']} {t['win']}, noise {t['noise']}, "
                  f"{name['loss']} {t['loss']}  (of {sum(t.values())})")
        dis = [f"{r['panel']} h{r['h']}" for r in rows
               if r["metric"] == m and r["count"]["verdict"] != r["model"]["verdict"]]
        print(f"  {m:7s} disagreements ({len(dis)}): {', '.join(dis) if dis else 'none'}")


def tally(rows, keep=lambda r: True):
    t = {sp: {k: 0 for k in ("win", "noise", "loss")} for sp in ("count", "model")}
    for r in rows:
        if keep(r):
            for sp in t:
                t[sp][r[sp]["verdict"]] += 1
    return t


def zero_split(panels, med, arms):
    """Where does the model-space Ebola MAE come from? Split the scored cells by whether the TRUE
    count is 0. CELL-POOLED abs error, 5-seed mean: a diagnostic of the mechanism, not the headline
    estimand (which is node-averaged country-macro) and never to be quoted against it."""
    print(f"\n{'=' * 108}\nEBOLA ZERO-CELL SPLIT  (cell-pooled mean abs error, 5-seed mean; "
          f"diagnostic only, not the headline estimand)\n{'=' * 108}")
    print(f"  {'arm':10s} {'h':>3} {'zero%':>6} | {'space':5s} {'cells':6s} {'pers':>9} {'few-shot':>9} "
          f"{'zero-shot':>9}")
    out = {}
    for a in arms:
        pan = panels[a]
        for h in HORIZONS:
            d = pan.h[h]
            W = d["W"] & d["keep"][:, None]
            zero = W & (d["Yc"] == 0)
            for sp in ("count", "model"):
                tf = (lambda x: x) if sp == "count" else (lambda x: forward(x, pan.sc))
                Y = d["Yc"] if sp == "count" else d["Ym"]
                e_p = np.abs(tf(pan.pers) - Y)
                e_f = np.mean([np.abs(tf(med[("encoder_ebola", a, s, h)]) - Y) for s in SEEDS], 0)
                e_z = np.mean([np.abs(tf(med[("encoder_ebola_zeroshot", a, s, h)]) - Y) for s in SEEDS], 0)
                for lab, msk in (("y==0", zero), ("y>0", W & ~zero)):
                    r = dict(pers=float(e_p[msk].mean()), fewshot=float(e_f[msk].mean()),
                             zeroshot=float(e_z[msk].mean()), n=int(msk.sum()))
                    out[f"{a}/h{h}/{sp}/{lab}"] = r
                    print(f"  {a:10s} {h:>3} {100 * zero.sum() / W.sum():6.1f} | {sp:5s} {lab:6s} "
                          f"{r['pers']:9.3f} {r['fewshot']:9.3f} {r['zeroshot']:9.3f}   n={r['n']}")
            pz = float((pan.pers[zero] == 0).mean())
            out[f"{a}/h{h}/pers_exact_on_zero"] = pz
            print(f"  {'':10s} {'':>3} {'':6s}   persistence predicts exactly 0 on {100 * pz:.1f}% of y==0 cells")
    return out


# --------------------------------------------------------------------------- main
def main_run():
    t0 = time.time()
    panels, scores, gates = {}, {}, {}
    info = {"min_1pq": np.inf, "max_z_err": 0.0, "neg_median_cells": 0, "median_cells": 0,
            "ebola_med": {}}
    for fam, prefix, dss, phase in FAMILIES:
        scores[fam] = {}
        for ds in dss:
            try:
                scores[fam][ds] = run_family(prefix, ds, phase, panels, info)
                gates[f"{fam}/{ds}"] = "pass"
            except GateFail as e:
                gates[f"{fam}/{ds}"] = f"FAIL: {e}"
                print(f"  GATE FAIL {fam}/{ds}: {e}  -> family/panel dropped")
    floors, fgates = {}, {}
    for ds, pan in panels.items():
        try:
            floors[ds] = floor_scores(pan, ds)
            fgates[ds] = "pass"
        except GateFail as e:
            fgates[ds] = f"FAIL: {e}"
            print(f"  GATE FAIL floors {ds}: {e}")

    print(f"\nSANITY GATE (count-space rebuild vs scored JSON, rmse+mae country_macro and n_nodes, "
          f"rel tol {TOL:g}; y == forward(raw) on scored cells):")
    for k, v in {**gates, **{f'naive/{d}': v for d, v in fgates.items()}}.items():
        print(f"  {k:36s} {v}")
    ygap = max(p.h[h]["y_gap"] for p in panels.values() for h in HORIZONS)
    print(f"  max |y - forward(raw)| on scored cells: {ygap:.2e};  min(1 + archived count) = "
          f"{info['min_1pq']:.4f} (log1p defined), worst float32 error bound {info['max_z_err']:.1e} z;"
          f"\n  negative medians on scored cells: "
          f"{info['neg_median_cells']:,} of {info['median_cells']:,} "
          f"({100 * info['neg_median_cells'] / max(info['median_cells'], 1):.1f}%)")

    ok = lambda fam, ds: gates.get(f"{fam}/{ds}") == "pass" and fgates.get(ds) == "pass"
    pers = {ds: floors[ds]["persistence"] for ds in floors}
    cells = lambda dss, ms: [(ds, h, m) for ds in dss for h in HORIZONS for m in ms]
    res = {}

    # (a) single-disease encoder vs persistence
    dss = [d for d in DEV if ok("single", d)]
    res["a_single_vs_persistence"] = compare(scores["single"], pers, cells(dss, METRICS), "a")
    print_rows("(a) SINGLE-DISEASE encoder vs PERSISTENCE  (d% < 0 = encoder better; win = interval "
               "below zero)", res["a_single_vs_persistence"],
               note="persistence pinball = its degenerate quantile set, i.e. MAE/2 per cell (WIS-style)")

    # (b) transfer cost: LDO3 adapted vs single, seed-paired
    dss = [d for d in DEV if ok("single", d) and ok("ldo3", d)]
    rows = compare(scores["ldo3"], scores["single"], cells(dss, METRICS), "b")
    for r in rows:
        r["attributable"] = (r["panel"], r["h"]) not in NOT_ATTRIBUTABLE
    res["b_ldo3_vs_single"] = rows
    print_rows("(b) TRANSFER COST: LDO3 adapted vs SINGLE-DISEASE, seed-paired  (win = transfer better)",
               rows, win="better", loss="worse",
               note="COVID h10/h15 are shown but are 'not attributable' (LDO3_Results.md:246)")
    rm = lambda r: r["attributable"] and r["metric"] in ("rmse", "mae")
    t36 = tally(rows, rm)
    print(f"\n  36-CELL TALLY (rmse+mae, attributable), this script's seed-paired t on per-seed values:")
    for sp in ("count", "model"):
        print(f"    {sp:5s}: better {t36[sp]['win']}, within noise {t36[sp]['noise']}, "
              f"worse {t36[sp]['loss']}  (of {sum(t36[sp].values())})")
    print("    documented (LDO3_Results.md:9): better 1, within noise 10, worse 25. That used an origin "
          "bootstrap on\n    the CELL-POOLED macro of 5-seed ENSEMBLES. Different instrument; a "
          "mismatch in counts is expected, not a defect.")
    res["b_tally_36"] = t36

    # (c) Ebola, both arms: zero-shot vs few-shot, and each vs persistence
    arms = [a for a in ARMS if ok("ebola_fewshot", a) and ok("ebola_zeroshot", a)]
    res["c_ebola_zs_vs_fs"] = compare(scores["ebola_zeroshot"], scores["ebola_fewshot"],
                                      cells(arms, METRICS), "c")
    fence = ("seed-paired t, seed variance only. NOT the ebola_ci.py district bootstrap, NOT a "
             "pre-registered result, not a rescore of it.")
    print_rows("(c1) EBOLA zero-shot vs few-shot  (win = zero-shot better)", res["c_ebola_zs_vs_fs"],
               win="zs", loss="fs", note=fence)
    for fam, lab in (("ebola_fewshot", "few-shot"), ("ebola_zeroshot", "zero-shot")):
        k = f"c_ebola_{lab.replace('-', '')}_vs_persistence"
        res[k] = compare(scores[fam], pers, cells(arms, METRICS), "c")
        print_rows(f"(c) EBOLA {lab} vs PERSISTENCE  (win = encoder better)", res[k], note=fence)

    # point-mean zero-shot vs few-shot count, the CLAUDE.md "31 of 32" observation, both spaces
    pm = {sp: sum(1 for r in res["c_ebola_zs_vs_fs"] if r["metric"] in ("rmse", "mae")
                  and r[sp]["diff"] < 0) for sp in ("count", "model")}
    n_pm = sum(1 for r in res["c_ebola_zs_vs_fs"] if r["metric"] in ("rmse", "mae"))
    print(f"\n  zero-shot point mean below few-shot (rmse+mae, no interval): count {pm['count']} of "
          f"{n_pm}, model {pm['model']} of {n_pm}")
    res["c_zs_point_wins_rmse_mae"] = dict(pm, of=n_pm)
    res["c_ebola_zero_split"] = zero_split(panels, info["ebola_med"], arms)

    runtime = round(time.time() - t0, 1)
    raw = {fam: {ds: {str(s): {str(h): v for h, v in sv.items()} for s, sv in d.items()}
                 for ds, d in fd.items()} for fam, fd in scores.items()}
    fl = {ds: {f: {str(h): v for h, v in fv.items()} for f, fv in d.items()} for ds, d in floors.items()}
    payload = dict(
        what="model-space (log1p then z-score, bundle scaler, no clip) rescore of archived medians, "
             "beside count space; country_macro aggregation; seed-paired t interval (t=2.776, n=5)",
        caveats=["model space weights every district's relative error equally; counts weight big "
                 "districts", "Ebola uses one pooled scale, so its model-space units are not "
                 "comparable to the per-node dev panels", "seed-paired t prices seed variance only; "
                 "not the LDO3 origin bootstrap and not the ebola_ci.py district bootstrap"],
        gates=gates, floor_gates=fgates, max_y_gap=ygap, min_1pq=info["min_1pq"],
        max_float32_z_err=info["max_z_err"],
        negative_median_cells=[info["neg_median_cells"], info["median_cells"]],
        runs=raw, floors=fl, comparisons=res, runtime_s=runtime)
    p = rpath("model_space_rescore.json", make=True)
    p.write_text(json.dumps(payload, indent=1, default=float))
    print(f"\nwrote {p}   runtime {runtime}s")
    return 0 if all(v == "pass" for v in {**gates, **fgates}.values()) else 1


# --------------------------------------------------------------------------- selfcheck
def _selfcheck():
    rng = np.random.default_rng(0)
    # 1. round trip through the runners' own float32 inverse, including counts in (-1, 0)
    N, K = 40, 200
    sc = {"mean": rng.uniform(0.5, 6.0, N), "std": rng.uniform(0.3, 2.0, N)}
    sc32 = {k: v.astype(np.float32).astype(np.float64) for k, v in sc.items()}
    z = rng.normal(0, 2, (N, K)).astype(np.float32)
    z[:, :20] = (-sc32["mean"][:, None] / sc32["std"][:, None] - rng.uniform(0.05, 0.8, (N, 20)))
    c = inverse32(z, sc32)
    neg = (c < 0) & (c > -1)
    assert neg.sum() >= N * 10, f"control void: only {neg.sum()} counts landed in (-1, 0)"
    err = np.abs(forward(c, sc32) - z.astype(np.float64))
    # The bound main() enforces, ULP / ((1 + c) * std), must hold on EVERY cell, including the near -1
    # ones (1 + c down to ~1e-5 here). The 1e-5 slack covers float32 rounding of z*std+mean itself.
    bound = ULP / ((1.0 + c) * sc32["std"][:, None]) + 1e-5
    assert (err <= bound).all(), f"float32 bound broken: worst err/bound {(err / bound).max():.2f}"
    near = (1.0 + c) < 1e-2
    assert near.any() and err[near].max() > 1e-5, "control void: no near -1 cells drawn"
    # 2. PLANTED clip at 0 (what apply_scaler does) must break it on exactly those cells
    clipped = np.abs(forward(np.clip(c, 0, None), sc32) - z)
    assert clipped[neg].min() > 1e-3 and clipped[~neg & (c > 0)].max() < 1e-4, "clip control void"
    print(f"ok  forward(inverse32(x)) == x within the float32 bound ULP/((1+c)*std) on all {err.size} "
          f"cells, incl. {int(neg.sum())} counts in (-1,0); worst err {err.max():.1e} at 1+c = "
          f"{(1.0 + c).min():.1e}, and {err[~near].max():.1e} away from -1")
    print(f"ok  a planted clip at 0 moves every (-1,0) cell: min {clipped[neg].min():.1e} z, "
          f"median {np.median(clipped[neg]):.2f} z, while positive cells stay put")
    # 3. country macro, not pooled
    v, ctry = np.array([1., 3., 10., 20., 100.]), np.array(list("aabbc"))
    assert macro(v, np.ones(5, bool), ctry) == np.mean([2., 15., 100.]) != v.mean()
    print("ok  macro = mean within country then over countries, not a pooled mean")
    # 4. vectorised node stats and pinball == score.py on a real run, and the gate bites
    import score
    ds, s = "influenza_japan", 42
    zq = load_archive("encoder", ds, s)
    pan = Panel(ds, "test", zq["origins"])
    d = pan.h[3]
    Q = zq["h3__quantiles"].astype(np.float64)
    ns = node_stats(Q[..., MED], d["Yc"], d["W"], Q)
    for i in np.flatnonzero(d["keep"])[:10]:
        m = d["W"][i]
        assert np.isclose(ns["rmse"][i], score._rmse(Q[i, m, MED], d["Yc"][i, m]), rtol=1e-12)
        assert np.isclose(ns["pinball"][i], score.crps(Q[i, m], d["Yc"][i, m]).mean() / 2, rtol=1e-12)
    res = {h: pan.score(h, zq[f"h{h}__quantiles"][..., MED].astype(np.float64)) for h in HORIZONS}
    want = json_macro(f"encoder__{ds}__seed{s}.json", lambda m: not m.endswith("_mc"))
    gate(res, want, "selfcheck")
    # MUTATION A: origin axis shifted by one step (forecast for origin k scored at k+1) must fail.
    # ponytail: a +1-count nudge on one cell moves japan's macro by ~3e-7 relative, under TOL, so it
    # is not used; the gate resolves 1e-6 relative, which catches a wrong cell set, not a 1-count typo.
    shifted = {h: pan.score(h, np.roll(zq[f"h{h}__quantiles"][..., MED].astype(np.float64), 1, axis=1))
               for h in HORIZONS}
    # MUTATION B: the wrong quantile (q05 for the median) must fail the gate
    wrong = {h: pan.score(h, zq[f"h{h}__quantiles"][..., 0].astype(np.float64)) for h in HORIZONS}
    for lab, r in (("origin axis shifted by one", shifted), ("q05 read as median", wrong)):
        try:
            gate(r, want, lab)
        except GateFail:
            continue
        raise AssertionError(f"MUTATION survived the gate: {lab}")
    print(f"ok  node stats + pinball == score.py on {ds} seed{s}; gate passes on the real archive and "
          f"FAILS on both planted mutations (origin axis shifted by one; q05 read as the median)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    return _selfcheck() if a.selfcheck else main_run()


if __name__ == "__main__":
    sys.exit(main())

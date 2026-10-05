"""encoder_dengue_subset.py -- the encoder's SAVED dengue forecasts rescored on the classical subset.

The classical baselines (run_classical.py) score dengue on the 1/3 stratified node subset
(export_baseline._kept_indices, "2392/7165", 2051 scored). The encoder record scores all 6161. This
puts the encoder on the classical subset with no retraining: read the point forecast (the median
slot, exactly what train/loop.py scored) out of results/single/encoder__dengue__seed*__quantiles.npz
and push it through the SAME call that wrote the classical records (run_classical.score_records).

Three gates, all must pass before a number is written:
  A  full-panel rescore through train.loop.score_predictions reproduces the archived JSON
     (model "encoder") rmse and mae country_macro to 1e-6 relative, per seed and horizon.
  B  the subset is the classical subset: node_subset, n_nodes, n_countries match every arima,
     sarima and gbm record per horizon; kept node ids equal the GNN export meta when it is on disk.
  C  no subset persistence record exists anywhere, so: (C1) full-panel persistence rebuilt through
     train.loop reproduces results/naive/naive__dengue.json to 1e-6; (C2) subset persistence and the
     subset encoder both match an independent route, the archived float32 per-node scores restricted
     to the subset and macro-averaged (tolerance 1e-5, float32 bound).

Verdict rule: two-sided 95% t at n=5 (t=2.776) on encoder minus comparator, paired by seed for GBM,
one-sample against the deterministic ARIMA/SARIMA/persistence value. Same rule as Table 2 of the
Milestone 6 report; --selfcheck proves it reproduces that table's 16-cell tallies.

    conda run -n ebola-train python -m diagnostics.encoder_dengue_subset
    conda run -n ebola-train python -m diagnostics.encoder_dengue_subset --selfcheck

Writes results/misc/encoder_dengue_subset.json (NOT results/single/, whose glob feeds capacity_probe).
# ponytail: encoder_mc is not archived as quantiles, so it is not rescored here.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import bundles
import export_baseline
import run_classical
from train import loop

BASE = Path(__file__).resolve().parent.parent
SINGLE, BL, NAIVE = BASE / "results/single", BASE / "results/baselines", BASE / "results/naive"
OUT = BASE / "results/misc/encoder_dengue_subset.json"
EXPORT_META = BASE / "baselines/_exported/dengue/meta.json"
SEEDS, H = (42, 52, 62, 72, 82), bundles.HORIZONS
T95, REL_TOL, F32_TOL = 2.776, 1e-6, 1e-5
METRICS = ("rmse", "mae", "nrmse")
PANELS4 = ("influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states")
# Milestone 6 report Table 2, (lower, none, higher) for the classical model vs the comparator
TABLE2 = {("gbm", "encoder", "rmse"): (0, 2, 14), ("gbm", "encoder", "mae"): (0, 3, 13),
          ("sarima", "encoder", "rmse"): (2, 4, 10), ("sarima", "encoder", "mae"): (2, 3, 11),
          ("gbm", "persistence", "rmse"): (8, 0, 8), ("gbm", "persistence", "mae"): (8, 0, 8),
          ("sarima", "persistence", "rmse"): (13, 0, 3), ("sarima", "persistence", "mae"): (13, 0, 3)}


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def _cm(recs, model, metric, h):
    """{field: value} of the one record matching (model, metric, h)."""
    hit = [r for r in recs if r["model"] == model and r["metric"] == metric and r["horizon"] == h]
    assert len(hit) == 1, (model, metric, h, len(hit))
    return hit[0]


# --------------------------------------------------------------------------- #
# forecasts
# --------------------------------------------------------------------------- #
def median_preds(seed, te, T, shift=0):
    """{h: [N,T]} count-space point forecast from the quantile archive, placed at t+h exactly as
    loop.py did. Median slot UNSORTED, which is what loop.py scored. `shift` is the selfcheck plant."""
    z = np.load(SINGLE / f"encoder__dengue__seed{seed}__quantiles.npz")
    assert np.array_equal(z["origins"], np.asarray(te)), f"seed{seed}: archive origins != test origins"
    mi = int(np.where(np.isclose(z["quantiles"], 0.5))[0][0])
    out = {}
    for h in H:
        q = z[f"h{h}__quantiles"]
        assert q.shape[:2] == (T[0], len(te)), f"seed{seed} h{h}: archive shape {q.shape}"
        p = np.zeros(T, dtype=np.float64)
        p[:, np.asarray(te) + h + shift] = q[:, :, mi]
        out[h] = p
    return out


def persistence(b, te, shift=0):
    """loop.py's own persistence; `shift` (selfcheck plant) reads y_{t-shift} instead of y_t."""
    if not shift:
        return loop.naive_predictions(b, te)[0]["persistence"]
    out = {h: np.zeros(b.raw.shape) for h in H}
    for t in te:
        for h in H:
            out[h][:, t + h] = b.raw[:, t - shift]
    return out


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def gate_a(b, te, seed, pred):
    """Full-panel rescore vs the archived JSON, model 'encoder'. Returns (ok, worst_rel, rows)."""
    recs, _, _ = loop.score_predictions("encoder", "dengue", seed, pred, b, te)
    arch = json.loads((SINGLE / f"encoder__dengue__seed{seed}.json").read_text())
    worst, rows, ok = 0.0, {}, True
    for h in H:
        for m in METRICS:
            mine, theirs = _cm(recs, "encoder", m, h), _cm(arch, "encoder", m, h)
            r = _rel(mine["country_macro"], theirs["country_macro"])
            rows[f"h{h}_{m}"] = dict(rescored=mine["country_macro"], archived=theirs["country_macro"],
                                     rel=r, n_nodes=mine["n_nodes"])
            if m in ("rmse", "mae"):        # nrmse was backfilled from float32 per-node rmse; reported only
                worst = max(worst, r)
                ok &= r <= REL_TOL and mine["n_nodes"] == theirs["n_nodes"]
    return ok, worst, rows


def classical_records():
    out = {}
    for m in ("arima", "sarima"):
        out[m] = {h: json.loads((BL / f"{m}__dengue__h{h}__seedNA.json").read_text()) for h in H}
    out["gbm"] = {(h, s): json.loads((BL / f"gbm__dengue__h{h}__seed{s}.json").read_text())
                  for h in H for s in SEEDS}
    return out


def gate_b(sub_recs, kept, node_subset, cl):
    """Subset identity vs every classical record. Returns (ok, detail list)."""
    bad = []
    pools = [(f"{m} h{h}", cl[m][h], h) for m in ("arima", "sarima") for h in H]
    pools += [(f"gbm h{h} seed{s}", cl["gbm"][(h, s)], h) for h in H for s in SEEDS]
    for tag, recs, h in pools:
        for m in METRICS:
            c, e = _cm(recs, recs[0]["model"], m, h), _cm(sub_recs, "encoder", m, h)
            for k in ("node_subset", "n_nodes", "n_countries"):
                if c[k] != e[k]:
                    bad.append(f"{tag} {m} {k}: classical {c[k]} vs encoder-subset {e[k]}")
    if node_subset != cl["arima"][3][0]["node_subset"]:
        bad.append(f"node_subset string {node_subset} != classical {cl['arima'][3][0]['node_subset']}")
    ids_note = "export meta absent, id check skipped"
    if EXPORT_META.exists():
        b = bundles.load("dengue")
        want = json.loads(EXPORT_META.read_text())["node_ids"]
        got = [b.meta["node_ids"][i] for i in kept]
        ids_note = f"kept ids == GNN export meta ({len(want)} ids)"
        if got != want:
            bad.append(f"kept node ids differ from {EXPORT_META.relative_to(BASE)}")
            ids_note = "kept ids DIFFER from GNN export meta"
    return not bad, bad, ids_note


def _pernode_macro(path, kept, h, metric):
    """Independent route: archived per-node float32 scores, restricted to `kept`, country-macro."""
    z = np.load(path, allow_pickle=True)
    idx, ctry, v = z[f"h{h}__node_idx"], z[f"h{h}__country"], z[f"h{h}__{metric}"].astype(np.float64)
    sel = np.isin(idx, kept) & ~np.isnan(v)
    per = {c: v[sel & (ctry == c)].mean() for c in np.unique(ctry[sel])}
    return float(np.mean(list(per.values()))), int(sel.sum())


def gate_c(b, te, kept, node_subset, pers):
    """C1 full-panel persistence vs naive__dengue.json; C2 subset persistence vs per-node route."""
    full, _, _ = loop.score_predictions("persistence", "dengue", None, pers, b, te)
    arch = json.loads((NAIVE / "naive__dengue.json").read_text())
    eo = b.origins(w=bundles.W, phase="test")
    sub = run_classical.score_records("persistence", "dengue", None, {h: pers[h][kept] for h in H},
                                      b, kept, eo, "test", node_subset)
    ok, rows = True, {}
    for h in H:
        for m in ("rmse", "mae"):
            r1 = _rel(_cm(full, "persistence", m, h)["country_macro"],
                      _cm(arch, "persistence", m, h)["country_macro"])
            pv, pn = _pernode_macro(NAIVE / "naive__dengue__persistence__pernode.npz", kept, h, m)
            s = _cm(sub, "persistence", m, h)
            r2 = _rel(s["country_macro"], pv)
            ok &= r1 <= REL_TOL and r2 <= F32_TOL and pn == s["n_nodes"]
            rows[f"h{h}_{m}"] = dict(full_rel_vs_naive_json=r1, subset=s["country_macro"],
                                     subset_rel_vs_pernode=r2, n_nodes=s["n_nodes"])
    return ok, rows, sub


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #
def _mean_sd(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(xs.mean()), float(xs.std(ddof=1)) if xs.size > 1 else 0.0


def verdict(a, b, t=T95):
    """a, b: per-seed lists (paired) or a scalar for b. Sign is a - b; 'lower' = a has lower error."""
    d = np.asarray(a, float) - (np.asarray(b, float) if np.ndim(b) else float(b))
    m, sd = _mean_sd(d)
    hw = t * sd / math.sqrt(d.size)
    return ("lower" if m + hw < 0 else "higher" if m - hw > 0 else "none"), m, hw


def _seedvals(path_fmt, model, metric, h):
    return [_cm(json.loads(Path(path_fmt.format(s=s)).read_text()), model, metric, h)["country_macro"]
            for s in SEEDS]


def table2(t=T95):
    """Recompute Milestone 6 Table 2 (4 panels x 4 horizons) from disk under rule `t`."""
    tally = {k: [0, 0, 0] for k in list(TABLE2) + [("arima", "encoder", m) for m in ("rmse", "mae")]}
    means = {(m, met): [0, 0] for m in ("arima", "sarima") for met in ("rmse", "mae")}
    pos = {"lower": 0, "none": 1, "higher": 2}
    for p in PANELS4:
        naive = json.loads((NAIVE / f"naive__{p}.json").read_text())
        for h in H:
            for met in ("rmse", "mae"):
                enc = _seedvals(str(SINGLE / f"encoder__{p}__seed{{s}}.json"), "encoder", met, h)
                gbm = _seedvals(str(BL / f"gbm__{p}__h{h}__seed{{s}}.json"), "gbm", met, h)
                c = {m: _cm(json.loads((BL / f"{m}__{p}__h{h}__seedNA.json").read_text()), m, met, h)
                     ["country_macro"] for m in ("arima", "sarima")}
                per = _cm(naive, "persistence", met, h)["country_macro"]
                cmp = {("gbm", "encoder"): (gbm, enc), ("sarima", "encoder"): (c["sarima"], enc),
                       ("arima", "encoder"): (c["arima"], enc),      # not in Table 2; feeds the 20-cell tally
                       ("gbm", "persistence"): (gbm, per), ("sarima", "persistence"): (c["sarima"], per)}
                for (cm, comp), (x, y) in cmp.items():
                    # deterministic side goes second so the one-sample form applies
                    v = (verdict(x, y, t)[0] if np.ndim(x) else
                         {"lower": "higher", "higher": "lower", "none": "none"}[verdict(y, x, t)[0]])
                    tally[(cm, comp, met)][pos[v]] += 1
                for m in ("arima", "sarima"):
                    means[(m, met)][0 if np.mean(enc) < c[m] else 1] += 1
    return {k: tuple(v) for k, v in tally.items()}, means


def _t2_match(t2):
    return all(t2[k] == v for k, v in TABLE2.items())


def tally20(t2, means16, comp):
    """Encoder-perspective win/noise/loss over 5 panels x 4 horizons: Table 2's 16 cells + dengue subset."""
    out = {}
    for m in ("arima", "sarima", "gbm"):
        for met in ("rmse", "mae"):
            lo, no, hi = t2[(m, "encoder", met)]                 # classical lower = encoder loss
            dv = [comp[f"h{h}_{met}"]["encoder_vs"][m] for h in H]
            t = dict(win=hi + sum(d["verdict"] == "win" for d in dv), noise=no + sum(d["verdict"] == "noise" for d in dv),
                     loss=lo + sum(d["verdict"] == "loss" for d in dv))
            if m != "gbm":
                w16, l16 = means16[(m, met)]
                t["straight_mean_win"] = w16 + sum(d["straight_mean"] == "win" for d in dv)
                t["straight_mean_loss"] = l16 + sum(d["straight_mean"] == "loss" for d in dv)
            out[f"{m}_{met}"] = t
    return out


# --------------------------------------------------------------------------- #
# main run
# --------------------------------------------------------------------------- #
def run(seeds=SEEDS, write=True):
    t0 = time.time()
    b = bundles.load("dengue")
    te = b.origins(phase="test")
    eo = b.origins(w=bundles.W, phase="test")
    assert te == eo, "encoder test origins != classical eval origins"
    kept, node_subset = run_classical._kept(b, "dengue")
    cl = classical_records()
    gates = {"A": {}, "B": None, "C": None}
    enc_full, enc_sub, xcheck = {}, {}, {}
    for s in seeds:
        pred = median_preds(s, te, b.raw.shape)
        ok, worst, rows = gate_a(b, te, s, pred)
        gates["A"][s] = dict(ok=ok, worst_rel=worst, rows=rows)
        print(f"gate A seed{s}: {'PASS' if ok else 'FAIL'}  worst rel {worst:.2e}")
        if not ok:
            raise SystemExit(f"GATE A FAILED seed{s}; stopping, nothing written")
        enc_full[s] = {k: v["rescored"] for k, v in rows.items()}
        sub = run_classical.score_records("encoder", "dengue", s, {h: pred[h][kept] for h in H},
                                          b, kept, eo, "test", node_subset)
        if s == seeds[0]:
            okb, bad, ids_note = gate_b(sub, kept, node_subset, cl)
            gates["B"] = dict(ok=okb, problems=bad, ids=ids_note)
            print(f"gate B: {'PASS' if okb else 'FAIL'}  ({ids_note})")
            if not okb:
                raise SystemExit("GATE B FAILED:\n  " + "\n  ".join(bad[:10]))
        enc_sub[s] = {f"h{h}_{m}": _cm(sub, "encoder", m, h)["country_macro"] for h in H for m in METRICS}
        for h in H:
            for m in ("rmse", "mae"):
                pv, _ = _pernode_macro(SINGLE / f"encoder__dengue__seed{s}__pernode.npz", kept, h, m)
                xcheck[f"seed{s}_h{h}_{m}"] = _rel(enc_sub[s][f"h{h}_{m}"], pv)
        del pred
    okc, crows, psub = gate_c(b, te, kept, node_subset, persistence(b, te))
    okx = max(xcheck.values()) <= F32_TOL
    gates["C"] = dict(ok=okc and okx, persistence=crows, encoder_subset_vs_pernode_worst=max(xcheck.values()))
    print(f"gate C: {'PASS' if okc and okx else 'FAIL'}  encoder subset vs per-node route worst rel "
          f"{max(xcheck.values()):.2e}")
    if not (okc and okx):
        raise SystemExit("GATE C FAILED: " + json.dumps(crows, indent=1))

    # comparison
    comp = {}
    for h in H:
        for m in METRICS:
            k = f"h{h}_{m}"
            e = [enc_sub[s][k] for s in seeds]
            ef = [enc_full[s][k] for s in seeds]
            ar = _cm(cl["arima"][h], "arima", m, h)["country_macro"]
            sa = _cm(cl["sarima"][h], "sarima", m, h)["country_macro"]
            gb = [_cm(cl["gbm"][(h, s)], "gbm", m, h)["country_macro"] for s in seeds]
            row = dict(encoder_subset=dict(zip(map(str, seeds), e), mean=_mean_sd(e)[0], sd=_mean_sd(e)[1]),
                       encoder_full=dict(zip(map(str, seeds), ef), mean=_mean_sd(ef)[0], sd=_mean_sd(ef)[1]),
                       arima=ar, sarima=sa,
                       gbm=dict(zip(map(str, seeds), gb), mean=_mean_sd(gb)[0], sd=_mean_sd(gb)[1]))
            if m != "nrmse":
                pv = _cm(psub, "persistence", m, h)["country_macro"]
                row["persistence_subset"] = pv
            vs = {}
            for name, other in (("arima", ar), ("sarima", sa), ("gbm", gb)) + (
                    (("persistence", row["persistence_subset"]),) if m != "nrmse" else ()):
                v, dm, hw = verdict(e, other)
                vs[name] = dict(verdict={"lower": "win", "higher": "loss", "none": "noise"}[v],
                                mean_diff=dm, half_width=hw,
                                straight_mean="win" if np.mean(e) < np.mean(other) else "loss")
            row["encoder_vs"] = vs
            comp[k] = row

    t2, means16 = table2()
    facts = dict(
        sarima_equals_arima=all(_cm(cl["sarima"][h], "sarima", m, h)["country_macro"] ==
                                _cm(cl["arima"][h], "arima", m, h)["country_macro"] for h in H for m in METRICS),
        fallback_nodes=cl["sarima"][3][0]["fallback_nodes"], kept_nodes=int(kept.size),
        scored_nodes=_cm(cl["sarima"][3], "sarima", "rmse", 3)["n_nodes"],
        fallback_denominator_note="sarima_run counts fell_back over range(Nk), Nk = kept.size, so the "
                                  "denominator is the 2392 kept nodes, not the 2051 scored ones",
        under8_train_cells=_under8(b, kept),
        space="count-space country_macro, node-averaged, constant-truth nodes excluded")
    res = dict(script="diagnostics/encoder_dengue_subset.py", seeds=list(seeds), node_subset=node_subset,
               kept_sha256=hashlib.sha256(kept.astype(np.int64).tobytes()).hexdigest(),
               rule=f"two-sided 95% t, n=5, t={T95}; GBM paired by seed, ARIMA/SARIMA/persistence one-sample",
               gates=gates, comparison=comp, facts=facts,
               table2_recomputed={f"{a}_vs_{b_}_{c}": v for (a, b_, c), v in t2.items()},
               table2_matches=_t2_match(t2), tally_20_cells=tally20(t2, means16, comp),
               straight_means_16={f"{m}_{met}": dict(encoder_wins=v[0], encoder_losses=v[1])
                                  for (m, met), v in means16.items()},
               encoder_mc="not archived as quantiles; not rescored", runtime_s=round(time.time() - t0, 1))
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=2))
        print(f"wrote {OUT.relative_to(BASE)}")
    _print(res)
    return res


def _under8(b, kept):
    """Deterministic slice of the fallback: kept nodes with < 8 train cells (sarima_run line 1)."""
    tm = b.masks()["train"][kept].astype(bool).sum(1) < 8
    test = b.masks()["test"][kept].astype(bool).any(1)
    return dict(kept_with_lt8_train=int(tm.sum()), of_which_have_test_cells=int((tm & test).sum()))


def _print(res):
    c = res["comparison"]
    print("\ndengue, country-macro, count space. encoder = 5-seed mean (sd). verdict vs encoder.")
    for m in ("rmse", "mae"):
        print(f"\n{m.upper():5s} enc_full     enc_subset        arima/sarima  gbm mean(sd)     persist"
              "   v_arima v_gbm v_pers")
        for h in H:
            r = c[f"h{h}_{m}"]
            ef, es, g = r["encoder_full"], r["encoder_subset"], r["gbm"]
            v = r["encoder_vs"]
            print(f"h{h:<4d} {ef['mean']:8.3f}  {es['mean']:8.3f} ({es['sd']:.3f})  {r['arima']:10.3f}  "
                  f"{g['mean']:8.3f} ({g['sd']:.3f})  {r['persistence_subset']:8.3f}   "
                  f"{v['arima']['verdict']:6s} {v['gbm']['verdict']:6s} {v['persistence']['verdict']}")
    print(f"\nTable 2 recomputed matches published: {res['table2_matches']}")
    print(f"straight means, 16 non-dengue cells: {res['straight_means_16']}")
    print(f"20 cells, encoder win/noise/loss: {json.dumps(res['tally_20_cells'])}")
    print(f"facts: {json.dumps(res['facts'])}")
    print(f"runtime {res['runtime_s']} s")


# --------------------------------------------------------------------------- #
# selfcheck: every plant must FAIL its gate
# --------------------------------------------------------------------------- #
def selfcheck():
    b = bundles.load("dengue")
    te = b.origins(phase="test")
    kept, node_subset = run_classical._kept(b, "dengue")
    cl = classical_records()
    fails = []

    def expect(name, ok_should_be, ok):
        tag = "ok " if ok == ok_should_be else "BAD"
        print(f"{tag} {name}: gate {'passed' if ok else 'failed'} (expected {'pass' if ok_should_be else 'fail'})")
        if ok != ok_should_be:
            fails.append(name)

    pred = median_preds(42, te, b.raw.shape)
    expect("A clean seed42", True, gate_a(b, te, 42, pred)[0])
    expect("A origins shifted -1 week", False, gate_a(b, te, 42, median_preds(42, te, b.raw.shape, shift=-1))[0])
    expect("A forecasts x1.001", False, gate_a(b, te, 42, {h: p * 1.001 for h, p in pred.items()})[0])
    expect("A wrong seed archive (52 vs 42 json)", False,
           gate_a(b, te, 42, median_preds(52, te, b.raw.shape))[0])

    eo = b.origins(w=bundles.W, phase="test")
    sub = lambda k: run_classical.score_records("encoder", "dengue", 42, {h: pred[h][k] for h in H},
                                                b, k, eo, "test", f"{k.size}/{len(b.meta['node_ids'])}")
    expect("B clean subset", True, gate_b(sub(kept), kept, node_subset, cl)[0])
    wrong = export_baseline._stratified_subsample(b, export_baseline.SUBSAMPLE_FRAC,
                                                  export_baseline.SUBSAMPLE_SEED + 1)
    okw, badw, idsw = gate_b(sub(wrong), wrong, f"{wrong.size}/{len(b.meta['node_ids'])}", cl)
    expect("B wrong subset (subsample seed+1)", False, okw)
    print(f"    caught by: {sorted({p.split(':')[0].split()[-1] for p in badw})[:6]} ; {idsw}")
    full = np.arange(len(b.meta["node_ids"]))
    expect("B full panel posing as subset", False,
           gate_b(sub(full), full, node_subset, cl)[0])

    expect("C clean persistence", True, gate_c(b, te, kept, node_subset, persistence(b, te))[0])
    expect("C persistence lagged one week", False,
           gate_c(b, te, kept, node_subset, persistence(b, te, shift=1))[0])

    t2, _ = table2()
    expect("rule reproduces Table 2", True, _t2_match(t2))
    expect("straight-means rule (t=0) reproduces Table 2", False, _t2_match(table2(t=0.0)[0]))
    if fails:
        raise SystemExit(f"SELFCHECK FAILED: {fails}")
    print("selfcheck: every plant caught, every clean input passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    selfcheck() if a.selfcheck else run()
    print(f"total {time.time() - t0:.1f} s")
    sys.exit(0)

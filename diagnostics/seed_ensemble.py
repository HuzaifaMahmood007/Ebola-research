"""Seed-ensemble the single-disease ceiling, off the archived quantiles. No GPU, no retraining.

    conda run -n ebola-train python -m diagnostics.seed_ensemble --selfcheck
    conda run -n ebola-train python -m diagnostics.seed_ensemble
    conda run -n ebola-train python -m diagnostics.seed_ensemble --datasets dengue

WHY THIS AND NOT SOMETHING CLEVERER. The ceiling has two embarrassing cells -- dengue h3 loses to
persistence and influenza_us-regions h10/h15 lose to seasonal-naive -- and the ceiling is the
denominator of every transfer number, so a weak ceiling makes the transfer gap unreadable. It does
NOT need to be single-disease SOTA; that is not the contribution. Seed ensembling is the one
accuracy lever the client's work order explicitly allows ("take it, it's free"), and it is free here
in the strict sense: `write_quantiles` already archived the count-space forecast for all 25 runs, so
this is a read.

WHY IT MUST HELP, AND BY HOW MUCH IS THE ONLY QUESTION. For squared error, MSE(mean of K forecasts)
= mean(MSE) - (the across-seed variance), so RMSE of the ensemble is <= the mean of the per-seed
RMSEs, strictly whenever seeds disagree. dengue h3 has a per-seed sd of 6.04 on a mean of 42.06, so
there is real variance to collect. This is arithmetic, not a hypothesis -- which is why the useful
output is the SIZE of the gain, and whether it clears the floor.

WHAT IT IS NOT. The ensemble is over the count-space MEDIAN, so it is comparable to `encoder` and
NOT to `encoder_mc`. The mean-correction is added in MODEL space before the scaler is inverted
(`train/loop.py:222`), and the archive holds only post-inversion counts, so `encoder_mc` cannot be
rebuilt from these files. Reported as the median arm, against the median arm.

THE CONTROL THAT MAKES IT TRUSTWORTHY. --selfcheck rebuilds ONE seed's predictions from its own
archive and rescores them, and requires the result to reproduce that seed's released JSON. If the
reader mis-indexes the origin axis, picks the wrong quantile, or scores different cells, that check
fails -- and every number below would otherwise be quietly wrong in the same direction.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

import bundles
import score
from models import MEDIAN_IDX
from results_paths import rpath
from train.loop import score_predictions

SEEDS = (42, 52, 62, 72, 82)
PANELS = ("dengue", "influenza_japan", "influenza_us-regions", "influenza_us-states",
          "covid_us-states")
POINT = ("rmse", "mae", "pcc")
LOWER_BETTER = ("rmse", "mae")
HEADLINE = {"dengue": "country_macro"}


def field(ds):
    return HEADLINE.get(ds, "node_mean")


def _archive(ds, seed, prefix="encoder"):
    return rpath(f"{prefix}__{ds}__seed{seed}__quantiles.npz")


def ensemble_preds(ds, seeds=SEEDS, prefix="encoder"):
    """{h: [N,T] count-space mean-over-seeds median forecast}, plus the shared origin vector.

    One horizon at a time and one seed at a time: dengue is [7165, 630, 5] float32 per horizon, so
    holding all five seeds x four horizons at once is ~1.8 GB for a column we then throw away.
    """
    paths = [(s, _archive(ds, s, prefix)) for s in seeds]
    have = [(s, p) for s, p in paths if Path(p).exists()]
    if len(have) < 2:
        return None, None, [s for s, _ in have]

    origins = None
    for s, p in have:
        o = np.load(p)["origins"]
        if origins is None:
            origins = o
        elif not np.array_equal(origins, o):
            # A seed scored on different origins cannot be averaged into the others: the columns
            # would not line up in time and the ensemble would silently mix horizons.
            raise SystemExit(f"{ds} seed{s}: origin vector differs from seed{have[0][0]}")

    b = bundles.load(ds)
    N, T = b.X.shape[0], b.X.shape[1]
    pred_by_h = {}
    for h in bundles.HORIZONS:
        acc = np.zeros((N, len(origins)), dtype=np.float64)
        for _, p in have:
            z = np.load(p)
            acc += z[f"h{h}__quantiles"][:, :, MEDIAN_IDX].astype(np.float64)
            z.close()
        acc /= len(have)
        full = np.zeros((N, T), dtype=np.float64)
        for k, t in enumerate(origins):
            full[:, t + h] = acc[:, k]
        pred_by_h[h] = full
    return pred_by_h, (b, origins), [s for s, _ in have]


def per_seed(ds, metric, h, model="encoder"):
    """The released per-seed values for one cell, from the JSON records."""
    out = []
    for f in sorted(glob.glob(str(rpath(f"encoder__{ds}__seed*.json")))):
        if "smoke" in Path(f).name:
            continue
        for r in json.load(open(f)):
            if r["model"] == model and r["horizon"] == h and r["metric"] == metric:
                out.append(r[field(ds)])
    return out


def floors(ds, metric, h):
    """{naive: value} for one cell."""
    p = rpath(f"naive__{ds}.json")
    if not Path(p).exists():
        return {}
    return {r["model"]: r[field(ds)] for r in json.load(open(p))
            if r["horizon"] == h and r["metric"] == metric}


def report(datasets, seeds=SEEDS):
    print(f"\n{'=' * 104}\nSEED-ENSEMBLE CEILING :: mean of the per-seed count-space MEDIAN "
          f"forecast (comparable to `encoder`, not `encoder_mc`)\n{'=' * 104}")
    fixed, still = [], []
    for ds in datasets:
        pred, ctx, got = ensemble_preds(ds, seeds)
        if pred is None:
            print(f"\n  {ds}: fewer than 2 archives on disk ({got}) -- nothing to ensemble")
            continue
        b, origins = ctx
        recs, _, _ = score_predictions("encoder_ens", ds, None, pred, b, origins)
        ens = {(r["horizon"], r["metric"]): r[field(ds)] for r in recs}

        print(f"\n  {ds}   field={field(ds)}   seeds {got}")
        for m in POINT:
            better = "higher" if m == "pcc" else "lower"
            print(f"    {m.upper()} ({better}=better)")
            print(f"      {'h':>3} | {'per-seed mean':>18} | {'ensemble':>10} | {'gain':>8} | "
                  f"{'best floor':>18} | verdict")
            for h in bundles.HORIZONS:
                vals = per_seed(ds, m, h)
                if not vals or (h, m) not in ens:
                    continue
                sm = st.mean(vals)
                ssd = st.stdev(vals) if len(vals) > 1 else 0.0
                e = ens[(h, m)]
                gain = (sm - e) / sm * 100 if m in LOWER_BETTER else (e - sm) / abs(sm) * 100
                fl = floors(ds, m, h)
                if fl:
                    bn = (min(fl, key=fl.get) if m in LOWER_BETTER else max(fl, key=fl.get))
                    bv = fl[bn]
                    beat = (e < bv) if m in LOWER_BETTER else (e > bv)
                    marg = ((bv - e) / bv * 100) if m in LOWER_BETTER else (e - bv)
                    verdict = f"{'BEATS' if beat else 'loses to'} {bn} ({marg:+.1f}%)"
                    if m in LOWER_BETTER:
                        (fixed if beat else still).append(
                            f"{ds} h{h} {m}: ens {e:.2f} vs {bn} {bv:.2f} ({marg:+.1f}%)")
                else:
                    bv, verdict = float("nan"), "no floor on disk"
                print(f"      {h:>3} | {sm:10.3f} +-{ssd:6.3f} | {e:10.3f} | {gain:+7.2f}% | "
                      f"{bv:10.3f} {'':6s} | {verdict}")

    print(f"\n  Ensemble = mean over seeds of the count-space median, scored through score.py on "
          f"the SAME cells as every per-seed run (score_predictions, same origins, same masks).")
    print(f"  'gain' is the ensemble against the MEAN of the per-seed values. For squared error a "
          f"gain is guaranteed by Jensen whenever seeds disagree; the size is the finding.")
    if fixed:
        print(f"\n  CLEARS ITS FLOOR ({len(fixed)}):")
        for s in fixed:
            print(f"    + {s}")
    if still:
        print(f"\n  STILL BELOW ITS FLOOR ({len(still)}):")
        for s in still:
            print(f"    - {s}")
    print()


def compare_transfer(datasets, seeds=SEEDS, transfer="encoder_ldo3", B=10000, rng_seed=0):
    """Ensembled ceiling vs ensembled transfer arm -- the like-for-like fight.

    WHY SYMMETRY IS THE POINT. Ensembling only the ceiling would compare our best single-disease
    system against a single transfer run, which is the same reference mismatch that produced the
    retracted +24% Week-3 headline, just pointing the other way. Both sides get the same treatment
    or neither does.

    TWO ESTIMANDS, NEVER MIXED (audit M1). The delta is reported on the node-averaged country-macro,
    which is the headline everywhere else in this project. The INTERVAL is a paired bootstrap over
    origins on the CELL-POOLED country-macro, because that is the statistic the sufficient stats
    support -- and it is the same instrument LDO3_Results.md's 25-of-36 tally already uses. They are
    printed as separate columns with their own labels. Printing the cell-pooled interval beside the
    node-averaged point is exactly the defect M1 was raised about.

    Both arms are resampled with the SAME origin draw, which is what makes it paired: the shared
    origin-to-origin difficulty cancels instead of being counted twice.
    """
    from analysis import _cnt_matrix, _macro_dist, _point_macro

    print(f"\n{'=' * 112}\nSYMMETRIC ENSEMBLE :: ceiling(encoder) vs transfer({transfer}), both "
          f"5-seed ensembles of the count-space median\n{'=' * 112}")
    tally = {"transfer better": 0, "within noise": 0, "transfer worse": 0}
    rng = np.random.default_rng(rng_seed)
    for ds in datasets:
        c_pred, c_ctx, c_got = ensemble_preds(ds, seeds, "encoder")
        t_pred, t_ctx, t_got = ensemble_preds(ds, seeds, transfer)
        if c_pred is None or t_pred is None:
            print(f"\n  {ds}: archives missing (ceiling {c_got}, transfer {t_got}) -- skipped")
            continue
        if not np.array_equal(c_ctx[1], t_ctx[1]):
            raise SystemExit(f"{ds}: ceiling and transfer scored different origins; not comparable")
        if c_got != t_got:
            print(f"\n  {ds}: WARNING ceiling seeds {c_got} != transfer seeds {t_got}")

        b, origins = c_ctx
        c_rec, _, c_po = score_predictions("ens", ds, None, c_pred, b, origins)
        t_rec, _, t_po = score_predictions("ens", ds, None, t_pred, b, origins)
        c_pt = {(r["horizon"], r["metric"]): r[field(ds)] for r in c_rec}
        t_pt = {(r["horizon"], r["metric"]): r[field(ds)] for r in t_rec}

        print(f"\n  {ds}   seeds {c_got}   node-averaged field={field(ds)}")
        print(f"    {'h':>3} {'metric':>5} | {'ceiling':>10} {'transfer':>10} {'delta%':>8} "
              f"(node-avg) | {'delta% (cell-pooled)':>21} {'95% CI':>22} | verdict")
        for h in bundles.HORIZONS:
            K = c_po[f"h{h}__sae"].shape[0]
            CNT = _cnt_matrix(K, B, rng)                       # ONE draw, applied to both arms
            for m in ("rmse", "mae"):
                cp, tp = c_pt.get((h, m)), t_pt.get((h, m))
                if cp is None or tp is None:
                    continue
                d_node = (cp - tp) / cp * 100                  # + means transfer BETTER
                args = (lambda po: (po[f"h{h}__sae"], po[f"h{h}__sse"], po[f"h{h}__n"]))
                c_cell, t_cell = _point_macro(*args(c_po), m), _point_macro(*args(t_po), m)
                d_cell = (c_cell - t_cell) / c_cell * 100
                cd = _macro_dist(*args(c_po), CNT, m)
                td = _macro_dist(*args(t_po), CNT, m)
                rel = (cd - td) / cd * 100                     # paired: same CNT on both arms
                lo, hi = np.nanpercentile(rel, [2.5, 97.5])
                if lo <= 0 <= hi:
                    v = "within noise"
                else:
                    v = "transfer better" if lo > 0 else "transfer worse"
                tally[v] += 1
                print(f"    {h:>3} {m:>5} | {cp:10.3f} {tp:10.3f} {d_node:+8.1f} "
                      f"{'':11s} | {d_cell:+21.1f} {f'[{lo:+.1f}, {hi:+.1f}]':>22} | {v}")

    tot = sum(tally.values())
    print(f"\n  TALLY over {tot} cells: {tally['transfer better']} transfer better, "
          f"{tally['within noise']} within noise, {tally['transfer worse']} transfer worse.")
    print(f"  Reference: the 5-seed ENSEMBLED single-disease ceiling on the same dataset and the "
          f"same origins. Positive delta = transfer better.")
    print(f"  The delta% column is NODE-AVERAGED (the project headline). The CI is a paired "
          f"bootstrap over origins (B={B:,}) on the CELL-POOLED macro, printed beside its OWN "
          f"point so the two estimands are never mixed (audit M1).")
    print(f"  Ensembling consumes the seed axis, so the seed-paired t-interval used in "
          f"LDO3_Results.md cannot be computed here; that table stays as the per-seed record.\n")


def _selfcheck():
    """One seed, rebuilt from its own archive and rescored, must reproduce its released record.

    This is the whole trust argument. A wrong quantile index, a shifted origin axis or a different
    eval mask would all still produce plausible numbers -- and would move every ensemble cell in the
    same direction, so nothing downstream would look odd."""
    ds, seed = "influenza_us-regions", 42
    p = _archive(ds, seed)
    if not Path(p).exists():
        sys.exit(f"selfcheck needs {p}")

    pred, ctx, got = ensemble_preds(ds, seeds=(seed,) * 2)   # "ensemble" of one seed == that seed
    b, origins = ctx
    recs, _, _ = score_predictions("encoder", ds, seed, pred, b, origins)
    rebuilt = {(r["horizon"], r["metric"]): r[field(ds)] for r in recs}

    released = {}
    for r in json.load(open(rpath(f"encoder__{ds}__seed{seed}.json"))):
        if r["model"] == "encoder":
            released[(r["horizon"], r["metric"])] = r[field(ds)]

    checked = 0
    for k, want in released.items():
        if k not in rebuilt or want is None or (isinstance(want, float) and np.isnan(want)):
            continue
        got_v = rebuilt[k]
        assert abs(got_v - want) <= 1e-6 * max(1.0, abs(want)), \
            f"{ds} seed{seed} {k}: rebuilt {got_v} != released {want}"
        checked += 1
    assert checked >= 12, f"only {checked} cells compared -- the control is too weak to trust"

    # MUTATION: read the wrong quantile and the check must FAIL. Without this, a reader that
    # happened to reproduce for an unrelated reason would pass silently.
    z = np.load(p)
    q = z[f"h3__quantiles"]
    assert not np.allclose(q[:, :, MEDIAN_IDX], q[:, :, 0]), \
        "control void: the 5% and 50% quantiles are identical on this panel, so a wrong index " \
        "would reproduce anyway -- pick a different panel for the selfcheck"
    z.close()

    # the ensemble must actually average: two distinct seeds must not equal either one alone
    pr2, ctx2, _ = ensemble_preds(ds, seeds=(42, 52))
    r2, _, _ = score_predictions("encoder", ds, None, pr2, ctx2[0], ctx2[1])
    e2 = {(r["horizon"], r["metric"]): r[field(ds)] for r in r2 if r["model"] == "encoder"}
    diffs = sum(1 for k in released if k in e2 and abs(e2[k] - released[k]) > 1e-9)
    assert diffs > 0, "a 2-seed ensemble reproduced seed 42 exactly -- it is not averaging"

    print(f"selfcheck ok: {checked} cells rebuilt from the archive reproduce the released "
          f"{ds} seed{seed} record to 1e-6; the median index is distinguishable on this panel; "
          f"a 2-seed ensemble differs from either seed alone in {diffs} cells")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(PANELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--vs-transfer", nargs="?", const="encoder_ldo3", default=None,
                    help="symmetric comparison: ensembled ceiling vs an ensembled transfer family")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()
    if a.vs_transfer:
        compare_transfer(a.datasets, tuple(a.seeds), a.vs_transfer, B=a.boot)
        return 0
    report(a.datasets, tuple(a.seeds))
    return 0


if __name__ == "__main__":
    sys.exit(main())

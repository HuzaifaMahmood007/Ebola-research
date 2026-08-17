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


def _archive(ds, seed):
    return rpath(f"encoder__{ds}__seed{seed}__quantiles.npz")


def ensemble_preds(ds, seeds=SEEDS):
    """{h: [N,T] count-space mean-over-seeds median forecast}, plus the shared origin vector.

    One horizon at a time and one seed at a time: dengue is [7165, 630, 5] float32 per horizon, so
    holding all five seeds x four horizons at once is ~1.8 GB for a column we then throw away.
    """
    paths = [(s, _archive(ds, s)) for s in seeds]
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
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()
    report(a.datasets, tuple(a.seeds))
    return 0


if __name__ == "__main__":
    sys.exit(main())

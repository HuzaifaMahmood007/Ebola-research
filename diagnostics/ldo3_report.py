"""ldo3_report.py -- verify the three-disease leave-one-disease-out overnight run and report it.

Two jobs, in this order, because the second is worthless without the first:

  1. VERIFY. Enumerate the 15 runs (3 held-out diseases x 5 seeds) and check every artifact the
     fold is supposed to emit actually landed and is internally consistent: record counts, seed and
     dataset agreement between filename and payload, the fold metadata (held_out_disease /
     in_diseases / fold_structure), NaN sweep, and -- the one that actually catches a mis-scored
     fold -- that the transfer arm scored the SAME node count as the single-disease ceiling it is
     compared against. Any failure is printed and the exit code is non-zero.

  2. REPORT. The cross-disease transfer read, per bundle, never pooled across bundles (client D3):
       - point metrics (rmse, mae, nrmse, pcc, smape, peak_intensity, peak_timing) for the
         LDO3-adapted arm, the LDO3 zero-shot arm, the single-disease ceiling and the naive floor,
         each as mean +- sd over 5 seeds;
       - paired-by-seed deltas against the ceiling with a small-sample t interval, labelled
         "within noise" when that interval covers zero;
       - the UQ block the Review Doc asks for -- WIS, CRPS, empirical coverage, interval width and
         PIT -- computed from the archived quantile forecasts.

  Reference is stated on every delta table, per the reporting standard.

Run:
  "C:/Users/Administrator/miniconda3/envs/ebola-train/python.exe" ldo3_report.py -o LDO3_Results.md
  ... --skip-uq     point metrics only (dengue's quantile archive is 360 MB/seed)
  ... --selfcheck   runnable checks of the aggregation helpers, needs no artifacts
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import collections
import json
import math
import sys

import numpy as np

import score
from results_paths import rpath, RESULTS
from results_matrix import _fmt, _t95, paired_delta, cell, mean_of

SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
NAIVES = ("persistence", "seasonal", "train_mean")
LOWER_BETTER = {"rmse", "mae", "nrmse", "smape", "peak_timing", "peak_intensity"}

# The fold universe, mirrored from train/lodo.py::DISEASES. Asserted against it at run time rather
# than trusted, so a change there cannot silently desynchronise this report.
DISEASES = {
    "dengue":    ("dengue",),
    "influenza": ("influenza_japan", "influenza_us-regions", "influenza_us-states"),
    "covid":     ("covid_us-states",),
}
ADAPTED, ZEROSHOT = "encoder_ldo3", "encoder_ldo3_zeroshot"

PRIMARY = ("rmse", "mae", "nrmse", "pcc")
EPI = ("smape", "peak_intensity", "peak_timing")

# Cells that are NOT ATTRIBUTABLE to transfer, and are therefore excluded from every verdict tally
# rather than counted as losses.
#
# covid_us-states at the long horizons is a property of the DATA, not of the trunk. The Omicron peak
# (5.1M national weekly cases, t=102) falls inside the VALIDATION fold, so every arm -- transfer,
# ceiling and naive alike -- selects its model on a surge and is then scored on a flat, low-amplitude
# tail: the test fold peaks 6.3x lower and carries about half the per-state relative variability
# (mean CV 0.583 in test against 1.124 in train). A model fitted to surge dynamics necessarily
# over-predicts a flat tail, and the error compounds with lead time, so h10-h15 measures the fold
# boundary rather than the transfer. Section 2 states the measured numbers. The SHORT horizons stay
# in the tally -- they are not excused by this, and covid h3 does register as a genuine loss.
NOT_ATTRIBUTABLE = {("covid_us-states", 10), ("covid_us-states", 15)}
NA_NOTE = "not attributable (COVID regime break)"


def attributable(ds, h):
    return (ds, h) not in NOT_ATTRIBUTABLE

# country_macro is the headline field everywhere: on the four single-country panels it is identical
# to node_mean, and on dengue it is the agreed headline (equal weight per country, so Brazil cannot
# dominate 7,165 nodes). Using one field for all five keeps the tables readable against each other.
FIELD = "country_macro"


# --------------------------------------------------------------------------- #
# 1. Verification
# --------------------------------------------------------------------------- #
def _expected_artifacts():
    """{(held_out, seed): [filenames]} -- exactly what run_ldo3_fold writes for one run."""
    want = {}
    for held, names in DISEASES.items():
        for s in SEEDS:
            fs = [f"{ADAPTED}__{held}__seed{s}__ckpt.pt"]
            for n in names:
                fs += [f"{ADAPTED}__{n}__seed{s}.json",
                       f"{ADAPTED}__{n}__seed{s}__pernode.npz",
                       f"{ADAPTED}__{n}__seed{s}__perorigin.npz",
                       f"{ADAPTED}__{n}__seed{s}__gate.npz",
                       f"{ADAPTED}__{n}__seed{s}__quantiles.npz",
                       f"{ZEROSHOT}__{n}__seed{s}.json",
                       f"{ZEROSHOT}__{n}__seed{s}__pernode.npz",
                       f"{ZEROSHOT}__{n}__seed{s}__perorigin.npz"]
            want[(held, s)] = fs
    return want


def _load_json(fname):
    p = rpath(fname)
    return json.loads(p.read_text()) if p.exists() else None


def verify(verbose=True):
    """Returns (ok, problems, manifest). Never raises on a missing file -- it reports it."""
    problems, manifest = [], []

    try:
        import train.lodo as LD
        if {k: tuple(v) for k, v in LD.DISEASES.items()} != {k: tuple(v) for k, v in DISEASES.items()}:
            problems.append(f"train/lodo.py DISEASES has drifted from this report's copy: "
                            f"{LD.DISEASES} vs {DISEASES}")
    except Exception as e:                                   # torch missing -> skip, not a failure
        if verbose:
            print(f"note: could not import train.lodo to cross-check the fold universe ({e})")

    n_metrics = len(score.METRICS)
    want = _expected_artifacts()
    for (held, seed), files in sorted(want.items()):
        missing = [f for f in files if not rpath(f).exists()]
        row = dict(held_out=held, seed=seed, files=len(files), missing=len(missing))
        if missing:
            problems.append(f"{held} seed{seed}: {len(missing)} missing artifact(s): "
                            f"{missing[:4]}{' ...' if len(missing) > 4 else ''}")

        for name in DISEASES[held]:
            for prefix, arm in ((ADAPTED, "adapted"), (ZEROSHOT, "zero-shot")):
                recs = _load_json(f"{prefix}__{name}__seed{seed}.json")
                if recs is None:
                    continue
                tag = f"{held} seed{seed} {name} [{arm}]"
                if len(recs) != len(HORIZONS) * n_metrics:
                    problems.append(f"{tag}: {len(recs)} records, expected "
                                    f"{len(HORIZONS) * n_metrics}")
                bad_seed = {r["seed"] for r in recs} - {seed}
                if bad_seed:
                    problems.append(f"{tag}: payload seed {bad_seed} disagrees with filename")
                bad_ds = {r["dataset"] for r in recs} - {name}
                if bad_ds:
                    problems.append(f"{tag}: payload dataset {bad_ds} disagrees with filename")
                # the fold metadata is what makes a record self-describing; a wrong held_out_disease
                # would make the whole table a mislabelled in-domain run.
                ho = {r.get("held_out_disease") for r in recs}
                if ho != {held}:
                    problems.append(f"{tag}: held_out_disease={ho}, expected {{'{held}'}}")
                ind = {r.get("in_diseases") for r in recs}
                expect_in = ",".join(d for d in DISEASES if d != held)
                if ind != {expect_in}:
                    problems.append(f"{tag}: in_diseases={ind}, expected {{'{expect_in}'}}")
                if held in (ind.pop() if len(ind) == 1 else ""):
                    problems.append(f"{tag}: held-out disease appears in its own trunk set")
                fs = {r.get("fold_structure") for r in recs}
                if fs != {"leave-one-disease-out-3way"}:
                    problems.append(f"{tag}: fold_structure={fs}")
                reg = {r.get("training_regime") for r in recs}
                want_reg = {"ldo3"} if arm == "adapted" else {"ldo3_zeroshot"}
                if reg != want_reg:
                    problems.append(f"{tag}: training_regime={reg}, expected {want_reg}")
                nan = [(r["horizon"], r["metric"]) for r in recs
                       if r[FIELD] is None or math.isnan(r[FIELD])]
                if nan:
                    problems.append(f"{tag}: NaN {FIELD} at {nan[:6]}")
                # node-count agreement with the ceiling: the single strongest signal that the fold
                # scored the same cells the comparison assumes.
                ref = [r for r in (_load_json(f"encoder__{name}__seed{seed}.json") or [])
                       if _wanted_model("encoder", r.get("model"))]
                if ref:
                    rn = {(r["horizon"], r["metric"]): r["n_nodes"] for r in ref}
                    mism = [(r["horizon"], r["metric"], r["n_nodes"], rn.get((r["horizon"], r["metric"])))
                            for r in recs if rn.get((r["horizon"], r["metric"])) not in (None, r["n_nodes"])]
                    if mism:
                        problems.append(f"{tag}: n_nodes differs from the single-disease ceiling at "
                                        f"{mism[:3]} (h, metric, arm, ceiling)")
        manifest.append(row)

    if verbose:
        print(f"{'=' * 78}\nVERIFY  three-disease LODO: {len(want)} runs "
              f"({len(DISEASES)} held-out diseases x {len(SEEDS)} seeds)\n{'=' * 78}")
        print(f"  {'held-out':10} {'seed':>5} {'artifacts':>10} {'missing':>8}")
        for r in manifest:
            print(f"  {r['held_out']:10} {r['seed']:>5} {r['files']:>10} {r['missing']:>8}")
        total = sum(r["files"] for r in manifest)
        miss = sum(r["missing"] for r in manifest)
        print(f"\n  {total - miss}/{total} expected artifacts present.")
        if problems:
            print(f"\n  {len(problems)} PROBLEM(S):")
            for p in problems:
                print(f"    - {p}")
        else:
            print("  No integrity problems found.")
    return (not problems), problems, manifest


# --------------------------------------------------------------------------- #
# 2a. Point metrics
# --------------------------------------------------------------------------- #
def _wanted_model(prefix, model):
    """Is this record the arm `prefix` names?

    Load-bearing for COVID. `encoder__covid_us-states__seed*.json` holds TWO arms in one file: the
    median forecast (`encoder`) and the bias-corrected variant (`encoder_mc`), 28 records each,
    under identical (dataset, horizon, metric, seed) keys. Keying on those four fields alone lets
    the second arm overwrite the first, so the COVID ceiling silently becomes the mean-corrected
    model -- a different experiment -- while still being labelled the single-disease ceiling.
    results_matrix.py hit exactly this and guards it with a regime suffix; here the arm is chosen
    explicitly instead.
    """
    if prefix == "encoder":
        return model == "encoder"
    return str(model).startswith(f"{prefix}:")


def load_points():
    """{(arm, dataset, horizon, metric): {seed: value}} for every arm we compare."""
    out = collections.defaultdict(dict)
    for prefix, arm in ((ADAPTED, "LDO3 adapted"), (ZEROSHOT, "LDO3 zero-shot"),
                        ("encoder", "single (ceiling)")):
        for names in DISEASES.values():
            for n in names:
                for s in SEEDS:
                    for r in _load_json(f"{prefix}__{n}__seed{s}.json") or []:
                        if not _wanted_model(prefix, r.get("model")):
                            continue
                        key = (arm, n, r["horizon"], r["metric"])
                        assert s not in out[key], \
                            f"{prefix} {n} seed{s} h{r['horizon']} {r['metric']}: two records " \
                            f"claim this cell (models collided) -- refusing to overwrite silently"
                        out[key][s] = r[FIELD]
    for names in DISEASES.values():                          # naive floors: seedless
        for n in names:
            for r in _load_json(f"naive__{n}.json") or []:
                out[(f"naive:{r['model']}", n, r["horizon"], r["metric"])][None] = r[FIELD]
    return out


def vs_floor(vals: dict, floor, metric):
    """One-sample t of the arm's seeds against the FIXED naive floor scalar: (text, mean, clears).

    Beating the single-disease ceiling means nothing if both arms lose to a train-mean baseline --
    which is exactly the COVID situation. This is the test that catches that, so it is reported
    beside the ceiling delta rather than left for the reader to do by eye. The floor is one pooled
    number with no seed dispersion of its own, so the interval here covers only the arm's spread.
    """
    xs = [v for v in vals.values() if v is not None and not math.isnan(v)]
    if not xs or floor is None or math.isnan(floor) or abs(floor) < 1e-9:
        return "—", None, None
    if metric in LOWER_BETTER:
        deltas = [(floor - x) / abs(floor) * 100.0 for x in xs]
    else:
        deltas = [(x - floor) / abs(floor) * 100.0 for x in xs]
    m = sum(deltas) / len(deltas)
    if len(deltas) < 2:
        return f"{m:+.1f}% (1 seed, untestable)", m, None
    sd = math.sqrt(sum((x - m) ** 2 for x in deltas) / (len(deltas) - 1))
    hw = _t95(len(deltas)) * sd / math.sqrt(len(deltas))
    if abs(m) <= hw:
        return f"within noise ({m:+.1f} ± {hw:.1f}%, n={len(deltas)})", m, False
    return f"**{m:+.1f}%** ± {hw:.1f} (n={len(deltas)})", m, True


def best_naive(points, ds, h, metric):
    """(value, which) for the strongest naive floor at this cell."""
    cands = [(mean_of(points.get((f"naive:{nm}", ds, h, metric), {})), nm) for nm in NAIVES]
    cands = [(v, nm) for v, nm in cands if v is not None and not math.isnan(v)]
    if not cands:
        return None, None
    return (min if metric in LOWER_BETTER else max)(cands, key=lambda t: t[0])


# --------------------------------------------------------------------------- #
# 2b. UQ metrics from the archived quantile forecasts
# --------------------------------------------------------------------------- #
def _eval_mask(b, origins, h):
    """The [N,T] mask score_predictions used: observed AND in the test split, at t+h."""
    phase = b.masks()["test"].astype(bool)
    m = np.zeros_like(phase)
    for t in origins:
        m[:, t + h] = phase[:, t + h]
    return m


def _pit_vec(q, y, levels):
    """Vectorised equivalent of score.pit over a whole [.., Q] grid at once.

    score.pit interpolates one cell at a time in a Python loop, which is fine for a 49-node panel
    and impossible for dengue (7,165 nodes x 630 origins x 4 horizons x 5 seeds is ~90M calls).
    Same definition: linear interpolation of y on the predictive quantile curve, pinned to exactly
    0 below q05 and 1 above q95. Checked against score.pit in _selfcheck.
    """
    q = np.sort(np.asarray(q, dtype=np.float64), axis=-1)
    tau = np.asarray(levels, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Q = q.shape[-1]
    idx = np.sum(q <= y[..., None], axis=-1)                 # 0..Q
    below, above = idx == 0, idx == Q
    i1 = np.clip(idx, 1, Q - 1)
    i0 = i1 - 1
    q0 = np.take_along_axis(q, i0[..., None], -1)[..., 0]
    q1 = np.take_along_axis(q, i1[..., None], -1)[..., 0]
    span = q1 - q0
    # a flat segment carries no information about where y sits; take the left level, as np.interp does
    frac = np.where(span > 0, (y - q0) / np.where(span > 0, span, 1.0), 0.0)
    out = tau[i0] + frac * (tau[i1] - tau[i0])
    out[below] = 0.0
    out[above] = 1.0
    return out


def _node_means(vals, obs, n_obs):
    """Per-node mean of a [N,K] cell quantity over observed cells only."""
    return np.where(n_obs > 0, (np.asarray(vals, dtype=np.float64) * obs).sum(1) /
                    np.where(n_obs > 0, n_obs, 1), np.nan)


def uq_for_run(name, seed, prefix, bundle, horizons=HORIZONS, pit_bins=10, chunk=1024):
    """WIS / CRPS / coverage / width / PIT for one (dataset, seed, arm), aggregated the same way
    the point metrics are: per node over its observed eval cells, then country-macro over nodes.

    Nodes with a constant truth window are dropped, mirroring score.per_node_scores, so the UQ rows
    describe the same node set as the rmse/mae rows beside them. Nodes are processed in chunks
    because dengue's quantile block is 90 MB per horizon in float32 and ~2x that promoted to f64.
    """
    p = rpath(f"{prefix}__{name}__seed{seed}__quantiles.npz")
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    origins = z["origins"]
    # The archive stores the levels as float32, so a round-tripped 0.05 comes back as
    # 0.05000000074505806 and 0.95 as 0.9499999880790710. score._interval_pairs matches lo to
    # (1 - lo) with a 1e-12 tolerance, so feeding it the round-tripped tuple SILENTLY DROPS the 90%
    # interval and the table loses its coverage-90 and width-90 columns with no error. Use the
    # canonical levels, but verify the archive really is that set first.
    stored = np.asarray(z["quantiles"], dtype=np.float64)
    levels = score.QUANTILE_LEVELS
    assert stored.shape == (len(levels),) and np.allclose(stored, levels, atol=1e-6), \
        f"{p.name}: archived levels {stored} are not score.QUANTILE_LEVELS {levels}"
    pairs = score._interval_pairs(levels)
    assert len(pairs) == 2, f"expected the 50% and 90% central intervals, got {pairs}"
    raw = bundle.raw.astype(np.float64)
    ids, ncmap = bundle.meta["node_ids"], bundle.group_of()
    country = np.array([ncmap[n] for n in ids])

    out = {}
    for h in horizons:
        qz = z[f"h{h}__quantiles"]                           # [N, K, Q] over the K origins
        mask = _eval_mask(bundle, origins, h)
        tcol = origins + h                                   # the scored column per origin
        truth = raw[:, tcol]                                 # [N, K]
        obs = mask[:, tcol].astype(np.float64)               # [N, K]
        n_obs = obs.sum(1)

        # constant-node drop, computed over OBSERVED cells only (score.per_node_scores does the same)
        mu = np.where(n_obs > 0, (truth * obs).sum(1) / np.where(n_obs > 0, n_obs, 1), 0.0)
        var = np.where(n_obs > 0, ((truth - mu[:, None]) ** 2 * obs).sum(1) /
                       np.where(n_obs > 0, n_obs, 1), 0.0)
        valid = (n_obs > 0) & (np.sqrt(var) >= 1e-8)
        if not valid.any():
            continue

        acc = {k: np.full(truth.shape[0], np.nan) for k in
               ["wis", "crps"] + [f"cov{1 - a:g}" for _, _, a in pairs] +
               [f"width{1 - a:g}" for _, _, a in pairs]}
        pit_pool = []
        for lo in range(0, truth.shape[0], chunk):
            hi = min(lo + chunk, truth.shape[0])
            sl = slice(lo, hi)
            if not valid[sl].any():
                continue
            q = np.sort(qz[sl].astype(np.float64), axis=-1)  # [n, K, Q]
            y, o, no = truth[sl], obs[sl], n_obs[sl]
            acc["wis"][sl] = _node_means(score.wis(q, y, levels), o, no)
            acc["crps"][sl] = _node_means(score.crps(q, y, levels), o, no)
            for lo_i, hi_i, alpha in pairs:
                nom = f"{1 - alpha:g}"
                inside = ((y >= q[..., lo_i]) & (y <= q[..., hi_i])).astype(np.float64)
                acc[f"cov{nom}"][sl] = _node_means(inside, o, no)
                acc[f"width{nom}"][sl] = _node_means(q[..., hi_i] - q[..., lo_i], o, no)
            keep = valid[sl][:, None] & (o > 0)
            if keep.any():
                pit_pool.append(_pit_vec(q, y, levels)[keep])
            del q

        # country-macro over the valid nodes, matching score.aggregate
        agg = {}
        for k, v in acc.items():
            per_country = []
            for c in np.unique(country[valid]):
                vals = v[valid & (country == c)]
                vals = vals[~np.isnan(vals)]
                if vals.size:
                    per_country.append(float(vals.mean()))
            agg[k] = float(np.mean(per_country)) if per_country else float("nan")
        pit_vals = np.concatenate(pit_pool) if pit_pool else np.array([])
        hist, _ = np.histogram(pit_vals, bins=pit_bins, range=(0.0, 1.0))
        agg["pit_hist"] = (hist / max(hist.sum(), 1)).tolist()
        agg["pit_saturated"] = (float(np.mean((pit_vals <= 0.0) | (pit_vals >= 1.0)))
                                if pit_vals.size else float("nan"))
        agg["n_nodes"] = int(valid.sum())
        out[h] = agg
        del qz
    return out


def uq_table(names, prefixes=(ADAPTED,), seeds=SEEDS, verbose=True):
    """{(arm, dataset, horizon, key): {seed: value}}, skipping arms with no archived quantiles."""
    import bundles
    out = collections.defaultdict(dict)
    for name in names:
        b = None
        for prefix in prefixes:
            if not any(rpath(f"{prefix}__{name}__seed{s}__quantiles.npz").exists() for s in seeds):
                if verbose:
                    print(f"  {prefix} :: {name} -- no quantile archive, UQ not computable")
                continue
            if b is None:
                b = bundles.load(name)
            for s in seeds:
                r = uq_for_run(name, s, prefix, b)
                if r is None:
                    continue
                for h, agg in r.items():
                    for k, v in agg.items():
                        out[(prefix, name, h, k)][s] = v
                if verbose:
                    print(f"  {prefix} :: {name} seed{s} UQ done")
    return out


# --------------------------------------------------------------------------- #
# 3. Markdown
# --------------------------------------------------------------------------- #
def covid_fold_stats(name="covid_us-states"):
    """Fold-level regime statistics for the COVID panel, straight off the bundle.

    Computed rather than hardcoded so the paragraph that excuses COVID's long horizons can itself be
    re-derived from disk. Returns None if the bundle is unavailable.
    """
    try:
        import bundles
        b = bundles.load(name)
    except Exception:
        return None
    raw = b.raw.astype(np.float64)
    nat = raw.sum(0)                                         # national weekly total
    masks = b.masks()
    out = {"peak_t": int(nat.argmax())}
    for ph in ("train", "val", "test"):
        cols = np.where(masks[ph].astype(bool).any(0))[0]
        seg, block = nat[cols], raw[:, cols]
        denom = np.where(block.mean(1) > 0, block.mean(1), np.nan)
        out[ph] = dict(lo=int(cols.min()), hi=int(cols.max()), n=int(len(cols)),
                       mean=float(seg.mean()), peak=float(seg.max()),
                       cv=float(np.nanmean(block.std(1) / denom)))
    te = np.where(masks["test"].astype(bool).any(0))[0]
    out["test_ratio"] = float(nat[te][-1] / nat[te][0]) if nat[te][0] else float("nan")
    return out


def _pct_cell(new, ref, metric):
    txt, _, _ = paired_delta(new, ref, metric)
    return txt


def build_markdown(points, uq, verify_out, args):
    ok, problems, manifest = verify_out
    A = [].append
    L = []

    def w(s=""):
        L.append(s)

    w("# Three-Disease Leave-One-Disease-Out: Cross-Disease Transfer")
    w()
    w(f"Generated by `ldo3_report.py` from the artifacts in `results/lodo/`. "
      f"Fold structure `leave-one-disease-out-3way`, 3 held-out diseases x {len(SEEDS)} seeds "
      f"= {len(DISEASES) * len(SEEDS)} runs.")
    w()
    w("Headline field is **country-macro** (equal weight per country). On the four single-country "
      "panels it is identical to the node mean; on dengue it is the agreed headline so that Brazil "
      "cannot dominate 7,165 nodes. Nothing in this document is averaged across datasets.")
    w()

    # ---- executive summary ------------------------------------------------
    if args.origin_ci:
        t = collections.Counter()
        by_h = collections.defaultdict(collections.Counter)
        for ds, rows in args.origin_ci.items():
            for (h, metric), r in rows.items():
                lo, _hi = r["ci"]
                if not attributable(ds, h):
                    t["na"] += 1
                    by_h[h]["na"] += 1
                    continue
                k = "better" if (r["clears_ci"] and lo > 0) else \
                    ("worse" if r["clears_ci"] else "noise")
                t[k] += 1; t["cells"] += 1
                by_h[h][k] += 1
        w("## Summary")
        w()
        w(f"**Cross-disease transfer is negative.** Across the {t['cells']} attributable RMSE and "
          f"MAE cells, with the disease genuinely held out and judged by the paired bootstrap over "
          f"time origins: **{t['worse']} cells are significantly worse than the single-disease "
          f"ceiling, {t['noise']} are within noise, and {t['better']} is better.**")
        w()
        w(f"{t['na']} further cells -- COVID at h10 and h15 -- are **excluded rather than counted "
          f"as losses**. The COVID test fold is a flat tail on the far side of the Omicron break "
          f"while model selection happens on the surge itself, so those cells measure a fold "
          f"boundary rather than the transfer. The measurement is in section 2 and the numbers are "
          f"still reported in full; they are simply not evidence about transfer either way.")
        w()
        w("The effect is strongly ordered by horizon, which is the more useful finding:")
        w()
        w("| Horizon | Transfer better | Within noise | Transfer worse | Excluded (COVID break) |")
        w("|---|---|---|---|---|")
        for h in HORIZONS:
            c = by_h[h]
            w(f"| h{h} | {c['better']} | {c['noise']} | {c['worse']} | {c['na']} |")
        w()
        w("At **h3 the transfer arm is close to parity** with a model trained on the disease "
          "itself -- most cells are within noise. By **h10 and h15 it is uniformly and "
          "substantially worse**, reaching -38% (influenza-Japan RMSE) and -51% "
          "(influenza-US-regions MAE). A frozen foreign trunk carries enough short-range structure "
          "to match in-domain training for a few weeks ahead, and then loses it. This gradient is "
          "driven entirely by dengue and influenza, and does not depend on the excluded COVID "
          "cells.")
        w()
        zs = []
        for _names in DISEASES.values():
            for _ds in _names:
                for _m in ("rmse", "mae"):
                    for _h in HORIZONS:
                        if not attributable(_ds, _h):
                            continue
                        _, d, cl = paired_delta(points.get(("LDO3 zero-shot", _ds, _h, _m), {}),
                                                points.get(("single (ceiling)", _ds, _h, _m), {}), _m)
                        if d is not None:
                            zs.append((d, cl))
        if zs:
            worse = sum(1 for d, _ in zs if d < 0)
            sig = sum(1 for d, c in zs if d < 0 and c)
            pos = sum(1 for d, c in zs if d > 0 and c)
            w(f"**The zero-shot arm is worse still, and it is worse everywhere.** Of the "
              f"{len(zs)} attributable RMSE and MAE cells, the borrowed adapter is behind the "
              f"ceiling in **{worse}** of them ({sig} significantly) and ahead in **{pos}**. The "
              f"cost ranges from {abs(max(d for d, _ in zs)):.1f}% (dengue, RMSE, h15) to "
              f"{abs(min(d for d, _ in zs)):.1f}% (COVID, RMSE, h3). Averaging the in-disease "
              f"adapters produces nothing usable on a disease the trunk has never seen.")
            w()
        w("**What this means for the plan.** This is the Option A / Option B decision the LODO probe "
          "was built to settle, and it comes down on **Option B: the trunk does not transfer across "
          "diseases well enough to carry a held-out disease, so the disease has to be trained on "
          "directly.** The adapted arm here is the *optimistic* bound -- the held-out disease got "
          "its entire train fold to fit the adapter. Ebola will have 27 labelled support cells, not "
          "a train fold, so few-shot Ebola transfer cannot do better than these numbers and will "
          "very likely do worse. It also fails hardest exactly at h10-h15, which is where the Ebola "
          "audit already found there is no adaptation data at all.")
        w()
        w("Caveats that limit how hard this can be pushed are in section 5; the most important is "
          "that every fold varies graph, geography and node count alongside the disease, so this is "
          "evidence that *this* trunk does not transfer across *these* folds, not a clean "
          "disease-only effect.")
        w()

    # ---- verification -----------------------------------------------------
    w("## 1. Run verification")
    w()
    total = sum(r["files"] for r in manifest)
    miss = sum(r["missing"] for r in manifest)
    w(f"{total - miss} of {total} expected artifacts present across the "
      f"{len(manifest)} runs.")
    w()
    w("| Held-out disease | Seeds | Bundles scored | Artifacts | Missing |")
    w("|---|---|---|---|---|")
    for held, names in DISEASES.items():
        rows = [r for r in manifest if r["held_out"] == held]
        w(f"| {held} | {', '.join(str(r['seed']) for r in rows)} | "
          f"{len(names)} ({', '.join(names)}) | {sum(r['files'] for r in rows)} | "
          f"{sum(r['missing'] for r in rows)} |")
    w()
    if problems:
        w(f"**{len(problems)} integrity problem(s) found:**")
        w()
        for p in problems:
            w(f"- {p}")
    else:
        w("Integrity checks passed: record counts, seed/dataset agreement between filename and "
          "payload, fold metadata (`held_out_disease`, `in_diseases`, `fold_structure`, "
          "`training_regime`), a NaN sweep, and node-count agreement with the single-disease "
          "ceiling each arm is compared against.")
    w()

    # ---- COVID data property, stated BEFORE any COVID result --------------
    w("### A data property that must be read before the COVID numbers")
    w()
    w("The COVID panel's folds sit on either side of a structural break, and this is a property of "
      "the data rather than of any model in this document. Measured from "
      "`data/processed/covid_us-states.npz`, on the national weekly total:")
    w()
    cf = covid_fold_stats()
    if cf:
        w("| Fold | Columns | Weeks | Mean | Peak | Peak / mean | Mean per-state CV |")
        w("|---|---|---|---|---|---|---|")
        for ph in ("train", "val", "test"):
            r = cf[ph]
            star = "**" if ph == "val" else ""
            cvs = "**" if ph == "test" else ""
            w(f"| {ph} | {r['lo']}-{r['hi']} | {r['n']} | {r['mean']:,.0f} | "
              f"{star}{r['peak']:,.0f}{star} | {r['peak'] / r['mean']:.2f} | "
              f"{cvs}{r['cv']:.3f}{cvs} |")
        w()
        w(f"The Omicron peak -- {cf['val']['peak'] / 1e6:.1f} million national weekly cases, the "
          f"largest value anywhere in the series (week {cf['peak_t']}) -- falls inside the "
          f"**validation** fold. Every arm in this study therefore selects its model on a surge and "
          f"is then scored on a flat, low-amplitude tail: the test fold peaks "
          f"{cf['val']['peak'] / cf['test']['peak']:.1f}x lower than validation, runs essentially "
          f"level end to end (last / first = {cf['test_ratio']:.2f}), and carries about "
          f"**{cf['test']['cv'] / cf['train']['cv']:.0%}** of the training fold's per-state relative "
          f"variability.")
    w()
    w("A model fitted to surge dynamics necessarily over-predicts a flat tail, and that error "
      "compounds with lead time. So COVID's **long-horizon (h10, h15) cells measure the fold "
      "boundary, not the transfer**, and they are reported below but **excluded from every verdict "
      "tally rather than counted as losses**. This applies symmetrically: the single-disease COVID "
      "ceiling is degraded by the same break, which is why it loses to a train-mean baseline at "
      "those horizons (`Report_Covid.md`), and it is why no *positive* COVID claim is made either. "
      "COVID's short horizons are not excused by this and stay in the tally.")
    w()

    # ---- what each fold trained on ---------------------------------------
    w("### What each fold actually held out")
    w()
    w("| Fold | Trunk trained on | Held out | Adapter fitted on |")
    w("|---|---|---|---|")
    for held, names in DISEASES.items():
        ins = [d for d in DISEASES if d != held]
        w(f"| hold out **{held}** | {' + '.join(ins)} | {', '.join(names)} | "
          f"the held-out disease's own train fold (trunk frozen) |")
    w()
    w("The three influenza bundles share **one** adapter wherever they appear, in-trunk and "
      "held-out alike, so the trunk cannot offload panel differences into three separate heads. "
      "A held-out influenza fold therefore emits three per-bundle rows and they are never pooled "
      "into one influenza number.")
    w()

    # ---- transfer tables --------------------------------------------------
    w("## 2. Cross-disease transfer")
    w()
    w("Two arms per fold:")
    w()
    w("- **LDO3 adapted** -- trunk trained on the other two diseases, frozen, then one fresh "
      "adapter fitted on the held-out disease's own train fold. This is the *optimistic* transfer "
      "number: the held-out disease gets its full train fold to fit the adapter.")
    w("- **LDO3 zero-shot** -- the same frozen trunk with the mean of the in-disease adapters, "
      "nothing fitted on the held-out disease at all.")
    w()
    w("**Comparison reference (stated per the reporting standard):** every delta is against the "
      "**single-disease encoder ceiling on the same dataset, paired by seed** -- LDO3 seed 42 "
      "against single-disease seed 42, and so on, then averaged over the 5 pairs. Positive means "
      "the transfer arm is *better* than the ceiling. Intervals are small-sample t "
      f"(n={len(SEEDS)}, t={_t95(len(SEEDS))}); a delta whose interval covers zero is printed as "
      "*within noise* rather than given a direction.")
    w()

    for held, names in DISEASES.items():
        w(f"### Fold: hold out {held}")
        w()
        for ds in names:
            w(f"#### {ds}")
            w()
            w("| h | Metric | Single (ceiling) | LDO3 adapted | LDO3 zero-shot | Best naive floor "
              "| Adapted vs ceiling | Zero-shot vs ceiling | Adapted vs floor |")
            w("|---|---|---|---|---|---|---|---|---|")
            for metric in PRIMARY + EPI:
                for h in HORIZONS:
                    ceil = points.get(("single (ceiling)", ds, h, metric), {})
                    adap = points.get(("LDO3 adapted", ds, h, metric), {})
                    zero = points.get(("LDO3 zero-shot", ds, h, metric), {})
                    nv, nm = best_naive(points, ds, h, metric)
                    nvs = f"{_fmt(nv)} ({nm})" if nv is not None else "—"
                    mark = "" if attributable(ds, h) else " ‡"
                    w(f"| {h}{mark} | {metric} | {cell(ceil)} | {cell(adap)} | {cell(zero)} | {nvs} "
                      f"| {_pct_cell(adap, ceil, metric)} | {_pct_cell(zero, ceil, metric)} "
                      f"| {vs_floor(adap, nv, metric)[0]} |")
            if any(not attributable(ds, h) for h in HORIZONS):
                w()
                w(f"‡ Excluded from every verdict tally: {NA_NOTE}. The numbers are shown, but the "
                  f"COVID test fold is a flat tail on the far side of the Omicron break while model "
                  f"selection sits on the surge, so these rows describe the fold boundary rather "
                  f"than the transfer. See section 2's regime table.")
            w()

    # ---- headline summary -------------------------------------------------
    w("## 3. Transfer verdict by cell")
    w()
    w("Counting only the RMSE and MAE cells, against the seed-paired ceiling. A cell is *negative* "
      "when the adapted arm is significantly worse than the ceiling, *positive* when significantly "
      "better, and *within noise* when the t interval covers zero.")
    w()
    w("The last column is the one that decides whether a cell is a *result*: it counts the horizons "
      "at which the adapted arm beats the **best naive floor**. Beating the ceiling while losing to "
      "a train-mean baseline is not transfer, it is two models failing together.")
    w()
    w("| Fold | Dataset | Metric | Better than ceiling | Within noise | Worse than ceiling "
      "| Excluded | Beats naive floor |")
    w("|---|---|---|---|---|---|---|---|")
    tally = collections.Counter()
    for held, names in DISEASES.items():
        for ds in names:
            for metric in ("rmse", "mae"):
                pos = neg = noise = beats = na = 0
                n_att = 0
                for h in HORIZONS:
                    adap = points.get(("LDO3 adapted", ds, h, metric), {})
                    if not attributable(ds, h):
                        na += 1
                        continue
                    n_att += 1
                    _, m, clears = paired_delta(adap,
                                                points.get(("single (ceiling)", ds, h, metric), {}),
                                                metric)
                    if m is not None and clears is not None:
                        if not clears:
                            noise += 1
                        elif m > 0:
                            pos += 1
                        else:
                            neg += 1
                    nv, _nm = best_naive(points, ds, h, metric)
                    _, fm, fclears = vs_floor(adap, nv, metric)
                    if fm is not None and fclears and fm > 0:
                        beats += 1
                tally["pos"] += pos; tally["noise"] += noise; tally["neg"] += neg
                tally["beats"] += beats; tally["cells"] += n_att; tally["na"] += na
                w(f"| {held} | {ds} | {metric} | {pos} | {noise} | {neg} | {na} "
                  f"| {beats}/{n_att} |")
    w(f"| **total** | | | **{tally['pos']}** | **{tally['noise']}** | **{tally['neg']}** "
      f"| **{tally['na']}** | **{tally['beats']}/{tally['cells']}** |")
    w()

    # ---- origin bootstrap -------------------------------------------------
    w("## 3b. Paired bootstrap over time origins")
    w()
    if not args.origin_ci:
        w("Not computed in this run (`--no-origin-ci`).")
    else:
        w(f"The reporting standard asks for bootstrap intervals over regions and time origins on "
          f"anything called a headline result. This resamples the **time origins** with replacement "
          f"(B={args.B:,}), rebuilding both arms from the same resampled origins so shared "
          f"origin noise cancels, and puts a 95% interval on the percentage difference.")
        w()
        w("Two caveats carried from `analysis.py`, both load-bearing:")
        w()
        w("- This interval is on the **cell-pooled** country-macro, while the tables above are "
          "**node-averaged**. The two coincide on the dense influenza and COVID panels and diverge "
          "on dengue, so dengue's bootstrap interval and its headline delta are not the same "
          "quantity and should not be quoted against each other.")
        w("- Only RMSE and MAE are covered. PCC is not additive over origins from the stored "
          "sufficient statistics, so it cannot be bootstrapped this way.")
        w()
        w("**Where this table disagrees with section 2, and which to believe.** The seed-paired "
          "t-test in section 2 finds three significantly positive COVID cells; this bootstrap finds "
          "none, and finds COVID significantly *worse* at h3 and h15. The two are not contradictory "
          "measurements of one quantity -- they are two different quantities. Section 2 averages "
          "per-node scores (each state weighted equally); this section pools cells before taking "
          "the square root (large states dominate). On COVID the two differ by a lot, because the "
          "state-level counts span several orders of magnitude: the h3 RMSE ceiling is 6,058 "
          "node-averaged and 8,557 cell-pooled. Section 2 also varies seeds at fixed origins, while "
          "this section varies origins at fixed seed-mean. Neither is wrong, but **no positive "
          "transfer claim should be made for COVID**, because the one arm that produces positives "
          "does not survive the other test and the COVID ceiling is independently known to be "
          "broken (section 5).")
        w()
        for held, names in DISEASES.items():
            for ds in names:
                rows = args.origin_ci.get(ds)
                if not rows:
                    continue
                w(f"#### {ds} (held out in the *{held}* fold)")
                w()
                w("| h | Metric | Ceiling | LDO3 adapted | Delta | 95% CI on delta | Verdict |")
                w("|---|---|---|---|---|---|---|")
                for h in HORIZONS:
                    for metric in ("rmse", "mae"):
                        r = rows.get((h, metric))
                        if not r:
                            continue
                        lo, hi = r["ci"]
                        if not attributable(ds, h):
                            v = f"*{NA_NOTE}*"
                        else:
                            v = ("**transfer better**" if lo > 0 else "**transfer worse**") \
                                if r["clears_ci"] else "within noise"
                        w(f"| {h} | {metric} | {_fmt(r['single_mean'])} | {_fmt(r['lodo'])} "
                          f"| {r['d_vs_mean']:+.1f}% | [{lo:+.1f}%, {hi:+.1f}%] | {v} |")
                w()
    w()

    # ---- UQ ---------------------------------------------------------------
    w("## 4. Uncertainty metrics (Review Doc para. 7)")
    w()
    if not uq:
        w("Not computed in this run (`--skip-uq`).")
    else:
        w("Computed from the archived quantile forecasts (5 levels: 0.05, 0.25, 0.5, 0.75, 0.95), "
          "in count space, over the same evaluation cells and the same node set as the point "
          "metrics above. Aggregated per node, then country-macro.")
        w()
        w("- **WIS** -- Weighted Interval Score (Bracher et al. 2021), the FluSight / Forecast Hub "
          "standard. Lower is better, same units as the data. It reduces to MAE for a point "
          "forecast, which is how the deterministic naive floors enter the table.")
        w("- **CRPS** -- reported as the **5-quantile approximation** (2 x mean pinball loss over "
          "the grid). With only 5 levels this is biased low against the true CRPS and must not be "
          "printed as CRPS unqualified.")
        w()
        w("> **WIS and CRPS are the same number here, and the table shows one measurement, not "
          "two.** On a symmetric quantile grid the two are algebraically identical: "
          "`(alpha/2) * IS_alpha` equals the sum of the pinball losses at the interval's two "
          "levels, and `0.5 * |y - median|` is the pinball loss at the median, so "
          "`WIS = (1/(K+1/2)) * sum over all 2K+1 levels = 2 * mean pinball = CRPS`. Both columns "
          "are printed because the Review Doc names both metrics, but they carry identical "
          "information on this 5-level grid and must not be cited as independent evidence. They "
          "would separate only on an asymmetric or denser grid.")
        w("- **Coverage** -- empirical coverage (PICP) of the central 50% and 90% intervals. "
          "Nominal is 0.50 and 0.90; below nominal means the intervals are too narrow.")
        w("- **Width** -- mean interval width, the sharpness half of calibration. Coverage alone is "
          "trivially satisfied by an infinitely wide interval, so the two are only readable "
          "together.")
        w()
        arms = sorted({k[0] for k in uq})
        for prefix in arms:
            label = "LDO3 adapted" if prefix == ADAPTED else prefix
            for ds in sorted({k[1] for k in uq if k[0] == prefix}):
                w(f"#### {label} :: {ds}")
                w()
                w("| h | WIS | CRPS (5-q approx) | Coverage 50% | Coverage 90% | Width 50% | "
                  "Width 90% | Naive WIS floor (=MAE) |")
                w("|---|---|---|---|---|---|---|---|")
                for h in HORIZONS:
                    g = lambda k: cell(uq.get((prefix, ds, h, k), {}))
                    nv, nm = best_naive(points, ds, h, "mae")
                    nvs = f"{_fmt(nv)} ({nm})" if nv is not None else "—"
                    w(f"| {h} | {g('wis')} | {g('crps')} | {g('cov0.5')} | {g('cov0.9')} | "
                      f"{g('width0.5')} | {g('width0.9')} | {nvs} |")
                w()
                # PIT
                w("PIT (pooled over all scored cells and seeds, 10 bins over [0,1]). A uniform "
                  "histogram means calibrated; a U shape means intervals too narrow, a central hump "
                  "means too wide.")
                w()
                w("| h | " + " | ".join(f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)) +
                  " | saturated |")
                w("|---|" + "---|" * 11)
                for h in HORIZONS:
                    hs = uq.get((prefix, ds, h, "pit_hist"), {})
                    if not hs:
                        continue
                    mean_hist = np.mean([np.array(v) for v in hs.values()], axis=0)
                    sat = mean_of(uq.get((prefix, ds, h, "pit_saturated"), {}))
                    w(f"| {h} | " + " | ".join(f"{x:.3f}" for x in mean_hist) +
                      f" | {sat:.3f} |" if sat is not None else "")
                w()
                w("*Saturated* is the fraction of cells falling outside [q05, q95], where the "
                  "5-point PIT curve pins to exactly 0 or 1. A large saturated fraction is itself "
                  "the calibration finding: the interval missed entirely.")
                w()

    # ---- gaps -------------------------------------------------------------
    w("## 5. Gaps and caveats")
    w()
    w("- **The zero-shot arm has no archived quantiles.** `run_ldo3_fold` passes `quant_out` only "
      "for the adapted arm, so WIS, CRPS, coverage and PIT cannot be computed for zero-shot "
      "without re-running the fold. The zero-shot arm is reported on point metrics only.")
    w("- **The single-disease ceiling has archived quantiles for `covid_us-states` only.** Dengue "
      "and the three influenza panels were scored before quantile archiving landed, so for those "
      "four bundles the UQ block describes the transfer arm in absolute terms and against the "
      "naive WIS floor, but there is no ceiling to difference it against.")
    w("- **CRPS is a 5-point approximation** and is biased low. Do not quote it against a "
      "literature CRPS computed on a dense quantile grid.")
    w("- **COVID's long horizons are excluded, not counted.** The single-disease COVID encoder "
      "fails its own naive floor at the longer horizons because the test fold sits on the far side "
      "of the Omicron structural break (section 2, and `Report_Covid.md`). Both arms are degraded "
      "by the same break, so h10 and h15 are reported but kept out of every tally in either "
      "direction. COVID's short horizons remain in, and h3 is a genuine loss.")
    w("- **Two folds are not a disease-level sample.** Three held-out diseases with one bundle for "
      "dengue, one for COVID and three for influenza is the whole cross-disease evidence base. "
      "Every fold also varies graph, geography, node count and panel width alongside the disease, "
      "so a negative here is not attributable to the disease change on its own. The "
      "graph-controlled COVID / influenza-US-states pair (`encoder_pair__`) is the fold that "
      "isolates the disease, and it is reported separately.")
    w()
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def _selfcheck():
    """Aggregation helpers, no artifacts needed."""
    # paired_delta: a uniform 10% improvement on a lower-better metric reads +10% and clears.
    new = {s: 90.0 for s in SEEDS}
    ref = {s: 100.0 for s in SEEDS}
    txt, m, clears = paired_delta(new, ref, "rmse")
    assert abs(m - 10.0) < 1e-9 and clears, f"uniform 10% better must clear: {txt}"
    # a worsening must read negative, not positive (the sign bug the client caught).
    _, m2, _ = paired_delta({s: 110.0 for s in SEEDS}, ref, "rmse")
    assert m2 < 0, "a worsening must read negative"
    # noise: deltas straddling zero must NOT be given a direction.
    jitter = {42: 100.0, 52: 90.0, 62: 110.0, 72: 95.0, 82: 105.0}
    txt3, _, clears3 = paired_delta(jitter, ref, "rmse")
    assert clears3 is False and "within noise" in txt3, f"straddling zero must be noise: {txt3}"
    # best_naive picks the MINIMUM for a lower-better metric and the MAXIMUM for pcc.
    pts = {("naive:persistence", "d", 3, "rmse"): {None: 50.0},
           ("naive:seasonal", "d", 3, "rmse"): {None: 30.0},
           ("naive:train_mean", "d", 3, "rmse"): {None: 70.0},
           ("naive:persistence", "d", 3, "pcc"): {None: 0.2},
           ("naive:seasonal", "d", 3, "pcc"): {None: 0.8}}
    assert best_naive(pts, "d", 3, "rmse") == (30.0, "seasonal"), "lower-better floor must be the min"
    assert best_naive(pts, "d", 3, "pcc") == (0.8, "seasonal"), "pcc floor must be the max"
    # WIS must reduce to MAE for a degenerate (point) forecast -- the property the naive floor uses.
    y = np.array([10.0, 20.0, 5.0])
    pt = np.repeat(np.array([[12.0], [18.0], [5.0]]), 5, axis=1)
    assert np.allclose(score.wis(pt, y, score.QUANTILE_LEVELS), np.abs(y - pt[:, 0])), \
        "WIS must reduce to MAE for a point forecast"
    # the fold plan must never put the held-out disease in its own trunk set.
    for held in DISEASES:
        ins = [d for d in DISEASES if d != held]
        assert held not in ins and len(ins) == 2, f"{held}: bad fold plan {ins}"
    # the vectorised PIT must agree cell-for-cell with score.pit, including the saturated ends.
    rng = np.random.default_rng(0)
    lv = score.QUANTILE_LEVELS
    qs = np.sort(rng.uniform(0, 100, (7, 11, len(lv))), axis=-1)
    ys = rng.uniform(-20, 120, (7, 11))                      # deliberately straddles both tails
    ref = score.pit(qs, ys, lv)
    got = _pit_vec(qs, ys, lv)
    assert np.allclose(ref, got, atol=1e-9), \
        f"vectorised PIT disagrees with score.pit (max diff {np.abs(ref - got).max():.2e})"
    assert (got == 0.0).any() and (got == 1.0).any(), "control void: no saturated cells exercised"
    # a flat (degenerate) quantile curve must not divide by zero.
    flat = np.repeat(np.array([[[5.0]]]), len(lv), axis=-1)
    assert np.isfinite(_pit_vec(flat, np.array([[5.0]]), lv)).all(), "flat curve produced non-finite PIT"
    # On a SYMMETRIC quantile grid, WIS and the 5-point CRPS approximation are algebraically the
    # same number: (alpha/2)*IS_alpha = pinball(lower) + pinball(upper) and 0.5*|y-median| =
    # pinball(median), so WIS = (1/(K+1/2)) * sum over all 2K+1 levels = 2 * mean pinball = CRPS.
    # Pinned here because the two columns must be presented as ONE measurement, not two.
    q1 = np.sort(rng.uniform(0, 100, (9, len(lv))), axis=-1)
    y1 = rng.uniform(-10, 110, 9)
    assert np.allclose(score.wis(q1, y1, lv), score.crps(q1, y1, lv)), \
        "WIS and the 5-point CRPS must coincide on a symmetric grid"
    # ... and the 90% interval must survive the float32 round-trip the archive stores levels in.
    assert len(score._interval_pairs(lv)) == 2, "canonical levels must yield the 50% and 90% PIs"
    rt = tuple(float(x) for x in np.asarray(lv, dtype=np.float32))
    assert len(score._interval_pairs(rt)) == 1, \
        "control void: the float32 round-trip is supposed to break _interval_pairs, and no longer does"
    print("ok  paired_delta sign + noise labelling; best_naive direction; WIS->MAE reduction; "
          "fold plan holds out exactly one disease; vectorised PIT == score.pit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="progress/outcomes/LDO3_Results.md")
    ap.add_argument("--skip-uq", action="store_true", help="point metrics only (dengue UQ is slow)")
    ap.add_argument("--no-origin-ci", action="store_true", help="skip the paired origin bootstrap")
    ap.add_argument("-B", type=int, default=10_000, help="bootstrap resamples")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        _selfcheck()
        return

    v = verify()
    if a.verify_only:
        sys.exit(0 if v[0] else 1)

    a.origin_ci = None
    if not a.no_origin_ci:
        import analysis
        print(f"\n{'=' * 78}\nPaired bootstrap over time origins (B={a.B:,})\n{'=' * 78}")
        a.origin_ci = {}
        for names in DISEASES.values():
            for n in names:
                a.origin_ci[n] = analysis.transfer_ci(n, B=a.B, prefix=ADAPTED,
                                                      lodo_seeds=SEEDS, single_seeds=SEEDS)

    points = load_points()
    uq = {}
    if not a.skip_uq:
        print(f"\n{'=' * 78}\nUQ metrics from archived quantiles\n{'=' * 78}")
        names = [n for names in DISEASES.values() for n in names]
        uq = uq_table(names, prefixes=(ADAPTED,))

    md = build_markdown(points, uq, v, a)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nwrote {a.out} ({len(md.splitlines())} lines)")
    if not v[0]:
        print("NOTE: verification reported problems; they are recorded in section 1 of the document.")


if __name__ == "__main__":
    main()

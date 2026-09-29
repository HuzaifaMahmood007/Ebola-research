"""ebola_ci.py -- the pre-registered Ebola interval, on the statistic we actually report.

    conda run -n ebola-train python ebola_ci.py                 # both arms, both regimes, RMSE+MAE
    conda run -n ebola-train python ebola_ci.py --metric rmse
    conda run -n ebola-train python ebola_ci.py --selfcheck     # logic only, no data, seconds
    conda run -n ebola-train python ebola_ci.py --floor persistence --bonferroni 32   # sensitivity

MULTIPLICITY, AND WHY THE FLOOR FAMILY HAS TO BE NAMED. `--bonferroni M` widens the percentiles to
2.5/M and 100-2.5/M, an overall alpha of 0.05 spread over M cells. It defaults to off, so every
number produced before it existed is unchanged. The canonical family is PERSISTENCE ONLY, both
metrics, both arms, both regimes, four horizons = 32 cells, of which 14 clear zero at 95 percent
(`Reports/Phase0_to_Now_Audit.md:257`). Against support_mean it is 13 of 32 and all 64 together give
27, so a count quoted without its floor family is unreadable: that ambiguity is what produced two
passes disagreeing 9 against 7 on "the 14". Pass `--floor persistence` whenever you quote a count.
This is a sensitivity check on an already-scored frozen pre-registration. It does not rescore Ebola,
which is why the verdict block is skipped under --bonferroni.

WHY THIS EXISTS -- audit findings M1, M2, M3 (Reports/Phase0_to_Now_Audit.md).

M1, THE ESTIMAND MISMATCH. The Ebola headline is a NODE-AVERAGED country-macro: per-node metric,
mean within country, mean over countries (`score.py:255-278`, pinned at `ebola_report.py:47`). The
interval that has been adjudicating the pre-registration is a CELL-POOLED country-macro rebuilt from
per-(origin,country) sufficient statistics (`analysis.py:71-90`). On Ebola RMSE those differ by
1.5-2.0x, and the tell is that the headline point 38.20 sits OUTSIDE its own quoted interval
[38.349, 104.537]. A point and an interval on different estimands cannot adjudicate anything, so
this recomputes the interval on the reported statistic. `assert_reproduces_headline()` refuses to
run unless the point value rebuilt from the per-node archive matches the scored JSON, which is what
makes "same estimand" a checked fact rather than a claim.

M3, THE MISSING AXIS. `Ebola_Prereg.md:188-189` requires a bootstrap "over districts and time
origins" and no amendment narrows it; only origins were ever resampled, because `train/loop.py`
collapses districts into 3 countries before writing the per-origin archive. The district axis is
recoverable from `*__pernode.npz`, which is why this reads that file and not the other one.

M2, THE TWO RULES. `ebola_report.py:91-99` calls a cell "within noise" when the 5 per-seed
improvements straddle zero -- an order-statistic sign check pricing ONLY initialisation variance.
The origin bootstrap seed-averages its inputs and so carries NO seed dispersion. Each prices one
variance component and they disagree on 27 of 64 cells. Here the district resample is drawn ONCE per
draw and applied to all five seeds, and the five seed-level differences are pooled, so a single
interval carries district variance AND seed variance. `--crosstab` prints the disagreement.

READ-ONLY. Consumes archived per-node scores and writes nothing. It cannot disturb a run the
pre-registration allows exactly once, and it needs no re-scoring.

STRATIFIED, AND DELIBERATELY SO. Districts are resampled WITHIN country, each country to its own
size. The estimand weights the three countries equally, so an unstratified draw would let a country
vanish and would fold country-composition variance into what is meant to be a district interval --
acute here, because Ebola has only three countries. `--unstratified` runs the other way as a
sensitivity; the difference belongs in the methods section, not in a footnote.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import ebola_report as ER
from results_paths import rpath

ARMS = ("ebola_L12", "ebola_L20")
REGIMES = (("encoder_ebola", "few-shot"), ("encoder_ebola_zeroshot", "zero-shot"))
FLOORS = ("persistence", "support_mean")          # prereg A5 dropped seasonal-naive
HORIZONS = (3, 5, 10, 15)
SEEDS = (42, 52, 62, 72, 82)                      # the frozen five
B_DEFAULT = 10_000
PCT_DEFAULT = (2.5, 97.5)                         # uncorrected 95%; --bonferroni M overrides it


def load_pernode(path, h, metric):
    """(values, country, node_idx) for one horizon, NaN kept -- aggregate() drops them, so do we."""
    z = np.load(path, allow_pickle=True)
    return (z[f"h{h}__{metric}"].astype(np.float64),
            z[f"h{h}__country"], z[f"h{h}__node_idx"].astype(np.int64))


def country_macro(vals, country, idx=None):
    """score.py aggregate()'s country_macro, exactly: mean within country, then mean over countries.

    `idx` selects nodes (a bootstrap draw); None means all. NaN node values are dropped, and a
    country left with no finite value is dropped from the macro rather than poisoning it -- both
    behaviours copied from score.py:269-274."""
    v = vals if idx is None else vals[idx]
    c = country if idx is None else country[idx]
    per_country = []
    for name in np.unique(c):
        sel = v[c == name]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            per_country.append(sel.mean())
    return float(np.mean(per_country)) if per_country else float("nan")


def strata(country):
    """{country: positions} -- the resampling strata, one per country."""
    return {name: np.flatnonzero(country == name) for name in np.unique(country)}


def draw(country, rng, stratified=True):
    """One bootstrap resample of districts. Stratified keeps all three countries at their own size."""
    if not stratified:
        return rng.integers(0, len(country), size=len(country))
    return np.concatenate([pos[rng.integers(0, len(pos), size=len(pos))]
                           for pos in strata(country).values()])


def assert_reproduces_headline(arm, family, seed, metric, tol=2e-3):
    """The gate on M1: the per-node rebuild must equal the scored JSON's country_macro.

    Without this the module could silently compute a third estimand and nobody would notice -- which
    is precisely the failure being fixed."""
    recs = [r for r in json.load(open(rpath(f"{family}__{arm}__seed{seed}.json"), encoding="utf-8"))
            if r["metric"] == metric]
    got = {}
    for h in HORIZONS:
        vals, country, _ = load_pernode(rpath(f"{family}__{arm}__seed{seed}__pernode.npz"), h, metric)
        got[h] = country_macro(vals, country)
    for r in recs:
        h, want = r["horizon"], float(r["country_macro"])
        if h in got and not np.isclose(got[h], want, rtol=tol, atol=tol):
            raise SystemExit(
                f"ESTIMAND MISMATCH {family} {arm} seed{seed} {metric} h{h}: "
                f"per-node rebuild {got[h]:.6f} != scored country_macro {want:.6f}. "
                f"Refusing to publish an interval on a statistic that is not the reported one.")
    return got


def bonferroni_pct(m):
    """Two-sided bootstrap percentiles for a Bonferroni family of size m at overall alpha=0.05.

    m=16 -> [0.15625, 99.84375], m=32 -> [0.078125, 99.921875]. m=1 returns the uncorrected
    [2.5, 97.5], which is the anchor that shows the formula is the plain one."""
    if m < 1:
        raise SystemExit(f"--bonferroni needs a family size of at least 1, got {m}")
    half = 2.5 / m
    return (half, 100.0 - half)


def paired_ci(arm, family, metric, floor, h, B=B_DEFAULT, seed=0, stratified=True, seeds=SEEDS,
              pct=PCT_DEFAULT):
    """Pooled district bootstrap of (encoder - floor) on the node-averaged country-macro.

    One district resample per draw, shared across all five seeds (paired against the floor and
    paired across seeds), then the five seed-level differences are POOLED -- so the interval carries
    district variance and seed variance in one number, which is the M2 fix."""
    rng = np.random.default_rng(seed)
    fl_v, fl_c, fl_i = load_pernode(rpath(f"naive__{arm}__{floor}__pernode.npz"), h, metric)

    enc = []
    for s in seeds:
        v, c, i = load_pernode(rpath(f"{family}__{arm}__seed{s}__pernode.npz"), h, metric)
        if not np.array_equal(i, fl_i):
            raise SystemExit(f"node misalignment {arm} {family} seed{s} h{h} vs {floor}: "
                             f"{len(i)} vs {len(fl_i)} nodes. The pairing would be meaningless.")
        enc.append(v)

    point = float(np.mean([country_macro(v, fl_c) for v in enc])) - country_macro(fl_v, fl_c)
    diffs = np.empty(B * len(enc))
    k = 0
    for _ in range(B):
        idx = draw(fl_c, rng, stratified)
        base = country_macro(fl_v, fl_c, idx)
        for v in enc:
            diffs[k] = country_macro(v, fl_c, idx) - base
            k += 1
    lo, hi = np.nanpercentile(diffs, list(pct))
    return dict(point=point, lo=float(lo), hi=float(hi), n_nodes=len(fl_i),
                wins=bool(hi < 0), loses=bool(lo > 0))


def report(metric, B, stratified, crosstab, levels=PCT_DEFAULT, floor_names=FLOORS):
    # NOTE the names: `pct` and `floors` are already taken inside this function by a percentage and
    # by ER.naive_floors, so the two new arguments must not reuse either word.
    label_lv = "95%" if levels == PCT_DEFAULT else f"Bonferroni pct [{levels[0]:g}, {levels[1]:g}]"
    print(f"\n{'=' * 100}")
    print(f"EBOLA {metric.upper()} -- paired DISTRICT bootstrap on the node-averaged country-macro "
          f"(B={B}, {'stratified by country' if stratified else 'UNSTRATIFIED'}, {label_lv})")
    print(f"  floor family: {', '.join(floor_names)}")
    print(f"{'=' * 100}")
    disagree = same = 0
    wins = 0
    for arm in ARMS:
        floors = ER.naive_floors(arm, metric)
        for family, label in REGIMES:
            for s in (42, 52, 62, 72, 82):
                assert_reproduces_headline(arm, family, s, metric)
            recs = ER.load_records(arm, family)
            vals_by_h = ER.by_horizon(recs, metric)
            role = "PRIMARY" if arm == "ebola_L12" else "secondary"
            print(f"\n  {arm} ({role})  {label}   [estimand verified against all 5 scored JSONs]")
            for h in HORIZONS:
                seed_vals = vals_by_h.get(h)
                for floor in floor_names:
                    r = paired_ci(arm, family, metric, floor, h, B=B, stratified=stratified,
                                  pct=levels)
                    wins += int(r["wins"])
                    flag = "WIN " if r["wins"] else ("LOSE" if r["loses"] else "n.s.")
                    line = (f"    h{h:<3} vs {floor:13s} d={r['point']:+8.3f} "
                            f"CI [{r['lo']:+8.3f}, {r['hi']:+8.3f}] {flag}  N={r['n_nodes']}")
                    if crosstab and seed_vals:
                        pct, noisy = ER.skill(seed_vals, floors.get(h, {}).get(floor))
                        rule_a = "within noise" if noisy else f"{pct:+.1f}%"
                        agrees = (noisy and not r["wins"] and not r["loses"]) or \
                                 (not noisy and pct > 0 and r["wins"]) or \
                                 (not noisy and pct < 0 and r["loses"])
                        same, disagree = (same + 1, disagree) if agrees else (same, disagree + 1)
                        line += f"   | seed-sd rule: {rule_a:>13s} {'ok' if agrees else 'DISAGREES'}"
                    print(line)
    n_cells = len(ARMS) * len(REGIMES) * len(HORIZONS) * len(floor_names)
    print(f"\n  {metric.upper()} vs {'+'.join(floor_names)}: {wins} of {n_cells} cells clear zero "
          f"at {label_lv}.")
    if crosstab:
        print(f"  M2 cross-tab: the seed-sd rule and this interval disagree on "
              f"{disagree} of {same + disagree} cells.")
    return wins, n_cells


def prereg_verdict(B, stratified):
    """Ebola_Prereg.md:196-198 -- few-shot beats persistence at h3 or h5 on the PRIMARY arm with the
    interval clearing zero. Adjudicated on the reported estimand, over districts, for the first time."""
    print(f"\n{'=' * 100}\nPRE-REGISTERED CRITERION, on the reported statistic and E6's district axis"
          f"\n{'=' * 100}")
    hit = False
    for metric in ("rmse", "mae"):
        for h in (3, 5):
            r = paired_ci("ebola_L12", "encoder_ebola", metric, "persistence", h, B=B,
                          stratified=stratified)
            hit = hit or r["wins"]
            print(f"  h{h} {metric.upper():5s} d={r['point']:+8.3f} CI [{r['lo']:+8.3f}, "
                  f"{r['hi']:+8.3f}]  {'CLEARS ZERO' if r['wins'] else 'spans zero'}")
    print(f"\n  VERDICT: the pre-registered positive result is {'MET' if hit else 'NOT met'}, "
          f"now adjudicable because point and interval share an estimand.")


def _selfcheck():
    """The three things that can silently produce a plausible wrong number."""
    v = np.array([1.0, 3.0, 10.0, 20.0])
    c = np.array(["a", "a", "b", "b"])
    # 1. country_macro is mean-within-country then mean-over-countries, NOT a pooled mean.
    assert country_macro(v, c) == np.mean([2.0, 15.0]) == 8.5, "country_macro must weight countries equally"
    v2 = np.array([1.0, 3.0, 10.0, 20.0, 100.0])
    c2 = np.array(["a", "a", "b", "b", "c"])
    assert country_macro(v2, c2) == np.mean([2.0, 15.0, 100.0]), "a third country must count equally"
    # CONTROL: a pooled mean would give 26.8 and must NOT be what we compute
    assert abs(country_macro(v2, c2) - float(np.mean(v2))) > 10, "must not collapse to a pooled mean"
    # 2. NaN nodes are dropped, and a country with no finite value drops out entirely.
    vn = np.array([1.0, np.nan, 10.0, 20.0])
    assert country_macro(vn, c) == np.mean([1.0, 15.0]), "NaN nodes must be dropped, not propagated"
    vd = np.array([np.nan, np.nan, 10.0, 20.0])
    assert country_macro(vd, c) == 15.0, "an all-NaN country must drop out, not poison the macro"
    # 3. Stratified draws keep every country present at its own size; unstratified need not.
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = draw(c2, rng, stratified=True)
        got = c2[idx]
        assert len(idx) == len(c2), "stratified draw must preserve N"
        assert set(np.unique(got)) == {"a", "b", "c"}, "stratified draw must keep all countries"
        assert (got == "c").sum() == 1, "each stratum keeps its own size"
    emptied = any(set(np.unique(c2[draw(c2, rng, stratified=False)])) != {"a", "b", "c"}
                  for _ in range(200))
    assert emptied, "CONTROL: unstratified draws must be able to lose a country (that is why we stratify)"
    print("ok  country_macro reproduces score.py aggregate(): equal weight per country, NaN dropped")
    print("ok  an all-NaN country drops out rather than poisoning the macro")
    print("ok  stratified draws preserve all three countries; unstratified can lose one")
    # 4. Bonferroni percentiles: the anchor M=16 must reproduce the documented pair, and the
    #    default path must be untouched so every existing number stays bit-identical.
    assert bonferroni_pct(1) == PCT_DEFAULT, "M=1 must be the uncorrected 95% interval"
    assert bonferroni_pct(16) == (0.15625, 99.84375), "M=16 anchor from the 2026-09-23 pass"
    assert bonferroni_pct(32) == (0.078125, 99.921875), "M=32, the canonical persistence family"
    # CONTROL: correcting must WIDEN the interval, never narrow it.
    d = np.random.default_rng(0).normal(size=200_000)
    w16, w32 = (np.diff(np.percentile(d, list(bonferroni_pct(m))))[0] for m in (16, 32))
    w95 = np.diff(np.percentile(d, list(PCT_DEFAULT)))[0]
    assert w95 < w16 < w32, "a bigger family must give a wider interval, so fewer survivors"
    print("ok  bonferroni_pct: M=1 is the plain 95%, M=16 and M=32 match the documented levels")
    print("ok  CONTROL: a larger family widens the interval (16 then 32), it never narrows it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", nargs="+", default=["rmse", "mae"])
    ap.add_argument("-B", type=int, default=B_DEFAULT)
    ap.add_argument("--unstratified", action="store_true",
                    help="resample districts globally; sensitivity only, see the module docstring")
    ap.add_argument("--no-crosstab", action="store_true", help="skip the M2 seed-sd comparison")
    ap.add_argument("--bonferroni", type=int, default=None, metavar="M",
                    help="sensitivity only: Bonferroni-correct the interval for a family of M "
                         "cells (M=32 is the canonical persistence family). Omit for plain 95%%.")
    ap.add_argument("--floor", nargs="+", default=list(FLOORS), choices=list(FLOORS),
                    help="which naive floors to score; the canonical family is persistence alone")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); return
    pct = PCT_DEFAULT if a.bonferroni is None else bonferroni_pct(a.bonferroni)
    total = 0
    for m in a.metric:
        total += report(m, a.B, not a.unstratified, not a.no_crosstab, pct, tuple(a.floor))[0]
    if a.bonferroni is not None:
        print(f"\n  SENSITIVITY TOTAL: {total} cells clear zero across metrics "
              f"{'+'.join(a.metric)} vs {'+'.join(a.floor)} at Bonferroni divisor {a.bonferroni}.")
        print("  The frozen pre-registration is adjudicated at 95% and is NOT rescored here, so "
              "the verdict block is skipped under --bonferroni.")
        return
    prereg_verdict(a.B, not a.unstratified)
    print("\nNOTE: this interval resamples DISTRICTS. The origin axis needs per-(origin,node) "
          "sufficient stats, which the archives do not carry; E6 names both and this delivers one.")


if __name__ == "__main__":
    main()

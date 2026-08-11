"""ebola_report.py -- read the scored Ebola case study off disk. G3, the headline deliverable.

    conda run -n ebola-train python ebola_report.py                # both arms, RMSE + MAE
    conda run -n ebola-train python ebola_report.py --metric rmse  # one metric
    conda run -n ebola-train python ebola_report.py --selfcheck    # logic only, no data, seconds

WHY THIS EXISTS. `train/ebola.py` writes 20 JSON record files and nothing in the repo reads them.
The case study is the required deliverable and its numbers were sitting unread. This is a reader,
not a re-scorer: it opens the archives, aggregates over seeds, and prints. It computes no forecast
and writes no artifact, so it cannot disturb a run the pre-registration allows exactly once.

WHAT IT REPORTS, AND THE RULES IT FOLLOWS.

  * DISPERSION ALWAYS (Week-4 work order standing rule). Every cell is mean +- sd over the 5 seeds,
    never a bare point estimate.
  * THE REFERENCE IS LABELLED. Skill against the naive floors is stated as a percentage with the
    floor named, because "improves by 12%" is meaningless without saying over what.
  * NEVER AVERAGE ACROSS ARMS. L12 and L20 are different pre-registered support sets, not two runs
    of one thing, so they never share a mean.
  * WITHIN NOISE. A skill figure whose seed sd covers zero is printed as "within noise" rather than
    as a number that looks like a result (D3).

THE FLOORS ARE DETERMINISTIC. `naive__ebola_L{12,20}.json` holds persistence and support_mean at one
value per (horizon, metric) with no seed dimension, so the comparison is a 5-seed mean against a
constant. The sd shown on a skill column is therefore the encoder's own seed spread, not a paired
difference, and it is labelled as such.

PREREG INTEGRITY IS CHECKED, NOT ASSUMED. Every record carries `prereg_sha256`, `support_cells` and
`support_cutoff`. If the 20 files do not all agree on those per arm, the run mixed configurations and
the table is not a single experiment; this refuses to print rather than average across them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

EBOLA_DIR = "results/ebola"
NAIVE = "results/naive/naive__{arm}.json"
ARMS = ("ebola_L12", "ebola_L20")
REGIMES = (("encoder_ebola", "few-shot"), ("encoder_ebola_zeroshot", "zero-shot"))
HORIZONS = (3, 5, 10, 15)
FIELD = "country_macro"          # the project's headline aggregate; node_mean printed beside it


def load_records(arm, family):
    """All seed records for one (arm, regime). Returns [] if the family was never written."""
    out = []
    for p in sorted(glob.glob(os.path.join(EBOLA_DIR, f"{family}__{arm}__seed*.json"))):
        with open(p, encoding="utf-8") as f:
            out += json.load(f)
    return out


def check_prereg(recs, label):
    """Every record must agree on the frozen config. Returns the shared (sha, cells, cutoff)."""
    keys = {(r.get("prereg_sha256"), r.get("support_cells"), r.get("support_cutoff")) for r in recs}
    if len(keys) != 1:
        raise SystemExit(f"{label}: records disagree on the frozen config -> {keys}\n"
                         f"This is not one experiment; refusing to average across it.")
    return keys.pop()


def by_horizon(recs, metric, field=FIELD):
    """{h: [value per seed]} for one metric."""
    out = defaultdict(list)
    for r in recs:
        if r["metric"] == metric:
            out[r["horizon"]].append(float(r[field]))
    return out


def naive_floors(arm, metric):
    """{h: {model: value}} for the deterministic floors. {} if the file is absent."""
    path = NAIVE.format(arm=arm)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)
    out = defaultdict(dict)
    for r in recs:
        if r["metric"] == metric:
            out[r["horizon"]][r["model"]] = float(r[FIELD])
    return out


def skill(vals, floor):
    """(mean % improvement over the floor, whether the seed spread covers zero improvement).

    Lower error is better, so improvement = (floor - ours) / floor. `within_noise` is true when the
    per-seed improvements straddle zero, i.e. some seeds beat the floor and some lose to it."""
    if floor is None or floor == 0 or not vals:
        return None, False
    imp = [(floor - v) / floor * 100.0 for v in vals]
    return float(np.mean(imp)), (min(imp) <= 0.0 <= max(imp))


def report(metric, arms=ARMS):
    for arm in arms:
        floors = naive_floors(arm, metric)
        best_name = None
        for family, regime in REGIMES:
            recs = load_records(arm, family)
            if not recs:
                print(f"\n{arm} {regime}: no records on disk"); continue
            sha, cells, cutoff = check_prereg(recs, f"{arm} {regime}")
            seeds = sorted({r["seed"] for r in recs})
            vals = by_horizon(recs, metric)
            nodes = {r["n_nodes"] for r in recs}

            print(f"\n=== {arm}  {regime}  {metric.upper()} ({FIELD}) ===")
            print(f"    support {cells} cells to {cutoff} | seeds {seeds} | nodes {sorted(nodes)}")
            print(f"    prereg {sha}")
            print(f"    {'h':>3s} {'mean +- sd':>20s} {'n':>3s} | "
                  f"{'vs persistence':>16s} {'vs support_mean':>17s}")
            for h in HORIZONS:
                v = vals.get(h, [])
                if not v:
                    continue
                cell = f"{np.mean(v):10.2f} +- {np.std(v, ddof=1):7.2f}"
                bits = []
                for fl in ("persistence", "support_mean"):
                    s, noisy = skill(v, floors.get(h, {}).get(fl))
                    bits.append("        n/a" if s is None else
                                ("within noise" if noisy else f"{s:+9.1f}%"))
                print(f"    {h:3d} {cell:>20s} {len(v):3d} | {bits[0]:>16s} {bits[1]:>17s}")
            if floors:
                best_name = ", ".join(sorted(next(iter(floors.values())).keys()))
        if best_name:
            print(f"\n    reference floors for {arm}: {best_name} "
                  f"(deterministic, one value per horizon, no seed dimension; the sd above is the "
                  f"encoder's own seed spread)")


def _selfcheck():
    """The two rules that can quietly produce a wrong table: the skill sign, and the noise flag."""
    # improvement is positive when OUR error is LOWER than the floor. A sign slip here would print
    # every regression as a win, which is the single most damaging bug this file could have.
    s, noisy = skill([50.0, 50.0, 50.0], 100.0)
    assert abs(s - 50.0) < 1e-9, f"half the floor's error is +50% skill, got {s}"
    assert not noisy
    s, _ = skill([200.0, 200.0, 200.0], 100.0)
    assert abs(s + 100.0) < 1e-9, "CONTROL: twice the floor's error must be -100%, not +100%"

    # a spread that straddles the floor must read as noise, not as its mean
    s, noisy = skill([90.0, 110.0], 100.0)
    assert noisy, "seeds on both sides of the floor must flag within-noise"
    assert abs(s - 0.0) < 1e-9
    _, noisy = skill([80.0, 90.0], 100.0)
    assert not noisy, "CONTROL: seeds all beating the floor must NOT be flagged as noise"

    # missing or zero floor must degrade to n/a, never to a divide-by-zero or a fake number
    assert skill([1.0], None) == (None, False) and skill([1.0], 0.0) == (None, False)
    assert skill([], 100.0) == (None, False)

    # the prereg guard must refuse a mixed set, and pass a clean one
    ok = [dict(prereg_sha256="a", support_cells=59, support_cutoff="2014-06-28")] * 3
    assert check_prereg(ok, "t") == ("a", 59, "2014-06-28")
    try:
        check_prereg(ok + [dict(prereg_sha256="b", support_cells=59,
                                support_cutoff="2014-06-28")], "t")
        raise AssertionError("CONTROL: a mixed prereg hash must refuse, not average")
    except SystemExit:
        pass

    print("ok  skill sign: lower error is positive; twice the floor reads -100%, not +100%")
    print("ok  noise flag: seeds straddling the floor read 'within noise'; all-beating does not")
    print("ok  missing/zero floor degrades to n/a")
    print("ok  prereg guard refuses records that disagree on the frozen config")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", nargs="+", default=["rmse", "mae"])
    ap.add_argument("--arm", nargs="+", default=list(ARMS))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); return
    for m in a.metric:
        report(m, a.arm)
    print("\nNOTE: L12 and L20 are separate pre-registered support sets. They are never averaged.")


if __name__ == "__main__":
    main()

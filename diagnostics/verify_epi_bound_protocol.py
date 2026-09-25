"""verify_epi_bound_protocol.py -- read the numbers back OUT of
progress/decisions/Epi_Bound_Lambda_Protocol.md and recompute every one from disk.

Deliberately does NOT import ablation/run_epi_ablation.py. That runner is the thing under test, so a
bug shared with it would verify itself. Instead:
  * r_max, the sweep's two percentage columns and the per-panel probe are rebuilt here from the raw
    rate arrays and the quantile archives. ablation.epi_penalty is imported for observed_rates /
    probe_rates / GAPS / PAIRS only, because that is the penalty module, not the runner, and the
    AGGREGATION (quantile then max/median/min across datasets, then the violation fractions) is
    re-derived in this file.
  * the runner's constants, its tag() rule, its gate threshold and the line number of the old
    significance rule are scraped from its SOURCE TEXT with regex and compared against the protocol.
    tag() is reimplemented here from the scraped format strings and never called.
  * every statistic (t critical value, false-call rate, paired sd, flag counts, runtimes) is
    reimplemented locally.

    python diagnostics/verify_epi_bound_protocol.py
    python diagnostics/verify_epi_bound_protocol.py --mutate    # corrupt the doc and the inputs
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DOC = ROOT / "progress" / "decisions" / "Epi_Bound_Lambda_Protocol.md"
RUNNER = ROOT / "ablation" / "run_epi_ablation.py"
PENALTY = ROOT / "ablation" / "epi_penalty.py"
RMAX_JSON = ROOT / "ablation" / "epi_rmax.json"
LOGS = {"epi_p90max": ROOT / "results" / "reports" / "epi_p90max.log",
        "epi_p99max": ROOT / "results" / "reports" / "epi_p99max.log"}

SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
METRICS = ("rmse", "mae")
PRINTED = ("rmse", "mae", "pcc")            # what --report prints; PCC decides nothing
# the four panels the arms train on, dengue excluded. Order matches the protocol's tables.
PANELS = ("influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states")
# the five (quantile, aggregator) bounds the protocol's section 2 table rows describe
BOUNDS = ((0.99, "max"), (0.90, "max"), (0.99, "median"), (0.99, "min"), (0.95, "median"))


# --------------------------------------------------------------------------- #
# Disk side
# --------------------------------------------------------------------------- #
def _rmax(rates, dev, q, agg):
    """r_max re-derived here: each dataset's per-gap quantile, then one aggregator across datasets."""
    from ablation.epi_penalty import GAPS
    f = {"max": max, "median": lambda v: float(np.median(v)), "min": min}[agg]
    return {g: float(f([float(np.quantile(rates[n][g], q)) for n in dev])) for g in GAPS}


def _data_frac(rates, names, rm):
    """Fraction of REAL observed training transitions above the bound, pooled over `names`."""
    v = t = 0
    for n in names:
        for g, r in rates[n].items():
            v += int((r > rm[g]).sum())
            t += len(r)
    return 100.0 * v / max(t, 1)


def _probe_cell(by_gap, rm):
    """(violations, total, worst ratio) for one (panel, seed) against one bound."""
    v = t = 0
    w = 0.0
    for g, r in by_gap.items():
        v += int((r > rm[g]).sum())
        t += len(r)
        w = float("inf") if rm[g] <= 0 else max(w, float(r.max() / rm[g]))
    return v, t, w


def _logs():
    """{tag: {(dataset, seed): (minutes, mean train penalty)}} from the two released run logs."""
    pat = re.compile(r"^\s+(\S+) seed(\d+) done in ([\d.]+) min\s+mean train penalty = ([\d.e+-]+)")
    out = {}
    for tag, p in LOGS.items():
        cells = {}
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pat.match(line)
            if m:
                cells[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
        out[tag] = cells
    return out


def _runtime(logs):
    """Runtime model: mean measured minutes per cell PER PANEL over both released logs, then the
    protocol's arm shapes. Per-panel matters because us-states is 6x covid."""
    per = {}
    for ds in PANELS:
        v = [c[(ds, s)][0] for c in logs.values() for s in SEEDS if (ds, s) in c]
        per[ds] = float(np.mean(v))
    arm4 = 5 * sum(per[ds] for ds in PANELS)                 # one 4-panel arm, 20 cells
    covid_arm = 5 * per["covid_us-states"]                   # one COVID-only arm, 5 cells
    flu = 5 * sum(per[ds] for ds in PANELS if ds != "covid_us-states")
    allmin = [t for c in logs.values() for t, _ in c.values()]
    return dict(per_panel=per, cell_lo=min(allmin), cell_hi=max(allmin),
                grid_h=(4 * arm4 + 2 * covid_arm) / 60.0,   # the protocol's 90-cell grid
                two_p99max_arms_h=2 * arm4 / 60.0,          # two FULL 4-panel p99max arms
                thirty_nulls_h=2 * flu / 60.0,              # only the 30 flu cells of those arms
                covid_cell_min=per["covid_us-states"])


def _rec(path, field="node_mean"):
    return {(r["horizon"], r["metric"]): r[field]
            for r in json.loads(Path(path).read_text()) if r["model"] == "encoder"}


def _sd(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(((x - x.mean()) ** 2).sum() / (len(x) - 1)))


def _old_rule_flags(tag, metrics=PRINTED):
    """How many cells the released report flagged under the OLD rule, |mean d| >= sd d.

    Over PRINTED (PCC included) that is the grid the protocol's "three flags" and "48 cells" refer
    to; over METRICS only it is the "one of the three sits on a deciding metric" clause."""
    flags, cells = 0, 0
    for ds in PANELS:
        for h in HORIZONS:
            for m in metrics:
                d = [_rec(ROOT / "ablation" / "single" / f"encoder__{ds}__seed{s}__{tag}.json")[(h, m)]
                     - _rec(ROOT / "results" / "single" / f"encoder__{ds}__seed{s}.json")[(h, m)]
                     for s in SEEDS]
                cells += 1
                mu, sd = float(np.mean(d)), _sd(d)
                if sd > 0 and abs(mu) >= sd:
                    flags += 1
    return flags, cells


def _sources():
    """Runner and penalty-module constants, scraped from source text. Nothing is called."""
    rs = RUNNER.read_text(encoding="utf-8")
    ps = PENALTY.read_text(encoding="utf-8")
    g = lambda s, pat, fl=re.M: re.search(pat, s, fl)
    lines = rs.splitlines()
    rule_lines = [i + 1 for i, l in enumerate(lines) if "abs(dm) < dsd" in l]
    # tag(): read the two format strings and the backward-compatibility condition, do not call tag()
    tag_base = g(rs, r'base = f"epi_p\{int\(round\(q \* 100\)\)\}\{aggregator\}"') is not None
    tag_lam = g(rs, r'return base if lam == DEFAULT_LAMBDA else f"\{base\}_lam\{lam:g\}"') is not None
    return dict(
        expdir=g(rs, r'^EXPDIR = ROOT / "experiments" / "([\w_]+)"').group(1),
        protocol=g(rs, r'^PROTOCOL = ROOT / "progress" / "decisions" / "([^"]+)"').group(1),
        model=g(rs, r'^MODEL = "(\w+)"').group(1),
        default_field=g(rs, r'return HEADLINE\.get\(ds, "(\w+)"\)').group(1),
        tag_base=tag_base, tag_lam=tag_lam,
        default_lambda=float(g(ps, r"^DEFAULT_LAMBDA = ([\d.]+)").group(1)),
        rule_lines=rule_lines,
        share_formula=g(rs, r"share = \(a\.lam \* pen_mean / pin_mean\)") is not None,
        gate=float(g(rs, r"if v == v and v < ([\d.]+)").group(1)),
        rec_fields=sorted(set(re.findall(r"\b(epi_\w+)=", rs))),
        n_planted=len(re.findall(r"lambda q, ag, lam=DEFAULT_LAMBDA:", rs)),
        seeds_default=g(rs, r'"--seeds".*?default=list\(L\.SEEDS\)', re.S) is not None,
        dengue_cost=g(rs, r"Dengue alone is ~([\d.]+) h for 5 seeds, (\d+)% of the").groups(),
        writes_ebola="results/ebola" in rs or 'RESULTS = ROOT / "results" / "ebola"' in rs,
        runner_flags=set(re.findall(r'add_argument\("(--[\w-]+)"', rs)),
        penalty_flags=set(re.findall(r'add_argument\("(--[\w-]+)"', ps)),
    )


def _local_tag(q, agg, lam, default_lambda):
    """tag() reimplemented from the scraped format strings. The runner is never imported."""
    base = f"epi_p{int(round(q * 100))}{agg}"
    return base if lam == default_lambda else f"{base}_lam{lam:g}"


def disk(overrides=None):
    ov = overrides or {}
    import bundles
    from ablation.epi_penalty import GAPS, PAIRS, observed_rates, probe_rates
    dev = list(bundles.DEV_BUNDLE_NAMES)
    rates = {n: observed_rates(n) for n in dev}
    if "rate_scale" in ov:                                   # input mutation: move one panel's rates
        n, g, f = ov["rate_scale"]
        rates[n] = {gg: (r * f if gg == g else r) for gg, r in rates[n].items()}
    pr = probe_rates(PANELS, SEEDS)

    d = {"gaps": GAPS, "pairs": tuple(PAIRS), "dev": dev, "seeds": SEEDS}
    d["rmax"] = {b: _rmax(rates, dev, *b) for b in BOUNDS}
    d["data_pct"] = {b: _data_frac(rates, PANELS, d["rmax"][b]) for b in BOUNDS}
    # "model >bound" is scored on seed 42 only, which is what --sweep prints at its default --seeds
    d["model_pct"] = {}
    d["probe"] = {}
    for b in BOUNDS:
        rm = d["rmax"][b]
        cells = {ds: [_probe_cell(pr[(ds, s)], rm) for s in SEEDS] for ds in PANELS}
        d["probe"][b] = cells
        v42 = sum(cells[ds][0][0] for ds in PANELS)
        t42 = sum(cells[ds][0][1] for ds in PANELS)
        d["model_pct"][b] = 100.0 * v42 / t42
    if "probe_viol" in ov:                                   # input mutation
        b, ds, i, v = ov["probe_viol"]
        row = list(d["probe"][b][ds][i])
        row[0] = v
        d["probe"][b][ds][i] = tuple(row)

    d["rmax_json"] = json.loads(RMAX_JSON.read_text())
    logs = _logs()
    if "log_pen" in ov:                                      # input mutation
        tag, key, f = ov["log_pen"]
        t, p = logs[tag][key]
        logs[tag][key] = (t, p * f)
    d["logs"] = logs
    d["pen"] = {tag: {k: p for k, (_, p) in c.items()} for tag, c in logs.items()}
    d["runtime"] = _runtime(logs)
    d["released"] = {tag: len(list((ROOT / "ablation" / "single").glob(f"encoder__*__seed*__{tag}.json")))
                     for tag in ("epi_p90max", "epi_p99max")}
    d["ref_records"] = sum((ROOT / "results" / "single" / f"encoder__{ds}__seed{s}.json").exists()
                           for ds in PANELS for s in SEEDS)
    d["released_keys"] = sorted(json.loads(
        (ROOT / "ablation" / "single" / "encoder__influenza_japan__seed42__epi_p90max.json").read_text())[0])
    d["tcrit"] = float(stats.t.ppf(0.975, 4))
    d["fcall"] = float(2 * (1 - stats.t.cdf(math.sqrt(5), 4)))
    if "fcall" in ov:
        d["fcall"] = ov["fcall"]
    d["flags"] = {tag: _old_rule_flags(tag) for tag in ("epi_p90max", "epi_p99max")}
    d["flags_dec"] = {tag: _old_rule_flags(tag, METRICS) for tag in ("epi_p90max", "epi_p99max")}
    d["src"] = _sources()
    if "src" in ov:
        d["src"] = dict(d["src"], **ov["src"])
    d["loop_seeds"] = tuple(int(x) for x in re.search(
        r"^SEEDS = \(([\d, ]+)\)", (ROOT / "train" / "loop.py").read_text(encoding="utf-8"),
        re.M).group(1).split(","))
    from results_paths import rpath
    d["route"] = rpath("encoder__covid_us-states__seed42__epi_p99median_lam100.json",
                       root=ROOT / "experiments" / "epi_bound_lambda")
    return d


# --------------------------------------------------------------------------- #
# Document side
# --------------------------------------------------------------------------- #
def _cells(line):
    return [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]


def _eq(printed, value):
    """printed string equals value rounded to the printed precision. Handles 2.2e-05 too."""
    p = str(printed).replace(",", "").replace("%", "").strip().rstrip(".")   # may end a sentence
    mant, _, exp = p.partition("e")
    dec = len(mant.split(".")[1]) if "." in mant else 0
    tol = 0.5 * 10 ** (-dec) * (10 ** int(exp) if exp else 1)
    return abs(float(p) - value) <= tol + 1e-12


def _bound_of(label):
    """'p99 median (new)' -> (0.99, 'median')."""
    m = re.search(r"p(\d+)\s+(max|median|min)", label)
    return (int(m.group(1)) / 100.0, m.group(2)) if m else None


def check(text, d, verbose=True):
    fails = []
    n_checks = [0]

    def ok(cond, msg):
        n_checks[0] += 1
        if not cond:
            fails.append(msg)

    flat = re.sub(r"\s+", " ", text)
    src = d["src"]
    rt = d["runtime"]
    pen90, pen99 = d["pen"]["epi_p90max"], d["pen"]["epi_p99max"]

    # ---------------------------------------------------------------- section 1
    m = re.search(r"mean train penalty is ([\d.]+e-\d+) to ([\d.]+e-\d+)", flat)
    ok(m is not None, "section 1 p90max penalty range sentence not found")
    if m:
        ok(_eq(m.group(1), min(pen90.values())),
           f"p90max penalty low: doc {m.group(1)}, disk {min(pen90.values()):.3e}")
        ok(_eq(m.group(2), max(pen90.values())),
           f"p90max penalty high: doc {m.group(2)}, disk {max(pen90.values()):.3e}")
    # the share range is stated against a pinball "of order LO to HI"; both endpoints must be
    # reachable from the measured penalties with a pinball inside that stated order
    mp = re.search(r"pinball loss of order ([\d.]+) to (\d+)", flat)
    ms = re.search(r"worth roughly ([\d.]+) to ([\d.]+) percent of the objective", flat)
    ok(mp is not None and ms is not None, "section 1 share-of-objective sentence not found")
    if mp and ms:
        plo, phi = float(mp.group(1)), float(mp.group(2))
        for name, pen, pct in (("low", min(pen90.values()), float(ms.group(1))),
                               ("high", max(pen90.values()), float(ms.group(2)))):
            implied = pen / (pct / 100.0)
            ok(plo <= implied <= phi,
               f"share {name} {pct}% needs a pinball of {implied:.3f}, outside the stated {plo} to {phi}")
    zero99 = [k for k, v in pen99.items() if v == 0.0]
    ok(len(zero99) == 10 and {ds for ds, _ in zero99} == {"influenza_us-regions", "influenza_us-states"},
       f"p99max identically-zero cells: disk {len(zero99)} on {sorted({ds for ds, _ in zero99})}")
    ok("those ten cells were the baseline retrained" in flat, "the 'ten cells' sentence changed")
    # COVID is 1.5x to 3x above every other panel, from epi_rmax.json's per_dataset block
    pd = d["rmax_json"]["per_dataset"]
    ratios = [pd["covid_us-states"][str(g)] / pd[n][str(g)]
              for g in d["gaps"] for n in d["dev"] if n != "covid_us-states"]
    m = re.search(r"COVID sits ([\d.]+)x to (\d+)x above every other panel", flat)
    ok(m is not None, "the COVID 1.5x-3x sentence not found")
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        ok(lo <= min(ratios) and max(ratios) <= hi,
           f"COVID ratio span is {min(ratios):.3f} to {max(ratios):.3f}, doc claims {lo} to {hi}")
        ok(min(ratios) < lo + 0.15 and max(ratios) > hi - 0.35,
           f"doc range {lo}-{hi} is loose against the disk span {min(ratios):.2f}-{max(ratios):.2f}")
    ok(all(_eq(f"{d['rmax_json']['r_max'][str(g)]:.10f}", d["rmax"][(0.99, "max")][g]) for g in d["gaps"]),
       "epi_rmax.json r_max does not reproduce from the raw rates at p99 max")
    ok(all(abs(d["rmax_json"]["r_max"][str(g)] - pd["covid_us-states"][str(g)]) < 1e-12 for g in d["gaps"]),
       "`max` does not select COVID at all three gaps in epi_rmax.json")

    # ---------------------------------------------------------------- section 2
    m = re.search(r"Chosen: quantile ([\d.]+), aggregator (\w+)", flat)
    ok(m is not None, "the chosen-bound sentence not found")
    chosen = (float(m.group(1)), m.group(2)) if m else (0.99, "median")
    ok(chosen in BOUNDS, f"chosen bound {chosen} is not one of the swept bounds")
    for g, p in zip(d["gaps"], re.search(
            r"r_max = ([\d.]+) at gap 2, ([\d.]+) at gap 3, ([\d.]+) at gap 5", flat).groups()):
        ok(_eq(p, d["rmax"][chosen][g]),
           f"chosen r_max gap {g}: doc {p}, disk {d['rmax'][chosen][g]:.4f}")
    ok(f"calibrated over all {len(d['dev'])} dev datasets" in flat
       or "over all five dev datasets" in flat and len(d["dev"]) == 5,
       f"the doc's dev-set size disagrees with bundles.DEV_BUNDLE_NAMES ({len(d['dev'])})")

    # the arm table in section 3 starts its rows the same way, so exclude anything naming a lambda
    rows = [_cells(l) for l in text.splitlines()
            if re.match(r"\|\s*\**p\d\d (max|median|min)", l.strip()) and "lambda" not in l]
    ok(len(rows) == 5, f"section 2 bound table has {len(rows)} rows, want 5")
    seen = set()
    for c in rows:
        b = _bound_of(c[0])
        ok(b in d["rmax"], f"unknown bound label {c[0]!r}")
        if b not in d["rmax"]:
            continue
        seen.add(b)
        parts = [x.strip() for x in c[1].split("/")]
        ok(len(parts) == len(d["gaps"]), f"{c[0]}: {len(parts)} r_max values, want {len(d['gaps'])}")
        for g, p in zip(d["gaps"], parts):
            ok(_eq(p, d["rmax"][b][g]), f"{c[0]} r_max g{g}: doc {p}, disk {d['rmax'][b][g]:.4f}")
        ok(_eq(c[2], d["data_pct"][b]), f"{c[0]} data >bound: doc {c[2]}, disk {d['data_pct'][b]:.3f} %")
        ok(_eq(c[3], d["model_pct"][b]),
           f"{c[0]} model >bound: doc {c[3]}, disk {d['model_pct'][b]:.3f} % (4 panels, seed 42)")
    ok(seen == set(BOUNDS), f"section 2 table covers {sorted(seen)}, want all five bounds")

    p99m, p90m = d["rmax"][(0.99, "median")], d["rmax"][(0.90, "max")]
    ok(p99m[2] > p90m[2] and p99m[3] > p90m[3] and p99m[5] < p90m[5],
       "'looser at gaps 2 and 3 but tighter at gap 5' does not hold on disk")
    ok(d["model_pct"][(0.99, "median")] > d["model_pct"][(0.90, "max")]
       and d["data_pct"][(0.99, "median")] < d["data_pct"][(0.90, "max")],
       "'p99 median fires more than p90max and calls fewer real transitions implausible' fails")
    mp = re.search(r"`PAIRS` is `\(\((.*?)\)\)`", flat)
    ok(mp is not None, "the PAIRS sentence not found")
    if mp:
        doc_pairs = tuple(tuple(int(x) for x in p.split(","))
                          for p in re.findall(r"\((\d+,\d+)\)", "((" + mp.group(1) + "))"))
        ok(doc_pairs == d["pairs"], f"doc PAIRS {doc_pairs} vs module {d['pairs']}")
        g5 = sum(1 for a, b in d["pairs"] if b - a == 5)
        g2 = sum(1 for a, b in d["pairs"] if b - a == 2)
        ok(g5 == 2 and g2 == 1 and len(d["pairs"]) == 4,
           f"gap counts: gap5 {g5}, gap2 {g2}, pairs {len(d['pairs'])}")
    m = re.search(r"Calling ([\d.]+) percent of real observed transitions implausible", flat)
    ok(m is not None and _eq(m.group(1), d["data_pct"][(0.95, "median")]),
       f"p95 median data fraction in prose: doc {m and m.group(1)}, "
       f"disk {d['data_pct'][(0.95, 'median')]:.3f}")

    # ---------------------------------------------------------------- section 3
    arms = [_cells(l) for l in text.splitlines()
            if re.match(r"\|\s*\**p\d\d (max|median|min), lambda \d+", l.strip())]
    ok(len(arms) == 6, f"arm table has {len(arms)} rows, want 6")
    total_cells, comparisons, covid_only = 0, 0, 0
    for c in arms:
        b = _bound_of(c[0])
        lam = float(re.search(r"lambda (\d+)", c[0]).group(1))
        want_tag = _local_tag(b[0], b[1], lam, src["default_lambda"])
        ok(c[1].strip("`") == want_tag, f"{c[0]} tag: doc {c[1]}, rule gives {want_tag}")
        npanel = len(PANELS) if c[2].isdigit() and int(c[2]) == len(PANELS) else 1
        ok(c[2] == str(len(PANELS)) or "COVID only" in c[2], f"{c[0]} panels column {c[2]!r}")
        if "COVID only" in c[2]:
            covid_only += 1
        ok(int(c[3]) == npanel * len(SEEDS),
           f"{c[0]} cells: doc {c[3]}, {npanel} panels x {len(SEEDS)} seeds = {npanel * len(SEEDS)}")
        total_cells += int(c[3])
        comparisons += npanel * len(HORIZONS) * len(METRICS)
        for g, p in zip(d["gaps"], [x.strip() for x in c[4].split("/")]):
            ok(_eq(p, d["rmax"][b][g]), f"{c[0]} r_max g{g}: doc {p}, disk {d['rmax'][b][g]:.4f}")
    m = re.search(r"(\d+) cells\. Seeds are ([\d, ]+) in every arm", flat)
    ok(m is not None, "the '90 cells / seeds' sentence not found")
    if m:
        ok(int(m.group(1)) == total_cells, f"stated total {m.group(1)}, arm rows sum to {total_cells}")
        doc_seeds = tuple(int(x) for x in m.group(2).split(","))
        ok(doc_seeds == d["loop_seeds"] == SEEDS,
           f"doc seeds {doc_seeds} vs train.loop.SEEDS {d['loop_seeds']}")
    ok(src["seeds_default"], "the runner's --seeds default is no longer list(L.SEEDS)")
    m = re.search(r"about ([\d.]+) h for five seeds, (\d+) percent of the job", flat)
    ok(m is not None and (m.group(1), m.group(2)) == src["dengue_cost"],
       f"dengue cost: doc {m and m.groups()}, runner docstring {src['dengue_cost']}")
    jm = float(np.mean([pen99[("influenza_japan", s)] for s in SEEDS]))
    m = re.search(r"Japan fires at about 1e-(\d+) there, which even at lambda (\d+) leaves the term "
                  r"near 1e-(\d+)", flat)
    ok(m is not None, "the Japan-fires sentence not found")
    if m:
        ok(-int(m.group(1)) == math.floor(math.log10(jm)),
           f"Japan p99max penalty mean {jm:.3e}, doc says 1e-{m.group(1)}")
        ok(-int(m.group(3)) == math.floor(math.log10(jm * float(m.group(2)))),
           f"Japan mean x{m.group(2)} = {jm * float(m.group(2)):.3e}, doc says 1e-{m.group(3)}")
    m = re.search(r"buy (\d+) guaranteed nulls for ([\d.]+) h of GPU", flat)
    ok(m is not None, "the 30-nulls sentence not found")
    if m:
        want = 2 * (len(PANELS) - 1) * len(SEEDS)
        ok(int(m.group(1)) == want, f"guaranteed nulls: doc {m.group(1)}, 2 arms x 3 panels x 5 seeds = {want}")
        cands = (rt["two_p99max_arms_h"], rt["thirty_nulls_h"])
        ok(min(abs(float(m.group(2)) - c) for c in cands) <= 0.15,
           f"2.9 h claim: doc {m.group(2)}, two full p99max arms {cands[0]:.2f} h, "
           f"the 30 flu cells alone {cands[1]:.2f} h")
    ok(covid_only == 2, f"{covid_only} COVID-only arms in the table, want 2")
    # the 40 released lambda-1 records really exist, 20 per bound
    m = re.search(r"The (\d+) released lambda-1 records in `ablation/single/`", flat)
    ok(m is not None and int(m.group(1)) == sum(d["released"].values()),
       f"released records: doc {m and m.group(1)}, disk {d['released']}")
    ok(d["released"] == {"epi_p90max": 20, "epi_p99max": 20}, f"released record counts {d['released']}")

    # ---------------------------------------------------------------- section 4
    ok(src["model"] == "encoder" and src["default_field"] == "node_mean"
       and "model `encoder`, field `node_mean`" in flat,
       f"reference model/field: runner {src['model']}/{src['default_field']}")
    ok(d["ref_records"] == len(PANELS) * len(SEEDS),
       f"only {d['ref_records']} of {len(PANELS) * len(SEEDS)} reference records exist")
    doc_fields = re.findall(r"`(epi_\w+)`", flat)
    for f in ("epi_quantile", "epi_aggregator", "epi_lam", "epi_r_max", "epi_penalty_mean",
              "epi_pinball_mean", "epi_penalty_share", "epi_protocol_sha256"):
        ok(f in doc_fields and f in src["rec_fields"],
           f"record field {f}: in doc {f in doc_fields}, written by runner {f in src['rec_fields']}")
    ok(not any(k.startswith("epi_") for k in d["released_keys"]),
       "a released lambda-1 record already carries epi_ fields, so the 'only in the filename' claim is wrong")

    # ---------------------------------------------------------------- section 5
    m = re.search(r"report (\d+) of (\d+) planted bugs caught", flat)
    ok(m is not None and int(m.group(1)) == int(m.group(2)) == src["n_planted"],
       f"planted-bug count: doc {m and m.groups()}, runner plants {src['n_planted']}")
    m = re.search(r"COVID seed 42 at `(\w+)`, about ([\d.]+) min", flat)
    ok(m is not None, "the pilot-cell sentence not found")
    if m:
        ok(m.group(1) == _local_tag(0.99, "median", 100.0, src["default_lambda"]),
           f"pilot tag {m.group(1)} is not the p99 median lambda 100 arm")
        cv = [t for c in d["logs"].values() for (ds, _), (t, _) in c.items() if ds == "covid_us-states"]
        ok(0.5 * rt["covid_cell_min"] <= float(m.group(2)) <= 2.0 * rt["covid_cell_min"],
           f"pilot cell: doc about {m.group(2)} min, but the {len(cv)} released COVID epi cells "
           f"averaged {rt['covid_cell_min']:.2f} min (range {min(cv):.1f} to {max(cv):.1f})")
    m = re.search(r"The remaining (\d+) cells, one chained command, about ([\d.]+) h", flat)
    ok(m is not None, "the 89-cells/6-h sentence not found")
    if m:
        ok(int(m.group(1)) == total_cells - 1,
           f"remaining cells: doc {m.group(1)}, grid {total_cells} minus the pilot")
        ok(abs(float(m.group(2)) - rt["grid_h"]) <= 0.5,
           f"grid runtime: doc {m.group(2)} h, per-panel log means give {rt['grid_h']:.2f} h")
    m = re.search(r"Per-cell time across them ranges ([\d.]+) to ([\d.]+) min", flat)
    ok(m is not None, "the per-cell range sentence not found")
    if m:
        ok(_eq(m.group(1), rt["cell_lo"]) and _eq(m.group(2), rt["cell_hi"]),
           f"per-cell range: doc {m.group(1)} to {m.group(2)}, logs give "
           f"{rt['cell_lo']:.1f} to {rt['cell_hi']:.1f}")
    ok(src["expdir"] == "epi_bound_lambda" and "experiments/epi_bound_lambda/single/" in flat
       and d["route"].parent.name == "single" and d["route"].parent.parent.name == src["expdir"],
       f"new records route to {d['route'].parent}, doc claims experiments/{src['expdir']}/single/")
    ok(not src["writes_ebola"] and "Nothing here writes to `results/ebola/`" in flat,
       "the runner mentions results/ebola")
    ok(src["protocol"] == DOC.name, f"the runner freezes {src['protocol']}, not {DOC.name}")

    # ---------------------------------------------------------------- section 6
    m = re.search(r"t\(0\.975, 4\) = ([\d.]+)", flat)
    ok(m is not None and _eq(m.group(1), d["tcrit"]),
       f"t critical value: doc {m and m.group(1)}, scipy {d['tcrit']:.6f}")
    m = re.search(r"`run_epi_ablation\.py:(\d+)`", flat)
    ok(m is not None and int(m.group(1)) in src["rule_lines"],
       f"old-rule line: doc {m and m.group(1)}, `abs(dm) < dsd` is on line(s) {src['rule_lines']}")
    ok("within noise unless |mean d| >= sd d" in flat, "the old rule is no longer quoted verbatim")
    ok("paired t of at least sqrt(n)" in flat, "the sqrt(n) equivalence sentence changed")
    m = re.search(r"so about p = ([\d.]+) at five seeds", flat)
    ok(m is not None and _eq(m.group(1), d["fcall"]),
       f"false-call rate: doc {m and m.group(1)}, 2*(1 - t.cdf(sqrt(5), 4)) = {d['fcall']:.4f}")
    m = re.search(r"Across the (\d+) printed cells of one released arm.*?it yields about ([\d.]+) "
                  r"false flags by chance; the released p90max report produced (\w+)", flat)
    ok(m is not None, "the 48-cells / 4.3-flags sentence not found")
    if m:
        flags90, cells90 = d["flags"]["epi_p90max"]
        ok(int(m.group(1)) == cells90,
           f"cell count: doc {m.group(1)}, the p90max printed grid is {cells90} "
           f"({len(PANELS)} panels x {len(HORIZONS)} horizons x {len(PRINTED)} metrics)")
        ok(_eq(m.group(2), int(m.group(1)) * d["fcall"]),
           f"expected false flags: doc {m.group(2)}, {m.group(1)} x {d['fcall']:.4f} = "
           f"{int(m.group(1)) * d['fcall']:.3f}")
        ok({"one": 1, "two": 2, "three": 3, "four": 4}.get(m.group(3)) == flags90,
           f"p90max flags: doc '{m.group(3)}', recomputed under |mean d| >= sd d = {flags90}")
        ok(flags90 < int(m.group(1)) * d["fcall"], "the 'below that expectation' clause fails")
    # "only one of the three sits on a deciding metric": on RMSE and MAE alone, p90max flags exactly 1
    flags90_dec, _ = d["flags_dec"]["epi_p90max"]
    m2 = re.search(r"only (\w+) of the three sits on a deciding metric", flat)
    ok(m2 is not None and {"one": 1, "two": 2, "three": 3}.get(m2.group(1)) == flags90_dec,
       f"deciding-metric flags: doc '{m2 and m2.group(1)}', RMSE/MAE only gives {flags90_dec}")
    m = re.search(r"The new arms add (\d+) deciding comparisons.*?would carry roughly (\d+) "
                  r"spurious flags", flat)
    ok(m is not None, "the 144-comparisons sentence not found")
    if m:
        ok(int(m.group(1)) == comparisons,
           f"comparisons: doc {m.group(1)}, arm table gives {comparisons} "
           f"(panels x {len(HORIZONS)} horizons x {len(METRICS)} deciding metrics)")
        ok(_eq(m.group(2), int(m.group(1)) * d["fcall"]),
           f"spurious flags: doc {m.group(2)}, {m.group(1)} x {d['fcall']:.4f} = "
           f"{int(m.group(1)) * d['fcall']:.2f}")
    # the gate
    m = re.search(r"inertness gate, at (\d+) percent", flat)
    mg = re.search(r"whose share stays under ([\d.]+)", flat)
    ok(m is not None and mg is not None, "the inertness-gate sentences not found")
    if m and mg:
        ok(float(mg.group(1)) == src["gate"] and float(m.group(1)) / 100.0 == src["gate"],
           f"gate: doc {m.group(1)} percent / {mg.group(1)}, runner {src['gate']}")
    ok(src["share_formula"] and "lambda * mean_train_penalty / mean_train_pinball" in flat,
       "the epi_penalty_share formula in the doc does not match the runner")
    m = re.search(r"at measured p90max lambda-1 shares of ([\d.]+) to ([\d.]+) percent", flat)
    ok(m is not None and (m.group(1), m.group(2)) == (ms.group(1), ms.group(2)) if (m and ms) else False,
       f"the gate's restated share range {m and m.groups()} differs from section 1's {ms and ms.groups()}")
    flu_us = max(pen90[(ds, s)] for ds in ("influenza_us-regions", "influenza_us-states") for s in SEEDS)
    jp_cv = min(pen90[(ds, s)] for ds in ("influenza_japan", "covid_us-states") for s in SEEDS)
    ok(jp_cv > flu_us,
       f"the claimed Japan/COVID vs US-panel asymmetry fails: min(Japan, COVID) {jp_cv:.3e} "
       f"is not above max(US panels) {flu_us:.3e}")

    # per-panel probe, section 6
    pm, pmin, p95 = (0.99, "median"), (0.99, "min"), (0.95, "median")
    ur = d["probe"][pm]["influenza_us-regions"]
    m = re.search(r"zero violations of its ([\d,]+) constrained intervals", flat)
    ok(m is not None and int(m.group(1).replace(",", "")) == ur[0][1],
       f"us-regions total: doc {m and m.group(1)}, disk {ur[0][1]}")
    ok(all(v == 0 for v, _, _ in ur),
       f"us-regions p99 median violations {[v for v, _, _ in ur]}, doc claims zero on every seed")
    ok(all(v == 0 for v, _, _ in d["probe"][pmin]["influenza_us-regions"]),
       f"us-regions p99 min violations {[v for v, _, _ in d['probe'][pmin]['influenza_us-regions']]}")
    u95 = d["probe"][p95]["influenza_us-regions"]
    # p99 median worst-ratio span, the new number the coordinator added
    w99 = [w for _, _, w in ur]
    m = re.search(r"worst prediction reaches only ([\d.]+) to ([\d.]+)x the bound", flat)
    ok(m is not None, "the us-regions p99 median worst-ratio sentence not found")
    if m:
        ok(_eq(m.group(1), min(w99)) and _eq(m.group(2), max(w99)),
           f"us-regions p99 median worst ratio: doc {m.group(1)} to {m.group(2)}, disk "
           f"{min(w99):.2f} to {max(w99):.2f}")
        ok(max(w99) < 1.0, "us-regions p99 median worst ratio is not below 1")
    # p95 median worst-ratio span
    w95 = [w for _, _, w in u95]
    m = re.search(r"worst prediction still only reaches ([\d.]+) to ([\d.]+)x", flat)
    ok(m is not None, "the us-regions p95 median worst-ratio sentence not found")
    if m:
        ok(_eq(m.group(1), min(w95)) and _eq(m.group(2), max(w95)),
           f"us-regions p95 median worst ratio: doc {m.group(1)} to {m.group(2)}, disk "
           f"{min(w95):.2f} to {max(w95):.2f}")
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "nine": 9}
    m = re.search(r"at most (\w+) per seed even at p95 median", flat)
    ok(m is not None and words.get(m.group(1)) == max(v for v, _, _ in u95),
       f"us-regions p95 median worst seed: doc '{m and m.group(1)}', disk {[v for v, _, _ in u95]}")
    us = d["probe"][pm]["influenza_us-states"]
    m = re.search(r"it is (\d+) to (\d+) of ([\d,]+), about ([\d.]+) percent", flat)
    ok(m is not None, "the us-states violation sentence not found")
    if m:
        v = [x for x, _, _ in us]
        ok(int(m.group(1)) == min(v) and int(m.group(2)) == max(v),
           f"us-states p99 median violations: doc {m.group(1)} to {m.group(2)}, disk {sorted(v)}")
        ok(int(m.group(3).replace(",", "")) == us[0][1],
           f"us-states total: doc {m.group(3)}, disk {us[0][1]}")
        ok(abs(float(m.group(4)) - 100 * sum(v) / (len(v) * us[0][1])) <= 0.015,
           f"us-states rate: doc {m.group(4)} percent, disk "
           f"{100 * sum(v) / (len(v) * us[0][1]):.4f} percent")
    jv = sum(v for v, _, _ in d["probe"][pm]["influenza_japan"])
    tv = sum(v for ds in PANELS for v, _, _ in d["probe"][pm][ds])
    m = re.search(r"At p99 median, ([\d,]+) of the ([\d,]+) violations across the four panels are "
                  r"`influenza_japan`, which is ([\d.]+) percent", flat)
    ok(m is not None, "the Japan-share sentence not found")
    if m:
        ok(int(m.group(1).replace(",", "")) == jv, f"Japan violations: doc {m.group(1)}, disk {jv}")
        ok(int(m.group(2).replace(",", "")) == tv, f"total violations: doc {m.group(2)}, disk {tv}")
        ok(_eq(m.group(3), 100 * jv / tv), f"Japan share: doc {m.group(3)}, disk {100 * jv / tv:.2f}")
    m = re.search(r"non-zero penalty on us-regions \(([\d.]+e-\d+)\) where the probe says zero", flat)
    ok(m is not None, "the us-regions 2.3e-05 sentence not found")
    if m:
        cand = [pen90[("influenza_us-regions", s)] for s in SEEDS]
        ok(any(_eq(m.group(1), v) for v in cand),
           f"us-regions p90max penalty: doc {m.group(1)}, disk seeds {['%.3e' % v for v in cand]}")
        ok(all(v == 0 for v, _, _ in d["probe"][(0.90, "max")]["influenza_us-regions"]),
           "the probe does NOT say zero for us-regions at p90max: "
           f"{[v for v, _, _ in d['probe'][(0.90, 'max')]['influenza_us-regions']]}")

    # ---------------------------------------------------------------- section 11
    for flag in ("--selfcheck", "--sweep", "--probe", "--quantile", "--aggregator", "--seeds"):
        ok(flag in src["penalty_flags"], f"epi_penalty.py has no {flag}")
    for flag in ("--selfcheck", "--mutate-selfcheck", "--report"):
        ok(flag in src["runner_flags"], f"run_epi_ablation.py has no {flag}")
    ok("--quantile 0.99 --aggregator median --seeds 42 52 62 72 82" in flat,
       "the reproduce block no longer runs the chosen bound over all five seeds")

    if verbose:
        print(f"{'OK' if not fails else 'FAIL'}: {n_checks[0]} checks, {len(fails)} failures")
        for f in fails:
            print("  " + f)
    return fails


def mutate(text, d):
    """Every corruption must be caught: digit transpositions in the document, and broken inputs so a
    correct document is forced to fail."""
    doc_mut = [
        ("r_max, chosen bound prose", "r_max = 1.0397 at gap 2", "r_max = 1.0937 at gap 2"),
        ("r_max, table cell", "**1.0397 / 0.7838 / 0.5666**", "**1.0397 / 0.7388 / 0.5666**"),
        ("data >bound column", "| **4.157 %** |", "| **4.517 %** |"),
        ("model >bound column", "| **1.626 %** |", "| **1.262 %** |"),
        ("penalty magnitude bound", "2.2e-05 to 6.5e-04", "2.2e-05 to 5.6e-04"),
        ("p90max us-regions penalty", "us-regions (2.3e-05)", "us-regions (3.2e-05)"),
        ("Japan share", "which is 96.7 percent", "which is 69.7 percent"),
        ("us-regions interval total", "of its 9,400 constrained", "of its 9,040 constrained"),
        ("us-regions p95 median count", "at most one per seed", "at most nine per seed"),
        ("us-regions p99 median worst ratio", "reaches only 0.53 to 0.71x", "reaches only 0.35 to 0.71x"),
        ("us-regions p95 median worst ratio", "0.79 to 1.05x", "0.97 to 1.05x"),
        ("deciding-metric flag count", "one of the three sits", "two of the three sits"),
        ("arm-table cell count", "| `epi_p99median_lam100` | 4 | 20 |", "| `epi_p99median_lam100` | 4 | 2 |"),
        ("arm-table total", "90 cells. Seeds are", "80 cells. Seeds are"),
        ("t critical value", "t(0.975, 4) = 2.7764", "t(0.975, 4) = 2.7746"),
        ("gate threshold", "under 0.01 has its null", "under 0.10 has its null"),
        ("gate percent", "inertness gate, at 1 percent", "inertness gate, at 5 percent"),
        ("false-call rate", "0.09 at five seeds", "0.90 at five seeds"),
        ("expected false flags", "about 4.3", "about 3.4"),
        ("comparison count", "add 144 deciding comparisons", "add 414 deciding comparisons"),
        ("old-rule line number", "`run_epi_ablation.py:196`", "`run_epi_ablation.py:164`"),
        ("COVID dominance range", "COVID sits 1.5x to 3x", "COVID sits 5.1x to 3x"),
        ("per-cell time range", "ranges 1.0 to 26.7 min", "ranges 1.0 to 62.7 min"),
        ("grid runtime", "one chained command, about 6 h", "one chained command, about 16 h"),
        ("released record count", "The 40 released lambda-1 records", "The 44 released lambda-1 records"),
    ]
    caught = 0
    for name, a, b in doc_mut:
        assert a in text, f"mutation anchor missing for {name}"
        f = check(text.replace(a, b, 1), d, verbose=False)
        caught += bool(f)
        print(f"  {'caught' if f else 'MISSED'}: doc {name}")

    inputs = [
        ("COVID gap-2 rates +20% (moves r_max at p99 max)",
         lambda: disk({"rate_scale": ("covid_us-states", 2, 1.20)})),
        ("us-regions p99 median violations 0 -> 3 on seed 42",
         lambda: disk({"probe_viol": ((0.99, "median"), "influenza_us-regions", 0, 3)})),
        ("p90max us-regions seed52 penalty x10",
         lambda: disk({"log_pen": ("epi_p90max", ("influenza_us-regions", 52), 10.0)})),
        ("false-call rate forced to 0.05", lambda: disk({"fcall": 0.05})),
        ("runner gate threshold 0.05", lambda: disk({"src": {"gate": 0.05}})),
        ("runner default lambda 2.0", lambda: disk({"src": {"default_lambda": 2.0}})),
    ]
    for name, build in inputs:
        f = check(text, build(), verbose=False)
        caught += bool(f)
        print(f"  {'caught' if f else 'MISSED'}: input {name}")
    total = len(doc_mut) + len(inputs)
    print(f"mutate: {caught} of {total} corruptions caught")
    return caught == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    text = DOC.read_text(encoding="utf-8")
    d = disk()
    fails = check(text, d)
    if a.mutate:
        return 0 if (mutate(text, copy.deepcopy(d)) and not fails) else 1
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

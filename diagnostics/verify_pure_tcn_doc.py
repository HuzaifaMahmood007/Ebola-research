"""verify_pure_tcn_doc.py -- read the numbers back OUT of Pure_TCN_Graph_Removal_2026-09-28.md and
recompute every one of them from the per-seed records on disk with independent code.

The point is to catch a document that has drifted from its artifacts: a stale mean or delta, a flipped
verdict, a tally that no longer sums, a flagged cell dropped from its table, a within-noise cell listed
as flagged, a prose count that no longer matches. So this deliberately does NOT import
experiments.pure_tcn, experiments.gateoff_fresh or ablation.run_epi_ablation: no loader, paired delta or
verdict function is shared with the code that wrote the summaries. Every value is re-derived from the
seed JSONs. The two summary JSONs are only cross-checked against that recompute, never used as a source.

Checks:
  1. completeness: 4 arms x 2 panels x 5 seeds on disk, record stamps right, seed and dataset fields
     match the filename, node_mean equals country_macro; the doc's new-file count.
  2. reproduction: fresh vs archived gate-off identity counts, whole-file identity, max relative diff,
     and the prose claim that the degree test is identical against either reference.
  3. tally table: all six (comparison, panel) rows, including the "what d measures" label.
  4. pure-TCN vs learned table: all 24 cells (means, d, sd, verdict), plus every prose count about it.
  5. h5 split table: 4 rows of three means and three differences, each row adding up; the h5-only and
     opposite-direction claims.
  6. per-seed table: pure-TCN minus gate-off at h5, 4 rows x 5 seeds, and the sign counts in prose.
  7. flagged-cell table: every row matches disk, and coverage both ways for the two split comparisons.
  8. seed-spread table and its three prose counts.
  9. other prose: the null false-flag rate, the COVID PCC sign claim, every manuscript number, the word
     arithmetic of the quoted sentences, and the runtime figures quoted from gate_ablation.log.
 10. summaries: experiments/pure_tcn__summary.json and gateoff_fresh__summary.json agree with the
     recompute, so a stale summary fails here.

Not checked, because they are not results: dates, file timestamps, the runtimes quoted from file
timestamps, line references, and the selfcheck's 1.498 control shift. The manuscript's current word
count and whether the quoted :326 sentence is still in it are printed as info only, because both move
when the manuscript is edited.

"About the same amount" is operationalised as a ratio of 0.8 to 1.25 between the mixing and degree
differences at h5.

    conda run -n ebola-train python diagnostics/verify_pure_tcn_doc.py
    conda run -n ebola-train python diagnostics/verify_pure_tcn_doc.py --mutate   # the check on the check
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "progress" / "outcomes" / "Pure_TCN_Graph_Removal_2026-09-28.md"
MANUSCRIPT = ROOT / "Reports" / "Manuscript_v2.md"
GATE_LOG = ROOT / "results" / "reports" / "gate_ablation.log"
PANELS = ("covid_us-states", "influenza_japan")
SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
POINT = ("rmse", "mae", "pcc")
ERROR = ("rmse", "mae")
N_METRICS = 7
MODEL = "encoder"
FIELD = "node_mean"
ARMS = {"learned": "results/single/encoder__{ds}__seed{s}.json",
        "gateoff": "ablation/single/encoder__{ds}__seed{s}__gateoff.json",
        "fresh": "experiments/gateoff_fresh__{ds}__seed{s}.json",
        "pure": "experiments/pure_tcn__{ds}__seed{s}.json"}
# doc label -> (reference arm, arm, label when the removed part helped, label when it hurt)
COMPARISONS = {"pure-TCN vs learned": ("learned", "pure", "GRAPH HELPS", "graph HURTS"),
               "pure-TCN vs gate-off": ("gateoff", "pure", "DEGREE HELPS", "degree HURTS"),
               "gate-off vs learned": ("learned", "gateoff", "mixing HELPS", "mixing HURTS")}
WHAT = {"pure-TCN vs learned": "the whole graph", "pure-TCN vs gate-off": "the degree feature",
        "gate-off vs learned": "neighbour mixing"}
SPLIT = ("pure-TCN vs gate-off", "gate-off vs learned")
WORD_H = {"three": 3, "five": 5, "ten": 10, "fifteen": 15}
SAME_BAND = (0.8, 1.25)


# --------------------------------------------------------------------------- #
# Disk side: independent loader, paired delta and verdict
# --------------------------------------------------------------------------- #
def load_disk():
    """values[arm][ds][(metric, h)][seed] for the encoder model over all seven metrics, plus the raw
    record lists (stamps and whole-file identity need them)."""
    values = {a: {ds: {} for ds in PANELS} for a in ARMS}
    raw = {a: {ds: {} for ds in PANELS} for a in ARMS}
    for a, pat in ARMS.items():
        for ds in PANELS:
            for s in SEEDS:
                p = ROOT / pat.format(ds=ds, s=s)
                if not p.exists():
                    continue
                recs = json.loads(p.read_text(encoding="utf-8"))
                raw[a][ds][s] = recs
                for r in recs:
                    if r["model"] == MODEL:
                        values[a][ds].setdefault((r["metric"], r["horizon"]), {})[s] = r[FIELD]
    return dict(values=values, raw=raw)


def _mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _verdict(dm, dsd, metric, helps, hurts):
    if dsd == 0 or abs(dm) < dsd:
        return "within noise"
    removed_helped = (dm > 0) if metric in ERROR else (dm < 0)   # worse without the part => it helped
    return helps if removed_helped else hurts


def paired(values, ds, ref, arm, metric, h, helps="helps", hurts="hurts"):
    r, a = values[ref][ds][(metric, h)], values[arm][ds][(metric, h)]
    seeds = sorted(set(r) & set(a))
    d = [a[s] - r[s] for s in seeds]
    dm, dsd = _mean_sd(d)
    (rm, rsd), (am, asd) = _mean_sd([r[s] for s in seeds]), _mean_sd([a[s] for s in seeds])
    return dict(ref_mean=rm, ref_sd=rsd, arm_mean=am, arm_sd=asd, d=dm, sd=dsd, n=len(d),
                pos=sum(x > 0 for x in d), per_seed=dict(zip(seeds, d)),
                verdict=_verdict(dm, dsd, metric, helps, hurts))


def derive(disk):
    v = disk["values"]
    cells = {}
    for name, (ref, arm, he, hu) in COMPARISONS.items():
        for ds in PANELS:
            for m in POINT:
                for h in HORIZONS:
                    cells[(name, ds, m, h)] = paired(v, ds, ref, arm, m, h, he, hu)
    fresh = {(ds, m, h): paired(v, ds, "fresh", "pure", m, h, "DEGREE HELPS", "degree HURTS")
             for ds in PANELS for m in POINT for h in HORIZONS}
    return {**disk, "cells": cells, "vs_fresh": fresh}


def arm_sd(disk, arm, ds, m, h):
    return _mean_sd([disk["values"][arm][ds][(m, h)][s] for s in SEEDS])[1]


def arm_mean(disk, arm, ds, m, h):
    return _mean_sd([disk["values"][arm][ds][(m, h)][s] for s in SEEDS])[0]


# --------------------------------------------------------------------------- #
# Document side
# --------------------------------------------------------------------------- #
def _num(x):
    return float(str(x).replace(",", "").replace("+", "").strip())


def _tol(printed):
    """Half a unit in the last printed digit, plus float slack."""
    s = str(printed).replace(",", "").replace("+", "").replace("-", "").strip()
    return 0.5 * (10 ** -len(s.split(".")[1]) if "." in s else 1.0) + 1e-9


def _off(printed, got):
    return abs(_num(printed) - got) > _tol(printed)


def _flat(text):
    return re.sub(r"\s+", " ", text)


def _rows(text, rx):
    return [m for m in (rx.match(line.strip()) for line in text.splitlines()) if m]


def _find(flat, rx, what, problems):
    m = re.search(rx, flat)
    if not m:
        problems.append(f"{what}: sentence not found (its shape changed)")
    return m


REPRO_RE = re.compile(r"^\|\s*(covid_us-states|influenza_japan)\s*\|\s*(\d+) of (\d+)\s*\|\s*(\d+) of (\d+)"
                      r"\s*\|\s*(\d+) of (\d+)\s*\|\s*([\d.eE+-]+)\s*\|$")
TALLY_RE = re.compile(r"^\|\s*(pure-TCN vs learned|pure-TCN vs gate-off|gate-off vs learned)\s*\|\s*([^|]+?)"
                      r"\s*\|\s*(covid_us-states|influenza_japan)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|$")
MAIN_RE = re.compile(r"^\|\s*(covid_us-states|influenza_japan)\s*\|\s*(RMSE|MAE|PCC)\s*\|\s*(\d+)\s*\|"
                     r"\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*([+-]?[\d.]+)\s*\|\s*([\d.]+)\s*\|"
                     r"\s*(within noise|GRAPH HELPS|graph HURTS)\s*\|$")
SPLIT_RE = re.compile(r"^\|\s*(covid_us-states|influenza_japan)\s*\|\s*(RMSE|MAE)\s*\|\s*([\d.]+)\s*\|"
                      r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*\|"
                      r"\s*([+-][\d.]+)\s*\|$")
SEED_RE = re.compile(r"^\|\s*(covid_us-states|influenza_japan)\s*\|\s*(RMSE|MAE) at h5\s*\|"
                     + r"\s*([+-]?[\d.]+)\s*\|" * 5 + r"$")
FLAG_RE = re.compile(r"^\|\s*(covid_us-states|influenza_japan)\s*\|\s*(pure-TCN vs gate-off|gate-off vs learned)"
                     r"\s*\|\s*(RMSE|MAE|PCC)\s*\|\s*(\d+)\s*\|\s*([+-]?[\d.]+)\s*\|\s*([\d.]+)\s*\|"
                     r"\s*(\d+) of (\d+)\s*\|\s*(DEGREE HELPS|degree HURTS|mixing HELPS|mixing HURTS|within noise)"
                     r"\s*\|$")
SPREAD_RE = re.compile(r"^\|\s*(covid_us-states|influenza_japan)\s*\|\s*(RMSE)\s*\|\s*(learned|pure-TCN)\s*\|"
                       + r"\s*([\d.]+)\s*\|" * 4 + r"$")


# --------------------------------------------------------------------------- #
def check_completeness(text, disk):
    problems = []
    raw = disk["raw"]
    want = {"pure": dict(protocol="EXPLORATORY", gate_mode="off", ltr="off", encoder_version="v1_pure_tcn"),
            "fresh": dict(protocol="EXPLORATORY", gate_mode="off"),
            "gateoff": dict(gate_mode="off"),
            "learned": dict(gate_mode="learned")}
    absent = {"fresh": ("ltr", "encoder_version"), "gateoff": ("protocol", "ltr", "encoder_version"),
              "learned": ("protocol", "ltr", "encoder_version"), "pure": ()}
    for a in ARMS:
        for ds in PANELS:
            missing = [s for s in SEEDS if s not in raw[a][ds]]
            if missing:
                problems.append(f"{a} {ds}: seed files missing for {missing}")
            for s, recs in raw[a][ds].items():
                enc = [r for r in recs if r["model"] == MODEL]
                if len(enc) != N_METRICS * len(HORIZONS):
                    problems.append(f"{a} {ds} seed {s}: {len(enc)} encoder records, expected "
                                    f"{N_METRICS * len(HORIZONS)}")
                for r in recs:
                    bad = [k for k, x in want[a].items() if r.get(k) != x] + [k for k in absent[a] if k in r]
                    if r.get("seed") != s or r.get("dataset") != ds or bad:
                        problems.append(f"{a} {ds} seed {s}: record stamps wrong ({bad or 'seed/dataset'})")
                        break
                    if r["model"] == MODEL and r["node_mean"] != r["country_macro"]:
                        problems.append(f"{a} {ds} seed {s}: node_mean != country_macro")
                        break
    new = sum(len(raw[a][ds]) for a in ("pure", "fresh") for ds in PANELS)
    m = _find(_flat(text), r"All (\d+) new seed files are present", "new-file count", problems)
    if m and int(m.group(1)) != new:
        problems.append(f"doc says {m.group(1)} new seed files, disk has {new}")
    return problems


def check_reproduction(text, disk):
    problems = []
    v, raw = disk["values"], disk["raw"]
    rows = {m.group(1): m for m in _rows(text, REPRO_RE)}
    if set(rows) != set(PANELS):
        problems.append(f"reproduction table: parsed panels {sorted(rows)}, expected {list(PANELS)}")
    for ds, m in rows.items():
        keys = [(k, s) for k in v["gateoff"][ds] for s in SEEDS]
        same = lambda k, s: v["fresh"][ds][k][s] == v["gateoff"][ds][k][s]
        pt = [(k, s) for k, s in keys if k[0] in POINT]
        disk_pt, disk_all = (sum(same(k, s) for k, s in pt), len(pt)), (sum(same(k, s) for k, s in keys), len(keys))
        files = sum([{k: x for k, x in r.items() if k != "protocol"} for r in raw["fresh"][ds][s]]
                    == raw["gateoff"][ds][s] for s in SEEDS)
        rel = []
        for k, s in keys:
            a, b = v["fresh"][ds][k][s], v["gateoff"][ds][k][s]
            rel.append(abs(a - b) / abs(b) if b else (0.0 if a == b else math.inf))
        for name, doc, got in (("RMSE/MAE/PCC identical", (int(m.group(2)), int(m.group(3))), disk_pt),
                               ("all-metric identical", (int(m.group(4)), int(m.group(5))), disk_all),
                               ("whole files identical", (int(m.group(6)), int(m.group(7))), (files, len(SEEDS)))):
            if doc != got:
                problems.append(f"{ds} {name}: doc {doc[0]} of {doc[1]}, disk {got[0]} of {got[1]}")
        printed = m.group(8)
        if (_num(printed) == 0 and max(rel) != 0) or (_num(printed) != 0 and _off(printed, max(rel))):
            problems.append(f"{ds} max relative difference: doc {printed}, disk {max(rel):.3g}")
    if _find(_flat(text), r"the same whether the reference is the archived or the fresh gate-off",
             "archived-or-fresh claim", problems):
        diff = [k for k, c in disk["vs_fresh"].items()
                if (c["d"], c["sd"], c["verdict"]) != (disk["cells"][("pure-TCN vs gate-off",) + k]["d"],
                                                       disk["cells"][("pure-TCN vs gate-off",) + k]["sd"],
                                                       disk["cells"][("pure-TCN vs gate-off",) + k]["verdict"])]
        if diff:
            problems.append(f"degree test differs between archived and fresh gate-off at {diff[:3]}")
    return problems


def check_tally(text, disk):
    problems = []
    rows = _rows(text, TALLY_RE)
    seen = set()
    for m in rows:
        name, what, ds = m.group(1), m.group(2).strip(), m.group(3)
        seen.add((name, ds))
        _, _, he, hu = COMPARISONS[name]
        vs = [disk["cells"][(name, ds, mt, h)]["verdict"] for mt in POINT for h in HORIZONS]
        got = (vs.count(he), vs.count(hu), vs.count("within noise"))
        doc = tuple(int(m.group(i)) for i in (4, 5, 6))
        if what != WHAT[name]:
            problems.append(f"tally {name} {ds}: labelled '{what}', it measures '{WHAT[name]}'")
        if doc != got:
            problems.append(f"tally {name} {ds}: doc helps/hurts/noise {doc}, disk {got}")
    want = {(n, ds) for n in COMPARISONS for ds in PANELS}
    if seen != want or len(rows) != len(want):
        problems.append(f"tally table: parsed {len(rows)} rows, missing {sorted(want - seen)}")
    return problems


def check_main_table(text, disk):
    problems = []
    rows = _rows(text, MAIN_RE)
    seen = set()
    for m in rows:
        ds, mt, h = m.group(1), m.group(2).lower(), int(m.group(3))
        seen.add((ds, mt, h))
        c = disk["cells"][("pure-TCN vs learned", ds, mt, h)]
        for name, printed, got in (("learned", m.group(4), c["ref_mean"]), ("pure-TCN", m.group(5), c["arm_mean"]),
                                   ("d", m.group(6), c["d"]), ("sd", m.group(7), c["sd"])):
            if _off(printed, got):
                problems.append(f"pure vs learned {ds} {mt} h{h}: {name} printed {printed}, disk {got:+.3f}")
        if m.group(8) != c["verdict"]:
            problems.append(f"pure vs learned {ds} {mt} h{h}: verdict {m.group(8)!r}, disk {c['verdict']!r}")
    want = {(ds, mt, h) for ds in PANELS for mt in POINT for h in HORIZONS}
    if seen != want or len(rows) != len(want):
        problems.append(f"pure vs learned table: parsed {len(rows)} rows, missing {sorted(want - seen)}")

    flat = _flat(text)
    err = [(ds, mt, h) for ds in PANELS for mt in ERROR for h in HORIZONS]
    flag = lambda k: disk["cells"][("pure-TCN vs learned",) + k]["verdict"] != "within noise"
    n_flag_err = sum(flag(k) for k in err)
    for rx, what in ((r"\*\*(\d+) of (\d+) error cells differ beyond seed noise\.\*\*", "headline count"),):
        m = _find(flat, rx, what, problems)
        if m and (int(m.group(1)), int(m.group(2))) != (n_flag_err, len(err)):
            problems.append(f"{what}: doc {m.group(1)} of {m.group(2)}, disk {n_flag_err} of {len(err)}")
    for rx, what in ((r"moved none of the (\d+) error cells beyond seed noise", "plain-words count"),
                     (r"moved none of (\d+) forecast-error comparisons beyond seed noise", "headline text count"),
                     (r"moved none of these panels' (\d+) error cells beyond seed noise", "replacement text count")):
        m = _find(flat, rx, what, problems)
        if m and (n_flag_err != 0 or int(m.group(1)) != len(err)):
            problems.append(f"{what}: doc says none of {m.group(1)}, disk has {n_flag_err} flagged of {len(err)}")
    lower = {ds: sum(disk["cells"][("pure-TCN vs learned", ds, mt, h)]["arm_mean"]
                     < disk["cells"][("pure-TCN vs learned", ds, mt, h)]["ref_mean"]
                     for mt in ERROR for h in HORIZONS) for ds in PANELS}
    m = _find(flat, r"numerically lower error than the learned model in (\d+) of (\d+) error cells "
                    r"\(COVID (\d+) of (\d+), Japan (\d+) of (\d+)\)", "numerically-lower count", problems)
    if m:
        doc = tuple(int(g) for g in m.groups())
        got = (sum(lower.values()), len(err), lower["covid_us-states"], len(err) // 2,
               lower["influenza_japan"], len(err) // 2)
        if doc != got:
            problems.append(f"numerically-lower count: doc {doc}, disk {got}")
    flagged = [k for k, c in disk["cells"].items() if k[0] == "pure-TCN vs learned" and c["verdict"] != "within noise"]
    m = _find(flat, r"The one flag is COVID PCC h10: learned (-?[\d.]+), pure-TCN (-?[\d.]+), d (-?[\d.]+) "
                    r"with sd ([\d.]+)\. Both arms are negative there", "the one flag", problems)
    if m:
        k = ("pure-TCN vs learned", "covid_us-states", "pcc", 10)
        if flagged != [k]:
            problems.append(f"'the one flag': disk flags {flagged}")
        c = disk["cells"][k]
        for name, printed, got in (("learned", m.group(1), c["ref_mean"]), ("pure", m.group(2), c["arm_mean"]),
                                   ("d", m.group(3), c["d"]), ("sd", m.group(4), c["sd"])):
            if _off(printed, got):
                problems.append(f"the one flag: {name} printed {printed}, disk {got:+.3f}")
        if not (c["ref_mean"] < 0 and c["arm_mean"] < 0):
            problems.append("the one flag: 'both arms are negative' is false on disk")
    m = _find(flat, r"One flag in (\d+) cells", "one flag in N", problems)
    if m:
        cov = [disk["cells"][("pure-TCN vs learned", "covid_us-states", mt, h)]["verdict"]
               for mt in POINT for h in HORIZONS]
        if (len(cov) - cov.count("within noise"), len(cov)) != (1, int(m.group(1))):
            problems.append(f"one flag in {m.group(1)} cells: disk {len(cov) - cov.count('within noise')} "
                            f"in {len(cov)}")
    return problems


def check_split(text, disk):
    problems = []
    rows = _rows(text, SPLIT_RE)
    seen = set()
    for m in rows:
        ds, mt = m.group(1), m.group(2).lower()
        seen.add((ds, mt))
        cell = lambda name: disk["cells"][(name, ds, mt, 5)]["d"]
        got = (arm_mean(disk, "learned", ds, mt, 5), arm_mean(disk, "gateoff", ds, mt, 5),
               arm_mean(disk, "pure", ds, mt, 5), cell("gate-off vs learned"), cell("pure-TCN vs gate-off"),
               cell("pure-TCN vs learned"))
        names = ("learned", "gate-off", "pure-TCN", "gate-off - learned", "pure - gate-off", "pure - learned")
        for i, (name, g) in enumerate(zip(names, got)):
            if _off(m.group(3 + i), g):
                problems.append(f"h5 split {ds} {mt}: {name} printed {m.group(3 + i)}, disk {g:.1f}")
        a, b, c = (_num(m.group(i)) for i in (6, 7, 8))
        if abs((a + b) - c) > 0.15:
            problems.append(f"h5 split {ds} {mt}: printed differences do not add up ({a} + {b} != {c})")
    want = {(ds, mt) for ds in PANELS for mt in ERROR}
    if seen != want or len(rows) != len(want):
        problems.append(f"h5 split table: parsed {len(rows)} rows, missing {sorted(want - seen)}")

    flat = _flat(text)
    both = {(ds, mt, h) for ds in PANELS for mt in ERROR for h in HORIZONS
            if all(disk["cells"][(n, ds, mt, h)]["verdict"] != "within noise" for n in SPLIT)}
    m = _find(flat, r"Only at the (\w+)-week horizon did both parts move error beyond seed noise, and there, on "
                    r"both panels", "h5-only claim", problems)
    if m:
        h = WORD_H.get(m.group(1))
        if both != {(ds, mt, h) for ds in PANELS for mt in ERROR}:
            problems.append(f"h5-only claim: doc says the {m.group(1)}-week horizon on both panels, disk has "
                            f"both parts flagged at {sorted(both)}")
    if _find(flat, r"At h5, on both panels and on both error metrics, the two parts move error in opposite "
                   r"directions by about the same amount", "opposite-and-equal claim", problems):
        for ds in PANELS:
            for mt in ERROR:
                mix = disk["cells"][("gate-off vs learned", ds, mt, 5)]["d"]
                deg = disk["cells"][("pure-TCN vs gate-off", ds, mt, 5)]["d"]
                ratio = abs(deg) / abs(mix) if mix else math.inf
                if not (mix < 0 < deg) or not (SAME_BAND[0] <= ratio <= SAME_BAND[1]):
                    problems.append(f"opposite-and-equal claim fails at {ds} {mt} h5: mixing d {mix:+.1f}, "
                                    f"degree d {deg:+.1f}")
    return problems


def check_per_seed(text, disk):
    problems = []
    rows = _rows(text, SEED_RE)
    seen = set()
    if "| seed 42 | seed 52 | seed 62 | seed 72 | seed 82 |" not in text:
        problems.append("per-seed table header is not in seed order 42 52 62 72 82")
    for m in rows:
        ds, mt = m.group(1), m.group(2).lower()
        seen.add((ds, mt))
        ps = disk["cells"][("pure-TCN vs gate-off", ds, mt, 5)]["per_seed"]
        for i, s in enumerate(SEEDS):
            if _off(m.group(3 + i), ps[s]):
                problems.append(f"per-seed {ds} {mt} seed {s}: printed {m.group(3 + i)}, disk {ps[s]:+.1f}")
    want = {(ds, mt) for ds in PANELS for mt in ERROR}
    if seen != want or len(rows) != len(want):
        problems.append(f"per-seed table: parsed {len(rows)} rows, missing {sorted(want - seen)}")
    d = [x for ds in PANELS for mt in ERROR
         for x in disk["cells"][("pure-TCN vs gate-off", ds, mt, 5)]["per_seed"].values()]
    flat = _flat(text)
    m = _find(flat, r"All (\d+) per-seed differences are positive", "per-seed positive count", problems)
    if m and (int(m.group(1)) != len(d) or not all(x > 0 for x in d)):
        problems.append(f"per-seed positive: doc all {m.group(1)}, disk {sum(x > 0 for x in d)} of {len(d)}")
    m = _find(flat, r"all (\d+) per-seed differences have the same sign", "per-seed same-sign count", problems)
    if m and (int(m.group(1)) != len(d) or not (all(x > 0 for x in d) or all(x < 0 for x in d))):
        problems.append(f"per-seed same sign: doc all {m.group(1)}, disk not all one sign over {len(d)}")
    return problems


def check_flagged(text, disk):
    problems = []
    rows = _rows(text, FLAG_RE)
    in_doc = set()
    for m in rows:
        ds, name, mt, h = m.group(1), m.group(2), m.group(3).lower(), int(m.group(4))
        key = (name, ds, mt, h)
        in_doc.add(key)
        c = disk["cells"][key]
        if c["verdict"] == "within noise":
            problems.append(f"{key}: listed as flagged, disk says within noise")
            continue
        if m.group(9) != c["verdict"]:
            problems.append(f"{key}: verdict {m.group(9)!r}, disk {c['verdict']!r}")
        for nm, printed, got in (("d", m.group(5), c["d"]), ("sd", m.group(6), c["sd"])):
            if _off(printed, got):
                problems.append(f"{key}: {nm} printed {printed}, disk {got:+.3f}")
        if (int(m.group(7)), int(m.group(8))) != (c["pos"], c["n"]):
            problems.append(f"{key}: seeds with d > 0 printed {m.group(7)} of {m.group(8)}, disk {c['pos']} of {c['n']}")
    on_disk = {k for k, c in disk["cells"].items() if k[0] in SPLIT and c["verdict"] != "within noise"}
    for k in sorted(on_disk - in_doc):     # the other direction, a within-noise row, is reported above
        problems.append(f"{k} is flagged on disk but missing from the flagged-cell table")

    flat = _flat(text)
    if _find(flat, r"Mixing helps in no cell on either panel, and the degree feature hurts in no cell on either "
                   r"panel", "no-mixing-help claim", problems):
        bad = [k for k, c in disk["cells"].items() if c["verdict"] in ("mixing HELPS", "degree HURTS")]
        if bad:
            problems.append(f"no-mixing-help claim: disk has {bad}")
    if _find(flat, r"On Japan, mixing also raises error at RMSE h3 and MAE h10 and lowers correlation at h10, "
                   r"where the degree feature is within noise", "Japan mixing-only cells", problems):
        for mt, h in (("rmse", 3), ("mae", 10), ("pcc", 10)):
            mix = disk["cells"][("gate-off vs learned", "influenza_japan", mt, h)]["verdict"]
            deg = disk["cells"][("pure-TCN vs gate-off", "influenza_japan", mt, h)]["verdict"]
            if mix != "mixing HURTS" or deg != "within noise":
                problems.append(f"Japan mixing-only claim fails at {mt} h{h}: mixing {mix}, degree {deg}")
    return problems


def check_spread(text, disk):
    problems = []
    rows = _rows(text, SPREAD_RE)
    seen = set()
    for m in rows:
        ds, arm = m.group(1), {"learned": "learned", "pure-TCN": "pure"}[m.group(3)]
        seen.add((ds, arm))
        for i, h in enumerate(HORIZONS):
            got = arm_sd(disk, arm, ds, "rmse", h)
            if _off(m.group(4 + i), got):
                problems.append(f"spread {ds} {arm} h{h}: printed {m.group(4 + i)}, disk {got:.1f}")
    want = {(ds, a) for ds in PANELS for a in ("learned", "pure")}
    if seen != want or len(rows) != len(want):
        problems.append(f"spread table: parsed {len(rows)} rows, missing {sorted(want - seen)}")

    flat = _flat(text)
    smaller = lambda ds, mt: [h for h in HORIZONS if arm_sd(disk, "pure", ds, mt, h) < arm_sd(disk, "learned", ds, mt, h)]
    m = _find(flat, r"On COVID the pure-TCN seed spread is smaller than the learned model's at all (\d+) horizons",
              "COVID RMSE spread", problems)
    if m and (int(m.group(1)) != len(HORIZONS) or len(smaller("covid_us-states", "rmse")) != len(HORIZONS)):
        problems.append(f"COVID RMSE spread: doc all {m.group(1)}, disk smaller at {smaller('covid_us-states', 'rmse')}")
    m = _find(flat, r"On Japan it is larger at (\d+) of (\d+) horizons and smaller only at h(\d+)",
              "Japan RMSE spread", problems)
    if m:
        sm = smaller("influenza_japan", "rmse")
        if (int(m.group(1)), int(m.group(2)), [int(m.group(3))]) != (len(HORIZONS) - len(sm), len(HORIZONS), sm):
            problems.append(f"Japan RMSE spread: doc larger at {m.group(1)} of {m.group(2)}, smaller at "
                            f"h{m.group(3)}; disk smaller at {sm}")
    m = _find(flat, r"On COVID MAE it is smaller at (\d+) of (\d+) horizons, h(\d+) being the exception",
              "COVID MAE spread", problems)
    if m:
        sm = smaller("covid_us-states", "mae")
        exc = [h for h in HORIZONS if h not in sm]
        if (int(m.group(1)), int(m.group(2)), [int(m.group(3))]) != (len(sm), len(HORIZONS), exc):
            problems.append(f"COVID MAE spread: doc smaller at {m.group(1)} of {m.group(2)} except "
                            f"h{m.group(3)}; disk smaller at {sm}")
    return problems


def _quote(text, start):
    """The text of the first '> ' blockquote line that starts with `start`."""
    for line in text.splitlines():
        if line.startswith("> " + start):
            return line[2:].strip()
    return None


def check_prose(text, disk):
    problems = []
    flat = _flat(text)
    cells = disk["cells"]

    m = _find(flat, r"\|t\| > sqrt\((\d+)\) = ([\d.]+) at (\d+) degrees of freedom, which happens ([\d.]+) "
                    r"percent of the time, about ([\d.]+) false flags per (\d+) cells", "null flag rate", problems)
    if m:
        n = len(SEEDS)
        p = 2 * stats.t.sf(math.sqrt(n), n - 1)
        k = len(POINT) * len(HORIZONS)
        if (int(m.group(1)), int(m.group(3)), int(m.group(6))) != (n, n - 1, k):
            problems.append(f"null flag rate: doc n={m.group(1)} df={m.group(3)} cells={m.group(6)}, "
                            f"expected {n}, {n - 1}, {k}")
        for name, printed, got in (("threshold", m.group(2), math.sqrt(n)), ("percent", m.group(4), 100 * p),
                                   ("expected flags", m.group(5), k * p)):
            if _off(printed, got):
                problems.append(f"null flag rate: {name} printed {printed}, recomputed {got:.3f}")

    m = _find(flat, r"COVID correlation is negative at ((?:h\d+, )*h\d+ and h\d+) in all three arms",
              "COVID PCC sign", problems)
    if m:
        hs = [int(x) for x in re.findall(r"h(\d+)", m.group(1))]
        pos = [(a, h) for a in ("learned", "gateoff", "pure") for h in hs
               if arm_mean(disk, a, "covid_us-states", "pcc", h) >= 0]
        if pos:
            problems.append(f"COVID PCC sign: not negative at {pos}")

    mix = {h: cells[("gate-off vs learned", "influenza_japan", "rmse", h)] for h in (3, 5)}
    for rx, what in ((r"The ([\d.]+) and ([\d.]+) are gate-off numbers", "gate-off numbers"),
                     (r"removing the graph improves RMSE by ([\d.]+) at \*h\* = 3 and ([\d.]+) at \*h\* = 5",
                      "manuscript quote numbers"),
                     (r"removing neighbour mixing improves RMSE by ([\d.]+) at \*h\* = 3 and ([\d.]+) at \*h\* = 5",
                      "replacement numbers")):
        m = _find(flat, rx, what, problems)
        if m:
            for printed, h in ((m.group(1), 3), (m.group(2), 5)):
                if _off(printed, -mix[h]["d"]):
                    problems.append(f"{what}: h{h} printed {printed}, disk gate-off improvement {-mix[h]['d']:.3f}")

    m = _find(flat, r"Japan RMSE moves by (-?[\d.]+) \(sd ([\d.]+)\) at h3 and (-?[\d.]+) \(sd ([\d.]+)\) at h5, "
                    r"and COVID h5 RMSE by (-?[\d.]+) \(sd ([\d.]+)\), all within noise", "whole-graph numbers", problems)
    if m:
        want = (("influenza_japan", 3), ("influenza_japan", 5), ("covid_us-states", 5))
        for i, (ds, h) in enumerate(want):
            c = cells[("pure-TCN vs learned", ds, "rmse", h)]
            if _off(m.group(1 + 2 * i), c["d"]) or _off(m.group(2 + 2 * i), c["sd"]):
                problems.append(f"whole-graph numbers {ds} h{h}: printed {m.group(1 + 2 * i)} "
                                f"(sd {m.group(2 + 2 * i)}), disk {c['d']:+.3f} (sd {c['sd']:.3f})")
            if c["verdict"] != "within noise":
                problems.append(f"whole-graph numbers {ds} h{h}: 'within noise' is false on disk")

    m = _find(flat, r"slightly larger than the gate-off one \(([\d.]+) against ([\d.]+)\)", "h3 comparison", problems)
    if m:
        pure3 = -cells[("pure-TCN vs learned", "influenza_japan", "rmse", 3)]["d"]
        if _off(m.group(1), pure3) or _off(m.group(2), -mix[3]["d"]) or not pure3 > -mix[3]["d"]:
            problems.append(f"h3 comparison: printed {m.group(1)} against {m.group(2)}, disk {pure3:.3f} "
                            f"against {-mix[3]['d']:.3f}")

    # word arithmetic of the quoted sentences, counted the way wc -w counts
    old = _quote(text, "Those are concentrated on influenza-Japan, where removing the graph")
    new = _quote(text, "Those are concentrated on influenza-Japan, where removing neighbour mixing")
    head = _quote(text, "In an exploratory follow-up")
    m = _find(flat, r"the explanation is (\d+) words, the replacement (\d+), so the edit adds (\d+) words net\. "
                    r"Using the full headline in its place would add (\d+)", "word arithmetic", problems)
    if m and None in (old, new, head):
        problems.append("word arithmetic: a quoted sentence is missing")
    elif m:
        expl = old[old.index("On those panels"):]
        repl = new[new.index("An exploratory check"):]
        got = (len(expl.split()), len(repl.split()), len(new.split()) - len(old.split()),
               len(head.split()) - len(expl.split()))
        if tuple(int(g) for g in m.groups()) != got:
            problems.append(f"word arithmetic: doc {tuple(int(g) for g in m.groups())}, counted {got}")
    m = _find(flat, r"The manuscript is ([\d,]+) words by wc -w on \d{4}-\d{2}-\d{2}, ([\d,]+) over the ([\d,]+) "
                    r"internal target", "word target arithmetic", problems)
    if m and int(_num(m.group(1))) - int(_num(m.group(3))) != int(_num(m.group(2))):
        problems.append(f"word target arithmetic: {m.group(1)} - {m.group(3)} != {m.group(2)}")

    # runtimes for the untested panels, from the tracked 2026-08-17 gate-off log
    log = {}
    for ds, mins in re.findall(r"^\s+(\S+) seed\d+ done in ([\d.]+) min", GATE_LOG.read_text(), re.M):
        log.setdefault(ds, []).append(float(mins))
    m = _find(flat, r"US-regions and US-states at ([\d.]+) to ([\d.]+) min per seed and dengue at ([\d.]+) to "
                    r"([\d.]+) min per seed, about ([\d.]+) h for five", "untested-panel runtimes", problems)
    if m:
        us = log["influenza_us-regions"] + log["influenza_us-states"]
        got = (min(us), max(us), min(log["dengue"]), max(log["dengue"]), sum(log["dengue"]) / 60)
        for printed, g in zip(m.groups(), got):
            if _off(printed, g):
                problems.append(f"untested-panel runtimes: printed {printed}, log gives {g:.1f}")
    return problems


def check_summaries(text, disk):
    """The committed summary JSONs must agree with the recompute. Relative tolerance 1e-9."""
    problems = []
    close = lambda a, b: a == b or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    S = json.loads((ROOT / "experiments" / "pure_tcn__summary.json").read_text())
    ref_to_cmp = {"learned": "pure-TCN vs learned", "gate-off": "pure-TCN vs gate-off"}
    for ds in PANELS:
        blk = S.get(ds)
        if not blk:
            problems.append(f"pure_tcn__summary.json has no block for {ds}")
            continue
        for c in blk["cells"]:
            mine = disk["cells"][(ref_to_cmp[c["reference"]], ds, c["metric"], c["horizon"])]
            pairs = (("ref_mean", "ref_mean"), ("ref_sd", "ref_sd"), ("pure_mean", "arm_mean"),
                     ("pure_sd", "arm_sd"), ("d_mean", "d"), ("d_sd", "sd"))
            bad = [k for k, j in pairs if not close(c[k], mine[j])]
            if bad or c["n"] != mine["n"] or c["verdict"] != mine["verdict"]:
                problems.append(f"pure_tcn summary {ds} {c['reference']} {c['metric']} h{c['horizon']}: "
                                f"differs from the seed files on {bad or 'n/verdict'}")
        for ref, tally in blk["tallies"].items():
            name = ref_to_cmp[ref]
            vs = [disk["cells"][(name, ds, mt, h)]["verdict"] for mt in POINT for h in HORIZONS]
            if {k: vs.count(k) for k in tally} != tally:
                problems.append(f"pure_tcn summary {ds} tally vs {ref}: {tally} differs from the seed files")
    G = json.loads((ROOT / "experiments" / "gateoff_fresh__summary.json").read_text())
    label = {"learned": "learned", "archived gate-off": "gateoff", "fresh gate-off": "fresh", "pure-TCN": "pure"}
    for ds in PANELS:
        blk = G.get(ds)
        if not blk:
            problems.append(f"gateoff_fresh__summary.json has no block for {ds}")
            continue
        v = disk["values"]
        for c in blk["reproduction"]["cells"]:
            k = (c["metric"], c["horizon"])
            ident = sum(v["fresh"][ds][k][s] == v["gateoff"][ds][k][s] for s in SEEDS)
            if c["identical"] != ident or (c["max_rel"] != 0 and ident == len(SEEDS)):
                problems.append(f"gateoff_fresh summary {ds} {k}: identical {c['identical']}, seed files {ident}")
        for c in blk["degree"]["cells"]:
            mine = disk["vs_fresh"][(ds, c["metric"], c["horizon"])]
            if not (close(c["d_mean"], mine["d"]) and close(c["d_sd"], mine["sd"])) or c["verdict"] != mine["verdict"]:
                problems.append(f"gateoff_fresh summary {ds} degree {c['metric']} h{c['horizon']}: differs "
                                f"from the seed files")
        for mt, row in blk["h5"].items():
            for lab, arm in label.items():
                if not close(row[lab], arm_mean(disk, arm, ds, mt, 5)):
                    problems.append(f"gateoff_fresh summary {ds} h5 {mt} {lab}: {row[lab]:.3f} differs from "
                                    f"the seed files {arm_mean(disk, arm, ds, mt, 5):.3f}")
    return problems


CHECKS = (("completeness", check_completeness),
          ("reproduction (fresh vs archived gate-off)", check_reproduction),
          ("tally table", check_tally),
          ("pure-TCN vs learned table and its prose", check_main_table),
          ("h5 split table and its prose", check_split),
          ("per-seed h5 table and its prose", check_per_seed),
          ("flagged-cell table, coverage and its prose", check_flagged),
          ("seed-spread table and its prose", check_spread),
          ("other prose (null rate, signs, manuscript numbers, word arithmetic, runtimes)", check_prose),
          ("summary JSONs agree with the seed files", check_summaries))


def run(text, disk, verbose=True):
    allp = []
    for name, fn in CHECKS:
        try:
            problems = fn(text, disk)
        except (KeyError, ValueError, ZeroDivisionError) as e:   # a broken input must fail, not crash
            problems = [f"{name} raised {type(e).__name__}: {e}"]
        allp += problems
        if verbose:
            print(f"  {'FAIL' if problems else 'ok  '}  {name}"
                  + (f"  ({len(problems)} problem(s))" if problems else ""))
            for p in problems:
                print(f"        - {p}")
    return allp


def info(text):
    """Printed, never failed: both move whenever the manuscript is edited."""
    if MANUSCRIPT.exists():
        ms = MANUSCRIPT.read_text(encoding="utf-8")
        old = _quote(text, "Those are concentrated on influenza-Japan, where removing the graph")
        print(f"  info  manuscript is {len(ms.split())} words now (wc -w); the quoted :326 text is "
              f"{'still present' if old and old in ms else 'NOT present (edited since?)'}")


# --------------------------------------------------------------------------- #
def _drop_line(text, start):
    for line in text.splitlines(keepends=True):
        if line.startswith(start):
            return text.replace(line, "", 1)
    return text


def mutate(text, disk):
    """Corrupt the document, then the disk, one defect at a time; every corruption MUST be caught."""
    r = lambda a, b: text.replace(a, b, 1)
    muts = [
        ("tally inflated", r("| the whole graph | covid_us-states | 1 | 0 | 11 |",
                             "| the whole graph | covid_us-states | 2 | 0 | 10 |")),
        ("tally label swapped", r("| pure-TCN vs gate-off | the degree feature | covid_us-states |",
                                  "| pure-TCN vs gate-off | neighbour mixing | covid_us-states |")),
        ("main table stale d", r("| -100.595 | 1077.100 |", "| -190.595 | 1077.100 |")),
        ("main table verdict flipped", r("| -0.041 | 0.160 | within noise |", "| -0.041 | 0.160 | GRAPH HELPS |")),
        ("main table stale mean", r("| 734.888 | 682.245 |", "| 734.888 | 662.245 |")),
        ("main table row deleted", _drop_line(text, "| influenza_japan | RMSE | 5 | 841.106")),
        ("bold error-cell count wrong", r("**0 of 16 error cells differ", "**1 of 16 error cells differ")),
        ("numerically-lower count wrong", r("in 11 of 16 error cells (COVID 4 of 8, Japan 7 of 8)",
                                            "in 12 of 16 error cells (COVID 5 of 8, Japan 7 of 8)")),
        ("the one flag stale", r("d -0.038 with sd 0.025", "d -0.058 with sd 0.025")),
        ("h5 split difference stale", r("| -962.9 | +862.3 | -100.6 |", "| -962.9 | +962.3 | -100.6 |")),
        ("h5 split mean stale", r("| 841.1 | 779.4 | 838.2 |", "| 841.1 | 799.4 | 838.2 |")),
        ("per-seed sign flipped", r("| +25.0 | +103.4 |", "| -25.0 | +103.4 |")),
        ("per-seed prose count wrong", r("All 20 per-seed differences are positive",
                                         "All 19 per-seed differences are positive")),
        ("flagged row deleted", _drop_line(text, "| influenza_japan | gate-off vs learned | MAE | 10 |")),
        ("within-noise cell listed as flagged",
         r("| influenza_japan | gate-off vs learned | RMSE | 5 |",
           "| influenza_japan | gate-off vs learned | MAE | 3 | -17.564 | 22.239 | 1 of 5 | mixing HURTS |\n"
           "| influenza_japan | gate-off vs learned | RMSE | 5 |")),
        ("flagged verdict flipped", r("| 5 of 5 | mixing HURTS |", "| 5 of 5 | mixing HELPS |")),
        ("flagged sd stale", r("| +862.322 | 349.181 |", "| +862.322 | 449.181 |")),
        ("flagged seed count wrong", r("| -962.918 | 790.682 | 0 of 5 |", "| -962.918 | 790.682 | 2 of 5 |")),
        ("seed spread stale", r("| 214.0 | 701.1 |", "| 314.0 | 701.1 |")),
        ("Japan spread count wrong", r("larger at 3 of 4 horizons and smaller only at h15",
                                       "larger at 2 of 4 horizons and smaller only at h15")),
        ("COVID MAE spread exception wrong", r("h5 being the exception", "h10 being the exception")),
        ("null rate wrong", r("which happens 8.9 percent of the time", "which happens 5.0 percent of the time")),
        ("reproduction count wrong", r("| covid_us-states | 60 of 60 | 140 of 140 |",
                                       "| covid_us-states | 59 of 60 | 140 of 140 |")),
        ("reproduction max rel wrong", r("| 5 of 5 | 0 |", "| 5 of 5 | 0.01 |")),
        ("new-file count wrong", r("All 20 new seed files are present", "All 18 new seed files are present")),
        ("whole-graph h5 number stale", r("-2.9 (sd 53.9) at h5", "-29.0 (sd 53.9) at h5")),
        ("gate-off numbers stale", r("The 45.3 and 61.7 are gate-off numbers", "The 45.3 and 67.1 are gate-off numbers")),
        ("h3 comparison stale", r("(52.6 against 45.3)", "(25.6 against 45.3)")),
        ("word arithmetic wrong", r("so the edit adds 8 words net", "so the edit adds 3 words net")),
        ("headline horizon wrong", r("Only at the five-week horizon", "Only at the ten-week horizon")),
        ("headline count wrong", r("moved none of 16 forecast-error comparisons", "moved none of 12 forecast-error comparisons")),
        ("COVID PCC sign claim widened", r("COVID correlation is negative at h5, h10 and h15",
                                           "COVID correlation is negative at h3, h5, h10 and h15")),
        ("dengue runtime stale", r("about 8.5 h for five", "about 6.0 h for five")),
    ]
    print(f"{'=' * 78}\nMutation test: {len(muts)} document corruptions and 3 disk corruptions, each must "
          f"be caught\n{'=' * 78}")
    escaped = []
    for name, bad in muts:
        if bad == text:
            print(f"  NO-OP       {name}  (mutation did not change the text)")
            escaped.append(name)
            continue
        problems = run(bad, disk, verbose=False)
        print(f"  {'caught' if problems else 'NOT CAUGHT':<11} {name}"
              + (f"  ({len(problems)} problem(s))" if problems else ""))
        if not problems:
            escaped.append(name)

    # the check must read the disk, not only the document: break an input and the clean doc must fail
    def broken(arm, ds, key, s, fn):
        d = copy.deepcopy({"values": disk["values"], "raw": disk["raw"]})
        d["values"][arm][ds][key][s] = fn(d["values"][arm][ds][key][s])
        return derive(d)
    for name, bad in (("disk: one pure-TCN seed value nudged",
                       broken("pure", "covid_us-states", ("rmse", 5), 42, lambda x: x + 500.0)),
                      ("disk: one archived gate-off value changed in the 5th digit",
                       broken("gateoff", "influenza_japan", ("mae", 10), 62, lambda x: x * 1.0001)),
                      ("disk: one learned PCC value moved",
                       broken("learned", "influenza_japan", ("pcc", 10), 52, lambda x: x - 0.02))):
        problems = run(text, bad, verbose=False)
        print(f"  {'caught' if problems else 'NOT CAUGHT':<11} {name}"
              + (f"  ({len(problems)} problem(s))" if problems else ""))
        if not problems:
            escaped.append(name)
    total = len(muts) + 3
    print(f"\n{len(escaped)} of {total} mutation(s) escaped. The verifier does not check what it claims to."
          if escaped else f"\nAll {total} mutations caught.")
    return escaped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc", nargs="?", default=str(DOC))
    ap.add_argument("--mutate", action="store_true", help="mutation-test the verifier itself")
    a = ap.parse_args()

    p = Path(a.doc)
    if not p.exists():
        sys.exit(f"no such document: {p}")
    text = p.read_text(encoding="utf-8")
    disk = derive(load_disk())

    print(f"{'=' * 78}\nVerifying {p.name} against the seed records on disk\n{'=' * 78}")
    problems = run(text, disk)
    info(text)

    if a.mutate:
        print()
        if mutate(text, disk):
            sys.exit(2)

    if problems:
        print(f"\n{len(problems)} PROBLEM(S). The document does not match its artifacts.")
        sys.exit(1)
    print(f"\nOK: {p.name} matches the seed records on disk.")


if __name__ == "__main__":
    main()

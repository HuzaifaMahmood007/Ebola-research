"""verify_shufadj_doc.py -- read the numbers back OUT of the shuffled-adjacency outcome doc and
recompute them from disk.

The point is to catch a document that has drifted from its artifacts: a stale delta left after a
re-run, a flipped verdict, a tally that no longer sums, a significant cell dropped from the table, or a
within-noise cell smuggled in as significant.

This deliberately does NOT import ablation.run_shuffle_adjacency or run_epi_ablation's loaders. Every
value is re-derived straight from the JSON records with a local paired-delta. A bug shared with the
generator would otherwise verify itself, which is the failure mode this file exists to prevent.

Checks:
  1. Completeness: 25 shufadj records present, each carrying the shuffle meta (seed, hash, frac>0).
  2. The full 60-cell tally recomputed from disk equals the tally printed in the doc.
  3. Every significant row in the doc table matches a fresh paired-by-seed recompute: delta, sd, and
     the HELPS/HURTS verdict, to the precision printed.
  4. Coverage: every cell that is significant on disk appears in the table, and no within-noise cell is
     listed there.

    python diagnostics/verify_shufadj_doc.py
    python diagnostics/verify_shufadj_doc.py --mutate      # the check on the check
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "results" / "single"
SHUF = ROOT / "ablation" / "single"
PANELS = ("influenza_japan", "influenza_us-regions", "influenza_us-states",
          "covid_us-states", "dengue")
SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
POINT = ("rmse", "mae", "pcc")
LOWER = ("rmse", "mae")
SHUF_BASE = 20260921
META = ("shuffle_adj_seed", "shuffle_adj_hash", "shuffle_adj_frac_displaced")


def field(ds):
    return "country_macro" if ds == "dengue" else "node_mean"


# --------------------------------------------------------------------------- #
# Disk side: independent loader + paired delta
# --------------------------------------------------------------------------- #
def _seed_from(name):
    for part in name.replace(".json", "").split("__"):
        if part.startswith("seed"):
            try:
                return int(part[4:])
            except ValueError:
                return None
    return None


def _load(pattern):
    """{(horizon, metric): {seed: value}} for the 'encoder' model only (skip encoder_mc/smoke)."""
    out = {}
    for f in glob.glob(str(pattern)):
        if "smoke" in Path(f).name:
            continue
        s = _seed_from(Path(f).name)
        if s is None:
            continue
        for r in json.load(open(f)):
            if r["model"] != "encoder":
                continue
            out.setdefault((r["horizon"], r["metric"]), {})[s] = r[field(r["dataset"])]
    return out


def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def _paired(base, abl):
    seeds = sorted(set(base) & set(abl))
    d = [abl[s] - base[s] for s in seeds]
    if not d:
        return None, None, 0
    m, sd = _mean_sd(d)
    return m, sd, len(d)


def _verdict(dm, dsd, n, m):
    if n < 2 or dsd == 0 or abs(dm) < dsd:
        return "within noise"
    shuf_worse = (dm > 0) if m in LOWER else (dm < 0)
    return "helps" if shuf_worse else "hurts"


def disk_cells():
    """{(panel, metric, h): (delta, sd, n, verdict)} over all 60 cells."""
    cells = {}
    for ds in PANELS:
        base = _load(REAL / f"encoder__{ds}__seed*.json")
        abl = _load(SHUF / f"encoder__{ds}__seed*__shufadj.json")
        for m in POINT:
            for h in HORIZONS:
                dm, dsd, n = _paired(base.get((h, m), {}), abl.get((h, m), {}))
                if dm is None:
                    continue
                cells[(ds, m, h)] = (dm, dsd, n, _verdict(dm, dsd, n, m))
    return cells


def check_completeness():
    problems = []
    present = 0
    for ds in PANELS:
        for s in SEEDS:
            p = SHUF / f"encoder__{ds}__seed{s}__shufadj.json"
            if not p.exists():
                problems.append(f"missing record: {p.name}")
                continue
            present += 1
            for r in json.loads(p.read_text()):
                if not all(k in r for k in META):
                    problems.append(f"{p.name}: record missing shuffle meta")
                    break
                if r.get("shuffle_adj_seed") != SHUF_BASE + s:
                    problems.append(f"{p.name}: shuffle_adj_seed {r.get('shuffle_adj_seed')} "
                                    f"!= {SHUF_BASE + s}")
                    break
                f = r.get("shuffle_adj_frac_displaced")
                if not (f and f > 0):
                    problems.append(f"{p.name}: displaced fraction {f} is not > 0")
                    break
    if present != 25 and not problems:
        problems.append(f"expected 25 records, found {present}")
    return problems


# --------------------------------------------------------------------------- #
# Document side
# --------------------------------------------------------------------------- #
TALLY_RE = re.compile(r"Verdict over 60 cells:\s*(\d+)\s*the real graph helps,\s*(\d+)\s*the real "
                      r"graph hurts,\s*(\d+)\s*within noise", re.I)
# | panel | metric | h | real | shuf | delta | sd | verdict |
ROW_RE = re.compile(r"^\|\s*([a-z_\-]+)\s*\|\s*(rmse|mae|pcc)\s*\|\s*(\d+)\s*\|"
                    r"\s*[-\d.]+\s*\|\s*[-\d.]+\s*\|\s*([-\d.]+)\s*\|\s*([\d.]+)\s*\|"
                    r"\s*real graph (HELPS|HURTS)\s*\|", re.I)


def _tol(printed):
    """Half a unit in the last printed digit."""
    s = printed.strip().lstrip("+-")
    return 0.5 * (10 ** -len(s.split(".")[1]) if "." in s else 1.0)


def parse_doc(text):
    tally = None
    m = TALLY_RE.search(text)
    if m:
        tally = tuple(int(g) for g in m.groups())
    rows = []
    for line in text.splitlines():
        mm = ROW_RE.match(line.strip())
        if mm:
            ds, met, h, delta, sd, verd = mm.groups()
            rows.append((ds, met.lower(), int(h), delta, sd, verd.lower()))
    return tally, rows


def check_tally(text, cells):
    problems = []
    tally, _ = parse_doc(text)
    if tally is None:
        return ["tally line 'Verdict over 60 cells: ...' not found"]
    helps = sum(1 for v in cells.values() if v[3] == "helps")
    hurts = sum(1 for v in cells.values() if v[3] == "hurts")
    noise = sum(1 for v in cells.values() if v[3] == "within noise")
    if tally != (helps, hurts, noise):
        problems.append(f"doc tally {tally} != disk {(helps, hurts, noise)} over {len(cells)} cells")
    return problems


def check_rows(text, cells):
    problems = []
    _, rows = parse_doc(text)
    if not rows:
        return ["no significant-cell rows parsed from the table"]
    for ds, met, h, delta_s, sd_s, verd in rows:
        key = (ds, met, h)
        if key not in cells:
            problems.append(f"{key}: row in doc has no cell on disk")
            continue
        dm, dsd, n, v = cells[key]
        if v == "within noise":
            problems.append(f"{key}: doc lists it significant, disk says within noise")
            continue
        if v != verd:
            problems.append(f"{key}: doc verdict {verd}, disk {v}")
        if abs(dm - float(delta_s)) > _tol(delta_s):
            problems.append(f"{key}: doc delta {delta_s}, disk {dm:.4f}")
        if abs(dsd - float(sd_s)) > _tol(sd_s):
            problems.append(f"{key}: doc sd {sd_s}, disk {dsd:.4f}")
    return problems


def check_coverage(text, cells):
    problems = []
    _, rows = parse_doc(text)
    in_doc = {(ds, met, h) for ds, met, h, *_ in rows}
    on_disk = {k for k, v in cells.items() if v[3] != "within noise"}
    for miss in sorted(on_disk - in_doc):
        problems.append(f"{miss} is significant on disk but missing from the table")
    for extra in sorted(in_doc - on_disk):
        problems.append(f"{extra} is in the table but not significant on disk")
    return problems


CHECKS = (("completeness", lambda t, c: check_completeness()),
          ("tally", check_tally),
          ("significant rows", check_rows),
          ("coverage", check_coverage))


def run(text, cells, verbose=True):
    allp = []
    for name, fn in CHECKS:
        problems = fn(text, cells)
        allp += problems
        if verbose:
            print(f"  {'FAIL' if problems else 'ok  '}  {name}"
                  + (f"  ({len(problems)} problem(s))" if problems else ""))
            for p in problems:
                print(f"        - {p}")
    return allp


# --------------------------------------------------------------------------- #
def mutate(text, cells):
    """Corrupt the document one defect at a time; every corruption MUST be caught."""
    muts = []
    m = TALLY_RE.search(text)
    if m:
        muts.append(("tally inflated",
                     text.replace(m.group(0), "Verdict over 60 cells: 12 the real graph helps, "
                                  "2 the real graph hurts, 46 within noise", 1)))
    # a stale delta in a significant row
    _, rows = parse_doc(text)
    if rows:
        ds, met, h, delta_s, sd_s, verd = rows[0]
        line_frag = f"| {delta_s} | {sd_s} | real graph {verd.upper()} |"
        bad_frag = f"| -0.0001 | {sd_s} | real graph {verd.upper()} |"
        if line_frag in text:
            muts.append(("stale delta in a significant row", text.replace(line_frag, bad_frag, 1)))
    # a flipped verdict on a real row
    for ds, met, h, delta_s, sd_s, verd in rows:
        flip = "HURTS" if verd == "helps" else "HELPS"
        frag = f"| {delta_s} | {sd_s} | real graph {verd.upper()} |"
        if frag in text:
            muts.append(("verdict flipped",
                         text.replace(frag, f"| {delta_s} | {sd_s} | real graph {flip} |", 1)))
            break
    # a dropped significant row
    for line in text.splitlines():
        if ROW_RE.match(line.strip()):
            muts.append(("a significant row deleted", text.replace(line + "\n", "", 1)))
            break

    print(f"{'=' * 78}\nMutation test: {len(muts)} corruptions, each must be caught\n{'=' * 78}")
    escaped = []
    for name, bad in muts:
        problems = run(bad, cells, verbose=False)
        status = "caught" if problems else "NOT CAUGHT"
        print(f"  {status:<11} {name}" + (f"  ({len(problems)} problem(s))" if problems else ""))
        if not problems:
            escaped.append(name)
    if escaped:
        print(f"\n{len(escaped)} mutation(s) escaped. The verifier does not check what it claims to.")
    else:
        print(f"\nAll {len(muts)} mutations caught.")
    return escaped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc", nargs="?",
                    default=str(ROOT / "progress" / "outcomes" / "Shuffled_Adjacency_2026-09-23.md"))
    ap.add_argument("--mutate", action="store_true", help="mutation-test the verifier itself")
    a = ap.parse_args()

    p = Path(a.doc)
    if not p.exists():
        sys.exit(f"no such document: {p}")
    text = p.read_text(encoding="utf-8")
    cells = disk_cells()

    print(f"{'=' * 78}\nVerifying {p.name} against the artifacts on disk\n{'=' * 78}")
    problems = run(text, cells)

    if a.mutate:
        print()
        escaped = mutate(text, cells)
        if escaped:
            sys.exit(2)

    if problems:
        print(f"\n{len(problems)} PROBLEM(S). The document does not match its artifacts.")
        sys.exit(1)
    print(f"\nOK: {p.name} matches the artifacts on disk.")


if __name__ == "__main__":
    main()

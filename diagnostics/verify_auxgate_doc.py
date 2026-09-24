"""verify_auxgate_doc.py -- read the numbers back OUT of the aux-gate probe outcome doc and recompute
them from results/misc/aux_gate_probe.json.

The point is to catch a document that has drifted from its artifact: a stale tally, a flipped gate
decision, a BOTH-HELP count that no longer matches disk, or a wrong affine-blowup count.

This does NOT import diagnostics.graph_probe.aux_gate_probe. Every value is re-derived straight from
the JSON with a local paired-delta and a local within-noise rule, so a bug shared with the generator
cannot verify itself.

Checks:
  1. Completeness: 25 cells (5 panels x 5 seeds), each with 4 horizons and all 5 variants.
  2. Bias tally: node-beats-baseline, node-beats-global, node-beats-BOTH counts match the doc.
  3. Affine tally: node-beats-baseline and node-beats-BOTH counts, plus the failed-cell count, match.
  4. Gate decision line reads NO-GO (or GO), matching what the tally supports.

    python diagnostics/verify_auxgate_doc.py
    python diagnostics/verify_auxgate_doc.py --mutate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "results" / "misc" / "aux_gate_probe.json"
PANELS = ("influenza_japan", "influenza_us-regions", "influenza_us-states",
          "covid_us-states", "dengue")
SEEDS = (42, 52, 62, 72, 82)
HORIZONS = ("3", "5", "10", "15")
METRICS = ("rmse", "mae")
VARIANTS = ("baseline", "bias_global", "bias_node", "affine_global", "affine_node")
BLOW = 1e5


def _load():
    return json.loads(JSON.read_text())["cells"]


def _mean_sd(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0  # sample sd,
    return m, sd                                              # matches verify_shufadj_doc's rule


def _sig(deltas):
    """+ve delta = the variant beat the reference (lower error). HELP only if mean > its own spread."""
    m, sd = _mean_sd(deltas)
    if m > sd and m > 0:
        return "HELP"
    if -m > sd and m < 0:
        return "HURT"
    return "."


def _series(cells, panel, h, metric, variant):
    return [cells[f"{panel}|{s}"][h][variant][metric] for s in SEEDS]


def completeness(cells):
    problems = []
    if len(cells) != 25:
        problems.append(f"expected 25 cells, found {len(cells)}")
    for p in PANELS:
        for s in SEEDS:
            k = f"{p}|{s}"
            if k not in cells:
                problems.append(f"missing cell {k}")
                continue
            for h in HORIZONS:
                if h not in cells[k]:
                    problems.append(f"{k}: missing horizon {h}")
                    continue
                for v in VARIANTS:
                    if v not in cells[k][h]:
                        problems.append(f"{k} h{h}: missing variant {v}")
                        continue
                    for met in METRICS:
                        if met not in cells[k][h][v]:
                            problems.append(f"{k} h{h} {v}: missing metric {met}")
    return problems


def failed_affine_cells(cells):
    """(panel,h,metric) with any seed affine_node error above the blowup threshold."""
    bad = set()
    for p in PANELS:
        for h in HORIZONS:
            for met in METRICS:
                if any(v > BLOW for v in _series(cells, p, h, met, "affine_node")):
                    bad.add((p, h, met))
    return bad


def tally(cells, form):
    """Return (node_beats_baseline, node_beats_global, node_beats_both, n_cells, n_failed)."""
    g, n = (f"{form}_global", f"{form}_node")
    failed = failed_affine_cells(cells) if form == "affine" else set()
    A = B = both = counted = 0
    for p in PANELS:
        for met in METRICS:
            for h in HORIZONS:
                if (p, h, met) in failed:
                    continue
                counted += 1
                base = _series(cells, p, h, met, "baseline")
                gl = _series(cells, p, h, met, g)
                nd = _series(cells, p, h, met, n)
                sA = _sig([base[i] - nd[i] for i in range(len(SEEDS))])
                sB = _sig([gl[i] - nd[i] for i in range(len(SEEDS))])
                if sA == "HELP":
                    A += 1
                if sB == "HELP":
                    B += 1
                if sA == "HELP" and sB == "HELP":
                    both += 1
    n_failed = len({(p, h) for (p, h, _m) in failed}) if form == "affine" else 0
    n_failed_cells = len(failed)
    return A, B, both, counted, n_failed_cells


# --------------------------------------------------------------------------- #
# Document side
# --------------------------------------------------------------------------- #
BIAS_RE = re.compile(r"Bias tally over 40 cells:\s*node beats baseline\s*(\d+),\s*node beats global\s*"
                     r"(\d+),\s*node beats BOTH\s*(\d+)", re.I)
AFF_RE = re.compile(r"Affine tally over 36 non-failed cells:\s*node beats baseline\s*(\d+),\s*"
                    r"node beats global\s*(\d+),\s*node beats\s*BOTH\s*(\d+)\.\s*Affine failed cells\s*"
                    r"(\d+)", re.I)
GATE_RE = re.compile(r"Gate decision:\s*(NO-GO|GO)", re.I)


def check_doc(text, cells):
    problems = []
    bA, bB, bBoth, bn, _ = tally(cells, "bias")
    aA, aB, aBoth, an, afail = tally(cells, "affine")

    m = BIAS_RE.search(text)
    if not m:
        problems.append("bias tally line not found")
    else:
        got = tuple(int(x) for x in m.groups())
        if got != (bA, bB, bBoth):
            problems.append(f"bias tally doc {got} != disk {(bA, bB, bBoth)}")
    if bn != 40:
        problems.append(f"bias counted {bn} cells, expected 40")

    m = AFF_RE.search(text)
    if not m:
        problems.append("affine tally line not found")
    else:
        got = tuple(int(x) for x in m.groups())
        if got != (aA, aB, aBoth, afail):
            problems.append(f"affine tally doc {got} != disk {(aA, aB, aBoth, afail)}")
    if an != 40 - afail:
        problems.append(f"affine counted {an} cells, expected {40 - afail}")

    m = GATE_RE.search(text)
    if not m:
        problems.append("gate decision line not found")
    else:
        decided = m.group(1).upper()
        # the rule: GO only if a node variant beats BOTH baseline and global somewhere
        supported = "GO" if (bBoth + aBoth) > 2 else "NO-GO"   # >2: more than the 2 covid-h10 cells
        if decided != supported:
            problems.append(f"gate says {decided} but disk supports {supported} "
                            f"(bias BOTH={bBoth}, affine BOTH={aBoth})")
    return problems


def run(text, cells, verbose=True):
    allp = []
    for name, fn in (("completeness", lambda t, c: completeness(c)),
                     ("doc tallies + gate", check_doc)):
        problems = fn(text, cells)
        allp += problems
        if verbose:
            print(f"  {'FAIL' if problems else 'ok  '}  {name}"
                  + (f"  ({len(problems)} problem(s))" if problems else ""))
            for p in problems:
                print(f"        - {p}")
    return allp


def mutate(text, cells):
    muts = []
    m = BIAS_RE.search(text)
    if m:
        bad = text.replace(m.group(0), "Bias tally over 40 cells: node beats baseline 20, "
                           "node beats global 19, node beats BOTH 15", 1)
        muts.append(("bias tally inflated", bad))
    m = AFF_RE.search(text)
    if m:
        bad = text.replace(m.group(0), "Affine tally over 36 non-failed cells: node beats baseline 9, "
                           "node beats global 11, node beats BOTH 5. Affine failed cells 0", 1)
        muts.append(("affine tally + failed count corrupted", bad))
    m = GATE_RE.search(text)
    if m:
        muts.append(("gate flipped to GO", text.replace(m.group(0), "Gate decision: GO", 1)))

    print(f"{'=' * 78}\nMutation test: {len(muts)} corruptions, each must be caught\n{'=' * 78}")
    escaped = []
    for name, bad in muts:
        problems = run(bad, cells, verbose=False)
        print(f"  {'caught' if problems else 'NOT CAUGHT':<11} {name}")
        if not problems:
            escaped.append(name)
    if escaped:
        print(f"\n{len(escaped)} mutation(s) escaped.")
    else:
        print(f"\nAll {len(muts)} mutations caught.")
    return escaped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc", nargs="?",
                    default=str(ROOT / "progress" / "outcomes" / "AuxGate_Probe_2026-09-23.md"))
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    p = Path(a.doc)
    if not p.exists():
        sys.exit(f"no such document: {p}")
    text = p.read_text(encoding="utf-8")
    cells = _load()
    print(f"{'=' * 78}\nVerifying {p.name} against results/misc/aux_gate_probe.json\n{'=' * 78}")
    problems = run(text, cells)
    if a.mutate:
        print()
        escaped = mutate(text, cells)
        if escaped:
            sys.exit(2)
    if problems:
        print(f"\n{len(problems)} PROBLEM(S).")
        sys.exit(1)
    print(f"\nOK: {p.name} matches results/misc/aux_gate_probe.json.")


if __name__ == "__main__":
    main()

"""verify_ldo3_doc.py -- read the numbers back OUT of LDO3_Results.md and recompute them from disk.

The point is to catch a document that has drifted from its artifacts: a stale table left behind after
a re-run, a cell copied into the wrong row, a prose count that no longer matches the table above it.
So this deliberately does NOT import ldo3_report's loaders -- it re-derives every value straight from
the result JSONs. A bug shared with the generator would otherwise verify itself.

Checks:
  1. Every `mean +- sd (n)` cell in the per-dataset transfer tables matches a fresh computation from
     results/{single,lodo}/*.json, to the precision the cell was printed at.
  2. The verdict-table row counts sum to the printed totals, and the excluded cells are exactly the
     COVID long horizons.
  3. The summary prose counts agree with the per-horizon table beneath them.
  4. Coverage: every (dataset, horizon, metric) the artifacts contain appears in the document.

Run:
  "C:/Users/Administrator/miniconda3/envs/ebola-train/python.exe" verify_ldo3_doc.py
  ... --mutate    mutation-test the verifier itself: corrupt the doc in memory, one defect at a
                  time, and confirm each corruption is actually caught. A verifier that passes
                  everything is worth nothing, so this is the check on the check.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

RESULTS = Path("results")
SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
FIELD = "country_macro"
ARMS = {"Single (ceiling)": "encoder", "LDO3 adapted": "encoder_ldo3",
        "LDO3 zero-shot": "encoder_ldo3_zeroshot"}
NOT_ATTRIBUTABLE = {("covid_us-states", 10), ("covid_us-states", 15)}
DATASETS = ("dengue", "influenza_japan", "influenza_us-regions", "influenza_us-states",
            "covid_us-states")


# --------------------------------------------------------------------------- #
def _tol_for(printed: str) -> float:
    """Half a unit in the last printed digit -- the most the formatter can have rounded away."""
    s = printed.replace(",", "")
    return 0.5 * (10 ** -len(s.split(".")[1]) if "." in s else 1.0)


def _parse_num(s: str):
    s = s.strip().replace(",", "").replace("**", "")
    try:
        return float(s)
    except ValueError:
        return None


CELL_RE = re.compile(r"^\s*([-\d,\.]+)\s*±\s*([-\d,\.]+)\s*\((\d+)\)\s*$")


def _is_arm(prefix, model):
    """`encoder__covid_us-states__seed*.json` holds both the median arm (`encoder`) and the
    bias-corrected arm (`encoder_mc`) under the same horizon/metric/seed keys, so a recompute that
    does not filter on the model silently averages two different experiments together."""
    if prefix == "encoder":
        return model == "encoder"
    return str(model).startswith(f"{prefix}:")


def recompute(prefix, ds, h, metric):
    """(mean, sd, n) over seeds, straight from the result JSONs. No shared code with the generator."""
    vals = []
    for s in SEEDS:
        for sub in ("single", "lodo", "joint"):
            p = RESULTS / sub / f"{prefix}__{ds}__seed{s}.json"
            if p.exists():
                for r in json.loads(p.read_text()):
                    if (r["horizon"] == h and r["metric"] == metric
                            and _is_arm(prefix, r.get("model"))):
                        vals.append(r[FIELD])
                break
    if not vals:
        return None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, None, len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, sd, len(vals)


def check_tables(text):
    """Walk the per-dataset transfer tables and re-derive every arm cell."""
    problems, checked = [], 0
    ds = None
    for line in text.splitlines():
        m = re.match(r"^####\s+(\S+)\s*$", line)
        if m and m.group(1) in DATASETS:
            ds = m.group(1)
            continue
        if re.match(r"^####\s", line) and not (m and m.group(1) in DATASETS):
            ds = None
        if ds is None or not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 6:
            continue
        h = _parse_num(cols[0].replace("‡", ""))
        metric = cols[1]
        if h is None or int(h) not in HORIZONS:
            continue
        for j, arm in ((2, "Single (ceiling)"), (3, "LDO3 adapted"), (4, "LDO3 zero-shot")):
            cm = CELL_RE.match(cols[j])
            if not cm:
                continue
            got_m, got_sd, got_n = _parse_num(cm.group(1)), _parse_num(cm.group(2)), int(cm.group(3))
            ref = recompute(ARMS[arm], ds, int(h), metric)
            if ref is None:
                problems.append(f"{ds} h{int(h)} {metric} [{arm}]: printed a value but no artifact "
                                f"on disk")
                continue
            rm, rsd, rn = ref
            if rn != got_n:
                problems.append(f"{ds} h{int(h)} {metric} [{arm}]: doc says n={got_n}, disk has "
                                f"{rn} seeds")
            if abs(rm - got_m) > _tol_for(cm.group(1)):
                problems.append(f"{ds} h{int(h)} {metric} [{arm}]: doc mean {got_m} vs disk "
                                f"{rm:.6g}")
            if rsd is not None and abs(rsd - got_sd) > _tol_for(cm.group(2)):
                problems.append(f"{ds} h{int(h)} {metric} [{arm}]: doc sd {got_sd} vs disk "
                                f"{rsd:.6g}")
            checked += 1
    return problems, checked


def check_verdict_totals(text):
    """Row counts must sum to the printed total row, and exclusions must be the COVID long horizons."""
    problems = []
    block = re.search(r"## 3\. Transfer verdict by cell(.*?)^## ", text, re.S | re.M)
    if not block:
        return ["section 3 (verdict by cell) is missing"], 0
    rows, total = [], None
    for line in block.group(1).splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) != 8:
            continue
        if cols[0].startswith("**total**"):
            total = [_parse_num(c) for c in cols[3:7]]
            continue
        nums = [_parse_num(c) for c in cols[3:7]]
        if all(n is not None for n in nums):
            rows.append((cols[1], nums))
    if total is None:
        return ["section 3 has no total row"], 0
    for i, label in enumerate(("better", "noise", "worse", "excluded")):
        s = sum(r[1][i] for r in rows)
        if abs(s - total[i]) > 1e-9:
            problems.append(f"verdict table: '{label}' column sums to {s:g} but the total row says "
                            f"{total[i]:g}")
    # exclusions must land only on the COVID long horizons: 2 metrics x 2 horizons = 4 cells
    for ds, nums in rows:
        expect = sum(1 for h in HORIZONS if (ds, h) in NOT_ATTRIBUTABLE)
        if abs(nums[3] - expect) > 1e-9:
            problems.append(f"verdict table: {ds} excludes {nums[3]:g} cells, expected {expect}")
    return problems, len(rows)


def check_summary_prose(text):
    """The counts asserted in prose must equal the per-horizon table under them."""
    problems = []
    m = re.search(r"\*\*(\d+) cells are significantly worse than the single-disease ceiling, "
                  r"(\d+) are within noise, and (\d+) is better\.\*\*", text)
    if not m:
        return ["summary prose: could not find the worse/noise/better sentence"], 0
    prose = [int(x) for x in m.groups()]
    m2 = re.search(r"Across the (\d+) attributable", text)
    prose_total = int(m2.group(1)) if m2 else None
    m3 = re.search(r"^(\d+) further cells", text, re.M)
    prose_na = int(m3.group(1)) if m3 else None

    tbl = re.search(r"\| Horizon \| Transfer better.*?\n\|[-\| ]+\n((?:\|.*\n)+)", text)
    if not tbl:
        return ["summary prose: per-horizon table missing"], 0
    better = noise = worse = na = 0
    for line in tbl.group(1).strip().splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        better += int(cols[1]); noise += int(cols[2]); worse += int(cols[3]); na += int(cols[4])
    if [worse, noise, better] != prose:
        problems.append(f"summary prose says worse/noise/better = {prose} but the per-horizon table "
                        f"sums to {[worse, noise, better]}")
    if prose_total is not None and prose_total != better + noise + worse:
        problems.append(f"summary says {prose_total} attributable cells, table has "
                        f"{better + noise + worse}")
    if prose_na is not None and prose_na != na:
        problems.append(f"summary says {prose_na} excluded cells, table has {na}")
    return problems, 1


def check_coverage(text):
    """Every dataset must have its own transfer table, with a full 4-horizon x 7-metric grid.

    Matches the heading as a WHOLE LINE. Substring matching is not enough: `#### influenza_japan`
    also occurs inside `#### influenza_japan (held out in the *influenza* fold)` in the bootstrap
    section, so a deleted transfer table would still look present.
    """
    problems = []
    heads = set(re.findall(r"^####\s+(\S+)\s*$", text, re.M))
    for ds in DATASETS:
        if ds not in heads:
            problems.append(f"{ds}: no per-dataset transfer table in the document")
            continue
        block = re.search(rf"^####\s+{re.escape(ds)}\s*$(.*?)^#### ", text, re.S | re.M)
        if not block:
            problems.append(f"{ds}: transfer table section could not be delimited")
            continue
        seen = set()
        for line in block.group(1).splitlines():
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            h = _parse_num(cols[0].replace("‡", "")) if cols else None
            if h is not None and int(h) in HORIZONS and len(cols) > 1:
                seen.add((int(h), cols[1]))
        n_metrics = len({m for _, m in seen})
        if len(seen) != len(HORIZONS) * n_metrics or n_metrics < 7:
            problems.append(f"{ds}: table has {len(seen)} (horizon, metric) rows over {n_metrics} "
                            f"metrics; expected a full {len(HORIZONS)}x7 grid")
    return problems, len(DATASETS)


CHECKS = (("table cells vs disk", check_tables),
          ("verdict totals", check_verdict_totals),
          ("summary prose vs table", check_summary_prose),
          ("dataset coverage", check_coverage))


def run(text, verbose=True):
    allp = []
    for name, fn in CHECKS:
        p, n = fn(text)
        allp += p
        if verbose:
            print(f"  {'FAIL' if p else 'ok  '}  {name:26} ({n} checked)"
                  + (f"  {len(p)} problem(s)" if p else ""))
    if verbose and allp:
        print()
        for p in allp:
            print(f"    - {p}")
    return allp


# --------------------------------------------------------------------------- #
def mutate(text):
    """Corrupt the document one way at a time; each corruption MUST be caught.

    Without this the verifier's own passes prove nothing -- a check that never fires reads exactly
    like a check that always passes.
    """
    cases = {}
    # (1) a wrong mean in a data cell
    m = CELL_RE.match("")
    for line in text.splitlines():
        if line.startswith("| 3 | rmse |") and "±" in line:
            cols = line.split("|")
            cm = CELL_RE.match(cols[3])
            if cm:
                bad = float(cm.group(1).replace(",", "")) * 1.5
                cols[3] = f" {bad:,.1f} ± {cm.group(2)} ({cm.group(3)}) "
                cases["corrupted mean"] = text.replace(line, "|".join(cols))
                break
    # (2) a wrong seed count
    for line in text.splitlines():
        if "± " in line and "(5)" in line and line.startswith("|"):
            cases["corrupted seed count"] = text.replace(line, line.replace("(5)", "(4)", 1))
            break
    # (3) prose that disagrees with its table
    cases["prose/table mismatch"] = re.sub(
        r"\*\*(\d+) cells are significantly worse", r"**99 cells are significantly worse", text)
    # (4) a verdict total that does not match its rows
    cases["verdict total mismatch"] = re.sub(
        r"(\| \*\*total\*\* \| \| \| )\*\*(\d+)\*\*", r"\g<1>**77**", text)
    # (5) a whole dataset table dropped
    cases["missing dataset table"] = text.replace("#### influenza_japan\n", "#### GONE\n")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--doc", default="progress/outcomes/LDO3_Results.md")
    ap.add_argument("--mutate", action="store_true", help="mutation-test the verifier itself")
    a = ap.parse_args()

    text = Path(a.doc).read_text(encoding="utf-8")
    print(f"{'=' * 78}\nVERIFY DOC  {a.doc}  ({len(text.splitlines())} lines)\n{'=' * 78}")
    problems = run(text)

    if a.mutate:
        print(f"\n{'=' * 78}\nMUTATION TEST -- each corruption must be caught\n{'=' * 78}")
        missed = []
        for name, bad in mutate(text).items():
            if bad == text:
                missed.append(f"{name}: mutation did not change the document (control void)")
                print(f"  VOID  {name}")
                continue
            caught = len(run(bad, verbose=False)) > len(problems)
            print(f"  {'ok  ' if caught else 'MISS'}  {name}")
            if not caught:
                missed.append(f"{name}: corruption not detected")
        if missed:
            print()
            for m in missed:
                print(f"    - {m}")
            print("\nThe verifier is not trustworthy until every mutation is caught.")
            sys.exit(2)
        print("\n  All mutations caught: the checks above actually fire.")

    print()
    if problems:
        print(f"{len(problems)} problem(s). The document does not match the artifacts on disk.")
        sys.exit(1)
    print("Document matches the artifacts on disk.")


if __name__ == "__main__":
    main()

"""verify_anil_doc.py -- read the numbers back OUT of ANIL_Results.md and recompute them from disk.

The point is to catch a document that has drifted from its artifacts: a stale table left behind
after a re-run, a cell copied into the wrong row, a tally that no longer sums, a prose count that
contradicts the table beneath it.

So this deliberately does NOT import `anil_report`'s loaders, its path helpers, or
`results_matrix.paired_delta`. Every value is re-derived straight from `results/misc/anil_*.json`
with a local implementation of the paired delta and its own t table. A bug shared with the generator
would otherwise verify itself, which is the failure mode this file exists to prevent.

Checks:
  1. Every `delta vs control` cell in the primary tables matches a fresh paired-by-seed computation,
     to the precision it was printed at, and its significance label matches too.
  2. Every `vs reference` cell in the secondary tables likewise.
  3. Each fold's printed tally matches the labels in its own table, and the summary table's per-fold
     rows match the per-fold section tallies.
  4. The summary prose counts agree with the summary table.
  5. Coverage: every (fold, arm) on disk appears in the seeds table, with the right seeds, and every
     scored cell appears in that fold's primary table.

    python diagnostics/verify_anil_doc.py
    python diagnostics/verify_anil_doc.py --mutate     # the check on the check
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

RESULTS = Path("results")
SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
FIELD = "country_macro"
LEGACY_FOLD = "dengue2flu"
FOLDS = (LEGACY_FOLD, "dengue", "influenza", "covid")
ARMS = ("anil", "control")

# Own t table, deliberately not imported. Two-sided 95%, keyed by df = n - 1.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228}

DISEASES = {"dengue": ("dengue",),
            "influenza": ("influenza_japan", "influenza_us-regions", "influenza_us-states"),
            "covid": ("covid_us-states",)}


# --------------------------------------------------------------------------- #
def _tol_for(printed: str) -> float:
    """Half a unit in the last printed digit -- the most the formatter can have rounded away."""
    s = printed.replace(",", "").replace("**", "").replace("%", "").strip().lstrip("+")
    return 0.5 * (10 ** -len(s.split(".")[1]) if "." in s else 1.0)


def _num(s):
    try:
        return float(str(s).replace(",", "").replace("**", "").replace("%", "").strip())
    except ValueError:
        return None


def _out_json(fold, arm):
    tag = "" if fold == LEGACY_FOLD else ("ldo3" + fold + "_")
    return RESULTS / "misc" / f"anil_{tag}affine_{arm}.json"


def _rows(fold, arm):
    p = _out_json(fold, arm)
    if not p.exists():
        return {}
    return {r["seed"]: r for r in json.loads(p.read_text(encoding="utf-8"))}


def _cells(fold, arm):
    """{'ds|hH': {seed: rmse}} straight from the resume JSON."""
    out = {}
    for seed, r in _rows(fold, arm).items():
        for c, v in (r.get("rmse") or {}).items():
            out.setdefault(c, {})[seed] = v
    return out


def _reference(fold):
    out = {}
    if fold == LEGACY_FOLD:
        p = RESULTS / "misc" / "capacity_probe_5seed.json"
        if not p.exists():
            return out
        for per_seed in json.loads(p.read_text(encoding="utf-8")).get("cross_disease", []):
            for row in per_seed:
                if row.get("label") == "affine (current)":
                    for c, v in row["rmse"].items():
                        out.setdefault(c, {})[row["seed"]] = v
        return out
    for ds in DISEASES[fold]:
        for s in SEEDS:
            p = RESULTS / "lodo" / f"encoder_ldo3__{ds}__seed{s}.json"
            if not p.exists():
                continue
            for r in json.loads(p.read_text(encoding="utf-8")):
                if r["metric"] == "rmse":
                    out.setdefault(f"{r['dataset']}|h{r['horizon']}", {})[s] = float(r[FIELD])
    return out


def paired(new: dict, ref: dict):
    """(mean, halfwidth, n, clears) -- improvement percent, positive = `new` better. Local."""
    ds = []
    for s in sorted(set(new) & set(ref)):
        a, b = new[s], ref[s]
        if a is None or b is None or abs(b) < 1e-12:
            continue
        ds.append((b - a) / abs(b) * 100.0)
    if not ds:
        return None, None, 0, None
    n = len(ds)
    m = sum(ds) / n
    if n < 2:
        return m, None, n, None
    sd = math.sqrt(sum((x - m) ** 2 for x in ds) / (n - 1))
    hw = _T95.get(n - 1, 1.96) * sd / math.sqrt(n)
    return m, hw, n, abs(m) > hw


# --------------------------------------------------------------------------- #
# Parsing the document
# --------------------------------------------------------------------------- #
FOLD_H3_RE = re.compile(r"^### Fold `([^`]+)`")
FOLD_H4_RE = re.compile(r"^### `([^`]+)`\s*$")
ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(h\d+)\s*\|(.*)\|\s*$")
TALLY_RE = re.compile(r"Tally:\s*\*\*(\d+)\s*better,\s*(\d+)\s*worse,\s*(\d+)\s*within noise")
SUMROW_RE = re.compile(r"^\|\s*`([^`]+)`(?:\s*\(first run\))?\s*\|[^|]*\|[^|]*\|\s*(\d+)\s*\|"
                       r"\s*(\d+)\s*\|\s*(\d+)\s*\|")
SEEDROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(anil|control)\s*\|\s*([\d, ]+)\s*\((\d+)\)\s*\|")
PROSE_RE = re.compile(r"better in \*\*(\d+)\*\* of (\d+) cells, worse in \*\*(\d+)\*\*, and within "
                      r"noise in \*\*(\d+)\*\*")

# "within noise (-4.0 ± 18.7%, n=5)"  |  "**-1.4%** ± 1.2 (n=5)"  |  "+0.5% (1 seed, untestable)"
NOISE_RE = re.compile(r"within noise \(([-+\d.]+)\s*±\s*([\d.]+)%,\s*n=(\d+)\)")
SIG_RE = re.compile(r"\*\*([-+][\d.]+)%\*\*\s*±\s*([\d.]+)\s*\(n=(\d+)\)")


def _parse_delta(text):
    """(mean, halfwidth, n, clears) from a rendered delta cell, or None if unrecognised."""
    t = text.strip()
    if t in ("—", ""):
        return None
    m = NOISE_RE.search(t)
    if m:
        return float(m.group(1)), float(m.group(2)), int(m.group(3)), False
    m = SIG_RE.search(t)
    if m:
        return float(m.group(1)), float(m.group(2)), int(m.group(3)), True
    return None


def _sections(text):
    """{fold: {'primary': [(cell, raw)], 'secondary': [(cell, raw_anil, raw_ctrl)], 'tally': (..)}}"""
    out = {f: dict(primary=[], secondary=[], tally=None) for f in FOLDS}
    fold = None
    mode = None
    for line in text.splitlines():
        if line.startswith("## 3."):
            mode = "primary"
            fold = None
            continue
        if line.startswith("## 4."):
            mode = "secondary"
            fold = None
            continue
        if line.startswith("## 5."):
            mode = None
            fold = None
            continue
        m = FOLD_H3_RE.match(line) or FOLD_H4_RE.match(line)
        if m and m.group(1) in FOLDS:
            fold = m.group(1)
            continue
        if fold is None or mode is None:
            continue
        m = TALLY_RE.search(line)
        if m and mode == "primary":
            out[fold]["tally"] = tuple(int(g) for g in m.groups())
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        ds, h, rest = m.group(1), m.group(2), m.group(3)
        parts = [p.strip() for p in rest.split("|")]
        key = f"{ds}|{h}"
        if mode == "primary" and len(parts) >= 2:
            out[fold]["primary"].append((key, parts[0], parts[1]))
        elif mode == "secondary" and len(parts) >= 2:
            out[fold]["secondary"].append((key, parts[0], parts[1]))
    return out


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_primary(text):
    problems = []
    secs = _sections(text)
    for fold in FOLDS:
        rows = secs[fold]["primary"]
        if not rows:
            problems.append(f"{fold}: no primary (vs control) table rows found")
            continue
        anil, ctrl = _cells(fold, "anil"), _cells(fold, "control")
        for key, delta_txt, sig_txt in rows:
            got = _parse_delta(delta_txt)
            if got is None:
                problems.append(f"{fold} {key}: unparseable delta cell {delta_txt!r}")
                continue
            pm, phw, pn, pclears = got
            m, hw, n, clears = paired(anil.get(key, {}), ctrl.get(key, {}))
            if m is None:
                problems.append(f"{fold} {key}: printed a delta but no artifact pairs support it")
                continue
            if n != pn:
                problems.append(f"{fold} {key}: printed n={pn}, artifacts pair on n={n}")
            if abs(m - pm) > _tol_for(f"{pm}"):
                problems.append(f"{fold} {key}: printed mean {pm:+.2f}%, recomputed {m:+.2f}%")
            if phw is not None and hw is not None and abs(hw - phw) > _tol_for(f"{phw}"):
                problems.append(f"{fold} {key}: printed halfwidth {phw}, recomputed {hw:.2f}")
            if pclears != clears:
                problems.append(f"{fold} {key}: printed significance {pclears}, recomputed {clears}")
            label = sig_txt.strip().lower()
            want = "yes" if clears else ("within noise" if clears is False else "untestable")
            if label != want:
                problems.append(f"{fold} {key}: significance label {label!r} should be {want!r}")
    return problems


def check_secondary(text):
    problems = []
    secs = _sections(text)
    for fold in FOLDS:
        ref = _reference(fold)
        if not ref:
            problems.append(f"{fold}: no reference on disk to check the secondary table against")
            continue
        for key, a_txt, c_txt in secs[fold]["secondary"]:
            for arm, txt in (("anil", a_txt), ("control", c_txt)):
                got = _parse_delta(txt)
                if got is None:
                    continue
                pm, _phw, pn, pclears = got
                m, _hw, n, clears = paired(_cells(fold, arm).get(key, {}), ref.get(key, {}))
                if m is None:
                    problems.append(f"{fold} {key} [{arm}]: printed a delta with no support")
                    continue
                if n != pn:
                    problems.append(f"{fold} {key} [{arm}]: printed n={pn}, recomputed n={n}")
                if abs(m - pm) > _tol_for(f"{pm}"):
                    problems.append(f"{fold} {key} [{arm}]: printed {pm:+.2f}%, "
                                    f"recomputed {m:+.2f}%")
                if pclears != clears:
                    problems.append(f"{fold} {key} [{arm}]: printed significance {pclears}, "
                                    f"recomputed {clears}")
    return problems


def check_tallies(text):
    """Each fold's printed tally must match the labels in its own table AND the summary row."""
    problems = []
    secs = _sections(text)
    summary = {m.group(1): tuple(int(g) for g in m.groups()[1:])
               for m in (SUMROW_RE.match(l) for l in text.splitlines()) if m}
    for fold in FOLDS:
        rows = secs[fold]["primary"]
        if not rows:
            continue
        b = w = n = 0
        for _key, delta_txt, _sig in rows:
            got = _parse_delta(delta_txt)
            if got is None:
                continue
            m, _hw, _n, clears = got
            if clears and m > 0:
                b += 1
            elif clears and m < 0:
                w += 1
            elif clears is False:
                n += 1
        printed = secs[fold]["tally"]
        if printed is None:
            problems.append(f"{fold}: no 'Tally:' line under its primary table")
        elif printed != (b, w, n):
            problems.append(f"{fold}: tally line says {printed}, its own rows say {(b, w, n)}")
        if fold in summary and summary[fold] != (b, w, n):
            problems.append(f"{fold}: summary table row {summary[fold]} != its section's "
                            f"{(b, w, n)}")
        elif fold not in summary:
            problems.append(f"{fold}: missing from the summary table")
    return problems


def check_summary_prose(text):
    """The headline sentence must equal the sum of the summary table."""
    problems = []
    m = PROSE_RE.search(text)
    if not m:
        return ["summary prose counts not found (the headline sentence changed shape)"]
    pb, ptot, pw, pn = (int(g) for g in m.groups())
    rows = [tuple(int(g) for g in mm.groups()[1:])
            for mm in (SUMROW_RE.match(l) for l in text.splitlines()) if mm]
    if not rows:
        return ["summary table rows not found"]
    b = sum(r[0] for r in rows)
    w = sum(r[1] for r in rows)
    n = sum(r[2] for r in rows)
    if (pb, pw, pn) != (b, w, n):
        problems.append(f"prose says better={pb} worse={pw} noise={pn}; "
                        f"summary table sums to {b}/{w}/{n}")
    if ptot != b + w + n:
        problems.append(f"prose says {ptot} cells; the table accounts for {b + w + n}")
    return problems


def check_coverage(text):
    """Nothing on disk may be silently absent from the document."""
    problems = []
    secs = _sections(text)
    seen = {m.group(1) + "/" + m.group(2): (m.group(3), int(m.group(4)))
            for m in (SEEDROW_RE.match(l) for l in text.splitlines()) if m}
    for fold in FOLDS:
        for arm in ARMS:
            rows = _rows(fold, arm)
            if not rows:
                continue
            key = f"{fold}/{arm}"
            if key not in seen:
                problems.append(f"{key}: scored on disk but absent from the seeds table")
                continue
            listed = [int(x) for x in seen[key][0].replace(" ", "").split(",") if x]
            if sorted(listed) != sorted(rows):
                problems.append(f"{key}: seeds table says {sorted(listed)}, disk has {sorted(rows)}")
            if seen[key][1] != len(rows):
                problems.append(f"{key}: seeds table count {seen[key][1]} != {len(rows)} on disk")
        on_disk = set(_cells(fold, "anil")) & set(_cells(fold, "control"))
        in_doc = {k for k, _d, _s in secs[fold]["primary"]}
        for miss in sorted(on_disk - in_doc):
            problems.append(f"{fold}: cell {miss} is scored on disk but missing from the table")
    return problems


CHECKS = (("primary table", check_primary),
          ("secondary table", check_secondary),
          ("tallies", check_tallies),
          ("summary prose", check_summary_prose),
          ("coverage", check_coverage))


def run(text, verbose=True):
    all_problems = []
    for name, fn in CHECKS:
        problems = fn(text)
        all_problems += problems
        if verbose:
            print(f"  {'FAIL' if problems else 'ok  '}  {name}"
                  + (f"  ({len(problems)} problem(s))" if problems else ""))
            for p in problems:
                print(f"        - {p}")
    return all_problems


# --------------------------------------------------------------------------- #
def mutate(text):
    """Corrupt the document one defect at a time; every corruption MUST be caught.

    A verifier that passes everything is worth nothing. Each mutation below is a real failure mode:
    a stale number, a flipped significance label, a tally that no longer sums, a prose count left
    behind after a re-run, and a dropped row.
    """
    muts = []

    # a stale number in a primary cell
    m = SIG_RE.search(text)
    if m:
        muts.append(("stale mean in a significant cell",
                     text.replace(m.group(0), m.group(0).replace(m.group(1), "-9.9"), 1)))
    # a flipped significance label, targeted at a real DATA row rather than at the summary table
    # header, which also ends in "| within noise |" and would make this mutation vacuous.
    for line in text.splitlines():
        mm = ROW_RE.match(line)
        if mm and line.rstrip().endswith("| within noise |"):
            muts.append(("significance label flipped to yes",
                         text.replace(line, line[:line.rindex("| within noise |")] + "| yes |", 1)))
            break
    # a tally that no longer matches its own rows
    m = TALLY_RE.search(text)
    if m:
        muts.append(("tally line inflated",
                     text.replace(m.group(0), "Tally: **3 better, 0 worse, 9 within noise", 1)))
    # prose left stale after a re-run
    m = PROSE_RE.search(text)
    if m:
        muts.append(("summary prose stale",
                     text.replace(m.group(0), m.group(0).replace(f"**{m.group(3)}**", "**7**"), 1)))
    # a dropped table row
    for line in text.splitlines():
        mm = ROW_RE.match(line)
        if mm:
            muts.append(("a primary row deleted", text.replace(line + "\n", "", 1)))
            break
    # a seed removed from the coverage table
    for line in text.splitlines():
        mm = SEEDROW_RE.match(line)
        if mm and "82" in mm.group(3):
            muts.append(("a seed dropped from the seeds table",
                         text.replace(line, line.replace(", 82", "", 1), 1)))
            break

    print(f"{'=' * 78}\nMutation test: {len(muts)} corruptions, each must be caught\n{'=' * 78}")
    escaped = []
    for name, bad in muts:
        problems = run(bad, verbose=False)
        status = "caught" if problems else "NOT CAUGHT"
        print(f"  {status:<11} {name}"
              + (f"  ({len(problems)} problem(s))" if problems else ""))
        if not problems:
            escaped.append(name)
    if escaped:
        print(f"\n{len(escaped)} mutation(s) escaped. The verifier does not check what it claims to.")
    else:
        print(f"\nAll {len(muts)} mutations caught.")
    return escaped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc", nargs="?", default="progress/outcomes/ANIL_Results.md")
    ap.add_argument("--mutate", action="store_true", help="mutation-test the verifier itself")
    a = ap.parse_args()

    p = Path(a.doc)
    if not p.exists():
        sys.exit(f"no such document: {p}")
    text = p.read_text(encoding="utf-8")

    print(f"{'=' * 78}\nVerifying {p} against the artifacts on disk\n{'=' * 78}")
    problems = run(text)

    if a.mutate:
        print()
        escaped = mutate(text)
        if escaped:
            sys.exit(2)

    if problems:
        print(f"\n{len(problems)} PROBLEM(S). The document does not match its artifacts.")
        sys.exit(1)
    print(f"\nOK: {p} matches the artifacts on disk.")


if __name__ == "__main__":
    main()
"""verify_v2_result_doc.py -- read the numbers back OUT of V2_Deviation_Result_2026-09-25.md and
recompute every one of them from the records on disk with independent code.

The point is to catch a document that has drifted from its artifacts: a stale delta left after a
re-run, a flipped verdict label, a confidence bound copied into the wrong row, a header statistic that
no longer matches the val files. So this deliberately does NOT import `ablation.run_v2_deviation`: no
its record readers, its paired delta, its power function or its seed-count formula. Every value is
re-derived straight from the JSON records with a local paired-t implementation and its own noncentral-t
power. A bug shared with the runner would otherwise verify itself, which is the failure mode this file
exists to prevent.

Checks:
  1. The protocol sha256 quoted in the doc equals the sha256 of the protocol file.
  2. The branch parameter count quoted in the doc equals the count stored in the records.
  3. The stage-1 spread s and the seed count N and power quoted in the header are recomputed from the
     validation files and the protocol formula, and match.
  4. Every cell of the v2graph-vs-v1 verdict table (ref mean, arm mean, d, both CI bounds, verdict
     label) matches a fresh paired-by-seed computation, to the precision it was printed at.
  5. Every d in the arms-vs-v1 attribution table matches.
  6. Every cell of the v2graph-vs-controls table (d, both CI bounds, verdict) matches.
  7. The PCC h3 collapse quoted in prose (v1 mean, v2graph mean, d) matches.

    conda run -n ebola-train python diagnostics/verify_v2_result_doc.py
    conda run -n ebola-train python diagnostics/verify_v2_result_doc.py --mutate   # the check on the check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "progress" / "outcomes" / "V2_Deviation_Result_2026-09-25.md"
PROTOCOL = ROOT / "progress" / "decisions" / "V2_Deviation_Protocol.md"
ABL = ROOT / "ablation" / "single"
RES = ROOT / "results" / "single"
DS = "covid_us-states"
SEEDS = tuple(42 + 10 * i for i in range(15))          # 42, 52, ..., 182
STAGE1 = SEEDS[:5]
FIELD = "country_macro"
MODEL = "encoder"
DELTA = 0.033
# doc label -> record filename tag
TAG = {"v1": "v1", "v2graph": "v2graph", "v2nograph": "v2nograph", "v2shuffled": "v2shuf"}


# --------------------------------------------------------------------------- #
# Reading records, independently of the runner
# --------------------------------------------------------------------------- #
def _cells(path, split=None):
    """{(horizon, metric): country_macro} from one record file, encoder model only."""
    out = {}
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        if r["model"] != MODEL:
            continue
        if split is not None and r.get("split") != split:
            raise AssertionError(f"{path} carries split={r.get('split')}, wanted {split}")
        out[(r["horizon"], r["metric"])] = r[FIELD]
    return out


def _v1_test(s):
    p = RES / f"encoder__{DS}__seed{s}.json"
    return p if p.exists() else ABL / f"encoder__{DS}__seed{s}__v1ref.json"


def _arm_test(s, label):
    return ABL / f"encoder__{DS}__seed{s}__{TAG[label]}.json"


def _v1_val(s):
    tag = "v1" if s in STAGE1 else "v1ref"
    return ABL / f"encoder__{DS}__seed{s}__{tag}__val.json"


def _arm_val(s, label):
    return ABL / f"encoder__{DS}__seed{s}__{TAG[label]}__val.json"


def series(which, label, h, m, split="test", seeds=SEEDS):
    """{seed: value} for one arm at one (h, metric), test or val."""
    out = {}
    for s in seeds:
        if which == "v1":
            p = _v1_test(s) if split == "test" else _v1_val(s)
        else:
            p = _arm_test(s, label) if split == "test" else _arm_val(s, label)
        out[s] = _cells(p, split=None if split == "test" else "val")[(h, m)]
    return out


def paired(arm, ref, higher_better=False):
    """(ref mean, arm mean, d mean, lo, hi, verdict) for d = arm - ref per shared seed."""
    seeds = sorted(set(arm) & set(ref))
    d = [arm[s] - ref[s] for s in seeds]
    n = len(d)
    rm = sum(ref[s] for s in seeds) / n
    am = sum(arm[s] for s in seeds) / n
    dm = sum(d) / n
    sd = math.sqrt(sum((x - dm) ** 2 for x in d) / (n - 1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    lo, hi = dm - half, dm + half
    if higher_better:                                   # PCC: a higher value is the better one
        verdict = "better" if lo > 0 else ("worse" if hi < 0 else "noise")
    else:                                               # error: a lower value is the better one
        verdict = "worse" if lo > 0 else ("better" if hi < 0 else "noise")
    return rm, am, dm, lo, hi, verdict


def stage1_s():
    """The largest of the three paired relative sds at h3 RMSE on the VALIDATION split, over the five
    stage-1 seeds. Independent re-derivation of the protocol's s."""
    v1 = series("v1", "v1", 3, "rmse", split="val", seeds=STAGE1)
    v1m = sum(v1.values()) / len(v1)
    rels = []
    for ref_label, arm_label in (("v1", "v2graph"), ("v2nograph", "v2graph"), ("v2shuffled", "v2graph")):
        ref = v1 if ref_label == "v1" else series("arm", ref_label, 3, "rmse", split="val", seeds=STAGE1)
        arm = series("arm", arm_label, 3, "rmse", split="val", seeds=STAGE1)
        d = [arm[s] - ref[s] for s in STAGE1]
        dm = sum(d) / len(d)
        sd = math.sqrt(sum((x - dm) ** 2 for x in d) / (len(d) - 1))
        rels.append(sd / v1m)
    return max(rels)


def _power(s, n):
    k = DELTA / s
    crit = stats.t.ppf(0.975, n - 1)
    return float(stats.nct.sf(crit, n - 1, k * math.sqrt(n)))


def seed_count(s):
    """n_z = max(5, ceil(((1.96+0.84) s / delta)^2)); t-correct up to the smallest n with power >= 0.80;
    N = min(that, 134). Independent of the runner's implementation."""
    n_z = max(5, math.ceil(((1.96 + 0.84) * s / DELTA) ** 2))
    n = n_z
    while _power(s, n) < 0.80:
        n += 1
    return min(n, 134)


# --------------------------------------------------------------------------- #
# Parsing the document
# --------------------------------------------------------------------------- #
def _num(x):
    return float(str(x).replace(",", "").replace("+", "").strip())


def _tol(printed):
    s = str(printed).replace(",", "").replace("+", "").replace("-", "").strip()
    return 0.5 * (10 ** -len(s.split(".")[1]) if "." in s else 1.0)

# | RMSE | 3 | 5306.760 | 10444.773 | +5138.013 | [+3669.034, +6606.992] | worse |
MAIN_RE = re.compile(
    r"^\|\s*(RMSE|MAE)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([+-][\d.]+)\s*\|"
    r"\s*\[\s*([+-][\d.]+)\s*,\s*([+-][\d.]+)\s*\]\s*\|\s*(\w+)\s*\|")
# | v2graph vs v1 | +5138.013 | +11521.231 | +3464.536 | +6232.448 |
ATTR2_RE = re.compile(
    r"^\|\s*(v2\w+) vs v1\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*\|"
    r"\s*([+-][\d.]+)\s*\|")
# | v2graph vs v2nograph | RMSE | 3 | +545.588 | [-245.477, +1336.653] | noise |
ATTR3_RE = re.compile(
    r"^\|\s*v2graph vs (v2\w+)\s*\|\s*(RMSE|MAE)\s*\|\s*(\d+)\s*\|\s*([+-][\d.]+)\s*\|"
    r"\s*\[\s*([+-][\d.]+)\s*,\s*([+-][\d.]+)\s*\]\s*\|\s*(\w+)\s*\|")


def check_header(text):
    problems = []
    m = re.search(r"sha256\s*\n?`?([0-9a-f]{64})`?", text)
    want = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if not m:
        problems.append("no protocol sha256 found in the doc")
    elif m.group(1) != want:
        problems.append(f"protocol sha256 in doc {m.group(1)[:12]}.. != file {want[:12]}..")

    m = re.search(r"branch adds ([\d,]+) parameters", text)
    disk = json.loads((ABL / f"encoder__{DS}__seed42__v2graph.json").read_text())
    bp = [r for r in disk if r["model"] == MODEL][0]["v2_branch_params"]
    if not m:
        problems.append("no branch parameter count found")
    elif int(_num(m.group(1))) != bp:
        problems.append(f"branch params in doc {m.group(1)} != records {bp}")

    s = stage1_s()
    m = re.search(r"s = ([\d.]+)", text)
    if not m:
        problems.append("no s value found in the doc")
    elif abs(_num(m.group(1)) - s) > _tol(m.group(1)):
        problems.append(f"s in doc {m.group(1)} != recomputed {s:.4f}")

    N = seed_count(s)
    m = re.search(r"N = (\d+) seeds", text)
    if not m:
        problems.append("no seed count N found in the doc")
    elif int(m.group(1)) != N:
        problems.append(f"N in doc {m.group(1)} != recomputed {N}")

    pw = _power(s, N)
    m = re.search(r"([\d.]+) chance of finding", text)
    if not m:
        problems.append("no power value found in the doc")
    elif abs(_num(m.group(1)) - pw) > _tol(m.group(1)):
        problems.append(f"power in doc {m.group(1)} != recomputed {pw:.3f}")
    return problems


def check_main_table(text):
    problems = []
    seen = 0
    for line in text.splitlines():
        mm = MAIN_RE.match(line)
        if not mm:
            continue
        seen += 1
        metric, h = mm.group(1).lower(), int(mm.group(2))
        p_rm, p_am, p_d, p_lo, p_hi, p_v = (mm.group(i) for i in range(3, 9))
        arm = series("arm", "v2graph", h, metric)
        ref = series("v1", "v1", h, metric)
        rm, am, dm, lo, hi, verdict = paired(arm, ref)
        for name, printed, got in (("ref mean", p_rm, rm), ("arm mean", p_am, am),
                                   ("d", p_d, dm), ("CI lo", p_lo, lo), ("CI hi", p_hi, hi)):
            if abs(_num(printed) - got) > _tol(printed):
                problems.append(f"v2graph vs v1 {metric} h{h}: {name} printed {printed}, "
                                f"recomputed {got:+.3f}")
        if p_v.strip().lower() != verdict:
            problems.append(f"v2graph vs v1 {metric} h{h}: verdict {p_v!r} should be {verdict!r}")
    if seen != 8:
        problems.append(f"main table: parsed {seen} rows, expected 8")
    return problems


def check_attr2_table(text):
    problems = []
    cols = ((3, "rmse"), (5, "rmse"), (3, "mae"), (5, "mae"))
    seen = 0
    for line in text.splitlines():
        mm = ATTR2_RE.match(line)
        if not mm:
            continue
        seen += 1
        label = mm.group(1)
        for i, (h, metric) in enumerate(cols):
            printed = mm.group(2 + i)
            arm = series("arm", label, h, metric)
            ref = series("v1", "v1", h, metric)
            _rm, _am, dm, _lo, _hi, _v = paired(arm, ref)
            if abs(_num(printed) - dm) > _tol(printed):
                problems.append(f"{label} vs v1 {metric} h{h}: d printed {printed}, "
                                f"recomputed {dm:+.3f}")
    if seen != 3:
        problems.append(f"arms-vs-v1 table: parsed {seen} rows, expected 3")
    return problems


def check_attr3_table(text):
    problems = []
    seen = 0
    for line in text.splitlines():
        mm = ATTR3_RE.match(line)
        if not mm:
            continue
        seen += 1
        ref_label, metric, h = mm.group(1), mm.group(2).lower(), int(mm.group(3))
        p_d, p_lo, p_hi, p_v = mm.group(4), mm.group(5), mm.group(6), mm.group(7)
        arm = series("arm", "v2graph", h, metric)
        ref = series("arm", ref_label, h, metric)
        _rm, _am, dm, lo, hi, verdict = paired(arm, ref)
        for name, printed, got in (("d", p_d, dm), ("CI lo", p_lo, lo), ("CI hi", p_hi, hi)):
            if abs(_num(printed) - got) > _tol(printed):
                problems.append(f"v2graph vs {ref_label} {metric} h{h}: {name} printed {printed}, "
                                f"recomputed {got:+.3f}")
        if p_v.strip().lower() != verdict:
            problems.append(f"v2graph vs {ref_label} {metric} h{h}: verdict {p_v!r} "
                            f"should be {verdict!r}")
    if seen != 8:
        problems.append(f"v2graph-vs-controls table: parsed {seen} rows, expected 8")
    return problems


def check_pcc_prose(text):
    """The h3 PCC collapse is prose, so no table check touches it."""
    problems = []
    m = re.search(r"collapses from ([\d.]+) to (-?[\d.]+)\s+\(d (-?[\d.]+)\)", text)
    if not m:
        return ["PCC h3 collapse sentence not found (its shape changed)"]
    p_v1, p_arm, p_d = m.group(1), m.group(2), m.group(3)
    arm = series("arm", "v2graph", 3, "pcc")
    ref = series("v1", "v1", 3, "pcc")
    rm, am, dm, _lo, _hi, _v = paired(arm, ref, higher_better=True)
    for name, printed, got in (("v1 pcc", p_v1, rm), ("v2graph pcc", p_arm, am), ("d", p_d, dm)):
        if abs(_num(printed) - got) > _tol(printed):
            problems.append(f"PCC h3 collapse: {name} printed {printed}, recomputed {got:+.3f}")
    return problems


CHECKS = (("header (sha, params, s, N, power)", check_header),
          ("v2graph-vs-v1 verdict table", check_main_table),
          ("arms-vs-v1 attribution table", check_attr2_table),
          ("v2graph-vs-controls table", check_attr3_table),
          ("PCC h3 collapse prose", check_pcc_prose))


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
    """Corrupt the document one defect at a time; every corruption MUST be caught."""
    muts = []
    # a stale delta in a significant main-table cell
    muts.append(("stale delta in the main table",
                 text.replace("+5138.013", "+9999.999", 1)))
    # a flipped verdict label in the main table
    muts.append(("main-table verdict flipped worse->better",
                 text.replace("| +5138.013 | [+3669.034, +6606.992] | worse |",
                              "| +5138.013 | [+3669.034, +6606.992] | better |", 1)))
    # a confidence bound moved
    muts.append(("main-table CI bound corrupted",
                 text.replace("[+3669.034, +6606.992]", "[+3000.000, +6606.992]", 1)))
    # a stale reference mean
    muts.append(("main-table ref mean stale",
                 text.replace("| RMSE | 3 | 5306.760 |", "| RMSE | 3 | 5999.999 |", 1)))
    # the header spread s made stale
    muts.append(("header s stale", text.replace("s = 0.0412", "s = 0.0600", 1)))
    # the seed count changed
    muts.append(("header N wrong", text.replace("N = 15 seeds", "N = 21 seeds", 1)))
    # a stale delta in the arms-vs-v1 attribution table
    muts.append(("attribution-table delta stale",
                 text.replace("| v2nograph vs v1 | +4592.425 |", "| v2nograph vs v1 | +7777.777 |", 1)))
    # a within-noise verdict flipped in the controls table
    muts.append(("controls-table verdict flipped noise->worse",
                 text.replace("| +545.588 | [-245.477, +1336.653] | noise |",
                              "| +545.588 | [-245.477, +1336.653] | worse |", 1)))
    # a stale PCC collapse number
    muts.append(("PCC collapse number stale",
                 text.replace("collapses from 0.436 to -0.014", "collapses from 0.436 to -0.500", 1)))
    # the branch parameter count made wrong
    muts.append(("branch param count wrong",
                 text.replace("branch adds 10,224 parameters", "branch adds 12,000 parameters", 1)))
    # the protocol sha corrupted
    muts.append(("protocol sha corrupted",
                 text.replace("82069f4138099ae154b30e4c20e214ac3181b2975e2d9c9f5349cf091c8189ef",
                              "00000f4138099ae154b30e4c20e214ac3181b2975e2d9c9f5349cf091c8189ef", 1)))

    print(f"{'=' * 78}\nMutation test: {len(muts)} corruptions, each must be caught\n{'=' * 78}")
    escaped = []
    for name, bad in muts:
        if bad == text:
            print(f"  NO-OP       {name}  (mutation did not change the text)")
            escaped.append(name)
            continue
        problems = run(bad, verbose=False)
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
    ap.add_argument("doc", nargs="?", default=str(DOC))
    ap.add_argument("--mutate", action="store_true", help="mutation-test the verifier itself")
    a = ap.parse_args()

    p = Path(a.doc)
    if not p.exists():
        sys.exit(f"no such document: {p}")
    text = p.read_text(encoding="utf-8")

    print(f"{'=' * 78}\nVerifying {p.name} against the records on disk\n{'=' * 78}")
    problems = run(text)

    if a.mutate:
        print()
        escaped = mutate(text)
        if escaped:
            sys.exit(2)

    if problems:
        print(f"\n{len(problems)} PROBLEM(S). The document does not match its artifacts.")
        sys.exit(1)
    print(f"\nOK: {p.name} matches the records on disk.")


if __name__ == "__main__":
    main()

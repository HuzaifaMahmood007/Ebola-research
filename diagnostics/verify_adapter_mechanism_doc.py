"""Verify progress/outcomes/Adapter_Mechanism_2026-09-16.md against results/misc/ebolashift__adapter_mechanism.json.

House rule (CLAUDE.md section 7): parse the numbers OUT of the document and recompute them from disk.
Third in the series, after verify_feature_shift_doc.py and verify_fewshot_sim_doc.py.

Checks the Q1 table cell by cell, the Q2 column-profile block and its summary table, the per-column
support counts printed as a code block, and the prose ranges ("76 and 85 per cent", "5 to 11 times")
which are recomputed from the cells rather than trusted.

  conda run -n ebola-train python -m diagnostics.verify_adapter_mechanism_doc
  conda run -n ebola-train python -m diagnostics.verify_adapter_mechanism_doc --mutate
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re

import numpy as np

DOC = pathlib.Path("progress/outcomes/Adapter_Mechanism_2026-09-16.md")
JSON = pathlib.Path("results/misc/ebolashift__adapter_mechanism.json")
COLS = ["n_sup", "rank", "cond", "out_of_span", "extrap_share", "extrap_median"]


def flatten(text):
    """Doc text as one line, with markdown blockquote markers dropped.

    A plain re.sub(r"\s+", " ") is not enough: a claim wrapped across two lines of a `>` blockquote
    flattens to "... 5 to 12 times > beyond the fitted range", and every contiguous-phrase check
    below then fails on a document whose numbers are correct. That happened, and the doc was right.
    """
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*>\s?", "", text))


def _num(s):
    s = s.replace("*", "").replace("%", "").replace("x", "").strip()
    return None if s in ("n/a", "-", "") else float(s)


def mean(cells, arm, h, key):
    v = [c[key] for c in cells if c["arm"] == arm and c["h"] == h and not math.isnan(c[key])]
    return float(np.mean(v)) if v else float("nan")


def check_q1(text, data, fails):
    cells = data["cells"]
    n = 0
    for m in re.finditer(r"^\|\s*(L12|L20)\s*\|\s*(\d+)\s*\|(.+)\|\s*$", text, re.M):
        arm, h = f"ebola_{m.group(1)}", int(m.group(2))
        vals = [_num(c) for c in m.group(3).split("|")]
        if len(vals) != len(COLS):
            fails.append(f"{arm} h{h}: {len(vals)} cells, expected {len(COLS)}")
            continue
        n += 1
        for key, said in zip(COLS, vals):
            got = mean(cells, arm, h, key)
            pct = key in ("out_of_span", "extrap_share")
            if pct and got == got:
                got *= 100
            if said is None and (got != got):
                continue
            if said is None or got != got:
                fails.append(f"{arm} h{h} {key}: doc {said!r}, disk {got!r}")
            elif key == "cond":
                if abs(math.log10(said) - math.log10(got)) > 0.02:
                    fails.append(f"{arm} h{h} cond: doc {said:.3e}, disk {got:.3e}")
            else:
                tol = 0.5 if key in ("n_sup", "rank") else (0.051 if pct else 0.006)
                if abs(said - got) > tol:
                    fails.append(f"{arm} h{h} {key}: doc {said}, disk {got:.4f} (tol {tol})")
    if n != 8:
        fails.append(f"parsed {n} Q1 rows, expected 8")


def check_q2(text, data, fails):
    prof = data["profiles"]
    for arm, short in (("ebola_L12", "L12"), ("ebola_L20", "L20")):
        p = prof[arm]
        want = f"{arm}  cols 0..{p['last_col']}   {p['per_column']}".replace("'", "")
        flat = flatten(text)
        if re.sub(r"\s+", " ", want) not in flat:
            fails.append(f"Q2 per-column line for {arm} does not match disk: expected {p['per_column']}")
        # summary table row: | label | L12 | L20 |
    flat = flatten(text)
    for label, key, fmt in (("support cells", "total", "{:d}"),
                            ("longest run of wholly unreported columns", "longest_blackout",
                             "**{:d}**"),
                            ("share of cells in the final two columns", "tail2_share", "**{:.1%}**")):
        a, b = prof["ebola_L12"][key], prof["ebola_L20"][key]
        want = f"| {label} | {fmt.format(a)} | {fmt.format(b)} |"
        if re.sub(r"\s+", " ", want) not in flat:
            fails.append(f"Q2 summary row {label!r} should read {fmt.format(a)} / {fmt.format(b)}")


def check_prose(text, data, fails):
    cells = data["cells"]
    flat = flatten(text)
    # A range quoted off the seed-MEAN table is narrower than the range across individual seeds.
    # Both are legitimate; conflating them is not, so each phrasing is checked against its own source.
    def rng(key, arms_h, agg):
        if agg == "mean":
            v = [mean(cells, a, h, key) for a, h in arms_h]
        else:
            v = [c[key] for c in cells if (c["arm"], c["h"]) in arms_h and not math.isnan(c[key])]
        v = [x for x in v if not math.isnan(x)]
        return min(v), max(v)

    fitted = {(c["arm"], c["h"]) for c in cells if c["n_sup"] > 0}
    for key, unit, tmpls in (
            ("extrap_share", 100, ("Between {lo} and {hi} per cent",
                                   "{lo} to {hi} per cent of the map's effect")),
            ("extrap_median", 1, ("**{lo} to {hi} times**",
                                  "{lo} to {hi} times beyond the fitted range"))):
        mlo, mhi = rng(key, fitted, "mean")
        slo, shi = rng(key, fitted, "seed")
        if key == "extrap_share":
            mlo, mhi = round(mlo * unit), round(mhi * unit)
            slo, shi = round(slo * unit), round(shi * unit)
        else:
            mlo, mhi = math.floor(mlo), math.ceil(mhi)
            slo, shi = math.floor(slo), math.ceil(shi)
        for t in tmpls:
            if t.format(lo=mlo, hi=mhi) not in flat:
                fails.append(f"prose: {key} seed-mean range should read '{t.format(lo=mlo, hi=mhi)}'")
        # and the wider per-seed spread must be disclosed somewhere, not silently dropped
        if f"{slo} to {shi} across individual seeds" not in flat:
            fails.append(f"prose: {key} per-seed spread should be disclosed as "
                         f"'{slo} to {shi} across individual seeds'")
    # h15 on L12 must really have zero rows for the "random projection" claim to stand
    z = [c["n_sup"] for c in cells if c["arm"] == "ebola_L12" and c["h"] == 15]
    if any(z):
        fails.append(f"prose: L12 h15 claimed to have no fitting rows but disk shows {z}")


def check(text):
    data = json.loads(JSON.read_text())
    fails = []
    check_q1(text, data, fails)
    check_q2(text, data, fails)
    check_prose(text, data, fails)
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    text = DOC.read_text(encoding="utf-8")

    if a.mutate:
        bad = 0
        for name, old, new in (("q1 share", "**83.2%** | **5.42x**", "**88.2%** | **5.42x**"),
                               ("q1 rank", "| 48 | 45 |", "| 48 | 51 |"),
                               ("q1 cond", "3.68e05", "3.68e03"),
                               ("q2 per-column", "0, 20, 34]", "0, 20, 44]"),
                               ("q2 blackout", "| **1** | **6** |", "| **1** | **8** |"),
                               ("q2 tail share", "| **27.1%** | **47.8%** |", "| **27.1%** | **41.8%** |"),
                               ("prose range", "Between 76 and 85 per cent", "Between 70 and 85 per cent"),
                               ("prose extrap", "**5 to 11 times**", "**5 to 14 times**")):
            m = text.replace(old, new, 1)
            if m == text:
                print(f"  {name:<15} TARGET NOT FOUND"); bad += 1; continue
            f = check(m)
            print(f"  {name:<15} {'caught: ' + f[0][:62] if f else '*** NOT CAUGHT ***'}")
            bad += 0 if f else 1
        print("\nall mutation targets caught" if not bad else f"\n{bad} BLIND SPOT(S)")
        raise SystemExit(1 if bad else 0)

    fails = check(text)
    if fails:
        print(f"{len(fails)} MISMATCH(ES) between doc and disk:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("ok: 8 Q1 rows x 6 columns, the Q2 per-column profiles and summary table, and the four\n"
          "    recomputed prose ranges all match results/misc/ebolashift__adapter_mechanism.json")


if __name__ == "__main__":
    main()

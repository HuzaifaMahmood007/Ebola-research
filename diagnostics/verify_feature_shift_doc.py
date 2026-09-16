"""Verify progress/outcomes/Ebola_Feature_Shift_2026-09-15.md against the JSON it was written from.

House rule (CLAUDE.md section 7): before signing off a results document, parse the numbers OUT of the
document and recompute them from disk. Same job as diagnostics/verify_anil_doc.py.

It parses the rendered tables and one prose claim, not a machine-readable block, because the rendered
text is what a reader quotes and therefore what can go stale. Tolerances are set just above the
display rounding of each column, so a real staleness (a number from an older run) fails while a
redisplay of the same number does not.

`--mutate` corrupts one cell and asserts the checker catches it, so a checker that silently passes
everything cannot masquerade as a green light.

  conda run -n ebola-train python -m diagnostics.verify_feature_shift_doc
  conda run -n ebola-train python -m diagnostics.verify_feature_shift_doc --mutate
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

DOC = pathlib.Path("progress/outcomes/Ebola_Feature_Shift_2026-09-15.md")
JSON = pathlib.Path("results/misc/ebolashift__feature_shift.json")
COLS = ["n_sup", "rank", "cond", "shift_med", "shift_max", "outside_dims", "pred_gap"]

# tolerance per column, set from how the doc displays it (%.2f -> 0.005, %.1f%% -> 0.05, count -> exact)
ATOL = {"n_sup": 0.5, "rank": 0.5, "shift_med": 6e-3, "shift_max": 6e-3,
        "outside_dims": 0.051, "pred_gap": 6e-3}


def _num(s):
    s = s.replace("*", "").replace("%", "").strip()
    return None if s in ("n/a", "-", "") else float(s)


def doc_rows(text):
    """{(arm, h): {col: value}} from the main results table."""
    out = {}
    for m in re.finditer(r"^\|\s*(L12|L20)\s*\|\s*(\d+)\s*\|(.+)\|\s*$", text, re.M):
        vals = [_num(c) for c in m.group(3).split("|")]
        assert len(vals) == len(COLS), \
            f"row {m.group(1)} h{m.group(2)} has {len(vals)} cells, expected {len(COLS)}"
        out[(f"ebola_{m.group(1)}", int(m.group(2)))] = dict(zip(COLS, vals))
    return out


def doc_windows(text):
    """{'L12 support'|'L20 support'|'query': (pad, observed)} as fractions."""
    out = {}
    for m in re.finditer(r"^\|\s*(L12 support|L20 support|query)[^|]*\|\s*\*{0,2}([\d.]+)%\*{0,2}"
                         r"\s*\|\s*\*{0,2}([\d.]+)%\*{0,2}\s*\|", text, re.M):
        out[m.group(1)] = (float(m.group(2)) / 100, float(m.group(3)) / 100)
    return out


def disk_mean(cells, arm, h, col):
    v = [x[col] for x in cells if x["arm"] == arm and x["h"] == h
         and x[col] is not None and np.isfinite(x[col])]
    return float(np.mean(v)) if v else float("nan")


def check(text, data):
    cells, fails = data["cells"], []

    def cmp(label, said, got, atol):
        blank_said = said is None
        blank_got = got is None or not np.isfinite(got)
        if blank_said and blank_got:
            return
        if blank_said or blank_got:
            fails.append(f"{label}: doc says {said!r}, disk says {got!r}")
        elif abs(said - got) > atol:
            fails.append(f"{label}: doc says {said}, disk says {got:.6g} (tol {atol})")

    rows = doc_rows(text)
    for (arm, h), said in rows.items():
        assert any(x["arm"] == arm and x["h"] == h for x in cells), \
            f"{arm} h{h} is in the doc but not in the JSON"
        for col in COLS:
            got = disk_mean(cells, arm, h, col)
            if col == "cond":
                # spans 1e4 to 1e20; compare exponents, not values
                cmp(f"{arm} h{h} cond(log10)",
                    None if said[col] is None else float(np.log10(said[col])),
                    float(np.log10(got)) if np.isfinite(got) and got > 0 else float("nan"), 0.02)
            else:
                cmp(f"{arm} h{h} {col}", said[col],
                    got * 100 if col == "outside_dims" else got, ATOL[col])

    w = doc_windows(text)
    for key, arm, pk, ok in (("L12 support", "ebola_L12", "support_pad", "support_obs"),
                             ("L20 support", "ebola_L20", "support_pad", "support_obs"),
                             ("query", "ebola_L12", "query_pad", "query_obs")):
        assert key in w, f"window row {key!r} not found in the doc"
        cmp(f"{key} pad", w[key][0], data[arm][pk], 5.1e-4)   # doc quotes pad to 0.1%, not 1%
        cmp(f"{key} observed", w[key][1], data[arm][ok], 5.1e-4)

    # The prose claim a reader is most likely to quote, recomputed rather than trusted.
    ranks = [x["rank"] for x in cells if x["arm"] == ebola_primary(data) and x["n_sup"] > 0]
    lo, hi = 64 - max(ranks), 64 - min(ranks)
    if not re.search(rf"\b{lo} to {hi} of the 64\b", text):
        fails.append(f"prose: unconstrained-directions claim should read '{lo} to {hi} of the 64', "
                     f"not found in the doc")
    return fails, len(rows)


def ebola_primary(data):
    return "ebola_L12"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mutate", action="store_true",
                    help="corrupt one doc number in memory; the check MUST then fail")
    a = ap.parse_args()

    text, data = DOC.read_text(encoding="utf-8"), json.loads(JSON.read_text())
    n_seeds = len({x["seed"] for x in data["cells"]})

    if a.mutate:
        bad = text.replace("| 48 | **45** |", "| 48 | **51** |", 1)
        assert bad != text, "mutation target not found; the table format changed"
        fails, _ = check(bad, data)
        assert fails, "MUTATION TEST FAILED: the checker passed a doc carrying a wrong rank"
        print(f"mutation test ok, checker caught it:\n  {fails[0]}")
        return

    fails, n = check(text, data)
    assert n == 8, f"expected 8 result rows in the doc, parsed {n}"
    if fails:
        print(f"{len(fails)} MISMATCH(ES) between doc and disk:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print(f"ok: {n} result rows x {len(COLS)} columns, 3 window rows and 1 prose claim all match\n"
          f"    {JSON}  (means over {n_seeds} seeds)")


if __name__ == "__main__":
    main()

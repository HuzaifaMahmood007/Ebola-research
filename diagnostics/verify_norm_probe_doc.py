"""Verify progress/outcomes/Norm_Probe_2026-09-16.md against the experiments/ artifacts.

House rule (CLAUDE.md section 7): parse the numbers OUT of the document and recompute them from disk.
Fourth in the series, after verify_feature_shift_doc.py, verify_fewshot_sim_doc.py and
verify_adapter_mechanism_doc.py.

Sources:
  EXP 2 table + counts -> experiments/norm_probe__seed*.json (recomputed, not read from the summary)
  EXP 1 table          -> experiments/norm_probe__seed42.json
  parity table         -> recomputed live from the bundles via experiments.norm_probe.devnorm_scaler
  prose counts         -> recomputed from the 40 paired cells

The parity table is recomputed from the data rather than from a stored number on purpose: it is the
caveat that bounds every EXP 2 claim, so it must not be able to go stale silently.

  conda run -n ebola-train python -m diagnostics.verify_norm_probe_doc
  conda run -n ebola-train python -m diagnostics.verify_norm_probe_doc --mutate
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob
import json
import math
import pathlib
import re

import numpy as np

DOC = pathlib.Path("progress/outcomes/Norm_Probe_2026-09-16.md")
ARMS = ("ebola_L12", "ebola_L20")
HS = (3, 5, 10, 15)


def flatten(text):
    """Doc as one line, blockquote markers dropped, so a claim wrapped across lines still matches."""
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*>\s?", "", text))


def _num(s):
    s = s.replace("*", "").replace("%", "").replace("+", "").strip()
    return None if s in ("n/a", "-", "") else float(s)


def exp2_rows():
    rows = []
    for f in sorted(glob.glob("experiments/norm_probe__seed*.json")):
        if "summary" in f:
            continue
        rows += [r for r in json.loads(pathlib.Path(f).read_text())["rows"] if r["exp"] == "exp2"]
    assert rows, "no exp2 rows on disk"
    return rows


def check_exp2(text, rows, fails):
    n_seeds = len({r["seed"] for r in rows})
    seen = 0
    for m in re.finditer(r"^\|\s*(L12|L20)\s*\|\s*(\d+)\s*\|(.+)\|\s*$", text, re.M):
        arm, h = f"ebola_{m.group(1)}", int(m.group(2))
        cells = [c.strip() for c in m.group(3).split("|")]
        if len(cells) != 8:
            continue                                   # not the EXP 2 table
        seen += 1
        c = [r for r in rows if r["arm"] == arm and r["h"] == h]
        if len(c) != n_seeds:
            fails.append(f"{arm} h{h}: {len(c)} seeds on disk, expected {n_seeds}")
            continue
        for key, said in zip(("pooled_zero", "pooled_few", "devnorm_zero", "devnorm_few"), cells[:4]):
            got = float(np.mean([r[key] for r in c]))
            v = _num(said)
            if v is None or abs(v - got) > 0.006:
                fails.append(f"{arm} h{h} {key}: doc {said!r}, disk {got:.4f}")
        counts = {"df<pz": ("devnorm_few", "pooled_zero"), "df<dz": ("devnorm_few", "devnorm_zero"),
                  "dz<pz": ("devnorm_zero", "pooled_zero")}
        for lbl, said in zip(("df<pz", "df<dz", "dz<pz"), cells[4:7]):
            a, b = counts[lbl]
            got = sum(1 for r in c if r[a] < r[b])
            mm = re.match(r"\*{0,2}(\d+)/(\d+)\*{0,2}$", said.strip())
            if not mm or int(mm.group(1)) != got or int(mm.group(2)) != n_seeds:
                fails.append(f"{arm} h{h} {lbl}: doc {said!r}, disk {got}/{n_seeds}")
        d = [(r["devnorm_few"] - r["pooled_zero"]) / r["pooled_zero"] * 100 for r in c]
        mm = re.search(r"([+-]?[\d.]+)%\s*\[\s*([+-][\d.]+)\s*,\s*([+-][\d.]+)\s*\]", cells[7])
        if not mm:
            fails.append(f"{arm} h{h}: could not parse the delta cell {cells[7]!r}")
        else:
            for lbl, said, got in (("mean", float(mm.group(1)), float(np.mean(d))),
                                   ("min", float(mm.group(2)), float(np.min(d))),
                                   ("max", float(mm.group(3)), float(np.max(d)))):
                if abs(said - got) > 0.06:
                    fails.append(f"{arm} h{h} delta {lbl}: doc {said}, disk {got:.2f}")
    if seen != 8:
        fails.append(f"parsed {seen} EXP 2 rows, expected 8")


def check_exp1(text, fails):
    rows = [r for r in json.loads(pathlib.Path("experiments/norm_probe__seed42.json").read_text())["rows"]
            if r["exp"] == "exp1"]
    by = {(r["panel"], r["h"]): r["cost_pct"] for r in rows}
    seen = 0
    for m in re.finditer(r"^\|\s*`([\w-]+)`\s*\|\s*\+?([\d.]+)%\s*\|\s*\+?([\d.]+)%\s*\|"
                         r"\s*\+?([\d.]+)%\s*\|\s*\+?([\d.]+)%\s*\|", text, re.M):
        panel = m.group(1)
        if (panel, 3) not in by:
            continue
        seen += 1
        for h, said in zip(HS, [float(m.group(i)) for i in (2, 3, 4, 5)]):
            got = by[(panel, h)]
            if abs(said - got) > 0.06:
                fails.append(f"exp1 {panel} h{h}: doc {said}, disk {got:.2f}")
    if seen != 4:
        fails.append(f"parsed {seen} EXP 1 panel rows, expected 4")
    lo, hi = min(by.values()), max(by.values())
    flat = flatten(text).replace("*", "")
    # EVERY occurrence must be right, not just one. A "is the correct string present?" check passes a
    # document where the claim appears twice and only one copy was corrupted -- the mutation sweep
    # caught exactly that on three separate claims.
    rng = re.findall(r"\+([\d.]+)% to \+([\d.]+)%", flat)
    if not rng:
        fails.append(f"exp1 prose range '+{lo:.1f}% to +{hi:.1f}%' not found at all")
    for a, b in rng:
        if abs(float(a) - lo) > 0.06 or abs(float(b) - hi) > 0.06:
            fails.append(f"exp1 prose range: doc '+{a}% to +{b}%', disk '+{lo:.1f}% to +{hi:.1f}%'")
    cw = re.findall(r"(\d+) of (\d+) cells worse", flat)
    if not cw:
        fails.append(f"exp1 prose should say '{len(rows)} of {len(rows)} cells worse'")
    for a, b in cw:
        if (int(a), int(b)) != (len(rows), len(rows)):
            fails.append(f"exp1 'cells worse': doc {a} of {b}, disk {len(rows)} of {len(rows)}")


def check_parity(text, fails):
    """Recompute the partial-parity table live from the bundles."""
    import bundles
    from experiments.norm_probe import devnorm_scaler, rescale
    flat = flatten(text)
    cent = {}
    for arm, col in (("ebola_L12", 0), ("ebola_L20", 1)):
        b = bundles.load(arm)
        s = b.masks()["support"].astype(bool)
        sc, zn = devnorm_scaler(b)
        own = sum(1 for i in range(b.M.shape[0])
                  if s[i].any() and np.log1p(b.raw[i][s[i]].astype(np.float64)).std() >= 1e-8)
        distinct = len(set(zip(np.round(sc["mean"], 6), np.round(sc["std"], 6))))
        rescale(b, sc, zero_nodes=zn)
        d = float(np.mean([abs(b.X[..., 0][i][s[i]].mean()) for i in range(b.M.shape[0]) if s[i].any()]))
        for label, val in (("with any support cell", int(s.any(1).sum())),
                           ("getting a genuine per-node scale", own),
                           ("with no support cell, pinned to 0.0 input", len(zn)),
                           ("distinct scales across 61 districts", distinct)):
            row = re.search(rf"\| {re.escape(label)} \| \*{{0,2}}(\d+)\*{{0,2}} \| \*{{0,2}}(\d+)\*{{0,2}} \|", flat)
            if not row:
                fails.append(f"parity row {label!r} not found")
            elif int(row.group(col + 1)) != val:
                fails.append(f"parity {label!r} {arm}: doc {row.group(col+1)}, disk {val}")
        cent[arm] = d
    # The centring row carries both arms, and "0.52" also appears in prose, so a bare-number check
    # passes when only the table cell is wrong. Validate the row itself, both columns.
    row = re.search(r"\| support districts' mean distance from zero \| ([\d.]+) → \*{0,2}([\d.]+)\*{0,2}"
                    r" \| ([\d.]+) → \*{0,2}([\d.]+)\*{0,2} \|", flat)
    if not row:
        fails.append("parity: the 'mean distance from zero' row was not found")
    else:
        for i, arm in enumerate(ARMS):
            said = float(row.group(2 + i * 2))
            if abs(said - cent[arm]) > 0.006:
                fails.append(f"parity centring {arm}: doc {said}, disk {cent[arm]:.4f}")


def check_prose(text, rows, fails):
    # Prose checks run on a bold-stripped copy. Markdown lets emphasis fall either side of a number
    # ("only **14 times**" vs "only **14** times") and a literal match then fails on a document whose
    # number is correct. The tables above keep their asterisks, where the bolding is load-bearing.
    flat = flatten(text).replace("*", "")
    n = len(rows)
    dz = sum(1 for r in rows if r["devnorm_zero"] < r["pooled_zero"])
    df = sum(1 for r in rows if r["devnorm_few"] < r["pooled_zero"])
    dfpf = sum(1 for r in rows if r["devnorm_few"] < r["pooled_few"])
    # every occurrence, not just the first (see the note in check_exp1)
    pc = re.findall(r"(\d+) of (\d+) paired cells", flat)
    if not pc:
        fails.append(f"prose: dz<pz should read '{dz} of {n} paired cells'")
    for a, b in pc:
        if (int(a), int(b)) != (dz, n):
            fails.append(f"prose 'of {b} paired cells': doc {a}, disk {dz}")
    if f"only {df} of {n} cells" not in flat:
        fails.append(f"prose: df<pz should read 'only {df} of {n} cells'")
    if f"only {dfpf} times" not in flat:
        fails.append(f"prose: devnorm_few<pooled_few should read 'only {dfpf} times'")
    # the L20 h10 unanimous cell and the spread figures
    c = [r for r in rows if r["arm"] == "ebola_L20" and r["h"] == 10]
    if sum(1 for r in c if r["devnorm_few"] < r["pooled_few"]) != len(c):
        fails.append("prose claims L20 h10 is unanimous (5 of 5) but disk disagrees")
    for h, key, lo_hi in ((3, "pooled_few", None), (3, "devnorm_few", None)):
        c3 = [r[key] for r in rows if r["arm"] == "ebola_L20" and r["h"] == h]
        for v in (min(c3), max(c3)):
            if f"{v:.2f}" not in flat:
                fails.append(f"prose: L20 h3 {key} bound {v:.2f} not found in the doc")


def check(text):
    rows = exp2_rows()
    fails = []
    check_exp2(text, rows, fails)
    check_exp1(text, fails)
    check_parity(text, fails)
    check_prose(text, rows, fails)
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    text = DOC.read_text(encoding="utf-8")

    if a.mutate:
        bad = 0
        for name, old, new in (("exp2 mean", "| 22.85 |", "| 22.55 |"),
                               ("exp2 count", "| **0/5** | 0/5 | **0/5** |", "| **1/5** | 0/5 | **0/5** |"),
                               ("exp2 delta", "+15.9% [+12.2, +19.2]", "+15.9% [+13.2, +19.2]"),
                               ("exp1 cell", "| +28.5% |", "| +25.8% |"),
                               ("exp1 range", "+4.6% to +101.5%", "+4.6% to +110.5%"),
                               ("parity pinned", "| **43** | 25 |", "| **41** | 25 |"),
                               ("parity centring", "0.70 → **0.52**", "0.70 → **0.42**"),
                               ("prose 0 of 40", "0 of 40 paired cells", "2 of 40 paired cells"),
                               ("prose 14", "only **14 times**", "only **19 times**")):
            m = text.replace(old, new, 1)
            if m == text:
                print(f"  {name:<17} TARGET NOT FOUND"); bad += 1; continue
            f = check(m)
            print(f"  {name:<17} {'caught: ' + f[0][:58] if f else '*** NOT CAUGHT ***'}")
            bad += 0 if f else 1
        print("\nall mutation targets caught" if not bad else f"\n{bad} BLIND SPOT(S)")
        raise SystemExit(1 if bad else 0)

    fails = check(text)
    if fails:
        print(f"{len(fails)} MISMATCH(ES) between doc and disk:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("ok: 8 EXP 2 rows (means, three sign counts, mean/min/max deltas), 4 EXP 1 panel rows and\n"
          "    its prose range, the parity table recomputed live from the bundles, and the paired-cell\n"
          "    prose counts all match experiments/norm_probe__seed*.json")


if __name__ == "__main__":
    main()

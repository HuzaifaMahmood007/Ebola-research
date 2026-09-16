"""Verify progress/outcomes/Fewshot_Init_Probe_2026-09-15.md against the artifacts on disk.

House rule (CLAUDE.md section 7): parse the numbers OUT of the document and recompute them from disk
before signing it off. Companion to diagnostics/verify_feature_shift_doc.py.

Four independent sources are checked, because the document leans on four:
  results table   -> results/misc/fewshotsim__summary__seed42.json
  diagnosis table -> results/lodo/encoder_ldo3{,_zeroshot}__*__seed*.json  (the 4-of-100 count)
  covid reference -> results/lodo/encoder_ldo3full{,_zeroshot}__covid_us-states__seed42.json
  Step 0 recap    -> results/misc/ebolashift__feature_shift.json

  conda run -n ebola-train python -m diagnostics.verify_fewshot_sim_doc
  conda run -n ebola-train python -m diagnostics.verify_fewshot_sim_doc --mutate
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import re

import numpy as np

DOC = pathlib.Path("progress/outcomes/Fewshot_Init_Probe_2026-09-15.md")
SUMMARY = pathlib.Path("results/misc/fewshotsim__summary__seed42.json")
SHIFT = pathlib.Path("results/misc/ebolashift__feature_shift.json")
ARMS = ["zeroshot", "fresh", "freshzs15", "warm"]


def _num(s):
    s = s.replace("*", "").replace("%", "").replace("+", "").strip()
    return None if s in ("-", "", "n/a") else float(s)


def mae_by_h(path):
    return {r["horizon"]: r["country_macro"]
            for r in json.loads(pathlib.Path(path).read_text()) if r["metric"] == "mae"}


def check_results_table(text, fails):
    rows = {(r["template"], r["placement"]): r for r in json.loads(SUMMARY.read_text())
            if r["arm"] == "zeroshot"}
    by = {(r["template"], r["placement"], r["arm"]): r["mae"]
          for r in json.loads(SUMMARY.read_text())}
    n = 0
    for m in re.finditer(r"^\|\s*(L12|L20)\s*\|\s*`([\w-]+)`\s*\|(.+)\|\s*$", text, re.M):
        tmpl, place = m.group(1), m.group(2)
        cells = [c.strip() for c in m.group(3).split("|")]
        if len(cells) != 6:
            fails.append(f"{tmpl} {place}: {len(cells)} cells, expected 6")
            continue
        assert (tmpl, place) in rows, f"{tmpl} {place} in doc but not in the summary"
        n += 1
        for arm, said in zip(ARMS, cells[:4]):
            got = by.get((tmpl, place, arm))
            said = _num(said)
            if said is None and got is None:
                continue
            if said is None or got is None:
                fails.append(f"{tmpl} {place} {arm}: doc {said!r}, disk {got!r}")
            elif abs(said - got) > 0.05:
                fails.append(f"{tmpl} {place} {arm}: doc {said}, disk {got:.4f}")
        # the derived warm-vs-fresh percentage must follow from the two columns, not be typed
        fr, wm = by.get((tmpl, place, "fresh")), by.get((tmpl, place, "warm"))
        said_d = _num(cells[4])
        got_d = (wm - fr) / fr * 100
        if said_d is None or abs(said_d - got_d) > 0.05:
            fails.append(f"{tmpl} {place} warm-vs-fresh: doc {said_d}, recomputed {got_d:.2f}")
        # and the adaptation verdict must agree with fresh vs zeroshot
        zs = by[(tmpl, place, "zeroshot")]
        word = cells[5].replace("*", "").strip()
        if word == "helps" and not fr < zs:
            fails.append(f"{tmpl} {place}: doc says adaptation helps but fresh {fr:.1f} >= zs {zs:.1f}")
        if word == "hurts" and not fr > zs:
            fails.append(f"{tmpl} {place}: doc says adaptation hurts but fresh {fr:.1f} <= zs {zs:.1f}")
    if n != 8:
        fails.append(f"parsed {n} result rows, expected 8")
    return n


def zeroshot_win_counts():
    """{panel: [wins per seed]} recomputed from the archived LDO3 records."""
    res = collections.defaultdict(dict)
    for pat, key in (("results/lodo/encoder_ldo3__*__seed*.json", "ad"),
                     ("results/lodo/encoder_ldo3_zeroshot__*__seed*.json", "zs")):
        for p in glob.glob(pat):
            parts = os.path.basename(p).split("__")
            res[(parts[1], parts[2].split(".")[0])][key] = mae_by_h(p)
    agg = collections.defaultdict(list)
    for (panel, _), d in sorted(res.items()):
        if "ad" in d and "zs" in d:
            agg[panel].append(sum(1 for h in (3, 5, 10, 15) if d["zs"][h] < d["ad"][h]))
    return agg


def check_diagnosis_table(text, flat, fails):
    agg = zeroshot_win_counts()
    n, tot_said = 0, None
    for m in re.finditer(r"^\|\s*`([\w-]+)`\s*\|\s*(\d+)\s*\|\s*([\d,\s]+?)\s*\|\s*(\d+)\s*/\s*(\d+)\s*\|",
                         text, re.M):
        panel, seeds = m.group(1), int(m.group(2))
        said_list = [int(x) for x in m.group(3).split(",")]
        said_tot, said_den = int(m.group(4)), int(m.group(5))
        if panel not in agg:
            fails.append(f"diagnosis: panel {panel} in doc but no records on disk")
            continue
        n += 1
        got = agg[panel]
        if seeds != len(got):
            fails.append(f"{panel}: doc says {seeds} seeds, disk has {len(got)}")
        if said_list != got:
            fails.append(f"{panel}: doc per-seed {said_list}, disk {got}")
        if said_tot != sum(got) or said_den != 4 * len(got):
            fails.append(f"{panel}: doc total {said_tot}/{said_den}, disk {sum(got)}/{4*len(got)}")
    if n != 5:
        fails.append(f"parsed {n} diagnosis rows, expected 5")
    grand = sum(sum(v) for v in agg.values()), sum(4 * len(v) for v in agg.values())
    if not re.search(rf"\*\*{grand[0]} / {grand[1]}\*\*", flat):
        fails.append(f"diagnosis grand total should read '{grand[0]} / {grand[1]}'")
    if not re.search(rf"wins \*\*{grand[1]-grand[0]} of {grand[1]}\*\* cells", flat):
        fails.append(f"prose should say adaptation wins {grand[1]-grand[0]} of {grand[1]} cells")


def check_reference(text, fails):
    ad = mae_by_h("results/lodo/encoder_ldo3full__covid_us-states__seed42.json")
    zs = mae_by_h("results/lodo/encoder_ldo3full_zeroshot__covid_us-states__seed42.json")
    for label, d in (("adapted", ad), ("zero-shot", zs)):
        want = " / ".join(f"{d[h]:.1f}" for h in (3, 5, 10, 15))
        if want not in text:
            fails.append(f"reference {label} series should read '{want}'")


def check_step0_recap(text, fails):
    d = json.loads(SHIFT.read_text())
    a = d["ebola_L12"]
    # Match the FULL phrase, not the bare number. "72.5%" also appears in the design table, so a
    # number-only check passes even when the recap sentence has been corrupted -- the mutation sweep
    # caught exactly that.
    for frag in (f"{a['support_pad']*100:.1f}% zero padding at {a['support_obs']*100:.1f}% observation",
                 f"full windows at {a['query_obs']*100:.1f}%"):
        if frag not in text:
            fails.append(f"Step 0 recap: '{frag}' not found in the doc")
    ranks = [int(np.mean([c["rank"] for c in d["cells"] if c["arm"] == "ebola_L12" and c["h"] == h]))
             for h in (3, 5, 10, 15)]
    want = ", ".join(str(r) for r in ranks[:3]) + f" and {ranks[3]} of 64"
    if want not in text:
        fails.append(f"Step 0 recap: rank series should read '{want}'")


def check(text):
    fails = []
    # Tables need the raw text (line-anchored regexes); prose phrases need whitespace collapsed,
    # because markdown wraps them across lines and a literal `in text` then silently fails.
    flat = re.sub(r"\s+", " ", text)
    check_results_table(text, fails)
    check_diagnosis_table(text, flat, fails)
    check_reference(flat, fails)
    check_step0_recap(flat, fails)
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    text = DOC.read_text(encoding="utf-8")

    if a.mutate:
        n_bad = 0
        for name, old, new in (("results cell", "| 12402.5 | **7829.5** |", "| 12402.5 | **7529.5** |"),
                               ("derived pct", "+37.1%", "+31.7%"),
                               ("diagnosis row", "| 0, 1, 0, 0, 0 |", "| 0, 2, 0, 0, 0 |"),
                               ("grand total", "**4 / 100**", "**7 / 100**"),
                               ("reference", "5931.3 / 12450.0", "5931.3 / 12480.0"),
                               ("step0 recap", "72.5% zero padding", "76.5% zero padding")):
            m = text.replace(old, new, 1)
            if m == text:
                print(f"  {name:<15} TARGET NOT FOUND"); n_bad += 1; continue
            f = check(m)
            print(f"  {name:<15} {'caught: ' + f[0][:60] if f else '*** NOT CAUGHT ***'}")
            n_bad += 0 if f else 1
        print("\nall mutation targets caught" if not n_bad else f"\n{n_bad} BLIND SPOT(S)")
        raise SystemExit(1 if n_bad else 0)

    fails = check(text)
    if fails:
        print(f"{len(fails)} MISMATCH(ES) between doc and disk:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("ok: 8 result rows (values, derived percentages and verdict words), 5 diagnosis rows plus\n"
          "    grand total, the covid per-horizon reference series, and the Step 0 recap all match disk")


if __name__ == "__main__":
    main()

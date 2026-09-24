"""Verify progress/outcomes/Adapter_Constraint_2026-09-22.md against the experiments/ artifacts.

House rule (CLAUDE.md section 7): parse the numbers OUT of the document and recompute them from disk,
prose counts and the verdict-bearing aggregates included, not just the table. Then a --mutate sweep
plants a defect in each load-bearing number and confirms the check catches it.

Sources, all recomputed live, never read from the summary:
  MAE table (few/zero/constr/clip/oracle/gap/recovery) -> experiments/adapter_constraint__seed*.json
  anchors (t=1 == archived few, t=0 == archived zero)  -> results/ebola/*.json, read-only
  selected-t lists, span ranks, aggregate recoveries    -> the same seed JSONs

  conda run -n ebola-train python -m diagnostics.verify_adapter_constraint_doc
  conda run -n ebola-train python -m diagnostics.verify_adapter_constraint_doc --mutate
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re

import numpy as np

DOC = pathlib.Path("progress/outcomes/Adapter_Constraint_2026-09-22.md")
ARMS = ("ebola_L12", "ebola_L20")
HS = (3, 5, 10, 15)
SEEDS = (42, 52, 62, 72, 82)
TGRID = [round(x, 3) for x in np.linspace(0, 1, 11)]
DAMAGE = 0.5          # gap threshold (MAE) for "a cell with real damage to recover"


def flatten(text):
    return re.sub(r"\s+", " ", text)


def rows_on_disk():
    rows = []
    for f in sorted(glob.glob("experiments/adapter_constraint__seed*.json")):
        if "summary" in f:
            continue
        d = json.loads(pathlib.Path(f).read_text())
        assert d["protocol"] == "EXPLORATORY", f"{f} not EXPLORATORY"
        rows += d["rows"]
    return rows


def archived_mae(fam, arm, seed):
    recs = json.loads(pathlib.Path(f"results/ebola/{fam}__{arm}__seed{seed}.json").read_text())
    return {r["horizon"]: r["country_macro"] for r in recs if r["metric"] == "mae"}


def sel_mae(r, h, metric="mae"):
    return next(g[metric][str(h)] for g in r["grid"] if g["t"] == r["selected_t"])


def oracle_mae(r, h):
    return min(g["mae"][str(h)] for g in r["grid"])


def cell(rows, arm, h):
    rs = [r for r in rows if r["arm"] == arm]
    few = float(np.mean([r["few_mae"][str(h)] for r in rs]))
    zero = float(np.mean([r["zero_mae"][str(h)] for r in rs]))
    con = float(np.mean([sel_mae(r, h) for r in rs]))
    clip = float(np.mean([r["clip_mae"][str(h)] for r in rs]))
    orac = float(np.mean([oracle_mae(r, h) for r in rs]))
    gap = few - zero
    # recovery is defined only where there is real damage to recover (gap > DAMAGE). A near-zero or
    # negative gap makes the ratio meaningless, and the doc marks exactly those cells n/a.
    defined = gap > DAMAGE
    rec = (few - con) / gap if defined else None
    rc = (few - clip) / gap if defined else None
    ro = (few - orac) / gap if defined else None
    return dict(few=few, zero=zero, con=con, clip=clip, orac=orac, gap=gap, rec=rec, rc=rc, ro=ro)


def _num(s):
    s = s.replace("*", "").replace("%", "").replace("+", "").strip()
    return None if s in ("n/a", "-", "") else float(s)


def _pct(s):
    s = s.replace("*", "").strip()
    if s in ("n/a", "-", ""):
        return None
    return float(s.replace("%", "").replace("+", ""))


def check(text):
    rows = rows_on_disk()
    fails = []

    # completeness
    seeds = sorted({r["seed"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    if seeds != list(SEEDS):
        fails.append(f"seeds on disk {seeds} != {list(SEEDS)}")
    if set(arms) != set(ARMS):
        fails.append(f"arms on disk {arms} != {list(ARMS)}")
    if len(rows) != 10:
        fails.append(f"{len(rows)} arm-seed rows on disk, expected 10")
    for r in rows:
        if [g["t"] for g in r["grid"]] != TGRID:
            fails.append(f"{r['arm']} seed{r['seed']}: t-grid is not the 11-point grid")

    # anchors: t=1 == archived few, t=0 == archived zero
    maxd = 0.0
    for r in rows:
        g1 = next(g for g in r["grid"] if g["t"] == 1.0)
        g0 = next(g for g in r["grid"] if g["t"] == 0.0)
        af = archived_mae("encoder_ebola", r["arm"], r["seed"])
        az = archived_mae("encoder_ebola_zeroshot", r["arm"], r["seed"])
        for h in HS:
            maxd = max(maxd, abs(g1["mae"][str(h)] - af[h]), abs(g0["mae"][str(h)] - az[h]),
                       abs(r["few_mae"][str(h)] - af[h]), abs(r["zero_mae"][str(h)] - az[h]))
    if maxd >= 1e-3:
        fails.append(f"anchor deviation {maxd:.2e} >= 1e-3 (t=1 vs few / t=0 vs zero)")

    # MAE table, every cell
    seen = 0
    pat = (r"^\|\s*(L12|L20)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
           r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([+\-][\d.]+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$")
    for m in re.finditer(pat, text, re.M):
        arm = f"ebola_{m.group(1)}"; h = int(m.group(2))
        seen += 1
        c = cell(rows, arm, h)
        for key, said in (("few", m.group(3)), ("zero", m.group(4)), ("con", m.group(5)),
                          ("clip", m.group(6)), ("orac", m.group(7))):
            v = _num(said)
            if v is None or abs(v - c[key]) > 0.006:
                fails.append(f"{arm} h{h} {key}: doc {said!r}, disk {c[key]:.4f}")
        gd = _num(m.group(8))
        if gd is None or abs(gd - c["gap"]) > 0.006:
            fails.append(f"{arm} h{h} gap: doc {m.group(8)!r}, disk {c['gap']:+.4f}")
        for lbl, said, got in (("recovery", m.group(9), c["rec"]),
                               ("recovery_clip", m.group(10), c["rc"]),
                               ("oracle", m.group(11), c["ro"])):
            sv = _pct(said)
            if got is None:                       # gap ~ 0, doc must say n/a
                if sv is not None:
                    fails.append(f"{arm} h{h} {lbl}: doc {said!r} but disk gap~0 (should be n/a)")
            elif sv is None or abs(sv - got * 100) > 0.6:
                fails.append(f"{arm} h{h} {lbl}: doc {said!r}, disk {got * 100:+.1f}%")
    if seen != 8:
        fails.append(f"parsed {seen} MAE table rows, expected 8")

    # selected-t lists, mean and median, per arm
    flat = flatten(text)
    for arm, tag in ((("ebola_L12"), "L12"), (("ebola_L20"), "L20")):
        ts = [r["selected_t"] for r in sorted([x for x in rows if x["arm"] == arm], key=lambda r: r["seed"])]
        mm = re.search(rf"{tag}: \[([\d., ]+)\], mean (\d+\.\d+), median (\d+\.\d+)", flat)
        if not mm:
            fails.append(f"selected-t line for {tag} not found")
            continue
        doc_list = [float(x) for x in mm.group(1).split(",")]
        if doc_list != ts:
            fails.append(f"{tag} selected-t: doc {doc_list}, disk {ts}")
        if abs(float(mm.group(2)) - float(np.mean(ts))) > 0.005:
            fails.append(f"{tag} selected-t mean: doc {mm.group(2)}, disk {np.mean(ts):.3f}")
        if abs(float(mm.group(3)) - float(np.median(ts))) > 0.005:
            fails.append(f"{tag} selected-t median: doc {mm.group(3)}, disk {np.median(ts):.3f}")

    # aggregate recoveries over the damaged cells (gap > 0.5)
    dmg = [(arm, h) for arm in ARMS for h in HS if cell(rows, arm, h)["gap"] > DAMAGE]
    rec_all = [cell(rows, a, h)["rec"] * 100 for a, h in dmg]
    rec_l12 = [cell(rows, "ebola_L12", h)["rec"] * 100 for h in HS if ("ebola_L12", h) in dmg]
    rec_l20 = [cell(rows, "ebola_L20", h)["rec"] * 100 for h in HS if ("ebola_L20", h) in dmg]
    clip_all = [cell(rows, a, h)["rc"] * 100 for a, h in dmg]
    for label, disk, pat_s in (
        ("avg", np.mean(rec_all), r"\+(\d+) per cent of the damage on average"),
        ("L12 mean", np.mean(rec_l12), r"\+(\d+) per cent mean on L12"),
        ("L20 mean", np.mean(rec_l20), r"\+(\d+) per cent mean on L20"),
        ("clip avg", np.mean(clip_all), r"almost nothing \(\+(\d+) per cent"),
    ):
        hits = re.findall(pat_s, flat)
        if not hits:
            fails.append(f"aggregate '{label}' pattern not found (disk {disk:+.1f}%)")
        for hnum in hits:
            if abs(float(hnum) - disk) > 0.6:
                fails.append(f"aggregate '{label}': doc {hnum}, disk {disk:.1f}")
    if len(dmg) != 7:
        fails.append(f"{len(dmg)} damaged cells (gap>{DAMAGE}) on disk, doc says 7")

    # span ranks, both arms, from disk
    r0 = {arm: next(r for r in rows if r["arm"] == arm)["span_rank"] for arm in ARMS}
    for tag, arm in (("L12", "ebola_L12"), ("L20", "ebola_L20")):
        want = r0[arm]
        mm = re.search(rf"{tag} \{{h3:(\d+), h5:(\d+), h10:(\d+), h15:(\d+)\}}", flat)
        if not mm:
            fails.append(f"span-rank line for {tag} not found")
            continue
        for i, h in enumerate(HS):
            if int(mm.group(i + 1)) != want[str(h)]:
                fails.append(f"{tag} span rank h{h}: doc {mm.group(i + 1)}, disk {want[str(h)]}")

    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    text = DOC.read_text(encoding="utf-8")

    if a.mutate:
        bad = 0
        for name, old, new in (
            ("mae few cell", "| L12 | 3 | 24.69 |", "| L12 | 3 | 24.99 |"),
            ("mae constr cell", "| 26.05 | 23.55 | 23.81 |", "| 26.05 | 23.55 | 24.81 |"),
            ("recovery cell", "+2.50 | **+90%** |", "+2.50 | **+70%** |"),
            ("recovery clip cell", "+10.28 | **+9%** | +0% |", "+10.28 | **+9%** | +20% |"),
            ("gap cell", "| +10.74 |", "| +9.74 |"),
            ("selected-t list", "L20: [0.8, 0.8, 1.0, 0.9, 0.9]", "L20: [0.8, 0.8, 0.8, 0.9, 0.9]"),
            ("selected-t mean", "mean 0.34, median 0.40", "mean 0.44, median 0.40"),
            ("aggregate avg", "+45 per cent of the damage on average", "+55 per cent of the damage on average"),
            ("aggregate L12", "+84 per cent mean on L12", "+74 per cent mean on L12"),
            ("clip aggregate", "almost nothing (+1 per cent", "almost nothing (+11 per cent"),
            ("span rank", "L12 {h3:45, h5:37, h10:17, h15:0}", "L12 {h3:45, h5:37, h10:27, h15:0}"),
        ):
            m = text.replace(old, new, 1)
            if m == text:
                print(f"  {name:<20} TARGET NOT FOUND"); bad += 1; continue
            f = check(m)
            print(f"  {name:<20} {'caught: ' + f[0][:56] if f else '*** NOT CAUGHT ***'}")
            bad += 0 if f else 1
        print("\nall mutation targets caught" if not bad else f"\n{bad} BLIND SPOT(S)")
        raise SystemExit(1 if bad else 0)

    fails = check(text)
    if fails:
        print(f"{len(fails)} MISMATCH(ES) between doc and disk:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("ok: 8 MAE table rows (few/zero/constr/clip/oracle/gap and three recovery columns), the\n"
          "    t=1/t=0 anchors against the archived arms, both selected-t lists with mean and median,\n"
          "    the four aggregate recoveries, the 7 damaged-cell count and the span ranks all match\n"
          "    experiments/adapter_constraint__seed*.json")


if __name__ == "__main__":
    main()

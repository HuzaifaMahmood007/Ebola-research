"""Verify progress/outcomes/Disease_Mean_Floor_2026-09-21.md against disk.

Parses the numbers OUT of the document and recomputes them, rather than trusting the run that
wrote it. Checks the four result tables cell by cell, the prose claims that carry a number, the
ordering claim that disease_mean is a WEAKER floor than support_mean, and the win/loss counts.

Blockquote-aware: claims wrapped across lines of a `>` block flatten correctly (the same trap that
produced a false failure in verify_adapter_mechanism_doc.py).

Run:  conda run -n ebola-train python -m diagnostics.verify_disease_mean_doc
      conda run -n ebola-train python -m diagnostics.verify_disease_mean_doc --doc path/to.md
"""
import argparse
import json
import re
import sys

import numpy as np

import bundles
from bundles import HORIZONS
from experiments.disease_mean_floor import disease_means, floor_preds, macro, model_macro
from train.loop import score_predictions

DOC = "progress/outcomes/Disease_Mean_Floor_2026-09-21.md"
ARMS = ("ebola_L12", "ebola_L20")
TOL = 5e-4          # the doc prints 3 decimals
PCT_TOL = 0.06      # the doc prints 1 decimal on percentages


def flatten(text):
    """Doc text as one line, with markdown blockquote markers dropped.

    A claim wrapped across two lines of a `>` block otherwise keeps its `>` mid-phrase once
    whitespace is collapsed, and a contiguous-string search misses a correct document."""
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*>\s?", "", text))


def table_rows(text, header_marker, labels):
    """{label: [h3,h5,h10,h15]} from the markdown table following header_marker."""
    seg = text.split(header_marker, 1)[1]
    out = {}
    for line in seg.splitlines():
        if not line.strip().startswith("|"):
            if out:
                break
            continue
        cells = [c.strip().strip("*`") for c in line.strip().strip("|").split("|")]
        if cells[0] in labels and cells[0] not in out:
            try:
                out[cells[0]] = [float(c) for c in cells[1:5]]
            except ValueError:
                continue
    return out


def recompute():
    """{arm: {metric: {series: {h: value}}}} straight from the bundles and archives."""
    truth = {}
    for arm in ARMS:
        b = bundles.load(arm)
        te = b.origins(phase="query")
        recs = []
        for m, preds in floor_preds(b, te).items():
            recs += score_predictions(m, arm, None, preds, b, te, phase="query")[0]
        with open("results/naive/naive__%s.json" % arm) as fh:
            ref = json.load(fh)
        truth[arm] = {}
        for metric in ("rmse", "mae"):
            truth[arm][metric] = {
                "persistence": macro(ref, "persistence", metric),
                "support_mean": macro(ref, "support_mean", metric),
                "disease_mean": macro(recs, "disease_mean", metric),
                "disease_mean_pernode": macro(recs, "disease_mean_pernode", metric),
                "MODEL zero-shot": model_macro(arm, "encoder_ebola_zeroshot", metric),
                "MODEL few-shot": model_macro(arm, "encoder_ebola", metric),
            }
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=DOC)
    a = ap.parse_args()
    with open(a.doc, encoding="utf-8") as fh:
        text = fh.read()
    flat = flatten(text)
    truth = recompute()
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    # --- 1. the two floor tables, cell by cell -----------------------------------------------
    labels = {"persistence", "support_mean", "disease_mean", "disease_mean_pernode",
              "MODEL zero-shot", "MODEL few-shot"}
    for metric, marker in (("rmse", "`ebola_L12`, RMSE:"), ("mae", "`ebola_L12`, MAE:")):
        rows = table_rows(text, marker, labels)
        chk(len(rows) == 6, "%s table: found %d of 6 rows" % (metric, len(rows)))
        for name, vals in rows.items():
            for h, v in zip(HORIZONS, vals):
                want = truth["ebola_L12"][metric][name][h]
                chk(abs(v - want) <= TOL,
                    "L12 %s %s h%d: doc %.3f disk %.3f" % (metric, name, h, v, want))

    # --- 2. the percentage table -------------------------------------------------------------
    pct = re.findall(
        r"\|\s*(L12|L20)\s*\|\s*(RMSE|MAE)\s*\|\s*`(disease_mean|disease_mean_pernode)`\s*\|"
        r"\s*([-+][\d.]+)\s*\|\s*([-+][\d.]+)\s*\|\s*([-+][\d.]+)\s*\|\s*([-+][\d.]+)\s*\|"
        r"\s*\*\*(\d)/4\*\*\s*\|", text)
    chk(len(pct) == 8, "percentage table: found %d of 8 rows" % len(pct))
    for arm_s, met_s, floor, *rest in pct:
        arm = "ebola_" + arm_s
        metric = met_s.lower()
        vals = [float(x) for x in rest[:4]]
        wins = int(rest[4])
        zs = truth[arm][metric]["MODEL zero-shot"]
        fl = truth[arm][metric][floor]
        got = [(fl[h] - zs[h]) / fl[h] * 100.0 for h in HORIZONS]
        for h, v, w in zip(HORIZONS, vals, got):
            chk(abs(v - w) <= PCT_TOL,
                "pct %s %s %s h%d: doc %+.1f disk %+.1f" % (arm_s, met_s, floor, h, v, w))
        chk(wins == sum(1 for w in got if w > 0),
            "wins %s %s %s: doc %d disk %d" % (arm_s, met_s, floor, wins,
                                               sum(1 for w in got if w > 0)))

    # --- 3. the ordering claim: weaker in 15 of 16, with the exception named -----------------
    # An earlier draft claimed "every horizon on both metrics" and this check refuted it, so the
    # count AND the identity of the exception are both pinned. Loosening either hides a regression.
    claim = "`disease_mean` is a WEAKER floor than `support_mean` in 15 of the 16 cells"
    chk(claim in flat, "prose: the 15-of-16 WEAKER-floor claim is missing or reworded")
    harder = [(arm, metric, h)
              for arm in ARMS for metric in ("rmse", "mae") for h in HORIZONS
              if truth[arm][metric]["disease_mean"][h] <= truth[arm][metric]["support_mean"][h]]
    chk(len(harder) == 1, "ordering: %d cells have disease_mean at or below support_mean, doc says 1"
                          % len(harder))
    chk(harder == [("ebola_L20", "rmse", 5)],
        "ordering: the exception is %s, doc names ebola_L20 h5 RMSE" % harder)
    if harder == [("ebola_L20", "rmse", 5)]:
        gap = (truth["ebola_L20"]["rmse"]["support_mean"][5]
               - truth["ebola_L20"]["rmse"]["disease_mean"][5])
        m = re.search(r"marginally harder, by (\d+\.\d+)", flat)
        chk(m is not None, "prose: the L20 h5 gap sentence is missing")
        if m:
            chk(abs(gap - float(m.group(1))) < 5e-4,
                "ordering: the L20 h5 gap is %.3f, doc says %s" % (gap, m.group(1)))
        chk("41.747 against `support_mean`'s 42.042" in flat,
            "prose: the L20 h5 exception values are missing or wrong")

    # --- 4. prose numbers --------------------------------------------------------------------
    b12 = bundles.load("ebola_L12")
    g, per = disease_means(b12)
    raw, obs = b12.raw.astype(np.float64), (b12.M == 1)
    sm_mask = b12.masks()["support"].astype(bool)
    sup = np.array([raw[i, sm_mask[i]].mean() if sm_mask[i].any() else 0.0
                    for i in range(raw.shape[0])])
    chk(abs(g - 16.2726) < 5e-5, "prose: disease mean is %.4f, doc says 16.2726" % g)
    chk("16.2726" in flat, "prose: 16.2726 not stated")
    chk(abs(sup[sm_mask.any(1)].mean() - 7.5713) < 5e-5,
        "prose: L12 support-window mean is %.4f, doc says 7.5713" % sup[sm_mask.any(1)].mean())
    n_zero_sup = int((sup == 0).sum())
    chk(n_zero_sup == 47, "prose: support_mean zero districts on L12 are %d, doc says 47"
                          % n_zero_sup)
    chk("47 of 61 districts forecast a flat zero" in flat, "prose: the 47-of-61 claim is missing")
    n_zero_new = int(((per == 0) & obs.any(1)).sum())
    chk(n_zero_new == 4, "prose: new-floor zero districts are %d, doc says 4" % n_zero_new)
    chk(all((raw[i, obs[i]] == 0).all() for i in np.where((per == 0) & obs.any(1))[0]),
        "prose: a new-floor zero district is a fallback, contradicting the doc")

    # --- 5. the win/loss headline ranges -----------------------------------------------------
    allpct = {}
    for arm in ARMS:
        for metric in ("rmse", "mae"):
            zs = truth[arm][metric]["MODEL zero-shot"]
            for floor in ("disease_mean", "disease_mean_pernode"):
                fl = truth[arm][metric][floor]
                allpct.setdefault(floor, []).extend(
                    (fl[h] - zs[h]) / fl[h] * 100.0 for h in HORIZONS)
    w = allpct["disease_mean"]
    chk(all(v > 0 for v in w) and len(w) == 16, "disease_mean: zero-shot does not win 16 of 16")
    chk(abs(min(w) - 7.5) < 0.06 and abs(max(w) - 25.1) < 0.06,
        "disease_mean range is %.1f to %.1f, doc says 7.5 to 25.1" % (min(w), max(w)))
    chk("16 of 16 cells by 7.5 to 25.1 per cent" in flat, "prose: the 7.5-to-25.1 claim is missing")
    L = [-v for v in allpct["disease_mean_pernode"]]
    chk(all(v > 0 for v in L) and len(L) == 16,
        "disease_mean_pernode: zero-shot does not lose 16 of 16")
    chk(abs(min(L) - 9.5) < 0.06 and abs(max(L) - 28.3) < 0.06,
        "pernode loss range is %.1f to %.1f, doc says 9.5 to 28.3" % (min(L), max(L)))
    chk("16 of 16 cells by 9.5 to 28.3 per cent" in flat, "prose: the 9.5-to-28.3 claim is missing")

    # --- 6. the EXPLORATORY fence must be present and must match the record ------------------
    chk("EXPLORATORY" in flat, "the EXPLORATORY stamp is missing from the document")
    with open("experiments/disease_mean_floor.json") as fh:
        chk(json.load(fh)["protocol"] == "EXPLORATORY", "the record is not stamped EXPLORATORY")

    if fails:
        print("FAIL: %d check(s)" % len(fails))
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("ok  %s verified against disk" % a.doc)
    print("    2 floor tables x 6 series x 4 horizons, 8 percentage rows with win counts,")
    print("    32 ordering checks, 8 prose numbers, 2 headline ranges, EXPLORATORY fence")


if __name__ == "__main__":
    main()

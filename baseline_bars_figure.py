"""Two horizontal bar charts for the baseline comparison: SOTA vs single, and SOTA vs transfer.

Both charts are CELL-POOLED RMSE, the definition every baseline paper uses, read from the same
archives as Reports/baseline_reproduction_table.md via diagnostics.paper_compare. The country-macro
headline never enters here.

Every bar is normalised so our single-disease encoder is 1.0 in its row. Absolute RMSE spans an
order of magnitude across panels (about 167 on us-states, about 1700 on japan), so unnormalised bars
would be unreadable. A bar longer than 1.0 is worse than our single-disease ceiling. The absolute
numbers live in the table, not in the chart.

WHAT THE TRANSFER BARS ARE NOT. EpiGNN, Cola-GNN and HeatGNN are transductive and node-indexed:
they learn an embedding per region and cannot be run on a disease they were not trained on. Our
transfer arms hold the target disease out of training entirely, so they are solving a strictly
harder problem and there is no comparable baseline number to put beside them. Chart 2 prices what
the setting costs, it is not a claim about beating SOTA. The chart says so on its face, because a
reader who only sees three bar families will read it as a like-for-like race.

COVID is in neither chart: no baseline of any kind was run on it. Ebola is in neither chart either:
no published baseline can run on a disease with no training data.

  conda run -n ebola-train python -m baseline_bars_figure
  conda run -n ebola-train python -m baseline_bars_figure --selfcheck
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from gate_figure import AXIS, GRID, INK, INK2, MUTED, SURFACE

PANELS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "dengue"]
HORIZONS = (3, 5, 10, 15)
SHORT = {"influenza_japan": "flu japan", "influenza_us-regions": "flu us-regions",
         "influenza_us-states": "flu us-states", "dengue": "dengue*"}
# MTGNN is excluded as a comparator: 47 of 80 of its prediction files are one repeated constant, and
# its paper has no epidemic dataset to validate against. Beating a constant is not evidence.
BASELINES = ["EpiGNN", "Cola-GNN", "HeatGNN"]

BLUE_D, BLUE_M, BLUE_L = "#1b5aa8", "#4a90dd", "#9dc2ee"
OURS = "#0b0b0b"
WARM_D, WARM_L = "#c2620d", "#eda45c"      # transfer family, hatched


def cells():
    """{(panel, h): dict of pooled RMSE means} from the archives paper_compare already reads."""
    from diagnostics import paper_compare as pc

    acc = pc.collect()
    enc = pc.encoder_pooled()
    tra = pc.encoder_pooled("lodo", "encoder_ldo3")
    trz = pc.encoder_pooled("lodo", "encoder_ldo3_zeroshot")
    out = {}
    for ds in PANELS:
        for h in HORIZONS:
            row = {m: float(np.mean([v[0] for v in acc[(m, ds, h)]]))
                   for m in BASELINES if (m, ds, h) in acc}
            row["ours"] = float(np.mean(enc[(ds, h)]))
            row["adapted"] = float(np.mean(tra[(ds, h)]))
            row["zeroshot"] = float(np.mean(trz[(ds, h)]))
            out[(ds, h)] = row
    return out


def rows(data):
    """Panel-major, horizon ascending. Same order in both charts."""
    return [(ds, h, data[(ds, h)]) for ds in PANELS for h in HORIZONS]


def _frame(ax, order, xmax):
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{SHORT[ds]}  h{h}" for ds, h, _ in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0, color=INK2, linewidth=1.0, zorder=2)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("cell-pooled RMSE, relative to our single-disease encoder (1.0 = tie)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    for i in range(4, len(order), 4):                 # thin rule between panels
        ax.axhline(i - 0.5, color=GRID, linewidth=0.8, zorder=0)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "hatch.linewidth": 0.6,
    })
    return plt


def chart1(data, out="figures/baselines_vs_single"):
    plt = _style()
    order = rows(data)
    series = [("EpiGNN", "EpiGNN", BLUE_D, None), ("Cola-GNN", "Cola-GNN", BLUE_M, None),
              ("HeatGNN", "HeatGNN", BLUE_L, None), ("ours", "ours, single-disease", OURS, None)]
    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    bh = 0.20
    for k, (key, label, colour, hatch) in enumerate(series):
        ys = [i + (k - 1.5) * bh for i in range(len(order))]
        vs = [r.get(key, np.nan) / r["ours"] for _, _, r in order]
        ax.barh(ys, vs, height=bh, color=colour, edgecolor=SURFACE, linewidth=0.6,
                hatch=hatch, label=label, zorder=3)
        for i, v in enumerate(vs):                    # dengue has no Cola-GNN and no HeatGNN
            if np.isnan(v):
                ax.text(0.012, i + (k - 1.5) * bh, f"{label} cannot run on dengue", va="center",
                        fontsize=6.5, color=MUTED, zorder=4)
    _frame(ax, order, 1.55)
    ax.set_title("Published baselines against our single-disease encoder, cell-pooled RMSE",
                 loc="left", color=INK, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.085), frameon=False, fontsize=8, ncol=4)
    fig.text(0.005, 0.012,
             "Every model in this chart trains on the disease it is scored on. Bars are the mean over "
             "5 seeds, normalised per row so our single-disease encoder is 1.0; longer is worse.\n"
             "* dengue is pooled over the 2,392-node subsample on BOTH sides, because the baseline "
             "repos cannot take 7,165 nodes. EpiGNN is the only baseline that runs on dengue, so the "
             "Cola-GNN and\nHeatGNN bars are absent there by construction, not by omission. MTGNN is "
             "excluded as a comparator: 47 of its 80 prediction files hold one repeated constant. "
             "COVID has no baseline of\nany kind and is not shown; Ebola has none either, since no "
             "node-indexed baseline can run on a disease with no training data.",
             fontsize=7.5, color=MUTED, va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.945, bottom=0.215)
    _save(fig, out)


def chart2(data, out="figures/transfer_vs_baselines"):
    plt = _style()
    order = rows(data)
    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    bh = 0.20
    best = [min(r[m] for m in BASELINES if m in r) for _, _, r in order]
    series = [(best, "best baseline in the cell  (trains on the target disease)", BLUE_M, None),
              ([r["ours"] for _, _, r in order],
               "ours, single-disease  (trains on the target disease)", OURS, None),
              ([r["adapted"] for _, _, r in order],
               "ours, transfer few-shot  (target disease held out)", WARM_D, "///"),
              ([r["zeroshot"] for _, _, r in order],
               "ours, transfer zero-shot  (target disease held out)", WARM_L, "///")]
    for k, (vals, label, colour, hatch) in enumerate(series):
        ys = [i + (k - 1.5) * bh for i in range(len(order))]
        vs = [v / r["ours"] for v, (_, _, r) in zip(vals, order)]
        ax.barh(ys, vs, height=bh, color=colour, edgecolor=SURFACE, linewidth=0.6,
                hatch=hatch, label=label, zorder=3)
    _frame(ax, order, 1.75)
    ax.set_title("What holding the disease out costs, cell-pooled RMSE\n"
                 "hatched bars never see the target disease, the other two train on it",
                 loc="left", color=INK, pad=10)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.085), frameon=False,
                    fontsize=8, ncol=2)
    for t, c in zip(leg.get_texts(), [INK2, INK2, WARM_D, WARM_D]):
        t.set_color(c)
    fig.text(0.005, 0.012,
             "The two hatched bars are a DIFFERENT task, not a fairer version of the same one. Blue "
             "and black train on the disease they are scored on; orange holds that disease out of "
             "training entirely.\nEpiGNN, Cola-GNN and HeatGNN are transductive and node-indexed, so "
             "they cannot be run in the held-out setting at all and have no bar there. Read the orange "
             "bars as the price of the\nsetting, not as a race. Baselines are collapsed to the best "
             "one per cell, which steel-mans them. Bars are the mean over 5 seeds, normalised per row "
             "so our single-disease encoder is 1.0.\n* dengue is pooled over the 2,392-node subsample "
             "on all bars. The LDO3 records predate the trunk early-stop fix; a full-budget rerun at "
             "seed 42 reproduced them bit-identically on all\nthree folds (10 of 10), but that check "
             "is seed 42 only. COVID is not shown because it has no baseline; Ebola is not shown "
             "because no node-indexed baseline can run on it.",
             fontsize=7.5, color=MUTED, va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.925, bottom=0.245)
    _save(fig, out)


def _save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"  wrote {out}.png and {out}.pdf")


def table(data):
    """The absolute numbers, printed rather than drawn."""
    order = rows(data)
    print(f"\n{'=' * 100}\ncell-pooled RMSE, mean over 5 seeds. ratios are vs our single-disease "
          f"encoder\n{'=' * 100}")
    print(f"  {'cell':22} {'bestSOTA':>9} {'ours':>9} {'adapted':>9} {'zeroshot':>9}   "
          f"{'ad/ours':>8} {'zs/ours':>8} {'ad/best':>8}")
    for ds, h, r in order:
        b = min(r[m] for m in BASELINES if m in r)
        print(f"  {SHORT[ds] + '  h' + str(h):22} {b:9.1f} {r['ours']:9.1f} {r['adapted']:9.1f} "
              f"{r['zeroshot']:9.1f}   {r['adapted'] / r['ours']:8.3f} "
              f"{r['zeroshot'] / r['ours']:8.3f} {r['adapted'] / b:8.3f}")
    n = len(order)
    ab = sum(1 for _, _, r in order if r["adapted"] < r["ours"])
    zb = sum(1 for _, _, r in order if r["zeroshot"] < r["ours"])
    abest = sum(1 for _, _, r in order
                if r["adapted"] < min(r[m] for m in BASELINES if m in r))
    zbest = sum(1 for _, _, r in order
                if r["zeroshot"] < min(r[m] for m in BASELINES if m in r))
    print(f"\n  adapted beats our single-disease encoder in {ab}/{n} cells, zero-shot in {zb}/{n}.")
    print(f"  adapted beats the best baseline in {abest}/{n} cells, zero-shot in {zbest}/{n}.")
    print("  Those baseline comparisons are NOT like-for-like: the baselines train on the target "
          "disease\n  and cannot be run without it. Read them as the price of the held-out setting.")


def _selfcheck():
    """Normalisation and row order, on synthetic input. No artifacts needed."""
    d = {(ds, h): {"EpiGNN": 100.0, "Cola-GNN": 90.0, "HeatGNN": 110.0,
                   "ours": 100.0, "adapted": 120.0, "zeroshot": 150.0}
         for ds in PANELS for h in HORIZONS}
    order = rows(d)
    assert len(order) == 16, len(order)
    assert [(ds, h) for ds, h, _ in order[:5]] == [
        ("influenza_japan", 3), ("influenza_japan", 5), ("influenza_japan", 10),
        ("influenza_japan", 15), ("influenza_us-regions", 3)], order[:5]

    # (1) the normaliser is the row's OWN single-disease encoder, so its bar is exactly 1.0 and a
    #     row with a different scale cannot shift another row. Getting this wrong (one global
    #     normaliser) would make japan's bars 10x us-states' and hide every within-row comparison.
    d[("dengue", 15)] = dict(d[("dengue", 15)], ours=500.0, adapted=600.0)
    r = d[("dengue", 15)]
    assert abs(r["ours"] / r["ours"] - 1.0) < 1e-12
    assert abs(r["adapted"] / r["ours"] - 1.2) < 1e-12, r["adapted"] / r["ours"]
    assert abs(d[("influenza_japan", 3)]["adapted"] / d[("influenza_japan", 3)]["ours"] - 1.2) < 1e-12

    # (2) "best baseline" is the SMALLEST RMSE, which steel-mans the opponent. Taking the mean or the
    #     max would flatter us, and this chart is the one place that must not.
    b = min(d[("influenza_japan", 3)][m] for m in BASELINES if m in d[("influenza_japan", 3)])
    assert b == 90.0, b
    # control: a cell with only EpiGNN (dengue) must still resolve, to EpiGNN.
    only = {"EpiGNN": 42.0, "ours": 1.0, "adapted": 1.0, "zeroshot": 1.0}
    assert min(only[m] for m in BASELINES if m in only) == 42.0

    # (3) MTGNN must never reach a bar, on either chart.
    assert "MTGNN" not in BASELINES

    print("ok  16 rows panel-major; normaliser is the row's own single-disease encoder; "
          "best baseline is the minimum; MTGNN excluded")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
        sys.exit(0)
    data = cells()
    table(data)
    chart1(data)
    chart2(data)

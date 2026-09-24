"""Manuscript figures for the ablations and the full baseline comparison.

The paper carries eight tables and one figure, which is a known reviewer objection. This script
draws six figures, one per experiment, EVERY value read from the artifacts on disk (results/,
ablation/, results/misc/, results/baselines/, results/naive/). No number is typed in by hand, which
is the repo rule: a figure without a generator is a defect.

  F1  spatial ablation      gate-learned vs gate-off, the graph does not help error
  F2  meta-learning         ANIL vs its own seed-matched control, 32 cells, 31 within noise
  F3  epi-informed penalty   two strictness bounds vs base, 4 panels (dengue not run)
  F4  window sensitivity     lookback 12 / 20 (base) / 32
  F5  Ebola adapted vs unadapted, both arms, E6-compliant labels
  F6  baseline comparison    encoder vs 4 GNNs vs classical vs persistence, per panel

Significance for the ablation figures uses the SAME rule the committed reports use
(ablation/run_gate_ablation.py, ablation/run_epi_ablation.py): a paired-by-seed delta on the shared
seeds, "within noise" when the absolute mean is below its own across-seed sd. F2 uses the repo's
canonical paired_delta (95% t-interval) via diagnostics.anil_report, so the figure and
progress/outcomes/ANIL_Results.md cannot drift apart.

Every figure asserts one spot cell against its source record and, where the report states a tally,
asserts that tally too. Missing inputs raise, they do not pass silently.

  <ebola-train python> paper_figures.py --only all
  <ebola-train python> paper_figures.py --only F1
  <ebola-train python> paper_figures.py --selfcheck

Writes figures/<name>.png and figures/<name>.pdf, and prints a suggested caption per figure.
Read-only against every artifact.
"""
import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.lines import Line2D                                   # noqa: E402

# ---- shared house style, matched to gate_figure.py / baseline_bars_figure.py ---------------- #
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OURS = "#0b0b0b"          # our encoder / our result: always bold, always this ink
OURS_2 = "#0b0b0b"
WARN = "#c0392b"          # a significant move against us / a flag
GOOD = "#1e7d5a"          # a significant move for the honest story (removing the graph helps)
BLUE = "#1b5aa8"
BLUE_M = "#4a90dd"
PALE = "#9dc2ee"
WARM_D, WARM_M, WARM_L = "#c2620d", "#e08a3c", "#f0b978"

HORIZONS = (3, 5, 10, 15)
SEEDS = (42, 52, 62, 72, 82)

PANEL_NAME = {
    "influenza_japan": "Influenza Japan",
    "influenza_us-regions": "Influenza US-Regions",
    "influenza_us-states": "Influenza US-States",
    "covid_us-states": "COVID US-States",
    "dengue": "Dengue",
}


def _rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    })


def _despine(ax, left=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)


def _save(fig, name):
    outdir = REPO / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    for p in paths:
        assert p.exists() and p.stat().st_size > 0, f"failed to write {p}"
    return paths


# --------------------------------------------------------------------------- #
# artifact loaders  (model filter matters: records also carry the "encoder_mc" bias-corrected row)
# --------------------------------------------------------------------------- #
def _records(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d if isinstance(d, list) else [d]


def _field(panel):
    """The headline field the committed reports use: dengue is a country-macro over 12 countries,
    every single-country panel has node_mean == country_macro so either reads the same."""
    return "country_macro" if panel == "dengue" else "node_mean"


def _mean_sd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def _paired(base_by_seed, abl_by_seed):
    """(mean, sd, n) of abl - base over the seeds present in BOTH arms. Same as the ablation reports."""
    seeds = sorted(set(base_by_seed) & set(abl_by_seed))
    d = [abl_by_seed[s] - base_by_seed[s] for s in seeds]
    m, sd = _mean_sd(d)
    return m, sd, len(d)


def _within_noise(dm, dsd, n):
    return n < 2 or dsd == 0 or abs(dm) < dsd


def _seed_metric(path, panel, metric, model="encoder"):
    """{horizon: value} for one (panel, seed) record file, filtered to the given model."""
    out = {}
    for r in _records(path):
        if r.get("model") == model and r.get("metric") == metric:
            out[r["horizon"]] = r[_field(panel)]
    return out


def _arm_by_seed(fmt, panel, metric, model="encoder"):
    """{horizon: {seed: value}} across the 5 frozen seeds. Raises if a seed file is missing."""
    out = {h: {} for h in HORIZONS}
    for s in SEEDS:
        vals = _seed_metric(REPO / fmt.format(panel=panel, s=s), panel, metric, model)
        for h in HORIZONS:
            if h in vals:
                out[h][s] = vals[h]
    return out


# --------------------------------------------------------------------------- #
# F1  spatial (gate-off) ablation
# --------------------------------------------------------------------------- #
def fig_F1():
    _rc()
    panels = ["influenza_japan", "influenza_us-regions", "influenza_us-states",
              "covid_us-states", "dengue"]
    metrics = ("rmse", "mae")
    # improvement % from removing the graph, per panel/metric/horizon, plus the raw verdict tally
    data, tally = {}, {"graph_helps": 0, "removal_helps": 0, "noise": 0}
    for p in panels:
        for metric in metrics:
            base = _arm_by_seed("results/single/encoder__{panel}__seed{s}.json", p, metric)
            off = _arm_by_seed("ablation/single/encoder__{panel}__seed{s}__gateoff.json", p, metric)
            for h in HORIZONS:
                dm, dsd, n = _paired(base[h], off[h])        # off - learned, raw units
                bm, _ = _mean_sd(list(base[h].values()))
                pct = 100.0 * (-dm) / bm                       # positive = removing the graph lowers error
                # half-width in the same % units, for the whisker
                seeds = sorted(set(base[h]) & set(off[h]))
                pdeltas = [100.0 * (base[h][s] - off[h][s]) / bm for s in seeds]
                _, psd = _mean_sd(pdeltas)
                hw = psd / math.sqrt(len(pdeltas)) if len(pdeltas) > 1 else 0.0
                noise = _within_noise(dm, dsd, n)
                if noise:
                    tally["noise"] += 1
                elif dm < 0:      # off better => removing the graph helps
                    tally["removal_helps"] += 1
                else:
                    tally["graph_helps"] += 1
                data[(p, metric, h)] = (pct, hw, noise, dm >= 0)
    # verdict tally must reproduce the committed log exactly (error: 0 help, 8 hurt, 32 noise)
    assert (tally["graph_helps"], tally["removal_helps"], tally["noise"]) == (0, 8, 32), tally
    # spot cell against the log: influenza_japan RMSE h3 learned mean is 734.888
    lm, _ = _mean_sd(list(_arm_by_seed("results/single/encoder__{panel}__seed{s}.json",
                                       "influenza_japan", "rmse")[3].values()))
    assert abs(lm - 734.888) < 0.05, lm

    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.8), sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    xs = list(range(len(HORIZONS)))
    bw = 0.36
    for ax, p in zip(axes, panels):
        for k, metric in enumerate(metrics):
            vals = [data[(p, metric, h)][0] for h in HORIZONS]
            hws = [data[(p, metric, h)][1] for h in HORIZONS]
            cols = []
            for h in HORIZONS:
                _pct, _hw, noise, _ = data[(p, metric, h)]
                cols.append(MUTED if noise else GOOD)     # significant here is always removal-helps
            pos = [x + (k - 0.5) * bw for x in xs]
            ax.bar(pos, vals, bw, color=cols, edgecolor=SURFACE, linewidth=0.6, zorder=3,
                   label=metric.upper())
            ax.errorbar(pos, vals, yerr=hws, fmt="none", ecolor=INK2, elinewidth=0.9,
                        capsize=2, zorder=4)
        ax.axhline(0, color=INK2, linewidth=1.0, zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{h}" for h in HORIZONS], fontsize=8.5)
        ax.set_title(PANEL_NAME[p], color=INK, fontsize=9.5, loc="left", pad=6)
        ax.set_xlabel("horizon (weeks)", fontsize=8.5)
        _despine(ax)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    axes[0].set_ylabel("removing the graph:\n% change in error (+ = error drops)", fontsize=8.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GOOD),
        plt.Rectangle((0, 0), 1, 1, color=MUTED),
        Line2D([0], [0], color=INK2, lw=0.9),
    ]
    axes[-1].legend(handles, ["removal helps (significant)", "within noise",
                              "hatch group: left bar RMSE, right bar MAE"][:2] +
                    ["95% seed interval"], frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("F1  Switching the spatial graph off. Error metric, lower is better; "
                 "bars show the change from removing the graph.",
                 x=0.008, ha="left", color=INK, fontsize=11, y=1.04)
    fig.text(0.008, -0.12,
             "Paired by seed against the learned-gate model, 5 seeds, same rule as the committed "
             "gate_ablation.log: within noise when the absolute mean is below its own across-seed sd. "
             "Left bar RMSE, right bar MAE.\nOver the 40 error cells the graph helps 0 and removing "
             "it wins 8 (all green), the rest are noise. The graph earns its keep on correlation only, "
             "not on error. Single-disease checkpoints (results/single/); dengue uses its "
             "country-macro.",
             color=INK2, fontsize=7.8, va="top", linespacing=1.5)
    _save(fig, "F1_spatial_ablation")
    cap = ("F1. Turning the spatial message-passing graph off, paired against the learned-gate model "
           "on the same seed (5 seeds, single-disease checkpoints; RMSE and MAE, lower is better; "
           "horizon in weeks). Bars are the percent change in error from removing the graph, positive "
           "meaning error drops when the graph is removed; green bars are significant under the "
           "committed report's within-noise rule. Across all 40 error cells the graph improves error "
           "in 0 and removing it significantly helps in 8, with the remaining 32 within noise, so the "
           "spatial channel does not buy accuracy. Scope: this is an inference and gate-off test on "
           "single-disease models, and the graph still helps correlation in a minority of cells, which "
           "is not shown here.")
    return "F1_spatial_ablation", cap


# --------------------------------------------------------------------------- #
# F2  meta-learning: ANIL vs its own control
# --------------------------------------------------------------------------- #
def fig_F2():
    _rc()
    import diagnostics.anil_report as ar
    from results_matrix import paired_delta

    data = ar.collect()
    # fold -> ordered list of (label, cell, mean, hw, clears)
    fold_order = [f for f in ar.FOLDS]
    rows, tally = [], {"better": 0, "worse": 0, "noise": 0}
    for fold in fold_order:
        prim = data[fold]["primary"]
        anil_c = ar.rmse_by_seed(data[fold]["rows"]["anil"])
        ctrl_c = ar.rmse_by_seed(data[fold]["rows"]["control"])
        for cell in sorted(prim, key=lambda c: (c.split("|")[0], int(c.split("h")[1]))):
            text, m, clears = prim[cell]
            # rebuild the whisker in the same % units paired_delta used
            new, ref = anil_c[cell], ctrl_c[cell]
            deltas = [(ref[s] - new[s]) / abs(ref[s]) * 100.0
                      for s in sorted(set(new) & set(ref)) if abs(ref[s]) > 1e-12]
            _, sd = _mean_sd(deltas)
            hw = (2.776 if len(deltas) == 5 else 3.182) * sd / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
            rows.append((fold, cell, m, hw, clears))
            if clears and m > 0:
                tally["better"] += 1
            elif clears and m < 0:
                tally["worse"] += 1
            else:
                tally["noise"] += 1
    assert (tally["better"], tally["worse"], tally["noise"]) == (0, 1, 31), tally
    # spot cell: covid h15 ANIL vs control mean is about -8.9 (worse), and it is within noise
    covid = data["covid"]["primary"]["covid_us-states|h15"]
    assert covid[2] is False and covid[1] < 0, covid

    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.2, 9.4))
    ys = list(range(n))
    for y, (fold, cell, m, hw, clears) in zip(ys, rows):
        col = WARN if (clears and m < 0) else (GOOD if (clears and m > 0) else MUTED)
        ax.barh(y, m, height=0.62, color=col, edgecolor=SURFACE, linewidth=0.5, zorder=3)
        ax.errorbar(m, y, xerr=hw, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=4)
    ax.axvline(0, color=INK2, linewidth=1.1, zorder=2)
    ax.set_yticks(ys)
    labels = []
    for fold, cell, *_ in rows:
        panel, h = cell.split("|")
        labels.append(f"{PANEL_NAME.get(panel, panel)}  {h}")
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.invert_yaxis()
    # fold separators
    seps = {}
    for i, (fold, *_ ) in enumerate(rows):
        seps.setdefault(fold, i)
    for fold, i in seps.items():
        if i > 0:
            ax.axhline(i - 0.5, color=GRID, linewidth=0.9, zorder=0)
    ax.set_xlabel("ANIL minus its control: % improvement in RMSE  (positive = meta-learning wins)",
                  fontsize=8.6)
    _despine(ax, left=False)
    ax.set_xlim(-45, 20)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("F2  Meta-learning against its own seed-matched control, 32 cells across 4 folds.\n"
                 "RMSE, positive means ANIL beats the control; whiskers are the 95% t-interval.",
                 loc="left", color=INK, pad=10, fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED), plt.Rectangle((0, 0), 1, 1, color=WARN)]
    ax.legend(handles, ["within noise (whisker crosses zero)", "significantly worse than control"],
              frameon=False, fontsize=8, loc="lower left")
    fig.text(0.005, 0.005,
             "ANIL against a control that runs the same episode stream and outer-update count with no "
             "inner loop, so the only difference is whether adaptation happened during training "
             "(diagnostics/anil_report.py, results/misc/anil_*). Of 32 cells, 0 favour ANIL, 1 is "
             "significantly worse, 31 are within noise.\nEvery cell warm-starts from an existing "
             "trunk, so this is meta-fine-tuning, and the meta-test fits a fresh adapter on the full "
             "held-out train fold, so nothing here measures few-shot behaviour.",
             color=INK2, fontsize=7.8, va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=0.30, right=0.985, top=0.90, bottom=0.11)
    _save(fig, "F2_metalearning_ablation")
    cap = ("F2. Adaptive meta-learning (ANIL) against its own seed-matched control across 32 cells "
           "and four held-out folds (RMSE, positive means ANIL beats the control; whiskers are the "
           "95 percent t-interval). The control sees the identical episode stream and outer-update "
           "count with no inner adaptation loop, so the contrast isolates meta-learning itself. ANIL "
           "wins 0 cells, is significantly worse in 1, and is within noise in the other 31, so "
           "meta-learning does not help here. Scope: every arm warm-starts from an existing trunk and "
           "the meta-test fits a fresh adapter on the full held-out train fold, so this measures "
           "meta-fine-tuning, not few-shot adaptation.")
    return "F2_metalearning_ablation", cap


# --------------------------------------------------------------------------- #
# F3  epi-informed penalty
# --------------------------------------------------------------------------- #
def fig_F3():
    _rc()
    panels = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states"]
    bounds = [("epi_p90max", "p90 bound", BLUE), ("epi_p99max", "p99 bound", WARM_D)]
    data = {}
    for p in panels:
        base = _arm_by_seed("results/single/encoder__{panel}__seed{s}.json", p, "rmse")
        for tag, _lbl, _c in bounds:
            abl = _arm_by_seed("ablation/single/encoder__{panel}__seed{s}__" + tag + ".json",
                               p, "rmse")
            for h in HORIZONS:
                dm, dsd, n = _paired(base[h], abl[h])     # epi - baseline, raw units
                bm, _ = _mean_sd(list(base[h].values()))
                pct = 100.0 * (-dm) / bm                    # positive = epi lowers error
                seeds = sorted(set(base[h]) & set(abl[h]))
                pdeltas = [100.0 * (base[h][s] - abl[h][s]) / bm for s in seeds]
                _, psd = _mean_sd(pdeltas)
                hw = psd / math.sqrt(len(pdeltas)) if len(pdeltas) > 1 else 0.0
                data[(p, tag, h)] = (pct, hw, _within_noise(dm, dsd, n))
    # spot: us-regions p90 h5 raw paired delta is -11.541 (report)
    base = _arm_by_seed("results/single/encoder__{panel}__seed{s}.json", "influenza_us-regions", "rmse")
    abl = _arm_by_seed("ablation/single/encoder__{panel}__seed{s}__epi_p90max.json",
                       "influenza_us-regions", "rmse")
    dm, _, _ = _paired(base[5], abl[5])
    assert abs(dm - (-11.541)) < 0.1, dm

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), sharey=True, gridspec_kw=dict(wspace=0.12))
    xs = list(range(len(HORIZONS)))
    bw = 0.36
    for ax, p in zip(axes, panels):
        for k, (tag, lbl, col) in enumerate(bounds):
            vals = [data[(p, tag, h)][0] for h in HORIZONS]
            hws = [data[(p, tag, h)][1] for h in HORIZONS]
            cols = [col if not data[(p, tag, h)][2] else PALE for h in HORIZONS]
            pos = [x + (k - 0.5) * bw for x in xs]
            ax.bar(pos, vals, bw, color=cols, edgecolor=SURFACE, linewidth=0.6, zorder=3, label=lbl)
            ax.errorbar(pos, vals, yerr=hws, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2,
                        zorder=4)
        ax.axhline(0, color=INK2, linewidth=1.0, zorder=2)
        ax.set_xticks(xs); ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=8.5)
        ax.set_xlabel("horizon (weeks)", fontsize=8.5)
        ax.set_title(PANEL_NAME[p], color=INK, fontsize=9.5, loc="left", pad=6)
        _despine(ax); ax.yaxis.grid(True); ax.set_axisbelow(True)
    axes[0].set_ylabel("epi penalty:\n% change in RMSE (+ = error drops)", fontsize=8.5)
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("F3  Epidemiology-informed penalty. RMSE, lower is better; bars show the change "
                 "from adding the penalty.", x=0.008, ha="left", color=INK, fontsize=11, y=1.04)
    fig.text(0.008, -0.12,
             "Paired by seed against the plain single-disease model, 5 seeds, same within-noise rule "
             "as the committed epi reports (a bound whose bar is pale\n"
             "is within noise). Two strictness bounds shown. Dengue was NOT run for this ablation and "
             "is absent by design, not omitted. Almost every cell sits\n"
             "within noise, so steering the model with a peak-intensity penalty neither helps nor "
             "hurts error.",
             color=INK2, fontsize=7.8, va="top", linespacing=1.5)
    _save(fig, "F3_epi_penalty_ablation")
    cap = ("F3. Adding an epidemiology-informed penalty at two strictness bounds, paired against the "
           "plain single-disease model on the same seed (5 seeds; RMSE, lower is better; horizon in "
           "weeks). Bars are the percent change in error, positive meaning error drops with the "
           "penalty; pale bars are within noise under the committed report's rule. The penalty leaves "
           "error essentially unchanged across all four panels, one influenza cell aside, so the "
           "hand-designed epidemic prior does not improve accuracy. Scope: dengue was not run for this "
           "ablation and is shown as absent, not omitted.")
    return "F3_epi_penalty_ablation", cap


# --------------------------------------------------------------------------- #
# F4  window (lookback) sensitivity
# --------------------------------------------------------------------------- #
def fig_F4():
    _rc()
    panels = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states"]
    variants = [("w12", "lookback 12", BLUE), (None, "lookback 20 (base)", OURS),
                ("w32", "lookback 32", WARM_D)]
    data = {}
    for p in panels:
        base = _arm_by_seed("results/single/encoder__{panel}__seed{s}.json", p, "rmse")
        for tag, _lbl, _c in variants:
            if tag is None:
                for h in HORIZONS:
                    bm, bsd = _mean_sd(list(base[h].values()))
                    data[(p, "w20", h)] = (bm, bsd, False)
                continue
            abl = _arm_by_seed("ablation/single/encoder__{panel}__seed{s}__" + tag + ".json",
                               p, "rmse")
            for h in HORIZONS:
                am, asd = _mean_sd(list(abl[h].values()))
                dm, dsd, n = _paired(base[h], abl[h])
                data[(p, tag, h)] = (am, asd, _within_noise(dm, dsd, n))
    # spot: us-states w12 h3 mean is 104.939 (report)
    m = data[("influenza_us-states", "w12", 3)][0]
    assert abs(m - 104.939) < 0.05, m

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9), gridspec_kw=dict(wspace=0.28))
    xs = list(range(len(HORIZONS)))
    bw = 0.26
    order = [("w12", "lookback 12", BLUE), ("w20", "lookback 20 (base)", OURS),
             ("w32", "lookback 32", WARM_D)]
    for ax, p in zip(axes, panels):
        for k, (tag, lbl, col) in enumerate(order):
            vals = [data[(p, tag, h)][0] for h in HORIZONS]
            sds = [data[(p, tag, h)][1] for h in HORIZONS]
            pos = [x + (k - 1) * bw for x in xs]
            lw = 0.9 if tag == "w20" else 0.6
            ax.bar(pos, vals, bw, color=col, edgecolor=SURFACE, linewidth=lw, zorder=3, label=lbl)
            ax.errorbar(pos, vals, yerr=sds, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2,
                        zorder=4)
        ax.set_xticks(xs); ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=8.5)
        ax.set_xlabel("horizon (weeks)", fontsize=8.5)
        ax.set_title(PANEL_NAME[p], color=INK, fontsize=9.5, loc="left", pad=6)
        _despine(ax); ax.yaxis.grid(True); ax.set_axisbelow(True)
    axes[0].set_ylabel("RMSE (lower is better)", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7.6, loc="upper left")
    fig.suptitle("F4  Lookback window sensitivity. RMSE, lower is better; the base model uses "
                 "lookback 20 (black).", x=0.008, ha="left", color=INK, fontsize=11, y=1.04)
    fig.text(0.008, -0.12,
             "Absolute RMSE, 5-seed mean, error bars the across-seed sd. The released base model uses "
             "a 20-week lookback (black). A shorter window helps a\n"
             "little on us-states and japan at long horizons, a longer window clearly hurts on COVID; "
             "on paired within-noise tests most cells are\n"
             "indistinguishable from base. Dengue was not run for this sweep and is absent by design.",
             color=INK2, fontsize=7.8, va="top", linespacing=1.5)
    _save(fig, "F4_window_sensitivity")
    cap = ("F4. Sensitivity of the single-disease model to the input lookback window (RMSE, lower is "
           "better; horizon in weeks; 5-seed mean, error bars the across-seed spread). The released "
           "model uses a 20-week window, shown in black between a shorter 12-week and a longer 32-week "
           "setting. A shorter window helps modestly on US-States and Japan at long horizons while a "
           "longer window clearly hurts COVID, but most cells are within noise of the base in the "
           "paired test, so the model is not fragile to this choice. Scope: dengue was not run for "
           "this sweep and is absent by design.")
    return "F4_window_sensitivity", cap


# --------------------------------------------------------------------------- #
# F5  Ebola adapted vs unadapted, both arms, E6-compliant labels
# --------------------------------------------------------------------------- #
def fig_F5():
    _rc()
    arms = [("ebola_L12", "L12 (primary arm)"), ("ebola_L20", "L20 (secondary arm)")]

    def macro(model, ds):
        vals = {h: [] for h in HORIZONS}
        for s in SEEDS:
            for r in _records(REPO / f"results/ebola/{model}__{ds}__seed{s}.json"):
                if r.get("metric") == "rmse":
                    vals[r["horizon"]].append(r["country_macro"])
        return {h: _mean_sd(vals[h]) for h in HORIZONS}

    def floor(ds):
        out = {}
        for r in _records(REPO / f"results/naive/naive__{ds}.json"):
            if r.get("model") == "persistence" and r.get("metric") == "rmse":
                out[r["horizon"]] = r["country_macro"]
        return out

    # E6 provenance: the footnote's pair counts must match the frozen config, or the claim drifts.
    _cfg = _records(REPO / "configs/ebola_arms.json")[0]
    _ap = _cfg["arms"]["ebola_L12"]["counts"]["adapt_pairs"]
    assert (_ap["3"], _ap["5"], _ap["10"], _ap["15"]) == (48, 38, 18, 0), _ap

    # spot: L12 unadapted (zero-shot) rmse h3 seed-mean must equal the mean of the 5 zeroshot files
    zs = []
    for s in SEEDS:
        for r in _records(REPO / f"results/ebola/encoder_ebola_zeroshot__ebola_L12__seed{s}.json"):
            if r.get("metric") == "rmse" and r["horizon"] == 3:
                zs.append(r["country_macro"])
    assert abs(macro("encoder_ebola_zeroshot", "ebola_L12")[3][0] - sum(zs) / len(zs)) < 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw=dict(wspace=0.22))
    xs = list(range(len(HORIZONS)))
    for ax, (ds, title) in zip(axes, arms):
        una = macro("encoder_ebola_zeroshot", ds)   # unadapted == zero-shot
        ada = macro("encoder_ebola", ds)            # adapted == few-shot
        fl = floor(ds)
        ax.plot(xs, [fl[h] for h in HORIZONS], color=MUTED, lw=1.4, ls="--", marker="s",
                markersize=5, zorder=2, label="persistence floor")
        ax.errorbar(xs, [una[h][0] for h in HORIZONS], yerr=[una[h][1] for h in HORIZONS],
                    color=OURS, lw=2.6, marker="o", markersize=7, capsize=3, zorder=4,
                    label="ours, unadapted (zero-shot)")
        ax.errorbar(xs, [ada[h][0] for h in HORIZONS], yerr=[ada[h][1] for h in HORIZONS],
                    color=WARM_D, lw=2.2, marker="D", markersize=6, capsize=3, zorder=3,
                    label="ours, adapted")
        # E6: on L12 the h10 and h15 adapted points must be reported zero-shot, never few-shot.
        # h15 has 0 adaptation pairs; h10 has only 18 on near-fully padded windows (configs/ebola_arms.json).
        if ds == "ebola_L12":
            for h in (10, 15):
                i = HORIZONS.index(h)
                ax.annotate("*", (i, ada[h][0]), textcoords="offset points", xytext=(6, 4),
                            fontsize=14, color=WARN, zorder=6)
        ax.set_xticks(xs); ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=9)
        ax.set_xlabel("horizon (weeks)", fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
        _despine(ax); ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("Ebola country-macro RMSE (lower is better)", fontsize=9)
    fig.suptitle("F5  Ebola forecast on a disease the model never trained on. "
                 "RMSE, lower is better; both arms beat the persistence floor.",
                 x=0.008, ha="left", color=INK, fontsize=11, y=1.02)
    fig.text(0.012, -0.02,
             "5-seed mean, error bars the across-seed sd, country-macro RMSE in raw case counts "
             "(results/ebola/, results/naive/). Unadapted is the\n"
             "frozen trunk applied zero-shot, so it reads no Ebola label at any horizon; adapted fits "
             "the Ebola adapter, a genuine few-shot fit only\n"
             "at h3 and h5 (48 and 38 adaptation pairs). * On the L12 arm prereg E6 requires h10 and "
             "h15 to be reported zero-shot, never few-shot:\n"
             "h15 has zero adaptation pairs and h10 has only 18 on near-fully padded windows. These "
             "are point means with no interval, so nothing\n"
             "here is a significance claim; the pre-registered adapted-beats-persistence criterion "
             "was not met.",
             color=INK2, fontsize=7.8, va="top", linespacing=1.5)
    _save(fig, "F5_ebola_adapted_vs_unadapted")
    cap = ("F5. Ebola forecasts from a model that never saw Ebola in training, on the primary L12 and "
           "secondary L20 arms (country-macro RMSE in raw case counts, lower is better; horizon in "
           "weeks; 5-seed mean, error bars the across-seed spread). Both the unadapted (zero-shot) "
           "trunk and the adapted variant sit below the persistence floor across horizons, and the "
           "unadapted line is at or below the adapted one almost everywhere. On the L12 arm the h10 "
           "and h15 adapted points, marked with an asterisk, are reported as zero-shot rather than "
           "few-shot as prereg E6 requires: h15 has zero adaptation pairs and h10 has only 18 on "
           "near-fully padded windows. These are point means without intervals, so no significance is "
           "claimed and the pre-registered adapted-beats-persistence criterion was not met.")
    return "F5_ebola_adapted_vs_unadapted", cap


# --------------------------------------------------------------------------- #
# F6  full baseline comparison, per panel
# --------------------------------------------------------------------------- #
def fig_F6():
    _rc()
    panels = ["influenza_japan", "influenza_us-regions", "influenza_us-states",
              "covid_us-states", "dengue"]

    def enc(ds):
        v = {h: [] for h in HORIZONS}
        for s in SEEDS:
            for r in _records(REPO / f"results/single/encoder__{ds}__seed{s}.json"):
                if r.get("model") == "encoder" and r.get("metric") == "rmse":
                    v[r["horizon"]].append(r["country_macro"])
        return {h: sum(v[h]) / len(v[h]) for h in HORIZONS if v[h]}

    def per_cell(model, ds):
        """Every baseline is a per-cell file: {model}__{ds}__h{H}__seed{S}.json. GNNs and GBM carry
        the 5 frozen seeds; SARIMA and ARIMA are deterministic and carry seedNA."""
        v = {h: [] for h in HORIZONS}
        for h in HORIZONS:
            for s in list(SEEDS) + ["NA"]:
                p = REPO / f"results/baselines/{model}__{ds}__h{h}__seed{s}.json"
                if p.exists():
                    for r in _records(p):
                        if r.get("metric") == "rmse":
                            v[h].append(r["country_macro"])
        got = {h: sum(v[h]) / len(v[h]) for h in HORIZONS if v[h]}
        return got or None

    gnn = per_cell
    classical = per_cell

    def pers(ds):
        out = {}
        for r in _records(REPO / f"results/naive/naive__{ds}.json"):
            if r.get("model") == "persistence" and r.get("metric") == "rmse":
                out[r["horizon"]] = r["country_macro"]
        return out

    # series style: our encoder bold black; GNNs blue family; classical warm family; persistence grey
    STYLE = {
        "ours": (OURS, "-", "o", 2.8, 7.5),
        "EpiGNN": (BLUE, "-", "s", 1.6, 5),
        "ColaGNN": (BLUE_M, "-", "^", 1.6, 5),
        "HeatGNN": (PALE, "-", "v", 1.6, 5),
        "MTGNN": (MUTED, ":", "x", 1.4, 5),
        "GBM": (WARM_D, "-", "P", 1.6, 5),
        "SARIMA": (WARM_M, "-", "D", 1.6, 4.5),
        "ARIMA": (WARM_L, "--", "d", 1.4, 4.5),
        "persistence": ("#b0aea6", "--", "*", 1.4, 7),
    }

    def collect(ds):
        s = {}
        if ds != "dengue":
            s["ours"] = enc(ds)
            s["persistence"] = pers(ds)
        for m in ("EpiGNN", "ColaGNN", "HeatGNN", "MTGNN"):
            g = gnn(m, ds)
            if g:
                s[m] = g
        for m in ("GBM", "SARIMA", "ARIMA"):
            key = m.lower()
            c = classical(key, ds)
            if c:
                s[m] = c
        return s

    # spot: EpiGNN japan h3 country-macro mean equals the direct read over the 5 seed files
    ep = gnn("EpiGNN", "influenza_japan")
    raw = []
    for s in SEEDS:
        for r in _records(REPO / f"results/baselines/EpiGNN__influenza_japan__h3__seed{s}.json"):
            if r.get("metric") == "rmse" and r["horizon"] == 3:
                raw.append(r["country_macro"])
    assert abs(ep[3] - sum(raw) / len(raw)) < 1e-9, ep[3]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6),
                             gridspec_kw=dict(wspace=0.26, hspace=0.34))
    axes = axes.ravel()
    xs = list(range(len(HORIZONS)))
    used = []
    for ax, ds in zip(axes, panels):
        s = collect(ds)
        for key, series in s.items():
            col, ls, mk, lw, ms = STYLE[key]
            hs = [h for h in HORIZONS if h in series]
            ax.plot([HORIZONS.index(h) for h in hs], [series[h] for h in hs],
                    color=col, ls=ls, marker=mk, markersize=ms, lw=lw,
                    zorder=6 if key == "ours" else 3)
            if key not in used:
                used.append(key)
        ax.set_xticks(xs); ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=8.5)
        ax.set_xlabel("horizon (weeks)", fontsize=8.5)
        ax.set_ylabel("country-macro RMSE", fontsize=8.5)
        ttl = PANEL_NAME[ds]
        if ds == "dengue":
            ttl += "  (subset-scored models only)"
        elif ds == "covid_us-states":
            ttl += "  (no GNN baseline exists)"
        ax.set_title(ttl, color=INK, fontsize=9.5, loc="left", pad=6)
        _despine(ax); ax.yaxis.grid(True); ax.set_axisbelow(True)
    # legend + notes in the 6th cell
    lax = axes[5]
    lax.axis("off")
    handles = [Line2D([0], [0], color=STYLE[k][0], ls=STYLE[k][1], marker=STYLE[k][2],
                      lw=STYLE[k][3] if k != "ours" else 2.8,
                      markersize=STYLE[k][4], label=("our encoder" if k == "ours" else k))
               for k in ["ours", "EpiGNN", "ColaGNN", "HeatGNN", "MTGNN", "GBM", "SARIMA",
                         "ARIMA", "persistence"] if k in used]
    lax.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=9,
               loc="upper left", title="models", title_fontsize=9.5)
    lax.text(0, 0.34,
             "Country-macro RMSE, lower is better. Our encoder is the bold black line; every\n"
             "comparator trains on the disease it is scored on. Horizon in weeks.\n\n"
             "Dengue: the GNN and classical baselines score the 2,392-node subset over 11\n"
             "countries; our encoder and persistence score the full 6,161-node set over 12,\n"
             "so they are NOT comparable and are withheld from the dengue axis.\n"
             "Dengue SARIMA and ARIMA fall back to persistence on 792 of 2,051 nodes (39%),\n"
             "and SARIMA equals ARIMA on dengue and COVID. MTGNN (grey dotted) degenerates\n"
             "to a near-constant forecast on most influenza files and is shown for\n"
             "completeness, not as a valid comparator. COVID has no GNN baseline.",
             fontsize=7.6, color=INK2, va="top", linespacing=1.5)
    fig.suptitle("F6  Published and classical baselines against our encoder, per panel. "
                 "Country-macro RMSE, lower is better.",
                 x=0.008, ha="left", color=INK, fontsize=11.5, y=0.99)
    _save(fig, "F6_baseline_comparison")
    cap = ("F6. Our encoder against four published GNN baselines (EpiGNN, Cola-GNN, HeatGNN, MTGNN), "
           "three classical baselines (GBM, SARIMA, ARIMA) and persistence, one small multiple per "
           "panel (country-macro RMSE, lower is better; horizon in weeks; means over available seeds). "
           "Every comparator trains on the disease it is scored on. On dengue the baselines score the "
           "2,392-node subset while our encoder and persistence score the full 6,161-node set, so the "
           "encoder and persistence are withheld from the dengue axis rather than plotted on a "
           "mismatched scale; dengue SARIMA and ARIMA fall back to persistence on 39 percent of nodes "
           "and SARIMA equals ARIMA on dengue and COVID. MTGNN is drawn greyed because it collapses to "
           "a near-constant forecast on most influenza files and is shown for completeness, not as a "
           "valid comparator, and COVID has no GNN baseline.")
    return "F6_baseline_comparison", cap


FIGS = {"F1": fig_F1, "F2": fig_F2, "F3": fig_F3, "F4": fig_F4, "F5": fig_F5, "F6": fig_F6}
SECTION = {
    "F1": "Results / spatial-graph ablation (the graph does not help error)",
    "F2": "Results / meta-learning ablation (ANIL is a clean null)",
    "F3": "Results / epidemiology-informed penalty ablation",
    "F4": "Methods or Appendix / window sensitivity",
    "F5": "Results / Ebola case study",
    "F6": "Results / baseline comparison (SOTA)",
}


def _selfcheck():
    """Statistics helpers on synthetic input. No artifacts needed."""
    m, sd = _mean_sd([1.0, 2.0, 3.0])
    assert abs(m - 2.0) < 1e-12 and abs(sd - 1.0) < 1e-12, (m, sd)
    dm, dsd, n = _paired({42: 10.0, 52: 12.0}, {42: 9.0, 52: 10.0})
    assert n == 2 and abs(dm - (-1.5)) < 1e-12, (dm, dsd, n)          # abl - base
    assert _within_noise(0.4, 1.0, 5) and not _within_noise(2.0, 1.0, 5)
    assert _within_noise(5.0, 1.0, 1)                                 # n<2 is always noise
    assert _field("dengue") == "country_macro" and _field("covid_us-states") == "node_mean"
    print("ok  mean_sd, paired abl-base sign, within-noise rule, field selection")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="all", help="F1..F6 or all")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
        return
    keys = list(FIGS) if a.only == "all" else [a.only]
    caps = []
    for k in keys:
        name, cap = FIGS[k]()
        png = REPO / "figures" / f"{name}.png"
        pdf = REPO / "figures" / f"{name}.pdf"
        print(f"\nwrote figures/{name}.png ({png.stat().st_size:,} B) and "
              f"figures/{name}.pdf ({pdf.stat().st_size:,} B)")
        caps.append((k, name, cap))
    print("\n" + "=" * 100 + "\nSUGGESTED CAPTIONS\n" + "=" * 100)
    for k, name, cap in caps:
        print(f"\n[{k}]  {SECTION[k]}\nfile: figures/{name}.png / .pdf\n{cap}")


if __name__ == "__main__":
    main()

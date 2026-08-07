"""data_quality.py -- per-bundle data-quality audit + figures.

Answers one question: are the six harmonised bundles similar enough that ONE shared
encoder can read them as a single representation? Computes completeness, signal
quality, graph health and split geometry per bundle, then the cross-bundle
comparability block (channel schema, input-distribution distance) that decides it.

    python data_quality.py            # figs/quality/*.png + data_quality.md

Everything is read from data/processed/*.npz through bundles.load -- no re-derivation
from raw sources, so what is measured here is exactly what the trunk is fed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wasserstein_distance

import bundles as B

OUT = Path("figs/quality")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
# sequential = ONE hue, light -> dark (dataviz: never a rainbow for magnitude)
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#9ec3ec", BLUE, "#123a68"])
MISS = LinearSegmentedColormap.from_list("miss", ["#f7d9cb", BLUE])   # 0=missing 1=observed

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 8.5, "axes.titlesize": 9.5, "figure.dpi": 130,
})


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _runs(row: np.ndarray, value) -> list[int]:
    """Lengths of maximal runs equal to `value` in a 1-D array."""
    eq = row == value
    if not eq.any():
        return []
    d = np.diff(np.concatenate(([0], eq.view(np.int8), [0])))
    return (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist()


def _interior_gap(m_row: np.ndarray) -> int:
    """Longest unobserved run BETWEEN the node's first and last observation.

    Leading/trailing unobserved stretches are a shorter series, not a hole -- counting
    them as gaps would make a late-onset node look like a broken one.
    """
    obs = np.flatnonzero(m_row)
    if obs.size < 2:
        return 0
    inner = m_row[obs[0]:obs[-1] + 1]
    runs = _runs(inner, 0)
    return max(runs) if runs else 0


def metrics(b: B.Bundle) -> dict:
    N, T, F = b.X.shape
    M = b.M.astype(bool)
    raw = b.raw.astype(float)
    dates = pd.DatetimeIndex(b.meta["dates"])
    obs_per_node = M.sum(1)
    live = obs_per_node > 0                       # nodes with any observation at all

    v = raw[M]                                    # observed target values, count space
    zero = (v == 0)
    # per-node dispersion on observed cells only; nodes with <2 obs are undefined
    cv, flat = [], []
    for i in range(N):
        x = raw[i][M[i]]
        if x.size >= 2:
            cv.append(x.std() / x.mean() if x.mean() > 0 else np.nan)
            # flat runs: consecutive identical OBSERVED values -- carried/derived reporting
            fr = [r for r in _runs(np.diff(x), 0) if r >= 2]
            flat.append(sum(r + 1 for r in fr) / x.size)
    cv, flat = np.array(cv, float), np.array(flat, float)

    A = b.A_geo.copy()
    np.fill_diagonal(A, 0)
    deg = (A > 0).sum(1)

    enc = b.transfer_view()[:, :, 0][M]           # what the shared encoder actually reads
    inp_zero = float((b.X[:, :, 0] == 0).mean())  # incl. unobserved cells, which ship as 0.0

    return {
        "name": b.name, "N": N, "T": T, "F": F,
        "t_res": b.meta.get("t_res", "weekly"), "steps_per_year": b.meta.get("steps_per_year"),
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "years": round((dates[-1] - dates[0]).days / 365.25, 1),
        "cells": N * T, "observed": int(M.sum()),
        "completeness": float(M.mean()),
        "node_completeness": obs_per_node / T,
        "dead_nodes": int((~live).sum()),
        "median_node_completeness": float(np.median(obs_per_node[live] / T)) if live.any() else 0.0,
        "worst_node_completeness": float((obs_per_node[live] / T).min()) if live.any() else 0.0,
        "gaps": np.array([_interior_gap(b.M[i]) for i in range(N)]),
        "zero_share": float(zero.mean()),
        "input_zero_share": inp_zero,
        "cv": cv, "flat_share": float(np.nanmean(flat)) if flat.size else np.nan,
        "zero_var_nodes": int((np.nan_to_num(cv) == 0).sum()),
        "tail_ratio": float(np.percentile(v, 99) / max(np.median(v[v > 0]), 1e-9)) if v.size else np.nan,
        "max_count": float(v.max()) if v.size else np.nan,
        "deg": deg, "isolated_nodes": int((deg == 0).sum()),
        "graph_density": float((A > 0).mean()),
        "splits": {p: int(m.sum()) for p, m in b.masks().items()},
        "origins": {p: len(b.origins(phase=p)) for p in b.masks()},
        "origins_total": len(b.origins()),
        "features": b.meta["feature_names"],
        "enc": enc.astype(np.float32),
        "enc_mean": float(enc.mean()), "enc_std": float(enc.std()),
        "raw": raw, "M": M, "dates": dates,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _binned(mat: np.ndarray, max_rows: int = 320) -> np.ndarray:
    """Row-bin a [N,T] matrix so a 7,165-node panel is still one screen of pixels."""
    N = mat.shape[0]
    if N <= max_rows:
        return mat
    edges = np.linspace(0, N, max_rows + 1).astype(int)
    return np.stack([mat[a:b].mean(0) for a, b in zip(edges[:-1], edges[1:]) if b > a])


def panel(m: dict) -> Path:
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=0.85, wspace=0.3)
    d, M = m["dates"], m["M"]

    # 1 -- observation map: the completeness question, answered as a picture
    ax = fig.add_subplot(gs[0, :])
    order = np.argsort(-M.sum(1))                     # densest node at the top
    ax.imshow(_binned(M[order].astype(float)), aspect="auto", cmap=MISS, vmin=0, vmax=1,
              interpolation="nearest",
              extent=[0, m["T"], m["N"], 0])
    ax.set_title(f"Observation map — {m['completeness']:.1%} of {m['cells']:,} cells observed "
                 f"(blue = observed, pink = missing); nodes sorted by completeness", loc="left")
    ax.set_ylabel("node"); ax.set_xlabel("")
    tk = np.linspace(0, m["T"] - 1, 6).astype(int)
    ax.set_xticks(tk); ax.set_xticklabels([str(d[i].date()) for i in tk], fontsize=7)

    # 2 -- per-node completeness
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(m["node_completeness"] * 100, bins=25, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(m["median_node_completeness"] * 100, color=ORANGE, lw=2)
    ax.set_title("Completeness per node", loc="left")
    ax.set_xlabel("% of weeks observed"); ax.set_ylabel("nodes")
    ax.text(0.97, 0.92, f"median {m['median_node_completeness']:.0%}", transform=ax.transAxes,
            ha="right", color=ORANGE, fontsize=8)

    # 3 -- interior gaps
    ax = fig.add_subplot(gs[1, 1])
    g = m["gaps"]
    ax.hist(g, bins=max(1, min(25, int(g.max()) + 1)), color=BLUE,
            edgecolor="white", linewidth=0.4)
    ax.set_title("Longest interior gap per node", loc="left")
    ax.set_xlabel("consecutive missing weeks"); ax.set_ylabel("nodes")
    ax.text(0.97, 0.92, f"max {int(g.max())}w  ·  {(g > 4).mean():.0%} of nodes >4w",
            transform=ax.transAxes, ha="right", color=INK2, fontsize=8)

    # 4 -- target distribution (log): tail weight and zero inflation
    ax = fig.add_subplot(gs[1, 2])
    v = m["raw"][M]
    pos = v[v > 0]
    ax.hist(np.log10(pos), bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_title("Observed incidence (log₁₀, non-zero)", loc="left")
    ax.set_xlabel("log₁₀ cases"); ax.set_ylabel("cells")
    ax.text(0.97, 0.92, f"{m['zero_share']:.0%} of observed cells are 0\n"
                        f"p99/median = {m['tail_ratio']:.0f}×",
            transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8)

    # 5 -- epidemic curve with coverage
    ax = fig.add_subplot(gs[2, 0])
    tot = np.where(M, m["raw"], np.nan)
    ax.plot(d, np.nansum(tot, 0), color=BLUE, lw=1.6)
    ax.set_title("Total observed incidence per week", loc="left")
    ax.set_ylabel("cases"); ax.tick_params(axis="x", labelsize=7)
    ax2 = ax.twiny(); ax2.axis("off")                # keep a single y-axis (no dual scale)

    # 6 -- graph degree
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(m["deg"], bins=25, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_title("Geographic-graph degree", loc="left")
    ax.set_xlabel("neighbours (self-loop excluded)"); ax.set_ylabel("nodes")
    ax.text(0.97, 0.92, f"isolated {m['isolated_nodes']}  ·  density {m['graph_density']:.3f}",
            transform=ax.transAxes, ha="right", color=INK2, fontsize=8)

    # 7 -- split geometry
    ax = fig.add_subplot(gs[2, 2])
    ph = list(m["splits"])
    ax.bar(ph, [m["splits"][p] for p in ph], color=BLUE, width=0.6)
    for i, p in enumerate(ph):
        ax.text(i, m["splits"][p], f"{m['splits'][p]:,}\n{m['origins'][p]} origins*",
                ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_title("Split geometry (observed cells)", loc="left")
    ax.set_xlabel("*an origin can serve targets in two phases, so these overlap",
                  fontsize=6.5)
    ax.margins(y=0.28); ax.set_ylabel("cells")

    fig.suptitle(f"{m['name']}  —  N={m['N']}  T={m['T']} {m['t_res']} steps  "
                 f"({m['start']} → {m['end']}, {m['years']}y)  ·  F={m['F']} channels",
                 x=0.008, ha="left", fontsize=12, weight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{m['name']}.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def compare(ms: list[dict]) -> Path:
    names = [m["name"] for m in ms]
    short = [n.replace("influenza_", "flu ").replace("covid_", "covid ").replace("_", " ")
             for n in names]
    fig = plt.figure(figsize=(11.5, 7.6))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.35)

    def barh(ax, vals, title, fmt, note=""):
        o = np.argsort(vals)
        ax.barh(np.arange(len(vals)), np.array(vals)[o], color=BLUE, height=0.62)
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels([short[i] for i in o], fontsize=7.5)
        for j, i in enumerate(o):
            ax.text(vals[i], j, " " + fmt(vals[i]), va="center", fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left"); ax.margins(x=0.22)
        if note:
            ax.set_xlabel(note, fontsize=7)

    barh(fig.add_subplot(gs[0, 0]), [m["completeness"] for m in ms],
         "Completeness (observed cells)", lambda x: f"{x:.0%}")
    barh(fig.add_subplot(gs[0, 1]), [m["zero_share"] for m in ms],
         "Zero-inflation (observed cells = 0)", lambda x: f"{x:.0%}")
    barh(fig.add_subplot(gs[0, 2]), [m["input_zero_share"] for m in ms],
         "Encoder input degeneracy", lambda x: f"{x:.0%}",
         "share of X[:,:,0] exactly 0.0 — the covariate shift the trunk sees")
    barh(fig.add_subplot(gs[1, 0]), [m["N"] for m in ms], "Nodes (N)", lambda x: f"{int(x):,}")
    barh(fig.add_subplot(gs[1, 1]), [m["T"] for m in ms], "Time steps (T)", lambda x: f"{int(x):,}")
    barh(fig.add_subplot(gs[1, 2]), [m["origins_total"] for m in ms],
         "Training origins available", lambda x: f"{int(x):,}",
         "windows of W=20 with all horizons in range")

    # encoder-view distributions: the "single representation" question, drawn
    ax = fig.add_subplot(gs[2, :2])
    for i, m in enumerate(ms):
        x = m["enc"]
        x = x[np.random.default_rng(0).integers(0, x.size, min(x.size, 200_000))]
        ax.hist(x, bins=np.linspace(-2, 6, 65), histtype="step", lw=1.7,
                color=SERIES[i % 6], label=short[i], density=True)
    ax.set_yscale("log")
    ax.set_title("What the shared encoder reads: normalised incidence, observed cells", loc="left")
    ax.set_xlabel("z-scored incidence (channel 0)"); ax.set_ylabel("density (log)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    # pairwise distance between those distributions
    ax = fig.add_subplot(gs[2, 2])
    rng = np.random.default_rng(0)
    samp = [m["enc"][rng.integers(0, m["enc"].size, min(m["enc"].size, 60_000))] for m in ms]
    D = np.zeros((len(ms), len(ms)))
    for i in range(len(ms)):
        for j in range(len(ms)):
            if i < j:
                D[i, j] = D[j, i] = wasserstein_distance(samp[i], samp[j])
    im = ax.imshow(D, cmap=SEQ)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(short, rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(short, fontsize=6.5)
    for i in range(len(ms)):
        for j in range(len(ms)):
            ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if D[i, j] > D.max() * 0.62 else INK)
    ax.set_title("Distribution distance (Wasserstein)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    fig.suptitle("Cross-bundle comparability — can one encoder read all six?",
                 x=0.008, ha="left", fontsize=12, weight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "_comparison.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, D


# --------------------------------------------------------------------------- #
def _md(df: pd.DataFrame) -> str:
    """Markdown table. pandas' own to_markdown needs `tabulate`; three lines needs nothing."""
    head = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = [[str(i)] + [str(v) for v in r] for i, r in zip(df.index, df.to_numpy())]
    return "\n".join(["| " + " | ".join(head) + " |",
                      "|" + "|".join(["---"] * len(head)) + "|",
                      *["| " + " | ".join(r) + " |" for r in rows]])


def report(ms: list[dict], D: np.ndarray) -> str:
    rows = []
    for m in ms:
        rows.append({
            "bundle": m["name"], "N": m["N"], "T": m["T"], "cadence": m["t_res"],
            "span": f"{m['start']}→{m['end']}", "years": m["years"],
            "complete%": round(m["completeness"] * 100, 1),
            "median node%": round(m["median_node_completeness"] * 100, 1),
            "max gap (w)": int(m["gaps"].max()),
            "zeros%": round(m["zero_share"] * 100, 1),
            "input 0%": round(m["input_zero_share"] * 100, 1),
            "flat runs%": round(m["flat_share"] * 100, 1),
            "zero-var nodes": m["zero_var_nodes"],
            "p99/median": round(m["tail_ratio"], 1),
            "isolated": m["isolated_nodes"],
            "origins": m["origins_total"],
            "channels": m["F"],
        })
    df = pd.DataFrame(rows).set_index("bundle")
    names = [m["name"] for m in ms]
    dist = pd.DataFrame(D.round(3), index=pd.Index(names, name="bundle"), columns=names)
    md = ["# Dataset quality audit\n",
          f"Generated by `data_quality.py` from `data/processed/*.npz`. Figures: `{OUT}/`.\n",
          "## Per-bundle metrics\n", _md(df),
          "\n\n## Encoder-view distribution distance (Wasserstein, z-space)\n", _md(dist),
          "\n"]
    return "\n".join(md)


def main():
    ms = [metrics(B.load(n)) for n in B.BUNDLE_NAMES]
    for m in ms:
        print("wrote", panel(m))
    p, D = compare(ms)
    print("wrote", p)
    Path("progress/planning/data_quality.md").write_text(report(ms, D), encoding="utf-8")
    print("wrote progress/planning/data_quality.md")
    print(json.dumps({m["name"]: {k: round(float(m[k]), 4) for k in
                                  ("completeness", "zero_share", "input_zero_share")}
                      for m in ms}, indent=1))


def _selfcheck():
    """The two hand-rolled bits: run-length gaps and interior-only gap accounting."""
    assert _runs(np.array([1, 1, 0, 1, 1, 1]), 1) == [2, 3]
    assert _runs(np.array([0, 0, 0]), 1) == []
    assert _interior_gap(np.array([0, 0, 1, 0, 0, 0, 1, 0], dtype=np.uint8)) == 3
    assert _interior_gap(np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.uint8)) == 0, \
        "trailing unobserved is a shorter series, not a gap"
    assert _interior_gap(np.array([1], dtype=np.uint8)) == 0
    x = np.arange(12).reshape(6, 2).astype(float)
    assert _binned(x, 3).shape == (3, 2) and np.allclose(_binned(x, 3)[0], [1, 2])
    assert _binned(x, 99).shape == (6, 2), "no binning when N is already small"
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
    main()
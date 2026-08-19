"""Work order 8e -- the learned graph gate as a figure.

The client's ask (Review Doc.md): "the learned graph gate is already an interpretable result and
we're under-using it. Gate values of 0.27 to 0.60 scaling with region count and density is a
figure. Build it."

Reads the per-node gate readouts already on disk (`results/single/encoder__{ds}__seed{S}__gate.npz`,
5 panels x 5 seeds, written by train.loop.gate_spatial_readout) and the bundle adjacency for the
graph geometry. No model, no GPU, no training -- everything this needs was archived at train time.

  python -m gate_figure              # writes figures/gate.png + gate.pdf, prints the table
  python -m gate_figure --selfcheck  # aggregation + geometry asserts, needs no artifacts

THE HONEST READ, and it is printed under the figure rather than left for a reviewer to find:
mean degree is near-constant across the five panels (3.2 to 5.7), so density = 2E/N(N-1) is
essentially 1/N here. Panels B and C are therefore ONE ordering shown twice, not two independent
pieces of evidence for the same claim.
"""
import argparse
import sys

import numpy as np

from results_paths import rpath

DEV = ["influenza_us-regions", "influenza_japan", "influenza_us-states",
       "covid_us-states", "dengue"]
SEEDS = (42, 52, 62, 72, 82)
SHORT = {"influenza_us-regions": "flu us-regions", "influenza_japan": "flu japan",
         "influenza_us-states": "flu us-states", "covid_us-states": "covid us-states",
         "dengue": "dengue"}

# dataviz house tokens, light surface. One hue: every point is direct-labelled, so colour carries
# no identity and a 5-slot categorical palette would fail the all-pairs CVD gate for nothing.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"


def density(A):
    """Undirected edge density, self-loops excluded. Returns (n_nodes, n_edges, density, mean_deg)."""
    N = A.shape[0]
    off = A.copy()
    np.fill_diagonal(off, 0)
    nz = int((off > 0).sum())            # directed count; A is symmetric so this is 2E
    return N, nz, nz / (N * (N - 1)), nz / N


def collect(names=DEV, seeds=SEEDS, prefix="encoder"):
    """Per-panel gate readout. Keeps the per-SEED means (the error bars) and the pooled per-node
    vector (the distribution) -- analysis.gate_reads pools both away, which is why this reloads."""
    import bundles
    out = []
    for name in names:
        per_seed, nodes = [], []
        for s in seeds:
            p = rpath(f"{prefix}__{name}__seed{s}__gate.npz")
            if not p.exists():
                continue
            g = np.load(p)["mean_g"].astype(np.float64)
            per_seed.append(g.mean())
            nodes.append(g)
        if not per_seed:
            print(f"  skip {name}: no gate npz", file=sys.stderr)
            continue
        N, E, dens, deg = density(bundles.load(name).A_geo)
        pooled = np.concatenate(nodes)
        assert nodes[0].shape[0] == N, f"{name}: gate readout is {nodes[0].shape[0]} nodes, graph is {N}"
        out.append(dict(name=name, short=SHORT[name], n_seeds=len(per_seed),
                        g=float(np.mean(per_seed)),
                        sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                        N=N, edges=E // 2, density=dens, mean_deg=deg,
                        pooled=pooled, q=np.percentile(pooled, [25, 50, 75])))
    return out


def _label_groups(rows, key, gtol=0.02):
    """One label per VISUAL point. Panels that coincide in x and (to gtol) in g get a single
    shared-suffix label -- two labels stacked on one marker read as a rendering bug, and the
    coincidence here is the finding (same graph, different disease, same gate)."""
    groups = {}
    for r in rows:
        k = (r[key], round(r["g"] / gtol))
        groups.setdefault(k, []).append(r)
    out = []
    for members in groups.values():
        x = members[0][key]
        g = float(np.mean([m["g"] for m in members]))
        if len(members) == 1:
            out.append((x, g, members[0]["short"]))
            continue
        words = [m["short"].split() for m in members]
        n = 0                                   # longest common trailing word run
        while n < min(len(w) for w in words) and len({w[-(n + 1)] for w in words}) == 1:
            n += 1
        stem = " ".join(words[0][len(words[0]) - n:]) if n else "+".join(w[0] for w in words)
        out.append((x, g, f"{stem} ×{len(members)}"))
    return out


def table(rows):
    print(f"\n{'=' * 104}\nlearned graph gate g, single-disease encoder, mean over "
          f"{len(SEEDS)} seeds +- seed sd\n{'=' * 104}")
    print(f"  {'panel':18} {'nodes':>6} {'edges':>7} {'density':>9} {'mean deg':>9} "
          f"{'g mean':>8} {'seed sd':>8} {'per-node IQR':>18}")
    for r in rows:
        print(f"  {r['short']:18} {r['N']:6d} {r['edges']:7d} {r['density']:9.4f} "
              f"{r['mean_deg']:9.2f} {r['g']:8.3f} {r['sd']:8.3f} "
              f"  [{r['q'][0]:.3f}, {r['q'][2]:.3f}]")
    degs = [r["mean_deg"] for r in rows]
    print(f"\n  mean degree spans {min(degs):.1f}..{max(degs):.1f} while node count spans "
          f"{min(r['N'] for r in rows)}..{max(r['N'] for r in rows)}, so density ~ 1/N on this set.")
    print("  Panels B and C are the same ordering twice, NOT two independent confirmations.")


def figure(rows, out="figures/gate"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    })
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.0, 4.5),
                                        gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.28))

    # ---- A: per-node distribution, pooled over seeds ----------------------------------------- #
    order = sorted(rows, key=lambda r: r["N"])
    bp = axA.boxplot([r["pooled"] for r in order], widths=0.5, showfliers=False,
                     patch_artist=True, medianprops=dict(color=SURFACE, linewidth=1.6),
                     boxprops=dict(facecolor=BLUE, edgecolor="none"),
                     whiskerprops=dict(color=AXIS, linewidth=1.0),
                     capprops=dict(color=AXIS, linewidth=1.0))
    for b in bp["boxes"]:
        b.set_alpha(0.85)
    axA.set_xticks(range(1, len(order) + 1))
    axA.set_xticklabels([f"{r['short']}\nN={r['N']:,}" for r in order], fontsize=8)
    axA.set_ylabel("per-node gate  $g$")
    axA.set_title("A · gate distribution across nodes", loc="left", color=INK, pad=10)
    axA.set_ylim(0, 1)
    axA.yaxis.grid(True); axA.set_axisbelow(True)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # ---- B and C: the two scalings the client asked for ---------------------------------------#
    # Label offsets in points, per panel: the x-ordering reverses between B and C (density ~ 1/N),
    # so one offset table cannot serve both. Coincident panels get ONE label -- flu and covid
    # us-states sit on the same graph (N=49, 206 edges) and the same gate, so two labels on one
    # marker would read as a collision rather than as the finding it is.
    # Rule in both panels: the two points that sit at the same height (us-regions and us-states,
    # both g ~ 0.37) go opposite ways, or their labels overlap -- they are only ~20% of the panel
    # width apart and each label is wider than that gap.
    offs = {"B": {"flu us-regions": (0, -21, "center"), "flu japan": (0, -21, "center"),
                  "us-states ×2": (0, 14, "center"), "dengue": (0, 14, "center")},
            "C": {"flu us-regions": (-12, -21, "center"), "flu japan": (0, -21, "center"),
                  "us-states ×2": (0, 14, "center"), "dengue": (0, 14, "center")}}
    for ax, key, xlab, title, pad, tag in (
            (axB, "N", "region count  $N$  (log)", "B · gate vs region count", 2.2, "B"),
            (axC, "density", "graph density  $2E/N(N{-}1)$  (log)", "C · gate vs density", 2.2, "C")):
        for r in rows:
            ax.errorbar(r[key], r["g"], yerr=r["sd"], fmt="o", markersize=9,
                        color=BLUE, ecolor=AXIS, elinewidth=1.4, capsize=3,
                        markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3)
        for x, g, text in _label_groups(rows, key):
            dx, dy, ha = offs[tag][text]
            ax.annotate(text, (x, g), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, fontsize=8, color=INK2, zorder=4)
        xs = [r[key] for r in rows]
        ax.set_xscale("log")
        ax.set_xlim(min(xs) / pad, max(xs) * pad)      # keep the extreme points off the spines
        ax.set_xlabel(xlab)
        ax.set_ylim(0, 0.8)
        ax.set_title(title, loc="left", color=INK, pad=10)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axB.set_ylabel("mean gate  $g$   (5 seeds, $\\pm$ sd)")

    fig.text(0.008, 0.015,
             "Learned spatial gate $g$, single-disease encoder, 5 seeds per panel. Error bars are the "
             "seed sd; box A is the per-node spread pooled over seeds (IQR, whiskers 1.5×IQR, fliers "
             "hidden).\n"
             "flu us-states and covid us-states share an identical graph (N=49, 206 edges) and land on "
             "the same gate (0.373 vs 0.375) — the gate reads graph structure, not disease. "
             "Mean degree is 3.2–5.7 across all\nfive panels, so density ≈ 1/N here: B and C are one "
             "ordering shown twice, not independent evidence. The four panels below N=50 are within "
             "noise of each other except japan;\nthe rise is carried by dengue alone, so B and C show "
             "a range across graph sizes, not a fitted trend over five points.",
             fontsize=7.5, color=MUTED, va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.30)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"\n  wrote {out}.png and {out}.pdf")


def _selfcheck():
    """Aggregation and geometry, on synthetic input. No artifacts needed."""
    # (1) density + mean degree on a known graph: a 4-cycle has 4 edges, density 4/6.
    A = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], float)
    N, nz, d, deg = density(A)
    assert (N, nz) == (4, 8) and abs(d - 8 / 12) < 1e-12 and abs(deg - 2.0) < 1e-12, (N, nz, d, deg)
    # control: self-loops must not count as edges, or every density is inflated by 1/(N-1).
    Al = A.copy(); np.fill_diagonal(Al, 1.0)
    assert density(Al)[1] == 8, "self-loops leaked into the edge count"

    # (2) the error bar is the spread of SEED means, not of nodes. Two seeds whose node vectors
    #     differ wildly but whose means agree must give sd 0 -- getting this backwards would print
    #     the node spread as run-to-run variation and overstate it by an order of magnitude.
    per_seed = [np.array([0.0, 1.0]).mean(), np.array([0.5, 0.5]).mean()]
    assert abs(np.std(per_seed, ddof=1)) < 1e-12, "seed sd picked up within-seed node spread"
    # control: genuinely different seed means must register.
    assert np.std([0.2, 0.6], ddof=1) > 0.2, "seed sd is not measuring anything"

    # (3) ddof=1 -- the 5-seed sd is a sample sd. ddof=0 understates it by sqrt(4/5) = 11%.
    x = [0.30, 0.35, 0.40, 0.45, 0.50]
    assert abs(np.std(x, ddof=1) - 0.0790569) < 1e-6, np.std(x, ddof=1)
    assert np.std(x, ddof=0) < np.std(x, ddof=1), "ddof control inverted"

    # (4) coincident panels collapse to one label; distinct ones must NOT be merged.
    mk = lambda s, n, g: dict(short=s, N=n, g=g)
    got = dict((t, (x, round(y, 3))) for x, y, t in _label_groups(
        [mk("flu us-states", 49, 0.373), mk("covid us-states", 49, 0.375),
         mk("flu japan", 47, 0.271)], "N"))
    assert got == {"us-states ×2": (49, 0.374), "flu japan": (47, 0.271)}, got
    # control: same x but gates far apart is a real difference and must stay two labels.
    two = _label_groups([mk("a x", 49, 0.20), mk("b x", 49, 0.60)], "N")
    assert len(two) == 2, two

    print("ok  density excludes self-loops; error bars are seed-level; sample sd (ddof=1); "
          "coincident labels merge, distinct ones do not")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--prefix", default="encoder")
    ap.add_argument("--out", default="figures/gate")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    else:
        rows = collect(prefix=a.prefix)
        if not rows:
            sys.exit("no gate archives found")
        table(rows)
        figure(rows, out=a.out)

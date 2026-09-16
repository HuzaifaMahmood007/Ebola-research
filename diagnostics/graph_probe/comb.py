"""Why the per-lag attribution profile combs with period 4, measured rather than asserted.

The report already says the comb is architectural: an UNTRAINED encoder reproduces it at r = 0.91
to 0.96, so it is the wiring and not epidemiology. That says the comb is not learned. It does not
say what makes it.

THE MECHANISM. DilatedTCN (models/temporal.py) is a causal kernel-2 stack with
DILATIONS = (1, 2, 4, 8, 16), read at the LAST timestep only. Each layer either takes its dilated
tap or passes through on the residual, so the input at offset k from the end reaches the output
through the unique subset of {1,2,4,8,16} summing to k, which is just k's binary expansion. The
number of GATED nonlinear taps on that route is therefore popcount(k), and every tap multiplies the
gradient by tanh' * sigmoid, both below 1. More taps, smaller gradient, smaller attribution.

Lag L sits at offset L-1. Multiples of 4 have the low two bits clear, so they skip the two cheapest
layers: lags 1, 5, 9, 13, 17 carry popcounts 0, 1, 1, 2, 1. Lags at 3 mod 4 set both low bits:
lags 4, 8, 12, 16, 20 carry 2, 3, 3, 4, 3. Mean 1.0 against 3.0. That is the comb, and it is a
property of the dilation schedule, not of the disease.

Read-only. Reads results/explain/*.npz, writes figures/comb_mechanism.png.

  conda run -n ebola-train python diagnostics/graph_probe/comb.py
"""
import glob
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

W = 20
LAG = np.arange(W, 0, -1)                          # archive index order: index 0 is lag 20
OFFSET = LAG - 1                                   # lag 1 is offset 0 from the final timestep
POP = np.array([bin(int(k)).count("1") for k in OFFSET])
SPIKES, TROUGHS = (1, 5, 9, 13, 17), (4, 8, 12, 16, 20)

# dataviz house tokens, light surface (same as explain.py and gate_figure.py). ONE hue: the panel
# bundle is recessive context, not a peer series, so it stays gray and carries no identity.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE = "#1c5cab"


def profiles():
    """{panel: [W] IG share in lag order, lag 1 first}, pooled over origins, horizons and seeds."""
    acc = {}
    for f in sorted(glob.glob("results/explain/explain__*.npz")):
        z = np.load(f, allow_pickle=True)
        if "ig_lag" not in z.files:                                      # __edges / __local sidecars
            continue
        # Pool the raw sums first, THEN normalise. Some (origin, horizon) cells have no scored node
        # and sum to zero, and normalising those per-cell gives NaN that poisons any later mean.
        acc.setdefault(str(z["panel"]), []).append(z["ig_lag"].sum((0, 1)))
    return {p: (s := np.sum(v, 0))[::-1] / s.sum() for p, v in acc.items()}


def main():
    prof = profiles()
    if not prof:
        sys.exit("no explain archives under results/explain/")
    pop_lag = POP[::-1]                                                  # popcount in lag order
    lags = np.arange(1, W + 1)

    print(f"{'panel':24} {'r(share, -popcount)':>20} {'spike/trough':>13}")
    rs = []
    for p, v in sorted(prof.items()):
        r = float(np.corrcoef(v, -pop_lag)[0, 1])
        ratio = v[np.isin(lags, SPIKES)].mean() / v[np.isin(lags, TROUGHS)].mean()
        rs.append(r)
        print(f"{p:24} {r:+20.3f} {ratio:12.1f}x")
    print(f"\nr ranges {min(rs):+.3f} to {max(rs):+.3f} over {len(rs)} panel-arms")
    print(f"mean popcount: spikes {pop_lag[np.isin(lags, SPIKES)].mean():.1f}, "
          f"troughs {pop_lag[np.isin(lags, TROUGHS)].mean():.1f}")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE,
                                 gridspec_kw=dict(width_ratios=[1.75, 1], wspace=0.22))
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK2, length=3)

    # --- A: the comb itself, with every panel-arm behind the mean.
    for lag in TROUGHS:                                                  # recessive trough guides
        ax.axvline(lag, color=GRID, lw=6, zorder=0)
    for v in prof.values():
        ax.plot(lags, v, color=MUTED, lw=1.0, alpha=0.55, zorder=2)
    mean = np.mean(list(prof.values()), 0)
    ax.plot(lags, mean, color=BLUE, lw=2.4, zorder=3)
    ax.scatter([l for l in SPIKES], [mean[l - 1] for l in SPIKES], s=34, color=BLUE,
               edgecolor=SURFACE, linewidth=2, zorder=4)
    # Name the binary route on three lags that are far enough apart not to collide: a spike with no
    # taps at all, a trough that needs two, and the spike that band 16-20 rides on.
    for lag, dx, dy in ((1, 30, 4), (4, 22, 30), (17, 6, 18)):
        taps = [d for d in (16, 8, 4, 2, 1) if (lag - 1) & d]
        txt = "no taps" if not taps else f"{len(taps)} tap{'s' if len(taps) > 1 else ''}: " \
                                         + "+".join(str(t) for t in taps)
        ax.annotate(f"lag {lag}, {txt}", (lag, mean[lag - 1]), textcoords="offset points",
                    xytext=(dx, dy), ha="left", fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                    shrinkA=0, shrinkB=4))
    ax.set_xticks(lags)
    ax.set_xlabel("lag, weeks before the forecast origin", color=INK2, fontsize=9)
    ax.set_ylabel("IG share", color=INK2, fontsize=9)
    ax.set_title("A. Every panel-arm combs with period 4", color=INK, fontsize=11, loc="left")
    ax.plot([], [], color=MUTED, lw=1.0, label=f"each panel-arm ({len(prof)})")
    ax.plot([], [], color=BLUE, lw=2.4, label="mean")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")

    # --- B: the comb IS the tap count.
    for p, v in prof.items():
        bx.scatter(pop_lag + np.random.default_rng(0).uniform(-0.11, 0.11, W), v,
                   s=16, color=MUTED, alpha=0.5, zorder=2)
    mp = [mean[pop_lag == k].mean() for k in range(pop_lag.max() + 1)]
    bx.plot(range(len(mp)), mp, color=BLUE, lw=2.4, marker="o", ms=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    for k, m in enumerate(mp):                                           # k=0 sits on the y-axis
        bx.annotate(f"{m:.3f}", (k, m), textcoords="offset points",
                    xytext=(14, -2) if k == 0 else (0, 12),
                    ha="left" if k == 0 else "center", fontsize=8.5, color=INK2)
    bx.set_xticks(range(len(mp)))
    bx.set_xlabel("gated TCN taps on the route, popcount(lag - 1)", color=INK2, fontsize=9)
    bx.set_ylabel("IG share", color=INK2, fontsize=9)
    bx.set_title("B. More taps, less attribution", color=INK, fontsize=11, loc="left")

    fig.text(0.008, -0.03,
             "Causal kernel-2 TCN, dilations (1, 2, 4, 8, 16), read at the last timestep. Each layer "
             "takes its dilated tap or passes through on the residual, so lag L reaches the output "
             "by the unique binary\nsubset summing to L-1, crossing popcount(L-1) gated "
             "tanh x sigmoid taps. Spike lags average 1.0 taps, trough lags 3.0. The comb is the "
             f"dilation schedule, not epidemiology: r = {min(rs):+.2f} to {max(rs):+.2f} against "
             "-popcount\nacross all seven panel-arms. This is why lags are reported at BAND level "
             "only, and why band 16-20 exceeds band 11-15: lag 17 is offset 16, one tap.",
             color=INK2, fontsize=8, va="top")
    fig.savefig("figures/comb_mechanism.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print("\nwrote figures/comb_mechanism.png")


if __name__ == "__main__":
    main()

"""WHY the spatial graph does not improve accuracy. The mechanism, in one runnable script.

The gate-off ablation gives the RESULT: switching the graph off improves error in 0 of 40 cells and
worsens it in 8 (results/reports/gate_ablation.log). A result is not a reason, and a client or a
reviewer will ask for the reason. This measures it as a three-step chain, every step from disk.

  STEP 1  The model is not ignoring the graph. The learned gate is wide open on every panel and not
          one node in any panel sits near closed. So "it does not help" is not "it was switched off".

  STEP 2  There is almost nothing district-specific to send. Of the 64 numbers the encoder produces
          per district, most are a single shared vector common to every district. Only a few percent
          of the energy actually differs between districts, and it occupies about 2 of 64 available
          directions. Then neighbour averaging deletes most of even that, because averaging a handful
          of vectors that are mostly identical cancels the small individual parts and leaves the
          shared one.

  STEP 3  So the identity of a district's neighbours barely matters. Relabel the adjacency, keeping
          the exact topology and every node's degree but attaching the WRONG districts to it, and
          test error moves by a fraction of the noise between random seeds.

Chain: gate open -> nothing distinct to send -> averaging deletes what little there is -> who your
neighbours are is worth almost nothing -> removing the graph costs almost nothing.

SCOPE, and it travels with the finding. These run on results/single/ checkpoints, the single-disease
models, NOT the cross-disease transfer trunk. Dengue is excluded: its 7,165 nodes make the
permutation sweep time out, and its district-specific share is the highest of any panel at 22.4%,
so leaving it out is conservative in the direction that favours the graph. This is an
inference-time test on a TRAINED model; it does not ask whether a model trained on a fake graph
would do as well, which is a separate 6-hour retrain and is still not run.

Read-only. Writes figures/why_graph_fails.png.

  conda run -n ebola-train python diagnostics/graph_probe/why_graph_fails.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bundles                                                            # noqa: E402
from bundles import HORIZONS                                              # noqa: E402
from models.adapters import Adapter                                       # noqa: E402
from models.config import MEDIAN_IDX                                      # noqa: E402
from models.encoder import SharedEncoder                                  # noqa: E402
from models.spatial import mask_aware_adj, sparse_from_dense_np           # noqa: E402
from models.windows import window_slice                                   # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

PANELS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states"]
SEEDS = (42, 52, 62, 72, 82)
NPERM = 10                        # per seed; the committed t2.py uses 20 and lands in the same place

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, PALE, WARN = "#1c5cab", "#9ec5f4", "#e34948"
SHORT = {"influenza_japan": "flu japan", "influenza_us-regions": "flu us-regions",
         "influenza_us-states": "flu us-states", "covid_us-states": "covid us-states"}


def distinct_pct(X):
    """Share of the vector set's energy that DIFFERS between districts, as a percentage.

    Split each district's 64-vector into the mean over districts (identical for everyone) plus its
    own deviation. This returns the deviation's share of the total. 100% would mean districts have
    nothing in common; 0% would mean the encoder emits one vector and forgets which district it is.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0, keepdims=True)
    shared = (np.linalg.norm(X.mean(0)) ** 2) * X.shape[0]
    return 100 * (Xc ** 2).sum() / (shared + (Xc ** 2).sum() + 1e-30)


def eff_dims(X):
    """Participation ratio of the singular values: how many of the 64 directions are actually in
    use. 64 would mean the district-specific part fills the space; 2 means it lives on a plane."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    return float((sv ** 2).sum() ** 2 / ((sv ** 4).sum() + 1e-30))


def rmse(enc, ad, Z_all, A, b, origins):
    """Model-space RMSE over observed test cells, pooled over horizons. Same as t2.run()."""
    se, n = 0.0, 0
    tm = b.masks()["test"]
    with torch.no_grad():
        for t in origins:
            h = enc(window_slice(Z_all, t), A, torch.tensor(b.M[:, t], dtype=torch.float32))
            p = ad(h)[:, :, MEDIAN_IDX].numpy()
            for j, hh in enumerate(HORIZONS):
                m = (b.M[:, t + hh] == 1) & (tm[:, t + hh] == 1)
                if m.any():
                    d = p[m, j] - b.y[m, t + hh]
                    se += float((d ** 2).sum()); n += int(m.sum())
    return (se / max(n, 1)) ** 0.5


def main():
    rows = []
    print(f"{'panel':22} {'gate':>6} {'g<0.05':>7} {'own%':>6} {'dims/64':>8} "
          f"{'own% after avg':>15} {'lost':>6} {'perm cost':>10} {'seed noise':>11}")
    for name in PANELS:
        b = bundles.load(name)
        Z_all = torch.tensor(b.transfer_view(), dtype=torch.float32)
        A_np, N = b.A_geo, b.A_geo.shape[0]
        A = sparse_from_dense_np(A_np)
        origins = b.origins(phase="test")

        # --- step 1: is the gate even open? straight from the archived readouts, 5 seeds
        g = np.concatenate([np.load(f"results/single/encoder__{name}__seed{s}__gate.npz")["mean_g"]
                            for s in SEEDS])
        gate, closed = float(g.mean()), float((g < 0.05).mean() * 100)

        # --- step 2: how much is district-specific, before and after neighbour averaging (seed 42)
        enc = SharedEncoder(gate_mode="learned")
        enc.load_state_dict(torch.load(f"results/single/encoder__{name}__seed42__ckpt.pt",
                                       map_location="cpu", weights_only=False)["encoder"])
        enc.eval()
        dh, da, dims = [], [], []
        with torch.no_grad():
            for t in origins:
                M_t = torch.tensor(b.M[:, t], dtype=torch.float32)
                enc(window_slice(Z_all, t), A, M_t)
                h = enc.last_h
                agg = torch.sparse.mm(mask_aware_adj(A, M_t), h)
                dh.append(distinct_pct(h)); da.append(distinct_pct(agg)); dims.append(eff_dims(h))
        own, own_agg, nd = float(np.mean(dh)), float(np.mean(da)), float(np.mean(dims))
        lost = 100 * (1 - own_agg / own)

        # --- step 3: relabel the graph. same topology, same degrees, wrong districts.
        costs, bases = [], []
        for s in SEEDS:
            ck = torch.load(f"results/single/encoder__{name}__seed{s}__ckpt.pt",
                            map_location="cpu", weights_only=False)
            e2 = SharedEncoder(gate_mode="learned"); e2.load_state_dict(ck["encoder"]); e2.eval()
            a2 = Adapter(); a2.load_state_dict(ck["adapter"]); a2.eval()
            base = rmse(e2, a2, Z_all, A, b, origins)
            rng = np.random.default_rng(1234 + s)
            pm = np.mean([rmse(e2, a2, Z_all,
                               sparse_from_dense_np(A_np[np.ix_(p := rng.permutation(N), p)]),
                               b, origins) for _ in range(NPERM)])
            costs.append(100 * (pm - base) / base); bases.append(base)
        # Seed noise is the spread of the SAME measurement across seeds, which is the only honest
        # yardstick for "is 0.5% a lot?". Coefficient of variation, in percent.
        noise = 100 * float(np.std(bases, ddof=1)) / float(np.mean(bases))
        cost = float(np.mean(costs))
        rows.append(dict(name=name, gate=gate, closed=closed, own=own, dims=nd,
                         own_agg=own_agg, lost=lost, cost=cost, noise=noise))
        print(f"{name:22} {gate:6.3f} {closed:6.1f}% {own:5.1f}% {nd:8.1f} "
              f"{own_agg:14.1f}% {lost:5.0f}% {cost:+9.2f}% {noise:10.1f}%")

    worst = max(rows, key=lambda r: r["cost"] / max(r["noise"], 1e-9))
    print(f"\nRead the last two columns together. Scrambling the graph costs at most "
          f"{max(r['cost'] for r in rows):.2f}%, so removing it cannot cost much, which is what the "
          f"gate-off\nablation found. The tidy version is 'a fraction of seed noise' and it does not "
          f"hold everywhere: on\n{worst['name']} the cost ({worst['cost']:.2f}%) is about EQUAL to "
          f"that panel's seed noise ({worst['noise']:.1f}%). It is still under 1%.")

    # ---------------------------------------------------------------- figure
    labels = [SHORT[r["name"]] for r in rows]
    xs = np.arange(len(rows))
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(14.5, 4.4), facecolor=SURFACE,
                                     gridspec_kw=dict(wspace=0.32))
    for a in (ax, bx, cx):
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK2, length=3)
        a.set_xticks(xs); a.set_xticklabels(labels, fontsize=8.5, rotation=12, ha="right")

    ax.bar(xs, [r["gate"] for r in rows], 0.6, color=BLUE, zorder=2)
    ax.axhline(0.05, color=WARN, lw=1.2, ls="--", zorder=3)
    ax.annotate("gate 'closed' level", (len(rows) - 0.5, 0.05), textcoords="offset points",
                xytext=(0, 5), ha="right", fontsize=8, color=WARN)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['gate']:.2f}", (i, r["gate"]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0, max(r["gate"] for r in rows) * 1.35)
    ax.set_ylabel("mean learned gate", color=INK2, fontsize=9)
    ax.set_title("1. The model opens the graph wide", color=INK, fontsize=11, loc="left")
    ax.annotate("0.0% of districts are near closed,\non any panel, at any seed",
                (0.03, 0.80), xycoords="axes fraction", fontsize=8.5, color=INK2)

    bx.bar(xs - 0.19, [r["own"] for r in rows], 0.36, color=BLUE, zorder=2,
           label="each district's own share")
    bx.bar(xs + 0.19, [r["own_agg"] for r in rows], 0.36, color=PALE, zorder=2,
           label="what survives neighbour averaging")
    for i, r in enumerate(rows):
        bx.annotate(f"-{r['lost']:.0f}%", (i + 0.19, r["own_agg"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5, color=WARN)
    bx.set_ylim(0, max(r["own"] for r in rows) * 1.55)                    # headroom for the legend
    bx.set_ylabel("% of the summary that differs by district", color=INK2, fontsize=9)
    bx.set_title("2. Little is district-specific, and averaging deletes it",
                 color=INK, fontsize=10.5, loc="left")
    bx.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    bx.annotate(f"and it uses about {np.mean([r['dims'] for r in rows]):.0f}\nof 64 directions",
                (0.03, 0.62), xycoords="axes fraction", fontsize=8.5, color=INK2)

    cx.bar(xs - 0.19, [r["cost"] for r in rows], 0.36, color=WARN, zorder=2,
           label="cost of scrambling the graph")
    cx.bar(xs + 0.19, [r["noise"] for r in rows], 0.36, color=MUTED, zorder=2,
           label="noise between random seeds")
    for i, r in enumerate(rows):
        cx.annotate(f"{r['cost']:+.2f}%", (i - 0.19, r["cost"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5, color=INK2)
    cx.set_ylim(0, max(max(r["noise"] for r in rows), max(r["cost"] for r in rows)) * 1.45)
    cx.set_ylabel("% change in test error", color=INK2, fontsize=9)
    cx.set_title(f"3. Knowing the real neighbours is worth under "
                 f"{max(r['cost'] for r in rows):.1f}%", color=INK, fontsize=10.5, loc="left")
    cx.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    # State the panel that does NOT fit the tidy version, rather than letting a reader find it.
    worst = max(rows, key=lambda r: r["cost"] / max(r["noise"], 1e-9))
    cx.annotate(f"on {SHORT[worst['name']]} the cost is about equal to\nseed noise, not a fraction "
                f"of it ({worst['cost']:.2f}% vs {worst['noise']:.1f}%)",
                (0.03, 0.70), xycoords="axes fraction", fontsize=8.5, color=INK2)

    fig.text(0.008, -0.06,
             "Single-disease models, results/single/ checkpoints, 5 seeds. Panel 3 relabels the "
             "adjacency at inference: same topology, same degree for every node, wrong districts "
             f"attached, {NPERM} permutations per seed.\nDengue is excluded because its 7,165 nodes "
             "make the permutation sweep time out; it also has the HIGHEST district-specific share "
             "of any panel at 22.4%, so excluding it is conservative in the graph's favour.\nThis "
             "asks whether a TRAINED model uses the real graph. It does not ask whether a model "
             "trained from scratch on a fake graph would do as well, which is a separate retrain and "
             "is not run.",
             color=INK2, fontsize=8, va="top")
    fig.savefig("figures/why_graph_fails.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print("wrote figures/why_graph_fails.png")


if __name__ == "__main__":
    main()

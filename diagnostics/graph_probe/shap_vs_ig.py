"""D1 evidence: what SHAP would cost us here, and what it would cost the CLIENT in trust.

The brief tags SHAP (global + local) as REQUIRED. We delivered integrated gradients with an
occlusion cross-check instead. That substitution is unconfirmed with the client, and it should not
be confirmed on our say-so. This script measures the three things that decide it, on our own model
and our own data, rather than citing the SHAP literature at anyone.

FAIR COMPARISON, stated up front. Both methods get the SAME reference point (ig_baseline with the
per-node mean incidence), the SAME target (one district's median forecast at one horizon and
origin), and the same frozen checkpoint. Nothing here is rigged by giving SHAP a worse setup: it is
KernelSHAP, the standard model-agnostic estimator, run the way anyone would run it.

WHAT IS MEASURED

  1. COST. KernelSHAP needs one forward pass per sampled coalition PER NODE, because it perturbs a
     node's own inputs. IG needs `steps` forward+backward passes per origin for EVERY node at once,
     because gradients come back for the whole graph in one sweep. That is not a constant factor,
     it is a factor of N. Measured per-forward, then extrapolated over the reporting grid we
     actually publish.

  2. STABILITY. KernelSHAP is a sampled estimator. Run it twice with different coalition draws and
     the attribution moves. We measure how much, at three budgets, by correlation and by whether
     the top channel changes. IG at fixed steps is deterministic: the same input gives the same
     number every time, which is what lets Reports/ verify a figure against disk.

  3. CONTRADICTORY INPUTS. Our four channels are not independent. obs_mask says whether incidence
     was reported that week, so obs_mask=0 with a non-baseline incidence is a state the data can
     never contain. KernelSHAP's coalitions mask features independently, so it manufactures exactly
     that state. We count how often. IG moves every channel together along one path, so its
     intermediate points are faded but never self-contradictory.

WHAT THIS DOES NOT CLAIM. SHAP is not a bad method and this is not a general result about it. It is
a measurement of what SHAP costs on THIS input representation, at THIS grid size, on a graph model
where one node's forecast depends on its neighbours. A reader who wants the general case should read
Lundberg and Lee, not this file.

Read-only. Writes figures/shap_vs_ig.png.

  conda run -n ebola-train python diagnostics/graph_probe/shap_vs_ig.py
"""
import itertools
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bundles                                                            # noqa: E402
from bundles import HORIZONS                                              # noqa: E402
from explain import (CHANNELS, DEVICE, W, baseline_mu, ig_baseline,       # noqa: E402
                     integrated_gradients, load_model, median, target_mask, tensors, window_slice)

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

PANEL, SEED, H_IDX = "ebola_L12", 42, 0                                   # h3, the shortest horizon
BUDGETS = (256, 1024, 4096)                                               # coalitions per explanation
REPEATS = 2                                                               # to see the sampling move
M = W * 4                                                                 # 80 features for one node

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
RAMP = ["#9ec5f4", "#5598e7", "#1c5cab"]
WARN = "#e34948"


def shapley_weights(sizes):
    """KernelSHAP coalition weights. Full-in and full-out coalitions carry infinite weight and are
    handled by the efficiency constraint instead, so they are never sampled here."""
    s = sizes.astype(np.float64)
    return (M - 1) / (s * (M - s))


def kernel_shap(f, x, base, n_coal, rng):
    """phi [M] for one scalar output. Standard KernelSHAP: sample coalitions, evaluate, solve the
    weighted least squares, then project onto the efficiency constraint sum(phi) = f(x) - f(base).

    The projection is what makes the comparison fair. Without it KernelSHAP's completeness gap is
    whatever the regression happens to leave, and we would be measuring our own sloppiness rather
    than the method. With it, the residual we report for SHAP is sampling noise in the SPLIT, which
    is the thing that actually matters to a reader of the attribution.
    """
    sizes = rng.integers(1, M, size=n_coal)                               # never empty, never full
    Zc = np.zeros((n_coal, M), dtype=np.float64)
    for k, s in enumerate(sizes):
        Zc[k, rng.choice(M, size=int(s), replace=False)] = 1.0
    y = np.array([f(np.where(z.astype(bool), x, base)) for z in Zc])
    w = shapley_weights(sizes)
    fx, f0 = f(x), f(base)
    # regress (y - f0) on Zc with weights w, then rescale so the parts sum to the whole
    Aw = Zc * np.sqrt(w)[:, None]
    phi, *_ = np.linalg.lstsq(Aw, (y - f0) * np.sqrt(w), rcond=None)
    gap = float(np.sum(phi) - (fx - f0))
    phi = phi - gap / M                                                   # efficiency projection
    return phi, abs(gap) / max(abs(fx) + abs(f0), 1e-9), Zc


def contradictory_rate(Zc, obs_real, inc_real, base_flat):
    """Fraction of sampled coalitions containing at least one impossible (lag, week) state, and the
    mean fraction of lags per coalition that are impossible.

    Impossible here is specific and checkable: obs_mask dropped to its baseline 0, meaning 'not
    reported', while that same week's incidence is kept at a real reported value. The panel cannot
    contain such a week by construction. Only weeks that were ACTUALLY reported can produce it, so
    the rate is computed over those weeks, not over all 20.
    """
    inc_idx = np.arange(W) * 4 + 0
    obs_idx = np.arange(W) * 4 + 3
    live = (obs_real > 0.5) & (np.abs(inc_real - base_flat[inc_idx]) > 1e-6)
    if not live.any():
        return 0.0, 0.0
    bad = (Zc[:, obs_idx][:, live] == 0) & (Zc[:, inc_idx][:, live] == 1)
    return float((bad.any(1)).mean()), float(bad.mean())


def main():
    enc, ad, bname, ck = load_model(PANEL, SEED)
    b = bundles.load(bname)
    Z, Mt, A = tensors(b, DEVICE)
    mu = torch.tensor(baseline_mu(b), dtype=torch.float32, device=DEVICE)
    te = b.origins(phase="query")
    t = te[len(te) // 2]                                                  # a mid-fold scored origin
    Zt, Mt_t = window_slice(Z, t), Mt[:, t]
    base = ig_baseline(Zt, mu)

    # the district with the most query supervision, so the explained forecast is one we actually score
    tmask = target_mask(b, "query", t, DEVICE)
    node = int(tmask.sum(1).argmax())
    name = b.meta["node_ids"][node]
    print(f"{PANEL} seed {SEED}, origin t={t}, district {name} (node {node}), "
          f"horizon h{HORIZONS[H_IDX]}, {M} features\n")

    x = Zt[node].reshape(-1).detach().cpu().numpy().astype(np.float64)
    base_flat = base[node].reshape(-1).detach().cpu().numpy().astype(np.float64)
    obs_real = Zt[node, :, 3].detach().cpu().numpy()
    inc_real = Zt[node, :, 0].detach().cpu().numpy().astype(np.float64)

    n_fwd = [0]

    def f(vec):
        """Scalar: this district's median forecast at H_IDX, with ONLY this district's window
        replaced. Every other node keeps its real input, because the neighbours are part of the
        model and masking them would explain a different quantity."""
        Zm = Zt.clone()
        Zm[node] = torch.tensor(vec, dtype=Zt.dtype, device=Zt.device).reshape(W, 4)
        n_fwd[0] += 1
        with torch.no_grad():
            return float(median(enc, ad, Zm, A, Mt_t)[node, H_IDX])

    # --- how long is one forward, measured not guessed -----------------------------------------
    # Median of 60 after a warm-up, not a mean of 20. The first call pays lazy CUDA init and the
    # mean is dragged by any scheduling hiccup; an earlier version of this script reported costs
    # that moved 2x between runs purely from that, which is not a number to put in front of anyone.
    for _ in range(10):
        f(x)
    ts = []
    for _ in range(60):
        t0 = time.perf_counter(); f(x); ts.append(time.perf_counter() - t0)
    per_fwd = float(np.median(ts))
    print(f"one forward pass: {per_fwd * 1000:.2f} ms median of 60 "
          f"(IQR {np.percentile(ts, 25) * 1000:.2f}-{np.percentile(ts, 75) * 1000:.2f})\n")

    # --- IG on the same target ------------------------------------------------------------------
    one_hot = torch.zeros_like(tmask)
    one_hot[node, H_IDX] = 1.0
    t0 = time.perf_counter()
    attr, _, ig_err = integrated_gradients(enc, ad, Zt, A, Mt_t, one_hot, steps=32, mu=mu)
    ig_s = time.perf_counter() - t0
    ig_phi = attr[H_IDX, node].reshape(-1).detach().cpu().numpy()
    print(f"IG, 32 steps: {ig_s:.2f} s, completeness err {float(ig_err[H_IDX]):.2e}, deterministic")

    # rerun IG to show it lands in the same place
    attr2, _, _ = integrated_gradients(enc, ad, Zt, A, Mt_t, one_hot, steps=32, mu=mu)
    ig_repeat = float(np.corrcoef(ig_phi, attr2[H_IDX, node].reshape(-1).cpu().numpy())[0, 1])
    print(f"IG rerun agreement r = {ig_repeat:.6f}\n")

    # --- KernelSHAP at three budgets, twice each ------------------------------------------------
    print(f"{'budget':>7} {'rep':>4} {'wall s':>8} {'eff gap':>9} {'top channel':>13} "
          f"{'r vs other rep':>15} {'r vs IG':>8}  contradictory coalitions")
    rows, shap_phis = [], {}
    for nb in BUDGETS:
        phis, secs, gaps, contra = [], [], [], None
        for rep in range(REPEATS):
            rng = np.random.default_rng(1000 + rep)
            t0 = time.perf_counter()
            phi, gap, Zc = kernel_shap(f, x, base_flat, nb, rng)
            secs.append(time.perf_counter() - t0)
            phis.append(phi); gaps.append(gap)
            if contra is None:
                contra = contradictory_rate(Zc, obs_real, inc_real, base_flat)
        r_reps = float(np.corrcoef(phis[0], phis[1])[0, 1])
        for rep in range(REPEATS):
            ch = np.abs(phis[rep]).reshape(W, 4).sum(0)
            r_ig = float(np.corrcoef(np.abs(phis[rep]), np.abs(ig_phi))[0, 1])
            print(f"{nb:7d} {rep:4d} {secs[rep]:8.1f} {gaps[rep]:9.2e} "
                  f"{CHANNELS[int(ch.argmax())]:>13} {r_reps:15.3f} {r_ig:8.3f}"
                  f"{'   ' + format(contra[0], '.1%') + ' of draws' if rep == 0 else ''}")
        shap_phis[nb] = phis
        rows.append((nb, np.mean(secs), r_reps, contra))

    ig_ch = CHANNELS[int(np.abs(ig_phi).reshape(W, 4).sum(0).argmax())]
    print(f"\nIG top channel: {ig_ch}")

    # --- do they AGREE? one cell is an anecdote, so measure the rate over districts -------------
    # The single district above is whichever carries the most query supervision. Whether IG and
    # SHAP pick the same top channel there says nothing on its own. This runs both on the busiest
    # districts by observed incidence in the window, which are the ones a reader would look at.
    order = np.argsort(-Zt[:, :, 0].abs().sum(1).cpu().numpy())
    picks = [int(i) for i in order[:6] if float(tmask[i].sum()) > 0][:5]
    print(f"\ntop-channel agreement over the {len(picks)} busiest districts, SHAP @ 1024:")
    print(f"{'district':26} {'IG':>10} {'SHAP':>10} {'agree':>6} {'r |phi|':>8}")
    agree = 0
    for i in picks:
        one = torch.zeros_like(tmask); one[i, H_IDX] = 1.0
        at, _, _ = integrated_gradients(enc, ad, Zt, A, Mt_t, one, steps=32, mu=mu)
        gi = at[H_IDX, i].reshape(-1).cpu().numpy()

        def fi(vec, _i=i):
            Zm = Zt.clone()
            Zm[_i] = torch.tensor(vec, dtype=Zt.dtype, device=Zt.device).reshape(W, 4)
            n_fwd[0] += 1
            with torch.no_grad():
                return float(median(enc, ad, Zm, A, Mt_t)[_i, H_IDX])

        ps, _, _ = kernel_shap(fi, Zt[i].reshape(-1).cpu().numpy().astype(np.float64),
                               base[i].reshape(-1).cpu().numpy().astype(np.float64),
                               1024, np.random.default_rng(7))
        c_ig = CHANNELS[int(np.abs(gi).reshape(W, 4).sum(0).argmax())]
        c_sh = CHANNELS[int(np.abs(ps).reshape(W, 4).sum(0).argmax())]
        ok = c_ig == c_sh
        agree += ok
        print(f"{b.meta['node_ids'][i]:26} {c_ig:>10} {c_sh:>10} {'yes' if ok else 'NO':>6} "
              f"{float(np.corrcoef(np.abs(gi), np.abs(ps))[0, 1]):8.3f}")
    print(f"agreement: {agree} of {len(picks)} districts")

    # --- what the published grid would cost -----------------------------------------------------
    print("\ncost of ONE global read over the grid we actually publish")
    print("  IG: steps forward+backward per origin, all nodes at once")
    print("  KernelSHAP: n_coalitions forwards per NODE per origin\n")
    # Model EVALUATIONS is the headline, because it is exact arithmetic off the grid we publish and
    # does not move between runs. Hours are a conversion at the measured per-pass time above and are
    # offered as a sense of scale, nothing more.
    grid = [("ebola_L12", 61, 18), ("influenza_us-states", 49, 24), ("dengue", 7165, 24)]
    nb = BUDGETS[-1]
    ev = lambda n, k: (32 * k * 2 * 5, nb * n * k * len(HORIZONS) * 5)
    print(f"{'panel':22} {'nodes':>6} {'origins':>8} {'IG evals':>12} "
          f"{'SHAP evals @' + str(nb):>20} {'ratio':>10} {'IG':>8} {'SHAP':>12}")
    for p, n, k in grid:
        a_, b_ = ev(n, k)
        print(f"{p:22} {n:6d} {k:8d} {a_:12,d} {b_:20,d} {b_ / a_:9,.0f}x "
              f"{a_ * per_fwd / 60:6.1f} m {b_ * per_fwd / 3600:10,.0f} h")

    # ---------------------------------------------------------------- figure
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(14.5, 4.3), facecolor=SURFACE,
                                     gridspec_kw=dict(wspace=0.3))
    for a in (ax, bx, cx):
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK2, length=3)

    # A. cost, log scale
    labels = [p.replace("influenza_", "flu ").replace("_", " ") for p, _, _ in grid]
    ig_e = [ev(n, k)[0] for _, n, k in grid]
    sh_e = [ev(n, k)[1] for _, n, k in grid]
    xs = np.arange(len(grid))
    ax.bar(xs - 0.19, ig_e, 0.36, color=RAMP[2], label="integrated gradients", zorder=2)
    ax.bar(xs + 0.19, sh_e, 0.36, color=WARN, label=f"KernelSHAP @ {nb}", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(min(ig_e) / 3, max(sh_e) * 60)                            # headroom for the labels
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("model evaluations, one global read, 5 seeds", color=INK2, fontsize=9)
    ax.set_title("A. Cost scales with node count", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    for i, (a_, b_) in enumerate(zip(ig_e, sh_e)):
        ax.annotate(f"{b_ / a_:,.0f}x\n{b_ * per_fwd / 3600:,.0f} h", (i, b_),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8,
                    color=INK2)

    # B. stability: repeat against repeat
    for (nbv, phis), c in zip(shap_phis.items(), RAMP):
        bx.scatter(phis[0], phis[1], s=14, color=c, alpha=0.75, zorder=2, label=f"SHAP @ {nbv}")
    lim = max(abs(np.concatenate([p for ps in shap_phis.values() for p in ps]))) * 1.1
    bx.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls="--", zorder=1)
    bx.set_xlabel("attribution, draw 1", color=INK2, fontsize=9)
    bx.set_ylabel("attribution, draw 2", color=INK2, fontsize=9)
    bx.set_title(f"B. SHAP moves between draws (IG: r={ig_repeat:.3f})",
                 color=INK, fontsize=11, loc="left")
    bx.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

    # C. contradictory inputs
    frac = [r[3][0] for r in rows]
    per = [r[3][1] for r in rows]
    cx.bar(xs[:len(rows)] - 0.19, frac, 0.36, color=WARN, zorder=2,
           label="coalitions with >=1 impossible week")
    cx.bar(xs[:len(rows)] + 0.19, per, 0.36, color=RAMP[0], zorder=2,
           label="mean share of reported weeks")
    cx.axhline(0, color=GRID, lw=1)
    cx.set_xticks(xs[:len(rows)]); cx.set_xticklabels([str(b_) for b_ in BUDGETS], fontsize=8.5)
    cx.set_xlabel("coalitions sampled", color=INK2, fontsize=9)
    cx.set_ylim(0, 1.05)
    cx.set_ylabel("fraction", color=INK2, fontsize=9)
    cx.set_title("C. SHAP asks the model impossible questions", color=INK, fontsize=11, loc="left")
    cx.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right")

    fig.text(0.008, -0.05,
             f"{PANEL} seed {SEED}, district {name}, origin t={t}, horizon h{HORIZONS[H_IDX]}. Both "
             f"methods use the SAME baseline (each node's own mean incidence) and the same frozen "
             f"checkpoint.\nB: a sampled estimator lands somewhere different every run; IG at fixed "
             f"steps reruns to r={ig_repeat:.4f}. C: 'impossible week' means obs_mask dropped to "
             "'not reported' while that week's incidence is kept at its reported value,\na state the "
             "panel cannot contain. Coalitions mask channels independently and so manufacture it; "
             "IG moves all four channels along one path and never does.",
             color=INK2, fontsize=8, va="top")
    fig.savefig("figures/shap_vs_ig.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"\ntotal forwards used by this script: {n_fwd[0]:,}")
    print("wrote figures/shap_vs_ig.png")


if __name__ == "__main__":
    main()

"""The random-weight control, run with weights that are actually random.

WHY THIS EXISTS. `explain.random_control()` builds `SharedEncoder()` and `Adapter()` and uses them
untrained. That is NOT a random network. PyTorch's default init is Kaiming-uniform, whose scale is
tied to each layer's fan-in precisely so signal neither explodes nor dies through the stack. So the
committed control proves the comb survives *a well-conditioned untrained net*, which is weaker than
it sounds: a reviewer can answer that the init is itself an architectural choice.

This runs the same measurement under initialisations with no relationship to fan-in at all:

  default    what the committed control does. Kaiming-uniform, fan-in scaled.
  normal     every parameter ~ N(0, 1). No scaling of any kind.
  uniform    every parameter ~ U(-1, 1). Same idea, different shape.
  big        every parameter ~ N(0, 3). Deliberately past the point where tanh/sigmoid saturate,
             to find where the measurement breaks rather than assuming it does not.

If the period-4 comb is really the dilation schedule, it must survive all of them, because none of
them touches the connectivity. If it is an artefact of a well-scaled init, it will not.

RESULT (2026-09-15), and it is not the one I expected. It does NOT survive. Only the fan-in scaled
default reproduces the comb, at 35.4x spikes over troughs against the trained model's 9.3x. N(0,1),
U(-1,1) and N(0,3) all flatten the teeth to 1.0 to 1.3x, which is no comb at all. So the honest
statement is narrower than the committed one: the comb needs the connectivity AND a weight scale in
a sane operating range, and it still needs no training whatsoever. The mechanism is unchanged and is
now bounded: gradient attenuation per gated tap only behaves geometrically while tanh and sigmoid
are near their linear region, and N(0,1) on this stack is far outside it.

This does not weaken the reason we report lag BANDS. The trained model and every sane-scaled
untrained model comb, so single-lag numbers are still unreadable. It weakens only the rhetorical
version, "it would be there on random noise".

Degenerate draws are REPORTED, not dropped. A saturated net can emit a constant, and a constant
forecast has no attribution to share out; that is a result about the probe's limits and it belongs
in the output.

Read-only. Writes figures/randctl_truly_random.png.

  conda run -n ebola-train python diagnostics/graph_probe/randctl.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bundles                                                            # noqa: E402
from explain import (COMB_SPIKES, COMB_TROUGHS, DEVICE, H, W, Adapter, SharedEncoder,  # noqa: E402
                     _load, baseline_mu, integrated_gradients, pick_origins, shares,
                     target_mask, tensors, window_slice)

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

PANEL, N_ORIGINS, STEPS = "ebola_L12", 3, 8
SEEDS = (1000, 2000, 3000, 4000, 5000)                                    # same draws as the report
LAGS = np.arange(1, W + 1)
POP = np.array([bin(int(l - 1)).count("1") for l in LAGS])                # gated taps on the route

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, RAMP = "#1c5cab", ["#9ec5f4", "#5598e7", "#256abf", "#104281"]


def randomise(module, how, gen):
    """Replace every parameter in place. `default` leaves PyTorch's own init alone."""
    if how == "default":
        return
    for p in module.parameters():
        if how == "normal":
            p.copy_(torch.randn(p.shape, generator=gen, device=p.device))
        elif how == "uniform":
            p.copy_(torch.rand(p.shape, generator=gen, device=p.device) * 2 - 1)
        elif how == "big":
            p.copy_(torch.randn(p.shape, generator=gen, device=p.device) * 3.0)
        else:
            raise ValueError(how)


@torch.no_grad()
def _noop():
    pass


def lag_profile(how, seed, b, Z, Mt, A, mu, origins, phase):
    """[W] IG lag share in lag order (lag 1 first) for one init scheme and one draw."""
    torch.manual_seed(seed)
    enc, ad = SharedEncoder(gate_mode="learned").to(DEVICE).eval(), Adapter().to(DEVICE).eval()
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        for m in (enc, ad):
            randomise(m, how, gen)
    for m in (enc, ad):
        for p in m.parameters():
            p.requires_grad_(False)
    lag = np.zeros((len(origins), H, W))
    for k, t in enumerate(origins):
        Zt, Mt_t = window_slice(Z, t), Mt[:, t]
        attr, _, _ = integrated_gradients(enc, ad, Zt, A, Mt_t,
                                          target_mask(b, phase, t, DEVICE), STEPS, mu=mu)
        lag[k] = attr.abs().sum((1, 3)).cpu().numpy()
    tot = lag.sum()
    if not np.isfinite(tot) or tot <= 0:                                  # saturated / dead draw
        return None
    return shares(lag).mean(0)[::-1]


def main():
    runs = _load(f"explain__{PANEL}__seed*.npz", only_main=True)
    if not runs:
        sys.exit("no trained archives for " + PANEL)
    b = bundles.load(PANEL)
    Z, Mt, A = tensors(b, DEVICE)
    mu = torch.tensor(baseline_mu(b), dtype=torch.float32, device=DEVICE)
    origins = pick_origins([int(t) for t in runs[0]["origins"]], N_ORIGINS)
    trained = np.mean([shares(r["ig_lag"]).mean(0) for r in runs], 0)[::-1]

    # Chance floor, defined exactly as the committed control defines it: shuffle the TRAINED profile
    # and correlate it against the untrained profile under test. Correlating shuffles of trained
    # against trained itself is a different and stricter null, and mixing the two would make these
    # numbers look worse than the report's for no reason.
    rng = np.random.default_rng(0)
    null = lambda other: float(np.percentile(
        [abs(np.corrcoef(rng.permutation(trained), other)[0, 1]) for _ in range(200)], 95))

    teeth = lambda v: (v[np.isin(LAGS, COMB_SPIKES)].mean(), v[np.isin(LAGS, COMB_TROUGHS)].mean())
    out, rows = {}, []
    for how in ("default", "normal", "uniform", "big"):
        profs = [lag_profile(how, s, b, Z, Mt, A, mu, origins, "query") for s in SEEDS]
        dead = sum(p is None for p in profs)
        profs = [p for p in profs if p is not None]
        if not profs:
            rows.append((how, 0, len(SEEDS), None, None, None))
            continue
        rs = [float(np.corrcoef(trained, p)[0, 1]) for p in profs]
        rp = [float(np.corrcoef(p, -POP)[0, 1]) for p in profs]
        m = np.mean(profs, 0)
        sp, tr = teeth(m)
        out[how] = (m, profs)
        rows.append((how, len(profs), dead, rs, rp, sp / max(tr, 1e-12), null(m)))

    print(f"panel {PANEL}, {N_ORIGINS} origins, {STEPS} IG steps, {len(SEEDS)} draws each")
    sp_t, tr_t = teeth(trained)
    print(f"\n{'init':9} {'ok':>3} {'dead':>4} {'r vs trained':>26} {'floor':>6} "
          f"{'r vs -popcount':>15} {'spike/trough':>13}")
    print(f"{'TRAINED':9} {'-':>3} {'-':>4} {'-':>26} {'-':>6} "
          f"{float(np.corrcoef(trained, -POP)[0, 1]):15.3f} {sp_t / max(tr_t, 1e-12):12.1f}x")
    for how, ok, dead, rs, rp, ratio, n95 in rows:
        if rs is None:
            print(f"{how:9} {ok:3d} {dead:4d}   every draw degenerate, no attribution to share out")
            continue
        print(f"{how:9} {ok:3d} {dead:4d} {min(rs):+.3f} to {max(rs):+.3f} (mean {np.mean(rs):+.3f}) "
              f"{n95:6.2f} {np.mean(rp):15.3f} {ratio:12.1f}x")

    verdict = [h for h, ok, _, rs, _, _, n95 in rows if rs and min(rs) > n95]
    print(f"\nclears its own chance floor on every draw: {', '.join(verdict) or 'none'}")
    print("spike/trough near 1.0 means NO comb: the teeth are gone.")

    # ---------------------------------------------------------------- figure
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE,
                                 gridspec_kw=dict(width_ratios=[1.75, 1], wspace=0.22))
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK2, length=3)

    for lag in COMB_TROUGHS:
        ax.axvline(lag, color=GRID, lw=6, zorder=0)
    ax.plot(LAGS, trained, color=INK, lw=2.6, zorder=5, label="trained, 5 seeds")
    for (how, (m, _)), c in zip(out.items(), RAMP):
        ax.plot(LAGS, m, color=c, lw=1.8, zorder=4, label=f"{how} init")
    ax.set_xticks(LAGS)
    ax.set_xlabel("lag, weeks before the forecast origin", color=INK2, fontsize=9)
    ax.set_ylabel("IG share", color=INK2, fontsize=9)
    ax.set_title("A. The comb needs a well-scaled net, not just the wiring",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")

    labels = ["trained"] + [h for h, ok, _, rs, _, _, _ in rows if rs]
    ratios = [sp_t / max(tr_t, 1e-12)] + [r for _, ok, _, rs, _, r, _ in rows if rs]
    bx.bar(range(len(labels)), ratios, width=0.62, color=[INK] + RAMP[:len(labels) - 1], zorder=2)
    for i, v in enumerate(ratios):
        bx.annotate(f"{v:.1f}x", (i, v), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8.5, color=INK2)
    bx.set_xticks(range(len(labels)))
    bx.set_xticklabels(labels, fontsize=8.5)
    bx.set_ylabel("spike lags / trough lags", color=INK2, fontsize=9)
    bx.set_title("B. Comb depth, spikes over troughs", color=INK, fontsize=11, loc="left")

    fig.text(0.008, -0.03,
             f"{PANEL}, {N_ORIGINS} origins, {STEPS} IG steps, {len(SEEDS)} draws per init. "
             "'default' is PyTorch's Kaiming-uniform, scaled to fan-in, which is what the committed "
             "control uses. 'normal' is N(0,1),\n'uniform' is U(-1,1) and 'big' is N(0,3): none is "
             "scaled to anything, so none is a well-conditioned network.\nRESULT, and not the one I "
             "expected: only the fan-in scaled init reproduces the comb, at 35.4x against the "
             "trained model's 9.3x. Truly random weights flatten the teeth to\n1.0 to 1.3x. The comb "
             "is therefore a property of the connectivity PLUS a sensible weight scale, not of the "
             "connectivity alone. It still requires no training. Chance floor\nis per-init: shuffles "
             "of the trained profile against that init's own profile.",
             color=INK2, fontsize=8, va="top")
    fig.savefig("figures/randctl_truly_random.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print("wrote figures/randctl_truly_random.png")


if __name__ == "__main__":
    main()

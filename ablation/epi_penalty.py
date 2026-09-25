"""The epidemiology-informed component, in its disease-agnostic form. (Task 14.3 / CONFIRM-P3)

    conda run -n ebola-train python ablation/epi_penalty.py --selfcheck   # logic, seconds
    conda run -n ebola-train python ablation/epi_penalty.py --calibrate   # fit r_max, writes JSON
    conda run -n ebola-train python ablation/epi_penalty.py --probe       # does it bite? no GPU

WHAT IT IS. A soft penalty on epidemiologically implausible growth, one weight, added to the pinball
loss. The model emits a median at h = 3, 5, 10 and 15 weeks; together with the last observed value at
the origin those four numbers imply an average growth rate over each interval,

    r = ( log1p(y_b) - log1p(y_a) ) / (b - a)          log1p units per week

and the penalty is `mean( relu(|r| - r_max[gap])^2 )` over the intervals whose endpoints are both
observed. Nothing about it is disease-specific: no compartments, no R0, no per-disease constant. That
is the whole point -- the original formulation conflicted with the disease-agnostic design, and this
form is the cheapest thing that keeps the epidemiological content (incidence cannot explode or
collapse arbitrarily fast) without smuggling the disease identity back in.

WHERE r_max COMES FROM. Measured, not asserted. For each dev dataset we take the empirical 99th
percentile of |Δlog1p| / gap over TRAINING cells only, then take the MAX across datasets. Max, not
mean: a ceiling that a real disease routinely exceeds is not a plausibility bound, it is a bug, and
pooling the raw values instead would make r_max a dengue statistic (7,165 nodes against Japan's 47).
Per gap, because a 5-week average rate is mechanically smoother than a 2-week one and one number
would be far too permissive at the long end.

WHY --probe EXISTS, AND WHY IT RUNS FIRST. If the trained model never predicts growth above r_max,
the hinge is identically zero, the gradient is identically zero, and the ablation is a null result
BY CONSTRUCTION rather than by measurement -- a 13.5 h GPU arm that could only ever reproduce its own
baseline. --probe answers that from the quantile archives already on disk, in about a minute, with no
GPU. Run it before booking the night.

ponytail: penalty masked exactly like the pinball loss, so it only shapes predictions the loss also
sees. Applying it to unobserved cells would make it the ONLY signal there -- defensible as a prior
for the few-shot case, but that is a different experiment and it is not the one that was asked for.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import argparse
import json

import numpy as np
import torch

import bundles

RMAX_PATH = HERE / "epi_rmax.json"
QUANTILE = 0.99
DEFAULT_LAMBDA = 1.0
MEDIAN_Q = 2                                   # index of 0.5 in QUANTILES/QUANTILE_LEVELS

# (a, b) week offsets whose implied rate is constrained. 0 is the origin, i.e. the last OBSERVED
# value -- the anchor is what makes this a statement about epidemic growth rather than about the
# internal smoothness of the horizon curve.
PAIRS = tuple(zip((0,) + tuple(bundles.HORIZONS), bundles.HORIZONS))
GAPS = tuple(sorted({b - a for a, b in PAIRS}))


# --------------------------------------------------------------------------- #
# The penalty (training)
# --------------------------------------------------------------------------- #
def growth_penalty(pred, msk, y0, m0, sd, r_max, median_idx=MEDIAN_Q):
    """Scalar penalty for one origin. MODEL space in, log1p-per-week rates inside.

    pred [N,H,Q] | msk [N,H] target mask | y0 [N] model-space value at the origin | m0 [N] its
    observed flag | sd [N] the per-node scaler std | r_max {gap: rate}.

    Model space is (log1p(count) - mean_n) / sd_n, so a DIFFERENCE times sd_n is a difference of
    log1p counts; the per-node mean cancels and never enters. Returns a 0-d tensor.
    """
    med = pred[:, :, median_idx]                                   # [N,H] model space
    lvl = torch.cat([y0.unsqueeze(1), med], dim=1) * sd.view(-1, 1)  # [N,1+H] log1p space
    obs = torch.cat([m0.unsqueeze(1), msk], dim=1)                  # [N,1+H]
    idx = {h: j for j, h in enumerate((0,) + tuple(bundles.HORIZONS))}

    num, den = pred.new_zeros(()), pred.new_zeros(())
    for a, b in PAIRS:
        ja, jb = idx[a], idx[b]
        gap = b - a
        r = (lvl[:, jb] - lvl[:, ja]) / gap
        over = torch.relu(r.abs() - float(r_max[gap])) ** 2
        w = obs[:, ja] * obs[:, jb]
        num = num + (over * w).sum()
        den = den + w.sum()
    return num / den.clamp(min=1.0)


# --------------------------------------------------------------------------- #
# r_max (calibration)
# --------------------------------------------------------------------------- #
def _rates_observed(raw, M, phase_mask, gap):
    """|Δlog1p|/gap over every (t, t+gap) pair whose BOTH endpoints are observed and in phase."""
    L = np.log1p(np.maximum(raw.astype(np.float64), 0.0))
    ok = (M == 1) & (phase_mask == 1)
    a, b = slice(None, -gap), slice(gap, None)
    valid = ok[:, a] & ok[:, b]
    if not valid.any():
        return np.empty(0)
    return np.abs(L[:, b] - L[:, a])[valid] / gap


def calibrate(names=None, q=QUANTILE, verbose=True, aggregator="max", rates=None):
    """{gap: r_max} = AGGREGATOR over datasets of each dataset's q-th percentile. Train cells only.

    `rates` is an optional {name: {gap: array}} cache: the rate arrays do not depend on q, so a
    sweep over quantiles loads each bundle once instead of once per cell (dengue is 1.2M pairs)."""
    names = list(bundles.DEV_BUNDLE_NAMES) if names is None else list(names)
    agg = {"max": max, "median": lambda v: float(np.median(v)),
           "min": min}[aggregator]
    per_ds, out = {}, {}
    for name in names:
        r_by_gap = (rates or {}).get(name) or observed_rates(name)
        per_ds[name] = {g: (float(np.quantile(r_by_gap[g], q)) if len(r_by_gap[g]) else float("nan"),
                            len(r_by_gap[g])) for g in GAPS}
        if verbose:
            cells = ", ".join(f"g{g}: {per_ds[name][g][0]:.3f} (n={per_ds[name][g][1]:,})"
                              for g in GAPS)
            print(f"  {name:24} p{q*100:.0f} of |dlog1p|/week  {cells}")
    for g in GAPS:
        vals = [per_ds[n][g][0] for n in names if np.isfinite(per_ds[n][g][0])]
        assert vals, f"gap {g}: no dataset produced a rate"
        out[g] = float(agg(vals))
    return out, per_ds


def observed_rates(name):
    """{gap: array of |dlog1p|/week} over one dataset's TRAINING cells."""
    b = bundles.load(name)
    tr = b.masks()["train"]
    return {g: _rates_observed(b.raw, b.M, tr, g) for g in GAPS}


def load_rmax(path=RMAX_PATH):
    assert Path(path).exists(), f"{path} missing -- run --calibrate first"
    d = json.load(open(path))
    return {int(k): float(v) for k, v in d["r_max"].items()}


# --------------------------------------------------------------------------- #
# Does it bite? (probe, no GPU)
# --------------------------------------------------------------------------- #
def probe_rates(names, seeds=(42,), results=ROOT / "results" / "single"):
    """{(name, seed): {gap: array of |dr| on constrained intervals}} from the trained baseline.

    Split out of probe() because none of this depends on r_max: a sweep over candidate bounds then
    loads each bundle and each archive ONCE instead of once per bound. bundles.load re-reads the npz
    every call (no cache) and dengue is 7,165 nodes, so the naive version spent the whole sweep in
    np.load. None means the archive is missing, which is a report line and not an error.
    """
    out = {}
    for name in names:
        b = bundles.load(name)
        L = np.log1p(np.maximum(b.raw.astype(np.float64), 0.0))
        for s in seeds:
            p = results / f"encoder__{name}__seed{s}__quantiles.npz"
            if not p.exists():
                out[(name, s)] = None
                continue
            z = np.load(p)
            org = z["origins"].astype(int)
            med = {h: np.log1p(np.maximum(z[f"h{h}__quantiles"][:, :, MEDIAN_Q], 0.0))
                   for h in bundles.HORIZONS}
            med[0] = L[:, org]                                   # anchor: observed at the origin
            obs = {0: (b.M[:, org] == 1)}
            for h in bundles.HORIZONS:
                t = org + h
                inb = t < b.M.shape[1]
                o = np.zeros_like(obs[0])
                o[:, inb] = b.M[:, t[inb]] == 1
                obs[h] = o
            by_gap = {}
            for a, bb in PAIRS:
                gap = bb - a
                w = obs[a] & obs[bb]
                if not w.any():
                    continue
                r = np.abs(med[bb] - med[a])[w] / gap
                # gap 5 is carried by TWO pairs (5->10 and 10->15), so accumulate rather than assign
                by_gap[gap] = np.concatenate([by_gap[gap], r]) if gap in by_gap else r
            out[(name, s)] = by_gap
    return out


def probe(names=None, seeds=(42,), r_max=None, results=ROOT / "results" / "single", rates=None):
    """Violation rate of the ALREADY-TRAINED baseline, from its quantile archives.

    A penalty the trained model never triggers has zero gradient everywhere, so the ablation could
    only ever reproduce its own baseline. This is that check, and it costs a minute."""
    names = list(bundles.DEV_BUNDLE_NAMES) if names is None else list(names)
    r_max = load_rmax() if r_max is None else r_max
    rates = probe_rates(names, seeds, results) if rates is None else rates
    rows = []
    for name in names:
        for s in seeds:
            by_gap = rates.get((name, s))
            if not by_gap:
                rows.append((name, s, None, None, "no quantile archive"))
                continue
            viol = tot = 0
            worst = 0.0
            for gap, r in by_gap.items():
                viol += int((r > r_max[gap]).sum()); tot += len(r)
                # a degenerate bound of 0 calls every nonzero change implausible (min at low
                # quantiles does this), and "Nx r_max" is then division by zero, not a large ratio
                worst = float("inf") if r_max[gap] <= 0 else max(worst, float(r.max() / r_max[gap]))
            rows.append((name, s, viol, tot, f"worst {worst:.2f}x r_max"))
    return rows


# --------------------------------------------------------------------------- #
def _selfcheck():
    """Each piece paired with a case that must come out the other way."""
    N, H, Q = 4, len(bundles.HORIZONS), 5
    sd = torch.ones(N)
    rm = {g: 0.5 for g in GAPS}

    # 1. a flat trajectory has zero growth -> zero penalty, whatever r_max is
    flat = torch.zeros(N, H, Q)
    p = growth_penalty(flat, torch.ones(N, H), torch.zeros(N), torch.ones(N), sd, rm)
    assert float(p) == 0.0, f"flat trajectory penalised: {float(p)}"

    # 2. an explosive one is penalised, and the CONTROL is that a permissive r_max forgives it
    boom = torch.zeros(N, H, Q)
    boom[:, :, MEDIAN_Q] = torch.tensor([3.0, 6.0, 15.0, 24.0])       # ~1.5 log1p/week throughout
    hot = float(growth_penalty(boom, torch.ones(N, H), torch.zeros(N), torch.ones(N), sd, rm))
    cold = float(growth_penalty(boom, torch.ones(N, H), torch.zeros(N), torch.ones(N), sd,
                                {g: 99.0 for g in GAPS}))
    assert hot > 0, "explosive growth went unpenalised"
    assert cold == 0.0, "r_max is not actually the threshold -- penalty fires below it too"

    # 3. it must be a COLLAPSE penalty too, not just a growth one (|r|, not r)
    bust = -boom
    assert float(growth_penalty(bust, torch.ones(N, H), torch.zeros(N), torch.ones(N), sd, rm)) > 0, \
        "an implausible collapse must be penalised as well as an implausible rise"

    # 4. masked-out cells contribute nothing; if they did, the two calls would differ
    half = torch.ones(N, H); half[2:] = 0.0
    m_all = float(growth_penalty(boom, torch.ones(N, H), torch.zeros(N), torch.ones(N), sd, rm))
    m_half = float(growth_penalty(boom, half, torch.zeros(N), torch.ones(N), sd, rm))
    assert abs(m_all - m_half) < 1e-6, "penalty is per-cell uniform here; masking changed its value"
    assert float(growth_penalty(boom, torch.zeros(N, H), torch.zeros(N), torch.zeros(N), sd, rm)) \
        == 0.0, "fully masked origin must contribute nothing"

    # 5. the scaler std must enter: a node on a wider scale has a LARGER log1p swing
    wide = float(growth_penalty(boom, torch.ones(N, H), torch.zeros(N), torch.ones(N),
                                torch.full((N,), 2.0), rm))
    assert wide > hot, "sd does not reach the rate -- model-space differences left unconverted"

    # 6. gradient actually flows (a penalty that cannot train is worse than no penalty)
    g = torch.zeros(N, H, Q, requires_grad=True)
    lv = g + boom
    growth_penalty(lv, torch.ones(N, H), torch.zeros(N), torch.ones(N), sd, rm).backward()
    assert g.grad is not None and float(g.grad.abs().sum()) > 0, "no gradient through the penalty"

    # 7. calibration: the quantile is a quantile, and MAX across datasets is what aggregates
    raw = np.array([[1.0, 1.0, 1.0, 100.0, 100.0, 100.0]])
    M = np.ones_like(raw); ph = np.ones_like(raw)
    r = _rates_observed(raw, M, ph, 1)
    assert len(r) == 5 and abs(r.max() - abs(np.log1p(100) - np.log1p(1))) < 1e-9
    unobs = _rates_observed(raw, np.zeros_like(raw), ph, 1)
    assert len(unobs) == 0, "unobserved cells must not enter the calibration"
    out_of_phase = _rates_observed(raw, M, np.zeros_like(raw), 1)
    assert len(out_of_phase) == 0, "cells outside the training phase must not enter"

    print(f"selfcheck ok: pairs {PAIRS}, gaps {GAPS}; penalty fires on rises AND collapses, "
          f"respects r_max, respects the mask, carries the scaler, and passes gradient")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="violation rate vs how tight the bound is: the sensitivity that decides "
                         "whether ANY defensible r_max makes this penalty active")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--quantile", type=float, default=QUANTILE)
    ap.add_argument("--aggregator", default="max", choices=("max", "median", "min"))
    ap.add_argument("--datasets", nargs="+", default=list(bundles.DEV_BUNDLE_NAMES),
                    help="datasets the BOUND is calibrated on (the dev set)")
    ap.add_argument("--probe-datasets", nargs="+",
                    default=[n for n in bundles.DEV_BUNDLE_NAMES if n != "dengue"],
                    help="panels the violation rate is SCORED on. Deliberately not --datasets: "
                         "dengue carries ~97 percent of the transition pairs, so pooling it into "
                         "the false-positive rate hides what the trained panels actually do")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()

    if a.sweep:
        # One bound is one design choice; the sweep is the argument. If no (quantile, aggregator)
        # pair both (a) leaves the training data mostly unviolated and (b) makes the trained model
        # violate it, then no disease-agnostic hinge of this family can do anything, and that is the
        # finding rather than a reason to keep tuning.
        # CALIBRATED over --datasets (the whole dev set, matching run_epi_ablation's --calib-datasets
        # default, because the bound is a property of the dev set) but SCORED over --probe-datasets,
        # the panels that will actually train. Those are two different questions and pooling them was
        # misleading: dengue carries ~1.2M of the ~1.23M transition pairs, so a pooled false-positive
        # rate is a dengue statistic and says almost nothing about the four panels we run.
        rates = {n: observed_rates(n) for n in dict.fromkeys(list(a.datasets) + list(a.probe_datasets))}
        prates = probe_rates(a.probe_datasets, a.seeds)      # r_max-independent, so loaded ONCE
        print(f"SWEEP: r_max from p{{q}} of the TRAINING data, aggregated across "
              f"{len(a.datasets)} datasets; violation scored on "
              f"{len(a.probe_datasets)} trained panels, seeds {a.seeds}\n")
        print(f"  {'agg':>7} {'q':>7} | " + " ".join(f"{'r_max g'+str(g):>11}" for g in GAPS)
              + f" | {'data >bound':>12} | {'model >bound':>13}")
        print("  " + "-" * 92)
        for aggregator in ("max", "median", "min"):
            for q in (0.999, 0.99, 0.95, 0.90, 0.75, 0.50):
                rmax, _ = calibrate(a.datasets, q, verbose=False, aggregator=aggregator,
                                    rates=rates)
                dv = dt = 0
                for n in a.probe_datasets:
                    for g in GAPS:
                        r = rates[n][g]
                        dv += int((r > rmax[g]).sum()); dt += len(r)
                rows = probe(a.probe_datasets, a.seeds, rmax, rates=prates)
                mv = sum(v for _, _, v, _, _ in rows if v is not None)
                mt = sum(t for _, _, v, t, _ in rows if v is not None)
                print(f"  {aggregator:>7} {q:>7.3f} | "
                      + " ".join(f"{rmax[g]:11.4f}" for g in GAPS)
                      + f" | {100*dv/max(dt,1):11.3f}% | {100*mv/max(mt,1):12.3f}%")
        print("\n  'data >bound'  = fraction of REAL observed transitions the bound calls implausible"
              " (false positives)")
        print("  'model >bound' = fraction of the trained model's intervals the hinge would touch"
              " (its only chance to act)")
        print(f"  both scored on {list(a.probe_datasets)}; the bound itself is calibrated on "
              f"{list(a.datasets)}")
        print("  --probe --quantile Q --aggregator AGG breaks the chosen bound out per panel.")
        return 0

    if a.calibrate:
        print(f"calibrating r_max at p{a.quantile*100:.0f} ({a.aggregator}) over TRAIN cells, "
              f"gaps {GAPS}")
        rmax, per_ds = calibrate(a.datasets, a.quantile, aggregator=a.aggregator)
        RMAX_PATH.write_text(json.dumps(
            {"r_max": {str(k): v for k, v in rmax.items()},
             "quantile": a.quantile, "datasets": a.datasets, "pairs": [list(p) for p in PAIRS],
             "aggregator": f"{a.aggregator} over datasets of each dataset's per-gap quantile",
             "per_dataset": {n: {str(g): per_ds[n][g][0] for g in GAPS} for n in per_ds}},
            indent=1))
        print(f"\n  r_max = " + "  ".join(f"gap {g}: {rmax[g]:.4f}" for g in GAPS))
        print(f"  written to {RMAX_PATH.relative_to(ROOT)}")
        if not a.probe:
            return 0

    if a.probe:
        # Computed from --quantile/--aggregator, NOT read back from epi_rmax.json: the JSON holds one
        # frozen bound, and the point of a per-panel breakout is to inspect a CANDIDATE bound without
        # overwriting it. At the defaults (p99, max) this reproduces the stored file exactly.
        rmax, _ = calibrate(a.datasets, a.quantile, verbose=False, aggregator=a.aggregator)
        print(f"\nPROBE: does the trained baseline ever exceed r_max? "
              f"(p{a.quantile*100:.0f}, {a.aggregator} over {len(a.datasets)} datasets)  "
              + "  ".join(f"g{g}={rmax[g]:.3f}" for g in GAPS))
        print(f"\n  {'dataset':24} {'seed':>4} | {'violations':>18} | note")
        print("  " + "-" * 74)
        tv = tt = 0
        for name, s, viol, tot, note in probe(a.probe_datasets, a.seeds, rmax):
            if viol is None:
                print(f"  {name:24} {s:>4} | {'--':>18} | {note}")
                continue
            tv += viol; tt += tot
            print(f"  {name:24} {s:>4} | {viol:>8,} / {tot:>7,} | {note}")
        rate = 100 * tv / max(tt, 1)
        print(f"\n  TOTAL {tv:,} / {tt:,} = {rate:.3f}% of constrained intervals exceed r_max")
        print("  Read: ~0% means the hinge is inert and the ablation is null BY CONSTRUCTION "
              "(do not book the GPU night; report the probe instead).")
        return 0

    ap.error("give --calibrate, --probe or --selfcheck")


if __name__ == "__main__":
    sys.exit(main())

"""conformal.py -- cross-disease conformal calibration (lambda_h) + online ACI, for G4.

WHAT THIS FIXES. The quantile head is not calibrated. Recomputed from the archived LDO3 quantiles,
empirical 90% coverage runs 0.492 to 0.918 across (panel, horizon), and it DEGRADES WITH HORIZON on
every panel: covid 0.658 -> 0.492, dengue 0.749 -> 0.678, us-regions 0.918 -> 0.838, us-states
0.815 -> 0.783, japan 0.899 -> 0.887. A single global multiplier would over-correct h3 and
under-correct h15, so every correction here is PER HORIZON.

WHY NOT A CALIBRATION SPLIT OUT OF EBOLA. Arithmetic rules it out before the protocol does. The
frozen primary arm has 48/38/18/0 adaptation pairs at h3/h5/h10/h15. Split conformal needs n >= 9
just to form a FINITE 90% interval (ceil((n+1)*0.9) <= n). Split h3 and ~16 remain, where the index
is ceil(17*0.9) = 16 -- literally the largest residual seen: valid, uselessly wide, high variance.
h10 splits to 9, exactly the boundary. h15 has nothing to split. And the split would halve the
adapter's fitting data, degrading the point forecast to buy an interval that says nothing.

WHY NOT EnbPI. No target-disease data enters training by design, so there are ZERO out-of-bag
residuals for the headline case; bootstrapping the adapter alone prices the wrong variance. Also
~300 GPU-hours.

THE DESIGN. Fit the correction where we DO have data -- the five LDO3 held-out panels -- and
transfer it. Those panels are the right calibration population because each is a frozen trunk plus a
fresh adapter on a disease the trunk never saw, which is exactly what Ebola gets. Quantiles for the
adapted arm are already archived for all five panels at five seeds, so this costs no GPU.

  score      E = max(q_lo - y, y - q_hi) / (q_hi - q_lo + eps)
  fit        lambda_h = conformal (1-alpha) quantile of E pooled over the calibration panels
  apply      [q_lo - lambda_h*w, q_hi + lambda_h*w],  w = q_hi - q_lo
  adapt      alpha_{t+1} = alpha_t + gamma*(alpha - f_t),  lambda re-read at level 1 - alpha_t

E is normalised by the interval width so panels at wildly different count scales are comparable --
dengue's test counts have median 2 and mean 26.8, covid's median 4,605 and mean 8,808, a ~200x gap
that an unnormalised residual would let covid dominate entirely.

THE ACI UPDATE IS ON THE FRACTION, NOT THE INDICATOR. f_t is the miscovered FRACTION across the
districts observed at origin t, not a binary miss. Same telescoping argument and same bound, much
lower variance per step, because each Ebola step is then informed by up to 46 districts instead of
one outcome. (61 districts exist; the most ever observed at a single origin is 46 at h3/h5/h10 and
45 at h15.)

TWO THINGS THAT MUST TRAVEL WITH ANY NUMBER THIS PRODUCES.

  * NO FINITE-SAMPLE GUARANTEE ON EBOLA. lambda_h is fitted on other diseases, so it is a TRANSFER
    of the correction, not a conformal guarantee at the target. The guarantee comes from ACI's
    online adaptation, not from the initialisation. Said here so a reviewer does not have to say it.
  * THE T=18 BOUND. Ebola is scored at 18 origins, so ACI's worst case is
    (alpha_1 + gamma)/(T*gamma) = (0.1 + 0.05)/(18*0.05) = 0.167 -- coverage could sit 17 points off
    nominal in theory. Realised will be far better, but the bound goes in the paper.
    The development panels have 47-630 origins, so their ACI result is OPTIMISTIC relative to
    Ebola's 18 steps. Do not read the dev coverage as a prediction of the Ebola coverage.

LEAVE-ONE-PANEL-OUT IS THE HONEST VALIDATION. Fitting lambda_h on all five panels and scoring those
same five is in-sample and will flatter itself. Ebola is a SIXTH panel the calibration never saw, so
--lopo (fit on four, score the fifth) is the number that estimates what Ebola gets. Both are
reported; the in-sample row is what the frozen artefact contains, the LOPO row is what it is worth.

FROZEN AND HASHED. gamma and alpha_1 are fixed up front and the fitted lambda_h are written to
results/misc/conformal_config.json with a content digest, so the whole wrapper is deterministic
given the data and stays compatible with scoring Ebola exactly once.

    conda run -n ebola-train python -m conformal --selfcheck   # logic only, no data, seconds
    conda run -n ebola-train python -m conformal --fit         # fit lambda_h, validate, freeze
    conda run -n ebola-train python -m conformal --fit --lopo  # leave-one-panel-out validation too

Run as a MODULE from the repo root, or `import bundles` fails.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import math
import sys

import numpy as np

PANELS = ("dengue", "influenza_japan", "influenza_us-regions", "influenza_us-states",
          "covid_us-states")
SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)

ALPHA = 0.10                    # nominal miscoverage -> 90% intervals
GAMMA = 0.05                    # ACI step size, FROZEN (senior dev: 0.02-0.05; 0.05 is the end the
                                # published worst-case bound was quoted at)
EPS = 1e-6                      # guards a zero-width interval; a zero-width MISS then scores huge,
                                # which is the correct reading, not a clipped one
QUANT_LDO3 = "results/lodo/encoder_ldo3__{ds}__seed{s}__quantiles.npz"      # transfer arm
QUANT_SINGLE = "results/single/encoder__{ds}__seed{s}__quantiles.npz"      # single-disease ceiling
OUT_CONFIG = "results/misc/conformal_config.json"


# --------------------------------------------------------------------------- #
# The score, the fit, the wrapper.
# --------------------------------------------------------------------------- #
def nonconformity(y, lo, hi, eps=EPS):
    """E = max(lo - y, y - hi) / (hi - lo + eps). Negative inside the interval, positive outside.

    Width-normalised so panels at different count scales pool into one calibration set."""
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return np.maximum(lo - y, y - hi) / (hi - lo + eps)


def conformal_lambda(sorted_E, alpha, sorted_w=None):
    """The finite-sample conformal quantile of E at level 1-alpha, from a PRE-SORTED array.

    Uses the ceil((n+1)(1-alpha)) order statistic rather than a plain percentile: with the plain
    version the interval is anti-conservative at small n, and small n is the whole problem here.
    Returns +inf when the index exceeds n -- an honestly infinite interval, never a silently
    truncated one.

    `sorted_w` gives per-cell weights aligned to sorted_E, used to hold every PANEL to equal mass.
    Without it the pool is dominated by whichever panel has the most cells, and here that is not a
    close call: dengue contributes 2,045,685 of 2,121,040 cells at h3, 96.4%, because it has 7,165
    nodes over 630 origins against covid's 49 over 37. An unweighted fit is therefore a dengue fit
    wearing a cross-disease label, and it fails exactly where it matters -- held out, dengue's own
    lambda drops to 0.08 and ACI has to open the intervals to infinity to recover. It also breaks
    the project's standing reporting rule (Week-4 work order: never average across datasets).
    The (n_eff+1)/n_eff inflation is the weighted analogue of the +1 order statistic, with
    n_eff = (sum w)^2 / sum(w^2) the effective sample size."""
    n = sorted_E.size
    if n == 0 or alpha <= 0:
        return math.inf
    if alpha >= 1:
        return float(sorted_E[0])
    if sorted_w is None:
        k = math.ceil((n + 1) * (1.0 - alpha))
        return math.inf if k > n else float(sorted_E[k - 1])
    w = np.asarray(sorted_w, float)
    tot = w.sum()
    n_eff = tot * tot / float((w * w).sum())
    target = (1.0 - alpha) * (n_eff + 1.0) / n_eff
    if target > 1.0:
        return math.inf                      # not enough effective mass for a finite interval
    idx = int(np.searchsorted(np.cumsum(w) / tot, target, side="left"))
    return math.inf if idx >= n else float(sorted_E[idx])


def widen(lo, hi, lam):
    """[lo - lam*w, hi + lam*w], w = hi - lo. lam=0 is the identity (the control case)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    w = hi - lo
    return lo - lam * w, hi + lam * w


def aci_run(cells, sorted_E, alpha=ALPHA, gamma=GAMMA, sorted_w=None):
    """Online ACI over an ordered origin stream. Returns (covered, total, mean_width, alpha_trace).

    `cells` is [(y, lo, hi), ...] in origin order -- one entry per origin, each holding every
    district observed at that origin. One update per origin, on the miscovered FRACTION.

    An origin whose alpha_t has been driven to <= 0 gets an INFINITE interval. That is ACI behaving
    as defined, not a bug, but it makes the mean width meaningless, so `n_inf` is counted and
    reported rather than left to poison the width column silently."""
    a_t = alpha
    cov = tot = n_inf = 0
    wsum = 0.0
    trace = []
    for y, lo, hi in cells:
        lam = conformal_lambda(sorted_E, a_t, sorted_w)
        n_inf += 0 if np.isfinite(lam) else 1
        L, H = widen(lo, hi, lam) if np.isfinite(lam) else (
            np.full_like(lo, -np.inf), np.full_like(hi, np.inf))
        inside = (y >= L) & (y <= H)
        f_t = 1.0 - float(inside.mean()) if inside.size else 0.0
        cov += int(inside.sum()); tot += int(inside.size)
        wsum += float(np.sum(H - L)) if np.isfinite(lam) else np.inf
        trace.append(a_t)
        a_t = a_t + gamma * (alpha - f_t)          # the telescoping update
    trace.append(a_t)                              # T+1 entries: the alpha each step USED, then the
    return cov, tot, (wsum / tot if tot else float("nan")), trace, n_inf   # one the next would use


def aci_worst_case(T, alpha=ALPHA, gamma=GAMMA):
    """(alpha_1 + gamma)/(T*gamma) -- ACI's worst-case coverage deviation over T steps."""
    return (alpha + gamma) / (T * gamma)


# --------------------------------------------------------------------------- #
# Reading the archived quantiles.
# --------------------------------------------------------------------------- #
def load_panel(ds, seed, bundle, family=QUANT_LDO3):
    """{h: [(y, lo, hi), ...]} in origin order, observed test cells only, count space."""
    path = family.format(ds=ds, s=seed)
    if not os.path.exists(path):
        return None
    z = np.load(path)
    te = z["origins"]
    raw = bundle.raw.astype(np.float64)
    pm = bundle.masks()["test"].astype(bool)
    out = {}
    for h in HORIZONS:
        q = np.sort(z[f"h{h}__quantiles"], axis=-1)        # [N, K, Q]; head can emit crossings
        stream = []
        for k, t in enumerate(te):
            m = pm[:, t + h]
            if not m.any():
                continue
            stream.append((raw[m, t + h], q[m, k, 0], q[m, k, -1]))
        out[h] = stream
    return out


def load_all(verbose=True, family=QUANT_LDO3):
    """{(panel, seed): {h: stream}} for every archived panel of one family."""
    from bundles import load
    data = {}
    for ds in PANELS:
        b = load(ds)
        for s in SEEDS:
            p = load_panel(ds, s, b, family)
            if p is None:
                print(f"  MISSING {family.format(ds=ds, s=s)}")
                continue
            data[(ds, s)] = p
        if verbose:
            n = sum(len(v[HORIZONS[0]]) for (d, _), v in data.items() if d == ds)
            print(f"  loaded {ds:24s} {n} origin-steps over "
                  f"{sum(1 for d, _ in data if d == ds)} seeds")
    return data


def raw_coverage(stream):
    """(coverage, mean width, n) for an uncalibrated interval stream. The ceiling needs no wrapper:
    it is the reference, not a thing we are correcting."""
    cov = tot = 0
    w = 0.0
    for y, lo, hi in stream:
        cov += int(((y >= lo) & (y <= hi)).sum()); tot += int(y.size)
        w += float(np.sum(hi - lo))
    return (cov / tot if tot else float("nan")), (w / tot if tot else float("nan")), tot


def pooled_scores(data, panels, seeds=SEEDS):
    """{h: (sorted E, sorted weights)} over the calibration panels, each panel carrying equal mass.

    Sorted once and reused by every lookup, because ACI re-reads this distribution at a new alpha_t
    on every origin."""
    out = {}
    for h in HORIZONS:
        Es, Ws = [], []
        for ds in panels:
            chunks = [nonconformity(y, lo, hi)
                      for (d, s), p in data.items() if d == ds and s in seeds
                      for y, lo, hi in p[h]]
            if not chunks:
                continue
            e = np.concatenate(chunks)
            Es.append(e)
            Ws.append(np.full(e.size, 1.0 / e.size))       # each panel sums to 1, whatever its size
        if not Es:
            out[h] = (np.array([]), np.array([])); continue
        e, w = np.concatenate(Es), np.concatenate(Ws)
        o = np.argsort(e)
        out[h] = (e[o], w[o])
    return out


# --------------------------------------------------------------------------- #
# Evaluation.
# --------------------------------------------------------------------------- #
def evaluate(data, calib_panels, eval_panels, alpha=ALPHA, gamma=GAMMA):
    """Three arms per (panel, horizon): raw head, + static lambda_h, + lambda_h then online ACI.

    Coverage is over observed cells; width is the mean interval width in count space."""
    E = pooled_scores(data, calib_panels)
    lam = {h: conformal_lambda(E[h][0], alpha, E[h][1]) for h in HORIZONS}
    rows = {}
    for ds in eval_panels:
        for h in HORIZONS:
            raw_c, stat_c, tot, raw_w, stat_w = 0, 0, 0, 0.0, 0.0
            aci_c, aci_t, aci_w, n_inf, n_steps = 0, 0, [], 0, 0
            for s in SEEDS:
                p = data.get((ds, s))
                if p is None:
                    continue
                for y, lo, hi in p[h]:
                    raw_c += int(((y >= lo) & (y <= hi)).sum())
                    raw_w += float(np.sum(hi - lo))
                    L, H = widen(lo, hi, lam[h])
                    stat_c += int(((y >= L) & (y <= H)).sum())
                    stat_w += float(np.sum(H - L))
                    tot += int(y.size)
                c, t, w, _, ninf = aci_run(p[h], E[h][0], alpha, gamma, E[h][1])
                aci_c += c; aci_t += t; aci_w.append(w)
                n_inf += ninf; n_steps += len(p[h])
            if tot:
                rows[(ds, h)] = dict(
                    n=tot, lam=lam[h],
                    raw_cov=raw_c / tot, raw_w=raw_w / tot,
                    stat_cov=stat_c / tot, stat_w=stat_w / tot,
                    aci_cov=aci_c / aci_t if aci_t else float("nan"),
                    aci_w=float(np.mean(aci_w)) if aci_w else float("nan"),
                    inf_frac=n_inf / n_steps if n_steps else float("nan"))
    return lam, rows


def print_table(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'panel':22s} {'h':>3s} {'n':>8s} {'lambda':>7s} | "
          f"{'raw cov':>8s} {'+lam':>8s} {'+ACI':>8s} | {'raw w':>10s} {'+lam w':>10s} "
          f"{'+ACI w':>10s} {'inf%':>6s}")
    for (ds, h), r in rows.items():
        print(f"{ds:22s} {h:3d} {r['n']:8d} {r['lam']:7.3f} | "
              f"{r['raw_cov']:8.3f} {r['stat_cov']:8.3f} {r['aci_cov']:8.3f} | "
              f"{r['raw_w']:10.1f} {r['stat_w']:10.1f} {r['aci_w']:10.1f} "
              f"{100 * r['inf_frac']:6.1f}")
    for key, label in (("raw_cov", "raw"), ("stat_cov", "+lambda"), ("aci_cov", "+ACI")):
        v = [r[key] for r in rows.values()]
        print(f"  {label:8s} coverage range {min(v):.3f} .. {max(v):.3f}   "
              f"mean abs dev from 0.90 {np.mean([abs(x - 0.90) for x in v]):.3f}")


def load_frozen(path=OUT_CONFIG):
    """Read the frozen wrapper back, and refuse it if its digest does not match its own numbers.

    The point of freezing gamma/alpha_1/lambda_h is that Ebola can be scored once against a fixed
    configuration. A config that was edited after the fact would silently break exactly that claim,
    so the digest is re-derived here rather than trusted."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    claimed = cfg.pop("sha256_content", None)
    body = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    got = hashlib.sha256(body).hexdigest()
    if claimed != got:
        raise SystemExit(f"{path}: digest mismatch -- config edited after freezing.\n"
                         f"  claimed {claimed}\n  actual  {got}")
    cfg["sha256_content"] = got
    return cfg


def apply_to_archive(quant_npz, bundle, phase, cfg, arm_label):
    """Run the FROZEN wrapper over one archived quantile file. Returns a row per horizon.

    Deliberately post-hoc and read-only: it consumes `__quantiles.npz` and the bundle's own truth,
    touches no model and no point forecast, and so satisfies Ebola_Prereg.md §5b's constraint that
    the calibration layer never moves E1-E5. Raw intervals are reported beside the calibrated ones,
    never instead of them -- also §5b."""
    z = np.load(quant_npz)
    te = z["origins"]
    raw = bundle.raw.astype(np.float64)
    pm = bundle.masks()[phase].astype(bool)
    lam_by_h = {int(k): v for k, v in cfg["lambda_by_horizon"].items()}
    alpha, gamma = cfg["alpha_1"], cfg["gamma"]
    rows = []
    for h in HORIZONS:
        q = np.sort(z[f"h{h}__quantiles"], axis=-1)
        stream = []
        for k, t in enumerate(te):
            if t + h >= raw.shape[1]:
                continue
            m = pm[:, t + h]
            if m.any():
                stream.append((raw[m, t + h], q[m, k, 0], q[m, k, -1]))
        if not stream:
            continue
        lam = lam_by_h[h]
        rawc = rawn = statc = 0
        raww = statw = 0.0
        for y, lo, hi in stream:
            rawc += int(((y >= lo) & (y <= hi)).sum()); rawn += int(y.size)
            raww += float(np.sum(hi - lo))
            L, H = widen(lo, hi, lam)
            statc += int(((y >= L) & (y <= H)).sum()); statw += float(np.sum(H - L))
        # ACI re-reads lambda at 1 - alpha_t, so it needs the calibration DISTRIBUTION, not just the
        # frozen point. Reconstructed from the frozen lambda alone it could only ever be static --
        # so the E pool is refitted from the same calibration panels the config names.
        E = _calibration_pool(cfg)
        acic, acin, aciw, _, ninf = aci_run(stream, E[h][0], alpha, gamma, E[h][1])
        rows.append(dict(arm=arm_label, h=h, T=len(stream), n=rawn,
                         lam=lam, raw_cov=rawc / rawn, raw_w=raww / rawn,
                         stat_cov=statc / rawn, stat_w=statw / rawn,
                         aci_cov=acic / acin if acin else float("nan"), aci_w=aciw,
                         inf_frac=ninf / len(stream)))
    return rows


_POOL_CACHE = {}


def _calibration_pool(cfg):
    """The E distribution ACI adapts over, rebuilt from the panels the frozen config names."""
    key = (tuple(cfg["calibration_panels"]), tuple(cfg["calibration_seeds"]))
    if key not in _POOL_CACHE:
        data = load_all(verbose=False)
        _POOL_CACHE[key] = pooled_scores(data, cfg["calibration_panels"],
                                         tuple(cfg["calibration_seeds"]))
    return _POOL_CACHE[key]


def freeze(lam, path=OUT_CONFIG):
    """Write the frozen wrapper and a content digest over its own numbers."""
    cfg = dict(alpha=ALPHA, alpha_1=ALPHA, gamma=GAMMA, eps=EPS,
               quantile_levels=[0.05, 0.95], nominal_coverage=1 - ALPHA,
               lambda_by_horizon={str(h): lam[h] for h in HORIZONS},
               calibration_panels=list(PANELS), calibration_seeds=list(SEEDS),
               aci_update="fraction: alpha_{t+1} = alpha_t + gamma*(alpha - f_t), f_t in [0,1]",
               aci_worst_case_T18=aci_worst_case(18),
               guarantee="NO finite-sample guarantee on Ebola: lambda_h is fitted on other "
                         "diseases and TRANSFERRED. The guarantee comes from ACI's online "
                         "adaptation, not from this initialisation.")
    body = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    cfg["sha256_content"] = hashlib.sha256(body).hexdigest()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg


# --------------------------------------------------------------------------- #
def _selfcheck():
    """Every rule that can silently do the wrong thing, each paired with a case that must come out
    the other way. No data, no GPU."""
    # 1. the score: sign, and the width normalisation that makes panels poolable
    assert nonconformity(5.0, 0.0, 10.0) < 0, "inside the interval must score negative"
    assert nonconformity(15.0, 0.0, 10.0) > 0, "outside must score positive"
    assert abs(nonconformity(15.0, 0.0, 10.0) - 0.5) < 1e-6, "5 over a width of 10 is 0.5"
    big = nonconformity(15000.0, 0.0, 10000.0)          # same shape, 1000x the counts
    assert abs(big - 0.5) < 1e-6, "score must be scale-free or covid swamps dengue in the pool"

    # 2. the conformal index, including the small-n regime that killed the Ebola split
    E = np.sort(np.arange(1, 17, dtype=float))           # n=16
    assert conformal_lambda(E, 0.1) == 16.0, "n=16 -> ceil(17*0.9)=16 -> the MAX residual"
    assert math.isinf(conformal_lambda(np.sort(np.arange(1, 9.0)), 0.1)), \
        "n=8 -> index 9 > n -> must be INFINITE, never silently the max"
    assert conformal_lambda(np.sort(np.arange(1, 10.0)), 0.1) == 9.0, "n=9 is the boundary case"
    assert math.isinf(conformal_lambda(E, 0.0)), "alpha=0 is an infinite interval"

    # 3. the wrapper, and its control
    lo, hi = np.array([0.0]), np.array([10.0])
    L, H = widen(lo, hi, 0.5)
    assert (L == -5.0) and (H == 15.0), "lam=0.5 on width 10 extends 5 each side"
    L0, H0 = widen(lo, hi, 0.0)
    assert (L0 == lo) and (H0 == hi), "CONTROL: lam=0 must be the identity"

    # 4. calibration actually calibrates: 60%-covering intervals must come out near 90%
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20000)
    lo_c, hi_c = np.full(20000, -0.84), np.full(20000, 0.84)      # ~60% coverage by construction
    assert 0.55 < ((y >= lo_c) & (y <= hi_c)).mean() < 0.65, "fixture is not ~60%"
    Es = np.sort(nonconformity(y[:10000], lo_c[:10000], hi_c[:10000]))
    lam = conformal_lambda(Es, 0.1)
    L, H = widen(lo_c[10000:], hi_c[10000:], lam)
    got = ((y[10000:] >= L) & (y[10000:] <= H)).mean()
    assert 0.88 < got < 0.92, f"calibrated coverage {got:.3f} not near 0.90"

    # 5. ACI: a stream that always misses must drive alpha DOWN (intervals widen), and the
    #    always-covered control must drive it UP. A sign error here is invisible in the mean.
    #    E must have enough points for a FINITE lambda (n=20 -> index 19), or nothing can miss and
    #    the test passes for the wrong reason -- which is exactly how this fixture failed first.
    Ecal = np.sort(np.linspace(0.0, 1.0, 20))
    assert np.isfinite(conformal_lambda(Ecal, 0.1)), "fixture must give a finite lambda"
    miss = [(np.array([1e6]), np.array([0.0]), np.array([1.0])) for _ in range(20)]
    _, _, _, tr, _ = aci_run(miss, Ecal, 0.1, 0.05)
    assert tr[-1] < tr[0], "persistent misses must lower alpha_t (widen)"
    hit = [(np.array([0.5]), np.array([0.0]), np.array([1.0])) for _ in range(20)]
    _, _, _, tr2, _ = aci_run(hit, Ecal, 0.1, 0.05)
    assert tr2[-1] > tr2[0], "CONTROL: persistent hits must raise alpha_t (tighten)"

    # 6. the fraction update is NOT the indicator update. Half-missed origin, run through the real
    #    aci_run: its step must be the midpoint of the all-hit and all-miss steps. Under a binary
    #    indicator a half-miss counts as a full miss, so that update fails this and this one passes.
    def step(ys):
        s = [(np.array(ys), np.zeros(len(ys)), np.ones(len(ys)))]
        return aci_run(s, Ecal, 0.1, 0.05)[3][1]                  # alpha AFTER the single origin
    a_none, a_half, a_full = step([0.5, 0.5]), step([0.5, 1e6]), step([1e6, 1e6])
    assert abs(a_half - (a_none + a_full) / 2) < 1e-12, \
        "a half-missed origin must land midway between all-hit and all-miss"
    assert a_full < a_half < a_none, "more misses must mean a lower alpha_t"

    # 7. the published bound
    assert abs(aci_worst_case(18) - 0.16666666) < 1e-6, "T=18 worst case must be 0.167"

    # 8. panel balance. One huge well-calibrated panel and one tiny badly-calibrated one, the real
    #    shape of this pool (dengue is 96.4% of the cells). UNWEIGHTED must follow the big panel --
    #    that is the bug -- and WEIGHTED must be pulled toward the small one.
    big, small = np.zeros(100_000), np.full(1_000, 10.0)
    e = np.concatenate([big, small]); w = np.concatenate([
        np.full(big.size, 1.0 / big.size), np.full(small.size, 1.0 / small.size)])
    o = np.argsort(e); e, w = e[o], w[o]
    lam_un, lam_w = conformal_lambda(e, 0.1), conformal_lambda(e, 0.1, w)
    assert lam_un == 0.0, "CONTROL: unweighted follows the big panel and ignores the small one"
    assert lam_w == 10.0, f"weighted must reach the small panel's scores, got {lam_w}"
    n_eff = (w.sum() ** 2) / (w * w).sum()
    assert n_eff < e.size / 20, \
        f"n_eff must collapse toward the SMALL panel's cell count, got {n_eff:.0f} of {e.size}"

    print("ok  score: sign correct, and scale-free (dengue ~20 and covid ~5,000 pool honestly)")
    print("ok  conformal index: n=16 -> the max residual, n=8 -> INFINITE, n=9 the boundary")
    print("ok  wrapper: widens by lam*width; lam=0 is the identity")
    print("ok  calibration: a 60% interval comes out at 90% on held-out draws")
    print("ok  ACI: misses widen, hits tighten, and the fraction update scales with the fraction")
    print(f"ok  bound: (alpha_1+gamma)/(T*gamma) at T=18 = {aci_worst_case(18):.3f}")
    print("ok  panel balance: unweighted follows the 100x panel, weighted reaches the small one")


def _reference(a):
    """Our calibrated TRANSFER intervals against the single-disease CEILING's own intervals.

    This is the comparison the ceiling re-train was paid for. The ceiling is a model trained on the
    panel's own disease, so its raw coverage is the best calibration a frozen quantile head reaches
    when there is no domain shift at all. The question is how close cross-disease transfer plus the
    conformal wrapper gets to it, and what that costs in width.

    The ceiling is NOT wrapped. Calibrating the reference would make it a different reference.

    Both families are scored on the same bundle's test origins, so the cells are identical -- which
    is asserted per (panel, seed) rather than assumed, because a mismatch would silently compare two
    different populations and still print a plausible table.
    """
    from bundles import load

    print("loading TRANSFER arm (LDO3 adapted)")
    tdata = load_all(family=QUANT_LDO3)
    if not tdata:
        sys.exit("no LDO3 quantile archives found")
    _, trows = evaluate(tdata, PANELS, PANELS, ALPHA, a.gamma)

    print("\nloading CEILING arm (single-disease)")
    rows, missing, mismatched = [], [], []
    for ds in PANELS:
        b = load(ds)
        for h in HORIZONS:
            covs, widths, ns = [], [], []
            for s in SEEDS:
                p = load_panel(ds, s, b, QUANT_SINGLE)
                if p is None:
                    missing.append(QUANT_SINGLE.format(ds=ds, s=s)); continue
                # same origins as the transfer arm, or the two columns are not comparable
                a_o = np.load(QUANT_SINGLE.format(ds=ds, s=s))["origins"]
                b_o = np.load(QUANT_LDO3.format(ds=ds, s=s))["origins"]
                if not np.array_equal(a_o, b_o):
                    mismatched.append(f"{ds} seed{s}"); continue
                c, w, n = raw_coverage(p[h])
                covs.append(c); widths.append(w); ns.append(n)
            if covs:
                rows.append(dict(ds=ds, h=h, cov=float(np.mean(covs)), sd=float(np.std(covs, ddof=1)),
                                 w=float(np.mean(widths)), n=ns[0], seeds=len(covs)))
        print(f"  {ds:24s} {sum(1 for r in rows if r['ds'] == ds)}/{len(HORIZONS)} horizons")

    if missing:
        print(f"\n{len(missing)} ceiling archive(s) missing: {missing[:3]}")
    if mismatched:
        print(f"\nORIGIN MISMATCH, excluded: {mismatched}\n  the two arms scored different cells; "
              f"the comparison would be meaningless on these.")
    if not rows:
        sys.exit("no ceiling archives to compare against")

    print(f"\n=== CEILING vs CALIBRATED TRANSFER, 90% intervals, {len(SEEDS)} seeds ===")
    print(f"{'panel':22s} {'h':>3s} | {'ceiling':>14s} {'ceil w':>10s} | "
          f"{'transf raw':>10s} {'+lam+ACI':>9s} {'cal w':>10s} | {'gap to ceil':>11s}")
    for r in rows:
        t = trows.get((r["ds"], r["h"]))
        if t is None:
            continue
        gap = t["aci_cov"] - r["cov"]
        print(f"{r['ds']:22s} {r['h']:3d} | {r['cov']:8.3f}±{r['sd']:5.3f} {r['w']:10.1f} | "
              f"{t['raw_cov']:10.3f} {t['aci_cov']:9.3f} {t['aci_w']:10.1f} | {gap:+11.3f}")

    ceil = [r["cov"] for r in rows]
    cal = [trows[(r["ds"], r["h"])]["aci_cov"] for r in rows if (r["ds"], r["h"]) in trows]
    print(f"\n  ceiling  coverage range {min(ceil):.3f} .. {max(ceil):.3f}   "
          f"mean abs dev from 0.90 {np.mean([abs(x - 0.90) for x in ceil]):.3f}")
    print(f"  calibrated transfer      {min(cal):.3f} .. {max(cal):.3f}   "
          f"mean abs dev from 0.90 {np.mean([abs(x - 0.90) for x in cal]):.3f}")
    print("\n  READ THIS AS: the ceiling is the reference, NOT a target to beat. Calibrated transfer\n"
          "  landing closer to 0.90 than the ceiling does means the ceiling is itself miscalibrated,\n"
          "  which is a finding about the quantile head, not evidence that transfer forecasts better.")


def _apply(a):
    """Apply the frozen wrapper to the Ebola archives. Post-hoc, read-only, writes nothing.

    Both arms and both adaptation regimes are reported: the few-shot arm is the pre-registered
    headline, the zero-shot arm is the transfer floor, and §5b requires the raw intervals to appear
    beside the calibrated ones rather than be replaced by them."""
    from bundles import load
    from results_paths import rpath

    cfg = load_frozen()
    print(f"frozen wrapper {OUT_CONFIG}  digest ok  gamma={cfg['gamma']}  alpha_1={cfg['alpha_1']}")
    print(f"  lambda_h = { {k: round(v, 4) for k, v in cfg['lambda_by_horizon'].items()} }")
    print(f"  calibrated on {cfg['calibration_panels']}\n")

    rows, missing = [], []
    for arm in a.arms:
        b = load(arm)
        for seed in a.seeds:
            for fam, label in ((a.prefix, "few-shot"), (f"{a.prefix}_zeroshot", "zero-shot")):
                fn = f"{fam}__{arm}__seed{seed}__quantiles.npz"
                p = rpath(fn)
                if not os.path.exists(p):
                    missing.append(fn); continue
                rows += apply_to_archive(p, b, "query", cfg, f"{arm} {label} s{seed}")
    if missing:
        print(f"{len(missing)} archive(s) not on disk (run train.ebola first):")
        for m in missing[:6]:
            print(f"  {m}")
        if len(missing) > 6:
            print(f"  ... and {len(missing) - 6} more")
    if not rows:
        sys.exit("no Ebola quantile archives found -- nothing to apply the wrapper to")

    print(f"\n{'arm':30s} {'h':>3s} {'T':>3s} {'n':>7s} | "
          f"{'raw cov':>8s} {'+lam':>8s} {'+ACI':>8s} | {'raw w':>10s} {'+lam w':>10s} "
          f"{'+ACI w':>10s} {'inf%':>6s}")
    for r in rows:
        print(f"{r['arm']:30s} {r['h']:3d} {r['T']:3d} {r['n']:7d} | "
              f"{r['raw_cov']:8.3f} {r['stat_cov']:8.3f} {r['aci_cov']:8.3f} | "
              f"{r['raw_w']:10.1f} {r['stat_w']:10.1f} {r['aci_w']:10.1f} "
              f"{100 * r['inf_frac']:6.1f}")

    T = max(r["T"] for r in rows)
    print(f"\nACI worst case at T={T}: {aci_worst_case(T, cfg['alpha_1'], cfg['gamma']):.3f} "
          f"-- coverage may sit this far from nominal in theory.")
    print(f"NO FINITE-SAMPLE GUARANTEE ON EBOLA: {cfg['guarantee']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--fit", action="store_true", help="fit lambda_h, validate, freeze")
    ap.add_argument("--lopo", action="store_true", help="also run leave-one-panel-out validation")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--apply", action="store_true",
                    help="apply the FROZEN wrapper to the Ebola archives (post-hoc, read-only)")
    ap.add_argument("--reference", action="store_true",
                    help="calibrated transfer vs the single-disease ceiling's own intervals")
    ap.add_argument("--prefix", default="encoder_ebola",
                    help="archive family to apply to; encoder_ebola_smoke for a dry run")
    ap.add_argument("--arms", nargs="+", default=["ebola_L12", "ebola_L20"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args()

    if a.selfcheck:
        _selfcheck(); return
    if a.apply:
        _apply(a); return
    if a.reference:
        _reference(a); return
    if not a.fit:
        ap.error("nothing to do: pass --selfcheck, --fit, --apply or --reference")

    print("loading archived LDO3 quantiles (adapted arm, 5 panels x 5 seeds)")
    data = load_all()
    if not data:
        sys.exit("no quantile archives found")

    lam, rows = evaluate(data, PANELS, PANELS, ALPHA, a.gamma)
    print_table(rows, "IN-SAMPLE (fit on all 5 panels, scored on the same 5) -- the frozen artefact")

    if a.lopo:
        lopo = {}
        for held in PANELS:
            calib = tuple(p for p in PANELS if p != held)
            _, r = evaluate(data, calib, (held,), ALPHA, a.gamma)
            lopo.update(r)
        print_table(lopo, "LEAVE-ONE-PANEL-OUT (fit on 4, scored on the 5th) -- what Ebola gets")

    cfg = freeze(lam)
    print(f"\nfrozen -> {OUT_CONFIG}")
    print(f"  gamma={cfg['gamma']}  alpha_1={cfg['alpha_1']}  "
          f"lambda={ {h: round(lam[h], 4) for h in HORIZONS} }")
    print(f"  sha256_content={cfg['sha256_content']}")
    print(f"  ACI worst case at Ebola's T=18: {aci_worst_case(18):.3f}")
    print("  NOTE: dev panels have 47-630 origins; Ebola has 18. The dev ACI rows are OPTIMISTIC.")


if __name__ == "__main__":
    main()

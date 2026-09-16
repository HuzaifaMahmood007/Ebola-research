"""STEP 0: is the Ebola few-shot adapter fit on trunk features it never sees again at scoring time?

The withdrawn explanation for "few-shot hurts on Ebola" was over-parameterisation, and it does not
reproduce. This asks a different question that costs no training at all.

`ebola_L12`'s support mask IS the observation mask over the first 13 columns -- it is not a sampling
design laid over an observed panel. So a cell that is "not support" inside the fit window was NEVER
OBSERVED: no label, input incidence 0, obs_mask channel 0. And `train.ebola.support_origins` returns
t in [0..9], whose 20-week input windows are mostly zero left-pad (models/windows.py:18-20 already
flags "~89% pad" at h10). The scored query origins t in [19,36] have full, densely observed windows.

With the trunk frozen the Adapter is EXACTLY affine in trunk features (models/adapters.py:14-16). So
the few-shot arm is a 64-dim affine fit whose design matrix may live in a degenerate corner of feature
space, evaluated somewhere else entirely. That is a mechanism orthogonal to sample size, which is
exactly why a parameter-counting story failed.

This script measures the gap directly on checkpoints already on disk. It reads NO Ebola label and
scores nothing: the prediction column compares two adapters against each other, never against truth,
so the pre-registration's score-once rule is untouched.

  conda run -n ebola-train python -m diagnostics.ebola_feature_shift --selfcheck
  conda run -n ebola-train python -m diagnostics.ebola_feature_shift --seeds 42 52 62 72 82
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import json

import numpy as np
import torch

import bundles
from bundles import HORIZONS
from models import Adapter, SharedEncoder, sparse_from_dense_np
from models.config import MEDIAN_IDX
from results_paths import RESULTS, rpath
from train.ebola import _precompute_features, support_origins
from train.loop import DEVICE

ARMS = ("ebola_L12", "ebola_L20")
SEEDS = (42, 52, 62, 72, 82)
PREFIX = "encoder_ebola"

# The frozen pair profile, from configs/ebola_arms.json. Asserted, not trusted: if the arm on disk
# has moved, every number below is about a different experiment.
EXPECTED_PAIRS = {"ebola_L12": {3: 48, 5: 38, 10: 18, 15: 0},
                  "ebola_L20": {3: 102, 5: 92, 10: 72, 15: 54}}


def _load(arm, seed, device):
    """(encoder, mean adapter, fitted adapter) for one arm-seed. All three already exist on disk."""
    trunk = torch.load(rpath(f"{PREFIX}__alldev__seed{seed}__ckpt.pt", root=RESULTS),
                       map_location=device, weights_only=False)
    fitted = torch.load(rpath(f"{PREFIX}__{arm}__seed{seed}__ckpt.pt", root=RESULTS),
                        map_location=device, weights_only=False)
    enc = SharedEncoder(gate_mode="learned").to(device); enc.load_state_dict(trunk["encoder"]); enc.eval()
    # the all-dev trunk's `adapter` slot holds _mean_adapter's output -- the zero-shot head (ebola.py:292)
    zero = Adapter().to(device); zero.load_state_dict(trunk["adapter"]); zero.eval()
    few = Adapter().to(device); few.load_state_dict(fitted["adapter"]); few.eval()
    return enc, zero, few


def design_rows(feats, mask, origins, d):
    """{h: [n_rows, d]} -- the feature rows a fit (or an eval) at horizon h actually touches.

    A row exists for (node i, origin t) exactly when the phase mask has a target at column t+h. That
    is the same predicate `targets_and_mask` uses to build the loss weight (models/windows.py:32), so
    these are literally the rows of the affine problem, not a proxy for them."""
    out, T = {}, mask.shape[1]
    for h in HORIZONS:
        rows = [feats[t][np.where(mask[:, t + h])[0]] for t in origins
                if t + h < T and mask[:, t + h].any()]
        out[h] = (torch.cat(rows).double().cpu().numpy() if rows else np.zeros((0, d)))
    return out


def compare(F_sup, F_qry):
    """How far is the fit region from the eval region, and is the fit region even full rank?"""
    n_sup, n_qry = len(F_sup), len(F_qry)
    if n_sup == 0 or n_qry == 0:
        return dict(n_sup=n_sup, n_qry=n_qry, rank=0, cond=float("nan"),
                    shift_med=float("nan"), shift_max=float("nan"),
                    norm_ratio=float("nan"), outside_box=float("nan"),
                    outside_dims=float("nan"))
    sd = F_qry.std(0); sd[sd < 1e-12] = np.nan                  # a dead dim shifts by nothing measurable
    shift = np.abs(F_sup.mean(0) - F_qry.mean(0)) / sd
    lo, hi = F_sup.min(0), F_sup.max(0)
    beyond = (F_qry < lo) | (F_qry > hi)
    # `outside_box` (ANY dim beyond) is weak evidence on its own: with 18-102 support rows in 64 dims
    # the per-dim box is necessarily tight, so it saturates at ~100% almost by construction.
    # `outside_dims` -- the average SHARE of dims a query row is beyond on -- does not saturate and is
    # the number to quote.
    return dict(
        n_sup=n_sup, n_qry=n_qry,
        rank=int(np.linalg.matrix_rank(F_sup)),
        cond=float(np.linalg.cond(F_sup)),
        shift_med=float(np.nanmedian(shift)), shift_max=float(np.nanmax(shift)),
        norm_ratio=float(np.linalg.norm(F_sup, axis=1).mean() /
                         np.linalg.norm(F_qry, axis=1).mean()),
        outside_box=float(beyond.any(1).mean()),
        outside_dims=float(beyond.mean()))


def pred_gap(zero, few, F_qry, h_idx, device):
    """Median |few-shot - zero-shot| on query features, in units of the zero-shot spread.

    Label-free by construction: it contrasts two adapters, never a truth value. Big here means
    adaptation moved the forecast a long way; paired with a big feature shift that movement is
    extrapolation, not learning."""
    if len(F_qry) == 0:
        return float("nan")
    x = torch.tensor(F_qry, dtype=torch.float32, device=device)
    with torch.no_grad():
        a = zero(x)[:, h_idx, MEDIAN_IDX].cpu().numpy()
        b = few(x)[:, h_idx, MEDIAN_IDX].cpu().numpy()
    s = a.std()
    return float(np.median(np.abs(b - a)) / s) if s > 1e-12 else float("nan")


def window_occupancy(b, origins):
    """How much real data is in the input windows at these origins? (pad fraction, observed fraction)"""
    M = b.M.astype(bool); N, T = M.shape
    W = bundles.W
    pad, obs = [], []
    for t in origins:
        lo = max(0, t - (W - 1))
        real = M[:, lo:t + 1]
        pad.append((W - real.shape[1]) / W)
        obs.append(real.mean() if real.size else 0.0)
    return float(np.mean(pad)), float(np.mean(obs))


def run(seeds, device):
    rows, payload = [], {}
    for arm in ARMS:
        b = bundles.load(arm)
        Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=device)
        Mt = torch.tensor(b.M, dtype=torch.float32, device=device)
        A = sparse_from_dense_np(b.A_geo).to(device)
        smask = b.masks()["support"].astype(bool)
        qmask = b.masks()["query"].astype(bool)
        so, te = support_origins(b), b.origins(phase="query")

        got = {h: int(smask[:, h:].sum()) for h in HORIZONS}
        assert got == EXPECTED_PAIRS[arm], f"{arm}: support pairs {got} != frozen {EXPECTED_PAIRS[arm]}"

        s_pad, s_obs = window_occupancy(b, so)
        q_pad, q_obs = window_occupancy(b, te)
        print(f"\n{arm}: support origins {so[0]}..{so[-1]} (pad {s_pad:.0%}, observed {s_obs:.1%})"
              f"  |  query origins {te[0]}..{te[-1]} (pad {q_pad:.0%}, observed {q_obs:.1%})")

        for seed in seeds:
            enc, zero, few = _load(arm, seed, device)
            d = Adapter().gamma.numel()
            F_sup = design_rows(_precompute_features(enc, Z, A, Mt, so, device), smask, so, d)
            F_qry = design_rows(_precompute_features(enc, Z, A, Mt, te, device), qmask, te, d)
            for j, h in enumerate(HORIZONS):
                m = compare(F_sup[h], F_qry[h])
                m.update(arm=arm, seed=seed, h=h,
                         pred_gap=pred_gap(zero, few, F_qry[h], j, device))
                rows.append(m)
        payload[arm] = dict(support_origins=[so[0], so[-1]], query_origins=[te[0], te[-1]],
                            support_pad=s_pad, support_obs=s_obs, query_pad=q_pad, query_obs=q_obs)

    payload["cells"] = rows
    return rows, payload


def table(rows):
    def g(cells, k):
        v = [x[k] for x in cells if not np.isnan(x[k])]
        return float(np.mean(v)) if v else float("nan")         # all-nan cell (h15 on L12) -> nan, no warning

    fmt = lambda v, s: (s % v) if np.isfinite(v) else "-"
    print(f"\n{'arm':<11}{'h':>3}{'n_sup':>7}{'rank':>6}{'cond':>11}{'shift med':>11}"
          f"{'shift max':>11}{'norm':>7}{'out dims':>10}{'pred gap':>10}")
    print("-" * 87)
    for arm in ARMS:
        for h in HORIZONS:
            c = [r for r in rows if r["arm"] == arm and r["h"] == h]
            if not c:
                continue
            print(f"{arm:<11}{h:>3}{c[0]['n_sup']:>7}{g(c,'rank'):>6.1f}"
                  f"{fmt(g(c,'cond'), '%.2e'):>11}"
                  f"{fmt(g(c,'shift_med'), '%.2f'):>11}{fmt(g(c,'shift_max'), '%.2f'):>11}"
                  f"{fmt(g(c,'norm_ratio'), '%.2f'):>7}"
                  f"{fmt(g(c,'outside_dims')*100, '%.1f%%'):>10}"
                  f"{fmt(g(c,'pred_gap'), '%.2f'):>10}")
    print("""
read:
  n_sup      rows in the affine design matrix at this horizon (the fit sees exactly these)
  rank       numerical rank of that matrix, out of 64 feature dims. rank < 64 means the fit is
             underdetermined and the unconstrained directions are set by init + weight decay alone
  cond       condition number. huge means near-collinear rows, so small label noise moves the fit a lot
  shift med  median |mean(support) - mean(query)| per feature dim, in units of the QUERY sd.
             1.0 = the average support feature sits a full query standard deviation away
  shift max  worst dim, same units
  norm       mean ||support feature|| / mean ||query feature||. far from 1.0 = different scale entirely
  out dims   average SHARE OF THE 64 DIMS on which a query row falls outside the support rows'
             min/max range. this is the extrapolation measure to quote. the cruder "any dim outside"
             version is in the JSON as outside_box, and it saturates near 100% by construction with
             this few rows in 64 dims, so it is not evidence on its own
  pred gap   median |few-shot - zero-shot| forecast on query features, in units of the zero-shot
             spread. label-free. large + large shift = adaptation moved the answer using a fit that
             never saw this region""")


def selfcheck():
    """Cheap, no trunk needed: the arms on disk must still carry the frozen pair profile, and the
    support window must really be the observation mask over a calendar prefix."""
    for arm in ARMS:
        b = bundles.load(arm)
        s = b.masks()["support"].astype(bool)
        got = {h: int(s[:, h:].sum()) for h in HORIZONS}
        assert got == EXPECTED_PAIRS[arm], f"{arm}: {got} != {EXPECTED_PAIRS[arm]}"
        cols = np.where(s.any(0))[0]
        prefix = b.M.astype(bool)[:, :cols.max() + 1]
        assert (s[:, :cols.max() + 1] == prefix).all(), \
            f"{arm}: support is NOT the observation mask over its calendar prefix -- the premise of " \
            f"this diagnostic does not hold"
        # Causality binds support COLUMNS against query CELL columns, not against origin indices: an
        # origin is an input-window position, and a query origin's window is allowed to read the
        # support period. This mirrors gate_splits (tests/test_leakage.py:169-176), which compares the
        # dates of columns holding support cells against those holding query cells.
        qcols = np.where(b.masks()["query"].astype(bool).any(0))[0]
        assert cols.max() < qcols.min(), \
            f"{arm}: support reaches col {cols.max()} but a query cell sits at {qcols.min()}"
        print(f"  ok {arm}: pairs {got}, support == M[:, :{cols.max()+1}], "
              f"{int(s.sum())} cells / {int(s.any(1).sum())} districts, "
              f"query cells start col {qcols.min()}")
    print("selfcheck passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    selfcheck()
    rows, payload = run(a.seeds, DEVICE)
    table(rows)
    out = rpath("ebolashift__feature_shift.json", root=RESULTS, make=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

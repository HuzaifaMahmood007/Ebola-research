"""Why the Ebola adapter loses to the borrowed one, and why 20 weeks of support is worse than 12.

Two questions, one piece of linear algebra. With the trunk frozen the Adapter is EXACTLY affine in
trunk features (models/adapters.py:14-16): folding FiLM into the head gives A = W diag(gamma) and
c = W beta + b, reproducing the module to 0.0 (asserted below). So few-shot adaptation is a 64-dim
affine fit, and everything about it is measurable without training anything.

Q1  Why does fitting not help?
    The fit is determined by the SUPPORT features and then applied to the QUERY features. This
    measures how much of what the fit changes is applied outside the region the support ever covered,
    per singular direction of the support design matrix.

Q2  Why is the 20-week arm worse than the 12-week arm, when it has nearly twice the data?
    Its extra support is not more of the same. This counts the support cells per column, which is
    where the six-week reporting blackout shows up, and reports how concentrated the extra data is.

Reads no Ebola label and scores nothing: every quantity is a property of features and weights.
Companion to diagnostics/ebola_feature_shift.py, which measured the shift; this one attributes it.

  conda run -n ebola-train python -m diagnostics.adapter_mechanism --selfcheck
  conda run -n ebola-train python -m diagnostics.adapter_mechanism --seeds 42 52 62 72 82
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import json

import numpy as np
import torch

import bundles
import score
from bundles import HORIZONS
from models import Adapter, SharedEncoder, sparse_from_dense_np
from results_paths import RESULTS, rpath
from train.ebola import _precompute_features, support_origins

ARMS = ("ebola_L12", "ebola_L20")
SEEDS = (42, 52, 62, 72, 82)
PREFIX = "encoder_ebola"
RANK_TOL = 1e-8          # relative to the largest singular value
MIN_SHARE = 0.01         # a direction must carry 1% of the change to enter the median


def affine(ad):
    """(A, c) with pred = h @ A.T + c. FiLM folds into the head exactly; asserted in selfcheck."""
    W, b = ad.head.weight.detach(), ad.head.bias.detach()
    return (W @ torch.diag(ad.gamma.detach())).double().numpy(), (W @ ad.beta.detach() + b).double().numpy()


def design_rows(feats, mask, origins, d=64):
    """{h: [n, d]} -- the feature rows the fit (or the scoring) actually consumes at horizon h."""
    out = {}
    T = mask.shape[1]
    for h in HORIZONS:
        r = [feats[t][np.where(mask[:, t + h])[0]] for t in origins
             if t + h < T and mask[:, t + h].any()]
        out[h] = torch.cat(r).double().cpu().numpy() if r else np.zeros((0, d))
    return out


def column_profile(b):
    """Support cells per column, and how concentrated the tail is. The blackout shows up here."""
    s = b.masks()["support"].astype(bool)
    hi = int(np.where(s.any(0))[0].max())
    per = [int(s[:, c].sum()) for c in range(hi + 1)]
    runs, cur = [], 0
    for n in per:                                   # longest run of wholly unreported columns
        cur = cur + 1 if n == 0 else 0
        runs.append(cur)
    return dict(last_col=hi, per_column=per, total=int(s.sum()),
                longest_blackout=int(max(runs)),
                tail2_share=float(sum(per[-2:]) / max(sum(per), 1)))


def attribute(Sh, Qh, D):
    """Split the change the fit makes, D = A_fewshot - A_zeroshot, across the support's own directions.

    For each singular direction v_i of the support design matrix:
      s_rms  how far the SUPPORT data spread along v_i  (sigma_i / sqrt(n))
      q_rms  how far the QUERY data spreads along v_i
      extrap q_rms / s_rms. Above 1 means the fit is being used beyond where it was determined.
      share  that direction's share of the change D makes to a query prediction, ||D v_i|| * q_rms.

    `extrap_share` is the headline: the fraction of the change that acts along directions where the
    query outruns the support. The MEDIAN extrapolation factor is taken over directions carrying at
    least MIN_SHARE of the change, because a direction with s_rms ~ 1e-4 produces a ratio in the
    hundreds and would otherwise set the summary on its own."""
    if len(Sh) == 0:
        return dict(n_sup=0, rank=0, cond=float("nan"), out_of_span=1.0,
                    extrap_share=1.0, extrap_median=float("nan"))
    U, sv, Vt = np.linalg.svd(Sh, full_matrices=False)
    k = int((sv > sv.max() * RANK_TOL).sum())
    V = Vt[:k]
    par = Qh @ V.T @ V
    out_of_span = float(((Qh - par) ** 2).sum() / max((Qh ** 2).sum(), 1e-30))

    s_rms = sv[:k] / np.sqrt(len(Sh))
    q_rms = np.sqrt(((Qh @ V.T) ** 2).mean(0))
    extrap = q_rms / np.maximum(s_rms, 1e-12)
    w = np.linalg.norm(D @ V.T, axis=0) * q_rms
    w = w / max(w.sum(), 1e-30)
    big = w >= MIN_SHARE
    return dict(n_sup=int(len(Sh)), rank=k, cond=float(sv.max() / sv[k - 1]),
                out_of_span=out_of_span,
                extrap_share=float(w[extrap > 1.0].sum()),
                extrap_median=float(np.median(extrap[big])) if big.any() else float("nan"))


def run(seeds, device):
    nQ = len(score.QUANTILE_LEVELS)
    cells, profiles = [], {}
    for arm in ARMS:
        b = bundles.load(arm)
        profiles[arm] = column_profile(b)
        Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=device)
        Mt = torch.tensor(b.M, dtype=torch.float32, device=device)
        A_adj = sparse_from_dense_np(b.A_geo).to(device)
        sm, qm = b.masks()["support"].astype(bool), b.masks()["query"].astype(bool)
        so, te = support_origins(b), b.origins(phase="query")
        for seed in seeds:
            tk = torch.load(rpath(f"{PREFIX}__alldev__seed{seed}__ckpt.pt", root=RESULTS),
                            map_location=device, weights_only=False)
            fk = torch.load(rpath(f"{PREFIX}__{arm}__seed{seed}__ckpt.pt", root=RESULTS),
                            map_location=device, weights_only=False)
            enc = SharedEncoder(gate_mode="learned").to(device)
            enc.load_state_dict(tk["encoder"]); enc.eval()
            zero = Adapter().to(device); zero.load_state_dict(tk["adapter"])
            fit = Adapter().to(device); fit.load_state_dict(fk["adapter"])
            Af, _ = affine(fit); Az, _ = affine(zero)
            S = design_rows(_precompute_features(enc, Z, A_adj, Mt, so, device), sm, so)
            Q = design_rows(_precompute_features(enc, Z, A_adj, Mt, te, device), qm, te)
            for j, h in enumerate(HORIZONS):
                m = attribute(S[h], Q[h], Af[j * nQ:(j + 1) * nQ] - Az[j * nQ:(j + 1) * nQ])
                m.update(arm=arm, seed=seed, h=h)
                cells.append(m)
    return cells, profiles


def table(cells, profiles):
    print("\nQ2. Where the support cells actually sit, per column")
    for arm in ARMS:
        p = profiles[arm]
        print(f"  {arm}: cols 0..{p['last_col']} -> {p['per_column']}")
        print(f"      {p['total']} cells | longest fully unreported run: {p['longest_blackout']} "
              f"columns | share in the last 2 columns: {p['tail2_share']:.1%}")

    print("\nQ1. What the fitted map changes, and where it is applied")
    print(f"{'arm':<11}{'h':>3}{'n_sup':>7}{'rank':>6}{'cond':>11}{'out of span':>13}"
          f"{'extrapolating':>15}{'median extrap':>15}")
    print("-" * 81)
    for arm in ARMS:
        for h in HORIZONS:
            c = [x for x in cells if x["arm"] == arm and x["h"] == h]
            def g(k):                                   # all-nan cell (h15 on L12) -> nan, no warning
                v = [x[k] for x in c if not np.isnan(x[k])]
                return float(np.mean(v)) if v else float("nan")
            cond = g("cond")
            print(f"{arm:<11}{h:>3}{c[0]['n_sup']:>7}{g('rank'):>6.1f}"
                  f"{(('%.2e' % cond) if np.isfinite(cond) else '-'):>11}"
                  f"{g('out_of_span'):>12.1%}{g('extrap_share'):>15.1%}"
                  f"{(('%.2fx' % g('extrap_median')) if np.isfinite(g('extrap_median')) else '-'):>15}")
    print("""
read:
  n_sup         rows of the affine design matrix at this horizon: the whole of what the fit sees
  rank          of 64. below 64 means some directions are set by the random init, not by data
  cond          condition number of the support design matrix
  out of span   share of QUERY feature energy lying outside everything the support spanned. the fit
                cannot have learned anything about this part, so the random init still governs it
  extrapolating share of the change the fit makes to a query forecast that acts along directions
                where the query data spreads FURTHER than the support data ever did
  median extrap how far beyond its fitted range a contributing direction is pushed, median over the
                directions carrying at least 1% of the change""")


def selfcheck(device):
    # The folding is exact ALGEBRA, so it must be checked in the module's own precision. `affine()`
    # returns float64 because the SVD downstream needs it, and comparing a float64 reconstruction
    # against a float32 forward leaves ~3e-7 of pure rounding, which is not evidence of anything.
    torch.manual_seed(1)
    ad = Adapter(); ad.eval()
    h = torch.randn(9, 64)
    with torch.no_grad():
        ref = ad(h)
        A32 = ad.head.weight @ torch.diag(ad.gamma)
        c32 = ad.head.weight @ ad.beta + ad.head.bias
        got = (h @ A32.T + c32).view_as(ref)
    dev = float((got - ref).abs().max())
    assert dev == 0.0, f"FiLM does not fold into an affine map exactly: max deviation {dev}"
    A64, c64 = affine(ad)
    d64 = float(np.abs((h.double().numpy() @ A64.T + c64).reshape(ref.shape)
                       - ref.double().numpy()).max())
    print(f"  ok adapter is exactly affine once FiLM is folded in (float32 deviation {dev}; "
          f"the float64 path used for the SVD differs by {d64:.1e}, which is float32 rounding)")

    for arm, exp in (("ebola_L12", 59), ("ebola_L20", 113)):
        p = column_profile(bundles.load(arm))
        assert p["total"] == exp, f"{arm}: {p['total']} support cells, expected {exp}"
        print(f"  ok {arm}: {p['total']} cells over cols 0..{p['last_col']}, "
              f"longest blackout {p['longest_blackout']}, tail-2 share {p['tail2_share']:.1%}")

    # a direction the query never uses must not be able to carry any of the change
    S = np.eye(4, 64) * 3.0
    Q = np.zeros((5, 64)); Q[:, :4] = 1.0
    a = attribute(S, Q, np.ones((5, 64)))
    assert abs(a["out_of_span"]) < 1e-12, f"synthetic: query inside span but out_of_span={a['out_of_span']}"
    Q2 = np.zeros((5, 64)); Q2[:, 60:] = 1.0
    a2 = attribute(S, Q2, np.ones((5, 64)))
    assert a2["out_of_span"] > 0.999, f"synthetic: query fully outside span but {a2['out_of_span']}"
    print("  ok attribution: query inside span -> 0% out of span; fully outside -> 100%")
    print("selfcheck passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu")           # 64-dim SVDs; the GPU buys nothing and CPU is reproducible
    if a.selfcheck:
        return selfcheck(dev)
    selfcheck(dev)
    cells, profiles = run(a.seeds, dev)
    table(cells, profiles)
    out = rpath("ebolashift__adapter_mechanism.json", root=RESULTS, make=True)
    out.write_text(json.dumps(dict(cells=cells, profiles=profiles), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

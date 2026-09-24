"""EXPLORATORY. The t6 neighbour-signal probe on Ebola 2014: do a district's neighbours' past
deviations predict its future deviation better than random districts' do, on the data-scarce disease?

This is a DATA probe. No model, no checkpoint, no adapter. It reads the COMPLETE observed Ebola series,
query weeks included, so under experiments/README.md it is exploratory only: it is never the case-study
result, never compared with the pre-registered record without this label, and never used to select
anything that touches the frozen arms. It writes nothing to results/ or data/. The frozen arm hashes
are verified against configs/ebola_arms.json before and after the run.

Design, same as diagnostics/graph_probe/t6_neighbour_signal.py except where Ebola forces a change:
  space      ebola_L12 X[:,:,0] (log1p, pooled per-disease scaler of the primary arm). raw and M are
             identical across ebola / ebola_L12 / ebola_L20 and the scalers are single constants, so
             deviations differ between arms by a constant factor only; L20 is recomputed as a check.
  weeks      the dense period only, weeks 20..51. Weeks 0..12 have 2 to 6 reporting districts, 13..18
             are the six-week reporting blackout, and week 19 is the backlog dump after it (1,399
             cases). Every feature and target cell is in weeks 20..51 (asserted).
  deviation  value minus the mean of OBSERVED districts in the same country that week (3 countries).
             21 of 146 edges cross a border; a cross-border neighbour's deviation is relative to its
             own country.
  split      chronological by TARGET week: fit on targets in weeks 20..39, score on 40..51. Origins
             are horizon-specific, t in [23, 51-h], because a common origin set (t6's) would leave no
             score rows at short horizons on a 52-week panel. Fit features use fit-period cells only.
  model      pooled ridge, alpha 1.0, A = own 4 lags + 4 observed flags, B = A + observed-neighbour
             mean over the same 4 lags (row-normalised, no self-loop).
  null       200 relabels from train.loop.permute_adjacency, plus 200 within-country relabels.
  WIN        gain > 0 and gain > null 95th percentile, as in t6.

Pre-committed readability rule, written before any gain was seen (row counts only were looked at):
  a horizon is READABLE only if it has >= 200 fit rows AND >= 200 score rows AND the Ebola positive
  control (0.5 SD spread planted through the real map, 25% of deviation variance) WINS there.
  At split week 40 the row counts are h3 602/394, h5 517/394, h10 293/394, h15 86/394, so h15 is
  unreadable by construction. Unreadable horizons are reported and never counted.
Also run: the shuffled-base false-alarm check (20 draws), as in t6b.

Run:  conda run -n ebola-train python -m experiments.t6_ebola_neighbour_signal
Writes experiments/t6_ebola_neighbour_signal.json (protocol EXPLORATORY).
"""
import json
import datetime
import os
import sys

import numpy as np

import bundles
from train.loop import permute_adjacency
from train.ebola import load_manifest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "diagnostics", "graph_probe"))
from t6_neighbour_signal import (deviation, nbr_feats, lagstack, ridge_mse,  # noqa: E402
                                 permute_within_groups, K, PERM_BASE, POS_SD, POS_LAG)

ARM, CHECK_ARM = "ebola_L12", "ebola_L20"
HS = (3, 5, 10, 15)
D0, SPLIT = 20, 40
MIN_ROWS = 200
N_PERM, N_WITHIN, WRONG_DRAWS = 200, 200, 20
WRONG_SEED0 = PERM_BASE - 900
OUT = "experiments/t6_ebola_neighbour_signal.json"


def load(arm):
    b = bundles.load(arm)
    x = b.X[:, :, 0].astype(np.float64)
    T = x.shape[1]
    week = np.arange(T)[None, :]
    dense = b.M.astype(bool) & (week >= D0)
    _, gid = np.unique([b.group_of()[n] for n in b.meta["node_ids"]], return_inverse=True)
    return dict(x=x, dense=dense, fitobs=dense & (week < SPLIT), gid=gid, A=b.A_geo, T=T)


def probe(P, n_perm, n_within, level=False):
    """level=True adds each district's mean fit-period deviation as a feature to BOTH A and B, so a
    neighbour term that only encodes 'this is a high-level area' (possible here: Ebola uses one pooled
    scale, so district levels stay inside the deviation) has nothing left to add."""
    x, dense, fitobs, gid, A, T = P["x"], P["dense"], P["fitobs"], P["gid"], P["A"], P["T"]
    d_fit, d_all = deviation(x, fitobs, gid), deviation(x, dense, gid)
    cnt = fitobs.sum(1)
    lvl = np.where(cnt > 0, np.where(fitobs, d_fit, 0.0).sum(1) / np.maximum(cnt, 1), 0.0)
    rows = {}
    for h in HS:
        ts = np.arange(D0 + K - 1, T - h)
        fi, ft = np.nonzero(fitobs[:, ts + h]); ft = ts[ft]
        si, st = np.nonzero(dense[:, ts + h] & (ts + h >= SPLIT)[None, :]); st = ts[st]
        assert (ft + h < SPLIT).all() and (st + h >= SPLIT).all(), f"h{h}: split leak"
        assert (ft - (K - 1) >= D0).all() and (st - (K - 1) >= D0).all(), f"h{h}: feature before week {D0}"
        Af = np.hstack([np.ones((len(fi), 1)), lagstack(d_fit, fi, ft), lagstack(fitobs.astype(float), fi, ft)])
        As = np.hstack([np.ones((len(si), 1)), lagstack(d_all, si, st), lagstack(dense.astype(float), si, st)])
        if level:
            Af, As = np.hstack([Af, lvl[fi, None]]), np.hstack([As, lvl[si, None]])
        rows[h] =(fi, ft, si, st, Af, As, d_fit[fi, ft + h], d_all[si, st + h])

    mseA = {h: ridge_mse(r[4], r[6], r[5], r[7]) for h, r in rows.items()}

    def gains(Amat):
        nbf, _ = nbr_feats(Amat, d_fit, fitobs, False)
        nbs, _ = nbr_feats(Amat, d_all, dense, False)
        out = {}
        for h, (fi, ft, si, st, Af, As, yf, ys) in rows.items():
            mB = ridge_mse(np.hstack([Af, lagstack(nbf, fi, ft)]), yf, np.hstack([As, lagstack(nbs, si, st)]), ys)
            out[h] = 100 * (mseA[h] - mB) / mseA[h]
        return out

    real = gains(A)
    nulls = {}
    for kind, n in (("global", n_perm), ("within_country", n_within)):
        acc = {h: [] for h in HS}
        for p in range(n):
            Ap = permute_adjacency(A, PERM_BASE + p)[0] if kind == "global" else \
                permute_within_groups(A, gid, PERM_BASE + p)[0]
            for h, g in gains(Ap).items():
                acc[h].append(g)
        nulls[kind] = acc
    res = {}
    for h, (fi, ft, si, st, Af, As, yf, ys) in rows.items():
        c = {"n_fit_rows": int(len(fi)), "n_score_rows": int(len(si)), "mse_A": mseA[h],
             "var_target_score": float(np.var(ys)),
             "deviation_share_of_var": float(np.var(ys) / np.var(x[si, st + h])),
             "gain_real_pct": real[h]}
        for kind, acc in nulls.items():
            if not acc[h]:
                continue
            nl = np.array(acc[h]); p95 = float(np.percentile(nl, 95))
            c[kind] = {"n_perm": len(nl), "null_p95_pct": p95, "null_median_pct": float(np.median(nl)),
                       "p_value": float((1 + (nl >= real[h]).sum()) / (1 + len(nl))),
                       "win": bool(real[h] > 0 and real[h] > p95)}
        res[str(h)] = c
    return res


def plant(P, A_inject, shuffle_seed=None):
    """0.5 SD spread planted through A_inject on the dense cells; optional within-country row shuffle."""
    Q = dict(P)
    if shuffle_seed is not None:
        rng = np.random.default_rng([shuffle_seed, 1])
        q = np.arange(len(P["gid"]))
        for g in np.unique(P["gid"]):
            idx = np.nonzero(P["gid"] == g)[0]
            q[idx] = idx[rng.permutation(len(idx))]
        assert (q != np.arange(len(q))).any()
        for k in ("x", "dense", "fitobs"):
            Q[k] = P[k][q]
    d = deviation(Q["x"], Q["dense"], Q["gid"])
    nb, _ = nbr_feats(A_inject, d, Q["dense"], False)
    term = np.zeros_like(nb); term[:, POS_LAG:] = nb[:, :-POS_LAG]
    term = np.where(Q["dense"], term, 0.0)
    fc = Q["fitobs"]
    beta = POS_SD * d[fc].std() / term[fc].std()
    Q["x"] = np.where(Q["dense"], Q["x"] + beta * term, Q["x"])
    return Q, float(beta)


def main():
    load_manifest(verify=True)
    print("ok frozen arm hashes verified (before)")
    P = load(ARM)
    b20 = bundles.load(CHECK_ARM)
    assert np.array_equal(bundles.load(ARM).raw, b20.raw) and np.array_equal(bundles.load(ARM).M, b20.M), \
        "arms differ in raw or M; the one-arm probe does not cover both"

    real = probe(P, N_PERM, N_WITHIN)
    pos_q, beta = plant(P, P["A"])
    pos = probe(pos_q, N_PERM, 0)
    wins_fa = {h: 0 for h in HS}; fa_gains = {h: [] for h in HS}
    for r in range(WRONG_DRAWS):
        s = WRONG_SEED0 - r
        W, _ = permute_adjacency(P["A"], s)
        Q, _ = plant(P, W, shuffle_seed=s)
        rr = probe(Q, N_PERM, 0)
        for h in HS:
            wins_fa[h] += int(rr[str(h)]["global"]["win"]); fa_gains[h].append(rr[str(h)]["gain_real_pct"])
    l20 = probe(load(CHECK_ARM), 0, 0)
    lvl = probe(P, N_PERM, N_WITHIN, level=True)
    pos_lvl = probe(pos_q, N_PERM, 0, level=True)   # power check: can the level-controlled probe still see timing?

    out = {"protocol": "EXPLORATORY", "script": "experiments/t6_ebola_neighbour_signal.py",
           "date": datetime.date.today().isoformat(), "arm": ARM,
           "note": "reads the complete observed series, query weeks included; data probe only, no model; "
                   "never the case-study result; nothing written to results/",
           "design": {"dense_weeks": [D0, P["T"] - 1], "split_target_week": SPLIT, "lags": K,
                      "min_rows_readable": MIN_ROWS, "positive_sd_frac": POS_SD, "positive_beta": beta},
           "horizons": {}}
    print(f"\n{'h':>3s} {'fit':>4s} {'score':>5s} {'share':>5s} | {'gain%':>7s} {'p95':>6s} {'p':>5s} win "
          f"| {'wc p95':>6s} win | {'pos%':>6s} win | FA wins | {'L20 gain':>8s} | readable")
    for h in HS:
        c, pc = real[str(h)], pos[str(h)]
        readable = c["n_fit_rows"] >= MIN_ROWS and c["n_score_rows"] >= MIN_ROWS and pc["global"]["win"]
        c.update(positive={"gain_pct": pc["gain_real_pct"], **pc["global"]},
                 false_alarm={"draws": WRONG_DRAWS, "wins": wins_fa[h],
                              "floor_size_pct": float(np.percentile(fa_gains[h], 95)),
                              "gains_pct": fa_gains[h]},
                 l20_gain_pct=l20[str(h)]["gain_real_pct"], readable=bool(readable),
                 share_of_total_variance_pct=c["gain_real_pct"] * c["mse_A"] * c["deviation_share_of_var"]
                 / c["var_target_score"],
                 level_controlled={**{k: lvl[str(h)][k] for k in ("gain_real_pct", "global", "within_country")},
                                   "positive_gain_pct": pos_lvl[str(h)]["gain_real_pct"],
                                   "positive": pos_lvl[str(h)]["global"]})
        out["horizons"][str(h)] = c
        g, w = c["global"], c["within_country"]
        print(f"{h:3d} {c['n_fit_rows']:4d} {c['n_score_rows']:5d} {c['deviation_share_of_var']:5.2f} | "
              f"{c['gain_real_pct']:+7.3f} {g['null_p95_pct']:+6.3f} {g['p_value']:5.3f} {'WIN' if g['win'] else ' no'} | "
              f"{w['null_p95_pct']:+6.3f} {'WIN' if w['win'] else ' no'} | {pc['gain_real_pct']:+6.2f} "
              f"{'WIN' if pc['global']['win'] else ' no'} | {wins_fa[h]:2d}/{WRONG_DRAWS} | "
              f"{c['l20_gain_pct']:+8.3f} | {'yes' if readable else 'NO'}")
    print("\nlevel-controlled variant (district fit-period mean deviation added to A and B):")
    for h in HS:
        c = out["horizons"][str(h)]; lc = c["level_controlled"]
        print(f"  h{h:<2d} gain {lc['gain_real_pct']:+7.3f}  p95 {lc['global']['null_p95_pct']:+6.3f} "
              f"{'WIN' if lc['global']['win'] else ' no'}  within p95 {lc['within_country']['null_p95_pct']:+6.3f} "
              f"{'WIN' if lc['within_country']['win'] else ' no'} | planted {lc['positive_gain_pct']:+6.2f} "
              f"{'WIN' if lc['positive']['win'] else ' no'} | share of total variance (primary) "
              f"{c['share_of_total_variance_pct']:+.2f}%")
    rd = [h for h in HS if out["horizons"][str(h)]["readable"]]
    out["summary"] = {"readable_horizons": rd,
                      "readable_wins_global": [h for h in rd if out["horizons"][str(h)]["global"]["win"]],
                      "readable_wins_within_country": [h for h in rd if out["horizons"][str(h)]["within_country"]["win"]]}
    out["summary"]["level_controlled_wins_global"] = [
        h for h in rd if out["horizons"][str(h)]["level_controlled"]["global"]["win"]]
    out["summary"]["level_controlled_power"] = [
        h for h in rd if out["horizons"][str(h)]["level_controlled"]["positive"]["win"]]
    load_manifest(verify=True)                     # raises SystemExit before anything is written
    out["frozen_arm_hashes_verified"] = ["before", "after"]
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nreadable horizons {rd}; wins (global null) {out['summary']['readable_wins_global']}; "
          f"wins (within-country null) {out['summary']['readable_wins_within_country']}")
    print(f"ok frozen arm hashes verified (after). wrote {OUT}  (protocol=EXPLORATORY)")


if __name__ == "__main__":
    main()

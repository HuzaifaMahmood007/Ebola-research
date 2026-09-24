"""TEST 6: is there spatial-spread signal in the RAW DATA at all? No model, no checkpoints.

Question. After removing the shared weekly wave, does a district's neighbours' PAST deviation predict
that district's FUTURE deviation better than a random relabelled set of districts' past deviation?
This decides whether the graph failure (gate-off, t2 relabel, D2 shuffled retrain) is a property of
the data or of the model.

Space. The same model space the encoder sees: bundle X[:,:,0] = log1p then per-node z-score with the
released TRAIN-period scaler (asserted against a train-only refit below). Observed cells only (obs mask).

Deviation. d[i,t] = x[i,t] - mean over OBSERVED districts in i's group at t. Group = country on dengue
(12 countries, no cross-country edges), the whole panel on the four single-country panels.

Features at origin t, k = 0..3 lags:
  A  own-only       1, d[i,t-k], obs[i,t-k]                          (9 columns)
  B  own + nbr      A + nbrmean[i,t-k]                               (13 columns)
  C  own + nbr + sp B + nbrspread[i,t-k] (max minus min)             (17 columns, secondary)
nbrmean is ROW-normalised over OBSERVED neighbours, no self-loop (a plain average in d units; own
history is already in A). 0 when no neighbour is observed. nbrspread is 0 with fewer than 2 observed.

Split rule. Origins are bundles.origins() (t >= 19, t+15 <= T-1). A row (i,t,h) is a FIT row when the
target cell (i,t+h) is in the bundle's train mask, a SCORE row when it is in the test mask. Val rows are
never used. For fit rows every feature is built from TRAIN-phase cells only (phase-aware obs mask), so
no val/test cell enters fitting under the real map or any relabelled map. Score rows read every
observed past cell, as the model does at forecast time. Nothing is re-scaled.

Model. Pooled ridge over all districts, closed form, alpha = 1.0 on the unscaled features, intercept
unpenalised. gain = 100 * (MSE_A - MSE_B) / MSE_A on score rows.

Null. B rebuilt from train.loop.permute_adjacency (degree-preserving relabel, identity refused).
Real neighbours WIN a cell if gain_real > 0 AND gain_real > the 95th percentile of the null gains.
200 relabels on the four small panels, 50 on dengue (about 7 s each). Dengue also gets a STRICTER
secondary null, 30 relabels restricted to within-country (real dengue edges never cross a border, a
global relabel does), reported alongside; the pre-committed win rule uses the global null.

Controls (run FIRST; a positive-control failure stops the script before any real panel is scored):
  positive   influenza_japan copy, x' = x + beta * nbrmean_real(d)[t-3], beta set so the injected term
             has SD = 0.5 x SD(d) on train cells (25% of the deviation variance). Must WIN at h3 and h5.
             Also run at 0.25 x SD (6.25% of variance) for sensitivity, no pass requirement.
  wrong-map  FIXED 2026-09-24. 20 draws: shuffle district identities in the base (so the real map is
             meaningless), plant the same 0.5 SD signal through a relabelled map, probe with the real
             map. Real-map WINs must be <= 3 of 20 at h3 and at h5 (P(>=4 | chance) = 0.016).
             The original version planted on the REAL Japan base and only checked magnitude (gain at h3
             under half the strong arm's). It passed while flagging WIN at every horizon, because the
             real base already carries real-map signal. Recorded as wrong_map_realbase_legacy.
  negative   median null gain per cell; flagged if |median| > 1% of MSE_A.

Run:  conda run -n ebola-train python diagnostics/graph_probe/t6_neighbour_signal.py
      (add --quick for 20 permutations everywhere, a smoke run; never cite its output)
Writes results/misc/t6_neighbour_signal.json. Reads only data/processed bundles.
"""
import os, sys, json, time, datetime, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.chdir(r"F:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import numpy as np
import scipy.sparse as sp
import bundles
from to_schema import fit_scalers_masked
from train.loop import permute_adjacency

PANELS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states", "dengue"]
HORIZONS = bundles.HORIZONS
K = 4
ALPHA = 1.0
PERM_BASE = 20260924
N_PERM = {"dengue": 50}           # ~7 s per dengue permutation; default below for everything else
N_PERM_DEFAULT = 200
N_WITHIN = {"dengue": 30}         # within-country relabel null, only where there is more than one group
POS_SD = 0.5                      # injected SD as a fraction of SD(d), pass-required
POS_SD_WEAK = 0.25                # sensitivity only
POS_LAG = 3
NEG_FLAG_PCT = 1.0
OUT = "results/misc/t6_neighbour_signal.json"


# --------------------------------------------------------------------------- data side
def load_panel(name):
    b = bundles.load(name)
    x = b.X[:, :, 0].astype(np.float64)
    M = b.M.astype(bool)
    ms = b.masks()
    train, test = ms["train"].astype(bool), ms["test"].astype(bool)
    groups = [b.group_of()[n] for n in b.meta["node_ids"]]
    _, gid = np.unique(groups, return_inverse=True)
    # the space the encoder sees must be the TRAIN-fitted scaler
    # float64: on float32 raw a constant train window gets sd ~6e-8, dodging the 1e-8 degenerate test
    s = fit_scalers_masked(b.raw.astype(np.float64), ms["train"],
                           groups=groups if b.meta.get("node_country") else None)
    diff = np.maximum(np.abs(s["mean"] - b.scaler["mean"]), np.abs(s["std"] - b.scaler["std"]))
    n_exact = int((diff < 1e-4).sum())
    assert n_exact == len(diff), (f"{name}: shipped scaler differs from a train-only refit on "
                                  f"{len(diff) - n_exact} nodes (max {diff.max()}); space is not train-fitted")
    # cross-group edges would let one country's wave leak in as 'spatial'
    ii, jj = np.nonzero(b.A_geo)
    xg = int((gid[ii] != gid[jj]).sum())
    return dict(b=b, x=x, M=M, train=train, test=test, gid=gid, A=b.A_geo,
                scaler_exact_nodes=n_exact, scaler_max_diff=float(diff.max()),
                n_groups=int(gid.max() + 1), cross_group_edges=xg)


def deviation(x, obs, gid):
    d = np.zeros_like(x)
    for g in np.unique(gid):
        s = gid == g
        o = obs[s]
        cnt = o.sum(0)
        mu = np.where(o, x[s], 0.0).sum(0) / np.maximum(cnt, 1)
        d[s] = np.where(o, x[s] - mu, 0.0)
    return d


def nbr_feats(A_np, d, obs, spread, chunk=256):
    """Row-normalised mean over observed neighbours (no self-loop), and max-min spread."""
    W = sp.csr_matrix(A_np.astype(np.float64))
    o = obs.astype(np.float64)
    den = W @ o
    nb = np.where(den > 0, (W @ (d * o)) / np.maximum(den, 1e-12), 0.0)
    if not spread:
        return nb, None
    Wb = sp.csr_matrix((A_np != 0).astype(np.float64))
    cnt = Wb @ o
    N, T = d.shape
    deg = np.diff(Wb.indptr)
    rows = np.nonzero(deg > 0)[0]
    starts = Wb.indptr[:-1][rows]
    cols = Wb.indices
    spr = np.zeros((N, T))
    for c0 in range(0, T, chunk):
        c1 = min(T, c0 + chunk)
        dv, ov = d[cols, c0:c1], obs[cols, c0:c1]
        mx = np.maximum.reduceat(np.where(ov, dv, -np.inf), starts, axis=0)
        mn = np.minimum.reduceat(np.where(ov, dv, np.inf), starts, axis=0)
        v = np.where(cnt[rows, c0:c1] >= 2, mx - mn, 0.0)
        spr[rows, c0:c1] = v
    return nb, spr


def rows_for(P, phase, h, origins):
    """(i, t) index arrays of rows whose TARGET (i, t+h) is in `phase`."""
    mask = P[phase]
    tt = np.array(origins)
    sub = mask[:, tt + h]                             # [N, n_origins]
    i, k = np.nonzero(sub)
    return i, tt[k]


def lagstack(F, i, t):
    return np.stack([F[i, t - k] for k in range(K)], 1)


def ridge_mse(Xf, yf, Xs, ys):
    P = np.eye(Xf.shape[1]) * ALPHA
    P[0, 0] = 0.0
    w = np.linalg.solve(Xf.T @ Xf + P, Xf.T @ yf)
    r = Xs @ w - ys
    return float(np.mean(r * r))


# --------------------------------------------------------------------------- probe
def probe(P, n_perm, spread=True, tag="", n_within=0, level=False):
    """P holds x, M, train, test, gid, A. Returns per-horizon results dict.

    level=True (added 2026-09-24, default off so the published run is unchanged) appends each
    district's mean TRAIN-phase deviation as one more column to A and B alike. It asks whether the
    neighbour gain is just 'big districts sit next to big districts' (see t6c_level_check.py)."""
    x, M, train, test, gid, A = P["x"], P["M"], P["train"], P["test"], P["gid"], P["A"]
    T = x.shape[1]
    origins = P["b"].origins()
    assert min(origins) >= K - 1 and max(origins) + max(HORIZONS) <= T - 1, "origin range broken"
    d_fit, d_all = deviation(x, train, gid), deviation(x, M, gid)
    cnt = train.sum(1)
    lvl = np.where(cnt > 0, np.where(train, d_fit, 0.0).sum(1) / np.maximum(cnt, 1), 0.0)

    rows, own = {}, {}
    for h in HORIZONS:
        fi, ft = rows_for(P, "train", h, origins)
        si, st = rows_for(P, "test", h, origins)
        assert len(fi) and len(si), f"{tag} h{h}: empty fit or score set"
        # leakage: fit targets train-only, score targets test-only, disjoint cells, no overlap
        assert train[fi, ft + h].all() and not test[fi, ft + h].any(), f"{tag} h{h}: fit row off-phase"
        assert test[si, st + h].all() and not train[si, st + h].any(), f"{tag} h{h}: score row off-phase"
        assert not np.intersect1d(fi * T + ft + h, si * T + st + h).size, \
            f"{tag} h{h}: a cell is both fit and score"
        # every own-lag cell a fit row reads is train-phase or unobserved (time ordering holds)
        for k in range(K):
            assert not (M[fi, ft - k] & ~train[fi, ft - k]).any(), f"{tag} h{h}: fit reads non-train own cell"
        ones_f, ones_s = np.ones((len(fi), 1)), np.ones((len(si), 1))
        Af = np.hstack([ones_f, lagstack(d_fit, fi, ft), lagstack(train.astype(float), fi, ft)])
        As = np.hstack([ones_s, lagstack(d_all, si, st), lagstack(M.astype(float), si, st)])
        if level:
            Af, As = np.hstack([Af, lvl[fi, None]]), np.hstack([As, lvl[si, None]])
        yf, ys = d_fit[fi, ft + h], d_all[si, st + h]
        rows[h] = (fi, ft, si, st, yf, ys)
        own[h] = (Af, As)

    def arms(Amat):
        nbf, spf = nbr_feats(Amat, d_fit, train, spread)
        nbs, sps = nbr_feats(Amat, d_all, M, spread)
        out = {}
        for h in HORIZONS:
            fi, ft, si, st, yf, ys = rows[h]
            Af, As = own[h]
            Bf = np.hstack([Af, lagstack(nbf, fi, ft)]); Bs = np.hstack([As, lagstack(nbs, si, st)])
            r = {"B": ridge_mse(Bf, yf, Bs, ys)}
            if spread:
                r["C"] = ridge_mse(np.hstack([Bf, lagstack(spf, fi, ft)]), yf,
                                   np.hstack([Bs, lagstack(sps, si, st)]), ys)
            out[h] = r
        return out, (nbf, nbs)

    mseA = {h: ridge_mse(own[h][0], rows[h][4], own[h][1], rows[h][5]) for h in HORIZONS}
    real, (nbf_r, nbs_r) = arms(A)
    # with the REAL map, phase-aware and all-observed neighbour terms agree on every fit row
    if P["cross_group_edges"] == 0:
        for h in HORIZONS:
            fi, ft = rows[h][0], rows[h][1]
            assert np.allclose(nbf_r[fi, ft], nbs_r[fi, ft]), f"{tag} h{h}: real-map fit features leak"

    def run_null(relabel, n):
        null = {h: {"B": [], "C": []} for h in HORIZONS}
        frac, same = [], 0
        for p in range(n):
            Ap, fr = relabel(PERM_BASE + p)
            frac.append(fr)
            if np.array_equal(Ap, A):
                same += 1
            res, _ = arms(Ap)
            for h in HORIZONS:
                for a in res[h]:
                    null[h][a].append(100 * (mseA[h] - res[h][a]) / mseA[h])
        assert same <= 0.05 * n, f"{tag}: {same}/{n} relabels reproduce the real map, null degenerate"
        if n == 0:                    # gains only, no null (floor script's real-base arm)
            return null, {"n_perm": 0}
        return null, {"n_perm": n, "perm_seed_base": PERM_BASE, "null_automorphic_draws": same,
                      "frac_displaced_min": float(min(frac)), "frac_displaced_mean": float(np.mean(frac))}

    def cells(null):
        o = {}
        for h in HORIZONS:
            o[str(h)] = {}
            for a in (["B", "C"] if spread else ["B"]):
                g = 100 * (mseA[h] - real[h][a]) / mseA[h]
                nl = np.array(null[h][a])
                if not len(nl):
                    o[str(h)][a] = {"mse_real": real[h][a], "gain_real_pct": float(g), "win": None}
                    continue
                p95 = float(np.percentile(nl, 95))
                o[str(h)][a] = {"mse_real": real[h][a], "gain_real_pct": float(g),
                                "null_p95_pct": p95, "null_median_pct": float(np.median(nl)),
                                "null_mean_pct": float(nl.mean()), "null_sd_pct": float(nl.std(ddof=1)),
                                "p_value": float((1 + (nl >= g).sum()) / (1 + len(nl))),
                                "win": bool(g > 0 and g > p95)}
        return o

    def global_relabel(seed):
        Ap, meta = permute_adjacency(A, seed)
        return Ap, meta["shuffle_adj_frac_displaced"]

    null, nmeta = run_null(global_relabel, n_perm)
    out = dict(nmeta, horizons={})
    cg = cells(null)
    for h in HORIZONS:
        fi, _, si, st, _, ys = rows[h]
        xs = x[si, st + h]
        out["horizons"][str(h)] = {"n_fit_rows": int(len(fi)), "n_score_rows": int(len(si)),
                                   "var_target_score": float(np.var(ys)),
                                   "deviation_share_of_var": float(np.var(ys) / np.var(xs)),
                                   "mse_A": mseA[h], **cg[str(h)]}
    if n_within:                      # stricter null: relabel only inside each group (country)
        def within(seed):
            return permute_within_groups(A, gid, seed)
        wnull, wmeta = run_null(within, n_within)
        out["within_group_null"] = dict(wmeta, horizons=cells(wnull))
    return out


def permute_within_groups(A_np, gid, seed):
    """permute_adjacency, restricted so every node is relabelled onto a node of its OWN group.
    Same guarantees, asserted the same way: not the identity, edges and degree multisets kept."""
    N = A_np.shape[0]
    rng = np.random.default_rng(seed)
    p = np.arange(N)
    for g in np.unique(gid):
        idx = np.nonzero(gid == g)[0]
        p[idx] = idx[rng.permutation(len(idx))]
    frac = float((p != np.arange(N)).mean())
    assert frac > 0.0, f"within-group permutation seed {seed} is the identity"
    assert (gid[p] == gid).all(), "within-group relabel moved a node across groups"
    Ap = A_np[np.ix_(p, p)]
    assert int((Ap != 0).sum()) == int((A_np != 0).sum()), "edge count changed"
    assert np.array_equal(np.sort((Ap != 0).sum(1)), np.sort((A_np != 0).sum(1))), "degrees changed"
    return Ap, frac


def inject(P, A_inject, sd_frac):
    """Synthetic spread on a copy: x' = x + beta * nbrmean_{A_inject}(d)[t - POS_LAG]."""
    d = deviation(P["x"], P["M"], P["gid"])
    nb, _ = nbr_feats(A_inject, d, P["M"], False)
    term = np.zeros_like(nb)
    term[:, POS_LAG:] = nb[:, :-POS_LAG]
    tr = P["train"]
    beta = sd_frac * d[tr].std() / term[tr].std()
    Q = dict(P)
    Q["x"] = np.where(P["M"], P["x"] + beta * term, P["x"])
    return Q, float(beta)


def shuffled_base_wrong_map(P, seed, n_perm, sd_frac=POS_SD, shuffle=True):
    """False-alarm test for the probe. Shuffle district identities in the BASE data (whole rows of x,
    M, train, test move together, inside each group), so the real map is meaningless for it; then
    plant spread through a relabelled map W; then probe with the REAL map. A correct probe wins here
    at about the nominal 5% rate. shuffle=False is the 2026-09-24 legacy arm (plant on the real base),
    which is NOT a false-alarm test: it inherits whatever real-map signal the base already carries."""
    N = P["x"].shape[0]
    base = dict(P)
    if shuffle:
        rng = np.random.default_rng([seed, 1])       # independent of W's stream below
        q = np.arange(N)
        for g in np.unique(P["gid"]):
            idx = np.nonzero(P["gid"] == g)[0]
            q[idx] = idx[rng.permutation(len(idx))]
        assert (q != np.arange(N)).any(), f"base shuffle seed {seed} is the identity"
        assert (P["gid"][q] == P["gid"]).all(), "base shuffle moved a district across groups"
        for k in ("x", "M", "train", "test"):
            base[k] = P[k][q]
    W, _ = permute_adjacency(P["A"], seed)
    planted, beta = inject(base, W, sd_frac)
    return probe(planted, n_perm, spread=False, tag=f"wrongmap-{seed}"), beta


WRONG_DRAWS = 20                  # fixed control: draws of (base shuffle, planting map)
WRONG_SEED0 = PERM_BASE - 100     # draw r uses seed WRONG_SEED0 - r; disjoint from the null seeds
WRONG_MAX_WINS = 3                # P(X >= 4 | n=20, p=0.05) = 0.016


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 permutations everywhere; smoke only")
    ap.add_argument("--controls-only", action="store_true",
                    help="run the three controls, write results/misc/t6_neighbour_signal__controls.json, stop")
    ap.add_argument("--break-wrongmap", action="store_true",
                    help="self-test: plant the wrong-map control on the REAL base; the assert MUST fire")
    a = ap.parse_args()
    nperm = (lambda n: 20) if a.quick else (lambda n: N_PERM.get(n, N_PERM_DEFAULT))
    T0 = time.time()
    out = {"script": "diagnostics/graph_probe/t6_neighbour_signal.py",
           "date": datetime.date.today().isoformat(), "quick": a.quick,
           "design": {"space": "bundle X[:,:,0]: log1p + per-node z-score, released train scaler",
                      "deviation": "x minus mean over observed districts in group at t; group = "
                                   "country on dengue, whole panel otherwise",
                      "lags": K, "horizons": list(HORIZONS), "ridge_alpha": ALPHA,
                      "neighbour_term": "row-normalised mean over observed neighbours, no self-loop",
                      "win_rule": "gain_real > 0 and gain_real > null 95th percentile",
                      "fit_rows": "target cell in train mask; features from train-phase cells only",
                      "score_rows": "target cell in test mask; features from all observed past cells"},
           "controls": {}, "panels": {}}

    # ------------------------------------------------ controls first
    P = load_panel("influenza_japan")
    n = nperm("influenza_japan")
    pos, beta = inject(P, P["A"], POS_SD)
    print(f"[positive] japan, injected SD = {POS_SD} x SD(d), beta={beta:.3f}")
    r_pos = probe(pos, n, spread=False, tag="positive")
    weak, beta_w = inject(P, P["A"], POS_SD_WEAK)
    r_weak = probe(weak, n, spread=False, tag="positive-weak")
    A_wrong, meta_w = permute_adjacency(P["A"], PERM_BASE - 1)
    wrong, beta_x = inject(P, A_wrong, POS_SD)
    r_wrong = probe(wrong, n, spread=False, tag="wrong-map")
    for lab, r in (("strong", r_pos), ("weak", r_weak), ("wrong-map", r_wrong)):
        print("  " + lab.ljust(9) + "  ".join(
            f"h{h}: {r['horizons'][str(h)]['B']['gain_real_pct']:+6.2f}% "
            f"(p95 {r['horizons'][str(h)]['B']['null_p95_pct']:+5.2f}, "
            f"{'WIN' if r['horizons'][str(h)]['B']['win'] else 'no'})" for h in HORIZONS))
    pos_pass = all(r_pos["horizons"][str(h)]["B"]["win"] for h in (3, 5))
    # FIXED 2026-09-24. The original check was magnitude-only ("wrong-map gain at h3 < half the strong
    # arm's"), never read the WIN flag, and ran on the REAL Japan base, which already carries real-map
    # signal. It passed while every horizon was flagged WIN. Kept below as a legacy record, no assert.
    # The replacement plants through a wrong map on a base where the real map is meaningless and
    # asserts the WIN rate stays at chance.
    wins_wm = {h: 0 for h in HORIZONS}
    draws = []
    for r in range(WRONG_DRAWS):
        rw, bw = shuffled_base_wrong_map(P, WRONG_SEED0 - r, n, shuffle=not a.break_wrongmap)
        draws.append({h: rw["horizons"][str(h)]["B"]["gain_real_pct"] for h in HORIZONS})
        for h in HORIZONS:
            wins_wm[h] += int(rw["horizons"][str(h)]["B"]["win"])
    print(f"  shuffled-base wrong-map: wins over {WRONG_DRAWS} draws " +
          "  ".join(f"h{h}: {wins_wm[h]}" for h in HORIZONS))
    wrong_pass = all(wins_wm[h] <= WRONG_MAX_WINS for h in (3, 5))
    out["controls"] = {
        "positive": {"panel": "influenza_japan", "sd_frac": POS_SD, "variance_share": POS_SD ** 2,
                     "lag": POS_LAG, "beta": beta, "pass_rule": "WIN at h3 and h5",
                     "passed": pos_pass, **r_pos},
        "positive_weak": {"sd_frac": POS_SD_WEAK, "variance_share": POS_SD_WEAK ** 2,
                          "beta": beta_w, **r_weak},
        "wrong_map_shuffled_base": {
            "draws": WRONG_DRAWS, "seed0": WRONG_SEED0, "sd_frac": POS_SD,
            "pass_rule": f"real-map WINs <= {WRONG_MAX_WINS} of {WRONG_DRAWS} at h3 and at h5",
            "wins": {str(h): wins_wm[h] for h in HORIZONS},
            "gains_pct": {str(h): [d[h] for d in draws] for h in HORIZONS}, "passed": wrong_pass},
        "wrong_map_realbase_legacy": {
            "inject_perm_seed": PERM_BASE - 1, "beta": beta_x,
            "note": "magnitude-only legacy check on the real base; not a false-alarm test, no assert",
            "legacy_rule": "real-map gain at h3 < 0.5 x positive-control gain at h3",
            "legacy_passed": bool(r_wrong["horizons"]["3"]["B"]["gain_real_pct"]
                                  < 0.5 * r_pos["horizons"]["3"]["B"]["gain_real_pct"]),
            **r_wrong}}
    assert pos_pass, "POSITIVE CONTROL FAILED: the probe cannot detect injected spread. Stop, do not interpret."
    if a.break_wrongmap:              # persist the self-test before the assert decides, so it is checkable
        with open("results/misc/t6_neighbour_signal__breaktest.json", "w") as f:
            json.dump({"script": out["script"], "date": out["date"], "mode": "wrong-map plant on REAL base",
                       "wins": {str(h): wins_wm[h] for h in HORIZONS}, "draws": WRONG_DRAWS,
                       "assert_would_pass": wrong_pass}, f, indent=2)
    if a.break_wrongmap and wrong_pass:
        sys.exit(f"SELF-TEST FAILED: the real-base plant did not trip the wrong-map assert ({wins_wm}); "
                 f"the check cannot catch what it exists for")
    assert wrong_pass, (f"WRONG-MAP CONTROL FAILED: a signal planted through a wrong map on a shuffled base "
                        f"wins with the real map above chance ({wins_wm}). The null is miscalibrated. Stop.")
    if a.controls_only:
        path = "results/misc/t6_neighbour_signal__controls.json"
        with open(path, "w") as f:
            json.dump({"script": out["script"], "date": out["date"], "controls": out["controls"]}, f, indent=2)
        print(f"controls passed; wrote {path}")
        return

    # ------------------------------------------------ the five dev panels
    for name in PANELS:
        t0 = time.time()
        P = load_panel(name)
        nw = (20 if a.quick else N_WITHIN.get(name, 0)) if P["n_groups"] > 1 else 0
        r = probe(P, nperm(name), spread=True, tag=name, n_within=nw)
        r.update(N=int(P["x"].shape[0]), T=int(P["x"].shape[1]), n_groups=P["n_groups"],
                 deviation_group="country" if P["n_groups"] > 1 else "whole panel",
                 cross_group_edges=P["cross_group_edges"],
                 scaler_exact_nodes=P["scaler_exact_nodes"], scaler_max_diff=P["scaler_max_diff"],
                 seconds=round(time.time() - t0, 1))
        out["panels"][name] = r
        for h in HORIZONS:
            c = r["horizons"][str(h)]
            print(f"{name:22s} h{h:<2d} B {c['B']['gain_real_pct']:+7.3f}% p95 {c['B']['null_p95_pct']:+7.3f} "
                  f"med {c['B']['null_median_pct']:+7.3f} p={c['B']['p_value']:.3f} "
                  f"{'WIN' if c['B']['win'] else '   '} | C {c['C']['gain_real_pct']:+7.3f}% "
                  f"p95 {c['C']['null_p95_pct']:+7.3f} {'WIN' if c['C']['win'] else ''}")
        if "within_group_null" in r:
            w = r["within_group_null"]
            print("  within-country null (" + str(w["n_perm"]) + " perms): " + "  ".join(
                f"h{h} p95 {w['horizons'][str(h)]['B']['null_p95_pct']:+.3f} "
                f"{'WIN' if w['horizons'][str(h)]['B']['win'] else 'no'}" for h in HORIZONS))
        print(f"  ({r['seconds']}s, {r['n_perm']} perms)")
    assert set(out["panels"]) == set(PANELS), f"missing panels: {set(PANELS) - set(out['panels'])}"

    # ------------------------------------------------ negative control + verdict
    neg = {f"{p}|h{h}": out["panels"][p]["horizons"][str(h)]["B"]["null_median_pct"]
           for p in PANELS for h in HORIZONS}
    flagged = [k for k, v in neg.items() if abs(v) > NEG_FLAG_PCT]
    out["controls"]["negative"] = {"rule": f"|median null gain| <= {NEG_FLAG_PCT}% in every cell",
                                   "max_abs_median_pct": max(abs(v) for v in neg.values()),
                                   "flagged_cells": flagged, "passed": not flagged}
    wins = {p: sum(out["panels"][p]["horizons"][str(h)]["B"]["win"] for h in HORIZONS) for p in PANELS}
    total = sum(wins.values())
    multi = [p for p in PANELS if wins[p] >= 2]
    if len(multi) >= 2:
        bucket = "BROAD SIGNAL"
    elif len(multi) == 1:
        bucket = "LOCALISED SIGNAL"
    elif total <= 1:
        bucket = "NO SIGNAL"
    else:
        bucket = "UNCLASSIFIED (scattered single wins)"
    # POST HOC, labelled as such: the same count with a minimum effect size on the winning gain
    floors = {}
    for fl in (1.0, 2.0):
        wf = {p: sum(out["panels"][p]["horizons"][str(h)]["B"]["win"]
                     and out["panels"][p]["horizons"][str(h)]["B"]["gain_real_pct"] >= fl
                     for h in HORIZONS) for p in PANELS}
        floors[f"{fl:.0f}pct"] = {"wins_per_panel": wf, "total": sum(wf.values()),
                                  "panels_with_2plus": [p for p in PANELS if wf[p] >= 2]}
    out["verdict"] = {"wins_per_panel": wins, "total_wins": total, "of_cells": 20,
                      "panels_with_2plus": multi, "bucket": bucket,
                      "post_hoc_size_floor": floors,
                      "rule": "NO SIGNAL: total <= 1 and no panel >= 2; LOCALISED: exactly one panel "
                              ">= 2; BROAD: two or more panels >= 2 each"}
    out["seconds_total"] = round(time.time() - T0, 1)
    path = OUT.replace(".json", "__QUICK_smoke.json") if a.quick else OUT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwins per panel {wins}, total {total}/20 -> {bucket}")
    print(f"negative control: max |median null| {out['controls']['negative']['max_abs_median_pct']:.3f}%"
          f", flagged {flagged}")
    print(f"wrote {path} ({out['seconds_total']}s)")


if __name__ == "__main__":
    main()

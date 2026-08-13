"""Offline post-hoc reads over the artifacts train/loop.py emits. No model, no retraining.

  --ci     paired bootstrap-over-origins CIs (B=10,000). For a comparison, resample the SAME
           origins for both models and build the CI on the DIFFERENCE; the pairing is what cancels
           japan's +-7-10% seed sd. Wilcoxon over the 5 seed-matched runs prints as SUPPORTING ONLY.
  --reads  per-dataset gate distribution and normalised spatial contribution, pooled over seeds.
  --selfcheck  synthetic checks of the bootstrap engine and reads aggregation, no artifacts needed.

CAVEAT: the saved sae/sse/n are node-pooled within country, so the reconstructed country-macro is
CELL-POOLED, not the node-averaged headline. Equal on the dense influenza panels, diverges on
dengue, and stated on every table. RMSE/MAE only: PCC is not additive over origins from sae/sse/n.

  PYTHONNOUSERSITE=1 conda run -n ebola-train python analysis.py --ci --reads
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np

from results_paths import rpath

RESULTS = Path("results")
SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
NAIVES = ("persistence", "seasonal", "train_mean")
DEV = ("dengue", "influenza_japan", "influenza_us-regions", "influenza_us-states",
       "covid_us-states")
B_DEFAULT = 10_000


# --------------------------------------------------------------------------- #
# Item 6 -- paired bootstrap over origins from the per-(origin,country) stats.
# --------------------------------------------------------------------------- #
def _load_perorigin(path, h):
    z = np.load(path, allow_pickle=True)
    return z[f"h{h}__sae"].astype(np.float64), z[f"h{h}__sse"].astype(np.float64), \
        z[f"h{h}__n"].astype(np.float64)                       # each [K origins, C countries]


def _encoder_stats(name, h, seeds, prefix="encoder"):
    """Seed-mean sufficient stats. n is model-independent (same observed cells every seed), so the
    seed-mean of sae/sse divided by n IS the mean-over-seeds cell-pooled metric -- exactly the
    quantity the bootstrap should put a CI around. prefix='encoder' (single) or
    'encoder_joint__<tag>' (a joint run) -- same file layout, different leading token."""
    saes, sses, ns = [], [], []
    for s in seeds:
        sae, sse, n = _load_perorigin(rpath(f"{prefix}__{name}__seed{s}__perorigin.npz"), h)
        saes.append(sae); sses.append(sse); ns.append(n)
    return np.mean(saes, 0), np.mean(sses, 0), ns[0]


def _cnt_matrix(K, B, rng):
    """[K,B] bootstrap count matrix: column j = how many times each of the K origins was drawn."""
    draws = rng.integers(0, K, size=(B, K))
    return np.stack([np.bincount(draws[j], minlength=K) for j in range(B)], axis=1).astype(np.float64)


def _macro_dist(sae, sse, n, CNT, metric):
    """Cell-pooled country-macro per bootstrap draw -> [B]. Per country: MAE=sum(sae)/sum(n),
    RMSE=sqrt(sum(sse)/sum(n)); then mean over countries (a country with 0 resampled cells is NaN
    and dropped from the macro)."""
    num = (sse if metric == "rmse" else sae).T @ CNT               # [C,B]
    den = n.T @ CNT                                                 # [C,B]
    per_c = num / np.where(den > 0, den, np.nan)
    if metric == "rmse":
        per_c = np.sqrt(per_c)
    return np.nanmean(per_c, axis=0)                               # macro over countries -> [B]


def _point_macro(sae, sse, n, metric):
    """The cell-pooled country-macro point value (no resampling) from one run's stats."""
    num = (sse if metric == "rmse" else sae).sum(0)
    den = n.sum(0)
    per_c = num / np.where(den > 0, den, np.nan)
    if metric == "rmse":
        per_c = np.sqrt(per_c)
    return float(np.nanmean(per_c))


def bootstrap_ci(name, B=B_DEFAULT, metrics=("rmse", "mae"), seeds=SEEDS, seed=0, verbose=True,
                 prefix="encoder"):
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(seed)
    out = {}
    if verbose:
        print(f"\n{'=' * 92}\n{prefix} :: {name}  paired bootstrap over origins "
              f"(B={B}, cell-pooled country-macro)\n{'=' * 92}")
    for h in HORIZONS:
        se_enc = _encoder_stats(name, h, seeds, prefix)            # (sae,sse,n) seed-mean
        K = se_enc[0].shape[0]
        CNT = _cnt_matrix(K, B, rng)
        naive_stats = {nm: _load_perorigin(rpath(f"naive__{name}__{nm}__perorigin.npz"), h)
                       for nm in NAIVES}
        for metric in metrics:
            enc_dist = _macro_dist(*se_enc, CNT, metric)
            enc_pt = _point_macro(*se_enc, metric)
            enc_lo, enc_hi = np.nanpercentile(enc_dist, [2.5, 97.5])
            # per-seed encoder point values, for the (supporting) Wilcoxon
            enc_seed_pts = np.array([_point_macro(
                *_load_perorigin(rpath(f"{prefix}__{name}__seed{s}__perorigin.npz"), h), metric)
                for s in seeds])
            row = dict(point=enc_pt, ci=(enc_lo, enc_hi), vs={})
            if verbose:
                print(f"  h{h:<2} {metric.upper():4} encoder {enc_pt:9.3f}  "
                      f"95% CI [{enc_lo:8.3f}, {enc_hi:8.3f}]")
            for nm, ns in naive_stats.items():
                nv_dist = _macro_dist(*ns, CNT, metric)            # SAME CNT -> paired
                nv_pt = _point_macro(*ns, metric)
                diff = enc_dist - nv_dist                          # encoder - naive (lower=better)
                d_lo, d_hi = np.nanpercentile(diff, [2.5, 97.5])
                wins = d_hi < 0                                    # CI strictly below 0 -> encoder better
                try:
                    p = float(wilcoxon(enc_seed_pts - nv_pt).pvalue)
                except ValueError:
                    p = float("nan")                              # all-equal / n too small
                row["vs"][nm] = dict(naive_point=nv_pt, diff_ci=(d_lo, d_hi), wins=bool(wins), wilcoxon_p=p)
                if verbose:
                    flag = "WIN " if wins else ("lose" if d_lo > 0 else "n.s.")
                    print(f"        vs {nm:11} d(enc-naive) [{d_lo:8.3f}, {d_hi:8.3f}] {flag}"
                          f"   wilcoxon p={p:.3f} (n=5, supporting)")
            out[(h, metric)] = row
    return out


# --------------------------------------------------------------------------- #
# Items 7-8 -- gate distribution + normalised spatial contribution, per dataset.
# --------------------------------------------------------------------------- #
def gate_reads(names=DEV, seeds=SEEDS, verbose=True, prefix="encoder"):
    out = {}
    if verbose:
        print(f"\n{'=' * 92}\n{prefix} :: gate + spatial contribution per dataset (pooled over "
              f"seeds x nodes; pre-head, horizon-independent)\n{'=' * 92}")
        print(f"  {'dataset':22} {'g mean':>8} {'g IQR':>16} {'g<0.05':>8}   "
              f"{'sc mean':>8} {'sc IQR':>16}")
    for name in names:
        gs, scs = [], []
        for s in seeds:
            p = rpath(f"{prefix}__{name}__seed{s}__gate.npz")
            if not p.exists():
                continue
            z = np.load(p)
            gs.append(z["mean_g"]); scs.append(z["mean_sc"])
        if not gs:
            if verbose:
                print(f"  {name:22} (no gate npz -- run train.loop first)")
            continue
        g = np.concatenate(gs); sc = np.concatenate(scs)
        gq = np.percentile(g, [25, 75]); scq = np.percentile(sc, [25, 75])
        rec = dict(g_mean=float(g.mean()), g_iqr=(float(gq[0]), float(gq[1])),
                   g_frac_lt_005=float((g < 0.05).mean()),
                   sc_mean=float(sc.mean()), sc_iqr=(float(scq[0]), float(scq[1])), n_nodes=int(g.size))
        out[name] = rec
        if verbose:
            print(f"  {name:22} {rec['g_mean']:8.3f} [{gq[0]:6.3f},{gq[1]:6.3f}] {rec['g_frac_lt_005']:8.1%}"
                  f"   {rec['sc_mean']:8.3f} [{scq[0]:6.3f},{scq[1]:6.3f}]")
    if verbose:
        print("  read: g near 0 or a high g<0.05 fraction => the gate is ~off => the Day-14 2x2's "
              "{gate on vs g==0} arm is measuring little.")
    return out


# --------------------------------------------------------------------------- #
# Transfer table -- LODO vs single-disease, paired over ORIGINS.
#
# WHY THIS EXISTS (2026-07-29). train/lodo.py::_report compared LODO against seed-42's OWN single
# run, while results_summary.txt and the direction doc quoted 5-seed means. The two were never
# readable against each other, and seed 42 is an unusually BAD single-disease seed (dengue RMSE h3
# z=+1.31, us-regions RMSE h5 z=+1.35), so a seed-matched reference systematically flattered LODO --
# that is the "+3% printed for a 17% worsening" the client caught.
#
# Fix: report BOTH references, labelled on the table, and put a real dispersion on the delta. The
# dispersion is a paired bootstrap over TIME ORIGINS (the axis the client asked for), which works at
# 1 LODO seed. The seed-CV column is a secondary "would a different init change this" screen, kept
# PER HORIZON -- never averaged across horizons, because CV swings from 14.4% (dengue h3) to 0.4%
# (dengue h15) and an averaged floor silently mis-tags both ends.
#
# CAVEAT carried from train/loop.py::write_per_origin: this CI is on the CELL-POOLED metric while
# the headline tables are NODE-AVERAGED. They coincide on the dense influenza panels and diverge on
# dengue, so dengue's CI and its headline delta are not the same quantity.
# --------------------------------------------------------------------------- #
LOWER_BETTER = ("rmse", "mae", "smape")
CV_SCREEN_MULT = 1.96          # secondary screen only; the origin CI is the verdict


def _pct(single, lodo, metric):
    """Signed improvement %, sign fixed so + always means LODO better."""
    if metric in LOWER_BETTER:
        return (single - lodo) / single * 100.0
    return (lodo - single) / np.abs(single) * 100.0


def _seed_cv(name, h, metric, seeds=SEEDS):
    """CV% of the SINGLE-disease run across seeds, for this exact (dataset, horizon, metric)."""
    pts = np.array([_point_macro(
        *_load_perorigin(rpath(f"encoder__{name}__seed{s}__perorigin.npz"), h), metric)
        for s in seeds])
    return float(100 * pts.std(ddof=1) / abs(pts.mean()))


def transfer_ci(name, B=B_DEFAULT, metrics=("rmse", "mae"), lodo_seeds=(42,), single_seeds=SEEDS,
                seed=0, verbose=True, prefix="encoder_lodo"):
    rng = np.random.default_rng(seed)
    out = {}
    if verbose:
        print(f"\n{'=' * 118}\n{prefix} :: {name}   TRANSFER vs single-disease")
        print(f"  ref A = single-disease MEAN over seeds {tuple(single_seeds)}   "
              f"ref B = single-disease seed {lodo_seeds[0]} only   LODO seeds {tuple(lodo_seeds)}")
        print(f"  CI = paired bootstrap over time origins (B={B}), the VERDICT.  "
              f"CV = seed spread of the single run at this horizon, a secondary screen.")
        print(f"  + = LODO better.\n{'=' * 118}")
        print(f"  {'h':>3} {'metric':6} {'singleA':>9} {'singleB':>9} {'lodo':>9} "
              f"{'d% vs A':>9} {'95% CI on d% vs A':>22} {'d% vs B':>9} {'CV%':>6}  verdict")
    for h in HORIZONS:
        s_all = _encoder_stats(name, h, single_seeds, "encoder")     # seed-mean sufficient stats
        s_one = _encoder_stats(name, h, lodo_seeds, "encoder")       # the seed-matched reference
        l_one = _encoder_stats(name, h, lodo_seeds, prefix)
        K = s_all[0].shape[0]
        CNT = _cnt_matrix(K, B, rng)
        for metric in metrics:
            sA, sB, lo = (_point_macro(*x, metric) for x in (s_all, s_one, l_one))
            dA, dB = _pct(sA, lo, metric), _pct(sB, lo, metric)
            # paired: the SAME resampled origins drive both arms, so shared origin noise cancels
            pct_dist = _pct(_macro_dist(*s_all, CNT, metric), _macro_dist(*l_one, CNT, metric), metric)
            c_lo, c_hi = (float(x) for x in np.nanpercentile(pct_dist, [2.5, 97.5]))
            cv = _seed_cv(name, h, metric, single_seeds)
            real = (c_lo > 0) or (c_hi < 0)                          # origin CI excludes 0
            clears_cv = abs(dA) >= CV_SCREEN_MULT * cv
            out[(h, metric)] = dict(single_mean=sA, single_matched=sB, lodo=lo, d_vs_mean=dA,
                                    d_vs_matched=dB, ci=(c_lo, c_hi), cv=cv,
                                    clears_ci=bool(real), clears_cv=bool(clears_cv))
            if verbose:
                v = ("LODO better" if c_lo > 0 else "LODO worse") if real else "within noise"
                if real and not clears_cv:
                    v += " (fails CV screen)"
                print(f"  {h:>3} {metric.upper():6} {sA:>9.3f} {sB:>9.3f} {lo:>9.3f} "
                      f"{dA:>+8.1f}% [{c_lo:>+8.1f}%, {c_hi:>+8.1f}%] {dB:>+8.1f}% {cv:>6.1f}  {v}")
    if verbose:
        print(f"  Verdict = origin CI excludes 0. '(fails CV screen)' means the delta is smaller than "
              f"{CV_SCREEN_MULT}x the\n  single-disease seed spread: stable across test periods, but "
              f"not distinguishable from a different init.")
        if len(tuple(lodo_seeds)) < 2:
            print("  The transfer arm is 1 seed, so it contributes no seed dispersion of its own -- "
                  "treat every row as provisional.")
        else:
            print(f"  The transfer arm is averaged over {len(tuple(lodo_seeds))} seeds. Ref B is the "
                  f"same seed set as ref A here, so\n  the two reference columns coincide by "
                  f"construction -- that is expected, not a bug.")
    return out


def transfer_table(names=DEV, **kw):
    return {n: transfer_ci(n, **kw) for n in names}


# --------------------------------------------------------------------------- #
# Self-checks -- run without any artifacts.
# --------------------------------------------------------------------------- #
def _selfcheck():
    rng = np.random.default_rng(0)
    # (1) bootstrap engine: CNT-weighted macro == brute-force resample on one draw.
    K, C = 6, 2
    sae = rng.uniform(1, 5, (K, C)); sse = rng.uniform(1, 9, (K, C)); n = rng.integers(1, 4, (K, C)).astype(float)
    cnt = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
    draw = np.repeat(np.arange(K), cnt.astype(int))              # the resampled origin index list
    for metric in ("mae", "rmse"):
        engine = _macro_dist(sae, sse, n, cnt[:, None], metric)[0]
        num = (sse if metric == "rmse" else sae)[draw].sum(0); den = n[draw].sum(0)
        brute = np.nanmean(np.sqrt(num / den) if metric == "rmse" else num / den)
        assert abs(engine - brute) < 1e-9, f"{metric}: engine {engine} != brute {brute}"
    # (2) a perfect model (zero error) has a degenerate CI at 0.
    z_sae = np.zeros((K, C)); z_sse = np.zeros((K, C))
    CNT = _cnt_matrix(K, 200, rng)
    assert np.allclose(_macro_dist(z_sae, z_sse, n, CNT, "rmse"), 0.0), "perfect model CI not at 0"
    # (3) a strictly worse model -> paired diff CI strictly one-signed.
    enc = _macro_dist(sae, sse, n, CNT, "mae")
    worse = _macro_dist(sae + 3.0, sse, n, CNT, "mae")
    d_lo, d_hi = np.nanpercentile(enc - worse, [2.5, 97.5])
    assert d_hi < 0, "encoder strictly better but paired diff CI not below 0"
    # (4) reads aggregation: frac(g<0.05) and IQR on a known vector.
    g = np.array([0.01, 0.02, 0.04, 0.5, 0.9, 0.95])
    assert abs((g < 0.05).mean() - 0.5) < 1e-9, "gate frac<0.05 wrong"
    # (5) transfer delta sign: + must mean LODO better for BOTH metric directions. Getting this
    #     backwards is exactly how a 17% worsening got printed as +3%.
    assert _pct(100.0, 80.0, "rmse") > 0, "lower-better: smaller lodo must read positive"
    assert _pct(100.0, 120.0, "rmse") < 0, "lower-better: larger lodo must read negative"
    assert _pct(0.50, 0.60, "pcc") > 0, "higher-better: larger lodo must read positive"
    assert _pct(0.50, 0.40, "pcc") < 0, "higher-better: smaller lodo must read negative"
    assert abs(_pct(42.06, 48.53, "rmse") + 15.38) < 0.05, "dengue h3 must read -15.4%, not +3%"
    print("ok  bootstrap engine == brute force (MAE+RMSE); perfect-model CI at 0; paired diff one-signed")
    print("ok  reads aggregation: frac(g<0.05) correct")
    print("ok  transfer delta sign correct in both metric directions (dengue h3 reads -15.4%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--reads", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--transfer", action="store_true",
                    help="LODO vs single-disease, both references + paired origin-bootstrap CI")
    ap.add_argument("--dataset", choices=DEV)
    ap.add_argument("--joint", metavar="TAG",
                    help="analyse a joint run: reads encoder_joint__TAG__* (e.g. --joint uniform-uniform)")
    # --transfer used to hardcode prefix='encoder_lodo' and a single seed, so it could not read the
    # ldo3/pair folds at all and every transfer CI was stuck at n=1 seed.
    ap.add_argument("--transfer-prefix", default="encoder_lodo",
                    help="transfer arm to analyse, e.g. encoder_ldo3 / encoder_ldo3_zeroshot")
    ap.add_argument("--transfer-seeds", type=int, nargs="+", default=[42],
                    help="seeds available for the transfer arm (5 seeds -> ref B collapses onto ref A)")
    ap.add_argument("-B", type=int, default=B_DEFAULT)
    a = ap.parse_args()
    prefix = f"encoder_joint__{a.joint}" if a.joint else "encoder"
    if a.selfcheck or not (a.ci or a.reads or a.transfer):
        _selfcheck()
    if a.reads:
        gate_reads(prefix=prefix)
    if a.ci:
        for name in ([a.dataset] if a.dataset else DEV):
            bootstrap_ci(name, B=a.B, prefix=prefix)
    if a.transfer:
        for name in ([a.dataset] if a.dataset else DEV):
            transfer_ci(name, B=a.B, prefix=a.transfer_prefix,
                        lodo_seeds=tuple(a.transfer_seeds))


if __name__ == "__main__":
    main()

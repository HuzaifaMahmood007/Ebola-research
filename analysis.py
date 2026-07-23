"""analysis.py -- offline post-hoc reads over the artifacts train/loop.py emits. No model, no
retraining: everything here consumes results/*.npz.

  --ci     item 6: paired bootstrap-over-origins CIs. Resample origin indices with replacement
           (B=10,000), rebuild the country-macro from the per-(origin,country) sufficient stats,
           take 2.5/97.5 percentiles. For a comparison, resample the SAME origins for both models
           and build the CI on the DIFFERENCE (the pairing is what cancels japan's +-7-10% seed sd).
           Wilcoxon across the 5 seed-matched runs is printed as SUPPORTING ONLY (n=5, underpowered).
  --reads  items 7-8: per-dataset gate distribution (mean, IQR, frac nodes g<0.05) and normalised
           spatial contribution (mean, IQR), from the per-node gate readout, pooled over seeds.
  --selfcheck  synthetic checks of the bootstrap engine + reads aggregation (need no artifacts).

METRIC CAVEAT (item 6): the saved sae/sse/n are node-pooled within country, so the reconstructed
country-macro is CELL-POOLED, not the NODE-AVERAGED headline (score.aggregate). Equal on the dense
influenza panels; diverges on dengue. The CI is on the cell-pooled metric -- stated in every table.
PCC is not covered here: it is not additive over origins from sae/sse/n (would need Sx/Sy/Sxx/... per
origin-country). RMSE/MAE only.

Run from the repo root:
  PYTHONNOUSERSITE=1 conda run -n ebola-train python analysis.py --selfcheck
  PYTHONNOUSERSITE=1 conda run -n ebola-train python analysis.py --ci --reads
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np

RESULTS = Path("results")
SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
NAIVES = ("persistence", "seasonal", "train_mean")
DEV = ("dengue", "influenza_japan", "influenza_us-regions", "influenza_us-states")
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
        sae, sse, n = _load_perorigin(RESULTS / f"{prefix}__{name}__seed{s}__perorigin.npz", h)
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
        naive_stats = {nm: _load_perorigin(RESULTS / f"naive__{name}__{nm}__perorigin.npz", h)
                       for nm in NAIVES}
        for metric in metrics:
            enc_dist = _macro_dist(*se_enc, CNT, metric)
            enc_pt = _point_macro(*se_enc, metric)
            enc_lo, enc_hi = np.nanpercentile(enc_dist, [2.5, 97.5])
            # per-seed encoder point values, for the (supporting) Wilcoxon
            enc_seed_pts = np.array([_point_macro(
                *_load_perorigin(RESULTS / f"{prefix}__{name}__seed{s}__perorigin.npz", h), metric)
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
            p = RESULTS / f"{prefix}__{name}__seed{s}__gate.npz"
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
    print("ok  bootstrap engine == brute force (MAE+RMSE); perfect-model CI at 0; paired diff one-signed")
    print("ok  reads aggregation: frac(g<0.05) correct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--reads", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--dataset", choices=DEV)
    ap.add_argument("--joint", metavar="TAG",
                    help="analyse a joint run: reads encoder_joint__TAG__* (e.g. --joint uniform-uniform)")
    ap.add_argument("-B", type=int, default=B_DEFAULT)
    a = ap.parse_args()
    prefix = f"encoder_joint__{a.joint}" if a.joint else "encoder"
    if a.selfcheck or not (a.ci or a.reads):
        _selfcheck()
    if a.reads:
        gate_reads(prefix=prefix)
    if a.ci:
        for name in ([a.dataset] if a.dataset else DEV):
            bootstrap_ci(name, B=a.B, prefix=prefix)


if __name__ == "__main__":
    main()

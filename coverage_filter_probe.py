"""coverage_filter_probe.py -- what a "drop the sparse nodes" filter actually costs.

Tests the proposal: keep only well-observed dengue nodes, evaluate Ebola only on
complete cells. Measures the four things that decide whether that is a noise filter
or a sampling change:

  1. survival    -- how many nodes / cells / origins survive each coverage threshold
  2. graph       -- what the geographic graph looks like after the subset
  3. MNAR        -- whether completeness correlates with the signal (selection bias)
  4. windows     -- window-level input coverage, the unit the encoder actually reads

Read-only. Touches no bundle. Prints a report.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

import bundles as B

W, HORIZONS = B.W, B.HORIZONS
MAX_H = B.MAX_H
THRESH = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]


def survival(b: B.Bundle) -> None:
    M = b.M.astype(bool)
    N, T = M.shape
    comp = M.mean(1)
    country = b.group_of()
    ids = b.meta["node_ids"]
    obs_total = M.sum()

    print(f"\n{'thr':>5} {'nodes':>7} {'%nodes':>7} {'obs cells':>12} {'%obs':>6} "
          f"{'countries':>10} {'isolated':>9} {'components':>11}")
    for t in THRESH:
        keep = comp >= t if t > 0 else np.ones(N, bool)
        if keep.sum() == 0:
            print(f"{t:>5.0%} {0:>7}   -- nothing survives --")
            continue
        A = b.A_geo[np.ix_(keep, keep)].copy()
        np.fill_diagonal(A, 0)
        deg = (A > 0).sum(1)
        ncomp, _ = connected_components(csr_matrix(A > 0), directed=False)
        cs = {country[ids[i]] for i in np.flatnonzero(keep)}
        print(f"{t:>5.0%} {keep.sum():>7,} {keep.mean():>6.1%} {M[keep].sum():>12,} "
              f"{M[keep].sum() / obs_total:>5.1%} {len(cs):>10} {int((deg == 0).sum()):>9,} "
              f"{ncomp:>11,}")


def mnar(b: B.Bundle) -> None:
    """Is missingness related to the signal? If yes, filtering on it moves the distribution."""
    M = b.M.astype(bool)
    comp = M.mean(1)
    raw = b.raw.astype(float)
    mean_inc = np.array([raw[i][M[i]].mean() if M[i].any() else np.nan for i in range(len(comp))])
    ok = ~np.isnan(mean_inc)
    r, p = spearmanr(comp[ok], mean_inc[ok])
    print(f"  spearman(node completeness, node mean incidence) = {r:+.3f}  (p={p:.1e}, "
          f"n={ok.sum():,})")
    for t in (0.0, 0.5):
        keep = (comp >= t) & ok if t else ok
        v = raw[np.ix_(keep)][M[keep]]
        print(f"  thr {t:.0%}: kept {keep.sum():>5,} nodes | median obs incidence "
              f"{np.median(v):>7.1f} | mean {v.mean():>9.1f} | zero-share {(v == 0).mean():>5.1%}")

    country = b.group_of()
    ids = b.meta["node_ids"]
    cs = np.array([country[i] for i in ids])
    print("\n  country mix, top 8 by node count (full -> thr 50%):")
    keep = comp >= 0.5
    for c, n in sorted(((c, (cs == c).sum()) for c in set(cs)), key=lambda x: -x[1])[:8]:
        k = ((cs == c) & keep).sum()
        print(f"    {c:<24} {n:>6,} -> {k:>6,}  ({k / n:>5.1%} survive)")
    print(f"    {'countries wiped out':<24} "
          f"{sum(1 for c in set(cs) if ((cs == c) & keep).sum() == 0):>6,} of {len(set(cs)):,}")


def windows(b: B.Bundle, sample: int = 400) -> None:
    """Coverage of the W=20 INPUT window per (node, origin) -- the unit the trunk reads."""
    M = b.M.astype(bool)
    N, T = M.shape
    origins = b.origins()
    rng = np.random.default_rng(0)
    take = origins if len(origins) <= sample else \
        [origins[i] for i in rng.choice(len(origins), sample, replace=False)]
    cov = np.concatenate([M[:, t - W + 1:t + 1].mean(1) for t in take])
    tgt = np.concatenate([np.stack([M[:, t + h] for h in HORIZONS]).all(0) for t in take])
    print(f"  input-window coverage over {len(take):,} sampled origins x {N:,} nodes "
          f"= {cov.size:,} pairs")
    for t in (1.0, 0.9, 0.5, 0.2):
        print(f"    windows with >= {t:>4.0%} of the 20 lookback steps observed: {(cov >= t).mean():>6.2%}")
    print(f"    windows that are fully observed AND have all 4 horizons observed: "
          f"{((cov >= 1.0) & tgt).mean():>6.2%}")
    print(f"    windows with >=50% coverage AND all 4 horizons observed:          "
          f"{((cov >= 0.5) & tgt).mean():>6.2%}")


def ebola_eval(b: B.Bundle) -> None:
    M = b.M.astype(bool)
    N, T = M.shape
    comp = M.mean(1)
    q, s = b.masks()["query"].astype(bool), b.masks()["support"].astype(bool)
    print(f"  node completeness: max {comp.max():.1%}  median {np.median(comp):.1%}  "
          f"nodes at 100% = {(comp == 1.0).sum()}")
    print(f"  weeks with every node observed: {(M.all(0)).sum()} of {T}")
    print(f"  eval already masked to observed cells? split&M == split -> "
          f"{np.array_equal(q & M, q)} (query), {np.array_equal(s & M, s)} (support)")
    print(f"  support cells {s.sum()} over {(s.sum(1) > 0).sum()} nodes | "
          f"query cells {q.sum()} over {(q.sum(1) > 0).sum()} nodes")
    print("  scorable query cells per horizon (origin t, target t+h in query & observed):")
    for h in HORIZONS:
        n = sum(int((q[:, t + h] & M[:, t + h]).sum()) for t in b.origins() if t + h < T)
        print(f"    h={h:<3} {n:>6,} cells")


def main():
    d = B.load("dengue")
    print("=" * 84)
    print("DENGUE — survival under a node-coverage filter")
    print("=" * 84)
    survival(d)
    print("\n" + "=" * 84)
    print("DENGUE — is missingness informative (MNAR)?")
    print("=" * 84)
    mnar(d)
    print("\n" + "=" * 84)
    print("DENGUE — window-level coverage (what the encoder actually reads)")
    print("=" * 84)
    windows(d)

    print("\n" + "=" * 84)
    print("EBOLA — can it be evaluated on 'complete weeks/nodes'?")
    print("=" * 84)
    ebola_eval(B.load("ebola"))

    print("\n" + "=" * 84)
    print("CONTROL — the same window check on a 100%-complete panel")
    print("=" * 84)
    windows(B.load("influenza_us-states"))


def _selfcheck():
    """The window-coverage arithmetic, on a hand-built bundle-shaped mask."""
    M = np.zeros((2, 40), np.uint8)
    M[0, :] = 1                                  # node 0 fully observed
    M[1, ::2] = 1                                # node 1 observed every other week
    cov = M[:, 0:W].mean(1)
    assert cov[0] == 1.0 and abs(cov[1] - 0.5) < 1e-9, cov
    A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], float)
    n, _ = connected_components(csr_matrix(A > 0), directed=False)
    assert n == 2, f"two components expected, got {n}"
    print("selfcheck ok\n")


if __name__ == "__main__":
    _selfcheck()
    main()

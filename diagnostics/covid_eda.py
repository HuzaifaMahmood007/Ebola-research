"""EDA for the covid_us-states bundle, against influenza_us-states on the identical node set."""
import numpy as np
import pandas as pd

import bundles as B
from to_schema import US_STATES_49

pd.set_option("display.width", 200)
c, f = B.load("covid_us-states"), B.load("influenza_us-states")
raw, M, dates = c.raw.astype(float), c.M.astype(bool), pd.DatetimeIndex(c.meta["dates"])
N, T = raw.shape
S = np.array(US_STATES_49)

print("=" * 78); print("1. SHAPE / CALENDAR / SPLITS"); print("=" * 78)
print(f"X {c.X.shape}  raw {raw.shape}  A_geo {c.A_geo.shape}  C {c.C.shape}")
print(f"calendar   {dates[0].date()} -> {dates[-1].date()}   T={T} weeks   cadence W-SAT")
for ph, m in c.masks().items():
    cols = np.flatnonzero(m.any(0))
    print(f"  {ph:6s} weeks {len(cols):4d}  {dates[cols[0]].date()} -> {dates[cols[-1]].date()}"
          f"   cells {int(m.sum()):,}  origins {len(c.origins(phase=ph))}")

print("\n" + "=" * 78); print("2. MISSINGNESS"); print("=" * 78)
print(f"M density                 {M.mean():.6f}   (unobserved cells: {int((~M).sum())})")
print(f"obs_mask channel unique   {np.unique(c.X[:, :, 3])}")
zero = raw == 0
print(f"raw cells exactly zero    {zero.mean():.1%}   ({int(zero.sum()):,} of {raw.size:,})")
lead = np.array([np.argmax(r > 0) if (r > 0).any() else T for r in raw])
print(f"leading all-zero weeks    min {lead.min()}  median {int(np.median(lead))}  max {lead.max()}"
      f"  ({S[lead.argmax()]})")
print("\nzero-increment weeks per YEAR (a proxy for reporting cadence, not for absence of disease):")
yr = dates.year
for y in sorted(set(yr)):
    sel = yr == y
    print(f"  {y}   weeks={sel.sum():3d}   zero-increment cells {zero[:, sel].mean():6.1%}"
          f"   states with >50% zero weeks: {(zero[:, sel].mean(1) > 0.5).sum():2d}/49")
worst = np.argsort(-zero[:, yr >= 2022].mean(1))[:6]
print("  worst 2022+ reporters:", ", ".join(f"{S[i]} {zero[i, yr>=2022].mean():.0%}" for i in worst))

print("\n" + "=" * 78); print("3. TARGET DISTRIBUTION"); print("=" * 78)
v = raw[M]
q = np.percentile(v, [50, 90, 99, 99.9])
print(f"n={v.size:,}  mean {v.mean():,.1f}  sd {v.std():,.1f}  max {v.max():,.0f} "
      f"({S[np.unravel_index(raw.argmax(), raw.shape)[0]]}, "
      f"{dates[np.unravel_index(raw.argmax(), raw.shape)[1]].date()})")
print(f"median {q[0]:,.0f}   p90 {q[1]:,.0f}   p99 {q[2]:,.0f}   p99.9 {q[3]:,.0f}")
print(f"skew {float(pd.Series(v).skew()):.1f}   zeros {(v == 0).mean():.1%}")

print("\n-- Doubt.md §3.1 predictor: count-space mean vs expm1(mean(log1p)) per node --")
for nm, b in (("covid", c), ("influenza", f)):
    r, mm = b.raw.astype(float), b.M.astype(bool)
    tm = b.masks()["train"].astype(bool)
    rat = []
    for i in range(r.shape[0]):
        s = r[i, tm[i]]
        if s.size and s.mean() > 0:
            rat.append(s.mean() / np.expm1(np.mean(np.log1p(s))))
    rat = np.array(rat)
    print(f"  {nm:10s} median {np.median(rat):6.2f}x   p90 {np.percentile(rat,90):7.2f}x   "
          f"max {rat.max():8.2f}x   (bigger => bigger bias_c)")

print("\n-- §1.3 graph diagnostic: variance of node-level means of X[:,:,0] --")
for nm in ("covid_us-states", "influenza_us-states", "influenza_japan", "dengue"):
    b = B.load(nm)
    print(f"  {nm:22s} {b.X[:, :, 0].mean(1).var():.4f}")

print("\n" + "=" * 78); print("4. REGIME SHIFT ACROSS THE 50/20/30 CUT"); print("=" * 78)
for ph, m in c.masks().items():
    cols = np.flatnonzero(m.any(0))
    blk = raw[:, cols]
    print(f"  {ph:6s} national mean/wk {blk.sum(0).mean():10,.0f}   peak wk {blk.sum(0).max():10,.0f}"
          f"   node-mean {blk.mean():8,.1f}")
tr = raw[:, np.flatnonzero(c.masks()["train"].any(0))]
te = raw[:, np.flatnonzero(c.masks()["test"].any(0))]
print(f"  test/train national mean ratio {te.sum(0).mean()/tr.sum(0).mean():.2f}x "
      f"-- the wild-type/Alpha -> Omicron+ shift")

print("\n" + "=" * 78); print("5. WAVES (national, top 6 weeks)"); print("=" * 78)
nat = raw.sum(0)
for i in np.argsort(-nat)[:6]:
    print(f"  {dates[i].date()}  {nat[i]:12,.0f}   ({nat[i]/np.median(nat):5.1f}x median week)")

print("\n" + "=" * 78); print("6. COVID vs INFLUENZA ON THE SAME 49 NODES"); print("=" * 78)
ct, ft = c.raw.sum(1), f.raw.sum(1)
print(f"  graph identical      {np.array_equal(c.A_geo, f.A_geo)}")
print(f"  covariates identical {np.allclose(c.C, f.C)}")
print(f"  node ids disjoint    {not (set(c.meta['node_ids']) & set(f.meta['node_ids']))}")
print(f"  T                    covid {c.raw.shape[1]}  vs influenza {f.raw.shape[1]}")
print(f"  spearman(node totals) {pd.Series(ct).corr(pd.Series(ft), method='spearman'):.3f}"
      f"   pearson {pd.Series(ct).corr(pd.Series(ft)):.3f}")
print(f"  dynamic range (max/median node total)  covid {ct.max()/np.median(ct):.1f}x   "
      f"influenza {ft.max()/np.median(ft):.1f}x")
top = np.argsort(-ct)[:5]
print("  top-5 by covid total:", ", ".join(f"{S[i]} {ct[i]:,.0f}" for i in top))
print("  same states by flu  :", ", ".join(f"{S[i]} {ft[i]:,.0f}" for i in top))

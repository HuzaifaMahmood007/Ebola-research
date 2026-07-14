"""Build the three ColaGNN influenza bundles and assert their calendars and geography.

The matrices and adjacencies are the shipped benchmark files, used unchanged, so the influenza
results sit directly beside the published baselines. The ColaGNN and EpiGNN copies are
byte-identical; EpiGNN's 50x50 state adjacency does not pair with the 49-node state matrix and
is not used.

The three are kept as INDEPENDENT bundles and never concatenated: they have disjoint node sets
and different calendars, and transfer here operates through shared model parameters rather than
shared indices. That independence is asserted below rather than merely intended.

Everything this script prints, it asserts.
"""
import hashlib
import pathlib

import numpy as np
import pandas as pd

from to_schema import INFLUENZA_START, load_influenza

RAW = pathlib.Path("data/Final datasets/influenza")
GADM_DIR = "data/gadm"

# Published spans. The shipped matrices are undated, so a length that disagrees means the
# recorded start date is wrong for that matrix.
DATASETS = {
    "japan":      dict(matrix="japan.txt",     adj="japan-adj.txt",  T=348, N=47),
    "us-regions": dict(matrix="region785.txt", adj="region-adj.txt", T=785, N=10),
    "us-states":  dict(matrix="state360.txt",  adj="state-adj.txt",  T=360, N=49),
}


def verify_checksums(d: pathlib.Path) -> int:
    """Fail loud if a cached raw drifted from the checksummed snapshot."""
    sums = {}
    for line in (d / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*")] = digest
    for name, want in sums.items():
        got = hashlib.sha256((d / name).read_bytes()).hexdigest()
        assert got == want, f"{name}: checksum drift\n  want {want}\n  got  {got}"
    return len(sums)


print(f"checksums       {verify_checksums(RAW)} files verified against SHA256SUMS.txt\n")

bundles = {}
for name, spec in DATASETS.items():
    dt = load_influenza(
        str(RAW / spec["matrix"]),
        str(RAW / spec["adj"]),
        dataset=name,
        ratios=(0.5, 0.2, 0.3),
        gadm_dir=GADM_DIR,          # builds the static covariates C
    )
    bundles[name] = dt

    N, T, F = dt.X.shape
    m, A = dt.meta, dt.A_geo

    # A necessary condition on the anchor, though nothing like a sufficient one (see below).
    assert (T, N) == (spec["T"], spec["N"]), \
        f"{name}: got [T={T},N={N}], published span is [T={spec['T']},N={spec['N']}] " \
        f"-- do not trust INFLUENZA_START, override start_date explicitly"

    print(f"influenza:{name}")
    print(f"  X={dt.X.shape}  A_geo={A.shape} ({m['A_geo_kind']})  M={dt.M.shape}")
    print(f"  calendar      {m['dates'][0].date()} -> {m['dates'][-1].date()}  "
          f"({T} weeks, anchored {INFLUENZA_START[name]})")

    # THE anchor guard. The row count above is not sufficient: it matches the published span
    # under a wrong anchor too, and both US sets were once 39 weeks out, which put the
    # seasonality channels ~180 degrees out of phase. Seasonal phase is the real evidence:
    # these are northern-hemisphere series, so the aggregate must peak in winter.
    agg = pd.Series(
        pd.read_csv(RAW / spec["matrix"], header=None).to_numpy(dtype=float).sum(1),
        index=pd.DatetimeIndex(m["dates"]),
    )
    by_month = agg.groupby(agg.index.month).mean()
    peak_m, trough_m = int(by_month.idxmax()), int(by_month.idxmin())
    print(f"  seasonality   peak {pd.Timestamp(2020, peak_m, 1):%b}  "
          f"trough {pd.Timestamp(2020, trough_m, 1):%b}  "
          f"amplitude {by_month.max() / max(by_month.min(), 1e-9):.0f}x")
    assert peak_m in (12, 1, 2, 3), (
        f"{name}: aggregate ILI peaks in month {peak_m}, not winter -- the date anchor is "
        f"wrong. sin_doy/cos_doy would be out of phase. Recover the true start (see "
        f"INFLUENZA_START) rather than trusting the published year range.")
    assert trough_m in (6, 7, 8, 9, 10), \
        f"{name}: ILI troughs in month {trough_m}, not summer -- anchor is wrong"
    print(f"  features      {m['feature_names']}  core_idx={m['core_feature_idx']}")
    print(f"  split         {m['split_scheme']}  |  scaler {m['scaler_scope']}")
    print(f"  mask density  {dt.M.mean():.4f}  (shipped matrices are fully observed)")

    deg = A.sum(1)
    iso = m["isolated_nodes"]
    print(f"  graph         {int(A.sum() // 2)} undirected edges  "
          f"symmetric={np.array_equal(A, A.T)}  self-loops={int(np.diag(A).sum())}")
    print(f"                degree: mean={deg.mean():.2f}  min={int(deg.min())}  "
          f"max={int(deg.max())}  isolated={len(iso)}"
          f"{'  ' + ','.join(iso) if iso else ''}")

    # A mis-joined geometry would be SILENT -- plausible numbers on the wrong nodes, and nothing
    # downstream complains -- so C is checked against independently known geography. The
    # largest-area node also confirms the recovered node ordering: Alaska must be the biggest US
    # state, Hokkaido the biggest Japanese prefecture.
    lat, lon, area = dt.C[:, 0], dt.C[:, 1], dt.C[:, 2]
    biggest = int(area.argmax())
    print(f"  covariates    C={dt.C.shape} {m['covariates']}  "
          f"lat [{lat.min():.1f},{lat.max():.1f}]  lon [{lon.min():.1f},{lon.max():.1f}]")
    print(f"                largest: {m['node_ids'][biggest]} @ {area[biggest]:,.0f} km2"
          f"   total {area.sum():,.0f} km2")
    assert not (dt.C == 0).all(), f"{name}: C must be populated, not the zero placeholder"
    box = dict(japan=(24, 46, 122, 146)).get(name, (18, 72, -180, -66))   # US spans AK + HI
    assert box[0] <= lat.min() and lat.max() <= box[1], f"{name}: centroid latitude out of range"
    assert box[2] <= lon.min() and lon.max() <= box[3], f"{name}: centroid longitude out of range"
    assert (area > 0).all(), f"{name}: non-positive area -- geometry join is wrong"
    EXPECT_BIGGEST = {"japan": "japan_10", "us-states": "us-states_1"}    # Hokkaido, Alaska
    if name in EXPECT_BIGGEST:
        assert m["node_ids"][biggest] == EXPECT_BIGGEST[name], (
            f"{name}: largest-area node is {m['node_ids'][biggest]}, expected "
            f"{EXPECT_BIGGEST[name]} -- the node ordering or the GADM join is wrong")

    assert np.array_equal(A, A.T), f"{name}: shipped adjacency must stay symmetric"
    assert np.diag(A).sum() == 0, f"{name}: diagonal must be zeroed (encoder adds I once)"

    # The edges must be reused verbatim: zeroing the diagonal may not change a single
    # off-diagonal entry, nor isolate a node that HAD a neighbour in the shipped file. Nodes
    # isolated here are inherited -- the self-loop was their only entry -- and are never patched
    # with an invented edge, which would forfeit the comparability the shipped graph exists for.
    shipped = pd.read_csv(RAW / spec["adj"], header=None).to_numpy(dtype=np.float32)
    off = shipped.copy()
    np.fill_diagonal(off, 0.0)
    assert np.array_equal(A, off), f"{name}: off-diagonal edges must be bit-for-bit shipped"
    assert set(np.flatnonzero(off.sum(1) == 0)) == set(np.flatnonzero(deg == 0)), \
        f"{name}: diag-zeroing isolated a node that HAD a neighbour -- would be a real bug"

    dt.check()
    print(f"  .check()      PASSED\n")

# --------------------------------------------------------------------------------------- #
# The H1N1 anchor pin -- the sharpest evidence that the us-regions calendar is right, so it is
# a build-time gate rather than a one-off check. The 2009 H1N1 autumn wave is the series' all-
# time maximum and its only major autumn peak, so the global argmax must land on the week ending
# 2009-10-24. Nothing else pins the anchor to a single week: the winter-peak test above leaves a
# few weeks of ambiguity.
# --------------------------------------------------------------------------------------- #
reg = pd.read_csv(RAW / DATASETS["us-regions"]["matrix"], header=None).to_numpy(float).sum(1)
peak_date = pd.DatetimeIndex(bundles["us-regions"].meta["dates"])[int(reg.argmax())]
assert peak_date == pd.Timestamp("2009-10-24"), (
    f"us-regions: series maximum lands on {peak_date.date()}, not the 2009 H1N1 peak week "
    f"(2009-10-24) -- the date anchor has drifted")
print(f"H1N1 ANCHOR PIN us-regions global max -> {peak_date.date()} "
      f"= the 2009 H1N1 fall wave, ILINet's all-time peak week. Anchor confirmed.\n")

# --------------------------------------------------------------------------------------- #
# Three INDEPENDENT bundles, never concatenated. They share neither a node set nor a calendar,
# and transfer operates through shared model parameters rather than shared indices. These
# asserts make "do not concatenate" a checked property rather than a convention that someone
# later breaks by stacking the matrices.
# --------------------------------------------------------------------------------------- #
names = list(bundles)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        na, nb = set(bundles[a].meta["node_ids"]), set(bundles[b].meta["node_ids"])
        assert not (na & nb), f"{a} and {b} share node ids -- node sets must be disjoint"
        da = pd.DatetimeIndex(bundles[a].meta["dates"])
        db = pd.DatetimeIndex(bundles[b].meta["dates"])
        assert not (len(da) == len(db) and (da == db).all()), \
            f"{a} and {b} have an identical calendar -- they are meant to be distinct streams"
print(f"INDEPENDENCE    {len(names)} bundles, pairwise-disjoint node sets and distinct "
      f"calendars -- kept separate, never concatenated")

# The shared encoder must not be able to tell which disease it is looking at. The transfer view
# is the enforcement point: identical core channels across every bundle.
cores = {
    n: tuple(dt.meta["feature_names"][i] for i in dt.meta["core_feature_idx"])
    for n, dt in bundles.items()
}
assert len(set(cores.values())) == 1, f"core channels diverge across bundles: {cores}"
for n, dt in bundles.items():
    assert dt.transfer_view().shape[2] == len(cores[n]), f"{n}: transfer_view width"
print(f"TRANSFER VIEW   identical core channels across all 3 bundles: "
      f"{list(next(iter(cores.values())))}")
print("COVID           excluded from the schema: the development set is dengue + influenza. "
      "COVID entered the reproduction study only.")

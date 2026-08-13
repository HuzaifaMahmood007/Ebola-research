"""Build the COVID-19 US-states bundle on the SAME 49 nodes and shipped graph as influenza_us-states.

    conda run -n ebola python covid_load.py [--refresh]

Reusing ColaGNN's ordering and `state-adj.txt` verbatim means the only new thing in the repo is a
case series, and it buys the point no other cell can make: COVID and influenza sit on an IDENTICAL
graph with IDENTICAL covariates, so transfer between them isolates DISEASE transfer from GRAPH
transfer. Three costs, stated rather than hidden: the 49-state set EXCLUDES Florida (~6.5% of US
population, dropped by ColaGNN because ILINet does not report it) and DC, since a 50th node would
fork the adjacency and forfeit the identical-graph property; the shipped adjacency is land
contiguity built for ILI and must not be described as COVID-specific; and COVID 2020-2023 is
NPI-dominated, so transfer may reflect policy response rather than pathogen dynamics.

Cumulative -> incidence differences the running MAXIMUM, the same mass-preserving transform as
Ebola: NYT revises cumulative counts downward on occasion and differencing raw would release those
corrections back as fresh incidence, the defect that fabricated 35.8% of the Ebola target. Nothing is
masked, unlike Ebola: absence of a row before a state's first case is a genuine zero, so M is
all-ones and the LOCF input fill is a no-op here. Everything this script prints, it asserts."""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import urllib.request

import numpy as np
import pandas as pd

from build_datasets import write
from to_schema import US_STATES_49, load_influenza

NYT_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
RAW_DIR = pathlib.Path("data/Final datasets/covid")
CSV = RAW_DIR / "us-states.csv"
ADJ = pathlib.Path("data/Final datasets/influenza/state-adj.txt")   # the SHIPPED ColaGNN graph
MATRIX = RAW_DIR / "covid_state_weekly.txt"                          # [T,49], built here
GADM_DIR = "data/gadm"
NAME = "covid_us-states"

# NYT froze on 2023-03-23 (a Thursday), so the final W-SAT bucket is partial and is dropped.
LAST_COMPLETE_WEEK = "2023-03-18"


def fetch(refresh: bool = False) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if refresh or not CSV.exists():
        print(f"downloading      {NYT_URL}")
        urllib.request.urlretrieve(NYT_URL, CSV)
    digest = hashlib.sha256(CSV.read_bytes()).hexdigest()
    print(f"raw              {CSV}  {CSV.stat().st_size:,} bytes\n"
          f"                 sha256 {digest}")
    return digest


def build_matrix() -> pd.DataFrame:
    """NYT daily cumulative -> [T, 49] weekly incidence in ColaGNN's exact column order."""
    df = pd.read_csv(CSV, parse_dates=["date"], usecols=["date", "state", "cases"])
    have = set(df["state"].unique())
    missing = [s for s in US_STATES_49 if s not in have]
    assert not missing, f"NYT is missing states from the 49-node set: {missing}"

    piv = (df[df["state"].isin(US_STATES_49)]
           .pivot_table(index="date", columns="state", values="cases", aggfunc="last")
           .reindex(columns=US_STATES_49))                     # exact node order, not alphabetical luck
    assert list(piv.columns) == US_STATES_49, "column order drifted from US_STATES_49"

    # Daily grid: forward-fill the cumulative (a missing day is "no new report", not zero cases),
    # then fill the LEADING gap with 0 -- before a state's first case its cumulative genuinely is 0.
    piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="D")).ffill().fillna(0.0)

    wk = piv.resample("W-SAT").last()                          # last cumulative in each MMWR week
    wk = wk[wk.index <= pd.Timestamp(LAST_COMPLETE_WEEK)]
    env = wk.cummax()                                          # monotone envelope
    inc = env.diff().iloc[1:]                                  # first week's increment is not identifiable
    inc = inc[inc.sum(axis=1) > 0]                             # drop the all-zero pre-outbreak head

    # Mass conservation, checked against the SOURCE column rather than against the transform:
    # increments must sum to env[last] - env[first_kept_week - 1], exactly.
    ref = env.loc[inc.index[0]:].iloc[-1] - env.loc[:inc.index[0]].iloc[-2]
    got = inc.sum(axis=0)
    assert np.allclose(got.to_numpy(), ref.to_numpy(), rtol=0, atol=1e-6), \
        f"mass not preserved; worst state {(got - ref).abs().idxmax()}"
    assert (inc.to_numpy() >= 0).all(), "negative incidence survived the cummax envelope"
    return inc


def main(refresh: bool = False) -> None:
    digest = fetch(refresh)
    inc = build_matrix()
    T, N = inc.shape
    start = inc.index[0]
    assert start.dayofweek == 5, f"calendar anchor {start.date()} is not a Saturday (W-SAT)"
    np.savetxt(MATRIX, inc.to_numpy(), fmt="%.0f", delimiter=",")
    print(f"matrix           {MATRIX}  [T={T}, N={N}]  "
          f"{inc.index[0].date()} -> {inc.index[-1].date()}")

    dt = load_influenza(str(MATRIX), str(ADJ), dataset=NAME, start_date=str(start.date()),
                        ratios=(0.5, 0.2, 0.3), gadm_dir=GADM_DIR,
                        disease="covid:us-states", covariates_as="us-states")
    m, A = dt.meta, dt.A_geo
    dt.meta["source"] = ("New York Times covid-19-data (us-states.csv, archived 2023-03-23); "
                         "graph reused from ColaGNN (CIKM 2020) state-adj.txt")
    dt.meta["raw_sha256"] = digest
    dt.meta["excluded_nodes"] = ["Florida", "District of Columbia"]
    dt.meta["npi_confounded"] = True

    print(f"\ncovid:us-states")
    print(f"  X={dt.X.shape}  A_geo={A.shape} ({m['A_geo_kind']})  M={dt.M.shape}")
    print(f"  calendar      {m['dates'][0].date()} -> {m['dates'][-1].date()}  ({T} weeks)")
    print(f"  features      {m['feature_names']}  core_idx={m['core_feature_idx']}")
    print(f"  split         {m['split_scheme']}  |  scaler {m['scaler_scope']}")
    print(f"  mask density  {dt.M.mean():.4f}  (NYT is dense; nothing is imputed)")

    # ---- the Omicron anchor pin. The matrices carry no dates of their own, so the calendar is an
    # assertion, not an input. US COVID cases peaked in mid-January 2022 by a very wide margin
    # (Omicron) -- a single unambiguous global maximum, the same role the H1N1 week plays for
    # us-regions. A drifted anchor moves this and puts sin_doy/cos_doy out of phase.
    national = dt.raw.sum(0)
    peak = pd.DatetimeIndex(m["dates"])[int(national.argmax())]
    ratio = national.max() / np.median(national)
    assert pd.Timestamp("2022-01-01") <= peak <= pd.Timestamp("2022-01-31"), (
        f"national COVID peak lands on {peak.date()}, not January 2022 (Omicron) -- the date "
        f"anchor has drifted")
    print(f"  OMICRON PIN   national max -> {peak.date()} ({ratio:.1f}x the median week). "
          f"Anchor confirmed.")

    # ---- the graph is the shipped one, bit for bit (same gate influenza_load.py applies)
    shipped = pd.read_csv(ADJ, header=None).to_numpy(dtype=np.float32)
    off = shipped.copy()
    np.fill_diagonal(off, 0.0)
    assert np.array_equal(A, off), "off-diagonal edges must be bit-for-bit the shipped graph"
    assert np.array_equal(A, A.T) and np.diag(A).sum() == 0, "graph must stay symmetric, diag-zero"

    # ---- identical geography to influenza_us-states, DISJOINT node ids. Both halves matter: the
    # shared graph is the point of the bundle, and colliding ids would silently merge two diseases.
    import bundles as B
    flu = B.load("influenza_us-states")
    assert np.array_equal(A, flu.A_geo), "COVID graph differs from influenza_us-states"
    assert np.allclose(dt.C, flu.C), "COVID covariates differ from influenza_us-states"
    assert not (set(m["node_ids"]) & set(flu.meta["node_ids"])), "node ids collide with influenza"
    biggest = int(dt.C[:, 2].argmax())
    assert m["node_ids"][biggest] == f"{NAME}_1", \
        f"largest-area node is {m['node_ids'][biggest]}, expected Alaska at index 1"
    print(f"  geography     A_geo and C identical to influenza_us-states; node ids disjoint; "
          f"largest = {m['node_ids'][biggest]} (Alaska)")

    # ---- the transfer view must be indistinguishable in SHAPE and NAMES from every other bundle
    core = tuple(m["feature_names"][i] for i in m["core_feature_idx"])
    flu_core = tuple(flu.meta["feature_names"][i] for i in flu.meta["core_feature_idx"])
    assert core == flu_core, f"core channels diverge from influenza: {core} vs {flu_core}"
    assert dt.transfer_view().shape == (N, T, 4), f"transfer_view {dt.transfer_view().shape}"
    print(f"  transfer view {list(core)}  -- identical to the other bundles")

    dt.check()
    path = write(NAME, dt)
    print(f"  .check()      PASSED")
    print(f"\nwrote            {path}  ({path.stat().st_size / 1e6:.2f} MB)")
    # Corrected against covid_eda.md: the train->test level shift is 1.03x, i.e. essentially none.
    # The problem is inside VAL, which owns the Omicron peak (5.14M, 6.3x the largest test week) and
    # runs at 2.8x both other folds. Early stopping AND the val-fitted bias correction are both
    # selected on it.
    print("NOTE             the Omicron peak (2022-01-15) falls in the VAL fold: val runs at 2.8x "
          "train and 2.8x test,\n                 and its peak is 6.3x the largest test week "
          "(train->test level ratio is only 1.03x).\n                 Early stopping and the "
          "val-fitted bias_c are both compromised for THIS bundle. See covid_eda.md §4.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download us-states.csv")
    main(**vars(ap.parse_args()))

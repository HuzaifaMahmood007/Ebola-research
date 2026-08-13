"""COVID-19 US-states driver: fetch the source, build through the schema, assert everything.

    conda run -n ebola python loaders/covid_load.py [--refresh]

Same shape as the other three drivers: `to_schema.load_covid` builds the tensors and
`build_datasets.py` writes the bundle. This file does the two things that are COVID's alone --
fetching the NYT source, which is downloaded rather than placed by hand, and the gates below -- and
it writes nothing.

The bundle sits on the SAME 49 nodes and the SAME shipped ColaGNN graph as influenza_us-states, so
transfer between the two isolates DISEASE transfer from GRAPH transfer; every other cross-disease
cell in this study confounds them. Three costs travel with that and must travel with any number
derived from it: the 49-state set excludes Florida (ILINet does not report it, so ColaGNN dropped it)
and DC, roughly 6.5% of the US population; the adjacency is land contiguity built for ILI and is not
a COVID-specific graph; and 2020-2023 is NPI-dominated, so transfer may reflect policy response
rather than pathogen dynamics.

Everything this script prints, it asserts.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import to_schema as ts
from build_datasets import COVID_CSV, COVID_MATRIX, FLU, GADM, INFLUENZA, RAW_SHA256

NYT_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
ADJ = FLU / INFLUENZA["us-states"]["adj"]          # the SHIPPED ColaGNN graph
NAME = "covid:us-states"


def fetch(refresh: bool = False) -> str:
    """Download us-states.csv if absent, then verify it against the pinned hash.

    The pin is the point: NYT_URL reads the repository's `master` branch, which is mutable by
    construction. The repo was archived in March 2023 so in practice it is frozen, but 'in practice'
    is not a guarantee, and a silently changed source would rebuild every COVID number in the study
    without a signal."""
    COVID_CSV.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not COVID_CSV.exists():
        print(f"downloading      {NYT_URL}")
        urllib.request.urlretrieve(NYT_URL, COVID_CSV)
    digest = ts._sha256(str(COVID_CSV))
    want = RAW_SHA256[COVID_CSV]
    if digest != want:
        sys.exit(f"CHECKSUM MISMATCH for {COVID_CSV}\n  expected {want}\n  got      {digest}\n"
                 f"The published source has changed, or the download is corrupt. Nothing was built.")
    print(f"raw              {COVID_CSV}  {COVID_CSV.stat().st_size:,} bytes\n"
          f"                 sha256 {digest}  VERIFIED")
    return digest


def main(refresh: bool = False) -> None:
    fetch(refresh)
    dt = ts.load_covid(str(COVID_CSV), str(ADJ), str(COVID_MATRIX), gadm_dir=GADM)
    m, A = dt.meta, dt.A_geo
    N, T = dt.M.shape

    print(f"\n{NAME}")
    print(f"  X={dt.X.shape}  A_geo={A.shape} ({m['A_geo_kind']})  M={dt.M.shape}")
    print(f"  calendar      {m['dates'][0].date()} -> {m['dates'][-1].date()}  ({T} weeks)")
    print(f"  features      {m['feature_names']}  core_idx={m['core_feature_idx']}")
    print(f"  split         {m['split_scheme']}  |  scaler {m['scaler_scope']}")
    print(f"  mask density  {dt.M.mean():.4f}  (NYT is dense; nothing is imputed)")

    # ---- the Omicron anchor pin. The matrices carry no dates of their own, so the calendar is an
    # assertion, not an input. US COVID cases peaked in mid-January 2022 by a very wide margin, a
    # single unambiguous global maximum -- the role the H1N1 week plays for us-regions. A drifted
    # anchor moves this and puts sin_doy/cos_doy out of phase.
    national = dt.raw.sum(0)
    peak = pd.DatetimeIndex(m["dates"])[int(national.argmax())]
    ratio = national.max() / np.median(national)
    assert pd.Timestamp("2022-01-01") <= peak <= pd.Timestamp("2022-01-31"), (
        f"national COVID peak lands on {peak.date()}, not January 2022 (Omicron) -- the date "
        f"anchor has drifted")
    print(f"  OMICRON PIN   national max -> {peak.date()} ({ratio:.1f}x the median week). Confirmed.")

    # ---- the graph is the shipped one, bit for bit (the same gate influenza_load.py applies)
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
    assert m["node_ids"][biggest] == "covid_us-states_1", \
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
    print(f"  .check()      PASSED")
    print(f"\nnothing written: `python build_datasets.py` builds and writes this bundle.")
    # Corrected against covid_eda.md: the train->test level shift is 1.03x, i.e. essentially none.
    # The problem is inside VAL, which owns the Omicron peak (5.14M, 6.3x the largest test week) and
    # runs at 2.8x both other folds. Early stopping AND the val-fitted bias correction select on it.
    print("NOTE             the Omicron peak (2022-01-15) falls in the VAL fold: val runs at 2.8x "
          "train and 2.8x test,\n                 and its peak is 6.3x the largest test week "
          "(train->test level ratio is only 1.03x).\n                 Early stopping and the "
          "val-fitted bias_c are both compromised for THIS bundle. See covid_eda.md §4.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download us-states.csv")
    main(**vars(ap.parse_args()))

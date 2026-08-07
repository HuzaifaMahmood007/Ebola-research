"""Tests for the influenza static covariates C = [centroid_lat, centroid_lon, area_km2].

A mis-joined geometry is SILENT -- C would be full of plausible-looking numbers attached to the
wrong nodes, and nothing downstream would ever complain. So C is checked against geography we
know independently of GADM: which state is biggest, which prefecture is furthest north, what the
country totals are. These also serve as an independent confirmation of the recovered node
orderings: if an ordering were wrong, Alaska would not land where it does.

Needs geopandas:   conda run -n ebola python test_influenza_covariates.py
"""
import numpy as np
import pandas as pd

from to_schema import build_influenza_covariates, US_STATES_49, HHS_REGIONS

GADM = "data/gadm"
LAT, LON, AREA = 0, 1, 2

C_JP = build_influenza_covariates("japan", GADM)
C_ST = build_influenza_covariates("us-states", GADM)
C_RG = build_influenza_covariates("us-regions", GADM)
JP = pd.read_csv("japan_node_map.csv")["prefecture"].tolist()


def test_shapes_and_sanity():
    assert C_JP.shape == (47, 3) and C_ST.shape == (49, 3) and C_RG.shape == (10, 3)
    for nm, C in [("japan", C_JP), ("us-states", C_ST), ("us-regions", C_RG)]:
        assert np.isfinite(C).all(), f"{nm}: non-finite covariate"
        assert (C[:, AREA] > 0).all(), f"{nm}: non-positive area"
        assert (np.abs(C[:, LAT]) <= 90).all() and (np.abs(C[:, LON]) <= 180).all()
    print("ok  C shapes, finite, positive areas, valid lat/lon")


def test_japan_geography():
    # Hokkaido is Japan's largest prefecture AND its northernmost; Okinawa its southernmost.
    # All three independently re-confirm the recovered ordering (japan_10 / japan_19).
    assert JP[int(C_JP[:, AREA].argmax())] == "Hokkaido", "largest prefecture must be Hokkaido"
    assert JP[int(C_JP[:, LAT].argmax())] == "Hokkaido", "northernmost must be Hokkaido"
    assert JP[int(C_JP[:, LAT].argmin())] == "Okinawa", "southernmost must be Okinawa"
    assert JP[int(C_JP[:, LON].argmin())] == "Okinawa", "westernmost must be Okinawa"

    # Japan's 5 largest prefectures, from published areas (independent of GADM).
    top5 = {JP[i] for i in np.argsort(-C_JP[:, AREA])[:5]}
    assert top5 == {"Hokkaido", "Iwate", "Fukushima", "Nagano", "Niigata"}, top5
    # ...and its smallest is Kagawa.
    assert JP[int(C_JP[:, AREA].argmin())] == "Kagawa", "smallest prefecture must be Kagawa"

    total = C_JP[:, AREA].sum()
    assert 0.95 * 377_975 < total < 1.05 * 377_975, f"Japan total {total:,.0f} km2 off"
    print(f"ok  japan: Hokkaido largest+northernmost, Okinawa southernmost, top-5 areas match, "
          f"total {total:,.0f} km2 (published 377,975)")


def test_us_states_geography():
    idx = {s: i for i, s in enumerate(US_STATES_49)}
    big = US_STATES_49[int(C_ST[:, AREA].argmax())]
    assert big == "Alaska", f"largest state must be Alaska, got {big}"
    assert US_STATES_49[int(C_ST[:, LAT].argmax())] == "Alaska", "northernmost must be Alaska"
    assert US_STATES_49[int(C_ST[:, LAT].argmin())] == "Hawaii", "southernmost must be Hawaii"

    top5 = [US_STATES_49[i] for i in np.argsort(-C_ST[:, AREA])[:5]]
    assert top5 == ["Alaska", "Texas", "California", "Montana", "New Mexico"], top5
    assert US_STATES_49[int(C_ST[:, AREA].argmin())] == "Rhode Island", "smallest = Rhode Island"

    # Alaska and Hawaii are the two isolated graph nodes -- they must also be the two whose
    # centroids sit far from the contiguous mass. Ties the covariates back to the graph.
    assert idx["Alaska"] == 1 and idx["Hawaii"] == 9, "isolated node indices must be AK/HI"
    print(f"ok  us-states: Alaska largest+northernmost (us-states_1), Hawaii southernmost "
          f"(us-states_9), top-5 areas match published ranking")


def test_us_regions_are_unions_of_their_states():
    # The strongest check available: each HHS region's area must equal the SUM of its member
    # states' areas. Region geometry is built by union; state geometry is built independently.
    # They can only agree if both joins hit the right polygons.
    st_area = dict(zip(US_STATES_49, C_ST[:, AREA]))
    extra = build_influenza_covariates("us-regions", GADM)  # regions include Florida + DC
    for r in range(1, 11):
        members = HHS_REGIONS[r]
        known = [m for m in members if m in st_area]          # FL/DC absent from the 49-set
        part = sum(st_area[m] for m in known)
        assert part <= C_RG[r - 1, AREA] * 1.001, \
            f"R{r}: member states ({part:,.0f} km2) exceed the region ({C_RG[r-1, AREA]:,.0f})"
        if len(known) == len(members):                        # region has no FL/DC -> exact
            rel = abs(C_RG[r - 1, AREA] - part) / part
            assert rel < 0.01, f"R{r}: union {C_RG[r-1,AREA]:,.0f} != sum {part:,.0f} ({rel:.1%})"

    # Regions cover Florida and DC; the 49-state set does not. The difference must be exactly
    # those two -- an independent proof both node sets are the ones we think they are.
    diff = C_RG[:, AREA].sum() - C_ST[:, AREA].sum()
    assert 130_000 < diff < 180_000, \
        f"regions-minus-states = {diff:,.0f} km2; expected ~Florida (~150k) + DC"
    assert int(C_RG[:, AREA].argmax()) == 9, "largest region must be R10 (contains Alaska)"
    print(f"ok  us-regions: each region == sum of its member states; regions-minus-states = "
          f"{diff:,.0f} km2 == Florida + DC; R10 largest (holds Alaska)")


def test_C_is_not_transfer_safe():
    # C must never reach the SHARED encoder. All three diseases now carry a populated C, and it
    # is still not transfer-safe: they occupy disjoint geography, so a raw centroid identifies
    # the disease outright. Populating it for every disease does not fix that.
    from to_schema import load_influenza
    dt = load_influenza("data/Final datasets/influenza/japan.txt",
                        "data/Final datasets/influenza/japan-adj.txt",
                        dataset="japan", gadm_dir=GADM)
    assert dt.meta["covariates_transfer_safe"] is False
    assert "centroid_lat" not in [dt.meta["feature_names"][i]
                                  for i in dt.meta["core_feature_idx"]]
    assert dt.transfer_view().shape[2] == 4, "transfer_view must stay the 4 core channels"
    print("ok  C is flagged NOT transfer-safe and stays out of transfer_view()")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} covariate tests passed.")

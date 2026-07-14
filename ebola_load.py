"""Build the Ebola few-shot bundle from the production file and assert its properties.

Run ebola_audit.py first: confirming that the file carries usable signal is the precondition
for everything here.

Everything this prints, it asserts. A silent mis-join or a leaked scaler would produce
plausible-looking numbers and nothing downstream would complain.

Run:  conda run -n ebola python ebola_load.py     (needs openpyxl + geopandas + libpysal)
"""
from __future__ import annotations

import numpy as np

import to_schema as ts

XLSX = r"data/Final datasets/data-ebola-public.xlsx"
GADM = r"data/gadm"

# GADM 4.1 district counts, as an independent check that each country joined its own
# layer and not a neighbour's: GIN 34 prefectures (L2), LBR 15 counties (L1), SLE 14 (L2).
EXPECT_NODES = {"guinea": 32, "liberia": 15, "sierra leone": 14}

# The 2014 epidemic's defining feature is CROSS-BORDER spread through the
# Guéckédou-Lofa-Kailahun tri-border area. If the graph does not carry these three
# edges it has severed the single most important transmission pathway in the data.
TRIBORDER = [("guinea|gueckedou", "liberia|lofa"),
             ("guinea|gueckedou", "sierra leone|kailahun"),
             ("liberia|lofa", "sierra leone|kailahun")]


def main() -> None:
    dt = ts.load_ebola(XLSX, countries=ts.EBOLA_CORE_COUNTRIES,
                       few_shot_support_weeks=2, gadm_dir=GADM)
    m = dt.meta
    ids = m["node_ids"]
    N, T, F = dt.X.shape
    sup = m["split"]["support_mask"]
    qry = m["split"]["query_mask"]
    rep = m["adjacency_report"]

    print(f"source   {XLSX}")
    print(f"sha256   {ts._sha256(XLSX)}")
    print(f"\nDiseaseTensors  X={dt.X.shape}  A_geo={dt.A_geo.shape}  M={dt.M.shape}")
    print(f"features        {m['feature_names']}  core_idx={m['core_feature_idx']}")
    print(f"transfer_view   {dt.transfer_view().shape}")
    print(f"calendar        {m['dates'][0].date()} -> {m['dates'][-1].date()}  ({T} weeks)")
    print(f"role/split      {m['role']} / {m['split_scheme']}")
    print(f"scaler          {m['scaler']['transform']} / {m['scaler_scope']}")
    print(f"mask density    {dt.M.mean():.4f}  ({int(dt.M.sum()):,} observed of {N*T:,})")

    # ---------------------------------------------------------------- nodes
    by_country = {}
    for n in ids:
        by_country.setdefault(n.split("|", 1)[0], []).append(n)
    print(f"\nNODES  {N}")
    for c in sorted(by_country):
        print(f"  {c:<16} {len(by_country[c]):>3}")
    for c, want in EXPECT_NODES.items():
        got = len(by_country.get(c, []))
        assert got == want, f"{c}: expected {want} districts, got {got}"

    print(f"\nDROPPED (declared, with reasons)  {len(m['nodes_dropped'])}")
    for n, why in sorted(m["nodes_dropped"].items()):
        print(f"  {n:<52} {why}")

    print(f"\nLOAD-TIME ALIASES (data repair)  {len(m['name_aliases_applied'])}"
          f"   incl. the Port -> Port Loko merge")
    print(f"GADM NAME FIXES (join only, GADM's defects stay out of our node ids):")
    for ours, theirs in sorted(rep["gadm_name_fixes"].items()):
        print(f"    {ours:<34} -> GADM {theirs.split('|', 1)[1]!r}")

    # ---------------------------------------------------------------- graph
    print(f"\nA_geo  {m['A_geo_kind']}   {rep['n_edges']} undirected edges   "
          f"{rep['n_isolated']} isolated   degree mean {dt.A_geo.sum(1).mean():.2f}")
    print(f"CROSS-BORDER edges: {rep['n_cross_border']}   (block_diagonal="
          f"{m['graph_is_block_diagonal']})")
    for a, b in rep["cross_border_edges"]:
        star = "  <-- tri-border" if (a, b) in TRIBORDER or (b, a) in TRIBORDER else ""
        print(f"    {a:<28} -- {b}{star}")

    assert np.allclose(dt.A_geo, dt.A_geo.T), "A must be symmetric for a GNN"
    assert rep["n_isolated"] == 0, "k-NN fallback must leave no district degenerate"
    assert np.all(np.diag(dt.A_geo) == 0), "self-loops are the encoder's job (A_hat=A+I)"
    idx = {n: i for i, n in enumerate(ids)}
    for a, b in TRIBORDER:
        assert dt.A_geo[idx[a], idx[b]] == 1.0, \
            f"MISSING tri-border edge {a} -- {b}: the graph has severed the outbreak's " \
            f"main transmission pathway"
    assert rep["n_cross_border"] >= len(TRIBORDER)

    # ---------------------------------------------------------------- few-shot + leakage
    print(f"\nFEW-SHOT  support={int(sup.sum())} cells   query={int(qry.sum())} cells")
    print(f"  support weeks/district: {sup.sum(1).min()}-{sup.sum(1).max()}")

    # Disclosures — real properties of the data that evaluation must not stumble over.
    raw_inc = ts.invert_scaler(dt.y, m["scaler"]) * dt.M     # observed incidence only
    zero = [n for i, n in enumerate(ids) if raw_inc[i].sum() < 0.5]
    print(f"  nodes with NO query cell (graph-only, never scored): "
          f"{m['nodes_without_query']}")
    print(f"  zero-incidence nodes (reported, but never a single case): {zero}")
    print(f"     -> trivially predictable (emit 0). Kept: they are real districts and")
    print(f"        valid message-passing neighbours. Their scaler cannot degenerate —")
    print(f"        Ebola's is pooled per-disease, not per-node.")

    assert (sup & qry).sum() == 0, "support and query must be disjoint"
    assert (qry <= dt.M).all(), "query must be a subset of observed"
    assert (sup <= dt.M).all(), "support must be a subset of observed"
    assert m["ebola_first_week_masked"] is True
    assert m["scaler_scope"] == "per_disease_support"
    # pooled per-disease scaler: one (mean,std) broadcast across every node
    assert np.allclose(m["scaler"]["mean"], m["scaler"]["mean"][0])
    # and it is fit on the support cells ALONE
    raw = ts.invert_scaler(dt.y, m["scaler"])
    fit = sup.astype(bool)
    assert np.isclose(m["scaler"]["mean"][0], np.log1p(raw[fit]).mean(), atol=1e-4), \
        "the scaler must be the pooled log1p mean over support cells only"

    # deaths exists for Ebola alone, so it must stay out of the shared encoder
    assert dt.transfer_view().shape[-1] == 4 and F == 5
    assert "deaths_norm" in m["feature_names"] and "deaths_norm" not in ts.CORE_FEATURES

    # ---------------------------------------------------------------- static covariates
    # A mis-joined geometry is SILENT: C would hold plausible numbers sitting on the wrong
    # nodes and nothing downstream would complain. Check it against West-African geography
    # known independently of GADM.
    LAT, LON, AREA = 0, 1, 2
    C = dt.C
    print(f"\nC  {C.shape}  {m['covariates']}   {m['covariates_status']}")
    assert C.shape == (N, 3) and not np.allclose(C, 0), "C must be populated"

    area = {n: float(C[i, AREA]) for i, n in enumerate(ids)}
    for c in sorted(by_country):
        tot = sum(area[n] for n in by_country[c])
        print(f"  {c:<16} {len(by_country[c]):>3} districts   {tot:>9,.0f} km2")

    # every centroid must land in West Africa (4-13 N, 16-7 W)
    assert C[:, LAT].min() > 4 and C[:, LAT].max() < 13, "a centroid left West Africa"
    assert C[:, LON].min() > -16 and C[:, LON].max() < -7, "a centroid left West Africa"

    # Published national areas. LBR and SLE are COMPLETE here (15/15 counties, 14/14
    # districts), so their sums must land on the published figure — a strong check, since
    # the units were joined individually and can only add up if every one of them is right.
    # Compare against LAND area, not total: Liberia's total (111,369 km2) includes ~15,000
    # km2 of water, and GADM ships land polygons. Guinea has 32 of its 34 prefectures
    # (Koubia and Mandiana never reported), so it must come in UNDER the country total.
    lbr = sum(area[n] for n in by_country["liberia"])
    sle = sum(area[n] for n in by_country["sierra leone"])
    gin = sum(area[n] for n in by_country["guinea"])
    assert abs(lbr - 96_320) / 96_320 < 0.05, f"Liberia {lbr:,.0f} != ~96,320 km2 (land)"
    assert abs(sle - 71_620) / 71_620 < 0.05, f"Sierra Leone {sle:,.0f} != ~71,620 km2 (land)"
    assert gin < 245_857, f"Guinea {gin:,.0f} exceeds the whole country (32 of 34 units)"

    def _big(c, f=max):
        return f(by_country[c], key=lambda n: area[n])

    print(f"  largest  liberia={_big('liberia')}  sierra leone={_big('sierra leone')}")
    print(f"  smallest liberia={_big('liberia', min)}  "
          f"sierra leone={_big('sierra leone', min)}")
    assert _big("liberia") == "liberia|nimba", "Liberia's largest county is Nimba"
    assert _big("liberia", min) == "liberia|montserrado", "its smallest is Montserrado"
    assert _big("sierra leone") == "sierra leone|koinadugu", "SLE's largest is Koinadugu"
    assert _big("sierra leone", min) == "sierra leone|western area urban", \
        "SLE's smallest is Western Area Urban (the Freetown peninsula)"

    # N/S extremes: Guinea is the northern country, Liberia the southern one
    north, south = ids[int(C[:, LAT].argmax())], ids[int(C[:, LAT].argmin())]
    print(f"  northernmost={north}   southernmost={south}")
    assert north.startswith("guinea"), "the northernmost district must be Guinean"
    assert south.startswith("liberia"), "the southernmost district must be Liberian"

    # C and A_geo must agree: adjacent districts have nearby centroids. This is the check
    # that catches a scrambled join — it cross-validates C against the graph.
    e = np.array([[np.hypot(*(C[i, :2] - C[j, :2]))]
                  for i, j in zip(*np.where(np.triu(dt.A_geo, 1) > 0))])
    print(f"  centroid distance across the {len(e)} edges: mean {e.mean():.2f}deg  "
          f"max {e.max():.2f}deg")
    assert e.max() < 3.0, "an 'adjacent' pair is far apart -> C is joined to the wrong nodes"

    assert m["covariates_transfer_safe"] is False, \
        "C is NOT transfer-safe: the diseases occupy disjoint geography, so a raw " \
        "centroid identifies the disease outright. Single-disease use only."

    dt.check()
    print("\n.check() PASSED — Ebola bundle is schema-valid, leakage-safe, C is joined to")
    print("the right districts, and the tri-border transmission edges survive in the graph.")


if __name__ == "__main__":
    main()

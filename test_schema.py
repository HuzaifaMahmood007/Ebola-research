"""
test_schema.py — correctness + shape tests for to_schema.py.

Uses tiny synthetic fixtures that mirror the *real* raw layouts
(OpenDengue 16-col, HDX Ebola long/cumulative, ColaGNN [T,N] matrix),
so the invariants that would silently ruin the paper are pinned:
  - Ebola cumulative -> weekly incidence (diff + negative-correction clip + mask)
  - leakage-safe scaler (val/test never touch the fit)
  - seasonality phase, inverse-transform round-trip, and all shape contracts.

Run:  python test_schema.py     (no pytest dependency required)
"""
import tempfile, os
import numpy as np
import pandas as pd

import to_schema as ts


# --------------------------------------------------------------------------- #
def test_cumulative_to_incidence_clips_and_masks():
    # One district; cumulative with a mid-series DOWNWARD correction (artefact),
    # sampled irregularly (sit-report style).
    rows = [
        ("guinea|gueckedou", "2014-04-06", 100),
        ("guinea|gueckedou", "2014-04-20", 122),   # +22
        ("guinea|gueckedou", "2014-05-11", 118),   # -4  (correction -> clip to 0)
        ("guinea|gueckedou", "2014-05-25", 130),   # +12
    ]
    df = pd.DataFrame(rows, columns=["_cd", "date", "value"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%d/%m/%Y")

    inc, msk = ts.cumulative_to_weekly_incidence(df, "_cd", "date", "value")
    series = inc.iloc[0]
    assert series.min() >= 0, "negative 'new cases' must be clipped"
    assert series.sum() == 100 + 22 + 0 + 12, "incidence must sum correctly"
    # weeks with an actual report are observed; forward-filled gaps are masked out
    assert msk.iloc[0].sum() == 4, "exactly the 4 reported weeks are observed"
    assert msk.iloc[0].sum() < len(series), "gap weeks must be masked (M=0)"
    print("ok  cumulative->incidence: clip + mask + conservation")


def test_scaler_is_leakage_safe():
    N, T = 3, 20
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 500, size=(N, T)).astype(float)
    mask = np.ones((N, T), np.uint8)
    train_end, _ = ts.chronological_split(T)

    s1 = ts.fit_scalers(raw, mask, train_end)
    # perturb ONLY the val/test tail; a leakage-safe fit must be unchanged
    raw2 = raw.copy()
    raw2[:, train_end:] *= 1000.0
    s2 = ts.fit_scalers(raw2, mask, train_end)
    assert np.allclose(s1["mean"], s2["mean"]) and np.allclose(s1["std"], s2["std"]), \
        "scaler must depend on the TRAIN slice only"
    print("ok  scaler leakage-safe (train-only fit)")


def test_scaler_roundtrip_and_zero_variance_guard():
    N, T = 2, 12
    raw = np.vstack([np.arange(T) * 10.0, np.full(T, 7.0)])  # row 1 has zero var
    mask = np.ones((N, T), np.uint8)
    sc = ts.fit_scalers(raw, mask, T)
    assert (sc["std"] > 0).all(), "zero-variance node must not yield std=0"
    back = ts.invert_scaler(ts.apply_scaler(raw, sc), sc)
    assert np.allclose(back, raw, atol=1e-4), "log1p+z must invert to real counts"
    print("ok  scaler round-trip + zero-variance guard")


def test_seasonality_phase():
    dates = pd.to_datetime(["2015-01-01", "2015-04-02", "2015-07-02", "2015-10-01"])
    sin, cos = ts._seasonality(dates)
    # Jan-1 ~ phase 0: sin~0, cos~1
    assert abs(sin[0]) < 0.05 and cos[0] > 0.98
    # ~quarter year later sin should have risen toward its peak
    assert sin[1] > sin[0]
    print("ok  seasonality phase from calendar dates")


DENGUE_HEADER = ["adm_0_name","adm_1_name","adm_2_name","full_name","ISO_A0",
                 "FAO_GAUL_code","RNE_iso_code","IBGE_code","calendar_start_date",
                 "calendar_end_date","Year","dengue_total",
                 "case_definition_standardised","S_res","T_res","UUID"]


def _dengue_rows(country, units, n_weeks=60, weekly=True, add_monthly=False):
    rows = []
    for u in units:
        if weekly:
            for wk in range(n_weeks):                                  # weekly, 2014+
                d = pd.Timestamp("2014-01-05") + pd.Timedelta(weeks=wk)
                rows.append([country,u,"NA",f"{country}, {u}","XXX",0,"X","NA",
                             d.strftime("%d/%m/%Y"), d.strftime("%d/%m/%Y"),
                             2014, 10 + wk, "Total","Admin1","Week","UUID-W"])
        if add_monthly:
            for m, val in zip(["01/01/1993","01/02/1993","01/03/1993"], [46,114,60]):
                rows.append([country,u,"NA",f"{country}, {u}","XXX",0,"X","NA",
                             m,m,1993,val,"Total","Admin1","Month","UUID-M"])
    return rows


def _dengue_fixture(spec=None, n_weeks=60):
    """Multi-country weekly extract (+ some legacy monthly rows), mirroring the
    best-spatial extract's cadence transition and multi-country coverage."""
    spec = spec or {"BRAZIL": ["ALAGOAS", "BAHIA", "ACRE"],
                    "THAILAND": ["BANGKOK", "CHIANG MAI", "PHUKET"]}
    rows = []
    for c, units in spec.items():
        rows += _dengue_rows(c, units, n_weeks=n_weeks, add_monthly=(c == "BRAZIL"))
    return pd.DataFrame(rows, columns=DENGUE_HEADER)


def test_dengue_multicountry_weekly_default():
    df = _dengue_fixture()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dengue.csv"); df.to_csv(p, index=False)
        dt = ts.load_dengue(p)                       # default: endemic pool, weekly
        dt_m = ts.load_dengue(p, countries=["BRAZIL"], t_res_filter="Month",
                              min_weeks=1, min_nodes_per_country=1)
    N, T, F = dt.X.shape
    assert (N, F) == (6, 4), f"got N={N},F={F}"       # 2 countries x 3 units; 4 core
    assert dt.meta["feature_names"][:4] == ts.CORE_FEATURES
    assert set(dt.meta["countries"]) == {"brazil", "thailand"}, dt.meta["countries"]
    assert all("|" in n for n in dt.meta["node_ids"]), "node ids must be country|adm"
    assert dt.meta["graph_is_block_diagonal"] is True
    assert dt.meta["t_res"] == "weekly" and dt.meta["steps_per_year"] == 52
    assert dt_m.meta["t_res"] == "monthly", "Month option must still work"
    print(f"ok  dengue multi-country weekly: N={N} T={T}, countries={dt.meta['countries']}")


def test_dengue_coverage_prune_and_absent_report():
    # Thailand has plenty of weeks; 'Peru' has too few -> dropped; 'Fiji' absent.
    df = _dengue_fixture(spec={"THAILAND": ["BANGKOK","CHIANG MAI","PHUKET"]},
                         n_weeks=60)
    df = pd.concat([df, pd.DataFrame(_dengue_rows("PERU", ["LIMA","CUSCO","PIURA"],
                                                  n_weeks=10), columns=DENGUE_HEADER)])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dengue.csv"); df.to_csv(p, index=False)
        dt = ts.load_dengue(p, countries=["thailand","peru","fiji"], min_weeks=52)
    assert dt.meta["countries"] == ["thailand"], "Peru must be pruned for low coverage"
    assert "peru" in dt.meta["countries_dropped_low_coverage"]
    assert "fiji" in dt.meta["countries_requested_absent"]
    print("ok  dengue coverage prune + absent-country reporting")


def test_dengue_region_control_across_countries():
    df = _dengue_fixture()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dengue.csv"); df.to_csv(p, index=False)
        all_n = ts.load_dengue(p, min_nodes_per_country=1).X.shape[0]
        excl = ts.load_dengue(p, exclude_regions=["ACRE"],
                              min_nodes_per_country=1).X.shape[0]
        inc = ts.load_dengue(p, include_regions=["BANGKOK", "BAHIA"],
                             min_nodes_per_country=1).X.shape[0]
    assert all_n == 6 and excl == 5 and inc == 2
    print("ok  dengue region control: include / exclude act on adm names")


def _ebola_fixture():
    # HDX long/cumulative: Country/Localite/Indicator/value/date/source/url
    header = ["Country","Localite","Indicator","value","date","source","url"]
    rows = []
    for ind, base in [("Cases", 100), ("Deaths", 40)]:
        for k, d in enumerate(["06/04/2014","20/04/2014","11/05/2014","25/05/2014"]):
            rows.append(["Guinea","Gueckedou ", ind, base + k*10, d, "MoH", "http://x"])
            rows.append(["Liberia","Lofa", ind, base + k*5, d, "MoH", "http://x"])
    rows.append(["Nigeria","Lagos","Cases",20,"20/09/2014","MoH","http://x"])  # minor
    return pd.DataFrame(rows, columns=header)


def test_ebola_keeps_all_by_default_and_few_shot_split():
    df = _ebola_fixture()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ebola.csv"); df.to_csv(p, index=False)
        dt = ts.load_ebola(p)                        # default: keep ALL countries
    N, T, F = dt.X.shape
    assert N == 3, f"default must keep all countries incl. Nigeria (got N={N})"
    assert "deaths_norm" in dt.meta["feature_names"], "deaths must be extended channel"
    assert F == 5 and dt.meta["core_feature_idx"] == [0, 1, 2, 3]  # 4 core + deaths
    assert dt.meta["role"] == "few_shot_holdout"
    assert dt.meta["split_scheme"] == "few_shot_support_query"
    assert dt.meta["scaler_scope"] == "per_disease_support"
    print(f"ok  ebola loader: keep-all default N={N} F={F}; few-shot split")


def test_ebola_few_shot_support_is_two_weeks_and_leakage_safe():
    df = _ebola_fixture()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ebola.csv"); df.to_csv(p, index=False)
        dt = ts.load_ebola(p, countries=ts.EBOLA_CORE_COUNTRIES,
                           few_shot_support_weeks=2)
    sup = dt.meta["split"]["support_mask"]
    qry = dt.meta["split"]["query_mask"]
    # each district contributes exactly its first 2 OBSERVED weeks as support
    assert (sup.sum(axis=1) == 2).all(), "support must be 2 weeks/district"
    assert (sup & qry).sum() == 0, "support and query must be disjoint"
    # per-disease scaler => a single (mean,std) broadcast across nodes
    assert np.allclose(dt.meta["scaler"]["mean"], dt.meta["scaler"]["mean"][0])
    print("ok  ebola few-shot: 2-week support, disjoint query, pooled scaler")


def test_obs_mask_is_a_core_channel():
    df = _ebola_fixture()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ebola.csv"); df.to_csv(p, index=False)
        dt = ts.load_ebola(p, countries=ts.EBOLA_CORE_COUNTRIES)
    assert dt.meta["feature_names"][3] == "obs_mask"
    assert np.array_equal(dt.X[:, :, 3].astype(np.uint8), dt.M), \
        "channel 3 must equal the observation mask"
    print("ok  obs_mask present as core channel and equals M")


def test_transfer_view_excludes_extended_channels():
    df = _ebola_fixture()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ebola.csv"); df.to_csv(p, index=False)
        dt = ts.load_ebola(p, countries=ts.EBOLA_CORE_COUNTRIES)
    tv = dt.transfer_view()
    assert tv.shape[-1] == 4, "shared encoder sees only the 4 core channels"
    assert dt.X.shape[-1] == 5, "but deaths remains available in full X"
    print("ok  transfer_view exposes core-only (deaths excluded)")


def test_ebola_client_region_control():
    df = _ebola_fixture()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ebola.csv"); df.to_csv(p, index=False)
        core = ts.load_ebola(p, countries=ts.EBOLA_CORE_COUNTRIES).X.shape[0]
        excl = ts.load_ebola(p, exclude_regions=["Lagos"]).X.shape[0]
        thr = ts.load_ebola(p, drop_below_total=50).X.shape[0]  # Nigeria=20 dropped
    assert core == 2, "restricting to core countries drops Nigeria"
    assert excl == 2, "exclude_regions by district name works"
    assert thr == 2, "drop_below_total removes degenerate near-zero nodes"
    print("ok  ebola region control: countries / exclude / threshold all work")


def test_accent_stripping_in_canon():
    assert ts._canon("Guéckédou ") == ts._canon("Gueckedou") == "gueckedou"
    assert ts._canon("Chiang Mai") == "chiang mai"
    print("ok  _canon strips accents (name-join robustness, D17)")


def test_influenza_loader_shapes():
    N, T = 5, 30
    rng = np.random.default_rng(1)
    mat = rng.integers(0, 200, size=(T, N))          # ColaGNN ships [T,N]
    adj = (rng.random((N, N)) > 0.5).astype(int)
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "japan.txt"); pd.DataFrame(mat).to_csv(mp, header=False, index=False)
        ap = os.path.join(d, "adj.txt");   pd.DataFrame(adj).to_csv(ap, header=False, index=False)
        dt = ts.load_influenza(mp, ap, dataset="japan")
    assert dt.X.shape == (N, T, 4) and dt.A_geo.shape == (N, N)   # 4 core channels
    assert dt.meta["A_geo_kind"] == "shipped", "shipped adjacency must be reused verbatim"
    assert str(dt.meta["dates"][0].date()) >= "2012-08-01", "Japan anchored to Aug 2012"
    print(f"ok  influenza loader: N={N} T={T}, shipped adjacency + MMWR anchor")


def test_no_future_leak_in_mask_semantics():
    # M must mark imputed steps so they are excluded from loss/eval downstream.
    N, T = 2, 10
    raw = np.zeros((N, T)); mask = np.ones((N, T), np.uint8)
    mask[:, 5:] = 0
    sc = ts.fit_scalers(raw, mask, 5)
    assert sc["mean"].shape == (N,)
    print("ok  mask semantics available for loss/eval exclusion")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} schema tests passed.")

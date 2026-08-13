"""
test_schema.py — correctness + shape tests for to_schema.py.

Uses tiny synthetic fixtures that mirror the *real* raw layouts
(OpenDengue 16-col, HDX Ebola long/cumulative, ColaGNN [T,N] matrix),
so the invariants that would silently ruin the paper are pinned:
  - Ebola cumulative -> weekly incidence (running-maximum envelope + mask; mass-conserving)
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
    # sampled irregularly (sit-report style). Dates are ISO — the real sheet stores
    # real datetimes, and `dayfirst` silently corrupted them (same bug as dengue).
    rows = [
        ("guinea|gueckedou", "2014-04-06", 100),   # FIRST report: C_0, increment UNDEFINED
        ("guinea|gueckedou", "2014-04-20", 122),   # +22
        ("guinea|gueckedou", "2014-05-11", 118),   # -4  (correction -> clip to 0)
        ("guinea|gueckedou", "2014-05-25", 130),   # +12
    ]
    df = pd.DataFrame(rows, columns=["_cd", "date", "value"])

    inc, msk = ts.cumulative_to_weekly_incidence(df, "_cd", "date", "value")
    series = inc.iloc[0]
    assert series.min() >= 0, "incidence must never be negative"
    # The first reported week yields NO incidence: with no C_-1 the increment is not
    # identifiable, and assigning C_0 (=100) would fabricate the whole back-log.
    #
    # The downward step 122 -> 118 is a data-entry dropout, not a fall in a cumulative count, so the
    # transform differences the RUNNING MAXIMUM: the 118 is lifted back to 122 and the next report
    # scores 130 - 122 = 8. This assertion previously expected 22 + 0 + 12 = 34, which is what
    # CLIPPING gives -- it zeroes the fall and then counts the recovery to 122 a second time. That is
    # the C1 defect that fabricated 8,786 cases across 34 districts, 35.8% of the Ebola target, and
    # the test went on asserting it after the transform was fixed.
    assert list(series[series > 0]) == [22.0, 8.0], f"expected 22 then 8, got {list(series)}"
    # THE property, and the one that would have caught C1 on its own: increments cannot invent mass.
    # Their sum is exactly C_last - C_first, whatever the reporting artefacts in between.
    assert series.sum() == 130 - 100, \
        f"mass not conserved: {series.sum()} != C_last - C_first = 30"
    assert msk.iloc[0].sum() == 2, "2 usable increments from 4 reports (week 0 and the dropout)"
    assert msk.iloc[0].sum() < len(series), "gap weeks must be masked (M=0)"
    print("ok  cumulative->incidence: running max, mass conserved, no fabricated week-0 onset")


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
                             d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"),
                             2014, 10 + wk, "Total","Admin1","Week","UUID-W"])
        if add_monthly:
            for m, val in zip(["1993-01-01","1993-02-01","1993-03-01"], [46,114,60]):
                rows.append([country,u,"NA",f"{country}, {u}","XXX",0,"X","NA",
                             m,m,1993,val,"Total","Admin1","Month","UUID-M"])
    return rows


def _dengue_rows_lvl(country, units, level, n_weeks=60, parent="PROV",
                     start="2014-01-05"):
    """Weekly rows at a specified S_res level. Admin1 -> unit in adm_1_name;
    Admin2 -> unit in adm_2_name (parent in adm_1_name); Admin0 -> both 'NA'."""
    rows = []
    for u in units:
        a1 = u if level == "Admin1" else (parent if level == "Admin2" else "NA")
        a2 = u if level == "Admin2" else "NA"
        for wk in range(n_weeks):
            d = pd.Timestamp(start) + pd.Timedelta(weeks=wk)
            rows.append([country, a1, a2, f"{country}, {u}", "XXX", 0, "X", "NA",
                         d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"),
                         d.year, 10 + wk, "Total", level, "Week", "UUID"])
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


# Weekly sit-report dates used by the Ebola fixture (each lands in its own W-SAT week).
_EB_DATES = ["2014-04-06", "2014-04-13", "2014-04-20", "2014-04-27",
             "2014-05-04", "2014-05-11"]


def _ebola_fixture():
    """Mirrors the REAL OCHA ROWCA sheet, not the loader's original guess:
    columns Country/Localite/Category/Value/Date (capitalised), ISO dates, an object
    `Value` column carrying junk, 8 Category variants, the `National` aggregate, a
    multi-district blob row, an Excel-epoch date, and Sierra Leone's `Port Loko`
    series torn across two labels. Every one of those is in the production file."""
    header = ["Country", "Localite", "Category", "Value", "Date", "Sources", "Link"]
    rows = []

    # cumulative CASES per district; gueckedou carries a downward revision (-5) at k=3
    series = {
        ("Guinea", "Gueckedou"):            [100, 110, 130, 125, 160, 175],
        ("Guinea", "Macenta"):              [50, 55, 60, 70, 80, 90],
        ("Liberia", "Lofa County"):         [20, 25, 30, 35, 40, 45],
    }
    for (country, loc), vals in series.items():
        for d, v in zip(_EB_DATES, vals):
            rows.append([country, loc, "Cases", v, d, "WHO", "http://x"])
            rows.append([country, loc, "Deaths", v // 2, d, "WHO", "http://x"])

    # Sierra Leone: ONE district reported under two labels in strictly disjoint eras.
    # 'Port Loko' stops after week 3; 'Port' picks the SAME cumulative up at week 4.
    for d, v in zip(_EB_DATES[:3], [10, 20, 30]):
        rows.append(["Sierra Leone", "Port Loko", "Cases", v, d, "WHO", "http://x"])
    for d, v in zip(_EB_DATES[3:], [40, 50, 60]):
        rows.append(["Sierra Leone", "Port", "Cases", v, d, "WHO", "http://x"])

    # --- rows that MUST be cleaned out ---
    for d in _EB_DATES:
        rows.append(["Guinea", "National", "Cases", 9999, d, "WHO", "x"])
    rows.append(["Guinea", "Guekedou, Macenta, Nzerekore and Kissidougou", "Cases", 86,
                 _EB_DATES[0], "WHO", "x"])              # declared multi-district blob
    rows.append(["Guinea", "Gueckedou", "Cases", 5, "1900-01-04", "WHO", "x"])  # Excel epoch
    rows.append(["Guinea", "Macenta", "Cases", " ", _EB_DATES[2], "WHO", "x"])  # junk Value
    rows.append(["Guinea", "Macenta", "Cases", "-", _EB_DATES[3], "WHO", "x"])  # junk Value

    # Sierra Leone non-districts: 'Freetown' is a CITY inside Western Area Urban, and
    # 'Western Area' is the PARENT of Western Area Urban+Rural and overlaps them in time.
    rows.append(["Sierra Leone", "Freetown", "Cases", 5, _EB_DATES[0], "WHO", "x"])
    for d, v in zip(_EB_DATES, [100, 200, 300, 400, 500, 600]):
        rows.append(["Sierra Leone", "Western Area", "Cases", v, d, "WHO", "x"])
    for d, v in zip(_EB_DATES, [60, 70, 80, 90, 100, 110]):
        rows.append(["Sierra Leone", "Western Area Urban", "Cases", v, d, "WHO", "x"])

    # --- Category variants that must NOT become the forecasting target ---
    for d in _EB_DATES:
        rows.append(["Guinea", "Gueckedou", "New cases", 7, d, "WHO", "x"])
        rows.append(["Guinea", "Gueckedou", "New Cases", 7, d, "WHO", "x"])
        rows.append(["Guinea", "Gueckedou", "Confirmed cases", 3, d, "WHO", "x"])
        rows.append(["Guinea", "Gueckedou", "Suspected Cases", 2, d, "WHO", "x"])

    rows.append(["Nigeria", "Lagos", "Cases", 20, "2014-09-20", "WHO", "x"])   # minor country
    return pd.DataFrame(rows, columns=header)


def _ebola_csv(d, df=None):
    p = os.path.join(d, "ebola.csv")
    (df if df is not None else _ebola_fixture()).to_csv(p, index=False)
    return p


def test_ebola_reads_the_real_column_layout_and_targets_only_headline_Cases():
    # 'Cases' is the target (cumulative). 'New cases'/'Confirmed'/'Suspected' must NOT
    # be summed into it, and the New cases/New Cases casing pair must not double-count.
    with tempfile.TemporaryDirectory() as d:
        dt = ts.load_ebola(_ebola_csv(d), countries=ts.EBOLA_CORE_COUNTRIES)
    i = dt.meta["node_ids"].index("guinea|gueckedou")
    # gueckedou cumulative 100,110,130,125,160,175. Week 0 is masked (unidentifiable back-log) and
    # the 125 dropout is masked, so observed incidence is 10+20+30+15 = 75 = C_last - C_first.
    # Summing New/Confirmed/Suspected into the target would push this above 75; so would the old
    # clipping transform, which reported 80.
    raw = ts.invert_scaler(dt.y, dt.meta["scaler"])[i]
    assert round(float((raw * dt.M[i]).sum())) == 175 - 100, \
        "target must be diff(Cases) only — New/Confirmed/Suspected must not leak in"
    assert "deaths_norm" in dt.meta["feature_names"], "Deaths is the extended channel"
    print("ok  ebola: real Category/Value/Date layout; only headline Cases is the target")


def test_ebola_drops_national_blobs_and_excel_epoch_dates():
    with tempfile.TemporaryDirectory() as d:
        dt = ts.load_ebola(_ebola_csv(d), countries=ts.EBOLA_CORE_COUNTRIES)
    ids = dt.meta["node_ids"]
    assert not any("national" in n for n in ids), "'National' is not a district"
    assert not any("," in n or " and " in n for n in ids), "blob rows are not nodes"
    # 'Freetown' is a city inside Western Area Urban; 'Western Area' is the parent of
    # Western Area Urban+Rural and overlaps them -> keeping it would double-count.
    assert "sierra leone|freetown" not in ids
    assert "sierra leone|western area" not in ids
    # the real district survives under its CORRECT name — GADM ships 'Western Urban',
    # but GADM's naming is absorbed at the join (EBOLA_GADM_FIX), not in our node ids.
    assert "sierra leone|western area urban" in ids
    assert all(pd.Timestamp(t).year >= 2014 for t in dt.meta["dates"]), \
        "Excel-epoch dates (year<2014) must be filtered out"
    # every removal is a decision someone wrote down; the builder drops nothing on its own
    assert dt.meta["nodes_dropped"], "declared drops must be recorded in meta"
    print("ok  ebola cleaning: National + blobs + Excel-epoch + non-district units removed")


def test_ebola_unlisted_blob_label_fails_loud():
    # A blob label we have NOT declared must halt the build, not silently become a node.
    df = _ebola_fixture()
    df.loc[len(df)] = ["Guinea", "Kindia and Coyah", "Cases", 12,
                       _EB_DATES[0], "WHO", "x"]
    with tempfile.TemporaryDirectory() as d:
        try:
            ts.load_ebola(_ebola_csv(d, df), countries=ts.EBOLA_CORE_COUNTRIES)
        except ValueError as e:
            assert "kindia and coyah" in str(e).lower()
            print("ok  ebola: an undeclared multi-district blob fails loud")
            return
    raise AssertionError("an undeclared blob label must raise, not be silently kept")


def test_ebola_first_observed_week_is_masked_not_a_backlog():
    # A district's FIRST report gives C_0 but no C_-1, so its incidence is UNDEFINED.
    # The old code set it to C_0, dumping the district's whole cumulative back-log into
    # week 0. On the production file that fabricated 4.1% of all incidence and handed
    # 10 of 64 districts their all-time maximum as a week-0 spike.
    with tempfile.TemporaryDirectory() as d:
        dt = ts.load_ebola(_ebola_csv(d), countries=ts.EBOLA_CORE_COUNTRIES)
    assert dt.meta["ebola_first_week_masked"] is True, "must be recorded in meta"
    i = dt.meta["node_ids"].index("guinea|gueckedou")
    # cumulative 100,110,130,125,160,175. The 125 is a data-entry dropout, not a fall in a
    # cumulative count, so the running-maximum transform (C1) lifts it back to 130 and MASKS that
    # week rather than scoring it as an observed zero. Increments: 10, 20, [masked], 30, 15.
    # These figures were 5 / 80 / 35 while the transform still CLIPPED, which zeroed the fall and
    # then counted the recovery to 130 again -- 5 cases out of 80 fabricated, in miniature exactly
    # the defect C1 removed from the production file.
    obs = ts.invert_scaler(dt.y, dt.meta["scaler"])[i][dt.M[i] == 1]
    assert int(dt.M[i].sum()) == 4, "6 reports, minus the unidentifiable week 0 and the dropout"
    assert round(float(obs.sum())) == 175 - 100, \
        f"mass not conserved: {round(float(obs.sum()))} != C_last - C_first = 75"
    assert round(float(obs.max())) == 30, "the recovery week is 160-130, not 160-125"
    print("ok  ebola: first observed week masked, dropout masked, mass conserved")


def test_ebola_support_is_real_increments_not_the_backlog():
    # If week 0 were observed it would BE support week #1 for EVERY district, so the
    # fabricated cumulative back-log would set the normalisation for the whole disease.
    with tempfile.TemporaryDirectory() as d:
        dt = ts.load_ebola(_ebola_csv(d), countries=ts.EBOLA_CORE_COUNTRIES)
    sup = dt.meta["split"]["support_mask"].astype(bool)
    raw = ts.invert_scaler(dt.y, dt.meta["scaler"])
    i = dt.meta["node_ids"].index("guinea|gueckedou")
    got = sorted(round(float(v)) for v in raw[i][sup[i]])
    assert got == [10, 20], f"support must be the first two REAL increments, got {got}"
    assert round(float(raw[sup].max())) < 50, \
        "no support cell may carry a district's cumulative back-log (gueckedou C_0=100)"
    # the pooled per-disease scaler is exactly the log1p mean over those support cells
    assert np.isclose(dt.meta["scaler"]["mean"][0], np.log1p(raw[sup]).mean(), atol=1e-4)
    print("ok  ebola: support = real increments; pooled scaler fit on them alone")


def test_ebola_scaler_depends_only_on_support_cells():
    # The Ebola scaler must depend on SUPPORT cells only.
    # NOTE the subtlety this test had to learn: you cannot perturb one week of a
    # CUMULATIVE series in isolation — inflating C_0 also changes the week-1 increment.
    # Perturbing a late (query) week is the clean probe: it moves only that increment.
    df = _ebola_fixture()
    with tempfile.TemporaryDirectory() as d:
        a = ts.load_ebola(_ebola_csv(d, df), countries=ts.EBOLA_CORE_COUNTRIES)
    big = df.copy()
    m = (big["Category"] == "Cases") & (big["Date"] == _EB_DATES[-1])
    big.loc[m, "Value"] = big.loc[m, "Value"].astype(float) * 1000    # query week only
    with tempfile.TemporaryDirectory() as d:
        b = ts.load_ebola(_ebola_csv(d, big), countries=ts.EBOLA_CORE_COUNTRIES)
    assert np.allclose(a.meta["scaler"]["mean"], b.meta["scaler"]["mean"]), \
        "a query-week value must never move the support scaler"
    assert np.allclose(a.meta["scaler"]["std"], b.meta["scaler"]["std"])
    print("ok  ebola: support scaler is immune to query-set values (no leakage)")


def test_ebola_torn_series_reunited_by_alias():
    # 'Port' is 'Port Loko' truncated mid-compilation: the labels are strictly disjoint
    # in time and the cumulative continues across the handover. Two half-series, each
    # with its own scaler and a fake back-log at the seam, is a data bug — not two nodes.
    with tempfile.TemporaryDirectory() as d:
        dt = ts.load_ebola(_ebola_csv(d), countries=ts.EBOLA_CORE_COUNTRIES)
    ids = dt.meta["node_ids"]
    assert "sierra leone|port" not in ids, "'Port' must merge into 'Port Loko'"
    assert "sierra leone|port loko" in ids
    i = ids.index("sierra leone|port loko")
    # 6 weekly reports, week 0 masked -> 5 observed, cumulative 10..60 -> 5 x 10 cases
    assert int(dt.M[i].sum()) == 5, "the reunited series must be continuous (5 observed)"
    raw = ts.invert_scaler(dt.y, dt.meta["scaler"])[i]
    assert round(float((raw * dt.M[i]).sum())) == 50, "merged incidence must be 10+10+10+10+10"
    print("ok  ebola: torn Port/Port Loko series reunited into one continuous node")


def test_ebola_nonnumeric_values_are_coerced_not_fatal():
    # The production sheet carries 7 junk Value cells (' ', '-'). They must be dropped
    # BEFORE the de-duplication that keeps the last row per (node, date) — otherwise a
    # junk cell silently overwrites that district's real cumulative count for the week.
    with tempfile.TemporaryDirectory() as d:
        dt = ts.load_ebola(_ebola_csv(d), countries=ts.EBOLA_CORE_COUNTRIES)
    dt.check()
    i = dt.meta["node_ids"].index("guinea|macenta")   # junk rows sit on its weeks 2 & 3
    obs = ts.invert_scaler(dt.y, dt.meta["scaler"])[i][dt.M[i] == 1]
    # cumulative 50,55,60,70,80,90 -> increments 5,5,10,10,10 (week 0 masked)
    assert int(dt.M[i].sum()) == 5 and round(float(obs.sum())) == 40, \
        "a junk Value must not displace the district's real cumulative count"
    print("ok  ebola: non-numeric Value cells coerced to missing, real counts intact")


def test_ebola_keeps_all_by_default_and_few_shot_split():
    with tempfile.TemporaryDirectory() as d:
        dt = ts.load_ebola(_ebola_csv(d))            # default: keep ALL countries
    N, T, F = dt.X.shape
    assert "nigeria|lagos" in dt.meta["node_ids"], "default must keep all countries"
    assert "deaths_norm" in dt.meta["feature_names"], "deaths must be extended channel"
    assert F == 5 and dt.meta["core_feature_idx"] == [0, 1, 2, 3]  # 4 core + deaths
    assert dt.meta["role"] == "few_shot_holdout"
    assert dt.meta["split_scheme"] == "few_shot_support_query"
    assert dt.meta["scaler_scope"] == "per_disease_support"
    print(f"ok  ebola loader: keep-all default N={N} F={F}; few-shot split")


def test_ebola_few_shot_support_is_two_weeks_and_leakage_safe():
    with tempfile.TemporaryDirectory() as d:
        dt = ts.load_ebola(_ebola_csv(d), countries=ts.EBOLA_CORE_COUNTRIES,
                           few_shot_support_weeks=2)
    sup = dt.meta["split"]["support_mask"]
    qry = dt.meta["split"]["query_mask"]
    # support = the first 2 OBSERVED weeks (i.e. the first 2 real increments — week 0
    # is masked, so support can never be the fabricated back-log).
    assert (sup.sum(axis=1) == 2).all(), "support must be 2 weeks/district"
    assert (sup & qry).sum() == 0, "support and query must be disjoint"
    assert (qry <= dt.M).all(), "query must be a subset of observed"
    # per-disease scaler => a single (mean,std) broadcast across nodes
    assert np.allclose(dt.meta["scaler"]["mean"], dt.meta["scaler"]["mean"][0])
    print("ok  ebola few-shot: 2-week support, disjoint query, pooled scaler")


def test_obs_mask_is_a_core_channel():
    with tempfile.TemporaryDirectory() as d:
        dt = ts.load_ebola(_ebola_csv(d), countries=ts.EBOLA_CORE_COUNTRIES)
    assert dt.meta["feature_names"][3] == "obs_mask"
    assert np.array_equal(dt.X[:, :, 3].astype(np.uint8), dt.M), \
        "channel 3 must equal the observation mask"
    print("ok  obs_mask present as core channel and equals M")


def test_transfer_view_excludes_extended_channels():
    with tempfile.TemporaryDirectory() as d:
        dt = ts.load_ebola(_ebola_csv(d), countries=ts.EBOLA_CORE_COUNTRIES)
    tv = dt.transfer_view()
    assert tv.shape[-1] == 4, "shared encoder sees only the 4 core channels"
    assert dt.X.shape[-1] == 5, "but deaths remains available in full X"
    print("ok  transfer_view exposes core-only (deaths excluded)")


def test_ebola_client_region_control():
    with tempfile.TemporaryDirectory() as d:
        p = _ebola_csv(d)
        base = ts.load_ebola(p).meta["node_ids"]
        core = ts.load_ebola(p, countries=ts.EBOLA_CORE_COUNTRIES).meta["node_ids"]
        excl = ts.load_ebola(p, exclude_regions=["Macenta"]).meta["node_ids"]
    assert "nigeria|lagos" in base and "nigeria|lagos" not in core, \
        "restricting to core countries drops Nigeria"
    assert len(core) == len(base) - 1
    assert "guinea|macenta" not in excl, "exclude_regions by district name works"
    print("ok  ebola region control: countries / exclude both work")


def test_accent_stripping_in_canon():
    assert ts._canon("Guéckédou ") == ts._canon("Gueckedou") == "gueckedou"
    assert ts._canon("Chiang Mai") == "chiang mai"
    print("ok  _canon strips accents (name-join robustness)")


def test_influenza_anchors_are_week_ending_saturdays():
    # pd.date_range(start, freq="W-SAT") SILENTLY snaps to the next Saturday if `start` is not
    # one -- shifting the whole calendar and the seasonality phase with it, with no error. Every
    # anchor must already be an MMWR week-ending Saturday.
    for k, v in ts.INFLUENZA_START.items():
        d = pd.Timestamp(v)
        assert d.dayofweek == 5, f"INFLUENZA_START[{k}]={v} is a {d.day_name()}, not a Saturday"
        assert pd.date_range(d, periods=1, freq=ts.WEEK_ANCHOR)[0] == d, f"{k}: W-SAT snap"
    # The US sets begin at the CDC flu-season boundary (MMWR week ~40), NOT in January.
    # Anchoring them to January puts the calendar 39 weeks out and inverts the seasonal phase.
    for k in ("us-regions", "us-states"):
        wk = pd.Timestamp(ts.INFLUENZA_START[k]).isocalendar().week
        assert 39 <= wk <= 41, f"{k} must start at the flu-season boundary, got ISO week {wk}"
    print("ok  influenza anchors: week-ending Saturdays; US sets on the season boundary")


def test_influenza_loader_shapes():
    N, T = 5, 30
    rng = np.random.default_rng(1)
    mat = rng.integers(0, 200, size=(T, N))          # ColaGNN ships [T,N]
    adj = (rng.random((N, N)) > 0.5).astype(int)
    adj = np.maximum(adj, adj.T)                     # shipped graphs are symmetric...
    np.fill_diagonal(adj, 1)                         # ...and carry a self-loop per node
    adj[0] = 0; adj[:, 0] = 0; adj[0, 0] = 1         # node 0: self-loop ONLY (an island)
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "japan.txt"); pd.DataFrame(mat).to_csv(mp, header=False, index=False)
        ap = os.path.join(d, "adj.txt");   pd.DataFrame(adj).to_csv(ap, header=False, index=False)
        dt = ts.load_influenza(mp, ap, dataset="japan")
    assert dt.X.shape == (N, T, 4) and dt.A_geo.shape == (N, N)   # 4 core channels
    assert dt.meta["A_geo_kind"] == "shipped(diag_zeroed)"
    assert str(dt.meta["dates"][0].date()) >= "2012-08-01", "Japan anchored to Aug 2012"

    # ONE graph convention across all diseases: _contiguity() zeroes the diagonal for
    # dengue/Ebola, so influenza must too, or the shared encoder's  = A + I would give
    # influenza self-weight 2 against dengue's 1 -- a structural tell for disease identity.
    assert np.diag(dt.A_geo).sum() == 0, "diagonal must be zeroed (encoder adds I once)"

    # ...but the EDGES stay bit-for-bit shipped: comparability is the sole reason we reuse
    # this graph. Zeroing the diagonal must not add, drop, or reweight an off-diagonal edge.
    off = adj.astype(np.float32).copy(); np.fill_diagonal(off, 0.0)
    assert np.array_equal(dt.A_geo, off), "off-diagonal edges must be reused verbatim"

    # A node whose only shipped entry was its self-loop is now degree-0. That isolation is
    # INHERITED, never patched with an invented k-NN edge (which _contiguity does do for
    # dengue/Ebola): the encoder's uniform  = A + I restores it to self-aggregation only.
    assert dt.meta["node_ids"][0] in dt.meta["isolated_nodes"]
    assert dt.meta["requires_encoder_self_loops"] is True
    print(f"ok  influenza loader: N={N} T={T}, verbatim edges + zeroed diag + MMWR anchor")


def test_no_future_leak_in_mask_semantics():
    # M must mark imputed steps so they are excluded from loss/eval downstream.
    N, T = 2, 10
    raw = np.zeros((N, T)); mask = np.ones((N, T), np.uint8)
    mask[:, 5:] = 0
    sc = ts.fit_scalers(raw, mask, 5)
    assert sc["mean"].shape == (N,)
    print("ok  mask semantics available for loss/eval exclusion")


def test_node_col_for_rejects_admin0_and_unknown():
    cols = ts.DENGUE_COLS
    assert ts._node_col_for("Admin1", cols) == cols["adm1"]
    assert ts._node_col_for("Admin2", cols) == cols["adm2"]
    for bad in ["Admin0", "auto", "admin1", "national", ""]:
        raised = False
        try:
            ts._node_col_for(bad, cols)
        except ValueError:
            raised = True
        assert raised, f"level {bad!r} must raise (no silent adm0 fallback)"
    print("ok  _node_col_for rejects Admin0/unknown (no silent adm0 fallback)")


def test_per_country_split_gives_every_country_train_and_is_chronological():
    T = 60
    node_ids = ["a|n1", "a|n2", "a|n3", "b|n1", "b|n2"]
    obs = np.zeros((5, T), np.uint8)
    obs[0, 0:10] = 1    # a|n1 early
    obs[1, 0:10] = 1    # a|n2 early
    obs[2, 8:10] = 1    # a|n3 late-within-A: its cells fall in country A's TEST window
    obs[3, 40:60] = 1   # b|n1 late (would starve under a single global split)
    obs[4, 40:60] = 1   # b|n2 late
    sp = ts.per_country_chronological_split(node_ids, obs, ratios=(0.5, 0.2, 0.3))
    tr, va, te = sp["train_mask"], sp["val_mask"], sp["test_mask"]
    assert np.array_equal((tr | va | te), obs), "every observed cell assigned exactly once"
    assert (tr & va).sum() == 0 and (tr & te).sum() == 0 and (va & te).sum() == 0
    assert ((tr | va | te) & (1 - obs)).sum() == 0, "no imputed cell may enter a split"
    # every COUNTRY gets train cells — the starvation fix
    assert tr[[0, 1, 2]].sum() > 0 and tr[[3, 4]].sum() > 0, "country b must have train"
    g_train_end, _ = ts.chronological_split(T)          # global cut ~ week 30
    assert obs[3, :g_train_end].sum() == 0, "b would have ZERO train under a global split"
    # C3: the per-node fallback was REMOVED. a|n3's cells all lie inside country A's test window,
    # so it gets NO train cell -- it takes A's boundary like every other A node. This assertion used
    # to require the opposite (tr[2].sum() >= 1). Re-cutting a late node on its own timeline placed
    # its training cells inside its neighbours' test period, and because a graph network aggregates
    # over neighbours that pulled test-period data into the training pass: 475 dengue nodes, 31% of
    # observed cells, in columns that mixed training and test.
    assert tr[2].sum() == 0, "C3: a node whose cells sit in its country's test window must not train"
    # and the invariant that replaced the fallback: within a country, a week belongs to ONE phase.
    for c in ("a", "b"):
        rows = [k for k, n in enumerate(node_ids) if n.startswith(f"{c}|")]
        for t in range(T):
            phases = sum(int(m[rows, t].any()) for m in (tr, va, te))
            assert phases <= 1, f"country {c} week {t} is in {phases} phases -- C3 violated"
    for i in range(5):
        w_tr, w_va, w_te = np.where(tr[i])[0], np.where(va[i])[0], np.where(te[i])[0]
        if len(w_tr) and len(w_va): assert w_tr.max() < w_va.min()
        if len(w_va) and len(w_te): assert w_va.max() < w_te.min()
        if len(w_tr) and len(w_te): assert w_tr.max() < w_te.min()
    print("ok  per-country split: every country trains, chronological, one phase per country-week")


def test_per_country_split_single_obs_node_takes_its_country_boundary():
    # A node with a single observed week, late in its country's span. Under C3 it does NOT get a
    # train cell of its own: it takes country c's boundary like every other c node, and its one
    # observation lands in whichever phase that boundary puts it.
    #
    # This test previously required the opposite, on the grounds that a node with no train cell has
    # an undefined per-node scaler (mean 0 / std 1). That reasoning is obsolete: D1 replaced the
    # blanket guard with a COUNTRY-POOLED fallback computed from training cells only, so such a node
    # is normalised on its country's statistics rather than handed a fabricated train cell. Buying a
    # defined scaler by re-cutting the node on its own timeline is exactly the leak C3 removed.
    T = 20
    ids = ["c|n1", "c|n2", "c|solo"]
    obs = np.zeros((3, T), np.uint8)
    obs[0, 0:10] = 1
    obs[1, 0:10] = 1
    obs[2, 18] = 1                     # one observation, late in the country's span
    sp = ts.per_country_chronological_split(ids, obs)
    tr, va, te = sp["train_mask"], sp["val_mask"], sp["test_mask"]
    assert tr[2].sum() == 0, "C3: a late solo node must not be re-cut on its own timeline"
    assert (tr | va | te)[2, 18] == 1, "its observation is still assigned to exactly one phase"
    assert tr[[0, 1]].sum() > 0, "the country itself must still have training data"
    print("ok  per-country split: single-observation node takes its country's boundary")


def test_per_country_split_scaler_is_leakage_safe():
    # Leakage gate for the per-country path: the per-node scaler must be fit on TRAIN
    # cells only, even across disjoint country spans.
    N, T = 4, 80
    rng = np.random.default_rng(3)
    raw = rng.integers(1, 500, size=(N, T)).astype(float)
    mask = np.ones((N, T), np.uint8)
    ids = ["a|1", "a|2", "b|1", "b|2"]
    mask[:2, 40:] = 0          # country a observes weeks 0..39
    mask[2:, :40] = 0          # country b observes weeks 40..79 (disjoint span)
    sp = ts.per_country_chronological_split(ids, mask)
    fit = mask.astype(bool) & sp["train_mask"].astype(bool)
    assert (fit.sum(1) > 0).all(), "every node must have a train cell -> defined scaler"
    s1 = ts.fit_scalers_masked(raw, fit)
    raw2 = raw.copy()
    raw2[~sp["train_mask"].astype(bool)] *= 1000.0     # perturb every non-train cell
    s2 = ts.fit_scalers_masked(raw2, fit)
    assert np.allclose(s1["mean"], s2["mean"]) and np.allclose(s1["std"], s2["std"]), \
        "per-country scaler must depend on TRAIN cells only (val/test never leak)"
    print("ok  per-country split scaler leakage-safe (train-only, every node defined)")


def test_resolve_country_levels_finest_available_and_exclusions():
    rows = []
    rows += _dengue_rows_lvl("BRAZIL", ["ACRE", "BAHIA", "ALAGOAS"], "Admin2", n_weeks=60)
    rows += _dengue_rows_lvl("JAPAN", ["TOKYO", "OSAKA", "KYOTO"], "Admin1", n_weeks=60)
    rows += _dengue_rows_lvl("JAPAN", ["SHIBUYA"], "Admin2", n_weeks=60)   # 1 Admin2 unit only
    rows += _dengue_rows_lvl("PERU", ["PERU"], "Admin0", n_weeks=60)       # national aggregate only
    rows += _dengue_rows_lvl("CHILE", ["SANTIAGO", "VALPARAISO"], "Admin1", n_weeks=60)  # 2 < 3
    df = pd.DataFrame(rows, columns=DENGUE_HEADER)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dengue.csv"); df.to_csv(p, index=False)
        levels, report = ts.resolve_country_levels(
            p, countries=["brazil", "japan", "peru", "chile"],
            min_weeks=52, min_nodes_per_country=3)
    assert levels["brazil"] == "Admin2", "finest-available: Brazil has adequate Admin2"
    assert levels["japan"] == "Admin1", "Japan Admin2 too sparse (1<3) -> fall back to Admin1"
    assert "peru" not in levels and report["peru"]["excluded_reason"] == "admin0_only"
    assert "chile" not in levels and report["chile"]["excluded_reason"] == "insufficient_coverage"
    assert set(levels.values()) <= {"Admin1", "Admin2"}, "Admin0 is never a chosen level"
    print("ok  resolve_country_levels: finest-available + admin0_only/insufficient exclusions")


def test_dengue_admin2_same_name_in_two_states_stays_two_nodes():
    """Admin-2 leaf names are NOT unique within a country. In the real extract Brazil
    carries 239 adm_2 names shared by 2-5 different states ('bom jesus' is five distinct
    municipalities, ~1500km apart). Keying nodes on 'country|adm2' merged them and
    pivot_table(aggfunc='sum') added their case counts together -- inflating incidence
    and making the GADM polygon join ambiguous. The node key must carry the parent adm1."""
    rows = []
    for state in ["PIAUI", "SANTA CATARINA"]:                 # same municipality name...
        rows += _dengue_rows_lvl("BRAZIL", ["BOM JESUS"], "Admin2",
                                 n_weeks=60, parent=state)    # ...two different states
    rows += _dengue_rows_lvl("BRAZIL", ["BONITO"], "Admin2", n_weeks=60, parent="BAHIA")
    df = pd.DataFrame(rows, columns=DENGUE_HEADER)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dengue.csv"); df.to_csv(p, index=False)
        dt = ts.load_dengue(p, countries=["brazil"], level="auto",
                            min_weeks=52, min_nodes_per_country=3)
    ids = dt.meta["node_ids"]
    assert len(ids) == 3, f"two 'Bom Jesus' municipalities must stay distinct, got {ids}"
    bom = [n for n in ids if n.endswith("|bom jesus")]
    assert len(bom) == 2, f"expected 2 distinct Bom Jesus nodes, got {bom}"
    assert bom[0] != bom[1] and all(n.count("|") == 2 for n in ids), \
        "Admin2 node id must be 'country|adm1|adm2'"
    # and the merge must not have summed their counts into one series
    assert dt.M.shape[0] == 3
    print("ok  Admin2 nodes keyed by parent adm1 (same-name municipalities stay distinct)")


def test_dengue_aliases_reunite_a_series_split_across_two_spellings():
    """OpenDengue reports one unit under two spellings in DISJOINT eras -- the DR's
    'salcedo' (2009-13) and 'hermanas mirabal' (2006-08) are the same province (renamed in
    2007). Keyed verbatim that is two nodes, each with half a history and a per-node scaler
    fit on that half. Aliases are applied at LOAD time (before the pivot) so the two halves
    merge back into one continuous series."""
    rows = []
    rows += _dengue_rows_lvl("DR", ["SALCEDO"], "Admin1", n_weeks=60, start="2009-01-03")
    rows += _dengue_rows_lvl("DR", ["HERMANAS MIRABAL"], "Admin1", n_weeks=60,
                             start="2006-01-07")                  # earlier era, same province
    rows += _dengue_rows_lvl("DR", ["AZUA", "BARAHONA"], "Admin1", n_weeks=60,
                             start="2006-01-07")
    df = pd.DataFrame(rows, columns=DENGUE_HEADER)
    aliases = {"dr": {"units": {"hermanas mirabal": "salcedo"}}}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dengue.csv"); df.to_csv(p, index=False)
        raw = ts.load_dengue(p, countries=["dr"], level="Admin1", min_weeks=1,
                             min_nodes_per_country=1)
        merged = ts.load_dengue(p, countries=["dr"], level="Admin1", min_weeks=1,
                                min_nodes_per_country=1, name_aliases=aliases)
    assert raw.X.shape[0] == 4, "unaliased: the province is split into two nodes"
    assert merged.X.shape[0] == 3, "aliased: the two halves must merge into ONE node"
    ids = merged.meta["node_ids"]
    assert "dr|hermanas mirabal" not in ids and "dr|salcedo" in ids
    i = ids.index("dr|salcedo")
    # the merged node must carry BOTH eras' observations, not just one
    assert merged.M[i].sum() == 120, f"merged series must span both eras, got {merged.M[i].sum()}"
    print("ok  load-time aliases reunite a series split across two spellings/eras")


def test_dengue_gadm_typos_never_reach_node_ids():
    """GADM's own defects are absorbed at the JOIN, never at LOAD, so our node ids keep the
    correct spelling. Folding them into the load-time map is what once put `japan|naoasaki` and
    `peru|huanuco|huenuco` into the bundle, where they would have been frozen into the released
    file and printed in the paper's node table.

    This checks the COMMITTED maps and needs no geopandas: the fix is a partition of two maps,
    and the way it regresses is someone adding a GADM quirk to the wrong one.
    """
    from dengue_aliases import DENGUE_NAME_ALIASES, DENGUE_GADM_FIX

    # the two GADM 4.1 typos live in the JOIN-time map...
    assert DENGUE_GADM_FIX["japan"]["nagasaki"] == "naoasaki"
    assert DENGUE_GADM_FIX["peru"]["huanuco|huanuco"] == "huanuco|huenuco"

    # ...and no load-time alias may TARGET a GADM misspelling, because a load-time target
    # becomes the node id. This is the assertion that actually pins the defect.
    gadm_typos = {"naoasaki", "huanuco|huenuco"}
    for country, m in DENGUE_NAME_ALIASES.items():
        for src, tgt in (m.get("units") or {}).items():
            assert ts._canon(tgt) not in gadm_typos, (
                f"DENGUE_NAME_ALIASES[{country!r}][{src!r}] -> {tgt!r} targets a GADM typo. "
                f"A load-time alias renames the NODE ID; GADM's defects belong in "
                f"DENGUE_GADM_FIX, which is applied at the join only.")

    # gadm_fix must be inert without a graph build: it may not touch the series or the ids
    rows = _dengue_rows_lvl("JPN", ["NAGASAKI", "TOKYO"], "Admin1", n_weeks=60)
    df = pd.DataFrame(rows, columns=DENGUE_HEADER)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dengue.csv"); df.to_csv(p, index=False)
        dt = ts.load_dengue(p, countries=["jpn"], level="Admin1", min_weeks=1,
                            min_nodes_per_country=1, gadm_fix=DENGUE_GADM_FIX)
    assert "jpn|nagasaki" in dt.meta["node_ids"], "the node must keep the CORRECT spelling"
    assert "jpn|naoasaki" not in dt.meta["node_ids"], "GADM's typo must never become a node id"
    print("ok  GADM's typos absorbed at the join, never in dengue's node ids")


def test_dengue_unmappable_nodes_dropped_only_when_declared():
    """A node with no GADM polygon is dropped ONLY via the explicit, reasoned
    DENGUE_UNMAPPABLE list -- never silently by the graph builder."""
    rows = _dengue_rows_lvl("ARG", ["BERMEJO", "FORMOSA", "GHOSTDEPT"], "Admin1", n_weeks=60)
    df = pd.DataFrame(rows, columns=DENGUE_HEADER)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dengue.csv"); df.to_csv(p, index=False)
        kept = ts.load_dengue(p, countries=["arg"], level="Admin1", min_weeks=1,
                              min_nodes_per_country=1)
        pruned = ts.load_dengue(p, countries=["arg"], level="Admin1", min_weeks=1,
                                min_nodes_per_country=1,
                                unmappable={"arg": {"ghostdept": "no such department"}})
    assert kept.X.shape[0] == 3, "without the declaration the node stays"
    assert pruned.X.shape[0] == 2, "declared-unmappable node must be dropped"
    assert pruned.meta["nodes_dropped_unmappable"] == ["arg|ghostdept"], \
        "the drop must be recorded in meta for the audit note"
    print("ok  unmappable nodes dropped only when explicitly declared, and recorded")


def test_load_dengue_auto_mixed_levels_and_per_country_split():
    # Disjoint spans on purpose: Brazil(Admin2) early 2005, Japan(Admin1) late 2018.
    # A single global split would starve Japan; per-country must give it train cells.
    rows = []
    rows += _dengue_rows_lvl("BRAZIL", ["ACRE", "BAHIA", "ALAGOAS"], "Admin2",
                             n_weeks=80, start="2005-01-01")
    rows += _dengue_rows_lvl("JAPAN", ["TOKYO", "OSAKA", "KYOTO"], "Admin1",
                             n_weeks=80, start="2018-01-06")
    rows += _dengue_rows_lvl("PERU", ["PERU"], "Admin0", n_weeks=80)  # excluded, no node
    df = pd.DataFrame(rows, columns=DENGUE_HEADER)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dengue.csv"); df.to_csv(p, index=False)
        dt = ts.load_dengue(p, countries=["brazil", "japan", "peru"], level="auto")
    m = dt.meta
    assert m["country_levels"] == {"brazil": "Admin2", "japan": "Admin1"}
    assert set(m["node_levels"].values()) <= {"Admin1", "Admin2"}, "no Admin0 nodes"
    for n, lvl in m["node_levels"].items():                 # one level per country
        assert lvl == m["country_levels"][m["node_country"][n]]
    assert m["countries_excluded"]["peru"] == "admin0_only"
    assert "level_report" in m
    assert m["split_scheme"] == "per_country_chronological_50_20_30"
    tr = m["split"]["train_mask"]
    ids = m["node_ids"]; nc = m["node_country"]
    for c in ["brazil", "japan"]:
        idx = [i for i, n in enumerate(ids) if nc[n] == c]
        assert tr[idx].sum() > 0, f"{c} must have train cells under the per-country split"
    dt.check()
    print("ok  load_dengue auto: mixed levels, node meta stamped, per-country split trains all")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} schema tests passed.")

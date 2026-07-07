"""
to_schema.py — Standardised, disease-agnostic input schema loaders.

Coerces dengue (OpenDengue), influenza (ColaGNN-shipped matrices) and Ebola
(HDX sub-national, cumulative) into one `DiseaseTensors` bundle:

    {X:[N,T,F], A_geo:[N,N], A_mob:[N,N]|None, C:[N,S], M:[N,T], y:[N,T], meta}

See schema_spec.md for the design rationale and the [CONFIRM] items.
Adjacency from GADM shapefiles requires geopandas (optional at import time);
the core tensor logic does not.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Feature layout (see schema_spec.md §3). Core block is transfer-safe.
# `obs_mask` is a core channel (client decision D15): the shared encoder must
# know which steps are imputed vs observed — critical for Ebola's many gaps.
# --------------------------------------------------------------------------- #
CORE_FEATURES = ["incidence_norm", "sin_doy", "cos_doy", "obs_mask"]
CORE_FEATURE_IDX = [0, 1, 2, 3]

# Weekly resample/label anchor. MMWR epi-weeks run Sunday->Saturday, so pandas
# 'W-SAT' (week ending Saturday) is the correct anchor (client decision D6).
WEEK_ANCHOR = "W-SAT"


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
@dataclass
class DiseaseTensors:
    X: np.ndarray                    # [N, T, F] float32
    A_geo: np.ndarray               # [N, N]   float32
    C: np.ndarray                    # [N, S]   float32
    M: np.ndarray                    # [N, T]   uint8
    y: np.ndarray                    # [N, T]   float32 (model space)
    meta: dict = field(default_factory=dict)
    A_mob: Optional[np.ndarray] = None   # [N, N] float32 or None

    def check(self) -> "DiseaseTensors":
        """Assert the schema invariants. Cheap; call it everywhere."""
        N, T, F = self.X.shape
        assert self.A_geo.shape == (N, N), "A_geo must be [N,N]"
        assert self.C.shape[0] == N, "C must have N rows"
        assert self.M.shape == (N, T), "M must be [N,T]"
        assert self.y.shape == (N, T), "y must be [N,T]"
        assert F >= len(CORE_FEATURES), "X must contain the core channels"
        assert len(self.meta["node_ids"]) == N, "node_ids must align to N"
        assert len(self.meta["dates"]) == T, "dates must align to T"
        assert self.M.dtype == np.uint8, "M must be uint8 (1 obs / 0 missing)"
        if self.A_mob is not None:
            assert self.A_mob.shape == (N, N), "A_mob must be [N,N]"
        # row order is the single source of truth
        assert self.X.dtype == np.float32
        return self

    def transfer_view(self) -> np.ndarray:
        """The ONLY channels the shared cross-disease encoder may see (D9):
        the core, transfer-safe block. Extended channels (deaths, etc.) are
        single-disease only and are excluded here by construction."""
        return self.X[:, :, self.meta["core_feature_idx"]]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _canon(s) -> str:
    """Canonical id: strip accents/diacritics, lowercase, collapse whitespace.
    Accent-stripping matters because geometry is joined to series on NAMES, not
    codes (client declined to provide code columns, D17), so 'Guéckédou' must
    equal 'Gueckedou' and 'Chiang Mai ' equal 'chiang mai'."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


# First MMWR epi-week per ColaGNN dataset. The repo's matrices are UNDATED
# (README: "rows indicate timestamps (weeks)... arranged in chronological
# order"), so these are anchored from the published ranges — Japan-Prefectures
# Aug 2012–Mar 2019 (348 wk), US-Regions 2002–2017 (785 wk), US-States
# 2010–2017 (360 wk) [ColaGNN, CIKM'20; EpiGNN, ECML-PKDD'22]. A day-level
# offset is immaterial to the seasonality phase; the year/month is what matters.
INFLUENZA_START = {
    "us-regions": "2002-01-05", "region": "2002-01-05",
    "us-states": "2010-01-09", "state": "2010-01-09",
    "japan": "2012-08-04", "japan-prefectures": "2012-08-04",
}


def _apply_node_filter(df: pd.DataFrame, node_col: str,
                       include=None, exclude=None) -> pd.DataFrame:
    """Client-facing region control: keep only `include`, drop any `exclude`
    (both matched on canonical names). Either may be None = no constraint."""
    if include is not None:
        inc = {_canon(x) for x in include}
        df = df[df[node_col].isin(inc)]
    if exclude:
        exc = {_canon(x) for x in exclude}
        df = df[~df[node_col].isin(exc)]
    return df


def _seasonality(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """sin/cos of day-of-year — resolution-invariant annual phase."""
    doy = dates.dayofyear.to_numpy().astype(np.float64)
    frac = 2.0 * np.pi * doy / 365.25
    return np.sin(frac), np.cos(frac)


def fit_scalers_masked(raw_counts: np.ndarray, fit_mask: np.ndarray,
                       per_disease: bool = False) -> dict:
    """Fit log1p+zscore stats using ONLY the cells where `fit_mask` is true.
    This is the general leakage-safe primitive: pass a train-slice mask for
    development diseases, or a few-shot *support* mask for the held-out disease.
    `per_disease=True` pools all cells into one (mean,std) — the right choice
    when each node has too few fit cells for a stable per-node scaler (e.g. a
    2-week Ebola support set)."""
    N, T = raw_counts.shape
    logc = np.log1p(np.clip(raw_counts, 0, None))
    fm = fit_mask.astype(bool)
    if per_disease:
        flat = logc[fm]
        mu = np.full(N, flat.mean() if flat.size else 0.0)
        sd = np.full(N, flat.std() if flat.size else 1.0)
    else:
        mu, sd = np.zeros(N), np.ones(N)
        for i in range(N):
            v = logc[i][fm[i]]
            if v.size:
                mu[i], sd[i] = v.mean(), v.std()
    sd[sd < 1e-8] = 1.0  # guard zero-variance nodes (sporadic Ebola districts)
    return {"transform": "log1p_z", "mean": mu.astype(np.float32),
            "std": sd.astype(np.float32)}


def fit_scalers(raw_counts: np.ndarray, mask: np.ndarray, train_end: int,
                per_disease: bool = False) -> dict:
    """Leakage-safe scaler for development diseases: fit on observed TRAIN cells
    only (mask==1 and t < train_end)."""
    fit_mask = mask.astype(bool).copy()
    fit_mask[:, train_end:] = False
    return fit_scalers_masked(raw_counts, fit_mask, per_disease=per_disease)


def apply_scaler(raw_counts: np.ndarray, scaler: dict) -> np.ndarray:
    logc = np.log1p(np.clip(raw_counts, 0, None))
    return ((logc - scaler["mean"][:, None]) / scaler["std"][:, None]).astype(np.float32)


def invert_scaler(norm: np.ndarray, scaler: dict) -> np.ndarray:
    """Model space -> real counts, for RMSE/MAE reporting."""
    logc = norm * scaler["std"][:, None] + scaler["mean"][:, None]
    return np.expm1(logc)


def cumulative_to_weekly_incidence(df: pd.DataFrame, node_col: str, date_col: str,
                                   value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HDX Ebola: cumulative sit-report counts -> weekly new incidence.

    Steps (schema_spec §5): weekly resample (last cumulative per epi-week,
    ffill) -> diff -> clip negatives -> mask ffilled weeks.
    Returns (incidence[N,T] wide, mask[N,T] wide), both indexed by node & week.
    """
    df = df.copy()
    df["_node"] = df[node_col].map(_canon)
    df["_date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["_date"])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    inc_frames, mask_frames = {}, {}
    for node, g in df.groupby("_node"):
        s = g.set_index("_date")[value_col].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        # last cumulative value observed within each MMWR epi-week (Sun–Sat)
        weekly_obs = s.resample(WEEK_ANCHOR).last()
        observed = weekly_obs.notna()               # weeks with an actual report
        weekly_cum = weekly_obs.ffill()             # carry cumulative forward
        new = weekly_cum.diff()
        new.iloc[0] = weekly_cum.iloc[0]            # first week ~ outbreak onset
        new = new.clip(lower=0)                     # reporting-correction artefacts
        inc_frames[node] = new
        mask_frames[node] = observed.astype(np.uint8)

    inc = pd.DataFrame(inc_frames).T                # [N, T]
    msk = pd.DataFrame(mask_frames).T.reindex_like(inc).fillna(0).astype(np.uint8)
    inc = inc.fillna(0.0)
    return inc, msk


def _assemble_X(inc_norm: np.ndarray, dates: pd.DatetimeIndex, obs_mask: np.ndarray,
                extended: Optional[dict] = None) -> tuple[np.ndarray, list]:
    """Stack core channels [incidence, sin, cos, obs_mask] (+ optional extended).
    obs_mask (1=observed, 0=imputed) is core so the shared encoder can tell real
    observations from imputed gaps (D15)."""
    N, T = inc_norm.shape
    sin, cos = _seasonality(dates)
    sin = np.broadcast_to(sin, (N, T)).astype(np.float32)
    cos = np.broadcast_to(cos, (N, T)).astype(np.float32)
    chans = [inc_norm, sin, cos, obs_mask.astype(np.float32)]
    names = list(CORE_FEATURES)
    if extended:
        for nm, arr in extended.items():
            chans.append(arr.astype(np.float32))
            names.append(nm)
    X = np.stack(chans, axis=-1).astype(np.float32)   # [N,T,F]
    return X, names


def chronological_split(T: int, ratios=(0.5, 0.2, 0.3)) -> tuple[int, int]:
    """Return (train_end, val_end) index cuts. Default 50/20/30 (EpiGNN)."""
    tr = int(round(T * ratios[0]))
    va = int(round(T * (ratios[0] + ratios[1])))
    return tr, va


# --------------------------------------------------------------------------- #
# Adjacency (GADM shapefiles) — geopandas optional
# --------------------------------------------------------------------------- #
def build_adjacency(shapefile_path: str, node_ids: Sequence[str],
                    name_field: str, knn_fallback: int = 4,
                    name_aliases: Optional[dict] = None) -> np.ndarray:
    """Queen contiguity from a GADM shapefile, aligned to node_ids, with a
    k-NN centroid fallback for isolated units. Requires geopandas.

    Because we join on NAMES not codes (D17), `name_aliases` lets you hand-map
    residual mismatches (data name -> shapefile name, canonicalised). Any node
    still unmatched raises with the full list so it can be aliased — never
    silently dropped or misaligned."""
    try:
        import geopandas as gpd
        from libpysal.weights import Queen, KNN
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "build_adjacency needs geopandas + libpysal: "
            "`pip install geopandas libpysal`"
        ) from e

    aliases = {_canon(k): _canon(v) for k, v in (name_aliases or {}).items()}
    want = [aliases.get(_canon(n), _canon(n)) for n in node_ids]

    gdf = gpd.read_file(shapefile_path)
    gdf["_node"] = gdf[name_field].map(_canon)
    gdf = gdf.drop_duplicates("_node").set_index("_node").reindex(want)
    if gdf.geometry.isna().any():
        missing = [n for n, g in zip(node_ids, gdf.geometry.isna()) if g]
        raise ValueError(
            f"Shapefile has no geometry for {len(missing)} node(s): {missing}. "
            f"Add them to name_aliases (data_name -> shapefile_name).")

    gdf = gdf.reset_index()
    N = len(gdf)
    A = np.zeros((N, N), dtype=np.float32)
    w = Queen.from_dataframe(gdf, use_index=False)
    for i, neighbours in w.neighbors.items():
        for j in neighbours:
            A[i, j] = 1.0
    # k-NN fallback for isolated nodes (islands / no reporting neighbour)
    isolated = np.where(A.sum(1) == 0)[0]
    if len(isolated):
        wk = KNN.from_dataframe(gdf, k=knn_fallback)
        for i in isolated:
            for j in wk.neighbors[i]:
                A[i, j] = 1.0
    return A


def build_dengue_adjacency(shapefiles_by_country: dict, node_ids: Sequence[str],
                           name_field: str, knn_fallback: int = 4,
                           name_aliases: Optional[dict] = None) -> np.ndarray:
    """Block-diagonal geographic adjacency for a multi-country dengue graph.
    Contiguity is built *within* each country (from that country's shapefile)
    and assembled block-diagonally — there are no cross-border edges, which is
    correct: nodes in different countries are not geographic neighbours.

    node_ids are 'country|adm'; `shapefiles_by_country` maps canonical country
    name -> shapefile path."""
    N = len(node_ids)
    A = np.zeros((N, N), dtype=np.float32)
    pos = {n: i for i, n in enumerate(node_ids)}
    for country, shp in shapefiles_by_country.items():
        c = _canon(country)
        c_nodes = [n for n in node_ids if n.split("|", 1)[0] == c]
        if not c_nodes:
            continue
        adm_names = [n.split("|", 1)[1] for n in c_nodes]
        block = build_adjacency(shp, adm_names, name_field, knn_fallback,
                                name_aliases=name_aliases)  # [k,k]
        for a, na in enumerate(c_nodes):
            for b, nb in enumerate(c_nodes):
                A[pos[na], pos[nb]] = block[a, b]
    return A


# --------------------------------------------------------------------------- #
# DENGUE — OpenDengue best-spatial extract (16-col layout from the sample)
# --------------------------------------------------------------------------- #
DENGUE_COLS = dict(
    adm0="adm_0_name", adm1="adm_1_name", adm2="adm_2_name",
    start="calendar_start_date", value="dengue_total",
    t_res="T_res", s_res="S_res",
)

# WHO/CDC dengue-endemic countries (canonical, lowercase) — the default
# development set. This is the *candidate* pool; the loader keeps only those
# that actually clear the weekly subnational coverage thresholds in the extract
# (reported in meta['coverage']), so the final set is data-driven, not asserted.
DENGUE_ENDEMIC = [
    # Americas
    "brazil", "mexico", "colombia", "peru", "argentina", "bolivia", "paraguay",
    "nicaragua", "venezuela", "ecuador", "dominican republic", "honduras",
    "guatemala",
    # South & Southeast Asia / Western Pacific
    "thailand", "vietnam", "malaysia", "philippines", "indonesia", "sri lanka",
    "cambodia", "laos", "india", "bangladesh", "singapore",
]


def dengue_country_coverage(csv_path: str, level: str = "Admin1",
                            t_res_filter: str = "Week",
                            cols: dict = DENGUE_COLS) -> pd.DataFrame:
    """Week-2 audit helper: per-country weekly subnational coverage in the
    extract (#nodes, max weeks/node, date span). Use this to confirm which
    endemic countries actually have enough signal before locking the set."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    if t_res_filter is not None and cols["t_res"] in df:
        df = df[df[cols["t_res"]].map(_canon) == _canon(t_res_filter)]
    if level and cols["s_res"] in df:
        df = df[df[cols["s_res"]].map(_canon) == _canon(level)]
    node_col = {"Admin1": cols["adm1"], "Admin2": cols["adm2"]}.get(level, cols["adm0"])
    df["_c"] = df[cols["adm0"]].map(_canon)
    df["_adm"] = df[node_col].map(_canon)
    df["_date"] = pd.to_datetime(df[cols["start"]], dayfirst=True, errors="coerce")
    rows = []
    for c, g in df.dropna(subset=["_date"]).groupby("_c"):
        per_node = g.groupby("_adm")["_date"].nunique()
        rows.append(dict(country=c, n_nodes=int(per_node.size),
                         max_weeks_per_node=int(per_node.max()),
                         start=g["_date"].min(), end=g["_date"].max(),
                         endemic=c in DENGUE_ENDEMIC))
    return pd.DataFrame(rows).sort_values("n_nodes", ascending=False)


def load_dengue(csv_path: str, countries=None, level: str = "Admin1",
                t_res_filter: str = "Week", cols: dict = DENGUE_COLS,
                include_regions: Optional[Sequence[str]] = None,
                exclude_regions: Optional[Sequence[str]] = None,
                min_weeks: int = 52, min_nodes_per_country: int = 3,
                ratios=(0.5, 0.2, 0.3),
                shapefiles: Optional[dict] = None) -> DiseaseTensors:
    """OpenDengue -> DiseaseTensors, MULTI-COUNTRY (dengue-endemic development
    set). `dengue_total` is per-period incidence (not cumulative); NA/suppressed
    -> M=0.

    countries:
      None (default) -> the DENGUE_ENDEMIC pool, intersected with the file;
      a str          -> a single country; a list -> exactly those.
    Node id is 'country|adm' (block-diagonal graph: intra-country contiguity,
    no cross-border edges — an ocean between Brazil and Thailand is not an edge).
    A country is kept only if >= `min_nodes_per_country` of its nodes have
    >= `min_weeks` observed weeks; kept/dropped and per-country coverage are
    recorded in meta['coverage']. `t_res_filter="Week"` -> the weekly (2014+)
    portion; "Month" -> legacy monthly; include/exclude_regions act on adm name.
    Pass `shapefiles={'brazil': path, ...}` to build the block-diagonal graph now;
    otherwise adjacency is deferred to Week 2."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df["_c"] = df[cols["adm0"]].map(_canon)

    if countries is None:
        want = set(DENGUE_ENDEMIC)
    elif isinstance(countries, str):
        want = {_canon(countries)}
    else:
        want = {_canon(c) for c in countries}
    present = set(df["_c"].unique())
    requested_missing = sorted(want - present)
    sel = sorted(want & present)
    assert sel, f"None of the requested dengue countries are in the file: {sorted(want)}"
    df = df[df["_c"].isin(sel)]

    if t_res_filter is not None and cols["t_res"] in df:
        df = df[df[cols["t_res"]].map(_canon) == _canon(t_res_filter)]
    node_col = {"Admin1": cols["adm1"], "Admin2": cols["adm2"]}.get(level, cols["adm0"])
    if level and cols["s_res"] in df:
        df = df[df[cols["s_res"]].map(_canon) == _canon(level)]

    df["_adm"] = df[node_col].map(_canon)
    df = _apply_node_filter(df, "_adm", include_regions, exclude_regions)
    df["_node"] = df["_c"] + "|" + df["_adm"]
    df["_date"] = pd.to_datetime(df[cols["start"]], dayfirst=True, errors="coerce")
    df["_val"] = pd.to_numeric(df[cols["value"]], errors="coerce")
    df = df.dropna(subset=["_date", "_adm"])
    assert len(df) > 0, "No dengue rows after filtering (check country/level/t_res)."

    weekly = (t_res_filter is None and _canon(df[cols["t_res"]].iloc[0]) == "week") \
        or (t_res_filter is not None and _canon(t_res_filter) == "week")
    # Bin each record into its MMWR epi-week (Sun–Sat) or month PERIOD, rather
    # than assuming the source dates already fall on a fixed weekday. This makes
    # alignment robust regardless of how OpenDengue labels week starts.
    period_freq = WEEK_ANCHOR if weekly else "M"
    df["_bin"] = df["_date"].dt.to_period(period_freq)
    wide = df.pivot_table(index="_node", columns="_bin", values="_val", aggfunc="sum")
    full_idx = pd.period_range(wide.columns.min(), wide.columns.max(), freq=period_freq)
    wide = wide.reindex(columns=full_idx)

    # data-driven coverage prune: keep endemic countries with enough weekly signal
    node_country = {n: n.split("|", 1)[0] for n in wide.index}
    obs_per_node = wide.notna().sum(axis=1)
    coverage, kept_countries = {}, []
    for c in sel:
        c_nodes = [n for n in wide.index if node_country[n] == c]
        good = [n for n in c_nodes if obs_per_node[n] >= min_weeks]
        coverage[c] = dict(n_nodes=len(c_nodes), n_nodes_ge_min_weeks=len(good),
                           max_weeks=int(obs_per_node[c_nodes].max()) if c_nodes else 0)
        if len(good) >= min_nodes_per_country:
            kept_countries.append(c)
    keep_nodes = [n for n in wide.index if node_country[n] in kept_countries]
    assert keep_nodes, (
        f"No dengue country cleared coverage (min_weeks={min_weeks}, "
        f"min_nodes_per_country={min_nodes_per_country}). Coverage: {coverage}")
    wide = wide.loc[keep_nodes]

    node_ids = list(wide.index)
    how = "end" if weekly else "start"
    dates = pd.DatetimeIndex(wide.columns.to_timestamp(how=how).normalize())
    raw = wide.to_numpy(dtype=np.float64)
    mask = (~np.isnan(raw)).astype(np.uint8)   # NA/suppressed -> missing
    raw = np.nan_to_num(raw, nan=0.0)

    A_geo, A_geo_kind = None, "deferred_block_diagonal"
    if shapefiles:
        name_field = cols["adm1"] if level == "Admin1" else cols["adm2"]
        A_geo = build_dengue_adjacency(shapefiles, node_ids, name_field)
        A_geo_kind = "queen+knn_block_diagonal"

    dt = _finalise(raw, mask, node_ids, dates, ratios,
                   disease="dengue", adm_level=level,
                   t_res="weekly" if weekly else "monthly",
                   A_geo=A_geo, A_geo_kind=A_geo_kind,
                   source="OpenDengue; Clarke J et al., Sci Data 2024;11(1):296")
    dt.meta["countries"] = kept_countries
    dt.meta["countries_dropped_low_coverage"] = [c for c in sel if c not in kept_countries]
    dt.meta["countries_requested_absent"] = requested_missing
    dt.meta["coverage"] = coverage
    dt.meta["graph_is_block_diagonal"] = True
    return dt


# --------------------------------------------------------------------------- #
# EBOLA — HDX sub-national (long, cumulative: Country/Localite/Indicator/value/date)
# --------------------------------------------------------------------------- #
EBOLA_COLS = dict(country="Country", district="Localite", indicator="Indicator",
                  value="value", date="date")
EBOLA_CORE_COUNTRIES = ["guinea", "liberia", "sierra leone"]


def load_ebola(csv_path: str, cols: dict = EBOLA_COLS,
               countries: Optional[Sequence[str]] = None,
               exclude_regions: Optional[Sequence[str]] = None,
               drop_below_total: Optional[float] = None,
               few_shot_support_weeks: int = 2,
               ratios=(0.5, 0.2, 0.3), shapefile: Optional[str] = None) -> DiseaseTensors:
    """HDX Ebola (cumulative, long) -> DiseaseTensors. Converts
    cumulative->weekly incidence; deaths as an extended channel.
    Node id = 'country|district'.

    Region control is the CLIENT'S choice (not silently applied):
      - `countries=None` keeps every country in the file (incl. Nigeria/Senegal/
        Mali). Pass `EBOLA_CORE_COUNTRIES` to restrict to Guinea/Liberia/
        Sierra Leone.
      - `exclude_regions` drops named districts or 'country|district' ids.
      - `drop_below_total` drops nodes whose total incidence is below a
        threshold (a principled way to remove degenerate near-zero districts).

    Few-shot protocol (D22): the SUPPORT set is the first `few_shot_support_weeks`
    OBSERVED weeks per district (default 2 = "14 days"); the rest are the query
    set the model is scored on. The incidence scaler is fit on the pooled support
    observations only (per-disease), because 2 weeks per node is far too few for
    a stable per-node scaler — and fitting on anything beyond support would leak.
    Recommendation for the main experiment: `countries=EBOLA_CORE_COUNTRIES`."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    if countries is not None:
        keep = {_canon(c) for c in countries}
        df = df[df[cols["country"]].map(_canon).isin(keep)].copy()
    df["_cd"] = df[cols["country"]].map(_canon) + "|" + df[cols["district"]].map(_canon)

    cases = df[df[cols["indicator"]].map(_canon) == "cases"]
    deaths = df[df[cols["indicator"]].map(_canon) == "deaths"]

    inc, msk = cumulative_to_weekly_incidence(cases, "_cd", cols["date"], cols["value"])
    dth, _ = cumulative_to_weekly_incidence(deaths, "_cd", cols["date"], cols["value"])

    # region exclusion (match full id OR the district part) + small-node drop
    if exclude_regions:
        exc = {_canon(x) for x in exclude_regions}
        keep = [n for n in inc.index
                if n not in exc and n.split("|", 1)[-1] not in exc]
        inc = inc.loc[keep]
        msk = msk.loc[keep]
    if drop_below_total is not None:
        keep = inc.index[inc.sum(axis=1) >= drop_below_total]
        inc, msk = inc.loc[keep], msk.loc[keep]
    assert len(inc) > 0, "No Ebola nodes left after region filtering."

    node_ids = list(inc.index)
    dates = pd.DatetimeIndex(inc.columns)
    dth = dth.reindex(index=node_ids, columns=inc.columns).fillna(0.0)

    raw = inc.to_numpy(dtype=np.float64)
    mask = msk.to_numpy().astype(np.uint8)
    deaths_raw = dth.to_numpy(dtype=np.float64)

    # Support = first `few_shot_support_weeks` OBSERVED weeks per district.
    support_mask = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        obs_idx = np.where(mask[i] == 1)[0][:max(0, few_shot_support_weeks)]
        support_mask[i, obs_idx] = 1

    return _finalise(raw, mask, node_ids, dates, ratios,
                     disease="ebola", adm_level="Admin2", t_res="weekly",
                     role="few_shot_holdout", support_mask=support_mask,
                     extended_raw={"deaths_norm": deaths_raw},
                     shapefile=shapefile, name_field=cols["district"],
                     source="HDX ebola-cases-2014 (WHO sit-reps); "
                            "NEJM 2014 10.1056/NEJMoa1411100")


# --------------------------------------------------------------------------- #
# INFLUENZA — ColaGNN-shipped [T,N] matrix + adjacency (fit-free graph)
# --------------------------------------------------------------------------- #
def load_influenza(matrix_path: str, adj_path: str, dataset: str = "japan",
                   start_date: Optional[str] = None, ratios=(0.5, 0.2, 0.3),
                   header=None) -> DiseaseTensors:
    """ColaGNN influenza: `matrix_path` is [T,N] ILI counts, `adj_path` its
    shipped adjacency. Graph is reused verbatim (comparability). Matrices ship
    without dates, so `start_date` (first epi-week) is resolved per dataset from
    INFLUENZA_START; override if the shipped file's first week differs."""
    if start_date is None:
        start_date = INFLUENZA_START.get(_canon(dataset), "2012-01-01")
    mat = pd.read_csv(matrix_path, header=header).to_numpy(dtype=np.float64)  # [T,N]
    raw = mat.T                                                              # [N,T]
    N, T = raw.shape
    A_geo = pd.read_csv(adj_path, header=None).to_numpy(dtype=np.float32)
    assert A_geo.shape == (N, N), f"shipped adjacency {A_geo.shape} != N={N}"

    node_ids = [f"{dataset}_{i}" for i in range(N)]
    dates = pd.date_range(start_date, periods=T, freq=WEEK_ANCHOR)
    mask = np.ones((N, T), dtype=np.uint8)

    return _finalise(raw, mask, node_ids, dates, ratios,
                     disease=f"influenza:{dataset}", adm_level="Admin1",
                     t_res="weekly", A_geo=A_geo, A_geo_kind="shipped",
                     source="ColaGNN (CIKM 2020) shipped influenza benchmark")


# --------------------------------------------------------------------------- #
# Shared finaliser: normalise (leakage-safe), assemble X, pack meta, check.
# --------------------------------------------------------------------------- #
def _finalise(raw, mask, node_ids, dates, ratios, disease, adm_level, t_res,
              role="development", extended_raw=None, support_mask=None,
              A_geo=None, A_geo_kind="pending", shapefile=None, name_field=None,
              source="") -> DiseaseTensors:
    N, T = raw.shape
    train_end, val_end = chronological_split(T, ratios)
    few_shot = (role == "few_shot_holdout" and support_mask is not None)

    # Choose the leakage-safe fit set: TRAIN slice for dev diseases, the few-shot
    # SUPPORT set for the held-out disease (pooled per-disease — 2 wk/node is too
    # few for a stable per-node scaler).
    if few_shot:
        fit_mask = mask.astype(bool) & support_mask.astype(bool)
        scaler = fit_scalers_masked(raw, fit_mask, per_disease=True)
    else:
        fit_mask = None
        scaler = fit_scalers(raw, mask, train_end)

    # Neutral imputation: imputed steps -> 0 in normalised space (per-node mean),
    # NOT the transform of a raw 0. The obs_mask channel flags them (D15).
    inc_norm = np.where(mask == 1, apply_scaler(raw, scaler), 0.0).astype(np.float32)

    extended = None
    if extended_raw:
        extended = {}
        for nm, arr in extended_raw.items():
            if few_shot:
                sc = fit_scalers_masked(arr, fit_mask, per_disease=True)
            else:
                sc = fit_scalers(arr, mask, train_end)   # own leakage-safe scaler
            extended[nm] = np.where(mask == 1, apply_scaler(arr, sc), 0.0).astype(np.float32)

    X, feature_names = _assemble_X(inc_norm, dates, mask, extended)
    y = inc_norm.copy()

    if A_geo is None:
        if shapefile:
            A_geo = build_adjacency(shapefile, node_ids, name_field)
            A_geo_kind = "queen+knn"
        else:
            A_geo = np.zeros((N, N), dtype=np.float32)   # deferred; build in Week 2
            A_geo_kind = "deferred"

    steps_per_year = 52 if t_res == "weekly" else 12
    if few_shot:
        # Real support/query split (no longer provisional): support = the fitted
        # few weeks; query = observed & not support.
        query_mask = (mask.astype(bool) & ~support_mask.astype(bool)).astype(np.uint8)
        split = {"scheme": "few_shot_support_query",
                 "support_weeks": int(support_mask.sum(1).max()),
                 "support_mask": support_mask.astype(np.uint8),
                 "query_mask": query_mask}
        split_scheme = "few_shot_support_query"
    else:
        split = {"train_end": train_end, "val_end": val_end}
        split_scheme = "fixed_50_20_30"

    meta = dict(
        disease=disease, role=role, node_ids=node_ids, adm_level=adm_level,
        dates=list(dates), t_res=t_res, steps_per_year=steps_per_year,
        feature_names=feature_names, core_feature_idx=CORE_FEATURE_IDX,
        scaler=scaler, scaler_scope=("per_disease_support" if few_shot else "per_node_train"),
        split=split, split_scheme=split_scheme,
        A_geo_kind=A_geo_kind, A_mob_available=False,
        # D20: covariates deferred by client ("manage later"); C is a flagged
        # placeholder to be populated (centroid lat/lon, area) in Week 2.
        covariates_status="PLACEHOLDER — populate lat/lon/area in Week 2 (D20 deferred)",
        shapefile_ref=shapefile, source=source,
    )
    return DiseaseTensors(X=X, A_geo=A_geo, C=np.zeros((N, 3), np.float32),
                          M=mask, y=y, meta=meta).check()

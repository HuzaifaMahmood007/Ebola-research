"""
to_schema.py — Standardised, disease-agnostic input schema loaders.

Coerces dengue (OpenDengue), influenza (ColaGNN-shipped matrices) and Ebola
(HDX sub-national, cumulative) into one `DiseaseTensors` bundle:

    {X:[N,T,F], A_geo:[N,N], A_mob:[N,N]|None, C:[N,S], M:[N,T], y:[N,T], meta}

Adjacency and static covariates need geopandas (imported lazily); the core tensor
logic does not.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _sha256(path: str) -> str:
    """SHA-256 of a file, recorded in meta as provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Feature layout. Only the core block is transfer-safe; `obs_mask` is core so the
# encoder can tell imputed steps from observed ones.
CORE_FEATURES = ["incidence_norm", "sin_doy", "cos_doy", "obs_mask"]
CORE_FEATURE_IDX = [0, 1, 2, 3]

# MMWR epi-weeks run Sunday->Saturday, so 'W-SAT' is the resample/label anchor.
WEEK_ANCHOR = "W-SAT"


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
@dataclass
class DiseaseTensors:
    """One disease's harmonised bundle.

    `raw` is the unscaled weekly incidence the scaler was fit from. It is carried so a
    scaler can be refit (rolling origins) or audited without inverting `y`.
    """

    X: np.ndarray                    # [N, T, F] float32
    A_geo: np.ndarray                # [N, N]   float32
    C: np.ndarray                    # [N, S]   float32
    M: np.ndarray                    # [N, T]   uint8 (1 observed / 0 missing)
    y: np.ndarray                    # [N, T]   float32 (model space)
    meta: dict = field(default_factory=dict)
    A_mob: Optional[np.ndarray] = None   # [N, N] float32 or None
    raw: Optional[np.ndarray] = None     # [N, T] float32, unscaled incidence

    def check(self) -> "DiseaseTensors":
        """Assert the schema invariants. Cheap; call it everywhere."""
        N, T, F = self.X.shape
        assert self.A_geo.shape == (N, N), "A_geo must be [N,N]"
        assert self.C.shape[0] == N, "C must have N rows"
        assert self.M.shape == (N, T), "M must be [N,T]"
        assert self.y.shape == (N, T), "y must be [N,T]"
        if self.raw is not None:
            assert self.raw.shape == (N, T), "raw must be [N,T]"
        assert F >= len(CORE_FEATURES), "X must contain the core channels"
        assert len(self.meta["node_ids"]) == N, "node_ids must align to N"
        assert len(self.meta["dates"]) == T, "dates must align to T"
        assert self.M.dtype == np.uint8, "M must be uint8 (1 obs / 0 missing)"
        if self.A_mob is not None:
            assert self.A_mob.shape == (N, N), "A_mob must be [N,N]"
        assert self.X.dtype == np.float32
        return self

    def transfer_view(self) -> np.ndarray:
        """The core channels — the only block the shared cross-disease encoder may see.
        Extended channels (deaths, C) are single-disease and are excluded here."""
        return self.X[:, :, self.meta["core_feature_idx"]]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _canon(s) -> str:
    """Canonical id: strip accents/diacritics, lowercase, collapse whitespace.
    Geometry joins to series on names, not codes, so 'Guéckédou' must equal 'Gueckedou'."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


# First MMWR epi-week per ColaGNN dataset. The shipped matrices are undated, so these
# anchors were recovered from the series and are asserted on every build. Both US sets
# start at MMWR week 40 (the CDC flu-season boundary), NOT January: a January anchor puts
# the calendar 39 weeks out and inverts the sin_doy/cos_doy phase.
INFLUENZA_START = {
    "us-regions": "2002-10-05", "region": "2002-10-05",
    "us-states": "2010-10-09", "state": "2010-10-09",
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
                       per_disease: bool = False,
                       groups: Optional[Sequence[str]] = None) -> dict:
    """Fit log1p+zscore stats using only the cells where `fit_mask` is true.

    The leakage-safe primitive: pass a train-slice mask for development diseases, or a
    few-shot support mask for the held-out disease. `per_disease=True` pools every cell
    into one (mean, std), for when each node has too few fit cells for a stable per-node
    scale (a 2-week Ebola support set).

    `groups` (one label per node, e.g. its country) supplies a fallback for nodes the per-node
    fit cannot scale: those with no fit cell, and those whose fit cells are constant (zero
    variance in the train window, though not necessarily in the held-out period). Both would
    otherwise get sd=1 and enter the model un-normalised, on a different scale from every
    neighbour. They instead take their group's pooled statistics, computed from that group's
    fit cells only and so still visible at train time.
    """
    N, T = raw_counts.shape
    logc = np.log1p(np.clip(raw_counts, 0, None))
    fm = fit_mask.astype(bool)
    if per_disease:
        flat = logc[fm]
        mu = np.full(N, flat.mean() if flat.size else 0.0)
        sd = np.full(N, flat.std() if flat.size else 1.0)
    else:
        mu, sd = np.zeros(N), np.ones(N)
        degenerate = []                              # no fit cells, or a constant fit window
        for i in range(N):
            v = logc[i][fm[i]]
            if v.size:
                mu[i], sd[i] = v.mean(), v.std()
            if not v.size or sd[i] < 1e-8:
                degenerate.append(i)
        if degenerate and groups is not None:
            by_group = {}                            # pool only nodes that actually vary
            for i in range(N):
                v = logc[i][fm[i]]
                if v.size and v.std() >= 1e-8:
                    by_group.setdefault(groups[i], []).append(v)
            for i in degenerate:
                pool = by_group.get(groups[i])
                if pool:
                    v = np.concatenate(pool)
                    mu[i], sd[i] = v.mean(), v.std()
    sd[sd < 1e-8] = 1.0  # last resort: a whole group constant in its fit window
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
    """Raw counts -> model space (log1p + per-node z-score)."""
    logc = np.log1p(np.clip(raw_counts, 0, None))
    return ((logc - scaler["mean"][:, None]) / scaler["std"][:, None]).astype(np.float32)


def invert_scaler(norm: np.ndarray, scaler: dict) -> np.ndarray:
    """Model space -> real counts, for RMSE/MAE reporting."""
    logc = norm * scaler["std"][:, None] + scaler["mean"][:, None]
    return np.expm1(logc)


def cumulative_to_weekly_incidence(df: pd.DataFrame, node_col: str, date_col: str,
                                   value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative sit-report counts -> weekly new incidence.

    Resample to the epi-week (last cumulative per week), forward-fill, then difference the
    running maximum. Forward-filled weeks are masked, as are weeks whose report falls below
    the running maximum, which are corrupt reports rather than observed zeros.

    Each node's FIRST observed week is masked: at a node's first report there is no
    preceding cumulative, so the increment is not identifiable. Assigning C_0 there
    would instead dump the node's entire back-log into week 0.

    Increments sum to max(C) - C_first per node, which cumulative_reference_mass() computes
    independently and the leakage suite checks.

    Returns (incidence[N,T], mask[N,T]), both indexed by node and week.
    """
    df = df.copy()
    df["_node"] = df[node_col].map(_canon)
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    # Non-numeric Value cells (' ', '-') are not reports. Must be dropped BEFORE the
    # de-dup below, or keep="last" lets one overwrite that week's real cumulative count.
    df = df.dropna(subset=["_date", value_col])

    inc_frames, mask_frames = {}, {}
    for node, g in df.groupby("_node"):
        s = g.set_index("_date")[value_col].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        weekly_obs = s.resample(WEEK_ANCHOR).last()  # last cumulative in each epi-week
        observed = weekly_obs.notna()                # weeks with an actual report
        weekly_cum = weekly_obs.ffill()

        # A cumulative count cannot fall. Where the report does fall, the report is wrong --
        # a single-week data-entry dropout, not a downward revision. Differencing the monotone
        # envelope absorbs it: the week itself yields no increment, and the recovery yields
        # only the rise above the previous high-water mark. Clipping the difference instead
        # (max(0, dC)) would release the whole climb back to the prior level as new cases,
        # which is how one Sierra Leonean district came to carry 2.5x the cases it ever
        # recorded. The envelope conserves mass by construction: the increments sum to
        # max(C) - C_first, which is what cumulative_reference_mass() checks.
        envelope = weekly_cum.cummax()
        new = envelope.diff()
        # Weeks whose report sits below the running maximum are corrupt, not observations of
        # zero. Masked, so they are neither scored nor fed to a scaler.
        dropout = observed & (weekly_obs < envelope.shift())
        observed = observed & ~dropout

        new.iloc[0] = 0.0                            # no preceding cumulative ->
        observed.iloc[0] = False                     # week 0 is not an observation
        inc_frames[node] = new
        mask_frames[node] = observed.astype(np.uint8)

    inc = pd.DataFrame(inc_frames).T                # [N, T]
    msk = pd.DataFrame(mask_frames).T.reindex_like(inc).fillna(0).astype(np.uint8)
    inc = inc.fillna(0.0)
    return inc, msk


def cumulative_reference_mass(df: pd.DataFrame, node_col: str, date_col: str,
                              value_col: str) -> dict[str, float]:
    """Per node, max(C) - C_first on the epi-week grid: the total new cases a cumulative series
    can imply.

    Gives the leakage suite a mass reference the increment logic did not produce. It shares the
    weekly resampling with cumulative_to_weekly_incidence -- both must agree on what a week is --
    but nothing else: no differencing, no envelope, no clipping. That is enough, because the
    fabrication this guards against lives entirely in the increment step. A weekly series summing
    to more than this has invented cases.
    """
    d = df.copy()
    d["_node"] = d[node_col].map(_canon)
    d["_date"] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["_date", value_col])
    ref = {}
    for node, g in d.groupby("_node"):
        s = g.set_index("_date")[value_col].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        weekly = s.resample(WEEK_ANCHOR).last().dropna()
        ref[node] = float(weekly.max() - weekly.iloc[0])
    return ref


def _assemble_X(inc_norm: np.ndarray, dates: pd.DatetimeIndex, obs_mask: np.ndarray,
                extended: Optional[dict] = None) -> tuple[np.ndarray, list]:
    """Stack the core channels [incidence, sin, cos, obs_mask], then any extended ones."""
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


def per_country_chronological_split(node_ids: Sequence[str], obs_mask: np.ndarray,
                                    ratios=(0.5, 0.2, 0.3),
                                    country_of=None) -> dict:
    """Chronological 50/20/30, cut per country on that country's own observed weeks.

    A single global positional cut on the union calendar would starve every country whose
    reporting begins after it (zero train cells, hence an undefined scaler). Cutting each
    country separately gives every one a real train/val/test, and all of a country's nodes
    share the boundary, so at any week a whole country-block is in one phase.

    Only observed cells are assigned, and every node of a country takes that country's
    boundary without exception. A node whose reporting begins after its country's train
    boundary therefore gets no train cell: it is scored, and remains a neighbour in the
    graph, but its scaler comes from its country's pooled train statistics rather than from
    its own history (fit_scalers_masked(groups=...)). Re-cutting such a node on its own weeks
    would put its train cells inside its neighbours' test period, and a GNN aggregates over
    neighbours -- which is the leak this split exists to prevent.

    Returns dict(train_mask, val_mask, test_mask : [N,T] uint8,
                 bounds : {country -> (train_end_col, val_end_col)},
                 nodes_without_train : [node ids scored but never trained on]).
    """
    if country_of is None:
        country_of = lambda n: n.split("|", 1)[0]
    N, T = obs_mask.shape
    train = np.zeros((N, T), np.uint8)
    val = np.zeros((N, T), np.uint8)
    test = np.zeros((N, T), np.uint8)

    def _cut(cols, ratios):
        n = len(cols)
        tr, va = chronological_split(n, ratios)
        return set(cols[:tr]), set(cols[tr:va]), set(cols[va:])

    groups = {}
    for i, n in enumerate(node_ids):
        groups.setdefault(country_of(n), []).append(i)

    bounds = {}
    for c, idx in groups.items():
        obs_cols = np.where(obs_mask[idx].any(axis=0))[0]
        if obs_cols.size == 0:
            continue
        tr_cols, va_cols, _ = _cut(list(obs_cols), ratios)
        bt = max(tr_cols) if tr_cols else -1
        bv = max(va_cols) if va_cols else bt
        bounds[c] = (int(bt), int(bv))
        for i in idx:
            row = np.where(obs_mask[i] == 1)[0]
            if row.size == 0:
                continue
            for t in row:
                if t in tr_cols:
                    train[i, t] = 1
                elif t in va_cols:
                    val[i, t] = 1
                else:
                    test[i, t] = 1

    no_train = [node_ids[i] for i in range(N)
                if obs_mask[i].any() and not train[i].any()]
    return dict(train_mask=train, val_mask=val, test_mask=test, bounds=bounds,
                nodes_without_train=no_train)


# --------------------------------------------------------------------------- #
# Adjacency (GADM 4.1 shapefiles)
# --------------------------------------------------------------------------- #

# ISO3 codes for the locked dengue development set.
GADM_ISO3 = {
    "argentina": "ARG", "bolivia": "BOL", "brazil": "BRA", "colombia": "COL",
    "dominican republic": "DOM", "ecuador": "ECU", "japan": "JPN", "mexico": "MEX",
    "nicaragua": "NIC", "panama": "PAN", "peru": "PER", "taiwan": "TWN",
}

# Countries aggregated to their parent unit because GADM has no polygon at the level the
# data reports. Taiwan reports 287 townships; GADM's deepest Taiwanese layer is the 22
# counties. Townships partition counties, so summing them is exact.
DENGUE_AGGREGATE_TO_PARENT = {"taiwan"}

# Countries whose GADM key is not the usual level/field pair. Taiwan's counties sit at level
# 2, but its NAME_1 is a legacy province layer absent from the data, so NAME_2 alone keys it.
GADM_KEY_OVERRIDE = {"taiwan": (2, ["NAME_2"])}


def _gadm_gdf(gadm_dir: str, iso3: str, level: str, country: Optional[str] = None):
    """Load a GADM country layer, keyed the way dengue node ids are: Admin1 -> 'name_1',
    Admin2 -> 'name_1|name_2'. The Admin2 key carries the parent because leaf names repeat
    within a country (Brazil has 239 municipality names used in 2-5 states each)."""
    import geopandas as gpd
    zp = Path(gadm_dir) / f"gadm41_{iso3}_shp.zip"
    if not zp.exists():
        raise FileNotFoundError(
            f"GADM 4.1 shapefile not found: {zp}. Download from "
            f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso3}_shp.zip")
    if country in GADM_KEY_OVERRIDE:
        lvl, fields = GADM_KEY_OVERRIDE[country]
    else:
        lvl = {"Admin1": 1, "Admin2": 2}[level]
        fields = ["NAME_1"] if level == "Admin1" else ["NAME_1", "NAME_2"]
    gdf = gpd.read_file(zp, layer=f"gadm41_{iso3}_{lvl}")
    key = gdf[fields[0]].map(_canon)
    for f in fields[1:]:
        key = key + "|" + gdf[f].map(_canon)
    gdf["_key"] = key
    return gdf


def _centroid_area(gdf) -> np.ndarray:
    """[centroid_lat, centroid_lon, area_km2] per polygon, in the gdf's row order.

    Computed in an equal-area projection and only then converted back to degrees: a degree
    is not a unit of area, and the distortion grows with latitude.
    """
    import geopandas as gpd
    ea = gpd.GeoSeries(gdf.geometry.values, crs=gdf.crs).to_crs(EQUAL_AREA_CRS)
    cent = ea.centroid.to_crs("EPSG:4326")
    return np.column_stack([cent.y.to_numpy(),            # latitude  (deg)
                            cent.x.to_numpy(),            # longitude (deg)
                            ea.area.to_numpy() / 1e6      # area (km^2)
                            ]).astype(np.float32)


def _contiguity(gdf, knn_fallback: int = 4) -> np.ndarray:
    """Queen contiguity over the given polygons (row order = output order), with a
    k-NN centroid fallback so islands are never degenerate for message passing.
    Symmetrised: Queen is symmetric but KNN is not, and a GNN needs A == A.T."""
    from libpysal.weights import Queen, KNN
    n = len(gdf)
    A = np.zeros((n, n), dtype=np.float32)
    if n < 2:
        return A
    w = Queen.from_dataframe(gdf, use_index=False)
    for i, neighbours in w.neighbors.items():
        for j in neighbours:
            A[i, j] = 1.0
    isolated = np.where(A.sum(1) == 0)[0]
    if len(isolated):
        wk = KNN.from_dataframe(gdf, k=min(knn_fallback, n - 1))
        for i in isolated:
            for j in wk.neighbors[i]:
                A[i, j] = 1.0
    np.fill_diagonal(A, 0.0)
    return np.maximum(A, A.T)


def build_dengue_adjacency(node_ids: Sequence[str], country_levels: dict,
                           gadm_dir: str, knn_fallback: int = 4,
                           name_aliases: Optional[dict] = None,
                           gadm_fix: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Block-diagonal adjacency for the mixed-level dengue graph, plus its covariates C.

    Contiguity is built within each country and assembled block-diagonally: there are no
    cross-border edges, because dengue's countries are not adjacent. Each country sits at
    one level and joins GADM at that level, so no single global name field can serve the
    graph:

        Admin1 node 'japan|tokyo'             -> GADM NAME_1
        Admin2 node 'brazil|piaui|bom jesus'  -> GADM (NAME_1, NAME_2)

    `name_aliases` is per country -- {country: {node_tail: gadm_key}} -- keyed on the
    node-id tail ('adm1' at Admin1, 'adm1|adm2' at Admin2).

    `gadm_fix` bends the lookup key to match a misspelling GADM itself ships. It is applied
    here at the join and nowhere else, so our node ids never inherit GADM's typos.

    Any node that cannot be matched raises with the full diagnostic; nothing is dropped
    silently. C is built in this same pass because the polygons are already reindexed into
    node order here, which is what guarantees C's rows align to node_ids.

    Returns (A [N,N], C [N,3], report of per-country node/edge/isolate counts).
    """
    N = len(node_ids)
    A = np.zeros((N, N), dtype=np.float32)
    C = np.zeros((N, 3), dtype=np.float32)
    aliases = {_canon(c): {_canon(k): _canon(v) for k, v in m.items()}
               for c, m in (name_aliases or {}).items()}
    gfixes = {_canon(c): {_canon(k): _canon(v) for k, v in m.items()}
              for c, m in (gadm_fix or {}).items()}

    by_country: dict = {}
    for i, n in enumerate(node_ids):
        by_country.setdefault(n.split("|", 1)[0], []).append(i)

    unmatched, report = {}, {}
    for country, idx in sorted(by_country.items()):
        if country not in GADM_ISO3:
            raise KeyError(f"No GADM ISO3 mapping for country {country!r}; add it to GADM_ISO3.")
        level = country_levels[country]
        gdf = _gadm_gdf(gadm_dir, GADM_ISO3[country], level, country=country)
        amap = aliases.get(country, {})
        gfix = gfixes.get(country, {})

        pos = {}                                  # gadm key -> row (first wins)
        for r, k in enumerate(gdf["_key"]):
            pos.setdefault(k, r)

        rows, miss = [], []
        for i in idx:
            tail = node_ids[i].split("|", 1)[1]   # 'adm1' | 'adm1|adm2'
            key = amap.get(tail, tail)
            r = pos.get(gfix.get(key, key))       # gadm_fix bends the key, not the node id
            (miss if r is None else rows).append(node_ids[i] if r is None else r)
        if miss:
            unmatched[country] = miss
            continue

        block_gdf = gdf.iloc[rows].reset_index(drop=True)   # polygons in node order
        block = _contiguity(block_gdf, knn_fallback)
        ia = np.asarray(idx)
        A[ia[:, None], ia[None, :]] = block
        C[ia] = _centroid_area(block_gdf)                   # same rows -> same order
        report[country] = dict(level=level, n_nodes=len(idx),
                               n_edges=int(block.sum() // 2),
                               n_isolated=int((block.sum(1) == 0).sum()))

    if unmatched:
        total = sum(len(v) for v in unmatched.values())
        detail = "\n".join(
            f"  {c} ({len(v)} unmatched): {v[:20]}{' ...' if len(v) > 20 else ''}"
            for c, v in sorted(unmatched.items()))
        raise ValueError(
            f"GADM name-join failed for {total} node(s) across {len(unmatched)} country/ies. "
            f"Add 'data_key -> gadm_key' entries to name_aliases[country]; keys are the "
            f"node-id tail ('adm1' at Admin1, 'adm1|adm2' at Admin2):\n{detail}")

    return A, C, report


# --------------------------------------------------------------------------- #
# DENGUE — OpenDengue best-spatial extract (16-col layout from the sample)
# --------------------------------------------------------------------------- #
DENGUE_COLS = dict(
    adm0="adm_0_name", adm1="adm_1_name", adm2="adm_2_name",
    start="calendar_start_date", value="dengue_total",
    t_res="T_res", s_res="S_res", case_def="case_definition_standardised",
)

# WHO/CDC dengue-endemic countries, canonicalised to the adm_0_name spellings the extract
# actually uses. This is only the CANDIDATE POOL: the loader keeps a country only if it
# clears the weekly subnational coverage thresholds, so the final set is data-driven.
DENGUE_ENDEMIC = [
    # Africa
    "angola", "benin", "burkina faso", "cabo verde", "cameroon",
    "central african republic", "chad", "cote d'ivoire", "eritrea", "ethiopia",
    "ghana", "guinea", "kenya", "mali", "mauritania", "mauritius", "mayotte",
    "reunion", "sao tome and principe", "senegal", "seychelles", "sudan",
    "togo", "united republic of tanzania",
    # Americas & Caribbean
    "anguilla", "antigua and barbuda", "argentina", "aruba", "bahamas",
    "barbados", "belize", "bermuda", "bolivia",
    "bonaire, saint eustatius and saba", "brazil", "cayman islands", "chile",
    "colombia", "costa rica", "cuba", "curacao", "dominica",
    "dominican republic", "ecuador", "el salvador", "french guiana", "grenada",
    "guadeloupe", "guatemala", "guyana", "haiti", "honduras", "jamaica",
    "martinique", "mexico", "montserrat", "nicaragua", "panama", "paraguay",
    "peru", "puerto rico", "saint barthelemy", "saint kitts and nevis",
    "saint lucia", "saint martin", "saint vincent and the grenadines",
    "sint maarten", "suriname", "trinidad and tobago",
    "turks and caicos islands", "uruguay", "venezuela", "virgin islands (uk)",
    "virgin islands (us)",
    # Asia
    "afghanistan", "bangladesh", "bhutan", "brunei darussalam", "cambodia",
    "hong kong", "india", "indonesia", "japan",
    "lao people's democratic republic", "macau", "malaysia", "myanmar", "nepal",
    "oman", "philippines", "saudi arabia", "singapore", "sri lanka", "taiwan",
    "thailand", "timor-leste", "viet nam", "yemen",
    # Europe & Pacific
    "france", "guam", "italy", "kiribati", "marshall islands", "niue", "spain",
    "tokelau",
]


def _node_col_for(level: str, cols: dict = DENGUE_COLS) -> str:
    """Map a subnational level to its node-name column, raising on anything else.

    Admin0 is deliberately absent and must raise rather than default: a national aggregate
    is a singleton with no intra-country neighbours, so it is never a node.
    """
    try:
        return {"Admin1": cols["adm1"], "Admin2": cols["adm2"]}[level]
    except KeyError:
        raise ValueError(
            f"Unsupported dengue node level {level!r}; expected 'Admin1' or 'Admin2' "
            f"(Admin0 is not a node level).")


def resolve_country_levels(csv_path: str, countries=None, t_res_filter: str = "Week",
                           min_weeks: int = 52, min_nodes_per_country: int = 3,
                           cols: dict = DENGUE_COLS) -> tuple[dict, dict]:
    """Choose one subnational level per candidate country: the FINEST that clears the gate.

    Admin2 if at least `min_nodes_per_country` of its Admin2 nodes each clear `min_weeks`
    observed weeks, else Admin1 if that clears, else exclude the country. Admin0 is never a
    node level. The extract resolves each location-period to a single level, so one level per
    country cannot double-count a parent province against its own child districts.

    Returns (levels, report):
      levels : {country -> 'Admin1'|'Admin2'}, kept countries only
      report : {country -> dict(n_nodes_admin1, n_usable_admin1, n_nodes_admin2,
                n_usable_admin2, chosen | excluded_reason)}, where excluded_reason is
                'admin0_only', 'insufficient_coverage' or 'absent_from_file'.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    if t_res_filter is not None and cols["t_res"] in df:
        df = df[df[cols["t_res"]].map(_canon) == _canon(t_res_filter)]
    df = df.copy()
    df["_c"] = df[cols["adm0"]].map(_canon)
    df["_lvl"] = df[cols["s_res"]].map(_canon)
    df["_wk"] = pd.to_datetime(df[cols["start"]],
                               errors="coerce").dt.to_period(WEEK_ANCHOR)
    df["_val"] = pd.to_numeric(df[cols["value"]], errors="coerce")
    df = df.dropna(subset=["_wk"])
    df = df[df["_val"].notna()]            # a week counts as observed only if reported

    if countries is None:
        cand = set(DENGUE_ENDEMIC)
    elif isinstance(countries, str):
        cand = {_canon(countries)}
    else:
        cand = {_canon(c) for c in countries}
    present = set(df["_c"].unique())

    def _usable(c: str, lvl_canon: str, col: str) -> tuple[int, int]:
        g = df[(df["_c"] == c) & (df["_lvl"] == lvl_canon)].copy()
        if not len(g):
            return 0, 0
        g["_n"] = g[col].map(_canon)
        g = g[~g["_n"].isin({"na", "nan", ""})]
        # The Admin2 key must carry the parent adm1: leaf names repeat within a country, and
        # a leaf-only key pools distinct municipalities, overstating coverage here.
        if lvl_canon == _canon("Admin2"):
            g["_p"] = g[cols["adm1"]].map(_canon)
            g = g[~g["_p"].isin({"na", "nan", ""})]
            # Aggregated countries count parent units, not the leaves they sum from.
            g["_n"] = g["_p"] if c in DENGUE_AGGREGATE_TO_PARENT else g["_p"] + "|" + g["_n"]
        if not len(g):
            return 0, 0
        obs = g.groupby("_n")["_wk"].nunique()
        return int(obs.size), int((obs >= min_weeks).sum())

    order = ["Admin2", "Admin1"]                 # finest-available
    levels, report = {}, {}
    for c in sorted(cand):
        if c not in present:
            report[c] = dict(excluded_reason="absent_from_file")
            continue
        n1, u1 = _usable(c, "admin1", cols["adm1"])
        n2, u2 = _usable(c, "admin2", cols["adm2"])
        rep = dict(n_nodes_admin1=n1, n_usable_admin1=u1,
                   n_nodes_admin2=n2, n_usable_admin2=u2)
        usable_by = {"Admin1": u1, "Admin2": u2}
        chosen = next((lvl for lvl in order
                       if usable_by[lvl] >= min_nodes_per_country), None)
        if chosen is not None:
            levels[c] = chosen
            rep["chosen"] = chosen
        else:
            rep["excluded_reason"] = ("insufficient_coverage" if (n1 + n2) > 0
                                      else "admin0_only")
        report[c] = rep
    return levels, report


def dengue_country_coverage(csv_path: str, level: str = "Admin1",
                            t_res_filter: str = "Week",
                            cols: dict = DENGUE_COLS) -> pd.DataFrame:
    """Audit helper: per-country weekly subnational coverage in the extract
    (n_nodes, max weeks per node, date span, endemic flag)."""
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


def load_dengue(csv_path: str, countries=None, level: str = "auto",
                t_res_filter: str = "Week", cols: dict = DENGUE_COLS,
                include_regions: Optional[Sequence[str]] = None,
                exclude_regions: Optional[Sequence[str]] = None,
                min_weeks: int = 52, min_nodes_per_country: int = 3,
                ratios=(0.5, 0.2, 0.3), gadm_dir: Optional[str] = None,
                name_aliases: Optional[dict] = None,
                gadm_fix: Optional[dict] = None,
                unmappable: Optional[dict] = None,
                exclude_countries: Optional[Sequence[str]] = None) -> DiseaseTensors:
    """OpenDengue -> DiseaseTensors, multi-country. `dengue_total` is per-period incidence,
    not cumulative, so it is never differenced. NA/suppressed values become M=0, not 0 counts.

    level:
      "auto" (default)  one level per country, finest available (resolve_country_levels).
      "Admin1"/"Admin2" force a single level everywhere (for ablations).

    countries: None -> the DENGUE_ENDEMIC pool intersected with the file; a str -> one
    country; a list -> exactly those. A country is kept only if at least
    `min_nodes_per_country` of its nodes clear `min_weeks`; the prune is at country level, so
    every node of a kept country is retained.

    Node ids are 'country|adm1' at Admin1 and 'country|adm1|adm2' at Admin2. The Admin2 key
    carries its parent because leaf names are not unique within a country.

    Adjacency: pass `gadm_dir` (holding gadm41_<ISO3>_shp.zip) to build the block-diagonal
    graph. Left None, A_geo stays a zero placeholder. `name_aliases` repairs the DATA and is
    applied here at load time; `gadm_fix` repairs GADM and is applied only at the join.

    Splits are per-country chronological 50/20/30, each country cut on its own observed span.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df["_c"] = df[cols["adm0"]].map(_canon)

    if countries is None:
        requested = set(DENGUE_ENDEMIC)
    elif isinstance(countries, str):
        requested = {_canon(countries)}
    else:
        requested = {_canon(c) for c in countries}
    if exclude_countries:
        requested -= {_canon(c) for c in exclude_countries}
    present = set(df["_c"].unique())
    requested_missing = sorted(requested - present)

    # --- resolve exactly one level per country ---
    if level == "auto":
        # pass `requested` (not None) so exclude_countries is honoured here too
        levels, level_report = resolve_country_levels(
            csv_path, countries=sorted(requested),
            t_res_filter=t_res_filter, min_weeks=min_weeks,
            min_nodes_per_country=min_nodes_per_country, cols=cols)
    else:
        _node_col_for(level, cols)               # validate (raises on Admin0/unknown)
        levels = {c: level for c in sorted(requested & present)}
        level_report = None
    sel = sorted(levels)
    assert sel, (f"No dengue country cleared level resolution "
                 f"(min_weeks={min_weeks}, min_nodes_per_country={min_nodes_per_country}).")

    # --- per-country level-correct row selection ---
    if t_res_filter is not None and cols["t_res"] in df:
        df = df[df[cols["t_res"]].map(_canon) == _canon(t_res_filter)]
    df = df[df["_c"].isin(sel)].copy()
    df["_lvl"] = df[cols["s_res"]].map(_canon)
    df["_want"] = df["_c"].map({c: _canon(levels[c]) for c in sel})
    df = df[df["_lvl"] == df["_want"]]           # keep only each country's chosen level
    assert len(df) > 0, "No dengue rows after per-country level selection."

    is_a2 = (df["_want"] == _canon("Admin2")).to_numpy()
    # Aggregated countries key on the parent; the pivot below sums their leaves into it.
    agg = is_a2 & df["_c"].isin(DENGUE_AGGREGATE_TO_PARENT).to_numpy()
    keyed = is_a2 & ~agg                                     # genuine Admin2 (parent-keyed)

    df["_adm"] = np.where(keyed, df[cols["adm2"]].map(_canon),
                          df[cols["adm1"]].map(_canon))      # leaf unit name
    df["_par"] = np.where(keyed, df[cols["adm1"]].map(_canon), "")   # parent adm1
    df["_nlvl"] = np.where(is_a2, "Admin2", "Admin1")
    df = _apply_node_filter(df, "_adm", include_regions, exclude_regions)
    df = df[~df["_adm"].isin(["na", "nan", ""])]
    df = df[~df["_par"].isin(["na", "nan"])]                 # "" is legitimate at Admin1

    # The Admin2 node id carries its parent. Keyed on the leaf alone, the pivot would sum
    # same-named municipalities from different states into one node and leave the GADM join
    # ambiguous.
    tail = pd.Series(np.where(df["_par"].to_numpy() != "", df["_par"] + "|" + df["_adm"],
                              df["_adm"]), index=df.index)

    # Aliases run BEFORE the pivot because they do two jobs: they make the geometry join
    # exact, AND they reunite units whose series the source split across two spellings in
    # disjoint eras. Summing the halves here restores one continuous series; a join-time map
    # could not, since the pivot has already separated them.
    # Per country: {country: {parents: {...}, units: {...}}}. Parents remap first, so a unit
    # key can be written against the already-corrected parent name.
    if name_aliases:
        pmap = {(_canon(c), _canon(k)): _canon(v)
                for c, m in name_aliases.items()
                for k, v in (m.get("parents") or {}).items()}
        umap = {(_canon(c), _canon(k)): _canon(v)
                for c, m in name_aliases.items()
                for k, v in (m.get("units") or {}).items()}
        if pmap:
            par = [pmap.get((c, p), p) for c, p in zip(df["_c"], df["_par"])]
            df["_par"] = par
            tail = pd.Series(np.where(df["_par"].to_numpy() != "",
                                      df["_par"] + "|" + df["_adm"], df["_adm"]),
                             index=df.index)
        if umap:
            tail = pd.Series([umap.get((c, t), t) for c, t in zip(df["_c"], tail)],
                             index=df.index)

    # Units with no GADM polygon and no defensible merge target. Dropped only from this
    # explicit, reasoned list; the graph builder itself never drops a node.
    dropped_unmappable = []
    if unmappable:
        umm = {(_canon(c), _canon(k)) for c, m in unmappable.items() for k in m}
        hit = np.array([(c, t) in umm for c, t in zip(df["_c"], tail)])
        if hit.any():
            dropped_unmappable = sorted({f"{c}|{t}" for c, t, h
                                         in zip(df["_c"], tail, hit) if h})
            df, tail = df[~hit], tail[~hit]

    df["_node"] = df["_c"] + "|" + tail
    df["_date"] = pd.to_datetime(df[cols["start"]], errors="coerce")
    df["_val"] = pd.to_numeric(df[cols["value"]], errors="coerce")
    df = df.dropna(subset=["_date", "_adm"])
    assert len(df) > 0, "No dengue rows after filtering (check country/level/t_res)."

    weekly = (t_res_filter is None and _canon(df[cols["t_res"]].iloc[0]) == "week") \
        or (t_res_filter is not None and _canon(t_res_filter) == "week")
    period_freq = WEEK_ANCHOR if weekly else "M"
    df["_bin"] = df["_date"].dt.to_period(period_freq)

    # Aggregation diagnostics. The pivot sums multiple rows per (node, period), which is only
    # legitimate for sub-period reports of the SAME case definition:
    #   within_week_summed -- (node, period) pairs carrying >1 row
    #   case_def_mixed     -- (node, period) pairs mixing definitions; must be 0
    within_week_summed = int(df.duplicated(subset=["_node", "_bin"]).sum())
    if cols.get("case_def") and cols["case_def"] in df:
        case_def_mixed = int((df.groupby(["_node", "_bin"])[cols["case_def"]]
                              .nunique() > 1).sum())
    else:
        case_def_mixed = 0
    node_level_map = dict(zip(df["_node"], df["_nlvl"]))

    wide = df.pivot_table(index="_node", columns="_bin", values="_val", aggfunc="sum")
    full_idx = pd.period_range(wide.columns.min(), wide.columns.max(), freq=period_freq)
    wide = wide.reindex(columns=full_idx)

    # coverage report + country-level prune
    node_country = {n: n.split("|", 1)[0] for n in wide.index}
    obs_per_node = wide.notna().sum(axis=1)
    coverage, kept_countries = {}, []
    for c in sel:
        c_nodes = [n for n in wide.index if node_country[n] == c]
        good = [n for n in c_nodes if obs_per_node[n] >= min_weeks]
        coverage[c] = dict(n_nodes=len(c_nodes), n_nodes_ge_min_weeks=len(good),
                           max_weeks=int(obs_per_node[c_nodes].max()) if c_nodes else 0,
                           level=levels[c])
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

    split_masks = per_country_chronological_split(node_ids, mask, ratios)

    adj_report, C_geo = None, None
    if gadm_dir:
        # name_aliases already ran above, so node tails are the corrected names and the join
        # is an identity lookup; anything unmatched raises. gadm_fix is applied inside, at the
        # join only. The same pass yields C, row-aligned to node_ids.
        A_geo, C_geo, adj_report = build_dengue_adjacency(
            node_ids, {c: levels[c] for c in kept_countries}, gadm_dir, gadm_fix=gadm_fix)
        A_geo_kind = "queen+knn_block_diagonal"
    else:
        A_geo, A_geo_kind = None, "deferred_block_diagonal"

    dt = _finalise(raw, mask, node_ids, dates, ratios,
                   disease="dengue",
                   adm_level=(level if level != "auto" else "per_country"),
                   t_res="weekly" if weekly else "monthly",
                   A_geo=A_geo, A_geo_kind=A_geo_kind, split_masks=split_masks,
                   source="OpenDengue; Clarke J et al., Sci Data 2024;11(1):296")

    dt.meta["countries"] = kept_countries
    dt.meta["country_levels"] = {c: levels[c] for c in kept_countries}
    dt.meta["node_levels"] = {n: node_level_map[n] for n in node_ids}
    dt.meta["node_country"] = {n: node_country[n] for n in node_ids}
    dt.meta["coverage"] = coverage
    dt.meta["level_report"] = level_report
    excluded = {}
    if level_report:
        for c, rep in level_report.items():
            reason = rep.get("excluded_reason")
            if reason and reason != "absent_from_file":
                excluded[c] = reason
    for c in sel:
        if c not in kept_countries:
            excluded.setdefault(c, "insufficient_coverage")
    dt.meta["countries_excluded"] = excluded
    dt.meta["countries_dropped_low_coverage"] = sorted(excluded)
    dt.meta["countries_requested_absent"] = requested_missing
    dt.meta["graph_is_block_diagonal"] = True
    dt.meta["within_week_summed_rows"] = within_week_summed
    dt.meta["case_def_mixed_node_weeks"] = case_def_mixed
    dt.meta["shapefile_source"] = "GADM 4.1 (gadm.org)" if gadm_dir else None
    dt.meta["adjacency_report"] = adj_report
    dt.meta["nodes_dropped_unmappable"] = dropped_unmappable
    dt.meta["aggregated_to_parent"] = sorted(
        set(kept_countries) & DENGUE_AGGREGATE_TO_PARENT)

    # Static covariates C — same contract as influenza:
    # C is raw/unstandardised: it is static, so it carries no leakage risk and scaling is the
    # encoder's job. It is NOT transfer-safe: the diseases occupy disjoint geography, so a raw
    # centroid identifies the disease. Single-disease use only; transfer_view() excludes it.
    if C_geo is not None:
        dt.C = C_geo
        dt.meta["covariates"] = ["centroid_lat", "centroid_lon", "area_km2"]
        dt.meta["covariates_status"] = "populated from GADM 4.1"
        dt.meta["covariates_source"] = "GADM 4.1 (gadm.org)"
        dt.meta["covariates_transfer_safe"] = False
        dt.check()
    return dt


# --------------------------------------------------------------------------- #
# EBOLA — OCHA ROWCA / HDX sub-national compilation (long, cumulative)
# --------------------------------------------------------------------------- #
EBOLA_COLS = dict(country="Country", district="Localite", indicator="Category",
                  value="Value", date="Date")
EBOLA_CORE_COUNTRIES = ["guinea", "liberia", "sierra leone"]
EBOLA_SHEET = "ROWCA Ebola All Sec Review"
# The outbreak window. The sheet carries Excel-epoch date artefacts outside it.
EBOLA_YEARS = (2014, 2016)

# Matches a Localite naming several districts at once ("Guekedou, Macenta and Kissidougou").
# Such a row is not a node: no polygon exists for it and its counts belong to districts we
# cannot separate. Detected structurally, so an UNDECLARED blob label halts the build rather
# than silently becoming a node.
_EBOLA_BLOB_RE = re.compile(r"[,()]|\band\b", flags=re.I)

# Labels that are never a district, in any country.
EBOLA_DROP_LABELS = {
    "national": "national aggregate, not a district",
}

# Declared node removals, each with its reason. The loader drops nothing else.
EBOLA_DROP_NODES = {
    # multi-district blob labels present in the sheet (early-outbreak aggregates)
    "guinea|( guekedou, macenta and kissidougou":
        "multi-district blob (early-outbreak aggregate)",
    "guinea|( guekedou, macenta and kissidougou)":
        "multi-district blob (early-outbreak aggregate)",
    "guinea|guekedou, macenta, nzerekore and kissidougou":
        "multi-district blob (early-outbreak aggregate)",
    "guinea|dabola and djingaraye":
        "multi-district blob (early-outbreak aggregate)",
    "guinea|( guekedou, macenta,kissidougou,lofa,kankan and conakry)":
        "multi-district blob (spans 2 countries)",
    # Sierra Leone units that are not districts
    "sierra leone|freetown":
        "city inside Western Area Urban, not a district; 1 report, no polygon",
    "sierra leone|western area":
        "parent of Western Area Urban+Rural and overlaps both in time; keeping it "
        "alongside its children would triple-count the same cases",
}

# Applied at LOAD time, before the pivot. These repair the DATA. Kept separate from the GADM
# join below so that a defect in GADM can never rename one of our nodes.
EBOLA_NAME_ALIASES = {
    "liberia": {
        # The source is internally inconsistent: 11 of the 15 counties carry a 'County'
        # suffix and 4 do not. Normalise to the bare name.
        "bomi county": "bomi", "bong county": "bong", "gbarpolu county": "gbarpolu",
        "lofa county": "lofa", "margibi county": "margibi",
        "maryland county": "maryland", "montserrado county": "montserrado",
        "nimba county": "nimba", "river gee county": "river gee",
        "rivercess county": "rivercess", "sinoe county": "sinoe",
    },
    "sierra leone": {
        # One district torn across two labels: 'port loko' reports to 2014-11-26 and 'port'
        # continues the SAME cumulative from 2014-11-27 (1041 -> 1923). The label was
        # truncated mid-compilation. The eras are disjoint, so the merge is unambiguous.
        "port": "port loko",
    },
}

# Applied at JOIN time only: our (correct) district name -> the key GADM actually ships.
# GADM's defects are GADM's and must not enter our node namespace.
EBOLA_GADM_FIX = {
    "guinea": {"yomou": "yamou"},            # GADM typo; the prefecture is Yomou
    "liberia": {"gbarpolu": "gbapolu"},      # GADM typo; the county is Gbarpolu
    "sierra leone": {                        # GADM drops the 'Area' from both names
        "western area urban": "western urban",
        "western area rural": "western rural",
    },
}

# GADM layer per country. The level is MIXED: Liberia's counties are level 1, while Guinea's
# prefectures and Sierra Leone's districts are level 2. District names are unique within a
# country, so the leaf field alone keys them (unlike dengue's Admin2).
EBOLA_GADM = {
    "guinea":       ("GIN", 2, "NAME_2"),   # 34 prefectures
    "liberia":      ("LBR", 1, "NAME_1"),   # 15 counties, level 1 not 2
    "sierra leone": ("SLE", 2, "NAME_2"),   # 14 districts
}


def _read_ebola(path: str) -> pd.DataFrame:
    """The production compilation ships as .xlsx (one sheet); fixtures use .csv."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=EBOLA_SHEET)
    return pd.read_csv(path, sep=None, engine="python")


def build_ebola_adjacency(node_ids: Sequence[str], gadm_dir: str,
                          knn_fallback: int = 4) -> tuple[np.ndarray, np.ndarray, dict]:
    """District contiguity for Guinea + Liberia + Sierra Leone, plus the covariates C.

    Deliberately NOT block-diagonal. Unlike dengue's countries, these three are physically
    contiguous and this was one outbreak that spread across them, so cross-border edges carry
    real signal and are kept, counted and reported. Contiguity is built over the union of all
    districts.

    Any district with no GADM polygon raises with the full diagnostic; nothing is dropped.
    Isolated districts fall back to k-NN centroids so no node is degenerate for message
    passing. C is built in the same pass, which is what aligns its rows to node_ids.

    Returns (A [N,N], C [N,3], report).
    """
    try:
        import geopandas as gpd
    except ImportError as e:                      # pragma: no cover - env dependent
        raise ImportError("build_ebola_adjacency needs geopandas + libpysal") from e

    geoms = []
    for country, (iso3, lvl, field) in EBOLA_GADM.items():
        zp = Path(gadm_dir) / f"gadm41_{iso3}_shp.zip"
        if not zp.exists():
            raise FileNotFoundError(
                f"GADM 4.1 shapefile not found: {zp}. Download from "
                f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso3}_shp.zip")
        g = gpd.read_file(zp, layer=f"gadm41_{iso3}_{lvl}")
        g["_key"] = country + "|" + g[field].map(_canon)
        geoms.append(g[["_key", "geometry"]])
    gadm = pd.concat(geoms, ignore_index=True).drop_duplicates("_key").set_index("_key")

    # our node name -> the key GADM ships. Applied to the lookup only, never to a node id.
    fix = {f"{c}|{_canon(k)}": f"{c}|{_canon(v)}"
           for c, m in EBOLA_GADM_FIX.items() for k, v in m.items()}
    want = [fix.get(n, n) for n in node_ids]

    miss = [n for n, w in zip(node_ids, want) if w not in gadm.index]
    if miss:
        raise ValueError(
            f"GADM name-join failed for {len(miss)} Ebola district(s): {miss}. "
            f"Add 'our_name -> gadm_name' entries to EBOLA_GADM_FIX[country].")

    gdf = gpd.GeoDataFrame(gadm.loc[want].reset_index(), geometry="geometry",
                           crs=geoms[0].crs)                  # polygons in node order
    A = _contiguity(gdf, knn_fallback)
    C = _centroid_area(gdf)                                   # same rows -> same order

    country_of = [n.split("|", 1)[0] for n in node_ids]
    cross = [(node_ids[i], node_ids[j])
             for i, j in zip(*np.where(np.triu(A, 1) > 0))
             if country_of[i] != country_of[j]]
    report = dict(n_nodes=len(node_ids), n_edges=int(A.sum() // 2),
                  n_isolated=int((A.sum(1) == 0).sum()),
                  n_cross_border=len(cross), cross_border_edges=cross,
                  gadm_name_fixes={n: w for n, w in zip(node_ids, want) if n != w})
    return A, C, report


def load_ebola(csv_path: str, cols: dict = EBOLA_COLS,
               countries: Optional[Sequence[str]] = None,
               exclude_regions: Optional[Sequence[str]] = None,
               few_shot_support_weeks: int = 2,
               few_shot_support_cutoff: Optional[str] = None,
               ratios=(0.5, 0.2, 0.3),
               gadm_dir: Optional[str] = None) -> DiseaseTensors:
    """OCHA ROWCA Ebola (long, cumulative) -> DiseaseTensors. Node id = 'country|district'.

    The target is the headline `Cases` series, which is cumulative with downward revisions,
    so it is differenced and clipped at 0. `Deaths` (also cumulative) becomes the extended,
    single-disease channel `deaths_norm`; it is never a core channel. The remaining Category
    values are never summed into the target.

    Cleaning is unconditional, because none of it is optional for this file: out-of-window
    Excel-epoch dates, the National aggregate, the multi-district blobs and the two Sierra
    Leone non-districts. Every removal is declared with a reason in EBOLA_DROP_* and recorded
    in meta['nodes_dropped']. An undeclared blob label raises.

    countries=None keeps every country in the file; pass EBOLA_CORE_COUNTRIES for the main
    experiment. `exclude_regions` drops named districts.

    Few-shot protocol. Two support regimes:

      calendar-prefix (`few_shot_support_cutoff` set): support = every observed cell dated on
        or before the cutoff; query = every observed cell after it. This is calendar-causal by
        construction -- no query cell is ever earlier in time than a support cell -- which is
        the only reading of "the first weeks of an emerging outbreak" that survives review.
        Districts entering after the cutoff receive no support (the genuine few-shot case).

      per-district (`few_shot_support_cutoff` None): support = the first
        `few_shot_support_weeks` OBSERVED weeks of each district. Retained for comparison; it
        is NOT calendar-causal, because districts enter at different dates.

    Because week 0 is masked (cumulative_to_weekly_incidence), support holds real increments
    rather than a back-log. The scaler is fit on the pooled support cells only: far too few
    per node for a stable per-node scale, and fitting past support would leak the outbreak's
    magnitude.

    Pass `gadm_dir` to build the district graph.
    """
    df = _read_ebola(csv_path).copy()
    df["_country"] = df[cols["country"]].map(_canon)
    df["_dist"] = df[cols["district"]].map(_canon)

    # 1. date window; drops the Excel-epoch artefacts before anything else sees them
    dts = pd.to_datetime(df[cols["date"]], errors="coerce")
    df = df[dts.notna() & dts.dt.year.between(*EBOLA_YEARS)]

    # 2. country restriction
    if countries is not None:
        df = df[df["_country"].isin({_canon(c) for c in countries})]

    # 3. declared drops
    lab = {_canon(k) for k in EBOLA_DROP_LABELS}
    nod = {_canon(k) for k in EBOLA_DROP_NODES}
    df["_node"] = df["_country"] + "|" + df["_dist"]
    hit = df["_dist"].isin(lab) | df["_node"].isin(nod)
    dropped = sorted(set(df.loc[hit, "_node"]))
    df = df[~hit]

    # 4. Raise on any UNDECLARED blob label. A new aggregate label in a refreshed file is a
    #    decision for a human, not something to ingest as if it were one district.
    undeclared = sorted({d for d in df["_dist"].unique() if _EBOLA_BLOB_RE.search(d)})
    if undeclared:
        raise ValueError(
            f"{len(undeclared)} undeclared multi-district 'blob' label(s) in the Ebola "
            f"compilation: {undeclared}. These name several districts at once and have no "
            f"polygon. Add each to EBOLA_DROP_NODES with its reason, or alias it.")

    # 5. aliases, before the pivot: they repair the data
    amap = {(_canon(c), _canon(k)): _canon(v)
            for c, m in EBOLA_NAME_ALIASES.items() for k, v in m.items()}
    df["_dist"] = [amap.get((c, d), d) for c, d in zip(df["_country"], df["_dist"])]
    df["_node"] = df["_country"] + "|" + df["_dist"]

    # 6. target = cumulative Cases; deaths = extended channel
    df["_cat"] = df[cols["indicator"]].map(_canon)      # collapses the New/Cases casings
    cases = df[df["_cat"] == "cases"]
    deaths = df[df["_cat"] == "deaths"]
    assert len(cases) > 0, "No Ebola 'Cases' rows after cleaning."

    inc, msk = cumulative_to_weekly_incidence(cases, "_node", cols["date"], cols["value"])
    dth, _ = cumulative_to_weekly_incidence(deaths, "_node", cols["date"], cols["value"])

    # 7. region control
    if exclude_regions:
        exc = {_canon(x) for x in exclude_regions}
        keep = [n for n in inc.index
                if n not in exc and n.split("|", 1)[-1] not in exc]
        inc, msk = inc.loc[keep], msk.loc[keep]
    assert len(inc) > 0, "No Ebola nodes left after region filtering."

    node_ids = list(inc.index)
    dates = pd.DatetimeIndex(inc.columns)
    dth = dth.reindex(index=node_ids, columns=inc.columns).fillna(0.0)

    raw = inc.to_numpy(dtype=np.float64)
    mask = msk.to_numpy().astype(np.uint8)
    deaths_raw = dth.to_numpy(dtype=np.float64)

    # 8. support set. Week 0 is already masked, so support holds real increments, not back-log.
    support_mask = np.zeros_like(mask)
    if few_shot_support_cutoff is not None:
        # calendar-prefix: every observed cell dated on or before the cutoff is support.
        cutoff = pd.Timestamp(few_shot_support_cutoff)
        pre = np.asarray(dates <= cutoff)
        support_mask[:, pre] = mask[:, pre]
        support_scheme = f"calendar_prefix<= {cutoff.date()}"
    else:
        # per-district: the first few_shot_support_weeks observed weeks of each district.
        for i in range(mask.shape[0]):
            obs_idx = np.where(mask[i] == 1)[0][:max(0, few_shot_support_weeks)]
            support_mask[i, obs_idx] = 1
        support_scheme = f"per_district_first_{few_shot_support_weeks}_obs_weeks"

    A_geo, C_geo, adj_report, A_geo_kind = None, None, None, "deferred"
    if gadm_dir:
        A_geo, C_geo, adj_report = build_ebola_adjacency(node_ids, gadm_dir)
        A_geo_kind = "queen+knn_cross_border"

    dt = _finalise(raw, mask, node_ids, dates, ratios,
                   disease="ebola", adm_level="per_country", t_res="weekly",
                   role="few_shot_holdout", support_mask=support_mask,
                   extended_raw={"deaths_norm": deaths_raw},
                   A_geo=A_geo, A_geo_kind=A_geo_kind,
                   source="OCHA ROWCA / HDX sub-national Ebola compilation "
                          "(WHO situation reports), 2014-03-24 -> 2015-03-28")

    q = dt.meta["split"]["query_mask"]
    dt.meta["ebola_first_week_masked"] = True
    dt.meta["few_shot_support_scheme"] = support_scheme
    dt.meta["nodes_dropped"] = {
        n: EBOLA_DROP_NODES.get(n) or EBOLA_DROP_LABELS.get(n.split("|", 1)[-1])
        for n in dropped}
    dt.meta["name_aliases_applied"] = {f"{c}|{k}": f"{c}|{v}" for (c, k), v in amap.items()}
    dt.meta["node_country"] = {n: n.split("|", 1)[0] for n in node_ids}
    dt.meta["countries"] = sorted({n.split("|", 1)[0] for n in node_ids})
    dt.meta["graph_is_block_diagonal"] = False    # cross-border edges are real here
    dt.meta["shapefile_source"] = "GADM 4.1 (gadm.org)" if gadm_dir else None
    dt.meta["adjacency_report"] = adj_report
    # A district whose observed cells all fall in the support set yields no query cell: it is a
    # graph node that is never scored. Recorded so evaluation cannot silently count it as one.
    dt.meta["nodes_without_query"] = [n for i, n in enumerate(node_ids) if q[i].sum() == 0]
    ref = cumulative_reference_mass(cases, "_node", cols["date"], cols["value"])
    dt.meta["cumulative_reference"] = {n: ref[n] for n in node_ids if n in ref}

    # Same C contract as dengue and influenza: raw, static, and NOT transfer-safe (the
    # diseases occupy disjoint geography, so a centroid identifies the disease).
    if C_geo is not None:
        dt.C = C_geo
        dt.meta["covariates"] = ["centroid_lat", "centroid_lon", "area_km2"]
        dt.meta["covariates_status"] = "populated from GADM 4.1"
        dt.meta["covariates_source"] = "GADM 4.1 (gadm.org)"
        dt.meta["covariates_transfer_safe"] = False
        dt.check()
    return dt


# --------------------------------------------------------------------------- #
# INFLUENZA — static covariates C (GADM 4.1)
# --------------------------------------------------------------------------- #
# The 49 US-States columns are the 50 states alphabetically, minus Florida (ILINet does not
# report it). Verified against real contiguity, not assumed: under this ordering every shipped
# edge is a genuine land-contiguity edge and the two isolated nodes are Alaska and Hawaii.
US_STATES_49 = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
    "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
]
# The 10 HHS regions in order 1-10, which is the us-regions node order. Verified against GADM
# contiguity. These DO include Florida, which the 49-state set excludes.
HHS_REGIONS = {
    1: ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"],
    2: ["New Jersey", "New York"],
    3: ["Delaware", "District of Columbia", "Maryland", "Pennsylvania", "Virginia",
        "West Virginia"],
    4: ["Alabama", "Florida", "Georgia", "Kentucky", "Mississippi", "North Carolina",
        "South Carolina", "Tennessee"],
    5: ["Illinois", "Indiana", "Michigan", "Minnesota", "Ohio", "Wisconsin"],
    6: ["Arkansas", "Louisiana", "New Mexico", "Oklahoma", "Texas"],
    7: ["Iowa", "Kansas", "Missouri", "Nebraska"],
    8: ["Colorado", "Montana", "North Dakota", "South Dakota", "Utah", "Wyoming"],
    9: ["Arizona", "California", "Hawaii", "Nevada"],
    10: ["Alaska", "Idaho", "Oregon", "Washington"],
}
# GADM ships 'Naoasaki', a typo for Nagasaki. Unaliased it silently drops a prefecture from
# the join. (Hyogo's macron needs no entry: _canon strips it.)
GADM_JPN_FIX = {"naoasaki": "nagasaki"}
EQUAL_AREA_CRS = "EPSG:6933"   # World Cylindrical Equal Area; areas in m^2, not degrees


def build_influenza_covariates(dataset: str, gadm_dir: str,
                               japan_map: str = "japan_node_map.csv") -> np.ndarray:
    """Static per-node covariates C = [centroid_lat, centroid_lon, area_km2] from GADM.

    Rows align to the loader's node_ids because index order is the node order:
      japan       the recovered prefecture map (japan_nodes.py)
      us-states   the verified alphabetical-minus-Florida ordering
      us-regions  the 10 HHS regions, each the union of its member states' geometry

    Areas are computed in an equal-area projection, never in degrees. Values are raw: C is
    static, so it carries no leakage risk and scaling is the encoder's job.
    """
    import geopandas as gpd                       # optional dep; only needed when C is built

    ds = _canon(dataset)
    if ds in ("japan", "japan-prefectures"):
        gdf = gpd.read_file(f"zip://{gadm_dir}/gadm41_JPN_shp.zip!gadm41_JPN_1.shp")
        key = gdf["NAME_1"].map(lambda n: GADM_JPN_FIX.get(_canon(n), _canon(n)))
        gdf = gdf.assign(_k=key).set_index("_k")
        order = [_canon(p) for p in pd.read_csv(japan_map)["prefecture"]]
        missing = set(order) - set(gdf.index)
        assert not missing, f"japan: prefectures absent from GADM: {sorted(missing)}"
        geoms = gdf.loc[order].geometry

    elif ds in ("us-states", "state"):
        gdf = gpd.read_file(f"zip://{gadm_dir}/gadm41_USA_shp.zip!gadm41_USA_1.shp")
        gdf = gdf.assign(_k=gdf["NAME_1"].map(_canon)).set_index("_k")
        order = [_canon(s) for s in US_STATES_49]
        missing = set(order) - set(gdf.index)
        assert not missing, f"us-states: absent from GADM: {sorted(missing)}"
        geoms = gdf.loc[order].geometry

    elif ds in ("us-regions", "region"):
        gdf = gpd.read_file(f"zip://{gadm_dir}/gadm41_USA_shp.zip!gadm41_USA_1.shp")
        gdf = gdf.assign(_k=gdf["NAME_1"].map(_canon))
        geoms = []
        for r in range(1, 11):                     # HHS 1..10 == node order
            members = {_canon(s) for s in HHS_REGIONS[r]}
            sel = gdf[gdf["_k"].isin(members)]
            assert len(sel) == len(members), \
                f"us-regions: R{r} matched {len(sel)} of {len(members)} states in GADM"
            geoms.append(sel.geometry.union_all())
        geoms = gpd.GeoSeries(geoms, crs=gdf.crs)

    else:
        raise ValueError(f"no covariate source defined for influenza dataset {dataset!r}")

    ea = gpd.GeoSeries(geoms.values, crs=gdf.crs).to_crs(EQUAL_AREA_CRS)
    cent = ea.centroid.to_crs("EPSG:4326")
    C = np.column_stack([cent.y.to_numpy(),                 # centroid latitude  (deg)
                         cent.x.to_numpy(),                 # centroid longitude (deg)
                         ea.area.to_numpy() / 1e6])         # area (km^2)
    return C.astype(np.float32)


# --------------------------------------------------------------------------- #
# INFLUENZA — ColaGNN-shipped [T,N] matrix + adjacency (fit-free graph)
# --------------------------------------------------------------------------- #
def load_influenza(matrix_path: str, adj_path: str, dataset: str = "japan",
                   start_date: Optional[str] = None, ratios=(0.5, 0.2, 0.3),
                   header=None, gadm_dir: Optional[str] = None) -> DiseaseTensors:
    """ColaGNN influenza: `matrix_path` is [T,N] ILI counts, `adj_path` its shipped adjacency.

    The shipped edges are reused verbatim, so results stay comparable to the published
    baselines; only the diagonal is normalised away (see below). The matrices ship undated, so
    `start_date` is resolved per dataset from INFLUENZA_START unless overridden.
    """
    if start_date is None:
        start_date = INFLUENZA_START.get(_canon(dataset), "2012-01-01")
    mat = pd.read_csv(matrix_path, header=header).to_numpy(dtype=np.float64)  # [T,N]
    raw = mat.T                                                              # [N,T]
    N, T = raw.shape
    A_geo = pd.read_csv(adj_path, header=None).to_numpy(dtype=np.float32)
    assert A_geo.shape == (N, N), f"shipped adjacency {A_geo.shape} != N={N}"

    # The shipped adjacencies carry a self-loop on every node, while _contiguity() zeroes the
    # diagonal for dengue and Ebola. One convention across diseases is a correctness property:
    # the encoder applies A_hat = A + I once, so a graph arriving with diag=1 would give
    # influenza self-weight 2 against dengue's 1 -- a structural tell for disease identity.
    # Information-preserving: the shipped matrix is exactly A_geo + I.
    np.fill_diagonal(A_geo, 0.0)

    # Some shipped nodes have no neighbour: their only entry WAS the self-loop just cleared.
    # This is inherited from the shipped graph, not introduced here. They are deliberately not
    # k-NN-patched (as _contiguity does for dengue/Ebola) -- inventing an edge the shipped
    # graph lacks would destroy the comparability that is the sole reason to reuse it, and
    # these nodes are anonymous indices with no geometry to fall back on.
    #   The encoder MUST add I before any D^-1/2 normalisation, or these rows divide by zero.
    isolated = np.flatnonzero(A_geo.sum(1) == 0)

    node_ids = [f"{dataset}_{i}" for i in range(N)]
    dates = pd.date_range(start_date, periods=T, freq=WEEK_ANCHOR)
    mask = np.ones((N, T), dtype=np.uint8)

    dt = _finalise(raw, mask, node_ids, dates, ratios,
                   disease=f"influenza:{dataset}", adm_level="Admin1",
                   t_res="weekly", A_geo=A_geo, A_geo_kind="shipped(diag_zeroed)",
                   source="ColaGNN (CIKM 2020) shipped influenza benchmark")
    dt.meta["isolated_nodes"] = [node_ids[i] for i in isolated]
    dt.meta["requires_encoder_self_loops"] = True
    dt.meta["shipped_adjacency_sha256"] = _sha256(adj_path)
    dt.meta["shipped_matrix_sha256"] = _sha256(matrix_path)

    # Same C contract as dengue and Ebola: raw, static, and not transfer-safe. Without
    # gadm_dir, C stays the all-zero placeholder _finalise built.
    if gadm_dir:
        dt.C = build_influenza_covariates(dataset, gadm_dir)
        assert dt.C.shape == (N, 3), f"C {dt.C.shape} != ({N}, 3)"
        dt.meta["covariates"] = ["centroid_lat", "centroid_lon", "area_km2"]
        dt.meta["covariates_status"] = "populated from GADM 4.1"
        dt.meta["covariates_source"] = "GADM 4.1 (gadm.org)"
        dt.meta["covariates_transfer_safe"] = False
        dt.check()
    return dt


# --------------------------------------------------------------------------- #
# Rolling-origin backtest scaffold
# --------------------------------------------------------------------------- #
def build_rolling_origins(node_ids: Sequence[str], obs_mask: np.ndarray,
                          split_scheme: str, ratios=(0.5, 0.2, 0.3),
                          n_origins: int = 5, horizon: int = 1) -> Optional[dict]:
    """Expanding-window, single-step rolling-origin cut points for the backtest.

    Only the cut points are stored: an origin is one integer per group, and
    `rolling_origin_masks()` expands it into masks on demand. Materialising the mask pairs
    here would cost hundreds of MB in the .npz to encode what one integer implies.

    The contract the stored indices enforce: at origin k the model may see only cells with
    t <= origin_k, and the scaler (plus any learned graph structure) must be REFIT on that
    slice alone. Nothing here carries a scaler, so there is nothing to reuse by accident.

    Grouping mirrors the headline split, or the origins would contradict it:
      per-country  origins per country, on that country's own observed span. A single global
                   origin on the union calendar could sit before one country's first report
                   and after another's last.
      fixed        one origin sequence on the shared calendar.
      few-shot     returns None. Rolling origin is meaningless for a held-out disease whose
                   protocol IS the support set: expanding the window past support is the leak.

    Origins sit on observed columns only, equally spaced from the headline train_end to the
    last observed column, so every origin has real data on both sides of the cut.

    Returns {scheme, n_origins, horizon, group_of, origins: {group: [col, ...]}} or None.
    """
    if split_scheme.startswith("few_shot"):
        return None

    per_country = split_scheme.startswith("per_country")
    groups: dict = {}
    for i, n in enumerate(node_ids):
        groups.setdefault(n.split("|", 1)[0] if per_country else "__all__", []).append(i)

    origins: dict = {}
    for g, idx in sorted(groups.items()):
        cols = np.where(obs_mask[idx].any(axis=0))[0]      # this group's observed calendar
        if cols.size < n_origins + 2:
            continue                       # too short to backtest
        start = int(round(cols.size * ratios[0]))          # the headline train_end, in-group
        last = cols.size - horizon                         # keep `horizon` steps to score on
        if last <= start:
            continue
        picks = np.unique(np.linspace(start, last - 1, n_origins).round().astype(int))
        origins[g] = [int(cols[p]) for p in picks]

    if not origins:
        return None
    return {"scheme": "expanding_window_single_step",
            "n_origins": n_origins, "horizon": horizon,
            "group_of": "country" if per_country else "all",
            "refit_per_origin": True,      # applies to the scaler AND the graph
            "origins": origins}


def rolling_origin_masks(dt: "DiseaseTensors", k: int) -> tuple[np.ndarray, np.ndarray]:
    """Materialise (train_mask, eval_mask) for the k-th rolling origin, on demand.

    train = observed cells at t <= origin; eval = observed cells in (origin, origin+horizon].
    Both are restricted to observed cells, so an imputed week can never be fit on or scored.

    Refit the scaler on `train_mask`. Do NOT reuse dt.meta['scaler']: it belongs to the
    headline split and has seen data past this origin.
    """
    ro = dt.meta.get("rolling_origins")
    if not ro:
        raise ValueError(f"{dt.meta['disease']} has no rolling origins "
                         f"(few-shot bundles have none by design)")
    N, T = dt.M.shape
    train = np.zeros((N, T), np.uint8)
    ev = np.zeros((N, T), np.uint8)
    per_country = ro["group_of"] == "country"
    h = ro["horizon"]
    for i, n in enumerate(dt.meta["node_ids"]):
        g = n.split("|", 1)[0] if per_country else "__all__"
        cuts = ro["origins"].get(g)
        if not cuts or k >= len(cuts):
            continue                       # group not backtested at this origin
        o = cuts[k]
        train[i, : o + 1] = dt.M[i, : o + 1]
        ev[i, o + 1 : o + 1 + h] = dt.M[i, o + 1 : o + 1 + h]
    return train, ev


def _finalise(raw, mask, node_ids, dates, ratios, disease, adm_level, t_res,
              role="development", extended_raw=None, support_mask=None,
              A_geo=None, A_geo_kind="pending",
              split_masks=None, source="") -> DiseaseTensors:
    """Normalise (leakage-safe), assemble X, pack meta, check."""
    N, T = raw.shape
    train_end, val_end = chronological_split(T, ratios)
    few_shot = (role == "few_shot_holdout" and support_mask is not None)

    # The leakage-safe fit set:
    #   few-shot    the support set, pooled per-disease (2 weeks/node is too few per-node)
    #   split_masks the per-country train mask, per-node
    #   otherwise   the global train slice t < train_end
    if few_shot:
        fit_mask = mask.astype(bool) & support_mask.astype(bool)
        scaler = fit_scalers_masked(raw, fit_mask, per_disease=True)
    elif split_masks is not None:
        fit_mask = mask.astype(bool) & split_masks["train_mask"].astype(bool)
        # groups = country: a node whose reporting starts after its country's train boundary
        # has no fit cell of its own and takes the country's pooled train statistics.
        countries = [n.split("|", 1)[0] for n in node_ids]
        scaler = fit_scalers_masked(raw, fit_mask, per_disease=False, groups=countries)
    else:
        fit_mask = None
        scaler = fit_scalers(raw, mask, train_end)

    # Imputed steps become 0 in NORMALISED space, i.e. the per-node mean. This is a neutral
    # value, not the transform of a raw 0, which would assert "zero cases". obs_mask flags them.
    inc_norm = np.where(mask == 1, apply_scaler(raw, scaler), 0.0).astype(np.float32)

    extended = None
    if extended_raw:
        extended = {}
        for nm, arr in extended_raw.items():
            if few_shot or split_masks is not None:
                sc = fit_scalers_masked(
                    arr, fit_mask, per_disease=few_shot,
                    groups=None if few_shot else [n.split("|", 1)[0] for n in node_ids])
            else:
                sc = fit_scalers(arr, mask, train_end)   # own leakage-safe scaler
            extended[nm] = np.where(mask == 1, apply_scaler(arr, sc), 0.0).astype(np.float32)

    X, feature_names = _assemble_X(inc_norm, dates, mask, extended)
    y = inc_norm.copy()

    if A_geo is None:                                # no gadm_dir -> zero placeholder
        A_geo = np.zeros((N, N), dtype=np.float32)
        A_geo_kind = "deferred"

    steps_per_year = 52 if t_res == "weekly" else 12
    if few_shot:
        query_mask = (mask.astype(bool) & ~support_mask.astype(bool)).astype(np.uint8)
        split = {"scheme": "few_shot_support_query",
                 "support_weeks": int(support_mask.sum(1).max()),
                 "support_mask": support_mask.astype(np.uint8),
                 "query_mask": query_mask}
        split_scheme = "few_shot_support_query"
    elif split_masks is not None:
        split = {"scheme": "per_country_chronological_50_20_30",
                 "train_mask": split_masks["train_mask"].astype(np.uint8),
                 "val_mask": split_masks["val_mask"].astype(np.uint8),
                 "test_mask": split_masks["test_mask"].astype(np.uint8),
                 "country_bounds": split_masks["bounds"],
                 "nodes_without_train": split_masks.get("nodes_without_train", [])}
        split_scheme = "per_country_chronological_50_20_30"
    else:
        split = {"train_end": train_end, "val_end": val_end}
        split_scheme = "fixed_50_20_30"

    rolling = build_rolling_origins(node_ids, mask, split_scheme, ratios)   # None for few-shot

    meta = dict(
        disease=disease, role=role, node_ids=node_ids, adm_level=adm_level,
        dates=list(dates), t_res=t_res, steps_per_year=steps_per_year,
        feature_names=feature_names, core_feature_idx=CORE_FEATURE_IDX,
        scaler=scaler, scaler_scope=("per_disease_support" if few_shot else "per_node_train"),
        split=split, split_scheme=split_scheme, rolling_origins=rolling,
        A_geo_kind=A_geo_kind, A_mob_available=False,
        # overwritten by the loader when a gadm_dir is supplied
        covariates_status="PLACEHOLDER — no gadm_dir given, C is all-zero",
        source=source,
    )
    return DiseaseTensors(X=X, A_geo=A_geo, C=np.zeros((N, 3), np.float32),
                          M=mask, y=y, meta=meta,
                          raw=raw.astype(np.float32)).check()

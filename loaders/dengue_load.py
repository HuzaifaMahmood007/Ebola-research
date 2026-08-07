"""Build the dengue bundle and assert its geography, then report it.

Countries are not hand-picked: the WHO/CDC endemic list is only the candidate pool, and the
loader prunes it with the locked coverage thresholds, choosing the finest administrative level
each country can support.

Adjacency is built per country from GADM and assembled block-diagonally, with no cross-border
edges. Any unmatched node halts the build rather than being dropped.

Everything this script prints, it asserts.
"""
import os

import numpy as np

from to_schema import load_dengue
from dengue_aliases import (DENGUE_NAME_ALIASES, DENGUE_GADM_FIX, DENGUE_UNMAPPABLE,
                            validate_aliases)

CSV = r"data/Final datasets/OpenDengue_Best_Spacial.csv"
GADM_DIR = r"data/gadm"
MIN_WEEKS = 52                # one full seasonal cycle
MIN_NODES_PER_COUNTRY = 3

# Set to exclude a country (e.g. while its shapefile is still downloading).
SKIP = set(os.environ.get("DENGUE_SKIP", "").split(",")) - {""}

dt = load_dengue(
    CSV,
    countries=None,           # the endemic candidate pool, intersected with the file
    level="auto",             # finest level each country supports
    t_res_filter="Week",      # weekly slice only; the monthly history is not interpolated
    min_weeks=MIN_WEEKS,
    min_nodes_per_country=MIN_NODES_PER_COUNTRY,
    ratios=(0.5, 0.2, 0.3),
    gadm_dir=GADM_DIR,
    name_aliases=DENGUE_NAME_ALIASES,   # load time: repairs the data
    gadm_fix=DENGUE_GADM_FIX,           # join time: absorbs GADM's typos
    unmappable=DENGUE_UNMAPPABLE,
    exclude_countries=SKIP or None,
)

# Guard against typos in the alias file itself.
bad = validate_aliases(dt.meta["country_levels"], GADM_DIR)
assert not bad, f"aliases that do not resolve to a GADM key: {bad}"

# GADM's defects are absorbed at the join and must never reach a node id. This regresses the
# moment someone adds a GADM quirk to the load-time map, so it is asserted rather than assumed.
_gadm_typos = ["japan|naoasaki", "peru|huanuco|huenuco"]
_leaked = [n for n in _gadm_typos if n in dt.meta["node_ids"]]
assert not _leaked, f"GADM typos leaked into node ids (belongs in DENGUE_GADM_FIX): {_leaked}"
for _n in ["japan|nagasaki", "peru|huanuco|huanuco"]:
    assert _n in dt.meta["node_ids"], f"expected correctly-spelt node {_n!r} in the bundle"

N, T, F = dt.X.shape
m, A = dt.meta, dt.A_geo

print(f"DiseaseTensors  X={dt.X.shape}  A_geo={A.shape} ({m['A_geo_kind']})  "
      f"M={dt.M.shape}  y={dt.y.shape}")
print(f"features        {m['feature_names']}  core_idx={m['core_feature_idx']}")
print(f"calendar        {m['dates'][0].date()} -> {m['dates'][-1].date()}  ({T} weeks)")
print(f"split           {m['split_scheme']}")
print(f"scaler          {m['scaler']['transform']} / {m['scaler_scope']}")
print(f"mask density    {dt.M.mean():.4f}  ({int(dt.M.sum())} observed of {N*T})")
print(f"shapefiles      {m['shapefile_source']}")

# ---- graph -----------------------------------------------------------------
deg = A.sum(1)
print(f"\nGRAPH  {int(A.sum()//2)} undirected edges  |  symmetric={np.array_equal(A, A.T)}"
      f"  |  self-loops={int(np.diag(A).sum())}")
print(f"       degree: mean={deg.mean():.2f}  min={int(deg.min())}  max={int(deg.max())}"
      f"  |  isolated nodes={int((deg == 0).sum())}")

print(f"\nPER-COUNTRY BLOCKS ({len(m['countries'])} countries / {N} nodes):")
rep = m["adjacency_report"]
for c in sorted(rep, key=lambda x: -rep[x]["n_nodes"]):
    r = rep[c]
    print(f"  {c:<22} {r['level']:<7} nodes={r['n_nodes']:>5}  edges={r['n_edges']:>6}"
          f"  isolated={r['n_isolated']:>3}")

# block-diagonality: no edge may cross a country boundary
countries = np.array([m["node_country"][n] for n in m["node_ids"]])
cross = (A > 0) & (countries[:, None] != countries[None, :])
print(f"\ncross-border edges (must be 0): {int(cross.sum())}")
assert cross.sum() == 0, "block-diagonal violated: found cross-border edges"

# ---- static covariates C = [centroid_lat, centroid_lon, area_km2] ------------
lat, lon, area = dt.C[:, 0], dt.C[:, 1], dt.C[:, 2]
print(f"\ncovariates    C={dt.C.shape} {m['covariates']}  ({m['covariates_status']})")
print(f"  lat  {lat.min():8.2f} .. {lat.max():7.2f}   lon {lon.min():8.2f} .. {lon.max():7.2f}")
print(f"  area {area.min():8.1f} .. {area.max():9.1f} km^2   (median {np.median(area):.1f})")
print(f"  transfer_safe={m['covariates_transfer_safe']}  (geography identifies the disease)")

assert np.isfinite(dt.C).all(), "non-finite covariate"
assert (area > 0).all(), "non-positive area"
assert (np.abs(lat) <= 90).all() and (np.abs(lon) <= 180).all(), "centroid off-globe"

# Row alignment is the thing that would fail SILENTLY, so check it against geography:
# every node's centroid must fall inside its own country's bounding box. A shuffled C
# would put Brazilian municipalities in Japan and this would catch it.
#
# Boxes include each country's OFFSHORE territory, not just its mainland. The first run
# with mainland-only boxes flagged exactly three nodes -- Fernando de Noronha (Brazil's
# Atlantic archipelago, 18.6 km2), Providencia (Colombia's Caribbean island, 23.2 km2) and
# Lienkiang/Matsu (Taiwan's islands off the Fujian coast, 32.7 km2). All three are real and
# their centroids are correct; that they land exactly where remote islands belong is itself
# evidence C is row-aligned. The boxes were widened, not the data.
BOX = {  # country -> (lat_min, lat_max, lon_min, lon_max), generous
    "brazil": (-34, 6, -74, -32),      # lon to -32: Fernando de Noronha
    "colombia": (-5, 14, -82, -66),    # lat to 14: Providencia / San Andres
    "argentina": (-56, -21, -74, -53), "peru": (-19, 0, -82, -68),
    "japan": (24, 46, 122, 154),       "dominican republic": (17, 20, -73, -68),
    "mexico": (14, 33, -118, -86),     "ecuador": (-5, 2, -92, -75),
    "taiwan": (21, 27, 118, 123),      # lat to 27: Matsu (Lienkiang)
    "nicaragua": (10, 15, -88, -82),
    "panama": (7, 10, -83, -77),       "bolivia": (-23, -9, -70, -57),
}
bad = [n for i, n in enumerate(m["node_ids"])
       if not (BOX[m["node_country"][n]][0] <= lat[i] <= BOX[m["node_country"][n]][1]
               and BOX[m["node_country"][n]][2] <= lon[i] <= BOX[m["node_country"][n]][3])]
assert not bad, f"{len(bad)} node centroid(s) outside their own country: {bad[:10]}"
print(f"  row alignment: all {len(m['node_ids'])} centroids fall inside their own country")

dt.check()
print("\n.check() PASSED")

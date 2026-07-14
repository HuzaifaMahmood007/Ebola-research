"""Recover the Japan node ordering from the shipped benchmark -> japan_node_map.csv

The benchmark ships the matrix and its adjacency with NO name list and no documented column
order. The ordering matches neither JIS prefecture order nor romaji-alphabetical, so the 47
nodes are anonymous, which blocks any join to population, geography or mobility.

Recovery uses three independent lines of evidence. Each is weak alone; together they are
decisive.

  1. STRUCTURE. Build true prefecture contiguity from GADM and solve for the permutation onto
     the shipped graph. Queen contiguity has exactly 86 edges -- the shipped graph's count --
     and the two graphs are ISOMORPHIC. The isolated prefectures are Hokkaido and Okinawa,
     Japan's only two with no land border. This pins 41 of 47 nodes uniquely. The remaining 6
     sit in 3 automorphism orbits that structure physically cannot separate: {Hokkaido, Okinawa}
     (both isolated), and the symmetric Shikoku pairs {Kagawa, Kochi} and {Ehime, Tokushima}.

  2. MAGNITUDE. ILI counts track population, and NOTHING about the graph knows population, so
     this is genuinely independent evidence. Across all 47 nodes the recovered labelling gives
     Spearman rho = +0.949 between mean ILI and prefecture population; the top-8 come out as
     Tokyo/Kanagawa/Saitama/Osaka/Aichi/Fukuoka/Chiba/Hokkaido (Japan's most populous) and the
     bottom as Tottori/Shimane (its two least). Within each ambiguous orbit the higher-ILI node
     is the more populous one, consistently.

  3. PHASE. Okinawa is subtropical and known for out-of-season influenza. japan_19 is the LEAST
     winter-concentrated of all 47 prefectures (74.0% of its ILI in Dec-Mar vs a national mean
     of 92.2%, z = -5.05). Seasonal phase is independent of both structure and magnitude, and it
     pins japan_19 = Okinawa, hence japan_10 = Hokkaido.

CONFIDENCE is recorded per node, honestly -- see the `confidence` column. The Kagawa/Kochi pair
is the weakest call (population separates them by only 1.38x against an ILI ratio of 1.22x); a
swap there would change nothing structurally (the nodes are automorphic, so the GRAPH is
identical either way) and matters only if covariates are later joined to those two nodes.

Requires geopandas/libpysal:  conda run -n ebola python japan_nodes.py
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from libpysal.weights import Queen
from scipy.stats import spearmanr

from to_schema import _canon, INFLUENZA_START, WEEK_ANCHOR

RAW = "data/Final datasets/influenza"
GADM = "zip://data/gadm/gadm41_JPN_shp.zip!gadm41_JPN_1.shp"

# 2020 census, millions. Used ONLY to break automorphism ties and to validate the mapping --
# never as a model input (that would be a covariate, and C is populated from GADM, not here).
POP = {
    "Hokkaido": 5.22, "Aomori": 1.24, "Iwate": 1.21, "Miyagi": 2.30, "Akita": 0.96,
    "Yamagata": 1.07, "Fukushima": 1.83, "Ibaraki": 2.87, "Tochigi": 1.93, "Gunma": 1.94,
    "Saitama": 7.34, "Chiba": 6.28, "Tokyo": 14.05, "Kanagawa": 9.24, "Niigata": 2.20,
    "Toyama": 1.03, "Ishikawa": 1.13, "Fukui": 0.77, "Yamanashi": 0.81, "Nagano": 2.05,
    "Gifu": 1.98, "Shizuoka": 3.63, "Aichi": 7.54, "Mie": 1.77, "Shiga": 1.41, "Kyoto": 2.58,
    "Osaka": 8.84, "Hyogo": 5.47, "Nara": 1.32, "Wakayama": 0.92, "Tottori": 0.55,
    "Shimane": 0.67, "Okayama": 1.89, "Hiroshima": 2.80, "Yamaguchi": 1.34, "Tokushima": 0.72,
    "Kagawa": 0.95, "Ehime": 1.33, "Kochi": 0.69, "Fukuoka": 5.14, "Saga": 0.81,
    "Nagasaki": 1.31, "Kumamoto": 1.74, "Oita": 1.12, "Miyazaki": 1.07, "Kagoshima": 1.59,
    "Okinawa": 1.47,
}
# GADM 4.1 quirks: 'Hyōgo' carries a macron (_canon strips it) and 'Naoasaki' is a GADM TYPO
# for Nagasaki. Unaliased, the join silently loses a prefecture.
GADM_FIX = {"naoasaki": "Nagasaki"}


def main() -> pd.DataFrame:
    gdf = gpd.read_file(GADM).sort_values("NAME_1").reset_index(drop=True)
    by_canon = {_canon(k): k for k in POP}
    names = [GADM_FIX.get(_canon(n)) or by_canon[_canon(n)] for n in gdf["NAME_1"]]
    assert set(names) == set(POP), f"GADM/POP mismatch: {set(names) ^ set(POP)}"

    A = (Queen.from_dataframe(gdf, use_index=False).full()[0] > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    ship = pd.read_csv(f"{RAW}/japan-adj.txt", header=None).to_numpy(float)
    np.fill_diagonal(ship, 0.0)
    assert A.sum() == ship.sum(), \
        f"GADM queen has {int(A.sum()//2)} edges, shipped has {int(ship.sum()//2)}"

    G_gadm, G_ship = nx.from_numpy_array(A), nx.from_numpy_array(ship)
    gm = nx.algorithms.isomorphism.GraphMatcher(G_gadm, G_ship)
    assert gm.is_isomorphic(), "GADM contiguity is not isomorphic to the shipped graph"
    labelings = list(gm.isomorphisms_iter())          # 8, from the graph's automorphisms

    mat = pd.read_csv(f"{RAW}/japan.txt", header=None).to_numpy(float)
    mean_ili = mat.mean(0)

    # Step 2: among the structurally-valid labelings, take the one whose ILI best tracks
    # population. Structure is blind to population, so this is independent evidence.
    rho, best = max(
        ((spearmanr(mean_ili, [POP[names[g]] for g, _ in sorted(m.items(), key=lambda kv: kv[1])]).statistic, m)
         for m in labelings),
        key=lambda t: t[0],
    )
    node_of = {s: names[g] for g, s in best.items()}

    # Which nodes could structure alone NOT resolve? Only those rest on evidence 2 and 3.
    cand = {s: {names[g] for m in labelings for g, s2 in m.items() if s2 == s} for s in range(47)}
    ambiguous = {s for s, v in cand.items() if len(v) > 1}

    # Step 3: the Okinawa phase check -- independent of both structure and magnitude.
    dates = pd.date_range(INFLUENZA_START["japan"], periods=mat.shape[0], freq=WEEK_ANCHOR)
    winter = mat[np.isin(dates.month, [12, 1, 2, 3])].sum(0) / mat.sum(0)
    z = (winter - winter.mean()) / winter.std()
    okinawa = int(np.argmin(winter))
    assert node_of[okinawa] == "Okinawa", (
        f"phase check contradicts the mapping: the least winter-concentrated prefecture is "
        f"japan_{okinawa} = {node_of[okinawa]}, but subtropical Okinawa is expected")
    assert z[okinawa] < -3, f"Okinawa's out-of-season signature is weak (z={z[okinawa]:.2f})"

    out = pd.DataFrame({
        "node_id": [f"japan_{i}" for i in range(47)],
        "prefecture": [node_of[i] for i in range(47)],
        "population_m": [POP[node_of[i]] for i in range(47)],
        "mean_ili": mean_ili.round(1),
        "winter_share": winter.round(4),
        "confidence": ["structure" if i not in ambiguous else
                       ("structure+magnitude+phase" if i in (10, 19) else
                        "structure+magnitude") for i in range(47)],
    })
    out.to_csv("japan_node_map.csv", index=False)

    print(f"GADM queen contiguity: {int(A.sum()//2)} edges == shipped; ISOMORPHIC")
    print(f"{len(labelings)} structurally-valid labelings; {47 - len(ambiguous)}/47 nodes pinned "
          f"by structure alone")
    print(f"chosen labeling: Spearman rho(mean ILI, population) = {rho:+.4f}")
    print(f"phase check: least winter-concentrated = japan_{okinawa} = Okinawa "
          f"({winter[okinawa]:.1%} vs mean {winter.mean():.1%}, z={z[okinawa]:+.2f})")
    print(f"\nambiguous orbits resolved by evidence 2/3:")
    for s in sorted(ambiguous):
        print(f"  japan_{s:<2} -> {node_of[s]:<10} (of {sorted(cand[s])})")
    print(f"\nwrote japan_node_map.csv")
    return out


if __name__ == "__main__":
    main()

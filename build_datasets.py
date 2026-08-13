"""Regenerate every disease bundle from the cached raw sources, in one deterministic pass.

    conda run -n ebola python build_datasets.py [--check-deterministic]

    1. verify every raw input by checksum, including the 16 GADM shapefiles;
    2. build the 6 bundles: dengue, influenza (japan / us-regions / us-states), ebola, covid;
    3. run the leakage suite and its negative controls;
    4. write one .npz per disease to data/processed/, plus the config and a pip freeze.

The leakage suite GATES THE WRITE: nothing is written if a gate fails, so the only obtainable
artifact is a clean one. Bundles are .npz (arrays + a JSON meta blob), not pickles: a pickle couples
the file to the classes and library versions that wrote it, and unpickling executes code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

import to_schema as ts
from dengue_aliases import DENGUE_GADM_FIX, DENGUE_NAME_ALIASES, DENGUE_UNMAPPABLE
from to_schema import load_dengue, load_ebola, load_influenza

RAW = pathlib.Path("data/Final datasets")
GADM = "data/gadm"
OUT = pathlib.Path("data/processed")

DENGUE_CSV = RAW / "OpenDengue_Best_Spacial.csv"
EBOLA_XLSX = RAW / "data-ebola-public.xlsx"
FLU = RAW / "influenza"
COVID_CSV = RAW / "covid" / "us-states.csv"
COVID_MATRIX = RAW / "covid" / "covid_state_weekly.txt"   # derived from COVID_CSV, not a source

# The influenza six are pinned by FLU/SHA256SUMS.txt; these three have no manifest of their own.
# Every hash is of the file AS PUBLISHED, so a reader can hash their own download and compare.
# COVID_CSV is fetched rather than placed by hand -- `python loaders/covid_load.py --refresh`
# downloads it -- but it is verified here on exactly the same terms as the rest. A source that is
# downloaded is more exposed to drift, not less, so exempting it would have been backwards.
RAW_SHA256 = {
    DENGUE_CSV: "0f59280ed6795d8a0f7e87f900397db46cab6f2740106718c25c05699b9ababc",
    EBOLA_XLSX: "2d679a31a66f912da93fe0bd73da82a3c36b0d1143b1fcc298bc921d5f28f9f2",
    COVID_CSV:  "0b202f6ac8bad66b70b9e344c07155e3f1ce9338f86192af072b56ce448dfc43",
}

INFLUENZA = {
    "japan":      dict(matrix="japan.txt",     adj="japan-adj.txt"),
    "us-regions": dict(matrix="region785.txt", adj="region-adj.txt"),
    "us-states":  dict(matrix="state360.txt",  adj="state-adj.txt"),
}

# Every parameter that moves a number, in one place. Written to data/processed/config.json.
CONFIG = dict(
    dengue=dict(level="auto", t_res_filter="Week", min_weeks=52,
                min_nodes_per_country=3, ratios=[0.5, 0.2, 0.3]),
    influenza=dict(ratios=[0.5, 0.2, 0.3], adjacency="shipped(diag_zeroed)"),
    covid=dict(ratios=[0.5, 0.2, 0.3], adjacency="shared with influenza_us-states (identical graph)",
               excluded_nodes=["Florida", "District of Columbia"], npi_confounded=True),
    ebola=dict(few_shot_support_cutoff="2014-05-24", countries="EBOLA_CORE_COUNTRIES"),
    shapefiles="GADM 4.1",
    rolling_origins=dict(n_origins=5, horizon=1),
)


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> None:
    """Check every raw input against its recorded checksum, and exit on any mismatch.

    This runs before the build, not after: a drifted raw changes every number in the audit
    note, and the bundles would still pass .check().
    """
    for p, want in RAW_SHA256.items():
        got = _sha256(p)
        if got != want:
            sys.exit(f"CHECKSUM DRIFT  {p}\n  want {want}\n  got  {got}\n"
                     f"The cached raw is not the file the audit note describes. Stop.")
        print(f"ok   {p.name}")

    sums = {}
    for line in (FLU / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.lstrip("*")] = digest
    for name, want in sums.items():
        if _sha256(FLU / name) != want:
            sys.exit(f"CHECKSUM DRIFT  influenza/{name}. Stop.")
    print(f"ok   influenza: {len(sums)} matrices verified against SHA256SUMS.txt")

    # The shapefiles back every graph and every covariate; a re-cut release would change
    # every edge with no other visible signal.
    r = subprocess.run([sys.executable, "fetch_gadm.py", "--verify"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"GADM verification FAILED:\n{r.stdout}\n{r.stderr}")
    print("ok   GADM 4.1: 16/16 shapefiles verified")


def build_all() -> dict[str, ts.DiseaseTensors]:
    """Build the five bundles in a fixed order. Also imported by the leakage suite."""
    b: dict[str, ts.DiseaseTensors] = {}

    c = CONFIG["dengue"]
    b["dengue"] = load_dengue(
        str(DENGUE_CSV), level=c["level"], t_res_filter=c["t_res_filter"],
        min_weeks=c["min_weeks"], min_nodes_per_country=c["min_nodes_per_country"],
        ratios=tuple(c["ratios"]), gadm_dir=GADM,
        name_aliases=DENGUE_NAME_ALIASES,   # load time: repairs the data
        gadm_fix=DENGUE_GADM_FIX,           # join time: absorbs GADM's typos
        unmappable=DENGUE_UNMAPPABLE)

    for name, spec in INFLUENZA.items():
        b[f"influenza:{name}"] = load_influenza(
            str(FLU / spec["matrix"]), str(FLU / spec["adj"]), dataset=name,
            ratios=tuple(CONFIG["influenza"]["ratios"]), gadm_dir=GADM)

    b["ebola"] = load_ebola(
        str(EBOLA_XLSX), countries=ts.EBOLA_CORE_COUNTRIES,
        few_shot_support_cutoff=CONFIG["ebola"]["few_shot_support_cutoff"], gadm_dir=GADM)

    # COVID reuses influenza_us-states' SHIPPED graph and 49-node ordering bit for bit, so the two
    # bundles sit on an IDENTICAL A_geo with IDENTICAL covariates. Holding one out therefore varies
    # the DISEASE and nothing else; every other cross-disease cell in this study confounds disease
    # with graph, geography and node count. The equality is asserted, not assumed -- see the gate
    # below and loaders/covid_load.py.
    b["covid:us-states"] = ts.load_covid(
        str(COVID_CSV), str(FLU / INFLUENZA["us-states"]["adj"]), str(COVID_MATRIX),
        ratios=tuple(CONFIG["covid"]["ratios"]), gadm_dir=GADM)
    assert np.array_equal(b["covid:us-states"].A_geo, b["influenza:us-states"].A_geo), \
        "COVID and influenza_us-states must share a bit-identical graph -- the point of the bundle"

    return b


def _jsonable(o):
    """Coerce a meta value to plain Python so the JSON blob stays inspectable.

    The split masks and scaler params do NOT come through here: write() lifts them out as real
    arrays first, so they keep their dtype rather than surviving a JSON round-trip.
    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def write(name: str, dt: ts.DiseaseTensors) -> pathlib.Path:
    """Write one bundle to data/processed/<name>.npz."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name.replace(':', '_')}.npz"

    arrays = dict(X=dt.X, A_geo=dt.A_geo, C=dt.C, M=dt.M, y=dt.y)
    if dt.A_mob is not None:
        arrays["A_mob"] = dt.A_mob
    if dt.raw is not None:
        # Unscaled incidence. Required, not a convenience: the rolling-origin backtest refits
        # the scaler at each origin, which is impossible without the scaler's own input.
        arrays["raw"] = dt.raw

    # The split masks are data, not metadata: they are [N,T] and they decide what the model may
    # see. They ride as arrays so they keep their dtype; only the scalar parts stay in meta.
    meta = dict(dt.meta)
    split = dict(meta.pop("split"))
    for k in list(split):
        if isinstance(split[k], np.ndarray):
            arrays[f"split_{k}"] = split[k]
            split[k] = f"<array:split_{k}>"
    meta["split"] = split

    sc = dict(meta.pop("scaler"))
    for k in ("mean", "std"):
        arrays[f"scaler_{k}"] = np.asarray(sc[k])
    meta["scaler"] = {k: f"<array:scaler_{k}>" for k in ("mean", "std")}

    arrays["meta_json"] = np.asarray(json.dumps(meta, default=_jsonable))
    np.savez_compressed(path, **arrays)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-deterministic", action="store_true",
                    help="build twice and assert the arrays hash identically")
    args = ap.parse_args()

    print("=== 1. verifying raw inputs ===")
    verify_inputs()

    print("\n=== 2. building bundles ===")
    bundles = build_all()
    for k, dt in bundles.items():
        N, T, F = dt.X.shape
        print(f"  {k:22s} N={N:5d} T={T:5d} F={F}  {dt.meta['split_scheme']}")

    print("\n=== 3. leakage suite — gates the write ===")
    from tests.test_leakage import run_leakage_suite, self_test
    # self_test plants real leaks and confirms the gates catch them. Running it on every build
    # is the difference between "the suite passed" and "the suite could have failed".
    if not (run_leakage_suite(bundles) and self_test(bundles)):
        print("\nLEAKAGE SUITE FAILED — refusing to write data/processed/.")
        return 1

    print("\n=== 4. writing data/processed/ ===")
    for k, dt in bundles.items():
        p = write(k, dt)
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    (OUT / "env.txt").write_text(freeze)
    print(f"  {OUT / 'config.json'}\n  {OUT / 'env.txt'}")

    if args.check_deterministic:
        print("\n=== 5. determinism check (rebuild + rehash) ===")
        again = build_all()
        for k in bundles:
            for f in ("X", "A_geo", "C", "M", "y"):
                a, bb = getattr(bundles[k], f), getattr(again[k], f)
                if not np.array_equal(a, bb):
                    print(f"  NONDETERMINISTIC: {k}.{f}")
                    return 1
        print("  ok  two builds produced identical arrays for all 5 bundles")

    print("\nBUILD COMPLETE — all bundles regenerated, leakage-clean, and packaged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

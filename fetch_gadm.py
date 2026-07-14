"""Fetch the GADM 4.1 shapefiles the project's graphs and covariates are built from.

GADM 4.1 is a frozen, versioned, publicly-hosted release, so a fetch script + SHA-256
manifest gives the same reproducibility guarantee as vendoring the 461 MB into the DVC
remote -- at zero storage, and without redistributing it (GADM's licence forbids
redistribution for commercial use).

    python fetch_gadm.py            # download whatever is missing, verify every file
    python fetch_gadm.py --verify   # verify only; never touch the network

Exits non-zero on any checksum mismatch, so it can gate build_datasets.py

Provenance: https://gadm.org  (GADM 4.1, released 2022-07-16)
Used by: dengue graph + C (12 countries), Ebola graph (GIN/LBR/SLE), influenza C (JPN/USA).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

GADM_DIR = Path("data/gadm")
URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso3}_shp.zip"

# SHA-256 of every gadm41_<ISO3>_shp.zip this project reads. A mismatch means GADM re-cut the
# release, or the download is corrupt, and the graphs would change with no other visible signal.
# Hence: fail, do not warn.
MANIFEST = {
    # dengue -- the 12 locked development countries (to_schema.GADM_ISO3)
    "ARG": "effdb6717d46faa9853a9b8e9c8af29f41f25b356426e8450a47651a97b652e3",
    "BOL": "769d9976e2ad3bcefdce1bcab90580217a1eca325145a37c4b2e6ce81abdcfe4",
    "BRA": "f3a2e4739afbfd25eee0b0fe54dd8ef9d6421ad5b0e589d2796417c81304d5a2",
    "COL": "31a297106f0a51c51551ac039ae6dfc7973795dcb6dd580a5e432f17150fd179",
    "DOM": "47c556bf296912948346fcfaf083ce59f2dfb2632e8980275cf4e3e5bde7faf6",
    "ECU": "3ee6170958937972f940cc4113b71c4dfca8f01954fc8b83ddb697495d353880",
    "MEX": "bf157d29b6ddfb1395e7fa3b5782ab72e2d37643bf893f5d8e5ff25600a5efd7",
    "NIC": "93227a5702009410d63f3300eef90c52065842fa53d83018efcbaea03ab4708c",
    "PAN": "d3779e1bb81a53b07a48d2821aebad2a45e802517cdfc1db8eef8338cf4b418e",
    "PER": "9ad3553b8ab209e3486736469b5cb02be7b3e7db3547f7a50f4e3baf03d1beca",
    "TWN": "95fbce1839c96d011a08e1163f4f42e71caf0c23d5e91e0ad9e0b621d67e18a4",
    # dengue + influenza:japan  (JPN serves both: the dengue graph and Japan's C)
    "JPN": "ca44b4e357784a553c7301093cdd1d366a6661e0b171f756dc19649240aa20fa",
    # influenza:us-states + us-regions  (C only -- the shipped adjacency is reused verbatim)
    "USA": "63d30dc9c870f3e8bb835be3da46221cfbc413cdedacb6544098c1f671f24807",
    # ebola -- the three core countries (to_schema.EBOLA_GADM)
    "GIN": "ccf7426221fa95ee24770eddeff0ce8bb1e0515468e84f434906165594e7191e",
    "LBR": "d70b934961c6886100a096ad91a86129f42649099916c8011a11aeb4249c9249",
    "SLE": "ac367df651ec1577b3fd9eb1b5a0cee0da8d536e2dd0e479c71a1e03eeadc7f5",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="verify existing files only; do not download")
    args = ap.parse_args()

    GADM_DIR.mkdir(parents=True, exist_ok=True)
    bad: list[str] = []

    for iso3, want in sorted(MANIFEST.items()):
        zp = GADM_DIR / f"gadm41_{iso3}_shp.zip"

        if not zp.exists():
            if args.verify:
                print(f"MISSING  {zp}")
                bad.append(iso3)
                continue
            url = URL.format(iso3=iso3)
            print(f"fetching {iso3} <- {url}")
            tmp = zp.with_suffix(".zip.part")          # never leave a half file at the real path
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(zp)

        got = sha256(zp)
        if got == want:
            print(f"ok       {zp.name}")
        else:
            print(f"MISMATCH {zp.name}\n  expected {want}\n  got      {got}")
            bad.append(iso3)

    if bad:
        print(f"\nFAILED: {len(bad)} file(s) missing or corrupt: {', '.join(sorted(bad))}")
        print("A checksum mismatch means GADM re-cut the release or the download is corrupt.")
        print("Do NOT rebuild the bundles until this is resolved -- the graphs would change.")
        return 1

    print(f"\nAll {len(MANIFEST)} GADM 4.1 shapefiles present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

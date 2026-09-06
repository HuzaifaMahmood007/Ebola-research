"""ebola_masking_audit.py -- M7 disclosure: per-district masked weeks, trailing runs, and how often the envelope's premise holds.

READ ONLY. Touches no bundle and no envelope logic. `ebola_L12.npz` is a hash-frozen pre-registered
arm; re-basing it would engage the pre-registration and is deliberately not done here.

to_schema.py:227-231 masks any week whose cumulative report sits below the running maximum, on the
stated premise that such a fall is "a single-week data-entry dropout, not a downward revision". A
dropout returns to the prior high-water mark quickly; a sustained lower branch is a revision. So:
count the downward steps and count how many recover within three weeks.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
os.chdir(Path(__file__).resolve().parent.parent)
import to_schema as ts

df = ts._read_ebola("data/Final datasets/data-ebola-public.xlsx")
c = ts.EBOLA_COLS
df = df[df[c["indicator"]].astype(str).str.strip().str.lower().eq("cases")].copy()
df["_node"] = df[c["district"]].map(ts._canon)
df["_country"] = df[c["country"]].map(ts._canon)
df["_date"] = pd.to_datetime(df[c["date"]], errors="coerce")
df[c["value"]] = pd.to_numeric(df[c["value"]], errors="coerce")
df = df.dropna(subset=["_date", c["value"]])

# Restrict to the districts that actually reach the released bundle. The raw sheet also carries the
# National aggregate, multi-district blobs, non-core countries and two Sierra Leone non-districts,
# all dropped by load_ebola with a declared reason. Counting them would inflate every figure here.
import json
_z = np.load("data/processed/ebola_L12.npz", allow_pickle=True)
_meta = json.loads(str(_z["meta_json"]))
KEEP = {nid.split("|", 1)[1] for nid in _meta["node_ids"]}
print(f"released bundle districts: {len(KEEP)}")
# Apply the loader's own name repairs first. The source is internally inconsistent (11 of
# Liberia's 15 counties carry a "County" suffix, 4 do not; one Sierra Leone district is torn
# across two labels), and without this "lofa county" never matches the bundle's "lofa".
df["_node"] = [ts.EBOLA_NAME_ALIASES.get(co, {}).get(no, no)
               for co, no in zip(df["_country"], df["_node"])]
df = df[df["_country"].isin(ts.EBOLA_CORE_COUNTRIES) & df["_node"].isin(KEEP)].copy()
_seen = set(df["_node"])
print(f"bundle districts matched in the raw sheet: {len(_seen & KEEP)} of {len(KEEP)}")
_miss = sorted(KEEP - _seen)
if _miss:
    print(f"  unmatched: {_miss}")

RECOVER_WITHIN = 3
rows, total_falls, recovered = [], 0, 0

for node, g in df.groupby("_node"):
    s = g.set_index("_date")[c["value"]].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    wk = s.resample(ts.WEEK_ANCHOR).last()
    observed = wk.notna()
    cum = wk.ffill()
    env = cum.cummax()
    fell = observed & (wk < env.shift())
    idx = list(wk.index)
    n_fall = int(fell.sum())
    if n_fall == 0:
        continue
    n_rec = 0
    for i, when in enumerate(idx):
        if not bool(fell.get(when, False)):
            continue
        prior = env.shift().get(when)
        window = wk.iloc[i + 1: i + 1 + RECOVER_WITHIN].dropna()
        if len(window) and (window >= prior).any():
            n_rec += 1
    total_falls += n_fall
    recovered += n_rec
    # trailing run of masked weeks at the end of the node's observed span
    tail = 0
    for when in reversed(idx):
        if bool(fell.get(when, False)):
            tail += 1
        elif bool(observed.get(when, False)):
            break
    rows.append((node, n_fall, n_rec, tail))

print(f"nodes with at least one downward step: {len(rows)}")
print(f"TOTAL downward steps masked: {total_falls}")
print(f"recovered to the prior high-water mark within {RECOVER_WITHIN} weeks: "
      f"{recovered} ({100 * recovered / total_falls:.0f}%)")
print(f"=> the stated 'single-week dropout' premise holds for {100 * recovered / total_falls:.0f}% "
      f"of them; it fails for {100 * (1 - recovered / total_falls):.0f}%")
print(f"districts terminating in >=4 consecutive masked weeks: "
      f"{sum(1 for r in rows if r[3] >= 4)}")
top10 = sum(r[1] for r in sorted(rows, key=lambda r: -r[1])[:10])
print(f"the 10 districts with the most masked weeks hold {top10} of {total_falls} "
      f"({100 * top10 / total_falls:.0f}%), while {len(rows)} districts are touched in all")
print()

print("| district | masked weeks | of which recover within 3 wk | trailing masked run |")
print("|---|---|---|---|")
for node, n_fall, n_rec, tail in sorted(rows, key=lambda r: (-r[1], r[0])):
    b = "**" if tail >= 4 else ""
    print(f"| {node} | {n_fall} | {n_rec} | {b}{tail}{b} |")



# --- source pin, checked before any number above is trusted -----------------------------
import hashlib

import build_datasets as _bd

_h = hashlib.sha256(Path("data/Final datasets/data-ebola-public.xlsx").read_bytes()).hexdigest()
assert _h == _bd.RAW_SHA256[_bd.EBOLA_XLSX], "ebola source does not match its pin; numbers void"
print()
print("source sha256 matches build_datasets.RAW_SHA256")


# --- --check: is the filed 3.5.1 table still what this computes? ------------------------------
if "--check" in sys.argv:
    import re
    doc = Path("progress/planning/data_audit.md").read_text(encoding="utf-8")
    sec = doc.split("#### 3.5.1")[1].split("### 3.6 ")[0]
    filed = dict(re.findall(r"^\| ([a-z ]+) \| (\d+) \|", sec, re.M))
    live = {n: str(f) for n, f, _r, _t in rows}
    bad = []
    for n, f in live.items():
        if filed.get(n) != f:
            bad.append(f"{n}: doc {filed.get(n)}, disk {f}")
    for n in filed:
        if n not in live:
            bad.append(f"{n}: in doc, not computed")
    for label, want in [("368 removed cells", total_falls), ("only 115 return", recovered)]:
        if str(want) not in sec:
            bad.append(f"prose lost {label!r} (computed {want})")
    if f"{100 * recovered / total_falls:.0f} percent" not in sec:
        bad.append("prose recovery percentage does not match")
    for b in bad:
        print("MISMATCH:", b)
    print("data_audit 3.5.1:", "verified" if not bad else f"{len(bad)} mismatches")
    raise SystemExit(1 if bad else 0)

"""Verify two properties of the real OpenDengue file that the loader depends on.

  (1) Case counts are per-period incidence, NOT cumulative -> the series is not differenced.
  (2) Every weekly record falls in its true epidemiological week.

Both were assumed by the loader and are checked here against the actual file. A third property,
the count of zero-variance nodes, is reported but not gated -- those zeros are true, not a
defect, and the nodes are deliberately retained.

Run: python test_dengue_7_1.py
"""
import numpy as np
import pandas as pd

from to_schema import DENGUE_COLS as C, WEEK_ANCHOR, _canon

CSV = r"data/Final datasets/OpenDengue_Best_Spacial.csv"

df = pd.read_csv(CSV, sep=None, engine="python")
df = df[df[C["t_res"]].map(_canon) == "week"].copy()
df["_d"] = pd.to_datetime(df[C["start"]], errors="coerce")
df["_v"] = pd.to_numeric(df[C["value"]], errors="coerce")
assert df["_d"].isna().sum() == 0, "weekly dates failed to parse (do NOT reintroduce dayfirst)"

# --- (1) per-period incidence, not cumulative ------------------------------------
# Key each node at ITS OWN level, exactly as load_dengue does. Keying Admin2 rows on
# adm_1_name collapses municipalities into states (2.08M of 2.24M rows would share a
# key) and makes the diff meaningless.
lvl = df[C["s_res"]].map(_canon)
df = df[lvl.isin(["admin1", "admin2"])].copy()          # Admin0 is never a node level
lvl = df[C["s_res"]].map(_canon)
df["_n"] = df[C["adm0"]].map(_canon) + "|" + np.where(
    lvl == "admin2",
    df[C["adm2"]].astype(str).map(_canon),
    df[C["adm1"]].astype(str).map(_canon))

big = df.dropna(subset=["_d", "_v"]).groupby("_n").filter(lambda g: len(g) >= 52)

# A constant series (e.g. an all-zero Japanese prefecture) is trivially non-decreasing
# and says NOTHING about cumulative-vs-incidence. Exclude it from the test, then count
# it separately below — conflating the two is what made the first version of this test
# fire a false positive.
nuniq = big.groupby("_n")["_v"].nunique()
flat = nuniq[nuniq <= 1].index
varying = big[~big["_n"].isin(flat)]

mono = (varying.sort_values("_d").groupby("_n")["_v"]
        .apply(lambda s: bool((s.diff().dropna() >= 0).all())))
frac = mono.mean() if len(mono) else 0.0
print(f"(1) non-constant series: {len(mono)}   monotone non-decreasing: {mono.sum()} ({frac:.1%})")
assert frac < 0.05, f"{frac:.1%} of series look CUMULATIVE — target would need diffing"

# --- (2) MMWR W-SAT binning ------------------------------------------------------
# OpenDengue weekly records run Sunday->Saturday; W-SAT weeks end Saturday. Each record
# must land on its own W-SAT period start, whatever the labelling convention.
dow = df["_d"].dt.dayofweek.value_counts()              # Mon=0 .. Sun=6
assert dow.index.tolist() == [6], f"weekly records must start on Sunday, got {dow.to_dict()}"

per = df["_d"].dt.to_period(WEEK_ANCHOR)
landed = (per.dt.start_time.dt.normalize() == df["_d"].dt.normalize()).mean()
print(f"(2) records starting Sunday: 100%   landing on their W-SAT week start: {landed:.4%}")
assert landed > 0.999, "records are not binning into their own epi-week — W-SAT anchor wrong"

end = pd.to_datetime(df["calendar_end_date"], errors="coerce")
span = (end - df["_d"]).dt.days
seven = (span == 6).mean()
print(f"(2b) records spanning exactly 7 days: {seven:.4%}   "
      f"(8-day outliers: {int((span == 7).sum())})")
assert seven > 0.99, "weekly records do not span 7 days"

# --- zero-variance diagnostic (REPORTED, not a gate) -----------------------------
# Nodes whose observed cells never vary carry no forecastable signal, yet they clear the
# coverage gate (it counts observed weeks, never asks whether they vary) and their scaler
# degenerates to mean=0/std=1 via the sd<1e-8 guard — silently; .check() still passes.
# Confined to Japan (29 of 47 prefectures, all-zero across 312 observed weeks) + 1 Ecuador
# node, and those zeros are TRUE (dengue is non-endemic in Japan), so they are KEPT per
# client decision. They own 0.4% of observed cells, so node-micro scoring is unaffected;
# the exposure is the country-macro headline (decision 6), where Japan is 1/12 of the
# score and 62% of its nodes are constant. Recorded here, deliberately not enforced.
zv = big[big["_n"].isin(flat)].groupby("_n").agg(obs=("_v", "size"), vmax=("_v", "max"))
allzero = zv[zv["vmax"] == 0]
print(f"\n(3) zero-variance nodes (>=52 wk, reported only): {len(zv)}  of which all-zero: {len(allzero)}")
by_country = allzero.index.to_series().str.split("|").str[0].value_counts()
for c, n in by_country.items():
    print(f"      {c:<12} {n:>3} all-zero nodes")

print("\n7.1 PASSED — per-period incidence confirmed; W-SAT binning correct.")

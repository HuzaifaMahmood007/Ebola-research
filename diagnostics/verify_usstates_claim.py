"""verify_usstates_claim.py -- check the manuscript's US-states paragraph against both aggregations.

The paragraph makes a claim that is true under one aggregation and false under the other, which is
precisely why it is in the paper. So it needs checking under both: country-macro RMSE paired by seed
from the run JSONs, and cell-pooled RMSE from the filed reproduction table. If either moves, the
paragraph is wrong and this fails.

    conda run -n ebola-train python -m diagnostics.verify_usstates_claim

Original docstring: Check the manuscript's new US-states paragraph against both aggregations on disk."""
import collections, json, re
import os, sys
from pathlib import Path
import numpy as np
os.chdir(Path(__file__).resolve().parent.parent)

def recs(paths, metric="rmse"):
    out = collections.defaultdict(dict)
    for f in paths:
        for r in json.load(open(f)):
            if r["metric"] == metric:
                out[(r["model"], r["dataset"], r["horizon"])].setdefault(r.get("seed"), r["country_macro"])
    return out

base = recs(sorted(Path("results/baselines").glob("*__influenza_us-states__*.json")))
enc  = recs(sorted(Path("results/single").glob("encoder__influenza_us-states__*.json")))
ours = {h: v for (m, d, h), v in enc.items() if m == "encoder"}

print("country-macro RMSE, us-states, paired by seed (level when |mean| < sd):")
verdicts = {}
for model in ("EpiGNN", "HeatGNN"):
    row = []
    for h in (3, 5, 10, 15):
        b, o = base[(model, "influenza_us-states", h)], ours[h]
        s = sorted(set(b) & set(o))
        d = np.array([b[k] - o[k] for k in s])
        m, sd = d.mean(), d.std(ddof=1)
        v = "level" if abs(m) < sd else ("better" if m > 0 else "worse")
        row.append(v)
    verdicts[model] = row
    print(f"  {model:8} h3/h5/h10/h15 = {row}")

tbl = Path("Reports/baseline_reproduction_table.md").read_text(encoding="utf-8")
pooled = {}
for model in ("EpiGNN", "HeatGNN"):
    # column 8 is "encoder vs reproduced"; column 6 is "delta vs published" and grabbing that
    # one instead is exactly the mistake this comment exists to stop repeating.
    vals = []
    for line in tbl.splitlines():
        f = [c.strip() for c in line.split("|")]
        if len(f) > 9 and f[1] == model and f[2] == "influenza_us-states":
            vals.append(float(f[8].rstrip("%")))
    pooled[model] = vals
    print(f"  {model:8} pooled deltas = {vals}  (positive = we are worse)")

doc = Path("Reports/Manuscript_v2.md").read_text(encoding="utf-8")
bad = []
if verdicts["EpiGNN"] != ["worse", "level", "level", "better"]:
    bad.append(f"EpiGNN macro pattern changed: {verdicts['EpiGNN']}")
if verdicts["HeatGNN"] != ["level"] * 4:
    bad.append(f"HeatGNN macro pattern changed: {verdicts['HeatGNN']}")
lo, hi = min(pooled["EpiGNN"]), max(pooled["EpiGNN"])
if not all(v > 0 for v in pooled["EpiGNN"]):
    bad.append("EpiGNN pooled: not behind at every horizon any more")
if f"{lo:.1f}% to {hi:.1f}%" not in doc:
    bad.append(f"doc range does not match pooled {lo:.1f}% to {hi:.1f}%")
n_worse = sum(v > 0 for v in pooled["HeatGNN"])
if f"behind HeatGNN at three of four" not in doc or n_worse != 3:
    bad.append(f"HeatGNN pooled worse count is {n_worse}, doc says three of four")
for b in bad:
    print("MISMATCH:", b)
print("manuscript us-states paragraph:", "verified" if not bad else f"{len(bad)} mismatches")
raise SystemExit(1 if bad else 0)

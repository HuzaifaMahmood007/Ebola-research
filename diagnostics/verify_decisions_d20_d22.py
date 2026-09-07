"""verify_decisions_d20_d22.py -- re-derive the disk-backed numbers in decisions D20 to D22.

Each of the three is a decision NOT to run something, and each rests on a measurement. If the
measurement moves, the decision has to be revisited rather than quietly inherited. D21 and D22 also
assert that a specific sentence is already in the manuscript, so those are checked too: a decision
that says "already disclosed" is wrong the moment the disclosure is edited out.

    conda run -n ebola-train python -m diagnostics.verify_decisions_d20_d22
"""
import os, sys
os.chdir(Path(__file__).resolve().parent.parent) if False else None
import json, re
import os
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parent.parent)

t = Path("progress/decisions/decisions.md").read_text(encoding="utf-8")
sec = t.split("## D20")[1].split("## Reversed or superseded")[0]
bad = []

# D20: zero-shot quantile archives
zs_json = len(list(Path("results/lodo").glob("encoder_ldo3_zeroshot__*.json")))
zs_q = len(list(Path("results/lodo").glob("encoder_ldo3_zeroshot__*__quantiles.npz")))
ad_q = len(list(Path("results/lodo").glob("encoder_ldo3__*__quantiles.npz")))
m = re.search(r"holds (\d+) `encoder_ldo3_zeroshot__\*\.json` records and \*\*(\d+) matching", sec)
if not m or (int(m.group(1)), int(m.group(2))) != (zs_json, zs_q):
    bad.append(f"D20 zero-shot counts: doc {m.groups() if m else None}, disk ({zs_json}, {zs_q})")
m = re.search(r"against (\d+) present for the adapted", sec)
if not m or int(m.group(1)) != ad_q:
    bad.append(f"D20 adapted quantiles: doc {m.group(1) if m else None}, disk {ad_q}")

# D21: gate tally, straight from the log
metric, tally = None, {"err": [0, 0], "pcc": [0, 0]}
for line in Path("Reports/gate_ablation.log").read_text(errors="replace").splitlines():
    h = re.match(r"^\s{4}(RMSE|MAE|PCC)\b", line)
    if h:
        metric = "pcc" if h.group(1) == "PCC" else "err"
        continue
    if metric and re.match(r"^\s+(3|5|10|15) \|", line):
        tally[metric][0] += "GATE HELPS" in line
        tally[metric][1] += "gate HURTS" in line
m = re.search(r"helps \*\*(\d+) of (\d+)\*\* error cells and hurts (\d+)", sec)
if not m or (int(m.group(1)), int(m.group(3))) != tuple(tally["err"]):
    bad.append(f"D21 error tally: doc {m.groups() if m else None}, log {tally['err']}")
m = re.search(r"helps (\d+) of (\d+) correlation cells", sec)
if not m or int(m.group(1)) != tally["pcc"][0]:
    bad.append(f"D21 pcc tally: doc {m.group(1) if m else None}, log {tally['pcc'][0]}")

# D21 + D22: the manuscript sentences the decisions claim already exist
ms = Path("Reports/Manuscript_v2.md").read_text(encoding="utf-8")
for label, needle in [
    ("D21 shuffled-adjacency disclosure", "We did not run a shuffled-adjacency arm"),
    ("D22 median/MAE sentence", "The direction is conservative, so we lead with MAE"),
]:
    if needle not in ms:
        bad.append(f"{label} no longer in the manuscript: {needle!r}")

# D22: the COVID penalty, at source
rc = Path("progress/outcomes/Report_Covid.md").read_text(encoding="utf-8")
if "+8.8 / +21.4 / +47.3 / +27.4 %" not in rc:
    bad.append("D22 COVID penalty not found at Report_Covid.md")
if "8.8 / 21.4 / 47.3 / 27.4 percent worse" not in sec:
    bad.append("D22 doc no longer quotes the COVID penalty")

for b in bad:
    print("MISMATCH:", b)
print("D20-D22:", "verified" if not bad else f"{len(bad)} mismatches")
raise SystemExit(1 if bad else 0)

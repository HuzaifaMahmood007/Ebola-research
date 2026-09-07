"""verify_status.py -- re-derive every number in progress/STATUS.md from the artifacts.

STATUS.md is now the ONLY status document; seven dated summaries were deleted on its authority.
That makes it the single point of failure for "what is the state of this project", so it gets the
same treatment as every results document here: numbers parsed back out and recomputed from disk.

It also guards the retirement itself in both directions, so neither a silent restore of a deleted
summary nor an accidental deletion of the retained Doubt.md passes unnoticed.

    conda run -n ebola-train python -m diagnostics.verify_status
"""
import os
import glob
import re
from pathlib import Path
import numpy as np

os.chdir(Path(__file__).resolve().parent.parent)

t = Path("progress/STATUS.md").read_text(encoding="utf-8")
bad = []
def eq(label, got, pat):
    m = re.search(pat, t)
    if not m:
        bad.append(f"{label}: sentence not found"); return
    said = int(m.group(1).replace(",", ""))
    if said != got:
        bad.append(f"{label}: doc {said}, disk {got}")

n = lambda p: len(glob.glob(p))
# section 2 table
for fam, pat in [("single","single-disease ceilings \| (\d+) \|"),
                 ("lodo","all LODO families \| (\d+) \|"),
                 ("ebola","Ebola case study \| (\d+) \|"),
                 ("baselines","published baselines \| (\d+) \|"),
                 ("joint","joint multi-disease \| (\d+) \|"),
                 ("naive","naive floors \| (\d+) \|"),
                 ("misc","including ANIL \| (\d+) \|")]:
    eq(f"records {fam}", n(f"results/{fam}/*.json"), pat)
eq("quantiles single", n("results/single/*__quantiles.npz"), r"ceilings \| \d+ \| (\d+) \|")
eq("quantiles lodo", n("results/lodo/*__quantiles.npz"), r"LODO families \| \d+ \| (\d+) \|")
eq("quantiles ebola", n("results/ebola/*__quantiles.npz"), r"case study \| \d+ \| (\d+) \|")
eq("checkpoints", len(list(Path("results").rglob("*ckpt.pt"))), r"\*\*(\d+) trained checkpoints\*\*")
eq("tracked logs", n("results/reports/*.log"), r"\*\*(\d+) decision-bearing run logs\*\*")
eq("ldo3 zeroshot records", n("results/lodo/encoder_ldo3_zeroshot__*.json"),
   r"zero-shot arm has (\d+) records")
eq("ldo3 zeroshot quantiles", n("results/lodo/encoder_ldo3_zeroshot__*__quantiles.npz"),
   r"and \*\*(\d+)\*\* archives")
eq("single ckpts", n("results/single/encoder__*ckpt.pt"), r"(\d+) single-disease checkpoints now")
eq("manuscript words", len(Path("Reports/Manuscript_v2.md").read_text(encoding='utf-8').split()),
   r"\*\*Manuscript\*\* \| ([\d,]+) words")
eq("tracked Reports", len([l for l in __import__("subprocess").run(
        ["git","ls-files","Reports/"],capture_output=True,text=True).stdout.splitlines()]),
   r"tracks (\d+) files in `Reports/`")

# LDO3 breakdown must sum to the lodo total
m = re.search(r"LDO3 proper (\d+), two-way LDO (\d+), population LODO (\d+), and the\s*\n?"
              r"graph-controlled `encoder_pair` arm (\d+)", t)
if not m:
    bad.append("LDO3 breakdown sentence not found")
elif sum(int(g) for g in m.groups()) != n("results/lodo/*.json"):
    bad.append(f"LDO3 breakdown sums to {sum(int(g) for g in m.groups())}, dir has {n('results/lodo/*.json')}")

# MTGNN constant count
const = sum(len(np.unique((lambda a: a[np.isfinite(a)])(
    np.asarray(np.load(f, allow_pickle=True)["preds"], float)))) == 1
    for f in glob.glob("baselines/_preds/MTGNN__*.npz"))
eq("MTGNN constant", const, r"\*\*(\d+) of the 80 prediction files hold")

# Section 6 names eight documents. Seven were deleted on 2026-09-07 and must stay deleted;
# Doubt.md is deliberately kept because eight source files cite it. Guard both directions, so
# neither a silent restore nor an accidental deletion of the retained one goes unnoticed.
named = re.findall(r"^\| `([A-Za-z0-9_]+\.md)` \| 2026", t, re.M)
if len(named) != 8:
    bad.append(f"section 6 lists {len(named)} retired documents, expected 8")
for name in named:
    exists = Path("progress/summaries", name).exists()
    if name == "Doubt.md" and not exists:
        bad.append("Doubt.md was deleted; 8 source files cite it by section")
    if name != "Doubt.md" and exists:
        bad.append(f"{name} is back in progress/summaries/ but STATUS.md calls it retired")

for b in bad:
    print("MISMATCH:", b)
print("STATUS.md:", "verified" if not bad else f"{len(bad)} mismatches")
raise SystemExit(1 if bad else 0)

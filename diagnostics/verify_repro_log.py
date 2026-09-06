"""verify_repro_log.py -- re-derive every disk-backed number in Reports/reproduction_failure_log.md.

The log states counts that came off the artifacts: how many baseline prediction files collapsed to a
single value, how many nodes are isolated in each exported graph, how many dengue nodes survive the
subsample, and how many scored records each baseline has. Standing repo rule is that a results
document is signed off only after its numbers are parsed back OUT of it and recomputed from disk, so
that is what this does. It reads the markdown, not the script that wrote it.

Published figures (the Cola-GNN / HeatGNN tolerance table, the STOEP rows) are transcribed from
papers and are deliberately NOT checked here -- there is nothing on disk to check them against.

Note `baselines/` is gitignored, so this only runs on a machine that still holds the prediction
archives. It exits 2, not 1, if they are missing, to keep "cannot check" distinct from "wrong".

    conda run -n ebola-train python diagnostics/verify_repro_log.py [--mutate]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "Reports" / "reproduction_failure_log.md"
PRED = ROOT / "baselines" / "_preds"
EXPORTED = ROOT / "baselines" / "_exported"
SCORED = ROOT / "results" / "baselines"

FNAME = re.compile(r"(\w+)__([\w-]+)__h(\d+)__seed(\d+)\.npz")


# ---------------------------------------------------------------- recompute

def constant_counts() -> dict[str, tuple[int, int]]:
    """model -> (files with exactly one distinct finite value, total files)."""
    const: dict[str, int] = {}
    total: dict[str, int] = {}
    for f in sorted(PRED.glob("*.npz")):
        m = FNAME.match(f.name)
        assert m, f"unparsed prediction file {f.name}"
        model = m.group(1)
        a = np.asarray(np.load(f, allow_pickle=True)["preds"], dtype=float)
        a = a[np.isfinite(a)]
        total[model] = total.get(model, 0) + 1
        if len(np.unique(a)) == 1:
            const[model] = const.get(model, 0) + 1
    return {k: (const.get(k, 0), v) for k, v in total.items()}


def isolated_counts() -> dict[str, int]:
    """exported panel -> number of degree-0 nodes, self-loops excluded."""
    out = {}
    for d in sorted(EXPORTED.iterdir()):
        adj = np.loadtxt(d / "adj.txt", delimiter=",")
        off = adj - np.diag(np.diag(adj))
        out[d.name] = int((off > 0).sum(axis=1).__eq__(0).sum())
    return out


def dengue_nodes() -> tuple[int, int]:
    full = np.load(ROOT / "data" / "processed" / "dengue.npz", allow_pickle=True)["X"].shape[0]
    kept = json.loads((EXPORTED / "dengue" / "meta.json").read_text())["n_nodes"]
    return full, kept


def stoep_shapes() -> dict[str, tuple]:
    """Shapes of the only dataset STOEP's repo ships. Settles which paper row our run matches."""
    p = ROOT / "baselines" / "STOEP-Epidemic-Forecasting" / "data" / "jp20200401_20210921.npy"
    d = np.load(p, allow_pickle=True).item()
    return {k: np.asarray(v).shape for k, v in d.items()}


def scored_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for f in SCORED.glob("*.json"):
        model = f.name.split("__")[0]
        out[model] = out.get(model, 0) + 1
    return out


# ------------------------------------------------------------------- checks

def check(text: str) -> list[str]:
    bad: list[str] = []

    # Section C table rows: | MTGNN | **47** | 80 |
    stated = {}
    for m in re.finditer(r"^\|\s*(MTGNN|EpiGNN|Cola-GNN|HeatGNN)\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*(\d+)\s*\|$",
                         text, re.M):
        stated[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    assert len(stated) == 4, f"expected 4 constant-count rows, parsed {len(stated)}"

    actual = constant_counts()
    alias = {"Cola-GNN": "ColaGNN"}
    for model, (c, t) in stated.items():
        ac, at = actual[alias.get(model, model)]
        if (c, t) != (ac, at):
            bad.append(f"constant counts {model}: doc says {c}/{t}, disk says {ac}/{at}")

    # Prose: "47 of 80 MTGNN prediction files" must agree with the table.
    m = re.search(r"\*\*(\d+) of (\d+) MTGNN prediction files", text)
    assert m, "prose sentence naming the MTGNN constant count not found"
    if (int(m.group(1)), int(m.group(2))) != actual["MTGNN"]:
        bad.append(f"prose MTGNN count {m.group(1)}/{m.group(2)} != disk {actual['MTGNN']}")

    # Isolated nodes.
    iso = isolated_counts()
    m = re.search(r"influenza_japan has (\d+) isolated nodes?,\s*\n?influenza_us-states has (\d+), "
                  r"influenza_us-regions has (\d+)", text)
    assert m, "isolated-node sentence not found"
    for panel, got in zip(("influenza_japan", "influenza_us-states", "influenza_us-regions"), m.groups()):
        if int(got) != iso[panel]:
            bad.append(f"isolated nodes {panel}: doc says {got}, disk says {iso[panel]}")

    # Dengue node counts and the retained fraction.
    full, kept = dengue_nodes()
    m = re.search(r"([\d,]+) nodes against ([\d,]+), so ([\d.]+) percent retained", text)
    assert m, "dengue node-count sentence not found"
    d_full, d_kept, d_pct = (int(m.group(1).replace(",", "")),
                             int(m.group(2).replace(",", "")), float(m.group(3)))
    if (d_full, d_kept) != (full, kept):
        bad.append(f"dengue nodes: doc says {d_full}/{d_kept}, disk says {full}/{kept}")
    if abs(d_pct - 100 * kept / full) > 0.05:
        bad.append(f"dengue retained pct: doc says {d_pct}, disk says {100 * kept / full:.1f}")

    # Scored record counts named in the Cola/Heat entry.
    sc = scored_counts()
    m = re.search(r"EpiGNN (\d+) records including dengue, MTGNN (\d+)\s*\n?including dengue, "
                  r"Cola-GNN (\d+) and HeatGNN (\d+)", text)
    assert m, "scored-record sentence not found"
    for model, got in zip(("EpiGNN", "MTGNN", "ColaGNN", "HeatGNN"), m.groups()):
        if int(got) != sc[model]:
            bad.append(f"scored records {model}: doc says {got}, disk says {sc[model]}")

    # STOEP: the shipped shapes are what closes A3, so they are quoted and must hold.
    st = stoep_shapes()
    m = re.search(r"`od \(([\d, ]+)\)`, `node \(([\d, ]+)\)` and\s*\n?`SIR \(([\d, ]+)\)`", text)
    assert m, "STOEP shipped-shape sentence not found"
    for key, got in zip(("od", "node", "SIR"), m.groups()):
        want = tuple(int(x) for x in got.split(","))
        if want != st[key]:
            bad.append(f"STOEP {key} shape: doc says {want}, disk says {st[key]}")
    if st["node"][1] != 47 or st["node"][0] != 539:
        bad.append("STOEP shipped data is no longer the 539-day 47-prefecture COVID panel")

    # The survivor table must call MTGNN unusable and EpiGNN usable.
    if "| MTGNN | no | no | **no** |" not in text:
        bad.append("survivor table no longer marks MTGNN unusable")
    if "**Two usable comparators on influenza, one on dengue.**" not in text:
        bad.append("survivor-count sentence missing or changed")

    return bad


MUTATIONS = [
    ("MTGNN constant count in the table", lambda t: t.replace("| MTGNN | **47** | 80 |", "| MTGNN | **41** | 80 |")),
    ("MTGNN constant count in the prose", lambda t: t.replace("**47 of 80 MTGNN prediction files", "**44 of 80 MTGNN prediction files")),
    ("a clean baseline made dirty", lambda t: t.replace("| EpiGNN | 0 | 80 |", "| EpiGNN | 3 | 80 |")),
    ("isolated node count", lambda t: t.replace("influenza_us-states has 2", "influenza_us-states has 4")),
    ("dengue subsample size", lambda t: t.replace("7,165 nodes against 2,392", "7,165 nodes against 2,400")),
    ("dengue retained fraction", lambda t: t.replace("so 33.4 percent retained", "so 30.0 percent retained")),
    ("scored record count", lambda t: t.replace("Cola-GNN 60 and HeatGNN 60", "Cola-GNN 80 and HeatGNN 60")),
    ("MTGNN quietly promoted to usable", lambda t: t.replace("| MTGNN | no | no | **no** |", "| MTGNN | no | yes | **yes** |")),
    ("STOEP shipped node count", lambda t: t.replace("`node (539, 47, 4)`", "`node (539, 11, 4)`")),
]


def main() -> int:
    if not PRED.exists() or not EXPORTED.exists():
        print("baselines/ artifacts absent (gitignored); cannot verify")
        return 2
    text = DOC.read_text(encoding="utf-8")

    if "--mutate" in sys.argv:
        rc = 0
        for label, mutate in MUTATIONS:
            corrupted = mutate(text)
            assert corrupted != text, f"mutation '{label}' changed nothing; it no longer targets the doc"
            try:
                caught = bool(check(corrupted))
            except AssertionError:
                caught = True                      # a structural break is also a catch
            print(f"  {'CAUGHT ' if caught else 'MISSED '} {label}")
            rc |= 0 if caught else 1
        print("mutation test:", "all caught" if rc == 0 else "SOME MISSED")
        return rc

    bad = check(text)
    for b in bad:
        print("MISMATCH:", b)
    print(f"{DOC.name}: {'all disk-backed numbers verified' if not bad else f'{len(bad)} mismatches'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""verify_g5_scope.py -- recompute every disk-backed number in the G5 scoping note.

The note is a decision record, and its decisions rest on measurements: how many checkpoints exist
(which is what makes attribution inference-only), how the gate-off ablation actually fell (which is
what forbids framing neighbour attribution as accuracy), and which panels have a constant obs_mask
(which is what makes that channel's attribution meaningless on some panels and a leak on others).
If any of those moved, the scope has to move with them.

Standing repo rule: numbers get parsed OUT of the document and recomputed, prose included.

    conda run -n ebola-train python -m diagnostics.verify_g5_scope [--mutate]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOC = ROOT / "progress" / "decisions" / "G5_Explainability_Scope.md"
GATE_LOG = ROOT / "Reports" / "gate_ablation.log"

PANELS = {
    "influenza_japan": "influenza_japan",
    "influenza_us-regions": "influenza_us-regions",
    "influenza_us-states": "influenza_us-states",
    "covid_us-states": "covid_us-states",
    "dengue": "dengue",
    "ebola_L12 / L20": "ebola_L12",
}


# ---------------------------------------------------------------- recompute

def checkpoints() -> dict[str, int]:
    """family -> count, families keyed the way the note groups them."""
    out = {"total": 0, "single": 0, "ebola": 0, "ebola_smoke": 0, "ldo3": 0, "ldo3full": 0}
    for p in (ROOT / "results").rglob("*ckpt.pt"):
        out["total"] += 1
        n = p.name
        if n.startswith("encoder__"):
            out["single"] += 1
        elif n.startswith("encoder_ebola_smoke__"):
            out["ebola_smoke"] += 1
        elif n.startswith("encoder_ebola__"):
            out["ebola"] += 1
        elif n.startswith("encoder_ldo3full__"):
            out["ldo3full"] += 1
        elif n.startswith("encoder_ldo3__"):
            out["ldo3"] += 1
    return out


def gate_tally() -> dict[str, tuple[int, int, int]]:
    """metric family -> (helps, hurts, within noise), straight from the ablation log."""
    metric, out = None, {"error": [0, 0, 0], "pcc": [0, 0, 0]}
    for line in GATE_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s{4}(RMSE|MAE|PCC)\b", line)
        if m:
            metric = "pcc" if m.group(1) == "PCC" else "error"
            continue
        # Only real data rows. The log's own legend line ("'within noise' = |mean| < sd") carries
        # both a pipe and the phrase, and counting it inflates PCC to 21 cells against a stated 20.
        if metric is None or not re.match(r"^\s+(3|5|10|15) \|", line):
            continue
        if "GATE HELPS" in line:
            out[metric][0] += 1
        elif "gate HURTS" in line:
            out[metric][1] += 1
        elif "within noise" in line:
            out[metric][2] += 1
    return {k: tuple(v) for k, v in out.items()}


def obs_mask_stats() -> dict[str, tuple[bool, float]]:
    """panel -> (is constant 1.0, fraction of cells observed)."""
    out = {}
    for label, stem in PANELS.items():
        z = np.load(ROOT / "data" / "processed" / f"{stem}.npz", allow_pickle=True)
        v = z["X"][:, :, 3].astype(float)
        v = v[np.isfinite(v)]
        u = np.unique(v)
        out[label] = (u.size == 1 and float(u[0]) == 1.0, float((v == 1).mean()))
    return out


def surface() -> tuple[list[str], list[int], int, tuple[int, int]]:
    import bundles
    z = np.load(ROOT / "data" / "processed" / "ebola_L12.npz", allow_pickle=True)
    m = json.loads(str(z["meta_json"]))
    d = np.load(ROOT / "data" / "processed" / "dengue.npz", allow_pickle=True)["X"]
    return (m["feature_names"], m["core_feature_idx"], bundles.W,
            (z["X"].shape[0], z["X"].shape[1]))


# ------------------------------------------------------------------- checks

def check(text: str) -> list[str]:
    bad: list[str] = []

    # --- checkpoint counts
    ck = checkpoints()
    m = re.search(r"\*\*(\d+) trained checkpoints are on disk\.\*\*", text)
    assert m, "checkpoint-count sentence not found"
    if int(m.group(1)) != ck["total"]:
        bad.append(f"checkpoints: doc says {m.group(1)}, disk has {ck['total']}")

    for label, key, pat in [
        ("single", "single", r"single-disease encoders, five panels \| (\d+) \|"),
        ("ebola", "ebola", r"Ebola arms .*? \| (\d+), plus (\d+) smoke \|"),
        ("ldo3", "ldo3", r"LDO3 trunks \| (\d+), plus (\d+) full-budget \|"),
    ]:
        m = re.search(pat, text)
        assert m, f"{label} checkpoint row not found"
        if int(m.group(1)) != ck[key]:
            bad.append(f"{label} checkpoints: doc says {m.group(1)}, disk has {ck[key]}")
    m = re.search(r"Ebola arms .*? \| \d+, plus (\d+) smoke \|", text)
    if int(m.group(1)) != ck["ebola_smoke"]:
        bad.append(f"ebola smoke ckpts: doc says {m.group(1)}, disk has {ck['ebola_smoke']}")
    m = re.search(r"LDO3 trunks \| \d+, plus (\d+) full-budget \|", text)
    if int(m.group(1)) != ck["ldo3full"]:
        bad.append(f"ldo3full ckpts: doc says {m.group(1)}, disk has {ck['ldo3full']}")

    # --- gate-off ablation table
    gt = gate_tally()
    m = re.search(r"\| error \(RMSE \+ MAE\) \| (\d+) \| \*\*(\d+)\*\* \| (\d+) \| (\d+) \|", text)
    assert m, "gate error row not found"
    cells, helps, hurts, noise = (int(g) for g in m.groups())
    if (helps, hurts, noise) != gt["error"] or cells != sum(gt["error"]):
        bad.append(f"gate error row: doc says {cells}/{helps}/{hurts}/{noise}, "
                   f"log gives {sum(gt['error'])}/{gt['error']}")
    m = re.search(r"\| correlation \(PCC\) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|", text)
    assert m, "gate PCC row not found"
    cells, helps, hurts, noise = (int(g) for g in m.groups())
    if (helps, hurts, noise) != gt["pcc"] or cells != sum(gt["pcc"]):
        bad.append(f"gate PCC row: doc says {cells}/{helps}/{hurts}/{noise}, "
                   f"log gives {sum(gt['pcc'])}/{gt['pcc']}")
    m = re.search(r"\| all \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|", text)
    assert m, "gate total row not found"
    tot = tuple(a + b for a, b in zip(gt["error"], gt["pcc"]))
    if tuple(int(g) for g in m.groups()[1:]) != tot or int(m.group(1)) != sum(tot):
        bad.append(f"gate total row: doc says {m.groups()}, log gives {sum(tot)}/{tot}")

    # --- the load-bearing prose claim behind 5.1
    if not re.search(r"helps error in zero of forty cells", text):
        bad.append("the 'zero of forty' framing sentence is gone")
    elif gt["error"][0] != 0 or sum(gt["error"]) != 40:
        bad.append(f"doc says zero of forty error cells; log gives "
                   f"{gt['error'][0]} of {sum(gt['error'])}")

    # --- obs_mask table
    om = obs_mask_stats()
    for label, (is_const, frac) in om.items():
        row = re.search(rf"^\| {re.escape(label)} \| (.+?) \| (.+?) \|$", text, re.M)
        assert row, f"obs_mask row for {label} not found"
        said_const = "constant 1.0" in row.group(1)
        said_frac = float(row.group(2).replace("**", ""))
        if said_const != is_const:
            bad.append(f"obs_mask {label}: doc says constant={said_const}, disk says {is_const}")
        if abs(said_frac - frac) > 5e-5:
            bad.append(f"obs_mask {label}: doc says {said_frac}, disk says {frac:.4f}")

    m = re.search(r"Only ([\d.]+)\s*\n?percent of dengue input cells are observed", text)
    assert m, "dengue observed-fraction sentence not found"
    if abs(float(m.group(1)) / 100 - om["dengue"][1]) > 5e-5:
        bad.append(f"dengue observed fraction in prose: doc says {m.group(1)}%, "
                   f"disk says {100 * om['dengue'][1]:.2f}%")

    n_const = sum(1 for v in om.values() if v[0])
    m = re.search(r"structurally zero on (\w+) of five panels", text)
    assert m, "obs_mask structural-zero sentence not found"
    words = {"three": 3, "four": 4, "five": 5}
    if words.get(m.group(1)) != n_const:
        bad.append(f"obs_mask constant panels: doc says {m.group(1)}, disk has {n_const}")

    # --- input surface
    names, core, w, (n_ebola, t_ebola) = surface()
    for i, nm in enumerate(names[:4]):
        if not re.search(rf"^\| {i} \| `{re.escape(nm)}` \|", text, re.M):
            bad.append(f"channel {i} row missing or renamed; bundle says `{nm}`")
    if not re.search(r"\*\*`\[N, 20, 4\]`\*\*", text) or w != 20:
        bad.append(f"input surface: doc says [N, 20, 4], bundles.W is {w}")
    if list(core) != [0, 1, 2, 3] or "deaths_norm" not in names:
        bad.append(f"ebola channel story wrong: core={core}, names={names}")
    if not re.search(rf"{n_ebola} districts over {t_ebola} weeks", text):
        bad.append(f"ebola shape: bundle is {n_ebola} nodes x {t_ebola} steps")

    m = re.search(r"dengue at \*\*([\d,]+) nodes over ([\d,]+) steps\*\*", text)
    assert m, "dengue scale sentence not found"
    dx = np.load(ROOT / "data" / "processed" / "dengue.npz", allow_pickle=True)["X"]
    if (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))) != dx.shape[:2]:
        bad.append(f"dengue scale: doc says {m.groups()}, bundle is {dx.shape[:2]}")

    # --- the SHAP claim sites must still be there; the note exists to close them
    for path, needle in [("PROJECT.md", "SHAP"), ("PROJECT.md", "SHAP global + local")]:
        if needle not in (ROOT / path).read_text(encoding="utf-8"):
            bad.append(f"{path} no longer contains {needle!r}; the note's section 1 is stale")

    return bad


MUTATIONS = [
    ("checkpoint total", lambda t: t.replace("**118 trained checkpoints", "**126 trained checkpoints")),
    ("single-disease checkpoint count", lambda t: t.replace("five panels | 26 |", "five panels | 25 |")),
    ("gate error row: a help appears", lambda t: t.replace("| error (RMSE + MAE) | 40 | **0** | 8 | 32 |",
                                                           "| error (RMSE + MAE) | 40 | **3** | 8 | 29 |")),
    ("gate PCC row", lambda t: t.replace("| correlation (PCC) | 20 | 6 | 1 | 13 |",
                                         "| correlation (PCC) | 20 | 9 | 1 | 10 |")),
    ("gate total row", lambda t: t.replace("| all | 60 | 6 | 9 | 45 |", "| all | 60 | 8 | 9 | 43 |")),
    ("the zero-of-forty framing removed", lambda t: t.replace("helps error in zero of forty cells",
                                                              "helps error in few cells")),
    ("dengue obs_mask fraction", lambda t: t.replace("| dengue | varies | **0.2175** |",
                                                     "| dengue | varies | **0.7175** |")),
    ("dengue obs_mask prose desynced", lambda t: t.replace("Only 21.75\npercent of dengue", "Only 71.75\npercent of dengue")),
    ("a varying panel called constant", lambda t: t.replace("| dengue | varies |", "| dengue | **constant 1.0** |")),
    ("constant-panel count", lambda t: t.replace("structurally zero on four of five panels",
                                                 "structurally zero on three of five panels")),
    ("input window", lambda t: t.replace("**`[N, 20, 4]`**", "**`[N, 12, 4]`**")),
    ("ebola shape", lambda t: t.replace("61 districts over 52 weeks", "61 districts over 60 weeks")),
    ("dengue scale", lambda t: t.replace("dengue at **7,165 nodes over 1,409 steps**",
                                         "dengue at **7,165 nodes over 1,400 steps**")),
    ("a channel renamed", lambda t: t.replace("| 3 | `obs_mask` |", "| 3 | `missing_mask` |")),
]


def main() -> int:
    text = DOC.read_text(encoding="utf-8")

    if "--mutate" in sys.argv:
        rc = 0
        for label, mutate in MUTATIONS:
            corrupted = mutate(text)
            assert corrupted != text, f"mutation '{label}' changed nothing; it no longer targets the doc"
            try:
                caught = bool(check(corrupted))
            except AssertionError:
                caught = True
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

"""verify_paper_table.py -- check Reports/baseline_reproduction_table.md against itself and the disk.

Two independent jobs, because the document makes two kinds of claim.

1. The table is generator output, so it must still BE the generator's output. Re-run
   paper_compare and compare row for row. This catches a stale file after any scoring change.
2. Every prose tally is a claim ABOUT the table -- how many cells reproduce, how wide the spread is,
   how many cells our encoder wins. Those are re-derived from the filed table text, parsed back out
   of the document, never from the numbers I had in hand while writing it.

Standing repo rule: a results document is signed off only after its numbers are parsed OUT of it and
recomputed. Prose counts and range claims count, not just tables.

`baselines/` is gitignored, so job 1 cannot run without the prediction archives. It exits 2, not 1,
in that case, to keep "cannot check" distinct from "wrong".

    conda run -n ebola-train python -m diagnostics.verify_paper_table [--mutate]
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOC = ROOT / "Reports" / "baseline_reproduction_table.md"
PRED = ROOT / "baselines" / "_preds"

ROW = re.compile(
    r"^\| (Cola-GNN|EpiGNN|HeatGNN|MTGNN|\(none\)) \| ([\w-]+) \| (\d+) \| (.+?) \| (.+?) \| (.+?) \| "
    r"(.+?) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \| (\d+)/(\d+)/(\d+)/(\d+) \| (.*?) \|$",
    re.M)
BASELINES = ("Cola-GNN", "EpiGNN", "HeatGNN")
HORIZONS = (3, 5, 10, 15)


def parse(text: str) -> list[dict]:
    rows = []
    for m in ROW.finditer(text):
        def pct(s):
            s = s.strip()
            return None if s in {"—", "--", ""} else float(s.rstrip("%"))

        def val(s):
            s = s.strip().split("±")[0].strip()
            return None if s in {"—", "--", "", "no run"} else float(s)

        rows.append(dict(model=m.group(1), dataset=m.group(2), h=int(m.group(3)),
                         pub=m.group(4).strip(), repro=val(m.group(5)), d_pub=pct(m.group(6)),
                         enc=val(m.group(7)), d_enc=pct(m.group(8)),
                         tr_a=val(m.group(9)), d_tr_a=pct(m.group(10)),
                         tr_z=val(m.group(11)), d_tr_z=pct(m.group(12)),
                         seeds=tuple(int(m.group(i)) for i in (13, 14, 15, 16)),
                         flag=m.group(17).strip()))
    return rows


def cells(rows, with_covid=False):
    """One entry per (dataset, horizon), carrying our three columns and the BEST baseline.

    The table repeats each cell once per baseline model, so any tally over cells has to collapse
    them first or it counts Japan h3 three times and dengue h3 once.
    """
    out = {}
    for r in rows:
        if r["dataset"] == "covid_us-states" and not with_covid:
            continue
        k = (r["dataset"], r["h"])
        c = out.setdefault(k, dict(enc=r["enc"], tr_a=r["tr_a"], tr_z=r["tr_z"],
                                   d_tr_a=r["d_tr_a"], d_tr_z=r["d_tr_z"], best=None))
        if r["model"] in BASELINES and r["repro"] is not None:
            c["best"] = r["repro"] if c["best"] is None else min(c["best"], r["repro"])
    return out


def check(text: str) -> list[str]:
    bad: list[str] = []
    rows = parse(text)
    if len(rows) != 60:
        return [f"expected 60 table rows, parsed {len(rows)}"]

    def by(model=None, dataset=None):
        return [r for r in rows
                if (model is None or r["model"] == model)
                and (dataset is None or r["dataset"] == dataset)]

    # --- Cola-GNN: 12 comparable cells, stated range.
    d = [r["d_pub"] for r in by("Cola-GNN")]
    assert all(x is not None for x in d), "a Cola-GNN cell lost its published number"
    m = re.search(r"All 12 comparable cells land between \+([\d.]+) and \+([\d.]+) percent of the\s*\n?"
                  r"published figure", text)
    assert m, "Cola-GNN range sentence not found"
    if (len(d), round(min(d), 1), round(max(d), 1)) != (12, float(m.group(1)), float(m.group(2))):
        bad.append(f"Cola-GNN: doc says 12 cells in [{m.group(1)}, {m.group(2)}], "
                   f"table has {len(d)} in [{min(d):.1f}, {max(d):.1f}]")
    if not all(x > 0 for x in d):
        bad.append("Cola-GNN: doc says every cell is worse than published; table disagrees")

    # --- EpiGNN: 12 comparable influenza cells (dengue has no published number), stated range.
    d = [r["d_pub"] for r in by("EpiGNN") if r["d_pub"] is not None]
    m = re.search(r"All 12 comparable cells land between \+([\d.]+) and \+([\d.]+) percent\.", text)
    assert m, "EpiGNN range sentence not found"
    if (len(d), round(min(d), 1), round(max(d), 1)) != (12, float(m.group(1)), float(m.group(2))):
        bad.append(f"EpiGNN: doc says 12 cells in [{m.group(1)}, {m.group(2)}], "
                   f"table has {len(d)} in [{min(d):.1f}, {max(d):.1f}]")

    jp = [r["d_pub"] for r in sorted(by("EpiGNN", "influenza_japan"), key=lambda r: (3, 5, 10, 15).index(r["h"]))]
    m = re.search(r"Japan is the loose one at \+([\d.]+), \+([\d.]+), \+([\d.]+) and \+([\d.]+) percent", text)
    assert m, "EpiGNN Japan sentence not found"
    if [round(x, 1) for x in jp] != [float(g) for g in m.groups()]:
        bad.append(f"EpiGNN Japan: doc says {m.groups()}, table says {jp}")

    m = re.search(r"seed standard deviation of ([\d.]+) on a mean of ([\d.]+)", text)
    assert m, "EpiGNN Japan h5 dispersion sentence not found"
    cell = re.search(r"^\| EpiGNN \| influenza_japan \| 5 \| \d+ \| ([\d.]+) ± ([\d.]+) \|", text, re.M)
    assert cell, "EpiGNN japan h5 row not found"
    if (m.group(2), m.group(1)) != (cell.group(1), cell.group(2)):
        bad.append(f"EpiGNN japan h5: doc says {m.group(2)} ± {m.group(1)}, "
                   f"row says {cell.group(1)} ± {cell.group(2)}")

    # --- HeatGNN: exactly three comparable cells.
    d = [r["d_pub"] for r in by("HeatGNN") if r["d_pub"] is not None]
    m = re.search(r"exactly three comparable cells out of twelve: (-?[\d.]+), \+([\d.]+) and \+([\d.]+)\s*\n?percent", text)
    assert m, "HeatGNN comparable-cell sentence not found"
    want = sorted([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    if len(by("HeatGNN")) != 12 or sorted(round(x, 1) for x in d) != want:
        bad.append(f"HeatGNN: doc says 3 of 12 at {want}, table has {len(d)} of "
                   f"{len(by('HeatGNN'))} at {sorted(d)}")

    # --- MTGNN: no published number anywhere.
    mt = by("MTGNN")
    m = re.search(r"column is empty for all (\d+) cells", text)
    assert m, "MTGNN empty-column sentence not found"
    if len(mt) != int(m.group(1)) or any(r["d_pub"] is not None for r in mt):
        bad.append(f"MTGNN: doc says {m.group(1)} cells with no published number, "
                   f"table has {len(mt)} rows, {sum(r['d_pub'] is None for r in mt)} empty")

    # --- Encoder win/loss tally over the non-MTGNN cells. The COVID rows carry no baseline, so
    #     they have no "encoder vs reproduced" number and must not enter this tally either.
    usable = [r for r in rows if r["model"] not in ("MTGNN", "(none)")]
    gap = [f"{r['model']} {r['dataset']} h{r['h']}" for r in usable if r["d_enc"] is None]
    if gap:
        bad.append(f"rows with a baseline but no encoder-vs-reproduced number: {gap}")
        usable = [r for r in usable if r["d_enc"] is not None]
    wins = sum(1 for r in usable if r["d_enc"] < 0)
    losses = sum(1 for r in usable if r["d_enc"] > 0)
    m = re.search(r"over the (\d+) cells that exclude MTGNN, the encoder is better in\s*\n?"
                  r"\*\*(\d+)\*\* and worse in \*\*(\d+)\*\*", text)
    assert m, "encoder win/loss sentence not found"
    if (len(usable), wins, losses) != tuple(int(g) for g in m.groups()):
        bad.append(f"encoder tally: doc says {m.groups()}, table gives "
                   f"({len(usable)}, {wins}, {losses})")

    # --- Per-panel claims.
    dn = [r["d_enc"] for r in by("EpiGNN", "dengue")]
    m = re.search(r"all four horizons against EpiGNN, by ([\d.]+) to ([\d.]+) percent on matched nodes", text)
    assert m, "dengue margin sentence not found"
    if not all(x < 0 for x in dn) or sorted(round(abs(x), 1) for x in dn)[::len(dn) - 1] != \
            [float(m.group(1)), float(m.group(2))]:
        bad.append(f"dengue margins: doc says {m.group(1)} to {m.group(2)}, table says {dn}")

    jp = [r for r in usable if r["dataset"] == "influenza_japan" and r["model"] != "Cola-GNN"]
    m = re.search(r"\*\*We win on Japan\*\*, (\d+) of (\d+) cells against EpiGNN and HeatGNN", text)
    assert m, "Japan win sentence not found"
    if (sum(1 for r in jp if r["d_enc"] < 0), len(jp)) != (int(m.group(1)), int(m.group(2))):
        bad.append(f"Japan wins: doc says {m.group(1)}/{m.group(2)}, table gives "
                   f"{sum(1 for r in jp if r['d_enc'] < 0)}/{len(jp)}")

    st = [r["d_enc"] for r in by("EpiGNN", "influenza_us-states")]
    m = re.search(r"Against EpiGNN we are worse at all four horizons, \+([\d.]+) to \+([\d.]+) percent", text)
    assert m, "US-States loss sentence not found"
    if not all(x > 0 for x in st) or [round(min(st), 1), round(max(st), 1)] != \
            [float(m.group(1)), float(m.group(2))]:
        bad.append(f"US-States vs EpiGNN: doc says +{m.group(1)} to +{m.group(2)}, table says {st}")

    # --- Every dengue row must carry the subsample flag; nothing else may.
    for r in rows:
        has = "2,392-node subsample" in r["flag"]
        if (r["dataset"] == "dengue") != has:
            bad.append(f"subsample flag wrong on {r['model']} {r['dataset']} h{r['h']}")

    # --- The dengue correction table must agree with the main table's encoder column.
    for m in re.finditer(r"^\| (3|5|10|15) \| ([\d.]+) \| ([\d.]+) \| \+[\d.]+% \| "
                         r"-[\d.]+% \| \*\*(-[\d.]+)%\*\* \|$", text, re.M):
        h, matched, margin = int(m.group(1)), float(m.group(3)), float(m.group(4))
        row = [r for r in by("EpiGNN", "dengue") if r["h"] == h][0]
        cell = re.search(rf"^\| EpiGNN \| dengue \| {h} \| — \| .+? \| — \| ([\d.]+) ± ", text, re.M)
        if not cell or float(cell.group(1)) != matched:
            bad.append(f"dengue correction h{h}: matched RMSE {matched} not in the main table")
        if round(row["d_enc"], 1) != margin:
            bad.append(f"dengue correction h{h}: margin {margin} != main table {row['d_enc']}")

    bad += check_transfer(text, rows)
    return bad


def check_transfer(text: str, rows: list[dict]) -> list[str]:
    """Columns 4 and 5, and every prose tally that talks about them.

    Same discipline as the rest of the file: the counts are re-derived from the filed table text,
    not from the numbers I had while writing the section.
    """
    bad: list[str] = []
    c = cells(rows)
    cov = cells(rows, with_covid=True)
    cov = {k: v for k, v in cov.items() if k[0] == "covid_us-states"}
    if len(c) != 16 or len(cov) != 4:
        return [f"expected 16 baseline cells and 4 COVID cells, got {len(c)} and {len(cov)}"]

    # --- COVID rows must state the absence rather than leave a gap.
    for r in rows:
        if r["dataset"] == "covid_us-states":
            if r["model"] != "(none)" or r["repro"] is not None or r["seeds"] != (0, 5, 5, 5):
                bad.append(f"COVID h{r['h']}: expected a (none) row with no baseline run, "
                           f"got {r['model']} repro={r['repro']} seeds={r['seeds']}")

    # --- Transfer against our own single-disease ceiling, 16 cells.
    a = [v["d_tr_a"] for v in c.values()]
    z = [v["d_tr_z"] for v in c.values()]
    m = re.search(r"Few-shot transfer is better in \*\*(\d+)\*\* of the 16 and worse\s*\n?"
                  r"in \*\*(\d+)\*\*, spanning (-[\d.]+) to \+([\d.]+) percent\. Zero-shot is worse in "
                  r"\*\*(\d+)\*\* of 16, spanning \+([\d.]+) to\s*\n?\+([\d.]+) percent", text)
    assert m, "transfer vs single-disease sentence not found"
    got = (sum(1 for x in a if x < 0), sum(1 for x in a if x > 0), round(min(a), 1), round(max(a), 1),
           sum(1 for x in z if x > 0), round(min(z), 1), round(max(z), 1))
    want = (int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)),
            int(m.group(5)), float(m.group(6)), float(m.group(7)))
    if got != want:
        bad.append(f"transfer vs single: doc says {want}, table gives {got}")

    ca, cz = [v["d_tr_a"] for v in cov.values()], [v["d_tr_z"] for v in cov.values()]
    m = re.search(r"few-shot is better in (\d+) of 4 and zero-shot worse in (\d+) of 4,\s*\n?"
                  r"by as much as \+([\d.]+) percent", text)
    assert m, "COVID transfer sentence not found"
    got = (sum(1 for x in ca if x < 0), sum(1 for x in cz if x > 0), round(max(cz), 1))
    if got != (int(m.group(1)), int(m.group(2)), float(m.group(3))):
        bad.append(f"COVID transfer: doc says {m.groups()}, table gives {got}")

    # --- The horizon gradient, h3 and h15 per panel.
    m = re.search(r"Few-shot goes (-[\d.]+) to \+([\d.]+) percent on\s*\n?Japan, (-[\d.]+) to "
                  r"\+([\d.]+) on US-Regions and (-[\d.]+) to \+([\d.]+) on US-States", text)
    assert m, "horizon gradient sentence not found"
    for i, ds in enumerate(("influenza_japan", "influenza_us-regions", "influenza_us-states")):
        want = (float(m.group(2 * i + 1)), float(m.group(2 * i + 2)))
        got = (round(c[(ds, 3)]["d_tr_a"], 1), round(c[(ds, 15)]["d_tr_a"], 1))
        if got != want:
            bad.append(f"{ds} h3/h15 few-shot cost: doc says {want}, table gives {got}")
    m = re.search(r"Dengue\s*\n?runs the other way, \+([\d.]+) down to \+([\d.]+)", text)
    assert m, "dengue reversal sentence not found"
    got = (round(c[("dengue", 3)]["d_tr_a"], 1), round(c[("dengue", 15)]["d_tr_a"], 1))
    if got != (float(m.group(1)), float(m.group(2))):
        bad.append(f"dengue h3/h15 few-shot cost: doc says {m.groups()}, table gives {got}")

    # --- Transfer against the BEST baseline in each cell. The minimum, because a steel-manned
    #     opponent is the only honest one here.
    m = re.search(r"best baseline in the cell in \*\*(\d+)\*\* of 16, zero-shot in \*\*(\d+)\*\* of 16\.\s*\n?"
                  r"(\w+) of those five few-shot cells\s*\n?are dengue and all three zero-shot cells "
                  r"are dengue", text)
    assert m, "best-baseline tally sentence not found"
    wa = [k for k, v in c.items() if v["tr_a"] < v["best"]]
    wz = [k for k, v in c.items() if v["tr_z"] < v["best"]]
    words = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    if (len(wa), len(wz)) != (int(m.group(1)), int(m.group(2))):
        bad.append(f"best-baseline tally: doc says {m.group(1)}/{m.group(2)}, "
                   f"table gives {len(wa)}/{len(wz)}")
    if sum(1 for k in wa if k[0] == "dengue") != words[m.group(3)] or \
            any(k[0] != "dengue" for k in wz):
        bad.append(f"best-baseline dengue split: doc says {m.group(3)} of the few-shot wins are "
                   f"dengue and all zero-shot wins are, table gives {wa} and {wz}")

    # --- The naive floors, recomputed from results/naive/ rather than read off any table.
    m = re.search(r"a per-node train-mean forecast scores\s*\n?\*\*([\d.]+) / ([\d.]+) / ([\d.]+) / "
                  r"([\d.]+)\*\* at h3/h5/h10/h15", text)
    assert m, "dengue train-mean floor sentence not found"
    floor = naive_pooled("dengue", "train_mean")
    if floor and [round(x, 1) for x in floor] != [float(g) for g in m.groups()]:
        bad.append(f"dengue train_mean floor: doc says {m.groups()}, disk gives "
                   f"{[round(x, 1) for x in floor]}")
    m = re.search(r"Persistence pooled on the same nodes is\s*\n?([\d.]+) / ([\d.]+) / ([\d.]+) / "
                  r"([\d.]+)\.", text)
    assert m, "dengue persistence floor sentence not found"
    pers = naive_pooled("dengue", "persistence")
    if pers and [round(x, 1) for x in pers] != [float(g) for g in m.groups()]:
        bad.append(f"dengue persistence floor: doc says {m.groups()}, disk gives "
                   f"{[round(x, 1) for x in pers]}")

    # --- The two quartets quoted in the prose must be the table's own numbers.
    m = re.search(r"Our few-shot transfer scores ([\d.]+) / ([\d.]+) /\s*\n?([\d.]+) / ([\d.]+), so at "
                  r"h10 and h15", text)
    assert m, "dengue few-shot quartet sentence not found"
    got = [round(c[("dengue", h)]["tr_a"], 1) for h in HORIZONS]
    if got != [float(g) for g in m.groups()]:
        bad.append(f"dengue few-shot quartet: doc says {m.groups()}, table gives {got}")
    m = re.search(r"and so is EpiGNN at ([\d.]+) and\s*\n?([\d.]+)\.", text)
    assert m, "dengue EpiGNN long-horizon sentence not found"
    epi = [r["repro"] for r in rows
           if (r["model"], r["dataset"]) == ("EpiGNN", "dengue") and r["h"] in (10, 15)]
    if [round(x, 1) for x in epi] != [float(g) for g in m.groups()]:
        bad.append(f"dengue EpiGNN h10/h15: doc says {m.groups()}, table gives {epi}")

    # --- The claim that hangs off those floors: at h10 and h15 both we and EpiGNN are WORSE than a
    #     flat forecast, and no dengue horizon clears BOTH floors. If either flips, the paragraph is
    #     wrong in the other direction and has to be rewritten, not softened.
    if floor and pers:
        for i, h in enumerate(HORIZONS):
            tr = c[("dengue", h)]["tr_a"]
            if (tr > floor[i]) != (h in (10, 15)):
                bad.append(f"dengue h{h}: few-shot transfer {tr} vs flat floor {floor[i]:.1f} "
                           f"no longer matches the paragraph")
            if (tr > pers[i]) != (h in (3, 5)):
                bad.append(f"dengue h{h}: few-shot transfer {tr} vs persistence {pers[i]:.1f} "
                           f"no longer matches the paragraph")
            if tr < floor[i] and tr < pers[i]:
                bad.append(f"dengue h{h}: few-shot transfer {tr} now clears BOTH floors "
                           f"({floor[i]:.1f}, {pers[i]:.1f}); the section says none does")

    # --- The transfer columns themselves, recomputed from the per-NODE archives. The generator
    #     builds them from the per-ORIGIN sufficient statistics, so this is a genuinely different
    #     route to the same quantity, not the generator marking its own homework.
    for col, sub, prefix in (("enc", "single", "encoder"),
                             ("tr_a", "lodo", "encoder_ldo3"),
                             ("tr_z", "lodo", "encoder_ldo3_zeroshot")):
        disk = pooled_from_pernode(sub, prefix)
        if not disk:
            print(f"note: no {prefix} archives on disk, column {col} not recomputed")
            continue
        for k, v in cells(rows, with_covid=True).items():
            if k not in disk:
                bad.append(f"{col} {k}: no {prefix} archive on disk")
                continue
            want = float(np.mean(disk[k]))
            if abs(v[col] - want) > 0.05:
                bad.append(f"{col} {k[0]} h{k[1]}: doc says {v[col]}, "
                           f"per-node archives give {want:.1f}")
            if len(disk[k]) != 5:
                bad.append(f"{col} {k[0]} h{k[1]}: {len(disk[k])} seeds on disk, expected 5")

    return bad


def pooled_from_pernode(subdir: str, prefix: str) -> dict:
    """{(dataset, h): [rmse per seed]} pooled node-side: sqrt(sum(n_i * rmse_i^2) / sum(n_i)).

    Dengue is restricted to the exported node set, the same way the table is, because the two sides
    would otherwise be pooling different populations.
    """
    from diagnostics.paper_compare import subsample_rows

    out: dict = {}
    for p in sorted((ROOT / "results" / subdir).glob(f"{prefix}__*__pernode.npz")):
        ds = p.name[: -len("__pernode.npz")].split("__")[-2]
        keep = subsample_rows(ds)
        z = np.load(p, allow_pickle=True)
        for h in HORIZONS:
            idx = z[f"h{h}__node_idx"].astype(np.int64)
            nc = z[f"h{h}__n_cells"].astype(np.float64)
            r = z[f"h{h}__rmse"].astype(np.float64)
            m = np.isfinite(r) & (nc > 0)
            if keep is not None:
                m &= np.isin(idx, keep)
            if m.any():
                out.setdefault((ds, h), []).append(
                    float(np.sqrt((nc[m] * r[m] ** 2).sum() / nc[m].sum())))
    return out


def naive_pooled(dataset: str, floor: str) -> list[float] | None:
    """The naive floor in the SAME pooled metric and on the same node set as the table."""
    from diagnostics.paper_compare import subsample_rows

    p = ROOT / "results" / "naive" / f"naive__{dataset}__{floor}__pernode.npz"
    if not p.exists():
        print(f"note: {p.name} absent, floor not recomputed")
        return None
    keep = subsample_rows(dataset)
    z = np.load(p, allow_pickle=True)
    out = []
    for h in HORIZONS:
        idx = z[f"h{h}__node_idx"].astype(np.int64)
        nc = z[f"h{h}__n_cells"].astype(np.float64)
        r = z[f"h{h}__rmse"].astype(np.float64)
        m = np.isfinite(r) & (nc > 0)
        if keep is not None:
            m &= np.isin(idx, keep)
        out.append(float(np.sqrt((nc[m] * r[m] ** 2).sum() / nc[m].sum())))
    return out


def table_is_current(text: str) -> list[str]:
    """Re-run the generator and confirm the filed rows are still what it emits."""
    from diagnostics import paper_compare

    buf = io.StringIO()
    real, sys.stdout = sys.stdout, buf
    try:
        paper_compare.main(as_md=True)
    finally:
        sys.stdout = real
    fresh = [ln for ln in buf.getvalue().splitlines() if ROW.match(ln)]
    filed = [ln for ln in text.splitlines() if ROW.match(ln)]
    if fresh == filed:
        return []
    out = [f"filed table is stale: {len(filed)} rows filed, {len(fresh)} regenerated"]
    for a, b in zip(filed, fresh):
        if a != b:
            out.append(f"  filed: {a}")
            out.append(f"  fresh: {b}")
    return out[:9]


MUTATIONS = [
    ("Cola-GNN range widened", lambda t: t.replace("between +1.7 and +15.0 percent", "between +1.7 and +25.0 percent")),
    ("EpiGNN Japan figure", lambda t: t.replace("at +18.7, +23.5, +18.1 and +8.7 percent", "at +18.7, +13.5, +18.1 and +8.7 percent")),
    ("EpiGNN japan h5 dispersion", lambda t: t.replace("seed standard deviation of 322.1", "seed standard deviation of 22.1")),
    ("HeatGNN comparable-cell count", lambda t: t.replace("exactly three comparable cells out of twelve", "exactly five comparable cells out of twelve")),
    ("MTGNN cell count", lambda t: t.replace("column is empty for all 16 cells", "column is empty for all 20 cells")),
    ("encoder win tally", lambda t: t.replace("better in\n**28** and worse in **12**", "better in\n**32** and worse in **8**")),
    ("dengue margin range", lambda t: t.replace("by 6.0 to 28.9 percent on matched nodes", "by 16.0 to 28.9 percent on matched nodes")),
    ("Japan win count", lambda t: t.replace("**We win on Japan**, 7 of 8 cells", "**We win on Japan**, 8 of 8 cells")),
    ("US-States losses softened", lambda t: t.replace("worse at all four horizons, +2.8 to +13.1 percent", "worse at all four horizons, +2.8 to +5.1 percent")),
    ("subsample flag dropped from a dengue row", lambda t: t.replace(
        "| EpiGNN | dengue | 15 | — | 504.7 ± 11.5 | — | 474.3 ± 1.9 | -6.0% | 482.5 ± 0.2 | +1.7% "
        "| 480.9 ± 0.2 | +1.4% | 5/5/5/5 | both sides on the 2,392-node subsample |",
        "| EpiGNN | dengue | 15 | — | 504.7 ± 11.5 | — | 474.3 ± 1.9 | -6.0% | 482.5 ± 0.2 | +1.7% "
        "| 480.9 ± 0.2 | +1.4% | 5/5/5/5 |  |")),
    ("dengue correction table desynced", lambda t: t.replace("| 15 | 315.6 | 474.3 |", "| 15 | 315.6 | 464.3 |")),
    # --- columns 4 and 5.
    ("transfer-vs-single tally inflated", lambda t: t.replace(
        "better in **4** of the 16 and worse\nin **12**", "better in **9** of the 16 and worse\nin **7**")),
    ("zero-shot range softened", lambda t: t.replace(
        "spanning +1.4 to\n+53.3 percent", "spanning +1.4 to\n+13.3 percent")),
    ("COVID transfer blow-up softened", lambda t: t.replace(
        "by as much as +204.6 percent", "by as much as +24.6 percent")),
    ("horizon gradient flattened", lambda t: t.replace(
        "Few-shot goes -5.9 to +38.7 percent on\nJapan", "Few-shot goes -5.9 to +8.7 percent on\nJapan")),
    ("dengue reversal misstated", lambda t: t.replace(
        "runs the other way, +40.4 down to +1.7", "runs the other way, +10.4 down to +1.7")),
    ("best-baseline win count inflated", lambda t: t.replace(
        "best baseline in the cell in **5** of 16, zero-shot in **3** of 16",
        "best baseline in the cell in **9** of 16, zero-shot in **7** of 16")),
    ("dengue caveat dropped from the best-baseline wins", lambda t: t.replace(
        "Four of those five few-shot cells\nare dengue", "One of those five few-shot cells\nare dengue")),
    ("train-mean floor moved above us", lambda t: t.replace(
        "**484.5 / 483.4 / 480.2 / 477.0**", "**484.5 / 483.4 / 490.2 / 497.0**")),
    ("persistence floor misquoted", lambda t: t.replace(
        "234.7 / 350.1 / 560.7 / 645.9", "234.7 / 350.1 / 460.7 / 545.9")),
    ("dengue few-shot quartet desynced from the table", lambda t: t.replace(
        "scores 435.1 / 467.4 /\n   482.3 / 482.5", "scores 435.1 / 467.4 /\n   442.3 / 442.5")),
    ("EpiGNN dengue long-horizon figures desynced", lambda t: t.replace(
        "EpiGNN at 500.1 and\n   504.7.", "EpiGNN at 400.1 and\n   404.7.")),
    # a transfer NUMBER in the table itself, which only the per-node recomputation can catch
    ("transfer RMSE altered in the table", lambda t: t.replace("1980.3 ± 45.8", "1880.3 ± 45.8")),
    ("zero-shot RMSE altered in the table", lambda t: t.replace("2148.4 ± 5.9", "2048.4 ± 5.9")),
    ("COVID absence rewritten as a normal row", lambda t: t.replace(
        "| (none) | covid_us-states | 10 | — | no run | — | 19091.1 ± 3014.9 | — |",
        "| EpiGNN | covid_us-states | 10 | — | 1.0 ± 0.0 | — | 19091.1 ± 3014.9 | — |")),
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
    if PRED.exists():
        bad += table_is_current(text)
    else:
        print("note: baselines/_preds absent (gitignored), table freshness not checked")
    for b in bad:
        print("MISMATCH:", b)
    print(f"{DOC.name}: {'verified' if not bad else f'{len(bad)} mismatches'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

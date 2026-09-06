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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOC = ROOT / "Reports" / "baseline_reproduction_table.md"
PRED = ROOT / "baselines" / "_preds"

ROW = re.compile(
    r"^\| (Cola-GNN|EpiGNN|HeatGNN|MTGNN) \| ([\w-]+) \| (\d+) \| (.+?) \| (.+?) \| (.+?) \| "
    r"(.+?) \| (.+?) \| (\d+)/(\d+) \| (.*?) \|$", re.M)


def parse(text: str) -> list[dict]:
    rows = []
    for m in ROW.finditer(text):
        def pct(s):
            s = s.strip()
            return None if s in {"—", "--", ""} else float(s.rstrip("%"))
        rows.append(dict(model=m.group(1), dataset=m.group(2), h=int(m.group(3)),
                         pub=m.group(4).strip(), d_pub=pct(m.group(6)),
                         d_enc=pct(m.group(8)), flag=m.group(11).strip()))
    return rows


def check(text: str) -> list[str]:
    bad: list[str] = []
    rows = parse(text)
    if len(rows) != 56:
        return [f"expected 56 table rows, parsed {len(rows)}"]

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

    # --- Encoder win/loss tally over the non-MTGNN cells.
    usable = [r for r in rows if r["model"] != "MTGNN"]
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

    return bad


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
        "| EpiGNN | dengue | 15 | — | 504.7 ± 11.5 | — | 474.3 ± 1.9 | -6.0% | 5/5 | both sides on the 2,392-node subsample |",
        "| EpiGNN | dengue | 15 | — | 504.7 ± 11.5 | — | 474.3 ± 1.9 | -6.0% | 5/5 |  |")),
    ("dengue correction table desynced", lambda t: t.replace("| 15 | 315.6 | 474.3 |", "| 15 | 315.6 | 464.3 |")),
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

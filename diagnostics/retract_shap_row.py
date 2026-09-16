"""D2: retract the SHAP claim from the comparison table the client already holds.

`Reports/Phase1/RelatedWork_CompetitiveAnalysis_Benchmark.docx` contains a comparison table whose
"Ours" row claims "yes - SHAP (glob.+loc.)". We have never had SHAP. We have integrated gradients
with an occlusion cross-check, which the client accepted on 2026-09-16 (decision D1, evidence in
Reports/G5_Method_Decision_Brief.md).

WHY A SCRIPT AND NOT A HAND EDIT. The original is a document already in the client's hands, so the
correction has to be exact, reviewable and repeatable. A script states precisely which cell moved
and refuses to run if the document is not the one it expects. A hand edit leaves no record of what
was changed and cannot be checked later.

THE ORIGINAL IS NEVER MODIFIED. The corrected copy is written alongside it with a dated filename, so
both versions exist and it is obvious which is which.

  conda run -n ebola-train python diagnostics/retract_shap_row.py          # writes the corrected copy
  conda run -n ebola-train python diagnostics/retract_shap_row.py --check  # verify only, writes nothing
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from docx import Document

SRC = Path("Reports/Phase1/RelatedWork_CompetitiveAnalysis_Benchmark.docx")
OUT = SRC.with_name(SRC.stem + "_corrected_2026-09-16.docx")
TABLE, ROW, COL = 0, 11, 7
OLD = "yes — SHAP (glob.+loc.)"
NEW = "yes — integrated gradients + occlusion (glob.+loc.)"


def locate(doc):
    """The cell to change, found by CONTENT rather than trusted by index.

    The indices above are what we measured, but a document that has been re-saved or edited by
    anyone else can shift them. So the indices are checked against the row label and the cell text,
    and anything unexpected stops the run rather than rewriting the wrong cell.
    """
    if len(doc.tables) <= TABLE:
        sys.exit(f"expected at least {TABLE + 1} tables, found {len(doc.tables)}")
    t = doc.tables[TABLE]
    if len(t.rows) <= ROW:
        sys.exit(f"table {TABLE} has {len(t.rows)} rows, expected more than {ROW}")
    row = t.rows[ROW]
    label = row.cells[0].text.strip()
    if label.lower() != "ours":
        sys.exit(f"row {ROW} of table {TABLE} is labelled {label!r}, not 'Ours'. Refusing to edit.")
    cell = row.cells[COL]
    if "shap" not in cell.text.lower():
        sys.exit(f"the cell at row {ROW} col {COL} reads {cell.text.strip()!r} and does not mention "
                 f"SHAP. Either it is already corrected or the layout moved. Refusing to edit.")
    return cell


def set_text(cell, text):
    """Replace a cell's text while keeping its formatting.

    python-docx has no 'set cell text and keep the style' call. Writing cell.text= would drop the
    run formatting, so the first run is rewritten and the rest emptied: the first run carries the
    font the rest of the table uses.
    """
    paras = [p for p in cell.paragraphs if p.runs]
    if not paras:
        cell.text = text
        return
    runs = paras[0].runs
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""
    for p in paras[1:]:
        for r in p.runs:
            r.text = ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report the cell, change nothing")
    a = ap.parse_args()
    if not SRC.exists():
        sys.exit(f"missing {SRC}")

    doc = Document(str(SRC))
    cell = locate(doc)
    print(f"source : {SRC}")
    print(f"cell   : table {TABLE}, row {ROW} ('{doc.tables[TABLE].rows[ROW].cells[0].text.strip()}'), "
          f"column {COL}")
    print(f"before : {cell.text.strip()!r}")
    print(f"after  : {NEW!r}")
    if a.check:
        print("\n--check: nothing written")
        return 0

    set_text(cell, NEW)
    doc.save(str(OUT))

    # Read the saved file back and confirm the change landed and nothing else moved. A correction
    # that is not verified is just an assertion with extra steps.
    back = Document(str(OUT))
    got = back.tables[TABLE].rows[ROW].cells[COL].text.strip()
    if got != NEW:
        sys.exit(f"wrote {OUT} but the cell reads {got!r}. Not correcting anything on that.")
    src_doc = Document(str(SRC))
    moved = [(ti, ri, ci)
             for ti, (t0, t1) in enumerate(zip(src_doc.tables, back.tables))
             for ri, (r0, r1) in enumerate(zip(t0.rows, t1.rows))
             for ci, (c0, c1) in enumerate(zip(r0.cells, r1.cells))
             if c0.text != c1.text]
    if moved != [(TABLE, ROW, COL)]:
        sys.exit(f"cells that changed: {moved}. Exactly one was expected.")
    print(f"\nok  wrote {OUT}")
    print(f"ok  exactly one cell differs from the original: table {TABLE} row {ROW} col {COL}")
    print(f"ok  {SRC.name} is untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

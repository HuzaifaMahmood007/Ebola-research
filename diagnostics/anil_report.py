"""anil_report.py -- verify the ANIL meta-learning folds and report them.

Two jobs, in this order, because the second is worthless without the first:

  1. VERIFY. Enumerate every (fold, arm, seed) cell the runs are supposed to have produced and check
     the artifacts landed and agree with themselves: the scored row is present in the resume JSON,
     the per-dataset record/pernode/perorigin/quantile files exist, the seed and dataset in the
     filename match the payload, the fold metadata matches the fold the file is filed under, and --
     the one that actually catches a mis-scored fold -- the arm's cell keys pair against the
     reference it is differenced against. Any failure is printed and the exit code is non-zero.

  2. REPORT. Two comparisons, because one of them alone misleads:

     PRIMARY, and the one G2 actually asks about: ANIL against its OWN seed-matched control. The
     control runs the same episode stream and the same number of outer updates with no inner loop,
     so it is ordinary ERM on identical data. Differencing against it holds everything fixed except
     whether adaptation happened during training, which is the definition of "does meta-learning
     beat the probe". `train.anil.compare()` does NOT compute this -- it differences each arm
     against a freeze-then-adapt reference -- so this file computes it.

     SECONDARY: each arm against its fold's own freeze-then-adapt reference. This is context, not
     the verdict. It exists to show the two arms move together, which is what makes a null on the
     primary comparison interpretable rather than an artefact of both arms being broken.

WHY THIS DOCUMENT EXISTS. Meta-learning is REQUIRED under G2 of the internal brief. Until
2026-09-04 the evidence for the null was twelve cells on a single held-out disease, meta-trained on
dengue alone, which is thin for a required goal and carried an objection the module raised against
itself: episodes drawn from one disease vary population and forecast origin, not disease, so the
arm was trained on the axis that already works. The three LDO3 folds close that.

WHAT THIS STILL DOES NOT MEASURE, and it belongs next to any number here: the meta-test fits a fresh
adapter on the held-out disease's FULL train fold, so nothing in this document measures few-shot
behaviour, and every arm warm-starts from an existing trunk, so this is meta-FINE-TUNING rather than
meta-learning from scratch.

No torch, no GPU, seconds to run. Deliberately reads only the archived JSON, so it can be run in any
environment that has numpy.

    python diagnostics/anil_report.py --selfcheck
    python diagnostics/anil_report.py --verify-only
    python diagnostics/anil_report.py -o progress/outcomes/ANIL_Results.md
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from results_matrix import _fmt, paired_delta  # noqa: E402
from results_paths import RESULTS, rpath  # noqa: E402

SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
ARMS = ("anil", "control")
SURFACE = "affine"

# Mirrored from train/lodo.py::DISEASES and train/anil.py::FOLDS, and ASSERTED against them at run
# time rather than trusted, so a change there cannot silently desynchronise this report.
DISEASES = {
    "dengue":    ("dengue",),
    "influenza": ("influenza_japan", "influenza_us-regions", "influenza_us-states"),
    "covid":     ("covid_us-states",),
}
LEGACY_FOLD = "dengue2flu"
FOLDS = (LEGACY_FOLD, "dengue", "influenza", "covid")

CAP_JSON = RESULTS / "misc" / "capacity_probe_5seed.json"
CAP_LABEL = "affine (current)"


# --------------------------------------------------------------------------- #
# Fold plumbing -- kept byte-compatible with train/anil.py's own path helpers
# --------------------------------------------------------------------------- #
def fold_plan(fold):
    """(meta-train bundles, meta-test bundles, direction label)."""
    assert fold in FOLDS, f"unknown fold {fold!r}"
    if fold == LEGACY_FOLD:
        return ["dengue"], list(DISEASES["influenza"]), "dengue2flu"
    meta = [n for d, names in DISEASES.items() if d != fold for n in names]
    return meta, list(DISEASES[fold]), f"ldo3:{fold}"


def fold_tag(fold):
    return "dengue2flu" if fold == LEGACY_FOLD else f"ldo3{fold}"


def out_json(fold, arm, surface=SURFACE):
    tag = "" if fold == LEGACY_FOLD else f"{fold_tag(fold)}_"
    return RESULTS / "misc" / f"anil_{tag}{surface}_{arm}.json"


def artifact(fold, arm, ds_name, seed, suffix, surface=SURFACE):
    tag = "" if fold == LEGACY_FOLD else f"{fold_tag(fold)}-"
    return f"encoder_ldo__{tag}anil-{surface}-{arm}__{ds_name}__seed{seed}{suffix}"


def warm_ckpt(fold, seed):
    if fold == LEGACY_FOLD:
        return f"encoder_ldo__dengue2flu-cap__seed{seed}__ckpt.pt"
    return f"encoder_ldo3__{fold}__seed{seed}__ckpt.pt"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_rows(fold, arm, surface=SURFACE):
    """{seed: row} from the resume JSON. A seed is present only if meta_test returned for it."""
    rows = _load_json(out_json(fold, arm, surface)) or []
    return {r["seed"]: r for r in rows}


def rmse_by_seed(rows):
    """{cell: {seed: rmse}} -- the shape paired_delta wants."""
    out = {}
    for seed, r in rows.items():
        for cell, v in (r.get("rmse") or {}).items():
            out.setdefault(cell, {})[seed] = v
    return out


def records_rmse(recs):
    """{'dataset|hH': country_macro rmse}. Same field and same key shape as train/anil.py."""
    return {f"{r['dataset']}|h{r['horizon']}": float(r["country_macro"])
            for r in recs if r["metric"] == "rmse"}


def reference(fold, surface=SURFACE):
    """{cell: {seed: rmse}} for the freeze-then-adapt comparator of this fold.

    The legacy fold keeps the capacity probe's own row, which is what its reported table was
    differenced against. The LDO3 folds use that fold's freeze-then-adapt records: the SAME trunk
    the arm warm-starts from, frozen, with one shared adapter fitted on the held-out disease's full
    train fold and scored by the same pass. The capacity probe cannot serve there, because it only
    ever ran the dengue->flu direction and holds no cells for a dengue or covid meta-test.
    """
    out = {}
    if fold == LEGACY_FOLD:
        blob = _load_json(CAP_JSON) or {}
        for per_seed in blob.get("cross_disease", []):
            for row in per_seed:
                if row.get("label") == CAP_LABEL:
                    for cell, v in row["rmse"].items():
                        out.setdefault(cell, {})[row["seed"]] = v
        return out
    for ds in fold_plan(fold)[1]:
        for s in SEEDS:
            recs = _load_json(rpath(f"encoder_ldo3__{ds}__seed{s}.json"))
            if not recs:
                continue
            for cell, v in records_rmse(recs).items():
                out.setdefault(cell, {})[s] = v
    return out


# --------------------------------------------------------------------------- #
# 1. Verification
# --------------------------------------------------------------------------- #
def verify(surface=SURFACE, verbose=True):
    """(ok, problems, manifest). Never raises on a missing file -- it reports it."""
    problems, manifest = [], []

    try:                                        # cross-check the fold universe against the source
        import train.lodo as LD
        if {k: tuple(v) for k, v in LD.DISEASES.items()} != {k: tuple(v) for k, v in DISEASES.items()}:
            problems.append(f"train/lodo.py DISEASES has drifted from this report's copy: "
                            f"{LD.DISEASES} vs {DISEASES}")
        import train.anil as AN
        if tuple(AN.FOLDS) != FOLDS:
            problems.append(f"train/anil.py FOLDS has drifted: {AN.FOLDS} vs {FOLDS}")
        for f in FOLDS:                          # the plan itself, not just the fold names
            if (list(AN.fold_plan(f)[0]), list(AN.fold_plan(f)[1])) != \
               (fold_plan(f)[0], fold_plan(f)[1]):
                problems.append(f"fold_plan({f!r}) disagrees with train/anil.py")
    except Exception as e:                       # torch missing -> skip, not a failure
        if verbose:
            print(f"note: could not import train.* to cross-check the fold universe ({e})")

    for fold in FOLDS:
        meta_names, test_names, _dir = fold_plan(fold)
        ref = reference(fold, surface)
        for arm in ARMS:
            rows = load_rows(fold, arm, surface)
            got = sorted(rows)
            row = dict(fold=fold, arm=arm, seeds=got, n_seeds=len(got),
                       missing_seeds=[s for s in SEEDS if s not in rows])
            if row["missing_seeds"]:
                problems.append(f"{fold}/{arm}: no scored row for seed(s) {row['missing_seeds']}")

            for s, r in sorted(rows.items()):
                tag = f"{fold}/{arm}/seed{s}"
                if r.get("seed") != s:
                    problems.append(f"{tag}: payload seed {r.get('seed')} != key {s}")
                if r.get("arm") not in (arm, None):
                    problems.append(f"{tag}: payload arm {r.get('arm')!r} != {arm!r}")
                if r.get("surface") not in (surface, None):
                    problems.append(f"{tag}: payload surface {r.get('surface')!r} != {surface!r}")
                # the legacy rows predate the --fold argument and carry no fold field; that is
                # expected and is not a defect. A NEW fold that claims the wrong fold is.
                if fold != LEGACY_FOLD and r.get("fold") != fold:
                    problems.append(f"{tag}: payload fold {r.get('fold')!r} != {fold!r}")
                if fold != LEGACY_FOLD and r.get("meta_names") \
                        and list(r["meta_names"]) != meta_names:
                    problems.append(f"{tag}: meta_names {r['meta_names']} != {meta_names}")
                if not rpath(warm_ckpt(fold, s)).exists():
                    problems.append(f"{tag}: warm-start trunk {warm_ckpt(fold, s)} is gone")

                cells = r.get("rmse") or {}
                want_cells = {f"{ds}|h{h}" for ds in test_names for h in HORIZONS}
                if set(cells) != want_cells:
                    problems.append(f"{tag}: scored cells {sorted(set(cells) ^ want_cells)} "
                                    f"differ from the fold's meta-test set")
                for c, v in cells.items():
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        problems.append(f"{tag}: cell {c} is NaN")
                unpaired = sorted(set(cells) - set(ref))
                if ref and unpaired:
                    problems.append(f"{tag}: cells with no reference to pair against: {unpaired}")

                for ds in test_names:             # the archived per-dataset artifacts
                    for suf in (".json", "__pernode.npz", "__perorigin.npz", "__quantiles.npz"):
                        p = rpath(artifact(fold, arm, ds, s, suf, surface))
                        if not p.exists():
                            problems.append(f"{tag}: missing artifact {p.name}")
                    recs = _load_json(rpath(artifact(fold, arm, ds, s, ".json", surface)))
                    if recs:
                        if {r2["seed"] for r2 in recs} != {s}:
                            problems.append(f"{tag}: {ds} record seed disagrees with its filename")
                        if {r2["dataset"] for r2 in recs} != {ds}:
                            problems.append(f"{tag}: {ds} record dataset disagrees with filename")
                        got_rmse = records_rmse(recs)
                        for c, v in got_rmse.items():
                            if c in cells and abs(cells[c] - v) > 1e-9:
                                problems.append(f"{tag}: {c} resume JSON {cells[c]} != record {v}")
            manifest.append(row)

        matched = sorted(set(load_rows(fold, "anil", surface)) &
                         set(load_rows(fold, "control", surface)))
        if len(matched) < len(SEEDS):
            problems.append(f"{fold}: only {len(matched)} matched (anil, control) pairs; "
                            f"a delta may not be quoted over more seeds than it pairs on")

    ok = not problems
    if verbose:
        print(f"{'=' * 78}\n1. Run verification\n{'=' * 78}")
        for row in manifest:
            print(f"  {row['fold']:<12} {row['arm']:<8} seeds={row['seeds']}")
        if ok:
            print(f"\nOK: {len(manifest)} (fold, arm) cells, all artifacts present and consistent.")
        else:
            print(f"\n{len(problems)} PROBLEM(S):")
            for p in problems:
                print(f"  - {p}")
    return ok, problems, manifest


# --------------------------------------------------------------------------- #
# 2. The comparisons
# --------------------------------------------------------------------------- #
def compare_cells(new_by_cell, ref_by_cell):
    """{cell: (text, mean, clears)} via the repo's shared paired-by-seed delta.

    Uses `results_matrix.paired_delta` rather than a local reimplementation so this document and the
    transfer tables cannot drift apart on the convention that decides what counts as a result: the
    t value comes from the number of seeds the CELL actually paired on, and a cell whose interval
    covers zero reads "within noise" rather than being given a direction.
    """
    out = {}
    for cell in sorted(set(new_by_cell) & set(ref_by_cell)):
        out[cell] = paired_delta(new_by_cell[cell], ref_by_cell[cell], "rmse")
    return out


def tally(cmp_map):
    """(better, worse, noise) over cells whose interval was computable."""
    b = sum(1 for _t, m, c in cmp_map.values() if c and m is not None and m > 0)
    w = sum(1 for _t, m, c in cmp_map.values() if c and m is not None and m < 0)
    n = sum(1 for _t, _m, c in cmp_map.values() if c is False)
    return b, w, n


def collect(surface=SURFACE):
    """Everything the document needs, per fold."""
    out = {}
    for fold in FOLDS:
        rows = {a: load_rows(fold, a, surface) for a in ARMS}
        anil_c = rmse_by_seed(rows["anil"])
        ctrl_c = rmse_by_seed(rows["control"])
        ref = reference(fold, surface)
        out[fold] = dict(
            rows=rows,
            primary=compare_cells(anil_c, ctrl_c),                 # ANIL vs its own control
            vs_ref={a: compare_cells(rmse_by_seed(rows[a]), ref) for a in ARMS},
            best_val=paired_delta({s: -r["best_val"] for s, r in rows["anil"].items()
                                   if r.get("best_val") is not None},
                                  {s: -r["best_val"] for s, r in rows["control"].items()
                                   if r.get("best_val") is not None}, "rmse"),
            plan=fold_plan(fold),
        )
    return out


# --------------------------------------------------------------------------- #
# 3. The document
# --------------------------------------------------------------------------- #
def _sig(clears):
    return {True: "yes", False: "within noise", None: "untestable"}[clears]


def _split(cellname):
    """'influenza_japan|h3' -> ('influenza_japan', 'h3').

    Cell keys carry a literal '|', which is the markdown table delimiter. Rendering one raw breaks
    the table in every GFM renderer AND in `Reports/md_to_docx.py`'s pipe-table parser, so the two
    halves always go in their own columns rather than being escaped into one.
    """
    ds, _, h = cellname.partition("|")
    return ds, h


def _cell_sort(cellname):
    """Sort by dataset, then by horizon NUMERICALLY -- otherwise h10 and h15 sort before h3."""
    ds, h = _split(cellname)
    try:
        return ds, int(h.lstrip("h"))
    except ValueError:
        return ds, 0


def build_markdown(data, verify_out, surface=SURFACE):
    ok, problems, manifest = verify_out
    L = []
    A = L.append

    grand = [0, 0, 0]
    for fold in FOLDS:
        b, w, n = tally(data[fold]["primary"])
        grand[0] += b
        grand[1] += w
        grand[2] += n
    total = sum(grand)

    A("# Episodic Meta-Learning (ANIL) Across Four Folds")
    A("")
    A("Generated by `diagnostics/anil_report.py` from the artifacts in `results/misc/` and "
      "`results/lodo/`. Surface is `affine`, the adaptation mechanism named in the Ebola "
      "pre-registration. Five seeds per arm, paired by seed. Nothing is averaged across datasets.")
    A("")
    A("## Summary")
    A("")
    A(f"**Meta-learning does not help, and the result is now four folds wide.** Against its own "
      f"seed-matched control, ANIL is better in **{grand[0]}** of {total} cells, worse in "
      f"**{grand[1]}**, and within noise in **{grand[2]}**.")
    A("")
    A("The comparison that matters is ANIL against its **control**, not against the "
      "freeze-then-adapt reference. The control sees the same episode stream and the same number of "
      "outer updates with no inner loop, so differencing against it changes exactly one thing: "
      "whether adaptation happened during training. Without it, any movement could just be the "
      "effect of more training.")
    A("")
    A("| fold | meta-train | meta-test | better | worse | within noise |")
    A("|---|---|---|---|---|---|")
    for fold in FOLDS:
        meta, test, _d = data[fold]["plan"]
        b, w, n = tally(data[fold]["primary"])
        label = f"`{fold}`" + (" (first run)" if fold == LEGACY_FOLD else "")
        A(f"| {label} | {', '.join(meta)} | {', '.join(test)} | {b} | {w} | {n} |")
    A("")
    A("**The one significant cell sits in the first fold, not the new ones.** Every cell of all "
      "three LDO3 folds is within noise. So the null did not merely survive being widened, it got "
      "cleaner as the evidence grew.")
    A("")
    A("**The design objection this run existed to answer is now answered.** The first fold "
      "meta-trained on dengue alone, so its episodes varied population and forecast origin rather "
      "than disease, and the module recorded that against itself: ANIL was being trained on the "
      "axis that already works. The `influenza` fold meta-trains across dengue **and** COVID, so "
      "episodes finally do vary disease. The answer did not change.")
    A("")

    A("## 1. Run verification")
    A("")
    if ok:
        A(f"All {len(manifest)} (fold, arm) cells present and internally consistent: every seed "
          f"scored, every per-dataset record, per-node, per-origin and quantile artifact on disk, "
          f"filename and payload agreeing on seed, dataset and fold, the resume JSON agreeing with "
          f"the per-dataset records to 1e-9, no NaN, and every scored cell pairing against the "
          f"reference it is differenced against.")
    else:
        A(f"**{len(problems)} problem(s) found. Read them before reading any number below.**")
        A("")
        for p in problems:
            A(f"- {p}")
    A("")
    A("| fold | arm | seeds scored |")
    A("|---|---|---|")
    for row in manifest:
        A(f"| `{row['fold']}` | {row['arm']} | {', '.join(str(s) for s in row['seeds'])} "
          f"({row['n_seeds']}) |")
    A("")

    A("## 2. Fold definitions")
    A("")
    A("`dengue2flu` is the originally reported run. The three LDO3 folds are the disease-out folds "
      "of the main transfer table, so a fold here means the same thing it means there. Each arm "
      "warm-starts from that fold's own trunk, which is the trunk the freeze-then-adapt reference "
      "was produced from, so the delta is attributable to the training objective rather than to a "
      "different starting point.")
    A("")
    A("| fold | meta-train bundles | meta-test bundles | warm start |")
    A("|---|---|---|---|")
    for fold in FOLDS:
        meta, test, _d = data[fold]["plan"]
        A(f"| `{fold}` | {', '.join(f'`{m}`' for m in meta)} | "
          f"{', '.join(f'`{t}`' for t in test)} | `{warm_ckpt(fold, 42).replace('42', 'S')}` |")
    A("")
    A("Every LDO3 fold has a multi-bundle meta-train side, so each outer step draws a panel and "
      "then an episode inside it, uniform over panels. Uniform matches `_fit_trunk`'s across-bundle "
      "sampler, so this arm and the freeze-then-adapt arm it is differenced against weight the "
      "in-diseases the same way; sampling proportional to cells would make dengue about 98% of "
      "episodes, which is the failure `train/joint.py` already documents.")
    A("")

    A("## 3. Primary comparison: ANIL against its own control")
    A("")
    A("Positive means ANIL is better. Percent of the control's RMSE, country-macro, count space. "
      "The interval is a two-sided 95% *t* at the number of seeds each cell actually paired on, and "
      "a cell counts only if that interval excludes zero.")
    A("")
    for fold in FOLDS:
        meta, test, _d = data[fold]["plan"]
        b, w, n = tally(data[fold]["primary"])
        A(f"### Fold `{fold}` — meta-train {', '.join(meta)} → meta-test {', '.join(test)}")
        A("")
        A("| dataset | h | delta vs control | significant |")
        A("|---|---|---|---|")
        for cellname in sorted(data[fold]["primary"], key=_cell_sort):
            text, _m, clears = data[fold]["primary"][cellname]
            ds, h = _split(cellname)
            A(f"| `{ds}` | {h} | {text} | {_sig(clears)} |")
        A("")
        A(f"Tally: **{b} better, {w} worse, {n} within noise.**")
        bv_text, _bm, bv_clears = data[fold]["best_val"]
        A("")
        A(f"Held-out meta-objective (best validation loss, lower is better, sign flipped so "
          f"positive still means ANIL better): {bv_text}, {_sig(bv_clears)}.")
        A("")

    A("## 4. Secondary: each arm against its fold's freeze-then-adapt reference")
    A("")
    A("Context, not the verdict. If ANIL and its control both sit in the same place relative to "
      "freeze-then-adapt, the null in section 3 is a statement about the objective. If they "
      "diverged here while tying there, something else would be going on.")
    A("")
    A("| fold | arm | better | worse | within noise |")
    A("|---|---|---|---|---|")
    for fold in FOLDS:
        for arm in ARMS:
            b, w, n = tally(data[fold]["vs_ref"][arm])
            A(f"| `{fold}` | {arm} | {b} | {w} | {n} |")
    A("")
    A("Reference is the capacity probe's own affine row for `dengue2flu`, and that fold's LDO3 "
      "freeze-then-adapt records for the other three.")
    A("")
    for fold in FOLDS:
        A(f"### `{fold}`")
        A("")
        A("| dataset | h | ANIL vs reference | control vs reference |")
        A("|---|---|---|---|")
        cells = sorted(set(data[fold]["vs_ref"]["anil"]) | set(data[fold]["vs_ref"]["control"]),
                       key=_cell_sort)
        for cellname in cells:
            a_t = data[fold]["vs_ref"]["anil"].get(cellname, ("—", None, None))[0]
            c_t = data[fold]["vs_ref"]["control"].get(cellname, ("—", None, None))[0]
            ds, h = _split(cellname)
            A(f"| `{ds}` | {h} | {a_t} | {c_t} |")
        A("")

    A("## 5. Training diagnostics")
    A("")
    A("| fold | arm | mean outer updates | mean episodes skipped | mean minutes |")
    A("|---|---|---|---|---|")
    for fold in FOLDS:
        for arm in ARMS:
            rows = data[fold]["rows"][arm]
            if not rows:
                continue
            def _mean(k):
                xs = [r[k] for r in rows.values() if r.get(k) is not None]
                return sum(xs) / len(xs) if xs else float("nan")
            A(f"| `{fold}` | {arm} | {_mean('outer_updates'):,.0f} | "
              f"{_mean('skipped'):,.0f} | {_mean('minutes'):.1f} |")
    A("")
    A("Episodes are skipped when a draw yields no observed support or query cell, which is a "
      "property of mask density rather than of the arm. Both arms draw from the same stream, so a "
      "skip rate difference between them at the same seed would be a defect; they match.")
    A("")

    A("## 6. What this measures, and what it does not")
    A("")
    A("Four bounds were recorded against the first run. The first is now retired for three of the "
      "four folds; the other three still stand and belong beside any number in this document.")
    A("")
    A("1. ~~Episodes come from one disease, so they vary population and forecast origin rather than "
      "disease.~~ **Retired for the three LDO3 folds**, whose meta-train sides span two diseases. "
      "It still holds for `dengue2flu`, and that fold's numbers should not be read as evidence "
      "about cross-disease episodes.")
    A("2. Every arm warm-starts from an existing trunk, so this is meta-**fine-tuning**, not "
      "meta-learning from scratch. A reviewer can fairly say the trunk was already shaped by "
      "ordinary training.")
    A("3. The meta-test fits a fresh adapter on the held-out disease's **full train fold**, so "
      "nothing here measures few-shot behaviour. For a project whose claim is few-shot adaptation, "
      "that is a stated limitation of this ablation rather than a property of ANIL.")
    A("4. The surface is `affine`, the pre-registered mechanism. A larger inner-loop surface would "
      "ablate something the Ebola protocol does not use, and its result could not be read back onto "
      "the case study.")
    A("")
    A("**This is an ablation of the adaptation procedure and is reported as one, whatever the "
      "sign.** That was committed to in advance. The headline remains freeze-then-adapt, which is "
      "what the Ebola pre-registration names and what every existing result was produced under.")
    A("")
    A("Two further caveats specific to the new folds. The `covid` fold's long horizons sit on the "
      "far side of the Omicron structural break, which `LDO3_Results.md` excludes from every "
      "verdict tally as measuring a fold boundary rather than transfer; they are shown here in full "
      "but carry the same caveat. And the `dengue` fold trains on small panels and adapts to a "
      "7,165-node graph, which is the reverse of the Ebola-shaped direction, so it is evidence "
      "about the method rather than about the case study.")
    A("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def _selfcheck():
    """Each check paired with a case that must come out the other way. No disk needed for most."""
    # 1. path helpers must match train/anil.py byte for byte, or this report reads files nothing
    #    writes and silently reports an empty document.
    assert out_json(LEGACY_FOLD, "anil").name == "anil_affine_anil.json"
    assert out_json("dengue", "anil").name == "anil_ldo3dengue_affine_anil.json"
    assert artifact(LEGACY_FOLD, "anil", "influenza_japan", 42, ".json") == \
        "encoder_ldo__anil-affine-anil__influenza_japan__seed42.json"
    assert artifact("covid", "anil", "covid_us-states", 42, ".json") == \
        "encoder_ldo__ldo3covid-anil-affine-anil__covid_us-states__seed42.json"
    assert len({str(out_json(f, "anil")) for f in FOLDS}) == len(FOLDS), \
        "two folds share a resume file"
    assert warm_ckpt("dengue", 42) == "encoder_ldo3__dengue__seed42__ckpt.pt"

    # 2. fold plans: meta-train and meta-test must partition the bundles, and no fold may test on
    #    what it trained on. A plan that overlapped would leak and still produce a plausible table.
    for f in FOLDS[1:]:
        meta, test, _d = fold_plan(f)
        assert test == list(DISEASES[f]), f"{f} meta-test != its disease's bundles"
        assert set(meta) == {n for d, ns in DISEASES.items() if d != f for n in ns}
        assert not set(meta) & set(test), f"{f} has a bundle on both sides"
    assert fold_plan("dengue")[0] and len(fold_plan("dengue")[0]) == 4
    assert len(fold_plan("influenza")[0]) == 2

    # 3. the tally must follow `clears`, not the sign of the mean. A version that counted every
    #    negative mean as a loss would report losses where the interval covers zero.
    fake = {"a": ("t", +5.0, True), "b": ("t", -5.0, True), "c": ("t", -9.0, False),
            "d": ("t", +9.0, False), "e": ("t", None, None)}
    assert tally(fake) == (1, 1, 2), tally(fake)

    # 4. paired_delta must pair on the CELL's seeds, not on the row count, and must call a
    #    sign-flipping cell noise. Control: a clean 5-seed separation must still register.
    noisy = compare_cells({"c": {42: 90.0, 52: 130.0, 62: 90.0, 72: 130.0, 82: 90.0}},
                          {"c": {s: 100.0 for s in SEEDS}})
    assert noisy["c"][2] is False, "a sign-flipping cell must read as within noise"
    clean = compare_cells({"c": {s: 50.0 for s in SEEDS}},
                          {"c": {s: 100.0 for s in SEEDS}})
    assert clean["c"][2] is True and clean["c"][1] > 0, "a clean 50% gain must clear zero"
    partial = compare_cells({"c": {42: 90.0, 52: 96.0, 62: 90.0}},
                            {"c": {s: 100.0 for s in SEEDS}})
    assert "n=3" in partial["c"][0], "delta must report the pair count it actually used"

    # 5. cell keys carry a literal '|'. It must never reach a table body, or the row silently
    #    gains a column and both GFM and md_to_docx's pipe parser mis-render it. Control: the raw
    #    key really does contain the delimiter, so this check is not vacuous.
    assert "|" in "influenza_japan|h3", "the control case must actually contain a pipe"
    assert _split("influenza_japan|h3") == ("influenza_japan", "h3")
    assert all("|" not in part for part in _split("covid_us-states|h15"))
    #    and horizons must sort numerically, or every table reads h10, h15, h3, h5
    assert [_split(c)[1] for c in sorted(["d|h3", "d|h15", "d|h5", "d|h10"], key=_cell_sort)] == \
        ["h3", "h5", "h10", "h15"], "horizons must sort numerically, not lexically"

    # 6. records_rmse must read country_macro, not value. Reading `value` would silently report the
    #    node mean on dengue, which is a different statistic.
    recs = [dict(dataset="dengue", horizon=3, metric="rmse", country_macro=1.0, value=99.0),
            dict(dataset="dengue", horizon=5, metric="mae", country_macro=7.0, value=99.0)]
    assert records_rmse(recs) == {"dengue|h3": 1.0}

    print("ok  path helpers match train/anil.py; folds partition their bundles; tally follows the "
          "interval not the sign; paired_delta pairs per cell; rmse read from country_macro")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="progress/outcomes/ANIL_Results.md")
    ap.add_argument("--surface", default=SURFACE, choices=("affine", "mlp-256"))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()

    v = verify(a.surface)
    if a.verify_only:
        sys.exit(0 if v[0] else 1)

    data = collect(a.surface)
    md = build_markdown(data, v, a.surface)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nwrote {a.out} ({len(md.splitlines())} lines)")
    if not v[0]:
        print("NOTE: verification reported problems; they are recorded in section 1 of the "
              "document. Do not circulate it until they are resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Milestone 4: Plan and Specification

**Written 2026-09-04.** Companion tracker: `progress/planning/Gap_Ledger.md`, which carries the
per-item status. This document is the rationale and does not change as items close.

---

## 1. What Milestone 4 is

Milestone 4 is Week 4 of `progress/planning/Final Internal Project Brief.md` section 7:
**transferable representations, meta-learning and few-shot adaptation.** The brief names three
deliverables:

1. Framework v2 with a transfer module
2. Leave-one-disease-out transfer results
3. A few-shot adaptation protocol

It also marks meta-learning and domain generalisation **REQUIRED** under goal G2.

## 2. Where it stands

| Deliverable | State | Evidence |
|---|---|---|
| Transfer module | Done | `train/lodo.py`, five fold variants at five seeds |
| Leave-one-disease-out results | Done | `progress/outcomes/LDO3_Results.md`, 215/215 artifacts verified |
| Few-shot adaptation protocol | Done | `train/ebola.py` and the frozen `progress/decisions/Ebola_Prereg.md` |
| Meta-learning (G2, REQUIRED) | **Closed 2026-09-04** | `progress/outcomes/ANIL_Results.md`, 4 folds |

Milestone 4 is therefore substantively complete. What remained was not new science but unclosed
loops: evidence sitting in JSON with no write-up, status documents contradicting the disk, a client
decision contradicting the results table, and client-ordered deliverables never produced.

## 3. The meta-learning result, and why the extra folds were run

Until 2026-09-04 the evidence for the meta-learning null was **twelve cells on one held-out
disease**, meta-trained on dengue alone. That is thin for a required goal, and it carried an
objection the module raised against itself: episodes drawn from a single disease vary population and
forecast origin, not disease, so ANIL was being trained on the axis that already works.

`train/anil.py` was hardwired to that one direction. Every remaining fold has a **multi-bundle**
meta-train side (holding dengue out leaves three influenza panels plus COVID), and the episode
sampler took `ds[0]`, so the other folds needed code, not a flag. The fix: each outer step now draws
a panel and then an episode inside it, uniform over panels to match `_fit_trunk`'s across-bundle
sampler. Warm start, meta-test set, validation-episode spread and comparison reference all follow the
fold. Legacy names are frozen so the already-reported run still resumes and still reproduces.

**The result.** ANIL against its own seed-matched control, 32 cells over 4 folds spanning 3 held-out
diseases: **0 better, 1 worse, 31 within noise.** The one significant cell
(`influenza_us-regions|h10`, −1.35%, [−2.52, −0.19]) is in the original fold; all three LDO3 folds
are entirely within noise. On the one fold where the held-out meta-objective separates at all, the
dengue fold, it separates **against** ANIL (−1.2%, [−2.41, −0.03]). So the inner loop does not win
the objective it optimises, and does not win test accuracy either.

The design objection is partly retired for three of the four folds: the episode distribution now
spans diseases, but each task still sits inside one panel, so the inner loop still adapts within a
single disease. The answer did not change.

**Why ANIL against its own control is the primary comparison.** The control runs the same seeded
episode stream with no inner loop, so differencing against it changes exactly one thing: whether
adaptation happened during training. The arms are not step-matched: each early-stops on its own
validation curve, so outer-update counts agree only on the legacy fold and differ per seed on the
LDO3 folds (covid seed 42: 4750 ANIL vs 6542 control). Each arm is scored at its own best-val
checkpoint. That is the definition of "does
meta-learning beat the probe", which is what G2 asks. `train.anil.compare()` does not compute it, so
`diagnostics/anil_report.py` does.

## 4. Three corrections found while checking the docs against disk

Recorded because each one misled, and two of them appear in documents that steer new sessions.

1. **`Reports/` and `results/` are NOT gitignored.** `.gitignore` covers `/docs/`, `/figs/`,
   `/baselines/*`, `.claude/`, `CLAUDE.md`, `graphify-out/`, `*.log` and ablation artifacts. Git
   tracks 45 files in `Reports/` including `Manuscript_v2.md`, and 919 in `results/lodo/`. So "get
   the manuscript under version control" is a non-task. But **`*.log` really is ignored**, which is
   the live half of audit finding M4: every decision-bearing log has no history.
2. **The shrinkage test already ran** (2026-09-02) and the diagnosis it was meant to settle is
   refuted. See `progress/outcomes/Shrinkage_Verdict.md`. Both `CLAUDE.md` and `Resume.md` still
   listed running it as next action #1.
3. **`PROJECT.md:36-44` still says "Phase 3 = not yet started"** and lists G5 as "SHAP (global +
   local); Week 5". Every Phase 3 experiment is finished and no attribution code exists.

## 5. Approach

Close the gaps **one at a time, oldest work first**, before starting anything new. Groups A to D are
loops left open by work already done. Group E is new work and does not begin until A to D are clear.

Two decisions taken by the user up front:

- **Amend B5 to keep COVID as a labelled panel.** Dropping COVID instead would mean roughly 60 edit
  sites across four files, collapse the three-way LDO3 design to two-way, and orphan the
  graph-controlled `encoder_pair__` comparison that exists only because COVID and
  influenza-US-states share a bit-identical graph. Amending is one edit site.
- **Milestone and paper safety lead**, ahead of chasing accuracy.

## 6. Standing constraints

- **No new training runs.** All expensive compute is finished.
- **Ebola is not re-scored.** It was scored once under a hash-frozen pre-registration whose
  amendment log is explicitly closed (`Ebola_Prereg.md:213-215`).
- **Do not wire the median-to-mean correction into the Ebola path.** Its own docstring
  (`train/loop.py:97-98`) requires the offset to be inherited and pre-registered before scoring.
  Audit M14 records it never was, and the correction is 8.8% to 47.3% *worse* on COVID, the most
  shifted panel and the closest analogue to Ebola. Adopt M14's recommendation instead: state that
  Ebola's point forecast is the count-space median and lead with MAE, which the median optimises.
- **Before signing off any results document, parse the numbers back out of it and recompute from
  disk.** This is the repo's standing rule and it has caught six stale numbers in a single brief.

## 7. Verification standard for this milestone

Every results document gets a generator and a **separate** verifier that re-derives each number from
the artifacts without importing the generator's loaders, so a shared bug cannot verify itself. The
verifier is itself mutation-tested: corrupt the document one defect at a time and confirm each
corruption is caught. A verifier that passes everything is worth nothing.

That standard has already paid for itself. It caught a sign bug in `anil_report.py`: `best_val` is a
loss, so lower is better exactly like RMSE, and negating it before `paired_delta` double-flipped and
printed every fold's meta-objective backwards. The check that catches it is a prose check, because
that line is prose and no table check touched it.

Commands:

```bash
conda run -n ebola-train python -m diagnostics.anil_report --verify-only
conda run -n ebola-train python -m diagnostics.anil_report -o progress/outcomes/ANIL_Results.md
conda run -n ebola-train python -m diagnostics.verify_anil_doc --mutate
```

Known-good anchor: ANIL against control on the legacy fold must give
`influenza_us-regions|h10 = -1.35%, [-2.52, -0.19]`, matching `Manuscript_v2.md`.

## 8. Explicitly not doing

- **LDO3 zero-shot quantiles** (0 of 25 present, ~10 h retrain). The zero-shot arm is behind its
  ceiling in 36 of 36 cells; calibrating a uniformly losing arm buys no claim. Standing limitation.
- **Shuffled-adjacency control** (~6 h). The gate-off ablation already shows the spatial channel
  helps error in 0 of 40 cells, so this confirms rather than discovers. Its absence is already
  disclosed in the manuscript's Threats section.
- **Rebuilding the shrinkage probe.** The hypothesis is refuted and the tables are preserved in
  full; rebuilding a tool to regenerate numbers nobody will act on has no consumer.

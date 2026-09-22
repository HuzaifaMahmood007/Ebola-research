# Milestone 6: Plan and Specification

**Written 2026-09-21.** Companion tracker: `todo_Milestone6.md` at the repo root, which carries the
per-item checkboxes. This document is the rationale and does not change as items close.

---

## 1. What Milestone 6 is

Milestone 6 is Week 6 of `progress/planning/Final Internal Project Brief.md:124-128`, days 26 to 30:
**ablations, robustness, writing and documentation.** The brief names three deliverables:

1. Ablation and robustness study
2. A submission-ready manuscript
3. A reproducibility package

None of this is new science. The science is finished and the thesis is settled. Milestone 6 turns a
finished body of work into something a journal and a stranger can both check.

## 2. Where it stands

| Deliverable | State | Evidence |
|---|---|---|
| Ablation and robustness study | **Partial.** Gate-off and epi-informed ablations done; four confirmatory runs queued | `ablation/single/` (210 records), the four runs in section 4 below |
| Submission-ready manuscript | Open | `Reports/Manuscript_v2.md`, 13,834 words, section 9.8 a stub, several false sentences |
| Reproducibility package | Open | README a month stale, no LICENSE, no CITATION, no installable env spec |

The honest thesis this milestone ships: a shared representation transfers to an unseen pathogen well
enough to beat naive floors and carries its uncertainty calibration with it, but every mechanism
added to *improve* transfer (spatial message passing, few-shot adaptation, meta-learning) fails to
help. That is a boundary-conditions paper and it is publishable as one. Milestone 6 does not try to
rescue any of the negative results. It documents them and closes the loose ends.

## 3. Approach

Cheap and independent work first, expensive and gated work last. The manuscript is the largest single
piece and needs no compute, so it runs in parallel with the cheap ablations. The one expensive run is
gated behind the cheap ones on purpose, so we do not spend six hours before the cheap tests have told
us what they tell us.

Two standing constraints carry over unchanged:

- **The Ebola pre-registration is never re-scored.** Everything touching Ebola in this milestone is
  exploratory, lives in `experiments/`, is labelled `EXPLORATORY`, and never touches the frozen
  `results/ebola/` records (`experiments/README.md:3-10`). The Bonferroni sensitivity is a percentile
  change on the read-only `ebola_ci.py`, not a re-score.
- **Before any results document is signed off, its numbers are parsed back out and recomputed from
  disk**, prose and significance labels included, and then one input is broken on purpose to confirm
  the check catches it. This is the repo's standing rule and it has caught six stale numbers in a
  single brief.

## 4. The four queued runs

Queued by the user's decision on 2026-09-21, in this order. Runs 1 to 3 are cheap and independent.
Run 4 is gated on runs 1 to 3 completing.

| # | run | cost | who runs it | goes where |
|---|---|---|---|---|
| 1 | Classical baselines: GBM on lag features plus ARIMA / SARIMA | 1 to 2 h, dengue ARIMA dominates | user's shell | dev panels to `results/baselines/`, Ebola arm exploratory to `experiments/` |
| 2 | Constrained-adapter causal test: refit the Ebola adapter ridge-regularised toward identity, selected on support only | minutes per seed | I run it | `experiments/`, exploratory |
| 3 | Input-vs-representation district-energy check | inference-only, minutes | I run it | machine-readable output plus a verifier |
| 4 | D2 shuffled-adjacency RETRAIN, gated on 1 to 3 | ~6 h | user's shell | ablation record plus manuscript update |

Plus a free add-on riding on run 3's cycle: rerun `ebola_ci.py` at Bonferroni percentiles
`[0.156, 99.844]` for a family of m = 16, as a multiplicity sensitivity.

### What each run buys the paper

- **Run 1 (classical baselines).** Closes an unfulfilled client order at `Review Doc.md:79`: "Add
  ARIMA or SARIMA and a gradient boosted model on lag features to the simple baseline set. The GBM in
  particular is hard to beat and I would much rather find that out from you than from a reviewer." That
  order is recorded nowhere as done or declined, and `results/baselines/` holds only the four GNNs. A
  reviewer will reach for the GBM first, so we run it first.
- **Run 2 (constrained adapter).** Closes the causal gap the mechanism doc names against itself at
  `Adapter_Mechanism_2026-09-16.md:122-124`: the extrapolation finding is geometric, not an error
  decomposition. Constraining the adapter toward identity and seeing whether error recovers is the
  causal step the geometry cannot take alone. Until this lands, the mechanism is written as geometric,
  never as *the* cause.
- **Run 3 (input-vs-representation).** Answers whether district structure was absent in the data or
  discarded by the encoder, the question `Session_Audit_2026-09-10.md:512-515` raises and did not run.
  Inference-only on `results/single/` checkpoints. Also produces the first machine-readable artifact
  for the graph-probe family, which currently exists only as prose.
- **Run 4 (D2 retrain).** Converts "we did not run a shuffled-adjacency arm"
  (`Manuscript_v2.md:332`, `:525`) from a disclosed gap into a closed test. It is the last remaining
  spatial test and the final close of the graph-does-not-help finding. See section 6 on why it was
  reopened.
- **Bonferroni add-on.** The manuscript controls multiplicity on the confirmatory family only
  (`Manuscript_v2.md:521`). A named family-wide Bonferroni sensitivity answers the reviewer who asks
  what happens under a 16-cell correction. A crude Gaussian estimate says the "14 comparisons clear
  zero" count thins to roughly 8, but the exact number needs the rerun and only the rerun's number is
  quotable.

## 5. The manuscript

`Reports/Manuscript_v2.md` is 13,834 words against a 12,000 internal target that no journal has
confirmed (the target journal is still open at brief `:200`). Over by 1,834, but the cut is larger,
because three sections must grow before anything is cut: section 9.8 explainability is a 44-word
PENDING stub (`:499-501`) while G5 closed 2026-09-16, there is no normalisation Threats paragraph, and
the graph-mechanism finding is absent. Plan for a gross cut of roughly 2,300 to 2,600 words to net out
at the limit with the new content in, and measure after every edit.

Beyond length, the manuscript carries content that is now false against disk and must be corrected:
two sentences say we did not run a shuffled-adjacency arm when the inference-only relabelling control
ran; one says we have no explanation for the adaptation inversion when a geometric mechanism now
exists; and one live pre-registration violation labels an L12 h15 table row "few-shot" where the
prereg binds h10 and h15 to zero-shot on the primary arm (`Ebola_Prereg.md:191-192`). The full list is
in `todo_Milestone6.md` Task 2.

## 6. Decline list, with reasons

Recorded so declined work does not leak back in.

- **Aggregator swap** (mean to sum, max or spread). Declined: it is a retrain, not an inference tweak,
  and the `t4` measurement counts information *available*, not information that *helps*. A high-variance
  channel can be noise.
- **L20-no-blackout counterfactual.** Declined: it needs a support window the frozen pre-registration
  does not contain (`Adapter_Mechanism_2026-09-16.md:125-126`).
- **D1** (LDO3 zero-shot quantiles, ~10 h retrain). Stays closed: the zero-shot arm is behind its
  ceiling in 36 of 36 cells, so calibrating a uniformly losing arm buys no claim (`Gap_Ledger.md:275`).
- **D3** (median-to-mean correction into the Ebola path). Stays closed: the prereg amendment log is
  closed and the correction is 8.8% to 47.3% worse on COVID, the closest analogue to Ebola
  (`Gap_Ledger.md:277`). The documentation fix is already in the manuscript.

### Why D2 was reopened when D1 and D3 were not

D2 (shuffled-adjacency retrain) was recorded as decided-no at `Gap_Ledger.md:276` on the cost-versus-
evidence argument: the gate-off ablation already showed the spatial channel helps error in 0 of 40
cells, so a six-hour retrain would confirm rather than discover. The user reopened it on 2026-09-21
under a different standard. This is the final close of the paper, D2 is the last remaining spatial
test, and the gate-off ablation and the inference-only relabelling control have both run. Under a
final-close standard no gate stays open, so the cost-versus-evidence rationale is superseded. D1 and
D3 are not spatial tests and their declines rest on evidence that has not changed, so they stay closed.
The ledger entry records both the original decision and the reopening.

## 7. Definition of done

Milestone 6 is done when all three hold:

1. **The manuscript is submission-ready.** At or under the word target, section 9.8 written, the
   normalisation Threats paragraph and the graph-mechanism finding added, every false sentence
   corrected, the E6 label fixed, the code URL real, and every number recomputed from disk against a
   verifier that has been shown to fail on a broken input.
2. **The reproducibility package is complete.** README rewritten to the current six-dataset state,
   LICENSE and CITATION present, an installable `ebola-train` env spec, and every stale record count
   (baselines 280, lodo 208) corrected wherever it appears.
3. **All evidence is committed.** The untracked `experiments/` results, the norm-probe and
   disease-mean-floor docs and verifiers, the three new figures with a generator each, and the 210
   ablation records freed from the `/ablation` gitignore line, all under version control. The two
   failing verifiers pass. The stale OPEN markers are cleared.

## 8. Explicitly not doing

- **No new science.** Every negative result stands as measured. Milestone 6 documents them, it does not
  re-litigate them.
- **No re-scoring of the Ebola pre-registration**, under any framing.
- **Aggregator swap, L20-no-blackout, D1, D3.** See section 6.

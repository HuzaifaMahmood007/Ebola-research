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

**Updated 2026-09-25.** The compute half of the milestone is CLOSED: runs 1 to 5 finished (section
9 carries the dated record), and the one live-tested spatial redesign failed its pre-registered
test (section 4B). What remains is writing and packaging.

| Deliverable | State | Evidence |
|---|---|---|
| Ablation and robustness study | **DONE 2026-09-25.** All queued runs finished and scored; every result doc carries a mutation-tested verifier | section 9 below; `progress/outcomes/` 2026-09-22 to 25 |
| Submission-ready manuscript | Open | `Reports/Manuscript_v2.md`, 13,834 words; the verified edit list is `progress/outcomes/Manuscript_Correctness_Pass_2026-09-23.md`, not yet applied |
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

## 4. The queued runs

Queued by the user's decision on 2026-09-21 (runs 1 to 4), plus run 5 added 2026-09-23. Runs 1 to 3
are cheap and independent. Run 4 is gated on runs 1 to 3 completing. Run 5 is itself two-staged with
its own hard gate, described in section 4A.

| # | run | cost | who runs it | goes where |
|---|---|---|---|---|
| 1 | Classical baselines: GBM on lag features plus ARIMA / SARIMA | 1 to 2 h, dengue ARIMA dominates | user's shell | dev panels to `results/baselines/`, Ebola arm exploratory to `experiments/` |
| 2 | Constrained-adapter causal test: refit the Ebola adapter ridge-regularised toward identity, selected on support only | minutes per seed | I run it | `experiments/`, exploratory |
| 3 | Input-vs-representation district-energy check | inference-only, minutes | I run it | machine-readable output plus a verifier |
| 4 | D2 shuffled-adjacency RETRAIN, gated on 1 to 3 | ~6 h | user's shell | ablation record plus manuscript update |
| 5 | Auxiliary district-identity objective: a gate probe first, then an overnight retrain only if the probe passes | probe minutes; retrain ~14 h if it runs | probe I run, retrain user's shell | probe to `results/misc/`, retrain to `ablation/` |

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
- **Bonferroni add-on. DONE 2026-09-28.** The manuscript controls multiplicity on the confirmatory
  family only (`Manuscript_v2.md:521`). The family-wide sensitivity is now run. The family is
  persistence only, both metrics, 32 cells; the divisor is 32; the "14 comparisons clear zero" count
  thins to **7**, 5 on RMSE and 2 on MAE. The old "roughly 8" Gaussian estimate is retired.

## 4A. Run 5: the auxiliary district-identity objective

**Added 2026-09-23, on the user's order.** This is the one run in the milestone that could still add a
mechanism to the encoder rather than only document one, so it is fenced tightly.

**The idea.** Add a second head to the encoder, a district classifier ("which district is this?")
trained on the representation `h` alongside the forecast loss, with weight `lambda_aux`. The forecast
loss and adapter are unchanged; the aux loss is the only new term. The bet is that this stops the
encoder collapsing district-specific structure. `progress/outcomes/Input_Energy_2026-09-22.md`
measured that collapse: the raw incidence input windows are 14 to 94 percent district-specific in
energy, and the encoder output `h` keeps only 3.8 to 22.4 percent, a three to five fold squash on
every one of the five panels. Forcing `h` to stay district-identifiable is meant to hold some of that
structure open for the spatial layer and the forecast head to use.

**The user's stated risk, kept verbatim in spirit.** If the squashed component was noise, forcing it
back in makes forecasts worse. Energy is not usefulness. A high-variance district channel can be
noise that per-node z-scoring amplified on small quiet nodes, and the encoder may have been right to
throw it away. The aux objective would then trade forecast accuracy for a district-classification
score nobody asked for.

**What bounds the upside, stated honestly.** D2 (`progress/outcomes/Shuffled_Adjacency_2026-09-23.md`)
measured that even the REAL adjacency, learned from scratch, earns only 2 to 4 percent and only on
dengue; the four small panels are within noise on error. The aux objective bets that a richer, more
district-specific `h` would raise that ceiling. That bet is untested, and D2 is the reason the ceiling
looks low.

**Two-stage structure with a hard gate.**

1. **Stage 1, the gate probe (cheap, minutes, I run it).** Before spending an overnight retrain, ask
   the cheaper question the probe can answer: is the discarded district information forecast-relevant
   at all? I fit per-node corrections on the FROZEN `results/single/` checkpoints, on the train period
   only, and check whether giving the model district identity post-hoc lowers TEST error beyond seed
   noise, contrasted against a district-agnostic correction of the same form. If a per-node correction
   beats the district-agnostic one beyond seed noise on any panel, the discarded structure is
   forecast-relevant and the retrain has a target. If nothing clears noise, the squash was justified
   and the overnight is not worth running. Design and leakage rule in `todo_Milestone6.md` section 1.6
   and in the script docstring (`diagnostics/graph_probe/aux_gate_probe.py`).

2. **Stage 2, the overnight retrain (only if the gate says GO).** Add the aux head and loss to
   `train/loop.py` behind a flag, mirror the sibling runner `ablation/run_shuffle_adjacency.py`, sweep
   a small `lambda_aux` grid or one justified value, and compare paired against `results/single/` under
   the same within-noise rule. The user runs it.

**The gate is decided on the probe numbers, not on judgement.** GO if the probe shows a beyond-noise
improvement from district identity on at least one panel, named with its size. NO-GO if nothing clears
noise, in which case the runner is NOT written; the no-go is recorded with numbers here and in the
todo, and the user decides whether to overrule. A single marginal cell is reported as borderline and
defaults to NO-GO, again with the user free to overrule. The gate outcome is recorded in both docs
either way.

**GATE OUTCOME: NO-GO (probe run 2026-09-23, scored 2026-09-24).** Full write-up and verifier:
`progress/outcomes/AuxGate_Probe_2026-09-23.md`, `diagnostics/verify_auxgate_doc.py` (passes,
mutation-tested 3 of 3). The decisive contrast is a per-node correction beating BOTH the baseline and
a district-agnostic global correction beyond seed noise. It fires in 2 of 40 bias cells (both
covid_us-states at h10) and 0 of 36 affine cells. The larger "node beats global" count (18 of 40 for
bias) is an artifact: on dengue and us-regions the global correction is itself worse than baseline, so
a per-node correction beating it just means it overfits less, while doing nothing beats both. The
richer per-district instrument (affine_node), the true analogue of what an aux head would produce,
never beats baseline, hurts on 8 cells, and blows up numerically on dengue's quiet z-scored nodes (up
to 3.07e7 error), a live instance of the user's stated risk. The one real district gain, covid h10, is
a per-district level offset achievable for free post-hoc, not a reason to retrain. Runner NOT written.
A separate, non-deciding observation: a post-hoc recalibration helps COVID (global level offset
+6.88% RMSE / +5.98% MAE at h10, clearing noise), worth a cheap follow-up but not the aux objective.

## 4B. Run 6: the V2 deviation channel, a pre-registered mechanism test. FAIL

**Added 2026-09-24 on the user's order, decided 2026-09-25.** After run 5's NO-GO, a model-free
probe (`progress/outcomes/Neighbour_Signal_2026-09-24.md`) found real neighbour signal in the raw
data on COVID (5.3 / 2.3 percent of target variance at h3 / h5) and dengue (about 2 percent at h3
to h10), none usable elsewhere, and none on Ebola once outbreak size is controlled. That created a
tension: the data carries neighbour signal, the encoder discards it (run 3), and the trained model
gains nothing from the real map (run 4). V2 tested the obvious remedy: a small deviation branch
(district minus the cross-sectional mean, the signal the encoder squashes) with its own spatial
mixer, added to the unchanged v1 path so v2 starts exactly at v1.

**How it was fenced.** Because the design was motivated by a probe that read test-period data, the
whole test was pre-registered: `progress/decisions/V2_Deviation_Protocol.md` (sha256 82069f41...)
was committed (39ecf34) BEFORE any COVID number existed. Three arms separate the causes (real map,
branch without graph, degree-preserving fake map). A two-stage design handled COVID's seed noise:
stage 1 measured the paired v2-vs-v1 spread on VALIDATION only (s = 0.0412), a pre-written formula
set N = 15 seeds, giving 0.822 power for the probe-implied +3.3 percent at h3 under a paired 95
percent t-interval rule. The runner enforces the order in code: a repro-check gate, no free arm
path, stage 1 blind to test, the report refusing until stage 2 is complete.

**Outcome, 2026-09-25: FAIL, and worse than a null.** v2 roughly DOUBLES COVID error: h3 RMSE
5306.8 to 10444.8 (paired d +5138, 95 percent CI [+3669, +6607]), h5 RMSE 8375.4 to 19896.6
(d +11521 [+7951, +15091]), significantly worse at h10 and h15 too, MAE the same, PCC at h3
collapsing from 0.436 to about zero. All three arms degrade identically (every v2graph vs
v2nograph and vs v2shuffled comparison within noise), so the deviation INPUT causes the damage and
the graph adds nothing on top. Per pre-registered criterion (c), no dengue run.

**What it means, fenced.** The user's stated risk from run 5 realised at scale: deviations that a
five-coefficient linear probe can exploit make a ten-thousand-parameter learned branch roughly
twice as bad. This is CONSISTENT WITH the encoder's compression working as a filter against
noise-dominated deviations rather than as a defect, but the exact failure mechanism (short-train
overfit, a train-to-test regime shift in COVID deviations, or optimisation interference) is NOT
established and must not be asserted. Do not write "the compression is a defence against
noise-dominated deviations" as a fact.

**DONE 2026-09-25.** Write-up `progress/outcomes/V2_Deviation_Result_2026-09-25.md`, verifier
`diagnostics/verify_v2_result_doc.py` (passes clean, mutation-tested 11 of 11). A read-only val-vs-test
check found the three v2 arms BEAT v1 on the validation split (10 to 20 percent at h3/h5) while losing
by 73 to 138 percent on test, so validation did not warn, which favours a regime-shift reading over a
val-blind overfit without settling it.

**What this closes.** The spatial-redesign question is closed for this paper. The compression finding
had one obvious remedy, the remedy was tested under a pre-registered protocol and failed, and no
further spatial redesign is planned.

**The manuscript gains one paragraph**, at the end of the graph-mechanism chain. Drafted here so it can
be dropped in during Task 2, and fenced so it does not overclaim causation:

> To test whether the encoder's compression of district structure is a fault to correct, I routed the
> district deviations directly to a learned spatial branch and retrained on COVID under a
> pre-registered protocol (15 seeds, three arms: the real map, no map, and a degree-preserving wrong
> map). The remedy did not help. It roughly doubled the short-horizon error, and a wrong map and no map
> did equal damage, so the harm came from the deviation input rather than the graph. A model-free
> linear probe recovers a small real neighbour signal from the same deviations (about 5 percent of
> target variance at h3), which is consistent with those deviations being noise-dominated at the
> capacity of a learned branch, though I do not establish the exact failure mechanism. On this
> evidence the encoder's compression behaves less like a defect to fix than like a filter a learned
> branch could not improve on.

**Fences on that paragraph, do not loosen them.** The pre-registration's FAIL row licenses "no effect
found at N seeds"; what we saw is stronger (the channel measurably hurt), so we may say that, but never
"the deviation channel has no effect" or "the graph is useless in general". The probe-versus-branch gap
is CONSISTENT WITH noise-dominated deviations, which is the risk the protocol named before the run, but
the three candidate mechanisms (short-train overfit, a train-to-test COVID regime shift, optimisation
interference) are not separated by this test. Do not write "the compression is a defence against
noise-dominated deviations" as an established fact.

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

### 5A. The epi-ablation sentence, two errors found 2026-09-25

`Manuscript_v2.md:497` currently reads:

> **The epidemiology-informed ablation is also a null.** Adding a growth-rate plausibility penalty to
> the training objective changes no cell beyond seed noise on any of the four small panels, because the
> model's forecasts are already smoother than the data at every bound we calibrated. We report that as a
> measured null rather than an untested design choice.

I checked this against the artifacts on 2026-09-25 and found two errors. The framing is fine: the
component is correctly presented as an ablation, the architecture is never sold as
epidemiology-informed, and no other sentence in the manuscript claims otherwise. Only these two clauses
are wrong.

**Error 1, "changes no cell beyond seed noise" contradicts its own report.** The verdict column of
`Reports/epi_p90max_report.txt` flags three cells:

| panel | metric | h | paired delta | verdict |
|---|---|---|---|---|
| influenza_us-regions | RMSE | 5 | -11.541 +- 9.462 | epi BETTER (n=5) |
| influenza_us-states | PCC | 3 | +0.002 +- 0.001 | epi BETTER (n=5) |
| covid_us-states | PCC | 10 | -0.017 +- 0.005 | epi WORSE (n=5) |

The conclusion still holds and the replacement wording is *stronger* than the current one, not weaker.
The report's rule is "within noise = |mean| < sd across seeds", so a cell flags at |mean| >= sd, which
at n=5 corresponds to roughly p = 0.09 rather than 0.05. Across 48 cells (4 panels x 3 metrics x 4
horizons) that predicts about 4.3 flags from pure noise. Three were observed, which is below chance.
The direction is also inconsistent: influenza_japan helps at h3/h5 and hurts at h10/h15 while
covid_us-states does the exact opposite, which is a noise signature rather than a mechanism. Say the
count and say why it is below chance. A reviewer who opens the report sees the `epi BETTER` labels
immediately, and "no cell" is the kind of claim that costs credibility for no gain.

**Error 2, the stated cause fits only one of the two bounds.** "The model's forecasts are already
smoother than the data at every bound we calibrated" is the p99 story, and p99 proves it: two panels
returned `+0.000 +- 0.000` across every cell, which is bit-identical runs and a gradient that is
identically zero. But `Reports/epi_p90max.log:10-29` records a nonzero mean train penalty at p90, from
2.2e-05 on influenza_us-states to 6.5e-04 on covid_us-states. A nonzero penalty means
`relu(|r| - r_max) > 0` somewhere, so the forecasts *did* cross the bar. The clause "at every bound" is
therefore false.

The true p90 cause is magnitude, not smoothness. With `lambda = 1.0` the penalty contributes roughly
0.005% to 0.1% of an objective whose pinball term is order 0.1 to 1 (targets are asserted to be about
unit scale at `train/loop.py:189`), so the term is about a thousand times too weak to steer training.
Two different nulls with two different causes, currently collapsed into the one cause that fits only
p99.

**A third fact worth one clause, because it explains Error 2's p99 half.** `ablation/epi_rmax.json`
shows the `max` aggregator selects covid_us-states at every gap, and COVID's per-gap p99 sits 1.5x to
3x above every other dataset (gap 2: covid 2.0355 against dengue 0.9359, japan 1.3554, us-regions
0.8704, us-states 1.0397). So `r_max` is a COVID statistic, and the flu panels were switched off by the
calibration rule rather than by any property of the model.

**Sequencing, so this sentence is not edited twice.** The user ordered a bound-and-lambda sweep on
2026-09-25, handed to a separate session: a disease-agnostic replacement for the `max` aggregator at
fixed lambda, plus lambda at 10 and 100 with the bound held fixed. If lambda = 100 moves anything, line
497 is rewritten rather than corrected. **Hold both fixes until that sweep reports.** The scope of the
null also needs stating either way: dengue was excluded from both bounds (~12.7 h for 5 seeds, 96% of
the job) and Ebola was never in scope, which is verified below.

**The sweep reported 2026-09-28. The hold is lifted, and line 497 is rewritten, not corrected.**
Result: `progress/outcomes/Epi_Bound_Lambda_2026-09-28.md`, verified by
`diagnostics/verify_epi_bound_doc.py` (393 checks, 45 of 45 mutations caught), commit `a0052b9`.
Pre-registered in `progress/decisions/Epi_Bound_Lambda_Protocol.md`, commit `b24063f`.

- New bound p99 median (one shared number, not any single panel's statistic), plus lambda 10 and 100.
  Six arms, 90 cells, four panels, dengue excluded.
- **No PASS in any of 18 arm-panel units.** 6 FAIL, 12 INCONCLUSIVE, under a paired 95 percent
  t-interval that replaces the old `|mean| >= sd` rule.
- Where the term became a real part of the objective (3.244 to 30.654 percent of the loss, Japan and
  COVID), it improved nothing on both RMSE and MAE at any horizon.
- **It hurt `influenza_japan` h10**, significantly on both metrics, in all four new Japan arms, up to
  about 12 percent RMSE. I ruled out trainer drift: seed 42's baseline reproduces exactly and shows
  the damage too. Likely cause, still a hypothesis: the shared bound calls 8 to 10 percent of Japan's
  own real training transitions implausible.
- The two US panels never reached the 1 percent gate, even at lambda 100, so they are INCONCLUSIVE.

What line 497 must now carry: it is no longer "a null". It is "did not help where it had weight, and
hurt one panel", with the US panels stated as untestable at these lambdas, and dengue and Ebola out of
scope. Error 1's three-flag discussion is superseded by the pre-registered rule. The epi component
joins spatial message passing, few-shot adaptation and meta-learning in the did-not-help list.

**Verified, so the paper can say it plainly.** The epi component is absent from the Ebola path. An
unfiltered grep returns zero hits in `train/ebola.py`, `configs/ebola_arms.json` and
`progress/decisions/Ebola_Prereg.md`, and none of the 20 scored records in `results/ebola/` carries an
epi field. It is a dev-panel ablation only.

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
  re-litigate them. (Runs 5 and 6 were user-ordered exceptions, both fenced by pre-committed gates,
  and both came back negative. The exception is now spent: the spatial redesign question is closed
  for this paper.)
- **No re-scoring of the Ebola pre-registration**, under any framing.
- **Aggregator swap, L20-no-blackout, D1, D3.** See section 6.

## 9. Progress record, 2026-09-21 to 2026-09-25

Dated outcomes of the queued runs. Detail and verifiers live in the named docs; numbers here were
verified against disk when written. When this section and the disk disagree, the disk is right.

**Run 1, classical baselines. DONE 2026-09-22, after one leak found and fixed.** `run_classical.py`
built and run: GBM (5 seeds) plus SARIMA / ARIMA on all dev panels, Ebola GBM arm exploratory. The
first SARIMA / ARIMA run was INVALID: statsmodels reads an integer `dynamic` as an offset relative
to `start`, so every "h-step forecast" was a one-step-ahead prediction reading the observed test
series; the giveaway was error flat or falling with horizon. Fixed (`dynamic=0` plus a per-node
explosion guard) and re-run clean. Verdicts on the corrected records: the encoder loses 1 of 20
like-for-like cells to a classical model (SARIMA, COVID h5); GBM loses to plain persistence in 8 of
16 clean cells; SARIMA beats persistence 13 of 20; seasonality pays only on influenza; dengue
SARIMA / ARIMA fall back to persistence on 792 of 2,051 districts (38.6 percent). Standing caution:
commit cb6e165 contains the leaked records; never quote SARIMA / ARIMA from that commit. The
dengue estimand split (classical and GNN on the 2,392-node subset, encoder and floors on the full
6,161) is permanent and every comparison states it.

**Run 2, constrained adapter. DONE 2026-09-22. Verdict: partially causal.**
`experiments/adapter_constraint.py`, both arms, 5 seeds, support-only selection. Shrinking the
fitted adapter toward the zero-effect map recovers 77 to 90 percent of the few-shot damage on L12
(selected t about 0.34) but only 9 to 19 percent on L20 (t about 0.88), whose full-rank support
defeats any honest support-only rule, itself a finding. The clip variant recovers about +1
percent, so the damage lies in the magnitude of the fitted change along weakly constrained in-span
directions, refining the mechanism doc's out-of-span framing. A literal ridge cannot run the test
through the prereg optimiser (gradient clipping saturates it); closed-form affine shrinkage
reproduces both archived arms exactly. `progress/outcomes/Adapter_Constraint_2026-09-22.md`,
verifier 11 of 11 mutations caught.

**Run 3, district energy. DONE 2026-09-22. Verdict: present in the data, discarded by the
encoder.** `diagnostics/graph_probe/t5_input_energy.py`: input incidence windows are 13.8 to 93.5
percent district-specific in energy; the encoder output retains 3.8 to 22.4 percent, a 3.2x to
5.0x reduction on every panel, with effective dimensionality already narrow at the input. Fences:
incidence channel only is the fair comparison, and energy accounting cannot prove the discarded
part was signal. First graph-probe result with a machine-readable artifact
(`results/misc/t5_input_energy.json`). `progress/outcomes/Input_Energy_2026-09-22.md`.

**Run 4, D2 shuffled-adjacency retrain. DONE 2026-09-23. Verdict: the map is barely needed, dengue
the one exception.** Full 5-panel x 5-seed matrix, 9.7 h measured (the ledger's ~6 h estimate was
wrong). 49 of 60 cells within noise of the real-graph model; the real map earns a small
seed-stable advantage in 9 cells, 8 of them dengue (PCC at all four horizons +0.034 to +0.042,
error at h10 / h15 by 2 to 4 percent); influenza_japan h10 favours the shuffled graph. The strong
sentence "the graph contributed nothing at any stage" is NOT supported; the supported sentence is
"training barely needs the real map". `progress/outcomes/Shuffled_Adjacency_2026-09-23.md`,
verifier 4 of 4 mutations caught. Ledger D2: reopened, run, scored, history preserved.

**Run 5, aux district-identity objective. NO-GO 2026-09-24.** Section 4A carries the gate outcome
in full. Runner never written; the one real district gain (COVID h10 level offset) is achievable
post-hoc for free.

**Neighbour-signal probe (t6 family), 2026-09-24, model-free.** Real neighbour signal on COVID
(5.3 / 2.3 percent of target variance at h3 / h5) and dengue (1.8 / 2.4 / 2.0 / 0.5 percent, also
clearing a stricter within-country null and a per-panel false-alarm floor about 300 to 600 times
smaller than the gains); Japan real but tiny (0.03 to 0.12 percent); US-states borderline;
US-regions none. Ebola, EXPLORATORY: the apparent signal is outbreak size; once each district's
level is controlled, neighbours add nothing at h3 / h5 (a powered null), h10 unpowered, h15
unreadable. The probe shows neighbours carry forecast information, not that disease spreads
between them. `progress/outcomes/Neighbour_Signal_2026-09-24.md`, verifier 52 of 52 mutations
caught across its revisions.

**Run 6, V2 deviation channel. Pre-registered FAIL 2026-09-25.** Section 4B carries it in full.

**The spatial story this leaves for the manuscript, every link tested:** district signal exists in
the raw data (t6) -> the encoder compresses it 3 to 5x (t5) -> mean aggregation deletes most of
the rest (t3 / t4) -> training barely needs the real map, dengue the small exception (D2) ->
forcing the signal back in makes the model roughly twice as bad under a pre-registered test (V2).
That is consistent with the compression behaving like a defence against noise-dominated deviations
rather than a defect, though the exact failure mechanism is not established. On Ebola, the data
carries no neighbour timing signal beyond outbreak size, so no spatial method should be expected to
help there.

**Multiplicity add-on: RESOLVED 2026-09-28.** The two passes had built different families and both
called the answer "the 14". The canonical family is persistence only, RMSE and MAE, both arms, both
regimes, four horizons, 32 cells, 14 clearing zero at 95 percent
(`Reports/Phase0_to_Now_Audit.md:257`). The divisor is 32, the conservative choice. Recomputed with
the new `--bonferroni` and `--floor` options on `ebola_ci.py`: the divisor-16 control reproduces the
documented 7 of 14, and divisor 32 gives the same 7 of 14, 5 on RMSE and 2 on MAE. Logs at
`results/reports/ebola_district_ci_bonferroni32.log` and `..._bonferroni16.log`. One clause needed
narrowing: h15 against persistence survives on both arms **on RMSE only**, not on MAE.

**Also produced this window:** figures F1 to F6 with the generator `paper_figures.py`
(mutation-tested asserts); the verified manuscript edit list
(`progress/outcomes/Manuscript_Correctness_Pass_2026-09-23.md`, not yet applied); the session
record `progress/Session_Progress_2026-09-21_to_23.md`; the handoff `Resume.md` (2026-09-25).
Committed through 39ecf34 except: the V2 / stage run records in the gitignored `ablation/` tree
(need `git add -f`), the V2 result doc and verifier (in progress at the pause), and this update.

**What remains, in order:** finish and commit the V2 result doc and records; pin and recompute
the Bonferroni family; apply the manuscript edit list and add the V2 paragraph; the word-count
trim (blocked on the client's venue and limit answer, open since 2026-09-21); the reproducibility
package (README, LICENSE, CITATION, env spec, repo URL); free the ablation records from the
gitignore and clear the stale markers per section 7.

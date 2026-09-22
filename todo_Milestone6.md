# Milestone 6 todo

**Scope.** Week 6 of `progress/planning/Final Internal Project Brief.md:124-128`, days 26 to 30:
ablations (with vs without the epidemiology-informed component, standard vs meta-learning),
robustness and sensitivity analysis, finalise every table and figure, draft the full manuscript,
and finalise the reproducibility package. Nothing outside that week is listed here.

**Position, written 2026-09-21.** Milestone 5 closed. Every required goal G1 to G5 is delivered,
the compute for the science is finished, and the honest thesis is settled: a shared representation
transfers to an unseen pathogen well enough to beat naive floors and carries its calibration with
it, while every mechanism added to *improve* transfer (spatial message passing, few-shot adaptation,
meta-learning) fails to help. That is a boundary-conditions paper and it is publishable as one.

Milestone 6 is not new science. It is three things: run a short queue of cheap confirmatory
ablations the paper and the client are owed, turn the finished manuscript into a submission-ready
document that is at the word limit and no longer contradicts the disk, and package the repository so
a stranger can reproduce it. Four runs are queued by the user's 2026-09-21 decision, three of them
cheap and one that reopens a previously-declined item for a final close. Status below is checked
against disk on 2026-09-21, not against the progress docs.

Brief tasks, and where each one stands:

| # | brief task | status |
|---|---|---|
| 1 | Ablations and robustness / sensitivity analysis | **PARTIAL.** Gate-off and epi-informed ablations already ran (`ablation/single/`, 210 records). Four confirmatory runs queued 2026-09-21, none run yet |
| 2 | Finalise all tables and figures, draft the full manuscript | **OPEN.** 13,834 words vs a 12,000 internal target (`wc -w Reports/Manuscript_v2.md`, disk 2026-09-21). Eight tables, one figure, nine figure-ready artifacts unused. Section 9.8 is a PENDING stub. Multiple sentences now false against disk |
| 3 | Finalise source code, README, reproducibility package | **OPEN.** README a month stale, no LICENSE, no CITATION, no installable env spec, three stale record counts, placeholder repo URL |

---

## Task 1. Ablations and robustness

Four runs are queued in a fixed order (user decision 2026-09-21). Runs 1 to 3 are cheap and
independent. Run 4 is the expensive one and is **gated on runs 1 to 3 completing**, because it is the
final close of a previously-declined item and there is no reason to spend six hours on it until the
cheap tests have landed. Dev-panel results go to `results/baselines/`. Everything touching Ebola is
**exploratory** and goes in `experiments/`, labelled `EXPLORATORY` in its own records, never touching
the frozen pre-registration in `results/ebola/` (`experiments/README.md:3-10`).

### 1.1 Classical baselines: GBM on lag features plus ARIMA / SARIMA

This closes an **unfulfilled client order**. `progress/decisions/Review Doc.md:79` reads: "Add ARIMA
or SARIMA and a gradient boosted model on lag features to the simple baseline set. The GBM in
particular is hard to beat and I would much rather find that out from you than from a reviewer." That
order is recorded nowhere as done or declined. `results/baselines/` holds only the four GNNs, 280
JSON records (`ls results/baselines/*.json | wc -l`, disk 2026-09-21).

- [ ] Finish the runner script `run_classical.py` (being written 2026-09-21, no compute in it yet)
- [ ] Dev-panel records land in `results/baselines/`
- [ ] The Ebola arm is **exploratory** and lands in `experiments/`, never the frozen prereg
- [ ] User runs it in their own shell, one to two hours, dominated by dengue ARIMA
- [ ] Fold the GBM and ARIMA/SARIMA rows into the baseline table (Task 2), reporting them under the
      same country-macro and cell-pooled conventions as the four GNNs so the comparison is like for like

**What it buys the paper.** A reviewer will reach for the GBM first, exactly as the client said. If
the GBM beats us on a panel, better we report it than have it found. If we beat it, the SOTA story is
stronger. Either way an open client order closes.

### 1.2 Constrained-adapter causal test

The adapter mechanism doc names its own gap. `progress/outcomes/Adapter_Mechanism_2026-09-16.md:122-124`:
"These are geometric quantities, not error decompositions. They show the fitted map is applied far
outside where it was determined; they do not prove that each percentage point of extrapolation costs
a measurable amount of MAE. The link to the scored outcome is by argument, not by regression."

- [ ] Refit the Ebola adapter ridge-regularised toward identity, selected on support only
- [ ] Exploratory, in `experiments/`, minutes per seed
- [ ] Report whether pulling the adapter toward identity recovers error, which is the causal step the
      geometric measurement cannot take on its own

**What it buys the paper.** It moves the adaptation-failure story from geometric ("the map is applied
far outside its fitted range") toward causal ("constraining it back recovers error"). Until this
lands, the extrapolation mechanism must be written as geometric, not as *the* cause. See Fences.

### 1.3 Input-vs-representation district-energy check

`Session_Audit_2026-09-10.md:512-515` section F item 1 specifies the test: "measure the
district-specific share of energy in the *inputs* and compare it to the share in the representation.
If the inputs carry 30% and the representation carries 4%, the encoder is throwing it away. If the
inputs carry 5%, it was never there to throw away. I did not run it."

- [ ] Inference-only on the `results/single/` checkpoints, no retrain
- [ ] Answers whether district structure was absent in the data or discarded by the encoder
- [ ] Machine-readable output plus a short verifier, since the t1 to t4 graph-probe numbers currently
      live only as prose in `Session_Audit_2026-09-10.md` with no machine-checkable artifact (Housekeeping)

**What it buys the paper.** The graph-mechanism finding says the encoder emits about 2 of 64 useful
directions. A reviewer will ask whether the data ever had more. This settles that one way or the other
and it is inference-only.

### 1.4 Bonferroni multiplicity sensitivity (free add-on)

- [ ] Rerun `ebola_ci.py` at Bonferroni percentiles `[0.156, 99.844]` for a family of m = 16
- [ ] Report how the "14 comparisons clear zero" count thins under a 16-cell family correction. A
      crude Gaussian estimate says it drops to roughly 8, with the wide h15 intervals least robust.
      **The exact number needs the rerun. Do not write "roughly 8" as a result.**
- [ ] Read-only, no re-scoring of the pre-registration. `ebola_ci.py` rebuilds the macro from
      `*__pernode.npz` and asserts it matches the scored JSON before printing

**What it buys the paper.** The manuscript already controls multiplicity on the confirmatory family
only (`Reports/Manuscript_v2.md:521`). A named Bonferroni sensitivity on the wider exploratory grid
answers the reviewer who asks what happens when you correct the whole family.

### 1.5 D2 shuffled-adjacency RETRAIN, gated on 1.1 to 1.3

**This reopens a decided-no item.** `progress/planning/Gap_Ledger.md:276` records D2 as "Do not run"
(recorded as decision-log D21). The user reopened it on 2026-09-21 for the final close, because it is
the last remaining spatial test. The gate-off ablation and the inference-only relabelling control both
already ran; this is the retrain version, which asks whether a model *trained* on a fake graph would
do as well, and it is the stronger of the two.

- [ ] **Do NOT start until runs 1.1, 1.2 and 1.3 have completed.**
- [ ] About 6 hours, user's shell
- [ ] Retrain on relabelled adjacency (same topology, same degrees, wrong districts), compare error to
      the real-graph trunk
- [ ] Update the ledger D2 entry (already done in this milestone, see Housekeeping) and update the two
      manuscript sentences that currently say we did not run a shuffled-adjacency arm

**What it buys the paper.** It converts "we did not run a shuffled-adjacency arm"
(`Reports/Manuscript_v2.md:332`, `:525`) from a disclosed gap into a closed test. If a fake-graph
retrain matches the real-graph trunk, the "the graph does not help error" finding gets its strongest
evidence. That is the whole point of the final close.

### 1.6 Close-by-wording items (no compute)

These are robustness points the reviewer will want that the artifacts already answer. They need a
sentence, not a run.

- [ ] **Rolling-origin evaluation.** Ebola scoring already aggregates 18 origins x 3 countries, and dev
      panels 37 to 589 origins, in the `__perorigin.npz` archives. State it in one sentence.
- [ ] **Multiplicity.** The manuscript policy at `:521` plus the Bonferroni sensitivity from 1.4.
- [ ] **Block bootstrap.** The `:519` disclosure stands. Ebola's 18 origins make blocks weak there, so
      say the block-bootstrap sensitivity was checked and leaves the long-horizon conclusions intact
      rather than applying it throughout.
- [ ] **Epi-ablation scope.** The epidemiology-informed ablation ran on 4 of 5 panels. State that
      dengue at 7,165 nodes was excluded for cost, so a reader knows the null is on the four small panels.

### 1.7 Declined, with recorded reason

Listed so they do not leak back in.

- **Aggregator swap** (mean to sum/max/spread). Declined: it is a retrain, not an inference tweak, and
  `t4` measures information *available*, not information that *helps* forecasting. High-variance is not
  the same as useful.
- **L20-no-blackout counterfactual.** Declined: it needs a support window the frozen pre-registration
  does not contain (`Adapter_Mechanism_2026-09-16.md:125-126`).
- **D1** (LDO3 zero-shot quantiles, ~10 h) and **D3** (median-to-mean into the Ebola path) stay closed
  (`Gap_Ledger.md:275,277`).

---

## Task 2. Manuscript

`Reports/Manuscript_v2.md` is tracked. It is 13,834 words against a 12,000 internal target
(`wc -w`, disk 2026-09-21). Note the target is internal and **no journal has confirmed it**; the
target journal is still an open item at brief `:200`.

**The word arithmetic.** Over by 1,834. But the cut is larger than 1,834, because three sections must
GROW before anything is cut:

- Section 9.8 explainability is a 44-word PENDING stub (`Reports/Manuscript_v2.md:499-501`) while G5
  closed 2026-09-16. Writing it properly adds words.
- There is no normalisation Threats paragraph (Gap Ledger E2). Adding it adds words.
- The graph-mechanism finding is entirely absent. Adding a compressed version adds words.

So plan for a gross cut of roughly 1,834 plus whatever 9.8, the Threats paragraph and the mechanism
paragraph add, call it 2,300 to 2,600 words of removal to net out at 12,000 with the new content in.
Measure after each edit; do not trust the estimate.

### 2.1 Content that must be written IN

- [ ] **Write Section 9.8 explainability.** It is a stub at `:499-501` and the conclusion repeats the
      stub at `:527` ("One required capability is not delivered... explainability") and `:543` ("The
      explainability component is registered and not yet run"). G5 closed 2026-09-16. Write it plainly:
      integrated gradients with an occlusion cross-check, T1/T2/T3 all PASS, IG-vs-occlusion agreement
      264 of 280, worst completeness error 0.0040 on the largest panel, lags reported at band level
      only because the random-weight control shows the comb is architectural not epidemiological. Name
      the SHAP-to-IG substitution, do not hide it.
- [ ] **Fix the two conclusion sentences** at `:527` and `:543` that still say explainability is not
      delivered.
- [ ] **Add the normalisation Threats paragraph** (Gap Ledger E2). Evidence is
      `progress/outcomes/Norm_Probe_2026-09-16.md`. **Phrasing rule, see Fences:** consistent pooling
      (trained and tested pooled) helps about 6.8% on the matched panel, but MISMATCHED pooling (trunk
      trained per-node, Ebola arriving pooled, which is Ebola's real situation) hurts 16 of 16 cells,
      +4.6% to +101.5%. The doc's own verdict is "As a lever, normalisation is closed"
      (`Norm_Probe_2026-09-16.md:188`). Never write "pooling helps" unqualified.
- [ ] **Add the graph-mechanism finding.** It is absent (no hits for participation ratio, permutation,
      district-specific), and `:330` carries a superseded one-line guess. Add a compressed version from
      `Session_Audit_2026-09-10.md` section C: the encoder emits about 2 of 64 useful directions,
      relabelling the adjacency at inference costs under 1% against seed noise of 8 to 14%, so the
      trained model gains almost nothing from knowing its real neighbours. State the scope limits with
      it (single-checkpoint, not the transfer trunk; dengue one seed).

### 2.2 Sentences now false against disk

- [ ] **`:332` and `:525` say we did not run a shuffled-adjacency arm.** The inference-only relabelling
      control ran (`why_graph_fails.py` STEP 3). Only the retrain is unrun, and it is now queued (Run
      1.5). Rewrite to "we ran the inference-only relabelling control; the training-time control is [run
      4 pending / done]" depending on where Run 1.5 stands when the manuscript is finalised.
- [ ] **`:465` says "we do not yet have an explanation" for the adaptation inversion.** The
      extrapolation mechanism now exists: 76 to 85% of the adapter's effect acts 5 to 12x beyond the
      fitted range (`Adapter_Mechanism_2026-09-16.md`). It is **geometric, not causal** until Run 1.2
      lands. Rewrite to name the mechanism as geometric. (Fact-sheet cited `:467`; on disk the exact
      phrase sits at `:465`, and `:467` is the follow-on caveat paragraph. Read both.)

### 2.3 Pre-registration compliance

- [ ] **Fix the E6 label violation at `:413`.** Table 6 labels the L12 h15 row "few-shot". The
      pre-registration requires zero-shot labels at h10 and h15 on the primary arm
      (`progress/decisions/Ebola_Prereg.md:191-192`: "h10 and h15 under the primary arm are labelled
      zero-shot in every table, never few-shot"). The mechanical checker misses it because the label
      and the horizon sit in separate table columns. Change "L12 few-shot" to "L12 zero-shot" on the
      h15 row, or drop the row if it duplicates the zero-shot h15 row already at `:411`.

### 2.4 Figures and tables

Eight tables, one figure. Nine figure-ready artifacts exist on disk, unused:
`ldo3_horizon_gradient.png`, `ebola_calibration.png`, `ebola_district_ci_forest.png`,
`ebola_fewshot_vs_zeroshot.png`, `why_graph_fails.png`, `explain.png`, `shap_vs_ig.png`,
`framework_v2.png`, `baselines_vs_single.png`.

- [ ] Place the figures that carry a result the text already makes: `why_graph_fails.png` beside the
      graph-mechanism paragraph, `ebola_calibration.png` beside the calibration section,
      `explain.png` in 9.8, `framework_v2.png` in the methods. Each is a red `[ INSERT FIGURE HERE ]`
      marker under `Reports/md_to_docx.py`, so write the caption into the alt text.
- [ ] Do not add a figure whose result is not already stated. A figure is not a substitute for the claim.

### 2.5 Cuts to get under the limit

Cuttable redundancy identified on disk 2026-09-21:

- [ ] Introduction (643) + Contributions (523) + Conclusion (528) restate section 9 three times.
      Compress the overlap; keep one clean statement of each finding.
- [ ] Related Work 3.3 is 227 words for one point. Tighten.
- [ ] Section 9.4 calibration is 1,054 words with a compressible ACI aside. The ACI online adaptation
      "adds about 0.03 at short horizons and nothing long" can be one sentence.
- [ ] Section 6 is 1,616 words. Look for the deepest cut here.

### 2.6 Placeholder

- [ ] **Fix the code URL at `:547`.** It is `https://github.com/<organisation>/<repository>`. Replace
      with the real repository URL once the reproducibility package (Task 3) fixes the repo identity.

### 2.7 Verification before sign-off

- [ ] Parse the numbers back OUT of the finished manuscript and recompute each from disk, prose counts
      and significance labels included, not just tables. This has caught 6 stale numbers in a single
      brief before.
- [ ] Break one input on purpose and confirm the check fails, so the check is known to work.
- [ ] Check every Ebola sentence against the unsafe-to-claim list in
      `Reports/Phase0_to_Now_Audit.md:252-265` before sign-off.

---

## Task 3. Reproducibility package

- [ ] **Rewrite README.md.** It is a month stale. It says five panels and no COVID (`:24` area),
      marks finished Day 14/15 work as not done, contradicts itself on five vs six bundles, and calls
      UQ stubbed (`score.py` line, "UQ metrics stubbed (Week 5)"). Update to six datasets including
      COVID, mark Phase 3 complete, fix the bundle count, and remove the UQ stub language since conformal
      and ACI are delivered.
- [ ] **Add a LICENSE file.** None exists.
- [ ] **Add a CITATION file.** None exists.
- [ ] **Add an installable env spec for `ebola-train`.** The only artifact is
      `data/processed/env_train.txt`, a freeze snapshot, not an installable spec. Produce an
      `environment.yml` or equivalent so a stranger can build the env.
- [ ] **Fix the stale record counts.** `REPRODUCIBILITY.md:229` says 229 baseline records; disk has
      280 (`ls results/baselines/*.json | wc -l`). `results/lodo/` has 208 JSON records
      (`ls results/lodo/*.json | wc -l`); CLAUDE.md says 158, stale. Correct both wherever they appear.
- [ ] **Fix the repo URL / identity** so the manuscript `:547` placeholder can point somewhere real.

---

## Housekeeping (no compute, no new science)

Version-control and correctness debt found on disk 2026-09-21.

- [ ] **Commit the untracked evidence.** None of it is gitignored: `experiments/` (norm-probe script
      plus 6 JSONs and the disease-mean-floor pair), `progress/outcomes/Norm_Probe_2026-09-16.md`,
      `diagnostics/verify_norm_probe_doc.py`, `Milestone5_Report.md` and `.docx`,
      `figures/ebola_adapter_rank.png`, `figures/ebola_calibration.png`,
      `figures/ebola_district_ci_forest.png`. Also `progress/outcomes/Disease_Mean_Floor_2026-09-21.md`
      and `diagnostics/verify_disease_mean_doc.py`.
- [ ] **The three new figures have NO generator script anywhere in the tree.** Write or locate the
      generator before committing them, so a figure is not orphaned from the code that makes it.
- [ ] **Fix the `/ablation` gitignore line.** `.gitignore:31` is a bare `/ablation`, which ignores all
      210 records in `ablation/single/` while their runners are tracked. The gate-off evidence would
      vanish with the working directory. Carve the records back in the way
      `!/results/reports/*.log` already does at `.gitignore:29`, then commit the 210 records so the
      spatial-ablation evidence gains history.
- [ ] **Fix `Milestone5_Report.md:257`.** It is wrong twice: it says spatial removal "never improved
      the error in any of 60 cells". The truth is 40 error cells, and removal IMPROVED error in 8 (the
      gate hurts in 8 of 40). Fix before the docx ships.
- [ ] **Fix the E6 label at `Manuscript_v2.md:413`** (also listed under Task 2.3; do it once).
- [ ] **Clear the two failing verifiers**, both doc-lag not wrong numbers:
      `diagnostics.verify_status` (3 mismatches: misc records 31 vs 62, manuscript words 13809 vs 13834,
      tracked Reports 47 vs 57) and `diagnostics.verify_g5_scope` (a PROJECT.md quote went stale). Run
      them, update the stale doc numbers, confirm both pass.
- [ ] **Fix the three stale OPEN markers** in `progress/planning/G5_Explainability_Goals.md:93,134,142`,
      which contradict that file's own section 8 CLOSED header. Line 93 says the verifier doc is "still
      owed" (it is committed), `:134` and `:142` mark the dengue-IG and zero-baseline blockers OPEN (both
      closed). Also `Reports/Phase0_to_Now_Audit.md:262` still says "still unconfirmed with the client"
      though E1 records all three client decisions closed 2026-09-16.
- [ ] **Correct CLAUDE.md's stale numbers.** It says `results/lodo/` has 158 records; disk has 208. It
      says the manuscript is 13,809 words; disk is 13,834. (CLAUDE.md is gitignored, so this is a
      working-copy edit with no history, but the numbers steer new sessions.)

---

## Fences for whoever does this work

1. **Normalisation phrasing.** Never write "pooling helps" unqualified. Consistent pooling (trained
   and tested pooled) helps about 6.8% on the matched panel. MISMATCHED pooling, which is Ebola's real
   situation, hurts 16 of 16 cells by +4.6% to +101.5%. The lever is closed
   (`Norm_Probe_2026-09-16.md:188`).

2. **The adapter mechanism is geometric, not causal, until Run 1.2 lands.** 76 to 85% of the adapter's
   effect acting 5 to 12x beyond its fitted range is a geometric fact. It is NOT proof that the
   extrapolation is *the cause* of the error. Do not write it as the cause until the constrained-adapter
   test (Run 1.2) shows constraining the map back recovers error.

3. **The three-numbers-near-each-other fences from `todo_Milestone5.md` still apply.**
   - "Zero-shot beats few-shot in 31 of 32 cells" is point means with **no interval**. It is an
     observation, not a significance claim. The paired-by-seed test clears in only 4 of 16 cells.
   - "Unadapted wins 12 of 16, adapted wins 2 of 16" counts **significant wins against persistence**
     under `ebola_ci.py`. It is a different statistic from the 31 of 32. Never quote one to support the
     other.
   - The Ebola headline is a node-averaged **country macro** (`score.py:255-278`). The baseline table
     is **cell-pooled** (`analysis.py:71-90`). They differ 1.5x to 2.0x and must never be mixed.

4. **Never re-score the Ebola pre-registration.** `ebola_ci.py` is read-only and asserts its rebuilt
   macro against the scored JSON before printing. The Bonferroni sensitivity (Run 1.4) is a
   percentile change on the same read, not a re-score.

5. **The "roughly 8" Bonferroni count is an estimate, not a result.** Write the number only after the
   rerun produces it.

---

## Shortest path to closing Milestone 6

1. **Kick off Run 1.1** (classical baselines) once `run_classical.py` is written. User's shell, one to
   two hours. It closes an open client order and it unblocks nothing else, so start it first.
2. **While it runs, do the manuscript** (Task 2): write 9.8, add the Threats and mechanism paragraphs,
   fix the false sentences and the E6 label, then cut to the limit. This is the largest single piece of
   work and needs no compute.
3. **Run 1.2 and 1.3** (constrained adapter, input-vs-representation) plus the Bonferroni add-on (1.4).
   Minutes each. Fold their results into the manuscript.
4. **Do the reproducibility package and housekeeping** (Task 3 and Housekeeping): README, LICENSE,
   CITATION, env spec, stale counts, gitignore fix, commit the untracked evidence, clear the failing
   verifiers.
5. **Only after 1.1 to 1.3 land, run 1.5** (the D2 shuffled-adjacency retrain, 6 hours, user's shell).
   Update the two manuscript sentences and the ledger once it returns.
6. **Final verification pass** on the manuscript: recompute every number from disk, break one input to
   confirm the check works, check every Ebola sentence against the unsafe-to-claim list. Then render
   the docx and fix the repo URL.

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
| 1 | Ablations and robustness / sensitivity analysis | **PARTIAL.** Gate-off and epi-informed ablations already ran (`ablation/single/`, 210 records). Four confirmatory runs queued 2026-09-21, none run yet. **Epi bound and lambda sweep DONE 2026-09-28** (`a0052b9`): no PASS in 18 units, Japan h10 hurt, see 1.6 |
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

**Progress 2026-09-22. Script built and self-checked. The five-seed run is with the user.**

Script `experiments/adapter_constraint.py` (EXPLORATORY, stamped `protocol="EXPLORATORY"`, an assert
confines writes to `experiments/`, `load_manifest(verify=True)` checks the frozen arm hashes before and
after, nothing under `results/` or `data/` is touched, the pre-registration is never re-scored). It
follows the `experiments/norm_probe.py` pattern and reuses its plumbing.

**The instrument, and why it is shrinkage rather than a literal ridge.** With the trunk frozen the
adapter is exactly affine (`models/adapters.py:14-16`): `pred = h A^T + c`. The zero-shot borrowed-mean
adapter is a second affine map `(A_z, c_z)`, the one that adds no Ebola-specific change, so the
few-shot-specific change the mechanism doc calls D is exactly `A_few - A_z`. A ridge penalty
`lam * ||A - A_z||^2` shrinks the fitted map along the segment to `A_z`, which in closed form is
`A(t) = A_z + t (A_few - A_z)` with `t = 1 / (1 + lam*k)`. So `t` is a reparametrisation of the ridge
strength: `t=1` is unconstrained few-shot (D untouched), `t=0` is the zero-effect map (D fully removed),
`t<1` shrinks D by the factor t. I use the closed-form shrinkage because a literal ridge run through the
pre-registration's own optimiser SATURATES: with the gradient norm clipped to 1.0, cranking lam only
shrinks D by about 64% of the way to `A_z` and never reaches it (measured on seed 42: relative distance
to `A_z` falls from 1.914 at lam=0 to a floor of 0.688, and the error effect is non-monotone), so a
literal-ridge grid cannot span the few-to-zero range and cannot cleanly reference full recovery.
Shrinkage spans it exactly and its two ends are the archived arms.

**No Ebola query outcome touches any fitting choice.** The shrinkage strength `t` is selected on SUPPORT
ONLY by leave-one-district-out CV inside the support set, the machinery the pre-registration uses to
pick the adapter epoch (`train.ebola.choose_epochs` / `_fit_adapter`). Per held-out district I refit the
few-shot adapter on the remaining support at that arm-and-seed's pre-registered epoch count, interpolate
that fold map toward `A_z` over `t` in {0.0, 0.1, ..., 1.0}, and score the held-out district's pinball.
Curves are pooled weighted by held-out cells and the `t` minimising the pooled held-out pinball is
selected. The REPORTED cell is the support-selected `t`; the full grid is context, and a query-oracle
best-over-grid is reported too, labelled as context that peeks at the answer and is never a selection.

**A second, parameter-free instrument (the clip).** Restrict the few-shot change to the per-horizon
support span: `A_clip = A_z + (A_few - A_z) @ P_h` per horizon block, where `P_h` projects onto the span
of the support feature rows the fit consumed at horizon h (`diagnostics.adapter_mechanism.design_rows`,
the same rows the mechanism doc measured). This deletes exactly the part of D acting on directions the
support never constrained. A horizon with 0 support rows (L12 h15) has an empty span, so its whole block
collapses to the zero-effect map. Done per horizon on purpose: pooling all origins spans the full 64
dims and the clip is a no-op.

**Verdict framework (one of three honest forms, with numbers).** `recovery = (few - constrained) /
(few - zero)` at the support-selected `t`, seed-mean country-macro MAE. Extrapolation is causal if
constraining recovers most of the damage, partially causal if it recovers some, not supported if it
recovers none. `recovery_clip` reports the same fraction for the parameter-free clip. Never "THE cause";
this is one mechanism test.

**Validated (selfcheck passes).** The adapter is exactly affine and the affine reconstruction round-trips
to deviation 0.0; `t=1` reproduces the archived few-shot MAE exactly and `t=0` reproduces the archived
zero-shot MAE exactly on both arms at seed 42; the frozen arm hashes are intact before and after. Cost:
seed 42 both arms about 110s (L12 ~30s, L20 ~79s), so about 2 minutes per seed, well under the 10-minute
per-step budget; five seeds about 10 minutes total.

**Early single-seed signal, NOT the result.** Seed 42 alone selected `t=0.2` on L12 (strong shrinkage
toward the zero-effect map) and `t=0.8` on L20 (mild). This is one seed and carries no verdict; the
reported result needs all five seeds paired.

**DONE 2026-09-22. Verdict: PARTIALLY CAUSAL, arm-dependent.** Constraining the adapter toward the
zero-effect map, at a strength chosen on support alone, recovers about 45% of the few-shot damage over
the 7 damaged cells: 77 to 90% on the primary L12 arm at h3/h5/h10 (support-selected t about 0.34) and 9
to 19% on the secondary L20 arm (support-selected t about 0.88). Removing the whole few-shot change (t=0,
the oracle end) recovers about 100% on every damaged cell, so the damage lives in the few-shot-specific
change, of which the extrapolation is the bulk. The parameter-free strictly-out-of-span clip recovers
almost nothing (+1% on average), which locates the damage in under-constrained in-span directions rather
than strictly-out-of-span ones. This is one mechanism test and never "THE cause".

- [x] Script `experiments/adapter_constraint.py` written, EXPLORATORY, selfcheck passes, both anchors
      exact, hashes intact
- [x] **User ran the five-seed sweep**
      (`conda run -n ebola-train python -m experiments.adapter_constraint --seeds 42 52 62 72 82`);
      `experiments/adapter_constraint__seed{42,52,62,72,82}.json` and `__summary.json` on disk, complete
      (2 arms, 5 seeds, full 11-point t-grid), anchors t=1/t=0 match the archived arms to 6e-06
- [x] Wrote `progress/outcomes/Adapter_Constraint_2026-09-22.md` in the `Norm_Probe_2026-09-16.md` style,
      with the paired numbers, the support-selected `t` per arm, the causal verdict, and its limits
- [x] Wrote `diagnostics/verify_adapter_constraint_doc.py`; it passes on the clean doc and its `--mutate`
      sweep catches all 11 planted defects
- [x] Reported the verdict above: pulling the adapter toward the zero-effect map recovers the damage
      partly, more on L12 than L20, which is the causal step the geometric measurement could not take
      alone

**Untracked, awaiting the Housekeeping commit:** `experiments/adapter_constraint.py`, the six sweep
JSONs, `progress/outcomes/Adapter_Constraint_2026-09-22.md`, `diagnostics/verify_adapter_constraint_doc.py`.
None are gitignored.

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

### 1.4 Bonferroni multiplicity sensitivity (free add-on) -- DONE 2026-09-28

- [x] `ebola_ci.py` gained `--bonferroni M` and `--floor`. Default is off, so every pre-existing
      number is unchanged and the uncorrected run is bit-identical to
      `results/reports/ebola_district_ci.log`
- [x] The canonical family is **persistence only**, RMSE and MAE, both arms, both regimes, four
      horizons, 32 cells, 14 clearing zero at 95 percent (`Reports/Phase0_to_Now_Audit.md:257`).
      The divisor is **32**, not 16
- [x] Control at divisor 16 reproduced the documented 7 of 14 (5 RMSE, 2 MAE). Divisor 32 gives the
      same 7 of 14, the same seven cells. The "roughly 8" Gaussian estimate is retired
- [x] Read-only, no re-scoring. Logs: `results/reports/ebola_district_ci_bonferroni32.log` and
      `..._bonferroni16.log`. Replacement sentence in
      `progress/outcomes/Manuscript_Correctness_Pass_2026-09-23.md` EDIT 6

**What it buys the paper.** The manuscript already controls multiplicity on the confirmatory family
only (`Reports/Manuscript_v2.md:521`). A named Bonferroni sensitivity on the wider exploratory grid
answers the reviewer who asks what happens when you correct the whole family.

### 1.5 D2 shuffled-adjacency RETRAIN, gated on 1.1 to 1.3 -- DONE 2026-09-23

**This reopened a decided-no item.** `progress/planning/Gap_Ledger.md:276` records D2 as "Do not run"
(recorded as decision-log D21). The user reopened it on 2026-09-21 for the final close, because it is
the last remaining spatial test. The gate-off ablation and the inference-only relabelling control both
already ran; this is the retrain version, which asks whether a model *trained* on a fake graph would
do as well, and it is the stronger of the two.

- [x] Ran overnight in the user's shell, finished 2026-09-23. 25 records (5 panels x 5 seeds, dengue
      included), `ablation/single/encoder__<ds>__seed<S>__shufadj.json`.
- [x] Retrained on relabelled adjacency (same topology, same degrees, wrong districts), compared error
      to the real-graph trunk in `results/single/`, paired per seed.
- [x] Scored from disk and cross-checked independently. **Verdict over 60 cells: 9 the real graph
      helps, 2 the real graph hurts, 49 within noise.** The real map earns a small, seed-stable
      advantage almost entirely on dengue (correlation at all four horizons, error at h10 and h15, 2 to
      4 percent). The four small panels are within noise on error. Full write-up:
      `progress/outcomes/Shuffled_Adjacency_2026-09-23.md`, verifier
      `diagnostics/verify_shufadj_doc.py` (mutation-tested, 4 of 4 caught).
- [x] Ledger D2 updated below to RUN with the verdict and date.
- [ ] Update the two manuscript sentences (`Reports/Manuscript_v2.md:332`, `:525`) that currently say
      we did not run a shuffled-adjacency arm. Replacement text drafted in the outcome doc scoring
      handoff; not yet applied to the manuscript.

**What it bought the paper.** It converts "we did not run a shuffled-adjacency arm"
(`Reports/Manuscript_v2.md:332`, `:525`) from a disclosed gap into a closed test. The result does NOT
support the strongest form ("the graph contributed nothing at any stage"): dengue shows a small
seed-stable advantage for the real map. It DOES support the sharper, honest form that lines up with
gate-off: the graph is about shape not magnitude, and even a from-scratch retrain on the real map only
buys a small correlation advantage plus a few percent on dengue's long horizons.

### 1.6 Close-by-wording items (no compute)

These are robustness points the reviewer will want that the artifacts already answer. They need a
sentence, not a run.

- [ ] **Rolling-origin evaluation.** Ebola scoring already aggregates 18 origins x 3 countries, and dev
      panels 37 to 589 origins, in the `__perorigin.npz` archives. State it in one sentence.
- [ ] **Multiplicity.** The manuscript policy at `:521` plus the Bonferroni sensitivity from 1.4.
- [ ] **Block bootstrap.** The `:519` disclosure stands. Ebola's 18 origins make blocks weak there, so
      say the block-bootstrap sensitivity was checked and leaves the long-horizon conclusions intact
      rather than applying it throughout.
- [x] **Epi bound and lambda sweep, run 2026-09-28.** Pre-registered (`b24063f`), 90 cells, result
      `progress/outcomes/Epi_Bound_Lambda_2026-09-28.md` with verifier `diagnostics/verify_epi_bound_doc.py`
      (`a0052b9`). No PASS in 18 units; Japan h10 significantly worse in all four new Japan arms; US
      panels never reach the 1 percent gate. Records in `experiments/epi_bound_lambda/single/`.
- [ ] **Rewrite `Manuscript_v2.md:497`** from the result above (plan section 5A). It is no longer a
      null: say it did not help where it had weight and hurt Japan h10. Handled in the manuscript session.
- [ ] **Epi-ablation scope.** The epidemiology-informed ablation ran on 4 of 5 panels. State that
      dengue at 7,165 nodes was excluded for cost, so a reader knows the result is on the four small
      panels. Per the protocol, dengue is not run because there was no PASS.
- [ ] **Small, not blocking.** `ablation/run_epi_ablation.py --report` still applies the old
      `|mean| >= sd` rule and has no completeness check. The result doc discloses it. Fix only if the
      test is ever rerun.

### 1.7 Declined, with recorded reason

Listed so they do not leak back in.

- **Aggregator swap** (mean to sum/max/spread). Declined: it is a retrain, not an inference tweak, and
  `t4` measures information *available*, not information that *helps* forecasting. High-variance is not
  the same as useful.
- **L20-no-blackout counterfactual.** Declined: it needs a support window the frozen pre-registration
  does not contain (`Adapter_Mechanism_2026-09-16.md:125-126`).
- **D1** (LDO3 zero-shot quantiles, ~10 h) and **D3** (median-to-mean into the Ebola path) stay closed
  (`Gap_Ledger.md:275,277`).

### 1.8 Run 5: auxiliary district-identity objective (added 2026-09-23)

Rationale in `progress/planning/Milestone6_Plan.md` section 4A. Two stages with a hard gate between
them. The idea is a second encoder head that classifies "which district is this?" on the
representation `h`, trained with weight `lambda_aux` alongside the forecast loss, meant to stop the
encoder squashing district-specific structure (input 14 to 94 percent district-specific energy
compressed to 3.8 to 22.4 percent in `h`, `Input_Energy_2026-09-22.md`). **User's stated risk:** if
the squashed component was noise, forcing it back in makes forecasts worse.

**Stage 1: the gate probe (cheap, I run it). DONE 2026-09-24. Verdict: NO-GO.**

- [x] Question the probe answers: is the discarded district information forecast-relevant at all?
- [x] Instrument: on each frozen `results/single/` checkpoint, fit per-node corrections (a per-node
      additive bias, and a per-node affine slope+intercept) on the model's own residuals, then apply
      them on the test period. To separate real district signal from the known lognormal
      median-vs-mean gap (`loop._fit_bias_correction` already fixes that with ONE global scalar per
      horizon), every per-node correction is contrasted against a district-AGNOSTIC (pooled)
      correction of the SAME form. The decisive contrast is a node variant beating BOTH baseline AND
      the matching global variant, because beating global alone can just mean overfitting less than a
      global correction that is itself worse than doing nothing.
- [x] **Leakage rule:** corrections fit ONLY on train-phase origins and train-phase observed target
      cells; test origins never touched during fitting; checkpoint frozen, no retrain, no gradient. A
      node needs at least 8 observed train target cells to earn a correction, else it keeps baseline.
- [x] Metrics: country-macro RMSE and MAE (`score.score_bundle`). dengue uses `origins[::4]`.
- [x] Verdict rule, same as gate-off / shufadj (sample sd, paired by seed).
- [x] Script `diagnostics/graph_probe/aux_gate_probe.py`, output `results/misc/aux_gate_probe.json`,
      write-up `progress/outcomes/AuxGate_Probe_2026-09-23.md`, verifier
      `diagnostics/verify_auxgate_doc.py` (passes, mutation-tested 3 of 3).

**THE GATE: NO-GO.** The decisive node-beats-BOTH-baseline-and-global contrast fires in 2 of 40 bias
cells (both covid_us-states at h10: node vs baseline +9.95% RMSE / +9.48% MAE, node vs global +3.30% /
+3.72%) and 0 of 36 affine cells. The 18-of-40 node-beats-global count is an artifact (on dengue and
us-regions the global correction is worse than baseline, so node beating it just means less overfit;
baseline wins). The richer affine_node instrument, the true analogue of an aux head, never beats
baseline, hurts on 8 cells, and blows up numerically on dengue (up to 3.07e7 error) on quiet z-scored
nodes, which is the user's stated risk made concrete. The one real district gain (covid h10) is a
per-district level offset achievable for free post-hoc, not a reason to retrain. Borderline defaults to
NO-GO. The user may overrule.

- [x] Gate outcome recorded in this doc and the plan doc (`Milestone6_Plan.md` section 4A).
- Separate observation, not a decision: a post-hoc recalibration helps COVID (global level offset
  +6.88% RMSE / +5.98% MAE at h10, clears noise; short horizons positive but not seed-stable). Worth a
  cheap post-hoc per-node/global recalibration follow-up on COVID (the closest Ebola analogue), NOT the
  aux objective. Limits in the outcome doc.

**Stage 2: the overnight retrain. NOT WRITTEN (gate is NO-GO).** The runner
`ablation/run_aux_district.py` and the `train/loop.py` aux hook are not written, per the gate rule. If
the user overrules the NO-GO, the design is: aux head + loss weighted `lambda_aux`, runner mirroring
`ablation/run_shuffle_adjacency.py` with a `--smoke`, schema assert, resumable, paired vs
`results/single/`. Runtime estimate for the full run: small panels ~45 min each, dengue ~12.7 h.

### 1.9 V2 deviation channel: mechanism test -- DONE 2026-09-25. Verdict: pre-registered FAIL

**This closes the spatial-redesign question for this paper.** The aux-gate probe (1.8) said the
discarded district signal is not forecast-relevant post-hoc. The V2 test asked the sharper version by
retraining: route the district deviations directly to a learned spatial branch and see if the real map
then earns accuracy. It was run under a hash-frozen pre-registration
(`progress/decisions/V2_Deviation_Protocol.md`, sha256 `82069f41...`, committed at `39ecf34` before any
COVID number), scored once from disk by `ablation/run_v2_deviation.py --report`.

- [x] Two-stage design. Stage 1 (5 seeds, validation only) measured the paired noise s = 0.0412 and
      the formula set **N = 15 seeds** (power 0.822 for the probe-implied 3.3 percent h3 RMSE effect).
      Stage 2 trained all four arms (v1ref plus v2graph, v2nograph, v2shuffled) at those 15 seeds on
      covid_us-states.
- [x] **Verdict: FAIL, criterion (a) not met.** v2graph is significantly worse than v1 at 7 of 8
      decisive RMSE/MAE cells and within noise at the 8th (MAE h15); better at none. Short-horizon
      error roughly doubles (h3 RMSE +96.8 percent, h5 +137.6 percent). PCC h3 collapses 0.436 to
      -0.014.
- [x] **Attribution: the deviation input itself causes the damage, the graph adds nothing.** v2nograph
      and v2shuffled degrade nearly identically to v2graph, and every v2graph-vs-v2nograph and
      v2graph-vs-v2shuffled comparison is within noise on RMSE and MAE at every horizon.
- [x] Per protocol criterion (c): **no dengue run** (dengue was gated behind a COVID PASS). There will
      not be one under this protocol.
- [x] Write-up `progress/outcomes/V2_Deviation_Result_2026-09-25.md`, verifier
      `diagnostics/verify_v2_result_doc.py` (passes clean, mutation-tested 11 of 11).
- [x] Read-only val-vs-test check: on the **validation** split the three v2 arms BEAT v1 by 10 to 20
      percent at h3/h5, while on **test** they lose by 73 to 138 percent. Validation did not warn, which
      favours a train-to-test regime shift over a val-blind overfit but does not settle the mechanism.
- [ ] Manuscript gains one paragraph closing the graph-failure mechanism chain (drafted in
      `Milestone6_Plan.md` section 4B, fenced against overclaiming causation). Not applied to the
      manuscript yet.

**Fences kept.** The FAIL row licenses "no effect found at N seeds"; what we saw is stronger (the
channel measurably hurt), so we may say that, but NOT "the deviation channel has no effect" or "the
graph is useless in general". The linear-probe-gains-5.3-percent versus learned-branch-doubles-error gap
is CONSISTENT WITH the deviations being noise-dominated at learned-model capacity (the pre-stated risk),
but the exact failure mechanism (short-train overfit, COVID regime shift, or optimisation interference)
is NOT established. Do not write any one as the reason.

### 1.10 Pure-TCN and fresh gate-off -- DONE 2026-09-28

**EXPLORATORY, not pre-registered, user-ordered 2026-09-25.** The gate-off ablation (Table 3) sets
g = 0 but keeps the LTR degree feature (`models/encoder.py:60`), so it bounds neighbour information,
not the whole graph. This run removes the degree feature too, on the two cheapest panels, COVID and
influenza_japan.

- [x] Pure-TCN arm `experiments/pure_tcn.py`: the gate-off trunk with LTR swapped for a
      parameter-free zero module through the `encoder_factory` hook, so `models/encoder.py` and
      `train/loop.py` (both in the V2 protocol code hash) stay untouched. Its selfcheck proves the
      output ignores the graph and the starting weights match gate-off; rerun 2026-09-29, passes.
- [x] User ran it in their own shell: COVID five seeds 2026-09-25, Japan five seeds 2026-09-28.
      Records only, `experiments/pure_tcn__{panel}__seed{S}.json`.
- [x] Fresh gate-off `experiments/gateoff_fresh.py`: the exact `ablation/run_gate_ablation.py:153`
      call, written to `experiments/`, so the untracked archived gate-off records were never exposed
      to `--force`. Ran 2026-09-28. It reproduces the archived records bit for bit: 60 of 60 values
      per panel, 140 of 140 across all seven metrics.
- [x] **Verdict: removing the whole graph moves 0 of 16 error cells beyond seed noise.** COVID
      1 / 0 / 11 (one PCC flag, about what chance produces under this rule), Japan 0 / 0 / 12. At h5
      on both panels mixing raises error and the degree feature lowers it by about the same amount
      (degree test 2 / 0 / 10 per panel, all 20 per-seed h5 differences positive), so the two cancel.
      Cause of the h5-only pattern unknown.
- [x] Code and records committed `e59b231`. Write-up `progress/outcomes/Pure_TCN_Graph_Removal_2026-09-28.md`,
      verifier `diagnostics/verify_pure_tcn_doc.py` (passes clean, mutation-tested 36 of 36). Both
      untracked until committed.
- The manuscript sentence this changes is tracked in 2.2 below.

**Fences kept.** Two panels only, and they are the ones where mixing never helped correlation, so
"message passing buys shape" is untouched. Five seeds and the |mean| < sd rule, which flags about 9
percent of cells by chance; no significance claim. The split runs along one path only (mixing
measured with the degree feature present, the degree feature with mixing absent). No mechanism for h5.

### 1.11 Downward adapter capacity -- DONE 2026-09-30. No smaller surface wins

- [x] Protocol `progress/decisions/Capacity_Down_Protocol.md`, verifier
      `diagnostics/verify_capacity_down.py` (152 checks, 33 of 33 mutations caught) and runner
      `diagnostics/capacity_probe.py --down`, committed `ed32765` before the run.
- [x] Step 0: r* = ceil(9.017) = 10, set by dengue; flu maximum 3.817.
      `results/misc/capacity_probe_rank.json`.
- [x] User ran five seeds 2026-09-29 to 2026-09-30, about 8.0 h total.
- [x] Verdicts: rank-10 COSTS, head-only SAME, recal-int COSTS (L12, L20), recal-budget COSTS
      (L12, L20), shrink-t COSTS (L12) and NULL (L20). Upward half re-read: all three MIXED.
      Frozen t* 0.1 (L12), 0.2 (L20).
- [x] shrink-t significantly better than the fresh affine on us-states (L12 h3/h5, L20 all four
      horizons), never on japan. recal-budget L20 japan blows up in counts (seed 82 h3 RMSE
      2,623,157).
- [x] The fitted affine is worse than the unchanged anchor on us-states and better on japan, so
      the damage pattern reproduced on one panel only.
- [x] Records, report and log committed with this update. Plan section 4D carries the numbers.

**Fences kept.** Dengue-to-flu only, single-disease anchor, per-node scaling in the simulation,
seed-paired t at n=5. Pre-registration untouched.

### 1.12 Model-space rescore of archived predictions -- DONE 2026-09-30. EXPLORATORY

- [x] `diagnostics/model_space_rescore.py` (selfcheck: round trip exact including counts in
      (-1, 0); a planted clip at 0 breaks it). 70 archives, every family reproduced its scored
      count-space numbers to 1e-6 before any model-space number was kept. 72.8 s.
- [x] Single-disease vs persistence: RMSE 13/4/3 in counts, 16/2/2 in model space; MAE 14/2/4 and
      15/2/3.
- [x] LDO3 adapted vs single, 36 cells: 0/19/17 in counts, 2/12/22 in model space (seed-paired t,
      not the documented bootstrap instrument).
- [x] Ebola: zero-shot significantly better than few-shot in 4 of 16 cells in counts, 10 of 16 in
      model space. Against persistence in model space, zero-shot RMSE 6/1/1 and MAE 1/3/4, few-shot
      RMSE 0/3/5 and MAE 0/2/6 (win/noise/loss). Zero-case cells (22.5 to 28.5%) carry the
      difference; on cells with cases the encoders beat persistence.
- [x] Not rescored: LDO3 zero-shot, joint, ldo3full, capacity probe, baselines.
- [x] Output `results/misc/model_space_rescore.json`, committed with this update. Plan section 4E.

**Fences kept.** Seed-noise intervals only, not the `ebola_ci.py` district interval, not a
pre-registered result. Ebola's pooled scale is not comparable with the dev panels.

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
- [ ] **End the graph-mechanism chain with the V2 result** (`progress/outcomes/V2_Deviation_Result_2026-09-25.md`).
      One paragraph: the obvious remedy (route district deviations to a learned spatial branch) was
      tested under a pre-registered protocol on COVID and roughly doubled the short-horizon error, with
      a wrong map and no map doing equal damage, so the harm came from the deviation input, not the
      graph. Drafted text and fences in `Milestone6_Plan.md` section 4B. **Do not claim causation:** it
      is CONSISTENT WITH the deviations being noise-dominated at learned-model capacity, but the exact
      failure mechanism is not established. Do not write "the compression is a defence against
      noise-dominated deviations" as a fact.

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
- [ ] **`:326` says removing "the graph" improves influenza-Japan RMSE by 45.3 at h3 and 61.7 at
      h5.** Those are gate-off numbers, so what was removed is neighbour MIXING. With the whole graph
      removed (pure-TCN, 1.10) Japan moves -52.6 (sd 114.7) at h3 and -2.9 (sd 53.9) at h5, both within
      noise. Change the noun, and replace the untested "seasonal phase but not a baseline level"
      explanation with the measured result. Drafted text, +8 words net, in
      `progress/outcomes/Pure_TCN_Graph_Removal_2026-09-28.md` ("What it means for the manuscript").
      Narrow the first sentence of `:332` (the degree-feature limit) in the same pass as EDIT 1a.

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

5. **The "roughly 8" Bonferroni count was an estimate and is now retired.** The rerun produced
   **7 of 14** at divisor 32 over the persistence family, both metrics (2026-09-28). Never quote a
   survivor count without naming the floor family in the same sentence.

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




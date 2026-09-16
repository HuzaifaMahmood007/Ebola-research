# Milestone 5 todo

**Scope.** Week 5 of `progress/planning/Final Internal Project Brief.md`, days 21 to 25: the Ebola
case study, calibrated uncertainty, and explainability. Nothing outside that week is listed here.

**Position, updated 2026-09-16 after the three client decisions closed.** The compute for this
milestone is finished and nothing is queued. **Tasks 1, 2 and 3 are all DONE.** The client decisions
closed on 2026-09-16 against a measured evidence pack, so G5 is signed off and Gap Ledger E1 reads
DONE. **One item is left in the whole milestone: the report, which does not exist.** Status below is
checked against disk, not against the progress docs.

Added 2026-09-15: section 1a now records **why** few-shot adaptation hurts on Ebola, which the
milestone previously had no answer for. The answer is partial and bounded, and the planned follow-up
run was cancelled on evidence rather than deferred. Read 1a before writing the task 4 report.

Brief tasks, and where each one actually stands:

| # | brief task | status |
|---|---|---|
| 1 | Run the Ebola out-of-distribution few-shot case study | **DONE** |
| 2 | Calibrated uncertainty plus calibration checks | **DONE** |
| 3 | Explainability outputs, SHAP global and local | **DONE 2026-09-16.** Delivered as integrated gradients with an occlusion cross-check; substitution accepted by the client on measured evidence, SHAP row retracted, neighbour figure accepted with its limitation |
| 4 | Compile case-study results with prediction intervals | **PARTIAL**, numbers exist, no milestone report |

---

## 1. Ebola case study, task 1. Done, do not re-run

- [x] 20 scored records plus 20 quantile archives in `results/ebola/`, both arms (L12 primary, L20
      secondary), both regimes (few-shot, zero-shot), 5 seeds
- [x] Scored ONCE under the hash-frozen pre-registration, `progress/decisions/Ebola_Prereg.md`
- [x] Estimand mismatch corrected in `ebola_ci.py`, so the point and the interval are finally the
      same statistic

**Nothing to do here.** Re-scoring would break the pre-registration. The only open item is how the
result is written up, which is task 4 below.

### 1a. WHY few-shot adaptation hurts. Two diagnostics run 2026-09-15

The case study is scored and closed, but until today the milestone had **no explanation** for its
central negative result, and the report has to say something about it. The parameter-counting story
was withdrawn on 2026-09-10 and nothing replaced it. Both diagnostics below are inference-only, read
no Ebola label, score nothing, and cost no GPU time.

- [x] **Step 0, the feature-shift measurement.** `diagnostics/ebola_feature_shift.py`, written up in
      `progress/outcomes/Ebola_Feature_Shift_2026-09-15.md`, machine-checked by
      `diagnostics/verify_feature_shift_doc.py` (mutation-swept, 9 targets, no blind spots).
      `ebola_L12`'s support mask **is** `M[:, :13]`, so the adapter fits on windows that are 72.5%
      zero padding at 5.2% observation and is then scored on full windows at 31.6%. Its affine design
      matrix is rank deficient at every horizon on the primary arm, 45 / 37 / 17 / 0 of 64 directions,
      so between 19 and 64 directions are set by the random initialisation rather than by data. At h15,
      with zero training rows, the adapted forecast still moves 5.13 zero-shot sd
- [x] **Step 1, the dev-fold rehearsal. Attempted, and it cannot be built.**
      `diagnostics/fewshot_sim.py`, written up in `progress/outcomes/Fewshot_Init_Probe_2026-09-15.md`,
      machine-checked by `diagnostics/verify_fewshot_sim_doc.py` (mutation-swept, 6 targets, no blind
      spots). Transplanting Ebola's exact support mask onto `covid_us-states`, with its sparsity,
      padding and 48/38/18/0 pair profile reproduced by construction, made adaptation **help**
      (7829.5 against zero-shot 12402.5). Checking the archived 5-seed records, zero-shot beats the
      adapted arm in **4 of 100** dev-fold cells. **No development fold fails the Ebola way**, so none
      can rehearse or repair it
- [x] **Step 2 cancelled.** It was gated on Step 1 not coming back degenerate. It did, so the ~14
      trunk retrains at 9 to 10 hours are not being run and nothing is queued

**What this buys the report.** A dev fold was given Ebola's sample size, sparsity, padded windows and
exact per-horizon pair profile and adaptation still helped, so **none of those is a sufficient cause**
of the Ebola damage. That closes off the two explanations a reviewer reaches for first, "59 labels is
too few" and "the windows are mostly padding". State it as a bounded negative, not as a solution: we
now know what it is not, and Step 0 shows the degeneracy is real but not decisive.

- [ ] **Next, and it is now the prime suspect: the normalisation mismatch.** The largest remaining
      difference between Ebola and every dev fold is `CLAUDE.md` section 6. The matched experiment,
      trunk trained per-node and tested pooled, has never been run.
      `progress/planning/Gap_Ledger.md:321-324` files it under "runs, none of them recommended", which
      should be revisited in light of this result. **Not started, and it is Week 6 work, not Milestone 5**

## 2. Uncertainty, task 2. Done

- [x] Conformal and ACI in `conformal.py`, fit and apply logs in `Reports/conformal_*.log`
- [x] The frozen cross-disease correction reads no Ebola outcome at all and lifts coverage from
      0.28-0.70 to 0.65-0.98. Online adaptation adds about 0.03 at short horizons and nothing long
- [x] District-level intervals rebuilt from `*__pernode.npz` in `ebola_ci.py`, read-only, asserts
      against the scored JSON before printing

**Nothing to do here either.** This is the strongest result in the milestone and it is already
machine-verified.

## 3. Explainability, task 3. DONE 2026-09-16

Gap Ledger row E1 reads **DONE**. All four of its blockers are closed: the verifier doc is committed,
the dengue completeness error turned out to be quadrature rather than a defect, the zero baseline was
replaced and every development panel rerun on the fix, and the three client decisions closed on
2026-09-16. Commits `b9783b5`, `17aedaa`, `b1d66f6`, `4aab8e6`, plus today's.

**The one that mattered turned out not to be a bug.** Dengue's completeness error was quadrature,
not mis-attribution, and it closed by running the integrator properly rather than by fixing
anything. Two generated reports were also found stale against their own archives on the way.

- [x] Blocker (c), the zero IG baseline. `ig_baseline()` and `baseline_mu()` in `explain.py`
- [x] Verifier document written and now committed, `progress/outcomes/G5_Explainability_Results.md`
- [x] **Ledger E1 and STATUS corrected.** Both said blocker (c) was unfixed and blocker (b) was
      0.0444, days after neither was true
- [x] **Regenerated `Reports/explaiability_report.md`**, byte-identical to the archives again. The
      same trap had also caught `results/reports/explain_report.txt`, which printed dengue at 64
      steps while the archives held 128. Two generated reports drifted from their own inputs inside
      five days
- [x] **Blocker (b), dengue IG completeness. Closed, and it was never a defect.** 0.0444 at 32
      steps, 0.0155 at 64, **0.0040 at 128**: 2.9x then 3.9x per doubling, against the 4x the
      midpoint Riemann rule predicts. Dengue was under-integrated, not mis-attributed, which is the
      panel where that would show at 7,165 nodes. At 128 steps it sits beside influenza_japan's
      0.0021
- [x] **Baseline split closed.** All four remaining development panels rerun at 128 steps on the
      fixed baseline. `mean |baseline_mu|` reads **exactly 0.0** on all four, so the no-op is now
      measured from the archives rather than argued from the scaler, and the entire report diff is
      four lines with only the step, err and gap columns moving. Both Ebola arms stay at 32 steps
      deliberately, since that is what their reported numbers were produced under
- [x] **Fixed `Reports/Phase0_to_Now_Audit.md:260`**, which claimed no attribution code exists
      anywhere in the project. The sentence stays on the unsafe list, because we have integrated
      gradients and not SHAP, but the stated reason is now the true one. Finding M12 also carries a
      half-closed note
- [x] **Gate figure placed.** `figures/gate.png` is Figure 1 in `Reports/Manuscript_v2.md` §9.1,
      replacing Table 3; tables renumbered 3 to 8. `Reports/md_to_docx.py` now renders a whole-line
      image as a red highlighted `[ INSERT FIGURE HERE: <path> ]` marker with the caption beneath,
      since figures are pasted into Word by hand here. Verified on the real manuscript, not just the
      self-check. Costs 25 words against the limit rather than saving any

### 3a. The three client decisions. ALL CLOSED 2026-09-16

Closed against `Reports/G5_Method_Decision_Brief.md`, which gave the client measured grounds rather
than our assurance. Two new figures and three new scripts were built for it.

- [x] **Decision 1, integrated gradients instead of SHAP. ACCEPTED.** Carried on two measured
      grounds. **Compute:** one global read costs SHAP 89,948,160 model evaluations against IG's
      5,760 on ebola_L12, and 14,086,963,200 against 7,680 on dengue, ratios of 15,616x and
      1,834,240x, because SHAP perturbs each district separately while IG returns every district in
      one sweep. **Agreement:** on the five busiest Ebola districts both methods pick the same top
      channel 5 of 5, |phi| correlation 0.963 to 0.990. Also on record: SHAP is a sampled estimator
      that reruns to r = 0.68 at 256 coalitions where IG reruns to r = 1.000000, and 41.6% to 44.9%
      of its coalitions contain an impossible week. Evidence
      `diagnostics/graph_probe/shap_vs_ig.py`, `figures/shap_vs_ig.png`
- [x] **Decision 2, retract the SHAP row. APPROVED AND DONE the same day.** Corrected copy at
      `Reports/Phase1/RelatedWork_CompetitiveAnalysis_Benchmark_corrected_2026-09-16.docx`, original
      untouched. Exactly one cell moved, table 0 row 11 ("Ours") column 7, from
      "yes - SHAP (glob.+loc.)" to "yes - integrated gradients + occlusion (glob.+loc.)". Done by
      `diagnostics/retract_shap_row.py`, which refuses to run on an unexpected row and re-reads the
      saved file to confirm nothing else changed
- [x] **Decision 3, neighbour figure accepted without an accuracy claim, and the MECHANISM
      delivered.** The client accepted that the graph does not improve accuracy and asked for the
      reason behind the 0-of-40 result, not just the result. Measured in
      `diagnostics/graph_probe/why_graph_fails.py`, `figures/why_graph_fails.png`, three steps: the
      gate is wide open (0.271 to 0.375, 0.0% of districts near closed, so the model is not ignoring
      the graph); only 3.8% to 12.3% of a district's summary is district-specific and it uses about
      2 of 64 directions; neighbour averaging destroys a further 14% to 60% of that; so relabelling
      the adjacency costs only +0.03% to +1.14% against seed noise of 0.7% to 4.8%. The brief also
      records what we TRIED: the learned gate the model could have closed and did not, two spatial
      layers rather than three, an epidemiology-informed training penalty at two strictness bounds
      (2 better, 1 worse, 45 within noise at p90max; 0 better, 0 worse, 48 within noise at p99max),
      and a measurement of alternative aggregators (spread preserves 24.5% to 26.7% against mean's
      3.3% to 5.0%, untested because it needs a retrain)

**Closed 2026-09-16.** G5 is DONE and was the last open REQUIRED goal.

### 3b. Two loose ends the work left behind

- [ ] **Decide what happens to `Reports/md_to_docx_patched.py`.** It embeds figures, which is the
      behaviour we deliberately replaced with red `[ INSERT FIGURE HERE ]` markers in
      `Reports/md_to_docx.py`. It is also what rendered `Milestone4_Report.docx` with
      `framework_v2.png` inside it. Leaving both is a trap: whoever picks the patched one next gets
      embedded images without meaning to. Delete it, or keep it and say in `REPRODUCIBILITY.md` what
      it is for
- [ ] **`Reports/Manuscript_v2.md:508` is still a `[PENDING]` placeholder**, and the SHAP-to-IG
      correction is not written into it. Tracked as ledger E2, and it cannot honestly be written
      until decision 1 above comes back

## 4. Compile the case study, task 4

The brief's Week 5 deliverables are the case-study results with uncertainty, the calibration
evaluation, and the explainability outputs. The numbers exist. The document does not. Milestone 4 got
`Milestone4_Report.md`, and there is no Milestone 5 equivalent.

**This is now the only real work left in Milestone 5.**

- [ ] **Write `Milestone5_Report.md`.** Follow the Milestone 4 structure and the house style in
      `Reports/md_to_docx.py`, Calibri 11, Light Grid Accent 1, 9pt cells. Figures go in as
      `![caption](../figures/x.png)` on their own line and render as a red INSERT FIGURE marker for
      manual pasting, so write the caption into the alt text
- [ ] Section on the case study result, stated without softening: the model beats persistence by
      16 / 15 / 16 / 44 percent at h3/h5/h10/h15 on the primary arm with no Ebola data at all, and
      14 comparisons clear zero on the corrected interval
- [ ] Section stating plainly that **the pre-registered criterion was NOT met**. It asked for the
      *adapted* model to beat persistence at h3 or h5 with the interval clearing zero. All four such
      intervals span zero, closest is h3 RMSE [-11.85, +0.35]. The wins come from the unadapted model
- [ ] **Section on why adaptation hurts**, using section 1a above. Say what we established and stop
      there: the fit region is genuinely degenerate (Step 0), and sample size, sparsity and padding
      are each insufficient to explain the damage (Step 1, 4 of 100 dev-fold cells). Do **not** claim
      we have the cause. Both docs are verified and quotable
- [ ] Section on calibration, which is the milestone's strongest result
- [ ] Section on explainability, marked PARTIAL, naming the SHAP-to-IG substitution rather than
      hiding it. The evidence is ready to quote: T1/T2/T3 all PASS, IG-versus-occlusion agreement
      264 of 280 with both seed-mean disagreements named, worst completeness error 0.0040 on the
      largest panel, and a random-weight control showing the lag comb is architectural rather than
      epidemiological, which is why lags are reported at band level only
- [ ] **Add the normalisation Threats paragraph** using the numbers in `CLAUDE.md` section 6. Ebola
      runs pooled scaling while every training panel runs per-node, district means sit 0.54 to 0.66
      off zero, and pooling HELPS on the matched panel by 6.8 percent. It helps most at h10 and h15,
      which is exactly where our Ebola wins are. Say it before a reviewer does
- [ ] **Run the numbers back out of the finished document and recompute them from disk** before
      sign-off, and mutation-test the verifier. This has caught 6 stale numbers in a single brief
      before
- [ ] **Check every Ebola sentence against the unsafe-to-claim list**, `Reports/Phase0_to_Now_Audit.md`

### Fences for whoever writes this

Three numbers live near each other and a reader will merge them if we let them.

1. "Zero-shot beats few-shot in 31 of 32 cells" is point means with **no interval**. It is an
   observation, not a significance claim. The paired-by-seed test clears in only 4 of 16 cells
2. "Unadapted wins 12 of 16, adapted wins 2 of 16" counts **significant wins against persistence**
   under `ebola_ci.py`. It is a different statistic from the 31 of 32. Never quote one to support
   the other
3. The Ebola headline is a node-averaged **country macro**. The baseline comparison table is
   **cell-pooled**. They must never be mixed

## 5. Explicitly not in Milestone 5

Listed so they do not leak in and inflate the milestone.

- Manuscript word count, **13,834** against 12,000, measured 2026-09-15. Up 25 from 13,809: the
  Figure 1 caption carries the density caveat and costs more than the table it replaced saved. That
  is Week 6
- The US-States weak-panel disclosure, ledger C9. Week 6
- The MTGNN sentence in `Reports/Week4_Experiments_Stakeholder_Brief.md:172-174`. Week 4 debt
- The retrain-based shuffled-adjacency control, about 6 hours. Optional, and it is an ablation,
  so Week 6 if it happens at all
- LDO3 zero-shot quantiles, about 10 hours retrain. Currently a stated limitation and staying one

---

**Shortest path to closing Milestone 5, as of 2026-09-15.** Two things, in this order.

1. **Write `Milestone5_Report.md`.** This is now the ONLY remaining item in Milestone 5. Every
   number it needs is on disk and verified, the explainability section can now be written plainly
   because D1 settled what it says, and it needs no GPU.

Everything else in this file is done.

# Milestone 5 todo

**Scope.** Week 5 of `progress/planning/Final Internal Project Brief.md`, days 21 to 25: the Ebola
case study, calibrated uncertainty, and explainability. Nothing outside that week is listed here.

**Position, updated 2026-09-15 after task 3 was worked through.** The compute for this milestone is
finished and nothing is queued. **Tasks 1, 2 and the whole engineering side of task 3 are done.**
Two things are left: three client decisions that no amount of code can move, and the milestone
report, which does not exist. Status below is checked against disk, not against the progress docs.

Brief tasks, and where each one actually stands:

| # | brief task | status |
|---|---|---|
| 1 | Run the Ebola out-of-distribution few-shot case study | **DONE** |
| 2 | Calibrated uncertainty plus calibration checks | **DONE** |
| 3 | Explainability outputs, SHAP global and local | **PARTIAL**, delivered as integrated gradients, not SHAP. Code, verification and reporting all done 2026-09-15; **blocked on the client, not on us** |
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

## 2. Uncertainty, task 2. Done

- [x] Conformal and ACI in `conformal.py`, fit and apply logs in `Reports/conformal_*.log`
- [x] The frozen cross-disease correction reads no Ebola outcome at all and lifts coverage from
      0.28-0.70 to 0.65-0.98. Online adaptation adds about 0.03 at short horizons and nothing long
- [x] District-level intervals rebuilt from `*__pernode.npz` in `ebola_ci.py`, read-only, asserts
      against the scored JSON before printing

**Nothing to do here either.** This is the strongest result in the milestone and it is already
machine-verified.

## 3. Explainability, task 3. Engineering done 2026-09-15

Gap Ledger row E1 is still **PARTIAL**, but **three of its four blockers are now closed and the
fourth is not ours**. Commits `b9783b5`, `17aedaa`, `b1d66f6`, `4aab8e6`. Nothing below needs a GPU,
a rerun or another line of code.

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

### 3a. The three client decisions, `G5_Explainability_Goals.md` section 8

These are the actual blockers. None of them is code and none of them can be closed by me alone.

- [ ] **Decision 1, integrated gradients instead of SHAP.** The brief says SHAP, global and local,
      and tags it REQUIRED. We built integrated gradients with an occlusion cross-check, which is
      defensible and arguably better here, but it is a deviation from signed scope and it has only
      been taken internally under C3. Get it in writing
- [ ] **Decision 2, retract the SHAP row. Urgent.**
      `RelatedWork_CompetitiveAnalysis_Benchmark.docx` is already with the client and asserts a
      capability we do not have. This is the one item on the list with an external cost that grows
      while it sits
- [ ] **Decision 3, accept a neighbour figure that does not explain accuracy.** The attribution
      describes what the model reads, not whether reading it helps. The gate-off ablation already
      answered the accuracy question separately, and negatively, in 0 of 40 cells

**Deferred by you on 2026-09-15**, code items closed first. Nothing further can be done on G5 until
these are answered, so the honest statement of G5's status is now "blocked on the client", not
"in progress".

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

1. **Send the client the three decisions**, decision 2 first, because they are holding a document
   that says we delivered SHAP and we did not. Until that comes back, G5 cannot close no matter what
   we build, and the milestone report cannot honestly describe the explainability section.
2. **Write `Milestone5_Report.md`.** Every number it needs is on disk and verified. This is the only
   remaining item that takes real time, and it needs no GPU.

Everything else in this file is done.

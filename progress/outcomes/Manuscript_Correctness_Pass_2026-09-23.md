# Manuscript v2 correctness pass, edit list (Milestone 6)

**Written 2026-09-23.** This records the edits to `Reports/Manuscript_v2.md` that the last two weeks
of evidence make necessary, together with the reason and the disk source for each. Per the standing
order I did **not** edit the manuscript itself. Each edit below gives the exact current text to match
and the exact replacement text, so whoever applies them can do so verbatim.

Baseline before any of these edits: `wc -w Reports/Manuscript_v2.md` = **13834 words**.
`python -m diagnostics.check_unsafe_claims` on the current manuscript: **no mechanical rule matched**
(3 refutation-exempt lines, all legitimate; 4 manual items). No em dashes anywhere in the new text.

Every number below was recomputed from its disk artifact before I wrote it. Sources and reproduce
commands are named per edit. One item did not reproduce as briefed: EDIT 6, the Bonferroni count.
**EDIT 6 was updated 2026-09-28**: the disputed family is settled as persistence only, the divisor
is 32 rather than 16, and the survivor count is 7 of 14 at both divisors.

---

## EDIT 1a. Shuffled-adjacency, gate-off limits paragraph (section 9.1, currently ~line 332)

**Reason.** The D2 retrain has now run and is scored and machine-verified
(`progress/outcomes/Shuffled_Adjacency_2026-09-23.md`, verifier `diagnostics/verify_shufadj_doc.py`,
which passes: completeness, tally, significant rows, coverage all ok). The sentence saying we "do
not run" a shuffled-adjacency arm is now false.

**Verified numbers** (from the outcome doc, re-verified against `ablation/single/*__shufadj.json`):
49 of 60 cells within noise; real-graph advantage only on dengue, on correlation at every horizon
(PCC deltas 0.034 to 0.042) and on error at h10 and h15 (roughly 2 to 4 percent).

MATCH THIS TEXT:

> And it does not separate "structure helps" from "any adjacency helps", which a shuffled-adjacency arm would do and which we do not run.

REPLACE WITH:

> And it does not separate "structure helps" from "any adjacency helps". A shuffled-adjacency retrain does separate them: training from scratch on a graph with the node labels permuted, so the topology and degree sequence are kept but the districts are wrong, matches the real-graph model in 49 of 60 cells, and the real district map earns a small advantage, stable across five seeds, only on dengue, on correlation at every horizon and on error at the two longest, by roughly 2 to 4 percent.

---

## EDIT 1b. Shuffled-adjacency, Threats paragraph (section 10, currently ~line 525)

**Reason.** Same as 1a. The Threats bullet also says we did not run the arm.

MATCH THIS TEXT:

> We did not run a shuffled-adjacency arm, which would separate "structure helps" from "any adjacency helps".

REPLACE WITH:

> A shuffled-adjacency retrain separates "structure helps" from "any adjacency helps". Training from scratch on a graph with permuted node labels, the same topology and degrees but the wrong districts, matches the real-graph model in 49 of 60 cells. The real district map earns a small advantage, stable across five seeds, only on dengue, our densest graph, on correlation at all four horizons and on error at the two longest, by roughly 2 to 4 percent, and it buys nothing on error on the four smaller panels. The graph's contribution is real but confined, and it is about shape rather than magnitude, consistent with the gate-off result.

---

## EDIT 2. Graph mechanism paragraph (section 9.1, replaces the "puzzling" guess, currently ~line 330)

**Reason.** The current one-line guess ("a gate reading the degree feature") is superseded by a
measured chain. The mechanism is what a reviewer will ask for.

**Verified numbers.**
- Raw incidence windows 14 to 94 percent district-specific in energy (13.8 to 93.5%), encoder output
  3.8 to 22.4 percent, a three to five fold reduction on every panel. Source
  `progress/outcomes/Input_Energy_2026-09-22.md` (t5), `results/misc/t5_input_energy.json`.
- Mean aggregation over neighbours removes a further 14 to 60 percent of what remains and deletes
  neighbour disagreement by construction. Source `Session_Audit_2026-09-10.md` section C.
- Gate open 0.27 to 0.60, no district near closed (0.0% below g=0.05). Source
  `Reports/gated+spatial_Contribution.txt:5-9` (g mean 0.271 japan to 0.604 dengue, 0.0% g<0.05).
- Relabelling adjacency at inference costs under 1 percent against seed noise of 8 to 14 percent.
  Source `CLAUDE.md` section 4 item 2 / `Session_Audit_2026-09-10.md` section C.

MATCH THIS TEXT:

> It also explains something otherwise puzzling. The learned gate is *highest* on the panel whose graph is emptiest, with roughly two thirds of dengue nodes having no observed neighbour at a typical origin, which fits a gate reading the degree feature rather than any real epidemiological coupling.

REPLACE WITH:

> We can also say why the graph buys so little, by following the district-specific signal through the model. The raw incidence windows are 14 to 94 percent district-specific in energy across the five development panels, meaning that is the share of each window that survives after the shared cross-district average is removed. The encoder's own output keeps only 3.8 to 22.4 percent, a three to five fold reduction on every panel, so most of what distinguishes one district from another is compressed away before the graph is ever consulted. Mean aggregation over neighbours then removes a further 14 to 60 percent of what is left, and it deletes neighbour disagreement by construction, because "neighbours agree" and "neighbours disagree widely" produce the same average. The gate is not the culprit: it sits open at 0.27 to 0.60 with no district anywhere near closed, so the model is not ignoring the channel, the channel simply has little district-shaped content left to pass. Consistent with that, relabelling the adjacency at inference, which hands each node a stranger's neighbours, costs under 1 percent against a seed-to-seed spread of 8 to 14 percent on the same runs. One honesty limit travels with this: an energy accounting shows structure is discarded, but it cannot establish that the discarded variation was forecastable signal rather than node-level noise amplified by our per-node scaling. These probes run on the single-disease checkpoints, and dengue is measured at one seed.

---

## EDIT 3. Adaptation mechanism (section 9.5.2, replaces "we do not yet have an explanation", ~line 465 to 467)

**Reason.** The mechanism is now measured, geometrically and then causally. The "no explanation"
sentence and the two guesses that follow it are behind the evidence.

**Verified numbers.** Sources `progress/outcomes/Adapter_Mechanism_2026-09-16.md` and
`progress/outcomes/Adapter_Constraint_2026-09-22.md` (verifier
`diagnostics/verify_adapter_constraint_doc`). With the trunk frozen the adapter is exactly affine
(max deviation 0.0). 76 to 85 percent of its effect acts along directions where the query data
spreads further than the support ever did, typically 5 to 12 times beyond the fitted range.
Shrinking toward the zero-effect map, strength chosen on support alone, recovers 77 to 90 percent on
the primary arm (h3/h5/h10) and 9 to 19 percent on the secondary. A hard clip to the fitted span
recovers almost nothing (+1 percent average).

MATCH THIS TEXT (the tail of the instability paragraph, ~line 465):

> More adaptation data producing a worse, wider and much less stable fit is a symptom rather than a result, and we do not yet have an explanation for it.

REPLACE WITH:

> More adaptation data producing a worse, wider and much less stable fit is a symptom, and we can now say what drives it. With the trunk frozen the adapter is exactly an affine map, verified to a deviation of 0.0, so it can be taken apart directly. Between 76 and 85 percent of the change it makes to a query forecast acts along directions where the query data spreads further than the support data ever did, typically 5 to 12 times beyond the range the fit was determined over. Shrinking that fitted map back toward the borrowed zero-effect map, at a strength chosen on the support data alone, recovers 77 to 90 percent of the damage on the primary arm at the three horizons that have support, and only 9 to 19 percent on the secondary arm, whose fuller-rank support hides from a support-only rule the fact that the query sits outside it. A hard clip to the fitted span recovers almost nothing, which locates the damage in the magnitude of the fitted change along weakly constrained directions rather than literally outside the span. This is one mechanism test, and we do not present it as the sole cause.

Then MATCH the guesses paragraph (~line 467) and REMOVE the two guesses, keeping the honest framing:

> **We are careful about how strongly we state this.** The comparison is across arms that share a trunk per seed, so a seed-paired test is appropriate, and it is significant in 4 of 16 cells. We report the inversion as a solid *observation* with a partly significant test behind it, not as an established rule. Two explanations fit. Fitting 1,428 parameters on 59 highly correlated support cells may overfit the early exponential phase of an outbreak, and the pooled support-only scaler is estimated from exactly the weeks that look least like the query period. Telling those apart is future work.

REPLACE WITH:

> **We are careful about how strongly we state this.** The comparison is across arms that share a trunk per seed, so a seed-paired test is appropriate, and it is significant in 4 of 16 cells. We report the inversion as a solid *observation* with a partly significant test behind it, not as an established rule. The affine analysis above says where the damage lives, in the few-shot-specific change applied outside the region it was fitted in, but it is a single mechanism test and does not rule out that the normalisation mismatch of Section 10 is upstream of why that change sits where it does.

---

## EDIT 4. Section 9.8 Explainability (replaces the [PENDING] stub, ~line 501) plus two contradicting sentences (~527, ~543)

**Reason.** The stub says nothing is delivered; explainability IS delivered and machine-verified
(`progress/outcomes/G5_Explainability_Results.md`, Gap Ledger E1 DONE; `Reports/explaiability_report.md`).

**Verified numbers** (from the G5 results doc, recomputed from `results/explain/`):
- IG with occlusion cross-check, agreement 264 of 280 per (seed, horizon), 54 of 56 seed-mean.
- SHAP substitution accepted on measured grounds: SHAP needs 15,616x (ebola_L12) to 1,834,240x
  (dengue) more model evaluations for a global read, and reruns unstably (r=0.68 at 256 coalitions)
  where IG reruns to r=1.000000.
- Worst completeness error 0.0040 at 128 steps (dengue).
- Per-node training-mean baseline on the Ebola arms.
- T1/T2/T3 falsification checks pass. Random-weight control shows the lag comb is architectural
  (r 0.912 to 0.963 against a shuffle ceiling below 0.609), which is why lags are reported at band
  level, not single-lag.

MATCH THIS TEXT (the stub):

> *[PENDING. A minimal global attribution over the four core channels is scoped and will be reported here. This version claims no attribution result, and the comparison in Section 3 makes no explainability claim on our behalf that the code does not currently support.]*

REPLACE WITH:

> We attribute each forecast over the four core channels with integrated gradients, a path method that assigns the change in a prediction from a neutral baseline to the inputs, and cross-check it with occlusion, which measures the effect of hiding one input at a time. The two agree on the top channel and top lag band in 264 of 280 per-seed-per-horizon cells, and 54 of 56 at seed mean, with both disagreements named rather than smoothed over. We deliver this in place of the SHAP read named in an earlier plan, and the substitution is accepted on measured grounds rather than convenience. A single global SHAP read costs 15,616 to 1,834,240 times more model evaluations than integrated gradients over the same grid, and at feasible sample sizes it reruns unstably, agreeing with itself at a correlation of 0.68 at 256 coalitions where integrated gradients reruns to 1.000000. On the five busiest Ebola districts the two methods pick the same top channel every time, so nothing is lost by the swap. The attribution completes its own path: the worst completeness error is 0.0040 at 128 integration steps, and on the Ebola arms the baseline is each district's own training-window mean rather than an all-zero reference, so the counterfactual is "this district at its typical level". Three falsification checks pass: incidence is the top channel, recent lags outweigh distant ones, and the influenza seasonality share exceeds Ebola's on both arms. A random-weight control reproduces the trained model's lag pattern at a correlation of 0.91 to 0.96 against a shuffle ceiling below 0.61, which shows the fine comb of lags is a fingerprint of the temporal convolution's dilation structure rather than a learned epidemiological memory, so we report lags at the level of coarse bands and not single weeks.

Optionally place `figures/explain.png` after this paragraph (it exists) using the insert marker,
caption from `explain.py` reporting; not required.

MATCH THIS TEXT (Threats, ~line 527):

> **One required capability is not delivered in this version**: explainability. It is scoped and registered, and nothing is claimed for it here.

REPLACE WITH:

> **Explainability is delivered as integrated gradients with an occlusion cross-check rather than the SHAP read named in an earlier plan.** The substitution is justified on measured compute and stability grounds in Section 9.8, and the two methods agree on which channel drives a forecast, so we report input attribution but not a SHAP-specific result.

MATCH THIS TEXT (Conclusion, ~line 543):

> Each of these is a constraint an operational early-warning system would have to design around, and none is visible without a build that holds the disease constant. What remains is stated plainly rather than promised loosely. The explainability component is registered and not yet run, and the case study rests on a single outbreak. The framework, the datasets, the evaluation protocol and every artefact behind these numbers are available, so each can be checked and extended.

REPLACE WITH:

> Each of these is a constraint an operational early-warning system would have to design around, and none is visible without a build that holds the disease constant. What remains is stated plainly rather than promised loosely. Input attribution is delivered as integrated gradients with an occlusion cross-check in place of SHAP, and the case study rests on a single outbreak. The framework, the datasets, the evaluation protocol and every artefact behind these numbers are available, so each can be checked and extended.

---

## EDIT 5. Normalisation Threats paragraph (section 10, add a new paragraph)

**Reason.** Section 10 does not yet disclose that Ebola runs a pooled scale while every training
panel runs per-node, a covariate shift we introduced and publish the diagnostic for. A reviewer will
find it.

**Verified numbers.** Sources `CLAUDE.md` section 6 and
`progress/outcomes/Norm_Probe_2026-09-16.md` (verifier built into `experiments/norm_probe.py`
`--selfcheck`). District means arrive 0.54 to 0.66 from zero (ebola_L20 0.5363, ebola_L12 0.6550)
against 0.0000 on influenza. On the matched development panel CONSISTENT pooling (trained and tested
pooled) helps 6.8 percent on average and most at long horizons. MISMATCHED pooling (per-node trunk
fed pooled inputs) hurts 16 of 16 development cells by 4.6 to 101.5 percent. Scoring Ebola per-node
instead never helped on any of 40 paired cells (dz<pz is 0 of 40).

ADD THIS PARAGRAPH (place it in section 10, after the transfer-folds threat, before the serial-dependence one):

> **Ebola runs a different normalisation from every training panel, and we introduced the mismatch ourselves.** The trunk is trained on inputs scaled district by district, so every district it ever saw sat centred on zero, while Ebola is scaled by one pooled per-disease statistic and its districts arrive 0.54 to 0.66 away from zero, against 0.0000 on influenza. We measured this from both ends. Feeding a development panel the Ebola way, a per-node trunk handed pooled inputs, makes it worse in all 16 cells by 4.6 to 101.5 percent, so the mismatch genuinely costs accuracy in the direction Ebola experiences it. That is not the same as saying pooling is bad: CONSISTENT pooling, trained and tested pooled, helps the matched panel by 6.8 percent on average and most at the long horizons, exactly where the Ebola wins are. What hurts is the MISMATCH, and the two must never be collapsed into a single claim about "pooling". We cannot simply score Ebola per-node instead, because 43 of its 61 districts have no support history to build a scale from, and the closest approximation the data allows was strictly worse: it never improved a single one of 40 paired cells on the unadapted arm. The pre-registered pooled configuration is therefore the best of everything we tried. The naive floors are scored in raw counts and never touch the scaler, so none of this changes the direction of the floor comparisons.

---

## EDIT 6. Multiplicity sensitivity sentence (section 10, next to the existing policy, ~line 521)

**RESOLVED 2026-09-28. The divisor is now 32 and the count is 7. Use the sentence at the bottom of
this section, not the divisor-16 draft that stood here before.**

**What was disputed.** Two passes computed a Bonferroni survivor count and disagreed, 9 against 7,
because they built two different 32-cell families and both called the answer "the 14". Pass A used
RMSE only across both naive floors. Pass B used both metrics against persistence only. Pass B was
right. `Reports/Phase0_to_Now_Audit.md:257` defines the canonical family: against persistence, 14 of
32 cells clear zero; against support_mean it is 13 of 32; all 64 together give 27. The same line says
the unadapted arm wins 12 of 16 and the adapted arm 2 of 16, which sums to exactly 14. So the
canonical family is **persistence only, both metrics (RMSE and MAE), both arms (ebola_L12 and
ebola_L20), both regimes, four horizons = 32 cells, 14 of which clear zero at 95 percent.**

**The divisor is 32**, matching that full family, not the 16 the old draft used. That is the
conservative choice and it is deliberate: buying extra survivors with a soft divisor would be a bad
trade for a paper whose credibility rests on reporting its nulls straight.

**What I found on disk.** I added `--bonferroni M` and `--floor` to `ebola_ci.py` and recomputed,
B=10000, seed 0, stratified. The uncorrected run is bit-identical to the archived
`results/reports/ebola_district_ci.log` on every persistence line, so nothing existing moved.

| family | clear at 95% | divisor 16 | divisor 32 |
|---|---|---|---|
| RMSE vs persistence (2 arms x 2 regimes x 4 h = 16) | 8 | 5 | 5 |
| MAE vs persistence (16) | 6 | 2 | 2 |
| **RMSE + MAE vs persistence (32), the canonical family** | **14** | **7** | **7** |

The same seven cells survive at both divisors, so the count does not depend on the choice between
them. The seven are, with the divisor-32 interval:

| metric | arm | regime | h | interval at divisor 32 |
|---|---|---|---|---|
| RMSE | ebola_L12 (primary) | adapted † | 15 | [-61.709, -2.449] |
| RMSE | ebola_L12 (primary) | zero-shot | 10 | [-18.138, -0.266] |
| RMSE | ebola_L12 (primary) | zero-shot | 15 | [-61.936, -2.568] |
| RMSE | ebola_L20 (secondary) | zero-shot | 3 | [-20.370, -0.449] |
| RMSE | ebola_L20 (secondary) | zero-shot | 15 | [-60.714, -0.939] |
| MAE | ebola_L12 (primary) | adapted † | 15 | [-29.811, -1.092] |
| MAE | ebola_L12 (primary) | zero-shot | 15 | [-29.949, -1.208] |

† Pre-registration E6 binds h10 and h15 on the primary arm to the zero-shot label, never few-shot,
because L12 has 0 fitted adaptation pairs at h15. Both adapted-named survivors sit in exactly that
cell, so under E6 neither may be reported as a few-shot win. Apply the same dagger footnote used for
Tables 6 and 8.

**The two clauses that were being treated as safe, rechecked at divisor 32.** The h15 against
persistence comparison survives on both arms **on RMSE only**: L12 at [-61.936, -2.568] and L20 at
[-60.714, -0.939]. On MAE only the primary arm's h15 survives, because L20 zero-shot h15 MAE widens
to [-29.394, +0.885] and no longer clears. Write "on RMSE", never the bare "on both arms". The
second clause holds and is stronger than before: 5 of the 7 survivors are zero-shot, and the other
2 are the primary-arm h15 cell that E6 labels zero-shot anyway, so no survivor comes from a cell the
pre-registration would count as an adapted win.

Reproduce, about 70 seconds per run:

```
conda run -n ebola-train python ebola_ci.py --metric rmse mae --floor persistence --no-crosstab --bonferroni 32
```

Archived at `results/reports/ebola_district_ci_bonferroni32.log`, with the divisor-16 control at
`results/reports/ebola_district_ci_bonferroni16.log`.

**Replacement sentence (verified), to add after the existing multiplicity policy:**

> As a sensitivity check, applying a Bonferroni correction across the full family of thirty-two comparisons against the persistence floor, which is two arms by two regimes by four horizons on RMSE and MAE (percentiles 0.078125 and 99.921875), leaves 7 of the 14 comparisons that clear zero at 95 percent still clear, 5 of them on RMSE and 2 on MAE. The h15 comparisons against persistence survive on both arms on RMSE, and every surviving cell is one the pre-registration labels unadapted.

Do NOT write "9 of the 14 RMSE comparisons"; that number does not reproduce from disk. Do not quote
any survivor count without naming the floor family in the same sentence, because the unnamed count
is half of why this went stale twice.

---

## EDIT 7. E6 label violations in Table 6 and Table 8

**Reason.** E6 (`progress/decisions/Ebola_Prereg.md:191-192`) binds h10 and h15 on the primary arm
to the zero-shot label, never few-shot. L12 has 0 adaptation pairs at h15 and 18 on near-fully padded
windows at h10. `figures/F5` already handles this with an asterisk marker plus a caption note; the
tables must match.

**Table 6 (coverage, ~line 413).** The "L12 few-shot" h15 row is a primary-arm h15 result under a
few-shot label. Add a marker and a footnote.

MATCH:

> | L12 few-shot | 15 | 0.275 ± .103 | 0.651 ± .138 | **0.658 ± .132** | 0.812 | 3 / 18 |

REPLACE WITH:

> | L12 few-shot † | 15 | 0.275 ± .103 | 0.651 ± .138 | **0.658 ± .132** | 0.812 | 3 / 18 |

Then extend the Table 6 caption (currently ends "...because its raw intervals are already far too
wide.") by appending:

> † On the primary L12 arm, h10 and h15 carry no fitted adaptation (0 pairs at h15, 18 padded pairs at h10), so per pre-registration E6 the adapted arm is reported as zero-shot at these horizons and the few-shot row is shown for completeness only.

**Table 8 (~lines 452 to 459).** The "Adapted, MAE" row spans h3/h5/h10/h15; at h10 and h15 the
primary adapted arm must be labelled zero-shot. Mark the two cells and footnote them. The numbers are
untouched.

MATCH:

> | Adapted, MAE | 24.69 ± 0.48 | 24.61 ± 0.62 | 26.05 ± 3.89 | 20.10 ± 0.52 |

REPLACE WITH:

> | Adapted, MAE | 24.69 ± 0.48 | 24.61 ± 0.62 | 26.05 ± 3.89 † | 20.10 ± 0.52 † |

And in the "Adapted skill vs persistence" row, the h10 cell already reads "within noise" and h15
reads "+37.0%"; add the same dagger to h10 and h15 there if the marker is to be consistent, or rely
on the single footnote. Append to the Table 8 caption (currently ends "...the model's own seed
spread."):

> † Per pre-registration E6, h10 and h15 on the primary arm are reported as zero-shot, not few-shot: h15 has zero adaptation pairs and h10 has only 18 on near-fully padded windows, so the "adapted" figures at these two horizons are the borrowed zero-shot map, matching Figure F5.

---

## EDIT 8. Place the six figures

**Reason.** The paper carries one figure against eight tables, a known reviewer objection.
`paper_figures.py` draws six, every value read from disk (it asserts spot cells and tallies).
Insert each on its own line as `![caption](../figures/<name>.png)`, which the docx path
(`Reports/md_to_docx.py`) renders as an INSERT FIGURE marker.

**Numbering.** The existing gate figure is "Figure 1" in 9.1. Adding six more in document order forces
a coherent renumber. Recommended document-order scheme (and the one downstream reference it touches):

| Figure | file | section | placement |
|---|---|---|---|
| Figure 1 | F4_window_sensitivity.png | 5.2 (~line 140) or a new Appendix | after the window-sweep sentence |
| Figure 2 | gate.png (existing) | 9.1 (~line 313) | unchanged position, renumbered from 1 to 2 |
| Figure 3 | F1_spatial_ablation.png | 9.1 | after Table 3 (gate-off table) |
| Figure 4 | F5_ebola_adapted_vs_unadapted.png | 9.5 | after Table 8 |
| Figure 5 | F6_baseline_comparison.png | 9.6 | after the baselines paragraph (~line 481) |
| Figure 6 | F2_metalearning_ablation.png | 9.7 | after the ANIL paragraph (~line 493) |
| Figure 7 | F3_epi_penalty_ablation.png | 9.7 | after the epi-penalty paragraph (~line 497) |

If F4 is preferred in an appendix (there is none today), gate.png can stay Figure 1 and F1..F6 number
2..7 in the order above. Note: `progress/outcomes/G5_Explainability_Results.md` refers to
"figures/gate.png is Figure 1"; if gate.png becomes Figure 2, update that reference too. Flagging, not
fixing, since that doc is outside this pass.

**Exact caption text to paste** (printed by `paper_figures.py`; do not retype, these are copied from
the script source). Each keeps its "lower is better" direction statement:

- **F1_spatial_ablation** (9.1): "F1. Turning the spatial message-passing graph off, paired against
  the learned-gate model on the same seed (5 seeds, single-disease checkpoints; RMSE and MAE, lower
  is better; horizon in weeks). Bars are the percent change in error from removing the graph, positive
  meaning error drops when the graph is removed; green bars are significant under the committed
  report's within-noise rule. Across all 40 error cells the graph improves error in 0 and removing it
  significantly helps in 8, with the remaining 32 within noise, so the spatial channel does not buy
  accuracy. Scope: this is an inference and gate-off test on single-disease models, and the graph
  still helps correlation in a minority of cells, which is not shown here."

- **F2_metalearning_ablation** (9.7): "F2. Adaptive meta-learning (ANIL) against its own seed-matched
  control across 32 cells and four held-out folds (RMSE, positive means ANIL beats the control;
  whiskers are the 95 percent t-interval). The control sees the identical episode stream and
  outer-update count with no inner adaptation loop, so the contrast isolates meta-learning itself.
  ANIL wins 0 cells, is significantly worse in 1, and is within noise in the other 31, so
  meta-learning does not help here. Scope: every arm warm-starts from an existing trunk and the
  meta-test fits a fresh adapter on the full held-out train fold, so this measures meta-fine-tuning,
  not few-shot adaptation."

- **F3_epi_penalty_ablation** (9.7): "F3. Adding an epidemiology-informed penalty at two strictness
  bounds, paired against the plain single-disease model on the same seed (5 seeds; RMSE, lower is
  better; horizon in weeks). Bars are the percent change in error, positive meaning error drops with
  the penalty; pale bars are within noise under the committed report's rule. The penalty leaves error
  essentially unchanged across all four panels, one influenza cell aside, so the hand-designed
  epidemic prior does not improve accuracy. Scope: dengue was not run for this ablation and is shown
  as absent, not omitted."

- **F4_window_sensitivity** (5.2 / Appendix): "F4. Sensitivity of the single-disease model to the
  input lookback window (RMSE, lower is better; horizon in weeks; 5-seed mean, error bars the
  across-seed spread). The released model uses a 20-week window, shown in black between a shorter
  12-week and a longer 32-week setting. A shorter window helps modestly on US-States and Japan at long
  horizons while a longer window clearly hurts COVID, but most cells are within noise of the base in
  the paired test, so the model is not fragile to this choice. Scope: dengue was not run for this
  sweep and is absent by design."

- **F5_ebola_adapted_vs_unadapted** (9.5): "F5. Ebola forecasts from a model that never saw Ebola in
  training, on the primary L12 and secondary L20 arms (country-macro RMSE in raw case counts, lower is
  better; horizon in weeks; 5-seed mean, error bars the across-seed spread). Both the unadapted
  (zero-shot) trunk and the adapted variant sit below the persistence floor across horizons, and the
  unadapted line is at or below the adapted one almost everywhere. On the L12 arm the h10 and h15
  adapted points, marked with an asterisk, are reported as zero-shot rather than few-shot as prereg E6
  requires: h15 has zero adaptation pairs and h10 has only 18 on near-fully padded windows. These are
  point means without intervals, so no significance is claimed and the pre-registered
  adapted-beats-persistence criterion was not met."

- **F6_baseline_comparison** (9.6): "F6. Our encoder against four published GNN baselines (EpiGNN,
  Cola-GNN, HeatGNN, MTGNN), three classical baselines (GBM, SARIMA, ARIMA) and persistence, one small
  multiple per panel (country-macro RMSE, lower is better; horizon in weeks; means over available
  seeds). Every comparator trains on the disease it is scored on. On dengue the baselines score the
  2,392-node subset while our encoder and persistence score the full 6,161-node set, so the encoder
  and persistence are withheld from the dengue axis rather than plotted on a mismatched scale; dengue
  SARIMA and ARIMA fall back to persistence on 39 percent of nodes and SARIMA equals ARIMA on dengue
  and COVID. MTGNN is drawn greyed because it collapses to a near-constant forecast on most influenza
  files and is shown for completeness, not as a valid comparator, and COVID has no GNN baseline."

To regenerate the figures and reprint the captions:
`conda run -n ebola-train python paper_figures.py --only all`.

---

## Fences checked

- Every touched Ebola sentence was checked against the unsafe list in
  `Reports/Phase0_to_Now_Audit.md:250-269`. None writes "criterion met" (the new 9.8 and Table
  captions preserve "criterion was not met"), none writes few-shot at L12 h10/h15 (EDIT 7 fixes the
  two that did), none quotes the 31-of-32 observation as a significance claim (not touched), and no
  new interval is quoted (EDIT 6 uses the `ebola_ci.py` district bootstrap on the country-macro, the
  matched estimand). No em dashes in any new text.
- `python -m diagnostics.check_unsafe_claims` should be rerun AFTER the edits are applied; on the
  current unedited manuscript it already passes with no mechanical hit. The EDIT 7 label fixes remove
  two of the four MANUAL-only concerns (regime-labels).

## Word count expectation

These edits add net prose (mechanism paragraphs, explainability section, normalisation paragraph,
six figure captions), so the count will GROW from 13834. The trim to 12000 is a separate pass and is
not attempted here.

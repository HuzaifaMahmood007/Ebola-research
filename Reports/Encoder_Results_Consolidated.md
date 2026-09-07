# Consolidated Encoder Results: Single-Disease, Joint, and Transfer Regimes

**Phase 3, Week 4 · Prepared 2026-07-30**

**Companion artifacts.** Full numeric matrix: `progress/outcomes/Results_Matrix.md` (superseded
2026-09-07, kept for provenance only, see the banner at the top of that file). Generator:
`results_matrix.py` (`--selfcheck` passes). Scoring authority: `score.py` (`_demo()` passes; seven
metrics live).

---

## Abstract

We evaluate a gated inductive spatio-temporal encoder across four training regimes — single-disease,
joint multi-dataset, leave-one-dataset-out (LODO) transfer, and leave-one-disease-out (LDO) transfer
— on four epidemiological panels at four forecast horizons, under a single scoring stack and the
full Week-4 metric set. Three findings are material.

**First, the previously reported transfer headline is substantially an artifact of the comparison
reference.** Holding the transfer prediction fixed and varying only the reference — seed-matched
versus five-seed mean — moves influenza-US-regions h3 from +19.9% to +9.9% and moves dengue h3 from
+2.9% to **−15.4%, a sign inversion**. This reproduces all three arithmetic discrepancies raised in
the Week-3 client review to the decimal place, and confirms the client's diagnosis exactly.

**Second, when the disease rather than the population is held out, the transfer benefit does not
survive.** Under the corrected leave-one-disease-out fold structure, **no panel shows a positive
cross-disease effect on RMSE or MAE that clears its noise floor at any horizon: twelve of sixteen
cells are significantly *negative*, four are within noise, none positive.** The population-transfer
regime that appeared strongest (+24.0% on influenza-Japan h3) is precisely the regime in which the
disease was never held out. The epidemiological peak metrics are a partial exception, reported in §4.5.

**Third, cross-disease zero-shot transfer fails outright on influenza**, with errors several times
the single-disease baseline. Adaptation is not an enhancement to the transfer story; it is load-bearing.

The direction of these results was anticipated by the client, who wrote that the honest number
"will come in weaker than the current headline." It does. We report it as the primary result.

---

## 1. Scope and provenance

This report covers **our encoder only**. Published-baseline reproduction is a separate workstream
(`paper_compare.py`); the HeatGNN queue was still executing when this was generated and no baseline
figures appear here.

Every number derives from JSON records in `results/{single,joint,lodo}/`, recomputed from disk by
`results_matrix.py`. No figure is carried forward from an earlier write-up. Where this report
disagrees with an earlier document, this report supersedes it.

Prior to analysis, `rescore_encoder.py` backfilled the Week-4 `nrmse` metric across all encoder
artifacts: **65/65 succeeded, 0 skipped on mask mismatch.** The backfill is verified rather than
assumed — for every (node, horizon) the reconstructed scored-cell count must equal the count the run
itself recorded, and any file with a single mismatch is skipped and reported rather than patched
with an inferred mask.

---

## 2. Methods

### 2.1 Aggregation

The headline aggregation is **country-macro**: per-node metrics over each node's observed evaluation
cells, averaged within a country, then averaged across countries with equal weight. This prevents
Brazil's 77% share of dengue nodes from becoming the dengue score. For the three influenza panels the
figure is identical to the node-mean by construction (one country each); only dengue's twelve
countries make the two differ.

Nodes whose ground truth is constant across the scored window are excluded by default. They are
trivially predictable and their Pearson correlation is undefined; they remain in the graph as
message-passing neighbours.

### 2.2 Scored population

| panel | scored nodes | countries |
|---|---|---|
| influenza_japan | 47 | 1 |
| influenza_us-regions | 10 | 1 |
| influenza_us-states | 49 | 1 |
| dengue | 6,161 | 12 |

Metrics are computed in **count space**, with the per-node scaler inverted before scoring.

### 2.3 Metric set

Seven per-node metrics: `rmse`, `mae`, `pcc`, `smape`, `peak_intensity`, `peak_timing`, and `nrmse`.

`nrmse` (= RMSE / mean(y), i.e. CV(RMSE)) was added in Week 4 in response to the review point that
"across 7,165 dengue regions raw RMSE is dominated by the biggest ones." It is undefined — not zero —
where a node's mean level is non-positive, since a node recording no cases has no scale to normalise
against. `sMAPE` excludes cells where truth and prediction are both zero rather than scoring them as
perfect, which matters because dengue is heavily imputed.

Uncertainty metrics (WIS, CRPS, empirical coverage, interval width, PIT) are implemented and
self-checked in `score.py` but **do not appear in this report**: they consume quantile predictions,
and no run currently archives them. This is the principal outstanding gap against the review's
metrics requirement.

### 2.4 Statistical protocol

Four rules from the client's reporting standards are enforced mechanically by the generator.

**Dispersion on every figure.** Cells report `mean ± sd (n seeds)`. A single-seed cell is labelled as
such and is never given a standard deviation.

**No averaging across datasets.** Every table is per-panel. Pooling dengue (6,161 nodes) with
US-regions (10 nodes) is not a supported operation.

**Deltas are paired by seed.** Both arms share the seed set, so pairing cancels shared initialisation
noise; treating them as independent would inflate the threshold by roughly √2 for no reason. Only
seeds present in both arms enter a delta. Intervals are two-sided 95% **t** intervals over the
per-seed paired deltas — at n = 4–5 the critical value is 2.78–3.18, and substituting the normal
quantile 1.96 would manufacture significance.

**"Within noise" rather than a direction.** A delta whose interval covers zero is printed as
`within noise`, never as a signed claim.

**Units.** Error-metric deltas are improvement percent, positive meaning better than reference.
`pcc` deltas are reported in **correlation points, never percent** — a correlation is signed and can
approach zero, where a percent change explodes or inverts for a model that genuinely improved.

### 2.5 Like-for-like verification

A delta is meaningful only if both arms scored the same nodes. Because `score.py` drops
constant-truth nodes, and a different prediction can change which nodes qualify, this is checked
rather than assumed: `n_nodes` is compared between every arm and its reference for each
(panel, horizon, seed).

**Result: PASS, zero mismatches.** Every regime scored an identical node population to its reference
in every comparison. All deltas in this report are like-for-like. This check exists because a
not-like-with-like comparison has already produced a misleading figure elsewhere in the project.

---

## 3. Experimental regimes

| regime | description | seeds |
|---|---|---|
| **single** | trained and evaluated on one panel; the reference arm | 5 (all panels) |
| **joint:uniform-uniform** | all panels co-trained, uniform sampler | 5 (all panels) |
| **joint:sqrt-uniform** | sampler variant probe | 1 |
| **LODO adapted** | leave-one-**dataset**-out, then adapter fitted | 1 (seed 42) |
| **LODO zero-shot** | leave-one-dataset-out, no adaptation | 1 (seed 42) |
| **LDO adapted** | leave-one-**disease**-out, then adapter fitted | 4 flu / 5 dengue |
| **LDO zero-shot** | leave-one-disease-out, no adaptation | 4 flu / 5 dengue |

The LODO/LDO distinction is the central methodological correction of Week 4 and the two must not be
conflated.

Under **LODO**, holding out influenza-Japan still leaves the encoder trained on influenza-US-regions
and influenza-US-states. The *population* is held out; the *disease* is not. This measures population
transfer.

Under **LDO**, all three influenza panels are held out together as a single fold, against dengue as
the other. This measures genuine cross-disease transfer, and it is the number the paper's central
claim depends on.

---

## 4. Results

### 4.1 Single-disease reference

Country-macro RMSE, mean ± sd over five seeds.

| panel | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| influenza_japan | 734.9 ± 58.5 | 841.1 ± 35.4 | 1,063.0 ± 74.9 | 1,031.4 ± 121.0 |
| influenza_us-regions | 613.2 ± 82.3 | 728.2 ± 76.3 | 787.8 ± 40.5 | 812.8 ± 68.1 |
| influenza_us-states | 113.4 ± 3.9 | 136.8 ± 4.3 | 151.5 ± 1.9 | 154.1 ± 4.2 |
| dengue | 42.06 ± 6.04 | 49.94 ± 6.05 | 56.21 ± 0.62 | 57.38 ± 0.22 |

Scale-normalised error (`nrmse`) makes the panels comparable in a way raw RMSE cannot. Dengue is the
hardest panel in relative terms (1.80 → 2.08 across horizons) despite having the smallest raw RMSE,
which is precisely the distortion the metric was introduced to correct. US-regions is the easiest
(0.47 → 0.66).

Correlation degrades sharply with horizon on dengue — `pcc` falls from 0.422 at h3 to **0.106 at
h15** — against a comparatively flat influenza-Japan profile (0.872 → 0.837). Long-horizon dengue
forecasting is close to uninformative under this encoder, which bears directly on the Ebola case
study, discussed in §6.

### 4.2 Joint training

Joint training with a uniform sampler is **indistinguishable from single-disease training** on the
overwhelming majority of cells. On dengue RMSE, three of four horizons are within noise and the
fourth is a statistically clear but practically negligible −0.9% ± 0.5. The pattern holds across
panels.

The `sqrt-uniform` sampler variant is a **single seed** and is reported for completeness only. Its
large apparent dengue regressions (−61.0% at h3) rest on one draw against a five-seed reference and
should not be interpreted.

This is a null result, and it is a useful one: it is the clean motivation for the freeze-then-adapt
architecture rather than an unfinished experiment.

### 4.3 Population transfer (LODO) — apparently strong, but 1 seed

Improvement % vs single, seed-matched reference (Reference A), RMSE:

| panel | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| influenza_japan | +24.0% | +4.5% | −25.4% | −22.6% |
| influenza_us-regions | +19.9% | +28.0% | +2.4% | −0.5% |
| influenza_us-states | +5.8% | +4.4% | −0.2% | −4.1% |
| dengue | +2.9% | +5.7% | −0.8% | −1.7% |

**Every cell in this table is a single seed and none is statistically testable.** The strong
short-horizon gains are the figures that drove the Week-3 headline. Two independent problems
undermine them, addressed in §4.4 and §4.5.

Note already the horizon structure: gains concentrate at h3–h5 and turn negative by h10–h15. Transfer,
as currently measured, helps least at exactly the long horizons.

### 4.4 Cross-disease transfer (LDO) — the honest number

Improvement % vs single, paired by seed, RMSE. Bold indicates the 95% t interval excludes zero.

| panel | h | LODO adapted (1 seed) | **LDO adapted (4–5 seeds)** |
|---|---|---|---|
| influenza_japan | h3 | +24.0% | within noise (+3.4 ± 35.6, n=4) |
| influenza_japan | h5 | +4.5% | **−16.9% ± 9.3** |
| influenza_japan | h10 | −25.4% | **−35.2% ± 19.3** |
| influenza_japan | h15 | −22.6% | **−51.3% ± 39.4** |
| influenza_us-regions | h3 | +19.9% | within noise (−4.7 ± 17.2, n=4) |
| influenza_us-regions | h5 | +28.0% | **−10.0% ± 8.5** |
| influenza_us-regions | h10 | +2.4% | **−32.6% ± 16.8** |
| influenza_us-regions | h15 | −0.5% | **−51.2% ± 25.0** |
| influenza_us-states | h3 | +5.8% | **−4.4% ± 3.7** |
| influenza_us-states | h5 | +4.4% | **−5.7% ± 4.4** |
| influenza_us-states | h10 | −0.2% | **−11.2% ± 5.2** |
| influenza_us-states | h15 | −4.1% | **−13.1% ± 5.3** |
| dengue | h3 | +2.9% | within noise (−9.8 ± 16.6, n=5) |
| dengue | h5 | +5.7% | within noise (−5.3 ± 13.9, n=5) |
| dengue | h10 | −0.8% | **−1.6% ± 1.4** |
| dengue | h15 | −1.7% | **−1.5% ± 0.5** |

**On RMSE and MAE, no cell shows a positive cross-disease transfer effect clearing its noise floor at
any horizon on any panel.** Twelve of sixteen are significantly negative and the other four are
within noise; the count is identical for MAE. (The peak metrics behave differently — see §4.5.) The three
apparently large positive LODO cells (+24.0%, +19.9%, +28.0%) all collapse to within-noise or
significantly negative under the corrected fold structure.

The client's stated hypothesis was that the strongest LODO results were the folds with the most
influenza leakage. The data support this: the two largest LODO gains occur on influenza panels where
two sibling influenza panels remained in training, and both vanish once influenza is held out as a
disease.

The degradation is also **monotone in horizon** on every panel — mildest at h3, most severe at h15
(influenza-Japan −51.3%, US-regions −51.2%). Cross-disease transfer is worst exactly where a genuine
outbreak-response application would need it most.

### 4.5 Epidemiological metrics — peak intensity and peak timing

The review named three metrics specifically: scale-normalised error, peak intensity error and peak
timing error, on the grounds that "epidemiology reviewers care about those specifically." All three
are computed for every regime. `nrmse` appears in §4.1; the two peak metrics are reported here.

**Single-disease reference.** `peak_intensity` is the absolute difference between predicted and true
peak height, in counts; `peak_timing` is the absolute difference between predicted and true peak
position, in observed-eval-cell steps (weeks on the dense influenza panels).

| panel | metric | h3 | h5 | h10 | h15 |
|---|---|---|---|---|---|
| influenza_japan | peak_intensity | 2,717 ± 532 | 2,885 ± 448 | 3,447 ± 621 | 3,471 ± 759 |
| influenza_us-regions | peak_intensity | 2,132 ± 448 | 2,535 ± 374 | 2,712 ± 631 | 2,766 ± 637 |
| influenza_us-states | peak_intensity | 408.7 ± 30.1 | 456.3 ± 18.0 | 481.9 ± 30.3 | 485.6 ± 25.4 |
| dengue | peak_intensity | 174.4 ± 50.9 | 198.0 ± 56.6 | 198.6 ± 17.1 | 229.4 ± 14.4 |
| influenza_japan | peak_timing | 4.87 ± 1.27 | 24.66 ± 2.49 | 26.30 ± 1.38 | 26.74 ± 1.89 |
| influenza_us-regions | peak_timing | 43.28 ± 13.91 | 55.04 ± 18.59 | 67.36 ± 10.43 | 67.76 ± 9.16 |
| influenza_us-states | peak_timing | 10.55 ± 1.01 | 22.36 ± 4.16 | 30.59 ± 2.39 | 27.69 ± 2.52 |
| dengue | peak_timing | 10.91 ± 0.84 | 18.35 ± 3.19 | 28.66 ± 3.04 | 31.98 ± 1.13 |

**Peak timing is weak in absolute terms and this should not be understated.** Only
influenza-Japan h3 (4.87 steps) is within a plausibly operational range. Influenza-US-regions is
43–68 steps off across all horizons. On a weekly panel these are not near-misses; the model is not
locating the peak. Peak intensity likewise runs to thousands of counts on the influenza panels.
These figures are reported here for the first time and they weaken, rather than support, any
operational-readiness claim.

**Cross-disease transfer effect (LDO adapted vs single, improvement %, paired by seed).** Bold = the
95% t interval excludes zero.

| panel | metric | h3 | h5 | h10 | h15 |
|---|---|---|---|---|---|
| influenza_japan | peak_intensity | +20.1 | −4.5 | −18.6 | −20.9 |
| influenza_us-regions | peak_intensity | −4.7 | −10.4 | −31.1 | −39.0 |
| influenza_us-states | peak_intensity | +9.4 | +4.2 | +3.6 | −1.0 |
| dengue | peak_intensity | −17.7 | −22.6 | **−30.3** | **−15.5** |
| influenza_japan | peak_timing | +10.7 | **+24.7** | +8.9 | −10.3 |
| influenza_us-regions | peak_timing | −19.7 | −49.3 | −9.9 | −26.2 |
| influenza_us-states | peak_timing | +7.0 | +14.4 | −0.9 | −0.5 |
| dengue | peak_timing | −24.2 | +15.3 | **+26.5** | +9.1 |

Unbolded cells are within noise.

**This is the one place a positive cross-disease effect clears the noise floor.** Two cells do so:
influenza-Japan h5 peak timing (+24.7%) and dengue h10 peak timing (+26.5%). Across the 32 peak cells
(2 metrics × 4 panels × 4 horizons) the full tally is **2 significantly positive, 2 significantly
negative, 28 within noise**. Two isolated positives out of 32 tests is consistent with multiple
comparisons and **must not be promoted to a finding**. It is recorded because suppressing it would
be selective reporting, not because it supports the transfer claim.

Peak intensity shows no positive clearing cell anywhere, and is significantly negative on dengue at
both long horizons.

### 4.6 The reference artifact — full reconciliation of the Week-3 discrepancies

Holding the transfer prediction fixed and varying only the reference:

- **Reference A — seed-matched.** Transfer at seed *s* against single-disease at the same seed *s*.
- **Reference B — five-seed mean.** The same transfer run against the mean of all five seeds.

| panel | h | Ref A | Ref B | client's independent arithmetic |
|---|---|---|---|---|
| influenza_us-regions | h3 | +19.9% | **+9.9%** | "613 to 553, which is a 9.8% improvement, but the table says +20%" |
| influenza_us-states | h3 | +5.8% | **+1.4%** | "113 to 112, which is under 1%, but says +6%" |
| influenza_japan | h5 | +4.5% | **+7.4%** | "841 to 779, about 7.4%, but says +4%" |
| dengue | h3 | +2.9% | **−15.4%** | "dengue h3 and h5 look sign inverted" |

**All four review observations are reproduced exactly.** Our five-seed means are 613.2, 113.4 and
841.1 against the client's 613, 113 and 841; our Reference B figures are +9.9%, +1.4% and +7.4%
against their 9.8%, "under 1%" and 7.4%. The client was computing Reference B throughout; the
Week-3 table was printing Reference A without labelling it. The two tables were never readable
against each other, exactly as reported.

The dengue case is the most consequential because the reference choice **inverts the sign**. The
cause is identifiable: single-disease dengue seed 42 scores 49.98 RMSE against 37.47, 47.13, 38.65
and 37.08 for the other four seeds. Seed 42 is an outlier, and a seed-matched delta against an
unusually weak reference reports an improvement that the five-seed mean contradicts. This is a
reference-selection artifact, not a defect in the delta formula. It is independently corroborated:
`analysis.py` carries a self-check asserting dengue h3 reads −15.4%, derived through a separate code
path, and the two agree.

**Consequence.** Every delta in this report states its reference on the table. No transfer figure
should be quoted without one.

### 4.7 Zero-shot transfer

Cross-disease zero-shot (LDO zero-shot) **fails on influenza**. Paired deltas against single-disease
RMSE reach −311.2% at influenza-Japan h3 and −558.8% at h10; that is, the un-adapted encoder produces
errors several times the single-disease baseline. Most cells carry intervals so wide they are
formally within noise, which should be read as instability rather than acceptability.

Dengue zero-shot degrades far more gracefully (all four horizons within noise), consistent with
dengue's larger node count providing a better-conditioned trunk.

Population-transfer zero-shot (LODO) is far less severe, but is a single seed.

**Implication.** The adapter is not a refinement on top of a transferable representation. Without
adaptation the cross-disease representation does not transfer to influenza at all. Since the Ebola
deployment is an adaptation step, this is the correct architecture — but it means the quality of the
Ebola result depends entirely on the adaptation data, which is the subject of §6.

---

## 5. Threats to validity

**LODO rests on one seed.** Every leave-one-dataset-out figure is seed 42. Given that seed 42 is
demonstrably an outlier on single-disease dengue (§4.5), single-seed LODO figures should not be
quoted as results. They are retained because the client requested both fold structures side by side.

**The LDO influenza fold has four seeds, not five.** Seed 42 is absent from the dengue→flu direction,
so those paired deltas rest on a four-seed overlap (t critical 3.18). The dengue fold has all five.

**No uncertainty quantification.** WIS, CRPS, coverage and PIT are implemented and self-checked but
unusable until runs archive quantile predictions. The calibrated-uncertainty claim is currently
unevidenced. This is the largest single gap against the review's metric requirements.

**`nrmse` is missing for LODO zero-shot.** Zero-shot runs write JSON only — `train/lodo.py` discards
the per-node arrays — so the backfill, which operates on `*__pernode.npz`, structurally cannot reach
them. The newer LDO zero-shot runs are unaffected, having recorded `nrmse` natively. Recovering it
for LODO zero-shot requires a rerun, not a rescore.

**Wide intervals on small folds.** Influenza-US-regions has ten nodes; several LDO intervals exceed
±30 percentage points. Where a cell reads "within noise," that frequently reflects limited power
rather than a demonstrated absence of effect, and the two must not be conflated.

**Joint `sqrt-uniform` is a single seed** and is not a confirmed arm.

**Peak-metric deltas can rest on fewer seeds.** A `peak_timing` reference cell of exactly zero has no
scale to divide by, so that seed is dropped from the pairing rather than producing an infinite
percentage. A peak_timing delta may therefore carry a smaller `n` than the same cell's RMSE delta;
the `n` is printed on every cell in `progress/outcomes/Results_Matrix.md` (superseded).

**Two isolated positive peak-timing cells** (§4.5) sit inside 32 peak tests of which 28 are within
noise. They are reported for completeness and are not treated as evidence of transfer.

---

## 6. Conclusions and implications for the Ebola case study

**The corrected fold structure does not support a positive cross-disease transfer claim.** This is the
primary result. The paper cannot presently claim that pre-training across diseases improves forecasting
on a held-out disease; on this evidence it degrades it, significantly so in twelve of sixteen cells and
most severely at long horizons.

**The reported Week-3 headline was substantially a reference artifact**, and the residue is population
transfer rather than disease transfer. Both of the client's Week-3 concerns are confirmed in full.

**Three implications for Ebola.**

The transfer effect is weakest at long horizons on every panel, and the Ebola audit established that
h10 and h15 have **zero adaptation examples** — they are necessarily zero-shot. §4.6 shows
cross-disease zero-shot failing outright on influenza. Long-horizon Ebola forecasting is therefore
being attempted in the single regime where this method is least supported by evidence.

The practical Ebola adaptation horizon is h3, on 16 origin-district pairs across 8 districts. Adaptation
is load-bearing (§4.6) and that is very little of it.

Because Ebola is scored exactly once against a pre-registered configuration, a weak result will not be
separable from insufficient adaptation data. This should be stated before the run, not after.

**Recommended next steps.** Complete the five-seed LODO arm so the population-transfer claim is
testable rather than anecdotal; archive quantile predictions so the uncertainty metrics can be
populated; and treat the meta-learning decision as directly relevant rather than parallel — the
freeze-then-adapt result reported here is the ablation baseline against which episodic meta-training
would have to demonstrate value, and it is currently a negative result.

---

## Appendix A. Reproduction

```
python score.py                         # scoring self-check, 7 metrics
python rescore_encoder.py --dry-run     # verify mask reconstruction (65/65)
python rescore_encoder.py               # backfill nrmse
python results_matrix.py --selfcheck    # sign, units, noise screen, guards
python results_matrix.py -o Results_Matrix.md
```

The matrix this document was written against, `progress/outcomes/Results_Matrix.md`, was superseded
on 2026-09-07 and is frozen for provenance. Note that `progress/outcomes/Results_Matrix.md` is also
the default `-o` path, so send a fresh run somewhere else if you do not want to overwrite the frozen
copy.

`results_matrix.py --selfcheck` verifies error-metric sign in both directions, pcc reported in points
and stable near zero, the noise screen both accepting and rejecting, seed-intersection pairing,
zero and NaN denominators dropped, small-sample t values, single-seed honesty, absence of scientific
notation, the like-for-like guard firing on unequal node counts, distinct routing of all six regime
prefixes, and that Reference A and Reference B can disagree in sign.

## Appendix B. Coverage

| regime | flu-japan | flu-us-regions | flu-us-states | dengue |
|---|---|---|---|---|
| single | 5 | 5 | 5 | 5 |
| joint:uniform-uniform | 5 | 5 | 5 | 5 |
| joint:sqrt-uniform | 1 | 1 | 1 | 1 |
| LODO adapted | 1 | 1 | 1 | 1 |
| LODO zero-shot | 1 | 1 | 1 | 1 |
| LDO adapted | 4 | 4 | 4 | 5 |
| LDO zero-shot | 4 | 4 | 4 | 5 |

Seeds drawn from {42, 52, 62, 72, 82}; the LDO influenza fold omits seed 42.

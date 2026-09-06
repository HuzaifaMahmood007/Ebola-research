# The Shrinkage Test: Verdict

**Run 2026-09-02. Verdict recorded 2026-09-04.**

**The script that produced these numbers was never saved.** A search of the working tree, the git
history and the repository archive finds the header strings of the two output logs in those logs and
nowhere else, so the probe was written ad hoc, run once, and lost. The numbers are therefore not
regenerable without rewriting it.

That is why the tables below are reproduced **in full** rather than cited. The source logs
(`Reports/shrinkage_test.log` and `Reports/shrinkage_honest.log`, duplicated under
`results/reports/`) match `*.log` in `.gitignore` and so have no version history either; if they are
deleted, this document is the only surviving record of the experiment.

Rebuilding the probe is a small job and worth doing only if someone wants to extend it: it needs the
archived count-space quantiles at `results/single/encoder__{panel}__seed{S}__quantiles.npz` (median
at quantile index 2), the raw counts and test mask from `bundles`, the per-node training mean, and
`score.py`'s country-macro aggregation. The verdict below does not depend on that rebuild.

---

## Why this test existed

Three symptoms pointed one way:

- the model loses to `train_mean` on COVID and on dengue at h15,
- 90% intervals cover about 50%,
- per-node scaling loses to blunter pooled scaling at long horizons.

The proposed chain was: per-node scaling amplifies noise on small quiet nodes, so the model learns
that noise is signal, so it predicts more variation than it can justify, giving over-jumpy point
forecasts and too-narrow intervals. **It was a hypothesis, not a finding.**

The test shrinks each prediction toward the per-node training mean,
`new = mean + lam * (pred - mean)`, and sweeps `lam`. `lam = 1` is the shipped model and `lam = 0` is
`train_mean`. The rule was fixed before the run, and is quoted verbatim from the log:

> VERDICT RULE: best lam well below 1 and falling with horizon supports the over-commitment
> diagnosis. Best lam ~1 refutes it and points at Test 2 (bias, not variance).

---

## Verdict

**The over-commitment diagnosis is NOT supported, and shrinkage is not a usable patch.** Three of
five panels want `lam > 1`, which is the opposite of shrinkage. The one headline number that appeared
to support the diagnosis is an artefact of averaging across panels that disagree in direction, which
is a thing this project's own reporting rules forbid.

Concretely, against the pre-stated rule:

| panel | best lam, h3 → h15 | reading |
|---|---|---|
| dengue | 0.74, 0.58, 0.54, 0.42 | below 1 and falling, **supports** the diagnosis |
| influenza_japan | 1.27, 1.30, 1.30, 1.30 | **pinned at the grid ceiling**, wants more variance |
| influenza_us-regions | 1.16, 1.20, 1.05, 1.02 | above 1 |
| influenza_us-states | 1.25, 1.21, 1.07, 1.04 | above 1 |
| covid_us-states | 0.50, 0.00, 0.00, 0.00 | **pinned at the grid floor**, degenerate (see below) |

Only dengue behaves as the hypothesis predicted. One panel of five is not a mechanism.

### Three things the log's own summary line got wrong

**1. "Mean best lam falls with horizon, 0.98 → 0.76" is an average across datasets.** The project's
standing rule is that dengue's 7,165 nodes and us-regions' 10 never enter the same mean. Applying
the rule dissolves the trend: the fall is produced by COVID sinking to 0.00 while three influenza
panels sit above 1.0, and averaging a floor against a ceiling produces a number that describes
neither.

**2. COVID's `lam = 0` is not a variance result at all.** At `lam = 0` the prediction *is*
`train_mean`, and the log's own "RMSE at best" for COVID at h5/h10/h15 equals its `train_mean` column
to the digit (5455.71, 5302.31, 5275.22). So the finding is "on COVID the training mean beats the
model", which is already documented in `Report_Covid.md` and is a consequence of the Omicron
structural break sitting between the validation and test folds. It says nothing about whether the
model over-commits variance, and since COVID supplies the entire mean gain, it should not be counted
as evidence for shrinkage anywhere.

**3. influenza_japan is clipped.** Its optimum sits at the grid ceiling of 1.30 at every horizon, so
the true optimum is outside the searched range and those four cells are unmeasured rather than
measured. Any rerun should widen the grid.

---

## The honest split, which is the number that decides it

The oracle sweep above picks `lam` on the same data it scores, so its gains are hindsight. The second
run fits `lam` on the early half of the test fold and scores it on the late half.

**Fitted `lam` helps in 10 cells, hurts in 8, and is flat in 2, of 20. Mean gain +6.0%, median
+0.5%, worst −32.7%, best +71.5%.**

The mean is carried entirely by COVID. Strip COVID out, as point 2 above says we must, and the
remaining sixteen cells are: small gains (+3.5, +2.9, +2.8, +2.1, +1.1, +1.0, +0.1) against large
losses (−32.7, −20.6, −13.3, −6.2, −4.4, −3.9, −3.1, −2.5, −0.4). **Dengue at h3 loses 32.7% with a
fitted `lam` of 1.70**, meaning dengue's short horizon wants substantially *more* variance, not less.

Against the pre-stated rule ("fitted beats lam=1 → real and exploitable, a no-retrain patch;
fitted ≈ lam=1 → hindsight only, a diagnostic not a fix"), this is the second case everywhere except
COVID, and on COVID it is a restatement of a known fold-boundary defect.

---

## What this changes

- **Do not apply shrinkage.** It is not a no-retrain accuracy patch. Recorded so the test is not run
  a third time.
- **The standing over-commitment diagnosis moves from "hypothesis to test" to "tested, not
  supported".** `CLAUDE.md` section 6 and `Resume.md` section 6 both still describe it as a live
  hypothesis with the shrinkage sweep as its untried first test, and both still list running that
  test as next action #1. Both are stale and are corrected alongside this note.
- **The evidence now points at bias rather than variance**, which is exactly where the pre-stated
  rule said a refutation should point. Two independent signs agree: the panels that reject shrinkage
  want `lam > 1`, i.e. their point forecasts sit too close to the node mean and under-reach the
  peaks; and the median-to-mean correction already built at `train/loop.py:84` measurably helps the
  same short horizons on the same panels (dengue h3 −16.2%, influenza_japan h3 −10.5%,
  influenza_us-regions h3 −11.5% RMSE). The pinball head predicts a log-space median while RMSE is
  minimised by the count-space mean, and on right-skewed counts that gap is a low point forecast, not
  an over-dispersed one.
- **The three symptoms still need an explanation**, and it is now a different one. Interval coverage
  in particular is untouched by this result: shrinking a point forecast says nothing about interval
  width, so the 90%-covers-50% problem remains open and is a calibration question, addressed by the
  conformal work rather than by scaling.

**What must not be claimed from this:** that per-node scaling is vindicated. This test refutes one
specific mechanism, over-commitment of variance. The separate, measured finding that pooled scaling
beats per-node scaling at long horizons on the matched influenza-US-states panel still stands and
still belongs in Threats.

---

## The tables, reproduced in full

### Oracle sweep: `lam` chosen on the same data it is scored on

`lam` over [0.0, 1.3]. `lam=1` is the shipped model, `lam=0` is `train_mean`.

| panel | h | model (lam=1) | best lam | RMSE at best | gain | train_mean (lam=0) |
|---|---|---|---|---|---|---|
| dengue | 3 | 41.98 | 0.74 | 40.81 | +2.8% | 57.08 |
| dengue | 5 | 49.95 | 0.58 | 47.03 | +5.8% | 57.06 |
| dengue | 10 | 56.21 | 0.54 | 55.10 | +2.0% | 57.02 |
| dengue | 15 | 57.38 | 0.42 | 56.56 | +1.4% | 56.97 |
| influenza_japan | 3 | 739.72 | 1.27 | 706.29 | +4.5% | 1236.47 |
| influenza_japan | 5 | 841.10 | 1.30 | 775.57 | +7.8% | 1390.62 |
| influenza_japan | 10 | 1064.01 | 1.30 | 1009.36 | +5.1% | 1548.32 |
| influenza_japan | 15 | 1031.14 | 1.30 | 988.83 | +4.1% | 1511.87 |
| influenza_us-regions | 3 | 611.35 | 1.16 | 596.42 | +2.4% | 1159.78 |
| influenza_us-regions | 5 | 728.28 | 1.20 | 713.43 | +2.0% | 1155.45 |
| influenza_us-regions | 10 | 795.04 | 1.05 | 794.28 | +0.1% | 1144.20 |
| influenza_us-regions | 15 | 814.55 | 1.02 | 814.43 | +0.0% | 1134.41 |
| influenza_us-states | 3 | 113.34 | 1.25 | 107.94 | +4.8% | 205.47 |
| influenza_us-states | 5 | 136.80 | 1.21 | 134.06 | +2.0% | 204.46 |
| influenza_us-states | 10 | 151.49 | 1.07 | 151.25 | +0.2% | 202.85 |
| influenza_us-states | 15 | 154.19 | 1.04 | 154.13 | +0.0% | 201.41 |
| covid_us-states | 3 | 5468.22 | 0.50 | 4817.40 | +11.9% | 5524.64 |
| covid_us-states | 5 | 8375.55 | 0.00 | 5455.71 | +34.9% | 5455.71 |
| covid_us-states | 10 | 12240.66 | 0.00 | 5302.31 | +56.7% | 5302.31 |
| covid_us-states | 15 | 11762.33 | 0.00 | 5275.22 | +55.2% | 5275.22 |

Cells where shrinking helps at all (best lam < 1.00): **8 of 20**. Mean RMSE gain from optimal
shrinkage: +10.2%, max +56.7%. Both figures are hindsight and both are dominated by COVID.

### Honest split: `lam` fitted on the early half of test, scored on the late half

| panel | h | nE/nL | lam=1 | lam_fit | fitted | gain | oracle lam | oracle headroom |
|---|---|---|---|---|---|---|---|---|
| dengue | 3 | 315/315 | 45.15 | 1.70 | 59.92 | −32.7% | 0.75 | +2.8% |
| dengue | 5 | 315/315 | 53.95 | 1.55 | 65.05 | −20.6% | 0.59 | +5.8% |
| dengue | 10 | 315/315 | 61.01 | 0.87 | 60.42 | +1.0% | 0.53 | +2.1% |
| dengue | 15 | 315/315 | 62.37 | 0.08 | 61.68 | +1.1% | 0.40 | +1.5% |
| influenza_japan | 3 | 52/52 | 945.11 | 0.75 | 1071.27 | −13.3% | 1.51 | +12.2% |
| influenza_japan | 5 | 52/52 | 1007.73 | 1.06 | 979.66 | +2.8% | 1.67 | +17.3% |
| influenza_japan | 10 | 52/52 | 856.73 | 1.69 | 883.14 | −3.1% | 1.29 | +4.9% |
| influenza_japan | 15 | 52/52 | 833.90 | 1.61 | 866.69 | −3.9% | 1.24 | +3.6% |
| influenza_us-regions | 3 | 117/118 | 569.30 | 1.09 | 549.40 | +3.5% | 1.23 | +5.6% |
| influenza_us-regions | 5 | 117/118 | 684.80 | 1.12 | 664.73 | +2.9% | 1.28 | +4.4% |
| influenza_us-regions | 10 | 117/118 | 777.78 | 1.05 | 776.89 | +0.1% | 1.05 | +0.1% |
| influenza_us-regions | 15 | 117/118 | 766.80 | 0.96 | 769.58 | −0.4% | 1.07 | +0.2% |
| influenza_us-states | 3 | 54/54 | 127.33 | 1.05 | 124.68 | +2.1% | 1.32 | +7.1% |
| influenza_us-states | 5 | 54/54 | 152.52 | 0.92 | 156.35 | −2.5% | 1.33 | +4.5% |
| influenza_us-states | 10 | 54/54 | 170.44 | 0.74 | 181.09 | −6.2% | 1.26 | +2.1% |
| influenza_us-states | 15 | 54/54 | 178.05 | 0.76 | 185.95 | −4.4% | 1.22 | +1.3% |
| covid_us-states | 3 | 24/25 | 5219.99 | 0.64 | 4530.89 | +13.2% | 0.42 | +15.8% |
| covid_us-states | 5 | 24/25 | 7788.74 | 0.00 | 4536.33 | +41.8% | 0.00 | +41.8% |
| covid_us-states | 10 | 24/25 | 12952.28 | 0.00 | 3693.78 | +71.5% | 0.00 | +71.5% |
| covid_us-states | 15 | 24/25 | 12683.96 | 0.00 | 4025.37 | +68.3% | 0.00 | +68.3% |

Note that the oracle `lam` differs between the two runs for the same panel and horizon, because the
second run's oracle is fitted on a half fold rather than the whole one. Where those two disagree in
direction, and influenza_japan h3 is 1.27 in the first table, 1.51 in the second and 0.75 fitted,
that cell is unstable and should not be read as a measurement in either direction.
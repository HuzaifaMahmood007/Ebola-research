# Epi-informed component: bound and lambda test protocol

I commit this file before the first new epi record exists. `ablation/run_epi_ablation.py` refuses to
train until this file is committed with no local edits, and stamps its sha256 into every record it
writes. Once committed this file is frozen. Any later change goes in a separate amendment file,
dated, and written before the data it affects, so the hash stamped into every record keeps matching
this file.

Nothing here writes to `results/ebola/`. Every new record lands in
`experiments/epi_bound_lambda/single/`. The 40 released lambda-1 records in `ablation/single/` are
read as reference and never modified.

---

## 1. The question

The epi component is a soft penalty on epidemiologically implausible growth, added to the TRAIN
objective only. The forward pass `pred = ad(enc(...))` is byte-identical with and without it, so this
is a question about the objective, not about the architecture.

    r = ( log1p(y_b) - log1p(y_a) ) / (b - a)          log1p units per week
    penalty = mean( relu(|r| - r_max[gap])^2 )
    loss = pinball_loss + lambda * penalty

Two bounds have already run at lambda 1 and both returned null. I established from disk why, and it
is not a mystery. At p90max the mean train penalty is 2.2e-05 to 6.5e-04 against a pinball loss of
order 0.1 to 1, so the term is worth roughly 0.005 to 0.1 percent of the objective. It is about a
thousand times too weak to steer the model. At p99max it is worse: the hinge is identically zero on
both US influenza panels across all five seeds, so those ten cells were the baseline retrained.

Two things therefore change together, and the protocol has to keep them separable:

1. **The bound.** `max` across datasets selects COVID at every gap, because COVID sits 1.5x to 3x
   above every other panel. `ablation/epi_rmax.json` shows the aggregated r_max equal to COVID's own
   per-gap quantile at all three gaps. So r_max was a COVID statistic, which is why p99 was switched
   off by construction on the flu panels.
2. **Lambda.** Whatever the bound, a term at 0.1 percent of the objective cannot be expected to move
   anything, so a null at lambda 1 is a statement about lambda.

## 2. The bound, fixed before any run

The aggregator is the lever, and the bound stays ONE shared number across diseases. A per-dataset
r_max would smuggle disease identity back into the trunk and break the disease-agnostic claim the
paper rests on, so that is not a legal move. Legal moves are the aggregator and the quantile.

**Chosen: quantile 0.99, aggregator median.** r_max = 1.0397 at gap 2, 0.7838 at gap 3, 0.5666 at
gap 5, calibrated over all five dev datasets.

Why median rather than max or min. All three produce one shared number, so all three are legal, and
the question is which single panel's statistic the number effectively becomes. `max` is driven
entirely by the most extreme panel, COVID at every gap. `min` is driven entirely by whichever panel
sits lowest, dengue at gaps 3 and 5 and `influenza_us-regions` at gap 2. `median` does not move when
one panel is an outlier, and that robustness is the property a disease-agnostic bound needs. Only the aggregator changes; the principled p99 quantile stays.

It is also a strictly better operating point than the p90max arm already on record, which I did not
expect and which contradicts the direction warning I was handed:

| bound | r_max g2 / g3 / g5 | real transitions called implausible | model intervals the hinge touches |
|---|---|---|---|
| p99 max (released) | 2.0355 / 1.8798 / 1.5097 | 0.177 % | 0.017 % |
| p90 max (released) | 0.8047 / 0.7168 / 0.6208 | 4.936 % | 1.279 % |
| **p99 median (new)** | **1.0397 / 0.7838 / 0.5666** | **4.157 %** | **1.626 %** |
| p99 min | 0.8704 / 0.6931 / 0.5040 | 6.029 % | 2.615 % |
| p95 median | 0.6119 / 0.4806 / 0.3822 | 12.426 % | 7.191 % |

p99 median both fires more than p90max and calls fewer real transitions implausible. The reason is
the pair structure: `PAIRS` is `((0,3),(3,5),(5,10),(10,15))`, so gap 5 carries two of the four
constrained intervals and gap 2 only one. p99 median is looser than p90max at gaps 2 and 3 but
tighter at gap 5, which is half of all intervals. Aggregate tightness is not a per-gap comparison.

p95 median was rejected despite its much larger surface. Calling 12.4 percent of real observed
transitions implausible is a ceiling live diseases routinely cross, and preventing exactly that is
the justification the `max` rule existed for. Lowering the quantile that far discards the
justification rather than improving the rule.

**When the bound was chosen, and why that is not peeking.** The two columns above come from
`ablation/epi_penalty.py --sweep`, which reads only the empirical distribution of real transitions in
TRAINING cells and the violation rate of the already-released baseline model's quantile archives, at
the sweep's default seed 42 and scored over the four trained panels. One statistic per column: the
first draft quoted a 5-seed pooled 7.318 percent in the p95 median row against seed-42 values
everywhere else, which is the estimand mismatch this project has already been burned by once. It
trains nothing and reads no test score of any arm in this protocol. This is the same structure as the
V2 protocol's stage 1, which computed validation spreads and explicitly never computed their mean.

## 3. The arms

Six, all on the four dev panels except where stated. Dengue is excluded: about 12.7 h for five seeds,
96 percent of the job, and gated per section 8.

| arm | tag | panels | cells | r_max g2 / g3 / g5 |
|---|---|---|---|---|
| p90 max, lambda 10 | `epi_p90max_lam10` | 4 | 20 | 0.8047 / 0.7168 / 0.6208 |
| p90 max, lambda 100 | `epi_p90max_lam100` | 4 | 20 | 0.8047 / 0.7168 / 0.6208 |
| p99 median, lambda 1 | `epi_p99median` | 4 | 20 | 1.0397 / 0.7838 / 0.5666 |
| p99 median, lambda 100 | `epi_p99median_lam100` | 4 | 20 | 1.0397 / 0.7838 / 0.5666 |
| p99 max, lambda 10 | `epi_p99max_lam10` | COVID only | 5 | 2.0355 / 1.8798 / 1.5097 |
| p99 max, lambda 100 | `epi_p99max_lam100` | COVID only | 5 | 2.0355 / 1.8798 / 1.5097 |

90 cells. Seeds are 42, 52, 62, 72, 82 in every arm, matching `train.loop.SEEDS`.

Lambda 1 is not retrained at p90max or p99max: those are the released `epi_p90max` and `epi_p99max`
records, which the runner reads from `ablation/single/`. Each bound therefore has a three-point
lambda ladder at 1, 10 and 100, with lambda 1 free at two of the three bounds.

The p99max arms are COVID only on purpose. The hinge is identically zero on both US influenza panels
at that bound across all five seeds, so lambda times zero is zero and those cells cannot move by
arithmetic. Japan fires at about 1e-7 there, which even at lambda 100 leaves the term near 1e-5,
three orders of magnitude under the gate in section 6. Running them would buy 30 guaranteed nulls for
2.9 h of GPU. COVID is the only panel where p99 is live.

**Model selection stays on val pinball in every arm.** The penalty enters the train objective only.
Changing the early-stopping criterion as well as the loss would make the arms differ in two ways at
once and nothing could be attributed.

## 4. Panel, seeds, scoring

Reference is the single-disease record at the SAME seed, `results/single/encoder__<ds>__seed<S>.json`,
model `encoder`, field `node_mean`. Deltas are paired per seed on the seed intersection, as
`paired_delta()` already does. An arm mean against a baseline mean is the reference mismatch the
client caught in Week 3 and it is not used here.

RMSE and MAE decide. PCC is printed and never decides. `encoder_mc`, the bias-corrected forecast, is
printed and never decides.

Each record additionally carries `epi_quantile`, `epi_aggregator`, `epi_lam`, `epi_r_max`,
`epi_penalty_mean`, `epi_pinball_mean`, `epi_penalty_share` and `epi_protocol_sha256`. Before this
protocol the released records recorded their bound only in the filename.

## 5. Order of work

1. `--selfcheck` and `--mutate-selfcheck` on the runner. The mutation test must report 3 of 3 planted
   bugs caught, or the tag assertions are decoration.
2. Commit this file. The runner refuses to train while it is missing or has uncommitted edits.
3. One cell first, COVID seed 42 at `epi_p99median_lam100`, about 1.5 min: the ten released COVID
   epi cells took 1.1 to 1.5 min each. It is a real cell of the grid, not a throwaway, and it
   confirms end to end that the record carries a non-zero `epi_penalty_share` of roughly 100x the
   lambda-1 value.
4. The remaining 89 cells, one chained command, about 6 h measured from the per-cell times in the
   two released logs, `results/reports/epi_p90max.log` and `results/reports/epi_p99max.log`.
   Per-cell time across them ranges 1.0 to 26.7 min because early stopping bites at different
   epochs, so the night may overrun. The runner skips finished cells, so the chain is restartable.
5. `--report` per arm, then the result document and its verifier.

## 6. The decision rule, and the inertness gate

**The rule.** For each cell (arm, panel, horizon, metric), d is arm minus reference at the same seed
over the five seeds. A cell counts only when the paired two-sided 95 percent t-interval of d excludes
zero: entirely below zero is better for RMSE and MAE, entirely above is worse. At n = 5 the critical
value is t(0.975, 4) = 2.7764.

This replaces the rule the released epi reports used, "within noise unless |mean d| >= sd d", which is
`run_epi_ablation.py:196`. That rule is equivalent to a paired t of at least sqrt(n), so about p =
0.09 at five seeds rather than 0.05, and its bar does not shrink as n grows. Across the 48 printed
cells of one released arm (4 panels x 4 horizons x 3 metrics, PCC included) it yields about 4.3
false flags by chance; the released p90max report produced three, below that expectation, and only
one of the three sits on a deciding metric. The new arms add 144 deciding comparisons (RMSE and MAE
only), which read raw under the old rule would carry roughly 13 spurious flags. The old rule is
printed beside every cell and carries no weight.

**The inertness gate, at 1 percent.** For every cell the runner records
`epi_penalty_share = lambda * mean_train_penalty / mean_train_pinball`. A panel whose share stays
under 0.01 has its null reported as a statement about lambda, NOT about the epi component. This is
pre-registered because the outcome is partly predictable: at measured p90max lambda-1 shares of 0.005
to 0.1 percent, lambda 100 should clear 1 percent on Japan and COVID and sit borderline at roughly 0.2
to 2 percent on the two US panels. That asymmetry must be disclosed rather than presented as a clean
null, and the threshold must be fixed before the numbers arrive or it becomes an excuse.

A cell whose penalty is identically zero is a separate and stronger case: it is the baseline
retrained, and it is reported as an inert penalty, never as "no effect of the epi component".

**What this design can and cannot detect.** The sweep also told me something the arms cannot fix, and
it belongs here rather than in the discussion of the results. On `influenza_us-regions` the trained
baseline exceeds no defensible bound at all: zero violations of its 9,400 constrained intervals in
every seed at p99 median, where its worst prediction reaches only 0.53 to 0.71x the bound; zero at
p99 min; and at most one per seed even at p95 median, where the worst prediction still only reaches
0.79 to 1.05x. On `influenza_us-states` it is 5 to 14 of 21,168, about 0.05 percent. At p99 median,
4,445 of the 4,596 violations across the four panels are `influenza_japan`, which is 96.7 percent.

So the aggregate "model intervals the hinge touches" column is a Japan statistic, in the same way
r_max under `max` was a COVID statistic. Two of the four panels are structurally near-inert at any
bound that keeps the plausibility justification. The gate above is what makes that reportable instead
of being read as four independent nulls.

One caveat on reading the probe: it measures the FINAL trained model on test origins, while
`mean train penalty` averages over all epochs including early ones where the model predicted wildly.
That is why p90max recorded a non-zero penalty on us-regions (2.3e-05) where the probe says zero. The
probe is a lower bound on how much the hinge fires during training, and it says the penalty acts early
and then goes quiet.

## 7. Pre-committed criteria

- **(a) PRIMARY.** An arm passes when, at some horizon, the t-interval lies entirely below zero on
  BOTH RMSE and MAE for that panel, AND no horizon on that panel shows the arm significantly worse on
  either metric, AND that panel's `epi_penalty_share` is at least 0.01. All three, on one panel, at
  one horizon.
- **(b) ATTRIBUTION.** Where (a) passes, the same panel and horizon is checked against the other
  lambda at the same bound and the other bound at the same lambda, to say whether the gain tracks
  lambda, the bound, or neither.
- **(c) FAIL.** (a) is not met and the share is at or above 0.01 on the panel in question. The term
  was a real fraction of the objective and still did not help.
- **(d) INCONCLUSIVE.** (a) is not met and the share is under 0.01. The term was too weak to steer
  training, so the arm tests lambda rather than the component.

Multiplicity is handled structurally, not by an alpha correction. The conjunction of two metrics at
one horizon plus the no-harm clause across all horizons means extra cells are a liability rather than
extra chances to win, which is the right shape for 144 comparisons. That is the same treatment the V2
protocol used.

| outcome | condition | what we may say | what we may not say |
|---|---|---|---|
| PASS | (a) met | The epi-informed penalty improves accuracy on that panel at that horizon, at the stated bound and lambda | That it helps in general, on other panels, or on Ebola. That the epi prior is validated |
| FAIL | (c) | At a bound and lambda where the penalty was a real fraction of the objective, it did not improve accuracy | That the epi prior is wrong in principle. That no bound could work |
| INCONCLUSIVE | (d) | At this bound the penalty is too weak to test at the lambdas we ran; the null is about lambda | That the component does not help. That the component is inert on Ebola |

Nothing else decides. PCC, `encoder_mc`, the old gate-rule column, and any subgroup are printed for
the reader and carry no verdict.

## 8. Gates for later steps

- **Dengue**, about 12.7 h for five seeds, only on PASS and only under its own dated amendment.
- **Ebola.** Nothing in this protocol touches `results/ebola/`. The frozen Ebola pre-registration is
  not reopened whatever the outcome. Any Ebola-side follow-up is exploratory, lives in
  `experiments/` or `ablation/`, and needs its own protocol.
- **Higher lambda.** If every arm lands INCONCLUSIVE, lambda 1000 is the next rung and needs its own
  amendment naming it before it runs. It does not get chosen after reading these numbers.
- **The manuscript.** No sentence about the epi component changes until the result document and its
  verifier are both committed.

## 9. After the data

This file does not change. `--report` reads the records from disk and refuses to decide on an arm
whose 20 cells (5 for the COVID-only arms) are not all present. The result document records this
file's sha256 and the commit that holds it.

## 10. Checks behind this protocol

- `ablation/epi_penalty.py --selfcheck` covers the penalty: it fires on rises and collapses, respects
  r_max, respects the mask, carries the scaler, and passes gradient.
- `ablation/run_epi_ablation.py --selfcheck` covers the comparison: pairing drops unmatched seeds,
  tags separate bounds and lambdas while lambda 1 stays backward compatible, and new arms write
  outside `ablation/`.
- `ablation/run_epi_ablation.py --mutate-selfcheck` plants three tag bugs and requires the selfcheck
  to fail on each.
- `diagnostics/verify_epi_bound_protocol.py` recomputes every number in this file from disk without
  importing the runner, and `--mutate` corrupts this file in memory and requires each corruption to be
  caught.

## 11. Reproduce, in order

```
conda run --no-capture-output -n ebola-train python ablation/epi_penalty.py --selfcheck
conda run --no-capture-output -n ebola-train python ablation/epi_penalty.py --sweep
conda run --no-capture-output -n ebola-train python ablation/epi_penalty.py --probe --quantile 0.99 --aggregator median --seeds 42 52 62 72 82
conda run --no-capture-output -n ebola-train python ablation/run_epi_ablation.py --selfcheck
conda run --no-capture-output -n ebola-train python ablation/run_epi_ablation.py --mutate-selfcheck
conda run --no-capture-output -n ebola-train python diagnostics/verify_epi_bound_protocol.py --mutate
```

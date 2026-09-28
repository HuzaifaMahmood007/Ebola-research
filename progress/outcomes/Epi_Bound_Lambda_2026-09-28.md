# Epi-informed component: bound and lambda test result

**Run and scored 2026-09-28.** This closes the pre-registered epi bound and lambda test. The protocol
was fixed and committed before any new number existed
(`progress/decisions/Epi_Bound_Lambda_Protocol.md`, sha256
`458184bc25d376c7fd59932f885b09ea04e3a3a6c29cbe9c8e5918ab7579b7f3`, commit
`b24063f16a691b502139d87e515972edf9f89551`). Every one of the 90 new records carries that sha256 in
its `epi_protocol_sha256` field.

**Verdict: no PASS in any of the 18 arm-panel units.** 6 FAIL, 12 INCONCLUSIVE. Where the penalty
became a real part of the objective, the six FAIL units (Japan and COVID at p90max lambda 10 and 100,
and at p99median lambda 100), it did not improve accuracy at any horizon on both error metrics. On Japan it made h10 significantly worse, in all four new arms.

---

## The question this answered

The epi component is a soft penalty on implausibly fast growth or collapse, added to the training
loss only. Two earlier arms at lambda 1 were null because the term was about 0.005 to 0.1 percent of
the objective, far too small to steer training, and the `max` bound was effectively COVID's own
number. This test asked two things:

1. Does a better disease-agnostic bound, p99 median instead of max, change the result? (Run 1)
2. With the bound held fixed, does raising lambda to 10 and 100 move results up or down? (Run 2)

## What ran

Six arms, 90 cells, seeds 42, 52, 62, 72, 82, every record under `experiments/epi_bound_lambda/single/`.
The released lambda-1 records in `ablation/single/` were read as reference and not modified.

| arm | panels | cells |
|---|---|---|
| `epi_p90max_lam10` | 4 | 20 |
| `epi_p90max_lam100` | 4 | 20 |
| `epi_p99median` | 4 | 20 |
| `epi_p99median_lam100` | 4 | 20 |
| `epi_p99max_lam10` | COVID only | 5 |
| `epi_p99max_lam100` | COVID only | 5 |

The log is `results/reports/epi_bound_lambda.log`: 90 "done in" lines, no errors.

## The verdict table

Rule, per protocol section 6: d is arm minus the released single-disease record at the same seed,
field `node_mean`, model `encoder`. A cell counts only when the paired two-sided 95 percent
t-interval of d excludes zero, t(0.975, 4) = 2.7764. PASS needs a win on BOTH RMSE and MAE at one
horizon, no significant harm on either metric at any horizon, and a share of at least 1 percent.
Share is `lambda * mean train penalty / mean train pinball`, averaged over the five seeds.

| arm | panel | share | win (both metrics) | harm | verdict |
|---|---|---|---|---|---|
| p90max lam10 | influenza_japan | 3.670 % | none | h10 RMSE, h10 MAE | FAIL |
| p90max lam10 | influenza_us-regions | 0.124 % | none | none | INCONCLUSIVE |
| p90max lam10 | influenza_us-states | 0.062 % | none | none | INCONCLUSIVE |
| p90max lam10 | covid_us-states | 3.244 % | none | none | FAIL |
| p90max lam100 | influenza_japan | 25.715 % | none | h10 RMSE, h10 MAE | FAIL |
| p90max lam100 | influenza_us-regions | 0.904 % | none | none | INCONCLUSIVE |
| p90max lam100 | influenza_us-states | 0.277 % | none | none | INCONCLUSIVE |
| p90max lam100 | covid_us-states | 30.654 % | none | none | FAIL |
| p99median lam1 | influenza_japan | 0.282 % | none | h10 RMSE, h10 MAE | INCONCLUSIVE |
| p99median lam1 | influenza_us-regions | 0.020 % | none | none | INCONCLUSIVE |
| p99median lam1 | influenza_us-states | 0.013 % | none | none | INCONCLUSIVE |
| p99median lam1 | covid_us-states | 0.432 % | none | none | INCONCLUSIVE |
| p99median lam100 | influenza_japan | 16.548 % | none | h10 RMSE, h10 MAE | FAIL |
| p99median lam100 | influenza_us-regions | 0.605 % | none | none | INCONCLUSIVE |
| p99median lam100 | influenza_us-states | 0.208 % | none | none | INCONCLUSIVE |
| p99median lam100 | covid_us-states | 17.521 % | none | none | FAIL |
| p99max lam10 | covid_us-states | 0.170 % | none | none | INCONCLUSIVE |
| p99max lam100 | covid_us-states | 0.762 % | none | none | INCONCLUSIVE |

Across the 144 deciding cells (RMSE and MAE), 8 are significantly worse, 4 significantly better, and
132 are within noise. All 8 worse cells are Japan h10. The 4 better cells never pair up on both
metrics at one horizon: Japan h3 MAE in three arms (its RMSE interval spans zero each time) and COVID
h15 RMSE at p99max lambda 100 (its MAE interval is [-811.475, +12.683]).

## The Japan h10 damage

| arm | share | h10 RMSE d [95% CI] | h10 MAE d [95% CI] |
|---|---|---|---|
| p99median lam1 | 0.282 % | +47.095 [+8.535, +85.655] | +19.377 [+4.144, +34.610] |
| p90max lam10 | 3.670 % | +127.751 [+17.560, +237.943] | +50.156 [+10.455, +89.857] |
| p99median lam100 | 16.548 % | +109.065 [+7.683, +210.447] | +44.741 [+1.146, +88.337] |
| p90max lam100 | 25.715 % | +99.732 [+26.275, +173.188] | +40.412 [+12.432, +68.391] |

The baseline h10 RMSE is about 1,063 averaged over seeds, so the largest damage is about 12 percent.

**I checked whether this is trainer drift rather than the penalty, and it is not.** By file date, four
of the five baseline Japan records date from 2026-08-11, and `train/loop.py` has changed since. But
the V2 repro check (`ablation/misc/v2_repro_check.json`) retrained Japan seed 42 with the trainer at
commit `39ecf34` and matched the released record exactly, max relative difference 0.0. The only
change to `train/loop.py` between `39ecf34` and the commit these arms ran under, `b24063f`, is 7
lines inside `if epi and train_mode:` that accumulate the pinball loss for the share; the baseline
path is untouched. I confirmed that by reading the diff, not by retraining. And seed 42 shows the
damage in every new arm:

| arm | s42 | s52 | s62 | s72 | s82 |
|---|---|---|---|---|---|
| p99max lam1, released, penalty about 1e-7 | -2.4 | -3.5 | -7.0 | +2.5 | +1.2 |
| p99median lam1 | +65.3 | +18.3 | +31.1 | +28.1 | +92.7 |
| p90max lam10 | +87.8 | +5.2 | +188.8 | +233.3 | +123.6 |
| p90max lam100 | +149.5 | +6.9 | +151.3 | +85.3 | +105.7 |
| p99median lam100 | +127.8 | -5.4 | +111.7 | +89.0 | +222.2 |

(Japan h10 RMSE, arm minus baseline, per seed.)

The first row is the control. A penalty of about 1e-7 barely moves the trained model, so a small
penalty does not scramble training into what is effectively a new random seed. The damage appears
once the penalty is non-trivial and stays there. That is a dose pattern a single unlucky baseline
cannot produce, because the near-inert arm is effectively the baseline and it shows nothing.

**The likely cause, a hypothesis.** The shared bound calls a large share of Japan's own real training
transitions implausible:

| bound | japan | us-regions | us-states | covid |
|---|---|---|---|---|
| p90 max | 9.97 % | 0.91 % | 1.49 % | 6.25 % |
| p99 median | 8.31 % | 0.83 % | 0.99 % | 5.97 % |

(Fraction of each panel's real observed TRAINING transitions above the bound, all three gaps pooled.)

So on Japan the penalty pushes against one real transition in ten to one in twelve, which is the fast
seasonal rise and fall of Japanese influenza. That is the failure the original `max` rule existed to
prevent: a ceiling a live disease routinely crosses is a bug, not a bound. It fits the damage landing
on Japan and not on the US panels, where the bound almost never binds. It does not explain why the
damage sits at h10 rather than h3 or h5, where Japan MAE is nominally better. That part is open.

## What is NOT established

- That the epi prior is wrong in principle. A per-panel bound would not misfire on Japan, but it is
  ruled out by design because it would carry disease identity into the trunk.
- That no bound in this family could help. We tested two bounds at up to three lambdas.
- Anything about Ebola. No Ebola record was read or written.
- The mechanism. The Japan false-positive rate is a candidate cause, not a tested one.

## The US panels

`influenza_us-regions` and `influenza_us-states` never reached the 1 percent gate, even at lambda 100
(highest seed-averaged share 0.904 percent, us-regions at p90max). They are INCONCLUSIVE in every
arm, as the sweep predicted before any GPU time: the trained baseline almost never exceeds any bound
that keeps the plausibility justification. For these two panels the null is a statement about the
panel and about lambda, not about the component.

## Deviations from the protocol, stated before a reader finds them

1. **`--report` does not apply the pre-registered rule.** The protocol's section 9 says the runner's
   `--report` decides and refuses to decide on an incomplete arm. It does neither: it still prints only
the old `|mean d| < sd d` rule, and it does not check completeness, which had no effect here because
all 90 records exist. I computed the
   verdicts above from the raw JSON records exactly as sections 6 and 7 state the rule, and
   `diagnostics/verify_epi_bound_doc.py` recomputes every one of them independently.
2. **The protocol does not say how the share is aggregated over seeds.** I used the mean over the five
   seeds. Under the minimum, no verdict changes. Under the maximum, two units move from INCONCLUSIVE to
   FAIL: us-regions at p90max lambda 100 (max 1.530 percent) and COVID at p99max lambda 100 (max
   1.324 percent). Neither reading produces a PASS.
3. **p99median lambda 1 Japan is INCONCLUSIVE by the frozen rule, but it shows significant harm.** Its
   share is 0.282 percent, under the gate, and it still made h10 significantly worse on both metrics.
   The gate's premise, that a term under 1 percent is too weak to steer training, is contradicted on
   Japan. The verdict stands as the rule gives it, and the harm is reported here rather than hidden
   behind the label.
4. **The released lambda-1 arms at p90max and p99max carry no share**, because the pinball
   denominator did not exist when they ran. They are the base of the two lambda ladders but are not
   units of this verdict table.

## What this closes

- **Run 1.** p99 median is a better-behaved bound than max on paper, one shared number that is not
  any single panel's statistic, and it did not change the outcome. At lambda 1 it was inert on three
  panels and harmful at Japan h10.
- **Run 2.** Raising lambda made the term a real part of the objective in the six FAIL units, 3.244
  to 30.654 percent, and it did not help anywhere. At p99max, COVID stayed under the gate even at
  lambda 100 (0.762 percent). On Japan it hurt h10 in every new arm; the released lambda 1 arm at
  p90max, the base of that ladder, is within noise there (RMSE d +19.668 [-19.442, +58.777]).
- **For the paper.** The epi-informed component joins spatial message passing, few-shot adaptation
  and meta-learning in the list of mechanisms added to improve transfer that did not help. Per the
  protocol's gates, the dengue arm is not run and no higher lambda is tried.

## Reproduce

```
conda run --no-capture-output -n ebola-train python ablation/run_epi_ablation.py --report --datasets influenza_japan influenza_us-regions influenza_us-states covid_us-states --quantile 0.99 --aggregator median --lam 100
conda run --no-capture-output -n ebola-train python diagnostics/verify_epi_bound_doc.py
conda run --no-capture-output -n ebola-train python diagnostics/verify_epi_bound_doc.py --mutate
```

Record inventory: 90 JSON records and 90 pernode archives under
`experiments/epi_bound_lambda/single/`, named `encoder__<panel>__seed<S>__<tag>.json`, with the six
tags in the "What ran" table.

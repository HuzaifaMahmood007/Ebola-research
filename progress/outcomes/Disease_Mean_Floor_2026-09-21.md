# The disease-mean floor: the bar we clear, and the bar we do not

**Run and recorded 2026-09-21.** Script `experiments/disease_mean_floor.py`. Output
`experiments/disease_mean_floor.json`. Arithmetic on archived predictions, about two seconds, no
GPU, no fitting.

**Everything here is EXPLORATORY and is stamped `protocol="EXPLORATORY"` in the record.** These
floors are built from the COMPLETE Ebola panel, query cells included, so they know the outcome of
the very cells they are scored on. They are not deployable forecasts, they are not pre-registered
(`progress/decisions/Ebola_Prereg.md:145` fixes persistence and support_mean only), and they must
never appear beside the pre-registered numbers without that label.

**Nothing was written to `results/` and no Ebola record was re-scored.** The model side is read out
of the 20 archived JSONs in `results/ebola/`. The score-once rule is untouched.

---

## In plain words

Our Ebola forecasts are currently graded against two very easy opponents. One repeats last week's
number. The other predicts a single flat value per district worked out from the opening weeks of the
outbreak, and because 47 of the 61 districts had reported nothing by then, that opponent predicts
**zero cases forever** for three quarters of the map. Beating it mostly proves we noticed the
outbreak existed.

So I built a harder opponent: a constant worked out from the **whole** outbreak, not just the quiet
start. It is allowed to see the future, which is exactly why it is hard to beat.

Two versions, because "average the outbreak, then average those values" can mean two things.

**Version one, one number for the whole country.** That number is **16.27 cases per week**. We beat
it everywhere, on both metrics, on both support windows, by 7.5 to 25.1 per cent. Good.

But here is the surprise. **This opponent is actually weaker than the one we already had**, in 15 of
the 16 cells. A single number for the whole map badly overpredicts all those quiet districts,
whereas the old floor's accidental zeros happened to be roughly right for them. So beating it is an
easier win, not a harder one, and it does not fix the weak-floor problem it was meant to fix.

**Version two, one number per district.** Each district gets its own average over the whole
outbreak. This is the real opponent. It has only 4 zero districts instead of 47, and all 4 genuinely
never reported a single case.

**We lose to it. Every horizon, both metrics, both arms.** By 10 to 24 per cent on the good arm, and
by as much as 85 per cent on the worst one.

That is not as bad as it sounds, and the reason matters. This opponent was handed 61 numbers taken
from the period it is being graded on. It knows which districts turned out big and which turned out
small. A real forecaster at the start of an outbreak cannot know that. So losing to it does not mean
the model is bad; it means that most of what makes Ebola districts differ is simply **how big each
district's outbreak ended up being**, and no amount of forecasting skill can recover that from
twelve quiet opening weeks.

It is worth writing down anyway, because a reviewer can build this opponent in ten minutes and we
would rather quote the number than be shown it.

---

## Why it was run

`Milestone5_Report.md` section 2.3 grades the model against persistence and support_mean only.
Auditing those floors on 2026-09-21 established two things that make the grading look generous:

- `train/ebola.py:268` gives a district with no support cell a floor of exactly `0.0`. On
  `ebola_L12` that is 43 of 61 districts with no support cell, plus 4 more whose support cells are
  all zero, so **47 of 61 districts forecast a flat zero**. On `ebola_L20` it is 33 of 61.
- No ceiling of any kind exists for Ebola. No published baseline can run on a disease it never saw
  (`export_baseline.py:105`), a single-disease Ebola model is forbidden in code
  (`train/loop.py:122`), and no literature comparison exists. Every comparator in the case study is
  a floor or our own sibling arm.

`progress/planning/Phase3_Developer_Execution_Guide.md:75` anticipated exactly this:

> "A large margin over a comparator that cannot in principle do the task is evidence for the regime
> being hard, not for the model being good."

This probe supplies the missing upper reference, with the leakage stated.

---

## What was built

Both floors are constants over the scored cells, fitted on every OBSERVED cell of the complete
panel:

| floor | construction | zero districts, L12 | zero districts, L20 |
|---|---|---|---|
| `disease_mean` | per-district full-panel mean, then averaged over districts. **One number: 16.2726** | 4 / 61 | 4 / 61 |
| `disease_mean_pernode` | each district keeps its own full-panel mean | 4 / 61 | 4 / 61 |
| `support_mean` (existing) | per-district mean over the SUPPORT window, 0.0 if no support cell | **47 / 61** | **33 / 61** |

The four zero districts under the new floors are districts that genuinely never reported a case, not
missing-data fallbacks. The selfcheck asserts this distinction directly.

`disease_mean` is identical on both arms (16.2726) because it never touches the support window. The
support-window mean over districts that have support is 7.5713 on L12 and 14.8688 on L20, so the
complete outbreak is roughly **twice** the level of L12's opening weeks.

### The gate, which is why these numbers are trustworthy

A new floor is just a number I invented unless the path that produced it can first produce a known
answer. Before reporting anything, the script rebuilds **persistence and support_mean** through the
same `score_predictions` call on the identical scored cells and compares against the published
`results/naive/naive__ebola_L{12,20}.json`. It exits if any of the 16 published values (2 floors x
2 metrics x 4 horizons) misses by more than 1e-9.

```
gate ok (ebola_L12): persistence and support_mean reproduced to 1e-09 on 8 cells
gate ok (ebola_L20): persistence and support_mean reproduced to 1e-09 on 8 cells
```

---

## Results

### Floor strength, country-macro, lower means a harder opponent

`ebola_L12`, RMSE:

| series | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| persistence | 43.405 | 44.340 | 46.006 | 50.677 |
| support_mean | 41.743 | 41.605 | 40.910 | 29.621 |
| `disease_mean` | 41.817 | 41.747 | 42.581 | 32.340 |
| **`disease_mean_pernode`** | **31.771** | **31.798** | **32.384** | **25.526** |
| MODEL zero-shot | 36.547 | 37.591 | 38.867 | 28.283 |
| MODEL few-shot | 38.195 | 37.984 | 41.191 | 28.520 |

`ebola_L12`, MAE:

| series | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| persistence | 25.899 | 27.402 | 28.526 | 31.916 |
| support_mean | 28.029 | 27.817 | 25.199 | 21.110 |
| `disease_mean` | 28.921 | 29.028 | 28.204 | 24.875 |
| **`disease_mean_pernode`** | **19.786** | **19.500** | **19.016** | **18.343** |
| MODEL zero-shot | 22.845 | 24.000 | 23.545 | 20.145 |
| MODEL few-shot | 24.687 | 24.609 | 26.050 | 20.102 |

**`disease_mean` is a WEAKER floor than `support_mean` in 15 of the 16 cells.** Its error is higher,
so it is easier to beat. The single exception is `ebola_L20` h5 on RMSE, where `disease_mean` is
41.747 against `support_mean`'s 42.042 and is therefore marginally harder, by 0.294. Every other
arm, horizon and metric goes the other way.

The reason is the long tail of quiet districts: `support_mean` predicts zero for them and is
accidentally close, while a single global constant predicts 16.27 and is badly wrong on all of them.
That effect is weaker on `ebola_L20`, whose 20-week support window leaves only 33 districts on the
zero fallback instead of 47, which is why the one exception sits on that arm. A global constant is
not the fix for a floor that is too weak.

### The model against the new floors

Percentage improvement, positive means the model is better.

| arm | metric | opponent | h3 | h5 | h10 | h15 | beats |
|---|---|---|---|---|---|---|---|
| L12 | RMSE | `disease_mean` | +12.6 | +10.0 | +8.7 | +12.5 | **4/4** |
| L12 | MAE | `disease_mean` | +21.0 | +17.3 | +16.5 | +19.0 | **4/4** |
| L20 | RMSE | `disease_mean` | +15.7 | +12.1 | +7.5 | +8.7 | **4/4** |
| L20 | MAE | `disease_mean` | +25.1 | +20.0 | +13.5 | +12.4 | **4/4** |
| L12 | RMSE | `disease_mean_pernode` | -15.0 | -18.2 | -20.0 | -10.8 | **0/4** |
| L12 | MAE | `disease_mean_pernode` | -15.5 | -23.1 | -23.8 | -9.8 | **0/4** |
| L20 | RMSE | `disease_mean_pernode` | -10.9 | -15.4 | -21.6 | -15.7 | **0/4** |
| L20 | MAE | `disease_mean_pernode` | -9.5 | -19.1 | -28.3 | -18.8 | **0/4** |

All rows above are the **zero-shot** arm. The few-shot arm is worse and repeats the pattern the rest
of the project has: it clears `disease_mean` 4 of 4 on L12 but only **1 of 4 on L20**, losing by up
to 26.8 per cent on RMSE and 26.8 on MAE, and it loses to `disease_mean_pernode` 0 of 4 on both
arms, by as much as 84.8 per cent at L20 h10 MAE.

---

## What is settled and what is not

**Settled.**

- The scoring path reproduces both published floors to 1e-9 on all 16 cells, so the new floor values
  come off the same machinery as the old ones.
- The disease mean is 16.2726 cases per week, against 7.5713 over L12's support window.
- The zero-shot model clears the global disease mean in 16 of 16 cells, by 7.5 to 25.1 per cent.
- The global disease mean is a weaker floor than `support_mean` in 15 of 16 cells, the exception
  being `ebola_L20` h5 RMSE where it is harder by 0.294, so adding it does not strengthen the
  grading.
- The per-district disease mean beats the zero-shot model in 16 of 16 cells, by 9.5 to 28.3 per
  cent, and beats the few-shot arm by up to 84.8 per cent.

**Not settled.**

1. **Both floors read the query period, so neither is a fair opponent.** `disease_mean_pernode`
   especially: 61 numbers fitted on the cells it is graded on. Losing to it is a measurement of how
   much district-level signal is unavailable in advance, not evidence about model quality. Any
   sentence that drops the leakage caveat is wrong.
2. **No interval.** These are point comparisons of seed-mean model macros against deterministic
   floors. `ebola_ci.py` was not run on them, so nothing here is a significance claim and no
   "clears zero" statement exists for any of these cells.
3. **This does not license a protocol change.** The pre-registration's floors are fixed at A5 and
   its amendment log is closed. These are an additional reference reported alongside, never a
   replacement, and the pre-registered criterion is adjudicated exactly as before.
4. **The four zero districts are shared by both new floors**, so neither is fully free of the
   zero-prediction issue. They differ from `support_mean`'s zeros in being real rather than
   fallbacks, which is the point, but the count is not zero.

---

## How this changes what we can say

Supported:

> The case study is graded against two naive floors, and the weaker of them predicts zero cases for
> 47 of 61 districts on the primary arm. Measured against a constant fitted on the complete outbreak
> the zero-shot model still wins in 16 of 16 cells by 7.5 to 25.1 per cent. Measured against a
> per-district constant fitted on the complete outbreak, which knows each district's eventual
> caseload and which no forecaster could possess in advance, the model loses in 16 of 16 cells by
> 9.5 to 28.3 per cent. Most of what distinguishes Ebola districts is the size each district's
> outbreak eventually reached, and that is not recoverable from twelve quiet opening weeks.

Not supported: any sentence calling `disease_mean_pernode` a fair baseline, any use of it without
the leakage caveat, and any claim that clearing `disease_mean` is a stronger result than clearing
`support_mean`. It is weaker in 15 of 16 cells. It is also not supported to say it is weaker in
**all** of them: `ebola_L20` h5 RMSE goes the other way.

---

## Reproducing

```
conda run -n ebola-train python -m experiments.disease_mean_floor --selfcheck
conda run -n ebola-train python -m experiments.disease_mean_floor
conda run -n ebola-train python -m diagnostics.verify_disease_mean_doc
```

`--selfcheck` runs automatically before the main path and asserts three things: the global constant
really is the mean of the per-district means, it differs from the support-window mean so it is a new
floor rather than a rename, and every zero district under the new floors is a district that truly
never reported a case rather than a missing-data fallback. The published-floor gate then runs per
arm before any new number is printed.

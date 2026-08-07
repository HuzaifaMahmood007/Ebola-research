# Graph-controlled pair run: COVID ↔ influenza_us-states

**Date:** 2026-08-04 · **Run:** 2026-08-03 22:13 → 23:01, 48 min, 5 seeds, both directions
**Records:** `results/lodo/encoder_pair__*` (adapted, 10 files) + `encoder_pair_zeroshot__*` (no adapter, 10 files)
**Regenerate:** `python compare_runs.py --metric rmse`

Metric is RMSE (`node_mean`), lower is better. Deltas are **paired per seed** against the
single-disease ceiling, so most of the 8–14% seed noise cancels. `sig` = two-sided paired t-test,
p ≤ 0.05, n = 5.

---

## 1. Headline: no positive transfer, and the design that would have detected it worked

| direction | arm | h3 | h5 | h10 | h15 |
|---|---|--:|--:|--:|--:|
| dengue-free trunk on **covid** → **influenza_us-states** | adapted | **+10.9%** sig | **+9.4%** sig | **+8.3%** sig | **+7.3%** sig |
| | no adapter | +27.8% sig | +26.9% sig | +35.7% sig | +36.0% sig |
| trunk on **influenza_us-states** → **covid** | adapted | +26.3% ns | −5.0% ns | +1.8% ns | +1.9% ns |
| | no adapter | +67.9% sig | +57.6% ns | +123.1% sig | +108.6% sig |

Positive = worse than that disease's own ceiling.

**The influenza direction is a clean, significant negative.** A trunk trained on COVID and adapted to
influenza is 7.3–10.9% worse than a trunk trained on influenza itself, significant at all four
horizons. One cell out of eight is negative (covid h5, −5.0%) and it is not significant.

**This is the result the fold was built to produce.** COVID and influenza_us-states share a
bit-identical `A_geo`, a bit-identical `C` and the same 49 nodes, so this fold varies the disease and
nothing else. Every previous cross-disease cell confounded disease with graph, geography, node count
and panel width simultaneously, which is why none of those negatives was ever attributable. **With the
graph confound removed, transfer is still negative.** The barrier is the representation, not the
graph — that is a real finding and it is worth more than another ambiguous cell.

## 2. The COVID ceiling has no skill, so half the table is unreadable

This is the most important thing the run surfaced, and it invalidates the COVID column above.

5-seed mean RMSE, encoder vs its own naive floors:

| dataset | h | encoder | persistence | seasonal | train_mean | verdict |
|---|--:|--:|--:|--:|--:|---|
| covid_us-states | 3 | 5,566 | 4,260 | 10,702 | 5,525 | **loses to persistence by 31%** |
| covid_us-states | 5 | 8,798 | 5,738 | 18,508 | 5,456 | **loses to train_mean by 61%** |
| covid_us-states | 10 | 12,264 | 10,068 | 29,875 | 5,302 | **loses to train_mean by 131%** |
| covid_us-states | 15 | 11,343 | 27,473 | 28,453 | 5,275 | **loses to train_mean by 115%** |
| influenza_us-states | 3 | 113 | 124 | 201 | 205 | encoder wins |
| influenza_us-states | 5 | 137 | 163 | 199 | 204 | encoder wins |
| influenza_us-states | 10 | 151 | 233 | 195 | 203 | encoder wins |
| influenza_us-states | 15 | 154 | 270 | 190 | 201 | encoder wins |

**Predicting "this state's average training week, forever" beats the COVID model by up to 131%.** The
single-disease COVID encoder has no skill beyond h3, and at h3 persistence still beats it.

So the COVID transfer column — "adapted is statistically indistinguishable from the ceiling at all
four horizons" — reads like a success and is not one. It means transfer matched a model that is
itself worse than a constant. **No COVID transfer number should be quoted until the COVID
single-disease model beats its own floors.**

Likely causes, all already documented in `covid_eda.md`:
- The **val fold owns Omicron** (2.8× both other folds; its peak is 6.3× the largest test week), so
  early stopping selects on an event that never recurs in test.
- COVID has only **60 train origins**, the fewest of any dev bundle.
- The test period is post-Omicron and comparatively flat, which is exactly the regime a constant
  predictor wins.

## 3. Adapter ablation: the adapter is doing nearly all the work

Improvement from fitting an adapter on the held-out disease, versus borrowing the mean of the
in-disease heads (the "no adapter" arm):

| direction | h3 | h5 | h10 | h15 |
|---|--:|--:|--:|--:|
| → influenza_us-states | 13.2% | 13.8% | 20.2% | 21.1% |
| → covid | 24.8% | 39.7% | 54.4% | 51.1% |

The transferred representation on its own is unusable: without a fitted adapter, error is 27–36%
above ceiling on influenza and 58–123% above on COVID. Every no-adapter cell but one is significantly
worse than ceiling, and on COVID the no-adapter arm is 2–5× the naive floor.

**Read carefully.** This is not evidence that the trunk transfers well and merely needs a read-out. It
is the opposite: 1,428 adapter parameters recover most of the achievable performance from a frozen
foreign trunk, and what they recover still lands 7–11% short of just training on the target disease.
That is consistent with the capacity probe, where a 15× larger read-out flipped zero cells positive
against the ceiling.

It also confirms the zero-shot arm is a weak baseline. `_mean_adapter` averages *parameters* rather
than functions, and the two heads emit into different per-node z-spaces, so the borrowed head is not
a principled starting point. Its large negative numbers should not be quoted as "zero-shot transfer
performance" without that caveat.

## 4. Trunks overfit almost immediately

Every trunk stopped early and used **15.5% of the 91,000-step budget** (median best at step 2,000,
median stop at 14,000; patience 12 × val_every 1000):

| held out | seed | best val @ | stopped @ |
|---|--:|--:|--:|
| covid | 42 / 52 / 62 / 72 / 82 | 1k / 2k / 3k / 2k / 3k | 13k / 14k / 15k / 14k / 15k |
| influenza_us-states | 42 / 52 / 62 / 72 / 82 | 1k / 3k / 2k / 2k / 2k | 13k / 15k / 14k / 14k / 14k |

A 142,305-parameter trunk overfits a single 49-node panel in about 2,000 steps, then degrades
monotonically (seed 82 covid: 0.1555 at step 3,000 → 0.1852 by step 13,000). The COVID-source trunk
is the worst case, training on 60 origins.

**Caveat this carries for the whole run:** what was measured is transfer *from a barely-trained,
immediately-overfitting trunk*. The 91,000-step default was calibrated for the dengue-containing
configuration, not for single small panels. This does not invalidate the negative result — the trunk
is at its own validation optimum, which is the best it can offer — but it does mean the pair fold is
not a test of what a well-trained trunk could transfer.

---

## What was achieved

1. **A clean, attributable negative.** Transfer is 7.3–10.9% worse than the ceiling on influenza, at
   all four horizons, significant, with graph, covariates, node set and node count all held identical.
   No previous cell in this project can rule out the graph as the cause. This one can.
2. **The COVID single-disease model is broken** and was caught before it entered a headline table.
   It loses to a constant by up to 131%.
3. **An adapter ablation with numbers**: 13–54% of achievable performance comes from the 1,428-param
   adapter, and the frozen representation alone is unusable.
4. **A measured overfitting regime** for small panels that changes how the step budget should be set.

## What to do next

- **Do not report any COVID transfer number** until COVID beats its own naive floors. Fix the target,
  not the transfer.
- The three-disease LDO3 run is the outstanding item; the pair fold does not substitute for it.
- The pair result is a publishable negative on its own terms and should be written as one, not
  buried. It answers a question the rest of the study cannot.

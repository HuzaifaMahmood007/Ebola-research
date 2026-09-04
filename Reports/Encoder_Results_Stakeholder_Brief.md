# Where the Encoder Actually Stands — Stakeholder Brief

**30 July 2026 · Phase 3, Week 4**

Plain-language companion to `Encoder_Results_Consolidated.md`. Same numbers, no statistics
background assumed. Every figure recomputed from run outputs this week; nothing carried over from
earlier write-ups.

---

## The short version

You asked us to fix the folds before spending the five seeds, because you suspected our transfer
result was measuring the wrong thing. **You were right, on both counts you raised.**

1. **The arithmetic you challenged was our error, and we can now show exactly what caused it.** Your
   three worked examples reproduce to the decimal point. The two tables genuinely could not be read
   against each other.
2. **With the folds corrected, the transfer benefit disappears.** On our two headline accuracy
   measures, there is now no case anywhere where training across diseases helps a genuinely
   held-out disease. In most cases it hurts.

This is a weaker result than our Week-3 summary implied. We think it is the real one.

---

## 1. What was wrong with the old number

We had two tables that used different comparison points without saying so.

Our model runs five times with different random starts ("seeds"), giving five slightly different
results. The transfer table compared against **one specific run**. The accuracy table reported the
**average of all five**. Neither is wrong on its own; comparing across them is.

Your three examples, and what we get:

| what you checked | you calculated | our old table said | cause |
|---|---|---|---|
| flu US-regions, 3 weeks ahead | 9.8% better | +20% | different comparison point |
| flu US-states, 3 weeks ahead | under 1% better | +6% | same |
| flu Japan, 5 weeks ahead | 7.4% better | +4% | same |

Our recomputed figures are **9.9%, 1.4% and 7.4%**. Your arithmetic was correct throughout.

**The dengue case is worse than a mismatch — it reverses the answer.** Reported as 2.9% *better*,
it is actually **15.4% worse** once compared against the five-run average. The cause is specific and
checkable: the single-disease dengue run we happened to compare against was an unusually bad one
(50.0 against roughly 37–47 for the other four). Measuring against a weak opponent made transfer
look good.

**Fixed going forward.** Every comparison table we produce now states its comparison point on the
table itself, and the check is automated rather than remembered.

---

## 2. The fold problem, and what the honest number looks like

Three of our four datasets are influenza. Holding out flu-Japan still left the model trained on
flu-US-regions and flu-US-states — so it had seen the disease, just not that population. That is
**population transfer**, and it is a real but much smaller claim than the one the paper needs.

We rebuilt the experiment to hold out the *disease*: all three influenza sets together on one side,
dengue on the other. That is **cross-disease transfer**, and it is the number the paper rests on.

### Side by side (forecast error, positive = transfer helps)

| dataset | weeks ahead | old fold (population) | corrected fold (disease) |
|---|---|---|---|
| flu Japan | 3 | +24.0% | no measurable effect |
| flu Japan | 15 | −22.6% | **−51.3% worse** |
| flu US-regions | 5 | +28.0% | **−10.0% worse** |
| flu US-regions | 15 | −0.5% | **−51.2% worse** |
| flu US-states | 3 | +5.8% | **−4.4% worse** |
| dengue | 3 | +2.9% | no measurable effect |
| dengue | 15 | −1.7% | **−1.5% worse** |

Across all 16 combinations of dataset and forecast horizon: **12 are significantly worse, 4 show no
measurable effect, and none is better.**

Note which results collapsed most. The three largest apparent wins under the old fold — **+28.0%,
+24.0% and +19.9%** — were all on influenza panels where other influenza data stayed in training.
Exactly the pattern you flagged.

**"No measurable effect" is not the same as "no difference."** It means the result is too unsteady
across runs to call a direction. On flu US-regions, which has only ten regions, our margin of error
is wide enough that a real effect could hide inside it.

---

## 3. The pattern that matters most for Ebola

Transfer performs worst at the longest forecast horizons — and gets steadily worse the further out
you look, on every dataset. At 15 weeks the corrected figures are around 50% worse on two of the
three influenza panels.

This collides directly with the Ebola audit. At 10 and 15 weeks ahead, Ebola has **no adaptation
data at all** — the model would have to work with no Ebola-specific tuning whatsoever. We tested that
mode, and on influenza it fails badly: errors several times worse than a normally trained model.

Put plainly: **the long-horizon Ebola forecasts land in the one setting where we have the least
evidence the method works.** Practical tuning is available only at the 3-week horizon, and only from
16 data points across 8 districts.

We are not recommending abandoning the Ebola case study. We are flagging, before it runs, that a weak
result will be impossible to separate from "there was not enough Ebola data to adapt on." Since we
score Ebola exactly once against a locked configuration, that ambiguity cannot be resolved afterwards.
Your instinct to narrow the horizon deliberately looks correct on this evidence.

---

## 4. The epidemiology metrics you asked for

You asked for peak timing error, peak intensity error, and an error measure that is not dominated by
the largest regions. All three are now computed and reported.

**Scale-normalised error changes the ranking.** By raw error dengue looks easiest (smallest numbers),
but that is only because dengue case counts are small. Adjusted for scale, **dengue is the hardest
dataset we have** and US-regions the easiest. Raw figures were flattering dengue.

**Peak timing is poor, and we would rather you heard it from us.** How close the model gets to the
true peak week:

| dataset | 3 weeks ahead | 15 weeks ahead |
|---|---|---|
| flu Japan | 4.9 weeks off | 26.7 weeks off |
| flu US-regions | 43.3 weeks off | 67.8 weeks off |
| flu US-states | 10.6 weeks off | 27.7 weeks off |
| dengue | 10.9 weeks off | 32.0 weeks off |

Only flu-Japan at 3 weeks is anywhere near operationally useful. On US-regions the model is not
locating the peak at all. This is a new measurement, it is unflattering, and it argues against any
"ready for deployment" framing.

**One honest caveat in the other direction.** Two of these peak measurements *do* show cross-disease
transfer helping (flu-Japan and dengue peak timing at particular horizons). But they sit among 32
such tests, 28 of which show nothing. Two hits out of 32 is roughly what pure chance produces. We are
recording them rather than hiding them, but we are not counting them as a win — and we would push
back if they appeared in a draft as evidence.

---

## 5. Also confirmed

**Joint training does nothing.** Training on everything at once performs the same as training on each
disease separately. This is a clean negative result and it is the motivation for the
freeze-then-adapt design, rather than an unfinished experiment.

**A quality check we ran on ourselves.** A comparison is only fair if both sides are scored on the
same regions. We now verify this automatically rather than assuming it, because an unfair comparison
has already produced a misleading figure elsewhere in this project. **All comparisons in this
document passed.**

---

## 6. What is still missing

**Uncertainty measures are not yet available.** WIS, CRPS, coverage and PIT are built and tested, but
our training runs do not currently save the outputs they need. **Until that is fixed, our
"calibrated uncertainty" claim has no evidence behind it.** This is the largest remaining gap against
your metrics list.

**The old fold structure has only one run.** Every population-transfer figure rests on a single
random start — and we now know that particular start was an outlier for dengue. Those numbers should
not be quoted.

**One transfer variant is missing a metric** and needs a rerun rather than a recalculation.

---

## 7. What this means for the paper

The central claim as previously framed — that cross-disease pre-training improves forecasting on a
new disease — **is not supported by our own corrected experiment.**

There is still a paper here, and arguably a better one:

- A **clean, well-evidenced negative result** on cross-disease transfer, which is genuinely useful to
  the field and rare in publication.
- A **methodological finding** that fold structure and comparison-point choice can manufacture a
  transfer effect that is not there — demonstrated with a reproduced sign inversion, not asserted.
- The **population-transfer result**, which is real, provided we stop describing it as cross-disease.
- The **reproduction failure log and score-once pre-registration**, which you have already adopted.

**On meta-learning.** This changes its status. The freeze-then-adapt approach was going to be the
comparison baseline for meta-learning — and that baseline is now a negative result. Meta-learning
stops being an additional feature and becomes the more likely route to a positive finding. That
strengthens the case for doing it, and we owe you a straight answer on whether it fits the schedule.

---

## Recommended decisions

1. **Report both fold structures**, as you asked, with the cross-disease number as the headline.
   Do not let the population-transfer figures stand unqualified.
2. **Fix the uncertainty gap** by saving quantile outputs on the next run. Cheap now, expensive later.
3. **Narrow the Ebola horizons** deliberately, on the evidence in §3, rather than being quietly weak
   at 15 weeks.
4. **Treat meta-learning as load-bearing**, not optional. We will bring the schedule answer this week.
5. **Complete the five runs for the old fold structure** if the population-transfer claim is going in
   the paper — one run is not enough to publish.

Happy to walk through any of this. The underlying numbers, and the code that produced them, are in
`Results_Matrix.md` and `results_matrix.py`; every figure here can be traced back to a run output.

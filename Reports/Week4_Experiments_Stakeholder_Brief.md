# Every Experiment We Ran, and What It Showed — Stakeholder Brief

**5 August 2026 · Phase 3, Week 4**

An account of every training setup we ran this week and what each one returned. No statistics
background assumed. Every figure below was recomputed directly from the run outputs before this
document was written; nothing is carried forward from an earlier draft.

---

## The short version

We ran eight distinct training setups, plus a full benchmark against published models — **312
training runs in total**, the largest of them a fifteen-run job that took just under eleven hours.

Four things came out of it.

**One. Training across diseases does not help a disease the model has not seen.** We tested this
four separate ways, each stricter than the one before. None of them found a benefit. In the largest
and most careful test, the shared model was worse than a model trained on the disease itself in
roughly half the comparisons, better in at most one, and too unsteady to call in the rest.

**Two. The gap widens the further ahead you forecast.** At three weeks out, a model built from other
diseases is roughly level with one trained on the disease directly. At ten and fifteen weeks it is
worse everywhere we can measure, by up to half again the error.

**Three. Our model is the strongest in the comparison, and the whole field loses to simple
baselines on this data.** Against EpiGNN, Cola-GNN and MTGNN, all re-run on our own pipeline, ours
is ahead far more often than behind. But ours beats a plain "next week resembles this week"
baseline in only 6 of 16 cases, and every published model we tested does the same or worse. That is
a property of these datasets and these forecast horizons, not a fault unique to our model.

**Four. One experiment found something that clearly works.** When we enlarged the small
disease-specific piece of the model, cross-disease performance improved by up to 23%. The identical
change applied within a single disease gained nothing at all. The limitation we had been measuring
was in that component, and it is specific to the transfer setting.

---

## What we ran

| Setup | What it does | Runs |
|---|---|---|
| One model per disease | Trains and tests on the same disease. The reference everything else is read against. | 25 |
| Everything at once | One model trained on all diseases together. | 6 |
| Hold out a population | Trains on three influenza panels and dengue, holds out one panel. | 4 |
| Hold out a disease, two-way | Influenza as one disease against dengue, in both directions. | 10 |
| **Hold out a disease, three-way** | **Dengue, influenza and COVID; each held out in turn.** | **15** |
| The controlled comparison | COVID against influenza on an identical map and identical inputs. | 10 |
| A larger tuning layer | Four sizes of the disease-specific component. | 8 |
| Seasonality removed | Influenza-Japan without its seasonal input. | 5 |
| Published models | EpiGNN, Cola-GNN, MTGNN and HeatGNN on our pipeline. | 229 |

Each setup below is reported against the same reference: **a model trained on the target disease
itself**, run five times from different random starting points, with the average taken. Where we say
a result is "too unsteady to call", we mean the spread across those five runs is wide enough that we
cannot honestly assign it a direction.

---

## One model per disease

This is the foundation everything else is measured against, and it is weaker than we would like.

**Our model beats simple baselines on one of five datasets.** On influenza across US states it wins
clearly, by 8% to 22% depending on the horizon. On dengue it is roughly level at the longer
horizons and behind at the shorter ones. On influenza in Japan it loses badly — 34% to 58% behind a
seasonal baseline, because Japanese influenza is strongly periodic and a model that simply repeats
last year's season is hard to beat. On COVID it loses by a wide margin at every horizon.

We report this plainly because it sets the ceiling on everything that follows. A transfer method is
being asked to match a reference that is itself often beaten by arithmetic.

## Everything at once

Training a single model on all diseases simultaneously performs the same as training each disease
separately. Across 32 comparisons, one was better, one was worse, and thirty were too unsteady to
call.

This is a clean and complete negative result. It is not an unfinished experiment.

## Hold out a population

Here the model trains on three influenza panels and dengue, and is tested on the influenza panel it
did not see. This was the setup that produced our earlier headline of 20% to 28% improvement.

**Two things are now clear about it.** It never held out a *disease* — when Japan was held out, the
model had still trained on influenza in the United States, so it had seen the disease and only
missed the population. And it was run only once, not five times, so no result from it can be tested
for steadiness. The figures from this setup should not be quoted.

## Hold out a disease, two-way

The first genuine cross-disease test: all three influenza panels treated as one disease on one side,
dengue on the other, both directions, five runs each.

Across 32 comparisons: **none better, 22 worse, 10 too unsteady to call.**

## Hold out a disease, three-way — the main run

This is the experiment the week was built around. COVID was added as a third disease, and each of
the three — dengue, influenza, COVID — was held out in turn while the model trained on the other
two. Fifteen runs, five random starts each, 10.8 hours on one graphics card. Every expected output
file was produced and checked.

Two arms were measured. In the first, the shared model is frozen and a small disease-specific piece
is fitted on the held-out disease. In the second, nothing at all is fitted on the held-out disease.

**The first arm.** Under two standard statistical tests, the shared model is significantly worse
than the disease-specific reference in 18 to 25 of the comparisons, and significantly better in at
most one. The two tests disagree about how many losses are large enough to be certain of; they agree
completely that there is no benefit.

**The pattern by horizon is the more useful result.** At three weeks ahead most comparisons are too
close to call. At five weeks the losses begin. By ten and fifteen weeks the shared model is worse in
every comparison we can attribute, reaching 38% worse on influenza in Japan and 51% worse on
influenza across US regions. Whatever the shared model carries between diseases, it holds for a few
weeks and then stops holding.

**The second arm fails completely.** With nothing fitted on the new disease, the model is worse in
every one of the 36 comparisons, by between 1% and 139%. Averaging what the model learned from other
diseases and applying it directly to a new one does not produce a usable forecast.

**One fold is not readable and we are not counting it.** COVID at ten and fifteen weeks measures a
data boundary rather than the model. The Omicron wave — the largest week in the entire COVID
series — falls inside the period we use to select the model, while the period we score on is a flat
tail with a peak six times lower and about half the week-to-week variation. Every model in the study
is tuned on a surge and then graded on a lull. Those four comparisons are reported in full but
excluded from the counts above, in both directions. COVID's shorter horizons are counted, and they
are losses.

## The controlled comparison

Every cross-disease test above changes more than the disease. Dengue has 7,165 regions across
twelve countries; influenza in Japan has 47; US regions has 10. The map, the geography and the
population all change at the same time as the pathogen, so a poor result cannot be pinned on the
disease alone.

COVID and influenza across US states are the one exception in the entire study: the same 49 states,
the same map, the same inputs. Holding one out changes the disease and nothing else. We ran both
directions, five times each.

Across 16 comparisons: **none better, 8 worse, 8 too unsteady to call.**

This is the cleanest cross-disease measurement we have, and it agrees with the others.

## A larger tuning layer

The one experiment that returned a clear positive.

The disease-specific component of our model is deliberately small — about 1,400 adjustable values,
sized for the 27 labelled examples Ebola will offer. We tested three larger versions against it, on
cross-disease transfer and, as a control, within a single disease.

**The larger component improved cross-disease performance by up to 23%, and by about 4% typically.
The same change within a single disease produced no gain at all — slightly worse, if anything.**

This tells us something specific: part of what we have been measuring as "transfer does not work"
was the tuning layer being too small to express the correction a new disease needs. It was a real
bottleneck, and it was specific to transfer rather than a general shortage of model capacity. It
does not overturn the results above — the gains are far smaller than the deficits — but it is a
genuine finding and it was obtained cheaply.

This run used a single random start, so it carries less weight than the five-run experiments.

## Against published models

We re-ran EpiGNN, Cola-GNN, MTGNN and HeatGNN on our own data, splits, horizons and scoring, five
times each — 229 runs. Running them identically is what makes the comparison fair, and it is a
condition we set before looking at any result.

**Our model is the strongest of the group.** Against EpiGNN it is better in 9 of 16 comparisons and
worse in 1. Against Cola-GNN it is better in 4 of 12 and worse in none. Against MTGNN it is better
in 12 of 16. HeatGNN currently runs on only one dataset and two horizons, where the two are level.

**And every model in the group, including ours, loses to simple baselines more often than it wins.**
Ours beats the best naive baseline in 6 of 16 dataset-and-horizon combinations; EpiGNN in 5, Cola-GNN
in 5 of 12, MTGNN in 2. We state this alongside the head-to-head result rather than separately,
because reporting only the head-to-head would leave a misleading impression.

---

## Adding COVID

COVID was brought in this week as a third development disease, and the work to add it was
substantial: a new data loader, a rebuilt input pipeline, and a diagnostic investigation when the
first results looked wrong.

They looked wrong because they were measuring the data, not the model. The COVID series contains a
structural break — Omicron — that is larger than everything around it, and it falls between the
period used to tune the model and the period used to grade it. A model fitted to a surge
necessarily over-predicts the flat period that follows, and the error grows with forecast distance.
COVID's own disease-specific model is *negatively* correlated with the truth at five weeks and
beyond, meaning it moves in the opposite direction to the real case counts.

The COVID dataset is sound and the pipeline is correct. The problem is where the epidemic's shape
falls relative to our fixed date boundaries.

## Uncertainty, measured for the first time

Until this week our forecasts saved only a single number per prediction, which meant we could not
evaluate the uncertainty ranges at all — a stated requirement we had no evidence for. The runs now
save the full range, and we have measured it.

**The result is better than expected.** On influenza in Japan, a disease the shared model had never
seen, our stated 50% range contained the truth 49.4% of the time and our 90% range contained it
89.9% of the time. Those are close to exactly right. The uncertainty ranges are honest even where
the central forecast is not accurate.

Two gaps remain. The no-tuning arm did not save these outputs, so it cannot be assessed. And of the
disease-specific reference models, only COVID saved them, so for the other four datasets we can
report our own calibration but cannot compare it against the reference.

## What we found wrong in our own work

Three issues surfaced during checking, and all three are recorded here rather than quietly fixed.

**A results file was holding two different models under the same labels.** The COVID output
contained both the standard forecast and an experimental bias-corrected variant, stored so that
reading the file naively returned whichever appeared last. This inflated the COVID reference and
manufactured **three false "transfer helps" results**. It was caught by an automated check that
recomputes every published number from the raw outputs, and every figure in this document was
produced after the fix.

**A 90% uncertainty range was being dropped silently.** A numerical precision issue meant one of the
two uncertainty bands was discarded without any error being raised. Had it not been caught, the
calibration result above would have been reported with half its evidence missing.

**The shared model stopped training very early.** In every one of the fifteen main runs, training
halted after using between 1% and 16% of its allotted budget, because performance on the diseases it
was training on stopped improving. The learning rate had barely moved from its starting value when
training ended. The transfer results above are therefore measured on models that stopped early by
their own criterion, and we cannot say from this run what a longer-trained shared model would do.

## Where these numbers are weak

**Three diseases is a small basis for a claim about diseases.** Influenza contributes three of the
five datasets, dengue and COVID one each. Every conclusion about cross-disease behaviour rests on
that.

**Only one comparison isolates the disease.** As described above, all the others change the map and
the geography at the same time.

**The reference is an advantage the real application will not have.** In every transfer test, the
held-out disease was given its complete training history to fit its disease-specific component.
Ebola will offer 27 labelled examples. These results are therefore the most favourable case, not a
typical one.

**The largest deficits fall at ten and fifteen weeks**, which is where the Ebola data provides no
material to tune on at all.

---

Every number in this document can be traced to a run output. The detailed tables are in
`LDO3_Results.md` and `Results_Matrix.md`; the code that produced them, and the automated check that
verifies each published figure against the raw results, are in `ldo3_report.py` and
`verify_ldo3_doc.py`.

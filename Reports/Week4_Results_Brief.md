# What We Built, and What It Shows — Stakeholder Brief

**11 August 2026 · Phase 3**

This covers the work completed since the 5 August brief, organised around the five contributions the
paper now claims. Every number is the mean over five fixed random starts, and every claim of
"better" carries an interval built by resampling rather than a single comparison.

---

## The short version

**One. A new disease turns out to be expressible as a small, bounded correction to a shared
representation of epidemics.** This is the central technical claim and the architecture is built to
enforce it: the model sees only four channels of case data, no geographic covariates, no disease
identifier, and no parameter anywhere that is indexed by location. It cannot memorise a disease or a
map. What it learns has to be general, because nothing else is available to it.

**Two. We can now say how large that correction has to be.** A five-seed experiment across four
adapter sizes shows extra capacity helps cross-disease in 7 of 36 comparisons, by up to 20%, and
helps within a disease in 0 of 12. The correction is real, it is small, and it is needed only when
crossing between diseases.

**Three. Transfer has a boundary, and we have located it.** It holds at long horizons and fails at
short ones. This is now measured rather than asserted, and it held up when we removed the one
plausible artefact that could have produced it.

**Four. Our forecasts carry calibrated uncertainty for the first time.** The model's own confidence
ranges covered about half of what they claimed. A correction fitted entirely on other diseases
brings them to target on a disease it has never seen.

**Five. The evaluation framework itself is a contribution.** Different experimental designs lead to
different conclusions on the same data, and we can now demonstrate that directly rather than assert
it.

---

## One: the architecture, and the claim underneath it

**The design decision is the scientific claim.** If a shared model can be transferred to a new
pathogen by adjusting only a small number of parameters, then epidemics of different diseases share
enough structure to be worth modelling jointly. The architecture is built so that this claim can
fail honestly if it is false.

Four restrictions make it testable. The model reads four channels of case counts and nothing else,
enforced in code rather than by convention. Geographic covariates are withheld. Influenza
self-loops are removed, so a doubled self-weight cannot quietly identify which disease is being
forecast. Graphs of mixed spatial resolution are handled block-diagonally rather than being forced
onto a common grid. No parameter is indexed by location, so a model trained on 47 Japanese
prefectures applies unchanged to 7,165 dengue regions or the roughly fifty Ebola districts.

**The Ebola result is the strongest evidence for this claim so far.** Trained on influenza, dengue
and COVID, and given no Ebola data whatsoever, the model beats the standard benchmark by 16% at
three weeks, 15% at five, 16% at ten and 44% at fifteen. On the secondary configuration the pattern
repeats: 19%, 17%, 14% and 42%.

**Fourteen of those comparisons survive an interval that excludes zero, and they now span every
horizon.** An earlier version of this brief reported eight, all at ten and fifteen weeks. That count
came from an interval computed on a different statistic than the one being reported, and it
understated the result. Recomputed correctly, the model with no Ebola data beats the benchmark with
a clean interval at three weeks as well as at ten and fifteen. The short-horizon result is real; we
had been measuring it with the wrong instrument.

---

## Two: how large the correction has to be

**We measured it directly, at five seeds, with a control.** Four adapter sizes were fitted on a
frozen shared model, once across diseases and once within a disease as a control. Within each seed
the same shared model serves every size, so the only thing varying is the adapter. Comparisons are
seed-paired, because run-to-run variation alone reaches 9.7% on some cells and would otherwise
swamp an effect of this size.

**The finding is specific.** Larger adapters improve cross-disease forecasting in 7 of 36
comparisons, with the largest gain 19.8% and an interval of +5.8% to +33.9%. **Every one of those
improvements falls at the fifteen-week horizon.** In the within-disease control the same sizes
improve 0 of 12.

**It is not free, and we report both directions.** Three of the 36 cross-disease comparisons are
significantly *worse* with a larger adapter, and one of the 12 within-disease comparisons is worse
too. Extra capacity is not a general improvement; it is a targeted one that costs something
elsewhere.

**What it means.** The gap between transferring a model and training one from scratch is partly a
capacity problem, and only at long range. Where transfer already works, more capacity does not help
and can hurt; where it struggles, a modest amount does. That is what a "bounded correction" looks
like when measured.

---

## Three: where transfer holds and where it fails

**On the development diseases the boundary is the horizon.** Cross-disease transfer degrades as the
forecast reaches further out, and the adapter-capacity experiment finds its only gains at the
longest horizon. Both point the same way.

**On Ebola the boundary is not the horizon, it is the adaptation step.** Corrected measurement
changes this. Against the benchmark, the model given no Ebola data wins 12 of 16 comparisons across
all four horizons; the version fitted on Ebola data wins 2 of 16, both at fifteen weeks. What
separates a win from a loss on the case study is not how far ahead we forecast, it is whether we
adapted at all.

**We removed the obvious artefact.** All fifteen transfer runs had stopped training early — a
mismatch between the stopping rule and the learning-rate schedule meant none reached its budget, so
the boundary could have been an artefact of undertrained models. We re-ran one fold with early
stopping effectively disabled, for the full 91,000 steps. The best model appeared at step 7,000 and
never improved again. The scored results came back **identical to the original run**. The extra
72,000 steps contained nothing better, and the boundary is not a training artefact.

**We checked the reference too.** Every transfer figure is a ratio against a model trained on that
disease alone, so if only the transfer side stopped early we would be comparing a half-trained model
to a fully-trained one. All 25 reference runs also stopped early, at 18 to 77 of 80 epochs. The
intuition that this makes the two sides cancel out is wrong, and the direction matters: the shared
model is at its own optimum while the reference may sit below its own, which means our reported
deficits **understate** the true gap rather than exaggerate it.

---

## Four: calibrated uncertainty

**The model's own confidence ranges were badly overconfident.** A range labelled "90% confident"
should contain the truth about nine times in ten. Across the five development diseases, the model's
ranges covered between 49% and 92% depending on disease and horizon. On Ebola they covered 50%. A
planner acting on those intervals would have been misled.

**The correction uses no data from the disease it is applied to.** We measured the error on five
diseases, derived a correction, froze it, and applied it to Ebola unseen. Tested the same way — each
panel corrected using only the other four — the corrected ranges land between 89% and 93% against a
90% target. On Ebola they land at 91%.

**The development figures are the optimistic ones**, and we say so rather than let them stand in for
Ebola. The correction improves itself as observations arrive, and the development panels give it
between 47 and 630 observations to do that. Ebola gives it eighteen.

**Reported as a range across all five panels, never a single flattering one**, and always beside the
uncorrected numbers rather than in place of them.

**One finding we did not expect.** The single-disease reference models are themselves poorly
calibrated, covering 53% to 93%. Our corrected cross-disease intervals sit closer to target than the
reference models' own intervals do. That is not evidence that transfer forecasts better; it means
the uncertainty component is under-dispersed on every disease we have, which is a defect worth
reporting in its own right.

---

## Five: the evaluation framework

Two experimental designs on the same data support different conclusions, and we can now show why.
Holding out a population and holding out a disease are different questions, and a design that
changes the map and the disease at once cannot tell you which one produced the effect. Our protocol
separates them, states the reference on every comparison, reports dispersion on every number, and
declines to name a direction when the interval covers zero.

This is offered as construction rather than critique: a more rigorous framework, and a demonstration
that different designs lead to different conclusions.

---

## Supporting evidence

Three further experiments inform the above without carrying claims of their own. Training on all
diseases jointly does not beat training on each separately. A model given no data from the target
disease performs comparably to one given a small amount, and in the Ebola case study it performs
better. And in the head-to-head against published models, ours is the strongest of the group while
every model in the group, ours included, loses to simple benchmarks more often than it wins. Taken
together these describe the same boundary from three directions.

---

## What we found wrong in our own work

**A permanent loss, caught with hours to spare.** The no-adaptation arm of the Ebola experiment was
not saving the numbers needed for any uncertainty analysis, and the protocol allows that experiment
to run exactly once. Found and fixed before it started.

**The uncertainty correction misbehaves on dengue.** Roughly one cell in nine there has a prediction
interval narrower than a single case. The correction multiplies interval widths, and multiplying a
near-zero width cannot fix it, so the method compensates by inflating every other interval in that
dataset by a factor of ten to twelve — which leaves them fifteen to eighteen times wider than the
reference model's. Ebola and the other four panels are unaffected: they contain no such intervals.
Documented and scheduled; the dengue column of that one comparison should not be quoted meanwhile.

**A labelling error in our own output.** Ten rows of the calibration output describe the ten- and
fifteen-week primary-arm results as adapted when the protocol requires them to be labelled otherwise,
because there is genuinely no adaptation data at those horizons. Being corrected.

---

## Where these numbers are weak

**The Ebola case study did not clear the bar we set in advance.** We pre-registered that success
meant the *adapted* model beating the benchmark at three or five weeks with the interval excluding
zero. It does not: all four of those intervals span zero. The wins we do have come from the
unadapted model, which is not what the criterion asked for. Reported as a miss, and the adaptation
step is the reason. The version fitted on Ebola data is worse than the version given none at every
horizon, and far less stable: run-to-run variation reaches 32%, against under 1.5% without it.

**We were adjudicating that criterion with the wrong instrument until an audit caught it.** The
reported accuracy figure and the interval being used to judge it were computed over different
units, so the point estimate did not even fall inside its own interval. Both are now computed the
same way, over districts, as the pre-registration always specified. The verdict on the criterion is
unchanged. Several individual results improved.

**The early-stopping check is one fold and one seed.** The other fourteen are inferred from it. Two
more folds would cost about seven hours and would close the inference.

**Five random starts is too few for the supporting statistical test.** With five, the smallest value
that test can return is 0.062, so it can never reach the conventional threshold regardless of effect
size. We rely on the resampling intervals instead.

**The uncertainty correction carries no mathematical guarantee on Ebola.** Its self-correcting
mechanism needs a long run of observations and Ebola provides eighteen. The coverage we report is
what was observed, not what is promised.

**One disease, one outbreak.** Everything about Ebola rests on the 2014 West African epidemic.

---

## Where this leaves the milestones

Measured against the six weekly milestones in the internal project brief, the first four are
complete and the fifth is two-thirds done. Weeks 1 to 3 — the literature review and baseline
selection, the multi-disease data engineering and audit, and the core framework with its benchmark
against published models — are delivered, with the epidemiology-informed component deliberately
re-scoped from the main build to an ablation, which was always its status as a suggestion rather
than a requirement. Week 4's contribution is delivered: leave-one-disease-out transfer results exist
across all five development diseases, and the Ebola adaptation protocol is built, frozen and hashed.
Week 5 is where the line falls. The Ebola case study has been scored once against its locked
configuration and calibrated uncertainty is complete — fitted on other diseases, frozen, applied and
checked — but explainability has not been started, and that is a required goal rather than a
suggested one. Week 6 has not begun. Against the brief's own definition of done, three of the five
conditions are met: the framework runs end to end on the standardised schema and is documented, it
is benchmarked against the agreed baselines on identical data, and the transfer and Ebola results
now carry calibrated intervals. The benchmark condition is the weakest of the three: two of the four
selected comparators are usable today, and one of the remaining two emits a constant prediction on
most of its output and must be retired rather than reported. The two outstanding conditions are the ablations plus explainability, and the manuscript
with its reproducibility package. Those are what the two-week extension to day 40 is for.

## What happens next

The adaptation-procedure ablation, the explainability work and the epidemiology-informed component
remain outstanding, followed by robustness, ablations and the reproducibility package. The
adaptation ablation is scoped as an ablation and will be reported as one whatever it returns.

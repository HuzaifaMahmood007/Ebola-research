# Explainability: three decisions, now closed

**Prepared 2026-09-16. For the client. All three decisions were answered on 2026-09-16 and the
outcomes are recorded below.** Goal G5 (explainability) is built, tested, verified and now signed
off. This document gives the evidence behind each decision rather than asking anyone to take our
word.

## Outcomes

| # | decision | outcome | the reason in one line |
|---|---|---|---|
| D1 | integrated gradients instead of SHAP | **ACCEPTED** | SHAP costs 12,544x to 1,834,240x more model runs here, and where both can be run they give the same answer, 5 of 5 districts |
| D2 | retract the SHAP row from the comparison table | **APPROVED AND DONE** | the row claimed a capability we do not have; corrected copy issued 2026-09-16 |
| D3 | accept a neighbour figure that does not explain accuracy | **ACCEPTED** | the graph is used but does not improve error in 0 of 40 cells, and we now have the reason as well as the result |

Detail and evidence follow.

Every number here was measured on our own model and our own data. The scripts are in the repository
and each one reruns from scratch.

---

## The three decisions, in one paragraph

We were asked for SHAP. We built a different method called integrated gradients, because SHAP does
not work at the size of this problem. **Decision 1** was whether to accept that substitution, and it
was accepted. **Decision 2** followed from it: a comparison table already in the client's hands said
we deliver SHAP, and that row has now been corrected. **Decision 3** was separate and concerned what
one figure is allowed to claim; it was accepted with the limitation attached.

---

## Decision 1. Integrated gradients instead of SHAP

### What both methods are trying to do

The model reads 20 weeks of history across four channels for each district, then produces a
forecast. Both methods answer the same question: **of everything the model read, which parts moved
the answer?** They differ only in how they find out.

**SHAP** works by hiding things. It hides a random subset of the 20 weeks, sees how the forecast
changes, and repeats that thousands of times until it can attribute credit fairly.

**Integrated gradients** works by fading things in. It starts from a neutral reference, which here
is each district at its own typical case level, and slides smoothly up to the real data, measuring
how sensitive the forecast is along the way.

Neither is a guess. Both come with a mathematical fairness guarantee. The difference is what they
cost and what they can be trusted to reproduce.

### The evidence

![**Figure 1.** *SHAP against integrated gradients on our own model. Both methods use the same reference point and the same frozen model, so nothing here is a rigged comparison. A: cost of one complete explainability read across the panels we publish, on a log scale. B: the same SHAP calculation run twice with different random draws, plotted against itself; points off the diagonal are the method disagreeing with itself. C: how often SHAP's hiding procedure builds an input that could never occur in real data.*](../figures/shap_vs_ig.png)

**1. Cost. SHAP is between 12,000 and 1.8 million times more expensive here, and the reason is
structural rather than fixable.**

| panel | districts | SHAP model runs | integrated gradients | ratio |
|---|---|---|---|---|
| Ebola (L12) | 61 | 89,948,160 | 5,760 | 15,616x |
| influenza, US states | 49 | 96,337,920 | 7,680 | 12,544x |
| dengue | 7,165 | 14,086,963,200 | 7,680 | 1,834,240x |

These are exact counts for one complete read across five random seeds, not estimates. The reason for
the gap is that SHAP has to hide and re-run for **each district separately**, so its cost multiplies
by the number of districts. Integrated gradients gets every district in the same sweep, so its cost
does not depend on district count at all. That is why the dengue row is so extreme: 7,165 districts.

In wall-clock terms on our hardware, the dengue read alone is somewhere around 14,000 GPU hours,
which is over eighteen months of continuous compute for one figure in one paper. The exact hour
figure moves a little between runs depending on machine load, which is precisely why we give you the
model-run counts as the headline: those are fixed arithmetic.

**2. Reproducibility. SHAP gives a different answer every time it runs. Integrated gradients does
not.**

We ran the same SHAP calculation twice on the same district, changing only the random draw:

| sampling budget | agreement between two runs | time per district |
|---|---|---|
| 256 | r = 0.68 | 1.0 s |
| 1,024 | r = 0.94 | 4.8 s |
| 4,096 | r = 0.98 | 17.0 s |
| integrated gradients | **r = 1.000000** | 0.8 s for all districts at once |

At a budget anyone could afford across this many districts, SHAP and SHAP disagree at r = 0.68. This
matters beyond tidiness. Our whole reporting standard on this project is that any number in a
document can be recomputed from the stored results and must come out the same. A method that lands
somewhere slightly different every run cannot be audited that way.

**3. SHAP asks our model questions that have no real-world answer.**

One of our four channels records whether a district reported at all that week. It is tied to the
case-count channel: if a district did not report, there is no case count. SHAP hides channels
independently, so it routinely builds weeks that say "this district did not report, and here is the
number it reported". **41.6% to 44.9% of SHAP's sampled inputs contain at least one such
impossible week.** The model has never seen anything like it in training, so whatever it answers
there is extrapolation, and that answer is then folded into the explanation.

Integrated gradients fades all four channels together along one path, so its intermediate points are
partial but never self-contradictory.

**4. And the important control: where both can be run, they agree.**

This is the part that should settle it rather than the cost. On the five busiest Ebola districts we
ran both methods properly and compared:

| district | integrated gradients | SHAP | agree |
|---|---|---|---|
| liberia, montserrado | past case counts | past case counts | yes |
| guinea, macenta | past case counts | past case counts | yes |
| liberia, bong | past case counts | past case counts | yes |
| sierra leone, port loko | past case counts | past case counts | yes |
| sierra leone, bombali | past case counts | past case counts | yes |

Five of five, with the detailed agreement between the two methods running from r = 0.96 to 0.99. So
integrated gradients is not giving you a different answer from SHAP. It is giving you **the same
answer, about fifteen thousand times cheaper, and identically every time it runs**.

**One disagreement, disclosed.** On a sixth district, guinea/beyla, which is almost empty of cases,
the two methods picked different top channels. We are telling you because it happened, not because
we think it changes the conclusion: in a district with almost no cases there is very little for
either method to find, and the two attributions still correlated at r = 0.86 to 0.99.

### What we also did, so the substitution is not a single method taking our word for it

We did not simply swap one method in and stop. Alongside integrated gradients we run **occlusion**,
which is the same hide-and-measure idea SHAP is built on, applied one piece at a time. The two
methods agree on the top channel and the top time window in **264 of 280** checks. Where they
disagree, we name the case rather than smoothing it over.

We also stated three predictions **in writing before running anything**, and required the method to
fail if they came out wrong. All three passed. That is a stronger discipline than either method
carries on its own.

### The decision

> **D1. ACCEPTED, 2026-09-16.** Integrated gradients with an occlusion cross-check is the delivery
> of G5, in place of SHAP. The reasons on record are the compute cost and the fact that the two
> methods give the same answer where both can be run.
>
> Our recommendation was yes. SHAP at this problem size is not a budget question, it is a method that
> would take years of compute for dengue and would still produce a figure nobody could reproduce
> exactly. Where the two can both be run, they agree.
>
> If your answer is no, that is a legitimate call and the consequence is concrete: G5 would have to
> be renegotiated in scope, most likely by cutting it down to a handful of districts on one panel,
> because a full SHAP read across all panels is not achievable.

---

## Decision 2. The comparison table you already have

`RelatedWork_CompetitiveAnalysis_Benchmark.docx` contains a comparison table whose row for our own
method reads **"yes, SHAP (global + local)"**.

That sentence is not true and has not been true at any point. It should read **"yes, integrated
gradients with occlusion cross-check (global + local)"**, which is true, evidenced, and if anything
a stronger claim than the original, because it comes with a reproducibility guarantee SHAP does not
offer.

We are raising this ourselves rather than waiting for a reviewer to find it. We would like to send a
corrected version of that document.

> **D2. APPROVED, 2026-09-16, and done the same day.**
> `Reports/Phase1/RelatedWork_CompetitiveAnalysis_Benchmark_corrected_2026-09-16.docx` is the
> corrected copy. The original is untouched, so both versions exist and it is clear which is which.
> Exactly one cell changed, table 0 row 11 ("Ours"), the explainability column:
>
> | | |
> |---|---|
> | was | `yes - SHAP (glob.+loc.)` |
> | now | `yes - integrated gradients + occlusion (glob.+loc.)` |
>
> The edit was made by `diagnostics/retract_shap_row.py`, which refuses to run unless it finds the
> expected row, and which reads the saved file back and fails if more than that one cell moved.

---

## Decision 3. The neighbour figure, in plain words

### What it is

The model does not look at each district on its own. Districts are connected to their geographic
neighbours, and information flows along those connections. One of our explainability outputs measures
that flow: for each of the 61 Ebola districts, we cut all of its connections and see how much its
forecast moves.

Two things come out of it, and both are worth having:

- **Districts with no Ebola data of their own lean harder on their neighbours.** The 43 districts
  where we had no case data score 0.143 on our influence measure, against 0.122 for the 18 districts
  that did have data. The model leans on neighbours most exactly where it has nothing else.
- **For 23 of those 43 data-less districts, the single strongest influence is a district that DID
  have data.** Information is flowing from the places we could observe into the places we could not.
  That is the mechanism the whole project is built on, visible directly.

### Why we need a decision about it

Because there is something that figure cannot be allowed to say.

Separately from all of this, we ran a test where we switch the district connections off entirely and
retrain. If the connections were improving the forecasts, accuracy would get worse when we remove
them. **It does not. Across 40 accuracy comparisons, switching off the connections improved accuracy
in 0 and made it worse in 8.** The connections help the shape of the forecast curve in some cases,
but they do not measurably reduce forecast error.

So we have two true statements that sound contradictory to a casual reader:

1. The model draws heavily on neighbouring districts, especially where it has no local data.
2. Drawing on neighbouring districts does not measurably improve its accuracy.

Both are ours, both are measured, and we will not publish the first without the second attached.

### Why the connections do not improve accuracy

A result is not a reason, and a reviewer will ask for the reason. We measured it. It is a chain of
three steps and each one is checkable.

![**Figure 2.** *Why the district connections do not improve accuracy. 1: the model is not ignoring them, the learned gate is wide open on every panel and not one district is near closed. 2: there is very little for the connections to carry, and averaging over neighbours deletes most of even that. 3: so attaching the wrong districts to the same graph barely changes the error.*](../figures/why_graph_fails.png)

**Step 1. The model is not ignoring the graph.** There is a learned switch that controls how much
neighbour information flows in. It could have closed it. It does the opposite: the switch sits
between 0.27 and 0.38 across panels, and **0.0% of districts on any panel at any seed sit near
closed**. So "the graph does not help" is not "the model turned the graph off". It is using it.

**Step 2. There is almost nothing district-specific for the graph to carry, and averaging deletes
most of what there is.** The model summarises each district as 64 numbers. We split that summary
into the part every district shares and the part that is genuinely that district's own:

| panel | the district's own share | uses how many of 64 directions | survives neighbour averaging | destroyed |
|---|---|---|---|---|
| influenza, Japan | 3.8% | 2.3 | 3.3% | 14% |
| influenza, US regions | 8.6% | 1.8 | 3.5% | 59% |
| influenza, US states | 12.3% | 2.5 | 5.0% | 60% |
| COVID, US states | 4.8% | 1.9 | 3.9% | 18% |

Read the first column first. Between **87% and 96% of what the model says about a district is
identical for every district**. Only a few percent is that district's own, and that few percent
occupies about **2 of the 64 directions available to it**. The model has a 64-dimensional vocabulary
for describing a district and is using roughly two words of it.

Then the second problem. Message passing averages over a district's neighbours. If everyone's
summary is mostly the same summary plus a small personal wobble, averaging a handful of them cancels
the wobbles and leaves the shared part. That is exactly what we measure: neighbour averaging
destroys **14% to 60%** of the little district-specific content that existed.

**Step 3. So the identity of the neighbours barely matters.** The test: keep the graph's exact
shape, keep every district's number of connections identical, but attach the **wrong districts** to
it and re-measure error. If the model were genuinely using who its neighbours are, error should jump.

| panel | cost of attaching the wrong districts | that panel's noise between random seeds |
|---|---|---|
| influenza, Japan | +0.03% | 1.9% |
| influenza, US regions | +1.14% | 4.8% |
| influenza, US states | +0.76% | 0.7% |
| COVID, US states | +0.26% | 1.7% |

**Knowing its real neighbours is worth at most 1.14% to the trained model.** On three of the four
panels that is a fraction of the run-to-run noise from simply changing the random seed. We are
flagging the one panel where the tidy version does not hold: on influenza US states the cost, 0.76%,
is about equal to that panel's seed noise of 0.7% rather than a fraction of it. It is still under
one percent.

**Put the chain together and the ablation result stops being surprising.** The gate is open, but
what flows through it is almost the same vector for every district, and averaging removes most of
what little was distinctive. By the time the neighbour message arrives it carries very little that
identifies who sent it. Remove it and you remove very little. That is why accuracy does not move.

**What this reason is worth, and what it is not.** These measurements run on the single-disease
models rather than the cross-disease transfer trunk, and they exclude dengue, whose 7,165 districts
make the permutation sweep time out. Excluding dengue is conservative: it has the highest
district-specific share of any panel at 22.4%, so it is the panel most favourable to the graph. And
this asks whether a **trained** model uses the real graph. It does not ask whether a model trained
from scratch on a fake graph would do just as well, which is a different and stronger test we have
not run.

### What we tried in order to make the spatial side pay off

The graph was not left to fail. Four things were tried before we concluded it does not help, and one
was deliberately not tried.

**1. We gave the model the option to reject the graph, and it declined.** The architecture includes a
learned switch per district, so the model itself can shut neighbour information out if it is
unhelpful. Trained with no pressure either way, it opened the switch to 0.27 to 0.38 and left
**0.0%** of districts near closed. The model votes to keep the graph. The graph then fails to earn
its keep on accuracy. Those two facts together are the finding.

**2. We used two message-passing layers rather than one**, the depth chosen on the grounds that three
or more over-smooths, meaning districts blur into each other. Our own measurements at depth 2 show
the blurring already happening: 87% to 96% of a district's summary is shared with every other
district. More depth would make that worse, not better.

**3. We added epidemiology to the training objective, at two strictness levels, and it changed
nothing.** This is the epidemiology-informed component the brief suggested. Rather than let the model
predict any trajectory, we penalised week-to-week growth the epidemiology says is implausible, with
the threshold set at two levels. Same trainer, same seeds, same early stopping, compared seed by
seed against the baseline:

| version | cells compared | better | worse | within noise |
|---|---|---|---|---|
| looser bound (p90max) | 48 | 2 | 1 | 45 |
| stricter bound (p99max) | 48 | 0 | 0 | 48 |

Ninety-six comparisons across four panels, four horizons, three metrics and five seeds, and **two
cells improved**. That is what chance looks like. The epidemiological prior did not rescue the
spatial side either.

**4. We measured whether a different way of combining neighbours could do better.** Averaging is not
the only option. On the same trained features we measured what each alternative could see:

| how neighbours are combined | district-specific content it preserves |
|---|---|
| mean, what we use | 3.3% to 5.0% |
| max | 2.9% to 8.8% |
| sum | 13.2% to 17.1% |
| spread, highest minus lowest | **24.5% to 26.7%** |

`max` collapses just as badly as the mean, so switching to it buys nothing. `sum` looks three to five
times better and the gain is an illusion: the sum is just the number of neighbours times the mean,
and that count is already fed into the model separately. Only **spread** is genuinely promising, and
it is the one thing averaging destroys by construction, because "my neighbours all agree" and "my
neighbours disagree wildly" produce the same average.

**5. What we deliberately did not run.** A version where the model is retrained from scratch on a
deliberately fake graph, roughly six hours of compute. We judged it would not change the conclusion,
since switching the graph off entirely already costs nothing, and we record it as a decision rather
than an oversight. It remains the stronger test and we say so in the paper.

**One thing this opens up rather than closes.** Averaging is a choice, not something the method
forces on us. We measured what other ways of combining neighbours could see, and the spread between
a district's neighbours, the difference between the highest and lowest, is 24.5% to 26.7%
district-specific, which is **higher than the district's own summary**. Averaging destroys it by
construction, because "my neighbours all agree" and "my neighbours disagree wildly" produce the same
average. That is a hypothesis about a possible improvement, not a result: information being
available is not the same as information that helps forecasting, and testing it would need a full
retrain. We mention it because it is the obvious next question and we would rather raise it than
have it raised at us.

### What the decision is

> **D3. ACCEPTED, 2026-09-16.** The neighbour figure ships, describing what the model reads, with
> the explicit statement that it is not a source of accuracy.
>
> In practice that means the figure ships with a caption saying, in substance: this shows where the
> model draws its information from, and our ablation shows that information does not improve
> accuracy.

**What saying yes achieves.** We keep the spatial half of the explainability deliverable, the part
that actually shows cross-district transfer happening, and we publish it in a form no reviewer can
attack, because we have stated the limitation before they can. Honest negative results about a
component are publishable and this project already has several; they are part of why the work reads
as careful rather than promotional.

**What saying no achieves.** We drop the figure. The explainability deliverable then covers time and
channels but says nothing about space, which is a visible gap given the model is a spatial model.

**What we will not do either way** is publish the figure without the caveat. That would let a reader
conclude the graph improves forecasts, which our own evidence says it does not.

---

## What these three decisions unlock

| decision | what it releases |
|---|---|
| D1 | G5 can be signed off. It is otherwise complete: code, tests, verification and figures are all finished and committed |
| D2 | the circulating comparison document stops asserting a capability we do not have |
| D3 | the spatial explainability figure can go into the manuscript, with its limitation attached |

G5 is the last open requirement in this milestone. Everything technical is done, verified and on
disk. These three answers are the only thing standing between it and closed.

---

## Where to check any of this

| claim | reruns with |
|---|---|
| the SHAP comparison, all three panels of Figure 1 | `diagnostics/graph_probe/shap_vs_ig.py` |
| channel and time-window attributions, the 264 of 280 agreement, the three pre-stated tests | `python -m explain --report` |
| the neighbour influence numbers | `results/reports/explain_report.txt`, Ebola neighbour ablation section |
| the 0 of 40 accuracy result | `results/reports/gate_ablation.log` |
| the three-step reason, all of Figure 2 | `diagnostics/graph_probe/why_graph_fails.py` |
| the aggregator hypothesis, 24.5% to 26.7% | `diagnostics/graph_probe/t4.py` |
| independent verification of all of the above | `progress/outcomes/G5_Explainability_Results.md` |

# Meta-Learning: Our Answer, and a Choice for You

**31 July 2026 · Phase 3, Week 4**

You asked for two things this week: pick a meta-learning method and tell you straight whether it fits
the schedule. Both answered below. The scope question at the end is genuinely yours to decide, so we
have set out the options and their costs rather than choosing for you.

---

## The short version

**The method: ANIL.** It is MAML applied only to the small adapter layer, not the whole network. We
are confident in this one and it is our call to make.

**The schedule: no, not as it stands.** The full version does not fit the fifteen working days left
without pushing into Weeks 5 and 6. One reduced version probably does. We will know for certain
within days from a timing measurement.

**One correction we owe you.** We previously told you that what we built is "not a linear probe". You
were right and we were wrong. Details in section 3.

**And one thing worth knowing before you decide anything.** There is a one-night experiment that
could show this whole approach will not work. It is ready to run. If it comes back negative, it saves
us a week and saves you a decision.

---

## 1. Why ANIL

Our model has two parts: a large shared "trunk" that learns general patterns, and a tiny adapter
(about 1,400 numbers) that tunes it to one specific disease.

The Ebola plan is to freeze the trunk and fit only the adapter on the small amount of Ebola data we
have. ANIL trains the model to be good at exactly that operation. Standard MAML would optimise for
adapting the whole network, which is not what we will ever do in practice.

The two alternatives are out for concrete reasons, not preference:

- **ProtoNet** is built for classification, sorting things into categories. We are predicting case
  counts, a continuous number. Making it fit would be a research project of its own.
- **Reptile** has no mechanism for the specific comparison you asked for. It cannot tell us "does
  meta-learning beat the simpler approach", which is the whole point of running it.

ANIL also makes that comparison clean. Our current freeze-and-adapt method is ANIL with one piece
swapped out, so comparing them changes exactly one thing.

---

## 2. Why the schedule does not work

The problem is ordering, not difficulty. The code itself is not hard.

Ebola gets scored **once**, against a configuration we lock beforehand. That is a deliberate part of
our credibility story. But it means meta-learning has to be completely finished before Ebola starts,
and Ebola has to finish before the uncertainty and explainability work begins. Nothing can run
alongside anything else.

So every day meta-learning takes is a day taken from Weeks 5 and 6, not a day shared with them.

Some real numbers. Meta-learning does not add to the existing training, it replaces it, so the full
set of training runs has to be done again. We measured our existing runs at roughly 55 minutes each,
about 9 to 10 hours of GPU time for a full set, on one machine running one job at a time.

We are on day 16 of 30. Weeks 5 and 6 are already full, and four items marked REQUIRED in the brief
have not been started.

---

## 3. Where we were wrong

We ran an adversarial review over our own first draft before sending it. It found three things worth
correcting to your face.

**You were right about the linear probe.** We claimed our adapter was something more sophisticated.
It is not. With the trunk frozen, the maths collapses exactly to a simple linear read-out, and we
confirmed this numerically. Your original description was accurate. This actually helps the proposal,
because that is precisely the setup ANIL is designed for.

**We quoted a stale number.** We described the adapter as 388 values. It is 1,428. The old figure
predates a change we made weeks ago and nobody updated it. Now fixed everywhere.

**We quoted two results that were not statistically solid.** Two of the dramatic failure numbers in
our draft do not clear the noise floor, which breaks the reporting rule you set. We have replaced
them with the ones that do hold up. The overall picture does not change, but the specific numbers we
led with were not defensible.

---

## 4. The honest problem with doing this now

We only have two diseases to develop on: dengue and flu.

Meta-learning works by practising on many different situations. If we hold out flu, we have only
dengue left to practise on, so the model can only practise varying *populations*, not *diseases*. And
population is the part that already works. Disease is the part that fails.

Put simply, we would be training it on the easy problem and testing it on the hard one. A reviewer
will spot this immediately, so we are raising it ourselves.

**Adding COVID-19 as a third disease is the real fix**, which is option D below. It would give us two
diseases to practise on instead of one. We should be clear that COVID is not in our pipeline at all
right now, so this is a genuine chunk of data engineering work, not a setting we can switch on.

**Second honest point.** Ebola has no usable data for adaptation at the 10 and 15 week horizons, none
at all. Meta-learning cannot help where there is nothing to adapt on. So the case for it rests on the
3 and 5 week horizons, and that is how we should judge it.

---

## 5. Your options

We are not picking between these. Each protects something different.

| | option | what it costs | what it protects |
|---|---|---|---|
| **A** | Extend by one week | One week of schedule | Everything. Full meta-learning, both comparisons, Weeks 5 and 6 untouched |
| **B** | Keep 30 days, cut secondary work | Some baseline and ablation work, listed below | Full meta-learning, at the cost of a thinner comparison section |
| **C** | Keep 30 days, do half the meta-learning | We test one direction instead of two | Weeks 5 and 6 intact, and the question gets a real answer rather than an assertion |
| **D** | Add COVID-19 as a third disease | Data engineering work, plus schedule | The only option that fixes the problem in section 4 |

**If you choose B**, the safest things to drop are the MTGNN investigation (that model is producing
broken output and cannot go in the comparison either way, so recording it as a failed reproduction is
both cheaper and more honest), the single-seed sampler probe, and the older transfer table which we
have already flagged as not quotable.

**If you choose C**, we would train on dengue and test on flu. Dengue gives us far more variety to
learn from, and it matches the real situation, learning from something large and applying it to
something small and sparse, which is exactly Ebola.

**Option D is additive.** It works alongside A, B or C rather than replacing them.

**One thing we would push back on:** cutting the number of repeat runs to save time. It would break
our own reliability standard and make it *harder* to show a positive result, not easier. Cut
experiments rather than repeats.

---

## 6. What we are doing meanwhile

Work that is useful no matter which option you pick.

**Tonight, the experiment that could end this early.** Everything above assumes a bigger adapter
could recover some of the lost accuracy. Nobody has tested that. We can, directly: take one trained
model and try progressively larger adapters on it.

If a much larger adapter recovers nothing, then the problem is the trunk, not the adapter, and
meta-learning will not fix it either, because meta-learning does not make the adapter bigger. That
would be a clear answer for one night of computing, before anyone commits a week.

If a larger adapter *does* help, we have a cheap partial fix in hand and a much stronger case for
going ahead.

**This week**, we are building the training machinery, taking the timing measurement that settles
whether option C fits, and running a single trial. None of it commits us to anything.

**Two gaps we have already closed.** We found that we were never saving our trained models to disk,
which meant every result rested on something we could not reproduce, and the Ebola plan had no saved
model to freeze. We also found we were discarding the uncertainty predictions at the moment we made
them, which is why our confidence intervals had no evidence behind them. Both fixed. Neither depended
on any decision from you.

---

## 7. What we need back

**A choice from section 5**, ideally together with the Ebola support set decision we sent separately.
Both of them lock the same configuration, and answering them a week apart costs a week either way.

We are not putting a deadline on you, but we should be straight that the cost here is roughly one day
of Week 5 or 6 for each day the question stays open.

---

## 8. The result we may have to report

Better said now than in Week 6. If meta-learning also comes back negative, our headline claim about
transferring between diseases does not survive, and the paper becomes a careful negative result
instead: a corrected experimental design, a demonstration that the earlier positive result was an
artifact, and a method that was properly tested rather than assumed.

That is genuinely publishable, and several of the models we compare against list this exact problem
as unsolved. But it is not the contribution the brief describes, and the brief marks it REQUIRED. So
that is a decision for you and Nora, and it is better raised now than late.

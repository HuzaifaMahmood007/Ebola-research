# Ebola Support Set: A Decision We Need From You

**Date:** 30 July 2026
**Prepared by:** modelling
**Status:** decision required before the Ebola case study is run

## Why we are asking

The Ebola dataset is split into two parts. The **support set** is the early part of the outbreak that
the model is allowed to learn from. The **query set** is everything after it, which we forecast and
score. Once we score the query set we score it exactly once, so the split has to be right before we
start.

While hardening the audit we found that the current split gives the model far less to learn from than
we assumed, and that the amount is not fixable by a small adjustment. The options are genuinely
different, and each one changes what the paper is able to claim. That makes it your call, not ours.

Every number in this document was recomputed from `data/processed/ebola.npz` on 30 July 2026.

## The dataset, in plain terms

The Ebola compilation covers **61 districts** across Guinea, Liberia and Sierra Leone, at **weekly**
resolution, for **52 weeks** (April 2014 to March 2015).

Only **41% of the possible district weeks were ever reported** (1,299 cells out of 3,172). A district
has on average 21.3 reported weeks. One district has a single reported week. The district adjacency
graph is healthy: 146 edges, no isolated districts, 21 cross border links.

## What the current split gives us

The current rule takes everything up to 24 May 2014 as support. That is weeks 1 to 7.

That yields **27 labelled cells across 9 of the 61 districts**. Spread across those nine districts the
counts are 7, 6, 3, 2, 2, 2, 2, 2 and 1.

The model does not learn from single cells, it learns from examples. An example is one district at one
forecast origin, where the answer we are trying to predict falls inside the support period. Counted
that way, the current split gives us:

| Forecast horizon | Learnable examples | Districts covered |
|---|---|---|
| 3 weeks ahead | 16 | 8 of 61 |
| 5 weeks ahead | 6 | 2 of 61 |
| 10 weeks ahead | **0** | 0 |
| 15 weeks ahead | **0** | 0 |

At 10 and 15 weeks ahead there is nothing to learn from at all. This is arithmetic, not a modelling
choice. The answer we would need to check against sits at week 10 or 15, and the support period stops
at week 7, so there is nowhere for it to land.

## The finding that shapes the options

We expected that simply extending the support period would fix this. It does, but not smoothly,
because of a gap in the source data.

| Week | Date | Districts reporting | Cases reported |
|---|---|---|---|
| 7 | 24 May 2014 | 2 | 7 |
| 8 | 31 May 2014 | 3 | 22 |
| 9 | 7 June 2014 | 11 | 71 |
| 11 | 21 June 2014 | 9 | 167 |
| 12 | 28 June 2014 | 7 | 50 |
| 13 to 18 | July to early Aug | **0** | **0** |
| 19 | 16 Aug 2014 | 20 | 1,399 |
| 20 | 23 Aug 2014 | 34 | 513 |

**Weeks 13 to 18 contain no reports at all, from any district.** Six consecutive empty weeks. The
outbreak reporting effectively restarts at week 19.

This means a 16 week support period gives **exactly the same data** as a 12 week one. There is no
point choosing anything between 13 and 18.

## One piece of good news

We initially believed that a longer support period would eat into the data we score on. **It does
not.** The number of scored forecasts stays identical at every option below, because the earliest
forecast we can score already sits at week 22 or later.

The scored set stays at **1,151 forecasts at 3 weeks, 1,075 at 5 weeks, 866 at 10 weeks and 642 at 15
weeks**, covering 61, 61, 59 and 58 districts. No option in this document reduces that.

So the cost of choosing a longer support period is not measured in lost evaluation. It is measured in
what the paper can claim, which is discussed at the end.

## The options

| Support length | Labelled cells | Districts | 3 wk | 5 wk | 10 wk | 15 wk | Scale check |
|---|---|---|---|---|---|---|---|
| 7 weeks (current) | 27 | 9 of 61 | 16 | 6 | 0 | 0 | 3.8x off |
| 8 weeks | 30 | 10 of 61 | 19 | 9 | 0 | 0 | 3.6x off |
| 12 weeks | 59 | 18 of 61 | 48 | 38 | 18 | 0 | 2.5x off |
| 16 weeks | identical to 12 weeks | | | | | | |
| 19 weeks | 79 | 22 of 61 | 68 | 58 | 38 | 20 | 0.8x |
| 20 weeks | 113 | 36 of 61 | 102 | 92 | 72 | 54 | 0.9x |

The four horizon columns are learnable examples. Forecasting at 15 weeks ahead is impossible below a
19 week support period, for the same arithmetic reason as before.

## What the scale check column means

Before the model sees case counts we convert them to a common scale, and afterwards we convert the
predictions back. The conversion factor is calculated **from the support set only**.

With the current 7 week support set, that calculation is done on the quiet opening weeks of the
outbreak. The result is that the model's yardstick says a typical week is about **2 cases**, while the
weeks we actually score average **19 cases** and reach a maximum of **1,428**. A third of the data we
score on (426 of 1,272 cells) sits outside what the yardstick treats as normal.

The scale check column is the ratio between the two. **1.0 would mean the yardstick was calibrated on
the right size of numbers.** The current split is 3.8x off. A 12 week support period improves it to
2.5x. At 19 or 20 weeks it lands at 0.8x to 0.9x, because the support period then includes the point
where the outbreak actually took off.

This matters because the model's adaptation layer is the part responsible for correcting scale, and
at the current split that layer has 16 examples to work from at 3 weeks ahead, and none at all at 10
and 15 weeks.

## A limitation that applies to every option

Our forecasting window looks back 20 weeks. Inside any support period shorter than 22 weeks, a full
20 week history cannot be assembled, so the earliest windows are partly padded with blanks.

| Support length | Share of the window that is padding, 3 wk ahead |
|---|---|
| 7 weeks | 89% |
| 12 weeks | 69% |
| 19 weeks | 53% |
| 20 weeks | 39% |

A completely unpadded window first becomes possible at a **22 week** support period for 3 week
forecasts, and would need **34 weeks** for 15 week forecasts. At 34 weeks the support period would
cover two thirds of the entire outbreak, so we are not proposing it.

## What each option means for the paper

This is the part that is genuinely a judgement call rather than a calculation.

**Staying at 7 or moving to 8 weeks** keeps the strongest version of the few shot claim, which is that
the method works on an emerging pathogen with almost no history. The cost is that 10 and 15 week
forecasts can only ever be reported as zero shot, the scale correction has very little to work with,
and the whole Ebola result rests on 16 examples across 8 districts.

**Moving to 12 weeks** roughly doubles what the model can learn from, to 59 cells across 18 districts,
and makes 10 week forecasts adaptable for the first time. 15 week forecasts remain permanently zero
shot. The scale mismatch improves but does not go away.

**Moving to 19 or 20 weeks** makes all four horizons adaptable and largely resolves the scale problem.
The cost is the framing. At 113 cells across 36 districts this is no longer a few shot result. It
becomes a statement that given roughly the first five months of an outbreak, the method forecasts the
remaining seven. That is still a real and defensible claim, and arguably a more operationally
realistic one, but it is a different claim from the one in the brief, and the adaptation layer was
sized against the 27 examples the current split provides.

## What we need from you

We are not recommending one. The numbers do not select an option on their own, because the choice is
about which claim the Ebola case study is meant to support.

The question is: **do you want the strongest few shot framing, accepting that two of the four horizons
can never be adapted and that the result rests on very few examples, or do you want all four horizons
and a correctly calibrated model, accepting that the headline becomes moderate data transfer rather
than few shot?**

Two things worth noting before you decide. Nothing in this decision changes the evaluation set, so no
option costs us scored forecasts. And 16 weeks should be removed from consideration entirely, since it
is identical to 12.

Once you choose, we will freeze the configuration, hash it, and score the Ebola set once against it,
as previously agreed.

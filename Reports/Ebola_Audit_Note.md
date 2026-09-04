# Ebola dataset audit: the numbers you asked for

**Date:** 2026-07-30 · **Source:** `data/processed/ebola.npz`, computed 2026-07-29 · **Record:** `decisions.md` D4

You asked for this in writing and called it the biggest open risk on the project. Here it is, with the
arithmetic run out loud as requested.

Two of your assumptions turn out to be wrong in our favour. Separately, the constraint you were
worried about is real, and it is one you already signed off: **[CONFIRM-P7]**, which you approved at
Day 13. What that approval did not have in front of it was the size of the bill. This note supplies
it, and adds two further problems we found only when hardening the audit last week.

---

## 1. The five things you asked for

| question | answer |
|---|---|
| Exact temporal resolution | **Weekly.** 61 districts × 52 weeks, 2014-03-24 → 2015-03-28 |
| Missingness across the 61 units | **41% observed** (1,299 of 3,172 cells). Mean 21.3 observed weeks per district, min 1, max 40 |
| Usable adjacency graph? | **Yes.** 146 edges, **0 isolated nodes**, 21 of them cross-border. GADM 4.1 |
| Labelled examples the support set gives | **27 observed cells, across only 9 of the 61 districts** |
| Usable time origins after the split | See §2. The answer depends on what you count, and that is the crux |

Support set is the calendar prefix ≤ 2014-05-24, i.e. weeks 1–7.

## 2. Your arithmetic, run out loud

You wrote: *"a 20 week input window plus a 15 week horizon eats 35 weeks per sample. If that data is
weekly we might be looking at something like 17 usable time origins before we split anything."*

**Per district, you are almost exactly right.** With T = 52, lookback w = 20 and horizon h, an origin
t must satisfy t ≥ 20 and t + h ≤ 52:

| horizon | origins per district |
|---|---|
| h3 | 30 |
| h5 | 28 |
| h10 | 23 |
| **h15** | **18** ← your estimate was 17 |

**But the evaluation unit is not the origin, it is the (district, origin) pair.** Every origin is
scored across all districts that have an observed target at that step. Pooling over the panel:

| | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| **Query/eval pairs** | **1,151** | **1,075** | **866** | **642** |
| Districts covered | 61 | 61 | 59 | 58 |

## 3. Correction 1: h10 and h15 *are* evaluable

You feared *"h10 and h15 may not be meaningfully evaluable on Ebola at all, and that matters
enormously because far horizon is where transfer currently underperforms."*

At h15 we have **642 observed pairs over 58 districts**. That is a scoreable test set. Your ~17
figure was per-district and the pooling over 61 districts rescues it.

The real characterisation is: **h10 and h15 are zero-shot, not unmeasurable.** We can score them; we
just cannot *adapt* at those horizons (§5). Those are very different problems and only the second one
is real.

## 4. Correction 2: the 27-example floor came from Ebola itself

You wrote: *"The encoder memo set the few-shot design floor at 27 labelled examples, taken from the
smallest development set. If Ebola comes in under that number, we calibrated the constraint against
the wrong thing."*

**This is backwards, and the good way round.** 27 *is* Ebola's own support-cell count. It was never
taken from a development set. So the constraint was calibrated against exactly the right thing, and
Ebola cannot "come in under it", it *is* the number.

## 5. What P7 actually costs, now that we have counted it

**To be clear about provenance: this constraint is not new, and we are not presenting it as a
discovery.** It was written up in `encoder_architecture_plan.md` §2.1 before you asked, escalated to
you as **[CONFIRM-P7]** (the execution guide called it "the most consequential" decision and "the
go/no-go on the Week-4 design"), and you settled it at Day 13. The plan stated it plainly:

> "Support targets end at 7; the earliest target is 22. The sets are disjoint. There are zero support
> origins, and the inner loop has nothing to supervise on."

> "h=3 admits origins t in [0,4], h=5 admits t in [0,2], and h=10 and h=15 admit none, ever."

**What was never quantified is how many labelled examples actually survive after missingness.** The
plan gives the origin ranges. It does not give the realised counts, because those depend on which
districts had reported by each week. That is the gap this audit closes, and the answer is smaller
than the origin ranges suggest.

A support cell only becomes a labelled adaptation example if it can be reached as a (district, origin)
pair with its target inside the support window. Counted properly:

| adaptation regime | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| Full 20-week lookback | **0** | **0** | **0** | **0** |
| With P7 left-pad (shortened, left-padded lookback) | **16 pairs / 8 districts** | **6 pairs / 2 districts** | **0** | **0** |

Three things follow:

**(a) The left-pad is load-bearing, not an optional variant.** With the full 20-week window there are
zero adaptation examples at *every* horizon, h3 included. Support ends at week 7 and the earliest
full-window origin is week 20, so the two never meet. This is the disjointness the plan predicted;
the audit confirms it holds against the built bundle, not just against the stated dates.

**(b) h5 adaptation is 6 examples on 2 districts.** This is the number the approval did not have.
"h=5 admits origins t in [0,2]" reads like three usable origins per district. After missingness it is
six pairs across two districts. The adaptation surface is **1,428 parameters**, so we would be fitting
1,428 values on six observations from two districts. That tells us nothing we could defend in review.

**(c) h3 is 16 pairs across 8 districts.** Also new, and also thinner than the origin range implies.

**Therefore: the practical adaptation horizon on Ebola is h3 only.** h5 is nominally available under
P7 and should not be used. h10 and h15 are zero-shot by construction, exactly as approved.

## 6. Two further problems, found when we hardened this audit

Neither of these was in scope when P7 was decided.

**(a) The scaler is calibrated on the wrong part of the outbreak.** Before the model sees case counts
we convert them to a common scale, and the conversion factor is computed **from the support set
only**. On the current 7-week support that calculation is done on the quiet opening weeks. The result
is a yardstick that treats a typical week as about **2.1 cases**, while the weeks we actually score
average **19.19 cases** and reach a maximum of **1,428**. **426 of the 1,272 scored cells (33%)** sit
beyond one standard deviation of what the scaler treats as normal.

This matters because the adaptation layer is the component responsible for correcting scale, and per
§5 it has 16 examples at h3 and none at h10 or h15. The part meant to fix the problem is the part with
almost no data.

**(b) Some districts are too sparse to score meaningfully.** Twelve districts have 10 or fewer
observed weeks, seven have 5 or fewer, and one has a single scored cell. None are dropped, so under a
per-district average that district carries the same weight as one scored on 35 cells.

**A related decision is now open.** Both problems above are sensitive to how long the support window
is, and the options are not evenly spaced, because weeks 13 to 18 of the source data contain no
reports at all from any district. We have set the options out with the numbers in a separate note,
`Ebola_Support_Set_Decision.md`, and deliberately left the choice to you, because it changes what the
Ebola case study is able to claim.

## 7. What this means for the horizon-set decision

This feeds straight into the horizon-set question you are taking to Nora:

- **h3**: the only horizon with genuine few-shot adaptation (16 pairs / 8 districts). Thin, but real.
- **h5**: reported as zero-shot, or dropped. Do not present it as adapted.
- **h10, h15**: zero-shot only, and scoreable (866 and 642 pairs). Worth keeping precisely because
  they are the honest far-horizon test.

The uncomfortable read: our adaptation story rests on **one horizon and 16 labelled pairs**. The
zero-shot story is much better supported. If the paper's Ebola claim leans on adaptation rather than
zero-shot transfer, it leans on those 16 pairs, and a reviewer will find that.

## 8. One thing to flag about how we get to score this at all

Ebola is scored **exactly once**, against a frozen, hashed, committed config (§8h pre-registration).
Everything above is computed from the data's structure and missingness. No model has been fit on
Ebola and no Ebola score exists. That is deliberate and it is what keeps the single-shot evaluation
honest.

Consequence worth naming: because we only get one shot, we currently have **no dev-set rehearsal of
the 27-cell regime**. We decided to defer the k-shot sweep, so the first time the procedure is
exercised at Ebola's real data budget will be on Ebola itself. If the result comes back weak we will
not be able to separate "transfer does not work" from "16 pairs cannot fit anything." Flagging it now
rather than after the fact.

---

**Bottom line.** Resolution weekly, graph usable with no isolated nodes, missingness 41%, query sets
comfortably scoreable at all four horizons. Your two specific worries were both misplaced.

The genuine constraint is the one you already approved as P7, priced for the first time: adaptation
exists only at h3, only with the left-pad, and only on **16 examples across 8 districts**. The
approval was sound on the structure. It did not have these counts in front of it, and h5 in
particular is far thinner than "admits origins t in [0,2]" reads.

Two further problems are new and were not in P7's scope: the scaler is calibrated on a period whose
mean case count is **3.8x lower** than the period we score on, and a handful of districts are too
sparse to carry the weight the averaging gives them. Both are affected by the support-window choice
now sitting with you in `Ebola_Support_Set_Decision.md`, which uses that same 3.8x ratio as its scale
check so the two notes can be read against each other.

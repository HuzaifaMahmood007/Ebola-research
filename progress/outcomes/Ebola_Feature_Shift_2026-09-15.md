# Why few-shot adaptation hurts on Ebola: the fit region is not the scoring region

**Run and recorded 2026-09-15.** Script: `diagnostics/ebola_feature_shift.py`. Output:
`results/misc/ebolashift__feature_shift.json`. Inference only, no training, about a minute to run.

**No Ebola label was read and nothing was scored.** The one prediction column compares two adapters
against each other, never against truth, so the pre-registration's score-once rule is untouched.

---

## In plain words

The model forecasts by looking at a 20 week window of case counts. For Ebola, the only weeks the
adapter is allowed to learn from are the first 12, right at the start of the outbreak, when almost
nothing had been reported yet. I measured what those learning windows actually contain: **72.5 per
cent of each one is zero padding** we insert because the outbreak had not been running long enough to
fill a 20 week window, and of the part that is real, **only 5.2 per cent of the cells carry any
observation at all.**

Then we ask the same adapter to forecast the rest of the outbreak. Those windows are full, no padding,
and about **31.6 per cent** of their cells carry real observations.

So we tune the adapter on near empty inputs and then use it on full ones. It is like calibrating a
scale with feathers and then weighing bricks on it.

There is a second problem on top of that. The adapter has 64 dials. On the primary 12 week arm the 48
training examples at h3 only ever move 45 of them. At h10, 18 examples move 17 dials. At h15 there are
no training examples at all, so nothing moves. **Every dial the data never touches keeps whatever
random value it started with**, because the protocol starts the adapter from scratch. At h15 the
adapted forecast still ends up 5.13 standard deviations away from the unadapted one, and all of that
movement is noise from the random starting values.

And the part that explains the result nobody could explain. The 20 week arm has more data, enough to
move all 64 dials at the short horizons, and it is numerically well behaved there. But its inputs
still come from the same quiet early period. So it tunes all 64 dials confidently to a region the
model is never scored in, and it moves the forecast **further** than the 12 week arm does. More data
made it worse rather than better. That is why counting parameters never explained this: the problem
was never how many dials there are, it was what we were showing them.

---

## Why I ran it

The Ebola headline is that few-shot adaptation hurts. Zero-shot beats few-shot in 31 of 32 cells, the
unadapted model wins 12 of 16 significant comparisons against persistence and the adapted model wins
2, and the pre-registered criterion was not met.

We had no explanation. The "about 50 parameters per observation" story was withdrawn on 2026-09-10
because it does not reproduce and predicts the wrong direction, since the 20 week arm has more data
per parameter and is hurt far more. `Session_Audit_2026-09-10.md:522-526` lists three untested guesses
and none of them concerns the adapter's inputs or its initialisation.

`Reports/Ebola_Audit_Note.md:159-163` also flagged, at the time, that we would not be able to separate
"transfer does not work" from "16 pairs cannot fit anything". This measures a third possibility that
is not either of those.

---

## The premise, checked before anything else

`ebola_L12`'s support mask is not a sampling design laid over an observed panel. It **is** the
observation mask over the first 13 columns. The script asserts this and refuses to continue otherwise:

```
ebola_L12: support == M[:, :13],  59 cells / 18 districts, query cells start col 19
ebola_L20: support == M[:, :21], 113 cells / 36 districts, query cells start col 21
pair profiles match configs/ebola_arms.json exactly: L12 48/38/18/0, L20 102/92/72/54
```

So a cell that is "not support" inside the fit window was never observed: no label, input incidence 0,
and obs_mask channel 0. With the trunk frozen the Adapter is exactly affine in trunk features
(`models/adapters.py:14-16`), so the few-shot arm is a 64 dimensional affine fit and the rows of its
design matrix are countable. `models/windows.py:18-20` already flagged the padding in its own
docstring.

Window occupancy, measured:

| origins | zero padding | observed share of the real part |
|---|---|---|
| L12 support, t 0 to 9 | **72.5%** | **5.2%** |
| L20 support, t 0 to 17 | 56.0% | 5.5% |
| query, t 19 to 36 | 0% | 31.6% |

---

## The measurement

Mean over seeds 42, 52, 62, 72, 82. `n_sup` is the literal row count of the affine design matrix at
that horizon.

| arm | h | n_sup | rank /64 | cond | shift med | shift max | out dims | pred gap |
|---|---|---|---|---|---|---|---|---|
| L12 | 3 | 48 | **45** | 2.43e18 | 0.61 | 7.38 | 18.7% | 0.84 |
| L12 | 5 | 38 | **37** | 8.63e17 | 0.62 | 8.28 | 25.8% | 0.56 |
| L12 | 10 | 18 | **17** | 2.26e17 | 0.63 | 8.87 | 40.6% | 1.96 |
| L12 | 15 | 0 | **0** | n/a | n/a | n/a | n/a | **5.13** |
| L20 | 3 | 102 | 64 | **1.61e04** | 0.56 | 5.17 | 15.6% | 1.03 |
| L20 | 5 | 92 | 64 | **2.20e04** | 0.55 | 5.70 | 18.7% | 0.69 |
| L20 | 10 | 72 | 60 | 2.36e18 | 0.58 | 6.91 | 19.6% | 1.92 |
| L20 | 15 | 54 | 34 | 7.07e19 | 0.62 | 11.03 | 31.5% | 2.91 |

Column meanings:

- **rank** is the numerical rank of the design matrix out of 64 feature dimensions. Below 64 means the
  fit is underdetermined and the unconstrained directions are set by initialisation and weight decay
  alone.
- **cond** is the condition number. Large means near collinear rows, so small label noise moves the
  fit a long way.
- **shift med** and **shift max** are the per dimension gap between the mean support feature and the
  mean query feature, in units of the query standard deviation.
- **out dims** is the average share of the 64 dimensions on which a query row falls outside the range
  the support rows covered. This is the extrapolation number to quote. The cruder "any dimension
  outside" version is in the JSON as `outside_box` and saturates at 100 per cent by construction with
  this few rows in 64 dimensions, so it is not evidence on its own.
- **pred gap** is the median absolute difference between the few-shot and zero-shot forecasts on query
  features, in units of the zero-shot spread. It is label free.

---

## What this explains

**The primary arm is rank deficient at every horizon.** 19 to 47 of the 64 feature directions are
never constrained by any Ebola label. What sets them is the random initialisation the pre-registration
specifies at `progress/decisions/Ebola_Prereg.md:126-127`, shrunk slightly by weight decay.

**The h15 column is the cleanest case.** Zero training rows, and the adapted forecast still lands 5.13
zero-shot standard deviations away from the unadapted one. That is precisely what `_check_e4`
(`train/ebola.py:195`) proves must happen: with no supervision the head block stays a uniform shrink
of its random initialisation. The gate was written to detect label leakage and it has been passing all
along. It was never wrong. It just also means the h15 forecast on the primary arm is close to a random
projection.

**The fit region is displaced from the scoring region.** The median feature dimension sits 0.55 to
0.63 query standard deviations away, the worst sits 5.2 to 11.0 away, and a query row lands outside
the support's observed range on 15.6 to 40.6 per cent of its dimensions.

**The L12 versus L20 inversion now has a candidate mechanism.** L20 at h3 and h5 is full rank and well
conditioned, 1.61e04 and 2.20e04 against L12's 1e17 to 1e18. So underdetermination does not explain why
L20 is hurt more. But L20's `pred gap` is larger at both horizons, 1.03 against 0.84 and 0.69 against
0.56. The reading is that L20 has enough rows to pin all 64 directions inside the support region, that
region is still displaced from the query region, and so it extrapolates further and with more
confidence. More support data buys a better determined fit to a region the model is not scored in.

This is orthogonal to parameter counting, which is why the withdrawn story failed and why nothing on
the Session Audit list of guesses covers it.

---

## What this does NOT establish

State these with the finding.

1. **`pred gap` is label free.** It measures displacement from the zero-shot head, not error. The step
   from "moved further" to "did worse" rests on the separately scored result that zero-shot wins, not
   on anything measured here.
2. **On L12, `rank` and `cond` are partly redundant.** A rank deficient matrix is singular by
   construction, so the 1e17 to 1e18 condition numbers on that arm are a restatement of the rank
   column, not independent evidence. Rank is the load bearing column.
3. **This is a property of the frozen arms and the all-dev trunk, measured on the checkpoints at
   `results/ebola/`.** It is not a counterfactual. It does not show that a better conditioned or better
   initialised fit would forecast better. That question needs the dev-fold rehearsal.
4. **The L20 mechanism above is a reading, not a proven chain.** Two label free quantities point the
   same way. It has not been tested against an alternative.
5. This does not retire the normalisation hypothesis in `CLAUDE.md` section 6. Both can be true, and
   the pooled-versus-per-node mismatch experiment is still unrun.

---

## What follows

The natural next step is the dev-fold rehearsal that `Reports/Ebola_Audit_Note.md:159-163` says we
never built: transplant the real `ebola_L12` support mask onto a held-out development panel, once with
the inputs blinded the way Ebola's are and once with them left clean, and see whether warm starting the
adapter from the zero-shot head recovers the loss. If it only recovers under the blinded placement, the
mechanism above is confirmed rather than merely measured. Full plan agreed separately; not started.

Note also that `progress/decisions/Ebola_Prereg.md:126-127` pins few-shot to a **fresh** adapter, so
anything that changes the initialisation on the Ebola arm is a protocol change needing
re-registration with the client, not a config edit. `Reports/Capacity_Probe_5Seed.md` records the same
constraint for the capacity result.

---

## Reproducing

```
conda run -n ebola-train python -m diagnostics.ebola_feature_shift --selfcheck
conda run -n ebola-train python -m diagnostics.ebola_feature_shift --seeds 42 52 62 72 82
```

Module form (`-m`) is required. Running the file by path puts `diagnostics/` on `sys.path` and the
`import bundles` fails.

`--selfcheck` runs without a trunk and asserts the three things this document rests on: the frozen pair
profiles still match `configs/ebola_arms.json`, the support mask really is the observation mask over a
calendar prefix, and the last support column precedes the first query cell column. It also runs
automatically before the main path, so the numbers cannot be produced against a moved arm.

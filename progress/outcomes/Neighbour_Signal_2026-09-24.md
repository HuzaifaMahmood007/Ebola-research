# Neighbour signal in the raw data (t6)

**Run 2026-09-24. Scored from disk the same day.** No model and no checkpoints. The user ordered this
before any spatial redesign. It asks whether the graph failure (gate-off ablation, inference relabel,
D2 shuffled retrain) is a property of the data or of the model.

## The question

Take out the shared weekly wave. After that, does a district's neighbours' past tell you more about
the district's future than the past of a random set of districts does?

## Verdict

Real neighbours win 13 of 20 cells.

Wins per panel: influenza_japan 4, influenza_us-regions 0, influenza_us-states 2, covid_us-states 3, dengue 4.

Bucket by the pre-committed rule: BROAD SIGNAL.

The rule was fixed before any result: NO SIGNAL if at most 1 win and no panel with 2 or more,
LOCALISED if exactly one panel has 2 or more, BROAD if two or more panels have 2 or more each. Four
panels have 2 or more wins.

The size of those wins is very uneven, and the calibrated reading below matters more than the count.
The case for a deviation channel rests on COVID (h3, h5) and dengue (h3 to h10). Japan is real but
tiny, US-states is borderline and carries no usable signal, and US-regions shows none.

On Ebola, run separately and EXPLORATORY (its own section below), the apparent neighbour signal is
district size, not timing. Once each district's own level is known, neighbours add nothing at h3 and
h5, and h10 cannot be judged.

## Design

- **Space.** The same input the encoder sees: incidence after log1p and a per-district z-score using
  the released TRAIN-period scaler. I checked the shipped scaler against a train-only refit on every
  district of all five panels and it matches exactly. The refit has to be done in float64. In float32,
  12 dengue districts with a constant train window get a spread of about 6e-8 from rounding, which
  dodges the 1e-8 "constant" test and skips the country fallback the real build correctly applied.
- **Deviation.** Each district's value minus the average of the observed districts in its group that
  week. The group is the country on dengue (12 countries), and the whole panel on the four
  single-country panels. Dengue has 0 edges that cross a border, out of 40,936.
- **Two predictors, both pooled ridge regressions** (linear, alpha 1.0, one fit shared by all districts):
  A uses the district's own last 4 weeks of deviation plus 4 observed flags. B adds the average
  deviation of its OBSERVED map neighbours over the same 4 weeks. That average is row-normalised with
  no self-loop, so it is a plain mean in the same units. The model itself uses a symmetric
  normalisation with a self-loop. I used the row mean because the district's own history is already
  in A.
- **Split.** Same origins as bundles.py. A row is used for fitting when its TARGET week is in the train
  mask, and for scoring when its target is in the test mask. Val is never used. Fitting rows are built
  from train-phase cells only, so no val or test cell enters a fit under the real map or any relabelled
  map. The script asserts this. Scoring rows read every observed past cell, as the model does.
- **Statistic.** gain = how much B cuts A's test error, as a percent of A's error.
- **Null.** B rebuilt on a relabelled map from train.loop.permute_adjacency: same shape, same number of
  neighbours per district, wrong districts. 200 relabels on the four small panels, 50 on dengue. A cell
  is a WIN when the gain is above zero AND above the 95th percentile of the relabelled gains.

## Controls

| control | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| positive 0.50 SD (gain %) | +21.788 | +8.931 | +0.969 | +0.379 |
| positive 0.25 SD (gain %) | +10.923 | +5.052 | +0.609 | +0.596 |
| legacy wrong-map, real base (gain %) | +0.620 | +0.546 | +0.676 | +0.911 |
| fixed wrong-map, shuffled base (wins of 20) | 1 | 1 | 1 | 1 |

**Positive control, passed.** On a copy of influenza_japan I planted spread: each district's value
gets a push from its real neighbours' deviation 3 weeks earlier. The push is sized at 0.5 times the
spread of the deviations, which is 25 percent of their variance. The probe had to WIN at h3 and h5. It
won at all four horizons, and also at 0.25 times (6.25 percent of variance). This certifies that the
probe finds a LINEAR neighbour effect of this size. It says nothing about effects of other shapes.

**Wrong-map control, and why the original run did not stop.** The original check planted the same
push through a relabelled map, on the real Japan data, then probed with the real map. It was flagged
WIN at every horizon, and the run carried on. The assert was magnitude-only: it asked whether the h3
gain was under half the positive control's h3 gain (0.620 against 21.788). It never read the WIN flag.
That check was also the wrong test. Real Japan data already carries a real-map signal (+1.214 at h3),
so the planted copy inherits it. Averaged over 20 wrong-map plants on the real base (t6b), the squared
error the real map removes at h3 is 0.00087, against 0.00095 with no plant at all: the plant barely
moves it. The legacy WIN flags were Japan's own neighbour signal showing through, not a false alarm.

**Fixed wrong-map control, passed.** I replaced it in t6. The new version shuffles which district owns
which series (whole rows move together, inside each group), so the real map means nothing for the data.
It then plants the push through a relabelled map and probes with the real map, 20 times. A calibrated
probe wins about 1 time in 20. The assert allows at most 3 of 20 at h3 and at h5, since 4 or more has
a 1.6 percent chance. Result: 1, 1, 1, 1 of 20. I also broke it on purpose by running the new check on
the real base: wins jumped to 20, 20, 13 and 17 of 20 and the assert fired with exit 1.

**Negative control, passed.** The median relabelled gain sits near zero in every cell. The largest
absolute median null gain is 0.220 percent, under the 1 percent flag.

## False-alarm check per panel

The same shuffled-base test as the fixed control, run on each panel with its own seeds, 20 draws and
200 relabels per draw (dengue: 6 draws and 20 relabels, see below). The rule was written before the
run: a false-alarm floor EXISTS if any horizon reaches the one-sided 5 percent binomial count, which is
4 or more of 20 (2 or more of 6). Floor size is the 95th percentile of the shuffled-base gains (the
largest of the 6 on dengue), and a real win below it is not read as signal.

| panel | h | false wins | draws | floor size % | real gain % | real win clears floor |
|---|---|---|---|---|---|---|
| influenza_japan | 3 | 0 | 20 | +0.127 | +1.214 | yes |
| influenza_japan | 5 | 2 | 20 | +0.210 | +1.007 | yes |
| influenza_japan | 10 | 1 | 20 | +0.120 | +0.284 | yes |
| influenza_japan | 15 | 1 | 20 | +0.117 | +0.698 | yes |
| influenza_us-states | 3 | 1 | 20 | +0.304 | +0.304 | no |
| influenza_us-states | 5 | 1 | 20 | +0.541 | +0.581 | yes |
| influenza_us-states | 10 | 2 | 20 | +0.654 | +0.627 | no |
| influenza_us-states | 15 | 3 | 20 | +0.865 | +0.249 | no |
| covid_us-states | 3 | 2 | 20 | +0.699 | +8.054 | yes |
| covid_us-states | 5 | 0 | 20 | +0.313 | +3.044 | yes |
| covid_us-states | 10 | 1 | 20 | +0.497 | -0.754 | no |
| covid_us-states | 15 | 1 | 20 | +0.472 | +1.321 | yes |
| dengue | 3 | 0 | 6 | +0.0020 | +5.543 | yes |
| dengue | 5 | 0 | 6 | +0.0037 | +5.281 | yes |
| dengue | 10 | 2 | 6 | +0.0046 | +2.865 | yes |
| dengue | 15 | 0 | 6 | +0.0022 | +0.673 | yes |

No floor exists on Japan, US-states or COVID. Pooled over the three panels, 15 of 240 shuffled-base
draws were flagged WIN, 6.25 percent, against a nominal 5. The relabel null is calibrated on these
panels.

**Dengue: the rule's trigger fired, and the floor is negligible.** Dengue ran the cheaper design, 6
draws of 20 relabels. At h10, 2 of 6 shuffled-base draws were flagged WIN, which meets the critical
count, so by the rule written in advance a floor EXISTS on dengue. Its size does not matter. The h10
floor is 0.0046 percent against a real gain of 2.865 percent, about 625 times smaller, and the
smallest ratio across the four horizons is about 307 at h15 (0.673 against 0.0022). Every dengue
horizon clears its own floor. The largest shuffled-base gain in all 24 dengue draws is 0.0046 percent.

**What 6 draws can and cannot rule out.** They rule out a false-alarm floor anywhere near the size of
the real dengue gains. They cannot pin down the false-alarm RATE: 2 of 6 fits a true rate anywhere from
4.3 to 77.7 percent (exact 95 percent binomial interval). A trigger is also not rare by chance: a
calibrated probe reaches 2 of 6 at a given horizon 3.3 percent of the time, and at one or more of four
horizons about 12.5 percent of the time if the horizons were independent, which they are not quite.
So the trigger says little about calibration either way, and the verdict rests on size, where the
margin is 307 to about 2,800 times. The within-country relabel null, the stricter null for dengue,
clears at 4 of 4 horizons as well (below).

US-regions was not run through this check because it has no wins to protect.

## Results

`gain` is the cut in the own-only model's test error, in percent. `share of total variance` translates
that into the cut in squared error as a percent of the total model-space target variance, i.e. gain
times A's error over the target's variance. It is approximate, it is in model space, and it is NOT
count-space RMSE.

| panel | h | gain % | null p95 % | p | verdict | share of total variance % |
|---|---|---|---|---|---|---|
| influenza_japan | 3 | +1.214 | +0.128 | 0.005 | WIN | +0.12 |
| influenza_japan | 5 | +1.007 | +0.162 | 0.005 | WIN | +0.11 |
| influenza_japan | 10 | +0.284 | +0.193 | 0.025 | WIN | +0.03 |
| influenza_japan | 15 | +0.698 | +0.156 | 0.005 | WIN | +0.07 |
| influenza_us-regions | 3 | -0.624 | +1.724 | 0.657 | no | -0.05 |
| influenza_us-regions | 5 | -1.708 | +2.034 | 0.776 | no | -0.23 |
| influenza_us-regions | 10 | -2.109 | +2.438 | 0.781 | no | -0.47 |
| influenza_us-regions | 15 | -0.587 | +2.303 | 0.642 | no | -0.17 |
| influenza_us-states | 3 | +0.304 | +0.315 | 0.065 | no | +0.05 |
| influenza_us-states | 5 | +0.581 | +0.535 | 0.055 | WIN | +0.14 |
| influenza_us-states | 10 | +0.627 | +0.625 | 0.055 | WIN | +0.21 |
| influenza_us-states | 15 | +0.249 | +0.642 | 0.184 | no | +0.09 |
| covid_us-states | 3 | +8.054 | +0.449 | 0.005 | WIN | +5.33 |
| covid_us-states | 5 | +3.044 | +0.390 | 0.005 | WIN | +2.28 |
| covid_us-states | 10 | -0.754 | +0.535 | 0.891 | no | -0.63 |
| covid_us-states | 15 | +1.321 | +0.851 | 0.025 | WIN | +1.17 |
| dengue | 3 | +5.543 | +0.005 | 0.020 | WIN | +1.79 |
| dengue | 5 | +5.281 | +0.007 | 0.020 | WIN | +2.44 |
| dengue | 10 | +2.865 | +0.006 | 0.020 | WIN | +1.97 |
| dengue | 15 | +0.673 | +0.005 | 0.020 | WIN | +0.52 |

p is the share of relabelled maps that did at least as well, with the standard plus-one correction.
The smallest possible p is 0.005 with 200 relabels and 0.020 with 50. The two US-states wins have
p = 0.055: they clear the interpolated 95th percentile by a hair, and that is as marginal as a WIN gets.

Dengue within-country null: 30 relabels that keep every district inside its own country. It clears at
4 of 4 horizons, with 95th percentiles of 0.004 or below.

On total model-space target variance, real neighbours remove 5.3, 2.3 and 1.2 percent on COVID at h3, h5 and h15, and 1.8, 2.4, 2.0 and 0.5 percent on dengue at h3, h5, h10 and h15.

A correction to a number that went round earlier: multiplying the gain by the deviation's share of
variance overstates dengue at short horizons, 4.3 instead of 1.8 at h3. The reason is that dengue's
own history already explains most of its h3 deviation, so the gain is taken off a much smaller error.

### What neighbour spread adds

Arm C adds the spread of the neighbours (max minus min), which t4 found to be the richest
district-specific channel in the representation. Secondary, and not part of the verdict.

| panel | h | own + mean (B) % | own + mean + spread (C) % |
|---|---|---|---|
| covid_us-states | 3 | +8.054 | +10.964 |
| covid_us-states | 5 | +3.044 | +6.036 |
| dengue | 3 | +5.543 | +5.385 |
| dengue | 5 | +5.281 | +5.159 |

Spread adds real value on COVID at short horizons and nothing on dengue.

### Is it just big districts next to big districts?

On Ebola (the EXPLORATORY section below) the neighbour gain vanished once each district's own average
level was added to the model. I checked the same thing on the dev panels with t6c: one extra column,
each district's mean train-phase deviation, added to A and B alike, then the real-map gain recomputed.

| panel | districts with a nonzero level | level SD | largest change in gain, points |
|---|---|---|---|
| influenza_japan | 0 of 47 | 0.0000 | 0.0000 |
| influenza_us-regions | 0 of 10 | 0.0000 | 0.0000 |
| influenza_us-states | 0 of 49 | 0.0000 | 0.0000 |
| covid_us-states | 0 of 49 | 0.0000 | 0.0000 |
| dengue | 6662 of 7165 | 0.2317 | 0.0588 |

On the four single-country panels the per-district z-score sets every district's train-period level to
exactly zero, so the extra column changes nothing. Dengue does have levels, because 78 percent of its
cells are unobserved and the set of reporting districts changes week to week. But adding them makes the
neighbour gain slightly larger, 5.543 to 5.602 at h3, not smaller. Neither panel that carries the design
case is level clustering. The script also asserts that its no-level gains equal the main run exactly.

## Calibrated reading

- **By the pre-committed rule: BROAD, 13 of 20.** That stands and is not re-bucketed.
- **The probe does not cry wolf.** No false-alarm floor on Japan, US-states or COVID. On dengue the
  rule's trigger fired at h10, but the floor there is about 625 times smaller than the real gain. The
  wins are not artifacts of the probe, and the dev wins are not level clustering either.
- **By size, only two panels matter.** COVID h3 and h5, 5.3 and 2.3 percent of total variance, and
  dengue h3, h5 and h10, 1.8, 2.4 and 2.0 percent. COVID h15 (1.2) is a modest win. Dengue h15 (0.5)
  is a win of trivial size. The same two panels survive a post-hoc 2 percent floor on the gain: COVID
  at 2 horizons and dengue at 3. That is still two panels with 2 or more each, so still BROAD. The
  floor was chosen after seeing the data, and I label it that way.
- **Japan is real but tiny.** Four wins, every one clears its false-alarm floor, and none is above
  0.12 percent of total variance. Not a design driver.
- **US-states is borderline, no usable signal.** Two wins at p = 0.055. The h10 win sits under its own
  false-alarm floor (0.627 against 0.654), and the two are 0.14 and 0.21 percent of total variance.
- **US-regions shows none.** Ten districts. The relabelled maps have 95th percentiles of 1.7 to 2.4
  percent, and the real map loses at every horizon.
- **Ebola (EXPLORATORY, not part of the verdict) shows no timing signal.** The first-pass wins at h3,
  h5 and h10 are district size. With size controlled, h3 and h5 are a real null and h10 has no power.
  Any gain a deviation channel brings is, on this evidence, scoped to the dev diseases.

The rule's wording for BROAD says "spread signal". What the probe actually measured is weaker:
neighbours carry forecast information. That is enough to justify trying a deviation channel, because a
model can use informative neighbours whatever the mechanism. It is not enough to claim spread. The
retrain is the user's call, not a default.

## How this fits what we already knew

Consistency only, not proof.

- D2 (the shuffled-adjacency retrain, 2026-09-23) found the trained model's real map helps error only
  on dengue, and never on COVID.
- t5 (input energy, 2026-09-22) found COVID's district-specific share falls from 23.8 percent in the
  input incidence to 4.8 percent in the encoder output. For dengue it falls from 93.5 to 22.4.
- Put together: on COVID the raw data carries neighbour-predictable deviation worth about 5 percent of
  target variance at h3, and the trained model's graph turns none of it into accuracy. One hypothesis
  is that the encoder compresses the deviation away before the graph ever sees it. That is the case
  for a deviation channel, stated as a hypothesis. Dengue, where more district structure survives the
  encoder, is also the one panel where the real map already helps a little.

## Ebola, EXPLORATORY

**EXPLORATORY. This cannot enter the case-study result.** It is a data probe with no model, no
checkpoint and no adapter. It reads the complete observed Ebola series, including the weeks the
pre-registered record was scored on, so it lives under the experiments/ policy: it writes nothing to
results/ or data/, it never re-scores the pre-registration, and it says nothing about the model's Ebola
forecasts or the pre-registered criterion. The frozen arm hashes were verified against
configs/ebola_arms.json before and after the run, and the output records both checks.

Script `experiments/t6_ebola_neighbour_signal.py`, output `experiments/t6_ebola_neighbour_signal.json`
(protocol EXPLORATORY).

**What changes from the dev probe, each forced by the data.**

- **Space.** The ebola_L12 input: log1p on one pooled scale for the whole disease. raw and M are
  identical across the three Ebola bundles and each scaler is a single constant, so the arms differ by
  a constant factor only. The L20 arm gives the same gains to within 0.023 points at h3 to h10.
- **Weeks 20 to 51 only.** Weeks 0 to 12 have 2 to 6 reporting districts, weeks 13 to 18 are the
  six-week reporting blackout, and week 19 is the backlog dump after it. Every feature and target cell
  is inside weeks 20 to 51, and the script asserts it.
- **Deviation per country**, Guinea, Liberia and Sierra Leone. 21 of the 146 edges cross a border.
- **Split.** Fit on target weeks 20 to 39, score on 40 to 51, origins set per horizon. I fixed the split
  from row counts alone, before any gain was computed.
- **Readability rule, fixed in advance.** A horizon counts only with at least 200 fit rows and 200
  score rows AND if a planted 0.5 SD neighbour signal is found there. h15 has 86 fit rows, so it was
  never going to be readable.
- **Nulls.** 200 global relabels, 200 within-country relabels, and the 20-draw shuffled-base
  false-alarm check.

**First pass: it looks like it works.**

| h | fit rows | score rows | gain % | null p95 % | within-country p95 % | planted % | planted found | false wins of 20 | readable |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 602 | 394 | +3.480 | +0.957 | +0.908 | +23.08 | yes | 1 | yes |
| 5 | 517 | 394 | +5.843 | +1.097 | +1.616 | +16.19 | yes | 2 | yes |
| 10 | 293 | 394 | +5.561 | +2.399 | +2.829 | +8.02 | yes | 1 | yes |
| 15 | 86 | 394 | -8.849 | +3.585 | +2.555 | -30.69 | no | 3 | no |

Real neighbours beat both nulls at h3, h5 and h10, the planted signal is found there, and false alarms
sit at chance.

**The catch: district size.** The dev panels use a per-district z-score, which centres every
district's train-period level at zero (see t6c above). Ebola uses one pooled scale for the whole
disease, so a district that is always big stays big inside its deviation. Big outbreaks sat next to big
outbreaks, and the neighbour average was mostly saying "this area is hot". I added each district's own
mean fit-period deviation as one more column to A and B alike, and reran the nulls and the planted check.

| h | gain with level % | null p95 % | within-country p95 % | planted % | planted found |
|---|---|---|---|---|---|
| 3 | +0.543 | +1.543 | +1.126 | +18.55 | yes |
| 5 | +0.396 | +1.458 | +1.415 | +3.66 | yes |
| 10 | -0.498 | +1.904 | +1.945 | +1.07 | no |
| 15 | -4.438 | +1.829 | +2.588 | -10.66 | no |

- **h3 and h5: a real null on timing.** The level-controlled probe still finds a planted timing signal
  there, and the real data shows none.
- **h10: no power.** After the level control the probe cannot find even the planted signal, so h10
  cannot be judged either way.
- **h15: unreadable**, by the rule fixed in advance.

So on Ebola the neighbour average told the model which districts were big, which the district's own
level already says. It added nothing about week-to-week timing at h3 and h5.

**Ebola limits, on top of the general ones below.**

- Small: 12 scored weeks, 394 scored rows per horizon, and 45 to 60 reporting districts a week.
- The level column is computed from fit-period cells, the same cells the fit targets come from. It is
  in-sample for fitting but never touches the scored weeks, and A and B get it identically.
- A null at h3 and h5 is a null for a LINEAR effect of about the planted size (25 percent of deviation
  variance). A smaller or nonlinear timing effect could be there and be missed.

## Limits

- **Linear, pooled model.** A nonlinear neighbour effect, or one that differs by district, could be
  missed. The positive control certifies linear detection at this effect size only.
- **Weekly resolution.** Anything faster than a week is folded into the shared wave or into the
  district's own lags.
- **The map is geographic, not mobility.** Two places linked by travel but not by a border are
  invisible to this probe.
- **Informative neighbours, not proven spread.** A WIN says neighbours carry forecast information. It
  cannot tell spread apart from a shared regional driver (weather, reporting practice) or from
  neighbours acting as a less noisy reading of the district's own state. A pooled lead-lag test cannot
  separate them either: on a two-way map every pair is counted in both directions, so it is symmetric
  by construction. A per-edge version would partly separate them, by asking for each pair whether one
  side consistently leads. A shared driver that moves across the map, such as a weather front, would
  still fool it.
- **Deviation definition.** The shared wave is the plain mean of observed districts, per country on
  dengue. A weighted mean, or a smoother regional baseline, would give different deviations.
- **Weak reference at long horizons.** On Japan and COVID at h10 and h15, the own-only model's test
  error is larger than the variance of the test deviations themselves, so it does worse than simply
  predicting their test-period average. The percent gains there are against a weak baseline.
- **Model space, not counts.** Every gain is in log1p z-score space. A probe gain does not promise the
  same gain in the model's count-space RMSE.
- **Not run.** A raw-count variant. On raw counts a pooled squared error is dominated by a few large
  districts, so it would not be a clean check.

## Reproduce

```
conda run -n ebola-train python diagnostics/graph_probe/t6_neighbour_signal.py
conda run -n ebola-train python diagnostics/graph_probe/t6_neighbour_signal.py --controls-only
conda run -n ebola-train python diagnostics/graph_probe/t6_neighbour_signal.py --controls-only --break-wrongmap
conda run -n ebola-train python diagnostics/graph_probe/t6b_wrongmap_floor.py
conda run -n ebola-train python diagnostics/graph_probe/t6b_wrongmap_floor.py --panels dengue --draws 6 --perms 20
conda run -n ebola-train python diagnostics/graph_probe/t6c_level_check.py
conda run -n ebola-train python -m experiments.t6_ebola_neighbour_signal
conda run -n ebola-train python diagnostics/verify_neighbour_signal_doc.py
conda run -n ebola-train python diagnostics/verify_neighbour_signal_doc.py --mutate
```

The first is about 9 minutes (dengue dominates). The `--break-wrongmap` run must exit 1. The dengue
floor run took 311 s and merges its result into `results/misc/t6b_wrongmap_floor.json`. The level check
takes about 20 s and the Ebola probe about 10 s.

Artifacts: `results/misc/t6_neighbour_signal.json` (main run, 553.4 s, produced before the wrong-map
fix; the fix does not touch the probe path, and t6b asserts the recomputed base gains are identical),
`results/misc/t6_neighbour_signal__controls.json` (the fixed controls),
`results/misc/t6_neighbour_signal__breaktest.json` (the deliberate break, written before the assert fires),
`results/misc/t6b_wrongmap_floor.json`, `results/misc/t6c_level_check.json`, and the EXPLORATORY
`experiments/t6_ebola_neighbour_signal.json`.

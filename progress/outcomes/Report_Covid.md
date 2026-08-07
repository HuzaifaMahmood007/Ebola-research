# Report_Covid.md — why the COVID single-disease encoder has no skill, and what does not fix it

**Date:** 2026-08-04
**Scope:** the `covid_us-states` panel only. The influenza, dengue and ebola panels are untouched by
everything below.
**Status:** diagnosis complete, two hypotheses tested and one rejected. No bundle was rebuilt, no
canonical record was overwritten, and the LDO3 run in flight was not disturbed.

---

## 0. Headline

The COVID single-disease encoder loses to a per-node constant by up to 131 per cent. This is **not a
training bug**. The panel breaks the stationarity assumption the whole protocol rests on, and it does
so in five separate places at once.

Two candidate fixes were tested end to end:

| test | hypothesis | outcome |
|---|---|---|
| `covid_val_probe.py` | model selection on the Omicron val fold is the cause | **partly confirmed** — 13–49% recovered |
| `covid_split_probe.py` | moving Omicron into train fixes it | **REJECTED** — worse at 3 of 4 horizons, seed noise tripled |

The conclusion is that **Omicron has no good home**. It is a ~6× structural break in a 164-week
series, and every chronological placement of it hurts a different mechanism. No split repairs this.

---

## 1. The problem, measured

`covid_us-states`: `X = (49, 164, 4)`, weekly, 2020-02-01 → 2023-03-18, split `fixed_50_20_30`
(`train_end=82`, `val_end=115`). 5-seed test RMSE (`node_mean`, count space), against its own naive
floors:

| h | encoder | persistence | train_mean | seasonal | verdict |
|--:|--:|--:|--:|--:|---|
| 3 | 5,566 | **4,260** | 5,525 | 10,702 | loses to persistence by 31% |
| 5 | 8,798 | 5,738 | **5,456** | 18,508 | **loses to train_mean by 61%** |
| 10 | 12,264 | 10,068 | **5,302** | 29,875 | **loses to train_mean by 131%** |
| 15 | 11,343 | 27,473 | **5,275** | 28,453 | **loses to train_mean by 115%** |

`influenza_us-states` — the same 49 nodes, the same graph, the same covariates — beats every floor at
every horizon. The failure is specific to COVID, not to the encoder.

All four figures above were reproduced independently from the archived quantiles
(`results/single/encoder__covid_us-states__seed*__quantiles.npz`) before any diagnosis was attempted,
and match `score.py` to the digit. The decomposition below therefore rests on verified arithmetic.

---

## 2. Root cause, in order of size

### 2.1 At h≥10 there is nothing to predict

Per-node lag-*h* autocorrelation of the target in z-space:

| lag | test fold | train fold |
|--:|--:|--:|
| 3 | **+0.528** | +0.867 |
| 5 | +0.319 | +0.704 |
| 10 | **+0.015** | +0.350 |
| 15 | **−0.048** | +0.282 |

The test fold carries **zero linear predictability at h10 and h15**. Train carries plenty — those are
the Delta and winter-2021 waves. The model learned real epidemic dynamics and the evaluation period
contains none of them.

**No model can beat a constant there.** This is a property of the data, provable without running
anything, and it is the single largest term in the deficit.

### 2.2 The encoder has genuinely negative skill — this is not a metric artefact

Count-space RMSE flatters constants, so both spaces were checked. Per-node then mean:

| h | encoder | persist | train_mean | | encoder z | persist z | train_mean z |
|--:|--:|--:|--:|---|--:|--:|--:|
| 3 | 5,566 | 4,260 | 5,525 | | 0.325 | 0.318 | 0.345 |
| 5 | 8,798 | 5,738 | 5,456 | | 0.491 | 0.412 | **0.366** |
| 10 | 12,264 | 10,068 | 5,302 | | 0.593 | 0.530 | **0.366** |
| 15 | 11,343 | 27,473 | 5,275 | | 0.573 | 0.623 | **0.378** |

The encoder loses to a per-node constant **in its own pinball loss space** from h5 onward. The mean
test per-node z std is 0.333, which is what a perfect constant predictor would score; the encoder is
at 0.593 by h10. The deficit cannot be argued away as an artefact of count-space scoring.

### 2.3 The val fold is the Omicron peak

National argmax is week 102 = **2022-01-15**, inside val (82–115). Fold levels:

| fold | weeks | national mean | national max |
|---|--:|--:|--:|
| train | 82 | 419,883 | 1,661,031 |
| **val** | 33 | **1,189,000** | **5,139,436** |
| test | 49 | 431,608 | 811,443 |

Val runs at **2.83×** both other folds and its peak week is **6.3×** the largest test week. Val also
requires **extrapolation**: its maximum is 3.1× anything in train.

Two mechanisms key off this fold, and both are damaged:

- **Early stopping** selects the checkpoint on it. The curve shows selection is noise-dominated
  anyway — epochs 3–32 all sit in 0.1220–0.1273.
- **The §3.1 bias correction** is *fitted* on it. `bias_c` comes out positive and is then applied to
  a test fold at a sixth of the scale. Measured cost of `encoder_mc` against `encoder`:
  **+8.8 / +21.4 / +47.3 / +27.4 %** at h3/h5/h10/h15. The correction makes COVID worse everywhere.

### 2.4 Train and test have different variance, so the model over-predicts

| | train | val | test |
|---|--:|--:|--:|
| z mean | +0.000 | +0.713 | +0.376 |
| z std | 1.000 | 0.537 | 0.427 |
| per-node z std (median) | 1.000 | 0.424 | **0.259** |

Test has roughly a **quarter** of the within-node variance of train. The model was taught to swing;
the test period asks it not to. Prediction/truth ratio confirms it: 0.98 / 0.88 / **1.17** / **1.35**
at h3/h5/h10/h15.

### 2.5 Smallest training budget in the project, least independent nodes

| bundle | train origins | train weeks |
|---|--:|--:|
| **covid_us-states** | **60** | 82 |
| influenza_japan | 152 | 174 |
| influenza_us-states | 158 | 180 |
| influenza_us-regions | 370 | 392 |

Against 142,305 trunk parameters. Mean pairwise node correlation in train z-space is **+0.912**
(influenza_us-states: +0.717), so the 49 states are close to one national trajectory of 60 points.

### 2.6 Count-space amplification, which magnifies all of the above

Median per-node scaler std is **2.405** on COVID against **1.130** on influenza_us-states. One z-unit
of error is therefore an **11.1×** multiplicative count error on COVID and **3.1×** on influenza. The
same z-space skill is roughly 3.5× more punishing here. This does not create the deficit (§2.2) but it
converts a 1.62× z-space gap at h10 into a 2.31× count-space gap.

---

## 3. Test 1 — is model selection the cause? Partly, and it is worth 13–49%

`covid_val_probe.py`. Three arms, one variable each, 3 seeds, test fold identical in every arm.
Arm A reproduces the canonical on-disk records seed-for-seed, which is the harness self-check.

| h | A baseline | B val←end of train | C val−omicron | train_mean | C vs A |
|--:|--:|--:|--:|--:|--:|
| 3 | 6,064 | 7,282 | **4,972** | 5,525 | **−18.0%** |
| 5 | 8,872 | 8,380 | **7,363** | 5,456 | **−17.0%** |
| 10 | 11,287 | 10,751 | **9,697** | 5,302 | **−14.1%** |
| 15 | 11,573 | 9,576 | **10,061** | 5,275 | **−13.1%** |

`encoder_mc`, which is fitted on the same fold, moves further: **−17.7 / −32.8 / −49.2 / −46.8 %**.
Under the shipped split the bias correction made things 47% worse; with the selection fold cleaned it
becomes the *better* arm. One change repaired two mechanisms, because both read the same fold.

**Arm B — carving val from the end of train — lost.** It helps at h5–h15 but costs 20 of 60 training
origins and h3 degrades 20%. Shrinking the smallest training budget in the project is not worth it.

**Why arm C cannot ship.** The excision rule was "drop val weeks exceeding the *test* fold's max".
That peeks at test to design the protocol. It is valid for establishing the mechanism, which it did,
and invalid as a shipped protocol. A train-only variant ("drop val weeks exceeding the *train* max")
would be legitimate but excises a narrower window and would recover less.

---

## 4. Test 2 — does a better split fix it? NO

`covid_split_probe.py`. Proposed `train_end=107, val_end=126` (65/12/23), which moves Omicron into
train and **excludes no data at all**:

```
train [0,107)    2020-02-01 .. 2022-02-12   contains the Omicron peak
val   [107,126)  2022-02-19 .. 2022-06-25   level-matched to test (1.07x)
test  [126,164)  2022-07-02 .. 2023-03-18   38 weeks
```

On paper this fixes everything §2.3 identified: no extrapolation anywhere (train max 5.14M ≥ val 710k
and test 811k), val/test level ratio 2.75× → **1.07×**, train origins 60 → **85 (+42%)**, and the h3/h5
test autocorrelation is unchanged. `Bundle.refit()` was used so the scaler and the trunk *inputs* move
with the new train window rather than leaking the old one.

It does not work. The comparable quantity across two different test folds is **model ÷ best naive
floor on that split's own test cells**; absolute RMSE is not comparable and is never differenced here.

| h | current 82/115 | proposed 107/126 | |
|--:|--:|--:|---|
| 3 | 1.31 | **1.95** | worse |
| 5 | 1.61 | **3.01** | worse |
| 10 | 2.31 | **2.47** | worse |
| 15 | 2.15 | 1.98 | ~flat |

Nothing crosses 1.00, so a naive baseline still wins at every horizon. Against `train_mean`
specifically the picture is better at long horizons (h15 2.74× → **1.38×**, h10 3.41× → 2.72× on
`encoder_mc`) but that is a smaller loss, not a win — and persistence gets *stronger* on the new test
fold (3,659 at h3 against 4,260), so the bar rose more than the model did.

**It also destabilised training.** Seed standard deviation as a percentage of the mean:

| h | current | proposed |
|--:|--:|--:|
| 3 | 16.4% | 15.7% |
| 5 | 9.6% | **29.1%** |
| 10 | 15.2% | **37.7%** |
| 15 | 12.9% | **31.8%** |

Two to three times noisier at 5 seeds. That alone disqualifies it for a reported number.

---

## 5. Why: Omicron has no good home

This is the transferable finding, and it generalises past this one split.

| placement | what breaks |
|---|---|
| **val** (shipped) | selection and `bias_c` target a regime absent from test — §2.3, costs 13–49% |
| **train** (tested) | the model learns 5M-scale excursions and emits them on a flat test fold — variance triples |
| **test** | extrapolating a 3.8× unprecedented event from a train fold containing nothing like it |

Omicron is a ~6× structural break sitting at 62% through a 164-week series. It is not a fold-placement
problem. The earlier framing — "put it in train and the protocol's intent is restored" — considered
only the selection mechanism and ignored what training on a 6× outlier does to the learned dynamics.
That framing was wrong and Test 2 is what falsified it.

A secondary consequence worth recording: **COVID's series is structurally front-loaded.** All the
dynamics live in the first ~65% and the tail is flat. Every chronological split therefore faces the
same fork — a test fold that is either flat (nothing to predict) or contains a wave (extrapolation).
Across every candidate split swept, test-fold lag-10 autocorrelation stayed at or below +0.01.

---

## 6. What this means for the programme

**The blocker as written is unachievable.** `Doubt.md` states *"no COVID transfer number should be
quoted until COVID beats its own naive floors."* At h10/h15 that bar cannot be cleared by any model,
because §2.1 shows there is no signal to clear it with. As written it blocks the COVID work
permanently and should be re-scoped by horizon.

**COVID survives as a transfer SOURCE, not a target.** The `covid → influenza_us-states` direction is
significantly negative at all four horizons and its validity rests on the *influenza* ceiling — which
beats every floor — not on the COVID model being good. That result is untouched by everything in this
document and remains the publishable finding from the pair fold. The `influenza → covid` direction is
the uninterpretable half.

**LDO3 is not invalidated.** Its dengue and influenza folds carry COVID in the *trunk*, and a trunk
trained partly on COVID transferring to a target with a sound ceiling is a legitimate experiment. Only
the covid-held-out fold is uninterpretable, which was already known and already documented.

### Recommended disposition

1. **Do not rebuild the bundle. Do not change the split.** Tested, rejected, evidence in §4.
2. **Report COVID as a transfer source**; scope the blocker to h3/h5 rather than all horizons.
3. **Report the single-disease failure as a disclosed limitation**, with §2.1 as the mechanism: a
   post-outbreak flat panel on which a per-node constant is near-optimal.
4. **Untested option with real upside — the `deaths` column.** The NYT source file already carries
   `deaths` alongside `cases`, so it needs no new data acquisition and keeps the identical graph, node
   set and covariates. Omicron was enormous in cases and far smaller in deaths, so the 6× break that
   drives this entire document may largely disappear. Roughly an hour to test. **Not yet measured —
   this is a hypothesis, not a result.**

---

## 7. Reproduction

| artefact | what it does |
|---|---|
| `covid_val_probe.py` | Test 1. Three val-fold arms, writes nothing to `results/`, self-checks arm A against disk. |
| `covid_split_probe.py` | Test 2. Proposed split with `Bundle.refit()`, writes nothing to `results/`. |
| `compare_runs.py --metric rmse` | pair-fold transfer table against ceilings and floors |
| `results_matrix.py` | full matrix; `DATASETS` now includes `covid_us-states` (it previously did not, which is why `Results_Matrix_CovidRun.md` contained no COVID data despite its name) |

Both probes call `train.loop.train_one` directly and never `run_dataset`, so no canonical
`encoder__covid_us-states__seed*.json` record was overwritten at any point.

**Caveats on strength.** Test 1 is 3 seeds and reports means without per-seed pairing. Test 2 is 5
seeds. The documented seed sd is 8–14% on this project, and Test 2's proposed arm ran at 29–38%, so
its negative verdict is robust while Test 1's magnitudes should be treated as directional.

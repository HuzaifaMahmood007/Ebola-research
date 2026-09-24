# Input vs representation: where the district structure goes

2026-09-22. Milestone 6 run 3, closing `Session_Audit_2026-09-10.md` section F item 1.

## The question

The graph probes found the encoder's output is only 3.8 to 22.4 percent district-specific in
energy, which is why the spatial graph buys nothing. That left two diagnoses that fit the same
measurements: either the raw data never carried district structure (the encoder is faithful), or
the data carries it and the encoder throws it away. Nothing run before today separated them.

## What I ran

`diagnostics/graph_probe/t5_input_energy.py`, inference only, seed 42 single-disease checkpoints
from `results/single/`, the same checkpoints t1 through t4 used. It computes the exact statistic
from t4 (district-specific share of energy: how much of a node-by-feature matrix survives after
removing the cross-district mean, as a share of the total) on three things per test origin: the
full 4-channel input window the encoder consumes, the incidence channel of that window alone, and
the encoder output h. It also computes effective dimensionality (participation ratio of the
singular values, roughly how many directions are in use) on each. As a consistency check the
script recomputes the published representation numbers and asserts it hits them; it reproduced
3.8 / 8.6 / 12.3 / 4.8 / 22.4 to the printed digit, and I broke the check on purpose to confirm
it fails loudly when the numbers do not match. Machine-readable summary:
`results/misc/t5_input_energy.json`. Reproduce with
`conda run -n ebola-train python diagnostics/graph_probe/t5_input_energy.py`, about 80 seconds,
nearly all of it dengue.

## The numbers

District-specific share of energy, mean over test origins:

| panel | input, incidence channel | input, all 4 channels | representation h | reduction (incidence to h) |
|---|---|---|---|---|
| influenza_japan | 13.8% | 3.2% | 3.8% | 3.6x |
| influenza_us-regions | 27.5% | 6.2% | 8.6% | 3.2x |
| influenza_us-states | 49.3% | 18.3% | 12.3% | 4.0x |
| covid_us-states | 23.8% | 5.0% | 4.8% | 5.0x |
| dengue | 93.5% | 40.8% | 22.4% | 4.2x |

Read the incidence column as the fair one. Of the four input channels, sin_doy and cos_doy are the
calendar and are identical across districts by construction, and obs_mask is constant 1 on the
fully observed panels, so the all-channel column is diluted by channels that cannot distinguish
districts by design. The incidence channel is the per-district signal the model is asked to
forecast.

Effective dimensionality of the district-specific part:

| panel | input incidence (of 20) | representation h (of 64) |
|---|---|---|
| influenza_japan | 3.9 | 2.3 |
| influenza_us-regions | 1.5 | 1.8 |
| influenza_us-states | 1.8 | 2.5 |
| covid_us-states | 2.8 | 1.9 |
| dengue | 4.0 | 4.6 |

## Verdict

Present in the data and mostly discarded by the encoder. The incidence input windows are 14 to 94
percent district-specific in energy, the encoder output keeps 3.8 to 22.4 percent, and the
reduction is 3x to 5x on every one of the five panels with no exception. The spatial layer gets
little district structure to mix not because the data lacks it but because the encoder has already
compressed most of it into the shared component before the graph is consulted. Of the two
diagnoses in the session audit, this rules out "the data never had it" and is consistent with
"training drove the encoder into a narrow solution".

One nuance the effective-dimensionality table adds: the narrowness in directions is not the
encoder's doing. The input's district-specific content already lives in about 2 to 4 directions,
and h uses about the same count. What the encoder changes is the share of energy, not the
dimensionality.

The one thing this measurement cannot decide, and the sentence must carry it: energy is not
usefulness. District-specific input energy can be noise, and per-node z-scoring amplifies quiet
nodes, so some of what the encoder removes may deserve removing. This test shows the encoder
discards district structure that exists; it does not show the discarded part was forecastable
signal. Dengue's 93.5 percent is the loudest example, since 78 percent of its input cells are
exact zeros in z-space and its districts barely share a common trajectory.

The manuscript sentence I would write: "The raw incidence windows are 14 to 94 percent
district-specific in energy across the five development panels, while the encoder's output retains
only 3.8 to 22.4 percent, a three to five fold reduction on every panel; the spatial layer
receives little district-level structure not because the data lacks it but because the encoder has
already compressed it away, although this energy accounting cannot establish that the discarded
variation was forecastable signal rather than node-level noise."

## Scope limits

- Single-disease checkpoints from `results/single/`, seed 42 only, not the transfer trunk. The two
  input columns are properties of the data and do not depend on the seed at all; only the
  representation column and the reduction factor could move with seed, and dengue in particular is
  one seed.
- Dengue uses every fourth test origin (158 of the origins), the same subsample `t_dengue.py`
  used, because of its 7,165 nodes.
- The statistic measures shares of energy, not forecasting usefulness. See the nuance above.
- The comparison covers the window input only. The LTR degree feature also feeds h
  (`models/encoder.py:60`) and is district-specific by construction; its contribution sits inside
  the h column, not the input columns.
- Input and representation have different raw dimensions (20 or 80 versus 64). The statistic is a
  scale-free share, so the comparison is of shares, which is the intended reading.

All numbers above live in `results/misc/t5_input_energy.json`, written by the script in the same
run that printed them.

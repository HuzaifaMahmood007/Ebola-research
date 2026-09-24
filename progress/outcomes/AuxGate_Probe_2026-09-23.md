# Aux-district-identity gate probe (Milestone 6 run 5, stage 1)

**Probe run 2026-09-23. Scoring completed 2026-09-24.** This is the cheap gate in front of the
auxiliary district-identity objective. It decides whether an overnight retrain is worth running, on
numbers, not judgement. Verdict up front: **NO-GO.** The runner is not written.

## The question

The aux objective would add a second encoder head that classifies "which district is this?" from the
representation `h`, trained alongside the forecast loss, to stop the encoder squashing district
structure. `Input_Energy_2026-09-22.md` measured that squash: raw incidence windows are 14 to 94
percent district-specific in energy, `h` keeps only 3.8 to 22.4 percent, a three to five fold
reduction on every panel. The objective is only worth an overnight retrain if the discarded structure
is forecast-USEFUL. The user's own stated risk: if the squashed component was noise, forcing it back
in makes forecasts worse. And D2 (`Shuffled_Adjacency_2026-09-23.md`) already bounds the upside: even
the real adjacency, learned from scratch, earns only 2 to 4 percent and only on dengue.

The probe asks the cheaper version of the question: **is the district information the encoder discards
forecast-relevant at all?**

## The probe and its leakage rule

On each frozen `results/single/` checkpoint (5 panels x 5 seeds), I fit per-node corrections on the
model's own forecast residuals and check whether giving the model district identity post-hoc lowers
TEST error. Five variants, all in the head's model space:

- `baseline`: the median forecast, no correction.
- `bias_global`: median + one scalar per horizon, pooled over all nodes. Uses NO district identity.
  This is the same shape as the global median-vs-mean fix `loop._fit_bias_correction` already ships.
- `bias_node`: median + one scalar per NODE per horizon. Adds district identity as a level offset.
- `affine_global`: one slope+intercept per horizon, pooled. No district identity.
- `affine_node`: per-node slope+intercept. Adds district identity as a level-and-scale map. This is
  the closest cheap analogue to what the aux objective would actually enable.

**Leakage rule.** Every correction is fit ONLY on train-phase origins and train-phase observed target
cells. Test origins are never touched during fitting. The checkpoint is frozen: no retrain, no
gradient. The model was trained on the train origins, so a per-node correction fit on the train period
that transfers to the held-out test period is the honest generalisation test: stable district
structure transfers, overfit noise does not. A node needs at least 8 observed train target cells to
earn a correction, else it keeps the baseline (identity).

Metrics are country-macro RMSE and MAE (`score.score_bundle`, the headline aggregation). dengue uses
`origins[::4]` (7,165 nodes, the `t_dengue` precedent). Verdict rule is the repo's within-noise rule,
paired by seed: a contrast counts only when the mean paired delta over the 5 seeds exceeds its own
spread across those seeds. Script: `diagnostics/graph_probe/aux_gate_probe.py`, output
`results/misc/aux_gate_probe.json`.

## Why node-vs-global is the decisive contrast, and why node-vs-baseline guards it

A global correction uses no district identity. The aux objective only has a target if district
identity adds something a district-AGNOSTIC correction cannot. So the district-identity signal is
`bias_node` beating `bias_global` (contrast B). But contrast B alone is a trap: a per-node correction
can "beat" a global one simply by overfitting less badly, while BOTH are worse than doing nothing. So
the honest test is the pair: a node variant is genuine district-identity evidence only when it beats
BOTH the baseline (contrast A) AND its matching global variant (contrast B) beyond seed noise.

## The tallies (recompute in `diagnostics/verify_auxgate_doc.py`)

**Bias tally over 40 cells: node beats baseline 4, node beats global 18, node beats BOTH 2.**

The 18-of-40 node-beats-global count looks encouraging and is misleading. On dengue and
influenza-us-regions the global correction HURTS relative to baseline (dengue global-vs-baseline is
-4 to -12 percent), and the per-node correction also hurts, just less. So the positive node-vs-global
signal there is an artifact of both corrections being worse than doing nothing. Baseline, with no
district correction at all, is the best of the three on those panels.

The 2 cells where the per-node correction beats BOTH baseline and global, the only genuine
district-identity evidence in the bias instrument, are both covid_us-states at h10:

| panel | metric | h | node vs baseline | node vs global |
|---|---|---|---|---|
| covid_us-states | rmse | 10 | +9.95% | +3.30% |
| covid_us-states | mae | 10 | +9.48% | +3.72% |

**Affine tally over 36 non-failed cells: node beats baseline 0, node beats global 11, node beats
BOTH 0. Affine failed cells 4.** The affine per-node instrument, the richer analogue of the aux
objective, NEVER beats baseline beyond noise and hurts on 8 cells. It also blows up numerically on 4
seed-cells: dengue h3 (rmse and mae, seed 42) and dengue h5 (rmse and mae, seed 62), with errors up to
3.07e7 against a baseline near 50. Cause: a per-node affine slope fit on dengue's near-zero-variance
quiet z-scored nodes extrapolates catastrophically once inverted through `expm1`. Those cells are
marked FAILED and excluded from the affine tallies. This failure is itself evidence, not just a
nuisance: it is the section-6 concern made concrete, per-node scaling amplifies quiet-node noise, and
it is a live instance of the user's stated risk that forcing squashed structure back in makes things
worse.

## Gate decision: NO-GO

The decisive contrast, a node variant beating BOTH baseline and a district-agnostic correction beyond
seed noise, fires in 2 of 40 bias cells (both covid h10) and 0 of 36 affine cells. That is not a
target worth an approximately 14 hour retrain. Reasons:

1. Where the per-node correction appears to win (dengue, us-regions), it wins only against a global
   correction that is itself worse than baseline. Doing nothing beats both. That is not district
   identity helping the forecast.
2. The one genuine district gain, covid_us-states h10, is a per-district LEVEL offset. It is a free
   post-hoc recalibration; it needs no retrain and no richer `h`. The aux objective is not the tool
   for it.
3. The richer per-district instrument, affine_node, which is the true analogue of what an aux head
   would produce, never beats baseline, hurts on 8 cells, and blows up on dengue. Pushing more
   district structure into the model looks like it costs accuracy, exactly the user's stated risk.

Under the original rule, and with borderline defaulting to NO-GO, this is a clear NO-GO. The user may
overrule.

## A separate observation, not a decision: post-hoc recalibration on COVID

COVID is the one panel where a post-hoc correction genuinely helps. The global level offset beats
baseline by roughly 5 to 8 percent at short horizons and clears the within-noise rule at h10
(global-vs-baseline +6.88% RMSE, +5.98% MAE). The short-horizon gains are positive but not seed-stable
under the strict rule. The per-node offset adds a little more at h10 (the covid cells above). This is a
recalibration story, and COVID is the most-shifted development panel and the closest analogue to Ebola.
It suggests a cheap follow-up, a free post-hoc per-node or global recalibration on COVID, entirely
separate from the aux objective. Limits: one panel, model-space fits scored in count space, and the
global half of it is already available through the shipped `_fit_bias_correction`, so only the per-node
half would be new. This is flagged for consideration, not decided here.

## Scope limits

- `results/single/` checkpoints only, not the transfer trunk.
- Corrections are fit in model space by least squares; the metric is count space (`expm1`). A
  model-space fit is not count-optimal, but node and global variants are fit identically, so the
  contrast is fair. This measures whether district identity is EXTRACTABLE post-hoc, a rough upper
  bound on what an aux head fed the same residuals could add; it is not the aux objective itself.
- dengue uses `origins[::4]`; its per-node fits are the least stable, which is where the affine
  blowups land.
- Tallies use point means over 5 seeds with the within-noise rule, not bootstrap intervals, and no
  multiplicity correction across cells.

## Reproduce

```
conda run -n ebola-train python diagnostics/graph_probe/aux_gate_probe.py
python diagnostics/verify_auxgate_doc.py
python diagnostics/verify_auxgate_doc.py --mutate
```

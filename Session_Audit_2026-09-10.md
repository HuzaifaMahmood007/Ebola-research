# Session Audit, 2026-09-10

What this is: a record of verification work done in a walkthrough of the model, the data and the
results. Findings only. No recommendations, no next actions.

Every number below was recomputed from disk for this document. I did not copy any figure out of an
existing progress document. Where a number I was handed did not reproduce, I wrote what the disk
says and flagged it in the "did not reproduce" list at the end.

The probe scripts are now in `diagnostics/graph_probe/` so the commands here can actually be run.
They were written into a session temp folder first, which does not survive. Delete the folder if you
do not want it. Environment for all of them is `ebola-train`.

---

## Summary

| # | Finding | Status | Evidence |
|---|---|---|---|
| A1 | SharedEncoder is 142,305 params, 87% of it the temporal stack | CONFIRMED | `diagnostics/graph_probe/pcount.py` |
| A2 | No parameter is indexed by node count, TCN receptive field 32 covers the 20-week window | CONFIRMED | `models/encoder.py:41-44`, `models/config.py:6` |
| A3 | The FiLM-folds-into-the-head claim is true but has no test anywhere | NEW gap | `models/adapters.py:14-16`, `tests/` |
| B1 | Dengue is 97.7% of every observed cell in the project | CONFIRMED | `diagnostics/graph_probe/sizes2.py` |
| B2 | COVID fits a 143,733-param model from 60 training origins | CONFIRMED | `diagnostics/graph_probe/orig.py` |
| C1 | The spatial branch is not a no-op, the residual does not kill it | NEW, hypothesis REFUTED | `diagnostics/graph_probe/t1.py` |
| C2 | 78% to 96% of every district vector is one shared vector | **NEW** | `diagnostics/graph_probe/t1b.py` |
| C3 | Scrambling which district is which costs 0.04% to 1.04% | **NEW** | `diagnostics/graph_probe/t2.py`, `t_dengue.py` |
| C4 | Neighbour averaging destroys 14% to 60% of the personal part, which is the mechanism | **NEW** | `diagnostics/graph_probe/t3.py`, `t_dengue.py` |
| C5 | The gate is wide open, it is not switched off | CONFIRMED | `Reports/gated+spatial_Contribution.txt:5-9` |
| C6 | Gate ablation splits 0 help / 8 hurt on error, 6 help / 1 hurt on correlation | CONFIRMED | `results/reports/gate_ablation.log:134` |
| D1 | 25 of 25 single-disease runs early-stopped | CONFIRMED | `diagnostics/epoch_budget_audit.py` |
| D2 | All 15 LDO3 trunks early-stopped on 14.3% to 29.7% of budget | CONFIRMED | same |
| D3 | Full-budget reruns are identical, and step 91000 is worse than best | CONFIRMED | `diagnostics/ldo3full_check.py`, `results/reports/ldo3full_*.log` |
| D4 | The 91,000-step budget was sized for data this project does not have | NEW reasoning | `train/lodo.py:115-133` |
| D5 | The Ebola epoch CV hits its 80-epoch cap in 6 of 10 runs | CONFIRMED | `results/reports/ebola_run.log` |
| E1 | Zero-shot beats few-shot in 31 of 32 point-mean cells | **NEW** framing | `diagnostics/graph_probe/eb3.py` |
| E2 | The arm with MORE support data is the arm adaptation damages MORE | **NEW** | same |
| F | Four things I could not settle | open | section F |

C and E are the two sections that are not reflected in any other document in this repo. I grepped
`Reports/*.md`, `progress/`, `PROJECT.md`, `README.md` and `Resume.md` for the permutation test and
for the collapse measurements and found nothing. C2, C3 and C4 are new evidence. E1 is a new
statistic, not a new claim, and section E says exactly how it differs from the claim already on
record.

---

## A. Architecture, checked by building the model

Command:

```
conda run -n ebola-train python diagnostics/graph_probe/pcount.py
```

Output, exact:

```
encoder total 142305
   tcn 124480     ltr 128     spatial 16640     gate 1057
adapter total 1428
   gamma (64,) 64     beta (64,) 64     head.weight (20, 64) 1280     head.bias (20,) 20
node-indexed params: []
TCN receptive field: 32
```

Read it as: 87.5% of the trunk is the dilated temporal stack. The spatial mixer is 11.7%. The gate
that decides how much spatial signal to let through is 0.7%. The whole per-disease adaptation
surface is 1,428 numbers.

`node-indexed params: []` is the C2 audit passing. No weight has a dimension equal to 10, 47, 49, 61
or 7165, the five graph sizes, so nothing in the encoder is tied to one country's district list.
That is what makes the encoder disease-agnostic by construction rather than by claim. The check
lives at `models/encoder.py:41-44` and the forbidden sizes at `models/config.py:12`.

Receptive field 32 against a 20-week lookback means the temporal stack can see the whole window. The
assertion that enforces it is `models/temporal.py:17`.

### A3. A documentation-versus-test gap

`models/adapters.py:14-16` says:

> With the trunk frozen this is EXACTLY an affine map of the frozen features (gamma folds into the
> head weight, verified to 0.0 deviation), so FiLM adds no expressive power over a plain linear
> head and the inner loop is ANIL in the exact sense.

I searched `tests/` for anything covering this and found nothing. A case-insensitive grep for
`film`, `fold` and `gamma` across `tests/` returns one unrelated hit about spelling maps at
`tests/test_schema.py:702`. So the sentence rests on a comment somebody wrote once, not on anything
that runs.

I checked it myself:

```
conda run -n ebola-train python diagnostics/graph_probe/fold.py
-> max abs deviation: 7.152557373046875e-07
```

The claim is true. 7.2e-07 on values of order 1 is float32 rounding, not a real difference. But
"verified to 0.0 deviation" is not literally right either, and nothing would catch it if somebody
changed `Adapter.forward`. The downside of leaving it as is: this comment is load-bearing for the
ANIL argument in the paper, and it is currently unprotected.

---

## B. How much data there actually is

Two different counts matter and they give very different impressions.

**Observed node-week cells** (`conda run -n ebola-train python diagnostics/graph_probe/sizes2.py`):

| bundle | shape | observed cells |
|---|---|---|
| dengue | (7165, 1409, 4) | 2,196,261 |
| influenza_us-states | (49, 360, 4) | 17,640 |
| influenza_japan | (47, 348, 4) | 16,356 |
| covid_us-states | (49, 164, 4) | 8,036 |
| influenza_us-regions | (10, 785, 4) | 7,850 |
| ebola | (61, 52, 5) | 1,299 |
| **total** | | **2,247,442** |

Dengue is 2,196,261 of 2,247,442, which is 97.7% of every cell in the project.

**Training origins** is the honest example count, because the trunk takes one optimizer step per
origin per dataset and an origin is the whole graph at one week
(`conda run -n ebola-train python diagnostics/graph_probe/orig.py`):

| bundle | nodes | weeks | train | val | test |
|---|---|---|---|---|---|
| dengue | 7165 | 1409 | 1138 | 594 | 630 |
| influenza_us-regions | 10 | 785 | 370 | 170 | 235 |
| influenza_us-states | 49 | 360 | 158 | 84 | 108 |
| influenza_japan | 47 | 348 | 152 | 82 | 104 |
| covid_us-states | 49 | 164 | 60 | 45 | 49 |

The point that falls out: COVID fits a 143,733-parameter model (142,305 trunk plus 1,428 adapter)
from 60 training examples. That is not a small model on a small dataset, it is a model with roughly
2,400 parameters for every example it sees.

Ebola has no train/val/test origins because it is not split that way. `orig.py` prints -1 for it.

---

## C. The graph does not help, and here is why. NEW.

This is the section that changes what the paper can say. Everything in it is inference only. I
loaded the `results/single/` checkpoints, ran them forward, and measured. Nothing was retrained.

### Scope limits, stated first

- All of this uses **single-disease** checkpoints from `results/single/`, not the transfer trunk.
- **Dengue is one seed only** (seed 42) with five permutations, because a single pass over dengue's
  630 test origins takes 306 seconds.
- This is **not** the retrain-based shuffled-adjacency control listed as optional in `CLAUDE.md`
  section 3. That one asks whether a model trained on a fake graph would do as well. Mine asks a
  different and much cheaper question: does the model we already trained use the real graph. Both
  are worth knowing and neither substitutes for the other.
- Effective dimensionality was measured on the four small panels only. `t_dengue.py` does not
  compute it.

### Test 1: the "it is a no-op" hypothesis. REFUTED.

The obvious first guess is that the residual connection at `models/spatial.py:73` lets the network
learn to ignore the spatial term, so the branch passes its input straight through:

```
h = F.gelu(sw(h) + nw(torch.sparse.mm(A_hat, h))) + h
```

It does not. From `conda run -n ebola-train python diagnostics/graph_probe/t1.py`, seed 42, test
origins:

| panel | shift from neighbour mixing | shift of the final gated vector | median gate |
|---|---|---|---|
| influenza_japan | 1.4541 | 0.3707 | 0.261 |
| influenza_us-regions | 1.1939 | 0.3911 | 0.330 |
| influenza_us-states | 1.0952 | 0.4023 | 0.372 |
| covid_us-states | 1.0003 | 0.3124 | 0.310 |

Both columns are the size of the change divided by the size of the original vector. The spatial
branch moves the representation by more than the length of the vector itself, and the final gated
output still sits 31% to 40% away from the graph-free version. The branch is doing real arithmetic.
Whatever is wrong is not that the wire is cut.

### Test 1b: 78% to 96% of every district vector is the same vector. NEW.

From `conda run -n ebola-train python diagnostics/graph_probe/t1b.py`:

| panel | raw pairwise cosine | cosine after removing the shared mean | district-specific share of energy | effective dimensions |
|---|---|---|---|---|
| influenza_japan | 0.981 | -0.018 | 3.8% | 2.30 |
| covid_us-states | 0.979 | 0.006 | 4.8% | 1.95 |
| influenza_us-regions | 0.938 | -0.165 | 8.6% | 1.76 |
| influenza_us-states | 0.925 | 0.011 | 12.3% | 2.50 |
| dengue (seed 42, from `t_dengue.py`) | not measured | not measured | 22.4% | not measured |

Raw cosines above 0.92 look like total collapse, every district identical. They are not. Once you
subtract the across-district mean, the centred cosines sit at roughly zero, which means the
districts genuinely do point in different directions from each other. They are not clones.

The real result is the third column. "District-specific share of energy" is the fraction of the
squared length of the 64-number vector that differs between districts, with the rest being one
common offset that every district carries. It is 3.8% on Japan and 22.4% on dengue. So between 78%
and 96% of what the encoder emits for a district is a shared vector that says nothing about that
district.

Effective dimensions is the participation ratio of the singular values, which is a way of asking how
many of the 64 directions are actually in use. It is 1.76 to 2.50. The part of the representation
that is personal to a district lives in about two dimensions out of sixty-four.

### Test 2: relabel the graph and nothing much happens. NEW.

I permuted the node labels of the adjacency matrix at inference. Same topology, same degree
sequence, same number of edges, every district simply handed somebody else's neighbours. 20
permutations per seed, 5 seeds, on the four small panels. 5 permutations on one seed for dengue.
Model-space RMSE over observed test cells, pooled over horizons.

```
conda run -n ebola-train python diagnostics/graph_probe/t2.py
conda run -n ebola-train python diagnostics/graph_probe/t_dengue.py
```

| panel | change in RMSE | sd across seeds |
|---|---|---|
| influenza_japan | +0.04% | 0.08 |
| covid_us-states | +0.26% | 0.42 |
| influenza_us-states | +0.74% | 0.35 |
| influenza_us-regions | +1.04% | 0.70 |
| dengue, 1 seed, 5 perms | +0.59% | perms ran +0.59 / +0.60 / +0.58 / +0.61 / +0.60 |

Both sides of this, stated straight.

**What it shows.** The real graph is better than a fake one with the same shape. All five panel
means are positive, and the dengue permutations are extremely tight around +0.59%, which for a
one-seed result is about as clean as it gets. So there is real information in knowing who your
actual neighbours are.

**What it costs.** The effect is 0.04% to 1.04%. Seed-to-seed variation on these same
single-disease runs is 8% to 14% (`Reports/Capacity_Probe_5Seed.md:7`). The graph is worth roughly
one tenth of the noise you get from changing the random seed. And the sign is not clean at the
individual-run level: 3 of the 20 seed cells came out negative, influenza_japan seed 42 at -0.09%,
covid_us-states seed 52 at -0.02% and covid_us-states seed 72 at -0.19%. Consistent at the
panel-mean level, not consistent run by run.

### Test 3: the mechanism. NEW.

From `conda run -n ebola-train python diagnostics/graph_probe/t3.py`. For each panel I measured the
district-specific share of energy in the representation `h`, then in the neighbour aggregate `A@h`,
then how far `A@h` moves when the graph is relabelled.

| panel | distinct% of h | distinct% of A@h | destroyed | relabel displacement | sqrt(distinct% of A@h) | ratio |
|---|---|---|---|---|---|---|
| influenza_japan | 3.8% | 3.3% | 14% | 0.1881 | 0.182 | 1.04 |
| covid_us-states | 4.8% | 3.9% | 18% | 0.2035 | 0.197 | 1.03 |
| influenza_us-regions | 8.6% | 3.5% | 59% | 0.1986 | 0.187 | 1.06 |
| influenza_us-states | 12.3% | 5.0% | 60% | 0.2554 | 0.224 | 1.14 |
| dengue, seed 42 | 22.4% | 15.5% | 31% | 0.4423 | 0.394 | 1.12 |

The last two columns are the finding. If relabelling the graph randomised the entire
district-specific part of the neighbour message and left the shared part untouched, the aggregate
would move by exactly the square root of the district-specific energy share. Measured displacement
divided by that prediction is 1.03 to 1.14 on all five panels. Relabelling therefore scrambles
essentially all of the personal content of the message and none of the shared content, and doing so
costs under 1% of error.

The chain, in order:

1. The encoder produces a representation that is 78% to 96% one shared vector.
2. Averaging over neighbours cancels most of the small remaining personal part. 14% to 60% of it is
   destroyed by the averaging alone.
3. So the message that arrives at a district is nearly independent of which districts sent it.
4. So scrambling the graph costs under 1%, and deleting the spatial channel entirely costs nothing
   on error.

Step 4 is what the existing gate ablation already measured. The new part is steps 1 to 3, which say
why.

### C5. The gate is open, not closed

Worth recording because the obvious counter-argument to the ablation is "the model just turned the
graph off, so of course removing it changed nothing". It did not.
`Reports/gated+spatial_Contribution.txt:5-9`:

| dataset | gate mean | gate IQR | fraction with gate < 0.05 | spatial contribution |
|---|---|---|---|---|
| dengue | 0.604 | [0.554, 0.664] | 0.0% | 0.637 |
| influenza_japan | 0.271 | [0.238, 0.297] | 0.0% | 0.395 |
| influenza_us-regions | 0.370 | [0.335, 0.414] | 0.0% | 0.490 |
| influenza_us-states | 0.373 | [0.319, 0.425] | 0.0% | 0.468 |

Not one node in any panel closes the gate below 0.05. The model opens it wide, pushes a large
spatial contribution through, and gets nothing for it.

### C6. Cross-check against the existing gate ablation

`results/reports/gate_ablation.log:134` reads: "TALLY over 60 cells: 6 the graph helps, 9 the graph
hurts, 45 within noise." I reparsed the log's own per-cell verdict lines and split them by metric:

| metric family | cells | helps | hurts | within noise |
|---|---|---|---|---|
| error, RMSE plus MAE | 40 | **0** | 8 | 32 |
| correlation, PCC | 20 | 6 | 1 | 13 |
| all | 60 | 6 | 9 | 45 |

That matches `CLAUDE.md` section 4 exactly. The graph never once helps error and helps correlation
in 6 of 20 cells. Two of the eight error cells where it hurts are dengue, RMSE h10 and MAE h10.

The log states its own ceiling on its last line, `results/reports/gate_ablation.log:136`:

> CEILING: g=0 removes neighbour mixing but keeps the LTR degree feature, so this bounds the value
> of NEIGHBOUR INFORMATION, not of the graph in total.

That caveat is real and it applies to my tests too. Node degree survives every one of my
permutations, because permuting labels preserves the degree sequence and each node keeps its own
degree under the relabel. So none of this bounds the value of degree as a feature.

---

## D. Training and the early-stop saga

### D1 and D2. Every run early-stopped

```
conda run -n ebola-train python -m diagnostics.epoch_budget_audit
```

Single-disease: 25 of 25 runs have a surviving log and 25 of those early-stopped. One of them,
`influenza_us-states` seed 52, is flagged by the audit itself as "stopped early (tail != patience,
check)", so it is early-stopped but the tail length does not match patience cleanly.

LDO3 trunks: all 15 early-stopped, using **14.3% to 29.7%** of the 91,000-step budget, with best
checkpoints between step 1000 and step 15000.

| fold | best-checkpoint steps across the 5 seeds | steps used |
|---|---|---|
| dengue | 4000, 4000, 1000, 4000, 5000 | 13000 to 17000 |
| influenza | 7000, 5000, 2000, 5000, 8000 | 14000 to 20000 |
| covid | 3000, 5000, 5000, 15000, 8000 | 15000 to 27000 |

The 15 held-out adapters also all early-stopped, at 41.2% to 93.8% of 80 epochs.

### D3. Running the full budget changes nothing, and running it longer makes it worse

```
conda run -n ebola-train python -m diagnostics.ldo3full_check
-> 10/10 records identical.
```

All ten full-budget records, five few-shot and five zero-shot, are identical to the truncated runs.
The full-budget trunk logs show why:

| fold | last step where validation improved | best val | val at step 91000 |
|---|---|---|---|
| covid | 3000 | 0.1811 | 0.1948 |
| dengue | 4000 | 0.1183 | 0.1506 |
| influenza | 7000 | 0.1768 | 0.1921 |

Source: `results/reports/ldo3full_covid_seed42.log`, `ldo3full_dengue_seed42.log` and
`ldo3full_influenza_seed42.log`, all run with `trunk_patience=999` so nothing could stop them.
Validation at the end of the budget is 7.6% to 27.3% worse than at the best checkpoint in all three.

The Ebola trunk runs with `TRUNK_PATIENCE = 30` (`train/ebola.py:47`) rather than LDO3's 12. Per
seed, from `results/reports/ebola_run.log`:

| seed | last improvement | best val | final step | final val |
|---|---|---|---|---|
| 42 | 3000 | 0.1601 | 33000 | 0.1729 |
| 52 | 5000 | 0.1612 | 35000 | 0.1718 |
| 62 | 3000 | 0.1605 | 33000 | 0.1751 |
| 72 | 5000 | 0.1628 | 35000 | 0.1722 |
| 82 | 2000 | 0.1590 | 32000 | 0.1714 |

Same picture with a longer fuse. Nothing improves after step 5000 and the extra 27,000 to 30,000
steps only make validation worse.

### D4. Why the budget was wrong. My reasoning, recorded.

The trunk loop at `train/lodo.py:115-133` takes one optimizer step per origin per dataset, with no
gradient accumulation. Each step draws one origin from each training panel, concatenates them into
one block graph, computes one loss and calls `opt.step()` once.

The smallest training panel is COVID at 60 train origins. In the folds where COVID is a training
panel, which is the dengue fold and the influenza fold, 91,000 steps means COVID's 60 examples are
revisited 1,516 times. By the dengue fold's best checkpoint at step 4000 they had already been
revisited about 67 times.

So the verdict on the early-stop defect: **the bug was real, its effect was zero, and the early stop
was preventing about 87,000 steps of active overfitting.** The 91,000-step budget was sized for a
dataset this project does not have. Both halves matter. Do not write this up as "the bug did not
matter" without the second half, because the reason it did not matter is that something else was
masking it.

One caveat on the arithmetic. In the covid-held-out fold, COVID is not a training panel at all, so
the 1,516 figure applies to the other two folds. The smallest training panel in the covid fold is
influenza_japan at 152 origins, which is 598 revisits over the full budget.

### D5. The Ebola epoch CV mostly does not select

`train/ebola.py:156` `choose_epochs` runs leave-one-district-out cross-validation inside the support
set to pick the adapter epoch count, so that the epoch count is not a free parameter chosen after
seeing the answer. It is a good idea and it mostly does not fire. From
`results/reports/ebola_run.log`, all ten lines:

| line | arm | support districts | held-out cells | chosen epoch | pinball at best | pinball at ep80 |
|---|---|---|---|---|---|---|
| 42 | L12 | 17 | 104 | 80 | 0.2724 | 0.2724 |
| 45 | L20 | 36 | 320 | 80 | 0.2692 | 0.2692 |
| 84 | L12 | 17 | 104 | 73 | 0.3135 | 0.3135 |
| 87 | L20 | 36 | 320 | 73 | 0.2710 | 0.2710 |
| 124 | L12 | 17 | 104 | 80 | 0.2746 | 0.2746 |
| 127 | L20 | 36 | 320 | 80 | 0.2599 | 0.2599 |
| 166 | L12 | 17 | 104 | 77 | 0.2928 | 0.2928 |
| 169 | L20 | 36 | 320 | 80 | 0.2677 | 0.2677 |
| 205 | L12 | 17 | 104 | 59 | 0.2794 | 0.2798 |
| 208 | L20 | 36 | 320 | 80 | 0.2626 | 0.2626 |

It chose the cap of 80 in **6 of 10 runs**. In 9 of the 10, the pinball loss at the chosen epoch is
identical to the pinball at epoch 80 to four decimal places, so even the runs that picked 73, 73 or
77 were choosing between values the CV could not separate. Only the epoch-59 run shows any
difference at all, 0.2794 against 0.2798, which is 0.14%.

That is a real limitation of the mechanism. The curve is flat, the CV has nothing to grip, and in
the majority of runs it hits the ceiling rather than selecting. Say so if the paper describes the
epoch count as cross-validated.

---

## E. Zero-shot beats few-shot across the board. NEW framing.

```
conda run -n ebola-train python diagnostics/graph_probe/eb3.py
```

Reading `results/ebola/*.json`, 5-seed means of `country_macro`, 2 arms by 4 metrics by 4 horizons
gives 32 cells. **Zero-shot wins 31 of 32.**

- `ebola_L12`, the primary arm: 15 of 16. The one loss is MAE at h15, 20.145 zero-shot against
  20.102 few-shot, a gap of 0.2%.
- `ebola_L20`, the pre-registered secondary arm: 16 of 16.

Worst damage on L20:

| metric | horizon | zero-shot | few-shot | change |
|---|---|---|---|---|
| MAE | 3 | 21.659 | 31.935 | +47% |
| MAE | 15 | 21.790 | 31.541 | +45% |
| MAE | 10 | 24.404 | 35.144 | +44% |
| RMSE | 3 | 35.235 | 47.035 | +33% |
| RMSE | 10 | 39.372 | 51.580 | +31% |

### E2. More support data makes adaptation worse, not better

From `configs/ebola_arms.json`:

| arm | support cells | support districts | adaptation pairs h3/h5/h10/h15 |
|---|---|---|---|
| ebola_L12 (primary) | 59 | 18 | 48 / 38 / 18 / 0 |
| ebola_L20 (secondary) | 113 | 36 | 102 / 92 / 72 / 54 |

L20 has roughly twice the support cells and three times the adaptation pairs of L12, and adapting on
it is far worse. L12 loses 1 of 16 cells to adaptation and its worst damage is single-digit
percentages. L20 loses 16 of 16 and its worst damage is 47%.

### Two caveats you must keep attached to this

**These are point means with no interval.** Nothing here is a significance claim. It is directional.

**This is a different statistic from the one already on record.** `CLAUDE.md` section 4 and
`Resume.md:58` say "unadapted wins 12 of 16, adapted wins 2 of 16". That counts which arm produced a
**significant win against persistence** under `ebola_ci.py`'s district bootstrap. Mine counts which
arm has the better point mean, head to head, with no floor involved and no interval. They are not
the same number and they must not be presented as if one supports the other.

**The audit already bans the strong version.** `Reports/Phase0_to_Now_Audit.md:256` lists
"Zero-shot beats few-shot / adaptation hurts" *as a tested result* on the unsafe-to-claim list. The
paired-by-seed test that the shared trunk permits is significant in only 4 of 16 cells, and the
per-origin bootstrap on the inversion has never been run. My 31 of 32 does not change that. It
strengthens the observation and it does not turn it into a test.

### The mechanism worth naming

The adapter has 1,428 free parameters and is fitted on 59 support cells on the primary arm. That is
24 parameters per support cell, or 14 per usable adaptation pair. It is very heavily
over-parameterised, so memorising the support set rather than learning from it is the obvious
reading.

`models/adapters.py:14-16` proves the frozen-trunk adapter is exactly an affine map of frozen
features, which makes the inner loop ANIL in the exact sense. `CLAUDE.md` section 4 records ANIL as
better than its seed-matched control in 0 of 32 cells across four folds. Those two facts fit
together: the thing that fails on Ebola is the same object that fails on the LDO3 folds.

I will not oversell the count argument, though, because it predicts the wrong direction. If the
problem were purely parameters per observation, L20 with 113 support cells (13 parameters per cell)
should be damaged *less* than L12 with 59 (24 per cell). It is damaged far more. Something other
than raw count is driving it. See section F.

---

## F. What I could not resolve

**1. Why the representation is 78% to 96% shared.** Two very different diagnoses fit the same
measurements.

- *The data really is like this.* Weekly case counts across districts may genuinely carry very
  little district-to-district structure once you know the national trajectory, in which case the
  encoder found the right answer and the graph has little to offer. Fix: none needed, report it.
- *Training drove the encoder into a narrow solution.* Pinball loss averaged over many nodes rewards
  getting the common level right and barely penalises missing the district-level deviation, so the
  optimiser may have collapsed onto the shared component. Fix: a different loss weighting or a term
  that rewards separating districts.

Nothing I ran separates them. A test that would: measure the district-specific share of energy in
the *inputs* and compare it to the share in the representation. If the inputs carry 30% and the
representation carries 4%, the encoder is throwing it away. If the inputs carry 5%, it was never
there to throw away. I did not run it.

**2. The FiLM fold has no test.** True to 7.2e-07, protected by nothing. Section A3.

**3. The Ebola epoch CV hits its ceiling more often than it selects.** 6 of 10 runs at the cap, and
in 9 of 10 the chosen epoch scores identically to the cap. Section D5.

**4. The over-parameterisation story does not explain E2.** L20 has more support data per adapter
parameter and is hurt more. Candidate explanations I did not test: L20's support window crosses the
six-week zero-report gap; L20's pooled scaler is fitted on different statistics (mean 1.791, sd
1.529 against L12's 1.440 and 1.214, `configs/ebola_arms.json`); L20's support set reaches districts
that are quiet during the query period. All three are guesses.

---

## Numbers I was handed that did not reproduce

I was asked to recompute everything rather than trust the figures I was given. Five did not come
back the same. The disk value is what I wrote above in every case.

1. **"Roughly 50 parameters per observation" for the Ebola adapter.** Does not reproduce on any
   denominator I can find. Disk: 1,428 parameters against 59 support cells on L12 is 24 per cell, or
   14 per adaptation pair. On L20 it is 13 per cell and 4.5 per pair. Still badly
   over-parameterised, but not 50, and the L20 figures cut against the argument rather than for it.
2. **"The Ebola trunk last improved at step 2000 and ran to 32000."** True for seed 82 only. Across
   the five seeds it is steps 2000 to 5000 and final steps 32000 to 35000. Table in D3.
3. **"The sign is consistent" on the permutation test.** Consistent at the panel-mean level, all
   five positive. Not consistent run by run: 3 of the 20 seed cells are negative.
4. **"Every ratio is 1.00 to 1.15" in test 3.** Measured range is 1.03 to 1.14. Directionally the
   same, but the bottom of the stated range is not attained by any panel.
5. **"Pinball at best equals pinball at ep80" in the Ebola epoch CV.** True in 9 of 10 runs. The
   epoch-59 run reads 0.2794 against 0.2798, a real if tiny difference. Minor, but the word "equals"
   is doing work in that sentence.

One more that is a rounding difference rather than a discrepancy: I was given the dengue permutation
series as +0.58 / +0.60 / +0.58 / +0.61 / +0.60. My run prints +0.59 / +0.60 / +0.58 / +0.61 /
+0.60. The mean is +0.59% either way.

Everything else in sections A through E reproduced to the digits quoted.

## How I checked the checks

Every number here came out of a parser or a model run, and a parser that silently misreads is worse
than no parser. So I broke the inputs on purpose and confirmed the checks noticed:

- Flipped one "GATE HELPS" to "gate HURTS" in an in-memory copy of `gate_ablation.log`. The metric
  split moved from PCC 6 help / 1 hurt to 5 help / 2 hurt. Detected.
- Subtracted 20.0 from one `encoder_ebola` L20 RMSE h3 record in memory. The zero-shot tally moved
  from 31/1 to 30/2. Detected.
- Zeroed one observed cell in the Ebola mask. The count moved from 1299 to 1298. Detected.

All three mutations were caught. No file on disk was modified by any of it.

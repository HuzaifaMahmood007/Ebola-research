# Reproduction Failure Log

**Opened 2026-09-06.** Ordered by client decision D10 and Week 4 work order section 1d:

> *"start the reproduction failure log now, every baseline we couldn't reproduce and exactly why...
> it's far easier to keep as you go than to reconstruct at the end."*

Every published model we intended to use as a comparator and could not, and exactly what stopped it.
One entry per model. Each entry says whether the blocker is a property of our data, our environment,
our run, or the published work.

The companion documents are `Published Model benchmarks/`, which holds the published numbers
transcribed verbatim, and the published-versus-ours reproduction table produced by
`diagnostics/paper_compare.py`, which covers the baselines that did reproduce.

**Numbers in this document were recomputed from the artifacts on 2026-09-06**, not copied from the
progress docs. The recomputation commands are named in each entry.

---

## How to read this: what "could not reproduce" should mean

The client's condition is *"if we can't get close, that baseline doesn't go in the comparison
table."* That needs a threshold, and the honest one is not tight.

HeatGNN's Table III re-ran Cola-GNN rather than copying its published numbers. Against Cola-GNN's own
paper, on the same dataset, the same horizon and the same metric:

| dataset | h | Cola-GNN's paper | HeatGNN's re-run of Cola-GNN | difference |
|---|---|---|---|---|
| Japan-Prefectures | 2 | 929 | 1168 | +25.7 percent worse |
| US-Regions | 2 | 480 | 552 | +15.0 percent worse |
| US-States | 2 | 136 | 148 | +8.8 percent worse |
| US-States | 5 | 202 | 224 | +10.9 percent worse |

Two published papers, same model, same data, same metric, and they disagree by 9 to 26 percent. That
is the published field's own reproduction tolerance. A re-run that lands inside roughly 25 percent of
a paper's figure is not a failed reproduction, and we should not claim it is. The entries below are
not near-misses. They are cases where no comparable number exists at all.

---

## Section A: never produced a number in our suite

### A1. MepoGNN: dropped, our data lacks the required input

Both MepoGNN variants need a mobility or origin-destination matrix. Our harmonised schema ships
`A_mob = None` and the flag `A_mob_available` is false for every bundle. The Dynamic variant needs a
full OD tensor. The Adaptive variant needs a static commuter matrix both to seed its graph and to
drive its SIR component. Neither can be synthesised from what we have.

**This is a property of our data, not a model failure.** MepoGNN's Phase 1 reproduction on the
paper's own Japan COVID data passed inside the published standard deviation bands. We are not saying
the model does not work. We are saying our disease panels carry no mobility layer, so the model
cannot be run on them at all.

**Cost to unblock:** acquiring commuter or movement data for twelve dengue countries and three
influenza panels. Out of scope for this project.

### A2. MSGNN: not run, environment blocker

MSGNN requires Ubuntu with CUDA 10.1. The project machine is Windows. No run was attempted.

Worth stating plainly, because solving the environment would not actually buy a comparator: MSGNN
evaluates weekly US-county COVID against live Forecast-Hub submissions, on horizons expressed in
weeks, over an evaluation window rather than a fixed test split. None of that maps onto our pipeline.
Reproducing it would validate the install and nothing else. The paper also publishes single values
with no dispersion, so there would be nothing to compare against statistically even if it ran.

**Cost to unblock:** a Linux box with a CUDA 10.1 toolchain, and then a comparison that still would
not be like for like.

### A3. STOEP: excluded, and the reason we had on file was wrong

**This entry is a correction. The old reason does not survive, but the exclusion does.**

STOEP was timeboxed and excluded on the grounds that the paper's table disagreed with the shipped
dataset and metric: overall RMSE 63.5 in the paper against 182.9 on our run, RAE 0.37 against 0.224,
which reads as a threefold scale gap and therefore a different dataset.

The paper prints **two** overall rows, not one:

| dataset | RMSE | MAE | SMAPE | RAE |
|---|---|---|---|---|
| Flu, Zhejiang, 11 cities | 63.5 | 32.4 | 38.8 | 0.37 |
| COVID-19, Japan, 47 prefectures | 169.1 | 68.8 | 45.4 | 0.21 |

63.5 and 0.37 are the **Flu** row. We compared against it. Our run reported 182.9 and 0.224, which
sits beside the **COVID-19** row at 169.1 and 0.21.

**Which dataset our run used is now settled.** The shipped repository contains exactly one data file,
`data/jp20200401_20210921.npy`, holding `od (539, 47, 47, 1)`, `node (539, 47, 4)` and
`SIR (539, 47, 3)`. That is 539 days over 47 nodes, matching the paper's COVID-19 protocol
(47 prefectures of Japan, 2020-04-01 to 2021-09-21) exactly. The Zhejiang flu data, 11 cities over
731 days, is **not shipped at all**, so our run could not have used it.

So the correct comparison is 182.9 against 169.1, **8.2 percent off on RMSE** and 6.7 percent off on
RAE. That is well inside the 9 to 26 percent tolerance the published field shows against itself. The
threefold gap was a table-selection error on our side. **STOEP was not a failed reproduction.**

The metric mismatch is separate and still real. The paper's third column is SMAPE and our run logged
MAPE. Those are different quantities and cannot be compared at all. It does not change the verdict,
because RMSE and RAE alone already put the run beside the COVID row.

**Why STOEP stays excluded anyway.** It needs a mobility input we do not have, exactly like MepoGNN.
The shipped `od` tensor is required, and the paper sources it from Facebook Movement Range Maps for
COVID and the Baidu Migration Map for flu. Our schema ships `A_mob = None`. STOEP reproduces fine on
its own data and still cannot be run on ours.

**Status: closed.** The exclusion is correct, the recorded reason for it was not, and the record now
says so. Any document repeating "STOEP's paper table disagrees with its shipped dataset by threefold"
should be corrected to the mobility blocker.

---

## Section B: ran, but not on the paper's protocol

These produced usable numbers. They are recorded here because the number is not a like-for-like
reproduction and any table carrying it must say so.

### B1. Dengue is not like for like

Our encoder is scored on the full dengue bundle at **7,165 nodes**. The baselines are scored on a
**2,392-node** stratified subsample, one third of the nodes, drawn per country so no country
disappears, with a fixed seed of 20260715 so the subset is identical across every model and every
re-run (`export_baseline.py:41-64`).

The reason is capacity. The baseline repositories were built for panels of at most about 49 nodes and
run out of memory at 7,165. Recomputed from `data/processed/dengue.npz` and
`baselines/_exported/dengue/meta.json`: 7,165 nodes against 2,392, so 33.4 percent retained.

**Consequence.** Any dengue row that puts our encoder next to a baseline is comparing two different
node populations. This is a disclosed caveat, not a defect, but it must appear wherever the dengue
row appears.

### B2. Cola-GNN and HeatGNN are influenza only

Both run on CPU-only environments in our setup, and the dengue one-third subsample is still too slow
there to finish. Verified against `results/baselines/`: EpiGNN 80 records including dengue, MTGNN 80
including dengue, Cola-GNN 60 and HeatGNN 60, both influenza only.

**Consequence.** On dengue, the only baseline in the published lineage is EpiGNN. Any statement about
how we compare on dengue rests on one comparator.

### B3. HeatGNN deviates from its paper by design

HeatGNN's `getLaplaceMat` omits the identity, so an isolated node produces a zero row, a division by
zero, and NaN through the rest of the forward pass. We add self-loops to HeatGNN's staged adjacency
to stop it (defect D7).

Recomputed degree-zero counts from the exported graphs: influenza_japan has 2 isolated nodes,
influenza_us-states has 2, influenza_us-regions has 0. The paper presumably never hit this because
its own graphs have no isolated nodes.

**Consequence.** Our HeatGNN is not exactly the published HeatGNN. Without the patch it produces NaN
rather than a worse number, so the alternative is no HeatGNN column at all. The deviation must be
stated on the comparison table.

HeatGNN's protocol also differs from ours in ways the patch does not touch: the paper uses a 60/20/20
split at horizons 2, 5, 7 and 12, where we use 50/20/30 at 3, 5, 10 and 15. Only h5 overlaps, so
HeatGNN has exactly one cell per dataset that can be read against the paper.

---

## Section C: ran, produced numbers, and the numbers are not usable

### C1. MTGNN collapsed to a constant on most runs

**47 of 80 MTGNN prediction files contain exactly one distinct value**, one number repeated for every
node at every forecast origin. Recomputed on 2026-09-06 directly from `baselines/_preds/*.npz`, and
the same check over every other baseline returns zero constant files:

| model | constant files | total |
|---|---|---|
| MTGNN | **47** | 80 |
| EpiGNN | 0 | 80 |
| Cola-GNN | 0 | 60 |
| HeatGNN | 0 | 60 |

Where it fails, by cell:

| cell | constant seeds | note |
|---|---|---|
| dengue h5, h10, h15 | 5 of 5 each | every seed collapsed, all three horizons |
| influenza_japan, all four horizons | 4 of 5 each | the surviving seed holds 16 to 75 distinct values over 4,888 cells |
| influenza_us-states, all four horizons | 3 of 5 each | the two survivors are fully varied |
| influenza_us-regions, all four horizons | 1 of 5 each | two further seeds hold 2 to 21 distinct values over 2,350 cells |
| dengue h3 | 0 of 5 | the only MTGNN cell that behaves like a trained model throughout |

Corroborating evidence: pooled PCC is negative on every influenza dataset, seed standard deviation is
exactly 0.00 wherever the runs are constant, and on Japan and dengue two different seeds produced
byte-identical prediction files. A model that ignores its seed and emits one number has not
converged. It has collapsed to something like a global mean.

**Consequence.** MTGNN cannot enter the comparison table, and every "we beat MTGNN" figure computed
from these files is meaningless, because the thing being beaten is a constant. Beating a constant is
already measured by our `train_mean` and persistence floors, which are honest about being floors.

**This is a failure of our run, not of the published model.** MTGNN's own paper reports on traffic,
solar, electricity and exchange-rate data and contains no epidemic dataset at all, so there is no
published epidemic number of its own to reproduce. Two independent third-party re-runs of MTGNN on
epidemic data, in the MepoGNN and STOEP papers, both obtained sensible numbers. The collapse is ours.

**Cost to unblock:** diagnose the collapse and re-run, unknown but likely a day of work. Not
recommended. MTGNN was never a like-for-like comparator, since its paper has no epidemic evaluation.

---

## Not a reproduction target

**LTGCN and GTGCN** are our own group's models on our own Brazil and Spain data. The published number
and our number are the same number, so there is nothing to reproduce. They are cited for design
context, not for the head-to-head table. Nothing about them is comparable to our pipeline anyway:
different countries, different spatial resolution, one-step forecasting, and a metric set including
MDA that we do not compute.

---

## What survives as a comparator

| model | dengue | influenza | usable | why not |
|---|---|---|---|---|
| EpiGNN | yes, subsampled | yes | **yes** | subsample caveat only |
| Cola-GNN | no | yes | **yes, influenza only** | CPU environment too slow for dengue |
| HeatGNN | no | yes | **yes, influenza only, one paper-comparable cell** | CPU speed, self-loop deviation, split mismatch |
| MTGNN | no | no | **no** | collapsed to a constant on 47 of 80 runs |
| MepoGNN | no | no | **no** | needs mobility data we do not have |
| MSGNN | no | no | **no** | needs Ubuntu and CUDA 10.1 |
| STOEP | no | no | **no** | needs mobility data we do not have, see A3 |

**Two usable comparators on influenza, one on dengue.** Any document claiming four clean baselines is
wrong.

---

## Open items in this log

1. **Correct the STOEP claim wherever it appears.** `Phase3_Week4_Work_Order.md` section 1d records
   STOEP as a threefold reproduction failure. It is an 8.2 percent reproduction blocked by missing
   mobility data. The old sentence must not reach the paper.
2. **A3, metric.** SMAPE against MAPE is still unsettled. It does not affect the verdict, but do not
   quote a STOEP percentage-error number until it is.
3. Nothing else here is actionable. A1, A2, B1, B2, B3 and C1 are closed as findings and simply need
   to be stated wherever the affected numbers appear.

---

## How the numbers here were checked

- Constant-prediction counts: every file in `baselines/_preds/*.npz` loaded and its distinct finite
  values counted, per model and per cell.
- Degree-zero node counts: off-diagonal adjacency from `baselines/_exported/*/adj.txt`, row degree
  counted.
- Dengue node counts: `data/processed/dengue.npz` array shape against
  `baselines/_exported/dengue/meta.json`.
- Baseline record counts: `results/baselines/` grouped by model and dataset.
- Published figures: transcribed in `Published Model benchmarks/`, never re-derived.

`baselines/` is gitignored, so the prediction archives behind Section C have no version history. The
counts above are the record.

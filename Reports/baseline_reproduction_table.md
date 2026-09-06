# Baseline Reproduction Table: published, reproduced, ours

**Written 2026-09-06.** Ordered by client decision D10 and Week 4 work order section 1c, which made
this a non-optional condition:

> *"validate each reproduction against the source paper's published numbers before using it as a
> comparator, and show me our reproduction next to their reported figure. If we can't get close,
> that baseline doesn't go in the comparison table."*

Three numbers per cell, all in one metric definition. Column 1 is transcribed from the paper's own
table. Column 2 is that baseline run on our pipeline. Column 3 is our encoder, the thing we are
actually claiming.

The baselines that produced no comparable number at all are in
[reproduction_failure_log.md](reproduction_failure_log.md). This document covers the ones that ran.

Regenerate with:

```
conda run -n ebola-train python -m diagnostics.paper_compare -o Reports/baseline_reproduction_table.md
```

which prints the table below. Check it with `diagnostics/verify_paper_table.py`.

---

## Read this before reading the table

**Every column is cell-pooled RMSE, and that is not our headline metric.** `score.py` reports a
per-node RMSE averaged over nodes and then over countries. Every one of these papers defines RMSE as
the square root of the mean over all scored cells. By Jensen those two always disagree, and the
mean-of-RMSEs is always the smaller number. A difference between the two aggregations would measure
the aggregation, not the reproduction. So this table is pooled throughout, and the country-macro
headline numbers in `Updated_Scores.txt` must never be mixed into it.

**The encoder never archived its predictions.** Column 3 is reconstructed from the per-origin error
sufficient statistics, where pooled RMSE is exactly `sqrt(sum(sse)/sum(n))`. The guard is that the
cell count implied by those statistics must equal the count in the matching `__pernode.npz`, or the
run is dropped loudly. No run was dropped.

**Dengue is pooled over the 2,392-node subsample on both sides**, and that correction matters more
than any other line in this document. See the next section.

**What counts as "close enough".** HeatGNN's Table III re-ran Cola-GNN rather than copying its
numbers, and disagreed with Cola-GNN's own paper by 8.8 to 25.7 percent on the same data, horizon
and metric. That is the published field's own reproduction tolerance. Judged against it, every
baseline in this table reproduces.

---

## The dengue correction, and why the first version of this table was wrong

The generator originally pooled the encoder over the full dengue bundle, 6,161 non-constant nodes of
7,165, while pooling the baselines over the 2,392-node subsample they were actually run on. The two
sides were different node populations, so the "encoder vs reproduced" column on dengue was measuring
the node set as much as the model.

I found this while filing the table, and priced it by restricting the encoder to the same 2,392
nodes:

| h | encoder, full 6,161 nodes | encoder, matched 2,392 nodes | shift | margin as first printed | margin, matched |
|---|---|---|---|---|---|
| 3 | 202.6 | 310.0 | +53.0% | -53.5% | **-28.9%** |
| 5 | 234.2 | 357.6 | +52.7% | -51.7% | **-26.2%** |
| 10 | 291.9 | 444.3 | +52.2% | -41.6% | **-11.2%** |
| 15 | 315.6 | 474.3 | +50.3% | -37.5% | **-6.0%** |

The encoder's own pooled dengue RMSE is about 50 percent higher on the subsample than on the full
panel. The subsample is stratified by country and is only 8.7 percent higher in mean incidence per
node, with a slightly lower median, so this is not a simple large-node effect. Pooled RMSE is
dominated by a small number of extreme nodes and the one-third draw happens to carry a
disproportionate share of them.

**Consequence.** We still beat EpiGNN on dengue at all four horizons, and the direction of every
dengue conclusion is unchanged. The margin is roughly half of what the uncorrected table said, and
at h15 it falls to 6.0 percent. Against seed dispersion of 474.3 ± 1.9 for us and 504.7 ± 11.5 for
EpiGNN that gap is still real, but it is no longer a large one.

`diagnostics/paper_compare.py` now restricts the encoder column to the exported node set for any
subsampled dataset, reusing `export_baseline._kept_indices` so the table cannot drift from the
export. The numbers below are the corrected ones.

---

## The table

| model | dataset | h | 1. published | 2. reproduced (ours) | Δ% vs published | 3. our encoder | encoder vs reproduced | seeds | flag |
|---|---|---|---|---|---|---|---|---|---|
| Cola-GNN | influenza_japan | 3 | 1051 | 1108.3 ± 82.2 | +5.5% | 1034.3 ± 69.1 | -6.7% | 5/5 |  |
| Cola-GNN | influenza_japan | 5 | 1117 | 1197.2 ± 93.6 | +7.2% | 1168.8 ± 56.9 | -2.4% | 5/5 |  |
| Cola-GNN | influenza_japan | 10 | 1372 | 1504.7 ± 65.0 | +9.7% | 1469.9 ± 108.1 | -2.3% | 5/5 |  |
| Cola-GNN | influenza_japan | 15 | 1475 | 1500.5 ± 50.5 | +1.7% | 1427.3 ± 171.1 | -4.9% | 5/5 |  |
| Cola-GNN | influenza_us-regions | 3 | 636 | 710.1 ± 33.2 | +11.6% | 766.3 ± 114.3 | +7.9% | 5/5 |  |
| Cola-GNN | influenza_us-regions | 5 | 855 | 927.1 ± 36.3 | +8.4% | 906.5 ± 98.8 | -2.2% | 5/5 |  |
| Cola-GNN | influenza_us-regions | 10 | 1134 | 1168.2 ± 72.4 | +3.0% | 973.7 ± 55.4 | -16.6% | 5/5 |  |
| Cola-GNN | influenza_us-regions | 15 | 1203 | 1239.2 ± 135.3 | +3.0% | 995.3 ± 93.9 | -19.7% | 5/5 |  |
| Cola-GNN | influenza_us-states | 3 | 167 | 187.7 ± 7.0 | +12.4% | 187.3 ± 8.1 | -0.2% | 5/5 |  |
| Cola-GNN | influenza_us-states | 5 | 202 | 232.4 ± 8.5 | +15.0% | 221.7 ± 9.2 | -4.6% | 5/5 |  |
| Cola-GNN | influenza_us-states | 10 | 241 | 264.1 ± 15.3 | +9.6% | 240.4 ± 5.1 | -9.0% | 5/5 |  |
| Cola-GNN | influenza_us-states | 15 | 237 | 249.3 ± 18.3 | +5.2% | 244.1 ± 8.0 | -2.1% | 5/5 |  |
| EpiGNN | dengue | 3 | — | 435.9 ± 13.8 | — | 310.0 ± 21.0 | -28.9% | 5/5 | both sides on the 2,392-node subsample |
| EpiGNN | dengue | 5 | — | 484.8 ± 25.6 | — | 357.6 ± 24.4 | -26.2% | 5/5 | both sides on the 2,392-node subsample |
| EpiGNN | dengue | 10 | — | 500.1 ± 20.9 | — | 444.3 ± 18.7 | -11.2% | 5/5 | both sides on the 2,392-node subsample |
| EpiGNN | dengue | 15 | — | 504.7 ± 11.5 | — | 474.3 ± 1.9 | -6.0% | 5/5 | both sides on the 2,392-node subsample |
| EpiGNN | influenza_japan | 3 | 996 | 1182.3 ± 65.7 | +18.7% | 1034.3 ± 69.1 | -12.5% | 5/5 |  |
| EpiGNN | influenza_japan | 5 | 1031 | 1273.1 ± 322.1 | +23.5% | 1168.8 ± 56.9 | -8.2% | 5/5 |  |
| EpiGNN | influenza_japan | 10 | 1441 | 1701.2 ± 98.9 | +18.1% | 1469.9 ± 108.1 | -13.6% | 5/5 |  |
| EpiGNN | influenza_japan | 15 | 1470 | 1598.0 ± 201.7 | +8.7% | 1427.3 ± 171.1 | -10.7% | 5/5 |  |
| EpiGNN | influenza_us-regions | 3 | 589 | 632.4 ± 36.1 | +7.4% | 766.3 ± 114.3 | +21.2% | 5/5 |  |
| EpiGNN | influenza_us-regions | 5 | 774 | 893.8 ± 55.4 | +15.5% | 906.5 ± 98.8 | +1.4% | 5/5 |  |
| EpiGNN | influenza_us-regions | 10 | 984 | 1053.3 ± 47.4 | +7.0% | 973.7 ± 55.4 | -7.6% | 5/5 |  |
| EpiGNN | influenza_us-regions | 15 | 1061 | 1107.4 ± 96.7 | +4.4% | 995.3 ± 93.9 | -10.1% | 5/5 |  |
| EpiGNN | influenza_us-states | 3 | 160 | 167.8 ± 7.2 | +4.9% | 187.3 ± 8.1 | +11.6% | 5/5 |  |
| EpiGNN | influenza_us-states | 5 | 186 | 195.9 ± 11.9 | +5.3% | 221.7 ± 9.2 | +13.1% | 5/5 |  |
| EpiGNN | influenza_us-states | 10 | 220 | 225.8 ± 8.2 | +2.6% | 240.4 ± 5.1 | +6.4% | 5/5 |  |
| EpiGNN | influenza_us-states | 15 | 236 | 237.4 ± 7.2 | +0.6% | 244.1 ± 8.0 | +2.8% | 5/5 |  |
| HeatGNN | influenza_japan | 3 | — | 1164.4 ± 184.8 | — | 1034.3 ± 69.1 | -11.2% | 5/5 |  |
| HeatGNN | influenza_japan | 5 | 1378 | 1155.3 ± 158.5 | -16.2% | 1168.8 ± 56.9 | +1.2% | 5/5 |  |
| HeatGNN | influenza_japan | 10 | — | 1694.5 ± 126.5 | — | 1469.9 ± 108.1 | -13.3% | 5/5 |  |
| HeatGNN | influenza_japan | 15 | — | 1518.5 ± 145.0 | — | 1427.3 ± 171.1 | -6.0% | 5/5 |  |
| HeatGNN | influenza_us-regions | 3 | — | 672.5 ± 36.5 | — | 766.3 ± 114.3 | +14.0% | 5/5 |  |
| HeatGNN | influenza_us-regions | 5 | 852 | 915.4 ± 63.2 | +7.4% | 906.5 ± 98.8 | -1.0% | 5/5 |  |
| HeatGNN | influenza_us-regions | 10 | — | 1148.5 ± 66.6 | — | 973.7 ± 55.4 | -15.2% | 5/5 |  |
| HeatGNN | influenza_us-regions | 15 | — | 1174.5 ± 68.3 | — | 995.3 ± 93.9 | -15.3% | 5/5 |  |
| HeatGNN | influenza_us-states | 3 | — | 166.7 ± 13.5 | — | 187.3 ± 8.1 | +12.4% | 5/5 |  |
| HeatGNN | influenza_us-states | 5 | 186 | 208.0 ± 15.8 | +11.8% | 221.7 ± 9.2 | +6.6% | 5/5 |  |
| HeatGNN | influenza_us-states | 10 | — | 238.1 ± 16.1 | — | 240.4 ± 5.1 | +0.9% | 5/5 |  |
| HeatGNN | influenza_us-states | 15 | — | 256.0 ± 19.7 | — | 244.1 ± 8.0 | -4.6% | 5/5 |  |
| MTGNN | dengue | 3 | — | 406.4 ± 13.2 | — | 310.0 ± 21.0 | -23.7% | 5/5 | both sides on the 2,392-node subsample |
| MTGNN | dengue | 5 | — | 490.9 ± 0.0 | — | 357.6 ± 24.4 | -27.1% | 5/5 | **5/5 constant**; both sides on the 2,392-node subsample |
| MTGNN | dengue | 10 | — | 487.7 ± 0.0 | — | 444.3 ± 18.7 | -8.9% | 5/5 | **5/5 constant**; both sides on the 2,392-node subsample |
| MTGNN | dengue | 15 | — | 484.4 ± 0.0 | — | 474.3 ± 1.9 | -2.1% | 5/5 | **5/5 constant**; both sides on the 2,392-node subsample |
| MTGNN | influenza_japan | 3 | — | 1662.8 ± 0.0 | — | 1034.3 ± 69.1 | -37.8% | 5/5 | **4/5 constant** |
| MTGNN | influenza_japan | 5 | — | 1882.2 ± 0.0 | — | 1168.8 ± 56.9 | -37.9% | 5/5 | **4/5 constant** |
| MTGNN | influenza_japan | 10 | — | 2118.5 ± 0.0 | — | 1469.9 ± 108.1 | -30.6% | 5/5 | **4/5 constant** |
| MTGNN | influenza_japan | 15 | — | 2068.4 ± 0.0 | — | 1427.3 ± 171.1 | -31.0% | 5/5 | **4/5 constant** |
| MTGNN | influenza_us-regions | 3 | — | 1561.2 ± 59.2 | — | 766.3 ± 114.3 | -50.9% | 5/5 | **1/5 constant** |
| MTGNN | influenza_us-regions | 5 | — | 1564.8 ± 13.7 | — | 906.5 ± 98.8 | -42.1% | 5/5 | **1/5 constant** |
| MTGNN | influenza_us-regions | 10 | — | 1562.8 ± 14.0 | — | 973.7 ± 55.4 | -37.7% | 5/5 | **1/5 constant** |
| MTGNN | influenza_us-regions | 15 | — | 1541.4 ± 41.9 | — | 995.3 ± 93.9 | -35.4% | 5/5 | **1/5 constant** |
| MTGNN | influenza_us-states | 3 | — | 474.1 ± 11.9 | — | 187.3 ± 8.1 | -60.5% | 5/5 | **3/5 constant** |
| MTGNN | influenza_us-states | 5 | — | 465.9 ± 1.4 | — | 221.7 ± 9.2 | -52.4% | 5/5 | **3/5 constant** |
| MTGNN | influenza_us-states | 10 | — | 449.7 ± 14.2 | — | 240.4 ± 5.1 | -46.5% | 5/5 | **3/5 constant** |
| MTGNN | influenza_us-states | 15 | — | 447.1 ± 8.4 | — | 244.1 ± 8.0 | -45.4% | 5/5 | **3/5 constant** |

The `seeds` column is baseline seeds over encoder seeds. Every cell is 5 over 5.

---

## Verdict per baseline, against the client's condition

### Cola-GNN: reproduces. Admit it.

All 12 comparable cells land between +1.7 and +15.0 percent of the published figure, every one of
them worse than published and none by much. Twelve of twelve inside the field's own tolerance. The
paper's protocol is already ours, chronological 50/20/30 on the same three influenza panels, so
these are like-for-like comparisons with no caveat needed.

### EpiGNN: reproduces, and Japan is the weakest of it. Admit it.

All 12 comparable cells land between +0.6 and +23.5 percent. The US panels are tight, +0.6 to +15.5
percent. Japan is the loose one at +18.7, +23.5, +18.1 and +8.7 percent, and its h5 reproduction
carries a seed standard deviation of 322.1 on a mean of 1273.1, so that arm is unstable across
seeds as well as offset from the paper. It is still inside the 26 percent the published field shows
against itself, but Japan is the cell a reviewer will pick, and the honest description is "reproduces
at the edge of tolerance on Japan, comfortably on the US panels".

EpiGNN is also the only baseline that runs on dengue, so every dengue comparison rests on it alone.

### HeatGNN: reproduces on the three cells that can be compared. Admit it, with the caveats.

The paper uses a 60/20/20 split at horizons 2, 5, 7 and 12; we use 50/20/30 at 3, 5, 10 and 15. Only
h5 overlaps, so there are exactly three comparable cells out of twelve: -16.2, +7.4 and +11.8
percent. All three are inside tolerance. The other nine HeatGNN rows have no published counterpart
and their `Δ% vs published` column is empty by construction, not by omission.

Two caveats travel with every HeatGNN number. The split differs from the paper's, so even the three
comparable cells are not exact. And our HeatGNN is patched: `getLaplaceMat` omits the identity and
produces NaN on isolated nodes, of which Japan and US-States have two each, so we add self-loops.
Without the patch there is no HeatGNN column at all.

### MTGNN: cannot be validated, and must not go in the comparison table.

**MTGNN's own paper contains no epidemic dataset.** It reports on traffic, solar, electricity,
exchange rate, METR-LA and PEMS-BAY. So there is no published number to validate against, and the
`Δ% vs published` column is empty for all 16 cells. On the client's condition alone that is
disqualifying: we cannot show our reproduction next to their reported figure because they never
reported one on this kind of data.

Independently, the runs are degenerate. 47 of 80 prediction files hold a single distinct value, and
the seed standard deviation is exactly 0.00 wherever they are, which is visible in the table as
`± 0.0`. Details in [reproduction_failure_log.md](reproduction_failure_log.md) section C1.

**Both reasons are sufficient on their own.** Every "we beat MTGNN" figure computed from these runs
is meaningless.

---

## What column 3 says about our encoder

Against the two usable comparators, over the 40 cells that exclude MTGNN, the encoder is better in
**28** and worse in **12**. That is not a clean sweep and the pattern is worth stating plainly:

- **We win on dengue**, all four horizons against EpiGNN, by 6.0 to 28.9 percent on matched nodes.
- **We win on Japan**, 7 of 8 cells against EpiGNN and HeatGNN.
- **We lose on US-States.** Against EpiGNN we are worse at all four horizons, +2.8 to +13.1 percent.
  Against HeatGNN we are worse at three of four. Against Cola-GNN we win all four. US-States is our
  weak panel and no document currently says so.
- **US-Regions is mixed**, and our margin grows with horizon: we lose at h3 against both EpiGNN and
  HeatGNN and win at h10 and h15 against all three.

These are pooled RMSE on single-disease runs, which is our best case. They are not the transfer
numbers and must not be quoted as if they were.

---

## Open items

1. **The dengue node-set correction is new as of 2026-09-06.** Any earlier document quoting an
   encoder-versus-baseline dengue margin near 40 or 50 percent is quoting the uncorrected figure and
   needs the number from this table instead.
2. **US-States losses are undisclosed.** No current document says we lose to EpiGNN on US-States at
   every horizon. It belongs in the manuscript's results section, not only here.
3. `paper_compare.py` computes a reproduced PCC per cell and never prints it. Harmless, but it is
   the source of a `Mean of empty slice` warning on the degenerate MTGNN cells.

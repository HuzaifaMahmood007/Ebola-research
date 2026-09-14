# Baseline Reproduction Table: published, reproduced, ours

**Written 2026-09-06. Transfer columns and the two figures added 2026-09-10.** Ordered by client
decision D10 and Week 4 work order section 1c, which made this a non-optional condition:

> *"validate each reproduction against the source paper's published numbers before using it as a
> comparator, and show me our reproduction next to their reported figure. If we can't get close,
> that baseline doesn't go in the comparison table."*

Five numbers per cell, all in one metric definition. Column 1 is transcribed from the paper's own
table. Column 2 is that baseline run on our pipeline. Column 3 is our single-disease encoder, which
trains on the disease it is scored on. Columns 4 and 5 are our cross-disease transfer encoder, which
does not: the target disease is held out of training entirely, few-shot in column 4 and zero-shot in
column 5. Read the section "Columns 4 and 5" before reading either of them, because they are not a
like-for-like race against the baselines and cannot be quoted as one.

The baselines that produced no comparable number at all are in
[reproduction_failure_log.md](reproduction_failure_log.md). This document covers the ones that ran.

Regenerate with:

```
conda run -n ebola-train python -m diagnostics.paper_compare -o Reports/baseline_reproduction_table.md
```

which prints the table below. Check it with `diagnostics/verify_paper_table.py`. The two figures come
from `conda run -n ebola-train python -m baseline_bars_figure`, which reads the same archives.

---

## Read this before reading the table

**Every column is cell-pooled RMSE, and that is not our headline metric.** `score.py` reports a
per-node RMSE averaged over nodes and then over countries. Every one of these papers defines RMSE as
the square root of the mean over all scored cells. By Jensen those two always disagree, and the
mean-of-RMSEs is always the smaller number. A difference between the two aggregations would measure
the aggregation, not the reproduction. So this table is pooled throughout, and the country-macro
headline numbers in `Updated_Scores.txt` must never be mixed into it.

**The encoder never archived its predictions.** Columns 3, 4 and 5 are reconstructed from the
per-origin error sufficient statistics, where pooled RMSE is exactly `sqrt(sum(sse)/sum(n))`. The
guard is that the cell count implied by those statistics must equal the count in the matching
`__pernode.npz`, or the run is dropped loudly. No run was dropped.

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

## Columns 4 and 5: what cross-disease transfer is, and what it is not

Columns 4 and 5 are the LDO3 records already on disk, `results/lodo/encoder_ldo3__*` and
`results/lodo/encoder_ldo3_zeroshot__*`, both arms, five seeds, four horizons. Nothing was retrained
for this table. In these runs the target disease is removed from training completely. Column 4 then
adapts on the target panel's own training window, which is the few-shot arm. Column 5 adapts on
nothing at all.

**This is not a like-for-like comparison, and the column must never be quoted as if it were.**
EpiGNN, Cola-GNN and HeatGNN are transductive and node-indexed: each learns an embedding per region
from the target panel's own history, so there is no way to run any of them on a disease they were
never trained on. The number does not exist for them and cannot be produced. Our encoder is the only
model in this group that can attempt the task at all. So the right reading of columns 4 and 5 is the
price of the setting, meaning how much worse our own model gets when the target disease is taken
away from it. It is not evidence that we beat SOTA, and where we do come out ahead of a baseline in
those columns the reason is almost always that the baseline is weak in that cell, not that transfer
is strong.

**One caveat travels with every transfer number.** The LDO3 records were written before the trunk
early-stop defect was found (`train/lodo.py:56`, patience 12 against a cosine schedule with
`T_max=91000` that never annealed). A full-budget rerun at seed 42 across all three folds produced
records that are bit-identical to the truncated ones, 10 of 10 in `diagnostics/ldo3full_check.py`,
because the best checkpoints landed at steps 3000, 4000 and 7000, long before patience could fire.
That check covers seed 42 only, one of five. It is strong evidence that the truncation did not bind,
not proof.

**Dengue transfer is restricted to the same 2,392 nodes** as columns 2 and 3, by the same code path.

**The last four rows are COVID and have no baseline.** No epidemic GNN in the comparison set was run
on COVID at all, so columns 1 and 2 read "no run" there rather than being left blank. Our own three
columns are filled, because those runs exist. COVID is in neither figure for the same reason.

---

## The table

| model | dataset | h | 1. published | 2. reproduced (ours) | Δ% vs published | 3. our encoder | encoder vs reproduced | 4. transfer adapted | adapted vs 3 | 5. transfer zero-shot | zero-shot vs 3 | seeds | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cola-GNN | influenza_japan | 3 | 1051 | 1108.3 ± 82.2 | +5.5% | 1034.3 ± 69.1 | -6.7% | 973.2 ± 84.7 | -5.9% | 1196.4 ± 112.5 | +15.7% | 5/5/5/5 |  |
| Cola-GNN | influenza_japan | 5 | 1117 | 1197.2 ± 93.6 | +7.2% | 1168.8 ± 56.9 | -2.4% | 1417.2 ± 111.3 | +21.3% | 1660.8 ± 100.7 | +42.1% | 5/5/5/5 |  |
| Cola-GNN | influenza_japan | 10 | 1372 | 1504.7 ± 65.0 | +9.7% | 1469.9 ± 108.1 | -2.3% | 1977.5 ± 32.5 | +34.5% | 2155.8 ± 7.4 | +46.7% | 5/5/5/5 |  |
| Cola-GNN | influenza_japan | 15 | 1475 | 1500.5 ± 50.5 | +1.7% | 1427.3 ± 171.1 | -4.9% | 1980.3 ± 45.8 | +38.7% | 2148.4 ± 5.9 | +50.5% | 5/5/5/5 |  |
| Cola-GNN | influenza_us-regions | 3 | 636 | 710.1 ± 33.2 | +11.6% | 766.3 ± 114.3 | +7.9% | 740.3 ± 39.0 | -3.4% | 927.4 ± 38.2 | +21.0% | 5/5/5/5 |  |
| Cola-GNN | influenza_us-regions | 5 | 855 | 927.1 ± 36.3 | +8.4% | 906.5 ± 98.8 | -2.2% | 965.3 ± 60.3 | +6.5% | 1162.6 ± 60.7 | +28.2% | 5/5/5/5 |  |
| Cola-GNN | influenza_us-regions | 10 | 1134 | 1168.2 ± 72.4 | +3.0% | 973.7 ± 55.4 | -16.6% | 1217.7 ± 103.1 | +25.1% | 1398.0 ± 48.0 | +43.6% | 5/5/5/5 |  |
| Cola-GNN | influenza_us-regions | 15 | 1203 | 1239.2 ± 135.3 | +3.0% | 995.3 ± 93.9 | -19.7% | 1324.1 ± 68.8 | +33.0% | 1497.0 ± 35.5 | +50.4% | 5/5/5/5 |  |
| Cola-GNN | influenza_us-states | 3 | 167 | 187.7 ± 7.0 | +12.4% | 187.3 ± 8.1 | -0.2% | 180.3 ± 5.9 | -3.7% | 199.7 ± 5.0 | +6.6% | 5/5/5/5 |  |
| Cola-GNN | influenza_us-states | 5 | 202 | 232.4 ± 8.5 | +15.0% | 221.7 ± 9.2 | -4.6% | 217.9 ± 9.2 | -1.7% | 242.3 ± 8.3 | +9.3% | 5/5/5/5 |  |
| Cola-GNN | influenza_us-states | 10 | 241 | 264.1 ± 15.3 | +9.6% | 240.4 ± 5.1 | -9.0% | 259.3 ± 5.2 | +7.9% | 287.6 ± 5.4 | +19.6% | 5/5/5/5 |  |
| Cola-GNN | influenza_us-states | 15 | 237 | 249.3 ± 18.3 | +5.2% | 244.1 ± 8.0 | -2.1% | 260.9 ± 8.8 | +6.9% | 307.2 ± 4.7 | +25.8% | 5/5/5/5 |  |
| EpiGNN | dengue | 3 | — | 435.9 ± 13.8 | — | 310.0 ± 21.0 | -28.9% | 435.1 ± 18.4 | +40.4% | 475.2 ± 1.5 | +53.3% | 5/5/5/5 | both sides on the 2,392-node subsample |
| EpiGNN | dengue | 5 | — | 484.8 ± 25.6 | — | 357.6 ± 24.4 | -26.2% | 467.4 ± 5.6 | +30.7% | 481.6 ± 1.8 | +34.7% | 5/5/5/5 | both sides on the 2,392-node subsample |
| EpiGNN | dengue | 10 | — | 500.1 ± 20.9 | — | 444.3 ± 18.7 | -11.2% | 482.3 ± 1.4 | +8.6% | 483.7 ± 0.5 | +8.9% | 5/5/5/5 | both sides on the 2,392-node subsample |
| EpiGNN | dengue | 15 | — | 504.7 ± 11.5 | — | 474.3 ± 1.9 | -6.0% | 482.5 ± 0.2 | +1.7% | 480.9 ± 0.2 | +1.4% | 5/5/5/5 | both sides on the 2,392-node subsample |
| EpiGNN | influenza_japan | 3 | 996 | 1182.3 ± 65.7 | +18.7% | 1034.3 ± 69.1 | -12.5% | 973.2 ± 84.7 | -5.9% | 1196.4 ± 112.5 | +15.7% | 5/5/5/5 |  |
| EpiGNN | influenza_japan | 5 | 1031 | 1273.1 ± 322.1 | +23.5% | 1168.8 ± 56.9 | -8.2% | 1417.2 ± 111.3 | +21.3% | 1660.8 ± 100.7 | +42.1% | 5/5/5/5 |  |
| EpiGNN | influenza_japan | 10 | 1441 | 1701.2 ± 98.9 | +18.1% | 1469.9 ± 108.1 | -13.6% | 1977.5 ± 32.5 | +34.5% | 2155.8 ± 7.4 | +46.7% | 5/5/5/5 |  |
| EpiGNN | influenza_japan | 15 | 1470 | 1598.0 ± 201.7 | +8.7% | 1427.3 ± 171.1 | -10.7% | 1980.3 ± 45.8 | +38.7% | 2148.4 ± 5.9 | +50.5% | 5/5/5/5 |  |
| EpiGNN | influenza_us-regions | 3 | 589 | 632.4 ± 36.1 | +7.4% | 766.3 ± 114.3 | +21.2% | 740.3 ± 39.0 | -3.4% | 927.4 ± 38.2 | +21.0% | 5/5/5/5 |  |
| EpiGNN | influenza_us-regions | 5 | 774 | 893.8 ± 55.4 | +15.5% | 906.5 ± 98.8 | +1.4% | 965.3 ± 60.3 | +6.5% | 1162.6 ± 60.7 | +28.2% | 5/5/5/5 |  |
| EpiGNN | influenza_us-regions | 10 | 984 | 1053.3 ± 47.4 | +7.0% | 973.7 ± 55.4 | -7.6% | 1217.7 ± 103.1 | +25.1% | 1398.0 ± 48.0 | +43.6% | 5/5/5/5 |  |
| EpiGNN | influenza_us-regions | 15 | 1061 | 1107.4 ± 96.7 | +4.4% | 995.3 ± 93.9 | -10.1% | 1324.1 ± 68.8 | +33.0% | 1497.0 ± 35.5 | +50.4% | 5/5/5/5 |  |
| EpiGNN | influenza_us-states | 3 | 160 | 167.8 ± 7.2 | +4.9% | 187.3 ± 8.1 | +11.6% | 180.3 ± 5.9 | -3.7% | 199.7 ± 5.0 | +6.6% | 5/5/5/5 |  |
| EpiGNN | influenza_us-states | 5 | 186 | 195.9 ± 11.9 | +5.3% | 221.7 ± 9.2 | +13.1% | 217.9 ± 9.2 | -1.7% | 242.3 ± 8.3 | +9.3% | 5/5/5/5 |  |
| EpiGNN | influenza_us-states | 10 | 220 | 225.8 ± 8.2 | +2.6% | 240.4 ± 5.1 | +6.4% | 259.3 ± 5.2 | +7.9% | 287.6 ± 5.4 | +19.6% | 5/5/5/5 |  |
| EpiGNN | influenza_us-states | 15 | 236 | 237.4 ± 7.2 | +0.6% | 244.1 ± 8.0 | +2.8% | 260.9 ± 8.8 | +6.9% | 307.2 ± 4.7 | +25.8% | 5/5/5/5 |  |
| HeatGNN | influenza_japan | 3 | — | 1164.4 ± 184.8 | — | 1034.3 ± 69.1 | -11.2% | 973.2 ± 84.7 | -5.9% | 1196.4 ± 112.5 | +15.7% | 5/5/5/5 |  |
| HeatGNN | influenza_japan | 5 | 1378 | 1155.3 ± 158.5 | -16.2% | 1168.8 ± 56.9 | +1.2% | 1417.2 ± 111.3 | +21.3% | 1660.8 ± 100.7 | +42.1% | 5/5/5/5 |  |
| HeatGNN | influenza_japan | 10 | — | 1694.5 ± 126.5 | — | 1469.9 ± 108.1 | -13.3% | 1977.5 ± 32.5 | +34.5% | 2155.8 ± 7.4 | +46.7% | 5/5/5/5 |  |
| HeatGNN | influenza_japan | 15 | — | 1518.5 ± 145.0 | — | 1427.3 ± 171.1 | -6.0% | 1980.3 ± 45.8 | +38.7% | 2148.4 ± 5.9 | +50.5% | 5/5/5/5 |  |
| HeatGNN | influenza_us-regions | 3 | — | 672.5 ± 36.5 | — | 766.3 ± 114.3 | +14.0% | 740.3 ± 39.0 | -3.4% | 927.4 ± 38.2 | +21.0% | 5/5/5/5 |  |
| HeatGNN | influenza_us-regions | 5 | 852 | 915.4 ± 63.2 | +7.4% | 906.5 ± 98.8 | -1.0% | 965.3 ± 60.3 | +6.5% | 1162.6 ± 60.7 | +28.2% | 5/5/5/5 |  |
| HeatGNN | influenza_us-regions | 10 | — | 1148.5 ± 66.6 | — | 973.7 ± 55.4 | -15.2% | 1217.7 ± 103.1 | +25.1% | 1398.0 ± 48.0 | +43.6% | 5/5/5/5 |  |
| HeatGNN | influenza_us-regions | 15 | — | 1174.5 ± 68.3 | — | 995.3 ± 93.9 | -15.3% | 1324.1 ± 68.8 | +33.0% | 1497.0 ± 35.5 | +50.4% | 5/5/5/5 |  |
| HeatGNN | influenza_us-states | 3 | — | 166.7 ± 13.5 | — | 187.3 ± 8.1 | +12.4% | 180.3 ± 5.9 | -3.7% | 199.7 ± 5.0 | +6.6% | 5/5/5/5 |  |
| HeatGNN | influenza_us-states | 5 | 186 | 208.0 ± 15.8 | +11.8% | 221.7 ± 9.2 | +6.6% | 217.9 ± 9.2 | -1.7% | 242.3 ± 8.3 | +9.3% | 5/5/5/5 |  |
| HeatGNN | influenza_us-states | 10 | — | 238.1 ± 16.1 | — | 240.4 ± 5.1 | +0.9% | 259.3 ± 5.2 | +7.9% | 287.6 ± 5.4 | +19.6% | 5/5/5/5 |  |
| HeatGNN | influenza_us-states | 15 | — | 256.0 ± 19.7 | — | 244.1 ± 8.0 | -4.6% | 260.9 ± 8.8 | +6.9% | 307.2 ± 4.7 | +25.8% | 5/5/5/5 |  |
| MTGNN | dengue | 3 | — | 406.4 ± 13.2 | — | 310.0 ± 21.0 | -23.7% | 435.1 ± 18.4 | +40.4% | 475.2 ± 1.5 | +53.3% | 5/5/5/5 | both sides on the 2,392-node subsample |
| MTGNN | dengue | 5 | — | 490.9 ± 0.0 | — | 357.6 ± 24.4 | -27.1% | 467.4 ± 5.6 | +30.7% | 481.6 ± 1.8 | +34.7% | 5/5/5/5 | **5/5 constant**; both sides on the 2,392-node subsample |
| MTGNN | dengue | 10 | — | 487.7 ± 0.0 | — | 444.3 ± 18.7 | -8.9% | 482.3 ± 1.4 | +8.6% | 483.7 ± 0.5 | +8.9% | 5/5/5/5 | **5/5 constant**; both sides on the 2,392-node subsample |
| MTGNN | dengue | 15 | — | 484.4 ± 0.0 | — | 474.3 ± 1.9 | -2.1% | 482.5 ± 0.2 | +1.7% | 480.9 ± 0.2 | +1.4% | 5/5/5/5 | **5/5 constant**; both sides on the 2,392-node subsample |
| MTGNN | influenza_japan | 3 | — | 1662.8 ± 0.0 | — | 1034.3 ± 69.1 | -37.8% | 973.2 ± 84.7 | -5.9% | 1196.4 ± 112.5 | +15.7% | 5/5/5/5 | **4/5 constant** |
| MTGNN | influenza_japan | 5 | — | 1882.2 ± 0.0 | — | 1168.8 ± 56.9 | -37.9% | 1417.2 ± 111.3 | +21.3% | 1660.8 ± 100.7 | +42.1% | 5/5/5/5 | **4/5 constant** |
| MTGNN | influenza_japan | 10 | — | 2118.5 ± 0.0 | — | 1469.9 ± 108.1 | -30.6% | 1977.5 ± 32.5 | +34.5% | 2155.8 ± 7.4 | +46.7% | 5/5/5/5 | **4/5 constant** |
| MTGNN | influenza_japan | 15 | — | 2068.4 ± 0.0 | — | 1427.3 ± 171.1 | -31.0% | 1980.3 ± 45.8 | +38.7% | 2148.4 ± 5.9 | +50.5% | 5/5/5/5 | **4/5 constant** |
| MTGNN | influenza_us-regions | 3 | — | 1561.2 ± 59.2 | — | 766.3 ± 114.3 | -50.9% | 740.3 ± 39.0 | -3.4% | 927.4 ± 38.2 | +21.0% | 5/5/5/5 | **1/5 constant** |
| MTGNN | influenza_us-regions | 5 | — | 1564.8 ± 13.7 | — | 906.5 ± 98.8 | -42.1% | 965.3 ± 60.3 | +6.5% | 1162.6 ± 60.7 | +28.2% | 5/5/5/5 | **1/5 constant** |
| MTGNN | influenza_us-regions | 10 | — | 1562.8 ± 14.0 | — | 973.7 ± 55.4 | -37.7% | 1217.7 ± 103.1 | +25.1% | 1398.0 ± 48.0 | +43.6% | 5/5/5/5 | **1/5 constant** |
| MTGNN | influenza_us-regions | 15 | — | 1541.4 ± 41.9 | — | 995.3 ± 93.9 | -35.4% | 1324.1 ± 68.8 | +33.0% | 1497.0 ± 35.5 | +50.4% | 5/5/5/5 | **1/5 constant** |
| MTGNN | influenza_us-states | 3 | — | 474.1 ± 11.9 | — | 187.3 ± 8.1 | -60.5% | 180.3 ± 5.9 | -3.7% | 199.7 ± 5.0 | +6.6% | 5/5/5/5 | **3/5 constant** |
| MTGNN | influenza_us-states | 5 | — | 465.9 ± 1.4 | — | 221.7 ± 9.2 | -52.4% | 217.9 ± 9.2 | -1.7% | 242.3 ± 8.3 | +9.3% | 5/5/5/5 | **3/5 constant** |
| MTGNN | influenza_us-states | 10 | — | 449.7 ± 14.2 | — | 240.4 ± 5.1 | -46.5% | 259.3 ± 5.2 | +7.9% | 287.6 ± 5.4 | +19.6% | 5/5/5/5 | **3/5 constant** |
| MTGNN | influenza_us-states | 15 | — | 447.1 ± 8.4 | — | 244.1 ± 8.0 | -45.4% | 260.9 ± 8.8 | +6.9% | 307.2 ± 4.7 | +25.8% | 5/5/5/5 | **3/5 constant** |
| (none) | covid_us-states | 3 | — | no run | — | 8483.5 ± 1251.5 | — | 12785.7 ± 3924.4 | +50.7% | 19671.7 ± 7622.2 | +131.9% | 0/5/5/5 | **no baseline exists for COVID** |
| (none) | covid_us-states | 5 | — | no run | — | 13725.8 ± 1204.0 | — | 14073.0 ± 1513.3 | +2.5% | 26916.2 ± 5629.0 | +96.1% | 0/5/5/5 | **no baseline exists for COVID** |
| (none) | covid_us-states | 10 | — | no run | — | 19091.1 ± 3014.9 | — | 18401.4 ± 4654.6 | -3.6% | 58142.3 ± 19740.2 | +204.6% | 0/5/5/5 | **no baseline exists for COVID** |
| (none) | covid_us-states | 15 | — | no run | — | 17820.4 ± 2312.4 | — | 21601.4 ± 2592.2 | +21.2% | 33702.9 ± 4497.6 | +89.1% | 0/5/5/5 | **no baseline exists for COVID** |

The `seeds` column is baseline seeds, then encoder seeds, then transfer few-shot, then transfer
zero-shot. Every cell is 5/5/5/5 except the four COVID rows, which are 0/5/5/5 because no baseline
was ever run on COVID.

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

## What columns 4 and 5 say about transfer

There are 16 cells with a baseline, four panels by four horizons, plus four COVID cells with none.

**Against our own single-disease ceiling.** Few-shot transfer is better in **4** of the 16 and worse
in **12**, spanning -5.9 to +40.4 percent. Zero-shot is worse in **16** of 16, spanning +1.4 to
+53.3 percent. On the four COVID cells few-shot is better in 1 of 4 and zero-shot worse in 4 of 4,
by as much as +204.6 percent. Holding the disease out costs error everywhere except a handful of
short-horizon cells, and holding it out with no adaptation at all costs error in every single cell
we can measure.

**The cost grows with horizon on the influenza panels.** Few-shot goes -5.9 to +38.7 percent on
Japan, -3.4 to +33.0 on US-Regions and -3.7 to +6.9 on US-States as the horizon goes 3 to 15. Dengue
runs the other way, +40.4 down to +1.7, and that is not transfer improving. See the floors below.

**Against the baselines, and this is the part that is easy to misread.** Few-shot transfer beats the
best baseline in the cell in **5** of 16, zero-shot in **3** of 16. Four of those five few-shot cells
are dengue and all three zero-shot cells are dengue. That is not a like-for-like win and it is not
evidence of transfer skill, for two separate reasons:

1. EpiGNN, Cola-GNN and HeatGNN train on the disease they are scored on and cannot be run without
   it. The transfer arms never see it. Comparing them measures two different tasks.
2. On dengue at the long horizons everything in the comparison, ours included, is at or below the
   level of a flat forecast. Pooled on the same 2,392 nodes, a per-node train-mean forecast scores
   **484.5 / 483.4 / 480.2 / 477.0** at h3/h5/h10/h15. Our few-shot transfer scores 435.1 / 467.4 /
   482.3 / 482.5, so at h10 and h15 it is WORSE than the flat floor, and so is EpiGNN at 500.1 and
   504.7. The "we beat the baseline on dengue h10 and h15" cells are two models that both lose to
   predicting the training mean. Only h3 and h5 clear that floor.

**And no dengue transfer cell clears both floors at once.** Persistence pooled on the same nodes is
234.7 / 350.1 / 560.7 / 645.9. Few-shot transfer beats train-mean at h3 and h5 but loses to
persistence there, and beats persistence at h10 and h15 but loses to train-mean there. There is no
horizon on dengue where cross-disease transfer beats both naive floors.

For context, the same floors already sink the single-disease runs on Japan, where a seasonal-naive
forecast beats every model in this table at every horizon. That is a known structural limit recorded
in `progress/outcomes/results_summary.txt` lines 30 to 35: a 20-week lookback cannot reach last
year's amplitude.

**Two figures carry this.** `figures/baselines_vs_single.png` is the 16 cells with the three
baselines against our single-disease encoder. `figures/transfer_vs_baselines.png` adds both transfer
arms with the baselines collapsed to the best one per cell, which steel-mans them. Both normalise
each row so our single-disease encoder is 1.0, because absolute RMSE spans an order of magnitude
across panels. The transfer bars are hatched and in a separate colour family, and the chart says on
its face that they never see the target disease, so the figure cannot be lifted out of this document
and read as three models racing on equal terms. COVID is in neither figure because it has no
baseline. Ebola is in neither figure because no node-indexed baseline can run on it at all.

---

## Open items

1. **The dengue node-set correction is new as of 2026-09-06.** Any earlier document quoting an
   encoder-versus-baseline dengue margin near 40 or 50 percent is quoting the uncorrected figure and
   needs the number from this table instead.
2. **US-States losses are undisclosed.** No current document says we lose to EpiGNN on US-States at
   every horizon. It belongs in the manuscript's results section, not only here.
3. `paper_compare.py` computes a reproduced PCC per cell and never prints it. Harmless, but it is
   the source of a `Mean of empty slice` warning on the degenerate MTGNN cells.

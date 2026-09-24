# Shuffled-adjacency retrain (ledger D2)

**Run finished 2026-09-23. Scored from disk the same day.** This is the last spatial control for the
paper. It asks a question the two earlier controls could not: does training on the real district map
buy anything over training on a fake one?

## What was run

Retrain the encoder from scratch on a graph whose node labels are permuted. The permutation keeps the
topology and the degree sequence exactly, so the model sees a graph with the same connectivity and the
same number of neighbours per node, but the wrong districts. Node histories, targets and masks stay in
their original order, so each node keeps its own data and is handed a stranger's neighbours. Everything
else is identical to the released single-disease runs: same trainer, same seeds, same 80 epochs, same
early stopping, same scoring. The only moved variable is who each node's neighbours are.

The comparison is paired per seed against the real-graph runs in `results/single/`. The rule for
"within noise" is the same one the gate-off ablation uses: a cell counts as a real effect only when
the mean paired difference across the five seeds is larger than its own spread across those seeds.

Code: `ablation/run_shuffle_adjacency.py`, training hook `train.loop.permute_adjacency`. Records:
`ablation/single/encoder__<panel>__seed<S>__shufadj.json`, 25 files (5 panels x 5 seeds).

## Completeness

All 25 records present and fresh (written 2026-09-22 23:59 to 2026-09-23 09:40). Every record carries
the shuffle metadata: the permutation seed (20260921 + training seed), a permutation hash, and the
displaced fraction, which ranges from 0.70 to 1.00 across runs, so no run was a silent no-op. Twenty
of the twenty-five permutation hashes are distinct. The five collisions are benign: influenza-US-states
and COVID-US-states have the same node count (49), so the same training seed produces the same
permutation index vector, applied to two different graphs. Each remains a genuine relabel of its own
panel's graph.

## Verdict over 60 cells: 9 the real graph helps, 2 the real graph hurts, 49 within noise

Sixty cells is 5 panels x 4 horizons x 3 metrics (RMSE, MAE, PCC). Scrambling the districts changes
nothing in 49 of them. The eleven cells that move:

| panel | metric | h | real graph | shuffled A | paired delta (shuf - real) | sd | verdict |
|---|---|---|---|---|---|---|---|
| influenza_japan | rmse | 10 | 1063.017 | 993.958 | -69.0594 | 41.4358 | real graph HURTS |
| influenza_japan | mae | 10 | 444.963 | 419.454 | -25.5089 | 15.6569 | real graph HURTS |
| influenza_us-states | pcc | 3 | 0.808 | 0.789 | -0.0185 | 0.0125 | real graph HELPS |
| dengue | rmse | 10 | 56.212 | 58.524 | 2.3115 | 1.9670 | real graph HELPS |
| dengue | rmse | 15 | 57.381 | 58.652 | 1.2701 | 0.7252 | real graph HELPS |
| dengue | mae | 10 | 26.609 | 27.553 | 0.9440 | 0.8151 | real graph HELPS |
| dengue | mae | 15 | 27.473 | 28.255 | 0.7821 | 0.6787 | real graph HELPS |
| dengue | pcc | 3 | 0.422 | 0.386 | -0.0363 | 0.0146 | real graph HELPS |
| dengue | pcc | 5 | 0.335 | 0.293 | -0.0420 | 0.0065 | real graph HELPS |
| dengue | pcc | 10 | 0.197 | 0.157 | -0.0399 | 0.0227 | real graph HELPS |
| dengue | pcc | 15 | 0.106 | 0.072 | -0.0336 | 0.0183 | real graph HELPS |

For RMSE and MAE a negative delta means the shuffled model had the lower error, so the real graph was
not helping. For PCC a negative delta means the shuffled model had the lower correlation, so the real
graph was helping.

## What this says

The strong form of the claim does not survive the disk. Training on the wrong districts does NOT
perfectly match training on the right ones, so I cannot say the graph contributed nothing at any stage.

The honest reading is narrower and it lines up with the gate-off finding. The real district map earns a
small, seed-stable advantage, and it is almost entirely on dengue, our densest graph and the one with
the highest learned gate. There it helps correlation at every horizon (PCC deltas 0.034 to 0.042, each
several seed-spreads clear of noise) and it helps error only at the two long horizons (dengue RMSE
+2.31 at h10 and +1.27 at h15, MAE +0.94 and +0.78), by roughly 2 to 4 percent. On the four smaller
panels the real map buys nothing on error: every RMSE and MAE cell there is within noise, and on
influenza-Japan at h10 the scrambled model is actually the better one. Only one small-panel correlation
cell moves (influenza-US-states PCC at h3, by 0.019).

So the picture from the earlier controls holds and is now sharper. The graph is about shape, not
magnitude. The gate-off ablation said the spatial channel helps correlation in a handful of cells and
never lowers error; the inference-only relabel said scrambling the graph on an already-trained model
costs under 1 percent; and this retrain says that even when the model is allowed to LEARN on the real
map from scratch, the map only earns a small correlation advantage plus a few percent on dengue's long
horizons. The mechanism chain survives with one honest edit: the data does carry district structure,
the encoder compresses most of it away, mean aggregation deletes most of the rest, and training barely
needs the real map. "Barely," not "never," and dengue is the measured exception.

## Scope limits

- The permutation preserves the degree sequence, and the LTR degree feature is permutation-invariant,
  so this arm isolates the value of neighbour IDENTITY, not of the graph as a whole. Same ceiling the
  gate-off arm carries.
- These are point means over five seeds with the within-noise rule the gate-off ablation uses. They
  are not bootstrap intervals, and no multiplicity correction is applied across the 60 cells.
- Dengue's advantage is small in absolute terms (2 to 4 percent on error, 0.03 to 0.04 on correlation).
  It is real across seeds but it does not rescue the graph as an accuracy component.
- The two influenza-Japan cells where the shuffled model wins on error at h10 are flagged as "real
  graph hurts." Read them as the real map carrying no advantage there, not as a strong claim that the
  wrong graph is better.

## Reproduce

```
conda run -n ebola-train python ablation/run_shuffle_adjacency.py --report
python diagnostics/verify_shufadj_doc.py
python diagnostics/verify_shufadj_doc.py --mutate
```

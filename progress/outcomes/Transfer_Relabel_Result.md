# Transfer relabel test A: result

Date: 2026-10-05. Protocol: `progress/decisions/Transfer_Relabel_Protocol.md`, frozen in commit
`6339393`, sha256 `5a8d4e087483c58fbd3abc76976e9a7197cf31abe18d979d4a72cef3150883ae`. Runner:
`experiments/transfer_relabel/run.py`. Verifier: `diagnostics/verify_transfer_relabel.py`. Records:
`experiments/transfer_relabel/ldo3__<fold>__seed<S>.json` (15 primary) and
`ldo3full_zeroshot__<fold>__seed42.json` (2 exploratory), plus `summary.json`. Lane: EXPLORATORY. The
records and this file are not committed yet.

The question: when a trunk trained on other diseases forecasts a disease it never trained on, does it
use which districts neighbour which? I relabelled the map among districts with the same number of
neighbours, at inference, 20 times per checkpoint, and measured how much RMSE and MAE moved.

## 1. Outcome: O2, no evidence of use

The protocol's O2 sentence, with only the bracketed slots filled in from `--report`:

> No held-out panel showed a rise in error of 2 percent or more when the map was relabelled. On
> influenza_japan and influenza_us-states the change was bounded inside plus or minus 2 percent; on
> dengue and covid_us-states the five-seed interval was too wide to decide. On influenza_us-regions
> the relabel changed too few neighbours to test.

The mandatory sentence, appended under every outcome:

> This is an inference-only test of trunks that were trained on the real map, using the head fitted
> on the held-out disease. It does not show whether a trunk trained on a wrong map would do as well.

"Showed no rise of 2 percent" refers to the five-seed means. The largest seed mean is +1.37 percent
(dengue h3 RMSE), but its interval reaches +3.92 percent, so a 2 percent cost there is not ruled out.
That is why dengue is INCONCLUSIVE rather than NOT USED.

## 2. What it means in plain words

Across 40 deciding cells (5 panels, 4 horizons, RMSE and MAE), the five-seed mean change from
scrambling the map runs from -0.78 to +1.37 percent. No cell reaches the 2 percent bar in either
direction. 36 cells are NOT USED, meaning the whole 95 percent interval sits inside plus or minus 2
percent. 4 are INCONCLUSIVE: dengue h3 (RMSE and MAE) and covid_us-states h15 (RMSE and MAE). So the
transferred trunk leans on the real map about as little as the single-disease models do. That fits
the earlier gate-off finding that the spatial channel helps error in 0 of 40 cells (CLAUDE.md section
4; I did not re-verify it for this document).

Two cells have an interval that excludes zero and are still NOT USED, because the effect is real but
under the 2 percent bar. influenza_japan h5 MAE is +0.32 percent [+0.10, +0.54], so the real map
helps by a third of a percent. covid_us-states h10 RMSE is -0.53 percent [-1.02, -0.05], so the
relabelled map does slightly better. One goes each way.

## 3. Per panel

"Strength" is the share of a typical district's neighbours that the deciding relabel actually
changed, averaged over the 20 draws. The protocol's gate is 0.50. The `global` column is an
unconstrained relabel that also changes the neighbour count the mixer sees. It is printed beside the
deciding one, and **it never decides**.

| panel | map seen by the trunk? (protocol label / what I found, section 4) | strength | verdict | largest deciding cell, mean [95% interval] | global, same panel, largest cell (never decides) |
|---|---|---|---|---|---|
| dengue | no / 47 of 7,165 nodes seen (the Japan island) | 0.999 | INCONCLUSIVE | h3 RMSE +1.37 [-1.17, +3.92] | h3 RMSE +2.10 [-0.98, +5.19], INCONCLUSIVE |
| influenza_japan | no / **yes, all 86 edges, inside dengue's map** | 0.833 | NOT USED | h3 RMSE -0.47 [-1.39, +0.45] | h15 MAE -0.61 [-1.38, +0.15], NOT USED |
| influenza_us-regions | no / no | 0.414 | WEAK | h10 MAE +0.22 [-0.52, +0.96] | h3 MAE -0.46 [-2.00, +1.07], INCONCLUSIVE |
| influenza_us-states | yes, bit-identical via covid / yes | 0.864 | NOT USED | h10 MAE -0.36 [-0.72, +0.01] | h3 RMSE +0.34 [-0.61, +1.30], NOT USED |
| covid_us-states | yes, bit-identical via influenza_us-states / yes | 0.864 | INCONCLUSIVE | h15 RMSE -0.78 [-3.71, +2.15] | h15 RMSE -2.22 [-4.52, +0.09], INCONCLUSIVE |

All values are percent change in error with the map relabelled, seed mean over 5 seeds. Positive means
the relabelled map forecast worse, so the real map was helping. influenza_us-regions has all 8 cells
NOT USED, and the strength gate turns that into WEAK as the protocol declared in advance. The full 40
cells are in the appendix.

Under `global`, covid_us-states h3 has intervals that clear zero on the worse side: RMSE +1.16
[+0.03, +2.28] and MAE +1.15 [+0.11, +2.18], both INCONCLUSIVE. `degclass` keeps the neighbour count
and shows +0.45 and +0.27 at the same cells. So the small h3 cost under `global` may come from the
changed neighbour count, not from the changed neighbours. This is a non-deciding row and I report it
as a lead only.

## 4. A correction to the protocol: influenza_japan's map was not new to the trunk

I wrote the protocol, and I labelled influenza_japan as "map not seen by the trunk". That was wrong.
I checked it on disk after the run, using `bundles.load` and `japan_node_map.csv`. The dengue panel
contains all 47 Japanese prefectures as a separate island, with no edges to any non-Japan dengue
node. When I match prefectures by name, that island contains **all 86** of influenza_japan's edges,
plus 8 sea links that influenza_japan does not have: Hokkaido to Aomori, Akita, Iwate and Miyagi,
and Okinawa to Kagoshima, Kumamoto, Miyazaki and Nagasaki. The influenza-fold trunk trained on
dengue, so it trained on Japan's whole map plus 8 edges. The protocol only checked for a
bit-identical `A_geo`, and that is why it missed this. The audit already lists "dengue contains the
same 47 Japanese prefectures as influenza_japan" (`Reports/Phase0_to_Now_Audit.md`, unsafe-to-claim
list).

What this changes. It does not change the outcome code. Only the O3 sentence reads the seen or unseen
label, and the outcome is O2. It does change what the result says about Ebola's situation, a map no
trunk has seen. Of the three panels declared unseen, influenza_us-regions is WEAK and influenza_japan
was in fact seen. That leaves dengue as the only cleanly unseen panel that passed the strength gate,
and dengue is INCONCLUSIVE. The dengue fold's trunk also saw dengue's Japan island through
influenza_japan, but that is 47 of 7,165 nodes.

Wording in the paper:

What you get: describing influenza_japan as "map seen through dengue" is accurate and survives a
reviewer who reads the data section.
What it costs: the test then has no confident verdict on any panel whose map is new to the trunk.
The NOT USED results are all on maps the trunk had seen.
My call: use the corrected label, because the other version is a claim the data contradicts.
Do not write "the transferred trunk ignores the map even on unseen geography".

## 5. Things that look odd

- **Dengue h3 comes from one seed.** Per seed, the h3 RMSE change is +4.57, +1.54, -1.05, +0.52 and
  +1.29 percent (seeds 42, 52, 62, 72, 82). Seed 42 alone is above 2 percent, and its spread over the
  20 draws is also the largest, 2.31 percent against 0.30 to 1.11 for the others. With one seed out
  of five, the five-seed interval cannot decide. I do not know why seed 42 is more map-sensitive.
- **Dengue h15 barely moves at all, and that says more about the forecast than about the map.** All
  five trunks score dengue h15 RMSE between 57.96 and 58.25, a seed spread of 0.18 percent. Relabels
  move each trunk's RMSE by -0.05 to +0.11 percent. A forecast that hardly changes with the seed or the map
  looks close to flat at that horizon. If so, NOT USED at dengue h15 is close to automatic. I did not
  test the flatness directly.
- **covid_us-states goes the wrong way at long horizons.** All four of its h10 and h15 cell means
  are negative, so the relabelled map does slightly better, while all four h3 and h5 means are
  positive. h10 RMSE clears zero (section 2). The h15 cells are wide, [-3.71, +2.15] for RMSE.
  Nothing here reaches HURTS.
- **The one saved zero-shot covid head goes clearly the wrong way.** See section 6.

## 6. Zero-shot rows, descriptive only, never a verdict

There are two saved zero-shot heads, one seed each (42), from the `ldo3full` trunk family, with early
stop disabled. That is a different trunk family from the primary, so these rows are never read
against it. The table shows the percent change from 20 `degclass` relabels: mean over draws, spread
(sd) over draws, and the lowest and highest draw. A spread over relabels is not an interval over
trunks.

| panel | cell | draw mean | draw sd | lowest draw | highest draw |
|---|---|---|---|---|---|
| covid_us-states | h3 RMSE | -0.80 | 0.62 | -1.83 | +0.90 |
| covid_us-states | h3 MAE | -0.73 | 0.72 | -1.85 | +1.10 |
| covid_us-states | h5 RMSE | -2.08 | 0.66 | -3.57 | -0.67 |
| covid_us-states | h5 MAE | -2.02 | 0.67 | -3.29 | -0.57 |
| covid_us-states | h10 RMSE | -1.02 | 0.38 | -2.00 | -0.20 |
| covid_us-states | h10 MAE | -1.32 | 0.37 | -2.29 | -0.45 |
| covid_us-states | h15 RMSE | -0.67 | 0.55 | -1.49 | +0.59 |
| covid_us-states | h15 MAE | -0.61 | 0.54 | -1.42 | +0.71 |
| dengue | h3 RMSE | +1.37 | 0.46 | +0.46 | +2.05 |
| dengue | h3 MAE | +0.87 | 0.29 | +0.24 | +1.25 |
| dengue | h5 RMSE | +0.49 | 0.29 | -0.13 | +0.94 |
| dengue | h5 MAE | -0.44 | 0.30 | -1.01 | +0.15 |
| dengue | h10 RMSE | -0.12 | 0.09 | -0.24 | +0.06 |
| dengue | h10 MAE | -0.54 | 0.17 | -0.80 | -0.19 |
| dengue | h15 RMSE | -0.27 | 0.14 | -0.54 | -0.03 |
| dengue | h15 MAE | -0.44 | 0.22 | -0.77 | -0.04 |

On the covid zero-shot head, all 20 relabels forecast better than the real map at h5 and at h10, by
about 2 percent at h5. That is one trunk and one seed, on a map the trunk had seen, so it is a lead
and not a finding. Do not quote it as "the map hurts zero-shot transfer". The dengue zero-shot h3
RMSE mean of +1.37 matches the primary dengue h3 mean to two decimals. That is a coincidence: the
values are +1.370 and +1.374, one is a spread over relabels of one trunk and the other a mean over
five trunks. It does not show the two families agree.

## 7. Compared with the single-disease probe

The single-disease probe (`diagnostics/graph_probe/t2.py`, `t_dengue.py`) found that relabelling
costs +0.04 to +1.04 percent (`Session_Audit_2026-09-10.md:225-228`; I quote it and did not re-run it
for this document). This test finds -0.78 to +1.37 percent across its 40 deciding cell means. The
transferred trunk is no more dependent on its map than the single-disease models were.

They are not like-for-like, for four reasons:
1. The probe's relabel was unconstrained, so each district got another district's degree feature as
   well as other neighbours (`t2.py:52-54` with the plain forward). Here the deciding relabel keeps
   every district's degree.
2. The probe scored scaled model-space RMSE pooled over all horizons and test cells (`t2.py:17-30`).
   This test scores count-space `country_macro` RMSE and MAE per horizon.
3. The probe used `results/single/` checkpoints trained on the scored disease. These are transfer
   trunks with an adapted head.
4. The probe's dengue figure is one seed.

## 8. Scope fences

- **Inference only.** Every trunk was trained on the real map. A null says the trained trunk does not
  lean on the map at forecast time. It does not show that a trunk trained on a wrong map would do as
  well. The D2 retrain found a dengue real-map advantage of about 2 to 4 percent at h10 and h15 that
  the inference-only probe missed (`progress/planning/Gap_Ledger.md`, D2). So a null here does not
  rule out a retrain effect.
- **Adapted head only** in the deciding family. Each checkpoint stores the adapter fitted on the
  held-out disease with the trunk frozen. The zero-shot head of these 15 runs was never saved.
  Nothing here supports a statement about zero-shot transfer.
- **ANIL is excluded.** The ANIL checkpoints store the meta-learned starting head, not the adapter
  that `meta_test` fitted and scored. On `ldo3covid` seed 42 the stored head's median forecast
  differs from the archived ANIL forecast by a median 54 to 98 percent. So no ANIL checkpoint can pass
  the drift gate without refitting, and refitting is training. Nothing here covers meta-learned
  trunks.
- **The influenza fold's three panels share one trunk per seed**, so they are not three independent
  pieces of evidence.
- **Not Ebola.** No sentence here is about Ebola. Section 4 explains why this test is further from
  Ebola's situation than the protocol assumed.
- **The verdicts depend on the 20 fixed relabels** (permutation seeds 7001 to 7020).
- **The 2 percent bar is coarse where seed noise is tiny.** At dengue h15 the seed spread is 0.18
  percent.

## 9. Checks run

- **Records complete.** 17 records: 15 primary (3 folds by 5 seeds) and 2 exploratory, each with 20
  `degclass` and 20 `global` draws per panel. All 17 carry the frozen protocol sha256. Drift gate:
  0.0 on metrics and 0.0 on quantiles for every panel of every record. The protocol file hashes to the
  frozen value on disk, and `git diff` shows no local edits to the protocol, runner or verifier. All
  ten code stamps in the records match the current files under the runner's line-ending-normalised
  hash.
- **Checkpoint stamps.** `checkpoint_sha256` uses the same line-ending-normalised hash, applied to a
  binary `.pt` file. So it matches all 17 checkpoints under that function, but it is **not** the
  value `sha256sum` prints, because every checkpoint contains 2 to 7 CRLF byte pairs. It still pins
  the file. Anyone checking it by hand must use the runner's `sha256()`. The verifier does not check
  checkpoint stamps.
- **`run.py --report`:** outcome O2. Re-running it left `summary.json` unchanged.
- **Verifier:** PASS, 17 records, O2, panel verdicts as in section 3. `--mutate`: 11 of 11
  corruptions caught, PASS. `--synthetic`: mixed and null built sets, runner and verifier agree with
  the built verdicts (O5 and O1), 11 of 11 mutations caught in each, PASS.
- **Independent recompute.** I wrote a short script from the protocol text alone, with no import of
  the runner or verifier, and recomputed every cell from the raw draws. It agrees with `summary.json`
  on all 483 values (cell means, interval ends, verdicts, panel verdicts and strengths, outcome,
  exploratory means and spreads) to within 2.1e-17. It agrees with the `--report` text on all 347
  printed values. It first disagreed on the 40 PCC cells by sign only. The runner reports PCC as real
  minus relabelled, so that "+" means "relabelled map worse", and my script had it the other way.
  With the sign aligned there are 0 mismatches. To show the check can fail, I made one seed's
  influenza_japan h10 MAE draws 3 percent worse in a copy of the records. The comparison flagged
  exactly that cell, and Japan flipped from NOT USED to INCONCLUSIVE.
- **This document.** I parsed the numbers back out of this file and checked them against the records.
  The count is in the hand-off note, not here, so this file does not cite a check of itself.

## 10. Reproduce

From the repo root (`conda` is not on the Git Bash PATH, so run these from PowerShell):

```
conda run --no-capture-output -n ebola-train python experiments/transfer_relabel/run.py --report
conda run --no-capture-output -n ebola-train python diagnostics/verify_transfer_relabel.py
conda run --no-capture-output -n ebola-train python diagnostics/verify_transfer_relabel.py --mutate
conda run --no-capture-output -n ebola-train python diagnostics/verify_transfer_relabel.py --synthetic
```

The scored run itself took about 95 minutes and must not be repeated for this result. It was:
`conda run --no-capture-output -n ebola-train python experiments/transfer_relabel/run.py --sha256 5a8d4e087483c58fbd3abc76976e9a7197cf31abe18d979d4a72cef3150883ae`.

## Appendix: all 40 deciding cells

Percent change in error with the map relabelled, seed mean over 5 seeds, 95 percent t-interval
(t = 2.7764). The PCC column is the absolute drop in correlation, real minus relabelled, printed only.

| panel | h | metric | degclass mean | 95% interval | verdict | global mean | global verdict | PCC drop, degclass |
|---|---|---|---|---|---|---|---|---|
| dengue | 3 | RMSE | +1.37 | [-1.17, +3.92] | INCONCLUSIVE | +2.10 | INCONCLUSIVE | +0.0124 |
| dengue | 3 | MAE | +1.13 | [-1.26, +3.52] | INCONCLUSIVE | +1.68 | INCONCLUSIVE |  |
| dengue | 5 | RMSE | +0.79 | [-0.38, +1.96] | NOT USED | +1.20 | INCONCLUSIVE | +0.0060 |
| dengue | 5 | MAE | +0.74 | [-0.48, +1.95] | NOT USED | +1.14 | INCONCLUSIVE |  |
| dengue | 10 | RMSE | +0.15 | [-0.13, +0.42] | NOT USED | +0.22 | NOT USED | -0.0025 |
| dengue | 10 | MAE | +0.24 | [-0.26, +0.73] | NOT USED | +0.35 | NOT USED |  |
| dengue | 15 | RMSE | +0.05 | [-0.04, +0.13] | NOT USED | +0.06 | NOT USED | +0.0049 |
| dengue | 15 | MAE | +0.06 | [-0.13, +0.25] | NOT USED | +0.09 | NOT USED |  |
| influenza_japan | 3 | RMSE | -0.47 | [-1.39, +0.45] | NOT USED | -0.53 | INCONCLUSIVE | -0.0002 |
| influenza_japan | 3 | MAE | +0.28 | [-0.44, +1.00] | NOT USED | +0.31 | INCONCLUSIVE |  |
| influenza_japan | 5 | RMSE | +0.14 | [-0.39, +0.66] | NOT USED | -0.17 | NOT USED | +0.0020 |
| influenza_japan | 5 | MAE | +0.32 | [+0.10, +0.54] | NOT USED | +0.17 | NOT USED |  |
| influenza_japan | 10 | RMSE | +0.13 | [-0.06, +0.32] | NOT USED | -0.26 | NOT USED | +0.0000 |
| influenza_japan | 10 | MAE | +0.12 | [-0.11, +0.35] | NOT USED | -0.44 | NOT USED |  |
| influenza_japan | 15 | RMSE | -0.09 | [-0.33, +0.15] | NOT USED | -0.47 | NOT USED | -0.0036 |
| influenza_japan | 15 | MAE | -0.07 | [-0.37, +0.23] | NOT USED | -0.61 | NOT USED |  |
| influenza_us-regions | 3 | RMSE | +0.12 | [-1.21, +1.45] | NOT USED | -0.32 | NOT USED | -0.0027 |
| influenza_us-regions | 3 | MAE | +0.17 | [-1.66, +1.99] | NOT USED | -0.46 | INCONCLUSIVE |  |
| influenza_us-regions | 5 | RMSE | +0.05 | [-0.79, +0.90] | NOT USED | -0.28 | NOT USED | -0.0047 |
| influenza_us-regions | 5 | MAE | +0.19 | [-0.87, +1.25] | NOT USED | -0.23 | NOT USED |  |
| influenza_us-regions | 10 | RMSE | +0.04 | [-0.69, +0.77] | NOT USED | -0.05 | NOT USED | -0.0032 |
| influenza_us-regions | 10 | MAE | +0.22 | [-0.52, +0.96] | NOT USED | +0.02 | NOT USED |  |
| influenza_us-regions | 15 | RMSE | -0.01 | [-0.58, +0.56] | NOT USED | -0.04 | NOT USED | +0.0001 |
| influenza_us-regions | 15 | MAE | +0.13 | [-0.42, +0.69] | NOT USED | -0.04 | NOT USED |  |
| influenza_us-states | 3 | RMSE | +0.20 | [-0.47, +0.87] | NOT USED | +0.34 | NOT USED | +0.0008 |
| influenza_us-states | 3 | MAE | +0.09 | [-0.51, +0.68] | NOT USED | +0.18 | NOT USED |  |
| influenza_us-states | 5 | RMSE | -0.18 | [-0.64, +0.27] | NOT USED | +0.16 | NOT USED | -0.0015 |
| influenza_us-states | 5 | MAE | -0.26 | [-0.79, +0.27] | NOT USED | -0.02 | NOT USED |  |
| influenza_us-states | 10 | RMSE | -0.21 | [-0.46, +0.04] | NOT USED | +0.10 | NOT USED | -0.0025 |
| influenza_us-states | 10 | MAE | -0.36 | [-0.72, +0.01] | NOT USED | +0.02 | NOT USED |  |
| influenza_us-states | 15 | RMSE | -0.15 | [-0.47, +0.17] | NOT USED | +0.08 | NOT USED | -0.0011 |
| influenza_us-states | 15 | MAE | -0.09 | [-0.37, +0.20] | NOT USED | +0.27 | NOT USED |  |
| covid_us-states | 3 | RMSE | +0.45 | [-0.27, +1.16] | NOT USED | +1.16 | INCONCLUSIVE | +0.0221 |
| covid_us-states | 3 | MAE | +0.27 | [-0.32, +0.85] | NOT USED | +1.15 | INCONCLUSIVE |  |
| covid_us-states | 5 | RMSE | +0.18 | [-0.33, +0.69] | NOT USED | -0.04 | NOT USED | +0.0069 |
| covid_us-states | 5 | MAE | +0.34 | [-0.21, +0.89] | NOT USED | +0.15 | NOT USED |  |
| covid_us-states | 10 | RMSE | -0.53 | [-1.02, -0.05] | NOT USED | -0.42 | NOT USED | +0.0030 |
| covid_us-states | 10 | MAE | -0.06 | [-0.79, +0.67] | NOT USED | -0.01 | NOT USED |  |
| covid_us-states | 15 | RMSE | -0.78 | [-3.71, +2.15] | INCONCLUSIVE | -2.22 | INCONCLUSIVE | +0.0035 |
| covid_us-states | 15 | MAE | -0.20 | [-2.29, +1.90] | INCONCLUSIVE | -1.24 | INCONCLUSIVE |  |

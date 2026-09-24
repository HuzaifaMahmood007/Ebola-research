# V2 deviation channel: mechanism test protocol

Written 2026-09-24 and revised the same day to the two-stage design the user chose, before any v2
number exists on COVID. The user commits this file before the first COVID run. The runner refuses to
train COVID until this file is committed with no local edits AND a passing repro-check record exists
for this file's hash. Once committed this file is frozen. Any later change goes in a separate
amendment file, dated, and written before the data it affects, so the hash stamped into every COVID
record keeps matching this file.

## 1. The question

Does routing district deviations to the graph make the graph contribute?

This is a mechanism test, not a replacement for the main model. Three measurements set it up:

- The encoder compresses district-specific energy 3 to 5 times between the raw incidence window and
  its output (`progress/outcomes/Input_Energy_2026-09-22.md`; COVID 23.8 percent in, 4.8 percent out).
- Retraining v1 on a relabelled map changes nothing on COVID error: every COVID RMSE and MAE cell is
  within noise (`progress/outcomes/Shuffled_Adjacency_2026-09-23.md`).
- The raw COVID data does carry neighbour information: after removing the shared weekly wave, real
  neighbours remove 5.3 and 2.3 percent of total model-space target variance at h3 and h5
  (`progress/outcomes/Neighbour_Signal_2026-09-24.md`).

The hypothesis: the graph fails because the encoder has squeezed out the district-level signal before
the graph sees it. If so, handing the graph the deviations directly should make the real map earn
accuracy. This protocol fixes, before any data, what result would support that and what would not.

**Note for the manuscript (it records what the wording must respect; it does not edit the paper).**
The earlier COVID graph nulls, the gate-off ablation and the D2 shuffled-adjacency retrain, were
judged at 5 seeds with the gate-ablation rule. On COVID their paired spreads at h3 RMSE were 694.1
(gate-off) and 809.2 (D2), which are 12.5 and 14.5 percent of v1's mean of 5565.8. Under that rule an
effect had to be about that large just to clear the bar, and larger still to be found reliably. So
those nulls mean "no effect of about 12 to 15 percent or more detected", not "no effect", and the
manuscript should say so.

## 2. The design, fixed

v1's path is kept exactly as released: h = TCN(Z) + LTR(deg), then its SpatialMixer and its gate
(`models/encoder.py`, unchanged). v2 adds one branch (`models/encoder_v2.py`, `DeviationEncoder`):

- **Deviation input.** Computed inside the encoder from Z's incidence channel (0) and obs_mask channel
  (3), so C1's 4-channel assert still holds. For each district i and each week k of the input window:
  dev[i,k] = x[i,k] minus the mean of x[j,k] over the OBSERVED districts j in i's country, same week k.
  Unobserved cells are set to 0. The country index is a per-node input bound per panel, never a
  parameter or a buffer. COVID is one country, so the mean is over all 49 states. Each week reads only
  itself, and the window ends at the origin, so nothing after the origin is read.
- **Branch.** A DilatedTCN on [dev, obs_mask] with width d_dev = 16, then its OWN SpatialMixer(16)
  over the arm's adjacency (section 3), then a Linear(16, 64) projection added to v1's output after
  v1's gate.
- **Combination, and why.** Add through a zero-initialised projection: v2 starts exactly at v1 and
  moves away only if the branch lowers the training loss, so v2 is a strict superset of v1. Apart
  from one floating-point effect described two bullets down, v2 departs from v1 only through what the
  branch learns.
- **Constraints.** C1: Z has exactly 4 channels (v1's assert). C2: no parameter is sized by N,
  asserted with `node_indexed_params` at construction and again after training. C3: every adjacency
  the branch mixes over carries a self-loop (`mask_aware_adj` on the real and shuffled maps, the
  identity on nograph).
- **Pairing with v1.** The v1 submodules are built first and draw the random stream exactly as v1
  does. The branch is built under a forked CPU random stream, so the Adapter built afterwards also
  gets v1's initial weights. The branch TCN runs with dropout 0, which draws no random numbers, so
  v1's dropout masks are unchanged. At the same seed every arm starts from v1's weights, sees the same
  origin order and the same dropout masks.
- **What pairing does NOT give.** Identical training paths. With the branch switched off, the loss
  and every v1 gradient are bit-identical to v1, but the gradient-clip norm is summed over 108
  gradient tensors instead of 58 (the branch's are all zero), moves by one float32 unit, and because
  clipping is active the nudge compounds: the smoke measured a 0.23 percent difference in the scored
  metrics after 3 epochs. With the clip norm taken over the same tensors in both runs, the records
  are bit-identical. So the arms share their start and their random draws, and still diverge, the
  same way the four earlier same-init COVID arms did. That divergence is what the paired spread in
  section 5 measures. `train_one`'s clip is left unchanged so v1 stays v1.
- **Determinism.** A v2 arm run twice at one seed gives identical records (smoke). Two things had to
  change to get there, both inside the branch only. The country sums in the deviation are a one-hot
  matrix product, because `index_add_` adds floats on the GPU in a varying order (repeats disagreed
  at 41 of 41 origins; the matrix product at 0). The branch's convolutions are computed as matrix
  products instead of through cuDNN, because at 16 channels cuDNN's weight-gradient kernels varied
  across repeats. Switching cuDNN to deterministic mode globally was rejected: it also moves v1's own
  convolutions, 0.15 percent after 3 epochs, which would break the pairing with the released v1
  records. The price is speed: a v2 run costs roughly twice a v1 run (1.8 to 2.4 times across the
  smokes I ran).

**Runtime basis.** Every runtime in this file uses nominal costs of 30 seconds per v1 COVID run (the
gaps between the D2 COVID records) and 65 seconds per v2 COVID run. The smokes measured 55 to 70
seconds per v2 run (a 3-epoch timing ratio, which jitters), so every runtime here is rough, to about
15 percent either way.

**Hyperparameters.** Everything shared with v1 is v1's value, unchanged: 80 epochs, patience 15,
early stopping on validation pinball only, AdamW lr 1e-3 and weight decay 1e-4, cosine schedule over
80 epochs, 8 origins per step, gradient clip 1.0, pinball loss over the 5 quantiles, the median as the
point forecast. The v2-only values are fixed here, not tuned, and none was chosen on validation or
test data:

| v2-only setting | value | why |
|---|---|---|
| branch width d_dev | 16 | fixed by the user's order |
| branch TCN dilations and kernel | (1, 2, 4, 8, 16), kernel 2 | v1's, so the receptive field is v1's 32 weeks |
| branch mixer layers | 2 | v1's |
| branch input channels | dev and obs_mask | obs_mask tells a real 0 from a missing cell; the data probe used the same pair |
| branch dropout | 0 | keeps v1's dropout masks, so the arms stay paired |
| projection init | zeros | the strict-superset argument above |
| branch init seed | torch initial seed + 1,000,003 | any fixed offset works; it only has to differ from the main stream |

No tuning happens anywhere in this protocol. If any value in this section has to change, that is a
new experiment: an amendment, and every arm reruns.

## 3. The arms

All three v2 arms run the same code and differ only in the adjacency the deviation branch mixes
over. v1's own path mixes over the real map in every arm.

| arm | branch adjacency | record suffix | what it isolates |
|---|---|---|---|
| v1 | (no branch) | none for the 5 released seeds (`results/single/encoder__covid_us-states__seed<S>.json`, read-only); `__v1ref` for seeds added in stage 2 | the reference |
| v2-graph | the real map | `__v2graph` | the full mechanism |
| v2-nograph | the identity, same parameters and depth | `__v2nograph` | "the deviation input helped" without neighbours |
| v2-shuffled | a degree-preserving relabel of the real map | `__v2shuf` | "any map helped" against "the real map helped" |

v2-shuffled uses `train.loop.permute_adjacency` with permutation seed 20260921 + training seed, the
same fake graphs the D2 retrain used. The identity permutation is refused.

**Adopted: relabel only the branch's map.** In v2-shuffled only the branch's map is relabelled, and
in v2-nograph only the branch's mixing is removed; v1's path keeps the real map in every arm.
Rationale: each comparison then moves exactly one thing, the branch's neighbours. What it costs:
v2-shuffled is not the same experiment as the D2 retrain, which relabelled the whole model (including
v1's neighbours and v1's per-district degree feature), so the two are not directly comparable.

## 4. Panel, seeds, scoring

- **Panel.** covid_us-states only: 49 states, one country, 60 train, 45 validation and 49 test origins.
- **Seeds.** One fixed sequence, written here before any data: seed i = 42 + 10 i for i = 0 to 133,
  that is 42, 52, 62, 72, 82, 92, ..., 1372, 134 seeds. Stage 1 uses the first 5, which are the five
  seeds v1 was released with. Stage 2 uses the first N. No other seed is trained or read.
- **Test scoring, exactly v1's.** `train.loop.train_one` scores the test fold once after early
  stopping, through `score.score_bundle`, in count space. The decision reads model `encoder` (the
  median forecast; `encoder_mc` is never read for the decision, ledger D3), field `country_macro`,
  which on single-country COVID equals `node_mean` (the report checks this), metrics RMSE and MAE at
  h3, h5, h10 and h15. PCC is printed and never decides.
- **Validation metrics, and exactly how they are made.** The released records carry test metrics
  only, so the runner makes validation metrics itself with `score_split` in
  `ablation/run_v2_deviation.py`:
  1. Take the val-selected model. For v1's five released seeds that is the released checkpoint
     `results/single/encoder__covid_us-states__seed<S>__ckpt.pt`, which holds the best-validation
     state `train_one` kept and `run_dataset` saved. For every v2 run and every added v1 seed it is
     the same state, in memory, right after training and before its checkpoint is written.
  2. For every validation origin (`bundle.origins(phase="val")`) and every horizon, invert the median
     quantile through the released scaler into the column of the target week.
  3. Score with `train.loop.score_predictions(..., phase="val")`, which applies the validation mask.
  This is `train_one`'s own test assembly run on another split. Applied to the released
  influenza_japan checkpoint on the test split, it reproduces the released japan record bit for bit
  (selfcheck). The files are `ablation/single/encoder__covid_us-states__seed<S>__<arm>__val.json`,
  every record marked `split: val`. Validation is the early-stopping split, so its error level is
  optimistic; stage 1 uses only the spread of paired differences, never the level of the arms.
- **Records.** `ablation/single/encoder__covid_us-states__seed<S>__{v2graph|v2nograph|v2shuf}.json`
  with a per-node archive, quantile archive and checkpoint beside each; the checkpoint is written last,
  so a run is done when its checkpoint exists. Schema: exactly the v1 key set for the same model plus
  12 v2 keys (`encoder_version`, `v2_arm`, `v2_dev_graph`, `v2_d_dev`, `v2_combine`, `v2_n_groups`,
  `v2_branch_params`, `v2_protocol_sha256`, `v2_dev_shuffle_seed`, `v2_dev_shuffle_hash`,
  `v2_dev_shuffle_frac`, `v2_dev_contrib`). `v2_dev_contrib` is the mean ratio of the branch's output
  size to v1's output size on validation. It is a readout, never a criterion.

## 5. Order of work: the repro gate, then two stages

### 5.1 Repro-check, enforced in code

`--repro-check` refuses to run unless this file is committed with no local edits. It retrains v1 on
influenza_japan seed 42 for the full 80 epochs with today's trainer and compares the result with the
released record, and writes `ablation/misc/v2_repro_check.json`: identical or not, the largest
relative difference, this file's sha256, the commit that holds it, and the sha256 of 12 code files
(`train/loop.py`, the seven `models/` files the encoder uses, `bundles.py`, `score.py`, `to_schema.py`
and the runner). Stage 1 and stage 2 refuse to start unless that record says identical, carries this
file's sha256, and matches the current code hashes. If it reports DIFFER, COVID stays blocked; the
difference is reported and the user decides. Why it gates: the v2 arms are paired against v1 records
made in August, and if today's trainer cannot reproduce them, every v2 minus v1 difference also
carries code drift.

### 5.2 Stage 1, the pilot

All three v2 arms at the 5 stage-1 seeds, paired against the released v1 records. Stage 1 computes
ONLY validation statistics: for each of the three comparisons (v2-graph against v1, against v2-nograph
and against v2-shuffled), the standard deviation of the paired per-seed differences in h3 RMSE on the
validation split, divided by v1's validation mean h3 RMSE. The mean of those differences is not
computed. Stage 1 prints nothing from the test split, and its summary file
(`ablation/misc/v2_stage1_summary.json`) holds only those statistics. The runs do write test records
to disk, because the final analysis needs them, but no command reads a test record before stage 2 is
complete. Stage 1 is 15 v2 runs, about 16 minutes.

### 5.3 The seed count, from code

- n_z = max(5, ceil(((1.96 + 0.84) s / delta)^2)), where s is the LARGEST of the three stage-1
  validation relative standard deviations and delta = 0.033.
- Small-sample t correction: n_t is the smallest n at or above n_z at which the exact power of the
  decision rule reaches 0.80. The decision rule is the paired two-sided 95 percent t-interval
  excluding zero on v2's side; its power uses the noncentral t with n - 1 degrees of freedom and
  noncentrality sqrt(n) delta / s.
- N = min(n_t, 134). If n_t is above 134, stage 2 runs 134 seeds and the report states that the test
  is underpowered for delta.
- `--seed-count` prints N from the stage-1 summary; `--stage2` recomputes it the same way. N is never
  typed by anyone.

delta is the probe-implied h3 RMSE change, 3.30 percent (section 6). **delta assumes the model
captures the full signal the data probe found. If v2 captures only part of it, the real effect is
smaller than delta and may be missed even at N seeds.** Two more limits: s is measured on validation
and the test spread may differ (if it is larger, N is too small), and the power is for h3 RMSE alone
while criterion (a) also needs MAE, so the real chance is a little lower.

Worked examples, fixed now so the formula cannot be reread after stage 1 (stage-2 hours use the
nominal runtime basis of section 2):

| s | n_z | n_t | N | chance of finding delta at N | stage 2 hours |
|---|---|---|---|---|---|
| 0.03 | 7 | 9 | 9 | 0.823 | 0.25 |
| 0.05 | 18 | 21 | 21 | 0.820 | 1.00 |
| 0.10 | 72 | 75 | 75 | 0.805 | 4.38 |
| 0.135 | 132 | 134 | 134 | 0.802 | 8.06 |
| 0.20 | 288 | 291 | 134, underpowered | 0.475 | 8.06 |

### 5.4 Stage 2

All four arms up to N seeds: v1 at the added seeds (92 onward, trained with today's trainer as
`__v1ref` records in `ablation/single/`, never in `results/single/`), plus the three v2 arms at the
same seeds. The runs go seed by seed, so an interruption leaves complete matched sets. The final
analysis uses all N seeds, stage-1 seeds included. Each added seed costs about 3.75 minutes (one v1
run and three v2 runs), so stage 2 is about 8.1 hours at the cap.

### 5.5 The report

`--report` refuses to read any test record until every one of the N x 4 runs exists, checked by file
existence alone. It then decides under section 7.

## 6. The decision rule, and what the design can detect

**The rule.** For each cell (comparison, horizon, metric), d is arm minus reference at the same seed
over the N seeds. A comparison counts only when the paired 95 percent t-interval of d excludes zero:
entirely below zero is better (lower error), entirely above is worse. The gate-ablation rule's verdict
(within noise when the absolute mean of d is below the standard deviation of d) is printed beside
every cell and carries no weight.

**Why the gate-ablation rule was replaced for this test.** Mean d at least sd d is the same as a
paired t statistic of at least the square root of the seed count. At 5 seeds a cell with no true
effect is called "better" 4.45 percent of the time. It is a bar on effect size, not on standard
error, so adding seeds does not lower it: at 10 seeds the same no-effect rate falls to 0.58 percent
and the effect it can find barely moves.

**Planning, before stage 1, from existing arms only (no v2 number).**
`python ablation/run_v2_deviation.py --power`, seconds, no GPU.

- v1's spread is the five released COVID seeds.
- The paired spread expected between v2 and v1 is planned from four arms that were already run on COVID
  at the same seeds with v1's initial weights (gate-off, shuffled adjacency, and the two epidemiology
  penalties): the median of their paired standard deviations against v1, with the range across the
  four. A v2 trained independently of v1 would sit near the square root of 2 times v1's spread, which
  is larger. Stage 1 replaces this planning spread with v2's own validation spread.
- The effect the data probe implies: the real-map cut in the probe's model-space squared error
  (mse_A minus mse_real in `results/misc/t6_neighbour_signal.json`), turned into an RMSE change against
  v1's own model-space squared error on the same test cells (read from v1's quantile archives). The
  same percentage is used for MAE. This assumes v2 captures the whole linear neighbour gain and v1
  captured none of it, and it is model space, not the count space the metrics use. It could be larger
  or smaller in counts, and by how much is unknown.

Noise on COVID, the five v1 seeds:

| horizon | metric | v1 mean | v1 sd | paired sd, planning | paired sd, range | rule's bar, % of v1 mean |
|---|---|---|---|---|---|---|
| 3 | rmse | 5565.8 | 915.0 | 751.7 | 509.8 to 878.6 | 13.5 |
| 3 | mae | 4352.6 | 806.4 | 632.3 | 424.4 to 735.0 | 14.5 |
| 5 | rmse | 8798.3 | 841.7 | 992.1 | 196.9 to 1262.4 | 11.3 |
| 5 | mae | 7129.3 | 639.9 | 861.9 | 159.3 to 1097.1 | 12.1 |
| 10 | rmse | 12263.8 | 1859.0 | 1221.1 | 737.6 to 1558.2 | 10.0 |
| 10 | mae | 9973.5 | 1462.0 | 982.6 | 519.6 to 1020.2 | 9.9 |
| 15 | rmse | 11343.2 | 1464.6 | 776.0 | 535.9 to 1532.3 | 6.8 |
| 15 | mae | 9676.3 | 1301.6 | 708.1 | 457.3 to 1478.5 | 7.3 |

What each rule can find, against what the probe implies. "Found 80 percent of the time" is in
percent of v1's mean.

| horizon | metric | probe-implied change % | effect in paired sds | found 80% of the time, gate rule, 5 seeds % | same, 10 seeds % | t-interval, 10 seeds % | t-interval, 20 seeds % | t-interval, 40 seeds % | chance of finding the probe effect, gate rule, 5 seeds | gate rule, 10 seeds | t-interval, 20 seeds | t-interval, 40 seeds | seeds for 80%, t-interval |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | rmse | 3.30 | 0.24 | 19.1 | 17.6 | 13.5 | 8.9 | 6.1 | 0.106 | 0.030 | 0.178 | 0.325 | 134 |
| 3 | mae | 3.30 | 0.23 | 20.5 | 18.9 | 14.5 | 9.6 | 6.6 | 0.101 | 0.027 | 0.160 | 0.288 | 155 |
| 5 | rmse | 0.77 | 0.07 | 15.9 | 14.7 | 11.2 | 7.4 | 5.1 | 0.058 | 0.010 | 0.047 | 0.062 | 1694 |
| 5 | mae | 0.77 | 0.06 | 17.1 | 15.7 | 12.0 | 8.0 | 5.5 | 0.057 | 0.009 | 0.046 | 0.058 | 1947 |

At 5 seeds under the gate rule the chance of finding the probe effect at h3 RMSE is 10.6 percent,
against a 4.45 percent chance of a false call. With that rule, 10 seeds are worse, not better (3.0
percent), because its bar does not shrink with seeds. Under the t-interval, 40 seeds find a 6.1
percent change at h3 RMSE 80 percent of the time, and 134 seeds are needed to find the 3.30 percent
the probe implies. These planning numbers are why the cap is 134.

**(a) is in effect an h3 test.** The probe-implied h5 change is 0.77 percent, which would need about
1,700 seeds. h5 stays in criterion (a) and is reported, but it cannot be powered.

Options considered before the two-stage design was chosen (runtimes on the nominal basis of section
2):

| option | runs | time | chance of finding the probe effect at h3 RMSE |
|---|---|---|---|
| 5 seeds, gate rule | 15 v2 | about 16 minutes | 0.106 |
| 10 seeds, gate rule | 5 v1 + 30 v2 | about 35 minutes | 0.030 |
| 40 seeds, t-interval rule | 35 v1 + 120 v2 | about 2.5 hours | 0.325 |
| 134 seeds, t-interval rule | 129 v1 + 402 v2 | about 8.3 hours | 0.80 |

Chosen: the two-stage design of section 5, which sets N from v2's own measured spread instead of
fixing it in advance, under the t-interval rule, capped at 134.

## 7. Pre-committed criteria

v2-graph is compared against v1, against v2-nograph and against v2-shuffled, each paired by seed over
the N seeds, under the section 6 rule. `decide()` in `ablation/run_v2_deviation.py` applies these
mechanically, and its selfcheck tests every branch below.

- **(a) PRIMARY.** v2-graph beats v1 at h3 or h5, on BOTH RMSE and MAE at that horizon (the t-interval
  lies entirely below zero for both), AND there is no horizon where the t-interval shows v2-graph
  significantly worse than v1 on either metric.
- **(b) ATTRIBUTION.** At a horizon that passed (a), v2-graph ALSO beats v2-nograph and v2-shuffled,
  on both RMSE and MAE, all at that one horizon.
- **(c) FAIL.** (a) is not met, for either reason: no short-horizon win, or a win with a significant
  loss elsewhere.

Outcomes and what each licenses:

| outcome | condition | what we may say | what we may not say |
|---|---|---|---|
| PASS | (a) and (b) | On COVID, a single-disease model that is handed district deviations gets accuracy from the real map, and neither the extra input alone nor a wrong map gives it. This supports "the encoder's compression is why the graph failed, and routing deviations fixes it" on COVID. | Anything about transfer, Ebola, or the other panels. That the compression is the whole story. |
| A_ONLY | (a) without (b) | The extra deviation input helped, not the graph. If v2-graph beats v2-nograph but not v2-shuffled, neighbour mixing helped but any map did. | That the real map helped. |
| FAIL | (a) not met | No effect found at N seeds. If N was not capped, the design had about 0.80 chance on h3 RMSE alone of finding delta, provided the test spread matches the stage-1 validation spread; if N was capped, the test was underpowered for delta. Stop: no dengue run. | That the deviation channel has no effect. That the graph is useless in general. |

Nothing else decides. PCC, `encoder_mc`, `v2_dev_contrib`, the gate-rule column, and any subgroup
are printed for the reader and carry no verdict.

## 8. Gates for later steps

- **Dengue.** Only if COVID is PASS, and only under its own amendment file, committed first, which
  also extends the runner: this runner has no dengue path. The amendment must fix seeds, runtime (a
  v1 dengue run is about 2.5 hours, so 3 arms at 5 seeds is about 38 hours before the branch's cost),
  the per-country deviation mean over dengue's 12 countries, and the same criteria on
  `country_macro`. The power check says dengue is no better placed than COVID at h3 and h5 (the probe
  effect is 0.07 to 0.19 paired sds) and somewhat better at h10 (about 0.33), so this gate does not
  hold back a better-powered panel.
- **A_ONLY or FAIL.** No dengue run for the graph question. Whether the "input helped" reading is
  worth its own experiment is a separate decision.
- **Ebola.** v2 on Ebola, if ever run, is EXPLORATORY only. It never re-scores the pre-registration,
  never writes to `results/ebola/`, and cannot enter the case-study result. This runner has no Ebola
  path. The Ebola data probe already found no neighbour timing signal at h3 and h5 once outbreak size
  is controlled.
- **Other panels and the transfer trunk.** Not in this protocol. Each would need its own.

## 9. After the data

Nothing in sections 2 to 8 changes once the first COVID record is written. A run that crashes is
rerun at the same seed and arm by starting the same stage again (a run counts as done only when its
checkpoint, its last write, exists), and the rerun is reported. A run is never repeated because of its result. The report
refuses to decide if any of the N x 4 runs is missing, if any record claims the wrong arm, or if the
records carry a protocol hash other than this file's; when it refuses, it prints no test number.

## 10. Phase 1 checks behind this protocol

- `--selfcheck`, 17 groups: C2 with a planted control; the branch convolutions equal to cuDNN's maths
  while v1's stay on cuDNN; deviation arithmetic by hand; same-week-only deviation with a planted
  leaky control; the v2 output at an origin unchanged when every later week is rewritten; the v1
  checkpoint loaded into v2 with the zero branch reproducing v1's forward output exactly on all three
  arms; arm guards; v1 and Adapter initial weights identical at a seed, with an unforked control that
  shifts them; identity permutation refused; the schema check caught by six mutants; the decision
  logic on nine hand-built grids; the seed count (formula, t correction, floor, cap, largest s);
  stage 1 reading validation files only, with a test sentinel that must not reach its output; the
  report refusing test while stage 2 is incomplete and following the t-interval where the two rules
  disagree; the repro gate and the stage guards under controlled state; no free COVID training path;
  and the validation scorer reproducing a released record on the test split.
- `--mutate-selfcheck`: ten bugs planted in the checker itself, one at a time (flipped verdict sign,
  harm ignored, leaky deviation, stray schema keys allowed, N from the smallest s, stage 1 reading
  test files, the report skipping its completeness check, the decision using the gate rule, the repro
  gate always open, the seed count without its t correction); the selfcheck fails on every one. The
  completeness mutant is caught by the report crashing on a missing file rather than by an assertion.
- `diagnostics/verify_v2_protocol.py --mutate`: every number in this file that comes from disk is
  recomputed by separate code that does not import the runner, the runner's constants are checked
  against this file, and corruptions of the file and of its inputs are each caught.
- `--smoke` on influenza_japan only: v2 with the branch gain set to 0 trains to records identical to
  v1's once the clip norm is summed over the same tensors (0.23 percent apart without that, see
  section 2); v2-graph run twice at one seed gives identical records; the validation scorer runs on a
  v2 checkpoint; then all three arms end to end with the schema checked. The branch adds 10,224
  parameters. The smoke writes its measurements to `ablation/misc/v2_smoke_summary.json`.
- v1 on this box is bit-identical run to run at a fixed seed (checked on influenza_japan, 3 epochs).

## 11. Reproduce, in order

```
conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --selfcheck
conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --mutate-selfcheck
conda run --no-capture-output -n ebola-train python diagnostics/verify_v2_protocol.py --mutate
conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --power
conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --smoke
conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --repro-check
conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --stage1
conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --seed-count
conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --stage2
conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --report
```

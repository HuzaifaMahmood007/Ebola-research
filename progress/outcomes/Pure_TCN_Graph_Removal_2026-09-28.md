# Removing the whole graph: pure-TCN and a fresh gate-off

**Runs 2026-09-25 to 2026-09-28. Recorded 2026-09-29.** Scripts `experiments/pure_tcn.py` and
`experiments/gateoff_fresh.py`. Records `experiments/pure_tcn__{panel}__seed{S}.json` and
`experiments/gateoff_fresh__{panel}__seed{S}.json`, plus `experiments/pure_tcn__summary.json` and
`experiments/gateoff_fresh__summary.json`. Code and records committed in `e59b231`.

**Everything here is EXPLORATORY and is stamped `protocol="EXPLORATORY"` in every new record.** It
was not pre-registered, it is not a scored result, and it must not appear beside the gate-off table
(Table 3 of the manuscript) without that label. Nothing under `results/` or `ablation/` was written;
both were only read.

Verifier: `diagnostics/verify_pure_tcn_doc.py`, which recomputes every number below from the seed
files.

---

## In plain words

The gate-off ablation behind Table 3 switched off neighbour mixing but left one piece of the graph
inside the model: a learned feature built from each district's number of neighbours (the degree
feature). So "the graph does not help error" had only been tested with part of the graph still in.
I removed the rest on the two cheapest panels, COVID and influenza-Japan. Removing the whole graph
moved none of the 16 error cells beyond seed noise.

Splitting the graph into its two parts gives a sharper picture at one horizon. At h5, on both
panels, neighbour mixing raises error and the degree feature lowers it by about the same amount, so
in the full model the two cancel. I do not know why this happens at h5 only.

I also retrained the gate-off arm from scratch on today's code. It reproduced the archived gate-off
records exactly, so Table 3's gate-off cells for these two panels stand.

---

## The question, and why it was asked

`ablation/run_gate_ablation.py` forces the spatial gate to g = 0, but the encoder still computes
`h = self.tcn(Z) + self.ltr(deg)` (`models/encoder.py:60`). `ltr` is `LTR`, a `Linear(1, 64)` on
`log1p(degree)` (`models/spatial.py:53-61`): one learned vector per district, derived from the graph.
The ablation's report says so every time it prints: "g=0 removes neighbour mixing but keeps the LTR
degree feature, so this bounds the value of NEIGHBOUR INFORMATION, not of the graph in total"
(`ablation/run_gate_ablation.py:115`, printed in `results/reports/gate_ablation.log`). The audit flagged
the same gap (`Reports/Phase0_to_Now_Audit.md:155`), and the manuscript states it as a limit
(`Reports/Manuscript_v2.md:332`). No run anywhere had removed the degree feature too.

So the question: does removing the WHOLE graph, mixing and degree together, change forecast error?

---

## Approach

### Three arms and one reproduction check

All on `covid_us-states` and `influenza_japan`, five seeds (42, 52, 62, 72, 82), model `encoder`,
test split, the `node_mean` field (equal to `country_macro` on these one-country panels).

| arm | what the model sees of the graph | records |
|---|---|---|
| learned (full model) | neighbour mixing through the learned gate, plus the degree feature | `results/single/encoder__{panel}__seed{S}.json` |
| gate-off | the degree feature only (g = 0) | `ablation/single/encoder__{panel}__seed{S}__gateoff.json`, 2026-08-17 |
| pure-TCN | nothing: the forecast comes from the temporal convolution alone | `experiments/pure_tcn__{panel}__seed{S}.json` |
| fresh gate-off (the check) | same as gate-off, retrained on today's code | `experiments/gateoff_fresh__{panel}__seed{S}.json` |

Three paired comparisons, d = arm minus reference at the same seed:

- **pure-TCN vs learned** measures the whole graph.
- **pure-TCN vs gate-off** measures the degree feature, with mixing already off.
- **gate-off vs learned** measures neighbour mixing, with the degree feature present. These are
  Table 3's own numbers for these two panels.

The verdict rule is the one the gate-off and shuffled-adjacency runs use: a cell is **within noise**
when |mean d| < sd of the five paired differences. Otherwise the label says which way it went, and
"helps" always means that removing the part made the cell worse.

### How pure-TCN is built

Through the existing `encoder_factory` hook in `train.loop.train_one` (`train/loop.py:179-183`). The
factory builds the ordinary `SharedEncoder(gate_mode="off")`, then replaces `enc.ltr` with a module
that has no parameters and returns 0.0, so the trunk computes `tcn(Z) + 0.0`. The full build runs
first, so the discarded LTR still used up its random draws, and the TCN, the spatial mixer and the
Adapter start from exactly the weights the gate-off arm starts from at the same seed. Training call:
`train_one(ds, s, gate_mode="off", encoder_factory=factory, verbose=False, gate_read=False)`. Records
are stamped `gate_mode="off"`, `ltr="off"`, `encoder_version="v1_pure_tcn"`. `gate_read=False` only
skips a gate readout that runs after the test predictions exist (`train/loop.py:285`), so it cannot
change a record.

`python -m experiments.pure_tcn --selfcheck` proves the construction, and every check carries a
planted control that must fail. I reran it on 2026-09-29 on today's code and it passes:

- pure-TCN output is identical under the real adjacency and an all-zero one, on both panels, while
  the gate-off control moves by up to 1.498.
- pure-TCN output equals `enc.tcn(Z)`, and the replacement module holds no parameters.
- the factory draws no CPU random numbers: RNG state, TCN, spatial and Adapter weights match a plain
  gate-off build exactly.

### How the fresh gate-off is built

`experiments/gateoff_fresh.py` makes exactly the call at `ablation/run_gate_ablation.py:153`,
`train_one(ds, s, epochs=80, verbose=False, gate_mode="off")` (80 is that script's `--epochs`
default), and writes to `experiments/`. Its selfcheck, also rerun 2026-09-29, confirms the trunk it
builds still reads the graph through the degree feature, and that the pure-TCN trunk does not.

### Completeness

All 20 new seed files are present: pure-TCN COVID written 2026-09-25 20:46 to 20:48, pure-TCN Japan
2026-09-28 13:47 to 13:54, fresh gate-off on both panels 2026-09-28 22:36 to 22:50. Every pure-TCN
record carries the four stamps above. Every fresh gate-off record carries `protocol="EXPLORATORY"` and
`gate_mode="off"` and no factory stamp. All four arms have all five seeds on both panels.

The Japan learned seed-42 file has a 2026-09-22 timestamp, but its content has not changed since its
first commit (`c23eb1d`, 2026-09-04), and `results/reports/gate_ablation.log`, written 2026-08-17,
prints the same learned-arm means and sds as today's files. The timestamp does not mark a changed
number.

---

## Results

### 1. The archived gate-off records reproduce exactly

| panel | RMSE, MAE, PCC values identical | all seven metrics identical | seed files identical apart from the protocol stamp | max relative difference |
|---|---|---|---|---|
| covid_us-states | 60 of 60 | 140 of 140 | 5 of 5 | 0 |
| influenza_japan | 60 of 60 | 140 of 140 | 5 of 5 | 0 |

The fresh run matches the archive exactly, not approximately. The script expected small drift (its
docstring cites cuDNN non-determinism in the dilated convolutions); none appeared, and I did not
establish why. As a result, every pure-TCN vs gate-off number below is the same whether the reference
is the archived or the fresh gate-off.

### 2. Verdicts at a glance

| comparison | what d measures | panel | helps | hurts | within noise |
|---|---|---|---|---|---|
| pure-TCN vs learned | the whole graph | covid_us-states | 1 | 0 | 11 |
| pure-TCN vs learned | the whole graph | influenza_japan | 0 | 0 | 12 |
| pure-TCN vs gate-off | the degree feature | covid_us-states | 2 | 0 | 10 |
| pure-TCN vs gate-off | the degree feature | influenza_japan | 2 | 0 | 10 |
| gate-off vs learned | neighbour mixing | covid_us-states | 0 | 2 | 10 |
| gate-off vs learned | neighbour mixing | influenza_japan | 0 | 5 | 7 |

Twelve cells per row: RMSE, MAE and PCC at h3, h5, h10 and h15.

### 3. Removing the whole graph: pure-TCN vs learned

**0 of 16 error cells differ beyond seed noise.** Pure-TCN has numerically lower error than the
learned model in 11 of 16 error cells (COVID 4 of 8, Japan 7 of 8), every one of them within noise.

| panel | metric | h | learned | pure-TCN | d (pure - learned) | sd of d | verdict |
|---|---|---|---|---|---|---|---|
| covid_us-states | RMSE | 3 | 5565.755 | 5405.942 | -159.813 | 1023.306 | within noise |
| covid_us-states | RMSE | 5 | 8798.302 | 8697.706 | -100.595 | 1077.100 | within noise |
| covid_us-states | RMSE | 10 | 12263.816 | 12573.753 | +309.937 | 2530.404 | within noise |
| covid_us-states | RMSE | 15 | 11343.231 | 11959.436 | +616.205 | 2159.810 | within noise |
| covid_us-states | MAE | 3 | 4352.600 | 4108.506 | -244.094 | 980.908 | within noise |
| covid_us-states | MAE | 5 | 7129.292 | 7118.292 | -11.000 | 877.503 | within noise |
| covid_us-states | MAE | 10 | 9973.489 | 10429.566 | +456.077 | 1851.767 | within noise |
| covid_us-states | MAE | 15 | 9676.309 | 10313.568 | +637.259 | 1861.689 | within noise |
| covid_us-states | PCC | 3 | 0.425 | 0.384 | -0.041 | 0.160 | within noise |
| covid_us-states | PCC | 5 | -0.176 | -0.212 | -0.036 | 0.059 | within noise |
| covid_us-states | PCC | 10 | -0.357 | -0.394 | -0.038 | 0.025 | GRAPH HELPS |
| covid_us-states | PCC | 15 | -0.324 | -0.378 | -0.054 | 0.060 | within noise |
| influenza_japan | RMSE | 3 | 734.888 | 682.245 | -52.643 | 114.713 | within noise |
| influenza_japan | RMSE | 5 | 841.106 | 838.179 | -2.927 | 53.858 | within noise |
| influenza_japan | RMSE | 10 | 1063.017 | 1032.356 | -30.661 | 103.825 | within noise |
| influenza_japan | RMSE | 15 | 1031.386 | 1023.461 | -7.925 | 160.649 | within noise |
| influenza_japan | MAE | 3 | 261.526 | 242.361 | -19.165 | 35.929 | within noise |
| influenza_japan | MAE | 5 | 323.461 | 323.959 | +0.498 | 23.635 | within noise |
| influenza_japan | MAE | 10 | 444.963 | 432.804 | -12.159 | 38.259 | within noise |
| influenza_japan | MAE | 15 | 443.764 | 439.461 | -4.302 | 56.494 | within noise |
| influenza_japan | PCC | 3 | 0.872 | 0.889 | +0.017 | 0.040 | within noise |
| influenza_japan | PCC | 5 | 0.899 | 0.914 | +0.015 | 0.018 | within noise |
| influenza_japan | PCC | 10 | 0.844 | 0.846 | +0.002 | 0.021 | within noise |
| influenza_japan | PCC | 15 | 0.837 | 0.836 | -0.001 | 0.035 | within noise |

The one flag is COVID PCC h10: learned -0.357, pure-TCN -0.394, d -0.038 with sd 0.025. Both arms
are negative there, so at h10 the forecasts move against the truth with or without the graph. One
flag in 12 cells is what this rule produces by chance alone (see Caveats).

### 4. Splitting the graph at h5: mixing against the degree feature

Seed means at h5, and the three paired differences.

| panel | metric | learned | gate-off | pure-TCN | gate-off - learned (mixing removed) | pure-TCN - gate-off (degree removed) | pure-TCN - learned (whole graph removed) |
|---|---|---|---|---|---|---|---|
| covid_us-states | RMSE | 8798.3 | 7835.4 | 8697.7 | -962.9 | +862.3 | -100.6 |
| covid_us-states | MAE | 7129.3 | 6392.5 | 7118.3 | -736.8 | +725.8 | -11.0 |
| influenza_japan | RMSE | 841.1 | 779.4 | 838.2 | -61.7 | +58.7 | -2.9 |
| influenza_japan | MAE | 323.5 | 301.3 | 324.0 | -22.1 | +22.6 | +0.5 |

Pure-TCN minus gate-off at h5, per seed:

| panel | difference | seed 42 | seed 52 | seed 62 | seed 72 | seed 82 |
|---|---|---|---|---|---|---|
| covid_us-states | RMSE at h5 | +834.5 | +362.6 | +1291.8 | +1066.2 | +756.6 |
| covid_us-states | MAE at h5 | +796.4 | +345.5 | +1000.7 | +974.8 | +511.4 |
| influenza_japan | RMSE at h5 | +25.0 | +103.4 | +40.4 | +17.8 | +107.2 |
| influenza_japan | MAE at h5 | +8.4 | +42.3 | +17.8 | +3.8 | +40.8 |

All 20 per-seed differences are positive.

**Reading.** At h5, on both panels and on both error metrics, the two parts move error in opposite
directions by about the same amount. Removing mixing lowers error, so mixing raises it. Removing the
degree feature (with mixing already off) raises error, so the degree feature lowers it. In the full
model they cancel, which is why removing the whole graph does nothing to error at h5. The cause of
the h5-only pattern is unknown, and I do not offer one.

### 5. Every flagged cell in the two split comparisons

| panel | comparison | metric | h | d | sd of d | seeds with d > 0 | verdict |
|---|---|---|---|---|---|---|---|
| covid_us-states | pure-TCN vs gate-off | RMSE | 5 | +862.322 | 349.181 | 5 of 5 | DEGREE HELPS |
| covid_us-states | pure-TCN vs gate-off | MAE | 5 | +725.761 | 288.621 | 5 of 5 | DEGREE HELPS |
| influenza_japan | pure-TCN vs gate-off | RMSE | 5 | +58.731 | 43.272 | 5 of 5 | DEGREE HELPS |
| influenza_japan | pure-TCN vs gate-off | MAE | 5 | +22.624 | 18.019 | 5 of 5 | DEGREE HELPS |
| covid_us-states | gate-off vs learned | RMSE | 5 | -962.918 | 790.682 | 0 of 5 | mixing HURTS |
| covid_us-states | gate-off vs learned | MAE | 5 | -736.761 | 668.566 | 1 of 5 | mixing HURTS |
| influenza_japan | gate-off vs learned | RMSE | 3 | -45.265 | 43.453 | 1 of 5 | mixing HURTS |
| influenza_japan | gate-off vs learned | RMSE | 5 | -61.658 | 46.174 | 0 of 5 | mixing HURTS |
| influenza_japan | gate-off vs learned | MAE | 5 | -22.126 | 14.266 | 0 of 5 | mixing HURTS |
| influenza_japan | gate-off vs learned | MAE | 10 | -24.007 | 23.866 | 1 of 5 | mixing HURTS |
| influenza_japan | gate-off vs learned | PCC | 10 | +0.011 | 0.008 | 5 of 5 | mixing HURTS |

Mixing helps in no cell on either panel, and the degree feature hurts in no cell on either panel. On
Japan, mixing also raises error at RMSE h3 and MAE h10 and lowers correlation at h10, where the
degree feature is within noise. Those three cells are why "the two parts cancel" is an h5 statement
only.

### 6. Seed spread

RMSE seed sd across the five seeds:

| panel | metric | arm | h3 | h5 | h10 | h15 |
|---|---|---|---|---|---|---|
| covid_us-states | RMSE | learned | 915.0 | 841.7 | 1859.0 | 1464.6 |
| covid_us-states | RMSE | pure-TCN | 214.0 | 701.1 | 832.6 | 1115.1 |
| influenza_japan | RMSE | learned | 58.5 | 35.4 | 74.9 | 121.0 |
| influenza_japan | RMSE | pure-TCN | 68.5 | 71.8 | 75.2 | 53.2 |

On COVID the pure-TCN seed spread is smaller than the learned model's at all 4 horizons. On Japan it
is larger at 3 of 4 horizons and smaller only at h15. On COVID MAE it is smaller at 3 of 4 horizons,
h5 being the exception. So this is a COVID RMSE observation, not a property of the arm.

---

## Decisions, with the reason for each

| decision | why | what it gave up |
|---|---|---|
| Build pure-TCN through the `encoder_factory` hook, not by editing `models/encoder.py` or `train/loop.py` | Both files are among the 12 in the V2 protocol's code hash (`ablation/run_v2_deviation.py:89-91`), and `DeviationEncoder` inherits `SharedEncoder.__init__` (`models/encoder_v2.py:102-104`), so an edit would change a hash-frozen protocol's code | The arm lives in an experiments script, not in the model's own `gate_mode` options, so a reader has to find it there |
| Replace LTR with a parameter-free module that returns zero | It draws no random numbers, so the starting weights match gate-off at the same seed and the only difference at step zero is the missing degree term | The spatial mixer is still built and still runs forward, so this is "graph removed from the output", not "graph code deleted" |
| Write to `experiments/`, stamp EXPLORATORY, skip any seed whose file exists, never overwrite | The user's instruction on 2026-09-25 | A deliberate retrain needs `--force`, and for pure-TCN also moving the old summary aside |
| Did NOT rerun `ablation/run_gate_ablation.py --force`; wrote `experiments/gateoff_fresh.py` instead | Without `--force` that script trains nothing, because the records exist. With `--force` it overwrites the archived gate-off records, which are gitignored (`.gitignore:31`, `/ablation`) and untracked (no `__gateoff` file is in `git ls-files`), so the originals would be lost for good | One more script in `experiments/` |
| COVID first, then Japan | The user's choice: the two cheapest panels, with both references on disk | The three panels where mixing helped correlation are untested (see Caveats) |
| Records only: no per-node archives, quantiles or checkpoints | A rerun is minutes per seed | No per-district or calibration reading of either arm |
| The same verdict rule as the gate-off and shuffled-adjacency runs | So the tallies line up with Table 3 | The rule is weak (see Caveats), and a stronger test would not be comparable to Table 3 |
| The pure-TCN summary refuses to overwrite a panel whose recomputed numbers changed; the fresh gate-off summary merges without refusing | For the fresh arm the seed files are the record, and every summary number is re-derived from them | The fresh summary can change without warning if a seed is retrained with `--force` |
| The user ran all training in their own shell | Project rule: anything taking minutes runs in the user's shell | No console log of either run was saved, so runtimes come from file timestamps only: about 0.5 min per COVID seed in both arms, 1.1 to 2.3 min between Japan pure-TCN seed files, 1.8 to 3.2 min between Japan fresh gate-off seed files |

---

## Caveats

- **A weak test.** Five seeds and the |mean d| < sd rule. With no real effect, the rule flags a cell
  when |t| > sqrt(5) = 2.236 at 4 degrees of freedom, which happens 8.9 percent of the time, about
  1.07 false flags per 12 cells. So COVID's single GRAPH HELPS flag is what chance alone produces.
  Nothing here is a significance claim. RMSE and MAE at the same horizon are not independent, so the
  two degree flags per panel are closer to one finding than two. What makes them more than a chance
  flag is that they land on the same horizon on both panels and all 20 per-seed differences have the
  same sign.
- **Two panels only, and they are the convenient ones.** These are the panels where mixing never
  helped correlation in Table 3. The panels where it did, US-regions, US-states and dengue, were not
  tested, so the manuscript's "message passing buys shape" claim is untouched by this run, and a
  reviewer could fairly call the panel choice convenient.
- **The split runs along one path only.** Mixing was measured with the degree feature present, and
  the degree feature with mixing absent. The other order, a model with mixing but no degree feature,
  was not run. So the two differences are not a proper additive split, and an interaction between
  mixing and degree would not show here.
- **The spatial mixer still exists** in the gate-off and pure-TCN arms. It runs forward, its output is
  multiplied by g = 0, and it receives zero gradient.
- **The training paths differ from the first step.** Pure-TCN and gate-off start from the same
  weights, but pure-TCN's forward lacks the degree term from step one, and dropping LTR's 128
  parameters changes the set that `clip_grad_norm_` scales (`train/loop.py:233`, `:237`). The paired
  difference therefore includes trajectory divergence that behaves like extra seed noise.
- **The h5-only pattern has no known cause.**
- **COVID correlation is negative at h5, h10 and h15 in all three arms.** COVID PCC differences at
  those horizons compare forecasts that already move against the truth.
- **"Degree feature" means `LTR(log1p(deg))`**, one learned vector per district computed from its
  neighbour count. It is derived from the graph, but it is not neighbour information: it carries
  nothing about what the neighbours are doing.
- **Code changed while the work ran.** `train/loop.py` was edited by the parallel epi session on
  2026-09-25 at 21:01, after the COVID pure-TCN seeds and before the Japan ones. The change sits
  entirely under `if epi and train_mode:`, which these runs never enter (committed in `b24063f`).
  `ablation/run_epi_ablation.py`, whose loaders the pure-TCN report reuses, changed after the COVID
  summary was first written; the summary rewrite on 2026-09-26 recomputed COVID from the seed files
  with no mismatch. The exact gate-off reproduction on 2026-09-28 confirms no drift on this path.
- **Not pre-registered.** The question was fixed before the run, but no protocol was frozen.
- **Single-disease trunks only.** Nothing here touches the transfer trunk or Ebola.

---

## What it means for the manuscript

Nothing below has been applied to `Reports/Manuscript_v2.md`. The edit is tracked as an open
checkbox in `todo_Milestone6.md` section 2.2.

**Proposed headline**, for wherever the graph finding is summarised:

> In an exploratory follow-up, removing the graph altogether, both neighbour mixing and the node-degree feature, moved none of 16 forecast-error comparisons beyond seed noise on influenza (Japan) and COVID across five paired seeds. Only at the five-week horizon did both parts move error beyond seed noise, and there, on both panels, mixing raised error and the degree feature lowered it by about the same amount, so the two cancel.

**The sentence that must change**, `Reports/Manuscript_v2.md:326`:

> Those are concentrated on influenza-Japan, where removing the graph improves RMSE by 45.3 at *h* = 3 and 61.7 at *h* = 5, and on COVID at *h* = 5. On those panels neighbouring units share a seasonal phase but not a baseline level, so mixing brings in bias along with the timing.

The 45.3 and 61.7 are gate-off numbers, so what was removed is neighbour MIXING, not the graph. With
the whole graph removed, Japan RMSE moves by -52.6 (sd 114.7) at h3 and -2.9 (sd 53.9) at h5, and
COVID h5 RMSE by -100.6 (sd 1077.1), all within noise. The Japan h3 point estimate for removing the
whole graph is slightly larger than the gate-off one (52.6 against 45.3); it is within noise because
the pure-TCN seeds spread more there, not because the difference vanished.

**Proposed replacement:**

> Those are concentrated on influenza-Japan, where removing neighbour mixing improves RMSE by 45.3 at *h* = 3 and 61.7 at *h* = 5, and on COVID at *h* = 5. An exploratory check that also removed the degree feature moved none of these panels' 16 error cells beyond seed noise: at *h* = 5 the degree feature offsets the mixing cost.

```
What you get: the "seasonal phase but not a baseline level" explanation is an untested mechanism,
  and the replacement is a measured result. The misnaming of what was removed is fixed.
What it costs: the explanation is 23 words, the replacement 31, so the edit adds 8 words net. Using
  the full headline in its place would add 46. The manuscript is 13,834 words by wc -w on
  2026-09-29, 1,834 over the 12,000 internal target. A reader also loses a plausible-sounding story
  for why mixing hurts, with nothing put in its place except "cause unknown".
My call: the compact replacement at :326, and the full headline only where it can replace an
  existing summary sentence, because the explanation it removes was never tested.
```

**Also touched, the first sentence of `:332`:** "Setting *g* = 0 removes neighbour mixing but keeps
the LTR degree feature, so the ablation measures the value of *neighbour information*, not of the
graph as a whole." It is still true of Table 3, but on COVID and Japan the whole graph has now been
removed as well. Narrow it in the same pass as EDIT 1a of
`progress/outcomes/Manuscript_Correctness_Pass_2026-09-23.md`, which rewrites that paragraph's second
sentence, so the paragraph is edited once. The opening of `:326`, "Switching the graph off costs
correlation", also means switching mixing off; same wording fix, lower priority.

**Table 3 stands.** Its gate-off cells for COVID and Japan reproduce exactly on today's code
(Results, section 1).

---

## What is still open

1. Whole-graph removal on US-regions, US-states and dengue, the panels where mixing helped
   correlation. The 2026-08-17 gate-off log puts US-regions and US-states at 1.5 to 2.5 min per seed
   and dengue at 63.8 to 136.6 min per seed, about 8.5 h for five.
2. The other half of the split: mixing on, degree feature off.
3. Why h5, and only h5.
4. The manuscript edits above.
5. This document and its verifier are untracked until they are committed.

---

## Reproduce

```
conda run --no-capture-output -n ebola-train python -m experiments.pure_tcn --selfcheck
conda run --no-capture-output -n ebola-train python -m experiments.gateoff_fresh --selfcheck
conda run -n ebola-train python diagnostics/verify_pure_tcn_doc.py
conda run -n ebola-train python diagnostics/verify_pure_tcn_doc.py --mutate
```

`python -m experiments.pure_tcn --report` and `python -m experiments.gateoff_fresh --report` rebuild
the two summaries from the seed files; the pure-TCN one refuses to write if any recorded number would
change. Retraining (`python -m experiments.pure_tcn`, `python -m experiments.gateoff_fresh`) skips every
seed whose file exists, so on this disk it trains nothing. COVID takes about 0.5 min per seed and Japan
about 2 to 3 min per seed.

# Doubt.md — Gravity mobility review, confirmed wins, and open bugs

**Date:** 2026-08-03
**Trigger:** proposal to add a uniform gravity-model `A_mob` to every disease bundle, so that mobility
could not act as a disease identifier for the shared encoder.
**Outcome:** proposal rejected. Two independent reviewers, same verdict. Several higher-yield problems
found instead.

---

## 1. Decision: do NOT build the gravity mobility adjacency

### 1.1 The code cannot consume a second adjacency

- `SharedEncoder.forward(self, Z, A, M_t=None)` — `models/encoder.py:57` — takes exactly one adjacency.
  One `normalise_adj` (`:59`), one `SpatialMixer` (`:64`), one gate (`:65`).
- `bundles.py:35` declares `A_geo` only; `bundles.py:122` `load()` never reads an `A_mob` key.
- `to_schema.py:1552` hardcodes `A_mob_available=False` as a literal.
- No baseline accepts a second adjacency: ColaGNN `baselines/colagnn/src/models.py:27`,
  HeatGNN `baselines/HeatGNN-14DB/src/models.py:39`, MTGNN `baselines/MTGNN/net.py:11`.
  `export_baseline.py:86` writes exactly one `adj.txt`.
- EpiGNN's `--extra` hook (`baselines/EpiGNN/src/models.py:46-50,122-139`) is the only second-adjacency
  path in the repo and it is unusable here: `baselines/EpiGNN/src/data.py:56-67` expects a directory of
  per-timestep CSV edge lists and `models.py:131` indexes it by time. It wants a time-varying OD tensor,
  not a static matrix.

Forking the four baselines to accept a second graph would void the client's reproduction-validation
condition (`Review Doc.md:73`), because a forked model is no longer the published model.

### 1.2 Raw gravity with a single global beta CREATES a stronger disease tell than the one it removes

`models/encoder.py:62` feeds `LTR(log1p(deg))` — a `Linear(1,64)` on node degree — directly into the
trunk, where the gate MLP then reads it.

| bundle | log1p(median degree) today | log1p(median degree), raw gravity |
|---|---|---|
| dengue | 2.08 | — (dense) |
| ebola | 1.95 | 10.84 |
| influenza_japan | 1.79 | 10.47 |
| influenza_us-states | 1.79 | 14.01 |
| influenza_us-regions | 1.70 | 14.81 |

Today the degree channel spans 0.38 nats across all five bundles and carries no information about which
disease is being processed. Under gravity it spans 4.3 nats and separates the bundles cleanly. That is a
disease identifier wired straight into the shared trunk, which is the exact failure the proposal existed
to prevent. `encoder_architecture_plan.md:249` already mandates per-graph normalisation of edge weights
for this reason; the "single fixed global beta" formulation omits it.

### 1.3 The per-node scaler already deletes gravity's payload

Gravity's information content over binary contiguity is population-weighted magnitude. The released
scaler is `scaler_scope="per_node_train"` (`to_schema.py:1552`), which removes cross-node level before
the mixer runs. Measured variance of node-level means of `X[:,:,0]`:

| dengue | ebola | japan | us-regions | us-states |
|---|---|---|---|---|
| 0.948 | 1.011 | **0.0016** | 0.0216 | 0.0657 |

`encoder_architecture_plan.md:193-199` predicted exactly this: per-node normalisation "erases exactly the
cross-node level gradient the spatial channel exists to propagate." The dengue/ebola variance near 1.0 is
largely an artefact of zero-fill (§3.2), not epidemiology.

### 1.4 The graph is already mostly empty at training time

`mask_aware_adj` (`models/encoder.py:63`) weights edge *(i,j)* by `M_t[j]`, so unobserved neighbours drop
out. Mean fraction of a node's neighbours observed at a training origin:

| dengue | ebola | influenza |
|---|---|---|
| 0.224 | 0.660 | 1.000 |

**67.3 per cent of dengue nodes have zero observed neighbours at a typical origin**, i.e. the spatial
channel is a self-loop for two thirds of the largest panel. Ebola 9.7 per cent, influenza ~4 per cent.
Improving the graph on top of that is bounded above by very little.

### 1.5 Nobody has established that the graph contributes anything

Of 2,584 encoder result records in `results/` and `ablation/`, **zero** are at `gate_mode='off'`. The
shuffled-adjacency negative control (`encoder_architecture_plan.md:472-473`) has never been run. What
exists shows only that the gate is open (g 0.238 japan → 0.682 dengue), not that opening it helps — and
the gate is highest on the panel where the graph is emptiest, which is what you would expect if
`Gate(h)` is reading `LTR(log1p deg)` rather than anything epidemiological.

### 1.6 Cost, against the remaining schedule

| | Days | Notes |
|---|---|---|
| Data | 3.5–4 | WorldPop is not in the repo (`data/raw/` empty; the only population data anywhere is a hardcoded 47-element list at `japan_nodes.py:114`). Raster acquisition for ~15 countries, zonal statistics over 7,332 GADM polygons, validation suite, DVC bundle regeneration. |
| Model | 2–2.5 | Dual-relation forward, fusion design, `LTR` degree-source decision, 16 encoder call sites across `train/loop.py`, `train/joint.py`, `train/lodo.py`, second block-diagonal in `block_diag_sparse`, new invariant gates. +3 days if baselines are forked. |
| Rerun | 3–5+ | All 2,584 encoder records invalidated. Measured per-origin cost at 5M nnz is 55× current; dengue LODO goes from 2–3 h to roughly 24 h per seed. |
| **Total** | **9–12** | At day 15 of 30, with meta-learning already declared not to fit (`Meta_Learning_Decision.md:139`). |

### 1.7 Corrections to claims made during the discussion

- **`A_geo` is binary** (all nonzeros exactly 1.0) in every bundle, but `models/spatial.py:34` carries edge
  values through `dinv[idx[0]] * val * dinv[idx[1]]`. Weights are **not** binarised on entry, so the
  earlier worry that gravity weights would be destroyed by the mixer was wrong.
- Message passing is **sparse** (`torch.sparse.mm`, `models/spatial.py:74`). No dense `[7165,7165]` matmul
  happens anywhere. The cost objection stands but by a different mechanism (55× measured, not OOM).
- **`C` is never fed to the encoder** — `tests/test_encoder_invariants.py:59-62` asserts that passing `C=`
  raises `TypeError`. So "gravity is just `C` rearranged" is weak: gravity would be genuinely new
  information to the trunk. That is the problem rather than the reassurance, because it launders `C` into
  the trunk through the adjacency, past a gate that only inspects the function signature.
- `A_geo_kind` for dengue is `"queen+knn_block_diagonal"`; ebola is `"queen+knn_cross_border"`; the three
  influenza sets are `"shipped(diag_zeroed)"`.
- `harness/` and `schema/` are empty directories.

### 1.8 The one argument that survives for the proposal

Dengue's graph is known mis-specified: block-diagonal with zero cross-border edges, on a premise the plan
itself marks False (`encoder_architecture_plan.md:267-269` — Brazil borders four of the twelve countries,
the Triple Frontier among them). A non-block-diagonal kernel would give dengue, 98.5 per cent of training
nodes, its first cross-border edges.

**If that is worth fixing, the cheap version is a distance-decay kernel, not gravity:**
`A_ij = exp(-d_ij / median(d))` from the GADM 4.1 centroids already in `C[:, :2]` in all five bundles,
plus symmetric k-NN augmentation, with per-graph median normalisation per
`encoder_architecture_plan.md:244-251`. About 0.5 day of data work instead of 4, no WorldPop, no new
external dependency, and it isolates the distance term from the population term so any gain is
attributable. `build_datasets.py:151-153` already has the `if dt.A_mob is not None` branch.

**Flag before doing it:** every bundle marks `covariates_transfer_safe: False`, and C1 /
`test_no_covariates` forbids `C` reaching the trunk. Edge weights are relational rather than node-indexed
and per-graph normalisation removes absolute scale as a tell, but this needs a recorded decision and an
amendment to the §8 gate, not a quiet import.

---

## 2. Confirmed wins

**W1. The encoder already beats the published GNNs on the common pipeline** (5-seed means, RMSE):

| panel / horizon | ours | Cola-GNN | EpiGNN |
|---|---|---|---|
| influenza_japan h3 | **735** | 804 | 876 |
| influenza_us-regions h15 | **813** | — | 954 |
| influenza_us-states | ≈ | — | ≈ |

At **142,305 trunk parameters** (`capacity_probe.log:95`), ~87 per cent of which sit in the temporal
branch.

**W2. It wins while handicapped.** `export_baseline.py:8` hands the baselines raw counts, so they fit
their own scaler under MSE and are mean-optimal in count space. The encoder optimises pinball over
quantiles in log1p + per-node z-score space and is median-optimal. The head-to-head is not like-for-like
on the target transform, and the handicap runs against us. Fixing §3.1 should widen the margin we already
hold.

**W3. Capacity is not the binding constraint.** `Reports/Capacity_Probe_Result.md`: a 15× larger read-out
gains **+0.2 per cent median in-domain** and up to +23.2 per cent cross-disease. The trunk is right-sized
for its own disease; the deficit is transfer-specific. This is a vote for ANIL / representation shaping
and against buying more parameters.

**W4. We now have a mechanism for the negative-transfer result** (§3.2), which was previously
unexplained. That converts the headline from a mystery into a testable, fixable hypothesis.

**W5. Population genuinely is predictive.** `japan_nodes.py:17-19` measures Spearman **ρ = +0.949**
between mean ILI and prefecture population. The instinct behind wanting mobility is sound; it is the
per-node scaler, not the absence of signal, that is currently blocking it.

**W6. The go/no-go experiment is already coded and free.** `gate_mode="off"` is implemented at
`models/encoder.py:19-24` and `tests/test_encoder_invariants.py:69-77` verifies exact graph-free nesting.
Five seeds × four panels is one night, and it settles the entire graph question before any further
investment.

**W7. Roughly 9–12 engineering days preserved**, on a project with 15 days left and a required
deliverable already declared not to fit.

---

## 3. Open bugs, and what each one costs

Ordered by consequence.

### 3.1 The objective predicts the count-space median; the metric rewards the count-space mean

Quantiles are equivariant under `expm1`, so `τ=0.5` in log space inverts to the **median** in count
space. RMSE is minimised by the **mean**. Ratio `mean(y) / expm1(mean(log1p y))` over observed train
cells:

| panel | median node | p90 | max |
|---|---|---|---|
| influenza_japan | **19.2×** | 23.9 | 30.1 |
| dengue | 4.2× | 24.5 | 2236 |
| ebola | 2.7× | 20.8 | 73.4 |
| influenza_us-states | 1.6× | 2.7 | 4.0 |

Fingerprint in the results: the encoder's RMSE/MAE ratio exceeds every naive floor's on every panel
(japan h3 2.81 vs seasonal 2.30 vs persistence 2.27). On Japan the encoder is within 10 per cent of
seasonal-naive on **MAE** (261 vs 238) while 34 per cent worse on **RMSE** (735 vs 547).

**Cost:** every RMSE and `peak_intensity` figure reported so far is depressed. `peak_intensity` reads
2,717 on a panel whose observed mean is 655. We were on track to report losing to a seasonal naive
baseline that we are not actually losing to.

**Fix:** fit a per-(panel, horizon) log-space bias correction `c` on the **val fold only**, report
`expm1(σ·(z_med + c) + μ)` for RMSE and peak intensity, keep the median for MAE and say so. Cheapest
first probe: archive quantiles, then re-score at τ=0.75. ~15–40 lines across `train/loop.py:137-139`,
`score.py`, `rescore_encoder.py`.
**Ebola caveat:** Ebola has no val split (C8), so its `c` must be inherited from the dev diseases and
pre-registered before the single Ebola scoring.

### 3.2 Unobserved input cells are zero-filled, which means "at this node's training mean"

Fraction of `X[:,:,0]` exactly zero:

| dengue | ebola | japan | us-regions | us-states |
|---|---|---|---|---|
| **78.2%** | **59.0%** | 0.0% | 0.0% | 0.0% |

A TCN whose gated activations were fitted where 78 per cent of inputs sit at exactly 0 with mask=0 is
then run where 0 per cent do and mask ≡ 1. The 1,428-parameter affine adapter cannot undo a covariate
shift of that size, and the capacity probe confirms it: `mlp-256` (21,780 params) recovers +23.2 per cent
at best, median +2.5 per cent.

**Cost:** this is the leading mechanical candidate for the zero-shot collapse (−311.2 per cent japan h3,
−558.8 per cent h10 from a dengue trunk) and therefore for the "transfer NEGATIVE in 12 of 16 cells"
headline. If it is the cause, the Option B strategic pivot was decided on contaminated evidence and the
true transfer result is unknown until this is rerun.

**Fix:** last-observation-carry-forward on channel 0, `obs_mask` still flagging it. No new channels, so
the C1 test is unaffected, and it is uniformly available across diseases. ~10–20 lines in
`bundles.py::transfer_view` or a trainer-side preprocessor.

### 3.3 `obs_mask` is a disease identifier inside the "disease-agnostic" core block

Channel 3 is constant 1.0 on all three influenza panels and varies on dengue and ebola.

**Cost:** the disease-agnostic claim is already partially false in the four core channels, independently
of anything to do with mobility. We were about to spend ten days protecting a property that leaks
elsewhere. This must be named in the limitations either way.

### 3.4 The reference arm has no saved weights and no saved quantiles

`train/loop.py:346-352` — `run_dataset` calls neither `write_checkpoint` nor `write_quantiles`. The
single-disease 5-seed runs are the reference arm for every delta in the manuscript.

**Cost:** any re-probe requires a full retrain, and G4 (calibrated uncertainty) is structurally
unevidenced — no WIS, CRPS, PICP or PIT can be computed without the quantiles.
**Fix:** call the two functions, then seed-ensemble the archived quantiles at inference rather than
averaging metrics. ~20–30 lines. Also cuts the 8–14 per cent seed noise floor, which is currently the
same order as every effect the paper is trying to measure (japan h15 sd 121 on 1031; us-regions h3 82 on
613; dengue h3 6.04 on 42.06).

### 3.5 Gradient and metric weight nodes ~700× differently

Loss is uniform over nodes in per-node z-space; the metric is count-space, where a node's sensitivity is
`exp(μ_i)·σ_i`. On dengue the top 1 per cent of nodes hold 30.2 per cent of count-space leverage
(max/median 698×), and on `dengue seed42 h3` **the top 1 per cent of nodes carry 95.1 per cent of total
SSE, the top 10 per cent carry 99.5 per cent** (median node RMSE 5.48 vs mean 23.05).

**Cost:** training optimises 6,161 equally-weighted nodes while the score is decided by roughly 60 of
them.
**Fix:** node weights `w_i ∝ exp(μ_i)·σ_i` from the **train-fold scaler only**, normalised to mean 1 per
graph, clipped at p95. The `w` argument already exists at `models/adapters.py:41`. Ebola's scaler is
`per_disease_support`, so its weights are uniform — state that.

### 3.6 Two planned figures are currently uninterpretable

- The gate figure (Work Order §8e) is at risk of showing graph density rather than disease geography,
  because `LTR` injects a time-constant per-node bias at `models/encoder.py:62` that the gate MLP then
  reads, and within a panel the gate is nearly constant. Re-run the gate readout with `LTR` zeroed before
  publishing it.
- The §3.1 mandatory diagnostic was never logged. Values are now measured (§1.3). By the plan's own rule
  (`encoder_architecture_plan.md:193-199`), a "graph adds nothing" result on influenza is currently an
  artefact of per-node normalisation, not a finding.

### 3.7 Documented protocol does not match shipped code

- `Bundle.refit()` (`bundles.py:52`) is called by nothing except `bundles._refit_check()`. All three
  trainers use the frozen headline scaler. Not a leak (it is train-fit), but C4 as written never ran, and
  `test_scaler_refit_per_origin` cannot be testing the trainer.
- `encoder_architecture_plan.md` §7 still states 3 layers / dilations `[1,2,4]` / RF 15. Shipped is
  `models/config.py:6` `DILATIONS=(1,2,4,8,16)`, 5 layers, RF 32. The stale table will be quoted into the
  manuscript if nobody fixes it.
- The `--norm {graph,node,none}` ablation (`encoder_architecture_plan.md:465`) is not implemented
  anywhere.

### 3.8 Wasted compute on every step

`SharedEncoder.forward` computes `normalise_adj(A)` and runs `(deg > 0).all()` and
`torch.isfinite(...).all()` on every forward, but `A_hat` is discarded whenever `M_t` is supplied — which
is always, in all three trainers. On dengue that is a ~41k-edge sparse construction plus two GPU→CPU
syncs per step, across 91k-step trunk runs.

---

## 4. Near miss worth recording

Had the gravity adjacency shipped as specified, `log1p(degree)` would have moved from a 1.70–2.08 band
(indistinguishable across diseases) to a 10.47–14.81 band (perfectly separating), installing a disease
identifier in the shared trunk. That would have invalidated the cross-disease transfer claim outright,
after roughly ten days of work and a full rerun of 2,584 encoder records. Caught in review, before spend.

---

## 5. Next steps, in order

1. **Overnight, free:** run `gate_mode='off'` and a row-shuffled `A_geo`, 5 seeds × 4 panels. This is a
   go/no-go on all further graph work, including the distance-decay variant in §1.8. If shuffled ≈ real,
   the spatial channel is decorative and that is a publishable honest negative.
2. **Fix §3.1** (median → mean bias correction) and **§3.4** (archive checkpoints + quantiles). Both are
   small, both are reversible, neither touches the frozen `.npz` bundles or the 86 data gates.
3. **Fix §3.2** (LOCF instead of zero-fill). This is the transfer fix, and it is upstream of ANIL — it
   makes the meta-learning arm a fair test rather than a rescue mission.
4. **Acquire COVID-19 data.** Primary: NYT `us-states.csv`
   (`https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv`) — one file,
   `date,state,fips,cases,deaths`, 2020-01-21 → 2023-03-23, archived and stable, **cumulative**, so the
   existing Ebola `cummax().diff()` + mask routine applies as-is. It reuses the 49-node
   `influenza_us-states` node set and ColaGNN's shipped adjacency, so no GADM work. Secondary: COVID at
   Japan prefectures against the existing 47-node `influenza_japan` set.
   *Scientific value:* same graph, same nodes, two diseases — isolates disease transfer from graph
   transfer, which none of the current cells do.
   *Caveat to disclose:* COVID 2020–2022 is NPI-dominated, so transfer to or from it may reflect policy
   response rather than pathogen dynamics.
5. **Re-run cross-disease transfer / LODO with the bugs fixed**, across dengue, influenza and COVID. The
   existing 12-of-16-negative result is not trustworthy until §3.2 is corrected, so this rerun is what
   determines whether Option B stands.

`A_mob` stays `None` on all bundles throughout. Uniformly absent is uniformly non-identifying, which is
exactly the property the gravity proposal was trying to buy.

---

## 6. Progress log

### 2026-08-03 — step 2 done (§3.1 + §3.4), and a new blocker found

**§3.4 fixed.** `train/loop.py` now writes the quantile archive and the trunk/adapter checkpoint on
every single-disease run. `write_quantiles` and `write_checkpoint` already existed and were already
routed by `results_paths` — `train_one` simply never produced the arrays and `run_dataset` never called
them. Wired via the `quant_out` / `run_out` out-parameter idiom `train/lodo.py:311-340` already uses, so
the four existing `recs, pn, po, gate = train_one(...)` call sites keep working unchanged.
Verified on the smoke run: `h3__quantiles` `[47, 104, 5]` at levels `(0.05, 0.25, 0.5, 0.75, 0.95)` over
104 test origins, and a checkpoint holding **142,305** encoder parameters, matching
`capacity_probe.log:95`. WIS/CRPS/PICP/PIT are now computable, so G4 has evidence available for the
first time.

**§3.1 fixed.** `_fit_bias_correction()` fits one log-space offset `c` per horizon by grid search on
the **val fold only**, minimising count-space RMSE. `c=0` is in the grid, so the fitted correction can
never be worse than uncorrected on the fold it was fitted on.

The corrected forecast ships as a **second model arm, `encoder_mc`, alongside the untouched `encoder`
arm in the same results file**, carrying `bias_c` on every record. It does not overwrite the median.
Rationale: the median is the calibrated forecast and the right point estimate for MAE; the corrected
one is the RMSE / peak-intensity point estimate. Shipping both lets the paper show the gap instead of
silently restating every number already reported.

Smoke evidence (influenza_japan, seed 42, **5 epochs — undertrained, directional only**):

| arm | RMSE h3 / h5 / h10 / h15 | MAE h3 / h5 / h10 / h15 |
|---|---|---|
| `encoder` | 721.9 / 839.0 / 1124.4 / 1212.0 | 245.0 / 307.2 / 458.6 / 510.4 |
| `encoder_mc` | **549.5 / 695.5 / 1070.1 / 1021.6** | **207.9 / 267.0 / 436.7 / 446.6** |
| `bias_c` | 0.15 / 0.10 / 0.05 / 0.20 | — |

RMSE at h3 falls 24 per cent. **Note against the prediction in §3.1: MAE improved too**, which it should
not if the median were already MAE-optimal. On a 5-epoch model the median is simply low, so both metrics
move together; on a fully trained model MAE may well go the other way. That is precisely why both arms
ship rather than one replacing the other — the question is now answerable from the records instead of
being assumed.

**Checks.** `python -m train.loop --selfcheck` runs a new `_bias_selfcheck()`: on a synthetic lognormal
target with sigma=1.2 the RMSE-optimal constant is the mean, so the fit must return c ≈ 0.72/1.2 = 0.60
(it returns 0.55) and must lower RMSE; on a target with no skew it must return exactly 0.0. Both arms of
the check were **mutation-tested** — collapsing the grid to `[0.0]` and removing `0.0` from the grid each
make it fail. The first mutation attempt did **not** fail, because `grid=BIAS_GRID` bound the constant at
def time; `_best_offset` now resolves the grid at call time so the constant is genuinely the knob.

### RESOLVED 2026-08-03 — `ebola-train` repaired, recorded versions intact

**Root cause, narrower and worse than first reported.** It was not `np.corrcoef` specifically: **every
BLAS call** in that env hard-crashed. `np.arange(1e6).sum()` (no BLAS) returned fine; `a @ a` on a 4×4
died with the same `0xC06D007F`. The env's numpy is the **conda-forge** build linked to conda-forge
MKL through the `libblas` shims:

```
libblas   3.11.0  8_h8455456_mkl      mkl  2026.1.0  (ships mkl_rt.3.dll)
libcblas  3.11.0  8_h2a3cdd5_mkl
liblapack 3.11.0  8_hf9ab0e9_mkl
```

MKL 2026.1.0 moved to versioned `mkl_rt.3.dll` / `mkl_core.3.dll`; the `build 8_*_mkl` shims resolve
against the older export set, so the ordinal lookup fails at first use. `ebola` was never affected
because its numpy is the **pypi** wheel, which bundles its own OpenBLAS and never touches the conda
MKL stack.

**Fix — swap the BLAS provider, change nothing else:**

```
conda install -p C:\Users\Administrator\miniconda3\envs\ebola-train -c conda-forge "libblas=*=*openblas"
```

Solve was minimal: `libblas` / `libcblas` / `liblapack` revised `*_mkl → *_openblas`, `libopenblas
0.3.33` added, `ca-certificates` bumped. Nothing else moved.

**Recorded versions preserved** — `numpy 2.4.6`, `scipy 1.17.1`, `torch 2.6.0+cu124`, CUDA available.
`data/processed/env_train.txt` therefore remains accurate (it records pip-visible versions, and none
of them changed); the BLAS provider is not captured there, which is itself worth knowing.

**Verified after the fix:** `a @ a` and `np.corrcoef` return; `score.score_bundle` (the call that
died) completes; all 6 bundles pass; `train.loop --selfcheck` passes all four checks;
`train.loop --smoke` completes in 17 s having previously killed the process.

**The env question is also answered empirically.** Same smoke, same seed, both envs:

| arm | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| `encoder` | +0.02% | +0.17% | +0.19% | +0.07% |
| `encoder_mc` | +0.02% | +0.19% | +0.18% | +0.09% |

`ebola-train` vs `ebola` differ by **at most 0.19%**, against a documented seed sd of 8–14%. `bias_c`
came out identical (`{3: 0.15, 5: 0.10, 10: 0.05, 15: 0.20}`). The torch difference (2.6.0+cu124 vs
2.12.1+cu126) is therefore not a re-baselining risk at the precision anything is reported to. **Run
the overnight job in `ebola-train`**, the recorded env.

**Rollback:** `env_train_snapshot_20260803.txt` at the repo root holds `conda list --explicit` plus
`pip freeze` taken immediately before the change.

### (superseded) BLOCKER: the recorded training environment is broken

`ebola-train` — the env recorded in `data/processed/env_train.txt` — **hard-crashes the Python process**
(exit `0xC06D007F`, no traceback) inside `np.corrcoef`, i.e. in `score._pcc` at `score.py:86`, reached
from `score.per_node_scores` (`score.py:250`) via `score_bundle` (`score.py:285`). Reproduced with
**numpy alone, no torch imported**, so it is a broken BLAS in that env, not the documented OpenMP clash
that `KMP_DUPLICATE_LIB_OK` handles.

| env | numpy | `np.corrcoef` | torch |
|---|---|---|---|
| `ebola-train` | 2.4.6 | **crashes the process** | 2.6.0+cu124, CUDA OK |
| `ebola` | 2.2.6 | OK | 2.12.1+cu126, CUDA OK |

Consequence: **any scoring run launched in `ebola-train` will die after training completes**, having
thrown away the trained model — which is exactly the failure §3.4 was written to prevent, and it would
have burned a full overnight before anyone noticed. The step-1 gate/shuffled-adjacency controls must not
be launched until this is resolved. The smoke above was run in `ebola` instead.

Open question before rerunning anything: the two envs differ in torch as well (2.6.0+cu124 vs
2.12.1+cu126), so switching envs is not numerically neutral against the existing 2,584 records. Either
repair numpy inside `ebola-train` (preferred, keeps the recorded env) or re-baseline in `ebola` and say
so. Not decided.

### 2026-08-03 — §3.2 fixed (LOCF), and the COVID bundle is built

**§3.2 fixed.** `bundles._locf()` + `transfer_view(locf=None)`, switched by the `LOCF_INPUT` module
constant (env `LOCF_INPUT=0` restores zero-fill). All three trainers call `transfer_view()` and
nothing else, so one switch covers `train/loop.py:124`, `train/lodo.py:160` and `train/joint.py:82`
with no signature change anywhere.

Inputs only: `y`, `raw`, `M` and every loss/eval mask are untouched, no channel is added (C1 gate
unaffected), and `obs_mask` still flags every filled cell. Strictly causal — cells before a node's
first observation have nothing to carry and stay at 0.

Verified (`python bundles.py`): **dengue exact-zero input mass 78.2% → 59.2%**, matching the audited
78.2% exactly. It does not reach 0% because dengue nodes enter at different dates and leading gaps
have nothing to carry forward; that is correct behaviour, not a shortfall.
Mutation-tested: a no-op fill and a *forward-peeking* fill are both caught. The causality arm is the
one that matters — a forward fill is leakage dressed as imputation and would pass every shape check.

**A/B probe: `locf_probe.py`** (repo root). Two arms, same seed / steps / epochs, differing only in
`bundles.LOCF_INPUT`; trunk on dengue, then flu zero-shot and adapted. It calls `train.lodo`
internals directly and **writes nothing to `results/`** — `run_ldo_fold()` would overwrite the
canonical `encoder_ldo__*` records the held 5-seed matrix lives in. The influenza panels are 100%
observed, so LOCF is a no-op there and every difference comes from the dengue trunk's input
distribution, which is the hypothesis.

### RESULT: the §3.2 hypothesis is NOT supported. LOCF is now default OFF.

seed 42, 3000 trunk steps, 15 adapter epochs, 7.3 / 7.5 min per arm:

| arm | LOCF better | mean change | worst cells |
|---|---|---|---|
| adapted | 7/12 | **−0.9%** | us-states h15 +3.1% |
| **zeroshot** | **1/12** | **+4.9% WORSE** | japan h3 **+29.7%**, h5 **+12.3%** |

The zero-shot arm is the one the −311% collapse was measured on, and the only arm this change was
meant to fix. It got worse in 11 of 12 cells. The adapted arm moved −0.9%, which is noise against the
documented 8–14% seed sd. **The prediction in §3.2 was wrong.**

**Plausible mechanism, which is why this is a finding rather than a failed patch.** Zero-fill puts an
unobserved dengue cell at that node's training mean; LOCF puts it at the last observed value and
*holds* it. On a series that is 78% gaps, that converts mean-reverting gaps into long flat runs at a
carried level. The influenza panels are dense and oscillating, so LOCF plausibly moved the dengue
input distribution **further** from flu, not closer. If that is right, the transfer barrier is
dengue's **gappiness itself**, and neither fill repairs it — which is a more useful thing to know
than "the fill was wrong", and it points the next attempt at the input representation rather than at
the imputation value.

**What changed:** `bundles.LOCF_INPUT` now defaults to `"0"`. `LOCF_INPUT=1` turns it back on. The
code and the self-checks stay in as a switchable ablation arm — a tested-and-rejected hypothesis with
numbers is worth more in the paper than a silent deletion.

**Caveat on the strength of this result.** Single seed, and the trunk ran 3,000 of 91,000 steps
(3.3%). The covariate-shift argument is about a *converged* trunk's fitted activation regime, so an
undertrained trunk is not a clean test of it. japan h3 at +29.7% is well outside seed noise, but
+3.2% and +2.1% cells are not. A 3-seed replication is ~45 min at these settings and would settle it.
**It should not block the overnight run** — the default is now the validated behaviour either way.

### COVID-19 bundle: built, gated, registered

`covid_load.py` → `data/processed/covid_us-states.npz`. **`X=(49, 164, 4)`, 2020-02-01 → 2023-03-18**,
origins 130 (train 60 / val 45 / test 49).

Source is NYT `us-states.csv` (archived 2023-03-23), sha256
`0b202f6ac8bad66b70b9e344c07155e3f1ce9338f86192af072b56ce448dfc43`. Cumulative → incidence uses the
**same mass-preserving `cummax().diff()` as Ebola** (§5 of the schema spec), checked against the
source column independently of the transform. Nothing is masked: NYT reports every state every day
once its first case lands, and absence before that is a genuine zero, so `M` is all-ones — which also
makes LOCF a no-op on COVID.

It reuses the **ColaGNN 49-state node set and `state-adj.txt` verbatim**, so it needed no new
geography, no new adjacency and no alias map. Gated on build:

- **Omicron anchor pin**: national max lands on **2022-01-15 at 13.8× the median week**. The matrices
  carry no dates, so the calendar is an assertion — this plays the role the H1N1 week plays for
  us-regions. A winter-peak seasonality gate was deliberately *not* copied over: COVID is not
  seasonal and that check would be meaningless here.
- Off-diagonal edges bit-for-bit identical to the shipped graph; symmetric; diagonal zeroed.
- `A_geo` and `C` **identical to `influenza_us-states`**, node ids **disjoint** from it, largest-area
  node is Alaska at index 1.
- Core channels and `transfer_view` shape identical to every other bundle.
- `dt.check()` passes. Encoder invariants pass across all six bundles, with COVID's self-loop-
  inclusive degree profile (min 1 / median 5 / max 9 / mean 5.20) exactly matching influenza_us-states
  — the shared graph, confirmed numerically.

**Why this node set:** COVID and influenza on an identical graph with identical covariates means a
transfer result between them isolates DISEASE transfer from GRAPH transfer. Every existing
cross-disease cell confounds the two.

**Costs, stated not hidden:** the 49-state set excludes **Florida** (ILINet does not report it, so
ColaGNN dropped it) and DC — Florida is ~6.5% of US population with a distinctively-timed burden, but
a 50th node would fork the adjacency and forfeit the identical-graph property that is the whole
point. The graph is land contiguity built for ILI; it is not a COVID-specific graph and must not be
described as one. COVID 2020-2022 is NPI-dominated throughout (`meta.npi_confounded = True`).

**EDA: [`covid_eda.md`](covid_eda.md)** (regenerate with `covid_eda.py`). Headlines:

- **Missingness is nil and needs no modelling.** `M` density 1.000000, zero unobserved cells, 3.4%
  exact zeros of which almost all are the pre-outbreak head (median 5 leading weeks). No reporting
  dropout in the 2022–23 tail — worst state is Nebraska at 6% zero weeks. LOCF is a no-op here.
- **`obs_mask` is constant 1.0**, so COVID joins the *influenza* side of the §3.3 channel-3 disease
  identifier. It does not add a new leak, and it does not fix the existing one.
- **Largest mean/median gap of any panel: 3.03× median node** (influenza_us-states 1.74× on the same
  nodes), so the §3.1 `encoder_mc` arm should matter more on COVID than anywhere else.
- **Cross-node level gradient 0.0090**, second flattest after japan and 7× flatter than
  influenza_us-states on the *same 49 nodes* — the spatial channel has even less to carry here.
- **Node rankings differ usefully from influenza**: Spearman 0.695 on node totals (New York 3rd by
  COVID, near-bottom by ILI). With graph and covariates bit-identical, a COVID↔influenza transfer cell
  isolates disease transfer from graph transfer. No existing cell can do that.

**Correction to what was written above.** The "train in wild-type/Alpha, test in Omicron+" warning was
wrong on the magnitude. The **train→test national level ratio is 1.03×** — the two folds sit at
essentially the same scale. The regime shift is **inside the val fold**, which owns the Omicron peak:
val runs at **2.8× both other folds** and its peak week (5.14M) is **6.3× the largest test week**.
That compromises two things for this bundle specifically: early stopping selects on it, and the new
§3.1 `bias_c` is *fitted* on it and then applied to a test fold roughly a sixth of that scale, so it
will over-correct. `covid_load.py`'s printed NOTE has been corrected to say this.

**Recommendation:** keep `fixed_50_20_30`. Protocol consistency across diseases is worth more than a
better-conditioned COVID split, and moving the cut for COVID alone would make its transfer cells
incomparable to every other cell. Disclose the val composition and treat COVID's `bias_c` and
early-stopping choice as known-fragile. If one thing changes, fit COVID's `bias_c` on train rather
than val and say so — do not move the split.

**Open, do not quote yet.** The §1.3 node-mean-variance statistic reads **0.948** for dengue in the
review above and **0.0234** when recomputed here, while japan (0.0016) and us-states (0.0657) match
exactly. Most likely a mean-over-all-T versus mean-over-observed-cells difference, which matters a lot
on a panel that is 78% zeros. Both are defensible statistics; only one belongs in the paper.

Registered in `bundles.BUNDLE_NAMES` and `analysis.DEV`. `FLU_NAMES` is an explicit tuple in
`train/lodo.py:224`, so COVID does not leak into the influenza group.

**`to_schema.load_influenza` gained two optional kwargs** (`disease`, `covariates_as`), three lines,
both defaulting to the previous behaviour exactly, so a non-influenza panel on the same node set and
shipped graph can reuse the loader instead of forking it.

### DECISION NEEDED before the overnight LODO

1. ~~**Which env.**~~ **RESOLVED** — `ebola-train` repaired (BLAS provider swapped to OpenBLAS,
   versions unchanged) and measured to agree with `ebola` to within 0.19%. Use `ebola-train`.
2. ~~**LDO directions.**~~ **RESOLVED** — three-disease fold implemented, see below.
3. ~~**Whether the LOCF arm is the default.**~~ **RESOLVED** — default OFF, hypothesis rejected.

### 2026-08-03 — three-disease leave-one-disease-out (`ldo3`)

The old `LDO_DIRECTIONS = ("flu2dengue", "dengue2flu")` is a two-disease split and cannot express
"hold out COVID". With three diseases there are three **folds**, not two directions.

`train/lodo.py` now carries both. The old two-direction code and its 84 `encoder_ldo__*` records are
untouched — the client asked for both fold structures reported separately (D1), and the new fold
writes under a distinct `encoder_ldo3__` prefix.

```python
DISEASES = {"dengue":    ("dengue",),
            "influenza": ("influenza_japan", "influenza_us-regions", "influenza_us-states"),
            "covid":     ("covid_us-states",)}
```

| held-out disease | trunk trains on | adapter_groups | held-out adapter scope |
|---|---|---|---|
| dengue | flu ×3 + covid | `[0,0,0,1]` | dengue |
| influenza | dengue + covid | `[0,1]` | all 3 flu bundles, one adapter |
| covid | dengue + flu ×3 | `[0,1,1,1]` | covid |

**Three design points, each of which was a bug waiting to happen:**

1. **`_fit_trunk` gained `adapter_groups`.** Neither `share_adapter=True` nor `False` can express the
   new case: when the trunk trains on dengue *and* influenza at once, dengue needs its own FiLM head
   while the three flu bundles must share one. `[0,1,1,1]` says exactly that. The two booleans are
   now the all-zeros and all-distinct special cases; passing both raises.
2. **`_mean_adapter` now dedupes by identity.** `_fit_trunk` returns one entry per in-*dataset*, so a
   shared head appears three times for influenza. Averaging the raw list would have weighted
   influenza 3/4 against dengue 1/4 — weighting a disease by how many bundles it happens to own,
   in a fold whose entire premise is that the disease is the unit. Single-source stays an exact
   identity, as the two-way fold documents.
3. **Every fold uses `_fit_shared_adapter`**, even where the held-out disease owns one bundle. This
   departs from `run_ldo_fold`, which uses `_fit_adapter_and_score` on its single-bundle side. Mixing
   the two here would make "hold out COVID" and "hold out influenza" differ by *fitting protocol* as
   well as by data. The protocols are near-identical anyway; comparability across the three folds is
   worth more than matching the old fold exactly.

Held-out influenza emits **three per-dataset rows that are never pooled** (D3): the bundles differ 60×
in cells and have disjoint calendars, so a mean over them would be arithmetic, not a measurement.

**Checks — `python -m train.lodo --smoke-ldo3`.** Asserts `DISEASES` partitions `DEV_BUNDLE_NAMES`
with no bundle in two diseases, that `DISEASES["influenza"]` has not drifted from `FLU_NAMES`, the
exact plan for all three folds, that no bundle is both in-trunk and held-out, C8, that the flu
bundles carry one group id in every fold, that `adapter_groups` produces shared-within/distinct-across
parameter sets, that the zero-shot head averages diseases rather than bundles (with a negative
control requiring the naive and deduped means to differ), and that single-source is an identity.
`--smoke` and `--smoke-ldo` still pass unchanged; `results_paths` routes 23 families including the
four new `ldo3` ones.

**End-to-end verified** on a throwaway `--ldo3 covid --trunk-steps 60 --epochs 2 --seed 999`: 7
artifacts routed to `results/lodo/`, 28 adapted + 28 zero-shot records, `fold_structure =
leave-one-disease-out-3way`, `held_out_disease = covid`, `in_diseases = dengue,influenza`, quantiles
`[49, 49, 5]`, checkpoint carrying the plan. Artifacts deleted afterwards; the 84 existing
`encoder_ldo__*` files were confirmed untouched.

### The overnight run

```
conda activate ebola-train
python -u -m train.lodo --all-ldo3 --seeds 42 2>&1 | Tee-Object results/reports/ldo3_run.log
```

Env is `ebola-train` (repaired). `LOCF_INPUT` defaults to off, which is the validated behaviour.

**Budget honestly:** extrapolating from `locf_probe.py` (dengue trunk, 3,000 steps ≈ 5 min), a full
91,000-step trunk is ~2.5 h where dengue is in the trunk, less for the hold-out-dengue fold whose
trunk is four small bundles. Early stopping (patience 12 × val_every 1000) may cut this short. One
seed across three folds is therefore roughly **5–8 h — an overnight run**. Five seeds is not: that is
25–40 h and needs to be spread across nights or run at reduced trunk steps. Start with seed 42, add
`--seeds 42 52 62 72 82` once the per-fold wall-clock is known.

### 2026-08-04 — graph-controlled pair run: results

Full write-up in [`Pair_Run_Analysis.md`](Pair_Run_Analysis.md); regenerate with
`python compare_runs.py --metric rmse`. Ran 2026-08-03 22:13→23:01 (48 min), 5 seeds, both
directions. 10 adapted + 10 zero-shot records, plus the COVID ceiling (5 seeds) and naive floors.

**Outcome 1 — a clean negative, and for the first time an attributable one.** Paired per seed against
each disease's own ceiling, RMSE, positive = worse:

| direction | arm | h3 | h5 | h10 | h15 |
|---|---|--:|--:|--:|--:|
| covid trunk → influenza_us-states | adapted | **+10.9%** sig | **+9.4%** sig | **+8.3%** sig | **+7.3%** sig |
| | no adapter | +27.8% sig | +26.9% sig | +35.7% sig | +36.0% sig |
| influenza_us-states trunk → covid | adapted | +26.3% ns | −5.0% ns | +1.8% ns | +1.9% ns |
| | no adapter | +67.9% sig | +57.6% ns | +123.1% sig | +108.6% sig |

The influenza direction is significantly negative at all four horizons. The value is the
attribution, not the sign: this fold holds `A_geo`, `C`, the node set and the node count identical,
so it varies the disease and nothing else. Every earlier cross-disease cell confounded disease with
graph, geography, node count and panel width at once, which is why none of those negatives could be
pinned on a cause. With the graph confound removed transfer is **still** negative, so the barrier is
the representation, not the graph.

**Outcome 2 — the COVID single-disease model has no skill, and half the table is therefore
unreadable.** 5-seed mean RMSE against its own naive floors:

| h | encoder | persistence | train_mean | verdict |
|--:|--:|--:|--:|---|
| 3 | 5,566 | 4,260 | 5,525 | loses to persistence by 31% |
| 5 | 8,798 | 5,738 | 5,456 | **loses to train_mean by 61%** |
| 10 | 12,264 | 10,068 | 5,302 | **loses to train_mean by 131%** |
| 15 | 11,343 | 27,473 | 5,275 | **loses to train_mean by 115%** |

influenza_us-states beats every floor at every horizon, so this is specific to COVID. Predicting each
state's average training week forever beats the COVID model by up to 131%. That makes the COVID
transfer column above worthless: "indistinguishable from the ceiling at all four horizons" reads as a
success and only means transfer matched a model that is worse than a constant.

**NO COVID TRANSFER NUMBER SHOULD BE QUOTED until COVID beats its own naive floors.** This also
applies to the LDO3 covid fold when it lands. Causes are all already in `covid_eda.md` §4: the val
fold owns Omicron so early stopping selects on an event absent from test; COVID has only 60 train
origins, the fewest of any dev bundle; and the test period is flat post-Omicron, exactly the regime a
constant predictor wins.

**Outcome 3 — adapter ablation.** Improvement from fitting an adapter on the held-out disease versus
borrowing the mean of the in-disease heads: influenza 13.2 / 13.8 / 20.2 / 21.1 %, covid 24.8 / 39.7 /
54.4 / 51.1 % at h3/h5/h10/h15. The frozen representation alone is unusable. Read the right way
round, this is not "the trunk transfers and just needs a read-out": 1,428 adapter parameters recover
most of what is achievable from a foreign trunk and it still lands 7–11% short of training on the
target, which matches the capacity probe where a 15× larger read-out flipped zero cells positive
against the ceiling. The zero-shot arm also remains a weak baseline for the reason in the review
below — `_mean_adapter` averages parameters rather than functions, and the two heads emit into
different per-node z-spaces.

**Outcome 4 — trunks overfit almost immediately on small panels.** Median best validation at step
**2,000**, median stop at **14,000**, i.e. **15.5% of the 91,000-step budget** (patience 12 ×
val_every 1000). Seed 82 covid ran 0.1555 at step 3,000 → 0.1852 by step 13,000, degrading at nearly
every checkpoint. A 142,305-parameter trunk overfits a single 49-node panel in ~2,000 steps, and the
COVID-source trunk trains on 60 origins. This does not invalidate the negative — each trunk sits at
its own validation optimum — but what was measured is transfer from a barely-trained trunk, and the
91,000-step default was calibrated for the dengue-containing configuration, not for single small
panels. Worth revisiting the budget/patience for small-panel trunks.

### Open, as of 2026-08-04

1. **LDO3 has not been run.** `encoder_ldo3*` = 0 files. `run_overnight.py` defaulted to the pair
   fold, so the no-flag invocation ran my recommendation instead of the requested experiment. The
   default is now LDO3 and `--pair` is the opt-in. Command:
   `python -u run_overnight.py --seeds 42 52 62 72 82 --skip-baselines`
2. **COVID single-disease must be fixed before any COVID transfer cell is reportable** (Outcome 2).
3. **`analysis.py` cannot read `ldo3` or `pair` records** — `transfer_ci` hardcodes
   `prefix="encoder_lodo"` and there is no CLI flag for it. Blocks bootstrap CIs on the new folds.
4. **`_mean_adapter` averages parameters, not functions** (measured 0.329 relative deviation), and
   the averaged heads live in different per-node z-spaces. The zero-shot arm is a weak baseline in
   every fold structure until this is addressed or explicitly caveated in the write-up.
5. **Step budget for small panels** — see Outcome 4.

### Also noticed, not fixed

`ablation/run_japan_noseason.py:48` does `recs, pernode = L.train_one(...)`, but `train_one` returns a
4-tuple and has since before this change. That ablation cannot currently run. Pre-existing, out of scope
here, flagged so it is not mistaken for fallout from step 2.

`test_schema.py::test_cumulative_to_incidence_clips_and_masks` fails at line 38
(`assert series.sum() == 22 + 0 + 12`, "week-0 back-log must NOT enter incidence"). **Pre-existing.**
Verified by extracting `cumulative_to_weekly_incidence`, `_finalise`, `fit_scalers_masked` and
`load_ebola` from `git show HEAD:to_schema.py` and comparing to the working copy — all four are
byte-identical, so nothing in the COVID work touched this path. It concerns the Ebola cumulative
transform and should be triaged on its own: either the test encodes the pre-correction expectation
(§5 says the first observed week is now *masked*) or the correction regressed. Not investigated here.

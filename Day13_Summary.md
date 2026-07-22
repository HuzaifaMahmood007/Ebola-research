# Phase 3 · Week 3 — Days 11–13 Summary & Handoff

**Date:** 2026-07-21. **Scope covered:** Day 11 (environment + bundle interface), Day 12
(the encoder), Day 13 (metrics + single-disease trainer + naive floors), a package reorganisation,
the project README, and an advisory code review. **Purpose:** a self-contained handoff so the next
session can resume at Day 14 without re-deriving anything.

**One-line status:** Days 11–13 are functionally complete and passing their gates; an advisory
review found **no Critical issues and no leakage**; nothing is committed yet; `ebola.npz`'s query
set was never touched.

---

## 1. What was built today

### Day 11 — environment + `bundles.py`
- **`ebola-train` conda env** created: python 3.11, `torch 2.6.0+cu124` (CUDA verified live on the
  RTX 3060), numpy 2.4.6, scipy 1.17.1, and later `pandas 3.0.3` (see §3.4). Pinned to
  [data/processed/env_train.txt](data/processed/env_train.txt).
- **[bundles.py](bundles.py)** — one interface over all five `.npz` bundles: `load()`, `masks()`,
  `transfer_view()`, `group_of()`, `refit()`, `origins()`. Absorbs the three format divergences
  (split masks vs train/val integers; 5 vs 4 channels; missing `node_country`) so nothing
  downstream branches on disease. Self-check passes: origin counts match spec exactly
  (dengue 1375, japan 314, us-regions 751, us-states 326), masks partition observed cells,
  `y == X[:,:,0]`.
- **[day11_diagnostics.py](day11_diagnostics.py)** → [results/day11_diagnostics.json](results/day11_diagnostics.json):
  the two §0.8/§0.9 close-out readings (see §3.1, §3.2).

### Day 12 — the encoder (`models/` package)
The locked design (`encoder_architecture_plan.md` Option 1): factorised inductive temporal-per-node
TCN → gated inductive spatial message passing → direct multi-horizon quantile head → light FiLM
adaptation. Split into:
- [models/config.py](models/config.py) — architecture constants (source of truth).
- [models/temporal.py](models/temporal.py) — `DilatedTCN`, 5 layers, dilations [1,2,4,8,16], kernel 2 → **RF 32 ≥ 20** (asserted).
- [models/spatial.py](models/spatial.py) — `normalise_adj` (identity-first, C3), `mask_aware_adj`, `LTR` (degree feature), `SpatialMixer` (GraphSAGE, sparse), `sparse_from_dense_np`.
- [models/encoder.py](models/encoder.py) — `Gate` (learned, `off`=g≡0), `SharedEncoder`, `spatial_contribution`, `node_indexed_params` (C2 audit).
- [models/adapters.py](models/adapters.py) — `Adapter` (FiLM + quantile head), `pinball_loss`, `shared_params`/`adaptation_params`.
- [models/windows.py](models/windows.py) — `window_slice`, `targets_and_mask`.
- **All 8 encoder invariants pass, every negative control fires** ([tests/test_encoder_invariants.py](tests/test_encoder_invariants.py)): RF≥20, C2 (no node-sized param), C3 (Â finite; no-`+I` → NaN), C1 (trunk rejects `C` and 5-channel input), gate-nesting (g=0 == graph-free), LTR degree reuse, 20-node toy overfit to pinball 0.009, size-agnostic run on N=47/49/7165.

### Day 13 — metrics + trainer + naive floors
- **[score.py](score.py) extended** — added `sMAPE` (0/0 cells excluded, not scored as perfect),
  `peak_intensity`, `peak_timing` to `METRICS`; CRPS/PICP/width stubbed for Week 5; the **Ebola
  metric rules pre-registered in the docstring** while no Ebola number exists. Aggregation
  unchanged. Self-check green (6161 nodes scored, 559 constant excluded).
- **[train/loop.py](train/loop.py)** — single-disease trainer: direct multi-horizon (one run = all
  4 horizons → 4 datasets × 5 seeds = **20 runs**), val-only early stopping, model-space-loss /
  count-space-metrics guards, writes `results/*.json` one record per
  `model × dataset × horizon × seed × metric`. Includes the three naive floors (persistence,
  seasonal-naive with recorded persistence fallback, per-node train mean).
- **Smoke test** ran end-to-end: japan trains cleanly (val_pinball 0.137→0.093 over 5 epochs),
  24 records, MAE by horizon 245/308/459/511. The full 20-run matrix (`python -m train.loop --all`)
  is **overnight compute, not yet run**.

### Also today
- **Package reorganisation** into `encoder_architecture_plan.md` §6 layout (`models/`, `train/`,
  `tests/`, `configs/`). Run scripts as modules from the repo root: `python -m train.loop`,
  `python -m tests.test_encoder_invariants`.
- **[configs/encoder_base.yaml](configs/encoder_base.yaml)** — the frozen hyperparameters (human record; code is the source of truth).
- **[README.md](README.md)** — project entry point: layout, environments, reproduce, invariants.
- **Advisory code review** (3 parallel reviewers) — findings in §5.

---

## 2. Decisions locked this week (CONFIRM-P1…P9)

Full record also in the `phase3-week3-encoder-decisions` memory. **All nine are now closed.**

| # | Decision | Outcome |
|---|---|---|
| P1 | Encoder family | Dilated causal TCN + GCN + **LTR** degree feature. RF 32 (kernel 2, dilations [1,2,4,8,16]). |
| P2 | Cross-disease sampling | **Uniform** over the 4 dev datasets (sqrt-weighting = Week-6 ablation). |
| P3 | Epi-informed component | **Ablation only**, not in the primary model. |
| P4 | Output head | **Quantile** {.05,.25,.5,.75,.95} + pinball loss; median = point forecast. |
| P5 | Adaptation scope | **FiLM + head only** (~388 params), sized vs Ebola's 27 support cells. |
| P6 | Baselines | EpiGNN, Cola-GNN, HeatGNN, MTGNN (control), MepoGNN (adaptive variant only). STOEP + MSGNN timeboxed, exclude+document if unresolved. |
| P7 | Ebola few-shot **(consequential)** | **Split protocol** — see §3.1. Zero-shot + light short-horizon adaptation; h=10/15 structurally zero-shot. |
| P8 | Spatial gate | **Learned gate** g=σ(MLP(h)), **LOCKED**. (Reversed from an earlier "unconditional" pick when the encoder was locked to plan Option 1 = "gated".) `off` mode = g≡0 ablation/nesting. |
| P9 | Topology-robust training | **Annealed edge dropout** p 0→0.3 over first 30%; report the {gate on, g≡0} × {aug none, edge_drop} 2×2; never perturb at eval. |

Also decided: **file layout** = the plan's `models/`/`train/`/`tests/`/`configs/` package split
(reversing an earlier single-file call, at the user's request).

---

## 3. Understandings uncovered (the non-obvious things)

These are the findings that were **not** already written down and that the next session should not
have to rediscover.

### 3.1 Ebola has ZERO support origins under the frozen protocol (§0.8) → P7
Support cells end at week index t=7 (calendar prefix through 2014-05-24); the earliest valid target
under w=20 is t+3=22. **Disjoint → 0 support origins.** Confirmed empirically
(`ebola {'support': 0, 'query': 18}`). Resolution (P7, the **split protocol**): adaptation is
separated from the frozen *evaluation* protocol. Left-pad short windows (t<19) with zeros +
`obs_mask=0` for **adaptation only**; h=10 and h=15 are **structurally zero-shot** for Ebola under
any windowing; the Week-5 headline is reframed as zero-shot + light short-horizon adaptation
(52 of 61 districts are zero-shot regardless). **This is unimplemented — the left-pad path is
Week-4 work** (and is review item #4 in §5).

### 3.2 The §0.9 "expect ≈ 0" diagnostic has an Ebola exception
Node-level-mean variance of `X[:,:,0]` is ≈0 **only over the fit window** and **only for per-node
scalers**. Measured (fit window): influenza ~1e-14 (dense, per-node), dengue **0.039** (near-zero;
the residual is ~475 no-train + constant-window nodes taking a country-pooled fallback), Ebola
**0.70** — because Ebola uses a **disease-pooled** scaler (`scaler_scope='per_disease_support'`),
not per-node, by few-shot design. A reader who blindly expects ≈0 on Ebola would mis-debug this;
the diagnostic records `scaler_scope` so it can't be mistaken for a defect. **The encoder adds no
second normalisation** — the released scaler is already applied.

### 3.3 The C3 negative control only manifests densely
`Â = D̃^{-1/2}(A+I)D̃^{-1/2}` without the `+I` should NaN on the 4 degree-0 influenza nodes
(japan_10/19, us-states Alaska/Hawaii). But in **sparse** form an isolated node simply has no edges,
so its `inf` degree never multiplies into any stored value and `A_hat.values()` stays finite — the
NaN only appears in the **dense** product (`inf × 0`). So: the negative control demonstrates the
defect on the small dense japan graph, and `SharedEncoder.forward` guards with **`assert (deg>0).all()`**
(which *does* catch a dropped `+I` even in sparse form). Robustness lesson for any future graph code.

### 3.4 Environment gotchas
- **`ebola-train` needs plain `pandas`** — `bundles.py` imports the frozen `to_schema.py` for the
  scaler primitives, and `to_schema` imports pandas at module top. Plain pandas has **none** of the
  GEOS/GDAL/PROJ stack that §0.7 warns against, so this is consistent with the design (the concern
  is geopandas' native stack, not pandas). It's pinned in `env_train.txt`.
- **Windows OpenMP collision** — `pandas` (MKL `libiomp5md.dll`) and `torch` (`libomp.dll`) both
  init OpenMP in one process → `OMP: Error #15`. `train/loop.py` sets
  `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` **before** importing numpy/torch/pandas.
  Any new entry point that imports both needs the same line.
- **`pip` in `conda run`** resolved to the global Python, not the env — always use
  `conda run -n ebola-train python -m pip …`, and freeze with `PYTHONNOUSERSITE=1` to exclude
  leaked global user-site packages.

### 3.5 Origin phase counts legitimately overlap and over-sum
The per-phase origin counts sum above the unfiltered total (e.g. dengue train 1138 + val 594 +
test 630 > 1375) because **one origin emits four horizon targets that can straddle phase
boundaries**. Cell-level phase membership still prevents any target cell being double-scored. This
is correct — not a bug to "fix."

---

## 4. Repo state & how to run

**Environments:** `ebola` (conda, data/geometry) and `ebola-train` (torch, training). Never
co-resolve them.

```bash
# data-layer checks
conda run -n ebola-train python bundles.py
conda run -n ebola-train python day11_diagnostics.py
conda run -n ebola-train python score.py

# encoder invariants (all gates + controls)
conda run -n ebola-train python -m tests.test_encoder_invariants

# training
conda run -n ebola-train python -m train.loop --smoke     # fast end-to-end check
conda run -n ebola-train python -m train.loop --all        # the 20-run matrix (OVERNIGHT, not yet run)
```
(Prefix training runs with `PYTHONNOUSERSITE=1` on Windows if global packages leak in.)

**Uncommitted — nothing has been committed this session.** New: `bundles.py`, `day11_diagnostics.py`,
`models/`, `train/`, `tests/`, `configs/`, `score.py` (modified), `README.md`,
`data/processed/env_train.txt`, `results/day11_diagnostics.json`, and this file. PROJECT.md §9
recommends committing the data code + these before relying on git. Suggested first commit: a clean
"Phase 3 Week 3 Days 11–13" commit.

---

## 5. Advisory code review — findings (UNFIXED, by user request)

Three parallel reviewers (encoder, training+metrics, data interface). **No Critical issues, no
leakage.** All three verdicts: "merge with fixes." The user chose to leave all code as-is for now;
these are logged for a future pass. Highest-leverage before Week 4/5 lean on them: #2, #3, #4.

**Important — clear fixes:**
1. **Gradient-accumulation flush is unclipped & unconditional** (`train/loop.py:97`). Trailing
   partial batch steps without `clip_grad_norm_`, on summed grads → can take a larger step than any
   full batch. Fix: clip before the trailing step; only step if grads pending.
2. **`refit` has no test, yet the checklist marks it done** (`bundles.py:52`). The §0.3 `X[:,:,0]`
   rebuild is the sharpest leakage trap with zero regression protection. Fix: add a test (y2==X2[:,:,0];
   X2[:,:,0] actually changed; negative control refitting only y must fail).
3. **`refit` hardcodes `per_disease=False` → wrong scaler for Ebola** (`bundles.py:62`). Ebola ships
   a pooled scaler; a Week-5 caller would silently get per-node. Fix: derive from
   `meta['scaler_scope']`, or assert dev-only.
4. **Ebola short-window path unimplemented; `window_slice` fails silently for t<19**
   (`models/windows.py:10`). Negative start index wraps with no error; §2.1/P7 left-padding is what
   makes Week-4 Ebola few-shot work. Fix: `assert t >= W-1` now; implement left-pad before Week 4.
5. **Mask-aware adjacency has zero test coverage; permutation-equivariance & masked-loss gates
   missing** (`models/spatial.py:38`). A §3.3 non-negotiable is untested; permutation-equivariance
   is the strongest check on the sparse index plumbing. Fix: add both tests.

**Important — judgment calls (need a decision):**
6. **Quantile crossing not prevented or flagged** (`models/adapters.py`). Plain Linear head can emit
   crossing quantiles → corrupts Week-5 PICP/interval-width. *Rec:* flag now, enforce monotonicity
   (sort or cumulative-softplus) at Week 5.
7. **Results schema can't produce the mandated stats** (`train/loop.py:170`). Task 13.2 wants
   bootstrap-CI *over origins* + paired Wilcoxon *across seeds*; only mean±sd over 5 seeds is
   reconstructable. *Rec:* persist per-node (or per-origin) scores, not just the two aggregates.
8. **Pinball reduction deviates from spec** (`models/adapters.py:31`): global mean vs plan §5's
   "per-horizon then sum" — material for Ebola where h=10/15 have zero support. *Rec:* per-horizon-then-sum.
9. **"g≡0" arm isn't truly graph-free** (`models/encoder.py:62`): the LTR degree feature is folded in
   before the gate and never gated off, yet code/manuscript call that cell "graph-free." *Rec:*
   reframe as "temporal + degree" rather than gate LTR off.

**Minor (batch later):** NaN→invalid JSON (`json.dumps` emits bare `NaN`); round-trip comment says
`≤1e-4` but real max err ~0.02 (guard still catches gross mis-ordering); seasonal-fallback rate
scoped over the full grid not scored cells; `--smoke` writes into `results/`; no val-NaN guard on
selection; `_demo` prints origin counts but doesn't assert them; §0.9 variance gate thin headroom
(0.039 vs 0.05); `node_indexed_params` misses buffers; no zero-diagonal guard in `normalise_adj`;
`sparse_from_dense_np` should be cached per bundle not rebuilt per origin.

---

## 6. What's next (Days 14–15) and what Week 4 needs

**Day 14 — multi-disease joint training (G2 foundation):**
- `train/sampler.py` (uniform primary per P2 + country-balanced ablation) and `train/topo_aug.py`
  (annealed edge dropout per P9) — not yet built.
- Joint training over the 4 dev datasets, gradient accumulation round-robin, one dataset per batch
  (different N/T — do **not** pad graphs to a common N).
- Record per-dataset help/hurt vs Day-13 single-disease numbers.

**Day 15 — baselines under the common pipeline (G6):**
- `export_baseline.py` producing ColaGNN-format `(matrix.txt, adj.txt)` + a split/scaler/mask
  sidecar. Carry **our** per-country dengue split and the dengue mask across (or the comparison is
  invalid). Baseline queue is ~320–400 runs, launches Day 15, completes in Week 4 (checkpoint Day 20).

**Appendix-C handoff (what Week 4's LODO/meta-learning/k-shot needs from Week 3):** `bundles.py` ✅,
the trunk/adapter split ✅ (`shared_params`/`adaptation_params`), the quantile head ✅, `score.py` +
`results/*.json` ✅, single-disease baseline-comparable numbers ⬜ (needs the overnight `--all` run),
the baseline queue ⬜ (Day 15), and a settled **P7** ✅ (but its left-pad implementation ⬜, review #4).

**Absolute rule still in force:** `ebola.npz`'s query set enters no loop that chooses anything until
Week 5.

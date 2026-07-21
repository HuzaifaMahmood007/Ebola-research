# Phase 3 — Developer Execution Guide (Days 11–30)
### Emerging Disease Forecasting Framework · The Modelling Contribution

**Who this is for:** the implementation developer, continuing from Phase 2. Every step is written so that no action requires a judgment call you don't have the information to make. Real decisions are marked **[CONFIRM-P*]** with a recommended default so you are never blocked.

**Read this first — what Phase 3 is, and is not.**
Phase 2 turned three surveillance sources into five harmonised, leakage-gated bundles. Phase 3 turns those bundles into **the paper's results**. Concretely, you will:
- **Build the disease-agnostic encoder** (G1) on the 4-channel `transfer_view`, and train it on the development diseases.
- **Establish transfer** (G2) — one shared encoder across dengue and influenza, evidenced by leave-one-disease-out.
- **Run the few-shot Ebola case study** (G3) under the frozen validation protocol, exactly once.
- **Add calibrated uncertainty** (G4, adaptive conformal) and **explanation** (G5, SHAP).
- **Re-run the trusted baselines under the common pipeline** (G6) and produce the head-to-head, significance battery, and ablations (G7).

You will **not**, in Phase 3: change any data code. `to_schema.py`, `build_datasets.py` and the leakage suite are **frozen**. If you believe a bundle is wrong, that is an escalation, not a patch — every number in `data_audit.md` is computed from these files and the manuscript quotes them. The single exception is `score.py`, which is a *metrics* module and is explicitly extended in Task 13.1.

The foundation is complete and unblocked: all ten audit decisions are closed (`data_audit.md` §5.4), the datasets and the code that built them are committed, and no [CONFIRM-D*] item from Phase 2 remains open. **The modelling clock is now the only risk.**

---

## 0. Reality checks you must internalise (read once, then act)

### 0.1 The datasets are finished. Do not re-derive anything they already state.
Row *i* of every array is `meta['node_ids'][i]`; column *t* is `meta['dates'][t]`. Nothing downstream re-derives this correspondence, and nothing downstream recomputes a scaler, a split, a graph or a mask except where §0.3 *requires* a refit. `data/processed/*.npz` are committed and were built by a gated, deterministic pipeline; treat them as read-only inputs.

Verified contents, so you can code against them without opening the files first:

| Bundle | `X` | Observed cells | Split representation | Rolling origins |
|---|---|---|---|---|
| `dengue` | (7165, 1409, 4) | 2,196,261 | `split_{train,val,test}_mask` arrays | 12 groups (per country), horizon 1 |
| `influenza_japan` | (47, 348, 4) | 16,356 | **integers** `meta['split']['train_end'/'val_end']` = 174 / 244 | 1 group, horizon 1 |
| `influenza_us-regions` | (10, 785, 4) | 7,850 | **integers** (fixed 50/20/30) | 1 group, horizon 1 |
| `influenza_us-states` | (49, 360, 4) | 17,640 | **integers** (fixed 50/20/30) | 1 group, horizon 1 |
| `ebola` | (61, 52, **5**) | 1,299 | `split_{support,query}_mask` (27 / 1,272) | **none, by design** |

**Three interfaces, not one — and this is the first thing that will bite you.** Dengue and Ebola ship split *masks*; the three influenza bundles ship split *integers* and no mask arrays at all. Ebola has five channels where the others have four (`deaths_norm` is extended). `meta['node_country']` exists for dengue and Ebola and **does not exist for influenza**. Task 11.2 exists to absorb all of this in one place so that no model or metric code ever branches on disease.

### 0.2 The three coded constraints are not advice (`data_audit.md` §4.6)
1. **The encoder must add the identity before any degree normalisation.** Verified: `influenza_japan` has two rows with adjacency row-sum zero (`japan_10` = Hokkaidō, `japan_19` = Okinawa), and `influenza_us-states` has two more (Alaska, Hawaii). Compute `Â = D̃^{-1/2}(A + I)D̃^{-1/2}` where `D̃` is the degree of `A + I`. Normalise `A` alone and those four rows divide by zero and produce NaN — silently, on one of five datasets, part-way through training. Each influenza bundle carries `meta['requires_encoder_self_loops'] = True`; assert on it at model construction.
2. **`C` must never reach the shared encoder.** The three diseases occupy disjoint geography, so a raw centroid identifies the disease outright. `transfer_view()` excludes `C` by construction — keep it that way. Single-disease runs may use `C` freely, and if you do, say so in the results table.
3. **The scaler must be refitted at every rolling origin** — and see §0.3, which is the part that is easy to get wrong.

### 0.3 Refitting the scaler means rebuilding the *input channel*, not just the target
This is the sharpest trap in the whole phase and it is not stated anywhere in the Phase-2 documents, because it only arises once you train.

Verified: `y == X[:, :, 0]` exactly, for every bundle. Both are `inc_norm` — the same array, written twice by `_finalise`. The released scaler belongs to the headline split and has seen data beyond every rolling origin.

So at rolling origin *k* you must:
```python
train_mask, eval_mask = rolling_origin_masks(dt, k)
scaler = fit_scalers_masked(raw, train_mask, per_disease=False, groups=countries)
inc_norm = np.where(M == 1, apply_scaler(raw, scaler), 0.0)
X = X.copy(); X[:, :, 0] = inc_norm        # <-- the step everyone forgets
y = inc_norm
```
Refitting for `y` and leaving `X[:, :, 0]` on the headline scaler leaks the test distribution into the model's *inputs* while every gate stays green — the leakage suite runs at build time, on the released bundles, and cannot see what your trainer does at origin *k*. Write one function that returns `(X, y, scaler)` for an origin and never touch the two separately.

### 0.4 Dengue is 98% of the development corpus — disease-balanced sampling is not optional
This is a **new finding for Phase 3**, and it is not the Brazilian dominance already handled in `data_audit.md` §1.11. That one is *within* dengue. This one is *across diseases*:

| | Dengue | Influenza (all three) | Dengue share |
|---|---|---|---|
| Nodes | 7,165 | 106 | **98.5 %** |
| Observed cells | 2,196,261 | 41,846 | **98.1 %** |

Sample training examples uniformly from the pooled development corpus and the shared encoder sees influenza in roughly one update in fifty. It will not be a disease-agnostic encoder; it will be a dengue encoder that has glanced at influenza — and the transfer claim (G2) is then unsupported by the training procedure that produced it. **Balance at the disease level** (see [CONFIRM-P2]). The uniform-vs-country-balanced question settled in Phase 2 is a *within-dengue* question and is unaffected; it remains the Week-6 ablation.

### 0.5 Ebola's query set is scored exactly once, and process is the only thing protecting it
Every hyperparameter — encoder architecture and optimisation, and the adaptation settings — is selected on the development diseases: the former on development validation folds, the latter by leave-one-development-disease-out. Those settings are then frozen and applied to Ebola **once**, with a fixed number of adaptation steps and no early stopping on query performance (`manuscript_reconciliation.md` §5).

There is no technical control preventing you from peeking; 1,272 query cells sit in a file you can load at any moment. Task 20.3 builds the control that makes an accidental second run visible. Until then, the rule is behavioural: **`ebola.npz`'s query mask does not enter any loop that chooses anything.**

### 0.6 The head-to-head must compare like with like, and Ebola has no fair comparator
Every baseline in the suite is single-disease by construction. Therefore:
- On the **development diseases**, the G6 comparison is *our encoder trained single-disease* against *the baselines trained single-disease*, on identical data, splits, horizons, metrics and seeds. The transfer-trained variant is a **separate, additional row** — reporting it as the head-to-head number would be comparing a model that saw four datasets against models that saw one.
- On **Ebola**, no baseline can do few-shot adaptation at all. The comparators are necessarily naive (persistence, seasonal-naive, a graph-mean predictor) plus any baseline *trained from scratch on 27 support cells*, which will fail. **Say so plainly in the paper.** A large margin over a comparator that cannot in principle do the task is evidence for the regime being hard, not for the model being good; the honest claim is that the framework produces usable forecasts where incumbents produce none.

### 0.7 There is no training environment yet, and the data environment must not become one
`environment.yml` pins `python=3.10, numpy, pandas, geopandas, libpysal, shapely, openpyxl`. There is **no PyTorch**, and there should not be: the training code reads `.npz` and needs `numpy + torch` only — it never touches geometry. Forcing `torch` and the GEOS/GDAL/PROJ stack to co-resolve in one conda environment is a solved-problem-turned-afternoon. Keep three environments (Task 11.1).

Hardware is an **RTX 3060, 12 GB** — note that the Phase-2 guide said 8 GB; it is 12. Adequate for everything here provided the dengue adjacency stays sparse: dense it is 205 MB in float32 with only 40,936 non-zeros (20,468 undirected edges). Use a sparse tensor and the whole graph fits comfortably.

---

## Week 3 — Core framework and the development result (Days 11–15) · G1, G2, G6

### Day 11 — Environment, the bundle access layer, and the windowing contract

**Objective:** build the one module every later day depends on, so that no model, metric or experiment file ever branches on which disease it is holding.

#### Task 11.1 — Three environments, and why
| Environment | Contents | Runs |
|---|---|---|
| Base (Python 3.11) | stdlib + numpy | `test_schema.py` — no geospatial deps, seconds |
| Conda `ebola` | the pinned Phase-2 stack | the build, the leakage suite, anything touching geometry |
| **New: `ebola-train`** | python 3.10/3.11 + `torch` (CUDA 12.x build) + numpy + scipy | all Phase-3 training, evaluation and analysis |

**Best option: a plain venv or conda env with `torch` from the official index, no geopandas.** Pin it and export `data/processed/env_train.txt` alongside the existing `env.txt` at the end of the phase. `scipy` earns its place for the significance battery (Friedman, Wilcoxon) and nothing else — do not add a stats framework for two tests.

#### Task 11.2 — `bundles.py`: one interface over five bundles
This module is the whole reason Phase 3 does not drown in special cases. It must expose:

```python
load(name) -> Bundle           # X, y, raw, M, A_geo, C, meta, and NORMALISED split masks
Bundle.masks()                 # always returns dict of [N,T] uint8 masks, whatever the bundle shipped
Bundle.transfer_view()         # X[:, :, meta['core_feature_idx']] — exactly 4 channels, always
Bundle.country_of()            # node -> group label; synthesises a single group for influenza
Bundle.refit(origin_or_mask)   # returns (X, y, scaler) per §0.3, as ONE object
```

Three normalisations it must perform, each corresponding to a divergence measured in §0.1:
- **Splits.** For influenza, expand `train_end`/`val_end` integers into `[N,T]` masks intersected with `M` (which is all ones there, but do it anyway so the code path is identical). For dengue and Ebola, read the shipped masks unchanged.
- **Channels.** Always slice with `meta['core_feature_idx']` for anything shared. Never assume `F == 4`; Ebola has 5.
- **Groups.** `meta['node_country']` is absent for influenza. Synthesise `{node: dataset_name}` so `score.py` receives a valid map and its country-macro degenerates to the node mean — which is exactly what the manuscript specifies for the single-panel datasets (`manuscript_reconciliation.md` §6). Do **not** patch the `.npz`.

#### Task 11.3 — The windowing contract, written down once
Per the frozen protocol: lookback **w = 20**, horizons **h ∈ {3, 5, 10, 15}**, **direct multi-horizon** (one forward pass emits all four; no autoregressive rollout).

An example is an **origin** *t*, not a node — the graph is consumed whole:
- inputs `X[:, t-19 : t+1, core]`, all *N* nodes;
- targets `y[:, t+h]` for each *h*;
- valid origins satisfy `t >= 19` and `t + max(H) <= T-1`;
- a target cell `(i, t+h)` is included only if `M[i, t+h] == 1` **and** it lies in the phase being trained or scored. **An example's phase is decided by where its target lands**, never by its input window.

**State this explicitly in the methods section, because a reviewer will ask:** when scoring a test-phase target, the input window legitimately contains test-period *observations*. That is what forecasting from a live surveillance stream means. Leakage-safety comes from elsewhere — the scaler saw only training cells, and no input at or beyond `t+h` is ever read. The leakage suite's "no future leakage" gate already pins the second property at the data layer.

**Deliverable (Day 11):** `ebola-train` environment; `bundles.py` with a self-check (`python bundles.py`) that loads all five, asserts `transfer_view().shape[-1] == 4` for every one, asserts the split masks partition the observed cells, and prints the window counts per dataset per phase.

---

### Day 12 — The disease-agnostic encoder (G1)

**Objective:** the shared spatio-temporal encoder, built so that Week 4 can meta-learn it and Week 5 can adapt it from 27 cells.

#### Task 12.1 — Architecture
**Options.**
- **(A) Dilated causal temporal convolution per node + graph convolution, stacked; direct multi-horizon head.** The MTGNN/EpiGNN lineage.
- **(B) GRU/LSTM per node + graph attention.** The Cola-GNN lineage.
- **(C) Temporal transformer + GNN.**

**Best option: (A).** Three reasons, and the third is the one that matters:
1. It is the family the trusted comparators come from, so the head-to-head is like-for-like rather than an architecture-class comparison dressed as a framework comparison.
2. No recurrence over 20 steps × 7,165 nodes — it parallelises over time and stays inside 12 GB.
3. **Week 4 needs second-order gradients through the encoder.** MAML-style inner loops backpropagate through the adaptation step; doing that through a recurrent net over 20 timesteps is where memory and wall-clock die. A convolutional trunk keeps the meta-learning tractable. Choosing (B) on Day 12 quietly makes Day 17 much harder.

Keep it small — roughly 2 temporal blocks (dilations 1, 2) and 2 graph layers, hidden width 64. This is a transfer and calibration paper, not a capacity paper, and a small trunk is what makes few-shot adaptation identifiable.

#### Task 12.2 — The three constraints, in code
- `Â = D̃^{-1/2}(A + I)D̃^{-1/2}`, identity added **before** the degree is computed. Assert `meta.get('requires_encoder_self_loops')` is honoured and that no row of `Â` contains NaN — one assertion, at construction, on all five graphs.
- The forward signature takes the **transfer view only**. Make it structurally impossible to pass `C` to the shared trunk: the shared module's signature has no `C` argument. A single-disease wrapper may add it.
- `A_geo` as a sparse tensor. Dengue's is block-diagonal (12 blocks, no cross-border edges); Ebola's is deliberately **not** block-diagonal and carries 21 cross-border edges including the Guéckédou–Lofa–Kailahun tri-border pathway. Both are handled by the same sparse code; nothing special-cases either.

#### Task 12.3 — The parameter partition, which is the design decision of the whole phase
The formalisation is *one shared encoder with light per-disease adaptation*. Split parameters into:
- **Shared trunk** — temporal blocks + graph layers. Disease-agnostic; the thing that transfers.
- **Per-disease adapter** — a FiLM-style affine modulation (per-channel scale and shift) plus the output head.

**Size the adapter against the Ebola support set, not against convenience.** Ebola support is **27 cells** across 9 of 61 districts. Fitting anything large on 27 cells is not few-shot learning, it is noise-fitting. A FiLM adapter over a width-64 trunk plus a linear head is a few thousand parameters and is a defensible thing to estimate from a handful of observations; a full fine-tune is not. This single constraint is what makes the Week-5 result credible, and it has to be designed in now — retrofitting it in Week 5 means re-running Week 4.

#### Task 12.4 — The output head: quantiles, not a Gaussian
G4 requires CRPS, PICP and interval width in **count space**, while the model works in `log1p + z-score` space.

**Options.** (A) point head + conformal intervals — gives PICP/width but no proper CRPS. (B) Gaussian head (mean, log-variance) with closed-form CRPS. (C) **quantile head** at levels {0.05, 0.25, 0.5, 0.75, 0.95} trained with pinball loss.

**Best option: (C), and the reason is decisive rather than stylistic.** `invert_scaler` is `expm1(σ·z + μ)` — strictly monotone. Quantiles are equivariant under monotone transforms, so a predicted 95th percentile in model space inverts to *exactly* the 95th percentile in count space. A Gaussian's mean and variance do not survive `expm1`: the inverted distribution is log-normal, its mean is not the inverted mean, and the closed-form CRPS you chose (B) for no longer applies. Option (C) makes every count-space uncertainty number exact by construction. CRPS follows from the quantile ladder; PICP and width read straight off the 5th and 95th; ACI (Day 22) adjusts the level online.

The median (τ = 0.5) is the point forecast for RMSE/MAE/PCC.

**Deliverable (Day 12):** `model.py` — shared trunk, per-disease adapter, quantile head; a `demo()` that builds it against all five graphs, asserts no NaN in `Â`, asserts the trunk's signature cannot accept `C`, and overfits a 20-node toy slice to near-zero loss (the smallest check that fails if the wiring is wrong).

---

### Day 13 — Single-disease training and the metric layer (G6 prerequisite)

**Objective:** a trained, scored, reproducible single-disease result on each development dataset — the row the baselines will be compared against.

#### Task 13.1 — Extend `score.py` to the full protocol
`score.py` today implements RMSE, MAE, PCC with the country-macro aggregation and the constant-node exclusion, and it is self-tested. Extend the `METRICS` dict — do not rewrite the aggregation, which is the part that has been verified (a Bolivia-only error moves the node mean 0.015 and the country-macro 0.833 ≈ 10/12).

Add: **sMAPE**; **peak intensity error** (`|max(pred) − max(truth)|` over the eval window, in counts); **peak timing error** (`|argmax(pred) − argmax(truth)|` in weeks); and, once Day 22 produces intervals, **CRPS**, **PICP**, **interval width**.

Two rules `score.py` already enforces and that the new metrics must not break: metrics are computed in **count space** (invert the scaler on predictions; truth is the bundle's `raw`), and **genuinely constant nodes are excluded** by default. Peak timing on a constant node is undefined, so the existing exclusion is exactly right — extend the self-check to prove it.

Headline aggregation per the manuscript: **country-macro for dengue**; **node-level for influenza and Ebola**, count-weighted alongside.

#### Task 13.2 — Train, selecting only on validation
Four development datasets, direct multi-horizon, **5 seeds {42, 52, 62, 72, 82}**. Because one forward pass emits all four horizons, this is 4 datasets × 5 seeds = **20 runs**, not 80.

Select architecture and optimisation hyperparameters on the **validation** folds only. Record the search you ran — a reviewer asking "how many configurations did you try?" should get a number, not a shrug.

#### Task 13.3 — Sanity floor before you believe anything
Before reporting any encoder number, score three trivial predictors on the same cells: **persistence** (`ŷ_{t+h} = y_t`), **seasonal naive** (`ŷ_{t+h} = y_{t+h-52}`), and the **per-node training mean**. If the encoder does not beat all three on every dataset and horizon, the problem is in the harness, not the architecture. This costs an hour and has caught this class of bug in every project that has bothered.

**Deliverable (Day 13):** extended `score.py` with its self-check passing; `train.py`; a results table (5 seeds, mean ± sd) for four development datasets × four horizons; the three naive floors alongside.

---

### Day 14 — Multi-disease joint training and the samplers (G2 foundation)

**Objective:** one encoder over all four development datasets, trained so that the transfer claim is supported by the procedure and not merely asserted.

#### Task 14.1 — Disease-balanced sampling  **[CONFIRM-P2]**
Per §0.4. **Default: sample the disease uniformly per batch** (each of the four development datasets equally likely), then sample origins within it. *Alternative:* proportional-to-data (which is 98% dengue) or square-root-frequency weighting as a middle path.

**Best: uniform over datasets**, with square-root weighting recorded as a Week-6 ablation if time permits. The encoder must be equally at home on a 10-node US-regions graph and a 7,165-node dengue graph, since Ebola's 61 nodes resemble neither in scale; a dengue-dominated encoder has no reason to have learned anything that transfers to a small graph.

Note that the four datasets have different *N* and different *T*. Since an example is a whole-graph origin, batches contain one dataset at a time — do not attempt to batch across diseases in one tensor. Gradient accumulation across a round-robin of the four is the simple, correct implementation.

#### Task 14.2 — Uniform vs country-balanced within dengue
Settled in Phase 2 (`data_audit.md` §5.4 item 7): **uniform node sampling is primary** — the encoder is data-proportional and therefore Brazil-weighted, which preserves the full training signal — and a **country-balanced sampler is built as an ablation**, decided empirically. Build both now; run the comparison in Week 6. Reporting is de-biased by the country-macro metric regardless, which is what makes keeping all nodes sound.

#### Task 14.3 — The epi-informed component  **[CONFIRM-P3]**
The brief marks this optional and the competitive analysis does not place it in the contested intersection — the gap is (disease-agnostic × transferable × few-shot × calibrated), and mechanistic hybridisation is *not* one of the four axes.

**Recommendation: not in the primary model; one ablation row.** The cheapest defensible form is a soft penalty on epidemiologically implausible week-over-week growth, added to the loss with a single weight. Build it as a switch, run it as an ablation in Week 6, and report the result honestly whichever way it falls. Building it into the primary model costs a week and dilutes a contribution that is already precisely located.

**Deliverable (Day 14):** `train.py --multi-disease` with both samplers; the joint-training result across the four development datasets, per-dataset, against the Day-13 single-disease numbers. Note whether joint training helps or hurts each dataset — either finding is publishable and the honest one is required.

---

### Day 15 — Launch the baselines under the common pipeline (G6)

**Objective:** get the trusted comparators running on our data, splits, horizons, metrics and seeds — the only version of the comparison that supports a SOTA claim.

**Scope note.** The baseline block is ~320–400 runs (Appendix B) and does not fit in a day. It is **compute-bound, not developer-bound**: the runs depend on nothing our encoder produces. Day 15's developer work is the adapter and the launch; the queue then grinds through Week 4, with a completeness checkpoint on **Day 20**.

#### Task 15.1 — One adapter, because three baselines share a lineage
Verified: `EpiGNN/src/`, `colagnn/src/` and `HeatGNN-14DB/src/` each contain the same `{data, layers, models, train, utils}.py` structure — they are one code lineage. They consume a `[T, N]` matrix plus an `[N, N]` adjacency, which is exactly what our bundles hold transposed.

**Best option: export, do not port.** Write `export_baseline.py` producing `(matrix.txt, adj.txt)` per bundle in the ColaGNN format, plus a JSON sidecar recording the split indices and the scaler. One ~40-line script covers three of the comparators. Porting our loader into each repository would mean maintaining three forks of code we do not own.

Two things the export must carry across, or the comparison is invalid:
- **The split.** The baselines apply their own internal 50/20/30. For influenza that coincides with ours (verified: `train_end=174, val_end=244` on Japan's 348 weeks is exactly 50/20/30). For **dengue it does not** — ours is per-country, cut on each country's own observed span. Pass our split explicitly; do not let a baseline re-cut dengue globally, which is the very failure the per-country split exists to prevent.
- **The mask.** Dengue is 21.75% observed. The baselines assume dense matrices. Imputed cells must be excluded from their loss and their scoring, or their numbers are computed on 78% fabricated zeros and the comparison is meaningless in our favour — which is worse than losing.

#### Task 15.2 — Which baselines can actually enter the table
| Model | Status entering Phase 3 | Action |
|---|---|---|
| **EpiGNN** | ✅ reproduced within ~1% | Core comparator. Re-run at h ∈ {3,5,10,15}, 5 seeds. |
| **Cola-GNN** | ⚠️ ran h=1; paper reports {2,3,5,10,15} | Core comparator. Re-run at **our** horizons. |
| **MTGNN** | ✅ reproduced (non-epidemic control) | Keep as the control row — it establishes what a general ST-GNN achieves without epidemic structure. |
| **HeatGNN** | ⚠️ ran h=1; compare in ×10³ units | Re-run at our horizons. **Watch the units** — the paper reports Japan RMSE in thousands. |
| **MepoGNN** | ✅ reproduced (Dynamic, Japan-COVID) | **Verified problem:** all three of its source files reference commuting/OD data, and `A_mob` is `None` for every disease (mobility declined in Phase 2, `data_audit.md` §2.8). The **Dynamic** variant cannot run on our data. Run the adaptive/learned-graph variant if the repo exposes one, and **disclose that the mobility-driven variant is out of scope for want of an OD matrix** — that is a property of the data, not a failure of the model, and it must be stated that way. |
| **STOEP** | ❌ paper table ≠ shipped dataset/metric | Timebox one day to resolve provenance. If unresolved, **exclude from the head-to-head and document why.** A comparator whose published numbers you cannot reproduce is not a comparator; reporting it anyway invites the reviewer to ask which number is wrong. |
| **MSGNN** | ⛔ not run (CUDA 10.1 / Linux) | Timebox WSL2. If it does not run, report "not reproducible in our environment" with the specific blocker. That is a legitimate, useful finding — the reproducibility appendix is a contribution in its own right. |

#### Task 15.3 — Standardise horizons and metric definitions before a single table is drawn
`PROJECT.md` §7 names this the top threat to any SOTA claim, and the baseline-reproduction record shows why: one model was run at h=1 against a paper reporting {2,3,5,10,15}; another reports RMSE in thousands; a third logs MAPE as a fraction where its paper prints percentage points. Fix the horizon grid at **{3, 5, 10, 15}** and the metric definitions at `score.py`'s, recompute every baseline's metrics **from its saved predictions using our `score.py`**, and never copy a number out of a baseline's own log.

**Deliverable (Day 15):** `export_baseline.py` with the split and mask carried across; the exported datasets; the baseline queue launched and logging to `results/`; a status note recording MepoGNN's mobility limitation, and the STOEP and MSGNN dispositions. Predictions saved per model/dataset/horizon/seed and every metric recomputed through `score.py` **as the queue completes**; completion checkpoint **Day 20**.

---

## Week 4 — Transfer, meta-learning and few-shot (Days 16–20) · G2, G3

### Day 16 — Leave-one-disease-out transfer, zero-shot
Train the shared encoder on three development datasets, evaluate on the fourth **with no adaptation**. Four folds × 5 seeds. This is the zero-shot transfer curve and the first direct evidence for G2.

Report against two references: the single-disease model trained on the held-out dataset (the ceiling) and the naive floors from Task 13.3. The interesting quantity is where zero-shot transfer sits between them.

**Watch for the tell.** If zero-shot dengue→influenza is surprisingly strong, check that no geographic or scale information is leaking through the input. The core channels are incidence (per-node z-scored, so scale-free by construction), two calendar-derived seasonality channels, and the mask. Northern/southern-hemisphere phase is a *legitimate* shared signal; anything beyond it deserves a second look.

### Day 17 — Meta-learning for disease-invariant representations
**Options.** (A) MAML / first-order MAML (Reptile) over diseases as tasks. (B) Domain-generalisation by adversarial disease-discrimination. (C) Plain multi-task joint training (the Day-14 model) as the meta-learning-free control.

**Best: first-order MAML (Reptile-style), with (C) as the control.** Full second-order MAML is the textbook choice and roughly triples memory for a benefit that is usually small at this scale; first-order variants are standard practice and keep the 7,165-node dengue graph inside 12 GB. The task distribution is the development diseases; the inner loop adapts **only the per-disease adapter** from Task 12.3, matching exactly what will be adapted on Ebola.

(B) is attractive because "disease-invariant" is literally what an adversarial discriminator optimises for — but the discriminator can trivially separate diseases by graph size, so it would learn to erase useful signal. Note the reasoning and move on.

### Day 18 — The few-shot adaptation protocol and the k-shot sweep
Define adaptation exactly as it will run on Ebola: freeze the shared trunk, adapt the per-disease adapter for a **fixed number of inner steps** at a fixed inner learning rate, on the support cells only, with the scaler fitted on pooled support cells only.

Sweep on the development LODO folds, treating each held-out development disease as a surrogate emerging pathogen under the same calendar-prefix support construction: **k ∈ {1, 2, 4, 8, all-support}** weeks, inner steps ∈ {1, 5, 10, 25}, inner LR over a small grid. The k-shot curve is a headline figure for G3 — it is the quantitative statement of how little data the framework needs.

**The surrogate must be built the same way Ebola's was**, i.e. a *calendar prefix*, not the first k observed weeks per node. Ebola's protocol was changed precisely because per-node support was not calendar-causal (`data_audit.md` §3.6): 85% of query cells preceded the last support cell, so the pooled scaler drew on peak-epidemic magnitudes. Selecting adaptation settings on a surrogate built the acausal way would tune for a regime Ebola does not present.

### Day 19 — Freeze
Write the selected configuration to a **read-only** `configs/frozen_ebola_protocol.json`: encoder hyperparameters, adapter definition, inner steps, inner LR, support scheme (`calendar_prefix <= 2014-05-24`), seeds, and a hash of the code that produced them. Everything after this point reads that file. Nothing after this point writes it.

Record in the same file *why* each value was chosen and which development fold chose it — that paragraph is §7 of the manuscript and it is much harder to reconstruct in Week 6 than to write today.

### Day 20 — Rehearse the Ebola run without scoring it

**Objective:** find every bug in the Ebola path while the query set is still sealed.

- **Task 20.1 — Dry-run on structure only.** Load `ebola.npz`, run adaptation on the 27 support cells, produce query-shaped predictions, and assert shapes, absence of NaN, and that the pooled-support scaler is the one in the bundle. **Do not compute a metric.**
- **Task 20.2 — Verify the zero-shot majority is handled.** 52 of 61 districts have **no support cell** — they are pure zero-shot, forecast from the graph and from the 9 districts that reported early. Assert that predictions exist for all 61 and that `meta['nodes_without_query']` is empty (verified: it is). This is the paper's central regime; a bug here that surfaces on Day 21 costs the single evaluation.
- **Task 20.3 — Build the one-shot control.** `run_ebola_once.py` writes `results/ebola_query_receipt.json` containing the frozen-config hash, the code hash, a timestamp and the resulting scores, and **refuses to run if that file exists**. Re-running requires an explicit flag and *appends* a new receipt rather than overwriting. The receipt file is the audit trail that the query set was scored once; it is also what you show a reviewer who asks.

**Deliverable (Week 4):** LODO zero-shot and few-shot results across development diseases; the k-shot curve; `configs/frozen_ebola_protocol.json`; a rehearsed, unscored Ebola path; `run_ebola_once.py`.

---

## Week 5 — Ebola, uncertainty and explanation (Days 21–25) · G3, G4, G5

### Day 21 — The Ebola case study, run once (G3)
Execute `run_ebola_once.py` with the frozen configuration, 5 seeds. Score the query set through `score.py` — node-level averages, count-weighted alongside, per the manuscript.

Report the breakdown that makes the result meaningful:
- **9 districts with support** vs **52 zero-shot districts**, scored separately. The zero-shot subset is the genuine emerging-outbreak regime and is the number the paper's claim rests on.
- Per-country (Guinea 32, Liberia 15, Sierra Leone 14).
- Against the naive floors and against any baseline trained on the 27 support cells.

**Two disclosures travel with every Ebola magnitude, without exception.** First, **gap-lumping** (`data_audit.md` §3.4.3): 7% of inter-report intervals exceed one week, the longest is 24 weeks, and the whole increment is stamped on the week the report arrived — Montserrado carries 1,428 cases on 2014-10-25. The curve's *shape* is correct and mass is conserved exactly, but weekly peak magnitudes are inflated and their neighbours flattened. Peak-intensity error on Ebola inherits this directly. Second, four Guinean prefectures record **no new cases at all** across their observed weeks; they are real districts and valid neighbours, retained in the graph and excluded from scoring by the constant-node rule.

### Day 22 — Calibrated uncertainty (G4)
Implement **adaptive conformal inference** (Gibbs & Candès 2021): maintain `α_t`, update `α_{t+1} = α_t + γ(α − err_t)` where `err_t` is 1 if the realised value fell outside the interval, and emit the `α_t`-level interval from the quantile head. Online along the calendar, **no held-out calibration set** — which is precisely why it was chosen, since Ebola has none.

Report **CRPS**, **PICP** and **mean interval width** at the 90% level, in count space, from the inverted quantile ladder (exact, per Task 12.4).

Run it on the development test sets too, and **cross-check against split conformal** there — development diseases *do* have a validation fold, so split conformal is available as an independent calibration reference. Agreement between the two on development is the evidence that the ACI implementation is correct before it is trusted on Ebola, where no such cross-check exists.

Calibration must be reported per horizon. Coverage that holds at h=3 and collapses at h=15 is the normal failure mode and is a finding, not something to average away.

### Day 23 — Uncertainty under distribution shift, and the zero-shot districts
The interesting calibration question for this paper is not whether PICP ≈ 90% on dengue's test fold. It is whether coverage survives on a **held-out disease** and, within it, on the **52 zero-shot districts**. Report PICP and width for: development test folds; LODO transfer; Ebola support-carrying districts; Ebola zero-shot districts. Widening intervals on the zero-shot subset is the *correct* behaviour and should be shown, not apologised for — a model that knows it is uncertain where it has no data is the calibration claim.

### Day 24 — Explainability (G5)
**Options.** (A) KernelSHAP — model-agnostic, prohibitively expensive on a 7,165-node graph. (B) **GradientSHAP / DeepSHAP** over the input tensor. (C) Attention weights — not SHAP, and not faithful.

**Best: (B), scoped tightly.**
- **Global:** mean |SHAP| per input channel (incidence, sin-doy, cos-doy, mask) and per lag (1–20), per disease. The expected, checkable result is that recent lags and incidence dominate, with the seasonality channels carrying more weight on influenza than on Ebola — influenza is strongly seasonal and a 52-week Ebola outbreak has no seasonal cycle to learn. If that ordering does not appear, distrust the attribution before distrusting the epidemiology.
- **Local:** a small number of Ebola districts at epidemiologically meaningful weeks — a tri-border district around the October 2014 national peak (2014-10-25, 2,688 cases/week) is the obvious case study.
- **Spatial:** neighbour attribution by edge ablation — mask one neighbour's contribution and measure the change. For the zero-shot districts this answers the question the whole paper poses: *where does a forecast for a district we have never observed actually come from?*

Timebox this to one day. It is a section of the paper, not a research programme.

### Day 25 — Consolidate
Every result to date into one machine-readable store (`results/*.json`, one record per model × dataset × horizon × seed × metric). Every table and figure in Week 6 is generated from this file by a script — no number is ever typed into the manuscript by hand. That is what makes the final consistency check mechanical rather than an act of faith.

**Deliverable (Week 5):** the Ebola result with its receipt; calibration across all five datasets and both transfer regimes; SHAP global/local/spatial; the consolidated results store.

---

## Week 6 — Ablations, robustness and writing (Days 26–30) · G7

### Day 26 — Ablations
Each isolates one claim. Run on development data; only the support-window ablation touches Ebola, and it is a *parameter* variation declared in advance (`data_audit.md` §5.4 item 10), not a second bite at the query set.

| Ablation | Isolates |
|---|---|
| Transfer-trained vs single-disease | G2 — does sharing help at all? |
| Meta-learning vs plain joint training | whether the meta-learning earns its complexity |
| Uniform vs **country-balanced** sampling | the Phase-2 deferred question (`data_audit.md` §5.4 item 7) |
| Disease-uniform vs proportional sampling | §0.4 — the new cross-disease imbalance |
| With / without the epi-informed penalty | [CONFIRM-P3] |
| Constant nodes included vs excluded (Japan) | the agreed scoring sensitivity (§1.7) |
| Core-4 only vs + extended channels (single-disease) | what disease-agnosticism actually costs |
| Ebola support window: calendar cutoff vs variable | comparability, declared in advance |

The core-4 ablation is worth emphasising: it quantifies the price of the framework's central constraint. If withholding `deaths_norm` and `C` costs little, the disease-agnosticism argument gets much stronger; if it costs a lot, that is an honest limitation and a reviewer will respect its being measured rather than avoided.

### Day 27 — Rolling-origin robustness and the significance battery
**Rolling origins.** Five expanding-window origins per group, from `meta['rolling_origins']` — 12 country groups for dengue, one for each influenza set, **none for Ebola by design** (expanding the window past support *is* the leakage the few-shot design forbids). Refit the scaler at every origin **per §0.3 — rebuild `X[:, :, 0]`, not just `y`**.

One gotcha, measured: the stored origins were generated with **horizon 1**, so the backtest is single-step, exactly as `data_audit.md` §4.4 describes it. If you want a multi-horizon backtest, `build_rolling_origins(..., horizon=h)` is a pure function of `node_ids`, the observed mask and the split scheme — regenerate the cut points without rebuilding any dataset. Whichever you use, say which in the paper.

**Significance.** Diebold–Mariano per model pair per dataset per horizon, with the Harvey–Leybourne–Newbold small-sample correction. DM operates on a *series* of loss differentials, so reduce the node panel to one value per time step (mean loss across scored nodes at that step) before testing, and state that reduction. Then Friedman across the (dataset × horizon) blocks with models as treatments, and Nemenyi post-hoc for the critical-difference diagram. Five seeds throughout; report mean ± sd and test on seed-averaged predictions.

### Day 28 — Apply the manuscript reconciliation
`manuscript_reconciliation.md` holds eight exact old→new edits. **Item 1 first**: eq. (4) still prints the retracted clip formula that fabricated 35.8% of the Ebola target. It is a correctness defect and it is in a methods description, so it must not survive into a submitted draft — and no results table quotes it, which is the only reason it is still merely a defect and not a retraction.

Then items 2–8: Table 3's dengue and Ebola rows; §6.3's spatial-level description; the three per-district→calendar-prefix support passages; the new §7 model-selection paragraph; the country-macro metric; the block-diagonal-as-simplification correction; and the influenza calendar-provenance sentence. Add the NIID references (IASR 40(11) 2019, 39(11) 2018).

### Day 29 — Tables and figures, generated not typed
From `results/*.json`: the head-to-head (with the single-disease/transfer rows clearly distinguished per §0.6), the LODO transfer table, the k-shot curve, calibration by horizon and regime, the critical-difference diagram, the ablation table, the SHAP panels, and the Ebola case study with its zero-shot breakdown.

### Day 30 — Reproducibility package and the final consistency pass
`env_train.txt`; the frozen config; the Ebola receipt; seeds; one command that regenerates every table from the stored results. Then a final pass in which **every number in the manuscript is checked against `results/*.json`** — mechanical, because nothing was typed by hand.

---

## Phase-3 "definition of done" checklist
- [ ] `bundles.py` presents one interface over all five bundles; split, channel and group divergences absorbed in one place.
- [ ] Encoder honours the three coded constraints; identity added before degree normalisation asserted on all five graphs; `C` structurally excluded from the shared trunk.
- [ ] Scaler refits rebuild `X[:, :, 0]` as well as `y` (§0.3), verified by a test.
- [ ] Single-disease results on four development datasets, 5 seeds, beating all three naive floors.
- [ ] Multi-disease training with disease-balanced sampling; both dengue samplers implemented.
- [ ] Baselines re-run under the common pipeline at h ∈ {3,5,10,15}, metrics recomputed through `score.py`; MepoGNN mobility limitation, STOEP and MSGNN dispositions documented.
- [ ] LODO zero-shot and few-shot transfer results; k-shot curve on calendar-prefix surrogates.
- [ ] `configs/frozen_ebola_protocol.json` written before any Ebola scoring, and read-only thereafter.
- [ ] Ebola scored **once**, with a receipt; zero-shot (52) and support-carrying (9) districts reported separately; gap-lumping disclosed with every magnitude.
- [ ] ACI intervals with CRPS/PICP/width per horizon and per regime; cross-checked against split conformal on development.
- [ ] SHAP global, local and spatial-neighbour attributions.
- [ ] All eight ablations; rolling-origin backtest with per-origin refit; DM + Friedman–Nemenyi.
- [ ] `manuscript_reconciliation.md` fully applied, item 1 first.
- [ ] Every table and figure generated from `results/*.json`; reproducibility package complete.

## Decisions to confirm before locking (send to client)
- **[CONFIRM-P1]** Encoder family: **dilated temporal convolution + GCN, direct multi-horizon** (default) vs recurrent + attention. Default chosen partly to keep Week-4 meta-learning tractable.
- **[CONFIRM-P2]** Cross-disease sampling: **uniform over the four development datasets** (default) vs proportional-to-data (98% dengue) vs square-root weighting. This is distinct from the within-dengue sampler already settled in Phase 2.
- **[CONFIRM-P3]** Epidemiology-informed component: **ablation only, not in the primary model** (default) vs built into the primary.
- **[CONFIRM-P4]** Output head: **quantile / pinball** (default, because quantiles invert exactly through `expm1`) vs Gaussian mean-variance.
- **[CONFIRM-P5]** Adaptation scope: **per-disease FiLM adapter + head only** (default, sized against 27 Ebola support cells) vs full fine-tune.
- **[CONFIRM-P6]** Baseline suite for the head-to-head: **EpiGNN, Cola-GNN, HeatGNN, MTGNN (control), MepoGNN (adaptive variant only)** — with STOEP and MSGNN excluded-and-documented unless their blockers resolve within their timeboxes.

---

## Appendix A — Measured facts the trainer must handle

Verified directly against `data/processed/*.npz`, not inferred from documentation.

- `y == X[:, :, 0]` exactly, in every bundle. Both are `inc_norm`. See §0.3.
- Split representations differ: dengue and Ebola ship `[N,T]` masks; the three influenza bundles ship **integers only** (`train_end`, `val_end`) and no mask arrays.
- `meta['node_country']` exists for **dengue and Ebola only**. Calling `score.score_bundle` on influenza without synthesising a group map raises `KeyError`.
- Ebola's `X` has **5** channels (`deaths_norm` extended); all others have 4. `meta['core_feature_idx'] == [0,1,2,3]` in all five — `transfer_view()` is identical across diseases, which is the disease-agnosticism guarantee.
- Four influenza nodes have adjacency row-sum zero: `japan_10`, `japan_19`, `us-states_1`, `us-states_9`. Add the identity before degree normalisation.
- Dengue `A_geo` dense is **205 MB** float32 with **40,936** non-zeros (20,468 undirected edges). Use sparse tensors.
- Ebola: 27 support cells, 1,272 query cells, 9 of 61 districts carry support, `meta['nodes_without_query']` is empty — every district is scored.
- Rolling origins exist for the four development bundles at **horizon 1** (dengue grouped per country, 12 groups); Ebola has none.
- Dengue is **98.1%** of development observed cells and **98.5%** of development nodes.

## Appendix B — Run matrix and compute budget

RTX 3060, 12 GB. Direct multi-horizon means one run yields all four horizons, which is the single largest saving available — take it.

| Block | Runs | Note |
|---|---|---|
| Single-disease encoder | 4 datasets × 5 seeds = **20** | Day 13 |
| Multi-disease joint | 2 samplers × 5 seeds = **10** | Day 14 |
| LODO transfer (zero + few-shot) | 4 folds × 5 seeds = **20** | Days 16–18 |
| k-shot sweep | 5 k × 4 folds × 3 seeds ≈ **60** (short runs, adapter only) | Day 18 |
| Ebola | 5 seeds, **once** | Day 21 |
| Ablations | 8 × 5 seeds ≈ **40** | Day 26 |
| Rolling origin | 5 origins × 4 datasets × 3 seeds = **60** | Day 27 |
| **Baselines** | 4–5 models × 4 datasets × **4 horizons** × 5 seeds ≈ **320–400** | Day 15 — **the largest block by far** |

The baselines dominate, because unlike our encoder they train one model per horizon. If the schedule slips, cut baseline **seeds** before cutting baseline **models** — a narrower comparison across three seeds is far more defensible than a five-seed comparison against two comparators. Whatever is cut, `log()` it in the paper: a silently truncated comparison reads as a complete one.

## Appendix C — Risks carried into Phase 3

- **The single Ebola evaluation.** Mitigated by the Day-19 freeze, the Day-20 rehearsal and the Day-20.3 receipt. Process, not code, is the control until Day 20.
- **Baseline comparability** (`PROJECT.md` §7). Horizons, metric definitions and units standardised on Day 15; every metric recomputed from saved predictions through our `score.py`, never copied from a baseline's log.
- **Cross-disease imbalance** (§0.4). New in Phase 3; addressed by [CONFIRM-P2] and measured by an ablation.
- **Scaler refit rebuilding only `y`** (§0.3). The one leak the build-time gates structurally cannot see. Pin it with a test.
- **Transfer through disjoint geography.** `C` and any geographic feature stay out of the shared encoder; enforced by the shared trunk's signature.
- **Manuscript–dataset drift.** Eq. (4) still prints the retracted clip. Apply `manuscript_reconciliation.md` item 1 early, so it cannot be quoted anywhere.
- **Schedule.** Weeks 3–6 are the entire novel contribution and the baseline re-runs are the largest single compute block. The data is no longer the risk.

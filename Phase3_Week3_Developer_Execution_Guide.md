# Phase 3 · Week 3 — Developer Execution Guide (Days 11–15)
### Emerging Disease Forecasting Framework · Core Framework Build & Development Training

> **Revision 2 — Day 11 complete; architecture reconciled with `encoder_architecture_plan.md`.**
>
> **Status:** Day 11 ✅ COMPLETE · Day 12 ⬜ blocked pending §0.8 · Days 13–15 ⬜
>
> Five corrections land in this revision. Two are defects in revision 1 that would have shipped
> silently, and one is a go/no-go on the Week-4 design:
>
> | # | Correction | Where |
> |---|---|---|
> | **A** | **Ebola has zero support origins** under the frozen protocol — Day 11's own origin-count printout should already show this | **§0.8** |
> | **B** | **The temporal stack's receptive field is 4, not 20** — dilations (1,2) cannot see the lookback window | **Task 12.1** |
> | **C** | The released scaler is already **per-node** log1p+z — do **not** add a second normalisation, and scope the spatial claim accordingly | **§0.9** |
> | **D** | Add a **learned spatial gate** so the encoder nests the graph-free model | **Task 12.5** |
> | **E** | Pre-register Ebola metric handling, loss space and statistical reporting before any numbers exist | Tasks 13.1, 13.2 |
>
> Corrections A and B block Day 12. Do not start `model.py` until both are resolved.
> `encoder_architecture_plan.md` is the architecture reference; this guide is the schedule.
> Where they disagreed, this revision reconciles them — noted inline each time.

**Who this is for:** the implementation developer, continuing from Phase 2. Every step is written so that no action requires a judgment call you don't have the information to make. Real decisions are marked **[CONFIRM-P*]** with a recommended default so you are never blocked.

**Read this first — what Week 3 is, and is not.**
Phase 2 delivered five harmonised, leakage-gated bundles. Week 3 turns them into a **trained, scored, reproducible development result** — the foundation every later week builds on. Concretely, you will:
- **Stand up the training stack** and one access layer that hides the five bundles' differences (Day 11).
- **Build the disease-agnostic encoder** on the 4-channel `transfer_view`, honouring the three coded constraints (Day 12).
- **Train and score single-disease** on the four development datasets, with a full metric layer (Day 13).
- **Train multi-disease** with disease-balanced sampling — the procedure the G2 transfer claim rests on (Day 14).
- **Launch the baseline re-runs** under the common pipeline, which then complete in the background through Week 4 (Day 15).

You will **not**, in Week 3: touch `ebola.npz` (Week 5 — and see §0.5), run leave-one-disease-out transfer or meta-learning (Week 4), add uncertainty or SHAP (Week 5), or run ablations, rolling-origin backtests or the significance battery (Week 6).

You will also **not change any data code**. `to_schema.py`, `build_datasets.py`, `test_leakage.py` and the five `.npz` are **frozen**. Every number in `data_audit.md` is computed from them and the manuscript quotes them; if you believe a bundle is wrong, that is an escalation, not a patch. The single exception is `score.py`, which is a *metrics* module and is explicitly extended in Task 13.1.

**Week 3 is where all six Phase-3 design decisions get made.** [CONFIRM-P1] through [CONFIRM-P6] all land in these five days. Send them to the client on Day 11; none of them blocks you, because each has a default, but each is expensive to reverse after Week 4 starts.

---

## 0. Reality checks you must internalise (read once, then act)

### 0.1 The datasets are finished, and they do not present one interface
Row *i* of every array is `meta['node_ids'][i]`; column *t* is `meta['dates'][t]`. Nothing re-derives this. `data/processed/*.npz` are committed, deterministic and gated — treat them as read-only inputs.

Verified directly from the files, so you can code against this table without opening them:

| Bundle | `X` | Observed cells | Split representation | `node_country`? |
|---|---|---|---|---|
| `dengue` | (7165, 1409, 4) | 2,196,261 | `split_{train,val,test}_mask` arrays | ✅ |
| `influenza_japan` | (47, 348, 4) | 16,356 | **integers** `train_end=174`, `val_end=244` | ❌ |
| `influenza_us-regions` | (10, 785, 4) | 7,850 | **integers** (fixed 50/20/30) | ❌ |
| `influenza_us-states` | (49, 360, 4) | 17,640 | **integers** (fixed 50/20/30) | ❌ |
| `ebola` | (61, 52, **5**) | 1,299 | `split_{support,query}_mask` | ✅ |

**Three divergences, and each will bite a different file if you don't absorb them on Day 11.** Dengue and Ebola ship split *masks*; the three influenza bundles ship split *integers* and **no mask arrays at all**. Ebola has five channels where the others have four (`deaths_norm` is extended, never core). `meta['node_country']` is **absent for influenza**, so calling `score.score_bundle` on it raises `KeyError`. Task 11.2 exists so that no model, metric or experiment file ever branches on which disease it is holding.

### 0.2 The three coded constraints are not advice (`data_audit.md` §4.6)
1. **The encoder must add the identity before any degree normalisation.** Verified: four influenza nodes have adjacency row-sum zero — `japan_10` (Hokkaidō), `japan_19` (Okinawa), `us-states_1` (Alaska), `us-states_9` (Hawaii). Compute `Â = D̃^{-1/2}(A + I)D̃^{-1/2}` where `D̃` is the degree of `A + I`. Normalise `A` alone and those four rows divide by zero and produce NaN — silently, on two of five datasets, part-way through training. Each influenza bundle carries `meta['requires_encoder_self_loops'] = True`; assert on it at model construction.
2. **`C` must never reach the shared encoder.** The three diseases occupy disjoint geography, so a raw centroid identifies the disease outright. `transfer_view()` excludes `C` by construction. Enforce it structurally (Task 12.2), not by discipline.
3. **The scaler must be refitted at every rolling origin.** Week 6 consumes this, but you build the function on Day 11 — and §0.3 is the part that is easy to get wrong.

### 0.3 Refitting the scaler means rebuilding the *input channel*, not just the target
Not stated anywhere in the Phase-2 documents, because it only arises once you train.

Verified: **`y == X[:, :, 0]` exactly, in every bundle.** Both are `inc_norm` — the same array, written twice by `_finalise`. The released scaler belongs to the headline split.

So any refit must do this, as one indivisible operation:
```python
scaler = fit_scalers_masked(raw, fit_mask, per_disease=False, groups=countries)
inc_norm = np.where(M == 1, apply_scaler(raw, scaler), 0.0)
X = X.copy(); X[:, :, 0] = inc_norm        # <-- the step everyone forgets
y = inc_norm
```
Refit `y` and leave `X[:, :, 0]` on the headline scaler and you have leaked the evaluation distribution into the model's *inputs* while every gate stays green — the leakage suite runs at build time, on the released bundles, and cannot see what your trainer does. Week 3 trains on the headline split and needs no refit, but **build `Bundle.refit()` returning `(X, y, scaler)` as one object on Day 11** so that Week 6 cannot get it wrong.

### 0.4 Dengue is 98% of the development corpus — disease-balanced sampling is not optional
A **new finding for Phase 3**, and distinct from the Brazilian dominance already settled in `data_audit.md` §1.11. That one is *within* dengue. This one is *across* diseases:

| | Dengue | Influenza (all three) | Dengue share |
|---|---|---|---|
| Nodes | 7,165 | 106 | **98.5 %** |
| Observed cells | 2,196,261 | 41,846 | **98.1 %** |

Sample training examples uniformly from the pooled development corpus and the shared encoder sees influenza in roughly one update in fifty. It will not be a disease-agnostic encoder; it will be a dengue encoder that has glanced at influenza — and the G2 transfer claim is then unsupported by the very procedure that produced it. Balance at the disease level: **[CONFIRM-P2]**, Task 14.1.

### 0.5 You do not open `ebola.npz` this week
Ebola's query set is scored **exactly once**, in Week 5, under a configuration frozen at the end of Week 4. There is no technical control preventing an early peek — 1,272 query cells sit in a file you can load at any moment — and the control that makes a second run visible is not built until Day 20.

Week 3's rule is therefore behavioural and absolute: **Ebola enters no loop that chooses anything.** Not for a sanity check, not for a shape assertion, not "just to see". Everything you need to size the design against Ebola is already in this document (61 nodes, 52 weeks, 27 support cells, 9 of 61 districts carrying support) and in `data_audit.md` §3.

### 0.6 The head-to-head must compare like with like
Every baseline in the suite is single-disease by construction. So on the development diseases, the G6 comparison is **our encoder trained single-disease** (Day 13) against **the baselines trained single-disease** (Day 15), on identical data, splits, horizons, metrics and seeds.

The multi-disease model from Day 14 is a **separate, additional row**. Reporting it as the head-to-head number would compare a model that saw four datasets against models that saw one — which a reviewer will catch, and which would discredit the honest result sitting next to it. Keep the two rows distinct in the results store from the first day they exist.

### 0.7 There is no training environment yet, and the data environment must not become one
`environment.yml` pins `python=3.10, numpy, pandas, geopandas, libpysal, shapely, openpyxl`. There is **no PyTorch**, and there should not be: the training code reads `.npz` and needs `numpy + torch` only — it never touches geometry. Forcing `torch` and the GEOS/GDAL/PROJ stack to co-resolve in one conda environment is a solved problem turned into an afternoon.

Hardware is an **RTX 3060, 12 GB** — the Phase-2 guide said 8 GB; it is 12. Adequate for everything here **provided the dengue adjacency stays sparse**: dense it is 205 MB in float32 with only 40,936 non-zeros (20,468 undirected edges).

### 0.8 BLOCKER — Ebola has zero support origins, and Day 11's output already shows it

Day 11's deliverable prints origin counts *per dataset per phase*, for all five bundles. **Read the
Ebola row of that printout before doing anything else.** The expected value is
`ebola {'support': 0, 'query': ...}`, and if it is 0 the Week-4 few-shot design does not work as
specified.

The arithmetic, which needs no access to query values (this is schedule metadata, not a peek —
§0.5 is not violated):

- Ebola is 61 districts × **T = 52** weekly steps, 2014-04-05 → 2015-03-28.
- Support is a calendar prefix through **2014-05-24** = week index **t = 7** (04-05, 04-12, 04-19,
  04-26, 05-03, 05-10, 05-17, 05-24). All 27 support cells lie in `t ∈ [0, 7]`.
- Task 11.3's contract requires `t ≥ 19` and `t + 15 ≤ 51`, so the earliest target cell is
  `t + 3 = 22`.
- Support targets end at 7. The earliest target is 22. **Disjoint. Zero support origins, and the
  Week-5 inner loop has nothing to supervise on.**

**Resolution — separate the adaptation protocol from the evaluation protocol.** Only the latter is
frozen, so this is legitimate and needs no bundle change:

1. **Left-pad short windows, for adaptation only.** Permit origins `t < 19`, left-padding to 20
   steps with zeros and `obs_mask = 0`. The mask channel exists for exactly this, and a dilated
   causal conv handles it natively. Task 11.3's evaluation contract is untouched.
2. **h = 10 and h = 15 are structurally zero-shot for Ebola.** A support target needs `t + h ≤ 7`:
   h=3 admits origins `t ∈ [0,4]`, h=5 admits `t ∈ [0,2]`, h=10 and h=15 admit **none, ever, under
   any windowing choice**. Adaptation supervises short horizons only; long-horizon Ebola measures
   pure transfer.
3. **Reframe the Week-5 headline as zero-shot with light short-horizon adaptation.** Only 9 of 61
   districts have any support cell, so 52 districts are zero-shot regardless. This is the honest
   framing and the stronger claim.

This is a **new decision for the client — [CONFIRM-P7]**, and it is the most consequential of the
seven because it changes what Week 5 can claim. Send it with the other six.

### 0.9 The released scaler is already per-node — do not normalise twice

`fit_scalers_masked` returns `mean` and `std` of shape `[N]`: the transform is **log1p + per-node
z-score**, fitted per node over the fit window. So `X[:, :, 0]` arrives at the trunk with each
node already standardised to mean 0, sd 1 in log space.

**Three consequences, and the third is a manuscript claim.**

1. **Do not add a second instance/window normalisation inside the encoder.** An earlier
   architecture draft specified a RevIN-style window norm; against this pipeline it would
   double-normalise, distort the sin/cos channels' relative amplitude, and interact badly with the
   §0.3 refit. **The encoder consumes `transfer_view()` as-is.**
2. **Add one diagnostic on Day 12** — the variance of node-level means in `X[:, :, 0]`. Expect
   ≈ 0. Log it once per dataset so the next reader does not rediscover this by debugging a
   confusing ablation.
3. **Scope the spatial claim to timing, not magnitude.** Per-node standardisation removes
   cross-node *level* differences, so the graph layers cannot propagate "my neighbour has more
   cases than I do". What survives — and what the spatial channel actually transports — is
   **phase and shape co-movement**: does my neighbour turn upward when I do. Epidemic wave
   propagation is a timing phenomenon, so this is real signal, but the manuscript must say which
   signal it is.

   This is not a defect to fix. Cross-node magnitude is a proxy for population and therefore for
   geography and therefore for disease identity — exactly what constraint §0.2-2 forbids. **The
   per-node scaler is a consequence of the disease-agnosticism constraint, not an oversight**, and
   the cost it imposes is the loss of magnitude-gradient information. State that trade-off
   explicitly; a reviewer who spots it unaided will assume it was accidental.

---

## Day 11 — Environment, the bundle access layer, and the windowing contract ✅ **COMPLETE**

**Objective:** build the module every later day depends on, so that no model, metric or experiment file ever branches on which disease it is holding.

> **Completion note.** `bundles.py` is delivered and its self-check passes: five bundles load,
> `transfer_view()` is 4 channels for every one (including Ebola's 5-channel `X`), the split masks
> partition the observed cells, `y == X[:, :, 0]`, `group_of()` covers every node, and origin
> counts print per dataset per phase. The three §0.1 divergences are absorbed in `_normalise_masks`
> and `_groups`, and `refit()` rebuilds `X[:, :, 0]` rather than `y` alone (§0.3).
>
> **Two close-out items before Day 12 — both are readings of Day 11's own output, not new work:**
>
> - [ ] **Ebola support-origin count**, from the per-phase printout. If it is `0`, §0.8 applies and
>       **[CONFIRM-P7]** goes to the client. This is the go/no-go on the Week-4 design.
> - [ ] **Node-level-mean variance of `X[:, :, 0]`**, one number per dataset (§0.9). Expect ≈ 0.
>       Record it — Task 12.6's ablation cannot be interpreted without it.
>
> Origin counts before phase filtering should match the guide exactly (dengue 1,375, us-regions
> 751, us-states 326, japan 314); that agreement is the check that the windowing contract was
> implemented as written.

### Task 11.1 — Three environments, and why
| Environment | Contents | Runs |
|---|---|---|
| Base (Python 3.11) | stdlib + numpy | `test_schema.py` — no geospatial deps, seconds |
| Conda `ebola` | the pinned Phase-2 stack | the build, the leakage suite, anything touching geometry |
| **New: `ebola-train`** | python 3.10/3.11 + `torch` (CUDA 12.x) + numpy + scipy | all Phase-3 training and evaluation |

**Best option: a plain venv or conda env with `torch` from the official index, no geopandas.** Pin it and export `data/processed/env_train.txt` alongside the existing `env.txt`. `scipy` earns its place for Week 6's significance battery and nothing else — do not add a statistics framework for two tests.

### Task 11.2 — `bundles.py`: one interface over five bundles
This module is the whole reason Phase 3 does not drown in special cases.

```python
load(name) -> Bundle           # X, y, raw, M, A_geo, C, meta + NORMALISED split masks
Bundle.masks()                 # always dict of [N,T] uint8, whatever the bundle shipped
Bundle.transfer_view()         # X[:, :, meta['core_feature_idx']] — exactly 4 channels, always
Bundle.group_of()              # node -> group label; synthesises one group for influenza
Bundle.refit(fit_mask)         # returns (X, y, scaler) as ONE object, per §0.3
Bundle.origins(w, H, phase)    # valid origins for a phase, per the contract below
```

Three normalisations, one per divergence in §0.1:
- **Splits.** For influenza, expand `train_end`/`val_end` into `[N,T]` masks intersected with `M` (all ones there, but do it anyway so the code path is identical). For dengue and Ebola, read the shipped masks unchanged.
- **Channels.** Always slice with `meta['core_feature_idx']`. Never assume `F == 4`; Ebola has 5.
- **Groups.** Synthesise `{node: dataset_name}` where `meta['node_country']` is absent, so `score.py` receives a valid map and its country-macro degenerates to the node mean — which is exactly what the manuscript specifies for the single-panel influenza datasets (`manuscript_reconciliation.md` §6). **Do not patch the `.npz`.**

### Task 11.3 — The windowing contract, written down once
Per the frozen protocol: lookback **w = 20**, horizons **h ∈ {3, 5, 10, 15}**, **direct multi-horizon** — one forward pass emits all four, no autoregressive rollout.

An example is an **origin** *t*, not a node — the graph is consumed whole:
- inputs `X[:, t-19 : t+1, core]`, all *N* nodes;
- targets `y[:, t+h]` for each *h*;
- valid origins satisfy `t >= 19` and `t + 15 <= T-1`;
- a target cell `(i, t+h)` is included only if `M[i, t+h] == 1` **and** it lies in the phase being trained or scored. **An example's phase is decided by where its target lands**, never by its input window.

Origin counts before phase filtering, so you can check your loop: dengue **1,375**, us-regions **751**, us-states **326**, japan **314**.

**State this in the methods section, because a reviewer will ask:** when scoring a test-phase target, the input window legitimately contains test-period *observations*. That is what forecasting from a live surveillance stream means. Leakage-safety comes from elsewhere — the scaler saw only training cells, and no input at or beyond `t+h` is ever read.

**Deliverable (Day 11):** `ebola-train` environment; `bundles.py` with a self-check (`python bundles.py`) that loads all five, asserts `transfer_view().shape[-1] == 4` for every one, asserts the split masks partition the observed cells, asserts `y == X[:,:,0]`, and prints origin counts per dataset per phase. [CONFIRM-P1]–[CONFIRM-P6] sent to the client.

---

## Day 12 — The disease-agnostic encoder (G1)

**Objective:** the shared spatio-temporal encoder, built so that Week 4 can meta-learn it and Week 5 can adapt it from 27 cells.

### Task 12.1 — Architecture  **[CONFIRM-P1]**
**Options.** (A) Dilated causal temporal convolution + graph convolution, stacked; direct multi-horizon head — the MTGNN/EpiGNN lineage. (B) GRU/LSTM + graph attention — the Cola-GNN lineage. (C) Temporal transformer + GNN.

**Best option: (A).** Three reasons, and the third is the one that matters:
1. It is the family the trusted comparators come from, so Day 15's head-to-head is like-for-like rather than an architecture-class comparison dressed up as a framework comparison.
2. No recurrence over 20 steps × 7,165 nodes — it parallelises over time and stays inside 12 GB.
3. **Week 4 needs second-order gradients through the encoder.** MAML-style inner loops backpropagate through the adaptation step, and doing that through a recurrent net over 20 timesteps is where memory and wall-clock die. Choosing (B) today quietly makes Day 17 much harder.

Keep it small — 2 graph layers, hidden width 64, plus the LTR node-feature encoding below
(Task 12.1a). This is a transfer and calibration paper, not a capacity paper, and a small trunk is
what makes few-shot adaptation identifiable at all.

> **CORRECTION B — the temporal stack must be resized, and this is a blocking defect.**
> Revision 1 specified "2 temporal blocks (dilations 1, 2)". For a dilated causal convolution the
> receptive field is `RF = 1 + Σ (k−1)·dᵢ`, so kernel 2 with dilations (1, 2) gives **RF = 4** —
> the encoder would see 4 of the 20 timesteps in the lookback window and ignore the other 16.
>
> It would still train, still converge, and still produce plausible numbers. The only symptom
> would be a baffling ablation in which `w = 20` and `w = 4` perform identically, and by then
> Weeks 4–5 would be built on it.
>
> **Use kernel 2 with dilations (1, 2, 4, 8, 16)** → `RF = 1 + (1+2+4+8+16) = 32 ≥ 20`. Kernel 3
> with (1, 2, 4, 8) → RF 31 is an equivalent alternative. At width 64 the extra layers cost very
> little; this does not disturb Task 12.3's small-trunk sizing, because the parameter growth is in
> the *shared* trunk, not the adapter.
>
> **Assert it at construction — this is a Day-12 gate, not a code comment:**
> ```python
> rf = 1 + sum((kernel_size - 1) * d for d in dilations)
> assert rf >= W, f"TCN receptive field {rf} < lookback {W}"
> ```
> Paired negative control: dilations (1, 2) must fail the assertion.

### Task 12.1a — Local Transmission Risk (LTR): one component adopted from the EpiGNN lineage

A literature check against EpiGNN's actual published architecture (not just its name) found three components the "Architecture" description above simplifies away: LTR, GTR and RAGL. Two do not survive this project's own constraints and are **not** adopted:
- **GTR** computes full self-attention over all *N* nodes — a dense *N*×*N* matrix. At dengue's N = 7,165 that is the same ~205 MB dense-tensor liability §0.7 already ruled out for `A_geo`. Adding it would reopen a memory problem this guide already solved.
- **RAGL**'s adaptive edge gate uses `W_s ∈ R^{N×N}`, a **learnable parameter matrix sized to one dataset's node count**. It cannot be a shared-weight component of a trunk that runs across N = 10, 47, 49, 61 and 7,165 — porting it as published would silently break the disease-agnostic-trunk constraint (§0.2-2), the same class of silent failure as an unrefit `X[:,:,0]` (§0.3).

**LTR survives**, and is now **in scope for Day 12**: it encodes each node's degree as a learned feature,
```
h_i^LTR = W_LTR · d_i + b_LTR
```
where `d_i` is `Â`'s self-loop-inclusive degree — **the same `D̃` already computed for Task 12.2's normalisation step below; do not recompute it** — and `W_LTR, b_LTR` are shared, disease-agnostic parameters (scalar → hidden-width-64). Add `h_i^LTR` into each node's feature vector before the graph-convolution layers. Cost is one linear layer (~130 parameters) — it does not move the "small trunk" sizing in Task 12.3.

**Deferred to Week 6, not Day 12:** a constraint-clean version of RAGL's idea — an adaptive gate over `A_geo`'s existing edges via a small bilinear form (`sigmoid(hᵢᵀ W hⱼ)`, `W ∈ R^{64×64}`, ~4K params, N-independent, unlike `W_s`) — is a legitimate ablation candidate, in the same spirit as Task 14.3's epidemiology-informed penalty. It is left out of Day 12 because Task 12.3 sizes the trunk small on purpose for few-shot identifiability, and adding capacity before Task 13.3's naive-floor check has run is premature.

### Task 12.2 — The three constraints, in code
- `Â = D̃^{-1/2}(A + I)D̃^{-1/2}`, identity added **before** the degree is computed. Assert at construction, on all five graphs, that no entry of `Â` is NaN — one assertion, and it is the difference between a clean run and a silent two-dataset corruption.
- The shared trunk's forward signature takes the **transfer view only and has no `C` argument**. Make passing covariates to it a `TypeError`, not a code-review question. A single-disease wrapper may add `C` freely; if you use it, say so in the results table.
- `A_geo` as a sparse tensor. Dengue's is block-diagonal (12 country blocks, no cross-border edges); Ebola's is deliberately **not** block-diagonal and carries 21 cross-border edges including the Guéckédou–Lofa–Kailahun tri-border pathway. The same sparse code handles both; nothing special-cases either.

### Task 12.3 — The parameter partition, which is the design decision of the phase  **[CONFIRM-P5]**
The formalisation is *one shared encoder with light per-disease adaptation*. Split parameters into:
- **Shared trunk** — temporal blocks + graph layers. Disease-agnostic; the thing that transfers.
- **Per-disease adapter** — FiLM-style affine modulation (per-channel scale and shift) plus the output head.

**Size the adapter against the Ebola support set, not against convenience.** Ebola support is **27 cells** across 9 of 61 districts. Fitting anything large on 27 cells is not few-shot learning, it is noise-fitting. A FiLM adapter over a width-64 trunk plus a linear head is a few thousand parameters and is a defensible thing to estimate from a handful of observations; a full fine-tune is not.

This constraint is what makes the Week-5 result credible, and it must be designed in **now**. Retrofitting it in Week 5 means re-running Week 4.

### Task 12.4 — The output head: quantiles, not a Gaussian  **[CONFIRM-P4]**
G4 (Week 5) requires CRPS, PICP and interval width in **count space**, while the model works in `log1p + z-score` space.

**Options.** (A) point head + conformal intervals — PICP/width but no proper CRPS. (B) Gaussian head (mean, log-variance) with closed-form CRPS. (C) **quantile head** at levels {0.05, 0.25, 0.5, 0.75, 0.95}, pinball loss.

**Best option: (C), and the reason is decisive rather than stylistic.** `invert_scaler` is `expm1(σ·z + μ)` — strictly monotone. Quantiles are equivariant under monotone transforms, so a predicted 95th percentile in model space inverts to *exactly* the 95th percentile in count space. A Gaussian's mean and variance do not survive `expm1`: the inverted distribution is log-normal, its mean is not the inverted mean, and the closed-form CRPS you chose (B) for no longer applies. Option (C) makes every count-space uncertainty number exact by construction.

The median (τ = 0.5) is the point forecast for RMSE/MAE/PCC. Build the head now even though calibration is Week 5 — switching heads later invalidates Week 3 and Week 4 results.

### Task 12.5 — The spatial gate: make the encoder nest the graph-free model  **[CONFIRM-P8]**

The diseases do not share a transmission kernel. Dengue spreads over short-range vector-suitable
geography; Ebola spread by distance- and population-weighted human movement through the
Guéckédou–Lofa–Kailahun corridor. And §0.9 has just narrowed what the graph can carry to timing
co-movement. A mis-specified or weakly-informative graph is therefore the expected case, not the
exception — and the dengue graph is *known* mis-specified, since `data_audit.md` §1.11 records its
no-cross-border-edges justification as **False** (Brazil borders four of the twelve countries).

So do not hard-wire the graph in. Mix it in through a learned gate:

```
g       = sigmoid(MLP(h))          # per node, d → d/4 → 1; data-dependent, N-independent
h_out   = (1 - g) · h + g · h_spatial
```

Three properties, and the first is why it is required rather than nice:

- **Graceful degradation.** If geography is uninformative for a disease, `g → 0` and the model
  collapses to the graph-free temporal forecaster. The spatial channel can then only help, never
  hurt — which is what makes the whole spatial apparatus safe to carry into Week 5.
- **Constraint-clean.** The MLP is `64 → 16 → 1`, ~1K parameters, no dimension sized by `N`. Same
  test as RAGL's `W_s` failed in Task 12.1a, and this passes it.
- **Reportable — but not as raw `g`.** A raw gate value is not scale-free: it is comparable across
  diseases only if `‖h_spatial‖` and `‖h‖` are, and they are not. Log the **normalised spatial
  contribution** instead:
  ```python
  contrib = (g * h_s.norm(dim=-1)) / ((1 - g) * h.norm(dim=-1) + g * h_s.norm(dim=-1) + 1e-8)
  ```
  Mean and IQR per dataset per epoch. Keep raw `g` as a debug metric only, and never put it in a
  figure caption.

Pair it with a Day-12 gate: forcing `g = 0` must reproduce the temporal-only model **exactly**.

### Task 12.6 — Two diagnostics that gate the interpretation of later ablations

Cheap to add now, and expensive to be missing when Day 14's numbers look strange.

1. **Node-level-mean variance of `X[:, :, 0]`**, per dataset (§0.9). Expect ≈ 0. This is the
   evidence that "the graph adds nothing" — should Task 14.4 report it — is a consequence of the
   per-node scaler and the disease-agnosticism constraint, not of the architecture.
2. **`Â` degree distribution**, per dataset, logged once. Dengue's block-diagonal graph and
   Ebola's cross-border graph have materially different degree profiles, and LTR (Task 12.1a)
   feeds degree directly into the trunk as a learned feature. Worth a line in the limitations:
   degree is a structural quantity that correlates with which dataset is being processed, so LTR
   is the one shared component with a residual disease-identifiability surface. It is retained
   because the signal is epidemiologically motivated and the alternative is discarding local
   transmission structure entirely — but say so rather than leaving it to be found.

**Deliverable (Day 12):** `model.py` — shared trunk (temporal blocks + graph layers + LTR
node-feature encoding + spatial gate), per-disease adapter, quantile head; a `demo()` that builds
it against all five graphs and asserts: no NaN in `Â`; the trunk rejects `C` with a `TypeError`;
LTR reuses `Â`'s degree vector rather than recomputing one; **`RF ≥ 20` (Correction B)**;
**`g = 0` reproduces the temporal-only model exactly**; **no shared-trunk parameter has any
dimension in {10, 47, 49, 61, 7165}**; and it overfits a 20-node toy slice to near-zero loss (the
smallest check that fails if the wiring is wrong). Each assertion ships with a negative control
that plants the defect and requires the assertion to fire — a gate that cannot fail proves nothing,
which is the Phase-2 discipline carried forward.

> **Reconciliation note.** `encoder_architecture_plan.md` §3.6 specified a point head with masked
> Huber loss. **Task 12.4 supersedes it:** the quantile head wins on the `expm1` equivariance
> argument, which is decisive and which the plan did not account for. The loss is **pinball, not
> Huber**. Similarly, the plan's §3.1 window normalisation is **dropped** per §0.9. The plan's
> remaining architecture — factorised temporal-then-spatial, inductive mixing, gate, small adapter
> — stands unchanged.

---

## Day 13 — Single-disease training and the metric layer (G6 prerequisite)

**Objective:** a trained, scored, reproducible single-disease result on each development dataset — the row the baselines will be compared against.

### Task 13.1 — Extend `score.py` to the full protocol
`score.py` today implements RMSE, MAE and PCC with the country-macro aggregation and the constant-node exclusion, and it is self-tested (a Bolivia-only error moves the node mean 0.015 and the country-macro 0.833 ≈ 10/12). **Extend the `METRICS` dict; do not rewrite the aggregation**, which is the verified part.

Add now: **sMAPE**, **peak intensity error** (`|max(pred) − max(truth)|` over the eval window, in counts), **peak timing error** (`|argmax(pred) − argmax(truth)|` in weeks). Leave CRPS/PICP/width as stubs — Week 5 fills them.

**Pin the sMAPE definition explicitly, because there are two in circulation.** Use `mean(2·|ŷ−y| / (|y|+|ŷ|)) × 100`, range 0–200. Cells where `y == ŷ == 0` are undefined and must be **excluded from the mean, not counted as zero error** — dengue is 78% imputed and Ebola carries four districts that never record a case, so a "0/0 counts as perfect" convention would hand the model a large free credit on exactly the sparsest data. Write that rule into the docstring and pin it with a self-check.

Two rules `score.py` already enforces and the new metrics must not break: metrics are computed in **count space** (invert the scaler on predictions; truth is the bundle's `raw`), and **genuinely constant nodes are excluded** by default. Peak timing on a constant node is undefined, so that exclusion is exactly right — extend the self-check to prove it.

Headline aggregation per the manuscript: **country-macro for dengue**; **node-level for influenza**, count-weighted alongside.

**Pre-register the Ebola metric rules now, in Week 3, while no Ebola number exists.** Ebola is not
scored until Week 5, which is precisely why the rules should be fixed here — a choice made in
Week 5 with the data in hand cannot be shown to be independent of it, and a reviewer is entitled to
assume the worst. Ebola runs at mask density 0.2175 with four districts that never record a case,
so PCC is near-meaningless on much of the panel and sMAPE is undefined at zero:

- **MAE and RMSE are primary for Ebola.**
- **PCC** is reported only over districts with ≥ 5 observed non-zero cells, with the qualifying
  district count printed beside it.
- **sMAPE** inherits the 0/0-excluded rule above; if the qualifying cell count falls below 30% of
  the query set, drop sMAPE for Ebola entirely and say so.
- **Peak timing** is undefined for districts with no peak; exclude them and report the count.

Write these into the `score.py` docstring this week, with the same status as the sMAPE rule.

### Task 13.2 — Train, selecting only on validation
Four development datasets, direct multi-horizon, **5 seeds {42, 52, 62, 72, 82}**. Because one forward pass emits all four horizons, this is 4 × 5 = **20 runs**, not 80.

Select architecture and optimisation hyperparameters on the **validation** folds only. Record the search you ran — a reviewer asking "how many configurations did you try?" should get a number, not a shrug.

**Pin the loss space, and assert it.** Pinball loss is computed in **model space** (`log1p + z`),
where targets are ≈ unit-scale; **all metrics are computed in count space** after `expm1(σ·z + μ)`.
These two facts are easy to state and easy to get backwards under the §0.3 refit, so guard them:

```python
assert targets.abs().median() < 10, "loss is not being computed in model space"
```

plus a round-trip test, `invert_scaler(apply_scaler(x)) ≈ x` to 1e-5. Mis-ordering these two
transforms is the single easiest silent bug in the trainer, and it is invisible in the loss curve.

**Statistical reporting — decide it now, because five seeds with no intervals will not survive
review.** All comparisons are **paired on identical origin sets**. Report **mean ± 95% bootstrap
CI over origins** as the primary evidence, and a **paired Wilcoxon signed-rank across seed-matched
runs** as supporting evidence for every headline comparison (ours vs each baseline, and each
ablation). With n = 5 seeds the test is underpowered: lead with the interval, and do not claim
significance off five points. `scipy` is already in `ebola-train` for exactly this (Task 11.1).

### Task 13.3 — The sanity floor, before you believe any encoder number
Score three trivial predictors on the same cells: **persistence** (`ŷ_{t+h} = y_t`), **seasonal naive** (`ŷ_{t+h} = y_{t+h−52}`), and the **per-node training mean**. If the encoder does not beat all three on every dataset and horizon, the problem is in the harness, not the architecture.

One implementation note: seasonal naive needs `y[t+h−52]` to exist *and be observed*. On dengue, at 21.75% mask density, it frequently is not — fall back to persistence for those cells and record how often the fallback fired, rather than silently scoring a different predictor than the one you named.

This costs an hour and has caught this class of bug in every project that has bothered to do it.

**Deliverable (Day 13):** extended `score.py` with its self-check passing; `train.py`; a results table (5 seeds, mean ± sd) for four development datasets × four horizons; the three naive floors alongside; everything written to `results/*.json` in the record-per-`model × dataset × horizon × seed × metric` form that Week 6's tables are generated from.

---

## Day 14 — Multi-disease joint training and the samplers (G2 foundation)

**Objective:** one encoder over all four development datasets, trained so the transfer claim is supported by the procedure and not merely asserted.

### Task 14.1 — Disease-balanced sampling  **[CONFIRM-P2]**
Per §0.4. **Default: sample the dataset uniformly per batch** (each of the four equally likely), then sample origins within it. *Alternatives:* proportional-to-data (98% dengue) or square-root-frequency weighting as a middle path.

**Best: uniform over datasets**, with square-root weighting recorded as a Week-6 ablation. The encoder must be equally at home on a 10-node US-regions graph and a 7,165-node dengue graph, because Ebola's 61 nodes resemble neither in scale — a dengue-dominated encoder has no reason to have learned anything that transfers to a small graph.

Implementation note: the four datasets have different *N* and *T*, and an example is a whole-graph origin, so **batches contain one dataset at a time**. Do not attempt to batch across diseases in one tensor. Gradient accumulation over a round-robin of the four, stepping the optimiser once per round, is the simple and correct implementation.

### Task 14.2 — Uniform vs country-balanced within dengue
Settled in Phase 2 (`data_audit.md` §5.4 item 7): **uniform node sampling is primary** — the encoder is data-proportional and therefore Brazil-weighted, which preserves the full training signal — and a **country-balanced sampler is built as an ablation**, decided empirically. Build both now; the comparison runs in Week 6. Reporting is de-biased by the country-macro metric regardless, which is what makes keeping all 7,165 nodes sound.

### Task 14.3 — The epidemiology-informed component  **[CONFIRM-P3]**
The brief marks this optional, and the competitive analysis does not place it in the contested intersection — the gap is (disease-agnostic × transferable × few-shot × calibrated), and mechanistic hybridisation is **not** one of the four axes.

**Recommendation: not in the primary model; one ablation row.** The cheapest defensible form is a soft penalty on epidemiologically implausible week-over-week growth, added to the loss behind a single weight. Build it as a switch today, run it as an ablation in Week 6, report the result honestly whichever way it falls. Building it into the primary model costs a week and dilutes a contribution that is already precisely located.

### Task 14.4 — Topology-robust training, and the 2×2 that keeps it honest  **[CONFIRM-P9]**

The encoder will be asked in Week 5 to run on a graph it never trained on, built by a different
process (Ebola's cross-border contiguity) for a disease with a different transmission kernel. And
per `data_audit.md` §1.11, one of the *training* graphs is already known mis-specified. So train
for graph mis-specification rather than hoping it does not matter: perturb the topology each
origin with **edge dropout** `p ∈ [0, 0.3]`, optionally light random rewiring. Ship as
`--topo-aug {none,edge_drop,rewire}`, default `edge_drop`. **Never perturb at evaluation.**

**This interacts with Task 12.5's gate, and run naively the two cancel.** Under aggressive
augmentation the optimal policy is `g → 0` everywhere: the encoder learns the graph is unreliable
*because you made it unreliable*, and the paper's two spatial contributions neutralise each other.
Two mitigations, both required:

- **Anneal.** Ramp `p_drop` linearly 0 → 0.3 over the first 30% of training, so the gate learns on
  a trustworthy graph before robustness pressure arrives. `--topo-aug-schedule {constant,linear}`,
  default `linear`.
- **Report the 2×2, not two margins.** The primary spatial table is
  `{gate on, g ≡ 0} × {topo-aug none, edge_drop}`, 5 seeds per cell. Two separate one-way
  ablations cannot distinguish "augmentation helps" from "augmentation suppressed the gate and the
  temporal trunk carried the result", and that distinction is the whole claim.

Interpret the result against Task 12.6's diagnostics. Given §0.9, a small spatial contribution is a
plausible and reportable outcome — but it must be attributed correctly, to the per-node scaler and
the disease-agnosticism constraint rather than to the architecture.

**Deliverable (Day 14):** `train.py --multi-disease` with both samplers and `--topo-aug`; the joint-training result across the four development datasets, per dataset, against Day 13's single-disease numbers; the Task 14.4 2×2 with normalised spatial contribution logged per cell. **Record whether joint training helps or hurts each dataset** — either finding is publishable, and the honest one is required.

---

## Day 15 — Launch the baselines under the common pipeline (G6)

**Objective:** get the comparators running on our data, splits, horizons, metrics and seeds — the only version of the comparison that supports a SOTA claim.

**Scope note, and read this before planning your day.** The baseline block is ~320–400 runs (4–5 models × 4 datasets × **4 horizons** × 5 seeds — unlike our encoder, they train one model per horizon). That is the largest compute block in the entire phase and it does not fit in a day. But it is **compute-bound, not developer-bound**: the runs depend on nothing our encoder produces. Day 15's developer work is the adapter and the launch; the queue then grinds through Week 4, with a completeness checkpoint on **Day 20**.

### Task 15.1 — One adapter, because three baselines share a lineage
Verified: `EpiGNN/src/`, `colagnn/src/` and `HeatGNN-14DB/src/` each contain the same `{data, layers, models, train, utils}.py` structure — they are one code lineage, consuming a `[T, N]` matrix plus an `[N, N]` adjacency, which is exactly what our bundles hold transposed.

**Best option: export, do not port.** Write `export_baseline.py` producing `(matrix.txt, adj.txt)` per bundle in the ColaGNN format, plus a JSON sidecar carrying the split indices and the scaler. One ~40-line script covers three comparators. Porting our loader into each repository means maintaining three forks of code we do not own.

Two things the export **must** carry across, or the comparison is invalid:
- **The split.** The baselines apply their own internal 50/20/30. For influenza that coincides with ours (verified: `train_end=174, val_end=244` on Japan's 348 weeks is exactly 50/20/30). For **dengue it does not** — ours is per-country, cut on each country's own observed span. Pass our split explicitly; do not let a baseline re-cut dengue globally, which is precisely the failure the per-country split exists to prevent (`data_audit.md` §1.8).
- **The mask.** Dengue is 21.75% observed and the baselines assume dense matrices. Imputed cells must be excluded from their loss and their scoring, or their numbers are computed on 78% fabricated zeros — a comparison rigged **in our favour**, which is worse than losing one.

### Task 15.2 — Which baselines can actually enter the table  **[CONFIRM-P6]**
| Model | Status entering Phase 3 | Action |
|---|---|---|
| **EpiGNN** | ✅ reproduced within ~1% | Core comparator. Re-run at h ∈ {3,5,10,15}, 5 seeds. |
| **Cola-GNN** | ⚠️ ran h=1; paper reports {2,3,5,10,15} | Core comparator. Re-run at **our** horizons. |
| **MTGNN** | ✅ reproduced (non-epidemic control) | Control row — establishes what a general ST-GNN achieves without epidemic structure. |
| **HeatGNN** | ⚠️ ran h=1; **units in ×10³** | Re-run at our horizons. Watch the units: the paper reports Japan RMSE in thousands. |
| **MepoGNN** | ✅ reproduced (Dynamic, Japan-COVID) | **Verified problem:** all three source files reference commuting/OD data, and `A_mob` is `None` for every disease (mobility declined in Phase 2, `data_audit.md` §2.8). The **Dynamic variant cannot run on our data.** Run the adaptive/learned-graph variant if the repo exposes one, and disclose that the mobility-driven variant is out of scope for want of an OD matrix — a property of our data, not a failure of the model, and it must be stated that way. |
| **STOEP** | ❌ paper table ≠ shipped dataset/metric | Timebox one day. If unresolved, **exclude and document why.** A comparator whose published numbers you cannot reproduce is not a comparator; reporting it anyway invites the reviewer to ask which number is wrong. |
| **MSGNN** | ⛔ not run (CUDA 10.1 / Linux) | Timebox WSL2. If it does not run, report "not reproducible in our environment" with the specific blocker — a legitimate and useful finding for the reproducibility appendix. |

### Task 15.3 — Standardise horizons and metric definitions before a single table is drawn
`PROJECT.md` §7 names this the top threat to any SOTA claim, and the reproduction record shows why: one model ran at h=1 against a paper reporting {2,3,5,10,15}; another reports RMSE in thousands; a third logs MAPE as a fraction where its paper prints percentage points.

Fix the horizon grid at **{3, 5, 10, 15}** and the metric definitions at `score.py`'s. Save every baseline's **raw predictions**, then recompute every metric through **our** `score.py`. **Never copy a number out of a baseline's own log** — that is how three different metric conventions end up in one table.

**Deliverable (Day 15):** `export_baseline.py` with the split and mask carried across; the five exported datasets; the baseline queue launched and logging to `results/`; a status note recording MepoGNN's mobility limitation and the STOEP and MSGNN dispositions. Completion checkpoint: **Day 20**.

---

## Week-3 "definition of done" checklist
- [x] `ebola-train` environment created and pinned; `env_train.txt` exported.
- [x] `bundles.py` presents one interface over all five bundles; split, channel and group divergences absorbed in one place; self-check passes.
- [x] `Bundle.refit()` returns `(X, y, scaler)` as one object, rebuilding `X[:, :, 0]` (§0.3), pinned by a test.
- [ ] **Ebola support-origin count read from Day 11's printout; §0.8 resolution adopted if it is 0; [CONFIRM-P7] sent.**
- [ ] **Node-level-mean variance of `X[:,:,0]` recorded per dataset (§0.9); no second normalisation added inside the encoder.**
- [ ] Encoder honours the three coded constraints: identity added before degree normalisation (asserted, all five graphs, no NaN in `Â`); `C` structurally rejected by the shared trunk; adjacency sparse.
- [ ] **Temporal receptive field ≥ 20, asserted at construction, with dilations (1,2) failing the paired negative control (Correction B).**
- [ ] **No shared-trunk parameter has any dimension in {10, 47, 49, 61, 7165}, asserted.**
- [ ] **Spatial gate implemented; `g = 0` reproduces the temporal-only model exactly; normalised spatial contribution logged, raw `g` debug-only.**
- [ ] **Every Day-12 assertion ships with a negative control that plants the defect and requires the assertion to fire.**
- [ ] LTR degree-encoding feature implemented in the shared trunk, reusing `Â`'s degree vector rather than recomputing one (Task 12.1a).
- [ ] Shared trunk / per-disease adapter split implemented, adapter sized against 27 support cells.
- [ ] Quantile head at {0.05, 0.25, 0.5, 0.75, 0.95}, pinball loss; median is the point forecast.
- [ ] `score.py` extended with sMAPE (definition pinned, 0/0 excluded), peak intensity and peak timing; self-check passes; count-space and constant-node rules intact.
- [ ] Ebola metric rules pre-registered in `score.py`'s docstring while no Ebola number exists (Task 13.1).
- [ ] Loss space asserted (pinball in model space, metrics in count space) and scaler round-trip tested (Task 13.2).
- [ ] Single-disease results: 4 datasets × 4 horizons × 5 seeds, beating all three naive floors everywhere, reported as mean ± 95% bootstrap CI.
- [ ] Multi-disease results with disease-balanced sampling; both dengue samplers implemented; per-dataset help/hurt recorded.
- [ ] Task 14.4 2×2 (gate × topo-aug) run with annealed augmentation; spatial contribution attributed against the §0.9 diagnostic.
- [ ] Baseline export adapter carries our split **and** our mask; queue launched; dispositions documented.
- [ ] Everything in `results/*.json`, one record per `model × dataset × horizon × seed × metric`. No number typed by hand.
- [ ] `ebola.npz` untouched.
- [ ] [CONFIRM-P1]–[CONFIRM-P6] sent to the client.

## Decisions to confirm (send Day 11 — all six land this week)
- **[CONFIRM-P1]** Encoder family: **dilated temporal convolution + GCN, direct multi-horizon, plus an LTR degree-encoding feature adopted from the EpiGNN lineage** (default) vs recurrent + attention. Default chosen partly to keep Week-4 meta-learning tractable; LTR added after a literature check against EpiGNN's published architecture found it constraint-clean and cheap (Task 12.1a) — its siblings GTR and RAGL were checked and rejected in the same pass.
- **[CONFIRM-P2]** Cross-disease sampling: **uniform over the four development datasets** (default) vs proportional-to-data (98% dengue) vs square-root weighting. Distinct from the within-dengue sampler already settled in Phase 2.
- **[CONFIRM-P3]** Epidemiology-informed component: **ablation only, not in the primary model** (default) vs built into the primary.
- **[CONFIRM-P4]** Output head: **quantile / pinball** (default, because quantiles invert exactly through `expm1`) vs Gaussian mean-variance.
- **[CONFIRM-P5]** Adaptation scope: **per-disease FiLM adapter + head only** (default, sized against 27 Ebola support cells) vs full fine-tune.
- **[CONFIRM-P6]** Baseline suite: **EpiGNN, Cola-GNN, HeatGNN, MTGNN (control), MepoGNN (adaptive variant only)**, with STOEP and MSGNN excluded-and-documented unless their blockers resolve within their timeboxes.

**Three further decisions raised by this revision — send with the other six.**

- **[CONFIRM-P7] — the consequential one.** Ebola few-shot adaptation under the frozen protocol
  (§0.8). **Default: adopt the split protocol** — left-padded short windows for adaptation only,
  h = 10 and h = 15 acknowledged as structurally zero-shot, and the Week-5 headline reframed as
  zero-shot with light short-horizon adaptation. The alternative is re-cutting the Ebola support
  definition, which changes a frozen bundle and invalidates the Phase-2 gate suite — not
  recommended. **This changes what Week 5 can claim, so it should be confirmed, not assumed.**
- **[CONFIRM-P8]** Spatial gate (Task 12.5): **learned gate, default on** vs unconditional graph
  mixing. Default chosen so the encoder nests the graph-free model and the spatial channel cannot
  degrade Week-5 transfer.
- **[CONFIRM-P9]** Topology-robust training (Task 14.4): **annealed edge dropout, default on, with
  the 2×2 reported** vs fixed-graph training. Default chosen because one training graph is already
  known mis-specified (`data_audit.md` §1.11).

---

## Appendix A — Measured facts the trainer must handle

Verified directly against `data/processed/*.npz`, not inferred from documentation.

- `y == X[:, :, 0]` exactly, in every bundle. Both are `inc_norm`. See §0.3.
- Split representations differ: dengue and Ebola ship `[N,T]` masks; the three influenza bundles ship **integers only** (`train_end`, `val_end`) and no mask arrays.
- `meta['node_country']` exists for **dengue and Ebola only**. `score.score_bundle` on influenza without a synthesised group map raises `KeyError`.
- Ebola's `X` has **5** channels (`deaths_norm`, extended); all others have 4. `meta['core_feature_idx'] == [0,1,2,3]` in all five — `transfer_view()` is identical across diseases, which *is* the disease-agnosticism guarantee.
- Four influenza nodes have adjacency row-sum zero: `japan_10`, `japan_19`, `us-states_1`, `us-states_9`. Add the identity **before** degree normalisation.
- Dengue `A_geo` dense is **205 MB** float32 with **40,936** non-zeros (20,468 undirected edges). Use sparse tensors.
- Dengue mask density **0.2175**; influenza all three **1.000**; Ebola **0.4095**.
- Dengue is **98.1%** of development observed cells and **98.5%** of development nodes.
- Valid origins at w=20, max h=15, before phase filtering: dengue **1,375**, us-regions **751**, us-states **326**, japan **314**.
- Dengue splits: 1,249,109 train / 524,972 val / 422,180 test cells; 475 nodes have no training cell and take their country's pooled scaler.

## Appendix B — Week-3 run matrix

RTX 3060, 12 GB. Direct multi-horizon means one run yields all four horizons — the single largest saving available.

| Block | Runs | Day | Bound by |
|---|---|---|---|
| Single-disease encoder | 4 datasets × 5 seeds = **20** | 13 | overnight wall-clock |
| Multi-disease joint | 2 samplers × 5 seeds = **10** | 14 | overnight wall-clock |
| **Baselines** | 4–5 models × 4 datasets × 4 horizons × 5 seeds ≈ **320–400** | launched 15, **completes in Week 4** | compute, not developer |

If the schedule slips, cut baseline **seeds** before cutting baseline **models** — a three-seed comparison across five comparators is far more defensible than a five-seed comparison across two. Whatever is cut, state it in the paper: a silently truncated comparison reads as a complete one.

## Appendix C — What Week 3 hands to Week 4

Week 4 (Days 16–20: leave-one-disease-out transfer, meta-learning, the k-shot sweep, and the Day-19 freeze of the Ebola protocol) cannot start without:

1. **`bundles.py`** — LODO iterates over datasets and cannot afford per-disease branching.
2. **The trunk/adapter split** — the meta-learning inner loop adapts *only* the adapter, and the k-shot sweep measures exactly that.
3. **The quantile head** — changing the head after Week 4 invalidates every transfer result.
4. **`score.py` and `results/*.json`** — the LODO tables are generated, not typed.
5. **The single-disease baseline-comparable numbers** — the ceiling that zero-shot transfer is measured against.
6. **The baseline queue running** — its Day-20 checkpoint sits inside Week 4.
7. **A settled [CONFIRM-P7]** (§0.8). Week 4's k-shot sweep and Day-19 protocol freeze both assume
   Ebola support origins exist. If the count is 0 and the split protocol is not adopted, the sweep
   has nothing to sweep over and Day 19 freezes a protocol that cannot run. **This is the one
   Week-3 item that can silently invalidate a whole Week-4 day**, so close it during Day 12 rather
   than discovering it on Day 18.

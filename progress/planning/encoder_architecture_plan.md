# Week 3 — Disease-Agnostic Shared Spatio-Temporal Encoder: Architecture Plan

**Status:** decided. This document freezes the encoder architecture for Phase 3 and is the
implementation contract. Where a choice remains open it is marked **[OPEN]** with a default that
is safe to build against — never block on one.

**Decision.** The shared encoder is **Option 1: a factorised, strictly inductive
temporal-per-node encoder followed by gated inductive spatial message passing**, with a direct
multi-horizon head and a light per-disease adaptation surface.

**Reads:** `PROJECT.md` §8 (Week 3), `data_audit.md` §4.6 (the three coded constraints) and
§1.11 / §3.x (graph construction), `bundles.py` (the only data interface), `schema_spec.md`.

**Revision 2 — post-review.** Ten defects were found in revision 1 and are resolved here. Four
were blocking and are worth knowing about before reading further:

| # | Defect | Where fixed |
|---|---|---|
| R10 | **Ebola has zero support origins** — support cells end at t=7, earliest target is t=22 | **§2.1** |
| R1 | TCN receptive field was **8, not 15** — would have ignored 60% of the lookback silently | §3.2 |
| R2 | Per-node normalisation **erases the signal the spatial channel propagates** | §3.1 |
| R5 | `A_mob` is absent from the released bundles — single-relation is forced, not chosen | §3.3 |

Non-blocking fixes: gate/augmentation interaction (§4, §9.1), gate reporting must be
norm-normalised (§3.4), the disease-agnosticism claim was overstated (§8), loss space (§5),
Ebola metric degeneracy (§5), statistical reporting (§9).

**Build order.** §2.1 diagnostic → §3.2/§3.1/§3.3 fixes → model code → §8 gates green → ablations.
Do not reorder; §2.1 is a go/no-go on the Week-4 design.

---

## 0. Why this architecture (the one-paragraph defence)

The transfer claim rests on *one set of shared weights* running unchanged over graphs of
10 / 47 / 49 / 61 / 7,165 nodes across disjoint geographies. That single requirement eliminates
most of the state of the art: Graph WaveNet, MTGNN and AGCRN all carry a learned node-embedding
dictionary `E ∈ ℝ^{N×d}`, and EpiGNN carries a learnable `W^s ∈ ℝ^{N×N}` — parameters indexed by
node, sized by `N`, and therefore both untransferable and a direct disease tell. Cola-GNN, STAN,
MepoGNN and CausalGNN are single-disease by construction or bake in compartmental parameters that
violate disease-agnosticism if placed in the trunk. The closest transferable prior art — PEMS
(ICLR 2024) and CAPE (2025) — proves cross-disease pre-training works but is **purely temporal
with no spatial encoder**. An inductive spatial channel over a shared temporal trunk is precisely
the unoccupied cell, and it is the contribution.

A factorised (temporal-then-spatial) trunk is preferred over STGCN-style interleaving for a
second reason established in review: the diseases do not share a transmission kernel. Dengue
spreads over short-range vector-suitable geography; Ebola spread by distance- and
population-weighted human movement through the Guéckédou–Lofa–Kailahun tri-border corridor.
A mis-specified graph is therefore expected, not exceptional. Factorisation **quarantines** that
error in one gated channel; interleaving lets it contaminate the temporal representation at every
block. The gate (§3.4) makes the encoder *nest* the graph-free model, so the spatial channel can
only help — and the gate value itself becomes a reportable quantity.

---

## 1. Non-negotiable constraints (fail the build if violated)

These are enforced by tests in §8, not by discipline.

| # | Constraint | Source | Enforcement |
|---|---|---|---|
| C1 | Shared encoder sees **only** `bundle.transfer_view()` — exactly 4 channels | audit §4.6.2 | `test_no_covariates` |
| C2 | **No parameter in the shared trunk may have any dimension equal to `N`** | inductivity | `test_no_node_indexed_params` |
| C3 | Identity added **before** degree normalisation | audit §4.6.1 | `test_isolated_node_finite` |
| C4 | Scaler **refit at every rolling origin** via `bundle.refit(fit_mask)` | audit §4.6.3 | `test_scaler_refit_per_origin` |
| C5 | Direct multi-horizon `h ∈ {3,5,10,15}`, **no autoregressive rollout** | protocol | `test_no_rollout` (single forward pass) |
| C6 | Lookback `w = 20`; an example is an **origin**, consuming the whole graph | `bundles.py` | `test_window_shape` |
| C7 | Loss and metrics computed **only on observed cells**, phase decided by where the **target** lands | `bundles.origins()` | `test_masked_loss` |
| C8 | Ebola gets **no validation split**; all selection on development diseases | PROJECT.md §5 | `test_ebola_never_in_selection` |

**C2 is the load-bearing one.** It is what makes the transfer claim mechanically true rather than
asserted. Implement it as a runtime assertion over `named_parameters()` given the set of node
counts {10, 47, 49, 61, 7165}: no shared-trunk parameter may have a dimension in that set.

---

## 2. Data interface

The encoder consumes `bundles.py` and nothing else. Do not open `.npz` directly; do not branch on
disease name anywhere in model or training code.

```python
from bundles import load, DEV_BUNDLE_NAMES, W, HORIZONS
b = load("influenza_japan")
Z = b.transfer_view()          # [N, T, 4]  the ONLY model input
A = b.A_geo                    # [N, N]
origins = b.origins(phase="train")
X, y, scaler = b.refit(fit_mask)   # per rolling origin, C4
```

**Batching.** One training example = one `(bundle, origin t)` pair. The input slice is
`Z[:, t-w+1 : t+1, :]` → `[N, 20, 4]`; targets are `y[:, t+h]` for `h ∈ HORIZONS` → `[N, 4]`,
each masked by `M[:, t+h] & phase_mask[:, t+h]`.

**Mixed-disease minibatches.** Do **not** pad graphs to a common `N`. Accumulate gradients over
one graph at a time (a "batch" = k origins drawn across bundles, stepped sequentially with
`loss.backward()` each, one `optimizer.step()` per k). This preserves size-agnosticism, keeps
memory flat, and makes the uniform-vs-country-balanced sampler ablation (PROJECT.md §8) a
sampler-level change only.

---

## 2.1 BLOCKER — Ebola few-shot is structurally impossible under the frozen protocol

**Run this before writing any model code.** It is a go/no-go on the entire Week-4 design.

```python
for name in BUNDLE_NAMES:
    b = load(name)
    print(name, {p: len(b.origins(phase=p)) for p in b.masks()})
```

**Expected result: `ebola {'support': 0, 'query': ...}`.** The arithmetic:

- Ebola is 61 districts × **T = 52** weekly steps, 2014-04-05 → 2015-03-28.
- The support set is a calendar prefix through **2014-05-24**, which is week index **t = 7**
  (04-05, 04-12, 04-19, 04-26, 05-03, 05-10, 05-17, 05-24). All 27 support cells lie in `t ∈ [0,7]`.
- `origins()` requires `t ≥ w−1 = 19` and `t + 15 ≤ T−1 = 51`, so the earliest possible target
  cell is `t + 3 = 22`.
- Support targets end at 7; the earliest target is 22. **The sets are disjoint. There are zero
  support origins, and the inner loop has nothing to supervise on.**

*(Derived from the audit's stated dates, not from the `.npz` — confirm with the snippet above.)*

### Resolution: separate the adaptation protocol from the evaluation protocol

Only the evaluation protocol is frozen, so this is legitimate.

1. **Left-pad short windows, for adaptation only.** Permit origins `t < 19`, left-padding the
   window to 20 steps with zeros and `obs_mask = 0`. The mask channel exists for exactly this and
   a causal TCN handles it natively. The evaluation protocol is untouched.
2. **Accept that h=10 and h=15 are structurally zero-shot for Ebola.** Even with padding, a
   support target needs `t + h ≤ 7`: h=3 admits origins `t ∈ [0,4]`, h=5 admits `t ∈ [0,2]`, and
   **h=10 and h=15 admit none, ever.** Adaptation therefore supervises short horizons only, and
   long-horizon Ebola performance measures pure transfer. This is a finding to state in the
   manuscript, not a defect to hide — and a reviewer will find it if you don't.
3. **Reframe the headline as zero-shot with light short-horizon adaptation.** Only 9 of 61
   districts have any support cell at all, so 52 districts are zero-shot regardless. That is both
   the honest framing and the stronger claim.

Do not proceed to §3 until the diagnostic has been run and the outcome recorded.

---

## 3. Architecture

```
Z [N, 20, 4]
   │
   ├─► InstanceNorm over the window (per node, incidence channel only)   §3.1
   │
   ├─► TemporalEncoder: dilated causal TCN, shared across all nodes      §3.2
   │        nodes → batch dim;  [N,20,4] → [N, d]        d = 64
   │
   ├─► SpatialMixer: L layers of mask-aware inductive message passing    §3.3
   │        h_s = MP(h, Â)      Â = D̃^{-1/2}(A+I)D̃^{-1/2}   (identity FIRST)
   │
   ├─► Gate: h_out = (1-g)·h + g·h_s,  g = σ(MLP(h)) ∈ (0,1)             §3.4
   │
   ├─► [per-disease] FiLM adapter: γ ⊙ h_out + β                          §3.5
   │
   └─► Head: Linear(d → |H|)  →  ŷ [N, 4]   direct, one shot             §3.6
```

### 3.1 Window normalisation — **SUPERSEDED: do not normalise inside the encoder**

> **Revision 3.** `fit_scalers_masked` returns `mean`/`std` of shape `[N]` — the released transform
> is already **log1p + per-node z-score**. `X[:, :, 0]` therefore arrives at the trunk with every
> node standardised. Adding any second normalisation would double-normalise, distort the sin/cos
> channels' relative amplitude, and collide with the C4 refit. **The encoder consumes
> `transfer_view()` as-is.**
>
> This also resolves the concern below in an unexpected direction: cross-node magnitude information
> is *already* gone, and it is gone by necessity, not oversight — node-level magnitude is a proxy
> for population, hence geography, hence disease identity, which C1 forbids. So the spatial channel
> transports **phase and shape co-movement**, not magnitude gradients. That is real signal (wave
> propagation is a timing phenomenon) but it is a narrower claim, and the manuscript must make it.
>
> Retain only the **diagnostic**: variance of node-level means in `X[:, :, 0]`, per dataset, logged
> once. Expect ≈ 0. Without it, a small spatial contribution in §9.1 will be misattributed to the
> architecture. See `Phase3_Week3_Developer_Execution_Guide.md` §0.9.

*Historical, for the record — the superseded specification:*

#### ~~Window normalisation — per-graph, not per-node~~
Instance normalisation (RevIN-style) on the incidence channel **only**; sin/cos and mask pass
through untouched. Store `(μ, σ)` and re-apply on the output so predictions return to model space.
Orthogonal to — not a replacement for — the C4 rolling-origin scaler refit. Guard `σ` with
`ε = 1e-5` (Ebola has all-zero windows).

**Statistics are computed per-graph-per-window (over all nodes jointly), not per node.** This is
a correction, and it matters more than it looks. Per-node normalisation z-scores each node's
window independently, which erases exactly the cross-node level gradient the spatial channel
exists to propagate — a surging district and a quiet district both arrive at the mixer as
mean-0/var-1 sequences. Ablation 1 would then likely report "the graph adds nothing", and the
cause would be §3.1 deleting the signal before §3.3 ever ran. Per-graph normalisation still
solves the cross-disease scale problem it was introduced for (a 7,165-node dengue graph and a
61-node Ebola graph look alike to the trunk) without flattening within-graph structure.

Expose as an ablation axis, `--norm {graph,node,none}`, default **`graph`**.

**Required diagnostic (gates the interpretation of ablation 1).** Log the variance of node-level
window means *after* normalisation. If it collapses to ≈ 0 under `--norm node`, that is direct
evidence the spatial signal was destroyed upstream, and any "graph adds nothing" result under
that setting is an artifact and must not be reported as a finding.

### 3.2 TemporalEncoder (shared, inductive)
Dilated causal TCN — the Graph-WaveNet temporal block with the adaptive-adjacency machinery
removed. **5 layers, dilations `[1, 2, 4, 8, 16]`, kernel 2**, gated activation
`tanh(W_f * x) ⊙ σ(W_g * x)`, residual + skip connections, GELU, dropout 0.1. Output: take the
last timestep of the skip sum → `h ∈ ℝ^{N×d}`, `d = 64`.

**Receptive field must be asserted, not assumed.** For a dilated causal TCN,
`RF = 1 + Σ (k−1)·dᵢ`. The specified stack gives `1 + (1+2+4+8+16) = 32 ≥ w = 20`. An earlier
draft specified kernel 2 with `[1,2,4]`, which is RF **8** — it would have silently ignored 60%
of the lookback window while training, converging and producing plausible numbers. Enforce at
construction:

```python
rf = 1 + sum((kernel_size - 1) * d for d in dilations)
assert rf >= W, f"TCN receptive field {rf} < lookback {W}"
```

This is a §8 gate with a negative control: dilations `[1,2,4]` at kernel 2 must fail.

Nodes go into the batch dimension, so the module is node-count-agnostic for free and satisfies C2
trivially. **[OPEN]** A 1-layer GRU (`d=64`) is the drop-in alternative; build the TCN first,
keep the interface identical, and make the swap an ablation (`--temporal {tcn,gru}`).

### 3.3 SpatialMixer (inductive, mask-aware)
`L = 2` layers. Each layer:

1. **Self-loop before normalisation** (C3): `Ã = A + I`, then `D̃ = diag(Ã·1)`,
   `Â = D̃^{-1/2} Ã D̃^{-1/2}`. Never invert `D` from `A` alone — four influenza nodes (Alaska,
   Hawaii, Hokkaidō, Okinawa) have degree 0 and will produce `NaN`. Assert `Â` is finite.
2. **Mask-aware aggregation.** Ebola reporting is sparse; smoothing over unobserved neighbours
   propagates reporting artefacts as epidemiological signal. Renormalise over *observed*
   neighbours only: weight edge `(i,j)` at origin `t` by `M[j, t]`, then re-normalise rows. A node
   with no observed neighbours falls back to its self-loop — which C3 guarantees exists.
3. **Update:** `h ← GELU(Â h W_self_and_neigh)` with GraphSAGE-style separate self and neighbour
   weights (`W_self`, `W_neigh` ∈ ℝ^{d×d}) plus residual. Both are `d×d` — no `N` anywhere.

**[OPEN] Multi-relational spatial input — deferred, not cancelled.** Running parallel channels
over contiguity / distance-decay / k-NN with attention over *relations* (not nodes) is the
principled answer to kernel mis-specification. It is deferred because a distance-decay adjacency
is a **new artifact not present in the frozen bundles**, and centroids currently live in `C`.
Building it requires a gated addition to `build_datasets.py` plus an audit entry, and the edge
weights must be normalised per-graph (e.g. by that graph's median edge length) so absolute scale
cannot act as a disease tell. Ship the single-relation version first; open this only if §9's
decision threshold fires.

**Single-relation is forced, not chosen — state this in the manuscript.** The Week-3 constraints
list an optional mobility adjacency `A_mob` for Japan/US, but `bundles.Bundle` has no `A_mob`
field and `bundles.load()` never reads one: the released `.npz` bundles carry geographic
adjacency only. Mobility was not carried into the frozen bundles, so multi-relational spatial
mixing is out of scope for this phase as a matter of data availability. Say so explicitly in the
limitations — silence here reads as oversight.

### 3.4 The gate (this is a contribution, not plumbing)
`g = σ(MLP(h))`, a per-node scalar from a small `d → d/4 → 1` MLP — data-dependent, so inductive
and C2-clean. `h_out = (1 - g) · h + g · h_spatial`.

Three reasons this is required:
- **Graceful degradation.** If geography is uninformative for a disease, `g → 0` and the encoder
  collapses to the graph-free temporal forecaster. The spatial channel can only help.
- **Defect mitigation.** The audit records the dengue graph as **block-diagonal with zero
  cross-border edges**, justified on a premise it marks *False* (Brazil borders four of the twelve
  countries; the Triple Frontier is among them). The dengue topology is known-mis-specified. The
  gate bounds the damage.
- **It is reportable** — but *not* as raw `g`. A raw gate value is not scale-free: `g` is
  comparable across diseases only if `‖h_spatial‖` and `‖h‖` are, and they are not. A model can
  reach identical effective mixing with `g = 0.2` and a large-norm spatial branch. Report the
  **normalised spatial contribution** instead:

```python
contrib = (g * h_s.norm(dim=-1)) / ((1 - g) * h.norm(dim=-1) + g * h_s.norm(dim=-1) + 1e-8)
```

  Log mean and IQR of `contrib` per disease per epoch; keep raw `g` as a debug metric only. Never
  put an uncalibrated scalar in a figure caption.

### 3.5 Adaptation surface (what the inner loop touches)
Everything above is **shared**. Per-disease adaptation is confined to:
- a **FiLM** modulation `γ, β ∈ ℝ^d` (2·64 = 128 params), and
- the **head** (`d × |H|` + bias = 260 params).

≈ 388 adaptable parameters per disease. This is deliberate: Ebola's 27 support *cells* (see §2.1
— they yield very few usable adaptation origins, and none at h=10/15) can move a few hundred
parameters, and cannot possibly discover a transmission topology. The burden of
learning structure sits in pre-training and in graph construction, never in the inner loop.
Week 4's MAML inner loop adapts exactly this set; the trunk is the outer loop. Build the
`adaptation_params()` / `shared_params()` split now so Week 4 is a training-loop change only.

### 3.6 Head — **quantile, not point**

> **Revision 3, superseding the point head + Huber loss specified here.** The Week-3 guide's Task
> 12.4 wins on an argument this plan did not account for: `invert_scaler` is `expm1(σ·z + μ)`,
> strictly monotone, and **quantiles are equivariant under monotone transforms**. A predicted 95th
> percentile in model space inverts to exactly the 95th percentile in count space. A Gaussian's
> mean and variance do not survive `expm1`. Since Week 5 needs CRPS, PICP and interval width in
> count space, the quantile head makes every uncertainty number exact by construction.

`Linear(d → |H| × |Q|)` = `d → 20`, emitting all four horizons × five quantile levels
{0.05, 0.25, 0.5, 0.75, 0.95} in a single forward pass (C5). **Pinball loss, not Huber** — which
also retires the δ=1.0 scale question in §5; the loss-space assertion there still stands. The
median (τ = 0.5) is the point forecast for RMSE/MAE/PCC. No decoder, no scheduled sampling, no
rollout.

---

## 4. Topology-robust pre-training (the headline ablation)

During development-disease training, perturb the graph every origin:
- edge dropout `p ∈ [0, 0.3]`;
- random rewiring of a small edge fraction;
- **[OPEN]** contiguity ↔ k-NN swap, once k-NN adjacencies exist.

The encoder never learns to trust a single topology; it is explicitly trained *for* graph
mis-specification, which is the Ebola condition and — per the audit — the dengue condition too.
Ship as `--topo-aug {none,edge_drop,rewire,full}`, default `edge_drop`. **Never perturb at
evaluation.**

**This is in direct tension with the §3.4 gate, and the two can cancel.** Under aggressive
augmentation the optimal policy is `g → 0` everywhere: the encoder learns the graph is unreliable
*because you made it unreliable*, and the paper's two novel components neutralise each other.
Two mitigations, both required:

- **Anneal.** Ramp `p_drop` linearly from 0 → 0.3 over the first 30% of training, so the gate
  learns on a trustworthy graph before robustness pressure arrives. Ship as
  `--topo-aug-schedule {constant,linear}`, default **`linear`**.
- **Measure the interaction, not the margins.** The primary table is the **2×2**
  `{gate on, g≡0} × {topo-aug none, edge_drop}` on Ebola, not two separate one-way ablations
  (§9.1). One-way ablations cannot distinguish "augmentation helps" from "augmentation killed the
  gate and the temporal trunk carried it".

---

## 5. Loss, metrics, memory

**Loss.** **Pinball loss** over the five quantile levels (§3.6, revised), masked to observed target cells, averaged per horizon then summed. *(Superseded: masked Huber δ=1.0.)*
Huber over MSE because Ebola and dengue are heavy-tailed and zero-inflated. Cells where
`M[:, t+h] == 0` contribute exactly zero — assert the gradient is unchanged when those targets are
overwritten with garbage (a negative control, §8).

**Loss space is normalised space, and this must be asserted.** δ=1.0 is scale-dependent, and two
transforms are in play (§3.1 instance norm, C4 refit scaler), so "Huber δ=1.0" is meaningless
without pinning the space. The loss is computed **after both transforms**, where targets are
≈ unit-scale and δ=1.0 sits near one standard deviation. Guard it:

```python
assert targets.abs().median() < 10, "loss is not being computed in normalised space"
```

All **metrics** are computed in **count space**, after inverting both transforms. Add a
round-trip test: `denorm(norm(x)) ≈ x` to 1e-5. Double-applying or mis-ordering these two
transforms is the single easiest silent bug in the pipeline.

**Metrics.** RMSE / MAE / PCC / sMAPE from the existing `metrics.py`. Wire **`score.py`'s
country-macro** into the eval loop as the dengue headline (per node → mean within country → mean
across the 12 countries equally), so Brazil's 77% node share does not dominate. Report node-mean
alongside it for transparency.

**Ebola metric handling is pre-registered here, before any numbers are seen.** Ebola is
zero-inflated at mask density 0.2175 with several all-zero districts, where PCC is near-
meaningless and sMAPE is undefined at zero. Fix now, so the choice cannot look post-hoc:

- **MAE and RMSE are primary for Ebola.**
- **PCC** is reported only over districts with ≥ 5 observed non-zero cells, and the qualifying
  district count is reported alongside it.
- **sMAPE** uses `denominator + ε` with ε = 1.0 count, or is dropped for Ebola entirely — decide
  now, record the decision, do not revisit after seeing results.

**Memory (8 GB, 7,165 dengue nodes).** Full-graph forward is comfortable: `[7165, 20, 64]` fp32
≈ 37 MB per activation. Keep `A` as a sparse COO tensor, use `torch.sparse.mm`, enable AMP
(`autocast`), and hold `d = 64`, `L = 2`. Fallbacks in order if OOM: (1) gradient checkpointing on
the TCN, (2) neighbour sampling / subgraph batching over dengue only, (3) `d = 48`.
Do **not** densify a 7,165² matrix — that alone is 205 MB fp32 and will end the run.

---

## 6. Module layout

```
models/
  encoder.py        SharedEncoder (TemporalEncoder + SpatialMixer + Gate)
  temporal.py       DilatedTCN, GRUEncoder
  spatial.py        InductiveMixer, normalise_adj (identity-first), mask_aware_adj
  adapters.py       FiLM, Head, shared_params()/adaptation_params()
  windows.py        origin → (Z_window, targets, target_mask), instance norm + inverse
train/
  loop.py           training/eval loop, per-origin scaler refit, gradient accumulation
  sampler.py        uniform (primary) | country-balanced (ablation)
  topo_aug.py       edge dropout / rewiring
tests/
  test_encoder_invariants.py    the §8 gates
configs/
  encoder_base.yaml
```

`SharedEncoder.forward(Z, A, M_t) -> h [N, d]` is the only public entry point. It must never
receive a disease name, `C`, or anything else.

---

## 7. Hyperparameters and selection protocol

| Param | Default | Notes |
|---|---|---|
| `d` (hidden) | 64 | matches the reference epidemic GNNs; fits 8 GB |
| TCN layers / dilations | 3 / [1,2,4] | receptive field 15 ≤ w |
| Spatial layers `L` | 2 | 3+ over-smooths; ablate |
| Dropout | 0.1 | |
| Optimiser | AdamW, lr 1e-3, wd 1e-4 | cosine decay, 5-epoch warmup |
| Grad clip | 1.0 | |
| Seeds | 42, 52, 62, 72, 82 | frozen protocol, 5-seed reporting |
| Early stopping | dev val, patience 15 | **never on Ebola** (C8) |

**Selection protocol (C8, non-negotiable).** All encoder hyperparameters are selected on the
development diseases' validation folds; adaptation settings by leave-one-development-disease-out.
Then **frozen**. Ebola's query set is scored **once**, with a fixed number of adaptation steps and
no early stopping on query. Ebola carries no validation split by design. Any code path that reads
an Ebola metric before the final run is a correctness bug, and `test_ebola_never_in_selection`
exists to catch it.

---

## 8. Test gates — including negative controls

Mirror the Phase-2 discipline: a gate that cannot fail proves nothing. Each gate below ships with
a paired control that plants the defect and requires the gate to fail.

| Gate | Assertion | Negative control |
|---|---|---|
| `test_no_node_indexed_params` | no shared param has a dim ∈ {10,47,49,61,7165} | add an `nn.Parameter(torch.zeros(47))` → must fail |
| `test_size_agnostic` | weights trained on 47 nodes run on 61 and 7,165 unchanged | reshape a weight to `N` → must fail |
| `test_isolated_node_finite` | output finite for a degree-0 node | normalise without `+I` → must produce `NaN` |
| `test_no_covariates` | `C` unreachable from the trunk; input is exactly 4 channels | pass 5 channels → must fail |
| `test_masked_loss` | garbage in unobserved targets leaves the gradient bit-identical | drop the mask → must fail |
| `test_scaler_refit_per_origin` | scaler stats differ across two origins | reuse the headline scaler → must fail |
| `test_permutation_equivariance` | permuting nodes permutes outputs identically | break with a positional term → must fail |
| `test_no_rollout` | exactly one forward pass per prediction | |
| `test_gate_nests_graphfree` | forcing `g=0` reproduces the temporal-only model exactly | |
| `test_receptive_field` | TCN `RF ≥ w = 20` (§3.2) | kernel 2 with `[1,2,4]` (RF 8) → must fail |
| `test_norm_roundtrip` | `denorm(norm(x)) ≈ x` to 1e-5 (§5) | |

**What permutation equivariance does and does not prove.** It proves node-order independence.
It does **not** prove disease-agnosticism, and the plan should not claim it does. The encoder can
still infer which pathogen it is looking at from `N`, graph density, seasonal amplitude and mask
sparsity — none of which any test here removes. The defensible claim is narrower and exactly two
things: **no static covariate reaches the trunk (C1), and no parameter is node-indexed (C2)**.
That is disease-agnosticism enforced at the *parameterisation* level, not the *information*
level. Concede this explicitly in the manuscript; a reviewer will otherwise write
"the model can identify the pathogen from dynamics alone" and be right.

---

## 9. Ablations (these become the paper's tables)

1. **PRIMARY — the 2×2:** `{gate on, g ≡ 0} × {topo-aug none, edge_drop}`, 5 seeds each,
   scored on **Ebola**. This replaces the two separate one-way ablations, for the reason given
   in §4: run separately they cannot distinguish "augmentation helps" from "augmentation
   suppressed the gate and the temporal trunk carried the result".
2. **Normalisation:** `--norm {graph,node,none}`. Report the §3.1 node-level-mean-variance
   diagnostic beside it. A "graph adds nothing" result under `--norm node` is an artifact, not a
   finding, and must not be reported as one.
3. **Fusion:** factorised (ours) vs interleaved STGCN-style — tests the quarantine argument.
4. **Temporal module:** TCN vs GRU.
5. **Spatial depth:** `L ∈ {1,2,3}`.
6. **Sampler:** uniform (primary) vs country-balanced.
7. **Mask-aware aggregation:** on vs off.
8. **Negative control:** shuffled adjacency — performance *must* degrade on dengue; if it does
   not, the spatial channel is decorative and that must be reported honestly.

**Statistical reporting (required — 5 seeds with no intervals will not survive review).** All
comparisons are **paired on identical origin sets**. Report mean ± **95% bootstrap CI over
origins** as the primary evidence, plus a **paired Wilcoxon signed-rank across seed-matched runs**
as supporting evidence for every headline comparison (ours vs each baseline, and each ablation).
With n = 5 seeds the test is underpowered — lead with the CI and do not over-claim significance
off five points.

**Baselines to run under the identical pipeline:** PEMS and CAPE (temporal-only pre-trained,
the closest prior art), a TSFM zero-shot, STGCN, and Graph WaveNet / AGCRN retrained per-graph —
the last as the demonstration that adaptive-graph models win in-distribution but **cannot move
weights across graph sizes at all**.

**Decision thresholds that change the plan.** If the spatial channel adds < 2–3% over a
temporal-only baseline on Ebola few-shot, demote spatial mixing to a light residual, lead with the
temporal trunk, and open the multi-relational work in §3.3. If dengue OOMs after all three
fallbacks, drop to STGCN-style Chebyshev `K ≤ 2`.

---

## 10. Definition of done (Week 3)

- [ ] **§2.1 origin diagnostic run and outcome recorded** (blocks everything else)
- [ ] `SharedEncoder` trains on all four development bundles with **one set of weights**
- [ ] All §8 gates pass **and** every negative control fails as designed
- [ ] 5-seed results on dev test splits, with 95% bootstrap CIs: RMSE/MAE/PCC/sMAPE + dengue
      country-macro; Ebola reported per the §5 pre-registered metric rules
- [ ] Normalised spatial contribution (§3.4) logged per disease; raw `g` debug-only
- [ ] §3.1 node-level-mean-variance diagnostic logged for each `--norm` setting
- [ ] `shared_params()` / `adaptation_params()` split exists and is unit-tested (Week 4 depends on it)
- [ ] Encoder runs a forward pass on the Ebola bundle **without any weight reshaping** — the
      mechanical proof of the transfer claim (no scoring, no adaptation, no metric read; C8)
- [ ] Trusted baselines re-run under the common pipeline (identical data/splits/horizons/metrics/seeds)
- [ ] Config + seeds + env captured; everything committed (PROJECT.md §9 traceability debt)

---

## 11. Risks

- **Ebola few-shot may be infeasible as specified** (§2.1). This is the top risk and it is
  resolvable, but not by ignoring it — h=10 and h=15 admit no support target under any windowing
  choice, so the long-horizon Ebola claim is zero-shot by construction.
- **The gate may collapse to 0 everywhere**, making the spatial channel decorative. This is a
  *finding*, not a failure — but it must be detected (§9.1 2×2 + the shuffled-adjacency control)
  and reported, not hidden. Note the §4 tension: aggressive augmentation can *cause* the collapse,
  so a collapsed gate under `edge_drop` is not evidence that geography is uninformative.
- **Normalisation can manufacture a false negative** (§3.1). Per-node normalisation removes the
  cross-node level gradient the spatial channel propagates; the resulting "graph adds nothing"
  would be an artifact. The default is per-graph and the diagnostic is mandatory.
- **Dengue's block-diagonal graph is a known defect** (audit §1.11: the "countries are not
  adjacent" justification is marked False). Topology randomisation mitigates; the honest framing
  in the paper is that the encoder is trained to be robust to exactly this class of error. Do not
  quietly rebuild the dengue graph mid-phase — that invalidates the frozen bundles and the 86 gates.
- **Over-smoothing at `L ≥ 3`** on the dense dengue graph; keep `L = 2` and ablate.
- **Instance norm interacts with the C4 scaler refit** — they are orthogonal but easy to
  double-apply. `test_scaler_refit_per_origin` plus an explicit round-trip test (normalise →
  denormalise → identity) should cover it.
- **Schedule.** Weeks 3–6 hold the entire novel contribution. Build the trunk minimally, get the
  gates green, then ablate — do not gold-plate the encoder before Week 4's meta-learning proves
  the adaptation surface is the right size.

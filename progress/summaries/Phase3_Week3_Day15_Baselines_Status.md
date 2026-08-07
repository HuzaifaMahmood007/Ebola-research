# Phase 3 · Week 3 · Day 15 — Baselines under the common pipeline (G6)

**Date:** 2026-07-27 · **Scope:** stand up the baseline comparators on our data / splits / horizons /
metrics / seeds, so the SOTA claim rests on beating *these*, not the naive floors. Ebola untouched.

## Suite (client-confirmed)

| Model | Env | Device | Datasets | Runs |
|---|---|---|---|---|
| **EpiGNN** | `epignn` (torch 1.11 cu113) | GPU | dengue-⅓ + 3 influenza | 4 × 4h × 5s = 80 |
| **Cola-GNN** | `colagnn` (torch cpu) | CPU | 3 influenza | 3 × 4h × 5s = 60 |
| **HeatGNN** | `heatgnn` (torch 1.13 cpu) | CPU | 3 influenza | 3 × 4h × 5s = 60 |
| **MTGNN** (control) | `mtgnn` *(to create, GPU)* | GPU | dengue-⅓ + 3 influenza | 4 × 5s = 20 (rollout, all-h/run) |
| MepoGNN | — | — | **DROPPED** | — |

## Decisions (each surfaced to the client, chosen by them)

- **Data:** export transposes our [N,T] → **[T,N] raw counts** + dense N×N adjacency. Each baseline
  uses its **own native scaler**; we rescore in count space.
- **Split — influenza:** native **50/20/30** (matches ours). Must pass `--train .5 --val .2 --test .3`
  to every model — Cola/Heat default to 0.6/0.2/0.2 and would otherwise train past our val boundary.
- **Split — dengue:** **per-cell mask patch** (per-country split, leakage-free) — EpiGNN + MTGNN only.
- **Dengue scale:** subsampled to **1/3 nodes stratified by country** (deterministic seed 20260715 →
  2392/7165, all 12 countries kept). Repos built for ≤49 nodes can't hold 7,165.
- **Cola/Heat:** **influenza-only** — their envs are CPU and dengue-⅓ (2,392 nodes) is too slow on CPU.
  Dengue among the lineage is **EpiGNN only** (GPU).
- **MepoGNN dropped:** both variants require a mobility/OD matrix we don't have (`A_mob=None`). Dynamic
  needs the OD tensor; Adaptive still needs a static commuter matrix to seed its graph + drive SIR
  propagation. Property of our data, not a model failure → reproducibility-appendix note.
- **Scoring:** every run dumps **count-space test predictions**; `score_baseline.py` rescores them
  through **our `score.py`** (country-macro dengue, node-level influenza). **No baseline's own printed
  metric is ever used** (Task 15.3).
- **Caveat (disclosed):** the encoder stays scored on **full** dengue (7,165 nodes) while baselines run
  on the 1/3 subset — dengue is **not like-with-like**; state it in the paper.

## What's built and verified — ALL FOUR MODELS

- `export_baseline.py` — 4 dev datasets → `baselines/_exported/<ds>/` (matrix.txt, adj.txt, masks.npz,
  meta.json), dengue subsampled 2392/7165. Self-check passes.
- `export_mtgnn.py` — MTGNN's `{train,val,test}.npz` (x [B,20,N,1], y [B,15,N,1] with per-cell NaN
  masking) + `adj.pkl` (A_geo+I so MTGNN's `-eye` recovers A_geo) + `test_origins.npy`, in
  `baselines/MTGNN/data_ours/<ds>/`. Origin counts match the guide (dengue 1375, us-regions 751,
  us-states 326, japan 314). Self-check passes.
- `score_baseline.py` — rescores `baselines/_preds/*.npz` → `results/baselines/*.json` in the encoder's
  record schema.
- **Patches:** EpiGNN/Cola/Heat `train.py` dump count-space test preds; **EpiGNN `data.py`+`train.py`**
  carry our per-cell dengue split via a mask-driven loader + masked loss (influenza path untouched);
  **MTGNN** `trainer.py` null_val→nan (keeps true zeros, masks our NaN cells), seed enabled, preds
  saved at {3,5,10,15} from the rollout.
- **Verified end-to-end in each model's own env:** EpiGNN (influenza **and** dengue masked-split),
  Cola-GNN, MTGNN. HeatGNN patch is identical to Cola's (same lineage) — smoke once on first launch.

> **Note:** the prediction/result files currently in `_preds/` and `results/baselines/` are from short
> **smoke** runs (2–8 epochs) — mechanism checks only. The full-epoch runs below overwrite them.

## Run commands — full matrix

Run from a shell where `conda` is on PATH (PowerShell). Data is staged. Seeds {42,52,62,72,82}.

**Lineage models** — one `--horizon` per run, horizons {3,5,10,15}; always pass the split ratios:

```powershell
# EpiGNN (env: epignn, GPU) — dengue + 3 influenza. From the EpiGNN repo root.
cd baselines\EpiGNN
conda run --no-capture-output -n epignn python src/train.py `
  --dataset dengue --sim_mat dengue-adj --window 20 --horizon 3 --seed 42 `
  --train .5 --val .2 --test .3 --batch 16
#   dengue is big: use --batch 16 (or 8) to fit 12 GB. Influenza can use the default batch.
#   repeat for dataset in {dengue, influenza_japan, influenza_us-regions, influenza_us-states},
#            horizon in {3,5,10,15}, seed in {42,52,62,72,82}.

# Cola-GNN (env: colagnn, CPU) — 3 influenza only. From colagnn/src.
cd baselines\colagnn\src
conda run --no-capture-output -n colagnn python train.py --model cola_gnn `
  --dataset influenza_japan --sim_mat influenza_japan-adj `
  --window 20 --horizon 3 --seed 42 --train .5 --val .2 --test .3

# HeatGNN (env: heatgnn, CPU) — 3 influenza only. From HeatGNN-14DB/src.
cd baselines\HeatGNN-14DB\src
conda run --no-capture-output -n heatgnn python train.py --model HeatGNN `
  --dataset influenza_japan --sim_mat influenza_japan-adj `
  --window 20 --horizon 3 --seed 42 --train .5 --val .2 --test .3
```

**MTGNN** (env: mtgnn, GPU) — dengue + 3 influenza. **One run per (dataset, seed)** emits every
horizon. From `baselines/MTGNN`. `--subgraph_size` must be < num_nodes (use 8 for us-regions' 10 nodes;
20 is fine elsewhere):

```powershell
cd baselines\MTGNN
conda run --no-capture-output -n mtgnn python train_multi_step.py `
  --data data_ours/dengue --adj_data data_ours/dengue/adj.pkl --num_nodes 2392 `
  --in_dim 1 --seq_in_len 20 --seq_out_len 15 --buildA_true False --gcn_true True `
  --subgraph_size 20 --device cuda:0 --seed 42 --save ./save_ours/ --batch_size 16
#   num_nodes per dataset: dengue 2392, influenza_japan 47, influenza_us-regions 10 (subgraph_size 8),
#   influenza_us-states 49.  repeat for seed in {42,52,62,72,82}.
```

After any batch of runs, rescore everything in `_preds/` through our `score.py`:

```powershell
conda run -n ebola-train python score_baseline.py    # -> results/baselines/*.json
```

## Compute note
EpiGNN dengue ≈ 24 s/epoch on the 3060 (early-stops). MTGNN rollout != our direct multi-horizon, so
its long-horizon error compounds — stated as a caveat. Lineage models re-train once per horizon; MTGNN
once per (dataset, seed).

## Other suite dispositions (unchanged from reproduction record)

- **STOEP** — paper table ≠ shipped dataset/metric; timeboxed, excluded-and-documented unless resolved.
- **MSGNN** — not reproducible in our environment (CUDA 10.1 / Linux); report the blocker in the
  reproducibility appendix.

## Completion checkpoint: **Day 20** (queue is compute-bound and runs through Week 4).

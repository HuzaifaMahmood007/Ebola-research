# Emerging Disease Forecasting Framework

**A generalizable spatio-temporal framework for emerging infectious-disease forecasting** —
transferable representations, few-shot adaptation, calibrated uncertainty, faithful explanation.

An emerging pathogen leaves almost no history at the moment a forecast matters most. Rather than
learn from one disease's own long record, this framework **learns transferable spatio-temporal
structure from data-rich diseases (dengue, influenza) and carries it, few-shot, to a data-scarce
one (Ebola)** — with calibrated uncertainty and faithful explanation. The contribution is occupying
a precise, empty intersection in the literature: a forecaster that is simultaneously
disease-agnostic by construction, transferable across diseases, few-shot adaptable to a genuinely
unseen pathogen, and calibrated + explained.

The authoritative, current description of the science and plan is **[PROJECT.md](PROJECT.md)**.
This README is the engineering entry point: layout, environments, and how to reproduce.

---

## Status

| Phase | Scope | State |
|---|---|---|
| Phase 1 | Formalisation, schema, evaluation protocol, competitive analysis, baseline reproduction | ✅ complete |
| Phase 2 | Five harmonised, leakage-gated datasets (dengue, 3× influenza, Ebola) | ✅ complete |
| **Phase 3** | The modelling contribution (Weeks 3–6) | **in progress** |

**Phase 3 · Week 3 (the core framework build):**
- Day 11 ✅ `bundles.py` (one interface over five bundles) + `ebola-train` environment.
- Day 12 ✅ the disease-agnostic encoder (`models/`), all invariant gates green.
- Day 13 ✅ `score.py` full metric layer + single-disease trainer (`train/loop.py`) + naive floors.
- Day 14 ⬜ multi-disease joint training and the samplers.
- Day 15 ⬜ baseline re-runs under the common pipeline.

`ebola.npz`'s query set is **untouched** until Week 5, by design.

---

## Repository layout

```
data/processed/*.npz        the five frozen, gated datasets (+ config.json, env.txt, env_train.txt)
bundles.py                  ONE interface over all five bundles — no code downstream branches on disease
score.py                    metrics: country-macro aggregation, sMAPE/peak/PCC; UQ metrics stubbed (Week 5)
models/                     the shared spatio-temporal encoder (encoder_architecture_plan.md §6)
  config.py                 architecture constants (source of truth; configs/encoder_base.yaml mirrors)
  temporal.py               DilatedTCN (receptive field asserted >= lookback)
  spatial.py                identity-first normalisation, mask-aware adjacency, GraphSAGE mixer, LTR
  encoder.py                SharedEncoder = temporal + LTR + gated spatial mixing; the C1/C2/C3 guards
  adapters.py               per-disease FiLM + quantile head; pinball loss; shared/adaptation split
  windows.py                origin -> input window + per-horizon targets/masks
train/loop.py               single-disease training + naive floors -> results/*.json
tests/test_encoder_invariants.py   the §8 gates, each with a negative control that must fire
configs/encoder_base.yaml   the frozen hyperparameters (human record)
loaders/                    the four per-disease entry points -> data/processed/*.npz (python -m loaders.<name>)
diagnostics/                one-off probes, audits, reports -- day11_diagnostics.py, capacity_probe.py,
                             ldo3_report.py, data_quality.py, and 14 more (python -m diagnostics.<name>)
progress/                   dated progress notes, decisions, results write-ups, planning docs (not code)
```

Phase-2 data-build code (frozen — do not edit): `to_schema.py`, `build_datasets.py`,
`fetch_gadm.py`, `dengue_aliases.py`, `japan_*.py`, and the Phase-2 tests `test_leakage.py`
(86 gates + 6 negative controls), `test_schema.py`, `test_dengue_7_1.py`,
`test_influenza_covariates.py`. `dengue_coverage.py` and `ebola_audit.py` are diagnostics
now (`diagnostics/`), not part of the frozen build path.

The four per-disease entry points that build `data/processed/*.npz` live in `loaders/`
(covid, dengue, ebola, influenza) and run as modules from the repo root, e.g.
`python -m loaders.covid_load`.

---

## Environments

Two conda environments, deliberately separate (the geometry stack and the training stack never
co-resolve):

| Env | Contents | Runs |
|---|---|---|
| `ebola` | `environment.yml` — python 3.10, numpy, pandas, geopandas, libpysal, shapely, openpyxl | the data build + leakage suite (anything touching geometry) |
| `ebola-train` | python 3.11 + torch (CUDA 12.4) + numpy + scipy + pandas | all Phase-3 training and evaluation |

```bash
# data / build env
conda env create -f environment.yml

# training env (pinned in data/processed/env_train.txt)
conda create -n ebola-train python=3.11 numpy scipy pandas -c conda-forge
conda run -n ebola-train pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Windows note: `pandas`' MKL and `torch` both link OpenMP; `train/loop.py` sets
`KMP_DUPLICATE_LIB_OK=TRUE` before importing them.

---

## Reproduce

Run scripts as modules from the repo root (the `models`/`train`/`tests`/`loaders`/`diagnostics`
packages need it):

```bash
# rebuild the five datasets (ebola env) — deterministic, gated by 86 checks
conda run -n ebola python build_datasets.py

# data-layer checks (ebola-train env)
conda run -n ebola-train python bundles.py                     # five-bundle self-check
conda run -n ebola-train python -m diagnostics.day11_diagnostics  # §0.8/§0.9 close-out readings
conda run -n ebola-train python score.py                       # metric self-check

# encoder invariants (all gates + negative controls)
conda run -n ebola-train python -m tests.test_encoder_invariants

# training
conda run -n ebola-train python -m train.loop --smoke          # fast end-to-end check
conda run -n ebola-train python -m train.loop --all            # the 20-run single-disease matrix
```

Every number lands in `results/*.json`, one record per `model × dataset × horizon × seed × metric`.
Week-6 tables are generated from these — no number is typed by hand.

---

## Design invariants (enforced in code, not by discipline)

The transfer claim rests on a shared encoder that is *mechanically* disease-agnostic. These are
gated in `tests/test_encoder_invariants.py`:

- **C1** the shared trunk sees only the 4 core transfer channels — `C` (static covariates) is a
  `TypeError`, not a review comment; disjoint geography makes a raw centroid a disease tell.
- **C2** no shared-trunk parameter has any dimension in {10, 47, 49, 61, 7165} — the five graph
  sizes — so one weight set runs on every graph unchanged.
- **C3** the identity is added **before** degree normalisation (`Â = D̃^{-1/2}(A+I)D̃^{-1/2}`);
  four influenza nodes have degree 0 and would divide by zero otherwise.
- The scaler is per-node z-score already applied; the encoder adds **no** second normalisation
  (§0.9). Any rolling-origin refit rebuilds `X[:,:,0]`, not just `y` (`Bundle.refit`, §0.3).
- Pinball loss is model space; **all metrics are count space** (`score.py`, after `invert_scaler`).

---

## Key documents

- [PROJECT.md](PROJECT.md) — the current source of truth (thesis, status, plan).
- [progress/planning/encoder_architecture_plan.md](progress/planning/encoder_architecture_plan.md) — the frozen encoder design.
- [progress/planning/Phase3_Week3_Developer_Execution_Guide.md](progress/planning/Phase3_Week3_Developer_Execution_Guide.md) — the day-by-day Week-3 schedule.
- [progress/planning/data_audit.md](progress/planning/data_audit.md) — the authoritative data-methods record (86 gates, every alteration declared).
- [progress/planning/schema_spec.md](progress/planning/schema_spec.md) — the `DiseaseTensors` contract.
- `progress/` — dated summaries, decisions, results write-ups (see `progress/summaries|decisions|outcomes|planning`).
- `Reports/` — Phase-1 and Phase-2 reports and the manuscript.

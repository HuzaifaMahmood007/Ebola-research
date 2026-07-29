# MTGNN — published numbers

**Paper:** Wu, Pan, Long, Jiang, Chang, Zhang. *Connecting the Dots: Multivariate Time Series
Forecasting with Graph Neural Networks.* KDD 2020. arXiv:2005.11650
**Source:** <https://arxiv.org/pdf/2005.11650> · code <https://github.com/nnzhan/MTGNN>

## ⚠ MTGNN's paper contains no epidemic data

Its published evaluation is traffic / solar / electricity / exchange-rate and two traffic-forecasting
graph benchmarks. **There is no Japan-Prefectures, US-Regions or US-States row to validate against.**
This is consistent with its role in our suite: the non-epidemic control / floor.

Two ways to satisfy the client's condition for MTGNN — pick one and say which:

1. **Validate off-domain.** Run MTGNN on METR-LA and compare to Table 3 below. Proves our
   installation is faithful; says nothing about epidemic performance.
2. **Validate against a third-party published epidemic number.** MepoGNN and STOEP both re-ran MTGNN
   on Japan COVID and printed the result (rows quoted below). Not the authors' own number, but it is
   published, epidemic, and citable.

Recommendation: **(2), with (1) as the fallback** — it is the only route that puts MTGNN on an
epidemic axis, and it costs nothing extra because those tables are already transcribed here.

## Protocol

| | |
|---|---|
| Split, single-step datasets | chronological **60% / 20% / 20%** |
| Split, METR-LA / PEMS-BAY | chronological **70% / 20% / 10%** |
| Input length | 168 (single-step) · 12 (multi-step) |
| Output length | 1 (single-step) · 12 (multi-step) |
| Metrics | RSE + CORR (single-step) · MAE / RMSE / MAPE (multi-step) |

Dataset statistics (Table 1):

| dataset | #samples | #nodes | sample rate | in | out |
|---|---|---|---|---|---|
| traffic | 17,544 | 862 | 1 hour | 168 | 1 |
| solar-energy | 52,560 | 137 | 10 min | 168 | 1 |
| electricity | 26,304 | 321 | 1 hour | 168 | 1 |
| exchange-rate | 7,588 | 8 | 1 day | 168 | 1 |
| metr-la | 34,272 | 207 | 5 min | 12 | 12 |
| pems-bay | 52,116 | 325 | 5 min | 12 | 12 |

## Table 2 — single-step, MTGNN row

| dataset | metric | h=3 | h=6 | h=12 | h=24 |
|---|---|---|---|---|---|
| Solar-Energy | RSE | **0.1778** | 0.2348 | **0.3109** | **0.4270** |
| | CORR | **0.9852** | 0.9726 | **0.9509** | 0.9031 |
| Traffic | RSE | **0.4162** | 0.4754 | **0.4461** | **0.4535** |
| | CORR | **0.8963** | 0.8667 | **0.8794** | **0.8810** |
| Electricity | RSE | **0.0745** | 0.0878 | **0.0916** | **0.0953** |
| | CORR | **0.9474** | 0.9316 | **0.9278** | **0.9234** |
| Exchange-Rate | RSE | 0.0194 | 0.0259 | 0.0349 | 0.0456 |
| | CORR | 0.9786 | 0.9708 | 0.9551 | 0.9372 |

MTGNN+sampling row: Solar RSE 0.1875 / 0.2521 / 0.3347 / 0.4386 · Traffic 0.4170 / 0.4435 / 0.4469 /
0.4537 · Electricity 0.0762 / 0.0862 / 0.0938 / 0.0976 · Exchange 0.0212 / 0.0271 / 0.0350 / 0.0454.

Paper's own claim: lowers RSE by 7.24% / 3.88% / 4.83% at horizons 3 / 12 / 24 on traffic. It
explicitly **fails to improve on exchange-rate** (small graph, few training examples).

## Table 3 — multi-step, MTGNN row

| dataset | | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| METR-LA | MAE | 2.69 | 3.05 | 3.49 |
| | RMSE | 5.18 | 6.17 | 7.23 |
| | MAPE | 6.86% | 8.19% | 9.87% |
| PEMS-BAY | MAE | 1.32 | 1.65 | 1.94 |
| | RMSE | 2.79 | 3.74 | 4.49 |
| | MAPE | 2.77% | 3.69% | 4.53% |

Reference points on METR-LA: DCRNN 2.77/5.38/7.30% at h3; GraphWaveNet 2.69/5.15/6.90%;
MRA-BGCN 2.67/5.12/6.80%. MTGNN's claim is **on-par without a pre-defined graph**, not best-in-class.

## Table 4 — METR-LA ablation, mean ± sd over 10 runs (validation set)

| | MTGNN | w/o GC | w/o Mix-hop | w/o Inception | w/o CL |
|---|---|---|---|---|---|
| MAE | 2.7715 ± 0.0119 | 2.8953 ± 0.0054 | 2.7975 ± 0.0089 | 2.7772 ± 0.0100 | 2.7828 ± 0.0105 |
| RMSE | 5.8070 ± 0.0512 | 6.1276 ± 0.0339 | 5.8549 ± 0.0474 | 5.8251 ± 0.0429 | 5.8248 ± 0.0366 |
| MAPE | 0.0778 ± 0.0009 | 0.0831 ± 0.0009 | 0.0779 ± 0.0009 | 0.0778 ± 0.0010 | 0.0784 ± 0.0009 |

**This is the only dispersion MTGNN's paper publishes** — useful if we go route (1), because it gives
a real tolerance band to judge our repro against.

## Third-party published MTGNN numbers on epidemic data (route 2)

**MepoGNN** (Japan COVID, 47 prefectures, 6:1:1, mean ± 95% CI over seeds 0–4):

| | 3d | 7d | 14d | overall |
|---|---|---|---|---|
| RMSE | 297.6 ± 19.2 | 363.5 ± 37.9 | 443.5 ± 15.4 | 363.2 ± 20.5 |
| MAE | 102.4 ± 6.7 | 130.9 ± 13.1 | 168.3 ± 8.1 | 130.0 ± 8.3 |
| MAPE | 40.6 ± 0.8 | 49.1 ± 1.7 | 68.0 ± 2.9 | 50.7 ± 1.6 |
| RAE | 0.31 ± 0.02 | 0.39 ± 0.04 | 0.50 ± 0.02 | 0.39 ± 0.03 |

**STOEP** (same Japan COVID dataset; SMAPE not MAPE): 3d 307.2 / 105.8 / 41.1 / 0.32 · 7d 382.4 /
137.5 / 50.1 / 0.41 · 14d 443.5 / 168.3 / 80.2 / 0.52 · overall 363.2 / 135.0 / 50.7 / 0.49.

The two agree on RMSE to 3 significant figures at 14d and overall (443.5, 363.2) — strong evidence
both re-ran the same public MTGNN configuration. That makes these numbers safe to cite.

## Notes for our validation table

- Our pipeline has MTGNN at **80/80 predictions, complete** — it is the only baseline fully done.
- Whichever route we pick, state on the table that MTGNN's own paper is non-epidemic. A reviewer who
  checks will find that immediately.

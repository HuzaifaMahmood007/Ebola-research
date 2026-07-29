# MepoGNN — published numbers

**Paper:** Cao, Jiang, Yang, Fan, Song, Shibasaki. *MepoGNN: Metapopulation Epidemic Forecasting with
Graph Neural Networks.* ECML-PKDD 2022, pp. 453–468.
**Extended version (numbers below):** *Metapopulation Graph Neural Networks: Deep Metapopulation
Epidemic Modeling with Human Mobility.* arXiv:2306.14857
**Source:** <https://arxiv.org/pdf/2306.14857> · code <https://github.com/deepkashiwa20/MepoGNN>

## Protocol the numbers are valid under

| | |
|---|---|
| Data | daily confirmed COVID-19 cases, **47 prefectures of Japan** |
| Mobility | Facebook Movement Range Maps + 2015 census OD — **required, not optional** |
| Input / output | 14 days in → 14 days out |
| Split | **6 : 1 : 1** chronological (test contains Japan's 5th infection wave) |
| Horizons reported | 3, 7, 14 days ahead + overall |
| Metrics | RMSE, MAE, MAPE, RAE |
| Repeats | seeds 0–4, reported as **mean ± 95% CI** |
| Training | curriculum learning, batch 32, MAE loss, Adam lr 1e-3, wd 1e-8, early stop patience 20, max 300 epochs, 4×2080Ti |

## Table I — MepoGNN rows

| variant | | 3d | 7d | 14d | overall |
|---|---|---|---|---|---|
| **MepoGNN (Dyn)** | RMSE | **135.9 ± 17.8** | **160.6 ± 4.5** | **253.2 ± 7.5** | **186.1 ± 5.0** |
| | MAE | **52.7 ± 4.6** | **67.6 ± 1.2** | 107.0 ± 3.0 | **74.3 ± 2.0** |
| | MAPE | **34.2 ± 0.7** | 41.7 ± 0.9 | 62.0 ± 2.0 | 44.4 ± 0.8 |
| | RAE | 0.16 ± 0.01 | **0.20 ± 0.00** | 0.32 ± 0.01 | **0.22 ± 0.01** |
| **MepoGNN (Adp)** | RMSE | 141.0 ± 7.2 | 174.6 ± 10.1 | 261.1 ± 16.0 | 196.2 ± 11.3 |
| | MAE | 54.3 ± 2.3 | 69.7 ± 4.2 | **105.1 ± 7.3** | 75.4 ± 4.7 |
| | MAPE | 34.9 ± 0.8 | 41.4 ± 1.6 | **60.1 ± 3.2** | **44.0 ± 1.6** |
| | RAE | 0.16 ± 0.01 | 0.21 ± 0.01 | 0.32 ± 0.02 | 0.23 ± 0.01 |

## Best baselines on the same table (overall RMSE)

| model | 3d | 7d | 14d | overall |
|---|---|---|---|---|
| ColaGNN | 221.7 ± 40.7 | 300.6 ± 61.2 | 388.3 ± 23.2 | 310.7 ± 31.4 |
| GraphWaveNet | 223.8 ± 46.6 | 259.9 ± 52.2 | 389.8 ± 20.8 | 294.7 ± 40.9 |
| AGCRN | 223.5 ± 28.5 | 253.1 ± 37.7 | 390.4 ± 105.8 | 322.7 ± 136.7 |
| MTGNN | 297.6 ± 19.2 | 363.5 ± 37.9 | 443.5 ± 15.4 | 363.2 ± 20.5 |
| DCRNN | 305.0 ± 9.8 | 323.8 ± 15.9 | 377.9 ± 11.1 | 335.0 ± 11.8 |
| STGCN | 375.6 ± 18.8 | 381.1 ± 17.7 | 430.2 ± 15.8 | 389.5 ± 7.9 |

## Notes for our validation table

- **Status in our suite: DROPPED** — both variants require a mobility / OD matrix and our schema ships
  `A_mob = None`. Dynamic needs the OD tensor; Adaptive needs a static commuter matrix to seed the
  graph and drive the SIR component. This is a property of **our data**, not a model failure, and it
  belongs in `reproduction_failure_log.md` **Section A** phrased that way.
- The Phase-1 reproduction of MepoGNN (Dynamic) passed *in or better than the published std bands* —
  the bands are the `± 95% CI` column above. That claim can now be checked cell by cell.
- This paper is one of the two published sources of an **MTGNN number on epidemic data** — see
  [MTGNN.md](MTGNN.md).
- The ECML-PKDD 2022 conference version and this extended arXiv version may differ in places. The
  numbers above are the **arXiv 2306.14857** version. If the conference table is cited instead, say
  which.

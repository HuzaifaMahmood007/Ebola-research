# MSGNN — published numbers

**Paper:** Qiu, Tan et al. *MSGNN: Multi-scale Spatio-temporal Graph Neural Network for Epidemic
Forecasting.* Data Mining and Knowledge Discovery 38:2348 (2024). arXiv:2308.15840
**Source:** <https://arxiv.org/pdf/2308.15840> · <https://link.springer.com/article/10.1007/s10618-024-01035-w>

## Protocol the numbers are valid under

| | |
|---|---|
| Data | JHU CSSE, **50 US states + 3,142 counties**, 2020-03-01 → 2021-07-01 (487 dates) |
| Target | **weekly** reported confirmed cases (daily is too noisy) |
| Eval window | Feb 2021 → Jul 2021, forecast issued each Sunday |
| Horizons | 1, 2, 3 weeks ahead |
| Subsets | **@500** and **@100** (top-N counties by case volume) |
| Metrics | MAE, MAPE, RMSE — computed at **county level**, then averaged |
| Baselines | live COVID-19 Forecast Hub submissions (Zoltar project 44) |

Dataset statistics (Table 1):

| level | series | #location | #dates | min | max | ave |
|---|---|---|---|---|---|---|
| US-State | confirmed | 50 | 487 | 0 | 73,854 | 1,660 |
| US-State | deaths | 50 | 487 | 0 | 4,417 | 450 |
| US-County | confirmed | 3,142 | 487 | 0 | 34,497 | 62 |
| US-County | deaths | 3,142 | 487 | 0 | 761 | 15 |

## Table 2 — MSGNN row

| subset | metric | 1 wk | 2 wk | 3 wk |
|---|---|---|---|---|
| @500 | MAE | **121.3** | **502.2** | **959.6** |
| | MAPE | **0.340** | **0.439** | **0.584** |
| | RMSE | **302.2** | **977.8** | 1867.8 |
| @100 | MAE | **321.5** | **1360.4** | **2588.5** |
| | MAPE | **0.283** | **0.432** | **0.571** |
| | RMSE | **594.8** | **1990.5** | 3840.9 |

## Strongest Forecast-Hub baselines on the same table

| method | @500 RMSE 1/2/3wk | @100 RMSE 1/2/3wk |
|---|---|---|
| COVIDhub-ensemble | 312.3 / 1035.9 / 1878.5 | 599.5 / 2108.6 / 3822.6 |
| CU-nochange | 327.7 / 1119.2 / 2006.1 | 643.4 / 2304.4 / 4144.3 |
| Microsoft-DeepSTIA | 341.3 / 980.0 / **1808.5** | 677.2 / 1992.7 / 3911.9 |
| CEID Walk | 371.8 / 1021.5 / 1724.7 | 742.6 / 2016.2 / 3931.2 |
| USC-SI kJalpha | 366.0 / 1072.7 / 2015.6 | 650.1 / 2081.0 / 3935.7 |

MSGNN's own claim: best MAE/MAPE/RMSE at 1 wk on both subsets; ~10% MAE gain over the best baseline
(132.2 → 121.3). It is **not** best on RMSE at 3 wk — CEID Walk (1724.7) and Microsoft-DeepSTIA
(1808.5) beat it on @500.

## Notes for our validation table

- **Status in our suite: NOT RUN.** Requires Ubuntu + CUDA 10.1; the project machine is Windows.
  `reproduction_failure_log.md` **Section A**, environment blocker.
- Even if the environment were solved, the evaluation is **not comparable to our pipeline**: weekly
  US-county COVID against live Forecast-Hub submissions, on horizons expressed in weeks, over an
  eval window rather than a fixed test split. Reproducing it would validate the install and nothing
  more. Worth saying explicitly rather than leaving it as an unexplained gap.
- No dispersion published — single values throughout.

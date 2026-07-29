# STOEP — published numbers

**Paper:** *Prior Knowledge-enhanced Spatio-temporal Epidemic Forecasting* — Spatio-Temporal
mechanism-aware Epidemic Predictor (STOEP). IJCAI-format submission, cites baselines up to ICML/KDD
2025.
**Local copy:** [docs/Prior Knowledge-enhanced Spatio-temporal Epidemic Forecasting.pdf](../docs/Prior%20Knowledge-enhanced%20Spatio-temporal%20Epidemic%20Forecasting.pdf)
**Code:** <https://github.com/stdi-lab/STOEP>

## ⚠ Read this before quoting a STOEP number

The paper reports on **two datasets**, and their overall rows are ~2.7× apart. Our failure log
currently records STOEP as irreproducible on the basis of a 63.5-vs-182.9 gap. **63.5 is the Flu
row.** Our run's RMSE 182.9 / RAE 0.224 sits next to the **COVID-19** row (169.1 / 0.21) — roughly
**8% off, not 3× off**. The shipped repo defaults to the COVID dataset.

| what we compared | RMSE | RAE |
|---|---|---|
| our run | 182.9 | 0.224 |
| paper, **Flu** overall (what we compared to) | 63.5 | 0.37 |
| paper, **COVID-19** overall (what we probably ran) | **169.1** | **0.21** |

**Action:** confirm which dataset the STOEP run used before it is written up as a Section-A failure.
If it was the COVID dataset, STOEP is an ~8% reproduction and may belong in the comparison table.
The SMAPE-vs-MAPE metric mismatch is a separate, still-open issue — the paper's third column is
**SMAPE**, and our run logged MAPE. Those are different numbers and cannot be compared at all.

## Protocol

| | |
|---|---|
| COVID-19 dataset | daily confirmed cases, **47 prefectures of Japan**, 2020-04-01 → 2021-09-21 (539 days), NHK source |
| Flu dataset | daily confirmed flu, **11 cities of Zhejiang Province, China**, 2023–2024 (731 days) |
| Mobility | Facebook Movement Range Maps (COVID) / Baidu Migration Map (Flu) — required |
| Split | **6 : 1 : 1** chronological, non-overlapping |
| Horizons | 3, 7, 14 days ahead + overall |
| Metrics | RMSE, MAE, **SMAPE**, RAE |
| Hardware | single NVIDIA RTX 5060, Adam + curriculum learning |

## Table 1 — COVID-19 dataset (Japan), STOEP row

| | 3d | 7d | 14d | overall |
|---|---|---|---|---|
| RMSE | **125.3** | **149.3** | **230.1** | **169.1** |
| MAE | **49.8** | **63.1** | **97.9** | **68.8** |
| SMAPE | **35.9** | 43.1 | 61.6 | 45.4 |
| RAE | **0.15** | **0.19** | 0.29 | **0.21** |

Nearest baseline, MepoGNN: 139.2 / 177.5 / 250.6 / **188.5** RMSE. Paper claims 10.3% lower RMSE,
8.9% lower MAE, 8.7% lower RAE than MepoGNN overall.

## Table 2 — Flu dataset (Zhejiang), STOEP row

| | 3d | 7d | 14d | overall |
|---|---|---|---|---|
| RMSE | **51.9** | **65.2** | **70.5** | **63.5** |
| MAE | **24.8** | **34.4** | 37.8 | **32.4** |
| SMAPE | **31.6** | **39.8** | 45.3 | **38.8** |
| RAE | **0.28** | **0.39** | 0.44 | **0.37** |

Nearest baseline, MepoGNN: 54.1 / 72.9 / 84.0 / **72.0** RMSE.

## MTGNN and ColaGNN rows from this table (third-party epidemic numbers)

| model | COVID overall RMSE/MAE/SMAPE/RAE | Flu overall |
|---|---|---|
| MTGNN [KDD20] | 363.2 / 135.0 / 50.7 / 0.49 | 143.4 / 87.6 / 71.9 / 0.88 |
| ColaGNN [CIKM20] | 294.2 / 106.4 / 50.2 / 0.33 | 126.7 / 112.1 / 110.0 / 1.27 |
| MepoGNN [ECML22] | 188.5 / 75.5 / 47.1 / 0.23 | 72.0 / 35.6 / 39.8 / 0.40 |
| CausalGNN [AAAI22] | 317.1 / 121.4 / 49.9 / 0.37 | 123.1 / 78.3 / 96.1 / 0.88 |

Note MTGNN's COVID overall RMSE here (363.2) matches MepoGNN's paper exactly — see [MTGNN.md](MTGNN.md).

## Notes for our validation table

- This local PDF is a **2025/2026-era version** (its baseline list includes TimeKAN ICLR25, DUET KDD25,
  LightGTS ICML25). If our original STOEP comparison used an earlier version's table, that is a second
  possible source of the mismatch. Pin the version we cite.
- STOEP is not currently a comparator in our suite; it was timeboxed and excluded. The finding above
  may reopen that. Cost of re-checking: reading which dataset flag the run used — minutes, not a rerun.

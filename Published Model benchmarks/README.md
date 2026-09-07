# Published Model Benchmarks

**Created:** 2026-07-30 · **Why:** client condition D10 / Work Order §1c — *"validate each reproduction
against the source paper's published numbers before using it as a comparator, and show me our
reproduction next to their reported figure. If we can't get close, that baseline doesn't go in the
comparison table."*

One file per baseline. Each file holds **only what the paper printed**, transcribed verbatim, with the
dataset, split, horizon set and metric definition that number is valid under. Nothing in here is our
number. The `our repro` column is filled in from `results/baselines/*.json` when the runs land.

**Rule:** a published number is never re-derived. If we can't cite a table for it, it doesn't go in
the "their reported" column.

---

## Index

| file | model | paper's own eval data | horizons | split | metric |
|---|---|---|---|---|---|
| [EpiGNN.md](EpiGNN.md) | EpiGNN | Japan-Pref, US-Regions, US-States, Spain/Australia-COVID | 3,5,10,15 (flu) · 3,7,14 (covid) | **50/20/30** | RMSE, PCC |
| [Cola-GNN.md](Cola-GNN.md) | Cola-GNN | Japan-Pref, US-Regions, US-States | 2,3,5,10,15 | **50/20/30** | RMSE, PCC |
| [HeatGNN.md](HeatGNN.md) | HeatGNN | Japan-Pref, US-Region, US-State, Australia-COVID | 2,5,7,12 | 60/20/20 | RMSE ×10³, PCC |
| [MTGNN.md](MTGNN.md) | MTGNN | solar, traffic, electricity, exchange-rate, METR-LA, PEMS-BAY | 3,6,12,24 · 3,6,12 | 60/20/20 · 70/20/10 | RSE/CORR · MAE/RMSE/MAPE |
| [MepoGNN.md](MepoGNN.md) | MepoGNN | Japan COVID, 47 prefectures | 3,7,14 d | 6:1:1 | RMSE, MAE, MAPE, RAE |
| [MSGNN.md](MSGNN.md) | MSGNN | JHU CSSE US counties (@500/@100) | 1,2,3 wk | Forecast-Hub eval window | MAE, MAPE, RMSE |
| [STOEP.md](STOEP.md) | STOEP | Japan COVID + Zhejiang Flu | 3,7,14 d | 6:1:1 | RMSE, MAE, **SMAPE**, RAE |
| [LTGCN-GTGCN.md](LTGCN-GTGCN.md) | LTGCN / GTGCN | Brazil, Spain | 1-step | split-first, 3-fold rolling | RMSE, SMAPE, MDA |

Sources: PDFs 1–3, 7–8 are in [docs/](../docs/). MTGNN, MepoGNN, MSGNN were fetched from arXiv
(URLs in each file).

---

## Three things this exercise turned up

### 1. STOEP is probably not a failed reproduction — we read the wrong table

`Phase3_Week4_Work_Order.md` §1d (retired 2026-09-07, in git history) recorded STOEP as *"paper table ≠ shipped dataset/metric. Overall
RMSE 63.5 in paper vs 182.9 on the run … RAE 0.37 vs 0.224. ~⅓ scale → different dataset."*

The paper prints **two** overall rows:

| dataset | RMSE | MAE | SMAPE | RAE |
|---|---|---|---|---|
| Flu (Zhejiang, 11 cities) | **63.5** | 32.4 | 38.8 | **0.37** |
| COVID-19 (Japan, 47 prefectures) | **169.1** | 68.8 | 45.4 | **0.21** |

63.5 / 0.37 is the **Flu** row. Our run reported RMSE 182.9, RAE 0.224 — which sits next to the
**COVID-19** row (169.1 / 0.21), i.e. **~8% off, not ~3×**. The shipped repo defaults to the COVID
dataset. The SMAPE-vs-MAPE mismatch is real and still needs settling, but the scale gap looks like a
table-selection error on our side, not a broken reproduction.

**Action:** re-check the STOEP run's dataset before it goes in the failure log as Section A. It may
belong in the comparison table instead. Detail in [STOEP.md](STOEP.md).

### 2. Cola-GNN's and EpiGNN's paper protocols are already our pipeline protocol

Both papers use **chronological 50%/20%/30%** on **Japan-Prefectures / US-Regions / US-States** — the
same split and the same three source datasets as our influenza bundles. Cola-GNN's paper horizons
{2,3,5,10,15} contain our {3,5,10,15} outright; EpiGNN's are {3,5,10,15} exactly.

So the "rerun at paper horizons on paper data" for these two is much smaller than budgeted — the only
delta from our pipeline runs is *their shipped copy of the data* vs *our harmonised bundle*. HeatGNN
is the genuine exception: 60/20/20 and horizons {2,5,7,12}, so it needs its own run.

### 3. Published re-runs of the same model disagree by up to 26%

HeatGNN's Table III re-ran Cola-GNN rather than copying its numbers. Against Cola-GNN's own paper, on
the same dataset and horizon:

| dataset | h | Cola-GNN's paper | HeatGNN's re-run of Cola-GNN | Δ |
|---|---|---|---|---|
| Japan-Prefectures | 2 | 929 | 1168 | **+25.7% worse** |
| US-Regions | 2 | 480 | 552 | +15.0% worse |
| US-States | 2 | 136 | 148 | +8.8% worse |
| US-States | 5 | 202 | 224 | +10.9% worse |

Two published papers, same model, same data, same metric, differences of 9–26%. This is the honest
tolerance band for "can't get close" and it is worth quoting in `reproduction_failure_log.md` — it
sets the bar the client's condition should actually be judged against.

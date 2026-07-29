# Cola-GNN — published numbers

**Paper:** Deng, Wang, Zhao, Liu, Ning. *Cola-GNN: Cross-location Attention based Graph Neural
Networks for Long-term ILI Prediction.* CIKM 2020.
**Local copy:** [docs/Cola-GNN Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction.pdf](../docs/Cola-GNN%20Cross-location%20Attention%20based%20Graph%20Neural%20Networks%20for%20Long-term%20ILI%20Prediction.pdf)

## Protocol the numbers are valid under

| | |
|---|---|
| Split | **chronological 50% / 20% / 30%** — same as our pipeline |
| Normalisation | 0–1 per location, fitted **on training data only** |
| Lookback | T = 20 weeks |
| Lead times | 2, 3, 5, 10, 15 — contains our {3,5,10,15} |
| Metrics | RMSE (↓), PCC (↑). Raw counts |
| Datasets | Japan-Prefectures (47), US-States (49, Florida dropped), US-Regions (10 HHS) |

## Table 3 — RMSE, Cola-GNN row

| dataset | lt=2 | lt=3 | lt=5 | lt=10 | lt=15 |
|---|---|---|---|---|---|
| Japan-Prefectures | **929** | **1051** | **1117** | **1372** | **1475** |
| US-Regions | **480** | **636** | **855** | **1134** | **1203** |
| US-States | **136** | **167** | **202** | **241** | **237** |

## Table 3 — PCC, Cola-GNN row

| dataset | lt=2 | lt=3 | lt=5 | lt=10 | lt=15 |
|---|---|---|---|---|---|
| Japan-Prefectures | 0.915 | 0.901 | 0.890 | 0.813 | 0.753 |
| US-Regions | 0.946 | 0.909 | 0.835 | 0.717 | 0.639 |
| US-States | 0.955 | 0.933 | 0.897 | 0.822 | 0.856 |

Relative gain over second-best claimed by the paper (RMSE): Japan 6.7 / 5.7 / 1.1 / 11.0 / 3.4 %;
US-Regions 5.3 / 7.6 / 4.6 / 2.0 / 2.3 %; US-States 8.7 / 7.2 / 5.2 / 7.3 / 5.2 %.

## Second-best baseline on the same table (ST-GCN), for context

| dataset | lt=2 | lt=3 | lt=5 | lt=10 | lt=15 |
|---|---|---|---|---|---|
| Japan-Prefectures | 996 | 1115 | 1129 | 1541 | 1527 |
| US-Regions | 697 | 807 | 1038 | 1290 | 1286 |
| US-States | 189 | 209 | 256 | 289 | 292 |

## Ablation (Table 4, Japan-Prefectures RMSE) — for sanity-checking a rerun

| variant | 2 | 3 | 5 | 10 | 15 |
|---|---|---|---|---|---|
| w/o temp | 912 | 1115 | 1310 | 1388 | 1517 |
| w/o loc | 942 | 1154 | 1199 | 1470 | 1576 |
| w/o geo | 1075 | 1105 | 1219 | 1417 | 1502 |
| full Cola-GNN | 929 | 1050 | 1117 | 1372 | 1475 |

(The full-model h=3 entry reads 1050 in Table 4 and 1051 in Table 3 — a rounding inconsistency in the
paper, not a transcription error here.)

## Notes for our validation table

- **Protocol matches ours** (50/20/30, T=20, RMSE+PCC, raw counts) and the horizon set is a superset
  of ours. The "rerun at paper horizons on paper data" is therefore cheap.
- Our earlier reproduction ran **h=1**, which is outside the paper's grid entirely — that is why it
  could not be validated, not because the model failed.
- **Independent re-runs of this model differ a lot.** HeatGNN's paper re-ran Cola-GNN and got
  Japan h2 = 1168 against this paper's 929 (+25.7%); US-States h5 = 224 against 202 (+10.9%). See
  [README.md](README.md) §3. Judge our own Δ% against that band, not against 0%.

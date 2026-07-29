# LTGCN / GTGCN — published numbers

**Paper:** Ezzat et al. *Spatio-temporal epidemic forecasting with graph-based transformer.*
International Journal of Health Geographics (2026) 25:33.
**Local copy:** [docs/Spatio-temporal epidemic forecasting with graph-based transformer.pdf](../docs/Spatio-temporal%20epidemic%20forecasting%20with%20graph-based%20transformer.pdf)

**Status: first-party.** These are our own group's models on our own Brazil/Spain data. There is no
external claim to reproduce — the "published number" and the "our number" are the same number. It is
listed here for completeness of the suite, not as a validation target.

## Protocol

| | |
|---|---|
| Data | Brazil: 5,300+ municipalities, 2020–2023, three subsets (unfiltered / cleaned 1,305 / top-40). Spain: provinces |
| Split | **split-first** (to prevent leakage), 3-fold rolling-origin cross-validation |
| Metrics | RMSE, SMAPE, **MDA** (mean directional accuracy) |
| Variants | LTGCN (node-wise temporal attention) · GTGCN (flattened global attention) · LinearTGCN |

GTGCN is evaluated **only on the top-40 subset** — full global attention over N=1,305 nodes is
O((NT)²) and computationally infeasible.

## Table 6 — Brazil, three subsets

| subset | metric | Persistence | GCRN | GraphWaveNet | LinearTGCN | LTGCN | GTGCN |
|---|---|---|---|---|---|---|---|
| Top 40 cities | RMSE | 5482.07 | **3872.72** | 3878.59 | 3933.80 | 3873.63 | 3882.10 |
| | SMAPE | 99.50% | 88.08% | 100% | 84.32% | **83.47%** | 99.74% |
| | MDA | 31.58% | 47.39% | **57.71%** | 49.68% | 52.02% | 43.30% |
| Cleaned (1,305) | RMSE | 1035.15 | 733.23 | 733.29 | **733.04** | 733.29 | not run |
| | SMAPE | 108.68% | 95.22% | 95.46% | **93.54%** | 95.43% | not run |
| | MDA | **55.78%** | 37.85% | 48.51% | 37.50% | 37.35% | not run |
| Full (all cities) | RMSE | 705.57 | 499.42 | 499.45 | **499.37** | 499.40 | not run |
| | SMAPE | 101.72% | 89.58% | 89.63% | **88.27%** | 89.11% | not run |
| | MDA | **74.60%** | 22.71% | 22.68% | 22.47% | 23.09% | not run |

## Table 8 — Spain, full dataset

| model | RMSE | SMAPE | MDA |
|---|---|---|---|
| Persistence | 1035.96 | 34.00% | 0.98% |
| GCRN | 1422.58 | 38.76% | 65.27% |
| Graph WaveNet | 1670.83 | 40.28% | **67.97%** |
| **LinearTGCN** | **682.92** | **24.74%** | 72.52% |
| LTGCN | 1763.79 | 46.52% | 64.18% |
| GTGCN | 1838.79 | 51.72% | 63.55% |

## Table 7 / 9 — fold-by-fold, avg ± std (the dispersion, which the single-value tables hide)

Brazil top-40 RMSE: Persistence 2434.98 ± 2178.20 · GCRN 1791.92 ± 1507.52 · LTGCN 1825.62 ± 1487.62 ·
GTGCN 1865.82 ± 1462.99.
Spain RMSE: Persistence 448.63 ± 415.72 · GCRN 544.77 ± 620.71 · LTGCN 670.00 ± 773.43 ·
GTGCN 704.63 ± 802.07.

**Fold 3 always tests against a COVID-19 peak window and produces RMSE 3–4× folds 1–2.** The sd
exceeds the mean on Spain. Any single-value RMSE from this paper is close to meaningless on its own —
which is the same point the client's dispersion rule (D3) is making.

## Notes

- The paper's own conclusion is that **LinearTGCN — a linear model — matches or beats both
  transformers** on structured data, and that a naïve persistence forecast achieves near-zero
  directional accuracy (MDA 0.98% Spain) while scoring competitively on RMSE and SMAPE. Directly
  relevant to Work Order §8a: simple floors are hard to beat, and RMSE alone hides it.
- RMSE here is biased toward high-population cities; the paper argues SMAPE and MDA are the more
  informative metrics for regional comparison. Same argument as our own scale-normalised-error item
  (§7).
- Nothing here is comparable to our pipeline (different countries, different resolution, 1-step
  forecasting, MDA not in our metric set). Cite for design context, not for the head-to-head table.

# HeatGNN — published numbers

**Paper:** Zheng, Jiang et al. *Epidemiology-informed Graph Neural Network for Heterogeneity-aware
Epidemic Forecasting.* (arXiv 2411.17372; IEEE journal format)
**Local copy:** [docs/Epidemiology-informed Graph Neural Network for Heterogeneity-aware Epidemic Forecasting.pdf](../docs/Epidemiology-informed%20Graph%20Neural%20Network%20for%20Heterogeneity-aware%20Epidemic%20Forecasting.pdf)

## Protocol the numbers are valid under

| | |
|---|---|
| Split | **60% / 20% / 20%** — **differs from our pipeline's 50/20/30** |
| Horizons | h ∈ {2, 5, 7, 12} — **differs from our {3,5,10,15}** |
| Metrics | RMSE (↓) **scaled ×10³ in the table**, PCC (↑) |
| Epochs | 1500, early stopping patience 200, batch size 32 (D6) |
| Datasets | Japan-Prefectures (47×348), US-Region (10×785), US-State (49×360), Australia-COVID (8×556) |

**Read the scaling.** Table III prints RMSE ÷ 1000. US-State h=2 reads `0.142` and means **142**
patient counts. Any comparison against our raw-count RMSE must multiply by 10³ first.

## Table III — HeatGNN row (RMSE ×10³ / PCC)

| dataset | h=2 | h=5 | h=7 | h=12 |
|---|---|---|---|---|
| Japan-Prefectures RMSE | 1.149 (=1149) | **1.378** (=1378) | **1.735** (=1735) | 1.685 (=1685) |
| Japan-Prefectures PCC | **0.917** | **0.884** | 0.780 | 0.773 |
| US-Region RMSE | **0.541** (=541) | **0.852** (=852) | **0.922** (=922) | 1.024 (=1024) |
| US-Region PCC | **0.941** | **0.866** | **0.856** | **0.824** |
| US-State RMSE | **0.142** (=142) | **0.186** (=186) | **0.200** (=200) | **0.240** (=240) |
| US-State PCC | **0.953** | 0.921 | **0.911** | 0.870 |
| Australia-COVID RMSE | **0.315** (=315) | **0.334** (=334) | **0.411** (=411) | **0.466** (=466) |
| Australia-COVID PCC | 0.993 | 0.993 | 0.991 | 0.986 |

## Nearest competitors on the same table (RMSE ×10³)

| model | Japan h2/h5/h7/h12 | US-Region | US-State | Australia |
|---|---|---|---|---|
| Cola-GNN | 1.168 / 1.573 / 1.758 / 1.690 | 0.552 / 0.871 / 1.054 / 1.011 | 0.148 / 0.224 / 0.228 / 0.241 | 0.368 / 0.406 / 0.463 / 0.533 |
| Epi-GNN | 1.503 / 1.448 / 1.738 / 1.695 | 0.563 / 0.881 / 0.938 / 1.025 | 0.150 / 0.190 / 0.205 / 0.251 | 0.325 / 0.381 / 0.416 / 0.491 |
| Epi-Cola-GNN | 1.118 / 1.544 / 2.549 / 2.164 | 0.547 / 0.872 / 1.160 / 1.365 | 0.148 / 0.219 / 0.268 / 0.303 | 0.803 / 0.854 / 0.912 / 1.124 |
| STGCN | 1.274 / 1.400 / 1.478 / 1.812 | 0.727 / 0.965 / 1.017 / 1.116 | 0.205 / 0.260 / 0.280 / 0.296 | 0.696 / 0.649 / 0.706 / 0.721 |

## Ablation (Table IV, RMSE ×10³) — sanity check for a rerun

| variant | Japan h2/h5/h7/h12 | US-Region | US-State | Australia |
|---|---|---|---|---|
| w/o PL | 1.230 / 1.453 / 1.804 / 1.759 | 0.631 / 0.943 / 1.055 / 1.141 | 0.146 / 0.188 / 0.192 / 0.223 | 0.487 / 0.460 / 0.451 / 0.455 |
| w/o TG | 1.229 / 1.439 / 1.762 / 1.657 | 0.740 / 1.039 / 1.142 / 1.182 | 0.149 / 0.202 / 0.215 / 0.238 | 0.350 / 0.366 / 0.362 / 0.429 |
| w/o TG+EIEL | 1.279 / 1.452 / 1.765 / 1.675 | 0.727 / 0.977 / 1.119 / 1.188 | 0.153 / 0.206 / 0.217 / 0.227 | 0.461 / 0.481 / 0.449 / 0.435 |
| HeatGNN | 1.149 / 1.378 / 1.735 / 1.685 | 0.541 / 0.852 / 0.922 / 1.024 | 0.142 / 0.186 / 0.200 / 0.240 | 0.315 / 0.334 / 0.411 / 0.466 |

## Notes for our validation table

- **This is the one baseline that genuinely needs a separate paper-protocol rerun**: both the split
  (60/20/20) and the horizon set {2,5,7,12} differ from our pipeline. Our earlier repro ran h=1,
  which is off-grid entirely.
- The paper's own numbers are single values with no dispersion, so our Δ% has nothing to be judged
  against statistically. Quote the Δ, note the absence of a published sd.
- **Our runs deviate from the paper by design:** we add self-loops to HeatGNN's staged adjacency
  because `getLaplaceMat` omits the identity and NaNs on isolated nodes (D7). Japan and US-States each
  have 2 degree-0 nodes; the paper never hit this because it presumably did not. **This deviation must
  appear on the validation table and in `reproduction_failure_log.md` Section B.**
- Australia-COVID is not in our bundle set, so three of the four dataset rows are usable to us.

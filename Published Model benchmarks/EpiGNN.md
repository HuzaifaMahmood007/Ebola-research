# EpiGNN — published numbers

**Paper:** Xie, Zhang, Wang, Yang, Xu. *EpiGNN: Exploring Spatial Transmission with Graph Neural
Network for Regional Epidemic Forecasting.* ECML-PKDD 2022.
**Local copy:** [docs/EpiGNN Exploring Spatial Transmission with Graph Neural Network for Regional Epidemic Forecasting.pdf](../docs/EpiGNN%20Exploring%20Spatial%20Transmission%20with%20Graph%20Neural%20Network%20for%20Regional%20Epidemic%20Forecasting.pdf)

## Protocol the numbers are valid under

| | |
|---|---|
| Split | **chronological 50% train / 20% val / 30% test** — same as our pipeline |
| Lookback | T = 20 — same as ours |
| Horizons | h ∈ {3, 5, 10, 15} influenza · {3, 7, 14} COVID |
| Metrics | RMSE (↓), PCC (↑). Raw counts, no scaling |
| Repeats | 5 runs with different random initialisation (paper reports a single value, not a sd) |
| Hardware | Nvidia Tesla K80, PyTorch 1.9.1 / CUDA 11.1 |

Dataset statistics as printed (Table 1):

| dataset | regions | length | min | max | mean | sd | granularity |
|---|---|---|---|---|---|---|---|
| Japan-Prefectures | 47 | 348 | 0 | 26635 | 655 | 1711 | weekly |
| US-Regions | 10 | 785 | 0 | 16526 | 1009 | 1351 | weekly |
| US-States | 49 | 360 | 0 | 9716 | 223 | 428 | weekly |
| Australia-COVID | 8 | 556 | 0 | 9987 | 539 | 1532 | daily |
| Spain-COVID | 35 | 122 | 0 | 4623 | 38 | 269 | daily |

## Table 2 — influenza, EpiGNN row

| dataset | metric | h=3 | h=5 | h=10 | h=15 |
|---|---|---|---|---|---|
| Japan-Prefectures | RMSE | **996** | **1031** | 1441 | 1470 |
| | PCC | 0.904 | 0.908 | 0.739 | 0.773 |
| US-Regions | RMSE | **589** | **774** | **984** | **1061** |
| | PCC | 0.912 | 0.842 | 0.749 | 0.694 |
| US-States | RMSE | **160** | **186** | **220** | **236** |
| | PCC | 0.935 | 0.907 | 0.865 | 0.861 |

Best-baseline context on the same table (Cola-GNN\*, reported from its own paper):

| dataset | metric | h=3 | h=5 | h=10 | h=15 |
|---|---|---|---|---|---|
| Japan-Prefectures | RMSE | 1051 | 1117 | **1372** | **1475** |
| US-Regions | RMSE | 636 | 855 | 1134 | 1203 |
| US-States | RMSE | 167 | 202 | 241 | 237 |

## Table 3 — COVID-19, RMSE

| model | Spain h3 | h7 | h14 | Australia h3 | h7 | h14 |
|---|---|---|---|---|---|---|
| EpiGNN | 135.54 | 162.51 | 186.41 | 71.42 | 153.07 | 287.90 |
| EpiGNN_exter (uses mobility) | 129.90 | 145.33 | 178.73 | — | — | — |

## Table 4 — runtime (s/epoch, single GPU) and params, h=5

| | Japan runtime / params | US-Regions | US-States |
|---|---|---|---|
| EpiGNN | 0.10 / 11K | 0.14 / 9K | 0.07 / 12K |
| Cola-GNN | 0.14 / 9K | 0.13 / 7K | 0.15 / 9K |

## Notes for our validation table

- **The protocol matches ours exactly** (50/20/30, T=20, h∈{3,5,10,15}, RMSE+PCC, raw counts). The
  only difference between "paper repro" and "pipeline run" is their shipped data copy vs our
  harmonised bundle. Budget the rerun accordingly — it is close to free.
- Prior reproduction: reproduces within ~1% on US-Regions h5 (`[[baseline-reproduction-status]]`).
  That number needs re-confirming against the row above (589/774/984/1061) before it is quoted.
- EpiGNN reports Cola-GNN's numbers **as printed in Cola-GNN's paper** (marked `*`), not re-run, and
  they match Cola-GNN's Table 3 exactly. Two papers agreeing is a useful transcription check.
- Paper's own headline: 5.6% lower RMSE than the best baseline on influenza, 13.4% on COVID.

# COVID-19 US-states bundle — EDA

**Date:** 2026-08-03
**Bundle:** `data/processed/covid_us-states.npz`, built by [`covid_load.py`](covid_load.py)
**Regenerate:** `conda run -n ebola python covid_eda.py`
**Source:** NYT `covid-19-data/us-states.csv` (archived 2023-03-23), sha256 `0b202f6ac8bad66b70b9e344c07155e3f1ce9338f86192af072b56ce448dfc43`

> Read §4 first. It is the finding that changes how this bundle should be used.

---

## 1. Shape, calendar, splits

```
X (49, 164, 4)   raw (49, 164)   A_geo (49, 49)   C (49, 3)
calendar   2020-02-01 -> 2023-03-18    T = 164 weeks    cadence W-SAT (MMWR)
```

| phase | weeks | span | cells | origins |
|---|---|---|---|---|
| train | 82 | 2020-02-01 → 2021-08-21 | 4,018 | 60 |
| val | 33 | 2021-08-28 → 2022-04-09 | 1,617 | 45 |
| test | 49 | 2022-04-16 → 2023-03-18 | 2,401 | 49 |

Standard `fixed_50_20_30`, `per_node_train` scaler — identical protocol to the other development
bundles. 164 weeks is the shortest development panel (influenza_us-states has 360 on the same nodes).

## 2. Missingness — clean, and there is nothing to impute

```
M density                 1.000000     unobserved cells: 0
obs_mask channel unique   [1.]
raw cells exactly zero    3.4%         (274 of 8,036)
leading all-zero weeks    min 0   median 5   max 7  (West Virginia)
```

`M` is all-ones by construction: NYT reports every state every day once that state's first case
lands, and absence before that is a genuine zero (no cases yet), not a missing observation. The
274 exact zeros are almost entirely the pre-outbreak head — median 5 leading weeks, max 7.

**Consequences worth stating.** COVID contributes **no zero-fill covariate shift**, so the LOCF input
fill is a no-op on this bundle; and `obs_mask` (core channel 3) is **constant 1.0** here, exactly as
it is on the three influenza panels. That means COVID *joins the influenza side* of the disease
identifier already documented in `Doubt.md` §3.3 rather than adding a new one — channel 3 still
separates {influenza, covid} from {dengue, ebola}.

Reporting cadence held up better than expected, which was the main risk with a 2022–23 tail:

| year | weeks | zero-increment cells | states with >50% zero weeks |
|---|---|---|---|
| 2020 | 48 | 10.6% | 0/49 |
| 2021 | 52 | 0.1% | 0/49 |
| 2022 | 53 | 0.6% | 0/49 |
| 2023 | 11 | 1.1% | 0/49 |

The 2020 figure is real epidemiology (few cases early), not dropout. Worst 2022+ reporters are
Nebraska 6%, Indiana 5%, Mississippi 3% — no state degrades into a mostly-flat series, so the
weekly-reporting transition many states made in 2022 did **not** damage this panel.

## 3. Target distribution

```
n = 8,036   mean 11,798.9   sd 28,786.2
median 4,562   p90 26,550   p99 111,658   p99.9 403,927
max 836,753  (California, week ending 2022-01-15)
skew 12.0    zeros 3.4%
```

Heavy right skew, as expected for weekly counts, though far less extreme than dengue (skew 178).

### Predictor for the §3.1 bias correction

Per node, `mean(y) / expm1(mean(log1p y))` over observed **train** cells — the gap between the
count-space mean (what RMSE rewards) and the count-space median (what the quantile head predicts):

| bundle | median node | p90 | max |
|---|---|---|---|
| **covid_us-states** | **3.03×** | 3.40× | 4.04× |
| influenza_us-states | 1.74× | 2.55× | 3.38× |

COVID's gap is **~1.7× larger than influenza's on the identical nodes**, so it should fit a visibly
larger `bias_c` than the influenza panels do, and the `encoder_mc` arm should matter more here.

### §1.3 graph diagnostic — variance of node-level means of `X[:,:,0]`

| bundle | variance |
|---|---|
| influenza_us-states | 0.0657 |
| dengue | 0.0234 |
| **covid_us-states** | **0.0090** |
| influenza_japan | 0.0016 |

COVID's cross-node level gradient is **7× flatter than influenza's on the same 49 nodes**. Per-node
z-scoring removes more from COVID than from flu, so the spatial channel has even less to propagate
here than on the panel it shares a graph with. That reinforces the §1.4/§1.5 conclusion: settle the
`gate_mode='off'` / shuffled-adjacency controls before investing anything further in graph structure.

> **Discrepancy to resolve, not to paper over.** The reviewer reported dengue at **0.948** for this
> statistic; this run gets **0.0234**. japan (0.0016) and us-states (0.0657) match exactly, so the
> method agrees everywhere except dengue. The likely cause is the denominator: dengue is 78% exact
> zeros, so a node mean taken over **all T** is crushed toward 0, while a mean over **observed cells
> only** is not. Both are defensible; they are different statistics and only one of them belongs in
> the paper. Do not quote either number until it is pinned down.

## 4. The split, and the finding that matters

| phase | national mean/week | peak week | node-mean |
|---|---|---|---|
| train | 419,883 | 1,661,031 | 8,569.0 |
| **val** | **1,189,000** | **5,139,436** | **24,265.3** |
| test | 431,608 | 811,443 | 8,808.3 |

**The Omicron peak lands inside the validation fold.** The national maximum (2022-01-15, 5.14M cases,
13.8× the median week) sits at val week 21 of 33.

Three consequences, all of which bite:

1. **The val fold is not representative of anything else.** Its mean week is **2.8× train and 2.8×
   test**, and its peak is **6.3× the largest week in test**. Early stopping selects the checkpoint on
   this fold, so model selection for COVID is dominated by a single once-in-the-panel event that never
   recurs in the scored period.
2. **The new §3.1 `bias_c` is fit on val.** It will be fit against Omicron magnitudes and then applied
   to a test fold that is roughly a sixth of that scale. Expect it to over-correct on COVID
   specifically. This is the first bundle where the val-only fitting rule is actively harmful rather
   than merely conservative.
3. **My earlier "train → test regime shift" note was wrong and is corrected here.** The
   **test/train national mean ratio is 1.03×** — train and test sit at almost identical levels. The
   regime shift is not across the train/test boundary; it is a spike *inside val*. The warning printed
   by `covid_load.py` overstated it and has been corrected.

**Recommendation:** keep `fixed_50_20_30`. Protocol consistency across diseases is worth more than a
better-conditioned COVID split, and moving the cut only for COVID would make its transfer cells
incomparable to every other cell. Disclose the val composition, and treat COVID's `bias_c` and its
early-stopping choice as known-fragile. If one thing is changed, the defensible option is to fit
COVID's `bias_c` on the train fold instead of val and say so explicitly — not to move the split.

## 5. Waves (national, top 6 weeks)

| week ending | cases | × median week |
|---|---|---|
| 2022-01-15 | 5,139,436 | 13.8× |
| 2022-01-22 | 4,689,090 | 12.6× |
| 2022-01-08 | 4,440,894 | 11.9× |
| 2022-01-29 | 3,531,492 | 9.5× |
| 2022-01-01 | 2,429,782 | 6.5× |
| 2022-02-05 | 1,994,389 | 5.4× |

All six largest weeks are Omicron, January–February 2022. This is what the build-time **Omicron
anchor pin** asserts, and it is the calendar's only strong anchor — the matrices carry no dates of
their own, so this plays the role the 2009 H1N1 week plays for `influenza_us-regions`.

## 6. COVID vs influenza on the identical 49 nodes

```
graph identical        True
covariates identical   True
node ids disjoint      True
T                      covid 164   vs   influenza 360
```

| | covid | influenza |
|---|---|---|
| dynamic range (max/median node total) | 8.9× | 13.6× |
| Spearman (node totals, paired) | \multicolumn{2}{c}{0.695} | |
| Pearson (node totals, paired) | \multicolumn{2}{c}{0.547} | |

Top-5 states by COVID total, against the same states' influenza totals:

| state | covid | influenza |
|---|---|---|
| California | 12,153,078 | 298,131 |
| Texas | 8,437,665 | 382,394 |
| New York | 6,800,040 | 66,014 |
| Illinois | 4,099,399 | 324,968 |
| Pennsylvania | 3,533,937 | 75,326 |

**This is the reason the bundle exists.** Spearman 0.695 means the two diseases rank the same 49
states *similarly but far from identically* — New York is 3rd by COVID burden and near the bottom by
ILI, Texas leads flu but trails California on COVID. Because the graph and the covariates are
bit-identical, a transfer result between these two bundles isolates **disease** transfer from
**graph** transfer, which no existing cross-disease cell can do.

---

## Summary for the run plan

**Good.** No missingness to model, no reporting dropout, a strong unambiguous calendar anchor, an
identical graph to an existing panel, and node rankings that differ enough from influenza for the
comparison to be informative.

**Watch.** The val fold owns the Omicron peak, which compromises both early stopping and the new
`bias_c` for this bundle specifically. COVID has the largest mean/median gap of any panel (3.03×), so
the `encoder_mc` arm matters most here. Its cross-node level gradient is the second flattest in the
set (0.0090), so expect the spatial channel to carry little. `obs_mask` is constant, so COVID sits on
the influenza side of the channel-3 identifier rather than fixing it.

**Do not quote** the dengue node-mean variance until the 0.948 vs 0.0234 discrepancy in §3 is
resolved.

# `schema_spec.md` — Standardised, Disease-Agnostic Input Schema

**Project:** Generalizable Spatio-Temporal Framework for Emerging Infectious Disease Forecasting
**Phase:** 1 · Task 4 (the contract every dataset, baseline wrapper, and metric depends on)
**Status:** v0.3, reconciled against the shipped bundles on **2026-09-07**. The build is finished and
frozen. Every number below was re-derived from `data/processed/*.npz` through `bundles.load()`, not
carried over from an earlier draft. Corrections made in that pass are marked **Corrected 2026-09-07**
and state the old claim before the new one, because a spec that hides its own errors is worth less than
one that never had any. This closes audit finding 25 (`Reports/Phase0_to_Now_Audit.md:223`), which
recorded this file as behind the bundles on COVID, admin level, horizons and cross-border policy.

---

## 0. Design principle

The architecture is disease-agnostic **because the input is disease-agnostic**. Every disease and every
baseline is coerced into **one** representation: a small set of arrays plus metadata. Nothing below is
specific to a disease; disease-specific facts live only in `meta` and in the *optional* extended feature
channels, never in the tensor shapes or the core channels.

**Corrected 2026-09-07.** That sentence used to name the diseases as "dengue, influenza, Ebola". Three
diseases was right when it was written and is wrong now. The released build is **six bundles covering
four diseases**:

| bundle | N | T | F | role |
|---|---|---|---|---|
| `dengue` | 7,165 | 1,409 | 4 | development |
| `influenza_japan` | 47 | 348 | 4 | development |
| `influenza_us-regions` | 10 | 785 | 4 | development |
| `influenza_us-states` | 49 | 360 | 4 | development |
| `covid_us-states` | 49 | 164 | 4 | development |
| `ebola` (three builds: the retired L7, `ebola_L12`, `ebola_L20`) | 61 | 52 | 5 | few-shot holdout |

`bundles.BUNDLE_NAMES` lists the six names. COVID-19 US-states is the **third development disease**.
Client decision **B5** originally kept COVID-19 out of the harmonised schema entirely. B5 was
**amended 2026-09-06** under the client's own Week 3 instruction (`Review Doc.md:97`), and COVID-19 is
now a development panel labelled as such in every table
(`progress/decisions/client_decisions.md`, the amendment under B5).

Reproduce the table with:
`conda run -n ebola python -c "import bundles; [print(n, bundles.load(n).X.shape, bundles.load(n).meta['role']) for n in bundles.BUNDLE_NAMES]"`

A single disease dataset is a `DiseaseTensors` bundle:

| Symbol | Shape | dtype | Meaning |
|---|---|---|---|
| `X` | `[N, T, F]` | float32 | N nodes × T steps × F features. **Channel 0 is normalised incidence** (the target signal). |
| `A_geo` | `[N, N]` | float32 | Static geographic adjacency (contiguity; optionally distance-weighted). |
| `A_mob` | `[N, N]` or `None` | float32 | Mobility / OD-flow adjacency where available (directed). |
| `C` | `[N, 3]` | float32 | Static node covariates: `centroid_lat`, `centroid_lon`, `area_km2` (§8). Populated in every bundle. Never reaches the shared encoder. |
| `M` | `[N, T]` | uint8 | Missingness mask over the **target**: 1 = observed, 0 = imputed/missing. |
| `y` | `[N, T]` | float32 | Target series in **model space** (normalised). Real-count target recoverable via `meta.scaler`. |
| `meta` | dict | — | Everything needed to interpret, invert, and reproduce the bundle (see §6). |

All `N` and `T` are **per disease**. Diseases do **not** share a node set or a time grid, and they are
not required to — the framework operates on one graph/sequence at a time and transfers *representations*,
not indices.

---

## 1. Node set `N` (spatial units)

One row of the graph = one spatial unit. The admin level is fixed within a **country**, not within a
disease: influenza, COVID and Ebola each sit at one level throughout, and dengue deliberately does not.
See the admin-level bullet below.

| Disease | Unit / level | Released N | Source of units |
|---|---|---|---|
| **Dengue** | the **finest level each country supports**: Admin2 for five countries, Admin1 for seven, 12 countries in total | **7,165** | OpenDengue `adm_1_name` / `adm_2_name`, per-country GADM 4.1 shapefiles |
| **Influenza** | US-Regions (10) / US-States (49) / Japan-Prefectures (47) | 10 / 49 / 47 | ColaGNN-shipped matrices + adjacency |
| **COVID-19** | US-States, the same 49 nodes and the bit-identical graph as influenza US-states | **49** | NYT `us-states.csv`, ColaGNN `state-adj.txt` |
| **Ebola** | Admin2 (district), **Guinea + Liberia + Sierra Leone** | **61** | HDX `Localite`, GADM 4.1 level-2 |

**Corrected 2026-09-07.** The dengue row used to say "Admin1 (state/province)" with "~150–250" nodes.
Both are wrong on the released build, by a factor of about thirty on N. See the admin-level bullet below.
The Ebola row said "~60–65"; it is exactly 61. The COVID-19 row did not exist.

### Node-set decisions (academic)
- **Dengue spans the dengue-endemic countries, not one country.** The development set is the
  WHO/CDC dengue-endemic pool (Americas: Brazil, Mexico, Colombia, Peru, Argentina, Bolivia, Paraguay,
  Venezuela, Ecuador, and others; South/Southeast Asia & Western Pacific: Thailand, Vietnam, Malaysia,
  Philippines, Indonesia, Sri Lanka, Cambodia, Laos, India, Bangladesh, Singapore), intersected with what the
  extract actually covers. This is what makes dengue genuinely *data-rich and spatially diverse* — the premise
  of the transferable-representation goal (G2) — and it enables an intermediate **leave-one-country-out**
  transfer check *within* dengue, a useful robustness result short of full cross-disease transfer.
  - **The country list is a candidate pool, resolved by data, not asserted.** `load_dengue` keeps an endemic
    country only if ≥ `min_nodes_per_country` (default 3) of its units **at a given level** have ≥
    `min_weeks` (default 52) of observed weekly data; per-country coverage and the kept/dropped/absent lists
    are written to `meta['coverage']`. `dengue_country_coverage()` prints the table.
    **Corrected 2026-09-07:** the old text said "its Admin1 units" and told the reader to run the coverage
    audit "in the Week-2 audit ... before locking". `resolve_country_levels` tests **both** Admin1 and
    Admin2 and picks the finest level that clears the gate (client decision **A1**), and the lock happened:
    the thresholds are confirmed as client decision **B3** and the result is the 12-country list below.
  - **The dengue graph is block-diagonal, and that is a deliberate simplification rather than a
    geographic fact.** Contiguity is built *within* each country from its own shapefile and assembled
    block-diagonally, so there are **no cross-border edges**. Verified on the released bundle: **0 of
    40,936 nonzero adjacency entries cross a country border**, and `meta['graph_is_block_diagonal']` is
    `True`. A GNN handles the disconnected components fine, because cross-country signal flows through the
    *shared weights*, not the graph. Node ids are `country|adm` to prevent collisions.
    **Corrected 2026-09-07:** the reason given in brackets used to be "nodes in different countries are not
    geographic neighbours". That is untrue. Brazil borders Colombia, Peru, Bolivia and Argentina, and the
    construction severs real transmission corridors, the Leticia/Tabatinga crossing and the Triple Frontier
    among them (client decision **D6**). The defensible reason is the one above: block-diagonal is what
    makes the mixed Admin1/Admin2 levels safe to combine. It costs real signal, and we say so rather than
    dress it up as geography.
  - **Corrected 2026-09-07: kept countries are NOT all at one admin level.** This bullet used to read
    "All kept countries must be at one admin level and one cadence (Admin1, weekly) so the tensor stays
    rectangular and comparable". It is false on the released build, and it was overridden deliberately:
    client decision **A1** takes the *finest* level each country actually supports, because forcing every
    country to Admin1 would have discarded Brazil's 5,517 municipalities, which are most of the dengue
    data. Only the **cadence** is uniform. `meta['country_levels']` records the level per country:

    | level | countries |
    |---|---|
    | Admin2 | argentina, brazil, colombia, peru, taiwan |
    | Admin1 | bolivia, dominican republic, ecuador, japan, mexico, nicaragua, panama |

    Rectangularity was never at risk, because that needs one *time* grid, not one admin level. What mixed
    levels really cost is that a municipality and a province are not comparable neighbours, and the
    block-diagonal graph in the bullet above is what contains that cost. `meta['level_report']` carries the
    per-country choice and the node counts it was made on. 57 candidate countries are dropped, every one of
    them because the extract carries only Admin0 rows for it: `meta['countries_excluded']` gives
    `admin0_only` as the reason for all 57, and `meta['countries_dropped_low_coverage']` is the identical
    set. One further node is dropped as unmappable, `argentina|formosa|grl. jose de san martin`, a source
    error rather than a coverage failure, and 7,697 rows are summed within their week.
    Released country list, 12 in total: argentina, bolivia, brazil, colombia, dominican republic, ecuador,
    japan, mexico, nicaragua, panama, peru, taiwan.
- **Ebola keeps its cross-border edges, which is the opposite of dengue, and on purpose.** Guinea,
  Liberia and Sierra Leone are physically contiguous, and 2014 was one epidemic that crossed those
  borders, with the Gueckedou / Lofa / Kailahun tri-border area as its defining transmission pathway. A
  country-separated graph would cut exactly the edges that carry the signal. So Ebola keeps them:
  `A_geo_kind = "queen+knn_cross_border"`, `meta['graph_is_block_diagonal']` is `False`, and I measured
  **42 of 292 nonzero adjacency entries crossing a national border**, which is **21 undirected
  cross-border edges out of 146**, including all three tri-border connections.
  `meta['adjacency_report']['cross_border_edges']` names every one of the 21, and the build
  re-checks them every time (client decision **C6**). **The two policies differ because the two situations
  differ**, not by oversight: dengue's twelve countries are separate epidemics observed at mismatched
  admin levels, Ebola's three countries are one epidemic at one admin level. This spec never used to
  state the Ebola half at all, which is one of the four gaps audit finding 25 named. Any document
  quoting a graph policy has to say which disease it means.
- **Ebola region selection is the client's choice, not baked in.** The finalized HDX file also carries
  Nigeria, Senegal and Mali, whose outbreaks were tiny and essentially single-node (Nigeria ≈20 cases,
  Senegal 1, Mali 8) — as district nodes they are degenerate near-zero series that distort per-node
  normalisation and break contiguity. The loader **keeps every country by default** and exposes explicit
  controls so the client decides: `countries=EBOLA_CORE_COUNTRIES` restricts to Guinea/Liberia/Sierra Leone;
  `exclude_regions=[...]` drops named districts; `drop_below_total=<n>` removes near-zero nodes by a principled
  threshold. **Recommendation for the main experiment:** restrict to the West-Africa core (the minor outbreaks
  add noise without strengthening the few-shot signal), with the all-countries version kept as a documented
  robustness check — but this is a switch you set, not one the schema makes for you.
  **Resolved 2026-09-07:** the recommendation was taken. Every released build has
  `meta['countries'] == ['guinea', 'liberia', 'sierra leone']` and 61 districts. Nigeria, Senegal and Mali
  are not in the bundle. No all-countries build exists in `data/processed/`, so the robustness check is an
  available option rather than a delivered artefact, and no document should cite it as one.
- **Node identity is canonicalised** (`country|adm_name`, lowercased, trimmed — the HDX `Localite` values
  carry trailing spaces) and stored in `meta.node_ids` so adjacency, covariates and series always align.

---

## 2. Time index `T`

- Every node in a disease shares **one** time grid (rectangular `X`). Nodes that start/end at different
  dates are placed on the **union** date range; absent steps are imputed for model input and marked `M=0`.
- Calendar timestamps for each step are stored in `meta.dates` (length `T`).

### Temporal-resolution decision — RESOLVED: everything is **weekly**
The client verified that the OpenDengue best-spatial extract **transitions to weekly from 2014 onward**.
(**Corrected 2026-09-07:** weekly-coded rows exist well before 2014. The released dengue bundle starts at
**1998-01-03**. 2014 is when weekly coding becomes near-universal, not when it starts. Nothing in the
loader depends on the date, which is the point of the `T_res` filter described just below.)
We therefore use the **weekly-coded portion** of the dengue extract, which makes all three diseases weekly on
an overlapping calendar:

| Disease | `T_res` used | Handling |
|---|---|---|
| Influenza | **weekly** (epi-week) | as-is (ColaGNN) |
| Ebola | **weekly** | irregular sit-reps → last-cumulative-per-week → differenced (§5) |
| Dengue | **weekly** (released span 1998-01-03 to 2024-12-28) | keep only `T_res == "Week"` rows; legacy monthly dropped |
| COVID-19 | **weekly** | NYT daily cumulative → last value per MMWR week → differenced from the running maximum, the same rule as Ebola |

**How it's selected (robustly).** The loader filters on the **`T_res` column itself** (`t_res_filter="Week"`),
not on a hardcoded year — so it captures exactly the weekly rows regardless of when each country switched, and
generalises if more countries are added. `T_res == "Month"` remains available as an option for a monthly-only
robustness study.

**Why this is the right call for a transfer paper.**
- One cadence gives a single lookback `w`, one horizon set, and `steps_per_year = 52` for **every**
  disease, so the shared temporal encoder sees comparable dynamics rather than having to be cadence-robust
  across a 4× resolution gap.
  **Corrected 2026-09-07:** this line used to say the horizon set is `h ∈ {1..4}` weeks. It is not, and no
  run ever used that. The frozen protocol is **`HORIZONS = (3, 5, 10, 15)`** weeks, direct multi-horizon,
  with lookback **`W = 20`** weeks (`bundles.py:45-46`). Every scored record under `results/` sits on that
  grid. The 20-week lookback in the old text was correct and is unchanged; only the horizons were a
  Phase-1 placeholder. Check with
  `conda run -n ebola python -c "import bundles; print(bundles.W, bundles.HORIZONS)"`.
- Seasonality (sin/cos day-of-year, §3) is now consistent across diseases.
- **Cost:** the monthly-coded rows are discarded from the harmonised set. Cadence consistency matters far
  more for the transfer claim than extra history. The dropped monthly series can feed an optional
  **monthly-only robustness appendix**, never the main cross-disease experiment.
  **Corrected 2026-09-07:** this bullet used to price the cost as "pre-2014 monthly dengue (~two decades)"
  and describe what survives as "2014–2023 weekly, ~10 years × ~27 Brazil states". Both halves are wrong on
  the released bundle. The surviving weekly span is **1998-01-03 to 2024-12-28, T = 1,409 weeks**, so about
  27 years and not 10, and Brazil enters as **5,517 Admin2 municipalities**, not 27 states. The filter on
  the `T_res` column, rather than a year cut, is what produced that, which is exactly why it was written on
  the column. Check with
  `conda run -n ebola python -c "import bundles; m=bundles.load('dengue').meta; print(m['dates'][0], m['dates'][-1], len(m['dates']))"`.
- **Leakage check:** the extract must not carry *both* a monthly and a weekly row for the same node-period
  (double counting). Filtering to a single `T_res` before the pivot prevents this; the audit note should
  confirm no residual overlap.

---

## 3. Feature channels `F` (fixed order, disease-agnostic by construction)

Channels are split into a **core** block that **every** disease is guaranteed to have (and that is the only
block used for cross-disease transfer) and an **extended** block that is disease-specific, optional, and
**excluded from the shared encoder** to avoid zero-padding degrading transfer.

**CORE (always present, transfer-safe) — `F_core = 4`:**
| idx | name | definition |
|---|---|---|
| 0 | `incidence_norm` | normalised new-case incidence — **the target signal every disease has** |
| 1 | `sin_doy` | `sin(2π · day_of_year / 365.25)` from `meta.dates` |
| 2 | `cos_doy` | `cos(2π · day_of_year / 365.25)` |
| 3 | `obs_mask` | 1 = observed, 0 = imputed. Lets the shared encoder distinguish real observations from filled gaps (**D15**). **Not harmless. See the leak note directly below.** |

> **Corrected 2026-09-07, and this correction is a live problem rather than a tidy-up.** Channel 3 used
> to be described as "Constant-1 for fully-observed diseases (harmless); informative for Ebola". Both
> halves mislead. I measured the mean of channel 3 on every released bundle:
>
> | bundle | mean of channel 3 |
> |---|---|
> | dengue | **0.2175** |
> | influenza_japan | 1.0000 |
> | influenza_us-regions | 1.0000 |
> | influenza_us-states | 1.0000 |
> | covid_us-states | 1.0000 |
> | ebola | **0.4095** |
>
> It is constant-1 on the three influenza panels **and** on COVID-19, and it is very far from constant on
> dengue: only 21.75 per cent of dengue cells are observed. That makes channel 3 a **near-perfect disease
> identifier sitting inside the block we call disease-agnostic**. A model can separate dengue from the
> influenza and COVID panels on this channel alone, without ever looking at incidence.
> This is recorded as an open bug at `progress/summaries/Doubt.md` §3.3 ("`obs_mask` is a disease
> identifier inside the 'disease-agnostic' core block"), and the COVID half is recorded in the same file
> under the 2026-08-03 entry "COVID-19 bundle: built, gated, registered", which notes that COVID joins the
> influenza side of the leak, adds nothing new to it and fixes nothing. **It has to be named in the paper's
> limitations.** None of this is a reason to delete the channel: without it the encoder cannot tell a real
> zero from a filled gap, which is a worse failure. It is a reason to stop calling the core block clean.
> Reproduce with
> `conda run -n ebola python -c "import bundles; [print(n, round(float(bundles.load(n).X[:,:,3].mean()),4)) for n in bundles.BUNDLE_NAMES]"`.

**EXTENDED (optional, single-disease only; names in `meta.feature_names`, transfer-flag in `meta.core_feature_idx`):**
| name | availability | notes |
|---|---|---|
| `deaths_norm` | Ebola (yes), dengue (partial), flu (no) | normalised deaths/severe; `0` + masked where unreported |
| `logpop_norm` | where a population table is joined | static, broadcast over `T` |
| `temp` / `humidity` / `precip` | dengue/flu if climate joined | **not** in the finalized raw files → absent by default; add via documented join |
| `lag_k` | model-side | never leaks (lag ≥ 1); constructed inside the model, not baked into `X` |

**Corrected 2026-09-07.** As released, **`deaths_norm` is the only extended channel that exists at all,
and it exists on Ebola only.** The table above said "dengue (partial)". Dengue ships `F = 4` and
`feature_names == ['incidence_norm', 'sin_doy', 'cos_doy', 'obs_mask']`, with no deaths channel.
Ebola ships `F = 5`, with `deaths_norm` at index 4. `logpop_norm`, `temp`, `humidity` and `precip`
are **not built** in any bundle; they stay in this table as the documented shape a future join would
take, not as something a reader can load today. `core_feature_idx == [0,1,2,3]` in every bundle, so
`transfer_view()` is identical across all four diseases, which is the disease-agnosticism guarantee.

> **Why the split matters academically.** Concatenating channels that are all-zero for some diseases
> (e.g. climate for Ebola) injects a spurious "this-disease-has-no-climate" signal the encoder can latch
> onto, quietly making the model disease-*aware* — the opposite of the claim. Keeping the shared encoder on
> the 4 core channels and confining extras to single-disease runs keeps the disease-agnostic claim honest.
> `DiseaseTensors.transfer_view()` returns exactly this core block, so the shared encoder cannot accidentally
> see an extended channel.

**Incidence transform (`incidence_norm`).** Counts are heavily right-skewed and vary orders of magnitude
across nodes/diseases, so: `log1p(count)` → standardise. Incidence is normalised **raw counts, not a
per-capita rate** (client decision **D10**) — no population is joined into channel 0. Standardisation stats are
**fit on the training split only** (§4). A node whose training window is constant (zero variance) takes its
**country's pooled training statistics** rather than `std := 1`, so it is never left un-normalised while its
neighbours are standardised (the pool is train-only, so no leakage).
Store `(transform, mean, std)` per node in `meta.scaler` so predictions invert back to real counts for
RMSE/MAE. **Imputed steps** (`obs_mask=0`) are set to **0 in normalised space** (the per-node mean), a neutral
fill — not the transform of a raw 0 — and the `obs_mask` channel flags them.

---

## 4. Normalisation & leakage safety (non-negotiable)

- **All weekly binning uses MMWR epi-weeks (Sun–Sat)** — client decision **D6**. Implemented as pandas
  `W-SAT` (week-ending-Saturday); dengue records are binned into their MMWR-week *period* rather than assumed
  to fall on a fixed weekday, so alignment is robust to how each source labels week starts.
- **Development diseases:** scaler fit on observed **train** cells only (mask==1 and `t < train_end`), per node
  (zero-variance guarded), applied to val/test. Never fit on val/test. Parameters stored for inverse-transform.
- **Few-shot holdout (Ebola):** the scaler is fit on the **support set only**, a **calendar prefix**: every
  observed cell dated on or before the arm's cutoff. It is pooled **per-disease** (one mean and one sd,
  `scaler_scope="per_disease_support"`), because the handful of early cells is far too few for a stable
  per-node scaler and fitting on anything past support would leak the outbreak scale into the "few-shot"
  result. The prefix is **calendar-causal** (no query cell precedes a support cell) and gated; see §5.
  Support and query masks are in `meta.split`.
  **Corrected 2026-09-07:** the released cutoff quoted here used to be **2014-05-24**. That build is
  retired for scoring. The two arms that were actually scored are **`ebola_L12`, cutoff 2014-06-28
  (primary)** and **`ebola_L20`, cutoff 2014-08-23 (secondary)**. The pooling and the causality property
  are unchanged; only the date moved. Full table in §5.
- **Chronological only.** No shuffling; dev split is a time cut (default **50/20/30**, EpiGNN; rolling-origin
  available — see `problem_and_protocol.md`).
- **Graph structure is static geography** (fit-free) or, if *learned* later, learned on train only.
- **No future-derived features**; horizon-`h` targets never appear in inputs (predict `t+h`, `h ≥ 1`, from a
  window ending at `t`; channel 0 at `t` is observed history, not the label).
- Leakage can't happen by accident: `fit_scalers*` and `apply_scaler` are **separate** steps — raw counts are
  assembled, the fit set (train slice or support) is chosen, then and only then are stats computed.

---

## 5. Ebola: cumulative → incidence, and the few-shot split (correctness, not preference)

HDX sub-national values are **cumulative** situation-report counts, not new cases. Feeding them in raw would
train the model on a monotone step function. Conversion, per `(country, district, indicator)` series:

1. Canonicalise district id (accent-stripped, §7); parse `date` as `dd/mm/yyyy`.
2. **Resample to MMWR weeks** (Sun–Sat): take the **last cumulative value** within each week; forward-fill weeks
   with no sit-report.
3. **Difference the running maximum** (`cummax().diff()`), not the raw series.
4. **Weeks whose report falls below the running maximum are masked (`M=0`).** They are corrupt reports, not
   observations of zero.
5. **First observed week is masked (`M=0`).** At a district's first report there is no preceding cumulative,
   so the increment is not identifiable.
6. Weeks that were forward-filled (no underlying report) are marked **`M=0`** so they are imputed for input
   continuity but **excluded from loss and evaluation**.
7. Deaths handled identically → `deaths_norm` extended channel.

**Steps 3–5 are corrections. The original spec was wrong on both, and the code implemented it faithfully.**

- **The original step 3–4 said "difference, then clip negatives to 0", on the reasoning that a decrease is a
  double-counting correction.** It is not. A cumulative count cannot fall; where the report falls, the report
  is a single-week data-entry dropout. Clipping zeroed the fall and then released the *recovery* — a climb
  back to cases already counted — as new incidence. This **fabricated 8,786 cases, 35.8 per cent of the Ebola
  target** (33,338 released against 24,552 real), put the six largest cells in the dataset there as artefacts,
  and displaced the national epidemic peak from late 2014 to February 2015. Differencing the monotone envelope
  is mass-preserving by construction: increments sum to `max(C) - C_first` exactly.
- **The original step 5 said "first observed week = its cumulative value".** Because the compilation opens
  months into an outbreak already under way, this assigned every district's entire back-log to its first week
  and fed those numbers straight into the pooled support scaler for the whole held-out disease.

**Invariant, enforced by a gate rather than by this prose:** a weekly series derived from a cumulative one must
sum to no more than `max(C) - C_first`, where that reference is computed **from the source column by
`cumulative_reference_mass()`** — independently of the transform under test. The original defect survived 83
gates precisely because no check compared the transform against anything the transform had not itself produced.

**Known limitation, not corrected:** differencing recovers *how many* cases accrued between two reports, not
*when* within that interval. Where a district falls silent and then files, the whole multi-week increment lands
on the week the report arrived. 7 per cent of inter-report intervals exceed one week (longest 24). No cases are
invented and the curve is correctly shaped, but peaks are inflated and adjacent weeks flattened. **Client
decision (A4): disclose and proceed** — the shape is right and both remedies (redistribute, or score only
contiguous runs) cost more than the distortion; quote any weekly Ebola magnitude with this stated.

**Few-shot protocol.** Ebola is `role="few_shot_holdout"`, *not* a 50/20/30 development disease. The
**support set** is a **calendar prefix**: every observed cell dated on or before the arm's cutoff is
support; every later observed cell is query. The support/query masks and the support-fit scaler are in
`meta.split` / `meta.scaler`, and `meta['few_shot_support_scheme']` carries the cutoff as a string. The
support-only scaler is enforced in code, not left to the training loop.

**Corrected 2026-09-07. This is the largest stale block in the file.** The paragraph above used to name a
single released cutoff, **2014-05-24**, and the counts that go with it. That build still exists on disk as
`data/processed/ebola.npz`, and it is deliberately **kept for provenance**, but it is **retired for
scoring**: nothing may be scored against it. Two arms were frozen and hashed *before* either was scored,
and those two are the ones the paper reports:

| arm | file | `support_weeks` | cutoff (`few_shot_support_scheme`) | support cells | query cells | districts with support | pure zero-shot districts |
|---|---|---|---|---|---|---|---|
| retired L7 | `data/processed/ebola.npz` | 7 | `calendar_prefix<= 2014-05-24` | 27 | 1,272 | 9 | 52 |
| **primary L12** | `data/processed/ebola_L12.npz` | 11 | `calendar_prefix<= 2014-06-28` | **59** | **1,240** | **18** | **43** |
| **secondary L20** | `data/processed/ebola_L20.npz` | 13 | `calendar_prefix<= 2014-08-23` | **113** | **1,186** | **36** | **25** |

The arms were frozen by `freeze_ebola_arms.py`, which writes `configs/ebola_arms.json` and re-derives every
count from the raw xlsx before it will write. The decision that chose them is **D16** in
`progress/decisions/decisions.md`, dated 2026-08-07, and the pre-registration is
`progress/decisions/Ebola_Prereg.md`. A third candidate, L19 (cutoff 2014-08-16), was considered and
recorded as rejected rather than quietly dropped.

Three counts that look like the same thing and are not, so nobody re-derives them wrongly.
(a) "L" is the **0-based index of the last support column**, not a count of weeks, so L12 is 13 columns
and L20 is 21. (b) The number of columns holding at least one support cell is 12 and 14, one fewer than
the column count in each case, because week 0 of a cumulative series carries no increment and is masked.
(c) `meta['split']['support_weeks']` is 11 and 13, and it is neither of those: it is
`support_mask.sum(1).max()` (`to_schema.py:1536`), the number of support weeks the **busiest single
district** has. **The cutoff date is the operative definition**; the L label exists only to match the
document the client decided from. And the **query set
is identical under both arms**, asserted by the freeze script, so the arms differ only in how much
labelled adaptation data they carry and never in what is evaluated. That is what makes them comparable.

**Never re-score the pre-registration.** It was scored once, on purpose. Re-scoring it destroys the thing
that makes it credible.

**What left the Ebola node set, in full.** `meta['nodes_dropped']` has **10** entries, not just the one
this file used to mention: five multi-district blobs from the early outbreak (one of them spanning two
countries), three national aggregates (`guinea|national`, `liberia|national`, `sierra leone|national`),
`sierra leone|freetown` (a city inside Western Area Urban, one report, no polygon of its own), and
`sierra leone|western area` (the parent that would have triple-counted its Urban and Rural children).
`meta['name_aliases_applied']` has **12** entries: eleven Liberian labels of the form "X county" mapped to
"x", plus `sierra leone|port` mapped to `sierra leone|port loko`, a truncated label whose cumulative series
continues from the next day. That leaves **61** districts across guinea, liberia and sierra leone, over
**52** weeks, **2014-04-05 to 2015-03-28**. `meta['ebola_first_week_masked']` is `True`, which is step 5
above.

**Calendar-causal by construction.** Because the split is a date threshold shared by every district, no
query cell is ever earlier in time than any support cell. This is verified by a build gate (every support
cell dated `<=` every query cell), with a negative control that plants an acausal support cell and requires
the gate to fail. **All three builds in the table above hold this property**, which is the entire point of
the design, so the reasoning here is unchanged and only the counts moved. On the primary arm `ebola_L12`:
59 support cells over the 18 districts reporting by the cutoff, 1,240 query cells, and **43 of 61 districts
are pure zero-shot**, which is the genuine emerging-outbreak regime.

**This replaced a per-district scheme that was not causal.** The earlier support set was the first *n* observed
weeks of *each district*; because districts enter at different dates, 85 per cent of query cells were earlier in
calendar time than the last support cell, so the pooled scaler normalising an early cell drew on peak-epidemic
magnitudes. The old suite could not detect this — it *defined* `support_mask` as the legitimate fit set — which
is why the calendar-causality gate checks the cells' **dates**, not the mask.

**Window size: RESOLVED, was deferred.** This used to read "the size of the early window may later widen
from the current eight-week prefix", left open as D22. It widened, and the question is closed. The cutoff
is a parameter, so it was a rebuild and not a code change, exactly as predicted. It was decided together
with the causality property, also as predicted, and both arms keep it. Two arms were frozen instead of one
so that the trade-off is measured rather than argued: a later cutoff buys labelled adaptation data and
spends held-out districts. L12 keeps 43 of 61 districts pure zero-shot; L20 keeps 25. Frozen by
`freeze_ebola_arms.py` under **D16**.

**One consequence worth stating here rather than only in the results.** On the primary arm, support reaches
column 12 and an h15 target needs column 15, so **the primary arm has zero adaptation data at h15**. h15 is
therefore zero-shot by arithmetic on that arm, and few-shot h15 must come out bit-identical to zero-shot
h15. Any difference there is a bug in the adaptation path, not a result. h10 has 18 adaptation pairs on 9
districts and is not a fitted adapter either. This is arithmetic from the schema, so it belongs in the
schema contract.

The same routine differences dengue only if a source is cumulative; OpenDengue `dengue_total` is already
per-period incidence, so dengue is used as-is (`NA`/suppressed → `M=0`).

---

## 6. `meta` contents (reproducibility contract)

`meta` must carry everything needed to interpret, invert and reproduce the bundle:

- `disease`. Released values: `"dengue"`, `"influenza:japan"`, `"influenza:us-regions"`,
  `"influenza:us-states"`, `"covid:us-states"`, `"ebola"`. Note that all three Ebola builds carry
  `disease == "ebola"`; the arm is identified by the **file name**, not by this key.
- `node_ids` — length-`N` canonical ids, **row order == every array's row order**
- `adm_level`. Released values: `"Admin1"` on the three influenza panels and COVID, `"per_country"` on
  dengue and on all three Ebola builds
- `dates`, length-`T` timestamps (MMWR week-ending Sat); `t_res` (`"weekly"` everywhere);
  `steps_per_year` (52 everywhere)
- `feature_names`, length-`F`. `core_feature_idx`, the transfer channels, `[0,1,2,3]` in **every** bundle
- `scaler` — per-node `(transform, mean, std)` for inverse-transform; `scaler_scope`
  (`"per_node_train"` on the five development panels | `"per_disease_support"` on Ebola)
- `split`. Three shapes, and code must not assume one: influenza and COVID ship `{train_end, val_end}`
  integers and no mask arrays; dengue ships `{scheme, train_mask, val_mask, test_mask, country_bounds,
  nodes_without_train}`; Ebola ships `{scheme, support_weeks, support_mask, query_mask}`.
  `split_scheme` released values: `"fixed_50_20_30"` (influenza, COVID) |
  `"per_country_chronological_50_20_30"` (dengue) | `"few_shot_support_query"` (Ebola).
  `bundles._normalise_masks` exists to absorb the three shapes so no model or metric branches on disease.
- `rolling_origins`, present on the five development panels, `null` on Ebola by design
- `role` (`"development"` | `"few_shot_holdout"`); `covariates_status`, `covariates`,
  `covariates_source`, `covariates_transfer_safe` (§8)
- `A_geo_kind`. Released values: `"shipped(diag_zeroed)"` (three influenza panels and COVID) |
  `"queen+knn_block_diagonal"` (dengue) | `"queen+knn_cross_border"` (Ebola). `A_mob_available` is
  `False` in every bundle.
- `requires_encoder_self_loops` and `isolated_nodes`, on the four **shipped-graph** bundles only. The
  shipped adjacency has its diagonal zeroed (§7, client decision **C5**), so a consumer that does not add
  self-loops loses each node's own signal. `isolated_nodes` lists the nodes with no neighbour at all:
  `['japan_10','japan_19']`, `[]` for us-regions, `['us-states_1','us-states_9']`, and
  `['covid_us-states_1','covid_us-states_9']`. This matters: it is the same class of defect as the HeatGNN
  isolated-node NaN, and it is in the metadata precisely so a wrapper can check rather than discover it.
- `graph_is_block_diagonal`, `True` on dengue, `False` on Ebola, absent on the shipped-graph bundles.
  `adjacency_report`, per-country edge and isolated counts on dengue; on Ebola it also names all 21
  cross-border edges.
- `countries`, `countries_dropped_low_coverage`, `countries_excluded`, `countries_requested_absent`,
  `coverage`, `country_levels`, `node_levels`, `level_report`, `node_country`,
  `nodes_dropped_unmappable`, `within_week_summed_rows`, `aggregated_to_parent`,
  `case_def_mixed_node_weeks` (dengue)
- `countries`, `node_country`, `nodes_dropped`, `name_aliases_applied`, `ebola_first_week_masked`,
  `few_shot_support_scheme`, `cumulative_reference`, `nodes_without_query` (Ebola)
- `raw_sha256`, `npi_confounded`, `excluded_nodes` (COVID only). `raw_sha256` is compared against
  `build_datasets.RAW_SHA256` and a mismatch **halts the build**, which matters because the loader fetches
  the NYT `master` branch, a reference that can move. `excluded_nodes` is
  `['Florida', 'District of Columbia']`. `npi_confounded` is `True` and is not decoration: the 2020 to 2023
  window is dominated by interventions.
- `shipped_adjacency_sha256`, `shipped_matrix_sha256` (influenza and COVID), the provenance of a reused graph
- `shapefile_source`, `"GADM 4.1 (gadm.org)"` on dengue and Ebola. `source`, the citation / DOI / URL,
  on every bundle

**Corrected 2026-09-07.** Four things in this list were wrong or missing.
1. `adm_level` was documented as `"Admin0|1|2"`. Dengue and Ebola ship `"per_country"`, because dengue
   genuinely mixes Admin1 and Admin2 (§1) and the real level is in `country_levels` / `node_levels`.
2. `A_geo_kind` was documented as `"queen"|"knn"|"shipped"|"queen+knn_block_diagonal"|"deferred..."`. None
   of `"queen"`, `"knn"`, `"shipped"` or `"deferred..."` is a released value. The three real ones are
   listed above.
3. `requires_encoder_self_loops`, `isolated_nodes`, `raw_sha256`, `npi_confounded` and `excluded_nodes`
   ship and were not in the contract at all. I confirmed each one by loading the bundles rather than by
   reading the loader.
4. The old last line promised `crs` and `shapefile_ref`. **Neither key ships in any bundle.** The
   provenance that does ship is `shapefile_source` on dengue and Ebola, and the two shipped-graph hashes on
   influenza and COVID. Anything downstream that reads `meta['crs']` will raise.

Dump the released keys yourself with:
`conda run -n ebola python -c "import bundles; [print(n, sorted(bundles.load(n).meta)) for n in bundles.BUNDLE_NAMES]"`

---

## 7. Adjacency construction `A_geo`

- **Influenza and COVID-19:** **reuse ColaGNN's shipped adjacency**, with one documented change. Do not
  rebuild it, because matching the baselines' graph is part of comparability. COVID-19 US-states reuses
  `state-adj.txt` too, so its `A_geo` and its `C` are **bit-identical** to `influenza_us-states`. That is
  not laziness, it is the whole design of that bundle: it varies the disease and nothing else, where every
  other cross-disease comparison in this study confounds disease with graph, geography and node count. It
  is what the graph-controlled `encoder_pair__` comparison rests on.
  **Corrected 2026-09-07:** the released flag is **`A_geo_kind="shipped(diag_zeroed)"`**, not `"shipped"`.
  The shipped matrices carry a self-connection on every node and ours do not. Left as shipped, influenza
  nodes would have carried twice the self-weight of dengue nodes inside the shared model, a structural
  signature from which the model could infer the disease (client decision **C5**). The operation is
  information-preserving and the real edges are reused unchanged, so comparability with the published
  baselines is intact, **but the encoder must now add self-loops itself**. `meta['requires_encoder_self_loops']`
  is `True` on all four of these bundles and exists so a consumer can check rather than discover it.
- **Dengue / Ebola:** GADM 4.1 shapefiles → **queen contiguity** (share a boundary) via `geopandas`. Row
  order aligned to `meta.node_ids`. **The two run opposite cross-border policies, and §1 explains why.**
  Dengue is **block-diagonal** across countries, intra-country contiguity only (`build_dengue_adjacency`,
  `A_geo_kind="queen+knn_block_diagonal"`, 0 of 40,936 nonzero entries cross a border). Ebola **keeps its
  cross-border edges** (`build_ebola_adjacency`, `A_geo_kind="queen+knn_cross_border"`, 42 of 292 nonzero
  entries cross a border, which is 21 undirected edges of 146). All adjacency in every bundle is
  **binary**: the only values in `A_geo` anywhere are 0.0 and 1.0.
- **Joins are by name, verified against codes where codes exist (D17).** The original D17 rationale — "the
  client will not supply GADM/GAUL code columns" — was **wrong**: OpenDengue ships `FAO_GAUL_code`,
  `RNE_iso_code` and `IBGE_code` at ~100 per cent. But a code join is not the fix it appears to be.
  `FAO_GAUL_code` is **not injective on our nodes** (7,443 → 7,027: 416 nodes would silently *merge*, because
  GAUL is a coarser ancestor geography — Taiwan alone collapses 287→2), so joining on it is unsafe. `RNE_iso`
  maps to GADM's `ISO_1` at only ~74 per cent, and GADM carries **no** code at all for the 1,476 Colombia/Peru/
  Argentina/Taiwan Admin-2 nodes. So the name join is unavoidable for four of five Admin-2 countries. It stays:
  `_canon` strips accents/diacritics/whitespace (*Guéckédou* == *gueckedou*), residual mismatches are fixed via
  the alias maps, and any unmatched node **raises with the full list** rather than being silently dropped.
  **Corrected 2026-09-07:** the count used to be given as "the `name_aliases` map (234 entries)". I counted
  the shipped module: `dengue_aliases.DENGUE_NAME_ALIASES` holds **231** entries across eight countries
  (brazil 83, colombia 101 plus 5 parent renames, taiwan 18, argentina 10, dominican republic 6, peru 4,
  nicaragua 3, ecuador 1), and there are **two** maps rather than one. The second,
  `DENGUE_GADM_FIX`, holds **2** entries (`japan: nagasaki -> naoasaki`, `peru: huanuco|huanuco ->
  huanuco|huenuco`) and runs at the join only, absorbing GADM's own typos without ever renaming one of our
  nodes. One node is genuinely unmappable and is listed with its reason in `DENGUE_UNMAPPABLE`.
  Where a code *does* exist it is used as an **oracle, not a key**: Brazil's `IBGE_code`
  matches GADM `CC_2` for all 5,517 nodes, and the name join lands on the identical polygon for every one
  (0 disagreements) — a stronger reproducibility claim than a code join, since it validates both the names and
  the codes. Recommended as a build gate.
- **Disconnected units** (islands; or a district with no reporting neighbour): fall back to **k-nearest
  centroids** (default k=4) for those nodes so the graph has no isolated vertices. GNN message passing
  degenerates on isolated nodes, so this is a correctness fix, not a tuning knob. It worked: on the
  released build `adjacency_report` reports `n_isolated = 0` for Ebola and for all twelve dengue country
  blocks.
  **Corrected 2026-09-07, twice.** First, the flag written is never the bare `"queen+knn"` this line
  promised; it is `"queen+knn_block_diagonal"` on dengue and `"queen+knn_cross_border"` on Ebola, so the
  cross-border policy is legible from the flag alone. Second, and more important, **the guarantee does not
  extend to the shipped graphs.** Influenza and COVID reuse ColaGNN's adjacency verbatim and are therefore
  allowed to keep isolated nodes: `japan_10`, `japan_19`, `us-states_1`, `us-states_9` and the two COVID
  copies of the latter pair. They are left isolated on purpose, for comparability, and listed in
  `meta['isolated_nodes']` instead of being repaired. Any consumer that assumes "no isolated vertices
  anywhere" is wrong on four of the six bundles.
- Store **raw adjacency** (binary or distance-weighted); let the model add self-loops / normalise so different
  baselines can apply their own convention.
- `A_mob`: Japan prefectures have OD flows (MepoGNN/STOEP) → directed `A_mob`. US flu can add commuting.
  Dengue/Ebola: `None` by default.

---

## 8. Covariates `C`

**RESOLVED. D20's deferral is closed.** `C` is `[N, 3]` and **populated in every one of the six bundles**:
`meta['covariates'] == ['centroid_lat', 'centroid_lon', 'area_km2']`,
`meta['covariates_status'] == "populated from GADM 4.1"`,
`meta['covariates_source'] == "GADM 4.1 (gadm.org)"`, and
`meta['covariates_transfer_safe'] == False` everywhere.

**Corrected 2026-09-07.** This section used to read "Deferred by the client (D20, 'manage later'). `C`
ships as a flagged `[N,3]` placeholder, to be populated in Week 2 from shapefiles". Week 2 happened. The
placeholder path still exists in the code and still writes
`covariates_status="PLACEHOLDER — no gadm_dir given, C is all-zero"` (`to_schema.py:1562`), but no
released bundle takes it.

Two things did not change and still hold. Population is **not** in the finalized raw set; if a population
table is joined later it becomes a fourth column and unlocks the `logpop_norm` extended channel. And
keeping `C` to shapefile-derived fields avoids silently introducing an un-provenanced covariate into a
publication.

One thing to add, because it is the operative rule now. **`C` never reaches the shared encoder.** That is
what `covariates_transfer_safe=False` means, it is client decision **C9**, and it is enforced rather than
asserted: `SharedEncoder.forward` has no `C` argument and `tests/test_encoder_invariants.py` proves that
passing one raises `TypeError`. Anyone proposing to feed geography into the trunk, including indirectly by
building an adjacency out of the centroids in `C[:, :2]`, needs a recorded decision and an amendment to
this section, not a quiet import.

---

## 9. Deliverable & acceptance (Phase-1 DoD)

- `schema_spec.md` (this file); `schema_decisions.md` (the annotated decision register).
  **Corrected 2026-09-07:** `schema_decisions.md` does not exist anywhere in the repo. The decision
  history it promised lives in `progress/decisions/decisions.md` and
  `progress/decisions/client_decisions.md` instead. Do not go looking for it.
- `to_schema.py`, exposing `load_dengue / load_influenza / load_ebola / load_covid → DiseaseTensors`, plus
  `dengue_country_coverage`, `fit_scalers*` / `apply_scaler` / `invert_scaler`, `build_adjacency` /
  `build_dengue_adjacency`, and `DiseaseTensors.transfer_view()`.
- `test_schema.py`. Shape contracts on **all four diseases across all six bundles**, plus the tricky
  invariants: cumulative→incidence (envelope + mask), leakage-safe train scaler, few-shot support +
  disjoint query + pooled scaler, `obs_mask` core channel, `transfer_view` core-only, accent-stripped
  joins, multi-country coverage prune.
- `test_leakage.py`. **101 gates** across the **six** datasets, gating the build, with **6 negative
  controls** that plant a real defect and require the corresponding gate to fail.
  **Corrected 2026-09-07:** this used to say "86 gates across the five datasets". The suite went from 86 to
  **101, all passing**, when COVID-19 was folded into `build_datasets.py` and came under the same fifteen
  per-dataset gates every other panel passes (`progress/planning/data_audit.md` §6.1, and the summary row
  at §5.3). Six negative controls is still correct and did not move. Before that refactor the COVID bundle
  was built outside `build_datasets.py` and passed through neither the leakage suite nor
  `--check-deterministic`; the refactor was verified by rebuilding and comparing bit-for-bit, so no COVID
  number in the study moved. A gate that has never failed establishes
  nothing: three defects (the clip above; a per-node split fallback that put training cells inside neighbours'
  test period; and an acausal per-district support set) passed 83 gates green and were caught only in
  adversarial review. The gate count is not a measure of coverage — the mass-conservation, phase-purity and
  calendar-causality gates were added precisely because the properties they test were previously untested.

## 10. Open items that still change the schema

**Reviewed against disk 2026-09-07. All four items that used to sit here are closed.** They are kept
below with their resolutions rather than deleted, because "how did this get decided" is a question this
file exists to answer. Two genuinely open items are listed after them.

- **`[CONFIRM-1]`** target = **cases** vs cases+deaths, that is, whether `deaths_norm` / `y_deaths` is a
  first-class target. **CLOSED: cases only.** `y` is incidence in every bundle. `deaths_norm` exists on
  Ebola alone, as channel 4, and `core_feature_idx == [0,1,2,3]` excludes it, so it is structurally
  unreachable from the shared encoder. No `y_deaths` array exists in any bundle or anywhere in the code.
  **Honest caveat:** I found the build settling this and I did not find a client sign-off recorded against
  the label `CONFIRM-1`. It is closed by what shipped, not by a minuted decision.
- **`[CONFIRM-3]`** influenza source = benchmark ILINet+Japan vs adding WHO FluNet. **CLOSED: benchmark
  only, FluNet not added.** Recorded at `Reports/Phase2/Progress.md:707` ("[CONFIRM-D2] resolved:
  benchmark-only. WHO FluNet NOT added") and confirmed by the client as decision **B1**: FluNet is
  country-level, is not sub-nationally resolved, is not the benchmark, and adding it would force a graph
  rebuild and forfeit comparability with the published baselines.
- Ebola region set for the *main* experiment. **CLOSED: the West-Africa core, as recommended.**
  `meta['countries'] == ['guinea', 'liberia', 'sierra leone']` on all three builds, 61 districts. Nigeria,
  Senegal and Mali are not in the released bundle. See also **C6** (cross-border edges kept) and **C7**
  (the three Sierra Leonean labels that are not districts).
- Dengue final country list, plus the Admin1-vs-Admin2 and coverage-threshold defaults. **CLOSED: 12
  countries.** Thresholds confirmed by the client as **B3**, at least 52 observed weeks per node and at
  least three such nodes per country, pruned at country level so no node of an included country is
  dropped. The level question was closed the other way from the old default: **A1** takes the finest level
  each country supports, which is why the list in §1 mixes Admin1 and Admin2.

**Still genuinely open, and both belong to this contract.**

- **Channel 3 is a disease identifier** (§3). Measured, undisclosed in the paper as of today, and it
  weakens the disease-agnostic claim inside the core block. The action is a limitations paragraph, not a
  code change. Tracked at `progress/summaries/Doubt.md` §3.3.
- **Ebola is normalised on a different scope from every training panel.** Ebola ships
  `scaler_scope="per_disease_support"`, one pooled scale for the whole disease, while all five development
  panels ship `per_node_train`. The encoder therefore trains on inputs centred at zero and is handed Ebola
  inputs that are not. That is a covariate shift the schema introduces itself. The measured divergences and
  the matched-panel pricing are in `progress/STATUS.md` §5; I did not re-derive them in this pass, and the
  one experiment that would settle it, training per-node and testing pooled, has not been run. It needs a
  Threats paragraph before a reviewer writes it for us.

**Resolved by client review:** D2 (dengue endemic multi-country), D3 (Ebola region control), D5 (weekly
everywhere), D6 (MMWR epi-weeks), D9 (encoder = core only), D10 (incidence = raw counts), D15 (`obs_mask` as a
core channel), D17 (name joins, accent-stripped, verified against codes — **rationale corrected, see above**),
D20 (covariates deferred, flagged. **The deferral is now closed and `C` is populated in every bundle,
see §8**),
D24 (Ebola cumulative→incidence — **method since corrected, see §5**), D27 (ColaGNN start weeks from the repo).

**Raised by the Phase-2 client review, and their resolutions** (see `client_decisions.md`,
`remediation_plan.md`):
- **Ebola few-shot causality** is RESOLVED: calendar-prefix support, now gated. (**Corrected 2026-09-07:**
  the cutoff quoted here used to be 2014-05-24. That build is retired for scoring. The scored arms are
  `ebola_L12` at 2014-06-28 and `ebola_L20` at 2014-08-23. The causality property is what was resolved and
  it holds on all three builds; only the date moved. See §5.)
- **Dengue case definitions** — RESOLVED: measured to a 2.8× (Mexico) / 1.6× (Bolivia) step on 41 nodes, not
  the ×220 of the raw-file group-by; kept as built and disclosed.
- **Ebola Western Area** — RESOLVED: parent stays dropped (Urban + Rural are already separate nodes; keeping
  the parent would triple-count), 61 nodes. (**Extended 2026-09-07:** Western Area is one of **ten**
  entries in `meta['nodes_dropped']`, not the only one. The full list is in §5. The client marked C7
  "Have a single Western Area node, instead of dropping it" rather than approving it, so if that
  instruction is ever actioned the node count changes and every Ebola number is rebuilt. As shipped, the
  parent is dropped.)
- **D22 (few-shot window size)** is **RESOLVED. It was resolved on 2026-08-07 by D16, and this file did
  not catch up until 2026-09-07.** It widened, exactly as the deferral anticipated, and it was decided together with the causality property. Two arms were frozen and
  hashed before either was scored: `ebola_L12` (primary, cutoff 2014-06-28) and `ebola_L20` (secondary,
  cutoff 2014-08-23). Frozen by `freeze_ebola_arms.py` into `configs/ebola_arms.json` under decision
  **D16** in `progress/decisions/decisions.md`, pre-registered in `progress/decisions/Ebola_Prereg.md`.
  Both were scored. See §5 for the counts and for the h15 consequence on the primary arm.
- **Ebola gap-lumping** — RESOLVED: disclose and proceed (A4); the shape is correct and both remedies cost
  more than the distortion.

# `schema_spec.md` — Standardised, Disease-Agnostic Input Schema

**Project:** Generalizable Spatio-Temporal Framework for Emerging Infectious Disease Forecasting
**Phase:** 1 · Task 4 (the contract every dataset, baseline wrapper, and metric depends on)
**Status:** v0.1 — defaults locked, `[CONFIRM]` items flagged for client sign-off before Week 2.

---

## 0. Design principle

The architecture is disease-agnostic **because the input is disease-agnostic**. Every disease —
dengue, influenza, Ebola — and every baseline is coerced into **one** representation: a small set
of arrays plus metadata. Nothing below is specific to a disease; disease-specific facts live only in
`meta` and in the *optional* extended feature channels, never in the tensor shapes or the core channels.

A single disease dataset is a `DiseaseTensors` bundle:

| Symbol | Shape | dtype | Meaning |
|---|---|---|---|
| `X` | `[N, T, F]` | float32 | N nodes × T steps × F features. **Channel 0 is normalised incidence** (the target signal). |
| `A_geo` | `[N, N]` | float32 | Static geographic adjacency (contiguity; optionally distance-weighted). |
| `A_mob` | `[N, N]` or `None` | float32 | Mobility / OD-flow adjacency where available (directed). |
| `C` | `[N, S]` | float32 | Static node covariates (centroid lat/lon, area; population if joined). |
| `M` | `[N, T]` | uint8 | Missingness mask over the **target**: 1 = observed, 0 = imputed/missing. |
| `y` | `[N, T]` | float32 | Target series in **model space** (normalised). Real-count target recoverable via `meta.scaler`. |
| `meta` | dict | — | Everything needed to interpret, invert, and reproduce the bundle (see §6). |

All `N` and `T` are **per disease**. Diseases do **not** share a node set or a time grid, and they are
not required to — the framework operates on one graph/sequence at a time and transfers *representations*,
not indices.

---

## 1. Node set `N` (spatial units)

One row of the graph = one spatial unit at a fixed admin level, consistent within a disease.

| Disease | Unit / level | Typical N | Source of units |
|---|---|---|---|
| **Dengue** | Admin1 (state/province), **dengue-endemic countries** | ~150–250 | OpenDengue `adm_1_name`, per-country GADM shapefiles |
| **Influenza** | US-Regions (10) / US-States (49) / Japan-Prefectures (47) | 10 / 49 / 47 | ColaGNN-shipped matrices + adjacency |
| **Ebola** | Admin2 (district), **Guinea + Liberia + Sierra Leone** | ~60–65 | HDX `Localite`, GADM level-2 |

### Node-set decisions (academic)
- **Dengue spans the dengue-endemic countries, not one country.** The development set is the
  WHO/CDC dengue-endemic pool (Americas: Brazil, Mexico, Colombia, Peru, Argentina, Bolivia, Paraguay,
  Venezuela, Ecuador, and others; South/Southeast Asia & Western Pacific: Thailand, Vietnam, Malaysia,
  Philippines, Indonesia, Sri Lanka, Cambodia, Laos, India, Bangladesh, Singapore), intersected with what the
  extract actually covers. This is what makes dengue genuinely *data-rich and spatially diverse* — the premise
  of the transferable-representation goal (G2) — and it enables an intermediate **leave-one-country-out**
  transfer check *within* dengue, a useful robustness result short of full cross-disease transfer.
  - **The country list is a candidate pool, resolved by data, not asserted.** `load_dengue` keeps an endemic
    country only if ≥ `min_nodes_per_country` (default 3) of its Admin1 units have ≥ `min_weeks` (default 52)
    of observed weekly data; per-country coverage and the kept/dropped/absent lists are written to
    `meta['coverage']`. Run `dengue_country_coverage()` in the Week-2 audit to see the table before locking.
  - **The graph is block-diagonal.** Contiguity is built *within* each country from its own shapefile and
    assembled block-diagonally; there are **no cross-border edges** (nodes in different countries are not
    geographic neighbours). A GNN handles the disconnected components fine — cross-country signal flows through
    the *shared weights*, not the graph. Node ids are `country|adm` to prevent collisions.
  - **All kept countries must be at one admin level and one cadence** (Admin1, weekly) so the tensor stays
    rectangular and comparable; countries offering only a coarser level or non-weekly data are dropped by the
    coverage gate rather than silently mixed in.
- **Ebola region selection is the client's choice, not baked in.** The finalized HDX file also carries
  Nigeria, Senegal and Mali, whose outbreaks were tiny and essentially single-node (Nigeria ≈20 cases,
  Senegal 1, Mali 8) — as district nodes they are degenerate near-zero series that distort per-node
  normalisation and break contiguity. The loader **keeps every country by default** and exposes explicit
  controls so the client decides: `countries=EBOLA_CORE_COUNTRIES` restricts to Guinea/Liberia/Sierra Leone;
  `exclude_regions=[...]` drops named districts; `drop_below_total=<n>` removes near-zero nodes by a principled
  threshold. **Recommendation for the main experiment:** restrict to the West-Africa core (the minor outbreaks
  add noise without strengthening the few-shot signal), with the all-countries version kept as a documented
  robustness check — but this is a switch you set, not one the schema makes for you.
- **Node identity is canonicalised** (`country|adm_name`, lowercased, trimmed — the HDX `Localite` values
  carry trailing spaces) and stored in `meta.node_ids` so adjacency, covariates and series always align.

---

## 2. Time index `T`

- Every node in a disease shares **one** time grid (rectangular `X`). Nodes that start/end at different
  dates are placed on the **union** date range; absent steps are imputed for model input and marked `M=0`.
- Calendar timestamps for each step are stored in `meta.dates` (length `T`).

### Temporal-resolution decision — RESOLVED: everything is **weekly**
The client verified that the OpenDengue best-spatial extract **transitions to weekly from 2014 onward**.
We therefore use the **weekly-coded portion** of the dengue extract, which makes all three diseases weekly on
an overlapping calendar:

| Disease | `T_res` used | Handling |
|---|---|---|
| Influenza | **weekly** (epi-week) | as-is (ColaGNN) |
| Ebola | **weekly** | irregular sit-reps → last-cumulative-per-week → differenced (§5) |
| Dengue | **weekly** (2014+) | keep only `T_res == "Week"` rows; legacy monthly dropped |

**How it's selected (robustly).** The loader filters on the **`T_res` column itself** (`t_res_filter="Week"`),
not on a hardcoded year — so it captures exactly the weekly rows regardless of when each country switched, and
generalises if more countries are added. `T_res == "Month"` remains available as an option for a monthly-only
robustness study.

**Why this is the right call for a transfer paper.**
- One cadence ⇒ a single lookback `w` (default 20 weeks), one horizon set `h ∈ {1..4}` weeks, and
  `steps_per_year = 52` for **every** disease — the shared temporal encoder sees comparable dynamics rather
  than having to be cadence-robust across a 4× resolution gap.
- Seasonality (sin/cos day-of-year, §3) is now consistent across diseases.
- **Cost:** pre-2014 monthly dengue (~two decades) is discarded from the harmonised set. This is worth it —
  2014–2023 weekly still gives ~10 years × ~27 Brazil states, and cadence consistency matters far more for the
  transfer claim than extra history. The dropped monthly series can feed an optional **monthly-only robustness
  appendix**, never the main cross-disease experiment.
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
| 3 | `obs_mask` | 1 = observed, 0 = imputed — lets the shared encoder distinguish real observations from filled gaps (**D15**). Constant-1 for fully-observed diseases (harmless); informative for Ebola. |

**EXTENDED (optional, single-disease only; names in `meta.feature_names`, transfer-flag in `meta.core_feature_idx`):**
| name | availability | notes |
|---|---|---|
| `deaths_norm` | Ebola (yes), dengue (partial), flu (no) | normalised deaths/severe; `0` + masked where unreported |
| `logpop_norm` | where a population table is joined | static, broadcast over `T` |
| `temp` / `humidity` / `precip` | dengue/flu if climate joined | **not** in the finalized raw files → absent by default; add via documented join |
| `lag_k` | model-side | never leaks (lag ≥ 1); constructed inside the model, not baked into `X` |

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
- **Few-shot holdout (Ebola):** the scaler is fit on the **support set only** — a **calendar prefix**, every
  observed cell dated on or before `few_shot_support_cutoff` (released **2014-05-24**) — pooled **per-disease**
  (one mean/std), because the handful of early cells is far too few for a stable per-node scaler and fitting on
  anything past support would leak the outbreak scale into the "few-shot" result. The prefix is
  **calendar-causal** (no query cell precedes a support cell) and gated; see §5. Support and query masks are in
  `meta.split`.
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

**Few-shot protocol (D22).** Ebola is `role="few_shot_holdout"`, *not* a 50/20/30 development disease. The
**support set** is a **calendar prefix**: every observed cell dated on or before `few_shot_support_cutoff`
(released value **2014-05-24**) is support; every later observed cell is query. The support/query masks and the
support-fit scaler are in `meta.split` / `meta.scaler`. The support-only scaler is enforced in code, not left
to the training loop.

**Calendar-causal by construction.** Because the split is a date threshold shared by every district, no query
cell is ever earlier in time than any support cell — verified by a build gate (every support cell dated `<=`
every query cell), with a negative control that plants an acausal support cell and requires the gate to fail.
On the released build: 27 support cells (the 9 districts reporting by the cutoff), 1,272 query cells, and 52 of
61 districts are pure zero-shot — the genuine emerging-outbreak regime.

**This replaced a per-district scheme that was not causal.** The earlier support set was the first *n* observed
weeks of *each district*; because districts enter at different dates, 85 per cent of query cells were earlier in
calendar time than the last support cell, so the pooled scaler normalising an early cell drew on peak-epidemic
magnitudes. The old suite could not detect this — it *defined* `support_mask` as the legitimate fit set — which
is why the calendar-causality gate checks the cells' **dates**, not the mask.

**Window size (D22, deferred).** The size of the early window — equivalently, how far the cutoff sits from the
record's start — may later widen from the current eight-week prefix. `few_shot_support_cutoff` is a parameter,
so this is a rebuild, not a code change; and it must be decided together with the causality property, since a
later cutoff pulls more of the epidemic into support.

The same routine differences dengue only if a source is cumulative; OpenDengue `dengue_total` is already
per-period incidence, so dengue is used as-is (`NA`/suppressed → `M=0`).

---

## 6. `meta` contents (reproducibility contract)

`meta` must carry everything needed to interpret, invert and reproduce the bundle:

- `disease` — `"dengue" | "influenza:japan" | "ebola" | ...`
- `node_ids` — length-`N` canonical ids, **row order == every array's row order**
- `adm_level` — `"Admin0|1|2"`; `spatial_unit` label
- `dates` — length-`T` timestamps (MMWR week-ending Sat / month start); `t_res`; `steps_per_year`
- `feature_names` — length-`F`; `core_feature_idx` — transfer channels (default `[0,1,2,3]`)
- `scaler` — per-node `(transform, mean, std)` for inverse-transform; `scaler_scope`
  (`"per_node_train"` | `"per_disease_support"`)
- `split` — dev: `{train_end, val_end}`; few-shot: `{scheme, support_weeks, support_mask, query_mask}`;
  `split_scheme` (`"fixed_50_20_30"` | `"few_shot_support_query"` | `"rolling_origin"`)
- `role` (`"development"` | `"few_shot_holdout"`); `covariates_status`
- `A_geo_kind` (`"queen"|"knn"|"shipped"|"queen+knn_block_diagonal"|"deferred..."`), `A_mob_available`
- `countries`, `countries_dropped_low_coverage`, `countries_requested_absent`, `coverage` (dengue)
- `crs`, `shapefile_ref` — adjacency provenance; `source` — citation/DOI/URL

---

## 7. Adjacency construction `A_geo`

- **Influenza:** **reuse ColaGNN's shipped adjacency** verbatim (`A_geo_kind="shipped"`). Do not rebuild it —
  matching the baselines' graph is part of comparability.
- **Dengue / Ebola:** GADM shapefiles → **queen contiguity** (share a boundary) via `geopandas`. Row order
  aligned to `meta.node_ids`. Dengue is **block-diagonal** across countries (intra-country contiguity only;
  `build_dengue_adjacency`).
- **Joins are by name, verified against codes where codes exist (D17).** The original D17 rationale — "the
  client will not supply GADM/GAUL code columns" — was **wrong**: OpenDengue ships `FAO_GAUL_code`,
  `RNE_iso_code` and `IBGE_code` at ~100 per cent. But a code join is not the fix it appears to be.
  `FAO_GAUL_code` is **not injective on our nodes** (7,443 → 7,027: 416 nodes would silently *merge*, because
  GAUL is a coarser ancestor geography — Taiwan alone collapses 287→2), so joining on it is unsafe. `RNE_iso`
  maps to GADM's `ISO_1` at only ~74 per cent, and GADM carries **no** code at all for the 1,476 Colombia/Peru/
  Argentina/Taiwan Admin-2 nodes. So the name join is unavoidable for four of five Admin-2 countries. It stays:
  `_canon` strips accents/diacritics/whitespace (*Guéckédou* == *gueckedou*), residual mismatches are fixed via
  the `name_aliases` map (234 entries), and any unmatched node **raises with the full list** rather than being
  silently dropped. Where a code *does* exist it is used as an **oracle, not a key**: Brazil's `IBGE_code`
  matches GADM `CC_2` for all 5,517 nodes, and the name join lands on the identical polygon for every one
  (0 disagreements) — a stronger reproducibility claim than a code join, since it validates both the names and
  the codes. Recommended as a build gate.
- **Disconnected units** (islands; or a district with no reporting neighbour): fall back to **k-nearest
  centroids** (default k=4) for those nodes so the graph has no isolated vertices, and flag them
  (`A_geo_kind="queen+knn"`). GNN message passing degenerates on isolated nodes, so this is a correctness fix,
  not a tuning knob.
- Store **raw adjacency** (binary or distance-weighted); let the model add self-loops / normalise so different
  baselines can apply their own convention.
- `A_mob`: Japan prefectures have OD flows (MepoGNN/STOEP) → directed `A_mob`. US flu can add commuting.
  Dengue/Ebola: `None` by default.

---

## 8. Covariates `C`

**Deferred by the client (D20 — "manage later").** `C` ships as a flagged `[N,3]` placeholder
(`meta.covariates_status`), to be populated in Week 2 from shapefiles (`centroid_lat`, `centroid_lon`,
`area_km2`). Population is **not** in the finalized raw set; if a population table (e.g. WorldPop/GADM
attributes) is joined later it
becomes `C[:, pop]` and unlocks the `logpop_norm` extended channel. Keeping `C` to shapefile-derived fields by
default avoids silently introducing an un-provenanced covariate into a publication.

---

## 9. Deliverable & acceptance (Phase-1 DoD)

- `schema_spec.md` (this file); `schema_decisions.md` (the annotated decision register).
- `to_schema.py` — `load_dengue / load_influenza / load_ebola → DiseaseTensors`, plus
  `dengue_country_coverage`, `fit_scalers*` / `apply_scaler` / `invert_scaler`, `build_adjacency` /
  `build_dengue_adjacency`, and `DiseaseTensors.transfer_view()`.
- `test_schema.py` — shape contracts on all three diseases **plus** the tricky invariants —
  cumulative→incidence (envelope + mask), leakage-safe train scaler, few-shot support + disjoint query +
  pooled scaler, `obs_mask` core channel, `transfer_view` core-only, accent-stripped joins, multi-country
  coverage prune.
- `test_leakage.py` — **86 gates** across the five datasets, gating the build, with **6 negative controls**
  that plant a real defect and require the corresponding gate to fail. A gate that has never failed establishes
  nothing: three defects (the clip above; a per-node split fallback that put training cells inside neighbours'
  test period; and an acausal per-district support set) passed 83 gates green and were caught only in
  adversarial review. The gate count is not a measure of coverage — the mass-conservation, phase-purity and
  calendar-causality gates were added precisely because the properties they test were previously untested.

## 10. Open items that still change the schema
- **`[CONFIRM-1]`** target = **cases** (default) vs cases+deaths → whether `deaths_norm`/`y_deaths` is a
  first-class target. *(Still open — not in the last review batch.)*
- **`[CONFIRM-3]`** influenza source = benchmark ILINet+Japan (default) vs adding WHO FluNet (adds a global
  Admin0 flu graph with a different `N`). *(Still open.)*
- Ebola region set for the *main* experiment — **client-controlled** (recommended: West-Africa core).
- Dengue final country list — whatever clears the Week-2 coverage audit (`dengue_country_coverage`), plus the
  Admin1-vs-Admin2 and coverage-threshold defaults.

**Resolved by client review:** D2 (dengue endemic multi-country), D3 (Ebola region control), D5 (weekly
everywhere), D6 (MMWR epi-weeks), D9 (encoder = core only), D10 (incidence = raw counts), D15 (`obs_mask` as a
core channel), D17 (name joins, accent-stripped, verified against codes — **rationale corrected, see above**),
D20 (covariates deferred, flagged),
D24 (Ebola cumulative→incidence — **method since corrected, see §5**), D27 (ColaGNN start weeks from the repo).

**Raised by the Phase-2 client review, and their resolutions** (see `client_decisions.md`,
`remediation_plan.md`):
- **Ebola few-shot causality** — RESOLVED: calendar-prefix support, cutoff 2014-05-24, now gated.
- **Dengue case definitions** — RESOLVED: measured to a 2.8× (Mexico) / 1.6× (Bolivia) step on 41 nodes, not
  the ×220 of the raw-file group-by; kept as built and disclosed.
- **Ebola Western Area** — RESOLVED: parent stays dropped (Urban + Rural are already separate nodes; keeping
  the parent would triple-count), 61 nodes.
- **D22 (few-shot window size)** — DEFERRED: the eight-week cutoff may later widen; decide with the causality
  property, since a later cutoff pulls more of the epidemic into support.
- **Ebola gap-lumping** — RESOLVED: disclose and proceed (A4); the shape is correct and both remedies cost
  more than the distortion.

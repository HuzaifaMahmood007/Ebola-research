# Phase 2 — Developer Execution Guide (Days 6–10)
### Emerging Disease Forecasting Framework · Multi-Disease Data Engineering & Harmonisation

**Who this is for:** the implementation developer, continuing from Phase 1. Every step is written so that no action requires a judgment call you don't have the information to make. Real decisions are marked **[CONFIRM]** with a recommended default so you are never blocked.

**Read this first — what Phase 2 is, and is not.**
Phase 2 turns the *contracts* fixed in Phase 1 (the schema spec, `to_schema.py`, the problem/protocol definition) into *artifacts*: three harmonised, leakage-safe disease bundles that Week 3 can load and train on without touching data code again. Concretely, you will:
- **Acquire and cache** the raw sources (OpenDengue, ColaGNN influenza matrices, the ROWCA/HDX Ebola compilation) with frozen provenance.
- **Harden the loaders** in `to_schema.py` against real data — they are first-draft stubs and *will* break on the actual files (proven below for Ebola).
- **Build the adjacency graphs** that Phase 1 deferred (GADM shapefiles → block-diagonal dengue graph; Ebola district contiguity), including the name-join machinery.
- **Produce and validate** the `DiseaseTensors` bundles + train/val/test (and few-shot support/query) splits, then **audit** coverage, resolution, and missingness — with an explicit confirmation of usable Ebola signal.

You will **not**, in Phase 2: train the encoder (Weeks 3–4), re-run baselines on the harmonised data (Week 3), or add uncertainty/explainability (Week 5). COVID-19 is **not** a development disease — it entered only Phase-1 reproduction and does **not** enter the harmonised schema (brief [CONFIRM-6], manuscript §6.3).

---

## 0. Reality checks you must internalise (read once, then act)

### 0.1 The loaders are contracts, not finished code — expect to patch them
`to_schema.py` encodes the *right design* (leakage-safe scalers, MMWR weekly anchor, core-only transfer view, few-shot support fitting). But it was written before the raw files were in hand. The Ebola loader in particular is coded against the clean national `ebola_data_db_format.csv` (columns `Indicator/value/date`), whereas the file we actually hold is the **OCHA ROWCA "All Sec Review" subnational compilation** with columns `Country/Localite/Category/Value/Date/Sources/Link`. It will not run unmodified. Budget Day 9 accordingly. Patches are specified in **Appendix B**.

### 0.2 Ebola signal is real — the #1 project risk is retired early, not assumed away
The brief flags Ebola scarcity as the top risk and asks Week 2 to *confirm usable signal before the few-shot story rests on it*. An audit of the file on disk (Appendix A) shows the story holds: **64 clean single-district series** across Guinea (32), Liberia (15), Sierra Leone (17), a **mean of ~27 observed weeks per district**, and **53 districts with ≥20 observed weeks**. Even after reserving a 2-week support set, the query set is ample. Do the audit *first thing* on Day 9 so a bad surprise (if the real production file differs) surfaces with two days of slack, not zero.

### 0.3 Harmonisation lives entirely at the data layer — the graph never sees a disease id
The whole architecture's disease-agnosticism (manuscript §6, §9) depends on you enforcing one rule here: **only the core, transfer-safe channels** (`incidence_norm, sin_doy, cos_doy, obs_mask`) are exposed to the shared encoder (`DiseaseTensors.transfer_view()`). Disease-specific extras (Ebola deaths, dengue climate) go in the *extended* block and are used only in single-disease runs. If a channel is present for one disease and absent for another and reaches the shared encoder, the encoder can infer *which* disease it is looking at from the missingness pattern — silently reintroducing the disease-specificity we are paying to remove (manuscript §6.1). This is a correctness property, not a nicety.

### 0.4 "Leakage-safe" is the deliverable's pass/fail gate, not a checkbox
Every claim in the paper rests on out-of-sample performance. Three invariants must hold for all three diseases and be *tested*, not asserted (Day 10): (i) every normalisation statistic and any learned graph structure is fit only on data the model may see at train time (train slice for dev diseases; the few-shot **support** set for Ebola); (ii) no feature is derived from future values; (iii) horizon-`h` targets never appear among inputs. The Ebola scaler is the sharp edge — fitting it on anything beyond the 2-week support set leaks the outbreak's magnitude into an "ostensibly few-shot" result (manuscript §6.4).

### 0.5 This is CPU/IO work — the 8 GB 3060 is irrelevant this week
No training happens in Phase 2. Bottlenecks are download bandwidth, `geopandas`/`libpysal` for contiguity, and pandas reshaping. Memory only matters if you accidentally build a dense adjacency for a huge single-country dengue graph; keep adjacency block-diagonal and per-country and it stays tiny.

---

## Day 6 — Dengue: acquisition, coverage audit, and locking the development country set

**Objective:** obtain the OpenDengue extract, decide (data-driven) which endemic countries carry enough weekly subnational signal, and freeze that set with provenance. Dengue is the largest, densest development disease and the backbone of the "general framework" claim, so its country set must be defensible.

### Task 6.1 — Acquire the OpenDengue "best spatial resolution" extract
**Options.**
- **(A) opendengue.org web extract** — convenient, but the "latest" download is a moving target and not directly citable.
- **(B) GitHub `OpenDengue/master-repo`** — all versions, good for diffing, but you must pick a release tag yourself.
- **(C) Figshare versioned DOI `10.6084/m9.figshare.24259573`** — a frozen, citable snapshot (Clarke et al., *Sci Data* 2024;11(1):296).

**Best option: (C) for the frozen working copy, recorded by DOI + version in `meta['source']`; keep (B) pinned to the same release as a mirror.** Reproducibility and citability dominate here — the paper must state exactly which version produced the tables. Use the **"best spatial resolution" extract** (the spatially-resolved product), *not* the national extract; the national file is only for burden context in prose. Cache the raw file under `data/raw/dengue/` with a SHA-256 checksum.

### Task 6.2 — Choose spatial level and temporal portion
- **Spatial level [CONFIRM-D1]:** **Admin-1** (manuscript Table 3). Alternative is Admin-2 (finer spatial coupling, but far sparser weekly coverage and a much harder name-join against shapefiles). *Best: Admin-1* — it matches the manuscript, keeps coverage high, and keeps the graph legible. Admin-2 is a future refinement, not this week's job.
- **Temporal portion:** **weekly only.** OpenDengue mixes monthly (legacy) and weekly (mostly 2014+) reporting. `load_dengue(..., t_res_filter="Week")` keeps the weekly slice. **Do not interpolate the monthly history up to weekly** — it fabricates observations and inflates the effective sample size (manuscript §6.2 makes this an explicit, principled refusal). The weekly slice still spans roughly a decade in the endemic countries of interest.

### Task 6.3 — Run the country-coverage audit and lock the set data-driven
Run `dengue_country_coverage(csv_path, level="Admin1", t_res_filter="Week")` (already in `to_schema.py`). It returns, per country: `n_nodes`, `max_weeks_per_node`, date span, and endemic-flag. Then let `load_dengue` prune with two thresholds:

- `min_weeks` — minimum observed weeks a node needs to count as usable. **Default 52** (one full seasonal cycle). *Alternative:* 104 (two cycles, stricter, fewer nodes). *Best: start at 52*, and if a country clears with a comfortable margin, note it; you want enough nodes to learn spatial structure, and one clean annual cycle is the minimum for the seasonality channel to mean anything.
- `min_nodes_per_country` — a country must contribute at least this many usable nodes to be worth a graph block. **Default 3.** A one- or two-node "country" contributes no intra-country spatial structure and just adds normalisation noise.

**Why data-driven, not hand-picked:** `DENGUE_ENDEMIC` in the code is a *candidate pool* (WHO/CDC endemic list), not the final set. Hand-fixing the country list invites the reviewer question "why these countries?"; a coverage rule with recorded thresholds answers it objectively. The kept/dropped lists and per-country coverage land in `meta['coverage']`, `meta['countries']`, and `meta['countries_dropped_low_coverage']` — that *is* your audit evidence for dengue.

**Deliverable (Day 6):** cached raw extract + checksum; `dengue_coverage.csv` (the audit table); the locked country set with thresholds recorded; a first `load_dengue(...)` run that returns a `DiseaseTensors` object passing `.check()` (adjacency still deferred — that's Day 7).

---

## Day 7 — Dengue: harmonisation, the block-diagonal graph, and the name-join problem

**Objective:** turn the raw extract into the finished dengue bundle *including* geography, and solve the name-matching that will otherwise silently corrupt the graph.

### Task 7.1 — Harden `load_dengue` against the real file
Expect: the 16-column OpenDengue layout to differ slightly from `DENGUE_COLS` (verify `adm_0_name/adm_1_name/calendar_start_date/dengue_total/T_res/S_res` exist; remap if not); dates that need `dayfirst=True` (handled); suppressed/`NA` `dengue_total` values that must become `M=0`, never `0` counts (handled — NA→mask 0, then `nan_to_num`). Confirm `dengue_total` is **per-period incidence, not cumulative** (it is — so *no* differencing, unlike Ebola). Re-check that MMWR binning (`to_period("W-SAT")`) lands each record in the right epi-week regardless of how OpenDengue labels week-starts.

### Task 7.2 — Build the block-diagonal geographic adjacency
This is the deferred Phase-1 item. Contiguity is built **within** each country and assembled block-diagonally — no cross-border edges (an ocean between Brazil and Thailand is not an edge; manuscript §6.2). `build_dengue_adjacency(shapefiles_by_country, node_ids, name_field)` does this via per-country `build_adjacency` (queen contiguity + k-NN fallback for isolated units).

**Shapefile source — the decision that governs the whole join:**
- **(A) GADM 4.1** (`gadm.org`) — the code's default; clean global Admin-1 polygons.
- **(B) geoBoundaries** — open-licence, sometimes better aligned to statistical names.
- **(C) Natural Earth / FAO GAUL** — coarser, but **this is what OpenDengue itself geomatches against** (its records are matched to RNaturalEarth/GAUL shapefiles).

**Best option — and the subtle trap:** OpenDengue's `adm_1_name` values follow *its own* geomatching source, so joining them to **GADM** names produces systematic mismatches (e.g. GADM's diacritics, alternate spellings, and "State of X" forms). Two defensible paths: **(i)** use **GADM 4.1** (keeps the whole project on one shapefile provenance, code default) and invest in a thorough `name_aliases` map per country; or **(ii)** source the shapefile from the **same lineage OpenDengue matched to** (Natural Earth/GAUL) to minimise the alias burden. *Recommendation: (i) GADM 4.1 + curated alias map*, because you will need GADM anyway for Ebola Admin-2, and one shapefile provenance across the project is cleaner to document than two. Budget real time for the alias map — this is where the day goes.

### Task 7.3 — Make the name-join fail loud, never silent
`build_adjacency` already raises with the full list of unmatched nodes rather than dropping them (manuscript §6.2: "any unit that cannot be matched halts construction with a diagnostic rather than being dropped silently"). Keep that behaviour. Workflow: run once, read the diagnostic, add `data_name → shapefile_name` entries to `name_aliases`, repeat until zero unmatched. A dropped node is a hole in the graph *and* a hole in the training set that you would never see downstream — the loud failure is a feature.

**Deliverable (Day 7):** finished dengue `DiseaseTensors` with `A_geo` (`A_geo_kind="queen+knn_block_diagonal"`, `graph_is_block_diagonal=True`), the per-country `name_aliases` map committed, and `meta` recording shapefile provenance.

---

## Day 8 — Influenza: three benchmark datasets, verbatim graphs, and the COVID decision

**Objective:** ingest the three ColaGNN influenza benchmarks *exactly as shipped* so our influenza numbers sit directly beside the published baselines, and formally close the COVID question.

### Task 8.1 — Acquire the influenza matrices + adjacency
**Options.**
- **(A) ColaGNN-shipped matrices** (`amy-deng/colagnn`, also mirrored in `Xiefeng69/EpiGNN`): Japan-Prefectures (47, weekly ILI, 2012–2019), US-Regions (10 HHS, weekly ILINet, 2002–2017), US-States (49, weekly ILI, 2010–2017), each with its **adjacency matrix**.
- **(B) Delphi Epidata `fluview`** (programmatic, `epidatpy`) — live and reproducible, but you'd rebuild the graph and lose exact comparability.
- **(C) WHO FluNet** — country-level, literally global, but not the benchmark and not sub-nationally resolved.

**Best option: (A), used unchanged, adjacency reused verbatim.** The manuscript commits to this precisely so "our influenza results are directly comparable to the published baselines" (§6.3) — reusing the shipped graph means comparisons happen on an identical graph. (B)/(C) are only for a *supplementary* "literally-global" framing if the client asks ([CONFIRM-D2], default: no). This keeps `A_geo_kind="shipped"` and requires no contiguity build for influenza at all.

### Task 8.2 — Anchor the undated matrices
The shipped matrices are `[T, N]` with **no dates** (the repo README states rows are weeks "in chronological order"). `load_influenza` resolves the first epi-week per dataset from `INFLUENZA_START` (Japan 2012-08-04, US-Regions 2002-01-05, US-States 2010-01-09) and generates a `W-SAT` weekly index. A day-level offset is immaterial — only the year/month drives the seasonality phase (`sin_doy/cos_doy`). Verify each matrix's row count matches the published span (e.g. US-Regions ≈ 785 weeks) before trusting the anchor; if a shipped file's length differs, override `start_date` explicitly.

### Task 8.3 — Keep the three datasets as separate bundles (do not concatenate)
Japan/US-Regions/US-States have **disjoint node sets and different calendars**. Transfer in this project operates through **shared model parameters, not shared indices** (manuscript §5): diseases "neither share a node set nor a common calendar." So produce **three independent `DiseaseTensors`**, each a self-contained development stream, rather than one stitched matrix. This also keeps leave-one-*disease*-out and per-dataset reporting clean.

### Task 8.4 — Formally exclude COVID from the schema
State it in writing (and in the audit note): COVID-19 datasets shipped with the baselines were used **only** in Phase-1 reproduction fidelity checks; the development set is **dengue + influenza** (brief [CONFIRM-6], confirmed). No COVID stream enters the harmonised schema. This pre-empts a reviewer asking why COVID appears in Phase 1 but not the framework.

**Deliverable (Day 8):** three influenza `DiseaseTensors` (`influenza:japan`, `influenza:us-regions`, `influenza:us-states`), each with shipped `A_geo` and a validated date anchor; a one-line COVID-exclusion note.

---

## Day 9 — Ebola: assembly, indicator selection, few-shot protocol, and the signal-confirmation audit

**Objective:** convert the messy ROWCA/HDX subnational compilation into the held-out, data-scarce Ebola bundle, and *confirm* (not assume) that it carries enough signal for the few-shot claim. This is the highest-risk REQUIRED task — do the audit first.

### Task 9.1 — Confirm usable signal FIRST (the brief's explicit gate)
Before any modelling decision, reproduce the coverage audit (Appendix A) on the *production* file: count clean single-district series per core country and their observed-week distributions. The file on disk gives **64 districts, mean ~27 weeks, 53 with ≥20 weeks** — comfortably usable. If the production file is materially thinner, escalate now; you have slack today, none on Day 10.

### Task 9.2 — Clean the compilation (small but non-negotiable)
From the audit, the debt is bounded: drop the **`National`** aggregate (not a district); drop **3 corrupt dates** (`year < 2014`, Excel-epoch artefacts); drop **11 multi-district "blob" rows** (`Localite` containing `, ( ) and`, e.g. *"Guekedou, Macenta and Kissidougou"*, an early-outbreak aggregate that is not a single node); and restrict to the **three core countries** (Guinea, Liberia, Sierra Leone), dropping the near-empty Mali/Nigeria/Senegal series that would distort per-node normalisation (manuscript §6.3). In code, `countries=EBOLA_CORE_COUNTRIES` plus an `exclude_regions` entry for `national` and the blob labels; the garbage-date filter is a one-line patch (Appendix B).

### Task 9.3 — Select the target indicator (a real modelling decision)
The file carries **eight** `Category` values, not the two the loader assumes: `Cases, Deaths, Confirmed cases, Probable cases, Suspected cases, New cases, New Cases, Suspected Cases`.
**Options for the forecasting target.**
- **(A) Headline `Cases` (cumulative) → difference → clip.** Densest series (~10k rows); matches manuscript Eq. (4) exactly. The audit confirms `Cases` is cumulative-with-downward-revisions (Conakry alone: 30 negative steps), which is *why* the `max(0, Cₜ − Cₜ₋₁)` clip exists — downward revisions are reporting reconciliation, not negative incidence.
- **(B) `New cases` directly.** Avoids differencing, but the values are sparser and split across inconsistent casing (`New cases`/`New Cases`), so coverage drops and you inherit a casing-merge chore.
- **(C) Sum of `Confirmed + Probable + Suspected`.** Epidemiologically explicit, but the three sub-series are not always co-reported, so the sum is defined only on their intersection — worse coverage than (A).

**Best option: (A).** It maximises coverage, matches the manuscript's stated method, and the clip rule is already justified in the paper. Use **`Deaths`** (also cumulative) as an **extended, single-disease-only** channel (`deaths_norm`) — never a core/transfer channel (0.3 rule). Canonicalise `Category` so `Cases`/`New Cases` casing collapses correctly.

### Task 9.4 — Cumulative → weekly new incidence
`cumulative_to_weekly_incidence` implements the manuscript's chain: resample to `W-SAT` taking the **last** cumulative value per epi-week, forward-fill, diff, set the first week to the onset level, `clip(lower=0)`, and mask forward-filled weeks as imputed (`M=0`). Two checks: (i) confirm forward-fill only bridges *internal* gaps, not a long trailing tail after reporting stops (a district that stops reporting should go to `M=0`, not carry a flat cumulative forever); (ii) confirm the first-week onset assignment isn't absorbing a huge back-log for late-joining districts — if it is, mask week 0 instead.

### Task 9.5 — District adjacency for the three countries
Build Admin-2 contiguity from **GADM** (same provenance as dengue) for GIN/LBR/SLE. The name-join is harder here than dengue and needs a curated `name_aliases`: **accents** (`Gueckedou → Guéckédou`, `Forecariah → Forécariah`, `Nzerekore → Nzérékoré`), **Liberia's `County` suffix** (`Lofa County → Lofa`), and **Sierra Leone's split units** (`Western area urban`/`Western area rural` → GADM's Western Area names). Use the same loud-failure loop as dengue. Isolated districts fall back to k-NN centroids so no node is degenerate for message passing.

### Task 9.6 — Freeze the few-shot support/query protocol
Per manuscript §5.2/§6.4 and code decision D22: the **support set is the first 2 *observed* weeks per district**; every later observed week is the **query set**. The incidence scaler is fit on the **pooled support observations only, per-disease** (`per_disease=True`) — two weeks per node is far too few for a stable per-node scale, and using anything past support would leak the outbreak's magnitude. `few_shot_support_weeks` is the knob; **default 2** ([CONFIRM-D3]). *Alternative:* 3–4 weeks (more adaptation data, but weaker "emerging-outbreak" realism and closer to a normal split). *Best: keep 2* — it is the operational regime an emerging outbreak actually presents, which is the paper's whole point.

**Deliverable (Day 9):** Ebola `DiseaseTensors` (`role="few_shot_holdout"`, `split_scheme="few_shot_support_query"`, disjoint support/query masks, pooled-support scaler), deaths as an extended channel, GADM Admin-2 `A_geo` with committed aliases, and the signal-confirmation numbers written into the audit note.

---

## Day 10 — Integration: leakage tests, the audit note, reproducible pipeline, and packaging

**Objective:** prove the bundles are leak-free, write the two documentation deliverables, and make the whole thing regenerate from one command.

### Task 10.1 — Automated leakage & invariant test suite
Turn §0.4 into `tests/test_leakage.py` — these are pass/fail gates, not spot checks:
- **Scaler provenance:** re-fit each dev scaler with the test slice zeroed out and assert identical params (i.e. test data never touched the scaler). For Ebola, assert the scaler depends only on support cells.
- **Support/query disjointness:** `support_mask & query_mask == 0` everywhere, and `query_mask ⊆ observed`.
- **No future leakage:** assert no feature column is a function of `t+h`; assert horizon targets aren't among inputs.
- **Mask/imputation:** imputed steps equal the per-node mean in normalised space (0), carry `M=0`, and are excluded from any loss/eval reduction.
- **Schema invariants:** every bundle passes `.check()`; `transfer_view()` returns exactly the 4 core channels for all diseases (identical `F_core` across diseases — the disease-agnosticism guarantee).

### Task 10.2 — Build the rolling-origin split scaffold (for Week 5 robustness)
The manuscript's protocol (§7) uses the fixed 50/20/30 split as headline **and** a rolling-origin, expanding-window backtest (≈5 single-step origins) as robustness, with scalers/graph **refit at each origin**. You don't run backtests this week, but generate and store the origin indices now so Week 5 just consumes them. *Alternative:* generate them later — rejected, because baking them in now guarantees the same leakage discipline (refit-per-origin) is applied consistently rather than reinvented under time pressure.

### Task 10.3 — Write the two documentation deliverables
- **`data_audit.md` (the audit note).** Per disease: sources + versions/DOIs/checksums; spatial units and level; temporal span and cadence (all weekly, MMWR `W-SAT`); **coverage** (dengue: kept/dropped countries + `meta['coverage']`; influenza: node counts + spans; Ebola: the 64-district / observed-week table); **missingness** (mask density per disease, and Ebola's imputed-week fraction); and an explicit **"Ebola signal confirmed"** verdict with the numbers behind it. This is the brief's named "data audit note."
- **`data_pipeline.md` (pipeline documentation).** How to regenerate everything, the schema contract, the alias maps and why they exist, and the `[CONFIRM-D*]` decisions taken. Cross-link `schema_spec.md` from Phase 1.

### Task 10.4 — One-command, deterministic regeneration
Wrap the three loaders in `build_datasets.py` that: reads cached raw (with checksum verification), builds all bundles, runs the leakage suite, writes `.npz` (or `.pt`) per disease into `data/processed/`, and dumps `pip freeze` + config. Determinism: fixed ordering of nodes/dates (already the schema's "single source of truth"), no reliance on dict/set iteration order for the graph. **Best packaging choice:** one `.npz` per disease (portable, no pickle-security or version-coupling issues) carrying `X, A_geo, A_mob, C, M, y` plus a JSON-serialised `meta`; *alternative* is a single pickled `dict` of `DiseaseTensors` (convenient but brittle across library versions and unsafe to share) — prefer `.npz`.

**Deliverable (Day 10):** `data/processed/{dengue,influenza_*,ebola}.npz`; `build_datasets.py`; passing `tests/test_leakage.py`; `data_audit.md`; `data_pipeline.md`.

---

## Phase-2 "definition of done" checklist
- [ ] Dengue: raw extract cached + checksummed; coverage audit run; country set locked data-driven; bundle passes `.check()`; block-diagonal `A_geo` built; alias map committed.
- [ ] Influenza: three benchmark bundles (Japan/US-Regions/US-States) with shipped adjacency and validated date anchors.
- [ ] Ebola: signal confirmed (district + observed-week counts); compilation cleaned (National/blobs/garbage-dates/non-core countries removed); `Cases`→weekly incidence via clip-diff; deaths as extended channel; GADM Admin-2 `A_geo` with aliases; few-shot support/query split frozen; pooled-support scaler.
- [ ] Harmonisation: all diseases on one weekly (`W-SAT`) schema; `transfer_view()` identical core-channel set across diseases; splits (fixed + few-shot) built; rolling-origin indices scaffolded.
- [ ] Leakage suite passes; `build_datasets.py` regenerates all `.npz` deterministically; env frozen.
- [ ] `data_audit.md` and `data_pipeline.md` written; `[CONFIRM-D*]` sent to client.

## Decisions to confirm before locking (send to client)
- **[CONFIRM-D1]** Dengue spatial level: **Admin-1** (default) vs Admin-2.
- **[CONFIRM-D2]** Influenza scope: benchmark ILINet + Japan only (default, SOTA-comparable) vs adding WHO FluNet for a literally-global signal (supplementary).
- **[CONFIRM-D3]** Ebola few-shot support window: **2 observed weeks** (default, per manuscript) vs 3–4.
- **[CONFIRM-D4]** Dengue coverage thresholds: `min_weeks=52`, `min_nodes_per_country=3` (defaults) — approve or tighten.
- **[CONFIRM-D5]** Shapefile provenance: **GADM 4.1** across dengue + Ebola (default) vs matching OpenDengue's Natural Earth/GAUL lineage for dengue.
- **[CONFIRM-D6]** Restate in writing: development set is **dengue + influenza**; COVID excluded from the schema.

---

## Appendix A — Ebola signal audit (from `dataebolapublic_best_yet.xlsx`, sheet *ROWCA Ebola All Sec Review*)
- **58,632** in-window rows (2014–2015); source: OCHA ROWCA compilation of WHO/situation-report data (consistent with the manuscript's HDX + WHO ERT provenance). File spans **2014-03 → 2015-03-28** (~1 year, the outbreak's most intense phase).
- **Countries (rows):** Guinea 27,067 · Sierra Leone 17,312 · Liberia 13,078 · (Mali 630, Nigeria 359, Senegal 189 → excluded).
- **Category values (8):** `Cases` (target, cumulative), `Deaths` (extended), plus `Confirmed/Probable/Suspected/New` variants (not used as target).
- **Cleaning debt:** 1 `National` aggregate; **11** multi-district blob rows (0.0%); **3** corrupt `year<2014` dates.
- **Usable signal (clean, core-country, `Cases`):** **64 districts** — Guinea 32, Liberia 15, Sierra Leone 17. Observed-weeks per district: mean **27.1**, median **30**, max **42**; **57** districts ≥8 weeks, **53** ≥20 weeks.
- **Cumulative-with-revisions confirmed:** `Cases` decreases occur (Kailahun 3, Kenema 6, Lofa 8, Conakry 30), validating the `max(0, Cₜ−Cₜ₋₁)` clip.
- **Verdict:** usable Ebola signal **confirmed**; the few-shot design (2-week support, remainder query) is well within the data.

## Appendix B — Patches `load_ebola()` needs for the real file
The shipped `EBOLA_COLS` won't match. Remap and extend:
- `EBOLA_COLS = dict(country="Country", district="Localite", indicator="Category", value="Value", date="Date")` (note **`Category`/`Value`/`Date`**, capitalised, not `Indicator/value/date`).
- Canonicalise `Category` so `Cases`/`New Cases` casing collapses; select `cases`/`deaths` after canonicalisation (the `_canon` step already lowercases — verify the filter still matches).
- Add a date sanity filter (`2014 ≤ year ≤ 2016`) inside/around `cumulative_to_weekly_incidence` to kill the Excel-epoch rows.
- Pass `countries=EBOLA_CORE_COUNTRIES` and `exclude_regions=["national", <5 blob labels>]`.
- Consider `drop_below_total` to remove any residual near-zero district after cleaning (principled small-node drop, per the docstring).
- Curate `name_aliases` for the GADM Admin-2 join (accents, `County` suffix, Western Area split) before calling `build_adjacency`.

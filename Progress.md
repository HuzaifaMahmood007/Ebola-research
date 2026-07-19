# Task 6.1

Acquired OpenDengue dataset from Figshare. Using the Best Spatial version
https://doi.org/10.6084/m9.figshare.24259573
V 1_3


MD5 checksum from orignal Source (the downloaded .zip):
6b2eba520c6b0831d3900138cb780118

Github Reference:
https://github.com/OpenDengue/master-repo/tree/main/data/releases/V1.3

Verification (local files):
- Spatial_extract_V1_3.zip     MD5    = 6b2eba520c6b0831d3900138cb780118  -> matches source, download intact
- Spatial_extract_V1_3.zip     SHA256 = 69f6f798bc732cf453e405725a1dab2faf595c37bc7b688e783f65ad2dd1e780
- OpenDengue_Best_Spacial.csv  SHA256 = 0f59280ed6795d8a0f7e87f900397db46cab6f2740106718c25c05699b9ababc  (extracted working copy)



# Task 6.2
Spatial level: Admin-1 (adm_1_name). Matches manuscript Table 3; keeps weekly coverage high and the graph legible. Admin-2 deferred as a future refinement.

Temporal portion: weekly only. Do NOT interpolate the pre-2014 monthly history up to weekly.
Reason: interpolating fabricates observations and inflates the effective sample size, contaminating the out-of-sample metrics the whole deliverable rests on (manuscript 6.2 + Phase-2 guide 6.2 make this an explicit refusal). Seasonality (sin_doy/cos_doy) is derived from the calendar date, so the monthly era is not needed to anchor seasonal phase. The weekly slice still spans ~a decade in the endemic countries.

Implementation: filter on the T_res column, not a hardcoded Year >= 2014 cutoff -> load_dengue(..., t_res_filter="Week"). Keeps every genuine weekly record and drops every monthly one. 

# Task 6.3
For countries with frequent,continuous and sporadic risk of dengue we are using WHO CDC data for dengue infected countries:
https://www.cdc.gov/dengue/areas-with-risk/index.html
Risk classification criteria
Classifications are reviewed every two years and align with CDC Yellow Book maps.

Frequent/continuous risk: evidence of more than 10 locally-acquired dengue cases in at least 3 of the previous 10 years. Areas with more than 75% of the country classified as frequent/continuous risk were also grouped in this category, even if some areas have limited or no reported dengue activity.
Sporadic/uncertain risk: evidence of at least 1 locally acquired dengue case during the last 10 years.

Ran the dengue_country_coverage with level="Admin1" and T_res_filter="Week", result stored in dengue_coverage.csv.

Dengue Loader run :

``` 
DiseaseTensors  X=(213, 1096, 4)  A_geo=(213, 213) (deferred)  M=(213, 1096)  y=(213, 1096)
features        ['incidence_norm', 'sin_doy', 'cos_doy', 'obs_mask']  core_idx=[0, 1, 2, 3]
transfer_view   (213, 1096, 4)
calendar        2002-01-05 -> 2022-12-31  (1096 weeks, weekly)
split           fixed_50_20_30  train_end=548  val_end=767
scaler          log1p_z / per_node_train
mask density    0.3028  (70686 observed cells of 233448)

KEPT 8 countries / 213 nodes:
  bolivia                nodes=  9  >=52wk=  9  max_weeks= 574  in_bundle=  9
  colombia               nodes= 36  >=52wk= 36  max_weeks=  88  in_bundle= 36
  dominican republic     nodes= 35  >=52wk= 35  max_weeks= 268  in_bundle= 35
  ecuador                nodes= 25  >=52wk= 16  max_weeks= 151  in_bundle= 25
  japan                  nodes= 47  >=52wk= 47  max_weeks= 312  in_bundle= 47
  mexico                 nodes= 32  >=52wk= 32  max_weeks= 603  in_bundle= 32
  nicaragua              nodes= 18  >=52wk= 18  max_weeks= 981  in_bundle= 18
  panama                 nodes= 11  >=52wk= 11  max_weeks= 248  in_bundle= 11

DROPPED (present at Admin1/Week but failed the gate): 0

NO Admin1/Week rows at all (endemic pool, absent at this level): 98
  afghanistan, angola, anguilla, antigua and barbuda, argentina, aruba, bahamas, bangladesh, barbados, belize, benin, bermuda, bhutan, bonaire, saint eustatius and saba, brazil, brunei darussalam, burkina faso, cabo verde, cambodia, cameroon, cayman islands, central african republic, chad, chile, costa rica, cote d'ivoire, cuba, curacao, dominica, el salvador, eritrea, ethiopia, france, french guiana, ghana, grenada, guadeloupe, guam, guatemala, guinea, guyana, haiti, honduras, hong kong, india, indonesia, italy, jamaica, kenya, kiribati, lao people's democratic republic, macau, malaysia, mali, marshall islands, martinique, mauritania, mauritius, mayotte, montserrat, myanmar, nepal, niue, oman, paraguay, peru, philippines, puerto rico, reunion, saint barthelemy, saint kitts and nevis, saint lucia, saint martin, saint vincent and the grenadines, sao tome and principe, saudi arabia, senegal, seychelles, singapore, sint maarten, spain, sri lanka, sudan, suriname, taiwan, thailand, timor-leste, togo, tokelau, trinidad and tobago, turks and caicos islands, united republic of tanzania, uruguay, venezuela, viet nam, virgin islands (uk), virgin islands (us), yemen

Per-country observed weeks on the union calendar (mean over its nodes):
  bolivia                  561.9 / 1096
  colombia                  88.0 / 1096
  dominican republic       224.8 / 1096
  ecuador                   82.0 / 1096
  japan                    312.0 / 1096
  mexico                   546.6 / 1096
  nicaragua                981.0 / 1096
  panama                   248.0 / 1096

.check() PASSED
```


## Task 6.3 — why we do NOT merge Admin-1 and Admin-2 into one bundle

Each country exists at only one level in the best-spatial extract, so "union across
levels" would be a mixed-resolution bundle: Admin-2 nodes for Brazil/Colombia/Taiwan/
Argentina/Peru + Admin-1 for Japan/DR/Mexico/Ecuador/Nicaragua/Panama/Bolivia.
Two ways to build that single bundle: (A) one load_dengue call picking level per
country; (B) two calls (Admin-1, Admin-2) then merge. Both make the SAME mixed-level
artifact. Issues found:

Inherent to any mixed-level single bundle:
1. Violates the schema contract. schema_spec.md §1: all kept countries must be at ONE
   admin level and one cadence "rather than silently mixed in." Influenza (Task 8.3) is
   the precedent — disjoint node-sets/calendars stay as SEPARATE bundles.
2. Global split starves most countries (CRITICAL). Countries span disjoint windows
   (Taiwan 1997-2024 ... Argentina 2018-2021). On the union ~1997-2024 calendar a global
   50/20/30 cut (train_end ~2011) puts 5 of 12 countries (Japan, Ecuador, Brazil, Panama,
   Argentina) entirely after train_end -> zero train cells -> per-node scaler falls back
   to mean0/std1 -> no learning signal, unnormalized inputs. Silent: .check() still passes.
3. meta cannot describe two levels. Single adm_level/split/scaler scalars mislabel half
   the nodes when serialized to .npz (Task 10.4) and in data_audit.md.
4. Mixed name_field adjacency. build_dengue_adjacency takes ONE name_field (adm_1 vs
   adm_2); a mixed graph either hard-fails the loud name-join or silently mis-joins.
5. Excessive imputation. Short-span countries (Argentina, Panama) become 80-90% M=0 on a
   27-yr calendar, distorting the shared encoder (the mask protects the loss, not the
   input windows fed to the encoder).

Specific to option (B) two-call merge:
6. Scaler leakage (CRITICAL). Concatenating two independently-fit per-node scalers (each
   fit on its own calendar's train_end) leaks train stats into the merged val/test region.
7. Calendar-reindex hazards: aligning two W-SAT period ranges (off-by-one on week-end
   normalization, float32/uint8 dtype drift, sin/cos recompute for padded weeks).

Option A also rejected (evaluated separately). It removes only the merge's *mechanical*
hazards (single _finalise -> no scaler reconciliation, no calendar reindex) but keeps every
intrinsic mixed-level problem above (schema §1 violation, dishonest meta, mixed name_field
adjacency, imputation blow-up) AND adds new silent-corruption risk: Colombia and Panama
exist at BOTH levels, so per-country level selection can double-count via pivot_table sum;
it also requires surgery on the tested load_dengue core. Worst of the three options.

DECISION (final): Admin-1 only, in line with the manuscript ([CONFIRM-D1], schema §1).
No merge, no mixed levels, no Option A, no two-bundle split.
- Build via the existing single call: load_dengue(csv, level="Admin1", t_res_filter="Week").
- Keep the Admin-1 endemic countries that clear the data-driven coverage gate
  (min_weeks=52, min_nodes_per_country=3): japan, dominican republic, mexico, ecuador,
  nicaragua, panama, bolivia (+ colombia only if its 2001-2003 Admin-1 legacy fragment
  clears the gate -- data-driven, not hand-picked).
- Drop the Admin-2-only countries (brazil, taiwan, argentina, peru, and Colombia's modern
  2006-2022 series): their weekly signal exists only at Admin-2 in the best-spatial extract,
  so they are out of scope for an Admin-1 bundle. Record as a known coverage limitation in
  data_audit.md.
- Drop the 4 non-endemic countries that surfaced in coverage (usa, solomon islands,
  pakistan, tonga): not on the WHO/CDC endemic list and all fail min_weeks=52.

STILL OPEN (correctness, not packaging -- flagged for before training / Day 10):
Even Admin-1 only, a GLOBAL positional 50/20/30 split on the union calendar (~2002-2022,
train_end ~2012) starves late-starting countries -> japan (2013), ecuador (2013),
panama (2017) get zero train cells and an undefined (mean0/std1) per-node scaler. This is
the real §0.4 leakage/validity gate and is independent of admin level. Recommended fix:
per-country chronological 50/20/30 on each country's own observed timeline (each node's
scaler fit on its own train window). Resolve before the splits are consumed.


## Task 6.3 — FINAL RESOLUTION (supersedes the "Admin-1 only" decision above)

Expert feedback + review moved the call from Admin-1-only to FINEST-AVAILABLE per country.
Verified against the full CSV: NO country-period carries two S_res levels (the best-spatial
extract already resolves each location-period to exactly one level), so per-country level
selection cannot double-count parent+child. A mixed-resolution graph is fine: it is
block-diagonal (no cross-country edges), scalers are per-node, and the encoder never sees
"level" -> disease-agnosticism is not violated (level is purely a data-layer concern).

RULE (level="auto"): for each endemic country pick the FINEST level that clears the gate --
Admin2 if >= min_nodes_per_country of its Admin2 nodes have >= min_weeks observed weeks,
else Admin1, else exclude. Level domain = {Admin1, Admin2}; Admin0 is NEVER a node level
(a national aggregate is a singleton with no spatial coupling). One level per country,
fixed across all time.

LOCKED SET (finest-available, min_weeks=52, min_nodes=3, country-level prune, all nodes kept):
  12 countries / 7030 nodes / union calendar 1997-2024 (1409 weeks). Brazil = 74.5% of nodes.
  Admin2: brazil(5239), colombia(989), taiwan(274), argentina(236), peru(115)
  Admin1: japan(47), dominican republic(35), mexico(32), ecuador(25), nicaragua(18),
          panama(11), bolivia(9)
  Excluded: admin0_only + insufficient_coverage countries; 4 non-endemic (usa, solomon
  islands, pakistan, tonga) are not in the pool.

DECISIONS:
1. Keep Brazil (do not drop). Dropping the largest dengue signal by a reporting artifact is
   indefensible to a reviewer. Brazil stays at Admin2 (5239 municipalities).
2. Rule = finest-available, NOT coarsest-adequate. Coarsest-adequate buys nothing here
   (Brazil has no Admin1 weekly data, so it stays Admin2 regardless) and would wrongly pick
   Colombia's stale 2001-2003 Admin1 fragment over its rich 2006-2022 Admin2 series.
3. Case-definition aggregation (case_definition_standardised duplicate rows): ON HOLD.
   Detect duplicate (node, period) rows at build time; if present, decide precedence then.
4. Prune at COUNTRY level only (keep all nodes of kept countries, incl. <52wk). Node-level
   pruning NOT applied (does not improve publishing quality -- decision 6 handles dominance).
5. SPLIT = per-country chronological 50/20/30 (headline) + rolling-origin backtest
   (robustness; manuscript §7 / Task 10.2). Global positional split is REPLACED: on the
   1997-2024 union it starves Brazil/Japan/Ecuador/Panama/Argentina (zero train cells ->
   undefined mean0/std1 per-node scaler). Per-country cut -> every country trainable, block
   phase-consistent for the GNN, leakage-safe. Per-node fallback for any node whose obs all
   fall past its country's train boundary. Documented as per-country in methods.
6. EVALUATION = country-macro-average headline (score per node, average within country, then
   average the 12 countries equally) + observed-cells-only scoring + node-micro secondary.
   Neutralizes Brazil's 74.5% node share WITHOUT dropping nodes -> makes decision 4 sound.
   Needs meta['node_country']. Country-balanced training sampler deferred to Week 3.
7. Storage is not the concern; the Day-7 cost is the Admin-2 NAME-JOIN (~6850 Admin-2 units:
   Brazil 5239 + Colombia 989 + Taiwan 274 + Argentina 236 + Peru 115) against GADM under
   fail-loud, with two alias maps (per level). Budget accordingly.
8. API: load_dengue(level="auto") default; level="Admin1"/"Admin2" forced overrides for ablations.

AUDIT stubs (write into data_audit.md):
- [AUDIT-DENG-1] Level domain {Admin1, Admin2}; Admin0 excluded (list admin0_only countries).
- [AUDIT-DENG-2] Finest-available per country (NOT coarsest); per-country level + node counts +
  max single-country node share (Brazil 74.5%).
- [AUDIT-DENG-3] Thresholds min_weeks=52, min_nodes_per_country=3; kept/excluded + reasons
  (admin0_only vs insufficient_coverage).
- [AUDIT-DENG-4] Country-macro-averaged evaluation (headline) + observed-only; node-micro secondary.
- [AUDIT-DENG-5] Case-definition aggregation policy (TBD -- decision 3).
- [AUDIT-DENG-6] Source = full OpenDengue CSV (Figshare DOI), weekly only, monthly NOT
  interpolated; truncated .xls NOT used.
- [AUDIT-DENG-SPLIT] Split = per-country chronological 50/20/30 + rolling-origin robustness.

## Task 6.3 — IMPLEMENTED (to_schema.py) and verified on real data

New/changed code (TDD, 21 tests pass in test_schema.py):
- `_node_col_for(level, cols)` -- replaces the silent `.get(level, adm_0_name)`; raises on
  Admin0/unknown. No node ever collapses to national level.
- `resolve_country_levels(...)` -- finest-available per country (Admin2-first, floor Admin1,
  Admin0 never a node); returns levels + per-country report with exclusion reasons.
- `per_country_chronological_split(...)` -- each country cut 50/20/30 on its OWN observed
  span; per-node fallback guarantees >=1 train cell even for single-observation nodes.
- `load_dengue(level="auto")` (new default) -- per-country level selection, per-country
  split, honest node-resolved meta (node_levels, node_country, country_levels,
  countries_excluded, level_report). "Admin1"/"Admin2" kept as forced ablation overrides.
- `_finalise(split_masks=...)` -- per-node scaler fit on the per-country TRAIN mask (leakage-safe).

Real-data verification (OpenDengue_Best_Spacial.csv, level="auto"):
  12 countries / 7030 nodes / union calendar 1997-2024 (1409 wk). Brazil 74.5%.
  Admin2: brazil, colombia, taiwan, argentina, peru; Admin1: the other 7.
  0 countries starved of train cells; 0 nodes with an undefined scaler; node levels subset
  of {Admin1,Admin2}; train|val|test partitions M disjointly; .check() PASSES.
  Excluded: 57 admin0_only endemic countries (national-only weekly rows).

Two bugs found by verification and fixed:
- dayfirst=True corrupted the real ISO dates (coerced ~19% of rows to NaT, wrongly excluding
  argentina/ecuador). Removed dayfirst; fixtures switched to ISO to mirror the real layout.
- single-observation nodes got no train cell; per-node fallback now guarantees one.

DECISION 3 (case-definition aggregation) RESOLVED = benign. Diagnostic on the real build:
  case_def_mixed_node_weeks = 0 -> NO (node,week) mixes case definitions, so the pivot sum
  never double-counts Suspected+Confirmed+Probable. The 74,952 within_week_summed_rows are
  legitimate sub-weekly reports of the SAME definition collapsed into one MMWR week.
  Note for [AUDIT-DENG-5]: case_definition_standardised is heterogeneous across the corpus
  (mostly 'Probable', then 'Total', with Confirmed/Suspected variants) -- record as a
  provenance note; per-node scaling handles magnitude, but the definition mix is worth stating.

Still deferred (not this component): level-aware block-diagonal adjacency (Day 7),
the automated leakage/invariant suite (Day 10, Task 10.1), and the data_audit.md write-up.


# Task 7.1 — harden load_dengue against the real file: PASSED

Three of the five items were already settled in 6.3 (column remap, dayfirst removed, NA ->
M=0); the two unverified ones now check out on the real file (test_dengue_7_1.py):
dengue_total is per-period incidence, NOT cumulative (0 of 5938 non-constant series monotone
-> no differencing, unlike Ebola), and W-SAT binning is correct (100% of weekly records start
Sunday, span 7 days, land on their own W-SAT period start). The test's first version
false-positived the cumulative check through two bugs of its own (it never excluded constant
series -- an all-zero series is trivially non-decreasing -- and keyed nodes on adm_1_name
regardless of level, collapsing Brazil's municipalities into states); both fixed. That false
positive surfaced 29 of Japan's 47 prefectures reporting zero dengue across all 312 observed
weeks (+1 Ecuador node): the zeros are TRUE (dengue is not endemic in Japan), so per client
decision they are kept, Japan is unchanged, and they are not added to meta -- written up
instead in data_audit.md §1.6 [AUDIT-DENG-4]. Next is Task 7.2, where 6.3 broke the existing
path: the bundle is mixed-level, so build_dengue_adjacency's single name_field cannot serve
both (Brazil/Colombia/Taiwan/Argentina/Peru join on adm_2_name, the other 7 on adm_1_name),
which is why load_dengue hard-codes A_geo=None and ignores shapefiles=; it needs a
per-country name_field, then the fail-loud GADM join over ~6,850 Admin-2 units.


# Task 7.2 — block-diagonal GADM 4.1 adjacency + the name-join: DONE  [CONFIRM-D5 = GADM 4.1]

Graph built and verified for ALL 12 countries. Zero unmatched nodes, zero isolated nodes,
zero cross-border edges, A == A.T, .check() PASSED. 25/25 tests pass.

  DiseaseTensors  X=(7165, 1409, 4)   mask density 0.2175 (2,196,261 observed of 10,095,485)
  A_geo (7165, 7165)  queen+knn_block_diagonal  20,468 undirected edges  degree mean 5.71
  brazil Admin2 5517 (16259 edges) · colombia Admin2 1066 (3021) · argentina Admin2 275 (547)
  peru Admin2 113 (238) · japan 47 · dominican republic 32 · mexico 32 · ecuador 24
  TAIWAN 22 (aggregated) · nicaragua 17 · panama 11 · bolivia 9
  Brazil = 77.0% of nodes (handled by country-macro evaluation, not by dropping nodes).

## 7.2.0 — BLOCKER FOUND FIRST: Admin-2 node ids were colliding (silent data corruption)

Node ids were `country|adm_name`. Admin-2 leaf names are NOT unique within a country:
Brazil carries **239 municipality names shared by 2–5 different states** ('bom jesus' is
five distinct municipalities, ~1,500 km apart), and **64,104 Brazil (name, week) cells**
merged 2+ states. `pivot_table(aggfunc="sum")` was therefore ADDING the case counts of
geographically distinct municipalities into one node. Colombia 82, Taiwan 13, Argentina 40
collided the same way; 413 collisions in total. Task 7.2 is impossible on top of this — a
node called `brazil|bom jesus` has no unambiguous GADM polygon, so the join would have
silently bound it to an arbitrary same-named municipality in the wrong state.

FIX: an Admin-2 node id now carries its parent state — `brazil|piaui|bom jesus`. Admin-1 is
unchanged (`japan|tokyo`). `resolve_country_levels` was hit too: it grouped by leaf name, so
collided municipalities POOLED their observed weeks (overstating coverage, understating
n_nodes) — enough to exclude a country from the gate outright. Both sites fixed, TDD.

  N 7030 -> 7443  (+413, exactly the collision count)
  within_week_summed_rows  74,952 -> 0   <-- the proof

**This retracts the Task 6.3 "DECISION 3 (case-definition aggregation) RESOLVED = benign"
finding.** Those 74,952 rows were NOT "legitimate sub-weekly reports of the SAME definition
collapsed into one MMWR week" — every one was a cross-municipality name collision. With the
parent in the key there are now ZERO duplicate (node, week) cells, so the pivot sums nothing
at all. [AUDIT-DENG-5] must be rewritten accordingly.

## 7.2.1 — the graph builder (mixed-level, which the old one could not do)

`build_dengue_adjacency` rewritten. Each country sits at exactly one level (Task 6.3), so
each block joins GADM at ITS level — a single global `name_field` cannot serve the graph,
which is exactly why Day 6 hard-coded A_geo=None:

    Admin1 node 'japan|tokyo'             -> GADM NAME_1
    Admin2 node 'brazil|piaui|bom jesus'  -> GADM (NAME_1, NAME_2)

Contiguity is queen + k-NN centroid fallback (islands), built WITHIN each country and
assembled block-diagonally (asserted: 0 cross-border edges). **Symmetrised** — Queen is
symmetric but KNN is not, and a GNN needs A == A.T; the old code returned an asymmetric
matrix whenever the k-NN fallback fired.

## 7.2.2 — aliases moved to LOAD time, because they fix a data bug as well as the join

Three further defects, none of which alias-curation-at-join-time would have fixed:

1. **A Cyrillic homoglyph in the source.** Peru's Marañón is encoded with U+0435 CYRILLIC
   SMALL LETTER IE where a Latin 'e' belongs. `_canon` strips accents but cannot repair a
   homoglyph, so that node could never match any shapefile, ever.
2. **Four provinces had their series torn in half.** OpenDengue reports one unit under two
   spellings in STRICTLY DISJOINT eras: DR `salcedo` (2009-13) vs `hermanas mirabal`
   (2006-08) — the province's actual 2007 rename — plus `maria trinidad sanches/sanchez`,
   `santiago. rodriguez` / `santiago rodriguez`, and Ecuador `zamora chincipe/chinchipe`.
   Each was TWO nodes, each holding half a history, each with a per-node scaler fit on that
   half. Disjointness makes the merge unambiguous.
3. **GADM has typos of its own.** Japan's Nagasaki is spelt `Naoasaki`; Peru's Huánuco is
   `Huenuco`. Aliased around.

So `DENGUE_NAME_ALIASES` (dengue_aliases.py) is applied at LOAD time, before the pivot: one
map that both makes the geometry join exact AND reunites the split series. `validate_aliases()`
asserts every alias TARGET is a real GADM key, so a typo in the map fails at build, not later.

Scale of the join: 219 unmatched (ex-Brazil, ex-Taiwan) — but **five Colombian department
names alone orphaned 100 municipalities** (`valle`->`valle del cauca`, `norte santander`->
`norte de santander`, `guajira`->`la guajira`, `san andres`->`san andres y providencia`,
`bogota`->`bogota d.c.`). The rest is editorial drift: `(cd)` corregimiento suffixes,
parenthetical municipal SEATS (inconsistently — sometimes the seat IS the GADM name, so each
was checked individually), Spanish numerals (`25 de mayo`->`veinticinco de mayo`), and GADM's
long official names (`cali`->`santiago de cali`, `cucuta`->`san jose de cucuta`). Every alias
was resolved by EXACT match against GADM, never fuzzy — fuzzy would happily bind
`antioquia|san andres` to `andes`, a different municipality.

## 7.2.3 — units GADM does not have, and the one node we drop

**Pre-split merges (7).** Municipalities created AFTER GADM 4.1's boundary vintage (Albania
2000, Guachené 2006, Tuchín 2007, San José de Uré 2007, Norosí; Peru's Datem del Marañón 2005
and Putumayo 2015). GADM has no polygon for the child, but its PARENT polygon still physically
CONTAINS the child's territory (GADM predates the split), so folding the child's counts back
into the parent is geometrically exact and lossless — strictly better than dropping the node.

**Nicaragua (1).** `zelaya central` and `region autonoma del atlantico sur` overlap FULLY
(981 weeks each, identical span) — they are concurrent MINSA health districts (SILAIS) that
partition the Atlántico Sur department, not era-variants. Summed into the one GADM department
polygon they represent. 18 -> 17 nodes.

**Taiwan (287 -> 22): [CLIENT DECISION].** GADM 4.1 ships NO township layer for Taiwan (levels
0/1/2 only; deepest = 22 counties), and Taiwan has ZERO Admin1 weekly rows to fall back on. Its
287 nodes are townships. No alias fixes a missing layer. Decision: **aggregate the 287
townships into their 22 parent counties** — townships partition counties, so the sum is exact
and double-counts nothing. All 22/22 counties join GADM level 2 (on NAME_2 alone; Taiwan's
NAME_1 is a 7-unit legacy province layer irrelevant to the data). The country is kept; the
"finest available level" rule now means "finest level GADM can actually back".

**Dropped: 1 node.** `argentina|formosa|grl. jose de san martin` — Formosa has exactly 9
departments and none is it (that department belongs to Chaco/Salta, both of which OpenDengue
also reports). A mislabelled source row that cannot be reassigned without inventing data.
Listed with its reason in `DENGUE_UNMAPPABLE` and recorded in `meta['nodes_dropped_unmappable']`;
the graph builder itself NEVER drops a node (§6.2 fail-loud is intact — every removal is a
decision someone wrote down).

Net node effect (ex-Brazil): colombia 1071->1066, DR 35->32 (= GADM's 32 provinces exactly),
ecuador 25->24, nicaragua 18->17, peru 115->113, argentina 276->275, taiwan 287->22.

## 7.2.4 — files + tests

- `to_schema.py` — parent-keyed Admin2 node ids; `build_dengue_adjacency` (mixed-level,
  fail-loud, symmetrised); `_gadm_gdf`/`_contiguity`; load-time alias + unmappable handling;
  `load_dengue(gadm_dir=..., name_aliases=..., unmappable=...)`.
- `dengue_aliases.py` — the committed alias map + `DENGUE_UNMAPPABLE` + `validate_aliases()`.
- `dengue_load.py` — Day-7 driver; asserts 0 cross-border edges and validates alias targets.
- `test_schema.py` — 24 pass, incl. 3 new: Admin2 same-name-different-state nodes stay
  distinct; load-time aliases reunite a split series; unmappable nodes drop only when declared.

## 7.2.5 — written into data_audit.md

- **[AUDIT-DENG-5] RETRACTED and corrected** — the "74,952 within-week summed rows are
  legitimate sub-weekly reports" finding was wrong; every one was a collision. `= 0` now.
- **[AUDIT-DENG-7]** §1.6 the GADM 4.1 block-diagonal graph + §1.6.1 the name-join.
- **[AUDIT-DENG-8]** §1.9 **JUDGMENT CALLS register (J1–J11)** — every place the data was
  changed rather than merely renamed, each with its reasoning, its alternative, and its
  residual risk. J1 Taiwan aggregation (client decision) · J2 seven post-GADM-vintage units
  folded into pre-split parents · J3 Nicaragua SILAIS sum (the one call resting on an
  administrative reading, not arithmetic — flagged as such) · J4 four torn series reunited ·
  J5 the one dropped node · J6 sibling disambiguation · J7 GADM's Santander mislabel ·
  J8 San Fernando two-polygon · J9 GADM's own typos · J10 the Cyrillic homoglyph ·
  J11 k-NN symmetrisation.
- **[AUDIT-DENG-9]** fail-loud is intact: the builder never drops a node; the only removals
  are the declared, reasoned `DENGUE_UNMAPPABLE` entries.
- §1.2 node counts restated (7,030 → 7,443 → 7,165) and §1.4 mask density voided/recomputed
  (21.75% on the final bundle).

## 7.2.6 — BRAZIL joined: 83 unmatched of 5,517, and it added NO judgment calls

All 27 states matched, so Brazil's residue was purely leaf-level and fell into three groups:

1. **71 punctuation.** OpenDengue strips apostrophes and hyphens that GADM keeps
   (`olho d agua` -> `olho d'agua`; `xique xique` -> `xique-xique`). Resolved by an exact match
   on the PUNCTUATION-COLLAPSED name ([^a-z0-9] removed) WITHIN the same state, accepted only
   where the collapsed key was UNIQUE in that state -- 0 ambiguous. That is an exact match
   modulo punctuation, NOT a fuzzy one. Materialised as explicit alias entries rather than left
   as a runtime rule, so the committed map stays the single auditable artifact.
2. **10 orthographic variants** (`itapage`->`itapaje`, `poxoreo`->`poxoreu`, `sao thome`->`sao
   tome`, `belem de sao francisco`->`belem do sao francisco`, ...).
3. **2 municipality RENAMES, with the two sources on opposite sides of each.** Augusto Severo
   (RN) was renamed Campo Grande in 2013 -- GADM has the NEW name, OpenDengue the old. Tabocao
   (TO) was renamed Fortaleza do Tabocao -- GADM has the OLD name, OpenDengue the new. Both
   verified 1:1 renames and NOT merges: the counterpart name is absent from the data in each
   case, so Brazil's node count is unchanged (5517 -> 5517).

Nothing in Brazil was merged, aggregated or dropped => J1-J11 remains the complete judgment-call
list, and no new [CONFIRM] item arises. TASK 7.2 IS CLOSED.


## 7.2.7 — STATIC COVARIATES C populated for dengue (D20 closed) [AUDIT-DENG-10]

C = [centroid_lat, centroid_lon, area_km2], (7165, 3) float32 -- the SAME contract as influenza
([AUDIT-FLU-8]) from the SAME GADM 4.1 provenance. Raw, not standardised (C is static -> no
leakage risk; scaling is the encoder's job). Areas/centroids computed in an EQUAL-AREA
projection (EPSG:6933), never in degrees.

  lat  -54.75 .. 43.37    lon  -115.08 .. 142.55
  area 3.6 .. 368,672.5 km^2  (median 445.2)

Built in the SAME pass as the adjacency, from each country's GADM polygons already reindexed
into node order -- so C's row alignment to node_ids is guaranteed by construction, not
re-derived (and Brazil's 212 MB shapefile is not read twice). Additionally ASSERTED at build:
every centroid must fall inside its own country's bounding box, so a shuffled C fails the build.

Alignment independently validated:
- extremes are the right real units -- smallest = brazil|minas gerais|santa cruz de minas at
  3.6 km2 (Brazil's smallest municipality, official 3.57); largest = bolivia|santa cruz at
  368,673 km2 (Bolivia's largest department, official 370,621).
- the 3 nodes that first failed the box check were REAL offshore islands (Fernando de Noronha
  18.6 km2, Providencia 23.2 km2, Lienkiang/Matsu 32.7 km2). Their centroids are correct; that
  remote islands land where remote islands belong is itself evidence the alignment holds. The
  boxes were widened, not the data.

>> C IS NOT TRANSFER-SAFE, AND THAT IS INTRINSIC -- not a gap to be filled on Day 9. The three
>> diseases occupy DISJOINT geography (dengue LatAm/Asia, influenza US/Japan, Ebola West
>> Africa), so a raw centroid IDENTIFIES THE DISEASE. Populating C for Ebola too would NOT make
>> it safe. C is single-disease-use only and is excluded from transfer_view() by construction
>> (§6.1); meta carries covariates_transfer_safe=False for dengue and influenza alike.
>> WEEK 3: using C in the SHARED encoder would silently reintroduce the disease-specificity the
>> framework exists to remove. If C is wanted there it must first be made geography-free
>> (per-disease standardisation, or area only).

NEXT: Day 9 (Ebola). Note guide §0.1 -- load_ebola() is written against a DIFFERENT file than
the one we hold (it expects Indicator/value/date; the ROWCA compilation has Category/Value/Date),
so Day 9 starts with the Appendix-B patches before any modelling. Signal-confirmation audit
(Task 9.1) runs FIRST -- it is the brief's explicit gate.


# Task 8.1 — acquire the influenza matrices + adjacency: DONE

Option (A), ColaGNN-shipped matrices used unchanged, adjacency reused (edges verbatim).
Acquisition was cheaper than budgeted: the files were already vendored under
baselines/colagnn/data/, and the ColaGNN and EpiGNN copies are BYTE-IDENTICAL (matching
SHA-256 on all six), so there was no version fork to adjudicate. EpiGNN's state-adj-50.txt
(50x50) does not pair with state360's 49 nodes and is not the shipped adjacency.

Cached to `data/Final datasets/influenza/` + SHA256SUMS.txt (verified on every load).
Reason for copying rather than pointing at baselines/: `baselines/*` is GITIGNORED, so the
matrices lived in no repo and no DVC remote -- Task 10.4's one-command regeneration would
have died on a fresh clone. Now under the same DVC-tracked data root as dengue.
STILL OPEN: `dvc add "data/Final datasets"` (run from the conda `ebola` env; dvc is not on
the Python 3.11 that runs the loaders).

Row counts match the published spans exactly (japan 348x47, us-regions 785x10, us-states
360x49). NOTE: I initially read that as validating the date anchors. It does NOT -- row count
constrains T, not the start date, and Task 8.2 below shows both US anchors were 39 weeks wrong
WHILE the row counts matched. Final calendars, after the Task 8.2 correction:
  japan       348 x 47   2012-08-04 -> 2019-03-30   86 edges   mask density 1.000
  us-regions  785 x 10   2002-10-05 -> 2017-10-14   16 edges   mask density 1.000
  us-states   360 x 49   2010-10-09 -> 2017-08-26  103 edges   mask density 1.000
All three fully observed; influenza contributes zero imputed steps. All .check() PASS,
transfer_view() identical 4-channel core across all three. 25/25 tests pass.


## Task 8.1 — THE DECISION: one self-loop convention across all diseases [AUDIT-FLU-4]

FINDING: all three shipped adjacencies carry a self-loop on every node (diag=N). The graphs
we build do not -- build_adjacency() ends with np.fill_diagonal(A, 0.0), so dengue and Ebola
arrive with diag=0.

WHY IT MATTERS (correctness, not tidiness): the shared encoder applies A_hat = A + I once. A
graph arriving with diag=1 gives influenza self-weight 2 against dengue's 1 -- a systematic
structural difference between diseases that changes every node's weighting under degree
normalisation. That is the §6.1 failure mode: a signal the encoder can use to infer WHICH
disease it is looking at. The alternative (teach the encoder to skip +I for influenza only)
is disease-specific special-casing -- the same disease-specificity re-entering by the back door.

DECISION (client-confirmed): zero the diagonal for influenza, matching dengue/Ebola.
A_geo_kind = "shipped(diag_zeroed)". Information-preserving: the shipped matrix is exactly
recoverable as A_geo + I. Comparability is NOT weakened -- only the diagonal is touched, the
off-diagonal edges are reused bit-for-bit (asserted at load time and in test_schema.py), and
after A_hat = A + I the model sees exactly the matrix ColaGNN itself consumes.

CONSEQUENCE -- 4 isolated nodes, and Week 3 must add I:
Zeroing the diagonal exposed what the self-loops had been masking: 4 nodes have NO neighbour
at all; their only entry WAS the self-loop. japan_10, japan_19, us-states_1, us-states_9
(us-regions has none). Inherited from the shipped graph, NOT introduced by us -- every one has
shipped row-sum 1, off-diagonal 0; it is asserted that zeroing never isolated a node that HAD
a real neighbour. Deliberately NOT k-NN-patched (build_adjacency does patch isolates for
dengue/Ebola): inventing an edge absent from the shipped graph destroys the exact comparability
that is the sole reason for reusing it, and these nodes are anonymous indices with no geometry
to fall back on. They stay degree-0 in storage and are made non-degenerate by the encoder's
uniform A_hat = A + I -- self-aggregation only, exactly ColaGNN's own behaviour.
  >> WEEK 3: the encoder MUST add I before any D^-1/2 normalisation or these 4 rows divide by
  >> zero -> NaN. Recorded as meta['requires_encoder_self_loops'] and meta['isolated_nodes'];
  >> asserted in influenza_load.py and test_schema.py so it cannot silently regress.


## Task 8.1 — node ordering: ALL THREE VERIFIED [AUDIT-FLU-5]

No name list ships with the data (README says only "columns indicate locations"); nodes are
anonymous indices. A wrong ordering assumption would silently corrupt any join to
geography/covariates, so all three orderings were TESTED, not assumed.

US-States VERIFIED: hypothesis "50 states alphabetically minus Florida (ILINet does not report
FL)" tested against real US land contiguity -- all 103 shipped edges are genuine contiguity
edges, ZERO spurious, 99.83% pairwise agreement over all 1176 pairs, and the isolated pair is
exactly Alaska + Hawaii, the two non-contiguous states. The only 2 real-contiguity pairs absent
from the shipped graph are Arizona-Colorado and New Mexico-Utah, which meet only at the Four
Corners POINT -> the shipped graph is clean ROOK contiguity (shared border segment), not queen.

US-Regions VERIFIED as HHS 1-10 (was "presumed"): all 16 shipped edges are genuine HHS
contiguity edges, ZERO spurious, degree sequence matches. GADM finds one EXTRA edge, R2-R5 -- a
GREAT LAKES WATER adjacency (GADM state polygons extend into the lakes). The shipped graph uses
strict LAND contiguity, the same convention that omitted the Four Corners point-touches above.

Japan RECOVERED -- see the "Japan node ordering RECOVERED" section below. This paragraph
originally read "Japan UNKNOWN, blocks any covariate join"; that is SUPERSEDED. Recovered by
GADM graph isomorphism + population + seasonal phase -> japan_node_map.csv.

NOTE also disclosed: influenza's shipped graphs are ROOK contiguity while build_adjacency()
produces QUEEN + k-NN. Inherent to reusing the shipped graph; cannot be reconciled without
rebuilding it, which would forfeit comparability. Disclosed, not fixed.


# Task 8.2 — anchor the undated matrices: BUG FOUND AND FIXED (39-week shift) [AUDIT-FLU-3]

THE GUIDE'S CHECK IS INSUFFICIENT. Task 8.2 says verify the row count against the published
span "before trusting the anchor". The row counts MATCHED and the anchor was still wrong by 39
weeks -- row count constrains T, not the start date. Recording this because the check that DOES
work is cheap and should be reused for any undated benchmark.

THE BUG: anchoring the US sets to January (the obvious reading of "2002-2017"/"2010-2017") put
both calendars 39 weeks out of phase. NOT cosmetic: sin_doy/cos_doy are computed from the
assigned date, so the seasonality channel sat ~180 deg out of phase -- telling the encoder
"summer" during every winter flu peak. Under the wrong anchor the aggregate ILI series peaked in
MAY and troughed in OCTOBER, the exact inverse of reality. It would have silently degraded every
US influenza result; neither .check() nor the row-count test would have flagged it.

THE CAUSE: ILINet is published BY INFLUENZA SEASON, so both US matrices begin at MMWR week 40
(the CDC season boundary), not in January.

RECOVERY: the 2009 H1N1 fall wave is ILINet's all-time maximum AND its only major AUTUMN peak,
which makes the series' global argmax a single-week pin on the calendar. It sits at row 368 of
region785 (9.0x the median week); pinning it to the published H1N1 peak week (2009-10-24) gives
start = 2002-10-05 = ISO week 40 of 2002. state360 needs the identical 39-week correction (its
rows are week-aligned to region785) -> 2010-10-09 = ISO week 40 of 2010. Cross-checked against 5
published CDC national peak weeks: 5/5 within 2 weeks, 3 exact.

  us-regions  2002-01-05 -> 2002-10-05  (+39 wk, MMWR wk 40)   H1N1 argmax pin; 5/5 CDC peaks
  us-states   2010-01-09 -> 2010-10-09  (+39 wk, MMWR wk 40)   week-aligned to us-regions
  japan       2012-08-04  UNCHANGED, CONFIRMED CORRECT          peak Feb, trough Aug, 377x
                                                                amplitude, 6/6 seasons Dec-Mar
After correction all three peak in FEBRUARY and trough in JULY/AUGUST, as northern-hemisphere
influenza must. Corrected spans also fit the published ranges better (us-regions 2002-10-05 ->
2017-10-14 = exactly 15 complete flu seasons).

GUARDS so it cannot regress (3 assertions, every build):
1. Seasonal phase -- each aggregate must peak Dec-Mar and trough Jun-Oct. THIS is the check that
   catches a shifted anchor; the row count is not.
2. H1N1 pin -- us-regions' global max must land exactly on 2009-10-24. The winter-peak test alone
   leaves +/-3 wk ambiguity; this pins it to a single week.
3. Saturday anchors (test_schema.py) -- every INFLUENZA_START value must already be a
   week-ending Saturday: pd.date_range(freq="W-SAT") SILENTLY snaps a non-Saturday start to the
   next Saturday, shifting the whole calendar with no error. US anchors additionally asserted to
   fall on the flu-season boundary (ISO week 39-41).
A whole-year (52 wk) offset stays immaterial -- only year/month drives seasonal phase.


# Task 8.3 — three independent bundles, never concatenated: DONE [AUDIT-FLU-2]

Japan / US-Regions / US-States have DISJOINT node sets and DIFFERENT calendars. Transfer here
operates through shared model PARAMETERS, not shared indices (manuscript §5: diseases "neither
share a node set nor a common calendar"), so they stay as three self-contained DiseaseTensors --
each with its own scaler, split and graph -- rather than one stitched matrix. Also keeps
leave-one-disease-out and per-dataset reporting clean.

Made a CHECKED property, not a convention: influenza_load.py asserts pairwise-disjoint node ids
and that no two calendars coincide, so "do not concatenate" cannot be quietly broken later by
someone stacking the matrices. transfer_view() returns the identical 4-channel core block for
all three (the §6.1 disease-agnosticism guarantee). 25/25 tests pass.


# Task 8.x — Japan node ordering RECOVERED (was logged as a blocker) [AUDIT-FLU-5]

Previously logged as "unknown, blocks any Japan covariate join". Now SOLVED -- japan_nodes.py ->
japan_node_map.csv. Three INDEPENDENT lines of evidence (each weak alone, decisive together):

1. STRUCTURE. GADM 4.1 queen contiguity for Japan's 47 prefectures has EXACTLY 86 edges -- the
   shipped graph's count -- and the two graphs are ISOMORPHIC. GADM's isolated prefectures are
   Hokkaido and Okinawa, Japan's only two with no land border. Pins 41/47 nodes uniquely. The
   other 6 sit in 3 AUTOMORPHISM ORBITS that structure physically cannot separate:
   {Hokkaido, Okinawa} (both isolated) + the symmetric Shikoku pairs {Kagawa, Kochi} and
   {Ehime, Tokushima}. I did NOT accept VF2's arbitrary pick among the 8 valid labelings.
2. MAGNITUDE. ILI tracks population; nothing about a graph knows population. Chosen labeling:
   Spearman rho = +0.949 (mean ILI vs prefecture population) over all 47. Top-8 come out as
   Tokyo/Kanagawa/Saitama/Osaka/Aichi/Fukuoka/Chiba/Hokkaido (Japan's most populous), bottom as
   Tottori/Shimane (its two least). Within every ambiguous orbit the higher-ILI node is the more
   populous one.
3. PHASE. Okinawa is subtropical -> out-of-season flu. japan_19 is the LEAST winter-concentrated
   of all 47 (74.0% of ILI in Dec-Mar vs national mean 92.2%, z = -5.05). Phase is independent of
   both structure and magnitude -> pins japan_19=Okinawa, hence japan_10=Hokkaido.

CONFIDENCE (recorded honestly per node in the CSV, not presented as uniform certainty):
  41/47                          structure alone            CERTAIN (no other labeling possible)
  japan_10=Hokkaido, _19=Okinawa structure+magnitude+phase  HIGH (z=-5.05 is decisive)
  japan_5=Ehime, _45=Tokushima   structure+magnitude        GOOD (ILI 2.00x vs pop 1.85x)
  japan_3=Kochi, _23=Kagawa      structure+magnitude        MODERATE -- THE WEAKEST CALL
                                                            (ILI 1.22x vs pop 1.38x)
A Kagawa/Kochi swap changes NOTHING structurally (they are automorphic -> the graph is identical
either way, no model result moves); it only mislabels two adjacent Shikoku prefectures in a
covariate join. Flagged as such rather than dressed up as certain.

GADM 4.1 name quirks handled (each would silently drop a prefecture): 'Hyogo' carries a MACRON
(_canon strips it), and 'Naoasaki' is a GADM TYPO for Nagasaki -> explicit alias.

japan_nodes.py re-asserts all three checks on every run (isomorphism, rho, Okinawa z-score), so
the recovery cannot silently rot. Needs geopandas: conda run -n ebola python japan_nodes.py


# Static covariates C + mobility A_mob — DECISIONS [AUDIT-FLU-8]

Both were empty for EVERY disease (C = zeros[N,3] flagged "PLACEHOLDER ... (D20 deferred)";
A_mob = None). Surfaced now because Week 2 IS the deferred window.

C  -> POPULATE PER DISEASE, INFLUENZA FIRST (client decision). Centroid lat/lon + area from
      GADM 4.1 ([CONFIRM-D5], same provenance as dengue/Ebola). Japan joins via the recovered
      node map; US-States via the verified alphabetical ordering; US-Regions by unioning each
      HHS region's member-state geometry.
      >> CORRECTNESS CONSTRAINT (easy to violate, so recorded): while C is populated for
      >> influenza but still all-zero for dengue/Ebola, C MUST NOT reach the shared encoder. A
      >> real C for one disease and a zero C for another is a perfect disease tell -- the same
      >> §6.1 failure mode as the self-loop diagonal, through a different door. transfer_view()
      >> returns only the 4 core channels and excludes C, so the guarantee holds BY
      >> CONSTRUCTION today. Keep it that way until all three diseases carry a populated C.
      >> Single-disease runs may use C freely.
A_mob -> DECLINED, None for all three (client decision). MepoGNN ships a real 47x47 directed
      Japan commute matrix (commute_jp.npy, max ~5.4M commuters, same node count as
      influenza:japan) -- tempting, but no mobility data exists for dengue or Ebola, so attaching
      it to Japan alone reintroduces exactly the asymmetry the diagonal fix removed; its own node
      ordering is undocumented; and mobility is outside the Phase-2 scope.

UNBLOCKED + DONE. GADM USA added by user -> C is now POPULATED for all three influenza bundles.
C = [centroid_lat, centroid_lon, area_km2], GADM 4.1, areas in an EQUAL-AREA projection
(EPSG:6933, never degrees). Raw values, not standardised (C is static -> no leakage risk;
scaling is the encoder's job). Japan joins via the recovered node map; us-states via the verified
alphabetical ordering; us-regions by unioning each HHS region's member states.

  japan       C=[47,3]  largest japan_10=Hokkaido 78,058 km2   total 372,468 (published 377,975)
  us-states   C=[49,3]  largest us-states_1=Alaska 1,507,385    total 9,325,229
  us-regions  C=[10,3]  largest us-regions_9=HHS R10 (Alaska)   total 9,472,856

A MIS-JOINED GEOMETRY WOULD BE SILENT (plausible numbers on the wrong nodes, nothing downstream
complains), so C is tested against geography known independently of GADM --
test_influenza_covariates.py, 5 tests, run: conda run -n ebola python test_influenza_covariates.py
  japan       largest AND northernmost = Hokkaido; southernmost+westernmost = Okinawa; smallest =
              Kagawa; top-5 areas EXACTLY the published ranking (Hokkaido, Iwate, Fukushima,
              Nagano, Niigata); total within 1.5% of published.
  us-states   largest+northernmost = Alaska; southernmost = Hawaii; smallest = Rhode Island;
              top-5 EXACTLY Alaska, Texas, California, Montana, New Mexico.
  us-regions  every region's area == the SUM of its member states' areas (region built by union,
              states built independently -> can only agree if BOTH joins are right). And
              regions-total MINUS states-total = 147,627 km2 = FLORIDA + DC exactly -- the units
              the region series includes and the 49-state series excludes.
These are a THIRD independent confirmation of the recovered node orderings: a wrong ordering
would not put Alaska on us-states_1 or Hokkaido on japan_10.

ALSO VERIFIED (was previously "presumed, not verified"): us-regions node order IS HHS 1-10. All
16 shipped edges are genuine HHS contiguity edges (ZERO spurious); GADM finds one EXTRA edge,
R2-R5, which is a GREAT LAKES WATER adjacency (GADM state polygons extend into the lakes). The
shipped graph uses strict LAND contiguity -- the same convention that made it omit the Four
Corners point-touches for us-states. Consistent; ordering confirmed.

NOTE: build_influenza_covariates needs geopandas, which is only in the conda `ebola` env, so it
is NOT covered by test_schema.py (python 3.11). It is covered by test_influenza_covariates.py +
the influenza_load.py driver asserts. Run both under conda.


# Task 8.4 — COVID formally excluded [CONFIRM-D6]

Written into data_audit.md §2.8. COVID datasets shipped with the baselines (japan-covid,
australia-covid, spain-covid, JHU global) were used ONLY in Phase-1 reproduction fidelity
checks. Development set is dengue + influenza. No COVID stream enters the harmonised schema
and no COVID bundle exists in data/processed/.
[CONFIRM-D2] resolved: benchmark-only. WHO FluNet NOT added (country-level, not sub-nationally
resolved, not the benchmark, and would need a graph built from scratch).


# DAY 8 — CLOSE-OUT: what is done, what is left

Guide's Day-8 deliverable: "three influenza DiseaseTensors (influenza:japan, influenza:us-regions,
influenza:us-states), each with shipped A_geo and a validated date anchor; a one-line
COVID-exclusion note."

DONE:
  8.1  acquire matrices + adjacency      cached + SHA-256, verified on every load
  8.2  anchor the undated matrices       39-week bug FOUND AND FIXED; 3 regression guards
  8.3  three independent bundles         disjoint node sets + calendars, ASSERTED not assumed
  8.4  COVID formally excluded           data_audit.md §2.8

BEYOND the guide (done because they were blocking or load-bearing, not scope creep):
  - Japan node ordering recovered (japan_node_map.csv) -- the guide never asks, but C could not
    be built without it, and "anonymous nodes forever" was a real dead end.
  - All three node orderings verified against independent geography.
  - Static covariates C populated for influenza (D20, the deferred Phase-1 item).
  - us-regions ordering verified as HHS 1-10.

TESTS
  python test_schema.py                                    25/25   (py3.11)
  conda run -n ebola python test_influenza_covariates.py    5/5    (needs geopandas)
  conda run -n ebola python influenza_load.py              builds all 3 bundles, all .check() PASS
  conda run -n ebola python japan_nodes.py                 regenerates the node map, self-asserting

LEFT FOR DAY 8:

  [x] `dvc add "data/Final datasets"` -- DONE. The pointer moved nfiles 3 -> 12,
      size 504,354,721 -> 559,281,566 bytes, so the 6 influenza matrices + SHA256SUMS.txt are
      under DVC. The .dvc file is modified-but-uncommitted in git.

  [ ] `dvc push` + commit the .dvc pointer. Without the push, the influenza raws exist in the
      local DVC cache only -- the Azure remote (azure://ebola-dataset/) does not have them, so a
      fresh clone still cannot `dvc pull` them. Task 8.1's provenance story is not closed until
      this lands. (Outward-facing -- your call, not run automatically.)

  [ ] DECIDE: data/gadm/ is UNTRACKED -- neither git nor DVC. 461 MB, 13 shapefiles.
      This is a REAL reproducibility hole, not bookkeeping: the dengue block-diagonal graph
      (Task 7.2) and influenza's C (Task 8.x) BOTH depend on it, and a fresh clone has neither.
      Task 10.4 ("one-command deterministic regeneration") cannot pass while it is untracked.
      Two defensible options:
        (a) `dvc add data/gadm` -- 461 MB to the Azure remote. Matches the existing data
            provenance pattern; robust to GADM ever going down or re-cutting a release.
        (b) a fetch script + SHA-256 manifest. GADM 4.1 is a FROZEN, versioned, publicly-hosted
            release (geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_<ISO3>_shp.zip), so the URLs
            are stable and citable. Keeps 461 MB of third-party public data out of our remote.
      RECOMMEND (b): it is third-party, frozen, and re-downloadable; checksums give the same
      reproducibility guarantee at ~0 storage. Note GADM's licence forbids redistribution for
      commercial use, which (a) arguably does -- another reason to prefer (b).
      Either way it must be resolved before Day 10.

NOT Day 8 (deliberately deferred; these belong to Day 10, do not do them now):
  - .npz packaging of the bundles into data/processed/   -> Task 10.4 (build_datasets.py)
  - automated leakage/invariant suite                    -> Task 10.1
  - rolling-origin split indices                         -> Task 10.2
  - data_pipeline.md                                     -> Task 10.3
  The influenza bundles currently exist only in memory when influenza_load.py runs. That is
  CORRECT for Day 8 -- persistence is Task 10.4's job, and doing it early would mean writing
  .npz twice (Ebola's schema lands on Day 9).

CARRIED INTO WEEK 3 (recorded so it is not rediscovered):
  - The encoder MUST add I before any D^-1/2 normalisation, or the 4 isolated influenza nodes
    (Alaska, Hawaii, Hokkaido, Okinawa) divide by zero -> NaN. meta['requires_encoder_self_loops'].
  - C must NOT reach the shared cross-disease encoder while it is populated for influenza and
    all-zero for dengue/Ebola -- that asymmetry is a perfect disease tell (§6.1).
    meta['covariates_transfer_safe'] = False; transfer_view() excludes C.

NEXT: Day 9 (Ebola). Task 9.1 is the signal-confirmation audit and the guide says do it FIRST --
it is the #1 project risk and today is the day with slack.


# DAY 9 — EBOLA

Bundle built and verified. 61 districts x 52 weeks, few-shot support/query, GADM graph with
the tri-border edges intact. 33/33 tests pass (was 25; +8 Ebola).

  DiseaseTensors  X=(61, 52, 5)  A_geo=(61,61)  M=(61,52)   mask density 0.5255
  calendar        2014-04-05 -> 2015-03-28  (52 weeks, W-SAT)
  role/split      few_shot_holdout / few_shot_support_query   scaler log1p_z/per_disease_support
  nodes           guinea 32 · liberia 15 · sierra leone 14
  A_geo           146 undirected edges · 21 CROSS-BORDER · 0 isolated · degree mean 4.79
  few-shot        support 121 cells · query 1,546 cells
  source sha256   937136a42685ab596d7eb9fc8864ae41d2c07cb546714704d3d31f0e7ae52972


## Task 9.1 — SIGNAL CONFIRMED (the brief's gate). ebola_audit.py

Every Appendix-A number reproduces EXACTLY on the production file. The #1 project risk is
retired: 64 clean single-district series (guinea 32, liberia 15, sierra leone 17), mean 27.1
observed weeks, median 30, max 42; 57 districts >=8 weeks, 53 >=20. 58,632 in-window rows,
2014-03-24 -> 2015-03-28. Conakry's 30 downward revisions reproduce to the digit.

'Cases' is CUMULATIVE WITH REVISIONS -- and far more pervasively than Appendix A implied:
only 5 of 64 series are monotone, and 58 of 64 carry at least one downward step. That is a
much stronger justification for the max(0, C_t - C_t-1) clip than the guide's four examples.

FOUR THINGS THE GUIDE DID NOT FLAG (all real, all now handled):
  1. 'National' is 1 LABEL but 3,613 ROWS (6.2% of the file). Appendix A's "1 National
     aggregate" reads as one row.
  2. 7 non-numeric `Value` cells (' ', '-'). Never mentioned. They must be dropped BEFORE
     the de-dup that keeps the last row per (node, date), or a junk cell silently
     OVERWRITES that district's real cumulative count for the week.
  3. 2 districts have 0 query weeks under a 2-week support set (min observed = 1 week).
  4. The week-0 back-log (see 9.4) -- the biggest correctness find of the day.


## Task 9.2 — CLEANING. Sierra Leone's "17 districts" are 14 + THREE ARTEFACTS

The guide's cleaning list (National, 11 blobs, 3 corrupt dates) is right but INCOMPLETE.
Sierra Leone has only 14 districts, yet reports 17 labels. The three extras are not
districts, and none of them appears anywhere in the guide:

  1. **'port' IS 'port loko', torn in two.** `port loko` runs 2014-06-17 -> **2014-11-26**;
     `port` begins **2014-11-27** -- the very next day -- and CONTINUES THE SAME CUMULATIVE
     (1041 -> 1923). The label was truncated mid-compilation. This is also the sole source
     of the worst week-0 anomaly: `port`'s C_0 = 1062 is not a back-log at all, it is Port
     Loko's inherited cumulative. MERGED at load time -> one continuous 35-week series, one
     of Sierra Leone's best. (Exactly dengue's J4 torn-series pattern.)
  2. **'western area' is the PARENT of Western Area Urban + Rural and OVERLAPS them in
     time** (its 3 reports run Jun-Oct 2014; the children start 2014-08-15). Keeping all
     three would triple-count the same cases. DROPPED -- the two children have 31-32 weeks
     each against the parent's 3, so the finer pair is strictly better.
  3. **'freetown' is a CITY inside Western Area Urban**, not a district. 1 report, 5 cases,
     no district polygon exists for it. DROPPED.

Every removal is DECLARED WITH ITS REASON in EBOLA_DROP_LABELS / EBOLA_DROP_NODES and
recorded in meta['nodes_dropped'] (10 drops). The loader never drops anything else.

FAIL-LOUD GUARD: blobs are ALSO detected structurally, and an UNDECLARED blob label RAISES
rather than silently becoming a node. So a refreshed file that introduces a new aggregate
label halts the build and asks a human, instead of quietly ingesting a 3-district blob as
if it were one place. (Pinned by test_ebola_unlisted_blob_label_fails_loud.)

  64 raw district LABELS  ->  61 BUNDLE NODES.  The delta is exactly these 3.
  Both numbers are right; ebola_audit.py now prints the reconciliation so they are never
  read as contradicting each other.


## Task 9.3 — INDICATOR: option (A), headline cumulative `Cases`. Confirmed, no change.

8 raw Category values collapse to 6 under _canon (the New cases/New Cases and Suspected
cases/Suspected Cases casing pairs merge). `Cases` (10,192 rows) is the densest and matches
manuscript Eq. (4). `Deaths` -> extended channel `deaths_norm`, never core (0.3 rule;
transfer_view() asserts 4 core channels). The other 5 categories are never summed in --
pinned by a test, because `_canon("New cases") == "new cases" != "cases"` is the only thing
stopping them, and that is too quiet a guarantee to leave untested.


## Task 9.4 — THE WEEK-0 BACK-LOG. The biggest find of Day 9.

`new.iloc[0] = weekly_cum.iloc[0]` assigned each district's ENTIRE cumulative-to-date as its
week-0 incidence. The guide told us to check this ("confirm the first-week onset assignment
isn't absorbing a huge back-log ... if it is, mask week 0 instead"). It is. Measured:

  * week 0 held **4.1% of ALL Ebola incidence** in the bundle (1,457 of 35,670 cases)
  * **10 of 64 districts** got their ALL-TIME MAXIMUM weekly value as a week-0 spike
  * worst: `sierra leone|port` C_0 = **1,062** vs a median weekly increment of **41** (26x)
  * 25 of 64 districts had C_0 above their own median weekly increment; 7 above 10x

And it is not merely a late-joiner problem. **The compilation opens on 2014-03-24, months
into an outbreak that began in Dec 2013** -- so EVERY district's first report is a back-log,
including the index districts (Guéckédou's C_0 = 85 against a median of 8).

The root cause is not a tuning issue, it is identifiability: at a district's first report we
have C_0 but no C_-1, so the increment **does not exist**. Assigning C_0 fabricates it.

FIX: the first observed week of every district is now MASKED (M=0). The mask exists for
precisely this case. Cost: 61 cells. It also removes a leak we had not noticed --
**under the old code week 0 was support week #1 for every district**, so those fabricated
back-logs (Port Loko's 1,062 included) were the dominant input to the POOLED support scaler,
i.e. the normalisation of the entire held-out disease was being set by invented numbers.

  meta['ebola_first_week_masked'] = True.  Pinned by 3 tests.


## Task 9.5 — GRAPH: NOT block-diagonal. A deliberate departure from dengue. [DECISION]

Dengue forbids cross-border edges because its countries are not adjacent -- "an ocean
between Brazil and Thailand is not an edge" (§6.2). **That rationale does not transfer.**
Guinea, Liberia and Sierra Leone are PHYSICALLY CONTIGUOUS, and this was ONE outbreak that
spread across them: the Guéckédou-Lofa-Kailahun tri-border area is the defining transmission
pathway of the 2014 epidemic. A block-diagonal graph would sever exactly the edges that
carry the signal -- a modelling error, not a convention choice.

So contiguity is built over the UNION of all 61 districts. Result: 146 edges, of which
**21 are cross-border**, and all three tri-border edges are present and ASSERTED on every
build (guéckédou-lofa, guéckédou-kailahun, lofa-kailahun). 0 isolated nodes, A == A.T,
diag = 0 (self-loops remain the encoder's job, consistent with influenza's F1 decision).
meta['graph_is_block_diagonal'] = False.

THE JOIN. GADM levels are MIXED, which the guide did not anticipate:

    guinea        GADM level 2 (34 prefectures)  key NAME_2
    liberia       GADM level 1 (15 COUNTIES)     key NAME_1   <-- level 1, NOT 2
    sierra leone  GADM level 2 (14 districts)    key NAME_2

Liberia's counties are GADM level 1; its level 2 is 66 districts the data never uses. So
Ebola is a mixed-level graph, exactly like dengue -- adm_level = "per_country".

The guide predicted accents would need aliasing (`Gueckedou -> Guéckédou`, `Nzerekore ->
Nzérékoré`). **They do not** -- `_canon` already strips them. The aliases actually needed are
a different problem entirely: TWO GADM TYPOS.

    guinea|yomou     -> GADM ships 'Yamou'    (Guinea has no prefecture called Yamou)
    liberia|gbarpolu -> GADM ships 'Gbapolu'
    sierra leone|western area urban/rural -> GADM drops the 'Area'

AND A CONVENTION FIX. Dengue folds GADM's quirks into its LOAD-time alias map, so dengue's
node ids inherit GADM's typos -- the dengue bundle literally contains a node called
**`japan|naoasaki`** and another called **`peru|huanuco|huenuco`** (dengue_aliases.py:314,
329). Influenza maps the other way and keeps the correct name. Ebola follows influenza:

    EBOLA_NAME_ALIASES  (LOAD time)  repairs the DATA   -- the port/port-loko merge, and
                                     Liberia's internally inconsistent 'County' suffix
                                     (11 of 15 counties carry it, 4 do not)
    EBOLA_GADM_FIX      (JOIN time)  absorbs GADM's defects -- they never rename our nodes

  >> CARRIED TO DAY 10 (not fixed here, out of scope): dengue's node ids carry GADM's
  >> typos. `japan|naoasaki` / `peru|huanuco|huenuco` would land in the paper's node table.
  >> One-line fix each, but it changes dengue node ids -> needs a rebuild + test pass.


## Task 9.6 — FEW-SHOT: support = 2 observed weeks. [CONFIRM-D3] = 2, kept.

Support = the first 2 OBSERVED weeks per district; every later observed week is query.
Because week 0 is masked, support is now the first two REAL INCREMENTS and can never be the
fabricated back-log. Scaler fit on the pooled support cells only, per-disease (2 weeks/node
is far too few for a per-node scale). support 121 cells / query 1,546 cells; support/query
disjoint and query ⊆ observed, both asserted.

LEAKAGE GATE (§0.4). Asserted on every build AND in the test suite: the scaler is exactly
the pooled log1p mean over support cells, and inflating a QUERY week 1000x does not move it.
  >> A subtlety worth recording for Day 10's leakage suite: you CANNOT probe this by
  >> perturbing week 0. The series is CUMULATIVE, so changing C_0 also changes the week-1
  >> increment. My first version of this test did exactly that and failed for the wrong
  >> reason. Perturb a QUERY week -- that moves one increment and nothing else.

DISCLOSURES (real properties, recorded so evaluation does not trip on them):
  - `guinea|lelouma` has 1 observed week -> support takes it, 0 query cells. It is a graph
    node that is NEVER SCORED. meta['nodes_without_query'].
  - 4 zero-incidence districts (guinea: dinguiraye, gaoual, lelouma, tougue) reported for
    6-41 weeks but never a NEW case -- their 1-3 cases predate their first report and sit in
    the masked back-log. Correct, not a bug. Trivially predictable (emit 0). Kept: real
    districts, valid message-passing neighbours. Unlike dengue's zero-variance nodes (§1.7)
    their scaler CANNOT degenerate -- Ebola's is pooled per-disease, not per-node.


## D20 — STATIC COVARIATES C: Ebola populated. All three diseases now carry a real C.

C = [centroid_lat, centroid_lon, area_km2] from GADM 4.1, equal-area projection (EPSG:6933),
raw/unstandardised. Cost one line: build_ebola_adjacency ALREADY holds the GADM polygons
reindexed into node order, so C falls out of the same pass as the graph -- row alignment to
node_ids is guaranteed by construction, not re-derived, and the shapefiles are read once.

  C  (61, 3)   guinea 32 / liberia 15 / sierra leone 14

>> THE CONSTRAINT DOES NOT LIFT, AND THE OLD REASON FOR IT WAS INCOMPLETE.
>> data_audit.md said C must stay out of the shared encoder "until all three diseases carry a
>> populated C, at which point C may be reconsidered as a shared input". All three now do --
>> and C is STILL not transfer-safe. The asymmetric-missingness tell is gone, but a stronger
>> one was always there: the diseases occupy DISJOINT GEOGRAPHY (ebola West Africa ~4-13N
>> 7-16W, dengue Latin America/Asia, influenza US/Japan), so a RAW CENTROID IDENTIFIES THE
>> DISEASE OUTRIGHT -- more cleanly than any missingness pattern could. Populating C cannot
>> fix that; it is intrinsic to what C is. covariates_transfer_safe = False on every bundle;
>> transfer_view() excludes C by construction. To share C at all, the coordinates must first
>> be made geography-free (centroid RELATIVE to its country centroid, or area alone) -- a
>> Week-3 modelling decision, not a data one. §2.10 and §3.10 corrected accordingly.

A MIS-JOINED GEOMETRY IS SILENT (plausible numbers on the wrong districts, nothing downstream
complains), so C is asserted on every build against West-African geography known INDEPENDENTLY
of GADM:
  liberia   all 15 counties  ->  95,918 km2 vs published LAND 96,320   (0.4%)
  sierra leone all 14 dists  ->  72,601 km2 vs 71,620 land             (1.4%)
  guinea    32 of 34 prefs   -> 229,694 km2, under the country's 245,857; the shortfall is
                                approx Koubia + Mandiana, the two that never reported
  largest/smallest  liberia = Nimba / Montserrado · sierra leone = Koinadugu / Western Area
                    Urban (the Freetown peninsula)
  N/S extremes      guinea|koundara (Guinea's northernmost) / liberia|maryland (Cape Palmas)
  C vs A_geo        centroid distance across the 146 edges: mean 0.83deg, max 1.66deg
The two COMPLETE-country sums are the strongest check: the units were joined INDIVIDUALLY, so
they can only add up to the national figure if every one of them hit the right polygon. The
last row cross-validates C against the graph -- a scrambled join puts "adjacent" nodes far apart.

  >> The area check FAILED on its first run (13.9% off) and the reference was wrong, not the
  >> join: Liberia's TOTAL area (111,369 km2) includes ~15,000 km2 of water, and GADM ships
  >> LAND polygons. Against land area it is 0.4%. Recorded because it is an easy way to
  >> mis-read a correct result as a broken one.


## Day 9 — files + tests

  to_schema.py      cumulative_to_weekly_incidence: week-0 masked, dayfirst removed, junk
                    Value dropped before de-dup. EBOLA_COLS -> the real Category/Value/Date
                    layout. _read_ebola (.xlsx/.csv dispatch). EBOLA_DROP_LABELS /
                    EBOLA_DROP_NODES / EBOLA_NAME_ALIASES / EBOLA_GADM_FIX / EBOLA_GADM.
                    build_ebola_adjacency (union graph, fail-loud, cross-border reported).
                    load_ebola rewritten.
  ebola_audit.py    Task 9.1 gate. Read-only. Asserts against Appendix A; escalates on drift.
  ebola_load.py     Day-9 driver. Everything it prints, it asserts.
  test_schema.py    33 tests (+8 Ebola). Fixture now mirrors the REAL sheet.

  python test_schema.py                      33/33   (py3.11, no geo deps)
  conda run -n ebola python ebola_audit.py   gate PASSES
  conda run -n ebola python ebola_load.py    builds the bundle, all asserts pass

  NOTE: openpyxl was added to the conda `ebola` env (pandas needs it to read .xlsx). The
  test fixtures stay CSV, so test_schema.py keeps running on py3.11 with no new deps.

LEFT / CARRIED:
  [x] data/gadm/ now holds GIN+LBR+SLE too (user added). RESOLVED -- see PRE-DAY-10 below.
  [x] dvc push (carried from Day 8). RESOLVED -- remote in sync.
  [ ] dengue node ids carry GADM's typos (see 9.5). Fix before the node table is published.
  [ ] data_audit.md §3 written (below). §4 harmonisation/leakage is Day 10.


# PRE-DAY-10 VERIFICATION GATE — Days 6-9 CONFIRMED COMPLETE

Everything below was RE-RUN, not read off this file. Two items this document still carried
as blockers turned out to be already resolved; one real blocker remains.

## Gates re-run (all pass)

  python test_schema.py                                    33/33
  conda run -n ebola python test_dengue_7_1.py             PASSED (per-period incidence, W-SAT)
  conda run -n ebola python test_influenza_covariates.py    5/5
  conda run -n ebola python ebola_audit.py                 Task 9.1 gate PASSES
  python fetch_gadm.py --verify                            16/16 shapefiles verified

CAUTION -- the tree had REGRESSED away from the state this file describes. On first run
test_schema.py did NOT pass: `_read_ebola(csv_path, sheet)` was calling a 1-arg function
(TypeError) and a `name_aliases` NameError fired in the dengue path. Both fixed; 33/33 now.
The lesson is that "33/33" written in a progress doc is not evidence -- re-run the suite
before trusting any day's close-out.

## Carried blockers: both CLOSED (this file was stale on both)

  [x] dvc push -- DONE. `dvc status -c` reports "Cache and remote 'ebola' are in sync."
      The Azure remote (azure://ebola-dataset/) holds the influenza matrices. Task 8.1's
      provenance story is closed.

  [x] data/gadm/ un-versioned -- DONE, and by the RECOMMENDED route (option (b), the fetch
      script + SHA-256 manifest -- NOT `dvc add`, whose 461 MB upload would arguably breach
      GADM's no-commercial-redistribution licence).

      fetch_gadm.py  -- new. stdlib only (urllib + hashlib), no new deps.
        python fetch_gadm.py            download whatever is missing, then verify every file
        python fetch_gadm.py --verify   verify only, never touches the network
      Pins all 16 zips by SHA-256. Downloads to a .part file and renames on success, so a
      broken download never lands at the real path. Exits non-zero on ANY mismatch -- so it
      can gate build_datasets.py (Task 10.4), which is the whole point: a GADM re-cut would
      silently CHANGE THE GRAPHS, and a warning would be read past.

      The 16 = dengue's 12 (ARG BOL BRA COL DOM ECU JPN MEX NIC PAN PER TWN) + Ebola's 3
      (GIN LBR SLE) + USA (influenza's C only -- the shipped adjacency is reused verbatim).
      JPN serves double duty: the dengue graph AND influenza:japan's C.

  Note `data/gadm.dvc` also exists and is staged. It is now REDUNDANT with fetch_gadm.py and
  should be dropped rather than committed -- committing it re-introduces the licence problem
  the fetch script exists to avoid. (data/.gitignore already ignores /gadm.)

## Ebola source SHA-256: VERIFIED

  937136a42685ab596d7eb9fc8864ae41d2c07cb546714704d3d31f0e7ae52972
  data/Final datasets/data-ebola-public_best_yet.xlsx  (1,968,712 bytes)

Recomputed from the file on disk; MATCHES the value recorded in data_audit.md §3.1 exactly.
The audit's provenance line was correct -- confirmed, not corrected.

## THE ONE REAL BLOCKER: dengue node ids carry GADM's typos

Confirmed live in the code, not just re-read from Day 9's note:
  dengue_aliases.py:314   "nagasaki": "naoasaki"                 <- GADM 4.1's typo
  dengue_aliases.py:329   "huanuco|huanuco": "huanuco|huenuco"   <- GADM 4.1's typo

Dengue applies its alias map at LOAD time (§7.2.2 -- it has to, because the same map also
reunites the four torn series, which a join-time map could not do). The consequence is that
these two entries rewrite the DATA's correct spelling into GADM's misspelling, and the node
id inherits it. The bundle contains `japan|naoasaki` and `peru|huanuco|huenuco`.

WHY IT MUST BE FIXED BEFORE DAY 10, NOT AFTER: Task 10.4 FREEZES node_ids into the .npz
files, and node_ids are the schema's "single source of truth" for ordering. They also land
in the paper's node table. Fixing after packaging = rebuild + repackage + re-verify.

SOLUTION (the fix is a split, not an edit -- see the reasoning):
  Ebola already solved this and dengue should follow it (that is decision E5). Ebola runs TWO
  maps because they do TWO different jobs:
      EBOLA_NAME_ALIASES  (LOAD time)  repairs the DATA        -- torn series, bad suffixes
      EBOLA_GADM_FIX      (JOIN time)  absorbs GADM's DEFECTS  -- never renames our nodes
  Dengue has only the first, so GADM's defects leak into the node ids.

  So: add a DENGUE_GADM_FIX applied at JOIN time (keyed on the CORRECT node id -> the GADM
  key), move exactly these two entries into it, and DELETE them from DENGUE_NAME_ALIASES.
  Node ids become `japan|nagasaki` and `peru|huanuco|huanuco`; the join still finds GADM's
  misspelt polygon. No other alias moves -- every other dengue alias genuinely repairs the
  DATA (OpenDengue's editorial drift) and correctly belongs at load time.

  NOT the alternative "just flip the mapping direction": the load-time map is applied before
  the pivot, so flipping it would rename GADM's key rather than the data's, and the join
  would then miss. The two maps have to be separate because they run at different times.

  COST: 2 node ids change -> rebuild dengue -> re-run test_schema.py + dengue_load.py.
  validate_aliases() asserts every alias TARGET is a real GADM key, so the new join-time map
  is self-checking. Also fixes the stale note in data_audit.md §0.


# TASK 10.0 — DENGUE GADM-TYPO FIX: DONE (the blocker above is CLOSED)

Implemented exactly the split described above. Verified: the rebuilt graph is BIT-IDENTICAL --
7165 nodes, 20,468 edges, mask density 0.2175, 0 unmatched, 0 cross-border. ONLY the two node
ids changed, which is precisely the intended blast radius.

  japan|naoasaki        -> japan|nagasaki
  peru|huanuco|huenuco  -> peru|huanuco|huanuco

  dengue_aliases.py   DENGUE_GADM_FIX (new, JOIN-time). The 2 GADM typos REMOVED from
                      DENGUE_NAME_ALIASES. validate_aliases() rewritten to check the
                      EFFECTIVE join key (gadm_fix applied on top of the load-time target) --
                      it would otherwise now wrongly reject the very entries the split fixed,
                      since `japan|nagasaki` is deliberately NOT a GADM key.
  to_schema.py        build_dengue_adjacency(gadm_fix=...) and load_dengue(gadm_fix=...).
                      One line does the work: pos.get(gfix.get(key, key)).
  dengue_load.py      passes DENGUE_GADM_FIX; ASSERTS both typos are absent from node_ids and
                      both correct names present -- the fix cannot silently regress.
  test_schema.py      +1 = 34/34. test_dengue_gadm_typos_never_reach_node_ids asserts no
                      LOAD-time alias may ever TARGET a known GADM misspelling. That is the
                      regression path (someone adds a GADM quirk to the wrong map), so that
                      is what the test forbids -- not merely the current two values.

WHY A SECOND MAP AND NOT A CORRECTED ENTRY (recorded because the cheap fix is wrong):
flipping the direction of the two old entries does NOT work. The load map is applied BEFORE
the pivot, so a flipped entry renames GADM's key rather than the data's and the join then
misses the polygon. Nor can the load map simply move to join time wholesale -- J4's torn
series must be summed BEFORE the pivot. The separation of TIME is the fix, not the direction.

Documented: data_audit.md §0 (defect note -> RESOLVED), §1.6.1 (the two-map table), §1.9 J9
(rewritten with the corrected reasoning + measured blast radius).

  python test_schema.py                   34/34
  conda run -n ebola python dengue_load.py   rebuild PASSES, all asserts pass, .check() PASSED


# DAY 10 — INTEGRATION: DONE

One command regenerates everything, 83 leakage gates pass, all 5 bundles packaged.

  conda run -n ebola python build_datasets.py --check-deterministic

    verify   OpenDengue + ebola.xlsx + 6 influenza matrices + 16 GADM shapefiles, by SHA-256
    build    dengue 7165x1409 · flu japan 47x348 / us-regions 10x785 / us-states 49x360 ·
             ebola 61x52
    gate     83 leakage/invariant gates PASS, 0 fail; 3 negative controls FIRE
    write    data/processed/*.npz + config.json + env.txt
    determinism  two builds -> identical arrays for all 5 bundles

  python test_schema.py    34/34


## Task 10.1 — THE LEAKAGE SUITE, AND THE BUG ITS NEGATIVE CONTROLS FOUND

83 gates across 5 bundles (test_leakage.py): scaler provenance · no-future-leakage · mask
semantics · split integrity · rolling-origin ordering · cross-disease agnosticism. They GATE
THE WRITE -- build_datasets.py writes nothing if one fails, because a leaking bundle in
data/processed/ is a bundle someone trains on.

>> THE FIRST VERSION OF THE SUITE WAS DECORATION, AND I ONLY FOUND OUT BECAUSE I TRIED TO
>> BREAK IT. It recovered raw counts by INVERTING y WITH THE VERY SCALER IT WAS TESTING
>> (raw = expm1(y*std + mean)). That is CIRCULAR: refitting on that reconstruction returns the
>> same scaler whether or not it leaked, because the reconstruction already carries the leak.
>>
>> Proof: I planted Ebola's scaler fit on SUPPORT+QUERY -- precisely the §6.4 leak the entire
>> few-shot claim rests on NOT happening -- and THE SUITE PASSED IT. All 83 gates green, on a
>> bundle that leaks the held-out outbreak's magnitude into its own normalisation.
>>
>> FIX: the bundle now carries `raw` (unscaled incidence, which no scaler ever touched) and
>> every scaler gate tests against that. The 3 negative controls are now PERMANENT and run on
>> every build (test_leakage.self_test): dev scaler fit on all cells · ebola scaler fit on
>> support+query · future target planted in an input channel. ALL THREE FIRE.
>>
>> The suite passing now MEANS something, because we know it can fail. A gate that cannot fail
>> is not a gate. If one thing from Day 10 is worth carrying into Week 3's model tests, it is
>> this: write the test, then plant the bug and confirm the test screams.

Shipping `raw` had a second, independent justification I nearly missed: TASK 10.2's BACKTEST
REFITS THE SCALER AT EACH ORIGIN, and a refit needs the scaler's INPUT. Without raw counts in
the bundle, Week 5 could not refit per origin even in principle -- it would fall back on the
headline scaler, which has seen every origin's future. The refit discipline is only enforceable
if the thing it refits from is present.


## Task 10.2 — ROLLING-ORIGIN SCAFFOLD

Expanding-window, single-step, 5 origins. ONLY THE CUT POINTS ARE STORED (one int per group per
origin) -- materialising 5 [N,T] mask pairs for dengue would be ~250 MB of uint8 encoding what
one integer implies. rolling_origin_masks(dt, k) expands a cut on demand.

Grouped to MATCH the headline split or the origins would contradict it: per-country for dengue
(a global origin on the 1997-2024 union calendar sits BEFORE panama's first report and AFTER
nicaragua's last), one shared sequence for influenza.

EBOLA GETS NONE, DELIBERATELY. Rolling origin is meaningless for a held-out disease whose
protocol IS the 2-week support set; expanding the training window past support IS the §6.4 leak.
Absence is a decision here, not an omission -- and the suite asserts it.


## Task 10.4 — PACKAGING

One .npz per disease, NOT a pickled dict of DiseaseTensors. A pickle couples the artifact to the
exact class definition and library versions that wrote it (rename a field, bump numpy -> the file
is unreadable) and UNPICKLING EXECUTES CODE, so it is unsafe to hand a reviewer. The .npz is
plain arrays (X, y, raw, M, A_geo, C, split masks, scaler params) + a JSON meta blob.

Determinism VERIFIED, not assumed (--check-deterministic rebuilds and compares every array).

  data/processed/dengue.npz              15.1 MB
  data/processed/influenza_japan.npz      0.2 MB
  data/processed/influenza_us-regions.npz 0.1 MB
  data/processed/influenza_us-states.npz  0.2 MB
  data/processed/ebola.npz                0.0 MB
  data/processed/config.json  (every knob that moves a number)
  data/processed/env.txt      (pip freeze)


## Task 10.3 — DOCS

  data_audit.md §4    harmonisation + leakage: the 5-bundle table, the disease-agnosticism
                      guarantee, the 83 gates, the circular-gate bug, the rolling-origin
                      scaffold, reproducible regeneration, and what Week 3 must not forget.
  data_pipeline.md    NEW. How to regenerate from a clean clone, what the .npz contains and how
                      to load it, the schema contract, the two-map alias design (and why the
                      obvious one-line fix is wrong), provenance + the GADM licence reasoning,
                      the [CONFIRM-D*] register, and the file map.


## CARRIED INTO WEEK 3 (all recorded in meta, none of it enforceable from inside a bundle)

  1. The encoder MUST add I before any D^-1/2 normalisation, or influenza's 4 isolated nodes
     (Alaska, Hawaii, Hokkaido, Okinawa) divide by zero -> NaN.
     meta['requires_encoder_self_loops'].
  2. C must NOT enter the SHARED encoder. All three diseases now have a populated C and it is
     STILL not transfer-safe: they occupy disjoint geography, so a raw centroid IDENTIFIES the
     disease. covariates_transfer_safe=False; transfer_view() excludes it by construction.
  3. REFIT THE SCALER AT EVERY ROLLING ORIGIN. meta['scaler'] is the headline split's and has
     seen data past every origin. `raw` is shipped precisely so this is possible.


## EBOLA SOURCE SWITCHED TO THE PUBLISHED HDX FILE (provenance fix)

Our working copy was NOT the file HDX publishes. Same data, different bytes.

  HDX  data-ebola-public.xlsx            1,955,645 B  sha256 2d679a31...5f28f9f2
  OURS data-ebola-public_best_yet.xlsx   1,968,712 B  sha256 937136a4...0e7ae52972

The '_best_yet' copy had been OPENED AND RE-SAVED IN EXCEL somewhere upstream of us -- that
rewrites the xlsx zip container without touching a single cell. Verified both ways:

  contents  IDENTICAL -- same 2 sheets, same 58,635 x 7 frame, df.equals() == True
  bundle    BIT-IDENTICAL -- 61 nodes / 52 wk / 121 support / 1,546 query; X, y, raw, M, A_geo,
            C, both split masks and the scaler mean+std all array_equal

So NO NUMBER ANYWHERE CHANGES. But the hash we published was the hash of a re-saved copy, not of
the public file. A reviewer who downloaded the resource, hashed it, and compared against the audit
note would have got a MISMATCH and reasonably concluded we had built on something other than the
public data. That is a provenance failure even though the data were right.

  FIXED: data/Final datasets/data-ebola-public.xlsx is now the source of record.
         build_datasets.py RAW_SHA256, ebola_load.py and ebola_audit.py all repointed.
         Rebuilt end to end: 83/83 leakage gates, 34/34 schema tests, ebola_audit gate passes,
         determinism check passes. Every audit figure reproduces (64 district labels, 3,613
         National rows, 5/64 monotone series, 61 bundle nodes).

  HDX resource id 9b68ab69-e0b3-4ff5-b2d8-90bf446acd52, last_modified 2015-11-24.

LESSON worth carrying: checksum the file AS THE SOURCE PUBLISHES IT, not the copy that happens to
be sitting on disk. An .xlsx that has been opened once is a different file. Our checksum was
honest about the bytes we used and useless for the purpose a checksum exists to serve.

SIDE EFFECT -- the Ebola reproducibility hole is CLOSED. I had recorded the compilation as a
"live, manually-updated file with no version number" and was going to recommend depositing a
snapshot somewhere citable so external readers could obtain the exact bytes. Not needed: the
resource has not been touched upstream since Nov 2015 and sits on a permanent URL, so it is
de facto frozen. All four raw sources are now publicly downloadable and checksum-verified, and
data_pipeline.md now leads with the source URLs for each.


# ADVERSARIAL REVIEW — THREE BLOCKING DEFECTS, TWO FIXED

Three independent reviewers (reproducibility, epidemiology, methodology) were run over the
Phase-2 deliverables before the client document was written. They found defects that invalidate
released numbers. ALL OF THEM PASSED THE 83-GATE SUITE GREEN. That is the finding that matters
most: the gate count was not a measure of coverage, and a green suite was evidence only about
the properties it happened to test.

Every claim below was re-verified against the production files before being acted on.


## B1 — THE EBOLA CLIP FABRICATED 36% OF THE TARGET. FIXED.

  to_schema.cumulative_to_weekly_incidence ended in:   new = weekly_cum.diff().clip(lower=0)

  A cumulative count cannot fall. Where the report falls, the report is a single-week data-entry
  dropout -- NOT a "downward revision", which is what data_audit.md claimed (and cited the
  58-of-64 downward-step count as SUPPORT for, when that count was evidence of the bug).

  Western Area Urban, verbatim from the source:

      2015-01-24  cum=2,612   diff    +43
      2015-01-31  cum=  613   diff -1,999  -> clipped to 0, recorded as an OBSERVED ZERO
      2015-02-07  cum=  631   diff    +18
      2015-02-14  cum=2,741   diff +2,110  -> RELEASED AS 2,110 NEW CASES IN ONE WEEK

  The clip zeroed the fall, then released the RECOVERY -- a climb back to cases already counted
  -- as new incidence.

  Measured on the affected build:
      +8,612 fabricated cases across 34 districts   (33,164 released vs 24,552 real, +35%)
      the six largest cells in the entire target were all clip artifacts
      Western Area Urban carried 7,462 cases against the 3,017 it ever recorded  (x2.5)
      national peak of our target: 2015-02-14 @ 4,982/wk
      actual West Africa peak:     late 2014, in steep decline by Feb 2015
      -> epidemic peak displaced ~3 months and inflated several-fold

  FIX: difference the running maximum (weekly_cum.cummax().diff()). Mass-preserving by
  construction. Weeks whose report falls below the running max are now MASKED -- they are
  corrupt reports, not observations of zero, and we were scoring and normalising on them.

  AFTER:  target 24,552 == reference 24,552 (exact).  Peak 2014-10-25 @ 2,688/wk.
          observed cells 1,667 -> 1,299 (density .5255 -> .4095); query 1,546 -> 1,178.
          The 368 lost cells are the corrupt weeks. That is a real reduction and the honest count.


## B4 — DENGUE SPLIT FALLBACK VIOLATED THE INVARIANT THE AUDIT ASSERTED. FIXED.

  per_country_chronological_split re-cut any node with no obs in its country's train window, on
  its OWN timeline -- putting its train cells inside its neighbours' TEST period. A GNN
  aggregates over neighbours, so that pulls the test period into the training forward pass.

  Measured: 475 nodes re-cut; 2,420 train cells after their country's val_end; 792 (country,week)
  columns holding >1 phase; 685,214 observed cells (31%) in mixed-phase columns.
  data_audit.md asserted "at any week an entire country block is in a single phase". False.

  FIX: fallback removed. Every node takes its country's boundary, no exception. The 475 nodes
  (brazil 379, peru 49, colombia 31, argentina 15, taiwan 1) keep their place in the graph and
  are still scored, but have no train cell of their own -- so they take their COUNTRY's pooled
  train statistics as their scaler (fit_scalers_masked(groups=...)). Country train cells are
  visible at train time, so no leakage. Recorded in meta["split"]["nodes_without_train"].
  No nodes and no cells dropped. train 1,249,109 / val 524,972 / test 422,180.


## THE GATES THAT WOULD HAVE CAUGHT THEM (written BEFORE the fixes, confirmed RED first)

  gate_mass_conservation  -- a weekly series derived from a cumulative one must sum to no more
      than max(C) - C_first, where the reference comes from cumulative_reference_mass(), which
      reads the SOURCE COLUMN and shares nothing with the transform but the weekly resample.
      The old suite compared the transform only against its own output.
  gate_phase_purity       -- no (country, week) column may hold >1 split phase.

  Both went RED on the unfixed build, with the numbers above. Then the fixes; then green.
  Two NEW negative controls (5 total, all firing): fabricated incidence past the reference, and
  a train cell planted in a test column. 85/85 gates, 5/5 controls.

  I also broke two gates while fixing the code: scaler-provenance failed because the gate did not
  know about the country-pooled fallback I had just added. Fixed by teaching the gate the
  grouping, NOT by loosening it -- the 1000x poisoning probe still holds.


## LEFT FOR THE CLIENT (in client_decisions.md Part A) -- NOT decided unilaterally

  A2  DENGUE CASE DEFINITIONS CHANGE MID-SERIES, ACROSS THE SPLIT.  ** BLOCKS MODELLING **
      peru      probable+confirmed 2000-23 -> total 2023-24   mean 24.5 -> 5,398.6   (x220)
      dominican probable 2006-13           -> total 2013-24    4.0 ->   191.3   (x48)
      mexico    confirmed 2002-13          -> total 2013-24   17.5 ->   231.3   (x13)
      bolivia   suspected 2004-09          -> total 2013-24   13.0 ->   141.4   (x11)
      panama    confirmed 2010-12          -> total 2013-24    5.5 ->    39.1   (x7)
      A per-node affine scaler CANNOT absorb a mid-series step change in what is being counted.
      Peru trains on probable+confirmed and is TESTED ON TOTAL. 42% of a 12-country macro.
      The existing case_def_mixed=0 check only asks whether a CELL mixes definitions. It never
      asks whether a NODE's definition is constant over time. It is not.

  A3  THE EBOLA SUPPORT SET IS NOT CALENDAR-CAUSAL.  ** BLOCKS MODELLING **
      support = first 2 observed weeks PER DISTRICT; districts enter at very different dates.
      support cells span calendar weeks 1-45 (2014-04-12 .. 2015-02-14).
      1,004 of 1,178 query cells (85%) are EARLIER in calendar time than the last support cell.
      The pooled scaler normalising a week-5 query cell is a function of week-45 peak magnitudes.
      The leakage suite STRUCTURALLY CANNOT catch this: _fit_mask() DEFINES support_mask as the
      legitimate fit set, so the sharpest gate is conditional on the assumption under attack.
      This is definitional -- what "few-shot on an emerging outbreak" means -- not a coding bug.

  A4  EBOLA GAP-LUMPING (not fixed; disclosed).
      Differencing recovers HOW MANY cases accrued between reports, not WHEN. 7% of intervals
      between observed weeks exceed 1 week (max gap 24w), so ~1 increment in 14 stamps several
      weeks of cases on a single week. Montserrado: 1,428 on 2014-10-25.
      No cases invented (the mass gate holds exactly) and the curve is correctly SHAPED, but
      peaks are inflated and the weeks either side flattened. Any per-week magnitude claim on
      Ebola inherits this.


## OTHER DOC FALSEHOODS CORRECTED (data_audit.md sec 5.3 now logs these)

  - zero-variance guard fires on 400 nodes, not "thirty". colombia 222, argentina 113, japan 29,
    brazil 18, taiwan 9, peru 6, mexico 3. AND 76 of them have real variance in their TEST cells
    -> the guard leaves those UN-NORMALISED (raw log1p scale, while every neighbour is
    standardised). That is a defect. Remedy = the same country-pooled fallback as B4. NOT applied.
  - NO SCORING CODE EXISTS. The country-macro metric that client decision A1 asks approval for is
    unimplemented. It was described in the audit as though it were a property of the release.
  - ebola: TWO districts have no query week (guinea|boke, guinea|lelouma), not one.
  - the dengue block-diagonal graph was justified by "its countries are not adjacent". FALSE.
    Brazil borders Colombia, Peru, Bolivia, Argentina. The graph severs the Leticia-Tabatinga and
    Triple Frontier dengue corridors. Defensible as a simplification; not as a geographic fact.
  - the 'new cases' series was rejected as "sparser". It carries 9,494 rows vs cases' 9,978 = 95%
    coverage. It is a DIRECTLY REPORTED incidence series and would have caught B1 on day one.
    Should be adopted as a cross-check.
  - no environment.yml exists; data_pipeline.md's first command fails.
  - ebola has no validation set (query IS test), so any hyperparameter choice is made on test.


# SESSION 3 CLOSE — A2/A3 resolved, P3 fixed, docs rewritten, env.yml added

  P3  zero-variance guard: FIXED. Constant-in-training nodes now take the country-pooled train
      scaler (same mechanism as nodes_without_train). Rebuild: guard fires on 0 nodes (was 400),
      0 un-normalised with test variance (was 76). Leakage-free (pool is train-only).

  A2  dengue case definitions: RESOLVED = do-nothing + disclose. The x220/x13/x11 were RAW-CSV
      group-by artifacts. Measured per node on EXTRACTED data:
        peru   x220 -> NONE (single-def; the 'total' era was a 52-row sliver never selected)
        dom.r. x48  -> NONE (single-def)
        mexico x13  -> 2.8x across the split (part def-change, part real dengue trend)
        bolivia x11 -> 1.6x, and the change sits INSIDE training
        panama x7   -> 1 node
      Real exposure: 41/7165 nodes. Kept as built, disclosed. Verified peru|piura|piura (peak
      4720) is 100% 'probable and confirmed' across 2000-2023.

  A3  ebola few-shot causality: RESOLVED = calendar-prefix support, cutoff 2014-05-24 (week 8).
      support = every observed cell dated <= cutoff. Build: 27 support / 1272 query, 9 districts
      supported, 52 zero-shot, all 61 scored, 0 without query. CAUSAL: last support 2014-05-24
      <= first query 2014-05-31. New gate `calendar-causal` + negative control (plant an acausal
      support cell). Replaces the per-district scheme (was 85% of query preceding last support).

  SUITE: 83 -> 86 gates, 5 -> 6 negative controls. Rebuilt, all green, deterministic (2 builds
      bit-identical). ebola mass still 24,552 == reference.

  INFLUENZA PROVENANCE (from the Q2 adversarial review, self-verified against Delphi Epidata):
      region785.txt IS CDC ILINet num_ili, HHS 1-10, epiweeks 200240-201741: 759/785 rows exact.
      state360.txt IS ILINet by state, 201040-201734: 16,890/17,327 cells (97.5%) exact.
      -> the '39-week bug' was OURS, not ColaGNN's; anchor 2002-10-05 now PROVEN not argued.
      Caveats recorded: col 30 = NY-excluding-NYC; Florida absent; JAPAN calendar still weakest
      (seasonal phase only, no row-level pin) -> pin against NIID/IDWR before publishing.

  DENGUE JOIN (from the Q1 adversarial review, self-verified): DON'T change it. FAO_GAUL_code is
      NOT injective (7443->7027, 416 nodes would silently MERGE). Name join lands on the exact
      IBGE polygon for all 5517 Brazil nodes (0 disagreements). 234 aliases (not 12 - my miscount).
      Use codes as an ORACLE gate, not a join key. D17 rationale corrected in schema_spec.

  env.yml: WRITTEN (python 3.10, numpy 2.2, pandas 2.3, geopandas 1.1, libpysal 4.13, shapely 2.1).

  A4  ebola gap-lumping: RESOLVED = disclose and proceed (client decision). No code change; the shape
      is correct and both remedies (redistribute / contiguous-runs-only) cost more than the distortion.
      Recorded in audit 3.4.3, schema_spec 5, client doc A4.

  DOCS REWRITTEN to measured numbers: client_decisions.md (A2/A3/A4 resolved), data_audit.md (1.7,
      1.10, 2.3 ILINet, 3.6 causal, 4.3 86 gates, 5.3/5.4), schema_spec.md (few-shot, D17, 4/9).

  DOC CONSISTENCY PASS after A3/P3 (stale numbers that flipped):
    - "two districts never scored (boke, lelouma)" -> ZERO under calendar-prefix; all 61 scored, 52
      are zero-shot. Fixed in client D4, audit 3.10 / 3.2 / 5.3.
    - zero-variance "76 mis-fire, not applied" -> FIXED (client D1, audit 1.7).
    - clip "8,612 / 36% / 1,546->1,178 eval cells" -> 8,786 / 35.8% / observed 1,667->1,299 (query is
      an A3 matter now). Fixed in client C1.
    - "83->85 gates" -> 86 gates / 6 controls everywhere (client 'what we got wrong', schema 9,
      remediation R3).

  STILL UNCOMMITTED. That is the one remaining action.


# STATUS SNAPSHOT -- SUPERSEDED (see SESSION 3 CLOSE at the top of this file)
# Left as a dated record of a mid-session state; every alarming note below was RESOLVED later.

CODE:   [SUPERSEDED] Now has uncommitted changes on top of a46142e: A3 calendar-prefix support,
        the calendar-causal gate, and the P3 zero-variance fallback. Last full run 86/86 gates,
        6/6 controls, ebola mass 24,552 == reference.

ARTIFACTS: [RESOLVED -- no longer stale] At this snapshot the released .npz predated the fixes
        (mass 33,338, the clip artifact still present) because verification ran build_all() in
        memory and nothing had called write(). FIXED: build_datasets.py was run; the released
        data/processed/*.npz now carry mass 24,552, 1,299 observed cells, calendar-prefix
        support -- verified on disk. This note is kept only to record that the gap existed and
        was closed.

DOCS:   [SUPERSEDED] All five .md docs and the .docx have since been rewritten to the measured
        numbers. Still UNCOMMITTED -- the one remaining action.

DVC:    data/gadm.dvc tracked deliberately. Settled, not an issue.


# CLIENT REVIEW RETURNED (client_decisions.md) -- ALL RESOLVED LATER (see SESSION 3 CLOSE)

PART A -- [SUPERSEDED] Was "all four unanswered, A2/A3 block modelling"; now all four resolved
          (A1 approved, A2 disclose, A3 calendar-prefix, A4 disclose). Nothing blocks modelling.

PART B -- B1,B3,B4,B5,B6 confirmed.
  B2  "Widen to three or four" -> *** DEFERRED by agreement. Stays at 2 for this release. ***
      few_shot_support_weeks is a param -> revisiting it later is a rebuild, not a code change.
      WHEN REVISITED, DECIDE WITH A3: more support weeks per district = support reaches FURTHER
      into calendar time = the acausality gets WORSE, not better.
      Also: 2 districts already yield no query cell at n=2; widening increases that.

PART C -- C1,C3,C4,C5,C6,C8,C9 approved.
  C2  *** NOT approved: "Need further elaboration ... short explanation, benefits, losses" ***
      Draft answer written in remediation_plan.md B-C2. Not sent yet.
  C7  *** RESOLVED: no change. Dropping Western Area is correct. ***
      Client had asked for a single Western Area node, then withdrew on learning that Western
      Area Urban + Western Area Rural are ALREADY separate nodes and the parent overlaps both
      in time -> keeping it would triple-count. Release stands as built: parent dropped, both
      children retained, 61 nodes. Also preserves the Urban/Rural split, which matters --
      Western Area Urban (Freetown) is the largest district of the epidemic and behaves
      nothing like the rural remainder.


# STILL OPEN (housekeeping, not blocking) -- updated at SESSION 3 CLOSE

  [x] environment.yml -- DONE (python 3.10, numpy 2.2, pandas 2.3, geopandas 1.1, libpysal 4.13).
  [x] 76 mis-firing zero-variance nodes -- FIXED (country-pooled fallback; 0 now un-normalised).
  [x] data_audit.md "33,164 released" -- FIXED to 33,338; fabrication 8,786 (+35.8%). Logged in
      the audit 5.3 corrections table.
  [ ] COMMIT -- nothing is in git yet. The one action that matters. (data/processed/ stays a
      derived artifact: regenerable from build_datasets.py; do not track the .npz.)
  [ ] ebola has no validation set (query IS test) -> hyperparameters get chosen on test. (P6)
  [ ] adopt 'new cases' as a cross-check series (95% coverage; would have caught the clip). (P7)
  [ ] pin Japan's influenza calendar against NIID/IDWR (seasonal-phase only); add IBGE code
      oracle gate; label state360 col 30 as NY-excluding-NYC. (P8, before publishing numbers)


Ebola dataset url
Sub-national time series data on Ebola cases and deaths in Guinea, Liberia, Sierra Leone, Nigeria, Senegal and Mali since March 2014
https://data.humdata.org/dataset/rowca-ebola-cases


























=== E1 COVERAGE ===
nodes=61 weeks=52 span=2014-04-05..2015-03-28
obs weeks/district mean=27.3 median=29 min=1 max=41
mask density=0.526 support=121 query=1546 >=20wk:54
=== E2 INTEGRITY ===
y negatives=842 totalmass=633 (ref 24552)
=== E3 CURVE ===
peak 2014-11-08 = 45
=== E4 A3 LEAKAGE ===
support 2014-04-12..2015-02-14  query 2014-04-26..2015-03-28
query cells earlier than last support: 85.0%
=== E5 A4 GAPS ===
intervals mean=1.11 >1wk=2.4% max=16
  spike sierra leone|western area urban 2015-02-14=6
  spike sierra leone|western area urban 2015-03-07=6
  spike liberia|montserrado 2014-10-25=6
  spike sierra leone|port loko 2015-02-14=6
  spike sierra leone|western area urban 2015-01-03=5
=== E6 FEATURES ===
feature_names=['incidence_norm', 'sin_doy', 'cos_doy', 'obs_mask', 'deaths_norm']  core_idx=[0, 1, 2, 3]
covariates_transfer_safe=False
corr(incidence_norm, deaths_norm) on observed = 0.788  (ebola-only feature)
=== E7 PATTERNS ===
lag-1 autocorr: mean=0.308 (n=57 nodes with variance)
spatial coupling corr(node, neighbour-mean incidence)=0.546
graph: edges=146 mean_deg=4.79 cross_border=21
incidence>0: median=1.4 p95=3 max=6 (heavy-tailed)
# Dengue Spatial-Level Resolution — Implementation Guide
### Phase 2 · patch to `to_schema.py` (`dengue_country_coverage`, `load_dengue`, `build_dengue_adjacency`)

**Scope.** Replace the single-global-level dengue path with a *per-country* level resolver that (a) never silently falls back to Admin0, and (b) prefers the coarsest adequate subnational level so no single country dominates the node set. This is a data-layer change only — `DiseaseTensors` and the encoder are untouched. Implement against the **full OpenDengue CSV**, not the truncated `.xls` (see §5.3).

---

## 1. The rule (implement exactly this)

For each candidate country (the `DENGUE_ENDEMIC` pool ∩ the file, weekly rows only):

1. **Level domain is `{Admin1, Admin2}`.** Admin0 is *not* a node level. A national aggregate is a block-diagonal singleton with no intra-country neighbours and therefore zero spatial-coupling signal — the exact thing the graph exists to exploit.
2. **Prefer the coarsest adequate level.** Choose **Admin1** if it yields ≥ `min_nodes_per_country` nodes each clearing `min_weeks` observed weeks. Only if Admin1 is inadequate, fall back to **Admin2**. (This is the inversion of "finest available": finest is what makes Brazil ~5,000 municipalities and ~84% of all nodes.)
3. **Exclude, with a logged reason, any country that clears neither level.** Two distinct reasons: `admin0_only` (no subnational rows at all — the 14 fallback countries) and `insufficient_coverage` (has subnational rows but too few usable nodes).
4. **One level per country, fixed across all time.** Never mix levels within a country (contiguity and the node set must be time-invariant).
5. **Record country + chosen level per node** so evaluation can macro-average over *countries*, not just nodes.

Net: no silent fallback anywhere; Admin1-preferred; coverage-driven; Admin0-only countries dropped explicitly.

---

## 2. Implementation steps

### 2.1 Kill the silent Admin0 fallback (do this first — it's a live bug once the filter loosens)
The current resolver is a `dict.get` whose default is the country column:
```python
node_col = {"Admin1": cols["adm1"], "Admin2": cols["adm2"]}.get(level, cols["adm0"])
```
Once you stop filtering to a single global `S_res`, any unexpected level token resolves to `adm_0_name` and collapses every node to national level. Replace with an explicit map that raises:
```python
_LEVEL_TO_COL = {"Admin1": cols["adm1"], "Admin2": cols["adm2"]}
def _node_col_for(level, cols):
    try:
        return {"Admin1": cols["adm1"], "Admin2": cols["adm2"]}[level]
    except KeyError:
        raise ValueError(f"Unsupported dengue node level {level!r}; expected Admin1 or Admin2.")
```
Admin0 is intentionally absent — asking for it is an error, not a fallback.

### 2.2 New function: `resolve_country_levels()` — the coverage-driven selector
This is the heart of the change. It runs the coverage logic at *both* subnational levels and returns the decision plus the audit trail.

```python
def resolve_country_levels(csv_path, countries=None, t_res_filter="Week",
                           min_weeks=52, min_nodes_per_country=3,
                           cols=DENGUE_COLS):
    """Return (levels, report) where
       levels : {canonical_country -> 'Admin1'|'Admin2'}   (kept countries only)
       report : per-country dict with n_usable_admin1, n_usable_admin2,
                chosen level or exclusion reason ('admin0_only'|'insufficient_coverage').
    Rule: prefer Admin1 if it clears the thresholds; else Admin2; else exclude."""
```
Logic per country (weekly rows only):
- Count **usable Admin1 nodes**: distinct `adm_1_name` among `S_res == "Admin1"` rows whose observed-week count ≥ `min_weeks`.
- Count **usable Admin2 nodes**: same among `S_res == "Admin2"` rows.
- Decide: `Admin1` if `n_usable_admin1 >= min_nodes_per_country`; elif `n_usable_admin2 >= min_nodes_per_country` → `Admin2`; else exclude. Reason is `admin0_only` when the country has zero Admin1 *and* zero Admin2 rows, otherwise `insufficient_coverage`.

Keep the whole `report` — it *is* the dengue portion of the data-audit note.

### 2.3 Update `dengue_country_coverage()` to report both levels
Today it filters to one `level` and reports one number. Change it to call the same per-country machinery and emit, per country: `n_nodes_admin1`, `n_usable_admin1`, `n_nodes_admin2`, `n_usable_admin2`, `max_weeks`, `endemic`, and the `resolve_country_levels` decision. This lets you eyeball the Admin1-vs-Admin2 trade-off per country before locking, and it's what you paste into the audit.

### 2.4 Update `load_dengue()` — per-country filter, guards, metadata
Replace the single global `S_res == level` filter with a per-country selection driven by `resolve_country_levels()`:

1. Call `resolve_country_levels(...)` → `levels`, `report`.
2. For each kept country, keep only rows where `S_res == levels[country]`, and set that country's node name from that level's column (`adm_1_name` or `adm_2_name`). Concatenate across countries.
3. Build `_node = country|adm` as now. Node ids are unique because country prefixes them; also stamp each node's level (from `levels`) for meta.
4. **Double-count guard.** Before the pivot's `aggfunc="sum"`, assert exactly one row per `(node, period)` at the chosen level — or, if `case_definition_standardised` produces several (§5.1), collapse deliberately, not by silent sum. A parent+child sum cannot occur because you kept only one level per country, but same-unit duplicate rows still can.
5. Run the existing `min_weeks` / `min_nodes_per_country` prune (it now operates on an already level-correct frame) and `_finalise`.
6. Populate meta (§3), including `report` and the excluded-country lists with reasons.

Signature change: replace `level="Admin1"` with `level="auto"` (the per-country resolver). Preserve `level="Admin1"`/`"Admin2"` as explicit overrides that force a single level for *all* countries, for ablations — but make `"auto"` the default.

### 2.5 Update `build_dengue_adjacency()` — per-country level-aware graph
Currently it takes a single global `name_field`, so it can't build a mixed-level graph. Make it level-aware:
```python
def build_dengue_adjacency(shapefiles_by_country, node_ids, levels_by_country,
                           name_field_by_level, knn_fallback=4, name_aliases=None):
    # for each country: pick shapefile + name_field from its chosen level,
    # build that block with build_adjacency(...), assemble block-diagonal.
```
- `shapefiles_by_country`: `{country: {"Admin1": path, "Admin2": path}}` (or a single path if only one level is used for that country).
- `name_field_by_level`: `{"Admin1": adm1_field, "Admin2": adm2_field}` in the shapefile schema.
- Each country's block uses *its* level's shapefile and name column; blocks are assembled block-diagonally exactly as now (no cross-border edges). The loud-failure name-join loop is unchanged, but now maintain the alias map **per level** (Admin1 province names and Admin2 district names live in different shapefile layers).

---

## 3. Metadata contract (what `meta` must carry after this change)
- `node_levels`: `{node_id -> 'Admin1'|'Admin2'}` — required for country/level-aware evaluation and adjacency.
- `country_levels`: `{country -> chosen level}`.
- `countries`: kept countries (as now).
- `countries_excluded`: `{country -> 'admin0_only'|'insufficient_coverage'}` — replaces/extends `countries_dropped_low_coverage`.
- `level_report`: the full per-country both-levels coverage table from `resolve_country_levels`.
- `node_country`: `{node_id -> country}` — enables country-macro-averaged metrics.
- `graph_is_block_diagonal`: `True` (unchanged).

---

## 4. Validation checks (add to the leakage/schema test suite)
- **No Admin0 nodes:** assert every value in `node_levels` ∈ `{Admin1, Admin2}`.
- **One level per country:** assert each country maps to a single level across all its nodes.
- **No parent/child co-presence:** assert no kept country has both an Admin1 node and an Admin2 node.
- **No dominance:** assert `max(country node share) ≤ cap` (set `cap` after inspecting the full CSV; flag, don't hard-fail, if exceeded — see §6).
- **Adjacency alignment:** assert `A_geo` is `[N,N]`, block-diagonal by country, and every node matched a polygon (the build already raises on unmatched).
- **Determinism:** node ordering fixed and reproducible from `resolve_country_levels` output.

---

## 5. Gotchas that will bite if unhandled

### 5.1 `case_definition_standardised` can create duplicate unit-period rows
The extract carries a case-definition column (e.g. *Suspected*). If a single unit-period appears under more than one definition, the pivot's `aggfunc="sum"` double-counts. Decide the aggregation explicitly — pick one standardised definition consistently, or a documented precedence — and assert one row per `(node, period)` afterwards. This is the dengue analogue of the Ebola indicator choice.

### 5.2 The name-join is per-level, not global
Admin1 joins province names to the Admin1 shapefile layer; Admin2 joins district names to the Admin2 layer. Diacritics and "State of X"/parenthetical forms differ between OpenDengue's GAUL/Natural-Earth lineage and GADM. Maintain **two** alias maps (one per level) and keep the halt-on-unmatched behaviour.

### 5.3 Use the full CSV — the `.xls` is truncated and audit-invalidating
`OpenDengue_Best_Spacial.xls` stops at exactly 65,535 rows (legacy Excel cap) and the file is alphabetical, so Brazil, Colombia, Mexico, Thailand, Vietnam, the Philippines are cut off. Run `resolve_country_levels` on the full CSV from the Figshare DOI / GitHub release; any level decision on the `.xls` is meaningless.

### 5.4 Weekly is the minority resolution
In OpenDengue, weekly ≈ 12k rows vs ≈ 53k monthly. After the weekly restriction (monthly *not* interpolated, per the manuscript), the usable subnational-weekly set is modest — which is exactly why keeping every weekly-capable country via per-country selection, rather than dropping half to a uniform level, matters.

---

## 6. Data-audit decisions to record (flag these in `data_audit.md`)

Record each of the following as an explicit, dated decision with its rationale, so the choice is defensible to a reviewer rather than buried in code:

- **[AUDIT-DENG-1] Node level domain = {Admin1, Admin2}; Admin0 excluded.** National aggregates are singleton nodes with no spatial coupling. **14 Admin0-only countries are excluded** from the development graph with reason `admin0_only`; list them. (They may be cited for national burden context in prose but do not enter the model.)
- **[AUDIT-DENG-2] Coarsest-adequate level, Admin1-preferred (not finest).** Motivation: at finest-available, Brazil's ~5,000 Admin2 municipalities are ~84% of all nodes, making the "general dengue representation" a Brazilian-municipality model and the transfer *source* mono-geography. Admin1-preference puts Brazil at 27 nodes and equalises granularity across countries. Record the resulting per-country node counts and the max single-country node share.
- **[AUDIT-DENG-3] Coverage thresholds.** `min_weeks` (default 52 = one seasonal cycle) and `min_nodes_per_country` (default 3). Record the values used and the kept/excluded lists with reasons (`admin0_only` vs `insufficient_coverage`).
- **[AUDIT-DENG-4] Evaluation is macro-averaged over countries, not only nodes.** Even at Admin1, node-macro metrics can tilt toward larger countries; report a country-macro-averaged variant as the headline so no single geography dominates the score. (Enabled by `node_country` in meta; a country-balanced sampler is left available for Week 3 but not applied at the data layer.)
- **[AUDIT-DENG-5] Case-definition aggregation.** State which `case_definition_standardised` policy was used and confirm one row per unit-period post-aggregation (no silent summation across definitions).
- **[AUDIT-DENG-6] Source = full OpenDengue CSV (Figshare DOI / GitHub release), weekly portion only.** Note explicitly that the truncated `.xls` was **not** used for the locked audit, and that monthly data is excluded (not interpolated).

---

## Definition of done (this component)
- [ ] `_node_col_for` raises on non-{Admin1,Admin2}; no `.get(..., adm0)` anywhere.
- [ ] `resolve_country_levels()` returns per-country level + full report; Admin1-preferred, Admin0-only excluded.
- [ ] `dengue_country_coverage()` reports both levels per country.
- [ ] `load_dengue(level="auto")` selects per country, guards duplicates, stamps `node_levels`/`node_country`/`countries_excluded`/`level_report`.
- [ ] `build_dengue_adjacency()` is level-aware (per-country shapefile + name_field); two alias maps maintained.
- [ ] Validation checks pass (no Admin0 nodes; one level per country; no parent/child co-presence; dominance flagged).
- [ ] All six **[AUDIT-DENG-*]** decisions written into `data_audit.md`.

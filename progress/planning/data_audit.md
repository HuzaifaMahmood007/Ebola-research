# Data Audit Note

**Emerging Disease Forecasting Framework — Phase 2**

This note documents the three disease datasets that constitute the harmonised input to the
forecasting framework. For each disease it records provenance and checksums, the spatial units
and administrative level, the temporal span and cadence, coverage and missingness, the
construction of the geographic graph, and every point at which the data were altered rather
than merely reformatted. A final section documents the harmonisation guarantees and the
leakage controls that gate the released datasets.

The note is written to be read on its own. A reader should not need to consult the source code
to discover a decision that changed the data, nor to learn why a number in the results table is
what it is.

**Contents**

1. Dengue (OpenDengue)
2. Influenza (ColaGNN benchmarks)
3. Ebola (OCHA ROWCA compilation)
4. Harmonisation and leakage
5. Summary of decisions

---

## 1. Dengue — OpenDengue

### 1.1 Provenance

| | |
|---|---|
| Source | OpenDengue, "best spatial resolution" extract |
| Citation | Clarke J et al., *Scientific Data* 2024;11(1):296 |
| Version | V1.3, Figshare DOI [10.6084/m9.figshare.24259573](https://doi.org/10.6084/m9.figshare.24259573) |
| Mirror | `OpenDengue/master-repo`, `data/releases/V1.3`, pinned to the same release |
| `Spatial_extract_V1_3.zip` | MD5 `6b2eba520c6b0831d3900138cb780118`; SHA-256 `69f6f798bc732cf453e405725a1dab2faf595c37bc7b688e783f65ad2dd1e780` |
| `OpenDengue_Best_Spacial.csv` | SHA-256 `0f59280ed6795d8a0f7e87f900397db46cab6f2740106718c25c05699b9ababc` |

The complete CSV extract is the source of record; the truncated spreadsheet distribution was not
used. Only weekly records are retained. The legacy monthly history is **not** interpolated to a
weekly cadence: doing so would fabricate observations and inflate the effective sample size, and
the resulting out-of-sample metrics would overstate performance. Because the seasonality channels
are derived from the calendar date rather than from the reporting cadence, discarding the monthly
era costs nothing in seasonal phase.

### 1.2 Spatial units

The admissible node levels are Admin-1 and Admin-2. **A national aggregate is never a node.** A
country represented by a single national series is a graph singleton with no intra-country
neighbours and therefore carries no spatial-coupling signal, which is the quantity the model
exists to exploit. Fifty-seven otherwise-endemic countries are excluded on this basis: their
weekly records exist only at national level.

Level is resolved **per country, at the finest level that clears the coverage thresholds**: Admin-2
if at least `min_nodes_per_country` of its Admin-2 units each clear `min_weeks` of observed data,
otherwise Admin-1, otherwise the country is excluded. The finest adequate level is preferred to
the coarsest adequate one because the latter would select Colombia's sparse 2001–2003 Admin-1
fragment in preference to its much richer 2006–2022 Admin-2 series, while gaining nothing for
Brazil, which has no Admin-1 weekly data at all.

Selecting one level per country cannot double-count a parent province against its own child
districts: the extract was verified to resolve each location-period to exactly one spatial level,
so parent and child never coexist for the same country and period.

**Node identity.** An Admin-1 node is keyed `country|adm1`. An Admin-2 node is keyed
`country|adm1|adm2` — that is, it carries its parent province. This is a correctness requirement
rather than a stylistic one, for reasons given in Section 1.5.

| Level | Countries (final node counts) |
|---|---|
| Admin-2 | Brazil (5,517), Colombia (1,066), Argentina (275), Peru (113), Taiwan (22, aggregated — Section 1.9) |
| Admin-1 | Japan (47), Dominican Republic (32), Mexico (32), Ecuador (24), Nicaragua (17), Panama (11), Bolivia (9) |

The dengue bundle comprises **12 countries and 7,165 nodes** on a union calendar of **1,409 weekly
steps** (1998–2024, MMWR week-ending-Saturday). Brazil contributes 77.0 per cent of all nodes; this
imbalance is addressed by the evaluation protocol (Section 1.7) rather than by discarding data.

The node count changed twice during construction, in both cases for cause: from 7,030 to 7,443 when
the Admin-2 key collision was corrected (Section 1.5), and from 7,443 to 7,165 when the geometry
reconciliation merged split series, folded post-vintage municipalities into their parents,
aggregated Taiwan, and dropped one unmappable node (Section 1.9). Each movement is itemised.

### 1.3 Coverage thresholds

A node must have at least **52 observed weeks** — one full seasonal cycle, below which the
seasonality channels cannot be estimated meaningfully — and a country must contribute at least
**three** such nodes, since a one- or two-node country supplies no intra-country spatial structure.

Pruning is applied at country level only: every node of a retained country is kept, including
those below 52 weeks. Node dominance is handled by the evaluation protocol, not by deletion.

The country set is therefore **data-driven rather than hand-selected**. The WHO/CDC endemic list
serves only as a candidate pool; membership of the final set is determined by the thresholds. The
retained and excluded lists, with per-country coverage, are recorded in the dataset metadata.

### 1.4 Temporal harmonisation

All records are binned to the MMWR epidemiological week (Sunday to Saturday). Two properties on
which the loader depends were verified against the actual file rather than assumed
([test_dengue_7_1.py](test_dengue_7_1.py)):

- **Case counts are per-period incidence, not cumulative**, so no differencing is applied (in
  contrast to Ebola). Of 5,938 non-constant weekly series, none is monotonically non-decreasing.
- **Week binning is correct.** Every weekly record begins on a Sunday, spans exactly seven days,
  and falls within its own epidemiological week.

Suppressed and missing case values are encoded as *missing* in the observation mask, never as a
count of zero. The distinction matters: a zero asserts that no cases occurred, whereas a mask
records that nothing is known.

Dates in this file are ISO-formatted. An early implementation parsed them day-first, which
silently coerced approximately 19 per cent of rows to invalid dates and thereby excluded Argentina
and Ecuador from the dataset entirely. Day-first parsing was removed.

**Mask density: 2,196,261 observed cells of 10,095,485 (21.75 per cent)**, over 7,165 nodes and
1,409 weeks.

### 1.5 The Admin-2 key collision

This subsection records a defect that materially corrupted an earlier version of the dataset, and
the correction of an incorrect finding that the defect had produced.

**The defect.** Nodes were originally keyed on country and unit name alone. Admin-2 unit names are
**not unique within a country**: Brazil contains 239 municipality names that are each shared by two
to five different states (*Bom Jesus*, for instance, names five distinct municipalities up to
1,500 km apart). Under a leaf-only key, 64,104 Brazilian (name, week) cells collided, and the
aggregation step therefore **summed the case counts of geographically distinct municipalities into
a single node**. Colombia (82), Argentina (40) and Taiwan (13) collided identically, for 413
collisions in total.

The same defect corrupted the coverage gate. Because the level resolver also grouped by leaf name,
collided municipalities pooled their observed weeks, overstating coverage and understating node
counts — enough, in principle, to admit or exclude a country wrongly.

**The correction.** An Admin-2 node key now carries its parent province.

| | Before | After |
|---|---|---|
| Nodes | 7,030 | 7,443 (an increase of exactly 413, the collision count) |
| Duplicate (node, week) cells | 74,952 | 0 |

**A previously-reported finding is hereby retracted.** An earlier diagnostic reported 74,952
within-week summed rows and concluded that these were legitimate sub-weekly reports of a single
case definition collapsed into one epidemiological week. That conclusion was wrong. With the
parent province in the key, the number of duplicate (node, week) cells is **zero**, which
demonstrates that every one of those 74,952 rows was a cross-municipality name collision and none
was a sub-weekly report. The original diagnostic could not distinguish the two cases because it
counted duplicates under the broken key.

**What survives.** No (node, week) cell mixes case definitions, so the aggregation never
double-counts suspected against confirmed against probable cases. That conclusion never depended
on the node key and is unaffected.

**A provenance caveat.** The case definition is nonetheless heterogeneous *across* the corpus —
predominantly *probable*, then *total*, with confirmed and suspected variants. Per-node scaling
absorbs the resulting differences in magnitude, but the mixture is a genuine property of the source
and is disclosed here rather than concealed by the normalisation.

The corrected behaviour is pinned by a regression test
(`test_dengue_admin2_same_name_in_two_states_stays_two_nodes`).

### 1.6 Geographic graph

**Shapefile provenance: GADM 4.1**, used throughout the project for both dengue and Ebola geometry
and for all static covariates. The alternative — sourcing dengue's geometry from the lineage
OpenDengue itself geomatched against — would have reduced the name-matching burden but required
two shapefile provenances, of different vintages and boundary conventions, to be documented and
reconciled.

The geometry is **fetched rather than vendored** ([fetch_gadm.py](fetch_gadm.py)). GADM 4.1 is a
frozen, versioned, publicly hosted release, so a checksum manifest of the sixteen required
shapefiles provides the same reproducibility guarantee at negligible storage cost; the licence also
forbids redistribution for commercial use. Verification runs without network access and fails on
any mismatch, so it gates the rebuild: a re-cut GADM release would otherwise change every graph
reported in this document without any visible signal.

**Construction.** Contiguity is queen adjacency with a k-nearest-centroid fallback for islands,
built *within* each country and assembled block-diagonally. There are **no cross-border edges**:
the dengue countries are not mutually adjacent, and an ocean between Brazil and Thailand is not an
edge. Each country joins the shapefile at its own level, so a single global name field cannot serve
the graph — which is why adjacency construction had to follow, rather than accompany, level
resolution.

| Node | Joins geometry on |
|---|---|
| `japan\|tokyo` (Admin-1) | province name |
| `brazil\|piaui\|bom jesus` (Admin-2) | (province, municipality) — the parent is part of the key |
| `taiwan\|changhua county` (aggregated) | county name alone |

The final graph:

```
7,165 nodes · 20,468 undirected edges · mean degree 5.71
0 unmatched · 0 isolated · 0 cross-border edges · symmetric
```

| Country | Level | Nodes | Edges | Isolated |
|---|---|---|---|---|
| Brazil | Admin-2 | 5,517 | 16,259 | 0 |
| Colombia | Admin-2 | 1,066 | 3,021 | 0 |
| Argentina | Admin-2 | 275 | 547 | 0 |
| Peru | Admin-2 | 113 | 238 | 0 |
| Japan | Admin-1 | 47 | 94 | 0 |
| Dominican Republic | Admin-1 | 32 | 72 | 0 |
| Mexico | Admin-1 | 32 | 67 | 0 |
| Ecuador | Admin-1 | 24 | 58 | 0 |
| Taiwan | Admin-2 (county) | 22 | 45 | 0 |
| Nicaragua | Admin-1 | 17 | 34 | 0 |
| Panama | Admin-1 | 11 | 17 | 0 |
| Bolivia | Admin-1 | 9 | 16 | 0 |

**Unmatched nodes halt construction.** A node that cannot be joined to a polygon raises an error
carrying the full diagnostic; the builder never discards a node of its own accord. A silently
dropped node would be a hole in the graph *and* in the training set, and nothing downstream would
reveal it. The only nodes ever removed from the bundle are the explicitly declared, individually
reasoned exclusions listed in Section 1.9.

**Symmetrisation.** Queen contiguity is symmetric, but the k-nearest-neighbour fallback is not, so
an earlier builder produced an asymmetric adjacency matrix whenever the island fallback fired. The
matrix is now symmetrised explicitly, as the graph neural network requires.

### 1.6.1 The name join

Geometry is joined to the case series on **names, rather than on codes**, because the source
supplies no code columns. Accents, case and whitespace are absorbed by canonicalisation; the
residue is genuine editorial difference between two independently maintained naming conventions,
and is resolved by a hand-curated, committed alias map ([dengue_aliases.py](dengue_aliases.py)).

**Two maps are required, and they run at different times.** This separation is load-bearing:

| Map | Applied | Repairs | Renames the node? |
|---|---|---|---|
| `DENGUE_NAME_ALIASES` | at load, before aggregation | **the data**: the source's editorial drift, series split across two spellings, a homoglyph | yes, by design |
| `DENGUE_GADM_FIX` | at the join only | **the shapefile's own defects**: two misspellings in GADM 4.1 | never |

The load-time map must run before aggregation because, besides correcting the join, it **reunites
units whose series the source split across two spellings** (Section 1.9). Those halves must be
summed before the data are pivoted into a node-by-week matrix, which is impossible at join time.

But a load-time alias necessarily **rewrites the node identifier**. Consequently, when the
shapefile's own misspellings were folded into that map, they propagated into the project's
published identifiers. The dataset briefly contained nodes named `japan|naoasaki` and
`peru|huanuco|huenuco`, reproducing GADM's typographical errors in identifiers that would have
appeared in the node table of the paper. The defect is corrected and documented in Section 1.9.

**Every alias was resolved by exact match against a real shapefile key; none by fuzzy matching.**
Approximate matching would have bound `antioquia|san andres` to `andes`, a different municipality
altogether. A validation routine asserts that every alias resolves to a key that exists in the
shapefile, so an error in the map fails at build time rather than resurfacing later as an
unexplained unmatched node.

**Scale of the problem.** Three hundred and two nodes failed the initial join. Five Colombian
department names alone orphaned one hundred municipalities (`valle` → `valle del cauca`,
`norte santander` → `norte de santander`, `guajira` → `la guajira`, `san andres` → `san andres y
providencia`, `bogota` → `bogota d.c.`). The remainder is editorial drift: corregimiento suffixes;
parenthetical municipal seats, applied inconsistently by the source — sometimes the seat *is* the
shapefile's name, so each case was checked individually; Spanish numerals (`25 de mayo` →
`veinticinco de mayo`); and long official forms (`cali` → `santiago de cali`).

**Brazil (83 of 5,517).** All twenty-seven states matched, so Brazil's residue is purely at the
municipal level and falls into three groups:

- **Seventy-one punctuation differences.** The source strips apostrophes and hyphens that the
  shapefile retains (`olho d agua` → `olho d'agua`; `xique xique` → `xique-xique`). These were
  resolved by exact match on the punctuation-stripped name *within the same state*, accepted only
  where that key was unique in the state; none was ambiguous. This is an exact match modulo
  punctuation, not an approximate one, and the results were written out as explicit alias entries
  so that the committed map remains the single auditable artefact.
- **Ten orthographic variants** (`itapage` → `itapaje`; `poxoreo` → `poxoreu`; and similar).
- **Two municipal renames, with the two sources on opposite sides of each.** *Augusto Severo* was
  renamed *Campo Grande* in 2013: the shapefile carries the new name and the case data the old.
  *Tabocão* was renamed *Fortaleza do Tabocão*: the shapefile carries the old name and the case data
  the new. Both were verified to be one-to-one renames rather than mergers — in each case the
  counterpart name is absent from the data — so no node count changes.

### 1.6.2 Static covariates

Each node carries `[centroid latitude, centroid longitude, area]`, derived from the same shapefile
provenance as the graph. Areas and centroids are computed in an equal-area projection and only then
converted back to degrees; a degree is not a unit of area, and the distortion grows with latitude.
Values are stored unstandardised, since these covariates are static and therefore carry no leakage
risk, and scaling is properly the model's concern.

**Row alignment is guaranteed by construction rather than re-derived.** The covariates are computed
in the same pass as the adjacency, from polygons already reindexed into node order, so a covariate
row and its node identifier cannot drift apart. The build additionally asserts that every centroid
falls within its own country's bounding box, so a shuffled covariate matrix would fail the build.

The extreme values land on the correct real-world units, which validates the join independently:

| | Node | Computed | Published |
|---|---|---|---|
| Smallest | Santa Cruz de Minas, Minas Gerais | 3.6 km² | 3.57 km² (Brazil's smallest municipality) |
| Largest | Santa Cruz, Bolivia | 368,673 km² | 370,621 km² (Bolivia's largest department) |

The three nodes that initially fell outside their country's bounding box proved to be genuine
offshore islands — Fernando de Noronha, Providencia, and Lienkiang. Their centroids are correct;
that remote islands are located where remote islands belong is itself evidence that the alignment
holds. The bounding boxes were widened; the data were not touched.

**These covariates are not transfer-safe, and this is intrinsic.** See Section 4.2.

### 1.7 Evaluation protocol and the zero-variance nodes

The headline metric is intended to be a **country-macro average**: scored per node, averaged within
a country, then averaged across the twelve countries with equal weight, with scoring restricted to
observed cells and a node-micro average reported as a secondary figure. This construction would
neutralise Brazil's 77 per cent share of nodes without discarding any data.

**This metric is now implemented** ([score.py](score.py)): scored per node over its observed test
cells, averaged within each country, then across the twelve countries with equal weight, with the
equal-node average reported alongside as the secondary figure. It operates on model predictions, so
the datasets themselves still carry no scores — but the scoring code is fixed and self-tested: a
Bolivia-only error of 10 moves the equal-node mean by 0.015 and the country-macro by 0.833 (≈ 10/12),
confirming the de-weighting behaves as intended.

**The zero-variance guard fires on 400 of the 7,165 nodes**, not on the thirty reported in earlier
versions of this note. The earlier figure counted only nodes that are constant across their *entire*
history; the guard fires on any node with no variance *within its training window*, which is a much
larger and quite different set.

| Country | Nodes with a zero-variance training window |
|---|---|
| Colombia | 222 |
| Argentina | 113 |
| Japan | 29 |
| Brazil | 18 |
| Taiwan | 9 |
| Peru | 6 |
| Mexico | 3 |
| **Total** | **400** |

Two distinct situations are conflated in that number, and they have opposite implications.

1. **Genuinely constant nodes.** Twenty-nine of Japan's forty-seven prefectures report zero dengue in
   every observed week. These zeros are true — dengue is not endemic in Japan, the only autochthonous
   transmission in the period being the small Tokyo outbreak of 2014. The nodes are retained because
   they remain valid neighbours for message passing.
2. **Nodes constant in training but not in test — a mis-fire, now corrected.** In an earlier build,
   76 of the 400 had real variance in their held-out cells, and the blanket `sd = 1` guard left them
   **un-normalised** — on the raw `log1p` scale while every neighbour was standardised. That was a
   defect, concentrated in Colombia (43), Brazil (18) and Taiwan (7).

   **The guard now applies the country-pooled fallback** (the same mechanism §1.8 uses for nodes with
   no training cell): a node whose training window is constant takes its country's pooled training
   statistics rather than `(0, 1)`. On the released build the blanket `sd = 1` guard fires on **zero
   nodes**, and **no node with held-out variance is left un-normalised**. The pooled statistics are
   computed from training cells only, so the fix introduces no leakage.

**Consequences that remain (for genuinely constant nodes).**

- The coverage gate counts *observed* weeks and never asks whether those observations vary, so a
  fully-reported all-zero node passes it with a perfect score.
- Under a country-macro headline Japan would carry one-twelfth of the score while 62 per cent of its
  nodes are trivially predictable — a model that emits zero is exactly right — so Japan's figure
  would substantially measure the prediction of a constant.

**Decided (both).** Genuinely-constant nodes are **excluded from scoring** while retained in the graph
as message-passing neighbours, and Japan is **also reported with them removed as a sensitivity**.
[score.py](score.py) implements the exclusion by default (`score_constant=False`): a node whose truth is
constant over the cells being scored is dropped, since a constant prediction is trivially exact there
and its correlation is undefined. On the released build 559 nodes are constant over their test window
and are excluded from the headline, and 6,161 are scored. The normalisation defect (above) is a
separate matter and is fixed.

### 1.8 Splits

The headline split is a **per-country chronological 50/20/30 partition**, each country cut on its
own observed span.

A single global positional cut on the 1997–2024 union calendar would starve every country whose
reporting begins after that cut. Brazil, Japan, Ecuador, Panama and Argentina would receive **zero
training cells**, and hence an undefined per-node scaler — silently, since no schema check would
fail. Cutting each country on its own timeline gives every country a genuine training, validation
and test partition.

**Every node of a country takes that country's boundary, without exception.** At any week an entire
country block is therefore in a single phase, which is what keeps the block-diagonal graph
temporally consistent: a graph neural network aggregates over neighbours, so a training cell sitting
beside a test cell in the same week would pull the test period into the training forward pass.

An earlier release did not have this property. A per-node fallback re-cut any node whose
observations fell entirely beyond its country's training boundary, in order to guarantee it at least
one training cell. Measured on that build, the fallback re-cut 475 nodes, placed 2,420 training
cells *after* their own country's validation boundary, and left 792 (country, week) columns holding
more than one phase — 685,214 observed cells, 31 per cent of the corpus, in mixed-phase columns. The
note asserted the single-phase invariant while the code violated it.

**The fallback has been removed.** The 475 affected nodes — Brazil 379, Peru 49, Colombia 31,
Argentina 15, Taiwan 1 — are nodes whose reporting begins after their country's training boundary.
They keep their place in the graph and are still scored, but they have no training cell of their own
and so take **their country's pooled training statistics** as their scaler. Those statistics are
computed from training cells only and are therefore visible at training time, so the substitution
introduces no leakage. The affected nodes are listed in `meta["split"]["nodes_without_train"]`.

Verified on the released build: no country is starved of training cells, every node has a defined
scaler, no (country, week) column mixes phases, and the three partitions divide the observed cells
disjointly. Scalers are fitted on training cells only. Each of these is a gate in the leakage suite
(§4.3) with a negative control that fails when the property is deliberately broken.

| | |
|---|---|
| Training cells | 1,249,109 |
| Validation cells | 524,972 |
| Test cells | 422,180 |
| Nodes with no training cell (country-pooled scaler) | 475 |

### 1.9 Alterations to the data

The geometry and the case series genuinely disagree in places, and the reconciliation could not be
completed by name-mapping alone. Each disagreement was resolved by a decision. All are listed here
so that a reviewer need never discover one by reading the source. Each is reversible, and none is
baked into the raw files.

Net effect on node counts: Colombia 1,071 → 1,066; Taiwan 287 → 22; Dominican Republic 35 → 32;
Peru 115 → 113; Ecuador 25 → 24; Nicaragua 18 → 17; Argentina 276 → 275; **Brazil 5,517 → 5,517
(unchanged)**.

**Brazil required no alterations.** All eighty-three of its unmatched nodes resolved to exact
shapefile targets. Nothing in Brazil was merged, aggregated or dropped.

---

**Taiwan: 287 townships aggregated into 22 counties.**
GADM 4.1 ships no township layer for Taiwan — its deepest Taiwanese layer is the 22 counties — and
Taiwan has no Admin-1 weekly records to fall back on. No alias can supply a missing layer.
*Decision:* retain Taiwan and aggregate its townships into their parent counties. Townships
partition counties exactly, so the sum is exact and nothing is double-counted; all 22 counties join
the shapefile cleanly. The alternatives were to drop Taiwan, or to introduce a second shapefile
provenance for one country.
*Consequence:* Taiwan's spatial resolution is reduced, and the "finest available level" rule of
Section 1.2 should be read as *the finest level the shapefile can actually support*.

**Seven units absent from the shapefile, folded into their pre-split parents.**
Colombia's Albania, Guachené, Tuchín, San José de Uré and Norosí, and Peru's Datem del Marañón and
Putumayo, were all created *after* the shapefile's boundary vintage, so no polygon exists for the
child unit.
*Decision:* fold each child's counts into its pre-split parent, whose polygon still physically
contains the child's territory precisely because the shapefile predates the split. The operation is
geometrically exact and lossless, and strictly preferable to discarding the node.
*Residual risk:* had the parent polygon excluded the child, this would misattribute counts. It does
not — the child's absence from the shapefile is itself the evidence that its territory remains
inside the parent.

**Nicaragua: two concurrent health districts summed into one department.**
`Zelaya Central` and `Región Autónoma del Atlántico Sur` overlap completely — 981 weeks each, over
an identical span. They are concurrent health districts that partition the Atlántico Sur
department, not two names for one unit in different eras.
*Decision:* sum both into the single departmental polygon they jointly represent (18 → 17 nodes).
*Residual risk:* if the autonomous region already reported the whole department, rather than the
remainder after Zelaya Central was carved out, this double-counts. The complete 981-week overlap of
two separately named units is strong evidence of a partition, but **this is the one merger here that
rests on an administrative reading rather than on arithmetic**, and it is flagged as such.

**Four provinces whose series were split across two spellings, reunited.**
The source reports one unit under two spellings in strictly disjoint eras:

| Province | First spelling | Second spelling | Temporal overlap |
|---|---|---|---|
| Hermanas Mirabal (DR) | `salcedo` (2009–13) | `hermanas mirabal` (2006–08) | none |
| María Trinidad Sánchez (DR) | `maria trinidad sanchez` (2006–08) | `maria trinidad sanches` (2012–13) | none |
| Santiago Rodríguez (DR) | `santiago rodriguez` (2012–13) | `santiago. rodriguez` (2006–08) | none |
| Zamora Chinchipe (Ecuador) | `zamora chinchipe` (74 wk) | `zamora chincipe` (2 wk) | none |

Salcedo to Hermanas Mirabal is the province's actual 2007 renaming. Each unit was previously **two
nodes, each holding half a history, each with a scaler fitted on that half**.
*Decision:* merge. The absence of temporal overlap makes the merge unambiguous. The Dominican
Republic then lands on exactly 32 nodes and Ecuador on exactly 24 — matching the shapefile's
province counts, which is independent corroboration.

**One node dropped: `argentina|formosa|grl. jose de san martin`.**
Formosa has exactly nine departments and this is not one of them; the department of that name
belongs to Chaco and Salta, both of which the source reports separately. This is a mislabelled
source row.
*Decision:* drop it. It cannot be reassigned without inventing data. The removal is declared with
its reason in the exclusion list and recorded in the dataset metadata, so the builder's guarantee —
that it never discards a node of its own accord — is preserved. Every removal is a decision someone
wrote down.

**Sibling disambiguation, resolved from evidence rather than guessed.**
`antioquia|san pedro` is ambiguous between *San Pedro de los Milagros* and *San Pedro de Urabá*;
`sucre|palmito` between *San Antonio de Palmito* and *Los Palmitos*.
*Resolution:* the source carries `san pedro de uraba` and `los palmitos` as separate nodes, so the
bare names must denote the other member of each pair. Approximate matching would have chosen wrongly.

**Two shapefile labelling anomalies, documented rather than silently accepted.**
The shapefile labels Santander's *San Andrés* with Antioquia's suffix; the polygon itself lies
inside Santander, so the join is geometrically correct and only the label is anomalous. Separately,
the shapefile splits the San Fernando *partido* into a mainland polygon and a delta-island polygon;
the node is bound to the mainland, where effectively the entire reporting population lives, at the
cost that the delta polygon contributes no contiguity edges.

**Shapefile misspellings, absorbed at the join.**
GADM 4.1 spells Nagasaki as *Naoasaki* and Peru's Huánuco province as *Huenuco*. The defect is
upstream: the case data are right and the shapefile is wrong.

These two entries were originally folded into the **load-time** alias map, which is where the
project repairs defects in the *case data*. Because a load-time alias rewrites the node identifier,
the dataset consequently shipped nodes named `japan|naoasaki` and `peru|huanuco|huenuco` — the
shapefile's typographical errors propagated into the project's own identifiers, where they would
have been frozen into the released dataset and printed in the paper's node table.

*The correction is a second map, not a corrected entry.* The two maps repair two different things
and must therefore run at two different times, as set out in Section 1.6.1. A join-time map now
absorbs the shapefile's defects and never renames a node.

It must be a separation in *time*, not a reversal of direction. Simply inverting the two entries
would not work: the load-time map is applied before aggregation, so a reversed entry would rename
the *shapefile's* key rather than the data's, and the join would then fail to find the polygon.
Nor can the load-time map be moved to join time wholesale, because the split series described above
must be summed before aggregation.

*Effect, measured.* The rebuilt graph is identical to its predecessor — 7,165 nodes, 20,468 edges,
mask density 0.2175, no unmatched nodes, no cross-border edges. **Only the two identifiers changed.**
A regression test now asserts that no load-time alias may target a known shapefile misspelling,
since the way this defect recurs is by someone adding a shapefile quirk to the wrong map.

**A homoglyph in the source data.** The source encodes Peru's *Marañón* with a Cyrillic character
where a Latin `e` belongs. Canonicalisation strips accents but cannot repair a homoglyph, so this
node could never have matched any shapefile. It is mapped explicitly. The finding is flagged because
it implies the extract may contain further homoglyphs elsewhere.

**Symmetrisation of the island fallback.** The k-nearest-neighbour fallback is directionally
asymmetric and, left uncorrected, produced an asymmetric adjacency matrix. It is now symmetrised.
This is a correctness fix, recorded because it changes edges relative to the earlier build.

### 1.10 Case definitions over time — measured, and milder than first reported

An adversarial review flagged five countries as changing their case definition mid-series, with
multipliers up to ×220, and this note's earlier draft repeated those figures. **They do not survive
measurement against the data that actually enters the model.** The multipliers were computed by grouping
the *raw source file* by `case_definition_standardised` and comparing group means; they are not
properties of any node's time series.

The correct measurement is per node, over its own history, on the extracted values:

| Country | Raw-CSV multiplier (earlier draft) | Extracted level step across the split | Notes |
|---|---|---|---|
| Peru | ×220 | **none** | every built Peru node is single-definition (`probable+confirmed`) throughout |
| Dominican Republic | ×48 | **none** | single-definition (`probable`) throughout |
| Mexico | ×13 | **2.8×** | `confirmed` (train) → `total` (test); part definition, part real trend |
| Bolivia | ×11 | **1.6×** | `suspected`→`total`, the change sitting *inside* the training window |
| Panama | ×7 | — | a single node |

Peru's "×220" was a 52-row sliver labelled `total` (one node, 2023–2024) that OpenDengue's
best-resolution extract never selected; the built Peru series is entirely `probable+confirmed`, mean
weekly 24.9. The largest Peru node, Piura (peak 4,720), is single-definition across its full 2000–2023
span. The genuine exposure is Mexico (a 2.8× step across the split, entangled with two decades of real
dengue growth) and Bolivia (1.6×, and mild because the model already trains on the later definition).
Together this is **41 of 7,165 nodes**.

**Decision: kept as built, and disclosed.** A 2.8× and a 1.6× shift, partly genuine trend, on 0.6 per
cent of nodes does not justify discarding Mexico's twenty-year history or Bolivia's modern series;
truncating to a single-definition era would remove real data to suppress a distortion smaller than the
trend it is entangled with. Any Mexico or Bolivia result should be read with the step in mind. See the
decisions note, Part A2.

The lesson is the same one that recurs in this note: a per-*node*-over-time question must be measured on
the per-node series, not on a raw-file group-by, which answers a different question.

### 1.11 Known limitations

- **Brazilian dominance.** Brazil supplies 77.0 per cent of nodes. Reporting is de-biased by the
  country-macro evaluation, now implemented (§1.7, [score.py](score.py)), rather than by discarding
  nodes. Training keeps **uniform node sampling as the primary regime** — the encoder is therefore
  data-proportional and Brazil-weighted, which preserves the full training signal — and a
  **country-balanced sampler is provided as an ablation** to measure whether balancing improves
  per-country accuracy and cross-disease transfer. Which is better is settled empirically in the
  modelling phase, not by decree.
- **Case definitions over time.** See Section 1.10. Measured to a 2.8× (Mexico) and 1.6× (Bolivia)
  level step on 41 nodes; kept as built and disclosed. The earlier ×220 figures were raw-file artefacts.
- **Zero-variance nodes.** See Section 1.7. The guard fires where a node's training window is constant;
  such nodes now inherit their country's pooled training scaler rather than being left un-normalised
  (§1.7). This closes the mis-fire that previously left 76 nodes on the raw scale.
- **The graph forbids cross-border edges.** The block-diagonal construction was justified on the
  grounds that the dengue countries are not adjacent. **That is false**: Brazil borders Colombia, Peru,
  Bolivia and Argentina, and the construction severs real transmission corridors — Leticia–Tabatinga
  and the Triple Frontier among them. The Ebola graph, by contrast, correctly keeps its cross-border
  edges (§3.7). The dengue choice is defensible only as a deliberate simplification, and it is recorded
  here as such rather than as a geographic fact.
- **Contiguity is an induced subgraph.** Edges exist only between nodes *present in the data*, so
  two municipalities adjacent only through a third that never reported have no edge between them.
  This is the correct semantics, but it means that node degree is a property of coverage as well as
  of geography.
- **Mixed spatial resolution.** Admin-1 and Admin-2 nodes coexist in one dataset. This is safe by
  construction — the graph is block-diagonal, scalers are per-node, and the level is never exposed to
  the model — but node counts are not comparable across countries.
- **Fifty-seven endemic countries are excluded** for want of subnational weekly data. This is real
  dengue burden that carries no spatial signal in this extract and therefore cannot enter a
  spatially-coupled model.

---

## 2. Influenza — ColaGNN benchmarks

### 2.1 Provenance

The influenza data are the **benchmark matrices shipped with ColaGNN, used unchanged, with the
adjacency reused**. The purpose is exact comparability: the framework's influenza results are
intended to sit directly beside published baseline results, which requires that the comparison
occur on an identical graph. Delphi's live surveillance API and the WHO's global influenza database
were both rejected on the same ground — either would require the graph to be rebuilt, forfeiting
that comparability.

| | |
|---|---|
| Source | ColaGNN (Deng et al., CIKM 2020) |
| Mirror | EpiGNN (ECML-PKDD 2022) — verified byte-identical by checksum |
| Working copy | `data/Final datasets/influenza/`, checksums verified on every load |

| File | SHA-256 |
|---|---|
| `japan.txt` | `85a549a8f45e6c0227bd21198373cfbd8269bb79111f4b866c8a839f51afb3dc` |
| `japan-adj.txt` | `c18d52daf6b57eb230b754d9212df0b5525a30ac2f9403547d0848c44eb2bc87` |
| `region785.txt` | `31e48b7623873224c62f4933978c8c7c4d3c15c8fc327481d8801331ac98b9a9` |
| `region-adj.txt` | `6305db391951c78ecf1b49b713b61a8478cc9b620ca9201f8257e5a72717e143` |
| `state360.txt` | `ccf2aabea34a5d539c2267144e10db4bcedb54f67846a9bd9234b3c4b42fe0c7` |
| `state-adj.txt` | `a7642da53a4551490a12dcade8e3d53b9af88757320d48f6feb58501d858ad7b` |

The matrices previously existed only inside an ignored directory, and a fresh checkout of the
repository could not have regenerated the datasets. They are now cached under the version-controlled
data root.

### 2.2 Three separate datasets, not concatenated

The three benchmarks have **disjoint node sets and different calendars**. Transfer in this framework
operates through shared model *parameters*, not through shared indices: the diseases neither share a
node set nor a common calendar. The three benchmarks are therefore produced as three independent
datasets rather than one stitched matrix, which additionally keeps leave-one-disease-out and
per-dataset reporting clean.

| Dataset | Nodes | Weeks | Calendar | Edges | Mask density |
|---|---|---|---|---|---|
| Japan | 47 prefectures | 348 | 2012-08-04 → 2019-03-30 | 86 | 1.000 |
| US regions | 10 HHS regions | 785 | 2002-10-05 → 2017-10-14 | 16 | 1.000 |
| US states | 49 states | 360 | 2010-10-09 → 2017-08-26 | 103 | 1.000 |

All three are **fully observed**; the shipped matrices contain no gaps, so influenza contributes no
imputed steps to the harmonised schema.

Each dataset carries its own scaler, split and graph. Independence is **asserted at build time**
rather than merely intended: node sets are checked to be pairwise disjoint and no two calendars may
coincide. The instruction not to concatenate is thereby a checked property rather than a convention
that a later contributor might unknowingly violate.

### 2.3 Date anchoring, and the provenance of the US matrices

The shipped matrices are **undated**; the documentation states only that rows are weeks in
chronological order. The calendar had to be recovered, and the published year range is not sufficient
to recover it. What began as a calendar reconstruction ended as a **provenance identification**: the
two US matrices are not merely *consistent with* CDC ILINet — they **are** ILINet, verified to the case.

**The identification.** `region785.txt` was matched cell-by-cell against CDC ILINet `num_ili` (the raw
patient counts, publicly queryable through the Delphi Epidata API) for the ten HHS regions over MMWR
epiweeks 200240–201741. **759 of its 785 rows are identical across all ten regions simultaneously**;
the 26 that differ are all dated 2016-10-08 or later and differ by revision-level amounts — the
signature of ILINet backfill against the benchmark's circa-2017 snapshot. `state360.txt` matches ILINet
by state over epiweeks 201040–201734 at **16,890 of 17,327 cells (97.5 per cent)**, the residual again
being post-hoc revision spread across states. The matrices are therefore ILINet at a known vintage, and
the released datasets can cite that provenance rather than a reconstruction.

**The anchor falls out of the identification, exactly.** Row 368 of the regional matrix equals ILINet
epiweek 200942 (week ending 2009-10-24, the H1N1 autumn peak) to the case, which fixes the start at
**2002-10-05 (MMWR week 40 of 2002)**. This is a far stronger pin than the earlier argument, which
rested on the H1N1 wave being the series maximum and was, as sole evidence, post-hoc — choosing the
maximum and then choosing the event that explains it. The 759-row match needs no such argument.

| Dataset | Original (wrong) anchor | Corrected anchor | Basis |
|---|---|---|---|
| US regions | 2002-01-05 | 2002-10-05 | 759/785 rows identical to ILINet epiweeks 200240–201741 |
| US states | 2010-01-09 | 2010-10-09 | 97.5% of cells identical to ILINet epiweeks 201040–201734 |
| Japan | 2012-08-04 | unchanged | seasonal phase only — see caveat below |

**The error this corrected.** Anchoring the US datasets to January — the obvious reading of "2002–2017"
and "2010–2017" — placed both calendars thirty-nine weeks out of phase, because US surveillance is
published by influenza *season* and both matrices begin at MMWR week 40. Under the January anchor the
seasonality channels were ~180° out: summer during every winter peak, the aggregate peaking in May. A
row-count test could not catch it, because the count matches the span under either anchor. The
identification above removes any residual doubt.

**Two provenance details to carry into any results table.** Column 30 of `state360.txt`, which the
ordering would name "New York", is ILINet's **NY-excluding-NYC** series (`ny_minus_jfk`); and **Florida
is absent** from the state matrix, consistent with ILINet not reporting it. Both are properties of
ILINet, not of our processing.

**Japan is now pinned to the week, and is no longer the project's weakest calendar.** The anchor
(2012-08-04) was formerly justified on seasonal phase alone (±4 weeks). It is now identified the way
the US matrices were — by matching the matrix's per-season national maxima to NIID's dated influenza
season peaks. The 2018/19 maximum lands on **2019-01-26** (NIID IASR 40(11): peak week 4 of 2019, at
57.09 patients/sentinel, the highest since 1999) and the 2017/18 maximum on **2018-02-03** (NIID IASR
39(11): peak week 5 of 2018) — each the exact epidemiological week NIID records. A one-week shift in
the anchor would move at least one maximum off its dated week, so the two-season match, together with
the paper's stated span (August 2012–March 2019, 348 weeks) and the matrix fingerprint (max 26,635 /
mean 655 / SD 1,711, matching ColaGNN Table 2), pins the anchor to the week. Re-asserted on every run
by [japan_calendar_pin.py](japan_calendar_pin.py).

**Regression guards.** Each aggregate series must peak December–March and trough June–October; the
regional matrix must match ILINet at its known vintage; and every start date must already be a
week-ending Saturday, because the date-range constructor silently advances a non-Saturday start,
shifting the whole calendar with no error raised.

### 2.4 The graph and the self-loop convention

**Finding.** All three shipped adjacency matrices carry a **self-loop on every node**. The graphs
built for dengue and Ebola do not.

**Why this matters.** The shared encoder applies the standard renormalisation, adding the identity
matrix once. A graph arriving with self-loops already present would therefore give influenza nodes
**twice the self-weight** of dengue nodes. This is a systematic structural difference between
diseases, and under degree normalisation it alters every node's weighting. It is precisely the
failure mode the framework exists to eliminate: a signal from which the encoder could infer *which
disease it is looking at*. The alternative — teaching the encoder to omit the identity for influenza
alone — is disease-specific special-casing, which reintroduces the same disease-specificity through
another door.

**Decision.** The diagonal is zeroed for influenza, matching dengue and Ebola.

**Comparability is not weakened.** The operation is information-preserving: the shipped matrix is
exactly recoverable by adding back the identity. Only the diagonal is touched, and the off-diagonal
edges are reused bit for bit, which is asserted at load time and in the test suite. After the encoder
adds the identity, the model sees precisely the matrix the published baselines consume.

**Consequence: four nodes are isolated.** Zeroing the diagonal exposed something the self-loops had
concealed — four nodes have no neighbours at all; their only entry *was* the self-loop.

| Dataset | Isolated nodes |
|---|---|
| Japan | 2 (Hokkaidō, Okinawa) |
| US regions | 0 |
| US states | 2 (Alaska, Hawaii) |

This isolation is **inherited from the shipped graph, not introduced here**: every isolated node has
a shipped row sum of one and an off-diagonal sum of zero. The build asserts that zeroing the diagonal
never isolated a node that had a genuine neighbour.

They are deliberately **not** patched with a nearest-neighbour edge, although the builder does
exactly that for dengue and Ebola. Inventing an edge absent from the shipped graph would destroy the
exact comparability that is the sole reason for reusing it, and these nodes are anonymous indices
with no geometry to fall back upon. They remain degree-zero in storage and are made non-degenerate by
the encoder's uniform addition of the identity, which restores them to self-aggregation only —
precisely the behaviour of the published baselines.

**A constraint on the model.** The encoder **must** add the identity before any degree normalisation,
or these four rows divide by zero. This requirement is recorded on every influenza dataset in its
metadata and asserted in both the loader and the test suite.

### 2.5 Node ordering

No name list ships with the data; the nodes are anonymous indices. A wrong ordering would silently
corrupt any join to geography, and the static covariates cannot be constructed at all without one, so
both orderings were **recovered and tested rather than assumed**.

**US states.** The hypothesis that the columns are the fifty states in alphabetical order, less
Florida (which the surveillance system does not report), was tested against real land contiguity. All
103 shipped edges are genuine contiguity edges, with none spurious; agreement is 99.83 per cent over
all 1,176 possible pairs; and the isolated pair is exactly Alaska and Hawaii, the two non-contiguous
states. The only two genuinely contiguous pairs *absent* from the shipped graph are Arizona–Colorado
and New Mexico–Utah, which meet only at the Four Corners *point*. The shipped graph is therefore
clean rook contiguity — a shared border segment — rather than queen.

**Japan** ([japan_nodes.py](japan_nodes.py) → [japan_node_map.csv](japan_node_map.csv)). The ordering
matches neither the standard prefecture order nor romaji-alphabetical order; both were tested and
disconfirmed. It was recovered instead from three *independent* lines of evidence, each weak alone but
decisive in combination.

1. **Structure.** Queen contiguity over Japan's forty-seven prefectures yields exactly 86 edges — the
   shipped graph's count — and the two graphs are isomorphic. The isolated prefectures are Hokkaidō
   and Okinawa, Japan's only two without a land border. This pins **41 of 47** nodes uniquely. The
   remaining six lie in three automorphism orbits that structure alone physically cannot separate:
   Hokkaidō and Okinawa (both isolated), and the structurally symmetric Shikoku pairs Kagawa/Kōchi and
   Ehime/Tokushima.
2. **Magnitude.** Case counts track population, and nothing about a graph knows population. The chosen
   labelling gives a Spearman correlation of +0.949 between mean case counts and prefecture population
   across all forty-seven nodes. The eight highest emerge as Tokyo, Kanagawa, Saitama, Osaka, Aichi,
   Fukuoka, Chiba and Hokkaidō — Japan's most populous prefectures — and the lowest as Tottori and
   Shimane, its two least. Within every ambiguous orbit, the higher-count node is the more populous.
3. **Seasonal phase.** Okinawa is subtropical and known for out-of-season influenza. One node is the
   least winter-concentrated of all forty-seven, with 74.0 per cent of its cases in December–March
   against a national mean of 92.2 per cent, five standard deviations below the mean. Seasonal phase is
   independent of both structure and magnitude, and it identifies Okinawa, and hence Hokkaidō.

| Nodes | Basis | Confidence |
|---|---|---|
| 41 of 47 | graph isomorphism alone | certain — no other labelling is structurally possible |
| Hokkaidō, Okinawa | structure, magnitude and phase | high — two independent confirmations |
| Ehime, Tokushima | structure and magnitude | good — case ratio 2.00 against population ratio 1.85 |
| Kōchi, Kagawa | structure and magnitude | **moderate — the weakest identification** (1.22 against 1.38) |

**On the weakest identification.** If Kagawa and Kōchi were transposed, nothing structural changes:
the two nodes are automorphic, so the graph is identical either way and no model result moves. The
transposition would matter only for a covariate join, where it would exchange two adjacent,
similarly-sized prefectures. This is recorded in the confidence column of the node map rather than
presented as certain.

**US regions.** Each region is a union of states, so its true contiguity can be constructed
independently and compared. Under the assumed ordering, all sixteen shipped edges are genuine
contiguity edges with none spurious, and the degree sequence matches. The independent construction
finds exactly one *additional* edge, which is a Great Lakes water adjacency: the state polygons extend
into the lakes. The shipped graph uses strict *land* contiguity — the same convention that made it
omit the Four Corners point-touches for the state dataset. The ordering is confirmed.

### 2.6 Splits and scalers

A fixed chronological 50/20/30 split, the standard protocol for the development diseases. Scalers are
fitted **per node on the training slice only**, so no validation or test cell contributes to a
normalisation statistic. Influenza requires no per-country split machinery: each dataset is a single
calendar with no late-starting blocks.

### 2.7 Alterations to the data

Influenza is the disease used *unchanged*, so this list is short by design. It is not, however, empty.

**The adjacency diagonal is zeroed.** Documented in full in Section 2.4. The operation is
information-preserving, the off-diagonal edges are reused bit for bit, and baseline comparability is
intact.

**Both US date anchors were corrected by thirty-nine weeks.** This is not a change to the *values* but
to the calendar attached to them — and the values are meaningless without it. Documented in full in
Section 2.3.

*Not done, although it could have been:* no node was dropped, no series smoothed, and no gap imputed
(there are none). No dataset was concatenated with another, no COVID-19 series was admitted, and no
mobility graph was attached.

### 2.8 Static covariates and mobility

Each node carries centroid and area, on the same contract and from the same provenance as dengue.
Japan joins through the recovered node map; US states through the verified alphabetical ordering; US
regions by taking the union of each region's member-state geometry.

| Dataset | Largest node | Total area |
|---|---|---|
| Japan | Hokkaidō, 78,058 km² | 372,468 km² (published: 377,975) |
| US states | Alaska, 1,507,385 km² | 9,325,229 km² |
| US regions | Region 10, which contains Alaska | 9,472,856 km² |

**A mis-joined geometry would fail silently** — the covariates would be full of plausible numbers
attached to the wrong nodes, and nothing downstream would complain. They are therefore tested against
geography known independently of the shapefile
([test_influenza_covariates.py](test_influenza_covariates.py)):

- **Japan.** The largest and northernmost prefecture is Hokkaidō; the southernmost and westernmost is
  Okinawa; the smallest is Kagawa. The five largest reproduce the published ranking exactly, and the
  total area is within 1.5 per cent of the published figure.
- **US states.** The largest and northernmost is Alaska; the southernmost is Hawaii; the smallest is
  Rhode Island. The five largest are exactly Alaska, Texas, California, Montana and New Mexico.
- **US regions.** Every region's area equals the **sum of its member states' areas**. The regional
  geometry is built by union while the state geometry is built independently, so the two can agree only
  if both joins reached the correct polygons. Moreover, the regional total *minus* the state total is
  147,627 km², which is exactly Florida plus the District of Columbia — precisely the units that the
  regional series includes and the 49-state series excludes.

These results serve as a third, independent confirmation of the recovered node orderings: had an
ordering been wrong, Alaska would not have landed where it did, nor Hokkaidō.

**Mobility is not populated** for any disease. A real commuting matrix exists for Japan and is
tempting for that dataset, but no equivalent exists for dengue or Ebola, so attaching it to Japan alone
would reintroduce exactly the cross-disease asymmetry that zeroing the adjacency diagonal removes. Its
own node ordering is also undocumented.

### 2.9 COVID-19 exclusion

Stated for the record: the COVID-19 datasets shipped with the baseline implementations were used
**only** in the Phase-1 reproduction fidelity checks. The development set is **dengue and influenza**.
No COVID-19 series enters the harmonised schema and no COVID-19 dataset is released. This is recorded
to pre-empt the reasonable question of why COVID-19 appears in the reproduction tables but not in the
framework.

### 2.10 Known limitations

- **Two Japanese node identities rest on non-structural evidence** (Kōchi and Kagawa), and thinly. A
  transposition changes no model result — the nodes are automorphic — but would mislabel two
  prefectures in any covariate join.
- **The contiguity convention differs from dengue and Ebola.** The shipped graphs are rook contiguity;
  the graphs built here are queen contiguity with a nearest-neighbour fallback. This is inherent to
  reusing the shipped graph verbatim, and reconciling it would mean rebuilding the graph and forfeiting
  baseline comparability. It is disclosed rather than fixed.
- **Isolated nodes carry no spatial signal.** Alaska, Hawaii, Hokkaidō and Okinawa receive only their
  own history through the graph. This is a property of the benchmark, identical for the published
  baselines, so it does not bias the comparison.
- **Florida is absent** from the state dataset — the surveillance system does not report it.

---

## 3. Ebola — OCHA ROWCA compilation

This is the **held-out, data-scarce disease** on which the framework's few-shot claim rests.

### 3.1 Provenance

| | |
|---|---|
| Source | OCHA ROWCA "Ebola All Sec Review" — a manual compilation of WHO and Ministry of Health situation reports |
| Landing page | [data.humdata.org/dataset/rowca-ebola-cases](https://data.humdata.org/dataset/rowca-ebola-cases) |
| Resource | `Data Ebola (Public).xlsx`, resource `9b68ab69-e0b3-4ff5-b2d8-90bf446acd52` ([direct download](https://data.humdata.org/dataset/f326cf9d-8e37-437d-9b85-5ea3d36e3d39/resource/9b68ab69-e0b3-4ff5-b2d8-90bf446acd52/download/data-ebola-public.xlsx)) |
| Working copy | `data/Final datasets/data-ebola-public.xlsx` |
| Size | 1,955,645 bytes |
| SHA-256 | `2d679a31a66f912da93fe0bd73da82a3c36b0d1143b1fcc298bc921d5f28f9f2` |
| Shape | 58,635 rows × 7 columns |
| Span | 2014-03-24 → 2015-03-28, the outbreak's most intense phase |
| Last modified upstream | 2015-11-24 |

**The recorded checksum is that of the file as published**, verified against a fresh download from the
landing page above. A reader may therefore download the resource, hash it, and confirm they hold the
same bytes the results in this note were computed from. The loader re-verifies the checksum on every
run, so a substituted or re-issued compilation cannot enter the dataset unnoticed.

**A correction to an earlier version of this note.** The working copy previously used was named
`data-ebola-public_best_yet.xlsx` and had a different checksum
(`937136a4…52972`, 1,968,712 bytes). That file had been **opened and re-saved in a spreadsheet
application** at some point before it reached the project, which rewrites the container without altering
a single cell. Its contents were verified identical to the published resource — the same two sheets, the
same 58,635 × 7 table, comparing equal cell for cell — and the dataset built from it was bit-identical:
the same 61 nodes over 52 weeks, with every array and both split masks equal. **No figure in this note
changes.** But the recorded hash was that of the re-saved copy, not
of the published file, so a reviewer hashing their own download would have obtained a mismatch and might
reasonably have concluded that the results rested on something other than the public data. The pristine
resource is now the source of record.

**On the volatility of this source.** It is a manually maintained compilation rather than a versioned
release, and it carries no version number, so the checksum is its only version identifier. In practice it
is stable: the resource has not been modified upstream since **November 2015**, and it is served from a
permanent URL, so it is re-fetchable in the same sense that a versioned release is. Should it
nonetheless be re-issued, the checksum will reveal the substitution, and the signal-confirmation audit
below re-asserts every count on each run, so a change in the file's *contents* would also fail loudly
rather than quietly shifting the numbers reported here.

The Phase-1 loader had been written against a different file — a clean national-level series — whereas
the file actually held is the subnational compilation, with different column names and a different
format. The loader was rewritten accordingly.

### 3.2 Confirmation of usable signal

Data scarcity for Ebola was identified at the outset as the project's principal risk, and confirmation
of usable signal was required *before* the few-shot design could rest upon it. The audit
([ebola_audit.py](ebola_audit.py)) reproduces the expected figures against the production file. **Every
number matches.**

| | Expected | Production file |
|---|---|---|
| Clean single-district case series | 64 | 64 |
| Guinea / Liberia / Sierra Leone | 32 / 15 / 17 | 32 / 15 / 17 |
| Mean observed weeks per district | 27.1 | 27.1 (median 30, maximum 42) |
| Districts with ≥ 20 observed weeks | 53 | 53 |
| In-window rows | 58,632 | 58,632 |

**Usable Ebola signal is confirmed.** Across 61 districts the compilation yields **1,299 observed cells**
after the corrections of §3.4; under the calendar-prefix support of §3.6 that splits into 27 support and
1,272 query cells. The few-shot design is comfortably within the data.

**The case series is cumulative and frequently falls**: only **5 of 64** series are monotonically
non-decreasing, and **58 of 64** contain at least one downward step.

An earlier version of this note cited those same figures as a *justification* for clipping negative
increments to zero, on the reasoning that downward steps are reporting reconciliation rather than
negative incidence. **That reasoning was wrong, and the figures were evidence of a defect rather than
of a licence.** Section 3.4 sets out what the downward steps actually are and what the clip did with
them.

**Four properties of the file were not anticipated** and are all handled:

1. The national aggregate is a single *label* but 3,613 *rows*, some 6.2 per cent of the file.
2. Seven `Value` cells are non-numeric. These must be discarded **before** the de-duplication step that
   retains the last row for each district and date; otherwise a non-numeric cell silently overwrites
   that district's genuine cumulative count for the week.
3. Some districts have very few observed weeks. Under the earlier per-district support scheme two of
   them (`guinea|boke`, `guinea|lelouma`) yielded no query week; under the calendar-prefix support of
   §3.6 all 61 districts are evaluated.
4. The week-zero backlog, described in Section 3.4.

### 3.3 Target indicator

The file carries eight category values, which canonicalisation collapses to six by merging
inconsistent capitalisations.

| Category | Rows | Role |
|---|---|---|
| Cases | 10,192 | **the forecasting target** — cumulative, differenced on its running maximum |
| Deaths | 10,156 | an extended, single-disease channel |
| Confirmed / probable / suspected / new | 37,258 | not used as the target |

The headline cumulative case series is used as the target: it is the densest series available and
matches the method stated in the manuscript. Summing confirmed, probable and suspected cases is
epidemiologically explicit, but the three sub-series are not always co-reported, so the sum is
defined only on their intersection and coverage is materially worse.

**The *new cases* series deserves reconsideration, and the grounds on which it was rejected do not
survive measurement.** It was declined as "sparser and inconsistently capitalised". The
capitalisation is handled by the same canonicalisation applied to every other field, and the series
carries **9,494 rows against the cumulative series' 9,978 — 95 per cent of the coverage**, which is
not a material loss. It is also a *directly reported* incidence series, so it requires no
differencing and cannot be corrupted by the class of defect described in §3.4. Had it been used as a
cross-check, that defect would have been caught immediately. It is not the target in this release —
changing the target now would invalidate the build — but it is **adopted as a cross-check series**
against the cumulative-derived target, so that the target's credibility rests on two independently
reported series rather than one.

Deaths are **never** a core channel. The transfer view returns exactly the four core channels for every
disease, so the shared encoder cannot exploit a deaths channel that exists for Ebola alone to infer
which disease it is observing. The remaining five categories are never summed into the target — a
property pinned by a test, because the only thing that currently prevents it is a string comparison,
which is too quiet a guarantee to leave unasserted.

### 3.4 Two defects in the cumulative-to-weekly conversion

**This is the substantive correctness finding for this dataset.** Both defects lived in the same
function. The first was found and corrected during Phase 2; the second survived that correction, was
documented in this note as a virtue, and was caught only in adversarial review.

#### 3.4.1 The clip, which fabricated 36 per cent of the target

The conversion ended in `max(0, ΔC)` — difference the cumulative series, then clip negative
increments to zero. The justification recorded in §3.2 was that a downward step is reporting
reconciliation rather than negative incidence.

**A cumulative count cannot fall. Where the report falls, the report is wrong** — and it is
overwhelmingly a single-week data-entry dropout, not a revision. Western Area Urban, the largest
district of the epidemic, taken verbatim from the source file:

| Week | Cumulative | Difference | Released as new cases |
|---|---|---|---|
| 2015-01-24 | 2,612 | +43 | 43 |
| 2015-01-31 | 613 | −1,999 | **0** (clipped; recorded as an observed zero) |
| 2015-02-07 | 631 | +18 | 18 |
| 2015-02-14 | 2,741 | **+2,110** | **2,110 in a single week** |

The 613 is a dropout: the series snaps straight back to trend. The clip zeroed the fall and then
released the *recovery* — the climb back to a level already counted — as new incidence. The lost mass
reappeared in full as a phantom spike.

Measured on the affected build:

| | |
|---|---|
| Fabricated cases | **+8,786 (+35.8 per cent)**: 33,338 released against 24,552 real, across 34 affected districts |
| The six largest cells in the entire target | all clip artefacts |
| Worst district | Western Area Urban carried 7,462 cases against the 3,017 it ever recorded — a factor of 2.5 |
| National peak of the released target | **2015-02-14 at 4,982 cases/week** |
| Actual peak of the West African epidemic | **late 2014**, and in steep decline by February 2015 |

The epidemic peak was displaced by roughly three months and inflated several-fold. Every Ebola
number in earlier versions of this note was computed on that target.

**Correction: the conversion now differences the running maximum of the cumulative series** rather
than clipping the difference. The monotone envelope absorbs a dropout — the affected week yields no
increment, and the recovery yields only the rise above the previous high-water mark, instead of
re-releasing mass that was already counted. Weeks whose report falls below the running maximum are
**masked**, because they are corrupt reports rather than observations of zero; previously they were
scored and normalised on as though a district had genuinely recorded no cases.

The correction is mass-conserving by construction: the weekly increments now sum to
`max(C) − C_first` per district, exactly. On the released build the target totals **24,552 against a
reference of 24,552**, and the national peak sits at **2014-10-25 (2,688 cases/week)**.

**Why no check caught it.** The suite compared the transform against nothing the transform had not
itself produced. A `cumulative_reference_mass()` function now computes the implied mass directly from
the source column — sharing only the weekly resampling with the transform, and none of the increment
logic — and a gate fails the build if the weekly series sums to more than that reference. The gate
has a negative control that fabricates incidence and requires it to fail (§4.3).

#### 3.4.2 The week-zero backlog

The conversion also assigned each district's **entire cumulative total to date** as its week-zero
incidence. Measured against the production file:

| | |
|---|---|
| Share of *all* Ebola incidence sitting in week zero | **4.1 per cent** (1,457 of 35,670 cases) |
| Districts whose week-zero value was their all-time maximum weekly value | **10 of 64** |
| Worst case | one district recorded 1,062 against a median weekly increment of 41 — a factor of 26 |
| Districts whose week-zero value exceeded their own median increment | 25 of 64 (7 by more than tenfold) |

*(These figures were measured before the §3.4.1 correction and are quoted against the target as it
then stood — hence the 35,670 denominator. They are retained because they are what motivated the
week-zero correction; they are not properties of the released dataset.)*

**This is not a late-joining-district problem.** The compilation itself opens on 2014-03-24, months
into an outbreak that began in December 2013, so *every* district's first report is a backlog —
including the index districts.

The root cause is **identifiability, not tuning**: at a district's first report the cumulative total is
known but its predecessor is not, so the increment *does not exist*. Assigning the cumulative total
fabricates it.

**Correction: the first observed week of every district is masked.** The observation mask exists for
exactly this case — a cell that is not observable. The cost is 61 cells, one per district.

**The correction also closed a leak that had not been looked for.** Under the original code, week zero
was the *first support week for every district*. Since the Ebola scaler is fitted on the pooled support
cells, those fabricated backlogs were the **dominant input to the normalisation of the entire held-out
disease**. Masking week zero removes the problem at its source.

#### 3.4.3 Gap-lumping — an acknowledged limitation, not corrected

Differencing a cumulative series recovers *how many* cases accrued between two reports, but not *when*
within that interval they occurred. Where a district reports every week the two coincide. Where it
falls silent and then files, the entire multi-week increment is stamped on the single week the report
arrived.

| | |
|---|---|
| Intervals between consecutive observed weeks that exceed one week | **7 per cent** |
| Longest gap | 24 weeks |
| Clearest instance | Montserrado, 1,428 cases stamped on the week of 2014-10-25 |

**No cases are invented** — the mass gate of §3.4.1 holds exactly — and the epidemic curve is
correctly *shaped*, peaking in late October 2014 where the epidemic genuinely peaked. But roughly one
increment in fourteen carries more than one week of cases on a single week's timestamp, which
inflates the peaks and flattens the weeks either side.

**Any per-week magnitude claim on Ebola inherits this.** The available remedies — distributing an
increment across the interval it spans, or scoring only on contiguous runs of observed weeks — both
change the target. **The decision (client, A4) is to disclose and proceed:** the shape is correct, and
both remedies cost more than the distortion is worth (one invents an unobserved within-gap
distribution, the other discards evaluation data). It stands as a stated limitation of the released
dataset, to be repeated wherever a weekly Ebola magnitude is quoted.

### 3.5 Spatial units

**Sierra Leone reports seventeen district labels, but the country has only fourteen districts.** The
three additional labels are not districts.

| | Raw labels | Dataset nodes |
|---|---|---|
| Guinea (prefectures) | 32 | 32 |
| Liberia (counties) | 15 | 15 |
| Sierra Leone (districts) | 17 | 14 |
| **Total** | **64** | **61** |

Both figures are correct: 64 counts raw *labels*, and 61 counts nodes after cleaning. The audit script
prints the reconciliation so that the two are never read as contradicting one another.

**Administrative levels are mixed across countries.** Liberia's fifteen counties sit at the first
administrative level of the shapefile, whereas Guinea's prefectures and Sierra Leone's districts sit at
the second. The Ebola dataset is therefore mixed-level, exactly as dengue is. District names are unique
*within* each country, so the node key needs no parent (unlike dengue's Admin-2 nodes). Note that Guinea
contains a genuine prefecture named *Mali*; the country prefix in the node key is what prevents it from
colliding with the country of the same name.

The dataset comprises **61 districts over 52 weekly steps** (2014-04-05 → 2015-03-28), with a **mask
density of 0.4095** (1,299 observed cells of 3,172).

The observed-cell count fell from 1,667 to 1,299 with the §3.4.1 correction. The 368 cells removed
are the weeks whose cumulative report fell below the running maximum: corrupt reports, previously
scored and normalised on as though they were observed zeros. This is a genuine reduction in Ebola
evaluation data and is the honest count.

### 3.6 Few-shot protocol

> **Superseded for scoring, 2026-08-07 (decision D16).** The cutoff of 2014-05-24 described below is
> the Phase-2 build and remains what `build_datasets.py` produces as `data/processed/ebola.npz`. The
> Ebola case study is scored against **two frozen arms** instead, cut at **2014-06-28** (primary) and
> **2014-08-23** (secondary). See §3.6a. Everything in this section about *why* the support set is a
> calendar prefix, how the scaler is fitted, and what the gates check applies unchanged to both arms;
> only the cutoff date moves.

The support set is a **calendar prefix**: every observed cell dated on or before a fixed cutoff is
support, and every later observed cell is query. The **first two *observed* weeks per district** was
the earlier scheme, retained only for comparison; see "Why a calendar prefix" below. Because week
zero is masked, the support set comprises *genuine increments* and can never contain the fabricated
backlog.

Under the Phase-2 cutoff of **2014-05-24**:

| | |
|---|---|
| Support | 27 cells (the 9 districts that had reported by 2014-05-24) |
| Query | 1,272 cells |
| Districts with support | 9 of 61 |
| Districts evaluated but with **no** support (pure zero-shot) | 52 of 61 |
| Districts never evaluated | 0 |
| Scaler | fitted on the **pooled support cells only** |

The scaler is pooled across districts rather than fitted per node because a handful of early cells is
far too few to estimate a stable per-node scale. Fitting on anything beyond the support set would leak
the outbreak's magnitude into an ostensibly few-shot result — the single sharpest leakage risk in the
project. The support and query sets are disjoint and partition the observed cells; both are gated at
build time.

That 52 of 61 districts are zero-shot is not a defect. It is the genuine emerging-outbreak regime: the
model must forecast a district it has never observed, from the graph and the few districts that reported
early. It is arguably the paper's central contribution.

**Why a calendar prefix, and what it replaced.** An earlier build used the first two *observed* weeks of
**each district** as its support. Because districts enter the record at widely different dates, that was
not calendar-causal: support cells spanned 2014-04-12 to 2015-02-14, and **1,004 of 1,178 query cells
(85 per cent) lay earlier in calendar time than the last support cell**. The pooled scaler normalising an
early district's week-5 cell was therefore a function of another district's week-45, peak-epidemic
magnitudes — the model was handed a hint about how large the outbreak would become.

**That defect was invisible to the suite, structurally.** The old fit-set check *defined* the support
mask as the legitimate fit set, so the sharpest gate was conditional on precisely the assumption under
attack. The calendar-prefix build closes it, and a **new gate now checks calendar-causality directly** —
every support cell is dated on or before every query cell — with a negative control that plants an
acausal support cell and requires the gate to fail (§4.3). The property that could not previously be
tested is now the thing the gate tests.

**The leakage gate** is checked on every build: the scaler is exactly the pooled mean over the support
cells, and inflating a *query* week a thousandfold does not move it.

**A methodological trap worth recording.** This property **cannot** be probed by perturbing week zero.
The series is *cumulative*, so altering the initial value also alters the week-one increment — the
perturbation is not local, and the test fails for the wrong reason. The perturbation must be applied to
a query week, which moves one increment and nothing else.

### 3.6a Support length: the two frozen arms (2026-08-07)

The Phase-2 cutoff of 2014-05-24 gives 27 support cells across 9 districts, which yields **0
adaptation pairs at h10 and h15** and only 6 at h5 across 2 districts. A support-length sweep
(`Reports/Ebola_Support_Set_Decision.md`) put the trade-off to the client without a recommendation:
keep the strongest few-shot framing and accept that the far horizons are permanently zero-shot, or
lengthen the support set and accept that the headline becomes moderate-data transfer.

**The client's decision (D16) was to run both, as two pre-registered arms**, frozen and hashed before
either is scored.

| | **primary** | **secondary** | dropped |
|---|---|---|---|
| label (see the note below) | L12 | L20 | L19 |
| cutoff, support = observed cells on or before | **2014-06-28** | **2014-08-23** | 2014-08-16 |
| support columns | 13 | 21 | 20 |
| support cells | 59 | 113 | 79 |
| districts with support | 18 of 61 | 36 of 61 | 22 of 61 |
| districts with no support (pure zero-shot) | 43 of 61 | 25 of 61 | |
| query cells | 1,240 | 1,186 | |
| adaptation pairs, h3 | 48 / 17d | 102 / 36d | |
| adaptation pairs, h5 | 38 / 14d | 92 / 36d | |
| adaptation pairs, h10 | 18 / 9d | 72 / 35d | |
| adaptation pairs, h15 | **0 / 0d** | 54 / 34d | |
| scaler, pooled log1p+z on support | mu 1.4402, sd 1.2145 | mu 1.7913, sd 1.5286 | |
| implied "typical week" | 3.2 cases | 5.0 cases | |
| query cells beyond 1 sd of that scaler | 333 / 1,240 (27%) | 192 / 1,186 (16%) | |
| artifact | `data/processed/ebola_L12.npz` | `data/processed/ebola_L20.npz` | not built |
| content sha256 | `08d657dc…` | `e9b9ac0b…` | |

An adaptation pair at horizon h needs a *support* target at column t+h, so it exists only where a
support cell sits at a column index of at least h. Input windows are left-padded, so the origin
itself never limits the count. The primary arm's h15 count is 0 by arithmetic, not by data quality:
support reaches column 12 and h15 needs a target at column 15.

**The query counts are identical under both arms.** A full-window query target sits at column 22 and
support reaches at most column 20, so no option below L=20 costs a single forecast. Query *cells*
fall (1,272 → 1,240 → 1,186) but those cells were never scoreable targets. The arms therefore differ
only in how much labelled adaptation data they carry, never in what is evaluated. Asserted by the
freeze script, not assumed.

**Two different counts, and only one is the evaluation size.**

| | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| **scored** pairs / districts | **757** / 57 | **766** / 57 | **765** / 59 | **642** / 58 |
| *reachable* pairs / districts | *1,151* / 61 | *1,075* / 61 | *866* / 59 | *642* / 58 |

*Reachable* counts per-horizon origins: every origin with a full window whose target lands inside the
panel, so h3 reaches origin 48 and h15 only 36. It is what §3.5 above and
`Reports/Ebola_Support_Set_Decision.md` quote. *Scored* uses **one common origin set for every
horizon**, t in [19, 36], 18 origins, which is what `bundles.origins()` returns and therefore what
`score_predictions` evaluates. Every other dataset in this project was scored that way, and it is
the right protocol: horizons are comparable to each other only if read at the same origins. The two
agree at h15, whose reach is the binding constraint; **quoting the reachable figures as the
evaluation size overstates h3/h5/h10 by 34 to 52 per cent.** Recorded as pre-registration amendment
A1, found by reading the scoring path before the run rather than after it.

**Scale calibration improves with support length and is still not good.** The mean scored cell is
about 19 cases under either arm, against an implied typical week of 3.2 (primary) and 5.0
(secondary); the Phase-2 build implied 2.1. The FiLM adapter is the component meant to correct this,
and under the primary arm it has 18 examples at h10 and none at h15. Any primary-arm h10/h15 number
is a zero-shot number and is labelled as such.

**The labels are off by one against the column counts.** The sweep table indexes outbreak weeks from
the raw first week 2014-03-24, whose incidence cell is masked. "L" is the 0-based index of the last
support column, not a count: L12 is 13 columns, L20 is 21. **The cutoff date is the operative
definition**, and the label is carried only so the arms match the document the decision was made
from.

Both arms are built from the raw xlsx in one pass by `freeze_ebola_arms.py`, which re-derives every
count in the table above and refuses to write if any has moved. It also re-checks the §3.6 properties
per arm: the masks partition the observed cells, no query cell precedes the last support cell, and
the scaler is exactly the pooled log1p mean and sd over the support cells. Manifest and hashes:
`configs/ebola_arms.json`. Re-verify at any time with `python freeze_ebola_arms.py --verify`.

### 3.7 Geographic graph

**The Ebola graph is deliberately *not* block-diagonal.** Cross-border edges are retained.

The dengue graph forbids cross-border edges because its countries are not mutually adjacent. **That
rationale does not transfer here.** Guinea, Liberia and Sierra Leone are physically contiguous, and the
2014 epidemic was a single outbreak that spread across them: the Guéckédou–Lofa–Kailahun tri-border area
is the defining transmission pathway of the epidemic. A block-diagonal graph would sever precisely the
edges that carry the epidemiological signal. This is a modelling error, not a convention choice.

Contiguity is therefore built over the union of all 61 districts:

```
61 nodes · 146 undirected edges · mean degree 4.79
0 unmatched · 0 isolated · 21 cross-border edges · symmetric · not block-diagonal
```

**All three tri-border edges are present and asserted on every build**; the build fails if any is
missing. The remaining eighteen cross-border edges are all geographically genuine (Forécariah–Kambia,
Nzérékoré–Nimba, Grand Cape Mount–Pujehun, and others) and are enumerated in the dataset metadata.

The diagonal is zero, consistent with the influenza convention, so that no disease reaches the encoder
with a different self-weight.

#### The name join

The expected difficulty was accents (*Gueckedou* against *Guéckédou*, and similar). These require no
handling: canonicalisation already strips diacritics. The aliases actually required address a different
problem — **two misspellings in the shapefile itself**.

| Node | Shapefile ships | |
|---|---|---|
| Guinea, Yomou | *Yamou* | a shapefile error; Guinea has no prefecture of that name |
| Liberia, Gbarpolu | *Gbapolu* | a shapefile error |
| Sierra Leone, Western Area Urban / Rural | *Western Urban* / *Western Rural* | the shapefile omits "Area" |

**The shapefile's defects are absorbed at the *join* and never rename a node.** Two separate maps do two
separate jobs: a load-time map repairs the *data* (the district merge described below, and Liberia's
inconsistent county suffix), and a join-time map absorbs the *shapefile's* quirks. This is the
separation that dengue initially lacked and now shares (Section 1.6.1).

**Unmatched districts halt construction.** Any district without a polygon raises an error carrying the
full diagnostic; the builder never discards a node.

### 3.8 Alterations to the data

**The first observed week of every district is masked.** Documented in full in Section 3.4. The
alternative — retaining the cumulative total as week-zero incidence — was rejected: it fabricates
4.1 per cent of all incidence, hands ten of sixty-four districts a false all-time maximum, and feeds
the fabrication directly into the support scaler. The cost is 61 observed cells. No residual risk is
identified: the quantity removed was never identifiable in the first place.

**`Port` merged into `Port Loko`: one district recorded under two labels.** `Port Loko` reports through
2014-11-26; `Port` begins on 2014-11-27 — the very next day — and continues the *same cumulative series*
(1,041 rising to 1,923) to the end of the file. The label was evidently truncated part-way through
compilation.
*Decision:* merge at load time. The two eras are strictly disjoint and the cumulative series is
continuous across the seam, so the merge is unambiguous.
*Consequence:* one continuous 35-week series — among Sierra Leone's best-covered — in place of two half
series, each of which would otherwise carry its own scaler and a fabricated backlog at the seam. This
merge is also what dissolves the worst week-zero anomaly in the file: the value of 1,062 noted in
Section 3.4 was never a backlog at all, but Port Loko's inherited cumulative total.

**Two Sierra Leonean units dropped, because they are not districts.**
*Western Area* is the **parent** of Western Area Urban and Western Area Rural, and it **overlaps both in
time**: its three reports run from June to October 2014, while its children begin on 2014-08-15.
Retaining all three would triple-count the same cases. The parent is dropped; the two children have
thirty-one and thirty-two observed weeks respectively against the parent's three, so the finer pair is
strictly preferable.
*Freetown* is a **city inside Western Area Urban**, not a district: one report, five cases, and no
district polygon exists for it. It is dropped. Folding its five cases into Western Area Urban was
considered and rejected, because it cannot be verified whether that district's cumulative total already
included them, and the magnitude (0.014 per cent of all cases) does not justify the risk of
double-counting.

Both removals are declared with their reasons and recorded in the dataset metadata. **The loader
discards nothing else.**

**The graph is not block-diagonal.** Documented in Section 3.7. The residual risk is that this departs
from the dengue convention, so the two graphs are built under different rules. The difference is
principled — adjacent against non-adjacent countries; one epidemic against many — and is stated in the
methods rather than concealed.

**Liberia's county suffix is normalised.** The source is internally inconsistent: eleven of the fifteen
counties carry a "County" suffix and four do not. All are normalised to the bare name at load time. This
is a naming correction, not a change to the data; no counts move.

**A refusal to change silently.** Multi-district aggregate labels — rows whose district field names
several districts at once, as an early-outbreak aggregate — are removed only from an explicitly declared
list. They are, however, *also* detected structurally, and an **undeclared** aggregate label **raises an
error**. A refreshed file that introduced a new aggregate label would therefore halt the build and ask a
human, rather than quietly ingesting a three-district aggregate as though it were a single place.

### 3.9 Static covariates

Each node carries centroid and area, on the same contract and from the same provenance as dengue and
influenza. Rows align to node identifiers by construction: the graph builder already holds the polygons
reindexed into node order, so the covariates fall out of the same pass rather than being re-derived.

**All three diseases now carry populated covariates. This does not make them transfer-safe.** See
Section 4.2.

### 3.10 Known limitations

- **Every district is scored, but 52 of 61 are zero-shot.** Under the calendar-prefix support (§3.6)
  all 61 districts have at least one query cell; `meta['nodes_without_query']` is empty. What replaces
  the old "unscored district" caveat is that 52 districts report only after the support cutoff and so
  carry *no support cell* — they are evaluated zero-shot, normalised by the pooled support scaler. This
  is the genuine emerging-outbreak regime, not a defect.
- **Four districts report no incidence at all.** Four Guinean prefectures reported for between six and
  forty-one weeks but never recorded a *new* case: their one to three cases predate their first report
  and therefore sit in the masked backlog. This is correct, not a defect. They are trivially predictable
  and are retained, being real districts and valid neighbours. Unlike the Japanese zero-variance nodes
  of Section 1.7, their scaler **cannot** degenerate, because the Ebola scaler is pooled across the
  disease rather than fitted per node.
- **Contiguity is an induced subgraph.** Guinea reports thirty-two of its thirty-four prefectures, and
  edges exist only between districts present in the data, so degree is a property of coverage as well as
  of geography.
- **Imputed weeks.** At a mask density of 0.5255, Ebola contributes by far the most imputed steps of the
  three diseases. Internal reporting gaps are forward-filled in *cumulative* space and flagged as
  missing. A district that ceases reporting is marked missing rather than carrying a flat cumulative
  total indefinitely.

---

## 4. Harmonisation and leakage

### 4.1 The released datasets

Five datasets, three diseases, one weekly schema. All are regenerated by a single command
([build_datasets.py](build_datasets.py)).

| Dataset | Role | Nodes | Weeks | Mask density | Split | Scaler fitted on |
|---|---|---|---|---|---|---|
| Dengue | development | 7,165 | 1,409 | 0.2175 | per-country chronological 50/20/30 | per node, own-country training cells |
| Influenza (Japan) | development | 47 | 348 | 1.000 | fixed 50/20/30 | per node, training slice |
| Influenza (US regions) | development | 10 | 785 | 1.000 | fixed 50/20/30 | per node, training slice |
| Influenza (US states) | development | 49 | 360 | 1.000 | fixed 50/20/30 | per node, training slice |
| Ebola | **few-shot holdout** | 61 | 52 | 0.5255 | support / query | pooled support cells |

### 4.2 Disease-agnosticism

The framework's claim to be disease-agnostic reduces, at the data layer, to a single rule: **only the
core, transfer-safe channels are exposed to the shared encoder.**

The transfer view returns exactly four channels — normalised incidence, two seasonality channels, and
the observation mask — in the same order and at the same indices, for all five datasets. This is
asserted across diseases in the test suite rather than merely intended.

Three things are deliberately excluded, and each exclusion is a correctness property rather than a
matter of tidiness:

| Excluded | Why its inclusion would disclose the disease's identity |
|---|---|
| The Ebola deaths channel | It exists for **one** disease only. The encoder could infer *which* disease it is observing from the channel's presence or absence alone. |
| The static covariates | The three diseases occupy **disjoint geography** — dengue in Latin America and Asia, influenza in the United States and Japan, Ebola in West Africa. A raw centroid therefore **identifies the disease outright**. |
| Adjacency self-loops | One shipped convention against one built convention would give influenza twice the self-weight of dengue under the standard renormalisation. |

**A correction concerning the static covariates.** An earlier version of this note gave the reason for
excluding them as *asymmetric availability* — a real covariate matrix for one disease against an
all-zero matrix for another would itself be a perfect disclosure of disease identity — and stated that
the constraint could be lifted once all three diseases carried populated covariates. All three now do,
**and the constraint still stands**, because that reasoning was incomplete. The asymmetry is gone, but a
stronger disclosure was present all along: the diseases occupy disjoint geography, so the coordinates
identify the disease directly. Populating the covariates cannot fix this; it is intrinsic to what they
are. Should they be wanted as a shared input, they must first be made geography-free — for instance, a
centroid expressed relative to its own country's centroid, or area alone. Single-disease models may use
them freely at any time.

### 4.3 The leakage suite

The leakage and invariant suite ([test_leakage.py](test_leakage.py)) comprises **86 pass/fail gates**
across the five datasets. They gate the build: the packaging step **writes nothing** if any gate fails,
on the principle that a dataset with a leak, once written, is a dataset somebody will train on.

| Gate group | What it establishes |
|---|---|
| Scaler provenance | The released scaler **reproduces from its declared fit cells alone**, and does not move when held-out cells are inflated a thousandfold. |
| No future leakage | The incidence channel at time *t* is the target at time *t*, with no shift; no channel equals the target at *t + h* for any tested horizon; the seasonality channels derive from the **calendar**, not from the series. |
| Mask semantics | Imputed cells take the value zero in *normalised* space — the per-node mean, a neutral value, and *not* the transform of a raw zero, which would assert that no cases occurred — carry a missing flag, and are excluded from every reduction. |
| Split integrity | For the development diseases, the three partitions divide the observed cells disjointly and every node has a defined scaler — its own training cells, or its country's. For Ebola, support and query are disjoint, both are subsets of the observed cells, and together they exhaust them. |
| **Mass conservation** | A weekly series derived from a cumulative one sums to no more than `max(C) − C_first`, where that reference is computed **from the source column by an independent routine**, not by the transform under test (§3.4.1). |
| **Phase purity** | No `(country, week)` column holds cells from more than one split phase, so a training cell never sits beside a test cell in a neighbourhood a graph network aggregates over (§1.8). |
| **Calendar causality** (few-shot) | Every Ebola support cell is dated on or before every query cell, computed from the cells' dates rather than from the mask that defines them — the property the earlier suite structurally could not test (§3.6). |
| Rolling origin | Each origin's training window lies strictly *before* its evaluation window, evaluation never touches an imputed cell, and the window expands monotonically. |
| Disease-agnosticism | Section 4.2, checked across all five datasets. |

**On the design of these gates, and two defects they were too weak to catch.**

The scaler gates are constructed as *probes*: a cell that the scaler must not be able to see is
corrupted, the scaler is refitted, and the gate requires that nothing moves. This design is deliberate,
because a gate that cannot fail establishes nothing.

That precaution proved necessary twice.

The **first** version of the suite recovered the raw counts by inverting the normalised series *using
the very scaler it was about to test*. This is circular: refitting on such a reconstruction returns the
same scaler whether or not that scaler leaked, because the reconstruction already carries the leak. The
flaw was exposed by planting a real leak — fitting the Ebola scaler on the support **and query** cells —
whereupon **the suite passed it, with all gates green**. The correction is that each dataset now carries
its **raw, unscaled incidence**, which no scaler has touched, and every scaler gate is evaluated against
that.

The **second** failure was more serious, because the suite was not circular but simply silent. A suite
of 83 gates passed a build in which 36 per cent of the Ebola target was fabricated (§3.4.1) and 31 per
cent of dengue's observed cells sat in mixed-phase columns (§1.8). Neither defect was reachable by any
gate then present: there was no check that compared the weekly transform against anything the transform
had not itself produced, and no check that the split honoured the invariant this note asserted for it.
Both defects were found in adversarial review, not by the suite. **The lesson is recorded here because
the count of gates is not a measure of their coverage**, and a green suite is evidence only about the
properties it actually tests.

**Six negative controls** now run on every build, each planting a real defect and requiring the
corresponding gate to fail: a development scaler fitted on all cells; the Ebola scaler fitted on support
and query; the target planted in an input channel; weekly incidence inflated past its cumulative
reference; a training cell placed inside a test column; and a support cell dated after a query cell. All
six fail, as they must. The last of these tests the calendar-causality gate that closes the acausal-
support defect of §3.6 — the very property the earlier suite could not detect, because it defined the
support mask as legitimate by construction rather than checking its dates.

### 4.4 Rolling-origin backtest scaffold

The evaluation protocol pairs the fixed split (the headline result) with an **expanding-window,
single-step rolling-origin backtest** (five origins) as a robustness check, with **the scaler and graph
refitted at each origin**. The backtest itself is run during the modelling phase; the origins are fixed
now. That ordering is deliberate: deferred, the instruction to refit at each origin is exactly the step
that becomes "fit once and reuse", which leaks the test distribution into all five origins simultaneously.

- **Only the cut points are stored** — one integer per group per origin — and the masks are expanded on
  demand. Materialising five mask pairs for dengue would cost hundreds of megabytes to encode what one
  integer implies.
- **Grouping mirrors the headline split**, or the origins would contradict it: per country for dengue,
  since a single global origin on the union calendar would fall *before* one country's first report and
  *after* another's last; a single shared sequence for influenza.
- **Ebola has no rolling origins, deliberately.** The construction is meaningless for a held-out disease
  whose protocol *is* the calendar-prefix support set: expanding the training window beyond support is
  exactly the leakage the few-shot design forbids. Its absence is a decision, and the suite asserts it.
- **The raw, unscaled incidence is released with every dataset.** This is not a convenience: a refit
  requires the scaler's *input*. Without it the backtest could not refit at each origin even in
  principle, and would fall back on the headline scaler — which has seen every origin's future.

### 4.5 Reproducibility

The build verifies every input by checksum — the dengue extract, the Ebola compilation, the six
influenza matrices, and all sixteen shapefiles — then constructs the five datasets, runs the full
leakage suite together with its negative controls, and only then writes the released files, alongside a
record of the configuration and the frozen environment.

A drifted input would silently change every number in this document, **and the datasets would still pass
their schema checks**. Verification therefore *stops* the build rather than warning.

**Packaging.** Each dataset is released as a compressed array archive rather than as a serialised Python
object. The latter would be more convenient and is the wrong artefact: it couples the file to the exact
class definitions and library versions that wrote it, and deserialising it executes code, which makes it
unsafe to hand to a reviewer. The archive contains plain arrays and a plain-text metadata record.

**Determinism is verified, not assumed.** Node and date order are the schema's single source of truth
and are fixed by the loaders. Two consecutive builds produce identical arrays for all five datasets.

### 4.6 Constraints carried into the modelling phase

Three constraints are properties of the datasets and cannot be enforced from within them.

1. **The encoder must add the identity matrix** before any degree normalisation, or the four isolated
   influenza nodes divide by zero.
2. **The static covariates must not enter the shared encoder**, since geography identifies the disease
   (Section 4.2). Single-disease models may use them freely.
3. **The scaler must be refitted at every rolling origin.** The released scaler belongs to the headline
   split and has seen data beyond every origin. The raw incidence is released precisely so that refitting
   is possible.

---

## 5. Summary of decisions

### 5.1 Principal decisions

| Question | Resolution |
|---|---|
| Dengue spatial level | Finest available per country (Admin-1 or Admin-2); a national aggregate is never a node |
| Dengue coverage thresholds | At least 52 observed weeks per node; at least 3 such nodes per country; pruning at country level only |
| Influenza scope | The three benchmark datasets only; a global country-level surveillance source was declined, being neither subnational nor the benchmark |
| Ebola support window | **Primary:** calendar prefix — every observation on or before 2014-05-24 (§3.6); a variable window size is retained as a parameter for comparability, but the calendar cutoff is the primary support set |
| Shapefile provenance | GADM 4.1 throughout, for every graph and every covariate |
| Development set | Dengue and influenza. COVID-19 entered the reproduction study only and enters no released dataset |
| Static covariates | Populated for all three diseases; **not** transfer-safe, and excluded from the shared encoder |
| Mobility graph | Declined for all diseases |

### 5.2 Where the data were changed

| Disease | Change |
|---|---|
| Dengue | Taiwan's townships aggregated to counties; seven post-vintage units folded into their parents; two Nicaraguan health districts summed into one department; four split series reunited; one mislabelled node dropped; two ambiguous names resolved from evidence; shapefile misspellings absorbed at the join; a source homoglyph mapped; the island fallback symmetrised |
| Influenza | The adjacency diagonal zeroed; both US date anchors corrected by thirty-nine weeks (the matrices identified as CDC ILINet at a known vintage, §2.3) |
| Ebola | **Weekly incidence taken from the running maximum of the cumulative series, and the weeks that fall below it masked (§3.4.1)**; the first observed week of every district masked; **the few-shot support set made a calendar prefix (cutoff 2014-05-24), replacing the acausal per-district scheme (§3.6)**; one district reunited across two labels; two non-districts dropped; the graph built with cross-border edges; the shapefile's misspellings absorbed at the join; Liberia's county suffix normalised |

### 5.3 Corrections to earlier releases of this note

Recorded so that a figure quoted from an earlier draft can be identified as superseded.

| Claim in an earlier draft | Corrected |
|---|---|
| Downward steps in the Ebola series justify clipping negative increments to zero | **False.** They are data-entry dropouts, and the clip fabricated 35.8 per cent of the target (§3.4.1) |
| The clip fabricated "8,612 cases; 33,164 released" | **8,786 cases; 33,338 released.** 33,164 was derived by adding a gate figure to the reference, conflating two quantities — the gate's 8,612 came from an earlier `cumulative_reference_mass()` that baselined on the first *daily* report rather than the first *weekly* value |
| Ebola: 1,667 observed cells, mask density 0.5255 | 1,299 observed cells, density 0.4095 — the corrupt weeks are now masked (§3.5). Query-cell count depends on the support scheme: 1,272 under the calendar prefix (§3.6) |
| Ebola: one (later: two) districts have no query week | **Zero** under the calendar-prefix support — all 61 are scored (§3.6); 52 are zero-shot |
| Thirty zero-variance dengue nodes | **400**, of which 76 are guard mis-fires that leave the node un-normalised (§1.7) |
| At any week an entire country block is in a single phase | Was **false** on the affected build: 792 mixed-phase columns, 31 per cent of observed cells. Now true, and gated (§1.8) |
| The country-macro metric neutralises Brazilian dominance | **Now true:** the metric is implemented and self-tested (§1.7, [score.py](score.py)); it operates on model predictions, so the datasets carry no scores themselves |
| The dengue countries are not adjacent, justifying a block-diagonal graph | **False.** Brazil borders four of them (§1.11) |
| The *new cases* series is too sparse to use | It carries 95 per cent of the cumulative series' coverage (§3.3) |
| Dengue case definitions: Peru ×220, DR ×48, Mexico ×13, Bolivia ×11, Panama ×7 | **Raw-file artefacts.** Measured per node on the extracted data: Peru and DR are single-definition (no change); the real steps are Mexico 2.8× and Bolivia 1.6× on 41 nodes (§1.10) |
| The Ebola support set is not calendar-causal (85% of query precedes last support) | **Resolved.** Calendar-prefix support, cutoff 2014-05-24; now gated (§3.6) |
| The 76 mis-firing zero-variance nodes ship un-normalised | **Resolved.** Country-pooled fallback; 0 nodes now un-normalised (§1.7) |
| The suite has 83 gates / five negative controls | 86 gates, six negative controls (§4.3) |

### 5.4 Decisions on the properties that affect interpretation

All four Part-A items are settled. None remains a blocker.

1. **Ebola gap-lumping** (§3.4.3) — **decided: disclose and proceed.** The curve's shape is correct;
   only the within-gap timing of a minority of cases is uncertain. Any weekly Ebola magnitude must be
   quoted with this stated.
2. **Dengue case definitions** (§1.10) — **decided: kept as built, disclosed.** The exposure measured
   far smaller than first reported (a 2.8×/1.6× step on 41 nodes, not the ×220 group-by figure).
3. **Ebola few-shot causality** (§3.6) — **resolved** by the calendar-prefix protocol, now gated.
4. **Dengue spatial resolution** (finest-available per country) — **acknowledged and approved** (client decision A1).

The two items an earlier draft listed as *blocking* — the case-definition change and the acausal
support set — are both resolved: the first by measurement, the second by a protocol change enforced by
a gate.

**Modelling-phase choices, now settled** — recorded so none is rediscovered as open once the modelling
work begins:

5. **Country-macro metric** (§1.7) — implemented and self-tested ([score.py](score.py)); the primary
   dengue headline, with an equal-node average reported alongside.
6. **Constant-node scoring** (§1.7) — genuinely-constant nodes are excluded from scoring and kept in
   the graph, with Japan-without-them reported as a sensitivity.
7. **Brazilian dominance** (§1.11) — uniform node sampling is the primary training regime (the encoder
   is data-proportional); a country-balanced sampler is provided as an ablation, and which is better is
   decided empirically rather than by decree.
8. **Japan calendar** (§2.3) — pinned to the week against NIID's dated season peaks
   ([japan_calendar_pin.py](japan_calendar_pin.py)).
9. **Ebola *new cases* series** (§3.3) — adopted as a cross-check against the cumulative-derived target.
10. **Ebola support window** (§3.6) — the calendar-prefix cutoff 2014-05-24 is the primary support set;
    a variable window size is retained as a parameter for comparability.

The three constraints of §4.6 (the encoder must add the identity; the covariates must not reach the
shared encoder; the scaler must be refit at every rolling origin) carry into the modelling phase as
coded guarantees, not open decisions.

### 5.5 What was deliberately not done

The monthly dengue history was **not** interpolated to a weekly cadence. The three influenza benchmarks
were **not** concatenated. The shipped influenza edges were **not** supplemented, so isolated nodes
remain isolated. For influenza, no node was dropped, no series smoothed and no gap imputed. No COVID-19
series was admitted. For Ebola, the *new cases* series was not used as the target — it is instead
**adopted as a cross-check** (§3.3) — the case sub-series were not summed into it, and the four
zero-incidence districts were not dropped.

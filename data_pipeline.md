# Data Pipeline

**Emerging Disease Forecasting Framework — Phase 2**

This document describes how the harmonised datasets are regenerated, what the released files
contain, and which interfaces a downstream model is obliged to respect. It is the operational
companion to the data audit note, which records what the data *are* and every decision taken in
preparing them; where a justification is needed here, this document states the conclusion and
refers to the audit note for the evidence.

**Contents**

1. Regeneration
2. The released files
3. The schema contract
4. Leakage controls
5. Provenance and versioning
6. Name matching
7. Source files

---

## 1. Regeneration

The entire pipeline is reproduced by a single command:

```bash
conda run -n ebola python build_datasets.py
conda run -n ebola python build_datasets.py --check-deterministic   # rebuild and compare
```

From a fresh checkout of the repository:

```bash
git clone <repository> && cd Ebola-Research
conda env create -f environment.yml

# acquire the raw sources — see Section 5 for the URLs and checksums
python fetch_gadm.py         # the sixteen shapefiles, fetched and verified
#   ... place the dengue extract, the Ebola compilation and the six influenza
#       matrices under data/Final datasets/  (all four are publicly downloadable)

conda run -n ebola python build_datasets.py
```

**Every raw input is publicly downloadable**, and the build verifies each against a recorded
checksum before use, so the datasets can be reproduced end to end from the published sources
alone. Section 5 gives the source and the checksum for each.

The build performs four steps, and the order is deliberate.

1. **Verification.** Every raw input is checked against a recorded checksum: the dengue extract, the
   Ebola compilation, the six influenza matrices, and all sixteen shapefiles. A drifted input would
   change every figure reported in the audit note, and the datasets would nonetheless pass their own
   schema checks. Verification therefore *halts* the build rather than emitting a warning.
2. **Construction.** The five datasets are built.
3. **Leakage gates.** The full leakage suite is run, together with its negative controls. On any
   failure the build **writes nothing**. A dataset containing a leak, once written to disk, is a
   dataset that somebody will train on.
4. **Packaging.** The five datasets are written, alongside a record of the configuration used and a
   snapshot of the environment.

### Environments

Two environments are maintained, and the separation is intentional.

| Environment | Use |
|---|---|
| Base (Python 3.11) | The schema unit tests. These carry **no geospatial dependencies** and run in seconds. The test fixtures are plain text precisely so that this remains true. |
| Conda (`ebola`) | Everything that touches geometry: the loaders, the leakage suite, and the build. |

---

## 2. The released files

The build writes one archive per disease to `data/processed/`, together with the configuration and
environment records.

| Array | Shape | Contents |
|---|---|---|
| `X` | *N* × *T* × *F* | Features. Channels 0 to 3 are the core block and are identical for every disease. |
| `y` | *N* × *T* | The target: normalised weekly incidence. |
| `raw` | *N* × *T* | Weekly incidence, **unscaled** — the scaler's own input (see Section 4). |
| `M` | *N* × *T* | The observation mask: one where observed, zero where missing or imputed. |
| `A_geo` | *N* × *N* | The adjacency matrix. The diagonal is zero for every disease. |
| `C` | *N* × 3 | Static covariates: centroid latitude, centroid longitude, and area. |
| `split_*` | *N* × *T* | The split masks — training, validation and test, or support and query. |
| `scaler_mean`, `scaler_std` | — | The fitted normalisation parameters. |
| `meta_json` | — | Node identifiers, dates, provenance, and the decisions taken. |

**Node identifiers and dates are the single source of truth for ordering.** Row *i* of every array —
features, target, mask, covariates, adjacency, and every split mask — is the node named at position
*i* of the identifier list. Nothing downstream re-derives this correspondence.

### Choice of format

Each dataset is released as a compressed array archive rather than as a serialised Python object. The
latter would be more convenient and is the wrong artefact: it couples the file to the exact class
definitions and library versions that wrote it, so that renaming a field or upgrading a library
renders the file unreadable, and deserialising it executes code, which makes it unsafe to hand to a
reviewer. The archive holds plain arrays and a plain-text metadata record, and can be read by any
tool that reads arrays.

### Loading a dataset

```python
import json, numpy as np

z = np.load("data/processed/ebola.npz", allow_pickle=False)
meta = json.loads(str(z["meta_json"]))

X, y, M, raw = z["X"], z["y"], z["M"], z["raw"]
support, query = z["split_support_mask"], z["split_query_mask"]

core = X[:, :, meta["core_feature_idx"]]   # the only block the shared encoder may see
```

---

## 3. The schema contract

Every disease is represented on a weekly epidemiological calendar. The shared, cross-disease encoder
may see **only the transfer view**: the four core channels — normalised incidence, two seasonality
channels, and the observation mask — identical in name, order and index across all five datasets.
Everything else is *extended*: available to a single-disease model, and withheld from the shared one.

**This is a correctness property, not a matter of tidiness.** If a channel exists for one disease and
not for another, and it reaches the shared encoder, then the encoder can infer *which disease it is
observing* from the pattern of presence and absence alone — silently reintroducing the
disease-specificity that the framework exists to remove. Three consequences follow, and each is
enforced at the data layer:

- The Ebola deaths channel is extended, never core.
- The static covariates are excluded from the transfer view. The three diseases occupy disjoint
  geography, so a raw centroid identifies the disease outright; populating the covariates for all
  three diseases, which has been done, does **not** make them transfer-safe. A single-disease model
  may use them freely.
- The shipped influenza adjacency has its diagonal zeroed to match the built graphs, since one
  convention carrying self-loops and another not would give influenza twice the self-weight of dengue
  under the standard renormalisation.

The reasoning behind each is given in the audit note.

### Obligations on the model

Three requirements are properties of the datasets and cannot be enforced from within them.

1. **The encoder must add the identity matrix** before any degree normalisation. Four influenza nodes
   are isolated — Alaska, Hawaii, Hokkaidō and Okinawa — and without the identity their rows divide
   by zero. This requirement is recorded in each influenza dataset's metadata.
2. **The static covariates must not enter the shared encoder**, for the reason given above.
3. **The scaler must be refitted at each rolling origin.** The released scaler belongs to the headline
   split and has seen data beyond every origin. The raw incidence is released so that refitting is
   possible.

---

## 4. Leakage controls

The leakage and invariant suite ([test_leakage.py](test_leakage.py)) comprises 83 pass/fail gates
across the five datasets, and it blocks the build: nothing is written if any gate fails.

**Where each scaler is fitted.** This is the leakage-safety property, stated concretely:

| Disease | The scaler is fitted on |
|---|---|
| Dengue | each node's own country's **training** cells, per node |
| Influenza | the **training** slice, per node |
| Ebola | the **support cells only**, pooled across the disease |

The Ebola case is the sharpest. Two weeks per node is far too little data to estimate a stable
per-node scale, so the scaler is pooled; and fitting on anything beyond the support set would leak the
magnitude of the outbreak into a result presented as few-shot.

**On the design of the gates.** The scaler gates are constructed as *probes*: a cell the scaler must
not be able to see is corrupted, the scaler is refitted, and the gate requires that nothing moves. The
suite additionally ships with **negative controls** — planted leaks, which the gates are required to
catch. This is not a stylistic preference. An earlier version of the suite recovered the raw counts by
inverting the normalised series with the very scaler under test, which is circular, and it
consequently passed a deliberately leaked Ebola scaler with every gate green. Each dataset therefore
now carries its raw, unscaled incidence, untouched by any scaler, and every scaler gate is evaluated
against that. The full account is in the audit note.

### Rolling-origin backtest

The origins for the expanding-window backtest are fixed at build time, although the backtest itself is
run later. Only the cut points are stored — one integer per group per origin — and the masks are
expanded on demand; materialising the masks would cost hundreds of megabytes to encode what a single
integer implies. Ebola has no rolling origins, deliberately: expanding the training window beyond the
support set is precisely the leakage that the few-shot design forbids.

---

## 5. Provenance and versioning

**All four sources are publicly available.** Each is pinned by checksum and verified at the start of
every build, so a reader can confirm they hold the same bytes these results were computed from.

| Source | Where to obtain it | Pinned by |
|---|---|---|
| OpenDengue, best-spatial extract, V1.3 | Figshare, DOI [10.6084/m9.figshare.24259573](https://doi.org/10.6084/m9.figshare.24259573) | DOI and checksum |
| Influenza benchmarks (6 files) | The ColaGNN repository; verified byte-identical to the EpiGNN mirror | Checksum manifest |
| Ebola compilation | [HDX](https://data.humdata.org/dataset/rowca-ebola-cases), resource `9b68ab69-e0b3-4ff5-b2d8-90bf446acd52` | Checksum |
| GADM 4.1 (16 shapefiles) | Fetched automatically by [fetch_gadm.py](fetch_gadm.py) | Checksum manifest |

**Versioning differs across the four, and the differences are worth understanding.**

OpenDengue is a frozen, citable release, so its DOI identifies the exact data. The influenza matrices
are a vendored benchmark, stable and verified against an independent mirror. GADM 4.1 is likewise a
frozen release.

**The Ebola compilation is the only source with no version number**, so its checksum is its only
version identifier. In practice it is stable — the resource has not been modified upstream since
November 2015 and is served from a permanent URL — but the checksum is what would reveal a
substitution, and the signal-confirmation audit re-asserts every count on each run, so a change in the
file's contents would also fail loudly.

The recorded Ebola checksum is that of the file **as published**. An earlier working copy had been
opened and re-saved in a spreadsheet application, which rewrites the container without altering any
cell; its contents and the dataset built from it were verified identical, but its hash was not the
published one, and a reviewer hashing their own download would have obtained a mismatch. The pristine
resource is now the source of record.

**The shapefiles are fetched rather than redistributed.** GADM 4.1 is a frozen, publicly hosted
release, so a checksum manifest affords the same reproducibility guarantee at negligible storage cost,
and its licence forbids redistribution for commercial purposes in any case. Verification runs without
network access and exits non-zero on any mismatch, so it gates the build: a re-cut release would
otherwise change every graph reported in the audit note with no visible signal.

---

## 6. Name matching

Geometry is joined to the case series on **names rather than on codes**, because the sources supply no
code columns. Canonicalisation absorbs differences of case, whitespace and accent; the residue is
genuine editorial difference between two independently maintained naming conventions, and is resolved
by hand-curated, committed alias maps.

**Each disease has two maps, and they run at different points in the pipeline** — a distinction that
is load-bearing rather than decorative:

| Map | Applied | Repairs | Renames the node? |
|---|---|---|---|
| `*_NAME_ALIASES` | at load, before aggregation | **the case data**: the source's editorial drift, series split across two spellings, a homoglyph | yes, by design |
| `*_GADM_FIX` | at the join only | **the shapefile's own defects**: its misspellings | never |

The load-time map must run before aggregation, because besides correcting the join it reunites units
whose series the source split across two spellings, and those halves must be summed before the data
are pivoted into a node-by-week matrix. But a load-time alias necessarily rewrites the node
identifier, so a shapefile misspelling folded into that map propagates into the project's own
identifiers. Absorbing the shapefile's defects at the join, in a separate map, is what prevents this.

Note that inverting the offending entries does not achieve the same thing. The load-time map runs
before aggregation, so a reversed entry would rename the *shapefile's* key rather than the data's, and
the join would then fail to find the polygon. It is the separation in *time* that is required, not a
change of direction.

**Every alias is resolved by exact match against a real shapefile key; none by approximate matching.**
Approximate matching would bind `antioquia|san andres` to `andes`, a different municipality
altogether. A validation routine asserts that every alias resolves to a key that exists, so an error
in the map fails at build time rather than resurfacing later as an unexplained unmatched node.

**Unmatched nodes halt construction.** Any node that cannot be joined to a polygon raises an error
carrying the full diagnostic, and the builder never discards a node of its own accord. The only nodes
ever removed from a dataset are those listed explicitly, each with its stated reason, in the declared
exclusion lists. Every removal is a decision that somebody wrote down.

---

## 7. Source files

| File | Role |
|---|---|
| `to_schema.py` | The schema, the three loaders, the graph builders, the splits, and the rolling-origin scaffold. |
| `build_datasets.py` | The build: verify, construct, gate on leakage, write. |
| `test_leakage.py` | The 83 leakage and invariant gates, with negative controls. |
| `test_schema.py` | Unit tests for the schema. No geospatial dependencies. |
| `fetch_gadm.py` | Retrieves and verifies the sixteen shapefiles against a checksum manifest. |
| `dengue_aliases.py` | The dengue alias maps and declared exclusions. |
| `dengue_load.py`, `influenza_load.py`, `ebola_load.py` | Per-disease drivers. Everything they report, they assert. |
| `ebola_audit.py` | The Ebola signal-confirmation audit. Read-only; escalates on any drift from the recorded figures. |
| `japan_nodes.py` | Recovers the Japanese prefecture ordering, writing `japan_node_map.csv`. |

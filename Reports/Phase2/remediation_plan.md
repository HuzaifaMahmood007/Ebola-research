# Phase 2 — Remediation Plan

Status of every defect found in the three adversarial reviews (reproducibility, epidemiology,
methodology), plus the items raised by the client's review of `client_decisions.md`.

**Current state:** all defects fixed; code has **uncommitted** changes (clip, split, zero-variance,
calendar-prefix support, +3 gates); `data/processed/*.npz` rebuilt and verified on disk; 86 gates / 6
negative controls green; deterministic. Docs rewritten to the measured numbers. **Nothing committed.**

Legend: **RESOLVED** — fixed and gated · **PENDING** — needs work · **BLOCKED** — needs a client answer.

---

## RESOLVED

### R1. The Ebola clip fabricated 35.8% of the target — FIXED

`cumulative_to_weekly_incidence` ended in `weekly_cum.diff().clip(lower=0)`. A cumulative count cannot
fall; where the report fell, it was a single-week data-entry dropout. The clip zeroed the fall and then
released the recovery — a climb back to cases already counted — as new incidence.

```
2015-01-24  cum=2,612   diff    +43
2015-01-31  cum=  613   diff -1,999  -> clipped to 0, recorded as an OBSERVED ZERO
2015-02-07  cum=  631   diff    +18
2015-02-14  cum=2,741   diff +2,110  -> RELEASED AS 2,110 NEW CASES IN ONE WEEK
```

| | Before | After |
|---|---|---|
| Target mass | 33,338 | **24,552** (= reference exactly) |
| Fabricated | **+8,786 (+35.8%)** | 0 |
| National peak | 2015-02-14 @ 4,982/wk | **2014-10-25 @ 2,688/wk** |
| Observed cells | 1,667 (density .5255) | **1,299** (density .4095) |

*(Query-cell counts are a function of the support scheme, not the clip; under the A3 calendar-prefix the
1,299 observed cells split 27 support / 1,272 query.)*

**Fix:** `weekly_cum.cummax().diff()` — mass-preserving by construction. Weeks whose report falls below
the running maximum are now masked; they were previously scored and normalised on as observed zeros.
The 368 lost cells are those corrupt weeks. That is a real reduction in evaluation data and the honest
count.

**Gated by:** `gate_mass_conservation`, which checks the weekly sum against
`cumulative_reference_mass()` — computed from the source column, sharing nothing with the transform but
the weekly resample. Negative control plants fabricated incidence and requires the gate to fail.

### R2. The dengue split fallback leaked across the graph — FIXED

`per_country_chronological_split` re-cut any node with no observation in its country's train window on
its *own* timeline, placing its train cells inside its neighbours' test period. A GNN aggregates over
neighbours, so this pulled the test period into the training forward pass.

Measured on the affected build: 475 nodes re-cut; 2,420 train cells after their country's `val_end`;
**792 (country, week) columns holding >1 phase; 685,214 observed cells (31%) in mixed-phase columns**.
`data_audit.md` asserted "at any week an entire country block is in a single phase" — false.

**Fix:** fallback removed. Every node takes its country's boundary, no exception. The 475 nodes
(brazil 379, peru 49, colombia 31, argentina 15, taiwan 1) keep their place in the graph and are still
scored, but have no train cell of their own, so they take their **country's pooled train statistics**
(`fit_scalers_masked(groups=...)`). Country train cells are visible at train time → no leakage.
Recorded in `meta["split"]["nodes_without_train"]`. No nodes and no cells dropped.

**Gated by:** `gate_phase_purity`. Negative control plants a train cell in a test column.

### R3. The gates could not fail — FIXED

Both defects above passed 83 gates green. The suite had no check comparing the weekly transform against
anything the transform had not itself produced, and no check that the split honoured the invariant the
docs asserted for it. Both new gates were **written first and confirmed RED on the unfixed build**,
then the fixes applied.

Now **86 gates, 6 negative controls**, all firing. Three are new — fabricated incidence, a train cell
in a test column, and (with A3) an acausal support cell — one per newly-added gate.

*Note:* fixing R2 broke the scaler-provenance gate, which did not know about the country-pooled
fallback. Repaired by teaching the gate the grouping, **not** by loosening it — the 1000× poisoning
probe still holds.

### R4. Documentation falsehoods — CORRECTED

Logged in `data_audit.md` §5.3. Principal items: the clip justification (cited the 58-of-64
downward-step count as *support* for the bug it was evidence of); "thirty zero-variance nodes" (actually
400); "an entire country block is in a single phase" (was false); "the country-macro metric neutralises
Brazilian dominance" (no scoring code exists); "the dengue countries are not adjacent" (Brazil borders
four of them); "*new cases* is too sparse" (95% coverage of the cumulative series).

`schema_spec.md` §5 updated: it was the origin of the clip reasoning and still instructed the reader to
"note it in the paper".

---

## PENDING — ours, no client input needed

### P1. Rebuild the released `.npz` — DONE

The artifacts were once an hour stale (built in memory, never written). Rebuilt repeatedly since; the
released `data/processed/ebola.npz` now carries mass 24,552, 1,299 observed cells, calendar-prefix
support. Verified on disk, not from timestamps. Determinism check passes.

### P2. `data_audit.md` mass figure — FIXED

Corrected 33,164 → **33,338** (measured off the artifact); fabrication is **8,786 (+35.8%)**. Logged in
the audit's §5.3 corrections table.

### P3. Zero-variance guard mis-fire — FIXED

Was firing on 400 nodes, mis-firing on 76 that had real test variance (left un-normalised). The guard
now applies the country-pooled fallback to any constant-in-training node. Measured on the rebuild:
**0 nodes** un-normalised. Leakage-free (pool is train-only). Audit §1.7 updated.

### P4. `environment.yml` — DONE

Written, pinned to the built-and-verified versions (python 3.10, numpy 2.2, pandas 2.3, geopandas 1.1,
libpysal 4.13, shapely 2.1, openpyxl 3.1), conda-forge.

### P5. Docs — REWRITTEN, still untracked

`client_decisions.md`, `data_audit.md`, `schema_spec.md`, `Progress.md`, `remediation_plan.md` rewritten
to the measured numbers (A2 2.8×/1.6×, A3 calendar-prefix, ILINet provenance, 86 gates). **Nothing
committed yet** — the one remaining action.

### P6. Ebola has no validation set — OPEN

Query *is* test, so any hyperparameter choice is made on the test set. Disclose or carve one out.
(Modelling-phase concern; not blocking the datasets.)

### P7. Adopt *new cases* as a cross-check — OPEN

95% coverage, directly reported, would have caught the clip. Recommended as a validation series next
revision.

### P8. Provenance follow-ups from the influenza review — OPEN

Add the IBGE code-oracle as a build gate; label `state360` column 30 as NY-excluding-NYC; **pin Japan's
calendar against NIID/IDWR** (still the weakest — seasonal phase only, ±4 weeks). Audit §2.3 states all
three plainly.

---

## BLOCKED — awaiting the client

### B-A1. Dengue spatial resolution — **unanswered**

Part A1 is still unmarked. Default is to proceed as built (finest-available per country, 7,165 nodes).

### B-A2. Dengue case definitions change mid-series — **unanswered · BLOCKS MODELLING**

| Country | Era 1 | Era 2 | Mean weekly |
|---|---|---|---|
The ×220 etc. were **raw-CSV group-by artifacts**. Measured per node on the *extracted* data:

| country | claimed | measured (extracted) |
|---|---|---|
| peru | ×220 | **none** — single-definition throughout (the `total` era was a 52-row sliver never selected) |
| dominican republic | ×48 | **none** — single-definition |
| mexico | ×13 | **2.8×** across the split (part definition, part real trend) |
| bolivia | ×11 | **1.6×**, change sits inside training |
| panama | ×7 | 1 node |

**RESOLVED: do-nothing + disclose** (client decision). Real exposure is 41/7,165 nodes at a 2.8×/1.6×
step, partly genuine dengue trend — not worth discarding Mexico's 20-year history. Audit §1.10 rewritten
to the measured numbers.

### B-A3. Ebola few-shot causality — **RESOLVED**

Was: support = first 2 obs weeks per district; 85% of query cells preceded the last support cell.
Now: **calendar-prefix support, cutoff 2014-05-24** (client decision, week 8). Support = every observed
cell dated <= cutoff. On the build: 27 support / 1,272 query, 9 districts supported, 52 zero-shot, all 61
scored, 0 without query. Causal by construction (last support 2014-05-24 <= first query 2014-05-31).
**A new gate (`calendar-causal`) enforces it**, with a negative control that plants an acausal support
cell. 86 gates, 6 controls, all green on the rebuild.

### B-A4. Ebola gap-lumping — **RESOLVED: disclose and proceed**

7% of inter-report intervals exceed one week (longest 24). Montserrado: 1,428 on 2014-10-25. No cases
invented, curve correctly shaped, but peaks inflated and adjacent weeks flattened.

Client decision: **disclose and proceed** — the shape is right and both remedies (redistribute across
the gap, or score only contiguous runs) cost more than the distortion. Documented in audit §3.4.3,
schema_spec §5, and the client doc A4. No code change.

### B-B2. Few-shot window size — **DEFERRED by agreement**

Now expressed as the cutoff *date* (2014-05-24) rather than a per-district week count. May widen later;
decide together with the causality property (a later cutoff pulls more of the epidemic into support).

### B-C2. Week-zero masking — **elaboration drafted, not yet sent**

Client asked for *"short explanation, benefits, losses"*. Answer drafted below and folded into the
client doc's C2. **Send it.**

> **What it does.** The first week we ever see a district, we mark it "not observed" instead of guessing
> a case count for it.
>
> **Why.** New cases = today's cumulative − last week's cumulative. At a district's *first* report there
> is no last week, so the quantity does not exist. The original code substituted the district's entire
> running total, which is every case it had ever recorded, in one week. The compilation opens in March
> 2014, months into an outbreak that began in December 2013, so this hit every district including the
> index ones.
>
> **Benefit.** It removes a fabricated number — for ten of sixty-four districts that fabricated value was
> their all-time weekly maximum. It also closes a leak: week zero was the *first support week for every
> district*, and Ebola's scaler is fitted on the pooled support set, so those invented back-logs were the
> dominant input to the normalisation of the entire held-out disease.
>
> **Loss.** 61 cells — one per district — out of 1,299. About 4.7% of the observed data, and it is data we
> never actually had.
>
> **Alternative if you prefer.** The week could be kept with an explicit "back-log" flag rather than
> masked. We do not recommend it: the flag would be an Ebola-only channel, which the shared encoder could
> use to identify the disease.

### B-C7. Ebola Western Area — **RESOLVED: dropping it is correct, no change**

Client had written *"Have a single Western Area node, instead of dropping it"*, then withdrew it on
learning that `Western Area Urban` and `Western Area Rural` are already separate nodes: the parent
label overlaps both in time, so keeping it would triple-count the same cases.

**No change. The release stands as built** — parent dropped, both children retained, 61 nodes. This
also preserves the Urban/Rural distinction, which matters: Western Area Urban (Freetown) is the single
largest district of the epidemic and behaves nothing like the rural remainder.

---

## Sequence — status

1. ~~R1–R4, P1–P4, P3, A2, A3~~ — **done.** Clip, split, zero-variance, calendar-prefix support all fixed
   and rebuilt; 86 gates / 6 negative controls green; deterministic; artifacts verified on disk.
2. ~~B-A2, B-A3, B-C7~~ — **resolved** (do-nothing+disclose; calendar-prefix cutoff 2014-05-24; parent
   dropped). Docs rewritten to the measured numbers.
3. ~~environment.yml~~ — **done.**
4. **Remaining, all non-blocking:**
   - **Commit** — nothing is committed yet (the one action that matters).
   - Send the client the C2 elaboration (drafted, folded into `client_decisions.md`).
   - A4 gap-lumping — client to confirm "disclose" (recommended).
   - P6 (Ebola validation set), P7 (`new cases` cross-check), P8 (IBGE oracle gate, NY label, **pin
     Japan's calendar**) — modelling-phase / next-revision items.

The datasets are internally consistent — source, artifacts, and docs now agree. Nothing blocks the
modelling phase. The gap-lumping disclosure and the Japan calendar pin are the two items worth doing
before the influenza and Ebola numbers are published.

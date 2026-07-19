# Manuscript reconciliation — Phase 1 → Phase 2

The Phase-1 manuscript (`Emerging_Disease_Forecasting_Manuscript.docx`) predates the Phase-2
data engineering and now describes methods the data build corrected or superseded. This memo
lists every passage to change, with the exact current text, a drop-in replacement in the
manuscript's register, and the reason (cross-referenced to `data_audit.md` and the released
bundles). Apply these before submission — item 1 is a correctness defect, not a wording nicety.

British spelling throughout, to match the manuscript. Verified against the released `.npz`
(dengue 1998-01-03→2024-12-28, 1,409 wk, 7,165 nodes/12 countries; Ebola 2014-04-05→2015-03-28,
52 wk, 61 nodes).

---

## 1. Eq. (4) — Ebola differencing  **[correctness — highest priority]**

§6.3 prints the exact clip that Phase-2 review found fabricated **35.8 % of the Ebola target**
(`data_audit.md` §3.4.1). Publishing it describes a broken method. The released build uses the
running-maximum envelope.

**Current:**
> Because these reports record cumulative cases, the reported quantities are differenced to
> recover new cases; for a cumulative series *C*ᵢ,ₜ we take
>
> *y*ᵢ,ₜ = max( 0, *C*ᵢ,ₜ − *C*ᵢ,ₜ₋₁ )  (4)
>
> where negative increments, which arise when totals are revised downward during data
> reconciliation, are treated as reporting artefacts and clipped to zero.

**Replace with:**
> Because these reports record cumulative cases, weekly incidence is recovered by differencing.
> A cumulative count cannot decrease; where a report falls below the running total it is a
> single-week data-entry dropout rather than a downward revision, so differencing the raw
> series and clipping negative increments would zero the fall and then release the subsequent
> recovery — a climb back to cases already counted — as phantom incidence. We therefore
> difference the running maximum (the monotone envelope) of the cumulative series,
>
> *y*ᵢ,ₜ = *M*ᵢ,ₜ − *M*ᵢ,ₜ₋₁ ,  where  *M*ᵢ,ₜ = maxₛ≤ₜ *C*ᵢ,ₛ  (4)
>
> so that a dropout yields no increment and its recovery contributes only the rise above the
> previous high-water mark. Two kinds of cell are treated as unobserved rather than as zero
> and are masked: each district's first observed week, at which no preceding cumulative exists
> and the increment is unidentifiable, and any week whose report falls below the running
> maximum. The transform is mass-conserving by construction — the weekly increments sum to
> maxₜ *C*ᵢ,ₜ − *C*ᵢ,first per district — and this is checked on every build against a
> reference computed independently from the source column.

*Why:* the clip displaced the national epidemic peak by ~3 months and inflated it several-fold;
the envelope reproduces the true late-October-2014 peak and totals 24,552 cases against a
24,552 reference (`data_audit.md` §3.4.1, verified in the released `ebola.npz`).

---

## 2. Table 3 — dengue and Ebola rows

**Current:**
> Dengue | OpenDengue (Clarke et al., 2024) | Admin-1 units, dengue-endemic countries | 2014–present | Weekly | Development
> Ebola  | HDX situation reports; WHO ERT (2014) | Admin-2 districts, GIN / LBR / SLE | 2014–2016 | Weekly | Few-shot holdout

**Replace with:**
> Dengue | OpenDengue (Clarke et al., 2024) | Finest available per country (Admin-1 or Admin-2); 12 endemic countries, 7,165 nodes | 1998–2024 | Weekly | Development
> Ebola  | HDX situation reports; WHO ERT (2014) | District level, mixed administrative level, GIN / LBR / SLE; 61 nodes | 2014–2015 | Weekly | Few-shot holdout

*Why:* the dengue set is finest-available per country (mostly Admin-2, e.g. Brazil's 5,517
municipalities), not Admin-1, and spans the full weekly record 1998–2024 (`data_audit.md`
§1.2, client decision A1). Ebola is mixed-level (Liberia's counties are GADM level 1; Guinea's
prefectures and Sierra Leone's districts are level 2) and the compilation runs 2014-04→2015-03.

---

## 3. §6.3 — dengue spatial-level description

**Current:**
> Development data for dengue are drawn from OpenDengue (Clarke et al., 2024), the most
> comprehensive openly available compilation of dengue surveillance, **at the first
> administrative level** and restricted to dengue-endemic countries; rather than fix the
> country list by hand, we retain a country only when enough of its units carry enough weekly
> observations to be usable, recording the per-country coverage that determined inclusion.

**Replace the italicised clause** so the sentence reads:
> Development data for dengue are drawn from OpenDengue (Clarke et al., 2024), the most
> comprehensive openly available compilation of dengue surveillance, **at the finest
> administrative level each country can support — Admin-2 where enough of its districts carry
> sufficient weekly observations, Admin-1 otherwise, and never a national aggregate** — and
> restricted to dengue-endemic countries; rather than fix the country list by hand, we retain
> a country only when enough of its units carry enough weekly observations to be usable,
> recording the per-country coverage that determined inclusion. The finest-available rule is
> what admits Brazil, whose weekly signal exists only at Admin-2; a uniform Admin-1 dataset
> would exclude the single largest dengue signal in the record.

*Why:* `data_audit.md` §1.2 and client decision A1. A mixed-resolution graph is safe here —
it is block-diagonal, scalers are per-node, and administrative level is never exposed to the
encoder.

---

## 4. §5.2, §6.4, §7 — Ebola support set (per-district → calendar-prefix)

Three passages still describe the **per-district** support scheme that review found acausal
(85 % of query cells preceded the last support cell; `data_audit.md` §3.6, client decision A3).

**§5.2 — current:**
> …is adapted to Ebola using only its two-week support set per district and evaluated on the
> remainder…

**Replace with:**
> …is adapted to Ebola using only a calendar-prefix support set — every district observation
> on or before a fixed early-outbreak cutoff — and evaluated on all later observations…

**§6.4 — current:**
> The held-out disease is handled differently, in keeping with its role as a few-shot target:
> Ebola is not partitioned by the development ratios; instead the first two observed weeks of
> each district form a support set on which the model adapts, and all subsequent observed weeks
> form the query set on which it is evaluated. The incidence normalisation for Ebola is
> estimated from the support observations only and pooled across districts rather than fitted
> per node, since two observations per district are far too few to estimate a stable per-node
> scale and any use of later weeks would leak the outbreak's magnitude into an ostensibly
> few-shot result.

**Replace with:**
> The held-out disease is handled differently, in keeping with its role as a few-shot target:
> Ebola is not partitioned by the development ratios. Instead the support set is a calendar
> prefix — every observation dated on or before a fixed cutoff (2014-05-24, the outbreak's
> eighth week) — and all later observations form the query set on which the model is evaluated.
> The calendar prefix is causal by construction: because districts enter the record at widely
> different dates, taking the first two observed weeks of *each district* would let one
> district's support cell post-date another district's query cells, so that a pooled scaler
> normalising an early query cell would draw on later, peak-epidemic magnitudes — handing the
> model a hint about how large the outbreak becomes. A calendar cutoff makes every support cell
> precede every query cell, a property checked on every build. The incidence normalisation is
> estimated from the pooled support observations only, since a handful of cells per district is
> far too few for a stable per-node scale and any use of later weeks would leak the outbreak's
> magnitude into an ostensibly few-shot result. Under this cutoff nine of the sixty-one
> districts carry support and the remaining fifty-two are evaluated zero-shot — the genuine
> emerging-outbreak regime, in which a district must be forecast from the graph and the few
> districts that reported early.

**§7 — current:**
> Ebola is exempt from both: its protocol is the few-shot support-to-query design in which the
> first two observed weeks per district form the support set and the remainder is the query,
> which is precisely the regime an emerging outbreak presents.

**Replace with:**
> Ebola is exempt from both: its protocol is the few-shot, calendar-prefix support-to-query
> design in which every observation on or before a fixed early-outbreak cutoff forms the
> support set and all later observations form the query, which is precisely the regime an
> emerging outbreak presents.

---

## 5. §7 — NEW paragraph: model selection and the held-out validation protocol

Ebola has no validation split (query is test). Add this paragraph to §7 so hyperparameter
selection is explicit and the held-out result is defensible.

**Insert:**
> Model selection never touches the held-out disease. Every hyperparameter — the shared
> encoder's architecture and optimisation settings, and the few-shot adaptation settings (the
> inner-loop learning rate, the number of adaptation steps, and which parameters adapt) — is
> selected on the development diseases alone: the former on the development validation folds,
> the latter by leave-one-development-disease-out, in which each development disease is held
> out in turn as a surrogate emerging disease and adapted under the same support-to-query
> protocol. These settings are then frozen and applied once to Ebola, whose query set is scored
> a single time, and the adaptation runs for a fixed number of steps rather than early-stopping
> on query performance. Ebola therefore carries no validation split of its own — a deliberate
> choice, since an emerging pathogen affords none, and carving one from its 1,272 query cells
> would both shrink the evaluation set and reintroduce the target-tuning the design exists to
> avoid. Predictive uncertainty for the held-out disease is produced by a time-series-valid
> conformal method that calibrates online over the query sequence (adaptive conformal
> inference; Gibbs and Candès, 2021), needing no held-out calibration set; robustness is
> established across the five seeds, with a sensitivity sweep of the adaptation settings
> reported rather than selected upon.

*Why:* the accepted Ebola validation protocol. Keeps the "genuinely unseen pathogen" claim
clean — the query set is evaluated exactly once.

---

## 6. §7 — evaluation metric (node-macro → country-macro for dengue)

**Current:**
> Per-disease tables are macro-averaged over nodes so that high-count hubs do not dominate the
> score, with a count-weighted variant reported alongside, since pure macro-averaging over many
> near-zero Ebola districts inflates scale-free metrics.

**Replace with:**
> Dengue accuracy is summarised by a country-macro average — scored per node over its observed
> test cells, averaged within each country, then averaged across the twelve countries with
> equal weight — so that Brazil, which supplies 77 per cent of dengue nodes, cannot dominate
> the headline figure; the equal-weight node average is reported alongside. Genuinely constant
> nodes, chiefly the dengue-free Japanese prefectures whose series are identically zero, are
> excluded from scoring, since a constant prediction is trivially exact and the correlation is
> undefined there, and are reported as a separate sensitivity. For the single-panel influenza
> datasets, and for Ebola where a macro over three countries is less meaningful, node-level
> averages are reported, count-weighted alongside the macro so that many near-zero districts do
> not inflate the scale-free metrics.

*Why:* the resolved country-macro metric (`data_audit.md` §1.7; reference implementation in
`score.py`, whose self-check shows a Bolivia-only error moves the node mean by 0.015 but the
country-macro by 0.833 ≈ 10/12).

---

## 7. §6.2 + Table 2 — the block-diagonal graph is a simplification, not a geographic fact

The manuscript justifies dengue's block-diagonal graph as though its countries do not touch.
They do — Brazil borders four of them (`data_audit.md` §1.11). State it as a deliberate
simplification, and add the contrasting Ebola choice.

**Table 2, A_geo row — current:**
> *A*geo | ℝ^{N×N} | Geographic adjacency (queen contiguity; block-diagonal across countries for dengue).

**Replace with:**
> *A*geo | ℝ^{N×N} | Geographic adjacency (queen contiguity; block-diagonal across countries for dengue, connected across borders for Ebola).

**§6.2 — current:**
> For dengue, whose development set spans several countries, adjacency is assembled
> block-diagonally — contiguity is computed within each country and no edges are placed between
> countries — and nodes are labelled by a country-qualified identifier to prevent collisions
> between like-named units; the resulting graph is disconnected across countries, which a graph
> network accommodates without difficulty, and cross-country regularities are captured through
> the shared encoder weights rather than through spurious edges.

**Replace with:**
> For dengue, whose development set spans several countries, adjacency is assembled
> block-diagonally — contiguity is computed within each country and no edges are placed between
> countries — and nodes are labelled by a country-qualified identifier to prevent collisions
> between like-named units. This is a deliberate simplification rather than a claim that the
> countries do not adjoin: several of them share borders, but suppressing cross-border edges
> keeps a graph of mixed administrative level well defined, is accommodated by a graph network
> without difficulty, and lets cross-country regularities be captured through the shared
> encoder weights rather than through a handful of long transmission corridors. For Ebola the
> opposite choice is made and cross-border edges are retained, because Guinea, Liberia and
> Sierra Leone are contiguous and the 2014 outbreak spread across them as a single epidemic:
> a block-diagonal graph would sever the Guéckédou–Lofa–Kailahun tri-border pathway that is the
> defining transmission route of that epidemic.

---

## 8. §6.3 (or §6.2) — one sentence: the influenza calendars are now source-identified

The influenza matrices ship undated; the manuscript is silent on how the calendars were fixed.
Add, where the influenza datasets are introduced:

**Insert:**
> The influenza matrices ship without dates; their calendars are recovered and verified against
> dated authoritative sources rather than assumed. The two United States matrices are identified
> cell-for-cell as CDC ILINet at a known vintage, and the Japanese matrix is pinned to the week
> by matching its per-season maxima to the influenza-season peaks dated by Japan's National
> Institute of Infectious Diseases (the 2018/19 and 2017/18 national peaks, weeks 4 of 2019 and
> 5 of 2018), so that the seasonality channels carry the correct phase.

*Why:* `data_audit.md` §2.3 (US ILINet identification) and the new `japan_calendar_pin.py`
(Japan two-season IDWR pin). Add NIID IASR 40(11) 2019 and 39(11) 2018 to the references.

---

## Not changed (deliberately)

- Horizons {3, 5, 10, 15} and lookback w = 20: a modelling choice, consistent with the
  Cola-GNN/EpiGNN convention. No change.
- The abstract and §9–§10: no now-false claim; leave as written.
- §8 reproduction tables: unaffected by Phase 2 (baselines run on their own shipped data).
- No Ebola/dengue *results* appear in the manuscript (Phase 1 is foundations only), so item 1
  is the sole place the retracted numbers could have propagated — and it is a method
  description, not a results table.

# Phase 2 — Decisions for Client Confirmation

**Emerging Disease Forecasting Framework · Multi-Disease Data Engineering & Harmonisation**

Three diseases are harmonised into five leakage-tested datasets, regenerable from public sources by a
single command. Full detail is in the accompanying **data audit note** and **pipeline documentation**.

**Part A records four decisions, resolved.** Two of them (the dengue case definitions and the Ebola
few-shot protocol) were flagged as blocking in an earlier draft; both are now settled — one by
measurement that shrank the problem, one by a concrete protocol change — and the reasoning is set out so
you can overrule either. The remaining parts record decisions already taken.

- **Part A** — four decisions, each with its resolution and an option to overrule.
- **Part B** — six confirmations where we took the agreed default and simply need it on record.
- **Part C** — decisions we had to take that were not anticipated, and which changed the data.
- **Part D** — properties of the data that affect how the results must be read.

Please mark each item and return this document. Where we recommend a course of action, the
recommendation is stated with its reasoning; you are of course free to overrule it.

---

## Part A — Requires your decision

### A1. Dengue spatial resolution

**What was agreed:** Admin-1 (provinces / states) for every country.

**What we built:** the **finest resolution each country can actually support** — Admin-2
(municipalities / districts) where the data allow, Admin-1 otherwise, decided per country by a
recorded coverage rule rather than by hand.

**Why we departed from the agreement.** Brazil, the largest dengue signal in the world and the backbone
of the "general framework" claim, **has no Admin-1 weekly data at all** in this extract; its weekly
reporting exists only at municipality level. An Admin-1-only dataset would therefore have **excluded
Brazil entirely**, and would have taken Colombia's rich modern Admin-2 series with it in favour of a
stale 2001–2003 Admin-1 fragment. We judged that indefensible to a reviewer and built the finer version.

**What it costs.** The dataset grows from roughly 200 nodes to **7,165**, and **Brazil supplies 77% of
all nodes**. That imbalance is real. The intended mitigation is a headline metric averaged **within each
country first and then across countries equally**, so that Brazil's five thousand municipalities count
for one twelfth of the score, exactly as Bolivia's nine departments do.

**You should know that this mitigation is specified but not yet built.** No scoring code exists in
Phase 2. Approving this item approves the *dataset*; the metric that makes it fair is work in the
modelling phase, and until it exists the imbalance is unmitigated.

| | |
|---|---|
| ☑ **Approve** the finest-available resolution as built (recommended) | |
| ☐ **Revert** to Admin-1 only, accepting the loss of Brazil and Colombia's modern series | |
| ☐ Discuss | |

---

### A2. Dengue — case definitions over time (measured, resolved: accept as built + disclose)

An earlier draft of this document flagged five countries as changing their case definition mid-series,
with multipliers up to ×220, and asked you to choose a remedy. **We then measured each figure against
the data that actually enters the model, and the picture is far milder than those numbers implied.** The
multipliers came from grouping the raw source file by definition; they are not properties of any node's
actual time series.

| Country | Earlier draft claimed | **Measured on the extracted data** |
|---|---|---|
| Peru | ×220 | **no change** — every Peru node is a single definition throughout |
| Dominican Republic | ×48 | **no change** — single definition throughout |
| Mexico | ×13 | **2.8×** level step across the split |
| Bolivia | ×11 | **1.6×**, and the change sits *inside* the training period |
| Panama | ×7 | one node, negligible |

Peru's "×220" was a 52-row sliver of a differently-labelled series that the source's best-resolution
extract never selected; the built Peru data is entirely one definition. The genuine exposure is Mexico
(a 2.8× step, part definition-change and part twenty years of real dengue growth) and Bolivia (1.6×,
mild, and the model already trains on the later definition). Together that is 41 of 7,165 nodes.

**Decision taken: keep the data as built, and disclose.** A 2.8× and a 1.6× shift — partly genuine
epidemic trend — on 0.6% of nodes does not justify discarding Mexico's twenty-year history or Bolivia's
modern series. The alternative, truncating each country to a single-definition era, would cost real data
to remove a distortion smaller than the trend it is entangled with. The shifts are documented in the
audit note so that any Mexico or Bolivia result is read in that light.

| | |
|---|---|
| ☑ **Accept as built, disclosed** (our measured recommendation) | |
| ☐ Truncate Mexico and Bolivia to a single-definition era (costs history; overrule if you prefer) | |
| ☐ Discuss | |

---

### A3. Ebola — few-shot protocol made calendar-causal (resolved)

The support set was the first two observed weeks **of each district**. Because districts enter the
record at very different dates, that was **not calendar-causal**: 85% of the evaluation cells lay
*earlier* in time than the last support cell, so the normalisation of an early district's data drew on
another district's peak-epidemic magnitudes — the model was handed a hint about how large the outbreak
would get.

**Decision taken: calendar-prefix support with a fixed cutoff of 2014-05-24** (eight weeks into the
record). Every observed cell on or before that date is support; everything after is evaluation. This is
causal by construction — no evaluation cell is ever earlier than a support cell — which is the only
reading of "we have seen the first weeks of an emerging outbreak" that survives review.

What it costs, on the released build:

| | |
|---|---|
| Support cells | 27 (across the 9 districts reporting by 2014-05-24) |
| Evaluation cells | 1,272 |
| Districts with support | 9 of 61 |
| Districts evaluated but with **no** support (pure zero-shot) | 52 of 61 |
| Districts never evaluated | 0 |

That most districts are zero-shot is not a defect — it is the genuine emerging-outbreak problem, and
arguably the paper's real contribution: the model must forecast a district it has never seen, from the
graph and the few districts that reported early. A **new gate now enforces calendar-causality on every
build**, with a negative control that fails when an acausal support cell is planted; this is the property
the previous suite structurally could not test.

*(The support-window size — two weeks vs three or four — remains open for a later revision, and is now
independent of this cutoff.)*

| | |
|---|---|
| ☑ **Calendar-prefix, cutoff 2014-05-24** (implemented) | |
| ☐ Move the cutoff earlier/later | |
| ☐ Discuss | |

---

### A4. Ebola — gap-lumping (resolved: disclose and proceed)

When a district reports every week, differencing its cumulative count gives clean weekly incidence. When
a district falls silent and then files, the **whole multi-week increment lands on the single week the
report arrived**. Seven per cent of intervals between reports exceed one week; the longest gap is 24
weeks; the clearest case is Montserrado, carrying 1,428 cases on the week of 2014-10-25.

No cases are invented, and the epidemic curve now peaks in late October 2014, where the epidemic
genuinely peaked. But the peaks are inflated and the quiet weeks either side are flattened.

**Decision taken: disclose and proceed.** The curve's shape is correct; only the within-gap *timing* of
a minority of cases is uncertain, and the two alternatives (spreading an increment across its gap, or
scoring only contiguous runs) either invent a distribution we do not observe or discard evaluation data.
The limitation is stated in the audit note and must be repeated wherever a weekly Ebola magnitude is
quoted.

| | |
|---|---|
| ☑ **Disclose and proceed** (implemented) | |
| ☐ **Spread** each increment evenly across the interval it spans (overrule if you prefer) | |
| ☐ **Score only on contiguous runs** of observed weeks (overrule if you prefer) | |
| ☐ Discuss | |

---

## Part B — Confirmations (the agreed default was taken)

These need no action beyond your acknowledgement, but we would like them on record.

### B1. Influenza scope
Benchmark datasets only — Japan, US regions, US states — used exactly as published, so that our
influenza results sit directly beside the published baselines on an identical graph. A global
country-level surveillance source was considered and declined: it is not sub-nationally resolved, is not
the benchmark, and would require the graph to be rebuilt, forfeiting comparability.

(Yes) Confirmed  ☐ Add the supplementary global source

### B2. Ebola few-shot support window
**Two observed weeks** per district, with every later week held out for evaluation. This is the
operational regime an emerging outbreak actually presents, which is the paper's central point. A wider
window would give the model more to adapt from but would weaken the claim.

*Note:* the **size** of the window is what this item confirms. **Where the window sits in calendar
time** is the unresolved question in A3, and it is the more consequential of the two.

☐ Confirmed at two weeks  (Yes) Widen to three or four

### B3. Dengue coverage thresholds
A node must have at least **52 observed weeks** (one full seasonal cycle) and a country must contribute
at least **three** such nodes. Pruning is applied at country level only, so every node of an included
country is retained. The country set is thus determined by a recorded rule, not hand-picked — which is
what allows us to answer the reviewer's inevitable "why these countries?"

(Yes) Confirmed  ☐ Tighten

### B4. Shapefile provenance
**GADM 4.1** throughout — every graph and every geographic covariate, for all three diseases. One
provenance to document rather than two.

(Yes) Confirmed

### B5. Development set and COVID-19
The development set is **dengue and influenza**. The COVID-19 datasets shipped with the baseline
implementations were used **only** in the Phase-1 reproduction checks; no COVID-19 series enters the
harmonised schema or any released dataset.

(Yes) Confirmed

### B6. Mobility data
**Not used**, for any disease. A real commuting matrix exists for Japan and is tempting, but no
equivalent exists for dengue or Ebola, so attaching it to influenza alone would let the model identify
which disease it is looking at — the precise failure the framework exists to prevent.

(Yes) Confirmed

---

## Part C — Decisions we had to take, which changed the data

None of these were anticipated in the original plan. Each is reversible and none is baked into the raw
files. All are documented in full, with their evidence, in the audit note.

### C1. Ebola — weekly incidence is now taken from the running maximum of the cumulative series
**The most consequential correction in Phase 2, and it came out of adversarial review rather than out of
our own testing.**

Our conversion from cumulative counts to weekly incidence differenced the series and clipped any
negative result to zero. The reasoning recorded at the time was that a falling cumulative count is a
reporting revision. **That reasoning was wrong.** A cumulative count cannot fall; where the report falls,
the report is a single-week data-entry dropout. Western Area Urban, verbatim from the source:

| Week | Cumulative | What we released |
|---|---|---|
| 2015-01-24 | 2,612 | 43 new cases |
| 2015-01-31 | **613** | 0 — recorded as a week with no cases |
| 2015-02-07 | 631 | 18 new cases |
| 2015-02-14 | 2,741 | **2,110 new cases in one week** |

The 613 is a typo; the series snaps back to trend a fortnight later. Clipping zeroed the fall and then
counted the *recovery* — a climb back to cases already counted once — as new incidence.

**What it did to the dataset:** **8,786 fabricated cases across 34 districts, 35.8% of the target.** The
six largest cells in the entire dataset were artefacts. The national epidemic peak was displaced from
late 2014 to **February 2015** and inflated several-fold.

**The fix** differences the running maximum instead, which conserves mass exactly, and masks the corrupt
weeks rather than scoring them as observed zeros. The Ebola target now totals 24,552 cases against an
independently-computed reference of 24,552, and peaks in late October 2014 where the epidemic really
peaked. Observed cells fall from 1,667 to 1,299 — the corrupt weeks are no longer counted as data.

(Yes) Approve

### C2. Ebola — the first observed week of every district is masked
The original method assigned each district's entire cumulative total to date as its *first week's* new
cases. Because the compilation opens months into an outbreak already under way, this fabricated a
backlog for every district — including the index districts — and fed those numbers straight into the
normalisation of the whole held-out disease. At a district's first report there is no preceding figure
to subtract from, so the increment does not exist and the week is marked unobserved. Cost: 61 cells.

☐ Approve
Need further elaboration on this, short explanation, benefits, losses 

### C3. Dengue — the per-node split fallback was removed
Nodes whose reporting began after their country's training cut-off were being re-cut on their own
timeline, which placed their training cells inside their neighbours' test period. Because a graph
network aggregates over neighbours, that pulled test-period data into the training pass — 475 nodes,
31% of observed cells sitting in columns that mixed training and test. Every node now takes its
country's boundary without exception. The 475 nodes keep their place in the graph and are still scored;
they take their **country's** training statistics for normalisation, which uses only data visible at
training time. No nodes and no cells were dropped.

(Yes) Approve

### C4. Taiwan aggregated from townships to counties
The shapefile has no township layer for Taiwan, and Taiwan has no coarser data to fall back on. Its 287
townships were summed into their 22 parent counties. Townships partition counties exactly, so the sum is
exact. The alternative was to drop Taiwan.

(Yes) Approve  ☐ Drop Taiwan instead

### C5. Influenza adjacency diagonal zeroed
The shipped influenza graphs carry a self-connection on every node; ours do not. Left as shipped,
influenza nodes would have carried **twice the self-weight** of dengue nodes inside the shared model — a
structural signature from which the model could infer the disease. The operation is
information-preserving and the actual edges are reused unchanged, so comparability with the published
baselines is intact.

(Yes) Approve

### C6. The Ebola graph keeps cross-border connections
Guinea, Liberia and Sierra Leone are physically contiguous, and the 2014 epidemic was a *single
outbreak* that spread across them: the Guéckédou–Lofa–Kailahun tri-border area is its defining
transmission pathway. A country-separated graph would sever exactly the connections that carry the
signal. Twenty-one cross-border edges, including all three tri-border connections, are verified present
on every build.

(Yes) Approve (recommended)  ☐ Enforce country separation for Ebola too

### C7. Ebola — three Sierra Leonean labels that are not districts
Sierra Leone reports seventeen labels but has fourteen districts. `Port` was found to be `Port Loko`
under a truncated label, continuing the same cumulative series from the next day, and was merged.
`Western Area` is the parent of two districts we already have and overlaps them in time, so keeping it
would have counted the same cases three times; it was dropped. `Freetown` is a city inside one of those
districts, with a single report and no boundary of its own; it was dropped.

☐ Approve
Have a single Western Area node, instad of dropping it.

### C8. Dengue splits are cut per country, not globally
A single global chronological cut would have given **zero training data** to Brazil, Japan, Ecuador,
Panama and Argentina — every country whose reporting begins after the cut — silently, with no error
raised. Each country is therefore cut on its own reporting timeline.

(Yes) Approve

### C9. Geographic covariates are computed but withheld from the shared model
Every node carries its centroid and area. These are **not** exposed to the shared cross-disease model,
and populating them for all three diseases did **not** make them safe to expose: the three diseases
occupy **disjoint parts of the world**, so a coordinate identifies the disease outright. Single-disease
models may use them freely. To share them, they would first have to be made geography-free — a modelling
decision for the next phase, not a data one.

(Yes) Noted

---

## Part D — Properties that affect how the results must be read

No decision required, but these must be understood before any number is quoted.

### D1. The zero-variance normalisation guard (fixed)
Where a node has no variation during its training window, a per-node scaler would divide by zero. An
earlier build substituted a scale of one, which fired on 400 dengue nodes and, worse, left the **76 of
them that do vary in the held-out period un-normalised** — fed to the model on the raw scale while every
neighbour was standardised.

**This is now corrected.** A constant-in-training node takes its country's pooled training statistics
(the same fallback used in C3), computed from training cells only. On the released build the blanket
guard fires on **zero nodes** and **no node with held-out variance is left un-normalised**. Genuinely
constant nodes (e.g. Japan's non-endemic prefectures) remain valid graph neighbours; the caveat on
*scoring* them is noted in D2.

### D2. Twenty-nine of Japan's forty-seven prefectures report zero dengue, always
These zeros are **true** — dengue is not endemic in Japan — and the nodes are kept, because they remain
valid neighbours in the graph. But under a country-averaged headline, Japan would carry one twelfth of
the score while **62% of its nodes are trivially predictable**: a model that always predicts zero is
exactly right. Any results table reporting Japan must state this alongside the score.

### D3. Two Japanese node identities are less than certain
The influenza benchmark ships no names for its nodes, so we recovered the Japanese prefecture ordering
from three independent lines of evidence. Forty-five of the forty-seven are certain. Two — Kōchi and
Kagawa — rest on a thinner population signal. **A transposition would change no model result**, because
the two are interchangeable within the graph; it would only mislabel two adjacent prefectures in a
geographic join. Recorded honestly rather than presented as certain.

### D4. Ebola — every district is now scored
Under the earlier per-district support scheme, `guinea|boke` and `guinea|lelouma` had too few observed
weeks to leave an evaluation cell. Under the calendar-prefix cutoff (A3), all **61 districts have at
least one evaluation cell** and none is unscored. Note the distinct property from A3: 52 districts are
*zero-shot* (they report only after the cutoff, so they have no support cell), but they are still
evaluated, normalised by the pooled support scaler.

### D5. Ebola — four districts recorded no new cases
Four Guinean prefectures reported for weeks but never recorded a *new* case; their handful of cases
predate their first report. This is correct, not a defect, and they are retained.

### D6. The dengue graph forbids cross-border edges, and the stated reason was wrong
The dengue graph is block-diagonal — no edges between countries. This was justified on the grounds that
the dengue countries are not adjacent. **They are:** Brazil borders Colombia, Peru, Bolivia and
Argentina, and the construction severs real dengue transmission corridors, the Leticia–Tabatinga
crossing and the Triple Frontier among them. The choice is defensible as a deliberate simplification —
it is what keeps mixed spatial resolutions safe to combine — but it is a simplification, not a
geographic fact, and it costs real signal.

### D7. The Ebola source file
We build from the file **as published**. An earlier working copy had been opened and re-saved in a
spreadsheet application, which changes the file's fingerprint without changing any data. We verified the
two were identical in content, then switched to the published file so that anyone downloading it can
confirm they hold the same bytes we used. The source has not been modified upstream since November 2015
and is served from a permanent address.

---

## What we got wrong, and how it was found

Several defects in this phase were caught in adversarial review — **after** our test suite had passed
them green, and after we had written them up as sound. The Ebola clip (C1) fabricated 35.8% of a
headline dataset and was described in our audit note as a *justified* decision, supported by a statistic
that was in fact evidence of the bug. The split fallback (C3) violated an invariant the same note
asserted. And the headline dengue case-definition figures (A2) were off by up to two orders of magnitude
because they were measured on a raw-file group-by rather than on the data that enters the model.

Our suite has grown from 83 gates to **86**, and from three negative controls to six. Every new gate
carries a deliberately-planted failure it is required to detect, so we know each one *can* fail —
including the calendar-causality gate (A3), which tests the very property the old suite structurally
could not. **We are telling you this because the number of tests passing was, on its own, worthless as
evidence**, and because the corrections that mattered most came from measuring our own claims rather
than from the suite.

---

## Sign-off

| | |
|---|---|
| Name | |
| Date | |
| A1 — dengue resolution | **Approved — finest-available per country** |
| A2 — dengue case definitions | Accepted as built, disclosed |
| A3 — Ebola few-shot definition | Calendar-prefix, cutoff 2014-05-24 |
| A4 — Ebola gap-lumping | Disclose and proceed |
| Any items to discuss | |

**All four Part-A decisions are resolved and reflected in the released build.** A1 is acknowledged
(finest-available per country); A2, A3 and A4 are implemented with the reasoning above and remain open
to your override. The modelling phase is no longer blocked. Parts B, C and D do not block that work;
they are recorded so that no decision affecting the data is discovered later by reading the code.

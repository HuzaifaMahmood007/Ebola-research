# Phase 3 · Week 3 — Results & Project-Direction Review

**Date:** 2026-07-24 · **Author:** modelling · **Audience:** project stakeholders (go/no-go on direction)
**Scope:** the two Week-3 overnight runs — single-disease (`train.loop --all`) and multi-disease joint
(`train.joint --all`, uniform+uniform) — scored through the frozen `score.py`, 4 development datasets ×
4 horizons × 5 seeds. **Ebola was not touched** (query set is Week-5 only, by protocol).

> **This document exists to answer one question:** do the Week-3 numbers support continuing on the
> planned direction (multi-disease transfer → few-shot Ebola), or do they call for a reframe? It reports
> what the runs actually show, states the risk plainly, and recommends a concrete next step. It does not
> pre-decide — the direction call is the stakeholders'.

---

## TL;DR

> **UPDATE 2026-07-27 — RESOLVED.** The LODO transfer probe (new **Finding 5**, §4b) answered the open
> question: **cross-disease transfer is POSITIVE** via freeze-then-adapt — the mechanism Ebola actually
> uses (frozen trunk + light adapter). It's strongest on the **smallest graphs** (us-regions 10n: +24%
> RMSE, now *beats* the naive floor it used to lose to) and at the **short horizons Ebola can adapt on**.
> Day-14's negative result was **joint co-training** (a tug-of-war over one shared trunk), not the real
> mechanism. **Recommendation is now Option A (continue), with a stronger story — pending a 5-seed
> confirmation.** The original analysis below is preserved; jump to §4b and §5 for the resolution.

> ### ⚠ SUPERSEDED 2026-07-29 — read this before the update above
>
> The 07-27 update is **wrong in its magnitudes and overstated in its conclusion**. Its numbers were
> referenced against seed 42's own single-disease run rather than the 5-seed mean used everywhere else
> in this document (full explanation and corrected table in §4b).
>
> Corrected position: the only cell that is both origin-stable and clears the seed-noise screen as a
> positive is **japan h3**. **us-regions' "+24%, biggest on the smallest graph" does not survive** —
> it is +13.1% against the right reference and sits inside its own seed noise. **Dengue, the only
> genuinely held-out *disease*, is negative at every horizon.**
>
> **The 5-seed confirmation named above is on hold.** The client's `Review Doc.md` requires the fold
> structure to be fixed first: three of our four datasets are influenza, so "leave-one-out" holds out
> a *population*, not a *disease*. Two tables are now required — leave-one-dataset-out (this one) and
> leave-one-disease-out (all three flu held out together vs dengue). See `Phase3_Week4_Work_Order.md`
> §2–§3 and `decisions.md` D1/D5.

1. **The encoder is a competent forecaster on one of four development datasets (US-states), mixed on
   the rest.** It cleanly beats all three trivial baselines only on US-states. On dengue it beats them
   at long horizons but *loses to "copy last week" at short horizons*; on Japan influenza it *loses to
   "copy last year" at every horizon*; US-regions is mixed. None of this is fatal on its own — short-horizon
   persistence dominance and seasonal-amplitude gaps are known, structural, and explainable — but it means
   the single-disease result is not yet a clean headline.

2. **The critical finding: multi-disease joint training does not help any dataset, and materially hurts
   the two smallest ones.** Pooling all four development diseases into one shared trunk left dengue and
   US-states roughly flat (±1%) and made **Japan −7% and US-regions −7% worse on error, with PCC down
   10–20% at long horizons.** The two datasets it hurt most are the two smallest graphs (10 and 47 nodes) —
   which are precisely the ones that resemble Ebola's scale (61 nodes). **This is negative transfer on the
   regime the whole thesis targets.**

3. **What it means:** the project's core novelty mechanism — "learn transferable structure from data-rich
   diseases and carry it to a data-scarce one (G2→G3)" — is, as instantiated by naive joint pooling, **not
   yet delivering.** This is a yellow flag for Weeks 4–5, which are built entirely on that shared trunk.

4. **What it does *not* mean:** naive joint pooling is not the same as the Week-4 transfer mechanism
   (leave-one-disease-out meta-learning / few-shot), and the current joint run used the weighting that is
   *worst-case* for small datasets. The finding is a warning to verify transfer before building on it — not
   proof that transfer is impossible.

5. **Recommendation:** **do not pivot yet, and do not proceed blindly.** Spend one cheap night (~15 runs,
   all code already built) on a **transfer-reality check** — the two alternative joint weightings plus a
   leave-one-disease-out zero-shot probe — then decide. Detailed options in §5.

---

## 1. What was run

| Run | Command | Output | Status |
|---|---|---|---|
| Single-disease | `train.loop --all` | 20 encoder runs + 3 naive floors × 4 datasets | ✅ complete |
| Multi-disease joint | `train.joint --all` (uniform+uniform) | 20 encoder runs (shared trunk, per-dataset adapters) | ✅ complete |
| Offline reads | `analysis.py --ci --reads` | paired origin-bootstrap CIs; gate/spatial contribution | ✅ complete |

All numbers are count-space, scored through the same `score.py` (country-macro for dengue, node-level for
influenza), 5 seeds {42,52,62,72,82}. Naive floors: **persistence** (`ŷ=y_t`), **seasonal-naive** (`ŷ=y_{t−52}`,
persistence fallback), **per-node train mean**.

---

## 2. Finding 1 — single-disease encoder vs the trivial baselines

The Week-3 sanity bar (Task 13.3) is: *the encoder should beat all three naive floors on every dataset and
horizon; if it does not, suspect the harness before the architecture.* **That bar is met cleanly on one of
four datasets.** Headline metric per dataset, mean over 5 seeds:

| Dataset | vs naive floors | Verdict |
|---|---|---|
| **US-states** (49 nodes) | Beats **all three** floors on RMSE/MAE/PCC/sMAPE at **all** horizons. e.g. h5 RMSE 137 vs persistence 163 (−16%); PCC 0.71 vs 0.53. | ✅ **clean win** |
| **Dengue** (7,165 nodes) | **Loses** to persistence at h3/h5 (h3 RMSE 42.1 vs 31.5). **Beats** train-mean and persistence at h10/h15; wins PCC at h5–h15; wins sMAPE h3–h10. Paired bootstrap confirms the sign at every horizon. | ⚠️ **wins long, loses short** |
| **US-regions** (10 nodes) | Beats seasonal at h3; loses to seasonal at h10/h15. Noisiest dataset (10 nodes → wide seed spread). | ⚠️ **mixed** |
| **Japan** (47 nodes) | **Loses** to seasonal-naive on RMSE/MAE/PCC at **every** horizon (h5 RMSE 841 vs 580). Wins only sMAPE at h3. | ❌ **loses to seasonal** |

**How worried to be about each miss:**

- **Short-horizon persistence loss (dengue h3/h5):** expected. At a 3-week horizon on a 78%-imputed panel,
  "last observed value" is an extremely strong and near-unbeatable baseline — this is universal in epidemic
  forecasting, not a defect. The encoder earns its keep at h10/h15, where persistence collapses and the
  encoder wins.
- **Japan seasonal loss:** structural and already diagnosed (`decisions_day_13.md` §3). The encoder *does*
  use seasonal phase (zeroing the sin/cos channels costs ~0.07 PCC — a controlled ablation confirms it), but
  the 20-week lookback cannot reach last year's *amplitude* (`y[t−52]`), which is exactly what seasonal-naive
  copies for free. Feeding `y[t−52]` was rejected on purpose: it breaks the 4-channel disease-agnostic
  constraint **and** does nothing for Ebola (no year-over-year history). This is a documented limitation of
  the frozen w=20 protocol, not a bug — but it does mean Japan is a weak headline dataset.
- **US-regions mixed:** 10 nodes; the seed spread is large enough that most differences are inside noise.
  Low information content.

**Read:** the single-disease encoder is real and clearly beats trivial methods where it matters (long
horizons, the larger/denser panels), but it is *not* a uniform slam-dunk over naive baselines, and a reviewer
running persistence and seasonal-naive will find the same misses we did. The paper's single-disease headline
should sit on US-states + dengue-long-horizon, with Japan/US-regions framed by their structural caveats.

---

## 3. Finding 2 — the transfer signal (multi-disease joint vs single) · **THE CRITICAL RESULT**

This is the number that speaks directly to the thesis. The G2 premise is that a shared encoder trained across
data-rich diseases learns structure that *helps* — the mechanism that is then supposed to carry, few-shot, to
Ebola. So the test is: **does joint multi-disease training beat single-disease training on the development
diseases?**

Mean change across the 4 horizons, **joint − single** (negative = joint worse on error; the sign is set so
"+" always means *joint helped*):

| Dataset | RMSE | MAE | PCC | Net |
|---|---|---|---|---|
| **Dengue** (7,165 nodes) | −0.4% | +0.3% | −3.1% | **flat** |
| **US-states** (49 nodes) | +0.1% | −0.6% | +0.8% | **flat** |
| **US-regions** (10 nodes) | **−4.9%** | **−7.3%** | **−10.2%** | **HURT** |
| **Japan** (47 nodes) | **−8.0%** | **−6.9%** | **−3.1%** | **HURT** |

Worst individual cells: **Japan h15 RMSE −18%**, **US-regions h15 PCC −21%**. No dataset, at any horizon, shows
a material *improvement* from joint training.

**What this says, read carefully:**

1. **Dengue is flat because dengue dominates.** It is 98.5% of the development nodes, so the shared trunk is
   effectively a dengue trunk regardless of what else is in the batch — joint ≈ single for it by construction.
   Dengue being flat tells us nothing about transfer; it's the small diseases that are the test.

2. **The small diseases — the actual test — got worse, not better.** Japan and US-regions are the datasets
   that would have to *benefit* from dengue's data for the transfer story to hold. They degraded by 5–10% on
   error and up to 21% on correlation. This is **negative transfer**.

3. **The datasets it hurt are the Ebola-shaped ones.** Ebola is 61 nodes. The two datasets joint training
   damaged most are 10 and 47 nodes — the small-graph regime Ebola lives in. The one place we most need
   transfer to work is the place this run shows it working *against* us.

**Two competing explanations, and why the difference matters:**

- **(a) Genuine negative transfer** — dengue and influenza have different enough dynamics that forcing one
  shared trunk degrades the minority disease. If true, the "transfer improves accuracy" story is in trouble
  and the contribution needs reframing.
- **(b) A weighting artifact** — the joint run used **uniform** weighting (25% gradient each), which *over-cycles*
  the tiny datasets: Japan's windows are seen ~24× for every one pass over dengue. Over-cycling small data
  invites overfitting, which would show up as exactly this val→test degradation. If true, a gentler weighting
  (sqrt, which we already built) could recover the small datasets and the story survives.

We cannot tell (a) from (b) from this single run — and that distinction is the whole decision. It is cheap to
resolve (§5).

---

## 4. Finding 3 — secondary reads (gate, calibration readiness)

- **The spatial gate is meaningfully on.** Per-dataset gate means run 0.27 (Japan) → 0.60 (dengue), with a
  0% fraction of near-off nodes. So the spatial channel is active and the planned {gate on vs g≡0} ablation
  would measure something real — the gate is *not* silently collapsed to a temporal-only model. (Caveat from
  the protocol: per-node scaling limits what the graph can carry to *timing* co-movement, not magnitude —
  a known, documented trade-off of the disease-agnosticism constraint, not a defect.)
- **Uncertainty/explainability (G4/G5) are Week-5 and untouched** — the quantile head is trained and in place,
  but CRPS/PICP/SHAP are not yet computed. Nothing here changes their plan.
- **P9 topology-augmentation was formally deferred** out of Week-3 scope this session (see
  `decisions_day_13.md` §9) — it is a robustness ablation whose value depends on the spatial channel being
  worth hardening, which is exactly what this review puts in question.

---

## 4b. Finding 5 — LODO resolves the transfer question (2026-07-27) · **DECISIVE**

The sqrt-weighting probe was inconclusive (undertrained), so we ran the clean, weighting-independent test:
**leave-one-disease-out** — train the trunk on 3 diseases, **freeze it**, fit one small adapter on the
held-out 4th (its own train fold), score its test fold. This is exactly the Week-5 Ebola mechanism, and
the *only* difference from the single-disease ceiling is the trunk's provenance (foreign vs same-disease).

> ### ⚠ CORRECTED 2026-07-29 — the original numbers in this section were referenced wrongly
>
> The table first published here compared LODO against **seed 42's own** single-disease run, while
> every other table in this document quotes **5-seed means**. The two were never readable against
> each other. Seed 42 is an unusually *bad* single-disease seed (dengue RMSE h3 z = +1.31,
> us-regions RMSE h5 z = +1.35), so a seed-matched reference systematically flattered LODO — a
> dengue h3 result that is a **15% worsening** against the 5-seed mean was printed as **+3%**.
> The percentage formula (`train/lodo.py:287`) was correct; the reference was not.
>
> **Superseded numbers** (do not re-quote): us-regions +24%/+26% short-h · japan +14%/+10% ·
> us-states +5%/+5% · dengue +4%/−6%.
>
> Corrected table below. `train/lodo.py::_report` now prints both references, and
> `python analysis.py --transfer` produces the CI. Raised by the client in `Review Doc.md`.

Corrected read (seed 42 LODO; **+% = LODO better**; reference stated per column). The **verdict** is
the paired bootstrap over time origins (B=10000) — the dispersion axis the client asked for, and the
only one available at one LODO seed. RMSE:

| held-out | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| **japan** (47n) | **+21.1%** `[+8.7, +31.5]` | +7.6% `[−10.5, +19.7]` *within noise* | **−20.0%** `[−30.3, −10.0]` | **−27.2%** `[−36.3, −18.0]` |
| **us-regions** (10n) | +13.1% `[+3.8, +20.4]` ¹ | +20.0% `[+15.7, +24.9]` ¹ | +9.2% `[+5.3, +13.8]` ¹ | +2.5% `[+0.5, +4.5]` ¹ |
| **us-states** (49n) | +2.5% *within noise* | +0.4% *within noise* | −1.8% `[−3.1, −0.8]` ¹ | −3.0% `[−5.2, −0.9]` ¹ |
| **dengue** (7165n) | **−20.9%** `[−26.7, −15.1]` ¹ | −10.4% `[−16.7, −4.3]` ¹ | **−4.4%** `[−7.0, −2.0]` | **−1.3%** `[−1.6, −0.9]` |

¹ origin CI excludes zero, but the delta is smaller than 1.96× the single-disease **seed** CV at that
horizon: stable across test periods, *not* distinguishable from a different random initialisation.

**Two axes, stated because they disagree.** The CI above is on the **cell-pooled** metric; the
headline tables in §2 are **node-averaged**. They coincide on the dense influenza panels and diverge
on dengue, so dengue's CI and its §2 headline are not the same quantity. Point values differ
accordingly (japan h3 single reads 1036 cell-pooled vs 734.9 node-averaged).

**What the corrected numbers support.** The only cell that is both origin-stable *and* clears the
seed-noise screen as a **positive** is **japan h3**. us-regions is positive and time-stable at every
horizon but seed-fragile throughout. us-states shows nothing at short horizons. **Dengue — the only
genuinely held-out *disease* in this fold structure — is negative at every horizon.** That is the
leakage pattern the client identified: the positive signal lives in the flu-leaky folds.

The earlier claim that transfer is "largest on the smallest graphs" does not survive the corrected
reference: us-regions' apparent +24% was +13.1% against the 5-seed mean and does not clear its own
seed noise (CV 14.9% at h3).

**Why this doesn't contradict Finding 2 — it explains it.** Joint co-training (Finding 2) shares one trunk
that is pulled between all diseases *during* training (uniform weighting over-cycling the tiny sets) → the
small diseases lose the tug-of-war. Freeze-then-adapt gives each disease a fixed general trunk + its own
adapter → no tug-of-war → transfer shows through. The negative Day-14 number measured a mechanism we do
**not** use for Ebola; the positive LODO number measures the one we do.

**Zero-shot (no adaptation) reference:** good PCC, poor RMSE — the trunk gets timing/shape right, the FiLM
adapter fixes scale. Exactly the intended division of labour, and further evidence the trunk carries real
cross-disease structure.

**Caveats (do not lock in yet):** 1 seed — confirm on 5 before it's paper-final (us-regions, 10 nodes, is
the noisy one); long-horizon transfer genuinely hurts, though (a) those horizons are Ebola-zero-shot anyway
and (b) the trunk early-stopped at steps 1k–9k of 91k, so a better-selected trunk may recover them.

---

## 5. What this means for direction — options & recommendation

The honest position: **the single-disease encoder works; the multi-disease transfer mechanism, as currently
run, does not.** Weeks 4–6 (LODO transfer, meta-learning, few-shot Ebola, calibration) are all built on the
shared trunk carrying useful cross-disease structure. This run is the first direct evidence on whether it
does — and the evidence is currently negative on the regime that matters. That is a genuine fork in the road,
so it deserves a deliberate decision rather than momentum.

### Option A — Continue as planned
Treat Week-3 joint pooling as not-yet-the-real-mechanism and proceed to Week-4 meta-learning/LODO on schedule.
- **Upside:** no schedule loss; meta-learning genuinely *is* a different (and often stronger) mechanism than
  naive pooling, so joint-hurt does not disprove it.
- **Downside:** if transfer is genuinely negative, we spend two more weeks building few-shot Ebola on a trunk
  that transfers badly, and discover it at Week 5 with no time to reframe. **Highest-risk option.**

### Option B — Reframe the contribution now
Shift the headline claim from *"cross-disease transfer improves accuracy"* to what the data can support:
*"a single disease-agnostic model that can forecast an emerging pathogen with no training history at all —
where no single-disease baseline can even be trained — and does so with calibrated uncertainty and
explanation."* Ebola has zero training history, so "transfer is flat vs single-disease on dev diseases" is
beside the point for Ebola: single-disease training is *not an option* there. The capability, calibration
(G4) and explanation (G5) become the contribution; the accuracy-gain-from-transfer claim is dropped.
- **Upside:** aligns the paper with a result the data actually backs; the Ebola-enablement + calibration +
  SHAP story is still novel and sits in the same empty intersection identified in the competitive analysis.
- **Downside:** a weaker-sounding accuracy narrative; requires rewriting the framing; premature if the
  negative transfer turns out to be a weighting artifact.

### Option C — One-night transfer-reality check, then decide **(recommended)**
Before committing Week 4, run the cheap diagnostics that separate "genuine negative transfer" from "weighting
artifact." **All code is already built; this is ~15 runs, one overnight:**
1. **Re-run joint with `sqrt` and `proportional` weighting** (~10 runs). If sqrt recovers Japan/US-regions,
   the negative transfer was over-cycling — Option A is safe. If they stay negative across all three
   weightings, transfer is genuinely negative.
2. **Leave-one-disease-out zero-shot probe** (~4 runs): train on 3 dev diseases, forecast the 4th with the
   trunk frozen. This is the Week-4 mechanism in miniature and the *direct* test of whether the trunk
   transfers at all. If LODO zero-shot on the small diseases is not catastrophic, transfer has legs.
3. **Decide on the evidence:** sqrt recovers *and* LODO transfers → **Option A** (continue). Transfer stays
   negative across weightings *and* LODO → **Option B** (reframe to the enablement/calibration story).

- **Upside:** converts a scary one-configuration result into an evidence-based fork, for the price of one
  night. De-risks the single most expensive commitment left in the project (the entire Week 4–5 build).
- **Downside:** one day of schedule; the diagnostics might come back ambiguous (in which case the LODO probe,
  not the weightings, is the tie-breaker — it is the mechanism the paper actually uses).

**Recommendation: ~~Option C~~ → resolved to OPTION A (2026-07-27).** Option C was run. The sqrt/proportional
weighting leg was inconclusive (undertrained), but the **LODO leg (§4b) was decisive: transfer is positive via
freeze-then-adapt**, the mechanism Ebola uses. So we **continue (Option A)** with the refined thesis —
*transferable representations + light per-disease adaptation beat single-disease training for small/emerging
data at short horizons*. **One gate before locking it:** reproduce the LODO result on 5 seeds (§6). If it holds,
Week 4 proceeds as planned; the reframe (Option B) is off the table unless the 5-seed run collapses the effect.

---

## 6. Immediate next actions

1. **[done]** LODO probe built (`train/lodo.py`) and run at seed 42 → **Option A** (§4b).
2. ~~**[next — the one gate]** **5-seed LODO confirmation.** Re-run the folds for seeds {52,62,72,82} (42 done)
   and re-report mean ± sd; the short-horizon transfer win must survive, especially us-regions (10 nodes).
   `python -m train.lodo --all --seed <S>` per seed (each `--all` ≈ 2–3 h, three folds include dengue). If
   time-boxed, the 3 small-disease held-out folds are the Ebola-relevant evidence — run those first.~~
   **COUNTERMANDED 2026-07-28 by the client (`Review Doc.md`).** Do **not** launch this yet. The fold
   structure must be fixed first — holding out one flu dataset while two other flu datasets stay in the
   trunk measures population transfer, not cross-disease transfer. Spend the seeds on the corrected
   folds. See `Phase3_Week4_Work_Order.md` §3 and `decisions.md` D1.
3. **[regardless]** Day-15 baselines under the common pipeline (G6) — EpiGNN/Cola-GNN/HeatGNN/MTGNN/MepoGNN.
   **Independent of the transfer story:** the paper's SOTA claim rests on beating *these*, not the naive
   floors, and we still have zero head-to-head numbers. Launch the export + queue.
4. **[housekeeping]** Commit the untracked `ablation/`, `train/lodo.py`, `results_paths.py`, the results
   reorganisation, and this review.

---

*Numbers in this document are generated from `results/*.json` via a read-only aggregation over 5 seeds; the
paired-bootstrap significance reads are in `results/reports/gated+spatial_Contribution.txt`. A condensed companion is
`results_summary.txt`.*

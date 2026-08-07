# Phase 3 · Week 4 — Work Order (from client `Review Doc.md`)

**Date:** 2026-07-29 · **Source:** client reply to Week-3 review · **Status:** this doc = what we do now.

Client back. Read Week 3. Made direction call. Some of our plan now dead. This doc = the new plan,
in order, with commands and done-criteria.

**Big thing:** 5-seed LODO run is **CANCELLED** until folds fixed. Do not launch it.
`Phase3_Week3_Results_and_Direction.md` §6 still says "run 5 seeds next" — that line is stale.

---

## 0. Rules that apply to EVERY number from now on

Client made these standing. Break one = table gets rejected.

| Rule | What it mean |
|---|---|
| Dispersion always | mean ± sd minimum. Headline number = bootstrap CI over regions AND time origins. No bare point estimates. |
| Never average across datasets | dengue 7165 nodes and us-regions 10 nodes never go in same mean. Already doing this. Keep. |
| Label the reference | every delta table says on the table what it compares to. This is what bit us (see §2). |
| Noise floor | cell under noise floor writes "within noise". Not a direction. Client said stop reading sub-5% joint cells. |
| Units + formula | "spatial contribution" column needs both. One sentence. |

---

## 1. Baselines — TOP PRIORITY, the Week-3 gate

Client: *"The baseline benchmark isn't really a third option. It's the Week 3 gate. Nothing gets
signed off without it."*

### 1a. Run status (checked 2026-07-29 16:xx, `baselines/_preds/`)

| Model | Files | Target | State |
|---|---|---|---|
| EpiGNN | 80 | 80 | ✅ done (dengue + 3 flu × 4h × 5s) |
| ColaGNN | 60 | 60 | ✅ done (3 flu × 4h × 5s) |
| HeatGNN | 5 | 60 | 🔄 **running** — only us-regions h3 done. us-regions h5/h10/h15 + japan + us-states left |
| MTGNN | 0 | 80 (=20 runs) | ❌ **not started** — env `mtgnn` exists, just not launched |

HeatGNN slow. us-regions h3 seed82 took ~4h. CPU env, no GPU. 55 files left = long. Budget for it.

MTGNN = 20 runs, one per (dataset, seed), each emits all 4 horizons from rollout. GPU. Faster.
**Launch MTGNN now in parallel — different env, different device, no clash with HeatGNN on CPU.**

Commands live in `Phase3_Week3_Day15_Baselines_Status.md` §"Run commands". Do not re-derive them.

### 1b. Rescore — after every batch

```powershell
conda run -n ebola-train python score_baseline.py    # -> results/baselines/*.json
```

`results/baselines/` currently holds **one stale smoke file** from 27/07
(`EpiGNN__dengue__h3__seed42.json`). It will be overwritten. Do not read it as a result.

Never copy a baseline's own printed metric. Always rescore through our `score.py`. (Task 15.3.)

### 1c. NEW condition — reproduction validation

Client added this. Not optional.

> *"validate each reproduction against the source paper's published numbers before using it as a
> comparator, and show me our reproduction next to their reported figure. If we can't get close,
> that baseline doesn't go in the comparison table."*

So: one extra table, **their number next to ours**, on the baseline's own paper dataset/horizon.
Some of this already exists in memory (EpiGNN reproduces within ~1% on region785 h5). Cola and Heat
were run at **wrong horizons** last time — paper reports Japan at h∈{2,3,5,10,15} (Cola) and
h∈{2,5,7,12} (Heat), we ran h=1. Those need a rerun **at paper horizons on the paper's own data**,
separate from our pipeline runs.

**Done =** table with columns `model | paper dataset | paper h | their reported | our repro | Δ%`.

### 1d. NEW — reproduction failure log, start NOW

Client: *"start the reproduction failure log now, every baseline we couldn't reproduce and exactly
why... it's far easier to keep as you go than to reconstruct at the end."* Publishable on its own.

New file: `reproduction_failure_log.md`. Seed it today with what we already know:

- **MepoGNN** — dropped. Both variants need mobility/OD matrix we don't have (`A_mob=None`).
  Dynamic needs OD tensor; Adaptive needs static commuter matrix to seed graph + drive SIR.
  Property of our data, not model failure.
- **MSGNN** — not run. Needs Ubuntu + CUDA 10.1. Windows box can't.
- **STOEP** — paper table ≠ shipped dataset/metric. Overall RMSE 63.5 in paper vs 182.9 on the run;
  paper uses SMAPE, run logs MAPE; RAE 0.37 vs 0.224. ~⅓ scale → different dataset. Timeboxed,
  exclude-and-document unless resolved.
- **Dengue not like-with-like** — encoder scored on full 7165 nodes, baselines on 1/3 subset (2392,
  stratified by country, seed 20260715). Repos built for ≤49 nodes OOM at 7165. Disclosed caveat.
- **Cola/Heat influenza-only** — CPU envs, dengue-⅓ too slow. Dengue among lineage = EpiGNN only.

---

## 2. Fix the transfer table — numbers are wrong as printed

Client caught two things. Both real. Both already diagnosed.

### 2a. The reference mismatch — client right

`train/lodo.py::_report` compares LODO vs **seed-42's own** single run.
`results_summary.txt` + direction doc report **5-seed means**. Two tables never comparable.
Formula at `train/lodo.py:287` is **correct** — the reference choice is the bug.

Real numbers, RMSE, `+%` = LODO better:

| held-out | h3 (s42 → 5-seed) | h5 | h10 | h15 |
|---|---|---|---|---|
| us-regions | +19.9 → **+9.9** | +28.0 → +17.8 | +2.4 → +4.0 | −0.5 → −2.8 |
| japan | +24.0 → +23.4 | +4.5 → +7.4 | −25.4 → −21.4 | −22.6 → −26.4 |
| us-states | +5.8 → **+1.4** (noise) | +4.4 → +1.5 | −0.2 → −0.7 | −4.1 → −1.3 |
| dengue | +2.9 → **−15.4** | +5.7 → −6.2 | −0.8 → −1.7 | −1.7 → −1.5 |

### 2b. Dengue sign flip — client right, cause is not a formula bug

Seed 42's dengue single run is an **outlier**: h3 RMSE 49.98 vs 5-seed mean 42.06. Seed-matched
reference flatters LODO by ~18 points. That is the whole "+3% printed for a 17% worsening".
Dengue PCC negative at every horizon under **either** reference.

**Action:** patch `_report` to print both references side by side, and label them on the table.
Cheap fix, `train/lodo.py:265-290`.

**Done =** transfer table where every delta names its reference, us-states cells say "within noise",
and dengue reads negative because it is negative.

---

## 3. Fix the folds — the reason the 5 seeds are on hold

Client: three of four datasets are influenza. Leave-one-out holds out a **population**, not a
**disease**. Biggest wins sit in the folds with most flu leakage. Reviewer will read it same way.

Report **two separate tables**:

1. **Leave-one-dataset-out** — what we have now. Keep it. Real result about population transfer.
2. **Leave-one-disease-out** — all 3 flu held out together as ONE fold, dengue as the other.
   This is the honest cross-disease number. This is what the central claim rests on.

### 3a. Half of table 2 already exists

`train/lodo.py:224`:

```python
in_names = [n for n in DEV_BUNDLE_NAMES if n != held_out]
```

`held_out="dengue"` → `in_names` = exactly the 3 flu sets. **The dengue fold we already ran IS the
flu→dengue leave-one-disease-out number.** Relabel it, do not rerun it. It is the −15.4% / −6.2% row.

### 3b. The other direction needs code

Trunk on dengue alone, adapt to the 3 flu. Architecture allows it, confirmed:

- `SharedEncoder.forward` (`models/encoder.py:57`) — no node-count param anywhere.
- `Adapter` (`models/adapters.py:12`) — `gamma[64]`, `beta[64]`, `Linear(64→20)`. No node dim.
  **One adapter serve all 3 flu graphs at once.** Free.
- `_prepare(names, device)` (`train/joint.py:70`) + `block_diag_sparse` take arbitrary name list.
  One-element list `["dengue"]` works fine.

Gap: `_fit_adapter_and_score` (`train/lodo.py:135`) takes ONE name, ONE bundle. Need multi-bundle
version fitting ONE shared adapter across 3 flu. Loop already exists in `_fit_trunk` — reuse
`_prepare` + `_origin_stream`, swap `ModuleList` for single `Adapter`.

Scoring stays **per-dataset** (3 rows). Nothing averaged across datasets. Rule respected.

### 3c. What to say about it in the write-up

Fold sides are lopsided. State it:

| side | nodes | train cells | train origins |
|---|---|---|---|
| influenza (3 sets) | 106 | 20,918 | 680 (3 graphs) |
| dengue | 7,165 | 1,249,109 | 1,138 (1 graph) |

60× the cells. Two directions **not comparable to each other** — each only reads against its own
single-disease ceiling.

Also: `dengue→flu` is the **Ebola-shaped** direction (train big, adapt to small unseen graph).
`flu→dengue` is backwards from Ebola. The free number is the less relevant one.

Also: two diseases = **two folds**. That is the ceiling on this table's weight. That is the concrete
argument for COVID coming back (§9).

One shared flu adapter across japan/us-regions/us-states is fine despite scale spread (raw means
655 / 1009 / 223) because the per-node scaler runs **before** the encoder — FiLM never sees it.
One sentence in the paper, not a design change.

### 3d. Then, and only then, the 5 seeds

Seeds {52,62,72,82} (42 done), both fold structures. `python -m train.lodo --all --seed <S>`,
≈2–3 h each. If timeboxed, run the small-disease folds first.

---

## 4. Meta-learning — client wants an answer THIS WEEK

Client: what we built (train on 3, freeze, fit adapter) is **transfer learning with a linear probe**,
not meta-learning. Brief has meta-learning + domain generalisation as **REQUIRED** under G2. And we
picked this encoder *because* it supports cheap second-order grads — that argument only pays if used.

Their call: build episodic meta-training across diseases. MAML / Reptile / ProtoNet — **our pick, we
justify it**. Freeze-and-adapt becomes the ablation. Then "does meta-learning beat a linear probe" is
a number, not an assertion.

> *"Scope it this week and tell me straight away if you think it doesn't fit the schedule. If it
> doesn't, I'll take the re-scope to Nora. Please don't quietly absorb it."*

**Do not quietly absorb it.** Two deliverables this week:
1. Which algorithm + why (one paragraph).
2. Straight yes/no on schedule fit.

Head start: `models/adapters.py:13` docstring already says *"the ONLY thing Week-4's MAML inner loop
touches; the trunk is the outer loop"*. Adapter is **1,428** params (d=64, |H|=4, |Q|=5); the 388
figure quoted before 2026-07-31 predates the five-quantile head. Inner loop is still small.

---

## 5. Ebola audit — client's #1 ask, numbers DONE, needs writing up

Client: *"the single thing I most need from you and I don't think the Week 2 audit note ever reached
me. It's now the biggest open risk."*

Computed 2026-07-29 from `data/processed/ebola.npz`:

- **Resolution:** weekly. N=61 districts, T=52 weeks (2014-03-24 → 2015-03-28).
- **Missingness:** 41% observed (1299 / 3172 cells). Mean 21.3 observed weeks/node, min 1, max 40.
- **Graph:** usable. 146 edges, **0 isolated nodes**, 21 cross-border. GADM 4.1.
- **Support set:** calendar prefix ≤2014-05-24 = weeks t=1..7 → **27 observed cells across only 9 of
  61 districts.**
- **Labelled adaptation examples** (support cells reachable as (node, origin) pairs):
  - Full 20-week window: **0 at every horizon** (support ends t=7, earliest full origin t=19).
  - With P7 left-pad (adaptation only): **h3 = 16 pairs / 8 nodes. h5 = 6 pairs / 2 nodes.
    h10 = 0. h15 = 0.**
- **Query / evaluation set:** 1151 / 1075 / 866 / 642 pairs at h3/h5/h10/h15, covering 61/61/59/58
  districts.

### Two corrections to send the client

1. They said the 27-example floor came from "the smallest development set". **Backwards.** 27 IS
   Ebola's own support-cell count. Constraint was calibrated against the right thing.
2. They feared h10/h15 "may not be meaningfully evaluable". **They are evaluable** — 642 pairs over
   58 districts at h15. Their "~17 origins" arithmetic was per-node; pooling over 61 districts saves
   it. h10/h15 are **zero-shot, not unmeasurable**.

### The real problem to raise instead

h5 adaptation = 6 examples on 2 districts. That is noise, not adaptation. Practical adaptation
horizon is **h3 only**. Feeds directly into the horizon-set decision they're taking to Nora.

---

## 6. Joint training — 2 days, then stop either way

Client agrees with our diagnosis (dengue ≈98% of examples → shared model becomes a dengue model that
glanced at flu). But: *"we haven't tried the obvious remedies and a reviewer will ask whether we
balanced the sampler."*

Try, then stop:
- balanced / temperature-scaled sampling
- per-disease loss reweighting
- per-dataset normalisation

Still fails after that → negative result gets **stronger**. Stops reading as unfinished experiment,
becomes clean motivation for freeze-then-adapt.

Hard stop at 2 days. Machinery exists in `train/joint.py` (samplers already loss-weights).

---

## 7. Metrics to add — RMSE/MAE/PCC not enough

Client: epidemic-forecasting reviewers won't accept our three. And we claim calibrated uncertainty.

Add to `score.py`:
- **WIS** — CDC FluSight / Forecast Hub standard. Non-negotiable.
- **CRPS**
- **Empirical coverage at nominal levels** + **PIT histograms**
- **Peak timing error** + **peak intensity error** — epi reviewers ask for these specifically
- **Scale-normalised error** alongside raw counts — across 7165 dengue regions raw RMSE is dominated
  by biggest regions

Note: `score.py` already has `peak_timing` / `peak_intensity` in `LOWER_BETTER` (`train/lodo.py:55`).
Check whether implemented or just named.

Quantile head already trained and in place → WIS/CRPS/coverage/PIT have the inputs they need.
Watch out: `models/adapters.py:21` flags plain `Linear` can emit **crossing quantiles** (q05 > q95).
Median unaffected, intervals are. Decision #6 = sort the 5 quantiles post-hoc **before** PICP/CRPS.
Do that, else coverage numbers are garbage.

---

## 8. More items, in client's priority order

| # | Item | Note |
|---|---|---|
| 8a | **ARIMA/SARIMA + GBM on lag features** into the naive floor set | GBM is hard to beat. Client: *"I would much rather find that out from you than from a reviewer."* |
| 8b | **Reserve conformal split NOW** | Must happen before Ebola work. Retrofitting painful. |
| 8c | **Pick conformal variant** | NOT textbook split conformal — assumes exchangeability, we break it on two axes (temporal dependence + shift onto unseen disease). Look at ACI / CQR / EnbPI, weighted variant for covariate shift. Pick one, tell client which. |
| 8d | **Explainability scoping note** | SHAP over ST-GNN at 7165 regions = serious compute. Only 4 input channels, so unclear what we attribute over (channels? lags? neighbours?). Integrated gradients probably more tractable. Deliverable = scoping note, not code. |
| 8e | **Gate figure — build it** | Client: already an interpretable result, we're under-using it. Gate mean 0.27 (japan) → 0.60 (dengue), scaling with region count/density. That's a figure. Cheap win. |
| 8f | **Seed ensembling — take it** | Client: *"close to free"*. We already train 5 seeds. Average their **quantiles** at inference, not mean-of-metrics. Lowers variance → every comparison easier to defend. Ebola-safe (ensemble the few-shot adapters too). |
| 8g | **Delta prediction — one test only** | ONE dataset, ONE seed. Bring the number, then decide. Doesn't win clearly → drop it AND write a paragraph why. Paragraph useful either way. |
| 8h | **Pre-registration before Ebola touched** | Freeze config, hash it, timestamp, commit. Score Ebola exactly once. Client adopting this as a **stated methodological contribution**, not internal process. |

---

## 9. What is NOT our work

**Deprioritised by client — do not spend time:**
- Japan losing to seasonal-naive. Diagnosis correct, explanation publishable as-is.
  **Do NOT add t−52 lag channel to main model** — Ebola has no prior year, channel permanently null
  for our actual target. Train/target mismatch to win a table cell. Benchmark-only variant at most,
  decision for later.
- Single-disease accuracy work beyond 8f and 8g.
- Epidemiology-informed component stays a **planned ablation**, not main build. Confirmed, not drifted.

**Client taking to Nora — not ours to decide:**
- Adding COVID-19 back as third development disease (they want it back; our §3c two-fold problem is
  the ammunition)
- Ebola horizon set, pending our audit (§5 — our h3-only finding lands here)
- Encoder confirmation (provisional since 20 July, everything downstream assumes it)
- Baseline shortlist sign-off (brief requires client sign-off, can't tell from records it happened)
- Schedule, journal, authorship, IP

Not blocked on any. Keep moving.

---

## 10. This week, expedited — the short list

Client named four:

1. **Ebola audit numbers** → §5. Numbers done. Write and send.
2. **Reconciled transfer table** → §2. Numbers done. Patch `_report`, relabel, re-issue.
3. **Read on whether meta-learning fits** → §4. Needs a decision from us.
4. **Baseline head-to-head** → §1. Running. HeatGNN 5/60, MTGNN 0/80 — launch MTGNN now.

Then, and only then: 5-seed LODO on the corrected folds (§3d).

---

## 11. Housekeeping — repo is dirty

Untracked, needs committing: `ablation/`, `train/lodo.py`, `results_paths.py`, `export_baseline.py`,
`export_mtgnn.py`, `score_baseline.py`, `run_baselines.py`, `make_results_doc.py`, `results_summary.txt`,
`test_run.py`, `Review Doc.md`, both Week-3 docs, this doc.

Modified, uncommitted: `analysis.py`, `decisions_day_13.md`, `train/loop.py`.

Also: `Phase3_Week3_Results_and_Direction.md` §6 item 2 still reads *"[next — the one gate] 5-seed
LODO confirmation"*. **Countermanded.** Mark it superseded by this doc before anyone else reads it.

---

## 12. One thing client liked — keep doing it

> *"flagging the single seed caveat yourself, without being asked, is exactly what I want from you.
> I'd rather hear a result is fragile from you than find out in review."*

Feasibility section going in the paper as a **stated contribution** (pairing every check with a
deliberately-broken version that must fail). Same for score-Ebola-once and the reproduction failure
log. Keep flagging our own weak numbers first.

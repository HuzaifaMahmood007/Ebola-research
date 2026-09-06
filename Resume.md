# Resume — session handoff

**Written 2026-09-01.** Everything a fresh chat needs to continue without re-deriving it. Read this
first, then `progress/summaries/Priority_Fix_Progress.md` §13 for the last engineering session and
`Reports/Phase0_to_Now_Audit.md` for the standing audit findings.

---

## 1. What the project is

A disease-agnostic spatio-temporal GNN. Train a shared encoder on data-rich diseases (dengue,
influenza x3 panels, COVID), then forecast a data-scarce emerging disease (Ebola, 2014 West Africa)
it has never seen. Deliverables are code plus a journal manuscript. Goals G1-G7 are in
`progress/planning/Final Internal Project Brief.md`.

Four experiments, and almost every file belongs to one:

| experiment | code | results | what it answers |
|---|---|---|---|
| single-disease | `train/loop.py` | `results/single/` (25) | what the architecture does with no transfer |
| transfer (LDO3) | `train/lodo.py` | `results/lodo/` (158) | what transfer costs |
| Ebola case study | `train/ebola.py` | `results/ebola/` (20) | the actual claim |
| published baselines | `run_baselines.py` | `results/baselines/` (280) | the SOTA comparison |

---

## 2. Current state

**All expensive compute is finished.** Nothing is pending except optional extras.

- Ebola: 20 scored records + 20 quantile archives, both arms (L12 primary, L20 secondary) x both
  regimes (few-shot, zero-shot) x 5 seeds. Scored ONCE under a hash-frozen pre-registration.
- Single-disease ceilings: 25 records + 25 quantile archives, all five panels.
- LDO3 transfer: 158 records.
- Baselines: EpiGNN 80, MTGNN 80, Cola-GNN 60, HeatGNN 60.
- ANIL: ran 2026-08-18 (`dengue2flu`) and 2026-09-04 (the three LDO3 disease-out folds), both arms,
  five seeds each. **Four folds, 32 cells, 3 held-out diseases, and the null holds.** Written up in
  `progress/outcomes/ANIL_Results.md`, generated and independently verified by
  `diagnostics/anil_report.py` and `diagnostics/verify_anil_doc.py`.

**Open / unfinished:**

| item | state |
|---|---|
| G5 explainability | **not built.** Zero attribution code in our source. Only REQUIRED goal with nothing written. |
| Manuscript v2 | `Reports/Manuscript_v2.md`, ~13,628 words against a 12,000 limit. It IS git-tracked; the earlier claim that `Reports/` is ignored was wrong. |
| COVID contradiction | **resolved 2026-09-06.** B5 amended in `progress/decisions/client_decisions.md` under the client's own Week 3 instruction (`Review Doc.md:97`). COVID is the third development disease. Ledger C1 DONE. |
| MTGNN sentence | **corrected 2026-09-06.** MTGNN is excluded from the brief's head-to-head and from its naive-floor tally, with a dated correction note and a re-rendered `.docx`. Ledger C2 DONE. |
| LDO3 zero-shot quantiles | ~10 h retrain, still a stated limitation |
| Shuffled-adjacency control | ~6 h, now optional since the gate does not help |

---

## 3. The results, as they actually stand

**Ebola.** With no Ebola data at all the model beats persistence by 16% / 15% / 16% / 44% at
h3/h5/h10/h15 (primary arm), and 19% / 17% / 14% / 42% on the secondary. On the corrected interval
(see §4) **14 comparisons clear zero, spanning every horizon.** Unadapted wins 12 of 16; adapted
wins 2 of 16.

**The pre-registered criterion was NOT met.** It required the *adapted* model to beat persistence at
h3 or h5 with the interval clearing zero. All four such intervals span zero (closest: h3 RMSE
[-11.85, +0.35]). The wins we have come from the *unadapted* model, which is not what the criterion
asked for.

**Transfer is negative.** Symmetric seed-ensembled comparison: 1 better, 10 within noise, 25 worse
over 36 cells. Verdict survived making the opponent stronger.

**The graph does not help accuracy.** Gate-off ablation, 25 cells: on error the spatial channel
helps in **0 of 40** and hurts in 8. It only helps correlation (6 of 20 PCC cells). Negative result
for the component the architecture was chosen for.

**Meta-learning does not help, across four folds.** Against its own seed-matched control ANIL is
better in 0 of 32 cells, worse in 1, within noise in 31. The one significant cell sits in the
original `dengue2flu` fold; all three LDO3 folds are entirely within noise. The standing objection
that episodes varied population and origin rather than disease is retired for the three LDO3 folds,
whose meta-train sides span two diseases, and the answer did not change.

**Calibration transfers, and that is the strong result.** The frozen cross-disease correction, which
reads no Ebola outcome whatsoever, lifts coverage from 0.28-0.70 up to 0.65-0.98. Online adaptation
adds ~0.03 at short horizons and nothing at long ones.

**The honest thesis:** a shared representation transfers to an unseen pathogen well enough to beat
naive floors and carries its uncertainty calibration with it, but every mechanism added to *improve*
transfer (spatial message passing, few-shot adaptation, meta-learning) fails to help. That is a
boundary-conditions paper and it is publishable as one.

---

## 4. Corrections made this session (do not regress these)

**M1 — the estimand mismatch. FIXED, committed as `ebola_ci.py`.**
The Ebola headline is a NODE-AVERAGED country-macro (`score.py:215-237`). The interval adjudicating
the pre-registration was a CELL-POOLED one (`analysis.py:63-83`). They diverge 1.5-2.0x on Ebola;
the headline 38.20 sat OUTSIDE its own quoted interval [38.349, 104.537]. Point and interval were
different statistics, so the criterion was never adjudicable as reported.
`ebola_ci.py` rebuilds the macro from `*__pernode.npz`, asserts it matches the scored JSON before
printing (so "same estimand" is checked, not claimed), resamples DISTRICTS (E6's missing axis, M3),
and shares one draw across the five seeds with pooled seed-level differences (M2). Read-only.
Result: wins went 8 -> 14 and stopped being h10/h15-only. Verdict unchanged, now defensible.

**Two things asserted this session that turned out to be wrong:**
1. "The pre-registered criterion is NOT met" — incomplete. It is not met AND was not adjudicable on
   the reported statistic. Corrected in memory.
2. "Pooled normalisation makes the task harder, so our Ebola numbers are a lower bound" — **wrong.**
   Measured: pooling HELPS (see §6). That was the comfortable assumption, not a measured one.

---

## 5. Experiments run this session

**Sampling ablation (free, already on disk, nobody had read it).** `uniform` vs `sqrt` weighting
across diseases. Uniform wins 9 of 16 cells, sqrt wins 2, 5 too close to call. Giving dengue MORE
weight made dengue **69% worse at h3 and 73% worse at h5**. More of a big noisy dataset made the
model worse at that dataset. Caveat: sqrt has one seed, uniform has five, so do not quote the exact
percentages; the direction is safe because the gaps dwarf the seed spread.
**Conclusion: uniform sampling is correct, and now measured rather than argued.**

**Pooled-vs-per-node normalisation.** See §6.

Logs: `results/reports/` — `ebola_district_ci.log`, `ebola_district_ci_unstratified.log`,
`pooled_scaler_cost.log`, `pooled_scaler_cost_usstates.log`, `verify_brief_numbers.log`.

---

## 6. The normalisation finding, and the open diagnosis

Ebola uses `scaler_scope=per_disease_support` (one pooled scale for the whole disease); every
training panel uses `per_node_train`. Measured divergence of district means from zero:

```
influenza (all three)  0.0000      dengue  0.0387
ebola_L12  0.6550      ebola_L20  0.5363
```

So the encoder trains on inputs centred at zero and is handed Ebola inputs scattered by ~+-0.8. That
is a covariate shift we introduced ourselves, and a reviewer will find it because we publish the
diagnostic.

**Priced it.** Retrained a dev panel with a pooled scaler, 5 seeds, everything else identical:

| panel | pooled severity | vs ebola | mean cost |
|---|---|---|---|
| influenza_japan | 0.0603 | 0.09x | -1.3% (nothing) |
| **influenza_us-states** | **0.5772** | **0.88x** | **-6.8% (POOLING HELPS)** |

On the matched panel pooling helps at every horizon: -1.1 / -4.3 / -10.1 / -11.8 % at h3/h5/h10/h15.
The long-horizon gains are ~3 seed SDs, so real.

**Not settled:** the experiment trained AND tested with pooling, consistently. Ebola's real situation
is a MISMATCH (trunk trained per-node, Ebola arrives pooled). That is an LDO3-shaped experiment and
has not been run. Also untested on dengue, where 7,165 wildly uneven nodes could flip the answer.

**Awkward implication for the paper:** pooling helps most at h10/h15, which is exactly where the
Ebola wins are. It does not invalidate them (naive floors are scored in raw counts and never touch
the scaler) but it must be stated before a reviewer states it.

### The standing diagnosis, and how to confirm it

Three symptoms pointing one way:
- loses to `train_mean` on COVID and dengue h15
- 90% intervals cover ~50%
- per-node scaling loses to blunter pooled scaling at long horizons

Proposed chain: per-node scaling amplifies noise on small quiet nodes -> the model learns noise is
signal -> it predicts more variation than it can justify -> over-jumpy point forecasts AND
too-narrow intervals. **This is a hypothesis, not established.**

**Test 1, the shrinkage test (free, post-hoc, run this first).** Take saved predictions, form
`new = mean + lam*(pred - mean)`, sweep lam in [0,1] per horizon, find the best.
- Diagnosis right: best lam well below 1 and falling with horizon. Also an immediate patch with no
  retraining.
- Diagnosis wrong: best lam near 1 everywhere.

**Test 2, bias vs variance.** Split the loss to `train_mean` into aim (bias) and wobble (variance).
Mostly wobble confirms over-commitment; mostly bias means it is the median-vs-mean issue instead,
whose fix already exists (`train/loop.py:84-116`, the `encoder_mc` arm) and is simply not wired into
the transfer/Ebola scoring path (`train/lodo.py:336-369`).

**Test 3, is it really ONE bug.** Correlate per-cell best-lam against per-cell coverage. Moving
together = one cause. Independent = two problems, and stop describing it as one fix.

---

## 7. Standing audit findings not yet closed

From `Reports/Phase0_to_Now_Audit.md` (71 agents, 45 findings survived adversarial verification,
16 refuted, 33 things confirmed done right, no CRITICAL survived):

- **M4** the criterion driver was untracked and unregenerable. `ebola_ci.py` closes the regenerable
  half; git-tracking the decision-bearing logs is still open (`progress/outcomes/` is not ignored).
- **M7** the Ebola cumulative envelope discards cells on a premise false for 69% of them.
- **M8** COVID absent from the authoritative data register (see §2).
- **M12** a client-facing document claims SHAP global+local; no attribution code exists.
- **M13** baselines are two usable comparators, not the four `PROJECT.md:40` still claims.
- The **"unsafe to claim"** list in that report names the exact sentences that must not appear in
  the paper. Check any Ebola sentence against it before publishing.

---

## 8. Conventions to keep

- **Slack updates**: date line alone, one prose paragraph, bold section headers, bullet dots,
  **no em dashes**, junior-dev terms.
- **Commits**: terse lowercase, no `Co-Authored-By` trailer.
- **Who runs what**: assistant runs evaluation/analysis directly; the user runs anything taking
  minutes in their own PowerShell and pastes output.
- **Verification rule**: before signing off a results doc, parse the numbers OUT of the doc and
  recompute them from disk. This caught 6 stale numbers in one brief this session.
- **Do not trust progress docs over disk.** Three times this session the doc was stale and the
  artifacts were right.
- **`/Reports/` and `/results/` are NOT gitignored.** That claim was wrong and stood for weeks. Git
  tracks 47 files in `Reports/`, including `Manuscript_v2.md`, and 1,650 under `results/`, of which
  919 are in `results/lodo/`. What `.gitignore` really covers is `/docs/`, `/figs/`, `/baselines/*`,
  `.claude/`, `CLAUDE.md`, `graphify-out/`, the ablation artifacts, and `*.log`.
- **`*.log` was the live half of that**, and it is now half-fixed: `results/reports/*.log` is carved
  back in as of 2026-09-06, so the 32 decision-bearing run logs have history. Logs anywhere else
  still do not.

---

## 9. Suggested next actions, ordered

1. **Decide G5.** It is REQUIRED and has no code. The scope is settled in
   `progress/decisions/G5_Explainability_Scope.md` (integrated gradients, not SHAP) and three
   decisions sit with the client, including retracting the SHAP row from a table they already hold.
2. **Write the Threats paragraph on normalisation** using the §6 numbers, before a reviewer does.
3. **Get the manuscript under the word limit.** ~13,628 against 12,000. It is already tracked.
4. **Disclose that US-States is our weak panel** (ledger C9). We lose to EpiGNN at all four horizons
   there and no document says so.
5. **Finish the remaining ledger items**, `progress/planning/Gap_Ledger.md` Groups C to E.

**Struck.** *Run the shrinkage test* was action #1 in both of these documents for days after it had
already run. Verdict in `progress/outcomes/Shrinkage_Verdict.md`: the over-commitment diagnosis is
**refuted as a general claim**. Under the honest split it helps 10 cells and hurts 8, median gain
+0.5%, the +6.0% mean is entirely COVID, and dengue h3 fits a lambda of 1.70, meaning dengue wants
*more* variance rather than less. Do not re-open it without new evidence.

**Also struck.** *Wire the median-to-mean correction into the transfer path* was action #2. It is a
decided no, recorded as ledger D3: the pre-registration's amendment log is closed and the correction
is 8.8% to 47.3% worse on COVID, the most-shifted panel and the closest analogue to Ebola. Adopt the
documentation fix instead, that the Ebola point forecast is the count-space median, and lead with
MAE.

# Project Status

**Written 2026-09-07. This is the single current status document.** It replaces the eight dated
progress summaries listed in section 6, which were snapshots taken between 21 July and 10 August and
had accumulated claims that are no longer true.

**Every number here was recomputed from the artifacts on 2026-09-07.** Where a retired document said
something different, the difference is recorded in section 6 rather than quietly dropped.

Read this, then `progress/planning/Gap_Ledger.md` for what is open and
`progress/decisions/decisions.md` for why each call was made.

---

## 1. What the project is

A disease-agnostic spatio-temporal GNN. Train a shared encoder on data-rich diseases, then forecast
a data-scarce emerging disease the model has never seen: Ebola, 2014 West Africa. Deliverables are
the code and a journal manuscript. Goals G1 to G7 are in
`progress/planning/Final Internal Project Brief.md`.

Two conda environments. `ebola` for data work, `ebola-train` for modelling. Always name the
environment.

---

## 2. What is on disk

**All expensive compute is finished. Nothing is queued.**

| family | records | quantile archives | what it answers |
|---|---|---|---|
| single-disease ceilings | 25 | 25 | what the architecture does with no transfer |
| transfer, all LODO families | 208 | 125 | what transfer costs |
| Ebola case study | 20 | 20 | the actual claim |
| published baselines | 280 | n/a | the SOTA comparison |
| joint multi-disease | 24 | n/a | does training everything at once help |
| naive floors | 7 | n/a | the honest reference |
| misc, including ANIL | 31 | n/a | meta-learning and probes |

**118 trained checkpoints**, which is what makes G5 attribution inference-only.
**32 decision-bearing run logs** now under version control (`results/reports/*.log`, carved out of
the `*.log` ignore rule on 2026-09-06).

The 208 transfer records break down as LDO3 proper 60, two-way LDO 120, population LODO 8, and the
graph-controlled `encoder_pair` arm 20. **Any document quoting 158 is stale**; that figure matches no
directory.

Three quantile gaps are real and deliberate: the LDO3 zero-shot arm has 25 records and **0** archives
(decision D20), and neither the two-way LDO nor the population LODO zero-shot arms are complete.

---

## 3. The results, stated accurately

Do not soften these.

**Ebola.** With no Ebola data at all the model beats persistence by 16 / 15 / 16 / 44 percent at
h3/h5/h10/h15 on the primary arm and 19 / 17 / 14 / 42 on the secondary. On the corrected district
bootstrap the wins span every horizon, not just the long ones.

**The pre-registered criterion was NOT met**, and on the originally reported statistic it was **not
adjudicable**. It required the *adapted* model to beat persistence at h3 or h5 with the interval
clearing zero. All four such intervals span zero, closest is h3 RMSE [-11.85, +0.35]. The wins we
have come from the unadapted model, which is not what the criterion asked for.

**Adaptation makes it worse.** The arm given no Ebola data beats the arm adapted on Ebola data at
every horizon and on both support sets, and it holds on the full predictive distribution: on WIS the
zero-shot arm is better in 7 of 8 cells. Reported as a strong observation with a partly significant
test behind it, not as an established rule.

**Transfer is negative.** Symmetric seed-ensembled comparison over 36 cells: 1 better, 10 within
noise, 25 worse. The verdict survived making the opponent stronger.

**The graph does not help accuracy.** Gate-off ablation over 60 cells: the spatial channel helps
error in **0 of 40** and hurts in 8. It helps correlation in 6 of 20. Negative result for the
component the architecture was chosen for.

**Meta-learning does not help, across four folds.** ANIL against its own seed-matched control is
better in 0 of 32 cells, worse in 1, within noise in 31. The one significant cell is in the original
`dengue2flu` fold; all three LDO3 folds are entirely within noise.

**Calibration transfers, and that is the strong result.** The frozen cross-disease correction, which
reads no Ebola outcome whatsoever, lifts coverage from 0.28-0.70 to 0.65-0.98. Online adaptation adds
about 0.03 at short horizons and nothing at long ones.

**Against the published baselines**, two usable comparators on influenza and one on dengue. Over the
40 non-MTGNN cells the encoder is better in 28 and worse in 12. **US-States is our weak panel**, and
which model leads there depends on the aggregation: on country-macro we lose to EpiGNN only at h3, on
cell-pooled RMSE we are behind at every horizon by 2.8 to 13.1 percent.

**The honest thesis.** A shared representation transfers to an unseen pathogen well enough to beat
naive floors and carries its uncertainty calibration with it, but every mechanism added to *improve*
transfer, meaning spatial message passing, few-shot adaptation and meta-learning, fails to help. That
is a boundary-conditions paper and it is publishable as one.

---

## 4. Corrections that must not regress

**The estimand mismatch, fixed in `ebola_ci.py`.** The Ebola headline is a node-averaged
country-macro; the interval originally used to adjudicate the pre-registration was cell-pooled. They
diverge 1.5x to 2.0x, and the headline 38.20 sat outside its own quoted interval. `ebola_ci.py`
rebuilds the macro from `*__pernode.npz`, asserts it matches the scored JSON before printing, and
resamples districts. It is read-only.

**The dengue population bug in `paper_compare.py`, fixed 2026-09-06.** It pooled the encoder over the
full 7,165-node bundle while pooling the baselines over the 2,392-node subsample they ran on. The
encoder's own dengue RMSE is about 50 percent higher on the subsample, so the printed margin was
roughly twice the real one. No published document ever carried the wrong figures.

**Three claims that were made and turned out wrong.** Do not repeat any of them.

1. "The criterion is not met" is incomplete. It is not met AND was not adjudicable as reported.
2. "Pooled normalisation makes the task harder, so our Ebola numbers are a lower bound" is wrong.
   Measured: pooling helps. See section 5.
3. "`Reports/` and `results/` are gitignored" is wrong. Git tracks 47 files in `Reports/` and 1,650
   under `results/`. Only `*.log` outside `results/reports/` is ignored.

---

## 5. The normalisation issue, live and unresolved

Ebola uses `scaler_scope=per_disease_support`, one pooled scale for the whole disease. Every training
panel uses `per_node_train`. Divergence of district means from zero: influenza 0.0000, dengue 0.0387,
ebola_L12 0.6550, ebola_L20 0.5363.

The encoder trains on inputs centred at zero and is handed Ebola inputs scattered by about 0.8 either
side. That is a covariate shift we introduced ourselves, and we publish the diagnostic, so a reviewer
will find it.

Priced on the matched panel, 5 seeds: on `influenza_us-states`, whose pooled severity is 0.88x
Ebola's, **pooling HELPS by 6.8 percent on average**, and at every horizon: -1.1 / -4.3 / -10.1 /
-11.8 percent at h3/h5/h10/h15. The long-horizon gains are about 3 seed SDs.

**Not settled:** that experiment trained AND tested with pooling, consistently. Ebola's real situation
is a mismatch, trunk trained per-node and Ebola arriving pooled. That experiment has not been run.
Also untested on dengue.

**Awkward for the paper:** pooling helps most at h10 and h15, exactly where the Ebola wins are. It
does not invalidate them, because naive floors are scored in raw counts and never touch the scaler,
but it must be stated in Threats before a reviewer states it for us. **That paragraph is still not
written.**

**The over-commitment diagnosis is refuted as a general claim.** The shrinkage test ran on
2026-09-02; under the honest split it helps 10 cells and hurts 8, median gain +0.5 percent, the +6.0
percent mean is entirely COVID, and dengue h3 fits a lambda of 1.70, meaning dengue wants *more*
variance. Do not re-open it without new evidence. Verdict in `progress/outcomes/Shrinkage_Verdict.md`.

---

## 6. What the retired summaries said, and what survives

**Seven of these eight are deleted as of 2026-09-07** and survive in git history. Each row records
what it contributed and where that content lives now, so nothing is lost by retiring it.

**`Doubt.md` is deliberately kept.** Eight source files cite it by section as the reason their code
is the way it is: `bundles.py:49`, `train/loop.py:85`, `run_overnight.py:26`, and five diagnostics
(`covid_eda.py`, `covid_split_probe.py`, `covid_val_probe.py`, `locf_probe.py`). Deleting it would
orphan every one of those comments, so it stays until that rationale is moved into the code itself.

| retired document | dated | what it carried | where it lives now |
|---|---|---|---|
| `Day13_Summary.md` | 2026-07-21 | Days 11-13 handoff: bundle interface, encoder, metrics, naive floors | `README.md`, `PROJECT.md` §6, `decisions_day_13.md` |
| `Phase3_Week3_Day15_Baselines_Status.md` | 2026-07-27 | the baseline suite plan and per-model environments | `Reports/reproduction_failure_log.md`, `REPRODUCIBILITY.md` |
| `Phase3_Week3_Results_and_Direction.md` | 2026-07-24 | the Week-3 go/no-go on direction | superseded by the LDO3 verdict, `decisions.md` D12 |
| `Phase3_Week4_Work_Order.md` | 2026-07-29 | the client's Week-4 orders | **all three deliverables now exist**; orders themselves in `progress/decisions/Review Doc.md` and `client_decisions.md` |
| `Doubt.md` | 2026-08-03 | the gravity-mobility rejection, the `obs_mask` disease-identifier leak, the LOCF finding | leak recorded in `G5_Explainability_Scope.md` §5.2 and `covid_eda.md`; rejection in `decisions.md` |
| `Pair_Run_Analysis.md` | 2026-08-04 | the graph-controlled COVID / influenza-US-states pair | `progress/outcomes/Results_Matrix.md`, `Manuscript_v2.md` §9 |
| `Day15_Progress.md` | 2026-07-30 | the Week-4 live tracker and the MTGNN degeneracy discovery | `Reports/reproduction_failure_log.md` §C1, `Reports/baseline_reproduction_table.md` |
| `Priority_Fix_Progress.md` | 2026-08-10 | the ANIL verdict, the conformal packet, pre-registration hardening, the overnight queue | `progress/outcomes/ANIL_Results.md`, `Ebola_Prereg.md`, `Manuscript_v2.md` §9.4 |

### Claims found stale when these were verified

Sixteen load-bearing claims were checked against disk on 2026-09-07. Thirteen held. Three did not:

1. **`Day13_Summary.md`: "nothing is committed yet."** The repository now has full history.
2. **`Day15_Progress.md`: MTGNN recorded as "80/80 COMPLETE" and a passing control.** The runs are
   complete but **47 of the 80 prediction files hold a single repeated value**. MTGNN is not a usable
   comparator and every "we beat MTGNN" figure derived from it is meaningless. Corrected in the
   stakeholder brief on 2026-09-06.
3. **`Doubt.md` §3.4: "the reference arm has no saved weights."** 25 single-disease checkpoints now
   exist, which is part of why G5 needs no retraining.

What held: all four baseline run counts, both `encoder_pair` arms at 10 records each, both Ebola arms
scored at 20 records, 25 ceiling quantile archives, all three Week-4 deliverables now present, and
`Doubt.md`'s `obs_mask` leak, which is still live and still undisclosed in the paper.

### Superseded constants that appear in the retired documents

Do not carry any of these forward: a **388-parameter** adapter (it is 1,428), an Ebola support set of
**27 cells / 9 districts** or a **2014-05-24** cutoff (superseded by D16), Ebola mask density
**0.5255** (it is 0.4095), **"better in 12 of 16 against MTGNN"**, and **158** transfer records.

---

## 7. What is open

| item | state |
|---|---|
| **G5 explainability** | code underway in `explain.py`, untracked at the time of writing. The only REQUIRED goal without a delivered result. Scope settled in `progress/decisions/G5_Explainability_Scope.md`: integrated gradients, not SHAP |
| **Manuscript** | 13,809 words against a 12,000 limit. Nine tables and zero figures, while `figures/gate.pdf` exists and was client-requested. §9.8 is a stub |
| **The normalisation Threats paragraph** | section 5 above has the numbers; the paragraph is not written |
| **LDO3 zero-shot calibration gap** | decided not to run (D20). The resulting limitation is **not yet stated in Threats** |
| **Three client decisions** | listed in section 8 |

Everything in Gap Ledger Groups A to D is closed. Group E is E1 and E2 above.

---

## 8. What is waiting on the client

All three come from the G5 scoping note.

1. **Adopt integrated gradients as G5, or renegotiate the goal.**
2. **Retract the SHAP row** from `Reports/Phase1/RelatedWork_CompetitiveAnalysis_Benchmark.docx`.
   This is the urgent one: it is a comparison table already in the client's hands asserting a
   delivered capability that does not exist. Audit finding M12.
3. **Confirm they accept a neighbour-attribution figure that cannot be described as explaining
   accuracy**, because the gate-off ablation forbids that framing.

---

## 9. How to work here

**Writing.** Plain English. **No em dashes.** Slack updates: date line alone, one prose paragraph,
bold section headers, bullet dots, junior-dev terms.

**Attribution.** Always "I found", never "the reviewer found" or "the agent found".

**Commits.** Terse lowercase. Never a `Co-Authored-By` trailer.

**Who runs what.** The assistant runs evaluation and analysis directly. The user runs anything taking
minutes in their own PowerShell and pastes the output back.

**Verification.** Before signing off any results document, parse the numbers OUT of the document and
recompute them from disk, then mutation-test the verifier. This has caught a sign bug, a population
bug and six stale numbers in a single brief.

**Search.** Query the graphify knowledge graph at `graphify-out/` before grepping. It is a snapshot,
so confirm on disk for anything changed since it was built.

**The traps this project has already hit.** Progress docs go stale and artifacts do not: when a
document and the disk disagree, **the disk is right**. The audit report has an explicit "unsafe to
claim" list; check any Ebola sentence against it, and note that one entry on that list is itself now
stale (it says all eight Ebola wins sit at h10/h15, which was true only of the superseded cell-pooled
interval). HeatGNN produces NaN on isolated nodes without the self-loop fix.

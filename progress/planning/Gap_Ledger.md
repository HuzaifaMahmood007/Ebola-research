# Gap Ledger

**Opened 2026-09-04.** Live status of every known open gap, worked one at a time, oldest work first.
Rationale lives in `progress/planning/Milestone4_Plan.md`; this file carries only what is done and
what is not.

**Provenance.** Items marked *(verified)* I checked against disk myself. Items marked *(audit)* come
from a sweep of `Reports/Phase0_to_Now_Audit.md` and the progress docs and have **not** each been
re-confirmed on disk; confirm before acting on one.

**Rule for this ledger:** an item is only DONE when the change is committed and, where it produces
numbers, a verifier re-derives them from the artifacts.

---

## Group A: loops left open by work already finished

| id | what | where | status |
|---|---|---|---|
| A1 | Meta-learning result had no write-up | `diagnostics/anil_report.py`, `diagnostics/verify_anil_doc.py`, `progress/outcomes/ANIL_Results.md` | **DONE** `6e9f15e` |
| A2 | Shrinkage experiment had no verdict, no script, no version control | `progress/outcomes/Shrinkage_Verdict.md` | **DONE** `38f5796` |
| A3 | Docs asserted the superseded one-fold meta-learning result | `CLAUDE.md`, `Resume.md`, `Manuscript_v2.md` 9.7 + conclusion | **DONE** `944e5ae` |
| A4 | `REPRODUCIBILITY.md` had no mention of ANIL at all | `REPRODUCIBILITY.md` | **DONE** `ae1eafe` |
| A5 | Stale line references in the two navigation docs | Five per doc, not the two expected. All corrected to the real definitions | **DONE** |

Notes carried out of Group A:

- The `anil_report.py` sign bug (`best_val` negated before `paired_delta`, double-flipping every
  fold's meta-objective) is fixed and now guarded by a selfcheck assertion and a verifier mutation.
- The manuscript grew 13,516 → **13,628 words** against a 12,000 limit, so the overage is now 1,628.
  Deliberate: the stronger four-fold claim was judged worth 112 words. E2 absorbs it.
- `CLAUDE.md` is gitignored, so its edits are local only and will not appear in any commit.
- A5 was five stale references per doc, not the two expected. Corrected to: `score.py:215-237`
  (`aggregate()`), `analysis.py:63-83` (`_macro_dist()`, the cell-pooled estimator),
  `train/loop.py:84-116` (`_fit_bias_correction()`), `train/lodo.py:336-369` (`_score()`), and
  `Reports/Week4_Experiments_Stakeholder_Brief.md:172-174`, which was also missing its directory.
- A sweep found 202 code line-references across 85 tracked markdown files, of which 8 point past the
  end of their file. Six of those 8 sit in dated historical records (the audit, `decisions.md`,
  `Doubt.md`) that are deliberately not rewritten, so **no permanent reference checker was added**:
  it would fail on history we have chosen to keep. Re-run the sweep by hand after any large doc edit.

---

## Group B: client-ordered deliverables never produced

| id | what | where | status |
|---|---|---|---|
| B1 | `reproduction_failure_log.md` does not exist anywhere in the repo | ordered by work order §1d and decision D10; content already drafted at `Phase3_Week4_Work_Order.md:75-95` and `Day15_Progress.md:429-450` | OPEN *(verified absent)* |
| B2 | Published-versus-ours reproduction table never written up | `diagnostics/paper_compare.py` computes it (degeneracy flag at line 95); no output saved anywhere | OPEN *(verified)* |
| B3 | Explainability scoping note (work order §8d) never written | recorded outstanding in three docs | OPEN *(audit)* |

B1 is the cheapest high-value item in the inventory: all six entries already exist in prose
elsewhere (MepoGNN needs an OD matrix we lack; MSGNN needs Ubuntu + CUDA 10.1; STOEP's paper table
disagrees with its shipped dataset and metric; dengue is not like-for-like at 7,165 nodes against a
2,392-node subsample; Cola and Heat are influenza-only; MTGNN emits a constant on 47 of 80 files).
It is assembly, not analysis.

---

## Group C: documents contradicting the disk or the client record

| id | what | where | status |
|---|---|---|---|
| C1 | Amend B5 to admit COVID as a labelled secondary panel | `progress/decisions/client_decisions.md:192-197` | OPEN, decision taken *(verified)* |
| C2 | MTGNN "better in 12 of 16" and "HeatGNN one dataset, two horizons" both false | `Reports/Week4_Experiments_Stakeholder_Brief.md:171-175` | OPEN *(verified)* |
| C3 | Status table stale: "Phase 3 not yet started", G5 as SHAP, MTGNN as a passing control | `PROJECT.md:36-44`, `PROJECT.md:166` | OPEN *(verified)* |
| C4 | Two navigation docs claim `Reports/` and `results/` are gitignored; they are not. `*.log` is | `CLAUDE.md` §7, `Resume.md` | OPEN *(verified)* |
| C5 | "Run the shrinkage test" still listed as next action #1 | `CLAUDE.md:219`, `Resume.md:212` | OPEN, depends on A2 *(verified)* |
| C6 | M8: three sites still deny COVID enters the schema | `data_audit.md:864-870`, `:1414-1423`, `:1569` | OPEN *(audit)* |
| C7 | M7: Ebola cumulative envelope discards cells on a premise false for 69% of them | `to_schema.py:226-233`; no per-district masked-week table in `data_audit.md` §3.5 | OPEN *(audit)* |
| C8 | Run the audit's "unsafe to claim" list over every Ebola sentence in the manuscript | `Reports/Phase0_to_Now_Audit.md:246-265` | OPEN *(audit)* |

**C7 carries a warning.** Re-basing would change `data/processed/ebola_L12.npz`, which is a
hash-frozen arm. The *disclosure* half is safe; the *fix* half engages the pre-registration and must
not be done casually.

**C8 is a review pass, not an edit**, and is the last Group C item because it gates Group E touching
the paper.

---

## Group D: pending runs, and the decisions about them

| id | what | decision | status |
|---|---|---|---|
| D1 | LDO3 zero-shot quantiles, 0 of 25 present, ~10 h | **Do not run.** Zero-shot is behind its ceiling in 36 of 36 cells; calibrating a uniformly losing arm buys no claim | decision recorded, needs writing into the limitation |
| D2 | Shuffled-adjacency control, ~6 h | **Do not run.** Gate-off already shows 0 of 40 error cells helped; absence already disclosed in Threats | decision recorded |
| D3 | Median-to-mean correction into the Ebola path | **Do not.** Prereg amendment log is closed; correction is 8.8–47.3% worse on the most-shifted panel. Adopt M14's documentation fix instead: point forecast is the count-space median, lead with MAE | OPEN as a doc task |
| D4 | Baselines | **Nothing pending.** EpiGNN 80/80, MTGNN 80/80, ColaGNN 60/60, HeatGNN 60/60, all scored | closed *(verified)* |

---

## Group E: new work, only after A to D are clear

| id | what | note | status |
|---|---|---|---|
| E1 | G5 explainability, the only REQUIRED goal with no code | 118 checkpoints on disk means this is pure inference: no retraining, no GPU night. Input surface is exactly `[N, 20, 4]`. Integrated gradients over channels and lags, inference-time occlusion as the faithfulness cross-check, neighbour edge ablation on Ebola only, plus the existing gate figure. Method and falsification test already specified at `Phase3_Developer_Execution_Guide.md:317-325` | OPEN |
| E2 | Manuscript: cut to 12,000 words, place a figure, replace the §9.8 stub, add the normalisation Threats paragraph | Currently 13,628. Intro 646 + Contributions 523 + Conclusion 525 is ~1,694 words of largely restated material; Related Work 3.1–3.5 another 1,009. **Nine tables and zero figures**, while `figures/gate.pdf` exists and was client-requested | OPEN |

**Two honesty constraints for E1.** The gate-off ablation says the spatial channel does not help
accuracy, so any neighbour attribution must be framed as *where the model draws from*, never as
*what makes it accurate*. And `obs_mask` is constant-1 on the fully observed development panels, so
its attribution is structurally near-zero there and only carries meaning on Ebola.

---

## The audit long tail

From the sweep of `Reports/Phase0_to_Now_Audit.md` and the progress docs. **Not individually
re-verified.** Kept here so it is not lost; confirm each against disk before acting.

**Cheap doc fixes:** stale Ebola mask density 0.5255 (actual 0.4095) in two live tables
(`data_audit.md:1403`, `:1423`); "new cases series adopted as a cross-check" asserted present-tense
with no such code (`data_audit.md:997`, `PROJECT.md:121-123`); retract the SHAP row from the
client-held comparison table (`Reports/Phase1/RelatedWork_CompetitiveAnalysis_Benchmark.docx`);
relabel `ebola_bootstrap.log`'s cell-pooled verdicts; docstrings claiming quantile sorting cannot
move the median when it moves in 100% of L12 h15 cells (`score.py:130-132`,
`models/adapters.py:31-32`); `models/windows.py:21-23` justifying the zero pad on the opposite
normalisation convention; a residual sweep of superseded constants (388-param adapter, 27-cell
support, "two folds" over three).

**Code items:** `analysis.py` hardcodes `NAIVES` and `--dataset choices=DEV` excludes both Ebola arms
(M4); carve `results/reports/*.log` out of `.gitignore:26` so decision-bearing logs get history;
`LOWER_BETTER` omits nrmse/peak_timing/peak_intensity (`analysis.py:186`); `node_mean` promised
beside `country_macro` but never printed (`ebola_report.py:47`); order-statistic "within noise" rule
still live and contradicting the bootstrap (`ebola_report.py:94`); no constant-prediction guard on
comparator readers; `invert_scaler` has no zero floor and 6,736 archived Ebola quantiles are negative
(`to_schema.py:189-192`); per-horizon regime labels print h10/h15 under a few-shot header; Ebola PCC
violates its own ≥5-cell rule; `per_country` computed but dropped from every record (M9);
`trunk_steps: 91000` advertised while trunks ran 32–35k; no BH/Holm on the 64-cell exploratory grid.

**Runs, none of them recommended:** `nrmse` missing for four zero-shot LODO files and structurally
un-backfillable; the two-disease LDO flu fold has 4 seeds not 5; the pooled-versus-per-node
*mismatch* experiment (trunk per-node, Ebola pooled) never run; HeatGNN and Cola paper-grid cells at
the papers' own horizons.

---

## Already closed but still written up as open

Do not redo these. *(audit, spot-checked)*

M1/M2/M3 estimand and district bootstrap, closed by `ebola_ci.py`. M5 oracle-feedback ACI, closed;
blast radius one cell. M6 Ebola UQ block, closed by `diagnostics/ebola_uq.py`. M10 spatial channel,
run and answered negatively. M11 uncited prior art, cited and the false sentence retracted in the
manuscript. HeatGNN's 51 parked runs, scored. ANIL "never started", it ran. The shrinkage test,
it ran. `Reports/` and `results/` being gitignored, they are not.

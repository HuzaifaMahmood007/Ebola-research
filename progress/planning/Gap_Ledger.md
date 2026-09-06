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
| B1 | `reproduction_failure_log.md` did not exist anywhere in the repo | `Reports/reproduction_failure_log.md`, verified by `diagnostics/verify_repro_log.py` | **DONE** |
| B2 | Published-versus-ours reproduction table never written up | `Reports/baseline_reproduction_table.md`, verified by `diagnostics/verify_paper_table.py` | **DONE** |
| B3 | Explainability scoping note (work order §8d) never written | `progress/decisions/G5_Explainability_Scope.md`, verified by `diagnostics/verify_g5_scope.py` | **DONE**, three decisions now sit with the client |

Notes carried out of B1:

- It was mostly assembly, but **not entirely**. Every disk-backed number was recomputed rather than
  copied: MTGNN 47 of 80 constant files confirmed, and the same check returns 0 for EpiGNN, Cola-GNN
  and HeatGNN; degree-0 nodes are 2 on japan, 2 on us-states, 0 on us-regions; dengue is 7,165
  against 2,392, so 33.4 percent retained; baseline records are 80/80/60/60.
- **The STOEP entry changed.** The recorded reason for excluding it, that the paper's table
  disagrees with the shipped dataset by roughly threefold, does not survive. The repo ships exactly
  one dataset, `od (539, 47, 47, 1)` and `node (539, 47, 4)`, which is the paper's COVID-19 panel,
  and the Zhejiang flu data is not shipped at all. Against the COVID row the run is **8.2 percent
  off on RMSE**, inside the field's own 9 to 26 percent tolerance. STOEP was not a failed
  reproduction. It stays excluded because it needs an OD tensor and our schema ships `A_mob = None`,
  which is the MepoGNN blocker, not a reproduction blocker.
- The old STOEP sentence still stands in `Phase3_Developer_Execution_Guide.md:243`,
  `Phase3_Week3_Developer_Execution_Guide.md:509` and `Phase3_Week4_Work_Order.md` §1d. Those are
  dated execution records and are **not** rewritten, same precedent as A5. The failure log is the
  live statement and it names the correction explicitly.
- `diagnostics/verify_repro_log.py` re-derives every disk-backed number from the artifacts and
  mutation-tests itself over 9 mutations, all caught. It exits 2 rather than 1 when `baselines/` is
  absent, since that directory is gitignored and "cannot check" is not "wrong".

Notes carried out of B2:

- **The generator had a population bug and it was load-bearing.** `paper_compare.py` pooled the
  encoder over the full dengue bundle (6,161 non-constant nodes of 7,165) while pooling the
  baselines over the 2,392-node subsample they actually ran on. The encoder's own dengue RMSE is
  about **50 percent higher** on the subsample, so the printed margin was roughly twice the real
  one. Corrected margins against EpiGNN are **-28.9 / -26.2 / -11.2 / -6.0 percent** at h3/h5/h10/h15,
  against -53.5 / -51.7 / -41.6 / -37.5 as first generated. Direction unchanged, magnitude halved,
  and h15 is now 6.0 percent against seed dispersion of ±1.9 and ±11.5.
- The fix restricts the encoder column to `export_baseline._kept_indices` for any subsampled dataset,
  so the table cannot drift from the export. The table also gained a `-o` flag, because the Windows
  console mangles the em dash and plus-minus on the way to a file.
- **No published document carried the uncorrected numbers.** I checked every tracked markdown for
  the four wrong margins and found none, so this never escaped the generator.
- **New finding, undisclosed anywhere: US-States is our weak panel.** Against EpiGNN the encoder is
  worse at all four horizons (+2.8 to +13.1 percent) and against HeatGNN at three of four. Over the
  40 non-MTGNN cells the encoder is better in 28 and worse in 12. This belongs in the manuscript
  results section, not only in the table. Tracked as **C9** below.
- MTGNN fails the client's condition twice over, and either reason alone is sufficient: its paper
  has no epidemic dataset, so all 16 cells have an empty `Δ% vs published` column, and 47 of 80 of
  its runs are constant.
- `diagnostics/verify_paper_table.py` re-runs the generator to prove the filed table is current, then
  re-derives every prose tally from the filed table text. 11 mutations, all caught.

Notes carried out of B3:

- **Decision taken: integrated gradients, not SHAP.** KernelSHAP does not fit dengue at 7,165 nodes
  over 1,409 steps, and GradientSHAP on a 4-channel 20-lag surface is integrated gradients with a
  sampler bolted on. Calling a sampled gradient approximation "SHAP" would also claim an efficiency
  axiom we would not be honouring, which is exactly the kind of sentence the audit's unsafe list
  exists to stop.
- **I corrected my own earlier planning claim.** I had written that `obs_mask` is constant-1 on the
  fully observed development panels. Measured: constant 1.0 on all three influenza panels **and
  COVID**, but it varies on **dengue**, where only **21.75 percent** of input cells are observed, and
  on Ebola at 40.95 percent. So attribution to that channel is structurally zero on four of five
  panels and a visible disease-identifier leak on the fifth. `Doubt.md` §3.3 had this and the
  planning text did not.
- Gate-off ablation recomputed from `Reports/gate_ablation.log`: over the 40 error cells the graph
  helps in **0** and hurts in 8; over the 20 PCC cells it helps in 6 and hurts in 1. That is what
  forbids framing neighbour attribution as a source of accuracy. The log's own ceiling caveat
  travels with it: g=0 removes neighbour mixing but keeps the LTR degree feature, so the tally bounds
  the value of neighbour information, not of the graph in total.
- 118 checkpoints confirmed on disk, so G5 is inference-only: 26 single-disease, 15 Ebola plus 3
  smoke, 15 LDO3 plus 3 full-budget, the balance ANIL and its controls.
- The trunk surface is `[N, 20, 4]` on every panel via `core_feature_idx`. The Ebola bundles carry a
  fifth channel, `deaths_norm`, that the trunk never sees, so it is not an attribution target.
- `diagnostics/verify_g5_scope.py` catches 14 mutations. Three of its first failures were verifier
  bugs, not document bugs, the worst being that the ablation log's own legend line
  (`'within noise' = |mean| < sd`) contains both a pipe and the phrase and was being counted as a
  21st PCC cell.
- **Three decisions now sit with the client** and are listed in the note: adopt integrated gradients
  or renegotiate G5; retract the SHAP row from the comparison table already in their hands and fix
  `PROJECT.md:39` and `:206` (that is C3's territory); and confirm they accept a neighbour figure
  that cannot be described as explaining accuracy.

---

## Group C: documents contradicting the disk or the client record

| id | what | where | status |
|---|---|---|---|
| C1 | Amend B5 to admit COVID as a labelled development panel | `progress/decisions/client_decisions.md:192-211`, amendment block under the original, which is kept as the returned record | **DONE** |
| C2 | MTGNN "better in 12 of 16" and "HeatGNN one dataset, two horizons" both false | `Reports/Week4_Experiments_Stakeholder_Brief.md`, corrected in place with a dated note, `.docx` re-rendered | **DONE** |
| C3 | Status table stale: "Phase 3 not yet started", G5 as SHAP, MTGNN as a passing control | `PROJECT.md` §2 goal table, §7 baseline table, §Week-5 list, §9 outputs row | **DONE** |
| C4 | Two navigation docs claim `Reports/` and `results/` are gitignored; they are not. `*.log` is | `CLAUDE.md` §7, `Resume.md` §2 and §8, `.gitignore` | **DONE**, and audit M4's live half is now half-closed |
| C5 | "Run the shrinkage test" still listed as next action #1 | `CLAUDE.md` §9, `Resume.md` §9, both replaced | **DONE** |
| C6 | M8: three sites still deny COVID enters the schema | `data_audit.md:864-870`, `:1414-1423`, `:1569` | OPEN *(audit)* |
| C7 | M7: Ebola cumulative envelope discards cells on a premise false for 69% of them | `to_schema.py:226-233`; no per-district masked-week table in `data_audit.md` §3.5 | OPEN *(audit)* |
| C8 | Run the audit's "unsafe to claim" list over every Ebola sentence in the manuscript | `Reports/Phase0_to_Now_Audit.md:246-265` | OPEN *(audit)* |
| C9 | US-States losses to EpiGNN and HeatGNN appear in no document | found in B2; `Reports/baseline_reproduction_table.md` has the cells, the manuscript results section does not | OPEN *(verified)* |

**C7 carries a warning.** Re-basing would change `data/processed/ebola_L12.npz`, which is a
hash-frozen arm. The *disclosure* half is safe; the *fix* half engages the pre-registration and must
not be done casually.

**C8 is a review pass, not an edit**, and is the last Group C item because it gates Group E touching
the paper.

Notes carried out of Group C:

- **C1.** The amendment was written 2026-09-06 and sat uncommitted until Group C was worked. Its
  three factual claims are now checked against disk rather than repeated: `covid_us-states` is
  49 nodes x 164 weeks x 4 channels, its `A_geo` is **bit-identical** to `influenza_us-states`
  (same sha256, `fddc2673a17e...`), and the client instruction it quotes is at `Review Doc.md:97`.
  The original two sentences are kept above the amendment as the record the client returned.
- **C1 turned up an off-by-one inside the manuscript.** Table 2 lists five development panels and
  the paragraph three lines below called COVID "a fourth training panel". Corrected to "a fifth
  training panel and a third training disease", which is what the table says.
- **C2.** Both flagged sentences were false. MTGNN is now excluded from the head-to-head rather than
  counted, with the constant-output diagnostic given in plain terms, and the naive-floor paragraph
  drops its MTGNN figure for the same reason. HeatGNN has finished on all three influenza panels at
  all four horizons: **better in 3 of 12, worse in none, 9 too close to call**, against the brief's
  "one dataset and two horizons".
- **C2: I reproduced the brief's own conventions before changing anything.** Its head-to-head rule is
  RMSE on `country_macro`, five seeds, level when |mean delta| < sd; that reproduces its EpiGNN
  "9 of 16, worse in 1" and Cola-GNN "4 of 12, worse in none" exactly. Its naive-floor rule is
  **sign only**, which reproduces "ours 6 of 16, EpiGNN 5, Cola-GNN 5 of 12, MTGNN 2" exactly. So the
  document uses two different rules in adjacent paragraphs. Both counts stand; the inconsistency is
  noted, not fixed, since fixing it would change numbers the client already has for no gain in truth.
- **C2: the correction note claims only what I checked.** First draft said every other figure in the
  document had been rechecked. That was untrue: I checked the head-to-head and naive-floor tallies,
  not the transfer or run-count figures. The note now says so.
- **C3.** Four sites fixed in `PROJECT.md`: the G1-G7 status table, the Phase-1 baseline outcome
  table, the Week-5 plan line promising SHAP, and the §9 outputs row. Phase 3 is now recorded as
  complete rather than not started, and G5 as scoped-not-built with a pointer to its decision.
- **C3 record counts are directory counts, recounted 2026-09-06 and checked back out of the file:**
  `results/single/` 25, `results/lodo/` **208**, `results/ebola/` 20, `results/baselines/` 280.
  Note 208, not the 158 that `CLAUDE.md` and `Resume.md` still carry. The 208 breaks down as
  LDO3 proper 60 (25 adapted, 25 zero-shot, 5 + 5 full-budget), two-way LDO 120, population LODO 8,
  and the graph-controlled `encoder_pair` arm 20. Whatever 158 counted, it is not the directory.
- **C3 closed audit finding M13** in the same edit: the baseline table now reads two usable
  comparators on influenza and one on dengue, with each exclusion named.
- **C4 recounted the tracked files**, since the figures in the ledger were themselves stale: git
  tracks **47** files in `Reports/` (not 45, my two new reports landed) and **1,650** under
  `results/`, of which 919 are `results/lodo/`.
- **C4 took the M4 decision rather than deferring it again.** `results/reports/*.log` is now carved
  out of the `*.log` rule and the 32 decision-bearing logs are committed. They total about 1.1 MB,
  which is nothing, and the shrinkage sweep nearly being lost is the argument. `*.log` elsewhere
  stays ignored, so M4 is half-closed, not closed.
- `Resume.md` also had two other stale rows fixed in passing: the manuscript word count (13,516 to
  13,628) and the MTGNN row, which C2 had already closed.
- **C5 struck two dead actions, not one.** The shrinkage test was #1 and had already run; its
  verdict, re-read from `Shrinkage_Verdict.md` rather than memory, is that the over-commitment
  diagnosis is refuted as a general claim (helps 10, hurts 8, flat 2 of 20; mean +6.0% but median
  +0.5%; dengue h3 fits lambda 1.70, wanting *more* variance). Wiring the median-to-mean correction
  into the transfer path was #2 and is decided against as D3. Both are now recorded as struck, with
  the reason, so they stop resurfacing.
- Both lists were rewritten around what is actually open, and C9 was promoted into them.
- **C4 retro-fixed a warning in `Shrinkage_Verdict.md`.** It said its two source logs had no version
  history and that the document was the only surviving record. Both are now committed under
  `results/reports/`, and the note says so.

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

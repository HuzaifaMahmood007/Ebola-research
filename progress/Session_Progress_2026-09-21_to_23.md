# Milestone 6 session record, 2026-09-21 to 2026-09-23

Written at a pause point on 2026-09-23. This is the session-level record of the Milestone 6
closing runs. Per-run detail lives in the outcome docs named below, each with its own verifier.
When this doc and the disk disagree, the disk is right.

## Where Milestone 6 stood at the start

Milestone 6 is Week 6 of `progress/planning/Final Internal Project Brief.md`: ablations and
robustness, submission-ready manuscript, reproducibility package. On 2026-09-21 I audited the
whole tree against the docs and found one order-of-magnitude gap: the client's baseline request
at `progress/decisions/Review Doc.md:79` (ARIMA or SARIMA plus a gradient boosted model on lag
features) had never been actioned and never been recorded as declined. That set the run plan:
classical baselines first, then the two cheap mechanism closers, then the reopened D2 retrain.
The plan is `progress/planning/Milestone6_Plan.md`, the task list is `todo_Milestone6.md`.

## Run 1. Classical baselines. DONE, with one leak found and fixed

- Built `run_classical.py`: GBM on lag features (sklearn HistGradientBoosting, direct
  multi-horizon, 5 seeds) and SARIMA/ARIMA (statsmodels, per node, fixed order (2,1,1),
  seasonal s=52 on influenza only). Same splits, origins, horizons, metrics and record schema
  as every other baseline. Ebola arm is EXPLORATORY under `experiments/`, never touches the
  frozen pre-registration.
- **I found a horizon leak in the first SARIMA/ARIMA run.** Reported error was flat or falling
  with horizon, which is impossible for a real forecast. Cause: statsmodels reads an integer
  `dynamic` argument as an offset relative to `start`, so dynamic forecasting never activated
  and every h-step value was a one-step-ahead prediction reading the observed test series.
  Fixed (`dynamic=0` plus a per-node explosion guard), re-run clean 2026-09-22. The leaked
  records were overwritten in place. Nothing from the leaked run is quotable anywhere.
- Honest verdicts on the corrected records (2-seed-SD noise rule, country-macro RMSE):
  - GBM beats our encoder in 0 of 16 like-for-like cells, and loses to plain persistence in
    8 of 16 clean cells (COVID h3/h5/h10, US-Regions all four, US-States h3).
  - SARIMA beats our encoder in exactly 1 of 20 cells (COVID h5; MAE also h10). Everywhere
    else the encoder wins (12) or the gap is noise (7). SARIMA beats persistence in 13 of 20.
  - Seasonality pays only on influenza (SARIMA 1.8 to 30.0 percent better than ARIMA there);
    SARIMA and ARIMA are identical on COVID and dengue.
  - Dengue: the explosion guard fell back to persistence on 792 of 2,051 districts (38.6
    percent), so dengue classical rows are a hybrid and are always footnoted as such.
  - Estimand fence, permanent: dengue classical and GNN records score the 2,392-node subset;
    the encoder and naive floors score the full 6,161 nodes. Never on one dengue axis.
- Open on this thread: the SARIMA/ARIMA test-window alignment is asserted by construction and
  not re-verifiable from the summary-only records.

## Run 2. Constrained-adapter causal test. DONE. Verdict: partially causal

`experiments/adapter_constraint.py`, EXPLORATORY, support-only selection (leave-one-district-out
inside support), both arms, 5 seeds. Write-up `progress/outcomes/Adapter_Constraint_2026-09-22.md`,
verifier `diagnostics/verify_adapter_constraint_doc.py` (11 planted mutations, 11 caught).

- Primary L12 arm: shrinking the fitted adapter toward the zero-effect map at the
  support-selected strength (t about 0.34) recovers 77 to 90 percent of the few-shot damage
  at h3/h5/h10. h15 has no damage to recover.
- Secondary L20 arm: only 9 to 19 percent recovered. L20's support is full or near-full rank,
  so support-only cross-validation cannot see that the query lies elsewhere and picks almost
  no shrinkage (t about 0.88). The damage is fully removable in the limit (t=0 recovers about
  100 percent everywhere), but an honest rule cannot find that setting on L20. That failure
  mode is itself a finding.
- The clip variant (delete only the strictly out-of-span part) recovers about +1 percent, so
  the damage lives in the magnitude of the fitted change along weakly constrained in-span
  directions, not literally outside the span. This refines the mechanism doc's framing.
- Instrument note: a literal ridge through the pre-registration's optimiser saturates under
  gradient clipping (never reaches the zero-effect map), so the test uses closed-form affine
  shrinkage whose two ends reproduce the archived arms exactly.
- Standing limit: one mechanism test, never "THE cause".

## Run 3. Input-vs-representation district energy. DONE. Verdict: present and discarded

`diagnostics/graph_probe/t5_input_energy.py`, machine-readable summary
`results/misc/t5_input_energy.json`, write-up `progress/outcomes/Input_Energy_2026-09-22.md`.
The script first reproduced the published representation shares (3.8/8.6/12.3/4.8/22.4) to the
printed digit before measuring anything new, with a mutation-tested assert.

- Input incidence windows are 13.8 to 93.5 percent district-specific in energy; the encoder
  output retains 3.8 to 22.4 percent. A 3.2x to 5.0x reduction on every panel.
- Effective dimensionality is already narrow at the input (about 2 to 4 directions), so the
  encoder changes the share of energy, not the dimensionality.
- Fences: the fair comparison is the incidence channel only (calendar channels are identical
  across districts by construction); and energy accounting cannot prove the discarded
  variation was forecastable signal rather than quiet-node noise amplified by per-node
  scaling. Single-disease checkpoints, dengue subsampled, seed 42.

## Run 4. D2 shuffled-adjacency retrain. DONE. Verdict: the map is barely needed, dengue excepted

Reopened by the user 2026-09-21 (ledger D2 history preserved). Runner
`ablation/run_shuffle_adjacency.py`, permutation hook in `train/loop.py` (tracked). The user ran
the full 5-panel x 5-seed matrix overnight 2026-09-22 to 23; measured 9.7 hours, not the ledger's
6 hour estimate. Scored 2026-09-23: `progress/outcomes/Shuffled_Adjacency_2026-09-23.md`,
verifier `diagnostics/verify_shufadj_doc.py` (4 of 4 mutations caught). All permutations verified
non-trivial (70 to 100 percent of nodes displaced; identity permutations are refused by design).

- 49 of 60 cells: the model trained on the wrong districts matches the real-graph model.
- The real map earns a small seed-stable advantage in 9 cells, 8 of them dengue: PCC at all
  four horizons (+0.034 to +0.042) and error at h10/h15 only (about 2 to 4 percent). Influenza
  Japan h10 favours the shuffled graph in 2 cells.
- The strong sentence "the graph contributed nothing at any stage" is NOT supported. The
  supported sentence: training barely needs the real map, and the one exception is dengue,
  which is also the panel where run 3 shows district information best survives the encoder.
  Consistent with gate-off: the graph helps shape, never error magnitude.
- The two false manuscript sentences (":332 and :525, "we did not run a shuffled-adjacency
  arm") have exact replacements drafted and are being applied in the manuscript pass.

## Multiplicity sensitivity. RUN, but the survivor count is DISPUTED and not yet quotable

`ebola_ci.py` rerun at Bonferroni percentiles for a 16-cell family ([0.15625, 99.84375]) by two
independent passes that disagree on the survivor count because they built different families
called "14": one counted 9 of 14 (RMSE-only, floors including support_mean); the other,
recomputing with ebola_ci's own loaders, counted 7 of 14 (5 RMSE + 2 MAE, persistence only,
reading the canonical 14 as 8 RMSE + 6 MAE vs persistence). This is the
three-numbers-near-each-other trap again. What both passes agree on, and what is safe to
carry: the long-horizon h15-versus-persistence intervals survive the correction on both arms,
and survivors are dominated by the unadapted zero-shot arm. BLOCKED before the manuscript
sentence: pin the canonical 14 to its defining artifact (which comparisons, which metric,
which floors) and recompute once. Do not write "9 of 14" or "7 of 14" until then; the
correctness-pass doc carries the same flag.

## Figures. DONE

Six manuscript figures, `figures/F1_*.png/pdf` through `F6_*`, generated by `paper_figures.py`
(repo root), which reads only disk artifacts and asserts per-figure spot checks
(mutation-tested: corrupting a source JSON kills the build before any figure writes). Every
figure carries its direction ("lower is better") on its face. F1 spatial ablation, F2 ANIL,
F3 epi penalty, F4 window sweep, F5 Ebola adapted vs unadapted (E6-compliant labels; the
footnote error claiming L12 h10 has zero adaptation pairs was caught and corrected to 18,
with the pair counts 48/38/18/0 pinned by an assert), F6 baseline comparison including the
corrected classical models. Suggested captions live in the script. `figures/` is tracked, so
the images can carry history once committed.

Session dashboard (classical baselines, live): https://claude.ai/artifact/AWXHQSfJxqqTauRbVLYtcW

## In flight at the pause point

Two agents were mid-task when the user paused; their results land in their own docs:

1. **Run 5 gate probe** (aux district-identity objective, user-ordered 2026-09-23): a cheap
   probe of whether the discarded district information is forecast-relevant at all, gating the
   overnight retrain. GO/NO-GO with numbers goes into `Milestone6_Plan.md`, `todo_Milestone6.md`
   and `progress/outcomes/AuxGate_Probe_2026-09-23.md`. The overnight runner is written only on
   a GO. Known bound on the upside, from run 4: even the real map earns only 2 to 4 percent,
   dengue only, on the current encoder.
2. **Manuscript correctness pass** on `Reports/Manuscript_v2.md`: the two D2 sentences, the
   graph mechanism paragraph, the adaptation mechanism paragraph, section 9.8 replacing the
   stub, the normalisation Threats paragraph (consistent vs mismatched pooling, never
   unqualified), the Bonferroni sentence, the E6 label fix in Tables 6 and 8, and placement of
   the six figures. Word count grows in this pass by design; the trim is the pass after.

## Decisions waiting on the user

1. **The commit.** Untracked and decision-bearing: `run_classical.py`, `paper_figures.py`, the
   twelve figure files, `experiments/` (norm probe and adapter constraint), four outcome docs
   with verifiers, `t5_input_energy.py` and its JSON, `diagnostics/verify_norm_probe_doc.py`,
   `Milestone5_Report.md`, `todo_Milestone6.md`, `progress/planning/Milestone6_Plan.md`, this
   file, and `ablation/run_shuffle_adjacency.py` (needs `git add -f`, `/ablation` is ignored).
   Everything decision-bearing since 2026-09-21 currently has no version history.
2. **The word-count target.** The 12,000 limit exists only in internal docs; no target journal
   is on record and the brief assigns that decision to the client. Confirm the venue before
   the trim pass cuts toward a number that may not exist.
3. **Run 5 GO/NO-GO overrule**, once the gate probe reports.

## Corrections to standing docs found this session

- `Reports/Milestone5_Report.md:257` is wrong twice (says spatial removal never improved error
  in any of 60 cells; truth is 40 error cells and removal significantly helps in 8). Still
  unfixed, and a rendered docx sits beside it.
- The D2 "about 6 hours" estimate in the ledger and plan was wrong; measured 9.7 hours with
  dengue, about 45 minutes without.
- CLAUDE.md staleness carried from the 2026-09-21 audit: LDO3 records are 208 not 158;
  `Reports/`, `results/` and `figures/` are tracked, not ignored; G5 is closed; the adaptation
  failure now has a mechanism; the normalisation section needs the consistent/mismatched split.
- `progress/STATUS.md` still fails `diagnostics.verify_status` on three doc-lag numbers and
  predates every run in this record. Refresh is queued in todo housekeeping, not done.

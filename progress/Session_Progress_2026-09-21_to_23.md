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

## Multiplicity sensitivity. RESOLVED 2026-09-28. Quotable

The dispute was two passes building two different 32-cell families and both calling the answer
"the 14". Pass A used RMSE only across both naive floors. Pass B used both metrics against
persistence only. Pass B was right, and the deciding artifact is
`Reports/Phase0_to_Now_Audit.md:257`: against persistence 14 of 32 cells clear zero, against
support_mean 13 of 32, all 64 together 27, and the 12-of-16 unadapted plus 2-of-16 adapted split
on that same line sums to exactly 14. So the canonical family is **persistence only, RMSE and
MAE, both arms, both regimes, four horizons, 32 cells, 14 clearing zero at 95 percent.**

The divisor is **32**, matching the full family, not the 16 the earlier draft used. That is the
user's call and it is the conservative one: buying extra survivors with a soft divisor would be a
bad trade for a paper that rests on reporting its nulls straight.

I added `--bonferroni M` and `--floor` to `ebola_ci.py` and recomputed, B=10000, seed 0,
stratified. The uncorrected run is bit-identical to the archived
`results/reports/ebola_district_ci.log` on every persistence line, so no existing number moved.
The divisor-16 control reproduced the documented 7 of 14 (5 RMSE, 2 MAE) exactly, which is what
licensed the divisor-32 run. At divisor 32 the answer is again **7 of 14, 5 RMSE and 2 MAE**, the
same seven cells, so the count does not depend on the choice of divisor. Logs:
`results/reports/ebola_district_ci_bonferroni32.log` and `..._bonferroni16.log`.

I mutation-tested the check three ways before believing it: disabling the correction moves the
RMSE count from 5 back to 8, adding 5 cases per district to the encoder drops it to 0, and the
estimand gate refuses a perturbed rebuild. All three were caught.

Both clauses that were being carried as safe, rechecked. The h15-versus-persistence comparison
survives on both arms **on RMSE only**, L12 [-61.936, -2.568] and L20 [-60.714, -0.939]; on MAE
only the primary arm survives, because L20 zero-shot h15 MAE widens to [-29.394, +0.885]. Write
"on RMSE", never the bare "on both arms". The unadapted-dominance clause holds and is stronger:
5 of 7 survivors are zero-shot and the other 2 are the primary-arm h15 cell that pre-registration
E6 labels zero-shot anyway, so no survivor is an adapted win. The replacement manuscript sentence
is in `progress/outcomes/Manuscript_Correctness_Pass_2026-09-23.md` EDIT 6, and it names the floor
family inside the sentence, which is the rule this dispute produced.

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

## Epi bound and lambda sweep. DONE 2026-09-28. Verdict: no PASS in 18 units

User-ordered 2026-09-25. Result `progress/outcomes/Epi_Bound_Lambda_2026-09-28.md`, verifier
`diagnostics/verify_epi_bound_doc.py` (393 checks, 45 of 45 mutations caught). Protocol
`progress/decisions/Epi_Bound_Lambda_Protocol.md`, sha256 `458184bc...b7f3`, commit `b24063f`,
verifier `diagnostics/verify_epi_bound_protocol.py` (185 checks, 31 of 31). Results commit `a0052b9`.
Records `experiments/epi_bound_lambda/single/`, 90 JSON plus 90 per-node archives.

**Why it ran.** The two released lambda-1 arms (p99max, p90max) were null for arithmetic reasons,
not measured ones. I found the p90 penalty was only 0.005 to 0.1 percent of the loss, and that "max
across datasets" made r_max COVID's own number, so at p99 the hinge never fired on either US panel.
The user ordered (1) a better disease-agnostic bound at lambda 1 and (2) lambda 10 and 100 at a
fixed bound.

**Process, in order.**

1. **Free sweep before any GPU** (`ablation/epi_penalty.py --sweep`). I fixed three defects in it
   first: `min` was missing from the grid; `bundles.load` has no cache, so every row reloaded every
   bundle (now 6 seconds for 18 rows); and the pooled percentages were a dengue statistic, because
   dengue carries about 97 percent of transition pairs (now calibrated on all five panels, scored on
   the four that train).
2. **Runner fixes.** Lambda is now in the filename, and lambda 1 stays byte-identical, so the 40
   released records still resolve. The selfcheck covers it and is mutation-tested, 3 of 3 caught; the
   mutation test first caught a bug in itself (it patched a second copy of the module). I found the
   skip-if-finished check had never worked: it looked in `ablation/` while records route to
   `ablation/single/`, so every earlier run retrained every cell. New arms write to `experiments/`.
   Records now carry bound, lambda, penalty share and protocol sha256. `train/loop.py` gained 2 code
   lines inside the epi branch to accumulate the pinball loss, the share's denominator.
3. **Protocol, checked before it was frozen.** An independently written verifier caught 4 wrong
   numbers of mine (a table cell on a different seed basis, a range's low end, a line citation, a
   runtime off by 4.7x) and 4 loose wordings. All were fixed before the commit.
4. **Smoke cell**, COVID seed 42 at p99 median lambda 100, to confirm the provenance fields. Then 89
   cells run overnight by the user.
5. **Scoring and write-up.** Verdicts computed from the raw records, because the runner's report does
   not apply the pre-registered rule (caveats below). A second independent verifier caught 3 prose
   errors in the results doc, all fixed before commit. No verdict or table cell changed.

**Decisions.**

- **Bound: p99 median** (user, from the sweep). It is one shared number that is not any single
  panel's statistic, and on the sweep it strictly beat the released p90max: more bite (1.626 vs 1.279
  percent of model intervals) with fewer false positives (4.157 vs 4.936 percent of real
  transitions). That contradicted the handoff's warning that it would fire less: gap 5 carries two
  of the four constrained intervals. p95 median was rejected at 12.4 percent false positives. A
  per-panel bound was ruled out because it carries disease identity into the trunk.
- **Arm set.** The user proposed p99max and p90max at lambda 10 and 100. I found p99max is
  identically zero on both US panels, so those 20 cells were guaranteed nulls. I cut p99max to COVID
  only and spent the time on the new bound. Final: six arms, 90 cells, about 6 h, dengue excluded.
- **Inertness gate at 1 percent of the objective** (user). A null under 1 percent is INCONCLUSIVE,
  a statement about lambda rather than the component.
- **Decision rule.** Paired 95 percent t-interval, replacing `|mean| >= sd` (about p = 0.09 at five
  seeds). Multiplicity handled structurally: a win on both RMSE and MAE at one horizon, plus no
  significant harm anywhere.
- **New records go to `experiments/`, not `ablation/`** (user), so the released runs stay untouched.

**Results.**

- **No PASS in 18 arm-panel units:** 6 FAIL, 12 INCONCLUSIVE. Of 144 deciding cells, 8 are worse, 4
  better and 132 within noise, and the 4 better cells never pair up on both metrics.
- Where the term was a real part of the loss, 3.244 to 30.654 percent (Japan and COVID at p90max
  lambda 10 and 100, and at p99median lambda 100), it improved nothing on both error metrics.
- **influenza_japan h10 got significantly worse on RMSE and MAE in all four new Japan arms,** up to
  about 12 percent RMSE. I ruled out trainer drift: the seed 42 baseline reproduces exactly and still
  shows the damage, and the near-inert released p99max arm moves Japan h10 by only about 7.
- **Likely cause, a hypothesis:** the shared bound calls 8.3 to 10.0 percent of Japan's own real
  training transitions implausible, so the penalty fights real flu seasons. It does not explain why
  the damage lands at h10 rather than h3 or h5.
- **US panels:** they never reached 1 percent even at lambda 100 (highest 0.904 percent), so they
  are INCONCLUSIVE, as the sweep predicted. us-regions exceeds no defensible bound at all.

**Caveats, all stated in the result doc.**

- The runner's `--report` still prints the old `|mean| < sd` rule and has no completeness check. The
  verdicts come from the raw records and were re-derived by the independent verifier.
- The protocol does not say how the share is averaged over seeds. I used the mean. The minimum
  changes nothing; the maximum flips 2 units from INCONCLUSIVE to FAIL. No reading gives a PASS.
- p99 median lambda 1 on Japan is INCONCLUSIVE by the rule (0.282 percent share), yet it caused
  significant h10 harm, which contradicts the gate's premise for Japan.
- The four new arms share one baseline, so the Japan harm is not four independent confirmations. The
  dose pattern and the near-inert control carry the argument.
- The probe reads the final model while the train penalty averages over all epochs: the penalty
  acts early and then goes quiet.
- Four of the five Japan baselines are cleared by file date and code diff, not by retraining.
- Dengue and Ebola are out of scope. The component is absent from the Ebola path.

**Open from this thread.**

- Rewrite `Reports/Manuscript_v2.md:497`: it calls the ablation a null. Manuscript session.
- Figure F3 plots only the lambda 1 arms under the old rule. Update it, or scope its caption.
- Fix the runner's `--report` only if this test is ever rerun.
- The result doc's "7 lines" in the drift paragraph: 5 of them are comments. Tighten it the next
  time the doc is touched.

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

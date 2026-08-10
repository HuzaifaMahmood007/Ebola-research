# Decision log — Phase 3 Week 4

**Started:** 2026-07-29 · **Progress:** `Day15_Progress.md` · **Work order:** `Phase3_Week4_Work_Order.md`

One row per decision that would be expensive to re-litigate. Client decisions are marked. Earlier
phases: `decisions_day_13.md`, `Phase3_Week3_Day15_Baselines_Status.md`.

---

## D1 · Direction: transfer, with the folds fixed first — **CLIENT**
2026-07-28, `Review Doc.md`

5-seed LODO spend is **on hold** until fold structure is fixed. Three of four dev datasets are
influenza, so leave-one-out holds out a *population*, not a *disease*. Report two tables:
leave-one-**dataset**-out (current, keep) and leave-one-**disease**-out (3 flu as one fold vs dengue).

**Why:** the two strongest folds are the ones with the most flu leakage. A reviewer reads that the
same way we now do. Spending seeds on the wrong fold structure buys nothing.

---

## D2 · Meta-learning is required, not optional — **CLIENT**
2026-07-28, `Review Doc.md`

Freeze-then-adapt is transfer learning with a linear probe, not meta-learning. Brief has meta-learning
as REQUIRED under G2, and the encoder was chosen for cheap second-order grads. Build episodic
meta-training (MAML / Reptile / ProtoNet — our pick, we justify). Freeze-and-adapt becomes the ablation.

**Why:** lets us answer "does meta-learning beat a linear probe" with a number instead of an assertion.
Client: *"don't quietly absorb it"* — if it doesn't fit the schedule, say so this week.

---

## D3 · Reporting standards, effective immediately — **CLIENT**
2026-07-28, `Review Doc.md`

Dispersion on every number (mean ± sd minimum; bootstrap CIs over regions and time origins for
headlines) · never average across datasets · state the comparison reference **on** the table · cells
under the noise floor say "within noise", not a direction.

---

## D4 · Ebola audit — figures settled
2026-07-29, computed from `data/processed/ebola.npz`

Weekly, N=61, T=52 (2014-03-24 → 2015-03-28), 41% observed (1299/3172), mean 21.3 obs weeks/node.
Graph usable: 146 edges, 0 isolated, 21 cross-border.

- **Support** = calendar prefix ≤2014-05-24 = weeks 1–7 → **27 cells across 9 of 61 districts**
- **Adaptation pairs:** full 20-week window → **0 at every horizon**. With P7 left-pad:
  **h3 = 16 pairs / 8 nodes, h5 = 6 pairs / 2 nodes, h10 = 0, h15 = 0**
- **Query/eval:** 1151 / 1075 / 866 / 642 pairs at h3/h5/h10/h15 over 61/61/59/58 districts

**Two corrections owed to the client:** the 27-example design floor came from **Ebola's own support**,
not the smallest dev set; and h10/h15 **are** evaluable — they are zero-shot, not unmeasurable
(their ~17-origin arithmetic was per-node; pooling over 61 districts rescues it).

**Real problem to raise instead:** h5 adaptation is 6 examples on 2 districts. Practical adaptation
horizon is **h3 only**.

---

## D5 · Transfer-table reference and dispersion
2026-07-29 · client complaint root-caused, method chosen with client

`train/lodo.py::_report` compared LODO to **seed-42's own** single run; the docs quoted **5-seed
means**. Never readable against each other. Seed 42 is an unusually bad single-disease seed
(dengue RMSE h3 z=+1.31, us-regions RMSE h5 z=+1.35), so a seed-matched reference flattered LODO —
that is the "+3% printed for a 17% worsening". **The formula at `train/lodo.py:287` is correct.**

**Decided:**
- Report **both references**, 5-seed mean primary, both labelled on the table
- **Paired origin-bootstrap CI** (B=10000) is the verdict — the axis the client asked for, and it
  works at 1 LODO seed
- **Seed-CV at 1.96×**, per horizon, as a secondary screen
- Correct the Week-3 docs **in place** with a dated correction note
- Patch `_report` so future runs are correct by construction

**Why per-horizon CV:** the floor swings from 14.4% (dengue h3) to 0.4% (dengue h15). An averaged
floor mis-tags both ends.

**Known wrinkle:** the origin CI is **cell-pooled**; headline tables are **node-averaged**. They
coincide on dense influenza and diverge on dengue. Document, don't reconcile.

**Deferred:** telling the client that 5 seeds will likely confirm the thin story rather than rescue
it. Client chose to run the seeds first and report what happens.

---

## D6 · HeatGNN epoch budget stays at the paper's 1500
2026-07-29

Verified in the paper (*Epidemiology-informed GNN for Heterogeneity-aware Epidemic Forecasting*,
Implementation Details): *"the early stopping strategy with the patience of 200 epochs, the batch
size to 32, and the number of epochs to 1500."* Split there is 60/20/20; we force 50/20/30 for
pipeline comparability (pre-existing, documented).

**Why it matters:** an earlier 150-epoch repro run was the **deviation**, not the standard. Cutting
epochs would have meant diverging from the paper while claiming we matched it. Speed must come from
parallelism instead.

`torch.autograd.set_detect_anomaly(True)` (`train.py:109`) ships on and stays on.

---

## D7 · HeatGNN NaN — isolated nodes, fixed with self-loops on HeatGNN's staged graph only
2026-07-29 · confirmed by `export_heatgnn.py --nan-test`

japan and us-states each carry **2 degree-0 nodes** (Okinawa; Alaska/Hawaii); us-regions carries
none — and us-regions was the only one that trained. `getLaplaceMat`
(`baselines/HeatGNN-14DB/src/utils.py:70-87`) never folds in the identity; the block that would is
**commented out**, leaving `sum(adj)+1e-12`, so an isolated node is scaled by ~1e12 rather than
erroring. Forward survives, backward overflows → `MulBackward0 returned nan values`.

**Test:** original graph NaN at seeds 999 **and** 992; same data + self-loops trains clean.

**Scope: HeatGNN's staged adjacency only.** EpiGNN/Cola/MTGNN already ran and are untouched. Note
this brings HeatGNN *into* line with our own encoder, which adds I before normalising
(`models/encoder.py:59-60`, C3 guard). **Disclose in paper + failure log.**

### D7a · EpiGNN's islands — document, do not rerun
EpiGNN's `getLaplaceMat` is byte-identical and equally lacks the identity; it survives only because
it has no SIR loss to amplify it. Quantified: japan islands are 1.25–1.42× other-node RMSE (rank 8
and 23 of 47); us-states islands are **0.31–0.38×** (*easier*, rank 47 and 27 of 49). Dropping them
shifts node_mean −1.0..−1.8% (japan) / +2.6..+2.9% (us-states) — **opposite signs, so idiosyncrasy
not bias.**

**Limit of this evidence:** it shows islands are not anomalous *in EpiGNN's own errors*. It does not
show what EpiGNN would score *with* self-loops. Only a rerun answers that, and we chose not to.

---

## D8 · The CSV skip trap
2026-07-29

`baselines/HeatGNN-14DB/src/train.py:99-105` exits immediately when `HeatGNN_results_test.csv` holds
a row matching (model, dataset, window, horizon, seed) — prints "Experiment exists", no training.
Our runner decides work from `_preds/*.npz`, so a **row without an npz** (crash or kill after the CSV
write) blocks that run forever.

`export_heatgnn.py` purges seed 990–999 rows and keeps a `.bak`. Real-seed collisions are not
auto-purged — they stop and report.

**Cost of learning this:** one wasted probe. Reusing seed 999 across combos made the second combo
exit in 6 s with no error.

---

## D9 · HeatGNN runtime model and launch config
2026-07-29 · measured, `export_heatgnn.py --probe`

**Cost tracks training windows, not nodes.** Steady s/epoch: us-regions 17.40 (368 windows, 10
nodes), japan 8.73 (150 windows, 47 nodes), us-states 8.84 (156 windows, 49 nodes). japan has 4.7×
the nodes and runs **2× faster**.

**Not thread-bound:** ±10% across thread settings; single-thread is *faster* on us-regions
(15.78 vs 17.40). Parallel is therefore near-linear.

**Do not project per-run cost as `s/epoch × 1500`** — runs early-stop near ~440 epochs. Use the
measured us-regions ground truth (**2.13 h/run** over 8 completed runs) scaled by the per-epoch ratio.

**Launched:** `--workers 8 --threads 1`, order **japan → us-states → us-regions** (the two unrun
datasets first, so a full-length run proves the self-loop fix early; us-regions is slowest per run
and tails the queue). 51 runs, ~8.9 h projected.

### D9a · Seed count stays at 5
An earlier decision to cut HeatGNN to 3 seeds was **reversed** once the real cost was measured —
at 8-way parallel the full 5-seed matrix fits inside the 1–1.5 day window, so the dispersion rule
(D3) is not broken and no disclosure is needed.

---

## D10 · Reproduction failure log — scope — **CLIENT**
2026-07-28/29

Two sections: **A** repos we could not reproduce (MepoGNN, MSGNN, STOEP), **B** comparisons we could
not make like-with-like (dengue ⅓-subsample non-comparability, Cola/Heat influenza-only on CPU,
HeatGNN's self-loop deviation, EpiGNN's island handling).

Client also requires each usable baseline validated against its source paper's published numbers
before it enters the comparison table — can't get close, it doesn't go in.

**Not created yet.**

---

## D11 · Independent audit of steps 1 and 2 — both GREEN, three fixes deferred
2026-07-29 · adversarial audit against artifacts on disk, not against claims

**Step 1 GREEN.** Run plan re-derived without trusting `remaining()` (60 target, 9 present, 51
missing, exact match, no already-done cell in the plan). Launch order confirmed japan → us-states →
us-regions. Self-loops verified present on **staged** adjacencies and absent from the **source**
exports — `staged − source` nonzero **only** on the diagonal for all three, source mtime 07-27,
two days before the fix. Isolated-node counts confirmed: japan 2 (idx 10, 19), us-states 2 (idx 1, 9),
us-regions 0. No probe artifacts. No in-matrix CSV orphan can block a run.

**Step 2 GREEN.** All 16 RMSE cells re-derived from raw JSON **without importing `analysis.py`** —
every cell within **0.03 pp**. Every number in both corrected tables independently reproduced, zero
discrepancies. `--selfcheck` exit 0, `_report('dengue',42)` prints −15.4% / +2.9% as claimed.

**Three fixes DEFERRED by decision, not oversight** (detail in `Day15_Progress.md`): the `--dry-run`
side-effect, the missing `-u`, and an `INFLUENZA`/`ORDER` display mismatch. All three only bite on a
**relaunch**; the queue is mid-flight and will not be relaunched during the review. **Apply only on
explicit confirmation.**

**Two audit findings corrected by later measurement:**
- The audit measured 0.22 cores/worker and projected a badly blown ETA. That sample was taken while
  MTGNN was still co-resident. Re-measured at 23:40 with MTGNN finished: **0.57 cores/worker, 4.53
  of 6 physical**. Revised ETA ~14.5 h, inside the window. Queue not restarted.
- The audit called the full-length self-loop proof unobservable. Partly true — but a NaN exits
  `train.py` non-zero, so it would surface as a `[FAIL]`. 8 workers alive 1,899 s at ~124 epochs with
  zero deaths. Held well past the 5-epoch proof; still **not** a completed 1500-epoch run.

**Findings accepted and left open:**
- `results/lodo/*.json` were produced by a version of `train/lodo.py` that was never committed and no
  longer exists, and `results/` is gitignored. **The transfer table's evidentiary base has zero
  version provenance.** Committing `train/lodo.py`, `analysis.py`, `results_paths.py` is the fix —
  a user decision, not taken here.
- `run_fold` is unchanged **by assertion, not evidence** — no pre-edit artifact exists to diff
  against. Structural argument only: the whole changed surface is reachable solely from `_report`,
  which `run_fold` calls at `:245` **after** all six artifacts are written, so it cannot alter output.
- Node identities behind the isolated indices (Okinawa / Alaska / Hawaii) are **unverified** —
  `meta.json` node_ids are anonymised (`us-states_0`…). Counts match; the names in the docstring
  and in D7 are an inference.
- `train/loop.py` was modified between the single runs and the LODO runs, which would break
  comparability if scoring had changed. Checked: the diff is **purely write-path routing**
  (`RESULTS / fname` → `rpath(...)`), no metric math. Artifacts remain comparable.

---

## D12 · Cross-disease transfer is negative — the Week-3 headline was leakage plus a reference artifact
2026-07-30, `Reports/Encoder_Results_Consolidated.md`, `Results_Matrix.md`

Under the corrected leave-one-**disease**-out fold (D1): **12 of 16 RMSE cells significantly negative,
4 within noise, 0 positive**, identical tally on MAE. The three largest LODO gains (+28.0%, +24.0%,
+19.9%) were all influenza panels with influenza still in training.

Separately, holding the transfer number fixed and varying only the comparison reference reproduces
**all three** of the client's arithmetic complaints to the decimal: us-regions h3 +19.9% (seed-matched)
vs **+9.9%** (5-seed mean) against their 9.8%; japan h5 +4.5% vs **+7.4%** against their 7.4%. Dengue
h3 **flips sign**, +2.9% → **−15.4%**, because single-disease dengue seed 42 is an outlier (49.98 vs
~37–47). Corroborated by `analysis.py`, which asserts −15.4% through a separate code path.

**Why it matters:** the paper's central claim as previously framed is not supported by our own
corrected experiment. `transfer-mechanism-finding`'s escape clause (*"Option B is off the table unless
the confirmation collapses the effect"*) has therefore **triggered — Option B is live.**

**Standing consequence:** every delta table states its comparison reference on the table, and the
check is automated (`results_matrix.py`), not remembered.

---

## D13 · Meta-learning: ANIL, and the schedule answer is no for the full scope
2026-07-31, `Reports/MAML_Decision.md` (answers D2)

**Algorithm — ours to call, and called.** MAML restricted to the adapter (**ANIL**), exact
second-order. ProtoNet excluded on task grounds (nearest-centroid classifier vs continuous
multi-horizon regression). Reptile excluded because it carries **no support/query meta-objective**, so
it cannot produce the controlled comparison D2 asks for — *not* on structural grounds, which the first
draft overstated.

**Schedule — no for the full scope**, option C may fit; a measured timing probe settles it. The
blocker is dependency order: ANIL sits upstream of a single-shot Ebola evaluation, so it serialises
Weeks 5 and 6 rather than running beside them.

**Scope is NOT decided here — deliberately.** Options A–D go to the client and Nora without a
recommendation, matching `Ebola_Support_Set_Decision.md`. Taking that decision for them is what this
document exists not to do.

**Three corrections we are on record for** (an adversarial audit; two re-verified directly):
- *"What we built is not a linear probe"* is **false**. Trunk frozen ⇒ `Adapter(h)` is an exact affine
  map of frozen features, `max deviation = 0.0`. **The client's description was right**, and the draft
  had simultaneously asserted ANIL while denying the property ANIL requires.
- The adapter is **1,428 params, not 388**; 388 predates the five-quantile head.
- **Nine trunk runs exist, not ten** — dengue→flu has 4 seeds (no seed 42).

**Also on record:** two zero-shot figures were within-noise cells quoted directionally, breaching D3;
the deficit range is **−1.5% to −51.3%**, not −16% to −51%; "monotone in horizon" is false for dengue.

**The objection we raise against ourselves:** with two dev diseases, LDO leaves **one** meta-train
disease, so episodes vary population and origin — the axis that already works. ANIL would be trained
on the non-problem. The structural fix is a third disease, which is why COVID is option D. **COVID is
not on disk** (no raw, no processed) — a full Phase-2 job, not a config flag. If it is added, the fold
must stay leave-one-**disease**-out.

---

## D14 · No further trunk run without checkpointing and quantile archiving
2026-07-31, `train/loop.py`

Both applied. Neither depends on any pending decision, both are cheap now and unrecoverable later.

- **Trunk checkpoints.** `find *.pt` outside `baselines/` returned **ZERO files**. `_fit_trunk` held
  the best state in memory and never wrote it, so **every LDO result rests on a trunk that no longer
  exists** — and the Ebola protocol ("freeze the trunk, fit the adapter") had nothing to freeze.
- **Quantile predictions.** Every scoring path took `[:, :, MEDIAN_IDX]` and discarded four of five
  quantiles at source. WIS, CRPS, coverage, interval width and PIT are implemented and self-checked
  but unusable without them, so the calibrated-uncertainty claim had **no evidence behind it**. This
  was the largest outstanding gap against D3's metric set and it blocked G4 entirely.

Wired via an optional `quant_out` dict so no call-site arity changed. Verified: checkpoint restores
trunk 142,305 + adapter 1,428 params exactly; `results_paths` self-check now covers 19 families.

---

## D15 · Capacity probe runs with an in-domain CONTROL arm, not alone
2026-07-31, `capacity_probe.py`

The probe asks whether a richer adaptation surface recovers any of the cross-disease deficit. **Run
alone it cannot answer that question**, and adding the control is what makes it interpretable.

- **Arm 1, cross-disease.** Dengue trunk, capacity ladder fitted on influenza. The transfer setting.
- **Arm 2, in-domain control.** *Same frozen trunk*, same ladder, fitted on dengue — the trunk's own
  disease. Costs no second trunk run.

**Why the control is not optional:** if bigger surfaces help arm 1, there are two explanations and
only the control separates them — (a) the surface was a genuine *transfer* bottleneck, the
interesting result, or (b) the adapter was simply undersized all along and helps everywhere, which
says nothing about transfer and is not an argument for ANIL. Reporting (b) as (a) would be exactly
the class of error D12 records.

Ladder is a strict capacity chain: affine **1,428** (the control, the surface that produced the LDO
result) → mlp-64 **5,460** → film+mlp-64 **5,588** → mlp-256 **21,780**. Only the surface varies;
`_fit_shared_adapter` holds epochs, patience, lr, wd, sampler and validation objective fixed.

Verdict is derived **mechanically** in code (five branches, all self-checked) so the morning call is
not a matter of taste: REPRESENTATION-BOUND / GENERAL UNDER-SIZING / ADAPTER-BOUND AND
TRANSFER-SPECIFIC / PARTIAL / INCONCLUSIVE.

**Scope of a null result, stated in advance:** it bounds what a *read-out* can recover from this
frozen representation. It does not prove no trunk can transfer. That is the correct bound for the
ANIL question, because ANIL keeps the read-out small by construction. Head capacity only — mid-trunk
FiLM is not tested, as that needs an encoder change and this run deliberately varies nothing else.

The probe **refuses to start unless `DEVICE` is cuda** (`--allow-cpu` to override). Confirmed:
torch 2.6.0+cu124, RTX 3060.

---

## D16 · Ebola support length: 12 weeks primary, 20 weeks pre-registered secondary — **CLIENT**
2026-08-07, closes the open question in `Reports/Ebola_Support_Set_Decision.md` (which stated the
trade-off and deliberately did not recommend). Full pre-registration: `Ebola_Prereg.md`.

**Two arms, both frozen and hashed before either is scored.** L=19 is dropped, and recorded in the
manifest as considered-and-rejected rather than silently absent.

| | **primary L12** | **secondary L20** | dropped |
|---|---|---|---|
| cutoff (support = every observed cell on or before) | **2014-06-28** | **2014-08-23** | 2014-08-16 |
| support cells / districts | 59 / 18 | 113 / 36 | 79 / 22 |
| adaptation pairs h3 / h5 / h10 / h15 | 48 / 38 / 18 / **0** | 102 / 92 / 72 / 54 | |
| scaler (pooled log1p+z on support) | mu 1.4402, sd 1.2145 | mu 1.7913, sd 1.5286 | |
| npz content sha256 | `08d657dc…` | `e9b9ac0b…` | |

**Query counts are identical under both arms**, asserted by the freeze script. A full-window query
target sits at column 22 and support reaches at most column 20, so neither arm costs a forecast. The
arms differ only in labelled adaptation data, never in what is evaluated, which is what makes them
comparable.

**Two counts, and only one is the evaluation size.** *Scored* = **757 / 766 / 765 / 642** pairs at
h3/h5/h10/h15 over 57/57/59/58 districts: one common origin set t in [19, 36] for every horizon,
which is what `bundles.origins()` returns and what `score_predictions` actually scores. *Reachable*
= 1,151 / 1,075 / 866 / 642 over 61/61/59/58: per-horizon origins, which is what the audit note and
`Ebola_Support_Set_Decision.md` quote. **The audit note's figures are not the evaluation size** and
overstate h3/h5/h10 by 34 to 52 per cent; they agree at h15, whose reach is the binding constraint.
Found by reading the scoring path before running it (pre-registration amendment A1). Every other
dataset in the project was scored on a common origin set, and that is the right protocol: horizons
are comparable only if read at the same origins.

**The primary arm has zero adaptation data at h15.** That is arithmetic: support reaches column 12,
h15 needs a target at column 15. h15 is therefore labelled zero-shot in every table, and few-shot
h15 must come out bit-identical to zero-shot h15 — any difference is a bug in the adaptation path,
not a result. h10 has 18 pairs on 9 districts and is not a fitted adapter either.

**Why the labels are off by one against the column counts.** The sweep table indexes outbreak weeks
from the raw first week 2014-03-24, whose incidence cell is masked (week 0 of a cumulative series
carries no increment). "L" is the 0-based index of the last support column, not a count: L12 is 13
columns, L20 is 21. **The cutoff date is the operative definition**; the label exists only to match
the document the client decided from.

**`data/processed/ebola.npz` (cutoff 2014-05-24, L7, 27 cells / 9 districts) is superseded for
scoring.** It stays as the Phase-2 build artifact and is what `build_datasets.py` still produces;
nothing may be scored against it. Scoring targets are read from `configs/ebola_arms.json`.

**C8 guards hardened as part of this.** They matched on the exact string `"ebola"`, so `ebola_L12`
would have walked straight past `assert name != "ebola"` in `train/loop.py` and the equivalents in
`train/lodo.py`, `train/joint.py`, `bundles.py`. All four now match on the `ebola` name prefix.

Rebuild and re-verify: `python freeze_ebola_arms.py` / `--verify`. The build re-derives every count
above from the raw xlsx and refuses to write if any has moved.

---

## D17 · The trunk early stop was real, and it cost nothing — measured, not argued
2026-08-06, `train/lodo.py`, `results/reports/ldo3full_influenza_seed42.log`

**The defect is real.** All 15 LDO3 trunks died on the same branch: each burned *exactly* 12,000
steps after its best checkpoint, which is `patience=12` x `val_every=1000`. None exhausted its
budget. `CosineAnnealingLR(T_max=91000)` was therefore scaled to a length early stopping guarantees
is never reached, so the lr at every selected checkpoint sat between 1.000e-3 and 9.3e-4 — the
schedule was a no-op and no trunk in that run could be called converged.

**The measurement.** Held-out influenza, seed 42, `--trunk-patience 999 --prefix encoder_ldo3full`,
91/91 val checks, ~2.8 h. `best=` reached **0.1768 at step 7,000 and never improved through step
91,000**. The scored records are **bit-identical** to the truncated run on all three flu datasets
(sha256 over the sorted record values) — necessarily so, since the same seed gives the same
trajectory, so both runs selected the same checkpoint. The extra 72,000 steps contain nothing better.

Japan h15 stays -43.3%, us-regions h15 stays -34.5%. **The verdict does not move.** D12 now stands
on a 91-point curve rather than an assumption.

Fix shipped anyway, because the mismatch will bite a longer-training configuration: `trunk_patience`
threaded through `run_ldo3_fold` (it was silently taking `_fit_trunk`'s default), `--trunk-patience`
and `--prefix` on the CLI, an argparse guard that **refuses** to run with the stop disabled under the
default prefix (it would overwrite the artifacts `LDO3_Results.md` and `verify_ldo3_doc.py` read),
and an `encoder_ldo3full__` route so both runs sit side by side.

**Scope of the claim: one fold, one seed.** The other 14 runs are inferred. Two more folds is ~7 h.

---

## D18 · The single-disease ceiling early-stops too — 25 of 25
2026-08-08, `diagnostics/epoch_budget_audit.py`, `diagnostics/recover_single_epochs.py`

Every transfer number is a ratio against the single-disease ceiling, so D17 made the ceiling an open
question: if only the transfer arm stopped early we benchmarked a half-trained model against a
fully-trained one, and the deficit is partly our own bug.

Training logs survived for only 5 of the 25 runs (covid, inside `overnight.log`); the other 20 were
trained on Day 13 and that stdout is gone. File mtimes are useless — all 20 JSONs are stamped within
two seconds of each other on 30 July, from a batch rescore, not from training. So the 20 were re-run
to recover the counts, through a library call that writes nothing and fingerprints all 954 artifacts
under `results/` before and after to prove it.

**Result: 25 of 25 early-stopped, 18 to 77 epochs of 80.** Not one used its budget.

Two things follow, and only the first is a clean win:

- **The attack is removed.** Nobody can say we crippled only the transfer side. Same rule, both arms.
- **The bias direction runs against transfer, not for it.** D17 shows the trunk is at its own
  optimum. If the ceiling sits *below* its optimum, `ceiling_error` is inflated and the reported
  deficit **understates** the true gap. The "both handicapped, so the deficits cancel" reading points
  the wrong way.

**Not "equally handicapped", and not uniform.** The arms stop on different clocks (80 epochs /
patience 15 epochs vs 91,000 steps / patience 12 checks of 1,000). And the behaviour varies by
dataset — dengue ran 57-77 of 80, us-regions 18-27 — so the bias is not constant across cells. Both
belong in the methods section as stated, not as a blanket claim.

**Still open: reproducibility.** The recovery straddled the 7 August restructure — 24 runs measured
the pre-restructure trainer, and only `us-regions seed 52` measured the current one. That one came
out **exact** across all 28 cells; the other 24 show worst-cell deltas of 0.7-65.8% concentrated in
`peak_timing`/`peak_intensity`, discrete indices where a one-week shift reads as a large relative
change. The script reports only the worst cell across all seven metrics and persists nothing, so
rmse/mae cannot be separated after the fact. **Re-run under current code with per-metric reporting
before anyone quotes the ceiling as reproducible.** ~2.5 h.

---

## D19 · Adapter capacity at five seeds: the gain is real, transfer-specific, and lands at h15
2026-08-08, `diagnostics/capacity_probe.py`, `Reports/Capacity_Probe_5Seed.md`

D15's ladder, promoted from one seed to five. One frozen dengue trunk **per seed**, shared by every
surface in both arms, so only the adaptation surface varies inside a seed and the trunk varies across
them. Every figure is a **seed-paired** delta against that seed's own affine control; intervals are
two-sided 95% t at n=5 (t=2.776), and a cell counts only if its interval excludes zero. 774 min.

| arm | cells | significantly better | significantly worse | best significant |
|---|---|---|---|---|
| cross-disease | 36 | **7** | 3 | **+19.8%** |
| in-domain control | 12 | **0** | 1 | — |

**The shape is the finding, not the headline number.** All 7 gains are at **h15** (us-regions +19.8,
+14.3, +12.2; japan +6.6, +6.4, +6.3; us-states +4.5). All 3 losses are at **japan h3** (-16.6 to
-22.5). The control gains nothing anywhere. So a richer adaptation surface buys accuracy at long
horizon and *costs* it at short horizon, and the effect is genuinely transfer-specific — D15's
alternative (b), general under-sizing, is ruled out by 0 of 12.

**The caveat that must travel with it.** h15 is exactly where the frozen Ebola primary arm has
**zero adaptation pairs** (D16: 48/38/18/0 at h3/h5/h10/h15). The one horizon this fix helps is the
one horizon Ebola cannot fit an adapter for. Stated in the same breath or the result oversells.

**It cannot change the Ebola arm on its own.** The support arms were frozen and hashed on 2026-08-07
and `Ebola_Prereg.md` defines few-shot as the FiLM-plus-head adapter — the affine control here.
Scoring Ebola against a surface chosen after that freeze would void the pre-registration, which is
itself a stated contribution. This is a development-fold mechanism result and a case to put to the
client for re-registration, not a config change.

One defect found and fixed en route: `stage1_fold` called `run_ldo_fold`, which also **writes**
`encoder_ldo__influenza_*__seed{S}.json`. Sweeping five seeds through it would have silently
overwritten four fifths of the two-disease LDO table (D1) with numbers from different code. It is now
trunk-only, under a distinct `dengue2flu-cap` checkpoint family.

---

## Reversed or superseded

| was | now | why |
|---|---|---|
| Cut HeatGNN to 3 seeds | 5 seeds kept (D9a) | parallel measurement made it affordable |
| Cut HeatGNN epochs to 150 | 1500, the paper's value (D6) | 150 was the deviation, not the standard |
| NaN caused by dead/constant nodes | isolated nodes (D7) | checked — no constant nodes exist |
| "5-seed LODO is the one gate" (`Phase3_Week3_Results_and_Direction.md` §6) | on hold pending fold fix (D1) | client countermanded |
| Freeze-then-adapt transfer is POSITIVE (Week-3 LODO, 1 seed) | **NEGATIVE** under the disease-level fold (D12) | fold leakage + a seed-matched reference against an outlier seed |
| "What we built is not a linear probe" (first MAML draft) | it **is** an exact affine read-out on frozen features (D13) | measured, `max deviation = 0.0`; the client was right |
| Adapter is ~388 params (4 code/doc sites) | **1,428** (D13) | 388 predates the five-quantile head |
| Reptile excluded "structurally" (first MAML draft) | excluded for having no support/query meta-objective (D13) | the structural argument described a variant nobody proposes |
| Ebola support = calendar prefix ≤2014-05-24, 27 cells / 9 districts (D4) | **≤2014-06-28 primary, ≤2014-08-23 secondary** (D16) | client decision; 27 cells gave 0 adaptation pairs past h5 |
| "Training halted after 1-16% of its budget" (Week-4 stakeholder brief) | halted at **14.3-29.7%**; 1.1-16.5% is where the *kept checkpoint* was selected (D17) | measured from `overnight.log`; substance right, wording wrong, and it went to the client that way |
| Capacity gain "up to +23.2%, median +2.5%" (1 seed, D15) | **7 of 36 cells clear zero, best +19.8%**, all at h15; control 0 of 12 (D19) | five seeds with seed-paired intervals |

# Day 15 Progress — Phase 3 Week 4

**Updated:** 2026-07-29 23:45 · **Work order:** `Phase3_Week4_Work_Order.md` · **Decisions:** `decisions.md`

Live state of the Week-4 work order. Numbers here are checked against disk, not remembered.

---

## Advisor review — independent audit, 2026-07-29

An adversarial verifier audited steps 1 and 2 against **artifacts on disk**, explicitly not against
our claims. Full decision record in `decisions.md` D11.

### Verdict: **STEP 1 GREEN · STEP 2 GREEN**

### What it proved independently

| check | result |
|---|---|
| Run plan re-derived without trusting `remaining()` | 60 target, 9 present, **51 missing, exact match**; no already-done cell appears in the plan |
| Launch order | japan → us-states → us-regions, confirmed in real dry-run output |
| Self-loops scoped correctly | `staged − source` nonzero **only on the diagonal**, all three datasets. Source exports untouched (mtime 07-27, two days pre-fix) |
| Isolated-node counts | japan 2 (idx 10, 19) · us-states 2 (idx 1, 9) · us-regions 0 |
| Probe artifacts | 0 `seed99*` npz, 0 seed-990-999 CSV rows |
| CSV skip trap | all 10 rows cross-checked; the one orphan is the upstream repo's own demo row, outside our 60-cell matrix. **Nothing will be silently skipped** |
| Transfer deltas | **all 16 RMSE cells re-derived from raw JSON without importing `analysis.py`** — every cell within **0.03 pp** |
| Corrected tables | every number in both reproduced. **Zero discrepancies** |
| `--selfcheck` / `_report` | exit 0; dengue h3 prints −15.4% / +2.9% as claimed |
| MTGNN | reached **80/80** during the audit |

### Defects it found — three, all deferred by decision

Listed under *Pending code fixes* below. All only bite on a **relaunch**; the queue is mid-flight.

### Two of its conclusions we corrected with fresh measurement

- It measured **0.22 cores/worker** and projected a badly blown ETA. That sample was taken while
  MTGNN was still co-resident. Re-measured at 23:40 with MTGNN finished: **0.57 cores/worker, 4.53
  of 6 physical.** Revised ETA ~14.5 h, inside the window. Queue **not** restarted.
- It called the full-length self-loop proof unobservable. Partly true, but a NaN exits `train.py`
  non-zero, so a failure surfaces as `[FAIL]` regardless of log buffering. 8 workers alive 1,899 s
  at ~124 epochs, **zero deaths**.

### Findings accepted and still open — read these before building on step 2

1. **The self-loop fix is not proven at full length.** ~124 epochs is not 1500. Do **not** write it
   up as proven until a japan run completes.
2. **The transfer table has zero version provenance.** `results/lodo/*.json` were produced by a
   version of `train/lodo.py` that was **never committed and no longer exists**, and `results/` is
   gitignored. Fix: commit `train/lodo.py`, `analysis.py`, `results_paths.py`. **Step 3 builds
   directly on these artifacts** — worth clearing first.
3. **`run_fold` is unchanged by assertion, not evidence.** No pre-edit artifact exists to diff.
   Structural argument only: the entire changed surface is reachable solely from `_report`, which
   `run_fold` calls at `:245` *after* all six artifacts are written, so it cannot alter output.
4. **Isolated-node identities are an inference, not verified.** `meta.json` node_ids are anonymised
   (`us-states_0`…), so Okinawa / Alaska / Hawaii is our reading of the counts, not a checked fact.
5. **Cleared, worth knowing:** `train/loop.py` was modified *between* the single runs and the LODO
   runs — which would break comparability if scoring had changed. It hadn't: the diff is purely
   write-path routing (`RESULTS / fname` → `rpath(...)`), no metric math. Artifacts are comparable.

### Signal

**GREEN to proceed to step 3** (fold restructure), with open item 2 above worth clearing first.

---

## Running right now

| job | state | ETA |
|---|---|---|
| **HeatGNN** `export_heatgnn.py --run --workers 8 --threads 1` | 8 workers live, 9/60 done, 51 queued, japan first | **~14.5 h → ~13:40 30 Jul** |
| **MTGNN** | **80/80 COMPLETE** | done 29 Jul |

**ETA revised up from 8.9 h** (measured 23:40, MTGNN finished): each worker gets **0.57 cores**,
total **4.53 of 6 physical** — 75% utilisation, a 1.75× per-worker slowdown from contention.
Effective ~15.3 s/epoch japan and us-states, ~30.5 us-regions. At ~440 epochs/run that is 1.87 h /
1.89 h / 3.73 h → 116 worker-hours ÷ 8. Fewer workers would land at roughly the same total, so the
queue is **not** worth restarting.

**Self-loop fix at full length — partially answered.** All 8 workers alive 1,899 s with 1,078 s CPU
each ≈ **124 epochs, zero deaths**. A NaN exits `train.py` non-zero, so a failure would surface as a
fast `[FAIL]` plus a replacement worker. Held far past the 5-epoch proof. **Not yet a full
1500-epoch run — do not write it up as proven.**

**Still to check when the first japan run lands (~01:00):** does it early-stop near us-regions'
~440 epochs? That assumption drives the whole ETA.

---

## Baseline reproduction vs published vs our encoder — three-way, 2026-07-30

Client condition §1c answered. Generated by `paper_compare.py`; published figures transcribed in
`Published Model benchmarks/`. Raw score dump in `Updated_Scores.txt`.

**All three columns are CELL-POOLED RMSE**, the definition every one of these papers uses
(`sqrt(mean over all scored cells)`). This is *not* our country-macro headline — `score.py` averages
per-node RMSEs, which by Jensen is always the smaller number. A Δ% between the two aggregations would
measure the aggregation, not the reproduction. Read this table against itself only; the country-macro
numbers live in `Updated_Scores.txt` and must never be mixed in.

The encoder never archived predictions, so its column is reconstructed from the per-origin error
sufficient statistics: pooled RMSE = `sqrt(sum(sse)/sum(n))` (`train/loop.py:252-256`). Verified per
run — the cell count implied by the sufficient stats must equal the count in the matching
`__pernode.npz`. **48/48 runs passed, 0 skipped.**

### 🔴 MTGNN IS NOT A USABLE BASELINE — it emits a single constant

Found while building this table. **47 of 80 MTGNN prediction files contain exactly one distinct
value** — one number repeated for every node at every origin.

| MTGNN | constant runs | evidence |
|---|---|---|
| dengue h5 / h10 / h15 | **5/5 seeds each** | every cell = 1.124; seed 42 and seed 82 predictions **byte-identical** |
| influenza_japan, all 4 h | **4/5 seeds each** | every cell = 564.797; seed 42 and seed 62 **byte-identical** |
| influenza_us-states, all 4 h | **3/5 seeds each** | — |
| influenza_us-regions, all 4 h | 1/5 seeds each | non-constant runs still only 10–21 distinct values across 2,360 cells |
| dengue h3 | 0/5 | the only MTGNN cell that looks like a trained model |

Corroborating: pooled PCC is **negative** on every influenza dataset (japan h5 −0.566, us-states h3
−0.463), and seed sd is exactly 0.00 wherever the runs are constant. A model that ignores its seed
and predicts one number is not converging — it has collapsed to something like a global mean.

**Consequences.** `Day15_Progress.md` §1 recorded MTGNN as "80/80 COMPLETE" and `PROJECT.md` §7 has
it as a reproduced pass and the non-epidemic floor. **Both are wrong as they stand.** MTGNN cannot
enter the comparison table, and every "we beat MTGNN" figure is meaningless — we are beating a
constant. It needs either a re-run with the collapse diagnosed, or a Section-A entry in
`reproduction_failure_log.md`. Note this is a failure of *our* run, not of the published model:
MTGNN's own paper reports on traffic/solar/electricity, and third-party epidemic re-runs (MepoGNN,
STOEP) both got sensible numbers from it.

### The table

| model | dataset | h | 1. published | 2. reproduced | Δ% vs published | 3. our encoder | encoder vs reproduced |
|---|---|---|---|---|---|---|---|
| Cola-GNN | japan | 3 | 1051 | 1108.3 ± 82.2 | +5.5% | 1034.3 ± 69.1 | −6.7% |
| Cola-GNN | japan | 5 | 1117 | 1197.2 ± 93.6 | +7.2% | 1168.8 ± 56.9 | −2.4% |
| Cola-GNN | japan | 10 | 1372 | 1504.7 ± 65.0 | +9.7% | 1469.9 ± 108.1 | −2.3% |
| Cola-GNN | japan | 15 | 1475 | 1500.5 ± 50.5 | +1.7% | 1427.3 ± 171.1 | −4.9% |
| Cola-GNN | us-regions | 3 | 636 | 710.1 ± 33.2 | +11.6% | 766.3 ± 114.3 | +7.9% |
| Cola-GNN | us-regions | 5 | 855 | 927.1 ± 36.3 | +8.4% | 906.5 ± 98.8 | −2.2% |
| Cola-GNN | us-regions | 10 | 1134 | 1168.2 ± 72.4 | +3.0% | 973.7 ± 55.4 | −16.6% |
| Cola-GNN | us-regions | 15 | 1203 | 1239.2 ± 135.3 | +3.0% | 995.3 ± 93.9 | −19.7% |
| Cola-GNN | us-states | 3 | 167 | 187.7 ± 7.0 | +12.4% | 187.3 ± 8.1 | −0.2% |
| Cola-GNN | us-states | 5 | 202 | 232.4 ± 8.5 | +15.0% | 221.7 ± 9.2 | −4.6% |
| Cola-GNN | us-states | 10 | 241 | 264.1 ± 15.3 | +9.6% | 240.4 ± 5.1 | −9.0% |
| Cola-GNN | us-states | 15 | 237 | 249.3 ± 18.3 | +5.2% | 244.1 ± 8.0 | −2.1% |
| EpiGNN | japan | 3 | 996 | 1182.3 ± 65.7 | **+18.7%** | 1034.3 ± 69.1 | −12.5% |
| EpiGNN | japan | 5 | 1031 | 1273.1 ± 322.1 | **+23.5%** | 1168.8 ± 56.9 | −8.2% |
| EpiGNN | japan | 10 | 1441 | 1701.2 ± 98.9 | **+18.1%** | 1469.9 ± 108.1 | −13.6% |
| EpiGNN | japan | 15 | 1470 | 1598.0 ± 201.7 | +8.7% | 1427.3 ± 171.1 | −10.7% |
| EpiGNN | us-regions | 3 | 589 | 632.4 ± 36.1 | +7.4% | 766.3 ± 114.3 | +21.2% |
| EpiGNN | us-regions | 5 | 774 | 893.8 ± 55.4 | **+15.5%** | 906.5 ± 98.8 | +1.4% |
| EpiGNN | us-regions | 10 | 984 | 1053.3 ± 47.4 | +7.0% | 973.7 ± 55.4 | −7.6% |
| EpiGNN | us-regions | 15 | 1061 | 1107.4 ± 96.7 | +4.4% | 995.3 ± 93.9 | −10.1% |
| EpiGNN | us-states | 3 | 160 | 167.8 ± 7.2 | +4.9% | 187.3 ± 8.1 | +11.6% |
| EpiGNN | us-states | 5 | 186 | 195.9 ± 11.9 | +5.3% | 221.7 ± 9.2 | +13.1% |
| EpiGNN | us-states | 10 | 220 | 225.8 ± 8.2 | +2.6% | 240.4 ± 5.1 | +6.4% |
| EpiGNN | us-states | 15 | 236 | 237.4 ± 7.2 | **+0.6%** | 244.1 ± 8.0 | +2.8% |
| HeatGNN | us-regions | 5 | 852 | 921.3 ± 71.4 | +8.1% | 906.5 ± 98.8 | −1.6% |
| HeatGNN | us-regions | 3 | — | 672.5 ± 36.5 | — (h3 off paper grid) | 766.3 ± 114.3 | +14.0% |
| EpiGNN | dengue | 3–15 | — | 435.9 → 504.7 | — | 202.6 → 315.6 | −37% to −54% ⚠ |
| MTGNN | all | all | — | see above | — | — | **EXCLUDED, degenerate** |

Δ% positive = our reproduction is worse than published. `encoder vs reproduced` negative = encoder better.

### Reads

1. **Every single reproduction is worse than published, none better.** Cola-GNN mean +7.7% (range
   +1.7 to +15.0), EpiGNN mean +9.7% (+0.6 to +23.5), HeatGNN +8.1% on its one overlapping cell. A
   systematic one-directional gap points at the two known protocol deltas — we run their model on
   *our harmonised bundle*, not their shipped data copy, and we report a 5-seed mean against what is
   often a single published value.
2. **This lands inside the tolerance the literature itself shows.** HeatGNN's paper re-ran Cola-GNN
   and missed Cola-GNN's own published Japan h2 by **+25.7%** (`Published Model benchmarks/README.md`
   §3). Our worst cell is +23.5%. Against that band, Cola-GNN and EpiGNN both **pass** §1c.
3. **EpiGNN's error is concentrated, not uniform.** us-states is near-exact (+0.6 to +5.3%), japan is
   the bad fold (+8.7 to +23.5%) with a huge ±322 sd at h5. Japan is also the dataset with the
   known seasonal-naive problem. Worth one look at whether the two share a cause.
4. **The encoder beats reproduced EpiGNN on japan at every horizon** (−8.2 to −13.6%) and **loses on
   us-states at every horizon** (+2.8 to +13.1%). Against Cola-GNN it wins 10 of 12 cells. This is
   the first honest head-to-head we have, and it is genuinely mixed — not a clean win.
5. **⚠ The dengue row is NOT like-with-like and must not be quoted as a win.** The encoder scores on
   full dengue (7,165 nodes); the baselines score on the 1/3 stratified subsample (2,392 nodes,
   different node set entirely). The −37% to −54% gap is largely that difference. Already a disclosed
   caveat (D10 Section B) — restating it here because the table makes it look like a headline result.

### Follow-ups this creates

- [ ] **Diagnose MTGNN's collapse** or log it as a Section-A failure. Blocks §1's sign-off either way.
- [ ] Correct `PROJECT.md` §7 (MTGNN "✅ Pass (control)") and §1 above ("80/80 done").
- [ ] `reproduction_failure_log.md` still not created — now has a fourth entry to carry.
- [ ] HeatGNN's remaining paper-grid cells (h2/h7/h12, japan + us-states) are not runnable from this
      queue: the queue runs *our* horizons {3,5,10,15}. The paper-protocol rerun is a separate job.
- [ ] All three of EpiGNN/Cola-GNN/HeatGNN emit **negative counts** on most runs (min −2,204 for
      EpiGNN japan). The papers do not clip either, so it is comparable — but it needs a sentence in
      the write-up before a reviewer asks.

---

## Pending code fixes — DEFERRED, do not apply mid-run

Found by the audit (`decisions.md` D11). All three only take effect on a **relaunch**, and we are not
relaunching during the review. **Confirm before applying.**

| # | file | fix | why it matters |
|---|---|---|---|
| 1 | `export_heatgnn.py` `run_all` | move the `if dry_run:` gate **above** `_stage_all()` | `--dry-run` is not side-effect-free: it rewrites the staged adjacencies, and `stage()` restores the RAW graph for a sub-millisecond window before `_add_self_loops` puts the identity back. A worker spawned inside that window reads the exact graph that NaNs. Verified idempotent (MD5 identical before/after), so nothing is currently damaged. |
| 2 | `export_heatgnn.py` `_cmd()` | add `-u` after `python` | logs are block-buffered at 5,614 bytes with **zero epoch lines**, so intra-run progress and any NaN warning are invisible until a run ends. Outcome visibility is unaffected (`[ok]`/`[FAIL]` comes from npz existence). |
| 3 | `export_heatgnn.py` `run_all` | summary loop iterates `INFLUENZA`, should be `ORDER` | cosmetic only — counts are right, display order is not. |

Also flagged, not code:
- `baselines/HeatGNN-14DB/src/HeatGNN_results_test.csv.bak` still holds the 6 probe rows (seeds
  990–995). Inert — `train.py` reads only the `.csv` — but one `mv` from resurrecting the skip trap.
- Both Week-3 docs cite `train/lodo.py:287` as the percentage formula; post-edit it lives at **:328**.

---

## §1 Baselines — the Week-3 gate

| model | preds | target | state |
|---|---|---|---|
| EpiGNN | 80 | 80 | done (dengue + 3 flu) · reproduces +0.6…+23.5% vs paper |
| ColaGNN | 60 | 60 | done (3 flu) · reproduces +1.7…+15.0% vs paper |
| MTGNN | 80 | 80 | files complete but **🔴 47/80 ARE A SINGLE CONSTANT — not usable**, see three-way section |
| HeatGNN | 9 | 60 | **running**, 51 left |

HeatGNN by dataset: us-regions 9/20, japan 0/20, us-states 0/20.

**Solved this session** (detail in `decisions.md` D6–D9):
- HeatGNN NaN on japan/us-states — isolated nodes, fixed with self-loops. Confirmed at 2 seeds.
- The CSV skip trap — stale `HeatGNN_results_test.csv` rows silently abort runs.
- Runtime model — cost tracks **training windows, not nodes**. japan (47n) is half the per-epoch
  cost of us-regions (10n).
- Not thread-bound: ±10% across thread settings, single-thread *faster* on us-regions.

**Still open in §1:**
- [x] Rescore everything — done 2026-07-30, 229/229 baseline files + 48/48 encoder artifacts. Added
      `nrmse` (scale-normalised RMSE, Review Doc ¶7) to `score.py`; encoder backfilled offline by
      `rescore_encoder.py` (no retrain). WIS/CRPS/coverage/interval-width/PIT implemented and
      self-checked, but they need quantile predictions and no run archives those yet.
- [x] Reproduction-vs-published table — done, see the three-way section above.
- [ ] Reproduction-vs-published table (Cola/Heat need reruns at **paper** horizons on **paper** data,
      60/20/20 split — separate from the pipeline runs)
- [ ] `reproduction_failure_log.md` — not created yet. Scope agreed: every blocker incl. data-side
      caveats, two sections.

---

## §2 Transfer table correction

**Done:**
- [x] Root-caused both client complaints. Reference mismatch confirmed; dengue "sign inversion" is a
      bad denominator (seed 42 is an outlier single-disease seed), not a formula bug.
- [x] Noise floor computed per horizon from the 5 single-disease seeds.
- [x] `analysis.py` — new `transfer_ci()` / `transfer_table()`, `--transfer` flag. Both references
      labelled, paired origin-bootstrap CI (B=10000) as the verdict, seed-CV screen at 1.96x as
      secondary. Self-check added that asserts dengue h3 reads −15.4%, not +3%.

- [x] `train/lodo.py::_report` — both references, per-horizon CV, `clears`/`NOISE` tag, and a pointer
      to `analysis.py --transfer` for the verdict CI. Verified: dengue h3 now prints
      `d% vs A = −15.4%` beside `d% vs B = +2.9%` — the client's complaint rendered on the table.
- [x] `Phase3_Week3_Results_and_Direction.md` — corrected table in §4b, dated correction block,
      superseded-numbers list, TL;DR marked superseded, §6 item 2 struck as countermanded.
- [x] `results_summary.txt` §3b — replaced with the corrected table and the same correction note.
- [x] Two-axis disagreement documented in both docs (cell-pooled CI vs node-averaged headline).

**Still open:**
- [ ] §2 is otherwise **done**. It re-opens only when the 5-seed run lands (§3), at which point the
      statistic should switch from origin-bootstrap-at-1-seed to a **paired-by-seed delta** — both
      arms share seeds, so pairing cancels shared init noise. Treating them as independent instead
      would raise the bar by ~√2 for no reason.

### What survives (RMSE, vs 5-seed mean, origin CI = verdict)

| fold | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| **japan** | **+21.1%** better | within noise | **−20.0%** worse | **−27.2%** worse |
| **us-regions** | +13.1% better¹ | +20.0% better¹ | +9.2% better¹ | +2.5% better¹ |
| **us-states** | within noise | within noise | −1.8% worse¹ | −3.0% worse¹ |
| **dengue** | **−20.9%** worse¹ | −10.4% worse¹ | **−4.4%** worse | **−1.3%** worse |

¹ origin CI excludes zero but the delta is under 1.96× the seed CV — stable across test periods,
not distinguishable from a different random init.

**Only cell that clears both screens as a positive: japan h3.** Dengue — the one genuinely
held-out *disease* — is negative at every horizon. LODO is 1 seed; everything is provisional.

---

## §3 Fold restructure — not started

- [ ] Relabel the existing dengue fold as the flu→dengue leave-one-**disease**-out result (it already
      is one; `train/lodo.py:224` computes `in_names` as exactly the 3 flu sets). No rerun needed.
- [ ] Build the dengue→flu direction: one shared adapter across the 3 flu bundles.
      `_fit_adapter_and_score` (`train/lodo.py:135`) takes one bundle; needs a multi-bundle variant.
- [ ] 5-seed LODO {52,62,72,82} on **both** fold structures — blocked until the above lands.

Architecture already verified as capable: `SharedEncoder` and `Adapter` have no node dimension;
`_prepare` and `block_diag_sparse` take arbitrary name lists.

---

## §4 Meta-learning — **client wants an answer this week**

- [ ] Pick MAML / Reptile / ProtoNet + one paragraph of justification
- [ ] Straight yes/no on schedule fit

Not started. Client said explicitly: *"don't quietly absorb it."* This is the item most at risk of
slipping silently.

---

## §5 Ebola audit — numbers done, write-up owed

- [x] All figures computed (see `decisions.md` D4)
- [ ] Send to client, including the two corrections: the 27-example floor came from **Ebola itself**,
      not the smallest dev set; and h10/h15 **are** evaluable (642 pairs / 58 districts), just
      zero-shot. Real problem to raise instead: **h5 adaptation is 6 examples on 2 districts.**

---

## §6–§8 Not started

Joint-training 2-day timebox · WIS/CRPS/coverage/PIT/peak metrics · ARIMA+GBM floors · conformal
split reservation and variant choice · explainability scoping · gate figure · seed ensembling ·
delta-prediction test · pre-registration.

---

## Housekeeping

Uncommitted: `analysis.py`, `decisions_day_13.md`, `train/loop.py`, `score.py` (new metrics).
Untracked: `ablation/`, `train/lodo.py`, `results_paths.py`, `export_*.py`, `score_baseline.py`,
`run_baselines.py`, `make_results_doc.py`, `results_summary.txt`, `test_run.py`, `Review Doc.md`,
both Week-3 docs, `Phase3_Week4_Work_Order.md`, this file, `decisions.md`,
`Published Model benchmarks/`, `paper_compare.py`, `rescore_encoder.py`, `Updated_Scores.txt`.

`Phase3_Week3_Results_and_Direction.md` §6 still reads *"[next — the one gate] 5-seed LODO
confirmation"*. **Countermanded** by the client. Mark superseded.

# Day 15 Progress — Phase 3 Week 4

**Updated:** 2026-07-30 22:35 · **Work order:** `Phase3_Week4_Work_Order.md` · **Decisions:** `decisions.md`

Live state of the Week-4 work order. Numbers here are checked against disk, not remembered.

---

## 🟢 §4 META-LEARNING ANSWERED + two archiving blockers closed (2026-07-31)

`Reports/MAML_Decision.md` written. **Deliverable 1 answered** (ANIL — MAML restricted to the adapter,
exact second-order). **Deliverable 2 answered** (No for full scope; option C may fit; measured probe
settles it). Scope options A–D are put to the client and Nora **without a recommendation** — the doc
deliberately takes no decision on their behalf, matching `Ebola_Support_Set_Decision.md`.

### The first draft was audited and three load-bearing claims were wrong

An adversarial advisor checked the draft against code on disk. Verdict: **materially flawed**. Two of
the three worst findings I re-verified myself before accepting:

| claim | reality |
|---|---|
| "what we built is **not** a linear probe" (a correction aimed at the client) | **False.** Trunk frozen ⇒ `Adapter(h) = (W·diag(γ))h + (W·β+b)`, an exact affine map. Measured `max deviation = 0.0`. **The client was right.** Worse, the draft asserted ANIL while denying the property ANIL requires |
| "388-parameter adapter" (used 3×, incl. the HVP-is-cheap argument) | **1,428.** 64+64+1280+20. 388 is the `\|Q\|=1` figure from before the 5-quantile head |
| "two directions × five seeds = ten runs", "no rerun at all" | **Nine runs on disk.** dengue→flu has 4 seeds (no 42). Cannot seed-pair a 5-seed ANIL arm against it |

Also corrected: two zero-shot headline numbers were **within-noise** cells quoted directionally
(breaking our own reporting rule); deficit range was cherry-picked (−16%…−51% → actually
**−1.5% to −51.3%**); "monotone in horizon" is **false for dengue** (−9.8→−1.5, improves with
horizon); the compute-matching rule used an ANIL-wrong cost model; Reptile's exclusion was a
strawman; wall-clock was declared unmeasurable but was **on disk**.

**Measured from untouched `gate.npz` mtimes: 41–59 min/run, median ~55, nine runs 02:44→09:58 on the
3060.** So a ten-run sweep is ~9–10 GPU-h. That number is now in the doc.

### Three things the draft never mentioned, all added

1. **The two-disease problem.** LDO with 2 diseases leaves **one** meta-train disease, so episodes can
   only vary population and origin — the axis that already *works* (+24.0% LODO). ANIL would train on
   the non-problem and be tested on the axis it never saw. Strongest reviewer objection there is.
2. **Ebola h10/h15 have zero adaptation pairs**, so ANIL's inner loop cannot run there — yet h15 is
   where the draft drew its motivating numbers. Case must be argued at h3/h5.
3. **COVID as option D**, with the fold-structure consequence: if COVID is added the fold must stay
   **leave-one-DISEASE-out**. Flagged that **COVID is not on disk at all** (no raw, no processed) —
   it is a full Phase-2 job, not a config flag.

### Code fixes applied

| fix | detail |
|---|---|
| `388` → `1,428` | `models/adapters.py`, `train/lodo.py`, `train/joint.py`, `Phase3_Week4_Work_Order.md` |
| **Trunk checkpointing** | `train.loop.write_checkpoint`. Before this, `find *.pt` outside `baselines/` returned **ZERO** — every LDO result rested on a trunk that no longer exists, and the Ebola protocol had no trunk to freeze |
| **Quantile archiving** | `train.loop.write_quantiles`. Every scoring path took `[:, :, MEDIAN_IDX]` and dropped 4 of 5 quantiles at source. Wired into `_score` (lodo) and `_test_dataset` (joint) via an optional `quant_out` dict, so no call-site arity changed. **Unblocks G4** |
| `adapter_factory` param | `_fit_shared_adapter`, defaulting to `Adapter` — bit-identical for existing callers, lets the capacity probe vary only the surface |

Verified: `results_paths` self-check now **19 families** (added `__quantiles.npz` / `__ckpt.pt`
routes); round-trip test confirms quantiles store `[N,K,Q]` + origins + levels, and a checkpoint
restores **trunk 142,305 + adapter 1,428 params exactly** (divergence 158.5 → 0.0).

### Overnight run ready — `capacity_probe.py`

**The highest-value run available, because it can invalidate the ANIL proposal before we spend a week
on it.** Premise never tested: does a *richer* adaptation surface recover any of the deficit?

- **Stage 1** runs LDO `dengue2flu` seed 42 — which simultaneously **closes the 4→5 seed gap**,
  produces the project's **first trunk checkpoint**, and the **first archived quantiles**. ~1 h.
- **Stage 2** fits four surfaces on that *same frozen trunk* — affine (control, 1,428) → mlp-64
  (5,460) → film+mlp-64 (5,588) → mlp-256 (21,780) — holding the fitting protocol fixed.

**The morning decision.** If no larger surface beats the affine control, the trunk does not carry
recoverable cross-disease signal and ANIL — which adds no read-out capacity — is unlikely to close a
51% gap. That is a decisive negative for one night. If a larger surface *does* help, there is a cheap
partial fix and a real prior for ANIL. The script writes `Reports/Capacity_Probe_Result.md` with a
mechanically-derived verdict so the call is not a matter of taste.

**Second arm added (D15) — the control that makes arm 1 readable.** Arm 2 runs the identical ladder
on the *same frozen trunk* but fitted on **dengue**, the trunk's own disease. No second trunk run.
Without it, "bigger adapter helps" is ambiguous between a real transfer fix and the adapter simply
having been undersized all along — and only the second story would be worthless for the ANIL case.

Verdict derived **mechanically**, five branches, all self-checked: REPRESENTATION-BOUND / GENERAL
UNDER-SIZING / ADAPTER-BOUND AND TRANSFER-SPECIFIC / PARTIAL / INCONCLUSIVE.

`--selfcheck` passes: all 4 surfaces emit `[N,H,Q]`, the ladder is strictly larger than the control,
the MLP is verified **genuinely non-affine** (else the probe would be vacuous), and every verdict
branch fires.

**GPU enforced.** The probe exits rather than start on CPU (`--allow-cpu` overrides). Confirmed
torch 2.6.0+cu124, `DEVICE=cuda`, RTX 3060.

```
conda run -n ebola-train python capacity_probe.py
```

### Client documents — 3 ready to send

| document | words | state |
|---|---|---|
| `Reports/Encoder_Results_Stakeholder_Brief.docx` | 1,459 | ready |
| `Reports/Ebola_Support_Set_Decision.docx` | 1,274 | ready |
| `Reports/MAML_Decision.docx` | 1,514 | ready |

`MAML_Decision.md` was **rewritten in the stakeholder register** (short, plain language, no parameter
counts or cost models) to match the encoder brief, on client instruction. All three verified: house
style matches `Week3_Results_Summary.docx`, **0 markdown leaks, exact numeric round-trip md→docx**.
The academic `Encoder_Results_Consolidated` stays internal — client asked for the stakeholder one only.

### Open / not done

- [ ] Run the probe. **Not launched.**
- [ ] `Meta_Learning_Decision.md` (superseded first draft, repo root) — **delete before anyone sends
      it by mistake.** It contains the "you're wrong about the linear probe" claim and the 388 figure.
- [ ] `Ebola_Audit_Note.md` — **do not send unchanged**, predates issues #3/#4.
- [ ] Conformal variant (ACI / CQR / EnbPI) still unanswered; client asked directly.
- [ ] Integrated Gradients scoping note still not written; client suggested it over SHAP.
- [ ] **`Reports/` is gitignored** (`.gitignore:6`), so none of the three client documents has version
      history — the same missing-provenance class the audit flagged for `results/lodo/*.json`.

---

## 🔵 EBOLA SUPPORT SET — hardened, and it is now a client decision (2026-07-30 evening)

Asked whether support-set *selection* was the only problem with the Ebola split. **It is not.** Tested
each candidate failure independently against `data/processed/ebola.npz`. Four issues, only one of
which a different selection rule fixes.

| # | issue | fixed by reselecting support? |
|---|---|---|
| 1 | calendar prefix lands in the sparsest weeks → 27 cells, **9 of 61 districts** | **yes** |
| 2 | support spans only **7 weeks**, so any h>7 has zero targets by construction | **no** |
| 3 | scaler fit on 27 early cells, applied at the epidemic peak | **no** |
| 4 | 12 districts have ≤10 observed weeks; 1 has a single query cell, yet carries full `node_mean` weight | **no** |

**#2 proved by counterfactual:** pretend every one of the 61 districts reported in every support week.
h3 goes 16 → 305 and h5 6 → 183, but **h10 and h15 stay at 0**. Perfect reporting changes nothing past
h7. That is arithmetic, not data quality.

**#3 quantified:** scaler is `log1p` then standardise, **pooled** (one μ/σ for all 61 nodes),
μ=1.130965 σ=1.159910 — and it equals `mean(log1p(support raw))` exactly, so it is fit on support
alone. It says a typical week is **2.1 cases**; the scored cells average **19.19** and reach **1,428**.
**426 of 1,272 query cells (33%)** sit beyond 1σ of what the scaler treats as normal. The FiLM adapter
is the component meant to correct scale, and it has 16 examples at h3 and none at h10/h15.

### The finding that shaped the options: a six-week reporting blackout

| t | date | districts | cases |
|---|---|---|---|
| 12 | 2014-06-28 | 7 | 50 |
| **13–18** | Jul–early Aug | **0** | **0** |
| 19 | 2014-08-16 | 20 | 1,399 |
| 20 | 2014-08-23 | 34 | 513 |

**A 16-week support set is byte-identical to a 12-week one.** 16 must be struck from the options.

### Support-length sweep, w=20 fixed (adaptation pairs / districts / mean window padding)

| L | cells | districts | h3 | h5 | h10 | h15 | scaler ratio |
|---|---|---|---|---|---|---|---|
| **7 (current)** | 27 | 9 | 16 / 8d / 89% | 6 / 2d / 90% | **0** | **0** | 3.8× |
| 8 | 30 | 10 | 19 / 9d / 86% | 9 / 3d / 87% | **0** | **0** | 3.6× |
| 12 | 59 | 18 | 48 / 17d / 69% | 38 / 14d / 72% | 18 / 9d / 89% | **0** | 2.5× |
| 16 | *identical to 12* | | | | | | |
| 19 | 79 | 22 | 68 / 22d / 53% | 58 / 22d | 38 / 21d | **20 / 20d** | 0.8× |
| 20 | 113 | 36 | 102 / 36d / 39% | 92 / 36d | 72 / 35d | **54 / 34d** | 0.9× |

### ⚠ Correction to an earlier claim in this file

I previously wrote that extending support **eats query**. **That is wrong.** Scored query pairs are
**identical at every L up to 20** — 1,151 / 1,075 / 866 / 642 at h3/h5/h10/h15, covering 61/61/59/58
districts. A full-window query target sits at t≥22, and support only reaches t≤20. Query *cells* drop
1,272→1,240 but those cells were never scoreable targets. **No option below L=20 costs a single scored
forecast.** The cost of a longer support set is to the *claim*, not the evaluation.

Full unpadded 20-week windows first appear at **L=22** (h3) and would need **L=34** for h15 — two
thirds of the outbreak. Not proposed.

### Left to the client, deliberately

`Reports/Ebola_Support_Set_Decision.md` + `.docx` (41 KB). House style verified: 0 markdown leaks,
0 em dashes, **214/214 numeric round-trip**, Calibri 11, 4× `Light Grid Accent 1`, 9pt cells,
Title + Heading 2 matching the other Week-4 reports.

The document states the trade-off and **does not recommend**: keep the strongest few-shot framing
(L=7/8/12, accepting h15 is permanently zero-shot and the scaler stays mis-calibrated), or take
L=19/20 for all four horizons and a working scaler, accepting the headline becomes moderate-data
transfer rather than few-shot. The adaptation layer was sized against 27 examples; L=20 gives 113.

**Blocked on the client's answer before the Ebola set can be scored.**

---

## 🔴 HEADLINE CHANGE — cross-disease transfer does not survive the fold fix (2026-07-30)

**The §3 LDO runs have landed** (Day-15 recorded "no real runs launched" — that is now stale) and
everything below in §2/§3 is superseded by the consolidated matrix.

**Deliverables built this session, all numbers recomputed from disk, none carried forward:**

| artifact | what it is |
|---|---|
| `results_matrix.py` | generator; `--selfcheck` passes |
| `Results_Matrix.md` | 768 (regime,dataset,horizon,metric) cells, 7 regimes × 4 datasets × 4 horizons |
| `Reports/Encoder_Results_Consolidated.md` | academic write-up |
| `Reports/Encoder_Results_Stakeholder_Brief.md` | client-facing, plain language |
| `Reports/Encoder_Results_Consolidated.docx` | rendered, 48,832 B |
| `Reports/Encoder_Results_Stakeholder_Brief.docx` | rendered, 41,890 B |
| `Reports/md_to_docx.py` | markdown → docx renderer, `--selfcheck` passes |

**Word output — markdown stays the single source, `.docx` is only a render**, so numbers cannot drift
between them. Regenerate with `python Reports/md_to_docx.py` using **BASE python** (`python-docx` is
not in the `ebola-train` env); close the file in Word first or the write fails `PermissionError`.

Styled to match `Reports/Week3_Results_Summary.docx` per the client's instruction — Calibri 11,
`Light Grid Accent 1` tables, 9 pt bold header / 9 pt body, level-0 title, no custom colours or rules.
Conformance checked against that reference file: **both docs MATCH**. An earlier bespoke design was
rejected and is not to be reintroduced.

Two converter bugs found and fixed, both now asserted in `_demo()`:
1. **Line-at-a-time parsing.** Markdown hard-wraps paragraphs, so a `**bold**` span crossing a line
   break leaked literal `**` and split one sentence into several Word paragraphs. Blocks are now
   accumulated and rendered only when the block ends.
2. **Non-recursive inline parsing.** `` **`nrmse` is missing** `` (code inside bold) kept its backticks.

Render verified: **0 leftover markdown** in paragraphs and cells in both docs, 0 broken sentences, and
an **exact numeric-token round-trip** md→docx (457 tokens academic, 77 stakeholder). The only benign
diffs are ordered-list digits (Word auto-numbers them) and fenced code blocks.

**Rescore done.** `score.py` self-check passes (7 metrics). `rescore_encoder.py`: **65/65 artifacts
backfilled with nrmse, 0 skipped on mask mismatch** (dry-run verified first).

### The three findings

1. **The Week-3 transfer headline was substantially a REFERENCE artifact.** Holding the transfer
   number fixed and varying only the reference reproduces **all three** of the client's arithmetic
   complaints to the decimal — us-regions h3 +19.9% (Ref A) vs **+9.9%** (Ref B) against their 9.8%;
   japan h5 +4.5% vs **+7.4%** against their 7.4%; us-states h3 +5.8% vs **+1.4%**. Dengue h3
   **flips sign**: +2.9% → **−15.4%**. Cause: single-disease dengue seed 42 is an outlier (49.98 vs
   ~37–47 for the other four). Independently corroborated — `analysis.py` asserts −15.4% via a
   separate code path.
2. **Cross-disease transfer (LDO) is negative.** On RMSE, **12 of 16 cells significantly negative,
   4 within noise, 0 positive** — identical tally for MAE. The three biggest LODO wins (+28.0%,
   +24.0%, +19.9%) were all flu panels with flu still in training, exactly the leakage the client
   predicted. Degradation is monotone in horizon, ~−51% at h15 on two panels.
3. **Cross-disease zero-shot fails on influenza** (−311% japan h3, −559% h10). Adaptation is
   load-bearing, not an enhancement.

### Client-required metrics — gap found and closed

Review Doc ¶7 names **three** metrics. First cut of the report shipped only `nrmse`; `peak_intensity`
and `peak_timing` were omitted "for readability". **Fixed** — all three now in the matrix and both
reports.

- **Peak timing is poor and previously unreported:** us-regions is 43–68 steps off the true peak at
  every horizon; only japan h3 (4.87) is near operational. Argues against deployment-readiness framing.
- **`nrmse` flips the difficulty ranking:** dengue is the HARDEST panel (1.80→2.08), not the easiest.
  Raw RMSE was flattering it.
- **One honest counter-signal:** 2 of 32 peak cells clear positive (japan h5 +24.7%, dengue h10
  +26.5% peak timing). Tally 2 pos / 2 neg / 28 within noise — consistent with multiple comparisons.
  Recorded, **not** promoted to a finding.

### Verification (client asked for a re-verify before sign-off)

- **264 assertions parsed OUT of the two documents and recomputed from the run JSONs — all match.**
  Direction is doc→data, so a stale figure cannot pass by agreeing with memory. Significance labels
  are checked too: every bold = clears, every "within noise" = interval covers zero.
- **Mutation test proves the verifier is not vacuous** — corrupting 2 numbers was caught, and only
  those 2. Coverage floor added (exits non-zero under 200 assertions).
- **Like-for-like `n_nodes` guard: PASS, zero mismatches.** Every arm scored the same node population
  as its reference, so no delta divides two different populations.
- Verifier currently in scratchpad, **not yet in the repo** — worth landing as `verify_docs.py`.
- Re-run after the docx work: still **ALL MATCH**, so the render did not disturb the numbers.

**Conventions saved to project memory** so this does not have to be re-derived: the `.docx` house
style + render pipeline (`report-docx-house-style`) and the verification protocol
(`results-doc-verification`). Also **corrected a now-false memory** — `transfer-mechanism-finding`
still asserted transfer was POSITIVE and named the 5-seed run as its gate; that gate ran and the
effect collapsed, so it is marked superseded. Its own escape clause ("Option B is off the table
unless the confirmation collapses the effect") has therefore **triggered — Option B is live.**

### Two counting errors I made and corrected

Both caught by machine-checking my own prose, not by eye: "nine of sixteen" → **twelve**; "26 other
peak cells" → **32 total, 28 within noise**. Flagged here because the write-up standard is that a
number in prose gets the same check as a number in a table.

### What this does to §4 meta-learning

**Raises its priority.** Freeze-then-adapt was to be the ablation baseline for meta-learning; that
baseline is now a **negative result**. Meta-learning stops being an add-on and becomes the more
likely route to a positive finding. Still owed: pick MAML/Reptile/ProtoNet + schedule yes/no.

### Still open from this session

- [ ] **UQ metrics remain unpopulated.** WIS/CRPS/coverage/PIT implemented and self-checked but need
      quantile predictions; no run archives them. Largest remaining gap vs ¶7. Cheap on the next
      run, expensive to retrofit.
- [ ] `nrmse` missing for `encoder_lodo_zeroshot__` (4 files, seed 42). Zero-shot writes JSON only —
      `train/lodo.py:418` discards the per-node arrays (`zrecs, _, _, _`) — so `rescore_encoder.py`,
      which globs `*__pernode.npz`, **structurally cannot** backfill it. Needs a rerun, not a rescore.
- [ ] **LDO flu fold is 4 seeds (52/62/72/82), missing seed 42**; dengue fold has all 5.
- [ ] LODO (leave-one-dataset-out) is still **1 seed** — and that seed is the known dengue outlier.
      Do not quote LODO figures; complete the 5 seeds if the population-transfer claim goes in.
- [ ] New files uncommitted.

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
| **HeatGNN** `export_heatgnn.py --run --workers 8 --threads 1` | **37/60**, runner + 8 workers alive. japan **20/20 DONE**, us-states 8/20, us-regions 9/20 | 23 left at 1.55 runs/h → **~14.8 h, ~13:20 on 31 Jul** |
| **MTGNN** | 80/80 files, but **degenerate — see below** | done 29 Jul |

**ETA is empirical now, not modelled:** 37 runs since 23:08 on 29 Jul = **1.55 runs/h** measured end
to end. The earlier 8.9 h estimate assumed ~440 epochs/run; the real figure is higher (see below), so
that estimate was optimistic. Contention measured at **0.57 cores/worker, 4.53 of 6 physical**.

### ✅ Self-loop fix — PROVEN at full length (closes the audit's open item 1)

`HeatGNN__influenza_japan__h3__seed42.log`: **736 epochs, zero `nan values`, `[day15] saved test
preds` present**, 27.90 s/epoch mean under 8-way contention. **All 20 japan runs completed.**
Across *every* HeatGNN log on disk: **0 files contain `nan values`.**

Note the epoch count: runs early-stop at ~736, not the ~440 assumed from us-regions, and not 1500.
That is the whole ETA gap.

### 🔴 MTGNN is degenerate — independently re-verified, worse than first recorded

| check | result |
|---|---|
| files with **exactly one** distinct value | **47 / 80** |
| files with **≤25** distinct values over 1.5M+ cells | **57 / 80** |
| **byte-identical** prediction arrays across different seeds | **32 pairs** |

Seed-inert: dengue h5/h10/h15 have seed 42 ≡ 52 ≡ 62 ≡ 72 ≡ 82.

**The collapse is horizon-structured, which is the diagnostic:** dengue **h3 is healthy** (596k–795k
distinct values) while h5/h10/h15 are **5/5 seeds constant**. We run `train_multi_step` with
`seq_out_len=15` and pull {3,5,10,15} from one rollout, so the rollout degenerates past the early
steps. Points at MTGNN's **curriculum-learning args** (`step_size1`/`step_size2`/`num_split`), not at
anything we did.

**It is not our export.** Same bundles, other models:

| model | files | constant | min distinct |
|---|---|---|---|
| EpiGNN | 80 | **0** | 2,359 |
| ColaGNN | 60 | **0** | 2,358 |
| HeatGNN | 37 | **0** | 2,359 |

So this is a Section-A reproduction failure, not a data bug.

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
| HeatGNN | **37** | 60 | **running**, 23 left, ~13:20 on 31 Jul |

HeatGNN by dataset (30 Jul 22:35): **japan 20/20 DONE**, us-states 8/20, us-regions 9/20.

**Solved this session** (detail in `decisions.md` D6–D9):
- HeatGNN NaN on japan/us-states — isolated nodes, fixed with self-loops. **Now proven at full
  length: 736 epochs, 0 NaN, all 20 japan runs complete, 0 NaN in any HeatGNN log on disk.**
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
- [x] ~~§2 re-opens when the 5-seed run lands.~~ **IT LANDED, AND §2 IS NOW SUPERSEDED.** The
      statistic did switch to a **paired-by-seed delta** as planned (t-based 95% interval; at n=4–5
      the critical value is 2.78–3.18, not 1.96 — using the normal quantile would manufacture
      significance). The table below is the 1-seed LODO picture and is **retained for history only**;
      the current numbers are in `Results_Matrix.md` §3 and `Reports/Encoder_Results_Consolidated.md`.
      Headline change: the surviving positive cells do **not** survive the disease-level fold.

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

## §3 Fold restructure — code built, **runs landed, results superseded this section** (2026-07-30)

> The heading used to read "runs not launched". They ran. The LDO results are in the headline section
> at the top of this file; the design record below is kept because the 12 decisions still stand.

All 12 design decisions taken with the client, then put through an adversarial review. Two decisions
were **reversed by that review**; three of the reviewer's own claims were wrong and rejected.

### Locked design

| # | decision |
|---|---|
| 1 | **ONE shared Adapter** across the 3 flu bundles (not 3) — if flu is one disease it gets one FiLM surface |
| 2 | Fit on **full pooled flu train folds** ("big pile"); k-shot sweep deferred |
| 3 | **Block-diagonal supergraph** at fit time, all 3 bundles per gradient step |
| 4 | **Uniform 1/3** per bundle, **no node weighting** — verified identical to what flu→dengue did (`dataset_weights(ds,"uniform")`, `_node_weight(...)→None`) |
| 5 | Dengue-only trunk via `_fit_trunk`, step-based, 91,000 steps — symmetric with the other direction |
| 6 | New **`encoder_ldo__`** prefix + **`fold_structure`** field on every record; route added to `results_paths.py` (self-check now 14 families) |
| 7 | Zero-shot reported, **labelled single-source** (it is one adapter, not a mean of 3) |
| 8 | ~~Block-diagonal at test~~ → **REVERSED: reuse `train/joint.py::_test_dataset`, solo graphs.** `_equiv_check` proves block == solo, so it is the same number with no second scoring path |
| 9 | **Three per-dataset rows**, no pooled "flu" row (D3) |
| 10 | Provenance commits — **DONE**, `6fc1a98` + `8a8a21f` |
| 11 | flu→dengue **rerun** with a shared trunk adapter, **all 5 seeds fresh**. Kills §3a's "relabel, no rerun" |
| 12 | 10 runs on GPU (RTX 3060, idle; HeatGNN is CPU so no contention) |

**On #11's real cost:** the reviewer called it ~15h of waste. It is **one marginal run**. §3d already
required seeds {52,62,72,82} for this fold because D3 demands dispersion and it has only seed 42 — so
4 runs were always coming. #11 only forfeits reuse of seed 42.

### Built and self-checked

- `results_paths.py` — `encoder_ldo__` / `encoder_ldo_zeroshot__` routes, 3 new demo cases. **Passes.**
- `train/lodo.py::_fit_trunk(share_adapter=True)` — one Adapter registered once, handed to every
  in-dataset. `share_adapter=False` keeps the old per-dataset behaviour.
- `train/lodo.py::_fit_shared_adapter` — new. Trunk frozen, ONE adapter over N bundles via the
  block-diagonal supergraph, pooled `_val_pinball` at uniform 1/3, epoch = one pass over the longest
  bundle's origins with shorter bundles cycling.
- `train/lodo.py::run_ldo_fold` — both directions, `fold_structure` + `ldo_direction` in run_meta.
- `_smoke_ldo()` — asserts shared-vs-separate adapter identity, trunk frozen, `fold_structure` on
  every record, records stay per-dataset.
- **Bug fixed en route:** the existing `_smoke` asserted `len(HORIZONS) * 6` records. `score.METRICS`
  became 7 when `nrmse` landed, so that assertion was already broken. Now `len(score.METRICS)`.

### Not done

- [x] ~~**No real runs launched.**~~ **SUPERSEDED 2026-07-30 — the runs landed.** On disk: 17
      `encoder_ldo__` + 17 `encoder_ldo_zeroshot__` record sets, `fold_structure:
      leave-one-disease-out` on every record, both `ldo_direction` arms present. Dengue fold has 5
      seeds; the flu fold has 4 (missing seed 42). Results consolidated in `Results_Matrix.md` and
      the two `Reports/` write-ups — see the headline section at the top of this file.
- [ ] **Dispersion machinery for the 3-row shape.** `_report` and `analysis.py --transfer` assume one
      held-out set per fold; dengue→flu yields 3 rows from 1 trunk run. Read-side only, so runs are
      not blocked — but the fold must not be reported until this lands, or it breaks D3 by omission.
      (Found by the reviewer; a genuine gap in the 12 decisions.)

### Reviewer claims rejected, with evidence

1. *"lodo.py and results_paths.py are untracked, commit before generating artifacts"* — **wrong**, both
   committed in `6fc1a98` before the review was dispatched. `git ls-files` confirms.
2. *"#11 costs ~15h"* — **wrong**, one marginal run (above).
3. *"a MAML trunk would waste the freeze-then-adapt seeds"* — **weak**; the client already said
   *"freeze-and-adapt then becomes the ablation"*, and an ablation still needs its numbers.

Architecture precondition still holds: `SharedEncoder`/`Adapter` have no node dimension; `_prepare`
and `block_diag_sparse` take arbitrary name lists.

---

## §5 Ebola audit — **SENT-READY**, written 2026-07-30

`Ebola_Audit_Note.md` — client-facing, answers all five questions, arithmetic run out loud as asked.

- Weekly, 61 × 52, 41% observed, graph usable (146 edges, 0 isolated, 21 cross-border).
- Client's per-district origin estimate of ~17 at h15 was **very nearly right** — it is 18. Their
  error was the unit: evaluation counts (district, origin) pairs, so h15 has **642 pairs / 58 districts**.
- **Correction 1:** h10/h15 *are* evaluable — zero-shot, not unmeasurable.
- **Correction 2:** the 27-example floor came from **Ebola's own support**, not the smallest dev set.
  Their reading was backwards, in our favour.
- ~~**The finding they did not raise:**~~ **CORRECTED 2026-07-30 — this was our overclaim.** The
  zero-adaptation result was **already ours and already theirs**: written in
  `encoder_architecture_plan.md:121-122` and `:133-135` before the client asked ("There are zero
  support origins" / "h=10 and h=15 admit none, ever"), escalated as **[CONFIRM-P7]**, and settled at
  Day 13 (`Day13_Summary.md:77`). What the audit genuinely adds is the **realised counts after
  missingness** — the plan gives origin ranges, not pair counts. h5 reads as "3 origins per district"
  in the plan; after missingness it is **6 pairs on 2 districts**. h3 is **16 pairs on 8 districts**.
  **Practical adaptation horizon is h3 only.**
- Flagged honestly: deferring the k-shot sweep means the first exercise of the 27-cell regime will be
  on Ebola itself, which is scored once. If it comes back weak we cannot separate "transfer fails"
  from "16 pairs fit nothing."

**SUPERSEDED IN PART, 2026-07-30 evening.** The audit was hardened after this note was written and
found **three further issues beyond support selection** (the 7-week span capping h>7 at zero
regardless of reporting, the scaler fit on the wrong distribution, and districts too sparse to score),
plus the **week 13–18 reporting blackout**. Full detail in the blue section at the top of this file.

Consequences for this note:
- The support set is now an **open client decision**, not a settled input.
  `Reports/Ebola_Support_Set_Decision.md` + `.docx` presents the options and does **not** recommend.
- The statement that a longer support set would cost query data is **wrong** and was corrected: no
  option below L=20 costs a single scored forecast.
- **Do not send `Ebola_Audit_Note.md` as-is.** It documents issues #1 and #2 but not #3 (scaler) or
  #4 (sparse districts), and it treats the 7-week support as fixed. Either fold the new findings in,
  or send it alongside the decision document so the client sees both.

---

## §4 Meta-learning — **client wants an answer this week · PRIORITY RAISED**

- [ ] Pick MAML / Reptile / ProtoNet + one paragraph of justification
- [ ] Straight yes/no on schedule fit

Not started. Client said explicitly: *"don't quietly absorb it."* Still the item most at risk of
slipping silently.

**Why it is now more than a checkbox (2026-07-30).** Freeze-then-adapt was to be the *ablation
baseline* meta-learning gets compared against — and that baseline is now a **negative result**
(§ headline: 12/16 LDO cells significantly negative, 0 positive). So meta-learning stops being an
add-on and becomes the more likely route to a positive finding. The client's own framing —
*"freeze-and-adapt then becomes the ablation"* — now cuts the other way: the ablation is what we have,
and it does not work. Answer this before the Ebola run, not after.

---

## §5 Ebola audit — numbers done, note written, **now blocked on a client decision**

- [x] All figures computed (see `decisions.md` D4)
- [x] Written up as `Ebola_Audit_Note.md`, 2026-07-30 — client-facing, both corrections included
- [x] **Audit hardened 2026-07-30 evening** — 3 further issues found beyond support selection, plus
      the week 13–18 blackout. See the blue section at the top of this file.
- [x] **Support-length sweep run** (L = 7/8/12/16/19/20/22/34, w=20 fixed), all recomputed from
      `data/processed/ebola.npz`.
- [x] **`Reports/Ebola_Support_Set_Decision.md` + `.docx`** written and style-verified. Decision
      deliberately left open.
- [x] **`Ebola_Audit_Note.md` rewritten 2026-07-30 evening.** Two fixes: (1) the P7 overclaim removed,
      §5 now opens "this constraint is not new, and we are not presenting it as a discovery", quotes
      the plan, credits CONFIRM-P7, and confines our contribution to the realised counts; (2) new §6
      added covering the scaler and the sparse districts, pointing at the support-set decision. All
      10 em dashes stripped. Sections renumbered 1-8.
- [ ] **Send both documents together.** Neither is sent. They cross-reference and share the 3.8x
      scale ratio deliberately, so they should not go out separately.
- [ ] **Client's answer on support length.** Blocks the Ebola run: the set is scored exactly once, so
      the split must be frozen and hashed first.

---

## §6–§8 Not started

Joint-training 2-day timebox · WIS/CRPS/coverage/PIT/peak metrics · ARIMA+GBM floors · conformal
split reservation and variant choice · explainability scoping · gate figure · seed ensembling ·
delta-prediction test · pre-registration.

---

## Housekeeping

Committed 2026-07-30: `6fc1a98` (train/lodo.py, analysis.py, results_paths.py — closes audit open
item 2) and `8a8a21f` (score.py metrics, rescore_encoder.py, paper_compare.py, Published Model
benchmarks/, Updated_Scores.txt, this file, decisions.md).

**Provenance defect created by that commit — CLOSED 2026-07-30, commit `1e8286e`.**
`paper_compare.py` was committed while importing `score_baseline.py`, which was untracked. Fixed by
tracking it. The full first-party import closure of `paper_compare.py` is now tracked: `score.py`,
`results_paths.py`, `score_baseline.py`, `bundles.py`. Import verified before committing, and a
pathspec commit was used so the two already-staged docs were left staged, not swept in.

**Residual, not the same defect:** `results_paths.py` is tracked but carries uncommitted working-tree
edits (the step-3 `encoder_ldo__` routes), so HEAD differs from disk. Clear before the three-way
table is cited externally.

Uncommitted: `decisions_day_13.md`, `train/loop.py`, plus today's step-3 edits to `train/lodo.py`
and `results_paths.py`, and new `Ebola_Audit_Note.md`.

**New and untracked, 2026-07-30 evening:** `Reports/Ebola_Support_Set_Decision.md` and
`Reports/Ebola_Support_Set_Decision.docx`.

**Note for anyone regenerating reports:** `Reports/Encoder_Results_Consolidated.docx` is currently
**open in Word** (a `~$` lock file sits beside it). `md_to_docx.py` will fail on that one file with
`PermissionError` until it is closed. It did not affect the decision-document render.
Untracked: `ablation/`, `train/lodo.py`, `results_paths.py`, `export_*.py`, `score_baseline.py`,
`run_baselines.py`, `make_results_doc.py`, `results_summary.txt`, `test_run.py`, `Review Doc.md`,
both Week-3 docs, `Phase3_Week4_Work_Order.md`, this file, `decisions.md`,
`Published Model benchmarks/`, `paper_compare.py`, `rescore_encoder.py`, `Updated_Scores.txt`.

`Phase3_Week3_Results_and_Direction.md` §6 still reads *"[next — the one gate] 5-seed LODO
confirmation"*. **Countermanded** by the client. Mark superseded.

# Priority Fix Progress

**2026-08-10.** Covers the ANIL verdict, the conformal decision packet, the verification of both
against the artifacts on disk, the state of `conformal.py`, and the pre-registration hardening that
had to close before the Ebola run started (§6) — now closed and proven end to end (§6.1).

**State:** every code gate is cleared and **the overnight queue has finished clean** (§9.3).
`run_tonight.py` ran `train.ebola --all` from 20:09:53 on 2026-08-10 through the three influenza
ceilings and dengue, ending 12:59:27 on 2026-08-11 with zero failures. Both Ebola arms are scored,
all 25 ceiling quantile archives exist, and the conformal wrapper is fitted, frozen and validated
(§5.1). **The results are now read and are in §10** - point forecast, Ebola calibration, and the
ceiling reference. They are first reads with no significance testing attached; §10.4 states plainly
what is still missing before any of it is reportable.

**The headline is an inversion nobody scoped for: zero-shot beats few-shot on 7 of 8 cells**, and
the same ordering shows up independently in the interval calibration (§10.1, §10.2).

Companion artifacts: `train/anil.py` (rebuilt after review, unrun), `conformal.py` (built,
self-checked, fitted and frozen, `--apply` landed), `run_tonight.py` (the overnight queue),
`results/misc/conformal_config.json` (balanced fit, digest `8e2b657d...`),
`progress/decisions/decisions.md` (D17-D19), `progress/decisions/Ebola_Prereg.md` (amendment A7 and
new section 5b), `train/ebola.py` and `results_paths.py` (Ebola zero-shot quantile archiving),
`train/lodo.py` (LDO3 zero-shot quantiles + zero-shot adapter in the checkpoint).

---

## 0. Priority order, as it now stands

Two independent reviews landed today and both moved the queue. The order below is the operative one.

| # | item | gate | resource |
|---|---|---|---|
| 0 | **Ebola pre-registration hardening (A7)** | **one-way door, closes when scoring starts** | **none, DONE and VERIFIED (section 6)** |
| 1 | Ebola case study (G3) | REQUIRED, built, pre-registered, unrun | GPU, tonight |
| 2 | Conformal wrapper (G4) | REQUIRED, was zero code | CPU, no Ebola dependency |
| 3 | Single-disease ceiling re-train, quantiles + checkpoints | gates the G4 reference comparison | GPU, ~13.1 h |
| 4 | ANIL ablation | not required, not tonight | GPU, 11-16 h |
| — | HeatGNN rescore | G6 gate | **parked on instruction, 2026-08-10 (section 7)** |
| — | LDO3 zero-shot quantiles | G4 completeness on the dev folds | **backlog, 15 trunk retrains (section 4)** |

Items 1 and 2 do not block each other and are scheduled for the same night. Item 0 was discovered
today, is irreversible once item 1 starts, and is now closed and proven end to end.

**Do not conflate the three zero-shot quantile gaps.** Different files, different fixes, very
different prices.

| arm | file | fix | state |
|---|---|---|---|
| Ebola case study | `train/ebola.py` | code, ships with the scored run | **done, verified (§6)** |
| LDO3 development folds | `train/lodo.py` | code done; the existing 15 runs need a **retrain**, not a rescore | **code done, data backlog (§4)** |
| single-disease ceilings | — | no zero-shot arm exists; ceilings are same-disease | n/a |

---

## 1. ANIL: worth running with changes, but fourth

`train/anil.py` is now 929 lines, untracked, **eight** self-checks passing, preflight passing on both
arms, second-order term confirmed live (-2.096e-03 exact against 0.0 first-order). **Zero training
has been run** and no `results/misc/anil_*.json` exists, so nothing is lost by parking it.

Two independent reviews were commissioned: one on technical accuracy, one on alignment with the
project's objectives. The alignment review's conclusions are in §1.2 below and stand unchanged. The
technical review found six defects that would have corrupted the number; all six are confirmed and
all six are now fixed (§1.1). The earlier "six self-checks passing" line in this document described
the pre-review module and is superseded.

### 1.1 Six confirmed defects, all fixed

Every one was verified directly, not accepted on the reviewer's word.

- **Support labels leaked into the query targets.** `gap` was counted in *origins* while the loss
  lives on targets at `t + h`. Verified: support origins [171, 172] give target weeks
  {174,175,176,177,181,182,186,187}; query origins [177, 178] give {180,181,182,183,187,188,192,193};
  the sets share **{181, 182, 187}**. The inner loop was fitting on labels the query was then scored
  against, so the meta-objective flattered ANIL. Disjointness requires a separation of
  `max(H) - min(H) = 12` in origin *value*; the default was 4. Now derived from the horizon set, not
  a tuneable, and enforced on origin values because train origins are not contiguous.
- **No validation, no early stop, no LR schedule in meta-training.** The ANIL trunk was the last
  iterate while the baseline trunk it is differenced against was best-val selected inside
  `_fit_trunk`. The delta therefore confounded the meta-objective with the model-selection rule -
  D17's defect with the sign inverted. Now: a fixed set of held-out episodes drawn from the **val**
  origins, cosine outer schedule, patience, best-val state restored, and a hard refusal to finish if
  training ended before the first validation check.
- **No fine-tuning control.** The ANIL arm received 8,000 extra dengue steps that the baseline never
  had, so a negative result would have been uninterpretable. Added `--arm control`: same episode
  stream, same number of outer updates, no inner loop, support and query cells pooled so both arms
  see identical data.
- **`compare()` took the t critical value from the seed-row count, not the cell's pair count.**
  With 5 rows and a reference covering 3, it applied t = 2.776 where 4.303 is correct - an interval
  ~35% too narrow, i.e. cells printed as significant that sit inside the noise floor, on exactly the
  rows where seeds are missing. This is a D3 violation. Fixed, with per-cell `n` now printed.
- **71% of episodes were discarded after paying for the trunk forwards.** Simulated at the module
  defaults on dengue: 5,680 of 8,000 draws had no observed support cell and 92 more no observed
  query cell, so `--outer-steps 8000` delivered ~2,228 real updates while paying two full 7,165-node
  forwards for most of the rest. The mask check now runs before the forward, and `--timing` reports
  the true skip rate and update yield rather than a step count that overstates the work by 3.6x.
- **Three smaller ones.** `--refit` was a no-op because the resume skip fired before `run_seed` ever
  consulted it. The checkpoint-reuse path never re-seeded, so resume did not reproduce the non-resume
  result. And every metric except RMSE was computed and thrown away - the exact D14 failure - so
  records, per-node, per-origin **and quantiles** are now all archived.

Two further fixes went in unprompted. `--from-scratch` was calling `_fit_trunk` at the default
`patience=12`, reproducing the precise early-stop defect D17 documented and A5 overrode to 30. And
self-check #3 was theatre: it hand-sliced the tensors and then asserted that tensor indexing works,
never calling the code that does the restricting. It now tests `_targets` directly and checks the
restricted rows are the *right* rows.

The self-check suite is now eight, each paired with a control that must fail the other way -
including one asserting the pre-fix `gap=4` case is *still* detectably overlapping, so the leak check
cannot pass vacuously, and one asserting the trunk is present in the outer optimiser, without which
the module would not be meta-learning at all and every earlier check would still have passed.

### 1.2 Four substantive gaps, from the alignment review — unchanged, still open

- **It answers a narrower question than D2 asked.** The client's words were episodic meta-training
  *across diseases*. `META_NAME = "dengue"` is one disease. The module raises this objection against
  itself at line 31; per-country episodes vary population and time, not disease, so the objection is
  structural and the sampling choice does not mitigate it.
- **There is no few-shot regime.** `meta_test` fits a fresh adapter for 80 epochs on the full train
  folds, which measures trunk quality under ERM adapter fitting. The headline claim is few-shot
  adaptation to 48 Ebola support pairs. The experiment does not test it.
- **The `mlp-256` surface collides with the pre-registration.** Every D19 gain sits at h15 and the
  frozen Ebola primary arm has zero h15 adaptation pairs (D16). `Ebola_Prereg.md` fixes the surface
  as the affine 1,428-parameter adapter. A positive ANIL result would force a re-registration
  request and delay Ebola scoring.
- **The scope decision was never answered.** `MAML_Decision.md` §5 put options A-D to the client and
  Nora and no answer is recorded in the decision log. Running option C now delivers reduced scope
  without the re-scope ever going to Nora.

### 1.3 The seven required changes: three done, four still open

| # | change | state |
|---|---|---|
| 1 | Fix the D3 reporting breaks in `compare()` | **done** - reference line, formula and units on the table, per-cell `n`, both references reported |
| 2 | Report against the affine reference too, since that is what D2 means by "the linear probe" | **done** - both references printed every run; it was already in the same JSON |
| 3 | Pre-commit what a positive result triggers | **done** - the run prints all four null-confounds at exit, and D19's precedent (re-registration is a client request, not a config change) is stated in the docstring |
| 4 | Send the scope answer to the client before running | **open, not ours to close alone** |
| 5 | Meta-train on dengue + covid, warm-started from the LDO3 influenza-held-out trunks | **open** - deliberately not half-built; it needs a different warm-start family and a different comparator |
| 6 | Use the affine surface rather than `mlp-256` | **open** - `--surface affine` exists and works; the default remains `mlp-256`, which is the client-side choice on record |
| 7 | Hold out districts in the episode, not time alone | **open** - stated as a known limit in `episode()`'s docstring rather than left to be discovered |

On item 5 the premise has moved and that is worth recording. D13 rejected cross-disease
meta-training because "COVID is not on disk". That is no longer true: `bundles.py` carries
`covid_us-states` in `DEV_BUNDLE_NAMES`, and `results/lodo/encoder_ldo3__influenza__seed{42,52,62,72,82}__ckpt.pt`
are five trunks trained on **dengue + covid** with influenza held out. The structural fix D13 named
is now a warm-start path change plus a disease-level draw, at similar GPU cost. Running the
single-disease version first is therefore the wrong order, and D13's objection should be re-read as
stale-in-its-premise rather than settled.

On item 6, the argument against `mlp-256` is stronger than the capacity-sweep win count suggested
when the choice was made, and that framing gap is ours: `Reports/MAML_Decision.md` §1 chose ANIL over
MAML *because* its meta-objective is the Ebola procedure differentiated, and that procedure is the
pre-registered 1,428-parameter FiLM+head. An inner loop over 21,780 parameters is not that procedure,
so the surface choice weakens the reason ANIL was selected at all. It also inverts D15, which scoped
the capacity probe as a bound *for* small-read-out ANIL. The counter-argument stands too: `mlp-256`
is the measured strongest rung and picking a weaker surface to protect an argument is its own bias.
The decision is open and belongs to the client-side call, not to this document.

**Decision: park it.** With the remaining four changes it becomes the Week-6 methodological ablation,
after the Ebola case study and the conformal calibration are complete. Five seeds is 11-16 GPU hours
and now two arms, which is the wrong claim on the bottleneck resource this week. The six corruption
defects being fixed does not promote it - it only means that when it does run, the number will mean
something.

---

## 2. Conformal: the decision packet

**Method: ACI, initialised from a cross-disease conformal calibration fitted on the five LDO3
held-out panels.** No retraining, no calibration split out of Ebola, and it still functions at h15
where there is no adaptation data.

**Split conformal is rejected on arithmetic, before the protocol is even reached.** The frozen
primary arm has 48/38/18/0 adaptation pairs at h3/h5/h10/h15. Split conformal needs n >= 9 merely to
form a finite 90% interval. Splitting h3 leaves about 16, where the index lands on
ceil(17 x 0.9) = 16, so the interval is literally the largest residual observed: valid, uselessly
wide, high variance. h10 splits to 9, exactly the boundary. h15 has nothing to split. The split
would also halve the adapter's fitting data, degrading the point forecast to buy an interval that
says nothing.

**EnbPI is dropped on two counts.** ~300 GPU-hours is out of budget, and the out-of-bag problem is
structural: no target-disease data enters training by design, so there are zero out-of-bag residuals
for the headline case, and bootstrapping the adapter alone prices the wrong variance.

**Trunk uncertainty is available but is not the dominant term.** Five seeds on disk are a free deep
ensemble and last-layer Laplace on the frozen trunk is near-free, but for Ebola the dominant term is
domain shift, and no ensemble prices domain shift however many members it has. ACI prices it
directly by adapting to observed miscoverage.

**The ACI update is on the miscoverage fraction, not the binary indicator.**

```
alpha_{t+1} = alpha_t + gamma * (alpha - f_t),   f_t in [0, 1]
```

Same telescoping argument and same bound, much lower variance per step, because each of the 18 Ebola
steps is informed by up to 46 observed districts instead of one outcome. The pooled
(district, origin) variant gives 1,151 steps at h3 but introduces ordering dependence and loses the
one-update-per-timestep reading, so the per-origin stream is kept.

**The correction is per horizon.** Under-coverage is horizon-dependent on every panel, so a single
global multiplier would over-correct h3 and under-correct h15.

**Two disclosures are mandatory in the write-up.** First, the cross-disease calibration carries no
finite-sample guarantee on Ebola; it is a transfer of the correction, and the guarantee comes from
ACI's online adaptation rather than from the initialisation. Second, the worst case at T = 18 is
(alpha_1 + gamma) / (T * gamma) = 0.167, so coverage could in theory sit 17 points off nominal. Both
go in the paper rather than waiting for a reviewer to derive them.

**Sequencing.** Build and validate on the five development panels with the quantiles already on
disk, then apply the frozen wrapper to Ebola once when the support config is locked. Neither blocks
the other.

---

## 3. Verification of the quoted numbers against disk

Coverage was recomputed from the archived LDO3 quantiles rather than read from the earlier summary:
ground truth is `b.raw` in count space, masked by the test phase mask at `t + h`, over 5 panels x 5
seeds. Scripts are in the session scratchpad (`verify_cov.py`, `verify_geom.py`).

| claim | disk | verdict |
|---|---|---|
| coverage 0.49-0.92 across horizons | 0.492 .. 0.918 (seed-mean cells) | exact |
| covid 0.492 at h15 | 0.492 | exact |
| us-regions 0.918 -> 0.838 | 0.918 -> 0.838 | exact |
| dengue 0.815 -> 0.753 | dengue is 0.749 -> 0.678 | **mislabelled** |
| the earlier 0.66-0.90 was h3 only | h3 range is 0.658 .. 0.918 | h3-only correct, top end is 0.92 |
| Ebola pairs 48/38/18/0 | 48/38/18/0 | exact |
| 18 origins | 18 scored origins, both arms | exact |
| up to 61 districts per step | 61 districts exist, max 46 observed at any one origin | roster 61, effective <= 46 |
| ~1,151 pooled steps at h3 | 1,151 exactly (reachable-origin count) | exact |
| split needs n >= 9 | 9 | exact |
| n = 16 -> ceil(17 x 0.9) = 16 = max residual | confirmed | exact |
| h10 split leaves 9, on the boundary | 18/2 = 9, and 9 is the minimum | exact |
| ACI bound ~ 0.167 | 0.15 / 0.9 = 0.1667 | exact |
| dengue ~20, covid ~5,000 counts | dengue mean 26.8 (median 2.0), covid median 4,605 | holds, ~200x gap |
| LDO3 quantiles for all five panels | 5 panels x 5 seeds present | exact |
| four ceilings missing quantiles | dengue + 3 influenza missing, covid 5/5 | exact |

**One correction to the record.** `0.815 -> 0.783` belongs to **influenza_us-states**, not dengue.
Dengue runs `0.749 -> 0.678` pooled and `0.800 -> 0.743` country-macro; neither reading gives 0.753.
The argument is unaffected and slightly strengthened, because dengue's h15 is worse than the figure
quoted. The label must be fixed before it reaches a document.

Full recomputed table, 5-seed mean +- sd, empirical 90% coverage, LDO3 adapted arm:

| panel | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| dengue | 0.749 +- 0.009 | 0.741 +- 0.003 | 0.710 +- 0.007 | 0.678 +- 0.008 |
| influenza_japan | 0.899 +- 0.006 | 0.900 +- 0.011 | 0.882 +- 0.025 | 0.887 +- 0.024 |
| influenza_us-regions | 0.918 +- 0.050 | 0.904 +- 0.063 | 0.862 +- 0.056 | 0.838 +- 0.064 |
| influenza_us-states | 0.815 +- 0.018 | 0.810 +- 0.016 | 0.790 +- 0.016 | 0.783 +- 0.010 |
| covid_us-states | 0.658 +- 0.081 | 0.712 +- 0.084 | 0.535 +- 0.097 | 0.492 +- 0.078 |

---

## 4. The ceiling job is a retrain, not a re-score, and it is ~13.1 h

**No checkpoints exist for the four missing panels.** Only covid has `__ckpt.pt`, at 5/5 seeds.
Dengue and the three influenza panels have json and npz only. There is no model to re-score, so the
quantile archive can only be produced by training again.

Priced from measurement, not estimate: the D18 recovery ran these exact 20 runs and
`results/reports/single_recovery.log` carries the per-run minutes.

| panel | 5 seeds | detail |
|---|---|---|
| dengue | 759.2 min (12.7 h) | 135.9 / 155.9 / 169.6 / 147.7 / 150.1 |
| influenza_japan | 11.0 min | |
| influenza_us-regions | ~11 min | 4 seeds logged at ~2.2 min each |
| influenza_us-states | 9.2 min | |
| **total** | **~790 min ~ 13.2 h** | log total for the 20 runs: 786.4 min |

**Dengue is 96% of the cost.** Splitting the job accordingly: the three influenza panels take about
35 minutes and get four of five reference panels; dengue is a separate 12.7 h block.

**Two consequences.**

- **This job and the D18 reproducibility re-run are the same 20 runs.** Run once, with
  `write_quantiles` and `write_checkpoint` and per-metric reporting all switched on. Two backlog
  items collapse into one.
- **D18's "~2.5 h" estimate for that re-run is wrong.** The measured figure is 13.1 h, about five
  times higher. That estimate is mine and it needs correcting in `decisions.md` before anyone
  schedules against it.

### 4.1 The LDO3 zero-shot arm is NOT cheap — correction

An earlier version of this document, and the verbal plan that came from it, said the LDO3 zero-shot
quantiles were "a `quant_out=` wiring plus a forward-pass rescore from the 15 saved trunks, no
training at all". **That is wrong, and the error is mine.** It was found by opening a checkpoint
instead of assuming what was in it:

```
results/lodo/encoder_ldo3__influenza__seed42__ckpt.pt
  top keys : ['encoder', 'adapter', 'meta']
  adapter  : ['gamma', 'beta', 'head.weight', 'head.bias']     <- the HELD-OUT adapter
```

The zero-shot arm does not use that adapter. It uses `borrowed = _mean_adapter(in_ads)`, the mean of
the in-disease heads, and `in_ads` lives in `_fit_trunk`'s return value and **was never written to
disk**. So the 15 LDO3 checkpoints can re-score the *adapted* arm — which already has quantiles and
therefore needs nothing — and cannot re-score the zero-shot arm at any price short of retraining the
trunk, which is what regenerates `in_ads`.

**Real cost: 15 LDO3 trunk retrains.** D17 measured one full-schedule trunk at ~2.8 h. This is a
backlog item with a price tag, not a same-night item, and it is the reason the row moved out of the
numbered queue in §0.

**Both halves of the fix are in, so this cannot recur.** `train/lodo.py` now passes `quant_out=` to
the zero-shot `_test_dataset` call and writes `{prefix}_zeroshot__{ds}__seed{S}__quantiles.npz`, and
`write_checkpoint` now carries `zeroshot_adapter=borrowed.state_dict()` alongside the trunk and the
held-out head. Any future LDO3 fold archives its zero-shot quantiles inline, and any future
checkpoint can reconstruct both arms. No rescore script was written, deliberately: there is no
artifact on disk it could run against, so it would be code built for a case that does not exist.

**What this costs us in the meantime.** `LDO3_Results.md` §5 already records the development-fold
zero-shot arm as having no UQ block. That stays true until someone pays the 15 retrains. It does not
touch the headline: the Ebola zero-shot arm is a different code path and does archive quantiles (§6).

---

## 5. `conformal.py`: built, self-checked, one defect found and fixed

New module at the repo root, alongside `score.py`. Stdlib and numpy only, no GPU, no new dependency.

**What it implements.**

```
score      E = max(q_lo - y, y - q_hi) / (q_hi - q_lo + eps)
fit        lambda_h = conformal (1-alpha) quantile of E over the calibration panels
apply      [q_lo - lambda_h * w, q_hi + lambda_h * w],  w = q_hi - q_lo
adapt      alpha_{t+1} = alpha_t + gamma * (alpha - f_t), lambda re-read at level 1 - alpha_t
```

Three design decisions worth recording.

- **The conformal index is the `ceil((n+1)(1-alpha))` order statistic, not a plain percentile.** A
  plain percentile is anti-conservative at small n, and small n is the entire problem. When the
  index exceeds n the function returns `+inf`, an honestly infinite interval rather than a silently
  truncated one.
- **Leave-one-panel-out is reported beside the in-sample fit.** Fitting on all five panels and
  scoring the same five is in-sample and flatters itself. Ebola is a sixth panel the calibration
  never saw, so LOPO is the number that estimates what Ebola gets. The in-sample row is what the
  frozen artefact contains; the LOPO row is what it is worth.
- **Infinite intervals are counted, not hidden.** When ACI drives `alpha_t <= 0` the interval is
  infinite by definition. That is ACI behaving as specified, but it makes a mean width meaningless,
  so the fraction of such origins is reported in its own column.

**Self-checks: eight, all passing, each paired with a control that must fail the other way.** Score
sign and scale-freedom; the conformal index at n = 16, n = 9 and n = 8; the wrapper and its
`lambda = 0` identity control; end-to-end calibration of a synthetic 60% interval to 90% on held-out
draws; ACI direction (misses widen, hits tighten); the fraction update landing midway between
all-hit and all-miss, which a binary-indicator update would fail; the published T = 18 bound; and
panel balance.

**The defect the first fit exposed: dengue dominates the pool.** At h3 dengue contributes 2,045,685
of 2,121,040 pooled cells, or 96.4%, because it has 7,165 nodes over 630 origins against covid's 49
over 37. An unweighted pooled fit is therefore a dengue fit wearing a cross-disease label. It failed
exactly where it mattered: under leave-one-panel-out, dengue's own lambda fell to 0.079-0.195, static
coverage reached only 0.725-0.778, and ACI could recover only by opening the intervals to infinite
width. It also breaches the project's own standing reporting rule from the Week-4 work order, which
is never to average across datasets.

**The fix, applied:** every panel carries equal mass in the pool via a weighted conformal quantile,
with the `(n_eff + 1) / n_eff` inflation as the weighted analogue of the `+1` order statistic and
`n_eff = (sum w)^2 / sum(w^2)`. A dedicated self-check now pins this: on a pool of one 100,000-cell
panel and one 1,000-cell panel, the unweighted quantile follows the large panel and the weighted one
reaches the small panel's scores.

Two self-check failures earlier in the session are recorded for the same reason as the pooling
defect: an ACI fixture that passed for the wrong reason because n = 3 made lambda infinite so nothing
could miss, and an `n_eff` assertion that misread effective cells as effective panels. Both were test
errors, not module errors, and both are fixed. The superseded unweighted lambda values
(1.533 / 1.790 / 2.552 / 3.400) are recorded here only so nobody quotes them by accident.

### 5.1 The balanced fit: it works, and ACI is the part that does the work

Run 2026-08-10, `python -m conformal --fit --lopo`, log at `results/reports/conformal_fit.log`.
Frozen at `lambda = {3: 0.1945, 5: 0.1809, 10: 0.4144, 15: 0.4584}`, gamma 0.05, alpha_1 0.10,
digest `8e2b657d...`.

| arm | raw coverage | + lambda | + lambda + ACI |
|---|---|---|---|
| in-sample (fit on 5, scored on 5) | 0.492 .. 0.918 | 0.759 .. 0.956 | **0.887 .. 0.927** |
| leave-one-panel-out (fit on 4, scored on the 5th) | 0.492 .. 0.918 | 0.731 .. 0.962 | **0.889 .. 0.931** |
| mean abs deviation from 0.90 | 0.124 | 0.055 / 0.066 | **0.012 / 0.012** |

**The target is met.** 0.49-0.92 goes to ~0.90 per horizon, and the horizon-dependence is gone:
covid h15, the worst cell in the project at 0.492, lands at 0.915 under LOPO.

**LOPO equals in-sample, and that is the finding that matters for Ebola.** Both arms reach a mean
absolute deviation of 0.012. Calibration fitted on four panels transfers to a fifth the calibration
never saw, which is the exact shape of what Ebola asks for. It is the strongest evidence available
that the transfer of the correction is sound, and it cost no GPU.

**Static lambda alone is NOT enough, and that vindicates the design rather than undermining it.**
The `+lambda` column still spans 0.731 to 0.962: dengue stays under-covered at 0.731-0.783 and
influenza over-covers at 0.93-0.96. ACI closes both. So the headline is "cross-disease initialisation
**plus** online adaptation", never lambda alone — which is precisely the pre-registered claim in §2,
that the guarantee comes from ACI's adaptation and not from the initialisation. That was an argument
this morning and is a measurement now.

**The cost, stated because coverage alone is free.** Intervals get wider: covid h10 goes 35,170 ->
76,015 in count space (2.2x), influenza_us-states h15 goes 344.8 -> 484.6 (1.4x), dengue h3 goes
91.0 -> 110.0 under static lambda. Sharpness is what buys the coverage and it belongs in the same
table, not a footnote.

**Two caveats that must travel with these numbers.**

- **Dengue LOPO has infinite intervals on 1.5-2.6% of origins.** ACI drove `alpha_t <= 0` there, so
  those origins cover trivially and dengue's LOPO ACI coverage is flattered by roughly that much.
  Its mean ACI width is `inf` and is not reportable. Every other cell in both tables is 0.0%.
- **The development panels have 47-630 origins; Ebola has 18.** ACI has far more room to adapt here
  than it will there, so **the dev ACI rows are optimistic as an Ebola forecast**. The T = 18 worst
  case remains 0.167 and still belongs in the paper.

### 5.2 `--apply`: the frozen wrapper over the Ebola archives

Added so the Ebola half of G4 is not blocked on writing code after the scored run exists.

`--apply` reads `{prefix}[_zeroshot]__{arm}__seed{S}__quantiles.npz` through `rpath`, so it finds
the scored run in `results/ebola/` and a dry run in `results/misc/` without being told which. It
consumes the archive and the bundle's own truth, touches no model and no point forecast, and prints
the raw interval beside the calibrated one — the three constraints `Ebola_Prereg.md` §5b binds the
calibration layer to. Both arms and both regimes (few-shot, zero-shot) are reported, with the
`T`-step ACI bound and the no-finite-sample-guarantee disclosure printed at the foot of every run so
neither can be dropped by whoever copies the table.

`load_frozen()` re-derives the config's sha256 from its own contents and refuses a file whose digest
does not match. Freezing gamma, alpha_1 and lambda_h is the thing that keeps Ebola a single-shot
score; a config that could be edited afterwards without anyone noticing would void that quietly.

---

## 6. The Ebola pre-registration was one edit away from a permanent loss — closed

This is the only item here that had a deadline measured in "before the next command runs", and it is
done.

**The defect.** `train/ebola.py` wrote records, per-node and per-origin for the zero-shot arm but
never called `write_quantiles`; the few-shot arm did. Collide that with `Ebola_Prereg.md` §5.1, "each
arm is scored exactly once", and §5a, "nothing here may be added once scoring has run", and the
consequence is not a gap but an amputation: **the moment the scored run started, the zero-shot arm
could never have WIS, CRPS, coverage, PIT or any calibration layer, ever**, because the only remedy
is a re-score the protocol forbids. `LDO3_Results.md` §5 records this same omission already costing
the development-fold zero-shot arm its entire UQ block. Repeating it on the run that cannot be redone
would have been the expensive version of a mistake we had already made cheaply.

**Fixed.** `run_arm` now passes `quant_out=zquant` and writes
`encoder_ebola_zeroshot__{arm}__seed{S}__quantiles.npz`. Verified that `train.lodo._score` fills the
dict in place, and that the new filename routes to `results/ebola/` while its dry-run twin routes to
`results/misc/`. Both cases are now pinned in `results_paths._demo`, which passes at 38 families
(was 36) - the routing was already correct, but an artifact this run produces exactly once should not
depend on a rule nobody tests.

**Amendment A7 and new §5b, registered before scoring, not after.** G4 is REQUIRED and its method is
not final, so §5b registers the *mechanism* and leaves the *variant* open under four binding
constraints: post-hoc on the archives only, never touching a point forecast or E1-E5; calibrated
intervals reported **beside** the raw ones rather than instead of them; variant and hyper-parameters
chosen on development folds and frozen before Ebola is touched; and any query-truth consumption
declared explicitly.

**The trade-off in that choice, stated rather than buried.** Naming ACI outright would be a stronger
pre-registration, and a reviewer can fairly call §5b permissive. The other side is that §5a makes the
registration uncorrectable after scoring, and the ACI-versus-EnbPI question is still out for review -
so naming a variant today risks registering one we do not use. Of two imperfect options this takes
the one whose failure mode is "slightly weak" rather than "permanently wrong". If the variant is
confirmed before the run starts, §5b should be tightened to name it.

### 6.1 Verified end to end, 2026-08-10

`PYTHONNOUSERSITE=1 conda run -n ebola-train python -m train.ebola --dry-run` — 400 trunk steps, seed
42, both arms, prefix `encoder_ebola_smoke` to `results/misc/`. **2.1 min, clean.**

**The fix landed.** Four quantile archives on disk, and the two that matter are the zero-shot pair
that did not exist before today:

```
encoder_ebola_smoke__ebola_L12__seed42__quantiles.npz            80,152 B
encoder_ebola_smoke__ebola_L20__seed42__quantiles.npz            80,993 B
encoder_ebola_smoke_zeroshot__ebola_L12__seed42__quantiles.npz   78,743 B   <- was never written
encoder_ebola_smoke_zeroshot__ebola_L20__seed42__quantiles.npz   79,124 B   <- was never written
```

**Read back end to end, not just listed.** `python -m conformal --apply --prefix encoder_ebola_smoke
--seeds 42` consumed all four, resolved `results/misc/` through `rpath` unprompted, and produced the
full 16-row table (2 arms x 2 regimes x 4 horizons) with the ACI bound and the guarantee disclosure
printed beneath it. That is the whole G4 chain proven on throwaway artifacts before the run that
cannot be redone. **The coverage numbers from it are meaningless** — a 400-step trunk against the
stale unweighted lambda — and they are not recorded here for that reason. The plumbing is what was
being tested, and the plumbing works.

**Four independent corroborations of the frozen arms fell out of the run.**

- Both arms verified against their registered content hashes before anything else ran.
- Support cells: **59** for L12, **113** for L20 — exactly the counts `freeze_ebola_arms.py` asserts.
- Scored origins: **18** on both arms, matching the geometry recomputed in §3.
- Cells per horizon came back **757 / 766 / 765 / 642** at h3/h5/h10/h15, identical to the
  independent count in `verify_geom.py`. Two code paths that share nothing agree.

**The E4 guard fired and passed, which is the interesting one.** On L12 the run reported h15 as
having zero support targets and the head block reducing to a uniform shrink, `c = 0.999990642` with
relative residual `1.4e-06` — decay only, no label signal. That is D16's `48/38/18/0` showing up as a
runtime assertion rather than a documented claim: at h15 the primary arm genuinely has nothing to fit
on, and the adapter is not quietly inventing a fit.

**One honest limit on what the dry run proves.** The trunk was *loaded* from a cached
`encoder_ebola_smoke__alldev__seed42__ckpt.pt`, not trained. The run therefore exercised adapter
fitting, scoring, archiving and routing, but not trunk training. Trunk training is the part every
other run in the project already exercises daily, so this is a small residual, but it is not zero.

---

## 7. HeatGNN: 51 finished runs sitting unscored — PARKED on instruction

**Status 2026-08-10: parked, not closed.** Deferred by explicit instruction while the Ebola and
conformal work holds the queue. Nothing below has changed; it is recorded so the item is picked up
deliberately rather than rediscovered.

Found while pricing the queue. `baselines/_preds/` holds all **60** HeatGNN prediction files - 3
influenza panels x 4 horizons x 5 seeds, complete. `results/baselines/` holds **9**.

The scored files stop mid-sequence at `us-regions__h5__seed72`: seed 82 of that cell is absent, and
japan and us-states were never reached. That is the signature of a scoring pass that died, not of a
filter or a deliberate subset.

This matters out of proportion to its cost. The baseline head-to-head is G6, and the client's
standing position in `Review Doc.md` is that it is the Week-3 gate - *"nothing gets signed off
without it"*. Closing it is one `score_baseline.py` run on CPU, minutes, with every prediction
already on disk. It is the highest value-per-minute item currently available and it does not touch
the GPU, so it costs the Ebola run nothing.

Worth checking whether this is the D8 CSV skip trap recurring rather than a plain crash; if it is,
the fix belongs in the scorer, not in a rerun.

---

## 8. Next

Ordered by what unblocks the most per unit of the bottleneck resource. There is one GPU, so anything
that does not need it should run **beside** the Ebola job, not behind it.

**Done since the last revision:**

- ~~`train.ebola --dry-run`~~ — **done, clean, §6.1.** The gate on the irreversible run is cleared.
- ~~`train/lodo.py` zero-shot quantile wiring~~ — **done**, plus the checkpoint fix that makes future
  folds rescorable (§4.1).
- ~~`conformal --apply`~~ — **done**, digest-checked, proven against the dry-run archives (§5.2).
- ~~`python -m conformal --fit --lopo`~~ — **done, §5.1.** Config regenerated and re-hashed
  (`8e2b657d...`); coverage 0.49-0.92 -> 0.889-0.931 under LOPO. `--apply` output is now quotable.
- ~~Launch the overnight queue~~ — **running.** `run_tonight.py` started `train.ebola --all` at
  20:09:53, then 15 influenza ceiling cells, then 5 dengue.

- ~~Commit the working tree~~ — **done**, 7 commits on `local`, tree clean, nothing pushed (§9).

**Running now, unattended, in this order (§9):**

1. `train.ebola --all`, both arms, all 5 seeds. `get_trunk` caches per seed and both arms share it,
   so a kill costs one trunk rather than five.
2. Three influenza ceilings with quantile archiving and checkpoint saving, ~35 min, 15 cells.
3. Dengue ceiling, 12.7 h, 5 cells. Last because it is 96% of that job's cost and therefore the
   thing worth losing if the night runs out.

**After the Ebola run exists:**

4. `python -m conformal --apply` on `encoder_ebola`. The Ebola half of G4, now a single command, and
   the config it reads is already fitted and frozen.

**CPU lane, no contention with any of the above:**

5. Gate figure (work order 8e) - the `*__gate.npz` files are already on disk and the client has
   already asked for it twice.
6. G5 explainability scoping note - a document, not code.

**GPU, last:**

7. ANIL, both arms, once items 4-7 of §1.3 are settled - item 4 is a client reply, not a code change.

**Backlog, priced, not scheduled:**

8. LDO3 zero-shot quantiles: 15 trunk retrains at ~2.8 h each (§4.1). Code is already in place so the
   next run of that family archives them inline; this item is only about the existing 15.
9. HeatGNN rescore (§7), parked on instruction.

**Record-keeping, any time:**

10. Correct the dengue coverage label and D18's ~2.5 h estimate in `decisions.md`. Both corrections
    are stated in §3 and §4 of this document but have not yet been carried into the decision log.
11. Send the client the meta-learning scope answer. `MAML_Decision.md` §5 put options A-D to them on
    2026-07-31 and no reply is recorded. Proceeding quietly with the reduced version is the specific
    thing `Review Doc.md` asked us not to do, and that holds whether ANIL runs or not. Tell them
    option D got cheaper (§1.3).

---

## 9. What shipped, and the queue that is running it

### 9.1 `run_tonight.py`

One GPU, so the GPU work is strictly serial and the order encodes what is worth losing. Ebola first
because it is the REQUIRED case study and the only irreversible step; the three influenza ceilings
next at ~35 min for four of five reference panels; dengue last because it alone is 12.7 h. The
conformal fit needs no GPU, so the script launches it detached rather than queued behind Ebola - the
only genuine parallelism available tonight, and it was already running separately when the queue was
started, hence `--skip-conformal`.

Two pieces of logic in it can silently do the wrong thing, so both are guarded and both are pinned by
a self-check with a control that must fail the other way.

- **The one-way door.** It refuses to start the Ebola step if any `encoder_ebola__*__seed*.json`
  already sits in `results/ebola/`. `Ebola_Prereg.md` §5.1 allows exactly one scored run, and a second
  would overwrite the first with nobody seeing it happen. `--force-ebola` is the only way past. The
  control case asserts the `encoder_ebola_smoke` dry-run prefix does **not** trip it.
- **Resume keys on the artifact, not a flag.** A ceiling cell counts as done only when its
  `__quantiles.npz` exists, because that archive is the entire point of the re-train. The control
  asserts that a `.json` and a `__pernode.npz` do **not** mark a cell finished: those already exist
  for all four panels, so keying on them would report the whole job complete and archive nothing.

### 9.2 The seven commits

Branch `local`, tree clean, **nothing pushed**.

| commit | what |
|---|---|
| `e9464ed` | Ebola zero-shot quantile archiving, prereg A7 + §5b |
| `38dc480` | LDO3 zero-shot quantiles, checkpoint keeps the borrowed adapter |
| `8d60649` | conformal wrapper: cross-disease lambda_h + online ACI, fitted and frozen |
| `8b1461f` | the overnight queue |
| `2faa75e` | ANIL built, self-checked, parked |
| `550fd05` | D17-D19 and the three diagnostics they rest on |
| `e956b71` | this document |

### 9.3 The queue finished clean, and what is still NOT in this document

**Finished 2026-08-11 12:59:27, zero non-zero exit codes across every step.** Dengue seed 82, the
last cell, took 143.0 min, in line with the 135.9-169.6 min per-seed figures §4 priced it from.

| artifact | state |
|---|---|
| Ebola scored records | **20 JSON** (2 arms x 5 seeds x few-shot/zero-shot) |
| Ebola quantile archives | **20**, so the zero-shot arm has its UQ material. §6 held. |
| single-disease ceiling quantiles | **25 of 25** (5 panels x 5 seeds). Was 5, covid only. |
| ceiling cells still missing quantiles | **0** |

**The G4 reference comparison is unblocked.** The four panels §4 was written about now have quantile
archives, so our calibration can finally be compared against the single-disease ceiling on all five
rather than on covid alone. The dengue 12.7 h block was the price and it has been paid.

The numbers those artifacts contain are now read and are in **§10**.

---

## 10. First read of the results

**These are first reads, not a write-up.** Every number below came off disk through a reader built
today, aggregated over the 5 seeds, and none of it has been interrogated. Nothing here has a
significance test attached, and §10.4 lists what is still missing before any of it can be reported.

Both arms carry their pre-registered identity in every record: L12 is 59 support cells to
2014-06-28 under prereg `08d657dc...`, L20 is 113 cells to 2014-08-23 under `e9b9ac0b...`. The
reader refuses to print if the 20 files disagree on that, and they do not. Scored once, as registered.

### 10.1 Point forecast, G3 (RMSE, `country_macro`, mean +- sd over 5 seeds)

Read with `ebola_report.py`. Skill columns are against the deterministic floors in
`results/naive/naive__ebola_L{12,20}.json`, and a cell whose seed spread straddles the floor is
printed as "within noise" rather than as a number.

| arm | regime | h3 | h5 | h10 | h15 |
|---|---|---|---|---|---|
| L12 | few-shot | 38.20 +- 0.49 | 37.98 +- 0.59 | 41.19 +- 4.33 | 28.52 +- 0.48 |
| L12 | **zero-shot** | **36.55 +- 0.46** | **37.59 +- 0.18** | **38.87 +- 0.14** | **28.28 +- 0.11** |
| L20 | few-shot | 47.03 +- 15.24 | 38.92 +- 0.74 | 51.58 +- 13.25 | 40.85 +- 10.70 |
| L20 | **zero-shot** | **35.24 +- 0.51** | **36.69 +- 0.22** | **39.37 +- 0.39** | **29.54 +- 0.31** |

Skill over persistence, zero-shot arms: L12 +15.8 / +15.2 / +15.5 / +44.2%, L20 +18.8 / +17.3 /
+14.4 / +41.7%. Over `support_mean` the margins are much thinner, +4.5 to +15.3%. The few-shot arms
beat persistence at h3 and h5 on L12 (+12.0, +14.3%) and read **within noise** on three of four L20
cells and on L12 h10.

**The headline finding is an inversion: zero-shot beats few-shot on 7 of 8 cells.** Fitting the
adapter on the support set makes the forecast worse, and the effect is largest where there is most
support data. L20 has 113 support cells against L12's 59, and its few-shot arm is the worst and by
far the least stable thing in the table: sd 15.24, 13.25 and 10.70 against zero-shot's 0.22 to 0.51.
Three of its four cells only read "within noise" because that variance is so large, which is a
statement about instability, not about being close to the floor.

This is consistent with the transfer result already settled on the development folds, where
cross-disease transfer is negative and adapter fitting does not repair it. It is now the result of
the pre-registered case study rather than a development-fold observation, and it is the paper's
headline whether or not it is the one that was wanted.

### 10.2 Calibration on Ebola, G4 (`conformal --apply`, empirical 90% coverage)

| arm | regime | h | raw cov | +ACI cov | raw w | ACI w |
|---|---|---|---|---|---|---|
| L12 | few-shot | 3 | 0.491 +- 0.062 | 0.862 +- 0.028 | 24.8 | 45.0 |
| L12 | few-shot | 5 | 0.437 +- 0.045 | 0.849 +- 0.020 | 22.2 | 48.0 |
| L12 | few-shot | 10 | 0.584 +- 0.105 | 0.941 +- 0.025 | 44.5 | 73.5 |
| L12 | few-shot | 15 | **0.275 +- 0.103** | **0.811 +- 0.016** | 6.5 | 32.7 |
| L12 | zero-shot | 3 | 0.460 +- 0.025 | 0.864 +- 0.008 | 20.5 | 36.5 |
| L12 | zero-shot | 5 | 0.421 +- 0.020 | 0.870 +- 0.008 | 20.0 | 35.9 |
| L12 | zero-shot | 10 | 0.396 +- 0.028 | 0.904 +- 0.009 | 22.1 | 44.1 |
| L12 | zero-shot | 15 | 0.363 +- 0.021 | 0.920 +- 0.007 | 23.1 | 43.1 |
| L20 | few-shot | 3 | 0.701 +- 0.112 | 0.943 +- 0.040 | 107.9 | 132.3 |
| L20 | few-shot | 5 | 0.677 +- 0.141 | 0.903 +- 0.029 | 98.1 | 138.5 |
| L20 | few-shot | 10 | 0.692 +- 0.161 | 0.977 +- 0.015 | 139.4 | 203.3 |
| L20 | few-shot | 15 | 0.670 +- 0.211 | 0.976 +- 0.012 | 108.2 | 171.9 |
| L20 | zero-shot | 3 | 0.541 +- 0.042 | 0.925 +- 0.011 | 36.9 | 48.4 |
| L20 | zero-shot | 5 | 0.480 +- 0.019 | 0.918 +- 0.006 | 38.9 | 52.6 |
| L20 | zero-shot | 10 | 0.413 +- 0.024 | 0.959 +- 0.005 | 50.5 | 77.6 |
| L20 | zero-shot | 15 | 0.375 +- 0.019 | 0.963 +- 0.005 | 55.9 | 91.3 |

Mean absolute deviation from 0.90: **L12 0.039, L20 0.045, few-shot 0.052, zero-shot 0.032.**

- **Raw Ebola coverage is far worse than any development panel**, 0.275 to 0.701 against the dev
  panels' 0.492 to 0.918. Domain shift onto a genuinely unseen disease costs more than any dev fold
  showed. That is the justification for the wrapper, measured on the target rather than argued.
- **The wrapper closes most of it but not all, and misses in opposite directions per arm.** L12
  under-covers, L20 over-covers. Neither reaches the 0.012 the dev panels did. **This was pre-stated
  in §5.1**: the dev panels have 47-630 origins and Ebola has 18, so the dev ACI rows were labelled
  optimistic before these numbers existed. The prediction held.
- **Every deviation sits inside the published T = 18 bound of 0.167.** The worst cell, L12 few-shot
  h15 at 0.811, is 0.089 off nominal. The bound is binding rather than decorative, and it was not
  breached.
- **L12 few-shot h15 is the worst cell and it is exactly the one D19 flagged.** Raw 0.275, calibrated
  0.811, the only cell still clearly under nominal. That is the horizon where the primary arm has
  **zero adaptation pairs** (D16: 48/38/18/0), which the E4 guard confirmed at runtime in §6.1. The
  adapter is fitted on nothing there and calibration cannot fully rescue it.
- **Zero-shot is better calibrated and far more stable**, 0.032 against 0.052, with seed sd of
  0.005-0.011 against few-shot's 0.012-0.040. Point accuracy (§10.1) and interval calibration now say
  the same thing about few-shot adaptation, by two independent routes.
- **`inf%` is 0.0 on all 80 rows.** The infinite-interval mode that hit dengue under LOPO does not
  occur here.

### 10.3 The single-disease ceiling is itself badly calibrated (`conformal --reference`)

The comparison the 12.7 h dengue block was paid for. The ceiling is a model trained on the panel's
own disease, so it carries no domain shift at all. It is **not** wrapped: calibrating the reference
would make it a different reference. Both arms are asserted to share the same origins per
(panel, seed), and none mismatched.

| | coverage range | mean abs dev from 0.90 |
|---|---|---|
| single-disease ceiling, raw | 0.531 .. 0.934 | **0.158** |
| calibrated transfer | 0.887 .. 0.927 | **0.012** |

Dengue degrades 0.756 to 0.624 across horizons, covid reaches 0.531 at h15, and only
influenza_us-regions sits near nominal (0.904-0.934). **A model with zero domain shift is
mis-calibrated by 0.158 on average.**

**This reframes the calibration story, and the reframing is the finding.** G4 was scoped as fixing
the transfer arm's under-coverage. It is actually that the five-quantile head is uncalibrated in
general and domain shift was never its main cause. Read the direction carefully: calibrated transfer
landing closer to nominal than the ceiling does **not** mean transfer forecasts better. It means the
ceiling's intervals are also wrong, and the ceiling is a weaker reference than the word implies.

Width is the cost and it is large: covid h10 is 25,394 for the ceiling against 78,288 calibrated
(~3x). The dengue h3 figure (52.0 against 944.7) is inflated by the infinite-interval origins
recorded in §5.1 and **should not be quoted as a clean comparison**.

### 10.4 What is still missing before any of this is reportable

- **No significance testing.** The zero-shot versus few-shot inversion in §10.1 is a comparison of
  means with no seed-paired interval attached. The arms share a trunk per seed, so a paired test is
  both available and the correct one. Until it is run, "zero-shot beats few-shot" is an observation.
- **No bootstrap CIs.** `__pernode.npz` and `__perorigin.npz` were written for all 20 records and
  neither has been touched. Those are the material for the per-node and over-origin intervals the
  standing rules ask for.
- **WIS, CRPS and PIT are not computed for Ebola.** They are implemented and self-checked in
  `score.py`, and the quantile archives now exist, but `--apply` reports coverage and width only. G4
  names the full metric set, so this is a genuine remaining gap rather than a nicety.
- **The L20 few-shot instability is unexplained.** sd of 15.24 on a mean of 47.03 is not a result, it
  is a symptom. More support data producing a worse and wildly less stable fit wants a cause before
  anyone writes a sentence about what few-shot adaptation does.
- **`--apply` prints per-seed rows and no aggregate.** The §10.2 table was produced with a throwaway
  script. Copied straight from the tool, the output would breach the project's own dispersion rule,
  so the aggregation belongs in the tool.

---

## 11. 2026-08-17 — the ceiling got stronger, and the transfer verdict did not move

Three things landed today, all read-only or cheap, none of them a retrain. Two close audit findings;
the third changes how the transfer result can be defended.

### 11.1 Seed ensembling the single-disease ceiling — free, and it was owed

`diagnostics/seed_ensemble.py`. The mean over the five seeds of the count-space median forecast,
rebuilt from the `__quantiles.npz` archives `write_quantiles` already wrote for all 25 ceiling runs.
**No GPU, no retraining.** This is the one accuracy lever the client's work order permits by name
("seed ensembling — take it, it's free").

For squared error `MSE(mean of K) = mean(MSE) − across-seed variance`, so a gain is arithmetic
whenever seeds disagree. The size is the only open question, and dengue h3 had a per-seed sd of 6.04
on a mean of 42.06 — real variance to collect.

| cell | per-seed mean | ensemble | vs best floor, before → after |
|---|---|---|---|
| dengue h3 RMSE | 42.06 ± 6.04 | **37.28** (+11.4%) | −33.4% → **−18.2%** vs persistence |
| dengue h5 RMSE | 49.94 ± 6.05 | **44.87** (+10.2%) | −14.0% → **−2.4%** vs persistence |
| influenza_us-regions h10 RMSE | 787.81 ± 40.5 | **736.95** (+6.5%) | −9.2% → **−2.2%** vs seasonal |
| influenza_us-regions h15 RMSE | 812.80 ± 68.1 | **760.99** (+6.4%) | −13.6% → **−6.4%** vs seasonal |

Three cells flipped that nobody asked for: `influenza_us-states` now clears its floor at **all four
horizons** on both RMSE and MAE, and dengue h10/h15 now **beat** train_mean (+3.7%/+0.1% RMSE,
+14.4%/+10.8% MAE). Across the grid the ceiling goes from **6 of 20 to 8 of 20** cells beating their
best naive floor on RMSE, and 7 to 8 on MAE.

**The control that makes this quotable.** `--selfcheck` rebuilds ONE seed from its own archive and
rescores it through `score_predictions`, and requires it to reproduce that seed's released JSON:
**28 cells to 1e-6**, plus a mutation asserting the median quantile index is distinguishable on the
selfcheck panel. A wrong quantile index or a shifted origin axis would have moved every ensemble
cell in the same direction and looked entirely plausible.

**`encoder_mc` cannot be ensembled this way.** The mean-correction is added in MODEL space before
the scaler is inverted (`train/loop.py:222`) and the archives hold post-inversion counts only. The
ensemble is the median arm and is comparable to `encoder`, never to `encoder_mc`.

### 11.2 dengue h3: concentrated, not uniform — and the aggregation is doing half the work

The residual −18.2% is not spread across the panel. Of 6,161 scored nodes, the **worst 1% (61 nodes)
hold 43.2% of the excess error**; the worst 5% hold 70.5%, the worst 25% hold 96.9%.

By country, encoder vs persistence RMSE at h3 on the shared node set: bolivia (9 nodes) **+105.9%**,
nicaragua (17) +57.5%, mexico (32) +30.4%, brazil (5,189) +21.5% — against colombia (718) **−8.6%**
and dominican republic (32) **−16.4%**, where the encoder wins.

And the estimand matters: on identical nodes, **country-macro is 49.98 vs 31.53 (−58.5%) while
node-mean is 23.05 vs 18.83 (−22%)**. The equal-weight country macro hands 1/12 of the score to a
9-node country. That is audit finding M9 live in the headline.

**The aggregation was NOT changed.** Introducing a minimum-node floor after seeing the result would
move our own number and is indefensible however good the reasoning; it goes in the paper as a stated
limitation with both columns shown. Japan (−32% to −56% vs seasonal) and COVID (−16% to −134%) are
disclosed, not fixed — the Japan gap is annual periodicity the w=20 window cannot reach, and w>=53
is blocked by both the receptive field (32) and the Ebola case study's 38-week span; COVID is the
Omicron fold boundary already recorded in `LDO3_Results.md` section 1.

### 11.3 The symmetric ensemble — the transfer verdict survives a stronger opponent

A stronger ceiling makes transfer look worse, because every transfer number is a ratio against it.
That is the honest consequence and it points the same way D18 already did. But ensembling **only**
the ceiling would be the Week-3 reference mismatch pointing the other way — our best single-disease
system against a single transfer run — so both arms get the same treatment or neither does.

`--vs-transfer`: ensembled ceiling against the ensembled `encoder_ldo3` adapted arm, same datasets,
same origins (asserted), and **one shared bootstrap draw applied to both arms** so the pairing is
real and origin-to-origin difficulty cancels.

**Result over the 36 attributable cells: 1 better, 10 within noise, 25 worse.**
The per-seed table in `LDO3_Results.md` says **1 / 10 / 25**. The horizon gradient matches cell for
cell:

| horizon | transfer better | within noise | transfer worse |
|---|---|---|---|
| h3 | 1 | 5 | 4 |
| h5 | 0 | 4 | 6 |
| h10 | 0 | 1 | 7 |
| h15 | 0 | 0 | 8 |

Identical to the published table. **The conclusion is invariant to a change of estimator (single
runs to ensembles), a change of significance instrument (seed-paired t at n=5 to paired origin
bootstrap at B=10,000), and a materially stronger reference.** Individual cells moved — japan h3
went from a loss to within noise (+9.8%), us-states h3 became a genuine transfer win (+3.5%, CI
[+1.0, +7.0]) — but no verdict count changed. This is worth more to the paper than the original
table was, because it pre-empts "you beat a weak baseline".

Three limits travel with it:

- **The seed axis is consumed.** An ensemble is one value, so the seed-paired t-interval cannot be
  computed on it. This is a SECOND table, not a replacement; the per-seed record remains the one
  carrying seed dispersion.
- **Point and interval are on different estimands and are kept in separate labelled columns.** The
  delta is node-averaged (the project headline); the CI is cell-pooled, because that is what the
  per-(origin, country) sufficient statistics support. Printing one beside the other as a single
  claim is defect M1 and is not done here.
- **Zero-shot cannot be treated this way at all** — that arm wrote **0** quantile archives, the
  omission `LDO3_Results.md` section 5 records. The symmetric treatment covers the adapted arm only,
  which is the optimistic bound the headline rests on. The zero-shot table stays single-run and
  labelled as such.

### 11.4 Gate-off ablation — built, not yet run (M10)

`ablation/run_gate_ablation.py` + `run_gate_night.py`. Across **13,177 records on disk `gate_mode`
is `learned`, null or absent and never once `off`** — so "the graph helps" is untested in both
directions. We have measured the gate's VALUE (g = 0.271 japan to 0.604 dengue, spatial contribution
0.40–0.64, `g<0.05` fraction 0.0% everywhere) but a gate being open is not the claim that it is
useful.

Identical trainer, identical seeds, identical 80 epochs; the only change is
`SharedEncoder(gate_mode="off")`, which forces g=0 and is already proven by
`tests/test_encoder_invariants.py` to nest the graph-free model exactly. Paired against the released
run at the same seed. ~13.5 h, dengue last (96% of the cost) so an interrupted night still leaves
four readable panels.

**Ceiling on the claim:** g=0 removes neighbour mixing but keeps the LTR degree feature, so this
bounds the value of NEIGHBOUR INFORMATION, not of the graph in total. A shuffled-adjacency arm is
the control that would separate "structure helps" from "any adjacency helps"; it is deliberately not
built, because if the graph does not help at all there is nothing left for a shuffle to distinguish.

### 11.5 Lag-h ACI — M5 closed in code

`aci_run()` now takes `lag`. The lag is not a tuned parameter: a forecast issued at origin *t* is
about week *t+h* and cannot be scored until *t+h*, so at step *k* only outcomes from origins
*j <= k-h* have landed. **lag = h, per horizon, forced by the definition of the horizon.**

`lag=1` is retained as the control and is asserted **bit-identical** to the pre-registered loop.
`--apply` now prints `+ACIlag` beside `+ACI`, with `upd` = how many origins ever adapted, and names
the gap between them as the size of the oracle.

At T=18 Ebola origins the honest stream adapts on **15/13/8/3 of 18** origins at h3/h5/h10/h15 — at
h15 fifteen of eighteen origins run at alpha_1 and ACI barely runs at all. That is the finding, not
a footnote: the old h15 lift from 0.464 to 0.794 was bought with up to 14 weeks of future truth.

The frozen wrapper config is **not** touched — its digest predates the first scored Ebola record and
that ordering is itself a claim.

### 11.6 Commits

`883314f` gate-off arm, lag-h ACI, seed ensemble off the archives ·
`bfba94c` symmetric ensemble, verdict does not move ·
this section.


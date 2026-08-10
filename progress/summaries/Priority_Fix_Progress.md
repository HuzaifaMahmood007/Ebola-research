# Priority Fix Progress

**2026-08-10.** Covers the ANIL verdict, the conformal decision packet, the verification of both
against the artifacts on disk, the state of `conformal.py`, and the pre-registration hardening that
had to close before the Ebola run started (§6) — now closed and proven end to end (§6.1).

**State at time of writing:** every code gate is cleared and the overnight queue is running.
`run_tonight.py` started `train.ebola --all` at **20:09:53 on 2026-08-10** and will follow it with
the three influenza ceilings and then dengue. The conformal wrapper is fitted, frozen and validated
(§5.1). Nothing in the queue is waiting on a line of code. What remains is compute, one client reply,
and the record-keeping in §8.

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

**Running now, in order:**

1. `train.ebola --all`, both arms, all 5 seeds. `get_trunk` caches per seed and both arms share it,
   so a kill costs one trunk rather than five.

**CPU lane, no contention:**

3. Gate figure (work order 8e) - the `*__gate.npz` files are already on disk and the client has
   already asked for it twice.
4. G5 explainability scoping note - a document, not code.

**GPU, after Ebola:**

5. Three influenza ceilings with quantile archiving and checkpoint saving, ~35 min - four of five
   reference panels for the price of one coffee.
6. Dengue ceiling, 12.7 h, scheduled separately. It is 96% of that job's cost.

**After the Ebola run exists:**

7. `python -m conformal --apply` on `encoder_ebola`, once step 2 has regenerated the config. This is
   the Ebola half of G4 and it is now a single command.

**GPU, last:**

8. ANIL, both arms, once items 4-7 of §1.3 are settled - item 4 is a client reply, not a code change.

**Backlog, priced, not scheduled:**

9. LDO3 zero-shot quantiles: 15 trunk retrains at ~2.8 h each (§4.1). Code is already in place so the
   next run of that family archives them inline; this item is only about the existing 15.
10. HeatGNN rescore (§7), parked on instruction.

**Record-keeping, any time:**

11. Correct the dengue coverage label and D18's ~2.5 h estimate in `decisions.md`.
12. Send the client the meta-learning scope answer. `MAML_Decision.md` §5 put options A-D to them on
    2026-07-31 and no reply is recorded. Proceeding quietly with the reduced version is the specific
    thing `Review Doc.md` asked us not to do, and that holds whether ANIL runs or not.
13. Commit the working tree: `conformal.py`, `train/anil.py` (parked), `train/ebola.py`,
    `train/lodo.py`, `results_paths.py`, `Ebola_Prereg.md`, `run_queue.py`, the three diagnostics
    files and D17-D19.

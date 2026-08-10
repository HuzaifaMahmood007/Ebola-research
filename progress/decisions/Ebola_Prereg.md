# Ebola pre-registration: support-set arms

**Frozen 2026-08-07, before either arm is scored.** Nothing in this document may be revised after
the first Ebola score is produced. If any of it turns out to be wrong, it stays as written and the
outcome is reported against it.

This closes the open item in `Reports/Ebola_Support_Set_Decision.md`, which stated the trade-off and
deliberately did not recommend.

---

## 1. The decision

Client, 2026-08-07:

> Ebola support set: 12 weeks primary, 20 weeks pre-registered as a labelled secondary arm, 19
> dropped. Both frozen and hashed before either is scored.

So there are two arms, not one. The 12-week arm carries the headline. The 20-week arm is a labelled
secondary result that is reported in full whatever it says, and is marked secondary in every table it
appears in. The 19-week option is dropped and is recorded in the manifest as considered and rejected,
so the arm set cannot later be read as the only thing we looked at.

A note on the labels. The sweep table the client read indexes outbreak weeks from the raw first week
2014-03-24, whose incidence cell is masked, because week 0 of a cumulative series carries no
increment. That label is the 0-based column index of the last support week in the built tensor, not a
column count. **The operative definition of each arm is its cutoff date.** The labels are kept only so
the arms match the document the decision was made from.

## 2. What is frozen

Built from the raw OCHA compilation in one pass by `freeze_ebola_arms.py`, which asserts every count
below before it is allowed to write. Manifest: `configs/ebola_arms.json`.

| artifact | sha256 (content digest) |
|---|---|
| `data/processed/ebola_L12.npz` (primary) | `08d657dcc3856fbd3319e509d828a9697eefb91d9a7b1f2b6abbdd38454178fd` |
| `data/processed/ebola_L20.npz` (secondary) | `e9b9ac0b44c2e3f71ba78a922a9125d1e0bc1b10e4e6547b3e8ce59855d28cc0` |
| `data/Final datasets/data-ebola-public.xlsx` (source) | `2d679a31a66f912da93fe0bd73da82a3c36b0d1143b1fcc298bc921d5f28f9f2` |

The content digest is taken over the arrays themselves, not the file bytes, because a `.npz` is a zip
and zip entries carry a wall-clock timestamp. The file-byte digest is recorded too, but the content
digest is the one that survives a rebuild. Re-check both at any time with:

```
conda run -n ebola python freeze_ebola_arms.py --verify
```

## 3. The two arms

| | **primary** | **secondary** | dropped |
|---|---|---|---|
| label | L12 | L20 | L19 |
| cutoff date (support is every observed cell on or before) | 2014-06-28 | 2014-08-23 | 2014-08-16 |
| support columns | 13 | 21 | 20 |
| support cells | 59 | 113 | 79 |
| support districts (of 61) | 18 | 36 | 22 |
| adaptation pairs h3 | 48 / 17d | 102 / 36d | |
| adaptation pairs h5 | 38 / 14d | 92 / 36d | |
| adaptation pairs h10 | 18 / 9d | 72 / 35d | |
| adaptation pairs h15 | **0 / 0d** | 54 / 34d | |

An adaptation pair at horizon h needs a support target at column t+h, so it exists only where a
support cell sits at a column index of at least h. Input windows are left-padded, so the origin
itself never limits the count.

**The primary arm has zero adaptation data at h15.** That is arithmetic, not a data-quality problem:
support reaches column 12 and h15 needs a target at column 15 or later.

### What is identical across the arms

| | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| **scored** query pairs (what the run evaluates) | **757** | **766** | **765** | **642** |
| scored districts | 57 | 57 | 59 | 58 |
| *reachable* query pairs (what the audit note quotes) | *1,151* | *1,075* | *866* | *642* |
| *reachable* districts | *61* | *61* | *59* | *58* |

Both rows are identical under both arms, and both are asserted by the freeze script. A full-window
query target sits at column 22 or later and support reaches at most column 20, so neither arm costs
a single forecast. The two arms are therefore directly comparable: same forecasts, different amount
of labelled adaptation data. The cost of the longer support set is to the claim, not to the
evaluation.

**The two rows are different quantities and only the first is the evaluation size** (amendment A1).
*Reachable* counts per-horizon origins: every origin with a full window whose target lands inside
the panel, so h3 reaches origin 48 and h15 only 36. *Scored* uses ONE common origin set for every
horizon, t in [19, 36], 18 origins, which is what `bundles.origins()` returns and therefore what
`score_predictions` actually scores. Every other dataset in this project was scored that way, and it
is the right protocol here too: horizons are comparable to each other only if read at the same
origins. The two agree at h15, whose reach is the binding constraint. Quoting the reachable numbers
as the evaluation size would overstate h3/h5/h10 by 34 to 52 per cent.

### Scaler consequence

The scaler is a pooled `log1p` plus z-score fit on the support cells alone, so each arm has its own.

| | mu | sd | implied typical week | query cells beyond 1 sd |
|---|---|---|---|---|
| primary L12 | 1.4402 | 1.2145 | 3.2 cases | 333 / 1,240 (27%) |
| secondary L20 | 1.7913 | 1.5286 | 5.0 cases | 192 / 1,186 (16%) |
| *(prior L7 build, for reference)* | 1.1310 | 1.1599 | 2.1 cases | 426 / 1,272 (33%) |

The mean scored cell is about 19 cases under either arm. Both arms are still mis-calibrated against
what they are scored on; the primary is less so than the L7 build we were on, and the secondary less
again. The FiLM adapter is the component meant to correct this, and under the primary arm it has 18
examples at h10 and none at h15.

---

## 4. Stated expectations, before anything is scored

Definitions used below, all fixed before the run (see the amendment log for what was added on
2026-08-07 after the code was read).

**The trunk.** One shared encoder trained jointly on all five development bundles: dengue,
influenza japan / us-regions / us-states, and covid us-states, with `adapter_groups=[0,1,1,1,2]` so
each *disease* gets one adaptation surface and the loss is rebalanced per disease rather than per
bundle. Nothing is held out, because Ebola is the held-out disease. 5 seeds (42, 52, 62, 72, 82),
91,000 steps, `trunk_patience=30`. No Ebola cell enters it.

**Zero-shot** means that trunk with the element-wise mean of the three trained in-disease adapters
applied to Ebola with no fitting, exactly the `_mean_adapter` reference used on every dev fold. No
Ebola label is touched at any point.

**Few-shot** means the same frozen trunk with one fresh Adapter (FiLM plus quantile head, 1,428
params) fit on that arm's support cells only.

**Adapter stopping rule** (amendment A2). Ebola has no validation split by construction, so the
epoch count is chosen by **leave-one-district-out cross-validation inside the support set**: hold
out one support district at a time, fit on the rest, record the held-district pinball per epoch,
take the epoch minimising the mean held-out curve, then refit on all support cells for that many
epochs. No query cell is read at any point. This is fixed here because with 1,428 parameters and 48
adaptation pairs at h3, "how long did you train the adapter" is otherwise a free parameter chosen
after seeing the answer.

**Short windows** (amendment A3). Adaptation origins sit at t < 19, so the input window is zero
left-padded to length 20 (`models/windows.py`, gate 10 in `tests/test_encoder_invariants.py`). Zero
is the per-node mean in normalised space, matching how every bundle encodes an unobserved cell. The
known weakness: sin_doy/cos_doy are zeroed too, which is not a point on the unit circle, so a padded
step carries an impossible calendar date, and at h10 on the primary arm a window is about 89 per
cent pad. That is the Day-13 protocol as recorded, and it is stated here rather than discovered
later.

**Reference floors.** `persistence` and `support_mean` on the same scored cells. `seasonal` is
dropped, not reported as a third floor: Ebola has T=52, so the t-52 lag is out of panel at every
scored origin and seasonal-naive would collapse to a duplicate of persistence.

These are predictions, and I expect to be reporting some of them as wrong.

**E1. Zero-shot loses to persistence at every horizon, on both arms.** Confidence: high. The LDO3
result is that cross-disease transfer is negative in 25 of 36 cells and that zero-shot fails on every
development fold. Ebola is a further shift than any fold in that run, so I have no basis for
expecting it to be the exception.

**E2. Few-shot beats zero-shot at h3 and h5, on both arms.** Confidence: moderate. This is the
weakest thing the adapter has to do, and h3/h5 are where it has the most examples.

**E3. Few-shot still does not beat persistence at any horizon on the primary arm.** Confidence:
moderate. If this is wrong it is the best outcome available here and I will say so plainly.

**E4. Primary arm, h15: the h15 block of the head must come out a uniform shrink of its
initialisation** (corrected twice, amendments A4 and A6). This is not a prediction, it is a
consequence of 0 adaptation pairs: with no h15 target in the support set the mask zeroes the h15
term of the pinball loss, so rows 15-19 of `head.weight` and `head.bias` receive a gradient that is
exactly zero. Adam's update with a zero gradient is exactly zero, so the only thing that moves the
block is AdamW's decoupled weight decay, which multiplies every element by the same
`(1 - lr_k * wd)` each step. The block must therefore satisfy `p_final = c * p_init` for a single
scalar `c` slightly below 1, to within float32 accumulation. Any departure from proportionality
means a label-driven gradient reached a horizon that has no labels, which is a bug in the adaptation
path, not a result.

**What I originally wrote here was wrong and is corrected before scoring.** The first version of
this document said few-shot and zero-shot h15 must be bit-identical *as forecasts*. They will not
be. `gamma` and `beta` are shared across all four horizons, so gradient from the h3/h5/h10 support
pairs moves the FiLM surface and therefore moves the h15 predictions, even though no h15 label
exists. Primary-arm h15 is therefore "no h15 supervision, FiLM fit on nearer horizons", which is
weaker than strict zero-shot and is labelled as such rather than as few-shot. h10 on the primary arm
has 18 pairs across 9 districts, which I do not consider a fitted adapter either; I expect no
separation from zero-shot beyond the noise floor.

**E5. The secondary arm beats the primary arm at h10 and h15.** Confidence: moderate. It has 72 and
54 adaptation pairs where the primary has 18 and 0. **If it does not, the failure is not
support-limited**, and no amount of extra Ebola labelling rescues the few-shot framing. That is the
single most informative cell in this exercise, and it is why the secondary arm is worth its compute
even though it weakens the headline from few-shot to moderate-data transfer.

**E6. Reporting.** Every number carries mean and standard deviation over the 5 frozen seeds (42, 52,
62, 72, 82) and a bootstrap confidence interval over districts and time origins. Any cell inside the
noise floor is written "within noise" and is given no direction. No averaging across diseases. The
comparison reference is stated on every delta table. h10 and h15 under the primary arm are labelled
zero-shot in every table, never few-shot. Interval metrics (WIS, CRPS, coverage, PIT) are reported
for **both** arms from their archived quantiles, and if a calibration layer is applied it appears
beside the raw intervals rather than in place of them (amendment A7, §5b).

**What would count as a positive result.** Few-shot beating persistence at h3 or h5 on the primary
arm, with the confidence interval clearing zero, at the frozen 5 seeds. Nothing weaker than that gets
written up as the method working.

## 5. Scoring protocol

1. Each arm is scored exactly once against the frozen file named above.
2. No Ebola cell of any kind enters trunk training, joint training, adapter fitting on the
   development folds, model selection, or early stopping. C8 guards are in `train/loop.py`,
   `train/lodo.py`, `train/joint.py` and `bundles.py`, and they now match on the `ebola` name prefix
   rather than the exact string, so `ebola_L12` and `ebola_L20` cannot slip past them.
3. Support cells are used for adapter fitting and for the scaler. Nothing else.
4. Both arms are reported, whatever they say. The secondary is labelled secondary everywhere.
5. If the run has to be repeated for a defect (a crash, a wrong checkpoint, a bug of the E4 kind),
   the repeat and its reason are recorded here before the re-scored numbers are used.

## 5a. Amendment log

Every change made to this document after it was first committed, with its reason. All of them
predate the first Ebola score. Nothing here may be added once scoring has run.

| # | date | change | why |
|---|---|---|---|
| A1 | 2026-08-07 | Evaluation size corrected from 1,151 / 1,075 / 866 / 642 to **757 / 766 / 765 / 642**; the old figures are kept as a separate *reachable* row | The audit note counts per-horizon origins. `bundles.origins()` uses one common origin set for all horizons, t in [19, 36], and that is what `score_predictions` scores. Found by reading the scoring path before running it, not after. h15 is unaffected. |
| A2 | 2026-08-07 | Adapter stopping rule fixed as leave-one-district-out CV inside the support set | Ebola has no validation split, so the epoch count was an unfixed free parameter. Chosen over a fixed epoch budget because 1,428 params on 48 pairs overfits fast and a fixed budget taken from the dev folds does not transfer: an Ebola epoch is ~13 origins against thousands. |
| A3 | 2026-08-07 | Short-window zero left-pad stated, with its known weakness | `window_slice` refused t < 19 outright, so no adaptation origin could be built at all. Implementing it was unavoidable; stating what the pad does to sin_doy/cos_doy is the part that belongs in a pre-registration. |
| A4 | 2026-08-07 | E4 corrected: it is the h15 head block that is bit-identical, not the h15 forecasts | `gamma`/`beta` are shared across horizons, so h3/h5/h10 gradient moves h15 predictions. The original claim was simply false and would have been reported as a bug on first contact with the data. |
| A6 | 2026-08-07 | E4 corrected a second time: the h15 head block is a **uniform shrink** of its initialisation, not bit-identical | The gate fired on the dry run at a drift of 1.0e-06. Cause is AdamW's *decoupled* weight decay, which shrinks a parameter whose gradient is exactly zero; Adam's own term is exactly zero there, so the block rescales and does not rotate. Proportionality across the 100 elements is a strictly sharper test than equality: it survives the optimiser detail, which carries no Ebola information, and is destroyed by any label-driven gradient. Tolerance is relative and sized for float32 accumulation over ~160 decay steps (predicted ~1e-6, measured 1.4e-6, gate at 1e-5). |
| A5 | 2026-08-07 | Trunk defined as all-five-dev-bundle joint, `trunk_patience=30`; floors fixed as persistence and support_mean | No all-dev trunk existed: every LDO3 checkpoint holds a disease out and the joint runs predate COVID and saved no checkpoints. Patience 30 against LDO3's 12 because the cosine schedule is scaled to a budget early stopping never reaches, and this run is scored once. Seasonal-naive is dropped because T=52 puts the t-52 lag out of panel at every scored origin. |
| A7 | 2026-08-10 | **Both arms archive count-space quantiles**, and a conformal calibration layer is permitted POST HOC on those archives under the four constraints in §5b | The zero-shot arm wrote records, per-node and per-origin but **not** quantiles (`train/ebola.py`), exactly the omission `LDO3_Results.md` §5 records costing the development-fold zero-shot arm its whole UQ block. Under §5.1 each arm is scored once, so a quantile array not written here can never be written and WIS, CRPS, coverage, PIT and any calibration layer would be permanently unavailable for zero-shot. Registered now because §5a forbids additions once scoring has run, and G4 (calibrated uncertainty) is a REQUIRED deliverable whose method is not yet finalised. |

## 5b. Uncertainty calibration (amendment A7)

G4 requires calibrated uncertainty and the method is not settled at the time of writing. Rather than
either freeze a variant we have not finished evaluating or leave the case study permanently unable to
carry one, the *mechanism* is registered here and the *variant* is left open under four constraints.
All four are binding, and the variant must be written into the amendment log **before** it is run
against these archives.

1. **Post hoc only.** Any calibration layer operates on the archived count-space quantiles
   (`*__quantiles.npz`, both arms, all seeds). It does not alter a point forecast, the adapter fit,
   the epoch chosen by A2, or any of E1–E5. Every number those expectations concern is fixed by the
   single scored run and is not revisited.
2. **Reported alongside, never instead.** Raw quantile-head intervals and their empirical coverage
   are reported for every cell whether or not a calibrated version exists. A calibrated interval is
   an additional column, not a replacement, so the uncalibrated result stays visible.
3. **Never selected on Ebola.** The variant and any hyper-parameter (for adaptive conformal
   inference, the step size) are chosen on development folds only and frozen before touching these
   archives. The intended variant is online ACI, because Ebola carries no calibration split by
   design (`PROJECT.md`), but that is a statement of intent here, not a registration.
4. **Truth consumption is declared.** An online method observes each origin's outcome after that
   origin has been forecast, in natural time order, and never before. If the method used consumes
   query truth in any other way, that is stated explicitly in the amendment log and in the paper.

This registration is what makes A7's archiving decision meaningful: without it the quantiles would be
written and then unusable, because §5a forbids adding a protocol after scoring has run.

## 6. Reproducing this

```
conda run -n ebola python freeze_ebola_arms.py            # rebuild both arms from the raw xlsx
conda run -n ebola python freeze_ebola_arms.py --verify   # rehash what is on disk
```

The build re-derives every count in section 3 from the raw file and refuses to write if any of them
has moved.

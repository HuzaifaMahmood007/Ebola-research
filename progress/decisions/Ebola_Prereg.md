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
| scored query pairs | 1,151 | 1,075 | 866 | 642 |
| scored districts | 61 | 61 | 59 | 58 |

Identical under both arms, and asserted by the freeze script. A full-window query target sits at
column 22 or later and support reaches at most column 20, so neither arm costs a single scored
forecast. The two arms are therefore directly comparable: same forecasts, different amount of
labelled adaptation data. The cost of the longer support set is to the claim, not to the evaluation.

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

Definitions used below. **Zero-shot** means the trunk trained on the development diseases with the
adapter left at initialisation, no Ebola label touched at any point. **Few-shot** means the same
trunk with the FiLM-plus-head adapter fit on that arm's support cells only. Reference is
`persistence` on the same scored cells, per `analysis.py` NAIVES; `seasonal` is undefined on Ebola,
which has no prior year.

These are predictions, and I expect to be reporting some of them as wrong.

**E1. Zero-shot loses to persistence at every horizon, on both arms.** Confidence: high. The LDO3
result is that cross-disease transfer is negative in 25 of 36 cells and that zero-shot fails on every
development fold. Ebola is a further shift than any fold in that run, so I have no basis for
expecting it to be the exception.

**E2. Few-shot beats zero-shot at h3 and h5, on both arms.** Confidence: moderate. This is the
weakest thing the adapter has to do, and h3/h5 are where it has the most examples.

**E3. Few-shot still does not beat persistence at any horizon on the primary arm.** Confidence:
moderate. If this is wrong it is the best outcome available here and I will say so plainly.

**E4. Primary arm, h15: few-shot and zero-shot must be bit-identical.** This is not a prediction, it
is a consequence of 0 adaptation pairs. Any difference between the two numbers is a bug in the
adaptation path, not a result, and will be treated that way. h10 on the primary arm has 18 pairs
across 9 districts, which I do not consider a fitted adapter; I expect no separation from zero-shot
beyond the noise floor.

**E5. The secondary arm beats the primary arm at h10 and h15.** Confidence: moderate. It has 72 and
54 adaptation pairs where the primary has 18 and 0. **If it does not, the failure is not
support-limited**, and no amount of extra Ebola labelling rescues the few-shot framing. That is the
single most informative cell in this exercise, and it is why the secondary arm is worth its compute
even though it weakens the headline from few-shot to moderate-data transfer.

**E6. Reporting.** Every number carries mean and standard deviation over the 5 frozen seeds (42, 52,
62, 72, 82) and a bootstrap confidence interval over districts and time origins. Any cell inside the
noise floor is written "within noise" and is given no direction. No averaging across diseases. The
comparison reference is stated on every delta table. h10 and h15 under the primary arm are labelled
zero-shot in every table, never few-shot.

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

## 6. Reproducing this

```
conda run -n ebola python freeze_ebola_arms.py            # rebuild both arms from the raw xlsx
conda run -n ebola python freeze_ebola_arms.py --verify   # rehash what is on disk
```

The build re-derives every count in section 3 from the raw file and refuses to write if any of them
has moved.

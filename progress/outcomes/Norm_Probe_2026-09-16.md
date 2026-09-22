# The normalisation mismatch, tested in both directions

**Run 2026-09-16, 23:43 to 23:53. Recorded 2026-09-17.** Script `experiments/norm_probe.py`.
Outputs `experiments/norm_probe__seed{42,52,62,72,82}.json` and
`experiments/norm_probe__exp2_summary.json`.

**Everything here is EXPLORATORY and is stamped `protocol="EXPLORATORY"` in every record.** It is not
a scored result, it does not amend `progress/decisions/Ebola_Prereg.md`, and it must never appear
beside the pre-registered numbers without that label. The scored Ebola record in `results/ebola/` is
untouched.

**The frozen hashes were the hard constraint and they held.** `load_manifest(verify=True)` re-hashes
both arms' array contents against `configs/ebola_arms.json` before the run, after every seed, and at
the end. All rescaling happens in memory on loaded bundles. Nothing was written to `results/` or
`data/`. Every scoring path reproduced its own archived record exactly before any exploratory number
was produced, on all five seeds.

---

## In plain words

The shared encoder was trained on inputs scaled district by district, so every district it ever saw
sat centred on zero. Ebola is fed in under a single pooled scale, so its districts arrive sitting
about 0.7 away from zero. That is a difference we introduced ourselves, and the obvious question is
whether it is the reason adaptation fails on Ebola.

I tested it from both ends. Feeding the development diseases the Ebola way made them worse
everywhere, by 5 to 102 per cent, so the mismatch genuinely costs accuracy in exactly the direction
Ebola experiences it. Feeding Ebola the development way, across five seeds, never helped once. Short
horizons were 10 to 19 per cent worse on every seed. A hint of a long-horizon improvement that
appeared at one seed did not survive the other four.

The reason Ebola cannot escape is simple: per-district scaling needs each district's own history, and
43 of Ebola's 61 districts have no support data at all. There is nothing to compute a scale from.

---

## Why it was run

`CLAUDE.md` section 6 records the mismatch as live and unresolved, and
`progress/planning/Gap_Ledger.md:321-324` lists the matched experiment under "runs, none of them
recommended", never executed. It became the prime suspect after
`progress/outcomes/Fewshot_Init_Probe_2026-09-15.md` ruled out sample size, sparsity and padding as
sufficient causes of the few-shot damage, leaving normalisation as the largest remaining difference
between Ebola and every development fold.

---

## What was tested

**Experiment 1, a development disease scored the Ebola way.** The trunk was trained on per-node
inputs; its held-out panel is fed **pooled** inputs instead, one log1p mean and standard deviation fit
on the train fold, Ebola-style. Same trunk, same adapter, same test cells. The input scaling is the
only thing that changes. Seed 42 only, because the `encoder_ldo3full` checkpoints it needs exist only
at that seed; the other seeds print a skip line. Deterministic inference, no fitting.

**Experiment 2, Ebola scored the development way.** A per-node scaler fit on each district's own
support cells, with the country fallback at `to_schema.py:158-165` for districts whose support is
absent or constant, and the districts with **no support cell at all** given a normalised incidence
channel of exactly 0.0000 so they arrive centred the way every development district does. Inversion to
counts uses each district's assigned statistics. The few-shot adapter is refit under the
pre-registered leave-one-district-out stopping rule. Both arms, five seeds, paired by seed.

### The partial-parity limit, which must be stated with any Experiment 2 number

This is **not** the development scheme. Ebola's data cannot support it.

| | L12 | L20 |
|---|---|---|
| districts | 61 | 61 |
| with any support cell | 18 | 36 |
| getting a genuine per-node scale | **11** | 19 |
| falling back to their country's pooled scale | 50 | 42 |
| with no support cell, pinned to 0.0 input | **43** | 25 |
| distinct scales across 61 districts | 14 | 22 |
| support districts' mean distance from zero | 0.70 → **0.52** | 0.63 → 0.50 |

So the "development-style" arm is country-pooled for 82 per cent of L12's districts and still sits at
0.52 from zero, against 0.0000 on influenza and covid. It closes roughly a quarter of the gap. This
experiment tests the closest approximation Ebola's data admits, not parity.

---

## Experiment 1: the mismatch is real and it has a price

Cost of pooled inputs on a per-node trunk, country-macro MAE, seed 42:

| panel | h3 | h5 | h10 | h15 |
|---|---|---|---|---|
| `covid_us-states` | +28.5% | +22.1% | +8.1% | +8.9% |
| `influenza_japan` | +44.4% | +18.8% | +5.3% | +4.6% |
| `influenza_us-regions` | +16.4% | +16.8% | +14.5% | +19.4% |
| `influenza_us-states` | +64.3% | +66.5% | +78.2% | +101.5% |

**Every panel, every horizon, 16 of 16 cells worse**, from +4.6% to +101.5%. This is the quotable
half of the run: deterministic inference, validated against archived records, no fitting, no seed
variance to argue about.

**Do not merge this with the other pooling result.** `CLAUDE.md` section 6 records that *consistent*
pooling, trained and tested pooled, **helps** by 6.8% on the matched panel. That is a different
experiment. Consistent pooling helps; **mismatched** pooling hurts. Both are true and a sentence about
"pooling" that does not say which is wrong in whichever direction it is read.

---

## Experiment 2: Ebola the development way, five seeds

Country-macro MAE, means over seeds 42/52/62/72/82. `pz` pooled zero-shot (the shipped
configuration), `pf` pooled few-shot, `dz` devnorm zero-shot, `df` devnorm few-shot. Counts are seeds
out of 5.

| arm | h | pz | pf | dz | df | df<pz | df<dz | dz<pz | df vs pz, mean [min,max] |
|---|---|---|---|---|---|---|---|---|---|
| L12 | 3 | 22.85 | 24.69 | 25.51 | 26.48 | **0/5** | 0/5 | **0/5** | +15.9% [+12.2, +19.2] |
| L12 | 5 | 24.00 | 24.61 | 25.94 | 26.32 | **0/5** | 1/5 | **0/5** | +9.7% [+7.8, +13.5] |
| L12 | 10 | 23.55 | 26.05 | 25.10 | 26.11 | 0/5 | 3/5 | 0/5 | +10.9% [+4.1, +26.1] |
| L12 | 15 | 20.14 | 20.10 | 22.12 | 20.91 | 2/5 | 4/5 | 0/5 | +3.8% [-1.2, +16.3] |
| L20 | 3 | 21.66 | 31.94 | 23.44 | 25.83 | **0/5** | 0/5 | **0/5** | +19.3% [+17.0, +20.4] |
| L20 | 5 | 23.23 | 25.75 | 25.40 | 25.79 | **0/5** | 1/5 | **0/5** | +11.0% [+8.0, +14.6] |
| L20 | 10 | 24.40 | 35.14 | 27.85 | 25.90 | 2/5 | 4/5 | 0/5 | +6.3% [-3.8, +32.2] |
| L20 | 15 | 21.79 | 31.54 | 26.47 | 28.79 | 1/5 | 3/5 | 0/5 | +31.7% [-7.3, +107.0] |

The L12 h15 row has **zero adaptation pairs**; its few-shot head is a random projection, not a fit
(`train/ebola.py:195 _check_e4`). It is not evidence in either direction and must not be quoted.

### The two questions this run was commissioned to answer

**Does the development-style scaler help at short horizons? No, and this is as firm as five paired
seeds get.** `df<pz` and `dz<pz` are **0/5 on both arms at h3 and h5**. Devnorm few-shot lands 9.7% to
19.3% worse than the shipped pooled zero-shot on the seed means, with every individual seed positive.

**Does it improve long horizons? No. That was seed noise.** At L20 h15 the improvement is **1/5**: one
seed at -7.3% and another at +107.0% on the same cell. L20 h10 is 2/5, mean +6.3%, range [-3.8, +32.2].

**The flattest result in the table: `dz<pz` is 0/5 on every row, which is 0 of 40 paired cells.** The
pre-registered pooled scaler beats the closest development-style approximation in every arm, horizon
and seed. Counting the few-shot arm too, devnorm beats the shipped configuration in only 5 of 40
cells, and 3 of those 5 are the L12 h15 random head or the single L20 h15 outlier seed.

### A reading withdrawn

After the seed-42 run alone I read the L20 h10 and h15 cells as evidence that adaptation damage
softens once the fit region and the use region are brought closer together, consistent with the
extrapolation mechanism in `progress/outcomes/Adapter_Mechanism_2026-09-16.md`. **That reading does
not survive five seeds and I withdraw it.** Seed 42 was the only winning seed at L20 h15. This is
exactly the failure mode a one-seed read invites, and it is why the multi-seed run was worth the hour.

### A secondary observation, fenced

Rescaling makes the few-shot arm **less seed-erratic in some cells but not as a general property**.
On L20 h3 the archived pooled few-shot ranges 23.95 to 53.46 across seeds, a spread of 29.51, while
devnorm few-shot ranges 25.31 to 26.16, a spread of 0.85. At L20 h10 the spreads are 29.24 against
7.60, and devnorm few-shot beats pooled few-shot **5 of 5 seeds** there, the only unanimous cell in
the run. But at L20 h15 devnorm is the *wider* of the two, 25.86 against 21.57, and across all 40
paired cells devnorm few-shot beats pooled few-shot only **14 times**. So this is a property of two
cells, not of the method. It is consistent with the extrapolation story and is not evidence for it.

---

## What is settled and what is not

**Settled.**

- The mismatch costs real accuracy in the direction Ebola experiences it: 16 of 16 development cells
  worse, +4.6% to +101.5%.
- Ebola cannot be given development-style normalisation. 43 of 61 districts have no support history,
  so 82 per cent of L12 falls back to country pooling and the result still sits at 0.52 from zero.
- The closest available approximation is **strictly worse**: 0 of 40 paired cells improved on the
  zero-shot arm, and the short horizons are worse on every seed.
- The pre-registered pooled configuration is therefore vindicated as the best of the alternatives
  tried, across seeds rather than at one.

**Not settled.**

1. **Whether the mismatch causes the few-shot adaptation failure.** A partial fix failing neither
   proves nor refutes causation, because the real fix requires history Ebola does not contain. The
   honest status is plausible but untestable with this data.
2. **Experiment 1 is one seed.** It is deterministic inference validated against archived records, so
   it carries no fitting variance, but it is a single trunk per fold.
3. **Five paired seeds support sign counts and "consistent across seeds", nothing stronger.** No
   significance test was run and no significance language appears above.
4. The L12 h15 cell is a random head throughout and is excluded from every claim.

---

## Where this leaves the hypothesis, and the wider position

**As a lever, normalisation is closed.** There is no configuration change available here that
improves the Ebola result, and no case to put to the client for re-registration.

**As an explanation it survives, weakened and untestable.** Experiment 1 shows the mismatch is
genuinely harmful in Ebola's direction; Experiment 2 shows the partial remedy does not recover it.

This completes a run of failed levers against the few-shot damage, each measured rather than assumed:
warm-start (unbuildable, no development fold fails the Ebola way,
`Fewshot_Init_Probe_2026-09-15.md`), post-hoc shrinkage (refuted, `Shrinkage_Verdict.md`),
meta-learning (better in 0 of 32 cells, `ANIL_Results.md`), adapter capacity (gains land at h15 where
the primary arm has no adaptation data, decision D19), and now rescaling (0 of 40). The pooled
zero-shot configuration fixed in the pre-registration is the best of everything tried. That is the
boundary-conditions thesis with more evidence behind it than it had.

---

## Reproducing

```
conda run -n ebola-train python -m experiments.norm_probe --selfcheck
conda run -n ebola-train python -m experiments.norm_probe --seeds 42 52 62 72 82
conda run -n ebola-train python -m experiments.norm_probe --seeds 42 --only exp1
```

About 10 minutes per seed, almost all of it Experiment 2's leave-one-district-out CV refit, with L20's
36 folds the slow half. One JSON is written per seed as it completes, so a crash costs one seed rather
than the run.

`--selfcheck` runs automatically before the main path and checks four things: the frozen arm hashes
against `configs/ebola_arms.json`, that L12 yields exactly 43 zero-support districts, that the
development-style scaler round-trips to real counts on observed cells, and that the pooled development
scaler really is a single shared mean and standard deviation.

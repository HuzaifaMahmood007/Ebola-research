# Step 1: the dev-fold rehearsal, and why it cannot be built

**Run and recorded 2026-09-15.** Script `diagnostics/fewshot_sim.py`, pilot on `covid_us-states` at
seed 42. Records: 28 files `results/misc/fewshotsim__covid_us-states__*__seed42.json` plus
`results/misc/fewshotsim__summary__seed42.json`.

**No Ebola cell was fitted on or scored.** Ebola appears only as the source of a mask pattern that is
copied onto a development panel. C8 is asserted in code before any fit.

**Verdict: the experiment is unrunnable as designed, and the reason is a finding in its own right.**
There is no development fold on which few-shot adaptation hurts, so there is no dev fold on which the
Ebola failure can be rehearsed or repaired. The 10 hours of trunk retraining planned for Step 2 would
have bought nothing and is cancelled.

---

## In plain words

We wanted to reproduce the Ebola failure somewhere we could experiment freely, then try fixing it by
giving the adapter a better starting point. So I took the real Ebola support mask, all 59 cells across
18 districts, and stamped it onto a development panel so the model would be learning from exactly as
little data, laid out exactly the same way, with the same near-empty input windows.

Then adaptation **helped**. It did not fail the way it fails on Ebola. So there was nothing to fix.

I checked whether any development panel fails the Ebola way, using results we already had on disk.
Across five panels, five seeds and four horizons, the no-adaptation model beats the adapted model in
**4 of 100** cells. On Ebola it wins most of them. Development folds and Ebola behave in opposite
directions, so no development fold can stand in for Ebola on this question.

The useful part: I gave a development fold Ebola's small sample, Ebola's sparsity, Ebola's padded
windows and Ebola's exact per-horizon data profile, and adaptation still helped. **So none of those
things is what breaks Ebola.** That kills the two explanations a reviewer reaches for first.

---

## What Step 1 was for

Step 0 (`progress/outcomes/Ebola_Feature_Shift_2026-09-15.md`) measured that the Ebola adapter fits on
trunk features from windows that are 72.5% zero padding at 5.2% observation, then gets scored on full
windows at 31.6%, and that its design matrix is rank deficient at every horizon on the primary arm
(45, 37, 17 and 0 of 64 directions).

That is a measurement, not a counterfactual. It shows the fit region is degenerate; it does not show
that repairing it would forecast better. Step 1 was meant to supply the counterfactual, and to fill the
hole flagged in `Reports/Ebola_Audit_Note.md:159-163`:

> "we currently have **no dev-set rehearsal of the 27-cell regime** ... If the result comes back weak
> we will not be able to separate 'transfer does not work' from '16 pairs cannot fit anything.'"

---

## Design

Step 0 found **two** degeneracies, not one, so placement was crossed 2x2 rather than contrasted once.

| placement | window padding | cells in block | isolates |
|---|---|---|---|
| `t0-blind` | yes (72.5%) | blinded | **Ebola exactly** |
| `t0-dense` | yes | intact | padding |
| `mid-blind` | no | blinded | sparsity |
| `mid-clean` | no | intact | neither, small-sample control |

Crossed with four initialisations: `zeroshot` (mean adapter, no fit), `fresh` (random, the protocol at
`Ebola_Prereg.md:126-127`), `freshzs15` (random except horizons with zero pairs, which get the
zero-shot head), `warm` (mean adapter, then fit). L2-SP was deferred to Step 2 since it only refines
`warm`.

### Construction, and three traps handled

**Transplant the real mask, not an invented one.** The 18x13 block is read from
`data/processed/ebola_L12.npz` (36x21 for L20) and laid on 18 seed-fixed nodes of the held-out panel.

**Pin the origin set.** Ebola's 48/38/18/0 profile exists only because origins clip at `t >= 0`.
Dropping the same block at column 40 makes every support cell reachable at every horizon, giving
59/59/59/59, so `mid` placements would have differed in *data volume* as well as feature quality and
the contrast would have been confounded. Origins are therefore pinned to `range(t0, t0+10)` for L12 and
`range(t0, t0+18)` for L20, the counts Ebola actually has. Verified: the profile then reproduces
exactly at both `t0 = 0` and `t0 = 40`.

**Blind the inputs the way Ebola is blinded.** Measured on the real bundle: an unobserved Ebola cell
has `X[...,0] == 0` and obs_mask `X[...,3] == 0`, while `sin_doy`/`cos_doy` stay live. Blinding
therefore touches channels 0 and 3 plus `M` and `y`, and leaves channels 1 and 2 alone. Asserted.

**One edit to shared code.** `train/ebola.py:_fit_adapter` and `choose_epochs` gained an
`init_state=None` kwarg. The seeding stays above the load, so the RNG is in an identical state whichever
init is used and the per-epoch origin permutation is the same stream across arms; a warm-versus-fresh
comparison at one seed then differs only in where the optimiser started. The default path is asserted
byte-identical by sha256 against a hash captured from the pre-edit code.

---

## Process, including the run that failed

The first pilot completed all 28 cells and reported **nan for every metric**. The epoch counts varied
sensibly (66, 79, 72, 60), so the fitting and the cross-validation had worked; the fault was in the
summary layer. `score_predictions` emits **long form**, one record per (metric, horizon) carrying a
`metric` field and a flat `country_macro`. My reader assumed a wide record with nested metric dicts, so
every lookup missed and returned an empty list, which averaged to nan rather than raising.

Two things made that cheap to recover from. Each cell is written to its own JSON as it finishes rather
than accumulated in memory, so the expensive half was already archived; and a `--from-disk` mode now
rebuilds the tables from those records without refitting. The corrected reader asserts on an empty
lookup instead of returning nan, so the same class of bug fails loudly next time.

**Checks that passed, and matter for trusting the numbers:**

1. The `zeroshot` arm reproduces `encoder_ldo3full_zeroshot__covid_us-states__seed42` to four decimal
   places at all four horizons. The mean adapter came from the right checkpoint slot and the scoring
   path is the same one that produced the archived records.
2. Pair profiles exact at both placements and both templates.
3. No test window reaches the transplanted block; the earliest test window starts at column 81 and the
   widest block ends at column 60.
4. `_fit_adapter(init_state=None)` sha256 unchanged.
5. `freshzs15` is numerically identical to `fresh` at h3/h5/h10 and differs only at h15, which is what
   it was built to do. That the arm behaves exactly as specified is evidence the init patching works.

---

## Results

`covid_us-states`, seed 42, country-macro MAE, lower is better.

| template | placement | zeroshot | fresh | freshzs15 | warm | warm vs fresh | adaptation |
|---|---|---|---|---|---|---|---|
| L12 | `t0-blind` | 12402.5 | **7829.5** | 11370.5 | 10736.2 | +37.1% | **helps** |
| L12 | `t0-dense` | 12402.5 | **8244.8** | 11378.4 | 15638.5 | +89.7% | helps |
| L12 | `mid-blind` | 12402.5 | 26813.3 | 29521.4 | 23874.9 | -11.0% | hurts |
| L12 | `mid-clean` | 12402.5 | 13306.3 | 15981.1 | **11333.5** | -14.8% | roughly neutral |
| L20 | `t0-blind` | 12402.5 | 8112.5 | - | **6888.1** | -15.1% | **helps** |
| L20 | `t0-dense` | 12402.5 | **6268.5** | - | 10629.6 | +69.6% | helps |
| L20 | `mid-blind` | 12402.5 | 18629.6 | - | 14162.4 | -24.0% | hurts |
| L20 | `mid-clean` | 12402.5 | 10806.2 | - | **8720.6** | -19.3% | helps |

**The decisive row is `t0-blind`, which reproduces Ebola's geometry exactly.** Adaptation helps there,
7829.5 against 12402.5 on L12 and 8112.5 against 12402.5 on L20. Ebola is the opposite. The phenomenon
under investigation is simply absent, so `warm` has nothing to repair and its column is noise: it swings
from -24.0% to +89.7% with no consistent direction.

---

## Diagnosis: no development fold is Ebola-like

This is the finding, and it rests on the existing 5-seed archives rather than on the probe. For each
panel and seed I compared the adapted arm against its own zero-shot arm, counting horizons where
**zero-shot wins**, which is the Ebola situation.

| panel | seeds | zero-shot wins, of 4 horizons per seed | total |
|---|---|---|---|
| `covid_us-states` | 5 | 0, 1, 0, 0, 0 | 1 / 20 |
| `dengue` | 5 | 0, 0, 1, 0, 0 | 1 / 20 |
| `influenza_japan` | 5 | 0, 0, 0, 0, 0 | 0 / 20 |
| `influenza_us-regions` | 5 | 1, 0, 0, 0, 1 | 2 / 20 |
| `influenza_us-states` | 5 | 0, 0, 0, 0, 0 | 0 / 20 |
| | | | **4 / 100** |

On development folds adaptation wins **96 of 100** cells. On Ebola it loses. The covid fold at full
train-fold adaptation shows the same thing per horizon: adapted 5193.8 / 6162.3 / 8974.5 / 11004.9
against zero-shot 5931.3 / 12450.0 / 13532.1 / 17696.6, adapted winning 4 of 4.

So the premise of Step 1, that a dev fold can be made to fail the Ebola way by giving it Ebola's data
budget, is false. Development folds do not fail that way at full data and they do not fail that way at
59 cells either.

---

## What this rules out

A development fold was given Ebola's support size, Ebola's sparsity, Ebola's zero-padded windows and
Ebola's exact per-horizon pair profile, and adaptation still helped. Therefore **none of the following
is a sufficient cause of the Ebola damage**:

- too few labels (59 cells, 48/38/18/0 pairs),
- mostly-padding input windows,
- mostly-unobserved cells inside the fit window,
- any combination of the three.

Combined with Step 0, the position is: the fit region really is degenerate and displaced, and that
degeneracy is real, but it is **not what breaks Ebola.**

One further nuance worth recording, because it cuts against a story we might otherwise have told.
`freshzs15` patches the h15 head with the zero-shot head at a horizon that has no training data. On
covid that made h15 **worse**, 22259.2 against `fresh`'s 8095.1, because covid's borrowed head is poor
at h15 (17696.6) and random numbers happened to beat it. So "a random head at an unsupervised horizon
is bad" is not a general truth. It depends on whether the borrowed head is good, and on Ebola the
borrowed head is good. The h15 mechanism from Step 0 survives for Ebola but does not generalise, and
must not be stated as a general claim.

---

## What this does NOT establish

1. **One panel, one seed, no interval.** The probe carries no significance claim. The load-bearing
   evidence for the diagnosis is the 4-of-100 table, which comes from the archived 5-seed records
   independently of the probe.
2. **The scaler was never matched.** Dev panels keep `per_node_train`; Ebola uses pooled
   `per_disease_support`. This was deliberate, to isolate sample size from normalisation, but it means
   the probe never tested the one difference that remains.
3. **It does not show adaptation cannot be repaired on Ebola**, only that a dev fold cannot be used to
   find out.
4. `mid-blind` was the only configuration where adaptation clearly hurt, and it is the one where the
   window goes dark in its final weeks. That rhymes with the six-week Ebola reporting gap, but it is an
   uncontrolled observation from a single cell and no claim is made.

---

## Where this leaves the question

The pre-specified branch for this outcome fires: **pivot to normalisation.** The largest remaining
difference between Ebola and every development fold is the one in `CLAUDE.md` section 6. The trunk
trains on `per_node_train` inputs centred at district mean 0.0000; Ebola arrives under pooled
`per_disease_support` with district means scattered 0.6550 (L12) and 0.5363 (L20) either side. That is a
covariate shift we introduced ourselves, we publish the diagnostic, and the matched experiment, trunk
trained per-node and tested pooled, has never been run. `progress/planning/Gap_Ledger.md:321-324` lists
it under "runs, none of them recommended", which should now be revisited given this result.

Step 2 as planned (all panels, five seeds, L2-SP, ~14 trunk retrains at 9-10 hours) is **cancelled**.
It was gated on this pilot not coming back degenerate, and it did.

---

## Reproducing

```
conda run -n ebola-train python -m diagnostics.fewshot_sim --selfcheck
conda run -n ebola-train python -m diagnostics.fewshot_sim --seed 42 --bundles covid_us-states
conda run -n ebola-train python -m diagnostics.fewshot_sim --from-disk --seed 42
```

Module form is required; running by path puts `diagnostics/` on `sys.path` and `import bundles` fails.
The full pilot takes roughly 25 to 35 minutes, most of it the L20 leave-one-district-out CV at 36 folds.
`--from-disk` rebuilds the tables from the archived per-cell records in seconds and fits nothing.
`--selfcheck` runs automatically before the main path, so the numbers cannot be produced against a moved
arm or a changed fit path.

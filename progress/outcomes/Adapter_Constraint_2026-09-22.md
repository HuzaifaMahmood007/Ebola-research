# Does constraining the Ebola adapter recover the few-shot damage? Partly.

**Run and recorded 2026-09-22.** Script `experiments/adapter_constraint.py`, both arms, 5 seeds
(42/52/62/72/82), an 11-point constraint grid. Outputs `experiments/adapter_constraint__seed*.json`
and `experiments/adapter_constraint__summary.json`. Adapter refits on CPU, about two minutes per seed.

**Everything here is EXPLORATORY and is stamped `protocol="EXPLORATORY"` in every record.** It is not a
scored result, it does not amend `progress/decisions/Ebola_Prereg.md`, and it must never appear beside
the pre-registered numbers without that label. `load_manifest(verify=True)` re-hashes both frozen arms
against `configs/ebola_arms.json` before and after every seed, all rescaling and refitting happens on
loaded checkpoints in memory, and nothing is written to `results/` or `data/`. Every constrained map is
validated first: the t=1 end reproduces the archived few-shot arm and the t=0 end reproduces the
archived zero-shot arm, both to 6e-06 MAE.

---

## In plain words

`progress/outcomes/Adapter_Mechanism_2026-09-16.md` showed, as geometry, that 76 to 85 per cent of what
the few-shot adapter changes is applied 5 to 12 times beyond the range the support data ever covered. It
said in its own limits section that this is geometry, not proof: it does not show that the extrapolation
is what causes the measured few-shot damage. This run is the causal step. If pulling the adapter back
toward the borrowed map that adds no Ebola-specific change shrinks the error, the extrapolation is
carrying the damage. If it does nothing, the mechanism story weakens.

I found the answer is **partially causal, and it splits by arm**. On the primary 12-week arm, choosing
how hard to pull on the support data alone recovers 77 to 90 per cent of the few-shot damage at h3, h5
and h10. On the secondary 20-week arm, the same support-only rule barely pulls at all and recovers only
9 to 19 per cent. Underneath that, the mechanism is real and complete: removing the whole few-shot
change (the t=0 end) recovers essentially 100 per cent of the damage on every damaged cell. So the
damage lives entirely in the few-shot-specific change, of which the extrapolation is the bulk. What
varies is whether the support-only selection is able to see that it should pull, and on the 20-week arm
it mostly cannot.

---

## What was tested

With the trunk frozen the adapter is exactly affine (`models/adapters.py:14-16`): `pred = h A^T + c`. The
zero-shot borrowed-mean adapter is a second affine map `(A_z, c_z)`, the one that adds no Ebola-specific
change, so the few-shot-specific change is `D = A_few - A_z`. A ridge penalty `lam*||A - A_z||^2` shrinks
the fitted map along the segment toward `A_z`, which in closed form is

```
A(t) = A_z + t (A_few - A_z),   c(t) = c_z + t (c_few - c_z),   t = 1 / (1 + lam*k)
```

so the strength `t` is a reparametrisation of the ridge strength: `t=1` is the unconstrained few-shot
map (D untouched), `t=0` is the zero-effect map (D removed), `t<1` shrinks D by the factor t. The grid is
`t` in {0.0, 0.1, ..., 1.0}. `t` is selected on **support only** by leave-one-district-out CV inside the
support set, the same machinery the pre-registration uses to pick the adapter epoch. Per held-out
district I refit the few-shot adapter on the remaining support at that arm-and-seed's pre-registered
epoch count, interpolate that fold map toward `A_z` over the grid, and score the held-out district. No
Ebola query cell is read in selection. The reported cell is the support-selected `t`; the query-oracle
best over the grid is reported as **context only**, since it peeks at the answer.

### Why shrinkage and not a literal ridge

A literal ridge run through the pre-registration's own optimiser **saturates**. With the gradient norm
clipped to 1.0, cranking `lam` shrinks D by only about 64 per cent of the way to `A_z` and never reaches
it (seed 42: the relative distance to `A_z` falls from 1.914 at `lam=0` to a floor of 0.688), and the
error effect is non-monotone. So a literal-ridge grid cannot span the full few-to-zero range or reference
full recovery. The closed-form shrinkage spans it exactly and its two ends are the archived arms.

### A second, parameter-free instrument (the clip)

Restrict the few-shot change to the per-horizon support span: `A_clip = A_z + (A_few - A_z) @ P_h`, where
`P_h` projects onto the span of the support feature rows the fit consumed at horizon h (the same rows the
mechanism doc measured). This deletes exactly the part of D acting on directions the support never
constrained. Support-span ranks are identical across seeds: L12 {h3:45, h5:37, h10:17, h15:0} and L20
{h3:64, h5:64, h10:60, h15:34}. No knob.

---

## The numbers

Country-macro MAE, seed means over 42/52/62/72/82. `few` is the archived unconstrained few-shot arm,
`zero` the archived borrowed-mean arm, `constr` the shrinkage at the support-selected t, `clip` the
parameter-free span restriction, `oracle` the best-over-grid (context only, peeks at the answer). `gap`
is `few - zero`, positive when the few-shot arm is worse and there is damage to recover.
`recovery = (few - constr) / gap`.

| arm | h | few | zero | constr | clip | oracle | gap | recovery | recovery clip | oracle (context) |
|---|---|---|---|---|---|---|---|---|---|---|
| L12 | 3 | 24.69 | 22.85 | 23.10 | 24.30 | 22.75 | +1.84 | **+86%** | +21% | +105% |
| L12 | 5 | 24.61 | 24.00 | 24.14 | 24.63 | 23.94 | +0.61 | **+77%** | -4% | +110% |
| L12 | 10 | 26.05 | 23.55 | 23.81 | 25.47 | 23.54 | +2.50 | **+90%** | +23% | +100% |
| L12 | 15 | 20.10 | 20.14 | 19.95 | 20.16 | 19.79 | -0.04 | n/a | n/a | n/a |
| L20 | 3 | 31.94 | 21.66 | 31.02 | 31.94 | 21.63 | +10.28 | **+9%** | +0% | +100% |
| L20 | 5 | 25.75 | 23.23 | 25.27 | 25.75 | 23.23 | +2.51 | **+19%** | -0% | +100% |
| L20 | 10 | 35.14 | 24.40 | 33.07 | 35.39 | 24.40 | +10.74 | **+19%** | -2% | +100% |
| L20 | 15 | 31.54 | 21.79 | 30.13 | 34.23 | 21.77 | +9.75 | **+14%** | -28% | +100% |

The L12 h15 cell has `gap` of -0.04, no damage to recover, so its recovery ratio is meaningless and is
excluded from every claim. Over the 7 cells that carry real damage (`gap > 0.5` MAE), the support-selected
constraint recovers **+45 per cent of the damage on average**: +84 per cent mean on L12 (h3/h5/h10), +15
per cent mean on L20 (h3/h5/h10/h15).

**Support-selected t, per seed.**
L12: [0.2, 0.3, 0.4, 0.4, 0.4], mean 0.34, median 0.40.
L20: [0.8, 0.8, 1.0, 0.9, 0.9], mean 0.88, median 0.90.
The primary arm pulls hard toward the zero-effect map; the secondary arm barely pulls, and one seed (62)
selects t=1.0, no constraint at all.

**Paired by seed, the recovery is consistent in sign, not a one-seed artefact.** On L12 the per-seed
recovery is +57 to +118% at h3, +15 to +95% at h5, and a tight +89 to +96% at h10. On L20 it is +0 to
+27% at h3, +0 to +28% at h5, +0 to +40% at h10, and +0 to +32% at h15 once the one gap-near-zero seed
outlier is set aside. Every damaged cell recovers a positive share on every seed that constrains at all.

**RMSE tells the same story.** L12 recovery is +93/+92/+99% at h3/h5/h10 (h15 gap tiny); L20 is
+8/+21/+19/+15%.

---

## The verdict

**Partially causal.** Constraining the adapter toward the zero-effect map, at a strength chosen on the
support data alone, recovers about 45 per cent of the few-shot damage on average, and that average hides
a clean split: 77 to 90 per cent on the primary 12-week arm at h3/h5/h10, 9 to 19 per cent on the
secondary 20-week arm. The extrapolation the mechanism doc measured is therefore carrying real, causal
weight on the error, not just geometry, but the support-only rule captures all of it only on the primary
arm.

Two facts keep this honest and stop it from being read as either "no effect" or "the whole story":

- **The damage is fully removable, and it lives in the few-shot-specific change.** The oracle column,
  which is the t=0 end, recovers essentially 100 per cent on every damaged cell. Removing D entirely
  removes the damage entirely. This is context, because t=0 is chosen by looking at the query answer, but
  it pins where the damage lives: in D, of which the extrapolation is 76 to 85 per cent by the mechanism
  doc. That is the causal core.
- **The strictly-out-of-span clip recovers almost nothing (+1 per cent on average).** Deleting only the
  part of D that acts outside the support span does not fix the error, and on L20, where the support is
  full or near-full rank, the clip is a no-op or worse (it hurts by 28 per cent at h15). So the damage is
  not concentrated on the directions the support never touched at all. It is in the general magnitude of D
  along under-constrained directions, which the scalar shrinkage catches and a hard span-projection does
  not. This refines the mechanism doc's "out-of-span" framing rather than confirming it literally.

---

## What is settled and what is not

**Settled.**

- The instrument is exact: t=1 reproduces the archived few-shot arm and t=0 the archived zero-shot arm,
  both to 6e-06 MAE, so this is measurement on the deployed maps, not a re-fit that could drift.
- The few-shot damage is fully attributable to the few-shot-specific change D: removing D recovers about
  100 per cent of it on every damaged cell.
- Constraining toward the zero-effect map on support-only signal recovers 77 to 90 per cent of the damage
  on the primary arm and 9 to 19 per cent on the secondary arm, consistent in sign across all five seeds.
- A literal ridge cannot do this test through the pre-registration's optimiser, because the gradient clip
  saturates it at about 64 per cent of D.

**Not settled.**

1. **This is one mechanism test. It is never "THE cause."** It shows that shrinking the few-shot change
   recovers the damage, and that the change is where the damage lives. It does not rule out that a
   different lever (normalisation, sample geometry) is upstream of why D sits where it does.
2. **The oracle and the clip both peek at, or bypass, the support-only rule.** The oracle chooses t on the
   query answer and is context only. The clip is parameter-free but its span is a hard threshold; a softer
   per-direction ridge would sit between the clip and the scalar shrinkage and was not run.
3. **Why the support-only rule under-constrains L20 is diagnosed, not proven.** L20's support is full or
   near-full rank and spans more of the feature space, so leave-one-district-out inside support does not
   reveal that the query sits far outside it, and the selection leaves recoverable damage on the table.
   This is consistent with the extrapolation mechanism but is an argument, not a separate measurement.

---

## How this changes what we can say

The adaptation-failure mechanism can move from "geometric" to "causal, partially, and arm-dependent." The
supported phrasing:

> Constraining the few-shot adapter toward the borrowed zero-effect map, at a strength chosen on the
> support data alone, recovers 77 to 90 per cent of the few-shot damage on the primary arm at h3, h5 and
> h10 and 9 to 19 per cent on the secondary arm. Removing the few-shot-specific change entirely recovers
> essentially all of the damage on every horizon, so the damage lives in that change, of which the
> out-of-fitted-range extrapolation is the bulk. The extrapolation is therefore carrying causal weight on
> the error and not only geometry, though a strictly-out-of-span restriction alone does not recover it,
> which locates the damage in under-constrained in-span directions rather than in directions the support
> never touched.

Phrasing that is **not** supported: any sentence calling this THE cause, quoting the oracle or the clip as
if it were the support-selected result, or reading the +45 per cent average without the arm split.

---

## Reproducing

```
conda run -n ebola-train python -m experiments.adapter_constraint --selfcheck
conda run -n ebola-train python -m experiments.adapter_constraint --seeds 42 52 62 72 82
conda run -n ebola-train python -m diagnostics.verify_adapter_constraint_doc
```

`--selfcheck` checks the frozen arm hashes, that the adapter is exactly affine and the affine
reconstruction round-trips to 0.0, and that t=1 and t=0 reproduce the archived few-shot and zero-shot
arms. The verifier parses the numbers out of this document and recomputes each from
`experiments/adapter_constraint__seed*.json`.

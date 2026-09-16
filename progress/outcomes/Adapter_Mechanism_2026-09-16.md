# Why the Ebola adapter loses to the borrowed one, and why 20 weeks is worse than 12

**Run and recorded 2026-09-16.** Script `diagnostics/adapter_mechanism.py`, 5 seeds, both arms.
Output `results/misc/ebolashift__adapter_mechanism.json`. Inference only, about a minute, no GPU.

**No Ebola label is read and nothing is scored.** Every quantity here is a property of the trunk's
features and the adapter's weights, so the pre-registration's score-once rule is untouched.

This answers the two questions left open by
`progress/outcomes/Ebola_Feature_Shift_2026-09-15.md` (which measured the shift) and
`progress/outcomes/Fewshot_Init_Probe_2026-09-15.md` (which ruled out sample size, sparsity and
padding as sufficient causes).

---

## The one fact that makes both questions answerable

With the trunk frozen the Adapter is **exactly affine** in trunk features. Folding FiLM into the head
gives `A = W·diag(gamma)` and `c = W·beta + b`, and `h·Aᵀ + c` reproduces the module with a maximum
deviation of **0.0** in its own float32 precision. `models/adapters.py:14-16` already stated this; the
script asserts it before computing anything.

So few-shot adaptation on Ebola is a 64-dimensional affine fit, `A` is a 20x64 matrix, and the whole
of it can be taken apart with an SVD instead of an experiment.

---

## Q1. Why does fitting not help?

**Because the fit is determined in one region and applied in another, and we can now say how much.**

Take the singular directions of the support design matrix. For each one, compare how far the support
data spread along it against how far the query data spreads, and weight by how hard the fitted map
pushes that direction. Mean over 5 seeds:

| arm | h | n_sup | rank /64 | cond | query energy out of span | share of change that extrapolates | median extrapolation |
|---|---|---|---|---|---|---|---|
| L12 | 3 | 48 | 45 | 3.68e05 | 3.3% | **83.2%** | **5.42x** |
| L12 | 5 | 38 | 37 | 2.57e04 | 5.4% | **85.3%** | **5.84x** |
| L12 | 10 | 18 | 17 | 8.17e04 | 15.4% | **75.7%** | **9.08x** |
| L12 | 15 | 0 | 0 | n/a | **100.0%** | **100.0%** | n/a |
| L20 | 3 | 102 | 64 | 1.61e04 | 0.0% | 77.7% | 5.40x |
| L20 | 5 | 92 | 64 | 2.19e04 | 0.0% | 83.2% | 5.21x |
| L20 | 10 | 72 | 60 | 1.18e06 | 0.6% | 83.4% | 8.02x |
| L20 | 15 | 54 | 34 | 5.12e04 | 8.5% | 82.0% | **11.15x** |

**Between 76 and 85 per cent of the change the fitted map makes to a query forecast acts along
directions where the query data spreads further than the support data ever did**, and the typical
contributing direction is pushed **5 to 12 times** beyond the range it was fitted over. Those are the
seed means, which is what the table shows; the individual seeds run wider, 71 to 89 across individual
seeds for the share and 3 to 14 across individual seeds for the extrapolation factor. Every seed of
every arm at every horizon sits above 70 per cent, so the effect is not carried by one fold.

The zero-shot map makes no such commitment. It was never fitted on Ebola, so it has no Ebola-specific
distortion to apply, and it comes from three development diseases' worth of dense data. Few-shot
loses because it replaces a generic map with one that is *correct on 59 support cells and
extrapolated everywhere it is actually used.*

Two horizons deserve separate mention.

- **L12 h15 is the pure case.** Zero support rows, so the fit is its random initialisation, decayed.
  100% of the query energy is outside a span that does not exist. `train/ebola.py:195 _check_e4`
  proves this must happen and has been passing all along. The primary arm's 15-week few-shot forecast
  is a random projection.
- **L12 h10 has 18 rows for 64 dimensions**, so 47 directions are never constrained by data and
  15.4% of query energy falls outside the span entirely.

### Why it is not simply "too few labels"

`Fewshot_Init_Probe_2026-09-15.md` transplanted exactly this sample size, sparsity and padding onto a
development panel and adaptation still helped there. The count is not the problem. The problem is
**where those rows sit relative to where the model is scored**, and that is a property of the Ebola
support window, not of its size.

---

## Q2. Why is the 20-week arm worse, with nearly twice the data?

**Because its extra data is not more of the same. It is a six-week blackout followed by a two-week
reporting burst.** Support cells per column:

```
ebola_L12  cols 0..12   [0, 2, 9, 8, 2, 2, 2, 2, 3, 11, 2, 9, 7]
ebola_L20  cols 0..20   [0, 2, 9, 8, 2, 2, 2, 2, 3, 11, 2, 9, 7, 0, 0, 0, 0, 0, 0, 20, 34]
                                                                 \___ six empty ___/  \__ 54 __/
```

| | L12 | L20 |
|---|---|---|
| support cells | 59 | 113 |
| longest run of wholly unreported columns | **1** | **6** |
| share of cells in the final two columns | **27.1%** | **47.8%** |

L20's eight extra columns contribute **nothing** in six of them and **54 cells in two**. Nearly half
of everything the 20-week arm learns from comes from two columns of post-blackout catch-up reporting,
a period that resembles neither the early outbreak before it nor the query period after it.

That combines badly with the second fact in the table above: **L20 is full rank (64 of 64) at h3 and
h5.** Full rank means there is no unconstrained subspace left where the initialisation survives.
Every one of the 64 directions is set by burst-dominated data, so the arm commits its whole map to a
distorted region instead of leaving most of it alone. Its long horizon is worse still: at h15 the
rank collapses to 34 while the median extrapolation rises to 11.15x, the highest in the table.

**So more support data made it worse by letting it commit harder to an unrepresentative window.**
That is the direction the withdrawn parameter-counting story got backwards: it predicted L20 should
be better because it has more data per parameter.

---

## What this does and does not settle

**Settled.**

- The adapter is exactly an affine map, verified to 0.0, so this is measurement rather than modelling.
- 76 to 85 per cent of what adaptation changes is applied outside the region it was fitted in, on
  every arm, horizon and seed.
- The primary arm's h15 few-shot forecast is a random projection, with zero fitting rows.
- L20's extra support is 6 blank columns plus 2 columns holding 47.8% of its cells.

**Not settled.**

1. **These are geometric quantities, not error decompositions.** They show the fitted map is applied
   far outside where it was determined; they do not prove that each percentage point of extrapolation
   costs a measurable amount of MAE. The link to the scored outcome is by argument, not by regression.
2. **The blackout story for L20 is consistent but not isolated.** I have not run the counterfactual
   of an L20-sized support drawn from unbroken reporting, which would separate "more data" from "data
   spanning a blackout". That would need a support window the pre-registration does not contain.
3. **Extrapolation is reported relative to the support spread per direction.** A direction where the
   support barely varies gives a large ratio by construction, which is why the summary uses the median
   over directions carrying at least 1% of the change, not the mean. The mean runs to the hundreds and
   is not quoted.
4. **The normalisation mismatch remains untested and unexcluded.** Ebola arrives pooled at district
   mean 0.655 against a trunk trained per-node at 0.0 (`CLAUDE.md` section 6). That could well be
   *why* the support features sit where they do, in which case it is upstream of everything here.

---

## How this changes what we can say

The project previously had no explanation for its central negative result. It now has a mechanism for
both halves of it, with the honest caveat that the mechanism is established geometrically rather than
by an error decomposition.

Phrasing that is supported:

> Few-shot adaptation fits a 64-dimensional affine map on as few as 18 support rows drawn from the
> first weeks of the outbreak, then applies it to a query period whose features extend 5 to 12 times
> beyond the fitted range along the directions that carry 76 to 85 per cent of the map's effect, both
> as seed means. The
> 20-week arm is hurt more because its additional support is a six-week reporting blackout followed by
> two columns carrying 48 per cent of its cells, and because at full rank it retains no unconstrained
> subspace to limit the damage.

Phrasing that is **not** supported: any sentence asserting this is *the* cause, or quantifying how
much MAE it accounts for.

---

## Reproducing

```
conda run -n ebola-train python -m diagnostics.adapter_mechanism --selfcheck
conda run -n ebola-train python -m diagnostics.adapter_mechanism --seeds 42 52 62 72 82
```

`--selfcheck` runs automatically before the main path and checks four things: FiLM folds into an
affine map with 0.0 deviation in float32, both arms still carry their frozen cell counts, and the
attribution returns 0% out-of-span for a query inside the support span and 100% for one entirely
outside it. That last pair is a positive and a negative control on the measure itself.

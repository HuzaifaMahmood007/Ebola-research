# G5 Explainability: Results and Verification

**Written 2026-09-10. Rewritten 2026-09-10 after the IG baseline fix and the two reruns.** This is
the committed verifier document called for in `progress/planning/G5_Explainability_Goals.md`
section 7, item 3. Gap Ledger row E1 (`progress/planning/Gap_Ledger.md:286`) lists this document as
one of four blockers keeping G5 at **PARTIAL, not DONE**. This document does not change that status.
It records what was checked, how, and what is still open. Internal engineering doc only, not
client-facing.

Every number below was recomputed by me today, straight from the archives in `results/explain/`,
with a script that re-implements the share, band, agreement and neighbour maths from scratch instead
of calling `explain.py`'s own reporting functions. Nothing here is transcribed from the previous
version of this document, from `Reports/explaiability_report.md`, or from pasted terminal output.

**What changed on disk since the first version of this document:**

1. `explain.py` gained `baseline_mu()` (line 146) and `ig_baseline()` (line 167). The IG and
   occlusion reference is no longer `torch.zeros_like(Zt)`.
2. dengue was rerun at 64 IG steps instead of 32.
3. both Ebola arms were rerun under the fixed baseline.

`results/explain/*.npz` (55 files), `results/reports/explain_report.txt`, `figures/explain.png` and
`figures/explain.pdf` are all regenerated. `Reports/explaiability_report.md` is **not**, see section 9.

---

## 1. How I verified this

Four checks, all run today:

1. **`conda run -n ebola-train python -m explain --selfcheck`**, the committed synthetic-model test
   suite. It passed clean, and it now names the baseline directly:
   > `ok  IG completes the path against ig_baseline and gradient x input does not, a zero baseline
   > fails the same residual so the check sees the reference, ig_baseline moves incidence only,
   > per-node scaling makes baseline_mu a no-op and pooled scaling does not, invisible channel gets 0
   > from IG and occlusion, forward_fixed_deg == forward and holds degree, gate-off is immune to edge
   > ablation, lag/band/share bookkeeping, the random-weight verdict flips on a failing draw, a
   > one-hot target completes on that node's own forecast and not on the pooled one, routing, count
   > inversion`

   Three of those items are new since the fix, and two of them are deliberate-failure checks on the
   baseline itself: a zero baseline must FAIL the residual test that `ig_baseline` passes, and
   `baseline_mu` must be a no-op under per-node scaling and must not be one under pooled scaling.

2. **My own recompute of every reported number**, from the 35 main archives, the 10 `__edges`
   archives and the 10 `__local` archives. I re-derived the channel shares, the lag-band shares, the
   completeness maxima and where they sit, T1/T2/T3, the IG-vs-occlusion agreement tallies, the
   neighbour-influence table and the local case. All of it matches
   `results/reports/explain_report.txt` as regenerated on disk.

3. **A separate check of the neighbour-hub finding**, not reusing `explain.py`'s
   `ebola_neighbours()`.

4. **Mutation tests**, because a verifier that cannot fail proves nothing. Scaling dengue's `err`
   array by 0.1 moved the reported worst cell from 0.0155 to 0.0016 and the ratio against the
   next-worst panel with it. Handing `randctl_verdict()` a failing draw of r = 0.30 flipped its
   wording to "NOT shown to be architectural". Both checks are sensitive, not vacuous.

I did not rerun the throwaway 219-check verification script mentioned in
`G5_Explainability_Goals.md:93-95`. It is not in the repo, by design, so I cannot rerun it and do not
claim its numbers as mine.

---

## 2. The IG baseline, as it now stands

The reference point that IG and occlusion measure against is built in one place, `ig_baseline()`
(`explain.py:167`), and both reads take it, so the faithfulness comparison stays a comparison of the
same counterfactual. It is **not** a zero baseline any more, and it is **not** fully moved either.
Per channel:

| channel | baseline | why |
|---|---|---|
| incidence | `baseline_mu(b)`, each node's own mean incidence in model space over the cells its scaler was fit on (train for the development panels, support for Ebola) | "this district at its typical level" is the counterfactual the report claims to measure against |
| sin_doy | **0** | the centre of the cyclic encoding, meaning no seasonal phase. Its fit-window mean is an arbitrary point that moves with window length, so it would encode a phase rather than a neutral reference |
| cos_doy | **0** | same |
| obs_mask | **0**, "not reported" | its fit-window mean is a reporting rate, about 0.6 on Ebola, a value the mask itself never takes. 0 is a state the channel really has, and it is what `window_slice()` already pads with |

So one channel of four moved. Say it that way. Three are still at zero, and the obs_mask artefact in
section 8 is a direct consequence of that choice.

**Why the incidence channel had to move.** The development panels use
`scaler_scope=per_node_train`, so z = 0 already is each node's own typical level. I measured the
per-node z-mean over the fit cells directly from the bundles: **exactly 0.000000 on
influenza_japan, influenza_us-regions, influenza_us-states and covid_us-states**. Ebola uses
`per_disease_support`, one pooled (mean, std) broadcast to all 61 districts, so z = 0 there is the
*disease* mean and the districts sit well off it. Measured from the `baseline_mu` array now archived
inside every Ebola run: **18 of 61 districts have a support cell and get a non-zero reference, mean
|mu| 0.7021 over those 18, max 1.3516**. The other 43 districts have no support cell at all, so 0 is
the only reference available for them and they legitimately keep it. Identical across all five
seeds, as it must be, since it is a property of the data and not of the model.

Dengue sits in between and is the one panel this is awkward for: it is `per_node_train`, but its
reference is not exactly zero. Measured, 5,934 of its 7,165 nodes carry a non-zero mu, mean |mu|
0.0441 over all nodes and 0.0218 over the 6,720 nodes ever scored in the test fold, max 1.0749. See
section 7 for what that does to the numbers, which is very little.

---

## 3. Goal-by-goal status (against `G5_Explainability_Goals.md` sections 2 and 3)

| # | goal | artifact | status |
|---|---|---|---|
| 1 | Global read: channel + lag-band shares, 7 panel-arms, 5 seeds, seed-mean and per-(seed,horizon) | `explain__<panel>__seed<s>.npz`, 35 of 35 present | **built, reverified today** |
| 2 | Faithfulness: IG vs occlusion agreement | printed in `--report` | **built, reverified today** |
| 3 | Spatial: Ebola edge ablation, degree held fixed | `*__edges.npz`, 10 files | **built, reverified today**, see section 6 |
| 4 | Local: Montserrado's own h3 forecast at the 2014-10-25 peak | `*__local.npz`, 10 files | **built, reverified today** |
| 5 | Gate figure placed in a document | `figures/gate.pdf` exists | **exists, still cited nowhere.** I grepped `Reports/Manuscript_v2.md` for `gate.pdf` and `gate.png`: zero hits. Still open. |
| 6 | Paper section, SHAP retracted, IG named throughout | `Reports/Manuscript_v2.md` section 9.8 | **not done.** Line 508 is still `[PENDING. A minimal global attribution ... is scoped and will be reported here. This version claims no attribution result...]` *(checked on disk 2026-09-10)*. Manuscript edit, out of scope for this internal doc. |

Goals 1 through 4 are code-complete and their outputs reverify. Goals 5 and 6 are writing work that
has not happened.

---

## 4. Global read, as recomputed from the archives today

Completeness is one residual over two denominators. `gap` = residual / |f(x) - f(0)|, the standard IG
ratio. `err` = residual / (|f(x)| + |f(0)|), the same residual against the size of the forecast.
Read `err` first, for the reason section 5 measures.

| panel | steps | cells | IG err (worst) | IG gap (worst) | incidence share |
|---|---|---|---|---|---|
| dengue | 64 | 63,753 | 0.0155 | 0.2967 | 0.630 +- 0.024 |
| influenza_japan | 32 | 4,136 | 0.0032 | 0.0635 | 0.461 +- 0.033 |
| influenza_us-regions | 32 | 920 | 0.0017 | 0.0931 | 0.497 +- 0.040 |
| influenza_us-states | 32 | 4,361 | 0.0024 | 0.0505 | 0.405 +- 0.020 |
| covid_us-states | 32 | 4,018 | 0.0003 | 0.0375 | 0.446 +- 0.035 |
| ebola_L12 | 32 | 2,930 | 0.0017 | 0.0645 | 0.564 +- 0.050 |
| ebola_L12_zeroshot | 32 | 2,930 | 0.0005 | 0.0016 | 0.570 +- 0.046 |

Ebola lag bands, IG share with the occlusion share beside it: `ebola_L12` 0.725 [0.697] on lags 1-5,
0.124 [0.122] on 6-10, 0.048 [0.052] on 11-15, 0.103 [0.129] on 16-20. `ebola_L12_zeroshot` 0.761
[0.756], 0.118 [0.110], 0.042 [0.045], 0.080 [0.089].

**Falsification tests**, all three reverified today from the archives:

- **T1** (incidence is the top channel): **PASS** on all 7 panels at seed-mean, horizon-averaged
  granularity. At the finer (seed, horizon) grain it **fails in 13 of 140 cells**: `influenza_us-states`
  9, `covid_us-states` 3, `influenza_us-regions` 1, all at h10 (6) and h15 (7), the winner being
  `sin_doy` 12 times and `cos_doy` once. Both granularities are printed, not just the passing one.
  Unchanged by the fix, since no failing cell is on an Ebola arm.
- **T2** (recent lags 1-5 outweigh 16-20): **PASS** on every panel. dengue 0.51 vs 0.14,
  `ebola_L12` 0.72 vs 0.10, `ebola_L12_zeroshot` 0.76 vs 0.08.
- **T3** (influenza seasonality share exceeds Ebola's on both arms): **PASS**. Lowest influenza value
  0.385 (`influenza_us-regions`) against the higher Ebola value 0.275 (`ebola_L12`); the zero-shot arm
  is 0.266. The margin narrowed slightly under the fixed baseline, from 0.385 vs 0.271 to 0.385 vs
  0.275, and the test still passes.

**Faithfulness.** Recomputed: seed-mean **54 of 56** panel-horizon cells agree on both the top channel
and the top lag band; per (seed, horizon) **263 of 280**. The two seed-mean misses are named, not
smoothed over: `covid_us-states` h5 and `ebola_L12_zeroshot` h15, both IG picking incidence where
occlusion picks sin_doy. Per-seed disagreement concentrates on `covid_us-states` (7) and
`ebola_L12_zeroshot` (4); dengue 2, `influenza_us-states` 2, `influenza_japan` 1,
`influenza_us-regions` 1. Both tallies are unchanged by the fix.

**Random-weight control.** I reran it directly rather than reading it off the report. An untrained
encoder reproduces the trained lag comb at r = 0.962, 0.963, 0.951, 0.963, 0.912 across the five
fixed draws, against a shuffle ceiling of |r| below **0.609** in 95 of 200 shuffles. Every draw clears
it, so the comb is architectural. Comb teeth, mean share at lags 1/5/9/13/17 against 4/8/12/16/20:
trained 0.123 vs 0.013, untrained 0.132 vs 0.004. This is why the report and the figure show lag
**bands** and not single lags. Reporting at single-lag resolution would publish a TCN dilation
fingerprint as an epidemiological memory effect.

---

## 5. The counterintuitive one: on ebola_L12, err improved and gap doubled

Panel-level, five seeds, worst cell over origins, horizons and seeds:

| arm | err before | err after | gap before | gap after |
|---|---|---|---|---|
| `ebola_L12` | 0.0031 | **0.0017** | 0.0333 | **0.0645** |
| `ebola_L12_zeroshot` | 0.0008 | **0.0005** | 0.0015 | **0.0016** |

The "after" column is my recompute from today's archives. The "before" column is read from the
pre-fix report committed at `9223e1b`, since the pre-fix archives were overwritten by the rerun. I did
not take the before column on trust, see the reproduction below.

**The reading is that this is expected and benign, and the data supports it.** I checked it rather
than asserting it, by rerunning IG on `ebola_L12` seed 72, the seed that carries the worst gap, over
its own 18 archived origins under BOTH references with everything else identical:

| quantity, seed 72 | zero baseline | fixed baseline |
|---|---|---|
| err, worst cell | 0.0019 | 0.0017 |
| gap, worst cell | 0.0019 | 0.0645 |
| residual, worst cell | 0.01759 | 0.01255 |
| residual, median over cells | 0.00948 | 0.00334 |

The recompute under the fixed baseline reproduces the archived seed-72 numbers exactly (gap 0.0645,
err 0.0017), so the two columns are the same code on the same inputs with only the reference moved.

**The residual fell in 72 of 72 cells.** The attribution did not get worse. It got better, because
the path from the baseline to the data is shorter, so 32 Riemann steps approximate it more closely.

What moved instead was gap's denominator. Measuring |f(x) - f(0)| directly, median over the 18
origins:

| horizon | gap denominator, zero baseline | gap denominator, fixed baseline | ratio |
|---|---|---|---|
| h3 | 36.10 | 30.60 | 0.85x |
| h5 | 20.03 | 15.52 | 0.78x |
| h10 | 35.94 | 26.40 | 0.74x |
| h15 | 6.65 | 3.68 | 0.55x |

It shrank in 72 of 72 cells, and the worst cell is far past the median: the smallest gap denominator
anywhere fell from 3.170 to **0.0359**, about 88x smaller. That single cell is where the 0.0645
comes from. err's denominator does not collapse the same way. It moved the other direction at three
horizons, 1.09x at h3, 1.10x at h5, 1.12x at h10, and only shrank at h15 (0.55x), yet err improved at
every horizon. So err's improvement is a real improvement in the residual, not a denominator artefact
running the other way.

This is exactly the failure mode `integrated_gradients()`'s own docstring (`explain.py:187-202`)
warns about: gap reads as broken attribution wherever the model's own output barely moves off the
baseline, which is a flat model, not a broken attribution. Moving the baseline closer to the data
makes the model look flatter by construction. **Read err. Do not quote gap on Ebola without this
paragraph beside it.**

One honest limit on this section. I reproduced the before/after comparison on seed 72 only, not on
all five seeds, because that is the seed that carries the maximum and one seed was enough to
separate the two explanations. The 0.0031 and 0.0333 panel-level "before" figures are from the
committed pre-fix report, not from an archive I still hold.

---

## 6. Ebola neighbour ablation, and the identical-output check

Recomputed from the 10 `__edges` archives, pooled over origins and horizons. Zero-shot districts (43)
have a higher mean relative neighbour influence than observed districts (18) on both arms: **0.143 vs
0.122** on `ebola_L12`, **0.100 vs 0.084** on `ebola_L12_zeroshot`. Medians 0.137 vs 0.116 and 0.096
vs 0.082. **23 of the 43** zero-shot districts draw their single strongest neighbour signal from an
observed district, on both arms.

**Those 23 slots do not funnel through one hub.** Ten distinct observed districts share them, with an
identical distribution on both arms:

```
guinea|macenta 5, liberia|nimba 4, guinea|dinguiraye 3, guinea|boffa 2, guinea|conakry 2,
liberia|montserrado 2, sierra leone|bo 2, guinea|gueckedou 1, guinea|telimele 1, liberia|margibi 1
```

Said plainly: the model is not routing every unobserved district's forecast through a single node. It
draws from a spread of 10 hubs, and the spread is stable across the few-shot and zero-shot arms,
which use different adapters over the same trunk.

**The identical-output check.** Every number in this section came out the same as the pre-fix run, to
the precision the report prints. I checked this rather than assuming it: I diffed the pre-fix report
committed at `9223e1b` against the regenerated one on disk, and both neighbour-ablation blocks are
line-for-line unchanged, including all 20 rows of the two district tables and every relative-influence
value. That is the right answer and it is evidence, not a coincidence. Edge ablation swaps the
adjacency and reads the forecast; it never touches the IG reference. `edge_ablation()`
(`explain.py:272`) takes no `mu` argument at all, and `run_panel` calls it without one. So a change
confined to the IG baseline must leave it untouched, and it did. The same argument covers the local
case study's forecast numbers, which come from the same call: `liberia|montserrado`, own forecast at
the 2014-10-25 national peak week, seed mean, h3 = 31.4, h5 = 19.1, h10 = 30.0, h15 = 4.3 cases per
week against 1,428 observed. Unchanged, as it should be.

What did change in the local archive is the attribution map itself, `ig_map`, which reads the
baseline. It is not printed in the report, so nothing published moved. The assertion at
`explain.py:722-723` still confirms `ig_map` differs from `ig_map_pooled`, so panel C of the figure is
the district's own forecast and not the national sum.

**This is not an accuracy claim and must not be read as one.** The gate-off ablation
(`Reports/gate_ablation.log`) found the spatial channel helps error in 0 of 40 cells and hurts in 8.
Neighbour attribution shows where the model draws from, not what makes it right. The 0 of 40 is also
a bound on neighbour information specifically, not on the graph overall, since the gate-off run keeps
the LTR degree feature live.

---

## 7. Blocker status

**(a) Manuscript and figure placement. Open, untouched.** `figures/gate.pdf` is cited in no document,
and `Reports/Manuscript_v2.md:508` is still a `[PENDING]` placeholder. Both verified on disk today.

**(b) Dengue IG completeness. PARTIAL, reduced but not closed.** Dengue's worst completeness error
fell from **0.0444 to 0.0155** (seed 52, origin index 22, h15, recomputed from
`explain__dengue__seed52.npz`). Its worst gap fell from 0.7114 to 0.2967. That is a 2.9x improvement
and dengue is still **4.77x the next-worst panel** (`influenza_japan` at 0.0032), so it remains the
outlier and the cause named in the scope note, under-integration on a 7,165-node surface, is reduced
rather than removed. Doubling the steps again would presumably help again, at about 10 s per origin
per seed against 5 s now.

**The confound I was told to expect is not there, and the real situation is slightly different.** I
checked whether the dengue gain belongs to the extra steps or to the new baseline, by rerunning IG on
the exact cell that produced the old worst number under all four combinations:

| dengue seed 52, origin idx 22 | err at h15 |
|---|---|
| 32 steps, zero baseline | 0.04441 |
| 32 steps, fixed baseline | 0.04436 |
| 64 steps, zero baseline | **0.01551** |
| 64 steps, fixed baseline | 0.01551 |

The step count carries the whole gain. The baseline moves it by about 0.1 percent. The 32-step
zero-baseline recompute also reproduces the previously reported 0.0444 exactly, which is a clean
reproduction of the pre-fix number.

The 64-step zero-baseline row is also the row that matches the archive, at h15 and at every other
horizon, and at the gap maximum of a second origin where the two references genuinely differ (0.2967
archived, 0.2967 zero, 0.2130 fixed). **So the dengue archive was produced under the OLD zero
baseline**, not the fixed one. That is consistent with the file times, dengue written 16:04 to 16:21
and `explain.py` last modified at 22:12, and with the archives themselves: the two Ebola runs carry a
`baseline_mu` record that the current code writes for every panel, and none of the five development
panels do.

What that means, stated straight: the published table mixes two baseline conventions. The five
development panels are zero-baseline; the two Ebola arms are fixed-baseline. On influenza and COVID
that is a distinction without a difference, because I measured their per-node reference at exactly
0.000000, so the fix is provably a no-op there. On dengue it is a small real difference that I
measured at two origins and found negligible for err and visible for gap at one of them. It is worth
one rerun of the five development panels for tidiness, about 22 minutes, and it is not worth blocking
anything on.

**(c) Ebola zero-baseline bug. CLOSED.** `integrated_gradients()` and `occlusion()` both take
`ig_baseline()` now, the incidence channel sits at each district's own support-window mean, the
reference is archived in every run so a reader can inspect it, and the selfcheck contains a
deliberate-failure test that a zero baseline must fail. The measured consequence is in section 5:
err improved, gap rose for a denominator reason that is understood and documented.

**(d) The three client decisions. OPEN, untouched.** `G5_Explainability_Goals.md` section 8 lists
adopting IG, retracting the SHAP row from the client-held comparison table, and confirming the
honesty framing on neighbour attribution. I found no newer decision record on disk referencing them,
so I am treating all three as still open. A missing record is not proof they were not answered, so
this is a "not found", not a "not answered".

---

## 8. The obs_mask artefact, still live, stated rather than fixed

`obs_mask` is constant 1.0 on the three influenza panels and on COVID. Nothing in those panels varies
it, so no finding about the model can rest on its share there. The scope note (5.2) predicted that
share would be structurally zero, because a constant input has no gradient. Measured, it is not:
**up to 0.182 by IG and 0.150 by occlusion**, the 0.182 being `covid_us-states`. The three influenza
panels sit at 0.112, 0.118 and 0.113. The scope note's prediction is refuted by the run.

The cause is the baseline, and our own per-channel decision keeps it. A path method charges obs_mask
for the entire move from "nothing observed" to "everything observed", a counterfactual those panels
never contain. Since we deliberately left obs_mask at 0 (section 2), the fix to the incidence channel
does not touch this and the artefact is unchanged.

**There is no clean single answer, which is why this is stated and not fixed.**

What moving obs_mask to its panel mean would get: it would remove the artefact on the four
constant-mask panels outright, because the reference would equal the data and the channel would
contribute nothing.

What it would cost: it would make Ebola worse. Ebola's fit-window mean for that channel is a
reporting rate of about 0.6, and the mask is a 0/1 indicator that never takes the value 0.6. The
baseline would become a state the data cannot be in, on the one panel the whole project is about.

My call: leave obs_mask at 0 and print the artefact, because a wrong number that is labelled is safer
than a clean number that is quietly wrong on the panel that matters. The report already says so in
its own text. Anyone quoting an obs_mask share on influenza or COVID is quoting an artefact.

One wording note for whoever edits the report next. Its obs_mask paragraph says "Both read from a
zero baseline". That sentence is still literally true on the four constant-mask panels, since their
measured per-node reference is exactly 0.000000 for every channel. It is no longer true as a general
statement about the module, and it should be narrowed to those panels before the paragraph is reused
anywhere else.

---

## 9. What I could not verify, and what is now stale

- **`Reports/explaiability_report.md` is stale.** It is dated 2026-09-08 and predates both reruns. It
  differs from the regenerated `results/reports/explain_report.txt` on 27 lines. The previous version
  of this document claimed the two were byte-identical; that claim was true when written and is false
  now. Regenerate it before anyone quotes it. I did not edit it, since report rendering is not this
  document's job.
- **The 219-check throwaway verification** from `G5_Explainability_Goals.md:93-95` is not in the repo
  and cannot be rerun. My mutation tests in section 1.4 are a real but smaller substitute.
- **The pre-fix per-cell archives are gone**, overwritten by the reruns. Where I quote a pre-fix
  number I say where it came from, and for the two that carry weight, dengue's 0.0444 and the
  ebola_L12 gap behaviour, I reproduced the pre-fix condition by rerunning under the zero baseline
  rather than trusting the old text.
- **Whether the three client decisions in section 7(d) have been answered** since 2026-09-07. Not
  found on disk. Treated as open.

---

## 10. Overall verdict

**PARTIAL, matching Gap Ledger E1. Not DONE.**

Blocker (c) is closed. Blocker (b) is reduced from 0.0444 to 0.0155 but dengue is still 4.77x every
other panel, and blocker (d) has not moved at all. Two of four blockers being open is what keeps this
at PARTIAL. Anyone reading a closed blocker as a closed goal would be reading it wrong.

What is solid: the code exists, runs, and reverifies clean end to end. `--selfcheck` catches its own
deliberately-broken inputs, including two new checks aimed at the baseline itself. Every table number
recomputes from the archives with independent code. T1, T2 and T3 pass, the IG-vs-occlusion agreement
is 54 of 56 with both disagreements named, the neighbour-hub finding holds under an independent
mutation-tested check, and the neighbour numbers came out identical across the fix in exactly the way
the code says they must.

What is still missing, and this document closes none of it:

1. Dengue IG completeness at h15 is 0.0155, improved but still the outlier.
2. `figures/gate.pdf` exists but is cited in no document.
3. `Reports/Manuscript_v2.md:508` is still a `[PENDING]` placeholder and the SHAP-to-IG correction is
   not written into it.
4. The three client decisions in `G5_Explainability_Goals.md` section 8 are unanswered as far as I can
   find on disk.
5. `Reports/explaiability_report.md` needs regenerating from the current archives.
6. The five development panels are still on the old zero baseline while the Ebola arms are on the
   fixed one. Harmless on influenza and COVID by measurement, near-harmless on dengue by measurement,
   untidy on paper.

None of the findings here should be read against the model's transfer performance. Every
neighbour-attribution and channel-attribution number in this document describes what the model reads,
not whether reading it helps. The gate-off ablation already answered the accuracy question
separately, and negatively, in 0 of 40 cells.

# G5 Explainability: Goals and Acceptance Criteria

**Written 2026-09-07, revised the same day after the ML review.** Ledger item E1
(`Gap_Ledger.md:286`). Decision record: `progress/decisions/G5_Explainability_Scope.md`.
*(disk)* = I recomputed it from `results/explain/` and the report.

---

## 1. What G5 is required to deliver

The brief requires it twice, `Final Internal Project Brief.md:74` and `:119`, "Produce
explainability outputs (SHAP global + local) **[REQUIRED]**". It is half of intersection property
(d), "calibrated and explained" (`PROJECT.md:23`).

**The brief's word is SHAP. The recorded decision is integrated gradients**
(`G5_Explainability_Scope.md:60-61`), taken after the client called SHAP over 7,165 regions "a
serious compute proposition" (`Review Doc.md:83`).

## 2. Goals

1. **Global read.** Attribution share per channel (4) and per lag **band** (4), per horizon, 7
   panel-arms at 5 seeds, printed at both seed-mean and per-(seed, horizon) granularity.
2. **Faithfulness.** Occlusion agrees with integrated gradients on the top channel and top lag band,
   per panel-arm and horizon, at seed-mean and per-seed. Disagreements are named, not resolved.
3. **Spatial.** Ebola edge ablation, degree held fixed so only the neighbour message moves, over 43
   zero-shot (no support labels) and 18 support-carrying districts of 61.
4. **Local.** One district at the 2014-10-25 national peak: `liberia|montserrado`, largest at the
   origin whose h3 target is the peak week, support-carrying *(disk)*. The IG target is that
   district's **own** median forecast per horizon, not the national sum.
5. **Gate figure.** `figures/gate.pdf` exists from `gate_figure.py`; no document places it.
6. **Paper section**, SHAP row retracted, method named as integrated gradients throughout.

## 3. Acceptance criteria

| goal | artifact | check that proves it |
|---|---|---|
| 1 | `explain__<panel>__seed<s>.npz`, 7 x 5 | selfcheck (1): IG completes the path, gradient x input fails the same assertion |
| 2 | agreement block in the report | seed-mean 54 of 56, per-seed 263 of 280 *(disk)*, both printed, the 2 misses named |
| 3 | `*__edges.npz` | selfcheck (3): gate-off cannot move under ablation, `forward_fixed_deg` holds degree |
| 4 | `*__local.npz` | selfcheck (4c): the target is that node's own forecast, not the pooled sum |
| 5 | `figures/gate.pdf` | cited in the manuscript |
| 6 | manuscript, corrected table | `check_unsafe_claims.py` catches a reinstated SHAP claim |
| all | `progress/outcomes/G5_Explainability_Results.md` | a verifier that recomputes every number from disk, mutation-tested |

`conda run -n ebola-train python -m explain --selfcheck` / `--report`.

## 4. What the goals may not claim

| constraint | one line |
|---|---|
| 5.1 | Neighbour attribution is *where the model draws from*, never *what makes it accurate*: gate-off helps error in 0 of 40 cells, correlation in 6 of 20. It keeps the LTR degree feature, so 0 of 40 bounds neighbour information, not the graph. |
| labels | Zero-shot means no support labels, not unobserved: 36 of the 43 are observed at some explain origin *(disk)*. |
| 5.2 | `obs_mask` varies only on dengue and Ebola, a partial disease identifier inside a disease-agnostic block. |
| 5.3 | Attribution describes a model whose transfer mechanisms did not help, and must not imply otherwise. |
| lags | No claim may rest on a single lag. Only the four bands are reportable. |

**New finding.** Scope note 5.2 predicted `obs_mask` attribution would be structurally zero on the
constant panels. It is not: seed-mean 0.182 by IG and 0.150 by occlusion on `covid_us-states`, 0.250
at worst *(disk)*. The zero baseline charges it for a 0 to 1 move those panels never contain, so
**5.2 needs a dated amendment.**

## 5. Pre-stated falsification tests

In `explain.report()`:

- **T1.** Incidence is the top channel. On horizon-averaged seed-mean shares it passes on all 7
  panel-arms. Per (seed, horizon) it fails in 13 of 140 cells, sin_doy 12 and cos_doy 1, all at h10
  and h15: us-states 9, COVID 3, us-regions 1 *(disk)*. Both are printed.
- **T2.** Lags 1-5 outweigh lags 16-20 on every panel-arm.
- **T3.** Seasonality share (sin + cos) on each influenza panel exceeds each Ebola arm. It compares
  trunks as well as diseases, single-disease encoders against the all-development one.

**If they fail, distrust the attribution before the epidemiology** (`Phase3_Developer_Execution_Guide.md:321`).

## 6. Out of scope

| excluded | reason |
|---|---|
| KernelSHAP | does not fit dengue at 7,165 nodes over 1,409 steps, and a sampled gradient approximation is not Shapley |
| Attention weights | not faithful; rejected at `Phase3_Developer_Execution_Guide.md:318` C |
| Any retraining | 118 checkpoints make this pure inference |
| Any Ebola re-scoring | scored once under a frozen pre-registration |
| Any accuracy claim for the spatial channel | forbidden by the gate-off result above |

## 7. Definition of done

Ledger rule (`Gap_Ledger.md:11-12`): DONE means committed, and a verifier re-derives every number.

1. `--all` over 7 panel-arms x 5 seeds archived. **Done: 35 of 35** *(disk)*, dev panels at 24
   origins and 32 IG steps, Ebola at 18 origins.
2. The 10 Ebola archives rebuilt 2026-09-07 after 9(b), 5.2 min, report and figure regenerated. T1,
   T2 and T3 all PASS seed-mean, agreement 54 of 56 *(disk)*.
3. `progress/outcomes/G5_Explainability_Results.md`: **still owed.** Verification has run, 219
   checks, 0 failures, 6 of 6 deliberate corruptions caught, but on a throwaway script. A committed
   verifier is part of this item.
4. Ledger E1 closed with the commit. `results/**/*.npz` is untracked, so the tracked evidence is the
   report and the outcome document.
5. The three client decisions recorded, answered or open.

## 8. Open decisions with the client

| # | decision | status |
|---|---|---|
| 1 | Adopt integrated gradients, or renegotiate G5 | **Open.** Taken internally under C3 |
| 2 | Retract the SHAP row from `RelatedWork_CompetitiveAnalysis_Benchmark.docx` | **Open, urgent.** It asserts a capability we do not have (M12, `Phase0_to_Now_Audit.md:260`) |
| 3 | Accept a neighbour figure that cannot be described as explaining accuracy | **Open.** No answer recorded |

## 9. Review findings (2026-09-07)

a. **CLOSED. The per-lag read was a TCN fingerprint.** Fixed by reporting lags at band level only
in the table and figure panel B, keeping the 20-lag array in the archives, and adding a
random-weight control to `--report`, about 8 s and skippable with `--no-randctl`. **The number
that proves it:** untrained encoders reproduce
the trained `ebola_L12` lag profile at r = 0.912 to 0.963 over 5 draws, against a chance ceiling of
|r| below 0.63 in 95 of 200 shuffles. Spike to trough the comb runs 10.5x trained, 37.1x untrained
*(disk)*. The report's verdict sentence is generated by `randctl_verdict()` from those two numbers
rather than written as fixed prose, so a run where the control failed would print the withheld
wording instead; `--selfcheck` feeds it a failing draw and checks the wording flips.

b. **CLOSED. The local panel attributed the wrong quantity.** IG targeted the sum over all scored
nodes, so Montserrado's map showed its effect on the national total. Fixed with one extra one-hot IG
pass at the case origin, archived as `ig_map` with the old map kept as `ig_map_pooled`; both Ebola
arms rerun. **The number that proves it:** panel C's title and the
caption print Montserrado's own h3 forecast, 31.4 cases/week seed-mean, against 1,428 observed for
2014-10-25 *(disk)*, with the gap-lumping cause named (`client_decisions.md` A4). Two checks hold it:
`--selfcheck` asserts a one-hot target completes on that node's own forecast and NOT on the pooled
one, and `--report` asserts the archived `ig_map` differs from `ig_map_pooled`, which is what a
pre-retarget archive on disk would fail. The 31.4 did not change with the retarget and was never
expected to: the retarget changes what IG explains, not what the model predicts.

c. **OPEN. Dengue IG is under-integrated at 32 steps**, completeness err 0.0444 at h15 against
0.0032 on the next worst panel *(disk)*. Rerun at 64 steps, about 12 minutes, and confirm err falls:

```
conda run -n ebola-train python -m explain --panels dengue --steps 64
conda run -n ebola-train python -m explain --report
```

d. **OPEN. The zero baseline is not the training mean on Ebola.** Averaged over every week, observed
or not, `ebola_L12` district means sit 0.32 from zero and 1.43 at worst under the pooled scaler,
against 0.16 on `influenza_japan` *(disk)*. That convention counts unobserved weeks as zero and so
understates it: over the weeks a value is actually read the figures are 0.80 and 2.37, while
`influenza_japan` is unchanged at 0.16 because its mask is constant 1. Either way the note at
`explain.py:38-41` does not hold on Ebola. Needs a one-line change at
`explain.py:162`, the zero `base` becoming the per-node training mean, then about 30 minutes:

```
conda run -n ebola-train python -m explain --panels ebola_L12 ebola_L12_zeroshot influenza_japan
conda run -n ebola-train python -m explain --report
```

Compare channel shares against this run before adopting either baseline.

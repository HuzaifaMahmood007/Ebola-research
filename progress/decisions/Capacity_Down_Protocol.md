# Capacity probe, downward half: protocol

Date: 2026-09-29. Written before any result of this run exists, and committed before the run starts.
The runner (`diagnostics/capacity_probe.py --down`) refuses to fit unless this file is committed with no
local edits. It stamps this file's sha256 into every per-seed cache, and the report refuses to decide
if any hash differs or any seed is missing. The numbers below are recomputed from disk by
`diagnostics/verify_capacity_down.py`, except the few cited from earlier documents or from the pre-run
S2 step check, which its docstring lists.

## What this run asks

The capacity probe only ever swept upward from the shipped 1,428 parameter affine adapter. The later
mechanism evidence (Adapter_Mechanism 2026-09-16, Adapter_Constraint 2026-09-22) points the other way:
too much freedom, not too little. A reviewer will ask why we never went down. This is the downward
half: each size is derived from the data budget, the decision rule is fixed here before any number
exists, and the whole thing is one overnight command.

Decisions already taken: test in **both regimes, inside the probe**; the verdict uses an **agreement
rule, no Bonferroni**.

## Corrections to the brief, all checked on disk

1. **Each surface adds 12 cross-disease and 4 in-domain cells, not 36 and 12.** 36 and 12 were the
   totals over three upward surfaces (`results/misc/capacity_probe_5seed.json`: 3 flu panels x 4
   horizons, and dengue x 4).
2. **The probe is the dengue-to-flu direction on its own five trunks**
   (`results/lodo/encoder_ldo__dengue2flu-cap__seed*__ckpt.pt`), not LDO3. Both hand the adapter the
   full train fold.
3. **The head-only surface (1,300) is not a smaller capacity.** With the trunk frozen the shipped
   adapter is exactly affine, and gamma and beta fold into the head (`models/adapters.py:14-16`), so it
   has 1,300 effective parameters. Head-only is the same function class, so it is a parametrisation
   control, not a rung. Real downward capacity starts below 1,300.
4. **The Ebola few-shot adapter starts from a fresh random init, not from the borrowed mean**
   (`train/ebola.py:106-148`, `init_state=None`). The "few-shot change" is the gap between two maps
   that were fitted separately.
5. **Small anchored surfaces cannot go through the Ebola fitting protocol.** `_fit_adapter` hard-codes
   `Adapter()` (`train/ebola.py:124`), so it cannot fit any other module. Its budget is also far too
   small for one: with 10 support origins (L12) or 18 (L20; the plan said 15), stepping every 8
   origins gives 2 or 3 steps per epoch, so a few hundred AdamW steps at lr 1e-3 under a cosine
   schedule, which would leave an anchored recalibration close to where it started. Small surfaces
   therefore get their own fit (section 2).
6. **A literal ridge saturates** under the gradient clip at 1.0, reaching only about 64 per cent of the
   way to the anchor (Adapter_Constraint, "Why shrinkage and not a literal ridge"). The shrinkage
   candidate therefore uses the closed-form interpolation, not a penalty term.
7. **Under the agreement rule below, the upward half has no clean win.** Every larger surface that
   helped at h15 (on japan plus a US panel) also hurt influenza_japan h3 significantly, so all three
   read MIXED. That comes from the archive at zero compute. The per-cell counts are 7 better, 3 worse,
   36 cells, all 7 at h15, all 3 at japan h3, best +19.8 per cent. mlp-64 japan h15 only just clears,
   with a lower bound of +0.03.
8. **The one development simulation on record did not reproduce the Ebola damage.** In
   `diagnostics/fewshot_sim.py`, covid, seed 42, with the Ebola geometry (t0-blind), adapting *helped*
   by 37 per cent on L12 and 35 per cent on L20 (country-macro MAE, fresh against zero-shot,
   `results/misc/fewshotsim__summary__seed42.json`). Only the mid-window placement hurt. This drives
   the most likely outcome below.
9. **The archived affine control cannot be reproduced bit for bit.** `_fit_shared_adapter` never seeds
   the adapter init (`train/lodo.py:278`), and every archived seed fitted its adapters straight after
   training its trunk in the same process. The downward half therefore refits its own affine control,
   seeded, and reports the drift from the archive. The drift is not a gate.
10. **`SURFACES` cannot simply be extended.** `train/anil.py:957-963` reads it, five places in the probe
    treat `SURFACES[0]` as the control, and a merged list would make any new or `--force` seed write
    6-rung caches beside the archived 4-rung ones. The new rungs go in a sibling list, `DOWN_SURFACES`,
    with the same control label.
11. **r* came out at 10, not 3 or 5.** The brief expected 272 or 440 parameters from the single-disease
    participation ratios (1.8 to 2.5 on flu, 4.6 on dengue). On the cap trunks, which are dengue
    trained, dengue itself reaches 9.017. The rule takes the maximum over all four panels, so r* = 10 and
    the rung has 860 parameters. Section 2 states what that costs.
12. **One luck-alone figure was mis-rounded.** Both panels clearing when independent is 0.0006, not
    0.0007: the exact value is 0.025 squared, 0.000625. Under a seeded Monte Carlo (section 3) the
    other figures reproduce within its noise; to two significant figures head-only's false DIFFERS
    rate is 0.019 (the brief said 0.018) and the whole sweep 0.028 (the brief said 0.03).
13. **"357 node-seasons" counts nodes x train origins / 52.** Counted by train weeks instead it would be
    402. The district bound below uses 357.
14. **The archived cross-disease fit averages 4.2 min per surface, not 4.3**, so regime F is about 81
    min per seed, not 82.
15. **The brief defines NULL as "no cell clears", which leaves one case open**: a significant gain in
    one panel that no second panel agrees with, and no harm. It reads NULL here, because agreement is
    the guard.

## 1. The sizing argument

**The output side is fixed.** The adapter emits 20 numbers per node, 4 horizons x 5 quantiles. The
masked pinball loss splits by horizon, so each horizon's 5 output rows are fitted only by the pairs that
have an observed target at that horizon. **The budget is therefore per horizon.** The shipped map spends
5 x 65 = 325 parameters per horizon, and the 5 rows share one scalar target, so the data mostly informs
one location per horizon plus a spread.

**The rule.** For a linear read-out with p free parameters fitted on n observations, the expected
out-of-sample excess error is about sigma^2 p / (n - p - 1) at in-range query points (the standard OLS
prediction-risk result). Keeping that excess under 10 per cent of the noise gives n >= 11p, which is the
familiar "10 observations per parameter" rule from clinical prediction modelling. So **budget per
horizon B_h = n_eff / 10**.

**Amount of data.** The effective n sits between the number of districts (every pair in a district
counts as one observation) and the number of pairs (every pair independent). Both ends are reported.

**Type of data, part 1: extrapolation.** At a query point x the variance term is x^T (X^T X)^-1 x. It
grows with the square of how far x sits beyond the support's spread. Adapter_Mechanism measured that
spread ratio kappa at 5.2 to 11.2 (seed means of the per-seed medians,
`results/misc/ebolashift__adapter_mechanism.json`). A parameter that multiplies features therefore
costs kappa^2 times more, so B_h = n / (10 kappa^2). Intercepts do not multiply features, so kappa does
not apply to them.

**Type of data, part 2: feature rank.** A read-out needs no more rank than the features actually
carry. Step 0 (section 2) measures it on the capacity trunks themselves.

Budget per horizon (Ebola counts from `configs/ebola_arms.json`, kappa from Adapter_Mechanism):

| arm | h | pairs | districts | kappa | B, pairs bound | B, district bound | B for feature-acting params (pairs / 10 kappa^2) | affine 325 is over budget by |
|---|---|---|---|---|---|---|---|---|
| L12 | 3 | 48 | 17 | 5.42 | 4.8 | 1.7 | 0.16 | 68x to 191x |
| L12 | 5 | 38 | 14 | 5.84 | 3.8 | 1.4 | 0.11 | 86x to 232x |
| L12 | 10 | 18 | 9 | 9.08 | 1.8 | 0.9 | 0.02 | 181x to 361x |
| L12 | 15 | 0 | 0 | n/a | 0 | 0 | 0 | no data at all |
| L20 | 3 | 102 | 36 | 5.40 | 10.2 | 3.6 | 0.35 | 32x to 90x |
| L20 | 5 | 92 | 36 | 5.21 | 9.2 | 3.6 | 0.34 | 35x to 90x |
| L20 | 10 | 72 | 35 | 8.02 | 7.2 | 3.5 | 0.11 | 45x to 93x |
| L20 | 15 | 54 | 34 | 11.15 | 5.4 | 3.4 | 0.04 | 60x to 96x |
| dev full fold (flu, 3 panels) | any | up to 18,586 | 106 nodes, about 357 node-seasons | not measured | up to 1,859 | about 36 | not measured | affine sits inside [36, 1,859] |

**What the rule decides.**

- **Ebola regime.** The rule decides this regime outright. The affine is 32x to 361x over budget
  whichever bound you take. After the extrapolation penalty, **not a single feature-acting parameter is
  fundable at any Ebola horizon** (every value is below 1). Only output-side parameters are fundable:
  per-horizon intercepts, plus a per-horizon slope on the borrowed forecast where the pairs bound
  reaches 2.
- **Dev full-fold regime.** The rule does not decide this one. The affine sits between the bounds, and
  the upward sweep's h15 gains show the lower bound is too pessimistic for long multi-season series. So
  the ladder has to decide, and the only principled reason to go down here is the input-side rank.
- **Shrinkage prediction.** The rule predicts a shrinkage factor of about B / 325 = 0.015 (L12 h3) to
  0.03 (L20 h3), meaning almost fully back to the anchor. That agrees with the Ebola oracle
  (Adapter_Constraint: t = 0 recovers about 100 per cent) and disagrees with support-only selection
  (0.34 on L12, 0.88 on L20). It is stated here so it cannot be read in afterwards.

## 2. The candidate ladder

Each candidate runs **only in the regime it is sized for**.

### Step 0: the rank the trunk actually uses (measured 2026-09-29)

Participation ratio (sum s^2)^2 / sum s^4 of the centred trunk output that the adapter reads
(`enc(Z, A, M)`, as `_fit_shared_adapter` feeds it), pooled over nodes x train origins, dengue on every
10th train origin (`--measure-rank`, `results/misc/capacity_probe_rank.json`). Inference only; it took
under a minute for all 20 cells.

| panel | 42 | 52 | 62 | 72 | 82 | max |
|---|---|---|---|---|---|---|
| influenza_japan | 2.521 | 1.850 | 1.931 | 2.031 | 1.852 | 2.521 |
| influenza_us-regions | 3.515 | 2.509 | 2.952 | 2.812 | 2.745 | 3.515 |
| influenza_us-states | 3.817 | 2.518 | 3.032 | 2.950 | 2.729 | 3.817 |
| dengue | 9.017 | 6.608 | 6.914 | 6.208 | 6.724 | 9.017 |

**r* = ceil(9.017) = 10**, so the rank rung has 84 x 10 + 20 = 860 parameters. `R_STAR = 10` is written
into the runner.

What that costs, stated before the run: r* is set by dengue, the trunk's own disease. On the three flu
panels the largest ratio is 3.817, so for the cross-disease arm, which carries the verdict, rank-10
leaves about 6 directions beyond what the flu features use. A flu-only rule would give r = 4 (356
parameters) and a sharper test of "capacity beyond the used directions hurts transfer", at the price of
giving the in-domain dengue read-out less rank than its features carry. The rule was fixed as the
maximum over all four panels so that no panel is under-ranked, and it is kept.

### Regime F: full train fold (the existing probe protocol, unchanged)

Cap trunks, `_fit_shared_adapter` unchanged, fitted from scratch, cross-disease arm (3 flu panels) and
in-domain arm (dengue). The affine control is refitted in the same session, seeded before every fit.

| label | params | rule behind the size | if it wins | if it loses or ties |
|---|---|---|---|---|
| `rank-10` | 860 = 84 r* + 20 at r* = 10 | r* = ceil(max participation ratio over 5 seeds x 4 panels), Step 0 | Even with abundant data, capacity beyond the directions the trunk uses hurts transfer. That links the read-out to the encoder's low effective rank. | The trunk's low-energy directions carry signal. A low participation ratio does not mean a low useful rank. The shipped affine is right-sized for data-rich adaptation, as the rule could not rule out. |
| `head-only` | 1,300 | same function class as affine: a **control**, not a candidate | (verdict DIFFERS or SAME) FiLM's redundant parameters change the fit through the optimiser, so every rung must be read net of parametrisation | expected: SAME |

Dropped, with reasons:

- **Shared quantile head plus per-horizon offset (340).** It assumes the horizon only shifts the
  forecast. The upward sweep contradicts that: all its gains sit at h15 and all its losses at h3. It is
  also over every Ebola budget.
- **FiLM-only over a borrowed head (128).** It acts on features, so the extrapolation penalty makes it
  unfundable at Ebola budgets.
- **Rank-1 or higher in regime S.** Rank-1 has 104 parameters, more than 3x the largest total Ebola
  budget (L20, pairs bound, 32).

### Regime S: Ebola budget (the support sweep inside `--down`)

- **Support.** The real L12 and L20 support masks are transplanted onto influenza_japan and
  influenza_us-states with `fewshot_sim.transplant`, t0-blind, which is the geometry declared as
  "Ebola" before any result. The Ebola support-mask geometry is read, no labels: `template()` reads only
  the frozen arm's support-mask pattern, no Ebola value enters any fit or score, and
  `_fit_shared_adapter`'s C8 assert is untouched.
  - Support nodes are fixed across seeds (`NODE_DRAW_SEED = 0`), as Ebola's districts are.
  - Both panels are fully observed and share the covid channel layout, so blinding means what it did
    on covid. No window touches the block: the first test window starts at column 210 on japan and 218
    on us-states, and the first validation window at column 140 on japan and 146 on us-states, against
    a block ending at column 20.
  - us-regions (10 nodes) cannot host an 18-row block.
- **Anchor.** The dengue affine on the same trunk, which is exactly arm 2's refitted control. It stands
  in for Ebola's borrowed mean: fitted on the source disease, never on flu.
- **Scoring.** The panel's full test fold, country-macro, RMSE and MAE per horizon.

| label | params | rule behind the size | if it wins | if it loses or ties |
|---|---|---|---|---|
| S0 `zeroshot` | 0 | reference, not a candidate | n/a | n/a |
| S1 `recal-int` | 4 nominal, one intercept per horizon shared across quantiles (3 carry data on L12) | the largest surface fundable under the district bound (L12 h3/h5 and every L20 horizon; at L12 h10, 0.9, it is marginally over and kept for a fixed shape) | Recalibration in the large is enough at this budget, the first rung of the classic model-updating ladder | The budget funds nothing useful. Zero-shot or the affine is as good. |
| S2 `recal-budget` | L12: 5 (h3 slope+int, h5 slope+int, h10 int, h15 none); L20: 8 | the largest recalibration the pairs bound funds per horizon: slope plus intercept where n/10 >= 2, intercept where >= 1, nothing where 0 | The surface size the data funds beats the pre-registered one. That is the direct answer to the reviewer. | Sizing to the budget is not the lever, at least on dev |
| S3 `shrink-t` | the fresh affine (1,428) shrunk toward the anchor by one factor t, effective size t x 1,300 | t is selected **off-target**: the argmin over {0, 0.1, ..., 1} of the *other* panel's validation-fold pinball at the same budget and seed. The frozen value for Ebola is the median over seeds of the pooled two-panel argmin, sha256-stamped in `results/misc/capacity_probe_tstar.json`. | A strength chosen before any target outcome transfers. That is a different and better mechanism than Adapter_Constraint's support-only selection, which failed on L20. | Most likely t near 1, because adapting helps on dev (correction 8). That would mean dev shrinkage cannot be transferred, which closes the idea for this project. |
| control `affine` | 1,428 | the pre-registered surface, fitted exactly as Ebola: `train.ebola.choose_epochs` + `_fit_adapter`, fresh init | n/a | n/a |

**How the regime-S surfaces are fitted.** Nothing here uses early stopping, because these surfaces are
inside their budget.

- **S1 is solved exactly.** In one intercept the pinball objective is convex and piecewise linear, so its
  minimum lies on one of the residual breakpoints. Evaluating all of them costs at most 510 points per
  horizon.
- **S2** is fitted by full-batch Adam at a constant lr of 1e-2 for 5,000 steps. The run asserts two
  checks and stops if either fails:
  - its support loss is no higher than S1's exact optimum, since S2 contains S1 at slope 1;
  - running it twice as long moves the loss by less than 1e-4 relative. A decaying learning rate would
    let this pass without convergence, which is why the rate is constant.
  - The step count was set by a pre-run check on seed 42 with two stand-in anchors (a random head and
    an affine fitted on influenza_us-regions), support loss only, no test score read. At 2,000 steps
    the largest doubling move over the 8 fits was 1.0e-4, at the limit; at 5,000 it was 1.0e-5. Every
    setting reached the same support loss to about 3e-5 relative, so the move is jitter from the
    constant rate, not a failure to converge. The dengue anchor was not used, so a breach on the real
    run is less likely, not impossible.
- **S3** uses the closed form A(t) = A_anchor + t (A_affine - A_anchor), and the same for c. Both panels'
  affines are fitted first, then each panel's t is read off the other panel.

**No in-domain arm in regime S.** The dengue anchor was fitted on the full dengue fold, which contains
any transplanted support cells, so the comparison would be rigged in favour of S0. Instead every surface
and the control are compared against S0, printed with no verdict.

## 3. Decision rule and multiplicity, fixed in advance

**Per cell, unchanged from the upward sweep.** Country-macro RMSE, seed-paired improvement over the same
seed's control, two-sided 95% t interval at n = 5 (t = 2.776). A cell is significantly better or worse
only if the interval excludes zero. MAE is stored and printed and carries no verdict: the deciding
metric stays RMSE so both halves compare directly.

**Verdicts (agreement rule, V2 and Epi style).** Panel agreement is the guard. Measured on the archived
per-seed deltas, japan vs us-states correlate at median r = 0.29, and the two US panels at 0.90. So two
US panels agreeing is nearly one piece of evidence, and every rule needs japan plus a US panel.

- **Regime F, `rank-10`** (12 cross cells decide, 4 in-domain cells label):
  - WIN: at some horizon, japan AND at least one US panel are significantly better, and no
    cross-disease cell is significantly worse. If the in-domain arm is also significantly better at
    that horizon, it is WIN-GENERAL, otherwise WIN-TRANSFER.
  - MIXED: that win pattern plus a significant harm in a cross-disease cell.
  - COSTS: harm but no win pattern.
  - NULL: no win pattern and no harm (correction 15).
- **Regime F, `head-only`** (12 cross cells): DIFFERS if at some horizon japan and a US panel clear in
  the same direction, otherwise SAME. Its 4 in-domain cells are printed only.
- **Regime S, per surface x budget.** The deciding cells are L12 at h3/h5/h10 and L20 at all four
  horizons, on both panels.
  - WIN: at h3 or h5, both panels are significantly better than the control, and no deciding cell is
    significantly worse.
  - MIXED, COSTS and NULL as above.
  - **L12 h15 is printed with no verdict.** There are no pairs there, so S1 and S2 equal the anchor and
    the control is a weight-decayed random head. Those cells would clear trivially.
  - h3/h5 is the pre-registered criterion's horizon set.
- **Upward half:** re-read under the same regime-F rule from the archive, reported beside the new one.
  Expected, and already confirmed by the runner's selfcheck: all three MIXED.

**Family count.**

- Regime F: rank-r* 16 cells plus head-only 12 cells = 28 cells, 2 verdicts.
- Regime S: 3 surfaces x (6 L12 + 8 L20) = 42 cells, 6 verdicts.
- **Total 70 deciding cells, 8 verdicts.**
- Descriptive only, no verdict: head-only's in-domain cells, L12 h15, every regime-S surface and the
  control against S0, all MAE cells, and the archived upward re-read.

**What luck alone produces, stated before any number exists.** Monte Carlo, 2,000,000 draws per
scenario, numpy seed 20260929, five per-seed deltas per cell drawn normal with the stated correlation
between panels, the same 95% t rule at n = 5 (t = 2.776). Horizons are treated as independent, which
overstates any-horizon rates, and the no-harm clause is ignored, so these are upper bounds.

- Under a global null, 70 cells at 95% give about 3.5 falsely significant cells, about 1.75 each way.
  That is why no single cell decides anything.
- The chance that both panels falsely clear at one horizon is 0.0016 at the measured r = 0.29, against
  0.0006 if independent and 0.012 at r = 0.90.
- So a regime-S verdict falsely WINs with probability about 0.003, and rank-r* about 0.009 (4 horizons,
  2 US panels).
- **Whole sweep: about 0.028, roughly a 1 in 35 chance that any verdict is a false WIN.** The no-harm
  clause pushes it lower. Head-only's two-sided DIFFERS has its own false rate of about 0.019, and it is
  not a WIN.

**Frozen like V2.** This protocol is committed before the run. Its sha256 is stamped into every
per-seed record. The report refuses to decide if any hash differs or any seed is missing.

## 4. How it runs

- Regime F then regime S, per seed, on the seed's existing cap trunk. Per-seed caches
  `results/misc/capacity_probe_down__seed{s}.json`, written as each seed finishes, so the run resumes.
- Outputs: `Reports/Capacity_Probe_Down.md`, `results/misc/capacity_probe_down.json`,
  `results/misc/capacity_probe_tstar.json`. Per-origin archives go to `results/lodo/` as
  `encoder_ldo__cap-down-*`, never colliding with the 80 archived upward files. The upward outputs are
  never written by this mode.
- Runtime. Regime F is 3 x (4.2 + 22.9) = about 81 min per seed, about 6.8 h for five, from the archived
  per-surface minutes (cross 2.0 to 6.5 min, in-domain 14.4 to 42.7 min per surface). Regime S took
  about 5 min per seed in the pre-run stand-in check (seed 42, 2,000 S2 steps), so about 6 min with
  5,000 and about half an hour for five seeds. Seed 42 runs first and prints its wall clock.

## 5. Which result produces which sentence

**Results that change the manuscript** (the adaptation-failure paragraph, and a pointer in Limitations):

- **S1 or S2 WIN.** "On development panels given the exact Ebola support pattern, recalibrating a
  borrowed head with only the 4 to 8 parameters that support can fund forecast better than refitting
  the full affine read-out at h[3/5] on both panels, with no significant loss at any horizon (5 seeds,
  seed-paired). The pre-registered few-shot surface is over-sized for its support by one to two orders
  of magnitude. This was found after the freeze and does not change the case-study result." This
  unlocks the gated Ebola follow-up below.
- **S3 WIN.** The same sentence, with "a shrinkage strength chosen on other development panels, fixed
  before any target outcome". t* is frozen for the follow-up.
- **rank-10 WIN-TRANSFER.** "A read-out restricted to the 10 directions the frozen trunk actually uses
  beat the full affine read-out cross-disease even with a full training fold, and did not in domain, so
  data-rich adaptation spends capacity on directions that do not transfer."

**Results that close the gate as a stated limitation** (the likely outcome, and still informative):

- **Regime S NULL, and the affine is not worse than S0** (the simulation does not reproduce the damage,
  as on covid): "Transplanting the Ebola support pattern onto development panels did not reproduce the
  few-shot damage. At that budget, surfaces sized to the data (4 to 8 parameters, or a transferred
  shrinkage) did not differ from the full affine read-out. Whether a smaller surface repairs the damage
  on Ebola itself is untested on the pre-registered data."
- **Regime S NULL or COSTS, and the affine is significantly worse than S0** (damage reproduced, not
  recovered): "Even where the development simulation reproduced the few-shot damage, a surface sized to
  its support did not recover it, so surface size is not the lever."
- **Regime F NULL or MIXED, head-only SAME:** "Swept in both directions on the development trunk, from
  rank-10 (860 parameters) to 21,780, no read-out beat the shipped affine map under a pre-registered
  agreement rule. Every larger surface that helped at h15 also hurt influenza_japan at h3. The shipped
  surface is right-sized for data-rich adaptation."
- **Combined, if everything is null:** "We swept the adaptation surface in both directions, sizing the
  smaller surfaces to the data an adapter actually receives. No surface from 4 to 21,780 parameters beat
  the shipped affine read-out, so we do not attribute the few-shot damage to the size of the adaptation
  surface."
- **Head-only DIFFERS**, whatever else happens: add "capacity comparisons in this ladder are confounded
  with parametrisation" to every sentence above.

**Gated Ebola follow-up, not part of this run.** Only if a regime-S surface WINs, or the frozen t* < 0.5:
`experiments/small_surface_ebola.py`, reusing adapter_constraint's loaders. It is stamped
`protocol="EXPLORATORY"`, runs `load_manifest(verify=True)` before and after, writes only to
`experiments/`, and is never placed beside the pre-registered numbers unlabelled. It needs no GPU.

## Scope limits, stated with any result

- It is one direction (dengue to flu) and one source disease. The dev anchor is a single-disease head,
  not Ebola's three-disease borrowed mean.
- The flu simulation keeps `per_node_train` scaling, so it does not reproduce Ebola's pooled-scaler
  covariate shift (CLAUDE.md section 6).
- t is transferred across two panels of the same disease, not across diseases.
- The downward control is a same-session refit, not the archived one (correction 9). The two halves
  share a trunk and a protocol, not an adapter init.
- r* is set by the trunk's own disease (Step 0), so rank-10 is a loose restriction on flu.
- The extrapolation factor on the flu simulation is not measured. It can be added from the saved maps
  with the `adapter_mechanism.py` statistic if a regime-S result needs a geometric explanation.

## Reproduce

```powershell
conda run -n ebola-train python -m diagnostics.capacity_probe --selfcheck
conda run -n ebola-train python -m diagnostics.capacity_probe --measure-rank
conda run -n ebola-train python -m diagnostics.verify_capacity_down
conda run --no-capture-output -n ebola-train python -m diagnostics.capacity_probe --down --skip-fold --seeds 42 52 62 72 82 2>&1 | Tee-Object results/reports/capacity_probe_down.log
conda run -n ebola-train python -m diagnostics.capacity_probe --down --report
conda run -n ebola-train python -m diagnostics.verify_capacity_down --result
```

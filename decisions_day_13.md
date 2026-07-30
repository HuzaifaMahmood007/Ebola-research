# Decisions — Day 13 review follow-up

Notes on the decisions we made after the Day-13 advisory review, plus the encoder
ideation we did before starting Day-14 joint training. Written so the next person
(or future me) doesn't have to re-argue any of this.

---

## 1. Review fixes #1–#5 (the clear ones — just did them)

These were flagged as "clear fixes" in the review, no debate needed:

- **#1 Grad-accumulation flush** — it was stepping the optimizer unconditionally at the
  end of an epoch, without gradient clipping, and it counted *all* origins (even skipped
  ones) toward the batch boundary. Fixed: clip before every step, only step when there
  are actually pending gradients, and count real `backward()` calls.
- **#2 `refit` had no test** — this is the sharpest leakage trap in the whole data layer,
  so I added a self-check that proves `refit` rebuilds `X[:,:,0]` together with `y` (a
  version that only rebuilt `y` would silently leak the eval distribution).
- **#3 `refit` used the wrong scaler for Ebola** — it hardcoded per-node scaling. Ebola
  ships a disease-pooled scaler. Fixed: derive it from `meta['scaler_scope']`.
- **#4 `window_slice` failed silently on short windows** — a negative start index wrapped
  around instead of erroring. Added `assert t >= W-1`. (The proper Ebola left-pad path is
  Week-4 work.)
- **#5 Untested adjacency** — added a `mask_aware_adj` unit test and an encoder
  permutation-equivariance test.

All green, plus the existing invariants still pass.

---

## 2. Judgment calls #6–#9 (these needed a decision)

### #6 Quantile crossing → **post-hoc sort at Week 5**
The plain Linear head can output crossing quantiles (q05 > q95). This only hurts the
prediction *intervals*, not the point forecast (the median). Rather than change the head
architecture now (which would force a full retrain), we'll just sort the 5 quantiles at
inference time in Week 5, before computing PICP/CRPS. Left a `ponytail:` comment in the
code so we don't forget. Upgrade path if sorting isn't enough: a monotone
cumulative-softplus head (but that means re-running everything).

### #7 Results schema → **persist granular scores**
The old records only saved two aggregates per run. You can't reconstruct a bootstrap CI or
a paired Wilcoxon test from a mean. So we now persist granular scores:
- First added **per-node** scores (all 6 metrics) as npz.
- Then (Day-14 request) added **per-origin × country** error sufficient stats (`sae`, `sse`,
  `n`) so we can bootstrap the metric *over origins* and still rebuild the country-macro CI.

### #8 Pinball loss reduction → **leave as global mean**
The plan says "per-horizon then sum," the code does one global mean. We checked: because
the released scaler is a per-node z-score, all four dev datasets sit at ~unit scale on the
train window, so the difference between the two reductions is tiny for the dev data. Not
worth changing right now.

### #9 "g≡0" arm framing → **leave as-is**
The complaint: the g≡0 ablation arm still includes the LTR degree feature, so calling it
"graph-free" is slightly wrong. But this is only a *label* on an ablation arm — it doesn't
touch the primary model at all. Fix the wording later if a reviewer asks; no code change.

---

## 3. Encoder — the seasonality investigation

After the overnight run, the encoder lost to the seasonal-naive baseline on the japan flu
data. First guess was "the model can't see seasonality." That turned out to be too strong —
the encoder *does* get sin/cos day-of-year as input channels, so it has the seasonal *phase*.

**We ran a proper test (japan only):** zero out the sin/cos channels and see what breaks.
- Result: performance got clearly worse (PCC dropped ~0.07 on every horizon, way past seed
  noise).
- **Conclusion: the encoder IS using the phase.** So this is not a "model ignores
  seasonality" bug.

**So what's actually missing?** The *amplitude anchor* — last year's level at this same point
in the season. That's exactly what seasonal-naive copies for free (`y[t-52]`), and our 20-week
lookback simply can't reach back a full year.

### Decision: **reject the obvious fix, accept the limitation**
The obvious fix is to feed `y[t-52]` as an extra input channel. We rejected it:
- It **breaks the C1 hard constraint** (the trunk must take exactly 4 core channels). We never
  break the hard constraints.
- More importantly, our real target is **Ebola**, which only has ~52 weeks of data and no
  year-over-year history at all. So the anchor helps the easy flu datasets and does *nothing*
  for the actual thesis. Seasonal-naive also fails on Ebola for the same reason.

So we **accept the seasonal-amplitude gap as a documented limitation**, not a bug. It's a
property of the frozen w=20 protocol, and it doesn't affect the Ebola story.

### Still open: persistence-anchor (delta) target
A constraint-safe idea we haven't decided on: predict the *change* from the last observed
value (`y[t+h] = y[t] + model_output`) instead of the absolute level. `y[t]` is already in the
window, so no new channel, no constraint broken. It would target the two problems the amplitude
anchor doesn't: losing to persistence at short horizons, and mean-reversion at long horizons.
**Not decided yet.**

---

## 4. Scaler / loss balance for joint training

Verified (didn't need a fix): because the released transform is a per-node z-score, all four
dev datasets arrive at ~unit scale on the train window (std ≈ 1.0 for all of them). So when we
move to joint multi-disease training, no per-dataset loss rescaling is needed — the only
imbalance is sample count, and uniform per-dataset sampling handles that. Green, ship as-is.

---

## 5. Day-14 schema prep (built this session)

To get ready for joint training without breaking the existing single-disease results:
- Added four run-metadata fields to every result record: `training_regime` (`single`/`joint`),
  `sampler`, `gate_mode`, `topo_aug`.
- Wired `gate_mode` to the actual encoder (`SharedEncoder(gate_mode=...)`), so it's real config,
  not just a label. Default `"learned"` is bit-identical to the old behaviour. This also unlocks
  the P9 `off` (g≡0) ablation arm later.
- Added the per-origin × country artifact (see #7) with a self-check.
- Wrote an idempotent `backfill_schema.py` to stamp the old result JSONs with the new fields.

**Important gotcha:** the old single runs didn't save their raw predictions, so the per-origin
artifact **cannot** be backfilled for them. To get per-origin data for the single runs we have
to re-run `train.loop --all` (which now writes everything natively). The backfill only adds the
metadata tags.

---

## 6. Joint training — block-diagonal supergraph (the real Day-14 work)

We build the joint multi-disease batch as **one block-diagonal supergraph**, not sequential
per-dataset gradient accumulation. The 4 dev datasets stack along the node axis into `[ΣN=7271, 20, 4]`
with a block-diagonal adjacency that has **no cross-dataset edges**. One forward, one backward, one step.

**Why it's safe (verified, not assumed).** We read every op in the encoder and confirmed **nothing
reduces over the node dimension**: the TCN treats nodes as the conv batch dim, spatial message passing
follows edges only, and there is **no BatchNorm** (the one layer that would blend datasets). So each
block's math is identical whether run alone or inside the supergraph — no cross-block edges means no
inter-dataset leakage. This is locked as a permanent gate: `--equiv` asserts block-diagonal forward ==
per-dataset solo forward, node-for-node (`atol 1e-5`).

**Architecture: one shared encoder + one adapter per dataset** (FiLM + head, ~388 params each, P5).
- Rejected the alternative (single shared adapter): one head fitting dengue *and* influenza scales
  would fight itself and undercut the whole shared-trunk claim.
- The per-dataset adapters do **not** make the project disease-specific — the *trunk* is the
  transferable asset; adapters are throwaways. A new disease (COVID, etc.) = one new adapter
  few-shot-fit with the trunk frozen, which is exactly the Ebola path (P5/P7).
- Locked as a second gate: `--routing` asserts a one-dataset loss gives every *other* adapter
  **exactly zero** gradient — the strict isolation Week-4's MAML inner loop depends on.

**Downsides accepted:** small datasets ride in every batch and get cycled hard (japan windows seen
~24× per dengue-pass), so overfitting leans on val early-stopping; and the gates add CPU cost to CI
(cheap — the 3 small datasets).

---

## 7. Samplers as loss weights + step-based training

Under block-diagonal batching the "sampler" **becomes loss weighting**, not sampling — every dataset
is in every batch, so the question is how much each pulls on the gradient. A naive mean over 7271
nodes hands dengue **98.5%** of the gradient (it is 7165 of the nodes). (This is the concrete
realisation of §4's "uniform per-dataset sampling handles the sample-count imbalance" — same idea,
implemented as a weight.)

Two **orthogonal** weight vectors:
- **Across datasets `w_i`** (`uniform` / `proportional` / `sqrt`), from **train `n_obs`**. `uniform`
  (P2 primary) → 25% each; `proportional` → the dengue-dominated pooled mean (98.4%); `sqrt` → the
  Week-6 middle ground (dengue ~82%). *Decision on the size measure:* `n_obs` over `N` — but verified
  it barely matters (proportional moves <0.5pp, sqrt ~2pp between the two), so we take `n_obs` counted
  over **train cells only** purely for leak-hygiene, at no real cost.
- **Within dengue `v_node`** (`uniform` / `country`). *Decision: per-**CELL** country balance
  (`v ∝ 1/train_obs_c`), not per-node.* We checked the data: observation density spans **97×** across
  dengue's 12 countries (nicaragua 981 obs/node vs argentina 10), and the small countries are the
  *dense* ones. Per-node balancing (`1/n_c`) would leave each country pulling ∝ its obs/node — i.e. it
  would **not** actually balance them. Per-cell does, and it's a fixed precomputed vector (same cost).
  The 3 influenza sets are single-country, so their vector is a no-op.

The two compose cleanly: `v` balances *inside* a dataset (each `L_i` stays a proper mean), `w` balances
*across*. `uniform+uniform` is **bit-identical** to the pre-weighting mean, and `pinball_loss` gained
only an optional `w=None` arg, so the single-disease trainer is untouched (all encoder invariants still
green). Locked as gate #3: `--equiv` asserts country mass is equal to 1.0000× and `w=None == w=ones`.

**Training is defined in STEPS, not epochs.** With 4 datasets at a fixed per-step ratio, "epoch" is
meaningless (one pass over dengue is ~24 passes over japan). We train for a step budget, validate every
`val_every` steps, and count patience in val-checks. Downside: you lose the intuitive "N passes over the
data" and have to pick step numbers by hand.

**Downsides accepted (discussed explicitly):** uniform weighting trades some of dengue's *own* peak
accuracy for cross-disease balance, and it amplifies the noisy small datasets (us-regions is 10 nodes
driving 25% of the gradient). We keep `sqrt` on deck as the Week-6 ablation precisely because "equal"
isn't provably optimal. The ablation surface also grows (3 schemes × 2 dengue vectors); only
`uniform+uniform` is the primary run.

---

## 8. Paired CIs and the "free reads" (items 6–9) — and the checkpoint finding

**The blocking finding:** the overnight `--all` run **never saved model weights** — `best_state` is an
in-memory clone, used to reload before the test forward, then discarded. There are no encoder
checkpoints on disk (the only `.pt` files are baselines). So "load the existing 20 checkpoints, no
retraining" is impossible as stated.

**Resolution: emit *derived artifacts* during the run instead of persisting models** (cheaper than
30 MB of checkpoints, and no reload infra). This costs one deterministic re-run of `train.loop --all`,
which now writes everything natively; the reads are then pure offline numpy that never touch the model
again.
- **Item 6 (paired CIs):** bootstrap over **origins**, not seeds. Resample origin indices with
  replacement (B=10,000), rebuild the country-macro from the per-`(origin,country)` sufficient stats
  (`sae/sse/n`), take 2.5/97.5 percentiles. For a comparison, **resample the same origins for both
  models and build the CI on the difference** — that pairing is what makes it robust to japan's ±7–10%
  seed sd. Wilcoxon across the 5 seed-matched runs is **supporting only** (n=5, underpowered, must not
  lead). Implemented in `analysis.py --ci`; engine verified against brute-force resampling.
- **Items 7–8 (gate + spatial reads):** a val forward while the trained model is still in scope emits
  per-node `mean_g` and normalised spatial contribution to `*__gate.npz`; `analysis.py --reads`
  aggregates to per-dataset gate mean/IQR/`frac(g<0.05)` and spatial mean/IQR. These are **pre-head, so
  horizon-independent** — reported per dataset, not per horizon. They tell us whether the gate is doing
  anything: near-zero `g` or a high `g<0.05` fraction ⇒ the gate is ~off ⇒ the Day-14 2×2's
  {gate on vs g≡0} arm is measuring little. (Toy-model preview: g≈0.36, so it's meaningfully on — but
  the real reads come from the re-run.)
- **Item 9 (§0.9 node-mean variance):** data-only, no model needed, and already computed in
  `results/day11_diagnostics.json` (dengue 0.039, influenza ~1e-14, ebola 0.70). Held, nothing to do.

**Metric caveat on item 6 (documented in code + every CI table):** the saved `sae/sse/n` are node-pooled
*within* each country, so the reconstructed country-macro is **cell-pooled**, whereas the headline
metric (`score.aggregate`) is **node-averaged** (mean of per-node metrics). They coincide on the dense
influenza panels and **diverge on dengue**. An exact-headline CI would need per-`(origin, node)` stats
(~1.8 GB for dengue) — deferred; the cell-pooled CI stands unless we decide we need the heavier artifact.
**PCC is not covered** by the origin bootstrap — it isn't additive from `sae/sse/n` (would need
Sx/Sy/Sxx/… per origin-country).

---

## 9. Day-14 advisory review + the fixes it triggered

After building the joint trainer, we ran an advisory pass — two independent reviewers, one on code
correctness/integration, one on Week-3 completion + protocol integrity.

**Verdict.** The joint trainer's *math* is clean: block-diagonal equivalence, adapter routing
isolation, and loss-weight balance are all verified by gates, and there's no leakage (weights
train-only, selection val-only, test unweighted). But the *integration/write-path* was broken — the
Day-14 code was committed but **never actually run against data**, so a set of seam bugs between the
joint trainer and the (separately-added) single-trainer per-origin changes went unnoticed.

**Bugs found and fixed:**

- **C1 (Critical) — joint `--all`/`--seed` crashed after training, wrote nothing.**
  `score_predictions` now returns a 3-tuple `(records, pernode, perorigin)`, but `run_joint` still
  unpacked a 2-tuple → `ValueError`. It only surfaced on a *real* run (not `--smoke`/`--equiv`), so
  it would have trained the full ~91k steps and then died before writing a single file. **Fix:**
  `_test_dataset` now returns the full 4-tuple `(records, pernode, perorigin, gate)`, and `run_joint`
  unpacks and writes all of it.
- **I2 (Important) — joint artifacts weren't at parity with single.** Joint records had no `run_meta`
  fields (a joint row was indistinguishable from a single one except by the model-name string), and
  the per-origin and gate artifacts were never written. **Fix:** `_test_dataset` passes
  `run_meta=dict(training_regime="joint", sampler="<scheme>-<balance>", gate_mode, topo_aug="none")`,
  and `run_joint` writes `*__perorigin.npz` and `*__gate.npz` for every joint run — the same set as
  the single trainer.
- **I3 (Important) — `analysis.py` couldn't see joint runs.** It globbed `encoder__*`, but joint files
  are `encoder_joint__<tag>__*`. **Fix:** added a `--joint TAG` flag (a `prefix` arg threaded through
  `bootstrap_ci`/`gate_reads`/`_encoder_stats`) so the same CI + gate-read code targets joint runs.
- **I4 (process gap) — the block-diagonal safety check never ran before the real training.**
  `_equiv_check` fired only under `--equiv`/`--smoke`, not in the `--all` run that produces the
  numbers. **Fix:** `run_joint` now calls `_equiv_check()` once before producing any results — if a
  future encoder change ever broke the no-cross-node-coupling premise, the run aborts instead of
  silently emitting leaked results.
- **Minor — threaded `gate_mode` into the joint trainer** (`--gate-mode learned|off`), so the joint
  g≡0 ablation arm is runnable and `run_meta.gate_mode` is honest.

All gates stay green after the fixes (`--equiv`, both self-checks), and a small end-to-end `run_joint`
on the 3 small datasets confirmed the write-path: records carry the schema fields, perorigin/gate npz
are produced, and `analysis.py --joint` reads them.

**Protocol integrity — intact (advisor-confirmed).** Ebola's query set is still untouched (both
trainers iterate `DEV_BUNDLE_NAMES`, ebola excluded by construction); selection is val-only; C1/C2/C3
invariants still hold.

**C8 made explicit (done).** The `ebola-never-in-selection` guarantee was previously only structural
(exclusion from `DEV_BUNDLE_NAMES`). It's now a runtime guard at the actual selection points —
`train_one` asserts `name != "ebola"`, `train_joint` asserts `"ebola" not in names` — so any future
caller that tries to pull ebola into trunk training/selection trips immediately, regardless of how the
dataset list was built (Week-5 few-shot is a separate adapter-fit path, deliberately not routed
through these). Named gates verify both: `train.loop --selfcheck` and `train.joint --equiv` each fire
a C8 control.

### Resolved: P9 `topo_aug` — **formally deferred out of Week-3 scope** (2026-07-24)
The advisor confirmed **annealed edge dropout (P9 / `topo_aug`) is not implemented** — it's only a
schema field defaulting `"none"`; there is no edge-dropout code or annealing schedule anywhere.
Consequence: the primary 2×2 ablation `{gate on, g≡0} × {aug none, edge_drop}` can't be run — the
`edge_drop` arm doesn't exist.

**Decision: defer.** P9 is a robustness *ablation*, not part of the primary model, and it is only
meaningful once we know the shared trunk is worth hardening — which the Day-14 transfer result now
puts in question (see `Phase3_Week3_Results_and_Direction.md`). Building edge-dropout + annealing +
the 2×2 this week would spend a day protecting a spatial channel whose value is unconfirmed. The
`topo_aug` schema field stays (`"none"` default), so nothing regresses; the 2×2 moves to **Week 6**
alongside the other ablations, or is dropped entirely if the direction review retires the spatial
apparatus. No code change required to defer.

---

## Open items / next up
- ~~Decide P9 `topo_aug`~~ — **DONE: deferred out of Week-3 scope** (§9), moved to Week 6 / conditional on the direction review.
- Decide the delta-target idea (yes/no) — §3.
- Run `train.loop --all` (the deterministic re-run) so the single runs emit `*__perorigin.npz` and
  `*__gate.npz`; then `python analysis.py --ci --reads`. Backfill only stamps metadata, not per-origin.
- **Run the joint matrix — now unblocked** (C1 fixed): `train.joint --all` (uniform+uniform primary),
  then `python analysis.py --joint uniform-uniform --ci --reads`, then produce the single-vs-joint
  help/hurt numbers (Day13 §6).
- Decide whether the item-6 CI needs exact-headline (node-averaged) stats for dengue, or the
  cell-pooled CI is acceptable (and whether PCC needs origin-bootstrap support at all) — §8.
- G6 baselines under the common pipeline — Day-15 → Week-4 by design; call it out, don't leave it
  silently open.


Residual / persistence-anchor target — predict y[t+h] − y[t] (or add a y[t] skip into the head), instead of the absolute level. The model then starts from persistence and only learns the correction, which directly attacks the short-horizon dengue loss (that's the horizon where persistence beats us). No new input channel, y[t] is already in the window → doesn't break C1 or Ebola. This is the single highest-leverage change and it's already half-decided in decisions_day_13.md §3. Downside: can add noise / over-mean-revert at h10/h15 — must check it doesn't cost us where we currently win.
Ensemble the 5 seeds — we already train 5 seeds per dataset; average their quantiles at inference instead of just reporting mean-of-metrics. Free variance reduction, and it helps most exactly where we're weakest (the noisy small datasets, Japan/US-regions). Ebola-safe (ensemble the few-shot adapters too). Downside: basically none; 5× inference cost, which is nothing.
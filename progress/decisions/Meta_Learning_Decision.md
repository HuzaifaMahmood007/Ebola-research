# Meta-Learning: Algorithm Decision and Schedule Assessment

**Phase 3, Week 4 - Section 4 - Prepared 2026-07-30**

**Answers the two deliverables requested this week.** Companion artifacts:
`Reports/Encoder_Results_Consolidated.md` (the LDO result this decision reacts to),
`Final_Internal_Project_Brief.docx` (G2), `models/adapters.py`, `train/lodo.py`.

---

## Summary

**Deliverable 1, algorithm.** MAML restricted to the adapter, that is **ANIL** (Almost No Inner Loop,
Raghu et al., ICLR 2020), with **exact second-order** gradients. Justification in section 1.

**Deliverable 2, schedule.** **No.** Meta-learning at the scope described does not fit the remaining
fifteen working days alongside Weeks 5 and 6 as written. It is not close, and the reason is
dependency order rather than implementation difficulty. Detail in sections 3 and 4. Re-scope options
for Nora in section 5.

We are not absorbing this. What we propose to do in Week 4 without a decision from Nora is in
section 6, and it is deliberately the smallest thing that produces a real number.

---

## 1. Algorithm and justification (one paragraph, as requested)

We will use MAML restricted to the adapter, which is ANIL, with exact second-order gradients: the
inner loop updates only the 388-parameter FiLM adapter on an episode support set, and the outer loop
backpropagates the query loss through that inner loop into the trunk. The decisive argument is
train-test consistency. The Ebola protocol is to freeze the trunk, fit the adapter on the support
set, and score the query set once; ANIL's meta-training objective is exactly that procedure
differentiated, so the trunk is optimised for the operation we will actually perform at deployment,
whereas full-network MAML optimises for an inner loop we will never run. ProtoNet is excluded on task
grounds, being a nearest-centroid classifier over class prototypes while our target is continuous
multi-horizon regression; making it apply would require a non-standard prototype-regression variant
that is itself an unbudgeted research contribution. Reptile is excluded on structural grounds, which
is stronger than a preference: its outer update is the difference between adapted and initial
parameters, so if the inner loop touches only the adapter then the trunk difference is identically
zero and the trunk never trains at all, and rescuing it would mean unfreezing the trunk in the inner
loop, which contradicts both `models/adapters.py` and the frozen-trunk deployment protocol. ANIL also
pays off the encoder-selection argument raised in the brief, because with a 388-parameter inner loop
the Hessian-vector product is genuinely cheap, so we can run true second-order meta-learning rather
than the first-order approximation, and a FOMAML arm turns "the second-order property was worth
paying for" into a measured number instead of an assertion. Finally, ANIL makes the comparison the
client actually asked for a controlled one: freeze-then-adapt is ANIL with the episodic outer loop
replaced by ERM, holding the trunk architecture, the adapter, the inner loop and the scoring stack
fixed, so "does meta-learning beat the probe" changes exactly one variable.

### 1.1 One correction to the framing, offered for accuracy not defence

What we built is not a linear probe. The FiLM adapter is a feature-wise affine modulation of
intermediate trunk activations, not a linear read-out head on frozen features. The substantive point
stands in full: it is frozen-trunk parameter-efficient adaptation fitted by ERM, and it is not
meta-learned. We record the distinction only so the manuscript does not carry a description a
reviewer can correct.

### 1.2 Domain generalisation, the other half of G2

G2's methodology pillar names "meta-learning / domain-generalisation". The domain-generalisation leg
is already partly discharged structurally rather than by an objective: `transfer_view()` exposes only
the four core channels, static covariates `C` are withheld from the shared encoder, and the influenza
self-loop convention was harmonised precisely because each of these is a feature from which the
encoder could infer which disease it is looking at. That is invariance enforced in code. If an
explicit objective is wanted, the cheapest honest addition is a worst-domain weighting over the
natural domains we already have (dengue's twelve countries plus the three influenza panels), because
it reuses the same episode sampler ANIL needs and costs one extra arm rather than one extra system.
We recommend holding this until the ANIL sampler has landed, and not before.

---

## 2. Why this is now load-bearing rather than a checkbox

The consolidated encoder results changed what this decision is for. Freeze-then-adapt was scoped as
the ablation baseline that meta-learning would be measured against. That baseline is now a negative
result: under the corrected leave-one-disease-out fold, **twelve of sixteen RMSE cells are
significantly negative, four are within noise, and none is positive**, with an identical tally on MAE,
and the degradation is monotone in horizon (influenza-Japan -51.3% and influenza-US-regions -51.2% at
h15). Cross-disease zero-shot fails outright on influenza (-311.2% at Japan h3, -558.8% at h10).

Two consequences follow.

**The mechanism ANIL targets is the mechanism that failed.** The zero-shot collapse says the trunk
representation does not transfer across diseases on its own; the adapted result says 388 parameters
of ERM-fitted modulation do not repair it. ANIL does not change the representation's capacity, it
changes what the representation is optimised *for*, namely being repairable by a small support-set
update. That is the correct intervention for this failure mode, which is the honest reason to expect
it might work.

**It is also the reason to pre-register the possibility that it does not.** A deficit of -16% to -51%
is large to close with a 388-parameter adaptation surface. Adapter capacity is therefore coupled to
this decision and must be selected on development folds only, never with sight of Ebola. We propose
fixing the capacity sweep to the development LDO folds and freezing it before the Ebola run, per the
existing validation protocol.

---

## 3. Why the answer on schedule is no

The blocker is not writing the inner loop. It is that ANIL sits **upstream of a single-shot
evaluation**, so it serialises everything behind it.

**The dependency chain, as it now stands:**

1. Client answer on the Ebola support-set length (currently open, `Reports/Ebola_Support_Set_Decision.md`).
2. ANIL trunk trained, selected on development folds, and frozen.
3. Ebola run, scored exactly once against that frozen configuration.
4. Conformal uncertainty and SHAP explainability on the resulting forecasts (G4, G5, both REQUIRED,
   both not started).
5. Ablations, robustness, tables, manuscript (G7, Week 6).

Step 2 cannot overlap step 3, because the protocol forbids it: Ebola is scored once against a frozen
configuration. If ANIL becomes the method, the Ebola case study must use the ANIL trunk, so every day
ANIL takes is a day removed from Weeks 5 and 6, not a day run in parallel with them.

**ANIL replaces trunk training, it does not extend it.** The current LDO results come from a
91,000-step ERM trunk per direction. Meta-training produces a different trunk, so the entire fold
structure must be run again: two directions times five seeds is ten trunk runs on the single RTX
3060, serial, plus the pilot and capacity sweep that precede them.

**Per-step cost.** With K inner steps, each outer step costs roughly K+1 trunk forward passes plus the
differentiated backward, so on the order of 6x to 11x an ERM step at K=5. We have not measured this on
our stack and will not quote a wall-clock figure we have not measured; a 200-outer-step timing probe
settles it in under an hour and we will run it first.

**There is a clean compute-matching rule that saves an ablation rerun.** Setting outer steps to
91,000 / (K+1) holds the number of trunk gradient evaluations constant, which makes the existing ERM
LDO runs a valid matched ablation with no rerun at all. That is the largest single saving available
here and we recommend taking it. The caveat is real and should be pre-registered: matched compute may
leave the meta arm under-trained relative to its own convergence, so a matched-outer-step arm is the
honest sensitivity check, and that arm is where the cost returns.

**Four REQUIRED items are still ahead of us and none of them is started.** G4 uncertainty (WIS, CRPS,
coverage, PIT are implemented and self-checked but no run archives quantile predictions, so the
calibrated-uncertainty claim is currently unevidenced), G5 explainability (SHAP, not scoped), the
Ebola case study itself (blocked), and the manuscript reconciliation, whose headline has just
inverted. Sections 6 through 8 of the Week-4 work order are untouched.

We are at **day 15 of 30**. Weeks 5 and 6 as written are already full.

### 3.1 One item that must ride on the meta-learning runs, whatever is decided

Quantile prediction archiving is cheap on a new run and expensive to retrofit. It is the largest
outstanding gap against the required metric set, and it is currently blocking G4 entirely. Whatever
happens to ANIL, **no further trunk run should be launched without quantile archiving turned on**.
Merging this into the meta-learning runs converts two jobs into one and is the second-largest saving
on the table.

---

## 4. What we are confident of, and what we are not

**Confident.** ANIL is the right algorithm, the exclusions of Reptile and ProtoNet are structural
rather than aesthetic, and the full scope does not fit the remaining fifteen days.

**Not confident, and stated as such.** We have not measured the per-step multiplier on our stack, so
the size of the overrun is estimated rather than known. The timing probe in section 6 replaces the
estimate with a measurement by end of Week 4. If the multiplier lands at the low end and the
compute-matching rule holds, option C in section 5 may fit without any extension.

---

## 5. Re-scope options, for Nora

Ranked by our preference.

| # | option | what it costs | what it protects |
|---|---|---|---|
| A | Extend by one week, to day 35 | One week of schedule | Everything. Full two-direction, five-seed ANIL, both ablation arms, Weeks 5 and 6 intact |
| B | Hold 30 days, cut secondary work | LODO five-seed completion, MTGNN collapse diagnosis, the joint `sqrt-uniform` probe, the paper-protocol Cola-GNN and HeatGNN reruns, the epidemiology-informed component | Full meta-learning scope, at the price of a thinner baseline and ablation section |
| C | Hold 30 days, descope meta-learning to one direction | The flu-to-dengue meta direction is not run | Weeks 5 and 6 intact, and G2 is answered with a real number rather than an assertion |

**On option B, what is safe to cut.** LODO is population transfer, rests on a single seed, and that
seed is the known dengue outlier; it is already flagged as unquotable and is not the paper's claim.
MTGNN emits a single constant on 47 of 80 files and cannot enter the comparison table either way, so
logging it as a Section-A reproduction failure is both cheaper and more honest than diagnosing it.
The `sqrt-uniform` probe is one seed and uninterpretable. The epidemiology-informed component is a
brief SUGGESTION, not a requirement.

**On option C, which direction to keep.** Dengue-to-influenza, and this is a technical argument
rather than a convenience. Episodic meta-training needs a task distribution, and the meta-train side
is where the tasks must come from. Meta-training on dengue gives twelve countries and 6,161 nodes to
draw episodes from; meta-training on influenza gives three panels of 47, 10 and 49 nodes, which is a
thin task distribution and a weak test of the method. The same direction also has the most headroom,
since influenza is where LDO transfer fails worst. Descoping the other way would test ANIL in the
regime where it has the least chance of a fair trial.

**What we do not recommend cutting: seeds.** Dropping from five seeds to three to save time would
breach the project's own dispersion standard and would push the t critical value from 2.78 to 4.30,
which makes a positive finding harder to establish, not easier. Cut arms, never seeds.

---

## 6. What we will do in Week 4 without waiting for a re-scope

The smallest thing that produces a real number. None of it is wasted under any of options A, B or C.

1. **Episode sampler and inner loop.** Episodes defined below the disease level (country and rolling
   origin for dengue, panel and rolling origin for influenza), with support size matched to the Ebola
   regime rather than chosen for convenience. Smoke test asserting inner-loop-touches-adapter-only,
   trunk-in-outer-loop, and second-order gradient flow to the trunk being non-zero.
2. **Timing probe, 200 outer steps.** Replaces the estimated multiplier with a measurement and
   settles whether option C fits inside 30 days.
3. **One-seed pilot, dengue-to-influenza, matched compute.** Go or no-go signal against the existing
   ERM ablation at the same trunk gradient budget.
4. **Quantile archiving switched on** for every run from here.

By end of Week 4 this yields a measured cost, a first ANIL-versus-probe number on one seed, and no
commitment we cannot reverse.

### 6.1 Two decisions we need back, together

The Ebola support-set answer and the meta-learning re-scope both gate Week 5, and both feed the same
frozen configuration. We recommend they are taken as one packet rather than sequentially, because
answering them a week apart costs a week either way.

---

## 7. The result we may have to report

Stated now rather than after the fact. If ANIL also comes back negative, the paper's cross-disease
transfer claim does not survive, and the contribution becomes a rigorous negative: a disease-agnostic
architecture, a corrected fold structure that separates population transfer from disease transfer, and
a meta-learning arm that was tested rather than asserted. Several of the models we benchmark against
name cross-disease transferability as open future work, so a careful negative on it is a real
contribution and is publishable. It is not, however, the contribution G2 describes, and G2 is marked
REQUIRED. That is a client and Nora decision, not ours, and it is better raised now than in Week 6.

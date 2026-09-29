# V2 deviation channel: the mechanism test result

**Run and scored 2026-09-25.** This closes the pre-registered V2 deviation-channel test. The protocol
was fixed and committed before any COVID number existed
(`progress/decisions/V2_Deviation_Protocol.md`, sha256
`82069f4138099ae154b30e4c20e214ac3181b2975e2d9c9f5349cf091c8189ef`, commit
`39ecf345abc0fcc7d1029d8a8b63d6faef6ca7c8`). The decision was read once, from disk, by
`ablation/run_v2_deviation.py --report`, which refuses to open a single test record until all four
arms exist at every seed.

**Verdict: FAIL, criterion (a) not met.** Handing the graph the district deviations directly did not
make the graph earn accuracy. It made the model worse, and a wrong map or no map at all did the same
damage.

## The question this answered

The earlier graph nulls (gate-off, the D2 shuffled-adjacency retrain) showed the trained model's
spatial channel does not lower error on COVID. One hypothesis for why: the encoder compresses the
district-specific signal away before the graph ever sees it
(`progress/outcomes/Input_Energy_2026-09-22.md`, COVID 23.8 percent district-specific energy in, 4.8
percent out). And the raw COVID data does carry neighbour information: after the shared weekly wave is
removed, a model-free linear probe finds real neighbours remove about 5.3 percent of total
model-space target variance at h3 and 2.3 percent at h5
(`progress/outcomes/Neighbour_Signal_2026-09-24.md`).

So the test: add one branch that computes each district's deviation from its country's weekly mean and
routes it to its own spatial mixer. If the compression is why the graph failed, giving the graph the
deviations directly should let the real map earn accuracy. The protocol fixed, before any data, what
result would support that and what would not.

## What ran

- **Panel.** covid_us-states only. 49 states, 60 train origins, 45 validation, 49 test.
- **Three v2 arms**, sharing `models/encoder_v2.py`, differing only in the map the deviation branch
  mixes over. v1's own path mixes over the real map in every arm, so each comparison moves exactly one
  thing.
  - `v2graph`, the real map, the full mechanism.
  - `v2nograph`, the identity (a district sees only its own deviation), isolating "the extra input
    helped" without neighbours.
  - `v2shuffled`, a degree-preserving relabel of the real map, isolating "any map helped" against "the
    real map helped".
- **v1**, the released single-disease run, plus extra seeds trained as `__v1ref` in stage 2.
- **The branch adds 10,224 parameters**, combined through a zero-initialised projection, so v2 starts
  exactly at v1 and moves only if the branch lowers the training loss.
- **Two stages.** Stage 1 trained the three arms at the 5 released seeds and read the validation split
  only, to measure the paired noise. The largest of the three relative standard deviations was
  s = 0.0412 (from v2graph against v2shuffled). The protocol's formula turned that into
  **N = 15 seeds** (42, 52, ..., 182), with a 0.822 chance of finding the probe-implied 3.3 percent h3
  RMSE effect. Stage 2 trained all four arms at those 15 seeds.
- **Decision rule.** For each cell, d is arm minus reference at the same seed. A comparison counts only
  when the paired 95 percent t-interval of d excludes zero. Lower error is better. PCC is printed and
  never decides.

## The verdict table

v2graph against v1, paired over 15 seeds, country-macro (which equals node-mean on single-country
COVID), median forecast. Lower is better. RMSE and MAE decide; PCC is printed only.

| metric | h | v1 mean | v2graph mean | d | 95% CI | verdict |
|---|---|---|---|---|---|---|
| RMSE | 3 | 5306.760 | 10444.773 | +5138.013 | [+3669.034, +6606.992] | worse |
| RMSE | 5 | 8375.418 | 19896.649 | +11521.231 | [+7950.992, +15091.469] | worse |
| RMSE | 10 | 12382.583 | 21368.728 | +8986.145 | [+5783.708, +12188.582] | worse |
| RMSE | 15 | 12077.801 | 13913.676 | +1835.875 | [+39.271, +3632.479] | worse |
| MAE | 3 | 4093.944 | 7558.480 | +3464.536 | [+2521.645, +4407.426] | worse |
| MAE | 5 | 6786.359 | 13018.807 | +6232.448 | [+4466.631, +7998.264] | worse |
| MAE | 10 | 10114.854 | 14748.664 | +4633.810 | [+3055.190, +6212.431] | worse |
| MAE | 15 | 10335.943 | 11500.228 | +1164.285 | [-339.813, +2668.382] | noise |

v2graph is significantly worse than v1 at seven of the eight decisive cells, and within noise at the
eighth (MAE h15). It is better at none. At h3 the error roughly doubles (RMSE +96.8 percent), and at
h5 it more than doubles (RMSE +137.6 percent).

**PCC, printed only, carries no verdict.** The h3 correlation collapses from 0.436 to -0.014
(d -0.450). Adding the branch did not just raise the error, it destroyed the short-horizon
correlation the model had.

Because there is no winning horizon on both RMSE and MAE, criterion (a) fails and `decide()` returns
**FAIL** without needing the attribution stage.

## The attribution reading

The three v2 arms degrade almost identically. v2nograph and v2shuffled are worse than v1 by the same
large margins as v2graph:

| comparison | RMSE h3 d | RMSE h5 d | MAE h3 d | MAE h5 d |
|---|---|---|---|---|
| v2graph vs v1 | +5138.013 | +11521.231 | +3464.536 | +6232.448 |
| v2nograph vs v1 | +4592.425 | +11055.043 | +3187.244 | +6105.379 |
| v2shuffled vs v1 | +4261.700 | +10212.949 | +3010.100 | +5596.376 |

And every direct comparison between v2graph and the two controls is within noise on RMSE and MAE at
every horizon:

| comparison | metric | h | d | 95% CI | verdict |
|---|---|---|---|---|---|
| v2graph vs v2nograph | RMSE | 3 | +545.588 | [-245.477, +1336.653] | noise |
| v2graph vs v2nograph | RMSE | 5 | +466.187 | [-2297.745, +3230.120] | noise |
| v2graph vs v2nograph | MAE | 3 | +277.291 | [-185.715, +740.298] | noise |
| v2graph vs v2nograph | MAE | 5 | +127.069 | [-1062.308, +1316.446] | noise |
| v2graph vs v2shuffled | RMSE | 3 | +876.313 | [-261.870, +2014.495] | noise |
| v2graph vs v2shuffled | RMSE | 5 | +1308.282 | [-1782.226, +4398.789] | noise |
| v2graph vs v2shuffled | MAE | 3 | +454.436 | [-246.318, +1155.190] | noise |
| v2graph vs v2shuffled | MAE | 5 | +636.071 | [-800.266, +2072.408] | noise |

So the real map buys nothing over a wrong map or no map. The damage comes from the **deviation input
itself**, not from the graph on top of it. Whatever the branch learned from the deviations hurt, and
whether it mixed those deviations over the real neighbours, wrong neighbours, or no neighbours made no
measurable difference.

## Interpretation, carefully fenced

The pieces sit oddly next to each other and I state them plainly rather than reconciling them by
force. A model-free linear probe on the same COVID deviations gains about 5.3 percent of total
target variance at h3 (`Neighbour_Signal_2026-09-24.md`). A learned branch of about 10,000 parameters
fed those same deviations roughly doubled the model's short-horizon error.

That gap is **consistent with** the deviations being noise-dominated at the capacity of a learned
model: a pooled linear ridge with heavy regularisation can skim a small, real, linear signal off the
top, while a flexible branch trained by gradient descent latches onto the noise in the same input and
generalises it the wrong way. This is exactly the risk the protocol named before the run: "delta
assumes the model captures the full signal the data probe found. If v2 captures only part of it, the
real effect is smaller than delta and may be missed even at N seeds." Here the branch did not miss a
small effect, it actively harmed, which is a stronger version of the same warning.

**What is NOT established.** The exact failure mechanism is not pinned down. Three candidates are all
live and this test does not separate them: the branch overfitting the short 60-origin COVID train
period, a train-to-test regime shift in the COVID deviations, or the branch's gradients interfering
with v1's optimisation. Do not write any one of them as the reason.

### Did validation warn, or did test collapse alone?

Cheap and read-only, to tell a val-blind overfit apart from a regime shift, I scored the same
val-selected models on the validation split from the `__val.json` records (`score_split`,
`ablation/run_v2_deviation.py`) and paired them against v1 the same way.

On the **validation** split the three v2 arms **beat** v1 by about 10 to 20 percent at h3 and h5 on
both RMSE and MAE, every cell clearing the t-interval. On the **test** split the same arms **lose** by
73 to 138 percent. Validation did not warn at all; it pointed the opposite way. So this is not the
plainest "val-blind overfit of the training loss" story, where validation would have degraded too. The
collapse is entirely on the test side, which is what a regime shift between the validation origins and
the later test origins would look like. This favours the regime-shift reading over the val-blind-overfit
one, but it does not settle it: validation is the early-stopping split, so its level is optimistic by
construction (protocol section 4), and short-train overfit that happens to align with the validation
period would look the same. The mechanism stays open.

## Scope, per the protocol

- **Criterion (c): no dengue run.** The protocol gates the dengue arm behind a COVID PASS (section 8).
  COVID is a FAIL, so there is no dengue run and there will not be one under this protocol.
- **COVID only.** This says nothing about the other panels, the transfer trunk, or Ebola. The Ebola
  data probe already found no neighbour timing signal at h3 and h5 once outbreak size is controlled
  (`Neighbour_Signal_2026-09-24.md`), and v2 on Ebola was never in scope.
- **Not a replacement model.** This was a mechanism test, not a candidate for the paper's main model.
  It stays where it is.
- **What the protocol licenses us to say (FAIL row, section 7).** No effect found at N seeds. Because N
  was not capped (15 well under the 134 cap), the design had about 0.82 chance on h3 RMSE alone of
  finding the probe-implied delta. We may NOT say the deviation channel has no effect, or that the graph
  is useless in general. What we observed is stronger than "no effect": the channel measurably hurt.

## What this closes for the paper

The spatial-redesign question is closed for this paper. The obvious remedy for the compression finding
was tested under a pre-registered protocol and made the model roughly twice as inaccurate on COVID,
with a wrong map and no map doing equal damage. On this test the encoder's compression looks less like
a defect to fix than a defence against district deviations a learned branch could not turn into
accuracy. The exact reason the branch fails is not established, so the paper states the outcome and the
consistency, not a causal mechanism.

## Reproduce

```
conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --report
conda run --no-capture-output -n ebola-train python diagnostics/verify_v2_result_doc.py
conda run --no-capture-output -n ebola-train python diagnostics/verify_v2_result_doc.py --mutate
```

Records: `ablation/single/encoder__covid_us-states__seed{42..182}__{v2graph,v2nograph,v2shuf}.json`
(45 test records, 15 seeds x 3 arms), the matching `__v1ref` records for seeds 92 to 182 and the five
released v1 seeds in `results/single/`, plus every `__val.json` beside them. Stage-1 summary:
`ablation/misc/v2_stage1_summary.json`. Repro-check: `ablation/misc/v2_repro_check.json` (identical,
max relative difference 0.0).

# G5 Explainability: scoping note

**Written 2026-09-06.** Ordered by Week 4 work order section 8d, which asked for a scoping note and
explicitly not for code:

> *"SHAP over ST-GNN at 7165 regions = serious compute. Only 4 input channels, so unclear what we
> attribute over (channels? lags? neighbours?). Integrated gradients probably more tractable.
> Deliverable = scoping note, not code."*

This note answers the three questions the work order raised, sets the method, and states the three
things the evidence already on disk forbids us from claiming. It is the decision record that lets
the build start.

**Every number here was recomputed from the artifacts on 2026-09-06.**

---

## 1. Where G5 actually stands

G5 is the only REQUIRED goal with no code. It is required twice, at
`Final Internal Project Brief.md:74` and `:119`, and it is half of intersection property (d) at
`PROJECT.md:23-29`.

**No attribution code exists anywhere in our source.** The only SHAP-bearing files in the tree are
third-party baseline checkouts.

It is nonetheless printed as delivered in three live places:

| where | what it says |
|---|---|
| `PROJECT.md:39` | G5 row reads "SHAP (global + local); Week 5" |
| `PROJECT.md:206` | "SHAP global + local explanations" |
| `Reports/Phase1/RelatedWork_CompetitiveAnalysis_Benchmark.docx` | the "Ours / this work" row reads "yes - SHAP (glob.+loc.)" |

The third is the serious one. It is a comparison table **already with the client** asserting a
delivered property that does not exist. That is audit finding M12, and it needs retracting or
replacing whichever way this note is decided.

---

## 2. Why not SHAP

Three reasons, in order of weight.

**It does not fit the compute.** KernelSHAP is model-agnostic and needs a large sample of coalitions
per explained instance. Our largest panel is dengue at **7,165 nodes over 1,409 steps**. There is no
version of that which is a one-day task, and the work order already flagged it.

**GradientSHAP would fit, and buys almost nothing over integrated gradients.** Both are path-attribution
methods over the same input tensor. GradientSHAP is integrated gradients with a sampled baseline
distribution and noise. On a **4-channel, 20-lag** surface the extra machinery adds variance to
manage and no interpretive power. If we are spending the day, spend it on the faithfulness check
rather than on the sampler.

**"SHAP" in the client-facing table implies a property we would not be delivering anyway.** Shapley
values carry an axiomatic efficiency guarantee. A sampled gradient approximation on a
non-independent, autocorrelated time-series input does not honour it. Calling the output SHAP would
be the kind of sentence the audit's "unsafe to claim" list exists to stop.

**Decision: integrated gradients, not SHAP.** Say so in the comparison table rather than quietly
letting "SHAP" stand.

---

## 3. What we attribute over, which the work order left open

The trunk's input surface is fixed and small. `transfer_view()` slices exactly four channels on
every panel, and the window is 20 steps, so a single forecast origin is a **`[N, 20, 4]`** tensor:

| index | channel | note |
|---|---|---|
| 0 | `incidence_norm` | log1p then z-score |
| 1 | `sin_doy` | |
| 2 | `cos_doy` | |
| 3 | `obs_mask` | see the warning below |

The Ebola bundles carry a fifth channel, `deaths_norm`, which `core_feature_idx` excludes from the
transfer view. The trunk never sees it, so it is not an attribution target.

That gives three attribution axes and they are all cheap:

- **Channel**, 4 of them. Global read, mean absolute attribution per channel per panel.
- **Lag**, 20 of them. Global read, and the one that answers "how far back does it look".
- **Neighbour**, by edge ablation rather than gradient. This is the axis the paper actually poses:
  where does a forecast for a district we have never observed come from?

---

## 4. What the checkpoints allow, and what they do not

**118 trained checkpoints are on disk.** Attribution is pure inference against them: no retraining,
no GPU night, no new results records.

| family | checkpoints |
|---|---|
| single-disease encoders, five panels | 26 |
| Ebola arms (`alldev`, `ebola_L12`, `ebola_L20`) | 15, plus 3 smoke |
| LDO3 trunks | 15, plus 3 full-budget |
| ANIL and its controls, plus the legacy fold | the balance |

One gap matters. The **single-disease reference arm never saved its predictions**, only metrics and
sufficient statistics. Attribution needs a forward pass, which the checkpoints give us, so this is
not blocking. It does mean attribution must be recomputed from weights rather than read off an
archive, and the numbers cannot be cross-checked against a stored prediction.

---

## 5. Three things the evidence forbids us from claiming

These are not stylistic. Each one contradicts a measurement already on disk.

### 5.1 Neighbour attribution may not be framed as a source of accuracy

The gate-off ablation is decisive and it is negative. Over the 60 cells, recomputed from
`Reports/gate_ablation.log`:

| metric family | cells | graph helps | graph hurts | within noise |
|---|---|---|---|---|
| error (RMSE + MAE) | 40 | **0** | 8 | 32 |
| correlation (PCC) | 20 | 6 | 1 | 13 |
| all | 60 | 6 | 9 | 45 |

**The spatial channel helps error in zero of forty cells.** It helps correlation in 6 of 20. So any
neighbour attribution must be written as *where the model draws from*, never as *what makes it
accurate*. The honest framing is that message passing shapes the correlation structure and not the
magnitude, which is what the ablation measured.

One caveat travels with the ablation itself and should travel with anything built on it: setting the
gate to zero removes neighbour mixing but keeps the LTR degree feature, so the tally bounds the value
of **neighbour information**, not of the graph in total.

### 5.2 The `obs_mask` channel is a disease identifier, and it is not constant on dengue

I checked all five panels directly. This corrects a claim I made earlier in planning:

| panel | `obs_mask` | fraction observed |
|---|---|---|
| influenza_japan | **constant 1.0** | 1.0000 |
| influenza_us-regions | **constant 1.0** | 1.0000 |
| influenza_us-states | **constant 1.0** | 1.0000 |
| covid_us-states | **constant 1.0** | 1.0000 |
| dengue | varies | **0.2175** |
| ebola_L12 / L20 | varies | **0.4095** |

Two consequences, and they point in opposite directions.

**Attribution to `obs_mask` is structurally zero on four of five panels**, because a constant input
has no gradient to attribute. Any per-channel bar chart will show a near-zero `obs_mask` bar on the
influenza panels and COVID, and that is an artefact of the data, not a finding about the model. Say
so before showing the chart.

**On dengue it is not an artefact, it is a leak.** Channel 3 is constant on four panels and varies on
two, which makes it a partial disease identifier inside the block we call disease-agnostic. Only 21.75
percent of dengue input cells are observed. That is already recorded in `Doubt.md` section 3.3 and
belongs in the limitations regardless of G5. Attribution is likely to make it visible, which is a
reason to do it rather than a reason not to.

### 5.3 Attribution explains a model whose transfer mechanisms did not work

The paper's standing result is that the shared representation transfers well enough to beat naive
floors and carries its calibration, while every mechanism added to improve transfer fails to help.
An explainability section that reads as though the model works well would contradict the rest of the
paper. G5's job here is to describe what the model attends to, not to argue that it is right to.

---

## 6. The build, when it happens

Not part of this note's deliverable. Recorded so the scope is unambiguous.

**Method.** Integrated gradients over the `[N, 20, 4]` input, per panel, from the existing
checkpoints. Global reads are mean absolute attribution per channel and per lag.

**Faithfulness cross-check.** Inference-time occlusion: zero a channel or a lag band and measure the
change in the forecast. Gradient attribution and occlusion should agree on the top channel and the
top lag band. Where they disagree, report the disagreement rather than picking the friendlier one.
This is the part of the day worth protecting.

**Spatial.** Neighbour edge ablation on **Ebola only**, 61 districts over 52 weeks, framed per 5.1.
Ebola is the panel where the question is interesting, because some districts are never observed.

**Local.** One case study, an Ebola district around the national peak week.

**Existing asset.** `figures/gate.pdf` and `gate.png` already exist, built by `gate_figure.py`. The
learned gate scaling with panel density is an interpretable result the client asked for and we are
under-using. It belongs in the G5 section and it costs nothing.

**Falsification test, stated before the run.** The expected ordering is that recent lags and
incidence dominate, and that the seasonality channels carry more weight on influenza than on Ebola,
since influenza is strongly seasonal and a 52-week outbreak has no seasonal cycle to learn. If that
ordering does not appear, distrust the attribution before distrusting the epidemiology.

**Cost.** One day. It is a section of the paper, not a research programme.

---

## 7. What needs a decision

1. **Adopt integrated gradients as G5 and say so.** The alternative is renegotiating the goal, which
   is also a legitimate answer, but it has to be an answer rather than a default.
2. **Retract or amend the SHAP row** in the client-held comparison table, and correct `PROJECT.md:39`
   and `:206`. This is required whichever way item 1 goes, because the current text asserts a
   delivered property that does not exist.
3. **Confirm the honesty framing in 5.1 is acceptable.** A neighbour-attribution figure that cannot
   be described as explaining accuracy is a weaker deliverable than the brief implies, and the client
   should know that before it is built rather than after.

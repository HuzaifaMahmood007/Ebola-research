**Internal Project Brief**

*A Generalizable Spatio-Temporal Framework for Emerging Infectious Disease Forecasting*

Engineering scope · research goals · milestones · deliverables · deadlines

|  |  |
| --- | --- |
| Audience | Internal — engineering team (implementation lead) |
| Purpose | Single reference for what we have decided, what needs to be built, who does what, and by when |
| Duration | 6 weeks, full-time (assume 5 working days/week → 30 working days). Start date: \_\_\_\_\_\_\_\_\_\_ |
| Pricing | Not included — this is an internal working document |
| Target outcome | A reproducible, novel methodological contribution suitable for a high-impact (ISI) journal |

|  |
| --- |
| **How to read the tags in this document**  Items are labelled so priorities are unambiguous:  **[REQUIRED]** — a decided, must-deliver requirement (client need or agreed scope).  **[SUGGESTION]** — our recommendation to strengthen the work; follow unless told otherwise, but it is our proposal rather than a fixed client requirement.  **[OUT OF SCOPE]** — deliberately excluded from this engagement. |

# 1. Project overview

We are building a **general, disease-agnostic framework for emerging infectious disease forecasting**. The architecture is not specialised to any single disease; it learns **transferable spatio-temporal representations** from data-rich diseases and adapts, with minimal modification, to a data-scarce emerging disease. Data-rich diseases are used to develop and validate the methodology; **Ebola is the headline case study** demonstrating generalisation to a scarce, real-world emerging disease.

**Why it is novel:** recent state-of-the-art models are largely single-disease or stay within one disease family, and several explicitly flag cross-disease transferability as open future work. A framework that is disease-agnostic by construction, transfers across diseases, adapts few-shot to a scarce disease, and reports calibrated uncertainty addresses a clear, identifiable research gap.

# 2. Research goals & objectives

The work is judged against these goals:

* G1 — A disease-agnostic spatio-temporal forecasting architecture operating on a standardised input schema. **[REQUIRED]**
* G2 — Transferable representations learned across multiple data-rich diseases. **[REQUIRED]**
* G3 — Demonstrated generalisation to a data-scarce emerging disease (Ebola) via few-shot adaptation. **[REQUIRED]**
* G4 — Calibrated uncertainty quantification on all forecasts. **[REQUIRED]**
* G5 — Explainability of forecasts. **[REQUIRED]**
* G6 — Competitive-or-better performance vs. recent SOTA baselines, re-run under identical conditions. **[REQUIRED]**
* G7 — A reproducible, submission-ready contribution (code + manuscript). **[REQUIRED]**

# 3. Scope — in and out

|  |  |
| --- | --- |
| **In scope** | **Out of scope** |
| * Spatio-temporal forecasting (cases / deaths) * Calibrated uncertainty quantification * Explainability of forecasts * Cross-disease transfer & few-shot adaptation * Empirical benchmark + full literature review | ✕ Regional risk stratification  ✕ Intervention priority ranking  *Reason: no agreed ground-truth labels, so they cannot be trained or validated rigorously and would weaken the paper. Natural future extension.* |

# 4. Methodology — what to build

Displaybarcode”Four design pillars, none specific to one disease:

* **Disease-agnostic spatio-temporal encoder** — temporal + graph components over a standardised schema (normalised incidence, spatial adjacency, generic covariates). **[REQUIRED]**
* **Epidemiology-informed inductive bias** — a metapopulation / compartmental-aware component (or physics-informed loss) for data efficiency and generalisation. **[SUGGESTION]**
* **Transferable representation learning with few-shot adaptation** — meta-learning / domain-generalisation so the encoder captures disease-invariant dynamics and adapts with minimal data. **[REQUIRED]**
* **Calibrated uncertainty** — conformal prediction is recommended (model-agnostic, post-hoc); a Bayesian approach is an alternative but would move UQ into the model build in Week 3. **[SUGGESTION]”qr**

# 5. Datasets

* **Dengue** — primary development disease (large, dense, spatially-resolved). **[REQUIRED]**
* **Influenza** — second development disease, to make the “general framework” claim credible (avoids a dengue-only model). **[SUGGESTION]**
* **COVID-19** — optional third development disease for added diversity. **[SUGGESTION]**
* **Ebola** — held-out, data-scarce case study for the few-shot generalisation test; assembled from outbreak situation reports. **[REQUIRED]**

|  |
| --- |
| **Note for the engineer**  All diseases must be mapped onto one **standardised input schema** so the architecture stays disease-agnostic. Run a data audit early (coverage, resolution, missingness) — Ebola data is small, so confirm the usable signal before relying on it. |

# 6. Deliverables (what the client needs)

All deliverables below are **required** unless tagged otherwise:

* Working, documented framework (source code) operating on the standardised schema
* Benchmark report: reproduced SOTA baselines vs. our framework on identical data
* Literature review / related-work section (structured SOTA comparison, research gap, how our method addresses limitations) — now fully owned by us
* Cross-disease transfer results (leave-one-disease-out)
* Ebola few-shot case study results, with prediction intervals
* Ablation studies + robustness / sensitivity analysis
* Explainability outputs (e.g. SHAP, global + local)
* Submission-ready manuscript (methods, experiments, results, discussion)
* Reproducibility package: code, configs, README, instructions

# 7. Milestones, tasks & deadlines

Six weekly milestones, full-time. “Target” shows working-day ranges from the agreed start date.

|  |  |
| --- | --- |
| **Week 1 — Benchmark, literature review & problem setup** | |
| Target | **Days 1–5** |
| Objective | Establish the research gap, the formal problem, and reproducible baselines. |
| Engineer tasks | * Write the full literature review & structured SOTA comparison *(now fully ours)* **[REQUIRED]** * Select 4–6 baselines (relevance, recency/impact, method-family coverage, public code, comparability); share shortlist with client before locking **[REQUIRED]** * Reproduce baselines using official public implementations where available **[REQUIRED]** * Define the formal problem, evaluation metrics & validation protocol **[REQUIRED]** * Specify the standardised, disease-agnostic input schema **[REQUIRED]** |
| Deliverables | * Literature-review / related-work draft * Baseline shortlist + selection rationale * Problem definition & evaluation protocol * Input schema specification |

|  |  |
| --- | --- |
| **Week 2 — Multi-disease data engineering & harmonisation** | |
| Target | **Days 6–10** |
| Objective | Build clean, harmonised, leak-free datasets. |
| Engineer tasks | * Acquire, parse & clean dengue data **[REQUIRED]** * Add influenza (and optionally COVID-19) development data **[SUGGESTION]** * Assemble the Ebola case-study dataset from outbreak situation reports **[REQUIRED]** * Map all diseases onto the standardised schema; build preprocessing + train/val/test splits **[REQUIRED]** * Run a data audit (coverage, resolution, missingness) — confirm usable Ebola signal **[REQUIRED]** |
| Deliverables | * Harmonised multi-disease datasets * Reproducible data pipeline + documentation * Data audit note |

|  |  |
| --- | --- |
| **Week 3 — Core framework build & development training** | |
| Target | **Days 11–15** |
| Objective | Implement and validate the core model on the data-rich diseases. |
| Engineer tasks | * Implement the disease-agnostic spatio-temporal encoder (temporal + graph) **[REQUIRED]** * Integrate the epidemiology-informed component **[SUGGESTION]** * Train on the data-rich diseases + hyperparameter tuning **[REQUIRED]** * Re-run the selected baselines on the harmonised data; benchmark vs. our framework **[REQUIRED]** |
| Deliverables | * Working framework (v1) + training code * Benchmark results vs. baselines on development diseases * Technical design note |

|  |  |
| --- | --- |
| **Week 4 — Transferable representations, meta-learning & few-shot adaptation** | |
| Target | **Days 16–20** |
| Objective | Add the core methodological contribution: transfer + few-shot adaptation. |
| Engineer tasks | * Implement meta-learning / domain-generalisation training **[REQUIRED]** * Disease-invariant representation learning **[REQUIRED]** * Leave-one-disease-out transfer evaluation across development diseases **[REQUIRED]** * Build the few-shot adaptation protocol for an unseen disease **[REQUIRED]** |
| Deliverables | * Framework (v2) with transfer module * Leave-one-disease-out transfer results * Few-shot adaptation protocol |

|  |  |
| --- | --- |
| **Week 5 — Ebola case study, uncertainty & explainability** | |
| Target | **Days 21–25** |
| Objective | Demonstrate generalisation to a data-scarce disease, with calibrated uncertainty and explanations. |
| Engineer tasks | * Run the Ebola out-of-distribution few-shot case study **[REQUIRED]** * Add calibrated uncertainty quantification (conformal recommended) + calibration checks **[REQUIRED]** * Produce explainability outputs (SHAP global + local) **[REQUIRED]** * Compile case-study results with prediction intervals **[REQUIRED]** |
| Deliverables | * Ebola case-study results with uncertainty * Calibration evaluation * Explainability outputs |

|  |  |
| --- | --- |
| **Week 6 — Ablations, robustness, writing & documentation** | |
| Target | **Days 26–30** |
| Objective | Establish robustness, then produce a submission-ready paper and reproducible release. |
| Engineer tasks | * Ablations: with vs. without epidemiology-informed component; standard vs. meta-learning **[REQUIRED]** * Robustness / sensitivity analysis; finalise all tables & figures **[REQUIRED]** * Draft the full manuscript (incl. literature review, methodology, experiments, results, discussion) **[REQUIRED]** * Finalise source code + README + reproducibility package; submission support **[REQUIRED]** |
| Deliverables | * Ablation & robustness study * Submission-ready manuscript * Reproducibility package |

|  |
| --- |
| **Post-submission (separate engagement)**  Reviewer response & one revision round (point-by-point response, requested re-runs, updated figures/tables) is handled **separately** after submission and is not part of the 6-week core schedule. |

# 8. Timeline at a glance

|  |  |  |  |
| --- | --- | --- | --- |
| **Week** | **Focus** | **Days** | **Cumulative** |
| W1 | Benchmark, literature review & problem setup | 1–5 | Day 5 |
| W2 | Multi-disease data engineering | 6–10 | Day 10 |
| W3 | Core framework build & training | 11–15 | Day 15 |
| W4 | Transfer, meta-learning & few-shot | 16–20 | Day 20 |
| W5 | Ebola case study, uncertainty & explainability | 21–25 | Day 25 |
| W6 | Ablations, robustness, writing & docs | 26–30 | Day 30 |

|  |
| --- |
| **Sequencing note**  Milestones run in dependency order, each building on the previous. Two activities span phases by design: baseline reproduction begins in Week 1 and is re-run on the harmonised data in Week 3 (identical-conditions comparison), and manuscript drafting starts in parallel from Week 4 so Week 6 is assembly and interpretation rather than writing from scratch. |

# 9. Roles & responsibilities

|  |  |
| --- | --- |
| **Party** | **Responsible for** |
| **Our team (engineering)** | All implementation, data engineering, experiments, the full literature review & benchmark, manuscript drafting, and the reproducibility package. |
| **Client (Nora)** | Direction & feedback, milestone-gate review and sign-off, decisions on target journal and authorship. Note: the client is no longer running a separate benchmark — the full literature review is now ours. |

# 10. Evaluation & validation protocol

* **Baselines re-run under identical conditions** (same datasets, splits, preprocessing, metrics, horizon) — the basis of any SOTA claim.
* **Development evaluation** on the data-rich diseases with standard forecasting metrics.
* **Generalisation** via leave-one-disease-out transfer, and few-shot adaptation to held-out Ebola.
* **Uncertainty** reported as calibrated prediction intervals, with calibration checks.
* **Ablations** isolating the epidemiology-informed component and the meta-learning contribution.

# 11. Required vs. suggested — quick reference

|  |  |
| --- | --- |
| **Item** | **Status** |
| Forecasting + calibrated uncertainty + explainability | **Required** |
| Disease-agnostic encoder + meta-learning transferable representations | **Required** |
| Ebola few-shot generalisation case study | **Required** |
| Full literature review + empirical benchmark (re-run baselines) | **Required** |
| Dengue as a development disease | **Required** |
| Adding influenza (and optionally COVID-19) as further development diseases | **Suggestion** |
| Epidemiology-informed component | **Suggestion** |
| Conformal UQ (vs. Bayesian alternative) | **Suggestion** |
| Regional risk stratification / intervention priority ranking | **Out of scope** |

# 12. Risks & mitigations

* **Ebola data scarcity** — confirm usable signal in the Week-2 audit; the design treats Ebola as a few-shot test precisely because the data is thin.
* **Tight 6-week schedule** — Weeks 3–4 are the densest; protect them by freezing scope and starting the write-up early.
* **Baseline reproduction** — prefer official code; where none exists, validate a from-scratch reimplementation against the paper’s reported numbers before comparing.
* **UQ method choice** — decide conformal vs. Bayesian before Week 3, as Bayesian moves UQ into the model build.

# 13. Definition of done

* Framework runs end-to-end on the standardised schema and is documented
* Our model is benchmarked against the agreed baselines on identical data
* Transfer + few-shot Ebola results produced with calibrated uncertainty
* Ablations and explainability completed
* Manuscript + reproducibility package ready for submission

# 14. Open questions to confirm before Week 1

* Final development-disease set: dengue + influenza only, or also COVID-19?
* UQ approach: conformal (recommended) or Bayesian?
* Start date and target journal.
* Authorship & IP terms.
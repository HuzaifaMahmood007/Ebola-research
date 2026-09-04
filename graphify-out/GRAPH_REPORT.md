# Graph Report - Ebola-Research  (2026-09-02)

## Corpus Check
- 129 files · ~279,575 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1688 nodes · 3356 edges · 94 communities (83 shown, 9 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 230 edges (avg confidence: 0.87)
- Token cost: 916,754 input · 0 output

## Community Hubs (Navigation)
- Overnight GPU Queue Runners
- LDO3 Transfer Training Loop
- ANIL Meta-Learning Trainer
- Shared Encoder Architecture
- Joint Multi-Disease Training
- Run Comparison and Adapters
- Schema Normalisation Layer
- Conformal Calibration Engine
- Scalers, Splits and Adjacency
- Baseline Staging and Export
- Leakage Gate Test Suite
- Encoder Option Menu
- Metric Layer (WIS, CRPS, PIT)
- LDO3 Report Builder
- Day 13 Review Decisions
- Ebola Support and Calibration
- Dataset Build and Freeze
- Ebola Support Pre-Registration
- Dengue Loader Tests
- Ebola Case Study Runner
- Data Audit and Corrections
- Published Baseline Scorecard
- Bundle Loading Interface
- Week 4 Decision Log
- Baseline Comparability Rules
- Bootstrap Analysis Engine
- COVID and MTGNN Export Probes
- Results Doc Verifier
- Negative Transfer Findings
- Ebola Schema Tests
- Provenance and Reporting Rules
- Data Quality Metrics
- Corrected Ebola Interval
- Reporting Standards Matrix
- Country-Macro Scoring Decisions
- Results Matrix Assembly
- Naive Floors and Fold Design
- Bundles and Conda Environments
- Baseline Reproduction Failures
- Frozen Encoder Hyperparameters
- Baseline Scoring and Comparison
- Gate Figure Panels
- Learned Spatial Gate Ablation
- Influenza Loader and Covariates
- Negative Control Test Gates
- Coverage Filter Probe
- Single-Run Epoch Recovery
- Seed Ensembling Comparison
- Ebola Report Generator
- COVID Skill Investigation
- Adapter Capacity Probe
- Ebola Uncertainty Table
- Artifact Path Routing
- Transfer View Feature Block
- Spatial Mixer Design Constraints
- Overnight Transfer Orchestrator
- Evaluation Protocol Freeze
- Encoder Rescoring Backfill
- Gate Figure Generator
- Meta-Learning Algorithm Choice
- LDO3 Negative Transfer Results
- Ebola Few-Shot Split Protocol
- Factorised Temporal-Spatial Encoder
- Quantile Archiving and Bias Correction
- Baseline Format Exporter
- Ebola One-Way Door
- Client Work Order and Folds
- Project Goals and Pillars
- Experiment Families and Floors
- LDO3 Full-Budget Check
- COVID Data Loader
- Calibrated Uncertainty Route
- COVID Split Probe
- Graph Topology Ablations
- ANIL Adaptation Surface
- The Honest Thesis
- Per-Node Score Aggregation
- Single-Disease Reference Reporting
- Day 11 Variance Diagnostics
- GADM Shapefile Fetcher
- Japan Calendar Pin
- Japan Seasonality Ablation
- MLP Adapter Surface
- Schema Backfill Script
- Cross-Border Graph Edges
- Diagnostics Package Init
- Loaders Package Init
- Weekly Epi-Week Cadence
- Tests Package Init
- Train Package Init
- MSGNN Not Reproducible
- STOEP Excluded

## God Nodes (most connected - your core abstractions)
1. `rpath()` - 57 edges
2. `load()` - 28 edges
3. `load_ebola()` - 27 edges
4. `SharedEncoder` - 26 edges
5. `window_slice()` - 24 edges
6. `_fit_trunk()` - 24 edges
7. `DiseaseTensors` - 23 edges
8. `_canon()` - 23 edges
9. `Adapter` - 22 edges
10. `pinball_loss()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `M1 Estimand Mismatch Fix (ebola_ci.py)` --semantically_similar_to--> `STOEP Table-Selection Error (Flu row read instead of COVID row)`  [INFERRED] [semantically similar]
  Resume.md → Published Model benchmarks/STOEP.md
- `Learned Spatial Gate Figure` --conceptually_related_to--> `Mask-Aware Adjacency Leaves the Graph Mostly Empty`  [INFERRED]
  figures/gate.pdf → progress/summaries/Doubt.md
- `Identical-Graph Gate Control` --references--> `COVID-19 us-states Bundle`  [INFERRED]
  figures/gate.pdf → progress/summaries/Doubt.md
- `Gate Figure LTR Confound Risk` --rationale_for--> `Learned Spatial Gate Figure`  [INFERRED]
  progress/summaries/Doubt.md → figures/gate.pdf
- `Gate Figure Deliverable` --rationale_for--> `Learned Spatial Gate Figure`  [INFERRED]
  progress/summaries/Phase3_Week4_Work_Order.md → figures/gate.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Mechanical Disease-Agnosticism: C1/C2/C3 Over the Transfer View** — readme_shared_encoder, readme_invariant_c1, readme_invariant_c2, readme_invariant_c3, project_transfer_view, project_disease_agnosticism_in_code [EXTRACTED 1.00]
- **The Boundary-Conditions Result Set** — resume_boundary_conditions_thesis, resume_negative_transfer, resume_graph_does_not_help, resume_anil_null, resume_calibration_transfers, resume_prereg_criterion_not_met [EXTRACTED 1.00]
- **Published-Number Validation Suite (D10)** — published_model_benchmarks_readme_validation_rule_d10, published_model_benchmarks_readme_tolerance_band, published_model_benchmarks_cola_gnn_colagnn, published_model_benchmarks_epignn_epignn, published_model_benchmarks_heatgnn_heatgnn, published_model_benchmarks_mtgnn_mtgnn, published_model_benchmarks_mepognn_mepognn, published_model_benchmarks_msgnn_msgnn, published_model_benchmarks_stoep_stoep, published_model_benchmarks_ltgcn_gtgcn_ltgcn [EXTRACTED 1.00]
- **Ebola Support-Set Freeze Packet** — progress_decisions_ebola_prereg_ebola_prereg, progress_decisions_ebola_prereg_primary_arm_l12, progress_decisions_ebola_prereg_secondary_arm_l20, progress_decisions_ebola_prereg_freeze_ebola_arms, progress_decisions_decisions_d16_ebola_support_length, progress_decisions_review_doc_preregistration_score_once [EXTRACTED 1.00]
- **Negative Cross-Disease Transfer Evidence Chain** — progress_outcomes_results_summary_lodo_probe_corrected, progress_decisions_decisions_d12_transfer_negative, progress_outcomes_ldo3_results_negative_transfer_finding, progress_outcomes_ldo3_results_option_b_decision, progress_decisions_ebola_prereg_expectation_e1, progress_decisions_meta_learning_decision_rigorous_negative_result [INFERRED 0.85]
- **Disease-Identity Leakage Controls** — progress_decisions_client_decisions_c5_influenza_adjacency_diagonal, progress_decisions_client_decisions_c9_geographic_covariates_withheld, progress_decisions_client_decisions_b6_mobility_not_used, progress_decisions_meta_learning_decision_domain_generalisation, progress_decisions_encoder_decision_filter_a_size_agnostic, progress_decisions_ebola_prereg_c8_guard [INFERRED 0.85]
- **The three coded constraints carried from the data layer into the model** — progress_planning_data_audit_three_modelling_constraints, progress_planning_data_audit_isolated_influenza_nodes, progress_planning_data_audit_disease_agnosticism_rule, progress_planning_phase3_developer_execution_guide_scaler_refit_trap, progress_planning_data_pipeline_model_obligations, progress_planning_encoder_architecture_plan_spatial_mixer [EXTRACTED 1.00]
- **The Ebola cumulative-to-weekly defect chain and its gates** — progress_planning_data_audit_clip_defect, progress_planning_data_audit_week_zero_backlog, progress_planning_data_audit_gap_lumping, progress_planning_data_audit_running_maximum_differencing, progress_planning_data_audit_mass_conservation_gate, progress_planning_data_audit_new_cases_crosscheck [EXTRACTED 1.00]
- **The Ebola few-shot protocol surface, from support definition to single scoring** — progress_planning_data_audit_calendar_prefix_support, progress_planning_data_audit_ebola_frozen_arms, progress_planning_encoder_architecture_plan_ebola_zero_support_origins, progress_planning_phase3_week3_developer_execution_guide_confirm_p7_split_protocol, progress_planning_phase3_developer_execution_guide_frozen_ebola_protocol, progress_planning_phase3_developer_execution_guide_ebola_scored_once, progress_planning_phase3_developer_execution_guide_run_ebola_once [INFERRED 0.95]
- **Ebola Few-Shot Protocol Constraint Chain** — progress_summaries_day13_summary_ebola_zero_support_origins, progress_summaries_day13_summary_p7_split_protocol, progress_summaries_day15_progress_ebola_support_set_decision, progress_summaries_day15_progress_reporting_blackout, progress_summaries_day15_progress_ebola_scaler_miscalibration, progress_summaries_priority_fix_progress_ebola_prereg_a7 [INFERRED 0.85]
- **Cross-Disease Transfer Negative Evidence Chain** — progress_summaries_phase3_week3_results_and_direction_finding2_joint_negative_transfer, progress_summaries_day15_progress_transfer_negative_headline, progress_summaries_pair_run_analysis_attributable_negative_transfer, progress_summaries_priority_fix_progress_symmetric_ensemble_verdict, progress_summaries_priority_fix_progress_zero_shot_inversion [INFERRED 0.85]
- **Spatial Channel Value Investigation** — progress_summaries_day13_summary_learned_spatial_gate_p8, progress_summaries_doubt_gate_off_gonogo, progress_summaries_doubt_degree_disease_identifier, progress_summaries_doubt_mask_aware_adj_empty_graph, progress_summaries_priority_fix_progress_gate_off_ablation_result, figures_gate_figure [INFERRED 0.85]
- **Five-Dataset Gate Comparison** — figures_gate_panel_a_gate_distribution, figures_gate_flu_us_regions, figures_gate_flu_japan, figures_gate_flu_us_states, figures_gate_covid_us_states, figures_gate_dengue [EXTRACTED 1.00]
- **Structure-Not-Disease Argument** — figures_gate_identical_graph_control, figures_gate_flu_us_states, figures_gate_covid_us_states, figures_gate_gate_reads_graph_structure_not_disease, figures_gate_learned_spatial_gate [EXTRACTED 1.00]
- **Caveats Limiting the Gate-Scaling Claim** — figures_gate_density_region_count_confound, figures_gate_no_fitted_trend_caveat, figures_gate_mean_degree_sparsity, figures_gate_panel_b_gate_vs_region_count, figures_gate_panel_c_gate_vs_density, figures_gate_dengue [EXTRACTED 1.00]

## Communities (94 total, 9 thin omitted)

### Community 0 - "Overnight GPU Queue Runners"
Cohesion: 0.05
Nodes (71): main(), parse(), How much of its training budget did each run actually use? (Client task 2,…, The parser attributes lines to the right run, and the ambiguous adapter label…, Last point reached, and the point the kept checkpoint was selected at., _row(), _selfcheck(), Track (+63 more)

### Community 1 - "LDO3 Transfer Training Loop"
Cohesion: 0.07
Nodes (65): invert_scaler(), Model space -> real counts, for RMSE/MAE reporting., Score one frozen arm at one seed: zero-shot, then few-shot. Returns the record…, persistence + support_mean on the arm's scored cells. Deterministic, so seed is…, run_arm(), run_floors(), Score one dataset's test fold through the shared encoder + its own adapter…, _test_dataset() (+57 more)

### Community 2 - "ANIL Meta-Learning Trainer"
Cohesion: 0.08
Nodes (50): artifact(), cap_reference(), compare(), country_index(), episode(), _features(), inner_adapt(), load_done() (+42 more)

### Community 3 - "Shared Encoder Architecture"
Cohesion: 0.09
Nodes (29): Per-disease adaptation surface: FiLM affine + quantile head, and the pinball…, Architecture constants -- the single code source of truth…, Gate, node_indexed_params(), SharedEncoder: TCN + LTR degree feature, then gated inductive spatial mixing.…, h_out = (1-g)*h + g*h_spatial, g = sigmoid(MLP(h)) per node: data-dependent,…, The reportable gate quantity: scale-free, unlike raw g, which stays debug-only., C2 audit: params whose shape touches a graph size. Empty list = clean. (+21 more)

### Community 4 - "Joint Multi-Disease Training"
Cohesion: 0.09
Nodes (45): pinball_loss(), pred [N,H,Q], target [N,H], mask [N,H] (1=observed & in phase), MODEL space.…, Windowing: an origin t -> input slice + per-horizon targets/masks. No instance…, Input window for origin t: Z[:, t-w+1 : t+1, :] -> [N, w, F]. For t < W-1 the…, For origin t: targets [N,H] (model space) and mask [N,H] (observed AND in-phase…, targets_and_mask(), window_slice(), no_grad (+37 more)

### Community 5 - "Run Comparison and Adapters"
Cohesion: 0.06
Nodes (40): _block(), _film_mlp(), FiLMMLPAdapter, gains_vs_control(), improvement(), interval(), _interval_table(), _mlp() (+32 more)

### Community 6 - "Schema Normalisation Layer"
Cohesion: 0.08
Nodes (34): Curated OpenDengue -> GADM 4.1 name maps for the dengue graph. Geometry joins…, Check that every alias resolves to a real GADM key, guarding against typos in…, validate_aliases(), Report per-country weekly coverage in the OpenDengue extract, before any…, is_blob(), main(), Confirm that the Ebola compilation carries usable signal, before anything is…, main() (+26 more)

### Community 7 - "Conformal Calibration Engine"
Cohesion: 0.10
Nodes (37): aci_run(), aci_worst_case(), _apply(), apply_to_archive(), _calibration_pool(), conformal_lambda(), evaluate(), freeze() (+29 more)

### Community 8 - "Scalers, Splits and Adjacency"
Cohesion: 0.07
Nodes (35): Refit the scaler on `fit_mask`; returns (X, y, scaler) as ONE object (guide…, DatetimeIndex, test_no_future_leak_in_mask_semantics(), test_per_country_split_gives_every_country_train_and_is_chronological(), test_per_country_split_scaler_is_leakage_safe(), test_per_country_split_single_obs_node_takes_its_country_boundary(), test_scaler_is_leakage_safe(), test_scaler_roundtrip_and_zero_variance_guard() (+27 more)

### Community 9 - "Baseline Staging and Export"
Cohesion: 0.11
Nodes (34): _add_self_loops(), _cleanup_probe(), _cmd(), _env(), main(), nan_test(), pred_name(), probe() (+26 more)

### Community 10 - "Leakage Gate Test Suite"
Cohesion: 0.13
Nodes (33): _fit_mask(), _gate(), gate_mask_semantics(), gate_mass_conservation(), gate_no_future_leakage(), gate_phase_purity(), gate_rolling_origins(), gate_scaler_provenance() (+25 more)

### Community 11 - "Encoder Option Menu"
Cohesion: 0.09
Nodes (33): B6: Mobility Data Not Used for Any Disease, C5: Influenza Adjacency Diagonal Zeroed, C9: Geographic Covariates Withheld from the Shared Model, AGCRN (Bai et al., NeurIPS 2020), CAPE (arXiv:2502.03393), DCRNN (Li et al., ICLR 2018), Menu of Disease-Agnostic Shared Spatio-Temporal Encoders, Filter A: No Node Identities, Size-Agnostic (+25 more)

### Community 12 - "Metric Layer (WIS, CRPS, PIT)"
Cohesion: 0.13
Nodes (26): coverage(), crps(), _demo(), _interval_pairs(), interval_score(), interval_width(), _median_index(), _nrmse() (+18 more)

### Community 13 - "LDO3 Report Builder"
Cohesion: 0.13
Nodes (27): attributable(), best_naive(), build_markdown(), covid_fold_stats(), _expected_artifacts(), _load_json(), load_points(), main() (+19 more)

### Community 14 - "Day 13 Review Decisions"
Cohesion: 0.11
Nodes (28): A1: Finest-Available Dengue Spatial Resolution, C4: Taiwan Aggregated from Townships to Counties, D2: Twenty-Nine Japanese Prefectures Report Zero Dengue, Block-Diagonal Joint Supergraph, C8 Runtime Ebola-Exclusion Guard, Cell-Pooled versus Node-Averaged Metric Caveat, Day 13 Review Follow-Up Decisions, Persistence-Anchor Delta Target (open) (+20 more)

### Community 15 - "Ebola Support and Calibration"
Cohesion: 0.10
Nodes (26): Days 11-13 Advisory Code Review, bundles.py Bundle Interface, Ebola Disease-Pooled Scaler Exception, Ebola Has Zero Support Origins, FiLM Adapter plus Quantile Head, Origin Phase Counts Legitimately Over-Sum, P7 Ebola Split Protocol, Quantile Crossing Risk (+18 more)

### Community 16 - "Dataset Build and Freeze"
Cohesion: 0.13
Nodes (23): build_all(), _jsonable(), main(), Path, Regenerate every disease bundle from the cached raw sources, in one…, Build the five bundles in a fixed order. Also imported by the leakage suite., Coerce a meta value to plain Python so the JSON blob stays inspectable. The…, Write one bundle to data/processed/<name>.npz. (+15 more)

### Community 17 - "Ebola Support Pre-Registration"
Cohesion: 0.12
Nodes (25): A2: Dengue Case Definitions Accepted as Built and Disclosed, A3: Calendar-Prefix Ebola Support with a Fixed Cutoff, A4: Ebola Gap-Lumping Disclosed, C1: Weekly Incidence from the Running Maximum, C2: First Observed Week of Every District Masked, C3: Per-Node Dengue Split Fallback Removed, C8: Dengue Splits Cut Per Country, D1: Zero-Variance Normalisation Guard Fixed (+17 more)

### Community 18 - "Dengue Loader Tests"
Cohesion: 0.11
Nodes (25): _dengue_fixture(), _dengue_rows(), _dengue_rows_lvl(), Weekly rows at a specified S_res level. Admin1 -> unit in adm_1_name; Admin2 ->…, Multi-country weekly extract (+ some legacy monthly rows), mirroring the best-…, Admin-2 leaf names are NOT unique within a country. In the real extract Brazil…, OpenDengue reports one unit under two spellings in DISJOINT eras -- the DR's…, GADM's own defects are absorbed at the JOIN, never at LOAD, so our node ids… (+17 more)

### Community 19 - "Ebola Case Study Runner"
Cohesion: 0.11
Nodes (20): Adapter, FiLM (gamma,beta) + quantile head. 1,428 params at d=64, |H|=4, |Q|=5. With the…, alldev_plan(), _check_e4(), ebola_naive_predictions(), get_trunk(), load_manifest(), main() (+12 more)

### Community 20 - "Data Audit and Corrections"
Cohesion: 0.12
Nodes (23): COVID-19 US-states bundle EDA, The Omicron peak lands inside the COVID validation fold, The Admin-2 key collision defect, Dengue case definitions over time - measured, not asserted, Clarke J et al., Scientific Data 2024;11(1):296 (OpenDengue), The clip defect that fabricated 36 per cent of the Ebola target, Register of corrections to earlier releases of the audit note, COVID-19 bundle - NYT US states (+15 more)

### Community 21 - "Published Baseline Scorecard"
Cohesion: 0.18
Nodes (23): Baseline Reproduction Scorecard (G6), Empty Intersection Research Gap, Cola-GNN, Cola-GNN Repro Ran Off-Grid at h=1, Deng et al., Cola-GNN, CIKM 2020, EpiGNN, Xie et al., EpiGNN, ECML-PKDD 2022, HeatGNN (+15 more)

### Community 22 - "Bundle Loading Interface"
Cohesion: 0.14
Nodes (18): _demo(), _groups(), load(), _locf(), _locf_check(), _normalise_masks(), ndarray, bundles.py -- one interface over the five harmonised disease bundles.… (+10 more)

### Community 23 - "Week 4 Decision Log"
Cohesion: 0.16
Nodes (22): D10: Reproduction Failure Log Scope, D11: Independent Audit of Steps 1 and 2, Both Green, D15: Capacity Probe Runs with an In-Domain Control Arm, D17: The Trunk Early Stop Was Real and Cost Nothing, D18: The Single-Disease Ceiling Early-Stops Too, 25 of 25, D19: Adapter Capacity at Five Seeds, D3: Reporting Standards Effective Immediately, D6: HeatGNN Epoch Budget Stays at the Paper's 1500 (+14 more)

### Community 24 - "Baseline Comparability Rules"
Cohesion: 0.10
Nodes (22): COVID vs influenza on the identical 49 nodes, Deng et al., Cola-GNN, CIKM 2020, EpiGNN, ECML-PKDD 2022, US matrices identified as CDC ILINet at a known vintage, Influenza bundles - ColaGNN benchmark matrices used unchanged, Influenza covariates validated against independently-known geography, Japan calendar pinned to the week against NIID season peaks, Japanese prefecture ordering recovered from three independent lines of evidence (+14 more)

### Community 25 - "Bootstrap Analysis Engine"
Cohesion: 0.20
Nodes (20): bootstrap_ci(), _cnt_matrix(), _encoder_stats(), gate_reads(), _load_perorigin(), _macro_dist(), main(), _pct() (+12 more)

### Community 26 - "COVID and MTGNN Export Probes"
Cohesion: 0.13
Nodes (19): canonical(), floors(), main(), patched_load(), Probe: is COVID's single-disease failure caused by selecting the checkpoint on…, Seed-matched RMSE already on disk -- arm A must reproduce this., Return new [N,T] masks. `test` is returned untouched in every mode -- asserted…, _recarve() (+11 more)

### Community 27 - "Results Doc Verifier"
Cohesion: 0.13
Nodes (20): check_coverage(), check_summary_prose(), check_tables(), check_verdict_totals(), _is_arm(), main(), mutate(), _parse_num() (+12 more)

### Community 28 - "Negative Transfer Findings"
Cohesion: 0.13
Nodes (21): C3 Negative Control Manifests Only Densely, P2 Uniform Cross-Disease Sampling, Doc-to-Disk Verification Protocol, Leave-One-Disease-Out Fold Restructure, Markdown-to-docx House Style Pipeline, Week-3 Transfer Headline Was a Reference Artifact, Results Matrix (768 Cells), Cross-Disease Transfer Is Negative (+13 more)

### Community 29 - "Ebola Schema Tests"
Cohesion: 0.23
Nodes (19): _ebola_csv(), _ebola_fixture(), test_schema.py — correctness + shape tests for to_schema.py. Uses tiny…, Mirrors the REAL OCHA ROWCA sheet, not the loader's original guess: columns…, test_ebola_client_region_control(), test_ebola_drops_national_blobs_and_excel_epoch_dates(), test_ebola_few_shot_support_is_two_weeks_and_leakage_safe(), test_ebola_first_observed_week_is_masked_not_a_backlog() (+11 more)

### Community 30 - "Provenance and Reporting Rules"
Cohesion: 0.11
Nodes (20): Uniform Within-Dengue Sampler (primary), Brazilian Dominance: Uniform Sampling Primary, Country-Macro Scoring Metric, results/*.json Record Schema (model x dataset x horizon x seed x metric), Datasets Committed, Raw Sources Behind DVC, COVID-19 Fetched From a Mutable Path, Therefore Pinned, Ebola Compilation: Its Checksum Is Its Only Version Identifier, Four Reporting Rules Enforced in the Readers (+12 more)

### Community 31 - "Data Quality Metrics"
Cohesion: 0.19
Nodes (19): _binned(), compare(), _interior_gap(), main(), _md(), metrics(), panel(), DataFrame (+11 more)

### Community 32 - "Corrected Ebola Interval"
Cohesion: 0.17
Nodes (19): assert_reproduces_headline(), country_macro(), draw(), load_pernode(), main(), paired_ci(), prereg_verdict(), ebola_ci.py -- the pre-registered Ebola interval, on the statistic we actually… (+11 more)

### Community 33 - "Reporting Standards Matrix"
Cohesion: 0.13
Nodes (20): D14: No Further Trunk Run Without Checkpointing and Quantile Archiving, No Model Weights Were Ever Saved, Seed Ensembling at Inference, Amendment A7: Both Arms Archive Count-Space Quantiles, E6: Reporting Standard for the Ebola Run, Post-Hoc Conformal Calibration Registration, Online Adaptive Conformal Inference, Quantile Archiving on Every Future Run (+12 more)

### Community 34 - "Country-Macro Scoring Decisions"
Cohesion: 0.13
Nodes (20): Brazilian dominance and the sampler decision, Constant nodes excluded from scoring, retained in the graph, Country-macro headline metric (score.py), Four Guinean districts that record no incidence at all, Zero-variance node guard and its mis-fire, Shuffled-adjacency negative control, G2 - Transferable representations across data-rich diseases, G7 - Reproducible submission-ready contribution (+12 more)

### Community 35 - "Results Matrix Assembly"
Cohesion: 0.18
Nodes (19): cell(), delta_vs_scalar(), _demo(), _fmt(), load(), main(), mean_of(), node_mismatch() (+11 more)

### Community 36 - "Naive Floors and Fold Design"
Cohesion: 0.13
Nodes (19): Identical-Graph Gate Control, Three Naive Forecast Floors, COVID-19 us-states Bundle, Three-Disease Leave-One-Disease-Out Fold (ldo3), LOCF Imputation Tested and Rejected, _mean_adapter Averages Parameters Not Functions, obs_mask Channel Leaks Disease Identity, Zero-Fill Covariate Shift Hypothesis (+11 more)

### Community 37 - "Bundles and Conda Environments"
Cohesion: 0.12
Nodes (18): env.txt Build Environment Freeze, torch 2.6.0+cu124 Training Stack, env_train.txt Training Environment Freeze, ebola Conda Environment (geometry stack), Dengue 12-Country Development Bundle (7,165 nodes), gate_mass_conservation (Ebola cumulative-to-weekly defect), gate_phase_purity (dengue split fallback defect), Leakage Suite and Negative Controls (+10 more)

### Community 38 - "Baseline Reproduction Failures"
Cohesion: 0.15
Nodes (18): Dengue Comparison Is Not Like-With-Like, HeatGNN Isolated-Node NaN Self-Loop Fix, MTGNN Degenerate Constant Collapse, Three-Way Baseline Reproduction Table, Encoder Beats Published GNNs on the Common Pipeline, Baseline Comparator Suite (G6), Dengue One-Third Stratified Subsample, export_baseline.py (+10 more)

### Community 39 - "Frozen Encoder Hyperparameters"
Cohesion: 0.14
Nodes (17): Dilated Causal TCN (receptive field 32 >= lookback 20), FiLM + Head Per-Disease Adapter (~388 params), Frozen Week-3 Encoder Hyperparameters, Config Is a Human Record, Code Is the Source of Truth, Quantile Head with Pinball Loss, Disease-Agnosticism Enforced in Code, Not Discipline, Disjoint Geography Identifies the Disease, Influenza Development Bundles (Japan / US-Regions / US-States) (+9 more)

### Community 40 - "Baseline Scoring and Comparison"
Cohesion: 0.19
Nodes (15): collect(), encoder_pooled(), _export(), main(), _ms(), pooled(), paper_compare.py -- the three-way picture (Work Order 1c): published |…, {(dataset, h): [rmse_per_seed]} -- pooled RMSE from the per-origin sufficient… (+7 more)

### Community 41 - "Gate Figure Panels"
Cohesion: 0.20
Nodes (17): covid us-states (N=49, gate 0.375), dengue (N=7,165, gate 0.60), Density / Region-Count Confound (B and C are one ordering shown twice), flu japan (N=47, gate 0.27), flu us-regions (N=10, gate 0.37), flu us-states (N=49, gate 0.373), Gate Reads Graph Structure, Not Disease, Graph Density 2E/N(N-1) (log axis) (+9 more)

### Community 42 - "Learned Spatial Gate Ablation"
Cohesion: 0.16
Nodes (17): Learned Spatial Gate Figure, Five-Seed Protocol (error bars are seed sd), Per-Node Gate Spread (IQR, 1.5x IQR whiskers, pooled over seeds), Dilated Causal TCN with Receptive Field 32, Eight Encoder Invariants and Negative Controls, Gated Inductive ST-Encoder, Learned Spatial Gate (CONFIRM-P8), P9 Annealed Edge Dropout (+9 more)

### Community 43 - "Influenza Loader and Covariates"
Cohesion: 0.14
Nodes (12): Path, Build the three ColaGNN influenza bundles and assert their calendars and…, Fail loud if a cached raw drifted from the checksummed snapshot., verify_checksums(), Tests for the influenza static covariates C = [centroid_lat, centroid_lon,…, test_C_is_not_transfer_safe(), test_us_regions_are_unions_of_their_states(), test_influenza_loader_shapes() (+4 more)

### Community 44 - "Negative Control Test Gates"
Cohesion: 0.16
Nodes (17): Calendar-causality gate for the few-shot split, Determinism verified, not assumed, The leakage and invariant suite (86 -> 101 gates), Six negative controls that plant real defects, Scaler gates built as probes, after a circularity failure, build_datasets.py - verify, construct, gate, write, Checksum-pinned public provenance for every raw input, loaders/covid_load.py - the one fetched source (+9 more)

### Community 45 - "Coverage Filter Probe"
Cohesion: 0.18
Nodes (12): Bundle, Valid origins t (0-indexed into T): t >= w-1 and t+H <= T-1. An example is an…, ebola_eval(), main(), mnar(), coverage_filter_probe.py -- what a "drop the sparse nodes" filter actually…, The window-coverage arithmetic, on a hand-built bundle-shaped mask., Is missingness related to the signal? If yes, filtering on it moves the… (+4 more)

### Community 46 - "Single-Run Epoch Recovery"
Cohesion: 0.17
Nodes (12): already_done(), compare_to_published(), fingerprint(), main(), Epochs-used for the 20 single-disease runs whose training log is gone (client…, Worst relative disagreement between the re-run and the published ceiling, on…, The repro comparison actually detects a difference, and the encoder_mc…, stdout -> console AND the log, so train_one's own epoch prints land in a file… (+4 more)

### Community 47 - "Seed Ensembling Comparison"
Cohesion: 0.25
Nodes (15): _archive(), compare_transfer(), ensemble_preds(), field(), floors(), main(), per_seed(), Seed-ensemble the single-disease ceiling, off the archived quantiles. No GPU,… (+7 more)

### Community 48 - "Ebola Report Generator"
Cohesion: 0.20
Nodes (15): by_horizon(), check_prereg(), load_records(), main(), naive_floors(), ebola_report.py -- read the scored Ebola case study off disk. G3, the headline…, The two rules that can quietly produce a wrong table: the skill sign, and the…, All seed records for one (arm, regime). Returns [] if the family was never… (+7 more)

### Community 49 - "COVID Skill Investigation"
Cohesion: 0.23
Nodes (16): Leave-One-District-Out Adapter Stopping Rule (A2), Pre-Registration Amendment Log, E2: Few-Shot Beats Zero-Shot at h3 and h5, E4: h15 Head Block Must Be a Uniform Shrink of Its Initialisation, Few-Shot FiLM Adapter (1,428 params), Zero Left-Pad for Short Windows (A3), All-Dev-Bundle Joint Trunk, Add COVID-19 Back as a Third Development Disease (+8 more)

### Community 50 - "Adapter Capacity Probe"
Cohesion: 0.16
Nodes (14): banner(), load_trunk(), n_params(), capacity_probe.py -- OVERNIGHT RUN. Is the cross-disease deficit caused by the…, {(dataset, horizon): country_macro rmse} out of a record list., The one frozen dengue trunk this seed's two arms share. TRUNK ONLY. That…, The ONE frozen trunk both sweeps share. Sharing it is what makes the control a…, Fit every surface in the ladder on the SAME frozen trunk over `names`, score,… (+6 more)

### Community 51 - "Ebola Uncertainty Table"
Cohesion: 0.19
Nodes (14): main(), mean_sd(), The pre-registered Ebola UQ block: WIS, CRPS, PIT, coverage and width. (audit…, The failure that matters is silent: reading the wrong fold. Ebola has no test…, report(), _selfcheck(), _eval_mask(), _node_means() (+6 more)

### Community 52 - "Artifact Path Routing"
Cohesion: 0.24
Nodes (12): main(), Gate-off ablation: does the spatial graph actually help, or is it decoration?…, The two things that would silently produce a wrong table: a mislabelled arm,…, report(), _selfcheck(), _demo(), Layout for results/: ONE prefix->subdir map so every reader and writer agrees…, The results/ subdir an artifact belongs in, decided from its filename alone. (+4 more)

### Community 53 - "Transfer View Feature Block"
Cohesion: 0.19
Nodes (14): Node-level-mean variance diagnostic and the dengue discrepancy, Window normalisation dropped - the released scaler is already per-node, bundles.py - one interface over five bundles, The windowing contract (w=20, h in {3,5,10,15}, direct multi-horizon), Day-11 close-out readings that gate Day 12, Scope the spatial claim to timing, not magnitude (§0.9), Core 4-channel transfer-safe feature block, Static covariates C deferred (decision D20) (+6 more)

### Community 54 - "Spatial Mixer Design Constraints"
Cohesion: 0.19
Nodes (14): Disease-agnosticism reduces to one data-layer rule, Four isolated influenza nodes (Alaska, Hawaii, Hokkaido, Okinawa), Mobility adjacency declined for every disease, Rolling-origin backtest scaffold, Influenza adjacency diagonal zeroed to match the built graphs, The three coded constraints carried into the modelling phase, Three obligations the datasets place on the model, Decision thresholds that change the plan (+6 more)

### Community 55 - "Overnight Transfer Orchestrator"
Cohesion: 0.24
Nodes (9): _hr(), main(), phase1_covid_baselines(), phase2_transfer(), preflight(), One command for the overnight transfer run. Preflight, COVID baselines, then…, stdout -> console AND the log file, so one command needs no shell pipe., COVID ceiling (5 seeds) + naive floors. (+1 more)

### Community 56 - "Evaluation Protocol Freeze"
Cohesion: 0.17
Nodes (12): Five-Seed Reporting Protocol {42,52,62,72,82}, Frozen Evaluation Protocol, EpiGNN Protocol Matches Our Pipeline Exactly, HeatGNN RMSE Printed Divided by 1000, Rolling-Origin Scaler Refit (Bundle.refit), bundles.content_sha256() Canonical Array Digest, Determinism Policy (build bit-deterministic, training not), Median-to-Mean Correction (encoder_mc arm, unwired) (+4 more)

### Community 57 - "Encoder Rescoring Backfill"
Cohesion: 0.23
Nodes (11): _aggregate_one(), backfill(), _dataset_of(), main(), _masks_for(), Path, rescore_encoder.py -- backfill nrmse onto encoder runs already on disk, without…, `encoder_joint__sqrt-uniform__dengue__seed42__pernode` -> `dengue` (token… (+3 more)

### Community 58 - "Gate Figure Generator"
Cohesion: 0.23
Nodes (10): collect(), density(), figure(), _label_groups(), Work order 8e -- the learned graph gate as a figure. The client's ask (Review…, Aggregation and geometry, on synthetic input. No artifacts needed., Undirected edge density, self-loops excluded. Returns (n_nodes, n_edges,…, Per-panel gate readout. Keeps the per-SEED means (the error bars) and the… (+2 more)

### Community 59 - "Meta-Learning Algorithm Choice"
Cohesion: 0.24
Nodes (12): D13: ANIL, and the Schedule Answer Is No for the Full Scope, D2: Meta-Learning Is Required, Not Optional, ANIL (MAML Restricted to the Adapter, Exact Second-Order), Dengue-to-Influenza as the Direction to Keep, Episode Sampler and Inner Loop, FiLM Adapter Is Not a Linear Probe (later reversed), ProtoNet Excluded, Reptile Excluded (+4 more)

### Community 60 - "LDO3 Negative Transfer Results"
Cohesion: 0.32
Nodes (12): E1: Zero-Shot Loses to Persistence at Every Horizon, Zero-Shot Mean-Adapter Definition, Where the Bootstrap Disagrees with the Seed-Paired Test, COVID Long Horizons Excluded, Not Counted as Losses, Every Fold Varies Graph, Geography and Node Count, Transfer Degrades Monotonically with Horizon, Three-Disease Leave-One-Disease-Out Results, Cross-Disease Transfer Is Negative (25 of 36 Cells) (+4 more)

### Community 61 - "Ebola Few-Shot Split Protocol"
Cohesion: 0.27
Nodes (12): Calendar-prefix few-shot support protocol, Two frozen Ebola support-length arms (L12 primary, L20 secondary), Scored vs reachable pairs (pre-registration amendment A1), C8 - Ebola never enters model selection, BLOCKER - Ebola has zero support origins under the frozen protocol, Rehearse the Ebola path without scoring it (Day 20), Ebola's query set is scored exactly once, and process is the only protection, configs/frozen_ebola_protocol.json (the Day-19 freeze) (+4 more)

### Community 62 - "Factorised Temporal-Spatial Encoder"
Cohesion: 0.21
Nodes (12): C2 - no shared-trunk parameter may have a dimension equal to N, TemporalEncoder - dilated causal TCN, shared and inductive, Factorised inductive temporal-then-spatial encoder (Option 1), Adaptation surface - FiLM modulation plus head (~388 params), Node-indexed parameters eliminate most of the state of the art, Memory budget and fallback order on the 12 GB card, PEMS (ICLR 2024) and CAPE (2025) - the closest transferable prior art, Receptive-field assertion (RF >= lookback) (+4 more)

### Community 63 - "Quantile Archiving and Bias Correction"
Cohesion: 0.18
Nodes (12): Quantile Archiving (write_quantiles), encoder_mc Log-Space Bias-Correction Arm, Median Objective versus Mean Metric Mismatch, Count-Space Node Leverage Imbalance, Pre-Registration Before Ebola Is Touched, Seed Ensembling of Quantiles, The Ceiling Job Is a Retrain, Not a Rescore, Ebola Pre-Registration Amendment A7 (+4 more)

### Community 64 - "Baseline Format Exporter"
Cohesion: 0.31
Nodes (10): _check(), export(), export_all(), _kept_indices(), ndarray, Path, export_baseline.py -- our five bundles -> the ColaGNN-lineage on-disk format…, Self-check: round-trip each export and prove the invariants a baseline run… (+2 more)

### Community 65 - "Ebola One-Way Door"
Cohesion: 0.22
Nodes (10): Early Stopping on val_pinball, Never on Ebola (C8), DiseaseTensors Schema Contract, Ebola Few-Shot Holdout Bundle (GIN/LBR/SLE, 61 nodes), Ebola Validation Protocol (select on dev, score once), Emerging Disease Forecasting Framework, MepoGNN Dropped: Our Schema Ships A_mob = None, bundles.py Single Interface Over All Bundles, The Ebola One-Way Door (scored exactly once) (+2 more)

### Community 66 - "Client Work Order and Folds"
Cohesion: 0.44
Nodes (10): D12: Cross-Disease Transfer Is Negative, D1: Transfer, With the Folds Fixed First, D5: Transfer-Table Reference and Dispersion, Client Review and Standing Work Order, Hold the Five Seeds Until the Folds Are Fixed, Leave-One-Dataset-Out Fold, Leave-One-Disease-Out Fold, LODO versus LDO Side by Side (+2 more)

### Community 67 - "Project Goals and Pillars"
Cohesion: 0.22
Nodes (10): Ebola signal-confirmation audit (ebola_audit.py), Risk: Ebola data scarcity, Epidemiology-informed inductive bias (metapopulation / physics-informed loss), Four design pillars (none disease-specific), Generalizable Spatio-Temporal Framework for EID Forecasting, G1 - Disease-agnostic spatio-temporal architecture, G3 - Few-shot generalisation to data-scarce Ebola, G5 - Explainability of forecasts (+2 more)

### Community 68 - "Experiment Families and Floors"
Cohesion: 0.20
Nodes (10): Fold 3 COVID Peak Makes Single-Value RMSE Meaningless, GTGCN, LinearTGCN Matches or Beats Both Transformers, Ezzat et al., Graph-Based Transformer, Int. J. Health Geographics 2026, Conformal / ACI Calibration Stage, The Experiment Families, Naive Floors (analytic, seedless), No Record Carries a Wall-Clock Field (+2 more)

### Community 69 - "LDO3 Full-Budget Check"
Cohesion: 0.42
Nodes (8): compare(), _digest(), _key(), main(), Did disabling the trunk early stop change any LDO3 result? (D17, extended to…, sha256 over the sorted (key, every numeric field) — order-independent, value-…, The influenza fold is the control: D17 already settled it as identical. Plus a…, _selfcheck()

### Community 70 - "COVID Data Loader"
Cohesion: 0.25
Nodes (8): fetch(), main(), COVID-19 US-states driver: fetch the source, build through the schema, assert…, Download us-states.csv if absent, then verify it against the pinned hash. The…, covid_weekly_matrix(), load_covid(), NYT daily cumulative -> [T, 49] weekly incidence in ColaGNN's exact column…, Build the COVID-19 US-states bundle from the NYT cumulative series.…

### Community 71 - "Calibrated Uncertainty Route"
Cohesion: 0.28
Nodes (9): bias_c mean-vs-median count-space correction, encoder_mc arm matters most on COVID, Encoder-view Wasserstein distance matrix (z-space), Quantile head with pinball loss, superseding the point head, Conformal prediction as the recommended UQ route, G4 - Calibrated uncertainty on all forecasts, Adaptive conformal inference (ACI) for calibrated intervals, Gibbs & Candes 2021, adaptive conformal inference (+1 more)

### Community 72 - "COVID Split Probe"
Cohesion: 0.36
Nodes (7): disk_current(), floors_for(), main(), new_bundle(), Probe: does the proposed COVID split (train_end=107, val_end=126) make the…, Naive floors scored on this bundle's own test cells, through the same scoring…, The shipped split's model and floor numbers, already on disk.

### Community 73 - "Graph Topology Ablations"
Cohesion: 0.39
Nodes (8): Dengue block-diagonal graph - a known mis-specification, Ebola graph is deliberately not block-diagonal, The gate x augmentation 2x2 (the primary ablation), The learned spatial gate (a contribution, not plumbing), Topology-robust pre-training (edge dropout, rewiring), [CONFIRM-P8] - learned spatial gate, default on, [CONFIRM-P9] - annealed edge dropout with the 2x2 reported, Queen contiguity with k-nearest-centroid island fallback

### Community 74 - "ANIL Adaptation Surface"
Cohesion: 0.36
Nodes (8): 1,428-Parameter Adaptation Surface, ANIL Chosen as the Meta-Learning Algorithm, Adaptation Capacity Probe, Frozen-Trunk Adapter Is an Exact Affine Map, Trunk Checkpointing (write_checkpoint), Meta-Learning Is Required Under G2, train/anil.py ANIL Module, ANIL Timing Probe Abort

### Community 75 - "The Honest Thesis"
Cohesion: 0.33
Nodes (7): Learned Spatial Gate (P8; g==0 is the ablation cell), Project Goals G1-G7, Phase 3 Modelling Contribution (Weeks 3-6), ANIL Meta-Learning Null Result (0 of 12 cells), The Honest Thesis: A Boundary-Conditions Paper, The Graph Does Not Help Accuracy (gate-off ablation), Cross-Disease Transfer Is Negative

### Community 76 - "Per-Node Score Aggregation"
Cohesion: 0.33
Nodes (6): aggregate(), per_node_scores(), Per-node metrics over observed eval cells. Arrays are [N, T] in COUNT space;…, country-macro (equal weight per country) and node-mean (equal weight per node).…, Convenience wrapper: node_country_map is meta['node_country'] (id -> country)., score_bundle()

### Community 77 - "Single-Disease Reference Reporting"
Cohesion: 0.47
Nodes (6): _field(), _load_recs(), (mean over seeds, CV%) of the single-disease run at this exact (dataset,…, Both references, labelled, plus the seed-noise screen. History (2026-07-29):…, _report(), _single_ref()

### Community 78 - "Day 11 Variance Diagnostics"
Cohesion: 0.50
Nodes (4): fit_window_mean_variance(), main(), Day-11 close-out readings (Week-3 guide §0.8, §0.9). Both are readings of Day…, Variance across nodes of each node's mean X[:,:,0] over its FIT (train/support)…

### Community 79 - "GADM Shapefile Fetcher"
Cohesion: 0.50
Nodes (4): main(), Path, Fetch the GADM 4.1 shapefiles the project's graphs and covariates are built…, sha256()

### Community 80 - "Japan Calendar Pin"
Cohesion: 0.50
Nodes (4): load_national(), Pin the Japan influenza calendar to the week, and re-assert it on every build.…, Return a weekly national-total ILI series indexed by week-ending Saturday., verify()

### Community 81 - "Japan Seasonality Ablation"
Cohesion: 0.67
Nodes (3): main(), mean_sd(), Japan-only ablation: zero sin_doy/cos_doy (transfer_view channels 1,2) and…

### Community 84 - "Cross-Border Graph Edges"
Cohesion: 0.67
Nodes (3): C6: Ebola Graph Keeps Cross-Border Connections, C7: Three Sierra Leonean Labels That Are Not Districts, D6: The Dengue Graph Forbids Cross-Border Edges

## Knowledge Gaps
- **58 isolated node(s):** `Japan Influenza Calendar Pin`, `Manuscript Reconciliation Memo`, `bundles.py Single Interface Over All Bundles`, `Shrinkage Test (post-hoc lambda sweep)`, `Median-to-Mean Correction (encoder_mc arm, unwired)` (+53 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 567 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `rpath()` connect `Artifact Path Routing` to `Corrected Ebola Interval`, `Overnight GPU Queue Runners`, `ANIL Meta-Learning Trainer`, `LDO3 Transfer Training Loop`, `Conformal Calibration Engine`, `LDO3 Report Builder`, `Single-Disease Reference Reporting`, `Seed Ensembling Comparison`, `Adapter Capacity Probe`, `Ebola Uncertainty Table`, `Ebola Case Study Runner`, `Overnight Transfer Orchestrator`, `Bootstrap Analysis Engine`, `Gate Figure Generator`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `load()` connect `Bundle Loading Interface` to `Baseline Format Exporter`, `LDO3 Transfer Training Loop`, `ANIL Meta-Learning Trainer`, `Shared Encoder Architecture`, `Joint Multi-Disease Training`, `Conformal Calibration Engine`, `Coverage Filter Probe`, `Day 11 Variance Diagnostics`, `Seed Ensembling Comparison`, `Ebola Uncertainty Table`, `Encoder Rescoring Backfill`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `train_one()` connect `LDO3 Transfer Training Loop` to `Shared Encoder Architecture`, `Joint Multi-Disease Training`, `Single-Run Epoch Recovery`, `Ebola Case Study Runner`, `Bundle Loading Interface`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `SharedEncoder` (e.g. with `LTR` and `SpatialMixer`) actually correct?**
  _`SharedEncoder` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Japan Influenza Calendar Pin`, `Manuscript Reconciliation Memo`, `bundles.py Single Interface Over All Bundles` to the rest of the system?**
  _58 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Overnight GPU Queue Runners` be split into smaller, more focused modules?**
  _Cohesion score 0.05443037974683544 - nodes in this community are weakly interconnected._
- **Should `LDO3 Transfer Training Loop` be split into smaller, more focused modules?**
  _Cohesion score 0.07372229760289462 - nodes in this community are weakly interconnected._
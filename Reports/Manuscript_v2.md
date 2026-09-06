# A Disease-Agnostic Shared Encoder for Emerging Infectious Disease Forecasting: Construction, Calibration, and the Limits of Cross-Disease Transfer

*One representation for many pathogens: what it buys, and where it stops working*

**Abstract**

Forecasting a new infectious disease is hardest at the point where it matters most, because early in an outbreak almost none of that pathogen's own history exists yet. The usual answer is to carry structure over from diseases that have plenty of data, and spatio-temporal graph neural networks provide the tools. Almost all of those models are still built one disease at a time, with spatial parameters tied to a single graph through the node set they were trained on, and cross-disease transfer left as future work.

This paper starts from the construction instead. We build a shared spatio-temporal encoder that is disease-agnostic *by construction* rather than by evaluation. It reads a four-channel representation common to every disease, holds no parameter whose size depends on the graph, and adapts to a new pathogen through a correction of 1,428 parameters, about 1% of the trunk. These properties are checked by machine rather than asserted: one weight set runs unchanged on graphs of 10, 47, 49, 61 and 7,165 nodes, a fifth input channel raises an error, and 101 leakage gates with planted negative controls must pass before the data is released.

We then use that construction as an instrument, because it holds the disease constant. Cross-disease transfer sits close to parity at three weeks and gets steadily worse, until at fifteen weeks a frozen foreign trunk loses to in-domain training in every attributable cell. That verdict survives a change of estimator, a change of significance test and a much stronger reference. Calibration transfers where accuracy does not: a conformal correction fitted entirely on other diseases lifts coverage on the unseen pathogen from a range of 0.28 to 0.70 up to near nominal, reading no target-disease outcome at all. A gate-off ablation measures the spatial channel itself. Message passing improves the *shape* of a forecast, its correlation with the observed curve, in 6 of 20 cells, and never improves error magnitude, which it degrades in 8 of 40. We keep the channel because shape is what an early-warning signal is read for, and because an inductive spatial channel is what makes the transfer question askable at all. A pre-registered Ebola case study, scored once against a protocol hashed beforehand, did not meet its stated success criterion, and we report it as such.

The contribution is the approach and the evidence about where it works and where it stops, not a claim that transfer solves the emerging-disease problem.

**Keywords:** epidemic forecasting; spatio-temporal graph neural networks; disease-agnostic representation; cross-disease transfer; conformal prediction; pre-registration.

---

## 1 Introduction

Every epidemic leaves a trail of surveillance data, but a new pathogen leaves almost none at the point a forecast is first needed. The early weeks of an outbreak are when public-health decisions matter most, and they are also when the record a data-driven model would rely on is thinnest. In the 2014 West African Ebola epidemic, the window in which a forecast could still have changed where treatment capacity was sent was measured in weeks. The district-level case series available at that moment ran to a few dozen observations in total.

That gap motivates a different starting assumption. Instead of learning from one disease's own long history, learn structure from diseases that have plenty of data, such as dengue and influenza with their years of spatially resolved records, and carry it over to a disease that has very little. The idea is appealing and it is not new. What has been missing is a build in which it can actually be *tested*, because the models that dominate the field are not designed to hold anything constant across diseases.

Graph neural networks became the standard answer because they use the two structures every notifiable-disease surveillance stream carries at once: temporal autocorrelation within each reporting unit, and spatial coupling between units that share population flows, borders or environmental drivers. Judged against what an operational early-warning system needs, the design space breaks into a handful of sub-problems: temporal dynamics, spatial coupling, mechanistic structure, uncertainty, explanation, and transfer to a disease or region where labelled history is scarce.

The literature has made strong progress on each of these on its own, and much less on a quieter structural problem. Most of these models *cannot* be moved between diseases without being rebuilt, and the barrier is one of dimensions rather than concepts. Models that learn their adjacency from node embeddings hold a dictionary **E** ∈ ℝ^(N×d), and models that learn a spatial attention matrix hold **W** ∈ ℝ^(N×N). Either way the parameters are tied to one graph. A model trained on 47 Japanese prefectures cannot be evaluated on 49 US states, let alone on 7,165 dengue districts, without re-initialising the very parts that hold the spatial knowledge. Cross-disease evaluation then means training a fresh model per disease and comparing scores, which says something about the method but nothing about transfer.

**This paper starts from the construction rather than from the result.** We build a spatio-temporal encoder in which being disease-agnostic is a structural property that can be checked by machine. Three commitments do the work. The shared trunk sees only a four-channel *transfer view*, identical in name, order and index across every dataset, and supplying a fifth channel is a runtime error rather than a review comment. No shared parameter has any dimension equal to any node count in the study, so one weight set runs on every graph unchanged. And static geographic covariates are kept out of the trunk entirely, because the diseases sit in largely separate parts of the world and a raw centroid is a disease label in disguise.

Having built it, we use it as an instrument. A model that holds the architecture, the representation, the optimiser and the evaluation constant across diseases can ask a question the field has mostly had to assume an answer to: *when does a shared epidemic representation transfer, and where does it fail?* We find a specific, repeatable answer with a clear pattern across horizons, and we report it whether or not it flatters the method. It does not, entirely. That is the point of building an instrument rather than a demonstration.

Section 2 states the contributions. Section 3 reviews the literature in two blocks and Section 4 states the gap that evidence supports. Sections 5 to 8 give the formalisation, the representation, the architecture and the protocol. Section 9 reports results and Section 10 the threats to validity.

---

## 2 Contributions

**C1. A disease-agnostic shared encoder, enforced in code rather than asserted.** The architecture treats a new pathogen as a small, bounded correction to a shared representation of epidemic dynamics. The shared trunk holds 142,305 parameters and **no parameter with any dimension in {10, 47, 49, 61, 7165}**, the five node counts in this study. The per-disease adaptation surface is a FiLM modulation plus a quantile head, totalling **1,428 parameters**, about 1% of the trunk. The same weight set ran unchanged at N = 47, 49 and 7,165 with no reshape. Each of these properties is a test with a planted negative control, not a design note. A gate-off ablation then measures the spatial channel itself: it improves the correlation of a forecast in 6 of 20 cells and error magnitude in none of 40. We keep it for shape and for the inductive property, and claim no accuracy gain from it.

**C2. A five-seed measurement of how large that correction has to be.** Sweeping four adaptation surfaces on a frozen trunk, seed-paired so only the surface varies within a seed, extra adaptation capacity helps across diseases in **7 of 36 comparisons** and within a disease in **0 of 12**. All seven gains fall at the fifteen-week horizon. The correction is real, small, and needed specifically when crossing between diseases rather than in general.

**C3. The conditions under which cross-disease transfer holds and fails.** Under a three-way leave-one-disease-out design, a frozen foreign trunk matches in-domain training at three weeks and loses everywhere by fifteen. Across the 36 attributable cells the verdict is **1 better, 10 within noise, 25 worse**, with a steady gradient across horizons, and it does not change with the estimator, the significance test, or the strength of the reference.

**C4. Calibrated uncertainty transferred across diseases.** The quantile head is badly calibrated out of the box: 0.492 to 0.918 against a 0.90 nominal target on the development panels, and **0.275 to 0.701 on the held-out pathogen**. A per-horizon conformal correction fitted entirely on other diseases, then frozen and applied unseen, brings coverage to near nominal **while reading no target-disease outcome at all**. The transferred correction does the work, and online adaptation adds about 0.03 at short horizons and nothing at long ones. We state the guarantee honestly: what transfers is the correction, and it carries no finite-sample guarantee on the held-out disease.

**C5. An evaluation protocol, and a demonstration that experimental design decides the conclusion.** We show directly, on the same data and the same model, that a leave-one-*dataset*-out design and a leave-one-*disease*-out design support different conclusions, and that a reference computed at one seed and one computed at five support different conclusions again. We offer this as a build, a more careful framework for a field whose designs vary widely, not as a criticism of any prior study.

Three further results in Section 9.6 support these without being raised to headline claims: joint multi-disease training does not beat per-disease training, a borrowed zero-shot adapter is behind in-domain training in every attributable cell, and every model in the group, ours included, loses to simple naive floors more often than it wins.

---

## 3 Related Work

The literature relevant to this paper splits into two blocks that rarely appear in the same survey. The first is the graph-based epidemic forecasting family, from which we take our architecture vocabulary and our baselines. The second is the non-graph, cross-disease pre-trained forecasters. These are the closest prior work to our transfer objective and are routinely left out of graph-focused reviews, including, in an earlier draft, ours. Reading only the first block produces a research gap wider than the evidence supports.

### 3.1 The traffic and sensor lineage

The architectural vocabulary of epidemic GNNs comes mostly from traffic and multivariate-sensor forecasting. That lineage is worth treating as a control group, since it shares the machinery but none of the epidemiological intent. STGCN (Yu, Yin and Zhu, 2018) and DCRNN (Li, Yu, Shahabi and Liu, 2018) set the convolutional and diffusion-recurrent templates for graph time series. Graph WaveNet (Wu et al., 2019) added the idea that matters most here, learning a self-adaptive dependency matrix from node embeddings so the graph becomes a parameter rather than an input. MTGNN (Wu et al., 2020) generalised that into an explicit graph-learning layer with dilated inception convolutions.

These models are strong because they impose no domain structure, and that is also their limit here. They carry no notion of transmission, no calibrated uncertainty, and were never asked to move between diseases. They set the floor an epidemic-specific contribution has to clear, and we reproduce MTGNN as a non-epidemic control, a role it fills more usefully than we expected (Section 9.6).

This is the family the dimensional barrier of Section 1 applies to most directly. Graph WaveNet's adaptive adjacency comes from a learned node-embedding dictionary **E** ∈ ℝ^(N×d), and MTGNN's graph-learning layer is the same idea with a different parameterisation. Both are *transductive*, so neither can be reused for our question.

### 3.2 Attention and transmission-risk models

The first sub-problem the epidemic community picked out was that geographic adjacency is a poor stand-in for epidemiological coupling. Two regions can be far apart yet move together epidemically through travel or shared seasonality. Cola-GNN (Deng et al., 2020) learns a directed, asymmetric location-aware attention matrix from recurrent hidden states and combines it with a normalised geographic adjacency through a learned gate, while a multi-scale dilated convolution reads short- and long-range temporal patterns. Its interpretability case study is an early example of the field taking explanation seriously. It is still single-disease by construction, trained separately for each lead time, and gives point forecasts with no uncertainty statement.

EpiGNN (Xie et al., 2022) encodes spatial transmission risk directly. A region-aware graph learner builds a directed correlation graph from temporal features and combines it with a degree-gated geographic adjacency, while local and global transmission-risk encodings summarise each region's exposure. It is one of the few incumbents tested on more than one disease, covering three influenza benchmarks and two COVID-19 datasets. Each dataset is fitted with its own model, though, so that is a claim about breadth of testing rather than about transfer, and it carries neither a mechanistic prior nor a calibrated interval. We use it as a competitor and, because it reproduces faithfully, as a load-bearing baseline. Its spatial component is size-locked through a **W**^s ∈ ℝ^(N×N) term, so it inherits the same transductive barrier.

### 3.3 Mechanistic hybrids

A parallel family accepts that purely data-driven graphs throw away hard-won epidemiological knowledge, and puts a compartmental prior back in. MepoGNN (Cao et al., 2022) embeds a metapopulation SIR process inside a spatio-temporal GNN, learning an adaptive graph from commuter surveys and a dynamic graph from origin-destination flows. CausalGNN (Wang et al., 2022) pairs an attention-based dynamic graph with a single-patch SIRD module, letting the mechanistic parameters and the graph representation refine each other. The trend has picked up since: BDGSTN (Mao et al., 2023) drives a decomposition-linear predictor with an SIR module, HeatGNN (Zheng et al., 2024) fits a time-varying SIR per location under a physics-informed loss, STOEP (Ruan et al., 2026) combines a metapopulation SIR with expert-guided thresholding of the transmission rate, and STAN (Gao et al., 2021) pairs graph attention with transmission dynamics learned from claims data.

This family is the most epidemiologically principled, and several members offer parameter-level interpretability. But with one partial exception discussed below, every one is trained and evaluated on a single disease, reports point forecasts without calibrated coverage, and lists cross-disease transfer as future work. HeatGNN is worth noting for a further reason. Its mechanistic graph construction assumes every node has neighbours, and it returns NaN outputs on graphs that contain isolated nodes. We hit that failure directly, and it motivated one of our own architectural constraints (Section 7.3).

### 3.4 Non-graph cross-disease pre-trained forecasters

This is the block a graph-focused survey leaves out, and leaving it out overstates the gap by a wide margin. Several recent systems reach cross-*disease* generality by dropping the spatial graph rather than by solving it.

CAPE and the PEMS family of pre-trained epidemic forecasters show transfer across pathogens directly: PEMS transfers to an unseen COVID-19 target, and CAPE reports zero-shot behaviour across seventeen diseases. epiFFORMA (Murph et al., 2026) uses the phrase *disease-agnostic* for an emerging-pathogen forecaster built on a light feature-and-ensemble design rather than a graph. Roster et al. (2022) is the closest published precedent for conditional cross-pathogen transfer, moving between dengue and Zika and between influenza and COVID-19. Panagopoulos, Nikolentzos and Vazirgiannis (2021) use model-agnostic meta-learning to carry a mobility GNN from data-rich to data-poor *countries*, showing cross-location transfer for a single disease.

The effect on our gap statement is direct: **it is not true that transfer has been shown across locations but not across diseases.** It has been shown across diseases, by models that drop spatial structure. What has not been shown is cross-disease transfer by a model that *keeps* an inductive spatial channel and also delivers calibrated uncertainty.

### 3.5 The closest incumbent

The work closest to what we are attempting is the dynamic spatio-temporal graph-attention model of Liu and Cao (2026). It forecasts influenza-like illness, hand-foot-and-mouth disease, dengue and respiratory syncytial virus together from a shared encoder with disease-specific probabilistic decoders, and reports properly calibrated quantile forecasts alongside attention- and SHAP-based explanations. On calibration and explanation it is the strongest incumbent.

Its multi-disease capability is still joint multi-task learning over a fixed panel of four diseases, not disease-agnosticism by construction. Its only cross-disease evidence is a leave-one-disease-out learning curve in a single figure rather than a few-shot protocol with held-out metrics, and no public code comes with it. It sharpens the gap rather than closing it, showing that calibration and explanation are achievable in a multi-disease setting while leaving disease-agnostic construction, and adaptation to an unseen data-scarce pathogen, unaddressed.

---

## 4 Research Gap

Reading the two blocks together supports a narrower and more defensible claim than a graph-only reading would.

Among **graph-based** epidemic models, the spatial and temporal modules have settled into a mature set of options, and mechanistic priors have gone from rare to common across the 2022 to 2026 cohort. Set against that, three properties are still thin. Calibrated uncertainty is close to absent, since the mechanistic hybrids report point forecasts and any confidence figures are usually run-to-run seed variability rather than a statement about predictive coverage. Explanation is common but shallow, mostly attention pictures and compartmental parameter read-out rather than input attribution. And cross-disease transfer is essentially absent: models tested on two diseases still fit each one separately, and the dominant parameterisations are transductive, so transfer is not just untried but structurally blocked.

Among **non-graph** cross-disease forecasters, transfer across pathogens is shown, but by dropping the spatial structure the graph literature exists to use, and usually without calibrated coverage.

**The gap is the overlap of the two blocks: no existing system combines an inductive spatial channel that survives a change of graph, transfer across diseases, and calibrated uncertainty.** Each property is handled well somewhere. The combination is handled nowhere. That is a narrower claim than "the intersection is empty", and it is the one the evidence supports.

We are clear about what this framing does *not* claim. It does not claim that a spatially structured model has to beat a non-spatial one on a new pathogen; one of our own results speaks to that and does not settle it in our favour. It claims the combination has not been built and evaluated, and that building it is what makes the question answerable.

---

## 5 Problem Formulation

We treat emerging-disease forecasting as multi-horizon, joint multi-node incidence prediction over the representation of Section 6. Because every disease is mapped onto that representation, the task statement is disease-agnostic by construction and applies unchanged to the development diseases and to the held-out case study.

Consider a disease observed over a graph of *N* spatial units at *T* weekly steps, held as a bundle of aligned arrays: a feature tensor **X** ∈ ℝ^(N×T×F) whose channel 0 is normalised incidence, an adjacency **A** (geographic **A**_geo, optionally mobility **A**_mob), static node covariates **C** ∈ ℝ^(N×S), an observation mask **M** ∈ {0,1}^(N×T) recording where a value was genuinely reported, and the target series **y** ∈ ℝ^(N×T).

One rule governs the bundle: row *i* means the same spatial unit and column *t* the same instant in every array, and nothing downstream works that out again. Node counts and time axes are per disease, and diseases share neither a node set nor a calendar, because transfer runs through shared parameters rather than shared indices.

| Symbol | Domain | Meaning |
|---|---|---|
| **X** | ℝ^(N×T×F) | Feature tensor; channel 0 is normalised incidence |
| **A**_geo | ℝ^(N×N) | Geographic adjacency (queen contiguity; block-diagonal across countries for dengue) |
| **A**_mob | ℝ^(N×N) | Optional mobility adjacency, where available |
| **C** | ℝ^(N×S) | Static node covariates (centroid latitude, longitude, area) |
| **M** | {0,1}^(N×T) | Observation mask: 1 where reported, 0 where imputed |
| **y** | ℝ^(N×T) | Target incidence series, invertible to counts |

**Table 1.** *Components of the per-disease input bundle. Shapes are per disease; N, T and F vary between diseases.*

### 5.1 Prediction task

Given a lookback window of *w* past weeks, the model predicts incidence *h* steps ahead at every node simultaneously,

**ŷ**_(i,t+h) = f_θ ( **X**_(i, t−w+1 : t), **A** ) for all i = 1 … N,  (1)

with forecast horizons *h* ∈ {3, 5, 10, 15} weeks reported separately, following the published EpiGNN and Cola-GNN convention so our results sit directly beside the baselines' tables, and a lookback of *w* = 20 weeks.

Forecasts are produced on raw case counts, not per-100,000 rates, because the assembled sources carry no consistent population denominators. Internal normalisation uses a train-split-only scaler and is inverted back to counts before evaluation, so all reported metrics are in real case units, and predictions are compared with observations only where the mask is set. Note what equation (1) does *not* take: the static covariates **C** are absent from the shared trunk's signature, enforced in code rather than left to convention (Section 7.2).

### 5.2 The lookback window

The choice *w* = 20 is a constraint set by the case study, not a tuned hyperparameter, and the arithmetic limits what the model can be expected to do. The Ebola panel spans 38 usable weeks, so the number of scoreable forecast origins is 38 − *w*: eighteen origins at *w* = 20, and none at all from *w* = 38. A window long enough to see the previous year's seasonal peak (*w* ≥ 53) therefore rules out evaluating the case study at all, and it is separately blocked by the temporal encoder's receptive field of 32.

This has a direct and unflattering consequence, which we report rather than bury. On strongly annual influenza panels the model cannot see the previous season's amplitude, and a seasonal-naive baseline that simply repeats last year beats it clearly (Section 9.6). A sweep of *w* ∈ {12, 20, 32} at five seeds on every development panel confirms this is a property of the window and not of the architecture: almost every cell is within seed noise, the one significant gain is *w* = 12 on a single Japan cell, and *w* = 32 is significantly *worse* on several COVID cells. A longer window, within the range we can use, does not recover the seasonal signal.

### 5.3 Transfer settings

Two evaluation settings test the generalisation the representation is meant to support.

In the **leave-one-disease-out** setting the trunk is trained on all but one development disease and evaluated on the one held out. A *disease* is the unit, not a dataset. Influenza owns three panels and they share one adaptation surface wherever they appear, because three separate surfaces would let the trunk push panel differences into the heads and never learn a flu-invariant representation. We report two arms: an adapted arm that fits a fresh adaptation surface on the held-out disease's own training fold with the trunk frozen, and a zero-shot arm that uses the mean of the in-disease surfaces with nothing fitted at all.

In the **few-shot** setting the model, having been trained on the development diseases, is adapted to a genuinely unseen pathogen using only a pre-registered calendar-prefix support set, then evaluated on the rest.

The adapted leave-one-disease-out arm is deliberately the *optimistic* bound. The held-out disease gets its entire training fold to fit the adaptation surface, far more data than a new pathogen ever supplies. If transfer fails there, it cannot succeed few-shot.

---

## 6 Standardised Input Representation

The framework treats spatio-temporal forecasting as a problem defined over a common interface rather than over one particular disease, and puts the whole harmonisation job in the data layer. Every disease is mapped onto the bundle of Section 5 before it reaches the network, so the architecture holds nothing specific to any one disease.

### 6.1 Feature construction and the transfer view

Feature channels are split into a **core** block common to every disease and an **extended** block that is disease-specific. Only the core reaches the shared encoder. The core is exactly four channels: normalised incidence, two seasonality channels derived from the calendar, and a binary observation indicator. Every disease genuinely has all four. The extended block holds quantities that exist for some diseases and not others, such as reported deaths, and only a single-disease model can use it. **The split is a correctness property, not tidiness.** A channel present for one disease and absent for another lets the shared encoder work out *which disease it is looking at* from presence and absence alone, putting back the disease-specificity the framework exists to remove.

Case counts are strongly right-skewed and span several orders of magnitude, so for unit *i* at time *t* with count *c*_(i,t) we apply a log stabilisation then standardisation,

x_(i,t,0) = ( ln(1 + c_(i,t)) − μ_i ) / σ_i  (2)

with location and scale estimated from the training portion only and then stored, so predictions map back to real counts. Nodes with no training variation take unit scale. Seasonality is encoded as

[ sin(2π τ_t / 365.25), cos(2π τ_t / 365.25) ]  (3)

with τ_t the day of year of step *t*. Taking the phase from the calendar date rather than a running index makes the encoding independent of reporting cadence, which puts diseases with different reporting rhythms on a common seasonal circle. Surveillance coverage is uneven, so the observation indicator *M*_(i,t) is a core channel. Imputed steps take the per-node mean in normalised space rather than the transform of a zero count, which would push a false low signal into every gap, and they never enter the loss or the evaluation.

### 6.2 Deriving weekly incidence from cumulative reports

Ebola situation reports record *cumulative* cases, so weekly incidence has to be recovered by differencing. The obvious approach, taking the first difference and clipping negative increments to zero, is wrong, and we report the correction because an earlier version of this work used the wrong form.

Cumulative series are revised downward during reconciliation, and they carry single-week dropouts where a district reports below its own previous total and then recovers. Clipping the negative increment and differencing normally at the next step *counts the recovery twice*: the fall is suppressed and the rise that follows is credited in full. On the Ebola compilation this invented **8,786 cases, 35.8% of the target**, and moved the national peak by about three months.

The correct transform tracks a running maximum,

y_(i,t) = max(C_(i,1..t)) − max(C_(i,1..t−1)),  (4)

masking any week whose reported cumulative value falls below that maximum. It conserves mass by construction. The released incidence sums to *C*_last − *C*_first exactly, checked per district against a reference computed by a separate routine from the source column. The corrected total is 24,552 cases, with the national peak on 2014-10-25 at 2,688 cases per week, against 2015-02-14 and 4,982 under the clipped transform.

The cost is that masked weeks are dropped. We measure that cost rather than calling it small: 368 downward steps removed, only 31% of which return to the previous high-water mark within three weeks, and two districts the source itself calls central to the outbreak keep only 7 and 17 of 30 scored weeks. Section 10 returns to this.

### 6.3 Spatial structure

Spatial relationships are held as a geographic adjacency matrix. For dengue and Ebola we build contiguity from GADM 4.1 boundary polygons, where two units are adjacent when their polygons share a boundary. We store the raw binary adjacency and leave self-loops and degree normalisation to the model. For dengue, whose development set covers several countries, the adjacency is block-diagonal, with country-qualified node identifiers so like-named units in different countries do not collide. Patterns shared across countries are picked up through shared encoder weights rather than false edges.

For influenza we reuse the adjacency matrices shipped with the benchmark datasets, so comparisons against the matching baselines run on an identical graph. Those matrices carry self-loops and ours do not, so we zero their diagonal. Two conventions in one study would give influenza twice the self-weight of dengue under the standard renormalisation, and self-weight is exactly the kind of systematic difference a shared encoder could use to tell the disease apart.

Surveillance records and boundary files are joined on administrative names rather than codes, so leftover mismatches are resolved by committed alias maps, each entry matched exactly against a real shapefile key. Any unit that cannot be matched stops the build instead of being dropped quietly.

### 6.4 Datasets

All diseases sit on a common weekly cadence, the MMWR epidemiological week. Interpolating monthly dengue history up to weekly resolution would invent observations and inflate the apparent sample size, while coarsening influenza down to monthly would throw away the resolution its baselines were built for.

| Disease | Source | Nodes | Weeks | Observed cells | Role |
|---|---|---|---|---|---|
| Dengue | OpenDengue V1.3 (Clarke et al., 2024) | 7,165 | 1,409 | 2,196,261 | Development |
| Influenza (Japan) | Cola-GNN benchmark | 47 | 348 | 16,356 | Development |
| Influenza (US regions) | Cola-GNN benchmark | 10 | 785 | 7,850 | Development |
| Influenza (US states) | Cola-GNN benchmark | 49 | 360 | 17,640 | Development |
| COVID-19 (US states) | New York Times | 49 | 164 | 8,036 | Development |
| Ebola (GIN/LBR/SLE) | HDX situation reports | 61 | 52 | 1,299 | Held-out case study |

**Table 2.** *Datasets mapped onto the standardised representation. Five development panels across three diseases; Ebola is the held-out case study and enters no training or selection decision.*

**COVID-19 is a fifth training panel and a third training disease, and we say so plainly**, because it trains the shared trunk behind every result in this paper, including the Ebola case study. Any statement that the model is "trained on dengue and influenza" is wrong. It reuses the influenza US-states 49-node ordering and shipped adjacency **exactly as they are**, so `covid_us-states` and `influenza_us-states` sit on a bit-identical graph with identical covariates, checked at build time. That is why we include it. Holding one of them out varies the *disease* and nothing else, while every other cross-disease comparison here mixes disease with graph, geography and node count.

Three costs come with that choice. The 49-state set leaves out Florida, about 6.5% of the US population, which ILINet does not report, and the District of Columbia. The adjacency is land contiguity built for influenza. And the 2020 to 2023 period is dominated by non-pharmaceutical interventions, so transfer to or from COVID may reflect policy response rather than how the pathogen behaves.

One further property affects every COVID result here. The Omicron peak, at 5,139,436 national weekly cases, is the largest value in the series and falls inside the **validation** fold. Every arm therefore selects on a surge and is scored on a flat tail that peaks 6.3 times lower. A model fitted to surge dynamics over-predicts a flat tail, and that error grows with lead time, so **COVID's long-horizon cells measure a fold boundary rather than transfer**. We leave them out of the verdict counts while still reporting them in full. The same break hits the single-disease COVID reference, which is why we make no positive COVID claim either.

### 6.5 Normalisation and leakage control

All series are split chronologically and never shuffled. Development diseases use a 50/20/30 train/validation/test partition following the Cola-GNN and EpiGNN convention, with every normalisation statistic estimated from the training portion alone.

Dengue's split is computed **per country**, not per node and not globally. A global boundary would starve countries whose reporting starts late. A per-node fallback, used in an earlier version of this pipeline, put 31% of observed cells in columns that were training for one node and test for its neighbour, which is a leak a graph network is unusually well placed to exploit, because message passing carries the neighbour's value straight into the node's representation.

Ebola is not split by the development ratios. A pre-registered calendar prefix forms the support set and all later observed weeks form the query set. Its normalisation uses support observations only, **pooled across districts**, because a handful of observations per district cannot support a stable per-node scale, and using later weeks would leak the size of the outbreak into what is supposed to be a few-shot result.

Three properties hold throughout. Normalisation statistics and any learned graph structure depend only on data the model is allowed to see at training time, no feature is derived from future values, and horizon-*h* targets never appear among the inputs.

**These are enforced by 101 pass/fail gates that block the data build.** Nothing is written if any gate fails, because once a dataset with a leak is on disk, somebody will train on it. The gates are *probes* rather than comparisons. A cell the scaler must not see is inflated a thousandfold, the scaler is refit, and the gate requires that nothing moves. Six **negative controls** plant real defects and require the matching gate to fail. This is not a style preference. An earlier version recovered raw counts by inverting the normalised series with the very scaler under test, which is circular, and it passed a deliberately leaked Ebola scaler with every gate green. Each dataset now ships its raw, unscaled incidence, and every scaler gate is checked against that.

---

## 7 Architecture

The architecture exists to make one claim testable: **that a new pathogen can be written as a small, bounded correction to a shared representation of epidemic dynamics.** Every design decision below serves that claim, and each one is enforced in code so the claim can fail honestly if it is false.

### 7.1 The shared trunk

The trunk maps the four-channel transfer view and an adjacency to a per-node latent representation, **h** ∈ ℝ^(N×d) with *d* = 64. It has three components.

A **dilated causal temporal convolution** with kernel size 2 and dilations {1, 2, 4, 8, 16} reads each node's history. The receptive field is 32 steps, and the code asserts at construction that it is at least the lookback *w* = 20. Nodes are the convolution's batch dimension, so the temporal component does not depend on node count.

A **local transmission-risk feature** reuses the degree of the normalised adjacency as one scalar input per node. This follows EpiGNN's transmission-risk encodings, cut back to the one quantity that carries no node identity.

A **gated inductive spatial mixer** does message passing in the GraphSAGE style, aggregating over neighbours, transforming, then combining, and blends the result with the purely temporal representation through a learned per-node gate:

**h**_out = (1 − g) ⊙ **h** + g ⊙ **h**_spatial,  g = σ(MLP(**h**))  (5)

The gate depends on the data and not on node count. It is computed from each node's own representation, not from a table indexed by node. Setting *g* ≡ 0 gives back the graph-free model exactly, which is what makes the spatial channel something we can switch off and test rather than an assumption (Section 9.1).

### 7.2 What the trunk is forbidden

Three constraints are enforced with tests that plant the corresponding defect and require the test to fail.

**No static covariates.** The trunk's forward signature does not accept **C**, and passing it raises a `TypeError`. Dengue, influenza and Ebola sit in largely separate parts of the world, so a raw centroid names the disease outright. We have populated the covariates for every disease, but that does not make them safe to transfer, and a single-disease model may use them freely. The COVID and influenza US-states pair is the one case where the argument does not apply, since they share coordinates exactly, and we exclude the covariates there anyway. It only takes one identifiable disease to break the property.

**No node-indexed parameters.** No shared parameter has any dimension in {10, 47, 49, 61, 7165}. The test walks every parameter and buffer and fails if any one matches, with a planted `nn.Parameter(47)` as the negative control. This is the constraint that separates our build from the transductive family of Section 3.1, and it is why one weight set ran unchanged at N = 47, 49 and 7,165.

**Exactly four input channels.** The trunk asserts its input's last dimension is 4, so Ebola's deaths channel is structurally unreachable by the shared model.

### 7.3 Identity before normalisation

The adjacency is symmetrically normalised as **Â** = **D̃**^(−1/2)(**A** + **I**)**D̃**^(−1/2), with the identity added *before* the degree is computed. Four influenza nodes, Alaska, Hawaii, Hokkaidō and Okinawa, have degree zero, and normalising before adding self-loops divides by zero on those rows. We hit this failure in a comparator model, which returned NaN outputs on exactly these nodes, so the ordering is a checked property of our implementation rather than a convention we inherited.

### 7.4 The per-disease correction

Adaptation is a FiLM modulation of the trunk's output followed by a quantile head:

**z** = γ ⊙ **h**_out + β,  **q** = W**z** + b  (6)

with γ, β ∈ ℝ^d learned per disease, and the head giving five quantile levels {0.05, 0.25, 0.5, 0.75, 0.95} at each of the four horizons. The whole surface is **1,428 parameters**, about 1% of the 142,305-parameter trunk. One forward pass gives all four horizons, so a run is one model per (dataset, seed) rather than one per horizon, and there is no autoregressive rollout to compound error.

Adapting to a new disease means fitting this surface with the trunk frozen. That is what "a small bounded correction" means in practice, and Section 9.2 measures how large it has to be.

We are precise about what this mechanism is. With the trunk frozen, the adaptation surface is an exact affine read-out of fixed features, checked directly with a maximum deviation from an affine map of 0.0. So we call it **adapter fitting on a frozen shared trunk**, not meta-learning. An episodic meta-learning arm is registered as an ablation of the adaptation *procedure* and reported in Section 9.7.

### 7.5 Loss and optimisation

Training minimises the pinball loss over the five quantile levels in model space, masked to observed cells, and all reported metrics are in count space after inverting the per-node scaler. Optimisation uses AdamW at learning rate 10⁻³, weight decay 10⁻⁴, gradient clipping at 1.0, gradient accumulation over 8 origins per step, a cosine schedule, a maximum of 80 epochs, and early stopping on validation pinball loss with patience 15.

Two effects of the quantile head touch every number we report. First, the point forecast is the **median**. That is the right choice for MAE, but it sits below the mean for a right-skewed count distribution, so RMSE penalises it slightly. We report a mean-corrected version alongside the median arm under its own name, never in place of it. Second, the head can produce crossing quantiles, so every interval metric sorts the levels before use.

---

## 8 Evaluation Protocol

The protocol was fixed in advance and applies in the same way to every baseline and to our own model, so accuracy differences reflect the model and not the pipeline.

**Splits.** Development diseases use the chronological 50/20/30 partition of Section 6.5. The Ebola case study uses its pre-registered support and query design. Model selection happens on the validation fold only, and the test fold is scored once.

**Horizons.** Accuracy is reported separately at *h* ∈ {3, 5, 10, 15} and never averaged across horizons. Error grows with lead time, and an averaged figure would hide the decline that is our central finding.

**Point metrics.** RMSE, MAE and Pearson correlation, the metric set used by the reference epidemic-GNN papers, plus sMAPE, peak-intensity error and peak-timing error. We keep MAPE and *R*² out of the headline tables: MAPE is undefined and blows up on weeks with near-zero counts, and *R*² adds little beyond RMSE and correlation.

**Aggregation.** Per-disease tables score each node over its observed evaluation cells, average within each country, then average countries with equal weight. This *country-macro* exists so Brazil's 77% share of dengue nodes cannot become the score, and the equal-node average is reported next to it. For the four single-country panels the two are identical by construction, and Section 10 records a limit of the country-macro that this study did not fix. Constant nodes, those with no variation over their scored cells, are left out of scoring because they are trivially predictable and correlation is undefined for them, but they stay in the graph as neighbours.

**Probabilistic metrics.** The weighted interval score, a five-quantile CRPS approximation, prediction-interval coverage and mean interval width at 90% nominal, and the probability integral transform. We label the CRPS a biased Riemann approximation on a five-level grid rather than calling it exact, and since WIS and CRPS come out numerically identical here we do not present them as two separate columns.

**Uncertainty method.** Conformal prediction, applied afterwards and independently of the model, in a variant valid for time series, because temporally dependent data breaks the exchangeability assumption behind ordinary split conformal. Section 9.4 gives the construction.

**Seeds and dispersion.** Every configuration is run at five fixed seeds {42, 52, 62, 72, 82}, and every reported cell carries dispersion. A single-seed cell prints on its own and is labelled as such, never as a mean.

**Significance.** Comparisons are paired. Where two arms share a seed set, the delta is computed per seed and summarised with a small-sample *t* interval. At *n* = 5 the two-sided 95% critical value is 2.776, not 1.96, and using the normal quantile would manufacture significance. Where an arm has no seed axis, we use a paired bootstrap over forecast origins at B = 10,000, applying the *same* resampled origins to both arms so shared origin difficulty cancels. A delta whose interval covers zero is reported as *within noise* rather than given a direction. Wilcoxon results over five seeds are supporting evidence only, since the exact two-sided floor at *n* = 5 is p = 0.0625.

**Comparison reference.** Every delta table states its reference in its own header. This answers a real failure in an earlier phase of this work, where a reference computed at one seed and one computed at five supported opposite conclusions about the same experiment.

---

## 9 Results

### 9.1 What the shared encoder buys, and what the graph contributes

Every structural claim in Section 7 holds, and each is checked by machine rather than asserted: the parameter-dimension test, the four-channel test, the covariate test, and *g* ≡ 0 reproducing the graph-free representation to within 10⁻⁶. The same weight set ran at N = 47, 49 and 7,165 with no reshape. No model in our comparison table can do this.

The learned gate is open on every panel, and how far it opens tracks the graph:

| Panel | gate *g* | spatial contribution | fraction with *g* < 0.05 |
|---|---|---|---|
| dengue | 0.604 | 0.637 | 0.0% |
| influenza (Japan) | 0.271 | 0.395 | 0.0% |
| influenza (US regions) | 0.370 | 0.490 | 0.0% |
| influenza (US states) | 0.373 | 0.468 | 0.0% |

**Table 3.** *Learned gate and normalised spatial contribution, pooled over seeds and nodes.*

**That table shows the spatial channel is used, not that it is useful.** An open gate is a fact about the model, not about the value of what flows through it. So we ran the ablation. Same trainer, same seeds, same budget, with *g* forced to zero, paired against the released run at the same seed across all five development panels and all four horizons. That gives 60 cells on RMSE, MAE and correlation.

The answer is specific, and it splits by what the metric measures:

| Metric family | Cells | Spatial channel helps | Hurts | Within noise |
|---|---|---|---|---|
| Error magnitude (RMSE, MAE) | 40 | **0** | 8 | 32 |
| Curve shape (PCC) | 20 | **6** | 1 | 13 |

**Table 4.** *Gate-off ablation, paired by seed against the learned gate. "Helps" means removing the spatial mixing made that cell significantly worse.*

**Message passing buys shape, not magnitude.** Switching the graph off costs correlation in 6 of 20 cells and gains it in one, by 0.011. The model tracks an epidemic's rise, turn and fall better when it can see its neighbours. That shows up on US-regions at long horizon (+0.062 at *h* = 10 and +0.107 at *h* = 15), on US-states, and on dengue at three of four horizons. It buys nothing on error. The spatial channel never significantly improves RMSE or MAE on any panel at any horizon, and it makes them worse in 8 of 40 cells. Those are concentrated on influenza-Japan, where removing the graph improves RMSE by 45.3 at *h* = 3 and 61.7 at *h* = 5, and on COVID at *h* = 5. On those panels neighbouring units share a seasonal phase but not a baseline level, so mixing brings in bias along with the timing.

**We keep the channel, and the ablation is what lets us say what keeping it is for.** Shape is what an early-warning signal is read for. A public-health user acts on when a curve turns, and a forecast that catches the turn while missing the level is more useful than one that does the opposite. The level is set by the temporal trunk, and the graph should not be asked to fix it. What we cannot claim, and do not, is that the graph lowers error. That claim was in our own design premise, and this ablation removes it.

It also explains something otherwise puzzling. The learned gate is *highest* on the panel whose graph is emptiest, with roughly two thirds of dengue nodes having no observed neighbour at a typical origin, which fits a gate reading the degree feature rather than any real epidemiological coupling.

Two limits bound the conclusion. Setting *g* = 0 removes neighbour mixing but keeps the LTR degree feature, so the ablation measures the value of *neighbour information*, not of the graph as a whole. And it does not separate "structure helps" from "any adjacency helps", which a shuffled-adjacency arm would do and which we do not run.

The second reason the channel stays is structural. It is *inductive*, so one parameter set runs unchanged across graphs of 10 to 7,165 nodes, which the transductive parameterisations of Section 3.1 cannot do. That is what makes the transfer question of Section 9.3 askable at all: a model rebuilt for each graph could not ask it.

### 9.2 How large the per-disease correction must be

If a new pathogen is a bounded correction to a shared representation, that bound can be measured. We fitted four adaptation surfaces of increasing capacity on a frozen trunk, once across diseases and once within a disease as a control. Within each seed the same frozen trunk serves every surface, so the only thing changing inside a seed is the surface itself. Every figure is a seed-paired delta against that seed's own affine control, with two-sided 95% *t* intervals at *n* = 5, and a cell counts only if its interval excludes zero.

| Arm | Cells | Significantly better | Significantly worse | Best significant gain |
|---|---|---|---|---|
| Cross-disease | 36 | **7** | 3 | **+19.8%** |
| In-domain control | 12 | **0** | 1 | n/a |

**Table 5.** *Adaptation-surface capacity at five seeds, seed-paired against the affine control.*

**The pattern is the finding, not the headline number.** All seven gains fall at *h* = 15: US-regions +19.8%, +14.3% and +12.2%, Japan +6.6%, +6.4% and +6.3%, and US-states +4.5%. All three losses fall at Japan *h* = 3, between −16.6% and −22.5%. The in-domain control gains nothing anywhere. Three things follow. A richer adaptation surface buys accuracy at long horizon and costs it at short horizon. The effect is specific to transfer rather than a sign that the model is too small in general, which the 0 of 12 in the control rules out. And the correction stays a modest fraction of the trunk.

**One caveat has to travel with this result.** The horizon where extra capacity helps, *h* = 15, is exactly the horizon where our pre-registered case-study arm has **zero** adaptation pairs. The one horizon this improvement addresses is the one an emerging-pathogen scenario cannot fit an adapter for. Said in the same breath, this is a bounded and honest result. Said on its own, it would oversell.

### 9.3 Where transfer holds, and where it fails

This is the central empirical result. Under the three-way leave-one-disease-out design of Section 5.3, each delta is measured against the single-disease model trained on that same dataset, which we call the *ceiling*, paired by seed.

**Across the 36 attributable cells: 25 significantly worse than the ceiling, 10 within noise, 1 better.** Four further cells, COVID at *h* = 10 and *h* = 15, are left out rather than counted as losses, for the fold-boundary reason given in Section 6.4.

The pattern across horizons is the more useful finding:

| Horizon | Transfer better | Within noise | Transfer worse |
|---|---|---|---|
| *h* = 3 | 1 | 5 | 4 |
| *h* = 5 | 0 | 4 | 6 |
| *h* = 10 | 0 | 1 | 7 |
| *h* = 15 | 0 | 0 | 8 |

**Table 6.** *Cross-disease transfer against the in-domain ceiling, by horizon.*

At three weeks the transfer arm is close to a model trained on the disease itself, and most cells are within noise. By ten and fifteen weeks it is worse everywhere and by a wide margin, reaching −38% on influenza-Japan RMSE and −51% on influenza-US-regions MAE. **A frozen foreign trunk carries enough short-range structure to match in-domain training for a few weeks ahead, then loses it.** Dengue and influenza drive the pattern, and it does not depend on the COVID cells we excluded.

The zero-shot arm is worse everywhere: behind the ceiling in **all 36** cells, significantly in 34, at costs from 1.4% to 138.8%. Averaging in-disease adaptation surfaces produces nothing usable on a disease the trunk has never seen.

#### 9.3.1 The result is not an artefact of undertraining

Every transfer number is a ratio, so undertraining on either side would corrupt it. We checked both.

All fifteen transfer trunks stopped early on the same branch, each running exactly 12,000 steps past its best checkpoint. None used up its 91,000-step budget, so the cosine schedule was set to a length early stopping guaranteed it would never reach, and the learning rate at every selected checkpoint sat between 1.00 × 10⁻³ and 9.3 × 10⁻⁴. The schedule did nothing, which left the transfer conclusion open. So we re-ran folds with early stopping turned off, running all 91,000 steps. Best validation loss came at **step 7,000 and never improved through step 91,000**, and the scored records came back **bit-identical** to the truncated run.

**All 25 single-disease reference runs also stopped early**, between 18 and 77 epochs of 80. The instinct that this cancels out is wrong, and the direction matters. The transfer trunk is demonstrably at its own optimum, so if the reference sits below its own, the reference error is inflated and our reported deficits **understate** the true gap.

#### 9.3.2 The result survives a stronger reference

A reviewer is entitled to ask whether the transfer arm was compared against a weak opponent. We made the reference stronger on purpose and re-ran the comparison.

Averaging the five seeds' forecasts is a free improvement. For squared error, the error of the mean of *K* forecasts is below the mean of the errors whenever the forecasts disagree, and it needed no retraining because the count-space forecasts of all 25 reference runs were already archived. It improved the reference clearly: dengue *h* = 3 RMSE went from 42.06 ± 6.04 to 37.28, and US-regions *h* = 15 from 812.80 ± 68.1 to 760.99. A stronger reference can only make transfer look worse. Ensembling only the reference would be a mismatch working against us, so **both arms got exactly the same treatment**: five-seed ensembles, the same datasets, the same forecast origins, and one shared bootstrap draw applied to both so the pairing is real.

**The verdict did not move: 1 better, 10 within noise, 25 worse.** The pattern across horizons repeated cell for cell, at 1/5/4, 0/4/6, 0/1/7 and 0/0/8. Individual cells did shift, and the shifts are informative. Japan *h* = 3 moved from a loss to within noise at +9.8%, and US-states *h* = 3 became a real transfer win at +3.5% with a bootstrap interval of [+1.0, +7.0]. No verdict count changed.

So the conclusion holds under a change of estimator (single runs to ensembles), a change of significance test (seed-paired *t* at *n* = 5 to a paired origin bootstrap at B = 10,000), and a much stronger reference. This is the strongest evidence in the paper that the horizon boundary belongs to the problem rather than to our experimental choices.

Two limits are worth stating. Ensembling uses up the seed axis, so the seed-paired interval cannot be computed on it. The ensemble table is a second analysis rather than a replacement, and the per-seed table is still the one that carries seed dispersion. And the zero-shot arm cannot be analysed this way, because its quantile forecasts were not archived.

### 9.4 Calibrated uncertainty

**The quantile head is not calibrated, and on the held-out disease it is badly off.** Coverage of the nominal 90% interval runs from 0.492 to 0.918 across the development panels, and from **0.275 to 0.701** on the Ebola query set, where the primary arm covers just 0.275 at *h* = 15 against a 0.90 target. Between 28% and 71% of Ebola cells fall completely outside the predicted quantile range. Coverage gets worse with horizon on every panel, so a single global multiplier would over-correct the short horizons and under-correct the long ones. Every correction here is per horizon.

**Carving a calibration split out of the case study is ruled out by arithmetic, not by preference.** The primary arm has 48, 38, 18 and 0 adaptation pairs at *h* = 3, 5, 10 and 15, and split conformal needs *n* ≥ 9 just to form a *finite* 90% interval. The *h* = 10 case splits to exactly that boundary, *h* = 15 has nothing to split, and any split would halve the adapter's fitting data. Ensemble batch prediction intervals are not available either: no target-disease data enters training, so there are zero out-of-bag residuals.

**The construction is therefore a cross-disease transfer of the correction.** We score each archived interval with a width-normalised conformity measure, *E* = max(q_lo − y, y − q_hi) / (q_hi − q_lo + ε), fit λ_h as the conformal (1 − α) quantile of *E* pooled over held-out development panels, then apply [q_lo − λ_h·w, q_hi + λ_h·w]. Width normalisation is necessary, not cosmetic. Dengue's test counts have a median of 2 and COVID's a median of 4,605, and an unnormalised residual would let COVID dominate across that gap. Online adaptation then updates α_t ← α_t + γ(α − f_t), where f_t is the *fraction* of districts missed at origin *t* rather than a binary miss. That gives the same bound with far less variance per step.

Two statements go with any number this produces. **There is no finite-sample guarantee on the held-out disease**, because λ_h is fitted on other diseases, so what transfers is the correction itself. And with only 18 scored origins, the online method's worst-case coverage deviation is (α₁ + γ)/(Tγ) = 0.167, which we state before a reviewer has to.

**One methodological correction is central, and we report it against ourselves.** An online conformal loop that updates at origin *t* using the outcome of origin *t* − 1 is valid at a one-step horizon and is an *oracle* at every longer one, because a forecast issued at *t* is about week *t* + *h* and cannot be scored until then. The lag is not a tuning parameter. It is forced by the definition of the horizon. Under an honest lag-*h* stream, only 15, 13, 8 and 3 of the 18 origins ever get feedback at *h* = 3, 5, 10 and 15. At the longest horizon, fifteen of eighteen origins are forecast at the starting α and the online mechanism barely runs.

The corrected results are below. `+λ` is the frozen cross-disease correction, which uses **no** target-disease truth. `+ACI-lag` adds the honest online stream. The oracle column is shown only so the size of the correction is visible.

| Arm | *h* | raw | +λ | **+ACI-lag** | +ACI (oracle) | origins adapting |
|---|---|---|---|---|---|---|
| L12 zero-shot | 3 | 0.460 ± .025 | 0.833 ± .014 | **0.863 ± .009** | 0.864 | 15 / 18 |
| L12 zero-shot | 5 | 0.421 ± .020 | 0.813 ± .042 | **0.862 ± .011** | 0.870 | 13 / 18 |
| L12 zero-shot | 10 | 0.396 ± .028 | 0.896 ± .012 | **0.899 ± .013** | 0.904 | 8 / 18 |
| L12 zero-shot | 15 | 0.363 ± .021 | 0.921 ± .009 | **0.921 ± .009** | 0.920 | 3 / 18 |
| L12 few-shot | 3 | 0.491 ± .062 | 0.829 ± .055 | **0.859 ± .032** | 0.862 | 15 / 18 |
| L12 few-shot | 15 | 0.275 ± .103 | 0.651 ± .138 | **0.658 ± .132** | 0.812 | 3 / 18 |

**Table 7.** *Coverage of the nominal 90% interval on the Ebola query set, mean ± sd over five seeds. The secondary L20 arm over-covers throughout, at 0.90 to 0.98 after correction, because its raw intervals are already far too wide.*

**Two things follow, and the first matters more.** The oracle made almost no difference. `+ACI-lag` matches the un-lagged column to within 0.01 in fifteen of sixteen cells. The one exception is the primary arm at *h* = 15, where removing the oracle costs 0.154 of coverage, from 0.812 down to 0.658. That is exactly the arm and horizon where only three of eighteen origins ever adapt, and which has zero adaptation pairs. The concern was real, its effect is limited to one cell, and every other calibrated figure in this paper stands without oracle feedback.

**The transferred correction does the work, not the online adaptation.** `+λ` on its own, fitted entirely on other diseases and reading no Ebola outcome at all, already lifts coverage from a range of 0.28 to 0.70 up to a range of 0.65 to 0.98. ACI adds about 0.03 at short horizons and nothing at long ones. That is a stronger result than the reverse would have been, and it is the form that is actually useful. A new pathogen gives you no outcomes to adapt on during the weeks when a forecast matters most, so a correction that arrives already fitted is one you can apply to the very first forecast you issue.

One diagnosis that came out of this analysis reframes the calibration story, and it stands on its own regardless of the corrected numbers. The single-disease reference models are themselves badly calibrated, covering 0.531 to 0.934 with a mean absolute deviation from nominal of 0.158. **A model with no domain shift at all is off by 0.158 on average.** Calibration is therefore not mainly a transfer problem. The five-quantile head is under-dispersed in general, and domain shift was never the main cause. Read the direction carefully. Calibrated transfer landing closer to nominal than the in-domain reference does *not* mean transfer forecasts better. It means the reference's intervals are also wrong.

### 9.5 The Ebola case study

The case study is the paper's held-out test of the whole construction. It is scored **exactly once**, against a configuration written down before any Ebola number existed.

**The pre-registration is real, not a formality.** Both support-set arms were built, validated and hashed before either was scored: a 12-week primary arm (cutoff 2014-06-28, 13 support columns, 59 support cells, digest `08d657dc…`) and a 20-week secondary arm (cutoff 2014-08-23, 21 columns, 113 cells, digest `e9b9ac0b…`). The training code re-hashes both at score time, refuses to run if anything has moved, and stamps the protocol digest into every record it writes. Ebola enters no early-stopping monitor, no hyperparameter selection and no trunk training.

The pre-registration also names the evaluation set. Both arms are scored on one common set of 18 origins, giving 757, 766, 765 and 642 scored pairs across 57 to 59 districts, figures we corrected *before* the run by reading the scoring path rather than assuming it. Every comparator scores bit-identical cells, so the difference between arms comes purely from the support set. We report two floors, persistence and a support-set mean. Seasonal-naive is absent because a 52-week panel puts the *t* − 52 lag outside the panel at every scored origin, making it a duplicate of persistence.

#### 9.5.1 The pre-registered criterion was not met

The registered criterion required the adapted arm to beat persistence at *h* = 3 or *h* = 5, with a confidence interval that excludes zero. **It does not.** On a paired bootstrap over districts, all four confirmatory cells span zero:

| Cell | Difference | 95% interval |
|---|---|---|
| *h* = 3 RMSE | −5.21 | [−11.85, +0.35] |
| *h* = 5 RMSE | −6.36 | [−14.91, +1.34] |
| *h* = 3 MAE | −1.21 | [−5.45, +2.39] |
| *h* = 5 MAE | −2.79 | [−8.51, +2.37] |

**Table 8.** *The pre-registered confirmatory family, primary arm, adapted. Negative favours the model.*

We report this as a miss. Reaching a *verdict* also meant fixing our own instrument. The headline statistic is a node-averaged country-macro, while the interval originally used to judge it was a cell-pooled quantity rebuilt from per-origin sufficient statistics. On this panel the two differ by a factor of 1.5 to 2.0, far enough that the reported point estimate fell outside its own quoted interval. A point estimate and an interval measuring different things cannot settle anything. The interval above is recomputed on the reported statistic, and it adds the district axis that the registered analysis named first and the original bootstrap left out.

This is what pre-registration is for, and it earned its keep in both directions. One registered expectation was cleanly proved wrong against us. Another, that zero-shot would lose to persistence at every horizon, was registered at high confidence and proved wrong in our favour.

#### 9.5.2 Adaptation makes it worse: the inversion

The most interesting finding in the case study is that **the arm given no Ebola data at all beats the arm adapted on Ebola data, at every horizon and on both support sets.**

| Arm, primary support set | *h* = 3 | *h* = 5 | *h* = 10 | *h* = 15 |
|---|---|---|---|---|
| Zero-shot, MAE | 22.85 ± 0.50 | 24.00 ± 0.17 | 23.55 ± 0.15 | 20.14 ± 0.12 |
| Adapted, MAE | 24.69 ± 0.48 | 24.61 ± 0.62 | 26.05 ± 3.89 | 20.10 ± 0.52 |
| Zero-shot skill vs persistence | +11.8% | +12.4% | +17.5% | +36.9% |
| Adapted skill vs persistence | +4.7% | +10.2% | within noise | +37.0% |

**Table 9.** *Primary arm, mean ± standard deviation over five seeds. Floors are deterministic, so the dispersion shown is the model's own seed spread.*

On the corrected district bootstrap, the wins that survive sit in the zero-shot arm. It beats persistence with intervals excluding zero at *h* = 3 ([−13.33, −1.54]), *h* = 10 ([−13.69, −2.04]) and *h* = 15 ([−44.46, −6.08]), while the adapted arm clears zero only at *h* = 15.

**This is not an artefact of the point estimator. It holds on the full predictive distribution.** Scoring the archived quantiles with the weighted interval score, which grades the whole forecast rather than just its median, the zero-shot arm is better in **7 of 8** arm-by-horizon cells: all four on the secondary arm and three of four on the primary. Only the primary arm at *h* = 10 favours adaptation, at 19.77 against 20.50. A median-versus-mean argument could explain a point-error inversion, but it cannot explain this one.

The instability tells us as much as the means do. The secondary arm, which has *more* support data, is both worse and far less stable. Its seed standard deviations are 15.24, 13.25 and 10.70 on means of 47.03, 51.58 and 40.85, against under 0.5 for the matching zero-shot cells. Its 90% intervals are wide to match, 94 to 138 in count units, with a seed spread reaching 52. More adaptation data producing a worse, wider and much less stable fit is a symptom rather than a result, and we do not yet have an explanation for it.

**We are careful about how strongly we state this.** The comparison is across arms that share a trunk per seed, so a seed-paired test is appropriate, and it is significant in 4 of 16 cells. We report the inversion as a solid *observation* with a partly significant test behind it, not as an established rule. Two explanations fit. Fitting 1,428 parameters on 59 highly correlated support cells may overfit the early exponential phase of an outbreak, and the pooled support-only scaler is estimated from exactly the weeks that look least like the query period. Telling those apart is future work.

One imbalance affects the margins. Our point forecast is the count-space median, which is right for MAE but sits low for RMSE, while both naive floors are raw count predictions with no transform bias. The comparison is handicapped on our side only. The direction is conservative, so we lead with MAE for this case study.

### 9.6 Supporting evidence

Three further experiments fill in the picture without carrying claims of their own.

**Joint multi-disease training does not beat per-disease training.** One trunk trained on all development panels at once, with per-disease adaptation surfaces, is worse on every panel: Japan at −6.6%, −4.4%, −4.5% and −19.8% across horizons, US-regions between −0.2% and −11.2%, and US-states and dengue at roughly zero. Other loss weightings are worse still, with square-root weighting costing dengue 61% and 62% at the two shortest horizons. This is a clean negative result and we report it as one.

**The reproduced baselines.** We re-ran EpiGNN, Cola-GNN and HeatGNN as epidemic-GNN comparators, and MTGNN as the non-epidemic control, on our own data, splits, horizons, scoring and five seeds, a condition we fixed before looking at any result. We never copy a baseline's own published metrics: each model exports count-space test predictions and every number is computed by our scorer. Against EpiGNN our model is better in 9 of 16 comparisons and worse in 1. Against Cola-GNN it is better in 4 of 12 and worse in none. HeatGNN covers all three influenza panels at all four horizons, where our model is better in 3 of 12 and worse in none, the remaining 9 being within seed noise.

MTGNN needs separate comment. It reproduces cleanly on its own traffic benchmark, which is what qualified it as a control. Moved onto epidemic panels it **breaks down**, returning a single constant value on 47 of 80 prediction files, with negative pooled correlation on every influenza set. We report this as a finding about how the traffic-forecasting lineage behaves on sparse epidemic count data, not as a win for our model. Beating a constant is not evidence of quality.

**Every model in the group, ours included, loses to simple naive floors more often than it wins.** On the development panels our model beats the best naive floor in 8 of 20 dataset-and-horizon cells on RMSE after ensembling, up from 6. EpiGNN wins 5 of 16, Cola-GNN 5 of 12, and MTGNN 2 of 16. Reporting only the head-to-head would give a misleading impression of where the field stands, ours included.

**Our weakest head-to-head panel is influenza US-states, and which model leads there depends on how the error is aggregated.** On the headline country-macro RMSE used throughout this paper we are worse than EpiGNN at *h* = 3, better at *h* = 15, and within seed noise at *h* = 5 and *h* = 10; against HeatGNN all four cells are within noise. On cell-pooled RMSE, which is the definition the baselines' own papers use, we are behind EpiGNN at every horizon by 2.8% to 13.1% and behind HeatGNN at three of four. The two aggregations disagree because a mean of per-node RMSEs and an RMSE pooled over all cells are different statistics, and by Jensen's inequality the first is always the smaller; US-states is the one panel where that gap is wide enough to change the ranking. We state the pooled comparison as well as our own because a reader coming from those papers will compute the pooled one.

Two panels are clearly weak, and we disclose them rather than patch them. Influenza-Japan loses to seasonal-naive by 32% to 56% at every horizon, for the lookback reason in Section 5.2, and COVID loses to its floors by 16% to 134%, for the fold-boundary reason in Section 6.4. Neither can be fixed with any lever inside the frozen protocol, and presenting them any other way would misrepresent where the method works.

### 9.7 Ablations of the adaptation procedure

Both ablations here test the adaptation *procedure*, not the headline result. The pre-registered case study is defined on the frozen-trunk adapter, so we report them whatever their sign and never re-frame the case study around them.

**Episodic meta-learning buys nothing.** We meta-trained the adaptation surface with ANIL, updating only that surface in the inner loop, against a control that sees the same episodes and the same number of outer updates with no inner loop. Without that control, any gain could just be the effect of more training. Both arms use the affine surface named in the pre-registration, at five seeds, paired by seed. We ran four folds: the original dengue-to-influenza direction, and the three leave-one-disease-out folds of Section 9.3, so the question is asked once per held-out disease.

Across 32 cells, **none is significantly better and one is significantly worse**: US-regions at *h* = 10, at −1.35% with an interval of [−2.52, −0.19]. That cell lies in the dengue-to-influenza fold, and every cell of all three disease-out folds is within noise. The held-out meta-objective agrees, and on the one fold where it separates at all, the dengue fold, it separates against ANIL, at −1.2% with an interval of [−2.41, −0.03]. So the inner loop does not even win the objective it optimises, while costing between 5% and 46% more wall-clock depending on the fold.

Three things bound how far this reads. Both arms warm-start from the transfer trunk, making this meta-fine-tuning rather than meta-learning from scratch. The meta-test fits a fresh adapter on full training folds, so nothing here measures few-shot behaviour. And episode yield varies with mask density: the dengue-to-influenza fold drew 2,100 usable outer updates from 8,000 episodes, while the disease-out folds drew 3,950 to 5,100. A fourth bound has been removed rather than restated. The original fold drew every episode from one disease, so episodes varied population and forecast origin rather than disease, which is the axis that already works; the influenza-held-out fold meta-trains across dengue and COVID, so its episodes do vary disease, and the answer did not change.

**The epidemiology-informed ablation is also a null.** Adding a growth-rate plausibility penalty to the training objective changes no cell beyond seed noise on any of the four small panels, because the model's forecasts are already smoother than the data at every bound we calibrated. We report that as a measured null rather than an untested design choice.

### 9.8 Explainability

*[PENDING. A minimal global attribution over the four core channels is scoped and will be reported here. This version claims no attribution result, and the comparison in Section 3 makes no explainability claim on our behalf that the code does not currently support.]*

---

## 10 Threats to Validity

We list the limitations most likely to affect a reader's confidence, including several we have not fixed.

**One outbreak, one pathogen.** Everything in the case study rests on the 2014 West African Ebola epidemic. A framework for emerging diseases tested on a single emergence is a demonstration, not a generalisation.

**The country-macro aggregation is less unbiased than its name suggests.** It averages twelve dengue countries with equal weight and applies no minimum-node floor, so a country contributing 9 scored nodes carries the same 1/12 weight as one contributing 5,189. At the shortest horizon this pulls the two aggregations far apart, with country-macro at 49.98 against node-mean at 23.05 on identical nodes. We report both columns everywhere, and we **deliberately did not add a minimum-node floor**, because picking an aggregation rule after seeing which one flatters the result cannot be defended however good the reasoning. Fixing it belongs in a pre-specified follow-up.

**Our own short-horizon dengue performance is a defect we have measured but not repaired.** The model loses to persistence at *h* = 3 by 18% after ensembling, and the loss is concentrated rather than spread out. The worst 1% of nodes carry 43% of the excess error, with the damage falling on small, high-variance countries such as Bolivia, at 9 nodes and +106%, while the model beats persistence outright on Colombia's 718 nodes and on the Dominican Republic. This is a fixable engineering problem rather than a limit of the approach, and we say so without having fixed it.

**The masked-week cost of the cumulative correction is real.** The running-maximum transform of Section 6.2 conserves mass but drops 368 weeks, only 31% of which recover within three weeks, and two districts central to the outbreak keep just 7 and 17 of 30 scored weeks. We have not tested a level-rebasing alternative.

**The transfer folds do not isolate disease perfectly.** Two of the three folds change graph, geography and node count as well as the disease. Only the COVID and influenza-US-states pair holds the graph bit-identical, and that pair is confounded by the fold boundary of Section 6.4. The honest statement is that this trunk does not transfer across *these* folds, which is a weaker claim than a clean disease-only effect.

**Serial dependence in the loss differences is not modelled.** Our origin bootstrap resamples independently where a moving-block scheme would fit better, and measured first-order autocorrelation reaches +0.91 on some cells. A block-bootstrap sensitivity check leaves the long-horizon case-study conclusions intact, but we have not applied it throughout.

**Multiplicity is controlled on the confirmatory family only.** The four-cell pre-registered family is the controlled comparison, and it failed. The wider grid is exploratory and labelled as such rather than corrected.

**Two measurement limits come from the protocol itself.** Five seeds are too few for the supporting non-parametric test, so we rely on the resampling intervals. And the five-level quantile grid is coarser than the eleven or twenty-three levels used by operational forecasting hubs, which is why the weighted interval score and CRPS coincide here and why our interval metrics are less fine-grained than a hub submission's.

**The spatial channel earns its place on shape, not on error magnitude**, and a reader is entitled to ask whether that justifies a graph model. It buys two things. It buys correlation with the observed curve, which is what an early-warning signal is read for, and it buys the inductive property that lets one parameter set span graphs of 10 to 7,165 nodes, which is what makes the cross-disease comparison possible. Neither is an accuracy claim. We did not run a shuffled-adjacency arm, which would separate "structure helps" from "any adjacency helps".

**One required capability is not delivered in this version**: explainability. It is scoped and registered, and nothing is claimed for it here.

---

## 11 Conclusion

We set out to build a forecasting model that is disease-agnostic by construction rather than by evaluation, and to use it to ask when a shared epidemic representation actually transfers.

The construction works, and it can be checked by machine. A shared trunk of 142,305 parameters with no node-indexed dimension reads a four-channel representation identical across six datasets, adapts to a new disease through 1,428 parameters, and runs unchanged on graphs from 10 to 7,165 nodes, over data held back by 101 leakage checks with planted negative controls. A new pathogen *can* be written as a small bounded correction, and Section 9.2 puts a bound on how large. Extra capacity helps in 7 of 36 cross-disease comparisons and 0 of 12 within a disease, all of it at the longest horizon.

The transfer result is a boundary rather than a win. A frozen foreign trunk matches in-domain training at three weeks and loses everywhere by fifteen, and that held under every attempt we made to break it: a full-budget retraining that came back bit-identical, a check that the reference was not handicapped differently, and a symmetric ensemble analysis that strengthened both arms and returned the same verdict from a different estimator and a different significance test. On the held-out pathogen the pre-registered criterion was not met, and the arm given no target data beat the arm adapted on it. What did transfer cleanly was the calibration. A conformal correction fitted on other diseases, reading no Ebola outcome, brings coverage from a range of 0.28 to 0.70 up to near nominal.

Two findings bear on our own design, and we state them plainly. The inductive spatial channel, the property that separates this build from the non-graph cross-disease forecasters, improves the shape of a forecast and not its magnitude: correlation in 6 of 20 cells, error in none of 40. We keep it because shape is what an early-warning signal is read for, and because the inductive property is what makes the transfer question askable across graphs of very different sizes. We do not keep it because message passing forecasts better, which on these panels it does not. Second, the field's uncertainty quantification, ours included, is under-dispersed for reasons that have little to do with domain shift. A model with no domain shift at all is off by 0.158 on average.

A third result belongs beside them. Meta-learning the adaptation surface does not help either, being better in none of 32 cells across four folds and worse in one, so adapting better is not what this problem needs.

Each of these is a constraint an operational early-warning system would have to design around, and none is visible without a build that holds the disease constant. What remains is stated plainly rather than promised loosely. The explainability component is registered and not yet run, and the case study rests on a single outbreak. The framework, the datasets, the evaluation protocol and every artefact behind these numbers are available, so each can be checked and extended.

---

**Code and data.** All code, harmonised datasets, configurations, pre-registration documents and the scripts that generate every table in this paper are available at `https://github.com/<organisation>/<repository>`.

---

## References

Cao, Q., Jiang, R., Yang, C., Fan, Z., Song, X., & Shibasaki, R. (2022). MepoGNN: Metapopulation epidemic forecasting with graph neural networks. *ECML-PKDD*, LNCS, 453–468.

Clarke, J., Lim, A., Gupte, P., Pigott, D. M., van Panhuis, W. G., & Brady, O. J. (2024). A global dataset of publicly available dengue case count data. *Scientific Data*, 11, 296.

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.

Deng, S., Wang, S., Rangwala, H., Wang, L., & Ning, Y. (2020). Cola-GNN: Cross-location attention based graph neural networks for long-term ILI prediction. *CIKM*.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.

Ezzat, M., Malek, Y. M., AbdelKader, T., & Badr, N. (2026). Spatio-temporal epidemic forecasting with graph-based transformer. *International Journal of Health Geographics*, 25, 33.

Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML*, 1126–1135.

Gao, J., Sharma, R., Qian, C., Glass, L. M., Spaeder, J., Romberg, J., Sun, J., & Xiao, C. (2021). STAN: Spatio-temporal attention network for pandemic prediction using real-world evidence. *JAMIA*, 28(4), 733–743.

Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.

Global Administrative Areas (GADM). (2022). GADM database of global administrative areas, version 4.1.

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.

Jin, M., Koh, H. Y., Wen, Q., Zambon, D., Alippi, C., Webb, G. I., King, I., & Pan, S. (2024). A survey on graph neural networks for time series. *IEEE TPAMI*, 46(12), 10466–10485.

Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). Diffusion convolutional recurrent neural network: Data-driven traffic forecasting. *ICLR*.

Liu, S., & Cao, L. (2026). Dynamic spatiotemporal graph attention networks for cross-regional multi-disease forecasting and intervention optimization. *Frontiers in Public Health*, 14, 1720620.

Liu, Z., Wan, G., Prakash, B. A., Lau, M. S. Y., & Jin, W. (2024). A review of graph neural networks in epidemic modeling. arXiv:2403.19852.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 4765–4774.

Mao, J., Han, Y., Tanaka, G., & Wang, B. (2023). Backbone-based dynamic graph spatio-temporal network for epidemic forecasting. arXiv:2312.00485.

Murph, A. C., et al. (2026). epiFFORMA: a disease-agnostic approach to forecasting emerging pathogens. *Nature Communications*, 17, 4255.

Panagopoulos, G., Nikolentzos, G., & Vazirgiannis, M. (2021). Transfer graph neural networks for pandemic forecasting. *AAAI*, 35(6), 4838–4845.

Roster, K., Connaughton, C., & Rodrigues, F. A. (2022). Machine learning methods for forecasting dengue and cross-pathogen transfer. *PLOS Computational Biology / PMC9222348*.

Ruan, S., Li, J., Wei, J., Xu, Z., Bao, J., Xu, J., Qiu, J., Wang, S., Wang, X., & Yuan, H. (2026). Prior knowledge-enhanced spatio-temporal epidemic forecasting (STOEP). arXiv:2602.22270.

Wang, L., Adiga, A., Chen, J., Sadilek, A., Venkatramanan, S., & Marathe, M. (2022). CausalGNN: Causal-based graph neural networks for spatio-temporal epidemic forecasting. *AAAI*, 36(11), 12191–12199.

WHO Ebola Response Team. (2014). Ebola virus disease in West Africa — the first nine months of the epidemic and forward projections. *NEJM*, 371(16), 1481–1495.

Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020). Connecting the dots: Multivariate time series forecasting with graph neural networks. *KDD*.

Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for deep spatial-temporal graph modeling. *IJCAI*, 1907–1913.

Xie, F., Zhang, Z., Li, L., Zhou, B., & Tan, Y. (2022). EpiGNN: Exploring spatial transmission with graph neural network for regional epidemic forecasting. *ECML-PKDD*, LNCS 13718.

Xu, C., & Xie, Y. (2021). Conformal prediction interval for dynamic time-series. *ICML*, PMLR 139.

Yu, B., Yin, H., & Zhu, Z. (2018). Spatio-temporal graph convolutional networks. *IJCAI*.

Zheng, Y., Jiang, W., Chen, T., Zhou, A., Zhan, C., Nguyen, Q. V. H., & Yin, H. (2024). Epidemiology-informed graph neural network for heterogeneity-aware epidemic forecasting (HeatGNN). arXiv:2411.17372.

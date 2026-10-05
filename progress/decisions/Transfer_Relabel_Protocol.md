# Transfer relabel: does a transferred trunk use the real map? Test protocol

The user commits this file before any relabelled forecast is scored. The runner
`experiments/transfer_relabel/run.py` refuses the scored run unless this file is committed with no
local edits AND its sha256 matches the stamp passed on the command line, and it writes that sha256
into every record. The sha256 is taken over the file's bytes with CRLF turned into LF, so a Windows
checkout cannot change it. Once committed this file is frozen. Any later change goes in a separate
amendment file, dated, and written before the data it affects, so the hash in every record keeps
matching this file.

Nothing here touches Ebola, `results/`, or any training code. No model is trained or fitted. Every
record lands in `experiments/transfer_relabel/`, which is the exploratory lane
(`experiments/README.md`), so records carry `lane: EXPLORATORY` and are not `results/` artifacts.

---

## 1. The question

Does a trunk trained on other diseases use the real map when it forecasts a disease it never trained
on? The paper needs a number for this because Ebola is exactly that case: a new disease on a map no
trunk has seen.

What is already known, and why it does not answer the question:

- The single-disease probe `diagnostics/graph_probe/t2.py` relabels the map at inference and finds a
  cost of +0.04 to +1.04 percent (CLAUDE.md section 4). It runs on `results/single/` checkpoints,
  each trained on the disease it is scored on. It says nothing about a transferred trunk.
- That probe also does not hold degree fixed. It builds `A_np[np.ix_(p, p)]` from an unconstrained
  `rng.permutation(N)` (`t2.py:52-54`) and calls the plain forward (`t2.py:23`), which computes
  degree from the graph it is handed (`models/encoder.py:57,60`). So each district's degree feature
  became some other district's degree. The degree multiset is kept, the per-district degree is not. Its numbers measure
  "wrong neighbours AND wrong degree feature" together.
- The D2 retrain (`progress/planning/Gap_Ledger.md:299-311`) trained single-disease models on a
  relabelled map and found a small seed-stable real-map advantage on dengue, about 2 to 4 percent at
  h10 and h15, that the inference-only probe did not show. So an inference-only null does not rule
  out an effect a retrain would find.

## 2. Checkpoint families

**Primary, the only family that decides.** The 15 LDO3 transfer checkpoints,
`results/lodo/encoder_ldo3__<fold>__seed<S>__ckpt.pt`, 3 folds by 5 seeds, each scored on its own
held-out panels as listed in the checkpoint's `adapter_scope`:

| fold | trunk trained on | held-out panels scored | map seen by the trunk in training? |
|---|---|---|---|
| dengue | influenza (3 panels), covid | dengue | no |
| influenza | dengue, covid | influenza_japan, influenza_us-regions, influenza_us-states | japan no, us-regions no, us-states **yes** |
| covid | dengue, influenza (3 panels) | covid_us-states | **yes** |

`covid_us-states` and `influenza_us-states` share a bit-identical map (`np.array_equal` on `A_geo`
is True, re-measured for this protocol). So in the covid fold the trunk trained on covid's exact map
through influenza, and in the influenza fold it trained on us-states' exact map through covid. The
three unseen-map panels, dengue, influenza_japan and influenza_us-regions, are the Ebola-shaped ones.
That split is declared here, before scoring, and the paper sentences in section 7 keep it.

**The head is the ADAPTED transfer arm, not zero-shot.** Each checkpoint stores the adapter fitted on
the held-out disease's train fold with the trunk frozen (`train/lodo.py:546`). That is the head the
archived `encoder_ldo3__<panel>__seed<S>.json` records were scored with. The zero-shot head of these
15 runs was never saved (`train/lodo.py:540-545`), so the zero-shot arm cannot be tested on them at
any price short of retraining the trunks. Measured before writing this: the stored head reproduces
the archived count-space quantile forecast of `encoder_ldo3__covid__seed42` with a maximum difference
of exactly 0.0, and the dry run re-checks every checkpoint.

**Exploratory, descriptive only, no verdict.** The two saved zero-shot heads,
`encoder_ldo3full__covid__seed42__ckpt.pt` and `encoder_ldo3full__dengue__seed42__ckpt.pt`, using
`meta.zeroshot_adapter`. The influenza `ldo3full` checkpoint has no zero-shot head, so it is not
here. Why they are in: they are the only zero-shot heads saved anywhere in the project, and Ebola's
wins against persistence come from the unadapted arm, so this is the nearest analogue. What they
cost: one seed each, so no interval and no verdict, and a different trunk family (`ldo3full`, early
stop disabled), so they are never read against the primary. They are printed as a draw mean with the
spread over draws. The stored zero-shot head reproduces its archived quantile forecast with a maximum
difference of exactly 0.0 (covid, measured).

**Left out, and why.**

- **ANIL, 40 checkpoints, both arms.** The ANIL checkpoint does not store the head that produced the
  ANIL records. `train/anil.py:611` writes the `ad` returned by `meta_train`, which is the
  meta-learned starting head. The archived ANIL records come from a fresh adapter fitted in
  `meta_test` (`train/anil.py:547`) that is never written. I checked this prediction against
  prediction, with no truth involved: on `ldo3covid` seed 42 the stored head's median forecast differs
  from the archived ANIL median forecast by a median 54 to 98 percent relative, both arms, all four
  horizons. So no ANIL checkpoint can pass the drift gate in section 5 without refitting an adapter,
  and refitting is training, which is out of scope. To bring ANIL in later: a dated amendment that
  refits the adapter under `meta_test`'s seeding, requires the refit to reproduce the archived ANIL
  records within the drift tolerance first (whether a GPU refit is reproducible to that level is
  unknown), and only then relabels. What leaving it out costs: this protocol says nothing about
  meta-learned trunks.
- **The graph-controlled pair, 10 checkpoints.** The only thing that sets the pair apart from the
  primary family is that its trunk trained on the held-out panel's exact map through the other
  disease. Two of the five primary panels already have that property. Adding the pair adds ten
  checkpoints and a second set of verdicts to choose between, without adding a question. What leaving
  it out costs: we do not learn whether a trunk trained on one disease on the identical map behaves
  differently from a two-disease trunk.
- The `ldo3full` adapted heads, the two-way LDO and LODO families, every single-disease checkpoint,
  and everything Ebola. Ebola is out by the pre-registration (`progress/decisions/Ebola_Prereg.md`,
  section 5): its arms are scored once and this protocol does not reopen them.

## 3. The perturbation

**Deciding perturbation: `degclass`, a relabel within degree classes.** Group the districts by their
number of neighbours. Inside each group, shuffle the labels. Build `A_perm = A[p][:, p]` and leave
node features, targets and masks in their own order. So district i keeps its own history and its own
neighbour count, and gets the neighbours of some other district with the same count. Because every
district's degree is unchanged, both places the model reads degree see the truth: the degree feature
(`models/encoder.py:60`) and the mixer's normalisation (`models/spatial.py:37-50`). Only WHO the
neighbours are changes. The forward goes through `explain.forward_fixed_deg` (`explain.py:240`),
which reads degree from the true graph. Under `degclass` that is identical to the plain forward, and
the runner's selfcheck asserts it.

**Printed beside every cell, never decides: `global`, an unconstrained relabel** of all districts,
run through the same `forward_fixed_deg`, so the degree feature reads the true degree. This is the
option the brief preferred. I did not make it the deciding one because it only holds the degree
FEATURE fixed. The mixer's normalisation still sees a new neighbour count for most districts (table
below), so a "map used" verdict under `global` could come from the changed count rather than from the
changed neighbours. What `degclass` gives up in exchange: on a small graph with few districts per
degree class it changes fewer neighbours.

**Strength, measured from the graph alone over the 20 draws below, before any scoring.** "Neighbours
changed" is, for each district that has at least one neighbour after the relabel, the share of
those neighbours that were not its real neighbours, averaged over districts and draws. "New count" is the share of
districts whose neighbour count changed.

| panel | districts | degree classes | degclass: neighbours changed | degclass: new count | global: neighbours changed | global: new count |
|---|---|---|---|---|---|---|
| dengue | 7,165 | 19 | 0.999 | 0.000 | 0.999 | 0.847 |
| influenza_japan | 47 | 9 | 0.833 | 0.000 | 0.921 | 0.764 |
| influenza_us-regions | 10 | 5 | **0.414** | 0.000 | 0.611 | 0.785 |
| influenza_us-states | 49 | 9 | 0.864 | 0.000 | 0.908 | 0.814 |
| covid_us-states | 49 | 9 | 0.864 | 0.000 | 0.908 | 0.814 |

Measured by the dry run with the permutation seeds in section 11, identical across all five seeds of
every fold. influenza_japan and the two us-states panels each have two districts with no neighbours.
Under `degclass` they swap with each other and stay without neighbours, so the map cannot move their
forecasts at all.

**Strength gate, at 0.50.** If the deciding perturbation changes under half of a typical district's
neighbours on a panel, that panel cannot return a null: a NOT USED or INCONCLUSIVE verdict there is
reported as WEAK, "the relabel was too weak to test". A USED or HURTS verdict stands, because a weak
perturbation that still moves error is still evidence. Declared now, from the table above:
influenza_us-regions, 10 districts in five degree classes of sizes 1, 2, 2, 4 and 1, is WEAK unless
it returns USED or HURTS.

**Draws.** 20 relabels per checkpoint per perturbation, permutation seeds 7001 to 7020 (section 11),
the same list for every checkpoint and panel, `numpy.random.default_rng(seed)`. Using one list
everywhere means seed-to-seed differences are differences between trunks, not between relabels. What
it costs: the verdict is conditional on these 20 relabels. The spread over draws is in every record,
so its size is visible. A relabel that leaves every label in place is refused.

## 4. Split, scorer, metrics

Identical to the archived LDO3 records. The test fold of each held-out panel,
`b.origins(phase="test")`. The point forecast is the median quantile, inverted to counts per node,
as `train/joint.py:_test_dataset` does. Scored by the unmodified `train.loop.score_predictions`, which
calls `score.score_bundle`. Count space. Horizons 3, 5, 10, 15. Field `country_macro`, the LDO3
headline (`diagnostics/ldo3_report.py:83`), which equals `node_mean` on the four single-country panels.

RMSE and MAE decide. PCC is printed as an absolute change and never decides.

## 5. The drift gate

Before any relabel on a panel, the runner forecasts that panel on the REAL map through the archived
path, `enc(Z, A, M)`, and scores it. This reproduces numbers already on disk, so it is not new
information. Two checks, both at a tolerance of 1e-6, where a difference is `|new - archived| /
max(|archived|, 1)`:

1. Every metric the archived record holds (7 metrics by 4 horizons), on both `country_macro` and
   `node_mean`, against `results/lodo/<stem>.json`.
2. The full count-space quantile forecast against `results/lodo/<stem>__quantiles.npz`, as the maximum
   absolute difference over the maximum archived value.

Any failure stops the run and that checkpoint writes no record. Measured before writing this, on the
GPU: 0.0 on every metric of `encoder_ldo3__dengue__seed42`, and 0.0 on every quantile of
`encoder_ldo3__covid__seed42`. 1e-6 leaves room for float noise and none for a different model.

## 6. The effect and the decision rule

**Per checkpoint.** For held-out panel, horizon h and metric m (RMSE or MAE), with `real` the score on
the real map and `perm_k` the score under relabel k:

    e = (1/20) * sum over k of (perm_k - real) / real

Positive means the relabelled map forecast worse, so the real map was helping.

**Per cell** (panel by horizon by metric): the mean of e over the 5 seeds, and a two-sided 95 percent
t-interval over the seeds, `mean +- 2.7764 * sd / sqrt(5)`. It is paired by construction: the same
checkpoint is scored with and without the real map, so trunk-to-trunk variation enters only through
the spread of e.

| cell verdict | condition |
|---|---|
| USED | mean >= +2 percent AND interval lower end > 0 |
| HURTS | mean <= -2 percent AND interval upper end < 0 |
| NOT USED | the whole interval inside (-2 percent, +2 percent) |
| INCONCLUSIVE | anything else |

The four are mutually exclusive. NOT USED needs the whole interval inside the band, so "too noisy to
tell" cannot pass as "not used". A significant change under 2 percent is NOT USED when its interval
fits the band, which is deliberate: it is real and too small to matter.

**Per panel**, over its 8 deciding cells (4 horizons by RMSE and MAE):

| panel verdict | condition |
|---|---|
| USED | at some horizon BOTH RMSE and MAE are USED, and at no horizon are both HURTS |
| HURTS | at some horizon both are HURTS, and at no horizon are both USED |
| MIXED | both of the above |
| NOT USED | all 8 cells NOT USED |
| INCONCLUSIVE | anything else |
| WEAK | the strength gate in section 3 turns a NOT USED or INCONCLUSIVE into WEAK |

**Per fold.** The dengue and covid folds are their one panel. The influenza fold is its three panels
listed separately and never pooled, as everywhere else in this project (D3).

**Why 2 percent.** It sits above the whole in-disease probe range, +0.04 to +1.04 percent, so if the
transferred trunks behaved like the in-disease ones this rule would call them NOT USED or
INCONCLUSIVE, never USED. (That probe also changed the degree feature, section 1, so it is a
reference point, not a like-for-like baseline.) And it sits under half the seed-to-seed spread of the
transferred model itself: the coefficient of variation across the 5 seeds of the archived LDO3
adapted records, `country_macro`, has a median of 4.9 percent over the 20 RMSE cells and 5.4 percent
over the 40 RMSE and MAE cells, range 0.2 to 31.8 percent. A map worth less than 2 percent is worth
less than half of what changing the random seed does. What it gives up: a real effect under 2 percent
is reported as NOT USED. On dengue h15 the seed spread is 0.2 percent, so 2 percent is coarse there.
One number for every cell is a choice made to keep the rule simple and fixed in advance.

**Multiplicity** is handled by structure, not by an alpha correction: 40 deciding cells, USED needs
both metrics at one horizon, and NOT USED needs all eight cells of a panel. Extra cells make a null
harder to reach, not a win easier.

## 7. Outcomes and the sentences the paper may use

The runner's `--report` names one outcome. The sentence for that outcome is used, filled in with
panel names and horizons from the report, and nothing stronger.

**O1, null.** Every panel that passed the strength gate is NOT USED.
> Relabelling the map among districts with the same number of neighbours, at inference, changed the
> transferred model's RMSE and MAE by less than 2 percent on [panels], with every 95 percent interval
> over five seeds inside plus or minus 2 percent. On these held-out diseases the transferred trunk
> does not use which districts neighbour which. [For each WEAK panel:] On [panel] the relabel changed
> too few neighbours to test.

**O2, no evidence of use.** No panel USED or HURTS, and at least one INCONCLUSIVE.
> No held-out panel showed a rise in error of 2 percent or more when the map was relabelled. On
> [NOT USED panels] the change was bounded inside plus or minus 2 percent; on [INCONCLUSIVE panels]
> the five-seed interval was too wide to decide. [For each WEAK panel:] On [panel] the relabel
> changed too few neighbours to test.

**O3, used.** At least one panel USED, none HURTS or MIXED.
> On [USED panels], relabelling the map raised the transferred model's RMSE and MAE by at least 2
> percent at [horizons], so the transferred trunk uses the real map there. On [other panels] it did
> not ([verdict per panel]).

If every USED panel is a seen-map panel (covid_us-states, influenza_us-states), this sentence is
mandatory as well:
> Every panel where the map mattered has a map the trunk had already seen in training through another
> disease. On the panels whose map was new to the trunk, which is Ebola's situation, [their verdicts].

**O4, hurts.** At least one panel HURTS, none USED or MIXED.
> On [panels], the transferred model forecast better with the map relabelled, by at least 2 percent
> on RMSE and MAE at [horizons]: the real map hurt the transferred trunk there.

**O5, mixed.** Any panel MIXED, or USED and HURTS panels both present. No one-line summary. A table of
panel verdicts with their horizons, and the O3 and O4 sentences for the panels they apply to.

**Always appended, under every outcome:**
> This is an inference-only test of trunks that were trained on the real map, using the head fitted
> on the held-out disease. It does not show whether a trunk trained on a wrong map would do as well.

**Never said, under any outcome.** Anything about Ebola, which this does not test. "The graph is
useless" or "a wrong map would train as well", which needs a retrain. Anything about zero-shot
transfer drawn from the primary family, which uses the adapted head. Anything about meta-learned
trunks, since ANIL is not in. The in-disease +0.04 to +1.04 percent placed beside these numbers as
like-for-like without saying that it also changed the degree feature. The `global` rows or the
exploratory rows quoted as the verdict.

## 8. Scope fences

- **Inference only.** Every trunk here was trained on the real map. A null says the trained trunk
  does not lean on the map at forecast time. It does not say the map carried nothing during training;
  D2 found a dengue effect only on retrain.
- **Adapted head only** in the deciding family. The zero-shot rows are one seed and descriptive.
- **Five held-out panels from three folds.** The influenza fold's three panels share one trunk per
  seed, so they are not independent evidence.
- **Not Ebola.** Ebola's map was never seen by any trunk; three of the five panels share that
  property, and that is as close as this protocol gets.

## 9. Order of work and runtime

1. `--selfcheck`: sparse relabel equals the dense one, `degclass` keeps every district's degree, an
   identity or automorphism changes nothing, the fixed-degree forward equals the archived path on the
   real map, the cell rule returns each verdict on hand-built inputs, and the write guard refuses a
   path outside the lane. It never scores a relabelled forecast.
2. `--dry --sha256 <this file's hash>`: the drift gate on all 17 checkpoints, the strength of every
   draw, and the time of ONE relabelled forward per panel, which is discarded unscored. Writes
   `experiments/transfer_relabel/dry_run.json`.
3. The user commits this file.
4. The scored run, in the user's shell. About 95 minutes: two dry runs projected 97 and 94. Dengue
   takes about 18.5 s per relabelled forward plus about 3.2 s to score, about 15 minutes per
   checkpoint, and its six checkpoints (five primary, one exploratory) are about 89 of the minutes. The four small panels take
   under 1 s per forward. The runner skips any checkpoint whose record already exists under this
   protocol, so a stopped run restarts where it left off.
5. `--report`, which writes `experiments/transfer_relabel/summary.json` and names the outcome.
6. `diagnostics/verify_transfer_relabel.py`, which recomputes every verdict from the raw records
   without importing the runner, checks every hash stamp and the drift gate, and compares with
   `summary.json`. `--mutate` corrupts the records in memory and requires every corruption to be
   caught. `--synthetic` runs the same checks on a built record set with known verdicts.
7. The result document, verified against disk before sign-off.

## 10. After the data

This file does not change. `--report` refuses to decide unless all 15 primary records exist and
every record carries this file's sha256. The result document records this sha256 and the commit
that holds it.

## 11. Machine-readable constants

The runner and the verifier both read their thresholds from this block and nowhere else.

<!-- constants -->
```
SEEDS = 42 52 62 72 82
PRIMARY_FOLDS = dengue influenza covid
EXPLORATORY_ZS_FOLDS = covid dengue
EXPLORATORY_SEED = 42
UNSEEN_MAP_PANELS = dengue influenza_japan influenza_us-regions
PERTURBATIONS = degclass global
DECIDING_PERTURBATION = degclass
PERM_SEEDS = 7001 7002 7003 7004 7005 7006 7007 7008 7009 7010 7011 7012 7013 7014 7015 7016 7017 7018 7019 7020
HORIZONS = 3 5 10 15
DECIDING_METRICS = rmse mae
PRINTED_METRICS = pcc
MATERIALITY = 0.02
T_CRIT = 2.7764
STRENGTH_FLOOR = 0.50
DRIFT_TOL = 1e-6
```

## 12. Reproduce, in order

```
conda run --no-capture-output -n ebola-train python experiments/transfer_relabel/run.py --selfcheck
conda run --no-capture-output -n ebola-train python experiments/transfer_relabel/run.py --dry --sha256 <hash>
conda run --no-capture-output -n ebola-train python experiments/transfer_relabel/run.py --sha256 <hash>
conda run --no-capture-output -n ebola-train python experiments/transfer_relabel/run.py --report
conda run --no-capture-output -n ebola-train python diagnostics/verify_transfer_relabel.py
conda run --no-capture-output -n ebola-train python diagnostics/verify_transfer_relabel.py --mutate
conda run --no-capture-output -n ebola-train python diagnostics/verify_transfer_relabel.py --synthetic
```

# Handoff 2026-09-26: verifier for the epi bound and lambda pre-registration

**RESOLVED 2026-09-26 (later same session).** The coordinator accepted all four discrepancies and all
four wording notes and edited the protocol (still uncommitted). I re-anchored the verifier checks and
mutations that pointed at removed text, added the requested new check (us-regions p99 median
worst-ratio span 0.53 to 0.71) plus a deciding-metric-flag check, and finished the mutation run.
Final state:

- `python diagnostics/verify_epi_bound_protocol.py` -> `OK: 185 checks, 0 failures`, exit 0.
- `python diagnostics/verify_epi_bound_protocol.py --mutate` -> `mutate: 31 of 31 corruptions caught`
  (25 document, 6 input), exit 0.
- No number still fails to reconcile.

The rest of this doc is the pre-resolution record, kept for history. Sections 3 and 4 below list the
discrepancies that were fixed; they are no longer open.

---

What I am doing, where I stopped, and exactly how to pick it up. Everything below was measured from
disk today. The protocol under test is NOT yet committed, which is the point: it is about to be
frozen, so a wrong number in it has to be caught now.

---

## 1. The job

Write `diagnostics/verify_epi_bound_protocol.py`, which reads every number back OUT of
`progress/decisions/Epi_Bound_Lambda_Protocol.md` and recomputes it from disk.

Hard rule I followed: the verifier does **not** import `ablation.run_epi_ablation`. That runner is
the thing under test, so a bug shared with it would verify itself. Instead:

- I import `ablation.epi_penalty` for `observed_rates`, `probe_rates`, `GAPS`, `PAIRS` only. That is
  the penalty module, not the runner.
- I re-derive the aggregation myself: each dataset's per-gap quantile, then max / median / min across
  datasets, then the two violation fractions. `_rmax`, `_data_frac`, `_probe_cell` in the new file.
- I scrape the runner's constants out of its **source text** with regex (`_sources()`), and I
  reimplement `tag()` locally from the scraped format strings (`_local_tag`). The runner's `tag()` is
  never called.
- Every statistic (t critical value, false-call rate, paired sd, old-rule flag counts, runtimes) is
  reimplemented locally.

Independence check that came for free: my reimplemented probe reproduces
`ablation/epi_penalty.py --probe --quantile 0.99 --aggregator median --seeds 42 52 62 72 82`
row for row, all 20 rows.

## 2. State of the file

`F:\Quickgen Projects\Research Paper\Ebola-Research\diagnostics\verify_epi_bound_protocol.py`
is written and runs. Current result:

```
FAIL: 182 checks, 4 failures
```

The four failures are real discrepancies in the protocol, listed in section 3. I did not soften any
check and I did not edit the protocol.

Structure mirrors `diagnostics/verify_v2_protocol.py` as instructed: a "Disk side" block of
recompute functions, a "Document side" with `_cells` / `_eq` / `check` / `mutate`, a closure-counted
`ok(cond, msg)` inside `check()`, `--mutate` as the only flag, exit 0 or 1.

## 3. The four protocol numbers that did NOT reconcile

Do not change these without deciding first. My read on each is below, but the call is the user's.

**(a) Section 2 table, p95 median, "model intervals the hinge touches" column.**
Protocol says **7.318 %**. Disk says **7.191 %** on the same basis as the other four rows.

I traced where 7.318 comes from. The other four rows of that table are seed 42 only, which is what
`epi_penalty.py --sweep` prints at its default `--seeds 42`. The p95 median row's percentage is the
**5-seed pooled** value. Measured both ways, all five bounds:

| bound | seed 42 | pooled over 5 seeds |
|---|---|---|
| p99 max | 0.017 | 0.012 |
| p90 max | 1.279 | 1.342 |
| p99 median | 1.626 | 1.539 |
| p99 min | 2.615 | 2.583 |
| p95 median | **7.191** | **7.318** |

So four rows are one command and one row is a different command. My call: change 7.318 to 7.191 so
the column is one statistic, because a table where one cell has a different seed basis is exactly
the estimand mismatch this project already got burned by on Ebola. The alternative, relabelling the
whole column as 5-seed, means recomputing all four other cells.

**(b) Section 6, `influenza_us-states` violations at p99 median.**
Protocol says **8 to 14** of 21,168. Disk says **5 to 14**. By seed
(42, 52, 62, 72, 82): `8, 10, 5, 14, 8`. Seed 62 is 5, so the low end of the range is wrong.
Confirmed twice, once by my reimplementation and once by the released `--probe` command.
"about 0.05 percent" in the same sentence is fine: disk gives 0.0425 percent.

**(c) Section 6, the line citation.** Protocol says the old rule lives at
`run_epi_ablation.py:146`. On disk `abs(dm) < dsd` is on **line 196**. Line 146 is inside the
`load_abl` docstring. The rule text itself, "within noise unless |mean d| >= sd d", is quoted
correctly.

**(d) Section 5 step 3, the pilot cell runtime.** Protocol says the COVID seed 42 cell at
`epi_p99median_lam100` is "about 6 min". The 10 released COVID epi cells took **1.1 to 1.5 min**,
mean 1.27. Nothing on disk supports 6 min for a COVID cell. This one is a forward-looking estimate
rather than a recomputable result, so it is the softest of the four, but it is off by about 4.7x and
a reader can check it against the logs we ship.

## 4. Things that DO reconcile but whose wording is loose

I did not fail the verifier on these. They are prose, and they are worth a decision before freezing.

1. **"worst prediction only 0.79 to 1.05x the bound"** (section 6, us-regions). That range is the
   **p95 median** worst-ratio span by seed, not p99 median. At p99 median us-regions' worst ratios
   are 0.53 to 0.71. The sentence is defensible because "p95 median" is the nearest antecedent, but
   it reads as though it covers all three bounds. I check it against p95 median and additionally
   assert the p99 median worst ratio stays under 1.
2. **"`min` is driven entirely by the least, currently dengue."** True at gaps 3 and 5. At gap 2 the
   minimum is `influenza_us-regions` (0.8704) not dengue (0.9359).
3. **"Three arms adding 144 comparisons."** 144 is right under two different decompositions:
   6 arms x panels x 4 horizons x 2 deciding metrics = 128 + 16 = 144, and also
   3 arms x 4 panels x 4 horizons x 3 printed metrics = 144. The word "Three" only works under the
   second reading, and the plan has six arms. I verify 144 against the arm table using the deciding
   metrics. Suggest rewording to "the new arms".
4. **"Across 48 cells ... the three flags the p90max arm produced."** Both numbers are right, but
   only if PCC is counted. 48 = 4 panels x 4 horizons x **3 printed metrics**, and the three flags
   are 1 RMSE flag (us-regions h5, -11.541 +- 9.462) plus 2 PCC flags. On RMSE and MAE alone the
   p90max arm produced **1** flag, not 3. Since the protocol says elsewhere that PCC never decides,
   the sentence mixes a printed-grid count into a decision-rule argument. The arithmetic is fine.

## 5. Everything the verifier checks, 182 checks

Grouped by protocol section so you can see coverage.

- **Section 1.** p90max penalty span 2.2e-05 to 6.5e-04 against all 20 log cells; the
  share-of-objective range 0.005 to 0.1 percent is checked for reachability from the measured
  penalties given the stated pinball order 0.1 to 1 (the released logs carry no pinball column, so
  this is the strongest available check); the 10 identically-zero p99max cells and that they are
  exactly the two US influenza panels; COVID 1.5x to 3x above every other panel from
  `ablation/epi_rmax.json` per_dataset (disk span 1.501 to 2.996, so the stated range is tight);
  `epi_rmax.json` reproduces from the raw rates at p99 max; `max` selects COVID at all three gaps.
- **Section 2.** All five table rows: three r_max values, "data >bound", "model >bound". The chosen
  bound prose (1.0397 / 0.7838 / 0.5666). "Looser at gaps 2 and 3, tighter at gap 5". "Fires more
  than p90max and calls fewer real transitions implausible". `PAIRS` as written in the doc against
  the module, and the gap-5-twice / gap-2-once count. The 12.4 percent p95 median figure.
- **Section 3.** All six arm rows: tag string against my local reimplementation of the tag rule,
  panels column, cell count = panels x 5 seeds, and the three r_max values. Total 90. Seeds against
  `train/loop.py:41` scraped as text. Dengue cost 12.7 h / 96 percent against the runner's docstring.
  Japan p99max penalty order 1e-7 and 1e-5 at lambda 100. 30 guaranteed nulls = 2 arms x 3 panels x
  5 seeds, and the 2.9 h against my runtime model. 40 released lambda-1 records, 20 per bound,
  actually on disk.
- **Section 4.** Reference model `encoder` and field `node_mean` against the runner source; all 20
  reference records exist; all eight `epi_*` record fields are both named in the doc and written by
  the runner; and a released lambda-1 record carries none of them, which is what makes "recorded
  their bound only in the filename" true.
- **Section 5.** 3 of 3 planted bugs against the count of mutants in the runner source. Pilot tag is
  the p99 median lambda 100 arm. 89 = 90 minus the pilot. 6 h grid runtime. 1.0 to 26.7 min per-cell
  range. New records route to `experiments/epi_bound_lambda/single/` via `results_paths.rpath`. The
  runner never writes `results/ebola`. The runner freezes this exact filename.
- **Section 6.** t(0.975, 4) = 2.7764 from `scipy.stats.t.ppf`. The line citation. The verbatim old
  rule and the sqrt(n) equivalence. p = 0.09 as 2*(1 - t.cdf(sqrt(5), 4)). 48 cells, 4.3 expected
  flags, 3 observed flags recomputed from records under the old rule. 144 comparisons and 13
  spurious flags. The 1 percent gate against the runner's `v < 0.01`. The share formula against the
  runner's source line. The restated share range matching section 1's. The Japan/COVID versus US
  asymmetry in p90max penalties. Per-panel probe: us-regions 0 of 9,400 every seed at p99 median and
  at p99 min, worst ratios, us-states 8-to-14 of 21,168 and its rate, Japan 4,445 of 4,596 = 96.7
  percent, us-regions p95 median worst-seed count, and that the probe really says zero for
  us-regions at p90max where the log recorded 2.3e-05.
- **Section 11.** Every flag in the reproduce block exists in the two scripts' argparse.

**Skipped on purpose:** the 12.7 h dengue figure is not recomputed, there is no dengue epi record. I
only cross-check that the protocol's number matches the runner's own docstring.

## 6. Runtime model, since two protocol numbers depend on it

`_runtime()` takes the mean measured minutes per cell **per panel** over both released logs, then
applies the protocol's arm shapes. Per-panel matters because us-states averages 7.89 min and COVID
1.27.

```
per panel (mean over both logs, 10 cells each):
  influenza_japan 4.18   influenza_us-regions 4.36   influenza_us-states 7.89   covid_us-states 1.27
one 4-panel arm (20 cells) 88.5 min
the 90-cell grid            4 x 88.5 + 2 x 6.35 = 366.7 min = 6.11 h   -> protocol "about 6 h" OK
two full p99max arms        2 x 88.5 = 177.0 min = 2.95 h              -> protocol "2.9 h" OK
the 30 flu cells alone      2 x 82.1 = 164.3 min = 2.74 h
per-cell range over both logs 1.0 to 26.7 min                          -> protocol OK
```

Note the protocol attributes the 1.0 to 26.7 min range to `results/reports/epi_p90max.log` alone.
That log's own range is 1.1 to 7.6 min. 1.0 and 26.7 come from `epi_p99max.log`. So the range is
correct over the **two** logs and misattributed to one. I check it over both, per the brief.

## 7. Where I stopped

`--mutate` was mid-run when I stopped. Progress so far:

- 22 document mutations defined, all digit transpositions or count changes. **The first 21 all
  report "caught".** I had not yet seen the last one print, nor the six input mutations.
- 6 input mutations defined: COVID gap-2 rates x1.20, us-regions p99 median violations 0 to 3,
  p90max us-regions seed52 penalty x10, false-call rate forced to 0.05, runner gate 0.05, runner
  default lambda 2.0. None have been observed yet.
- Anchor assertions are in, `assert a in text, f"mutation anchor missing for {name}"`, and they
  already earned their keep: three anchors failed because the protocol wraps lines. The trap is that
  `check()` matches against a whitespace-flattened copy while `mutate()` replaces in the **raw**
  text, so any anchor spanning a newline fails. Fixed anchors so far: `under 0.01 has its null`,
  `0.09 at five seeds`, `COVID sits 1.5x to 3x`.

**Next step, one command:**

```
C:\Users\Administrator\miniconda3\Scripts\conda.exe run --no-capture-output -n ebola-train `
  python diagnostics/verify_epi_bound_protocol.py --mutate
```

Expect the four failures from section 3 first, then 28 mutation lines, then
`mutate: N of 28 corruptions caught`. If any anchor assertion fires, shorten that anchor so it sits
on one line of the protocol. If any mutation reports MISSED, the check for that number is too loose
and needs tightening, not the mutation weakening.

The plain run is:

```
C:\Users\Administrator\miniconda3\Scripts\conda.exe run --no-capture-output -n ebola-train `
  python diagnostics/verify_epi_bound_protocol.py
```

Bare `conda` is not on PATH in the harness shells, hence the full path. Both commands take about a
minute; the mutate run rebuilds the disk dict 7 times.

## 8. Open decisions for the user

1. The four discrepancies in section 3. Fix the protocol before committing, or accept and document.
   The protocol is a pre-registration, so it must be right **before** the hash is stamped into any
   record.
2. The four loose wordings in section 4.
3. Once the numbers are settled, the protocol gets committed and only then can
   `ablation/run_epi_ablation.py` train, because `protocol_sha()` refuses while the file has
   uncommitted edits.
4. After any protocol edit, rerun both commands in section 7. The verifier is cheap.

## 9. Scratch files, safe to delete

Under the session scratchpad, not in the repo:
`explore.py`, `e2.py`, `e3.py` in
`C:\Users\ADMINI~1\AppData\Local\Temp\claude\f--Quickgen-Projects-Research-Paper-Ebola-Research\0ff0b3ef-ee34-4d9a-a4d3-7235cd80085d\scratchpad`.
They computed the five-bound sweep, the per-seed pooled percentages in section 3(a), and the
old-rule flag counts in section 4(4). All three are now folded into the verifier itself, so nothing
is lost if they go.

# Handoff 2026-09-26: extend the pure-TCN probe to influenza_japan

What I am doing, where I stopped, and how to pick it up. Everything below was measured from disk
between the evening of 2026-09-25 and the early hours of 2026-09-26. Stopped at the user's request,
in the middle of verification.

## UPDATE 2026-09-26 02:25: verification COMPLETE, sections 2, 5 and 6 below are now history

All six steps passed. Nothing left to verify; the user's run command (section 9) is ready.

- 1.498 question settled: both graphs binary, max degree 9 incl. self-loop, but different graphs
  (covid N=49 sum(A)=206, japan N=47 sum(A)=172). Selfcheck now prints that line per panel.
- Step 2: Japan mutation fires: `influenza_japan: pure-TCN output moved when the graph was emptied
  (max diff 1.5): the arm still sees A`.
- Step 3: refuse path mutation-tested (float +1e-9, verdict flip, missing block: all REFUSED, clean
  blocks written). `--report` rewrote the summary into keyed form, new sha256 `a4251b80...`. COVID:
  244 checks (144 floats exact, 24 n, 24 verdicts, 24 seed lists, 6 tallies, 3 meta, 19 brief
  values), 0 mismatches; planted breaks caught. Printed COVID block identical to pre-edit except
  CEILING wording.
- Step 4: Japan 1-epoch smoke 6.2 s, 3-epoch 9.6 s, so about 1.7 s per epoch plus 4.5 s fixed;
  worst case 2.3 min per seed, about 10 to 12 min for five. Nothing written. Meta gate_mode off,
  ltr off, encoder_version v1_pure_tcn.
- Step 5: default run trains influenza_japan 42 52 62 72 82, skips all five covid seeds.
- Step 6: five COVID seed sha256 unchanged. `train/loop.py` diff is entirely under
  `if epi and train_mode:`, so the `epi=None` path is unchanged in behaviour.

---

## 1. The job

Extend `experiments/pure_tcn.py` (untracked, written 2026-09-25) so it also runs `influenza_japan`.
The pure-TCN arm is the gate-off trunk with the degree term (LTR, a learned feature built from each
node's number of neighbours) swapped for a constant zero, so the model sees no graph at all.

Hard constraints from the brief:

- Edit ONLY `experiments/pure_tcn.py`. This handoff doc exists because the user asked for it.
- Do NOT run the 5-seed Japan job. The user runs it in their own shell.
- Do NOT commit.
- Do NOT add a fresh gate-off arm. That is the next step, separately.
- The 5 COVID seed files must not be retrained or overwritten. Hashes in section 4.

## 2. Where I stopped

**Code edit: done.** The file compiles, and `--selfcheck` passes on both panels (section 3).

**Verification: step 1 of 6 done.** Steps 2 to 6 are in section 6 with exact commands.

**Interrupted mid-check.** I was confirming why the gate-off control shifts by exactly 1.498 on BOTH
COVID and Japan (section 5). Not resolved yet. Resolve it before trusting the Japan control.

Nothing has been handed to the user yet. Do not hand over the run command until steps 2 to 6 pass.

## 3. What changed in `experiments/pure_tcn.py`

272 lines before, 317 after; `git diff --no-index` says 139 insertions, 94 deletions, most of it
re-indentation of selfcheck checks 1 and 2 and of the training loop in `main()`.

1. `DATASET` constant replaced by `DATASETS = ["covid_us-states", "influenza_japan"]` and a
   `--datasets` flag (`nargs="+"`, `choices=bundles.DEV_BUNDLE_NAMES`, default `DATASETS`). The old
   ponytail comment "Add a flag when a second panel is wanted" now reads "the two panels run so far.
   --datasets takes any DEV bundle; the others are unrun here."
2. `PURE` and `COMPARISONS` became functions `pure_glob(ds)` and `comparisons(ds)`.
   `seed_file(ds, s)` and `pending(ds, seeds, force, out)` take the dataset. The skip line now names
   the dataset.
3. `report()` split in three:
   - `report_one(ds)` is the old body, prints one block, returns the summary block or `None`. On no
     records it prints `no pure-TCN records yet` for that dataset only.
   - `report(datasets)` loops, then calls `write_summary` only if at least one block exists.
   - `write_summary(blocks, datasets)` writes ONE file keyed by dataset. It reads the current file,
     converts the old flat COVID-only form (`"dataset"` at top level) to keyed form, and for every
     dataset in this run that is already in the file compares `KEEP = (model, field, seeds, tallies,
     cells)` as canonical JSON (`json.dumps(sort_keys=True)`, exact on floats and NaN). Any mismatch
     prints `REFUSED to write ...` naming the keys and writes nothing. Panels not in this run are
     carried over untouched.
4. CEILING text is dataset-neutral: `at most {n} paired seeds on {ds}, and seed sd is the only noise
   scale, so |mean| < sd is a weak rule.` The old text said "COVID's seed sd is large".
5. `selfcheck(datasets)`: checks 1 (graph independence plus its negative control) and 2 (output
   equals `enc.tcn(Z)`) run inside a loop over every dataset. Checks 3 to 5 unchanged in logic;
   check 4's path assert covers every dataset's seed file.
6. `main()` loops over `a.datasets` for training; asserts and prints name the dataset.
7. Docstring: question covers both panels, comparisons use `{D}`, notes the default datasets and
   that the plain run trains Japan only while COVID seeds exist, adds the runtime (section 7), and
   now cites the TRACKED `results/reports/gate_ablation.log` instead of the gitignored
   `Reports/gate_ablation.log` (byte-identical, checked with `cmp`).

No em dashes, no non-ASCII. The two lines over 110 characters are untouched check 3 code.

### Decision I made in `write_summary`, flag it to the user

The brief asked for the reproduce-or-refuse check on COVID. I applied it to every panel already in
the file, not just COVID.

```
What you get: once Japan is in the summary, a later --report cannot silently change Japan either.
What it costs: a deliberate --force retrain makes the next write refuse. The user must move the old
  summary aside and rerun --report. One manual step, and the message says so.
My call: keep it, because a recorded result should only change on purpose.
```

Second, smaller choice: panels not in `--datasets` are carried over from the old file.
What you get: `--report --datasets influenza_japan` does not drop COVID from the one summary file.
What it costs: a carried block was not regenerated on that run; it was verified when it was written.

## 4. Protected files, baseline hashes (sha256), taken before any edit

```
9a72daddaccd33ae6615e4ecfe37f82007a3def66015ee6b2b7158361bb9a4fb  experiments/pure_tcn__covid_us-states__seed42.json
82ed13eb478f5b0e5059fc6f543e7abd2414a6edeefb9eeb450384db7eb1a7c9  experiments/pure_tcn__covid_us-states__seed52.json
ce8878c0951729dbf72a20e5ae5de65cb4b6b671cd52439aa0c32b134037cb7a  experiments/pure_tcn__covid_us-states__seed62.json
7cbd46532e02eae2e06285b1345317fff7b7006ecc16e7ca82c15fa3a677f519  experiments/pure_tcn__covid_us-states__seed72.json
3852a86491c180dc717352c9e10c99c02dc9a2a9344c4ec5a16cb09c3ff37506  experiments/pure_tcn__covid_us-states__seed82.json
1496599fa8663d3b50db03dbcb5c0dd336c1575643b4655dbbe0edcfdfbf4385  experiments/pure_tcn__summary.json   (old flat form, COVID only)
```

All six re-hashed after the edit and the selfcheck: unchanged. The summary has NOT been rewritten
yet; the first `--report` will rewrite it into keyed form (expected, the brief allows it).

Pre-edit copies, in the session scratchpad, which may be cleaned overnight:
`C:\Users\ADMINI~1\AppData\Local\Temp\claude\f--Quickgen-Projects-Research-Paper-Ebola-Research\efcdead1-8349-4826-912c-01fdf5ecbaa8\scratchpad\`
holds `pure_tcn_before.py`, `summary_before.json`, `report_before.txt`. If they are gone, the summary
baseline is still recoverable: it is the file with hash `1496599f...` above, so **copy
`experiments/pure_tcn__summary.json` somewhere safe before running `--report`**.

Before editing, I captured the OLD script's report with writing disabled. It matches every number
the brief quotes, e.g. vs learned RMSE h3 `5565.755 +-915.034 | 5405.942 +-213.968 | -159.813
+-1023.306 within noise`; vs gate-off RMSE h5 `7835.384 +-640.431 | 8697.706 +-701.117 | +862.322
+-349.181 DEGREE HELPS`; tallies vs learned 1 / 0 / 11, vs gate-off 2 / 0 / 10.

## 5. Selfcheck output (verification step 1), and the open question

```
ok covid_us-states: pure-TCN output identical under the real A and an all-zero A; control: gate-off moves by up to 1.498
ok covid_us-states: pure-TCN output equals enc.tcn(Z) (atol 1e-6), LTR holds no parameters; control: gate-off output differs from its tcn(Z)
ok influenza_japan: pure-TCN output identical under the real A and an all-zero A; control: gate-off moves by up to 1.498
ok influenza_japan: pure-TCN output equals enc.tcn(Z) (atol 1e-6), LTR holds no parameters; control: gate-off output differs from its tcn(Z)
ok factory draws no CPU RNG: RNG state, TCN, spatial and Adapter init match gate-off exactly; control: one rebuilt LTR shifts both
  covid_us-states seed 42: C:\Users\ADMINI~1\AppData\Local\Temp\tmpo07nz2gz\pure_tcn__covid_us-states__seed42.json exists, skipped (--force retrains it)
ok pure_tcn__covid_us-states__seed42.json and pure_tcn__summary.json resolve inside experiments/, existing seed skipped, --force retrains; control: _dump refuses ../
ok _seed_from('pure_tcn__covid_us-states__seed42.json') == 42; verdict signs right for rmse and pcc; control: seedless name and |mean| < sd both refused
selfcheck passed
```

**Open: the control shift is 1.498 on both panels.** LTR is `Linear(1, 64)` on `log1p(deg)`
(`models/spatial.py:60-61`), and with the gate off nothing else sees the graph, so the shift is
`max|w| * (log1p(deg_max) - log1p(1))`, same seed-0 weights on both panels. It matches only if both
panels have the same maximum self-loop-inclusive degree. My guess is 9 on both (Missouri and
Tennessee border 8 states, Nagano borders 8 prefectures). **Unverified.** It matters because an
identical number is also what you would see if the Japan loop were accidentally reading COVID's
graph. The command I was running when stopped:

```
PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE "/c/Users/Administrator/miniconda3/envs/ebola-train/python.exe" -c "
import torch, numpy as np, bundles
from models import sparse_from_dense_np, SharedEncoder
from models.encoder import normalise_adj
for ds in ('covid_us-states','influenza_japan'):
    b = bundles.load(ds)
    _, deg = normalise_adj(sparse_from_dense_np(b.A_geo))
    _, d0 = normalise_adj(sparse_from_dense_np(np.zeros_like(b.A_geo)))
    print(ds, 'N', b.A_geo.shape[0], 'deg max', float(deg.max()), 'deg min', float(deg.min()),
          'empty-A deg', torch.unique(d0).tolist(), 'A_geo values', np.unique(b.A_geo)[:5].tolist())
torch.manual_seed(0); e = SharedEncoder(gate_mode='off')
w = e.ltr.lin.weight.detach().abs().max().item()
print('max|w| seed 0', round(w, 4), 'predicted shift', round(w * (np.log1p(9) - np.log1p(1)), 4))
"
```

Pass condition: N is 49 and 47, both `deg max` are 9, predicted shift is 1.498. If `deg max`
differs between panels, the loop is not reading the right graph and that is a bug to fix first.
(`normalise_adj` import path is from `models/encoder.py:57`; confirm it is importable from there.)

## 6. Remaining verification, steps 2 to 6

All from the repo root in bash, with `PY="/c/Users/Administrator/miniconda3/envs/ebola-train/python.exe"`
and `PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE`.

**Step 2, mutation on Japan.** In memory only, make the factory keep the real LTR, run selfcheck on
Japan ALONE (with both panels, COVID fires first and Japan is never reached):

```
$PY -c "
import experiments.pure_tcn as P
from models import SharedEncoder
P.factory = lambda b, g, d: (SharedEncoder(gate_mode='off').to(d), dict(encoder_version='v1_pure_tcn', ltr='off'))
P.selfcheck(['influenza_japan'])
"
```

Expected: `AssertionError: influenza_japan: pure-TCN output moved when the graph was emptied (max
diff ...)`. Report the message.

**Step 3, report and COVID reproduction.** First copy the summary to the scratchpad. Then:

1. Mutation-test the refuse path in memory, nothing written: build blocks with `P.report_one`,
   nudge one COVID `d_mean` by 1e-9, replace `P._dump` with a function that raises, call
   `P.write_summary(blocks, P.DATASETS)`. Must print `REFUSED` and never call `_dump`. Then the
   unmutated blocks must reach `_dump`.
2. Run `$PY -m experiments.pure_tcn --report`. Expect the COVID block, Japan printing `no pure-TCN
   records yet`, and `wrote experiments\pure_tcn__summary.json: ['covid_us-states'], regenerated
   ['covid_us-states']`.
3. Programmatic check, not by eye: load the saved old summary (flat) and the new one
   (`["covid_us-states"]`), compare model, field, seeds, tallies, and every cell's ref_mean, ref_sd,
   pure_mean, pure_sd, d_mean, d_sd, n, verdict. 24 cells x 6 floats = 144 numbers plus 24 n, 24
   verdicts, 6 tally counts. Also check the brief's quoted values (RMSE h3 vs learned, RMSE h5 vs
   gate-off, both tallies) at 3 decimals against the new file. Report count checked and mismatches.
4. Diff the new printed COVID block against `report_before.txt`. Only the CEILING line and the
   `wrote` line may differ.

**Step 4, Japan smoke, nothing written.**

```
$PY -c "
import time, json, train.loop as L, experiments.pure_tcn as P
t = time.time()
recs, *_ = L.train_one('influenza_japan', 42, epochs=1, gate_mode='off', encoder_factory=P.factory, verbose=True, gate_read=False)
print('wall', round(time.time() - t, 1), 's,', len(recs), 'records')
print({k: v for k, v in recs[0].items() if not isinstance(v, (list, dict))})
"
```

Report wall time and meta (expect `gate_mode='off'`, `ltr='off'`, `encoder_version='v1_pure_tcn'`).
Afterwards confirm with `git status --short` and `ls -t experiments/ results/single/ | head` that
nothing was written.

**Step 5, dry print of the default run.**

```
$PY -c "import experiments.pure_tcn as P, train.loop as L; [print(ds, 'to train:', P.pending(ds, list(L.SEEDS), False)) for ds in P.DATASETS]"
```

Expected: COVID prints five `exists, skipped` lines and `[]`; Japan prints `[42, 52, 62, 72, 82]`.

**Step 6, integrity.** Re-hash the five COVID seed files against section 4. Run
`git status --short`. See section 8 for what it will show that is not mine.

## 7. Runtime estimate for the user's Japan run

Japan gate-off seeds took 2.4, 1.9, 1.9, 1.6, 2.7 min, 10.5 min for five
(`results/reports/gate_ablation.log:4-8`). The pure-TCN arm uses the same trainer, epochs and
patience, so expect roughly 10 min, but early stopping makes the epoch count differ per seed, so
this is an estimate, not a measurement. Refine it with the step 4 smoke (one epoch time times the
epoch counts in `results/reports/ceilings.log`, which runs 60 to 72 epochs per Japan seed).

## 8. Things on disk that contradict the brief

1. **`train/loop.py` is modified and it is not mine.** Changed 2026-09-25 21:01:58, after the COVID
   pure-TCN run finished (seed files 20:46 to 20:48). The diff adds a third slot to `epi["_seen"]`
   (pinball loss, for an "inertness gate") at `train/loop.py:220-227`. It belongs to the parallel
   epi-bound session (`progress/Session_Handoff_2026-09-26_Epi_Bound_Protocol_Verifier.md`).
   It cannot affect this arm: `train_one` defaults `epi=None` (`train/loop.py:157`) and
   `pure_tcn.py` never passes `epi`. Consequence to state: the Japan seeds will train on a
   `loop.py` that differs from the one the COVID seeds used, in that inert branch only. The brief's
   expected `git status` did not list it. Do not touch or revert it.
2. **Three more untracked files from that session**: `progress/decisions/Epi_Bound_Lambda_Protocol.md`,
   `diagnostics/verify_epi_bound_protocol.py` and the handoff above. Not strays of mine.
3. **This handoff doc** is new and untracked. The brief said edit only `pure_tcn.py`; the user
   asked for this doc directly.
4. The old docstring cited `Reports/gate_ablation.log`, which is gitignored (`.gitignore:26 *.log`).
   The tracked copy `results/reports/gate_ablation.log` is byte-identical and is now cited.

## 9. The command for the user, once steps 2 to 6 pass

PowerShell, from the repo root:

```
conda run --no-capture-output -n ebola-train python -m experiments.pure_tcn
```

It runs the selfcheck on both panels, skips the five COVID seeds, trains five Japan seeds (about
10 min), then prints both blocks and writes the keyed summary, refusing if COVID no longer
reproduces.

## 10. Goal this moves

Exploratory only, nothing scored. It feeds the G2 / graph-mechanism story ("the graph does not
help error"): the gate-off ablation still kept the degree term, and this arm is the first test with
no graph input at all. COVID result so far: vs learned 1 / 0 / 11, vs gate-off 2 / 0 / 10, degree
term helps at h5 RMSE and MAE only. Japan is the second panel. Still missing after Japan: the fresh
gate-off arm (next step, separately), and any per-node or quantile output (records only today).

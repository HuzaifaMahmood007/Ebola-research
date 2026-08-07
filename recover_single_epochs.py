"""Epochs-used for the 20 single-disease runs whose training log is gone (client task 2 follow-up).

READ THIS BEFORE QUOTING THE OUTPUT. This script does NOT recover the historical epoch counts.
That information died with the logs and no script can bring it back. `train/loop.py` carries 145
uncommitted insertions since the 20 dengue/influenza ceilings were trained (the bias-correction arm,
the gate change, the prediction-pipeline restructure), and those files contain only `encoder`
records while covid -- retrained 2026-08-03 on the current code -- contains `encoder` AND
`encoder_mc`. So the five ceilings in the comparison table were not all produced by the same
trainer, and re-running today measures TODAY's trainer.

What it does answer, both worth having:
  1. Does the current trainer early-stop on dengue and the three influenza panels, or run its
     80-epoch budget? Covid went 5-for-5 early (24-33 epochs). If the other four behave the same,
     the single-disease ceiling is handicapped exactly as the transfer arm is, and the deficits in
     the transfer table partly cancel. That is the paragraph the client asked for.
  2. Does the current trainer REPRODUCE the published ceiling numbers? If it does not, the ceiling
     every transfer comparison is measured against cannot be regenerated from the code in the repo,
     which is a bigger problem than the epoch question and needs to surface now rather than in
     review.

Nothing is written to results/single. `train_one` returns its records and writes nothing; the only
caller that writes is `run_dataset`/`main`, which this script never touches. That claim is CHECKED,
not asserted: every file under results/ is fingerprinted (mtime + size) before and after, and a
single byte of drift aborts the run.

Epoch lines go to results/reports/single_recovery.log in train_one's own print format, so
`python epoch_budget_audit.py` picks them up with no changes.

  python recover_single_epochs.py                       # all 20, cheap datasets first, resumable
  python recover_single_epochs.py --dataset dengue      # one dataset
  python recover_single_epochs.py --selfcheck           # no GPU needed
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Match the runs being reproduced. run_overnight.py pins the same value; setting it here explicitly
# rather than inheriting it from an import side-effect, because a silent LOCF flip would change the
# inputs and make every comparison below meaningless.
os.environ.setdefault("LOCF_INPUT", "0")

import argparse
import json
import sys
import time
from pathlib import Path

LOG = Path("results/reports/single_recovery.log")
RESULTS = Path("results")
# cheapest first: 10, 47 and 49 nodes take minutes; dengue is 7,165 nodes and ~2 h for five seeds.
# An abort partway therefore still leaves 15 of the 20 answered.
ORDER = ("influenza_us-regions", "influenza_japan", "influenza_us-states", "dengue")
SEEDS = (42, 52, 62, 72, 82)
HEADLINE = {"dengue": "country_macro"}          # else node_mean; mirrors train/lodo.py::_field
EPOCH_BUDGET = 80
REPRO_TOL = 1e-6                                # relative; anything above this is a real difference


class _Tee:
    """stdout -> console AND the log, so train_one's own epoch prints land in a file the audit reads."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.out = sys.__stdout__

    def write(self, s):
        self.out.write(s); self.out.flush()
        self.f.write(s)

    def flush(self):
        self.out.flush(); self.f.flush()


def fingerprint(root=RESULTS):
    """Every artifact under results/, by mtime and size. The no-write guarantee is checked against
    this, not trusted -- these are the ceiling files every transfer number is divided by."""
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in Path(root).rglob("*") if p.is_file()}


def already_done(ds, seed, log=LOG):
    """Resume: a (dataset, seed) that already has epoch lines in the log is skipped. A 2 h dengue
    leg should not restart from zero because the machine hiccuped on seed 4."""
    if not log.exists():
        return False
    return f"{ds} seed{seed} ep" in log.read_text(encoding="utf-8", errors="replace")


def compare_to_published(ds, seed, recs):
    """Worst relative disagreement between the re-run and the published ceiling, on the headline
    field, over the `encoder` arm only.

    `encoder` only because covid's file holds encoder AND encoder_mc under the same (horizon, metric)
    keys -- reading it naively returns whichever appears last, which is the collision that
    manufactured three false 'transfer helps' results on 2026-08-05. Cells whose published value is
    ~0 are skipped: pcc crosses zero and a relative error there is noise amplified, not signal.

    Returns (worst_pct, worst_key, n_compared) or None when there is no published file.
    """
    p = RESULTS / "single" / f"encoder__{ds}__seed{seed}.json"
    if not p.exists():
        return None
    pub = {(r["horizon"], r["metric"]): r
           for r in json.loads(p.read_text()) if r["model"] == "encoder"}
    fld = HEADLINE.get(ds, "node_mean")
    worst, worst_key, n = 0.0, None, 0
    for r in recs:
        if r["model"] != "encoder":
            continue
        o = pub.get((r["horizon"], r["metric"]))
        if o is None:
            continue
        a, b = o[fld], r[fld]
        if a is None or b is None or a != a or b != b or abs(a) < 1e-9:
            continue                                    # missing, NaN, or a near-zero denominator
        n += 1
        d = abs(b - a) / abs(a) * 100
        if d > worst:
            worst, worst_key = d, (r["horizon"], r["metric"])
    return worst, worst_key, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=ORDER, help="one dataset instead of all four")
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCH_BUDGET,
                    help="must match the published run's budget or the %% figures are meaningless")
    ap.add_argument("--force", action="store_true", help="redo runs already in the log")
    a = ap.parse_args()

    todo = [(d, s) for d in ([a.dataset] if a.dataset else ORDER)
            for s in ([a.seed] if a.seed else SEEDS)
            if a.force or not already_done(d, s)]
    if not todo:
        print("nothing to do; every requested run is already in the log (use --force to redo)")
        return

    before = fingerprint()
    sys.stdout = _Tee(LOG)
    print("=" * 96)
    print(f"SINGLE-DISEASE EPOCH RECOVERY  {len(todo)} run(s)  budget={a.epochs} epochs")
    print("NOTE: this measures TODAY's trainer. train/loop.py has changed since these ceilings were")
    print("      trained, so a non-zero repro delta below is expected and is itself a finding.")
    print("=" * 96)

    from train.loop import train_one              # imported late: argparse errors should not pay for torch

    rows, t_all = [], time.time()
    for i, (ds, seed) in enumerate(todo, 1):
        print(f"\n----- [{i}/{len(todo)}] {ds} seed{seed} -----")
        t0 = time.time()
        recs, _, _, _ = train_one(ds, seed, epochs=a.epochs, verbose=True)
        rows.append((ds, seed, compare_to_published(ds, seed, recs), time.time() - t0))

    print("\n" + "=" * 96)
    print(f"{'dataset':<24} {'seed':>5} | {'worst repro delta':>18} | {'at':>14} | {'cells':>5} | mins")
    print("=" * 96)
    for ds, seed, cmp_, dt in rows:
        if cmp_ is None:
            print(f"{ds:<24} {seed:>5} | {'no published file':>18} | {'--':>14} | {'--':>5} | {dt/60:5.1f}")
            continue
        worst, key, n = cmp_
        tag = "exact" if worst <= REPRO_TOL * 100 else f"{worst:.3f}%"
        print(f"{ds:<24} {seed:>5} | {tag:>18} | "
              f"{(f'h{key[0]} {key[1]}' if key else '--'):>14} | {n:>5} | {dt/60:5.1f}")
    print(f"\ntotal {(time.time()-t_all)/60:.1f} min")
    print("epoch counts -> python epoch_budget_audit.py")

    sys.stdout = sys.__stdout__
    after = fingerprint()
    changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k)
               and not k.endswith(("single_recovery.log",))]
    if changed:
        raise SystemExit("ABORT: this script must not modify results/. Changed:\n  "
                         + "\n  ".join(sorted(changed)[:20]))
    print(f"ok  results/ untouched ({len(before)} artifacts fingerprinted, none changed)")


def _selfcheck():
    """The repro comparison actually detects a difference, and the encoder_mc collision cannot fool
    it. Both arms matter: a comparator that always returns 'exact' would pass every run silently and
    is exactly how the 2026-08-05 false positives survived."""
    ds, seed = "dengue", 42
    p = RESULTS / "single" / f"encoder__{ds}__seed{seed}.json"
    if not p.exists():
        raise SystemExit(f"selfcheck needs {p}")
    pub = [r for r in json.loads(p.read_text()) if r["model"] == "encoder"]

    identical = compare_to_published(ds, seed, pub)
    assert identical is not None and identical[0] == 0.0, f"identical input must compare exact: {identical}"
    assert identical[2] > 0, "nothing was actually compared -- the check is void"

    # mutation: perturb ONE cell by 5% and require it to be caught, at the right key
    import copy
    mutated = copy.deepcopy(pub)
    fld = HEADLINE.get(ds, "node_mean")
    target = next(r for r in mutated if r[fld] not in (None, 0) and r[fld] == r[fld])
    target[fld] = target[fld] * 1.05
    worst, key, _ = compare_to_published(ds, seed, mutated)
    assert abs(worst - 5.0) < 1e-6, f"5% perturbation reported as {worst}"
    assert key == (target["horizon"], target["metric"]), f"blamed the wrong cell: {key}"

    # an encoder_mc record with a wildly wrong value must be IGNORED, not compared
    poisoned = copy.deepcopy(pub)
    for r in poisoned:
        r2 = dict(r); r2["model"] = "encoder_mc"; r2[fld] = 1e9
        poisoned.append(r2)
        break
    assert compare_to_published(ds, seed, poisoned)[0] == 0.0, \
        "encoder_mc leaked into the comparison -- this is the collision that faked 3 results"

    print("ok  repro comparison: exact on identical input, catches a 5% perturbation at the right "
          "cell, and ignores the encoder_mc arm")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

"""The two remaining full-budget LDO3 folds, back to back, then the comparison. (D17 -> all 3 folds)

    conda run --no-capture-output -n ebola-train python run_ldo3full.py
    conda run --no-capture-output -n ebola-train python run_ldo3full.py --dry        # plan only
    conda run --no-capture-output -n ebola-train python run_ldo3full.py --selfcheck  # logic, seconds

WHAT THIS SETTLES. D17 disabled the trunk early stop on ONE fold (influenza, seed 42) and found the
scored records bit-identical to the truncated run: the extra 72,000 steps contained nothing better.
The other 14 runs are inferred from that. These two folds close the inference across FOLDS at seed
42. They do not close it across seeds, and the write-up must keep saying so.

ORDER, AND WHY. Dengue first: it is the fold the client's fold-structure objection rests on, so it
is the one worth having if the box dies overnight. Covid second, and it is the more punishing test
-- dengue plus all three influenza panels sit in its trunk, the largest training set of the three
folds and therefore the one with the most room to keep improving past step 7,000.

THE TWO FLAGS ARE NOT OPTIONAL. `--trunk-patience 999` is the whole experiment (without it the stop
fires at 12 checks and this measures nothing new), and `--prefix encoder_ldo3full` is what keeps it
from overwriting the truncated artifacts LDO3_Results.md and verify_ldo3_doc.py read. train/lodo.py
already refuses that exact combination under the default prefix; _selfcheck pins it here too, so a
edit that drops either flag fails in seconds rather than after seven hours.

RESUME KEYS ON THE CHECKPOINT, NOT A FLAG. `{prefix}__{fold}__seed{seed}__ckpt.pt` is the last write
in run_ldo3_fold, so it exists only if that fold ran to completion. The per-bundle .json files are
written DURING the loop and a fold killed halfway leaves some behind; keying on those would skip a
half-finished fold and report the queue clean.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import sys
from pathlib import Path

from run_queue import LOGDIR, run, say as _say, wait_for_gpu

QLOG = LOGDIR / "ldo3full_queue.log"
CKPT_DIR = Path("results/lodo")
FOLDS = ("dengue", "covid")        # influenza is D17, already on disk
SEED = 42
PREFIX = "encoder_ldo3full"
TRUNK_STEPS = 91000
PATIENCE = 999                     # x1000 steps per val check, so >= TRUNK_STEPS disables the stop


def say(msg):
    _say(msg, log=QLOG)


def done(fold, seed=SEED, ckpt_dir=CKPT_DIR):
    return (ckpt_dir / f"{PREFIX}__{fold}__seed{seed}__ckpt.pt").exists()


def fold_cmd(fold, seed=SEED):
    return ["-m", "train.lodo", "--ldo3", fold, "--seed", str(seed),
            "--trunk-steps", str(TRUNK_STEPS), "--trunk-patience", str(PATIENCE),
            "--prefix", PREFIX]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=list(FOLDS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="re-run a fold whose checkpoint exists")
    ap.add_argument("--keep-going", action="store_true", help="run fold 2 even if fold 1 fails")
    ap.add_argument("--skip-wait", action="store_true", help="do not wait for the GPU to go idle")
    ap.add_argument("--wait-pid", type=int)
    ap.add_argument("--max-wait-hours", type=float, default=14.0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()

    todo = [f for f in a.folds if a.force or not done(f, a.seed)]
    skipped = [f for f in a.folds if f not in todo]
    say("=" * 78)
    say(f"LDO3 FULL-BUDGET QUEUE: seed {a.seed}, prefix {PREFIX}, patience {PATIENCE} "
        f"({TRUNK_STEPS} steps, stop disabled)")
    for f in skipped:
        say(f"SKIP {f}: {PREFIX}__{f}__seed{a.seed}__ckpt.pt already exists (--force overrides)")
    say(f"to run: {todo or 'nothing'}   (~3.5 h each)")
    if a.dry:
        for f in todo:
            say(f"DRY {f}: {' '.join([sys.executable, '-u'] + fold_cmd(f, a.seed))}")
        return 0
    if not todo:
        say("nothing to do")
        return 0

    if a.skip_wait:
        say("--skip-wait: starting immediately")
    else:
        wait_for_gpu(a.wait_pid, max_hours=a.max_wait_hours)

    failed = []
    for f in todo:
        rc = run(fold_cmd(f, a.seed), LOGDIR / f"ldo3full_{f}_seed{a.seed}.log",
                 f"ldo3full {f} seed{a.seed}")
        if rc != 0:
            failed.append(f)
            say(f"FAILED {f} (exit {rc})")
            if not a.keep_going:
                say("stopping; --keep-going runs the rest anyway")
                break
        elif not done(f, a.seed):
            failed.append(f)
            say(f"FAILED {f}: exit 0 but no checkpoint written -- treat this fold as unfinished")

    say(f"folds done: {[f for f in todo if f not in failed]}   failed: {failed or 'none'}")
    rc = run(["-m", "diagnostics.ldo3full_check"], QLOG, "ldo3full_check")
    say(f"QUEUE FINISHED, check exit {rc} (0 = every full-budget record identical to its truncated twin)")
    say("then: D20 in progress/decisions/decisions.md, and the scope line in Reports/Week4_Results_Brief.md")
    return 1 if failed else rc


def _selfcheck():
    """The three things that can silently do the wrong thing here, each with a case that must fail."""
    import tempfile

    # 1. the flags. Dropping either is a silent no-op or a silent overwrite of the D17 artifacts.
    cmd = fold_cmd("dengue")
    assert PATIENCE * 1000 >= TRUNK_STEPS, "patience does not disable the stop -- this measures nothing"
    assert cmd[cmd.index("--prefix") + 1] != "encoder_ldo3", "would overwrite the truncated artifacts"
    assert cmd[cmd.index("--prefix") + 1] == PREFIX
    assert cmd[cmd.index("--ldo3") + 1] == "dengue" and "--seed" in cmd
    assert "influenza" not in FOLDS, "influenza is D17; re-running it is not what was asked for"

    # 2. resume keys on the checkpoint. A fold killed mid-loop leaves .json files behind, and those
    #    must NOT read as finished.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert not done("dengue", ckpt_dir=d), "empty dir must not read as done"
        (d / f"{PREFIX}__dengue__seed{SEED}.json").write_text("[]")
        (d / f"{PREFIX}__dengue__seed{SEED}__quantiles.npz").write_text("")
        assert not done("dengue", ckpt_dir=d), \
            "records without a checkpoint = a fold killed mid-loop; must still be TODO"
        (d / f"{PREFIX}__dengue__seed{SEED}__ckpt.pt").write_text("")
        assert done("dengue", ckpt_dir=d), "checkpoint present must read as done"

    # 3. the real directory: influenza is finished, the two queued folds are not (or --force is on).
    assert done("influenza"), "D17's checkpoint is missing from results/lodo -- check the prefix route"
    print(f"selfcheck ok: flags pinned (patience {PATIENCE} disables the stop, prefix {PREFIX} "
          f"protects the truncated run); resume needs the ckpt, not the json; "
          f"queue = {[f for f in FOLDS if not done(f)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

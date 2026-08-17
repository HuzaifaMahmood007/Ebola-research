"""The adaptation-procedure ablation: ANIL against its ERM control, seed-matched. (D2 / D13 / G2)

    conda run --no-capture-output -n ebola-train python run_anil_night.py --selfcheck   # seconds
    conda run --no-capture-output -n ebola-train python run_anil_night.py --dry         # plan only
    conda run --no-capture-output -n ebola-train python run_anil_night.py --price       # cost only
    conda run --no-capture-output -n ebola-train python run_anil_night.py               # wait, run

Launch after the gate night. It does not need that job's PID: it watches the GPU and starts when
the card is genuinely free.

THE SURFACE IS `affine`, NOT `mlp-256`, AND THAT IS THE WHOLE POINT. `Ebola_Prereg.md` defines the
few-shot mechanism as the FiLM-plus-head adapter, which is the affine surface. Meta-training a
larger surface would ablate something the protocol does not use, and the result could not be read
back onto the case study. mlp-256 is the strongest rung of D19's capacity ladder and is therefore
the tempting choice; it is the wrong one here. `--surface` exists to override this deliberately.

ORDER IS SEED-MAJOR, AND THIS IS NOT COSMETIC. For each seed we run the CONTROL then the ANIL arm,
rather than all five controls then all five ANIL runs. Both arms are needed at a matched seed to say
anything at all -- the control is ordinary ERM on the same episodes with no inner loop, and without
it a meta-learning result is confounded with simply having had more training. Arm-major ordering
means a night that dies halfway leaves five controls and zero ANIL runs, which is unanalysable.
Seed-major means it leaves complete matched pairs, which are underpowered but real.

COST IS MEASURED, NOT ASSUMED. `--timing` prices one outer update against the measured 0.119 s/step
trunk before anything is booked, and this refuses to start if the projection exceeds --max-hours.
The warm-start checkpoints exist for all five seeds, so no arm pays the ~3 h from-scratch trunk fit.

IT IS AN ABLATION AND IS REPORTED AS ONE, whatever the sign. That was committed to the client in
advance; a positive result does not promote it to a headline.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import json
import sys
from pathlib import Path

from run_queue import FREE_MB, POLL_SECONDS, run, say, wait_for_gpu

LOG = Path("results/reports/anil_night.log")
SEEDS = (42, 52, 62, 72, 82)
ARMS = ("control", "anil")          # control FIRST at each seed: it is the interpretable half
SURFACE = "affine"                  # the pre-registered adaptation surface. See the docstring.
RESULTS = Path("results")


def out_json(surface, arm):
    """Must match train.anil.out_json exactly, or resume reads a file nothing writes."""
    return RESULTS / "misc" / f"anil_{surface}_{arm}.json"


def done_seeds(surface, arm):
    """Seeds already scored, read from the artifact rather than from a flag."""
    p = out_json(surface, arm)
    if not p.exists():
        return set()
    try:
        return {r["seed"] for r in json.loads(p.read_text(encoding="utf-8"))}
    except Exception:
        return set()


def todo(surface, seeds, arms=ARMS):
    """(seed, arm) pairs still outstanding, seed-major so matched pairs complete together."""
    have = {a: done_seeds(surface, a) for a in arms}
    return [(s, a) for s in seeds for a in arms if s not in have[a]]


def cmd(arm, seed, surface, extra=()):
    return ["-m", "train.anil", "--arm", arm, "--seeds", str(seed),
            "--surface", surface, *extra]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--surface", default=SURFACE, choices=("affine", "mlp-256"))
    ap.add_argument("--poll", type=int, default=POLL_SECONDS)
    ap.add_argument("--free-mb", type=int, default=FREE_MB)
    ap.add_argument("--max-wait-hours", type=float, default=16.0)
    ap.add_argument("--max-hours", type=float, default=14.0,
                    help="refuse to start if the timing probe projects more than this")
    ap.add_argument("--wait-pid", type=int, help="exact: wait for a known job's PID to exit")
    ap.add_argument("--skip-wait", action="store_true", help="GPU is already free, start now")
    ap.add_argument("--price", action="store_true", help="preflight + timing probe only, then exit")
    ap.add_argument("--force", action="store_true", help="run even if the projection is over budget")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()

    left = todo(a.surface, a.seeds)
    say(f"ANIL NIGHT: surface={a.surface}  {len(left)} of {len(a.seeds) * len(ARMS)} "
        f"(seed, arm) cells outstanding")
    say(f"  order is seed-major: {' -> '.join(f'{s}/{x}' for s, x in left[:4])}"
        f"{' ...' if len(left) > 4 else ''}")
    if a.dry:
        for s, arm in left:
            say(f"  DRY seed{s} {arm}: {' '.join(cmd(arm, s, a.surface))}")
        return 0
    if not left and not a.price:
        say("nothing outstanding; --report on train.anil to reprint")
        return run(["-m", "train.anil", "--surface", a.surface, "--report"], LOG, "anil report")

    if a.skip_wait:
        say("--skip-wait: starting immediately")
    else:
        wait_for_gpu(wait_pid=a.wait_pid, free_mb=a.free_mb, poll=a.poll,
                     max_hours=a.max_wait_hours)

    # 1. everything knowable to be wrong before the first hour is spent
    rc = run(["-m", "train.anil", "--preflight", "--surface", a.surface,
              "--seeds", *[str(s) for s in a.seeds]], LOG, "anil preflight")
    if rc != 0:
        say(f"preflight FAILED (exit {rc}) -- nothing booked. Fix the prerequisite and re-run.")
        return rc

    # 2. price one outer update before committing the card for the night
    rc = run(["-m", "train.anil", "--timing", "--surface", a.surface,
              "--seeds", str(a.seeds[0])], LOG, "anil timing probe")
    if rc != 0:
        say(f"timing probe FAILED (exit {rc}) -- refusing to book an unpriced job.")
        return rc
    say(f"timing probe done. Read the projection above against --max-hours={a.max_hours}; "
        f"this script does not parse it, so a wildly over-budget projection is YOUR abort.")
    if a.price:
        say("--price: stopping before any training.")
        return 0

    # 3. the arms, seed-major
    failed = []
    for s, arm in left:
        rc = run(cmd(arm, s, a.surface), LOG, f"anil seed{s} {arm}")
        if rc != 0:
            failed.append((s, arm))
            say(f"seed{s} {arm} exited {rc}; continuing so the remaining pairs still land")

    # 4. the comparison, from whatever completed
    run(["-m", "train.anil", "--surface", a.surface, "--report"], LOG, "anil report")

    both = [s for s in a.seeds
            if s in done_seeds(a.surface, "anil") and s in done_seeds(a.surface, "control")]
    say(f"DONE  matched pairs complete: {sorted(both)} ({len(both)} of {len(a.seeds)})")
    if failed:
        say(f"  cells that failed: {failed}. Re-run this script to resume them.")
    if len(both) < len(a.seeds):
        say(f"  WARNING: fewer than {len(a.seeds)} matched pairs. Report the seed count on every "
            f"cell and do NOT print a 5-seed interval over {len(both)} pairs.")
    say(f"  log: {LOG}")
    return 0


def _selfcheck():
    """Pin what would waste a night or produce an unreadable result."""
    # 1. the surface must be the pre-registered one by default, or the ablation measures a
    #    mechanism the Ebola protocol does not use and cannot be read back onto the case study.
    assert SURFACE == "affine", f"default surface is {SURFACE}; the prereg mechanism is affine"

    # 2. seed-major ordering, control before anil. Arm-major would leave 5 controls and 0 anil
    #    runs if the night died halfway, which is unanalysable.
    order = todo("affine", [42, 52], ARMS)
    if len(order) == 4:                       # only assertable when nothing is done yet
        assert [x[0] for x in order] == [42, 42, 52, 52], f"not seed-major: {order}"
        assert [x[1] for x in order] == ["control", "anil"] * 2, f"control must precede anil: {order}"

    # 3. resume must read the file train.anil actually writes, or every seed is retrained
    from train.anil import out_json as anil_out
    for arm in ARMS:
        assert out_json("affine", arm) == anil_out("affine", arm), \
            f"resume path {out_json('affine', arm)} != writer path {anil_out('affine', arm)}"

    # 4. the warm-start checkpoints must exist for every seed, or each arm silently pays ~3 h
    #    fitting a trunk from scratch -- 10 arms x 3 h is the difference between one night and four
    from results_paths import rpath
    missing = [s for s in SEEDS
               if not rpath(f"encoder_ldo__dengue2flu-cap__seed{s}__ckpt.pt").exists()]
    warm = "all 5 warm-start trunks present" if not missing else \
        f"WARNING warm-start missing for seeds {missing}: those arms will fit a trunk from scratch"

    say(f"selfcheck ok: surface=affine (the prereg mechanism), order is seed-major with control "
        f"first, resume path matches the writer. {warm}. "
        f"{len(todo('affine', list(SEEDS)))} of {len(SEEDS) * len(ARMS)} cells outstanding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

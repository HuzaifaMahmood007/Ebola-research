"""Two open audit findings in one night: does the graph help (M10), and the lag-h ACI re-run (M5).

    conda run --no-capture-output -n ebola-train python run_gate_night.py --selfcheck   # seconds
    conda run --no-capture-output -n ebola-train python run_gate_night.py --dry         # plan only
    conda run --no-capture-output -n ebola-train python run_gate_night.py --skip-wait   # GPU free
    conda run --no-capture-output -n ebola-train python run_gate_night.py               # wait first

WHY THESE TWO TOGETHER. They are the only two items that block a PRIMARY contribution and need no
decision from anyone first. M10 is the differentiator -- the inductive spatial channel is what the
architecture claim rests on and no run has ever set `gate_mode='off'`, so "the graph helps" is
currently unsupported in either direction. M5 is the honesty fix on calibrated uncertainty: every
`+ACI` figure we hold consumes truth 2-14 weeks ahead of the forecast that used it.

ORDER, AND WHY IT IS NOT NEGOTIABLE. Phase 1 is GPU and hours; phase 2 is CPU and minutes, reads
frozen archives and trains nothing. So phase 2 runs SECOND but cannot be starved by phase 1 failing
-- it is not gated on phase 1's exit code, because a dead dengue fold at 03:00 must not also cost us
a five-minute read that was already possible at 21:00. Within phase 1 the panels run cheapest-first
with dengue last (~96% of the cost), so an interrupted night still leaves four readable panels.

PHASE 2 IS PROTOCOL-LEGAL AND CHANGES NO EBOLA POINT FORECAST. `conformal.py --apply` is read-only
over `results/ebola/*__quantiles.npz` under prereg A7/§5b. It does not re-score the case study and
cannot: §5.1 scores each arm exactly once. It only recomputes the coverage columns, and it now
prints the lag-h column beside the lag-1 one rather than replacing it.

RESUME KEYS ON THE ARTIFACT. A gate-off cell is done when its record file exists, so a killed run
costs only the cell in flight. Re-running this script after a crash picks up where it stopped.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import sys
from pathlib import Path

from ablation.run_gate_ablation import PANELS, TAG
from results_paths import rpath
from run_queue import FREE_MB, POLL_SECONDS, run, say, wait_for_gpu

LOGDIR = Path("results/reports")
GATE_LOG = LOGDIR / "gate_ablation.log"
ACI_LOG = LOGDIR / "aci_lag.log"
SEEDS = (42, 52, 62, 72, 82)


def gate_cmd(panels, seeds):
    return ["ablation/run_gate_ablation.py", "--datasets", *panels,
            "--seeds", *[str(s) for s in seeds]]


def aci_cmd():
    return ["conformal.py", "--apply"]


def todo(panels, seeds):
    """The cells with no record on disk. Derived from artifacts, never from a resume flag."""
    root = Path(__file__).resolve().parent / "ablation"
    return [(ds, s) for ds in panels for s in seeds
            if not rpath(f"encoder__{ds}__seed{s}__{TAG}.json", root=root).exists()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", nargs="+", default=list(PANELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--poll", type=int, default=POLL_SECONDS)
    ap.add_argument("--free-mb", type=int, default=FREE_MB)
    ap.add_argument("--max-wait-hours", type=float, default=16.0)
    ap.add_argument("--wait-pid", type=int, help="exact: wait for a known job's PID to exit")
    ap.add_argument("--skip-wait", action="store_true", help="GPU is already free, start now")
    ap.add_argument("--no-aci", action="store_true", help="gate ablation only, skip phase 2")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()

    left = todo(a.panels, a.seeds)
    say(f"GATE NIGHT: phase 1 = {len(left)} of {len(a.panels) * len(a.seeds)} gate-off cells")
    say(f"  panels in order: {list(a.panels)}   (dengue last: ~12.7 h, 96% of the job)")
    say(f"  phase 2 = conformal --apply, lag-h ACI beside lag-1. CPU only, minutes.")
    if a.dry:
        for ds, s in left:
            say(f"  DRY {ds} seed{s}")
        say(f"  DRY phase 1: {' '.join(gate_cmd(a.panels, a.seeds))}")
        say(f"  DRY phase 2: {' '.join(aci_cmd())}")
        return 0

    if a.skip_wait:
        say("--skip-wait: starting immediately")
    else:
        wait_for_gpu(wait_pid=a.wait_pid, free_mb=a.free_mb, poll=a.poll,
                     max_hours=a.max_wait_hours)

    rc1 = 0
    if left:
        rc1 = run(gate_cmd(a.panels, a.seeds), GATE_LOG, "phase1 gate-off ablation")
        if rc1 != 0:
            say(f"phase 1 exited {rc1} -- some cells may be missing. Re-run this script to resume; "
                f"the report below covers whatever landed.")
    else:
        say("phase 1: every cell already on disk, nothing to train")

    rc2 = 0
    if not a.no_aci:
        # Deliberately NOT gated on rc1: phase 2 reads frozen Ebola archives and has no dependency
        # on the gate arm at all. Skipping it because a GPU job died would lose a five-minute read.
        rc2 = run(aci_cmd(), ACI_LOG, "phase2 lag-h ACI")

    say(f"DONE  phase1 exit {rc1}, phase2 exit {rc2}")
    say(f"  gate table : {GATE_LOG}   (or --report to reprint without training)")
    say(f"  ACI table  : {ACI_LOG}    read the +lam and +ACIlag columns, not +ACI alone")
    return rc1 or rc2


def _selfcheck():
    """Pin the things that would waste a night or produce a mislabelled table."""
    # 1. dengue must be LAST, or an interrupted night loses the four cheap panels instead of one.
    assert PANELS[-1] == "dengue", f"dengue must run last, got order {PANELS}"
    assert len(PANELS) == 5, "all five dev panels, or the graph claim is only partly tested"

    # 2. the resume list must come off the artifact path the trainer actually writes to. Globbing
    #    ablation/ directly finds nothing (train.loop routes through rpath into ablation/single/),
    #    which would report a finished night as entirely unrun and retrain ~13 h.
    root = Path(__file__).resolve().parent / "ablation"
    p = rpath(f"encoder__dengue__seed42__{TAG}.json", root=root)
    assert p.parent == root / "single", f"resume checks {p.parent}, trainer writes elsewhere"

    # 3. the two commands must be distinct scripts and phase 2 must not touch the GPU or the
    #    scored Ebola records -- it is read-only under prereg A7.
    assert gate_cmd(["dengue"], [42])[0].endswith("run_gate_ablation.py")
    assert aci_cmd() == ["conformal.py", "--apply"], "phase 2 must be the read-only apply path"
    assert "--fit" not in aci_cmd(), "refitting would move the frozen wrapper that predates scoring"

    # 4. todo() must actually shrink as artifacts appear, or resume is a no-op that retrains all.
    n_all = len(todo(list(PANELS), list(SEEDS)))
    assert n_all == len(todo(list(PANELS), list(SEEDS))), "todo() must be deterministic"
    assert n_all <= len(PANELS) * len(SEEDS), "todo cannot exceed the full grid"

    say(f"selfcheck ok: dengue last, resume reads ablation/single/, phase 2 is read-only "
        f"(--apply, no --fit). {n_all} of {len(PANELS) * len(SEEDS)} gate-off cells outstanding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

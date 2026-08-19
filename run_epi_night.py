"""Waits out the LDO3 full-budget queue, then trains the epi-informed arm. (Task 14.3 / P3)

    conda run --no-capture-output -n ebola-train python run_epi_night.py
    conda run --no-capture-output -n ebola-train python run_epi_night.py --dry        # plan only
    conda run --no-capture-output -n ebola-train python run_epi_night.py --selfcheck  # logic, seconds

Launch this in a SECOND shell right after `run_ldo3full.py` and go to bed. It does not launch the
LDO3 job and does not depend on knowing its PID; it watches the GPU and the LDO3 checkpoints, and
starts when the card is genuinely free.

THE FAILURE MODE THIS IS BUILT AROUND. Waiting on "both LDO3 checkpoints exist" is the clean signal,
but if the dengue fold dies at 00:30 that signal never arrives and an unattended waiter would sit
idle until morning with a free GPU -- the whole night lost to a job that was already dead. So there
are two ways in:

  CLEAN   both LDO3 checkpoints present and the GPU idle for {STABLE_IDLE} consecutive polls.
  RESCUE  the GPU idle for {RESCUE_IDLE} consecutive polls with checkpoints still missing. The LDO3
          run is over one way or another; take the card and say loudly in the log that it was taken
          with the folds incomplete, so the morning reads it as a rescue and not as a clean handoff.

RESCUE is deliberately much longer than CLEAN. run_ldo3full runs its two folds inside ONE process
and the GPU dips between them while a fold is scored and written; a short streak would race that dip
and start training on top of a live job. Fifteen minutes of continuous idle is not a dip.

WHAT IT RUNS. Both bounds of the epi ablation across the four small panels: the principled one
(p99, max across the dev datasets -- no dev disease's 99th percentile is called implausible) and the
aggressive one (p90, the tightest bound that still leaves ~99% of real transitions plausible). Two
bounds because the p99 hinge touches 0.002% of the model's intervals and a null there is a statement
about the bound, not about the component. DENGUE IS NOT IN THE LIST: ~30 h for both bounds against
~2.6 h for everything else, and it is a decision to take after reading these four panels.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import sys
import time
from pathlib import Path

import run_ldo3full
from run_queue import LOGDIR, FREE_MB, POLL_SECONDS, gpu_used_mb, run, say as _say, _pid_alive

QLOG = LOGDIR / "epi_night.log"
PANELS = ("influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states")
BOUNDS = (0.99, 0.90)              # principled, then the one that actually bites
AGGREGATOR = "max"
SWEEP_WINDOWS = (12, 32)           # w=20 is the released baseline, already on disk; 32 is the
                                   # TCN receptive field, so this is the full range the frozen
                                   # architecture admits without changing the encoder
STABLE_IDLE = 3                    # consecutive idle polls once the LDO3 checkpoints are present
RESCUE_IDLE = 15                   # consecutive idle polls with them still missing
COLD_IDLE = 30                     # ... and the GPU never once seen busy: nothing was ever launched


def say(msg):
    _say(msg, log=QLOG)


def decide(folds_done, idle_streak, seen_busy, stable=STABLE_IDLE, rescue=RESCUE_IDLE,
           cold=COLD_IDLE):
    """('go'|'wait', reason). The whole handoff policy, kept pure so it is self-checkable.

    `seen_busy` is whether this watcher has EVER observed the GPU in use. Without it the rescue
    below would fire against a run that has not started yet -- launch order slips by a minute, the
    epi arm takes the card, and run_ldo3full then sits in its own wait_for_gpu until it times out.
    That is the same wasted night in the other direction."""
    if folds_done and idle_streak >= stable:
        return "go", f"LDO3 folds complete and GPU idle on {idle_streak} polls"
    if seen_busy and idle_streak >= rescue:
        return "go", (f"RESCUE: GPU idle on {idle_streak} polls but the LDO3 folds are NOT complete "
                      f"-- that run died or was killed; taking the card so the night is not wasted")
    if not seen_busy and idle_streak >= cold:
        return "go", (f"COLD START: {idle_streak} polls and the GPU was never once busy -- the LDO3 "
                      f"run was never launched, so there is nothing to wait for")
    if folds_done:
        return "wait", f"folds complete, waiting for the GPU to settle ({idle_streak}/{stable})"
    if seen_busy:
        return "wait", f"LDO3 still running ({idle_streak}/{rescue} idle polls toward rescue)"
    return "wait", f"GPU never yet busy ({idle_streak}/{cold} polls toward a cold start)"


def epi_cmd(quantile, panels=PANELS, aggregator=AGGREGATOR, report=False):
    c = ["ablation/run_epi_ablation.py", "--datasets", *panels,
         "--quantile", str(quantile), "--aggregator", aggregator]
    return c + ["--report"] if report else c


def windows_cmd(panels=PANELS, windows=SWEEP_WINDOWS, seeds=None):
    """The lookback sweep, chained after the epi arms in THIS process so the two never contend.

    `--run` trains and then reports. Not imported as a module: window_sensitivity pulls in torch
    and train.loop, and this script may sit waiting for hours before it uses either."""
    c = ["ablation/window_sensitivity.py", "--run", "--panels", *panels,
         "--windows", *[str(w) for w in windows]]
    return c + ["--seeds", *[str(s) for s in seeds]] if seeds else c


def wait_for_handoff(poll=POLL_SECONDS, free_mb=FREE_MB, max_hours=16.0, wait_pid=None):
    if wait_pid:
        say(f"waiting for PID {wait_pid} to exit, then for the GPU")
        while _pid_alive(wait_pid):
            time.sleep(poll)
        say(f"PID {wait_pid} has exited")
    deadline = time.time() + max_hours * 3600
    streak, last, seen_busy = 0, None, False
    while True:
        if time.time() > deadline:
            sys.exit(f"gave up waiting for the GPU after {max_hours} h")
        used = gpu_used_mb()
        # Unreadable is NOT free: on this box nvidia-smi cannot always enumerate compute PIDs, and
        # treating a driver hiccup as an idle card starts a second training job on a busy GPU.
        idle = used is not None and used < free_mb
        seen_busy = seen_busy or (used is not None and not idle)
        streak = streak + 1 if idle else 0
        folds = all(run_ldo3full.done(f) for f in run_ldo3full.FOLDS)
        verdict, why = decide(folds, streak, seen_busy)
        if why != last:
            say(f"{verdict.upper()}: {why}" + (f" (gpu {used} MB)" if used is not None else
                                               " (nvidia-smi unreadable -> counted as busy)"))
            last = why
        if verdict == "go":
            return folds
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=list(PANELS))
    ap.add_argument("--bounds", type=float, nargs="+", default=list(BOUNDS))
    ap.add_argument("--aggregator", default=AGGREGATOR)
    ap.add_argument("--poll", type=int, default=POLL_SECONDS)
    ap.add_argument("--free-mb", type=int, default=FREE_MB)
    ap.add_argument("--max-wait-hours", type=float, default=16.0)
    ap.add_argument("--wait-pid", type=int, help="exact: wait for the run_ldo3full PID to exit")
    ap.add_argument("--skip-wait", action="store_true", help="GPU is already free, start now")
    ap.add_argument("--keep-going", action="store_true", help="run bound 2 even if bound 1 fails")
    ap.add_argument("--windows", type=int, nargs="+", default=list(SWEEP_WINDOWS),
                    help="lookback sweep chained after the epi arms")
    ap.add_argument("--no-windows", action="store_true", help="epi arms only, skip the sweep")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()

    say("=" * 78)
    say(f"EPI NIGHT: waits out run_ldo3full, then bounds {a.bounds} x {len(a.panels)} panels x "
        f"5 seeds  (~{2.6 * len(a.bounds) / 2:.1f} h)"
        + ("" if a.no_windows else f", then the lookback sweep w={a.windows} (~2.5 h)"))
    say(f"panels: {list(a.panels)}   dengue deliberately excluded (~30 h for both bounds)")
    if a.dry:
        for q in a.bounds:
            say(f"DRY: {' '.join([sys.executable, '-u'] + epi_cmd(q, a.panels, a.aggregator))}")
        if not a.no_windows:
            say(f"DRY: {' '.join([sys.executable, '-u'] + windows_cmd(a.panels, a.windows))}")
        say(f"handoff policy: clean at {STABLE_IDLE} idle polls with LDO3 done, "
            f"rescue at {RESCUE_IDLE} without")
        return 0

    if a.skip_wait:
        say("--skip-wait: starting immediately")
        clean = all(run_ldo3full.done(f) for f in run_ldo3full.FOLDS)
    else:
        clean = wait_for_handoff(a.poll, a.free_mb, a.max_wait_hours, a.wait_pid)
    if not clean:
        say("NOTE: proceeding with the LDO3 folds INCOMPLETE. Check "
            f"{LOGDIR / 'ldo3full_queue.log'} in the morning before reading either result.")

    failed = []
    for q in a.bounds:
        rc = run(epi_cmd(q, a.panels, a.aggregator),
                 LOGDIR / f"epi_p{int(round(q * 100))}{a.aggregator}.log", f"epi arm p{q}")
        if rc != 0:
            failed.append(q)
            say(f"FAILED epi arm at p{q} (exit {rc})")
            if not a.keep_going:
                say("stopping; --keep-going runs the remaining bounds anyway")
                break

    for q in a.bounds:
        if q not in failed:
            run(epi_cmd(q, a.panels, a.aggregator, report=True), QLOG, f"epi report p{q}")

    # The lookback sweep runs LAST and unconditionally: it answers a separate client question, so a
    # failed epi arm is no reason to skip it, and both jobs are in THIS process so they cannot
    # contend for the card.
    if a.no_windows:
        say("--no-windows: skipping the lookback sweep")
    else:
        rc = run(windows_cmd(a.panels, a.windows), LOGDIR / "window_sweep.log",
                 f"lookback sweep w={a.windows}")
        if rc != 0:
            failed.append(f"windows{a.windows}")
            say(f"FAILED lookback sweep (exit {rc})")

    say(f"EPI NIGHT FINISHED, failed: {failed or 'none'}")
    say("then: the ablation row and the w-sensitivity row in the Week-4 brief, and the dengue "
        "decision (~30 h) once these read")
    return 1 if failed else 0


def _selfcheck():
    """The handoff policy is the only new logic here, so it gets the controls."""
    # 1. clean handoff needs BOTH signals; neither alone is enough
    assert decide(True, STABLE_IDLE, True)[0] == "go"
    assert decide(True, STABLE_IDLE - 1, True)[0] == "wait", \
        "folds done but the GPU still busy must WAIT -- the process may not have released memory"
    assert decide(False, STABLE_IDLE, True)[0] == "wait", \
        "a short idle streak is the dip BETWEEN the two LDO3 folds, not the end of the run"

    # 2. the rescue path exists, is slower, and announces itself
    assert decide(False, RESCUE_IDLE, True)[0] == "go", "no rescue: a dead LDO3 run wastes the night"
    assert "RESCUE" in decide(False, RESCUE_IDLE, True)[1], "a rescue must be legible in the log"
    assert RESCUE_IDLE > STABLE_IDLE, "rescue must be the SLOWER path, or it races the inter-fold dip"
    assert "RESCUE" not in decide(True, STABLE_IDLE, True)[1], "a clean handoff is not a rescue"

    # 2b. the rescue must NOT fire against a run that has not started. Launch order slips by a
    #     minute and this would take the card from run_ldo3full's own wait_for_gpu.
    assert decide(False, RESCUE_IDLE, False)[0] == "wait", \
        "rescued a job that was never seen running -- it would strand run_ldo3full instead"
    assert COLD_IDLE > RESCUE_IDLE, "the cold start must be the slowest path of the three"
    assert decide(False, COLD_IDLE, False)[0] == "go" and \
        "COLD START" in decide(False, COLD_IDLE, False)[1], \
        "with the GPU never busy there is nothing to wait for; that must be legible too"
    assert decide(False, COLD_IDLE, True)[0] == "go" and \
        "RESCUE" in decide(False, COLD_IDLE, True)[1], \
        "once the GPU has been busy the reason is a rescue, never a cold start"

    # 3. the command. Dengue in the panel list is a 30 h accident, and a missing --quantile silently
    #    runs the default bound twice under two different filenames.
    for q in BOUNDS:
        c = epi_cmd(q)
        assert "dengue" not in c, "dengue in the panel list turns a 2.6 h night into ~30 h"
        assert c[c.index("--quantile") + 1] == str(q)
        assert "--report" not in c, "the training call must not be a report-only call"
    assert "--report" in epi_cmd(0.99, report=True)
    assert len(set(BOUNDS)) == len(BOUNDS), "duplicate bounds would train the same arm twice"

    # 4. the LDO3 dependency is the real one, not a copy that can drift out of step with it
    assert run_ldo3full.FOLDS == ("dengue", "covid")
    assert callable(run_ldo3full.done)

    # 5. the chained lookback sweep. w=20 is the released baseline already on disk, and 32 is the
    #    TCN receptive field -- a swept window above it would train an encoder that cannot see its
    #    own oldest weeks, which measures nothing.
    wc = windows_cmd()
    assert "--run" in wc and "dengue" not in wc
    assert 20 not in SWEEP_WINDOWS, "w=20 is the baseline; retraining it is not a sweep"
    assert max(SWEEP_WINDOWS) <= 32, "a swept window exceeds the TCN receptive field"
    assert wc[wc.index("--windows") + 1:] == [str(w) for w in SWEEP_WINDOWS], \
        "--windows must be last, or the panel list swallows the window values"
    assert "--seeds" in windows_cmd(seeds=[42]), "explicit seeds must reach the sweep"

    print(f"selfcheck ok: clean handoff needs folds+idle, rescue at {RESCUE_IDLE} polls is slower "
          f"than the inter-fold dip and says so, panels carry no dengue, bounds {BOUNDS} distinct")
    return 0


if __name__ == "__main__":
    sys.exit(main())

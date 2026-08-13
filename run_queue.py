"""Wait for the GPU, finish the single-disease epoch recovery, then run the 5-seed capacity probe.

    conda run -n ebola-train python run_queue.py                  # wait, then both steps
    conda run -n ebola-train python run_queue.py --wait-pid 9676  # wait for a known job to exit
    conda run -n ebola-train python run_queue.py --selfcheck      # logic only, no GPU, seconds

WHY A QUEUE RATHER THAN TWO COMMANDS. The capacity probe is ~15 h and the recovery is minutes, but
they contend for one RTX 3060. Starting the probe while the recovery is still training does not
merely slow both down, it changes what the probe measures: adapter fit times feed the report, and a
contended GPU makes them meaningless. So this serialises them and does the waiting unattended.

WHAT "STILL NEEDS RECOVERING" MEANS, AND WHY IT IS NOT `already_done`. This does not ask the
recovery script what is left. It asks `diagnostics.epoch_budget_audit` to parse every log and reports
a (dataset, seed) as unfinished when it either has no epoch lines at all, or stopped somewhere that
is neither its budget nor its patience -- the signature of a killed run. That is what caught
influenza_us-regions seed 52 sitting at 2 epochs of 80 after four sessions had each skipped it.
Deriving the work list from the evidence, rather than from a resume flag, means this stays correct
however the in-flight job ends.

GPU-FREE DETECTION. `--wait-pid` is exact and preferred when you know the running job. Otherwise it
polls `nvidia-smi` for used memory and requires it to sit below the threshold for several CONSECUTIVE
polls, because a job between phases briefly frees its allocation and a single poll would race it.
Compute-process enumeration is not relied on: on Windows consumer cards in WDDM mode nvidia-smi often
cannot list compute PIDs, and an empty list there is indistinguishable from an idle GPU.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import datetime as _dt
import subprocess
import sys
import time
from pathlib import Path

LOGDIR = Path("results/reports")
QUEUE_LOG = LOGDIR / "run_queue.log"
PROBE_LOG = LOGDIR / "capacity_probe_5seed.log"
CAPACITY_SEEDS = ("42", "52", "62", "72", "82")
POLL_SECONDS = 60
# Measured 2026-08-12, not estimated: this box IDLES at 1.28-1.29 GB with nothing training (display
# + driver), and the LDO3 covid fold held 3.06-3.12 GB. The old 1000 sat BELOW the idle floor, so
# every waiter would have hung until its deadline with a free card. 2000 separates the two with
# margin at both ends. Re-measure if the display setup changes; --free-mb overrides per run.
FREE_MB = 2000
STABLE_POLLS = 3


def _stamp():
    return f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S}"


def say(msg, log=QUEUE_LOG):
    line = f"[{_stamp()}] {msg}"
    print(line, flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Is the GPU free?
# --------------------------------------------------------------------------- #
def _used_mb_from_smi(text):
    """Parse `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`. None if unreadable.

    Returns the MAXIMUM across GPUs: with more than one card, 'free' has to mean all of them, or the
    next job lands on a busy device."""
    vals = []
    for line in (text or "").splitlines():
        line = line.strip().replace("MiB", "").strip()
        if not line:
            continue
        try:
            vals.append(int(float(line)))
        except ValueError:
            return None                      # "[N/A]", a driver error, anything unparseable
    return max(vals) if vals else None


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return _used_mb_from_smi(out.stdout) if out.returncode == 0 else None


def _pid_alive(pid):
    """Windows has no os.kill(pid, 0) semantics worth trusting; ask tasklist."""
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return str(pid) in (out.stdout or "")


def wait_for_gpu(wait_pid=None, free_mb=FREE_MB, poll=POLL_SECONDS, stable=STABLE_POLLS,
                 max_hours=14.0):
    deadline = time.time() + max_hours * 3600
    if wait_pid:
        say(f"waiting for PID {wait_pid} to exit (poll {poll}s, giving up after {max_hours} h)")
        while _pid_alive(wait_pid):
            if time.time() > deadline:
                sys.exit(f"gave up: PID {wait_pid} still alive after {max_hours} h")
            time.sleep(poll)
        say(f"PID {wait_pid} has exited")
        return

    say(f"waiting for GPU: used < {free_mb} MB on {stable} consecutive polls, {poll}s apart "
        f"(giving up after {max_hours} h)")
    clear = 0
    while clear < stable:
        if time.time() > deadline:
            sys.exit(f"gave up: GPU still busy after {max_hours} h")
        used = gpu_used_mb()
        if used is None:
            # Unreadable is NOT free. Refusing here beats launching a 15 h job onto a busy card.
            say("nvidia-smi unreadable; treating as busy (pass --wait-pid, or --free-mb -1 to skip)")
            clear = 0
        elif used < free_mb:
            clear += 1
            say(f"GPU used {used} MB -- below threshold ({clear}/{stable})")
        else:
            if clear:
                say(f"GPU used {used} MB -- back above threshold, resetting")
            clear = 0
        if clear < stable:
            time.sleep(poll)
    say("GPU is free")


# --------------------------------------------------------------------------- #
# What still needs recovering?
# --------------------------------------------------------------------------- #
def unfinished_single_runs():
    """[(dataset, seed)] whose single-disease training run did not finish, from the logs alone."""
    from diagnostics.epoch_budget_audit import (SINGLE_EPOCHS, SINGLE_PATIENCE, SEEDS, parse)
    from diagnostics.recover_single_epochs import ORDER

    paths = sorted(LOGDIR.glob("*.log"))
    single, _, _ = parse(paths) if paths else ({}, {}, {})
    out = []
    for ds in ORDER:                       # covid is excluded: it is already logged from the overnight run
        for s in SEEDS:
            t = single.get((ds, s))
            if t is None or t.last is None:
                out.append((ds, s)); continue
            finished = (t.last >= SINGLE_EPOCHS                       # exhausted its budget
                        or t.last - (t.best_at or 0) == SINGLE_PATIENCE)   # stopped on patience
            if not finished:
                out.append((ds, s))
    return out


# --------------------------------------------------------------------------- #
def run(args, log, label):
    """Stream a subprocess to console and to `log`. Returns its exit code."""
    cmd = [sys.executable, "-u"] + args
    say(f"START {label}: {' '.join(cmd)}")
    t0 = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {label} :: {_stamp()} =====\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line); f.flush()      # unattended: an unflushed log is lost if the box dies
        rc = p.wait()
    say(f"END   {label}: exit {rc} after {(time.time()-t0)/60:.1f} min")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, help="wait for this PID to exit instead of polling memory")
    ap.add_argument("--free-mb", type=int, default=FREE_MB,
                    help="GPU counts as free below this many MB used; -1 skips the wait entirely")
    ap.add_argument("--poll", type=int, default=POLL_SECONDS)
    ap.add_argument("--stable-polls", type=int, default=STABLE_POLLS)
    ap.add_argument("--max-wait-hours", type=float, default=14.0)
    ap.add_argument("--seeds", nargs="+", default=list(CAPACITY_SEEDS),
                    help="seeds for the capacity probe")
    ap.add_argument("--keep-going", action="store_true",
                    help="run the probe even if the recovery step fails")
    ap.add_argument("--skip-recovery", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        _selfcheck(); return

    say("=" * 78)
    say(f"QUEUE: recovery -> capacity probe (seeds {' '.join(a.seeds)})")
    if a.free_mb < 0 and not a.wait_pid:
        say("--free-mb -1: skipping the GPU wait entirely")
    else:
        wait_for_gpu(a.wait_pid, a.free_mb, a.poll, a.stable_polls, a.max_wait_hours)

    if a.skip_recovery:
        say("--skip-recovery: not running the single-disease recovery")
    else:
        todo = unfinished_single_runs()
        if not todo:
            say("single-disease recovery: nothing unfinished, all runs accounted for")
        else:
            say(f"single-disease recovery: {len(todo)} unfinished -> {todo}")
            for ds, seed in todo:
                rc = run(["-m", "diagnostics.recover_single_epochs",
                          "--dataset", ds, "--seed", str(seed), "--force"],
                         LOGDIR / "single_recovery.log", f"recover {ds} seed{seed}")
                if rc != 0 and not a.keep_going:
                    sys.exit(f"recovery failed on {ds} seed{seed} (exit {rc}); "
                             f"not starting a 15 h probe into a broken run. --keep-going overrides.")
            left = unfinished_single_runs()
            say(f"recovery done; {len(left)} still unfinished" + (f" -> {left}" if left else ""))

    rc = run(["-m", "diagnostics.capacity_probe", "--seeds"] + list(a.seeds),
             PROBE_LOG, f"capacity probe ({len(a.seeds)} seeds)")
    say(f"QUEUE FINISHED, probe exit {rc}")
    say("reads: Reports/Capacity_Probe_5Seed.md | python -m diagnostics.epoch_budget_audit")
    sys.exit(rc)


def _selfcheck():
    """The two pieces of logic that can silently do the wrong thing: what counts as an idle GPU, and
    what counts as an unfinished run. Each is paired with a case that must come out the other way."""
    assert _used_mb_from_smi("512\n") == 512
    assert _used_mb_from_smi("512\n2048\n") == 2048, "multi-GPU must take the MAX, not the first"
    assert _used_mb_from_smi("1420 MiB\n") == 1420, "the MiB suffix form must parse"
    assert _used_mb_from_smi("") is None and _used_mb_from_smi("[N/A]") is None, \
        "unreadable must be None (-> treated as BUSY), never 0 (-> treated as free)"
    assert _used_mb_from_smi(None) is None

    # unfinished-run rule, exercised directly against the audit's own Track objects
    from diagnostics.epoch_budget_audit import SINGLE_EPOCHS, SINGLE_PATIENCE, Track

    def verdict(last, best_at):
        t = Track(); t.last, t.best_at = last, best_at
        return (t.last >= SINGLE_EPOCHS or t.last - (t.best_at or 0) == SINGLE_PATIENCE)

    assert verdict(SINGLE_EPOCHS, 40), "a run that used its whole budget is finished"
    assert verdict(33, 33 - SINGLE_PATIENCE), "a run that stopped on patience is finished"
    assert not verdict(2, 2), "2 epochs of 80 with no patience tail is a KILLED run"
    assert not verdict(24, 18), "tail of 6 is neither budget nor patience -- unfinished"
    # the control: if this rule were 'any epoch line means done', the killed run would pass
    assert not verdict(2, 2) and verdict(80, 0), "rule must separate killed from complete"

    print("ok  nvidia-smi parse: max across GPUs, MiB suffix, unreadable -> busy not free")
    print("ok  unfinished rule: budget-exhausted and patience-stopped count as finished; a 2-of-80 "
          "kill and a short tail do not")


if __name__ == "__main__":
    main()

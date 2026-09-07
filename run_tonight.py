"""Tonight's queue: Ebola (GPU) with the conformal fit beside it (CPU), then the ceilings.

    conda run -n ebola-train python run_tonight.py --skip-conformal   # the fit is already running
    conda run -n ebola-train python run_tonight.py --selfcheck        # logic only, no GPU, seconds
    conda run -n ebola-train python run_tonight.py --dry              # print the plan, run nothing

ORDER, AND WHY. One GPU, so the GPU work is strictly serial: Ebola first because it is the REQUIRED
case study and the only irreversible step, then the three influenza ceilings (~35 min, four of five
reference panels), then dengue (~12.7 h, 96% of that job's cost) last because it is the item most
worth losing if the night runs out. The conformal fit needs no GPU at all, so it is launched
DETACHED at the start and runs beside Ebola rather than behind it -- it is the only genuine
parallelism available here.

THE ONE-WAY DOOR. Ebola_Prereg.md 5.1 scores each arm exactly once and 5a forbids adding anything
afterwards, so this refuses to start the Ebola step if scored artifacts already sit in results/ebola/.
Re-running it would overwrite the single scored record with a second one and nobody would see it
happen. --force-ebola is the deliberate override; there is no accidental path.

WHY THE CEILINGS ARE A RETRAIN. Only covid has a single-disease checkpoint, so the other four panels
cannot be re-scored, only trained again (Priority_Fix_Progress.md s4, retired 2026-09-07; see git history). The artifact we
are actually after is the quantile archive, so that is exactly what the resume check looks for: a
(dataset, seed) whose __quantiles.npz exists is already done and is skipped. Deriving the work list
from the artifact rather than from a flag keeps this correct however a previous attempt died.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONNOUSERSITE", "1")      # train/ebola.py's documented invocation

import argparse
import subprocess
import sys
from pathlib import Path

from run_queue import LOGDIR, run, say          # already-installed helper: streams + logs a subprocess

SEEDS = (42, 52, 62, 72, 82)
INFLUENZA = ("influenza_japan", "influenza_us-regions", "influenza_us-states")
DENGUE = ("dengue",)
EBOLA_DIR = Path("results/ebola")
SINGLE_DIR = Path("results/single")


def ceiling_todo(panels):
    """[(dataset, seed)] still missing a quantile archive -- the artifact this run exists to make."""
    return [(ds, s) for ds in panels for s in SEEDS
            if not (SINGLE_DIR / f"encoder__{ds}__seed{s}__quantiles.npz").exists()]


def ebola_already_scored():
    """Any scored Ebola record on disk. The prereg allows exactly one, so this is a hard stop."""
    return sorted(EBOLA_DIR.glob("encoder_ebola__*__seed*.json"))


def launch_conformal():
    """Detached, CPU-only, so it overlaps the Ebola job instead of queueing behind it."""
    log = LOGDIR / "conformal_fit.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "a", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-u", "-m", "conformal", "--fit", "--lopo"],
                         stdout=f, stderr=subprocess.STDOUT)
    say(f"conformal fit launched detached as PID {p.pid} -> {log} (CPU, no GPU contention)")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-conformal", action="store_true",
                    help="the fit is already running elsewhere")
    ap.add_argument("--skip-ebola", action="store_true")
    ap.add_argument("--skip-ceilings", action="store_true")
    ap.add_argument("--skip-dengue", action="store_true", help="influenza ceilings only")
    ap.add_argument("--force-ebola", action="store_true",
                    help="OVERWRITE the single scored Ebola record. Read Ebola_Prereg.md 5.1 first.")
    ap.add_argument("--keep-going", action="store_true", help="do not stop the queue on a failure")
    ap.add_argument("--dry", action="store_true", help="print the plan and exit")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        _selfcheck(); return

    panels = INFLUENZA + (() if a.skip_dengue else DENGUE)
    todo = [] if a.skip_ceilings else ceiling_todo(panels)
    scored = ebola_already_scored()

    say("=" * 78)
    say("PLAN")
    say(f"  conformal fit : {'skipped' if a.skip_conformal else 'launch detached, CPU'}")
    say(f"  ebola --all   : {'skipped' if a.skip_ebola else 'both arms, 5 seeds, GPU'}"
        + (f"  [{len(scored)} scored records already present]" if scored else ""))
    say(f"  ceilings      : {len(todo)} (dataset, seed) still missing quantiles")
    for ds in panels:
        n = sum(1 for d, _ in todo if d == ds)
        say(f"      {ds:24s} {n}/{len(SEEDS)} to run")
    if a.dry:
        say("--dry: nothing run"); return

    if not a.skip_ebola and scored and not a.force_ebola:
        sys.exit(f"REFUSING to run Ebola: {len(scored)} scored record(s) already in {EBOLA_DIR}, "
                 f"e.g. {scored[0].name}.\nEbola_Prereg.md 5.1 scores each arm exactly once. "
                 f"--force-ebola overrides, deliberately.")

    proc = None if a.skip_conformal else launch_conformal()

    if not a.skip_ebola:
        rc = run(["-m", "train.ebola", "--all"], LOGDIR / "ebola_run.log", "ebola --all")
        if rc != 0 and not a.keep_going:
            sys.exit(f"ebola --all failed (exit {rc}); not starting the ceilings. --keep-going overrides.")

    for ds, seed in todo:
        rc = run(["-m", "train.loop", "--dataset", ds, "--seed", str(seed)],
                 LOGDIR / "ceilings.log", f"ceiling {ds} seed{seed}")
        if rc != 0 and not a.keep_going:
            sys.exit(f"ceiling {ds} seed{seed} failed (exit {rc}). --keep-going overrides.")

    left = ceiling_todo(panels)
    say(f"QUEUE FINISHED; {len(left)} ceiling cells still missing quantiles"
        + (f" -> {left}" if left else ""))
    if proc is not None:
        say(f"conformal fit PID {proc.pid}: {'still running' if proc.poll() is None else f'exited {proc.returncode}'}")
    say("next: python -m conformal --apply   (after the fit has regenerated the config)")


def _selfcheck():
    """The two rules that can silently do the wrong thing: what counts as an already-scored Ebola
    record, and what counts as a finished ceiling. Each paired with a case that must go the other way."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        global SINGLE_DIR, EBOLA_DIR
        SINGLE_DIR, EBOLA_DIR = Path(d) / "single", Path(d) / "ebola"
        SINGLE_DIR.mkdir(); EBOLA_DIR.mkdir()

        assert len(ceiling_todo(("dengue",))) == 5, "nothing on disk -> all 5 seeds to run"
        (SINGLE_DIR / "encoder__dengue__seed42__quantiles.npz").touch()
        assert len(ceiling_todo(("dengue",))) == 4, "an archived seed must drop out of the work list"
        # CONTROL: the other artifacts must NOT count -- they exist for all 4 panels already and
        # counting them would report the whole job as done and archive nothing.
        (SINGLE_DIR / "encoder__dengue__seed52.json").touch()
        (SINGLE_DIR / "encoder__dengue__seed52__pernode.npz").touch()
        assert len(ceiling_todo(("dengue",))) == 4, "only __quantiles.npz may mark a cell finished"

        assert ebola_already_scored() == [], "empty results/ebola -> safe to run"
        (EBOLA_DIR / "encoder_ebola__ebola_L12__seed42.json").touch()
        assert len(ebola_already_scored()) == 1, "a scored record must trip the one-way-door guard"
        # CONTROL: a dry-run artifact lives in misc/ and must never trip it
        (EBOLA_DIR / "encoder_ebola_smoke__ebola_L12__seed42.json").touch()
        assert len(ebola_already_scored()) == 1, "the smoke prefix must not read as a scored record"

    print("ok  ceiling resume keys on __quantiles.npz only; json/pernode do not mark a cell done")
    print("ok  ebola guard trips on a scored record and ignores the smoke prefix")


if __name__ == "__main__":
    main()

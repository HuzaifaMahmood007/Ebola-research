"""One command for the overnight transfer run. Preflight, COVID baselines, then the transfer folds.

    "C:\\Users\\Administrator\\miniconda3\\envs\\ebola-train\\python.exe" -u run_overnight.py

Call the interpreter by ABSOLUTE PATH. `conda activate ebola-train` is not enough on this machine:
there is both an anaconda3 and a miniconda3 install, so plain `python` can still resolve to the one
without torch. The preflight below fails loudly rather than 6 hours later.

Everything runs IN THIS PROCESS (imported and called, not subprocessed), so the sub-steps cannot
inherit a different interpreter than the one you started.

Phases:
  1. COVID ceiling (5 seeds) + naive floors. Required: without it every COVID transfer cell prints
     "--", because there is no single-disease reference or floor to read against.
  2. The transfer folds. Default is the graph-controlled COVID <-> influenza_us-states pair at 5
     seeds; --ldo3 runs the three-disease fold instead.

Both phases are RESUMABLE. Anything already on disk is skipped, so a crash or a Ctrl-C costs only
the fold that was in flight. Pass --force to redo completed work.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("LOCF_INPUT", "0")          # validated default; see Doubt.md §3.2

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path

LOG = Path("results/reports/overnight.log")


class _Tee:
    """stdout -> console AND the log file, so one command needs no shell pipe."""
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.out = sys.__stdout__

    def write(self, s):
        self.out.write(s); self.out.flush()
        self.f.write(s)

    def flush(self):
        self.out.flush(); self.f.flush()


def _hr(t):
    return f"{t/3600:.2f} h" if t >= 3600 else f"{t/60:.1f} min"


def preflight():
    print(f"python      {sys.executable}")
    try:
        import torch
    except ModuleNotFoundError:
        sys.exit("FATAL: torch is not importable by this interpreter.\n"
                 "       Run with the absolute path to the ebola-train python:\n"
                 '       "C:\\Users\\Administrator\\miniconda3\\envs\\ebola-train\\python.exe" '
                 "-u run_overnight.py")
    import numpy as np
    # the BLAS crash that killed this env: every matmul died, not just corrcoef (Doubt.md §6).
    try:
        assert float((np.ones((4, 4)) @ np.ones((4, 4)))[0, 0]) == 4.0
    except BaseException:
        sys.exit("FATAL: numpy BLAS is broken in this env (a 4x4 matmul failed).\n"
                 '       Fix: conda install -p <env> -c conda-forge "libblas=*=*openblas"')
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if dev == "cuda" else "-"
    print(f"torch       {torch.__version__}  cuda={torch.cuda.is_available()}  {name}")
    print(f"numpy       {np.__version__}  (BLAS ok)")
    if dev != "cuda":
        sys.exit("FATAL: no CUDA. This run is ~6 h on GPU and days on CPU. Refusing to start.")

    import bundles
    print(f"LOCF_INPUT  {bundles.LOCF_INPUT}  (False is the validated default)")
    missing = [n for n in bundles.BUNDLE_NAMES
               if not (bundles.DATA_DIR / f"{n}.npz").exists()]
    if missing:
        sys.exit(f"FATAL: missing bundles {missing}. Build them first (e.g. python -m loaders.covid_load).")
    print(f"bundles     {len(bundles.BUNDLE_NAMES)} present: {bundles.BUNDLE_NAMES}")


def phase1_covid_baselines(force=False):
    """COVID ceiling (5 seeds) + naive floors."""
    import train.loop as L
    from results_paths import rpath
    name = "covid_us-states"
    have = [rpath(f"encoder__{name}__seed{s}.json").exists() for s in L.SEEDS]
    floors = rpath(f"naive__{name}.json").exists()
    if all(have) and floors and not force:
        print(f"\n[1/2] COVID baselines already on disk ({len(L.SEEDS)} seeds + floors). Skipping.")
        return 0.0
    print(f"\n[1/2] COVID baselines: {len(L.SEEDS)} seeds + naive floors "
          f"(have {sum(have)}/{len(L.SEEDS)} seeds, floors={floors})")
    t0 = time.time()
    L.run_dataset(name)                       # writes ceiling for every seed AND the naive floors
    dt = time.time() - t0
    print(f"[1/2] done in {_hr(dt)}")
    return dt


def phase2_transfer(mode, seeds, trunk_steps, epochs, force=False):
    import train.lodo as LD
    from results_paths import rpath
    if mode == "pair":
        folds, runner, prefix = list(LD.PAIR_DISEASES), LD.run_pair_fold, "encoder_pair"
        universe = LD.PAIR_DISEASES
    else:
        folds, runner, prefix = list(LD.LDO3_DISEASES), LD.run_ldo3_fold, "encoder_ldo3"
        universe = LD.DISEASES

    todo = []
    for s in seeds:
        for f in folds:
            done = all(rpath(f"{prefix}__{n}__seed{s}.json").exists() for n in universe[f])
            (todo if (force or not done) else []).append((f, s))
    skipped = len(folds) * len(seeds) - len(todo)
    print(f"\n[2/2] {mode}: {len(folds)} folds x {len(seeds)} seeds = {len(folds)*len(seeds)} runs"
          f"  ({skipped} already on disk, {len(todo)} to run)")
    print(f"      trunk_steps={trunk_steps}  epochs={epochs}  seeds={seeds}")

    t0 = time.time()
    for i, (f, s) in enumerate(todo, 1):
        ts = time.time()
        print(f"\n----- [{i}/{len(todo)}] {mode} fold={f} seed={s} "
              f"({_dt.datetime.now():%H:%M:%S}) -----")
        runner(f, s, trunk_steps=trunk_steps, epochs=epochs)
        el = time.time() - t0
        print(f"----- fold done in {_hr(time.time()-ts)} | elapsed {_hr(el)} | "
              f"projected total {_hr(el/i*len(todo))} -----")
    dt = time.time() - t0
    print(f"[2/2] done in {_hr(dt)}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    # DEFAULT IS LDO3. It was the pair until 2026-08-04, which meant a no-flag invocation silently
    # ran a recommendation instead of the requested experiment. The default is now the thing that
    # was asked for; --pair opts into the graph-controlled side study.
    ap.add_argument("--pair", action="store_true",
                    help="graph-controlled covid <-> influenza_us-states pair instead of LDO3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    ap.add_argument("--trunk-steps", type=int, default=91000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--force", action="store_true", help="redo work already on disk")
    ap.add_argument("--skip-baselines", action="store_true")
    a = ap.parse_args()

    sys.stdout = _Tee(LOG)
    t0 = time.time()
    print("=" * 78)
    print(f"OVERNIGHT RUN  started {_dt.datetime.now():%Y-%m-%d %H:%M:%S}  log -> {LOG}")
    print("=" * 78)
    preflight()

    mode = "pair" if a.pair else "ldo3"
    print(f"\nMODE: {mode}" + ("  (three-disease leave-one-disease-out: hold out dengue, influenza, "
                               "covid in turn;\n      each fold emits an ADAPTED arm and a "
                               "NO-ADAPTER zero-shot arm)" if mode == "ldo3" else ""))
    if mode == "ldo3" and len(a.seeds) == 1:
        print("NOTE: one seed is below the project's dispersion standard (seed sd is 8-14%).")

    d1 = 0.0 if a.skip_baselines else phase1_covid_baselines(a.force)
    d2 = phase2_transfer(mode, a.seeds, a.trunk_steps, a.epochs, a.force)

    print("\n" + "=" * 78)
    print(f"ALL DONE  baselines {_hr(d1)} | {mode} {_hr(d2)} | total {_hr(time.time()-t0)}")
    print(f"finished {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("next: python results_matrix.py -o progress/outcomes/Results_Matrix.md")
    print("=" * 78)


if __name__ == "__main__":
    main()

"""How much of its training budget did each run actually use? (Client task 2, 2026-08-06.)

Every transfer number in the project is a ratio: transfer arm over single-disease ceiling. If BOTH
arms stopped early, both are handicapped and the deficits partly cancel; if only the transfer arm
did, the gap is real. That is a paragraph of the paper either way, and it is answerable from the run
logs alone -- no retraining.

Three budgets are in play and they are NOT the same units:
  single-disease  train/loop.py::train_one        epochs=80,     patience=15 epochs
  LDO3 trunk      train/lodo.py::_fit_trunk       steps=91000,   patience=12 val checks (1 per 1000)
  LDO3 adapter    train/lodo.py::_fit_shared_adapter  epochs=80,  patience=15 epochs

Reported per run: used / budget, the point the KEPT checkpoint was selected, and the tail burned
after it. The tail is the diagnostic -- a tail exactly equal to patience means the run died on the
early-stop branch rather than exhausting its budget.

Runs with no surviving log are printed as MISSING rather than omitted. A silently short table would
read as "we checked and they were fine".

Run as a MODULE from the repo root; the paths below are relative to it.

  python -m diagnostics.epoch_budget_audit                  # all logs under results/reports
  python -m diagnostics.epoch_budget_audit --log path.log   # one log
  python -m diagnostics.epoch_budget_audit --selfcheck      # parser attribution, no logs needed
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

LOGS = Path("results/reports")
SINGLE_EPOCHS, SINGLE_PATIENCE = 80, 15
TRUNK_STEPS, TRUNK_VAL_EVERY, TRUNK_PATIENCE = 91000, 1000, 12
DEV = ("dengue", "influenza_japan", "influenza_us-regions", "influenza_us-states",
       "covid_us-states")
SEEDS = (42, 52, 62, 72, 82)
LDO3_FOLDS = ("dengue", "influenza", "covid")

# run_overnight.py's per-fold banner is the only place mode+fold+seed appear together, so it is what
# attributes an adapter/trunk line to a fold. `shared-adapter[us-states]` alone is ambiguous: it is
# emitted by BOTH the covid LDO3 fold and the influenza_us-states pair fold.
RE_BANNER = re.compile(r"-----\s*\[\d+/\d+\]\s+(ldo3|pair)\s+fold=(\S+)\s+seed=(\d+)")
RE_SINGLE = re.compile(r"^\s*(\S+)\s+seed(\d+)\s+ep(\d+)\s+val_pinball=([\d.]+)\s+best=([\d.]+)")
RE_TRUNK = re.compile(r"trunk\(-(\d+)\)\s+step\s*(\d+)\s+val=([\d.]+)\s+best=([\d.]+)")
RE_ADAPTER = re.compile(r"shared-adapter\[[^\]]*\]\s+ep(\d+)\s+val=([\d.]+)\s+best=([\d.]+)")


class Track:
    """Last point reached, and the point the kept checkpoint was selected at."""

    def __init__(self):
        self.last = None
        self.best_at = None
        self.best = float("inf")

    def see(self, x, val):
        self.last = x
        if val < self.best - 1e-9:
            self.best, self.best_at = val, x


def parse(paths):
    single, trunk, adapter = {}, {}, {}
    ctx = None
    for p in paths:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = RE_BANNER.search(line)
            if m:
                ctx = (m.group(1), m.group(2), int(m.group(3)))
                continue
            m = RE_SINGLE.search(line)
            if m:                                   # phase 1, no fold context by construction
                single.setdefault((m.group(1), int(m.group(2))), Track()) \
                      .see(int(m.group(3)) + 1, float(m.group(4)))
                continue
            m = RE_TRUNK.search(line)
            if m and ctx:
                trunk.setdefault(ctx, Track()).see(int(m.group(2)), float(m.group(3)))
                continue
            m = RE_ADAPTER.search(line)
            if m and ctx:
                adapter.setdefault(ctx, Track()).see(int(m.group(1)) + 1, float(m.group(2)))
    return single, trunk, adapter


def _row(used, budget, best_at, patience, unit):
    if used is None:
        return f"{'MISSING':>12} | {'--':>11} | {'--':>10} | no log survives"
    pct = 100 * used / budget
    tail = used - (best_at or 0)
    why = "EARLY STOP" if used < budget else "ran full budget"
    if used < budget and tail != patience * (TRUNK_VAL_EVERY if unit == "step" else 1):
        why = "stopped early (tail != patience -- check)"
    return (f"{used:>7} /{budget:>5} | {pct:>10.1f}% | {best_at if best_at is not None else '--':>10} "
            f"| {why}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="one log file; default = every *.log under results/reports")
    a = ap.parse_args()
    paths = [Path(a.log)] if a.log else sorted(LOGS.glob("*.log"))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f"no logs found under {LOGS}")
    print(f"logs read: {', '.join(str(p) for p in paths)}\n")
    single, trunk, adapter = parse(paths)

    print("=" * 100)
    print(f"SINGLE-DISEASE CEILING  (train_one: {SINGLE_EPOCHS} epochs, patience {SINGLE_PATIENCE})")
    print("=" * 100)
    print(f"{'dataset':<24} {'seed':>5} | {'epochs used':>12} | {'% budget':>11} | "
          f"{'best at ep':>10} | verdict")
    n_early = n_known = 0
    for ds in DEV:
        for s in SEEDS:
            t = single.get((ds, s))
            used = t.last if t else None
            if used is not None:
                n_known += 1
                n_early += used < SINGLE_EPOCHS
            print(f"{ds:<24} {s:>5} | "
                  f"{_row(used, SINGLE_EPOCHS, t.best_at if t else None, SINGLE_PATIENCE, 'epoch')}")
    print(f"\n  {n_known}/{len(DEV)*len(SEEDS)} runs have a surviving log; "
          f"{n_early} of those early-stopped.")

    print("\n" + "=" * 100)
    print(f"LDO3 TRUNK  (_fit_trunk: {TRUNK_STEPS} steps, patience {TRUNK_PATIENCE} checks "
          f"= {TRUNK_PATIENCE*TRUNK_VAL_EVERY} steps)")
    print("=" * 100)
    print(f"{'fold':<24} {'seed':>5} | {'steps used':>12} | {'% budget':>11} | "
          f"{'best at':>10} | verdict")
    for f in LDO3_FOLDS:
        for s in SEEDS:
            t = trunk.get(("ldo3", f, s))
            print(f"{f:<24} {s:>5} | "
                  f"{_row(t.last if t else None, TRUNK_STEPS, t.best_at if t else None, TRUNK_PATIENCE, 'step')}")

    print("\n" + "=" * 100)
    print(f"LDO3 HELD-OUT ADAPTER  (_fit_shared_adapter: {SINGLE_EPOCHS} epochs, "
          f"patience {SINGLE_PATIENCE})")
    print("=" * 100)
    print(f"{'fold':<24} {'seed':>5} | {'epochs used':>12} | {'% budget':>11} | "
          f"{'best at ep':>10} | verdict")
    for f in LDO3_FOLDS:
        for s in SEEDS:
            t = adapter.get(("ldo3", f, s))
            print(f"{f:<24} {s:>5} | "
                  f"{_row(t.last if t else None, SINGLE_EPOCHS, t.best_at if t else None, SINGLE_PATIENCE, 'epoch')}")


def _selfcheck():
    """The parser attributes lines to the right run, and the ambiguous adapter label is disambiguated
    by the banner rather than by its own text. Paired with a control that must FAIL if the banner is
    ignored -- the same shared-adapter[us-states] label is emitted by two different folds."""
    import tempfile
    log = ("----- [1/2] ldo3 fold=covid seed=42 (01:00:00) -----\n"
           "    trunk(-42) step  1000 val=0.5000 best=0.5000\n"
           "    trunk(-42) step  2000 val=0.6000 best=0.5000\n"
           "    shared-adapter[us-states] ep00 val=0.90 best=0.90\n"
           "    shared-adapter[us-states] ep01 val=0.80 best=0.80\n"
           "----- [2/2] pair fold=influenza_us-states seed=42 (02:00:00) -----\n"
           "    shared-adapter[us-states] ep00 val=0.10 best=0.10\n"
           "  covid_us-states seed42 ep00 val_pinball=0.3000 best=0.3000\n"
           "  covid_us-states seed42 ep01 val_pinball=0.4000 best=0.3000\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.log"
        p.write_text(log, encoding="utf-8")
        single, trunk, adapter = parse([p])
    assert trunk[("ldo3", "covid", 42)].last == 2000
    assert trunk[("ldo3", "covid", 42)].best_at == 1000, "best must be where val was LOWEST"
    # the two folds share an adapter label; only the banner separates them
    assert adapter[("ldo3", "covid", 42)].last == 2, "ldo3 adapter epochs mis-attributed"
    assert adapter[("pair", "influenza_us-states", 42)].last == 1, "pair fold absorbed ldo3 lines"
    assert single[("covid_us-states", 42)].last == 2
    assert single[("covid_us-states", 42)].best_at == 1
    print("ok  parser attributes trunk/adapter/single lines to the right run; the ambiguous "
          "shared-adapter[us-states] label is split by fold banner, not by its own text")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

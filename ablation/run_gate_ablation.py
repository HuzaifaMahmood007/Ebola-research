"""Gate-off ablation: does the spatial graph actually help, or is it decoration? (audit M10)

    conda run --no-capture-output -n ebola-train python ablation/run_gate_ablation.py --selfcheck
    conda run --no-capture-output -n ebola-train python ablation/run_gate_ablation.py --dry
    conda run --no-capture-output -n ebola-train python ablation/run_gate_ablation.py
    conda run --no-capture-output -n ebola-train python ablation/run_gate_ablation.py --report

WHY THIS EXISTS. The inductive spatial channel is the differentiator -- it is what separates this
framework from the temporal-only cross-disease forecasters, and the encoder decision rests on it.
Across all 13,177 records on disk `gate_mode` is `learned` or absent and **never once `off`**, so no
experiment anywhere has asked whether the graph earns its place. We have measured the gate's VALUE
(g = 0.271 japan .. 0.604 dengue, spatial contribution 0.40-0.64) but a gate being open is not the
same claim as a gate being useful. This run supplies the missing half.

WHAT IS COMPARED. Identical trainer, identical seeds, identical 80 epochs and patience, identical
early-stopping monitor; the ONLY difference is `SharedEncoder(gate_mode="off")`, which forces g=0.
The reference is the released single-disease run in results/single/ for the SAME seed, so the delta
is paired and one variable moved. `tests/test_encoder_invariants.py` already proves g=0 reproduces
the graph-free representation exactly, so the arm is what it says it is.

THE CEILING ON THE CLAIM, WHICH MUST TRAVEL WITH ANY NUMBER THIS PRODUCES. g=0 removes message
passing but NOT the LTR degree feature -- `encoder.forward` still adds `ltr(deg)`, so the arm is
graph-free in its mixing and graph-aware in one scalar per node. So this measures the value of
NEIGHBOUR INFORMATION, not of the graph in total. It also cannot separate "structure helps" from
"any adjacency helps": that needs a shuffled-adjacency arm.
# ponytail: gate-off only. Add the shuffled-A arm if this comes back positive -- if the graph does
# not help at all, there is nothing left for a shuffle control to distinguish.

READ IT LIKE THIS.
  gate-off ~= learned (within seed noise)  -> the graph is inert. Report it. It is a real finding
                                              and it retires an unsupported differentiator claim.
  gate-off  worse than learned             -> the graph helps, by the printed margin. This is the
                                              experiment that defends the contribution.
  gate-off  better than learned            -> the mixer is actively harmful and must be reported so.

COST. The four small panels are ~45 min at 5 seeds. Dengue alone is ~12.7 h and runs LAST, so a
night that dies still leaves four panels. Dengue is not optional here the way it was for the epi
bound: it has the densest graph and the highest gate, so it is the panel the claim most needs.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import argparse
import time

import bundles
import train.loop as L
from ablation.run_epi_ablation import (MODEL, POINT, LOWER_BETTER, _seed_from, field,
                                       load_arm, mean_sd, paired_delta)
from results_paths import rpath

TAG = "gateoff"

# Cheapest first, dengue last: it is 96% of the job, so an interrupted night still leaves the four
# small panels readable. Same ordering rule as run_tonight.py.
PANELS = ("influenza_japan", "influenza_us-regions", "influenza_us-states",
          "covid_us-states", "dengue")


def report(datasets, model=MODEL):
    print(f"\n{'=' * 100}\nGATE-OFF ABLATION :: g==0 (no spatial mixing)   model={model}   "
          f"reference = learned gate, SAME seed (paired)\n{'=' * 100}")
    tally = {"gate helps": 0, "gate hurts": 0, "within noise": 0}
    for ds in datasets:
        # rpath at BOTH ends: train.loop writes through it (into ablation/single/), so a reader that
        # globs HERE directly finds nothing and silently reports a finished arm as "not run".
        base = load_arm(rpath(f"encoder__{ds}__seed*.json"), _seed_from, model)
        abl = load_arm(rpath(f"encoder__{ds}__seed*__{TAG}.json", root=HERE), _seed_from, model)
        if not abl:
            print(f"\n  {ds}: no gate-off records -- not run")
            continue
        print(f"\n  {ds}   field={field(ds)}")
        for m in POINT:
            better = "higher" if m == "pcc" else "lower"
            print(f"    {m.upper()} ({better}=better)")
            print(f"      {'h':>3} | {'gate learned':>18} | {'gate off':>18} | "
                  f"{'paired d':>18} | verdict")
            for h in bundles.HORIZONS:
                b, a = base.get((h, m), {}), abl.get((h, m), {})
                if not a:
                    continue
                # BOTH columns over the SHARED seeds only, so off - learned on the printed numbers
                # equals the printed delta (the Week-3 reference-mismatch trap).
                shared = sorted(set(b) & set(a))
                bm, bsd = mean_sd([b[s] for s in shared])
                am, asd = mean_sd([a[s] for s in shared])
                dm, dsd, n = paired_delta(b, a)
                if n < 2 or dsd == 0 or abs(dm) < dsd:
                    sig = "within noise"
                else:
                    # turning the gate OFF got worse => the graph was doing work
                    off_worse = (dm > 0) if m in LOWER_BETTER else (dm < 0)
                    sig = "GATE HELPS" if off_worse else "gate HURTS"
                tally["gate helps" if sig == "GATE HELPS" else
                      "gate hurts" if sig == "gate HURTS" else "within noise"] += 1
                print(f"      {h:>3} | {bm:9.3f} +-{bsd:6.3f} | {am:9.3f} +-{asd:6.3f} | "
                      f"{dm:+9.3f} +-{dsd:6.3f} | {sig} (n={n})")

    tot = sum(tally.values())
    print(f"\n  TALLY over {tot} cells: {tally['gate helps']} the graph helps, "
          f"{tally['gate hurts']} the graph hurts, {tally['within noise']} within noise.")
    print(f"  Paired: each delta is (gate-off - learned) at the SAME seed, then averaged. Both mean "
          f"columns cover ONLY the seeds present in both arms. 'within noise' = |mean| < sd.")
    print(f"  CEILING: g=0 removes neighbour mixing but keeps the LTR degree feature, so this "
          f"bounds the value of NEIGHBOUR INFORMATION, not of the graph in total.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(PANELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(L.SEEDS))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="read what is on disk, train nothing")
    ap.add_argument("--model", default=MODEL, choices=("encoder", "encoder_mc"))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()
    if a.report:
        report(a.datasets, a.model)
        return 0

    # Resume keys on the ARTIFACT, not a flag: a cell is done when its record file exists.
    todo = [(ds, s) for ds in a.datasets for s in a.seeds
            if a.force or not rpath(f"encoder__{ds}__seed{s}__{TAG}.json", root=HERE).exists()]
    print(f"GATE-OFF ABLATION: {len(todo)} of {len(a.datasets) * len(a.seeds)} cells to train")
    if a.dry:
        for ds, s in todo:
            print(f"  DRY {ds} seed{s} -> ablation/single/encoder__{ds}__seed{s}__{TAG}.json")
        return 0
    if not todo:
        print("  nothing to train; --report to read it")
        report(a.datasets, a.model)
        return 0

    L.RESULTS = HERE                                   # send write_records / write_per_node here
    for ds, s in todo:
        t0 = time.time()
        recs, pernode, _, _ = L.train_one(ds, s, epochs=a.epochs, verbose=False, gate_mode="off")
        # The arm must actually BE the arm. train.loop stamps gate_mode into every record, so if a
        # future refactor drops the kwarg this fails here rather than printing a duplicate baseline
        # under an ablation filename -- the failure mode the epi run hit as an all-zero penalty.
        bad = {r.get("gate_mode") for r in recs} - {"off"}
        assert not bad, f"{ds} seed{s}: records claim gate_mode={bad}, expected 'off'"
        L.write_records(recs, f"encoder__{ds}__seed{s}__{TAG}.json")
        L.write_per_node(pernode, f"encoder__{ds}__seed{s}__{TAG}__pernode.npz")
        print(f"  {ds} seed{s} done in {(time.time() - t0) / 60:.1f} min")

    report(a.datasets, a.model)
    return 0


def _selfcheck():
    """The two things that would silently produce a wrong table: a mislabelled arm, and a verdict
    whose direction is inverted."""
    from models.encoder import SharedEncoder

    # 1. gate_mode='off' really zeroes the gate, and 'learned' really does not -- otherwise both
    #    arms are the same model and every cell reads 'within noise' for the wrong reason.
    import torch
    h = torch.randn(7, 64)
    assert float(SharedEncoder(gate_mode="off").gate.g(h).abs().max()) == 0.0, "g must be 0 when off"
    assert float(SharedEncoder(gate_mode="learned").gate.g(h).abs().max()) > 0.0, \
        "CONTROL: the learned gate must be non-zero, or 'off' is not removing anything"

    # 2. verdict direction. Turning the gate off and getting WORSE means the graph was helping.
    #    An inverted sign here would report the exact opposite of the finding, in both directions.
    def verdict(dm, dsd, m):
        if dsd == 0 or abs(dm) < dsd:
            return "within noise"
        off_worse = (dm > 0) if m in LOWER_BETTER else (dm < 0)
        return "GATE HELPS" if off_worse else "gate HURTS"

    assert verdict(+5.0, 1.0, "rmse") == "GATE HELPS", "rmse up with gate off => graph helped"
    assert verdict(-5.0, 1.0, "rmse") == "gate HURTS", "rmse down with gate off => mixer hurt"
    assert verdict(-0.05, 0.01, "pcc") == "GATE HELPS", "pcc down with gate off => graph helped"
    assert verdict(+0.05, 0.01, "pcc") == "gate HURTS", "pcc up with gate off => mixer hurt"
    assert verdict(+5.0, 9.0, "rmse") == "within noise", "a delta inside its own spread is noise"

    # 3. reader and writer must agree on the directory, and the tag must not collide with the
    #    released baseline glob (encoder__ds__seed*.json also matches the ablation filenames).
    p = rpath(f"encoder__dengue__seed42__{TAG}.json", root=HERE)
    assert p.parent == HERE / "single", f"records route to {p.parent}; globbing HERE would miss them"
    assert _seed_from(f"encoder__dengue__seed52__{TAG}.json") == 52, "seed must survive the suffix"
    assert rpath("encoder__dengue__seed42.json").parent != p.parent, \
        "baseline and ablation must live in different trees or the reference reads itself"

    print("selfcheck ok: gate off zeroes g (and learned does not), verdict signs correct in both "
          "metric directions, records route to ablation/single/ and cannot read the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Shuffled-adjacency RETRAIN (ledger D2): does the graph earn its place DURING LEARNING?

    conda run --no-capture-output -n ebola-train python ablation/run_shuffle_adjacency.py --selfcheck
    conda run --no-capture-output -n ebola-train python ablation/run_shuffle_adjacency.py --smoke
    conda run --no-capture-output -n ebola-train python ablation/run_shuffle_adjacency.py --dry
    conda run --no-capture-output -n ebola-train python ablation/run_shuffle_adjacency.py            # small panels, dengue last
    conda run --no-capture-output -n ebola-train python ablation/run_shuffle_adjacency.py --report

WHY THIS EXISTS. Ledger D2 (`progress/planning/Gap_Ledger.md:276`, reopened 2026-09-21). Two spatial
controls have already run and both say the graph does not help ACCURACY: the gate-off ablation (0 of
40 error cells helped) and the inference-only relabelling probe (permuting the adjacency's node labels
on the ALREADY-TRAINED model costs under 1% -- `diagnostics/graph_probe/why_graph_fails.py` STEP 3,
`Session_Audit_2026-09-10.md` section C). Both leave one question open. They act on a model that
learned on the REAL graph. Neither can say whether a model that TRAINED on a fake graph would have
done just as well. That is the only remaining spatial test, and it is what closes the
graph-does-not-help finding for the final paper.

WHAT IS COMPARED. Identical trainer, identical seeds, identical 80 epochs and patience, identical
early-stopping monitor. The ONLY difference is that the adjacency's node labels are permuted once, at
model construction, and held fixed for the entire run (`train.loop.permute_adjacency`, wired through
the new `shuffle_adj_seed` kwarg). Same topology, same degree sequence, WRONG districts. Node
features/targets/masks stay in their original order, so each node keeps its own history and is handed a
stranger's neighbours. The reference is the released single-disease run in results/single/ for the
SAME seed, so the delta is paired and exactly one variable moved.

WHAT THE PERMUTATION PRESERVES, AND WHY THAT MATTERS. A symmetric row+column relabel A[p][:, p] keeps
the edge count and the whole degree multiset unchanged (asserted in permute_adjacency, both unweighted
and weighted). So this arm cannot be explained away as "fewer edges" or "different sparsity": the model
sees a graph with the same connectivity statistics and only the identities scrambled. The permutation
is seeded (SHUF_BASE + train seed) and its hash + displaced fraction are stamped into every record;
identity permutations are REFUSED, because a no-op shuffle would silently reproduce the real-graph run
and manufacture a false null.

THE CEILING ON THE CLAIM, WHICH MUST TRAVEL WITH ANY NUMBER THIS PRODUCES. The LTR degree feature
(`models/encoder.py`, `models/spatial.py:LTR`) reads D~ from the permuted adjacency, and degree is
permutation-invariant, so the degree scalar is UNCHANGED by the shuffle. This arm therefore isolates
the value of NEIGHBOUR IDENTITY (message passing over the right districts), not of the graph's degree
information. That is the same ceiling the gate-off arm carries, stated the same way.

ANALYSIS PLAN -- how to read the result, fixed here so nobody improvises at scoring time. Read exactly
as the gate-off ablation is read (`ablation/run_gate_ablation.py`). Each cell is one (panel, horizon,
metric). The delta is (shuffle - real) at the SAME seed, averaged over the seeds present in BOTH arms;
both printed mean columns cover only those shared seeds, so shuffle_mean - real_mean reproduces the
delta exactly. The within-noise rule is identical: |paired mean| < its across-seed sd => "within
noise". Three outcomes, three verdicts:
  shuffle ~= real (within noise)  -> the graph is inert DURING LEARNING too. A model trained on the
                                     wrong districts is as good as one trained on the right ones, so
                                     the real adjacency carried no usable district information at any
                                     stage. This CLOSES D2 consistent with gate-off and the relabel
                                     probe, and it is the expected outcome given both.
  shuffle worse than real         -> training on the real graph buys something inference-time controls
                                     could not see: the graph helped while learning, by the printed
                                     margin. This is a NEW positive and must be reported as one; it
                                     would partially walk back the graph-does-not-help finding.
  shuffle better than real        -> the real adjacency is actively misleading during learning and the
                                     shuffle regularises. Report it as such.
The headline count is the tally over all cells, exactly like gate-off's. Dengue is one panel of five
and is NOT down-weighted; it has the densest graph and the highest gate, so it is the panel the claim
most needs.

COST. Same trainer and epochs as the single-disease ceiling, so the per-panel cost tracks
REPRODUCIBILITY.md:220 -- the three influenza panels are ~35 min combined at 5 seeds and covid is a
comparable small panel (four small panels ~45 min at 5 seeds), while dengue alone is ~12.7 h at 5
seeds because of its 7,165 nodes. The full 5-panel x 5-seed matrix is therefore ~13.5 h, ~94% of it
dengue. Dengue runs LAST for that reason: an interrupted night still leaves four readable panels. Same
cheapest-first ordering as run_gate_ablation.py.
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
import json
import time

import numpy as np

import bundles
import train.loop as L
from ablation.run_epi_ablation import (MODEL, POINT, LOWER_BETTER, _seed_from, field,
                                       load_arm, mean_sd, paired_delta)
from results_paths import rpath

TAG = "shufadj"

# Permutation seed = SHUF_BASE + train seed. Kept SEPARATE from the train seed (which seeds torch/np
# init) so the shuffle is reproducible without perturbing training reproducibility, and DISTINCT per
# seed so the five runs do not all see the same fake graph. 20260921 is the D2 reopen date.
SHUF_BASE = 20260921

# Cheapest first, dengue last: it is ~94% of the job, so an interrupted night still leaves the four
# small panels readable. Same ordering rule as run_gate_ablation.py / run_tonight.py.
PANELS = ("influenza_japan", "influenza_us-regions", "influenza_us-states",
          "covid_us-states", "dengue")

# The 13 keys a real gate-off / single-disease encoder record carries. The shufadj record adds three
# (shuffle_adj_seed/hash/frac_displaced), so this is a required-keys-present subset check, exactly the
# shape of run_classical._assert_schema.
_GATEOFF_REF = HERE / "single" / "encoder__influenza_japan__seed42__gateoff.json"


def perm_seed_for(seed):
    return SHUF_BASE + int(seed)


def report(datasets, model=MODEL):
    print(f"\n{'=' * 100}\nSHUFFLED-ADJACENCY RETRAIN :: wrong districts, same topology+degrees   "
          f"model={model}   reference = real graph, SAME seed (paired)\n{'=' * 100}")
    tally = {"real graph helps": 0, "real graph hurts": 0, "within noise": 0}
    for ds in datasets:
        # rpath at BOTH ends: train.loop writes through it (into ablation/single/), so a reader that
        # globs HERE directly finds nothing and silently reports a finished arm as "not run".
        base = load_arm(rpath(f"encoder__{ds}__seed*.json"), _seed_from, model)
        abl = load_arm(rpath(f"encoder__{ds}__seed*__{TAG}.json", root=HERE), _seed_from, model)
        if not abl:
            print(f"\n  {ds}: no shuffled-adjacency records -- not run")
            continue
        print(f"\n  {ds}   field={field(ds)}")
        for m in POINT:
            better = "higher" if m == "pcc" else "lower"
            print(f"    {m.upper()} ({better}=better)")
            print(f"      {'h':>3} | {'real graph':>18} | {'shuffled A':>18} | "
                  f"{'paired d':>18} | verdict")
            for h in bundles.HORIZONS:
                b, a = base.get((h, m), {}), abl.get((h, m), {})
                if not a:
                    continue
                # BOTH columns over the SHARED seeds only, so shuffle - real on the printed numbers
                # equals the printed delta (the Week-3 reference-mismatch trap).
                shared = sorted(set(b) & set(a))
                bm, bsd = mean_sd([b[s] for s in shared])
                am, asd = mean_sd([a[s] for s in shared])
                dm, dsd, n = paired_delta(b, a)
                if n < 2 or dsd == 0 or abs(dm) < dsd:
                    sig = "within noise"
                else:
                    # scrambling districts got WORSE => the real graph was doing work while learning
                    shuf_worse = (dm > 0) if m in LOWER_BETTER else (dm < 0)
                    sig = "REAL GRAPH HELPS" if shuf_worse else "real graph HURTS"
                tally["real graph helps" if sig == "REAL GRAPH HELPS" else
                      "real graph hurts" if sig == "real graph HURTS" else "within noise"] += 1
                print(f"      {h:>3} | {bm:9.3f} +-{bsd:6.3f} | {am:9.3f} +-{asd:6.3f} | "
                      f"{dm:+9.3f} +-{dsd:6.3f} | {sig} (n={n})")

    tot = sum(tally.values())
    print(f"\n  TALLY over {tot} cells: {tally['real graph helps']} the real graph helps learning, "
          f"{tally['real graph hurts']} the real graph hurts, {tally['within noise']} within noise.")
    print(f"  Paired: each delta is (shuffled - real) at the SAME seed, then averaged. Both mean "
          f"columns cover ONLY the seeds present in both arms. 'within noise' = |mean| < sd.")
    print(f"  CEILING: the LTR degree feature is permutation-invariant, so this bounds the value of "
          f"NEIGHBOUR IDENTITY, not of the graph in total (same ceiling as gate-off).\n")


def _train_cell(ds, s, epochs, tag=TAG):
    """One shuffled-adjacency cell: train, guard the arm label, write records + per-node archive.
    Returns the record list so the smoke path can schema-check it."""
    ps = perm_seed_for(s)
    recs, pernode, _, _ = L.train_one(ds, s, epochs=epochs, verbose=False, shuffle_adj_seed=ps)
    # The arm must actually BE the arm. train.loop stamps the shuffle meta into every record, so if a
    # future refactor drops the kwarg this fails here rather than writing a real-graph baseline under
    # a shufadj filename -- the failure mode the epi run hit as an all-zero penalty.
    seeds_seen = {r.get("shuffle_adj_seed") for r in recs}
    assert seeds_seen == {ps}, f"{ds} seed{s}: records carry shuffle_adj_seed={seeds_seen}, want {ps}"
    fracs = {r.get("shuffle_adj_frac_displaced") for r in recs}
    assert all(f and f > 0.0 for f in fracs), \
        f"{ds} seed{s}: a record has zero/absent displacement -- the shuffle was a no-op"
    L.write_records(recs, f"encoder__{ds}__seed{s}__{tag}.json")
    L.write_per_node(pernode, f"encoder__{ds}__seed{s}__{tag}__pernode.npz")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(PANELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(L.SEEDS))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="read what is on disk, train nothing")
    ap.add_argument("--model", default=MODEL, choices=("encoder", "encoder_mc"))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()
    if a.smoke:
        return _smoke()
    if a.report:
        report(a.datasets, a.model)
        return 0

    # Resume keys on the ARTIFACT, not a flag: a cell is done when its record file exists.
    todo = [(ds, s) for ds in a.datasets for s in a.seeds
            if a.force or not rpath(f"encoder__{ds}__seed{s}__{TAG}.json", root=HERE).exists()]
    print(f"SHUFFLED-ADJACENCY RETRAIN: {len(todo)} of {len(a.datasets) * len(a.seeds)} cells to train")
    if a.dry:
        for ds, s in todo:
            print(f"  DRY {ds} seed{s} (perm_seed={perm_seed_for(s)}) -> "
                  f"ablation/single/encoder__{ds}__seed{s}__{TAG}.json")
        return 0
    if not todo:
        print("  nothing to train; --report to read it")
        report(a.datasets, a.model)
        return 0

    L.RESULTS = HERE                                   # send write_records / write_per_node here
    done = 0
    for ds, s in todo:
        t0 = time.time()
        recs = _train_cell(ds, s, a.epochs)
        done += 1
        dt = (time.time() - t0) / 60
        frac = next(iter({r["shuffle_adj_frac_displaced"] for r in recs}))
        print(f"  {ds} seed{s} done in {dt:.1f} min  (displaced {frac:.1%} of nodes)  "
              f"[{done}/{len(todo)} cells, ETA ~{dt * (len(todo) - done):.0f} min at this rate]")

    report(a.datasets, a.model)
    return 0


def _smoke():
    """Smallest panel, one seed, truncated epochs: end to end through scoring and record writing.
    Verifies the record schema against a real gate-off record and that the permutation moved labels."""
    t0 = time.time()
    L.RESULTS = HERE
    ds, s = "influenza_japan", 42
    print(f"smoke: {ds} seed{s}, perm_seed={perm_seed_for(s)}, 3 epochs")
    recs = _train_cell(ds, s, epochs=3, tag=f"{TAG}__smoke")

    # 1. schema: every emitted record must carry AT LEAST the gate-off record's key set (extras ok).
    ref = json.loads(_GATEOFF_REF.read_text())[0]
    want = set(ref)
    got = set(recs[0])
    assert want <= got, f"schema drift vs {_GATEOFF_REF.name}: missing required keys {want - got}"
    extra = got - want
    assert extra == {"shuffle_adj_seed", "shuffle_adj_hash", "shuffle_adj_frac_displaced"}, \
        f"unexpected extra keys {extra}"
    print(f"  schema OK: {len(want)} gate-off keys present; extras = {sorted(extra)}")

    # 2. the arm is stamped and the permutation actually moved labels (identity would be a silent null)
    frac = recs[0]["shuffle_adj_frac_displaced"]
    assert frac > 0.0, "displacement is zero -- the permutation was the identity"
    print(f"  arm OK: shuffle_adj_seed={recs[0]['shuffle_adj_seed']}, "
          f"hash={recs[0]['shuffle_adj_hash']}, displaced {frac:.1%} of nodes")

    # 3. gate-off value on disk vs the smoke record, so a reader sees they share a schema
    print(f"  reference record: {_GATEOFF_REF.name} ({len(want)} keys)")
    print(f"smoke done in {time.time() - t0:.0f}s; wrote {len(recs)} records to "
          f"ablation/single/encoder__{ds}__seed{s}__{TAG}__smoke.json")
    return 0


def _selfcheck():
    """The two things that would silently produce a wrong table: a fake shuffle (identity, or one that
    changes the graph's statistics) and a verdict whose direction is inverted."""
    from train.loop import permute_adjacency

    # 1. the permutation is a PURE relabel: edge count and degree sequence preserved, labels moved.
    rng = np.random.default_rng(0)
    N = 40
    Araw = (rng.random((N, N)) < 0.15).astype(float)
    Araw = np.maximum(Araw, Araw.T)                      # symmetric, like a geo adjacency
    np.fill_diagonal(Araw, 0.0)
    Ap, meta = permute_adjacency(Araw, perm_seed=123)
    assert int((Ap != 0).sum()) == int((Araw != 0).sum()), "edge count changed -- not a pure relabel"
    assert np.array_equal(np.sort((Ap != 0).sum(1)), np.sort((Araw != 0).sum(1))), \
        "degree sequence changed -- not a pure relabel"
    assert meta["shuffle_adj_frac_displaced"] > 0.0, "a real shuffle must displace at least one node"
    assert not np.array_equal(Ap, Araw), "the permuted graph is identical -- nothing was shuffled"

    # 1b. identity is REFUSED. A permutation seed that yields the identity must raise, or a no-op arm
    #     silently reproduces the real-graph run and fakes a null. Search for such a seed on tiny N.
    caught = False
    for _seed in range(500):
        p = np.random.default_rng(_seed).permutation(2)
        if (p == np.arange(2)).all():
            try:
                permute_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]]), perm_seed=_seed)
            except AssertionError:
                caught = True
            break
    assert caught, "identity permutation was NOT refused -- a no-op arm could fake a null result"

    # 1c. reproducible: same seed => same permutation hash; different seed => (almost surely) different.
    _, m1 = permute_adjacency(Araw, perm_seed=123)
    _, m2 = permute_adjacency(Araw, perm_seed=124)
    assert m1["shuffle_adj_hash"] == meta["shuffle_adj_hash"], "same seed gave a different permutation"
    assert m1["shuffle_adj_hash"] != m2["shuffle_adj_hash"], "different seeds gave the same permutation"

    # 2. verdict direction. Scrambling districts and getting WORSE means the real graph was helping.
    #    An inverted sign here would report the exact opposite of the finding, in both directions.
    def verdict(dm, dsd, m):
        if dsd == 0 or abs(dm) < dsd:
            return "within noise"
        shuf_worse = (dm > 0) if m in LOWER_BETTER else (dm < 0)
        return "REAL GRAPH HELPS" if shuf_worse else "real graph HURTS"

    assert verdict(+5.0, 1.0, "rmse") == "REAL GRAPH HELPS", "rmse up with shuffle => real graph helped"
    assert verdict(-5.0, 1.0, "rmse") == "real graph HURTS", "rmse down with shuffle => real graph hurt"
    assert verdict(-0.05, 0.01, "pcc") == "REAL GRAPH HELPS", "pcc down with shuffle => real graph helped"
    assert verdict(+0.05, 0.01, "pcc") == "real graph HURTS", "pcc up with shuffle => real graph hurt"
    assert verdict(+5.0, 9.0, "rmse") == "within noise", "a delta inside its own spread is noise"

    # 3. reader and writer must agree on the directory, and the tag must not collide with the released
    #    baseline glob (encoder__ds__seed*.json also matches the ablation filenames).
    p = rpath(f"encoder__dengue__seed42__{TAG}.json", root=HERE)
    assert p.parent == HERE / "single", f"records route to {p.parent}; globbing HERE would miss them"
    assert _seed_from(f"encoder__dengue__seed52__{TAG}.json") == 52, "seed must survive the suffix"
    assert rpath("encoder__dengue__seed42.json").parent != p.parent, \
        "baseline and ablation must live in different trees or the reference reads itself"

    # 4. perm seed is distinct from the train seed, so the shuffle does not alias training randomness
    assert perm_seed_for(42) != 42 and perm_seed_for(42) != perm_seed_for(52), \
        "perm seed must be distinct from the train seed and distinct across seeds"

    print("selfcheck ok: shuffle is a pure relabel (edges+degrees preserved, labels moved), identity "
          "refused, reproducible by seed, verdict signs correct both directions, records route to "
          "ablation/single/ and cannot read the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())

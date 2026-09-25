"""Train the epi-informed arm and read it against the single-disease baseline. (Task 14.3 / P3)

    conda run --no-capture-output -n ebola-train python ablation/run_epi_ablation.py --dry
    conda run --no-capture-output -n ebola-train python ablation/run_epi_ablation.py --selfcheck
    conda run --no-capture-output -n ebola-train python ablation/run_epi_ablation.py \
        --datasets influenza_japan influenza_us-regions influenza_us-states covid_us-states
    conda run --no-capture-output -n ebola-train python ablation/run_epi_ablation.py --report

WHAT IS COMPARED. The identical trainer, identical seeds, identical early stopping (val pinball in
both arms -- the penalty enters the TRAIN objective only), against the single-disease runs already in
results/single/. Deltas are PAIRED per seed and the reference is named on the table, because an
arm-mean against a baseline-mean is the exact reference mismatch the client caught in Week 3: a seed
present in one arm and missing from the other silently shifts the comparison.

THE BOUND IS A CHOICE AND IT IS ON THE FILENAME. `--quantile`/`--aggregator` pick r_max, the tag
`epi_p{q}{agg}` goes into every artifact name, and two bounds therefore never overwrite each other.
Run `epi_penalty.py --sweep` first: it prints, for each bound, what fraction of REAL transitions the
bound calls implausible against what fraction of the model's own intervals it would touch. If the
second number is ~0 the arm cannot move and the run is not worth the GPU.

COST. Four small panels are ~45 min for 5 seeds. Dengue alone is ~12.7 h for 5 seeds, 96% of the
job, so it is a separate decision and `--datasets` is how you make it.
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
import glob
import hashlib
import json
import math
import subprocess
import time

import bundles
import train.loop as L
from ablation.epi_penalty import DEFAULT_LAMBDA, GAPS, calibrate
from results_paths import rpath

# New arms write HERE-independently, into their own tracked tree, so the 40 released lambda-1 records
# in ablation/single/ are never touched. rpath still routes on the `encoder__` prefix, so the files
# land in experiments/epi_bound_lambda/single/ -- that `single/` level is routing, not a choice.
EXPDIR = ROOT / "experiments" / "epi_bound_lambda"
PROTOCOL = ROOT / "progress" / "decisions" / "Epi_Bound_Lambda_Protocol.md"

HEADLINE = {"dengue": "country_macro"}
POINT = ("rmse", "mae", "pcc")
LOWER_BETTER = ("rmse", "mae")


def field(ds):
    return HEADLINE.get(ds, "node_mean")


def tag(q, aggregator, lam=DEFAULT_LAMBDA):
    """Artifact suffix for one arm. LAMBDA IS IN IT, because two lambda arms at the same bound
    otherwise collide on filename: without --force the second silently skips and the report reads
    stale lambda-1 records under the new arm's name, and with --force it overwrites and the first arm
    is gone. Neither failure announces itself.

    Backward compatible by construction -- lam == DEFAULT_LAMBDA reproduces `epi_p99max` and
    `epi_p90max` byte for byte, so the 40 released records still resolve."""
    base = f"epi_p{int(round(q * 100))}{aggregator}"
    return base if lam == DEFAULT_LAMBDA else f"{base}_lam{lam:g}"


def mean_sd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def paired_delta(base_by_seed, abl_by_seed):
    """(mean, sd, n) of the per-seed difference abl - base, over seeds present in BOTH arms.

    Pairing on the intersection is the whole point: a seed that only one arm has must be dropped,
    not folded into that arm's mean and compared across."""
    seeds = sorted(set(base_by_seed) & set(abl_by_seed))
    d = [abl_by_seed[s] - base_by_seed[s] for s in seeds]
    m, sd = mean_sd(d)
    return m, sd, len(d)


MODEL = "encoder"          # each record file also holds "encoder_mc", the bias-corrected forecast


def load_arm(pattern, key_seed, model=MODEL):
    """{(horizon, metric): {seed: value}} from a glob of record files, for ONE model variant.

    Filtering on the model is not optional. train.loop writes `recs + mcrecs`, so every file holds
    both `encoder` and `encoder_mc` at the same (horizon, metric); a key without the model lets the
    later record overwrite the earlier one, and the table then silently reports whichever variant
    happens to be written last. lodo._load_recs and compare_runs both key on the model; this used
    not to, which is how it printed mean-corrected levels under a bare 'baseline' heading."""
    out = {}
    for f in sorted(glob.glob(str(pattern))):
        if "smoke" in Path(f).name:
            continue
        s = key_seed(Path(f).name)
        if s is None:
            continue
        for r in json.load(open(f)):
            if r["model"] != model:
                continue
            out.setdefault((r["horizon"], r["metric"]), {})[s] = r[field(r["dataset"])]
    return out


def _seed_from(name):
    for part in name.replace(".json", "").split("__"):
        if part.startswith("seed"):
            try:
                return int(part[4:])
            except ValueError:
                return None
    return None


ARM_ROOTS = (EXPDIR, HERE)     # the new tree first, the released ablation/ tree second


def load_abl(ds, tg, model=MODEL, roots=ARM_ROOTS):
    """One arm's records, from EITHER tree.

    New arms land under experiments/epi_bound_lambda/, the released lambda-1 arms are in ablation/.
    Both must be readable or the lambda ladders have no base and the bound comparison has no
    reference. A plain merge needs no precedence rule because the tags cannot overlap: `epi_p90max`
    only ever exists in ablation/ and `epi_p90max_lam10` only in the new tree."""
    out = {}
    for r in roots:
        found = load_arm(rpath(f"encoder__{ds}__seed*__{tg}.json", root=r), _seed_from, model)
        for k, v in found.items():
            out.setdefault(k, {}).update(v)
    return out


def protocol_sha():
    """sha256 of the frozen protocol, refusing if it is missing or carries uncommitted edits.

    A pre-registration is only worth the name if the file was committed BEFORE any number existed.
    Stamped into every record, so a reader can tell which protocol scored it -- and so a record
    written against an edited protocol is detectable after the fact."""
    assert PROTOCOL.exists(), \
        f"{PROTOCOL.relative_to(ROOT)} missing -- pre-register the criterion before training"
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(PROTOCOL)],
                           cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    assert not dirty, (f"{PROTOCOL.name} has uncommitted changes -- commit the protocol before any "
                       f"number exists, or it is not frozen ({dirty})")
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def report(datasets, tg, model=MODEL):
    print(f"\n{'=' * 100}\nEPI-INFORMED ABLATION :: bound {tg}   model={model}   reference = "
          f"single-disease, SAME seed (paired)\n{'=' * 100}")
    any_rows = False
    for ds in datasets:
        # rpath at BOTH ends: train.loop writes through it (into ablation/single/), so a reader that
        # globs HERE directly finds nothing and silently reports "not run".
        base = load_arm(rpath(f"encoder__{ds}__seed*.json"), _seed_from, model)
        abl = load_abl(ds, tg, model)
        if not abl:
            print(f"\n  {ds}: no ablation records for bound {tg} -- not run")
            continue
        any_rows = True
        print(f"\n  {ds}   field={field(ds)}")
        for m in POINT:
            better = "higher" if m == "pcc" else "lower"
            print(f"    {m.upper()} ({better}=better)")
            print(f"      {'h':>3} | {'baseline':>18} | {'epi':>18} | {'paired d':>18} | verdict")
            for h in bundles.HORIZONS:
                b, a = base.get((h, m), {}), abl.get((h, m), {})
                if not a:
                    continue
                # BOTH columns over the SHARED seeds only, so epi - baseline on the printed numbers
                # equals the printed delta. Showing a 5-seed baseline mean beside a 1-seed epi mean
                # and a paired delta is the Week-3 reference mismatch reprinted in a new table.
                shared = sorted(set(b) & set(a))
                bm, bsd = mean_sd([b[s] for s in shared])
                am, asd = mean_sd([a[s] for s in shared])
                dm, dsd, n = paired_delta(b, a)
                # within noise unless the paired mean clears its own spread across seeds
                sig = "within noise" if (n < 2 or dsd == 0 or abs(dm) < dsd) else (
                    "epi BETTER" if (dm < 0) == (m in LOWER_BETTER) else "epi WORSE")
                print(f"      {h:>3} | {bm:9.3f} +-{bsd:6.3f} | {am:9.3f} +-{asd:6.3f} | "
                      f"{dm:+9.3f} +-{dsd:6.3f} | {sig} (n={n})")
    if not any_rows:
        print("\n  nothing to report yet.")
    print(f"\n  Paired: each delta is (epi - baseline) at the SAME seed, then averaged. Both mean "
          f"columns cover ONLY the seeds present in both arms, so epi - baseline reproduces the "
          f"delta column exactly. 'within noise' = |mean| < sd across seeds.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(bundles.DEV_BUNDLE_NAMES))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(L.SEEDS))
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--aggregator", default="max", choices=("max", "median", "min"))
    ap.add_argument("--calib-datasets", nargs="+", default=list(bundles.DEV_BUNDLE_NAMES),
                    help="datasets r_max is calibrated on. Deliberately NOT --datasets: the bound "
                         "is a property of the dev set, so training one panel must not silently "
                         "recalibrate it and land under a filename claiming the dev-wide value")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="read what is on disk, train nothing")
    ap.add_argument("--model", default=MODEL, choices=("encoder", "encoder_mc"),
                    help="which forecast to read: raw, or the bias-corrected one")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--mutate-selfcheck", action="store_true",
                    help="plant the tag bugs the selfcheck claims to catch and require it to fail")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()
    if a.mutate_selfcheck:
        return _mutate_selfcheck()

    tg = tag(a.quantile, a.aggregator, a.lam)
    if a.report:
        report(a.datasets, tg, a.model)
        return 0

    # rpath, not HERE/fname: records route into a `single/` subdir, so the old check looked for a path
    # that could never exist and every run silently retrained every cell. Fixing it is what makes the
    # overnight chain restartable after a crash.
    todo = [(ds, s) for ds in a.datasets for s in a.seeds
            if a.force or not rpath(f"encoder__{ds}__seed{s}__{tg}.json", root=EXPDIR).exists()]
    print(f"EPI ABLATION: bound {tg}, lambda {a.lam}, {len(todo)} of "
          f"{len(a.datasets) * len(a.seeds)} cells to train")
    if a.dry:
        for ds, s in todo:
            print(f"  DRY {ds} seed{s} -> "
                  f"{rpath(f'encoder__{ds}__seed{s}__{tg}.json', root=EXPDIR).relative_to(ROOT)}")
        return 0
    if not todo:
        print("  nothing to train; --report to read it")
        return 0

    r_max, _ = calibrate(a.calib_datasets, a.quantile, verbose=True, aggregator=a.aggregator)
    print(f"  r_max = " + "  ".join(f"gap {g}: {r_max[g]:.4f}" for g in GAPS)
          + f"   (calibrated on {len(a.calib_datasets)} datasets)")

    sha = protocol_sha()                               # refuses unless the protocol is committed
    print(f"  protocol {PROTOCOL.name} sha256 {sha[:16]}... (frozen)")

    L.RESULTS = EXPDIR                                 # send write_records / write_per_node here
    fired, shares = {}, {}
    for ds, s in todo:
        t0 = time.time()
        epi = dict(lam=a.lam, r_max=r_max)
        recs, pernode, _, _ = L.train_one(ds, s, epochs=a.epochs, verbose=False, epi=epi)
        tot, n, pin = epi.get("_seen", [0.0, 0, 0.0])
        pen_mean, pin_mean = tot / max(n, 1), pin / max(n, 1)
        # THE inertness number. A null from a term worth 0.005% of the objective is a statement about
        # lambda, not about the epi component, and the protocol's 1% gate is adjudicated on this.
        share = (a.lam * pen_mean / pin_mean) if pin_mean > 0 else float("nan")
        for r in recs:
            r.update(epi_quantile=a.quantile, epi_aggregator=a.aggregator, epi_lam=a.lam,
                     epi_r_max={str(g): round(r_max[g], 6) for g in GAPS},
                     epi_penalty_mean=pen_mean, epi_pinball_mean=pin_mean,
                     epi_penalty_share=share, epi_protocol_sha256=sha)
        L.write_records(recs, f"encoder__{ds}__seed{s}__{tg}.json")
        L.write_per_node(pernode, f"encoder__{ds}__seed{s}__{tg}__pernode.npz")
        fired[(ds, s)], shares[(ds, s)] = pen_mean, share
        print(f"  {ds} seed{s} done in {(time.time()-t0)/60:.1f} min  "
              f"mean train penalty = {pen_mean:.3e}  pinball = {pin_mean:.4f}  "
              f"lam*pen/pinball = {100*share:.4f}%")

    dead = [k for k, v in fired.items() if v == 0.0]
    if dead:
        print(f"\n  WARNING: the penalty was IDENTICALLY ZERO on {len(dead)} of {len(fired)} cells "
              f"({dead[:4]}{'...' if len(dead) > 4 else ''}).")
        print("  Those cells are the baseline retrained, not an ablation: report them as an inert "
              "penalty, not as 'no effect of the epi component'.")
    weak = [k for k, v in shares.items() if v == v and v < 0.01 and k not in dead]
    if weak:
        print(f"\n  INERT-BY-WEAKNESS: lam*penalty is under 1% of the pinball loss on {len(weak)} of "
              f"{len(shares)} cells. Per the protocol those nulls are INCONCLUSIVE, not FAIL: the "
              f"term was too weak to steer training, which is a statement about lambda.")
        for k in sorted(weak)[:8]:
            print(f"    {k[0]} seed{k[1]}: {100*shares[k]:.4f}%")
    report(a.datasets, tg, a.model)
    return 0


def _mutate_selfcheck():
    """The check on the check. Plant the bug each new assertion exists to catch and require a FAIL.

    A selfcheck that still passes against a broken tag() is decoration, and the tag bug is exactly the
    kind that leaves no trace in a log.

    Patches globals(), NOT `import ablation.run_epi_ablation`. Run as a script this file is __main__,
    so importing it by package name builds a SECOND module object and rebinding its `tag` leaves the
    running _selfcheck on the good one -- every mutation then reports MISSED for the wrong reason.
    (That is not hypothetical: the first version of this function did exactly that.)"""
    g = globals()
    good = g["tag"]
    muts = {
        "lambda-blind tag (the original bug)":
            lambda q, ag, lam=DEFAULT_LAMBDA: f"epi_p{int(round(q * 100))}{ag}",
        "lambda always appended (the 40 released records stop resolving)":
            lambda q, ag, lam=DEFAULT_LAMBDA: f"epi_p{int(round(q * 100))}{ag}_lam{lam:g}",
        "aggregator dropped (median and max arms collide)":
            lambda q, ag, lam=DEFAULT_LAMBDA: (f"epi_p{int(round(q * 100))}"
                                               + ("" if lam == DEFAULT_LAMBDA else f"_lam{lam:g}")),
    }
    caught = 0
    for name, bad in muts.items():
        # A mutant that agrees with the real tag everywhere tests nothing, and would report MISSED as
        # though the selfcheck were weak. Probe several points: mutant 2 differs ONLY at lambda 1.
        probes = ((0.99, "max", DEFAULT_LAMBDA), (0.90, "max", 10.0), (0.99, "median", 100.0))
        assert any(bad(*p) != good(*p) for p in probes), \
            f"mutant {name!r} is a no-op on every probe point -- the corruption tests nothing"
        g["tag"] = bad
        try:
            _selfcheck(verbose=False)
            print(f"  MISSED: {name}")
        except AssertionError:
            caught += 1
            print(f"  caught: {name}")
        finally:
            g["tag"] = good
    print(f"\nmutate-selfcheck: {caught} of {len(muts)} planted bugs caught")
    return 0 if caught == len(muts) else 1


def _selfcheck(verbose=True):
    """The comparison is the part that can silently lie, so that is what gets the controls."""
    # 1. pairing drops a seed the other arm lacks -- the Week-3 reference-mismatch bug
    base = {42: 10.0, 52: 20.0}
    abl = {42: 11.0}
    m, sd, n = paired_delta(base, abl)
    assert (n, m) == (1, 1.0), f"pairing must use only the shared seed, got n={n} mean={m}"
    unpaired = sum(abl.values()) / len(abl) - sum(base.values()) / len(base)
    assert abs(unpaired - m) > 1e-9, "control void: pick values where unpaired and paired differ"

    # 2. a complete pairing agrees with the difference of means, and carries a real spread
    m2, sd2, n2 = paired_delta({42: 10.0, 52: 20.0}, {42: 11.0, 52: 19.0})
    assert (n2, m2) == (2, 0.0) and sd2 > 0, \
        "paired mean must be 0 here, but the SPREAD is what makes it 'within noise'"

    # 2b. the printed columns must reconcile with the printed delta. Restricted to shared seeds,
    #     epi_mean - baseline_mean == paired mean; over ALL seeds it does not, which is the bug.
    shared = sorted(set(base) & set(abl))
    bm, _ = mean_sd([base[s] for s in shared])
    am, _ = mean_sd([abl[s] for s in shared])
    assert abs((am - bm) - m) < 1e-9, "shared-seed columns do not reproduce the paired delta"
    all_bm, _ = mean_sd(list(base.values()))
    assert abs((am - all_bm) - m) > 1e-9, \
        "control void: with these values the all-seed column happens to reconcile too"

    # 3. tags separate the bounds, or two runs overwrite each other
    assert tag(0.99, "max") != tag(0.90, "max") != tag(0.90, "median")
    assert tag(0.99, "max") == "epi_p99max"

    # 3b. LAMBDA must separate arms too. Without it two lambda arms at one bound collide on filename:
    #     --force overwrites the first, and without --force the second skips and the report reads the
    #     first arm's records under the second arm's name. Neither says anything in the log.
    assert tag(0.90, "max", 10.0) != tag(0.90, "max", 100.0), "lambda is absent from the tag"
    assert tag(0.90, "max", 1.0) != tag(0.90, "max", 10.0), "lambda 1 and lambda 10 collide"
    grid = {tag(q, ag, lm) for q in (0.99, 0.90) for ag in ("max", "median", "min")
            for lm in (1.0, 10.0, 100.0)}
    assert len(grid) == 2 * 3 * 3, f"tag grid collapses: {len(grid)} distinct of 18"

    # 3c. ... while the lambda-1 tag stays byte-identical, or the 40 released records stop resolving
    #      and both lambda ladders lose their base.
    assert tag(0.99, "max", DEFAULT_LAMBDA) == "epi_p99max", "backward compatibility broken at p99"
    assert tag(0.90, "max", DEFAULT_LAMBDA) == "epi_p90max", "backward compatibility broken at p90"
    assert tag(0.90, "max", 10.0) == "epi_p90max_lam10", \
        f"lambda must format without a trailing .0, got {tag(0.90, 'max', 10.0)!r}"

    # 3d. new arms must not land beside the released ones, and the reader must still see both trees
    p_new = rpath(f"encoder__dengue__seed42__{tag(0.99, 'median', 100.0)}.json", root=EXPDIR)
    assert EXPDIR in p_new.parents, f"a new arm would write outside {EXPDIR}: {p_new}"
    assert HERE not in p_new.parents, "a new arm writes into ablation/ -- released records at risk"
    assert EXPDIR in ARM_ROOTS and HERE in ARM_ROOTS, \
        "the reader must cover both trees, or the lambda ladder has no lambda-1 base"

    # 4. the headline field follows the dengue convention, or dengue is read on the wrong column
    assert field("dengue") == "country_macro" and field("influenza_japan") == "node_mean"

    # 4b. reader and writer must agree on the directory. train.loop writes through rpath, so a
    #     reader globbing HERE directly finds nothing and reports a finished arm as "not run".
    p = rpath("encoder__dengue__seed42__epi_p99max.json", root=HERE)
    assert p.parent != HERE, "records route into a subdir; globbing HERE would miss every file"
    assert p.parent == HERE / "single", f"unexpected route {p.parent}"

    # 4c. the bound must not depend on which datasets are TRAINED. Fake rate arrays, so no bundle
    #     is loaded: adding a faster-growing dataset to the calibration set must move r_max.
    import numpy as _np
    fake = {"slow": {g: _np.full(100, 0.1) for g in GAPS},
            "fast": {g: _np.full(100, 9.0) for g in GAPS}}
    one, _ = calibrate(["slow"], 0.9, verbose=False, rates=fake)
    both, _ = calibrate(["slow", "fast"], 0.9, verbose=False, rates=fake)
    assert one != both, "calibration scope does not move the bound -- control is void"
    med, _ = calibrate(["slow", "fast"], 0.9, verbose=False, aggregator="median", rates=fake)
    assert med != both, "aggregator does not move the bound -- the tag would be meaningless"

    # 5. seed parsing survives the ablation suffix (and refuses a filename without one)
    assert _seed_from("encoder__dengue__seed52__epi_p99max.json") == 52
    assert _seed_from("encoder__dengue__seed42.json") == 42
    assert _seed_from("encoder__dengue.json") is None

    if verbose:
        print("selfcheck ok: pairing drops unmatched seeds, tags separate bounds AND lambdas while "
              "lambda 1 stays backward compatible, new arms write outside ablation/, dengue reads "
              "country_macro, seed parsing survives the suffix")
    return 0


if __name__ == "__main__":
    sys.exit(main())

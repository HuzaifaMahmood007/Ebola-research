"""export_heatgnn.py -- probe + parallel runner for the HeatGNN leg of the Day-15 baseline matrix.

WHY THIS EXISTS. HeatGNN is the slow one: ~2h per run on us-regions (10 nodes) against ColaGNN's
~2 min on the same data, same env, same 1500-epoch budget. The cost is real per-epoch work (the
EIEL SIR-embedding module + a per-batch [B,m,m] Laplacian), not a misconfiguration -- the paper
fixes epochs=1500, patience=200, so we do not get to cut it. That leaves parallelism as the only
lever, and parallelism needs two numbers we do not have:

  1. what japan (47 nodes) and us-states (49 nodes) cost per epoch -- every timing so far is from
     the 10-node set, and the remaining 40 of 52 runs are on the big two;
  2. whether the model is thread-hungry -- if it uses BLAS threads, N workers fight each other and
     the speedup is not linear.

--probe answers both in ~5 minutes. --run then executes the matrix with the width it justifies.

Run from the repo root (a shell with conda on PATH):
    python export_heatgnn.py --probe                      # measure, project, delete its own junk
    python export_heatgnn.py --probe --epochs 8           # more epochs if 5 looks noisy
    python export_heatgnn.py --run --workers 6 --threads 1
    python export_heatgnn.py --run --workers 6 --threads 1 --dry-run

Resumable: --run skips any (dataset, horizon, seed) whose prediction file already exists, so a kill
and relaunch costs at most the in-flight runs.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from run_baselines import BL, CONDA, HORIZONS, INFLUENZA, PREDS, SEEDS, stage

SRC = BL / "HeatGNN-14DB" / "src"
LOGS = BL / "_logs"
PROBE_SEED = 999                       # never a real seed -> probe output can be deleted by pattern
PAPER_EPOCHS = 1500                    # paper: "the number of epochs to 1500", patience 200
EPOCH_RE = re.compile(r"time:\s*([0-9.]+)s")

# Ground truth, not a model: 8 completed us-regions runs, 04:39 -> 21:46 on 2026-07-29 = 2h08m each.
# Runs early-stop well before 1500 epochs, so per-epoch numbers cannot be multiplied by 1500 -- we
# scale this measured per-RUN cost by a measured per-EPOCH ratio instead.
US_REGIONS_REAL_H = 2.13
US_REGIONS_5EP_SPEP = 17.38            # steady mean of epochs 2-5 in the successful probe


def pred_name(ds, h, seed):
    return f"HeatGNN__{ds}__h{h}__seed{seed}.npz"


# Launch order, deliberately not INFLUENZA's order. japan and us-states are 0-of-20 and are the two
# that needed the self-loop fix, so they go first: a full 1500-epoch run there is the real proof the
# fix holds beyond a 5-epoch probe. us-regions is already 9-of-20 and is the slowest per run (2.13 h
# vs ~1.18 h), so it tails the queue where it costs least.
ORDER = ["influenza_japan", "influenza_us-states", "influenza_us-regions"]


def remaining():
    """(dataset, horizon, seed) triples with no prediction file yet. This IS the run plan."""
    return [(d, h, s) for d in ORDER for h in HORIZONS for s in SEEDS
            if not (PREDS / pred_name(d, h, s)).exists()]


def _cmd(ds, h, seed, epochs=None):
    c = [CONDA, "run", "--no-capture-output", "-n", "heatgnn", "python", "train.py",
         "--model", "HeatGNN", "--dataset", ds, "--sim_mat", f"{ds}-adj",
         "--window", "20", "--horizon", str(h), "--seed", str(seed),
         "--train", ".5", "--val", ".2", "--test", ".3"]        # our split, not the paper's 60/20/20
    if epochs is not None:
        c += ["--epochs", str(epochs)]
    return c


def _env(threads):
    """threads=None leaves torch alone; an int pins BLAS/OMP so N workers cannot oversubscribe."""
    import os
    e = dict(os.environ)
    if threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            e[k] = str(threads)
    return e


DATA = BL / "HeatGNN-14DB" / "data"


def _stage_all(selfloops=True):
    """Stage our exports, then (for HeatGNN only) fold the identity into the adjacency.

    WHY. japan and us-states each carry 2 degree-0 nodes (Okinawa; Alaska/Hawaii). On the raw graph
    HeatGNN NaNs in backward -- confirmed at two independent seeds (999, 992) -- while the same data
    with self-loops trains clean. getLaplaceMat (utils.py:70-87, copied verbatim from EpiGNN) never
    folds in the identity: the block that would is commented out, leaving `sum(adj) + 1e-12`, so an
    isolated node is scaled by ~1e12 rather than erroring. EpiGNN survives the same code because it
    has no SIR loss to amplify it; HeatGNN's EIEL module does.

    Scope is deliberate: HeatGNN's staged copy only. EpiGNN/Cola/MTGNN already ran and are untouched.
    Note this also brings HeatGNN INTO line with our own encoder, which adds the identity before
    normalising (models/encoder.py:59-60, the C3 guard). Disclose in the paper + failure log."""
    for ds in INFLUENZA:
        stage(ds, DATA)                       # re-copies from _exported, so this stays idempotent
    if not selfloops:
        return
    for ds in INFLUENZA:
        p = DATA / f"{ds}-adj.txt"
        iso = _add_self_loops(p, p)
        if iso:
            print(f"  {ds}: self-loops added, {len(iso)} isolated node(s) {iso}")


# --------------------------------------------------------------------------- #
# --nan-test: is the japan/us-states NaN caused by isolated nodes, or a bad seed?
#
# japan and us-states each carry 2 degree-0 nodes (islands: Okinawa; Alaska/Hawaii); us-regions
# carries none -- and us-regions is the only one that trains. HeatGNN's normalisers do not divide
# by zero, they divide by an epsilon (utils.py:86 `+1e-12`, HeatGNN.py:172 `+1e-8`), so an isolated
# node is scaled by ~1e12 instead of erroring. Forward survives; backward overflows, which is
# exactly the reported `MulBackward0 returned nan values`.
#
# Arm A: same data, adjacency with self-loops added (degree 0 -> 1). Arm B: original adjacency,
# different seed. A passes and B fails -> structural, fix the graph. Both fail -> not the graph.
# --------------------------------------------------------------------------- #
def _add_self_loops(src: Path, dst: Path):
    """A -> A+I, comma-separated dense text. Returns the indices that were isolated."""
    rows = [[float(x) for x in ln.split(",")] for ln in src.read_text().strip().splitlines()]
    iso = [i for i, r in enumerate(rows) if not any(v > 0 for v in r)]
    for i, r in enumerate(rows):
        if r[i] <= 0:
            r[i] = 1.0
    dst.write_text("\n".join(",".join(f"{v:g}" for v in r) for r in rows))
    return iso


def _purge_csv_rows(seeds=(), datasets=()):
    """Drop rows from HeatGNN_results_test.csv by seed or dataset, keeping a .bak.

    Needed because train.py:99-105 exits immediately when a row already matches
    (model, dataset, window, horizon, seed) -- a stale row blocks the real run forever."""
    import csv
    p = SRC / "HeatGNN_results_test.csv"
    if not p.exists():
        return 0
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return 0
    head, body = rows[0], rows[1:]
    i_seed, i_ds = head.index("seed"), head.index("dataset")
    keep = [r for r in body
            if not (len(r) > max(i_seed, i_ds)
                    and (r[i_seed] in {str(s) for s in seeds} or r[i_ds] in set(datasets)))]
    dropped = len(body) - len(keep)
    if dropped:
        p.replace(p.with_suffix(".csv.bak"))
        with open(p, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([head] + keep)
    return dropped


def nan_test(epochs=5):
    _stage_all(selfloops=False)                    # arm B must see the ORIGINAL graph
    LOGS.mkdir(parents=True, exist_ok=True)
    data_dir = DATA
    probe_ds = "influenza_japan_selfloop"          # own dataset name -> cannot collide in the CSV

    iso = _add_self_loops(data_dir / "influenza_japan-adj.txt", data_dir / f"{probe_ds}-adj.txt")
    (data_dir / f"{probe_ds}.txt").write_bytes((data_dir / "influenza_japan.txt").read_bytes())
    print(f"japan isolated nodes (degree 0): {len(iso)} -> {iso}   self-loops added\n")

    arms = [("A: japan + self-loops", probe_ds, 991),
            ("B: japan original adj, new seed", "influenza_japan", 992)]
    verdict = {}
    for label, ds, seed in arms:
        log = LOGS / f"nantest__{ds}__seed{seed}.log"
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            subprocess.run(_cmd(ds, 5, seed, epochs=epochs), cwd=str(SRC),
                           env=_env(1), stdout=fh, stderr=subprocess.STDOUT)
        txt = log.read_text(encoding="utf-8", errors="replace")
        times = [float(x) for x in EPOCH_RE.findall(txt)]
        nan = "nan values" in txt
        steady = times[1:] if len(times) > 1 else times
        mean = sum(steady) / len(steady) if steady else float("nan")
        verdict[label] = (not nan and len(times) >= epochs, mean, len(times))
        print(f"  {label:<34} {'NaN' if nan else 'ok ':<4} epochs={len(times)}  "
              f"mean={mean:6.2f}s/epoch  wall={time.time() - t0:5.1f}s")

    a_ok = verdict["A: japan + self-loops"][0]
    b_ok = verdict["B: japan original adj, new seed"][0]
    print("\n  READ:")
    if a_ok and not b_ok:
        print("    Self-loops fix it, a fresh seed does not -> STRUCTURAL (isolated nodes).")
        print("    Next decision: apply A+I to HeatGNN's staged adjacency only, or everywhere.")
    elif a_ok and b_ok:
        print("    Both train -> seed 999 was unlucky, the graph is not the problem.")
        print("    Cheapest path: keep the original adjacency, avoid the bad seed.")
    elif not a_ok and not b_ok:
        print("    Both NaN -> not the graph and not the seed. Something else; do not launch 52 runs.")
    else:
        print("    Self-loops NaN but plain seed trains -> self-loops are not the answer. Investigate.")

    # timing bonus: arm A/B give japan's per-epoch cost, which is what --probe failed to get
    got = [m for _, m, n in verdict.values() if m == m and n >= 2]
    if got:
        ratio = min(got) / US_REGIONS_5EP_SPEP
        print(f"\n  japan {min(got):.2f} s/epoch vs us-regions {US_REGIONS_5EP_SPEP:.2f} = {ratio:.2f}x")
        print(f"  -> japan/us-states run ~{US_REGIONS_REAL_H * ratio:.1f} h each "
              f"(us-regions ground truth {US_REGIONS_REAL_H:.2f} h over 8 runs)")
        _project_from_ratio(ratio)

    for f in (data_dir / f"{probe_ds}-adj.txt", data_dir / f"{probe_ds}.txt"):
        f.unlink(missing_ok=True)
    for p in PREDS.glob("HeatGNN__*seed99*.npz"):
        p.unlink()
    print(f"\n  cleaned probe artefacts; purged {_purge_csv_rows(seeds=range(990, 1000), datasets=[probe_ds])} CSV row(s)")


def _project_from_ratio(ratio):
    """us-regions has 8 real runs, so its per-run cost is measured, not modelled. Scale it by the
    per-epoch ratio to price japan/us-states rather than trusting a 5-epoch absolute."""
    todo = remaining()
    print("\n  " + "=" * 70)
    print(f"  {'dataset':<22}{'left':>5}{'h/run':>9}{'h total':>10}")
    grand = 0.0
    for ds in INFLUENZA:
        n = sum(1 for d, _, _ in todo if d == ds)
        h = US_REGIONS_REAL_H if ds == "influenza_us-regions" else US_REGIONS_REAL_H * ratio
        grand += h * n
        print(f"  {ds:<22}{n:>5}{h:>9.2f}{h * n:>10.1f}")
    print(f"  {'TOTAL sequential':<27}{'':>9}{grand:>10.1f} h  ({grand / 24:.1f} days)")
    for w in (4, 6, 8):
        print(f"      {w}-way parallel: {grand / w:6.1f} h  ({grand / w / 24:.2f} days)")


# --------------------------------------------------------------------------- #
# --probe
# --------------------------------------------------------------------------- #
def probe(epochs, threads_variants):
    _stage_all()
    LOGS.mkdir(parents=True, exist_ok=True)
    # us-regions first: we know its true cost (~2h08m/run over 8 runs), so it calibrates the probe
    # against reality and tells us how much the short-run warmup is distorting things.
    plan = [(ds, t) for ds in INFLUENZA for t in threads_variants]
    rows = []
    # Every combo gets its OWN seed. Sharing one seed made the second combo hit train.py:99-105
    # ("Experiment exists") and exit in 6 seconds without training -- a silent skip, not an error.
    for i, (ds, thr) in enumerate(plan):
        seed = 990 + i
        log = LOGS / f"probe__{ds}__thr{thr or 'default'}.log"
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            subprocess.run(_cmd(ds, 5, seed, epochs=epochs), cwd=str(SRC),
                           env=_env(thr), stdout=fh, stderr=subprocess.STDOUT)
        wall = time.time() - t0
        times = [float(x) for x in EPOCH_RE.findall(log.read_text(encoding="utf-8", errors="replace"))]
        # epoch 1 carries import/warmup and a test-eval (train.py:379 evaluates whenever val improves,
        # which early on is every epoch), so it is not representative. Drop it when we can.
        steady = times[1:] if len(times) > 1 else times
        mean = sum(steady) / len(steady) if steady else float("nan")
        rows.append(dict(ds=ds, thr=thr, wall=wall, n=len(times), mean=mean))
        print(f"  {ds:<22} thr={str(thr or 'default'):<7} wall={wall:6.1f}s  "
              f"epochs={len(times)}  mean={mean:6.2f}s/epoch", flush=True)
        if not times:
            print(f"      !! no epoch lines parsed -- read {log}", flush=True)

    _cleanup_probe()
    _project(rows)
    return rows


def _cleanup_probe():
    """Probe writes real prediction files at seeds 990-999. They are 5-epoch garbage and
    score_baseline.py would happily rescore them, so they die here -- along with their CSV rows,
    which would otherwise make a future run at the same seed exit instantly (train.py:99-105).
    run_baselines.done() documents the same trap."""
    killed = 0
    for p in PREDS.glob("HeatGNN__*__seed99*.npz"):
        p.unlink(); killed += 1
    rows = _purge_csv_rows(seeds=range(990, 1000))
    print(f"\n  cleaned {killed} probe prediction file(s), purged {rows} probe CSV row(s)")


def _project(rows):
    todo = remaining()
    per_ds = {}
    for ds in INFLUENZA:
        cand = [r for r in rows if r["ds"] == ds and r["mean"] == r["mean"]]
        if cand:
            per_ds[ds] = min(r["mean"] for r in cand)      # best thread setting for that dataset

    print("\n" + "=" * 78)
    print(f"PROJECTION at the paper's {PAPER_EPOCHS} epochs  ({len(todo)} runs left)")
    print("=" * 78)
    print(f"  {'dataset':<22}{'left':>5}{'s/epoch':>10}{'h/run':>8}{'h total':>10}")
    grand = 0.0
    for ds in INFLUENZA:
        n = sum(1 for d, _, _ in todo if d == ds)
        if ds not in per_ds:
            print(f"  {ds:<22}{n:>5}{'--':>10}{'--':>8}{'--':>10}")
            continue
        h_run = per_ds[ds] * PAPER_EPOCHS / 3600
        grand += h_run * n
        print(f"  {ds:<22}{n:>5}{per_ds[ds]:>10.2f}{h_run:>8.1f}{h_run * n:>10.1f}")
    print(f"  {'TOTAL sequential':<27}{'':>10}{'':>8}{grand:>10.1f}  ({grand / 24:.1f} days)")
    for w in (4, 6, 8):
        print(f"    at {w}-way parallel: {grand / w:6.1f} h  ({grand / w / 24:.2f} days)")

    thr_note = {}
    for ds in INFLUENZA:
        got = {r["thr"]: r["mean"] for r in rows if r["ds"] == ds and r["mean"] == r["mean"]}
        if len(got) > 1:
            thr_note[ds] = got
    if thr_note:
        print("\n  THREAD SENSITIVITY (decides whether workers fight each other):")
        for ds, got in thr_note.items():
            base = got.get(None) or got.get(max(got, key=lambda k: k or 0))
            for thr, m in sorted(got.items(), key=lambda kv: (kv[0] is None, kv[0])):
                rel = f"{m / base:.2f}x" if base else "--"
                print(f"    {ds:<22} thr={str(thr or 'default'):<7} {m:6.2f}s/epoch  ({rel} of default)")
        print("    ~1.0x at thr=1  -> not thread-bound -> parallel is near-linear, go wide (6-8).")
        print("    much slower at thr=1 -> BLAS-bound -> go narrow (3-4 workers, 3 threads each).")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# --run
# --------------------------------------------------------------------------- #
def run_all(workers, threads, dry_run=False):
    _stage_all()
    LOGS.mkdir(parents=True, exist_ok=True)
    todo = remaining()
    print(f"HeatGNN: {len(todo)} runs left, {workers} workers, threads={threads or 'default'}")
    for ds in INFLUENZA:
        print(f"    {ds:<22} {sum(1 for d, _, _ in todo if d == ds):>3}")
    if dry_run:
        for ds, h, s in todo:
            print("   would run", ds, f"h{h}", f"seed{s}")
        return
    if not todo:
        print("nothing to do."); return

    t_start = time.time()
    live, done_n, failed = [], 0, []
    while todo or live:
        while todo and len(live) < workers:
            ds, h, s = todo.pop(0)
            log = LOGS / f"HeatGNN__{ds}__h{h}__seed{s}.log"
            fh = open(log, "w", encoding="utf-8", errors="replace")
            p = subprocess.Popen(_cmd(ds, h, s), cwd=str(SRC), env=_env(threads),
                                 stdout=fh, stderr=subprocess.STDOUT)
            live.append((p, fh, (ds, h, s), time.time()))
            print(f"  [start] {ds} h{h} seed{s}   ({len(live)} live, {len(todo)} queued)", flush=True)
        time.sleep(5)
        for item in list(live):
            p, fh, (ds, h, s), t0 = item
            if p.poll() is None:
                continue
            live.remove(item); fh.close()
            ok = (PREDS / pred_name(ds, h, s)).exists()
            done_n += 1
            if not ok:
                failed.append((ds, h, s))
            mins = (time.time() - t0) / 60
            el = (time.time() - t_start) / 3600
            left = len(todo) + len(live)
            eta = (el / done_n) * left if done_n else 0
            print(f"  [{'ok  ' if ok else 'FAIL'}] {ds} h{h} seed{s}  {mins:5.1f} min   "
                  f"done {done_n}, left {left}, elapsed {el:.1f}h, eta {eta:.1f}h", flush=True)

    print(f"\nfinished in {(time.time() - t_start) / 3600:.1f}h. failures: {len(failed)}")
    for f in failed:
        print("   FAILED (no prediction file):", f)
    print("\nnext:  conda run -n ebola-train python score_baseline.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help="measure per-epoch cost, project, exit")
    ap.add_argument("--nan-test", action="store_true",
                    help="japan NaN: self-loops vs fresh seed, 2 short runs, plus a timing read")
    ap.add_argument("--run", action="store_true", help="run the remaining matrix in parallel")
    ap.add_argument("--epochs", type=int, default=5, help="probe epochs per run (default 5)")
    ap.add_argument("--workers", type=int, default=6, help="--run: concurrent processes")
    ap.add_argument("--threads", type=int, default=1,
                    help="--run/--probe: BLAS threads per process; 0 = leave torch alone")
    ap.add_argument("--dry-run", action="store_true", help="--run: print the plan, run nothing")
    a = ap.parse_args()

    if a.nan_test:
        nan_test(a.epochs)
    elif a.probe:
        # both thread settings, so one probe answers cost AND thread sensitivity
        probe(a.epochs, threads_variants=[None, 1])
    elif a.run:
        run_all(a.workers, a.threads or None, dry_run=a.dry_run)
    else:
        ap.print_help(); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

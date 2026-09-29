"""EXPLORATORY: a fresh gate-off arm on today's code, covid_us-states and influenza_japan. Do the
2026-08-17 gate-off records reproduce, and does the degree term still help at h5 on same-day code?

THE QUESTION. pure-TCN (experiments/pure_tcn.py: gate off AND no LTR degree term) is worse than the
archived gate-off arm at h5 only, on both panels, in all five seeds, RMSE and MAE
(experiments/pure_tcn__summary.json). h5 RMSE means, learned / archived gate-off / pure-TCN: covid
8798.3 / 7835.4 / 8697.7, japan 841.1 / 779.4 / 838.2. Two readings:
  (a) the degree term helps at h5 when mixing is off, and pure-TCN is worse because it lost it;
  (b) the archived gate-off records (ablation/single/encoder__{D}__seed{S}__gateoff.json) are the
      outlier, and pure-TCN only looks worse against them.
Those records are not in git (.gitignore ignores /ablation), and run_gate_ablation.py --force would
overwrite them, so this arm writes to experiments/ only and reads ablation/ and results/ only.

THE ARM. L.train_one(ds, s, **TRAIN_KW), with TRAIN_KW the arguments of
ablation/run_gate_ablation.py:153 verbatim (80 is that script's --epochs default). No encoder_factory,
so train_one builds the plain gate-off SharedEncoder with LTR kept. gate_read keeps its default, as on
Aug 17; it runs after the test predictions exist and cannot change a record. No commit touched
train/loop.py or models/ between d758809 (2026-08-13) and 191714f (2026-09-24), and the three since
add opt-in paths (permute_adjacency, encoder_factory, epi bookkeeping) this call never engages. torch
2.6.0+cu124 and numpy 2.4.6 match env_train_snapshot_20260803.txt. So on paper nothing on this path
changed; Table A is the test of that.

THE REPORT, per panel.
  A  fresh vs archived gate-off, d = fresh - archived: the reproduction check. Exact equality is not
     expected: the TCN backward runs on cuDNN dilated convs, measured non-deterministic at a fixed seed
     here (models/encoder_v2.py:77). The degree scatter_add_ (models/spatial.py:31) is not a source:
     A is 0/1 on both panels, so it adds small integers, exact in float32. With g=0 the mixer's
     sparse.mm output is multiplied by zero.
     Read the drift against the printed RMSE/MAE seed spread: far below it is drift, near it is a
     different run.
  B  pure-TCN vs fresh gate-off, d = pure - fresh: the degree test with both arms on today's code.
     Same layout and labels as pure_tcn's vs-gate-off table (DEGREE HELPS / degree HURTS / within noise).
  h5 RMSE and MAE means for learned, archived gate-off, fresh gate-off, pure-TCN, on shared seeds.

WHAT EACH OUTCOME MEANS.
  A drift far below seed spread, B still DEGREE HELPS at h5  -> reading (a). The Aug-17 records stand.
  A drift near or above seed spread, B within noise at h5    -> reading (b). The Aug-17 records do not
                                                                reproduce; retire the h5 degree claim.
  anything else                                              -> undecided at 5 seeds. Say so.

  conda run --no-capture-output -n ebola-train python -m experiments.gateoff_fresh --selfcheck
  conda run --no-capture-output -n ebola-train python -m experiments.gateoff_fresh
  conda run --no-capture-output -n ebola-train python -m experiments.gateoff_fresh --report

Run from the repo root. The Aug-17 gate-off took 0.4 to 0.6 min per seed on covid and 1.6 to 2.7 min
per seed on influenza_japan (results/reports/gate_ablation.log:19-23 and 4-8), so about 13 min for all
ten. One JSON per seed as it finishes; an existing seed file is skipped unless --force. Nothing here is
a scored result: every record carries protocol="EXPLORATORY".
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import bundles
import experiments.pure_tcn as P
import train.loop as L
from ablation.run_epi_ablation import MODEL, POINT, _seed_from, field, load_arm, mean_sd, paired_delta
from experiments.norm_probe import OUT, TAG, _dump
from models import sparse_from_dense_np, window_slice
from results_paths import rpath

TRAIN_KW = dict(epochs=80, verbose=False, gate_mode="off")   # ablation/run_gate_ablation.py:153 verbatim
SUMMARY = "gateoff_fresh__summary.json"


def seed_file(ds, s):
    return f"gateoff_fresh__{ds}__seed{s}.json"


def fresh_glob(ds):
    return OUT / f"gateoff_fresh__{ds}__seed*.json"


def pending(ds, seeds, force, out=None):
    """Seeds still to train. A seed is done when its file exists. `out` resolves at CALL time, so it
    always checks the directory _dump writes to (a def-time default would freeze the import value)."""
    out = OUT if out is None else out
    done = [s for s in seeds if (out / seed_file(ds, s)).exists() and not force]
    for s in done:
        print(f"  {ds} seed {s}: {out / seed_file(ds, s)} exists, skipped (--force retrains it)")
    return [s for s in seeds if s not in done]


def is_gateoff(r):
    """The plain gate-off arm: g=0, LTR kept, no factory meta. pure_tcn.factory, the one factory in the
    repo, stamps encoder_version and ltr="off"; the default path stamps neither."""
    return r.get("gate_mode") == "off" and r.get("ltr") != "off" and "encoder_version" not in r


def train(ds, seeds, force):
    todo = pending(ds, seeds, force)
    print(f"\nGATE-OFF FRESH {ds}: {len(todo)} of {len(seeds)} seeds to train")
    for s in todo:
        t0 = time.time()
        recs, _, _, _ = L.train_one(ds, s, **TRAIN_KW)
        # ponytail: records only, no pernode. Table A needs the records; a per-node drift check would
        # need pernode, a rerun is minutes.
        for r in recs:
            r["protocol"] = TAG
        bad = {(r.get("gate_mode"), r.get("ltr"), r.get("encoder_version")) for r in recs if not is_gateoff(r)}
        assert not bad, f"{ds} seed {s}: records claim (gate_mode, ltr, encoder_version) = {bad}"
        p = _dump(recs, seed_file(ds, s))
        print(f"  {ds} seed {s} done in {(time.time() - t0) / 60:.1f} min, {len(recs)} records -> {p}")


def pair_cells(ref, arm):
    """Every (metric, horizon) both arms hold, over their SHARED seeds, d = arm - ref."""
    cells = []
    for m in POINT:
        for h in bundles.HORIZONS:
            r, a = ref.get((h, m), {}), arm.get((h, m), {})
            shared = sorted(set(r) & set(a))
            if not shared:                         # nothing to pair: skip, never tally it as noise
                continue
            rel = {s: abs(a[s] - r[s]) / abs(r[s]) if r[s] else (0.0 if a[s] == r[s] else float("inf"))
                   for s in shared}
            worst = max(shared, key=rel.get)
            (rm, rsd), (am, asd) = mean_sd([r[s] for s in shared]), mean_sd([a[s] for s in shared])
            dm, dsd, n = paired_delta(r, a)
            cells.append(dict(metric=m, horizon=h, seeds=shared, ref_mean=rm, ref_sd=rsd, arm_mean=am,
                              arm_sd=asd, d_mean=dm, d_sd=dsd, n=n, max_rel=rel[worst],
                              max_rel_seed=worst, identical=sum(a[s] == r[s] for s in shared)))
    return cells


def show(cells, ref_name, arm_name, last, tail):
    for m in POINT:
        rows = [c for c in cells if c["metric"] == m]
        if rows:
            print(f"    {m.upper()} ({'higher' if m == 'pcc' else 'lower'}=better)")
            print(f"      {'h':>3} | {ref_name:>18} | {arm_name:>18} | {'paired d':>18} | {last}")
        for c in rows:
            print(f"      {c['horizon']:>3} | {c['ref_mean']:9.3f} +-{c['ref_sd']:6.3f} | "
                  f"{c['arm_mean']:9.3f} +-{c['arm_sd']:6.3f} | {c['d_mean']:+9.3f} +-{c['d_sd']:6.3f} | "
                  f"{tail(c)}")


def report(datasets):
    blocks = {}
    for ds in datasets:
        block = report_one(ds)
        if block:
            blocks[ds] = block
    if not blocks:
        return None
    # ponytail: carries other panels over, does not refuse a changed panel the way pure_tcn does. Every
    # number here is re-derived from the seed files, and those are the skip-protected artifact.
    f = OUT / SUMMARY
    merged = {**(json.loads(f.read_text()) if f.exists() else {}), **blocks}
    p = _dump(merged, SUMMARY)
    print(f"wrote {p}: {sorted(merged)}, regenerated {sorted(blocks)}")
    return p


def report_one(ds):
    (_, learned, _, _), (_, archived, _, _) = P.comparisons(ds)
    src = {"learned": learned, "archived gate-off": archived, "fresh gate-off": fresh_glob(ds),
           "pure-TCN": P.pure_glob(ds)}
    arm = {k: load_arm(v, _seed_from, MODEL) for k, v in src.items()}
    print(f"\n{'=' * 100}\nGATE-OFF FRESH :: {ds}   model={MODEL}   field={field(ds)}   {TAG}\n{'=' * 100}")
    for k, v in src.items():
        print(f"  {k:<18} {len(P._seeds(arm[k]))} seeds {P._seeds(arm[k])}  <- {v}")
    if not arm["fresh gate-off"]:
        print("\n  no fresh gate-off records yet")
        return None

    print("\n  A. REPRODUCTION   d = fresh - archived (2026-08-17), paired by seed")
    rep = pair_cells(arm["archived gate-off"], arm["fresh gate-off"])
    show(rep, "archived gate-off", "fresh gate-off", "max seed rel diff",
         lambda c: f"{c['max_rel']:.4%} (seed {c['max_rel_seed']}), {c['identical']}/{c['n']} identical")
    # PCC sits near 0 on covid (h5 mean -0.115), so a relative diff or sd/|mean| there is no scale: a small
    # absolute move reads as a huge percent. The yardstick is RMSE/MAE only; the all-cells max is kept.
    err = [c for c in rep if c["metric"] != "pcc"]
    top = lambda cs: max(cs, key=lambda c: c["max_rel"]) if cs else None
    at = lambda c: c and [c["metric"], c["horizon"], c["max_rel_seed"]]
    worst, worst_err = top(rep), top(err)
    spread = [c["ref_sd"] / abs(c["ref_mean"]) for c in err if c["ref_mean"]]
    if worst and worst_err and spread:
        say = lambda c: f"{c['max_rel']:.4%} ({c['metric']} h{c['horizon']} seed {c['max_rel_seed']})"
        print(f"  {ds}: max per-seed relative diff {say(worst)} over all {len(rep)} cells, {say(worst_err)} "
              f"over the {len(err)} RMSE/MAE cells; {sum(c['identical'] for c in rep)} of "
              f"{sum(c['n'] for c in rep)} values bit-identical. For scale, the archived seed spread "
              f"sd/|mean| on RMSE/MAE runs {min(spread):.2%} to {max(spread):.2%}.")

    print("\n  B. DEGREE TEST, same-day code   d = pure - fresh gate-off, paired by seed")
    deg = pair_cells(arm["fresh gate-off"], arm["pure-TCN"])
    tally = {"DEGREE HELPS": 0, "degree HURTS": 0, "within noise": 0}
    for c in deg:
        c["verdict"] = P.verdict(c["d_mean"], c["d_sd"], c["n"], c["metric"], "DEGREE HELPS", "degree HURTS")
        tally[c["verdict"]] += 1
    show(deg, "fresh gate-off", "pure-TCN", "verdict", lambda c: f"{c['verdict']} (n={c['n']})")
    print(f"    TALLY vs fresh gate-off, {len(deg)} cells: " + ", ".join(f"{k} {n}" for k, n in tally.items()))

    h5 = {}
    print()
    for m in ("rmse", "mae"):
        by = {k: v.get((5, m), {}) for k, v in arm.items()}
        shared = sorted(set.intersection(*(set(v) for v in by.values())))
        h5[m] = dict(seeds=shared, **{k: mean_sd([v[s] for s in shared])[0] for k, v in by.items()})
        print(f"  h5 {m.upper():<4} over seeds {shared}: " + " | ".join(f"{k} {h5[m][k]:.1f}" for k in src))

    n_max = max((c["n"] for c in rep + deg), default=0)
    ceiling = (f"CEILING: at most {n_max} paired seeds on {ds}, and seed sd is the only noise scale, so "
               f"|mean| < sd is a weak rule. Sign and spread only, no significance claim.")
    print(f"\n  {ceiling}\n")
    return dict(protocol=TAG, dataset=ds, model=MODEL, field=field(ds),
                seeds={k: P._seeds(v) for k, v in arm.items()},
                reproduction=dict(cells=rep, max_rel=worst and worst["max_rel"], max_rel_at=at(worst),
                                  max_rel_rmse_mae=worst_err and worst_err["max_rel"],
                                  max_rel_rmse_mae_at=at(worst_err),
                                  seed_spread_rmse_mae=[min(spread, default=None), max(spread, default=None)]),
                degree=dict(cells=deg, tally=tally), h5=h5, ceiling=ceiling)


def built(ds, kw):
    """(trunk, records) train_one produces from kw, untrained: epochs=0 takes no optimiser step and
    writes nothing, so this is the exact build and meta stamping the real run gets."""
    ro = {}
    recs, _, _, _ = L.train_one(ds, 42, **{**kw, "epochs": 0}, run_out=ro)
    return ro["enc"].cpu().eval(), recs


def graph_moves(enc, b):
    """max |out(real A) - out(all-zero A)| on real panel data. With g=0 only the LTR degree term can move it."""
    t = int(b.origins(phase="test")[0])
    Z = window_slice(torch.tensor(b.transfer_view(), dtype=torch.float32), t)
    Mt = torch.tensor(b.M[:, t], dtype=torch.float32)
    with torch.no_grad():
        return float((enc(Z, sparse_from_dense_np(b.A_geo), Mt) -
                      enc(Z, sparse_from_dense_np(np.zeros_like(b.A_geo)), Mt)).abs().max())


def inside(p, root):
    p, root = Path(p).resolve(), Path(root).resolve()
    return p == root or root in p.parents


def selfcheck(datasets):
    """Every check carries a planted control that must fail, or the check proves nothing."""
    before = sorted(os.listdir(OUT))

    # 1. outputs land in experiments/ only, never where the archived records live
    for p in [OUT / seed_file(d, 42) for d in datasets] + [OUT / SUMMARY]:
        assert inside(p, OUT) and not inside(p, "ablation") and not inside(p, L.RESULTS), f"{p} escapes experiments/"
    planted = rpath(f"encoder__{datasets[0]}__seed42__gateoff.json", root="ablation")   # what --force would hit
    assert inside(planted, "ablation") and not inside(planted, OUT), f"CONTROL void: {planted} not seen as ablation/"
    try:
        _dump({}, "../gateoff_fresh_escape_probe.json")
    except AssertionError:
        pass
    else:
        (OUT / "../gateoff_fresh_escape_probe.json").unlink()
        raise SystemExit("CONTROL void: _dump wrote outside experiments/")
    print(f"ok {seed_file(datasets[0], 42)} and {SUMMARY} resolve inside experiments/, not ablation/ or "
          f"results/; control: {planted} is caught, _dump refuses ../")

    # 2. an existing seed is skipped, --force retrains it
    ds = datasets[0]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert pending(ds, [42, 52], False, out=d) == [42, 52], "CONTROL void: an empty dir skipped a seed"
        (d / seed_file(ds, 42)).write_text("[]")
        assert os.listdir(d), "CONTROL void: the listing cannot see a written file"
        assert pending(ds, [42, 52], False, out=d) == [52], "an existing seed file was not skipped"
        assert pending(ds, [42, 52], True, out=d) == [42, 52], "--force did not retrain an existing seed"
    assert not d.exists(), f"temp dir {d} survived"
    print("ok existing seed skipped, --force retrains; control: an empty dir skips nothing")

    # 3. the trunk TRAIN_KW builds still reads the graph (LTR kept), and the record label says gate-off
    for ds in datasets:
        b = bundles.load(ds)
        enc, recs = built(ds, TRAIN_KW)
        enc_c, recs_c = built(ds, {**TRAIN_KW, "encoder_factory": P.factory})
        arm, ctl = graph_moves(enc, b), graph_moves(enc_c, b)
        assert arm > 0, (f"{ds}: the trunk TRAIN_KW builds did not move when the graph was emptied, so it "
                         f"has no degree term. This is not the Aug-17 gate-off arm")
        assert ctl == 0, f"CONTROL void on {ds}: the pure-TCN trunk moved by {ctl:.3g}"
        assert all(map(is_gateoff, recs)), f"{ds}: TRAIN_KW records fail the gate-off label check"
        assert not any(map(is_gateoff, recs_c)), f"CONTROL void on {ds}: pure-TCN records passed the label check"
        print(f"ok {ds}: TRAIN_KW trunk moves by up to {arm:.3f} when A is emptied (degree term present), "
              f"its records pass the label check; control: the pure-TCN trunk moves by {ctl:g} and its "
              f"records fail the label check")

    # 4. the reader keys the new filenames, and d = arm - ref points the verdict the right way
    assert _seed_from(seed_file("influenza_japan", 42)) == 42, "seed must parse from the fresh filename"
    assert _seed_from(SUMMARY) is None, "CONTROL void: the seedless summary name parsed"
    base = {(5, "rmse"): {s: 100.0 for s in L.SEEDS}}
    worse = {(5, "rmse"): {s: 110.0 + s % 4 for s in L.SEEDS}}          # d = 12,10,12,10,12
    call = lambda ref, a: P.verdict(*(pair_cells(ref, a)[0][k] for k in ("d_mean", "d_sd", "n")),
                                    "rmse", "DEGREE HELPS", "degree HURTS")
    assert call(base, worse) == "DEGREE HELPS", "a worse pure-TCN must read DEGREE HELPS"
    assert call(worse, base) == "degree HURTS", "CONTROL void: swapping the arms did not flip the verdict"
    print(f"ok _seed_from({seed_file('influenza_japan', 42)!r}) == 42, a worse pure-TCN reads DEGREE HELPS; "
          f"control: {SUMMARY} parses to None, swapped arms read degree HURTS")

    assert sorted(os.listdir(OUT)) == before, f"selfcheck left files: {set(os.listdir(OUT)) ^ set(before)}"
    print("selfcheck passed, experiments/ unchanged")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=bundles.DEV_BUNDLE_NAMES, default=P.DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(L.SEEDS))
    ap.add_argument("--force", action="store_true", help="retrain a seed whose file already exists")
    ap.add_argument("--report", action="store_true", help="read what is on disk, train nothing")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck(a.datasets)
    if a.report:
        return report(a.datasets)
    selfcheck(a.datasets)
    for ds in a.datasets:
        train(ds, a.seeds, a.force)
    report(a.datasets)


if __name__ == "__main__":
    main()

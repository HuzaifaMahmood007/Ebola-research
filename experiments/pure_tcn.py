"""EXPLORATORY: a truly graph-free, TCN-only trunk on covid_us-states and influenza_japan. What does the
WHOLE graph buy on each panel?

THE GAP. The gate-off ablation (ablation/run_gate_ablation.py) forces g=0 but still adds the LTR degree
feature, `h = self.tcn(Z) + self.ltr(deg)` at models/encoder.py:60. Its own CEILING line says so: it
measures NEIGHBOUR INFORMATION, not the graph in total. No run anywhere drops the degree term, so "the
graph does not help" has only been tested with one graph-derived scalar per node still in the model.

THE ARM. train.loop.train_one with gate_mode="off" and an encoder_factory (train/loop.py:179-183) that
builds the gate-off SharedEncoder and then swaps `enc.ltr` for a module with no parameters whose forward
returns 0.0, so the trunk computes tcn(Z) + 0.0. The swap happens AFTER the full build, so the discarded
LTR still consumed its init draws: at the same seed the TCN, the spatial mixer and the Adapter start from
exactly the weights the gate-off arm started from (asserted in --selfcheck).

TWO PAIRED COMPARISONS per panel {D}, same seeds, same trainer, same 80 epochs and patience,
d = pure - reference:
  vs learned gate  results/single/encoder__{D}__seed{S}.json
                   what the whole graph buys, mixing plus degree        GRAPH HELPS / graph HURTS
  vs gate-off      ablation/single/encoder__{D}__seed{S}__gateoff.json
                   what the degree term alone buys                      DEGREE HELPS / degree HURTS
Verdict rule as run_gate_ablation.report: within noise when n < 2, sd == 0 or |mean| < sd.

Nothing here is a scored result. Every record carries protocol="EXPLORATORY" and lands in experiments/
only (norm_probe._dump refuses any other path). train_one writes nothing itself and this script never
calls the loop's write_* functions, so results/ and ablation/ are only read.

  conda run --no-capture-output -n ebola-train python -m experiments.pure_tcn --selfcheck
  conda run --no-capture-output -n ebola-train python -m experiments.pure_tcn
  conda run --no-capture-output -n ebola-train python -m experiments.pure_tcn --report

--datasets defaults to covid_us-states influenza_japan; pass one name to run or report one panel. One
JSON per seed as it finishes, so a crash costs one seed; a seed whose file exists is skipped unless
--force, so with the covid seeds on disk the plain run trains influenza_japan only. The gate-off seeds
took 0.4 to 0.6 min each on covid and 1.6 to 2.7 min each on influenza_japan, 10.5 min for five
(results/reports/gate_ablation.log:19-23 and 4-8). The summary is ONE file keyed by dataset, and a
panel already in it must be reproduced exactly from disk or nothing is written.
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
import torch.nn as nn

import bundles
import train.loop as L
from ablation.run_epi_ablation import (LOWER_BETTER, MODEL, POINT, _seed_from, field, load_arm,
                                       mean_sd, paired_delta)
from experiments.norm_probe import OUT, TAG, _dump
from models import LTR, Adapter, SharedEncoder, normalise_adj, sparse_from_dense_np
from results_paths import rpath

DATASETS = ["covid_us-states", "influenza_japan"]   # ponytail: the two panels run so far. --datasets takes
                                                   # any DEV bundle; the others are unrun here.
SUMMARY = "pure_tcn__summary.json"
KEEP = ("model", "field", "seeds", "tallies", "cells")   # what a rerun must reproduce; ceiling is prose


def pure_glob(ds):
    return OUT / f"pure_tcn__{ds}__seed*.json"


def comparisons(ds):
    # (reference name, record glob, verdict when the removed part helped, verdict when it hurt)
    return (("learned", rpath(f"encoder__{ds}__seed*.json"), "GRAPH HELPS", "graph HURTS"),
            ("gate-off", rpath(f"encoder__{ds}__seed*__gateoff.json", root="ablation"),
             "DEGREE HELPS", "degree HURTS"))


class NoDegree(nn.Module):
    """Stands in for LTR. No parameters and no RNG draw, so the trunk computes tcn(Z) + 0.0."""
    def forward(self, deg):
        return 0.0


def factory(b, gate_mode, device):
    """encoder_factory for train.loop.train_one: the gate-off trunk with the degree term removed."""
    assert gate_mode == "off", f"pure-TCN is the gate-off trunk minus LTR, got gate_mode={gate_mode!r}"
    enc = SharedEncoder(gate_mode="off").to(device)   # full build, so LTR's init draws are still spent
    enc.ltr = NoDegree()                              # and TCN, spatial, Adapter init match gate-off
    # ponytail: two ceilings. (1) The spatial mixer is still built and still runs forward; g=0 multiplies
    # its output by zero, so it gets zero gradient, exactly as in the gate-off arm. (2) Init is matched to
    # gate-off, the training path is not: the forward loses ltr(deg) from step one, and LTR's 128
    # parameters leave the clip_grad_norm_ set (loop.py:192, 227), so the clip scale differs too.
    return enc, dict(encoder_version="v1_pure_tcn", ltr="off")


def seed_file(ds, s):
    return f"pure_tcn__{ds}__seed{s}.json"


def pending(ds, seeds, force, out=OUT):
    """Seeds still to train. Resume keys on the artifact: a seed is done when its file exists."""
    todo = []
    for s in seeds:
        if (out / seed_file(ds, s)).exists() and not force:
            print(f"  {ds} seed {s}: {out / seed_file(ds, s)} exists, skipped (--force retrains it)")
        else:
            todo.append(s)
    return todo


def verdict(dm, dsd, n, m, helps, hurts):
    """run_gate_ablation.report's rule on d = pure - reference: getting WORSE without a part means
    that part was helping."""
    if n < 2 or dsd == 0 or abs(dm) < dsd:
        return "within noise"
    removed_helped = (dm > 0) if m in LOWER_BETTER else (dm < 0)
    return helps if removed_helped else hurts


def _seeds(arm):
    return sorted({s for v in arm.values() for s in v})


def report(datasets):
    blocks = {}
    for ds in datasets:
        block = report_one(ds)
        if block:
            blocks[ds] = block
    return write_summary(blocks, datasets) if blocks else None


def write_summary(blocks, datasets):
    """ONE file keyed by dataset. A panel of this run already in the file must come back identical on
    KEEP, or nothing is written: a recorded result never changes as a side effect of a report. Panels
    not in this run are carried over untouched."""
    f = OUT / SUMMARY
    old = json.loads(f.read_text()) if f.exists() else {}
    old = {old["dataset"]: old} if "dataset" in old else old   # the 2026-09-25 covid-only file is flat
    canon = lambda x: json.dumps(x, sort_keys=True)            # exact on floats and NaN, tuple == list
    for ds in sorted(set(old) & set(datasets)):
        bad = [k for k in KEEP if ds not in blocks or canon(blocks[ds].get(k)) != canon(old[ds].get(k))]
        if bad:
            print(f"REFUSED to write {f}: {ds} regenerated from disk differs from the file on {bad}. The "
                  f"seed files changed since it was written. If that was deliberate (--force), move the "
                  f"old summary aside and rerun --report.")
            return None
    p = _dump({**old, **blocks}, SUMMARY)
    print(f"wrote {p}: {sorted({**old, **blocks})}, regenerated {sorted(blocks)}")
    return p


def report_one(ds):
    pure = load_arm(pure_glob(ds), _seed_from, MODEL)
    refs = {name: load_arm(pat, _seed_from, MODEL) for name, pat, _, _ in comparisons(ds)}
    print(f"\n{'=' * 100}\nPURE-TCN :: {ds}, no mixing and no degree term   model={MODEL}   "
          f"field={field(ds)}   {TAG}\n{'=' * 100}")
    for name, pat, _, _ in comparisons(ds):
        print(f"  reference {name:<9} {len(_seeds(refs[name]))} seeds {_seeds(refs[name])}  <- {pat}")
    print(f"  pure-TCN           {len(_seeds(pure))} seeds {_seeds(pure)}  <- {pure_glob(ds)}")
    if not pure:
        print("\n  no pure-TCN records yet")
        return None

    cells, tallies = [], {}
    for name, _, helps, hurts in comparisons(ds):
        ref = refs[name]
        tally = {helps: 0, hurts: 0, "within noise": 0}
        print(f"\n  PURE-TCN vs {name.upper()}   d = pure - {name}, paired by seed")
        for m in POINT:
            print(f"    {m.upper()} ({'higher' if m == 'pcc' else 'lower'}=better)")
            print(f"      {'h':>3} | {name:>18} | {'pure-TCN':>18} | {'paired d':>18} | verdict")
            for h in bundles.HORIZONS:
                r, p = ref.get((h, m), {}), pure.get((h, m), {})
                shared = sorted(set(r) & set(p))
                if not shared:                     # nothing to pair: skip, never tally it as noise
                    continue
                # both columns over the SHARED seeds only, so pure - ref reproduces the printed d
                rm, rsd = mean_sd([r[s] for s in shared])
                pm, psd = mean_sd([p[s] for s in shared])
                dm, dsd, n = paired_delta(r, p)
                v = verdict(dm, dsd, n, m, helps, hurts)
                tally[v] += 1
                cells.append(dict(reference=name, metric=m, horizon=h, seeds=shared, ref_mean=rm,
                                  ref_sd=rsd, pure_mean=pm, pure_sd=psd, d_mean=dm, d_sd=dsd, n=n,
                                  verdict=v))
                print(f"      {h:>3} | {rm:9.3f} +-{rsd:6.3f} | {pm:9.3f} +-{psd:6.3f} | "
                      f"{dm:+9.3f} +-{dsd:6.3f} | {v} (n={n})")
        tallies[name] = tally
        print(f"    TALLY vs {name}, {sum(tally.values())} cells: " +
              ", ".join(f"{k} {c}" for k, c in tally.items()))

    n_max = max(c["n"] for c in cells)
    ceiling = (f"CEILING: at most {n_max} paired seeds on {ds}, and seed sd is the only noise scale, so "
               f"|mean| < sd is a weak rule. Sign and spread only, no significance claim.")
    print(f"\n  {ceiling}\n")
    return dict(protocol=TAG, dataset=ds, model=MODEL, field=field(ds),
                seeds=dict(pure=_seeds(pure), **{k: _seeds(v) for k, v in refs.items()}),
                tallies=tallies, cells=cells, ceiling=ceiling)


def selfcheck(datasets):
    """Every check carries a planted control that must fail, or the check proves nothing."""
    for ds in datasets:                    # checks 1 and 2 on every panel: each has its own graph
        b = bundles.load(ds)
        t = int(b.origins(phase="test")[0])
        Z = torch.tensor(b.transfer_view()[:, t - 19:t + 1, :])       # [N, 20, 4], real panel data
        Mt = torch.tensor(b.M[:, t], dtype=torch.float32)
        A_real = sparse_from_dense_np(b.A_geo)
        A_none = sparse_from_dense_np(np.zeros_like(b.A_geo))          # +I makes every degree 1

        torch.manual_seed(0)
        enc, meta = factory(b, "off", "cpu")
        torch.manual_seed(0)
        plain = SharedEncoder(gate_mode="off")                          # same TCN weights, LTR kept
        enc.eval(); plain.eval()
        with torch.no_grad():
            out_r, out_0, tcn = enc(Z, A_real, Mt), enc(Z, A_none, Mt), enc.tcn(Z)
            pl_r, pl_0 = plain(Z, A_real, Mt), plain(Z, A_none, Mt)

        # 1. graph independence, first, because it is the claim the arm exists to make
        assert torch.equal(out_r, out_0), (f"{ds}: pure-TCN output moved when the graph was emptied "
                                           f"(max diff {float((out_r - out_0).abs().max()):.3g}): the "
                                           f"arm still sees A")
        assert not torch.allclose(pl_r, pl_0, atol=1e-6), \
            f"CONTROL void on {ds}: gate-off output did not move when the graph was emptied"
        print(f"ok {ds}: pure-TCN output identical under the real A and an all-zero A; control: "
              f"gate-off moves by up to {float((pl_r - pl_0).abs().max()):.3f}")
        print(f"   {ds} graph: N={b.A_geo.shape[0]}, sum(A)={float(b.A_geo.sum()):g}, "
              f"max degree incl. self-loop {float(normalise_adj(A_real)[1].max()):g}")

        # 2. nothing but the TCN reaches the output
        assert torch.allclose(out_r, tcn, atol=1e-6), f"{ds}: pure-TCN output is not enc.tcn(Z)"
        assert not torch.allclose(pl_r, plain.tcn(Z), atol=1e-6), \
            f"CONTROL void on {ds}: gate-off output equals its own tcn(Z), so the check cannot see LTR"
        assert meta == dict(encoder_version="v1_pure_tcn", ltr="off"), meta
        assert not list(enc.ltr.parameters()) and list(plain.ltr.parameters()), "LTR params not removed"
        print(f"ok {ds}: pure-TCN output equals enc.tcn(Z) (atol 1e-6), LTR holds no parameters; "
              f"control: gate-off output differs from its tcn(Z)")

    # 3. no CPU RNG drawn by the swap, so TCN, spatial and Adapter init match gate-off at the same seed
    def same(x, y):
        return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)
    torch.manual_seed(7); e1, _ = factory(b, "off", "cpu"); st1 = torch.random.get_rng_state(); a1 = Adapter()
    torch.manual_seed(7); e2 = SharedEncoder(gate_mode="off"); st2 = torch.random.get_rng_state(); a2 = Adapter()
    assert torch.equal(st1, st2), "the factory drew CPU RNG that a plain gate-off build does not"
    assert same(e1.tcn.state_dict(), e2.tcn.state_dict()), "TCN init differs from gate-off"
    assert same(e1.spatial.state_dict(), e2.spatial.state_dict()), "spatial init differs from gate-off"
    assert same(a1.state_dict(), a2.state_dict()), "Adapter init differs from gate-off"
    torch.manual_seed(7); SharedEncoder(gate_mode="off"); LTR(); a3 = Adapter()   # a rebuilt LTR draws
    assert not torch.equal(torch.random.get_rng_state(), st2) and not same(a3.state_dict(), a2.state_dict()), \
        "CONTROL void: an extra LTR build did not move the RNG state or the Adapter init"
    print("ok factory draws no CPU RNG: RNG state, TCN, spatial and Adapter init match gate-off exactly; "
          "control: one rebuilt LTR shifts both")

    # 4. outputs land in experiments/ only, and an existing seed is skipped unless --force
    ds = datasets[0]
    for f in (*[seed_file(d, 42) for d in datasets], SUMMARY):
        assert (OUT / f).resolve().parent == OUT.resolve(), f"{f} resolves outside experiments/"
    try:
        _dump({}, "../pure_tcn_escape_probe.json")
        raise SystemExit("CONTROL void: _dump wrote outside experiments/")
    except AssertionError:
        pass
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert pending(ds, [42, 52], False, out=d) == [42, 52], "CONTROL void: an empty dir skipped a seed"
        (d / seed_file(ds, 42)).write_text("[]")
        assert pending(ds, [42, 52], False, out=d) == [52], "an existing seed file was not skipped"
        assert pending(ds, [42, 52], True, out=d) == [42, 52], "--force did not retrain an existing seed"
    print(f"ok {seed_file(ds, 42)} and {SUMMARY} resolve inside experiments/, existing seed skipped, "
          f"--force retrains; control: _dump refuses ../")

    # 5. the reader can key the new filenames, and the verdict points the right way
    assert _seed_from(seed_file(ds, 42)) == 42, "seed must parse from the pure-TCN filename"
    assert _seed_from(f"pure_tcn__{ds}.json") is None, "CONTROL void: a seedless name parsed"
    assert verdict(+5.0, 1.0, 5, "rmse", "H", "X") == "H", "rmse up without the part => it helped"
    assert verdict(-0.05, 0.01, 5, "pcc", "H", "X") == "H", "pcc down without the part => it helped"
    assert verdict(-5.0, 1.0, 5, "rmse", "H", "X") == "X", "rmse down without the part => it hurt"
    assert verdict(+5.0, 9.0, 5, "rmse", "H", "X") == "within noise", "CONTROL: inside its own spread"
    print(f"ok _seed_from({seed_file(ds, 42)!r}) == 42; verdict signs right for rmse and pcc; control: "
          f"seedless name and |mean| < sd both refused")
    print("selfcheck passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=bundles.DEV_BUNDLE_NAMES, default=DATASETS)
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
        todo = pending(ds, a.seeds, a.force)
        print(f"\nPURE-TCN {ds}: {len(todo)} of {len(a.seeds)} seeds to train")
        for s in todo:
            t0 = time.time()
            recs, _, _, _ = L.train_one(ds, s, gate_mode="off", encoder_factory=factory,
                                        verbose=False, gate_read=False)
            # ponytail: records only. No pernode, quantiles or checkpoint; add them when a per-node
            # test needs them, a rerun is a few minutes per seed.
            for r in recs:
                r["protocol"] = TAG
            bad = {(r.get("gate_mode"), r.get("ltr")) for r in recs} - {("off", "off")}
            assert not bad, f"{ds} seed {s}: records claim (gate_mode, ltr) = {bad}, expected ('off', 'off')"
            p = _dump(recs, seed_file(ds, s))
            print(f"  {ds} seed {s} done in {(time.time() - t0) / 60:.1f} min, {len(recs)} records -> {p}")
    report(a.datasets)


if __name__ == "__main__":
    main()

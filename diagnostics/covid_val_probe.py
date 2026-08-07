"""Probe: is COVID's single-disease failure caused by selecting the checkpoint on the Omicron val fold?

Doubt.md measures COVID losing to a per-node constant by up to 131% at h>=10. Three candidate causes
were separated by diagnosis; this tests the one that is actually actionable -- MODEL SELECTION. The
val fold (weeks 82-115) owns the Omicron peak: it runs 2.8x the level of both other folds and its peak
week is 6.3x the largest test week. Early stopping and the sec 3.1 bias_c are both fitted on it.

THREE ARMS, one variable each:

  A  baseline     the shipped split. Must reproduce the canonical seed-matched record on disk, which
                  is the harness self-check -- if A drifts, the mask patch below is broken.
  B  val-from-train   val = the last `--val-frac` of the TRAIN window; test untouched. This is the
                  "proper protocol" fix, but it SHRINKS TRAIN, so B vs A moves two things at once.
  C  val-minus-omicron   val stays the shipped window with the surge weeks excised; train untouched.
                  This is the clean single-variable test of selection-on-Omicron. Read C first.

C isolates what B confounds. If C recovers most of the gap, selection is the cause and the fix is
cheap. If neither moves, the cause is the test fold having no long-horizon signal to predict (test
lag-10 autocorrelation is +0.015 against train's +0.350) and no split repairs that.

WRITES NOTHING TO results/. It calls train.loop.train_one directly and never run_dataset, which would
overwrite the canonical encoder__covid_us-states__seed*.json 5-seed records.

  python covid_val_probe.py --seeds 42 52 62

Run in `ebola-train` (repaired; Doubt.md sec 6). locf_probe.py's "use ebola" header predates that fix.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json

import numpy as np

import bundles

NAME = "covid_us-states"
_REAL_LOAD = bundles.load


def _recarve(b, mode, val_frac):
    """Return new [N,T] masks. `test` is returned untouched in every mode -- asserted by the caller."""
    m = {k: v.copy() for k, v in b.masks().items()}
    split = b.meta["split"]
    tr_end, va_end = split["train_end"], split["val_end"]
    T = b.raw.shape[1]
    t = np.arange(T)
    if mode == "baseline":
        return m, "shipped split"
    if mode == "val_from_train":
        cut = int(round(tr_end * (1.0 - val_frac)))
        m["train"] = (np.tile(t < cut, (b.raw.shape[0], 1)).astype(np.uint8) & b.M)
        m["val"] = (np.tile((t >= cut) & (t < tr_end), (b.raw.shape[0], 1)).astype(np.uint8) & b.M)
        return m, f"train=[0,{cut}) val=[{cut},{tr_end})"
    if mode == "val_minus_omicron":
        # Excise from val the contiguous run around its peak that exceeds ANYTHING the test fold
        # contains. Defining the cut by the test fold's own maximum keeps it from being a hand-picked
        # window: we remove exactly the regime the model will never be asked to predict.
        nat = b.raw.sum(0)
        test_max = nat[va_end:].max()
        peak = int(nat[tr_end:va_end].argmax()) + tr_end
        lo = peak
        while lo - 1 >= tr_end and nat[lo - 1] > test_max:
            lo -= 1
        hi = peak
        while hi + 1 < va_end and nat[hi + 1] > test_max:
            hi += 1
        drop = (t >= lo) & (t <= hi)
        m["val"] = (m["val"] & ~np.tile(drop, (b.raw.shape[0], 1))).astype(np.uint8)
        return m, f"val minus weeks [{lo},{hi}] (peak {nat[peak]:,.0f} > test max {test_max:,.0f})"
    raise ValueError(mode)


def patched_load(mode, val_frac):
    def _load(name):
        b = _REAL_LOAD(name)
        if name != NAME:
            return b
        base_test = b.masks()["test"].copy()
        m, desc = _recarve(b, mode, val_frac)
        # Invariants. These run on every arm; a broken patch fails here, not silently in the scores.
        assert np.array_equal(m["test"], base_test), "test fold moved -- arms are not comparable"
        assert not (m["train"] & m["val"]).any(), "train/val overlap"
        assert not (m["val"] & m["test"]).any(), "val/test overlap"
        assert m["train"].any() and m["val"].any(), "empty train or val fold"
        b._masks.update(m)
        _load.desc = desc
        return b
    _load.desc = ""
    return _load


def run_arm(mode, seeds, epochs, val_frac, patience):
    import train.loop as L
    loader = patched_load(mode, val_frac)
    bundles.load = loader
    try:
        b = loader(NAME)
        n_tr, n_va = len(b.origins(phase="train")), len(b.origins(phase="val"))
        print(f"\n=== arm {mode}  ({loader.desc})", flush=True)
        print(f"    origins: train={n_tr} val={n_va} test={len(b.origins(phase='test'))}", flush=True)
        out = {}
        for s in seeds:
            t0 = time.time()
            recs, _, _, _ = L.train_one(NAME, s, epochs=epochs, patience=patience, verbose=False,
                                        gate_read=False)
            for r in recs:
                if r["metric"] == "rmse":
                    out.setdefault((r["model"], r["horizon"]), {})[s] = r["node_mean"]
            print(f"    seed{s} done in {(time.time()-t0)/60:.1f} min", flush=True)
        return out, n_tr
    finally:
        bundles.load = _REAL_LOAD


def canonical(seeds):
    """Seed-matched RMSE already on disk -- arm A must reproduce this."""
    out = {}
    for s in seeds:
        p = f"results/single/encoder__{NAME}__seed{s}.json"
        if not os.path.exists(p):
            continue
        for r in json.loads(open(p, encoding="utf-8").read()):
            if r["metric"] == "rmse" and r["model"] == "encoder":
                out[(r["horizon"], s)] = r["node_mean"]
    return out


def floors():
    p = f"results/naive/naive__{NAME}.json"
    out = {}
    for r in json.loads(open(p, encoding="utf-8").read()):
        if r["metric"] == "rmse":
            out[(r["model"], r["horizon"])] = r["node_mean"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="arm B: fraction of the TRAIN window carved off as val")
    a = ap.parse_args()

    arms = {}
    ntr = {}
    for mode in ("baseline", "val_from_train", "val_minus_omicron"):
        arms[mode], ntr[mode] = run_arm(mode, a.seeds, a.epochs, a.val_frac, a.patience)

    can, fl = canonical(a.seeds), floors()
    print("\n" + "=" * 78)
    print("HARNESS CHECK -- arm A must reproduce the canonical records seed-for-seed")
    ok = True
    for h in (3, 5, 10, 15):
        for s in a.seeds:
            got, want = arms["baseline"].get(("encoder", h), {}).get(s), can.get((h, s))
            if want is None:
                continue
            d = abs(got - want) / want
            if d > 0.005:
                ok = False
                print(f"  MISMATCH h{h} seed{s}: probe {got:,.0f} vs disk {want:,.0f} ({d:+.2%})")
    print("  arm A reproduces disk" if ok else "  *** arm A DRIFTED -- treat every number below as suspect")

    print("\n" + "=" * 78)
    print(f"COVID single-disease test RMSE (node_mean), mean over seeds {a.seeds}")
    print(f"train origins: baseline={ntr['baseline']}  val_from_train={ntr['val_from_train']}  "
          f"val_minus_omicron={ntr['val_minus_omicron']}\n")
    for model in ("encoder", "encoder_mc"):
        print(f"-- {model}")
        print(f"{'h':>4} {'A baseline':>12} {'B val<-train':>13} {'C val-omicron':>14} "
              f"{'train_mean':>11} {'best floor':>11}   {'C vs A':>8}")
        for h in (3, 5, 10, 15):
            vals = {}
            for mode in arms:
                d = arms[mode].get((model, h), {})
                vals[mode] = np.mean(list(d.values())) if d else float("nan")
            tm = fl.get(("train_mean", h), float("nan"))
            bf = min(v for (m, hh), v in fl.items() if hh == h)
            cva = 100.0 * (vals["val_minus_omicron"] - vals["baseline"]) / vals["baseline"]
            print(f"h{h:<3d} {vals['baseline']:12,.0f} {vals['val_from_train']:13,.0f} "
                  f"{vals['val_minus_omicron']:14,.0f} {tm:11,.0f} {bf:11,.0f}   {cva:+7.1f}%")
        print()
    print("Negative 'C vs A' = excising Omicron from the selection fold HELPED.")
    print("A cell only counts as fixed if it also lands at or below train_mean.")
    print("\nCaveat: the scaler is the frozen headline scaler, fitted on the ORIGINAL train window, so")
    print("in arm B it saw the weeks now used as val. Test cells are untouched in every arm, so the")
    print("test scores compare cleanly; only arm B's val loss is mildly optimistic.")


if __name__ == "__main__":
    main()

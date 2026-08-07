"""Probe: does the proposed COVID split (train_end=107, val_end=126) make the panel winnable?

The shipped fixed_50_20_30 split puts the Omicron peak (2022-01-15, 5.14M national) in the VAL fold,
which is where early stopping and the sec 3.1 bias_c are both fitted. Doubt.md measures the resulting
model losing to a per-node constant by up to 131%. covid_val_probe.py established by a clean
single-variable test that the selection fold is worth 15-50% of that.

The proposed split moves Omicron into TRAIN. Nothing is excluded -- every week stays in the data:

    train [0,107)   2020-02-01 .. 2022-02-12   contains the Omicron peak
    val   [107,126) 2022-02-19 .. 2022-06-25   post-Omicron, level-matched to test (1.07x)
    test  [126,164) 2022-07-02 .. 2023-03-18   unchanged in character, 38 weeks

Changing train_end changes the SCALER, so this calls Bundle.refit() on the new train mask rather than
reusing the frozen headline scaler. refit() rebuilds X[:,:,0] as well as y, so the trunk's inputs move
with it -- reusing the shipped scaler here would leak the old train window into the new arm's inputs.

THE COMPARISON IS MODEL-vs-FLOOR, NOT RMSE-vs-RMSE. The two splits have different test folds, so their
absolute RMSEs are not comparable and are never differenced here. What transfers across splits is the
RATIO of the model to the naive floor computed on that split's own test cells.

WRITES NOTHING TO results/. Calls train.loop.train_one directly, never run_dataset.

  python covid_split_probe.py --seeds 42 52 62 72 82

Run in `ebola-train`.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import bundles

NAME = "covid_us-states"
_REAL_LOAD = bundles.load
NEW_TR, NEW_VA = 107, 126
HOR = (3, 5, 10, 15)


def new_bundle(name):
    b = _REAL_LOAD(name)
    if name != NAME:
        return b
    N, T = b.raw.shape
    t = np.arange(T)
    m = {"train": np.tile(t < NEW_TR, (N, 1)).astype(np.uint8) & b.M,
         "val": np.tile((t >= NEW_TR) & (t < NEW_VA), (N, 1)).astype(np.uint8) & b.M,
         "test": np.tile(t >= NEW_VA, (N, 1)).astype(np.uint8) & b.M}
    assert not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any()
    assert (((m["train"] | m["val"] | m["test"]) > 0) == (b.M > 0)).all(), \
        "folds must partition observed cells"
    b._masks.update(m)
    # train_end moved -> the scaler must move with it. refit() rebuilds X[:,:,0] too, so the trunk's
    # INPUTS are on the new scale as well; keeping the shipped scaler would feed the old train
    # window's normalisation into an arm whose whole point is a different train window.
    X, y, scaler = b.refit(m["train"])
    b.X, b.y, b.scaler = X, y, scaler
    return b


def floors_for(b, seed=0):
    """Naive floors scored on this bundle's own test cells, through the same scoring path."""
    import train.loop as L
    te = b.origins(phase="test")
    preds, fb = L.naive_predictions(b, te)
    out = {}
    for model, by_h in preds.items():
        recs, _, _ = L.score_predictions(model, NAME, seed, by_h, b, te)
        for r in recs:
            if r["metric"] == "rmse":
                out[(model, r["horizon"])] = r["node_mean"]
    return out, fb


def disk_current():
    """The shipped split's model and floor numbers, already on disk."""
    enc = {}
    for p in sorted(os.listdir("results/single")):
        if not (p.startswith(f"encoder__{NAME}__seed") and p.endswith(".json")):
            continue
        for r in json.loads(open(f"results/single/{p}", encoding="utf-8").read()):
            if r["metric"] == "rmse":
                enc.setdefault((r["model"], r["horizon"]), []).append(r["node_mean"])
    fl = {}
    for r in json.loads(open(f"results/naive/naive__{NAME}.json", encoding="utf-8").read()):
        if r["metric"] == "rmse":
            fl[(r["model"], r["horizon"])] = r["node_mean"]
    return enc, fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    a = ap.parse_args()

    import train.loop as L
    bundles.load = new_bundle
    try:
        b = new_bundle(NAME)
        print(f"NEW SPLIT train_end={NEW_TR} val_end={NEW_VA}")
        print(f"  origins train={len(b.origins(phase='train'))} val={len(b.origins(phase='val'))} "
              f"test={len(b.origins(phase='test'))}")
        print(f"  refit scaler: median per-node std {np.median(b.scaler['std']):.3f} "
              f"(shipped was 2.405)  amplification {np.exp(np.median(b.scaler['std'])):.1f}x")
        nat = b.raw.sum(0)
        print(f"  national max  train={nat[:NEW_TR].max():,.0f}  val={nat[NEW_TR:NEW_VA].max():,.0f}  "
              f"test={nat[NEW_VA:].max():,.0f}  -> no extrapolation\n", flush=True)

        nf, fb = floors_for(b)
        print(f"  naive floors scored (seasonal fallback rate {fb:.1%})\n", flush=True)

        got = {}
        for s in a.seeds:
            t0 = time.time()
            recs, _, _, _ = L.train_one(NAME, s, epochs=a.epochs, patience=a.patience,
                                        verbose=False, gate_read=False)
            for r in recs:
                if r["metric"] == "rmse":
                    got.setdefault((r["model"], r["horizon"]), []).append(r["node_mean"])
            print(f"    seed{s} done in {(time.time()-t0)/60:.1f} min", flush=True)
    finally:
        bundles.load = _REAL_LOAD

    denc, dfl = disk_current()

    print("\n" + "=" * 84)
    print(f"PROPOSED SPLIT {NEW_TR}/{NEW_VA} -- COVID single-disease, test RMSE, {len(a.seeds)} seeds")
    print(f"{'h':>4} {'encoder':>16} {'encoder_mc':>16} {'persistence':>12} {'train_mean':>11} "
          f"{'seasonal':>10}")
    for h in HOR:
        e, mc = np.array(got[("encoder", h)]), np.array(got[("encoder_mc", h)])
        print(f"h{h:<3d} {e.mean():9,.0f} ±{e.std(ddof=1):5,.0f} {mc.mean():9,.0f} ±{mc.std(ddof=1):5,.0f} "
              f"{nf[('persistence',h)]:12,.0f} {nf[('train_mean',h)]:11,.0f} {nf[('seasonal',h)]:10,.0f}")

    print("\n" + "=" * 84)
    print("THE COMPARABLE QUANTITY: model / best naive floor on that split's OWN test fold.")
    print("Absolute RMSE is NOT comparable across splits -- the test folds differ. <1.00 = model wins.\n")
    print(f"{'h':>4} | {'CURRENT 82/115':^27} | {'PROPOSED 107/126':^27}")
    print(f"{'':>4} | {'encoder':>12} {'encoder_mc':>13} | {'encoder':>12} {'encoder_mc':>13}")
    for h in HOR:
        cbf = min(v for (m, hh), v in dfl.items() if hh == h)
        nbf = min(v for (m, hh), v in nf.items() if hh == h)
        ce = np.mean(denc[("encoder", h)]) / cbf
        cm = np.mean(denc[("encoder_mc", h)]) / cbf
        ne = np.mean(got[("encoder", h)]) / nbf
        nm = np.mean(got[("encoder_mc", h)]) / nbf
        star = lambda x: f"{x:.2f}" + ("  WIN" if x < 1.0 else "     ")
        print(f"h{h:<3d} | {star(ce):>12} {star(cm):>13} | {star(ne):>12} {star(nm):>13}")

    print("\nAgainst train_mean specifically (the constant that beat us by up to 131%):")
    print(f"{'h':>4} | {'CURRENT':>18} | {'PROPOSED':>18}")
    for h in HOR:
        c = np.mean(denc[("encoder_mc", h)]) / dfl[("train_mean", h)]
        n = np.mean(got[("encoder_mc", h)]) / nf[("train_mean", h)]
        print(f"h{h:<3d} | {c:17.2f}x | {n:17.2f}x")


if __name__ == "__main__":
    main()

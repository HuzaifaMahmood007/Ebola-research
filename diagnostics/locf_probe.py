"""A/B probe for Doubt.md §3.2: does LOCF-filling the trunk's inputs improve dengue->flu transfer?

Calls train.lodo internals directly and WRITES NOTHING to results/. run_ldo_fold() would overwrite the
canonical encoder_ldo__* records the held 5-seed matrix lives in, so this deliberately does not use it.

Both arms share seed, trunk steps and adapter epochs; the ONLY difference is bundles.LOCF_INPUT. The
three influenza panels are 100% observed, so LOCF is a no-op on them -- every difference reported below
comes from the dengue TRUNK's input distribution, which is exactly the hypothesis under test.

Read the `zeroshot` block first: that is the arm the -311% collapse was measured on, and the arm the
1,428-parameter adapter cannot rescue.

  python locf_probe.py --steps 3000 --epochs 15 --seed 42

Run it in the `ebola` conda env, NOT `ebola-train` -- see Doubt.md §6, ebola-train's numpy 2.4.6 hard-
crashes the process inside np.corrcoef (score._pcc), which is reached by every scoring call here.
"""
import argparse
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import bundles
import train.lodo as L
from train.joint import _test_dataset

META = dict(training_regime="ldo", sampler="uniform-uniform", gate_mode="learned",
            topo_aug="none", fold_structure="leave-one-disease-out", ldo_direction="dengue2flu")


def arm(locf, seed, steps, epochs, device):
    bundles.LOCF_INPUT = locf                      # transfer_view() reads this global at call time
    b = bundles.load("dengue")
    zero_mass = float((b.transfer_view()[:, :, 0] == 0).mean())
    print(f"\n=== arm LOCF={locf}  (dengue exact-zero input mass {zero_mass:.1%}) ===", flush=True)
    t0 = time.time()
    enc, in_ads = L._fit_trunk(seed, ["dengue"], device, steps=steps, val_every=max(steps // 6, 1),
                               verbose=False)
    ad, ds = L._fit_shared_adapter(enc, list(L.FLU_NAMES), seed, device, epochs=epochs, verbose=False)
    borrowed = L._mean_adapter(in_ads, device)     # dengue's own adapter, unfitted on flu = zero-shot
    out = {}
    for d in ds:
        for tag, head in (("adapted", ad), ("zeroshot", borrowed)):
            recs, _, _, _ = _test_dataset(enc, head, d, seed, f"probe:{tag}", META)
            for r in recs:
                if r["metric"] == "rmse":
                    out[(d.name, tag, r["horizon"])] = r["node_mean"]
    print(f"    arm done in {(time.time()-t0)/60:.1f} min", flush=True)
    return out, zero_mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000, help="trunk steps per arm (default 91000)")
    ap.add_argument("--epochs", type=int, default=15, help="shared-adapter epochs (default 80)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    device = L.DEVICE
    print(f"device={device} seed={a.seed} trunk_steps={a.steps} adapter_epochs={a.epochs}", flush=True)

    off, zm_off = arm(False, a.seed, a.steps, a.epochs, device)
    on, zm_on = arm(True, a.seed, a.steps, a.epochs, device)

    print(f"\ndengue exact-zero input mass: {zm_off:.1%} (zero-fill) -> {zm_on:.1%} (LOCF)")
    print("\nRMSE, dengue trunk -> flu.  lower is better; delta<0 means LOCF helped.")
    print(f"{'dataset':22s} {'head':9s} {'h':>3s} {'zero-fill':>10s} {'LOCF':>10s} {'delta':>9s} {'%':>7s}")
    wins = {"adapted": [0, 0], "zeroshot": [0, 0]}
    for k in sorted(off):
        ds, tag, h = k
        a0, a1 = off[k], on[k]
        pct = 100.0 * (a1 - a0) / a0 if a0 else float("nan")
        wins[tag][0 if a1 < a0 else 1] += 1
        print(f"{ds:22s} {tag:9s} {h:3d} {a0:10.1f} {a1:10.1f} {a1-a0:+9.1f} {pct:+6.1f}%")
    for tag, (w, l) in wins.items():
        mean_pct = np.mean([100.0 * (on[k] - off[k]) / off[k] for k in off if k[1] == tag])
        print(f"\n{tag:9s}: LOCF better in {w}/{w+l} cells, mean change {mean_pct:+.1f}%")
    print("\nSingle seed, truncated trunk. Directional only -- not a result, a go/no-go.")


if __name__ == "__main__":
    main()

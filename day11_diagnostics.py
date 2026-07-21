"""Day-11 close-out readings (Week-3 guide §0.8, §0.9). Both are readings of Day 11's own
output, not new modelling. Writes results/day11_diagnostics.json so Task 12.6's ablation and
the [CONFIRM-P7] go/no-go have a recorded number rather than a remembered one.

  1. Ebola support-origin count (§0.8). Under the frozen protocol (support ends at t=7, the
     earliest target is t+3=22) support and target windows are disjoint -> 0 support origins,
     which triggers [CONFIRM-P7]. This is schedule metadata, not a query peek (§0.5 intact).

  2. Node-level-mean variance of X[:,:,0] over the FIT window, per dataset (§0.9). The per-node
     z-score guarantees mean~0 only over the cells the scaler was fit on, so the diagnostic is
     computed there, not over all-observed (where val/test drift makes it ~1 -- itself the reason
     §0.3 refits at rolling origins). "Expect ~0" is a *per-node-scaler* property:
       - the four development diseases use per_node_train  -> ~0 (machine-zero for dense influenza);
       - ebola uses per_disease_support (one pooled mean/std, few-shot design) -> NOT ~0, and that
         is correct. Recorded with scaler_scope so the ebola value is not mistaken for a defect.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import bundles

RESULTS = Path("results")


def fit_window_mean_variance(b: bundles.Bundle) -> tuple[float, int]:
    """Variance across nodes of each node's mean X[:,:,0] over its FIT (train/support) cells.
    Returns (variance, n_nodes_with_a_fit_cell)."""
    x0, m = b.X[:, :, 0], b.masks()
    fit = (m["train"] if "train" in m else m["support"]).astype(bool)
    means = np.array([x0[i, fit[i]].mean() for i in range(x0.shape[0]) if fit[i].any()])
    return float(means.var()), len(means)


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {"origins_by_phase": {}, "fit_window_node_mean_variance": {},
           "scaler_scope": {}, "n_fit_nodes": {}}
    for name in bundles.BUNDLE_NAMES:
        b = bundles.load(name)
        var, n_fit = fit_window_mean_variance(b)
        out["origins_by_phase"][name] = {p: len(b.origins(phase=p)) for p in b.masks()}
        out["fit_window_node_mean_variance"][name] = var
        out["n_fit_nodes"][name] = n_fit
        out["scaler_scope"][name] = b.meta["scaler_scope"]

    ebola_support = out["origins_by_phase"]["ebola"].get("support")
    out["confirm_p7_triggered"] = ebola_support == 0

    (RESULTS / "day11_diagnostics.json").write_text(json.dumps(out, indent=2))

    for name in bundles.BUNDLE_NAMES:
        print(f"{name:22s} scope={out['scaler_scope'][name]:20s} "
              f"fit-window node-mean var={out['fit_window_node_mean_variance'][name]:.3e} "
              f"(n_fit={out['n_fit_nodes'][name]})  origins={out['origins_by_phase'][name]}")
    print(f"\nEbola support origins = {ebola_support}  ->  [CONFIRM-P7] "
          f"{'TRIGGERED (§0.8 applies)' if out['confirm_p7_triggered'] else 'not triggered'}")

    # §0.9: per-node-scaled datasets must be ~0 over the fit window; the pooled ebola scaler is not.
    for name in bundles.DEV_BUNDLE_NAMES:
        assert out["fit_window_node_mean_variance"][name] < 0.05, \
            f"{name}: per-node fit-window var {out['fit_window_node_mean_variance'][name]:.3e} should be ~0"
    assert out["scaler_scope"]["ebola"] == "per_disease_support", "ebola scaler is not disease-pooled"
    print("ok  per-node-scaled dev diseases ~0 over fit window; ebola pooled by design; "
          "no second normalisation needed inside the encoder (§0.9)")


if __name__ == "__main__":
    main()

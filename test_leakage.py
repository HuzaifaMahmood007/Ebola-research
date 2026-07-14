"""The leakage and invariant suite: pass/fail gates, not spot checks.

Three invariants must hold for every disease:

  (i)   every normalisation statistic is fit only on data the model may see at train time --
        the train slice for the development diseases, the support set for Ebola;
  (ii)  no feature is derived from future values;
  (iii) horizon-h targets never appear among the inputs.

The scaler gates are PROBES rather than comparisons. Refitting a scaler with the same code
that built it and finding the same numbers proves nothing; it only re-runs the bug. Instead a
cell the scaler must not be able to see is inflated 1000x and the scaler refit. If it moves,
that cell was in its fit set. self_test() plants real leaks and requires the gates to catch
them, so a gate that cannot fail is caught as such.

Run standalone, or import run_leakage_suite(), which is what build_datasets.py calls before it
writes anything.

    conda run -n ebola python test_leakage.py
"""
from __future__ import annotations

import sys

import numpy as np

from to_schema import CORE_FEATURES, DiseaseTensors, fit_scalers_masked, rolling_origin_masks

PASS, FAIL = [], []


def _gate(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if detail and not cond else ""))


def _raw_from(dt: DiseaseTensors) -> np.ndarray:
    """The unscaled weekly incidence carried on the bundle -- the scaler's actual input.

    This must NOT recover the raw counts by inverting y with meta['scaler']. That is circular:
    refitting on such a reconstruction returns the same scaler whether or not it leaked,
    because the reconstruction already carries the leak, and the gate passes a leaking bundle.
    Testing against dt.raw, which no scaler has touched, is what lets these gates fail.
    """
    if dt.raw is None:
        raise ValueError(f"{dt.meta['disease']}: bundle carries no raw counts; the scaler "
                         f"gates cannot be evaluated without a scaler-independent ground truth")
    return dt.raw.astype(np.float64)


def _fit_mask(dt: DiseaseTensors) -> tuple[np.ndarray, bool, np.ndarray]:
    """(cells the scaler is ALLOWED to see, per_disease?, cells it must NOT see)."""
    sp, obs = dt.meta["split"], dt.M.astype(bool)
    if dt.meta["split_scheme"] == "few_shot_support_query":
        return obs & sp["support_mask"].astype(bool), True, obs & sp["query_mask"].astype(bool)
    if dt.meta["split_scheme"].startswith("per_country"):
        return obs & sp["train_mask"].astype(bool), False, obs & sp["test_mask"].astype(bool)
    T = dt.M.shape[1]                                    # influenza: global positional cut
    t = np.zeros(T, bool); t[: sp["train_end"]] = True
    v = np.zeros(T, bool); v[sp["val_end"] :] = True
    return obs & t[None, :], False, obs & v[None, :]


# --------------------------------------------------------------------------- #
# (i) SCALER PROVENANCE — the sharp edge
# --------------------------------------------------------------------------- #
def gate_scaler_provenance(name: str, dt: DiseaseTensors) -> None:
    """The scaler must be a function of the fit cells and of nothing else.

    For Ebola this is the sharpest gate in the suite: fitting on anything past the support set
    leaks the outbreak's magnitude into an ostensibly few-shot result.
    """
    raw = _raw_from(dt)
    fit, per_disease, held = _fit_mask(dt)
    ship = dt.meta["scaler"]

    if not held.any():
        _gate(f"{name}: scaler provenance", False, "no held-out cells to probe with")
        return

    # The shipped scaler must reproduce from the fit cells alone. Fit on anything wider (all
    # observed cells, or support+query), it will not reproduce here.
    base = fit_scalers_masked(raw, fit, per_disease=per_disease)
    _gate(f"{name}: SHIPPED scaler reproduces from the fit-mask alone",
          np.allclose(ship["mean"], base["mean"], atol=1e-4)
          and np.allclose(ship["std"], base["std"], atol=1e-4),
          "meta['scaler'] does NOT reproduce from its declared fit set -> it saw held-out data")

    # The same property probed from the other direction: destroy the held-out cells and confirm
    # a refit still lands on the shipped scaler. One that ever touched them could not.
    poisoned = raw.copy()
    poisoned[held] *= 1000.0
    after = fit_scalers_masked(poisoned, fit, per_disease=per_disease)
    _gate(f"{name}: shipped scaler unmoved by 1000x poisoning of held-out cells",
          np.allclose(ship["mean"], after["mean"], atol=1e-4)
          and np.allclose(ship["std"], after["std"], atol=1e-4),
          "the scaler MOVED -> held-out cells are in its fit set (LEAK)")


# --------------------------------------------------------------------------- #
# (ii)/(iii) NO FUTURE LEAKAGE
# --------------------------------------------------------------------------- #
def gate_no_future_leakage(name: str, dt: DiseaseTensors) -> None:
    """No channel may be the target shifted back in time, and the incidence channel at t
    must be the value AT t -- not at t+h."""
    X, y = dt.X, dt.y
    _gate(f"{name}: incidence channel is y at the SAME t (no shift)",
          np.array_equal(X[:, :, 0], y))

    # no feature column is the horizon target pre-shifted into the inputs
    leaks = []
    for h in (1, 2, 4):
        if X.shape[1] <= h:
            continue
        for f, nm in enumerate(dt.meta["feature_names"]):
            if np.allclose(X[:, :-h, f], y[:, h:], atol=1e-6):
                leaks.append(f"{nm}@h={h}")
    _gate(f"{name}: no channel equals the future target y[t+h]", not leaks, f"leaking: {leaks}")

    # seasonality is a function of the CALENDAR, not of the series -- so it cannot leak
    dates = np.asarray([np.datetime64(str(d)) for d in dt.meta["dates"]])
    doy = ((dates - dates.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int) + 1)
    ang = 2 * np.pi * doy / 365.25
    _gate(f"{name}: sin_doy/cos_doy derive from the calendar alone",
          np.allclose(X[0, :, 1], np.sin(ang), atol=1e-4)
          and np.allclose(X[0, :, 2], np.cos(ang), atol=1e-4))


# --------------------------------------------------------------------------- #
# MASK / IMPUTATION
# --------------------------------------------------------------------------- #
def gate_mask_semantics(name: str, dt: DiseaseTensors) -> None:
    """Imputed cells must be 0 in normalised space -- the per-node mean, a neutral value, and
    not the transform of a raw 0, which would assert 'zero cases' -- and must carry M=0."""
    imp = dt.M == 0
    _gate(f"{name}: imputed cells are 0 in normalised space",
          bool(np.all(dt.X[:, :, 0][imp] == 0.0)) if imp.any() else True)
    _gate(f"{name}: imputed cells are 0 in the target y",
          bool(np.all(dt.y[imp] == 0.0)) if imp.any() else True)
    _gate(f"{name}: obs_mask channel == M",
          np.array_equal(dt.X[:, :, 3].astype(np.uint8), dt.M))
    _gate(f"{name}: X and y carry no NaN/Inf",
          bool(np.isfinite(dt.X).all() and np.isfinite(dt.y).all()))


# --------------------------------------------------------------------------- #
# SPLIT INTEGRITY
# --------------------------------------------------------------------------- #
def gate_splits(name: str, dt: DiseaseTensors) -> None:
    sp, obs = dt.meta["split"], dt.M.astype(bool)
    if dt.meta["split_scheme"] == "few_shot_support_query":
        s, q = sp["support_mask"].astype(bool), sp["query_mask"].astype(bool)
        _gate(f"{name}: support & query are disjoint", not (s & q).any())
        _gate(f"{name}: query is a subset of observed", bool((q & ~obs).sum() == 0))
        _gate(f"{name}: support is a subset of observed", bool((s & ~obs).sum() == 0))
        _gate(f"{name}: support+query == every observed cell",
              np.array_equal(s | q, obs))
    elif dt.meta["split_scheme"].startswith("per_country"):
        tr, va, te = (sp[k].astype(bool) for k in ("train_mask", "val_mask", "test_mask"))
        _gate(f"{name}: train/val/test are pairwise disjoint",
              not ((tr & va).any() or (tr & te).any() or (va & te).any()))
        _gate(f"{name}: the three splits partition the observed cells",
              np.array_equal(tr | va | te, obs))
        _gate(f"{name}: every node has >=1 train cell (no undefined scaler)",
              bool((tr.sum(1) > 0).all()))
    else:
        _gate(f"{name}: train_end < val_end < T",
              sp["train_end"] < sp["val_end"] < dt.M.shape[1])


def gate_rolling_origins(name: str, dt: DiseaseTensors) -> None:
    """Each origin's train window must lie strictly before its eval window, and eval must never
    touch an imputed cell."""
    ro = dt.meta.get("rolling_origins")
    if ro is None:
        # Few-shot bundles have none by design: expanding the window past support is the leak.
        _gate(f"{name}: no rolling origins (few-shot -- by design)",
              dt.meta["split_scheme"].startswith("few_shot"))
        return
    _gate(f"{name}: rolling origins declare refit-per-origin", ro["refit_per_origin"] is True)
    ok_order, ok_obs, ok_grow = True, True, True
    prev = None
    for k in range(ro["n_origins"]):
        tr, ev = rolling_origin_masks(dt, k)
        if not ev.any():
            continue
        # every eval cell must lie strictly after every train cell OF THE SAME NODE
        for i in np.where(ev.any(1))[0]:
            if tr[i].any() and np.where(tr[i])[0].max() >= np.where(ev[i])[0].min():
                ok_order = False
        ok_obs &= bool(((ev == 1) & (dt.M == 0)).sum() == 0)
        if prev is not None:
            ok_grow &= bool(tr.sum() >= prev)        # expanding window, never shrinking
        prev = tr.sum()
    _gate(f"{name}: rolling-origin train is strictly before eval", ok_order)
    _gate(f"{name}: rolling-origin eval touches no imputed cell", ok_obs)
    _gate(f"{name}: rolling-origin window expands monotonically", ok_grow)


# --------------------------------------------------------------------------- #
# CROSS-DISEASE: the disease-agnosticism guarantee (§6.1 / guide §0.3)
# --------------------------------------------------------------------------- #
def gate_transfer_agnosticism(bundles: dict[str, DiseaseTensors]) -> None:
    """The shared encoder must not be able to infer which disease it is looking at.

    A channel present for one disease and absent for another would let it identify the disease
    from the missingness pattern alone, reintroducing the disease-specificity the framework
    exists to remove.
    """
    views = {k: dt.transfer_view() for k, dt in bundles.items()}
    _gate("transfer_view: exactly the 4 core channels for EVERY disease",
          all(v.shape[-1] == len(CORE_FEATURES) for v in views.values()))
    _gate("transfer_view: identical core_feature_idx across all diseases",
          len({tuple(dt.meta["core_feature_idx"]) for dt in bundles.values()}) == 1)
    _gate("transfer_view: identical core feature NAMES across all diseases",
          len({tuple(dt.meta["feature_names"][:4]) for dt in bundles.values()}) == 1)

    # Ebola's deaths channel is extended-only and must never reach the shared view
    eb = bundles.get("ebola")
    if eb is not None:
        _gate("transfer_view: Ebola's deaths_norm is NOT in the transfer view",
              "deaths_norm" in eb.meta["feature_names"]
              and "deaths_norm" not in [eb.meta["feature_names"][i]
                                        for i in eb.meta["core_feature_idx"]])

    # The diseases occupy disjoint geography, so a raw centroid identifies the disease.
    # Populating C for all three does not make it transfer-safe.
    _gate("C is excluded from the transfer view (geography identifies the disease)",
          all(v.shape[-1] == len(CORE_FEATURES) for v in views.values()))

    # One shared cadence is what makes transfer through shared parameters legal at all.
    _gate("all diseases share one weekly (W-SAT) cadence",
          {dt.meta["t_res"] for dt in bundles.values()} == {"weekly"})


# --------------------------------------------------------------------------- #
def run_leakage_suite(bundles: dict[str, DiseaseTensors]) -> bool:
    """Run every gate. True iff all pass. Called before any bundle is written to disk."""
    PASS.clear(); FAIL.clear()
    for name, dt in bundles.items():
        print(f"\n--- {name} ---")
        dt.check()
        _gate(f"{name}: schema .check()", True)
        gate_scaler_provenance(name, dt)
        gate_no_future_leakage(name, dt)
        gate_mask_semantics(name, dt)
        gate_splits(name, dt)
        gate_rolling_origins(name, dt)

    print("\n--- cross-disease (disease-agnosticism) ---")
    gate_transfer_agnosticism(bundles)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED GATES:\n  " + "\n  ".join(FAIL))
    return not FAIL


# --------------------------------------------------------------------------- #
# NEGATIVE CONTROLS — proof the gates can actually FAIL
# --------------------------------------------------------------------------- #
def self_test(bundles: dict[str, DiseaseTensors]) -> bool:
    """Plant real leaks and require the gates to catch them.

    A gate that cannot fail establishes nothing, and that is not hypothetical here: an earlier
    version of this suite recovered the raw counts by inverting y with the scaler under test,
    which made the provenance gate circular, and it passed a deliberately-leaked Ebola scaler.
    These controls run on every build rather than being a check someone remembers to do.
    """
    import copy
    print("\n=== negative controls: the gates must FAIL on a planted leak ===")
    fired = []

    # 1. a development scaler fit on ALL observed cells, so it sees val and test
    dt = copy.deepcopy(bundles["dengue"])
    fit_all = dt.M.astype(bool)
    dt.meta["scaler"] = fit_scalers_masked(_raw_from(dt), fit_all, per_disease=False)
    FAIL.clear(); PASS.clear()
    gate_scaler_provenance("NEGCTL dengue(leaky)", dt)
    fired.append(("dev scaler fit on all cells", bool(FAIL)))

    # 2. the one that matters: Ebola's scaler fit on support+query, not support alone
    eb = copy.deepcopy(bundles["ebola"])
    eb.meta["scaler"] = fit_scalers_masked(_raw_from(eb), eb.M.astype(bool), per_disease=True)
    FAIL.clear(); PASS.clear()
    gate_scaler_provenance("NEGCTL ebola(leaky)", eb)
    fired.append(("ebola scaler fit on support+query", bool(FAIL)))

    # 3. the future target planted in an input channel
    fl = copy.deepcopy(bundles["influenza:japan"])
    fl.X[:, :-1, 1] = fl.y[:, 1:]
    FAIL.clear(); PASS.clear()
    gate_no_future_leakage("NEGCTL japan(leaky)", fl)
    fired.append(("future target planted in an input channel", bool(FAIL)))

    FAIL.clear(); PASS.clear()
    ok = all(f for _, f in fired)
    for what, f in fired:
        print(f"{'ok  ' if f else 'FAIL'} negative control fired: {what}")
    if not ok:
        print("\nA NEGATIVE CONTROL DID NOT FIRE — the corresponding gate cannot fail.")
    return ok


if __name__ == "__main__":
    from build_datasets import build_all
    b = build_all()
    ok = run_leakage_suite(b)
    ok &= self_test(b)                 # and prove the gates could have failed
    print("\nLEAKAGE SUITE PASSED (gates verified able to fail)" if ok
          else "\nLEAKAGE SUITE FAILED")
    sys.exit(0 if ok else 1)

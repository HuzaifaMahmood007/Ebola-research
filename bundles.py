"""bundles.py -- one interface over the five harmonised disease bundles.

data/processed/*.npz are frozen, gated inputs (data_audit.md, schema_spec.md). This module
absorbs the three divergences the Week-3 guide documents -- split representation (masks vs
train_end/val_end integers), channel count (Ebola ships 5, everyone else 4) and node_country
presence (absent for influenza) -- in one place, so no model, metric or experiment file
downstream ever branches on which disease it is holding.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from to_schema import apply_scaler, fit_scalers_masked

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
BUNDLE_NAMES = ["dengue", "influenza_japan", "influenza_us-regions", "influenza_us-states",
                "covid_us-states", "ebola"]
DEV_BUNDLE_NAMES = tuple(n for n in BUNDLE_NAMES if n != "ebola")

W = 20                          # lookback, frozen protocol
HORIZONS = (3, 5, 10, 15)       # frozen protocol, direct multi-horizon
MAX_H = max(HORIZONS)

# Doubt.md §3.2. Unobserved INPUT cells ship as 0.0 in X[:,:,0], and 0.0 in per-node z-space is the
# node's training mean -- so an unobserved week is fed to the trunk as "an average week". The fraction
# of X[:,:,0] that is exactly zero is 78.2% on dengue and 59.0% on ebola, against 0.0% on all three
# influenza panels. A trunk fitted where 78% of inputs sit at exactly 0 is then run where 0% do, and
# that covariate shift is the leading mechanical candidate for the zero-shot collapse (-311% japan h3
# from a dengue trunk) that a 1,428-parameter affine adapter provably cannot absorb.
#
# LOCF carries the last OBSERVED value forward instead. It is strictly causal (only past cells are
# read), touches inputs only -- never y, raw, M or any loss/eval mask -- and adds no channel, so the
# 4-channel core and the C1 gate are untouched. obs_mask (channel 3) still flags every filled cell,
# so the trunk can still tell a carried value from a real one.
#
# DEFAULT OFF. The hypothesis was tested and did NOT hold -- locf_probe.py, seed 42, 3000 trunk steps:
#   adapted  arm: LOCF better in  7/12 cells, mean -0.9%  (noise, against an 8-14% seed sd)
#   zeroshot arm: LOCF better in  1/12 cells, mean +4.9% WORSE
# The zero-shot arm is the one the -311% collapse was measured on and the only arm this was meant to
# fix. It got worse, driven by japan h3 (+29.7%) and h5 (+12.3%).
#
# Plausible mechanism, which is why this is a finding rather than a failed patch: zero-fill puts an
# unobserved dengue cell at that node's training mean, whereas LOCF puts it at the last observed value
# and HOLDS it. On a series that is 78% gaps, that converts mean-reverting gaps into long flat runs at
# a carried level. The influenza panels are dense and oscillating, so LOCF plausibly moved the dengue
# input distribution FURTHER from flu rather than closer. If that is right, the transfer barrier is
# dengue's gappiness itself, and neither fill repairs it.
#
# Kept as a switchable ablation arm rather than deleted: LOCF_INPUT=1 turns it back on, and a
# tested-and-rejected hypothesis with numbers is worth more in the paper than a silent deletion.
# Single seed on a 3.3%-length trunk, so this is a go/no-go, not a measurement -- see Doubt.md §6.
LOCF_INPUT = os.environ.get("LOCF_INPUT", "0").lower() not in ("0", "false", "no")


def _locf(x, m):
    """Last-observation-carry-forward along axis 1 for [N,T] `x`, given observation mask [N,T] `m`.

    Standard cummax-of-indices trick: each cell reads the most recent observed index at or before it.
    Cells before a node's FIRST observation fall back to index 0, which is itself unobserved and
    therefore already 0.0 -- the only causal choice, since there is no earlier value to carry."""
    idx = np.where(m.astype(bool), np.arange(x.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return np.take_along_axis(x, idx, axis=1)


@dataclass
class Bundle:
    name: str
    X: np.ndarray             # [N, T, F]
    y: np.ndarray              # [N, T] == X[:, :, 0]
    raw: np.ndarray            # [N, T] count-space target
    M: np.ndarray               # [N, T] observation mask
    A_geo: np.ndarray            # [N, N]
    C: np.ndarray                 # [N, 3] static covariates -- never reaches the shared encoder
    meta: dict
    scaler: dict                    # {"mean": [N], "std": [N]} -- the released headline scaler
    _masks: dict                     # phase -> [N, T] uint8, normalised regardless of shipped form
    _groups: dict                     # node_id -> group label, synthesised for influenza

    def masks(self) -> dict:
        return self._masks

    def group_of(self) -> dict:
        return self._groups

    def transfer_view(self, locf: bool | None = None) -> np.ndarray:
        """X sliced to the core transfer-safe channels -- exactly 4, always.

        With `locf` (default: the LOCF_INPUT module switch) the incidence channel is
        carry-forward-filled on unobserved cells instead of zero-filled; see LOCF_INPUT above.
        Only channel 0 is touched -- sin_doy/cos_doy are complete by construction and obs_mask must
        keep flagging the filled cells."""
        view = self.X[:, :, self.meta["core_feature_idx"]]
        if LOCF_INPUT if locf is None else locf:
            view = view.copy()
            view[:, :, 0] = _locf(view[:, :, 0], self.M)
        return view

    def refit(self, fit_mask: np.ndarray):
        """Refit the scaler on `fit_mask`; returns (X, y, scaler) as ONE object (guide §0.3).

        Rebuilds X[:, :, 0], not just y -- a refit that only touches y leaves the model's
        *inputs* on the headline scaler, leaking the eval distribution in even though the
        leakage suite (which only inspects the released bundles) stays green.
        """
        groups = None
        if self.meta.get("node_country"):
            groups = [self.meta["node_country"][nid] for nid in self.meta["node_ids"]]
        # Ebola ships a disease-pooled scaler (scaler_scope='per_disease_support', §0.9): its 27
        # support cells are too few for a stable per-node scale. Hardcoding per-node here would
        # silently rescale Ebola wrong in a Week-5 few-shot refit. Derive it from the bundle.
        per_disease = self.meta.get("scaler_scope") == "per_disease_support"
        scaler = fit_scalers_masked(self.raw, fit_mask, per_disease=per_disease, groups=groups)
        inc_norm = np.where(self.M == 1, apply_scaler(self.raw, scaler), 0.0).astype(np.float32)
        X = self.X.copy()
        X[:, :, 0] = inc_norm
        return X, inc_norm.copy(), scaler

    def origins(self, w: int = W, H: int = MAX_H, phase: str | None = None) -> list[int]:
        """Valid origins t (0-indexed into T): t >= w-1 and t+H <= T-1.

        An example is an origin, not a node -- one origin consumes the whole graph. If
        `phase` is given, an origin is kept only when at least one target cell (i, t+h),
        h in HORIZONS, is both observed and in that phase -- the phase is decided by where
        the TARGET lands, never by the input window (Task 11.3).
        """
        T = self.X.shape[1]
        cand = [t for t in range(w - 1, T) if t + H <= T - 1]
        if phase is None:
            return cand
        pm = self._masks[phase]
        return [t for t in cand
                if any(np.any((self.M[:, t + h] == 1) & (pm[:, t + h] == 1)) for h in HORIZONS)]


def _normalise_masks(meta: dict, npz, M: np.ndarray) -> dict:
    """Split representation differs by bundle (§0.1): dengue/ebola ship [N,T] masks; the
    three influenza bundles ship train_end/val_end integers and no mask arrays at all.
    Always returns [N,T] uint8 masks, intersected with the observation mask M."""
    split = meta["split"]
    N, T = M.shape
    if "train_end" in split:                        # the three influenza bundles
        t = np.arange(T)
        train_end, val_end = split["train_end"], split["val_end"]
        raw_masks = {
            "train": np.tile(t < train_end, (N, 1)),
            "val": np.tile((t >= train_end) & (t < val_end), (N, 1)),
            "test": np.tile(t >= val_end, (N, 1)),
        }
    elif "support_mask" in split:                     # ebola
        raw_masks = {"support": npz["split_support_mask"], "query": npz["split_query_mask"]}
    else:                                               # dengue
        raw_masks = {"train": npz["split_train_mask"], "val": npz["split_val_mask"],
                     "test": npz["split_test_mask"]}
    return {phase: (arr.astype(np.uint8) & M) for phase, arr in raw_masks.items()}


def _groups(meta: dict, name: str) -> dict:
    if meta.get("node_country"):
        return dict(meta["node_country"])
    return {nid: name for nid in meta["node_ids"]}     # synthesise one group for influenza


def load(name: str) -> Bundle:
    npz = np.load(DATA_DIR / f"{name}.npz", allow_pickle=True)
    meta = json.loads(str(npz["meta_json"]))
    M = npz["M"]
    return Bundle(
        name=name, X=npz["X"], y=npz["y"], raw=npz["raw"], M=M, A_geo=npz["A_geo"], C=npz["C"],
        meta=meta, scaler={"mean": npz["scaler_mean"], "std": npz["scaler_std"]},
        _masks=_normalise_masks(meta, npz, M), _groups=_groups(meta, name),
    )


def _demo():
    for name in BUNDLE_NAMES:
        b = load(name)
        N, T, F = b.X.shape

        tv = b.transfer_view()
        assert tv.shape == (N, T, 4), f"{name}: transfer_view should be 4 channels, got {tv.shape}"
        assert np.array_equal(b.y, b.X[:, :, 0]), f"{name}: y != X[:, :, 0]"

        masks = b.masks()
        union = sum(masks.values())
        assert union.max() <= 1, f"{name}: split masks overlap"
        assert np.array_equal(union, b.M), f"{name}: split masks don't partition the observed cells"

        groups = b.group_of()
        assert set(groups) == set(b.meta["node_ids"]), f"{name}: group_of() missing nodes"

        all_origins = b.origins()
        by_phase = {phase: len(b.origins(phase=phase)) for phase in masks}
        print(f"ok  {name:22s} X={b.X.shape} origins={len(all_origins):5d}  "
              f"by-phase={by_phase}")

    print(f"ok  all {len(BUNDLE_NAMES)} bundles: transfer_view==4ch, y==X[:,:,0], masks partition M, "
          "group_of covers every node")
    _refit_check()
    _locf_check()


def _locf_check():
    """LOCF fills only what it should, reads only backwards, and leaves every other array alone.

    The causality arm is the one that matters: a fill that peeked forward would be leakage dressed
    as imputation, and it would not show up in any shape or dtype check."""
    x = np.array([[0.0, 0.0, 5.0, 0.0, 0.0, 7.0, 0.0]])            # cols 2 and 5 observed
    m = np.array([[0, 0, 1, 0, 0, 1, 0]], dtype=np.uint8)
    got = _locf(x, m)
    assert np.array_equal(got, [[0.0, 0.0, 5.0, 5.0, 5.0, 7.0, 7.0]]), f"LOCF wrong: {got}"
    assert np.array_equal(got[0, :2], [0.0, 0.0]), "leading gap must stay 0 -- nothing to carry"
    # causality: perturbing the LATER observation must not move any earlier cell
    x2 = x.copy(); x2[0, 5] = 999.0
    assert np.array_equal(_locf(x2, m)[0, :5], got[0, :5]), "LOCF read a FUTURE value -- leakage"
    # a fully-observed series is untouched (so the influenza panels are bit-identical either way)
    full = np.arange(7.0)[None, :]
    assert np.array_equal(_locf(full, np.ones((1, 7), np.uint8)), full), "observed cells moved"

    b = load("dengue")
    off, on = b.transfer_view(locf=False), b.transfer_view(locf=True)
    obs = b.M.astype(bool)
    assert np.array_equal(off[:, :, 0][obs], on[:, :, 0][obs]), "LOCF changed an OBSERVED input cell"
    assert np.array_equal(off[:, :, 1:], on[:, :, 1:]), "LOCF touched a non-incidence channel"
    assert np.array_equal(b.y, b.X[:, :, 0]), "LOCF must not write back into the bundle"
    zero_off = float((off[:, :, 0] == 0).mean())
    zero_on = float((on[:, :, 0] == 0).mean())
    assert zero_on < zero_off, f"LOCF did not reduce the zero mass ({zero_on} vs {zero_off})"
    print(f"ok  LOCF: causal, observed cells untouched, channels 1-3 untouched; "
          f"dengue exact-zero input mass {zero_off:.1%} -> {zero_on:.1%}")


def _refit_check():
    """The §0.3 leakage trap (review #2): refit must rebuild X[:,:,0], not only y -- else the
    model's INPUTS stay on the headline scaler while its targets move, leaking the eval
    distribution back in invisibly. Refit on a window the headline didn't use, so the scaler
    must move, then prove the rebuilt inputs track y (and that a y-only refit would not)."""
    b = load("influenza_japan")
    obs = b.M.astype(bool)
    fit = b.masks()["train"].astype(bool) | b.masks()["val"].astype(bool)   # != headline (train only)
    X2, y2, _ = b.refit(fit)
    assert np.array_equal(y2, X2[:, :, 0]), "refit: X[:,:,0] not rebuilt to match y (leakage trap)"
    assert not np.allclose(X2[:, :, 0][obs], b.X[:, :, 0][obs]), \
        "refit on a different window left inputs unchanged -- scaler was not re-applied to X"
    # negative control: the ORIGINAL inputs no longer equal the refit y, so a y-only refit
    # (rebuilding y but not X) would leave X[:,:,0] stale. That inconsistency is the bug refit avoids.
    assert not np.allclose(b.X[:, :, 0][obs], y2[obs]), \
        "control void: headline X[:,:,0] already == refit y (pick a window that moves the scaler)"
    print("ok  refit rebuilds X[:,:,0] together with y (§0.3); a y-only refit would leave inputs stale")


if __name__ == "__main__":
    _demo()

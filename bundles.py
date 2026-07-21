"""bundles.py -- one interface over the five harmonised disease bundles.

data/processed/*.npz are frozen, gated inputs (data_audit.md, schema_spec.md). This module
absorbs the three divergences the Week-3 guide documents -- split representation (masks vs
train_end/val_end integers), channel count (Ebola ships 5, everyone else 4) and node_country
presence (absent for influenza) -- in one place, so no model, metric or experiment file
downstream ever branches on which disease it is holding.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from to_schema import apply_scaler, fit_scalers_masked

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
BUNDLE_NAMES = ["dengue", "influenza_japan", "influenza_us-regions", "influenza_us-states", "ebola"]
DEV_BUNDLE_NAMES = tuple(n for n in BUNDLE_NAMES if n != "ebola")

W = 20                          # lookback, frozen protocol
HORIZONS = (3, 5, 10, 15)       # frozen protocol, direct multi-horizon
MAX_H = max(HORIZONS)


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

    def transfer_view(self) -> np.ndarray:
        """X sliced to the core transfer-safe channels -- exactly 4, always."""
        return self.X[:, :, self.meta["core_feature_idx"]]

    def refit(self, fit_mask: np.ndarray):
        """Refit the scaler on `fit_mask`; returns (X, y, scaler) as ONE object (guide §0.3).

        Rebuilds X[:, :, 0], not just y -- a refit that only touches y leaves the model's
        *inputs* on the headline scaler, leaking the eval distribution in even though the
        leakage suite (which only inspects the released bundles) stays green.
        """
        groups = None
        if self.meta.get("node_country"):
            groups = [self.meta["node_country"][nid] for nid in self.meta["node_ids"]]
        scaler = fit_scalers_masked(self.raw, fit_mask, per_disease=False, groups=groups)
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

    print("ok  all five bundles: transfer_view==4ch, y==X[:,:,0], masks partition M, "
          "group_of covers every node")


if __name__ == "__main__":
    _demo()

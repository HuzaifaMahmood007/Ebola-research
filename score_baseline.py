"""score_baseline.py -- rescore baseline predictions through OUR score.py (Day 15, Task 15.3).

Baselines each print their own metrics in their own conventions; we NEVER copy those (one model
logged MAPE/1e7, another RMSE in thousands). Instead every patched baseline dumps its COUNT-SPACE
test predictions to baselines/_preds/<model>__<dataset>__h<h>__seed<s>.npz, and this script:

  1. reassembles them into an [N, T] count array (scatter pred[row, node] -> P[node, target_time]),
  2. scores the OBSERVED TEST cells (the export's test mask, already intersected with M) through
     score.score_bundle -- identical aggregation to our encoder (country-macro dengue, node-level
     influenza), same count space,
  3. writes results/baselines/<...>.json in the same record-per-(model x dataset x horizon x seed x
     metric) schema the encoder uses, so Week-6's tables pool both without special-casing.

Runs in a numpy env (score.py is numpy-only); it does NOT import torch or any baseline code.
Dengue baseline rows are scored on the 1/3 stratified node subset (export meta records it); our
encoder stays on full dengue -- a disclosed non-like-with-like caveat on that one dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import score

BASE = Path(__file__).resolve().parent
PRED_DIR = BASE / "baselines" / "_preds"
EXPORT_DIR = BASE / "baselines" / "_exported"
OUT_DIR = BASE / "results" / "baselines"

# baseline rows carry the same record keys as encoder rows; these fields don't apply to baselines.
_NULL_META = dict(training_regime="single", sampler=None, gate_mode=None, topo_aug=None)


def _load_export(dataset: str):
    d = EXPORT_DIR / dataset
    meta = json.loads((d / "meta.json").read_text())
    raw_TN = np.loadtxt(d / "matrix.txt", delimiter=",")          # [T, N] counts, as fed to the baseline
    if raw_TN.ndim == 1:
        raw_TN = raw_TN.reshape(meta["n_steps"], meta["n_nodes"])
    truth = raw_TN.T.astype(np.float64)                            # [N, T] count-space truth
    z = np.load(d / "masks.npz")
    return meta, truth, {k: z[k] for k in z.files}


def score_pred_file(path: Path) -> list[dict]:
    z = np.load(path, allow_pickle=True)
    model, dataset = str(z["model"]), str(z["dataset"])
    horizon, seed = int(z["horizon"]), int(z["seed"])
    preds, tt = z["preds"], z["target_times"].astype(int)         # preds [n, N], tt [n]

    meta, truth, masks = _load_export(dataset)
    N, T = meta["n_nodes"], meta["n_steps"]
    assert preds.shape[1] == N, f"{path.name}: preds N={preds.shape[1]} != export N={N}"

    P = np.zeros((N, T), dtype=np.float64)
    for row, t in enumerate(tt):
        P[:, t] = preds[row]                                       # scatter into the target column

    # score the observed test cells at THIS horizon: (i, t) is scored iff it's an observed test cell
    # AND some prediction landed in column t (i.e. t was a baseline target time). Intersecting with
    # tt guards against scoring test cells the baseline never emitted a prediction for.
    test = masks["test"].astype(bool)                              # already & obs at export time
    covered = np.zeros(T, dtype=bool)
    covered[tt] = True
    mask = (test & covered[None, :]).astype(np.uint8)

    node_ids = meta["node_ids"]
    ncmap = dict(zip(node_ids, meta["node_country"]))
    agg, _ = score.score_bundle(P, truth, mask, node_ids, ncmap)

    sub = meta.get("subsample")
    records = []
    for metric, a in agg.items():
        records.append(dict(
            model=model, dataset=dataset, horizon=horizon, seed=seed, metric=metric,
            country_macro=a["country_macro"], node_mean=a["node_mean"],
            n_countries=a["n_countries"], n_nodes=a["n_nodes"],
            node_subset=(None if sub is None else f"{sub['kept_n']}/{sub['orig_n']}"),
            **_NULL_META))
    return records


def score_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PRED_DIR.glob("*.npz")) if PRED_DIR.exists() else []
    if not files:
        print(f"no prediction files in {PRED_DIR} yet -- run a baseline first")
        return
    for f in files:
        recs = score_pred_file(f)
        out = OUT_DIR / (f.stem + ".json")
        out.write_text(json.dumps(recs, indent=2))
        r = recs[0]
        print(f"ok  {f.stem:44s} rmse(macro)={_g(recs,'rmse'):.3f}  "
              f"pcc(macro)={_g(recs,'pcc'):.3f}  nodes={r['n_nodes']}  -> {out.relative_to(BASE)}")


def _g(recs, metric):
    for r in recs:
        if r["metric"] == metric:
            v = r["country_macro"]
            return float(v) if v is not None else float("nan")
    return float("nan")


if __name__ == "__main__":
    score_all()

"""export_mtgnn.py -- build MTGNN's multi_step input format from our exports (Day 15).

MTGNN's `train_multi_step` reads dataset_dir/{train,val,test}.npz, each holding
  x : [B, in_len, N, 1]   input windows (raw counts; MTGNN fits its own StandardScaler on x_train)
  y : [B, out_len, N, 1]  future targets
plus a predefined adjacency pickle (ids, id_to_ind, A). One rollout emits all `out_len` steps, so
one run per (dataset, seed) yields every horizon -- we pull {3,5,10,15} out of the rollout.

Our per-cell split is carried by NaN-masking y: a target cell (node, t+1+s) that is NOT in the phase
being written, or not observed, becomes NaN. MTGNN's masked_mae(..., null_val=nan) then excludes
exactly those cells from the loss while keeping true zero weeks (the reason we flip trainer's null_val
from 0.0 to nan). x is left dense (inputs may legitimately contain later observations -- guide 11.3).
Dengue uses the same 1/3 stratified subset as every other baseline (read from _exported/dengue).

adj pickle: we store A_geo + I, because train_multi_step does `load_adj(...) - eye`, so it recovers
A_geo. Run MTGNN with --buildA_true False so this predefined graph is actually used.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
EXPORT = BASE / "baselines" / "_exported"
MTGNN_DIR = BASE / "baselines" / "MTGNN"
OUT = MTGNN_DIR / "data_ours"
PREDS = BASE / "baselines" / "_preds"

IN_LEN = 20        # our lookback W  (-> --seq_in_len)
OUT_LEN = 15       # covers all horizons {3,5,10,15}; pull steps h-1 at scoring  (-> --seq_out_len)
DATASETS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "dengue"]  # dengue last (slow)
PHASE_FILE = {"train": "train", "val": "val", "test": "test"}

# training run matrix
SEEDS = [42, 52, 62, 72, 82]
HORIZONS = [3, 5, 10, 15]
# --subgraph_size must be < num_nodes; explicit where the default 20 would exceed the node count.
SUBGRAPH = {"dengue": 20, "influenza_japan": 20, "influenza_us-regions": 8, "influenza_us-states": 20}


def _load(dataset: str):
    d = EXPORT / dataset
    meta = json.loads((d / "meta.json").read_text())
    mat = np.loadtxt(d / "matrix.txt", delimiter=",")            # [T, N] counts
    if mat.ndim == 1:
        mat = mat.reshape(meta["n_steps"], meta["n_nodes"])
    z = np.load(d / "masks.npz")
    masks = {k: z[k].astype(bool) for k in z.files}              # [N, T]
    A = np.loadtxt(d / "adj.txt", delimiter=",")
    if A.ndim == 1:
        A = A.reshape(meta["n_nodes"], meta["n_nodes"])
    return meta, mat.astype(np.float32), masks, A.astype(np.float32)


def build(dataset: str):
    meta, mat, masks, A = _load(dataset)
    T, N = mat.shape
    obs = masks["obs"]                                           # [N, T]
    origins = list(range(IN_LEN - 1, T - OUT_LEN))              # need in_len history + out_len future
    # x is phase-independent; y gets NaN'd per phase.
    X = np.stack([mat[t - IN_LEN + 1: t + 1] for t in origins]).astype(np.float32)   # [B, in_len, N]
    X = X[..., None]                                             # [B, in_len, N, 1]
    Yc = np.stack([mat[t + 1: t + 1 + OUT_LEN] for t in origins]).astype(np.float32)  # [B, out_len, N]

    d = OUT / dataset
    d.mkdir(parents=True, exist_ok=True)
    kept = {}
    for phase in ("train", "val", "test"):
        ph = masks[phase]                                       # [N, T]
        y = Yc.copy()
        for s in range(OUT_LEN):
            tt = np.array(origins) + 1 + s                      # target time per origin at step s
            valid = (ph[:, tt] & obs[:, tt]).T                  # [B, N] -- observed & in this phase
            step = y[:, s, :]
            step[~valid] = np.nan
            y[:, s, :] = step
        has_cell = ~np.all(np.isnan(y.reshape(len(origins), -1)), axis=1)   # keep origins with >=1 cell
        xk, yk = X[has_cell], y[has_cell, ..., None]            # y -> [b, out_len, N, 1]
        np.savez_compressed(d / f"{PHASE_FILE[phase]}.npz", x=xk, y=yk)
        kept[phase] = np.array(origins)[has_cell]
    np.save(d / "test_origins.npy", kept["test"])               # for count-space pred target-times

    with open(d / "adj.pkl", "wb") as f:                        # load_adj does (-I); store A+I -> recovers A
        pickle.dump((meta["node_ids"], {n: i for i, n in enumerate(meta["node_ids"])},
                     (A + np.eye(N, dtype=np.float32))), f)

    print(f"ok  {dataset:22s} N={N} origins={len(origins)}  "
          f"train/val/test={[int(kept[p].size) for p in ('train','val','test')]}  -> {d.relative_to(BASE)}")
    return d


def build_all():
    for ds in DATASETS:
        build(ds)


def _conda_exe():
    """Resolve a directly-launchable conda.exe so this runs from any Python (subprocess can't find bare
    `conda`, nor run conda.bat without a shell). Prefers Scripts/conda.exe."""
    cands = [os.environ.get("CONDA_EXE")]
    w = shutil.which("conda")
    if w and w.lower().endswith(".exe"):
        cands.append(w)
    for root in (Path.home() / "miniconda3", Path.home() / "anaconda3",
                 Path(os.environ.get("CONDA_PREFIX", "") or ".")):
        cands.append(str(root / "Scripts" / "conda.exe"))
    for c in cands:
        if c and Path(c).exists():
            return c
    return shutil.which("conda") or "conda"


CONDA = _conda_exe()


def _num_nodes(dataset: str) -> int:
    return json.loads((EXPORT / dataset / "meta.json").read_text())["n_nodes"]


def run_training(seeds=SEEDS):
    """Loop datasets x seeds, one MTGNN run per (dataset, seed) -- the rollout emits every horizon.
    Resumable: skips a (dataset, seed) whose four horizon prediction files already exist. num_nodes
    comes from each dataset's meta; subgraph_size from SUBGRAPH (must be < num_nodes)."""
    print(f"using conda: {CONDA}")
    MTGNN_DIR.joinpath("save_ours").mkdir(exist_ok=True)
    for ds in DATASETS:
        if not (OUT / ds / "train.npz").exists():
            print(f"!! {ds} not built -- run `python export_mtgnn.py build` first", file=sys.stderr)
            continue
        nn = _num_nodes(ds)
        sg = SUBGRAPH.get(ds, min(20, nn - 1))
        for s in seeds:
            if all((PREDS / f"MTGNN__{ds}__h{h}__seed{s}.npz").exists() for h in HORIZONS):
                print(f"skip MTGNN {ds} seed{s}"); continue
            cmd = [CONDA, "run", "--no-capture-output", "-n", "mtgnn", "python", "train_multi_step.py",
                   "--data", f"data_ours/{ds}", "--adj_data", f"data_ours/{ds}/adj.pkl",
                   "--num_nodes", str(nn), "--in_dim", "1",
                   "--seq_in_len", str(IN_LEN), "--seq_out_len", str(OUT_LEN),
                   "--buildA_true", "False", "--gcn_true", "True", "--subgraph_size", str(sg),
                   "--device", "cuda:0", "--seed", str(s), "--save", "./save_ours/", "--batch_size", "16"]
            print(">>>", " ".join(cmd[1:]), flush=True)
            if CONDA.lower().endswith(".bat"):
                rc = subprocess.run(subprocess.list2cmdline(cmd), cwd=str(MTGNN_DIR), shell=True).returncode
            else:
                rc = subprocess.run(cmd, cwd=str(MTGNN_DIR)).returncode
            if rc != 0:
                print(f"!!! FAILED (continuing): MTGNN {ds} seed{s}", file=sys.stderr, flush=True)
    print("done. score with:  conda run -n ebola-train python score_baseline.py")


def _check():
    build_all()
    # round-trip us-regions: shapes correct, y has NaNs (masked cells), x has none, test origins align
    d = OUT / "influenza_us-regions"
    tr = np.load(d / "train.npz")
    assert tr["x"].shape[1:] == (IN_LEN, tr["x"].shape[2], 1), tr["x"].shape
    assert tr["y"].shape[1:] == (OUT_LEN, tr["y"].shape[2], 1), tr["y"].shape
    assert not np.isnan(tr["x"]).any(), "x must be dense (no NaN)"
    assert np.isnan(tr["y"]).any(), "y should carry NaN-masked cells"
    te = np.load(d / "test.npz")
    origins = np.load(d / "test_origins.npy")
    assert te["x"].shape[0] == origins.shape[0], (te["x"].shape, origins.shape)
    print(f"ok  MTGNN export round-trips (x dense, y NaN-masked, test origins aligned)")


if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "run"
    if arg == "build":                          # (re)build the npz/adj/test_origins, then self-check
        _check()
    elif arg == "run":                          # train all MTGNN runs (build first if data is missing)
        if not all((OUT / ds / "train.npz").exists() for ds in DATASETS):
            print("data_ours missing -> building first\n"); build_all()
        run_training()
    else:
        sys.exit("usage: python export_mtgnn.py [build|run]   (default: run)")

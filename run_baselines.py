"""run_baselines.py -- run EpiGNN, Cola-GNN, HeatGNN and MTGNN over all horizons and all 5 seeds.

Each baseline runs in its own conda env. Resumable: a run whose prediction file(s) already exist in
baselines/_preds is skipped, so re-running after an interruption picks up where it stopped -- and any
already-complete FULL run (e.g. EpiGNN dengue seed 42) is not repeated. Fast influenza cells run
before the slow dengue ones, so most of the matrix lands early.

Cola-GNN and HeatGNN are influenza-only (CPU envs); EpiGNN and MTGNN also do the dengue 1/3 subset.

CAUTION: `done()` trusts that any existing prediction file is a full run. If you ever left a smoke
(short-epoch) file in baselines/_preds, delete it first or it will be skipped and its garbage kept.

Run from a shell where `conda` is on PATH (PowerShell):
    python run_baselines.py                 # all three models
    python run_baselines.py --dry-run        # print the exact plan, run nothing
    python run_baselines.py mtgnn            # just one (or several) by name

Then score:  conda run -n ebola-train python score_baseline.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
BL = BASE / "baselines"
EXPORT = BL / "_exported"
PREDS = BL / "_preds"

SEEDS = [42, 52, 62, 72, 82]                # all five seeds
HORIZONS = [3, 5, 10, 15]
INFLUENZA = ["influenza_japan", "influenza_us-regions", "influenza_us-states"]
MTGNN_CFG = {                               # dataset -> (num_nodes, subgraph_size < num_nodes); dengue last
    "influenza_japan": (47, 20), "influenza_us-regions": (10, 8),
    "influenza_us-states": (49, 20), "dengue": (2392, 20),
}

DRY = False                                 # set by --dry-run


def _conda_exe():
    """Resolve a directly-launchable conda executable, so the script works even when launched with a
    non-conda Python (subprocess can't find bare `conda`, and can't run conda.bat without a shell).
    Prefers Scripts/conda.exe."""
    cands = [os.environ.get("CONDA_EXE")]          # conda shells set this to Scripts/conda.exe
    w = shutil.which("conda")
    if w and w.lower().endswith(".exe"):
        cands.append(w)
    for root in (Path.home() / "miniconda3", Path.home() / "anaconda3",
                 Path(os.environ.get("CONDA_PREFIX", "") or ".")):
        cands.append(str(root / "Scripts" / "conda.exe"))
    for c in cands:
        if c and Path(c).exists():
            return c
    return shutil.which("conda") or "conda"        # last resort (may be conda.bat -> run() uses a shell)


CONDA = _conda_exe()


def stage(dataset, data_dir, with_masks=False):
    """Copy our export into a repo's data/ dir (idempotent)."""
    if DRY:
        print(f"    (would stage {dataset} -> {data_dir})"); return
    src = EXPORT / dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "matrix.txt", data_dir / f"{dataset}.txt")
    shutil.copyfile(src / "adj.txt", data_dir / f"{dataset}-adj.txt")
    if with_masks:                          # EpiGNN dengue reads these to drive the per-cell split
        shutil.copyfile(src / "masks.npz", data_dir / f"{dataset}_masks.npz")


def done(fname):
    return (PREDS / fname).exists()


def run(cmd, cwd):
    print(">>>", " ".join(cmd), flush=True)
    if DRY:
        return True
    argv = [CONDA] + cmd[1:]                        # cmd[0] is the literal "conda" -> resolved exe
    if CONDA.lower().endswith(".bat"):              # a .bat must go through a shell
        rc = subprocess.run(subprocess.list2cmdline(argv), cwd=str(cwd), shell=True).returncode
    else:
        rc = subprocess.run(argv, cwd=str(cwd)).returncode
    if rc != 0:
        print("!!! FAILED (continuing):", " ".join(cmd), file=sys.stderr, flush=True)
    return rc == 0


def run_epignn():
    cwd = BL / "EpiGNN"
    for ds in INFLUENZA + ["dengue"]:       # fast influenza first, slow dengue last
        stage(ds, cwd / "data", with_masks=(ds == "dengue"))
        batch = "16" if ds == "dengue" else "128"
        for h in HORIZONS:
            for s in SEEDS:
                if done(f"EpiGNN__{ds}__h{h}__seed{s}.npz"):
                    print(f"skip EpiGNN {ds} h{h} seed{s}"); continue
                run(["conda", "run", "--no-capture-output", "-n", "epignn", "python", "src/train.py",
                     "--dataset", ds, "--sim_mat", f"{ds}-adj", "--window", "20", "--horizon", str(h),
                     "--seed", str(s), "--train", ".5", "--val", ".2", "--test", ".3", "--batch", batch], cwd)


def run_colagnn():
    cwd = BL / "colagnn" / "src"
    for ds in INFLUENZA:                    # Cola-GNN is influenza-only (CPU env)
        stage(ds, BL / "colagnn" / "data")
        for h in HORIZONS:
            for s in SEEDS:
                if done(f"ColaGNN__{ds}__h{h}__seed{s}.npz"):
                    print(f"skip Cola-GNN {ds} h{h} seed{s}"); continue
                run(["conda", "run", "--no-capture-output", "-n", "colagnn", "python", "train.py",
                     "--model", "cola_gnn", "--dataset", ds, "--sim_mat", f"{ds}-adj", "--window", "20",
                     "--horizon", str(h), "--seed", str(s), "--train", ".5", "--val", ".2", "--test", ".3"], cwd)


def run_heatgnn():
    cwd = BL / "HeatGNN-14DB" / "src"
    for ds in INFLUENZA:                    # HeatGNN is influenza-only (CPU env)
        stage(ds, BL / "HeatGNN-14DB" / "data")
        for h in HORIZONS:
            for s in SEEDS:
                if done(f"HeatGNN__{ds}__h{h}__seed{s}.npz"):
                    print(f"skip HeatGNN {ds} h{h} seed{s}"); continue
                run(["conda", "run", "--no-capture-output", "-n", "heatgnn", "python", "train.py",
                     "--model", "HeatGNN", "--dataset", ds, "--sim_mat", f"{ds}-adj", "--window", "20",
                     "--horizon", str(h), "--seed", str(s), "--train", ".5", "--val", ".2", "--test", ".3"], cwd)


def run_mtgnn():
    cwd = BL / "MTGNN"
    if not DRY:
        (cwd / "save_ours").mkdir(exist_ok=True)
    for ds, (nn, sg) in MTGNN_CFG.items():  # influenza first, dengue last (dict order)
        if not (cwd / "data_ours" / ds / "train.npz").exists():
            print(f"!! MTGNN data missing for {ds} -- run `python export_mtgnn.py` first",
                  file=sys.stderr); continue
        for s in SEEDS:
            # one run emits every horizon; skip only when all four files are present
            if all(done(f"MTGNN__{ds}__h{h}__seed{s}.npz") for h in HORIZONS):
                print(f"skip MTGNN {ds} seed{s}"); continue
            run(["conda", "run", "--no-capture-output", "-n", "mtgnn", "python", "train_multi_step.py",
                 "--data", f"data_ours/{ds}", "--adj_data", f"data_ours/{ds}/adj.pkl", "--num_nodes", str(nn),
                 "--in_dim", "1", "--seq_in_len", "20", "--seq_out_len", "15", "--buildA_true", "False",
                 "--gcn_true", "True", "--subgraph_size", str(sg), "--device", "cuda:0", "--seed", str(s),
                 "--save", "./save_ours/", "--batch_size", "16"], cwd)


RUNNERS = {"epignn": run_epignn, "colagnn": run_colagnn, "heatgnn": run_heatgnn, "mtgnn": run_mtgnn}

if __name__ == "__main__":
    argv = [a.lower() for a in sys.argv[1:]]
    DRY = "--dry-run" in argv
    which = [a for a in argv if not a.startswith("-")] or list(RUNNERS)
    unknown = [w for w in which if w not in RUNNERS]
    if unknown:
        sys.exit(f"unknown model(s) {unknown}; choose from {list(RUNNERS)}")
    print(f"using conda: {CONDA}")
    for name in which:
        print(f"\n===== {name} (seeds {SEEDS}, horizons {HORIZONS}){' [DRY-RUN]' if DRY else ''} =====")
        RUNNERS[name]()
    print("\ndone." if not DRY else "\ndry-run done.",
          " score with:  conda run -n ebola-train python score_baseline.py")

"""export_baseline.py -- our five bundles -> the ColaGNN-lineage on-disk format (Day 15, Task 15.1).

The baselines (EpiGNN / Cola-GNN / HeatGNN one lineage; MTGNN separately) consume a comma-delimited
[T, N] matrix + a dense N x N comma adjacency. Ours are [N, T] and dense-ndarray already, so this is
transpose + subset + dump. Ebola is NOT exported (Week 5 only, guide 0.5).

Decisions this script encodes (all client-confirmed Day 15 -- see memory/day15-baseline-decisions):
  * RAW COUNTS out. Each baseline fits its OWN native scaler; we hand it counts and rescore in count
    space. No scaler travels in the sidecar.
  * DENGUE subsampled to ~1/3 nodes, STRATIFIED BY COUNTRY, deterministic (fixed seed). Fixes the
    7,165-node memory wall on repos built for <=49 nodes; keeps every country so country-macro stays
    defined. Influenza exported whole.
  * PER-CELL SPLIT carried as [N,T] masks. Influenza's split is already a clean global 50/20/30, but
    dengue's is per-country -- a global time-cut would leak our test cells into baseline training
    (guide Task 15.1). The patched loaders read these masks instead of re-cutting by ratio.
  * OBSERVATION mask carried too: dengue is 21.75% observed and the baselines assume dense input.
    Scoring (score_baseline.py) uses it to score OBSERVED cells only -- never the 78% imputed fill.

Layout per dataset under baselines/_exported/<name>/:
  matrix.txt   [T, N] raw counts, comma, one timestep per row (baseline orientation)
  adj.txt      [N, N] dense adjacency, comma
  masks.npz    train/val/test/obs, each [N, T] uint8 (N = exported node count)
  meta.json    node_ids (export column order), node_country, window/horizons/seeds, subsample record
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import bundles

OUT_DIR = Path(__file__).resolve().parent / "baselines" / "_exported"
SUBSAMPLE_FRAC = 1.0 / 3.0
SUBSAMPLE_SEED = 20260715          # fixed: the node subset must be identical across every model + rerun
SUBSAMPLE_DATASETS = {"dengue"}    # only dengue hits the node-count wall; influenza exported whole


def _stratified_subsample(b: bundles.Bundle, frac: float, seed: int) -> np.ndarray:
    """Keep ceil(frac * n_c) nodes from EACH country c (>=1), deterministically. Returns sorted
    row indices into the bundle's node axis, so export order still follows the original ordering."""
    node_ids = b.meta["node_ids"]
    country = b.meta["node_country"]                       # dengue has it (checked at call site)
    by_country: dict[str, list[int]] = {}
    for i, nid in enumerate(node_ids):
        by_country.setdefault(country[nid], []).append(i)
    rng = np.random.default_rng(seed)
    kept: list[int] = []
    for c in sorted(by_country):                            # sorted -> order independent of dict insertion
        idx = np.array(by_country[c])
        k = max(1, math.ceil(len(idx) * frac))
        chosen = rng.choice(idx, size=min(k, len(idx)), replace=False)
        kept.extend(int(i) for i in chosen)
    return np.array(sorted(kept), dtype=int)


def _kept_indices(b: bundles.Bundle) -> tuple[np.ndarray, dict | None]:
    if b.name in SUBSAMPLE_DATASETS:
        if not b.meta.get("node_country"):
            raise ValueError(f"{b.name}: stratified subsample needs node_country")
        kept = _stratified_subsample(b, SUBSAMPLE_FRAC, SUBSAMPLE_SEED)
        rec = dict(frac=SUBSAMPLE_FRAC, seed=SUBSAMPLE_SEED,
                   orig_n=len(b.meta["node_ids"]), kept_n=int(kept.size))
        return kept, rec
    return np.arange(len(b.meta["node_ids"]), dtype=int), None


def export(name: str) -> Path:
    b = bundles.load(name)
    kept, subrec = _kept_indices(b)
    node_ids = [b.meta["node_ids"][i] for i in kept]
    groups = b.group_of()                                   # real country for dengue; dataset name for influenza
    node_country = [groups[nid] for nid in node_ids]

    raw = b.raw[kept]                                        # [N, T] counts
    A = np.asarray(b.A_geo)[np.ix_(kept, kept)]             # [N, N]
    masks = b.masks()                                        # phase -> [N, T] uint8, already intersected with M
    obs = b.M[kept].astype(np.uint8)

    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    # [T, N] for the baselines: time down the rows. %.6g keeps counts exact (they are small integers as floats).
    np.savetxt(d / "matrix.txt", raw.T, delimiter=",", fmt="%.6g")
    np.savetxt(d / "adj.txt", A, delimiter=",", fmt="%.6g")
    mask_out = {phase: masks[phase][kept].astype(np.uint8) for phase in masks}
    mask_out["obs"] = obs
    np.savez_compressed(d / "masks.npz", **mask_out)

    meta = dict(
        dataset=name, n_nodes=int(raw.shape[0]), n_steps=int(raw.shape[1]),
        window=bundles.W, horizons=list(bundles.HORIZONS), seeds=[42, 52, 62, 72, 82],
        phases=list(masks), node_ids=node_ids, node_country=node_country,
        has_true_node_country=bool(b.meta.get("node_country")), subsample=subrec,
    )
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    tag = f"subsample {subrec['kept_n']}/{subrec['orig_n']}" if subrec else "full"
    print(f"ok  {name:22s} -> {d.relative_to(OUT_DIR.parent)}  "
          f"[T,N]={raw.T.shape}  adj={A.shape}  {tag}")
    return d


def export_all() -> None:
    for name in bundles.DEV_BUNDLE_NAMES:                   # the 4 development datasets; ebola excluded by design
        export(name)


def _check():
    """Self-check: round-trip each export and prove the invariants a baseline run depends on."""
    export_all()
    for name in bundles.DEV_BUNDLE_NAMES:
        d = OUT_DIR / name
        meta = json.loads((d / "meta.json").read_text())
        N, T = meta["n_nodes"], meta["n_steps"]
        mat = np.loadtxt(d / "matrix.txt", delimiter=",")
        mat = mat.reshape(T, N) if mat.ndim == 1 else mat   # 1-col guard
        assert mat.shape == (T, N), f"{name}: matrix {mat.shape} != [T,N] {(T, N)}"
        adj = np.loadtxt(d / "adj.txt", delimiter=",")
        assert adj.shape == (N, N), f"{name}: adj {adj.shape} != [N,N]"
        m = np.load(d / "masks.npz")
        split = sum(m[p] for p in meta["phases"])
        assert split.max() <= 1, f"{name}: split masks overlap"
        assert np.array_equal(split, m["obs"]), f"{name}: split masks don't partition observed cells"
        assert len(meta["node_ids"]) == N == len(meta["node_country"])
        # matrix counts match the bundle's raw for the kept nodes (transpose round-trips)
        b = bundles.load(name)
        kept, _ = _kept_indices(b)
        assert np.allclose(mat.T, b.raw[kept], atol=1e-4), f"{name}: exported counts != bundle raw"

    # dengue subsample: ~1/3 of nodes, every country still present, deterministic across two calls
    dm = json.loads((OUT_DIR / "dengue" / "meta.json").read_text())
    b = bundles.load("dengue")
    orig_countries = {b.meta["node_country"][nid] for nid in b.meta["node_ids"]}
    assert set(dm["node_country"]) == orig_countries, "dengue: a country was dropped by the subsample"
    frac = dm["n_nodes"] / dm["subsample"]["orig_n"]
    assert 0.30 <= frac <= 0.40, f"dengue kept fraction {frac:.3f} not ~1/3"
    k1 = _stratified_subsample(b, SUBSAMPLE_FRAC, SUBSAMPLE_SEED)
    k2 = _stratified_subsample(b, SUBSAMPLE_FRAC, SUBSAMPLE_SEED)
    assert np.array_equal(k1, k2), "subsample not deterministic"
    print(f"ok  round-trip + partition hold; dengue kept {dm['n_nodes']}/{dm['subsample']['orig_n']} "
          f"({frac:.1%}) across all {len(orig_countries)} countries, deterministic")


if __name__ == "__main__":
    _check()

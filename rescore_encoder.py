"""rescore_encoder.py -- backfill nrmse onto encoder runs already on disk, without retraining.

Week-4 added `nrmse` (= rmse / mean(y), Review Doc para. 7) to score.METRICS. Baselines pick it up by
re-running score_baseline.py, because their COUNT-SPACE predictions are archived in
baselines/_preds/*.npz. The encoder's are NOT: train/loop.py writes per-node METRIC VALUES
(results/*/**__pernode.npz), never the predictions. Retraining every run to recover one metric costs
hours of GPU/CPU we do not have while the HeatGNN queue holds the box.

nrmse does not need the predictions. It needs `rmse` (stored, per node, per horizon) and `mean(y)`
over exactly the cells that node was scored on -- and that cell set is deterministic, defined at
loop.py:210-214 as

    mask_h[:, t + h] = phase_mask[:, t + h]   for t in origins

from the bundle's own test mask and test origins. So we rebuild the mask from data/processed/,
divide, and write nrmse back.

THE GUARD THAT MAKES THIS HONEST: we do not assume the rebuild is right, we prove it. For every
(node, horizon) the rebuilt cell count must equal the `n_cells` the run itself recorded. Any file
with a single mismatch is SKIPPED and reported -- never silently patched with a guessed mask. A
backfill you cannot verify is worse than a missing column.

Idempotent: re-running overwrites nrmse with the same values and does not duplicate JSON records.

    conda run -n ebola-train python rescore_encoder.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import bundles
import score
from results_paths import RESULTS

FAMILIES = ("single", "joint", "lodo")          # misc/ is smoke throwaways; naive/ has its own writer
_BUNDLE_CACHE: dict[str, object] = {}


def _dataset_of(stem: str) -> str:
    """`encoder_joint__sqrt-uniform__dengue__seed42__pernode` -> `dengue` (token before seed*)."""
    parts = stem.split("__")
    for i, p in enumerate(parts):
        if p.startswith("seed"):
            return parts[i - 1]
    raise ValueError(f"cannot read dataset from {stem!r}")


def _masks_for(dataset: str):
    """{h: [N,T] uint8} scored-cell masks + the bundle's count-space truth, exactly as loop.py built them."""
    if dataset not in _BUNDLE_CACHE:
        b = bundles.load(dataset)
        phase = b.masks()["test"].astype(np.uint8)
        origins = b.origins(phase="test")
        mh = {}
        for h in bundles.HORIZONS:
            m = np.zeros_like(phase)
            for t in origins:
                m[:, t + h] = phase[:, t + h]
            mh[h] = m
        _BUNDLE_CACHE[dataset] = (b.raw.astype(np.float64), mh, b.meta["node_ids"], b.group_of())
    return _BUNDLE_CACHE[dataset]


def _aggregate_one(values, countries):
    """score.aggregate's country-macro / node-mean for a single metric vector (NaNs dropped)."""
    per_country = {}
    for c in sorted(set(countries)):
        vals = [v for v, cc in zip(values, countries) if cc == c and not np.isnan(v)]
        if vals:
            per_country[c] = float(np.mean(vals))
    node_vals = [v for v in values if not np.isnan(v)]
    return dict(
        country_macro=float(np.mean(list(per_country.values()))) if per_country else float("nan"),
        node_mean=float(np.mean(node_vals)) if node_vals else float("nan"),
        n_countries=len(per_country), n_nodes=len(node_vals))


def backfill(path: Path, dry_run=False):
    """Returns (status, detail). status in {'ok', 'skip-mismatch', 'ok-already'}."""
    stem = path.name[: -len("__pernode.npz")]
    dataset = _dataset_of(stem)
    raw, mask_by_h, ids, ncmap = _masks_for(dataset)

    z = dict(np.load(path, allow_pickle=True))
    new_records = []
    for h in bundles.HORIZONS:
        idx = z[f"h{h}__node_idx"].astype(int)
        stored_n = z[f"h{h}__n_cells"].astype(int)
        rmse = z[f"h{h}__rmse"].astype(np.float64)
        countries = list(z[f"h{h}__country"])

        m = mask_by_h[h].astype(bool)[idx]                       # [S, T] rebuilt cell set
        rebuilt_n = m.sum(axis=1)
        bad = int(np.sum(rebuilt_n != stored_n))
        if bad:
            return "skip-mismatch", (f"h{h}: {bad}/{len(idx)} nodes disagree on n_cells "
                                     f"(rebuilt vs recorded) -- mask reconstruction is not exact")

        scale = np.array([raw[i][mi].mean() for i, mi in zip(idx, m)])
        with np.errstate(divide="ignore", invalid="ignore"):
            nrmse = np.where(scale > 0, rmse / scale, np.nan)
        z[f"h{h}__nrmse"] = nrmse.astype(np.float32)
        new_records.append((h, _aggregate_one(nrmse, countries)))

    if dry_run:
        return "ok", f"{dataset}: verified, {len(new_records)} horizons (dry run, nothing written)"

    np.savez_compressed(path, **z)

    jpath = path.with_name(stem + ".json")
    if jpath.exists():
        recs = json.loads(jpath.read_text())
        template = recs[0]
        keep = [r for r in recs if r["metric"] != "nrmse"]       # idempotent: drop any prior backfill
        for h, a in new_records:
            r = {k: template[k] for k in template}
            r.update(horizon=h, metric="nrmse", **a)
            keep.append(r)
        jpath.write_text(json.dumps(keep, indent=2))
    return "ok", f"{dataset}: nrmse written for {len(new_records)} horizons"


def main(dry_run=False):
    files = sorted(p for fam in FAMILIES for p in (RESULTS / fam).glob("*__pernode.npz"))
    if not files:
        print("no encoder __pernode.npz artifacts found")
        return 0
    counts = {"ok": 0, "skip-mismatch": 0}
    for p in files:
        status, detail = backfill(p, dry_run=dry_run)
        counts[status] = counts.get(status, 0) + 1
        if status != "ok":
            print(f"SKIP {p.parent.name}/{p.name}\n     {detail}")
    print(f"\n{counts['ok']}/{len(files)} artifacts backfilled with nrmse"
          f"{' (dry run)' if dry_run else ''}; {counts['skip-mismatch']} skipped on mask mismatch")
    return 1 if counts["skip-mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv))

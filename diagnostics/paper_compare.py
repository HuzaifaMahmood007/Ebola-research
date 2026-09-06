"""paper_compare.py -- the three-way picture (Work Order 1c): published | reproduced | our encoder.

Client condition, verbatim: "validate each reproduction against the source paper's published numbers
before using it as a comparator, and show me our reproduction next to their reported figure. If we
can't get close, that baseline doesn't go in the comparison table."

Three numbers per (model, dataset, horizon), all in ONE metric definition:

  1. PUBLISHED  -- transcribed from the paper's own table ("Published Model benchmarks/")
  2. REPRODUCED -- that baseline run on our pipeline, from baselines/_preds/*.npz
  3. OUR ENCODER -- results/single/encoder__<ds>__seed*, the thing we are actually claiming

WHY THIS IS NOT score_baseline.py's NUMBER. score.py's headline is per-node RMSE averaged over nodes
(then over countries). Every one of these papers defines RMSE as sqrt(mean over ALL scored cells) --
pooled, not averaged. By Jensen the two disagree, always, and mean-of-RMSEs <= RMSE-of-pool. A Delta%
between the two aggregations would measure the aggregation, not the reproduction. So every column
here is CELL-POOLED, and the country-macro headline is deliberately absent -- read it in
Updated_Scores.txt, never against these.

The encoder never archived its predictions (train/loop.py stores metrics, not preds), but it does
archive per-(origin, country) error sufficient statistics, and pooled RMSE = sqrt(sum(sse)/sum(n))
is exactly recoverable from them (loop.py:252-256). Guard: the cell count implied by those
sufficient stats must equal the cell count in the matching __pernode.npz, or the run is skipped.

Population match: score.py drops constant-truth nodes, and the encoder's sufficient stats are over
that same non-constant set, so the baseline pooled figure applies the identical filter. Without it
the two sides would be pooling different cells.

    conda run -n ebola-train python paper_compare.py [--md]
"""
from __future__ import annotations

import collections
import io
import pathlib
import sys

import numpy as np

from results_paths import RESULTS
from score_baseline import PRED_DIR, _load_export

# (model, our dataset name) -> {horizon: (published RMSE, published PCC)}.
# EpiGNN + Cola-GNN: same 50/20/30 chronological split and same source datasets as our pipeline.
# HeatGNN: paper is 60/20/20 and horizons {2,5,7,12}; only h5 overlaps our grid, and the split
# differs -- its Delta% carries that caveat and cannot be read as cleanly as the other two.
# MTGNN: its paper has NO epidemic dataset, so there is nothing to compare against here.
PUBLISHED = {
    ("EpiGNN", "influenza_japan"):       {3: (996, 0.904), 5: (1031, 0.908), 10: (1441, 0.739), 15: (1470, 0.773)},
    ("EpiGNN", "influenza_us-regions"):  {3: (589, 0.912), 5: (774, 0.842), 10: (984, 0.749), 15: (1061, 0.694)},
    ("EpiGNN", "influenza_us-states"):   {3: (160, 0.935), 5: (186, 0.907), 10: (220, 0.865), 15: (236, 0.861)},
    ("Cola-GNN", "influenza_japan"):     {3: (1051, 0.901), 5: (1117, 0.890), 10: (1372, 0.813), 15: (1475, 0.753)},
    ("Cola-GNN", "influenza_us-regions"): {3: (636, 0.909), 5: (855, 0.835), 10: (1134, 0.717), 15: (1203, 0.639)},
    ("Cola-GNN", "influenza_us-states"): {3: (167, 0.933), 5: (202, 0.897), 10: (241, 0.822), 15: (237, 0.856)},
    ("HeatGNN", "influenza_japan"):      {5: (1378, 0.884)},
    ("HeatGNN", "influenza_us-regions"): {5: (852, 0.866)},
    ("HeatGNN", "influenza_us-states"):  {5: (186, 0.921)},
}

_EXPORTS: dict[str, tuple] = {}


def _export(dataset):
    if dataset not in _EXPORTS:
        _EXPORTS[dataset] = _load_export(dataset)
    return _EXPORTS[dataset]


def pooled(path):
    """Cell-pooled RMSE and PCC over the scored test cells -- the papers' definition.

    Constant-truth nodes are excluded, matching score.per_node_scores and therefore matching the
    population behind the encoder's sufficient statistics.
    """
    z = np.load(path, allow_pickle=True)
    model, dataset = str(z["model"]), str(z["dataset"])
    h, seed = int(z["horizon"]), int(z["seed"])
    preds, tt = z["preds"], z["target_times"].astype(int)

    meta, truth, masks = _export(dataset)
    N, T = meta["n_nodes"], meta["n_steps"]
    P = np.zeros((N, T), dtype=np.float64)
    for row, t in enumerate(tt):
        P[:, t] = preds[row]

    covered = np.zeros(T, dtype=bool)
    covered[tt] = True
    m = masks["test"].astype(bool) & covered[None, :]
    for i in range(N):                                        # same constant-node rule as score.py
        mi = m[i]
        if mi.any() and truth[i][mi].std() < 1e-8:
            m[i] = False

    y, yh = truth[m], P[m]
    rmse = float(np.sqrt(np.mean((yh - y) ** 2)))
    pcc = float(np.corrcoef(yh, y)[0, 1]) if y.std() > 1e-8 and yh.std() > 1e-8 else float("nan")
    degenerate = len(np.unique(np.round(preds.astype(np.float64), 6))) == 1
    return model, dataset, h, seed, rmse, pcc, int(m.sum()), degenerate


_SUB_ROWS: dict[str, np.ndarray] = {}


def subsample_rows(dataset: str) -> np.ndarray | None:
    """Bundle row indices the baselines actually saw, or None if the whole panel was exported.

    Only dengue is subsampled (7,165 nodes will not fit repos built for <=49), and the encoder is
    scored on the whole bundle. Pooling the two sides over different node sets makes the
    'encoder vs reproduced' column measure the node set: the encoder's own dengue RMSE is about 50%
    higher on the subsample than on the full panel, so the uncorrected gap is roughly twice the real
    one. Reuses export_baseline's own selector so the table cannot drift from the export.
    """
    import export_baseline

    if dataset not in export_baseline.SUBSAMPLE_DATASETS:
        return None
    if dataset not in _SUB_ROWS:
        import bundles
        _SUB_ROWS[dataset] = export_baseline._kept_indices(bundles.load(dataset))[0]
    return _SUB_ROWS[dataset]


def encoder_pooled():
    """{(dataset, h): [rmse_per_seed]} -- cell-pooled encoder RMSE, on the baselines' node set.

    sqrt(sum(sse)/sum(n)) over every (origin, country) cell IS the cell-pooled RMSE, and the
    per-origin sufficient stats give it directly. They cannot be restricted by node, so a subsampled
    dataset is pooled from the per-node archive instead as sqrt(sum(n_i * rmse_i^2)/sum(n_i)), which
    is the same quantity node-side. The two routes agree to the printed precision on the full panel;
    the cell-count check below is what holds them together.
    """
    out = collections.defaultdict(list)
    for po in sorted((RESULTS / "single").glob("encoder__*__perorigin.npz")):
        stem = po.name[: -len("__perorigin.npz")]
        pn = po.with_name(stem + "__pernode.npz")
        if not pn.exists():
            print(f"  ! {stem}: no __pernode.npz to verify against, skipped", file=sys.stderr)
            continue
        dataset = stem.split("__")[1]
        keep_rows = subsample_rows(dataset)
        zo, zn = np.load(po, allow_pickle=True), np.load(pn, allow_pickle=True)
        for h in (3, 5, 10, 15):
            n = zo[f"h{h}__n"].astype(np.int64)
            sse = zo[f"h{h}__sse"].astype(np.float64)
            want = int(zn[f"h{h}__n_cells"].astype(np.int64).sum())
            got = int(n.sum())
            if got != want:
                print(f"  ! {stem} h{h}: per-origin cells {got} != per-node cells {want}, skipped",
                      file=sys.stderr)
                continue
            if keep_rows is None:
                out[(dataset, h)].append(float(np.sqrt(sse.sum() / max(got, 1))))
                continue
            idx = zn[f"h{h}__node_idx"].astype(np.int64)
            nc = zn[f"h{h}__n_cells"].astype(np.float64)
            r = zn[f"h{h}__rmse"].astype(np.float64)
            m = np.isfinite(r) & (nc > 0) & np.isin(idx, keep_rows)
            if not m.any():
                print(f"  ! {stem} h{h}: no scored node survives the subsample, skipped",
                      file=sys.stderr)
                continue
            out[(dataset, h)].append(float(np.sqrt((nc[m] * r[m] ** 2).sum() / nc[m].sum())))
    return out


def collect():
    acc = collections.defaultdict(list)
    for p in sorted(PRED_DIR.glob("*.npz")):
        model, ds, h, _seed, rmse, pcc, n, degen = pooled(p)
        acc[(model, ds, h)].append((rmse, pcc, n, degen))
    return acc


def _ms(a):
    a = np.asarray(a, dtype=np.float64)
    return a.mean(), (a.std(ddof=1) if a.size > 1 else 0.0)


def main(as_md=False):
    acc, enc = collect(), encoder_pooled()
    rows = []
    for (model, ds, h), vals in sorted(acc.items()):
        r_mean, r_sd = _ms([v[0] for v in vals])
        pub = PUBLISHED.get((model, ds), {}).get(h)
        degen = sum(1 for v in vals if v[3])
        e = enc.get((ds, h))
        e_mean, e_sd = _ms(e) if e else (float("nan"), float("nan"))
        rows.append(dict(
            model=model, dataset=ds, h=h, n_seeds=len(vals), degen=degen,
            pub=pub[0] if pub else None,
            repro=r_mean, repro_sd=r_sd,
            repro_pcc=np.nanmean([v[1] for v in vals]),
            enc=e_mean, enc_sd=e_sd, enc_seeds=len(e) if e else 0,
            d_pub=(100.0 * (r_mean - pub[0]) / pub[0]) if pub else None,
            d_enc=(100.0 * (e_mean - r_mean) / r_mean) if e and r_mean > 0 else None))

    def f(v, w=".1f", dash="--"):
        return dash if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, w)

    if as_md:
        print("| model | dataset | h | 1. published | 2. reproduced (ours) | Δ% vs published "
              "| 3. our encoder | encoder vs reproduced | seeds | flag |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for x in rows:
            flags = []
            if x["degen"]:
                flags.append(f"**{x['degen']}/{x['n_seeds']} constant**")
            if subsample_rows(x["dataset"]) is not None:
                flags.append("both sides on the 2,392-node subsample")
            flag = "; ".join(flags)
            print(f"| {x['model']} | {x['dataset']} | {x['h']} | {f(x['pub'], '.0f', '—')} | "
                  f"{f(x['repro'])} ± {f(x['repro_sd'])} | "
                  f"{f(x['d_pub'], '+.1f', '—')}{'%' if x['d_pub'] is not None else ''} | "
                  f"{f(x['enc'], '.1f', '—')} ± {f(x['enc_sd'], '.1f', '—')} | "
                  f"{f(x['d_enc'], '+.1f', '—')}{'%' if x['d_enc'] is not None else ''} | "
                  f"{x['n_seeds']}/{x['enc_seeds']} | {flag} |")
    else:
        print(f"{'model':10} {'dataset':22} {'h':>3} {'1.pub':>8} {'2.reproduced':>18} {'d%pub':>8} "
              f"{'3.encoder':>18} {'enc-v-rep':>10} {'sd':>5} {'flag':>14}")
        for x in rows:
            flag = f"{x['degen']}/{x['n_seeds']} CONST" if x["degen"] else ""
            print(f"{x['model']:10} {x['dataset']:22} {x['h']:>3} {f(x['pub'], '8.0f'):>8} "
                  f"{f(x['repro'], '10.1f')} +/-{f(x['repro_sd'], '5.1f')} {f(x['d_pub'], '+7.1f'):>8} "
                  f"{f(x['enc'], '10.1f')} +/-{f(x['enc_sd'], '5.1f')} "
                  f"{f(x['d_enc'], '+9.1f'):>10} {'':>5} {flag:>14}")
    return rows


if __name__ == "__main__":
    # -o writes UTF-8 itself: the table carries em dash and +- and the Windows console mangles both.
    out = None
    if "-o" in sys.argv:
        out = sys.argv[sys.argv.index("-o") + 1]
        buf = io.StringIO()
        real, sys.stdout = sys.stdout, buf
        try:
            main(as_md=True)
        finally:
            sys.stdout = real
        pathlib.Path(out).write_text(buf.getvalue(), encoding="utf-8")
        print(f"wrote {out}")
    else:
        main(as_md="--md" in sys.argv)

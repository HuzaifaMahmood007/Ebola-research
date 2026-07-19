"""Country-macro scoring for the multi-country bundles (dengue is the one that needs it).

Headline metric (data_audit.md 1.7 / Task 6.3 decision 6): score per node over its
observed eval cells, average within each country, then average the countries with EQUAL
weight. That is what stops Brazil's 77% node share from being the score. The equal-node
average is reported alongside as the secondary ("micro") figure.

Two rules the metric depends on, both settled here so a caller cannot get them wrong:

  * REAL COUNTS. Metrics are computed in count space, per manuscript 5.1. The model works
    in normalised space, so invert the per-node scaler on the predictions BEFORE calling
    this (invert_scaler in to_schema). Truth is the bundle's `raw`, which is already counts.

  * CONSTANT NODES. A node whose truth is constant over the cells being scored (Japan's 29
    always-zero dengue prefectures, chiefly) is trivially predictable -- emit its constant
    and you are exactly right -- and its Pearson correlation is undefined. Such nodes are
    excluded from scoring by default and kept in the graph as message-passing neighbours.
    Score them too (score_constant=True) only for the with-constants sensitivity row.

Horizon-resolved reporting is the caller's job: call this once per horizon h with that
horizon's predictions. The aggregation is identical each time.
"""
from __future__ import annotations

import collections
import numpy as np


def _rmse(yh, y):
    return float(np.sqrt(np.mean((yh - y) ** 2)))


def _mae(yh, y):
    return float(np.mean(np.abs(yh - y)))


def _pcc(yh, y):
    # Undefined when either side is flat; the caller has already dropped constant-truth
    # nodes, but a flat prediction can still happen, so guard both.
    if y.std() < 1e-8 or yh.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(yh, y)[0, 1])


METRICS = {"rmse": _rmse, "mae": _mae, "pcc": _pcc}


def per_node_scores(pred, truth, mask, node_country, score_constant=False):
    """Per-node metrics over observed eval cells. Arrays are [N, T] in COUNT space; mask==1
    marks a cell that is observed AND in the eval split (e.g. bundle split_test_mask).

    Returns {i: {country, n_cells, constant, rmse, mae, pcc}} for the nodes that were scored.
    """
    N, _ = truth.shape
    out = {}
    for i in range(N):
        m = mask[i].astype(bool)
        if not m.any():
            continue                                   # nothing to score for this node
        y, yh = truth[i][m], pred[i][m]
        constant = bool(y.std() < 1e-8)
        if constant and not score_constant:
            continue                                   # trivially predictable; kept in graph
        out[i] = dict(country=node_country[i], n_cells=int(m.sum()), constant=constant,
                      rmse=_rmse(yh, y), mae=_mae(yh, y), pcc=_pcc(yh, y))
    return out


def aggregate(node_scores):
    """country-macro (equal weight per country) and node-mean (equal weight per node).

    Returns {metric: {country_macro, node_mean, n_countries, n_nodes, per_country: {...}}}.
    NaN per-node values (an undefined PCC that slipped through) are dropped from the means.
    """
    by_country = collections.defaultdict(list)
    for s in node_scores.values():
        by_country[s["country"]].append(s)

    result = {}
    for metric in METRICS:
        per_country = {}
        for c, ss in sorted(by_country.items()):
            vals = [s[metric] for s in ss if not np.isnan(s[metric])]
            if vals:
                per_country[c] = float(np.mean(vals))
        node_vals = [s[metric] for s in node_scores.values() if not np.isnan(s[metric])]
        result[metric] = dict(
            country_macro=float(np.mean(list(per_country.values()))) if per_country else float("nan"),
            node_mean=float(np.mean(node_vals)) if node_vals else float("nan"),
            n_countries=len(per_country), n_nodes=len(node_vals),
            per_country=per_country)
    return result


def score_bundle(pred_counts, truth_counts, mask, node_ids, node_country_map,
                 score_constant=False):
    """Convenience wrapper: node_country_map is meta['node_country'] (id -> country)."""
    node_country = [node_country_map[n] for n in node_ids]
    ns = per_node_scores(pred_counts, truth_counts, mask, node_country, score_constant)
    return aggregate(ns), ns


# --------------------------------------------------------------------------- #
# Self-check: the aggregation must (1) be exact for a perfect predictor, (2) actually
# neutralise Brazil's dominance, (3) drop constant nodes. Runs on the released dengue bundle.
# --------------------------------------------------------------------------- #
def _demo():
    import json
    z = np.load("data/processed/dengue.npz", allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    raw = z["raw"].astype(np.float64)                  # truth, in counts
    test = z["split_test_mask"]
    ids, ncmap = meta["node_ids"], meta["node_country"]
    country = [ncmap[n] for n in ids]

    # (1) perfect predictor -> macro rmse/mae 0, pcc 1 on scored (non-constant) nodes
    agg, ns = score_bundle(raw.copy(), raw, test, ids, ncmap)
    assert abs(agg["rmse"]["country_macro"]) < 1e-9, agg["rmse"]
    assert abs(agg["mae"]["country_macro"]) < 1e-9
    assert abs(agg["pcc"]["country_macro"] - 1.0) < 1e-6, agg["pcc"]

    # (2) dominance test. Predict truth everywhere EXCEPT add a constant error of 10 to
    # every Bolivia cell (9 nodes). node-mean barely moves (Bolivia is 0.1% of nodes);
    # country-macro must move by ~10/12, because Bolivia is one of twelve equal countries.
    pred = raw.copy()
    for i, c in enumerate(country):
        if c == "bolivia":
            pred[i] = raw[i] + 10.0
    agg2, _ = score_bundle(pred, raw, test, ids, ncmap)
    macro_mae, node_mae = agg2["mae"]["country_macro"], agg2["mae"]["node_mean"]
    assert node_mae < 0.05, f"node-mean should be ~0 (Bolivia negligible), got {node_mae}"
    assert 0.7 < macro_mae < 0.9, f"country-macro should be ~10/12, got {macro_mae}"

    # (3) constant nodes excluded by default, scored when asked
    n_default = len(ns)
    _, ns_all = score_bundle(raw.copy(), raw, test, ids, ncmap, score_constant=True)
    assert len(ns_all) > n_default, "score_constant=True must score more nodes"
    n_const = sum(1 for s in ns_all.values() if s["constant"])
    print(f"ok  perfect-predictor macro exact; dominance neutralised "
          f"(node-mean {node_mae:.3f} vs country-macro {macro_mae:.3f} for a Bolivia-only error);")
    print(f"ok  {n_default} nodes scored, {n_const} constant nodes excluded by default")
    print(f"    country-macro is the mean of {agg['rmse']['n_countries']} per-country scores; "
          f"node-mean pools {agg['rmse']['n_nodes']} nodes")


if __name__ == "__main__":
    _demo()

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

METRICS added in Week 3 (Task 13.1), all per-node scalars over a node's observed eval cells
(in time order), count space:
  * sMAPE = mean(2*|yh-y| / (|y|+|yh|)) * 100, range 0-200. Cells with y==yh==0 are UNDEFINED and
    EXCLUDED from the mean -- never counted as zero error. dengue is ~78% imputed and Ebola has
    districts that never record a case, so a "0/0 == perfect" convention would hand a large free
    credit on exactly the sparsest data. A node whose every scored cell is 0/0 scores NaN.
  * peak_intensity = |max(yh) - max(y)| over the eval cells, in counts.
  * peak_timing = |argmax(yh) - argmax(y)| in observed-eval-cell steps (== weeks for the dense
    influenza panels; approximate where the mask is sparse). Undefined on a constant node, which
    is already excluded by default -- the self-check proves that exclusion.
CRPS / PICP / interval-width are UQ metrics left as stubs here; Week 5 (G4) fills them.

EBOLA METRIC RULES -- pre-registered now, in Week 3, while no Ebola number exists (Task 13.1). Ebola
is scored once in Week 5; fixing the rules here is what makes them demonstrably independent of the
data. Ebola runs at mask density 0.2175 with districts that never record a case, so PCC is
near-meaningless on much of the panel and sMAPE is undefined at zero:
  * MAE and RMSE are PRIMARY for Ebola.
  * PCC is reported only over districts with >= 5 observed non-zero cells, with the qualifying
    district count printed beside it.
  * sMAPE inherits the 0/0-excluded rule; if qualifying cells fall below 30% of the query set, drop
    sMAPE for Ebola entirely and say so.
  * peak_timing is undefined for districts with no peak; exclude them and report the count.
These are decisions, not defaults to revisit after seeing results.
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


def _smape(yh, y):
    """0-200 sMAPE; cells with y==yh==0 (denominator 0) are excluded, not scored as perfect."""
    denom = np.abs(y) + np.abs(yh)
    defined = denom > 0
    if not defined.any():
        return float("nan")               # every scored cell was 0/0 -> undefined for this node
    return float(np.mean(2.0 * np.abs(yh - y)[defined] / denom[defined]) * 100.0)


def _peak_intensity(yh, y):
    return float(abs(np.max(yh) - np.max(y)))


def _peak_timing(yh, y):
    return float(abs(int(np.argmax(yh)) - int(np.argmax(y))))


METRICS = {"rmse": _rmse, "mae": _mae, "pcc": _pcc,
           "smape": _smape, "peak_intensity": _peak_intensity, "peak_timing": _peak_timing}


# UQ metrics -- Week 5 (G4) fills these; they need the full quantile prediction, not a point
# forecast, so they do not belong in the point-metric METRICS dict above.
def _crps(*_a, **_k):                     # noqa: D401 - stub
    raise NotImplementedError("CRPS is a Week-5 deliverable (G4)")


def _picp(*_a, **_k):
    raise NotImplementedError("PICP is a Week-5 deliverable (G4)")


def _interval_width(*_a, **_k):
    raise NotImplementedError("interval width is a Week-5 deliverable (G4)")


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
        y, yh = truth[i][m], pred[i][m]                # observed eval cells, in time order
        constant = bool(y.std() < 1e-8)
        if constant and not score_constant:
            continue                                   # trivially predictable; kept in graph
        rec = dict(country=node_country[i], n_cells=int(m.sum()), constant=constant)
        rec.update({name: fn(yh, y) for name, fn in METRICS.items()})   # extend via METRICS, not here
        out[i] = rec
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

    # (4) sMAPE 0/0 rule: an all-(0,0) node is NaN (undefined), not 0 (perfect), and the excluded
    # cells are not silently scored as zero. Discriminating case: [0,0,5] vs [0,0,3] -> only the
    # third cell is defined, sMAPE = 2*2/(5+3)*100 = 50 (excluded), not 16.67 ("0/0 == 0").
    assert np.isnan(_smape(np.zeros(3), np.zeros(3))), "all-0/0 must be undefined, not perfect"
    assert abs(_smape(np.array([0., 0., 3.]), np.array([0., 0., 5.])) - 50.0) < 1e-9, "0/0 not excluded"
    assert _smape(np.array([5., 10.]), np.array([5., 10.])) == 0.0, "perfect predictor sMAPE must be 0"

    # (5) peak metrics: exact for a perfect predictor; peak_timing catches a shifted peak.
    yv = np.array([1., 5., 2., 8., 3.])
    assert _peak_intensity(yv, yv) == 0.0 and _peak_timing(yv, yv) == 0.0
    assert _peak_timing(np.array([8., 1., 1.]), np.array([1., 1., 8.])) == 2.0   # peak moved 2 steps
    # constant-truth node is excluded by default, so peak_timing is never computed on an undefined peak
    assert all(not s["constant"] for s in ns.values()), "a constant node leaked into the default scoring"

    print(f"ok  perfect-predictor macro exact; dominance neutralised "
          f"(node-mean {node_mae:.3f} vs country-macro {macro_mae:.3f} for a Bolivia-only error);")
    print(f"ok  {n_default} nodes scored, {n_const} constant nodes excluded by default")
    print(f"ok  sMAPE 0/0 excluded (all-0/0 -> NaN); peak intensity/timing exact for a perfect predictor")
    print(f"    metrics live: {list(METRICS)}")
    print(f"    country-macro is the mean of {agg['rmse']['n_countries']} per-country scores; "
          f"node-mean pools {agg['rmse']['n_nodes']} nodes")


if __name__ == "__main__":
    _demo()

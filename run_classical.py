"""run_classical.py -- the classical baselines the client asked for (Review Doc.md:79): a
gradient-boosted model on lag features, plus SARIMA/ARIMA. Sits beside run_baselines.py and
writes the SAME record schema the GNN baselines write, so results/baselines/ pools both.

WHAT IT RUNS
  --model gbm     sklearn HistGradientBoostingRegressor on the node's own recent count lags,
                  pooled across every node in a panel, ONE model per horizon (direct multi-step,
                  never recursive). 5 seeds (42/52/62/72/82), matching every baseline record.
  --model sarima  statsmodels SARIMAX per node, fitted on the TRAINING split only, then rolled
                  across the eval period WITHOUT re-estimation (get_prediction(dynamic=...) on the
                  filtered state, fixed params). Deterministic -> one record, seed=null, the
                  convention the naive floors use in results/naive/.
  --model arima   the same path with the seasonal order zeroed.

  --panel NAME    one dev panel; --all runs every dev panel (fast influenza first, dengue last).
  --ebola         the two frozen arms as an EXPLORATORY arm only (see EBOLA below).
  --smoke         influenza_japan, a handful of nodes, a small estimator budget, end to end in
                  under two minutes; writes to a temp dir, never into results/.
  --selfcheck     schema + mutation check, no data needed.

ENVIRONMENT (checked 2026-09-21). sklearn 1.7.2 lives in the `ebola` env, NOT in `ebola-train`,
and statsmodels is absent from BOTH. So GBM runs under `ebola`; SARIMA raises a clear ImportError
telling you to install statsmodels first. Nothing here imports torch, so `ebola` is enough:
    conda run -n ebola python run_classical.py --model gbm --all
    conda run -n ebola python run_classical.py --model sarima --all   # needs statsmodels installed

SCHEMA. Every dev-panel record carries exactly the keys an existing results/baselines/ record has
(model, dataset, horizon, seed, metric, country_macro, node_mean, n_countries, n_nodes, node_subset,
training_regime, sampler, gate_mode, topo_aug). Scoring is score.score_bundle -- the same authority
and the same country-macro count-space metric as the encoder and the naive floors, over the same
eval origins (bundles.origins(phase='test')) and the same test mask. Dengue uses the identical 1/3
stratified subsample the GNN baselines use (export_baseline._kept_indices), recorded in node_subset,
so the two sides pool like for like.

DENGUE is the cost driver: 2,392 subsampled nodes. GBM is trivial there (one pooled fit). SARIMA is
one fit per node, so ~2,392 fits; non-seasonal keeps that inside the hour.

EBOLA is EXPLORATORY and NOTHING here is the pre-registered result. The frozen prereg record in
results/ebola/ is never touched or re-scored. This arm: verifies the arm's content hash against
configs/ebola_arms.json (read-only), fits classical models on the SUPPORT cells, scores the SAME
query cells the naive floors use, and writes only into experiments/classical__<arm>__<model>.json
with protocol="EXPLORATORY". An assert forbids any write outside experiments/. The support window
is 13-21 weeks, shorter than the 20-week lookback, so a horizon whose target never lands in support
cannot be trained and falls back to persistence -- recorded per record. That non-fittability is
itself the honest point: classical models cannot learn Ebola from its own support alone.

# ponytail: deliberate ceilings, called out where they sit --
#   - GBM features are lags only (count space), no calendar/degree/cross-node engineering.
#   - SARIMA order is FIXED, not an AIC search per node (a reviewer can ask for the search).
#   - the seasonal term is on only where an annual cycle is real (the flu panels); off on covid
#     (waves are not annually periodic), dengue (2,392 gappy districts, unstable + the cost driver)
#     and ebola.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import json
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

import bundles
import export_baseline
import score

BASE = Path(__file__).resolve().parent
RESULTS_BL = BASE / "results" / "baselines"
EXPERIMENTS = BASE / "experiments"
CONFIG = BASE / "configs" / "ebola_arms.json"

SEEDS = (42, 52, 62, 72, 82)                 # identical to every baseline record (export_baseline.py)
HORIZONS = bundles.HORIZONS                   # (3, 5, 10, 15)
MAX_H = max(HORIZONS)
W = bundles.W                                  # 20, the encoder's lookback
W_EBOLA = 3                                    # ponytail: support is 13-21 wk < W; short lags only
MIN_PAIRS = 30                                 # below this a horizon's GBM cannot fit -> persistence

DEV_PANELS = ("influenza_japan", "influenza_us-regions", "influenza_us-states",
              "covid_us-states", "dengue")     # fast panels first, dengue (the cost driver) last
EBOLA_ARMS = ("ebola_L12", "ebola_L20")

# baseline rows carry these four null meta fields; they do not apply to a classical model.
NULL_META = dict(training_regime="single", sampler=None, gate_mode=None, topo_aug=None)

SARIMA_ORDER = (2, 1, 1)                        # ponytail: AR(2), one difference for trend, MA(1); fixed
SEASONAL = {                                    # (P, D, Q, s); s=52 weekly -> annual. arima zeroes this.
    "influenza_japan": (1, 0, 0, 52),
    "influenza_us-regions": (1, 0, 0, 52),
    "influenza_us-states": (1, 0, 0, 52),
    "covid_us-states": (0, 0, 0, 0),           # covid waves are not annually periodic
    "dengue": (0, 0, 0, 0),                    # 2,392 gappy districts: s=52 is unstable + too slow
}


# --------------------------------------------------------------------------- #
# panel plumbing
# --------------------------------------------------------------------------- #
def _kept(b, dataset):
    """(row indices scored, node_subset string). Dengue takes the GNN baselines' exact 1/3 subset."""
    if dataset == "dengue":
        idx = export_baseline._kept_indices(b)[0]
        return idx, f"{idx.size}/{len(b.meta['node_ids'])}"
    return np.arange(len(b.meta["node_ids"])), None


def _phases(dataset):
    return ("support", "query") if dataset.startswith("ebola") else ("train", "test")


def _lags(raw, origins, lookback):
    """[len(origins)*N, lookback] lag matrix. Row order is origin-major: block k is origin k over
    all N nodes, so predict().reshape(len(origins), N) recovers the (origin, node) grid."""
    return np.concatenate([raw[:, t - lookback + 1:t + 1] for t in origins], axis=0)


def _persist(pred_by_h, raw, origins):
    for t in origins:
        for h in HORIZONS:
            pred_by_h[h][:, t + h] = raw[:, t]


# --------------------------------------------------------------------------- #
# GBM: HistGradientBoostingRegressor on count-space lags, pooled over nodes, one model per horizon
# --------------------------------------------------------------------------- #
def gbm_run(b, kept, train_origins, eval_origins, feat_lookback, train_phase, seed, max_iter):
    from sklearn.ensemble import HistGradientBoostingRegressor

    raw = b.raw[kept].astype(np.float64)
    Nk, T = raw.shape
    tmask = b.masks()[train_phase][kept].astype(bool)     # observed AND in the training split
    pred_by_h = {h: np.zeros((Nk, T)) for h in HORIZONS}
    Xtr = _lags(raw, train_origins, feat_lookback) if train_origins else np.zeros((0, feat_lookback))
    Xev = _lags(raw, eval_origins, feat_lookback)
    fallbacks = []
    for h in HORIZONS:
        if train_origins:
            ytr = np.concatenate([raw[:, t + h] for t in train_origins])
            vtr = np.concatenate([tmask[:, t + h] for t in train_origins])
        else:
            ytr, vtr = np.zeros(0), np.zeros(0, bool)
        Xh, yh = Xtr[vtr], ytr[vtr]
        if yh.size < MIN_PAIRS:                            # cannot fit this horizon -> persistence floor
            for t in eval_origins:
                pred_by_h[h][:, t + h] = raw[:, t]
            fallbacks.append(h)
            continue
        m = HistGradientBoostingRegressor(random_state=seed, max_iter=max_iter)
        m.fit(Xh, yh)
        p = m.predict(Xev).reshape(len(eval_origins), Nk)
        for k, t in enumerate(eval_origins):
            pred_by_h[h][:, t + h] = np.maximum(p[k], 0.0)  # counts are non-negative
    return pred_by_h, fallbacks


# --------------------------------------------------------------------------- #
# SARIMA: per node, fit on train, roll over the eval period with fixed params (no re-estimation)
# --------------------------------------------------------------------------- #
def sarima_run(b, kept, eval_origins, train_phase, order, seasonal):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise SystemExit(
            "run_classical --model sarima needs statsmodels, which is absent from BOTH the `ebola`\n"
            "and `ebola-train` envs. Install it (nothing else is missing) and re-run:\n"
            "    conda run -n ebola python -m pip install statsmodels\n"
            "GBM has no such dependency; `--model gbm` runs today under the `ebola` env.")

    warnings.filterwarnings("ignore")                     # convergence chatter, one line per node
    raw = b.raw[kept].astype(np.float64)
    Mk = b.M[kept].astype(bool)
    tmask = b.masks()[train_phase][kept].astype(bool)
    Nk, T = raw.shape
    pred_by_h = {h: np.zeros((Nk, T)) for h in HORIZONS}
    fallback_nodes = 0
    for i in range(Nk):
        tcols = np.where(tmask[i])[0]
        if tcols.size < 8:                                # too few train points to fit anything stable
            for t in eval_origins:
                for h in HORIZONS:
                    pred_by_h[h][i, t + h] = raw[i, t]
            fallback_nodes += 1
            continue
        series = raw[i].copy()
        series[~Mk[i]] = np.nan                            # SARIMAX skips the update step on NaN cells
        train_end = int(tcols.max()) + 1
        try:
            res = SARIMAX(series[:train_end], order=order, seasonal_order=seasonal,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            res_full = res.apply(series, refit=False)      # fixed params, refilter the whole series
            for t in eval_origins:
                fc = np.asarray(res_full.get_prediction(
                    start=t + 1, end=t + MAX_H, dynamic=t + 1).predicted_mean)
                for h in HORIZONS:
                    v = fc[h - 1]
                    pred_by_h[h][i, t + h] = max(float(v) if np.isfinite(v) else raw[i, t], 0.0)
        except Exception:                                  # singular fit / LinAlg -> persistence for this node
            for t in eval_origins:
                for h in HORIZONS:
                    pred_by_h[h][i, t + h] = raw[i, t]
            fallback_nodes += 1
    return pred_by_h, fallback_nodes


# --------------------------------------------------------------------------- #
# scoring -> the baseline record schema
# --------------------------------------------------------------------------- #
def score_records(model, dataset, seed, pred_by_h, b, kept, eval_origins, eval_phase,
                  node_subset, extra=None):
    raw = b.raw[kept].astype(np.float64)
    pm = b.masks()[eval_phase][kept].astype(np.uint8)
    ids = [b.meta["node_ids"][i] for i in kept]
    ncmap = b.group_of()
    recs = []
    for h in HORIZONS:
        mask_h = np.zeros_like(pm)
        for t in eval_origins:
            mask_h[:, t + h] = pm[:, t + h]                # observed-and-in-phase target cells only
        agg, _ = score.score_bundle(pred_by_h[h], raw, mask_h, ids, ncmap)
        for metric, a in agg.items():
            r = dict(model=model, dataset=dataset, horizon=h, seed=seed, metric=metric,
                     country_macro=a["country_macro"], node_mean=a["node_mean"],
                     n_countries=a["n_countries"], n_nodes=a["n_nodes"],
                     node_subset=node_subset, **NULL_META)
            if extra:
                r.update(extra)
            recs.append(r)
    return recs


def _write_baseline(recs, out_dir):
    """One file per (model, dataset, horizon, seed), matching the GNN baseline layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for r in recs:
        by[(r["model"], r["dataset"], r["horizon"], r["seed"])].append(r)
    written = []
    for (m, d, h, s), rs in by.items():
        stem = f"{m}__{d}__h{h}__seed{'NA' if s is None else s}"
        p = out_dir / f"{stem}.json"
        p.write_text(json.dumps(rs, indent=2))
        written.append(p)
    return written


def _write_experiment(recs, arm, model):
    """EXPLORATORY: one file per (arm, model), confined to experiments/."""
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    p = (EXPERIMENTS / f"classical__{arm}__{model}.json").resolve()
    assert EXPERIMENTS.resolve() == p.parent, f"output escaping experiments/: {p}"
    p.write_text(json.dumps(dict(protocol="EXPLORATORY", arm=arm, model=model, records=recs), indent=2))
    return p


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def run_dev(panel, model, seeds, max_iter, out_dir):
    b = bundles.load(panel)
    kept, node_subset = _kept(b, panel)
    train_phase, eval_phase = _phases(panel)
    train_origins = b.origins(w=W, phase=train_phase)
    eval_origins = b.origins(w=W, phase=eval_phase)
    print(f"  {panel}: nodes={kept.size} train_origins={len(train_origins)} "
          f"eval_origins={len(eval_origins)} subset={node_subset or 'full'}")
    written = []
    if model == "gbm":
        for s in seeds:
            t0 = time.time()
            pred, fb = gbm_run(b, kept, train_origins, eval_origins, W, train_phase, s, max_iter)
            recs = score_records("gbm", panel, s, pred, b, kept, eval_origins, eval_phase, node_subset)
            written += _write_baseline(recs, out_dir)
            fbnote = f" (persistence fallback h={fb})" if fb else ""
            print(f"    gbm seed{s}: {time.time() - t0:.1f}s{fbnote}")
    else:
        seasonal = (0, 0, 0, 0) if model == "arima" else SEASONAL[panel]
        t0 = time.time()
        pred, fbn = sarima_run(b, kept, eval_origins, train_phase, SARIMA_ORDER, seasonal)
        recs = score_records(model, panel, None, pred, b, kept, eval_origins, eval_phase, node_subset)
        written += _write_baseline(recs, out_dir)
        print(f"    {model} order={SARIMA_ORDER} seasonal={seasonal}: {time.time() - t0:.1f}s "
              f"({fbn}/{kept.size} nodes fell back to persistence)")
    return written


def _verify_ebola_hash(arm):
    cfg = json.loads(CONFIG.read_text())
    want = cfg["arms"][arm]["sha256_content"]
    got = bundles.content_sha256(bundles.DATA_DIR / f"{arm}.npz")
    assert got == want, f"{arm}: content hash {got} != frozen {want} -- the arm has MOVED, refusing to run"
    return want[:12]


def run_ebola(arm, model, seeds, max_iter):
    tag = _verify_ebola_hash(arm)
    print(f"  {arm}: frozen content hash {tag}.. verified (read-only; the prereg is untouched)")
    b = bundles.load(arm)
    kept = np.arange(len(b.meta["node_ids"]))
    train_origins = b.origins(w=W_EBOLA, phase="support")   # short lags: support is 13-21 wk
    eval_origins = b.origins(w=W, phase="query")            # the prereg / naive-floor query cell set
    print(f"    train_origins={len(train_origins)} (support, lookback {W_EBOLA})  "
          f"eval_origins={len(eval_origins)} (query, matches the floors)")
    recs = []
    if model == "gbm":
        for s in seeds:
            pred, fb = gbm_run(b, kept, train_origins, eval_origins, W_EBOLA, "support", s, max_iter)
            recs += score_records("gbm", arm, s, pred, b, kept, eval_origins, "query", None,
                                  extra=dict(protocol="EXPLORATORY", lookback=W_EBOLA,
                                             persistence_fallback_h=fb))
    else:
        seasonal = (0, 0, 0, 0)                              # ebola has no annual cycle to fit
        pred, fbn = sarima_run(b, kept, eval_origins, "support", SARIMA_ORDER, seasonal)
        recs += score_records(model, arm, None, pred, b, kept, eval_origins, "query", None,
                              extra=dict(protocol="EXPLORATORY", nodes_fell_back=fbn))
    p = _write_experiment(recs, arm, model)
    print(f"    wrote {p.relative_to(BASE)}  (EXPLORATORY, {len(recs)} records)")
    return [p]


# --------------------------------------------------------------------------- #
# smoke + selfcheck
# --------------------------------------------------------------------------- #
def smoke(model):
    t0 = time.time()
    tmp = Path(tempfile.gettempdir()) / "classical_smoke"
    tmp.mkdir(exist_ok=True)
    b = bundles.load("influenza_japan")
    kept = np.arange(min(10, len(b.meta["node_ids"])))       # a handful of nodes
    # NB: do NOT truncate the origin set. The earliest test origins only have their LONG horizons in
    # the test split (the short ones still land in val), so a head-slice would score 0 cells at h3.
    # Japan is 47 nodes / ~150 origins, so the full origin sweep on 10 nodes is still well under 2 min.
    train_origins = b.origins(w=W, phase="train")
    eval_origins = b.origins(w=W, phase="test")
    print(f"smoke: influenza_japan, {kept.size} nodes, {len(train_origins)} train / "
          f"{len(eval_origins)} eval origins, model={model}")
    if model == "gbm":
        pred, fb = gbm_run(b, kept, train_origins, eval_origins, W, "train", 42, max_iter=20)
        recs = score_records("gbm", "influenza_japan", 42, pred, b, kept, eval_origins, "test", None)
    else:
        seasonal = (0, 0, 0, 0) if model == "arima" else SEASONAL["influenza_japan"]
        pred, fbn = sarima_run(b, kept, eval_origins, "train", SARIMA_ORDER, seasonal)
        recs = score_records(model, "influenza_japan", None, pred, b, kept, eval_origins, "test", None)
    written = _write_baseline(recs, tmp)
    _assert_schema(recs[0])                                  # the emitted record must match a baseline record
    print(f"smoke OK in {time.time() - t0:.1f}s: {len(recs)} records, {len(written)} files -> {tmp}")
    for r in recs[:2]:
        print(f"   {r['model']} h{r['horizon']} {r['metric']}={r['country_macro']:.3f} "
              f"(node_mean={r['node_mean']:.3f}, n_nodes={r['n_nodes']})")


def _baseline_keys():
    """The key set of a real results/baselines/ record, or a hard-coded fallback if none on disk."""
    for p in sorted(RESULTS_BL.glob("*.json")):
        recs = json.loads(p.read_text())
        if recs:
            return set(recs[0]), p.name
    return ({"model", "dataset", "horizon", "seed", "metric", "country_macro", "node_mean",
             "n_countries", "n_nodes", "node_subset", "training_regime", "sampler",
             "gate_mode", "topo_aug"}, "(no baseline file; hard-coded key set)")


def _assert_schema(rec):
    want, src = _baseline_keys()
    got = set(rec)
    assert got == want, f"schema drift vs {src}: missing {want - got}, extra {got - want}"


def selfcheck():
    want, src = _baseline_keys()
    print(f"schema reference: {src}  ({len(want)} keys)")
    sample = dict(model="gbm", dataset="influenza_japan", horizon=3, seed=42, metric="rmse",
                  country_macro=1.0, node_mean=1.0, n_countries=1, n_nodes=47,
                  node_subset=None, **NULL_META)
    _assert_schema(sample)
    print("ok  emitted record matches the baseline key set exactly")
    # mutation test: drop a key and confirm the check FAILS, so we know it can fail
    broken = dict(sample); broken.pop("node_subset")
    try:
        _assert_schema(broken)
    except AssertionError:
        print("ok  mutation caught: a record missing 'node_subset' is rejected")
    else:
        raise SystemExit("FAIL: schema check did not fire on a broken record -- the check is dead")
    # a second mutation: an extra key must also be caught
    broken2 = dict(sample); broken2["surprise"] = 1
    try:
        _assert_schema(broken2)
    except AssertionError:
        print("ok  mutation caught: an unexpected extra key is rejected")
    else:
        raise SystemExit("FAIL: schema check ignored an extra key")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["gbm", "sarima", "arima"], default="gbm")
    ap.add_argument("--panel", choices=DEV_PANELS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ebola", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--max-iter", type=int, default=300, help="GBM boosting iterations (ceiling)")
    a = ap.parse_args()

    if a.selfcheck:
        return selfcheck()
    if a.smoke:
        return smoke(a.model)

    t0 = time.time()
    print(f"model={a.model}, seeds={SEEDS if a.model == 'gbm' else '(deterministic, seed=null)'}")
    if a.ebola:
        for arm in EBOLA_ARMS:
            run_ebola(arm, a.model, SEEDS, a.max_iter)
    if a.all:
        for panel in DEV_PANELS:
            run_dev(panel, a.model, SEEDS, a.max_iter, RESULTS_BL)
    elif a.panel:
        run_dev(a.panel, a.model, SEEDS, a.max_iter, RESULTS_BL)
    elif not a.ebola:
        ap.error("give --panel NAME, or --all, or --ebola, or --smoke, or --selfcheck")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

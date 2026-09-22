"""The disease-mean floor: a constant that knows the whole outbreak.

EXPLORATORY. This floor is fitted on the COMPLETE Ebola panel, query period included, so it peeks
at the very cells it is scored on. It is not a deployable forecast and it is not a pre-registered
floor (Ebola_Prereg.md:145 fixes persistence and support_mean only). It is an evidential bar: what
a forecaster that had seen the entire outbreak could do with a single number.

Two variants, because "average the outbreak then average those values" admits two readings:
  disease_mean          one global constant: per-district full-panel mean, averaged over districts
  disease_mean_pernode  each district keeps its own full-panel mean. Direct sibling of
                        support_mean, which uses the support window instead of the full panel.

Nothing is written to results/. The scored Ebola records are never re-scored; the model side is
read out of the archived JSONs.

Run:  conda run -n ebola-train python -m experiments.disease_mean_floor
      conda run -n ebola-train python -m experiments.disease_mean_floor --selfcheck
"""
import argparse
import json
import sys

import numpy as np

import bundles
from bundles import HORIZONS
from train.loop import score_predictions
from train.ebola import ebola_naive_predictions

ARMS = ("ebola_L12", "ebola_L20")
SEEDS = (42, 52, 62, 72, 82)
TOL = 1e-9


def disease_means(b):
    """(global scalar, per-district vector) over every OBSERVED cell of the complete panel."""
    raw = b.raw.astype(np.float64)
    obs = (b.M == 1)
    per = np.array([raw[i, obs[i]].mean() if obs[i].any() else 0.0 for i in range(raw.shape[0])])
    return float(per[obs.any(1)].mean()), per


def floor_preds(b, te):
    """{model: {h: [N,T]}} for the two disease-mean variants, on the scored cells only."""
    n, t_len = b.raw.shape
    g, per = disease_means(b)
    out = {m: {h: np.zeros((n, t_len)) for h in HORIZONS}
           for m in ("disease_mean", "disease_mean_pernode")}
    for t in te:
        for h in HORIZONS:
            out["disease_mean"][h][:, t + h] = g
            out["disease_mean_pernode"][h][:, t + h] = per
    return out


def macro(recs, model, metric):
    return {r["horizon"]: r["country_macro"] for r in recs
            if r["model"] == model and r["metric"] == metric}


def published(arm):
    with open("results/naive/naive__%s.json" % arm) as fh:
        return json.load(fh)


def gate(arm, b, te):
    """Reproduce BOTH published floors through the same scorer, or refuse to go on.

    Without this the new floor is a number I made up. With it, the scoring path is shown to return
    two known answers on the identical cells before it is asked a new question."""
    ref = published(arm)
    got = []
    for m, preds in ebola_naive_predictions(b, te).items():
        got += score_predictions(m, arm, None, preds, b, te, phase="query")[0]
    bad = 0
    for r in ref:
        if r["metric"] not in ("rmse", "mae"):
            continue
        mine = macro(got, r["model"], r["metric"])[r["horizon"]]
        if abs(mine - r["country_macro"]) > TOL:
            print("  GATE FAIL %s %s %s h%d: published %.9f mine %.9f"
                  % (arm, r["model"], r["metric"], r["horizon"], r["country_macro"], mine))
            bad += 1
    if bad:
        sys.exit("gate failed on %s: %d published floor values not reproduced" % (arm, bad))
    print("  gate ok (%s): persistence and support_mean reproduced to %g on 8 cells" % (arm, TOL))


def model_macro(arm, prefix, metric):
    """Seed-mean country-macro of an archived Ebola arm."""
    out = {}
    for h in HORIZONS:
        vals = []
        for s in SEEDS:
            with open("results/ebola/%s__%s__seed%d.json" % (prefix, arm, s)) as fh:
                recs = json.load(fh)
            vals += [r["country_macro"] for r in recs
                     if r["horizon"] == h and r["metric"] == metric]
        out[h] = float(np.mean(vals))
    return out


def selfcheck():
    """Three properties that must hold whatever the data says."""
    b = bundles.load("ebola_L12")
    g, per = disease_means(b)
    raw, obs = b.raw.astype(np.float64), (b.M == 1)
    assert abs(g - per[obs.any(1)].mean()) < 1e-12, "global constant is not the mean of the means"
    sm = b.masks()["support"].astype(bool)
    sup = np.array([raw[i, sm[i]].mean() if sm[i].any() else 0.0 for i in range(raw.shape[0])])
    assert abs(g - sup[sm.any(1)].mean()) > 1e-6, "disease mean equals support mean, not a new floor"
    # Every zero in this floor must be a district that genuinely never reported a case, never the
    # no-data fallback support_mean leans on. That is the whole point of the comparison.
    zero = np.where((per == 0) & obs.any(1))[0]
    assert all((raw[i, obs[i]] == 0).all() for i in zero), "a zero here is a fallback, not real data"
    assert len(zero) < (sup == 0).sum(), "disease mean has no fewer zero districts than support mean"
    print("  selfcheck ok: global mean %.4f against support-window mean %.4f; %d of %d districts "
          "sit at zero and every one truly never reported a case, against %d for support_mean"
          % (g, sup[sm.any(1)].mean(), len(zero), int(obs.any(1).sum()), int((sup == 0).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    selfcheck()
    if a.selfcheck:
        return

    summary = {}
    for arm in ARMS:
        b = bundles.load(arm)
        te = b.origins(phase="query")
        g, per = disease_means(b)
        raw = b.raw.astype(np.float64)
        sm = b.masks()["support"].astype(bool)
        sup = np.array([raw[i, sm[i]].mean() if sm[i].any() else 0.0 for i in range(raw.shape[0])])
        print("\n=== %s" % arm)
        print("  disease mean (one constant, whole panel) = %.4f cases per week" % g)
        print("  support-window mean over districts that have support = %.4f" % sup[sm.any(1)].mean())
        print("  districts on a ZERO support_mean floor: %d/%d    on a zero disease_mean floor: %d/%d"
              % (int((sup == 0).sum()), len(sup), int((per == 0).sum()), len(per)))
        gate(arm, b, te)

        recs = []
        for m, preds in floor_preds(b, te).items():
            recs += score_predictions(m, arm, None, preds, b, te, phase="query")[0]
        ref = published(arm)

        for metric in ("rmse", "mae"):
            fl = {m: macro(recs, m, metric) for m in ("disease_mean", "disease_mean_pernode")}
            fl["persistence"] = macro(ref, "persistence", metric)
            fl["support_mean"] = macro(ref, "support_mean", metric)
            zs = model_macro(arm, "encoder_ebola_zeroshot", metric)
            fs = model_macro(arm, "encoder_ebola", metric)
            print("\n  -- %s country-macro, %s" % (metric.upper(), arm))
            head = "".join("%11s" % ("h%d" % h) for h in HORIZONS)
            print("  %-22s%s" % ("series", head))
            rows = (("persistence", fl["persistence"]), ("support_mean", fl["support_mean"]),
                    ("disease_mean", fl["disease_mean"]),
                    ("disease_mean_pernode", fl["disease_mean_pernode"]),
                    ("MODEL zero-shot", zs), ("MODEL few-shot", fs))
            for name, d in rows:
                print("  %-22s%s" % (name, "".join("%11.3f" % d[h] for h in HORIZONS)))
            for label, d in (("zero-shot", zs), ("few-shot", fs)):
                for fname in ("disease_mean", "disease_mean_pernode"):
                    imp = [(fl[fname][h] - d[h]) / fl[fname][h] * 100.0 for h in HORIZONS]
                    win = sum(1 for v in imp if v > 0)
                    print("  %-9s vs %-21s%s   beats it %d/4"
                          % (label, fname, "".join("%+10.1f%%" % v for v in imp), win))
            summary["%s__%s" % (arm, metric)] = dict(
                floors={k: {str(h): v[h] for h in HORIZONS} for k, v in fl.items()},
                model_zeroshot={str(h): zs[h] for h in HORIZONS},
                model_fewshot={str(h): fs[h] for h in HORIZONS})

    out = "experiments/disease_mean_floor.json"
    with open(out, "w") as fh:
        json.dump(dict(protocol="EXPLORATORY",
                       note="floor fitted on the complete panel including query cells; not "
                            "pre-registered; not a deployable forecast",
                       results=summary), fh, indent=2)
    print("\nwrote %s  (protocol=EXPLORATORY)" % out)


if __name__ == "__main__":
    main()

"""TEST 6c: is the dev-panel neighbour gain just 'big districts sit next to big districts'?

On Ebola (experiments/t6_ebola_neighbour_signal.py, EXPLORATORY) the neighbour gain vanished once each
district's own average level was given to the model: Ebola uses one pooled scale, so a district's
persistent size stays inside its deviation, and the neighbour mean was mostly reporting which area was
hot. The dev panels use a per-district z-score fitted on train, which should centre every district's
train-period level. This script checks that on disk rather than assuming it.

For each dev panel: recompute the t6 real-map gain with and without one extra column, the district's
mean TRAIN-phase deviation, added to A and B alike (t6.probe level=True). No null is rebuilt; the
question is whether the gain moves. The no-level gain is ASSERTED equal to
results/misc/t6_neighbour_signal.json, which certifies the level option left the default path untouched.

Run:  conda run -n ebola-train python diagnostics/graph_probe/t6c_level_check.py
Writes results/misc/t6c_level_check.json. Dengue takes about a minute.
"""
import os, sys, json, time, datetime
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.chdir(r"F:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "diagnostics", "graph_probe"))
import numpy as np
import t6_neighbour_signal as t6

REF = "results/misc/t6_neighbour_signal.json"
OUT = "results/misc/t6c_level_check.json"


def main():
    ref = json.load(open(REF))
    assert not ref["quick"], "reference is a quick smoke run"
    out = {"script": "diagnostics/graph_probe/t6c_level_check.py", "date": datetime.date.today().isoformat(),
           "reference": REF, "level_feature": "district mean train-phase deviation, added to A and B",
           "panels": {}}
    for name in t6.PANELS:
        t0 = time.time()
        P = t6.load_panel(name)
        base = t6.probe(P, 0, spread=False, tag=f"{name}-nolevel")
        lev = t6.probe(P, 0, spread=False, tag=f"{name}-level", level=True)
        d_fit = t6.deviation(P["x"], P["train"], P["gid"])
        cnt = P["train"].sum(1)
        lvl = np.where(cnt > 0, np.where(P["train"], d_fit, 0.0).sum(1) / np.maximum(cnt, 1), 0.0)
        hz = {}
        for h in t6.HORIZONS:
            g0 = base["horizons"][str(h)]["B"]["gain_real_pct"]
            want = ref["panels"][name]["horizons"][str(h)]["B"]["gain_real_pct"]
            assert abs(g0 - want) < 1e-9, f"{name} h{h}: no-level gain {g0} != {REF} {want}; default path moved"
            g1 = lev["horizons"][str(h)]["B"]["gain_real_pct"]
            hz[str(h)] = {"gain_pct": g0, "gain_level_pct": g1, "change_pct_points": g1 - g0}
        out["panels"][name] = {"level_sd": float(lvl.std()), "level_max_abs": float(np.abs(lvl).max()),
                               "nodes_level_nonzero": int((np.abs(lvl) > 1e-6).sum()), "N": int(len(lvl)),
                               "max_abs_change_pct_points": max(abs(v["change_pct_points"]) for v in hz.values()),
                               "horizons": hz, "seconds": round(time.time() - t0, 1)}
        r = out["panels"][name]
        print(f"{name:22s} level sd {r['level_sd']:.4f} nonzero {r['nodes_level_nonzero']:4d}/{r['N']:<5d} | " +
              " | ".join(f"h{h}: {hz[str(h)]['gain_pct']:+.3f} -> {hz[str(h)]['gain_level_pct']:+.3f}"
                         for h in t6.HORIZONS) + f"  ({r['seconds']}s)")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

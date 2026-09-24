"""TEST 6b: does the t6 neighbour probe have a false-alarm floor? Per-panel shuffled-base wrong-map test.

Why. In the t6 run the legacy wrong-map control (spread planted through a relabelled map on the REAL
Japan base, probed with the real map) was flagged WIN at every horizon (+0.62 / +0.55 / +0.68 / +0.91%).
That control cannot tell a false alarm from the real-map signal the Japan base already carries (+1.21%
at h3). This script separates the two.

Arms, per panel, R draws each (draw r uses seed SEED0 - r):
  shuffled  base district identities shuffled inside each group (whole rows of x, M, train, test move
            together), so the real map is meaningless for the base; 0.5 SD spread planted through a
            relabelled map W; probed with the REAL map against its own relabel null.
            THIS is the false-alarm test. A calibrated probe wins at about the nominal 5% rate.
  realbase  the legacy arm: same plant on the unshuffled base, gains only (no null). Shown so the doc
            can say where the original WIN flags came from.
  base      the unplanted real data, recomputed and ASSERTED equal to results/misc/t6_neighbour_signal.json
            (gain and MSE_A), which certifies the 2026-09-24 edits to t6 left the probe path unchanged.

Pre-committed rule, written before any result:
  A false-alarm FLOOR EXISTS on a panel if, at any horizon, shuffled-arm WINs reach the one-sided 5%
  binomial critical count for a 5% rate over R draws (R=20: 4 or more; R=6: 2 or more).
  Floor size is reported either way as the 95th percentile of the shuffled-arm real-map gains per
  horizon (the max when R < 20). A real-data win that sits below the floor size is not read as signal.

Run (small panels, about a minute):
  conda run -n ebola-train python diagnostics/graph_probe/t6b_wrongmap_floor.py
Dengue (handed to the user, see --help for the cost knobs):
  conda run -n ebola-train python diagnostics/graph_probe/t6b_wrongmap_floor.py --panels dengue --draws 6 --perms 20
Writes / merges results/misc/t6b_wrongmap_floor.json (one entry per panel). Reads bundles and the t6 JSON.
"""
import os, sys, json, time, datetime, argparse, math
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.chdir(r"F:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "diagnostics", "graph_probe"))
import numpy as np
import t6_neighbour_signal as t6

T6_JSON = "results/misc/t6_neighbour_signal.json"
OUT = "results/misc/t6b_wrongmap_floor.json"
SEED0 = 20260924 - 500
SMALL = ["influenza_japan", "influenza_us-states", "covid_us-states"]


def crit(R, p=0.05, alpha=0.05):
    """Smallest c with P(X >= c | R, p) <= alpha."""
    for c in range(R + 1):
        tail = sum(math.comb(R, k) * p ** k * (1 - p) ** (R - k) for k in range(c, R + 1))
        if tail <= alpha:
            return c
    return R + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=SMALL)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--perms", type=int, default=200)
    a = ap.parse_args()
    ref = json.load(open(T6_JSON))
    assert not ref["quick"], "t6 reference JSON is a quick smoke run; refuse"
    out = json.load(open(OUT)) if os.path.exists(OUT) else {"panels": {}}
    out.update(script="diagnostics/graph_probe/t6b_wrongmap_floor.py",
               date=datetime.date.today().isoformat(), reference=T6_JSON,
               rule="floor EXISTS if shuffled-arm WINs >= binomial critical count (5% rate, one-sided "
                    "0.05) at any horizon; floor size = 95th pct (max if R<20) of shuffled-arm gains")
    H = t6.HORIZONS
    for name in a.panels:
        t0 = time.time()
        P = t6.load_panel(name)
        # certify the probe path against the published run
        b0 = t6.probe(P, 0, spread=False, tag=f"{name}-base")
        for h in H:
            got, want = b0["horizons"][str(h)], ref["panels"][name]["horizons"][str(h)]
            assert abs(got["B"]["gain_real_pct"] - want["B"]["gain_real_pct"]) < 1e-9 and \
                abs(got["mse_A"] - want["mse_A"]) < 1e-12, \
                f"{name} h{h}: recomputed base gain differs from {T6_JSON}; probe path changed, stop"
        shuf = {h: [] for h in H}; wins = {h: 0 for h in H}; p95s = {h: [] for h in H}
        real = {h: [] for h in H}; dmse_real = {h: [] for h in H}
        for r in range(a.draws):
            s = SEED0 - r
            rs, _ = t6.shuffled_base_wrong_map(P, s, a.perms, shuffle=True)
            rr, _ = t6.shuffled_base_wrong_map(P, s, 0, shuffle=False)
            for h in H:
                c = rs["horizons"][str(h)]["B"]
                shuf[h].append(c["gain_real_pct"]); p95s[h].append(c["null_p95_pct"])
                wins[h] += int(c["win"])
                cr = rr["horizons"][str(h)]
                real[h].append(cr["B"]["gain_real_pct"])
                dmse_real[h].append(cr["mse_A"] - cr["B"]["mse_real"])
        c_crit = crit(a.draws)
        floor = {}
        for h in H:
            g = np.array(shuf[h])
            size = float(np.percentile(g, 95)) if a.draws >= 20 else float(g.max())
            bh = ref["panels"][name]["horizons"][str(h)]
            floor[str(h)] = {
                "shuffled_wins": wins[h], "shuffled_gain_mean_pct": float(g.mean()),
                "shuffled_gain_max_pct": float(g.max()), "floor_size_pct": size,
                "shuffled_null_p95_mean_pct": float(np.mean(p95s[h])),
                "shuffled_gains_pct": [float(v) for v in g],
                "realbase_gain_mean_pct": float(np.mean(real[h])),
                "realbase_dmse_mean": float(np.mean(dmse_real[h])),
                "base_gain_pct": bh["B"]["gain_real_pct"], "base_win": bh["B"]["win"],
                "base_dmse": bh["mse_A"] - bh["B"]["mse_real"],
                "base_clears_floor": bool(bh["B"]["win"] and bh["B"]["gain_real_pct"] > size)}
        exists = any(wins[h] >= c_crit for h in H)
        out["panels"][name] = {"draws": a.draws, "perms": a.perms, "seed0": SEED0,
                               "critical_wins": c_crit, "floor_exists": exists, "horizons": floor,
                               "seconds": round(time.time() - t0, 1)}
        print(f"{name}: floor {'EXISTS' if exists else 'absent'} (critical {c_crit} of {a.draws}), "
              f"{out['panels'][name]['seconds']}s")
        for h in H:
            f = floor[str(h)]
            print(f"  h{h:<2d} shuffled wins {f['shuffled_wins']:2d}/{a.draws}  gain mean {f['shuffled_gain_mean_pct']:+.3f}"
                  f" max {f['shuffled_gain_max_pct']:+.3f} floor {f['floor_size_pct']:+.3f} | real base "
                  f"{f['base_gain_pct']:+.3f} {'WIN' if f['base_win'] else '   '} clears {f['base_clears_floor']} | "
                  f"legacy realbase gain {f['realbase_gain_mean_pct']:+.3f} dMSE {f['realbase_dmse_mean']:.5f} "
                  f"vs base dMSE {f['base_dmse']:.5f}")
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""Figure 1: deviation branch against v1, paired over 15 seeds, test split, COVID (US states).

Run from the repository root:  python fig1_deviation_branch.py [repo_root] [out_dir]
Defaults: repo_root = ".", out_dir = "figures". Reads committed records only and asserts
known values before plotting, so a changed input stops the build."""
import os, sys
REPO = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT_DIR, exist_ok=True)
import json, math, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SEEDS = list(range(42, 183, 10)); assert len(SEEDS) == 15
H = [3, 5, 10, 15]; T = 2.1447866879   # t(0.975, 14)
def load(f):
    out = {}
    for r in json.load(open(f"{REPO}/{f}")):
        if r["model"] != "encoder": continue
        k = (r["metric"], r["horizon"]); assert k not in out; out[k] = r["node_mean"]
    assert len(out) == 28, f; return out
def ref(s):
    return load(f"results/single/encoder__covid_us-states__seed{s}.json" if s <= 82
                else f"ablation/single/encoder__covid_us-states__seed{s}__v1ref.json")
V1 = {s: ref(s) for s in SEEDS}
ARMS = [("v2graph", "Real map", "#2a78d6", "o"), ("v2nograph", "No map", "#eb6834", "s"),
        ("v2shuf", "Wrong map", "#1baf7a", "^")]
def stats_for(arm, m, h):
    A = {s: load(f"ablation/single/encoder__covid_us-states__seed{s}__{arm}.json") for s in SEEDS}
    d = [A[s][(m, h)] - V1[s][(m, h)] for s in SEEDS]; base = st.mean(V1[s][(m, h)] for s in SEEDS)
    mu = st.mean(d); half = T * st.stdev(d) / math.sqrt(15)
    return 100 * mu / base, 100 * (mu - half) / base, 100 * (mu + half) / base, mu
R = {(a, m, h): stats_for(a, m, h) for a, *_ in ARMS for m in ("rmse", "mae") for h in H}
assert abs(R[("v2graph", "rmse", 3)][3] - 5138.013) < 1e-3          # pre-registered verdict table
assert abs(R[("v2graph", "rmse", 3)][0] - 96.8) < 0.05 and abs(R[("v2graph", "rmse", 5)][0] - 137.6) < 0.05
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, m, title in zip(axes, ("rmse", "mae"), ("RMSE", "MAE")):
    ax.axhline(0, color="#898781", lw=1)
    for i, (arm, lab, col, mk) in enumerate(ARMS):
        xs = [j + (i - 1) * 0.2 for j in range(4)]
        pts = [R[(arm, m, h)] for h in H]
        ax.errorbar(xs, [p[0] for p in pts], yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]],
                    fmt=mk, color=col, ms=6, lw=1.5, capsize=0, mec="#fcfcfb", mew=1, label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{h} weeks" for h in H])
    ax.set_title(title, fontsize=9.5, color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("Change in error against the model\nwithout the branch (per cent)")
axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
fig.tight_layout()
out = os.path.join(OUT_DIR, "fig1_deviation_branch.png")
fig.savefig(out, dpi=200, facecolor="#fcfcfb"); print("wrote", out)
for a, *_ in ARMS:
    print(a, {f"{m}{h}": tuple(round(x, 1) for x in R[(a, m, h)][:3]) for m in ("rmse", "mae") for h in H})

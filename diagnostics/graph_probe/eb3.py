import os, json, glob, numpy as np, collections
os.chdir(r"f:\Quickgen Projects\Research Paper\Ebola-Research")
V = collections.defaultdict(list)
for f in glob.glob("results/ebola/*.json"):
    for r in json.load(open(f)):
        V[(r["model"], r["dataset"], r["metric"], r["horizon"])].append(r["country_macro"])
print(f"{'arm':10s} {'metric':7s} {'h':>3s} | {'zero-shot':>10s} {'few-shot':>10s} | winner")
print("-"*62)
tally = collections.Counter()
for ds in ("ebola_L12", "ebola_L20"):
    for met in ("rmse", "mae", "pcc", "smape"):
        for h in (3, 5, 10, 15):
            z = np.mean(V[("encoder_ebola_zeroshot", ds, met, h)])
            a = np.mean(V[("encoder_ebola", ds, met, h)])
            better_low = met != "pcc"
            win = "ZERO-SHOT" if ((z < a) == better_low) else "few-shot"
            tally[(ds, win)] += 1; tally[win] += 1
            print(f"{ds:10s} {met:7s} {h:3d} | {z:10.3f} {a:10.3f} | {win}")
print("\nTALLY", dict(tally))

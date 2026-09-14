import os, sys, numpy as np
os.chdir(r"f:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import bundles
tot_obs = 0
for n in bundles.BUNDLE_NAMES:
    try:
        b = bundles.load(n)
    except Exception as e:
        print(n, "LOAD FAIL", e); continue
    X = b.X
    M = getattr(b, "M", None)
    obs = int(M.sum()) if M is not None else X.shape[0]*X.shape[1]
    tot_obs += obs
    print(f"{n:24s} X={tuple(X.shape)}  observed cells={obs:,}")
print("TOTAL observed node-week cells across all bundles:", f"{tot_obs:,}")

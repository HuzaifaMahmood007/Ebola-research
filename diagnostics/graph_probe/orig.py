import os, sys
os.chdir(r"f:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import bundles
print(f"{'bundle':22s} {'N':>6s} {'T':>6s} {'train':>7s} {'val':>5s} {'test':>6s}")
for n in bundles.BUNDLE_NAMES:
    b = bundles.load(n)
    try:
        tr, va, te = len(b.origins(phase="train")), len(b.origins(phase="val")), len(b.origins(phase="test"))
    except Exception as e:
        tr = va = te = -1
    print(f"{n:22s} {b.X.shape[0]:6d} {b.X.shape[1]:6d} {tr:7d} {va:5d} {te:6d}")

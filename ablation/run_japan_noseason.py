"""Japan-only ablation: zero sin_doy/cos_doy (transfer_view channels 1,2) and compare to the
seasonal-phase baseline in results/. Settles ONE question: does the encoder USE the seasonal
phase it already receives?

  ablation ~= baseline (within seed noise) -> phase is UNUSED -> capacity/training problem (fixable)
  ablation  <  baseline (real degradation) -> phase IS used   -> residual gap is the missing
                                              year-over-year amplitude anchor (genuinely w=20-bound)

Writes ablation/encoder__influenza_japan__seed{S}__noseason.json (+ __pernode.npz) so the run is
inspectable, not in-memory. Reads the baseline from results/ (produced by `train.loop --all`).

Run from the repo root:
  PYTHONNOUSERSITE=1 conda run -n ebola-train python ablation/run_japan_noseason.py
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (pandas MKL + torch)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)                                 # so `import train.loop`, `import bundles` resolve

import json, glob, math
from pathlib import Path
from collections import defaultdict
import train.loop as L

DS = "influenza_japan"
SEEDS = L.SEEDS
HZ = [3, 5, 10, 15]
POINT = ["rmse", "mae", "pcc"]
FIELD = "node_mean"                                      # japan is a single synthesised group
RESULTS = Path(ROOT) / "results"


def mean_sd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), 0.0
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def main():
    L.RESULTS = Path(HERE)                                # send write_records / write_per_node here
    abl = defaultdict(list)
    for s in SEEDS:
        recs, pernode = L.train_one(DS, s, zero_channels=(1, 2), verbose=False)
        L.write_records(recs, f"encoder__{DS}__seed{s}__noseason.json")
        L.write_per_node(pernode, f"encoder__{DS}__seed{s}__noseason__pernode.npz")
        for r in recs:
            abl[(r["horizon"], r["metric"])].append(r[FIELD])
        print(f"  ablation seed{s} done -> ablation/encoder__{DS}__seed{s}__noseason.json")

    base = defaultdict(list)
    for f in glob.glob(str(RESULTS / f"encoder__{DS}__seed*.json")):
        if "smoke" in f or "noseason" in f:
            continue
        for r in json.load(open(f)):
            base[(r["horizon"], r["metric"])].append(r[FIELD])
    if not base:
        print(f"\nWARNING: no baseline found in {RESULTS} -- run `train.loop --all` first.")
        return

    print(f"\n{'=' * 80}\nJAPAN  seasonal-phase ablation (zero sin_doy/cos_doy)  field={FIELD}\n{'=' * 80}")
    for metric in POINT:
        better = "higher" if metric == "pcc" else "lower"
        print(f"\n  {metric.upper()} ({better}=better)")
        print(f"    {'h':>3} | {'baseline':>17} | {'no-season':>17} | delta (abl - base)")
        for h in HZ:
            bm, bs = mean_sd(base[(h, metric)])
            am, asd = mean_sd(abl[(h, metric)])
            d, rel = am - bm, ((am - bm) / bm * 100 if bm else float("nan"))
            print(f"    {h:>3} | {bm:8.3f} ±{bs:6.3f} | {am:8.3f} ±{asd:6.3f} | {d:+9.3f} ({rel:+6.1f}%)")
    print("\nRead: |delta| within ~1 seed-sd => phase unused (capacity/training). "
          "A clear worsening => phase used, gap is the amplitude anchor.")


if __name__ == "__main__":
    main()

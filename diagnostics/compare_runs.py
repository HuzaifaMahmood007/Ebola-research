"""Compare every transfer run on record against its single-disease ceiling and naive floor.

    python compare_runs.py [--metric rmse] [-o Transfer_Comparison.md]

Reads results/{single,naive,lodo,joint}/*.json, groups by (regime, dataset, horizon), and reports
the 5-seed mean, the paired per-seed delta against the ceiling, and a paired t-test. Paired matters:
transfer and ceiling share seeds, so the per-seed difference cancels most of the 8-14% seed noise
that swamps an unpaired comparison.

Headline field follows train.lodo.HEADLINE (country_macro for dengue, node_mean elsewhere).
Lower is better for every metric reported here, so a NEGATIVE delta% means transfer BEAT the ceiling.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

RESULTS = Path("results")
HEADLINE = {"dengue": "country_macro"}
REGIMES = (("encoder_pair_zeroshot__", "PAIR zero-shot"),
           ("encoder_pair__", "PAIR adapted"),
           ("encoder_ldo3_zeroshot__", "LDO3 zero-shot"),
           ("encoder_ldo3__", "LDO3 adapted"),
           ("encoder_ldo_zeroshot__", "LDO zero-shot"),
           ("encoder_lodo_zeroshot__", "LODO zero-shot"),
           ("encoder_ldo__", "LDO adapted"),
           ("encoder_lodo__", "LODO adapted"),
           ("encoder_joint__", "joint"),
           ("encoder__", "single"))


def _field(ds):
    return HEADLINE.get(ds, "node_mean")


def regime_of(fname, rec):
    for prefix, label in REGIMES:
        if fname.startswith(prefix):
            if label == "joint":
                label = f"joint:{rec.get('sampler') or '?'}"
            if str(rec.get("model", "")).startswith("encoder_mc"):
                label += " [mean-corr]"
            return label
    return None


def load(metric):
    """{(regime, dataset, horizon): {seed: value}} for one metric, plus naive floors."""
    out = collections.defaultdict(dict)
    floors = collections.defaultdict(dict)
    for fam in ("single", "joint", "lodo", "naive"):
        d = RESULTS / fam
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            if "smoke" in p.name:
                continue
            for r in json.loads(p.read_text()):
                if r["metric"] != metric:
                    continue
                v = r[_field(r["dataset"])]
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                if fam == "naive":
                    floors[(r["model"], r["dataset"], r["horizon"])][r.get("seed")] = v
                    continue
                reg = regime_of(p.name, r)
                if reg:
                    out[(reg, r["dataset"], r["horizon"])][r["seed"]] = v
    return out, floors


def _mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _t_p(diffs):
    """Two-sided paired t-test p-value, normal approx beyond the small-n table."""
    n = len(diffs)
    if n < 2:
        return None
    m, sd = _mean_sd(diffs)
    if sd == 0:
        return 0.0 if m != 0 else 1.0
    t = abs(m / (sd / math.sqrt(n)))
    df = n - 1
    # Student-t survival via the incomplete beta, series-free: use the standard normal tail with a
    # small-sample correction table for df<=8, which covers every cell here (n=5 -> df=4).
    crit = {1: (12.71, 63.66), 2: (4.303, 9.925), 3: (3.182, 5.841), 4: (2.776, 4.604),
            5: (2.571, 4.032), 6: (2.447, 3.707), 7: (2.365, 3.499), 8: (2.306, 3.355)}
    c05, c01 = crit.get(df, (1.96, 2.576))
    return 0.01 if t >= c01 else (0.05 if t >= c05 else 1.0)


def rows(data, floors, metric):
    ceilings = {(ds, h): v for (reg, ds, h), v in data.items() if reg == "single"}
    out = []
    for (reg, ds, h), seeds in sorted(data.items()):
        if reg.startswith("single"):
            continue
        ceil = ceilings.get((ds, h))
        vals = list(seeds.values())
        m, sd = _mean_sd(vals)
        d_mean = d_pct = p = None
        shared = sorted(set(seeds) & set(ceil)) if ceil else []
        if len(shared) >= 2:
            diffs = [seeds[s] - ceil[s] for s in shared]
            d_mean, _ = _mean_sd(diffs)
            cm, _ = _mean_sd([ceil[s] for s in shared])
            d_pct = 100.0 * d_mean / cm if cm else None
            p = _t_p(diffs)
        best_floor = None
        for (fm, fds, fh), fv in floors.items():
            if fds == ds and fh == h:
                v = list(fv.values())[0]
                if best_floor is None or v < best_floor[1]:
                    best_floor = (fm, v)
        out.append(dict(regime=reg, dataset=ds, horizon=h, n=len(vals), mean=m, sd=sd,
                        ceiling=(_mean_sd([ceil[s] for s in shared])[0] if shared else None),
                        d_pct=d_pct, p=p, floor=best_floor, n_paired=len(shared)))
    return out


def fmt(v, w=9, d=1):
    return f"{v:>{w},.{d}f}" if v is not None else f"{'--':>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="rmse")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    data, floors = load(a.metric)
    rs = rows(data, floors, a.metric)

    L = []
    A = L.append
    A(f"# Transfer comparison ({a.metric.upper()}, lower is better)\n")
    A("Paired against the single-disease ceiling on shared seeds. **Negative delta% = transfer BEAT "
      "the ceiling.** `p` is a two-sided paired t-test over seeds.\n")

    by_reg = collections.defaultdict(list)
    for r in rs:
        by_reg[r["regime"]].append(r)

    for reg in sorted(by_reg, key=lambda x: (("PAIR" not in x), x)):
        A(f"\n## {reg}\n")
        A("| dataset | h | n | transfer | ceiling | delta% | p | best naive floor |")
        A("|---|--:|--:|--:|--:|--:|:--:|--:|")
        for r in sorted(by_reg[reg], key=lambda r: (r["dataset"], r["horizon"])):
            fl = f"{r['floor'][1]:,.1f} ({r['floor'][0]})" if r["floor"] else "--"
            sig = "**sig**" if (r["p"] is not None and r["p"] <= 0.05) else ("ns" if r["p"] is not None else "--")
            dp = f"{r['d_pct']:+.1f}%" if r["d_pct"] is not None else "--"
            A(f"| {r['dataset']} | {r['horizon']} | {r['n']} | {r['mean']:,.1f} ± {r['sd']:,.1f} "
              f"| {fmt(r['ceiling'],0)} | {dp} | {sig} | {fl} |")

    # headline roll-up
    A("\n## Roll-up: cells better than ceiling, by regime\n")
    A("| regime | cells | beat ceiling | mean delta% | median delta% | significant |")
    A("|---|--:|--:|--:|--:|--:|")
    for reg in sorted(by_reg):
        rr = [r for r in by_reg[reg] if r["d_pct"] is not None]
        if not rr:
            continue
        ds = sorted(r["d_pct"] for r in rr)
        beat = sum(1 for r in rr if r["d_pct"] < 0)
        sig = sum(1 for r in rr if r["p"] is not None and r["p"] <= 0.05)
        med = ds[len(ds)//2] if len(ds) % 2 else (ds[len(ds)//2-1]+ds[len(ds)//2])/2
        A(f"| {reg} | {len(rr)} | **{beat}/{len(rr)}** | {sum(ds)/len(ds):+.1f}% | {med:+.1f}% | {sig} |")

    txt = "\n".join(L)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

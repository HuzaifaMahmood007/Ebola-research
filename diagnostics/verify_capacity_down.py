"""verify_capacity_down.py -- read the numbers back OUT of progress/decisions/Capacity_Down_Protocol.md
and recompute every one from disk, the formulas, and an independent seeded Monte Carlo.

Deliberately does NOT import diagnostics/capacity_probe.py. The runner's constants (R_STAR, S2_LR,
S2_STEPS) are read from its SOURCE TEXT and compared with the protocol, so a bug shared with the runner
cannot verify itself.

What is checked:
  1. The budget table, all 8 Ebola rows and the dev row: pairs, districts, kappa, both bounds, the
     feature-acting budget and the over-budget ratios, from configs/ebola_arms.json,
     results/misc/ebolashift__adapter_mechanism.json and the three flu bundles.
  2. Parameter counts: rank-r* = 84 r* + 20, head-only 1,300, affine 1,428, S1 4, S2 5 and 8 from the
     budget rule applied to the frozen config, and the section 5 ranges.
  3. The Step 0 table and r*, from results/misc/capacity_probe_rank.json and the runner's R_STAR.
  4. Family counts: 28 + 42 = 70 deciding cells, 2 + 6 = 8 verdicts, with the deciding horizons
     derived from which horizons carry pairs.
  5. The luck-alone rates, from this file's own Monte Carlo (numpy, n = 5, t = 2.776), each within its
     printed precision plus 3 Monte Carlo standard errors, and the Monte Carlo itself against the two
     values known exactly (0.05 per cell, 0.025 squared for independent panels).
  6. The archived upward half: panel correlations, 7 / 3 / 36, best gain, the closest lower bound, the
     all-MIXED re-read under this file's own copy of the regime-F rule, and the runtime basis; and the
     fewshot_sim covid figures behind correction 8.
  7. Support geometry: support origins, first validation and test window columns, the L20 total budget.
Not recomputed: the single-disease participation ratios the brief quoted (1.8 to 2.5, 4.6), the
Adapter_Constraint figures quoted in the corrections, and the pre-run S2 step check on stand-in anchors
(a scratch run that left no artifact). They are cited, not derived.

    python -m diagnostics.verify_capacity_down            # check, then corrupt doc and inputs; all caught
    python -m diagnostics.verify_capacity_down --result   # after the run: the result doc vs the JSON
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DOC = ROOT / "progress" / "decisions" / "Capacity_Down_Protocol.md"
RUNNER = ROOT / "diagnostics" / "capacity_probe.py"
CONFIG = ROOT / "configs" / "ebola_arms.json"
MECH = ROOT / "results" / "misc" / "ebolashift__adapter_mechanism.json"
RANK = ROOT / "results" / "misc" / "capacity_probe_rank.json"
ARCH = ROOT / "results" / "misc" / "capacity_probe_5seed.json"
RESULT_JSON = ROOT / "results" / "misc" / "capacity_probe_down.json"
SIM = ROOT / "results" / "misc" / "fewshotsim__summary__seed42.json"
HORIZONS, NQ, D = (3, 5, 10, 15), 5, 64
T5 = 2.776
CONTROL = "affine (current)"
FLU = ("influenza_japan", "influenza_us-regions", "influenza_us-states")
S_PANELS = ("influenza_japan", "influenza_us-states")
MC_N, MC_SEED = 2_000_000, 20260929
R_JS, R_US = 0.29, 0.90


# --------------------------------------------------------------------------- #
# Disk side
# --------------------------------------------------------------------------- #
def dev_fold():
    """The flu full fold and the regime-S window geometry, from the bundles."""
    import bundles
    from bundles import W
    pairs, nodes, node_seasons, weeks, val0, test0 = defaultdict(int), 0, 0.0, 0.0, {}, {}
    for n in FLU:
        b = bundles.load(n)
        M, tr = b.M.astype(bool), b.masks()["train"].astype(bool)
        orig = b.origins(phase="train")
        for h in HORIZONS:
            pairs[h] += int(sum((M[:, t + h] & tr[:, t + h]).sum() for t in orig))
        nodes += M.shape[0]
        node_seasons += M.shape[0] * len(orig) / 52
        weeks += M.shape[0] * int(tr.any(0).sum()) / 52
        val0[n] = min(b.origins(phase="val")) - (W - 1)          # first column any val window reads
        test0[n] = min(b.origins(phase="test")) - (W - 1)
    return dict(pairs=max(pairs.values()), nodes=nodes, node_seasons=node_seasons, by_weeks=weeks,
                val0=val0, test0=test0)


def upward(arch):
    """The archived upward half, recomputed with this file's own interval and regime-F rule."""
    deltas = defaultdict(list)
    for arm in ("cross_disease", "in_domain_control"):
        for rows in arch[arm]:
            base = next(r for r in rows if r["label"] == CONTROL)["rmse"]
            for r in rows:
                if r["label"] != CONTROL:
                    for c, v in r["rmse"].items():
                        deltas[(arm, r["label"], c)].append((base[c] - v) / abs(base[c]) * 100)

    def ci(xs):
        x = np.asarray(xs)
        half = T5 * x.std(ddof=1) / math.sqrt(len(x))
        return x.mean(), x.mean() - half, x.mean() + half

    cross = {(l, c): ci(x) for (a, l, c), x in deltas.items() if a == "cross_disease"}
    better = [(l, c, m, lo) for (l, c), (m, lo, hi) in cross.items() if lo > 0]
    worse = [(l, c) for (l, c), (m, lo, hi) in cross.items() if hi < 0]
    labels = sorted({l for l, _ in cross})
    verdicts = {}
    for l in labels:
        s = {c: (1 if lo > 0 else -1 if hi < 0 else 0) for (ll, c), (m, lo, hi) in cross.items() if ll == l}
        wins = [h for h in HORIZONS if s[f"influenza_japan|h{h}"] == 1
                and (s[f"influenza_us-regions|h{h}"] == 1 or s[f"influenza_us-states|h{h}"] == 1)]
        harm = -1 in s.values()
        verdicts[l] = "MIXED" if wins and harm else "WIN" if wins else "COSTS" if harm else "NULL"
    rjs, rus = [], []
    for l in labels:
        for h in HORIZONS:
            g = lambda p: deltas[("cross_disease", l, f"{p}|h{h}")]
            rjs.append(np.corrcoef(g("influenza_japan"), g("influenza_us-states"))[0, 1])
            rus.append(np.corrcoef(g("influenza_us-regions"), g("influenza_us-states"))[0, 1])
    mins = {arm: [r["minutes"] for rows in arch[arm] for r in rows] for arm in ("cross_disease", "in_domain_control")}
    params = max(r["params"] for rows in arch["cross_disease"] for r in rows)
    return dict(cells=len(cross), better=len(better), worse=len(worse),
                better_h15=all(c.endswith("|h15") for _, c, _, _ in better),
                worse_jh3=all(c == "influenza_japan|h3" for _, c in worse),
                best=max(m for *_, m, _ in better), closest_lo=min(lo for *_, lo in better),
                r_js=float(np.median(rjs)), r_us=float(np.median(rus)), verdicts=verdicts,
                xmin=min(mins["cross_disease"]), xmax=max(mins["cross_disease"]),
                xmean=float(np.mean(mins["cross_disease"])), imin=min(mins["in_domain_control"]),
                imax=max(mins["in_domain_control"]), imean=float(np.mean(mins["in_domain_control"])),
                top_params=params)


def runner_constants(src):
    g = lambda pat: re.search(pat, src, re.M)
    lr, steps = g(r"^S2_LR, S2_STEPS = ([\d.e-]+), (\d+)").groups()
    return dict(r_star=int(g(r"^R_STAR = (\d+)").group(1)), s2_lr=float(lr), s2_steps=int(steps),
                conv=bool(g(r"abs\(at_2n - at_n\) <= 1e-4 \* at_n")))


def disk(ov=None, dev=None):
    """Everything the protocol's numbers come from. `ov` injects corruptions for the mutation test."""
    ov = ov or {}
    cfg = json.loads(CONFIG.read_text())
    mech = json.loads(MECH.read_text())["cells"]
    rank = json.loads(RANK.read_text())
    arch = json.loads(ARCH.read_text())
    sim = json.loads(SIM.read_text())
    src = RUNNER.read_text(encoding="utf-8")
    for key, fn in ov.items():                               # each override mutates its own input
        if key == "cfg":
            fn(cfg)
        elif key == "mech":
            fn(mech)
        elif key == "rank":
            fn(rank)
        elif key == "arch":
            fn(arch)
        elif key == "src":
            src = fn(src)
        elif key == "sim":
            fn(sim)
    arms = {}
    for a in ("L12", "L20"):
        c = cfg["arms"][f"ebola_{a}"]
        arms[a] = dict(pairs={int(h): v for h, v in c["counts"]["adapt_pairs"].items()},
                       districts={int(h): v for h, v in c["counts"]["adapt_districts"].items()},
                       origins=c["support_columns"] - min(HORIZONS), block_end=c["support_columns"] - 1)
    ks = defaultdict(list)
    for c in mech:
        if c["extrap_median"] == c["extrap_median"]:          # nan where there is no support (L12 h15)
            ks[(c["arm"][-3:], c["h"])].append(c["extrap_median"])
    kappa = {k: float(np.mean(v)) for k, v in ks.items()}
    pr = {p: [v for _, v in sorted(s.items())] for p, s in rank["pr"].items()}
    mae = {(r["template"], r["placement"], r["arm"]): r["mae"] for r in sim if r["bundle"] == "covid_us-states"}
    gain = {t: 100 * (mae[(t, "t0-blind", "zeroshot")] - mae[(t, "t0-blind", "fresh")]) / mae[(t, "t0-blind", "zeroshot")]
            for t in ("L12", "L20")}
    mid_hurts = all(mae[(t, "mid-blind", "fresh")] > mae[(t, "mid-blind", "zeroshot")] for t in ("L12", "L20"))
    return dict(arms=arms, kappa=kappa, pr=pr, rank_json_r=rank["r_star"], runner=runner_constants(src),
                sim_gain=gain, sim_mid_hurts=mid_hurts,
                up=upward(arch), dev=dev or dev_fold())


def levels(pairs):
    """The protocol's budget rule, written again: 2 slope+intercept, 1 intercept, 0 nothing."""
    return {h: 2 if n >= 20 else 1 if n >= 10 else 0 for h, n in pairs.items()}


def monte_carlo(n=MC_N, seed=MC_SEED):
    """Luck alone under the agreement rule. Five normal per-seed deltas per cell, correlated between
    panels, the 95% t rule at n = 5. Per-horizon probabilities, horizons combined as independent."""
    rng = np.random.default_rng(seed)

    def signs(C, chunk=400_000):
        L = np.linalg.cholesky(np.asarray(C, dtype=float))
        out = []
        for s in range(0, n, chunk):
            X = rng.standard_normal((min(chunk, n - s), 5, len(C))) @ L.T
            m, half = X.mean(1), T5 * X.std(1, ddof=1) / math.sqrt(5)
            out.append(((m - half > 0).astype(np.int8) - (m + half < 0).astype(np.int8)))
        return np.concatenate(out)

    both = lambda r: float(((lambda s: (s[:, 0] == 1) & (s[:, 1] == 1))(signs([[1, r], [r, 1]]))).mean())
    cell = float((signs([[1.0]])[:, 0] != 0).mean())
    p0, pjs, pus = both(0.0), both(R_JS), both(R_US)
    s = signs([[1, R_JS, R_JS], [R_JS, 1, R_US], [R_JS, R_US, 1]])
    J, R, S = s[:, 0], s[:, 1], s[:, 2]
    qF = float(((J == 1) & ((R == 1) | (S == 1))).mean())
    qH = float((((J == 1) & ((R == 1) | (S == 1))) | ((J == -1) & ((R == -1) | (S == -1)))).mean())
    pS, pF, pH = 1 - (1 - pjs) ** 2, 1 - (1 - qF) ** 4, 1 - (1 - qH) ** 4
    se = lambda p: math.sqrt(p * (1 - p) / n)
    return dict(cell=cell, p0=p0, pjs=pjs, pus=pus, pS=pS, pF=pF, pH=pH,
                sweep=1 - (1 - pS) ** 6 * (1 - pF),
                se=dict(cell=se(cell), p0=se(p0), pjs=se(pjs), pus=se(pus), pS=2 * se(pjs),
                        pF=4 * se(qF), pH=4 * se(qH), sweep=12 * se(pjs) + 4 * se(qF)))


# --------------------------------------------------------------------------- #
# Document side
# --------------------------------------------------------------------------- #
def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _num(p):
    return float(p.replace(",", ""))


def _eq(printed, value):
    """printed string equals value rounded to the printed precision."""
    p = printed.replace(",", "")
    dec = len(p.split(".")[1]) if "." in p else 0
    return abs(float(p) - value) <= 0.5 * 10 ** (-dec) + 1e-9


def _near(printed, value, se):
    """Monte Carlo figure: within the printed precision plus 3 standard errors."""
    p = printed.replace(",", "")
    dec = len(p.split(".")[1]) if "." in p else 0
    return abs(float(p) - value) <= 0.5 * 10 ** (-dec) + 3 * se


def check(text, d, mc, verbose=True):
    fails, n = [], [0]

    def ok(cond, msg):
        n[0] += 1
        if not cond:
            fails.append(msg)

    flat = re.sub(r"\s+", " ", text)

    def grab(pat, k=None):
        m = re.search(pat, flat)
        ok(m is not None, f"prose not found: {pat}")
        if not m:
            return None
        g = tuple(x.rstrip(".") for x in m.groups())               # a number that ends a sentence
        return g if k is None else g[k - 1]

    ok("\u2014" not in text, "the protocol contains an em dash")
    A, up, dev, run = d["arms"], d["up"], d["dev"], d["runner"]

    # 1. budget table
    rows = [_cells(l) for l in text.splitlines() if re.match(r"\| L(12|20) \| \d+ \| ", l)]
    ok(len(rows) == 8, f"budget table has {len(rows)} Ebola rows, want 8")
    ratios = []
    for c in rows:
        arm, h = c[0], int(c[1])
        pairs, dist = A[arm]["pairs"][h], A[arm]["districts"][h]
        ok(int(c[2]) == pairs, f"{arm} h{h} pairs: doc {c[2]}, config {pairs}")
        ok(int(c[3]) == dist, f"{arm} h{h} districts: doc {c[3]}, config {dist}")
        ok(_eq(c[5], pairs / 10) and _eq(c[6], dist / 10), f"{arm} h{h} bounds: doc {c[5]}/{c[6]}")
        if pairs == 0:
            ok(c[4] == "n/a" and c[7] == "0" and c[8] == "no data at all", f"{arm} h{h} empty row: {c}")
            continue
        k = d["kappa"][(arm, h)]
        ok(_eq(c[4], k), f"{arm} h{h} kappa: doc {c[4]}, disk {k:.4f}")
        ok(_eq(c[7], pairs / (10 * k * k)), f"{arm} h{h} feature budget: doc {c[7]}, disk {pairs / (10 * k * k):.4f}")
        m = re.fullmatch(r"(\d+)x to (\d+)x", c[8])
        lo, hi = round(325 / (pairs / 10)), round(325 / (dist / 10))
        ok(m is not None and (int(m.group(1)), int(m.group(2))) == (lo, hi), f"{arm} h{h} ratio: doc {c[8]}, disk {lo}x to {hi}x")
        ratios += [lo, hi]
    m = grab(r"\| dev full fold \(flu, 3 panels\) \| any \| up to ([\d,]+) \| (\d+) nodes, about (\d+) node-seasons "
             r"\| not measured \| up to ([\d,]+) \| about (\d+) \| not measured \| affine sits inside \[(\d+), ([\d,]+)\] \|")
    if m:
        ok(_num(m[0]) == dev["pairs"], f"dev pairs: doc {m[0]}, disk {dev['pairs']}")
        ok(int(m[1]) == dev["nodes"], f"dev nodes: doc {m[1]}, disk {dev['nodes']}")
        ok(int(m[2]) == round(dev["node_seasons"]), f"node-seasons: doc {m[2]}, disk {dev['node_seasons']:.1f}")
        ok(_num(m[3]) == round(dev["pairs"] / 10) == _num(m[6]), f"dev pairs bound: doc {m[3]}/{m[6]}")
        ok(int(m[4]) == round(dev["node_seasons"] / 10) == int(m[5]), f"dev district bound: doc {m[4]}/{m[5]}")
    ks = [v for v in d["kappa"].values()]
    lo_k, hi_k = grab(r"spread ratio kappa at ([\d.]+) to ([\d.]+)") or ("0", "0")
    ok(_eq(lo_k, min(ks)) and _eq(hi_k, max(ks)), f"kappa range: doc {lo_k} to {hi_k}, disk {min(ks):.2f} to {max(ks):.2f}")
    g = grab(r"The affine is (\d+)x to (\d+)x over budget")
    ok(g is not None and ratios and (int(g[0]), int(g[1])) == (min(ratios), max(ratios)), f"overall ratio {g}")
    g = grab(r"B / 325 = ([\d.]+) \(L12 h3\) to ([\d.]+) \(L20 h3\)")
    ok(g is not None and _eq(g[0], A["L12"]["pairs"][3] / 10 / 325) and _eq(g[1], A["L20"]["pairs"][3] / 10 / 325),
       f"shrinkage prediction {g}")
    g = grab(r"The shipped map spends (\d+) x (\d+) = (\d+) parameters per horizon")
    ok(g is not None and (int(g[0]), int(g[1]), int(g[2])) == (NQ, D + 1, NQ * (D + 1)), f"per-horizon count {g}")
    ok(re.search(r"the ebola support-mask geometry is read, no labels", flat, re.I) is not None,
       "the no-labels sentence is missing")

    # 2. parameter counts
    r = run["r_star"]
    lvl = {a: levels(A[a]["pairs"]) for a in A}
    s2 = {a: sum(lvl[a].values()) for a in A}
    word = {2: "slope+int", 1: "int", 0: "none"}
    g = grab(r"\| `rank-(\d+)` \| (\d+) = 84 r\* \+ 20 at r\* = (\d+) \|")
    ok(g is not None and int(g[0]) == int(g[2]) == r and int(g[1]) == 84 * r + 20, f"rank rung {g}, runner r* {r}")
    g = grab(r"\| `head-only` \| ([\d,]+) \|")
    ok(g is not None and _num(g[0]) == D * 4 * NQ + 4 * NQ, f"head-only {g}")
    g = grab(r"\| control `affine` \| ([\d,]+) \|")
    ok(g is not None and _num(g[0]) == D * 4 * NQ + 4 * NQ + 2 * D, f"affine {g}")
    g = grab(r"\| S1 `recal-int` \| (\d+) nominal")
    ok(g is not None and int(g[0]) == len(HORIZONS), f"S1 {g}")
    desc = ", ".join(f"h{h} {word[lvl['L12'][h]]}" for h in HORIZONS)
    g = grab(r"\| S2 `recal-budget` \| L12: (\d+) \(([^)]*)\); L20: (\d+) \|")
    ok(g is not None and int(g[0]) == s2["L12"] and g[1] == desc and int(g[2]) == s2["L20"],
       f"S2 {g}, disk L12 {s2['L12']} ({desc}), L20 {s2['L20']}")
    g = grab(r"the fresh affine \(([\d,]+)\) shrunk toward the anchor by one factor t, effective size t x ([\d,]+)")
    ok(g is not None and _num(g[0]) == 1428 and _num(g[1]) == 1300, f"S3 {g}")
    g = grab(r"only the (\d+) to (\d+) parameters that support can fund")
    ok(g is not None and (int(g[0]), int(g[1])) == (len(HORIZONS), max(s2.values())), f"section 5 range {g}")
    g = grab(r"from rank-(\d+) \((\d+) parameters\) to ([\d,]+)")
    ok(g is not None and int(g[0]) == r and int(g[1]) == 84 * r + 20 and _num(g[2]) == up["top_params"], f"F range {g}")
    g = grab(r"No surface from (\d+) to ([\d,]+) parameters")
    ok(g is not None and int(g[0]) == len(HORIZONS) and _num(g[1]) == up["top_params"], f"combined range {g}")
    g = grab(r"restricted to the (\d+) directions")
    ok(g is not None and int(g[0]) == r, f"WIN-TRANSFER sentence r {g}")
    g = grab(r"Rank-1 has (\d+) parameters, more than (\d+)x the largest total Ebola budget \(L20, pairs bound, (\d+)\)")
    tot = sum(A["L20"]["pairs"].values()) / 10
    ok(g is not None and int(g[0]) == 104 and int(g[2]) == round(tot) and 104 / tot > int(g[1]), f"rank-1 budget {g}")
    g = grab(r"full-batch Adam at a constant lr of ([\d.e-]+) for ([\d,]+) steps")
    ok(g is not None and float(g[0]) == run["s2_lr"] and _num(g[1]) == run["s2_steps"],
       f"S2 optimiser: doc {g}, runner lr {run['s2_lr']}, steps {run['s2_steps']}")
    ok("less than 1e-4 relative" in flat and run["conv"], "S2 convergence limit disagrees with the runner")

    # 3. Step 0
    tab = {c[0]: c[1:] for c in (_cells(l) for l in text.splitlines()) if c[0] in d["pr"] and len(c) == 7}
    ok(len(tab) == 4, f"Step 0 table has {len(tab)} rows, want 4")
    for p, cells in tab.items():
        v = d["pr"][p]
        ok(all(_eq(x, y) for x, y in zip(cells[:5], v)) and _eq(cells[5], max(v)), f"Step 0 {p}: doc {cells}")
    allpr = [x for v in d["pr"].values() for x in v]
    flu = max(x for p, v in d["pr"].items() if p != "dengue" for x in v)
    g = grab(r"r\* = ceil\(([\d.]+)\) = (\d+)")
    ok(g is not None and _eq(g[0], max(allpr)) and int(g[1]) == math.ceil(max(allpr)) == r == d["rank_json_r"],
       f"r*: doc {g}, disk ceil({max(allpr):.3f}), runner {r}, rank json {d['rank_json_r']}")
    g = grab(r"so the rank rung has 84 x (\d+) \+ 20 = (\d+) parameters")
    ok(g is not None and int(g[0]) == r and int(g[1]) == 84 * r + 20, f"rank count {g}")
    g = grab(r"On the three flu panels the largest ratio is ([\d.]+)")
    ok(g is not None and _eq(g[0], flu), f"flu max {g}, disk {flu:.3f}")
    g = grab(r"A flu-only rule would give r = (\d+) \((\d+) parameters\)")
    ok(g is not None and int(g[0]) == math.ceil(flu) and int(g[1]) == 84 * math.ceil(flu) + 20, f"flu-only r {g}")
    g = grab(r"dengue itself reaches ([\d.]+)")
    ok(g is not None and _eq(g[0], max(d["pr"]["dengue"])), f"dengue max {g}")
    g = grab(r"so r\* = (\d+) and the rung has (\d+) parameters")
    ok(g is not None and int(g[0]) == r and int(g[1]) == 84 * r + 20, f"correction 11 {g}")

    # 4. family counts, deciding horizons derived from which horizons carry pairs
    dec = {a: sum(1 for h in HORIZONS if A[a]["pairs"][h] > 0) for a in A}
    fam_f = (12 + 4) + 12
    fam_s = 3 * (len(S_PANELS) * dec["L12"] + len(S_PANELS) * dec["L20"])
    g = grab(r"rank-r\* (\d+) cells plus head-only (\d+) cells = (\d+) cells, (\d+) verdicts")
    ok(g is not None and tuple(map(int, g)) == (16, 12, fam_f, 2), f"regime F family {g}")
    g = grab(r"(\d+) surfaces x \((\d+) L12 \+ (\d+) L20\) = (\d+) cells, (\d+) verdicts")
    ok(g is not None and tuple(map(int, g)) == (3, 2 * dec["L12"], 2 * dec["L20"], fam_s, 3 * 2), f"regime S family {g}")
    g = grab(r"Total (\d+) deciding cells, (\d+) verdicts")
    ok(g is not None and (int(g[0]), int(g[1])) == (fam_f + fam_s, 8), f"total family {g}")

    # 5. luck alone
    g = grab(r"Monte Carlo, ([\d,]+) draws per scenario, numpy seed (\d+)")
    ok(g is not None and _num(g[0]) == MC_N and int(g[1]) == MC_SEED, f"Monte Carlo settings {g}")
    ok(abs(mc["cell"] - 0.05) < 4 * mc["se"]["cell"], f"the Monte Carlo itself: per-cell rate {mc['cell']:.5f} is not 0.05")
    ok(abs(mc["p0"] - 0.025 ** 2) < 4 * mc["se"]["p0"], f"the Monte Carlo itself: independent {mc['p0']:.6f} is not 0.000625")
    tot = fam_f + fam_s
    g = grab(r"give about ([\d.]+) falsely significant cells, about ([\d.]+) each way")
    ok(g is not None and _eq(g[0], 0.05 * tot) and _eq(g[1], 0.025 * tot), f"false cells {g}")
    g = grab(r"is ([\d.]+) at the measured r = ([\d.]+), against ([\d.]+) if independent and ([\d.]+) at r = ([\d.]+)")
    if g:
        ok(_near(g[0], mc["pjs"], mc["se"]["pjs"]), f"both clear at r=0.29: doc {g[0]}, MC {mc['pjs']:.5f}")
        ok(_near(g[2], mc["p0"], mc["se"]["p0"]), f"both clear independent: doc {g[2]}, MC {mc['p0']:.5f}")
        ok(_near(g[3], mc["pus"], mc["se"]["pus"]), f"both clear at r=0.90: doc {g[3]}, MC {mc['pus']:.5f}")
        ok(float(g[1]) == R_JS and float(g[4]) == R_US, f"MC correlations {g[1]}, {g[4]}")
    g = grab(r"falsely WINs with probability about ([\d.]+), and rank-r\* about ([\d.]+)")
    ok(g is not None and _near(g[0], mc["pS"], mc["se"]["pS"]) and _near(g[1], mc["pF"], mc["se"]["pF"]),
       f"verdict false-WIN {g}, MC {mc['pS']:.4f} / {mc['pF']:.4f}")
    g = grab(r"Whole sweep: about ([\d.]+), roughly a 1 in (\d+) chance")
    ok(g is not None and _near(g[0], mc["sweep"], mc["se"]["sweep"]) and abs(int(g[1]) - 1 / mc["sweep"]) <= 1,
       f"whole sweep {g}, MC {mc['sweep']:.4f} (1 in {1 / mc['sweep']:.1f})")
    g = grab(r"DIFFERS has its own false rate of about ([\d.]+)")
    ok(g is not None and _near(g[0], mc["pH"], mc["se"]["pH"]), f"head-only DIFFERS {g}, MC {mc['pH']:.4f}")
    g = grab(r"0\.025 squared, ([\d.]+)")
    ok(g is not None and _eq(g[0], 0.025 ** 2), f"exact independent value {g}")

    # 6. archived upward half
    g = grab(r"japan vs us-states correlate at median r = ([\d.]+), and the two US panels at ([\d.]+)")
    ok(g is not None and _eq(g[0], up["r_js"]) and _eq(g[1], up["r_us"]) and float(g[0]) == R_JS and float(g[1]) == R_US,
       f"correlations {g}, archive {up['r_js']:.3f} / {up['r_us']:.3f}")
    g = grab(r"The per-cell counts are (\d+) better, (\d+) worse, (\d+) cells, all (\d+) at h15, all (\d+) at japan h3, "
             r"best \+([\d.]+) per cent")
    ok(g is not None and (int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4])) ==
       (up["better"], up["worse"], up["cells"], up["better"], up["worse"]) and up["better_h15"] and up["worse_jh3"]
       and _eq(g[5], up["best"]), f"upward counts {g}, archive {up['better']}/{up['worse']}/{up['cells']} best {up['best']:.2f}")
    g = grab(r"with a lower bound of \+([\d.]+)")
    ok(g is not None and _eq(g[0], up["closest_lo"]), f"closest lower bound {g}, archive {up['closest_lo']:.3f}")
    ok("so all three read MIXED" in flat and set(up["verdicts"].values()) == {"MIXED"} and len(up["verdicts"]) == 3,
       f"upward re-read {up['verdicts']}")
    g = grab(r"Regime F is 3 x \(([\d.]+) \+ ([\d.]+)\) = about (\d+) min per seed, about ([\d.]+) h for five")
    per_seed = 3 * (up["xmean"] + up["imean"])
    ok(g is not None and _eq(g[0], up["xmean"]) and _eq(g[1], up["imean"]) and _eq(g[2], per_seed)
       and _eq(g[3], 5 * per_seed / 60), f"runtime {g}, archive {up['xmean']:.2f} + {up['imean']:.2f}")
    g = grab(r"cross ([\d.]+) to ([\d.]+) min, in-domain ([\d.]+) to ([\d.]+) min per surface")
    ok(g is not None and [float(x) for x in g] == [up["xmin"], up["xmax"], up["imin"], up["imax"]], f"runtime range {g}")
    g = grab(r"averages ([\d.]+) min per surface, not 4\.3")
    ok(g is not None and _eq(g[0], up["xmean"]), f"correction 14 {g}")
    g = grab(r"adapting \*helped\* by (\d+) per cent on L12 and (\d+) per cent on L20")
    ok(g is not None and _eq(g[0], d["sim_gain"]["L12"]) and _eq(g[1], d["sim_gain"]["L20"]) and d["sim_mid_hurts"],
       f"fewshot_sim covid gains {g}, disk {d['sim_gain']}, mid-blind hurts {d['sim_mid_hurts']}")
    g = grab(r"Counted by train weeks instead it would be (\d+)")
    ok(g is not None and int(g[0]) == round(dev["by_weeks"]), f"node-seasons by weeks {g}, disk {dev['by_weeks']:.1f}")

    # 7. support geometry
    g = grab(r"with (\d+) support origins \(L12\) or (\d+) \(L20")
    ok(g is not None and (int(g[0]), int(g[1])) == (A["L12"]["origins"], A["L20"]["origins"]), f"support origins {g}")
    g = grab(r"first test window starts at column (\d+) on japan and (\d+) on us-states, and the first validation "
             r"window at column (\d+) on japan and (\d+) on us-states, against a block ending at column (\d+)")
    ok(g is not None and tuple(map(int, g)) == (dev["test0"]["influenza_japan"], dev["test0"]["influenza_us-states"],
                                                dev["val0"]["influenza_japan"], dev["val0"]["influenza_us-states"],
                                                A["L20"]["block_end"]), f"window columns {g}, disk {dev['test0']} {dev['val0']}")
    ok(all(dev[k][p] > A["L20"]["block_end"] for k in ("test0", "val0") for p in S_PANELS), "a window reaches the block")

    if verbose:
        print(f"{'OK' if not fails else 'FAIL'}: {n[0]} checks, {len(fails)} failures")
        for f in fails:
            print("  " + f)
    return fails


def mutate(text, d, mc):
    """Every corruption must be caught: edits to the document and breaks in the inputs. A corruption
    counts as caught only if it adds a failure the clean run does not have, so a failing baseline
    cannot make every mutation look caught."""
    base = set(check(text, d, mc, verbose=False))
    doc_mut = [
        ("budget pairs", "| L12 | 3 | 48 | 17 |", "| L12 | 3 | 46 | 17 |"),
        ("district bound", "| 5.42 | 4.8 | 1.7 | 0.16 |", "| 5.42 | 4.8 | 1.9 | 0.16 |"),
        ("feature budget", "| 11.15 | 5.4 | 3.4 | 0.04 |", "| 11.15 | 5.4 | 3.4 | 0.40 |"),
        ("over-budget ratio", "181x to 361x", "181x to 316x"),
        ("dev pairs", "up to 18,586", "up to 18,568"),
        ("Step 0 cell", "| dengue | 9.017 |", "| dengue | 9.071 |"),
        ("r*", "r* = ceil(9.017) = 10", "r* = ceil(9.017) = 9"),
        ("rank params", "| 860 = 84 r* + 20", "| 680 = 84 r* + 20"),
        ("head-only params", "| `head-only` | 1,300 |", "| `head-only` | 1,030 |"),
        ("S2 params", "; L20: 8 |", "; L20: 9 |"),
        ("S2 shape", "h10 int, h15 none", "h10 slope+int, h15 none"),
        ("total cells", "Total 70 deciding cells", "Total 72 deciding cells"),
        ("verdict count", "70 deciding cells, 8 verdicts", "70 deciding cells, 9 verdicts"),
        ("MC pair rate", "is 0.0016 at the measured", "is 0.0026 at the measured"),
        ("MC whole sweep", "Whole sweep: about 0.028", "Whole sweep: about 0.018"),
        ("correlation", "median r = 0.29", "median r = 0.39"),
        ("upward counts", "are 7 better, 3 worse", "are 8 better, 3 worse"),
        ("closest bound", "lower bound of +0.03", "lower bound of +0.30"),
        ("runtime", "about 81 min per seed", "about 18 min per seed"),
        ("window column", "column 210 on japan", "column 201 on japan"),
        ("L20 total budget", "(L20, pairs bound, 32)", "(L20, pairs bound, 23)"),
        ("S2 steps", "constant lr of 1e-2 for 5,000 steps", "constant lr of 1e-2 for 2,000 steps"),
        ("em dash", "Scope limits, stated", "Scope limits \u2014 stated"),
    ]
    caught = 0
    for name, a, b in doc_mut:
        assert a in text, f"mutation anchor missing for '{name}': {a!r}"
        f = set(check(text.replace(a, b, 1), d, mc, verbose=False)) - base
        caught += bool(f)
        print(f"  {'caught' if f else 'MISSED'}: doc {name}")

    def seti(path, value):
        def fn(obj):
            *head, last = path
            for k in head:
                obj = obj[k]
            obj[last] = value(obj[last])
        return fn

    def arch_one(arch):                      # one seed's mlp-64 japan h3 becomes a 30% gain, not a loss
        rows = arch["cross_disease"][0]
        base = next(r for r in rows if r["label"] == CONTROL)["rmse"]["influenza_japan|h3"]
        next(r for r in rows if r["label"] == "mlp-64")["rmse"]["influenza_japan|h3"] = 0.7 * base

    inputs = [
        ("config L12 h3 pairs 48 -> 49", dict(cfg=seti(["arms", "ebola_L12", "counts", "adapt_pairs", "3"], lambda v: v + 1))),
        ("config L20 h10 districts 35 -> 34", dict(cfg=seti(["arms", "ebola_L20", "counts", "adapt_districts", "10"], lambda v: v - 1))),
        ("config L12 support columns 13 -> 14", dict(cfg=seti(["arms", "ebola_L12", "support_columns"], lambda v: v + 1))),
        ("kappa +10%", dict(mech=lambda cells: [c.update(extrap_median=c["extrap_median"] * 1.1) for c in cells])),
        ("rank json dengue seed 42 -> 10.2", dict(rank=seti(["pr", "dengue", "42"], lambda v: 10.2))),
        ("runner R_STAR = 9", dict(src=lambda s: re.sub(r"^R_STAR = \d+", "R_STAR = 9", s, flags=re.M))),
        ("runner S2_STEPS = 2000", dict(src=lambda s: re.sub(r"^S2_LR, S2_STEPS = ([\d.e-]+), \d+",
                                                              r"S2_LR, S2_STEPS = \1, 2000", s, flags=re.M))),
        ("runner S2_LR = 1e-3", dict(src=lambda s: re.sub(r"^S2_LR, S2_STEPS = [\d.e-]+,", "S2_LR, S2_STEPS = 1e-3,",
                                                          s, flags=re.M))),
        ("archive: one seed's mlp-64 japan h3", dict(arch=arch_one)),
        ("fewshot_sim covid L12 t0-blind fresh MAE +10%",
         dict(sim=lambda rows: [r.update(mae=r["mae"] * 1.1) for r in rows if (r["bundle"], r["template"], r["placement"],
                                r["arm"]) == ("covid_us-states", "L12", "t0-blind", "fresh")])),
    ]
    for name, ov in inputs:
        f = set(check(text, disk(ov, dev=d["dev"]), mc, verbose=False)) - base
        caught += bool(f)
        print(f"  {'caught' if f else 'MISSED'}: input {name}")
    total = len(doc_mut) + len(inputs)
    print(f"mutate: {caught} of {total} corruptions caught")
    return caught == total


def result_mode():
    """After the run: every verdict in capacity_probe_down.json must appear in the newest
    progress/outcomes/Capacity_Down_Result_*.md as a table row naming the surface, the scope and the
    verdict, the JSON must carry the committed protocol's hash, and the document must quote it."""
    fails = []
    ok = lambda cond, msg: None if cond else fails.append(msg)
    if not RESULT_JSON.exists():
        print(f"FAIL: {RESULT_JSON.relative_to(ROOT)} does not exist yet; run --down --report first")
        return 1
    res = json.loads(RESULT_JSON.read_text())
    docs = sorted((ROOT / "progress" / "outcomes").glob("Capacity_Down_Result_*.md"))
    if not docs:
        print("FAIL: no progress/outcomes/Capacity_Down_Result_*.md yet")
        return 1
    text = docs[-1].read_text(encoding="utf-8")
    rows = [[c.strip("`* ") for c in _cells(l)] for l in text.splitlines() if l.startswith("|")]
    sha = hashlib.sha256(DOC.read_bytes()).hexdigest()
    ok(res["protocol_sha256"] == sha, "the result JSON was made under a different protocol hash")
    ok(sha[:12] in text, "the result document does not quote the protocol hash")
    ok("\u2014" not in text, "the result document contains an em dash")
    for v in res["verdicts"]:
        hit = [r for r in rows if v["surface"] in r and v["scope"] in r]
        ok(hit and all(v["verdict"] in r for r in hit),
           f"{v['surface']} {v['scope']}: JSON says {v['verdict']}, document rows {hit}")
    print(f"{'OK' if not fails else 'FAIL'}: result {docs[-1].name}, {len(res['verdicts'])} verdicts, "
          f"{len(fails)} failures")
    for f in fails:
        print("  " + f)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", action="store_true", help="check the result document after the run")
    a = ap.parse_args()
    if a.result:
        return result_mode()
    text = DOC.read_text(encoding="utf-8")
    d, mc = disk(), monte_carlo()
    print(f"Monte Carlo: pair at r=0.29 {mc['pjs']:.5f}, independent {mc['p0']:.6f}, r=0.90 {mc['pus']:.5f}, "
          f"regime S {mc['pS']:.4f}, rank {mc['pF']:.4f}, head-only {mc['pH']:.4f}, sweep {mc['sweep']:.4f}")
    fails = check(text, d, mc)
    return 0 if (mutate(text, d, mc) and not fails) else 1


if __name__ == "__main__":
    sys.exit(main())

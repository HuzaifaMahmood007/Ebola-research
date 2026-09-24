"""verify_neighbour_signal_doc.py -- read the numbers back OUT of the neighbour-signal outcome doc and
recompute them from disk.

This result drives a design decision (whether to build a deviation channel) and probably a manuscript
sentence, so every number, label and count in the doc is checked against the artifacts:

  results/misc/t6_neighbour_signal.json            main run (20 cells, controls, negative control)
  results/misc/t6_neighbour_signal__controls.json  the fixed wrong-map control
  results/misc/t6_neighbour_signal__breaktest.json the deliberate break of that control
  results/misc/t6b_wrongmap_floor.json             per-panel false-alarm floors (dengue may be pending)
  results/misc/t5_input_energy.json                the t5 numbers quoted as context
  results/misc/t6c_level_check.json                dev-panel level check ("is it just size?")
  experiments/t6_ebola_neighbour_signal.json       Ebola, EXPLORATORY: must stay labelled so, carry the
                                                   frozen-hash record, and never be sold as case-study

The dengue floor prose (trigger count, ratios, 6-draw binomial numbers) is recomputed from t6b with
scipy. The Ebola "21 of 146 cross-border edges" is recomputed from the ebola_L12 bundle.

Independence. The four small panels' real-map gains are RE-DERIVED here from the processed bundles
with a separate implementation (it does not import t6), and must match the JSON. Everything else is
parsed from the JSON and re-derived where a rule exists: WIN flags from gain and null p95, the bucket
from the pre-committed rule, pooled false-alarm counts, share of total variance. Dengue is not re-derived
(it needs the full 7,165-district pipeline); its rows are checked against the JSON only.

Dengue floor. While t6b_wrongmap_floor.json has no dengue entry, the doc MUST mark dengue's floor
PENDING. Once it lands, the doc must carry dengue's numbers and no PENDING anywhere.

    conda run -n ebola-train python diagnostics/verify_neighbour_signal_doc.py
    conda run -n ebola-train python diagnostics/verify_neighbour_signal_doc.py --mutate
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MISC = ROOT / "results" / "misc"
DOC = ROOT / "progress" / "outcomes" / "Neighbour_Signal_2026-09-24.md"
PANELS = ("influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states", "dengue")
SMALL = PANELS[:4]
FLOOR_SMALL = ("influenza_japan", "influenza_us-states", "covid_us-states")
HS = (3, 5, 10, 15)
ALPHA, K, W, MAXH = 1.0, 4, 20, 15


def tol(s):
    """Half a unit in the last printed digit."""
    s = s.strip().lstrip("+-")
    return 0.5 * (10 ** -len(s.split(".")[1]) if "." in s else 1.0) + 1e-12


# --------------------------------------------------------------------------- disk side
def load():
    d = {k: json.loads((MISC / f).read_text()) for k, f in (
        ("t6", "t6_neighbour_signal.json"), ("ctl", "t6_neighbour_signal__controls.json"),
        ("brk", "t6_neighbour_signal__breaktest.json"), ("fl", "t6b_wrongmap_floor.json"),
        ("t5", "t5_input_energy.json"), ("lvl", "t6c_level_check.json"))}
    d["eb"] = json.loads((ROOT / "experiments" / "t6_ebola_neighbour_signal.json").read_text())
    return d


def indep_small():
    """Separate re-implementation for the four fully-observed single-group panels. The observed-flag
    columns in t6 are constant 1 on these panels, and a penalised constant column next to an
    unpenalised intercept is fitted to exactly zero, so they are dropped here: same fit, different code."""
    import bundles
    out = {}
    for name in SMALL:
        b = bundles.load(name)
        x = b.X[:, :, 0].astype(np.float64)
        assert b.M.all(), f"{name}: independent path assumes full observation"
        ms = b.masks()
        tr, te = ms["train"].astype(bool), ms["test"].astype(bool)
        N, T = x.shape
        d = x - x.mean(0, keepdims=True)
        A = b.A_geo.astype(np.float64)
        deg = A.sum(1)
        nb = np.zeros_like(d)
        nz = deg > 0
        nb[nz] = (A[nz] @ d) / deg[nz, None]
        ts = np.arange(W - 1, T - MAXH)
        res = {}
        for h in HS:
            def rows(mask):
                i, k = np.nonzero(mask[:, ts + h])
                return i, ts[k]
            fi, ft = rows(tr)
            si, st = rows(te)

            def feats(i, t, with_nb):
                cols = [np.ones(len(i))] + [d[i, t - k] for k in range(K)]
                if with_nb:
                    cols += [nb[i, t - k] for k in range(K)]
                return np.stack(cols, 1)

            def mse(with_nb):
                Xf, Xs = feats(fi, ft, with_nb), feats(si, st, with_nb)
                P = ALPHA * np.eye(Xf.shape[1]); P[0, 0] = 0
                w = np.linalg.solve(Xf.T @ Xf + P, Xf.T @ d[fi, ft + h])
                return float(np.mean((Xs @ w - d[si, st + h]) ** 2))
            mA, mB = mse(False), mse(True)
            ys = d[si, st + h]
            res[h] = {"mse_A": mA, "gain": 100 * (mA - mB) / mA,
                      "share": float(np.var(ys) / np.var(x[si, st + h]))}
        out[name] = res
    b = bundles.load("ebola_L12")                     # Ebola cross-border edges, straight from the bundle
    g = np.array([b.group_of()[n] for n in b.meta["node_ids"]])
    ii, jj = np.nonzero(b.A_geo)
    out["eb_edges"] = (int((g[ii] != g[jj]).sum()) // 2, len(ii) // 2)
    return out


def cell(D, p, h):
    return D["t6"]["panels"][p]["horizons"][str(h)]


def total_share(c):
    return c["B"]["gain_real_pct"] * c["mse_A"] * c["deviation_share_of_var"] / c["var_target_score"]


def bucket_of(wins):
    multi = [p for p in PANELS if wins[p] >= 2]
    if len(multi) >= 2:
        return "BROAD SIGNAL"
    if len(multi) == 1:
        return "LOCALISED SIGNAL"
    if sum(wins.values()) <= 1:
        return "NO SIGNAL"
    return "UNCLASSIFIED"


# --------------------------------------------------------------------------- doc side
MAIN_RE = re.compile(r"^\|\s*(" + "|".join(PANELS) + r")\s*\|\s*(\d+)\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)"
                     r"\s*\|\s*([\d.]+)\s*\|\s*(WIN|no)\s*\|\s*([+-][\d.]+)\s*\|$")
FA_RE = re.compile(r"^\|\s*(" + "|".join(PANELS) + r")\s*\|\s*(\d+|all)\s*\|\s*(\d+|PENDING)\s*\|\s*(\d+|PENDING)"
                   r"\s*\|\s*([+-][\d.]+|PENDING)\s*\|\s*([+-][\d.]+|see results table)\s*\|\s*(yes|no|PENDING)\s*\|$")
C_RE = re.compile(r"^\|\s*(" + "|".join(PANELS) + r")\s*\|\s*(\d+)\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*\|$")
CTL_RE = re.compile(r"^\|\s*(positive 0\.50 SD|positive 0\.25 SD|legacy wrong-map, real base|fixed wrong-map, "
                    r"shuffled base)[^|]*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|$")


def S(p):
    return re.compile(p.replace(" ", r"\s+"))


PROSE = {
    "tally": S(r"Real neighbours win (\d+) of 20 cells"),
    "perpanel": S(r"Wins per panel: influenza_japan (\d+), influenza_us-regions (\d+), influenza_us-states (\d+), "
                  r"covid_us-states (\d+), dengue (\d+)\."),
    "bucket": S(r"Bucket by the pre-committed rule: ([A-Z ]+)\."),
    "shares": S(r"remove ([\d.]+), ([\d.]+) and ([\d.]+) percent on COVID at h3, h5 and h15, and ([\d.]+), "
                r"([\d.]+), ([\d.]+) and ([\d.]+) percent on dengue"),
    "pooled": S(r"(\d+) of (\d+) shuffled-base draws were flagged WIN, ([\d.]+) percent"),
    "neg": S(r"largest absolute median null gain is ([\d.]+) percent"),
    "within": S(r"clears at (\d) of 4 horizons"),
    "t5": S(r"falls from ([\d.]+) percent in the input incidence to ([\d.]+) percent in the encoder output\. "
            r"For dengue it falls from ([\d.]+) to ([\d.]+)\."),
    "legacy_dmse": S(r"at h3 is ([\d.]+), against ([\d.]+) with no plant"),
    "legacy_ratio": S(r"h3 gain was under half the positive control's h3 gain \(([\d.]+) against ([\d.]+)\)"),
    "break": S(r"wins jumped to (\d+), (\d+), (\d+) and (\d+) of 20"),
    "usstates": S(r"\(([\d.]+) against ([\d.]+)\), and the two are ([\d.]+) and ([\d.]+) percent"),
    "japan_max": S(r"none is above ([\d.]+) percent of total variance"),
    "negmax_seen": S(r"Four panels have 2 or more wins"),
}
BANNED = ("proves spread", "proven spread", "demonstrates spread", "shows spread", "evidence of spread",
          "evidence of transmission", "confirms spread", "criterion was met", "criterion is met",
          "criterion was satisfied")
EB1_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*\|"
                    r"\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*\|\s*(yes|no)\s*\|\s*(\d+)\s*\|\s*(yes|no)\s*\|$")
EB2_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*\|"
                    r"\s*([+-][\d.]+)\s*\|\s*(yes|no)\s*\|$")
LVL_RE = re.compile(r"^\|\s*(" + "|".join(PANELS) + r")\s*\|\s*(\d+) of (\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|$")
DPROSE = {
    "trigger": S(r"At h10, (\d+) of (\d+) shuffled-base draws were flagged WIN"),
    "ratio10": S(r"h10 floor is ([\d.]+) percent against a real gain of ([\d.]+) percent, about ([\d,]+) times smaller"),
    "ratiomin": S(r"smallest ratio across the four horizons is about (\d+) at h(\d+) \(([\d.]+) against ([\d.]+)\)"),
    "maxshuf": S(r"largest shuffled-base gain in all (\d+) dengue draws is ([\d.]+) percent"),
    "cp": S(r"anywhere from ([\d.]+) to ([\d.]+) percent \(exact 95 percent binomial interval\)"),
    "binom": S(r"reaches 2 of 6 at a given horizon ([\d.]+) percent of the time, and at one or more of four "
               r"horizons about ([\d.]+) percent"),
    "margin": S(r"margin is (\d+) to about ([\d,]+) times"),
    "lvl_dengue": S(r"neighbour gain slightly larger, ([\d.]+) to ([\d.]+) at h3"),
    "eb_l20": S(r"same gains to within ([\d.]+) points at h3 to h10"),
    "eb_h15": S(r"h15 has (\d+) fit rows"),
    "eb_edges": S(r"(\d+) of the (\d+) edges cross a border"),
    "eb_rows": S(r"(\d+) scored weeks, (\d+) scored rows per horizon"),
}


def rows(rx, text):
    return [m.groups() for m in (rx.match(l.strip()) for l in text.splitlines()) if m]


# --------------------------------------------------------------------------- checks
def c_artifacts(text, D, I):
    pr = []
    if D["t6"].get("quick"):
        pr.append("t6 JSON is a quick smoke run")
    for p in PANELS:
        if p not in D["t6"]["panels"]:
            pr.append(f"t6 JSON missing panel {p}")
    for p in FLOOR_SMALL:
        if p not in D["fl"]["panels"]:
            pr.append(f"t6b JSON missing panel {p}")
    return pr


def c_winflags(text, D, I):
    """JSON internal consistency: WIN == gain > 0 and gain > p95, for every cell."""
    pr = []
    for p in PANELS:
        for h in HS:
            b = cell(D, p, h)["B"]
            want = b["gain_real_pct"] > 0 and b["gain_real_pct"] > b["null_p95_pct"]
            if bool(b["win"]) != want:
                pr.append(f"{p} h{h}: JSON win flag {b['win']} disagrees with gain/p95 rule")
    return pr


def c_indep(text, D, I):
    pr = []
    for p in SMALL:
        for h in HS:
            c, r = cell(D, p, h), I[p][h]
            if abs(c["B"]["gain_real_pct"] - r["gain"]) > 1e-6:
                pr.append(f"{p} h{h}: independent gain {r['gain']:.6f} != JSON {c['B']['gain_real_pct']:.6f}")
            if abs(c["mse_A"] - r["mse_A"]) > 1e-9 * max(1, r["mse_A"]):
                pr.append(f"{p} h{h}: independent MSE_A {r['mse_A']:.9f} != JSON {c['mse_A']:.9f}")
            if abs(c["deviation_share_of_var"] - r["share"]) > 1e-9:
                pr.append(f"{p} h{h}: independent deviation share differs")
    return pr


def c_tally(text, D, I):
    pr = []
    wins = {p: sum(bool(cell(D, p, h)["B"]["win"]) for h in HS) for p in PANELS}
    m = PROSE["tally"].search(text)
    if not m:
        pr.append("tally sentence not found")
    elif int(m.group(1)) != sum(wins.values()):
        pr.append(f"doc tally {m.group(1)} != disk {sum(wins.values())}")
    m = PROSE["perpanel"].search(text)
    if not m:
        pr.append("per-panel wins sentence not found")
    else:
        got = dict(zip(PANELS, map(int, m.groups())))
        if got != wins:
            pr.append(f"doc per-panel wins {got} != disk {wins}")
    m = PROSE["bucket"].search(text)
    b = bucket_of(wins)
    if not m:
        pr.append("bucket sentence not found")
    elif m.group(1).strip() != b:
        pr.append(f"doc bucket {m.group(1)} != rule on disk {b}")
    if b != D["t6"]["verdict"]["bucket"]:
        pr.append(f"independent bucket {b} != JSON bucket {D['t6']['verdict']['bucket']}")
    if (sum(wins[p] >= 2 for p in PANELS) == 4) != bool(PROSE["negmax_seen"].search(text)):
        pr.append("'Four panels have 2 or more wins' does not match disk")
    return pr


def c_main(text, D, I):
    pr = []
    rs = rows(MAIN_RE, text)
    seen = set()
    for p, h, g, p95, pv, v, sh in rs:
        h = int(h); seen.add((p, h))
        c = cell(D, p, h); b = c["B"]
        for lab, s, val in (("gain", g, b["gain_real_pct"]), ("null p95", p95, b["null_p95_pct"]),
                            ("p", pv, b["p_value"]), ("share", sh, total_share(c))):
            if abs(float(s) - val) > tol(s):
                pr.append(f"{p} h{h}: doc {lab} {s}, disk {val:.4f}")
        if (v == "WIN") != bool(b["win"]):
            pr.append(f"{p} h{h}: doc verdict {v}, disk win={b['win']}")
    want = {(p, h) for p in PANELS for h in HS}
    for miss in sorted(want - seen):
        pr.append(f"{miss} missing from the results table")
    if len(rs) != len(seen):
        pr.append("duplicate rows in the results table")
    return pr


def c_shares_prose(text, D, I):
    pr = []
    m = PROSE["shares"].search(text)
    if not m:
        return ["share-of-total-variance sentence not found"]
    want = [total_share(cell(D, "covid_us-states", h)) for h in (3, 5, 15)] + \
           [total_share(cell(D, "dengue", h)) for h in HS]
    for s, w in zip(m.groups(), want):
        if abs(float(s) - w) > tol(s):
            pr.append(f"prose share {s} != disk {w:.3f}")
    m = PROSE["japan_max"].search(text)
    jm = max(total_share(cell(D, "influenza_japan", h)) for h in HS)
    if not m or jm > float(m.group(1)) + tol(m.group(1)):
        pr.append(f"Japan max share claim does not hold (disk max {jm:.3f})")
    m = PROSE["usstates"].search(text)
    if not m:
        pr.append("US-states floor sentence not found")
    else:
        fl = D["fl"]["panels"]["influenza_us-states"]["horizons"]["10"]
        want = [cell(D, "influenza_us-states", 10)["B"]["gain_real_pct"], fl["floor_size_pct"],
                total_share(cell(D, "influenza_us-states", 5)), total_share(cell(D, "influenza_us-states", 10))]
        for s, w in zip(m.groups(), want):
            if abs(float(s) - w) > tol(s):
                pr.append(f"US-states sentence {s} != disk {w:.4f}")
    return pr


def c_floor(text, D, I):
    pr = []
    fl = D["fl"]["panels"]
    rs = rows(FA_RE, text)
    seen = set()
    for p, h, fw, dr, fs, rg, cl in rs:
        if p == "dengue" and "dengue" not in fl:
            continue
        if p not in fl:
            pr.append(f"false-alarm row for {p} but no floor on disk"); continue
        if h == "all" or "PENDING" in (fw, dr, fs, cl):
            pr.append(f"{p}: false-alarm row is PENDING but the floor is on disk"); continue
        seen.add((p, int(h)))
        f = fl[p]["horizons"][h]
        if int(fw) != f["shuffled_wins"]:
            pr.append(f"{p} h{h}: doc false wins {fw}, disk {f['shuffled_wins']}")
        if int(dr) != fl[p]["draws"]:
            pr.append(f"{p} h{h}: doc draws {dr}, disk {fl[p]['draws']}")
        if abs(float(fs) - f["floor_size_pct"]) > tol(fs):
            pr.append(f"{p} h{h}: doc floor {fs}, disk {f['floor_size_pct']:.4f}")
        if rg != "see results table" and abs(float(rg) - cell(D, p, int(h))["B"]["gain_real_pct"]) > tol(rg):
            pr.append(f"{p} h{h}: doc real gain {rg} != t6")
        b = cell(D, p, int(h))["B"]
        clears = bool(b["win"]) and b["gain_real_pct"] > f["floor_size_pct"]
        if clears != f["base_clears_floor"]:
            pr.append(f"{p} h{h}: t6b clears flag inconsistent with t6 + floor")
        if (cl == "yes") != clears:
            pr.append(f"{p} h{h}: doc clears={cl}, disk {clears}")
    for p in fl:
        for h in HS:
            if (p, h) not in seen:
                pr.append(f"{p} h{h}: floor on disk but no row in the false-alarm table")
    # pooled count over the 20-draw panels
    pool = [p for p in fl if fl[p]["draws"] == 20]
    w = sum(fl[p]["horizons"][str(h)]["shuffled_wins"] for p in pool for h in HS)
    n = sum(fl[p]["draws"] for p in pool) * len(HS)
    m = PROSE["pooled"].search(text)
    if not m:
        pr.append("pooled false-alarm sentence not found")
    elif (int(m.group(1)), int(m.group(2))) != (w, n) or abs(float(m.group(3)) - 100 * w / n) > tol(m.group(3)):
        pr.append(f"doc pooled {m.group(1)} of {m.group(2)} ({m.group(3)}%), disk {w} of {n} ({100*w/n:.2f}%)")
    if "No floor exists on Japan, US-states or COVID" in text and any(fl[p]["floor_exists"] for p in FLOOR_SMALL):
        pr.append("doc says no floor exists, but a floor exists on disk")
    for p in fl:                     # the pre-committed rule, re-derived
        crit = fl[p]["critical_wins"]
        ex = any(fl[p]["horizons"][str(h)]["shuffled_wins"] >= crit for h in HS)
        if ex != fl[p]["floor_exists"]:
            pr.append(f"{p}: floor_exists flag disagrees with its own wins and critical count")
    # dengue pending logic
    if "dengue" not in fl:
        dr = [r for r in rs if r[0] == "dengue"]
        if not dr or dr[0][2] != "PENDING":
            pr.append("dengue floor is not on disk but the doc does not mark its row PENDING")
        if "Dengue's own floor is PENDING" not in text:
            pr.append("dengue floor is not on disk but the doc does not say it is PENDING")
    elif "PENDING" in text:
        pr.append("dengue floor is on disk but the doc still says PENDING")
    return pr


def c_controls(text, D, I):
    pr = []
    c6, cc = D["t6"]["controls"], D["ctl"]["controls"]
    src = {"positive 0.50 SD": ("gain", c6["positive"]), "positive 0.25 SD": ("gain", c6["positive_weak"]),
           "legacy wrong-map, real base": ("gain", c6["wrong_map"]),
           "fixed wrong-map, shuffled base": ("wins", cc["wrong_map_shuffled_base"])}
    rs = rows(CTL_RE, text)
    if {r[0] for r in rs} != set(src):
        pr.append(f"controls table rows {sorted(r[0] for r in rs)} incomplete")
    for lab, *vals in rs:
        kind, r = src[lab]
        for h, s in zip(HS, vals):
            if kind == "gain":
                v = r["horizons"][str(h)]["B"]["gain_real_pct"]
                if abs(float(s) - v) > tol(s):
                    pr.append(f"{lab} h{h}: doc {s}, disk {v:.4f}")
            elif int(s) != r["wins"][str(h)]:
                pr.append(f"{lab} h{h}: doc {s} wins, disk {r['wins'][str(h)]}")
    if not cc["wrong_map_shuffled_base"]["passed"] or not c6["positive"]["passed"]:
        pr.append("a control the doc reports as passed failed on disk")
    m = PROSE["neg"].search(text)
    negmax = max(abs(cell(D, p, h)["B"]["null_median_pct"]) for p in PANELS for h in HS)
    if not m or abs(float(m.group(1)) - negmax) > tol(m.group(1)):
        pr.append(f"negative-control max: doc {m and m.group(1)}, disk {negmax:.4f}")
    m = PROSE["break"].search(text)
    bw = [D["brk"]["wins"][str(h)] for h in HS]
    if not m or list(map(int, m.groups())) != bw or D["brk"]["assert_would_pass"]:
        pr.append(f"break-test sentence does not match disk {bw}")
    m = PROSE["legacy_ratio"].search(text)
    want = [c6["wrong_map"]["horizons"]["3"]["B"]["gain_real_pct"], c6["positive"]["horizons"]["3"]["B"]["gain_real_pct"]]
    if not m or any(abs(float(s) - w) > tol(s) for s, w in zip(m.groups(), want)):
        pr.append("legacy magnitude-check sentence does not match disk")
    m = PROSE["legacy_dmse"].search(text)
    fj = D["fl"]["panels"]["influenza_japan"]["horizons"]["3"]
    want = [fj["realbase_dmse_mean"], fj["base_dmse"]]
    if not m or any(abs(float(s) - w) > tol(s) for s, w in zip(m.groups(), want)):
        pr.append("legacy dMSE sentence does not match disk")
    return pr


def c_secondary(text, D, I):
    pr = []
    for p, h, bs, cs in rows(C_RE, text):
        c = cell(D, p, int(h))
        for s, v in ((bs, c["B"]["gain_real_pct"]), (cs, c["C"]["gain_real_pct"])):
            if abs(float(s) - v) > tol(s):
                pr.append(f"spread table {p} h{h}: doc {s}, disk {v:.4f}")
    wg = D["t6"]["panels"]["dengue"]["within_group_null"]["horizons"]
    nw = sum(bool(wg[str(h)]["B"]["win"]) for h in HS)
    for m in PROSE["within"].finditer(text):
        if int(m.group(1)) != nw:
            pr.append(f"within-country null: doc {m.group(1)} of 4, disk {nw}")
    if not list(PROSE["within"].finditer(text)):
        pr.append("within-country sentence not found")
    m = PROSE["t5"].search(text)
    t5 = D["t5"]["panels"]
    want = [t5["covid_us-states"]["distinct_pct"]["input_incidence"], t5["covid_us-states"]["distinct_pct"]["representation_h"],
            t5["dengue"]["distinct_pct"]["input_incidence"], t5["dengue"]["distinct_pct"]["representation_h"]]
    if not m or any(abs(float(s) - w) > tol(s) for s, w in zip(m.groups(), want)):
        pr.append("t5 context sentence does not match results/misc/t5_input_energy.json")
    return pr


def c_claims(text, D, I):
    pr = []
    low = text.lower()
    for ph in BANNED:
        for m in re.finditer(re.escape(ph), low):
            pre = low[max(0, m.start() - 14):m.start()]
            if not re.search(r"\b(not|no|nor)\b", pre):
                pr.append(f"unhedged claim '{ph}' at char {m.start()}")
    if "—" in text:
        pr.append("em dash present (house style)")
    return pr


def _num(s):
    return float(s.replace(",", ""))


def c_dengue_floor(text, D, I):
    """Dengue floor prose, recomputed from t6b. Skipped (and required absent) while dengue is pending."""
    fl = D["fl"]["panels"]
    if "dengue" not in fl:
        return []
    from scipy.stats import beta, binom
    pr = []
    dg = fl["dengue"]; hz = dg["horizons"]; R = dg["draws"]
    ratios = {h: hz[str(h)]["base_gain_pct"] / hz[str(h)]["floor_size_pct"] for h in HS}
    m = DPROSE["trigger"].search(text)
    if not m or (int(m.group(1)), int(m.group(2))) != (hz["10"]["shuffled_wins"], R):
        pr.append("dengue trigger sentence does not match disk")
    elif hz["10"]["shuffled_wins"] < dg["critical_wins"] or not dg["floor_exists"]:
        pr.append("doc says the trigger fired at h10 but disk does not meet the critical count")
    if "a floor EXISTS on dengue" in text and not dg["floor_exists"]:
        pr.append("doc says a floor exists on dengue, disk says not")
    m = DPROSE["ratio10"].search(text)
    if not m:
        pr.append("dengue h10 ratio sentence not found")
    else:
        f, g, r = m.groups()
        if abs(float(f) - hz["10"]["floor_size_pct"]) > tol(f) or abs(float(g) - hz["10"]["base_gain_pct"]) > tol(g) \
                or abs(_num(r) - ratios[10]) > 1.0:
            pr.append(f"dengue h10 ratio sentence ({f}, {g}, {r}) != disk ({hz['10']['floor_size_pct']:.4f}, "
                      f"{hz['10']['base_gain_pct']:.3f}, {ratios[10]:.1f})")
    m = DPROSE["ratiomin"].search(text)
    hmin = min(ratios, key=ratios.get)
    if not m or abs(int(m.group(1)) - ratios[hmin]) > 1.0 or int(m.group(2)) != hmin \
            or abs(float(m.group(3)) - hz[str(hmin)]["base_gain_pct"]) > tol(m.group(3)) \
            or abs(float(m.group(4)) - hz[str(hmin)]["floor_size_pct"]) > tol(m.group(4)):
        pr.append(f"smallest-ratio sentence does not match disk (h{hmin}, {ratios[hmin]:.1f})")
    m = DPROSE["margin"].search(text)
    if not m or abs(int(m.group(1)) - ratios[hmin]) > 1.0 or abs(_num(m.group(2)) / max(ratios.values()) - 1) > 0.02:
        pr.append(f"margin sentence does not match disk ({ratios[hmin]:.0f} to {max(ratios.values()):.0f})")
    allg = [g for h in HS for g in hz[str(h)]["shuffled_gains_pct"]]
    m = DPROSE["maxshuf"].search(text)
    if not m or int(m.group(1)) != len(allg) or abs(float(m.group(2)) - max(allg)) > tol(m.group(2)):
        pr.append(f"largest shuffled gain sentence does not match disk ({len(allg)} draws, {max(allg):.5f})")
    k = hz["10"]["shuffled_wins"]
    lo, hi = beta.ppf(0.025, k, R - k + 1) * 100, beta.ppf(0.975, k + 1, R - k) * 100
    m = DPROSE["cp"].search(text)
    if not m or abs(float(m.group(1)) - lo) > tol(m.group(1)) or abs(float(m.group(2)) - hi) > tol(m.group(2)):
        pr.append(f"binomial interval sentence does not match ({lo:.1f} to {hi:.1f})")
    p1 = binom.sf(dg["critical_wins"] - 1, R, 0.05)
    m = DPROSE["binom"].search(text)
    if not m or abs(float(m.group(1)) - 100 * p1) > tol(m.group(1)) \
            or abs(float(m.group(2)) - 100 * (1 - (1 - p1) ** 4)) > tol(m.group(2)):
        pr.append(f"trigger-probability sentence does not match ({100*p1:.1f}, {100*(1-(1-p1)**4):.1f})")
    return pr


def c_level(text, D, I):
    pr = []
    L = D["lvl"]["panels"]
    rs = rows(LVL_RE, text)
    if {r[0] for r in rs} != set(PANELS):
        pr.append("level-check table does not cover all five panels")
    for p, nz, n, sd, ch in rs:
        r = L[p]
        if (int(nz), int(n)) != (r["nodes_level_nonzero"], r["N"]):
            pr.append(f"level table {p}: {nz} of {n}, disk {r['nodes_level_nonzero']} of {r['N']}")
        for s, v in ((sd, r["level_sd"]), (ch, r["max_abs_change_pct_points"])):
            if abs(float(s) - v) > tol(s):
                pr.append(f"level table {p}: doc {s}, disk {v:.4f}")
    for p in PANELS:                              # the level run must sit on the published gains
        for h in HS:
            if abs(L[p]["horizons"][str(h)]["gain_pct"] - cell(D, p, h)["B"]["gain_real_pct"]) > 1e-9:
                pr.append(f"level JSON {p} h{h}: no-level gain differs from the main run")
    if "exactly zero" in text and any(L[p]["nodes_level_nonzero"] for p in SMALL):
        pr.append("doc says the single-country levels are exactly zero; disk disagrees")
    m = DPROSE["lvl_dengue"].search(text)
    hz = L["dengue"]["horizons"]["3"]
    if not m or abs(float(m.group(1)) - hz["gain_pct"]) > tol(m.group(1)) \
            or abs(float(m.group(2)) - hz["gain_level_pct"]) > tol(m.group(2)):
        pr.append("dengue level sentence does not match disk")
    if "slightly larger" in text and any(L["dengue"]["horizons"][str(h)]["change_pct_points"] <= 0 for h in HS):
        pr.append("doc says the dengue gain gets larger with level; disk has a horizon where it does not")
    return pr


def c_ebola(text, D, I):
    pr = []
    E = D["eb"]
    if E.get("protocol") != "EXPLORATORY":
        pr.append(f"Ebola JSON protocol is {E.get('protocol')!r}, not EXPLORATORY")
    if E.get("frozen_arm_hashes_verified") != ["before", "after"]:
        pr.append("Ebola JSON does not record frozen-hash checks before and after")
    if "## Ebola, EXPLORATORY" not in text:
        pr.append("Ebola section heading is not labelled EXPLORATORY")
    if "This cannot enter the case-study result." not in text:
        pr.append("Ebola section lacks the case-study exclusion sentence")
    H = E["horizons"]; mr = E["design"]["min_rows_readable"]
    yn = lambda b: "yes" if b else "no"
    r1 = rows(EB1_RE, text)
    if sorted(int(r[0]) for r in r1) != list(HS):
        pr.append("Ebola first-pass table does not have one row per horizon")
    for h, fit, sc, g, p95, wp95, pg, pf, fa, rd in r1:
        c = H[h]
        readable = c["n_fit_rows"] >= mr and c["n_score_rows"] >= mr and c["positive"]["win"]
        if readable != c["readable"]:
            pr.append(f"Ebola h{h}: JSON readable flag disagrees with the pre-committed rule")
        checks = [(fit, c["n_fit_rows"], "fit rows"), (sc, c["n_score_rows"], "score rows"),
                  (g, c["gain_real_pct"], "gain"), (p95, c["global"]["null_p95_pct"], "null p95"),
                  (wp95, c["within_country"]["null_p95_pct"], "within p95"), (pg, c["positive"]["gain_pct"], "planted")]
        for s, v, lab in checks:
            if abs(float(s) - v) > tol(s):
                pr.append(f"Ebola h{h} {lab}: doc {s}, disk {v}")
        if pf != yn(c["positive"]["win"]) or int(fa) != c["false_alarm"]["wins"] or rd != yn(c["readable"]):
            pr.append(f"Ebola h{h}: planted/false-alarm/readable labels do not match disk")
        if c["readable"] and c["global"]["win"] is False and "Real neighbours beat both nulls" in text:
            pr.append(f"Ebola h{h}: doc says both nulls beaten at a readable horizon; disk says no")
    r2 = rows(EB2_RE, text)
    if sorted(int(r[0]) for r in r2) != list(HS):
        pr.append("Ebola level table does not have one row per horizon")
    for h, g, p95, wp95, pg, pf in r2:
        lc = H[h]["level_controlled"]
        for s, v, lab in ((g, lc["gain_real_pct"], "gain"), (p95, lc["global"]["null_p95_pct"], "null p95"),
                          (wp95, lc["within_country"]["null_p95_pct"], "within p95"), (pg, lc["positive_gain_pct"], "planted")):
            if abs(float(s) - v) > tol(s):
                pr.append(f"Ebola level h{h} {lab}: doc {s}, disk {v}")
        if pf != yn(lc["positive"]["win"]):
            pr.append(f"Ebola level h{h}: planted-found label does not match disk")
    # the three verdict lines, re-derived
    for h in (3, 5):
        lc = H[str(h)]["level_controlled"]
        if lc["global"]["win"] or lc["within_country"]["win"] or not lc["positive"]["win"]:
            pr.append(f"Ebola h{h}: 'a real null on timing' does not hold on disk")
    if H["10"]["level_controlled"]["positive"]["win"]:
        pr.append("Ebola h10: doc says no power, disk says the planted signal is found")
    m = DPROSE["eb_l20"].search(text)
    d20 = max(abs(H[str(h)]["l20_gain_pct"] - H[str(h)]["gain_real_pct"]) for h in (3, 5, 10))
    if not m or abs(float(m.group(1)) - d20) > tol(m.group(1)):
        pr.append(f"Ebola L20 sentence does not match disk ({d20:.4f})")
    m = DPROSE["eb_h15"].search(text)
    if not m or int(m.group(1)) != H["15"]["n_fit_rows"]:
        pr.append("Ebola h15 fit-row sentence does not match disk")
    m = DPROSE["eb_rows"].search(text)
    nweeks = E["design"]["dense_weeks"][1] - E["design"]["split_target_week"] + 1
    if not m or int(m.group(1)) != nweeks or any(H[str(h)]["n_score_rows"] != int(m.group(2)) for h in HS):
        pr.append("Ebola scored weeks / rows sentence does not match disk")
    m = DPROSE["eb_edges"].search(text)
    if not m or (int(m.group(1)), int(m.group(2))) != I["eb_edges"]:
        pr.append(f"Ebola cross-border edge sentence does not match the bundle {I['eb_edges']}")
    return pr


CHECKS = (("artifacts", c_artifacts), ("JSON win flags", c_winflags), ("independent recompute", c_indep),
          ("tally and bucket", c_tally), ("results table", c_main), ("share prose", c_shares_prose),
          ("false-alarm floors", c_floor), ("controls", c_controls), ("secondary numbers", c_secondary),
          ("dengue floor prose", c_dengue_floor), ("level check", c_level), ("Ebola EXPLORATORY", c_ebola),
          ("claims and style", c_claims))


def run(text, D, I, verbose=True):
    allp = []
    for name, fn in CHECKS:
        pr = fn(text, D, I)
        allp += pr
        if verbose:
            print(f"  {'FAIL' if pr else 'ok  '}  {name}" + (f"  ({len(pr)} problem(s))" if pr else ""))
            for p in pr:
                print(f"        - {p}")
    return allp


# --------------------------------------------------------------------------- mutation test
def mutate(text, D, I):
    def sub(old, new):
        assert old in text, f"mutation anchor not found: {old!r}"
        return text.replace(old, new, 1)

    doc_muts = [
        ("tally inflated", sub("win 13 of 20 cells", "win 14 of 20 cells")),
        ("per-panel wins changed", sub("covid_us-states 3, dengue 4.", "covid_us-states 2, dengue 4.")),
        ("bucket changed", sub("rule: BROAD SIGNAL.", "rule: LOCALISED SIGNAL.")),
        ("stale gain in results row", sub("| covid_us-states | 3 | +8.054 |", "| covid_us-states | 3 | +8.154 |")),
        ("verdict flipped", sub("| +0.304 | +0.315 | 0.065 | no |", "| +0.304 | +0.315 | 0.065 | WIN |")),
        ("results row deleted", text.replace(next(l for l in text.splitlines() if l.startswith("| dengue | 15 | +0.673"))
                                             + "\n", "", 1)),
        ("share column stale", sub("| 0.005 | WIN | +5.33 |", "| 0.005 | WIN | +6.12 |")),
        ("prose share, relayed COVID number", sub("remove 5.3, 2.3", "remove 6.1, 2.3")),
        ("prose share, relayed dengue number", sub("and 1.8, 2.4, 2.0 and 0.5 percent", "and 4.3, 2.4, 2.0 and 0.5 percent")),
        ("pooled false alarms", sub("15 of 240 shuffled-base", "12 of 240 shuffled-base")),
        ("false-alarm row stale", sub("| influenza_japan | 5 | 2 | 20 |", "| influenza_japan | 5 | 0 | 20 |")),
        ("clears flag flipped", sub("| +0.654 | +0.627 | no |", "| +0.654 | +0.627 | yes |")),
        ("dengue reverted to PENDING", sub("| dengue | 3 | 0 | 6 | +0.0020 | +5.543 | yes |",
                                           "| dengue | all | PENDING | PENDING | PENDING | see results table | PENDING |")),
        ("dengue floor row stale", sub("| dengue | 10 | 2 | 6 | +0.0046 |", "| dengue | 10 | 0 | 6 | +0.0046 |")),
        ("dengue trigger count", sub("At h10, 2 of 6", "At h10, 1 of 6")),
        ("dengue ratio, relayed 570", sub("about 625 times smaller", "about 570 times smaller")),
        ("dengue smallest ratio", sub("about 307 at h15", "about 370 at h15")),
        ("dengue binomial interval", sub("4.3 to 77.7 percent", "4.3 to 47.7 percent")),
        ("dengue trigger probability", sub("about 12.5 percent", "about 21.5 percent")),
        ("dengue largest shuffled gain", sub("dengue draws is 0.0046 percent", "dengue draws is 0.0406 percent")),
        ("level table stale", sub("| dengue | 6662 of 7165 | 0.2317 | 0.0588 |", "| dengue | 6662 of 7165 | 0.2317 | 0.0088 |")),
        ("level table nonzero count", sub("| covid_us-states | 0 of 49 |", "| covid_us-states | 3 of 49 |")),
        ("dengue level prose", sub("5.543 to 5.602 at h3", "5.543 to 4.602 at h3")),
        ("Ebola gain stale", sub("| 5 | 517 | 394 | +5.843 |", "| 5 | 517 | 394 | +5.943 |")),
        ("Ebola h15 made readable", sub("| -30.69 | no | 3 | no |", "| -30.69 | no | 3 | yes |")),
        ("Ebola level gain stale", sub("| 3 | +0.543 |", "| 3 | +1.743 |")),
        ("Ebola h10 power flipped", sub("| +1.07 | no |", "| +1.07 | yes |")),
        ("Ebola EXPLORATORY label dropped", sub("## Ebola, EXPLORATORY", "## Ebola")),
        ("Ebola case-study exclusion dropped", sub("This cannot enter the case-study result.",
                                                   "This supports the case-study result.")),
        ("Ebola L20 claim", sub("within 0.023 points", "within 0.003 points")),
        ("Ebola edges", sub("21 of the 146 edges", "12 of the 146 edges")),
        ("Ebola scored rows", sub("12 scored weeks, 394 scored rows", "12 scored weeks, 494 scored rows")),
        ("pre-registration claim inserted", text + "\nOn this data the pre-registered criterion was met.\n"),
        ("positive control stale", sub("| +21.788 |", "| +12.788 |")),
        ("fixed control wins stale", sub("shuffled base (wins of 20) | 1 | 1 |", "shuffled base (wins of 20) | 0 | 1 |")),
        ("break-test numbers stale", sub("20, 20, 13 and 17", "20, 20, 13 and 7")),
        ("negative control stale", sub("null gain is 0.220 percent", "null gain is 0.120 percent")),
        ("within-country count", sub("clears at\n4 of 4 horizons", "clears at\n3 of 4 horizons")),
        ("t5 context stale", sub("from 23.8 percent", "from 32.8 percent")),
        ("spread table stale", sub("| +8.054 | +10.964 |", "| +8.054 | +9.964 |")),
        ("legacy dMSE stale", sub("at h3 is 0.00087", "at h3 is 0.00060")),
        ("US-states floor sentence stale", sub("(0.627 against 0.654)", "(0.627 against 0.554)")),
        ("unhedged spread claim", text + "\nThis proves spread between districts.\n"),
    ]
    D1 = copy.deepcopy(D); D1["t6"]["panels"]["covid_us-states"]["horizons"]["3"]["B"]["gain_real_pct"] += 0.5
    D2 = copy.deepcopy(D); D2["t6"]["panels"]["influenza_japan"]["horizons"]["10"]["B"]["win"] = False
    D3 = copy.deepcopy(D); D3["fl"]["panels"].pop("dengue", None)
    D4 = copy.deepcopy(D); D4["fl"]["panels"]["influenza_japan"]["horizons"]["5"]["shuffled_wins"] = 4
    D5 = copy.deepcopy(D); D5["eb"]["protocol"] = "SCORED"
    D6 = copy.deepcopy(D); D6["eb"].pop("frozen_arm_hashes_verified", None)
    D7 = copy.deepcopy(D); D7["eb"]["horizons"]["3"]["level_controlled"]["global"]["win"] = True
    D8 = copy.deepcopy(D); D8["lvl"]["panels"]["covid_us-states"]["nodes_level_nonzero"] = 3
    D9 = copy.deepcopy(D); D9["fl"]["panels"]["dengue"]["horizons"]["10"]["floor_size_pct"] *= 10
    input_muts = [("JSON gain perturbed (input broken)", D1), ("JSON win flag flipped (input broken)", D2),
                  ("dengue floor vanishes from disk, doc carries numbers", D3),
                  ("a floor appears on Japan (input broken)", D4),
                  ("Ebola JSON relabelled as scored", D5), ("Ebola frozen-hash record missing", D6),
                  ("Ebola level-controlled h3 becomes a WIN", D7), ("COVID gets nonzero levels", D8),
                  ("dengue h10 floor 10x larger", D9)]

    total = len(doc_muts) + len(input_muts)
    print(f"{'=' * 78}\nMutation test: {total} corruptions, each must be caught\n{'=' * 78}")
    escaped = []
    for name, bad in doc_muts:
        pr = run(bad, D, I, verbose=False)
        print(f"  {'caught' if pr else 'NOT CAUGHT':<11} doc:   {name}" + (f"  ({len(pr)})" if pr else ""))
        if not pr:
            escaped.append(name)
    for name, Dx in input_muts:
        pr = run(text, Dx, I, verbose=False)
        print(f"  {'caught' if pr else 'NOT CAUGHT':<11} input: {name}" + (f"  ({len(pr)})" if pr else ""))
        if not pr:
            escaped.append(name)
    print(f"\n{len(escaped)} escaped." if escaped else f"\nAll {total} mutations caught.")
    return escaped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc", nargs="?", default=str(DOC))
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    p = Path(a.doc)
    if not p.exists():
        sys.exit(f"no such document: {p}")
    text = p.read_text(encoding="utf-8")
    D = load()
    I = indep_small()
    print(f"{'=' * 78}\nVerifying {p.name} against the artifacts on disk\n{'=' * 78}")
    problems = run(text, D, I)
    print(f"  dengue false-alarm floor: {'ON DISK' if 'dengue' in D['fl']['panels'] else 'PENDING (doc must say so)'}")
    if a.mutate:
        print()
        if mutate(text, D, I):
            sys.exit(2)
    if problems:
        print(f"\n{len(problems)} PROBLEM(S). The document does not match its artifacts.")
        sys.exit(1)
    print(f"\nOK: {p.name} matches the artifacts on disk.")


if __name__ == "__main__":
    main()

"""Verify the transfer relabel records against the frozen protocol, without importing the runner.

Recomputes every cell, panel and outcome verdict from the raw per-draw metrics, checks every hash
stamp, the drift gate (re-read from results/lodo/), the draw list and the record set, then compares
with the runner's summary.json. Exit 1 on any failure.

  python diagnostics/verify_transfer_relabel.py              # the real records
  python diagnostics/verify_transfer_relabel.py --mutate     # corrupt real records in memory, all must be caught
  python diagnostics/verify_transfer_relabel.py --synthetic  # built records with known verdicts, cross-checked
                                                             # against run.py --report, then the mutation suite
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "experiments" / "transfer_relabel"
PROTOCOL = ROOT / "progress" / "decisions" / "Transfer_Relabel_Protocol.md"
LODO = ROOT / "results" / "lodo"
FOLD_PANELS = {"dengue": ["dengue"], "covid": ["covid_us-states"],
               "influenza": ["influenza_japan", "influenza_us-regions", "influenza_us-states"]}


def proto():
    raw = PROTOCOL.read_bytes().replace(b"\r\n", b"\n")
    text = raw.decode("utf-8")
    block = text[text.index("<!-- constants -->"):].split("```")[1]
    c = {}
    for ln in block.strip().split("\n"):
        k, v = ln.split("=", 1)
        c[k.strip()] = v.split()
    num = lambda k: float(c[k][0])
    return hashlib.sha256(raw).hexdigest(), dict(
        seeds=[int(x) for x in c["SEEDS"]], folds=c["PRIMARY_FOLDS"], perm=[int(x) for x in c["PERM_SEEDS"]],
        hs=[int(x) for x in c["HORIZONS"]], dec=c["DECIDING_METRICS"], pr=c["PRINTED_METRICS"],
        pert=c["DECIDING_PERTURBATION"][0], perts=c["PERTURBATIONS"], mat=num("MATERIALITY"),
        t=num("T_CRIT"), floor=num("STRENGTH_FLOOR"), tol=num("DRIFT_TOL"))


# --------------------------------------------------------------------------- #
# The rule, written again from the protocol text
# --------------------------------------------------------------------------- #
def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def rule(e, P):
    m = mean(e); half = P["t"] * sd(e) / math.sqrt(len(e)); lo, hi = m - half, m + half
    if m >= P["mat"] and lo > 0:
        v = "USED"
    elif m <= -P["mat"] and hi < 0:
        v = "HURTS"
    elif -P["mat"] < lo and hi < P["mat"]:
        v = "NOT USED"
    else:
        v = "INCONCLUSIVE"
    return m, lo, hi, v


def per_ckpt(rec, panel, pert, key):
    pn = rec["panels"][panel]; real = pn["real"][key][0]
    xs = [d["metrics"][key][0] for d in pn["draws"][pert]]
    if key.endswith("pcc"):
        return mean([real - x for x in xs])
    return mean([(x - real) / real for x in xs])


def recompute(recs, P):
    prim = [r for r in recs if r["role"] == "primary"]
    keys = [f"h{h}|{m}" for h in P["hs"] for m in P["dec"] + P["pr"]]
    cells, panels = {}, {}
    for pert in P["perts"]:
        cells[pert] = {}
        for f in P["folds"]:
            rs = sorted([r for r in prim if r["fold"] == f], key=lambda r: r["seed"])
            for panel in FOLD_PANELS[f]:
                cells[pert][panel] = {k: rule([per_ckpt(r, panel, pert, k) for r in rs], P) for k in keys}
    for panel, cs in cells[P["pert"]].items():
        f = next(f for f, ps in FOLD_PANELS.items() if panel in ps)
        r0 = min((r for r in prim if r["fold"] == f), key=lambda r: r["seed"])
        st = mean([d["nbr_changed"] for d in r0["panels"][panel]["draws"][P["pert"]]])
        both = lambda h, v: all(cs[f"h{h}|{m}"][3] == v for m in P["dec"])
        used = [h for h in P["hs"] if both(h, "USED")]
        hurt = [h for h in P["hs"] if both(h, "HURTS")]
        if used and hurt:
            v = "MIXED"
        elif used:
            v = "USED"
        elif hurt:
            v = "HURTS"
        elif all(cs[f"h{h}|{m}"][3] == "NOT USED" for h in P["hs"] for m in P["dec"]):
            v = "NOT USED"
        else:
            v = "INCONCLUSIVE"
        if st < P["floor"] and v in ("NOT USED", "INCONCLUSIVE"):
            v = "WEAK"
        panels[panel] = (v, used, hurt, st)
    vs = [p[0] for p in panels.values()]
    live = [v for v in vs if v != "WEAK"]
    out = ("O5" if "MIXED" in vs or ("USED" in vs and "HURTS" in vs) else "O3" if "USED" in vs
           else "O4" if "HURTS" in vs else "O1" if live and all(v == "NOT USED" for v in live) else "O2")
    return cells, panels, out


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def gap(a, b):
    if a != a and b != b:
        return 0.0
    return math.inf if (a != a or b != b) else abs(a - b) / max(abs(b), 1.0)


def verify(recs, summary, sha, P, synthetic):
    F = []
    if not recs:
        return ["no records"]
    for r in recs:
        tag = f"{r.get('family')}|{r.get('fold')}|{r.get('seed')}"
        if r.get("protocol_sha256") != sha:
            F.append(f"{tag}: protocol stamp {r.get('protocol_sha256')} is not this file's {sha}")
        if r.get("lane") != "EXPLORATORY":
            F.append(f"{tag}: lane label missing")
        if r.get("synthetic") is not synthetic:
            F.append(f"{tag}: synthetic flag {r.get('synthetic')} in a {'synthetic' if synthetic else 'real'} check")
        if r.get("perm_seeds") != P["perm"]:
            F.append(f"{tag}: perm_seeds differ from the protocol list")
        for panel, pn in r["panels"].items():
            dr = pn["drift"]
            dm = max(gap(a, b) for k in pn["archived"] for a, b in zip(pn["real"][k], pn["archived"][k]))
            if not dr.get("passed") or dm > P["tol"] or dr.get("quantiles", math.inf) > P["tol"]:
                F.append(f"{tag} {panel}: drift gate fails (metrics {dm:.2e}, flag {dr.get('passed')})")
            if set(pn["real"]) != set(pn["archived"]):
                F.append(f"{tag} {panel}: real and archived metric keys differ")
            if not synthetic:
                stem = {"ldo3": "encoder_ldo3", "ldo3full_zeroshot": "encoder_ldo3full_zeroshot"}[r["family"]]
                disk = {f"h{x['horizon']}|{x['metric']}": [x["country_macro"], x["node_mean"]]
                        for x in json.loads((LODO / f"{stem}__{panel}__seed{r['seed']}.json").read_text())}
                if any(gap(a, b) > 0 for k in disk for a, b in zip(pn["archived"].get(k, [math.nan] * 2), disk[k])):
                    F.append(f"{tag} {panel}: record's archived numbers differ from results/lodo/{stem}")
            for pert in P["perts"]:
                ds = pn["draws"].get(pert, [])
                if [d["perm_seed"] for d in ds] != P["perm"]:
                    F.append(f"{tag} {panel} {pert}: draws are not the protocol list in order")
                if len({d["perm_sha1"] for d in ds}) != len(ds):
                    F.append(f"{tag} {panel} {pert}: duplicate relabels")
                if any(d["moved"] <= 0 for d in ds):
                    F.append(f"{tag} {panel} {pert}: an identity relabel slipped through")
                if any("metrics" not in d for d in ds):
                    F.append(f"{tag} {panel} {pert}: a draw has no metrics (dry record?)")
    prim = [r for r in recs if r["role"] == "primary"]
    want = sorted((f, s) for f in P["folds"] for s in P["seeds"])
    have = sorted((r["fold"], r["seed"]) for r in prim)
    if have != want:
        F.append(f"primary set is {len(have)} records, protocol wants {len(want)}: missing {sorted(set(want) - set(have))}")
    for r in prim:
        if sorted(r["panels"]) != sorted(FOLD_PANELS[r["fold"]]):
            F.append(f"{r['fold']}|{r['seed']}: panels {sorted(r['panels'])}")
    # same relabels and same strength for every seed of a panel
    for f in P["folds"]:
        rs = [r for r in prim if r["fold"] == f]
        for panel in FOLD_PANELS[f]:
            for pert in P["perts"]:
                sig = {json.dumps([(d["perm_sha1"], round(d["nbr_changed"], 12)) for d in r["panels"][panel]["draws"][pert]])
                       for r in rs if panel in r["panels"]}
                if len(sig) > 1:
                    F.append(f"{panel} {pert}: relabels or strength differ across seeds")
    if len({json.dumps(r.get("code_sha256"), sort_keys=True) for r in recs}) > 1:
        F.append("records were written by different code (code_sha256 differs)")
    if F:
        return F
    cells, panels, out = recompute(recs, P)
    if summary is None:
        return ["no summary.json to compare"]
    if summary.get("protocol_sha256") != sha:
        F.append("summary.json stamped with another protocol")
    for pert, byp in cells.items():
        for panel, cs in byp.items():
            for k, (m, lo, hi, v) in cs.items():
                s = summary["cells"].get(pert, {}).get(panel, {}).get(k)
                if s is None or s["verdict"] != v or max(abs(s["mean"] - m), abs(s["lo"] - lo), abs(s["hi"] - hi)) > 1e-9:
                    F.append(f"cell {pert} {panel} {k}: recomputed {v} {m:+.5f}, summary {s and (s['verdict'], s['mean'])}")
    for panel, (v, used, hurt, st) in panels.items():
        s = summary["panels"].get(panel, {})
        if (s.get("verdict"), s.get("used_horizons"), s.get("hurts_horizons")) != (v, used, hurt) or abs(s.get("strength", -1) - st) > 1e-9:
            F.append(f"panel {panel}: recomputed {v} {used} {hurt}, summary {s.get('verdict')}")
    if summary.get("outcome") != out:
        F.append(f"outcome: recomputed {out}, summary {summary.get('outcome')}")
    return F


# --------------------------------------------------------------------------- #
# Mutations: every one must produce at least one failure
# --------------------------------------------------------------------------- #
def first(recs, fold, role="primary"):
    return next(r for r in recs if r["fold"] == fold and r["role"] == role)


def mutations(P):
    def m_stamp(R, S): R[0]["protocol_sha256"] = "0" * 64
    def m_draw(R, S):
        pn = first(R, "dengue")["panels"]["dengue"]
        for d in pn["draws"][P["pert"]]:
            d["metrics"]["h3|rmse"][0] = pn["real"]["h3|rmse"][0] * 1.5
    def m_drift(R, S):
        pn = first(R, "covid")["panels"]["covid_us-states"]; pn["real"]["h5|mae"][0] *= 1.01
    def m_flag(R, S): first(R, "influenza")["panels"]["influenza_japan"]["drift"]["passed"] = False
    def m_drop(R, S): R.remove(first(R, "influenza"))
    def m_seeds(R, S): R[-1]["perm_seeds"] = list(reversed(R[-1]["perm_seeds"]))
    def m_hash(R, S): first(R, "covid")["panels"]["covid_us-states"]["draws"]["global"][3]["perm_sha1"] = "deadbeef0000"
    def m_strength(R, S):
        for r in R:
            for d in r["panels"].get("influenza_us-regions", {}).get("draws", {}).get(P["pert"], []):
                d["nbr_changed"] = 0.95 if d["nbr_changed"] < P["floor"] else 0.10
    def m_verdict(R, S):
        p = next(iter(S["panels"])); S["panels"][p]["verdict"] = "USED" if S["panels"][p]["verdict"] != "USED" else "NOT USED"
    def m_outcome(R, S): S["outcome"] = "O1" if S["outcome"] != "O1" else "O3"
    def m_dry(R, S): del first(R, "dengue")["panels"]["dengue"]["draws"][P["pert"]][0]["metrics"]
    return [("protocol stamp changed in one record", m_stamp), ("dengue h3 rmse draws inflated 50%", m_draw),
            ("covid real metric moved 1% off archive", m_drift), ("drift flag set false", m_flag),
            ("one primary record dropped", m_drop), ("perm seed list reordered", m_seeds),
            ("one relabel hash altered", m_hash), ("us-regions strength flipped across the floor", m_strength),
            ("summary panel verdict altered", m_verdict), ("summary outcome altered", m_outcome),
            ("a draw lost its metrics", m_dry)]


def mutate(recs, summary, sha, P, synthetic):
    base = verify(recs, summary, sha, P, synthetic)
    if base:
        print("baseline does not verify, mutation test is void:\n  " + "\n  ".join(base[:10])); return False
    caught = 0
    for name, f in mutations(P):
        R, S = copy.deepcopy(recs), copy.deepcopy(summary)
        f(R, S)
        F = verify(R, S, sha, P, synthetic)
        caught += bool(F)
        print(f"  {'caught' if F else 'MISSED'}  {name}" + (f"  ->  {F[0][:90]}" if F else ""))
    n = len(mutations(P))
    print(f"mutations caught: {caught} of {n}")
    return caught == n


# --------------------------------------------------------------------------- #
# Synthetic record set with verdicts known by construction
# --------------------------------------------------------------------------- #
SCENARIOS = {   # panel -> (true relative effect, seed spread, strength); expected verdict; then outcome
    "mixed": ({"dengue": (0.05, 0.005, 0.999), "influenza_japan": (0.0, 0.002, 0.83),
               "influenza_us-regions": (0.0, 0.002, 0.43), "influenza_us-states": (0.0, 0.04, 0.87),
               "covid_us-states": (-0.05, 0.005, 0.87)},
              {"dengue": "USED", "influenza_japan": "NOT USED", "influenza_us-regions": "WEAK",
               "influenza_us-states": "INCONCLUSIVE", "covid_us-states": "HURTS"}, "O5"),
    "null": ({"dengue": (0.001, 0.002, 0.999), "influenza_japan": (0.0, 0.002, 0.83),
              "influenza_us-regions": (0.0, 0.002, 0.43), "influenza_us-states": (-0.003, 0.003, 0.87),
              "covid_us-states": (0.004, 0.003, 0.87)},
             {"dengue": "NOT USED", "influenza_japan": "NOT USED", "influenza_us-regions": "WEAK",
              "influenza_us-states": "NOT USED", "covid_us-states": "NOT USED"}, "O1"),
}
OFF = [-1.5, -0.5, 0.0, 0.5, 1.5]                         # exact seed offsets, sd known


def build(P, sha, scen):
    eff, _, _ = SCENARIOS[scen]
    keys = [f"h{h}|{m}" for h in P["hs"] for m in ["rmse", "mae", "pcc", "nrmse", "smape", "peak_timing", "peak_intensity"]]
    K = len(P["perm"]); dk = [(i - (K - 1) / 2) / K * 0.004 for i in range(K)]     # zero-mean draw offsets
    recs = []
    rows = [("ldo3", "primary", f, s, i) for f in P["folds"] for i, s in enumerate(P["seeds"])]
    rows += [("ldo3full_zeroshot", "exploratory", f, 42, 2) for f in ("covid", "dengue")]
    for fam, role, fold, seed, i in rows:
        rec = dict(lane="EXPLORATORY", synthetic=True, protocol_sha256=sha, family=fam, role=role, fold=fold,
                   seed=seed, head="adapted" if role == "primary" else "zeroshot", checkpoint=f"synthetic__{fold}__{seed}",
                   code_sha256={"x": "synthetic"}, perm_seeds=P["perm"], panels={})
        for panel in FOLD_PANELS[fold]:
            e0, s0, st = eff[panel]
            real = {k: [100.0, 100.0] if not k.endswith("pcc") else [0.8, 0.8] for k in keys}
            draws = {}
            for pert in P["perts"]:
                ds = []
                for j, ps in enumerate(P["perm"]):
                    e = e0 + s0 * OFF[i] + dk[j]
                    met = {k: ([v * (1 + e) for v in real[k]] if not k.endswith("pcc") else [0.8 - e, 0.8 - e]) for k in keys}
                    ds.append(dict(perm_seed=ps, perm_sha1=hashlib.sha1(f"{panel}{pert}{ps}".encode()).hexdigest()[:12],
                                   moved=0.9, nbr_changed=st, count_changed=0.0, metrics=met))
                draws[pert] = ds
            rec["panels"][panel] = dict(n_nodes=10, real=real, archived=copy.deepcopy(real), unseen_map=True,
                                        drift=dict(metrics=0.0, quantiles=0.0, tol=P["tol"], passed=True), draws=draws)
        recs.append(rec)
    return recs


def synthetic(sha, P):
    ok = True
    for scen, (_, want, want_out) in SCENARIOS.items():
        d = LANE / f"_synthetic_{scen}"
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        try:
            recs = build(P, sha, scen)
            for r in recs:
                (d / f"{r['family']}__{r['fold']}__seed{r['seed']}.json").write_text(json.dumps(r))
            run = subprocess.run([sys.executable, str(LANE / "run.py"), "--report", "--dir", str(d)],
                                 cwd=ROOT, capture_output=True, text=True)
            if run.returncode != 0:
                print(f"[{scen}] run.py --report failed:\n{run.stderr[-1500:]}"); ok = False; continue
            summary = json.loads((d / "summary.json").read_text())
            F = verify(recs, summary, sha, P, True)
            _, panels, out = recompute(recs, P)
            got = {p: v[0] for p, v in panels.items()}
            known = got == want and out == want_out
            print(f"[{scen}] runner and verifier agree: {not F}; verdicts match construction: {known} "
                  f"({out}; {got})")
            for x in F[:10]:
                print("   ", x)
            ok &= not F and known
            print(f"[{scen}] mutation suite:")
            ok &= mutate(recs, summary, sha, P, True)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return ok


def main():
    sha, P = proto()
    if "--synthetic" in sys.argv:
        ok = synthetic(sha, P)
    else:
        recs = [json.loads(p.read_text()) for p in sorted(LANE.glob("*__seed*.json"))]
        sp = LANE / "summary.json"
        summary = json.loads(sp.read_text()) if sp.exists() else None
        if not recs:
            sys.exit("no records in experiments/transfer_relabel/ yet: the scored run has not happened. "
                     "Use --synthetic to exercise this verifier.")
        if "--mutate" in sys.argv:
            ok = mutate(recs, summary, sha, P, False)
        else:
            F = verify(recs, summary, sha, P, False)
            for x in F:
                print("FAIL", x)
            ok = not F
            if ok:
                _, panels, out = recompute(recs, P)
                print(f"verified {len(recs)} records against protocol {sha[:12]}: outcome {out}; "
                      + "; ".join(f"{p} {v[0]}" for p, v in panels.items()))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

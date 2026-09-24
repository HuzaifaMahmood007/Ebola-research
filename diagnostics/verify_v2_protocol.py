"""verify_v2_protocol.py -- read the numbers back OUT of progress/decisions/V2_Deviation_Protocol.md and
recompute every one from disk.

Deliberately does NOT import ablation/run_v2_deviation.py: the power planning, the paired spreads, the
probe translation and the seed-count formula are re-derived here with separate code from the raw
records, the quantile archives, the probe JSONs and the smoke summary. The runner's constants are read
from its SOURCE TEXT and compared with the protocol. A bug shared with the runner would otherwise
verify itself.

What is checked:
  1. The noise table (8 rows) and the planning power table (4 rows), every cell, to the precision printed.
  2. The seed-count worked examples (5 rows): n_z, n_t, N, the underpowered flag, power, stage-2 hours.
  3. The options table: chances and runtimes.
  4. Every number in the prose that comes from disk: false-call rates, powers, seeds needed, delta,
     probe shares, input-energy shares, origin counts, the manuscript note on the earlier COVID nulls,
     the smoke's determinism and drift measurements, runtimes, the branch size, the dengue figures,
     the D2 COVID verdict, the number of hashed code files.
  5. The runner's constants (seed sequence, stage-1 seeds, rule, delta, z values, power target, floor,
     cap) agree with the protocol text.

    python diagnostics/verify_v2_protocol.py
    python diagnostics/verify_v2_protocol.py --mutate      # corrupt the doc, the inputs and the runner constants
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DOC = ROOT / "progress" / "decisions" / "V2_Deviation_Protocol.md"
RUNNER = ROOT / "ablation" / "run_v2_deviation.py"
SMOKE = ROOT / "ablation" / "misc" / "v2_smoke_summary.json"
SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
METRICS = ("rmse", "mae")
SAME_INIT = ("gateoff", "shufadj", "epi_p90max", "epi_p99max")
# the protocol's formula constants, stated independently of the runner
P_DELTA, P_ZA, P_ZB, P_TARGET, P_MIN, P_CAP = 0.033, 1.96, 0.84, 0.80, 5, 134


# --------------------------------------------------------------------------- #
# Disk side
# --------------------------------------------------------------------------- #
def _recs(path):
    return {(r["horizon"], r["metric"]): r["country_macro"]
            for r in json.loads(Path(path).read_text()) if r["model"] == "encoder"}


def _sd(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(((x - x.mean()) ** 2).sum() / (len(x) - 1)))


def _pw(k, n, rule):
    c = math.sqrt(n) if rule == "gate" else stats.t.ppf(0.975, n - 1)
    return float(stats.nct.sf(c, n - 1, k * math.sqrt(n)))


def _k80(n, rule):
    return optimize.brentq(lambda k: _pw(k, n, rule) - 0.8, 1e-6, 50.0)


def _seed_count(s):
    """The protocol's section 5.3, written again here: linear scan, no bisection."""
    k = P_DELTA / s
    n_z = max(P_MIN, math.ceil(((P_ZA + P_ZB) * s / P_DELTA) ** 2))
    n_t = n_z
    while _pw(k, n_t, "tint") < P_TARGET:
        n_t += 1
    N = min(n_t, P_CAP)
    return dict(n_z=n_z, n_t=n_t, N=N, under=n_t > P_CAP, power=_pw(k, N, "tint"))


def _model_mse(ds):
    import bundles
    b = bundles.load(ds)
    test = b.masks()["test"].astype(bool)
    mu, sd = b.scaler["mean"][:, None], b.scaler["std"][:, None]
    out = {}
    for h in HORIZONS:
        v = []
        for s in SEEDS:
            z = np.load(ROOT / "results" / "single" / f"encoder__{ds}__seed{s}__quantiles.npz")
            q = z[f"h{h}__quantiles"][:, :, list(np.round(z["quantiles"], 3)).index(0.5)].astype(np.float64)
            pm = (np.log1p(np.clip(q, 0, None)) - mu) / sd
            cols = z["origins"] + h
            e = (pm - b.y[:, cols].astype(np.float64))[test[:, cols]]
            v.append(float(np.mean(e * e)))
        out[h] = float(np.mean(v))
    return out


def power_rows(ds, overrides=None):
    ov = overrides or {}
    v1 = {s: _recs(ROOT / "results" / "single" / f"encoder__{ds}__seed{s}.json") for s in SEEDS}
    if "v1" in ov:
        h, m, s, f = ov["v1"]
        v1[s][(h, m)] *= f
    t6 = json.loads((ROOT / "results" / "misc" / "t6_neighbour_signal.json").read_text())
    probe = copy.deepcopy(t6["panels"][ds]["horizons"])
    if "mse_A" in ov:
        h, f = ov["mse_A"]
        probe[str(h)]["mse_A"] *= f
    mse = _model_mse(ds)
    rows = {}
    for h in HORIZONS:
        dmse = max(probe[str(h)]["mse_A"] - probe[str(h)]["B"]["mse_real"], 0.0)
        plaus = 100 * (1 - math.sqrt(1 - dmse / mse[h]))
        for m in METRICS:
            base = [v1[s][(h, m)] for s in SEEDS]
            sds = {}
            for tag in SAME_INIT:
                ps = [ROOT / "ablation" / "single" / f"encoder__{ds}__seed{s}__{tag}.json" for s in SEEDS]
                if all(p.exists() for p in ps):
                    sds[tag] = _sd([_recs(p)[(h, m)] - b for p, b in zip(ps, base)])
            sdd, mean = float(np.median(list(sds.values()))), float(np.mean(base))
            bar = 100 * sdd / mean
            k = plaus / bar
            r = dict(v1_mean=mean, v1_sd=_sd(base), sdd=sdd, lo=min(sds.values()), hi=max(sds.values()),
                     bar=bar, plaus=plaus, k=k, arm_sds=sds)
            for rule, ns in (("gate", (5, 10)), ("tint", (10, 20, 40))):
                for n in ns:
                    r[f"mde_{rule}_{n}"] = 100 * _k80(n, rule) * sdd / mean
            for rule, n in (("gate", 5), ("gate", 10), ("tint", 20), ("tint", 40), ("tint", 134)):
                r[f"pow_{rule}_{n}"] = _pw(k, n, rule)
            r["n80"] = next((n for n in range(3, 6000) if _pw(k, n, "tint") >= 0.8), None) if k > 0 else None
            rows[(h, m)] = r
    return rows


def _runner_constants(src):
    g = lambda pat, flags=re.M: re.search(pat, src, flags)
    base, step, count = map(int, g(r"^SEED_SEQ = tuple\((\d+) \+ (\d+) \* i for i in range\((\d+)\)\)").groups())
    za, zb = map(float, g(r"^Z_ALPHA, Z_BETA = ([\d.]+), ([\d.]+)").groups())
    nmin, ncap = map(int, g(r"^N_MIN, N_CAP = (\d+), (\d+)").groups())
    files = re.findall(r'"([^"]+\.py)"', g(r"^CODE_FILES = \((.*?)\)\n", re.M | re.S).group(1))
    return dict(seq=tuple(base + step * i for i in range(count)),
                stage1=int(g(r"^STAGE1_SEEDS = SEED_SEQ\[:(\d+)\]").group(1)),
                rule=g(r'^RULE = "(\w+)"').group(1), delta=float(g(r"^DELTA = ([\d.]+)").group(1)),
                za=za, zb=zb, target=float(g(r"^POWER_TARGET = ([\d.]+)").group(1)), nmin=nmin, ncap=ncap,
                code_files=files)


def disk(overrides=None):
    import bundles
    d = {"covid": power_rows("covid_us-states", overrides)}
    dg = power_rows("dengue")
    d["dengue_k_short"] = [dg[(h, m)]["k"] for h in (3, 5) for m in METRICS]
    d["dengue_k_h10"] = [dg[(10, m)]["k"] for m in METRICS]
    d["fp5"], d["fp10"] = float(stats.t.sf(math.sqrt(5), 4)), float(stats.t.sf(math.sqrt(10), 9))
    t6 = json.loads((ROOT / "results" / "misc" / "t6_neighbour_signal.json").read_text())
    ch = t6["panels"]["covid_us-states"]["horizons"]
    d["share"] = {h: ch[str(h)]["B"]["gain_real_pct"] * ch[str(h)]["mse_A"]
                  / (ch[str(h)]["var_target_score"] / ch[str(h)]["deviation_share_of_var"]) for h in (3, 5)}
    t5 = json.loads((ROOT / "results" / "misc" / "t5_input_energy.json").read_text())
    d["t5"] = t5["panels"]["covid_us-states"]["distinct_pct"]
    b = bundles.load("covid_us-states")
    d["origins"] = tuple(len(b.origins(phase=p)) for p in ("train", "val", "test"))
    d["smoke"] = json.loads(SMOKE.read_text())
    from models.encoder_v2 import DeviationEncoder
    d["branch_params"] = sum(p.numel() for n, p in DeviationEncoder().named_parameters() if n.startswith("dev_"))
    d2 = []
    for h in HORIZONS:
        for m in METRICS:
            dl = [_recs(ROOT / "ablation" / "single" / f"encoder__covid_us-states__seed{s}__shufadj.json")[(h, m)]
                  - _recs(ROOT / "results" / "single" / f"encoder__covid_us-states__seed{s}.json")[(h, m)]
                  for s in SEEDS]
            d2.append(abs(np.mean(dl)) < _sd(dl))
    d["d2_all_noise"] = all(d2)
    rep = (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    d["dengue_5seed_hours"] = float(re.search(r"dengue ~([\d.]+) h;", rep).group(1))
    d["runner"] = _runner_constants(RUNNER.read_text())
    return d


# --------------------------------------------------------------------------- #
# Document side
# --------------------------------------------------------------------------- #
def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _eq(printed, value):
    """printed string equals value rounded to the printed precision."""
    p = printed.replace(",", "")
    dec = len(p.split(".")[1]) if "." in p else 0
    return abs(float(p) - value) <= 0.5 * 10 ** (-dec) + 1e-9


def check(text, d, verbose=True):
    fails = []
    n_checks = [0]

    def ok(cond, msg):
        n_checks[0] += 1
        if not cond:
            fails.append(msg)
    flat = re.sub(r"\s+", " ", text)
    rows = d["covid"]
    h3, h5 = rows[(3, "rmse")], rows[(5, "rmse")]
    sm = d["smoke"]
    # runtime basis: the doc states nominal per-run costs and the range the smokes measured. Every
    # runtime is checked exactly against the nominal costs; the latest smoke must sit inside the range.
    m_nom = re.search(r"nominal costs of (\d+) seconds per v1 COVID run .*? and (\d+) seconds per v2 COVID run", flat)
    m_rng = re.search(r"The smokes measured (\d+) to (\d+) seconds per v2 run", flat)
    m_rat = re.search(r"\(([\d.]+) to ([\d.]+) times across the smokes", flat)
    ok(m_nom is not None and m_rng is not None and m_rat is not None, "runtime basis sentences not found")
    v1s, v2s = (float(m_nom.group(1)), float(m_nom.group(2))) if m_nom else (30.0, 65.0)
    if m_nom and m_rng and m_rat:
        lo, hi = float(m_rng.group(1)), float(m_rng.group(2))
        ok(abs(v1s - sm["v1_covid_run_seconds_median"]) <= 1.0,
           f"nominal v1 {v1s}s vs the D2 record gaps {sm['v1_covid_run_seconds_median']:.1f}s")
        ok(lo <= sm["v2_covid_run_seconds"] <= hi and lo <= v2s <= hi,
           f"latest smoke v2 {sm['v2_covid_run_seconds']:.1f}s or nominal {v2s}s outside the stated {lo}-{hi}s")
        ok(float(m_rat.group(1)) <= sm["v2_over_v1_time"] <= float(m_rat.group(2)),
           f"latest smoke ratio {sm['v2_over_v1_time']:.2f} outside the stated range")

    # 1. noise and planning power tables
    lines = [_cells(l) for l in text.splitlines() if re.match(r"\| (3|5|10|15) \| (rmse|mae) \|", l)]
    seven = [c for c in lines if len(c) == 7]
    wide = [c for c in lines if len(c) == 14]
    ok(len(seven) == 8, f"noise table has {len(seven)} rows, want 8")
    ok(len(wide) == 4, f"power table has {len(wide)} rows, want 4")
    for c in seven:
        r = rows[(int(c[0]), c[1])]
        lo, hi = [x.strip() for x in c[5].split("to")]
        for name, p, v in (("v1 mean", c[2], r["v1_mean"]), ("v1 sd", c[3], r["v1_sd"]), ("sd plan", c[4], r["sdd"]),
                           ("lo", lo, r["lo"]), ("hi", hi, r["hi"]), ("bar", c[6], r["bar"])):
            ok(_eq(p, v), f"noise h{c[0]} {c[1]} {name}: doc {p}, disk {v:.4f}")
    keys = ("plaus", "k", "mde_gate_5", "mde_gate_10", "mde_tint_10", "mde_tint_20", "mde_tint_40",
            "pow_gate_5", "pow_gate_10", "pow_tint_20", "pow_tint_40")
    for c in wide:
        r = rows[(int(c[0]), c[1])]
        for key, p in zip(keys, c[2:13]):
            ok(_eq(p, r[key]), f"power h{c[0]} {c[1]} {key}: doc {p}, disk {r[key]:.4f}")
        ok(int(c[13]) == r["n80"], f"power h{c[0]} {c[1]} n80: doc {c[13]}, disk {r['n80']}")

    # 2. seed-count worked examples
    ex = [_cells(l) for l in text.splitlines() if re.match(r"\| 0\.\d+ \| \d+ \| \d+ \| ", l)]
    ok(len(ex) == 5, f"worked-example table has {len(ex)} rows, want 5")
    for c in ex:
        sc = _seed_count(float(c[0]))
        ok(int(c[1]) == sc["n_z"], f"example s={c[0]} n_z: doc {c[1]}, disk {sc['n_z']}")
        ok(int(c[2]) == sc["n_t"], f"example s={c[0]} n_t: doc {c[2]}, disk {sc['n_t']}")
        ok(int(c[3].split(",")[0]) == sc["N"], f"example s={c[0]} N: doc {c[3]}, disk {sc['N']}")
        ok(("underpowered" in c[3]) == sc["under"], f"example s={c[0]} underpowered flag: doc '{c[3]}'")
        ok(_eq(c[4], sc["power"]), f"example s={c[0]} power: doc {c[4]}, disk {sc['power']:.4f}")
        hrs = (sc["N"] - 5) * (v1s + 3 * v2s) / 3600
        ok(_eq(c[5], hrs), f"example s={c[0]} stage-2 hours: doc {c[5]}, nominal {hrs:.4f}")

    # 3. options table
    opts = {5: ("pow_gate_5", 0, 15), 10: ("pow_gate_10", 5, 30), 40: ("pow_tint_40", 35, 120),
            134: ("pow_tint_134", 129, 402)}
    for n, (key, nv1, nv2) in opts.items():
        m = re.search(rf"\| {n} seeds, [^|]*\| ([^|]*) \| about ([\d.]+) (minutes|hours) \| ([\d.]+) \|", text)
        ok(m is not None, f"option row for {n} seeds not found")
        if not m:
            continue
        ok(f"{nv2} v2" in m.group(1) and (nv1 == 0 or f"{nv1} v1" in m.group(1)), f"option {n}: runs '{m.group(1)}'")
        t = nv1 * v1s + nv2 * v2s
        got = t / 60 if m.group(3) == "minutes" else t / 3600
        ok(_eq(m.group(2), got), f"option {n}: runtime doc {m.group(2)} {m.group(3)}, nominal {got:.3f}")
        ok(_eq(m.group(4), h3[key]), f"option {n}: chance doc {m.group(4)}, disk {h3[key]:.4f}")
    ok(h3["n80"] == 134 and h3["pow_tint_134"] >= 0.8, "134 seeds must be the first count at 80 percent")

    # 4. prose: (pattern, value, tolerance or None for printed precision)
    rn = d["runner"]
    ms = h3["arm_sds"]
    prose = [
        (r'with no true effect is called "better" ([\d.]+) percent', 100 * d["fp5"], None),
        (r"no-effect rate falls to ([\d.]+) percent", 100 * d["fp10"], None),
        (r"chance of finding the probe effect at h3 RMSE is ([\d.]+) percent", 100 * h3["pow_gate_5"], None),
        (r"against a ([\d.]+) percent chance of a false call", 100 * d["fp5"], None),
        (r"10 seeds are worse, not better \(([\d.]+) percent\)", 100 * h3["pow_gate_10"], None),
        (r"40 seeds find a ([\d.]+) percent change at h3 RMSE", h3["mde_tint_40"], None),
        (r"and (\d+) seeds are needed", h3["n80"], None),
        (r"needed to find the ([\d.]+) percent the probe implies", h3["plaus"], None),
        (r"delta is the probe-implied h3 RMSE change, ([\d.]+) percent", h3["plaus"], None),
        (r"The probe-implied h5 change is ([\d.]+) percent", h5["plaus"], None),
        (r"which would need about ([\d,]+) seeds", h5["n80"], 100),
        (r"real neighbours remove ([\d.]+) and [\d.]+ percent", d["share"][3], None),
        (r"real neighbours remove [\d.]+ and ([\d.]+) percent", d["share"][5], None),
        (r"COVID ([\d.]+) percent in", d["t5"]["input_incidence"], None),
        (r"percent in, ([\d.]+) percent out", d["t5"]["representation_h"], None),
        (r"one country, (\d+) train", d["origins"][0], None),
        (r"(\d+) validation and \d+ test origins", d["origins"][1], None),
        (r"validation and (\d+) test origins", d["origins"][2], None),
        (r"at h3 RMSE were ([\d.]+) \(gate-off\)", ms["gateoff"], None),
        (r"\(gate-off\) and ([\d.]+) \(D2\)", ms["shufadj"], None),
        (r"which are ([\d.]+) and [\d.]+ percent of v1's mean", 100 * ms["gateoff"] / h3["v1_mean"], None),
        (r"which are [\d.]+ and ([\d.]+) percent of v1's mean", 100 * ms["shufadj"] / h3["v1_mean"], None),
        (r"percent of v1's mean of ([\d.]+)", h3["v1_mean"], None),
        (r"summed over (\d+) gradient tensors", sm["grad_tensors_v2"], None),
        (r"gradient tensors instead of (\d+)", sm["grad_tensors_v1"], None),
        (r"measured a ([\d.]+) percent difference in the scored metrics after 3 epochs", 100 * sm["superset_raw_max_rel_diff"], None),
        (r"\(([\d.]+) percent apart without that", 100 * sm["superset_raw_max_rel_diff"], None),
        (r"own convolutions, ([\d.]+) percent after 3 epochs", 100 * sm["cudnn_flag_moves_v1_max_rel_diff"], None),
        (r"repeats disagreed at (\d+) of \d+ origins", sm["deviation_varying_index_add"], None),
        (r"repeats disagreed at \d+ of (\d+) origins", sm["deviation_origins_tested"], None),
        (r"the matrix product at (\d+)\)", sm["deviation_varying_matmul"], None),
        (r"Stage 1 is 15 v2 runs, about (\d+) minutes", 15 * v2s / 60, None),
        (r"Each added seed costs about ([\d.]+) minutes", (v1s + 3 * v2s) / 60, None),
        (r"stage 2 is about ([\d.]+) hours at the cap", 129 * (v1s + 3 * v2s) / 3600, None),
        (r"The branch adds ([\d,]+) parameters", d["branch_params"], None),
        (r"the sha256 of (\d+) code files", len(rn["code_files"]), None),
        (r"the (\w+) `models/` files the encoder uses", None, None),     # a word; checked below
        (r"is about ([\d.]+) hours, so 3 arms", d["dengue_5seed_hours"] / 5, None),
        (r"at 5 seeds is about (\d+) hours", 3 * d["dengue_5seed_hours"], None),
        (r"\(the probe effect is ([\d.]+) to [\d.]+ paired sds\)", min(d["dengue_k_short"]), None),
        (r"\(the probe effect is [\d.]+ to ([\d.]+) paired sds\)", max(d["dengue_k_short"]), None),
        (r"somewhat better at h10 \(about ([\d.]+)\)", min(d["dengue_k_h10"]), None),
    ]
    words = {"five": 5, "six": 6, "seven": 7, "eight": 8}
    for pat, v, tol in prose:
        m = re.search(pat, flat)
        ok(m is not None, f"prose not found: {pat}")
        if not m:
            continue
        p = m.group(1).rstrip(".")                       # a number that ends a sentence
        if v is None:                                    # the models/ count, written as a word
            n_models = sum(f.startswith("models/") for f in rn["code_files"])
            ok(words.get(p) == n_models, f"prose '{p} models/ files': runner hashes {n_models}")
        elif tol is None:
            ok(_eq(p, v), f"prose {pat}: doc {p}, disk {v}")
        else:
            ok(abs(float(p.replace(",", "")) - v) <= tol, f"prose {pat}: doc {p}, disk {v:.3f} (tol {tol})")
    ok("every COVID RMSE and MAE cell is within noise" in flat and d["d2_all_noise"], "D2 COVID claim")
    ok(sm["superset_equalised_identical"] and sm["v2_run_to_run_identical"], "smoke identity claims")
    ok(sm["branch_params"] == d["branch_params"], "branch params: smoke vs fresh build")

    # 5. runner constants against the protocol text
    ok(rn["seq"] == tuple(42 + 10 * i for i in range(134)) and "seed i = 42 + 10 i for i = 0 to 133" in flat
       and "..., 1372, 134 seeds" in flat, f"seed sequence {rn['seq'][:3]}..{rn['seq'][-1]} ({len(rn['seq'])})")
    ok(rn["stage1"] == 5 and rn["seq"][:5] == SEEDS and "Stage 1 uses the first 5" in flat, "stage-1 seeds")
    ok(rn["rule"] == "tint" and "paired 95 percent t-interval of d excludes zero" in flat, f"rule {rn['rule']}")
    ok(rn["delta"] == P_DELTA and f"delta = {P_DELTA}" in flat and _eq(f"{P_DELTA:.3f}", h3["plaus"] / 100),
       f"delta: runner {rn['delta']}, protocol {P_DELTA}, probe {h3['plaus'] / 100:.4f}")
    ok((rn["za"], rn["zb"]) == (P_ZA, P_ZB) and "ceil(((1.96 + 0.84) s / delta)^2)" in flat, "z values")
    ok(rn["target"] == P_TARGET and "reaches 0.80" in flat, "power target")
    ok((rn["nmin"], rn["ncap"]) == (P_MIN, P_CAP) and "n_z = max(5," in flat and "N = min(n_t, 134)" in flat,
       f"floor/cap {rn['nmin']}/{rn['ncap']}")
    if verbose:
        print(f"{'OK' if not fails else 'FAIL'}: {n_checks[0]} checks, {len(fails)} failures")
        for f in fails:
            print("  " + f)
    return fails


def mutate(text, d):
    """Every corruption must be caught: edits to the document, breaks in the inputs, and changed runner
    constants."""
    doc_mut = [
        ("noise mean", "| 3 | rmse | 5565.8 |", "| 3 | rmse | 5566.8 |"),
        ("noise range", "509.8 to 878.6", "509.8 to 887.6"),
        ("power cell", "| 0.106 | 0.030 | 0.178 | 0.325 | 134 |", "| 0.106 | 0.030 | 0.178 | 0.352 | 134 |"),
        ("n80", "| 0.106 | 0.030 | 0.178 | 0.325 | 134 |", "| 0.106 | 0.030 | 0.178 | 0.325 | 143 |"),
        ("prose power", "RMSE is 10.6 percent", "RMSE is 16.0 percent"),
        ("prose drift", "measured a 0.23 percent", "measured a 0.32 percent"),
        ("option runtime", "| about 2.5 hours |", "| about 1.5 hours |"),
        ("smoke range", "measured 55 to 70", "measured 58 to 70"),
        ("probe share", "remove 5.3 and 2.3 percent", "remove 5.3 and 3.2 percent"),
        ("dengue k", "0.07 to 0.19", "0.07 to 0.91"),
        ("manuscript note spread", "were 694.1", "were 649.1"),
        ("worked example N", "| 0.10 | 72 | 75 | 75 |", "| 0.10 | 72 | 75 | 57 |"),
        ("worked example underpowered flag", "| 134, underpowered |", "| 134 |"),
        ("delta", "delta = 0.033", "delta = 0.030"),
        ("seed sequence", "42 + 10 i for i = 0 to 133", "42 + 5 i for i = 0 to 133"),
        ("code-file count", "sha256 of 12 code files", "sha256 of 11 code files"),
    ]
    caught = 0
    for name, a, b in doc_mut:
        assert a in text, f"mutation anchor missing for '{name}': {a!r}"
        f = check(text.replace(a, b, 1), d, verbose=False)
        caught += bool(f)
        print(f"  {'caught' if f else 'MISSED'}: doc {name}")
    inputs = [("one v1 seed value +5%", dict(d, covid=power_rows("covid_us-states", {"v1": (3, "rmse", 52, 1.05)}))),
              ("probe mse_A h3 +10%", dict(d, covid=power_rows("covid_us-states", {"mse_A": (3, 1.10)}))),
              ("runner RULE = gate", dict(d, runner=dict(d["runner"], rule="gate"))),
              ("runner N_CAP = 100", dict(d, runner=dict(d["runner"], ncap=100))),
              ("runner DELTA = 0.05", dict(d, runner=dict(d["runner"], delta=0.05)))]
    for name, dd in inputs:
        f = check(text, dd, verbose=False)
        caught += bool(f)
        print(f"  {'caught' if f else 'MISSED'}: input {name}")
    total = len(doc_mut) + len(inputs)
    print(f"mutate: {caught} of {total} corruptions caught")
    return caught == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    text = DOC.read_text(encoding="utf-8")
    d = disk()
    fails = check(text, d)
    if a.mutate:
        return 0 if (mutate(text, d) and not fails) else 1
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

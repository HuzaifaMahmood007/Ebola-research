"""verify_epi_bound_doc.py -- read the numbers back OUT of Epi_Bound_Lambda_2026-09-28.md and
recompute every one of them from the records on disk with independent code.

The point is to catch a document that has drifted from its artifacts: a verdict label that no longer
follows from the rule, a share taken from the wrong seed aggregate, a CI bound moved between rows, a
false-positive rate computed against the wrong bound. So this deliberately does NOT import
`ablation.run_epi_ablation`: not its record readers, not its paired delta, not its report. Every value
is re-derived straight from the JSON records with a local paired t-interval. `ablation.epi_penalty` is
imported for `observed_rates`, `calibrate` and `GAPS` only, and only for the false-positive table; that
is the penalty module, not the runner under test. The runner's report() is read as SOURCE TEXT.

Checks, one group each:
  1. header: the protocol sha256 re-hashed and equal to the doc's digest and to the blob in the quoted
     commit, which predates every new record; all 90 records exist and every record carries the sha;
     share = lam * penalty / pinball on every record; 18 units, 6 FAIL, 12 INCONCLUSIVE; the six units
     named in prose are exactly the six FAIL units, each at or over the gate.
  2. "What ran": cells per arm, the log's "done in" count, the archive inventory.
  3. the 18-row verdict table: share, win horizons, harm cells, verdict re-derived from the rule.
  4. the 144-cell tally, the better cells and the COVID h15 MAE interval quoted in prose.
  5. the Japan h10 table, the baseline mean and the "about 12 percent".
  6. the per-seed table, including the released p99max lambda 1 control row.
  7. the false-positive table and "one real transition in ten to one in twelve".
  8. the US panels and the deviations: seed aggregation under min and max, the 0.904 figure, report()
     still on the old rule and with no completeness guard, and the Run 2 bullet (the 3.244 to 30.654
     FAIL-unit range, p99max COVID at 0.762, the released p90max lambda 1 Japan h10 interval).
  9. the V2 repro claim and the trainer-drift argument: the repro hashed the trainer at 39ecf34, and
     the diff to b24063f adds 7 lines, every changed line inside `if epi and train_mode:`.

    conda run --no-capture-output -n ebola-train python diagnostics/verify_epi_bound_doc.py
    conda run --no-capture-output -n ebola-train python diagnostics/verify_epi_bound_doc.py --mutate
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import datetime as dt
import functools
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DOC = ROOT / "progress" / "outcomes" / "Epi_Bound_Lambda_2026-09-28.md"
PROTOCOL = ROOT / "progress" / "decisions" / "Epi_Bound_Lambda_Protocol.md"
PROTOCOL_GIT = "progress/decisions/Epi_Bound_Lambda_Protocol.md"
RUNNER = ROOT / "ablation" / "run_epi_ablation.py"
LOG = ROOT / "results" / "reports" / "epi_bound_lambda.log"
LOG_P99MAX = ROOT / "results" / "reports" / "epi_p99max.log"
REPRO = ROOT / "ablation" / "misc" / "v2_repro_check.json"
NEW = ROOT / "experiments" / "epi_bound_lambda" / "single"
ABL = ROOT / "ablation" / "single"
RES = ROOT / "results" / "single"

SEEDS = (42, 52, 62, 72, 82)
HORIZONS = (3, 5, 10, 15)
METRICS = ("rmse", "mae")                  # the deciding metrics; PCC decides nothing
FIELD = "node_mean"
MODEL = "encoder"
GATE = 0.01
JP, UR, US, CV = "influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states"
PANELS = (JP, UR, US, CV)
# doc label -> (record tag, panels). Lambda is read from the records and checked against the tag.
ARMS = {"p90max lam10": ("epi_p90max_lam10", PANELS),
        "p90max lam100": ("epi_p90max_lam100", PANELS),
        "p99median lam1": ("epi_p99median", PANELS),
        "p99median lam100": ("epi_p99median_lam100", PANELS),
        "p99max lam10": ("epi_p99max_lam10", (CV,)),
        "p99max lam100": ("epi_p99max_lam100", (CV,))}
RELEASED = {"p90max lam1": "epi_p90max", "p99max lam1": "epi_p99max"}
SHORT = {"japan": JP, "Japan": JP, "us-regions": UR, "us-states": US, "covid": CV, "COVID": CV}
FP_BOUNDS = ((0.90, "max"), (0.99, "median"))
WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
LABEL = r"p\d\d(?:max|median) lam\d+"


# --------------------------------------------------------------------------- #
# Disk side, read once, independently of the runner
# --------------------------------------------------------------------------- #
def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args, binary=False):
    out = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True)
    if out.returncode != 0:
        return None
    return out.stdout if binary else out.stdout.decode("utf-8", "replace").strip()


@functools.lru_cache(maxsize=None)
def _commit(h):
    """(full hash, commit unix time, sha256 of the protocol blob in it or None). None if unresolved."""
    full = _git("rev-parse", "--verify", "--quiet", f"{h}^{{commit}}")
    if not full:
        return None
    ct = int(_git("log", "-1", "--format=%ct", full))
    blob = _git("show", f"{full}:{PROTOCOL_GIT}", binary=True)
    return full, ct, (hashlib.sha256(blob).hexdigest() if blob is not None else None)


@functools.lru_cache(maxsize=None)
def _blob(commit, path):
    """Bytes of `path` at `commit`, or None when either does not resolve."""
    return _git("show", f"{commit}:{path}", binary=True)


@functools.lru_cache(maxsize=None)
def _diff(c1, c2, path):
    """(added, removed, removed line numbers in c1, added line numbers in c2) from `git diff -U0`."""
    ns = _git("diff", "--numstat", c1, c2, "--", path)
    out = _git("diff", "-U0", c1, c2, "--", path)
    if ns is None or out is None:
        return None
    add, rem = (int(x) for x in ns.split()[:2]) if ns else (0, 0)
    old_ln, new_ln = [], []
    for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", out, re.M):
        a, b = int(m.group(1)), int(m.group(2) or 1)
        c, n = int(m.group(3)), int(m.group(4) or 1)
        old_ln += range(a, a + b)
        new_ln += range(c, c + n)
    return add, rem, old_ln, new_ln


def _block(src, head):
    """1-based (first, last) body lines of the one statement whose stripped text is `head`, by
    indentation. None when `head` is absent or not unique."""
    lines = src.splitlines()
    hits = [j for j, ln in enumerate(lines) if ln.strip() == head]
    if len(hits) != 1:
        return None
    i = hits[0]
    ind = len(lines[i]) - len(lines[i].lstrip())
    last = i
    for j in range(i + 1, len(lines)):
        if lines[j].strip():
            if len(lines[j]) - len(lines[j].lstrip()) <= ind:
                break
            last = j
    return i + 2, last + 1


@functools.lru_cache(maxsize=None)
def _load():
    """Everything read from disk, once. disk() derives from a copy, so input mutations never leak."""
    import bundles
    from ablation.epi_penalty import GAPS, observed_rates
    r = {"new": {}, "base": {}, "rel": {}}
    for tag, panels in ARMS.values():
        for ds in panels:
            for s in SEEDS:
                p = NEW / f"encoder__{ds}__seed{s}__{tag}.json"
                if p.exists():
                    r["new"][(tag, ds, s)] = _read(p)
    for ds in PANELS:
        for s in SEEDS:
            r["base"][(ds, s)] = _read(RES / f"encoder__{ds}__seed{s}.json")
            for tag in RELEASED.values():
                r["rel"][(tag, ds, s)] = _read(ABL / f"encoder__{ds}__seed{s}__{tag}.json")
    r["inv_json"] = sorted(p.name for p in NEW.glob("*.json"))
    r["inv_npz"] = sorted(p.name for p in NEW.glob("*__pernode.npz"))
    r["mtime_new_min"] = min(p.stat().st_mtime for p in NEW.glob("*.json"))
    r["mtime_rel_max"] = max((ABL / f"encoder__{ds}__seed{s}__{t}.json").stat().st_mtime
                             for ds in PANELS for s in SEEDS for t in RELEASED.values())
    r["mtime_base_jp"] = {s: (RES / f"encoder__{JP}__seed{s}.json").stat().st_mtime for s in SEEDS}
    r["proto_sha"] = _sha(PROTOCOL)
    r["proto_text"] = PROTOCOL.read_text(encoding="utf-8")
    r["loop_now"] = _sha(ROOT / "train" / "loop.py")
    r["log"] = LOG.read_text(encoding="utf-8", errors="replace")
    r["log_p99max"] = LOG_P99MAX.read_text(encoding="utf-8", errors="replace")
    r["runner"] = RUNNER.read_text(encoding="utf-8")
    r["repro"] = _read(REPRO)
    r["code_now"] = {f: (_sha(ROOT / f) if (ROOT / f).exists() else None)
                     for f in r["repro"].get("code_sha256", {})}
    # unix times of every commit that touched the trainer, for "train/loop.py has changed since"
    r["loop_commits"] = [int(x) for x in (_git("log", "--format=%ct", "--", "train/loop.py") or "").split()]
    r["dev"] = tuple(bundles.DEV_BUNDLE_NAMES)
    r["gaps"] = tuple(GAPS)
    r["rates"] = {n: observed_rates(n) for n in r["dev"]}
    return r


def _enc(recs):
    """{(horizon, metric): node_mean} over the encoder records of one file."""
    return {(x["horizon"], x["metric"]): x[FIELD] for x in recs if x["model"] == MODEL}


def _lam_of(tag):
    m = re.search(r"_lam(\d+)$", tag)
    return float(m.group(1)) if m else 1.0


def paired(arm, ref):
    """(d mean, lo, hi, verdict, {seed: d}) for d = arm - ref per shared seed. Lower is better."""
    seeds = sorted(set(arm) & set(ref))
    d = {s: arm[s] - ref[s] for s in seeds}
    n = len(d)
    dm = sum(d.values()) / n
    sd = math.sqrt(sum((x - dm) ** 2 for x in d.values()) / (n - 1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    lo, hi = dm - half, dm + half
    return dm, lo, hi, ("worse" if lo > 0 else ("better" if hi < 0 else "noise")), d


def rule(cells, share):
    """Protocol sections 6 and 7, restated here and nowhere imported."""
    win = [h for h in HORIZONS if all(cells[(h, m)][3] == "better" for m in METRICS)]
    harm = [(h, m) for h in HORIZONS for m in METRICS if cells[(h, m)][3] == "worse"]
    if win and not harm and share >= GATE:
        return "PASS"
    return "FAIL" if share >= GATE else "INCONCLUSIVE"


def _apply(r, kind, arg):
    """One input perturbation for --mutate. Asserts it landed, so a no-op cannot pass as caught."""
    if kind in ("rec", "base"):
        key, h, m, delta = arg
        recs = r["new" if kind == "rec" else "base"][key]
        hit = [x for x in recs if x["model"] == MODEL and x["horizon"] == h and x["metric"] == m]
        assert len(hit) == 1, f"{kind} perturbation missed {key} h{h} {m}"
        hit[0][FIELD] += delta
    elif kind == "share":                                  # the penalty moves, so the share moves
        tag, ds, f = arg
        keys = [(tag, ds, s) for s in SEEDS if (tag, ds, s) in r["new"]]
        assert keys, f"share perturbation missed {tag} {ds}"
        for k in keys:
            for x in r["new"][k]:
                x["epi_penalty_mean"] *= f
                x["epi_penalty_share"] *= f
    elif kind == "sha":
        assert arg in r["new"], f"sha perturbation missed {arg}"
        r["new"][arg][0]["epi_protocol_sha256"] = "0" * 64
    elif kind == "drop":
        assert arg in r["new"], f"drop perturbation missed {arg}"
        del r["new"][arg]
        name = f"encoder__{arg[1]}__seed{arg[2]}__{arg[0]}.json"
        r["inv_json"] = [n for n in r["inv_json"] if n != name]
    elif kind == "rate":
        ds, g, f = arg
        assert ds in r["rates"] and g in r["rates"][ds], f"rate perturbation missed {ds} g{g}"
        r["rates"] = dict(r["rates"])
        r["rates"][ds] = {gg: (a * f if gg == g else a) for gg, a in r["rates"][ds].items()}
    elif kind == "repro":
        assert all(k in r["repro"] for k in arg), f"repro perturbation missed {list(arg)}"
        r["repro"].update(arg)
    elif kind == "runner":
        old, new = arg
        assert old in r["runner"], f"runner perturbation anchor {old!r} missing"
        r["runner"] = r["runner"].replace(old, new)
    elif kind == "log_drop":
        lines = r["log"].splitlines()
        i = next(j for j, ln in enumerate(lines) if " done in " in ln)
        r["log"] = "\n".join(lines[:i] + lines[i + 1:])
    else:
        raise ValueError(kind)


def disk(ov=None):
    """Every statistic the checks compare against. `ov` perturbs the inputs, for --mutate only."""
    from ablation.epi_penalty import calibrate
    src = _load()
    r = {k: (v if k == "rates" else copy.deepcopy(v)) for k, v in src.items()}
    for kind, arg in (ov or ()):
        _apply(r, kind, arg)
    d = {k: r[k] for k in ("proto_sha", "proto_text", "loop_now", "inv_json", "inv_npz",
                           "mtime_new_min", "mtime_rel_max", "mtime_base_jp", "runner", "repro",
                           "code_now", "loop_commits", "dev", "gaps")}
    d["new_keys"] = set(r["new"])
    d["tag_panels"] = {}
    for tag, ds, _s in r["new"]:
        d["tag_panels"].setdefault(tag, set()).add(ds)
    d["tag_cells"] = {tag: sum(1 for k in r["new"] if k[0] == tag) for tag, _ in ARMS.values()}

    # per-record integrity: sha on EVERY record (encoder_mc too), one share per file, formula, lambda
    d["sha_bad"], d["share_multi"], d["formula_bad"], d["lam_bad"] = [], [], [], []
    for k, recs in r["new"].items():
        nbad = sum(1 for x in recs if x.get("epi_protocol_sha256") != r["proto_sha"])
        if nbad:
            d["sha_bad"].append((k, nbad, len(recs)))
        if len({x.get("epi_penalty_share") for x in recs}) != 1:
            d["share_multi"].append(k)
        for x in recs:
            want = x["epi_lam"] * x["epi_penalty_mean"] / x["epi_pinball_mean"]
            if abs(x["epi_penalty_share"] - want) > 1e-9 * max(1.0, abs(want)):
                d["formula_bad"].append(k)
                break
        if any(float(x["epi_lam"]) != _lam_of(k[0]) for x in recs):
            d["lam_bad"].append(k)
    d["stamped_rmax"] = {tag: {int(g): v for g, v in recs[0]["epi_r_max"].items()}
                         for (tag, _ds, _s), recs in r["new"].items()}

    base = {k: _enc(v) for k, v in r["base"].items()}
    d["base_jp"] = {s: base[(JP, s)] for s in SEEDS}
    units = {}
    for label, (tag, panels) in ARMS.items():
        for ds in panels:
            arm = {s: _enc(r["new"][(tag, ds, s)]) for s in SEEDS if (tag, ds, s) in r["new"]}
            if len(arm) < 2:
                continue
            cells = {(h, m): paired({s: arm[s][(h, m)] for s in arm},
                                    {s: base[(ds, s)][(h, m)] for s in SEEDS})
                     for h in HORIZONS for m in METRICS}
            sh = [r["new"][(tag, ds, s)][0]["epi_penalty_share"] for s in arm]
            share = {"mean": sum(sh) / len(sh), "min": min(sh), "max": max(sh)}
            units[(label, ds)] = dict(tag=tag, lam=_lam_of(tag), cells=cells, share=share,
                                      verdict={a: rule(cells, share[a]) for a in share})
    d["units"] = units
    d["rel"] = {}
    for label, tag in RELEASED.items():
        for ds in PANELS:
            arm = {s: _enc(r["rel"][(tag, ds, s)]) for s in SEEDS}
            d["rel"][(label, ds)] = {(h, m): paired({s: arm[s][(h, m)] for s in SEEDS},
                                                    {s: base[(ds, s)][(h, m)] for s in SEEDS})
                                     for h in HORIZONS for m in METRICS}
    d["rel_has_share"] = sorted(k for k, recs in r["rel"].items()
                                if any("epi_penalty_share" in x for x in recs))

    # false positives: fraction of each panel's real TRAINING transitions above the bound, gaps pooled
    d["fp"], d["rmax"] = {}, {}
    for q, agg in FP_BOUNDS:
        rm, _ = calibrate(list(r["dev"]), q, verbose=False, aggregator=agg, rates=r["rates"])
        d["rmax"][(q, agg)] = rm
        d["fp"][(q, agg)] = {}
        for ds in PANELS:
            v = sum(int((r["rates"][ds][g] > rm[g]).sum()) for g in r["gaps"])
            t = sum(len(r["rates"][ds][g]) for g in r["gaps"])
            d["fp"][(q, agg)][ds] = 100.0 * v / t

    d["log_done"] = sum(1 for ln in r["log"].splitlines() if re.search(r" done in [\d.]+ min", ln))
    d["log_err"] = sum(1 for ln in r["log"].splitlines()
                       if re.search(r"\b(error|traceback|exception)\b", ln, re.I))
    pen = [float(m.group(1)) for m in re.finditer(
        rf"^\s+{JP} seed\d+ done in [\d.]+ min\s+mean train penalty = ([\d.e+-]+)", r["log_p99max"], re.M)]
    d["p99max_jp_pen"] = (sum(pen) / len(pen), len(pen)) if pen else (float("nan"), 0)
    return d


# --------------------------------------------------------------------------- #
# Parsing the document
# --------------------------------------------------------------------------- #
class P(list):
    """A group's problem strings, plus how many checks produced them."""
    n = 0

    def ok(self, cond, msg):
        self.n += 1
        if not cond:
            self.append(msg)
        return bool(cond)


def _flat(text):
    return re.sub(r"\s+", " ", text)


def _num(x):
    return float(str(x).replace(",", "").replace("+", "").strip())


def _tol(printed):
    s = str(printed).replace(",", "").replace("+", "").replace("-", "").strip()
    return 0.5 * (10 ** -len(s.split(".")[1]) if "." in s else 1.0)


def _close(printed, value):
    return abs(_num(printed) - value) <= _tol(printed) + 1e-9


def _unit(short_panel, bound, lam):
    """('us-regions', 'p90max', '100') -> ('p90max lam100', 'influenza_us-regions')."""
    return f"{bound} lam{lam}", SHORT[short_panel]


def _cells_list(s, pat):
    s = s.strip()
    if s == "none":
        return set(), []
    items = [x.strip() for x in s.split(",")]
    got = [re.fullmatch(pat, x) for x in items]
    return {m.groups() if len(m.groups()) > 1 else m.group(1) for m in got if m}, \
           [x for x, m in zip(items, got) if not m]


# | `epi_p90max_lam10` | 4 | 20 |
RAN_RE = re.compile(r"^\|\s*`(epi_\w+)`\s*\|\s*(\d+|COVID only)\s*\|\s*(\d+)\s*\|\s*$")
# | p90max lam10 | influenza_japan | 3.670 % | none | h10 RMSE, h10 MAE | FAIL |
VERDICT_RE = re.compile(rf"^\|\s*({LABEL})\s*\|\s*([\w-]+)\s*\|\s*([\d.]+) %\s*\|\s*([^|]+?)\s*\|"
                        r"\s*([^|]+?)\s*\|\s*([A-Z]+)\s*\|\s*$")
# | p99median lam1 | 0.282 % | +47.095 [+8.535, +85.655] | +19.377 [+4.144, +34.610] |
JH10_RE = re.compile(rf"^\|\s*({LABEL})\s*\|\s*([\d.]+) %\s*\|"
                     r"\s*([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]\s*\|"
                     r"\s*([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]\s*\|\s*$")
# | p99max lam1, released, penalty about 1e-7 | -2.4 | -3.5 | -7.0 | +2.5 | +1.2 |
SEED_RE = re.compile(rf"^\|\s*({LABEL})([^|]*)\|" + r"\s*([+-][\d.]+)\s*\|" * 5 + r"\s*$")
# | p90 max | 9.97 % | 0.91 % | 1.49 % | 6.25 % |
FP_RE = re.compile(r"^\|\s*p(\d\d) (max|median|min)\s*\|" + r"\s*([\d.]+) %\s*\|" * 4 + r"\s*$")


def check_header(text, d):
    p, flat = P(), _flat(text)
    m = re.search(r"sha256 `([0-9a-f]{64})`", flat)
    if p.ok(m is not None, "no protocol sha256 found in the doc"):
        p.ok(m.group(1) == d["proto_sha"],
             f"protocol sha256 in doc {m.group(1)[:12]}.. != file {d['proto_sha'][:12]}..")
    m = re.search(r"commit `([0-9a-f]{7,40})`", flat)
    if p.ok(m is not None, "no commit hash found in the doc"):
        c = _commit(m.group(1))
        if p.ok(c is not None, f"commit {m.group(1)[:12]} in the doc does not resolve (git rev-parse)"):
            full, ct, blob = c
            p.ok(len(m.group(1)) < 40 or full == m.group(1), f"commit {m.group(1)} resolves to {full}")
            p.ok(blob == d["proto_sha"], f"protocol blob in commit {full[:10]} has sha "
                 f"{(blob or 'none')[:12]}.., file {d['proto_sha'][:12]}..")
            when = lambda t: dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
            p.ok(ct < d["mtime_new_min"], f"commit {full[:10]} at {when(ct)} is not before the first "
                 f"new record, written {when(d['mtime_new_min'])}")
            p.ok(d["mtime_rel_max"] < ct, f"a released lambda-1 record was written at "
                 f"{when(d['mtime_rel_max'])}, after the protocol commit {when(ct)}")

    expected = [(tag, ds, s) for tag, panels in ARMS.values() for ds in panels for s in SEEDS]
    missing = [k for k in expected if k not in d["new_keys"]]
    p.ok(not missing, f"{len(missing)} expected new record(s) missing on disk, e.g. {missing[:2]}")
    m = re.search(r"Every one of the (\d+) new records carries that sha256", flat)
    if p.ok(m is not None, "the '90 new records carry that sha256' sentence not found"):
        p.ok(int(m.group(1)) == len(expected) == len(d["new_keys"]),
             f"new records: doc {m.group(1)}, grid {len(expected)}, on disk {len(d['new_keys'])}")
    p.ok(not d["sha_bad"], f"records without the protocol sha256: {d['sha_bad'][:3]}")
    p.ok(not d["share_multi"], f"files whose records disagree on the share: {d['share_multi'][:3]}")
    p.ok(not d["formula_bad"], f"share != lam * penalty / pinball in {d['formula_bad'][:3]}")
    p.ok(not d["lam_bad"], f"epi_lam disagrees with the tag in {d['lam_bad'][:3]}")

    u = d["units"]
    tally = {v: sum(1 for x in u.values() if x["verdict"]["mean"] == v)
             for v in ("PASS", "FAIL", "INCONCLUSIVE")}
    m = re.search(r"no PASS in any of the (\d+) arm-panel units\.\*\* (\d+) FAIL, (\d+) INCONCLUSIVE", flat)
    if p.ok(m is not None, "the verdict headline sentence not found"):
        p.ok(int(m.group(1)) == len(u), f"units: doc {m.group(1)}, disk {len(u)}")
        p.ok(tally["PASS"] == 0, f"doc says no PASS, disk has {tally['PASS']}")
        p.ok(int(m.group(2)) == tally["FAIL"], f"FAIL count: doc {m.group(2)}, disk {tally['FAIL']}")
        p.ok(int(m.group(3)) == tally["INCONCLUSIVE"],
             f"INCONCLUSIVE count: doc {m.group(3)}, disk {tally['INCONCLUSIVE']}")
    m = re.search(r"On Japan it made h(\d+) significantly worse, in all (\w+) new arms", flat)
    if p.ok(m is not None, "the 'Japan h10 worse in all four new arms' sentence not found"):
        h = int(m.group(1))
        jp = {k: x for k, x in u.items() if k[1] == JP}
        p.ok(WORDS.get(m.group(2)) == len(jp), f"Japan arms: doc '{m.group(2)}', disk {len(jp)}")
        bad = [k[0] for k, x in jp.items() if not all(x["cells"][(h, mm)][3] == "worse" for mm in METRICS)]
        p.ok(not bad, f"Japan h{h} is not significantly worse on both metrics in {bad}")
    # "became a real part of the objective, the six FAIL units (Japan and COVID at p90max lambda 10 and
    # 100, and at p99median lambda 100), it did not improve accuracy"
    m = re.search(r"became a real part of the objective, the (\w+) FAIL units \(Japan and COVID at "
                  r"(p\d\d(?:max|median)) lambda (\d+) and (\d+), and at (p\d\d(?:max|median)) lambda (\d+)\), "
                  r"it did not improve accuracy", flat)
    if p.ok(m is not None, "the six-FAIL-units sentence not found"):
        named = {_unit(pan, b, lam) for pan in ("Japan", "COVID")
                 for b, lam in ((m.group(2), m.group(3)), (m.group(2), m.group(4)), (m.group(5), m.group(6)))}
        fail = {k for k, x in u.items() if x["verdict"]["mean"] == "FAIL"}
        p.ok(WORDS.get(m.group(1)) == len(named) == len(fail),
             f"FAIL units: doc '{m.group(1)}', named {len(named)}, disk {len(fail)}")
        p.ok(named == fail, f"named FAIL units {sorted(named)} != disk FAIL units {sorted(fail)}")
        weak = [(k, round(100 * u[k]["share"]["mean"], 3)) for k in named if k in u and u[k]["share"]["mean"] < GATE]
        p.ok(not weak, f"named units under the 1 percent gate: {weak}")
        won = [k for k in named if k in u and any(all(u[k]["cells"][(h, mm_)][3] == "better" for mm_ in METRICS)
                                                  for h in HORIZONS)]
        p.ok(not won, f"'did not improve accuracy', but these named units win on both metrics: {won}")
    m = re.search(r"t\(0\.975, 4\) = (\d+\.\d+)", flat)
    if p.ok(m is not None, "t critical value not found"):
        p.ok(_close(m.group(1), float(stats.t.ppf(0.975, 4))),
             f"t(0.975, 4): doc {m.group(1)}, scipy {stats.t.ppf(0.975, 4):.4f}")
    return p


def check_what_ran(text, d):
    p, flat = P(), _flat(text)
    m = re.search(r"(\w+) arms, (\d+) cells, seeds ((?:\d+, )+)every record", flat)
    if p.ok(m is not None, "the 'Six arms, 90 cells, seeds' sentence not found"):
        p.ok(WORDS.get(m.group(1).lower()) == len(ARMS), f"arms: doc '{m.group(1)}', grid {len(ARMS)}")
        p.ok(int(m.group(2)) == len(d["new_keys"]), f"cells: doc {m.group(2)}, disk {len(d['new_keys'])}")
        seeds = tuple(int(x) for x in m.group(3).strip(", ").split(","))
        p.ok(seeds == SEEDS, f"seeds: doc {seeds}, want {SEEDS}")
    rows = [RAN_RE.match(ln) for ln in text.splitlines()]
    rows = [mm for mm in rows if mm]
    p.ok(len(rows) == len(ARMS), f"'What ran' table: parsed {len(rows)} rows, expected {len(ARMS)}")
    total, seen = 0, set()
    for mm in rows:
        tag, pan, cells = mm.group(1), mm.group(2), int(mm.group(3))
        seen.add(tag)
        on_disk = d["tag_panels"].get(tag, set())
        if pan == "COVID only":
            p.ok(on_disk == {CV}, f"{tag}: doc 'COVID only', disk panels {sorted(on_disk)}")
        else:
            p.ok(int(pan) == len(on_disk), f"{tag}: doc {pan} panels, disk {len(on_disk)}")
        p.ok(cells == d["tag_cells"].get(tag, 0), f"{tag}: doc {cells} cells, disk {d['tag_cells'].get(tag, 0)}")
        total += cells
    p.ok(seen == {t for t, _ in ARMS.values()}, f"'What ran' covers {sorted(seen)}")
    p.ok(total == len(d["new_keys"]), f"'What ran' cells sum to {total}, disk {len(d['new_keys'])}")
    m = re.search(r'`results/reports/epi_bound_lambda\.log`: (\d+) "done in" lines, no errors', flat)
    if p.ok(m is not None, "the log sentence not found"):
        p.ok(int(m.group(1)) == d["log_done"], f"'done in' lines: doc {m.group(1)}, log {d['log_done']}")
        p.ok(d["log_err"] == 0, f"doc says no errors, log has {d['log_err']} error/traceback line(s)")
    m = re.search(r"(\d+) JSON records and (\d+) pernode archives", flat)
    if p.ok(m is not None, "the inventory sentence not found"):
        p.ok(int(m.group(1)) == len(d["inv_json"]), f"JSON: doc {m.group(1)}, disk {len(d['inv_json'])}")
        p.ok(int(m.group(2)) == len(d["inv_npz"]), f"pernode: doc {m.group(2)}, disk {len(d['inv_npz'])}")
        orphans = [n for n in d["inv_json"] if n.replace(".json", "__pernode.npz") not in d["inv_npz"]]
        p.ok(not orphans, f"records without a pernode archive: {orphans[:3]}")
    return p


def check_verdict_table(text, d):
    p = P()
    seen = []
    for ln in text.splitlines():
        mm = VERDICT_RE.match(ln)
        if not mm:
            continue
        key = (mm.group(1), mm.group(2))
        seen.append(key)
        x = d["units"].get(key)
        if not p.ok(x is not None, f"verdict table row {key} is not a unit on disk"):
            continue
        name = f"{key[0]} {key[1]}"
        p.ok(_close(mm.group(3), 100 * x["share"]["mean"]),
             f"{name}: share printed {mm.group(3)} %, disk {100 * x['share']['mean']:.3f} %")
        win, junk = _cells_list(mm.group(4), r"h(\d+)")
        p.ok(not junk, f"{name}: unparsed win item(s) {junk}")
        want = {str(h) for h in HORIZONS if all(x["cells"][(h, m)][3] == "better" for m in METRICS)}
        p.ok(win == want, f"{name}: win printed {sorted(win) or 'none'}, disk {sorted(want) or 'none'}")
        harm, junk = _cells_list(mm.group(5), r"h(\d+) (RMSE|MAE)")
        p.ok(not junk, f"{name}: unparsed harm item(s) {junk}")
        want = {(str(h), m.upper()) for h in HORIZONS for m in METRICS if x["cells"][(h, m)][3] == "worse"}
        p.ok(harm == want, f"{name}: harm printed {sorted(harm) or 'none'}, disk {sorted(want) or 'none'}")
        p.ok(mm.group(6) == x["verdict"]["mean"],
             f"{name}: verdict printed {mm.group(6)}, rule gives {x['verdict']['mean']}")
    p.ok(len(seen) == 18, f"verdict table: parsed {len(seen)} rows, expected 18")
    p.ok(len(set(seen)) == len(seen), "verdict table repeats a unit")
    p.ok(set(seen) == set(d["units"]), f"verdict table misses units {sorted(set(d['units']) - set(seen))}")
    return p


def check_tally(text, d):
    p, flat = P(), _flat(text)
    cells = [(k[0], k[1], h, m, v) for k, x in d["units"].items() for (h, m), v in x["cells"].items()]
    worse = [c for c in cells if c[4][3] == "worse"]
    better = [c for c in cells if c[4][3] == "better"]
    noise = [c for c in cells if c[4][3] == "noise"]
    m = re.search(r"Across the (\d+) deciding cells \(RMSE and MAE\), (\d+) are significantly worse, "
                  r"(\d+) significantly better, and (\d+) are within noise", flat)
    if p.ok(m is not None, "the 144-cell tally sentence not found"):
        for name, pr, got in (("deciding cells", m.group(1), len(cells)), ("worse", m.group(2), len(worse)),
                              ("better", m.group(3), len(better)), ("noise", m.group(4), len(noise))):
            p.ok(int(pr) == got, f"tally {name}: doc {pr}, disk {got}")
    m = re.search(r"All (\d+) worse cells are Japan h(\d+)", flat)
    if p.ok(m is not None, "the 'all 8 worse cells are Japan h10' sentence not found"):
        off = [c[:4] for c in worse if not (c[1] == JP and c[2] == int(m.group(2)))]
        p.ok(int(m.group(1)) == len(worse) and not off,
             f"worse cells: doc {m.group(1)} all Japan h{m.group(2)}, disk {len(worse)}, off-pattern {off}")
    m = re.search(rf"The (\d+) better cells never pair up on both metrics at one horizon: Japan h(\d+) MAE "
                  r"in (\w+) arms \(its RMSE interval spans zero each time\) and COVID h(\d+) RMSE at "
                  r"(p\d\d(?:max|median)) lambda (\d+) \(its MAE interval is \[([+-][\d.]+), ([+-][\d.]+)\]\)",
                  flat)
    if not p.ok(m is not None, "the better-cells sentence not found"):
        return p
    p.ok(int(m.group(1)) == len(better), f"better cells: doc {m.group(1)}, disk {len(better)}")
    pairs = [(k, h) for k, x in d["units"].items() for h in HORIZONS
             if all(x["cells"][(h, mm)][3] == "better" for mm in METRICS)]
    p.ok(not pairs, f"doc says the better cells never pair up, disk pairs at {pairs}")
    hj, hc = int(m.group(2)), int(m.group(4))
    cv_key = _unit("COVID", m.group(5), m.group(6))
    jp_b = [c for c in better if c[1] == JP]
    p.ok(all(c[2] == hj and c[3] == "mae" for c in jp_b) and len(jp_b) == WORDS.get(m.group(3)),
         f"Japan better cells: doc h{hj} MAE in '{m.group(3)}' arms, disk {[c[:4] for c in jp_b]}")
    for c in jp_b:
        _dm, lo, hi, _v, _ = d["units"][(c[0], JP)]["cells"][(c[2], "rmse")]
        p.ok(lo <= 0 <= hi, f"{c[0]} Japan h{c[2]} RMSE interval [{lo:+.3f}, {hi:+.3f}] does not span zero")
    cv_b = [c[:4] for c in better if c[1] != JP]
    p.ok(cv_b == [(cv_key[0], CV, hc, "rmse")], f"other better cells: doc {cv_key[0]} COVID h{hc} RMSE only, "
         f"disk {cv_b}")
    if cv_key in d["units"]:
        _dm, lo, hi, _v, _ = d["units"][cv_key]["cells"][(hc, "mae")]
        for name, pr, got in (("lo", m.group(7), lo), ("hi", m.group(8), hi)):
            p.ok(_close(pr, got), f"{cv_key[0]} COVID h{hc} MAE CI {name}: doc {pr}, disk {got:+.3f}")
    return p


def check_japan_h10(text, d):
    p, flat = P(), _flat(text)
    m = re.search(r"\| h(\d+) RMSE d \[95% CI\] \| h(\d+) MAE d \[95% CI\] \|", text)
    h = int(m.group(1)) if p.ok(m is not None and m.group(1) == m.group(2),
                                 "Japan table header not found or its horizons disagree") else 10
    seen = set()
    for ln in text.splitlines():
        mm = JH10_RE.match(ln)
        if not mm:
            continue
        label = mm.group(1)
        seen.add(label)
        x = d["units"].get((label, JP))
        if not p.ok(x is not None, f"Japan h{h} table row {label} is not a Japan unit"):
            continue
        p.ok(_close(mm.group(2), 100 * x["share"]["mean"]),
             f"{label} Japan share: printed {mm.group(2)} %, disk {100 * x['share']['mean']:.3f} %")
        for metric, off in (("rmse", 3), ("mae", 6)):
            dm, lo, hi, _v, _ = x["cells"][(h, metric)]
            for name, i, got in (("d", off, dm), ("CI lo", off + 1, lo), ("CI hi", off + 2, hi)):
                p.ok(_close(mm.group(i), got), f"{label} Japan h{h} {metric.upper()} {name}: printed "
                     f"{mm.group(i)}, disk {got:+.3f}")
    want = {k[0] for k in d["units"] if k[1] == JP}
    p.ok(seen == want, f"Japan h{h} table rows {sorted(seen)}, Japan units {sorted(want)}")
    m = re.search(r"The baseline h(\d+) RMSE is about ([\d,]+) averaged over seeds, so the largest damage "
                  r"is about (\d+) percent", flat)
    if p.ok(m is not None, "the baseline / 12 percent sentence not found"):
        hb = int(m.group(1))
        bm = sum(d["base_jp"][s][(hb, "rmse")] for s in SEEDS) / len(SEEDS)
        p.ok(_close(m.group(2), bm), f"Japan h{hb} baseline RMSE: doc about {m.group(2)}, disk {bm:.3f}")
        worst = max(x["cells"][(hb, "rmse")][0] for k, x in d["units"].items() if k[1] == JP)
        p.ok(_close(m.group(3), 100 * worst / bm),
             f"largest damage: doc about {m.group(3)} percent, disk {worst:+.3f} / {bm:.3f} = "
             f"{100 * worst / bm:.2f} percent")
    return p


def check_per_seed(text, d):
    p, flat = P(), _flat(text)
    m = re.search(r"^\|\s*arm\s*\|" + r"\s*s(\d+)\s*\|" * 5, text, re.M)
    order = tuple(int(x) for x in m.groups()) if m else SEEDS
    p.ok(m is not None and order == SEEDS, f"per-seed header seeds {order}, want {SEEDS}")
    m = re.search(r"\(Japan h(\d+) (RMSE|MAE), arm minus baseline, per seed\.\)", flat)
    h, metric = (int(m.group(1)), m.group(2).lower()) if p.ok(
        m is not None, "per-seed caption not found") else (10, "rmse")
    seen = []
    for ln in text.splitlines():
        mm = SEED_RE.match(ln)
        if not mm:
            continue
        label, extra = mm.group(1), mm.group(2)
        seen.append(label)
        if "released" in extra:
            cell = d["rel"].get((label, JP), {}).get((h, metric)) if label in RELEASED else None
            if not p.ok(cell is not None, f"per-seed released row {label} has no released arm"):
                continue
            pm = re.search(r"penalty about 1e-(\d+)", extra)
            mean, n = d["p99max_jp_pen"]
            if p.ok(pm is not None and label == "p99max lam1" and n == len(SEEDS),
                    f"released row {label}: penalty claim or its {len(SEEDS)} log lines not found"):
                p.ok(-int(pm.group(1)) == math.floor(math.log10(mean)),
                     f"released p99max Japan penalty: doc about 1e-{pm.group(1)}, log mean {mean:.3e}")
            p.ok(cell[3] == "noise", f"control row {label} is not within noise at h{h} {metric}: {cell[3]}")
        else:
            x = d["units"].get((label, JP))
            if not p.ok(x is not None, f"per-seed row {label} is not a Japan unit"):
                continue
            cell = x["cells"][(h, metric)]
        for i, s in enumerate(order):
            got = cell[4].get(s)
            if p.ok(got is not None, f"{label} seed {s}: no paired value on disk"):
                p.ok(_close(mm.group(3 + i), got), f"{label} s{s}: printed {mm.group(3 + i)}, disk {got:+.3f}")
    want = {k[0] for k in d["units"] if k[1] == JP} | {"p99max lam1"}
    p.ok(len(seen) == 5 and set(seen) == want, f"per-seed table rows {seen}, want {sorted(want)}")
    return p


def check_false_positive(text, d):
    p, flat = P(), _flat(text)
    m = re.search(r"^\|\s*bound\s*\|" + r"\s*([\w-]+)\s*\|" * 4, text, re.M)
    cols = [SHORT.get(c) for c in m.groups()] if m else []
    if not p.ok(sorted(cols, key=str) == sorted(PANELS), f"false-positive header columns {cols}"):
        cols = list(PANELS)
    seen = set()
    for ln in text.splitlines():
        mm = FP_RE.match(ln)
        if not mm:
            continue
        b = (int(mm.group(1)) / 100.0, mm.group(2))
        seen.add(b)
        if not p.ok(b in d["fp"], f"false-positive row p{mm.group(1)} {mm.group(2)} is not a tested bound"):
            continue
        for i, ds in enumerate(cols):
            p.ok(_close(mm.group(3 + i), d["fp"][b][ds]), f"false positives p{mm.group(1)} {mm.group(2)} "
                 f"{ds}: printed {mm.group(3 + i)} %, disk {d['fp'][b][ds]:.3f} %")
    p.ok(seen == set(FP_BOUNDS), f"false-positive table rows {sorted(seen)}, want {sorted(FP_BOUNDS)}")
    # the table's bound must be the bound the arms actually trained with, as stamped in the records
    for b, tags in (((0.90, "max"), ("epi_p90max_lam10", "epi_p90max_lam100")),
                    ((0.99, "median"), ("epi_p99median", "epi_p99median_lam100"))):
        for tag in tags:
            st = d["stamped_rmax"].get(tag, {})
            p.ok(st and all(abs(st[g] - d["rmax"][b][g]) < 5e-7 for g in d["gaps"]),
                 f"{tag} records stamp r_max {st}, recalibrated {d['rmax'][b]}")
    p.ok(len(d["dev"]) == 5, f"the bound is calibrated over {len(d['dev'])} dev datasets, not five")
    m = re.search(r"all (\w+) gaps pooled", flat)
    p.ok(m is not None and WORDS.get(m.group(1)) == len(d["gaps"]),
         f"gaps pooled: doc '{m and m.group(1)}', module {len(d['gaps'])}")
    m = re.search(r"pushes against one real transition in (\w+) to one in (\w+)", flat)
    if p.ok(m is not None and m.group(1) in WORDS and m.group(2) in WORDS,
            "the 'one real transition in ten to one in twelve' sentence not found"):
        # the damage arms use both bounds, so the range runs from the looser-rate bound to the tighter;
        # each end must be 1 / (Japan's rate at one bound), rounded, lowest N first
        ns = sorted(round(100.0 / d["fp"][b][JP]) for b in FP_BOUNDS)
        doc = [WORDS[m.group(1)], WORDS[m.group(2)]]
        p.ok(doc == ns, f"'one in {m.group(1)} to one in {m.group(2)}': disk gives one in {ns[0]} to one in "
             f"{ns[1]} (" + ", ".join(f"p{int(b[0] * 100)} {b[1]} {d['fp'][b][JP]:.3f} % = 1 in "
                                      f"{100 / d['fp'][b][JP]:.2f}" for b in FP_BOUNDS) + ")")
    for b in FP_BOUNDS:
        us = max(d["fp"][b][ds] for ds in (UR, US))
        hot = min(d["fp"][b][ds] for ds in (JP, CV))
        p.ok(us < hot, f"p{int(b[0] * 100)} {b[1]}: a US panel ({us:.2f} %) is not below Japan and COVID "
             f"({hot:.2f} %), so 'the bound almost never binds' on the US panels fails")
    return p


def check_us_and_deviations(text, d):
    p, flat = P(), _flat(text)
    u = d["units"]
    usu = {k: x for k, x in u.items() if k[1] in (UR, US)}
    m = re.search(rf"never reached the (\d+) percent gate, even at lambda (\d+) \(highest seed-averaged "
                  r"share ([\d.]+) percent, (us-regions|us-states) at (p\d\d(?:max|median))\)", flat)
    if p.ok(m is not None, "the US-panel gate sentence not found"):
        p.ok(float(m.group(1)) / 100 == GATE, f"gate: doc {m.group(1)} percent, protocol {100 * GATE:g}")
        over = [k for k, x in usu.items() if x["share"]["mean"] >= GATE]
        p.ok(not over, f"US units at or over the gate: {over}")
        top = max(usu, key=lambda k: usu[k]["share"]["mean"])
        p.ok(_close(m.group(3), 100 * usu[top]["share"]["mean"]),
             f"highest US share: doc {m.group(3)} percent, disk {100 * usu[top]['share']['mean']:.3f} ({top})")
        p.ok(top[1] == SHORT[m.group(4)] and top[0].startswith(m.group(5)),
             f"highest US share: doc {m.group(4)} at {m.group(5)}, disk {top}")
    p.ok("They are INCONCLUSIVE in every arm" in flat, "the 'INCONCLUSIVE in every arm' sentence not found")
    notinc = [k for k, x in usu.items() if x["verdict"]["mean"] != "INCONCLUSIVE"]
    p.ok(not notinc, f"US units not INCONCLUSIVE: {notinc}")

    # deviation 1: report() really still prints only the old |mean d| < sd d rule
    src = d["runner"]
    mm = re.search(r"^def report\(.*?(?=^def )", src, re.S | re.M)
    if p.ok(mm is not None, "def report( not found in ablation/run_epi_ablation.py"):
        body = mm.group(0)
        p.ok("abs(dm) < dsd" in body, "report() no longer contains the old `abs(dm) < dsd` rule")
        p.ok(not re.search(r"ppf|2\.7764|t\.interval|stats\.t", body),
             "report() now computes a t-interval, so 'does not apply the pre-registered rule' is stale")
    p.ok("`|mean d| < sd d`" in flat, "the doc no longer quotes the old rule")
    # ... and has no completeness guard, though protocol section 9 promises one
    m = re.search(r"`--report` decides and refuses to decide on an incomplete arm\. It does neither: it still "
                  r"prints only the old `\|mean d\| < sd d` rule, and it does not check completeness, which had "
                  r"no effect here because all (\d+) records exist", flat)
    if p.ok(m is not None, "the deviation 1 completeness sentence not found"):
        p.ok(re.search(r"`--report` reads the records from disk and refuses to decide on an arm whose \d+ cells",
                       _flat(d["proto_text"])) is not None,
             "protocol section 9 no longer says --report refuses an incomplete arm")
        if mm is not None:
            guard = re.findall(r"\bassert\b|\braise\b|sys\.exit|refus|incomplete|L\.SEEDS|len\(\s*(?:abl|a|b)\b"
                               r"|\bn\s*(?:!=|<)\s*(?:5|20|len)", mm.group(0))
            p.ok(not guard, f"report() now carries a completeness guard: {guard}")
        expected = sum(len(panels) * len(SEEDS) for _t, panels in ARMS.values())
        p.ok(int(m.group(1)) == expected and d["new_keys"] >= {(t, ds, s) for t, pans in ARMS.values()
                                                               for ds in pans for s in SEEDS},
             f"'all {m.group(1)} records exist': grid {expected}, on disk {len(d['new_keys'])}")

    # deviation 2: seed aggregation of the share
    m = re.search(r"Under the minimum, no verdict changes\. Under the maximum, (\w+) units move from "
                  r"INCONCLUSIVE to FAIL: (us-regions|us-states|COVID|Japan) at (p\d\d(?:max|median)) lambda "
                  r"(\d+) \(max ([\d.]+) percent\) and (us-regions|us-states|COVID|Japan) at "
                  r"(p\d\d(?:max|median)) lambda (\d+) \(max ([\d.]+) percent\)\. Neither reading produces "
                  r"a PASS", flat)
    if p.ok(m is not None, "the seed-aggregation sentence not found"):
        chg_min = [k for k, x in u.items() if x["verdict"]["min"] != x["verdict"]["mean"]]
        p.ok(not chg_min, f"under the minimum these verdicts change: {chg_min}")
        chg_max = {k: (x["verdict"]["mean"], x["verdict"]["max"]) for k, x in u.items()
                   if x["verdict"]["max"] != x["verdict"]["mean"]}
        named = {_unit(m.group(2), m.group(3), m.group(4)): m.group(5),
                 _unit(m.group(6), m.group(7), m.group(8)): m.group(9)}
        p.ok(WORDS.get(m.group(1)) == len(chg_max), f"units flipping under max: doc '{m.group(1)}', "
             f"disk {len(chg_max)} {sorted(chg_max)}")
        p.ok(set(chg_max) == set(named), f"flipping units: doc {sorted(named)}, disk {sorted(chg_max)}")
        p.ok(all(v == ("INCONCLUSIVE", "FAIL") for v in chg_max.values()),
             f"flips under max are not all INCONCLUSIVE to FAIL: {chg_max}")
        for k, pr in named.items():
            if p.ok(k in u, f"named unit {k} not on disk"):
                p.ok(_close(pr, 100 * u[k]["share"]["max"]),
                     f"{k[0]} {k[1]} max share: doc {pr} percent, disk {100 * u[k]['share']['max']:.3f}")
        p.ok(not any("PASS" in (x["verdict"]["min"], x["verdict"]["max"]) for x in u.values()),
             "a PASS appears under the min or max reading")

    # deviation 3: p99median lambda 1 Japan
    m = re.search(r"Its share is ([\d.]+) percent, under the gate, and it still made h(\d+) significantly "
                  r"worse on both metrics", flat)
    x = u.get(("p99median lam1", JP))
    if p.ok(m is not None and x is not None, "the p99median lambda 1 Japan sentence not found"):
        p.ok(_close(m.group(1), 100 * x["share"]["mean"]) and x["share"]["mean"] < GATE,
             f"p99median lam1 Japan share: doc {m.group(1)} percent, disk {100 * x['share']['mean']:.3f}")
        h = int(m.group(2))
        p.ok(all(x["cells"][(h, mm_)][3] == "worse" for mm_ in METRICS) and x["verdict"]["mean"] == "INCONCLUSIVE",
             f"p99median lam1 Japan: h{h} {[x['cells'][(h, mm_)][3] for mm_ in METRICS]}, "
             f"verdict {x['verdict']['mean']}")

    # deviation 4: the released lambda-1 arms carry no share
    p.ok("The released lambda-1 arms at p90max and p99max carry no share" in flat,
         "the 'released arms carry no share' sentence not found")
    p.ok(not d["rel_has_share"], f"released records that DO carry a share: {d['rel_has_share'][:3]}")

    # "What this closes"
    # Run 2: the share range over the FAIL units, the p99max exception, and the Japan h10 harm
    m = re.search(r"real part of the objective in the (\w+) FAIL units, (\d+\.\d+) to (\d+\.\d+) percent", flat)
    if p.ok(m is not None, "the Run 2 FAIL-unit share range sentence not found"):
        vals = {k: 100 * x["share"]["mean"] for k, x in u.items() if x["verdict"]["mean"] == "FAIL"}
        p.ok(WORDS.get(m.group(1)) == len(vals), f"FAIL units: doc '{m.group(1)}', disk {len(vals)}")
        if vals:
            lo_k, hi_k = min(vals, key=vals.get), max(vals, key=vals.get)
            p.ok(_close(m.group(2), vals[lo_k]) and _close(m.group(3), vals[hi_k]),
                 f"FAIL-unit share range: doc {m.group(2)} to {m.group(3)} percent, disk "
                 f"{vals[lo_k]:.3f} ({lo_k[0]} {lo_k[1]}) to {vals[hi_k]:.3f} ({hi_k[0]} {hi_k[1]})")
    m = re.search(r"At p99max, COVID stayed under the gate even at lambda (\d+) \((\d+\.\d+) percent\)", flat)
    if p.ok(m is not None, "the p99max COVID sentence not found"):
        k = (f"p99max lam{m.group(1)}", CV)
        if p.ok(k in u, f"{k} is not a unit"):
            p.ok(_close(m.group(2), 100 * u[k]["share"]["mean"]),
                 f"p99max lam{m.group(1)} COVID share: doc {m.group(2)} percent, disk "
                 f"{100 * u[k]['share']['mean']:.3f}")
        over = [(q[0], round(100 * x["share"]["mean"], 3)) for q, x in u.items()
                if q[0].startswith("p99max") and q[1] == CV and x["share"]["mean"] >= GATE]
        p.ok(not over, f"'stayed under the gate', but p99max COVID units at or over it: {over}")
    m = re.search(r"On Japan it hurt h(\d+) in every new arm; the released lambda (\d+) arm at "
                  r"(p\d\d(?:max|median)), the base of that ladder, is within noise there \(RMSE d ([+-][\d.]+) "
                  r"\[([+-][\d.]+), ([+-][\d.]+)\]\)", flat)
    if p.ok(m is not None, "the Run 2 Japan h10 sentence not found"):
        h = int(m.group(1))
        clean = [k[0] for k, x in u.items() if k[1] == JP
                 and not any(x["cells"][(h, mm_)][3] == "worse" for mm_ in METRICS)]
        p.ok(not clean, f"Japan units with no significant h{h} harm: {clean}")
        rk = (f"{m.group(3)} lam{m.group(2)}", JP)
        if p.ok(rk in d["rel"], f"released arm {rk[0]} not found"):
            dm, lo, hi, _v, _ = d["rel"][rk][(h, "rmse")]
            for name, pr, got in (("d", m.group(4), dm), ("CI lo", m.group(5), lo), ("CI hi", m.group(6), hi)):
                p.ok(_close(pr, got), f"released {rk[0]} Japan h{h} RMSE {name}: doc {pr}, disk {got:+.3f}")
            live = [mm_ for mm_ in METRICS if d["rel"][rk][(h, mm_)][3] != "noise"]
            p.ok(not live, f"released {rk[0]} Japan h{h} is not within noise on {live}")
    m = re.search(r"At lambda 1 it was inert on (\w+) panels and harmful at Japan h(\d+)", flat)
    if p.ok(m is not None, "the Run 1 lambda 1 sentence not found"):
        others = {k: x for k, x in u.items() if k[0] == "p99median lam1" and k[1] != JP}
        p.ok(WORDS.get(m.group(1)) == len(others), f"inert panels: doc '{m.group(1)}', disk {len(others)}")
        live = [k[1] for k, x in others.items()
                if x["share"]["mean"] >= GATE or any(c[3] != "noise" for c in x["cells"].values())]
        p.ok(not live, f"p99median lam1 is not inert on {live}")
        jx = u.get(("p99median lam1", JP))
        p.ok(jx is not None and any(jx["cells"][(int(m.group(2)), mm_)][3] == "worse" for mm_ in METRICS),
             f"p99median lam1 is not harmful at Japan h{m.group(2)}")
    p.ok("the dengue arm is not run" in flat and not any("dengue" in n for n in d["inv_json"]),
         "a dengue record exists under experiments/epi_bound_lambda/single/")
    p.ok("No Ebola record was read or written" in flat and not any("ebola" in n for n in d["inv_json"]),
         "an ebola record exists under experiments/epi_bound_lambda/single/")
    return p


def check_repro(text, d):
    p, flat = P(), _flat(text)
    rp = d["repro"]
    code = rp.get("code_sha256", {})
    loop = "train/loop.py"
    m = re.search(r"\(`ablation/misc/v2_repro_check\.json`\) retrained Japan seed (\d+) with the trainer at "
                  r"commit `([0-9a-f]{7,40})` and matched the released record exactly, max relative difference "
                  r"(\d+(?:\.\d+)?)", flat)
    if p.ok(m is not None, "the V2 repro sentence not found"):
        p.ok(rp.get("identical") is True, f"repro check identical = {rp.get('identical')}")
        p.ok(rp.get("max_rel_diff") == float(m.group(3)),
             f"max relative difference: doc {m.group(3)}, file {rp.get('max_rel_diff')}")
        p.ok(rp.get("dataset") == JP, f"repro dataset {rp.get('dataset')}")
        p.ok(rp.get("seed") == int(m.group(1)), f"repro seed: doc {m.group(1)}, file {rp.get('seed')}")
        blob = _blob(m.group(2), loop)
        if p.ok(blob is not None, f"commit `{m.group(2)}` or its {loop} does not resolve"):
            at = hashlib.sha256(blob).hexdigest()
            p.ok(code.get(loop) == at, f"repro check hashed {loop} as {str(code.get(loop))[:12]}.., "
                 f"`{m.group(2)}` holds {at[:12]}..")
    m = re.search(r"The only change to `train/loop\.py` between `([0-9a-f]{7,40})` and the commit these arms ran "
                  r"under, `([0-9a-f]{7,40})`, is (\d+) lines inside `(if epi and train_mode:)` that accumulate "
                  r"the pinball loss for the share; the baseline path is untouched\.", flat)
    if p.ok(m is not None, "the 'only change to train/loop.py' sentence not found"):
        c1, c2, n, head = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        diff = _diff(c1, c2, loop)
        old, new = _blob(c1, loop), _blob(c2, loop)
        if p.ok(diff is not None and old is not None and new is not None,
                f"`git diff {c1} {c2} -- {loop}` does not resolve"):
            add, rem, old_ln, new_ln = diff
            # the doc's "7 lines" counts the lines of the new version that differ: git's insertions
            p.ok(add == n, f"lines added to {loop} between {c1} and {c2}: doc {n}, git {add} (removed {rem})")
            ob = _block(old.decode("utf-8"), head)
            nb = _block(new.decode("utf-8"), head)
            if p.ok(ob is not None and nb is not None, f"`{head}` is missing or not unique in {c1} or {c2}"):
                out_old = [x for x in old_ln if not ob[0] <= x <= ob[1]]
                out_new = [x for x in new_ln if not nb[0] <= x <= nb[1]]
                p.ok(not out_old and not out_new, f"changed lines outside `{head}`: {c1} {out_old}, {c2} {out_new}")
        # "the commit these arms ran under": the trainer on disk is the one at c2 and nothing touched it since
        at2 = hashlib.sha256(new).hexdigest() if new is not None else None
        p.ok(at2 == d["loop_now"], f"{loop} on disk {d['loop_now'][:12]}.. is not the one at `{c2}`")
        later = _git("log", "--format=%h", f"{c2}..HEAD", "--", loop)
        p.ok(later == "", f"commits after `{c2}` touched {loop}: {later}")
        # nothing else the repro check hashed drifted between the repro run and these arms
        other = [f for f, h in code.items() if f != loop and d["code_now"].get(f) != h]
        p.ok(not other, f"files other than {loop} differ from the repro check's hashes: {other}")
    p.ok("I confirmed that by reading the diff, not by retraining." in flat,
         "the 'reading the diff, not by retraining' disclosure is gone")
    m = re.search(r"By file date, (\w+) of the (\w+) baseline Japan records date from (\d{4}-\d{2}-\d{2})", flat)
    if p.ok(m is not None, "the baseline-date sentence not found"):
        days = {s: dt.date.fromtimestamp(t).isoformat() for s, t in d["mtime_base_jp"].items()}
        on = [s for s, day in days.items() if day == m.group(3)]
        p.ok(WORDS.get(m.group(1).lower()) == len(on) and WORDS.get(m.group(2)) == len(SEEDS),
             f"Japan baselines dated {m.group(3)} by file time: doc '{m.group(1)} of {m.group(2)}', disk {days}")
        if on:
            newest = max(d["mtime_base_jp"][s] for s in on)
            p.ok(any(t > newest for t in d["loop_commits"]),
                 "no commit touched train/loop.py after those baseline records were written")
    m = re.search(r"seed (\d+) shows the damage in every new arm", flat)
    if p.ok(m is not None, "the 'seed 42 shows the damage' sentence not found"):
        s = int(m.group(1))
        neg = [k[0] for k, x in d["units"].items() if k[1] == JP and not x["cells"][(10, "rmse")][4].get(s, 0) > 0]
        p.ok(not neg, f"seed {s} Japan h10 RMSE is not worse than baseline in {neg}")
    return p


CHECKS = (("header (sha, commit, records, verdict counts)", check_header),
          ("'What ran' table, log, inventory", check_what_ran),
          ("18-row verdict table", check_verdict_table),
          ("144-cell tally and better-cell prose", check_tally),
          ("Japan h10 table, baseline, 12 percent", check_japan_h10),
          ("per-seed table and control row", check_per_seed),
          ("false-positive table", check_false_positive),
          ("US panels and deviations", check_us_and_deviations),
          ("V2 repro and trainer drift", check_repro))


def run(text, d, verbose=True):
    all_problems, n = [], 0
    for name, fn in CHECKS:
        problems = fn(text, d)
        n += problems.n
        all_problems += problems
        if verbose:
            print(f"  {'FAIL' if problems else 'ok  '}  {name}  [{problems.n} checks]"
                  + (f"  ({len(problems)} problem(s))" if problems else ""))
            for pr in problems:
                print(f"        - {pr}")
    return all_problems, n


# --------------------------------------------------------------------------- #
def mutate(text, d):
    """Corrupt the document, then the inputs, one defect at a time; every corruption MUST be caught.

    Caught means a problem the unmutated run did not already report. Without that, a document that
    already fails would make every mutation look caught."""
    before = set(run(text, d, verbose=False)[0])
    edits = [
        ("verdict label FAIL -> INCONCLUSIVE",
         "| 3.670 % | none | h10 RMSE, h10 MAE | FAIL |", "| 3.670 % | none | h10 RMSE, h10 MAE | INCONCLUSIVE |"),
        ("verdict-table share", "| covid_us-states | 3.244 % |", "| covid_us-states | 3.424 % |"),
        ("verdict-table win horizon", "| influenza_us-regions | 0.124 % | none |",
         "| influenza_us-regions | 0.124 % | h3 |"),
        ("verdict-table harm cell", "| 16.548 % | none | h10 RMSE, h10 MAE |", "| 16.548 % | none | h10 RMSE, h15 MAE |"),
        ("Japan-table CI bound", "+17.560", "+17.650"),
        ("Japan-table d", "+109.065", "+109.056"),
        ("Japan-table share", "| p90max lam100 | 25.715 % | +99.732", "| p90max lam100 | 25.751 % | +99.732"),
        ("per-seed delta", "| +123.6 |", "| +132.6 |"),
        ("per-seed control delta", "| -7.0 |", "| -0.7 |"),
        ("false-positive percentage", "| 9.97 % |", "| 9.79 % |"),
        ("tally noise count", "132 are within noise", "123 are within noise"),
        ("tally worse/better swapped", "8 are significantly worse, 4 significantly better",
         "4 are significantly worse, 8 significantly better"),
        ("COVID h15 MAE interval", "[-811.475, +12.683]", "[-811.475, +12.863]"),
        ("protocol digest", "`458184bc25d376c7", "`548184bc25d376c7"),
        ("commit hash", "`b24063f16a691b50", "`b24036f16a691b50"),
        ("1.530 max share", "(max 1.530 percent)", "(max 1.350 percent)"),
        ("1.324 max share", "1.324 percent)", "1.342 percent)"),   # the doc wraps "(max" onto the line above
        ("0.904 US figure", "share 0.904 percent", "share 0.940 percent"),
        ("about 12 percent", "about 12 percent", "about 21 percent"),
        ("about 1,063 baseline", "about 1,063", "about 1,036"),
        ("What-ran panels label", "| `epi_p99max_lam100` | COVID only | 5 |", "| `epi_p99max_lam100` | 4 | 5 |"),
        ("verdict counts", "6 FAIL, 12 INCONCLUSIVE", "12 FAIL, 6 INCONCLUSIVE"),
        ("six FAIL units named", "and at p99median lambda 100), it did", "and at p99median lambda 1), it did"),
        ("one in ten to one in twelve", "ten to one in twelve", "ten to one in eleven"),
        ("repro max rel diff", "max relative difference 0.0.", "max relative difference 0.1."),
        ("repro commit", "commit `39ecf34` and matched", "commit `39ecf43` and matched"),
        ("diff base commit", "between `39ecf34` and the commit", "between `39ecf43` and the commit"),
        ("7 lines in the diff", "`b24063f`, is 7", "`b24063f`, is 8"),   # the doc wraps "lines inside" below
        ("all 90 records exist", "all 90 records exist", "all 80 records exist"),
        ("Run 2 share low end 3.244", "FAIL units, 3.244", "FAIL units, 3.424"),   # "to 30.654" wraps below
        ("Run 2 share high end 30.654", "to 30.654 percent", "to 30.564 percent"),
        ("p99max COVID 0.762", "(0.762 percent)", "(0.726 percent)"),
        ("released p90max d", "(RMSE d +19.668", "(RMSE d +19.686"),
        ("released p90max CI lo 19.442", "[-19.442,", "[-19.424,"),
    ]
    inputs = [
        ("record: p90max lam10 Japan s42 h10 RMSE +5", [("rec", (("epi_p90max_lam10", JP, 42), 10, "rmse", 5.0))]),
        ("record: us-regions p90max lam100 penalty x1.2", [("share", ("epi_p90max_lam100", UR, 1.2))]),
        ("baseline: Japan s82 h10 RMSE -10", [("base", ((JP, 82), 10, "rmse", -10.0))]),
        ("training rates: Japan gap 2 x1.05", [("rate", (JP, 2, 1.05))]),
        ("record: one protocol sha corrupted", [("sha", ("epi_p99median", UR, 62))]),
        ("record: COVID p99max lam100 s82 missing", [("drop", ("epi_p99max_lam100", CV, 82))]),
        ("repro check: identical false", [("repro", {"identical": False})]),
        ("runner: report() moved to a t-interval", [("runner", ("abs(dm) < dsd", "hi < 0 < lo"))]),
        ("log: one 'done in' line lost", [("log_drop", None)]),
        ("repro check: loop.py hash is the current trainer's",
         [("repro", {"code_sha256": dict(d["repro"]["code_sha256"], **{"train/loop.py": d["loop_now"]})})]),
        ("runner: report() gains a completeness guard",
         [("runner", ("        abl = load_abl(ds, tg, model)\n",
                      "        abl = load_abl(ds, tg, model)\n"
                      "        assert all(len(v) == 5 for v in abl.values()), 'incomplete arm'\n"))]),
    ]
    print(f"{'=' * 78}\nMutation test: {len(edits)} document and {len(inputs)} input corruptions, "
          f"each must be caught\n{'=' * 78}")
    escaped = []
    for name, old, new in edits:
        n_hit = text.count(old)
        bad = text.replace(old, new, 1)
        if bad == text or n_hit != 1:
            print(f"  NO-OP       doc {name}  (anchor found {n_hit} times, want exactly 1)")
            escaped.append(name)
            continue
        fresh = set(run(bad, d, verbose=False)[0]) - before
        print(f"  {'caught' if fresh else 'NOT CAUGHT':<11} doc {name}" + (f"  ({len(fresh)} new)" if fresh else ""))
        if not fresh:
            escaped.append(name)
    for name, ov in inputs:
        try:
            dd = disk(ov)
        except (AssertionError, StopIteration) as e:
            print(f"  NO-OP       input {name}  ({e})")
            escaped.append(name)
            continue
        fresh = set(run(text, dd, verbose=False)[0]) - before
        print(f"  {'caught' if fresh else 'NOT CAUGHT':<11} input {name}" + (f"  ({len(fresh)} new)" if fresh else ""))
        if not fresh:
            escaped.append(name)
    total = len(edits) + len(inputs)
    print(f"\nmutations caught: {total - len(escaped)} of {total}")
    if escaped:
        print(f"{len(escaped)} mutation(s) escaped. The verifier does not check what it claims to.")
    return escaped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc", nargs="?", default=str(DOC))
    ap.add_argument("--mutate", action="store_true", help="mutation-test the verifier itself")
    a = ap.parse_args()

    p = Path(a.doc)
    if not p.exists():
        sys.exit(f"no such document: {p}")
    text = p.read_text(encoding="utf-8")

    print(f"{'=' * 78}\nVerifying {p.name} against the records on disk\n{'=' * 78}")
    d = disk()
    problems, n = run(text, d)
    print(f"\n{n} checks, {len(problems)} failure(s)")

    if a.mutate:
        print()
        escaped = mutate(text, d)
        if escaped:
            sys.exit(2)

    if problems:
        print(f"\n{len(problems)} PROBLEM(S). The document does not match its artifacts.")
        sys.exit(1)
    print(f"\nOK: {p.name} matches the records on disk.")


if __name__ == "__main__":
    main()

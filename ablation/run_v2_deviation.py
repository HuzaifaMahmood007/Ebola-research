"""V2 deviation channel: a MECHANISM TEST. Does routing district deviations to the graph make the graph
contribute? Protocol, fixed before any COVID number: progress/decisions/V2_Deviation_Protocol.md.

User commands, in this order (each refuses if the one before it has not passed):
    conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --repro-check
    conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --stage1
    conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --seed-count
    conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --stage2
    conda run --no-capture-output -n ebola-train python ablation/run_v2_deviation.py --report
Checks (seconds to minutes, never COVID training):
    ... --selfcheck | --mutate-selfcheck | --power | --smoke | --stage1 --dry | --stage2 --dry

THE ARMS. v1 is the released single-disease run in results/single/, read-only, plus extra v1 seeds
trained in stage 2 as __v1ref records. The three v2 arms share models/encoder_v2.py and differ ONLY in the
adjacency the deviation branch mixes over:
  v2graph     the real map (the same adjacency v1 mixes over)
  v2nograph   the identity: same parameters and depth, but a district sees only its own deviation
  v2shuffled  a degree-preserving relabel of the real map (train.loop.permute_adjacency), seed
              SHUF_BASE + train seed, the same fake graphs the D2 retrain used
v1's own path mixes over the REAL map in every arm, so each comparison moves exactly one thing.

TWO STAGES. The repro-check must pass first (v1 on influenza_japan reproduces its released record,
tied to the committed protocol hash and to the code's hashes). Stage 1 trains the three arms at the 5
v1 seeds and reports ONLY validation statistics: the sd of paired per-seed differences in h3 RMSE on
the validation split, relative to v1's validation mean. --seed-count turns the largest of the three
into N with the protocol's formula. Stage 2 trains all four arms up to N seeds of the fixed sequence.
--report refuses to read any test record until every one of the N x 4 runs exists.

There is no other training path for COVID. --smoke trains influenza_japan only. Dengue and Ebola have
no path here at all.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import json
import math
import subprocess
import time

import numpy as np
import torch

import bundles
import train.loop as L
from ablation.run_epi_ablation import _seed_from, mean_sd, paired_delta
from ablation.run_shuffle_adjacency import perm_seed_for
from models import MEDIAN_IDX, node_indexed_params, sparse_from_dense_np, window_slice
from models.encoder_v2 import D_DEV, DeviationEncoder, deviation, group_index
from results_paths import rpath
from to_schema import invert_scaler

# ---- fixed by the protocol. Changing any of these changes the experiment, and the protocol with it.
PROTOCOL = ROOT / "progress" / "decisions" / "V2_Deviation_Protocol.md"
DATASET = "covid_us-states"
ARMS = ("v2graph", "v2nograph", "v2shuffled")
TAGS = {"v2graph": "v2graph", "v2nograph": "v2nograph", "v2shuffled": "v2shuf", "v1": "v1", "v1ref": "v1ref"}
DEV_GRAPH = {"v2graph": "real", "v2nograph": "none", "v2shuffled": "shuffled"}
SEED_SEQ = tuple(42 + 10 * i for i in range(134))     # protocol section 5: 42, 52, ..., 1372
STAGE1_SEEDS = SEED_SEQ[:5]                           # the five seeds v1 was released with
COMPARISONS = (("v2graph", "v1"), ("v2graph", "v2nograph"), ("v2graph", "v2shuffled"))
RULE = "tint"             # decides: the paired 95% t-interval excludes zero. "gate" is printed only.
DELTA = 0.033             # the probe-implied h3 RMSE effect, as a fraction of v1
Z_ALPHA, Z_BETA = 1.96, 0.84
POWER_TARGET = 0.80
N_MIN, N_CAP = 5, 134
METRICS = ("rmse", "mae")               # verdict metrics; PCC is printed, never decides
FIELD = "country_macro"                 # the scored field; equals node_mean on single-country COVID
SHORT = (3, 5)                          # criterion (a) horizons (h5 cannot be powered; see protocol)
MODEL = "encoder"                       # the median forecast, as v1 is read (ledger D3)
RES = ROOT / "results"
V1_REF = RES / "single" / "encoder__influenza_japan__seed42.json"
V1_CKPT = RES / "single" / "encoder__influenza_japan__seed42__ckpt.pt"
SAME_INIT_ARMS = ("gateoff", "shufadj", "epi_p90max", "epi_p99max")    # power planning, paired-sd source
REPRO = HERE / "misc" / "v2_repro_check.json"
STAGE1_SUMMARY = HERE / "misc" / "v2_stage1_summary.json"
CODE_FILES = ("train/loop.py", "models/config.py", "models/encoder.py", "models/encoder_v2.py", "models/spatial.py",
              "models/temporal.py", "models/adapters.py", "models/windows.py", "bundles.py", "score.py",
              "to_schema.py", "ablation/run_v2_deviation.py")
V2_KEYS = frozenset({
    "encoder_version", "v2_arm", "v2_dev_graph", "v2_d_dev", "v2_combine", "v2_n_groups",
    "v2_branch_params", "v2_protocol_sha256", "v2_dev_shuffle_seed", "v2_dev_shuffle_hash",
    "v2_dev_shuffle_frac", "v2_dev_contrib"})
COMBINE = "add_zero_init_linear"


def protocol_sha256(path=None):
    return hashlib.sha256(Path(path or PROTOCOL).read_bytes()).hexdigest()


def code_sha256():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES}


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _protocol_committed():
    """True when the protocol is tracked and has no uncommitted edits."""
    rel = PROTOCOL.relative_to(ROOT).as_posix()
    return _git("ls-files", "--error-unmatch", rel).returncode == 0 and not _git(
        "status", "--porcelain", "--", rel).stdout.strip()


def seeds_file(ds, s, arm):
    return f"encoder__{ds}__seed{s}__{TAGS[arm]}"


# --------------------------------------------------------------------------- #
# Building an arm, scoring a split
# --------------------------------------------------------------------------- #
def make_factory(arm, perm_seed=None, dev_gain=1.0):
    """encoder_factory for train.loop.train_one: builds the arm's DeviationEncoder and its record meta."""
    assert arm in ARMS, arm
    assert (perm_seed is not None) == (arm == "v2shuffled"), "only the shuffled arm takes a perm seed"

    def factory(b, gate_mode, device):
        grp, labels = group_index(b.meta["node_ids"], b.group_of())
        enc = DeviationEncoder(gate_mode=gate_mode, d_dev=D_DEV, dev_graph=DEV_GRAPH[arm],
                               dev_gain=dev_gain).to(device)
        shuf = dict(v2_dev_shuffle_seed=None, v2_dev_shuffle_hash=None, v2_dev_shuffle_frac=None)
        A_dev = None
        if arm == "v2shuffled":
            A_perm, m = L.permute_adjacency(b.A_geo, perm_seed)       # refuses the identity
            assert m["shuffle_adj_frac_displaced"] > 0.0
            A_dev = sparse_from_dense_np(A_perm).to(device)
            shuf = dict(v2_dev_shuffle_seed=m["shuffle_adj_seed"], v2_dev_shuffle_hash=m["shuffle_adj_hash"],
                        v2_dev_shuffle_frac=m["shuffle_adj_frac_displaced"])
        enc.bind(group=torch.tensor(grp, dtype=torch.long, device=device), A_dev=A_dev)
        assert node_indexed_params(enc) == [], "C2"
        meta = dict(encoder_version="v2_deviation", v2_arm=arm, v2_dev_graph=DEV_GRAPH[arm],
                    v2_d_dev=D_DEV, v2_combine=COMBINE, v2_n_groups=len(labels),
                    v2_branch_params=sum(p.numel() for p in enc.branch_params()),
                    v2_protocol_sha256=protocol_sha256(), **shuf)
        return enc, meta
    return factory


def _panel_tensors(ds, device):
    b = bundles.load(ds)
    Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=device)
    Mt = torch.tensor(b.M, dtype=torch.float32, device=device)
    A = sparse_from_dense_np(b.A_geo).to(device)
    return b, Z, Mt, A


def score_split(enc, ad, ds, seed, phase):
    """Median-forecast records for one split of a trained model, assembled exactly as train_one assembles
    test: for every origin of that split and every horizon, invert the median quantile through the
    released scaler into the column of the target week, then score through train.loop.score_predictions
    on that split's mask. This is how the validation metrics the records do not carry are made."""
    dev = next(enc.parameters()).device
    b, Z, Mt, A = _panel_tensors(ds, dev)
    origins = b.origins(phase=phase)
    N, T = b.X.shape[0], b.X.shape[1]
    pred = {h: np.zeros((N, T), dtype=np.float64) for h in bundles.HORIZONS}
    enc.eval(); ad.eval()
    with torch.no_grad():
        for t in origins:
            med = ad(enc(window_slice(Z, t), A, Mt[:, t])).cpu().numpy()[:, :, MEDIAN_IDX]
            for j, h in enumerate(bundles.HORIZONS):
                pred[h][:, t + h] = invert_scaler(med[:, j:j + 1], b.scaler)[:, 0]
    recs, _, _ = L.score_predictions(MODEL, ds, seed, pred, b, origins, phase=phase)
    return recs


def load_v1_ckpt(ds, s, device):
    """The released v1 checkpoint: the val-selected state train_one kept, saved by run_dataset."""
    from models import Adapter, SharedEncoder
    ck = torch.load(rpath(f"encoder__{ds}__seed{s}__ckpt.pt", root=RES), map_location=device,
                    weights_only=False)
    assert ck["meta"].get("seed") == s and ck["meta"].get("dataset") == ds, f"checkpoint meta {ck['meta']}"
    enc, ad = SharedEncoder().to(device), Adapter().to(device)
    enc.load_state_dict(ck["encoder"]); ad.load_state_dict(ck["adapter"])
    return enc, ad


def write_val(ds, s, arm, recs, source, root=None):
    for r in recs:
        r.update(split="val", arm=arm, val_source=source, v2_protocol_sha256=protocol_sha256())
    rpath(seeds_file(ds, s, arm) + "__val.json", root=root or HERE, make=True).write_text(json.dumps(recs, indent=1))


def _read_val(ds, s, arm, root=None):
    """{(h, metric): value} from a VALIDATION file only. Refuses anything not marked split=val."""
    p = rpath(seeds_file(ds, s, arm) + "__val.json", root=root or HERE)
    recs = json.loads(p.read_text())
    assert recs and all(r.get("split") == "val" and r["model"] == MODEL for r in recs), f"{p.name} is not val"
    return {(r["horizon"], r["metric"]): r[FIELD] for r in recs}


def dev_contrib(enc, ds):
    """Secondary readout, never a criterion: mean over val origins and nodes of |P(branch)| / |v1 output|."""
    dev = next(enc.parameters()).device
    b, Z, Mt, A = _panel_tensors(ds, dev)
    enc.eval()
    vals = []
    with torch.no_grad():
        for t in b.origins(phase="val"):
            enc(window_slice(Z, t), A, Mt[:, t])
            vals.append(float((enc.last_dev_out.norm(dim=-1) / (enc.last_v1.norm(dim=-1) + 1e-8)).mean()))
    return round(float(np.mean(vals)), 6)


def assert_schema(recs, arm, ref=V1_REF):
    """Every v2 record carries EXACTLY the v1 key set of the same model plus V2_KEYS, the arm label is
    the arm, and the record set covers the same (model, horizon, metric) grid as the v1 reference."""
    ref_recs = json.loads(Path(ref).read_text())
    want = {}
    for r in ref_recs:
        want.setdefault(r["model"], set(r))
    grid = lambda rs: sorted((r["model"], r["horizon"], r["metric"]) for r in rs)
    assert grid(recs) == grid(ref_recs), "record grid differs from the v1 reference"
    for r in recs:
        base = want[r["model"]] | V2_KEYS
        keys = set(r)
        assert keys == base, (f"schema drift vs {Path(ref).name} ({r['model']}): "
                              f"missing {sorted(base - keys)}, unexpected {sorted(keys - base)}")
        assert r["v2_arm"] == arm and r["v2_dev_graph"] == DEV_GRAPH[arm], f"record claims {r['v2_arm']}"
        assert r["v2_protocol_sha256"] == protocol_sha256(), "record made under a different protocol"
        if arm == "v2shuffled":
            assert r["v2_dev_shuffle_frac"] and r["v2_dev_shuffle_frac"] > 0.0, "shuffle was a no-op"
        else:
            assert r["v2_dev_shuffle_seed"] is None, "a non-shuffled arm carries a shuffle seed"


def train_cell(ds, s, arm, epochs, smoke=False, dev_gain=1.0):
    """One (panel, seed, arm): train, check, write val file, records, archives, checkpoint LAST.
    Returns (test records, seconds). Prints nothing: the caller decides what may be shown."""
    t0 = time.time()
    quant, ro = {}, {}
    ps = perm_seed_for(s) if arm == "v2shuffled" else None
    recs, pernode, _, _ = L.train_one(ds, s, epochs=epochs, verbose=False, quant_out=quant, run_out=ro,
                                      encoder_factory=make_factory(arm, ps, dev_gain))
    enc = ro["enc"]
    assert isinstance(enc, DeviationEncoder) and enc.dev_graph == DEV_GRAPH[arm], "the arm is not the arm"
    assert node_indexed_params(enc) == [], "C2"
    c = dev_contrib(enc, ds)
    for r in recs:
        r["v2_dev_contrib"] = c
    assert_schema(recs, arm)
    stem = seeds_file(ds, s, arm) + ("__smoke" if smoke else "")
    if not smoke:
        write_val(ds, s, arm, score_split(enc, ro["ad"], ds, s, "val"), "in-memory val-selected model")
    L.write_records(recs, stem + ".json")
    L.write_per_node(pernode, stem + "__pernode.npz")
    L.write_quantiles(quant, ro["te"], stem + "__quantiles.npz")
    L.write_checkpoint(enc, ro["ad"], stem + "__ckpt.pt",
                       extra=dict(dataset=ds, seed=s, arm=arm, bias_c=ro["bias_c"],
                                  protocol_sha256=protocol_sha256()))
    return recs, time.time() - t0


def train_v1ref(ds, s, epochs):
    """A v1 seed the release does not have (stage 2). Same trainer, default path, no factory. Writes
    val file, records, per-node archive, checkpoint last. Never touches results/single/."""
    t0 = time.time()
    assert not rpath(f"encoder__{ds}__seed{s}.json", root=RES).exists(), "v1 already has this seed"
    ro = {}
    recs, pernode, _, _ = L.train_one(ds, s, epochs=epochs, verbose=False, run_out=ro)
    stem = seeds_file(ds, s, "v1ref")
    write_val(ds, s, "v1ref", score_split(ro["enc"], ro["ad"], ds, s, "val"), "in-memory val-selected model")
    L.write_records(recs, stem + ".json")
    L.write_per_node(pernode, stem + "__pernode.npz")
    L.write_checkpoint(ro["enc"], ro["ad"], stem + "__ckpt.pt",
                       extra=dict(dataset=ds, seed=s, arm="v1ref", bias_c=ro["bias_c"],
                                  protocol_sha256=protocol_sha256()))
    return recs, time.time() - t0


# --------------------------------------------------------------------------- #
# Stage 1: validation statistics only
# --------------------------------------------------------------------------- #
def stage1_stats(root=None):
    """The ONLY statistics stage 1 computes: per comparison, the sd of the paired per-seed differences
    in h3 RMSE on the VALIDATION split, and that sd as a fraction of v1's validation mean. Reads val
    files only. The mean of the differences is deliberately not computed or stored."""
    val = {a: {s: _read_val(DATASET, s, a, root)[(3, "rmse")] for s in STAGE1_SEEDS} for a in ("v1",) + ARMS}
    v1m = float(np.mean(list(val["v1"].values())))
    comps = {}
    for arm, ref in COMPARISONS:
        d = np.array([val[arm][s] - val[ref][s] for s in STAGE1_SEEDS])
        sd = float(np.sqrt(((d - d.mean()) ** 2).sum() / (len(d) - 1)))
        comps[f"{arm}_vs_{ref}"] = dict(sd=sd, rel_sd=sd / v1m)
    s_from = max(comps, key=lambda k: comps[k]["rel_sd"])
    return dict(dataset=DATASET, split="val", horizon=3, metric="rmse", field=FIELD, model=MODEL,
                seeds=list(STAGE1_SEEDS), v1_val_mean=v1m, comparisons=comps, s=comps[s_from]["rel_sd"],
                s_from=s_from, delta=DELTA, protocol_sha256=protocol_sha256())


def write_stage1_summary(root=None, path=None):
    summ = stage1_stats(root)
    Path(path or STAGE1_SUMMARY).write_text(json.dumps(summ, indent=1))
    print(f"STAGE 1 :: {DATASET}, VALIDATION split only, h3 RMSE ({FIELD}, median forecast), "
          f"seeds {summ['seeds']}")
    print(f"  v1 validation mean h3 RMSE {summ['v1_val_mean']:.3f}")
    for name, c in summ["comparisons"].items():
        print(f"  {name:24s} sd of paired differences {c['sd']:10.3f}  = {c['rel_sd']:.4f} of the v1 mean")
    print(f"  s = the largest = {summ['s']:.4f} ({summ['s_from']}). Next: --seed-count")
    return summ


def load_stage1_summary(root=None, path=None):
    """The stage-1 summary, re-derived from the val files and required to match, under this protocol."""
    p = Path(path or STAGE1_SUMMARY)
    assert p.exists(), f"no stage-1 summary at {p}; run --stage1 first"
    summ = json.loads(p.read_text())
    assert summ.get("protocol_sha256") == protocol_sha256(), "stage-1 summary made under another protocol"
    assert summ.get("split") == "val" and summ.get("seeds") == list(STAGE1_SEEDS), "not a stage-1 val summary"
    fresh = stage1_stats(root)
    for k in [f"{a}_vs_{r}" for a, r in COMPARISONS]:
        assert abs(fresh["comparisons"][k]["rel_sd"] - summ["comparisons"][k]["rel_sd"]) < 1e-12, \
            f"stage-1 summary disagrees with the val files at {k}"
    return summ


# --------------------------------------------------------------------------- #
# The seed count
# --------------------------------------------------------------------------- #
def power(k, n, rule):
    """P(the rule calls v2 better) when the true paired effect is k paired-sds in v2's favour, n seeds.
    gate: |mean d| >= sd d  <=>  t >= sqrt(n).  tint: t >= t_{0.975, n-1}. Gaussian seed noise."""
    from scipy import stats
    crit = math.sqrt(n) if rule == "gate" else stats.t.ppf(0.975, n - 1)
    return float(stats.nct.sf(crit, n - 1, k * math.sqrt(n)))


def _mde(k_power, n, rule):
    """Smallest standardized paired effect (delta / sd_d) found with probability k_power."""
    from scipy import optimize
    return optimize.brentq(lambda k: power(k, n, rule) - k_power, 1e-6, 50.0)


def seed_count(s):
    """Protocol section 5.3. n_z = max(N_MIN, ceil(((1.96 + 0.84) s / delta)^2)); the t correction is the
    smallest n >= n_z at which the exact power of the decision rule (paired two-sided 95% t-interval
    excluding zero on v2's side, noncentral t with n-1 df and noncentrality sqrt(n) delta / s) reaches
    0.80; N = min(that, N_CAP). If the corrected count exceeds the cap, N_CAP runs and the test is
    underpowered for delta."""
    if s <= 0:
        return dict(s=s, k=float("inf"), n_z=N_MIN, n_t=N_MIN, N=N_MIN, underpowered=False, power_at_N=1.0)
    k = DELTA / s
    n_z = max(N_MIN, math.ceil(((Z_ALPHA + Z_BETA) * s / DELTA) ** 2))
    ok = lambda n: power(k, n, "tint") >= POWER_TARGET
    lo = hi = n_z
    if not ok(hi):
        while not ok(hi):
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    n_t = hi
    N = min(n_t, N_CAP)
    return dict(s=s, k=k, n_z=n_z, n_t=n_t, N=N, underpowered=n_t > N_CAP, power_at_N=power(k, N, "tint"))


def _s_of(summ):
    """s is the LARGEST of the three stage-1 validation relative sds."""
    return max(c["rel_sd"] for c in summ["comparisons"].values())


def seed_count_from_summary(summ):
    assert set(summ["comparisons"]) == {f"{a}_vs_{r}" for a, r in COMPARISONS}, "need all three comparisons"
    return seed_count(_s_of(summ))


def _run_costs():
    """Per-run seconds for the ETA: v1 from the D2 COVID record gaps, v2 from the smoke's ratio."""
    p = HERE / "misc" / "v2_smoke_summary.json"
    if not p.exists():
        return None
    sm = json.loads(p.read_text())
    return sm["v1_covid_run_seconds_median"], sm["v2_covid_run_seconds"]


def print_seed_count(root=None, path=None):
    summ = load_stage1_summary(root, path)
    sc = seed_count_from_summary(summ)
    print(f"SEED COUNT :: s = {sc['s']:.4f} (largest stage-1 validation relative sd, {summ['s_from']}), "
          f"delta = {DELTA}")
    print(f"  normal formula n_z = max({N_MIN}, ceil(((1.96 + 0.84) * s / delta)^2)) = {sc['n_z']}")
    print(f"  t-corrected n_t = {sc['n_t']}  (smallest n >= n_z with exact power >= {POWER_TARGET})")
    print(f"  N = min(n_t, {N_CAP}) = {sc['N']}   seeds {SEED_SEQ[0]} to {SEED_SEQ[sc['N'] - 1]}")
    if sc["underpowered"]:
        print(f"  UNDERPOWERED: the formula wants {sc['n_t']} seeds; the cap runs {N_CAP}. At {N_CAP} seeds the "
              f"chance of finding delta on h3 RMSE is {sc['power_at_N']:.3f}, below {POWER_TARGET}.")
    else:
        print(f"  chance of finding delta on h3 RMSE alone at N: {sc['power_at_N']:.3f}")
    extra = sc["N"] - len(STAGE1_SEEDS)
    costs = _run_costs()
    eta = f", about {extra * (costs[0] + 3 * costs[1]) / 3600:.1f} h" if costs else ""
    print(f"  stage 2: {extra} new seeds x 4 arms = {4 * extra} runs{eta}. Next: --stage2")
    return sc


# --------------------------------------------------------------------------- #
# Reading test records and deciding. Only after stage 2 is complete.
# --------------------------------------------------------------------------- #
def _read(path, model=MODEL):
    out = {}
    for r in json.loads(Path(path).read_text()):
        if r["model"] == model:
            out[(r["horizon"], r["metric"])] = r
    return out


def _v1_path(ds, s, root=None):
    p = rpath(f"encoder__{ds}__seed{s}.json", root=RES)
    return p if p.exists() else rpath(seeds_file(ds, s, "v1ref") + ".json", root=root or HERE)


def missing_runs(seeds, root=None):
    """(seed, arm) pairs with no test record, by FILE EXISTENCE only: no test record is opened."""
    miss = [(s, "v1") for s in seeds if not _v1_path(DATASET, s, root).exists()]
    miss += [(s, a) for s in seeds for a in ARMS
             if not rpath(seeds_file(DATASET, s, a) + ".json", root=root or HERE).exists()]
    return miss


def cell_verdict(ref, arm, rule=None):
    """(mean d, sd d, n, verdict, (lo, hi)) for d = arm - ref per shared seed; lower is better.

    tint: better when the paired 95% t-interval lies entirely below 0, worse when entirely above.
    gate: within noise when |mean d| < sd d, exactly run_gate_ablation.py."""
    from scipy import stats
    rule = rule or RULE
    dm, dsd, n = paired_delta(ref, arm)
    half = stats.t.ppf(0.975, n - 1) * dsd / math.sqrt(n) if n >= 2 else float("inf")
    ci = (dm - half, dm + half)
    if n < 2 or dsd == 0:
        return dm, dsd, n, "noise", ci
    if rule == "gate":
        v = "noise" if abs(dm) < dsd else ("better" if dm < 0 else "worse")
    elif rule == "tint":
        v = "better" if ci[1] < 0 else ("worse" if ci[0] > 0 else "noise")
    else:
        raise ValueError(rule)
    return dm, dsd, n, v, ci


def decide(v):
    """The protocol section 7 decision, mechanically. v[(ref, h, metric)] is the verdict of v2graph
    against ref in {'v1', 'v2nograph', 'v2shuffled'}. Returns (label, reason)."""
    H = bundles.HORIZONS
    wins = [h for h in SHORT if all(v[("v1", h, m)] == "better" for m in METRICS)]
    harm = [(h, m) for h in H for m in METRICS if v[("v1", h, m)] == "worse"]
    if not wins:
        return "FAIL", "criterion (a) not met: no win over v1 at h3 or h5 on both RMSE and MAE"
    if harm:
        return "FAIL", (f"criterion (a) not met: wins at h{wins} but significantly worse than v1 at "
                        + ", ".join(f"h{h} {m}" for h, m in harm))
    both = lambda ref, h: all(v[(ref, h, m)] == "better" for m in METRICS)
    attr = [h for h in wins if both("v2nograph", h) and both("v2shuffled", h)]
    if attr:
        return "PASS", f"(a) and (b) met at h{attr}"
    ng = [h for h in wins if both("v2nograph", h)]
    sh = [h for h in wins if both("v2shuffled", h)]
    detail = (f"beats v2nograph at h{ng or '-'}, beats v2shuffled at h{sh or '-'}, "
              "never both at one winning horizon")
    return "A_ONLY", "(a) met, (b) not met: the extra input helped, not the graph. " + detail


def _verdicts(seeds, root=None):
    """Read the test records of all four arms over `seeds` and build the verdict grid under RULE, with
    the gate rule's verdict alongside. Returns (decision or None, grid, problems)."""
    v1 = {s: _read(_v1_path(DATASET, s, root)) for s in seeds}
    arms = {a: {s: _read(rpath(seeds_file(DATASET, s, a) + ".json", root=root or HERE)) for s in seeds}
            for a in ARMS}
    problems = []
    shas = {r["v2_protocol_sha256"] for a in ARMS for s in arms[a] for r in arms[a][s].values()}
    if shas != {protocol_sha256()}:
        problems.append(f"records carry protocol sha256 {sorted(shas)}, file is {protocol_sha256()}")
    for a in ARMS:
        for s, recs in arms[a].items():
            if any(r["v2_arm"] != a for r in recs.values()):
                problems.append(f"{a} seed{s}: a record claims another arm")
    for src in [v1] + list(arms.values()):
        for recs in src.values():
            for r in recs.values():
                if r["n_countries"] == 1 and r["country_macro"] != r["node_mean"]:
                    problems.append(f"{r['dataset']} single-country record has macro != node_mean")
    grid = {}
    val = lambda src, h, m: {s: src[s][(h, m)][FIELD] for s in src}
    refs = {"v1": v1, "v2nograph": arms["v2nograph"], "v2shuffled": arms["v2shuffled"]}
    pairs = [("v1", "v2graph"), ("v1", "v2nograph"), ("v1", "v2shuffled"),
             ("v2nograph", "v2graph"), ("v2shuffled", "v2graph")]
    for ref_name, arm_name in pairs:
        for h in bundles.HORIZONS:
            for m in METRICS + ("pcc",):
                a, r = val(arms[arm_name], h, m), val(refs[ref_name], h, m)
                dm, dsd, n, verdict, ci = cell_verdict(r, a)
                gate = cell_verdict(r, a, "gate")[3]
                if m == "pcc":                                   # higher is better for PCC
                    flip = {"better": "worse", "worse": "better", "noise": "noise"}
                    verdict, gate = flip[verdict], flip[gate]
                grid[(ref_name, arm_name, h, m)] = dict(ref=mean_sd(list(r.values())), arm=mean_sd(list(a.values())),
                                                        d=dm, sd=dsd, n=n, ci=ci, verdict=verdict, gate=gate)
    if problems:
        return None, grid, problems
    v = {(ref, h, m): grid[(ref, "v2graph", h, m)]["verdict"]
         for ref in ("v1", "v2nograph", "v2shuffled") for h in bundles.HORIZONS for m in METRICS}
    return decide(v), grid, problems


def decision_from_disk(root=None, path=None):
    """Stage-1 summary -> N -> every one of the N x 4 runs present (by file existence) -> read test.
    Returns (decision or None, grid, problems, seed count)."""
    summ = load_stage1_summary(root, path)
    sc = seed_count_from_summary(summ)
    seeds = SEED_SEQ[:sc["N"]]
    miss = missing_runs(seeds, root)
    if miss:
        return None, {}, [f"stage 2 incomplete: {len(miss)} of {4 * len(seeds)} runs missing"], sc
    dec, grid, problems = _verdicts(seeds, root)
    return dec, grid, problems, sc


def report(root=None, path=None):
    try:
        dec, grid, problems, sc = decision_from_disk(root, path)
    except AssertionError as e:                          # no or invalid stage-1 summary
        print(f"REPORT REFUSED, no test number read. {e}")
        return None
    if dec is None:
        print(f"REPORT REFUSED, no test number printed. {'; '.join(problems)}")
        return None
    print(f"\n{'=' * 110}\nV2 DEVIATION CHANNEL :: {DATASET}   field={FIELD}   model={MODEL}   N={sc['N']} seeds "
          f"({SEED_SEQ[0]}..{SEED_SEQ[sc['N'] - 1]})   decision rule: paired 95% t-interval\n"
          f"protocol sha256 {protocol_sha256()}\n{'=' * 110}")
    for (ref, arm, h, m), c in sorted(grid.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], kv[0][2])):
        (rm, rsd), (am, asd) = c["ref"], c["arm"]
        print(f"  {arm:10} vs {ref:10} {m:4} h{h:<2} | ref {rm:11.3f} +-{rsd:9.3f} | arm {am:11.3f} +-{asd:9.3f}"
              f" | d {c['d']:+10.3f} 95% [{c['ci'][0]:+10.3f}, {c['ci'][1]:+10.3f}] | {c['verdict']:6}"
              f" | gate rule: {c['gate']} (no weight)" + ("   [PCC: printed only]" if m == "pcc" else ""))
    print(f"\n  h5 is reported but cannot be powered: the probe-implied h5 effect is under 1 percent.")
    print(f"  N = {sc['N']} from s = {sc['s']:.4f}; chance of finding delta = {DELTA} on h3 RMSE alone: "
          f"{sc['power_at_N']:.3f}{'  UNDERPOWERED (cap reached)' if sc['underpowered'] else ''}")
    print(f"\n  DECISION: {dec[0]}  {dec[1]}\n")
    return dec


# --------------------------------------------------------------------------- #
# Power planning from disk (protocol section 6), seconds, no GPU.
# --------------------------------------------------------------------------- #
def _v1_model_mse(ds, seeds=STAGE1_SEEDS):
    """v1's model-space MSE of the median forecast on the test cells, per horizon, mean over seeds."""
    from to_schema import apply_scaler
    b = bundles.load(ds)
    test = b.masks()["test"].astype(bool)
    out = {}
    for h in bundles.HORIZONS:
        vals = []
        for s in seeds:
            z = np.load(rpath(f"encoder__{ds}__seed{s}__quantiles.npz", root=RES))
            qi = int(np.argmin(np.abs(z["quantiles"] - 0.5)))
            cols = z["origins"] + h
            med = np.maximum(z[f"h{h}__quantiles"][:, :, qi].astype(np.float64), 0.0)
            pm = apply_scaler(med, b.scaler).astype(np.float64)
            y, m = b.y[:, cols].astype(np.float64), test[:, cols]
            vals.append(float(np.mean((pm - y)[m] ** 2)))
        out[h] = float(np.mean(vals))
    return out


def power_check(ds=DATASET, verbose=True):
    from scipy import stats
    v1 = {s: _read(rpath(f"encoder__{ds}__seed{s}.json", root=RES)) for s in STAGE1_SEEDS}
    val = lambda h, m: {s: v1[s][(h, m)][FIELD] for s in STAGE1_SEEDS}
    t6 = json.loads((RES / "misc" / "t6_neighbour_signal.json").read_text())
    probe = t6["panels"][ds]["horizons"]
    mse_v1 = _v1_model_mse(ds)
    rows = []
    for h in bundles.HORIZONS:
        dmse = max(probe[str(h)]["mse_A"] - probe[str(h)]["B"]["mse_real"], 0.0)
        plaus = 100.0 * (1.0 - math.sqrt(1.0 - dmse / mse_v1[h]))
        for m in METRICS:
            base = val(h, m)
            v1m, v1sd = mean_sd(list(base.values()))
            sds, used = [], []
            for tag in SAME_INIT_ARMS:                       # dengue has gateoff and shufadj only
                ps = {s: HERE / "single" / f"encoder__{ds}__seed{s}__{tag}.json" for s in STAGE1_SEEDS}
                if not all(p.exists() for p in ps.values()):
                    continue
                sds.append(paired_delta(base, {s: _read(p)[(h, m)][FIELD] for s, p in ps.items()})[1])
                used.append(tag)
            assert sds, f"{ds}: no same-init arm on disk to plan the paired sd from"
            sdd = float(np.median(sds))
            bar = 100.0 * sdd / v1m
            k = plaus / bar
            row = dict(h=h, metric=m, v1_mean=v1m, v1_sd=v1sd, sdd_plan=sdd, sdd_lo=min(sds),
                       sdd_hi=max(sds), sdd_indep=math.sqrt(2) * v1sd, bar_pct=bar, plaus_pct=plaus,
                       dmse=dmse, mse_v1=mse_v1[h], k=k, sdd_arms=used)
            for rule, ns in (("gate", (5, 10)), ("tint", (5, 10, 20, 40))):
                for n in ns:
                    row[f"mde80_{rule}_{n}"] = 100.0 * _mde(0.8, n, rule) * sdd / v1m
                    row[f"pow_{rule}_{n}"] = power(k, n, rule)
            row["n80_tint"] = next((n for n in range(3, 5000) if power(k, n, "tint") >= 0.8), None) \
                if k > 0 else None
            row["fp_gate_5"] = float(stats.t.sf(math.sqrt(5), 4))
            row["fp_gate_10"] = float(stats.t.sf(math.sqrt(10), 9))
            rows.append(row)
    if verbose:
        print(f"POWER PLANNING :: {ds}, v1 = results/single seeds {list(STAGE1_SEEDS)}, field {FIELD}, "
              f"model {MODEL}. Existing arms only; no v2 number.")
        print(f"  paired-sd planning value = median paired sd of the same-init arms {rows[0]['sdd_arms']}")
        print(f"  probe effect = t6 real-map MSE cut (mse_A - mse_real), turned into an RMSE change against "
              f"v1's own model-space MSE")
        print(f"  false-positive rate of the gate rule, one direction: n=5 {rows[0]['fp_gate_5']:.4f}, "
              f"n=10 {rows[0]['fp_gate_10']:.4f}; tint rule 0.025 at any n\n")
        print("  h  metric   v1 mean    v1 sd | sd_d plan [lo, hi]      indep | bar% | probe% "
              "   k   | gate MDE80% n5  n10 | tint MDE80% n10  n20  n40 | P(gate) n5  n10 | P(tint) n20 "
              " n40 | n80 tint")
        for r in rows:
            print(f"  {r['h']:<2} {r['metric']:5} {r['v1_mean']:9.1f} {r['v1_sd']:8.1f} | {r['sdd_plan']:7.1f} "
                  f"[{r['sdd_lo']:6.1f}, {r['sdd_hi']:6.1f}] {r['sdd_indep']:7.1f} | {r['bar_pct']:4.1f} | "
                  f"{r['plaus_pct']:5.2f} {r['k']:5.2f} | {r['mde80_gate_5']:14.1f} {r['mde80_gate_10']:4.1f} | "
                  f"{r['mde80_tint_10']:15.1f} {r['mde80_tint_20']:4.1f} {r['mde80_tint_40']:4.1f} | "
                  f"{r['pow_gate_5']:10.3f} {r['pow_gate_10']:4.3f} | {r['pow_tint_20']:11.3f} "
                  f"{r['pow_tint_40']:4.3f} | {r['n80_tint']}")
    return rows


# --------------------------------------------------------------------------- #
# The repro gate
# --------------------------------------------------------------------------- #
def _record_diff(ra, rb):
    """Max abs relative difference over numeric fields of records matched on (model, horizon, metric)."""
    ka = {(r["model"], r["horizon"], r["metric"]): r for r in ra}
    kb = {(r["model"], r["horizon"], r["metric"]): r for r in rb}
    assert set(ka) == set(kb)
    worst, exact = 0.0, True
    for k in ka:
        for f in ("country_macro", "node_mean"):
            x, y = ka[k][f], kb[k][f]
            if not (x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y))):
                exact = False
                worst = max(worst, abs(x - y) / max(abs(x), 1e-12))
    return exact, worst


def _repro_check(ds="influenza_japan", s=42):
    """Does TODAY's trainer reproduce the released v1 record? The v2 arms are paired against v1 records
    made in August, so if it does not, every v2 - v1 delta also carries code drift. influenza_japan, not
    COVID. Full 80 epochs, a few minutes. Refuses unless the protocol is committed, so the passing record
    is tied to the committed protocol and to the hashes of the code that will train COVID."""
    assert _protocol_committed(), "commit the protocol (no local edits) before the repro-check"
    ref = json.loads(rpath(f"encoder__{ds}__seed{s}.json", root=RES).read_text())
    t0 = time.time()
    recs, *_ = L.train_one(ds, s, verbose=False)
    exact, worst = _record_diff(ref, recs)
    out = dict(identical=exact, max_rel_diff=worst, dataset=ds, seed=s, epochs=80, device=L.DEVICE,
               protocol_sha256=protocol_sha256(),
               protocol_commit=_git("log", "-1", "--format=%H", "--", PROTOCOL.relative_to(ROOT).as_posix()).stdout.strip(),
               code_sha256=code_sha256(), seconds=round(time.time() - t0, 1))
    REPRO.parent.mkdir(parents=True, exist_ok=True)
    REPRO.write_text(json.dumps(out, indent=1))
    print(f"repro-check {ds} seed{s}: today's v1 vs results/single record: "
          f"{'IDENTICAL' if exact else 'DIFFER'} (max rel diff {worst:.3e}), {out['seconds']:.0f}s -> {REPRO.name}")
    if not exact:
        print("  COVID stays blocked. Report the difference; the user decides.")
    return 0 if exact else 1


def _repro_ok(path=None):
    """(ok, reason). A passing repro record for THIS protocol file and THIS code."""
    p = Path(path or REPRO)
    if not p.exists():
        return False, "no repro-check record; run --repro-check"
    r = json.loads(p.read_text())
    if r.get("identical") is not True:
        return False, f"repro-check did not reproduce v1 (max rel diff {r.get('max_rel_diff')})"
    if r.get("protocol_sha256") != protocol_sha256():
        return False, "repro-check was run under a different protocol file"
    if r.get("code_sha256") != code_sha256():
        changed = sorted(f for f in CODE_FILES if (r.get("code_sha256") or {}).get(f) != code_sha256()[f])
        return False, f"code changed since the repro-check: {changed}"
    return True, "ok"


def _guard(stage):
    assert _protocol_committed(), ("the protocol must be committed with no local edits before any COVID "
                                   f"run: {PROTOCOL.relative_to(ROOT).as_posix()}")
    ok, why = _repro_ok()
    assert ok, f"COVID is blocked until a passing repro-check exists: {why}"
    if stage == 2:
        load_stage1_summary()
        miss = missing_runs(STAGE1_SEEDS)
        assert not miss, f"stage 1 is incomplete: {miss}"


# --------------------------------------------------------------------------- #
# The two stages
# --------------------------------------------------------------------------- #
def _ensure_v1_val(seeds, device):
    for s in seeds:
        if not rpath(seeds_file(DATASET, s, "v1") + "__val.json", root=HERE).exists():
            enc, ad = load_v1_ckpt(DATASET, s, device)
            write_val(DATASET, s, "v1", score_split(enc, ad, DATASET, s, "val"), "released checkpoint")


def _done(s, arm):
    """A run is done when its checkpoint exists: the checkpoint is its last write."""
    return rpath(seeds_file(DATASET, s, arm) + "__ckpt.pt", root=HERE).exists()


def _train_list(todo, epochs):
    done, spent = 0, 0.0
    for s, arm in todo:
        _, dt = (train_v1ref(DATASET, s, epochs) if arm == "v1ref" else train_cell(DATASET, s, arm, epochs))
        done += 1
        spent += dt
        print(f"  seed{s} {arm:10} done in {dt / 60:.1f} min   [{done}/{len(todo)}, "
              f"ETA ~{spent / done * (len(todo) - done) / 60:.0f} min]")


def run_stage1(epochs=80, dry=False):
    todo = [(s, a) for s in STAGE1_SEEDS for a in ARMS if not _done(s, a)]
    print(f"STAGE 1: {DATASET}, {len(todo)} of {len(STAGE1_SEEDS) * len(ARMS)} runs to train, "
          f"seeds {list(STAGE1_SEEDS)}; output is validation statistics only")
    if dry:
        for s, a in todo:
            print(f"  DRY seed{s} {a}")
        return 0
    _guard(1)
    L.RESULTS = HERE
    _ensure_v1_val(STAGE1_SEEDS, L.DEVICE)
    _train_list(todo, epochs)
    write_stage1_summary()
    return 0


def run_stage2(epochs=80, dry=False):
    if dry and not STAGE1_SUMMARY.exists():
        print("STAGE 2: no stage-1 summary yet, so N is not known. Run --stage1 first.")
        return 0
    if not dry:
        _guard(2)
    sc = seed_count_from_summary(load_stage1_summary())
    seeds = SEED_SEQ[len(STAGE1_SEEDS):sc["N"]]
    todo = [(s, a) for s in seeds for a in ("v1ref",) + ARMS if not _done(s, a)]
    print(f"STAGE 2: N = {sc['N']} (from the stage-1 summary), {len(todo)} runs to train over seeds "
          f"{seeds[0] if seeds else '-'}..{seeds[-1] if seeds else '-'}"
          + ("  UNDERPOWERED: cap reached" if sc["underpowered"] else ""))
    if dry:
        for s, a in todo[:12]:
            print(f"  DRY seed{s} {a}")
        print(f"  ... {max(len(todo) - 12, 0)} more" if len(todo) > 12 else "")
        return 0
    L.RESULTS = HERE
    _train_list(todo, epochs)
    print(f"  stage 2 complete: {sc['N']} seeds x 4 arms. Next: --report")
    return 0


# --------------------------------------------------------------------------- #
# Selfcheck (CPU where possible), mutation self-test, smoke (influenza_japan only)
# --------------------------------------------------------------------------- #
def _fake_panel(tmp, v1_by_seed, arm_fn, seeds, sentinel_test=False):
    """Write fabricated COVID test records for v1ref (seeds v1 lacks) and the three arms into tmp, built
    from the real released v1 COVID records. arm_fn(arm, h, metric, seed, x) -> value."""
    meta = dict(encoder_version="v2_deviation", v2_d_dev=D_DEV, v2_combine=COMBINE, v2_n_groups=1,
                v2_branch_params=1, v2_protocol_sha256=protocol_sha256(), v2_dev_shuffle_seed=None,
                v2_dev_shuffle_hash=None, v2_dev_shuffle_frac=None, v2_dev_contrib=0.0)
    for i, s in enumerate(seeds):
        base = v1_by_seed[STAGE1_SEEDS[i % 5]]
        jit = 1.0 + 0.01 * (i // 5)                     # later seeds are not exact copies
        if s not in STAGE1_SEEDS:
            out = [dict(r, seed=s, country_macro=r["country_macro"] * jit, node_mean=r["country_macro"] * jit)
                   for r in base]
            rpath(seeds_file(DATASET, s, "v1ref") + ".json", root=tmp, make=True).write_text(json.dumps(out))
        for arm in ARMS:
            out = []
            for r in base:
                x = 987654.321 + i if sentinel_test else arm_fn(arm, r["horizon"], r["metric"], s, r["country_macro"] * jit)
                out.append(dict(r, seed=s, country_macro=x, node_mean=x, v2_arm=arm, v2_dev_graph=DEV_GRAPH[arm], **meta))
            rpath(seeds_file(DATASET, s, arm) + ".json", root=tmp, make=True).write_text(json.dumps(out))


def _fake_val(tmp, vals):
    """vals[arm][seed] = h3 rmse on validation; writes split=val files for v1 and the three arms."""
    for arm, by_seed in vals.items():
        for s, x in by_seed.items():
            recs = [dict(model=MODEL, dataset=DATASET, horizon=h, seed=s, metric=m, country_macro=x * h / 3,
                         node_mean=x * h / 3, n_countries=1, n_nodes=49, split="val", arm=arm)
                    for h in bundles.HORIZONS for m in ("rmse", "mae", "pcc")]
            rpath(seeds_file(DATASET, s, arm) + "__val.json", root=tmp, make=True).write_text(json.dumps(recs))


def _selfcheck():
    import contextlib
    import io
    import shutil
    import tempfile
    from models import Adapter, SharedEncoder
    from models.temporal import DilatedTCN
    torch.manual_seed(0)

    # 1. C2: no parameter sized by N in any arm; the control (branch width 47) must be caught.
    for g in ("real", "none", "shuffled"):
        assert node_indexed_params(DeviationEncoder(dev_graph=g)) == []
    try:
        DeviationEncoder(d_dev=47); raise SystemExit("C2 control did not fire")
    except AssertionError:
        pass
    print("ok  C2: no v2 parameter sized by a graph size; control d_dev=47 caught")

    # 1b. the branch convs leave cuDNN but keep the maths and the parameters. v1's convs stay cuDNN.
    from models.encoder_v2 import _MatmulConv1d
    e = DeviationEncoder()
    convs = [m for m in e.dev_tcn.modules() if isinstance(m, torch.nn.Conv1d)]
    assert convs and all(type(m) is _MatmulConv1d for m in convs), "a branch conv is still cuDNN"
    assert all(type(m) is torch.nn.Conv1d for m in e.tcn.modules() if isinstance(m, torch.nn.Conv1d)), \
        "v1's convs must stay plain Conv1d"
    xin = torch.randn(7, 16, 40)
    for m in convs[1:]:
        want = torch.nn.functional.conv1d(xin[:, :m.in_channels], m.weight, m.bias, dilation=m.dilation)
        assert torch.allclose(m(xin[:, :m.in_channels]), want, atol=1e-5), "matmul conv != conv1d"
    print(f"ok  branch convs: {len(convs)} swapped to matmul, equal to conv1d to 1e-5; v1 convs untouched")

    # 2. deviation arithmetic on a hand case: 2 countries, one unobserved cell.
    Z = torch.zeros(4, 3, 4)
    Z[:, :, 0] = torch.tensor([[1., 2., 3.], [3., 4., 5.], [10., 0., 7.], [20., 30., 9.]])
    Z[:, :, 3] = 1.0
    Z[2, 1, 0], Z[2, 1, 3] = 0.0, 0.0                               # node 2 unobserved at week 1
    grp = torch.tensor([0, 0, 1, 1])
    dv = deviation(Z, grp)
    want = torch.tensor([[-1., -1., -1.], [1., 1., 1.], [-5., 0., -1.], [5., 0., 1.]])
    assert torch.allclose(dv, want), f"deviation arithmetic wrong:\n{dv}"
    assert not torch.allclose(deviation(Z, torch.zeros(4, dtype=torch.long)), dv), \
        "CONTROL: one group must differ from two groups here"
    print("ok  deviation: per-country, mask-aware, unobserved cells 0; one-group control differs")

    # 3. no future leak. (i) dev at week k moves only when week k moves; (ii) the whole v2 forward at
    #    origin t is blind to everything after t. Control: a leaky deviation (window mean) is caught.
    jb = bundles.load("influenza_japan")
    Zf = torch.tensor(jb.transfer_view(), dtype=torch.float32)
    A = sparse_from_dense_np(jb.A_geo)
    Mt = torch.tensor(jb.M, dtype=torch.float32)
    g0 = torch.zeros(Zf.shape[0], dtype=torch.long)
    t = 150
    win = window_slice(Zf, t)

    def moved_cols(fn):
        base = fn(win)
        bad = []
        for k in range(win.shape[1]):
            w2 = win.clone(); w2[:, k, 0] += 5.0
            diff = (fn(w2) - base).abs().amax(dim=0) > 0
            if diff.any() and (diff.nonzero().flatten().tolist() != [k]):
                bad.append(k)
        return bad
    assert moved_cols(lambda w: deviation(w, g0)) == [], "deviation reads another week"
    assert moved_cols(lambda w: (w[..., 0] - w[..., 0].mean()) * w[..., 3]) != [], \
        "CONTROL: a window-mean deviation must be caught"
    enc = DeviationEncoder(dev_graph="real").eval()
    with torch.no_grad():
        torch.nn.init.normal_(enc.dev_proj.weight, std=0.1)          # branch live for this check
        o1 = enc(window_slice(Zf, t), A, Mt[:, t])
        Zc = Zf.clone(); Zc[:, t + 1:, :] = torch.randn_like(Zc[:, t + 1:, :]) * 50
        Mc = Mt.clone(); Mc[:, t + 1:] = 0
        o2 = enc(window_slice(Zc, t), A, Mc[:, t])
    assert torch.equal(o1, o2), "v2 output at origin t changed when only weeks after t changed"
    print("ok  no leak: deviation is same-week only (window-mean control caught); v2 output at origin "
          "t is identical after rewriting every week after t")

    # 4. strict superset, forward: the v1 checkpoint loaded into v2 with the branch at its zero init
    #    reproduces v1's output EXACTLY, on every arm. Control: a nonzero projection must differ.
    ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
    v1 = SharedEncoder(); v1.load_state_dict(ck["encoder"]); v1.eval()
    ad = Adapter(); ad.load_state_dict(ck["adapter"]); ad.eval()
    A_sh, _ = L.permute_adjacency(jb.A_geo, perm_seed_for(42))
    for g in ("real", "none", "shuffled"):
        v2 = DeviationEncoder(dev_graph=g)
        miss, unexp = v2.load_state_dict(ck["encoder"], strict=False)
        assert not unexp and all(k.startswith("dev_") for k in miss) and miss, "v1 keys do not map into v2"
        v2.bind(A_dev=sparse_from_dense_np(A_sh) if g == "shuffled" else None).eval()
        with torch.no_grad():
            for tt in (40, 150, 250):
                w = window_slice(Zf, tt)
                assert torch.equal(ad(v1(w, A, Mt[:, tt])), ad(v2(w, A, Mt[:, tt]))), \
                    f"{g}: zero-branch v2 does not reproduce v1 exactly at t={tt}"
            torch.nn.init.normal_(v2.dev_proj.weight, std=0.1)
            assert not torch.equal(ad(v1(w, A, Mt[:, tt])), ad(v2(w, A, Mt[:, tt]))), "CONTROL did not differ"
    print("ok  strict superset: v1 checkpoint in v2 with the zero-init branch reproduces v1's forward "
          "output bit for bit on all 3 arms at 3 origins; a nonzero branch differs")

    # 5. the arms are what they say.
    with torch.no_grad():
        w = window_slice(Zf, 150)
        Ap = sparse_from_dense_np(A_sh)
        for g, should_move in (("none", False), ("real", True)):
            e = DeviationEncoder(dev_graph=g).eval()
            torch.nn.init.normal_(e.dev_proj.weight, std=0.1)
            e(w, A, Mt[:, 150]); b1 = e.last_dev_out.clone()
            e(w, Ap, Mt[:, 150]); b2 = e.last_dev_out.clone()
            assert (not torch.equal(b1, b2)) == should_move, f"{g}: branch sensitivity to the map is wrong"
    for g, kw in (("shuffled", {}), ("real", {"A_dev": Ap})):
        try:
            DeviationEncoder(dev_graph=g).bind(**kw)(w, A, Mt[:, 150]); raise SystemExit(f"{g} guard off")
        except AssertionError:
            pass
    print("ok  arms: nograph branch blind to the map, graph branch reads it; shuffled needs its map, "
          "real refuses a second one")

    # 6. pairing: same seed => same v1 init AND same Adapter init as the default path.
    torch.manual_seed(42); e1, a1 = SharedEncoder(), Adapter()
    torch.manual_seed(42); e2, a2 = DeviationEncoder(), Adapter()
    s1, s2 = e1.state_dict(), e2.state_dict()
    assert all(torch.equal(s1[k], s2[k]) for k in s1), "v1 submodule init differs"
    assert all(torch.equal(x, y) for x, y in zip(a1.state_dict().values(), a2.state_dict().values())), \
        "Adapter init differs: the branch leaked into the main RNG stream"
    torch.manual_seed(42); SharedEncoder(); DilatedTCN(in_ch=2, d=D_DEV); a3 = Adapter()
    assert not all(torch.equal(x, y) for x, y in zip(a1.state_dict().values(), a3.state_dict().values())), \
        "CONTROL: an unforked extra module must shift the Adapter init"
    print("ok  pairing: v2 at seed 42 has v1's trunk init and v1's Adapter init; unforked control shifts it")

    # 7. identity permutation refused (a no-op shuffled arm would fake a null).
    caught = False
    for sd in range(500):
        if (np.random.default_rng(sd).permutation(2) == np.arange(2)).all():
            try:
                L.permute_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]]), perm_seed=sd)
            except AssertionError:
                caught = True
            break
    assert caught, "identity permutation was not refused"
    try:
        make_factory("v2graph", perm_seed=5); raise SystemExit("perm-seed guard off")
    except AssertionError:
        pass
    print("ok  identity permutation refused; only the shuffled arm accepts a perm seed")

    # 8. schema check, mutation-tested against the real v1 record.
    ref = json.loads(V1_REF.read_text())
    meta = dict(encoder_version="v2_deviation", v2_arm="v2graph", v2_dev_graph="real", v2_d_dev=D_DEV,
                v2_combine=COMBINE, v2_n_groups=1, v2_branch_params=1, v2_protocol_sha256=protocol_sha256(),
                v2_dev_shuffle_seed=None, v2_dev_shuffle_hash=None, v2_dev_shuffle_frac=None,
                v2_dev_contrib=0.1)
    good = [dict(r, **meta) for r in ref]
    assert_schema(good, "v2graph")
    mutants = {
        "drop a v1 key": lambda rs: rs[3].pop("node_mean"),
        "add a stray key": lambda rs: rs[0].__setitem__("shuffle_adj_seed", 1),
        "drop a v2 key": lambda rs: rs[5].pop("v2_arm"),
        "wrong arm label": lambda rs: rs[7].__setitem__("v2_arm", "v2nograph"),
        "drop a record": lambda rs: rs.pop(),
        "wrong protocol hash": lambda rs: rs[2].__setitem__("v2_protocol_sha256", "0" * 64),
    }
    for name, mut in mutants.items():
        rs = [dict(r) for r in good]
        mut(rs)
        try:
            assert_schema(rs, "v2graph"); raise SystemExit(f"schema mutant not caught: {name}")
        except AssertionError:
            pass
    print(f"ok  schema: v1 keys + {len(V2_KEYS)} v2 keys pass; all {len(mutants)} mutants caught")

    # 9. the decision logic on hand-built grids, and both rules' verdicts.
    H = bundles.HORIZONS

    def grid(v1=None, ng=None, sh=None):
        g = {(r, h, m): "noise" for r in ("v1", "v2nograph", "v2shuffled") for h in H for m in METRICS}
        for rf, cells in (("v1", v1), ("v2nograph", ng), ("v2shuffled", sh)):
            for (h, m), x in (cells or {}).items():
                g[(rf, h, m)] = x
        return g
    win3 = {(3, "rmse"): "better", (3, "mae"): "better"}
    win5 = {(5, "rmse"): "better", (5, "mae"): "better"}
    assert decide(grid())[0] == "FAIL"
    assert decide(grid(v1={(3, "rmse"): "better"}))[0] == "FAIL", "one metric must not pass (a)"
    assert decide(grid(v1={(10, "rmse"): "better", (10, "mae"): "better"}))[0] == "FAIL", "h10 is not (a)"
    assert decide(grid(v1={**win3, (15, "mae"): "worse"}))[0] == "FAIL", "harm anywhere fails (a)"
    assert decide(grid(v1=win3))[0] == "A_ONLY"
    assert decide(grid(v1=win3, ng=win3))[0] == "A_ONLY", "beating nograph alone is not (b)"
    assert decide(grid(v1=win3, ng=win3, sh=win3))[0] == "PASS"
    assert decide(grid(v1={**win3, **win5}, ng=win3, sh=win5))[0] == "A_ONLY", "(b) must hold at ONE horizon"
    assert decide(grid(v1=win3, ng=win3, sh={(3, "rmse"): "better"}))[0] == "A_ONLY", "(b) needs both metrics"
    for rule in ("gate", "tint"):
        assert cell_verdict({1: 10., 2: 12., 3: 11.}, {1: 5., 2: 7., 3: 6.1}, rule)[3] == "better"
        assert cell_verdict({1: 10., 2: 12., 3: 11.}, {1: 15., 2: 17., 3: 16.1}, rule)[3] == "worse"
        assert cell_verdict({1: 10., 2: 12., 3: 11.}, {1: 11., 2: 11., 3: 12.}, rule)[3] == "noise"
    d = np.array([-1., -1., -1., -1., -1.]) + np.array([-0.9, 0.9, -0.9, 0.9, 0.0])   # mean -1, sd 0.9
    base = {s: 100.0 for s in range(5)}
    arm = {s: 100.0 + d[s] for s in range(5)}
    assert cell_verdict(base, arm, "gate")[3] == "better" and cell_verdict(base, arm, "tint")[3] == "noise", \
        "|mean| = 1.11 sd at n=5: the gate rule calls it, the t-interval (needs 1.24 sd) does not"
    lo, hi = cell_verdict(base, arm, "tint")[4]
    assert lo < 0 < hi, "the interval must straddle zero here"
    print("ok  decision: FAIL/A_ONLY/PASS on 9 hand grids incl. one-metric, h10, harm, split-horizon "
          "traps; verdict signs right under both rules; the rules differ where they must")

    # 10. the seed count: formula, t correction, floor, cap, largest s, integrity.
    for s in (0.05, 0.10, 0.135, 0.20):
        sc = seed_count(s)
        assert sc["n_z"] == max(N_MIN, math.ceil(((1.96 + 0.84) * s / 0.033) ** 2)), f"n_z wrong at s={s}"
        k = 0.033 / s
        assert power(k, sc["n_t"], "tint") >= 0.8 and (sc["n_t"] == sc["n_z"] or power(k, sc["n_t"] - 1, "tint") < 0.8), \
            f"n_t is not the smallest powered count at s={s}"
        assert sc["n_t"] >= sc["n_z"] and sc["N"] == min(sc["n_t"], 134) and sc["underpowered"] == (sc["n_t"] > 134)
    assert seed_count(0.001)["N"] == 5, "floor at 5"
    big = seed_count(0.30)
    assert big["N"] == 134 and big["underpowered"] and big["power_at_N"] < 0.8, "cap at 134, flagged"
    ns = [seed_count(s)["N"] for s in (0.02, 0.05, 0.08, 0.11, 0.14)]
    assert ns == sorted(ns) and len(set(ns)) > 1, "N must grow with s"
    assert seed_count(0.05)["n_t"] > seed_count(0.05)["n_z"], "the t correction must add seeds at small n"
    summ = dict(comparisons={"v2graph_vs_v1": {"rel_sd": 0.05}, "v2graph_vs_v2nograph": {"rel_sd": 0.20},
                             "v2graph_vs_v2shuffled": {"rel_sd": 0.10}})
    assert seed_count_from_summary(summ)["N"] == seed_count(0.20)["N"] != seed_count(0.05)["N"], \
        "N must come from the LARGEST s"
    print(f"ok  seed count: normal formula, exact t correction, floor 5, cap 134 flagged, grows with s, "
          f"uses the largest s (s=0.05 -> {seed_count(0.05)['N']}, 0.10 -> {seed_count(0.10)['N']}, "
          f"0.135 -> {seed_count(0.135)['N']}, 0.20 -> {seed_count(0.20)['N']})")

    # 11. stage 1 shows validation only: fabricate val files AND test files (the test ones carry a
    #     sentinel), run the stage-1 summary, and require the val-derived numbers and no sentinel.
    tmp = Path(tempfile.mkdtemp(prefix="v2selfcheck_"))
    try:
        vals = {"v1": dict(zip(STAGE1_SEEDS, [100., 110., 90., 105., 95.])),
                "v2graph": dict(zip(STAGE1_SEEDS, [98., 109., 87., 104., 96.])),
                "v2nograph": dict(zip(STAGE1_SEEDS, [104., 113., 96., 108., 84.])),
                "v2shuffled": dict(zip(STAGE1_SEEDS, [99., 108., 91., 103., 94.]))}
        _fake_val(tmp, vals)
        v1_real = {s: json.loads(rpath(f"encoder__{DATASET}__seed{s}.json", root=RES).read_text())
                   for s in STAGE1_SEEDS}
        _fake_panel(tmp, v1_real, None, STAGE1_SEEDS, sentinel_test=True)
        for s in STAGE1_SEEDS:                          # a v1 "test" file in tmp too, also sentinel
            rpath(seeds_file(DATASET, s, "v1") + ".json", root=tmp, make=True).write_text(
                json.dumps([{"model": MODEL, "horizon": 3, "metric": "rmse", "country_macro": 987654.321}]))
        buf = io.StringIO()
        sp = tmp / "summary.json"
        with contextlib.redirect_stdout(buf):
            summ = write_stage1_summary(root=tmp, path=sp)
        text = buf.getvalue() + sp.read_text()
        assert "98765" not in text, "stage 1 showed a test value"
        v1m = float(np.mean(list(vals["v1"].values())))
        for arm, rf in COMPARISONS:
            dd = np.array([vals[arm][s] - vals[rf][s] for s in STAGE1_SEEDS])
            want = float(np.std(dd, ddof=1)) / v1m
            assert abs(summ["comparisons"][f"{arm}_vs_{rf}"]["rel_sd"] - want) < 1e-12, f"stage-1 sd wrong ({arm})"
        assert "mean_d" not in text and summ["s"] == max(c["rel_sd"] for c in summ["comparisons"].values())
        bad = json.loads(rpath(seeds_file(DATASET, 52, "v2graph") + "__val.json", root=tmp).read_text())
        bad[0]["split"] = "test"
        rpath(seeds_file(DATASET, 52, "v2graph") + "__val.json", root=tmp).write_text(json.dumps(bad))
        try:
            stage1_stats(root=tmp); raise SystemExit("a non-val file was accepted by stage 1")
        except AssertionError:
            pass
        print("ok  stage 1: statistics come from val files only (hand-computed sds match, no test sentinel "
              "reaches the output or the summary); a record not marked val is refused")

        # 12. the report refuses test until stage 2 is complete, then decides under the t-interval rule.
        _fake_val(tmp, vals)                            # restore the good val files
        with contextlib.redirect_stdout(io.StringIO()):
            write_stage1_summary(root=tmp, path=sp)
        n_full = seed_count_from_summary(json.loads(sp.read_text()))["N"]
        assert n_full > 5, f"control void: this summary must ask for more than 5 seeds, got {n_full}"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dec = report(root=tmp, path=sp)
        assert dec is None and "REPORT REFUSED" in buf.getvalue() and "98765" not in buf.getvalue(), \
            "report must refuse, without reading test, while stage 2 is incomplete"
        seeds = SEED_SEQ[:n_full]
        win = lambda arm, h, m, s, x: x * 0.7 if (arm == "v2graph" and h == 3 and m in METRICS) else x
        _fake_panel(tmp, v1_real, win, seeds)
        assert decision_from_disk(root=tmp, path=sp)[0][0] == "PASS", "a clear h3 win on both metrics must PASS"
        # The rules disagree: at N seeds the t-interval needs only t_{.975,N-1}/sqrt(N) paired sds, the
        # gate rule a full one. Build |mean d| = 0.6 sd on v2graph at h3: the t-interval calls it, the
        # gate rule does not, and the decision must follow the t-interval (PASS, not FAIL).
        _fake_panel(tmp, v1_real, lambda a, h, m, s, x: x, seeds)          # every arm equal to v1
        e = np.array([(-1.0) ** i for i in range(len(seeds))]); e -= e.mean(); e /= e.std(ddof=1)
        off = dict(zip(seeds, 50.0 * (-1.0 + e / 0.6)))                   # mean -50, sd 50/0.6
        v1t = {s: _read(_v1_path(DATASET, s, tmp)) for s in seeds}
        for s in seeds:
            p = rpath(seeds_file(DATASET, s, "v2graph") + ".json", root=tmp)
            rs = json.loads(p.read_text())
            for r in rs:
                if r["model"] == MODEL and r["horizon"] == 3 and r["metric"] in METRICS:
                    r["country_macro"] = r["node_mean"] = v1t[s][(3, r["metric"])][FIELD] + off[s]
            p.write_text(json.dumps(rs))
        dec, g, probs, _ = decision_from_disk(root=tmp, path=sp)
        cell = g[("v1", "v2graph", 3, "rmse")]
        ratio = abs(cell["d"]) / cell["sd"]
        assert 0.55 < ratio < 0.65, f"control void, |mean|/sd = {ratio:.3f}"
        assert cell["gate"] == "noise" and cell["verdict"] == "better" and dec[0] == "PASS", \
            f"decision must follow the t-interval: gate={cell['gate']} tint={cell['verdict']} dec={dec}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"ok  report: refuses to read test while stage 2 is incomplete (N={n_full} asked, 5 present); "
          "PASS on a clear win; where the t-interval calls a win and the gate rule does not, the decision "
          "follows the t-interval")

    # 13. the repro gate: missing, failing, other protocol, other code -> blocked; passing -> open.
    tmp = Path(tempfile.mkdtemp(prefix="v2repro_"))
    try:
        good = dict(identical=True, max_rel_diff=0.0, protocol_sha256=protocol_sha256(), code_sha256=code_sha256())
        cases = {"missing": None, "differ": dict(good, identical=False, max_rel_diff=1e-3),
                 "other protocol": dict(good, protocol_sha256="0" * 64),
                 "other code": dict(good, code_sha256=dict(code_sha256(), **{"train/loop.py": "0" * 64}))}
        for name, rec in cases.items():
            p = tmp / f"{name}.json"
            if rec is not None:
                p.write_text(json.dumps(rec))
            assert not _repro_ok(p)[0], f"repro gate open on '{name}'"
        (tmp / "good.json").write_text(json.dumps(good))
        assert _repro_ok(tmp / "good.json")[0], "repro gate shut on a passing record"
        # the guard itself, under controlled state (never the live files)
        mod = sys.modules[__name__]
        saved = (mod._protocol_committed, mod.REPRO, mod.STAGE1_SUMMARY)

        def refuses(stage):
            try:
                _guard(stage)
                return False
            except AssertionError:
                return True
        try:
            mod.STAGE1_SUMMARY = tmp / "no_summary.json"
            mod._protocol_committed = lambda: False
            mod.REPRO = tmp / "good.json"
            assert refuses(1), "stage 1 ran without a committed protocol"
            mod._protocol_committed = lambda: True
            mod.REPRO = tmp / "missing.json"
            assert refuses(1), "stage 1 ran without a repro-check record"
            mod.REPRO = tmp / "differ.json"
            assert refuses(1), "stage 1 ran after a failing repro-check"
            mod.REPRO = tmp / "good.json"
            assert not refuses(1), "stage 1 refused with a committed protocol and a passing repro-check"
            assert refuses(2), "stage 2 ran without a stage-1 summary"
        finally:
            mod._protocol_committed, mod.REPRO, mod.STAGE1_SUMMARY = saved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok  repro gate: blocked when missing, failing, under another protocol or with changed code; "
          "stage 1 refuses without a commit or a passing repro record and opens with both; stage 2 "
          "refuses without a stage-1 summary")

    # 14. no free COVID path: the parser has no --arm or --dataset for training.
    for bad in (["--arm", "v2graph"], ["--dataset", "covid_us-states", "--arm", "v2graph"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _parser().parse_args(bad)
            raise SystemExit(f"parser accepted {bad}")
        except SystemExit as e:
            if "parser accepted" in str(e):
                raise AssertionError(str(e))
    print("ok  no free COVID path: --arm is gone; only --stage1 and --stage2 train COVID")

    # 15. the val scorer is train_one's test assembly on another split: applied to the released japan
    #     checkpoint on TEST it reproduces the released japan record.
    dev = L.DEVICE
    enc_j, ad_j = load_v1_ckpt("influenza_japan", 42, dev)
    exact, worst = _record_diff([r for r in json.loads(V1_REF.read_text()) if r["model"] == MODEL],
                                score_split(enc_j, ad_j, "influenza_japan", 42, "test"))
    assert worst < 1e-6, f"score_split on test does not reproduce the released record ({worst:.3e})"
    print(f"ok  val scorer: on the released japan checkpoint and the test split it reproduces the released "
          f"record ({'bit for bit' if exact else f'max rel diff {worst:.1e}'}, device {dev})")

    # 16. routing, protocol presence, seed parsing, the seed list.
    p = rpath(seeds_file(DATASET, 42, "v2shuffled") + ".json", root=HERE)
    assert p.parent == HERE / "single" and p.name.endswith("__v2shuf.json"), p
    assert rpath(seeds_file("influenza_japan", 42, "v2graph") + "__smoke.json", root=HERE).parent == HERE / "misc"
    assert _seed_from(p.name) == 42 and _seed_from(seeds_file(DATASET, 52, "v2graph") + "__val.json") == 52
    assert SEED_SEQ[:5] == (42, 52, 62, 72, 82) and SEED_SEQ[-1] == 1372 and len(set(SEED_SEQ)) == 134
    assert PROTOCOL.exists(), "the protocol must exist before anything runs"
    print(f"ok  routing and seeds: records to ablation/single, smoke to ablation/misc; seeds 42..1372 step 10; "
          f"protocol sha256 {protocol_sha256()[:16]}...")
    print("selfcheck ok")
    return 0


def _mutate_selfcheck():
    """The check on the check: plant bugs in this module, one at a time; the selfcheck must fail on
    every one. Seconds to a minute."""
    import contextlib
    import io
    mod = sys.modules[__name__]
    names = ("cell_verdict", "decide", "deviation", "assert_schema", "_s_of", "_read_val", "missing_runs",
             "RULE", "_repro_ok", "seed_count")
    orig = {k: getattr(mod, k) for k in names}

    def flipped(ref, arm, rule=None):
        dm, dsd, n, v, ci = orig["cell_verdict"](ref, arm, rule)
        return dm, dsd, n, {"better": "worse", "worse": "better"}.get(v, v), ci

    def lax(recs, arm, ref=V1_REF):
        try:
            orig["assert_schema"](recs, arm, ref)
        except AssertionError as e:
            if "unexpected" in str(e) and "missing []" in str(e):
                return
            raise

    def reads_test(ds, s, arm, root=None):
        rs = json.loads(rpath(seeds_file(ds, s, arm) + ".json", root=root or HERE).read_text())
        return {(r["horizon"], r["metric"]): r[FIELD] for r in rs if r["model"] == MODEL}

    def z_only(s):
        out = orig["seed_count"](s)
        return dict(out, n_t=out["n_z"], N=min(out["n_z"], N_CAP), underpowered=out["n_z"] > N_CAP)

    plants = (("flipped verdict sign", "cell_verdict", flipped),
              ("decision ignores harm", "decide", lambda v: orig["decide"]({k: ("noise" if x == "worse" else x) for k, x in v.items()})),
              ("leaky deviation", "deviation", lambda Z, g: (Z[..., 0] - Z[..., 0].mean()) * Z[..., 3]),
              ("schema allows stray keys", "assert_schema", lax),
              ("N from the smallest s", "_s_of", lambda summ: min(c["rel_sd"] for c in summ["comparisons"].values())),
              ("stage 1 reads the test file", "_read_val", reads_test),
              ("report skips the completeness check", "missing_runs", lambda seeds, root=None: []),
              ("decision uses the gate rule", "RULE", "gate"),
              ("repro gate always open", "_repro_ok", lambda path=None: (True, "ok")),
              ("seed count without the t correction", "seed_count", z_only))
    missed = []
    for name, attr, fn in plants:
        setattr(mod, attr, fn)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _selfcheck()
            missed.append(name)
            print(f"  MISSED: {name}")
        except Exception as e:                           # AssertionError, SystemExit, or a crash
            print(f"  caught: {name}  ({type(e).__name__})")
        except SystemExit:
            print(f"  caught: {name}  (SystemExit)")
        finally:
            setattr(mod, attr, orig[attr])
    print(f"mutate-selfcheck: {len(plants) - len(missed)} of {len(plants)} planted bugs caught")
    return 1 if missed else 0


def _smoke(epochs=3):
    """influenza_japan ONLY, seed 42, a few epochs. Never COVID."""
    L.RESULTS = HERE
    ds, s = "influenza_japan", 42
    print(f"smoke: {ds} seed{s}, {epochs} epochs, device {L.DEVICE}")

    # 1. strict superset in TRAINING (see protocol section 2): raw clip, then clip over nonzero grads.
    fac0 = lambda: make_factory("v2graph", None, dev_gain=0.0)
    t0 = time.time(); r_v1, *_ = L.train_one(ds, s, epochs=epochs, verbose=False); t_v1 = time.time() - t0
    t0 = time.time(); r_z, *_ = L.train_one(ds, s, epochs=epochs, verbose=False, encoder_factory=fac0())
    t_z = time.time() - t0
    exact_raw, worst_raw = _record_diff(r_v1, r_z)
    clip = torch.nn.utils.clip_grad_norm_
    torch.nn.utils.clip_grad_norm_ = lambda ps, mx, *a, **k: clip(
        [p for p in ps if p.grad is not None and bool(p.grad.any())], mx, *a, **k)
    try:
        r_v1e, *_ = L.train_one(ds, s, epochs=epochs, verbose=False)
        r_ze, *_ = L.train_one(ds, s, epochs=epochs, verbose=False, encoder_factory=fac0())
    finally:
        torch.nn.utils.clip_grad_norm_ = clip
    exact_eq, worst_eq = _record_diff(r_v1e, r_ze)
    print(f"  training superset, clip norm as train_one computes it: v1 vs v2graph(gain 0) records "
          f"{'IDENTICAL' if exact_raw else 'DIFFER'}, max rel diff {worst_raw:.3e} after {epochs} epochs")
    print(f"  training superset, clip norm over nonzero grads in both runs: records "
          f"{'IDENTICAL' if exact_eq else 'DIFFER'}, max rel diff {worst_eq:.3e}")
    print(f"  wall time v1 {t_v1:.1f}s, v2 gain-0 {t_z:.1f}s")
    assert exact_eq, "with the clip order equalised, v2 at gain 0 must train to v1's records exactly"

    # 1b. v2 itself is deterministic run to run.
    ra, *_ = L.train_one(ds, s, epochs=epochs, verbose=False, encoder_factory=make_factory("v2graph"))
    rb, *_ = L.train_one(ds, s, epochs=epochs, verbose=False, encoder_factory=make_factory("v2graph"))
    exact_det, worst_det = _record_diff(ra, rb)
    print(f"  determinism: v2graph twice at seed {s}: records {'IDENTICAL' if exact_det else 'DIFFER'} "
          f"(max rel diff {worst_det:.3e})")
    assert exact_det, "v2 is not deterministic run to run on this box"

    # 1c. the measurements behind the determinism design choices, so the protocol's numbers are on disk.
    torch.backends.cudnn.deterministic = True
    try:
        r_flag, *_ = L.train_one(ds, s, epochs=epochs, verbose=False)
    finally:
        torch.backends.cudnn.deterministic = False
    _, worst_flag = _record_diff(r_v1, r_flag)
    b, Z, Mt, A = _panel_tensors(ds, L.DEVICE)
    g0 = torch.zeros(Z.shape[0], dtype=torch.long, device=Z.device)

    def dev_index_add(Zw, group):                    # the first build, kept only to measure it
        x, obs = Zw[..., 0], Zw[..., 3]
        tot = x.new_zeros(1, x.shape[1]).index_add_(0, group, x * obs)
        cnt = x.new_zeros(1, x.shape[1]).index_add_(0, group, obs)
        return (x - (tot / cnt.clamp(min=1))[group]) * obs
    ts = list(range(19, 300, 7))
    vary = {}
    for name, fn in (("index_add", dev_index_add), ("matmul", deviation)):
        vary[name] = sum(any(not torch.equal(fn(window_slice(Z, t), g0), fn(window_slice(Z, t), g0))
                             for _ in range(19)) for t in ts)
    from models import Adapter, SharedEncoder, pinball_loss, targets_and_mask
    ymod = torch.tensor(b.y, dtype=torch.float32, device=Z.device)
    trm = torch.tensor(b.masks()["train"], dtype=torch.float32, device=Z.device)
    counts, norms = {}, {}
    for name, mk in (("v1", SharedEncoder), ("v2", lambda: DeviationEncoder(dev_gain=0.0))):
        torch.manual_seed(s)
        e, adp = mk().to(Z.device).train(), Adapter().to(Z.device)
        tgt, msk = targets_and_mask(ymod, Mt, trm, 150, Z.device)
        pinball_loss(adp(e(window_slice(Z, 150), A, Mt[:, 150])), tgt, msk).backward()
        ps = list(e.parameters()) + list(adp.parameters())
        counts[name] = sum(p.grad is not None for p in ps)
        norms[name] = float(torch.nn.utils.clip_grad_norm_(ps, 1.0))
    print(f"  cudnn.deterministic on moves v1 itself: max rel diff {worst_flag:.3e} after {epochs} epochs")
    print(f"  deviation repeats varying, of {len(ts)} origins x 20: index_add_ {vary['index_add']}, "
          f"one-hot matmul {vary['matmul']}")
    print(f"  gradient tensors v1 {counts['v1']}, v2 {counts['v2']}; clip norm v1 {norms['v1']:.9f}, "
          f"v2 gain-0 {norms['v2']:.9f}, gap {norms['v2'] - norms['v1']:+.3e}")

    # 2. each arm end to end, branch live, records + archives + checkpoint to ablation/misc; the val
    #    scorer on each trained arm.
    times = {}
    for arm in ARMS:
        t0 = time.time()
        recs, dt = train_cell(ds, s, arm, epochs, smoke=True)
        times[arm] = dt
        r0 = next(r for r in recs if r["model"] == MODEL and r["metric"] == "rmse" and r["horizon"] == 3)
        print(f"  {arm:10} {dt:5.1f}s  schema OK ({len(recs)} records, {len(set(r0))} keys)  "
              f"branch params {r0['v2_branch_params']}  dev_contrib {r0['v2_dev_contrib']:.4f}  "
              f"shuffle frac {r0['v2_dev_shuffle_frac']}  h3 rmse {r0[FIELD]:.2f}")
    ck = torch.load(rpath(seeds_file(ds, s, "v2graph") + "__smoke__ckpt.pt", root=HERE), map_location=L.DEVICE,
                    weights_only=False)
    e = make_factory("v2graph")(b, "learned", L.DEVICE)[0]
    e.load_state_dict(ck["encoder"])
    adp = Adapter().to(L.DEVICE); adp.load_state_dict(ck["adapter"])
    vr = score_split(e, adp, ds, s, "val")
    v3 = next(r for r in vr if r["metric"] == "rmse" and r["horizon"] == 3)
    print(f"  val scorer on the v2graph smoke checkpoint: {len(vr)} val records, h3 rmse {v3[FIELD]:.2f}")

    mt = sorted((HERE / "single" / f"encoder__covid_us-states__seed{x}__shufadj.json").stat().st_mtime
                for x in STAGE1_SEEDS)
    gaps = np.diff(mt)
    ratio = float(np.mean(list(times.values()))) / t_v1
    per = float(np.median(gaps)) * ratio
    print(f"  v1 COVID per run ~{np.median(gaps):.0f}s (median gap of shufadj COVID mtimes {np.round(gaps)}); "
          f"v2/v1 time ratio on this smoke {ratio:.2f} -> v2 COVID ~{per:.0f}s per run; "
          f"stage 1 (15 v2 runs) ~{15 * per / 60:.0f} min")
    summary = dict(dataset=ds, seed=s, epochs=epochs, device=L.DEVICE,
                   superset_raw_identical=exact_raw, superset_raw_max_rel_diff=worst_raw,
                   superset_equalised_identical=exact_eq, superset_equalised_max_rel_diff=worst_eq,
                   v2_run_to_run_identical=exact_det, v2_run_to_run_max_rel_diff=worst_det,
                   cudnn_flag_moves_v1_max_rel_diff=worst_flag, deviation_origins_tested=len(ts),
                   deviation_varying_index_add=vary["index_add"], deviation_varying_matmul=vary["matmul"],
                   grad_tensors_v1=counts["v1"], grad_tensors_v2=counts["v2"],
                   clip_norm_v1=norms["v1"], clip_norm_v2_gain0=norms["v2"],
                   arm_seconds=times, v1_seconds=t_v1, v2_over_v1_time=ratio,
                   v1_covid_run_seconds_median=float(np.median(gaps)), v2_covid_run_seconds=per,
                   branch_params=r0["v2_branch_params"], protocol_sha256=protocol_sha256())
    out = HERE / "misc" / "v2_smoke_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"smoke done; records and {out.name} in ablation/misc/")
    return 0


# --------------------------------------------------------------------------- #
def _parser():
    ap = argparse.ArgumentParser(description="v2 deviation channel, two-stage COVID mechanism test")
    g = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--repro-check", "--stage1", "--seed-count", "--stage2", "--report",
                 "--selfcheck", "--mutate-selfcheck", "--power", "--smoke"):
        g.add_argument(flag, action="store_true")
    ap.add_argument("--dry", action="store_true", help="with --stage1/--stage2: print the plan, train nothing")
    ap.add_argument("--power-dataset", default=DATASET, help="with --power only (planning, reads disk)")
    return ap


def main():
    a = _parser().parse_args()
    if a.selfcheck:
        return _selfcheck()
    if a.mutate_selfcheck:
        return _mutate_selfcheck()
    if a.power:
        power_check(a.power_dataset)
        return 0
    if a.smoke:
        return _smoke()
    if a.repro_check:
        return _repro_check()
    if a.stage1:
        return run_stage1(dry=a.dry)
    if a.seed_count:
        try:
            print_seed_count()
        except AssertionError as e:
            print(f"SEED COUNT REFUSED: {e}")
            return 1
        return 0
    if a.stage2:
        return run_stage2(dry=a.dry)
    if a.report:
        return 0 if report() is not None else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

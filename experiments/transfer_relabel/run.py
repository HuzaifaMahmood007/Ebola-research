"""Transfer relabel test, inference only: does a trunk trained on OTHER diseases use the real map when
it forecasts a held-out disease? Relabels the adjacency among districts of equal degree, scores with
the archived scorer. Rule and scope: progress/decisions/Transfer_Relabel_Protocol.md, whose sha256 is
stamped into every record. Writes only inside experiments/transfer_relabel/. Never touches Ebola.

  conda run -n ebola-train python experiments/transfer_relabel/run.py --selfcheck
  conda run -n ebola-train python experiments/transfer_relabel/run.py --dry --sha256 HEX
  conda run -n ebola-train python experiments/transfer_relabel/run.py --sha256 HEX       # scored run
  conda run -n ebola-train python experiments/transfer_relabel/run.py --report
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LANE = ROOT / "experiments" / "transfer_relabel"
PROTOCOL = ROOT / "progress" / "decisions" / "Transfer_Relabel_Protocol.md"
LODO = ROOT / "results" / "lodo"
CODE = ("experiments/transfer_relabel/run.py", "explain.py", "models/encoder.py", "models/spatial.py",
        "models/adapters.py", "score.py", "train/loop.py", "train/joint.py", "to_schema.py", "bundles.py")


# --------------------------------------------------------------------------- #
# Protocol: hash, constants, stamp
# --------------------------------------------------------------------------- #
def sha256(path):
    """LF-normalised, so a CRLF checkout cannot change the stamp (.gitattributes forces LF anyway)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def constants(text=None):
    """The fenced `KEY = v1 v2` block after the `<!-- constants -->` marker. The frozen doc is the one
    source; code never carries its own copy of a threshold."""
    text = PROTOCOL.read_text(encoding="utf-8") if text is None else text
    block = text.split("<!-- constants -->", 1)[1].split("```", 2)[1]
    raw = {k.strip(): v.split() for k, v in (ln.split("=", 1) for ln in block.strip().splitlines())}
    ints = ("SEEDS", "PERM_SEEDS", "HORIZONS", "EXPLORATORY_SEED")
    floats = ("MATERIALITY", "T_CRIT", "STRENGTH_FLOOR", "DRIFT_TOL")
    c = {k: ([int(x) for x in v] if k in ints else float(v[0]) if k in floats else v) for k, v in raw.items()}
    c["EXPLORATORY_SEED"], c["DECIDING_PERTURBATION"] = c["EXPLORATORY_SEED"][0], c["DECIDING_PERTURBATION"][0]
    assert c["DECIDING_PERTURBATION"] == c["PERTURBATIONS"][0], "deciding perturbation must run first"
    return c


def check_stamp(given, committed):
    got = sha256(PROTOCOL)
    if given != got:
        sys.exit(f"protocol sha256 is {got}, the stamp given is {given}: refusing to run")
    if committed:
        rel = PROTOCOL.relative_to(ROOT).as_posix()
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                                 capture_output=True).returncode == 0
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
        if not (tracked and clean):
            sys.exit(f"{rel} is not committed with no local edits: the scored run needs the frozen file")
    return got


def write(path, obj):
    path = Path(path).resolve()
    if LANE.resolve() not in path.parents:
        raise PermissionError(f"refusing to write outside {LANE}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


# --------------------------------------------------------------------------- #
# Perturbations (graph only, no data)
# --------------------------------------------------------------------------- #
def degclass_perm(deg, seed):
    """Shuffle district labels within each degree class: every district keeps its exact degree."""
    rng, p = np.random.default_rng(seed), np.arange(len(deg))
    for d in np.unique(deg):
        idx = np.flatnonzero(deg == d)
        p[idx] = idx[rng.permutation(len(idx))]
    return p


def global_perm(deg, seed):
    return np.random.default_rng(seed).permutation(len(deg))


PERTURB = {"degclass": degclass_perm, "global": global_perm}


def neighbours(A_np):
    return [set(np.flatnonzero(A_np[i]).tolist()) for i in range(A_np.shape[0])]


def strength(nb, p):
    """(share of each district's relabelled neighbours that are NOT its real neighbours, averaged over
    districts with at least one; share of districts whose neighbour count changed)."""
    pinv, ch, cnt = np.argsort(p), [], []
    for i, real in enumerate(nb):
        new = {int(pinv[m]) for m in nb[p[i]]}
        if new:
            ch.append(1 - len(new & real) / len(new))
        cnt.append(len(new) != len(real))
    return float(np.mean(ch)), float(np.mean(cnt))


def relabel(A, p):
    """Sparse A[p][:, p] without densifying: edge (a, b) moves to (pinv[a], pinv[b])."""
    import torch
    pinv = torch.as_tensor(np.argsort(p), device=A.device)
    return torch.sparse_coo_tensor(pinv[A.indices()], A.values(), A.shape).coalesce()


# --------------------------------------------------------------------------- #
# Model, forecast, score
# --------------------------------------------------------------------------- #
def plan(c):
    """(family, role, fold, seed, head, checkpoint, archived-stem template)."""
    rows = [("ldo3", "primary", f, s, "adapted", f"encoder_ldo3__{f}__seed{s}__ckpt.pt",
             "encoder_ldo3__{panel}__seed%d" % s) for f in c["PRIMARY_FOLDS"] for s in c["SEEDS"]]
    s = c["EXPLORATORY_SEED"]
    rows += [("ldo3full_zeroshot", "exploratory", f, s, "zeroshot", f"encoder_ldo3full__{f}__seed{s}__ckpt.pt",
              "encoder_ldo3full_zeroshot__{panel}__seed%d" % s) for f in c["EXPLORATORY_ZS_FOLDS"]]
    return rows


def load(ck, head):
    import torch
    from models import Adapter, SharedEncoder
    from train.loop import DEVICE
    b = torch.load(ck, map_location=DEVICE, weights_only=False)
    enc = SharedEncoder().to(DEVICE); enc.load_state_dict(b["encoder"]); enc.eval()
    ad = Adapter().to(DEVICE)
    ad.load_state_dict(b["adapter"] if head == "adapted" else b["meta"]["zeroshot_adapter"]); ad.eval()
    return enc, ad, b["meta"]


def forecast(enc, ad, d, A_mix=None, quant=False):
    """Count-space medians {h: [N, T]} over the test fold, plus {h: [N, K, Q]} quantiles if asked.
    A_mix None is the archived path enc(Z, A, M), as in train.joint._test_dataset. Otherwise the mixer
    gets A_mix and the degree feature reads the TRUE graph (explain.forward_fixed_deg)."""
    import torch
    from bundles import HORIZONS
    from explain import forward_fixed_deg
    from models import MEDIAN_IDX, window_slice
    from to_schema import invert_scaler
    T = d.b.X.shape[1]
    med = {h: np.zeros((d.N, T)) for h in HORIZONS}
    q = {h: np.zeros((d.N, len(d.te), 5), dtype=np.float32) for h in HORIZONS} if quant else None
    with torch.no_grad():
        for k, t in enumerate(d.te):
            Zt, Mt = window_slice(d.Z, t), d.Mt[:, t]
            h = enc(Zt, d.A_solo, Mt) if A_mix is None else forward_fixed_deg(enc, Zt, d.A_solo, A_mix, Mt)
            out = ad(h).cpu().numpy()
            for j, hh in enumerate(HORIZONS):
                med[hh][:, t + hh] = invert_scaler(out[:, j, MEDIAN_IDX:MEDIAN_IDX + 1], d.b.scaler)[:, 0]
                if quant:
                    for qi in range(out.shape[2]):
                        q[hh][:, k, qi] = invert_scaler(out[:, j, qi:qi + 1], d.b.scaler)[:, 0]
    return med, q


def score(d, med, seed):
    """The archived scorer, untouched: {'h3|rmse': [country_macro, node_mean], ...}."""
    from train.loop import score_predictions
    recs, _, _ = score_predictions("transfer_relabel", d.name, seed, med, d.b, d.te)
    return {f"h{r['horizon']}|{r['metric']}": [float(r["country_macro"]), float(r["node_mean"])] for r in recs}


def archived(stem):
    recs = json.loads((LODO / f"{stem}.json").read_text())
    return {f"h{r['horizon']}|{r['metric']}": [float(r["country_macro"]), float(r["node_mean"])] for r in recs}


def gap(a, b):
    """|a - b| / max(|b|, 1); NaN on both sides is agreement, NaN on one side is infinite drift."""
    if np.isnan(a) and np.isnan(b):
        return 0.0
    return float("inf") if np.isnan(a) or np.isnan(b) else abs(a - b) / max(abs(b), 1.0)


def drift(real, arch, q, npz):
    assert set(real) == set(arch), f"metric keys differ: {set(real) ^ set(arch)}"
    m = max(gap(a, b) for k in arch for a, b in zip(real[k], arch[k]))
    z = np.load(npz)
    qd = max(float(np.abs(q[h] - z[f"h{h}__quantiles"]).max()) / max(float(np.abs(z[f"h{h}__quantiles"]).max()), 1.0)
             for h in q)
    return m, qd


# --------------------------------------------------------------------------- #
# One checkpoint
# --------------------------------------------------------------------------- #
def run_one(row, c, sha, dry):
    import torch
    from train.joint import _prepare
    from train.loop import DEVICE
    fam, role, fold, seed, head, ck, stem = row
    out = LANE / f"{fam}__{fold}__seed{seed}.json"
    if not dry and out.exists() and json.loads(out.read_text()).get("protocol_sha256") == sha:
        print(f"[skip] {out.name} exists under this protocol"); return None
    enc, ad, meta = load(LODO / ck, head)
    from train.lodo import _ldo3_plan
    assert list(meta["adapter_scope"]) == _ldo3_plan(fold)[2], f"{ck}: panels {meta['adapter_scope']}"
    ds, _ = _prepare(list(meta["adapter_scope"]), DEVICE)
    rec = dict(lane="EXPLORATORY", synthetic=False, protocol_sha256=sha, family=fam, role=role, fold=fold,
               seed=seed, head=head, checkpoint=ck, checkpoint_sha256=sha256(LODO / ck),
               code_sha256={f: sha256(ROOT / f) for f in CODE}, perm_seeds=c["PERM_SEEDS"],
               device=str(DEVICE), torch=torch.__version__, panels={})
    for d in ds:
        t0 = time.time()
        med, q = forecast(enc, ad, d, quant=True)
        t1 = time.time()
        real = score(d, med, seed)
        t2 = time.time()
        arch = archived(stem.format(panel=d.name))
        dm, dq = drift(real, arch, q, LODO / f"{stem.format(panel=d.name)}__quantiles.npz")
        ok = dm <= c["DRIFT_TOL"] and dq <= c["DRIFT_TOL"]
        print(f"  {fam:18s} {fold:9s} seed{seed} {d.name:22s} drift metrics {dm:.2e} quantiles {dq:.2e} "
              f"tol {c['DRIFT_TOL']:.0e} {'PASS' if ok else 'FAIL'}  fwd {t1 - t0:.1f}s score {t2 - t1:.1f}s")
        if not ok:
            sys.exit(f"DRIFT GATE FAILED on {ck} / {d.name}: aborting, this checkpoint writes nothing")
        A_np = np.asarray(d.b.A_geo)
        deg, nb = (A_np != 0).sum(1), neighbours(A_np)
        P = dict(n_nodes=d.N, real=real, archived=arch, unseen_map=d.name in c["UNSEEN_MAP_PANELS"],
                 drift=dict(metrics=dm, quantiles=dq, tol=c["DRIFT_TOL"], passed=ok),
                 seconds=dict(real_forward=t1 - t0, score=t2 - t1), draws={})
        for name in c["PERTURBATIONS"]:
            rows = []
            for ps in c["PERM_SEEDS"]:
                p = PERTURB[name](deg, ps)
                moved = float((p != np.arange(len(p))).mean())
                assert moved > 0, f"perm seed {ps} is the identity on {d.name}: refusing a silent no-op"
                if name == "degclass":                     # degree of i after relabel is deg[p[i]]
                    assert np.array_equal(deg[p], deg), "degclass changed a district's degree"
                ch, cnt = strength(nb, p)
                r = dict(perm_seed=ps, perm_sha1=hashlib.sha1(p.astype(np.int64).tobytes()).hexdigest()[:12],
                         moved=moved, nbr_changed=ch, count_changed=cnt)
                if dry:
                    if name == c["PERTURBATIONS"][0] and ps == c["PERM_SEEDS"][0]:
                        ta = time.time()
                        forecast(enc, ad, d, relabel(d.A_solo, p))      # timed, discarded, NEVER scored
                        P["seconds"]["perm_forward"] = time.time() - ta
                else:
                    r["metrics"] = score(d, forecast(enc, ad, d, relabel(d.A_solo, p))[0], seed)
                rows.append(r)
            P["draws"][name] = rows
            print(f"      {name:8s} neighbours changed {np.mean([r['nbr_changed'] for r in rows]):.3f}  "
                  f"districts with new neighbour count {np.mean([r['count_changed'] for r in rows]):.3f}")
        rec["panels"][d.name] = P
    if not dry:
        write(out, rec)
    return rec


# --------------------------------------------------------------------------- #
# Report: the decision rule, computed from the raw records
# --------------------------------------------------------------------------- #
def cell(e, c):
    e = np.asarray(e, dtype=float)
    assert len(e) == len(c["SEEDS"]), f"rule needs {len(c['SEEDS'])} seeds, got {len(e)}"
    m = float(e.mean()); half = c["T_CRIT"] * float(e.std(ddof=1)) / np.sqrt(len(e))
    lo, hi, mat = m - half, m + half, c["MATERIALITY"]
    v = ("USED" if m >= mat and lo > 0 else "HURTS" if m <= -mat and hi < 0
         else "NOT USED" if lo > -mat and hi < mat else "INCONCLUSIVE")
    return dict(mean=m, lo=lo, hi=hi, verdict=v)


def effect(rec, panel, pert, key):
    """Per checkpoint: mean over draws of the change. Relative for rmse/mae, absolute drop for pcc,
    so positive always means the relabelled map forecast worse."""
    P = rec["panels"][panel]; real = P["real"][key][0]
    vals = [r["metrics"][key][0] for r in P["draws"][pert]]
    return float(np.mean([real - v for v in vals])) if key.endswith("pcc") else \
        float(np.mean([(v - real) / real for v in vals]))


def panel_verdict(cells, strength_, c):
    hs = c["HORIZONS"]
    used = [h for h in hs if all(cells[f"h{h}|{m}"]["verdict"] == "USED" for m in c["DECIDING_METRICS"])]
    hurt = [h for h in hs if all(cells[f"h{h}|{m}"]["verdict"] == "HURTS" for m in c["DECIDING_METRICS"])]
    allnot = all(cells[f"h{h}|{m}"]["verdict"] == "NOT USED" for h in hs for m in c["DECIDING_METRICS"])
    v = "MIXED" if used and hurt else "USED" if used else "HURTS" if hurt else "NOT USED" if allnot \
        else "INCONCLUSIVE"
    if strength_ < c["STRENGTH_FLOOR"] and v in ("NOT USED", "INCONCLUSIVE"):
        v = "WEAK"
    return dict(verdict=v, used_horizons=used, hurts_horizons=hurt, strength=strength_)


def outcome(verdicts):
    vs = list(verdicts.values())
    if "MIXED" in vs or ("USED" in vs and "HURTS" in vs):
        return "O5"
    if "USED" in vs:
        return "O3"
    if "HURTS" in vs:
        return "O4"
    live = [v for v in vs if v != "WEAK"]
    return "O1" if live and all(v == "NOT USED" for v in live) else "O2"


def report(c, sha, lane=LANE):
    recs = [json.loads(p.read_text()) for p in sorted(lane.glob("*__seed*.json"))]
    assert recs, f"no records in {lane}"
    bad = [r["checkpoint"] for r in recs if r["protocol_sha256"] != sha]
    assert not bad, f"records stamped with another protocol: {bad}"
    assert len({r["synthetic"] for r in recs}) == 1, "synthetic and real records mixed"
    prim = [r for r in recs if r["role"] == "primary"]
    want = {(f, s) for f in c["PRIMARY_FOLDS"] for s in c["SEEDS"]}
    have = {(r["fold"], r["seed"]) for r in prim}
    assert have == want and len(prim) == len(want), f"primary records missing: {sorted(want - have)}"
    keys = [f"h{h}|{m}" for h in c["HORIZONS"] for m in c["DECIDING_METRICS"] + c["PRINTED_METRICS"]]
    S = dict(protocol_sha256=sha, synthetic=recs[0]["synthetic"], cells={}, panels={}, exploratory={})
    for pert in c["PERTURBATIONS"]:
        S["cells"][pert] = {}
        for f in c["PRIMARY_FOLDS"]:
            rs = sorted((r for r in prim if r["fold"] == f), key=lambda r: r["seed"])
            for panel in rs[0]["panels"]:
                S["cells"][pert][panel] = {k: cell([effect(r, panel, pert, k) for r in rs], c) for k in keys}
                if pert == c["DECIDING_PERTURBATION"]:
                    st = float(np.mean([d["nbr_changed"] for d in rs[0]["panels"][panel]["draws"][pert]]))
                    S["panels"][panel] = panel_verdict(S["cells"][pert][panel], st, c)
    S["outcome"] = outcome({p: v["verdict"] for p, v in S["panels"].items()})
    for r in (r for r in recs if r["role"] == "exploratory"):
        for panel, P in r["panels"].items():
            S["exploratory"][f"{r['family']}|{panel}"] = {
                k: dict(mean=effect(r, panel, c["DECIDING_PERTURBATION"], k),
                        draw_sd=float(np.std([(d["metrics"][k][0] - P["real"][k][0]) / P["real"][k][0]
                                              for d in P["draws"][c["DECIDING_PERTURBATION"]]], ddof=1)))
                for k in keys if not k.endswith("pcc")}
    write(lane / "summary.json", S)
    tag = "SYNTHETIC " if S["synthetic"] else ""
    print(f"\n{tag}decision rule, perturbation {c['DECIDING_PERTURBATION']}, materiality "
          f"{c['MATERIALITY']:.0%}, 95% t over {len(c['SEEDS'])} seeds. + = relabelled map worse")
    for panel, pv in S["panels"].items():
        print(f"\n  {panel}  strength {pv['strength']:.3f}  ->  {pv['verdict']}"
              f"{'  (map unseen by trunk)' if panel in c['UNSEEN_MAP_PANELS'] else '  (map seen via another disease)'}")
        for k, v in S["cells"][c["DECIDING_PERTURBATION"]][panel].items():
            g = S["cells"]["global"][panel][k] if "global" in S["cells"] else None
            pcc = k.endswith("pcc")                       # printed only: no verdict word to quote
            print(f"    {k:10s} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] "
                  f"{'(pcc, abs)' if pcc else v['verdict']:12s}"
                  + (f" | global, never decides: {g['mean']:+.4f}{'' if pcc else ' ' + g['verdict']}" if g else ""))
    print(f"\n  OUTCOME {S['outcome']}  (sentence: protocol section 7)")
    for k, v in S["exploratory"].items():
        print(f"  exploratory {k}: " + "  ".join(f"{kk} {vv['mean']:+.4f}" for kk, vv in v.items()))
    return S


# --------------------------------------------------------------------------- #
# Selfcheck: graph-level and real-graph checks only, nothing permuted is scored
# --------------------------------------------------------------------------- #
def selfcheck():
    import torch
    import bundles
    from explain import forward_fixed_deg
    from models import sparse_from_dense_np, window_slice
    from train.joint import _prepare
    from train.loop import DEVICE
    c = constants()
    assert len(c["PERM_SEEDS"]) == len(set(c["PERM_SEEDS"])), "duplicate perm seeds"
    A_np = np.asarray(bundles.load("influenza_us-regions").A_geo)
    deg, nb = (A_np != 0).sum(1), neighbours(A_np)
    for name, f in PERTURB.items():
        p = f(deg, c["PERM_SEEDS"][0])
        dense = sparse_from_dense_np(A_np[np.ix_(p, p)]).to_dense()
        assert torch.equal(relabel(sparse_from_dense_np(A_np), p).to_dense(), dense), f"{name}: sparse != dense"
        assert sorted((A_np[np.ix_(p, p)] != 0).sum(1)) == sorted(deg), f"{name}: degree multiset moved"
    p = degclass_perm(deg, c["PERM_SEEDS"][0])
    assert np.array_equal((A_np[np.ix_(p, p)] != 0).sum(1), deg), "degclass moved a district's degree"
    assert degclass_perm(np.arange(5), 1).tolist() == list(range(5)), "singleton classes must stay put"
    assert strength(nb, np.arange(len(deg))) == (0.0, 0.0), "identity must change nothing"
    tri = [{1}, {0, 2}, {1}]                                   # path 0-1-2; swap the two ends
    assert strength(tri, np.array([2, 1, 0])) == (0.0, 0.0), "path end-swap is an automorphism"
    star = [{1}, {0}, {3}, {2}]                                # edges 0-1, 2-3; swap 1 and 2
    assert strength(star, np.array([0, 2, 1, 3]))[0] == 1.0, "every neighbour should change"
    # forward_fixed_deg: equals the archived path on the real graph; equals plain forward under degclass
    # (degree unchanged); differs under global (degree feature held at the truth).
    enc, ad, meta = load(LODO / f"encoder_ldo3__influenza__seed{c['SEEDS'][0]}__ckpt.pt", "adapted")
    d = _prepare(["influenza_us-regions"], DEVICE)[0][0]
    t = d.te[0]; Zt, Mt = window_slice(d.Z, t), d.Mt[:, t]
    with torch.no_grad():
        assert torch.equal(forward_fixed_deg(enc, Zt, d.A_solo, d.A_solo, Mt), enc(Zt, d.A_solo, Mt))
        Ad = relabel(d.A_solo, degclass_perm(deg, c["PERM_SEEDS"][0]))
        assert torch.allclose(forward_fixed_deg(enc, Zt, d.A_solo, Ad, Mt), enc(Zt, Ad, Mt), atol=1e-5)
        Ag = relabel(d.A_solo, global_perm(deg, c["PERM_SEEDS"][0]))
        assert not torch.allclose(forward_fixed_deg(enc, Zt, d.A_solo, Ag, Mt), enc(Zt, Ag, Mt), atol=1e-5)
    cc = dict(c, SEEDS=[1, 2, 3, 4, 5])
    off = np.array([-1.5, -0.5, 0, 0.5, 1.5])
    assert cell(0.05 + 0.002 * off, cc)["verdict"] == "USED"
    assert cell(-0.05 + 0.002 * off, cc)["verdict"] == "HURTS"
    assert cell(0.0 + 0.002 * off, cc)["verdict"] == "NOT USED"
    assert cell(0.01 + 0.03 * off, cc)["verdict"] == "INCONCLUSIVE"
    assert cell(0.015 + 0.001 * off, cc)["verdict"] == "NOT USED", "significant but immaterial is NOT USED"
    try:
        write(ROOT / "results" / "x.json", {}); raise AssertionError("write guard let a results/ path through")
    except PermissionError:
        pass
    print("selfcheck ok: sparse relabel == dense; degclass keeps every district's degree; identity and "
          "automorphism change nothing; fixed-degree forward == archived path on the real graph; "
          "cell rule gives USED/HURTS/NOT USED/INCONCLUSIVE as built; write guard holds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha256", help="protocol sha256 you expect; the run refuses on mismatch")
    ap.add_argument("--dry", action="store_true", help="drift gate + timing, NO permuted scoring")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dir", default=str(LANE), help="records dir for --report (inside the lane)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    c = constants()
    if a.report:
        return report(c, sha256(PROTOCOL), Path(a.dir))
    if not a.sha256:
        ap.error("--sha256 is required for --dry and for the scored run")
    sha = check_stamp(a.sha256, committed=not a.dry)
    t0, recs = time.time(), []
    for row in plan(c):
        recs.append(run_one(row, c, sha, a.dry))
    if a.dry:
        nP, K = len(c["PERTURBATIONS"]), len(c["PERM_SEEDS"])
        proj = sum(P["seconds"]["real_forward"] + P["seconds"]["score"]
                   + nP * K * (P["seconds"]["perm_forward"] + P["seconds"]["score"])
                   for r in recs for P in r["panels"].values())
        write(LANE / "dry_run.json", dict(protocol_sha256=sha, projected_seconds=proj, records=[
            {k: v for k, v in r.items() if k != "code_sha256"} for r in recs]))
        print(f"\nDRY RUN ok in {time.time() - t0:.0f}s. Every drift gate passed. Nothing permuted was "
              f"scored. Projected scored run: {proj / 60:.0f} min ({nP} perturbations x {K} draws)")
    else:
        print(f"\nscored run done in {(time.time() - t0) / 60:.1f} min; now run --report")


if __name__ == "__main__":
    main()

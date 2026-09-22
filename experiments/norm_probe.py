"""EXPLORATORY: what does the normalisation mismatch cost, in both directions?

Two runs, both off-protocol, both writing ONLY into experiments/:

  EXP 1  A development disease scored the Ebola way. The trunk was trained on per-node inputs; here
         its held-out panel arrives POOLED (one log1p mean/sd fit on the train fold, Ebola-style).
         Same trunk, same adapter, same test cells; the input scaling is the only change. This is
         the "trained per-node, used pooled" mismatch that Gap_Ledger.md:321-324 records as never
         run.

  EXP 2  Ebola scored the development way. Per-node scaler fit on each district's own support cells
         (country fallback for districts whose support is degenerate, to_schema.py:135-140), and the
         43 (L12) / 25 (L20) districts with NO support cell get their normalised incidence channel
         set to exactly 0.0000, so they arrive centred the way every dev district does. Inversion to
         counts uses each district's assigned stats. Fresh few-shot adapter refit under the new
         scale with the pre-registered stopping rule; zero-shot arm re-run on the new features.

THE FROZEN HASHES ARE THE ONE HARD CONSTRAINT. This script verifies both arms' content digests
against configs/ebola_arms.json BEFORE and AFTER the run (train.ebola.load_manifest(verify=True)
refuses to continue on a moved arm), rescales bundles in memory only, and never writes into
results/ or data/. Before trusting any off-protocol number, both scoring paths are validated by
reproducing the ARCHIVED matched records from the loaded checkpoints first.

None of this is the pre-registered Ebola result. Every record carries protocol="EXPLORATORY".

  conda run -n ebola-train python -m experiments.norm_probe --selfcheck
  conda run -n ebola-train python -m experiments.norm_probe --seeds 42
  conda run -n ebola-train python -m experiments.norm_probe --seeds 42 52 62 72 82

Multi-seed notes: EXP 2 pairs by seed (each seed's trunk and archived adapters are its own control)
and the summary reports SIGN COUNTS over seeds plus mean deltas, not significance -- five paired
seeds support "consistent across seeds", nothing stronger. EXP 1 runs only at seed 42, because the
encoder_ldo3full checkpoints it needs exist only there; other seeds skip it with a printed reason.
One JSON is written per seed as it completes, so a crash costs one seed, not the run.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import json
import pathlib

import numpy as np
import torch

import bundles
from bundles import HORIZONS
from models import Adapter, SharedEncoder, sparse_from_dense_np
from to_schema import apply_scaler, fit_scalers_masked, invert_scaler
from train.ebola import _fit_adapter, _precompute_features, choose_epochs, load_manifest, support_origins
from train.lodo import _score
from train.loop import DEVICE

OUT = pathlib.Path("experiments")
TAG = "EXPLORATORY"                                     # stamped into every record
EXP1 = {"covid_us-states": "covid", "influenza_japan": "influenza",
        "influenza_us-regions": "influenza", "influenza_us-states": "influenza"}
EBOLA_ARMS = ("ebola_L12", "ebola_L20")


def mae_by_h(recs):
    return {r["horizon"]: r["country_macro"] for r in recs if r["metric"] == "mae"}


def archived(path):
    return mae_by_h(json.loads(pathlib.Path(path).read_text()))


def _dump(obj, fname):
    p = OUT / fname
    assert OUT.resolve() in p.resolve().parents or p.resolve().parent == OUT.resolve(), \
        f"output escaping experiments/: {p}"
    p.write_text(json.dumps(obj, indent=2))
    return p


def rescale(b, sc, zero_nodes=()):
    """Apply a new scaler to a LOADED bundle, in memory. The .npz on disk is never touched.

    X channel 0 and y are rebuilt the way Bundle.refit does it (masked by M), b.scaler is replaced so
    _score's invert_scaler stays consistent with the forward pass, and `zero_nodes` then get their
    incidence channel forced to exactly 0.0 -- the value the dev panels' own districts centre on.
    Channels 1-3 (calendar, obs_mask) are untouched."""
    x0 = apply_scaler(b.raw, sc) * (b.M == 1)
    b.X[..., 0] = x0
    b.y = x0.astype(np.float32)
    b.scaler = sc
    if len(zero_nodes):
        b.X[..., 0][list(zero_nodes), :] = 0.0
        b.y[list(zero_nodes), :] = 0.0


def tensors(b, device):
    Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=device)
    Mt = torch.tensor(b.M, dtype=torch.float32, device=device)
    A = sparse_from_dense_np(b.A_geo).to(device)
    return Z, Mt, A


def score_arm(enc, ad, b, seed, model_name, meta, phase, device):
    Z, Mt, A = tensors(b, device)
    te = b.origins(phase=phase)
    recs, _, _, _ = _score(enc, ad, b, Z, Mt, A, te, b.name, seed, model_name,
                           dict(meta, protocol=TAG), phase=phase, device=device)
    return recs


# --------------------------------------------------------------------------- #
# EXP 1: dev disease, trunk per-node, inputs pooled.
# --------------------------------------------------------------------------- #
def exp1(seed, device):
    rows = []
    missing = [f for f in set(EXP1.values())
               if not pathlib.Path(f"results/lodo/encoder_ldo3full__{f}__seed{seed}__ckpt.pt").exists()]
    if missing:
        print(f"\nEXP 1 skipped at seed {seed}: no encoder_ldo3full checkpoint for {sorted(missing)} "
              f"(they exist only at seed 42)")
        return rows
    print("\nEXP 1: development panels scored with POOLED (Ebola-style) inputs on per-node trunks")
    for panel, fold in EXP1.items():
        ck = torch.load(f"results/lodo/encoder_ldo3full__{fold}__seed{seed}__ckpt.pt",
                        map_location=device, weights_only=False)
        enc = SharedEncoder(gate_mode="learned").to(device); enc.load_state_dict(ck["encoder"]); enc.eval()
        ad = Adapter().to(device); ad.load_state_dict(ck["adapter"]); ad.eval()

        # validation first: the untouched bundle must reproduce the archived record exactly
        b = bundles.load(panel)
        got = mae_by_h(score_arm(enc, ad, b, seed, "exp1_matched", dict(panel=panel), "test", device))
        ref = archived(f"results/lodo/encoder_ldo3full__{panel}__seed{seed}.json")
        for h in HORIZONS:
            assert abs(got[h] - ref[h]) < max(1e-3, 1e-5 * abs(ref[h])), \
                f"{panel} h{h}: matched rescore {got[h]} != archived {ref[h]} -- scoring path broken"

        # the mismatch: pooled log1p stats over the TRAIN fold (the analogue of Ebola's support fit)
        b2 = bundles.load(panel)
        sc = fit_scalers_masked(b2.raw, b2.masks()["train"], per_disease=True)
        rescale(b2, sc)
        mis = mae_by_h(score_arm(enc, ad, b2, seed, "exp1_pooled", dict(panel=panel), "test", device))
        for h in HORIZONS:
            d = (mis[h] - ref[h]) / ref[h] * 100
            rows.append(dict(exp="exp1", seed=seed, panel=panel, h=h,
                             matched=ref[h], pooled=mis[h], cost_pct=d))
        print(f"  {panel:<22} matched {ref[3]:9.1f} MAE(h3) -> pooled {mis[3]:9.1f}  "
              f"cost h3/h5/h10/h15: " + " / ".join(f"{(mis[h]-ref[h])/ref[h]*100:+.1f}%" for h in HORIZONS))
    return rows


# --------------------------------------------------------------------------- #
# EXP 2: Ebola, scaler per-node from support, zero-support districts pinned at 0.0.
# --------------------------------------------------------------------------- #
def devnorm_scaler(b):
    """(scaler, zero_nodes): per-node from support with country fallback; no-support districts are
    the ones whose incidence channel gets pinned to 0.0000."""
    s = b.masks()["support"].astype(bool)
    grp = [b.meta["node_country"][n] for n in b.meta["node_ids"]]
    sc = fit_scalers_masked(b.raw, s, per_disease=False, groups=grp)
    zero_nodes = np.where(~s.any(1))[0]
    return sc, zero_nodes


def exp2(seed, device):
    rows = []
    print("\nEXP 2: Ebola scored under a development-style scaler (zero-support districts pinned at 0.0)")
    tk = torch.load(f"results/ebola/encoder_ebola__alldev__seed{seed}__ckpt.pt",
                    map_location=device, weights_only=False)
    enc = SharedEncoder(gate_mode="learned").to(device); enc.load_state_dict(tk["encoder"]); enc.eval()
    zero = Adapter().to(device); zero.load_state_dict(tk["adapter"]); zero.eval()

    for arm in EBOLA_ARMS:
        # validation first: untouched bundle + archived adapters must reproduce the scored record
        b = bundles.load(arm)
        fk = torch.load(f"results/ebola/encoder_ebola__{arm}__seed{seed}__ckpt.pt",
                        map_location=device, weights_only=False)
        few = Adapter().to(device); few.load_state_dict(fk["adapter"]); few.eval()
        for model, ad, ref_file in ((f"exp2_matched_few", few, f"results/ebola/encoder_ebola__{arm}__seed{seed}.json"),
                                    (f"exp2_matched_zero", zero, f"results/ebola/encoder_ebola_zeroshot__{arm}__seed{seed}.json")):
            got = mae_by_h(score_arm(enc, ad, b, seed, model, dict(arm=arm), "query", device))
            ref = archived(ref_file)
            for h in HORIZONS:
                assert abs(got[h] - ref[h]) < max(1e-3, 1e-5 * abs(ref[h])), \
                    f"{arm} {model} h{h}: {got[h]} != archived {ref[h]} -- scoring path broken"

        # the exploratory arm: rescale in memory, refit few-shot under the pre-registered rule
        b2 = bundles.load(arm)
        sc, zn = devnorm_scaler(b2)
        rescale(b2, sc, zero_nodes=zn)
        assert float(np.abs(b2.X[..., 0][zn]).max(initial=0.0)) == 0.0
        print(f"  {arm}: {len(zn)} of {b2.M.shape[0]} districts have no support cell -> pinned 0.0")

        Z, Mt, A = tensors(b2, device)
        so = support_origins(b2)
        smask = torch.tensor(b2.masks()["support"], dtype=torch.float32, device=device)
        ymod = torch.tensor(b2.y, dtype=torch.float32, device=device)
        feats = _precompute_features(enc, Z, A, Mt, so, device)
        ep, _, _ = choose_epochs(feats, ymod, Mt, smask, so, seed, device, verbose=False)
        few2, _ = _fit_adapter(feats, ymod, Mt, smask, so, seed, ep, device)

        ref_few = archived(f"results/ebola/encoder_ebola__{arm}__seed{seed}.json")
        ref_zero = archived(f"results/ebola/encoder_ebola_zeroshot__{arm}__seed{seed}.json")
        dn_few = mae_by_h(score_arm(enc, few2, b2, seed, "exp2_devnorm_few",
                                    dict(arm=arm, adapter_epochs=ep, zero_pinned=len(zn)), "query", device))
        dn_zero = mae_by_h(score_arm(enc, zero, b2, seed, "exp2_devnorm_zero",
                                     dict(arm=arm, zero_pinned=len(zn)), "query", device))
        print(f"    {'h':>3} {'pooled zero':>12} {'pooled few':>12} {'devnorm zero':>13} {'devnorm few':>12}")
        for h in HORIZONS:
            rows.append(dict(exp="exp2", seed=seed, arm=arm, h=h,
                             pooled_zero=ref_zero[h], pooled_few=ref_few[h],
                             devnorm_zero=dn_zero[h], devnorm_few=dn_few[h], adapter_epochs=ep))
            print(f"    {h:>3} {ref_zero[h]:>12.2f} {ref_few[h]:>12.2f} {dn_zero[h]:>13.2f} {dn_few[h]:>12.2f}")
    return rows


def summarise_exp2(rows):
    """Paired-by-seed summary. Sign counts and mean deltas only -- five seeds support 'consistent
    across seeds' and nothing stronger, so no significance language is printed or stored."""
    seeds = sorted({r["seed"] for r in rows})
    n = len(seeds)
    out = []
    print(f"\nEXP 2 SUMMARY over seeds {seeds}  (country-macro MAE, means; counts are seeds of {n})")
    print(f"{'arm':<11}{'h':>3}{'pooled_zero':>12}{'pooled_few':>11}{'devnorm_zero':>13}"
          f"{'devnorm_few':>12} | {'df<pz':>6}{'df<dz':>6}{'dz<pz':>6} | {'d(df-pz)':>9}")
    print("-" * 100)
    for arm in EBOLA_ARMS:
        for h in HORIZONS:
            c = [r for r in rows if r["arm"] == arm and r["h"] == h]
            if len(c) != n:
                continue
            m = {k: float(np.mean([r[k] for r in c]))
                 for k in ("pooled_zero", "pooled_few", "devnorm_zero", "devnorm_few")}
            wins = {k: sum(1 for r in c if r[a] < r[b])
                    for k, (a, b) in (("df<pz", ("devnorm_few", "pooled_zero")),
                                      ("df<dz", ("devnorm_few", "devnorm_zero")),
                                      ("dz<pz", ("devnorm_zero", "pooled_zero")))}
            deltas = [(r["devnorm_few"] - r["pooled_zero"]) / r["pooled_zero"] * 100 for r in c]
            note = "  <- 0 adaptation pairs: random head, not a fit" \
                if arm == "ebola_L12" and h == 15 else ""
            out.append(dict(arm=arm, h=h, n_seeds=n, **m, **wins,
                            d_pct_mean=float(np.mean(deltas)),
                            d_pct_min=float(np.min(deltas)), d_pct_max=float(np.max(deltas))))
            print(f"{arm:<11}{h:>3}{m['pooled_zero']:>12.2f}{m['pooled_few']:>11.2f}"
                  f"{m['devnorm_zero']:>13.2f}{m['devnorm_few']:>12.2f} | "
                  f"{wins['df<pz']:>4}/{n}{wins['df<dz']:>4}/{n}{wins['dz<pz']:>4}/{n} | "
                  f"{np.mean(deltas):>+7.1f}% [{min(deltas):+.1f},{max(deltas):+.1f}]{note}")
    print("""
read:
  df<pz  seeds where devnorm FEW-SHOT beats POOLED ZERO-SHOT, the shipped best. This is the bar for
         "the dev-style scaler actually improves the horizon"
  df<dz  seeds where few-shot beats zero-shot UNDER devnorm, i.e. adaptation helps once rescaled
  dz<pz  seeds where devnorm zero-shot alone beats pooled zero-shot (scaler effect without adaptation)
  d(df-pz) devnorm few vs pooled zero, mean [min,max] percent over seeds. Negative = improvement.
  All EXPLORATORY, paired by seed, sign counts only -- 5 seeds cannot carry a significance claim.""")
    return out


def selfcheck():
    load_manifest(verify=True)
    print("  ok frozen arm hashes verified against configs/ebola_arms.json")
    b = bundles.load("ebola_L12")
    sc, zn = devnorm_scaler(b)
    assert len(zn) == 61 - 18, f"expected 43 zero-support districts on L12, got {len(zn)}"
    rt = invert_scaler(apply_scaler(b.raw, sc), sc)
    obs = b.M.astype(bool)
    assert np.allclose(rt[obs], b.raw[obs], atol=1e-4), "dev-style scaler does not round-trip"
    rescale(b, sc, zero_nodes=zn)
    assert float(np.abs(b.X[..., 0][zn]).max()) == 0.0, "pinned districts are not exactly 0.0"
    d = [b.X[..., 0][i][b.masks()["support"].astype(bool)[i]].mean()
         for i in range(61) if b.masks()["support"][i].any()]
    print(f"  ok L12 dev-style scaler: {len(zn)} districts pinned 0.0, round-trip exact, "
          f"support districts now centred at mean |avg| = {float(np.mean(np.abs(d))):.4f} "
          f"(pooled was 0.70)")
    b2 = bundles.load("covid_us-states")
    sc2 = fit_scalers_masked(b2.raw, b2.masks()["train"], per_disease=True)
    assert np.allclose(sc2["mean"], sc2["mean"][0]) and np.allclose(sc2["std"], sc2["std"][0]), \
        "pooled scaler must be one shared (mean, sd)"
    load_manifest(verify=True)
    print("  ok pooled dev scaler is a single shared (mean, sd); hashes verified again")
    print("selfcheck passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--only", choices=["exp1", "exp2"], default=None)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    selfcheck()
    all_exp2 = []
    for seed in a.seeds:
        print(f"\n================ seed {seed} ================")
        rows = []
        if a.only in (None, "exp1"):
            rows += exp1(seed, DEVICE)
        if a.only in (None, "exp2"):
            e2 = exp2(seed, DEVICE)
            rows += e2
            all_exp2 += e2
        load_manifest(verify=True)
        p = _dump(dict(protocol=TAG, seed=seed, rows=rows), f"norm_probe__seed{seed}.json")
        print(f"  seed {seed} done, hashes intact, wrote {p}")
    if len(a.seeds) > 1 and all_exp2:
        summary = summarise_exp2(all_exp2)
        p = _dump(dict(protocol=TAG, seeds=a.seeds, cells=summary), "norm_probe__exp2_summary.json")
        print(f"wrote {p}")
    load_manifest(verify=True)
    print("frozen arm hashes re-verified after the full run: intact")


if __name__ == "__main__":
    main()

"""train/lodo.py -- leave-one-disease-out transfer probe (the decisive G2/G3 test, Day 14 follow-up).

Question it answers: does the shared trunk, trained on OTHER diseases, carry structure that transfers
to a disease it NEVER saw? This is the dev-set proxy for the Week-5 Ebola mechanism (frozen trunk +
a small adapter fit on the new disease), and -- unlike the sqrt-weighting probe -- it is not
confounded by the across-dataset loss weighting.

Protocol, per held-out disease X:
  1. Train the shared trunk + per-disease adapters JOINTLY on the OTHER 3 dev diseases (block-diagonal
     supergraph, uniform weighting -- same machinery as train.joint). X is never in trunk training.
  2. FREEZE the trunk. Fit ONE fresh adapter (FiLM + quantile head, ~388 params) on X's OWN train
     fold, trunk frozen -- identical adapter-fit protocol to the single-disease trainer (epoch-based,
     val early-stop). This is the OPTIMISTIC transfer number: X gets its full train fold to fit the
     adapter, so if transfer fails here it will certainly fail few-shot on Ebola's 27 support cells.
  3. Score X's test fold. Also score a pure ZERO-SHOT reference: the mean of the 3 trained in-adapters
     applied to X with NO fitting (a rough lower bound -- adapters are all in per-node z-score space).

Read against the baselines already in results/:
  single-disease encoder (results/encoder__X__seed*.json)  = CEILING (trunk trained on X itself)
  naive floors           (results/naive__X.json)            = FLOOR
    LODO-adapter ~ ceiling  -> trunk transfers; a foreign trunk + small adapter recovers X  (Option A)
    LODO-adapter ~ floor    -> trunk does not transfer; you must train on the disease itself (Option B)

Ebola is never touched (only the 4 dev diseases; C8 guard inherited from train.joint's DEV list).

Run from the repo root:
  PYTHONNOUSERSITE=1 conda run -n ebola-train python -m train.lodo --held-out influenza_us-states
  PYTHONNOUSERSITE=1 conda run -n ebola-train python -m train.lodo --all           # all 4 folds
  PYTHONNOUSERSITE=1 conda run -n ebola-train python -m train.lodo --smoke          # fast, no dengue
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

import bundles
from bundles import DEV_BUNDLE_NAMES, HORIZONS
from models import (MEDIAN_IDX, Adapter, SharedEncoder, pinball_loss, sparse_from_dense_np,
                    targets_and_mask, window_slice)
from to_schema import invert_scaler
from train.joint import (_node_weight, _origin_stream, _prepare, _val_pinball, dataset_weights)
from train.loop import (DEVICE, RESULTS, _round_trip_ok, gate_spatial_readout, score_predictions,
                        write_gate, write_per_node, write_per_origin, write_records)
from results_paths import rpath

HEADLINE = {"dengue": "country_macro"}          # else node_mean
LOWER_BETTER = {"rmse", "mae", "smape", "peak_timing", "peak_intensity"}


def _field(ds):
    return HEADLINE.get(ds, "node_mean")


# --------------------------------------------------------------------------- #
# Step 1 -- train the shared trunk on the in-diseases (joint loop, returns the model).
# --------------------------------------------------------------------------- #
def _fit_trunk(seed, in_names, device, steps=91000, val_every=1000, patience=12, lr=1e-3, wd=1e-4,
               sampler="uniform", gate_mode="learned", verbose=True):
    """Mirror train.joint.train_joint's training loop, but RETURN (enc, in_adapters) instead of
    testing. node weights uniform (dengue balance off) to match the primary uniform-uniform run."""
    assert "ebola" not in in_names, "ebola must never enter trunk training (C8)"
    torch.manual_seed(seed); np.random.seed(seed)
    ds, A_block = _prepare(in_names, device)
    enc = SharedEncoder(gate_mode=gate_mode).to(device)
    adapters = nn.ModuleList([Adapter().to(device) for _ in in_names])
    params = list(enc.parameters()) + list(adapters.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    w_base = dataset_weights(ds, sampler)
    node_w = [_node_weight(d.name, d.b, "uniform", device) for d in ds]
    streams = [_origin_stream(d.tr, seed + i) for i, d in enumerate(ds)]

    best_val, best_state, bad = float("inf"), None, 0
    enc.train(); [ad.train() for ad in adapters]
    for step in range(1, steps + 1):
        ts = [next(s) for s in streams]
        Zc = torch.cat([window_slice(d.Z, ts[i]) for i, d in enumerate(ds)], dim=0)
        Mcol = torch.cat([ds[i].Mt[:, ts[i]] for i in range(len(ds))], dim=0)
        h = enc(Zc, A_block, Mcol)
        losses, wts = [], []
        for i, (ad, d) in enumerate(zip(adapters, ds)):
            tgt, msk = targets_and_mask(d.ymod, d.Mt, d.mtr, ts[i], device)
            if msk.sum() == 0:
                continue
            losses.append(pinball_loss(ad(h[d.start:d.end]), tgt, msk, w=node_w[i]))
            wts.append(w_base[i])
        if not losses:
            continue
        wt = torch.tensor(wts, device=device); wt = wt / wt.sum()
        loss = sum(a * L for a, L in zip(wt, losses))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if step % val_every == 0:
            val = _val_pinball(enc, adapters, ds, node_w, w_base)
            if val < best_val - 1e-5:
                best_val, bad = val, 0
                best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                              {k: v.detach().clone() for k, v in adapters.state_dict().items()})
            else:
                bad += 1
            if verbose:
                print(f"    trunk(-{seed}) step{step:6d} val={val:.4f} best={best_val:.4f}")
            enc.train(); [ad.train() for ad in adapters]
            if bad >= patience:
                break
    if best_state:
        enc.load_state_dict(best_state[0]); adapters.load_state_dict(best_state[1])
    return enc, adapters


def _mean_adapter(in_adapters, device):
    """Zero-shot reference: the element-wise mean of the trained in-adapters (all live in per-node
    z-score space, so the mean is a defensible generic adapter). No held-out data used."""
    ad = Adapter().to(device)
    ref = ad.state_dict()
    avg = {k: torch.stack([a.state_dict()[k].float() for a in in_adapters]).mean(0).to(ref[k].dtype)
           for k in ref}
    ad.load_state_dict(avg)
    return ad


# --------------------------------------------------------------------------- #
# Step 2 -- freeze trunk, fit a fresh adapter on the held-out disease (== single-disease protocol).
# --------------------------------------------------------------------------- #
def _fit_adapter_and_score(enc, name, seed, device, epochs=80, lr=1e-3, wd=1e-4, batch_origins=8,
                           patience=15, verbose=True):
    """Trunk FROZEN; train only a fresh Adapter on `name`'s train fold, val early-stop, then score
    its test fold. Same epoch/patience/batch protocol as train.loop.train_one, so the only difference
    vs the single-disease ceiling is the trunk's provenance (foreign vs same-disease)."""
    b = bundles.load(name)
    assert _round_trip_ok(b), f"{name}: scaler round-trip failed"
    Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=device)
    ymod = torch.tensor(b.y, dtype=torch.float32, device=device)
    Mt = torch.tensor(b.M, dtype=torch.float32, device=device)
    A = sparse_from_dense_np(b.A_geo).to(device)
    masks = {p: torch.tensor(m, dtype=torch.float32, device=device) for p, m in b.masks().items()}
    tr, va, te = b.origins(phase="train"), b.origins(phase="val"), b.origins(phase="test")

    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()                                            # trunk is a fixed feature extractor
    ad = Adapter().to(device)
    opt = torch.optim.AdamW(ad.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def run_phase(origins, train_mode):
        ad.train(train_mode)
        total, n, pending = 0.0, 0, 0
        order = np.random.permutation(origins) if train_mode else origins
        if train_mode:
            opt.zero_grad()
        for t in order:
            tgt, msk = targets_and_mask(ymod, Mt, masks["train" if train_mode else "val"], t, device)
            if msk.sum() == 0:
                continue
            with torch.no_grad():                         # trunk features are constants w.r.t. the adapter
                feat = enc(window_slice(Z, t), A, Mt[:, t])
            if train_mode:
                pred = ad(feat)
                loss = pinball_loss(pred, tgt, msk)
                loss.backward(); pending += 1
                if pending == batch_origins:
                    torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
                    opt.step(); opt.zero_grad(); pending = 0
            else:
                with torch.no_grad():
                    loss = pinball_loss(ad(feat), tgt, msk)
            total += float(loss) * int(msk.sum()); n += int(msk.sum())
        if train_mode and pending:
            torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        return total / max(n, 1)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        run_phase(tr, True)
        val = run_phase(va, False)
        sched.step()
        if val < best_val - 1e-5:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in ad.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"    adapter[{name}] ep{ep:02d} val={val:.4f} best={best_val:.4f}")
        if bad >= patience:
            break
    if best_state:
        ad.load_state_dict(best_state)
    return ad, b, Z, ymod, Mt, A, masks, va, te


def _score(enc, ad, b, Z, Mt, A, te, name, seed, model_name, run_meta, va=None, gate_read=False,
           device=DEVICE):
    """Forecast test fold (median = point), invert to counts, score through score.py."""
    enc.eval(); ad.eval()
    T = b.X.shape[1]
    pred_by_h = {h: np.zeros((b.X.shape[0], T), dtype=np.float64) for h in HORIZONS}
    with torch.no_grad():
        for t in te:
            med = ad(enc(window_slice(Z, t), A, Mt[:, t]))[:, :, MEDIAN_IDX].cpu().numpy()
            for j, h in enumerate(HORIZONS):
                pred_by_h[h][:, t + h] = invert_scaler(med[:, j:j + 1], b.scaler)[:, 0]
    recs, pernode, perorigin = score_predictions(model_name, name, seed, pred_by_h, b, te,
                                                 run_meta=run_meta)
    gate = gate_spatial_readout(enc, ad, Z, A, Mt, va, device) if (gate_read and va is not None) else None
    return recs, pernode, perorigin, gate


# --------------------------------------------------------------------------- #
# A fold: trunk on the other 3 -> adapter-fit + zero-shot on the held-out.
# --------------------------------------------------------------------------- #
def run_fold(held_out, seed, device=DEVICE, trunk_steps=91000, verbose=True):
    in_names = [n for n in DEV_BUNDLE_NAMES if n != held_out]
    if verbose:
        print(f"\n=== LODO fold: held-out={held_out}  trunk-train={in_names}  seed={seed} ===")
    enc, in_adapters = _fit_trunk(seed, in_names, device, steps=trunk_steps, verbose=verbose)

    meta = dict(training_regime="lodo", sampler="uniform-uniform", gate_mode="learned", topo_aug="none")
    ad, b, Z, ymod, Mt, A, masks, va, te = _fit_adapter_and_score(enc, held_out, seed, device,
                                                                  verbose=verbose)
    recs, pn, po, gate = _score(enc, ad, b, Z, Mt, A, te, held_out, seed, f"encoder_lodo:{held_out}",
                                meta, va=va, gate_read=True, device=device)
    write_records(recs, f"encoder_lodo__{held_out}__seed{seed}.json")
    write_per_node(pn, f"encoder_lodo__{held_out}__seed{seed}__pernode.npz")
    write_per_origin(po, f"encoder_lodo__{held_out}__seed{seed}__perorigin.npz")
    write_gate(gate, f"encoder_lodo__{held_out}__seed{seed}__gate.npz")

    zmeta = dict(meta, training_regime="lodo_zeroshot")
    borrowed = _mean_adapter(in_adapters, device)
    zrecs, _, _, _ = _score(enc, borrowed, b, Z, Mt, A, te, held_out, seed,
                            f"encoder_lodo_zeroshot:{held_out}", zmeta, device=device)
    write_records(zrecs, f"encoder_lodo_zeroshot__{held_out}__seed{seed}.json")
    if verbose:
        _report(held_out, seed)
    return recs, zrecs


# --------------------------------------------------------------------------- #
# Read the fold against the ceiling (single-disease) and floor (naive), headline field.
# --------------------------------------------------------------------------- #
def _load_recs(path):
    if not os.path.exists(path):
        return None
    fld = None
    out = {}
    for r in json.load(open(path)):
        fld = _field(r["dataset"])
        out[(r["horizon"], r["metric"], r["model"])] = r[fld]
    return out


ALL_SEEDS = (42, 52, 62, 72, 82)
CV_SCREEN_MULT = 1.96


def _single_ref(ho, m, h, seeds=ALL_SEEDS):
    """(mean over seeds, CV%) of the single-disease run at this exact (dataset, horizon, metric).

    The mean is the reference the write-ups quote; the CV is the noise floor. Both are per horizon
    on purpose -- the CV swings from ~14% (dengue h3) to ~0.4% (dengue h15), so one averaged floor
    would mis-tag both ends."""
    vals = []
    for s in seeds:
        d = _load_recs(rpath(f"encoder__{ho}__seed{s}.json"))
        if d and (h, m, "encoder") in d:
            vals.append(d[(h, m, "encoder")])
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2 or mean == 0:
        return mean, None
    sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return mean, 100 * sd / abs(mean)


def _report(ho, seed):
    """Both references, labelled, plus the seed-noise screen.

    History (2026-07-29): this used to compare LODO against seed-`seed`'s OWN single run while the
    write-ups quoted 5-seed means, so the two were never readable against each other. Seed 42 is an
    unusually BAD single-disease seed, which flattered LODO -- a dengue h3 result that is a 15%
    WORSENING against the 5-seed mean printed as +3% against seed 42. The percentage formula was
    always right; the reference was not. Both are now shown and named on the table.

    The verdict number is NOT here: `python analysis.py --transfer` gives the paired
    origin-bootstrap CI. This screen only answers "would a different init have done as well"."""
    fld = _field(ho)
    single_s = _load_recs(rpath(f"encoder__{ho}__seed{seed}.json"))
    lodo = _load_recs(rpath(f"encoder_lodo__{ho}__seed{seed}.json"))
    zs = _load_recs(rpath(f"encoder_lodo_zeroshot__{ho}__seed{seed}.json"))
    naive = _load_recs(rpath(f"naive__{ho}.json"))
    print(f"\n{'='*112}\nLODO transfer read :: held-out={ho}  seed={seed}  field={fld}")
    print(f"  ref A = single-disease MEAN over seeds {ALL_SEEDS}   ref B = single-disease seed {seed}"
          f"   floor = best naive   (+% = lodo better)")
    print(f"  'noise' = |d% vs A| < {CV_SCREEN_MULT}x the single-disease seed CV at that horizon."
          f"  Verdict CI: python analysis.py --transfer\n{'='*112}")
    for m in ["rmse", "mae", "pcc"]:
        print(f"\n  {m.upper()}  ({'higher' if m=='pcc' else 'lower'}=better)")
        print(f"    {'h':>3} | {'singleA':>10} | {'singleB':>10} | {'lodo-adapt':>10} | "
              f"{'zero-shot':>10} | {'best-naive':>10} | {'d% vs A':>8} | {'d% vs B':>8} | "
              f"{'CV%':>5} | screen")
        for h in HORIZONS:
            sA, cv = _single_ref(ho, m, h)
            sB = single_s.get((h, m, "encoder")) if single_s else None
            lv = lodo.get((h, m, f"encoder_lodo:{ho}")) if lodo else None
            zv = zs.get((h, m, f"encoder_lodo_zeroshot:{ho}")) if zs else None
            nv = None
            if naive:
                cands = [c for c in (naive.get((h, m, k)) for k in
                                     ("persistence", "seasonal", "train_mean")) if c is not None]
                if cands:
                    nv = (min if m in LOWER_BETTER else max)(cands)

            def pct(ref):
                if ref in (None, 0) or lv is None:
                    return None
                return ((ref - lv) / ref * 100) if m in LOWER_BETTER else ((lv - ref) / abs(ref) * 100)

            dA, dB = pct(sA), pct(sB)
            screen = "--"
            if dA is not None and cv is not None:
                screen = "clears" if abs(dA) >= CV_SCREEN_MULT * cv else "NOISE"
            f = lambda x: f"{x:10.3f}" if x is not None else f"{'--':>10}"
            g = lambda x: f"{x:+7.1f}%" if x is not None else f"{'--':>8}"
            print(f"    {h:>3} | {f(sA)} | {f(sB)} | {f(lv)} | {f(zv)} | {f(nv)} | "
                  f"{g(dA):>8} | {g(dB):>8} | {('--' if cv is None else f'{cv:5.1f}'):>5} | {screen}")
    print()


# --------------------------------------------------------------------------- #
def _smoke():
    """Fast end-to-end, no dengue: trunk on 2 small sets, held-out us-regions, tiny budgets."""
    dev = DEVICE
    enc, ins = _fit_trunk(42, ["influenza_japan", "influenza_us-states"], dev, steps=120,
                          val_every=60, patience=99, verbose=False)
    ad, b, Z, ymod, Mt, A, masks, va, te = _fit_adapter_and_score(
        enc, "influenza_us-regions", 42, dev, epochs=3, patience=99, verbose=False)
    recs, pn, po, g = _score(enc, ad, b, Z, Mt, A, te, "influenza_us-regions", 42,
                             "encoder_lodo:smoke", dict(training_regime="lodo", sampler="u",
                             gate_mode="learned", topo_aug="none"))
    assert len(recs) == len(HORIZONS) * 6, f"expected {len(HORIZONS)*6} records, got {len(recs)}"
    z = _mean_adapter(ins, dev)
    zr, _, _, _ = _score(enc, z, b, Z, Mt, A, te, "influenza_us-regions", 42, "encoder_lodo_zeroshot:smoke",
                         dict(training_regime="lodo_zeroshot", sampler="u", gate_mode="learned", topo_aug="none"))
    assert len(zr) == len(HORIZONS) * 6
    assert all("ebola" != n for n in DEV_BUNDLE_NAMES)   # C8
    print(f"smoke ok: adapter-fit + zero-shot both produced {len(recs)} records (held-out us-regions)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", choices=list(DEV_BUNDLE_NAMES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trunk-steps", type=int, default=91000)
    a = ap.parse_args()

    t0 = time.time()
    if a.smoke:
        _smoke()
    elif a.all:
        for ho in DEV_BUNDLE_NAMES:
            run_fold(ho, a.seed, trunk_steps=a.trunk_steps)
        print(f"all LODO folds done in {(time.time()-t0)/60:.1f} min")
    elif a.held_out:
        run_fold(a.held_out, a.seed, trunk_steps=a.trunk_steps)
        print(f"LODO fold {a.held_out} done in {(time.time()-t0)/60:.1f} min")
    else:
        ap.error("give --held-out NAME, or --all, or --smoke")


if __name__ == "__main__":
    main()

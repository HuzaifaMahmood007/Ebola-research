"""train/ebola.py -- the Ebola case study, scored ONCE against the locked config.

The protocol is `progress/decisions/Ebola_Prereg.md` (frozen 2026-08-07, amendments A1-A5) and the
split is `configs/ebola_arms.json`. Nothing here chooses anything: every free parameter the run
could have tuned was fixed in writing before this file was executed, and this module asserts the
frozen hashes before it will score.

  1. TRUNK. One shared encoder trained jointly on all five development bundles -- dengue, influenza
     japan / us-regions / us-states, covid us-states -- with adapter_groups [0,1,1,1,2], so each
     DISEASE gets one adaptation surface and the loss is rebalanced per disease rather than per
     bundle. Nothing is held out: Ebola is the held-out disease. trunk_patience=30 (A5). One trunk
     per seed, shared by both arms, so the arms differ only in their support set.
  2. ZERO-SHOT arm. The element-wise mean of the three in-disease adapters, applied with NO fitting
     (_mean_adapter, the same reference every dev fold used). No Ebola label is read.
  3. FEW-SHOT arm. One fresh Adapter fit on the arm's SUPPORT cells only. The epoch count comes from
     leave-one-district-out CV inside the support set (A2) -- no query cell is read at any point.
  4. SCORE the query set at the common origin set t in [19, 36] (A1).
  5. FLOORS. persistence and support_mean. Seasonal-naive is NOT reported: T=52 puts the t-52 lag
     out of panel at every scored origin, so it would be a duplicate of persistence (A5).

Run from the repo root:
  PYTHONNOUSERSITE=1 conda run -n ebola-train python -m train.ebola --dry-run      # throwaway, misc/
  PYTHONNOUSERSITE=1 conda run -n ebola-train python -m train.ebola --all          # the scored run
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import torch

import bundles
import score
from bundles import HORIZONS, W, content_sha256
from models import (Adapter, SharedEncoder, pinball_loss, sparse_from_dense_np, targets_and_mask,
                    window_slice)
from results_paths import RESULTS, rpath
from train.lodo import DISEASES, _fit_trunk, _mean_adapter, _score
from train.loop import (DEVICE, NAIVE_META, score_predictions, write_checkpoint, write_per_node,
                        write_per_origin, write_quantiles, write_records)

MANIFEST = pathlib.Path("configs/ebola_arms.json")
SEEDS = (42, 52, 62, 72, 82)
TRUNK_PATIENCE = 30           # A5; LDO3 used 12, and this run is scored once
ADAPTER_EPOCHS_MAX = 80       # the CV searches within this; encoder_base.yaml max_epochs


# --------------------------------------------------------------------------- #
# The locked config. Read from disk and verified, never restated here.
# --------------------------------------------------------------------------- #
def load_manifest(verify=True) -> dict:
    """The frozen arms. Re-hashes each bundle's array content and refuses to go on if it moved.

    This is the point of the pre-registration: if the split has drifted since it was frozen, the
    run is not the run that was registered, and it must fail rather than quietly score a different
    experiment."""
    man = json.loads(MANIFEST.read_text())
    if not verify:
        return man
    for name, arm in man["arms"].items():
        got = content_sha256(pathlib.Path(arm["npz"]))
        if got != arm["sha256_content"]:
            raise SystemExit(
                f"FROZEN ARM MOVED: {name}\n  registered {arm['sha256_content']}\n  on disk    {got}\n"
                f"The split is not the one {man['prereg']} registered. Refusing to score.")
    return man


def alldev_plan():
    """(in_names, adapter_groups) for the all-development-disease trunk: one group per DISEASE."""
    in_names, groups = [], []
    for g, dis in enumerate(DISEASES):
        for n in DISEASES[dis]:
            in_names.append(n); groups.append(g)
    return in_names, groups


# --------------------------------------------------------------------------- #
# Support-side windowing. Adaptation origins are SHORT (t < W-1) and left-padded (A3).
# --------------------------------------------------------------------------- #
def support_origins(b) -> list[int]:
    """Origins t >= 0 with at least one SUPPORT target at t+h for some h.

    Deliberately NOT bundles.origins(): that requires a full window (t >= W-1) and a common origin
    set across horizons, which on these arms yields ZERO adaptation origins -- support reaches
    column 12 (primary) or 20 (secondary) and the first full-window origin is 19. The comparability
    argument for a common origin set applies to SCORING, where horizons are read against each other;
    it does not apply to fitting, where discarding supervision buys nothing."""
    s = b.masks()["support"].astype(bool)
    T = s.shape[1]
    return [t for t in range(T) if any(t + h < T and s[:, t + h].any() for h in HORIZONS)]


def _precompute_features(enc, Z, A, Mt, origins, device):
    """Trunk output per origin, computed ONCE. The trunk is frozen for the whole adaptation step, so
    its features are constants: recomputing them inside the CV would multiply the encoder cost by
    (n_districts x n_epochs) for an identical result."""
    enc.eval()
    with torch.no_grad():
        return {t: enc(window_slice(Z, t), A, Mt[:, t]).detach() for t in origins}


def _fit_adapter(feats, ymod, Mt, fit_mask, origins, seed, epochs, device,
                 lr=1e-3, wd=1e-4, batch_origins=8, track_mask=None):
    """Fit ONE Adapter on the cells flagged by `fit_mask`. Trunk already frozen and precomputed.

    Returns (adapter, curve) where curve[e] is the pooled pinball on `track_mask` after epoch e
    (None when track_mask is None). Optimiser, lr, wd, clip and accumulation match
    lodo._fit_adapter_and_score exactly, so the only thing that differs from a dev-fold adapter fit
    is the data and the stopping rule."""
    torch.manual_seed(seed); np.random.seed(seed)
    ad = Adapter().to(device)
    opt = torch.optim.AdamW(ad.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    curve = []
    for ep in range(epochs):
        ad.train(True)
        opt.zero_grad(); pending = 0
        for t in np.random.permutation(origins):
            tgt, msk = targets_and_mask(ymod, Mt, fit_mask, int(t), device)
            if msk.sum() == 0:
                continue
            pinball_loss(ad(feats[int(t)]), tgt, msk).backward()
            pending += 1
            if pending == batch_origins:
                torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
                opt.step(); opt.zero_grad(); pending = 0
        if pending:
            torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        sched.step()
        if track_mask is not None:
            curve.append(_pooled_pinball(ad, feats, ymod, Mt, track_mask, origins, device))
    return ad, (curve if track_mask is not None else None)


def _pooled_pinball(ad, feats, ymod, Mt, mask, origins, device) -> float:
    """Cell-count-weighted pinball over `mask`. Pooled, not a mean of per-origin means: an origin
    with one held-out cell must not weigh as much as one with twelve."""
    ad.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for t in origins:
            tgt, msk = targets_and_mask(ymod, Mt, mask, t, device)
            k = int(msk.sum())
            if k == 0:
                continue
            tot += float(pinball_loss(ad(feats[t]), tgt, msk)) * k
            n += k
    return tot / max(n, 1)


def choose_epochs(feats, ymod, Mt, smask, origins, seed, device, epochs_max=ADAPTER_EPOCHS_MAX,
                  verbose=True):
    """Leave-one-district-out CV INSIDE the support set (A2) -> the epoch count for the real fit.

    Ebola has no validation split, so without this the epoch count is a free parameter chosen after
    seeing the answer. Districts, not cells, are the unit: support cells within a district are
    autocorrelated in time, so holding out cells would leak neighbours of the held-out point and
    report an optimistic curve.

    Returns (best_epoch, mean_curve, n_folds). Falls back to (epochs_max, None, 0) when fewer than
    two districts carry an adaptation target -- there is nothing to cross-validate on, and that is
    recorded rather than silently defaulted."""
    # A district participates only if it owns a support cell that is a TARGET at some horizon, i.e.
    # a cell at column >= min(HORIZONS). Districts whose support sits entirely in the first two
    # columns contribute no adaptation pair and would give an empty held-out curve.
    s = smask.detach().cpu().numpy()
    usable = [int(i) for i in np.where(s[:, min(HORIZONS):].sum(1) > 0)[0]]
    if len(usable) < 2:
        return epochs_max, None, 0

    curves, weights = [], []
    for i in usable:
        fit = smask.clone(); fit[i] = 0.0
        held = torch.zeros_like(smask); held[i] = smask[i]
        _, curve = _fit_adapter(feats, ymod, Mt, fit, origins, seed, epochs_max, device,
                                track_mask=held)
        n_held = int(sum(int(targets_and_mask(ymod, Mt, held, t, device)[1].sum()) for t in origins))
        curves.append(curve); weights.append(n_held)
    C = np.array(curves)                                   # [folds, epochs]
    w = np.array(weights, dtype=np.float64)
    mean_curve = (C * w[:, None]).sum(0) / w.sum()         # weight a fold by its held-out cells
    best = int(np.argmin(mean_curve)) + 1
    if verbose:
        print(f"    LODO-CV over {len(usable)} support districts ({int(w.sum())} held-out cells): "
              f"best epoch {best} (pinball {mean_curve.min():.4f}, "
              f"ep{epochs_max} {mean_curve[-1]:.4f})")
    return best, mean_curve.tolist(), len(usable)


def _check_e4(ad, smask, seed, device, arm_name, rtol=1e-5):
    """Pre-registration E4, second correction (A6): a horizon with NO support target must come out a
    UNIFORM SHRINK of its initialisation, p_final = c * p_init for one scalar c just under 1.

    Not "bit-identical", which is what A4 said and what this gate first fired on. The h15 rows do
    receive a gradient -- it is exactly zero, because the mask zeroes their term of the pinball loss
    -- but AdamW's weight decay is DECOUPLED, so it shrinks every parameter it is given each step
    whether or not the gradient is zero. With grad == 0 the Adam moment terms stay at zero and the
    only surviving update is p <- p * (1 - lr_k * wd), which is the same factor for every element of
    the block. So the block scales and does not rotate.

    That makes this a sharper gate than equality, not a weaker one: proportionality across 100
    elements is destroyed by any label-driven gradient, however small, while equality would also
    have been destroyed by an optimiser detail that carries no information about Ebola.

    `rtol` is relative to max|p_init| and is set for float32 ACCUMULATION, not for a single rounding.
    The shrink is applied element-wise once per optimiser step, so the error random-walks: ~160 steps
    at a per-step rounding of ~6e-9 predicts ~1e-6 relative, and the dry run measured 1.4e-6. A gate
    at float32 eps would fire on arithmetic; 1e-5 is roughly an order of magnitude above the
    accumulation and orders of magnitude below anything a real gradient would produce, since a real
    gradient would also be strongly NON-uniform across the block rather than a clean rescale."""
    torch.manual_seed(seed); np.random.seed(seed)
    fresh = Adapter().to(device)                      # same init _fit_adapter drew, same seed
    nQ = len(score.QUANTILE_LEVELS)
    for j, h in enumerate(HORIZONS):
        rows = slice(j * nQ, (j + 1) * nQ)
        if int(smask[:, h:].sum()) > 0:
            continue                                  # this horizon has supervision; nothing to check
        got = torch.cat([ad.head.weight[rows].reshape(-1), ad.head.bias[rows].reshape(-1)]).detach().cpu()
        ini = torch.cat([fresh.head.weight[rows].reshape(-1), fresh.head.bias[rows].reshape(-1)]).detach().cpu()
        c = float((got / ini).median())
        rel = float((got - c * ini).abs().max()) / float(ini.abs().max())
        if rel > rtol:
            raise SystemExit(
                f"{arm_name} seed {seed}: h{h} has 0 support targets but its head block is not a "
                f"uniform shrink of its initialisation (relative residual {rel:.2e} > {rtol:.0e}, "
                f"c={c:.9f}). A label-driven gradient reached a horizon with no labels -- E4/A6.")
        if not 0.0 <= 1.0 - c < 1e-3:
            raise SystemExit(f"{arm_name} seed {seed}: h{h} shrink factor c={c:.9f} is not a "
                             f"weight-decay-scale shrink. Something other than decay moved it.")
        print(f"    E4 ok: h{h} has 0 support targets; head block is a uniform shrink "
              f"(c={c:.9f}, rel residual {rel:.1e}) -- decay only, no label signal")


# --------------------------------------------------------------------------- #
# Naive floors. persistence + support_mean only (A5).
# --------------------------------------------------------------------------- #
def ebola_naive_predictions(b, te):
    """{model: {h: [N,T]}} in count space, on the same scored cells as the encoder.

    `support_mean` is the Ebola analogue of train_mean: the per-node mean over that arm's SUPPORT
    cells, which is the only history a few-shot method is allowed. A district with no support cell
    has no in-protocol history at all, so it takes 0 -- the same convention train_mean uses for a
    node with an empty train fold, and 43 of 61 districts are in that position on the primary arm.
    Seasonal-naive is absent by design: T=52, so t+h-52 < 0 at every scored origin and it would
    collapse to a copy of persistence rather than being a distinct floor."""
    N, T = b.raw.shape
    raw = b.raw.astype(np.float64)
    smask = b.masks()["support"].astype(bool)
    smean = np.array([raw[i, smask[i]].mean() if smask[i].any() else 0.0 for i in range(N)])
    out = {m: {h: np.zeros((N, T)) for h in HORIZONS} for m in ("persistence", "support_mean")}
    for t in te:
        for h in HORIZONS:
            out["persistence"][h][:, t + h] = raw[:, t]          # y_t
            out["support_mean"][h][:, t + h] = smean
    return out


# --------------------------------------------------------------------------- #
# One seed: trunk (cached) -> zero-shot + few-shot on every arm.
# --------------------------------------------------------------------------- #
def get_trunk(seed, device, prefix, trunk_steps=91000, trunk_patience=TRUNK_PATIENCE,
              val_every=1000, verbose=True):
    """The all-dev trunk for this seed, trained once and reused by both arms.

    Cached to results/ebola/: the two arms MUST share a trunk, or an L12-vs-L20 difference would
    partly be a difference between two trunk runs. Reloading also makes a re-score after a scoring
    bug cheap, which matters when the protocol says the set is scored once."""
    fname = f"{prefix}__alldev__seed{seed}__ckpt.pt"
    path = rpath(fname, root=RESULTS)
    in_names, groups = alldev_plan()
    if path.exists():
        payload = torch.load(path, map_location=device, weights_only=False)
        enc = SharedEncoder(gate_mode="learned").to(device)
        enc.load_state_dict(payload["encoder"])
        zero = Adapter().to(device)
        zero.load_state_dict(payload["adapter"])
        if verbose:
            print(f"  trunk seed {seed}: loaded {path}")
        return enc, zero, payload["meta"]

    if verbose:
        print(f"  trunk seed {seed}: training on {in_names} groups={groups} "
              f"steps={trunk_steps} patience={trunk_patience}")
    t0 = time.time()
    enc, in_ads = _fit_trunk(seed, in_names, device, steps=trunk_steps, patience=trunk_patience,
                             val_every=val_every, adapter_groups=groups, verbose=verbose)
    zero = _mean_adapter(in_ads, device)                  # deduped by identity -> 3 diseases, not 5
    meta = dict(in_names=in_names, adapter_groups=groups, trunk_steps=trunk_steps,
                trunk_patience=trunk_patience, val_every=val_every, seed=seed,
                wall_s=round(time.time() - t0, 1))
    write_checkpoint(enc, zero, fname, extra=meta)
    return enc, zero, meta


def run_arm(arm_name, arm, enc, zero_ad, seed, device, prefix, verbose=True):
    """Score one frozen arm at one seed: zero-shot, then few-shot. Returns the record lists."""
    b = bundles.load(arm_name)
    Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=device)
    ymod = torch.tensor(b.y, dtype=torch.float32, device=device)
    Mt = torch.tensor(b.M, dtype=torch.float32, device=device)
    A = sparse_from_dense_np(b.A_geo).to(device)
    smask = torch.tensor(b.masks()["support"], dtype=torch.float32, device=device)
    te = b.origins(phase="query")
    so = support_origins(b)
    assert te == [t for t in range(W - 1, b.X.shape[1]) if t + max(HORIZONS) <= b.X.shape[1] - 1], \
        "query origins are not the common origin set the pre-registration fixed (A1)"

    meta = dict(training_regime="ebola_fewshot", sampler="uniform-perdisease", gate_mode="learned",
                topo_aug="none", fold_structure="few-shot-support-query",
                arm=arm_name, arm_role=arm["role"], support_cutoff=arm["cutoff_date"],
                support_cells=arm["counts"]["support_cells"],
                prereg_sha256=arm["sha256_content"])
    zmeta = dict(meta, training_regime="ebola_zeroshot")

    # --- zero-shot: the borrowed mean adapter, no Ebola label read -------------------------------
    # zquant is NOT optional. The protocol scores each arm exactly once (section 5.1), so a quantile
    # array not written here can never be written: WIS, CRPS, coverage, PIT and any conformal wrapper
    # would be permanently unavailable for the zero-shot arm, and the only remedy would be a re-score
    # the pre-registration forbids. LDO3_Results.md section 5 records this exact omission costing the
    # zero-shot arm its entire UQ block on the development folds; it is not repeated on the run that
    # cannot be redone. Writing it costs a few hundred KB.
    zquant = {}
    zrecs, zpn, zpo, _ = _score(enc, zero_ad, b, Z, Mt, A, te, arm_name, seed,
                                f"{prefix}_zeroshot", zmeta, phase="query", device=device,
                                quant_out=zquant)
    write_records(zrecs, f"{prefix}_zeroshot__{arm_name}__seed{seed}.json")
    write_per_node(zpn, f"{prefix}_zeroshot__{arm_name}__seed{seed}__pernode.npz")
    write_per_origin(zpo, f"{prefix}_zeroshot__{arm_name}__seed{seed}__perorigin.npz")
    write_quantiles(zquant, te, f"{prefix}_zeroshot__{arm_name}__seed{seed}__quantiles.npz")

    # --- few-shot: LODO-CV picks the epoch, then refit on all support ----------------------------
    feats = _precompute_features(enc, Z, A, Mt, so, device)
    best_ep, curve, n_folds = choose_epochs(feats, ymod, Mt, smask, so, seed, device, verbose=verbose)
    ad, _ = _fit_adapter(feats, ymod, Mt, smask, so, seed, best_ep, device)

    _check_e4(ad, smask, seed, device, arm_name)

    quant = {}
    recs, pn, po, _ = _score(enc, ad, b, Z, Mt, A, te, arm_name, seed, prefix, meta,
                             phase="query", device=device, quant_out=quant)
    write_records(recs, f"{prefix}__{arm_name}__seed{seed}.json")
    write_per_node(pn, f"{prefix}__{arm_name}__seed{seed}__pernode.npz")
    write_per_origin(po, f"{prefix}__{arm_name}__seed{seed}__perorigin.npz")
    write_quantiles(quant, te, f"{prefix}__{arm_name}__seed{seed}__quantiles.npz")
    write_checkpoint(enc, ad, f"{prefix}__{arm_name}__seed{seed}__ckpt.pt",
                     extra=dict(meta, adapter_epochs=best_ep, cv_folds=n_folds, cv_curve=curve))
    if verbose:
        print(f"  {arm_name} seed {seed}: adapter fit for {best_ep} epochs on "
              f"{arm['counts']['support_cells']} support cells, scored {len(te)} origins")
    return recs, zrecs


def run_floors(arm_name, tag="naive", verbose=True):
    """persistence + support_mean on the arm's scored cells. Deterministic, so seed is null.

    `tag` exists only so --dry-run can write `naive_smoke__...`, which routes to results/misc/. The
    floors do not depend on the trunk, so a dry run would otherwise drop real, correct records into
    the scored record path -- correct content in the wrong place is still a provenance defect."""
    b = bundles.load(arm_name)
    te = b.origins(phase="query")
    all_recs = []
    for mname, preds in ebola_naive_predictions(b, te).items():
        # model name is the bare floor name, matching train.loop.run_dataset, so analysis.py's
        # naive__<ds>__<floor>__perorigin.npz convention keeps working unchanged.
        recs, pn, po = score_predictions(mname, arm_name, None, preds, b, te, phase="query",
                                         run_meta=NAIVE_META)
        write_per_node(pn, f"{tag}__{arm_name}__{mname}__pernode.npz")
        write_per_origin(po, f"{tag}__{arm_name}__{mname}__perorigin.npz")
        all_recs += recs
    write_records(all_recs, f"{tag}__{arm_name}.json")
    if verbose:
        print(f"  floors scored ({arm_name}): persistence, support_mean")
    return all_recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="both arms, all 5 seeds -- the scored run")
    ap.add_argument("--arm", default=None, help="one arm name, e.g. ebola_L12")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="1 seed, 400 trunk steps, prefix encoder_ebola_smoke -> results/misc/. "
                         "Exercises every path without writing into the scored record.")
    ap.add_argument("--prefix", default="encoder_ebola")
    ap.add_argument("--trunk-steps", type=int, default=91000)
    a = ap.parse_args()

    man = load_manifest()
    print(f"locked config: {MANIFEST}  frozen {man['frozen']}  prereg {man['prereg']}")
    print(f"  arms verified against their registered content hashes: {list(man['arms'])}")

    prefix, steps, val_every, tag = a.prefix, a.trunk_steps, 1000, "naive"
    seeds, arms = SEEDS, list(man["arms"])
    if a.dry_run:
        prefix, steps, val_every, seeds = "encoder_ebola_smoke", 400, 200, (42,)
        tag = "naive_smoke"       # -> results/misc/, so a dry run leaves the record path untouched
        print("  DRY RUN -- 400 trunk steps, artifacts go to results/misc/, nothing scored for the record")
    if a.arm:
        arms = [a.arm]
    if a.seed is not None:
        seeds = (a.seed,)
    if not (a.all or a.arm or a.seed is not None or a.dry_run):
        ap.error("pass --all for the scored run, or --arm/--seed/--dry-run")

    t0 = time.time()
    for arm_name in arms:
        run_floors(arm_name, tag=tag)
    for seed in seeds:
        enc, zero_ad, tmeta = get_trunk(seed, DEVICE, prefix, trunk_steps=steps, val_every=val_every)
        for arm_name in arms:
            run_arm(arm_name, man["arms"][arm_name], enc, zero_ad, seed, DEVICE, prefix)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min  ->  results/"
          f"{'misc' if a.dry_run else 'ebola'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

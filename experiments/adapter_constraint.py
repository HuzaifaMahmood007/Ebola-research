"""EXPLORATORY: does constraining the Ebola adapter toward the zero-effect map shrink the few-shot damage?

Milestone 6 run 2, the CAUSAL step the mechanism doc could not take on its own.
`progress/outcomes/Adapter_Mechanism_2026-09-16.md` established GEOMETRICALLY that 76-85% of what the
few-shot adapter changes acts along directions where the query data spreads 5-12x beyond the support
it was fitted on. Its own limits section (:122-124) says that is geometry, not an error decomposition:
it does not prove the extrapolation CAUSES the measured few-shot damage. This run closes that gap.

THE INSTRUMENT. With the trunk frozen the Adapter is EXACTLY affine (models/adapters.py:14-16):
pred = h A^T + c with A = W diag(gamma), c = W beta + b. The zero-shot ("borrowed mean") adapter is
another affine map (A_z, c_z), the one that adds no Ebola-specific change. The few-shot-specific change
the mechanism doc calls D is exactly A_few - A_z. A ridge penalty lam*||A - A_z||^2 on the affine fit
shrinks the fitted map along the segment toward A_z; in closed form that is exactly

    A(t) = A_z + t*(A_few - A_z),   c(t) = c_z + t*(c_few - c_z),   t = 1/(1 + lam*k) in [0,1]

so t is a reparametrisation of the ridge strength: t=1 is unconstrained few-shot (D untouched),
t=0 is the zero-effect map recovered (D fully removed), and t<1 shrinks D by the factor t. We use the
closed-form shrinkage as the instrument because a literal ridge run through the pre-registration's own
optimiser SATURATES: with the gradient norm clipped to 1.0, cranking lam only shrinks D by ~64% of the
way to A_z and never reaches it (measured; noted in the write-up). Shrinkage spans the full range
exactly and validates on both ends against the archived arms.
  t=1 -> reproduces the archived unconstrained few-shot arm (asserted)
  t=0 -> reproduces the archived zero-shot arm (asserted)
If constraining toward the zero-effect map (t<1) recovers the damage, extrapolation is causal. If it
recovers nothing, the mechanism story weakens and we say so.

A SECOND, PARAMETER-FREE INSTRUMENT (the "clip"). Restrict the few-shot-specific change to the support
row space: A_clip = A_z + (A_few - A_z) @ P, where P projects the 64-dim feature space onto the span
of the support features. This deletes exactly the part of D that acts on directions the support never
constrained -- the out-of-span change the mechanism doc flagged -- and keeps the rest. No knob.

NO EBOLA QUERY OUTCOME TOUCHES ANY FITTING CHOICE. The shrinkage strength t is selected on SUPPORT
ONLY, by leave-one-district-out CV inside the support set, the machinery the pre-registration uses to
pick the adapter epoch (train.ebola.choose_epochs / _fit_adapter). Per fold we refit the few-shot
adapter on the remaining support at the pre-registered epoch count, interpolate that fold map toward
A_z, and score the held-out district. The REPORTED cell is the support-selected t; the full grid is
context. A query-oracle best-over-grid is reported too, labelled as context that peeks at the answer.

PROTOCOL. Everything here is EXPLORATORY and stamped protocol="EXPLORATORY". It never re-scores the
frozen pre-registration, never writes into results/ or data/, and asserts the frozen arm hashes against
configs/ebola_arms.json before and after (train.ebola.load_manifest(verify=True)). The archived
few-shot and zero-shot records and checkpoints in results/ebola/ are read only, as paired anchors, and
are reproduced from the loaded checkpoints before any exploratory number is trusted.

  conda run -n ebola-train python -m experiments.adapter_constraint --selfcheck
  conda run -n ebola-train python -m experiments.adapter_constraint --seeds 42 52 62 72 82

Multi-seed: one JSON per seed as it completes, so a crash costs one seed. Paired by seed against the
archived unconstrained-adapted and zero-shot records. Five paired seeds support sign counts and mean
deltas, nothing stronger; no significance language is printed or stored.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import json
import time

import numpy as np
import torch

import bundles
import score
from bundles import HORIZONS
from diagnostics.adapter_mechanism import design_rows
from models import Adapter, SharedEncoder, sparse_from_dense_np, targets_and_mask
from train.ebola import (_fit_adapter, _pooled_pinball, _precompute_features, load_manifest,
                         support_origins)
from train.loop import DEVICE

# reuse the exploratory plumbing that norm_probe already validated
from experiments.norm_probe import OUT, TAG, archived, score_arm, tensors

EBOLA_ARMS = ("ebola_L12", "ebola_L20")
SEEDS = (42, 52, 62, 72, 82)
# t = 1 unconstrained few-shot, t = 0 the zero-effect (borrowed) map, t<1 shrinks the few-shot change
T_GRID = tuple(round(x, 3) for x in np.linspace(0.0, 1.0, 11))
RANK_TOL = 1e-8          # support-span rank threshold, relative to the largest singular value


def _dump(obj, fname):
    p = OUT / fname
    assert OUT.resolve() in p.resolve().parents or p.resolve().parent == OUT.resolve(), \
        f"output escaping experiments/: {p}"
    p.write_text(json.dumps(obj, indent=2))
    return p


def metric_by_h(recs, metric):
    return {r["horizon"]: r["country_macro"] for r in recs if r["metric"] == metric}


def to_affine(ad):
    """(A, c) of an adapter's frozen-trunk affine map: A = W diag(gamma), c = W beta + b."""
    with torch.no_grad():
        A = (ad.head.weight @ torch.diag(ad.gamma)).detach()
        c = (ad.head.weight @ ad.beta + ad.head.bias).detach()
    return A, c


def affine_adapter(A, c, device):
    """An Adapter whose forward is exactly h @ A^T + c (gamma=1, beta=0, head=A/b=c)."""
    ad = Adapter().to(device)
    with torch.no_grad():
        ad.gamma.copy_(torch.ones_like(ad.gamma))
        ad.beta.copy_(torch.zeros_like(ad.beta))
        ad.head.weight.copy_(A)
        ad.head.bias.copy_(c)
    ad.eval()
    return ad


def shrink(A_few, c_few, A_z, c_z, t):
    """A_z + t(A_few - A_z), c_z + t(c_few - c_z). t=1 few-shot, t=0 zero-effect map."""
    return A_z + t * (A_few - A_z), c_z + t * (c_few - c_z)


def clip_to_span(A_f, A_z, feats, smask, origins, device):
    """Restrict the few-shot change to the PER-HORIZON support span. Returns (A_clip, {h: rank}).

    The map A is [H*Q, 64]; rows [j*Q:(j+1)*Q] drive horizon j. For each horizon we build the span of
    the support FEATURE rows that the fit consumed at that horizon (diagnostics.adapter_mechanism.
    design_rows, the same rows the mechanism doc measured), project the change D=A_f-A_z onto it, and
    keep A_z on the part the support never constrained. A horizon with 0 support rows (L12 h15) has an
    empty span, so its whole block collapses to the zero-effect map. Done per horizon on purpose:
    pooling all origins spans the full 64 dims and the clip becomes a no-op."""
    nQ = len(score.QUANTILE_LEVELS)
    s = smask.detach().cpu().numpy().astype(bool)
    S = design_rows(feats, s, origins)          # {h: [n, 64]}
    A_clip = A_z.clone().double()
    Df = (A_f - A_z).double()
    ranks = {}
    for j, h in enumerate(HORIZONS):
        rows = slice(j * nQ, (j + 1) * nQ)
        X = S[h]
        if len(X) == 0:
            ranks[h] = 0
            continue                            # empty span -> block stays at the zero-effect map
        _, sv, Vt = np.linalg.svd(X, full_matrices=False)
        k = int((sv > sv.max() * RANK_TOL).sum())
        V = Vt[:k]
        P = torch.tensor(V.T @ V, dtype=torch.float64, device=device)
        A_clip[rows] = A_z[rows].double() + Df[rows] @ P
        ranks[h] = k
    return A_clip.float(), ranks


def pooled_pinball_affine(A, c, feats, ymod, Mt, mask, origins, device):
    ad = affine_adapter(A.float(), c.float(), device)
    return _pooled_pinball(ad, feats, ymod, Mt, mask, origins, device)


def choose_t(feats, ymod, Mt, smask, origins, seed, device, A_z, c_z, best_ep):
    """Leave-one-district-out CV inside support -> the shrinkage t, mirroring choose_epochs.

    Per held-out district: refit the few-shot adapter on the remaining support at the pre-registered
    epoch count, interpolate that fold map toward the zero-effect map over T_GRID, and score the
    held-out district's pinball. Curves are pooled weighted by held-out cells; the t minimising the
    pooled held-out pinball is selected. Reads NO query cell. Falls back to t=1 (unconstrained) when
    fewer than two districts carry an adaptation target."""
    s = smask.detach().cpu().numpy()
    usable = [int(i) for i in np.where(s[:, min(HORIZONS):].sum(1) > 0)[0]]
    if len(usable) < 2:
        return 1.0, [dict(t=t, cv=float("nan")) for t in T_GRID], 0
    tot = np.zeros(len(T_GRID)); wsum = 0.0
    for i in usable:
        fit = smask.clone(); fit[i] = 0.0
        held = torch.zeros_like(smask); held[i] = smask[i]
        ad_i, _ = _fit_adapter(feats, ymod, Mt, fit, origins, seed, best_ep, device)
        A_i, c_i = to_affine(ad_i)
        n_held = int(sum(int(targets_and_mask(ymod, Mt, held, t, device)[1].sum()) for t in origins))
        for j, tval in enumerate(T_GRID):
            At, ct = shrink(A_i, c_i, A_z, c_z, tval)
            tot[j] += pooled_pinball_affine(At, ct, feats, ymod, Mt, held, origins, device) * n_held
        wsum += n_held
    curve = (tot / max(wsum, 1)).tolist()
    grid = [dict(t=float(T_GRID[j]), cv=float(curve[j])) for j in range(len(T_GRID))]
    best_t = float(T_GRID[int(np.argmin(curve))])
    return best_t, grid, len(usable)


def run_arm(arm, seed, enc, zero_state, device, verbose=True):
    b = bundles.load(arm)
    Z, Mt, A = tensors(b, device)
    ymod = torch.tensor(b.y, dtype=torch.float32, device=device)
    smask = torch.tensor(b.masks()["support"], dtype=torch.float32, device=device)
    so = support_origins(b)
    feats = _precompute_features(enc, Z, A, Mt, so, device)

    # the deployed maps: archived few-shot (the damage baseline) and zero-shot (the constraint target)
    fk = torch.load(f"results/ebola/encoder_ebola__{arm}__seed{seed}__ckpt.pt",
                    map_location=device, weights_only=False)
    best_ep = int(fk["meta"]["adapter_epochs"])
    few = Adapter().to(device); few.load_state_dict(fk["adapter"]); few.eval()
    zero = Adapter().to(device); zero.load_state_dict(zero_state); zero.eval()
    A_f, c_f = to_affine(few); A_z, c_z = to_affine(zero)

    # anchors, reproduced from the loaded checkpoints
    arch_few = archived(f"results/ebola/encoder_ebola__{arm}__seed{seed}.json")
    arch_zero = archived(f"results/ebola/encoder_ebola_zeroshot__{arm}__seed{seed}.json")
    fewR = metric_by_h(score_arm(enc, few, b, seed, "c_few", dict(arm=arm), "query", device), "mae")
    zeroR = metric_by_h(score_arm(enc, zero, b, seed, "c_zero", dict(arm=arm), "query", device), "mae")
    for h in HORIZONS:
        assert abs(fewR[h] - arch_few[h]) < max(1e-3, 1e-5 * abs(arch_few[h])), \
            f"{arm} s{seed} h{h}: few rebuild {fewR[h]} != archived {arch_few[h]}"
        assert abs(zeroR[h] - arch_zero[h]) < max(1e-3, 1e-5 * abs(arch_zero[h])), \
            f"{arm} s{seed} h{h}: zero rebuild {zeroR[h]} != archived {arch_zero[h]}"

    # support-only selection of t
    best_t, tgrid, nf = choose_t(feats, ymod, Mt, smask, so, seed, device, A_z, c_z, best_ep)

    # score the query set across the full grid (context) and pull out the selected + oracle
    grid = []
    for tval in T_GRID:
        At, ct = shrink(A_f, c_f, A_z, c_z, tval)
        recs = score_arm(enc, affine_adapter(At.float(), ct.float(), device), b, seed,
                         "c_t", dict(arm=arm, t=tval), "query", device)
        grid.append(dict(t=float(tval), mae=metric_by_h(recs, "mae"), rmse=metric_by_h(recs, "rmse")))

    # the parameter-free clip: delete the out-of-span part of D, per horizon
    A_clip, ranks = clip_to_span(A_f, A_z, feats, smask, so, device)
    clip_recs = score_arm(enc, affine_adapter(A_clip, c_f.float(), device), b, seed,
                          "c_clip", dict(arm=arm, clip="span"), "query", device)

    sel = next(g for g in grid if g["t"] == best_t)
    return dict(
        arm=arm, seed=seed, best_ep=best_ep, selected_t=best_t,
        span_rank={str(h): ranks[h] for h in HORIZONS},
        few_mae={str(h): arch_few[h] for h in HORIZONS},
        zero_mae={str(h): arch_zero[h] for h in HORIZONS},
        selected_mae={str(h): sel["mae"][h] for h in HORIZONS},
        clip_mae={str(h): metric_by_h(clip_recs, "mae")[h] for h in HORIZONS},
        t_cv=tgrid, n_folds=nf,
        grid=[dict(t=g["t"], mae={str(h): g["mae"][h] for h in HORIZONS},
                   rmse={str(h): g["rmse"][h] for h in HORIZONS}) for g in grid],
    )


def summarise(rows):
    """Seed-mean few / constrained(selected) / clip / zero per (arm,h), and recovery of the damage.

    recovery = (few - constrained) / (few - zero): the share of the few-shot-minus-zero-shot gap the
    constraint closes at the SUPPORT-selected t. 1.0 = full recovery to zero-shot, 0.0 = none. The
    oracle column is the best-over-grid recovery, which peeks at the query answer -- CONTEXT ONLY."""
    seeds = sorted({r["seed"] for r in rows})
    out = []
    for arm in EBOLA_ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if len(rs) != len(seeds):
            continue
        for h in HORIZONS:
            hs = str(h)
            few = float(np.mean([r["few_mae"][hs] for r in rs]))
            zero = float(np.mean([r["zero_mae"][hs] for r in rs]))
            con = float(np.mean([r["selected_mae"][hs] for r in rs]))
            clip = float(np.mean([r["clip_mae"][hs] for r in rs]))
            oracle = float(np.mean([min(g["mae"][hs] for g in r["grid"]) for r in rs]))
            gap = few - zero
            rec = (few - con) / gap if abs(gap) > 1e-9 else float("nan")
            rec_or = (few - oracle) / gap if abs(gap) > 1e-9 else float("nan")
            rec_clip = (few - clip) / gap if abs(gap) > 1e-9 else float("nan")
            out.append(dict(arm=arm, h=h, n_seeds=len(seeds), few=few, zero=zero, constrained=con,
                            clip=clip, oracle=oracle, gap=gap, recovery=rec, recovery_clip=rec_clip,
                            recovery_oracle=rec_or,
                            selected_t_by_seed=[r["selected_t"] for r in rs]))
    return out


def print_summary(summary):
    print(f"\n{'=' * 104}")
    print("CONSTRAINED-ADAPTER CAUSAL TEST -- country-macro MAE, seed means")
    print(f"{'=' * 104}")
    print(f"{'arm':<11}{'h':>3}{'few':>9}{'zero':>9}{'constr':>9}{'clip':>9}{'gap':>8}"
          f"{'recov':>8}{'recov_clip':>12}{'oracle':>9}")
    for s in summary:
        print(f"{s['arm']:<11}{s['h']:>3}{s['few']:>9.2f}{s['zero']:>9.2f}{s['constrained']:>9.2f}"
              f"{s['clip']:>9.2f}{s['gap']:>+8.2f}{s['recovery']:>+8.0%}{s['recovery_clip']:>+12.0%}"
              f"{s['recovery_oracle']:>+9.0%}")
    print("""
read:
  few         archived unconstrained adapted arm (the damage baseline)
  zero        archived borrowed-mean arm (the zero-effect target the constraint pulls toward)
  constr      shrinkage to the zero-effect map at the SUPPORT-selected t, this run
  clip        parameter-free: the few-shot change restricted to the support span
  gap         few - zero. positive means few-shot is worse (there is damage to recover)
  recov       (few - constr)/(few - zero) at the support-selected t. THE verdict rides on this
  recov_clip  same recovery fraction for the parameter-free clip
  oracle      best-over-grid recovery, peeking at the query answer. CONTEXT ONLY, not a selection""")


def _load_trunk(seed, device):
    tk = torch.load(f"results/ebola/encoder_ebola__alldev__seed{seed}__ckpt.pt",
                    map_location=device, weights_only=False)
    enc = SharedEncoder(gate_mode="learned").to(device); enc.load_state_dict(tk["encoder"]); enc.eval()
    return enc, tk["adapter"]


def selfcheck(device=DEVICE):
    load_manifest(verify=True)
    print("  ok frozen arm hashes verified against configs/ebola_arms.json")

    torch.manual_seed(1)
    ad = Adapter().to(device); ad.eval()
    h = torch.randn(9, 64, device=device)
    with torch.no_grad():
        ref = ad(h)
        A, c = to_affine(ad)
        got = affine_adapter(A.float(), c.float(), device)(h)
    dev = float((got - ref).abs().max())
    assert dev == 0.0, f"affine reconstruction is not exact: {dev}"
    print(f"  ok adapter is exactly affine and affine_adapter round-trips it (deviation {dev})")

    seed = 42
    enc, zero_state = _load_trunk(seed, device)
    for arm in EBOLA_ARMS:
        b = bundles.load(arm)
        fk = torch.load(f"results/ebola/encoder_ebola__{arm}__seed{seed}__ckpt.pt",
                        map_location=device, weights_only=False)
        few = Adapter().to(device); few.load_state_dict(fk["adapter"]); few.eval()
        zero = Adapter().to(device); zero.load_state_dict(zero_state); zero.eval()
        A_f, c_f = to_affine(few); A_z, c_z = to_affine(zero)

        # t=1 shrinkage == archived few-shot MAE; t=0 == archived zero-shot MAE (both exact)
        for tval, ref_file in ((1.0, f"results/ebola/encoder_ebola__{arm}__seed{seed}.json"),
                               (0.0, f"results/ebola/encoder_ebola_zeroshot__{arm}__seed{seed}.json")):
            At, ct = shrink(A_f, c_f, A_z, c_z, tval)
            got = metric_by_h(score_arm(enc, affine_adapter(At.float(), ct.float(), device), b, seed,
                                        "sc", dict(arm=arm), "query", device), "mae")
            ref = archived(ref_file)
            for hh in HORIZONS:
                assert abs(got[hh] - ref[hh]) < max(1e-3, 1e-5 * abs(ref[hh])), \
                    f"{arm} t={tval} h{hh}: {got[hh]} != archived {ref[hh]}"
        print(f"  ok {arm}: t=1 reproduces archived few-shot, t=0 reproduces archived zero-shot")

    load_manifest(verify=True)
    print("  ok frozen arm hashes intact after selfcheck")
    print("selfcheck passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--arms", nargs="+", default=list(EBOLA_ARMS))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    selfcheck()
    all_rows = []
    for seed in a.seeds:
        print(f"\n================ seed {seed} ================")
        enc, zero_state = _load_trunk(seed, DEVICE)
        rows = []
        for arm in a.arms:
            t0 = time.time()
            r = run_arm(arm, seed, enc, zero_state, DEVICE)
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"  {arm} seed {seed}: selected t={r['selected_t']} (ep {r['best_ep']}, "
                  f"span ranks {r['span_rank']}), {r['wall_s']}s")
        all_rows += rows
        load_manifest(verify=True)
        _dump(dict(protocol=TAG, seed=seed, rows=rows), f"adapter_constraint__seed{seed}.json")
        print(f"  seed {seed} done, hashes intact")
    if len(a.seeds) > 1:
        summary = summarise(all_rows)
        print_summary(summary)
        _dump(dict(protocol=TAG, seeds=a.seeds, t_grid=list(T_GRID), cells=summary),
              "adapter_constraint__summary.json")
    load_manifest(verify=True)
    print("frozen arm hashes re-verified after the full run: intact")


if __name__ == "__main__":
    main()

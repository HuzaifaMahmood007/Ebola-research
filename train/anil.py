"""train/anil.py -- ANIL as an ABLATION of the adaptation procedure, single direction.

WHAT THIS IS AND IS NOT. This answers D2's question -- "does meta-learning beat a linear probe" --
as a number rather than an assertion. It is reported as an ablation of the ADAPTATION PROCEDURE,
never as a headline, and that framing does not change if the number comes back positive. The
headline remains freeze-then-adapt, because that is what the Ebola pre-registration names and what
every existing result was produced under.

THE ALGORITHM (D13). MAML restricted to the adaptation surface = ANIL, exact second order. Inner
loop touches ONLY the adapter; the outer loop updates the trunk and the adapter's initialisation.
`--first-order` drops the second-order term for memory/time, and is a stated deviation from D13.

THE THREE ARMS. All three share a seed's warm-start trunk, all three are scored by the identical
frozen-trunk-plus-fresh-adapter pass, so they differ in ONE thing each:

  * `anil`     -- meta-training as above.
  * `control`  -- the SAME episodes and the SAME number of outer updates with NO inner loop, i.e.
                  ordinary ERM on the same dengue cells. Without it the ANIL arm is confounded with
                  simply having had more dengue training than the baseline, and a negative result
                  would be uninterpretable. Run it; it is the same cost.
  * `baseline` -- not run here. It is the capacity probe's own row for this surface, already on disk
                  at five seeds, and `compare()` differences against it.

THE SURFACE. `--surface mlp-256` (default, 21,780 params) is D19's strongest rung: 3 significant
cross-disease wins (us-regions h15 +19.8%, japan h15 +6.6%, us-states h15 +4.5%) and the smallest h3
loss of the ladder (-16.6% vs mlp-64's -22.5%). Three things travel with that choice and belong in
the write-up whatever this run says:

  * EVERY significant gain in that sweep sits at h15, and the frozen Ebola PRIMARY arm has ZERO
    adaptation pairs at h15 (D16: 48/38/18/0). Worse, at h3 -- the only practical Ebola adaptation
    horizon per D4 -- this surface's one significant cell is a LOSS. The surface is selected on
    evidence that is positive where Ebola cannot use it and negative where Ebola will.
  * It is the only rung with a significant loss in D19's in-domain control (dengue h15, -1.0%).
  * `Reports/MAML_Decision.md` chose ANIL over MAML BECAUSE its meta-objective is the Ebola
    procedure differentiated, and that procedure is the pre-registered 1,428-param FiLM+head. An
    inner loop over 21,780 params is not that procedure, so this surface weakens the reason ANIL was
    picked. `--surface affine` runs the pre-registered surface instead; `compare()` reports against
    BOTH references regardless, because affine is what D2 means by "the linear probe".

THE DIRECTION. `--fold dengue2flu` (the default, and the run already reported) is dengue ->
influenza, matching capacity-probe arm 1: the Ebola-shaped direction (train big, adapt to a small
unseen graph; work order 3c), with baselines already on disk at five seeds.

THE OTHER FOLDS. `--fold dengue|influenza|covid` runs the disease-out folds of the LDO3 transfer
table: meta-train on that fold in-diseases, meta-test on the held-out one, warm starting from that
fold own LDO3 trunk and differenced against that fold own freeze-then-adapt row. They exist because
G2 calls meta-learning REQUIRED and a verdict resting on one held-out disease is not an answer to it.
Two things to carry into any write-up of them: `--fold influenza` is the SAME direction as the legacy
run but meta-trains across dengue AND covid, so it is the arm that answers the objection below; and
`--fold dengue` is the backwards-from-Ebola direction (work order 3c), so it is evidence about the
method, not about the case study. A covid fold also inherits the unresolved client decision B5.

THE OBJECTION WE RAISE AGAINST OURSELVES (D13, unretracted, and NOT mitigated here). Meta-training
on dengue ALONE means episodes vary population and time origin, not disease, so ANIL is trained on
the axis that already works. Episodes are drawn per-country, but no episode structure inside one
disease manufactures a cross-disease meta-objective. D13 called the fix a third disease and recorded
that COVID was not on disk; that has since changed -- `results/lodo/encoder_ldo3__influenza__seed*
__ckpt.pt` are five dengue+covid trunks with influenza held out. Meta-training across two diseases
off those trunks is therefore now possible at similar cost, and is an open design decision, NOT
something this module does. State the objection next to any number this produces.

WHAT THIS DOES NOT MEASURE. `meta_test` fits a fresh adapter on the FULL influenza train folds, so
this run asks "is the meta-trained trunk a better frozen feature extractor", not "does it adapt
better from few examples". There is no few-shot regime in it. For a project whose claim is few-shot
adaptation, that is a stated limitation of this ablation, not a property of ANIL.

    conda run -n ebola-train python -m train.anil --selfcheck              # logic only, seconds
    conda run -n ebola-train python -m train.anil --preflight              # prerequisites, seconds
    conda run -n ebola-train python -m train.anil --timing                 # prices an outer step
    conda run -n ebola-train python -m train.anil --arm anil --seeds 42 52 62 72 82
    conda run -n ebola-train python -m train.anil --arm control --seeds 42 52 62 72 82

Run as a MODULE from the repo root, or `import bundles` fails.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

import bundles
from bundles import HORIZONS
from models import SharedEncoder, pinball_loss, targets_and_mask, window_slice
from models.adapters import Adapter
from models.config import D_HIDDEN, QUANTILES
from results_paths import RESULTS, rpath
from train.joint import _prepare, _test_dataset
from train.lodo import FLU_NAMES, _fit_shared_adapter, _fit_trunk, _ldo3_plan
from train.loop import (DEVICE, write_checkpoint, write_per_node, write_per_origin, write_quantiles,
                        write_records)

SEEDS = (42, 52, 62, 72, 82)
META_NAME = "dengue"                  # the LEGACY fold's meta-train disease. See the objection above.
ARMS = ("anil", "control")

# --------------------------------------------------------------------------- #
# FOLDS. `dengue2flu` is the run already on disk and already reported (D13); its names, paths and
# every default are frozen so re-running it reproduces the same artifacts byte for byte. The three
# LDO3 folds are the disease-out folds of the main transfer table, and they exist so the
# meta-learning verdict is not one fold wide -- G2 calls meta-learning REQUIRED, and 12 cells on a
# single held-out disease is not an answer to it.
#
# Every LDO3 fold has a MULTI-BUNDLE meta-train side (holding dengue out leaves three influenza
# panels plus covid), which is why `meta_train` samples a panel per episode instead of taking ds[0].
# The legacy fold is the only single-bundle one and it still reads exactly as it did.
# --------------------------------------------------------------------------- #
LEGACY_FOLD = "dengue2flu"
FOLDS = (LEGACY_FOLD, "dengue", "influenza", "covid")


def fold_plan(fold):
    """(meta-train bundle names, meta-test bundle names, direction label) for one fold.

    The LDO3 folds defer to `train.lodo._ldo3_plan`, the same function that built the transfer table,
    so a fold here can never disagree with the fold of the same name there."""
    assert fold in FOLDS, f"unknown fold {fold!r}; want one of {FOLDS}"
    if fold == LEGACY_FOLD:
        return [META_NAME], list(FLU_NAMES), "dengue2flu"
    in_names, _groups, held_names = _ldo3_plan(fold)
    return list(in_names), list(held_names), f"ldo3:{fold}"


def fold_tag(fold):
    return "dengue2flu" if fold == LEGACY_FOLD else f"ldo3{fold}"
CAP_JSON = RESULTS / "misc" / "capacity_probe_5seed.json"    # the reference this is differenced against
# Two-sided 95% t, same table and convention as the capacity probe. scipy is not a dependency.
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


class MLPAdapter(nn.Module):
    """Byte-for-byte the capacity probe's `mlp-256` surface. Duplicated deliberately: importing from
    `diagnostics.capacity_probe` would make a training module depend on a diagnostic, and the probe's
    import side-effects do not belong in a training run. The self-check asserts this is state-dict
    identical to the probe's own factory, so drift is caught in seconds rather than after a run."""

    def __init__(self, d=D_HIDDEN, hidden=256, horizons=HORIZONS, quantiles=QUANTILES):
        super().__init__()
        self.nH, self.nQ = len(horizons), len(quantiles)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, self.nH * self.nQ))

    def forward(self, h):
        return self.net(h).view(h.shape[0], self.nH, self.nQ)


# label -> (factory, the label the capacity probe wrote into capacity_probe_5seed.json)
SURFACES = {"mlp-256": (lambda: MLPAdapter(hidden=256), "mlp-256"),
            "affine": (Adapter, "affine (current)")}


def surface_factory(name):
    return SURFACES[name][0]


def meta_ckpt(seed, surface, arm, fold=LEGACY_FOLD):
    """Own checkpoint family, keyed by surface AND arm so the three configurations cannot overwrite
    each other. Starts with `encoder_ldo__` so results_paths routes it to lodo/ unchanged, and
    carries `-anil` so it can never collide with the D1 two-disease table or the `-cap` trunks.
    D19 records what silently overwriting a results family costs."""
    return f"encoder_ldo__{fold_tag(fold)}-anil-{surface}-{arm}__seed{seed}__ckpt.pt"


def artifact(surface, arm, ds_name, seed, suffix, fold=LEGACY_FOLD):
    """The legacy fold keeps its exact existing names. A new fold carries its tag, which matters for
    fold="influenza": it meta-tests on the same three panels as the legacy fold, so without the tag
    it would overwrite results that are already reported."""
    tag = "" if fold == LEGACY_FOLD else f"{fold_tag(fold)}-"
    return f"encoder_ldo__{tag}anil-{surface}-{arm}__{ds_name}__seed{seed}{suffix}"


def warm_ckpt(seed, fold=LEGACY_FOLD):
    """This seed's frozen starting trunk. Legacy: the capacity probe's dengue trunk, which produced
    the baseline row `compare()` differences against. LDO3 folds: that fold's own trunk from the
    transfer run, so the ANIL arm starts where freeze-then-adapt started and the delta is
    attributable to the objective rather than to the trunk."""
    if fold == LEGACY_FOLD:
        return f"encoder_ldo__dengue2flu-cap__seed{seed}__ckpt.pt"
    return f"encoder_ldo3__{fold}__seed{seed}__ckpt.pt"


# --------------------------------------------------------------------------- #
# Episodes
# --------------------------------------------------------------------------- #
def country_index(b):
    """[N] array of country labels, aligned to bundle node order. Same derivation as
    train.joint.per_cell_country_weight, so an episode's 'population' means what the loss weighting
    already means by that word."""
    g, ids = b.group_of(), b.meta["node_ids"]
    return np.array([g[i] for i in ids])


def required_lag():
    """Smallest separation IN ORIGIN VALUE between the last support origin and the first query origin
    that makes their TARGET sets disjoint.

    This is the fix for a real defect. `gap` used to be counted in origins while the loss lives on
    targets at t+h for h in HORIZONS, so with gap=4 a support origin's h=10 target and a query
    origin's h=3 target landed on the SAME (node, week) cell: the inner loop was fitting on labels
    the query was then scored against. Disjointness needs

        min(qry) + min(H) > max(sup) + max(H)   <=>   min(qry) - max(sup) > max(H) - min(H)

    so the separation is a property of the horizon set, not a tuneable. It is enforced on origin
    VALUES rather than on index offsets because train origins are not guaranteed contiguous."""
    return max(HORIZONS) - min(HORIZONS)


def episode(origins, countries, rng, n_sup, n_qry, extra_lag=0):
    """One episode: (node subset, support origins, query origins).

    SHAPE OF THE TASK. A country stands in for 'a geography you have not adapted to yet'. Support
    origins precede query origins by at least `required_lag()`, so no target week is shared between
    them, and the query origins taken are the EARLIEST admissible ones -- the task is 'adapt, then
    forecast forward', not 'forecast a distant future'.

    Returns (node_idx, sup_ts, qry_ts) or None if no admissible block exists at this draw. The trunk
    still runs on the FULL graph; only the loss is restricted to node_idx. Subsetting the sparse
    adjacency instead would change what the encoder sees, making the episode a different model
    rather than a different task.

    KNOWN LIMIT, stated rather than discovered later: support and query share the same node subset,
    so the episode rehearses a temporal shift within a country, NOT adaptation to unseen districts.
    Ebola's actual shape is 18 support districts of 61 with the rest zero-shot. Holding nodes out as
    well would rehearse that, and is an open design decision.
    """
    tr = np.sort(np.asarray(origins))
    lag = required_lag() + extra_lag
    if len(tr) < n_sup + n_qry:
        return None
    hi = len(tr) - n_sup                                  # leave room for at least one query origin
    for _ in range(8):                                    # a few draws, then give up for this step
        i = int(rng.integers(0, hi))
        sup_ts = [int(t) for t in tr[i:i + n_sup]]
        cand = tr[tr > sup_ts[-1] + lag]
        if len(cand) >= n_qry:
            qry_ts = [int(t) for t in cand[:n_qry]]
            c = rng.choice(np.unique(countries))
            return np.flatnonzero(countries == c), sup_ts, qry_ts
    return None


def _targets(d, ts, node_idx, mask, device):
    """(tgt, msk) per origin, restricted to node_idx. Deliberately separated from the trunk forward:
    an episode with no observed support or query cell must be detected and skipped BEFORE paying for
    a 7,165-node forward, and at the module defaults that is the majority of draws."""
    idx = torch.as_tensor(node_idx, dtype=torch.long, device=device)
    out = []
    for t in ts:
        tgt, msk = targets_and_mask(d.ymod, d.Mt, mask, t, device)
        out.append((tgt[idx], msk[idx]))
    return idx, out


def _features(enc, d, A, ts, idx, grad=True):
    """Trunk features per origin, restricted to idx.

    Each origin keeps its own autograd graph, which is this module's memory ceiling: n_sup + n_qry
    live trunk graphs at dengue's 7,165 nodes. Hence the small defaults and --timing.
    ponytail: one origin per forward; batch them only if --timing says this is the bottleneck."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return [enc(window_slice(d.Z, t), A, d.Mt[:, t])[idx] for t in ts]


def _packs(feats, tms):
    return [(h, tgt, msk) for h, (tgt, msk) in zip(feats, tms)]


def _mean_pinball(ad, params, packs):
    """Mean pinball over a list of (h, tgt, msk), evaluated at `params` rather than at ad's own
    weights. functional_call is what makes the inner loop differentiable without mutating ad."""
    losses = [pinball_loss(functional_call(ad, params, (h,)), tgt, msk)
              for h, tgt, msk in packs if msk.sum() > 0]
    if not losses:
        return None
    return sum(losses) / len(losses)


def inner_adapt(ad, params, sup_packs, inner_steps, inner_lr, second_order):
    """ANIL's inner loop: `inner_steps` plain SGD steps on the ADAPTER only, trunk untouched.

    `create_graph=second_order` is the whole difference between exact ANIL and its first-order
    approximation. With it the returned fast weights remain a function of the trunk, so the outer
    gradient flows back through the support path as well as the query path. Without it the support
    path is severed and the trunk only ever hears about the query loss, which is a different
    algorithm; D13 committed to the exact one."""
    fast = dict(params)
    for _ in range(inner_steps):
        L = _mean_pinball(ad, fast, sup_packs)
        if L is None:
            return fast, None
        gs = torch.autograd.grad(L, list(fast.values()), create_graph=second_order)
        fast = {k: v - inner_lr * g for (k, v), g in zip(fast.items(), gs)}
    return fast, L


# --------------------------------------------------------------------------- #
# Meta-training
# --------------------------------------------------------------------------- #
def _val_episodes(d, countries, seed, n, n_sup, n_qry, extra_lag):
    """A FIXED set of validation episodes drawn once from the VAL origins.

    Fixed, because a resampled val set makes the early-stop signal mostly draw noise; val origins,
    because selecting on the training episodes is selecting on what we fit."""
    rng = np.random.default_rng(seed + 99991)
    eps, guard = [], 0
    while len(eps) < n and guard < n * 50:
        guard += 1
        e = episode(d.va, countries, rng, n_sup, n_qry, extra_lag)
        if e is not None:
            eps.append(e)
    return eps


def meta_val(enc, ad, panels, eps, inner_steps, inner_lr, arm, device):
    """Held-out meta-objective: adapt on each val episode's support, score its query. No second-order
    graph and no trunk gradient -- this is a measurement, not a step."""
    was_training = enc.training
    enc.eval(); ad.eval()
    tot, n = 0.0, 0
    for pi, (node_idx, sup_ts, qry_ts) in eps:
        d, A, _cs = panels[pi]
        idx, sup_tm = _targets(d, sup_ts, node_idx, d.mva, device)
        _, qry_tm = _targets(d, qry_ts, node_idx, d.mva, device)
        if not any(m.sum() > 0 for _, m in sup_tm) or not any(m.sum() > 0 for _, m in qry_tm):
            continue
        sup = _packs(_features(enc, d, A, sup_ts, idx, grad=False), sup_tm)
        qry = _packs(_features(enc, d, A, qry_ts, idx, grad=False), qry_tm)
        with torch.enable_grad():
            if arm == "anil":
                fast, _ = inner_adapt(ad, dict(ad.named_parameters()), sup, inner_steps, inner_lr,
                                      second_order=False)
            else:
                fast = dict(ad.named_parameters())
            L = _mean_pinball(ad, fast, qry)
        if L is not None:
            tot += float(L.detach()); n += 1
    if was_training:
        enc.train(); ad.train()
    return (tot / n) if n else float("inf")


def meta_train(seed, device, arm="anil", surface="mlp-256", outer_steps=8000, inner_steps=5,
               inner_lr=1e-2, outer_lr=1e-4, wd=1e-4, n_sup=2, n_qry=2, extra_lag=0,
               second_order=True, warm_start=True, val_every=250, patience=8, n_val_episodes=32,
               log_every=100, verbose=True, _budget_s=None, require_val=True, fold=LEGACY_FOLD):
    """Meta-train the trunk on this fold's episodes. Returns (enc, ad, stats).

    PANELS. The meta-train side of an LDO3 fold is several bundles, so each outer step first draws a
    PANEL and then an episode inside it. Uniform over panels, deliberately: it matches `_fit_trunk`'s
    uniform across-bundle sampler, so the meta arm and the freeze-then-adapt arm it is differenced
    against weight the in-diseases the same way. Cells-proportional sampling would make dengue 98% of
    episodes, which is the failure `train/joint.py` already documented.

    SELECTION. Early-stopped on a fixed held-out meta-objective with a cosine outer schedule, so the
    returned trunk is a best-val checkpoint. This is not optional polish: the baseline trunk this is
    differenced against was itself best-val selected inside `_fit_trunk`, so returning a last iterate
    here would confound the meta-objective with the model-selection rule and reproduce D17's defect
    with the sign flipped.

    ARMS. `arm="anil"` runs the inner loop. `arm="control"` runs the SAME episode stream and the same
    number of outer updates with no inner loop, pooling support and query cells so the two arms see
    identical data -- ordinary ERM fine-tuning, the control that makes a negative ANIL result mean
    anything.

    WARM START, AND WHAT IT COSTS US. By default the trunk starts from the capacity probe's own
    frozen dengue trunk for this seed, so the starting point is held fixed and the delta is
    attributable to the objective. It is also far cheaper than meta-training from scratch. The cost
    is real and belongs in the write-up: a reviewer can say the trunk was already shaped by ordinary
    training and that we therefore measure meta-FINE-TUNING. `--from-scratch` runs the pure version.
    """
    assert arm in ARMS, f"unknown arm {arm}"
    meta_names, _test_names, direction = fold_plan(fold)
    assert not any(x.startswith("ebola") for x in meta_names), \
        "ebola must never enter meta-training (C8)"
    torch.manual_seed(seed); np.random.seed(seed)
    # ONE _prepare per bundle rather than a block-diagonal supergraph: an episode lives inside a
    # single panel, so a shared graph would put other diseases' nodes in the trunk's receptive field
    # during a task. SharedEncoder carries no node dimension, so per-panel graphs cost only the
    # resident tensors.
    panels = []
    for _nm in meta_names:
        _ds, _A = _prepare([_nm], device)
        panels.append((_ds[0], _A, country_index(_ds[0].b)))
    d, A, countries = panels[0]

    if warm_start:
        ck = rpath(warm_ckpt(seed, fold))
        if not ck.exists():
            sys.exit(f"no warm-start trunk at {ck} for fold={fold}. Run that fold's trunk for "
                     f"seed {seed} first, or pass --from-scratch.")
        enc = SharedEncoder().to(device)
        enc.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["encoder"])
        if verbose:
            print(f"[anil] seed {seed} arm={arm} surface={surface}: warm start from {ck}")
    else:
        # trunk_patience=30 rather than _fit_trunk's default 12: D17 showed the cosine schedule is
        # scaled to a budget patience=12 never reaches, and the Ebola prereg overrode it to 30 for
        # the same reason (A5). Running the "pure" arm under a stop rule we have on record as broken
        # would be a defect we already documented.
        if verbose:
            print(f"[anil] seed {seed}: from scratch -- fitting a {meta_names} trunk first (~3 h)")
        enc, _ = _fit_trunk(seed, list(meta_names), device, patience=30, verbose=verbose)

    ad = surface_factory(surface)().to(device)
    for p in enc.parameters():
        p.requires_grad_(True)
    meta_params = list(enc.parameters()) + list(ad.parameters())
    opt = torch.optim.AdamW(meta_params, lr=outer_lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(outer_steps, 1))
    rng = np.random.default_rng(seed)
    # Spread the fixed val set over every panel, so the early-stop signal is the FOLD's
    # meta-objective and not whichever panel happened to sort first.
    val_eps = []
    for _pi, (_d, _A, _cs) in enumerate(panels):
        _k = n_val_episodes // len(panels) + (1 if _pi < n_val_episodes % len(panels) else 0)
        val_eps += [(_pi, e) for e in
                    _val_episodes(_d, _cs, seed + _pi, _k, n_sup, n_qry, extra_lag)]
    if not val_eps:
        sys.exit(f"seed {seed}: could not draw a single validation episode from {meta_names} val "
                 f"origins at n_sup={n_sup} n_qry={n_qry}. Early stopping would be unguarded.")

    enc.train(); ad.train()
    hist, updates, skipped, step, t0 = [], 0, 0, 0, time.time()
    best_val, best_state, bad = float("inf"), None, 0
    for step in range(1, outer_steps + 1):
        d, A, countries = panels[int(rng.integers(0, len(panels)))]
        ep = episode(d.tr, countries, rng, n_sup, n_qry, extra_lag)
        if ep is None:
            skipped += 1
            continue
        node_idx, sup_ts, qry_ts = ep
        # Masks first, forwards second. An episode with no observed cell is the majority case and
        # must cost nothing; the old order paid two full-graph forwards before discovering it.
        idx, sup_tm = _targets(d, sup_ts, node_idx, d.mtr, device)
        _, qry_tm = _targets(d, qry_ts, node_idx, d.mtr, device)
        if not any(m.sum() > 0 for _, m in sup_tm) or not any(m.sum() > 0 for _, m in qry_tm):
            skipped += 1
            continue

        sup = _packs(_features(enc, d, A, sup_ts, idx), sup_tm)
        qry = _packs(_features(enc, d, A, qry_ts, idx), qry_tm)
        if arm == "anil":
            fast, Ls = inner_adapt(ad, dict(ad.named_parameters()), sup, inner_steps, inner_lr,
                                   second_order)
            if Ls is None:
                skipped += 1
                continue
            L = _mean_pinball(ad, fast, qry)
        else:
            # Same episodes, same updates, no inner loop: support and query cells pooled so the two
            # arms see identical data and differ only in whether adaptation happened.
            L = _mean_pinball(ad, dict(ad.named_parameters()), sup + qry)
        if L is None:
            skipped += 1
            continue

        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        opt.step(); sched.step()
        updates += 1
        hist.append(float(L.detach()))

        if updates % val_every == 0:
            v = meta_val(enc, ad, panels, val_eps, inner_steps, inner_lr, arm, device)
            if v < best_val - 1e-5:
                best_val, bad = v, 0
                best_state = ({k: t.detach().clone() for k, t in enc.state_dict().items()},
                              {k: t.detach().clone() for k, t in ad.state_dict().items()})
            else:
                bad += 1
            if verbose:
                print(f"    anil({seed},{arm}) upd{updates:6d} val={v:.4f} best={best_val:.4f} "
                      f"({(time.time() - t0) / 60:.1f} min)")
            if bad >= patience:
                if verbose:
                    print(f"    anil({seed},{arm}) early stop at update {updates}")
                break
        if verbose and updates and updates % log_every == 0 and updates % val_every:
            print(f"    anil({seed},{arm}) upd{updates:6d} train={np.mean(hist[-log_every:]):.4f} "
                  f"({(time.time() - t0) / 60:.1f} min, {skipped} skipped)")
        if _budget_s is not None and time.time() - t0 > _budget_s:
            if verbose:
                print(f"    anil({seed},{arm}) wall-clock budget reached at update {updates}")
            break

    if not updates:
        sys.exit(f"seed {seed}: {step} episodes drawn, zero usable. Check --n-sup/--n-qry against "
                 f"the mask density of {meta_names} before rerunning.")
    if best_state is None:
        # `require_val=False` is for the TIMING PROBE only, which prices an outer update and throws
        # the model away. Every scoring path leaves this True, so a real arm can still never return
        # a last iterate -- the defect this arm exists to avoid.
        if require_val:
            sys.exit(f"seed {seed}: training ended before the first validation check "
                     f"(updates={updates}, val_every={val_every}). The trunk would be a last "
                     f"iterate, which is the defect this arm exists to avoid. Lower --val-every or "
                     f"raise --outer-steps.")
        if verbose:
            print(f"    [probe] no val check fired in {updates} updates; returning the last iterate. "
                  f"Timing only -- this model is not scored.")
    else:
        enc.load_state_dict(best_state[0]); ad.load_state_dict(best_state[1])

    stats = dict(seed=seed, arm=arm, surface=surface, fold=fold, direction=direction,
                 meta_names=list(meta_names), n_panels=len(panels),
                 episodes_drawn=step, outer_updates=updates,
                 skipped=skipped, best_val=best_val, minutes=(time.time() - t0) / 60,
                 train_first100=float(np.mean(hist[:100])), train_last100=float(np.mean(hist[-100:])),
                 second_order=second_order, warm_start=warm_start, inner_steps=inner_steps,
                 inner_lr=inner_lr, outer_lr=outer_lr, n_sup=n_sup, n_qry=n_qry,
                 required_lag=required_lag(), extra_lag=extra_lag)
    return enc, ad, stats


# --------------------------------------------------------------------------- #
# Meta-test -- deliberately the capacity probe's stage 2, unchanged
# --------------------------------------------------------------------------- #
def meta_test(enc, seed, device, surface, arm, verbose=True, archive=True, fold=LEGACY_FOLD):
    """Freeze the meta-trunk, fit ONE fresh adapter on the fold's HELD-OUT panels, score.

    Calls `_fit_shared_adapter` with the SAME factory, epochs, patience, lr, wd, sampler and
    validation objective the capacity probe used, so the only thing differing from that run's row for
    this surface is the trunk's provenance -- which is what makes the delta readable, and why the
    adapter is FRESH rather than warm-started from the meta-learned initialisation. The meta-learned
    init is saved to the checkpoint, so evaluating that variant later costs a scoring pass, not a
    re-run (D14's lesson about artifacts that cannot be reconstructed).

    `manual_seed` immediately before the fit: `_fit_shared_adapter` does not seed, so without this the
    adapter's initialisation is drawn from wherever the process RNG happens to be -- which differs
    between the train path and the checkpoint-reuse path, and made resume non-reproducible.
    """
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    torch.manual_seed(seed); np.random.seed(seed)
    _meta_names, test_names, direction = fold_plan(fold)
    ad, ds = _fit_shared_adapter(enc, list(test_names), seed, device, verbose=verbose,
                                 adapter_factory=surface_factory(surface))
    rmse = {}
    for d in ds:
        quant = {}
        recs, pn, po, _ = _test_dataset(enc, ad, d, seed, f"anil-{arm}:{surface}",
                                        dict(training_regime=f"anil_{arm}", surface=surface,
                                             direction=direction, fold=fold, seed=seed),
                                        quant_out=quant)
        rmse.update(_records_rmse(recs))
        if archive:
            # D14: every metric except rmse used to be computed and thrown away here. Records carry
            # mae/pcc/nrmse/coverage, and the quantiles are the only material WIS/CRPS/PIT can ever
            # be recomputed from. Seconds to write, a whole rerun to reconstruct.
            write_records(recs, artifact(surface, arm, d.name, seed, ".json", fold))
            write_per_node(pn, artifact(surface, arm, d.name, seed, "__pernode.npz", fold))
            write_per_origin(po, artifact(surface, arm, d.name, seed, "__perorigin.npz", fold))
            write_quantiles(quant, d.te, artifact(surface, arm, d.name, seed, "__quantiles.npz", fold))
    assert rmse, "meta_test produced no rmse cells -- the record schema has moved"
    return rmse


def _records_rmse(recs):
    """{'dataset|hH': country_macro rmse} out of a record list.

    MUST stay identical to `diagnostics.capacity_probe.rmse_of`, because the delta this module
    reports is against numbers that function produced. The field is `country_macro`, NOT `value`, and
    the self-check asserts the two agree so drift is caught in seconds rather than after a run.
    float(): score_predictions may hand back numpy scalars and json.dumps refuses them, which would
    raise at the incremental write, after the seed's training is already spent."""
    return {f"{r['dataset']}|h{r['horizon']}": float(r["country_macro"])
            for r in recs if r["metric"] == "rmse"}


def run_seed(seed, device, args):
    """Meta-train (or reuse) this seed's trunk, then score it.

    The checkpoint is written BEFORE scoring and reused on a later invocation, so a crash in the
    scoring pass costs a scoring pass rather than the hours that preceded it. The seed's row enters
    the JSON only after scoring returns, so resume still treats a killed seed as unfinished."""
    ck = rpath(meta_ckpt(seed, args.surface, args.arm, args.fold))
    if ck.exists() and not args.refit:
        print(f"[anil] seed {seed}: reusing meta-trunk {ck} (--refit to retrain)")
        blob = torch.load(ck, map_location=device, weights_only=False)
        enc = SharedEncoder().to(device)
        enc.load_state_dict(blob["encoder"])
        stats = dict(blob.get("meta", {}))
        stats.update(seed=seed, arm=args.arm, surface=args.surface, fold=args.fold,
                     reused_checkpoint=True)
    else:
        try:
            enc, ad, stats = meta_train(
                seed, device, arm=args.arm, surface=args.surface, outer_steps=args.outer_steps,
                inner_steps=args.inner_steps, inner_lr=args.inner_lr, outer_lr=args.outer_lr,
                n_sup=args.n_sup, n_qry=args.n_qry, extra_lag=args.extra_lag,
                second_order=not args.first_order, warm_start=not args.from_scratch,
                val_every=args.val_every, patience=args.patience, verbose=args.verbose,
                _budget_s=(args.max_hours * 3600 if args.max_hours else None), fold=args.fold)
        except torch.cuda.OutOfMemoryError:
            sys.exit(f"seed {seed}: CUDA OOM during meta-training. This module holds "
                     f"n_sup + n_qry = {args.n_sup + args.n_qry} trunk graphs live at once on "
                     f"this fold's full node set. Lower --n-sup/--n-qry, or --first-order "
                     f"(a stated deviation from D13), before rerunning.")
        # `extra` and `stats` both carry `seed`; dict(seed=..., **stats) is a TypeError, not a merge.
        write_checkpoint(enc, ad, meta_ckpt(seed, args.surface, args.arm, args.fold),
                         extra=dict(stats, regime="anil"))
    stats["rmse"] = meta_test(enc, seed, device, args.surface, args.arm, verbose=args.verbose,
                              fold=args.fold)
    return stats


# --------------------------------------------------------------------------- #
# Preflight, resume, and the comparison the run exists to produce
# --------------------------------------------------------------------------- #
def out_json(surface, arm, fold=LEGACY_FOLD):
    """Legacy fold keeps its existing filename, so resume still reads the rows already on disk."""
    tag = "" if fold == LEGACY_FOLD else f"{fold_tag(fold)}_"
    return RESULTS / "misc" / f"anil_{tag}{surface}_{arm}.json"


def preflight(seeds, args, device):
    """Everything knowable to be wrong BEFORE the first hour is spent, checked for EVERY seed up
    front. The failure this prevents is specific: the warm-start checkpoint used to be looked up
    inside `meta_train`, so a missing seed-82 trunk surfaced four seeds into an unattended run."""
    problems = []
    if not str(device).startswith("cuda") and not args.allow_cpu:      # DEVICE is a str here
        problems.append(f"DEVICE is {device}, not cuda. This is a multi-hour second-order run and on "
                        f"CPU it will not finish. --allow-cpu to override.")
    if not args.from_scratch:
        for s in seeds:
            if not rpath(warm_ckpt(s, args.fold)).exists():
                problems.append(f"seed {s}: no warm-start trunk at {rpath(warm_ckpt(s, args.fold))} "
                                f"for fold={args.fold} (run that fold trunk, or --from-scratch)")
    meta_names, test_names, _direction = fold_plan(args.fold)
    # EVERY meta-train panel is sized, not just the first. A fold whose second panel cannot yield an
    # episode would otherwise train on a silently smaller set than the one named on the table.
    for nm in meta_names:
        try:
            b = bundles.load(nm)
            need = args.n_sup + args.n_qry
            for phase in ("train", "val"):
                k = len(b.origins(phase=phase))
                if k < need:
                    problems.append(f"{nm} has {k} {phase} origins, an episode needs {need}")
            # An episode must be drawable in practice, not just in principle: the lag is on origin
            # VALUES and val origins are a short contiguous block, so this can fail even when the
            # count passes.
            cs = country_index(b)
            if episode(b.origins(phase="val"), cs, np.random.default_rng(0),
                       args.n_sup, args.n_qry, args.extra_lag) is None:
                problems.append(f"{nm}: no admissible VAL episode at n_sup={args.n_sup} "
                                f"n_qry={args.n_qry} lag={required_lag() + args.extra_lag}; early "
                                f"stopping would be unguarded. Lower --n-sup/--n-qry or --extra-lag.")
        except Exception as e:
            problems.append(f"could not load {nm} to size an episode: {e!r}")
    if args.fold != LEGACY_FOLD:
        for nm in test_names:
            if not rpath(f"encoder_ldo3__{nm}__seed{args.seeds[0]}.json").exists():
                print(f"[preflight] WARNING: no LDO3 record for {nm} seed {args.seeds[0]}; the run "
                      f"will produce absolute RMSE but no seed-paired delta.")
    if args.fold == LEGACY_FOLD and not CAP_JSON.exists():
        print(f"[preflight] WARNING: {CAP_JSON} missing -- the run will produce absolute RMSE but "
              f"no seed-paired delta against freeze-then-adapt.")
    if problems:
        sys.exit("preflight failed:\n  - " + "\n  - ".join(problems))
    print(f"[preflight] ok: fold={args.fold} meta-train={meta_names} -> meta-test={test_names}, "
          f"{len(seeds)} seeds, device {device}, arm={args.arm}, surface={args.surface}, "
          f"target lag {required_lag() + args.extra_lag} origins, "
          f"second_order={not args.first_order}, warm_start={not args.from_scratch}")


def load_done(path):
    """Resume: seeds already scored. An overnight job that dies on seed 62 must not redo 42 and 52,
    and a completed seed is proven by its row being IN the file -- the row is written only after
    meta_test returns, so a killed seed leaves nothing and is retried. Same rule and the same
    reasoning as recover_single_epochs' [run-complete] marker."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"[resume] {path} unreadable; starting fresh")
        return []


def cap_reference(surface):
    """{seed: {cell: rmse}} for the capacity probe's cross-disease rows at `surface` -- the
    freeze-then-adapt comparator on the SAME surface, direction and fitting protocol."""
    if not CAP_JSON.exists():
        return {}
    want = SURFACES[surface][1]
    blob = json.loads(CAP_JSON.read_text(encoding="utf-8"))
    out = {}
    for per_seed in blob.get("cross_disease", []):
        for row in per_seed:
            if row.get("label") == want:
                out[row["seed"]] = row["rmse"]
    return out


def ldo3_reference(fold, seeds=SEEDS):
    """{seed: {cell: rmse}} from the LDO3 freeze-then-adapt records for this fold.

    This is the comparator D2 actually asks for on a disease-out fold: the SAME trunk this arm warm
    starts from, frozen, with one shared adapter fitted on the held-out disease full train fold and
    scored by the same score.py pass. The capacity probe cannot serve here, because it only ever ran
    the dengue->flu direction, so its rows carry no cells for a dengue or covid meta-test.

    Missing seeds are simply absent; compare() takes its t critical value from the pair count, so a
    partial reference widens the interval rather than silently narrowing it."""
    _meta, test_names, _dir = fold_plan(fold)
    out = {}
    for sd in seeds:
        cells = {}
        for nm in test_names:
            f = rpath(f"encoder_ldo3__{nm}__seed{sd}.json")
            if not f.exists():
                continue
            try:
                cells.update(_records_rmse(json.loads(f.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError, KeyError):
                continue
        if cells:
            out[sd] = cells
    return out


def reference_for(fold, surface):
    """(reference, label, provenance) for this fold. The legacy fold keeps the capacity probe."""
    if fold == LEGACY_FOLD:
        return (cap_reference(surface), SURFACES[surface][1],
                "freeze-then-adapt, capacity probe, same surface/direction")
    return (ldo3_reference(fold), f"LDO3 freeze-then-adapt ({fold} held out)",
            "freeze-then-adapt, LDO3 fold of the same name, same seeds and scorer")


def compare(rows, ref, ref_label,
            provenance="freeze-then-adapt, capacity probe, same surface/direction"):
    """Seed-paired delta vs a freeze-then-adapt reference. Positive = the ANIL/control arm is better.

    Paired WITHIN a seed by construction: both arms descend from that seed's own warm-start trunk, so
    the pairing removes trunk-to-trunk variance exactly as it does in the capacity probe. Two-sided
    95% t at THAT CELL's pair count, and a cell counts only if its interval excludes zero (D3's
    noise-floor rule).

    The t critical value is taken from `len(v)`, the number of seeds this cell actually paired on --
    not from the number of seed rows. Those differ whenever a seed is missing from the reference or
    from the run, and using the row count there gives, for example, t=2.776 where 4.303 is correct: an
    interval 35 per cent too narrow, i.e. cells printed as significant that are inside the noise
    floor on exactly the rows where seeds are missing.
    """
    if not ref:
        print(f"\nno reference for '{ref_label}' on disk; absolute RMSE only.")
        return None
    cells = {}
    for r in rows:
        base = ref.get(r["seed"])
        if not base:
            continue
        for cell, v in r.get("rmse", {}).items():
            if cell in base and base[cell]:
                cells.setdefault(cell, []).append((base[cell] - v) / abs(base[cell]) * 100.0)
    if not cells:
        print(f"\nno seed paired against '{ref_label}'.")
        return None
    # D3: the reference, the formula and the units go ON the table, not in someone's memory.
    print(f"\nreference: {ref_label} ({provenance})")
    print("delta% = (reference_rmse - arm_rmse) / |reference_rmse| x 100; positive = arm better; "
          "units = per cent of reference RMSE (country-macro, count space)")
    print(f"\n{'cell':<28} {'n':>2} {'mean d%':>9} {'sd':>7} {'95% CI':>20}   verdict")
    better = worse = 0
    summary = {}
    for cell in sorted(cells):
        v = np.array(cells[cell], dtype=float)
        k = len(v)
        m = float(v.mean())
        sd = float(v.std(ddof=1)) if k > 1 else float("nan")
        t = T_CRIT.get(k)
        half = (t * sd / math.sqrt(k)) if (t and k > 1) else float("inf")
        lo, hi = m - half, m + half
        clears = math.isfinite(half) and (lo > 0 or hi < 0)
        better += clears and m > 0
        worse += clears and m < 0
        ci = f"[{lo:+.1f}, {hi:+.1f}]" if math.isfinite(half) else "n too small"
        print(f"{cell:<28} {k:>2} {m:>+9.1f} {sd:>7.1f} {ci:>20}   "
              f"{'yes' if clears else 'within noise'}")
        summary[cell] = dict(n=k, mean=m, sd=sd, lo=lo, hi=hi, clears_zero=bool(clears),
                             reference=ref_label)
    print(f"\n{better} of {len(cells)} cells significantly better than {ref_label}, "
          f"{worse} significantly worse.")
    return summary


def report(args):
    """Both references, always. mlp-256 answers 'did meta-training beat freeze-then-adapt on the same
    surface'; affine answers D2's actual question, 'does meta-learning beat the linear probe', and it
    is the pre-registered Ebola surface. Both sit in the same JSON, so reporting one is a choice, not
    a constraint."""
    path = out_json(args.surface, args.arm, args.fold)
    rows = load_done(path)
    if not rows:
        print(f"nothing scored yet in {path}")
        return {}
    print(f"\n=== fold={args.fold} arm={args.arm} surface={args.surface}, {len(rows)} seeds ===")
    out = {}
    if args.fold == LEGACY_FOLD:
        for ref_name in dict.fromkeys([args.surface, "affine"]):
            sm = compare(rows, cap_reference(ref_name), SURFACES[ref_name][1])
            if sm:
                out[ref_name] = sm
    else:
        ref, label, prov = reference_for(args.fold, args.surface)
        sm = compare(rows, ref, label, prov)
        if sm:
            out[args.fold] = sm
    if out:
        p = path.with_name(path.stem + "_summary.json")
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {p}")
    return out


def timing_probe(device, args, steps=120):
    """Price one outer UPDATE before booking anything. The comparator is the measured 3 h trunk at
    91,000 steps (D17) = ~0.119 s/step. Uses meta_train's own clock, which starts after setup, rather
    than the wall clock around it -- `_prepare` alone is ~2 s and would inflate a short estimate.

    `--outer-steps` counts EPISODES, and a large fraction are skipped (mask density), so the update
    count is well below the episode count. The first version drew 40 episodes with val_every=20, got
    13 usable updates, never reached a validation check, and died on the last-iterate guard -- which
    aborted a whole night's queue before anything was booked. Now: enough episodes that a check
    reliably fires, and `require_val=False` so a pricing pass cannot be killed by that guard even at
    a pathological skip rate. The model this returns is discarded."""
    seed = args.seeds[0]
    _, _, st = meta_train(seed, device, arm=args.arm, surface=args.surface, outer_steps=steps,
                          inner_steps=args.inner_steps, inner_lr=args.inner_lr,
                          outer_lr=args.outer_lr, n_sup=args.n_sup, n_qry=args.n_qry,
                          extra_lag=args.extra_lag, second_order=not args.first_order,
                          warm_start=not args.from_scratch, val_every=max(steps // 6, 1),
                          patience=99, log_every=10, verbose=args.verbose, require_val=False,
                          fold=args.fold)
    upd = max(st["outer_updates"], 1)
    per_update = st["minutes"] * 60 / upd
    trunk_per_step = 3 * 3600 / 91000
    print("\n" + "=" * 78)
    print(f"episodes drawn  : {st['episodes_drawn']}  ->  {upd} usable updates "
          f"({st['skipped']} skipped, {100 * st['skipped'] / max(st['episodes_drawn'], 1):.0f}%)")
    print(f"outer update    : {per_update:.3f} s   ({per_update / trunk_per_step:.1f}x a trunk step)")
    print(f"NOTE --outer-steps counts EPISODES DRAWN, not updates. At the observed skip rate, "
          f"N steps yields ~{upd / max(st['episodes_drawn'], 1):.2f}N updates.")
    for n in (2000, 4000, 8000):
        h = per_update * n * (upd / max(st['episodes_drawn'], 1)) / 3600
        print(f"  {n:5d} steps  : {h:5.2f} h/seed   {h * len(args.seeds):6.2f} h for "
              f"{len(args.seeds)} seeds")
    print(f"arm={args.arm} surface={args.surface} second_order={not args.first_order} "
          f"warm_start={not args.from_scratch}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--arm", choices=ARMS, default="anil",
                    help="anil = meta-training; control = same episodes/updates, no inner loop")
    ap.add_argument("--surface", choices=sorted(SURFACES), default="mlp-256",
                    help="mlp-256 = D19's strongest rung; affine = the pre-registered Ebola surface")
    ap.add_argument("--fold", choices=FOLDS, default=LEGACY_FOLD,
                    help="dengue2flu = the reported single-direction run; dengue/influenza/covid = "
                         "the LDO3 disease-out folds, meta-trained on that fold in-diseases")
    ap.add_argument("--outer-steps", type=int, default=8000, help="EPISODES DRAWN, not updates")
    ap.add_argument("--inner-steps", type=int, default=5)
    ap.add_argument("--inner-lr", type=float, default=1e-2)
    ap.add_argument("--outer-lr", type=float, default=1e-4)
    ap.add_argument("--n-sup", type=int, default=2)
    ap.add_argument("--n-qry", type=int, default=2)
    ap.add_argument("--extra-lag", type=int, default=0,
                    help=f"origins ON TOP of the {required_lag()} required to make support and query "
                         f"target weeks disjoint")
    ap.add_argument("--val-every", type=int, default=250, help="outer UPDATES between val checks")
    ap.add_argument("--patience", type=int, default=8, help="val checks without improvement")
    ap.add_argument("--first-order", action="store_true",
                    help="drop the second-order term (deviation from D13; state it if used)")
    ap.add_argument("--from-scratch", action="store_true",
                    help="meta-train a trunk from init instead of warm-starting the capacity trunk")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock cap on EACH seed's meta-training; it still gets scored")
    ap.add_argument("--refit", action="store_true",
                    help="redo a seed even if its checkpoint and scored row already exist")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="check every prerequisite for every seed and exit; starts nothing")
    ap.add_argument("--timing", action="store_true", help="price one outer update and exit")
    ap.add_argument("--report", action="store_true",
                    help="recompute the comparison from the existing JSON and exit; no training")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    a = ap.parse_args()

    if a.selfcheck:
        _selfcheck(); return
    if a.report:
        report(a); return
    preflight(a.seeds, a, DEVICE)
    if a.preflight:
        return
    if a.timing:
        timing_probe(DEVICE, a); return

    path = out_json(a.surface, a.arm, a.fold)
    rows = load_done(path)
    # --refit has to drop the seed from `done` too. It used to be consulted only inside run_seed,
    # which the resume skip never reached, so the flag silently did nothing.
    if a.refit:
        rows = [r for r in rows if r["seed"] not in set(a.seeds)]
    done = {r["seed"] for r in rows}
    if done:
        print(f"[resume] already scored: {sorted(done)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        if s in done:
            continue
        rows.append(run_seed(s, DEVICE, a))
        # Written after EVERY seed, not once at the end: a crash on the last seed used to cost every
        # earlier seed's result, and those are hours each.
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[seed {s}] scored; {path} updated ({len(rows)} seeds)")

    report(a)
    print(f"\nwrote {path}")
    print("ABLATION ONLY. This is an ablation of the adaptation procedure, not a headline, and that "
          "does not change if the sign is positive.")
    if a.fold == LEGACY_FOLD:
        one = ("(1) D13's objection -- dengue-only episodes vary population and origin, not disease")
    else:
        mtr, mte, _d = fold_plan(a.fold)
        one = (f"(1) episodes here DO vary disease -- meta-train {mtr} over {len(mtr)} panels, "
               f"held out {mte} -- which answers D13's objection for this fold but NOT for the "
               f"dengue2flu run, so do not read this back onto that table")
    print(f"Report with ALL of: {one}; (2) warm start, so this is meta-FINE-TUNING; (3) the "
          "meta-test fits a fresh adapter on FULL train folds, so nothing here measures few-shot; "
          "(4) the surface was selected on h15, where the Ebola primary arm has zero adaptation "
          "pairs.")


# --------------------------------------------------------------------------- #
def _selfcheck():
    """Each check is paired with a case that must come out the other way. No GPU, seconds."""
    torch.manual_seed(0)

    # 1. the surfaces have not drifted from the ones D19 measured -- shapes, not just a param count
    from diagnostics.capacity_probe import SURFACES as CAP_SURFACES, rmse_of
    cap = {lbl: f for lbl, f in CAP_SURFACES}
    for name, (_, cap_label) in SURFACES.items():
        mine, theirs = surface_factory(name)(), cap[cap_label]()
        assert {k: tuple(v.shape) for k, v in mine.state_dict().items()} == \
               {k: tuple(v.shape) for k, v in theirs.state_dict().items()}, \
            f"surface '{name}' has drifted from the capacity probe's '{cap_label}'"
    assert sum(p.numel() for p in surface_factory("mlp-256")().parameters()) == 21_780
    assert sum(p.numel() for p in surface_factory("affine")().parameters()) == 1_428

    # 2. episodes. The property that matters is TARGET-week disjointness, not origin ordering -- the
    #    earlier version checked only origins and passed while support labels at t+15 collided with
    #    query labels at t+3. The control is the old rule: a lag of 4 must NOT be disjoint.
    rng = np.random.default_rng(0)
    tr = np.arange(50, 200)
    cs = np.array(["a"] * 6 + ["b"] * 4)
    for _ in range(300):
        e = episode(tr, cs, rng, n_sup=2, n_qry=2)
        assert e is not None
        idx, sup, qry = e
        st = {t + h for t in sup for h in HORIZONS}
        qt = {t + h for t in qry for h in HORIZONS}
        assert not (st & qt), f"support and query share target weeks {sorted(st & qt)}"
        assert max(sup) < min(qry) and not (set(sup) & set(qry))
        assert len(idx) in (6, 4) and len(set(cs[idx])) == 1, "episode must be one country"
    assert required_lag() == max(HORIZONS) - min(HORIZONS) == 12
    sup_, qry_ = [171, 172], [177, 178]          # the old default (gap=4), the documented failure
    assert {t + h for t in sup_ for h in HORIZONS} & {t + h for t in qry_ for h in HORIZONS}, \
        "the pre-fix case must still be detectably overlapping, or this check proves nothing"
    assert episode(np.arange(3), cs, rng, n_sup=2, n_qry=2) is None, "short panel must refuse"

    # 3. the node restriction, tested where it actually happens (_targets), not by hand-slicing.
    #    A tautological version of this check passed while never calling the restricting code at all.
    from types import SimpleNamespace
    N, T = 10, 40
    d_ = SimpleNamespace(ymod=torch.arange(N * T, dtype=torch.float32).view(N, T),
                         Mt=torch.ones(N, T))
    keep = np.array([1, 3, 5])
    idx, tms = _targets(d_, [20], keep, torch.ones(N, T), "cpu")
    assert idx.tolist() == keep.tolist()
    assert tms[0][0].shape[0] == len(keep), "targets not restricted to the episode's nodes"
    exp = torch.stack([d_.ymod[keep, 20 + h] for h in HORIZONS], dim=1)
    assert torch.allclose(tms[0][0], exp), "restricted targets are the WRONG nodes (index misalign)"

    # 4. second order. Exact and first-order must DIFFER -- asserting only 'non-zero' would pass
    #    either way, since the query path reaches the trunk regardless.
    def meta_grad(second_order):
        torch.manual_seed(1)
        ad_ = surface_factory("mlp-256")()
        theta = torch.ones(1, requires_grad=True)
        hs, hq = torch.randn(5, D_HIDDEN), torch.randn(5, D_HIDDEN)
        m = torch.ones(5, len(HORIZONS))
        fast, _ = inner_adapt(ad_, dict(ad_.named_parameters()),
                              [(hs * theta, torch.randn(5, len(HORIZONS)), m)], 2, 1e-2, second_order)
        Lq = _mean_pinball(ad_, fast, [(hq, torch.randn(5, len(HORIZONS)), m)])
        # allow_unused: under first order theta is ABSENT from the graph, not zero-gradient, and
        # autograd raises rather than returning 0. That absence is exactly what is being asserted.
        g = torch.autograd.grad(Lq, theta, allow_unused=True)[0]
        return 0.0 if g is None else float(g)

    g2, g1 = meta_grad(True), meta_grad(False)
    assert abs(g1) < 1e-12 and abs(g2) > 1e-9, f"second-order term dead: {g2} vs {g1}"

    # 5. the inner loop must not mutate the real adapter, and the outer optimiser must contain the
    #    TRUNK. A meta_train that optimised only the adapter passed every earlier check in this file.
    ad_ = surface_factory("affine")()
    before = {k: v.detach().clone() for k, v in ad_.named_parameters()}
    inner_adapt(ad_, dict(ad_.named_parameters()),
                [(torch.randn(4, D_HIDDEN), torch.randn(4, len(HORIZONS)),
                  torch.ones(4, len(HORIZONS)))], 3, 1e-2, True)
    assert all(torch.equal(before[k], v) for k, v in ad_.named_parameters()), \
        "inner loop mutated the adapter's real weights; fast weights are not functional"
    src = __import__("inspect").getsource(meta_train)
    assert "list(enc.parameters()) + list(ad.parameters())" in src, \
        "the outer optimiser must contain the trunk, or this is not meta-learning"

    # 6. record schema. The field is `country_macro`; reading `value` raises only at the END of a run.
    fake = [dict(dataset="influenza_japan", horizon=3, metric="rmse", country_macro=1.0, value=99.0),
            dict(dataset="influenza_japan", horizon=5, metric="mae", country_macro=7.0, value=99.0)]
    assert _records_rmse(fake) == {"influenza_japan|h3": 1.0}
    assert _records_rmse(fake) == {f"{k[0]}|h{k[1]}": v for k, v in rmse_of(fake).items()}, \
        "_records_rmse has drifted from capacity_probe.rmse_of"

    # 7. compare(): the t value must follow the CELL's pair count, not the row count. The control is
    #    a cell paired on all 5 seeds; both live in one call, so a row-count t cannot satisfy both.
    import io, contextlib
    seeds_ = (42, 52, 62, 72, 82)
    ref_ = {s: ({"d|h3": 100.0} if s in (42, 52, 62) else {}) for s in seeds_}
    for s in seeds_:
        ref_[s]["d|h5"] = 100.0
    rows_ = [dict(seed=s, rmse={"d|h3": 90.0 + (i % 2) * 6, "d|h5": 90.0 if i % 2 else 130.0})
             for i, s in enumerate(seeds_)]
    with contextlib.redirect_stdout(io.StringIO()):
        summ = compare(rows_, ref_, "ref")
    assert summ["d|h3"]["n"] == 3 and summ["d|h5"]["n"] == 5, "pair counts wrong"
    v3 = np.array([10.0, 4.0, 10.0])
    exp_half = T_CRIT[3] * v3.std(ddof=1) / math.sqrt(3)
    assert abs((summ["d|h3"]["hi"] - summ["d|h3"]["mean"]) - exp_half) < 1e-9, \
        "t taken from the row count, not the cell's pair count -- intervals too narrow"
    assert not summ["d|h5"]["clears_zero"], "a sign-flipping cell must read within noise"

    # 8. cap_reference actually parses the real file, if it is there. A silent {} degrades the whole
    #    run to 'absolute RMSE only', which is the quietest possible failure.
    if CAP_JSON.exists():
        for name in SURFACES:
            r = cap_reference(name)
            assert len(r) == 5 and all(len(c) == 12 for c in r.values()), \
                f"cap_reference('{name}') parsed {len(r)} seeds; expected 5 x 12 cells"
        assert cap_reference("mlp-256")[42] != cap_reference("affine")[42], \
            "both surfaces returned the same reference rows -- the label mapping is wrong"

    # 9. artifact names route to lodo/ and collide with nothing already on disk
    for suf in (".json", "__pernode.npz", "__perorigin.npz", "__quantiles.npz"):
        p = rpath(artifact("mlp-256", "anil", "influenza_japan", 42, suf))
        assert p.parent.name == "lodo", f"{p} did not route to lodo/"
        assert not p.exists(), f"{p} already exists -- this run would overwrite it"
    assert not rpath(meta_ckpt(42, "mlp-256", "anil")).exists()

    # 10. FOLDS. Two things must hold at once: the legacy fold reproduces its existing names byte
    #     for byte (or resume rereads nothing and the reported run is orphaned), and every new fold
    #     writes somewhere else (or fold="influenza", which meta-tests on the same three panels,
    #     silently overwrites results that are already in a stakeholder brief).
    assert artifact("affine", "anil", "influenza_japan", 42, ".json") == \
        "encoder_ldo__anil-affine-anil__influenza_japan__seed42.json", \
        "legacy artifact name changed; the reported run would be orphaned"
    assert out_json("affine", "anil").name == "anil_affine_anil.json", \
        "legacy out_json changed; resume would not see the 5 seeds already scored"
    assert warm_ckpt(42) == "encoder_ldo__dengue2flu-cap__seed42__ckpt.pt"
    assert meta_ckpt(42, "affine", "anil") == \
        "encoder_ldo__dengue2flu-anil-affine-anil__seed42__ckpt.pt"
    for f_ in FOLDS:
        if f_ == LEGACY_FOLD:
            continue
        assert artifact("affine", "anil", "influenza_japan", 42, ".json", f_) != \
            artifact("affine", "anil", "influenza_japan", 42, ".json"), \
            f"fold {f_} shares the legacy artifact name and would overwrite it"
        assert out_json("affine", "anil", f_) != out_json("affine", "anil"), \
            f"fold {f_} shares the legacy out_json and would overwrite it"
        assert warm_ckpt(42, f_) == f"encoder_ldo3__{f_}__seed42__ckpt.pt"
    #     fold_plan must agree with the transfer table it borrows its folds from, and no fold may
    #     ever put ebola on either side (C8).
    from train.lodo import DISEASES as _DIS
    for f_ in FOLDS[1:]:
        mtr, mte, _d = fold_plan(f_)
        assert mte == list(_DIS[f_]), f"fold {f_} meta-tests on {mte}, not {list(_DIS[f_])}"
        assert set(mtr) == {x for k, v in _DIS.items() if k != f_ for x in v}, \
            f"fold {f_} meta-train side disagrees with lodo.DISEASES"
        assert not set(mtr) & set(mte), f"fold {f_} has a bundle on both sides"
    for f_ in FOLDS:
        mtr, mte, _d = fold_plan(f_)
        assert not any(x.startswith("ebola") for x in mtr + mte), "ebola entered a fold (C8)"
    assert fold_plan("dengue")[0] and len(fold_plan("dengue")[0]) == 4, \
        "the dengue fold must meta-train on 4 bundles (3 flu + covid)"
    #     the multi-panel path is what makes those 4 bundles reachable: assert it is actually wired,
    #     because a meta_train that silently kept ds[0] would train on one panel and still finish.
    src_mt = __import__("inspect").getsource(meta_train)
    assert "panels[int(rng.integers(0, len(panels)))]" in src_mt, \
        "meta_train does not draw a panel per outer step; a multi-bundle fold would train on one"
    assert "for _nm in meta_names:" in src_mt, "meta_train does not prepare every meta-train bundle"
    assert "meta_val(enc, ad, panels," in src_mt, "meta_val is not receiving the panel list"
    try:
        fold_plan("nope")
    except AssertionError:
        pass
    else:
        raise AssertionError("fold_plan accepted an unknown fold")

    print(f"ok  surfaces match the capacity probe by state-dict shape (mlp-256 21,780 / affine 1,428)")
    print(f"ok  episodes: support/query TARGET weeks disjoint at lag {required_lag()}; "
          f"the old gap=4 case is still detectably overlapping")
    print("ok  node restriction tested in _targets, values checked against the right rows")
    print(f"ok  second-order live: support-path meta-grad {g2:.3e} exact vs {g1:.3e} first-order")
    print("ok  inner loop leaves the real adapter untouched; trunk is in the outer optimiser")
    print("ok  rmse read from country_macro, agrees with capacity_probe.rmse_of")
    print("ok  compare(): t follows each cell's pair count (n=3 -> 4.303), sign-flip reads as noise")
    print("ok  cap_reference parses both surfaces; artifact names route to lodo/ and collide with nothing")
    print(f"ok  folds {FOLDS}: legacy names byte-identical, every LDO3 fold writes elsewhere, "
          f"fold plans agree with lodo.DISEASES, no ebola on either side, panel draw wired")


if __name__ == "__main__":
    main()

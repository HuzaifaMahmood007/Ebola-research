"""capacity_probe.py -- OVERNIGHT RUN. Is the cross-disease deficit caused by the adaptation surface,
or by the representation?

WHY THIS IS THE HIGHEST-VALUE RUN AVAILABLE. The whole meta-learning proposal rests on an untested
premise. Under the three-disease leave-one-disease-out fold at five seeds, cross-disease transfer is
significantly negative in 25 of 36 cells and zero-shot fails in all 36 (the "12 of 16" this note
originally quoted was the earlier two-disease fold). The ANIL argument is that shaping the trunk to
be "repairable by a small support-set update" fixes this. But ANIL does not add capacity to the adaptation surface -- it changes what the
trunk is optimised for. So there are two very different worlds:

  * ADAPTER-BOUND. A richer adaptation surface recovers some of the deficit. Then the surface was the
    bottleneck, there is a cheap partial fix available immediately, and there is a real prior that
    optimising the trunk for adaptability will pay. ANIL is worth a week.
  * REPRESENTATION-BOUND. A strictly larger, strictly more expressive surface recovers nothing. Then
    the frozen trunk simply does not carry the information, no read-out can recover it, and ANIL --
    which adds no capacity -- is unlikely to close a 51% gap. That is a decisive negative for one
    night of compute, obtained BEFORE asking for a schedule extension.

The comparison that carries the argument is `affine` vs the larger surfaces ON THE SAME FROZEN TRUNK.
Everything else -- fitting protocol, data, folds, scoring -- is held fixed by construction, because
we reuse `train.lodo._fit_shared_adapter` and pass only a different `adapter_factory`.

WHAT THIS RUN CAN AND CANNOT DECIDE, AFTER THE 2026-08-07 EBOLA FREEZE. `freeze_ebola_arms.py` has
built and hashed both support arms, and `progress/decisions/Ebola_Prereg.md` defines few-shot as "the
same trunk with the FiLM-plus-head adapter fit on that arm's support cells only". That names the
1,428-param affine surface -- the `affine (current)` control below. So a win for a larger surface here
is a DEVELOPMENT-FOLD MECHANISM RESULT and may not be swapped into the Ebola path without the client
re-registering; scoring Ebola against a surface chosen after the freeze would void the pre-registration
that is itself a stated contribution. This is the arrangement the meta-learning note asked for:
capacity selected on development folds only, never with sight of Ebola.

It is also, separately, MORE worth measuring than when it was written. The adapter was sized against
"27 labelled examples". The frozen primary arm (L12) gives 59 support cells over 18 districts, with
48/38/18/0 adaptation pairs at h3/h5/h10/h15. There is materially more support data to fit a surface
against than the 1,428-param figure was chosen for.

    conda run -n ebola-train python -m diagnostics.capacity_probe --selfcheck            # seconds
    conda run -n ebola-train python -m diagnostics.capacity_probe --seeds 42 52 62 72 82 # ~15 h
    conda run -n ebola-train python -m diagnostics.capacity_probe --skip-fold            # reuse trunks

The downward half (protocol progress/decisions/Capacity_Down_Protocol.md, committed before the run):

    conda run -n ebola-train python -m diagnostics.capacity_probe --measure-rank         # Step 0, minutes
    conda run -n ebola-train python -m diagnostics.capacity_probe --down --skip-fold --seeds 42 52 62 72 82
    conda run -n ebola-train python -m diagnostics.capacity_probe --down --report        # from caches

Run as a MODULE from the repo root. `python diagnostics/capacity_probe.py` puts diagnostics/ on
sys.path instead of the root and dies on `import bundles`.

Reads afterwards: `Reports/Capacity_Probe_5Seed.md` (multi-seed) or `Capacity_Probe_Result.md` (one).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import bundles
import score
from bundles import HORIZONS
from models import SharedEncoder
from models.adapters import Adapter
from models.config import D_HIDDEN, QUANTILES
from results_paths import RESULTS, rpath
from train.joint import _test_dataset
from train.lodo import FLU_NAMES, _fit_shared_adapter, _fit_trunk
from train.loop import DEVICE, write_checkpoint, write_per_origin

SEEDS = (42, 52, 62, 72, 82)
SEED = SEEDS[0]                       # single-seed default; `--seeds 42` reproduces the 2026-07-31 run
OUT_MD = Path("Reports/Capacity_Probe_Result.md")
OUT_MD5 = Path("Reports/Capacity_Probe_5Seed.md")
OUT_JSON = RESULTS / "misc" / "capacity_probe.json"
OUT_JSON5 = RESULTS / "misc" / "capacity_probe_5seed.json"
# Two-sided 95% t critical values. n=5 -> 2.776, which is the figure the meta-learning note quotes
# when it argues against dropping to three seeds. scipy is not a dependency of this repo.
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def trunk_ckpt(seed):
    """Deliberately NOT `encoder_ldo__dengue2flu__seed42__ckpt.pt`. That file was written by
    run_ldo_fold on 2026-07-31 under older code; this family is refitted under today's code so all
    five seeds form one valid paired sample and the probe is reproducible from the repo as it stands.
    Starts with `encoder_ldo__`, so it routes to lodo/ with no change to results_paths.py."""
    return f"encoder_ldo__dengue2flu-cap__seed{seed}__ckpt.pt"


# --------------------------------------------------------------------------- #
# The adaptation surfaces. Each must match Adapter's interface: forward(h[N,d]) -> [N, nH, nQ].
# They form a strict capacity ladder, so "bigger did not help" is an interpretable result.
# --------------------------------------------------------------------------- #
class MLPAdapter(nn.Module):
    """Nonlinear read-out. The current adapter is provably affine in the frozen features, so this is
    the first surface in the ladder that can represent anything the affine one cannot."""
    def __init__(self, d=D_HIDDEN, hidden=64, horizons=HORIZONS, quantiles=QUANTILES):
        super().__init__()
        self.nH, self.nQ = len(horizons), len(quantiles)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, self.nH * self.nQ))

    def forward(self, h):
        return self.net(h).view(h.shape[0], self.nH, self.nQ)


class FiLMMLPAdapter(nn.Module):
    """FiLM modulation followed by a nonlinear head. Unlike the shipped Adapter, the FiLM here is NOT
    absorbable into the next layer, because a nonlinearity sits between them -- so this is the
    smallest surface in which FiLM actually buys expressive power."""
    def __init__(self, d=D_HIDDEN, hidden=64, horizons=HORIZONS, quantiles=QUANTILES):
        super().__init__()
        self.nH, self.nQ = len(horizons), len(quantiles)
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, self.nH * self.nQ))

    def forward(self, h):
        h = self.gamma * h + self.beta
        return self.net(h).view(h.shape[0], self.nH, self.nQ)


def _mlp(hidden):
    return lambda: MLPAdapter(hidden=hidden)


def _film_mlp(hidden):
    return lambda: FiLMMLPAdapter(hidden=hidden)


# ladder, smallest first. "affine" is the control: it is exactly what produced the LDO result.
SURFACES = [
    ("affine (current)", Adapter),
    ("mlp-64", _mlp(64)),
    ("film+mlp-64", _film_mlp(64)),
    ("mlp-256", _mlp(256)),
]


def n_params(factory):
    return sum(p.numel() for p in factory().parameters())


# --------------------------------------------------------------------------- #
# The downward half (--down). A sibling list, not an extension of SURFACES: train/anil.py:957-963
# reads SURFACES, five places here treat SURFACES[0] as the control, and a merged list would make a
# new or --force seed write 6-rung caches beside the archived 4-rung ones. The control keeps its
# label, so paired_deltas, interval and tally read the downward rows unchanged.
# --------------------------------------------------------------------------- #
class LowRankAdapter(nn.Module):
    """Linear(64, r, bias=False) then Linear(r, 20): 84r + 20 params. With r=None it is a single
    Linear(64, 20), the head-only surface (1,300). With the trunk frozen the shipped Adapter is exactly
    that map with FiLM folded into the head (models/adapters.py:14-16), so head-only is the same
    function class: a parametrisation control, not a smaller capacity."""
    def __init__(self, r=None, d=D_HIDDEN, horizons=HORIZONS, quantiles=QUANTILES):
        super().__init__()
        self.nH, self.nQ = len(horizons), len(quantiles)
        out = self.nH * self.nQ
        self.net = (nn.Linear(d, out) if r is None else
                    nn.Sequential(nn.Linear(d, r, bias=False), nn.Linear(r, out)))

    def forward(self, h):
        return self.net(h).view(h.shape[0], self.nH, self.nQ)


# r* = ceil(max participation ratio of the centred trunk output) over the 5 cap trunks x 4 panels,
# measured by --measure-rank on 2026-09-29 (results/misc/capacity_probe_rank.json): ceil(9.017) = 10,
# the max on dengue seed 42. Flu alone tops out at 3.817 (us-states, seed 42).
R_STAR = 10
CONTROL = SURFACES[0][0]
DOWN_SURFACES = [                               # smallest first, the control last
    (f"rank-{R_STAR}", lambda: LowRankAdapter(R_STAR)),
    ("head-only", LowRankAdapter),
    (CONTROL, Adapter),
]


# --------------------------------------------------------------------------- #
# Reference: the single-disease 5-seed mean, the same reference results_matrix.py calls
# "reference B". Named against the generator, not against progress/outcomes/Results_Matrix.md,
# because that document was superseded on 2026-09-07 and is frozen for provenance.
# --------------------------------------------------------------------------- #
def single_reference():
    """{(dataset, horizon): mean country_macro rmse over the single-disease seeds}."""
    vals = collections.defaultdict(list)
    for p in sorted((RESULTS / "single").glob("encoder__*.json")):
        for r in json.loads(p.read_text()):
            if r["metric"] == "rmse":
                vals[(r["dataset"], r["horizon"])].append(r["country_macro"])
    return {k: sum(v) / len(v) for k, v in vals.items() if v}


def rmse_of(recs):
    """{(dataset, horizon): country_macro rmse} out of a record list."""
    return {(r["dataset"], r["horizon"]): r["country_macro"]
            for r in recs if r["metric"] == "rmse"}


def improvement(new, ref):
    """Improvement %, positive = better (lower error). Same convention as results_matrix.py."""
    if ref is None or new is None or ref == 0 or math.isnan(ref) or math.isnan(new):
        return None
    return (ref - new) / abs(ref) * 100.0


# --------------------------------------------------------------------------- #
def stage1_fold(seed, skip=False, verbose=True):
    """The one frozen dengue trunk this seed's two arms share. TRUNK ONLY.

    That restriction is a correctness fix, not an optimisation. The previous version called
    `run_ldo_fold("dengue2flu", seed)`, which also fits a shared flu adapter and then WRITES
    `encoder_ldo__influenza_{japan,us-regions,us-states}__seed{S}.json`. Those files exist on disk
    for seeds 52-82 (dated 2026-07-30) and are the two-disease LDO table the client asked to have
    reported separately (D1). Sweeping five seeds through the old path would have silently
    overwritten four fifths of that table with numbers produced by different code. It was also
    duplicated work: the affine row of arm 1 refits exactly the adapter run_ldo_fold had just fit.
    """
    ck = rpath(trunk_ckpt(seed))
    if ck.exists():
        print(f"[stage1] seed {seed}: reusing trunk {ck}")
        return ck
    if skip:
        sys.exit(f"--skip-fold given but no trunk at {ck}; run without --skip-fold first")
    print(f"[stage1] seed {seed}: fitting the dengue trunk (~70 min on the 3060)")
    t0 = time.time()
    enc, _ = _fit_trunk(seed, ["dengue"], DEVICE, verbose=verbose)
    write_checkpoint(enc, None, trunk_ckpt(seed),
                     extra=dict(fold="ldo", direction="dengue2flu", seed=seed, trunk_only=True))
    print(f"[stage1] seed {seed}: done in {(time.time() - t0) / 60:.1f} min -> {ck}")
    return ck


def load_trunk(ck_path):
    """The ONE frozen trunk both sweeps share. Sharing it is what makes the control a control."""
    enc = SharedEncoder().to(DEVICE)
    ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
    enc.load_state_dict(ck["encoder"])
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    print(f"[trunk] loaded {ck_path} "
          f"({sum(v.numel() for v in ck['encoder'].values()):,} params, frozen)")
    return enc


def sweep(enc, names, tag, seed, verbose=True, archive=True, surfaces=SURFACES, reseed=False,
          keep=None):
    """Fit every surface in the ladder on the SAME frozen trunk over `names`, score, return rows.

    The last three arguments serve the downward half and default to the upward run. `reseed` seeds
    the adapter init before each fit, so a fit no longer depends on whatever state the previous one
    left (the archived inits were never seeded, train/lodo.py:278). `keep` is an out-dict, the
    _score(quant_out=) pattern, that receives the fitted control adapter: on arm 2 that is the
    regime-S anchor."""
    out = []
    for label, factory in surfaces:
        t0 = time.time()
        print(f"\n[{tag}] seed {seed} surface '{label}' ({n_params(factory):,} params) "
              f"-- fitting on {list(names)}")
        if reseed:
            torch.manual_seed(seed)
        ad, ds = _fit_shared_adapter(enc, list(names), seed, DEVICE, verbose=verbose,
                                     adapter_factory=factory)
        if keep is not None and label == CONTROL:
            keep[label] = ad
        per = {}
        for d in ds:
            recs, _, po, _ = _test_dataset(enc, ad, d, seed, f"capacity:{tag}:{label}",
                                           dict(training_regime="capacity_probe", surface=label,
                                                arm=tag, seed=seed))
            per.update(rmse_of(recs))
            if archive:
                # Per-(origin, country) sufficient stats: seconds to write, a 15 h rerun to
                # reconstruct. Same lesson the quantile archives taught in Week 4 -- the only
                # bootstrap material that cannot be recovered after the fact.
                slug = "".join(c for c in label if c.isalnum() or c == "-")
                write_per_origin(po, f"encoder_ldo__cap-{tag}-{slug}__{d.name}"
                                     f"__seed{seed}__perorigin.npz")
        mins = (time.time() - t0) / 60
        out.append(dict(label=label, params=n_params(factory), minutes=round(mins, 1), seed=seed,
                        rmse={f"{k[0]}|h{k[1]}": v for k, v in per.items()}))
        print(f"[{tag}] seed {seed} '{label}' done in {mins:.1f} min")
    return out


# --------------------------------------------------------------------------- #
# Five-seed aggregation. The comparison is PAIRED WITHIN A SEED by construction: every surface in a
# seed reads against the affine control fitted on that seed's own trunk. Trunk-to-trunk variation is
# large (the single-disease seed CV runs 8-14%) and cancels inside a seed, so an unpaired five-seed
# comparison would drown a 2-20% surface effect in trunk noise.
# --------------------------------------------------------------------------- #
def paired_deltas(rows_by_seed):
    """{surface: {cell: [one delta per seed]}}, delta = improvement % over the affine control."""
    base_label = SURFACES[0][0]
    acc = {}
    for rows in rows_by_seed:
        base = next((r for r in rows if r["label"] == base_label), None)
        if not base:
            continue
        for r in rows:
            if r["label"] == base_label:
                continue
            for cell, v in r["rmse"].items():
                d = improvement(v, base["rmse"].get(cell))
                if d is not None:
                    acc.setdefault(r["label"], {}).setdefault(cell, []).append(d)
    return acc


def interval(xs):
    """(n, mean, sd, lo, hi) for a two-sided 95% paired t-interval. lo/hi are None when n < 2.

    t rather than a bootstrap: at n=5 a bootstrap over seeds resamples five points and its tails are
    an artefact of that, not evidence. The project already reasons in t critical values here."""
    n = len(xs)
    if n == 0:
        return 0, None, None, None, None
    m = sum(xs) / n
    if n < 2:
        return n, m, None, None, None
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    t = T_CRIT.get(n)
    if t is None:
        return n, m, sd, None, None
    h = t * sd / (n ** 0.5)
    return n, m, sd, m - h, m + h


def tally(acc):
    """(cells, significantly positive, significantly negative, best significant mean).

    "Significant" = the seed-paired 95% interval excludes zero. A point estimate that does not clear
    its own seed noise is precisely what the client rejected when this was a one-seed result, so the
    headline may only quote cells that clear it."""
    cells = pos = neg = 0
    best = None
    for per_cell in acc.values():
        for xs in per_cell.values():
            n, m, sd, lo, hi = interval(xs)
            cells += 1
            if lo is not None and lo > 0:
                pos += 1
                best = m if best is None else max(best, m)
            elif hi is not None and hi < 0:
                neg += 1
    return cells, pos, neg, (best if best is not None else 0.0)


def gains_vs_control(rows):
    """Per-cell improvement % of every larger surface over the affine control, same trunk."""
    base = next((r for r in rows if r["label"] == SURFACES[0][0]), None)
    if not base:
        return []
    g = []
    for r in rows:
        if r["label"] == base["label"]:
            continue
        for k, v in r["rmse"].items():
            d = improvement(v, base["rmse"].get(k))
            if d is not None:
                g.append(d)
    return g


def _block(A, rows, title, note):
    """One arm: surfaces, absolute RMSE, and change vs the affine control."""
    A(f"\n## {title}\n")
    A(f"\n{note}\n")
    keys = sorted({k for r in rows for k in r["rmse"]})
    A("\n| surface | params | fit (min) | " + " | ".join(keys) + " |")
    A("|---|---|---|" + "---|" * len(keys))
    for r in rows:
        A(f"| {r['label']} | {r['params']:,} | {r['minutes']} | " + " | ".join(
            f"{r['rmse'][k]:,.1f}" if k in r["rmse"] else "—" for k in keys) + " |")
    base = next((r for r in rows if r["label"] == SURFACES[0][0]), None)
    if base:
        A("\nChange vs the affine control on the same trunk, positive = better:\n")
        A("\n| surface | " + " | ".join(keys) + " |")
        A("|---|" + "---|" * len(keys))
        for r in rows:
            if r["label"] == base["label"]:
                continue
            A(f"| {r['label']} | " + " | ".join(
                (lambda d: f"{d:+.1f}%" if d is not None else "—")(
                    improvement(r["rmse"].get(k), base["rmse"].get(k))) for k in keys) + " |")


def read_verdict(cross, control):
    """Mechanical read, so the morning decision is not a matter of taste.

    The control is what makes the cross-disease number interpretable. A capacity gain that appears
    on BOTH arms means the adapter was simply undersized all along and says nothing about transfer;
    a gain that appears only cross-disease is transfer-specific and is the interesting result.
    """
    gc, gi = gains_vs_control(cross), gains_vs_control(control)
    if not gc:
        return "INCONCLUSIVE", "No cross-disease gains computed."
    bc, mc = max(gc), float(np.median(gc))
    bi, mi = (max(gi), float(np.median(gi))) if gi else (float("nan"), float("nan"))
    if bc < 2.0:
        return ("REPRESENTATION-BOUND (provisional)",
                f"No larger surface beats the affine control by more than {bc:+.1f}% on any "
                f"cross-disease cell (median {mc:+.1f}%). The frozen trunk does not carry "
                f"recoverable cross-disease signal, so no read-out can recover it. ANIL adds no "
                f"read-out capacity, so it is unlikely to close a deficit of this size. "
                f"RECOMMENDATION: do not spend a week on ANIL on this evidence.")
    if gi and bi >= 0.6 * bc:
        return ("GENERAL UNDER-SIZING, NOT TRANSFER-SPECIFIC (provisional)",
                f"Larger surfaces help cross-disease (best {bc:+.1f}%, median {mc:+.1f}%) but help "
                f"the in-domain control about as much (best {bi:+.1f}%, median {mi:+.1f}%). The "
                f"adapter was simply too small everywhere. That is a cheap win worth taking, but it "
                f"is NOT evidence about cross-disease transfer and is not an argument for ANIL.")
    if bc >= 10.0:
        return ("ADAPTER-BOUND AND TRANSFER-SPECIFIC (provisional)",
                f"Larger surfaces recover up to {bc:+.1f}% cross-disease (median {mc:+.1f}%) while "
                f"the in-domain control gains only {bi:+.1f}%. The adaptation surface was a real, "
                f"transfer-specific bottleneck. There is a cheap partial fix available now AND a "
                f"genuine prior that shaping the trunk for adaptability will pay. Strongest case "
                f"for ANIL this evidence could produce.")
    return ("PARTIAL (provisional)",
            f"Cross-disease best {bc:+.1f}% (median {mc:+.1f}%), in-domain best {bi:+.1f}%. Some "
            f"capacity effect, well short of the deficit. Neither reading is clean.")


def report(cross, control, seed=SEED):
    ref = single_reference()
    verdict, detail = read_verdict(cross, control)
    L = []
    A = L.append
    A("# Capacity Probe: is the deficit adapter-bound or representation-bound?\n")
    A(f"\nGenerated by `capacity_probe.py`, seed {seed}. **ONE frozen trunk (dengue-trained) is shared "
      "by every row in both arms.** Only the adaptation surface varies; the fitting protocol "
      "(80 epochs, patience 15, lr 1e-3, wd 1e-4, uniform sampler, pooled pinball validation) is held "
      "fixed by reusing `train.lodo._fit_shared_adapter`.\n")
    A("\n**Why there are two arms.** Arm 1 asks whether a bigger read-out recovers the cross-disease "
      "deficit. Arm 2 runs the identical ladder on the trunk's OWN disease. Without arm 2 a positive "
      "result is ambiguous: 'bigger adapter helps' could just mean the adapter was undersized all "
      "along, which would say nothing about transfer. The control separates those two stories.\n")

    _block(A, cross, "Arm 1 - cross-disease (dengue trunk, adapters fitted on influenza)",
           "This is the transfer setting. The affine row is exactly the surface that produced the "
           "reported LDO result.")
    _block(A, control, "Arm 2 - in-domain control (same trunk, adapters fitted on dengue)",
           "Same trunk, same ladder, but the trunk's own disease. Any gain here is a general "
           "capacity effect, not a transfer effect.")

    keys = sorted({k for r in cross for k in r["rmse"]})
    A("\n## Context: arm 1 vs the single-disease 5-seed mean (positive = better)\n")
    A("\nThe headline transfer comparison, for orientation only. **1 seed, not a significance test.**\n")
    A("\n| surface | " + " | ".join(keys) + " |")
    A("|---|" + "---|" * len(keys))
    for r in cross:
        cells = []
        for k in keys:
            ds, h = k.split("|h")
            d = improvement(r["rmse"].get(k), ref.get((ds, int(h))))
            cells.append(f"{d:+.1f}%" if d is not None else "—")
        A(f"| {r['label']} | " + " | ".join(cells) + " |")

    A(f"\n---\n\n## Read: **{verdict}**\n")
    A(f"\n{detail}\n")
    A("\n**Provisional, and here is exactly why.** One seed, one trunk, one direction. Head capacity "
      "only - a genuinely mid-trunk FiLM injection is not tested, because that needs a change to the "
      "encoder forward pass and this run deliberately changes nothing but the read-out. A null result "
      "bounds what a *read-out* can recover from this frozen representation; it does not prove that no "
      "trunk can transfer. That is the right bound for the ANIL question, because ANIL keeps the "
      "read-out small by construction.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(seed=SEED, verdict=verdict, detail=detail,
                                        cross_disease=cross, in_domain_control=control), indent=2))
    print(f"\n{'=' * 78}\nVERDICT: {verdict}\n\n{detail}\n{'=' * 78}")
    print(f"wrote {OUT_MD} and {OUT_JSON}")
    return verdict


def _interval_table(A, acc, title):
    """Per-cell seed-paired mean, sd and 95% interval. Never pooled across datasets (client rule)."""
    A(f"\n### {title}\n")
    if not acc:
        A("\nNo paired deltas available.\n")
        return
    A("\n| surface | dataset | h | mean Δ% | sd | 95% CI | clears zero |")
    A("|---|---|---|---|---|---|---|")
    for label, per_cell in acc.items():
        for cell in sorted(per_cell):
            ds, h = cell.split("|h")
            n, m, sd, lo, hi = interval(per_cell[cell])
            ci = f"[{lo:+.1f}, {hi:+.1f}]" if lo is not None else "—"
            mark = "yes" if lo is not None and (lo > 0 or hi < 0) else "within noise"
            sdt = f"{sd:.1f}" if sd is not None else "—"
            A(f"| {label} | {ds} | {h} | {m:+.1f}% | {sdt} | {ci} | {mark} |")


def report_multiseed(cross_by_seed, control_by_seed, seeds):
    """The five-seed promotion of the probe: same ladder, same two arms, intervals over seeds."""
    xacc, cacc = paired_deltas(cross_by_seed), paired_deltas(control_by_seed)
    xc, xp, xn, xbest = tally(xacc)
    cc, cp, cn, cbest = tally(cacc)

    L = []
    A = L.append
    A("# Adapter capacity: is the cross-disease deficit adapter-bound? (five seeds)\n")
    A(f"\nGenerated by `capacity_probe.py --seeds {' '.join(map(str, seeds))}`. "
      f"**Within each seed, one frozen dengue trunk is shared by every surface in both arms**, so "
      "the only thing that varies inside a seed is the adaptation surface. Across seeds the trunk "
      "varies too, which is what the intervals below are over.\n")
    A("\nEvery figure is a **seed-paired** delta: each surface is read against the affine control "
      "fitted on *that seed's own trunk*, and the five per-seed deltas give the interval. Pairing "
      "matters — the single-disease seed CV runs 8-14%, which would swamp a 2-20% surface effect if "
      "the comparison were unpaired.\n")
    A(f"\n**Intervals are two-sided 95% t at n={len(seeds)} (t={T_CRIT.get(len(seeds), float('nan')):.3f}).** "
      "A cell counts as a result only if its interval excludes zero.\n")

    A("\n## Headline\n")
    A(f"\n| arm | cells | significantly better than affine | significantly worse | best significant mean |")
    A("|---|---|---|---|---|")
    A(f"| cross-disease (arm 1) | {xc} | **{xp}** | {xn} | {xbest:+.1f}% |")
    A(f"| in-domain control (arm 2) | {cc} | {cp} | {cn} | {cbest:+.1f}% |")
    A("\nThe control is what makes arm 1 interpretable. A capacity gain that appears on both arms "
      "means the adapter was undersized everywhere and says nothing about transfer; a gain that "
      "appears only cross-disease is transfer-specific, and that is the mechanism claim.\n")

    _interval_table(A, xacc, "Arm 1 — cross-disease (dengue trunk, adapters fitted on influenza)")
    _interval_table(A, cacc, "Arm 2 — in-domain control (same trunk, adapters fitted on dengue)")

    A("\n## Per-seed absolute RMSE\n")
    for tag, by_seed in (("arm 1 cross-disease", cross_by_seed), ("arm 2 in-domain", control_by_seed)):
        if not by_seed:
            continue
        keys = sorted({k for rows in by_seed for r in rows for k in r["rmse"]})
        A(f"\n**{tag}**\n")
        A("\n| seed | surface | params | " + " | ".join(keys) + " |")
        A("|---|---|---|" + "---|" * len(keys))
        for rows in by_seed:
            for r in rows:
                A(f"| {r['seed']} | {r['label']} | {r['params']:,} | " + " | ".join(
                    f"{r['rmse'][k]:,.1f}" if k in r["rmse"] else "—" for k in keys) + " |")

    A("\n---\n\n## What this does and does not establish\n")
    A("\nIt bounds what a larger **read-out** can recover from a frozen representation. It does not "
      "show that no trunk transfers: the FiLM here sits at the head, and a mid-trunk injection would "
      "need a change to the encoder forward pass that this probe deliberately does not make. That is "
      "the right bound for the ANIL question, because ANIL keeps the read-out small by construction.\n")
    A("\nStill one direction (dengue → influenza) and one fold structure. The trunks use the shipped "
      "early-stopping settings, unchanged, so these five seeds remain comparable to the "
      "2026-07-31 single-seed run rather than differing in two things at once.\n")
    A("\n**This cannot change the Ebola arm on its own.** The support arms were frozen and hashed on "
      "2026-08-07, and `progress/decisions/Ebola_Prereg.md` defines few-shot as the FiLM-plus-head "
      "adapter — the affine control above. Scoring Ebola against a surface chosen after that freeze "
      "would void the pre-registration, which is itself a stated methodological contribution. A win "
      "here is a development-fold mechanism result and a case to put to the client for "
      "re-registration, not a config change.\n")

    OUT_MD5.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD5.write_text("\n".join(L) + "\n", encoding="utf-8")
    OUT_JSON5.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON5.write_text(json.dumps(dict(seeds=list(seeds), cross_disease=cross_by_seed,
                                         in_domain_control=control_by_seed), indent=2))
    print(f"\n{'=' * 78}")
    print(f"cross-disease: {xp} of {xc} cells significantly better than affine "
          f"(best {xbest:+.1f}%); in-domain control: {cp} of {cc} (best {cbest:+.1f}%)")
    print(f"{'=' * 78}\nwrote {OUT_MD5} and {OUT_JSON5}")


# --------------------------------------------------------------------------- #
# The downward half. Protocol: progress/decisions/Capacity_Down_Protocol.md, committed before the
# run and sha256-stamped into every per-seed cache. This half writes only capacity_probe_down*,
# capacity_probe_tstar.json, capacity_probe_rank.json and encoder_ldo__cap-down-* archives.
# --------------------------------------------------------------------------- #
PROTOCOL = Path("progress/decisions/Capacity_Down_Protocol.md")
OUT_MD_DOWN = Path("Reports/Capacity_Probe_Down.md")
OUT_JSON_DOWN = RESULTS / "misc" / "capacity_probe_down.json"
OUT_TSTAR = RESULTS / "misc" / "capacity_probe_tstar.json"
OUT_RANK = RESULTS / "misc" / "capacity_probe_rank.json"
US_PANELS = ("influenza_us-regions", "influenza_us-states")
S_PANELS = ("influenza_japan", "influenza_us-states")      # us-regions (10 nodes) cannot host 18 rows
S_BUDGETS = ("L12", "L20")
S_LABELS = ("zeroshot", "recal-int", "recal-budget", "shrink-t")        # S0..S3; the control is CONTROL
# 5,000 not 2,000: on 8 stand-in fits (2026-09-29) the doubling move reached 1.0e-4 at 2,000 steps and
# stayed at or below 1.0e-5 at 5,000, with the same loss either way. Constant-lr jitter, not divergence.
S2_LR, S2_STEPS = 1e-2, 5000


def down_cache(seed):
    return RESULTS / "misc" / f"capacity_probe_down__seed{seed}.json"


def _write_lf(path, text):
    """LF on every OS. Plain write_text emits CRLF on Windows against this repo's LF index, which git
    then shows as a whole-file change (it does exactly that to the upward report today)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def protocol_frozen():
    """Tracked and free of local edits, the V2 pattern (ablation/run_v2_deviation.py:111-115)."""
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True)
    rel = PROTOCOL.as_posix()
    return (PROTOCOL.exists() and git("ls-files", "--error-unmatch", rel).returncode == 0
            and not git("status", "--porcelain", "--", rel).stdout.strip())


def ebola_pairs(arm):
    """Adaptation pairs per horizon of a frozen Ebola arm: counts from configs/ebola_arms.json only."""
    m = json.loads(Path("configs/ebola_arms.json").read_text())
    return {int(h): n for h, n in m["arms"][f"ebola_{arm}"]["counts"]["adapt_pairs"].items()}


def budget_levels(pairs):
    """Protocol section 2, S2. The pairs bound funds pairs/10 parameters per horizon: 2 = slope plus
    intercept (>= 2 funded), 1 = intercept only (>= 1), 0 = nothing."""
    return {h: 2 if n / 10 >= 2 else 1 if n / 10 >= 1 else 0 for h, n in pairs.items()}


class AnchoredRecal(nn.Module):
    """A borrowed affine map (A, c) held fixed as buffers, recalibrated per horizon:
    y_h = (1 + da_h) * anchor_h + db_h, one slope and one intercept per horizon, shared by the five
    quantiles. `levels` {h: 0/1/2} gates them through 0/1 buffers, so a masked delta gets a zero
    gradient and never moves under plain Adam (no weight decay)."""
    def __init__(self, A, c, levels):
        super().__init__()
        self.nH, self.nQ = len(HORIZONS), len(QUANTILES)
        lv = torch.tensor([levels[h] for h in HORIZONS], device=A.device)
        self.register_buffer("A", A.detach().float().clone())
        self.register_buffer("c", c.detach().float().clone())
        self.register_buffer("ms", (lv >= 2).float())
        self.register_buffer("mi", (lv >= 1).float())
        self.da = nn.Parameter(torch.zeros(self.nH, device=A.device))
        self.db = nn.Parameter(torch.zeros(self.nH, device=A.device))

    def n_params(self):
        return int(self.ms.sum() + self.mi.sum())

    def forward(self, h):
        base = nn.functional.linear(h, self.A, self.c).view(h.shape[0], self.nH, self.nQ)
        return base * (1 + self.da * self.ms).view(1, -1, 1) + (self.db * self.mi).view(1, -1, 1)


def _support_rows(feats, ymod, Mt, smask, origins):
    """Every support pair in one full batch: (features [n, 64], targets [n, H], mask [n, H]).
    pinball_loss over it equals train.ebola._pooled_pinball on the support mask."""
    from models import targets_and_mask
    H, Y, K = [], [], []
    for t in origins:
        tgt, msk = targets_and_mask(ymod, Mt, smask, t, DEVICE)
        k = msk.sum(1) > 0
        H.append(feats[t][k]); Y.append(tgt[k]); K.append(msk[k])
    return torch.cat(H), torch.cat(Y), torch.cat(K)


def s1_exact(A, c, rows):
    """S1, intercept per horizon, solved EXACTLY. In one intercept the pinball objective is convex and
    piecewise linear, so its minimum sits on a residual breakpoint: evaluate every one (at most 102
    pairs x 5 quantiles = 510 per horizon). A horizon with no pair keeps the anchor. -> (module, loss)"""
    from models import pinball_loss
    Hr, Y, K = rows
    ad = AnchoredRecal(A, c, {h: 1 for h in HORIZONS})
    q = torch.tensor(QUANTILES, dtype=torch.float64, device=Hr.device)
    with torch.no_grad():
        base = ad(Hr).double()
        for j in range(len(HORIZONS)):
            sel = K[:, j] > 0
            if not bool(sel.any()):
                continue
            R = Y[sel, j].double().unsqueeze(1) - base[sel, j, :]            # [n, Q] residuals
            b = R.reshape(-1)
            E = R.unsqueeze(0) - b.view(-1, 1, 1)
            ad.db[j] = b[int(torch.maximum(q * E, (q - 1) * E).mean((1, 2)).argmin())].float()
        return ad, float(pinball_loss(ad(Hr), Y, K))


def fit_s2(A, c, levels, rows, s1_loss):
    """S2, the budget-sized recalibration: full-batch Adam at a CONSTANT lr (a decaying one would pass
    the doubling check without converging). Asserts (a) support loss <= S1's exact optimum, which S2
    contains at slope 1, and (b) running twice as long moves the loss < 1e-4 relative. Keeps the
    S2_STEPS state. -> (module, loss, loss after 2x steps)"""
    from models import pinball_loss
    Hr, Y, K = rows
    ad = AnchoredRecal(A, c, levels)
    opt = torch.optim.Adam([ad.da, ad.db], lr=S2_LR)
    for step in range(2 * S2_STEPS):
        if step == S2_STEPS:
            with torch.no_grad():
                at_n, keep = float(pinball_loss(ad(Hr), Y, K)), (ad.da.clone(), ad.db.clone())
        opt.zero_grad()
        pinball_loss(ad(Hr), Y, K).backward()
        opt.step()
    with torch.no_grad():
        at_2n = float(pinball_loss(ad(Hr), Y, K))
        ad.da.copy_(keep[0]); ad.db.copy_(keep[1])
    # ponytail: 1e-6 relative is float32 summation slack on "<=", not a tolerance on convergence
    assert at_n <= s1_loss * (1 + 1e-6), f"S2 support loss {at_n:.6f} above S1 exact {s1_loss:.6f}"
    assert abs(at_2n - at_n) <= 1e-4 * at_n, f"S2 not converged: {at_n:.6f} -> {at_2n:.6f} at 2x steps"
    return ad, at_n, at_2n


def transplanted(panel, budget):
    """A FRESH bundle (blinding mutates it) carrying the real Ebola support pattern at column 0,
    t0-blind: the geometry fewshot_sim declared as "Ebola" before any result. The Ebola support-mask
    geometry is read, no labels. -> (bundle, support origins, test origins, val origins)"""
    from bundles import W
    from diagnostics.fewshot_sim import NODE_DRAW_SEED, TEMPLATES, pair_counts, transplant
    assert not panel.startswith("ebola"), "C8: Ebola must never enter a dev-fold fit"
    b = bundles.load(panel)
    _, origins = transplant(b, budget, 0, True, np.random.default_rng(NODE_DRAW_SEED))
    assert pair_counts(b.masks()["support"].astype(bool), origins) == ebola_pairs(budget), \
        f"{panel} {budget}: the transplanted budget is not the frozen Ebola one"
    te, va = b.origins(phase="test"), b.origins(phase="val")
    width = TEMPLATES[budget]["width"]
    assert min(te) - (W - 1) > width and min(va) - (W - 1) > width, \
        f"{panel} {budget}: a test or validation window reaches the transplanted block (leakage)"
    return b, origins, te, va


def support_sweep(enc, anchor, seed, verbose=True, archive=True):
    """Regime S for one seed: the Ebola support budget on two flu panels, five surfaces each.

    Modelled on fewshot_sim.run_cell, not calling it, because run_cell writes fewshotsim__ records.
    `anchor` is arm 2's refitted affine control, the dengue map standing in for Ebola's borrowed mean.
    Returns ({budget: rows in sweep() format plus mae}, {budget: {panel: fitted maps and checks}})."""
    from experiments.adapter_constraint import (T_GRID, affine_adapter, pooled_pinball_affine, shrink,
                                                to_affine)
    from models import sparse_from_dense_np
    from train.ebola import _fit_adapter, _precompute_features, choose_epochs
    from train.lodo import _score

    A_z, c_z = to_affine(anchor)
    rows, detail = {}, {}
    for bud in S_BUDGETS:
        levels = budget_levels(ebola_pairs(bud))
        got = {lab: dict(rmse={}, mae={}) for lab in (*S_LABELS, CONTROL)}
        mins, fits, ctx, aff, n_par = collections.Counter(), {}, {}, {}, {}

        def score_as(label, ad, panel):
            t0 = time.time()
            recs, _, po, _ = _score(enc, ad, *ctx[panel], panel, seed, f"capacity:down-S{bud}:{label}",
                                    dict(training_regime="capacity_probe_down", surface=label,
                                         budget=bud, arm=f"down-S{bud}", seed=seed), device=DEVICE)
            for r in recs:
                if r["metric"] in ("rmse", "mae"):
                    got[label][r["metric"]][f"{panel}|h{r['horizon']}"] = r["country_macro"]
            if archive:
                slug = "".join(ch for ch in label if ch.isalnum() or ch == "-")
                write_per_origin(po, f"encoder_ldo__cap-down-S{bud}-{slug}__{panel}__seed{seed}"
                                     f"__perorigin.npz")
            mins[label] += time.time() - t0

        for panel in S_PANELS:
            b, origins, te, va = transplanted(panel, bud)
            Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=DEVICE)   # AFTER blinding
            ymod = torch.tensor(b.y, dtype=torch.float32, device=DEVICE)
            Mt = torch.tensor(b.M, dtype=torch.float32, device=DEVICE)
            A = sparse_from_dense_np(b.A_geo).to(DEVICE)
            smask = torch.tensor(b.masks()["support"], dtype=torch.float32, device=DEVICE)
            vmask = torch.tensor(b.masks()["val"], dtype=torch.float32, device=DEVICE)
            ctx[panel] = (b, Z, Mt, A, te)
            feats = _precompute_features(enc, Z, A, Mt, origins, DEVICE)

            t0 = time.time()          # the control: the pre-registered surface, fitted as Ebola's was
            ep, _, _ = choose_epochs(feats, ymod, Mt, smask, origins, seed, DEVICE, verbose=False)
            ad, _ = _fit_adapter(feats, ymod, Mt, smask, origins, seed, ep, DEVICE)
            aff[panel] = to_affine(ad)
            mins[CONTROL] += time.time() - t0
            score_as(CONTROL, ad, panel)
            score_as("zeroshot", anchor, panel)

            t0 = time.time()
            sup = _support_rows(feats, ymod, Mt, smask, origins)
            s1, l1 = s1_exact(A_z, c_z, sup)
            mins["recal-int"] += time.time() - t0
            score_as("recal-int", s1, panel)
            t0 = time.time()
            s2, l2, l2x = fit_s2(A_z, c_z, levels, sup, l1)
            mins["recal-budget"] += time.time() - t0
            score_as("recal-budget", s2, panel)

            t0 = time.time()          # this panel's shrink curve; it sets the OTHER panel's t
            fv = _precompute_features(enc, Z, A, Mt, va, DEVICE)
            curve = [pooled_pinball_affine(*shrink(*aff[panel], A_z, c_z, t), fv, ymod, Mt, vmask, va,
                                           DEVICE) for t in T_GRID]
            mins["shrink-t"] += time.time() - t0
            n_par.update({"recal-int": s1.n_params(), "recal-budget": s2.n_params()})
            fits[panel] = dict(epochs=ep, levels={str(h): v for h, v in levels.items()},
                               s1=dict(db=s1.db.tolist(), loss=l1),
                               s2=dict(da=s2.da.tolist(), db=s2.db.tolist(), loss=l2, loss_2x=l2x),
                               affine=dict(A=aff[panel][0].tolist(), c=aff[panel][1].tolist()),
                               val_curve=curve)

        for panel in S_PANELS:        # S3: t is read off the other panel's validation fold
            other = next(p for p in S_PANELS if p != panel)
            t = float(T_GRID[int(np.argmin(fits[other]["val_curve"]))])
            fits[panel]["t"] = t
            At, ct = shrink(*aff[panel], A_z, c_z, t)
            score_as("shrink-t", affine_adapter(At.float(), ct.float(), DEVICE), panel)

        n_par.update({"zeroshot": 0, "shrink-t": n_params(Adapter), CONTROL: n_params(Adapter)})
        rows[bud] = [dict(label=lab, params=n_par[lab], minutes=round(mins[lab] / 60, 2), seed=seed,
                          **got[lab]) for lab in (*S_LABELS, CONTROL)]
        detail[bud] = fits
        if verbose:
            for r in rows[bud]:
                print(f"[S {bud}] seed {seed} {r['label']:<17} " + "  ".join(
                    f"{k.split('_')[-1]}={v:,.1f}" for k, v in sorted(r["rmse"].items())))
    return rows, detail


# --- verdicts, protocol section 3 ------------------------------------------ #
def sig(xs):
    """+1 significantly better, -1 significantly worse, 0 within noise (seed-paired 95% t)."""
    n, m, sd, lo, hi = interval(xs)
    return 1 if lo is not None and lo > 0 else -1 if hi is not None and hi < 0 else 0


def _label(wins, harm):
    # NULL covers "no cell clears" and a lone gain no second panel agrees with: agreement is the guard
    return "MIXED" if wins and harm else "WIN" if wins else "COSTS" if harm else "NULL"


def verdict_rank(cross, dom):
    """Regime F rank-r*: WIN needs japan AND a US panel significantly better at one horizon and no
    cross-disease cell significantly worse. The in-domain arm only labels it GENERAL or TRANSFER."""
    s = {c: sig(x) for c, x in cross.items()}
    wins = [h for h in HORIZONS if s.get(f"influenza_japan|h{h}") == 1
            and any(s.get(f"{u}|h{h}") == 1 for u in US_PANELS)]
    v = _label(wins, -1 in s.values())
    if v != "WIN":
        return v
    return "WIN-GENERAL" if any(sig(dom.get(f"dengue|h{h}", [])) == 1 for h in wins) else "WIN-TRANSFER"


def verdict_head(cross):
    """Regime F head-only: DIFFERS if at one horizon japan and a US panel clear the same way."""
    s = {c: sig(x) for c, x in cross.items()}
    return "DIFFERS" if any(s.get(f"influenza_japan|h{h}") == d and any(s.get(f"{u}|h{h}") == d
                                                                         for u in US_PANELS)
                            for h in HORIZONS for d in (1, -1)) else "SAME"


def verdict_s(cells, budget):
    """Regime S: deciding cells are L12 h3/h5/h10 and L20 at every horizon, both panels. WIN needs both
    panels significantly better at h3 or h5 and no deciding cell significantly worse. L12 h15 has no
    pair, so its cells would clear trivially: printed, never decided."""
    s = {c: sig(x) for c, x in cells.items() if not (budget == "L12" and c.endswith("|h15"))}
    wins = [h for h in (3, 5) if all(s.get(f"{p}|h{h}") == 1 for p in S_PANELS)]
    return _label(wins, -1 in s.values())


def _caches_bad(caches, sha):
    """Why the downward caches cannot be decided on, or None."""
    miss = [s for s in SEEDS if s not in caches]
    bad = [s for s, d in caches.items() if d.get("protocol_sha256") != sha]
    rs = [s for s, d in caches.items() if d.get("r_star") != R_STAR]
    return (f"seeds {miss} have no cache" if miss else
            f"seeds {bad} carry a protocol hash other than the committed {sha[:12]}" if bad else
            f"seeds {rs} were fitted with an r* other than {R_STAR}" if rs else None)


def _relabel(rows_by_seed, base=None, metric="rmse"):
    """Rows reshaped so the untouched paired_deltas reads them against `base` and on `metric`.
    ponytail: swapping labels keeps paired_deltas byte-identical instead of giving it a base argument."""
    swap = {base: CONTROL, CONTROL: "affine (fitted)"} if base else {}
    return [[dict(r, label=swap.get(r["label"], r["label"]), rmse=r[metric]) for r in rows]
            for rows in rows_by_seed]


def drift(seed, cross, control):
    """Refitted vs archived affine control per cell (protocol correction 9). Reported, never a gate."""
    p = RESULTS / "misc" / f"capacity_probe__seed{seed}.json"
    if not p.exists():
        return None
    old = json.loads(p.read_text())
    out = {}
    for arm, new_rows, old_rows in (("cross", cross, old["cross"]), ("control", control, old["control"])):
        n = next(r for r in new_rows if r["label"] == CONTROL)["rmse"]
        o = next(r for r in old_rows if r["label"] == CONTROL)["rmse"]
        out[arm] = {k: dict(refit=n[k], archived=o[k], rel=(n[k] - o[k]) / o[k]) for k in n if k in o}
    return out


def tstar(ds):
    """Frozen t per budget: median over seeds of the argmin of the equal-weight two-panel val pinball."""
    from experiments.adapter_constraint import T_GRID
    per = {bud: {d["seed"]: float(T_GRID[int(np.argmin(np.mean(
        [d["support_detail"][bud][p]["val_curve"] for p in S_PANELS], 0)))]) for d in ds}
        for bud in S_BUDGETS}
    return per, {bud: float(np.median(list(v.values()))) for bud, v in per.items()}


def measure_rank(seeds=SEEDS):
    """Step 0, inference only. Participation ratio (sum s^2)^2 / sum s^4 of the CENTRED trunk output
    the adapter reads (enc(Z, A, M), as _fit_shared_adapter feeds it), pooled over nodes x train
    origins, per cap trunk and panel. The 64x64 scatter is accumulated in float64; rows are never
    held. r* = ceil of the largest value."""
    from models import window_slice
    from train.joint import _prepare
    pr, panels = {}, (*FLU_NAMES, "dengue")
    for s in seeds:
        enc = load_trunk(stage1_fold(s, skip=True))
        for name in panels:
            (d,), _ = _prepare([name], DEVICE)
            ts = d.tr[::10] if name == "dengue" else d.tr     # ponytail: dengue on 1 in 10 train origins
            s1 = torch.zeros(D_HIDDEN, dtype=torch.float64, device=DEVICE)
            s2 = torch.zeros(D_HIDDEN, D_HIDDEN, dtype=torch.float64, device=DEVICE)
            n = 0
            with torch.no_grad():
                for t in ts:
                    h = enc(window_slice(d.Z, t), d.A_solo, d.Mt[:, t]).double()
                    s1 += h.sum(0); s2 += h.T @ h; n += h.shape[0]
            ev = torch.linalg.eigvalsh(s2 - torch.outer(s1, s1) / n).clamp(min=0)
            pr[(s, name)] = float(ev.sum() ** 2 / (ev ** 2).sum())
            print(f"  seed {s} {name:<22} {len(ts):>4} origins x {d.N:>5} nodes  PR {pr[(s, name)]:.3f}")
    top = max(pr, key=pr.get)
    r = math.ceil(pr[top])
    print(f"\n{'panel':<22}" + "".join(f"{s:>8}" for s in seeds) + f"{'max':>8}")
    for name in panels:
        v = [pr[(s, name)] for s in seeds]
        print(f"{name:<22}" + "".join(f"{x:>8.3f}" for x in v) + f"{max(v):>8.3f}")
    print(f"\nr* = ceil({pr[top]:.3f}) = {r}   (max on {top[1]}, seed {top[0]}); "
          f"rank-{r} adapter = 84 x {r} + 20 = {84 * r + 20} params")
    _write_lf(OUT_RANK, json.dumps(dict(
        measured="2026-09-29", definition="participation ratio (sum s^2)^2 / sum s^4 of the centred "
        "trunk output, pooled over nodes x train origins; dengue on every 10th train origin",
        trunks=[trunk_ckpt(s) for s in seeds],
        pr={name: {str(s): pr[(s, name)] for s in seeds} for name in panels},
        max=pr[top], max_at=dict(seed=top[0], panel=top[1]), r_star=r), indent=2))
    print(f"wrote {OUT_RANK}")
    return r


def run_down(a):
    """--down: per seed, regime F (DOWN_SURFACES on both arms) then regime S, cached per seed."""
    if a.no_control:
        sys.exit("--down needs arm 2: its refitted affine control is the regime-S anchor")
    if not a.report:
        if not protocol_frozen():
            sys.exit(f"[down] REFUSING TO FIT: {PROTOCOL} must be committed with no local edits. The "
                     f"decision rule is frozen before any number exists.")
        sha = protocol_sha256()
        banner(allow_cpu=a.allow_cpu)
        v = not a.quiet
        for i, s in enumerate(a.seeds, 1):
            print(f"\n{'#' * 78}\n# down seed {s}   ({i}/{len(a.seeds)})\n{'#' * 78}")
            cache = down_cache(s)
            if cache.exists() and not a.force:
                print(f"[down seed {s}] reusing cached sweep {cache}")
                continue
            t0, keep, mins = time.time(), {}, {}
            enc = load_trunk(stage1_fold(s, skip=a.skip_fold, verbose=v))
            cross = sweep(enc, FLU_NAMES, "down-arm1-cross-disease", s, v, surfaces=DOWN_SURFACES,
                          reseed=True)
            mins["regime_f_cross"] = round((time.time() - t0) / 60, 1)
            t1 = time.time()
            control = sweep(enc, ["dengue"], "down-arm2-in-domain", s, v, surfaces=DOWN_SURFACES,
                            reseed=True, keep=keep)
            mins["regime_f_in_domain"] = round((time.time() - t1) / 60, 1)
            t1 = time.time()
            support, detail = support_sweep(enc, keep[CONTROL], s, verbose=v)
            mins["regime_s"] = round((time.time() - t1) / 60, 1)
            mins["total"] = round((time.time() - t0) / 60, 1)
            if protocol_sha256() != sha:
                sys.exit(f"[down] {PROTOCOL} changed during the run; seed {s} is not written")
            _write_lf(cache, json.dumps(dict(seed=s, protocol_sha256=sha, r_star=R_STAR, cross=cross,
                                             control=control, drift=drift(s, cross, control),
                                             support=support, support_detail=detail, minutes=mins),
                                        indent=2))
            print(f"[down seed {s}] wall clock {mins} -> {cache}")
        have = [s for s in SEEDS if down_cache(s).exists()]
        if len(have) < len(SEEDS):
            print(f"[down] the report needs all five seeds; cached so far {have}")
            return
    report_down()


def report_down():
    """The downward report, rebuilt from the five per-seed caches. Refuses on a missing seed or a
    protocol hash that differs from the committed protocol."""
    from experiments.adapter_constraint import T_GRID
    sha = protocol_sha256()
    caches = {s: json.loads(down_cache(s).read_text()) for s in SEEDS if down_cache(s).exists()}
    err = _caches_bad(caches, sha)
    if err:
        sys.exit(f"[down] REFUSING TO REPORT: {err}")
    ds = [caches[s] for s in SEEDS]
    rank = DOWN_SURFACES[0][0]
    xacc, cacc = paired_deltas([d["cross"] for d in ds]), paired_deltas([d["control"] for d in ds])
    arch = json.loads(OUT_JSON5.read_text())
    ux, uc = paired_deltas(arch["cross_disease"]), paired_deltas(arch["in_domain_control"])
    by_bud = {b: [d["support"][b] for d in ds] for b in S_BUDGETS}
    sacc = {b: paired_deltas(r) for b, r in by_bud.items()}
    verdicts = ([dict(surface=rank, regime="F", scope="cross-disease",
                      verdict=verdict_rank(xacc.get(rank, {}), cacc.get(rank, {}))),
                 dict(surface="head-only", regime="F", scope="cross-disease",
                      verdict=verdict_head(xacc.get("head-only", {})))]
                + [dict(surface=lab, regime="S", scope=b, verdict=verdict_s(sacc[b].get(lab, {}), b))
                   for b in S_BUDGETS for lab in S_LABELS[1:]]
                + [dict(surface=lab, regime="upward re-read", scope="cross-disease",
                        verdict=verdict_rank(ux.get(lab, {}), uc.get(lab, {}))) for lab, _ in SURFACES[1:]])
    per_t, tst = tstar(ds)
    moves = [f[p]["s2"] for d in ds for f in d["support_detail"].values() for p in S_PANELS]

    L = []
    A = L.append
    A("# Adapter capacity, downward half (five seeds)\n")
    A(f"\nGenerated by `capacity_probe.py --down --report` from the five per-seed caches "
      f"`results/misc/capacity_probe_down__seed*.json`. Protocol `{PROTOCOL.as_posix()}`, sha256 "
      f"`{sha}`, stamped in every cache. r* = {R_STAR}. Every figure is a seed-paired improvement "
      f"over the same seed's control, positive = better, two-sided 95% t at n=5 (t=2.776). The "
      f"deciding metric is country-macro RMSE; MAE is printed and decides nothing.\n")
    A("\n## Verdicts\n")
    A("\n| surface | regime | scope | verdict |")
    A("|---|---|---|---|")
    for v in verdicts:
        A(f"| {v['surface']} | {v['regime']} | {v['scope']} | **{v['verdict']}** |")
    A("\nRegime F decides on 28 cells (rank-r* 12 cross plus 4 in-domain, head-only 12 cross), "
      "regime S on 42 (3 surfaces x (6 L12 + 8 L20)): 70 deciding cells, 8 verdicts. The upward "
      "re-read is the archived upward half under the regime-F rule, descriptive only.\n")

    _interval_table(A, {k: xacc[k] for k in (rank, "head-only") if k in xacc},
                    "Regime F, cross-disease (dengue trunk, read-outs fitted on influenza)")
    _interval_table(A, {k: cacc[k] for k in (rank, "head-only") if k in cacc},
                    "Regime F, in-domain (dengue). rank-r* uses these to label a WIN; head-only's are "
                    "printed only")

    if all(d.get("drift") for d in ds):
        A("\n## Control drift: refitted affine against the archived one (not a gate)\n")
        A("\nThe archived affine control cannot be reproduced bit for bit, its init was never seeded, "
          "so this half refits its own, seeded. Drift = (refit - archived) / archived per seed. Seed "
          "CV = sd / mean of the archived control over the five trunks, the trunk-to-trunk noise.\n")
        A("\n| arm | cell | median drift | min | max | archived seed CV |")
        A("|---|---|---|---|---|---|")
        for arm, key in (("cross", "cross_disease"), ("control", "in_domain_control")):
            for c in sorted(ds[0]["drift"][arm]):
                rel = [100 * d["drift"][arm][c]["rel"] for d in ds]
                base = [next(r for r in rows if r["label"] == CONTROL)["rmse"][c] for rows in arch[key]]
                A(f"| {arm} | {c} | {np.median(rel):+.1f}% | {min(rel):+.1f}% | {max(rel):+.1f}% | "
                  f"{100 * np.std(base, ddof=1) / np.mean(base):.1f}% |")

    for b in S_BUDGETS:
        A(f"\n## Regime S, {b} support pattern on influenza_japan and influenza_us-states\n")
        A(f"\nBudget levels per horizon (2 slope and intercept, 1 intercept, 0 nothing): "
          f"{budget_levels(ebola_pairs(b))}."
          + (" **L12 h15 rows carry no verdict**: there is no pair, so S1 and S2 equal the anchor and "
             "the control is a weight-decayed random head." if b == "L12" else "") + "\n")
        _interval_table(A, {k: sacc[b][k] for k in S_LABELS[1:] if k in sacc[b]},
                        f"{b}: each surface against the fresh affine control (RMSE, deciding)")
        _interval_table(A, paired_deltas(_relabel(by_bud[b], base="zeroshot")),
                        f"{b}: every surface and the control against S0, the unchanged anchor "
                        f"(RMSE, descriptive)")
        _interval_table(A, paired_deltas(_relabel(by_bud[b], metric="mae")),
                        f"{b}: against the control on MAE (descriptive)")
        A(f"\n### {b}: shrinkage strength t, read off the other panel's validation fold\n")
        A("\n| seed | t for influenza_japan | t for influenza_us-states | pooled two-panel argmin |")
        A("|---|---|---|---|")
        for d in ds:
            f = d["support_detail"][b]
            A(f"| {d['seed']} | {f['influenza_japan']['t']} | {f['influenza_us-states']['t']} | "
              f"{per_t[b][d['seed']]} |")
        A(f"\nFrozen t* for {b} = **{tst[b]}** (median over seeds of the pooled argmin).\n")

    A("\n## Fitting checks\n")
    A(f"\nS2 passed both asserted checks on all {len(moves)} fits: support loss at or below S1's "
      f"exact optimum, and the largest relative move from doubling the steps was "
      f"{max(abs(m['loss_2x'] - m['loss']) / m['loss'] for m in moves):.1e} (limit 1e-4).\n")
    A("\n## Runtime\n")
    A("\n| seed | regime F cross (min) | regime F in-domain (min) | regime S (min) | total (min) |")
    A("|---|---|---|---|---|")
    for d in ds:
        m = d["minutes"]
        A(f"| {d['seed']} | {m['regime_f_cross']} | {m['regime_f_in_domain']} | {m['regime_s']} | "
          f"{m['total']} |")
    _write_lf(OUT_MD_DOWN, "\n".join(L) + "\n")

    iv = lambda acc: {lab: {c: dict(zip(("n", "mean", "sd", "lo", "hi"), interval(xs)))
                            for c, xs in cells.items()} for lab, cells in acc.items()}
    payload = dict(protocol_sha256=sha, seeds=list(SEEDS), t_grid=[float(t) for t in T_GRID],
                   rule="median over seeds of the argmin of the equal-weight two-panel val pinball",
                   per_seed=per_t, tstar=tst)
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    _write_lf(OUT_TSTAR, json.dumps(payload, indent=2))
    _write_lf(OUT_JSON_DOWN, json.dumps(dict(
        protocol_sha256=sha, seeds=list(SEEDS), r_star=R_STAR, verdicts=verdicts,
        intervals=dict(f_cross=iv(xacc), f_in_domain=iv(cacc),
                       **{f"s_{b}": iv(sacc[b]) for b in S_BUDGETS}),
        tstar=tst, tstar_sha256=payload["sha256"], minutes={d["seed"]: d["minutes"] for d in ds}),
        indent=2))
    print(f"\n{'=' * 78}")
    for v in verdicts:
        print(f"{v['surface']:<18} {v['regime']:<15} {v['scope']:<14} {v['verdict']}")
    print(f"t* {tst}  (sha256 {payload['sha256'][:16]})\n{'=' * 78}")
    print(f"wrote {OUT_MD_DOWN}, {OUT_JSON_DOWN} and {OUT_TSTAR}")


def _selfcheck_down():
    """Seconds, no training: the downward surfaces, the recalibration, the budget rule, both solvers,
    the verdict rule, the archived upward re-read, and the regime-S transplant wiring."""
    from diagnostics.fewshot_sim import TEMPLATES
    from experiments.adapter_constraint import affine_adapter, to_affine
    from results_paths import subdir_for

    # 1. surfaces
    assert isinstance(R_STAR, int) and R_STAR >= 1, f"R_STAR must be the measured integer, got {R_STAR}"
    for label, factory in DOWN_SURFACES:
        out = factory()(torch.randn(11, D_HIDDEN))
        assert out.shape == (11, len(HORIZONS), len(QUANTILES)), (label, out.shape)
    for r in (1, 2, 3, 5, R_STAR):
        assert n_params(lambda: LowRankAdapter(r)) == 84 * r + 20, r
    sizes = [n_params(f) for _, f in DOWN_SURFACES]
    assert sizes == [84 * R_STAR + 20, 1300, 1428], sizes
    assert DOWN_SURFACES[-1][0] == SURFACES[0][0] == CONTROL, "the control label must match the upward half"
    print(f"ok  downward ladder {[f'{l}={s:,}' for (l, _), s in zip(DOWN_SURFACES, sizes)]}, all emit [N,H,Q]")

    # 2. budget rule from the frozen config
    pairs = {b: ebola_pairs(b) for b in S_BUDGETS}
    assert pairs == {b: TEMPLATES[b]["pairs"] for b in S_BUDGETS}, "config and fewshot_sim disagree"
    lv = {b: budget_levels(p) for b, p in pairs.items()}
    assert lv["L12"] == {3: 2, 5: 2, 10: 1, 15: 0} and lv["L20"] == {3: 2, 5: 2, 10: 2, 15: 2}, lv
    print(f"ok  budget rule from configs/ebola_arms.json: L12 {lv['L12']}, L20 {lv['L20']}")

    # 3. AnchoredRecal: zero deltas reproduce the anchor; parameter counts
    torch.manual_seed(3)
    anc = Adapter()
    with torch.no_grad():
        for p in anc.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    A, c = to_affine(anc)
    h = torch.randn(9, D_HIDDEN)
    counts = {}
    with torch.no_grad():
        for name, levels in (("S1", {k: 1 for k in HORIZONS}), ("S2 L12", lv["L12"]), ("S2 L20", lv["L20"])):
            rec = AnchoredRecal(A, c, levels)
            assert float((rec(h) - affine_adapter(A, c, "cpu")(h)).abs().max()) == 0.0, name
            assert float((rec(h) - anc(h)).abs().max()) < 1e-5, name
            counts[name] = rec.n_params()
    assert counts == {"S1": 4, "S2 L12": 5, "S2 L20": 8}, counts
    print(f"ok  AnchoredRecal with zero deltas equals the folded anchor exactly; params {counts}")

    # 4. S1 exact against a brute-force grid, and S2's two asserted checks, on a toy problem
    from models import pinball_loss
    g = torch.Generator().manual_seed(5)
    Hr, Y = torch.randn(30, D_HIDDEN, generator=g), 2 * torch.randn(30, len(HORIZONS), generator=g)
    K = (torch.rand(30, len(HORIZONS), generator=g) < 0.6).float()
    K[:, -1] = 0                                                  # an empty horizon, like L12 h15
    s1, l1 = s1_exact(A, c, (Hr, Y, K))
    q = torch.tensor(QUANTILES, dtype=torch.float64)
    with torch.no_grad():
        base = AnchoredRecal(A, c, lv["L12"])(Hr).double()
    for j in range(len(HORIZONS)):
        sel = K[:, j] > 0
        if not bool(sel.any()):
            assert float(s1.db[j]) == 0.0, "a horizon with no pair must keep the anchor"
            continue
        R = Y[sel, j].double().unsqueeze(1) - base[sel, j, :]
        grid = torch.linspace(float(R.min()), float(R.max()), 20001, dtype=torch.float64)
        E = R.unsqueeze(0) - grid.view(-1, 1, 1)
        Lg = torch.maximum(q * E, (q - 1) * E).mean((1, 2))
        Ex = R - float(s1.db[j])
        le = float(torch.maximum(q * Ex, (q - 1) * Ex).mean())
        assert le <= float(Lg.min()) + 1e-6 and float(Lg.min()) - le < 1e-3, (j, le, float(Lg.min()))
    with torch.no_grad():
        assert abs(float(pinball_loss(s1(Hr), Y, K)) - l1) < 1e-6
    _, l2, l2x = fit_s2(A, c, lv["L20"], (Hr, Y, K), l1)
    print(f"ok  S1 exact matches a 20,001-point grid on every horizon; S2 {l2:.5f} <= S1 {l1:.5f}, "
          f"2x steps move {abs(l2x - l2) / l2:.1e}")

    # 5. verdict rule
    def acc(**cells):             # J3=1 -> influenza_japan|h3 clearing better, -1 worse, 0 noise
        full = dict(J="influenza_japan", R="influenza_us-regions", S="influenza_us-states", D="dengue")
        return {f"{full[c[0]]}|h{c[1:]}": [10.0 * s + e for e in (0.1, -0.1, 0.2, -0.2, 0.0)]
                for c, s in cells.items()}
    noise = acc(J3=0, J5=0, J10=0, J15=0, R3=0, R15=0, S3=0, S15=0)
    assert verdict_rank(noise, {}) == "NULL"
    assert verdict_rank({**noise, **acc(J15=1, R15=1)}, {}) == "WIN-TRANSFER"
    assert verdict_rank({**noise, **acc(J15=1, S15=1)}, acc(D15=1)) == "WIN-GENERAL"
    assert verdict_rank({**noise, **acc(J15=1, S15=1)}, acc(D3=1)) == "WIN-TRANSFER", "wrong horizon"
    assert verdict_rank({**noise, **acc(J15=1, R15=1, J3=-1)}, {}) == "MIXED"
    assert verdict_rank({**noise, **acc(J3=-1)}, {}) == "COSTS"
    assert verdict_rank({**noise, **acc(R15=1, S15=1)}, {}) == "NULL", "two US panels alone must not WIN"
    assert verdict_rank({**noise, **acc(J15=1)}, {}) == "NULL", "a lone panel must not WIN"
    assert verdict_head({**noise, **acc(J5=-1, S5=-1)}) == "DIFFERS"
    assert verdict_head({**noise, **acc(J5=1, S5=-1)}) == "SAME", "opposite directions do not agree"
    assert verdict_head({**noise, **acc(R5=1, S5=1)}) == "SAME"
    sn = acc(J3=0, J5=0, J10=0, J15=0, S3=0, S5=0, S10=0, S15=0)
    assert verdict_s({**sn, **acc(J3=1, S3=1)}, "L12") == "WIN"
    assert verdict_s({**sn, **acc(J3=1, S3=1, J15=-1, S15=-1)}, "L12") == "WIN", "L12 h15 must be ignored"
    assert verdict_s({**sn, **acc(J3=1, S3=1, J15=-1)}, "L20") == "MIXED", "L20 h15 decides"
    assert verdict_s({**sn, **acc(J10=1, S10=1)}, "L20") == "NULL", "wins count at h3 or h5 only"
    assert verdict_s({**sn, **acc(J15=1, S15=1)}, "L12") == "NULL"
    assert verdict_s({**sn, **acc(S10=-1)}, "L20") == "COSTS"
    assert verdict_s({**sn, **acc(J5=1)}, "L20") == "NULL", "one panel alone must not WIN"
    print("ok  verdicts: WIN-TRANSFER, WIN-GENERAL, MIXED, COSTS, NULL fire; US-only and lone-panel "
          "gains rejected; head-only DIFFERS/SAME; L12 h15 ignored, L20 h15 decides")

    # 6. the archived upward half, re-read under the regime-F rule
    arch = json.loads(OUT_JSON5.read_text())
    ux, uc = paired_deltas(arch["cross_disease"]), paired_deltas(arch["in_domain_control"])
    up = {lab: verdict_rank(ux[lab], uc.get(lab, {})) for lab, _ in SURFACES[1:]}
    assert set(up.values()) == {"MIXED"}, up
    print(f"ok  archived upward half re-read under the regime-F rule: {up}")

    # 7. regime-S wiring, no fit: both budgets on both panels
    for panel in S_PANELS:
        for bud in S_BUDGETS:
            b, origins, te, va = transplanted(panel, bud)
            s = b.masks()["support"].astype(bool)
            assert (s <= b.masks()["train"].astype(bool)).all(), "support escaped the train fold"
            print(f"ok  {panel} {bud}: {int(s.sum())} support cells, origins {origins[0]}..{origins[-1]}, "
                  f"pairs {ebola_pairs(bud)}, first val origin {min(va)}, first test origin {min(te)}")

    # 8. routing and cache refusal
    names = [f"encoder_ldo__cap-{t}-x__influenza_japan__seed42__perorigin.npz"
             for t in ("down-arm1-cross-disease", "down-arm2-in-domain", "down-SL12", "down-SL20")]
    assert all(subdir_for(n) == "lodo" and n.startswith("encoder_ldo__cap-down-") for n in names)
    assert subdir_for(down_cache(42).name) == subdir_for(OUT_TSTAR.name) == "misc"
    assert not {OUT_MD_DOWN, OUT_JSON_DOWN, OUT_TSTAR} & {OUT_MD, OUT_MD5, OUT_JSON, OUT_JSON5}
    good = {s: dict(protocol_sha256="x", r_star=R_STAR) for s in SEEDS}
    assert _caches_bad(good, "x") is None
    assert "no cache" in _caches_bad({s: good[s] for s in SEEDS[:4]}, "x")
    assert "protocol hash" in _caches_bad({**good, 52: dict(protocol_sha256="y", r_star=R_STAR)}, "x")
    print(f"ok  archives route to lodo/ as encoder_ldo__cap-down-*, caches to misc/; the report refuses "
          f"a missing seed or a stale hash; protocol committed and clean: {protocol_frozen()}")


def _selfcheck():
    """Wiring only, no training: shapes, the capacity ladder, and the read logic."""
    for label, factory in SURFACES:
        ad = factory()
        h = torch.randn(11, D_HIDDEN)
        out = ad(h)
        assert out.shape == (11, len(HORIZONS), len(QUANTILES)), (label, out.shape)
    sizes = [n_params(f) for _, f in SURFACES]
    assert sizes[0] == 1428, f"control must be the shipped 1,428-param adapter, got {sizes[0]}"
    assert all(s > sizes[0] for s in sizes[1:]), f"ladder must be strictly larger than control: {sizes}"
    # the point of MLPAdapter: it must NOT be affine, or the probe tests nothing
    ad = MLPAdapter()
    with torch.no_grad():
        for p in ad.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        a, b = torch.randn(1, D_HIDDEN), torch.randn(1, D_HIDDEN)
        lin = (ad(a + b) - ad(a) - ad(b) + ad(torch.zeros(1, D_HIDDEN))).abs().max()
    assert float(lin) > 1e-4, "MLP surface behaves affinely; the probe would be vacuous"
    # sign convention
    assert abs(improvement(90.0, 100.0) - 10.0) < 1e-9, "positive must mean better (lower error)"
    assert improvement(110.0, 100.0) < 0
    assert improvement(1.0, 0.0) is None and improvement(float("nan"), 1.0) is None

    # the verdict logic is what the morning decision reads, so exercise every branch
    def rows(base, other):
        return [dict(label=SURFACES[0][0], params=1, minutes=0, rmse={"d|h3": base}),
                dict(label="mlp-64", params=2, minutes=0, rmse={"d|h3": other})]
    flat = rows(100.0, 100.0)                       # no gain anywhere
    big_cross, big_ctrl = rows(100.0, 80.0), rows(100.0, 79.0)     # 20% both arms
    v, _ = read_verdict(flat, flat)
    assert v.startswith("REPRESENTATION-BOUND"), v
    v, _ = read_verdict(big_cross, big_ctrl)
    assert v.startswith("GENERAL UNDER-SIZING"), v          # control gains too -> not transfer
    v, _ = read_verdict(big_cross, flat)
    assert v.startswith("ADAPTER-BOUND AND TRANSFER-SPECIFIC"), v  # control flat -> transfer-specific
    v, _ = read_verdict(rows(100.0, 95.0), flat)
    assert v.startswith("PARTIAL"), v
    assert read_verdict([], [])[0] == "INCONCLUSIVE"

    # ---- five-seed machinery -------------------------------------------------
    n, m, sd, lo, hi = interval([10.0] * 5)
    assert (n, m, sd) == (5, 10.0, 0.0) and lo == hi == 10.0, "zero-variance interval must collapse"
    n, m, sd, lo, hi = interval([0.0, 5.0, 10.0, 15.0, 20.0])
    assert n == 5 and abs(m - 10.0) < 1e-9 and abs(sd - 7.90569) < 1e-4, (n, m, sd)
    assert lo > 0, "mean 10, sd 7.9, n=5 (t=2.776) should still clear zero"
    assert interval([-10.0, 0.0, 10.0, 20.0, 30.0])[3] < 0, "twice that spread must stop clearing"
    assert interval([1.0])[3] is None, "n=1 has no interval -- which is the whole reason for this task"

    # pairing must happen WITHIN a seed: three wildly different trunks, same 10% surface effect.
    # An unpaired reading of these would see a spread of 45-200 and find nothing.
    def _r(base, other, sd_):
        return [dict(label=SURFACES[0][0], params=1, minutes=0, seed=sd_, rmse={"d|h3": base}),
                dict(label="mlp-64", params=2, minutes=0, seed=sd_, rmse={"d|h3": other})]
    acc = paired_deltas([_r(100.0, 90.0, 1), _r(200.0, 180.0, 2), _r(50.0, 45.0, 3)])
    assert acc["mlp-64"]["d|h3"] == [10.0, 10.0, 10.0], f"pairing did not cancel trunk scale: {acc}"

    assert tally({"s": {"c": [10.0, 10.1, 9.9, 10.0, 10.0]}})[:3] == (1, 1, 0), "consistent gain"
    assert tally({"s": {"c": [-10.0] * 5}})[:3] == (1, 0, 1), "consistent loss must register as worse"
    assert tally({"s": {"c": [30.0, -20.0, 10.0, -25.0, 5.0]}})[:3] == (1, 0, 0), \
        "a noisy cell must clear neither direction -- this is the guard against the old 1-seed read"

    print("ok  4 surfaces emit [N,H,Q]; ladder strictly exceeds the 1,428-param control; "
          "MLP is genuinely non-affine; improvement sign correct; all 5 verdict branches fire")
    print("ok  seed-paired machinery: t-interval correct at n=5, pairing cancels trunk scale, "
          "tally counts only cells whose interval excludes zero")
    print(f"    ladder: {[f'{l}={s:,}' for (l, _), s in zip(SURFACES, sizes)]}")


def banner(allow_cpu=False):
    """Fail loudly rather than spend a night on the CPU by accident."""
    # train.loop.DEVICE is a plain string ("cuda"/"cpu"), not a torch.device -- accept either.
    dev = DEVICE if isinstance(DEVICE, str) else DEVICE.type
    # is_available() can be True while the device list is empty (CUDA_VISIBLE_DEVICES=""), and
    # get_device_name(0) then raises. Gate on the count so this check reports rather than crashes.
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    ok = n_gpu > 0 and str(dev).startswith("cuda")
    try:
        name = torch.cuda.get_device_name(0) if n_gpu else "n/a"
    except Exception as e:                       # never let the device CHECK be what kills the run
        name, ok = f"unavailable ({e})", False
    print(f"[device] torch {torch.__version__} | cuda_available={torch.cuda.is_available()} "
          f"| n_gpu={n_gpu} | DEVICE={DEVICE} | gpu={name}")
    if not ok and not allow_cpu:
        sys.exit("[device] REFUSING TO START: DEVICE is not cuda. This run is hours on a GPU and "
                 "far longer on CPU. Fix the environment, or pass --allow-cpu if you really mean it.")
    if ok:
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--skip-fold", action="store_true",
                    help="reuse an existing trunk checkpoint instead of running stage 1")
    ap.add_argument("--no-control", action="store_true",
                    help="skip arm 2; the cross-disease result is then ambiguous, see the report")
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED],
                    help="one sweep per seed, each on its own trunk. Five seeds is ~15 h on the "
                         "3060 (trunk 71 min + arm1 16 min + arm2 96 min per seed). RESUMABLE: a "
                         "finished seed is cached and skipped.")
    ap.add_argument("--force", action="store_true", help="redo seeds already cached")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--measure-rank", action="store_true",
                    help="Step 0 of the downward half: participation ratio of the cap trunks' output, "
                         "inference only, minutes")
    ap.add_argument("--down", action="store_true",
                    help="the downward half: DOWN_SURFACES on both arms plus the Ebola-budget support "
                         "sweep. Refuses unless the protocol is committed. RESUMABLE per seed.")
    ap.add_argument("--report", action="store_true",
                    help="with --down: rebuild the downward report from the five caches, fit nothing")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
        _selfcheck_down()
    elif a.measure_rank:
        banner(allow_cpu=a.allow_cpu)
        measure_rank()
    elif a.down:
        run_down(a)
    else:
        banner(allow_cpu=a.allow_cpu)
        t0 = time.time()
        cross_by_seed, control_by_seed = [], []
        for i, s in enumerate(a.seeds, 1):
            print(f"\n{'#' * 78}\n# seed {s}   ({i}/{len(a.seeds)})\n{'#' * 78}")
            cache = RESULTS / "misc" / f"capacity_probe__seed{s}.json"
            if cache.exists() and not a.force:
                print(f"[seed {s}] reusing cached sweep {cache}")
                d = json.loads(cache.read_text())
            else:
                enc = load_trunk(stage1_fold(s, skip=a.skip_fold, verbose=not a.quiet))
                d = dict(seed=s,
                         cross=sweep(enc, FLU_NAMES, "arm1-cross-disease", s, verbose=not a.quiet),
                         control=([] if a.no_control else
                                  sweep(enc, ["dengue"], "arm2-in-domain", s, verbose=not a.quiet)))
                # Written per seed, not at the end: a 15 h job that dies on seed 4 should cost one
                # seed, not the night. The overnight LDO3 run needed several restarts.
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(d, indent=2))
                print(f"[seed {s}] cached -> {cache}")
            cross_by_seed.append(d["cross"])
            if d["control"]:
                control_by_seed.append(d["control"])
        if len(a.seeds) == 1:
            report(cross_by_seed[0], control_by_seed[0] if control_by_seed else [], seed=a.seeds[0])
        else:
            report_multiseed(cross_by_seed, control_by_seed, a.seeds)
        if torch.cuda.is_available():
            print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"total {(time.time() - t0) / 60:.1f} min")

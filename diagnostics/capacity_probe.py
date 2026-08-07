"""capacity_probe.py -- OVERNIGHT RUN. Is the cross-disease deficit caused by the adaptation surface,
or by the representation?

WHY THIS IS THE HIGHEST-VALUE RUN AVAILABLE. The whole meta-learning proposal rests on an untested
premise. Under the corrected leave-one-disease-out fold, cross-disease transfer is negative in 12 of
16 RMSE cells. The ANIL argument is that shaping the trunk to be "repairable by a small support-set
update" fixes this. But ANIL does not add capacity to the adaptation surface -- it changes what the
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

IT ALSO CLOSES A KNOWN GAP FOR FREE. The dengue->flu direction is missing seed 42 (4 seeds on disk,
5 everywhere else). Stage 1 below runs exactly that fold, so the same night produces the fifth seed,
the first trunk checkpoint the project has ever saved, and the first archived quantile predictions
(G4 has been blocked on those). None of it is wasted under any scope decision.

    conda run -n ebola-train python capacity_probe.py                  # full run
    conda run -n ebola-train python capacity_probe.py --selfcheck      # wiring only, seconds
    conda run -n ebola-train python capacity_probe.py --skip-fold      # reuse an existing checkpoint

Reads in the morning: `Reports/Capacity_Probe_Result.md`.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
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
# Reference: the single-disease 5-seed mean, the same reference Results_Matrix.md uses.
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


def sweep(enc, names, tag, seed, verbose=True, archive=True):
    """Fit every surface in the ladder on the SAME frozen trunk over `names`, score, return rows."""
    out = []
    for label, factory in SURFACES:
        t0 = time.time()
        print(f"\n[{tag}] seed {seed} surface '{label}' ({n_params(factory):,} params) "
              f"-- fitting on {list(names)}")
        ad, ds = _fit_shared_adapter(enc, list(names), seed, DEVICE, verbose=verbose,
                                     adapter_factory=factory)
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

    OUT_MD5.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD5.write_text("\n".join(L) + "\n", encoding="utf-8")
    OUT_JSON5.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON5.write_text(json.dumps(dict(seeds=list(seeds), cross_disease=cross_by_seed,
                                         in_domain_control=control_by_seed), indent=2))
    print(f"\n{'=' * 78}")
    print(f"cross-disease: {xp} of {xc} cells significantly better than affine "
          f"(best {xbest:+.1f}%); in-domain control: {cp} of {cc} (best {cbest:+.1f}%)")
    print(f"{'=' * 78}\nwrote {OUT_MD5} and {OUT_JSON5}")


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
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
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

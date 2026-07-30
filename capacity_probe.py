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
from train.lodo import FLU_NAMES, _fit_shared_adapter, run_ldo_fold
from train.loop import DEVICE, write_checkpoint

SEED = 42
CKPT = f"encoder_ldo__dengue2flu__seed{SEED}__ckpt.pt"
OUT_MD = Path("Reports/Capacity_Probe_Result.md")
OUT_JSON = RESULTS / "misc" / "capacity_probe.json"


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
def stage1_fold(skip=False, verbose=True):
    """Run the dengue->flu LDO fold at seed 42: closes the missing-seed gap AND leaves a checkpoint."""
    ck = rpath(CKPT)
    if skip or ck.exists():
        if not ck.exists():
            sys.exit(f"--skip-fold given but no checkpoint at {ck}; run without --skip-fold first")
        print(f"[stage1] reusing existing checkpoint {ck}")
        return ck
    print(f"[stage1] running LDO fold dengue2flu seed {SEED} "
          f"(closes the 4->5 seed gap; writes ckpt + quantiles). Expect ~1 h on the 3060.")
    t0 = time.time()
    run_ldo_fold("dengue2flu", SEED, device=DEVICE, verbose=verbose)
    print(f"[stage1] done in {(time.time() - t0) / 60:.1f} min -> {ck}")
    return ck


def stage2_sweep(ck_path, verbose=True):
    """Fit each surface on the SAME frozen trunk, score all three flu bundles, return results."""
    enc = SharedEncoder().to(DEVICE)
    ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
    enc.load_state_dict(ck["encoder"])
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    print(f"[stage2] trunk loaded from {ck_path} ({sum(v.numel() for v in ck['encoder'].values()):,} params)")

    out = []
    for label, factory in SURFACES:
        t0 = time.time()
        print(f"\n[stage2] surface '{label}' ({n_params(factory):,} params) -- fitting")
        ad, ds = _fit_shared_adapter(enc, list(FLU_NAMES), SEED, DEVICE, verbose=verbose,
                                     adapter_factory=factory)
        per = {}
        for d in ds:
            recs, _, _, _ = _test_dataset(enc, ad, d, SEED, f"capacity:{label}",
                                          dict(training_regime="capacity_probe", surface=label))
            per.update(rmse_of(recs))
        mins = (time.time() - t0) / 60
        out.append(dict(label=label, params=n_params(factory), minutes=round(mins, 1),
                        rmse={f"{k[0]}|h{k[1]}": v for k, v in per.items()}))
        print(f"[stage2] '{label}' done in {mins:.1f} min")
    return out


def report(results):
    ref = single_reference()
    base = next((r for r in results if r["label"] == "affine (current)"), None)
    L = []
    A = L.append
    A("# Capacity Probe: is the deficit adapter-bound or representation-bound?\n")
    A(f"\nGenerated by `capacity_probe.py`. Fold: **dengue -> influenza**, seed {SEED}, ONE frozen "
      "trunk shared by every row. Only the adaptation surface varies; the fitting protocol "
      "(80 epochs, patience 15, lr 1e-3, wd 1e-4, uniform 1/3 sampler, pooled pinball validation) "
      "is held fixed by reusing `train.lodo._fit_shared_adapter`.\n")
    A("\n**How to read this.** The decisive comparison is each row against **`affine (current)`**, "
      "which is exactly the surface that produced the reported LDO result. If the larger surfaces "
      "do not beat it, the frozen representation does not carry the information and no read-out can "
      "recover it -- in which case ANIL, which adds no capacity, is unlikely to close the gap. If "
      "they do beat it, the surface was a real bottleneck.\n")

    A("\n## Adaptation surfaces\n")
    A("\n| surface | params | fit time (min) |")
    A("|---|---|---|")
    for r in results:
        A(f"| {r['label']} | {r['params']:,} | {r['minutes']} |")

    keys = sorted({k for r in results for k in r["rmse"]})
    A("\n## Country-macro RMSE (lower is better)\n")
    A("\n| surface | " + " | ".join(keys) + " |")
    A("|---|" + "---|" * len(keys))
    for r in results:
        A(f"| {r['label']} | " + " | ".join(
            f"{r['rmse'][k]:,.1f}" if k in r["rmse"] else "—" for k in keys) + " |")

    if base:
        A("\n## Change vs the current affine surface, same trunk (positive = better)\n")
        A("\n| surface | " + " | ".join(keys) + " |")
        A("|---|" + "---|" * len(keys))
        for r in results:
            if r["label"] == base["label"]:
                continue
            cells = []
            for k in keys:
                d = improvement(r["rmse"].get(k), base["rmse"].get(k))
                cells.append(f"{d:+.1f}%" if d is not None else "—")
            A(f"| {r['label']} | " + " | ".join(cells) + " |")

    A("\n## Context: same rows vs the single-disease 5-seed mean (positive = better)\n")
    A("\nThis is the headline transfer comparison, shown for orientation only. It is a **1-seed** "
      "figure and is not a significance test.\n")
    A("\n| surface | " + " | ".join(keys) + " |")
    A("|---|" + "---|" * len(keys))
    for r in results:
        cells = []
        for k in keys:
            ds, h = k.split("|h")
            d = improvement(r["rmse"].get(k), ref.get((ds, int(h))))
            cells.append(f"{d:+.1f}%" if d is not None else "—")
        A(f"| {r['label']} | " + " | ".join(cells) + " |")

    # the read, stated mechanically so the morning decision is not a matter of taste
    verdict = "INCONCLUSIVE"
    detail = ""
    if base:
        gains = []
        for r in results:
            if r["label"] == base["label"]:
                continue
            for k in keys:
                d = improvement(r["rmse"].get(k), base["rmse"].get(k))
                if d is not None:
                    gains.append(d)
        if gains:
            best, med = max(gains), float(np.median(gains))
            if best < 2.0:
                verdict = "REPRESENTATION-BOUND (provisional)"
                detail = (f"No larger surface beats the affine control by more than {best:+.1f}% on "
                          f"any cell (median {med:+.1f}%). On this evidence the frozen trunk does not "
                          f"carry recoverable cross-disease signal, and ANIL - which adds no capacity "
                          f"to the read-out - is unlikely to close a deficit of this size.")
            elif best >= 10.0:
                verdict = "ADAPTER-BOUND (provisional)"
                detail = (f"A larger surface recovers up to {best:+.1f}% (median {med:+.1f}%) over the "
                          f"affine control on the same trunk. The adaptation surface was a real "
                          f"bottleneck: there is a cheap partial fix available now, and a genuine "
                          f"prior that optimising the trunk for adaptability will pay.")
            else:
                verdict = "PARTIAL (provisional)"
                detail = (f"Best gain {best:+.1f}%, median {med:+.1f}%. Some capacity effect, but far "
                          f"short of the deficit. Neither reading is clean.")
    A(f"\n---\n\n## Read: **{verdict}**\n")
    A(f"\n{detail}\n")
    A("\n**Provisional, and here is exactly why.** One seed, one direction, one trunk. Head capacity "
      "only - a genuinely mid-trunk FiLM injection is not tested here because it requires changing "
      "the encoder forward pass, and this run deliberately changes nothing but the read-out. A null "
      "result bounds what a *read-out* can recover from this frozen representation; it does not prove "
      "no trunk can transfer. That is the correct bound for the ANIL question, because ANIL keeps the "
      "read-out small by construction.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(seed=SEED, verdict=verdict, results=results), indent=2))
    print(f"\n{'=' * 72}\nVERDICT: {verdict}\n{detail}\n{'=' * 72}")
    print(f"wrote {OUT_MD} and {OUT_JSON}")
    return verdict


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
    print("ok  4 surfaces emit [N,H,Q]; ladder strictly exceeds the 1,428-param control; "
          "MLP is genuinely non-affine; improvement sign correct")
    print(f"    ladder: {[f'{l}={s:,}' for (l, _), s in zip(SURFACES, sizes)]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--skip-fold", action="store_true",
                    help="reuse an existing trunk checkpoint instead of running stage 1")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    else:
        t0 = time.time()
        ck = stage1_fold(skip=a.skip_fold, verbose=not a.quiet)
        res = stage2_sweep(ck, verbose=not a.quiet)
        report(res)
        print(f"total {(time.time() - t0) / 60:.1f} min")

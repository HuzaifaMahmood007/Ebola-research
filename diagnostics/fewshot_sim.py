"""STEP 1: does the Ebola few-shot damage survive when you fix the adapter's starting point?

Step 0 (diagnostics/ebola_feature_shift.py, written up in
progress/outcomes/Ebola_Feature_Shift_2026-09-15.md) established that the Ebola adapter is fit on trunk
features drawn from windows that are 72.5% zero padding at 5.2% observation, then scored on full
windows at 31.6%, and that its design matrix is rank deficient at every horizon on the primary arm.
That is a measurement, not a counterfactual: it does not show that repairing the fit would forecast
better. This does.

It transplants the REAL ebola_L12 (or L20) support mask onto a held-out development panel and refits
the adapter four ways. Ebola is never loaded, never fitted on and never scored -- C8 is asserted below.

Step 0 found TWO degeneracies, so placement is crossed 2x2 rather than contrasted once:

    t0-blind   padded windows + blinded cells   = Ebola
    t0-dense   padded windows, cells intact     = padding only
    mid-blind  full windows + blinded cells     = sparsity only
    mid-clean  full windows, cells intact       = neither, the pure small-sample control

If `warm` beats `fresh` under mid-clean too, this is ordinary small-sample shrinkage, it says nothing
about Ebola, and the hypothesis is dead as an EXPLANATION even though the table looks like a win.

  conda run -n ebola-train python -m diagnostics.fewshot_sim --selfcheck
  conda run -n ebola-train python -m diagnostics.fewshot_sim --seed 42 --bundles covid_us-states
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import hashlib
import itertools
import json

import numpy as np
import torch

import bundles
import score
from bundles import HORIZONS, W
from models import Adapter, SharedEncoder, sparse_from_dense_np
from results_paths import RESULTS, rpath
from train.ebola import _fit_adapter, _precompute_features, choose_epochs
from train.lodo import _score
from train.loop import DEVICE, write_records

# Frozen truths this probe is built on. Asserted, never assumed.
TEMPLATES = {"L12": dict(src="ebola_L12", rows=18, width=13, n_origins=10,
                         pairs={3: 48, 5: 38, 10: 18, 15: 0}),
             "L20": dict(src="ebola_L20", rows=36, width=21, n_origins=18,
                         pairs={3: 102, 5: 92, 10: 72, 15: 54})}
PLACEMENTS = {"t0-blind": (0, True), "t0-dense": (0, False),
              "mid-blind": (40, True), "mid-clean": (40, False)}
ARMS = ("zeroshot", "fresh", "freshzs15", "warm")
NODE_DRAW_SEED = 0          # the support NODES are fixed across fit seeds, as Ebola's districts are
PREFIX = "fewshotsim"
# _fit_adapter(init_state=None) must stay byte-identical to the pre-init_state code. Captured from
# that code before the kwarg was added; see the docstring in train/ebola.py:_fit_adapter.
REFERENCE_FIT_SHA256 = "3a50afc3fd9fe6d5a1675e52dae9fff8ab93b0907a98775b26eb582a32fc7e5f"

FOLD_OF = {"covid_us-states": "covid", "influenza_japan": "influenza",
           "influenza_us-states": "influenza", "dengue": "dengue"}


# --------------------------------------------------------------------------- #
# The template, lifted from the frozen arm rather than invented.
# --------------------------------------------------------------------------- #
def template(name):
    """[rows, width] bool block: the real Ebola support pattern, rows that carry a cell."""
    spec = TEMPLATES[name]
    s = bundles.load(spec["src"]).masks()["support"].astype(bool)
    blk = s[np.where(s.any(1))[0]][:, :spec["width"]]
    assert blk.shape == (spec["rows"], spec["width"]), f"{name}: template is {blk.shape}"
    return blk


def pair_counts(smask, origins):
    """{h: n} adaptation pairs, counted the way the fit will actually consume them.

    A pair exists for (node i, origin t, horizon h) when the support mask has a cell at t+h. This is
    the same predicate targets_and_mask uses to build the loss weight (models/windows.py:32), so it is
    the row count of the affine problem, not a proxy."""
    T = smask.shape[1]
    return {h: int(sum(smask[:, t + h].sum() for t in origins if t + h < T)) for h in HORIZONS}


def transplant(b, name, t0, blind, rng):
    """Lay the template on `b` at column t0. Mutates the bundle we hold. Returns (nodes, origins).

    The origin set is PINNED to t0 .. t0+n_origins-1 rather than derived from the mask. Ebola's
    48/38/18/0 profile exists only because origins clip at t >= 0; dropping the same block at column 40
    would make every support cell reachable at every horizon (59/59/59/59), so a mid-placement would
    differ from t0 in DATA VOLUME as well as in feature quality and the contrast would be confounded.
    Pinning the count reproduces the profile at any t0 -- asserted below."""
    spec, blk = TEMPLATES[name], template(name)
    N, T = b.M.shape
    train = b.masks()["train"].astype(bool)
    assert spec["rows"] <= N, f"{b.name}: {N} nodes cannot host a {spec['rows']}-district template"

    nodes = np.sort(rng.choice(N, spec["rows"], replace=False))
    cols = np.arange(t0, t0 + spec["width"])
    smask = np.zeros((N, T), dtype=np.uint8)
    smask[np.ix_(nodes, cols)] = blk
    smask &= b.M.astype(np.uint8)                      # never invent an observation
    assert (smask.astype(bool) <= train).all(), "support escaped the train fold"

    if blind:
        # On Ebola a non-support cell inside the prefix was never observed: incidence 0, obs_mask 0.
        # sin_doy/cos_doy stay live there (measured on ebola_L12), so channels 1 and 2 are untouched.
        blk_cells = np.zeros((N, T), dtype=bool)
        blk_cells[np.ix_(nodes, cols)] = True
        blk_cells[:, cols] = True                      # the whole window, not just the chosen nodes
        kill = blk_cells & ~smask.astype(bool)
        b.M[kill] = 0
        b.y[kill] = 0.0
        b.X[..., 0][kill] = 0.0
        b.X[..., 3][kill] = 0.0
        smask &= b.M.astype(np.uint8)

    b._masks["support"] = smask                        # subset of train, NOT a fourth partition member
    origins = list(range(t0, t0 + spec["n_origins"]))
    got = pair_counts(smask.astype(bool), origins)
    assert got == spec["pairs"], f"{b.name} {name}@{t0}: pairs {got} != frozen {spec['pairs']}"
    return nodes, origins


# --------------------------------------------------------------------------- #
# Arms.
# --------------------------------------------------------------------------- #
def init_for(arm, seed, zero_sd, smask, origins, device):
    """The adapter state each arm starts from, or None for the protocol's fresh random init."""
    if arm == "fresh":
        return None
    if arm == "warm":
        return zero_sd
    # freshzs15: the fresh init everywhere EXCEPT horizons with no adaptation pair, which get the
    # zero-shot head. Those blocks are otherwise a uniform weight-decay shrink of random numbers
    # (train/ebola.py:195 _check_e4 proves it), so this separates "warm start helped the fit" from
    # "warm start stopped h15 being noise".
    torch.manual_seed(seed); np.random.seed(seed)      # same draw _fit_adapter will make
    sd = Adapter().to(device).state_dict()
    nQ, empty = len(score.QUANTILE_LEVELS), pair_counts(smask, origins)
    for j, h in enumerate(HORIZONS):
        if empty[h] == 0:
            r = slice(j * nQ, (j + 1) * nQ)
            sd["head.weight"][r] = zero_sd["head.weight"][r]
            sd["head.bias"][r] = zero_sd["head.bias"][r]
    return sd


def run_cell(b, enc, zero_ad, name, placement, arm, seed, device, verbose=True):
    """One (template, placement, arm) cell. Returns the record list from the panel's TEST fold."""
    Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=device)   # AFTER blinding
    ymod = torch.tensor(b.y, dtype=torch.float32, device=device)
    Mt = torch.tensor(b.M, dtype=torch.float32, device=device)
    A = sparse_from_dense_np(b.A_geo).to(device)
    smask_np = b.masks()["support"].astype(bool)
    smask = torch.tensor(smask_np, dtype=torch.float32, device=device)
    te = b.origins(phase="test")
    t0, _ = PLACEMENTS[placement]
    origins = list(range(t0, t0 + TEMPLATES[name]["n_origins"]))

    assert min(te) - (W - 1) > t0 + TEMPLATES[name]["width"], \
        f"a test window reaches the transplanted block (test origin {min(te)}, block ends " \
        f"{t0 + TEMPLATES[name]['width'] - 1}) -- this would be leakage"

    meta = dict(training_regime=f"fewshotsim_{arm}", template=name, placement=placement,
                arm=arm, t0=t0, blinded=PLACEMENTS[placement][1],
                sim_pairs=json.dumps(pair_counts(smask_np, origins)))

    if arm == "zeroshot":
        ad, epochs = zero_ad, 0
    else:
        feats = _precompute_features(enc, Z, A, Mt, origins, device)
        init = init_for(arm, seed, zero_ad.state_dict(), smask_np, origins, device)
        epochs, _, _ = choose_epochs(feats, ymod, Mt, smask, origins, seed, device,
                                     verbose=False, init_state=init)
        ad, _ = _fit_adapter(feats, ymod, Mt, smask, origins, seed, epochs, device, init_state=init)

    recs, _, _, _ = _score(enc, ad, b, Z, Mt, A, te, b.name, seed,
                           f"{PREFIX}_{arm}", meta, phase="test", device=device)
    write_records(recs, f"{PREFIX}__{b.name}__{name}__{placement}__{arm}__seed{seed}.json")
    if verbose:
        print(f"    {name:<4} {placement:<10} {arm:<10} epochs={epochs:<3} -> "
              f"{headline(recs, 'mae'):.4f} MAE")
    return recs


def headline(recs, metric, h=None):
    """country_macro, averaged over horizons unless one is named. The project's headline statistic.

    score_predictions emits LONG form -- one record per (metric, horizon) carrying a `metric` field
    and a flat `country_macro` -- not a wide record with nested metric dicts. Reading it the wide way
    silently yields nan for every cell rather than raising, which is exactly what it did on the first
    pilot run."""
    v = [r["country_macro"] for r in recs
         if r.get("metric") == metric and (h is None or r["horizon"] == h)]
    assert v, f"no {metric} records at horizon {h} -- record layout changed"
    return float(np.mean(v))


# --------------------------------------------------------------------------- #
def load_trunk(bundle, seed, device):
    """Encoder + the fold's mean adapter. Only encoder_ldo3full__* carries zeroshot_adapter."""
    fold = FOLD_OF[bundle]
    p = rpath(f"encoder_ldo3full__{fold}__seed{seed}__ckpt.pt", root=RESULTS)
    assert p.exists(), f"no trunk at {p}"
    ck = torch.load(p, map_location=device, weights_only=False)
    zs = ck["meta"].get("zeroshot_adapter")
    assert zs is not None, (
        f"{p.name} has no zeroshot_adapter, so its mean adapter cannot be recovered. Its `adapter` "
        f"slot is the adapter FITTED on the held-out disease's full train fold and using it as a warm "
        f"start would be leakage. That fold needs its trunk retrained (Step 2).")
    enc = SharedEncoder(gate_mode="learned").to(device); enc.load_state_dict(ck["encoder"]); enc.eval()
    ad = Adapter().to(device); ad.load_state_dict(zs); ad.eval()
    return enc, ad


def run(bundle_names, templates, placements, arms, seed, device):
    rows = []
    for bn in bundle_names:
        assert not bn.startswith("ebola"), "C8: Ebola must never enter a dev-fold fit"
        enc, zero_ad = load_trunk(bn, seed, device)
        print(f"\n{bn}  (trunk encoder_ldo3full__{FOLD_OF[bn]}__seed{seed})")
        for name, placement in itertools.product(templates, placements):
            for arm in arms:
                # freshzs15 == fresh wherever every horizon has pairs, so it is only run where it means
                # something. Running it on L20 would silently duplicate a row.
                if arm == "freshzs15" and min(TEMPLATES[name]["pairs"].values()) > 0:
                    continue
                b = bundles.load(bn)                    # FRESH bundle: blinding must not accumulate
                t0, blind = PLACEMENTS[placement]
                transplant(b, name, t0, blind, np.random.default_rng(NODE_DRAW_SEED))
                recs = run_cell(b, enc, zero_ad, name, placement, arm, seed, device)
                rows.append(dict(bundle=bn, template=name, placement=placement, arm=arm, seed=seed,
                                 **{f"{m}": headline(recs, m) for m in ("rmse", "mae", "pcc")},
                                 **{f"mae_h{h}": headline(recs, "mae", h) for h in HORIZONS}))
    return rows


def table(rows):
    for (bn, name) in sorted({(r["bundle"], r["template"]) for r in rows}):
        print(f"\n{bn}  template {name}   (country-macro MAE, lower is better)")
        print(f"{'placement':<11}{'zeroshot':>10}{'fresh':>10}{'freshzs15':>11}{'warm':>10}"
              f"{'warm-fresh':>12}{'verdict':>26}")
        print("-" * 90)
        for placement in PLACEMENTS:
            c = {r["arm"]: r for r in rows
                 if r["bundle"] == bn and r["template"] == name and r["placement"] == placement}
            if not c:
                continue
            g = lambda a: c[a]["mae"] if a in c else float("nan")
            zs, fr, wm = g("zeroshot"), g("fresh"), g("warm")
            d = (wm - fr) / fr * 100 if np.isfinite(fr) and fr else float("nan")
            if not np.isfinite(d):
                v = "-"
            elif d < -1:
                v = "warm helps"
            elif d > 1:
                v = "warm hurts"
            else:
                v = "no difference"
            if np.isfinite(fr) and np.isfinite(zs):
                v += ", adapt " + ("hurts" if fr > zs else "helps")
            print(f"{placement:<11}{zs:>10.4f}{fr:>10.4f}"
                  f"{(('%10.4f' % g('freshzs15')) if 'freshzs15' in c else '%10s' % '-'):>11}"
                  f"{wm:>10.4f}{d:>11.1f}%{v:>26}")
    print("""
read:
  zeroshot   the borrowed mean adapter, no fitting. the arm that wins on Ebola
  fresh      random init, the protocol Ebola_Prereg.md:126-127 fixes
  freshzs15  random init except horizons with 0 pairs, which get the zero-shot head (L12 only)
  warm       initialised from the mean adapter, then the same fit
  warm-fresh negative means warm-starting reduced error

  t0-blind reproduces Ebola (padded windows AND blinded cells). t0-dense isolates padding,
  mid-blind isolates sparsity, mid-clean is the small-sample control with neither.
  If warm helps under mid-clean too, this is ordinary shrinkage and does NOT explain Ebola.""")


# --------------------------------------------------------------------------- #
def selfcheck(device):
    # 1. the default fit path did not move when init_state was added
    torch.manual_seed(0)
    N, T, d = 8, 30, 64
    feats = {t: torch.randn(N, d) for t in range(4)}
    ad, _ = _fit_adapter(feats, torch.randn(N, T), torch.ones(N, T), torch.ones(N, T),
                         [0, 1, 2, 3], 42, 6, torch.device("cpu"))
    got = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                  for _, v in sorted(ad.state_dict().items()))).hexdigest()
    assert got == REFERENCE_FIT_SHA256, \
        f"_fit_adapter(init_state=None) CHANGED: {got} != {REFERENCE_FIT_SHA256}"
    print("  ok fit path unchanged by the init_state kwarg (sha256 matches pre-edit reference)")

    # 2/3. templates, and the pinned origin set reproducing the profile at BOTH placements
    for name, spec in TEMPLATES.items():
        blk = template(name)
        assert blk.sum() == {"L12": 59, "L20": 113}[name], f"{name}: {blk.sum()} cells"
        print(f"  ok {name} template {blk.shape}, {blk.sum()} cells / {spec['rows']} districts")
    for bn in ("covid_us-states",):
        for name, t0 in itertools.product(TEMPLATES, {p[0] for p in PLACEMENTS.values()}):
            b = bundles.load(bn)
            nodes, origins = transplant(b, name, t0, False, np.random.default_rng(NODE_DRAW_SEED))
            s = b.masks()["support"].astype(bool)
            assert (s <= b.masks()["train"].astype(bool)).all(), "support not inside train"
            assert min(b.origins(phase="test")) - (W - 1) > t0 + TEMPLATES[name]["width"]
            print(f"  ok {bn} {name}@t0={t0}: pairs {pair_counts(s, origins)}, "
                  f"support inside train, no test window reaches the block")

    # 4. blinding touches channels 0 and 3 only
    b0, b1 = bundles.load("covid_us-states"), bundles.load("covid_us-states")
    transplant(b1, "L12", 0, True, np.random.default_rng(NODE_DRAW_SEED))
    for ch in (1, 2):
        assert np.array_equal(b0.X[..., ch], b1.X[..., ch]), f"blinding altered channel {ch}"
    assert not np.array_equal(b0.X[..., 0], b1.X[..., 0]), "blinding changed nothing on channel 0"
    assert b1.M.sum() < b0.M.sum(), "blinding removed no observations"
    print(f"  ok blinding: channels 1,2 untouched; {int(b0.M.sum() - b1.M.sum())} cells unobserved")

    # 6. routing
    from results_paths import subdir_for
    f = f"{PREFIX}__covid_us-states__L12__t0-blind__warm__seed42.json"
    assert subdir_for(f) == "misc", f"{f} routes to {subdir_for(f)}, not misc"
    print("  ok output routes to results/misc/, colliding with no encoder_* family")
    print("selfcheck passed")


def rows_from_disk(seed):
    """Rebuild the summary from per-cell records already written, without refitting anything.

    The fits are the expensive half and they are already archived one JSON per cell, so a bug in the
    SUMMARY layer must never cost another run. This is also the reason run_cell writes each cell as it
    finishes rather than accumulating in memory."""
    import glob
    import re
    rows = []
    pat = re.compile(rf"{PREFIX}__(?P<b>.+?)__(?P<t>L\d+)__(?P<p>[\w-]+?)__(?P<a>\w+)__seed{seed}\.json$")
    for path in sorted(glob.glob(str(rpath(f"{PREFIX}__x.json", root=RESULTS).parent /
                                     f"{PREFIX}__*__seed{seed}.json"))):
        m = pat.search(path.replace("\\", "/"))
        if not m:
            continue                                   # the summary file itself
        recs = json.loads(open(path).read())
        rows.append(dict(bundle=m["b"], template=m["t"], placement=m["p"], arm=m["a"], seed=seed,
                         **{k: headline(recs, k) for k in ("rmse", "mae", "pcc")},
                         **{f"mae_h{h}": headline(recs, "mae", h) for h in HORIZONS}))
    assert rows, f"no per-cell records for seed {seed} in results/misc/"
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-disk", action="store_true",
                    help="rebuild the tables from per-cell records already written; fits nothing")
    ap.add_argument("--bundles", nargs="+", default=["covid_us-states"])
    ap.add_argument("--templates", nargs="+", default=list(TEMPLATES))
    ap.add_argument("--placements", nargs="+", default=list(PLACEMENTS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return selfcheck(DEVICE)
    if a.from_disk:
        rows = rows_from_disk(a.seed)
        table(rows)
        out = rpath(f"{PREFIX}__summary__seed{a.seed}.json", root=RESULTS, make=True)
        out.write_text(json.dumps(rows, indent=2))
        print(f"\nrebuilt from {len(rows)} per-cell records -> {out}")
        return
    selfcheck(DEVICE)
    rows = run(a.bundles, a.templates, a.placements, a.arms, a.seed, DEVICE)
    table(rows)
    out = rpath(f"{PREFIX}__summary__seed{a.seed}.json", root=RESULTS, make=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

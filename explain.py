"""G5 explainability: what the shared trunk reads, from the checkpoints already on disk.

Three reads, all inference-only against saved checkpoints. No training, no re-scoring, no new
results record is written or touched.

  1. Integrated gradients over the [N, 20, 4] trunk input (4 channels x 20 lags), per panel.
     Global reads are the share of |attribution| per channel and per lag BAND, per horizon. The
     20-lag array is archived but not reported: it combs with period 4 and an untrained encoder
     reproduces the comb, so single-lag resolution reads the TCN's dilations rather than
     epidemiology. report() runs that random-weight control and prints the correlation.
  2. Occlusion as the faithfulness cross-check: zero one channel or one lag and measure the
     change in the median forecast over the scored cells. IG and occlusion must agree on the
     top channel and the top lag band, or the disagreement is reported as a disagreement.
  2b. The Ebola local map targets that ONE district's own median forecast, not the sum over every
     scored node. The encoder mixes across neighbours, so a sum target says how a district's inputs
     move the national total, which is a different question and was the one the map used to answer.
  3. Neighbour edge ablation on Ebola: drop one edge, hold every node's degree fixed so only the
     message changes, and measure the change in each endpoint's forecast in cases/week. Answers
     "where does a forecast for a district we never observed draw from". It is framed as where
     the model draws from, never as what makes it accurate: the gate-off ablation says the graph
     helps error in 0 of 40 cells (Reports/gate_ablation.log).

Decision record: progress/decisions/G5_Explainability_Scope.md (integrated gradients, not SHAP).
The falsification test is stated there before any run: recent lags and incidence should dominate,
and the seasonality channels should carry more weight on influenza than on Ebola. If that ordering
does not appear, distrust the attribution before distrusting the epidemiology.

  python -m explain --selfcheck                          # synthetic model, no artifacts needed
  python -m explain --panels influenza_japan --seeds 42  # one panel-seed, ~25 s on a GPU
  python -m explain --all                                # 7 panel-arms x 5 seeds -> results/explain/
  python -m explain --report                             # archives -> tables, tests, figures/explain.*

Measured on one RTX GPU at the --all settings (24 origins, 32 IG steps), per seed: dengue 2.3 min
(7,165 nodes, and the only panel where IG dominates at 5 s an origin), each other development panel
about 0.4 min, ebola_L12 0.8 min and its zero-shot arm 0.5 min. --all is therefore about 26 min of
compute plus load/save, so run it in your own shell rather than inside a tool call.

ponytail: zero baseline throughout. In per-node z-space 0 is the node's training mean for
incidence, "unobserved" for obs_mask, and off the unit circle for sin/cos, which is exactly the
convention window_slice() already pads with. Upgrade path if a reviewer objects: a per-node
training-mean baseline, one line in integrated_gradients().
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP collision (see train/loop.py)

import argparse
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

import bundles
from bundles import DEV_BUNDLE_NAMES, HORIZONS, W
from models import (Adapter, MEDIAN_IDX, SharedEncoder, mask_aware_adj, normalise_adj,
                    sparse_from_dense_np, window_slice)
from results_paths import rpath

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (42, 52, 62, 72, 82)
CHANNELS = ("incidence", "sin_doy", "cos_doy", "obs_mask")
EBOLA_ARMS = ("ebola_L12", "ebola_L12_zeroshot")       # few-shot adapter, and the borrowed mean adapter
PANELS = tuple(DEV_BUNDLE_NAMES) + EBOLA_ARMS
IG_STEPS = 32
MAX_ORIGINS = 24                                          # evenly spaced over the scored fold
H = len(HORIZONS)
# window index w (0 = oldest) -> lag in weeks before the origin (1 = the origin week itself)
LAG_OF_INDEX = np.arange(W, 0, -1)
BANDS = ((1, 5), (6, 10), (11, 15), (16, 20))             # lag bands, most recent first
BAND_OF_INDEX = np.array([next(b for b, (lo, hi) in enumerate(BANDS) if lo <= lag <= hi)
                          for lag in LAG_OF_INDEX])
# The per-lag profile combs with period 4 on every panel. These are its teeth, and the random-weight
# control below measures how much of them is architecture rather than epidemiology.
COMB_SPIKES, COMB_TROUGHS = (1, 5, 9, 13, 17), (4, 8, 12, 16, 20)
# ponytail: the control is hard-wired to one panel-arm so it can run inside --report in about ten
# seconds. Ceiling: it bounds the fingerprint on ebola_L12 only, and a reviewer who wants the other
# six panel-arms has to change these constants and rerun.
# Five draws, not one: a single draw's r ranged 0.87 to 0.96 across torch seeds when I probed it, so
# one draw would let anyone quote the top of the range as if it were the number. The seeds are an
# arithmetic rule fixed before the run rather than seeds picked from the results.
RANDCTL_PANEL, RANDCTL_ORIGINS, RANDCTL_STEPS = "ebola_L12", 3, 8
RANDCTL_SEEDS = (1000, 2000, 3000, 4000, 5000)

# dataviz house tokens, light surface (same as gate_figure.py)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]              # blue <-> gray <-> red


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_model(panel, seed, device=DEVICE):
    """(enc, adapter, bundle name, checkpoint path), frozen and in eval mode."""
    if panel == "ebola_L12_zeroshot":
        ck, bname = rpath(f"encoder_ebola__alldev__seed{seed}__ckpt.pt"), "ebola_L12"
    elif panel.startswith("ebola_"):
        ck, bname = rpath(f"encoder_ebola__{panel}__seed{seed}__ckpt.pt"), panel
    else:
        ck, bname = rpath(f"encoder__{panel}__seed{seed}__ckpt.pt"), panel
    pl = torch.load(ck, map_location=device, weights_only=False)
    enc = SharedEncoder(gate_mode="learned").to(device)
    enc.load_state_dict(pl["encoder"])
    ad = Adapter().to(device)
    ad.load_state_dict(pl["adapter"])
    for m in (enc, ad):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return enc, ad, bname, ck


def tensors(b, device=DEVICE):
    Z = torch.tensor(b.transfer_view(), dtype=torch.float32, device=device)
    Mt = torch.tensor(b.M, dtype=torch.float32, device=device)
    A = sparse_from_dense_np(b.A_geo).to(device)
    return Z, Mt, A


def target_mask(b, phase, t, device=DEVICE):
    """[N, H]: the cells scored at origin t, exactly as train.loop.score_predictions folds them."""
    pm = b.masks()[phase]
    return torch.tensor(np.stack([pm[:, t + h] for h in HORIZONS], 1), dtype=torch.float32,
                        device=device)


def pick_origins(origins, k=MAX_ORIGINS):
    if len(origins) <= k:
        return list(origins)
    idx = np.unique(np.linspace(0, len(origins) - 1, k).round().astype(int))
    return [origins[i] for i in idx]


# --------------------------------------------------------------------------- #
# The three reads
# --------------------------------------------------------------------------- #
def median(enc, ad, Zt, A, Mt_t):
    return ad(enc(Zt, A, Mt_t))[:, :, MEDIAN_IDX]         # [N, H], model space


def integrated_gradients(enc, ad, Zt, A, Mt_t, tmask, steps=IG_STEPS):
    """Riemann-midpoint IG from a zero baseline, target = sum of the scored median forecasts per
    horizon. Returns attr [H, N, W, 4] and TWO completeness numbers per horizon, both measuring the
    same residual |sum(attr) - (f(x) - f(0))| against a different denominator:

      gap = residual / |f(x) - f(0)|   the standard IG completeness ratio.
      err = residual / (|f(x)| + |f(0)|)   the same residual against the size of the forecast.

    Both are reported because `gap` has a denominator that can collapse. Measured on ebola_L12
    seed 42: at h15 f(x) - f(0) falls to 0.17 while |f(x)| is 44, because that arm has no adaptation
    pairs at h15 and its head is near-constant, so `gap` reads 0.46 while the residual (7.9e-2) and
    `err` (8.9e-4) are in line with every other horizon. Reading `gap` alone there would say the
    attribution is broken when what is actually flat is the model. `err` is the one the selfcheck
    asserts on, since it is the scale-invariant statement that the path was integrated rather than
    approximated by a single gradient."""
    base = torch.zeros_like(Zt)
    grads = torch.zeros((H,) + tuple(Zt.shape), device=Zt.device)
    for k in range(steps):
        a = (k + 0.5) / steps
        Za = (base + a * (Zt - base)).detach().requires_grad_(True)
        f = median(enc, ad, Za, A, Mt_t)
        for j in range(H):
            g, = torch.autograd.grad((f[:, j] * tmask[:, j]).sum(), Za, retain_graph=j < H - 1)
            grads[j] += g
    attr = (Zt - base).unsqueeze(0) * grads / steps
    with torch.no_grad():
        fx = (median(enc, ad, Zt, A, Mt_t) * tmask).sum(0)
        f0 = (median(enc, ad, base, A, Mt_t) * tmask).sum(0)
    resid = (attr.flatten(1).sum(1) - (fx - f0)).abs()
    return attr, resid / (fx - f0).abs().clamp(min=1e-6), resid / (fx.abs() + f0.abs()).clamp(min=1e-6)


def occlusion(enc, ad, Zt, A, Mt_t, tmask):
    """Zero one channel (4) or one lag (20) at a time; mean |delta median| over the scored cells,
    per horizon. Same zero reference as IG, so the two reads are comparable. Returns
    (chan [H, 4], lag [H, W])."""
    with torch.no_grad():
        f = median(enc, ad, Zt, A, Mt_t)
        n = tmask.sum(0).clamp(min=1)

        def delta(Zo):
            return ((median(enc, ad, Zo, A, Mt_t) - f).abs() * tmask).sum(0) / n

        ch = torch.arange(4, device=Zt.device)
        chan = torch.stack([delta(Zt * (ch != c).float()) for c in range(4)], 1)
        wi = torch.arange(W, device=Zt.device).view(1, W, 1)
        lag = torch.stack([delta(Zt * (wi != w).float()) for w in range(W)], 1)
    return chan, lag


def forward_fixed_deg(enc, Zt, A_full, A_abl, Mt_t):
    """SharedEncoder.forward with the adjacency edited but the degree feature read from the FULL
    graph, so an edge ablation moves the neighbour message and nothing else. The gate reads h
    only, so it is fixed too. With A_abl is A_full this must equal enc(Zt, A_full, Mt_t) exactly,
    which the selfcheck asserts."""
    _, deg = normalise_adj(A_full)
    h = enc.tcn(Zt) + enc.ltr(deg)
    h_s = enc.spatial(h, mask_aware_adj(A_abl, Mt_t))
    h_out, _ = enc.gate(h, h_s)
    return h_out


def to_counts(f, scaler):
    """Model-space median [N, H] -> cases/week, the same inversion as to_schema.invert_scaler."""
    std = torch.tensor(scaler["std"], dtype=torch.float32, device=f.device).view(-1, 1)
    mean = torch.tensor(scaler["mean"], dtype=torch.float32, device=f.device).view(-1, 1)
    return torch.expm1(f * std + mean)


def undirected_edges(A_np):
    ii, jj = np.nonzero(np.triu(A_np, 1))
    return np.stack([ii, jj], 1)


def ablated(A_np, pairs, device=DEVICE):
    """Sparse adjacency with the given (i, j) entries zeroed both ways."""
    A2 = A_np.copy()
    for i, j in pairs:
        A2[i, j] = A2[j, i] = 0.0
    return sparse_from_dense_np(A2).to(device)


def edge_ablation(enc, ad, Zt, A, Mt_t, scaler, edges, A_edge, A_iso):
    """Per edge: change in each endpoint's count-space forecast when that one edge is dropped,
    [E, H, 2]. Per node: change when ALL its edges are dropped, [N, H]. Plus the full forecast
    [N, H] in counts."""
    with torch.no_grad():
        full = to_counts(ad(forward_fixed_deg(enc, Zt, A, A, Mt_t))[:, :, MEDIAN_IDX], scaler)
        ed = torch.zeros(len(edges), H, 2, device=Zt.device)
        for e, (i, j) in enumerate(edges):
            fa = to_counts(ad(forward_fixed_deg(enc, Zt, A, A_edge[e], Mt_t))[:, :, MEDIAN_IDX], scaler)
            ed[e, :, 0] = fa[i] - full[i]
            ed[e, :, 1] = fa[j] - full[j]
        iso = torch.zeros(Zt.shape[0], H, device=Zt.device)
        for i in range(Zt.shape[0]):
            fa = to_counts(ad(forward_fixed_deg(enc, Zt, A, A_iso[i], Mt_t))[:, :, MEDIAN_IDX], scaler)
            iso[i] = fa[i] - full[i]
    return ed, iso, full


# --------------------------------------------------------------------------- #
# One (panel, seed): run every read over the scored origins, archive
# --------------------------------------------------------------------------- #
def run_panel(panel, seed, device=DEVICE, max_origins=MAX_ORIGINS, steps=IG_STEPS, verbose=True):
    enc, ad, bname, ck = load_model(panel, seed, device)
    b = bundles.load(bname)
    phase = "query" if bname.startswith("ebola") else "test"
    all_origins = b.origins(phase=phase)
    origins = pick_origins(all_origins, max_origins)
    is_ebola = bname.startswith("ebola")
    if is_ebola:
        peak = int(b.raw.sum(0).argmax())
        t_case, d_case = peak - HORIZONS[0], int(b.raw[:, peak].argmax())
        # The local case study is the origin whose h3 target IS the national peak week. Subsampling
        # picks evenly spaced origins and has no reason to land on it, so without this it survives
        # only by luck and the local archive silently disappears at some --origins values.
        if t_case in all_origins and t_case not in origins:
            origins = sorted(set(origins) | {t_case})
    Z, Mt, A = tensors(b, device)
    K, N = len(origins), b.X.shape[0]
    out = dict(origins=np.array(origins, np.int32), horizons=np.array(HORIZONS, np.int32),
               ig_chan=np.zeros((K, H, 4), np.float64), ig_lag=np.zeros((K, H, W), np.float64),
               occ_chan=np.zeros((K, H, 4), np.float64), occ_lag=np.zeros((K, H, W), np.float64),
               gap=np.zeros((K, H), np.float64), err=np.zeros((K, H), np.float64),
               n_scored=np.zeros((K, H), np.int32), ig_steps=steps,
               checkpoint=str(ck), bundle=bname, panel=panel, seed=seed)
    if is_ebola:
        edges = undirected_edges(b.A_geo)
        A_edge = [ablated(b.A_geo, [tuple(e)], device) for e in edges]
        A_iso = [ablated(b.A_geo, [(i, j) for j in np.nonzero(b.A_geo[i])[0]], device)
                 for i in range(N)]
        support = b.masks()["support"]
        eout = dict(edges=edges.astype(np.int32), origins=out["origins"], horizons=out["horizons"],
                    edge_delta=np.zeros((len(edges), K, H, 2), np.float32),
                    iso_delta=np.zeros((N, K, H), np.float32), f_count=np.zeros((N, K, H), np.float32),
                    zero_shot=(support.sum(1) == 0), node_ids=np.array(b.meta["node_ids"]),
                    dates=np.array([str(d)[:10] for d in b.meta["dates"]]))
        local = None
    t0 = time.time()
    for k, t in enumerate(origins):
        Zt, Mt_t = window_slice(Z, t), Mt[:, t]
        tmask = target_mask(b, phase, t, device)
        attr, gap, err = integrated_gradients(enc, ad, Zt, A, Mt_t, tmask, steps)
        a = attr.abs()
        out["ig_chan"][k] = a.sum((1, 2)).cpu().numpy()
        out["ig_lag"][k] = a.sum((1, 3)).cpu().numpy()
        out["gap"][k], out["err"][k] = gap.cpu().numpy(), err.cpu().numpy()
        # how many cells each attribution target summed over, so a share can be read against the
        # amount of data behind it (it swings 13 to 46 across Ebola origins and horizons).
        out["n_scored"][k] = tmask.sum(0).cpu().numpy()
        oc, ol = occlusion(enc, ad, Zt, A, Mt_t, tmask)
        out["occ_chan"][k], out["occ_lag"][k] = oc.cpu().numpy(), ol.cpu().numpy()
        if is_ebola:
            ed, iso, full = edge_ablation(enc, ad, Zt, A, Mt_t, b.scaler, edges, A_edge, A_iso)
            eout["edge_delta"][:, k] = ed.cpu().numpy()
            eout["iso_delta"][:, k] = iso.cpu().numpy()
            eout["f_count"][:, k] = full.cpu().numpy()
            if t == t_case:
                # One extra IG pass, target one-hot on this district. The main pass targets the SUM
                # over every scored node, so attr[:, d_case] is how this district's inputs move the
                # NATIONAL total, not its own forecast: the encoder mixes across neighbours, so the
                # two are different quantities. The old map is kept beside the new one as evidence.
                # ponytail: hard-wired to the one case-study district. Ceiling: no general per-node
                # attribution API, and no second district gets a local map without editing this.
                tmask_local = torch.zeros_like(tmask)
                tmask_local[d_case] = 1.0
                attr_loc, _, _ = integrated_gradients(enc, ad, Zt, A, Mt_t, tmask_local, steps)
                local = dict(origin=t, district=d_case, name=b.meta["node_ids"][d_case], peak_week=peak,
                             peak_date=str(b.meta["dates"][peak])[:10],
                             ig_map=attr_loc[:, d_case].cpu().numpy(),      # [H, W, 4], signed, OWN target
                             ig_map_pooled=attr[:, d_case].cpu().numpy(),   # the old national-sum target
                             ig_target="own median forecast of this district, per horizon",
                             edge_delta=ed.cpu().numpy(), edges=edges.astype(np.int32),
                             f_count=full[d_case].cpu().numpy(),
                             truth_count=float(b.raw[d_case, peak]))
        if verbose and (k % 8 == 0 or k == K - 1):
            print(f"  {panel} seed {seed}: origin {k + 1}/{K}  IG err max {out['err'][:k + 1].max():.5f}"
                  f"  (gap max {out['gap'][:k + 1].max():.4f})  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(rpath(f"explain__{panel}__seed{seed}.npz", make=True), **out)
    if is_ebola:
        np.savez_compressed(rpath(f"explain__{panel}__seed{seed}__edges.npz", make=True), **eout)
        assert local is not None or t_case not in all_origins, \
            f"{panel}: origin {t_case} (h3 target = the national peak week) is scoreable but no local archive was built"
        if local is not None:
            np.savez_compressed(rpath(f"explain__{panel}__seed{seed}__local.npz", make=True), **local)
    return out


# --------------------------------------------------------------------------- #
# Report: archives -> shares, agreement, falsification tests, Ebola neighbours, figure
# --------------------------------------------------------------------------- #
# A main archive ends at the seed. The __edges and __local archives sit beside it and ALSO match a
# bare "...__seed*.npz" glob, so the main-archive reader has to exclude them by shape or it loads an
# edges file and dies on a missing ig_chan.
MAIN_ARCHIVE = re.compile(r"__seed\d+\.npz$")


def _load(pattern, only_main=False):
    paths = sorted(glob.glob(str(rpath(pattern))))
    if only_main:
        paths = [p for p in paths if MAIN_ARCHIVE.search(p)]
    return [dict(np.load(p, allow_pickle=True)) for p in paths]


def shares(arr):
    """[K, H, m] sums -> per-horizon share [H, m] pooled over the K origins."""
    s = arr.sum(0)
    return s / s.sum(1, keepdims=True).clip(min=1e-12)


def band_shares(lag_share):
    return np.stack([lag_share[:, BAND_OF_INDEX == b].sum(1) for b in range(len(BANDS))], 1)


def summarise(panel):
    """Per-seed shares for one panel, then mean and sd over seeds. Everything is [H, m]."""
    runs = _load(f"explain__{panel}__seed*.npz", only_main=True)
    if not runs:
        return None
    per = dict(ig_chan=[], ig_lag=[], occ_chan=[], occ_lag=[], gap=[], err=[], cells=[],
               origins=[], steps=[])
    for r in runs:
        per["ig_chan"].append(shares(r["ig_chan"])); per["ig_lag"].append(shares(r["ig_lag"]))
        per["occ_chan"].append(shares(r["occ_chan"])); per["occ_lag"].append(shares(r["occ_lag"]))
        per["gap"].append(r["gap"].max()); per["err"].append(r["err"].max())
        per["cells"].append(int(r["n_scored"].sum()))
        per["origins"].append(len(r["origins"])); per["steps"].append(int(r["ig_steps"]))
    s = {k: np.array(v) for k, v in per.items()}
    agg = {k: (s[k].mean(0), s[k].std(0, ddof=1) if len(runs) > 1 else np.zeros_like(s[k][0]))
           for k in ("ig_chan", "ig_lag", "occ_chan", "occ_lag")}
    agg["ig_band"] = np.array([band_shares(x) for x in s["ig_lag"]])
    agg["occ_band"] = np.array([band_shares(x) for x in s["occ_lag"]])
    # per-seed shares [S, H, m], kept so the report can print T1 and the agreement tally per
    # (seed, horizon) as well as on the seed mean. A seed-mean PASS can hide per-cell failures.
    agg["ig_chan_ps"], agg["occ_chan_ps"] = s["ig_chan"], s["occ_chan"]
    agg["gap_max"], agg["err_max"] = float(s["gap"].max()), float(s["err"].max())
    agg["cells"] = int(s["cells"].max())
    # the settings the numbers were produced at. A 6-origin/8-step smoke and a full 24/32 run write
    # the same filenames and print the same table, so the table has to say which one it is.
    agg["origins"], agg["steps"] = int(s["origins"].max()), int(s["steps"].max())
    agg["n_seeds"] = len(runs)
    return agg


def agreement(agg):
    """Per horizon: does occlusion pick the same top channel and top lag band as IG? Judged on
    the seed-mean shares."""
    ic, oc = agg["ig_chan"][0], agg["occ_chan"][0]
    ib, ob = agg["ig_band"].mean(0), agg["occ_band"].mean(0)
    return [(ic[j].argmax(), oc[j].argmax(), ib[j].argmax(), ob[j].argmax()) for j in range(H)]


def agreement_per_seed(agg):
    """The same two comparisons as agreement(), judged on each seed's own shares rather than the
    seed mean. Returns a bool array [S, H, 2] over (channel, band)."""
    return np.stack([agg["ig_chan_ps"].argmax(2) == agg["occ_chan_ps"].argmax(2),
                     agg["ig_band"].argmax(2) == agg["occ_band"].argmax(2)], -1)


def randctl_verdict(rs, null95):
    """(active, sentence) for the random-weight control. The whole point of the control is that an
    untrained encoder tracks the trained lag profile, so the report may only call the comb
    architectural when EVERY draw clears the shuffle ceiling. This is a function and not inline
    prose so the selfcheck can hand it a failing draw and watch the wording flip: as prose it would
    print "well above chance" whatever the numbers did, which is the exact mistake that let a TCN
    fingerprint be published as a memory finding."""
    if min(rs) > null95:
        return True, "Every draw clears that ceiling, so the comb is architectural, not learned."
    return False, (f"NOT every draw clears it (lowest r = {min(rs):.3f}), so the comb is NOT shown "
                   f"to be architectural.")


def random_control(panel=RANDCTL_PANEL, n_origins=RANDCTL_ORIGINS, steps=RANDCTL_STEPS,
                   seeds=RANDCTL_SEEDS, device=DEVICE):
    """Is the per-lag profile a property of the trained model, or of the architecture? Run IG with
    UNTRAINED SharedEncoder + Adapter weights at each of `seeds` and correlate each lag profile with
    the trained one. Needs the bundle but no checkpoint. Returns None if the archives are missing.

    Also returns a chance baseline: the 95th percentile of |r| between the trained profile and 200
    shuffles of itself. Without it a reader cannot tell whether r = 0.9 on a 20-point vector is
    large, and on this fixture it is not small (about 0.6)."""
    runs = _load(f"explain__{panel}__seed*.npz", only_main=True)
    if not runs:
        return None
    bname = "ebola_L12" if panel == "ebola_L12_zeroshot" else panel
    b = bundles.load(bname)
    phase = "query" if bname.startswith("ebola") else "test"
    Z, Mt, A = tensors(b, device)
    origins = pick_origins([int(t) for t in runs[0]["origins"]], n_origins)
    # index order -> lag order (lag 1 first) is a plain reverse, since LAG_OF_INDEX descends.
    trained = np.mean([shares(r["ig_lag"]).mean(0) for r in runs], 0)[::-1]
    teeth = lambda v: (float(np.mean([v[l - 1] for l in COMB_SPIKES])),
                       float(np.mean([v[l - 1] for l in COMB_TROUGHS])))
    rs, tr, rnd = [], [], None
    for sd in seeds:
        torch.manual_seed(sd)
        enc, ad = SharedEncoder(gate_mode="learned").to(device).eval(), Adapter().to(device).eval()
        for m in (enc, ad):
            for p in m.parameters():
                p.requires_grad_(False)
        lag = np.zeros((len(origins), H, W))
        for k, t in enumerate(origins):
            Zt, Mt_t = window_slice(Z, t), Mt[:, t]
            attr, _, _ = integrated_gradients(enc, ad, Zt, A, Mt_t, target_mask(b, phase, t, device), steps)
            lag[k] = attr.abs().sum((1, 3)).cpu().numpy()
        rnd = shares(lag).mean(0)[::-1]
        rs.append(float(np.corrcoef(trained, rnd)[0, 1]))
        tr.append(teeth(rnd))
    rng = np.random.default_rng(0)
    null = np.abs([np.corrcoef(rng.permutation(trained), rnd)[0, 1] for _ in range(200)])
    return dict(panel=panel, seeds=tuple(seeds), n_origins=len(origins), steps=steps,
                n_seeds=len(runs), r=rs, trained=trained,
                teeth_trained=teeth(trained), teeth_rnd=tuple(np.mean(tr, 0)),
                null95=float(np.percentile(null, 95)))


def ebola_neighbours(panel):
    """Seed-averaged neighbour influence on the Ebola arm. Returns per-node rows and edge table."""
    runs = _load(f"explain__{panel}__seed*__edges.npz")
    if not runs:
        return None
    ids, zs, edges = runs[0]["node_ids"], runs[0]["zero_shot"], runs[0]["edges"]
    N = len(ids)
    rel = np.zeros((len(runs), N))
    einf = np.zeros((len(runs), len(edges), 2))
    for r_i, r in enumerate(runs):
        rel[r_i] = np.abs(r["iso_delta"]).sum((1, 2)) / np.abs(r["f_count"]).sum((1, 2)).clip(min=1e-6)
        einf[r_i] = np.abs(r["edge_delta"]).mean((1, 2))
    rel_m, einf_m = rel.mean(0), einf.mean(0)
    rows = []
    for i in range(N):
        inc = [(einf_m[e, 0] if edges[e, 0] == i else einf_m[e, 1], edges[e, 1] if edges[e, 0] == i else edges[e, 0])
               for e in range(len(edges)) if i in edges[e]]
        if not inc:
            rows.append(dict(i=i, name=ids[i], zero_shot=bool(zs[i]), rel=rel_m[i], top=None, top_share=0.0,
                             top_observed=None, degree=0))
            continue
        tot = sum(v for v, _ in inc)
        v, j = max(inc)
        rows.append(dict(i=i, name=ids[i], zero_shot=bool(zs[i]), rel=rel_m[i], top=ids[j],
                         top_share=v / tot if tot > 0 else 0.0, top_observed=not bool(zs[j]), degree=len(inc)))
    return dict(rows=rows, n_seeds=len(runs), zero_shot=zs)


def report(out_txt="explain_report.txt", fig_out="figures/explain", randctl=True):
    lines = []
    P = lines.append
    P("G5 explainability: integrated gradients, occlusion cross-check, Ebola neighbour ablation")
    P("=" * 118)
    P("shares are |attribution| pooled over the scored origins, per horizon, then averaged over")
    P("horizons for this table; per-horizon values sit in results/explain/*.npz. Mean +- seed sd.")
    P("`gap` and `err` are the SAME IG completeness residual over two denominators, worst case over")
    P("origins, horizons and seeds: gap = residual / |f(x) - f(0)|, err = residual / (|f(x)|+|f(0)|).")
    P("Read err first. gap's denominator collapses wherever the model's own output barely moves off")
    P("the baseline, which is a flat model, not a broken attribution. `cells` is the scored count.")
    aggs = {p: summarise(p) for p in PANELS}
    aggs = {p: a for p, a in aggs.items() if a is not None}
    if not aggs:
        sys.exit("no explain archives found under results/explain/")

    P(f"\n{'panel':22} {'seeds':>5} {'orig':>4} {'step':>4} {'cells':>6} {'IG err':>7} {'IG gap':>7}  "
      + "  ".join(f"{c:>14}" for c in CHANNELS))
    for p, a in aggs.items():
        m, sd = a["ig_chan"][0].mean(0), a["ig_chan"][1].mean(0)
        P(f"{p:22} {a['n_seeds']:5d} {a['origins']:4d} {a['steps']:4d} {a['cells']:6d} "
          f"{a['err_max']:7.4f} {a['gap_max']:7.4f}  " +
          "  ".join(f"{m[c]:6.3f} +-{sd[c]:5.3f}" for c in range(4)))
    P("\nlag bands (weeks before the origin), IG share, then occlusion share in brackets")
    P(f"{'panel':22}  " + "  ".join(f"{f'{lo}-{hi}':>16}" for lo, hi in BANDS))
    for p, a in aggs.items():
        ib, ob = a["ig_band"].mean(0).mean(0), a["occ_band"].mean(0).mean(0)
        P(f"{p:22}  " + "  ".join(f"{ib[k]:6.3f} [{ob[k]:6.3f}]" for k in range(len(BANDS))))

    P("\nrandom-weight control on the per-lag read, and why lags are reported at band level")
    rc = random_control() if randctl else None
    if rc is None:
        P("  NOT RUN" + (" (--no-randctl)" if not randctl else
                         f" (no {RANDCTL_PANEL} archive to compare against)") +
          ". The band-level reporting below does not depend on it.")
    else:
        tt, st = rc["teeth_trained"], rc["teeth_rnd"]
        P(f"  IG on UNTRAINED SharedEncoder + Adapter weights (no checkpoint), {rc['panel']}, "
          f"{rc['n_origins']} origins, {rc['steps']} IG steps,")
        P(f"  torch seeds {', '.join(map(str, rc['seeds']))}, against the trained "
          f"{rc['n_seeds']}-seed profile on the same panel-arm.")
        P("  pearson r(trained lag profile, untrained lag profile) over the 20 lags, one per draw:")
        P("    " + ", ".join(f"{v:.3f}" for v in rc["r"]))
        P(f"  For scale, shuffling the trained profile against the same untrained one gives |r| below")
        P(f"  {rc['null95']:.2f} in 95 of 200 shuffles. {randctl_verdict(rc['r'], rc['null95'])[1]}")
        P("  The null on a 20-point vector is not small, which is why the ceiling is printed at all.")
        P(f"  comb teeth, mean share at lags {', '.join(map(str, COMB_SPIKES))} vs lags "
          f"{', '.join(map(str, COMB_TROUGHS))}:")
        P(f"    trained    {tt[0]:.3f} vs {tt[1]:.3f}   ({tt[0] / max(tt[1], 1e-9):.1f}x)")
        P(f"    untrained  {st[0]:.3f} vs {st[1]:.3f}   ({st[0] / max(st[1], 1e-9):.1f}x, "
          f"mean over {len(rc['seeds'])} draws)")
    rtxt = (f", and untrained encoders reproduce it at r = {min(rc['r']):.2f} to {max(rc['r']):.2f} "
            f"over {len(rc['seeds'])} draws" if rc else "")
    P("  Every panel's per-lag share combs with period 4, spiking at lags 1, 5, 9, 13 and 17 and")
    P(f"  collapsing at 4, 8, 12, 16 and 20{rtxt}.")
    # The conclusion follows the control's own verdict. Printed unconditionally it would say the comb
    # is architectural even on a run where the untrained encoder looked nothing like the trained one.
    if rc is None or randctl_verdict(rc["r"], rc["null95"])[0]:
        P("  Single-lag resolution therefore reads the dilated TCN's receptive field, not epidemiology,")
        P("  so the figure and the table above report lag BANDS only; the 20-lag arrays stay in")
        P("  results/explain/*.npz as the evidence. Band 16-20 exceeding band 11-15 is the dilation-16")
        P("  tap of DILATIONS = (1, 2, 4, 8, 16) reaching lag 17 in one hop, not a memory effect.")
        if rc is None:
            P("  That sentence rests on the earlier measured control, NOT on this run, which did not")
            P("  rerun it. Rerun without --no-randctl before quoting it.")
    else:
        P("  Single-lag resolution is NOT shown to be architectural by this run, so the claim about")
        P("  the dilated TCN's receptive field is withheld. The figure and the table above report lag")
        P("  BANDS only regardless, on the T2 comparison alone; the 20-lag arrays stay in")
        P("  results/explain/*.npz. Investigate before publishing either reading of the lag profile.")

    P("\nIG vs occlusion agreement on the top channel / top lag band, per horizon "
      "(a disagreement is reported, not resolved)")
    # one bool per printed comparison. argmax gives numpy ints, so `okc == oc` is a numpy bool and
    # `okc + okb` would be logical OR, not addition: that silently tallied 1 per cell instead of 2
    # and reported 12 of 24 on a smoke where every cell agreed. Sum Python ints, and tie the total
    # to the number of cells actually printed.
    marks, misses = [], []
    for p, a in aggs.items():
        cells = []
        for j, (ic, oc, ib, ob) in enumerate(agreement(a)):
            okc, okb = bool(ic == oc), bool(ib == ob)
            marks += [okc, okb]
            if not okc:
                misses.append(f"{p} h{HORIZONS[j]} channel (IG {CHANNELS[ic]} vs occlusion {CHANNELS[oc]})")
            if not okb:
                misses.append(f"{p} h{HORIZONS[j]} band (IG {BANDS[ib][0]}-{BANDS[ib][1]} vs "
                              f"occlusion {BANDS[ob][0]}-{BANDS[ob][1]})")
            cells.append(f"h{HORIZONS[j]}: {CHANNELS[ic][:4]}{'=' if okc else '!='}{CHANNELS[oc][:4]} "
                         f"{BANDS[ib][0]}-{BANDS[ib][1]}{'=' if okb else '!='}{BANDS[ob][0]}-{BANDS[ob][1]}")
        P(f"  {p:22} " + " | ".join(cells))
    n_ok, n_all = sum(int(m) for m in marks), len(marks)
    assert n_all == 2 * H * len(aggs), f"agreement tally counted {n_all}, expected {2 * H * len(aggs)}"
    # the seed mean is a vote of five, so a panel can agree on the mean while single seeds do not.
    ps = {p: agreement_per_seed(a) for p, a in aggs.items()}
    ps_ok, ps_all = sum(int(v.sum()) for v in ps.values()), sum(int(v.size) for v in ps.values())
    P(f"  agreement: seed-mean {n_ok} of {n_all}; per (seed, horizon) {ps_ok} of {ps_all}")
    P(f"  seed-mean disagreements: {'; '.join(misses) if misses else 'none'}")
    P("  per-seed disagreements by panel: " + (", ".join(
        f"{p} {int(v.size - v.sum())}" for p, v in ps.items() if int(v.sum()) < v.size) or "none"))

    P("\nfalsification tests, stated in G5_Explainability_Scope.md section 6 before the run")
    t1 = {p: CHANNELS[a["ig_chan"][0].mean(0).argmax()] for p, a in aggs.items()}
    P(f"  T1 incidence is the top channel on every panel, seed-mean and horizon-averaged: "
      f"{'PASS' if all(v == 'incidence' for v in t1.values()) else 'FAIL'}  "
      + ", ".join(f"{p}={v}" for p, v in t1.items()))
    # and the same test at the granularity it was measured at. Averaging over seeds and horizons
    # first can turn a set of failing cells into a PASS, so both are printed and the misses located.
    n_cells, by_p, by_h, by_w = 0, {}, {}, {}
    for p, a in aggs.items():
        for si in range(a["ig_chan_ps"].shape[0]):
            for j in range(H):
                n_cells += 1
                top = int(a["ig_chan_ps"][si, j].argmax())
                if top != 0:
                    by_p[p] = by_p.get(p, 0) + 1
                    by_h[HORIZONS[j]] = by_h.get(HORIZONS[j], 0) + 1
                    by_w[CHANNELS[top]] = by_w.get(CHANNELS[top], 0) + 1
    nbad = sum(by_p.values())
    srt = lambda d, f=str: ", ".join(f"{f(k)} {v}" for k, v in sorted(d.items(), key=lambda x: -x[1]))
    P(f"  T1 per (seed, horizon): {'PASS' if nbad == 0 else 'fails'} in {nbad} of {n_cells} cells"
      + ("" if nbad == 0 else f"; by panel {srt(by_p)}; by horizon {srt(by_h, lambda k: f'h{k}')}; "
                              f"winner when not incidence {srt(by_w)}"))
    t2 = {p: (a["ig_band"].mean(0).mean(0)[0], a["ig_band"].mean(0).mean(0)[-1]) for p, a in aggs.items()}
    P(f"  T2 lags 1-5 outweigh lags 16-20 on every panel: "
      f"{'PASS' if all(r > f for r, f in t2.values()) else 'FAIL'}  "
      + ", ".join(f"{p}={r:.2f}>{f:.2f}" for p, (r, f) in t2.items()))
    seas = {p: a["ig_chan"][0].mean(0)[1:3].sum() for p, a in aggs.items()}
    flu = [v for p, v in seas.items() if p.startswith("influenza")]
    ebo = [v for p, v in seas.items() if p.startswith("ebola")]
    if flu and ebo:
        P(f"  T3 seasonality share (sin+cos) on each influenza panel exceeds each Ebola arm: "
          f"{'PASS' if min(flu) > max(ebo) else 'FAIL'}  "
          + ", ".join(f"{p}={v:.3f}" for p, v in seas.items() if p.startswith(('influenza', 'ebola'))))
    else:
        # a pre-registered test that quietly disappears reads as a test that passed. Say it did not run.
        P(f"  T3 seasonality share, influenza vs Ebola: NOT RUN, needs both "
          f"(influenza archives: {len(flu)}, Ebola archives: {len(ebo)})")
    # Scope note 5.2 predicted this share would be "structurally zero" because a constant input has
    # no gradient. Measured, it is not: a zero-baseline path method charges obs_mask for the whole
    # move from "nothing observed" to "everything observed". Report the measurement, not the note.
    const = [p for p in aggs if p.startswith(("influenza", "covid"))]
    P("  obs_mask is constant 1.0 on the three influenza panels and COVID, so nothing in those panels")
    P("  varies it and no finding about the model can rest on its share there.")
    if const:
        ig_om = max(aggs[p]["ig_chan"][0].mean(0)[3] for p in const)
        oc_om = max(aggs[p]["occ_chan"][0].mean(0)[3] for p in const)
        P(f"  That share is NOT near zero: up to {ig_om:.3f} by IG and {oc_om:.3f} by occlusion on those")
        P("  panels. Both read from a zero baseline, so they charge obs_mask for the full 0 -> 1 move,")
        P("  a counterfactual those panels never contain. It is an artefact of the baseline, and the")
        P("  scope note's prediction that it would be structurally zero is refuted by this run.")

    for arm in EBOLA_ARMS:
        nb = ebola_neighbours(arm)
        if nb is None:
            continue
        rows = nb["rows"]
        zs = [r for r in rows if r["zero_shot"] and r["degree"] > 0]
        ob = [r for r in rows if not r["zero_shot"] and r["degree"] > 0]
        P(f"\nEbola neighbour ablation, {arm}, {nb['n_seeds']} seeds. Relative influence = "
          f"|forecast change with every edge of the district dropped| / |forecast|, cases/week,")
        P("pooled over origins and horizons. Degree held fixed, so only the neighbour message moves.")
        P(f"  zero-shot districts {len(zs)}: mean relative influence {np.mean([r['rel'] for r in zs]):.3f}, "
          f"median {np.median([r['rel'] for r in zs]):.3f}")
        P(f"  observed districts  {len(ob)}: mean relative influence {np.mean([r['rel'] for r in ob]):.3f}, "
          f"median {np.median([r['rel'] for r in ob]):.3f}")
        P(f"  zero-shot districts whose strongest neighbour is an OBSERVED district: "
          f"{sum(r['top_observed'] for r in zs)} of {len(zs)}")
        P(f"  {'district':28} {'rel':>6} {'deg':>3}  strongest neighbour (share of its edge influence)")
        for r in sorted(zs, key=lambda r: -r["rel"])[:10]:
            P(f"  {r['name']:28} {r['rel']:6.3f} {r['degree']:3d}  {r['top']} "
              f"({r['top_share']:.2f}, {'observed' if r['top_observed'] else 'zero-shot'})")
        P("  Read this as where the model draws from. The gate-off ablation says neighbour information")
        P("  helps error in 0 of 40 cells, so none of this may be described as a source of accuracy.")
        P("  That ablation zeroes the gate, which removes neighbour mixing but KEEPS the LTR degree")
        P("  feature, so the 0 of 40 bounds the value of neighbour information, not of the graph in total.")

    loc = _load("explain__ebola_L12__seed*__local.npz")
    if loc:
        assert "truth_count" in loc[0] and "ig_map_pooled" in loc[0], (
            "stale local archive: rerun `python -m explain --panels ebola_L12 ebola_L12_zeroshot`")
        # The published map has to be the per-node one. Both targets are archived side by side, so
        # this is a check on the artifact the figure reads, not only on the selfcheck's fixture. If
        # they matched, panel C would still be showing the national-sum attribution under a new title.
        assert not np.allclose(loc[0]["ig_map"], loc[0]["ig_map_pooled"], atol=1e-8), \
            "local ig_map equals ig_map_pooled, so panel C is still the national-sum map"
        fc, truth = np.mean([l["f_count"] for l in loc], 0), float(loc[0]["truth_count"])
        P(f"\nEbola local case, ebola_L12, {len(loc)} seeds. The IG target here is the district's OWN")
        P("median forecast per horizon, not the sum over every scored node. The encoder mixes across")
        P("neighbours, so those are different quantities, and the sum-target map answered a different")
        P("question: how this district's inputs move the national total.")
        P(f"  {str(loc[0]['name'])}, origin index {int(loc[0]['origin'])}, h3 target week "
          f"{str(loc[0]['peak_date'])}, the national peak")
        P("  own forecast, cases/week, seed mean:  "
          + "  ".join(f"h{HORIZONS[j]}={fc[j]:.1f}" for j in range(H)))
        P(f"  observed that week: {truth:,.0f} cases. Most of that gap is gap-lumping, not model error")
        P("  alone: a district that falls silent and then files puts the whole multi-week increment on")
        P(f"  the reporting week (client_decisions.md A4), and this one carries {truth:,.0f} on")
        P(f"  {str(loc[0]['peak_date'])}. The peaks are inflated and the quiet weeks either side flattened.")

    text = "\n".join(lines)
    print(text)
    rpath(out_txt, make=True).write_text(text, encoding="utf-8")
    print(f"\n  wrote {rpath(out_txt)}")
    figure(aggs, fig_out)


def figure(aggs, out="figures/explain"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.8})
    seq = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    div = LinearSegmentedColormap.from_list("div", DIVERGING)
    panels = list(aggs)
    short = [p.replace("influenza_", "flu ").replace("covid_us-states", "covid us-states")
             .replace("ebola_L12_zeroshot", "ebola L12 zero-shot").replace("ebola_L12", "ebola L12 few-shot")
             for p in panels]

    def pretty(node_id):
        """Bundle node id -> a name a reader can say out loud: 'liberia|montserrado' becomes
        'Montserrado, Liberia'. Country is first in the id and last in the label. Deeper ids keep
        their leaf as the place ('brazil|sao paulo|sao paulo' -> 'Sao Paulo, Brazil')."""
        parts = [p.strip() for p in str(node_id).split("|") if p.strip()]
        if len(parts) < 2:
            return str(node_id).title()
        return f"{parts[-1].title()}, {parts[0].title()}"
    # Two independent grids, not one 2x2. The rows need different column geometry: A/B are heatmaps
    # that want a tight gap, while D's y labels are full place names ("Gbarpolu, Liberia
    # (zero-shot)") that hang left off its axis and land on panel C if the gap is shared with the top
    # row. Positioning the grids here also means no subplots_adjust later, which is what used to move
    # the panels out from under their colourbars.
    # 9.0 in, not 8.6: the caption is 7 lines now (band-level note, and C's forecast against the
    # observed truth), and at 8.6 the last line ran into panel C's colourbar label.
    fig = plt.figure(figsize=(13.0, 9.0))
    # wspace on the top row has to hold BOTH panel A's colourbar and panel B's row labels, which are
    # the same panel names as A's and hang left off B's axis. 0.14 fits the colourbar alone.
    gs_top = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.40,
                              left=0.115, right=0.945, top=0.955, bottom=0.645)
    gs_bot = fig.add_gridspec(1, 2, width_ratios=[1, 1.9], wspace=0.44,
                              left=0.115, right=0.945, top=0.535, bottom=0.315)
    axA, axB = fig.add_subplot(gs_top[0, 0]), fig.add_subplot(gs_top[0, 1])
    axC, axD = fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])

    # A: channel shares, panels x channels
    M = np.array([aggs[p]["ig_chan"][0].mean(0) for p in panels])
    vA = max(0.5, M.max())
    imA = axA.imshow(M, cmap=seq, vmin=0, vmax=vA, aspect="auto")
    for r in range(M.shape[0]):
        for c in range(4):
            axA.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=8,
                     color=SURFACE if M[r, c] > 0.45 * vA else INK)
    axA.set_xticks(range(4)); axA.set_xticklabels(CHANNELS, fontsize=8)
    axA.set_yticks(range(len(panels))); axA.set_yticklabels(short, fontsize=8)
    axA.set_title("A · share of IG attribution by input channel", loc="left", color=INK, pad=8)
    axA.tick_params(length=0)

    # B: lag shares at BAND level, panels x 4 bands. NOT single-lag: the 20-lag profile combs with
    # period 4 on every panel and an untrained encoder reproduces it (the random-weight control in
    # report()), so single lags read the TCN's dilation pattern, not epidemiology. The 20-lag arrays
    # stay in results/explain/*.npz. Its OWN colourbar: a share over 4 bands and a share over 4
    # channels are the same quantity on different supports and land on different scales.
    L = np.array([aggs[p]["ig_band"].mean(0).mean(0) for p in panels])      # [P, 4], seeds, horizons
    vB = L.max()
    imB = axB.imshow(L, cmap=seq, vmin=0, vmax=vB, aspect="auto")
    for r in range(L.shape[0]):
        for c in range(len(BANDS)):
            axB.text(c, r, f"{L[r, c]:.2f}", ha="center", va="center", fontsize=8,
                     color=SURFACE if L[r, c] > 0.45 * vB else INK)
    axB.set_xticks(range(len(BANDS)))
    axB.set_xticklabels([f"{lo}-{hi}" for lo, hi in BANDS], fontsize=8)
    axB.set_yticks(range(len(panels))); axB.set_yticklabels(short, fontsize=8)
    axB.set_xlabel("weeks before the forecast origin (band 1-5 holds the origin week)")
    axB.set_title("B · share of IG attribution by lag band, per panel", loc="left", color=INK, pad=8)
    axB.tick_params(length=0)

    # C and D: the Ebola local case, few-shot arm, seed-averaged
    loc = _load("explain__ebola_L12__seed*__local.npz")
    if loc:
        ig = np.mean([l["ig_map"] for l in loc], 0)[0]                     # h3, [W, 4] signed, OWN target
        name, date = str(loc[0]["name"]), str(loc[0]["peak_date"])
        fc3 = float(np.mean([l["f_count"] for l in loc], 0)[0])            # this district's own h3
        truth = float(loc[0]["truth_count"])                                # observed, same week
        v = np.abs(ig).max()
        imC = axC.imshow(ig.T[:, ::-1], cmap=div, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
        axC.set_yticks(range(4)); axC.set_yticklabels(CHANNELS, fontsize=8)
        axC.set_xticks(range(0, W, 2)); axC.set_xticklabels([str(l) for l in range(1, W + 1, 2)], fontsize=7)
        axC.set_xlabel("weeks before the origin")
        # two lines, and the place name on the first: at one line this title reached into panel D's.
        axC.set_title(f"C · signed IG on {pretty(name)}'s OWN h3 forecast\n"
                      f"target week {date}: forecast {fc3:.1f} cases/week against {truth:,.0f} observed",
                      loc="left", color=INK, pad=8)
        axC.tick_params(length=0)
        edges, d = loc[0]["edges"], int(loc[0]["district"])
        ed = np.mean([l["edge_delta"] for l in loc], 0)                     # [E, H, 2]
        nb = ebola_neighbours("ebola_L12")
        ids, zs = np.array([r["name"] for r in nb["rows"]]), nb["zero_shot"]
        inc = [(abs(ed[e, 0, 0]) if edges[e, 0] == d else abs(ed[e, 0, 1]),
                edges[e, 1] if edges[e, 0] == d else edges[e, 0]) for e in range(len(edges)) if d in edges[e]]
        inc.sort()
        vals = [v for v, _ in inc]
        names = [f"{pretty(ids[j])} ({'zero-shot' if zs[j] else 'observed'})" for _, j in inc]
        axD.barh(range(len(vals)), vals, color=BLUE_RAMP[7], height=0.62)
        for k, val in enumerate(vals):
            # 3 decimals: on the case district these run 0.920 down to exactly 0, and ".1f" printed
            # the small ones as "0.0", which is a different statement from a measured zero.
            axD.text(val, k, f"  {val:.3f}", va="center", fontsize=8, color=INK2)
        axD.set_yticks(range(len(vals))); axD.set_yticklabels(names, fontsize=8)
        axD.set_xlabel("absolute change in the h3 forecast when that one edge is dropped, cases/week")
        axD.set_xlim(0, max(vals) * 1.35 if vals else 1)
        axD.set_title(f"D · neighbour influence on {pretty(name)}, edge ablation", loc="left",
                      color=INK, pad=8)
        axD.xaxis.grid(True); axD.set_axisbelow(True)
        for sp in ("top", "right"):
            axD.spines[sp].set_visible(False)
        # a bar at exactly 0.000 reads as a broken chart unless the reason is on the page. It is a
        # neighbour that filed nothing that week: mask_aware_adj weights edge (i, j) by the
        # neighbour's own observation mask, so that message is already zero before any ablation.
        zero_txt = ("A bar at exactly zero is a neighbour that filed no report that week: each edge is "
                    "weighted by the neighbour's own observation mask, so its message is already "
                    "zero and dropping it changes nothing.\n" if vals and min(vals) == 0 else "")
        c_txt = (f"C and D are one district at the national peak week. C targets that district's OWN "
                 f"median forecast, not the national sum: its h3 forecast is {fc3:.1f} cases/week "
                 f"against {truth:,.0f} observed for {date}.\n"
                 f"That week carries a whole multi-week reporting increment (gap-lumping, "
                 f"client_decisions.md A4), so the observed peak is inflated and the quiet weeks either "
                 f"side are flattened.\n" + zero_txt)
    else:
        c_txt = "C and D are empty: no Ebola local archive in this run.\n"
        imC = None
        for ax in (axC, axD):
            ax.text(0.5, 0.5, "no Ebola local archive yet", ha="center", va="center", color=MUTED)
            ax.set_axis_off()

    # Counted from the archives actually loaded, never hard-coded: this caption is the figure's own
    # claim about its evidence, and a subset run must not inherit a full run's sentence.
    n_dev = sum(1 for p in panels if not p.startswith("ebola"))
    n_ebo = sum(1 for p in panels if p.startswith("ebola"))
    n_seeds = sorted({aggs[p]["n_seeds"] for p in panels})
    const = [p for p in panels if p.startswith(("influenza", "covid"))]
    om_txt = (f"up to {max(aggs[p]['ig_chan'][0].mean(0)[3] for p in const):.2f}"
              if const else "not measured here")
    plural = lambda n, w: f"{n} {w}" + ("" if n == 1 else "s")
    seed_txt = plural(n_seeds[0], "seed") if len(n_seeds) == 1 else f"{min(n_seeds)}-{max(n_seeds)} seeds"
    fig.text(0.008, 0.012,
             f"Integrated gradients from a zero baseline over the [N, 20, 4] trunk input, from the saved "
             f"checkpoints: {plural(n_dev, 'development panel')} and {plural(n_ebo, 'Ebola L12 arm')}, "
             f"{seed_txt}. A and B are shares of absolute IG attribution, pooled over scored origins "
             f"and horizons.\n"
             f"B is reported at lag BAND level: the 20-lag profile combs with period 4 on every panel and "
             f"untrained encoders reproduce the comb, so single-lag resolution reads the dilated TCN's "
             f"receptive field, not epidemiology.\n"
             f"obs_mask is constant on the three influenza panels and COVID, so nothing there varies it. Its "
             f"share is still large ({om_txt}) because a zero baseline charges the channel for the whole "
             f"0 -> 1 move: an artefact of the baseline, not a use of observation status.\n"
             + c_txt +
             "Edge ablation holds every degree fixed, so only the neighbour message moves. None of D is a "
             "source of accuracy: the gate-off ablation shows neighbour information helps error in 0 of 40 "
             "cells.\n"
             "That ablation zeroes the gate but keeps the LTR degree feature, so the tally bounds neighbour "
             "information, not the graph in total. D shows where the model draws from, and nothing more.",
             fontsize=7.5, color=MUTED, va="bottom", linespacing=1.6)

    # Colourbars LAST, positioned from the final axes boxes. fig.colorbar(ax=[...]) steals space at
    # creation time and anything that re-lays-out the panels afterwards slides them out from under
    # it: that is what put the shared bar on top of lags 18-20 of panel B.
    for ax, mappable, label in ((axA, imA, "share of attribution"), (axB, imB, "share of attribution")):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.007, pos.y0, 0.0075, pos.height])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label(label, color=INK2, fontsize=8)
        cb.ax.tick_params(labelsize=7, length=0)
        cb.outline.set_edgecolor(AXIS)
    if imC is not None:
        # C's bar goes UNDERNEATH it. To the right it lands squarely on panel D's district names,
        # which are long enough ("Gbarpolu, Liberia (zero-shot)") to reach into the column gap.
        pos = axC.get_position()
        cax = fig.add_axes([pos.x0, pos.y0 - 0.075, pos.width * 0.55, 0.013])
        cb = fig.colorbar(imC, cax=cax, orientation="horizontal")
        cb.set_label("signed IG (h3, this district)", color=INK2, fontsize=8)
        cb.ax.tick_params(labelsize=7, length=0)
        cb.outline.set_edgecolor(AXIS)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"  wrote {out}.png and {out}.pdf")


# --------------------------------------------------------------------------- #
# Selfcheck: synthetic model, no artifacts
# --------------------------------------------------------------------------- #
def _selfcheck():
    torch.manual_seed(0)
    dev = DEVICE
    N = 6
    A_np = np.zeros((N, N))
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]:
        A_np[i, j] = A_np[j, i] = 1.0
    A = sparse_from_dense_np(A_np).to(dev)
    Zt = torch.randn(N, W, 4, device=dev)
    Mt_t = torch.ones(N, device=dev)
    tmask = torch.ones(N, H, device=dev)
    enc, ad = SharedEncoder().to(dev).eval(), Adapter().to(dev).eval()
    for m in (enc, ad):
        for p in m.parameters():
            p.requires_grad_(False)

    # (1) IG integrates the path. Assert on `err`, not on `gap`: this net is untrained, so f(x)-f(0)
    #     collapses to 0.002 at h15 while it is 0.36 at h3, and the ratio there is meaningless. The
    #     residual itself is 1e-4 at every horizon and does not move between 8 and 512 steps.
    attr, gap, err = integrated_gradients(enc, ad, Zt, A, Mt_t, tmask, steps=64)
    assert attr.shape == (H, N, W, 4), attr.shape
    assert attr.abs().sum() > 0
    assert (err < 1e-3).all(), f"IG completeness error {err.tolist()}"
    #     and the check discriminates: gradient x input, the un-integrated shortcut this whole
    #     module exists to avoid, must FAIL the same assertion on the same scale.
    with torch.no_grad():
        fx = (median(enc, ad, Zt, A, Mt_t) * tmask).sum(0)
        f0 = (median(enc, ad, torch.zeros_like(Zt), A, Mt_t) * tmask).sum(0)
    Za = Zt.detach().requires_grad_(True)
    g, = torch.autograd.grad((median(enc, ad, Za, A, Mt_t)[:, 0] * tmask[:, 0]).sum(), Za)
    shortcut = ((Zt * g).sum() - (fx - f0)[0]).abs() / (fx.abs() + f0.abs())[0]
    assert shortcut > 1e-3, f"gradient x input passed the completeness check ({shortcut:.2e}), so it cannot discriminate"

    # (2) a channel the model structurally cannot see gets exactly zero attribution AND zero
    #     occlusion delta, while a visible channel gets both. Control for both reads at once.
    enc.tcn.inp.weight[:, 3, :] = 0.0
    attr, _, _ = integrated_gradients(enc, ad, Zt, A, Mt_t, tmask, steps=8)
    assert attr[..., 3].abs().max() == 0.0, "IG attributed to an invisible channel"
    assert attr[..., 0].abs().max() > 0.0
    oc, ol = occlusion(enc, ad, Zt, A, Mt_t, tmask)
    assert oc.shape == (H, 4) and ol.shape == (H, W)
    assert oc[:, 3].abs().max() == 0.0, "occlusion moved on an invisible channel"
    assert oc[:, 0].min() > 0.0

    # (3) forward_fixed_deg is the real forward when nothing is ablated; with the gate off, no
    #     edge ablation can move anything; with it learned, some edge does.
    with torch.no_grad():
        ref, mine = enc(Zt, A, Mt_t), forward_fixed_deg(enc, Zt, A, A, Mt_t)
    assert torch.allclose(ref, mine, atol=1e-6), "forward_fixed_deg drifted from SharedEncoder.forward"
    edges = undirected_edges(A_np)
    A_edge = [ablated(A_np, [tuple(e)], dev) for e in edges]
    A_iso = [ablated(A_np, [(i, j) for j in np.nonzero(A_np[i])[0]], dev) for i in range(N)]
    scaler = {"mean": np.zeros(N), "std": np.ones(N)}
    off = SharedEncoder(gate_mode="off").to(dev).eval()
    off.load_state_dict({k: v for k, v in enc.state_dict().items() if not k.startswith("gate.")}, strict=False)
    ed, iso, full = edge_ablation(off, ad, Zt, A, Mt_t, scaler, edges, A_edge, A_iso)
    assert ed.abs().max() == 0.0 and iso.abs().max() == 0.0, "gate-off model moved under edge ablation"
    ed, iso, full = edge_ablation(enc, ad, Zt, A, Mt_t, scaler, edges, A_edge, A_iso)
    assert ed.shape == (len(edges), H, 2) and iso.shape == (N, H) and full.shape == (N, H)
    assert ed.abs().max() > 0.0, "learned gate but no edge moved anything"
    # the degree feature really is held. normalise_adj on the isolated graph gives a DIFFERENT
    # degree, so a plain enc(Zt, A_abl, Mt) would move the LTR feature as well as the message and
    # the read would confound the two. forward_fixed_deg must therefore differ from that plain call.
    with torch.no_grad():
        _, deg_full = normalise_adj(A)
        _, deg_iso = normalise_adj(A_iso[0])
        fixed, plain = forward_fixed_deg(enc, Zt, A, A_iso[0], Mt_t), enc(Zt, A_iso[0], Mt_t)
    assert not torch.allclose(deg_full, deg_iso), "isolation control void: degree did not change"
    assert not torch.allclose(fixed, plain), "forward_fixed_deg is reading degree from the ablated graph"

    # (4) lag bookkeeping: window index W-1 is the origin week (lag 1, band 0); index 0 is lag 20.
    assert LAG_OF_INDEX[W - 1] == 1 and LAG_OF_INDEX[0] == W
    assert BAND_OF_INDEX[W - 1] == 0 and BAND_OF_INDEX[0] == len(BANDS) - 1
    assert np.bincount(BAND_OF_INDEX).tolist() == [5, 5, 5, 5]
    # shares: pooled over origins, sum to one per horizon; bands sum to one too.
    x = np.random.rand(3, H, W)
    s = shares(x)
    assert np.allclose(s.sum(1), 1.0) and np.allclose(band_shares(s).sum(1), 1.0)
    # a pooled share is NOT the mean of per-origin shares: origin 0 carrying 100x the mass must
    # dominate. Getting this backwards would weight a near-empty origin like a full one.
    y = np.zeros((2, H, 4)); y[0, :, 0] = 100.0; y[1, :, 1] = 1.0
    assert shares(y)[0, 0] > 0.98, shares(y)[0]

    # (4b) band bookkeeping, now that the figure and the table report bands and NOT single lags.
    #      Each band must hold exactly the lags it names, and band_shares must route a lag to the
    #      band that names it. A silent off-by-one here would mislabel the published panel B.
    for k, (lo, hi) in enumerate(BANDS):
        assert sorted(LAG_OF_INDEX[BAND_OF_INDEX == k].tolist()) == list(range(lo, hi + 1)), k
    for lag in (1, 5, 6, 15, 16, 17, 20):
        one = np.zeros((1, H, W))
        one[0, :, int(np.nonzero(LAG_OF_INDEX == lag)[0][0])] = 1.0
        k = next(k for k, (lo, hi) in enumerate(BANDS) if lo <= lag <= hi)
        got = band_shares(shares(one))
        assert got[0, k] == 1.0 and np.allclose(got[0].sum(), 1.0), (lag, k, got[0])
    # a lag-order profile is the index-order profile reversed, which is what figure() and
    # random_control() both rely on. LAG_OF_INDEX descends, so this is the whole proof.
    assert (LAG_OF_INDEX[::-1] == np.arange(1, W + 1)).all()

    # (4bb) the random-weight control must be able to say NO. The report's sentence "the comb is
    #       architectural" is generated by randctl_verdict(), so hand it a failing draw and check
    #       the wording flips and names the offending r. As inline prose this sentence printed
    #       whatever the numbers did, which is how a TCN fingerprint got published as memory.
    assert randctl_verdict([0.912, 0.963], 0.63)[0] is True
    assert randctl_verdict([0.55, 0.963], 0.63)[0] is False, "control cannot fail, so it checks nothing"
    assert "0.550" in randctl_verdict([0.55, 0.963], 0.63)[1]
    assert randctl_verdict([0.63, 0.99], 0.63)[0] is False, "verdict must be strict at the ceiling"

    # (4c) the LOCAL Ebola map targets ONE district's own forecast, not the sum over every scored
    #      node. This is the assertion that would have caught the published panel C attributing the
    #      national total to a single district. Two halves: a one-hot target completes against that
    #      node's own f(x) - f(0), and it does NOT complete against the pooled one, so the check
    #      can tell the two apart. It also must not reproduce the pooled map for that node.
    d = 2
    tm_local = torch.zeros(N, H, device=dev)
    tm_local[d] = 1.0
    attr_loc, _, _ = integrated_gradients(enc, ad, Zt, A, Mt_t, tm_local, steps=64)
    with torch.no_grad():
        f_all = median(enc, ad, Zt, A, Mt_t)
        f0_all = median(enc, ad, torch.zeros_like(Zt), A, Mt_t)
    tot = attr_loc.flatten(1).sum(1)
    own = (tot - (f_all[d] - f0_all[d])).abs() / (f_all[d].abs() + f0_all[d].abs()).clamp(min=1e-6)
    pooled = ((tot - ((f_all - f0_all) * tmask).sum(0)).abs()
              / ((f_all * tmask).sum(0).abs() + (f0_all * tmask).sum(0).abs()).clamp(min=1e-6))
    assert (own < 1e-3).all(), f"local IG does not complete on the district's own forecast {own.tolist()}"
    assert pooled.max() > 1e-3, "check is void: the pooled and own-node targets agree on this fixture"
    attr_pool, _, _ = integrated_gradients(enc, ad, Zt, A, Mt_t, tmask, steps=64)
    assert not torch.allclose(attr_loc[:, d], attr_pool[:, d], atol=1e-6), \
        "the local map equals the pooled map for that node, so the local target is not local"

    # (5) routing: the archive lands in its own subdir, the text report in reports/. And the
    #     main-archive filter keeps the per-origin archive while rejecting the two Ebola side
    #     archives, which a bare "__seed*.npz" glob would hand to the share reader instead.
    assert rpath("explain__dengue__seed42.npz").parent.name == "explain"
    assert rpath("explain__ebola_L12__seed42__edges.npz").parent.name == "explain"
    assert rpath("explain_report.txt").parent.name == "reports"
    assert MAIN_ARCHIVE.search("explain__ebola_L12__seed42.npz")
    assert not MAIN_ARCHIVE.search("explain__ebola_L12__seed42__edges.npz")
    assert not MAIN_ARCHIVE.search("explain__ebola_L12__seed42__local.npz")

    # (6) to_counts matches the schema inversion.
    from to_schema import invert_scaler
    f = torch.randn(N, H, device=dev)
    sc = {"mean": np.random.rand(N), "std": np.random.rand(N) + 0.5}
    assert np.allclose(to_counts(f, sc).cpu().numpy(), invert_scaler(f.cpu().numpy(), sc), rtol=1e-4)

    print("ok  IG completes the path and gradient x input does not, invisible channel gets 0 from "
          "IG and occlusion, forward_fixed_deg == forward and holds degree, gate-off is immune to "
          "edge ablation, lag/band/share bookkeeping, the random-weight verdict flips on a failing "
          "draw, a one-hot target completes on that node's own forecast and not on the pooled one, "
          "routing, count inversion")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--all", action="store_true", help="every panel-arm, every seed")
    ap.add_argument("--panels", nargs="*", default=None, choices=PANELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--origins", type=int, default=MAX_ORIGINS)
    ap.add_argument("--steps", type=int, default=IG_STEPS)
    ap.add_argument("--report", action="store_true", help="aggregate the archives, no model")
    ap.add_argument("--no-randctl", action="store_true",
                    help="skip the untrained-encoder lag control inside --report (it needs the bundle)")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); return 0
    if a.report:
        report(randctl=not a.no_randctl); return 0
    panels = PANELS if a.all or not a.panels else a.panels
    seeds = SEEDS if a.all or not a.seeds else a.seeds
    t0 = time.time()
    print(f"device {DEVICE}; panels {list(panels)}; seeds {list(seeds)}; "
          f"<= {a.origins} origins; {a.steps} IG steps")
    for s in seeds:
        for p in panels:
            run_panel(p, s, max_origins=a.origins, steps=a.steps)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> results/explain/  "
          f"(next: python -m explain --report)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

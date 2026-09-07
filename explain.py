"""G5 explainability: what the shared trunk reads, from the checkpoints already on disk.

Three reads, all inference-only against saved checkpoints. No training, no re-scoring, no new
results record is written or touched.

  1. Integrated gradients over the [N, 20, 4] trunk input (4 channels x 20 lags), per panel.
     Global reads are the share of |attribution| per channel and per lag, per horizon.
  2. Occlusion as the faithfulness cross-check: zero one channel or one lag and measure the
     change in the median forecast over the scored cells. IG and occlusion must agree on the
     top channel and the top lag band, or the disagreement is reported as a disagreement.
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
                local = dict(origin=t, district=d_case, name=b.meta["node_ids"][d_case], peak_week=peak,
                             peak_date=str(b.meta["dates"][peak])[:10],
                             ig_map=attr[:, d_case].cpu().numpy(),          # [H, W, 4], signed
                             edge_delta=ed.cpu().numpy(), edges=edges.astype(np.int32),
                             f_count=full[d_case].cpu().numpy())
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


def report(out_txt="explain_report.txt", fig_out="figures/explain"):
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

    P("\nIG vs occlusion agreement on the top channel / top lag band, per horizon "
      "(a disagreement is reported, not resolved)")
    # one bool per printed comparison. argmax gives numpy ints, so `okc == oc` is a numpy bool and
    # `okc + okb` would be logical OR, not addition: that silently tallied 1 per cell instead of 2
    # and reported 12 of 24 on a smoke where every cell agreed. Sum Python ints, and tie the total
    # to the number of cells actually printed.
    marks = []
    for p, a in aggs.items():
        cells = []
        for j, (ic, oc, ib, ob) in enumerate(agreement(a)):
            okc, okb = bool(ic == oc), bool(ib == ob)
            marks += [okc, okb]
            cells.append(f"h{HORIZONS[j]}: {CHANNELS[ic][:4]}{'=' if okc else '!='}{CHANNELS[oc][:4]} "
                         f"{BANDS[ib][0]}-{BANDS[ib][1]}{'=' if okb else '!='}{BANDS[ob][0]}-{BANDS[ob][1]}")
        P(f"  {p:22} " + " | ".join(cells))
    n_ok, n_all = sum(int(m) for m in marks), len(marks)
    assert n_all == 2 * H * len(aggs), f"agreement tally counted {n_all}, expected {2 * H * len(aggs)}"
    P(f"  agreement: {n_ok} of {n_all} (panel x horizon x {{channel, band}})")

    P("\nfalsification tests, stated in G5_Explainability_Scope.md section 6 before the run")
    t1 = {p: CHANNELS[a["ig_chan"][0].mean(0).argmax()] for p, a in aggs.items()}
    P(f"  T1 incidence is the top channel on every panel: "
      f"{'PASS' if all(v == 'incidence' for v in t1.values()) else 'FAIL'}  "
      + ", ".join(f"{p}={v}" for p, v in t1.items()))
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
    fig = plt.figure(figsize=(13.0, 8.6))
    # wspace on the top row has to hold BOTH panel A's colourbar and panel B's row labels, which are
    # the same panel names as A's and hang left off B's axis. 0.14 fits the colourbar alone.
    gs_top = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.40,
                              left=0.115, right=0.945, top=0.94, bottom=0.60)
    gs_bot = fig.add_gridspec(1, 2, width_ratios=[1, 1.9], wspace=0.44,
                              left=0.115, right=0.945, top=0.46, bottom=0.22)
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

    # B: lag shares, panels x lags (lag 1 = origin week, at the left). Its OWN colourbar: a share
    # over 20 lags averages 0.05 where a share over 4 channels averages 0.25, so one bar covering
    # both would either flatten B or mislabel it. They are the same quantity on different supports.
    L = np.array([aggs[p]["ig_lag"][0].mean(0)[::-1] for p in panels])      # index -> lag order
    imB = axB.imshow(L, cmap=seq, vmin=0, vmax=L.max(), aspect="auto")
    axB.set_xticks(range(W)); axB.set_xticklabels([str(l) for l in range(1, W + 1)], fontsize=7)
    axB.set_yticks(range(len(panels))); axB.set_yticklabels(short, fontsize=8)
    axB.set_xlabel("weeks before the forecast origin (1 = the origin week)")
    axB.set_title("B · share of IG attribution by lag, per panel", loc="left", color=INK, pad=8)
    axB.tick_params(length=0)

    # C and D: the Ebola local case, few-shot arm, seed-averaged
    loc = _load("explain__ebola_L12__seed*__local.npz")
    if loc:
        ig = np.mean([l["ig_map"] for l in loc], 0)[0]                     # h3, [W, 4] signed
        name, date = str(loc[0]["name"]), str(loc[0]["peak_date"])
        v = np.abs(ig).max()
        imC = axC.imshow(ig.T[:, ::-1], cmap=div, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
        axC.set_yticks(range(4)); axC.set_yticklabels(CHANNELS, fontsize=8)
        axC.set_xticks(range(0, W, 2)); axC.set_xticklabels([str(l) for l in range(1, W + 1, 2)], fontsize=7)
        axC.set_xlabel("weeks before the origin")
        axC.set_title(f"C · signed IG, {pretty(name)}, h3 target {date}", loc="left", color=INK, pad=8)
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
            # 3 decimals: these run 0.218 down to 0.003, and ".1f" printed three of the four as "0.0"
            axD.text(val, k, f"  {val:.3f}", va="center", fontsize=8, color=INK2)
        axD.set_yticks(range(len(vals))); axD.set_yticklabels(names, fontsize=8)
        axD.set_xlabel("absolute change in the h3 forecast when that one edge is dropped, cases/week")
        axD.set_xlim(0, max(vals) * 1.35 if vals else 1)
        axD.set_title(f"D · neighbour influence on {pretty(name)}, edge ablation", loc="left",
                      color=INK, pad=8)
        axD.xaxis.grid(True); axD.set_axisbelow(True)
        for sp in ("top", "right"):
            axD.spines[sp].set_visible(False)
    else:
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
             f"obs_mask is constant on the three influenza panels and COVID, so nothing there varies it. Its "
             f"share is still large ({om_txt}) because a zero baseline charges the channel for the whole "
             f"0 -> 1 move: an artefact of the baseline, not a use of observation status.\n"
             "C and D are one district at the national peak week. Edge ablation holds every degree fixed, "
             "so only the neighbour message moves.\n"
             "None of D is a source of accuracy: the gate-off ablation shows neighbour information helps "
             "error in 0 of 40 cells. That ablation zeroes the gate but keeps the LTR degree feature, so the "
             "tally bounds the value of neighbour information, not of the graph in total.\n"
             "D shows where the model draws from, and nothing more.",
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
        cax = fig.add_axes([pos.x0, pos.y0 - 0.080, pos.width * 0.55, 0.013])
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
          "edge ablation, lag/band/share bookkeeping, routing, count inversion")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--all", action="store_true", help="every panel-arm, every seed")
    ap.add_argument("--panels", nargs="*", default=None, choices=PANELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--origins", type=int, default=MAX_ORIGINS)
    ap.add_argument("--steps", type=int, default=IG_STEPS)
    ap.add_argument("--report", action="store_true", help="aggregate the archives, no model")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); return 0
    if a.report:
        report(); return 0
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

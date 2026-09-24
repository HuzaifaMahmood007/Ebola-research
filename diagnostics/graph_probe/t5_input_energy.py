"""TEST 5: is the low district-specific energy already in the RAW INPUT, or created by the encoder?

t1b/t4 measured the district-specific share of energy in the encoder output h and got
3.8% (influenza_japan), 8.6% (influenza_us-regions), 12.3% (influenza_us-states),
4.8% (covid_us-states), 22.4% (dengue, t_dengue.py, origins[::4], one seed). This script computes
the SAME statistic, same code, on the input windows the encoder consumes (window_slice ->
[N, 20, 4]), and re-derives the representation numbers as a consistency check. If input share is
high and h share is low, the encoder discards district structure. If input share is already low,
the data never had it.

Statistic (identical to t4.distinct_pct): per test origin, for a node-by-feature matrix X,
distinct% = 100 * ||X - mean||^2 / (N*||mean||^2 + ||X - mean||^2), mean taken across nodes.
Averaged over test origins. Effective dimensionality = participation ratio of the singular values
of the centred matrix (t1b), also per origin, averaged.

Inputs measured:
  input_full      the [N, 20*4] flattened window, all 4 channels, exactly what the TCN eats.
                  Channels 1-2 (sin_doy, cos_doy) are identical across districts by construction,
                  and channel 3 (obs_mask) is constant 1 on the influenza panels, so this number is
                  dragged DOWN by shared-by-design channels. Reported because it is literally the input.
  input_incidence the [N, 20] incidence channel alone, the per-node z-scored case series. The fair
                  measure of district structure the DATA offers.
Not measured: the LTR degree feature (models/encoder.py:60). It is part of h but is a graph
property, not a window; its contribution is inside the h numbers, not the input numbers.

Inference-only. Reads results/single/ checkpoints (seed 42, matching t1/t1b/t4) and the processed
bundles. Writes ONE machine-readable summary: results/misc/t5_input_energy.json. Nothing else.
Dengue uses origins[::4], the same subsample t_dengue.py used for its distinct% pass.

Run:  conda run -n ebola-train python diagnostics/graph_probe/t5_input_energy.py
"""
import os, sys, json, time, datetime
os.chdir(r"F:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import numpy as np, torch
import bundles
from models.encoder import SharedEncoder
from models.spatial import sparse_from_dense_np
from models.windows import window_slice

SEED = 42
# published representation numbers from t4 (4 panels) and t_dengue (dengue), prose in
# Session_Audit_2026-09-10.md / CLAUDE.md section 4. The consistency assert below must hit these.
PUBLISHED_H = {"influenza_japan": 3.8, "influenza_us-regions": 8.6, "influenza_us-states": 12.3,
               "covid_us-states": 4.8, "dengue": 22.4}
PANELS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states",
          "dengue"]
TOL = 0.30   # published values are printed to 1 dp; anything beyond rounding is a real mismatch


def distinct_pct(X):
    """t4.distinct_pct verbatim, numpy [N, D] in, percent out."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0, keepdims=True)
    sh = (np.linalg.norm(X.mean(0)) ** 2) * X.shape[0]
    return 100 * (Xc ** 2).sum() / (sh + (Xc ** 2).sum() + 1e-30)


def effdim(X):
    """t1b participation ratio of singular values, on the centred matrix."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    return float((sv ** 2).sum() ** 2 / ((sv ** 4).sum() + 1e-30))


out = {"script": "diagnostics/graph_probe/t5_input_energy.py",
       "date": datetime.date.today().isoformat(), "seed": SEED,
       "statistic": "per-origin district-specific share of energy (t4.distinct_pct) and "
                    "participation-ratio effective dimensionality (t1b), mean over test origins",
       "panels": {}}

hdr = (f"{'panel':22s} {'N':>5s} {'org':>4s} | {'in_full%':>8s} {'in_inc%':>8s} {'h%':>6s} "
       f"{'pub h%':>6s} | {'ed_full':>7s} {'ed_inc':>6s} {'ed_h':>5s}")
print(hdr); print("-" * len(hdr))

for name in PANELS:
    t0 = time.time()
    b = bundles.load(name)
    Z_all = torch.tensor(b.transfer_view(), dtype=torch.float32)
    assert Z_all.shape[-1] == 4, f"{name}: transfer_view is not 4 channels"
    A = sparse_from_dense_np(b.A_geo)
    N = b.X.shape[0]
    enc = SharedEncoder(gate_mode="learned")
    ck = torch.load(f"results/single/encoder__{name}__seed{SEED}__ckpt.pt",
                    map_location="cpu", weights_only=False)
    enc.load_state_dict(ck["encoder"]); enc.eval()

    origins = list(b.origins(phase="test"))
    sub = 4 if name == "dengue" else 1          # t_dengue.py precedent, 7,165 nodes
    origins = origins[::sub]
    assert len(origins) > 0, f"{name}: no test origins"

    acc = {k: [] for k in ("in_full", "in_inc", "h", "ed_full", "ed_inc", "ed_h")}
    with torch.no_grad():
        for t in origins:
            Zw = window_slice(Z_all, t)                      # [N, 20, 4]
            assert Zw.shape == (N, bundles.W, 4), f"{name}: bad window shape {Zw.shape}"
            full = Zw.reshape(N, -1).numpy()                 # [N, 80]
            inc = Zw[:, :, 0].numpy()                        # [N, 20]
            enc(Zw, A, torch.tensor(b.M[:, t], dtype=torch.float32))
            h = enc.last_h.numpy()                           # [N, 64]
            for k, X in (("in_full", full), ("in_inc", inc), ("h", h)):
                d = distinct_pct(X)
                assert 0.0 <= d <= 100.0 and np.isfinite(d), f"{name}: distinct% out of range ({k}={d})"
                acc[k].append(d)
            acc["ed_full"].append(effdim(full)); acc["ed_inc"].append(effdim(inc))
            acc["ed_h"].append(effdim(h))
    m = {k: float(np.mean(v)) for k, v in acc.items()}

    # consistency check: our re-derivation of the representation number must hit the published one
    gap = abs(m["h"] - PUBLISHED_H[name])
    assert gap <= TOL, (f"{name}: recomputed h distinct% {m['h']:.2f} vs published "
                        f"{PUBLISHED_H[name]} (gap {gap:.2f} > {TOL}); comparison invalid, stop")

    out["panels"][name] = {
        "N": N, "n_origins_used": len(origins), "origin_subsample_stride": sub,
        "dims": {"input_full": int(full.shape[1]), "input_incidence": int(inc.shape[1]),
                 "representation": int(h.shape[1])},
        "distinct_pct": {"input_full": m["in_full"], "input_incidence": m["in_inc"],
                         "representation_h": m["h"], "representation_h_published": PUBLISHED_H[name]},
        "effective_dim": {"input_full": m["ed_full"], "input_incidence": m["ed_inc"],
                          "representation_h": m["ed_h"]},
        "seconds": round(time.time() - t0, 1)}
    print(f"{name:22s} {N:5d} {len(origins):4d} | {m['in_full']:7.1f}% {m['in_inc']:7.1f}% "
          f"{m['h']:5.1f}% {PUBLISHED_H[name]:6.1f} | {m['ed_full']:7.2f} {m['ed_inc']:6.2f} "
          f"{m['ed_h']:5.2f}")

path = "results/misc/t5_input_energy.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {path}")
print("read: in_inc% = district-specific share of energy in the incidence input window alone.")
print("      in_full% = same on all 4 channels; sin/cos/(mask on flu) are shared by construction.")
print("      h% = same statistic on the encoder output, must match the published column.")
print("      ed_* = participation-ratio effective dimensionality of the centred matrix.")

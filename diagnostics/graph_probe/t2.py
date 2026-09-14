"""TEST 2: permute the adjacency's node labels at inference. Same topology, wrong districts.
If test error does not move, the model is not using WHICH districts are neighbours."""
import os, sys, torch, numpy as np
os.chdir(r"f:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import bundles
from models.encoder import SharedEncoder
from models.adapters import Adapter
from models.spatial import sparse_from_dense_np
from models.windows import window_slice
from models.config import MEDIAN_IDX
from bundles import HORIZONS

PANELS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states"]
SEEDS = [42, 52, 62, 72, 82]
NPERM = 20

def run(enc, ad, Z_all, A, b, origins):
    """model-space RMSE over observed test cells, pooled over horizons"""
    se, n = 0.0, 0
    tm = b.masks()["test"]
    with torch.no_grad():
        for t in origins:
            h = enc(window_slice(Z_all, t), A, torch.tensor(b.M[:, t], dtype=torch.float32))
            p = ad(h)[:, :, MEDIAN_IDX].numpy()                    # [N, H] median
            for j, hh in enumerate(HORIZONS):
                m = (b.M[:, t + hh] == 1) & (tm[:, t + hh] == 1)
                if m.any():
                    d = p[m, j] - b.y[m, t + hh]
                    se += float((d ** 2).sum()); n += int(m.sum())
    return (se / max(n, 1)) ** 0.5

print(f"{'panel':22s} {'seed':>5s} {'RMSE real A':>12s} {'RMSE permuted A':>16s} {'change':>9s}")
print("-" * 70)
summary = {}
for name in PANELS:
    b = bundles.load(name)
    Z_all = torch.tensor(b.transfer_view(), dtype=torch.float32)
    A_np = b.A_geo
    A = sparse_from_dense_np(A_np)
    origins = b.origins(phase="test")
    N = A_np.shape[0]
    rows = []
    for s in SEEDS:
        ck = torch.load(f"results/single/encoder__{name}__seed{s}__ckpt.pt", map_location="cpu",
                        weights_only=False)
        enc = SharedEncoder(gate_mode="learned"); enc.load_state_dict(ck["encoder"]); enc.eval()
        ad = Adapter(); ad.load_state_dict(ck["adapter"]); ad.eval()
        base = run(enc, ad, Z_all, A, b, origins)
        perms = []
        rng = np.random.default_rng(1234 + s)
        for k in range(NPERM):
            p = rng.permutation(N)
            Ap = A_np[np.ix_(p, p)]
            perms.append(run(enc, ad, Z_all, sparse_from_dense_np(Ap), b, origins))
        pm = float(np.mean(perms))
        rows.append((base, pm, 100 * (pm - base) / base))
        print(f"{name:22s} {s:5d} {base:12.4f} {pm:16.4f} {100*(pm-base)/base:+8.2f}%")
    ch = [r[2] for r in rows]
    summary[name] = (np.mean(ch), np.std(ch, ddof=1))
    print(f"{'':22s} {'MEAN':>5s} {'':12s} {'':16s} {np.mean(ch):+8.2f}% (sd {np.std(ch, ddof=1):.2f})")
    print("-" * 70)
print("\nEach permutation keeps the graph's exact topology and degree sequence and only")
print(f"changes WHICH district sits at each position. {NPERM} permutations per seed, 5 seeds.")
print("A near-zero change means the model gains nothing from knowing its real neighbours.")

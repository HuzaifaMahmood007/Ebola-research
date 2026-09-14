"""Tests 1b/2/3 on dengue (7,165 nodes). Sparse-index permutation, so no dense [N,N] is ever built."""
import os, sys, torch, numpy as np, time
os.chdir(r"f:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import bundles
from models.encoder import SharedEncoder
from models.adapters import Adapter
from models.spatial import sparse_from_dense_np, mask_aware_adj
from models.windows import window_slice
from models.config import MEDIAN_IDX
from bundles import HORIZONS

NPERM, SEED = 5, 42
b = bundles.load("dengue")
N = b.A_geo.shape[0]
Z_all = torch.tensor(b.transfer_view(), dtype=torch.float32)
A = sparse_from_dense_np(b.A_geo)
idx, val = A.indices(), A.values()
origins = b.origins(phase="test")
print(f"dengue N={N} test origins={len(origins)}")

def perm_A(p):
    inv = torch.tensor(np.argsort(p), dtype=torch.long)
    return torch.sparse_coo_tensor(inv[idx], val, (N, N)).coalesce()

def distinct_pct(X):
    X = X.numpy().astype(np.float64); Xc = X - X.mean(0, keepdims=True)
    sh = (np.linalg.norm(X.mean(0)) ** 2) * X.shape[0]
    return 100 * (Xc ** 2).sum() / (sh + (Xc ** 2).sum() + 1e-30)

ck = torch.load(f"results/single/encoder__dengue__seed{SEED}__ckpt.pt", map_location="cpu",
                weights_only=False)
enc = SharedEncoder(gate_mode="learned"); enc.load_state_dict(ck["encoder"]); enc.eval()
ad = Adapter(); ad.load_state_dict(ck["adapter"]); ad.eval()
tm = b.masks()["test"]

def rmse(Ause):
    se, n = 0.0, 0
    with torch.no_grad():
        for t in origins:
            h = enc(window_slice(Z_all, t), Ause, torch.tensor(b.M[:, t], dtype=torch.float32))
            p = ad(h)[:, :, MEDIAN_IDX].numpy()
            for j, hh in enumerate(HORIZONS):
                m = (b.M[:, t + hh] == 1) & (tm[:, t + hh] == 1)
                if m.any():
                    d = p[m, j] - b.y[m, t + hh]; se += float((d ** 2).sum()); n += int(m.sum())
    return (se / max(n, 1)) ** 0.5

t0 = time.time(); base = rmse(A); print(f"RMSE real A = {base:.4f}   ({time.time()-t0:.0f}s per pass)")
rng = np.random.default_rng(99); ps = []
for k in range(NPERM):
    ps.append(rmse(perm_A(rng.permutation(N))))
    print(f"  perm {k+1}: {ps[-1]:.4f}  ({100*(ps[-1]-base)/base:+.2f}%)")
print(f"MEAN permuted = {np.mean(ps):.4f}   change {100*(np.mean(ps)-base)/base:+.2f}%")

dh, da, rel = [], [], []
with torch.no_grad():
    for t in origins[::4]:
        M_t = torch.tensor(b.M[:, t], dtype=torch.float32)
        enc(window_slice(Z_all, t), A, M_t); h = enc.last_h
        Amix = mask_aware_adj(A, M_t); agg = torch.sparse.mm(Amix, h)
        dh.append(distinct_pct(h)); da.append(distinct_pct(agg))
        Ap = mask_aware_adj(perm_A(rng.permutation(N)), M_t)
        rel.append(float((torch.sparse.mm(Ap, h) - agg).norm() / (agg.norm() + 1e-9)))
print(f"\ndistinct% h = {np.mean(dh):.1f}%   distinct% A@h = {np.mean(da):.1f}%   "
      f"lost {100*(1-np.mean(da)/np.mean(dh)):.0f}%")
print(f"|Ap@h - A@h| / |A@h| = {np.mean(rel):.4f}")

"""TEST 3: the mechanism. How much does the neighbour aggregate A@h depend on WHO the neighbours
are, and how much district-specific content survives the averaging?"""
import os, sys, torch, numpy as np
os.chdir(r"f:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import bundles
from models.encoder import SharedEncoder
from models.spatial import sparse_from_dense_np, mask_aware_adj
from models.windows import window_slice

def distinct_pct(X):
    X = X.numpy().astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    shared = (np.linalg.norm(X.mean(0)) ** 2) * X.shape[0]
    return 100 * (Xc ** 2).sum() / (shared + (Xc ** 2).sum() + 1e-30)

PANELS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states"]
print(f"{'panel':22s} | {'distinct% h':>11s} {'distinct% A@h':>14s} {'lost':>6s} | "
      f"{'|Ap@h - A@h| / |A@h|':>21s}")
print("-" * 82)
for name in PANELS:
    b = bundles.load(name)
    Z_all = torch.tensor(b.transfer_view(), dtype=torch.float32)
    A_np, N = b.A_geo, b.A_geo.shape[0]
    A = sparse_from_dense_np(A_np)
    enc = SharedEncoder(gate_mode="learned")
    enc.load_state_dict(torch.load(f"results/single/encoder__{name}__seed42__ckpt.pt",
                                   map_location="cpu", weights_only=False)["encoder"])
    enc.eval()
    dh, da, rel = [], [], []
    rng = np.random.default_rng(7)
    with torch.no_grad():
        for t in b.origins(phase="test"):
            M_t = torch.tensor(b.M[:, t], dtype=torch.float32)
            enc(window_slice(Z_all, t), A, M_t)
            h = enc.last_h
            Amix = mask_aware_adj(A, M_t)
            agg = torch.sparse.mm(Amix, h)
            dh.append(distinct_pct(h)); da.append(distinct_pct(agg))
            for _ in range(5):
                p = rng.permutation(N)
                Ap = mask_aware_adj(sparse_from_dense_np(A_np[np.ix_(p, p)]), M_t)
                aggp = torch.sparse.mm(Ap, h)
                rel.append(float((aggp - agg).norm() / (agg.norm() + 1e-9)))
    print(f"{name:22s} | {np.mean(dh):10.1f}% {np.mean(da):13.1f}% "
          f"{100*(1-np.mean(da)/np.mean(dh)):5.0f}% | {np.mean(rel):20.4f}")
print("\ndistinct% = share of the vector's energy that differs between districts.")
print("'lost' = how much of that district-specific content the neighbour average destroys.")
print("last column = how far the aggregate moves when the graph is relabelled. 0 = who your")
print("neighbours are makes no difference to what you receive.")

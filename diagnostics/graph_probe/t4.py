"""TEST 4: is MEAN aggregation the problem, or the representation feeding it?
Compare what mean / sum / max / min-max-spread aggregators can even see, on the same trained h.
If every aggregator collapses, the aggregator is not the bottleneck."""
import os, sys, torch, numpy as np
os.chdir(r"F:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import bundles
from models.encoder import SharedEncoder
from models.spatial import sparse_from_dense_np, mask_aware_adj
from models.windows import window_slice

def distinct_pct(X):
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0, keepdims=True)
    sh = (np.linalg.norm(X.mean(0)) ** 2) * X.shape[0]
    return 100 * (Xc ** 2).sum() / (sh + (Xc ** 2).sum() + 1e-30)

PANELS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states"]
print(f"{'panel':22s} | {'h':>6s} {'mean':>6s} {'sum':>6s} {'max':>6s} {'spread':>7s} | {'nbr disagree':>12s}")
print("-" * 86)
for name in PANELS:
    b = bundles.load(name)
    Z_all = torch.tensor(b.transfer_view(), dtype=torch.float32)
    A_np = b.A_geo; N = A_np.shape[0]
    A = sparse_from_dense_np(A_np)
    nbrs = [np.nonzero(A_np[i])[0] for i in range(N)]
    enc = SharedEncoder(gate_mode="learned")
    enc.load_state_dict(torch.load(f"results/single/encoder__{name}__seed42__ckpt.pt",
                                   map_location="cpu", weights_only=False)["encoder"])
    enc.eval()
    acc = {k: [] for k in ("h", "mean", "sum", "max", "spread", "dis")}
    with torch.no_grad():
        for t in b.origins(phase="test"):
            M_t = torch.tensor(b.M[:, t], dtype=torch.float32)
            enc(window_slice(Z_all, t), A, M_t)
            h = enc.last_h.numpy()
            Amix = mask_aware_adj(A, M_t)
            mean_ag = torch.sparse.mm(Amix, enc.last_h).numpy()
            sum_ag  = np.stack([h[nb].sum(0) if len(nb) else h[i] for i, nb in enumerate(nbrs)])
            max_ag  = np.stack([h[nb].max(0) if len(nb) else h[i] for i, nb in enumerate(nbrs)])
            spr_ag  = np.stack([(h[nb].max(0) - h[nb].min(0)) if len(nb) > 1 else np.zeros(h.shape[1])
                                for i, nb in enumerate(nbrs)])
            # how much do a node's neighbours disagree with EACH OTHER, relative to their size?
            d = [float(np.linalg.norm(h[nb] - h[nb].mean(0), axis=1).mean() /
                       (np.linalg.norm(h[nb], axis=1).mean() + 1e-9)) for nb in nbrs if len(nb) > 1]
            acc["h"].append(distinct_pct(h)); acc["mean"].append(distinct_pct(mean_ag))
            acc["sum"].append(distinct_pct(sum_ag)); acc["max"].append(distinct_pct(max_ag))
            acc["spread"].append(distinct_pct(spr_ag)); acc["dis"].append(np.mean(d) if d else np.nan)
    m = {k: np.nanmean(v) for k, v in acc.items()}
    print(f"{name:22s} | {m['h']:5.1f}% {m['mean']:5.1f}% {m['sum']:5.1f}% {m['max']:5.1f}% "
          f"{m['spread']:6.1f}% | {m['dis']:11.3f}")
print("\ndistinct% = share of a vector's energy that differs between districts (higher = more usable).")
print("'nbr disagree' = how far a node's neighbours sit from their own mean, relative to their size.")
print("Small disagreement means every aggregator is summarising near-identical inputs.")

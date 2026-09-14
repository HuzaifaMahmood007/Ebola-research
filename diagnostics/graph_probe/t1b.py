"""TEST 1b: is the high district-to-district similarity a real collapse, or just a shared offset?"""
import os, sys, torch, numpy as np
os.chdir(r"f:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import bundles
from models.encoder import SharedEncoder
from models.spatial import sparse_from_dense_np
from models.windows import window_slice

def stats(H):                       # H [N,64] torch
    X = H.numpy().astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)                 # remove the shared component
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T; iu = np.triu_indices(S.shape[0], k=1)
    cos_centred = np.median(S[iu])
    # share of total energy that is the shared mean vs the district-specific part
    shared = (np.linalg.norm(X.mean(0)) ** 2) * X.shape[0]
    spec = (Xc ** 2).sum()
    # participation ratio of the singular values = effective number of directions used
    sv = np.linalg.svd(Xc, compute_uv=False)
    pr = (sv ** 2).sum() ** 2 / ((sv ** 4).sum() + 1e-30)
    return cos_centred, spec / (shared + spec), pr

PANELS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states"]
print(f"{'panel':22s} {'N':>4s} | {'cosC(h)':>8s} {'cosC(hs)':>9s} {'cosC(hout)':>11s} | "
      f"{'distinct%':>9s} | {'effdim h':>9s} {'effdim hout':>12s}")
print("-" * 100)
for name in PANELS:
    b = bundles.load(name)
    Z_all = torch.tensor(b.transfer_view(), dtype=torch.float32)
    A = sparse_from_dense_np(b.A_geo)
    enc = SharedEncoder(gate_mode="learned")
    enc.load_state_dict(torch.load(f"results/single/encoder__{name}__seed42__ckpt.pt",
                                   map_location="cpu", weights_only=False)["encoder"])
    enc.eval()
    acc = {k: [] for k in ("ch", "chs", "cho", "frac", "prh", "pro")}
    with torch.no_grad():
        for t in b.origins(phase="test"):
            h_out = enc(window_slice(Z_all, t), A, torch.tensor(b.M[:, t], dtype=torch.float32))
            h, h_s = enc.last_h, enc.last_h_s
            c1, f1, p1 = stats(h); c2, _, _ = stats(h_s); c3, _, p3 = stats(h_out)
            acc["ch"].append(c1); acc["chs"].append(c2); acc["cho"].append(c3)
            acc["frac"].append(f1); acc["prh"].append(p1); acc["pro"].append(p3)
    m = {k: np.mean(v) for k, v in acc.items()}
    print(f"{name:22s} {b.X.shape[0]:4d} | {m['ch']:8.3f} {m['chs']:9.3f} {m['cho']:11.3f} | "
          f"{100*m['frac']:8.1f}% | {m['prh']:9.2f} {m['pro']:12.2f}")
print("\ncosC = pairwise cosine AFTER removing the across-district mean. 0 = unrelated, 1 = identical.")
print("distinct% = share of total energy that is district-specific rather than the shared offset.")
print("effdim = participation ratio of singular values: how many of the 64 directions are really used.")

"""TEST 1: does the spatial branch actually move the representation, and are districts distinct?"""
import os, sys, torch, numpy as np
os.chdir(r"f:\Quickgen Projects\Research Paper\Ebola-Research"); sys.path.insert(0, os.getcwd())
import bundles
from models.encoder import SharedEncoder
from models.spatial import sparse_from_dense_np
from models.windows import window_slice

def cos_offdiag(H):
    Hn = H / (H.norm(dim=1, keepdim=True) + 1e-9)
    S = (Hn @ Hn.T).numpy()
    iu = np.triu_indices(S.shape[0], k=1)
    return S[iu]

PANELS = ["influenza_japan", "influenza_us-regions", "influenza_us-states", "covid_us-states"]
print(f"{'panel':22s} {'N':>5s} {'|hs-h|/|h|':>11s} {'|hout-h|/|h|':>13s} {'g med':>7s} "
      f"{'cos(h)':>8s} {'cos(hs)':>8s} {'cos(hout)':>10s}")
print("-" * 92)
for name in PANELS:
    b = bundles.load(name)
    Z_all = torch.tensor(b.transfer_view(), dtype=torch.float32)
    A = sparse_from_dense_np(b.A_geo)
    Mnp = b.M
    enc = SharedEncoder(gate_mode="learned")
    ck = torch.load(f"results/single/encoder__{name}__seed42__ckpt.pt", map_location="cpu", weights_only=False)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    tm = b.masks()["test"]
    origins = [t for t in b.origins(phase="test")]
    rs, ro, gs, ch, chs, cho = [], [], [], [], [], []
    with torch.no_grad():
        for t in origins:
            Z = window_slice(Z_all, t)
            M_t = torch.tensor(Mnp[:, t], dtype=torch.float32)
            h_out = enc(Z, A, M_t)
            h, h_s, g = enc.last_h, enc.last_h_s, enc.last_g
            rs.append(float(((h_s - h).norm(dim=1) / (h.norm(dim=1) + 1e-9)).median()))
            ro.append(float(((h_out - h).norm(dim=1) / (h.norm(dim=1) + 1e-9)).median()))
            gs.append(float(g.median()))
            ch.append(np.median(cos_offdiag(h))); chs.append(np.median(cos_offdiag(h_s)))
            cho.append(np.median(cos_offdiag(h_out)))
    print(f"{name:22s} {b.X.shape[0]:5d} {np.mean(rs):11.4f} {np.mean(ro):13.4f} "
          f"{np.mean(gs):7.3f} {np.mean(ch):8.3f} {np.mean(chs):8.3f} {np.mean(cho):10.3f}")
print("\nread: |hs-h|/|h| = how far neighbour mixing moves the vector, as a fraction of its own size")
print("      |hout-h|/|h| = how far the FINAL vector ends up from the graph-free one")
print("      cos(.) = median pairwise cosine similarity BETWEEN DISTRICTS. 1.0 = all identical.")
print(f"      origins used per panel = test-phase origins")

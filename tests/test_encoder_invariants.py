"""The §8 encoder invariants (encoder_architecture_plan.md), each with a negative control that
plants the defect and requires the gate to fail -- a gate that cannot fail proves nothing (the
Phase-2 discipline). Runnable script, no framework:  python -m tests.test_encoder_invariants

Does NOT enter ebola into any loop that chooses anything (§0.5): ebola appears only in the
structural A_hat-finite and degree-distribution readings, which choose nothing."""
from __future__ import annotations

import inspect

import torch
import torch.nn as nn

import bundles
from models import (Adapter, SharedEncoder, node_indexed_params, normalise_adj, pinball_loss,
                    sparse_from_dense_np)
from models.spatial import LTR
from models.temporal import DilatedTCN


def main():
    torch.manual_seed(0)
    d = 64

    # Gate 1 -- receptive field >= 20 (Correction B); dilations (1,2) [RF=4] must fail.
    assert DilatedTCN().rf == 32
    try:
        DilatedTCN(dilations=(1, 2)); raise SystemExit("RF control did not fire")
    except AssertionError:
        pass
    print("ok  RF gate: RF=32>=20; control dilations(1,2) [RF=4] rejected")

    # Gate 2 -- no shared-trunk parameter has any dimension in the five graph sizes (C2).
    enc = SharedEncoder(d=d, gate_mode="learned")
    assert node_indexed_params(enc) == [], node_indexed_params(enc)
    ctrl = nn.Module(); ctrl.register_parameter("evil", nn.Parameter(torch.zeros(47)))
    assert node_indexed_params(ctrl), "C2 control did not fire"
    print("ok  C2 gate: no trunk param sized by N; control nn.Parameter(47) caught")

    # Gate 3 -- A_hat finite on all five graphs (C3); control: normalise the (small, dense) japan
    # graph WITHOUT +I -> isolated rows (japan_10/19) get deg=0 -> dinv=inf -> inf*0 = NaN.
    deg_dist = {}
    for name in bundles.BUNDLE_NAMES:
        A = sparse_from_dense_np(bundles.load(name).A_geo)
        A_hat, deg = normalise_adj(A)
        assert torch.isfinite(A_hat.values()).all() and (deg > 0).all(), f"{name}: bad A_hat"
        q = torch.quantile(deg, torch.tensor([0.0, 0.5, 1.0]))
        deg_dist[name] = (float(q[0]), float(q[1]), float(q[2]), float(deg.mean()))
    japan_dense = torch.tensor(bundles.load("influenza_japan").A_geo).float()
    dinv0 = japan_dense.sum(1).pow(-0.5)
    assert not torch.isfinite(dinv0[:, None] * japan_dense * dinv0[None, :]).all(), "no-+I control did not NaN"
    print("ok  C3 gate: A_hat finite on all 5 graphs; control (no +I) yields NaN on isolated nodes")

    # Gate 4 -- the trunk rejects covariates: forward is (Z,A,M_t), no C arg; 5 channels fail (C1).
    jb = bundles.load("influenza_japan")
    Z = torch.tensor(jb.transfer_view()[:, 100 - 19:101, :])           # [47,20,4]
    A = sparse_from_dense_np(jb.A_geo)
    try:
        enc(Z, A, C=torch.tensor(jb.C)); raise SystemExit("C-arg control did not fire")
    except TypeError:
        pass
    try:
        enc(torch.cat([Z, Z[..., :1]], dim=-1), A); raise SystemExit("5-channel control did not fire")
    except AssertionError:
        pass
    print("ok  C1 gate: trunk has no C argument (TypeError); 5-channel input rejected")

    # Gate 5 -- gate off / g=0 reproduces the temporal+LTR (graph-free) representation exactly.
    enc.eval()
    with torch.no_grad():
        h_gf = enc.tcn(Z) + enc.ltr(normalise_adj(A)[1])
        enc.gate.mode = "off"
        h_off = enc(Z, A)
    assert torch.allclose(h_off, h_gf, atol=1e-6), "g=0 does not nest the graph-free model"
    enc.gate.mode = "learned"
    print("ok  gate-nesting gate: g=0 reproduces the graph-free model exactly")

    # Gate 6 -- LTR structurally reuses the degree (its forward takes deg, never A).
    sig = inspect.signature(LTR.forward).parameters
    assert "deg" in sig and "A" not in sig
    print("ok  LTR gate: degree passed in from normalise_adj, not recomputed from A")

    # Gate 7 -- overfit a 20-node toy slice (japan) to near-zero pinball loss: the wiring works.
    enc2, ad = SharedEncoder(d=d), Adapter(d=d)
    N0 = 20
    Zt = torch.tensor(jb.transfer_view()[:N0, 100 - 19:101, :])
    At = sparse_from_dense_np(jb.A_geo[:N0, :N0])
    yt = torch.tensor(jb.y[:N0][:, [100 + h for h in bundles.HORIZONS]])       # [20,4], model space
    mt = torch.ones(N0, len(bundles.HORIZONS))
    opt = torch.optim.Adam(list(enc2.parameters()) + list(ad.parameters()), lr=3e-3)
    enc2.train()
    for _ in range(400):
        opt.zero_grad()
        loss = pinball_loss(ad(enc2(Zt, At)), yt, mt)
        loss.backward(); opt.step()
    assert loss.item() < 0.05, f"toy overfit did not converge: loss={loss.item():.4f}"
    print(f"ok  wiring gate: 20-node toy overfits to pinball loss {loss.item():.4f} (<0.05)")

    # Size-agnosticism: the SAME weights run on 47, 49 and 7,165 nodes with no reshape (C2, sparse).
    for name in ["influenza_japan", "influenza_us-states", "dengue"]:
        b = bundles.load(name)
        Zb = torch.tensor(b.transfer_view()[:, 100 - 19:101, :])
        with torch.no_grad():
            hb = enc2(Zb, sparse_from_dense_np(b.A_geo))
        assert hb.shape == (b.X.shape[0], d)
    print("ok  size-agnostic: one weight set ran on N=47, 49, 7165 with no reshape")

    print("\nA_hat self-loop-inclusive degree (min/median/max/mean) per dataset (Task 12.6.2):")
    for name, (lo, med, hi, mean) in deg_dist.items():
        print(f"    {name:22s} {lo:5.1f} / {med:5.1f} / {hi:6.1f} / {mean:5.2f}")
    print("\nall encoder invariants pass; every negative control fired.")


if __name__ == "__main__":
    main()

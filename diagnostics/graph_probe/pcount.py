import sys; sys.path.insert(0, r"f:\Quickgen Projects\Research Paper\Ebola-Research")
from models.encoder import SharedEncoder, node_indexed_params
from models.adapters import Adapter
e = SharedEncoder(); a = Adapter()
tot = lambda m: sum(p.numel() for p in m.parameters())
print("encoder total", tot(e))
for n, mod in [("tcn", e.tcn), ("ltr", e.ltr), ("spatial", e.spatial), ("gate", e.gate)]:
    print("  ", n, tot(mod))
print("adapter total", tot(a))
for n, p in a.named_parameters():
    print("  ", n, tuple(p.shape), p.numel())
print("node-indexed params:", node_indexed_params(e))
print("TCN receptive field:", e.tcn.rf)

import sys, torch; sys.path.insert(0, r"f:\Quickgen Projects\Research Paper\Ebola-Research")
from models.adapters import Adapter
torch.manual_seed(0)
a = Adapter()
with torch.no_grad():
    a.gamma.normal_(1, 0.3); a.beta.normal_(0, 0.3)
h = torch.randn(37, 64)
out = a(h)
W, b = a.head.weight, a.head.bias
Wf = W * a.gamma            # fold gamma into the columns
bf = b + W @ a.beta         # fold beta into the bias
flat = (h @ Wf.T + bf).view(37, a.nH, a.nQ)
print("max abs deviation:", float((out - flat).abs().max()))

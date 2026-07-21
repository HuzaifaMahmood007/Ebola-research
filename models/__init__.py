"""models -- the disease-agnostic shared spatio-temporal encoder (encoder_architecture_plan.md).

Public API so callers do `from models import SharedEncoder, Adapter, pinball_loss, ...` without
knowing the file split. Run scripts as modules from the repo root: `python -m train.loop`,
`python -m tests.test_encoder_invariants`."""
from .config import D_HIDDEN, DILATIONS, KERNEL, MEDIAN_IDX, NODE_COUNTS, QUANTILES, SPATIAL_LAYERS
from .temporal import DilatedTCN
from .spatial import LTR, SpatialMixer, mask_aware_adj, normalise_adj, sparse_from_dense_np
from .adapters import Adapter, adaptation_params, pinball_loss, shared_params
from .encoder import Gate, SharedEncoder, node_indexed_params, spatial_contribution
from .windows import targets_and_mask, window_slice

__all__ = [
    "SharedEncoder", "Adapter", "Gate", "DilatedTCN", "SpatialMixer", "LTR",
    "pinball_loss", "normalise_adj", "mask_aware_adj", "sparse_from_dense_np",
    "spatial_contribution", "node_indexed_params", "shared_params", "adaptation_params",
    "targets_and_mask", "window_slice",
    "QUANTILES", "MEDIAN_IDX", "D_HIDDEN", "DILATIONS", "KERNEL", "SPATIAL_LAYERS", "NODE_COUNTS",
]

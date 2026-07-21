"""Architecture constants -- the single code source of truth (configs/encoder_base.yaml mirrors
these for the human record). Reconciled per encoder_architecture_plan.md §3/§7 + guide Rev 2."""
from __future__ import annotations

D_HIDDEN = 64                                          # trunk width (plan §7)
DILATIONS = (1, 2, 4, 8, 16)                            # kernel-2 TCN -> RF 32 >= w=20 (Correction B)
KERNEL = 2
SPATIAL_LAYERS = 2                                      # plan §3.3 (3+ over-smooths)
DROPOUT = 0.1
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)              # pinball levels (P4)
MEDIAN_IDX = QUANTILES.index(0.5)                       # the point forecast for RMSE/MAE/PCC
NODE_COUNTS = frozenset({10, 47, 49, 61, 7165})        # the five graph sizes; C2 forbids these as param dims

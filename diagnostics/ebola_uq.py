"""The pre-registered Ebola UQ block: WIS, CRPS, PIT, coverage and width. (audit M6 / goal G4)

    conda run -n ebola-train python -m diagnostics.ebola_uq --selfcheck
    conda run -n ebola-train python -m diagnostics.ebola_uq
    conda run -n ebola-train python -m diagnostics.ebola_uq --arms ebola_L12

WHY THIS EXISTS. `Ebola_Prereg.md` E6 requires WIS, CRPS and PIT for both arms from the archived
quantiles, and the client's work order asks for them by name. All five metrics are implemented and
hand-checked in `score.py`, all 20 quantile archives are on disk, and until now the only consumer
for Ebola was the conformal wrapper, which reports coverage and width at the 90% level and nothing
else. G4 is a REQUIRED goal whose metric set was therefore complete on the development folds and
incomplete on the headline case study.

READ-ONLY, AND PROTOCOL-LEGAL. This computes nothing new and re-scores nothing: it reads frozen
`__quantiles.npz` archives and the bundle's own truth. Prereg 5.1 scores each arm exactly once, and
amendment A7 was registered precisely so these quantities stay computable post hoc. No point
forecast, no record and no checkpoint is touched.

WHAT IT REPORTS, AND THE RULES IT INHERITS.
  * Both arms x both regimes. L12 and L20 are different pre-registered support sets and are NEVER
    averaged together.
  * Dispersion on every cell: mean +- sd over the five seeds.
  * WIS and CRPS are printed side by side and are expected to AGREE. On a five-level grid they are
    the same number, and presenting them as two independent columns would be double-counting.
  * PIT is the continuous non-randomised transform. On sparse count data the randomised count PIT
    is the standard and this is not it -- stated here rather than left for a reviewer.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import statistics as st
import sys

import bundles
from diagnostics.ldo3_report import uq_table

ARMS = ("ebola_L12", "ebola_L20")
REGIMES = (("encoder_ebola", "few-shot"), ("encoder_ebola_zeroshot", "zero-shot"))
SEEDS = (42, 52, 62, 72, 82)
# The keys uq_for_run emits, named for the nominal level ("cov0.9", not "coverage_90"). Kept
# explicit AND asserted below, so a rename upstream fails loudly rather than silently dropping a
# pre-registered column -- which is exactly what a first pass of this reader did.
KEYS = ("wis", "crps", "cov0.5", "cov0.9", "width0.5", "width0.9", "pit_saturated", "n_nodes")
REQUIRED = ("wis", "crps", "cov0.5", "cov0.9", "width0.5", "width0.9")


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    return m, (st.stdev(xs) if len(xs) > 1 else 0.0)


def report(arms=ARMS, seeds=SEEDS):
    print(f"\n{'=' * 104}\nEBOLA UQ BLOCK :: prereg E6, read-only over the frozen quantile "
          f"archives\n{'=' * 104}")
    any_row = False
    for arm in arms:
        uq = uq_table([arm], prefixes=[p for p, _ in REGIMES], seeds=seeds,
                      verbose=False, phase="query")
        if not uq:
            print(f"\n  {arm}: no quantile archives -- nothing to report")
            continue
        for prefix, label in REGIMES:
            hs = sorted({h for (p, n, h, k) in uq if p == prefix})
            if not hs:
                print(f"\n  {arm} {label}: no archive")
                continue
            any_row = True
            print(f"\n  {arm}  {label}   ({len(seeds)} seeds, query set)")
            avail = [k for k in KEYS if any((prefix, arm, h, k) in uq for h in hs)]
            missing = [k for k in REQUIRED if k not in avail]
            assert not missing, (f"{arm} {label}: pre-registered UQ columns {missing} are absent. "
                                 f"uq_for_run emitted {sorted({k for (p,n,h,k) in uq if p==prefix})}")
            print("      h | " + " | ".join(f"{k:>16}" for k in avail))
            for h in hs:
                cells = []
                for k in avail:
                    m, sd = mean_sd(list(uq.get((prefix, arm, h, k), {}).values()))
                    cells.append(f"{m:9.3f}+-{sd:<6.3f}")
                print(f"    {h:>3} | " + " | ".join(f"{c:>16}" for c in cells))

            # WIS == CRPS on a 5-level grid. Assert rather than trust: if they ever diverge, one of
            # the two definitions has drifted and both columns are suspect.
            for h in hs:
                w = mean_sd(list(uq.get((prefix, arm, h, "wis"), {}).values()))[0]
                c = mean_sd(list(uq.get((prefix, arm, h, "crps"), {}).values()))[0]
                if w == w and c == c and abs(w - c) > 1e-6 * max(1.0, abs(w)):
                    print(f"      NOTE h{h}: WIS {w:.4f} != CRPS {c:.4f} -- expected identical on "
                          f"a 5-level grid; do not report both until this is explained")
    if not any_row:
        print("\n  nothing to report.")
        return
    print(f"\n  L12 and L20 are separate pre-registered support sets and are never averaged.")
    print(f"  WIS and CRPS coincide on this five-level grid, so they are ONE number reported twice, "
          f"not two independent scores. Coverage is nominal 0.50 / 0.90.")
    print(f"  PIT is the continuous non-randomised transform; the randomised count PIT is the "
          f"standard for sparse counts and this is not it.")
    print(f"  These are the RAW head's intervals. The conformal wrapper's corrected coverage is "
          f"reported separately by conformal.py --apply.\n")


def _selfcheck():
    """The failure that matters is silent: reading the wrong fold. Ebola has no test split, so the
    old hard-coded phase would raise -- and any phase that does NOT raise must be the query set."""
    b = bundles.load("ebola_L12")
    masks = b.masks()
    assert "query" in masks and "support" in masks, f"ebola masks are {sorted(masks)}"
    assert "test" not in masks, \
        "control void: ebola now has a test mask, so phase='test' would silently score the wrong fold"

    from diagnostics.ldo3_report import _eval_mask
    import numpy as np
    origins = b.origins()
    m_q = _eval_mask(b, origins, 3, "query")
    assert m_q.any(), "the query eval mask is empty -- nothing would be scored"
    # and the default must still be the dev-fold behaviour, or every LDO3 number silently moves
    dev = bundles.load("influenza_us-regions")
    assert np.array_equal(_eval_mask(dev, dev.origins(), 3),
                          _eval_mask(dev, dev.origins(), 3, "test")), \
        "the default phase is no longer 'test'; every development-fold UQ number would move"

    print("selfcheck ok: ebola exposes support/query and no test fold, the query eval mask is "
          "non-empty, and the default phase still reproduces the development-fold mask exactly")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()
    report(a.arms, tuple(a.seeds))
    return 0


if __name__ == "__main__":
    sys.exit(main())

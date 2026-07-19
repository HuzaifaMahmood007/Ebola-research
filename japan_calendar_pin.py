"""Pin the Japan influenza calendar to the week, and re-assert it on every build.

The ColaGNN matrices ship undated. The US two were identified to the cell against CDC
ILINet (data_audit 2.3). Japan had no such row-level pin and rested on seasonal phase alone
(+/- ~4 weeks) -- the weakest anchor in the project.

It is now pinned the same way the US H1N1 argmax pinned US-Regions: the matrix's per-season
national maxima are matched against NIID's authoritatively-dated season peaks. Two seasons,
each landing on the exact epidemiological week NIID reports:

  2018/19 season  NIID IASR 40(11): peak week 4 of 2019 (Jan 21-27), 57.09/sentinel, the
                  highest since 1999.  Matrix global max -> 2019-01-26 (epi-week 4 of 2019).
  2017/18 season  NIID IASR 39(11): peak week 5 of 2018, 54.3/sentinel.
                  Matrix season max -> 2018-02-03 (epi-week 5 of 2018).

A one-week error in the anchor would move at least one matrix maximum off its NIID week, so
the two matches together pin 2012-08-04 to the week. The paper's stated span (Aug 2012 ->
Mar 2019, 348 weeks) rules out any whole-year offset, and the matrix fingerprint (max 26635,
mean 655, SD 1711) matches ColaGNN Table 2, confirming this IS that matrix.

Run: python japan_calendar_pin.py   (numpy + pandas only; no geo deps)
Sources: NIID IASR 40(11) 2019 and IASR 39(11) 2018, Influenza season topic-of-the-month.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

JAPAN_MATRIX = "data/Final datasets/influenza/japan.txt"
ANCHOR = "2012-08-04"          # the value under test (INFLUENZA_START["japan"])

# NIID-dated national peaks -> the week-ending-Saturday label they fall on, and the season
# window to search for that season's maximum. (season label, W-SAT peak, window start, end)
NIID_PEAKS = [
    ("2018/19", "2019-01-26", "2018-09-01", "2019-06-30"),   # IASR 40(11): week 4 of 2019
    ("2017/18", "2018-02-03", "2017-09-01", "2018-06-30"),   # IASR 39(11): week 5 of 2018
]


def load_national(matrix_path: str = JAPAN_MATRIX, anchor: str = ANCHOR):
    """Return a weekly national-total ILI series indexed by week-ending Saturday."""
    mat = pd.read_csv(matrix_path, header=None).to_numpy(dtype=np.float64)   # [T, N]
    national = mat.sum(axis=1)                                               # per-week total
    dates = pd.date_range(anchor, periods=len(national), freq="W-SAT")
    return pd.Series(national, index=dates)


def verify(matrix_path: str = JAPAN_MATRIX, anchor: str = ANCHOR) -> None:
    s = load_national(matrix_path, anchor)

    # The anchor must already be a week-ending Saturday, or date_range silently snaps it.
    assert pd.Timestamp(anchor).day_name() == "Saturday", f"{anchor} is not a Saturday"

    # 1. Global maximum must be the 2018/19 peak week NIID dates to week 4 of 2019.
    global_peak = s.idxmax().date().isoformat()
    assert global_peak == "2019-01-26", (
        f"global max at {global_peak}, expected 2019-01-26 (NIID: 2018/19 peak, week 4 2019)")

    # 2. Each NIID-dated season peak must be that season's maximum in the matrix.
    for season, want, lo, hi in NIID_PEAKS:
        window = s[(s.index >= lo) & (s.index <= hi)]
        got = window.idxmax().date().isoformat()
        assert got == want, (
            f"{season}: matrix season max at {got}, NIID dates the peak to {want}")
        print(f"ok  {season} peak -> {got}  (NIID-dated week)  total={int(window.max())}")

    print(f"ok  Japan anchor {anchor} PINNED to the week: 2 NIID season peaks matched, "
          f"a one-week shift would break at least one")


if __name__ == "__main__":
    verify()

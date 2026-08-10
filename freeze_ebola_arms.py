"""Build, freeze and hash the two pre-registered Ebola support-set arms.

    conda run -n ebola python freeze_ebola_arms.py            # build, check, write, hash
    conda run -n ebola python freeze_ebola_arms.py --verify   # rehash what is on disk

Client decision (2026-08-07), closing the open item in `Reports/Ebola_Support_Set_Decision.md`:
12-week support is the PRIMARY arm, 20-week is a pre-registered labelled SECONDARY arm, and
the 19-week option is dropped. Both arms are built and hashed HERE, before either is scored.
The pre-registration that states the expected outcomes is `progress/decisions/Ebola_Prereg.md`.

On the "L" labels. The sweep table in the decision document indexes outbreak weeks from the
raw first week 2014-03-24, whose incidence cell is masked (week 0 of a cumulative series has
no increment). Its L is therefore the 0-based column index of the LAST support week in
`data/processed/ebola.npz`, not a column count. The operative definition of each arm is its
`cutoff_date`; the label is carried only so the arms can be matched to the document the
client read. `support_columns` is the unambiguous count.

Two digests are recorded per arm. `sha256_file` fixes the artifact exactly as written;
`sha256_content` is a canonical digest over the arrays themselves, and is the one that
survives a rebuild (a .npz is a zip, and zip entries carry a wall-clock timestamp).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

import to_schema as ts
from build_datasets import EBOLA_XLSX, GADM, OUT, RAW_SHA256, _sha256, write
from to_schema import load_ebola

MANIFEST = pathlib.Path("configs/ebola_arms.json")

# never re-declare W/HORIZONS here; a drift would be a silent wrong gate. content_sha256 lives in
# bundles so train/ebola.py can verify a frozen arm without importing this build stack.
from bundles import HORIZONS, W, content_sha256

# The frozen arms, with every count the decision document quotes stated up front. These are
# assertions, not printouts: if the loader or the raw file drifts, the arm is not the arm the
# client signed off on and the freeze must fail rather than quietly write a different split.
ARMS = {
    "ebola_L12": dict(
        role="primary",
        label_L=12,
        cutoff_date="2014-06-28",
        support_columns=13,
        expect=dict(support_cells=59, support_districts=18,
                    adapt_pairs={3: 48, 5: 38, 10: 18, 15: 0},
                    adapt_districts={3: 17, 5: 14, 10: 9, 15: 0}),
    ),
    "ebola_L20": dict(
        role="secondary_preregistered",
        label_L=20,
        cutoff_date="2014-08-23",
        support_columns=21,
        expect=dict(support_cells=113, support_districts=36,
                    adapt_pairs={3: 102, 5: 92, 10: 72, 15: 54},
                    adapt_districts={3: 36, 5: 36, 10: 35, 15: 34}),
    ),
}

# Dropped by the same decision. Recorded so the manifest says what was considered and rejected,
# not only what was kept -- a pre-registration that lists the surviving arms alone is worth less.
DROPPED = {"ebola_L19": dict(label_L=19, cutoff_date="2014-08-16",
                             reason="client decision 2026-08-07: 12 and 20 only")}

# Query counts, IDENTICAL under both arms (decision document, corrected claim): a full-window query
# target sits at t>=22 and support reaches at most t<=20, so no option below L=20 costs a forecast.
# Asserted, because the arms are only comparable if it holds.
#
# TWO different counts, and they are not interchangeable:
#
#   REACHABLE -- per-horizon origins. Every t with a full window whose target t+h lands in the
#     panel, so h3 reaches t=48 and h15 only t=36. This is what the audit note and the support-set
#     decision document quote. It counts query cells a horizon COULD reach.
#
#   SCORED -- one COMMON origin set for all horizons, t in [W-1, T-1-max(H)] = [19, 36], which is
#     what bundles.origins() returns and therefore what score_predictions actually scores. Every
#     other dataset in the project was scored this way, and it is the right protocol here: horizons
#     are only comparable to each other if they are read at the same origins.
#
# The two agree only at h15, whose reach is the binding constraint. Recording both, because quoting
# the reachable numbers as the evaluation size overstates h3/h5/h10 by 34-52%.
EXPECT_REACHABLE_PAIRS = {3: 1151, 5: 1075, 10: 866, 15: 642}
EXPECT_REACHABLE_DISTRICTS = {3: 61, 5: 61, 10: 59, 15: 58}
EXPECT_SCORED_PAIRS = {3: 757, 5: 766, 10: 765, 15: 642}
EXPECT_SCORED_DISTRICTS = {3: 57, 5: 57, 10: 59, 15: 58}
MAX_H = max(HORIZONS)


def _counts(M: np.ndarray, support: np.ndarray, query: np.ndarray) -> dict:
    """Every count the manifest quotes, recomputed from the built arrays."""
    T = M.shape[1]
    # An adaptation pair at horizon h needs a SUPPORT target at t+h, so it exists only where a
    # support cell sits at column >= h. Windows are left-padded, so the origin itself is free.
    adapt = {h: int(support[:, h:].sum()) for h in HORIZONS}
    adapt_d = {h: int((support[:, h:].sum(1) > 0).sum()) for h in HORIZONS}
    # Reachable: per-horizon origins, full window, target inside the panel.
    reach = {h: [t + h for t in range(W - 1, T) if t + h <= T - 1] for h in HORIZONS}
    # Scored: ONE common origin set for every horizon, exactly bundles.origins().
    origins = [t for t in range(W - 1, T) if t + MAX_H <= T - 1]
    scored = {h: [t + h for t in origins] for h in HORIZONS}
    return dict(support_cells=int(support.sum()),
                support_districts=int((support.sum(1) > 0).sum()),
                query_cells=int(query.sum()),
                adapt_pairs=adapt, adapt_districts=adapt_d,
                n_scored_origins=len(origins), scored_origins=[origins[0], origins[-1]],
                reachable_pairs={h: int(query[:, reach[h]].sum()) for h in HORIZONS},
                reachable_districts={h: int((query[:, reach[h]].sum(1) > 0).sum()) for h in HORIZONS},
                scored_pairs={h: int(query[:, scored[h]].sum()) for h in HORIZONS},
                scored_districts={h: int((query[:, scored[h]].sum(1) > 0).sum()) for h in HORIZONS})


def _check(name: str, dt, spec: dict) -> dict:
    """Gate the write. Split integrity first, then the pre-registered counts."""
    M = dt.M.astype(bool)
    support = dt.meta["split"]["support_mask"].astype(bool)
    query = dt.meta["split"]["query_mask"].astype(bool)
    dates = np.array([str(d)[:10] for d in dt.meta["dates"]])

    assert not (support & query).any(), f"{name}: support and query overlap"
    assert np.array_equal(support | query, M), f"{name}: masks do not partition the observed cells"
    # calendar-causality: no query cell may be earlier in time than the latest support cell.
    last_sup = np.where(support.any(0))[0].max()
    first_q = np.where(query.any(0))[0].min()
    assert first_q > last_sup, (f"{name}: query cell at t={first_q} precedes the last support "
                                f"cell at t={last_sup} -- the prefix is not calendar-causal")
    assert dates[last_sup] == spec["cutoff_date"], \
        f"{name}: last support week is {dates[last_sup]}, expected {spec['cutoff_date']}"
    assert last_sup + 1 == spec["support_columns"], \
        f"{name}: {last_sup + 1} support columns, expected {spec['support_columns']}"
    # the scaler must have seen support only: pooled log1p+z over the support cells exactly.
    logc = np.log1p(np.clip(dt.raw, 0, None))[support]
    mu, sd = dt.meta["scaler"]["mean"], dt.meta["scaler"]["std"]
    assert np.allclose(mu, logc.mean(), atol=1e-5) and np.allclose(sd, logc.std(), atol=1e-5), \
        f"{name}: scaler was not fit on the support cells alone"

    got = _counts(dt.M, support, query)
    for k, want in spec["expect"].items():
        want = {int(a): b for a, b in want.items()} if isinstance(want, dict) else want
        assert got[k] == want, f"{name}: {k} = {got[k]}, pre-registered {want}"
    for k, want in (("reachable_pairs", EXPECT_REACHABLE_PAIRS),
                    ("reachable_districts", EXPECT_REACHABLE_DISTRICTS),
                    ("scored_pairs", EXPECT_SCORED_PAIRS),
                    ("scored_districts", EXPECT_SCORED_DISTRICTS)):
        assert got[k] == want, f"{name}: {k} = {got[k]}, pre-registered {want}"
    return got


def _git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def build() -> int:
    src = _sha256(EBOLA_XLSX)
    if src != RAW_SHA256[EBOLA_XLSX]:
        sys.exit(f"CHECKSUM DRIFT  {EBOLA_XLSX}\n  want {RAW_SHA256[EBOLA_XLSX]}\n  got  {src}")
    print(f"ok   source {EBOLA_XLSX.name}  {src[:16]}...")

    arms = {}
    for name, spec in ARMS.items():
        dt = load_ebola(str(EBOLA_XLSX), countries=ts.EBOLA_CORE_COUNTRIES,
                        few_shot_support_cutoff=spec["cutoff_date"], gadm_dir=GADM)
        got = _check(name, dt, spec)
        path = write(name, dt)
        arms[name] = dict(
            **{k: v for k, v in spec.items() if k != "expect"},
            npz=str(path).replace("\\", "/"),
            sha256_file=_sha256(path), sha256_content=content_sha256(path),
            scaler=dict(mean=float(dt.meta["scaler"]["mean"][0]),
                        std=float(dt.meta["scaler"]["std"][0]), scope="pooled_log1p_z_on_support"),
            counts=got,
        )
        print(f"ok   {name:10s} {spec['cutoff_date']}  support {got['support_cells']:>3} cells / "
              f"{got['support_districts']:>2} districts   adapt {got['adapt_pairs']}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(
        frozen="2026-08-07",
        decision="client 2026-08-07: 12wk primary, 20wk pre-registered secondary, 19wk dropped",
        prereg="progress/decisions/Ebola_Prereg.md",
        source=dict(file=str(EBOLA_XLSX).replace("\\", "/"), sha256=src),
        lookback_w=W, horizons=list(HORIZONS),
        label_basis="L = 0-based column index of the last support week in data/processed/ebola.npz "
                    "(the decision-document sweep convention); cutoff_date is operative",
        identical_across_arms=dict(
            scored=dict(pairs=EXPECT_SCORED_PAIRS, districts=EXPECT_SCORED_DISTRICTS,
                        note="one common origin set t in [19, 36]; what score_predictions scores"),
            reachable=dict(pairs=EXPECT_REACHABLE_PAIRS, districts=EXPECT_REACHABLE_DISTRICTS,
                           note="per-horizon origins; what the audit note quotes, NOT the eval size")),
        arms=arms, dropped=DROPPED, git_head=_git_head(),
    ), indent=2) + "\n")
    print(f"\nwrote {MANIFEST}")
    return 0


def verify() -> int:
    man = json.loads(MANIFEST.read_text())
    bad = 0
    for name, arm in man["arms"].items():
        p = pathlib.Path(arm["npz"])
        if not p.exists():
            print(f"MISSING  {p}"); bad += 1; continue
        c = content_sha256(p)
        f = _sha256(p)
        ok_c, ok_f = c == arm["sha256_content"], f == arm["sha256_file"]
        print(f"{'ok  ' if ok_c else 'FAIL'} {name:10s} content {c[:16]}...  "
              f"file {'match' if ok_f else 'DIFFERS (rebuild: zip mtimes)'}")
        bad += not ok_c
    src_ok = _sha256(pathlib.Path(man["source"]["file"])) == man["source"]["sha256"]
    print(f"{'ok  ' if src_ok else 'FAIL'} source xlsx")
    bad += not src_ok
    print("\nFROZEN ARMS VERIFIED" if not bad else f"\n{bad} MISMATCH(ES) -- the arms have moved")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="rehash on-disk arms against the manifest")
    a = ap.parse_args()
    sys.exit(verify() if a.verify else build())

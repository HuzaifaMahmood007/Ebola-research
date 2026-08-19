"""Did disabling the trunk early stop change any LDO3 result?  (D17, extended to all 3 folds)

    python -m diagnostics.ldo3full_check            # every full-budget run on disk
    python -m diagnostics.ldo3full_check --selfcheck

Pairs each `encoder_ldo3full*` record with its truncated `encoder_ldo3*` twin at the same seed and
reports whether the scored values moved. Same seed -> same trajectory, so if the extra steps found
nothing better both runs select the same checkpoint and the records are bit-identical. A DIFFERENT
verdict is the interesting one and gets a per-horizon table, not just a flag.

Headline field follows train.lodo.HEADLINE (country_macro for dengue, node_mean elsewhere); the
sha256 is over ALL numeric fields, so it is the stricter test of the two.

ponytail: pairs by filename, not by importing train.lodo -- no torch, no fold table to drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LODO = Path("results/lodo")
FULL = "encoder_ldo3full"
TRUNC = "encoder_ldo3"
HEADLINE = {"dengue": "country_macro"}
NUMERIC = ("country_macro", "node_mean")


def _key(r):
    return (r["dataset"], r["horizon"], r["metric"], r["model"].replace("ldo3full", "ldo3"))


def _digest(recs):
    """sha256 over the sorted (key, every numeric field) — order-independent, value-sensitive."""
    body = sorted((_key(r), tuple(r.get(f) for f in NUMERIC)) for r in recs)
    return hashlib.sha256(repr(body).encode()).hexdigest()


def compare(full_path):
    trunc_path = full_path.with_name(full_path.name.replace(FULL, TRUNC, 1))
    if not trunc_path.exists():
        return full_path.name, None, f"no truncated twin at {trunc_path.name}"
    a, b = json.load(open(full_path)), json.load(open(trunc_path))
    same = _digest(a) == _digest(b)
    moved = []
    if not same:
        tr = {_key(r): r for r in b}
        for r in sorted(a, key=_key):
            k = _key(r)
            f = HEADLINE.get(r["dataset"], "node_mean")
            t = tr.get(k)
            if t is None:
                moved.append((k[1], k[2], r[f], None, None))
                continue
            if r[f] != t[f]:
                d = 100 * (r[f] - t[f]) / abs(t[f]) if t[f] else float("nan")
                moved.append((k[1], k[2], r[f], t[f], d))
    return full_path.name, same, moved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return _selfcheck()

    paths = sorted(p for p in LODO.glob(f"{FULL}*.json") if "smoke" not in p.name)
    if not paths:
        print(f"no {FULL}* records under {LODO}/ — nothing to check")
        return 1
    print(f"{'record':58} | verdict")
    print("-" * 90)
    dirty = []
    for p in paths:
        name, same, detail = compare(p)
        if same is None:
            print(f"{name:58} | SKIP: {detail}")
        elif same:
            print(f"{name:58} | IDENTICAL to truncated run")
        else:
            print(f"{name:58} | ** DIFFERS ** ({len(detail)} cells moved)")
            dirty.append((name, detail))
    for name, detail in dirty:
        print(f"\n{name}\n  {'h':>3} | {'metric':>6} | {'full':>12} | {'truncated':>12} | {'delta%':>8}")
        for h, m, fv, tv, d in detail:
            print(f"  {h:>3} | {m:>6} | {fv:12.4f} | "
                  f"{'--':>12}" if tv is None else
                  f"  {h:>3} | {m:>6} | {fv:12.4f} | {tv:12.4f} | {d:+7.2f}%")
    print(f"\n{len(paths) - len(dirty)}/{len(paths)} records identical."
          f"  Trunk convergence: grep 'best=' results/reports/ldo3full_*.log")
    return 1 if dirty else 0


def _selfcheck():
    """The influenza fold is the control: D17 already settled it as identical.

    Plus a planted mutation the digest is required to catch, so a green run means the comparison
    can fail, not just that it printed."""
    flu = sorted(LODO.glob(f"{FULL}__influenza_*.json"))
    assert flu, "no influenza full-budget records — D17's artifacts are missing"
    for p in flu:
        _, same, _ = compare(p)
        assert same is True, f"{p.name}: D17 says identical, compare() says {same}"

    recs = json.load(open(flu[0]))
    d0 = _digest(recs)
    bumped = [dict(r) for r in recs]
    f = HEADLINE.get(bumped[0]["dataset"], "node_mean")
    bumped[0][f] += 1e-9
    assert _digest(bumped) != d0, "digest ignored a 1e-9 change — it is not value-sensitive"
    assert _digest(list(reversed(recs))) == d0, "digest depends on record order"
    print(f"selfcheck ok: {len(flu)} influenza records identical (D17 control), "
          f"digest catches 1e-9 and ignores ordering")
    return 0


if __name__ == "__main__":
    sys.exit(main())

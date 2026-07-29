"""Layout for results/: ONE prefix->subdir map so every reader and writer agrees where an artifact
lives, and the tree can't drift back into a flat pile. Stdlib only -- analysis.py (numpy-only) imports
it without dragging in the torch/model stack.

  results/
    single/   encoder__<ds>__seed<S>[.json | __pernode.npz | __perorigin.npz | __gate.npz]
    joint/    encoder_joint__<tag>__<ds>__seed<S>...        (tag = sampler-balance, e.g. uniform-uniform)
    lodo/     encoder_lodo[_zeroshot]__<ds>__seed<S>...      (leave-one-disease-out transfer probe)
    naive/    naive__<ds>.json + naive__<ds>__<floor>__{pernode,perorigin}.npz
    reports/  *.txt / *.log  human-readable summaries and run logs
    misc/     anything unrouted, and *smoke* throwaways
  results/day11_diagnostics.json stays at the root -- it's the one git-tracked artifact and docs cite
  it by that exact path, so routing keeps it in place (subdir '.').

Every write goes through rpath(fname, make=True); every read that names a file goes through rpath(fname).
Change a location HERE and both ends follow.
"""
from pathlib import Path

RESULTS = Path("results")

# most-specific prefix first: encoder_lodo_zeroshot before encoder_lodo before encoder_joint before encoder.
_ROUTES = (
    ("encoder_lodo_zeroshot__", "lodo"),
    ("encoder_lodo__", "lodo"),
    ("encoder_joint__", "joint"),
    ("encoder__", "single"),
    ("naive__", "naive"),
)


def subdir_for(fname):
    """The results/ subdir an artifact belongs in, decided from its filename alone."""
    if "smoke" in fname:
        return "misc"
    if fname.startswith("day11_diagnostics"):
        return "."                       # stays at results/ root (tracked + doc-referenced)
    if fname.endswith((".txt", ".log")):
        return "reports"
    for prefix, sub in _ROUTES:
        if fname.startswith(prefix):
            return sub
    return "misc"


def rpath(fname, root=RESULTS, make=False):
    """Full path for an artifact, routed to its subdir. make=True creates the subdir (writers pass it)."""
    d = Path(root) / subdir_for(fname)
    if make:
        d.mkdir(parents=True, exist_ok=True)
    return d / fname


def _demo():
    """One runnable check: routing is stable and puts each artifact family where the docstring says."""
    cases = {
        "encoder__dengue__seed42.json": "single",
        "encoder__dengue__seed42__perorigin.npz": "single",
        "encoder_joint__uniform-uniform__dengue__seed42.json": "joint",
        "encoder_lodo__influenza_japan__seed42.json": "lodo",
        "encoder_lodo_zeroshot__influenza_japan__seed42.json": "lodo",
        "naive__dengue.json": "naive",
        "naive__dengue__persistence__perorigin.npz": "naive",
        "gated+spatial_Contribution.txt": "reports",
        "lodo_run.log": "reports",
        "encoder__influenza_japan__smoke.json": "misc",
        "day11_diagnostics.json": ".",
    }
    for fname, want in cases.items():
        got = subdir_for(fname)
        assert got == want, f"{fname}: routed to {got!r}, expected {want!r}"
    # day11 stays put (subdir '.'), so rpath is a no-op relocation for it
    assert rpath("day11_diagnostics.json") == RESULTS / "day11_diagnostics.json"
    assert rpath("encoder__dengue__seed42.json") == RESULTS / "single" / "encoder__dengue__seed42.json"
    print(f"ok  results_paths routes {len(cases)} artifact families correctly")


if __name__ == "__main__":
    _demo()

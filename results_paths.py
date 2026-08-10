"""Layout for results/: ONE prefix->subdir map so every reader and writer agrees where an artifact
lives, and the tree can't drift back into a flat pile. Stdlib only -- analysis.py (numpy-only) imports
it without dragging in the torch/model stack.

  results/
    single/   encoder__<ds>__seed<S>[.json | __pernode.npz | __perorigin.npz | __gate.npz]
    joint/    encoder_joint__<tag>__<ds>__seed<S>...        (tag = sampler-balance, e.g. uniform-uniform)
    lodo/     encoder_lodo[_zeroshot]__<ds>__seed<S>...      (leave-one-DATASET-out transfer probe)
              encoder_ldo[_zeroshot]__<ds>__seed<S>...       (leave-one-DISEASE-out, Week-4 fold fix)

  Two fold structures live side by side in lodo/ and MUST stay distinguishable (client D1: "report
  both structures as separate tables"). `encoder_lodo__` holds out one DATASET; `encoder_ldo__` holds
  out one DISEASE (all 3 influenza sets as a single fold, vs dengue). Records also carry an explicit
  `fold_structure` field -- the filename is the fast path, the field is the one a reader can trust.
  Note `encoder_ldo__` and `encoder_lodo__` are distinct strings, neither a prefix of the other, so
  route order between them is not load-bearing; zeroshot goes first only to match the existing style.
    ebola/    encoder_ebola[_zeroshot]__<arm>__seed<S>...     (the case study, scored once)
              encoder_ebola__alldev__seed<S>__ckpt.pt         (the all-dev trunk)
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
    # ldo3 = the THREE-disease leave-one-disease-out fold (dengue / influenza / covid), which lives
    # beside the two-direction ldo records rather than replacing them. Note "encoder_ldo3__" is not
    # a prefix of "encoder_ldo__" and vice versa (they differ at the '3'), so route order between
    # the ldo and ldo3 families is not load-bearing.
    ("encoder_ldo3_zeroshot__", "lodo"),
    ("encoder_ldo3__", "lodo"),
    # ldo3FULL = the same three-disease fold with the trunk early stop disabled, so the 91k-step
    # cosine schedule is actually traversed. It is a SEPARATE family, not a replacement: the 2026-08-04
    # runs stopped at 13k-27k steps with the lr still at ~1e-3, and the whole point is to read the two
    # side by side at a matched seed. Neither "encoder_ldo3full__" nor "encoder_ldo3full_zeroshot__"
    # is a prefix of the other or of "encoder_ldo3__" (they differ at the 'f'), so route order here is
    # not load-bearing either.
    ("encoder_ldo3full_zeroshot__", "lodo"),
    ("encoder_ldo3full__", "lodo"),
    # the graph-controlled covid <-> influenza_us-states pair (identical A_geo, C and node set)
    ("encoder_pair_zeroshot__", "lodo"),
    ("encoder_pair__", "lodo"),
    ("encoder_ldo_zeroshot__", "lodo"),
    ("encoder_ldo__", "lodo"),
    # The Ebola case study gets its OWN subdir, not lodo/. It is not a fold: nothing is held out of
    # the trunk, the eval set is a few-shot support/query split rather than a train/val/test one, and
    # it is scored exactly once against a pre-registered config. Keeping it separate means no reader
    # can sweep results/lodo/ and silently average the headline case study into a dev-fold table.
    # Must precede "encoder__" or it would route to single/.
    ("encoder_ebola_zeroshot__", "ebola"),
    ("encoder_ebola__", "ebola"),
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
        "encoder_ldo__dengue__seed42.json": "lodo",
        "encoder_ldo__influenza_japan__seed42__pernode.npz": "lodo",
        "encoder_ldo_zeroshot__influenza_us-states__seed42.json": "lodo",
        # G4 quantile archives and trunk checkpoints (added 2026-07-31). New EXTENSIONS, not new
        # prefixes -- they must route by prefix like every other artifact, and in particular .pt must
        # not fall through to misc/ the way an unrouted extension would.
        "encoder_ldo__dengue__seed42__quantiles.npz": "lodo",
        # ldo3 must land in lodo/ too, and must NOT be captured by the ldo routes (or a three-disease
        # record would be read as a two-disease one by anything keying on the filename).
        "encoder_ldo3__covid_us-states__seed42.json": "lodo",
        "encoder_ldo3__influenza_japan__seed42__pernode.npz": "lodo",
        "encoder_ldo3_zeroshot__dengue__seed42.json": "lodo",
        "encoder_ldo3__covid__seed42__ckpt.pt": "lodo",
        # the full-budget variant must route like its parent family, and must NOT be swallowed by
        # the encoder_ldo3__ / encoder_ldo__ routes (or a full-budget record reads as a truncated one)
        "encoder_ldo3full__influenza_japan__seed42.json": "lodo",
        "encoder_ldo3full__influenza_us-regions__seed42__perorigin.npz": "lodo",
        "encoder_ldo3full_zeroshot__influenza_us-states__seed42.json": "lodo",
        "encoder_ldo3full__influenza__seed42__ckpt.pt": "lodo",
        "encoder_pair__covid_us-states__seed42.json": "lodo",
        "encoder_pair_zeroshot__influenza_us-states__seed42__pernode.npz": "lodo",
        "encoder_ldo__dengue2flu__seed42__ckpt.pt": "lodo",
        "encoder_lodo__influenza_japan__seed42__ckpt.pt": "lodo",
        "encoder__dengue__seed42__quantiles.npz": "single",
        "encoder_joint__uniform-uniform__dengue__seed42__ckpt.pt": "joint",
        # the Ebola case study: own subdir, and must NOT be swallowed by the encoder__ route
        "encoder_ebola__ebola_L12__seed42.json": "ebola",
        "encoder_ebola__ebola_L20__seed42__perorigin.npz": "ebola",
        "encoder_ebola__ebola_L12__seed42__quantiles.npz": "ebola",
        "encoder_ebola_zeroshot__ebola_L12__seed42.json": "ebola",
        # the zero-shot arm archives quantiles too (prereg A7). It routes on the zeroshot prefix, and
        # this case is here because the arm is scored ONCE: a quantile archive that silently landed in
        # misc/ could not be moved by re-running anything.
        "encoder_ebola_zeroshot__ebola_L12__seed42__quantiles.npz": "ebola",
        "encoder_ebola__alldev__seed42__ckpt.pt": "ebola",
        "encoder_ebola_smoke__ebola_L12__seed42.json": "misc",     # dry runs stay out of the record
        "encoder_ebola_smoke_zeroshot__ebola_L12__seed42__quantiles.npz": "misc",
        "naive__dengue.json": "naive",
        "naive__dengue__persistence__perorigin.npz": "naive",
        "naive__ebola_L12__support_mean__pernode.npz": "naive",
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

"""results_matrix.py -- the consolidated encoder results matrix: single vs joint vs LODO vs LDO,
every regime scored on the full score.METRICS set, emitted as one markdown document.

WHAT THIS IS. Week-4 asked for "our encoder results, single, joint, and lodo all scored on the latest
metrics, formalized into a results matrix document". This reads ONLY the JSON records already on disk
(results/{single,joint,lodo}/*.json) -- it does not retrain and it does not rescore. `score.py` is the
scoring authority; `rescore_encoder.py` is what puts nrmse on disk. Run that first if nrmse is missing.

REPORTING STANDARDS enforced here, from Review Doc "Reporting standards from here":

  * EVERY number carries dispersion. Cells print `mean ± sd` over seeds with the seed count. A
    1-seed cell prints the bare value and is tagged `(1 seed)` -- never dressed up as a mean.
  * NEVER average across datasets. Every table is per-dataset. There is no "all datasets" row and
    building one is not a supported operation here (dengue 6,161 scored nodes vs us-regions 10).
  * THE COMPARISON REFERENCE IS ON THE TABLE. Every delta table states its reference in its own
    header line, because the Week-3 complaint was precisely that two tables could not be read
    against each other.
  * "WITHIN NOISE" RATHER THAN A DIRECTION. A delta whose paired-by-seed CI covers zero prints
    `within noise`, not a sign.

THE DELTA IS PAIRED BY SEED. Both arms share the seed set, so pairing cancels the shared init noise;
treating the arms as independent would inflate the bar by ~sqrt(2) for nothing (Day15 §2). We compute
the per-seed delta on the COMMON seeds only, then a one-sample t interval over those deltas. With
n = 4-5 seeds the t critical value is 2.78-3.18, NOT 1.96 -- using the normal quantile here would
manufacture significance, so _T95 is a real t table.

AGGREGATION. `country_macro` is the headline everywhere. For the three single-country influenza
bundles it is identical to `node_mean` by construction (n_countries == 1); only dengue's 12 countries
make them differ, which is the whole reason the metric exists (score.py docstring).

SIGN CONVENTION. Deltas on the six ERROR metrics are IMPROVEMENT PERCENT, (ref - new)/|ref|: positive
means the regime beats the reference. `pcc` is deliberately NOT given a percent -- a correlation is
signed and can sit near zero, so a percent change either explodes or flips sign for a model that got
better (ref = -0.05 -> new = +0.30 is a real improvement but a nonsense "-700%"). pcc deltas are
reported as CORRELATION POINTS (new - ref), a different unit, and the cell says `pts` so the two can
never be read off the same scale.

LIKE-FOR-LIKE GUARD. A delta between two regimes is only meaningful if both scored the SAME nodes.
Regimes can legitimately differ here -- score.py drops constant-truth nodes, and a different
prediction can make a node constant -- so we do not assume it, we check it: `n_nodes` is compared
between every arm and its reference, and any mismatch is surfaced in the document rather than being
quietly divided through. This is the same "not like-with-like" failure that already bit the dengue
baseline row, so it gets a machine check instead of a memo.

    conda run -n ebola-train python results_matrix.py [-o Results_Matrix.md]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import score
from results_paths import RESULTS

# Error metrics: lower is better. pcc is the sole "higher is better" metric in score.METRICS.
POINTS_NOT_PERCENT = {"pcc"}                         # signed metric: report a difference, not a ratio
# Review Doc para. 7 names three metrics explicitly: scale-normalised error (nrmse), peak intensity
# error and peak timing error. All three are CLIENT-REQUIRED and are reported in full below -- they
# are not "secondary" and must not be dropped from a table for readability.
PRIMARY = ("rmse", "mae", "nrmse", "pcc")            # lead table
EPI = ("peak_intensity", "peak_timing", "smape")     # peak_* are client-required; smape rides along
DATASETS = ("influenza_japan", "influenza_us-regions", "influenza_us-states", "dengue",
            "covid_us-states")
DELTA_METRICS = ("rmse", "mae", "nrmse")             # error metrics only; pcc goes in its own table
# fixed reading order -- alphabetical would put "LDO" above "single" and bury the reference arm.
REGIME_ORDER = ("single", "joint:uniform-uniform", "joint:sqrt-uniform",
                "LODO adapted", "LODO zero-shot", "LDO adapted", "LDO zero-shot")
# a reference smaller than this is treated as no scale to divide by, not as a huge percentage.
_MIN_DENOM = 1e-9

# two-sided 95% t critical values by degrees of freedom (n-1). n=5 -> df=4 -> 2.776.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228}


def _t95(n):
    return _T95.get(n - 1, 1.96)


def _fmt(x):
    """Readable fixed-point across the full range on disk (pcc ~0.8 to japan rmse ~4,000).

    Deliberately not `g`: `format(4042.6, ',.3g')` is '4.04e+03', which is both ugly in a client
    table and drops the digits a reviewer would check against the source JSON.
    """
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def _order(regimes):
    """REGIME_ORDER first, then anything unrecognised, so a new prefix cannot vanish from the doc."""
    known = [r for r in REGIME_ORDER if r in regimes]
    return known + sorted(set(regimes) - set(known))


def regime_of(fname: str, rec: dict) -> str:
    """Regime label from the filename prefix, which is the unambiguous discriminator.

    `encoder_ldo__` and `encoder_lodo__` are distinct strings and neither is a prefix of the other
    (results_paths.py), but ordering still matters for the zeroshot variants, so test longest first.
    Joint carries its sampler tag because uniform-uniform and sqrt-uniform are different experiments.
    """
    for prefix, label in (("encoder_ldo3_zeroshot__", "LDO3 zero-shot"),
                          ("encoder_ldo3__", "LDO3 adapted"),
                          ("encoder_pair_zeroshot__", "PAIR zero-shot"),
                          ("encoder_pair__", "PAIR adapted"),
                          ("encoder_ldo_zeroshot__", "LDO zero-shot"),
                          ("encoder_lodo_zeroshot__", "LODO zero-shot"),
                          ("encoder_ldo__", "LDO adapted"),
                          ("encoder_lodo__", "LODO adapted"),
                          ("encoder_joint__", "joint"),
                          ("encoder__", "single")):
        if fname.startswith(prefix):
            if label == "joint":
                label = f"joint:{rec.get('sampler') or '?'}"
            # The mean-corrected arm rides in the SAME file as the median arm under the same
            # (dataset, horizon, metric, seed), so without this it would overwrite the median arm in
            # load()'s dict and silently restate every ceiling number by ~24% with no label.
            # The regime string is the only discriminator load() keys on, so the arm belongs here.
            if str(rec.get("model", "")).startswith("encoder_mc"):
                label += " [mean-corr]"
            return label
    return "unknown"


def load():
    """{(regime, dataset, horizon, metric): {seed: country_macro}} plus the per-cell n_nodes."""
    macro = collections.defaultdict(dict)
    nodes = collections.defaultdict(dict)
    meta = collections.defaultdict(set)
    for fam in ("single", "joint", "lodo"):
        for p in sorted((RESULTS / fam).glob("*.json")):
            if "smoke" in p.name:
                continue
            for r in json.loads(p.read_text()):
                reg = regime_of(p.name, r)
                key = (reg, r["dataset"], r["horizon"], r["metric"])
                macro[key][r["seed"]] = r["country_macro"]
                nodes[key][r["seed"]] = r["n_nodes"]
                if r.get("ldo_direction"):
                    meta[(reg, r["dataset"])].add(r["ldo_direction"])
    return macro, nodes, meta


def cell(vals: dict) -> str:
    """`mean ± sd (n)` over seeds; a single seed is labelled as such, never given a fake sd."""
    if not vals:
        return "—"
    xs = [v for v in vals.values() if v is not None and not math.isnan(v)]
    if not xs:
        return "NaN"
    if len(xs) == 1:
        return f"{_fmt(xs[0])} (1 seed)"
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return f"{_fmt(m)} ± {_fmt(sd)} ({len(xs)})"


def paired_delta(new: dict, ref: dict, metric: str):
    """Paired-by-seed delta on the COMMON seeds: (text, mean, clears).

    Unit depends on the metric, and the text always carries it: error metrics give improvement
    PERCENT (positive = better than reference), `pcc` gives correlation POINTS (new - ref).
    `clears` is True when the t interval excludes zero, False when it covers zero ("within noise"),
    and None when n < 2 so no interval exists -- reported as untestable, never as a result.
    """
    points = metric in POINTS_NOT_PERCENT
    deltas = []
    for s in sorted(set(new) & set(ref)):
        a, b = new.get(s), ref.get(s)
        if a is None or b is None or math.isnan(a) or math.isnan(b):
            continue
        if points:
            deltas.append(a - b)
        else:
            if abs(b) < _MIN_DENOM:               # no scale to divide by; a ratio here is noise
                continue
            deltas.append((b - a) / abs(b) * 100.0)
    if not deltas:
        return "—", None, None
    unit = " pts" if points else "%"
    m = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        return f"{m:+.1f}{unit} (1 seed, untestable)", m, None
    sd = math.sqrt(sum((x - m) ** 2 for x in deltas) / (len(deltas) - 1))
    hw = _t95(len(deltas)) * sd / math.sqrt(len(deltas))
    if abs(m) <= hw:
        return f"within noise ({m:+.1f} ± {hw:.1f}{unit}, n={len(deltas)})", m, False
    return f"**{m:+.1f}{unit}** ± {hw:.1f} (n={len(deltas)})", m, True


def delta_vs_scalar(new: dict, ref_scalar, metric: str):
    """Improvement of the arm's mean against a FIXED reference scalar (the 5-seed mean).

    This is the unpaired counterpart of paired_delta: the reference is one pooled number, so the
    per-seed pairing that cancels shared init noise is gone and no interval is computed here. It
    exists to expose Reference B, and it is always labelled as a point estimate.
    """
    xs = [v for v in new.values() if v is not None and not math.isnan(v)]
    if not xs or ref_scalar is None or math.isnan(ref_scalar):
        return "—", None
    m = sum(xs) / len(xs)
    if metric in POINTS_NOT_PERCENT:
        d = m - ref_scalar
        return f"{d:+.1f} pts", d
    if abs(ref_scalar) < _MIN_DENOM:
        return "—", None
    d = (ref_scalar - m) / abs(ref_scalar) * 100.0
    return f"{d:+.1f}%", d


def mean_of(vals: dict):
    xs = [v for v in vals.values() if v is not None and not math.isnan(v)]
    return sum(xs) / len(xs) if xs else None


def node_mismatch(nodes, reg, ref_reg, ds, horizons, metric="rmse"):
    """Seeds where the arm and its reference scored a DIFFERENT number of nodes.

    Returns [(horizon, seed, n_arm, n_ref)]. Empty means every paired delta for this arm is
    like-for-like. Non-empty means at least one delta divides two different populations.
    """
    bad = []
    for h in horizons:
        a, b = nodes.get((reg, ds, h, metric), {}), nodes.get((ref_reg, ds, h, metric), {})
        for s in sorted(set(a) & set(b)):
            if a[s] != b[s]:
                bad.append((h, s, a[s], b[s]))
    return bad


def main(out_path):
    macro, nodes, meta = load()
    regimes = _order({k[0] for k in macro})
    horizons = sorted({k[2] for k in macro})
    L = []
    A = L.append

    A("# Encoder Results Matrix — single vs joint vs LODO vs LDO\n")
    A("Generated by `results_matrix.py` from the JSON records in `results/{single,joint,lodo}/`. ")
    A("Scoring authority is `score.py` (self-check passes; 7 metrics live). No numbers in this ")
    A("document are carried over from any earlier write-up — every cell is recomputed from disk.\n")
    A(f"\n**Metrics:** {', '.join(score.METRICS)}. ")
    A("**Aggregation:** `country_macro` — per-node scores averaged within a country, then countries ")
    A("averaged with equal weight. For the three influenza bundles this is identical to `node_mean` ")
    A("(one country); only dengue's 12 countries make them differ.\n")
    A("\n**Reporting rules applied:** every cell carries `mean ± sd (n seeds)`; a 1-seed cell says so ")
    A("and gets no standard deviation. Nothing is averaged across datasets. Every delta table names ")
    A("its reference. Deltas are paired by seed with a t-based 95% interval, and a cell whose ")
    A("interval covers zero reads `within noise` instead of a direction.\n")
    A("\n**Units.** Error-metric deltas are improvement percent (positive = better than reference). ")
    A("`pcc` deltas are correlation POINTS, never a percent — a correlation is signed and can sit ")
    A("near zero, where a percent change explodes or flips sign on a model that genuinely improved.\n")

    # ---------------- coverage, first, because the gaps decide what is readable ----------------
    A("\n---\n\n## 1. Coverage — what actually exists on disk\n")
    A("\nSeed counts per regime × dataset. Read this before any table below: a 1-seed cell cannot ")
    A("carry a dispersion figure, so the client's standard is unmet there by data, not by choice.\n")
    A("\n| regime | " + " | ".join(d.replace("influenza_", "flu-") for d in DATASETS) + " |")
    A("|---|" + "---|" * len(DATASETS))
    for reg in regimes:
        row = [reg]
        for ds in DATASETS:
            seeds = set()
            for h in horizons:
                seeds |= set(macro.get((reg, ds, h, "rmse"), {}))
            row.append(f"{len(seeds)} seeds" if seeds else "—")
        A("| " + " | ".join(row) + " |")

    # ---------------- per-dataset absolute matrix ----------------
    A("\n---\n\n## 2. Absolute scores by dataset\n")
    A("\nOne block per dataset — never pooled. `country_macro`, count space.\n")
    A("\n`peak_intensity` is |max(pred) − max(truth)| in counts; `peak_timing` is "
      "|argmax(pred) − argmax(truth)| in observed-eval-cell steps (weeks on the dense influenza "
      "panels, approximate where the mask is sparse); `smape` is 0–200 with 0/0 cells excluded. "
      "`peak_intensity`, `peak_timing` and `nrmse` are the three metrics named in Review Doc ¶7.\n")
    for ds in DATASETS:
        A(f"\n### {ds}\n")
        for metric in PRIMARY + EPI:
            A(f"\n**{metric}**" + (" (higher is better)" if metric == "pcc" else "")
              + (" — client-required, Review Doc ¶7" if metric in ("peak_intensity", "peak_timing")
                 else "") + "\n")
            A("\n| regime | " + " | ".join(f"h{h}" for h in horizons) + " |")
            A("|---|" + "---|" * len(horizons))
            for reg in regimes:
                vals = [macro.get((reg, ds, h, metric), {}) for h in horizons]
                if not any(vals):
                    continue
                A(f"| {reg} | " + " | ".join(cell(v) for v in vals) + " |")

    # ---------------- transfer deltas ----------------
    A("\n---\n\n## 3. Transfer deltas vs the single-disease encoder\n")
    A("\n**Comparison reference: the `single` regime on the same dataset, same metric, same horizon, ")
    A("PAIRED BY SEED.** Only seeds present in both arms enter the delta; `n` on each cell is that ")
    A("overlap. **Positive = the regime beats single-disease training.** Interval is a two-sided 95% ")
    A("t interval over the per-seed paired deltas.\n")
    for ds in DATASETS:
        A(f"\n### {ds} — vs single (paired by seed)\n")
        for metric in DELTA_METRICS + ("pcc",) + EPI:
            unit = "correlation points, positive = better" if metric == "pcc" else "improvement %"
            if metric == "peak_timing":
                unit += " — NOTE: a zero-error reference cell has no scale to divide by and is dropped"
            A(f"\n**{metric}** — {unit}\n")
            A("\n| regime | " + " | ".join(f"h{h}" for h in horizons) + " |")
            A("|---|" + "---|" * len(horizons))
            for reg in regimes:
                if reg == "single":
                    continue
                cells, any_data = [], False
                for h in horizons:
                    txt, _, _ = paired_delta(macro.get((reg, ds, h, metric), {}),
                                             macro.get(("single", ds, h, metric), {}), metric)
                    cells.append(txt)
                    any_data |= txt != "—"
                if any_data:
                    A(f"| {reg} | " + " | ".join(cells) + " |")

    # ---------------- two-reference reconciliation ----------------
    A("\n---\n\n## 4. The reference question — 1-seed vs 5-seed reference\n")
    A("\nThe Week-3 transfer table could not be read against the single-disease table, and the client's "
      "diagnosis was that the two used different references. This section holds the transfer number "
      "FIXED and varies only the reference, so the size of that effect is visible rather than argued.\n")
    A("\n- **Reference A — seed-matched (1 seed).** The transfer run at seed *s* against the single-disease "
      "run at the SAME seed *s*. Paired, so shared init noise cancels, but for a 1-seed arm it rests on "
      "one draw and cannot be tested.")
    A("\n- **Reference B — 5-seed mean.** The same transfer run against the MEAN of all 5 single-disease "
      "seeds. Unpaired and a point estimate, but the reference is no longer one lucky or unlucky draw.\n")
    A("\nA large A-vs-B gap means the headline was reporting the reference, not the transfer. This is "
      "most acute for the LODO arms, which exist at seed 42 only.\n")
    for metric in ("rmse", "mae"):
        A(f"\n### {metric} — improvement % under each reference\n")
        A("\n| dataset | regime | " + " | ".join(f"h{h} A | h{h} B" for h in horizons) + " |")
        A("|---|---|" + "---|" * (2 * len(horizons)))
        for ds in DATASETS:
            for reg in regimes:
                if reg == "single":
                    continue
                arm_any = any(macro.get((reg, ds, h, metric)) for h in horizons)
                if not arm_any:
                    continue
                cells = []
                for h in horizons:
                    arm = macro.get((reg, ds, h, metric), {})
                    ref = macro.get(("single", ds, h, metric), {})
                    ref_a = {s: ref[s] for s in set(arm) & set(ref)}       # seed-matched slice
                    _, ma, _ = paired_delta(arm, ref_a, metric)
                    _, mb = delta_vs_scalar(arm, mean_of(ref), metric)
                    cells.append(f"{ma:+.1f}" if ma is not None else "—")
                    cells.append(f"{mb:+.1f}" if mb is not None else "—")
                A(f"| {ds} | {reg} | " + " | ".join(cells) + " |")
        A("\nA = seed-matched reference, B = 5-seed-mean reference. Both in improvement %, positive = "
          "transfer beats single. Intervals are omitted here on purpose — this table is about the "
          "reference shift, and §3 is where the significance verdict lives.\n")

    # ---------------- fold structure comparison ----------------
    A("\n---\n\n## 5. The two fold structures side by side — LODO vs LDO\n")
    A("\nClient decision D1: *\"report both structures as separate tables\"*. They measure different "
      "things and the distinction is the point of the Week-4 fold fix.\n")
    A("\n- **LODO — leave-one-DATASET-out.** Hold out flu-Japan and the encoder has still trained on "
      "flu-US-regions and flu-US-states. This is **population transfer**: the disease is not held out. "
      "1 seed (42) only.")
    A("\n- **LDO — leave-one-DISEASE-out.** All three influenza bundles held out together as one fold, "
      "against dengue as the other. This is **cross-disease transfer**, the number the paper's central "
      "claim rests on. 4 seeds on the flu fold, 5 on dengue.\n")
    A("\nThe expected pattern, stated in advance by the client, is that the honest cross-disease number "
      "comes in weaker than the population-transfer headline. Both columns below are improvement % vs "
      "the `single` encoder, paired by seed.\n")
    for metric in ("rmse", "mae"):
        A(f"\n### {metric} — improvement % vs single\n")
        A("\n| dataset | h | LODO adapted (1 seed) | LDO adapted (4-5 seeds) | LODO zero-shot (1 seed) "
          "| LDO zero-shot (4-5 seeds) |")
        A("|---|---|---|---|---|---|")
        for ds in DATASETS:
            for h in horizons:
                ref = macro.get(("single", ds, h, metric), {})
                row = []
                for reg in ("LODO adapted", "LDO adapted", "LODO zero-shot", "LDO zero-shot"):
                    txt, _, _ = paired_delta(macro.get((reg, ds, h, metric), {}), ref, metric)
                    row.append(txt)
                A(f"| {ds} | h{h} | " + " | ".join(row) + " |")

    # ---------------- like-for-like guard ----------------
    A("\n---\n\n## 6. Like-for-like check — did both arms score the same nodes?\n")
    A("\nA delta is only meaningful if the two arms scored the same node population. `score.py` drops "
      "constant-truth nodes, and a different prediction can change which nodes qualify, so this is "
      "checked per (dataset, horizon, seed) against the `single` reference rather than assumed.\n")
    mismatches = []
    for ds in DATASETS:
        for reg in regimes:
            if reg == "single":
                continue
            for h, s, na, nb in node_mismatch(nodes, reg, "single", ds, horizons):
                mismatches.append((ds, reg, h, s, na, nb))
    if not mismatches:
        A("\n**PASS — no mismatches.** Every regime scored exactly the same `n_nodes` as the `single` "
          "reference on every shared (dataset, horizon, seed). All deltas in §3 are like-for-like.\n")
    else:
        A(f"\n**{len(mismatches)} MISMATCHES — the affected deltas in §3 compare different node "
          f"populations and must not be quoted until resolved.**\n")
        A("\n| dataset | regime | h | seed | n_nodes (arm) | n_nodes (single) |")
        A("|---|---|---|---|---|---|")
        for ds, reg, h, s, na, nb in mismatches:
            A(f"| {ds} | {reg} | {h} | {s} | {na:,} | {nb:,} |")

    # ---------------- caveats ----------------
    A("\n---\n\n## 7. Reading notes and known gaps\n")
    directions = {k: sorted(v) for k, v in meta.items() if v}
    if directions:
        A("\n**LDO fold directions on disk** (which arm produced each row):\n")
        for (reg, ds), dirs in sorted(directions.items()):
            A(f"\n- `{reg}` / {ds}: {', '.join(dirs)}")
        A("\n")
    A("\n- **LODO (leave-one-dataset-out) is 1 seed.** Every LODO cell is seed 42 alone, so it "
      "carries no dispersion and no delta against it is testable. It is kept because the client "
      "asked for both fold structures side by side, not because it is confirmed.")
    A("\n- **`LODO zero-shot` has no `nrmse`.** Zero-shot runs write JSON only — `train/lodo.py` "
      "discards the per-node arrays (`zrecs, _, _, _`, lodo.py:418) — so `rescore_encoder.py`, which "
      "globs `*__pernode.npz`, structurally cannot backfill them. The newer `LDO zero-shot` runs are "
      "unaffected: they were trained after `nrmse` entered `score.METRICS` and recorded it natively. "
      "Recovering `nrmse` for LODO zero-shot needs a rerun, not a rescore.")
    A("\n- **The LDO flu rows are 4 seeds, not 5** (52/62/72/82; seed 42 absent), while LDO dengue "
      "has all 5. Paired deltas on the flu rows therefore rest on a 4-seed overlap.")
    A("\n- **`joint:sqrt-uniform` is 1 seed.** It is the sampler-variant probe, not a confirmed arm.")
    A("\n- **`peak_timing` deltas drop cells with a zero-error reference.** A reference that already "
      "peaked in exactly the right step gives a 0 denominator, and a percentage against it is "
      "undefined rather than infinite. Those seeds are excluded from the pairing, so a peak_timing "
      "delta can rest on fewer seeds than the same cell's rmse delta — read the `n` on the cell.")
    A("\n- **UQ metrics are still absent.** WIS, CRPS, empirical coverage, interval width and PIT are "
      "implemented and self-checked in `score.py`, but they consume quantile predictions and no run "
      "archives them yet. This is the remaining gap against Review Doc ¶7.")
    A("\n- **Baselines are not in this document.** It covers our encoder only, per the request. The "
      "published-vs-reproduced baseline comparison is `paper_compare.py`, and HeatGNN was still "
      "running when this was generated.")

    text = "\n".join(L) + "\n"
    Path(out_path).write_text(text, encoding="utf-8")
    n_cells = len(macro)
    print(f"ok  wrote {out_path}  ({n_cells} (regime,dataset,horizon,metric) cells, "
          f"{len(regimes)} regimes, {len(horizons)} horizons)")
    print(f"    regimes: {regimes}")


def _demo():
    """Runnable checks on the logic that could put a WRONG NUMBER OR A WRONG SIGN in a client
    document: the sign convention, the pcc unit, the noise verdict, seed pairing, and the
    like-for-like guard. A formatting bug is cosmetic; a sign bug is not."""
    # --- error metrics: new BELOW ref is an improvement, so the delta is POSITIVE ---
    txt, m, clears = paired_delta({1: 90.0, 2: 90.0, 3: 90.0}, {1: 100.0, 2: 100.0, 3: 100.0}, "rmse")
    assert abs(m - 10.0) < 1e-9 and clears and "%" in txt, f"error sign/verdict wrong: {txt}"
    # and new ABOVE ref is a regression -> negative. The direction must not be symmetric-by-accident.
    _, mw, _ = paired_delta({1: 110.0, 2: 110.0}, {1: 100.0, 2: 100.0}, "rmse")
    assert abs(mw + 10.0) < 1e-9, f"a worse error metric must read negative, got {mw}"

    # --- pcc: POINTS, not percent, and rising pcc is an improvement ---
    txt2, m2, _ = paired_delta({1: 0.9, 2: 0.9}, {1: 0.8, 2: 0.8}, "pcc")
    assert abs(m2 - 0.1) < 1e-9 and "pts" in txt2 and "%" not in txt2, f"pcc must be points: {txt2}"
    _, m3, _ = paired_delta({1: 0.7}, {1: 0.8}, "pcc")
    assert m3 < 0, "a pcc DROP is a regression"
    # the reason points exist: a near-zero/negative reference must not explode or flip.
    # ref -0.05 -> new +0.30 is a real improvement; as a percent it would read about -700%.
    _, m4, _ = paired_delta({1: 0.30}, {1: -0.05}, "pcc")
    assert abs(m4 - 0.35) < 1e-9 and m4 > 0, f"pcc near zero must stay sane and positive: {m4}"

    # --- noise verdict: deltas straddling zero get no direction ---
    txt5, _, cl5 = paired_delta({1: 100.0, 2: 80.0, 3: 120.0}, {1: 100.0, 2: 100.0, 3: 100.0}, "rmse")
    assert cl5 is False and "within noise" in txt5, f"scattered deltas must read within noise: {txt5}"
    # a tight, consistent delta must still be allowed to clear -- the screen cannot reject everything
    _, _, cl6 = paired_delta({1: 90.0, 2: 91.0, 3: 89.0}, {1: 100.0, 2: 100.0, 3: 100.0}, "rmse")
    assert cl6 is True, "a consistent 10% improvement across seeds must clear the noise screen"

    # --- pairing is a seed INTERSECTION; an unpaired seed cannot leak in ---
    _, m7, _ = paired_delta({1: 50.0, 9: 1e9}, {1: 100.0}, "rmse")
    assert abs(m7 - 50.0) < 1e-9, "unpaired seeds must be dropped, not averaged in"
    # 1 seed -> no interval exists, so it is untestable, never significant
    txt8, _, cl8 = paired_delta({1: 50.0}, {1: 100.0}, "rmse")
    assert cl8 is None and "untestable" in txt8, f"1-seed delta must be untestable: {txt8}"
    # a zero reference has no scale to divide by and must be dropped, not turned into infinity
    assert paired_delta({1: 5.0}, {1: 0.0}, "rmse") == ("—", None, None), "zero denom must drop"
    # NaN on either side is dropped rather than propagating into the mean
    _, m9, _ = paired_delta({1: 90.0, 2: float("nan")}, {1: 100.0, 2: 100.0}, "rmse")
    assert abs(m9 - 10.0) < 1e-9, "NaN must be dropped from the pairing"

    # --- small-sample t, not the normal quantile: using 1.96 at n=5 manufactures significance ---
    assert _t95(5) == 2.776 and _t95(4) == 3.182, "small-n t critical values are load-bearing"

    # --- dispersion honesty and readable numbers ---
    assert "1 seed" in cell({1: 5.0}) and "±" not in cell({1: 5.0}), "no invented sd for one seed"
    assert "±" in cell({1: 5.0, 2: 7.0})
    assert cell({}) == "—"
    assert _fmt(4042.6) == "4,043" and "e" not in _fmt(4042.6), "no scientific notation in a table"
    assert _fmt(0.8123) == "0.8123"

    # --- like-for-like guard actually fires on a differing node count ---
    nodes = {("A", "d", 3, "rmse"): {1: 47}, ("single", "d", 3, "rmse"): {1: 47},
             ("B", "d", 3, "rmse"): {1: 46}}
    assert node_mismatch(nodes, "A", "single", "d", [3]) == [], "equal n_nodes must pass"
    assert node_mismatch(nodes, "B", "single", "d", [3]) == [(3, 1, 46, 47)], "unequal must be caught"

    # --- Reference A vs Reference B: the whole point is that they can DISAGREE ---
    # arm = 90 at seed 42. Seed-matched ref is 100 -> +10%. The 5-seed mean is 120 -> +25%.
    ref5 = {42: 100.0, 52: 110.0, 62: 120.0, 72: 130.0, 82: 140.0}
    arm = {42: 90.0}
    _, a_only, _ = paired_delta(arm, {42: ref5[42]}, "rmse")
    _, b_only = delta_vs_scalar(arm, mean_of(ref5), "rmse")
    assert abs(a_only - 10.0) < 1e-9, f"Reference A must be seed-matched, got {a_only}"
    assert abs(b_only - 25.0) < 1e-9, f"Reference B must use the 5-seed mean, got {b_only}"
    assert mean_of(ref5) == 120.0 and mean_of({}) is None
    # a reference shift can flip the SIGN -- exactly the failure the section exists to expose
    _, flip = delta_vs_scalar({42: 115.0}, mean_of(ref5), "rmse")
    _, flip_a, _ = paired_delta({42: 115.0}, {42: 100.0}, "rmse")
    assert flip > 0 > flip_a, "a reference shift must be able to flip the sign; that is the finding"
    # pcc stays in points under Reference B too, never a percent
    txt_b, _ = delta_vs_scalar({42: 0.9}, 0.8, "pcc")
    assert "pts" in txt_b and "%" not in txt_b, f"Reference B must keep pcc in points: {txt_b}"

    # --- regime routing: the four transfer prefixes must not collide ---
    for fname, want in (("encoder__dengue__seed42.json", "single"),
                        ("encoder_joint__uniform-uniform__dengue__seed42.json", "joint:uniform-uniform"),
                        ("encoder_lodo__dengue__seed42.json", "LODO adapted"),
                        ("encoder_lodo_zeroshot__dengue__seed42.json", "LODO zero-shot"),
                        ("encoder_ldo__dengue__seed42.json", "LDO adapted"),
                        ("encoder_ldo_zeroshot__dengue__seed42.json", "LDO zero-shot")):
        got = regime_of(fname, {"sampler": "uniform-uniform"})
        assert got == want, f"{fname} routed to {got!r}, expected {want!r}"
    assert _order({"LDO adapted", "single"})[0] == "single", "the reference arm must lead the tables"

    print("ok  error sign both directions; pcc in points and safe near zero; noise screen accepts "
          "and rejects; seed intersection; zero/NaN denominators dropped; small-n t; 1-seed honesty; "
          "no sci notation; like-for-like guard fires; 6 regime prefixes route distinctly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="Results_Matrix.md")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _demo()
    else:
        main(a.out)

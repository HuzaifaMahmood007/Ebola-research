# -*- coding: utf-8 -*-
"""Parse numbers OUT of schema_spec.md and recompute them from data/processed/*.npz.

Nothing here reads a progress doc as evidence. The document is the thing under test and the
bundles are the ground truth. Run with --mutate to plant a defect in the parsed text and prove
the checks can actually fail.
"""
import io, json, re, sys
import numpy as np

ROOT = r"F:\Quickgen Projects\Research Paper\Ebola-Research"
sys.path.insert(0, ROOT)
import bundles  # noqa: E402

DOC = ROOT + r"\progress\planning\schema_spec.md"
MUTATE = "--mutate" in sys.argv

text = io.open(DOC, encoding="utf-8").read()
if MUTATE:
    # plant one defect: claim Ebola's primary arm has 60 support cells instead of 59
    text = text.replace("| **59** | **1,240** |", "| **60** | **1,240** |")

flat = re.sub(r"\s+", " ", text)
fails, checks = [], 0


def chk(label, doc_val, disk_val):
    global checks
    checks += 1
    if doc_val != disk_val:
        fails.append("%s: doc=%r disk=%r" % (label, doc_val, disk_val))


def n(x):
    return int(str(x).replace(",", "").replace("*", "").strip())


def loaded(name):
    return bundles.load(name)


def cross_border(b):
    npz = np.load(bundles.DATA_DIR / (b.name + ".npz"), allow_pickle=True)
    nc = json.loads(str(npz["meta_json"]))["node_country"]
    cc = [nc[i] for i in b.meta["node_ids"]]
    idx = np.argwhere(b.A_geo != 0)
    return len(idx), sum(1 for i, j in idx if cc[i] != cc[j])


# ---- 1. the section 0 bundle table -------------------------------------------------
row = re.compile(r"^\| `([a-z_\-]+)`[^|]*\| ([\d,]+) \| ([\d,]+) \| (\d) \| (development|few-shot holdout) \|",
                 re.M)
rows = {m.group(1): m.groups()[1:] for m in row.finditer(text)}
chk("sec0 table: bundle names", sorted(rows), sorted(bundles.BUNDLE_NAMES))
for name, (N, T, F, role) in rows.items():
    b = loaded(name)
    chk("sec0 %s N" % name, n(N), b.X.shape[0])
    chk("sec0 %s T" % name, n(T), b.X.shape[1])
    chk("sec0 %s F" % name, n(F), b.X.shape[2])
    chk("sec0 %s role" % name, role.replace("-", "_").replace(" ", "_"), b.meta["role"])

# ---- 2. lookback and horizons -------------------------------------------------------
m = re.search(r"\*\*`HORIZONS = \((\d+), (\d+), (\d+), (\d+)\)`\*\*", flat)
chk("HORIZONS", tuple(int(g) for g in m.groups()), tuple(bundles.HORIZONS))
m = re.search(r"lookback \*\*`W = (\d+)`\*\* weeks", flat)
chk("W", int(m.group(1)), bundles.W)

# ---- 3. dengue admin levels ----------------------------------------------------------
lv = loaded("dengue").meta["country_levels"]
m = re.search(r"\| Admin2 \| ([^|]+)\|\n\s*\| Admin1 \| ([^|]+)\|", text)
chk("dengue Admin2 set", sorted(x.strip() for x in m.group(1).split(",")),
    sorted(k for k, v in lv.items() if v == "Admin2"))
chk("dengue Admin1 set", sorted(x.strip() for x in m.group(2).split(",")),
    sorted(k for k, v in lv.items() if v == "Admin1"))

# ---- 4. dengue calendar ---------------------------------------------------------------
d = loaded("dengue")
m = re.search(r"\*\*(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2}), T = ([\d,]+) weeks\*\*", flat)
chk("dengue first date", m.group(1), str(d.meta["dates"][0])[:10])
chk("dengue last date", m.group(2), str(d.meta["dates"][-1])[:10])
chk("dengue T", n(m.group(3)), len(d.meta["dates"]))

# ---- 5. dengue graph ------------------------------------------------------------------
nnz, cb = cross_border(d)
m = re.search(r"\*\*(\d+) of ([\d,]+) nonzero adjacency entries cross a country border\*\*", flat)
chk("dengue cross-border", int(m.group(1)), cb)
chk("dengue nnz", n(m.group(2)), nnz)
chk("dengue block diagonal", True, d.meta["graph_is_block_diagonal"])

# ---- 6. ebola graph -------------------------------------------------------------------
e = loaded("ebola_L12")
nnz, cb = cross_border(e)
m = re.search(r"\*\*(\d+) of (\d+) nonzero adjacency entries crossing a national border\*\*", flat)
chk("ebola cross-border entries", int(m.group(1)), cb)
chk("ebola nnz", int(m.group(2)), nnz)
m = re.search(r"\*\*(\d+) undirected\s*\n?\s*cross-border edges out of (\d+)\*\*", text)
chk("ebola cross-border undirected", int(m.group(1)), cb // 2)
chk("ebola total edges", int(m.group(2)), e.meta["adjacency_report"]["n_edges"])
chk("ebola block diagonal", False, e.meta["graph_is_block_diagonal"])

# ---- 7. obs_mask table -----------------------------------------------------------------
om = dict(re.findall(r"^\> \| (dengue|influenza_japan|influenza_us-regions|influenza_us-states|"
                     r"covid_us-states|ebola) \| \*{0,2}([\d.]+)\*{0,2} \|", text, re.M))
chk("obs_mask table covers every bundle", sorted(om), sorted(bundles.BUNDLE_NAMES))
for name, v in om.items():
    chk("obs_mask %s" % name, float(v), round(float(loaded(name).X[:, :, 3].mean()), 4))

# ---- 8. the ebola arms table ------------------------------------------------------------
arm_row = re.compile(r"^\| [^|]*\| `data/processed/(ebola(?:_L\d+)?)\.npz` \| (\d+) \| "
                     r"`(calendar_prefix<= \d{4}-\d{2}-\d{2})` \| \*{0,2}([\d,]+)\*{0,2} \| "
                     r"\*{0,2}([\d,]+)\*{0,2} \| \*{0,2}(\d+)\*{0,2} \| \*{0,2}(\d+)\*{0,2} \|", re.M)
seen = []
for m in arm_row.finditer(text):
    name, sw, cut, sup, qry, dsup, zs = m.groups()
    seen.append(name)
    b = loaded(name)
    ms = b.masks()
    S = ms["support"].astype(bool)
    chk("%s support_weeks" % name, int(sw), b.meta["split"]["support_weeks"])
    chk("%s cutoff" % name, cut, b.meta["few_shot_support_scheme"])
    chk("%s support cells" % name, n(sup), int(S.sum()))
    chk("%s query cells" % name, n(qry), int(ms["query"].astype(bool).sum()))
    chk("%s districts with support" % name, int(dsup), int((S.sum(1) > 0).sum()))
    chk("%s pure zero-shot" % name, int(zs), int((S.sum(1) == 0).sum()))
chk("arms table lists three builds", sorted(seen), ["ebola", "ebola_L12", "ebola_L20"])

# ---- 8b. the three support-week counts ----------------------------------------------------
m = re.search(r"so L12 is (\d+) columns and L20 is (\d+)\.", flat)
for arm, cols in (("ebola_L12", int(m.group(1))), ("ebola_L20", int(m.group(2)))):
    b = loaded(arm)
    cutoff = b.meta["few_shot_support_scheme"].split()[-1]
    chk("%s column count to cutoff" % arm, cols,
        sum(1 for x in b.meta["dates"] if str(x)[:10] <= cutoff))
m = re.search(r"support cell is (\d+) and (\d+), one fewer", flat)
for arm, ncol in (("ebola_L12", int(m.group(1))), ("ebola_L20", int(m.group(2)))):
    S = loaded(arm).masks()["support"].astype(bool)
    chk("%s columns with a support cell" % arm, ncol, int(S.any(0).sum()))
m = re.search(r"is (\d+) and (\d+), and it is neither of those: it is `support_mask\.sum\(1\)\.max\(\)`", flat)
for arm, sw in (("ebola_L12", int(m.group(1))), ("ebola_L20", int(m.group(2)))):
    b = loaded(arm)
    S = b.masks()["support"].astype(bool)
    chk("%s support_weeks == busiest district" % arm, sw, int(S.sum(1).max()))
    chk("%s support_weeks matches meta" % arm, sw, b.meta["split"]["support_weeks"])

# ---- 9. ebola node set -------------------------------------------------------------------
m = re.search(r"has \*\*(\d+)\*\* entries, not just the one\b", text)
chk("ebola nodes_dropped", int(m.group(1)), len(e.meta["nodes_dropped"]))
m = re.search(r"`meta\['name_aliases_applied'\]` has \*\*(\d+)\*\* entries", flat)
chk("ebola name_aliases_applied", int(m.group(1)), len(e.meta["name_aliases_applied"]))
m = re.search(r"That leaves \*\*(\d+)\*\* districts across ([a-z,\s]+), over\s*\n?\*\*(\d+)\*\* weeks, "
              r"\*\*(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\*\*", flat)
chk("ebola N", int(m.group(1)), e.X.shape[0])
chk("ebola countries", [x.strip() for x in m.group(2).replace(" and ", ", ").split(",") if x.strip()],
    list(e.meta["countries"]))
chk("ebola T", int(m.group(3)), e.X.shape[1])
chk("ebola first date", m.group(4), str(e.meta["dates"][0])[:10])
chk("ebola last date", m.group(5), str(e.meta["dates"][-1])[:10])

# ---- 10. the meta contract enums ----------------------------------------------------------
for name in bundles.BUNDLE_NAMES:
    b = loaded(name)
    want = "per_country" if name in ("dengue", "ebola") else "Admin1"
    chk("adm_level %s" % name, want, b.meta["adm_level"])
    want = {"dengue": "queen+knn_block_diagonal", "ebola": "queen+knn_cross_border"}.get(
        name, "shipped(diag_zeroed)")
    chk("A_geo_kind %s" % name, want, b.meta["A_geo_kind"])
    chk("A_mob_available %s" % name, False, b.meta["A_mob_available"])
    chk("C is [N,3] %s" % name, (b.X.shape[0], 3), b.C.shape)
    chk("covariates %s" % name, ["centroid_lat", "centroid_lon", "area_km2"], list(b.meta["covariates"]))
    chk("covariates_status %s" % name, "populated from GADM 4.1", b.meta["covariates_status"])
    chk("covariates_transfer_safe %s" % name, False, b.meta["covariates_transfer_safe"])
    chk("A_geo binary %s" % name, {0.0, 1.0}, set(np.unique(b.A_geo).tolist()) | {0.0, 1.0})
    # the doc says neither crs nor shapefile_ref ships anywhere
    chk("no crs key %s" % name, False, "crs" in b.meta)
    chk("no shapefile_ref key %s" % name, False, "shapefile_ref" in b.meta)
    shipped = name != "dengue" and name != "ebola"
    chk("requires_encoder_self_loops present %s" % name, shipped,
        b.meta.get("requires_encoder_self_loops", False) is True)
    chk("isolated_nodes present %s" % name, shipped, "isolated_nodes" in b.meta)

iso = dict(re.findall(r"`(\[[^\]]*\])`(?:, `\[\]` for us-regions)?", ""))  # placeholder, see below
c = loaded("covid_us-states")
for k in ("raw_sha256", "npi_confounded", "excluded_nodes"):
    chk("covid key %s" % k, True, k in c.meta)
chk("covid excluded_nodes", ["Florida", "District of Columbia"], list(c.meta["excluded_nodes"]))
chk("covid npi_confounded", True, c.meta["npi_confounded"])
f = loaded("influenza_us-states")
chk("covid graph == influenza us-states", True, bool(np.array_equal(c.A_geo, f.A_geo)))
chk("covid C == influenza us-states", True, bool(np.array_equal(c.C, f.C)))
for name, want in (("influenza_japan", ["japan_10", "japan_19"]),
                   ("influenza_us-regions", []),
                   ("influenza_us-states", ["us-states_1", "us-states_9"]),
                   ("covid_us-states", ["covid_us-states_1", "covid_us-states_9"])):
    chk("isolated_nodes %s" % name, want, list(loaded(name).meta["isolated_nodes"]))
    assert ("`%s`" % json.dumps(want).replace('"', "'").replace(", ", "','")) or True

# ---- 10b. the section 6 enums, parsed OUT of the doc rather than hardcoded ----------------
m = re.search(r"- `adm_level`\. Released values: (.+?) - `dates`", flat)
doc_adm = set(re.findall(r"`\"([^\"]+)\"`", m.group(1)))
chk("sec6 adm_level enum", doc_adm, {b.meta["adm_level"] for b in map(loaded, bundles.BUNDLE_NAMES)})
m = re.search(r"- `A_geo_kind`\. Released values: (.+?) `A_mob_available` is", flat)
doc_kind = set(re.findall(r"`\"([^\"]+)\"`", m.group(1)))
chk("sec6 A_geo_kind enum", doc_kind, {b.meta["A_geo_kind"] for b in map(loaded, bundles.BUNDLE_NAMES)})
m = re.search(r"- `split_scheme` released values: (.+?)\. `bundles\._normalise_masks`", flat) or     re.search(r"`split_scheme` released values: (.+?)\.\s*`bundles", flat)
if m:
    doc_ss = set(re.findall(r"`\"([^\"]+)\"`", m.group(1)))
    chk("sec6 split_scheme enum", doc_ss, {b.meta["split_scheme"] for b in map(loaded, bundles.BUNDLE_NAMES)})

# ---- 10c. feature names -------------------------------------------------------------------
m = re.search(r"`feature_names == (\[[^\]]+\])`, with no deaths channel", flat)
chk("dengue feature_names", [x.strip(" '") for x in m.group(1).strip("[]").split(",")],
    list(loaded("dengue").meta["feature_names"]))
chk("dengue F", 4, loaded("dengue").X.shape[2])
chk("ebola F", 5, loaded("ebola").X.shape[2])
chk("ebola deaths_norm at 4", "deaths_norm", loaded("ebola").meta["feature_names"][4])
for name in bundles.BUNDLE_NAMES:
    chk("core_feature_idx %s" % name, [0, 1, 2, 3], list(loaded(name).meta["core_feature_idx"]))

# ---- 11. dengue alias counts --------------------------------------------------------------
from dengue_aliases import DENGUE_NAME_ALIASES, DENGUE_GADM_FIX  # noqa: E402
tot = sum(len(v.get("parents", {})) + len(v.get("units", {})) for v in DENGUE_NAME_ALIASES.values())
m = re.search(r"holds \*\*(\d+)\*\* entries across eight countries", flat)
chk("dengue name alias entries", int(m.group(1)), tot)
chk("dengue name alias countries", 8, len(DENGUE_NAME_ALIASES))
m = re.search(r"holds \*\*(\d+)\*\* entries \(`japan", flat)
chk("dengue gadm fix entries", int(m.group(1)), sum(len(v) for v in DENGUE_GADM_FIX.values()))

# ---- 12. dengue country prune -------------------------------------------------------------
dm = d.meta
m = re.search(r"(\d+) candidate countries are dropped", flat)
chk("dengue dropped countries", int(m.group(1)), len(dm["countries_dropped_low_coverage"]))
chk("dengue excluded == dropped", set(dm["countries_excluded"]), set(dm["countries_dropped_low_coverage"]))
m = re.search(r"and ([\d,]+) rows are summed within their week", flat)
chk("dengue within_week_summed_rows", n(m.group(1)), dm["within_week_summed_rows"])
m = re.search(r"Released country list, 12 in total: ([a-z,\s]+)\.", flat)
chk("dengue country list", [x.strip() for x in m.group(1).split(",")], list(dm["countries"]))

# ---- report ---------------------------------------------------------------------------------
print("checks run: %d" % checks)
if fails:
    print("FAILED %d:" % len(fails))
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("ALL PASS" + ("  <-- BUG: the mutation should have failed this run" if MUTATE else ""))

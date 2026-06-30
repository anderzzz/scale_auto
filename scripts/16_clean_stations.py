"""Phase 16 — pre-analysis cleaning of the station layer, with PROVENANCE.

Raw stays immutable. cache/stations.gpkg (deduped OSM) is never modified; this phase
reads it and writes a NEW layer cache/stations_clean.gpkg plus a transform manifest
output/transforms.json and a human summary output/cleaning.md. Every record keeps its
original brand (brand_raw) and a per-row `transforms` trail, so any step is auditable
and reversible.

Transform chain (ordered):
  T1  brand_normalise   canonicalise brand spelling AND fold chain names that were
                        mis-filed in operator/name back into brand_norm
                        (brand_source records: original | from_operator | from_name).
  T2  rebrand           map defunct brands to current owner (Statoil→Circle K,
                        Shell→St1) — flagged separately as an ASSUMPTION.
  T3  classify          assign a `tier`: major | discount_chain | haulier_diesel |
                        marine | gas_altfuel | community_mack | independent |
                        untagged_osm (blank in OSM: indie + under-tagged automats).
                        `independent` = has a name OR a non-chain operator.
                        (uses richer OSM fuel:* tags from cache/fuel_alltags.json).
  T4  scope             include_cars = not marine and not gas_altfuel — i.e. drop
                        boat pumps and pure biogas/CNG/LPG that don't serve road cars
                        (kept in the file, just flagged out).

Re-runnable; regenerates fuel_alltags.json from the cached PBF if missing.
"""
import sys
import re
import json
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402

RAW = C.CACHE / "stations.gpkg"               # immutable input
OUT = C.CACHE / "stations_clean.gpkg"
ALLTAGS = C.CACHE / "fuel_alltags.json"
PBF = C.CACHE / "sweden-latest.osm.pbf"

# ---- rule tables (data, so they serialise into the manifest) ----------------
# (canonical_brand, parent_group, regex) — matched case-insensitively
CANON_RULES = [
    ("Circle K", "Circle K", r"circle\s*-?\s*k|circlek"),
    ("OKQ8",     "OKQ8",     r"okq8|ok-?\s*q8|ok q8"),
    ("Preem",    "Preem",    r"\bpreem\b"),
    ("St1",      "St1",      r"\bst1\b|\bst 1\b"),
    ("Ingo",     "St1",      r"\bingo\b"),
    ("Tanka",    "Tanka",    r"\btanka\b"),
    ("Qstar",    "Qstar",    r"q-?\s*star|qstar"),
    ("Bilisten", "Qstar",    r"bilisten"),
    ("Pump",     "Qstar",    r"^pump$"),
    ("din-X",    "din-X",    r"din-?\s*x\b|dinx"),
    ("Gulf",     "Gulf",     r"\bgulf\b"),
    ("Uno-X",    "Uno-X",    r"uno-?\s*x|unox"),
]
REBRAND_RULES = [   # defunct -> current owner (ASSUMPTION, flagged)
    ("Circle K", "Circle K", r"statoil"),
    ("St1",      "St1",      r"\bshell\b"),
]
# combined table; is_rebrand flag lets us tag provenance and apply with the SAME
# precedence as canon (brand_raw first, only then operator/name) — so a stale
# operator field never overrides an already-resolved current brand.
ALL_RULES = [(c, g, rx, False) for c, g, rx in CANON_RULES] + \
            [(c, g, rx, True) for c, g, rx in REBRAND_RULES]
MAJOR = {"Circle K", "OKQ8", "Preem", "St1"}
DISCOUNT = {"Ingo", "Tanka", "Qstar", "Bilisten", "Pump", "din-X", "Gulf", "Uno-X"}
HAULIER_RX = r"\bids\b|international diesel|\bsåifa\b|\bsaifa\b|\btrb\b|q8truck|maserfrakt|åkeri"
MARINE_RX = r"\bsjömack|\bskeppsfourn|gästhamn|båtmack|hamnmack|\bmarina\b|fyrudden|bootsmack"
COMMUNITY_RX = r"byamack|bymack|bygdemack|byns mack|sockenmack|byalag|byga?mack"
GAS_NAME_RX = r"fordonsgas|gasolfyll|svensk biogas|biogas|\be\.?on\b|polargas|clean fuel"

PETROL_KEYS = ("fuel:octane_95", "fuel:octane_98", "fuel:octane_100", "fuel:petrol", "fuel:e85")
DIESEL_KEYS = ("fuel:diesel", "fuel:GTL_diesel", "fuel:biodiesel", "fuel:HVO", "fuel:HGV_diesel")
GAS_KEYS = ("fuel:CNG", "fuel:LPG", "fuel:biogas", "fuel:lng", "fuel:propane")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_alltags():
    if C.exists_nonempty(ALLTAGS):
        return
    C.log("fuel_alltags.json missing — re-scanning PBF (~7 min)")
    import osmium
    rows = []
    for obj in osmium.FileProcessor(str(PBF)).with_locations():
        t = obj.tags
        if t.get("amenity") != "fuel":
            continue
        d = {tag.k: tag.v for tag in t}
        d["_osm_type"] = "node" if obj.is_node() else ("way" if obj.is_way() else "rel")
        d["_osm_id"] = obj.id
        rows.append(d)
    json.dump(rows, open(ALLTAGS, "w"))
    C.update_manifest(C.manifest_entry_for(ALLTAGS, "derived: all tags of OSM amenity=fuel"))


def match_all(s):
    """Return (canon, group, is_rebrand) for the first matching rule, else None."""
    if not s or str(s) == "nan":
        return None
    s = str(s).lower()
    for canon, grp, rx, reb in ALL_RULES:
        if re.search(rx, s):
            return canon, grp, reb
    return None


def resolve_brand(braw, op, nm):
    """brand_raw takes precedence; only consult operator then name if it is empty.
    Returns (brand_norm, brand_group, source)."""
    valid = braw is not None and str(braw).strip() and str(braw) != "nan"
    if valid:
        m = match_all(braw)
        if m:
            return m[0], m[1], ("rebrand" if m[2] else "original")
        return str(braw), str(braw), "original_kept"
    for fld, lab in ((op, "from_operator"), (nm, "from_name")):
        m = match_all(fld)
        if m:
            return m[0], m[1], (lab + "+rebrand" if m[2] else lab)
    return None, None, "unresolved"


def main():
    name = "Phase 16 — clean stations (provenance)"
    C.phase_start(name)
    C.set_seed()
    if not C.exists_nonempty(RAW):
        C.warn("stations.gpkg missing; run Phase 1"); C.phase_end(name); return
    ensure_alltags()
    tags = {(t["_osm_type"], int(t["_osm_id"])): t for t in json.load(open(ALLTAGS))}

    g = gpd.read_file(RAW)
    g["brand_raw"] = g["brand"]
    n = len(g)
    log = {"brand_normalise": {"from_operator": 0, "from_name": 0, "canon_spelling": 0},
           "rebrand": 0, "marine": 0, "gas_altfuel": 0, "haulier": 0,
           "community": 0, "tier_counts": {}}
    examples = {"from_operator": [], "from_name": [], "rebrand": [], "marine": [], "community": []}

    brand_norm, brand_group, brand_source, tier, prov = [], [], [], [], []
    is_marine, is_gas, include = [], [], []

    for r in g.itertuples():
        steps = []
        braw, op, nm = r.brand_raw, getattr(r, "operator", None), getattr(r, "name", None)
        # T1+T2: resolve brand with correct precedence (brand_raw > operator > name)
        bn, grp, src = resolve_brand(braw, op, nm)
        if src == "original" and bn != str(braw):
            log["brand_normalise"]["canon_spelling"] += 1; steps.append("canon_spelling")
        elif src == "from_operator":
            log["brand_normalise"]["from_operator"] += 1; steps.append("brand_from_operator")
            if len(examples["from_operator"]) < 8: examples["from_operator"].append(f"{op!r}→{bn}")
        elif src == "from_name":
            log["brand_normalise"]["from_name"] += 1; steps.append("brand_from_name")
            if len(examples["from_name"]) < 8: examples["from_name"].append(f"{nm!r}→{bn}")
        if "rebrand" in src:
            log["rebrand"] += 1; steps.append("rebrand")
            if len(examples["rebrand"]) < 8: examples["rebrand"].append(f"{braw or op or nm!r}→{bn}")
        # T3: classify tier (richer fuel tags)
        rec = tags.get((r.osm_type, int(r.osm_id)), {})
        has_pet = any(rec.get(k) not in (None, "no") for k in PETROL_KEYS)
        has_die = any(rec.get(k) not in (None, "no") for k in DIESEL_KEYS)
        has_gas = any(rec.get(k) not in (None, "no") for k in GAS_KEYS)
        blob = " ".join(str(x) for x in (nm, op, braw) if x)
        marine = bool(re.search(MARINE_RX, blob, re.I))
        # gas-only only for UNBRANDED stations (a branded liquid-fuel site that merely
        # also lists biogas shouldn't be scoped out on incomplete OSM fuel tags)
        gas_only = bn is None and (
            (has_gas and not has_pet and not has_die)
            or (bool(re.search(GAS_NAME_RX, blob, re.I)) and not has_pet and not has_die))
        haulier = bool(re.search(HAULIER_RX, blob, re.I))
        community = bool(re.search(COMMUNITY_RX, blob, re.I))
        if marine:
            tr = "marine"; log["marine"] += 1
            if len(examples["marine"]) < 8: examples["marine"].append(str(nm))
        elif gas_only:
            tr = "gas_altfuel"; log["gas_altfuel"] += 1
        elif haulier:
            tr = "haulier_diesel"; log["haulier"] += 1
        elif community:
            tr = "community_mack"; log["community"] += 1
            if len(examples["community"]) < 8: examples["community"].append(str(nm))
        elif bn in MAJOR:
            tr = "major"
        elif bn in DISCOUNT:
            tr = "discount_chain"
        elif (nm is not None and str(nm).strip() and str(nm) != "nan") or \
             (op is not None and str(op).strip() and str(op) != "nan"):
            # a name OR a (non-chain) operator => a real local independent
            tr = "independent"
        else:
            # genuinely blank in OSM: no brand, operator or name. NOT a market
            # segment — an OSM-completeness bucket (small independents + under-tagged
            # chain automats like the INGO/PUMP pumps OSM never tagged).
            tr = "untagged_osm"
        steps.append(f"tier={tr}")
        inc = tr not in ("marine", "gas_altfuel")

        brand_norm.append(bn); brand_group.append(grp); brand_source.append(src)
        tier.append(tr); is_marine.append(marine); is_gas.append(gas_only)
        include.append(inc); prov.append(";".join(steps))

    g["brand_norm"] = brand_norm
    g["brand_group"] = brand_group
    g["brand_source"] = brand_source
    g["tier"] = tier
    g["is_marine"] = is_marine
    g["is_gas_altfuel"] = is_gas
    g["include_cars"] = include
    g["transforms"] = prov
    from collections import Counter
    log["tier_counts"] = dict(Counter(tier))

    g.to_file(OUT, driver="GPKG")
    C.update_manifest(C.manifest_entry_for(OUT, "derived: cleaned+normalised stations (Phase 16)"))

    # brand histogram before/after (top 15)
    before = g["brand_raw"].fillna("(unbranded)").value_counts().head(15).to_dict()
    after = g["brand_norm"].fillna("(unresolved)").value_counts().head(15).to_dict()
    manifest = {
        "input": {"file": "cache/stations.gpkg", "sha256": sha256(RAW), "n_records": n},
        "output": {"file": "cache/stations_clean.gpkg", "n_records": len(g),
                   "n_include_cars": int(g["include_cars"].sum()),
                   "n_excluded": int((~g["include_cars"]).sum())},
        "transforms": [
            {"id": "T1", "name": "brand_normalise",
             "desc": "canonicalise brand spelling; fold chain names mis-filed in operator/name into brand_norm",
             "rules": [[c, grp, rx] for c, grp, rx in CANON_RULES],
             "counts": log["brand_normalise"], "examples": {k: examples[k] for k in ("from_operator", "from_name")}},
            {"id": "T2", "name": "rebrand", "desc": "ASSUMPTION: map defunct brands to current owner",
             "rules": [[c, grp, rx] for c, grp, rx in REBRAND_RULES],
             "count": log["rebrand"], "examples": examples["rebrand"]},
            {"id": "T3", "name": "classify_tier",
             "desc": "assign market tier using name/operator + richer OSM fuel:* tags",
             "marine_regex": MARINE_RX, "community_regex": COMMUNITY_RX, "haulier_regex": HAULIER_RX,
             "tier_counts": log["tier_counts"],
             "marine_examples": examples["marine"], "community_examples": examples["community"]},
            {"id": "T4", "name": "scope_include_cars",
             "desc": "include_cars = not marine and not gas_altfuel (boat pumps & pure biogas/CNG/LPG dropped from car analysis, kept in file)",
             "n_marine": log["marine"], "n_gas_altfuel": log["gas_altfuel"]},
        ],
        "brand_top15_before": before, "brand_top15_after": after,
    }
    C.write_json(C.OUTPUT / "transforms.json", manifest)

    # human summary
    tc = log["tier_counts"]
    L = ["# Station cleaning — transform provenance", "",
         f"Input `cache/stations.gpkg` (sha256 `{manifest['input']['sha256'][:12]}…`, n={n}) → "
         f"`cache/stations_clean.gpkg` (n={len(g)}). Raw is untouched; each row keeps `brand_raw` "
         "and a `transforms` trail.", "",
         "## What changed", "",
         f"- **T1 brand fold-in:** {log['brand_normalise']['from_operator']} brands recovered from "
         f"`operator`, {log['brand_normalise']['from_name']} from `name`, "
         f"{log['brand_normalise']['canon_spelling']} spelling-canonicalised "
         "(e.g. INGO→Ingo, Q-star→Qstar).",
         f"- **T2 rebrand (assumption):** {log['rebrand']} mapped (Statoil→Circle K, Shell→St1).",
         f"- **T4 scoped out of car analysis:** {log['marine']} marine/harbour + {log['gas_altfuel']} "
         f"pure-gas/biogas = {log['marine']+log['gas_altfuel']} excluded (kept in file, `include_cars=False`).",
         "", "## Tier counts", ""]
    for k, v in sorted(tc.items(), key=lambda kv: -kv[1]):
        L.append(f"- {k}: {v}")
    L += ["", "## 'Unbranded' before → after",
          f"- brand_raw unbranded: {int(g['brand_raw'].isna().sum())}",
          f"- still unresolved after fold-in: {int(g['brand_norm'].isna().sum())}",
          "", "Full rule tables and examples in `output/transforms.json`."]
    (C.OUTPUT / "cleaning.md").write_text("\n".join(L))
    C.log(f"[OK] stations_clean.gpkg n={len(g)} include_cars={int(g['include_cars'].sum())}")
    C.log(f"folded from_operator={log['brand_normalise']['from_operator']} "
          f"from_name={log['brand_normalise']['from_name']} rebrand={log['rebrand']} | "
          f"marine={log['marine']} gas={log['gas_altfuel']}")
    C.log(f"unbranded {int(g['brand_raw'].isna().sum())} → unresolved {int(g['brand_norm'].isna().sum())}")
    C.log(f"tiers: {log['tier_counts']}")
    C.phase_end(name)


if __name__ == "__main__":
    main()

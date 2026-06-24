"""Phase 9 — tourism "effective customer base" proxy, per kommun, from OSM.

Hypothesis: petrol demand in a kommun is driven not only by locally registered cars
but by a transient visitor base — overnight tourists and day-trippers who drive and
refuel locally. SCB guest-night statistics are county-level only (buries Åre inside
Jämtland), so we derive a kommun-resolution proxy from the same Geofabrik PBF that
Phase 1 uses for fuel: counts of accommodation and ski infrastructure per kommun.

One PBF pass (cached to cache/tourism_raw.gpkg) collects, for nodes and ways:
  accommodation : tourism in {hotel,hostel,guest_house,motel,apartment,chalet,
                  alpine_hut,wilderness_hut,camp_site,caravan_site}
  ski_lift      : aerialway in {chair_lift,gondola,drag_lift,t-bar,platter,...}
  resort        : leisure=resort | tourism in {theme_park,attraction}
Capacity tags (beds, capacity, rooms) are kept where present (sparse in OSM).

Then a spatial join to cache/units_kommun.gpkg gives per-kommun category counts,
written to cache/tourism_per_kommun.json (keyed by unit_id) for Phase 8 to consume
as a covariate. Also writes output/tourism_summary.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402
from shapely.geometry import Point, Polygon  # noqa: E402

PBF = C.CACHE / "sweden-latest.osm.pbf"
RAW = C.CACHE / "tourism_raw.gpkg"
KOM = C.CACHE / "units_kommun.gpkg"
OUT_JSON = C.CACHE / "tourism_per_kommun.json"
CRS = C.CONFIG["national_crs"]

ACCOMMODATION = {"hotel", "hostel", "guest_house", "motel", "apartment", "chalet",
                 "alpine_hut", "wilderness_hut", "camp_site", "caravan_site"}
SKI_LIFT = {"chair_lift", "gondola", "drag_lift", "t-bar", "j-bar", "platter",
            "rope_tow", "magic_carpet", "mixed_lift", "cable_car", "zip_line"}
RESORT_TOURISM = {"theme_park", "attraction"}
CAP_TAGS = ["beds", "capacity", "rooms", "stars"]


def categorize(tags):
    t = tags.get("tourism")
    if t in ACCOMMODATION:
        return "accommodation", t
    if tags.get("aerialway") in SKI_LIFT:
        return "ski_lift", tags.get("aerialway")
    if tags.get("leisure") == "resort" or t in RESORT_TOURISM:
        return "resort", (tags.get("leisure") or t)
    return None, None


def _num(v):
    try:
        return float(str(v).split(";")[0].strip())
    except (ValueError, TypeError, AttributeError):
        return None


def extract_raw():
    import osmium
    rows = []
    fp = osmium.FileProcessor(str(PBF)).with_locations()
    n = 0
    for obj in fp:
        n += 1
        if n % 10_000_000 == 0:
            C.log(f"  ...scanned {n/1e6:.0f}M objects, tourism collected {len(rows)}")
        cat, sub = categorize(obj.tags)
        if cat is None:
            continue
        rec = {"cat": cat, "sub": sub, "name": obj.tags.get("name")}
        for k in CAP_TAGS:
            rec[k] = _num(obj.tags.get(k))
        if obj.is_node():
            loc = obj.location
            if not loc.valid():
                continue
            rec.update(osm_type="node", geometry=Point(loc.lon, loc.lat))
        elif obj.is_way():
            coords = [(nd.lon, nd.lat) for nd in obj.nodes if nd.location.valid()]
            if len(coords) >= 3:
                g = Polygon(coords).centroid
            elif coords:
                g = Point(sum(x for x, _ in coords) / len(coords),
                          sum(y for _, y in coords) / len(coords))
            else:
                continue
            rec.update(osm_type="way", geometry=g)
        else:
            continue
        rows.append(rec)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    C.log(f"[OK] tourism raw: {len(gdf)} features "
          f"({gdf['cat'].value_counts().to_dict()})")
    return gdf


def aggregate(raw):
    kom = gpd.read_file(KOM)[["unit_id", "name", "geometry"]]
    pts = raw.to_crs(CRS)
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty].copy()
    j = gpd.sjoin(pts, kom[["unit_id", "geometry"]], how="inner", predicate="within")
    counts = (j.groupby(["unit_id", "cat"]).size().unstack(fill_value=0)
              .reindex(columns=["accommodation", "ski_lift", "resort"], fill_value=0))
    counts.columns = [f"tour_{c}" for c in counts.columns]
    # camp sites specifically (mountain/lake tourism flag)
    camp = (j[j["sub"].isin(["camp_site", "caravan_site", "chalet", "alpine_hut"])]
            .groupby("unit_id").size().rename("tour_camp_chalet"))
    beds = j.groupby("unit_id")["beds"].sum(min_count=1).rename("tour_beds")
    out = kom.merge(counts, on="unit_id", how="left").merge(
        camp, on="unit_id", how="left").merge(beds, on="unit_id", how="left")
    fillc = [c for c in out.columns if c.startswith("tour_")]
    out[fillc] = out[fillc].fillna(0)
    return out, j


def main():
    name = "Phase 9 — tourism proxy"
    C.phase_start(name)
    C.set_seed()
    if not PBF.exists():
        C.warn("PBF missing; run Phase 1 first."); C.phase_end(name); return

    if C.exists_nonempty(RAW):
        C.log(f"[SKIP] reusing cached {RAW.name}")
        raw = gpd.read_file(RAW)
    else:
        raw = extract_raw()
        raw.to_file(RAW, driver="GPKG")
        C.log(f"[OK] wrote {RAW.name}")

    agg, j = aggregate(raw)
    # JSON keyed by unit_id for Phase 8
    cols = [c for c in agg.columns if c.startswith("tour_")]
    rec = {r["unit_id"]: {c: int(r[c]) if c != "tour_beds" else float(r[c])
                          for c in cols} for _, r in agg.iterrows()}
    C.write_json(OUT_JSON, {"columns": cols, "by_unit": rec})
    C.update_manifest(C.manifest_entry_for(OUT_JSON, "derived: OSM tourism POIs per kommun"))
    agg.to_file(C.CACHE / "tourism_per_kommun.gpkg", driver="GPKG")

    summary = {
        "n_features": int(len(raw)),
        "by_category": {k: int(v) for k, v in raw["cat"].value_counts().items()},
        "kommun_with_any_accommodation": int((agg["tour_accommodation"] > 0).sum()),
        "kommun_with_ski_lift": int((agg["tour_ski_lift"] > 0).sum()),
        "top10_accommodation": agg.nlargest(10, "tour_accommodation")[
            ["name", "tour_accommodation", "tour_ski_lift", "tour_camp_chalet"]].to_dict("records"),
        "top10_ski_lift": agg.nlargest(10, "tour_ski_lift")[
            ["name", "tour_ski_lift", "tour_accommodation"]].to_dict("records"),
    }
    C.write_json(C.OUTPUT / "tourism_summary.json", summary)
    C.log(f"tourism per kommun: accommodation>0 in {summary['kommun_with_any_accommodation']}, "
          f"ski lift in {summary['kommun_with_ski_lift']} kommuner")
    C.phase_end(name)


if __name__ == "__main__":
    main()

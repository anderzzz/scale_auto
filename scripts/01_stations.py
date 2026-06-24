"""Phase 1 — Petrol stations P.

Stream amenity=fuel from the Geofabrik Sweden PBF (nodes + ways, way centroids),
reproject to national CRS, dedupe within dedup_radius_m, sanity-gate the count,
and save cache/stations.gpkg.

Heavy raw extraction (~7 min full-country scan) is cached to cache/stations_raw.gpkg
so reruns are fast.
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
RAW = C.CACHE / "stations_raw.gpkg"
OUT = C.CACHE / "stations.gpkg"
KEEP_TAGS = ["amenity", "name", "brand", "operator", "access", "disused", "fuel"]


def extract_raw():
    """Stream fuel nodes+ways from PBF -> GeoDataFrame in EPSG:4326."""
    import osmium

    rows = []
    fp = osmium.FileProcessor(str(PBF)).with_locations()
    n = 0
    for obj in fp:
        n += 1
        if n % 10_000_000 == 0:
            C.log(f"  ...scanned {n/1e6:.0f}M objects, fuel collected {len(rows)}")
        tags = obj.tags
        if tags.get("amenity") != "fuel":
            continue
        rec = {k: tags.get(k) for k in KEEP_TAGS}
        if obj.is_node():
            loc = obj.location
            if not loc.valid():
                continue
            rec.update(osm_type="node", osm_id=obj.id, geometry=Point(loc.lon, loc.lat))
        elif obj.is_way():
            coords = [(nd.lon, nd.lat) for nd in obj.nodes if nd.location.valid()]
            if len(coords) >= 3:
                c = Polygon(coords).centroid
            elif coords:
                c = Point(
                    sum(x for x, _ in coords) / len(coords),
                    sum(y for _, y in coords) / len(coords),
                )
            else:
                continue
            rec.update(osm_type="way", osm_id=obj.id, geometry=c)
        else:
            continue
        rows.append(rec)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    C.log(f"[OK] raw extraction: {len(gdf)} fuel features "
          f"(nodes={int((gdf.osm_type=='node').sum())}, ways={int((gdf.osm_type=='way').sum())})")
    return gdf


def dedupe(gdf, radius_m):
    """Cluster points within radius_m (connected components) and keep one per cluster."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    coords = np.column_stack([xs, ys])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius_m, output_type="ndarray")
    nn = len(gdf)
    if len(pairs):
        data = np.ones(len(pairs))
        m = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(nn, nn))
        n_comp, labels = connected_components(m + m.T, directed=False)
    else:
        labels = np.arange(nn)
    gdf = gdf.copy()
    gdf["cluster"] = labels
    gdf["_x"] = xs
    gdf["_y"] = ys
    gdf["_score"] = (
        gdf["brand"].notna().astype(int) * 2
        + gdf["name"].notna().astype(int)
        + (gdf["osm_type"] == "node").astype(int) * 0.5
    )

    # cluster centroid (averaged coords) and size
    cent = gdf.groupby("cluster")[["_x", "_y"]].mean()
    size = gdf.groupby("cluster").size().rename("cluster_size")

    # representative row per cluster: highest score, stable tie-break by index
    order = gdf.sort_values(["cluster", "_score"], ascending=[True, False])
    reps = order.drop_duplicates("cluster", keep="first").set_index("cluster")
    reps = reps.join(size)
    reps["_cx"] = cent["_x"]
    reps["_cy"] = cent["_y"]
    reps["geometry"] = gpd.points_from_xy(reps["_cx"], reps["_cy"])

    out = gpd.GeoDataFrame(reps, geometry="geometry", crs=gdf.crs).reset_index(drop=True)
    out = out.drop(columns=[c for c in ["_x", "_y", "_cx", "_cy", "_score"] if c in out.columns])
    return out


def main():
    name = "Phase 1 — petrol stations P"
    C.phase_start(name)
    if C.exists_nonempty(OUT):
        C.log(f"[SKIP] {OUT.name} already exists")
        g = gpd.read_file(OUT)
        C.phase_end(name, f"(cached, n={len(g)})")
        return

    if not PBF.exists():
        C.download(C.CONFIG["geofabrik_pbf"], PBF, min_bytes=100_000_000,
                   what="Geofabrik Sweden PBF")
    C.update_manifest(C.manifest_entry_for(PBF, C.CONFIG["geofabrik_pbf"]))

    # raw extraction (cached)
    if C.exists_nonempty(RAW):
        C.log(f"[SKIP] reusing cached raw extraction {RAW.name}")
        raw = gpd.read_file(RAW)
    else:
        raw = extract_raw()
        raw.to_file(RAW, driver="GPKG")
        C.log(f"[OK] wrote {RAW.name}")

    raw_count = len(raw)

    # optionally drop access=private
    priv_mask = (raw["access"].astype("string").str.lower() == "private").fillna(False)
    n_private = int(priv_mask.sum())
    raw = raw[~priv_mask].copy()
    C.log(f"dropped {n_private} access=private features")

    # reproject to national CRS, then dedupe
    proj = raw.to_crs(C.CONFIG["national_crs"])
    deduped = dedupe(proj, C.CONFIG["dedup_radius_m"])
    dedup_count = len(deduped)

    # brand histogram
    brands = (deduped["brand"].fillna(deduped["operator"]).fillna("(unbranded/unknown)")
              .astype("string").str.strip())
    brand_hist = brands.value_counts().head(25).to_dict()

    # sanity gate
    lo, hi = C.CONFIG["station_total_tolerance"]
    anchor = C.CONFIG["expected_station_total"]
    in_band = lo <= dedup_count <= hi
    completeness = {
        "raw_features": raw_count,
        "dropped_private": n_private,
        "deduped_count": dedup_count,
        "drivkraft_anchor": anchor,
        "tolerance_band": [lo, hi],
        "within_band": bool(in_band),
        "ratio_to_anchor": round(dedup_count / anchor, 3),
    }
    if not in_band:
        C.warn(f"COMPLETENESS: deduped station count {dedup_count} OUTSIDE band [{lo},{hi}] "
               f"(anchor {anchor}). Carrying caveat into report.")
    else:
        C.log(f"completeness OK: {dedup_count} stations within band [{lo},{hi}] "
              f"(anchor {anchor}, ratio {completeness['ratio_to_anchor']})")

    deduped.to_file(OUT, driver="GPKG")
    C.update_manifest(C.manifest_entry_for(OUT, "derived: dedup of OSM amenity=fuel"))

    C.write_json(C.OUTPUT / "stations_summary.json", {
        "completeness": completeness,
        "brand_histogram_top25": brand_hist,
    })
    C.log(f"raw={raw_count} private_dropped={n_private} deduped={dedup_count}")
    C.log(f"top brands: {dict(list(brand_hist.items())[:8])}")
    C.phase_end(name, f"(stations={dedup_count})")


if __name__ == "__main__":
    C.set_seed()
    main()

"""Phase 2 — spatial units + population, with station counts P.

Builds, for each unit definition, a table (unit_id, name, S, P, geometry, is_border):
  (a) tätort                      cache/units_tatort.gpkg        [PRIMARY]
  (b) tätort + småort             cache/units_tatort_smaort.gpkg
  (c) kommun (LAU)                cache/units_kommun.gpkg        [ADMINISTRATIVE]
  (d) FUA (best-effort)           cache/units_fua.gpkg           [robustness]

P = number of deduped stations whose point falls inside the unit polygon.
is_border = unit lies within border_buffer_km of the Norway/Finland land border.
P=0 units are kept. Also computes the rural (outside-all-tätort) station share.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402

CRS = C.CONFIG["national_crs"]
BUF_M = C.CONFIG["border_buffer_km"] * 1000.0
LAU = C.CACHE / "lau_2024_3035.gpkg"
TAT = C.CACHE / "tatorter_2023.geojson"
SMA = C.CACHE / "smaorter_2023.geojson"
STATIONS = C.CACHE / "stations.gpkg"

WFS = "https://geodata.scb.se/geoserver/stat/wfs"
TAT_LAYER = "stat:Tatorter_2023"
SMA_LAYER = "stat:Smaorter_2023"
LAU_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/lau/gpkg/"
           "LAU_RG_01M_2024_3035.gpkg")


def wfs_url(layer):
    return (f"{WFS}?service=wfs&version=2.0.0&request=GetFeature&typeNames={layer}"
            f"&outputFormat=application/json&srsName=EPSG:3006")


def fetch_inputs():
    """Download SCB tätort/småort (WFS) and GISCO LAU if not already cached."""
    if not C.host_up("geodata.scb.se"):
        C.warn("geodata.scb.se unreachable — tätort/småort fetch may fail")
    C.download(wfs_url(TAT_LAYER), TAT, min_bytes=100_000,
               what=f"SCB WFS {TAT_LAYER}")
    C.update_manifest(C.manifest_entry_for(TAT, wfs_url(TAT_LAYER)))
    if C.CONFIG.get("include_smaorter", True):
        C.download(wfs_url(SMA_LAYER), SMA, min_bytes=100_000,
                   what=f"SCB WFS {SMA_LAYER}")
        C.update_manifest(C.manifest_entry_for(SMA, wfs_url(SMA_LAYER)))
    C.download(LAU_URL, LAU, min_bytes=1_000_000, what="GISCO LAU 2024")
    C.update_manifest(C.manifest_entry_for(LAU, LAU_URL))


def load_stations():
    s = gpd.read_file(STATIONS).to_crs(CRS)
    s = s[s.geometry.notna() & ~s.geometry.is_empty].copy()
    s["station_idx"] = np.arange(len(s))
    return s


def neighbor_border_buffer():
    """Buffered Norway+Finland land border, in national CRS."""
    nb = gpd.read_file(LAU, where="CNTR_CODE IN ('NO','FI')").to_crs(CRS)
    union = nb.geometry.union_all()
    return union.buffer(BUF_M)


def count_stations(units, stations, id_col):
    """Spatial join: count stations within each unit polygon. Keeps P=0 units."""
    j = gpd.sjoin(stations[["station_idx", "geometry"]], units[[id_col, "geometry"]],
                  how="inner", predicate="within")
    counts = j.groupby(id_col).size().rename("P")
    out = units.merge(counts, on=id_col, how="left")
    out["P"] = out["P"].fillna(0).astype(int)
    return out, set(j["station_idx"].unique())


def finalize(units, id_col, name_col, pop_col, stations, border_buf):
    u = units.copy()
    u = u[u.geometry.notna() & ~u.geometry.is_empty].copy()
    if id_col == name_col:
        u["unit_id"] = u[id_col].astype(str)
        u["name"] = u[name_col].astype(str)
        u = u.rename(columns={pop_col: "S"})
    else:
        u = u.rename(columns={id_col: "unit_id", name_col: "name", pop_col: "S"})
    u["unit_id"] = u["unit_id"].astype(str)
    u["S"] = pd.to_numeric(u["S"], errors="coerce")
    u = u[u["S"].notna()].copy()
    u["S"] = u["S"].astype(float)
    u, covered = count_stations(u, stations, "unit_id")
    u["is_border"] = u.geometry.intersects(border_buf)
    u = u[["unit_id", "name", "S", "P", "is_border", "geometry"]]
    return gpd.GeoDataFrame(u, geometry="geometry", crs=CRS), covered


def save(units, path):
    units.to_file(path, driver="GPKG")
    C.update_manifest(C.manifest_entry_for(path, "derived: units + station counts"))
    C.log(f"[OK] {Path(path).name}: n={len(units)} | P>0={int((units['P']>0).sum())} "
          f"| zeros={int((units['P']==0).sum())} | border={int(units['is_border'].sum())} "
          f"| sum P={int(units['P'].sum())}")


def build_fua(stations, border_buf, summary):
    """Best-effort Eurostat GISCO FUA (Urban Audit). Skip gracefully on failure."""
    out = C.CACHE / "units_fua.gpkg"
    if C.exists_nonempty(out):
        C.log(f"[SKIP] {out.name} exists")
        return
    try:
        import requests
        base = "https://gisco-services.ec.europa.eu/distribution/v2/urau"
        files = C.retry(lambda: requests.get(f"{base}/urau-2021-files.json", timeout=60).json(),
                        what="GISCO URAU files.json", tries=3)
        # pick a FUA gpkg in EPSG:3035
        gpkgs = files.get("gpkg", {})
        # need RG (region polygons), NOT LB (label points); prefer EPSG:3035
        fua_key = next((k for k in gpkgs if "FUA" in k.upper() and "RG" in k.upper()
                        and "3035" in k), None)
        if fua_key is None:
            fua_key = next((k for k in gpkgs if "FUA" in k.upper() and "RG" in k.upper()), None)
        if fua_key is None:
            raise RuntimeError(f"no FUA gpkg found; keys={list(gpkgs)[:6]}")
        url = f"{base}/gpkg/{fua_key}"
        dest = C.CACHE / fua_key.replace("/", "_")
        C.download(url, dest, min_bytes=10_000, what=f"GISCO {fua_key}")
        C.update_manifest(C.manifest_entry_for(dest, url))
        fua = gpd.read_file(dest)
        # filter to Sweden
        cc = next((c for c in fua.columns if c.upper() in ("CNTR_CODE", "CNTR_ID")), None)
        if cc:
            fua = fua[fua[cc].astype(str).str.upper().str.startswith(("SE",))].copy()
        fua = fua.to_crs(CRS)
        namec = next((c for c in fua.columns if "NAME" in c.upper()), fua.columns[0])
        idc = next((c for c in fua.columns if c.upper() in ("URAU_CODE", "FUA_CODE", "GISCO_ID")),
                   None) or namec
        popc = next((c for c in fua.columns if "POP" in c.upper() or "FUA_P" in c.upper()), None)
        pop_source = "GISCO attribute"
        if popc is None:
            # FUA boundary file carries no population -> aggregate kommun (LAU) population
            # by kommun centroid-in-FUA. FUAs are built from LAUs, so this is consistent.
            kom = gpd.read_file(LAU, where="CNTR_CODE='SE'").to_crs(CRS)
            kom_pts = kom.copy()
            kom_pts["geometry"] = kom.geometry.representative_point()
            jj = gpd.sjoin(kom_pts[["POP_2024", "geometry"]],
                           fua[[idc, "geometry"]], how="inner", predicate="within")
            popmap = (jj.groupby(idc)["POP_2024"].sum()
                      .reset_index().rename(columns={"POP_2024": "S_pop"}))
            fua = fua.merge(popmap, on=idc, how="left")
            popc = "S_pop"
            pop_source = "aggregated kommun POP_2024 (centroid-in-FUA)"
        u, _ = finalize(fua, idc, namec, popc, stations, border_buf)
        u = u[u["S"] > 0].copy()
        save(u, out)
        summary["fua"] = {"n": len(u), "pop_col": popc, "pop_source": pop_source, "source": url}
    except Exception as e:  # noqa: BLE001
        C.warn(f"FUA build failed (non-fatal, robustness extra): {e}")
        summary["fua"] = {"status": "failed", "error": str(e)}


def main():
    name = "Phase 2 — units + population"
    C.phase_start(name)
    fetch_inputs()
    stations = load_stations()
    n_stations = len(stations)
    border_buf = neighbor_border_buffer()
    summary = {"n_stations": n_stations}

    # (a) tätort
    out_t = C.CACHE / "units_tatort.gpkg"
    tat = gpd.read_file(TAT).to_crs(CRS)
    u_tat, covered_tat = finalize(tat, "tatortskod", "tatort", "bef", stations, border_buf)
    if not C.exists_nonempty(out_t):
        save(u_tat, out_t)
    else:
        C.log(f"[SKIP] {out_t.name} exists")

    # rural share = stations outside ALL tätort polygons
    rural = n_stations - len(covered_tat)
    summary["tatort"] = {"n": len(u_tat), "stations_in_tatort": len(covered_tat)}
    summary["rural_share"] = {
        "stations_total": n_stations,
        "stations_outside_tatort": rural,
        "rural_fraction": round(rural / n_stations, 4),
    }
    C.log(f"rural share: {rural}/{n_stations} = {rural/n_stations:.1%} stations outside all tätorter")

    # (b) tätort + småort
    out_ts = C.CACHE / "units_tatort_smaort.gpkg"
    if not C.exists_nonempty(out_ts):
        sma = gpd.read_file(SMA).to_crs(CRS)
        u_sma, _ = finalize(sma, "smaort", "smaort", "bef", stations, border_buf)
        u_sma["unit_id"] = "SMA_" + u_sma["unit_id"].astype(str)
        combined = gpd.GeoDataFrame(
            pd.concat([u_tat.assign(kind="tatort"), u_sma.assign(kind="smaort")],
                      ignore_index=True),
            geometry="geometry", crs=CRS)
        save(combined, out_ts)
        summary["tatort_smaort"] = {"n": len(combined)}
    else:
        C.log(f"[SKIP] {out_ts.name} exists")

    # (c) kommun (LAU)
    out_k = C.CACHE / "units_kommun.gpkg"
    if not C.exists_nonempty(out_k):
        lau = gpd.read_file(LAU, where="CNTR_CODE='SE'").to_crs(CRS)
        u_kom, covered_kom = finalize(lau, "GISCO_ID", "LAU_NAME", "POP_2024", stations, border_buf)
        save(u_kom, out_k)
        summary["kommun"] = {"n": len(u_kom), "stations_assigned": len(covered_kom),
                             "stations_total": n_stations}
        C.log(f"kommun coverage: {len(covered_kom)}/{n_stations} stations fell in a kommun")
    else:
        C.log(f"[SKIP] {out_k.name} exists")

    # (d) FUA (best-effort)
    build_fua(stations, border_buf, summary)

    C.write_json(C.OUTPUT / "units_summary.json", summary)
    C.phase_end(name)


if __name__ == "__main__":
    C.set_seed()
    main()

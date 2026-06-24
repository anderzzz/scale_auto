# Methods

National CRS: **EPSG:3006** (SWEREF99 TM, metric). All geometries reprojected to it. Random seed 17.

## Datasets used (exact)

| Role | Source | Layer / file | Vintage | Key fields |
|---|---|---|---|---|
| Petrol stations P | Geofabrik OSM extract | `sweden-latest.osm.pbf`, `amenity=fuel` (nodes + way centroids), MD5-verified | OSM snapshot 2026-06 | `amenity`, `brand`, `operator`, `access` |
| Tätorter (primary) | SCB GeoServer WFS `geodata.scb.se/geoserver/stat/wfs` | `stat:Tatorter_2023` (WFS 2.0.0 GetFeature, EPSG:3006) | 2023 | `tatortskod`, `tatort`, `bef` (pop), `kommun`, `lan` |
| Småorter (low-S tail) | same WFS | `stat:Smaorter_2023` | 2023 | `smaort`, `bef` (50–199 inhabitants) |
| Kommun boundaries + pop | Eurostat GISCO LAU | `LAU_RG_01M_2024_3035.gpkg`, `CNTR_CODE='SE'` (290 units) | 2024 | `GISCO_ID`, `LAU_NAME`, `POP_2024` |
| FUA (robustness) | Eurostat GISCO Urban Audit | `URAU_RG_100K_2021_3035_FUA.gpkg`, `CNTR_CODE='SE'` (12 FUAs) | 2021 | `URAU_CODE`, `URAU_NAME`; population aggregated from LAU |
| Land border (leakage flag) | GISCO LAU NO+FI polygons, dissolved, buffered 10 km | 2024 | `CNTR_CODE` |
| Registered cars (denominator swap) | SCB PXWeb / Trafikanalys `TK/TK1001/TK1001A/FordonTrafik`, passenger cars in use, per kommun | latest year | `Region`, `Fordonsslag=10` |

## Follow-up analyses (Phases 6–7)

- **Station typology (Phase 6):** each station is labelled automat (unmanned) / full-service / vehicle-gas-biogas / unknown by a documented brand keyword map (approximate — used only for deviation inspection, never to split β). Residuals are log(observed/expected) from the kommun NB GLM; a Dorling cartogram (area ∝ population, iterative overlap repulsion) shows them population-weighted.
- **Denominator swap (Phase 7):** the kommun NB GLM is refit with registered passenger cars as S instead of population, to test how much of the sublinearity is lower urban car ownership rather than provision economies.

## Parameters

- `dedup_radius_m` = 50
- `border_buffer_km` = 10
- `expected_station_total` = 2700
- `station_total_tolerance` = [1800, 3600]
- `bootstrap_n` = 1000
- `include_smaorter` = True

## Procedure

1. Stream `amenity=fuel` from the PBF with pyosmium (low-memory `FileProcessor` with node locations); way/area features reduced to polygon centroids.
2. Drop `access=private`; reproject to EPSG:3006; dedupe by clustering points within 50 m (KD-tree pairs → connected components), keeping one representative (preferring branded/named) at the cluster centroid.
3. For each unit definition build (unit_id, name, S, P, geometry, is_border); P = stations spatially within the polygon; **P=0 units retained**; is_border = within buffer of the NO/FI land border.
4. Model A = NB GLM (Poisson where not overdispersed, dispersion reported), log link, log E[P]=a+β·log S; H₀:β=1 via z=(β̂−1)/SE. Model B = positives-only OLS of log10 P on log10 S. Bootstrap β (resample units, refit). (log S)² term + segmented breakpoint for the single-power-law check. Leakage = refit minus border units.

## Temporal mismatch & limitations

- OSM fuel = continuously edited ~2026 snapshot; population = SCB 2023 (tätort/småort) and GISCO 2024 (kommun) / 2021 (FUA delineation). Stations opening or closing between vintages are not reflected in S.
- OSM completeness is uneven; brand/operator tagging is partial (hence an '(unbranded/unknown)' bucket).
- FUA population is approximated (kommun aggregation); FUA truncates the size range.
- The land-border leakage flag captures Norway/Finland but not the Öresund sea crossing to Denmark.

## OSM brand histogram (top, deduped)

- OKQ8: 478
- Preem: 462
- (unbranded/unknown): 424
- St1: 365
- Circle K: 336
- Ingo: 245
- Qstar: 216
- Tanka: 157
- din-X: 104
- Gulf: 55
- Bilisten: 20
- Din-X: 16
- Oljeshejkerna Johnsson AB: 13
- SÅIFA: 11
- Fordonsgas: 11

## Rural share
512 of 3263 stations (15.7%) fall outside all tätorter.

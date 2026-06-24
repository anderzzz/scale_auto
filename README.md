# Petrol-station population scaling for Sweden

Estimates the scaling exponent **β** in **P = C · Sᵝ**, where *P* = number of petrol
stations in a spatial unit and *S* = its population, and tests whether scaling is
**sublinear (β < 1)**. Because β depends on how spatial units are drawn (the
**Modifiable Areal Unit Problem, MAUP**) and on cross-boundary usage, β is estimated
under **four unit definitions** and the spread across them is reported as a primary
result.

## Headline result

| Unit definition | β | 95% CI | verdict |
|---|--:|---|---|
| Tätort (settlement) | **0.78** | [0.76, 0.79] | sublinear |
| Tätort + småort | **0.81** | [0.79, 0.82] | sublinear |
| FUA (functional) | **0.66** | [0.61, 0.71] | sublinear |
| Kommun (administrative) | **0.59** | [0.55, 0.64] | sublinear |

All four definitions are sublinear and exclude β = 1, but the magnitude swings from
0.59 to 0.81 depending on the unit — that MAUP spread is the point. Full write-up in
[`output/report.md`](output/report.md).

## Run

```bash
./run_all.sh
```

Creates a Python 3.12 venv if needed, then runs the phases. Every phase **skips work
whose output already exists and is non-empty**, so re-runs are cheap and a partial run
can be resumed. The report is **regenerated after every phase**, so a late failure
still leaves a valid (partial) deliverable. Single random seed = 17.

## Pipeline

| Phase | Script | Output |
|---|---|---|
| 1 | `scripts/01_stations.py` | `cache/stations.gpkg` — deduped OSM `amenity=fuel` (P) |
| 2 | `scripts/02_units.py` | `cache/units_*.gpkg` — units with S, P, is_border |
| 3 | `scripts/03_regressions.py` | `output/results.json` — NB/Poisson GLM, OLS, bootstrap |
| 4 | `scripts/04_charts.py` | `output/charts/*.png` |
| 5 | `scripts/05_report.py` | `output/report.md`, `output/methods.md` |
| — | `scripts/99_summary.py` | stdout summary |

`scripts/common.py` holds config loading, timestamped logging to `logs/status.log`,
retry-with-exponential-backoff downloads, and the manifest writer.

## Data sources

- **Stations (P):** Geofabrik `sweden-latest.osm.pbf`, `amenity=fuel` (nodes + way
  centroids), MD5-verified, streamed with pyosmium.
- **Tätorter / småorter (S):** SCB GeoServer WFS `geodata.scb.se/geoserver/stat/wfs`
  (`stat:Tatorter_2023`, `stat:Smaorter_2023`), population field `bef`.
- **Kommuner (S + boundaries):** Eurostat GISCO LAU 2024 (`CNTR_CODE='SE'`, `POP_2024`).
- **FUA:** Eurostat GISCO Urban Audit 2021 FUA polygons; population aggregated from LAU.

See [`output/methods.md`](output/methods.md) for exact layers, vintages, field names,
parameters and limitations, and [`output/manifest.json`](output/manifest.json) for
every downloaded file.

## Method notes

- **P = 0 units are kept** (dropping zeros biases β upward by survivorship). Model A is a
  Negative-Binomial / Poisson count GLM with log link; H₀: β = 1 is tested with
  z = (β̂ − 1)/SE. Model B (positives-only OLS) is reported only to compare with the
  classic US literature and is flagged as zero-truncation-biased.
- `config.yaml` holds all parameters (dedup radius, border buffer, completeness band,
  bootstrap N, seed).

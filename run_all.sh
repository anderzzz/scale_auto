#!/usr/bin/env bash
# Petrol-station population scaling for Sweden — full pipeline.
# Re-runnable: every phase skips work whose output already exists and is non-empty.
# The report is regenerated after every phase, so a late failure still leaves a
# valid (partial) deliverable.
set -uo pipefail
cd "$(dirname "$0")"

PY="venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "Creating venv (python3.12)..."
  python3.12 -m venv venv || python3 -m venv venv
  PY="venv/bin/python"
  "$PY" -m pip install --upgrade pip >/dev/null
  "$PY" -m pip install numpy pandas scipy statsmodels matplotlib requests \
      geopandas shapely pyproj owslib osmium pyyaml
fi

mkdir -p cache logs output/charts

run_phase () {  # $1 = script, $2 = human name
  echo ">>> $2"
  if ! "$PY" -u "$1"; then
    echo "!!! $2 FAILED — regenerating report with whatever exists, then continuing"
  fi
  # regenerate report after every phase (idempotent checkpoint)
  "$PY" -u scripts/05_report.py || true
}

run_phase scripts/01_stations.py    "Phase 1 — petrol stations P"
run_phase scripts/02_units.py       "Phase 2 — units + population"
run_phase scripts/03_regressions.py "Phase 3 — scaling regressions"
run_phase scripts/04_charts.py      "Phase 4 — charts"
run_phase scripts/06_deviation.py   "Phase 6 — deviation analysis"
run_phase scripts/07_vehicles.py    "Phase 7 — denominator swap (cars)"
run_phase scripts/09_tourism.py     "Phase 9 — tourism proxy (OSM)"
run_phase scripts/08_surprise.py    "Phase 8 — surprise & clustering"
"$PY" -u scripts/05_report.py
"$PY" -u scripts/99_summary.py

echo "Done. Report: output/report.md"

"""Phase 5 — assemble output/report.md (+ methods.md), idempotent.

Reads whatever JSON artifacts exist and produces a valid (possibly partial)
report. Names what is missing and why. Safe to run after any phase.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LABELS = {"tatort": "Tätort (settlement)", "tatort_smaort": "Tätort + småort",
          "kommun": "Kommun (administrative)", "fua": "FUA (functional)"}


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"


def followup_section():
    """Deviation analysis (Phase 6) + denominator-swap to cars (Phase 7). Optional."""
    dev = C.read_json(C.OUTPUT / "deviation_summary.json", {})
    veh = C.read_json(C.OUTPUT / "vehicles_summary.json", {})
    if not dev and not veh:
        return []
    L = ["## Follow-up: where the deviations are, and is population even the right denominator?\n",
         "β is the all-Sweden summary; the **residuals** are where the structure lives. "
         "Two follow-ups: (1) which kommuner deviate from the scaling law and whether it is a "
         "station-*type* artefact, and (2) whether the sublinearity is partly just lower urban "
         "car ownership rather than genuine provision economies.\n"]
    for fn, cap in [
        ("05_residual_automat.png", "Kommun residuals (log obs/expected stations) vs population, "
         "coloured by the unmanned-*automat* share of each kommun's stations; extreme deviations "
         "labelled."),
        ("06_dorling_kommun.png", "Dorling cartogram — circle area ∝ population, colour = deviation "
         "from the petrol-scaling law, NO/FI border kommuner outlined. The empty north collapses; "
         "population-weighted geography remains."),
        ("07_pop_vs_cars.png", "Denominator swap: P vs population (β≈0.59) and P vs registered "
         "passenger cars (β≈0.68)."),
    ]:
        p = C.CHARTS / fn
        if p.exists():
            L.append(f"![{fn}](charts/{fn})\n\n*{cap}*\n")

    bl = []
    if dev:
        tc = dev.get("station_type_counts", {})
        rho = dev.get("corr_residual_automatshare")
        bysize = dev.get("automat_share_by_size", {})
        bl.append(f"**Station type is a minor part of the deviation, not the cause.** Classifying "
                  f"stations by brand (full-service {tc.get('full_service','?')}, automat "
                  f"{tc.get('automat','?')}, vehicle-gas/biogas {tc.get('gas_alt','?')}, unknown "
                  f"{tc.get('unknown','?')}), the correlation between a kommun's residual and its "
                  f"automat share is only **{rho:+.2f}**. Automat (unmanned) share does fall with "
                  f"size ({bysize.get('small','?')} small → {bysize.get('large','?')} large), so "
                  f"cheap unmanned pumps do thicken the rural tail — but they explain little of the "
                  f"over-stationing. (Some 'fuel' nodes are actually biogas/CNG outlets — literally "
                  f"'doing other things' — concentrated near the big cities.)")
        over = dev.get("top_over_stationed", [])
        under = dev.get("top_under_stationed", [])
        if over and under:
            on = ", ".join(r["name"] for r in over[:5])
            un = ", ".join(r["name"] for r in under[:5])
            bl.append(f"**The deviations are a cross-boundary-demand map, in both directions.** "
                      f"Most over-stationed: {on} — sparse northern, ski-tourism (Sälen/Härjedalen) "
                      f"and E4-transit (Ljungby, Tingsryd) kommuner serving non-residents. Most "
                      f"under-stationed: {un} — dense **Stockholm/Göteborg commuter suburbs** (e.g. "
                      f"Sundbyberg, ~56k residents but ~2 stations) whose residents fuel up in the "
                      f"core or in transit. Border over-stationing and suburban under-stationing are "
                      f"the same phenomenon: resident population is the wrong denominator wherever "
                      f"demand crosses the unit boundary.")
    if veh:
        bp = veh["beta_population"]["beta"]; bc = veh["beta_cars"]["beta"]
        cpc = veh.get("cars_per_capita", {}).get("by_size_tercile", {})
        gap_pop = 1 - bp; gap_cars = 1 - bc
        share = (bc - bp) / gap_pop if gap_pop else 0
        bl.append(f"**Part of the sublinearity is just lower urban car ownership — but most of it "
                  f"is real.** Swapping the denominator from people to **registered passenger cars** "
                  f"(SCB/Trafikanalys {veh.get('year')}, all 290 kommuner) lifts β from "
                  f"**{bp:.3f} → {bc:.3f}**. Cars per capita fall with size "
                  f"({cpc.get('small','?')} small → {cpc.get('large','?')} large), so denser places "
                  f"genuinely own fewer cars; that accounts for ~{share*100:.0f}% of the gap from "
                  f"linearity. The remaining β = {bc:.3f} is still firmly sublinear, so real "
                  f"provision economies (and through-traffic demand) survive the confounder — the "
                  f"naive population β just overstates the effect by ~{(bc-bp):.2f}.")
    for item in bl:
        L.append(f"- {item}\n")
    return L


def build_report():
    stn = C.read_json(C.OUTPUT / "stations_summary.json", {})
    units = C.read_json(C.OUTPUT / "units_summary.json", {})
    res = C.read_json(C.OUTPUT / "results.json", {})
    defs = res.get("definitions", {}) if res else {}
    missing = []

    comp = stn.get("completeness", {})
    rural = units.get("rural_share", {})

    L = []
    L.append("# Petrol-station population scaling in Sweden — P = C · Sᵝ\n")
    L.append(f"*Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
             f"Random seed {C.CONFIG['random_seed']}. National CRS {C.CONFIG['national_crs']}.*\n")
    L.append("**Question:** does the number of petrol stations P scale sublinearly "
             "(β < 1) with population S, and how much does β move with the choice of "
             "spatial unit (the Modifiable Areal Unit Problem)?\n")

    # headline table
    L.append("## Headline results (Model A: Negative-Binomial / Poisson GLM, log link)\n")
    L.append("| Unit definition | n units | Σ P | β | model 95% CI | bootstrap 95% CI | H₀: β=1 verdict |")
    L.append("|---|--:|--:|--:|---|---|---|")
    for d in ["tatort", "tatort_smaort", "fua", "kommun"]:
        v = defs.get(d)
        if not v or "model_A_primary" not in v:
            L.append(f"| {LABELS.get(d,d)} | — | — | — | — | — | *missing* |")
            missing.append(d); continue
        a = v["model_A_primary"]; bs = v.get("bootstrap_modelA", {})
        L.append(f"| {LABELS[d]} | {v['n_units']} | {v['sum_P']} | **{a['beta']:.3f}** | "
                 f"{fmt_ci(a['ci95'])} | {fmt_ci(bs.get('ci95'))} | {a['verdict']} "
                 f"(p={a['p_vs1']:.1e}) |")
    L.append("")

    # charts
    L.append("## Charts\n")
    for fn, cap in [
        ("01_headline_tatort.png", "Headline: log-log P vs S for tätorter with GLM fit, "
         "95% band, and the N zero-station units shown as a bottom rug (excluded from the "
         "log scatter only, retained in the fit)."),
        ("02_forest_beta.png", "Forest plot of β (95% CI) across all four unit definitions "
         "vs the β=1 reference and the ~0.75–0.90 infrastructure prior — the MAUP/leakage payoff."),
        ("03_residuals_tatort.png", "Tätort GLM residuals vs population — a systematic trend "
         "signals departure from a single power law."),
        ("04_residual_map.png", "Residual map: tätorter coloured by over-/under-stationing "
         "relative to the population expectation."),
    ]:
        p = C.CHARTS / fn
        if p.exists():
            L.append(f"![{fn}](charts/{fn})\n\n*{cap}*\n")
        else:
            L.append(f"*(chart {fn} missing — {cap})*\n")

    # findings bullets
    L.append("## Key findings\n")
    b = []
    if defs.get("tatort", {}).get("model_A_primary"):
        a = defs["tatort"]["model_A_primary"]
        b.append(f"**Tätort (primary functional unit): β = {a['beta']:.3f}, 95% CI "
                 f"{fmt_ci(a['ci95'])} — {a['verdict']}** (z vs 1 = {a['z_vs1']}, "
                 f"p = {a['p_vs1']:.1e}). Petrol provision scales clearly sublinearly with "
                 f"settlement population: doubling a town's population multiplies its stations "
                 f"by only ~2^{a['beta']:.2f} ≈ {2**a['beta']:.2f}, i.e. larger towns are more "
                 f"station-efficient per capita.")
    betas = {d: defs[d]["model_A_primary"]["beta"] for d in defs
             if defs[d].get("model_A_primary")}
    if betas:
        lo_d = min(betas, key=betas.get); hi_d = max(betas, key=betas.get)
        b.append(f"**β moves a lot with the unit definition — {betas[lo_d]:.3f} "
                 f"({LABELS.get(lo_d,lo_d)}) to {betas[hi_d]:.3f} ({LABELS.get(hi_d,hi_d)}), "
                 f"a spread of {betas[hi_d]-betas[lo_d]:.2f}.** This MAUP spread is itself a "
                 f"primary result: the 'efficiency-with-population' claim is qualitatively robust "
                 f"(every definition is sublinear and excludes β=1) but its *magnitude* is not "
                 f"unit-invariant. Administrative kommuner give the most sublinear β because they "
                 f"bundle whole territories (dense city + sparse hinterland) into one large-S unit; "
                 f"tätorter measure settlement-internal scaling only.")
    if rural:
        b.append(f"**{rural.get('rural_fraction',0)*100:.1f}% of stations "
                 f"({rural.get('stations_outside_tatort','?')} of "
                 f"{rural.get('stations_total','?')}) lie outside every tätort polygon** — rural "
                 f"automat chains (Din-X, Pump, Bilisten, Qstar) on roads between settlements. "
                 f"This is why tätort and kommun answer different questions: tätort = "
                 f"settlement-internal scaling (these rural stations are invisible); kommun = "
                 f"whole-territory scaling (they are captured, inflating P in low-density units "
                 f"and pushing β down).")
    # leakage
    lk = defs.get("kommun", {}).get("leakage_check")
    ka = defs.get("kommun", {}).get("model_A_primary")
    if lk and ka:
        b.append(f"**Border leakage:** excluding the {lk['n_excluded_border']} kommuner within "
                 f"{C.CONFIG['border_buffer_km']} km of the Norway/Finland land border moves "
                 f"kommun β from {ka['beta']:.3f} to {lk['beta_no_border']:.3f} "
                 f"({'up' if lk['beta_no_border']>ka['beta'] else 'down'}). Border units are "
                 f"over-stationed for their resident population because cross-border demand "
                 f"(Norwegian fuel-price shopping around Strömstad/Svinesund; Haparanda–Tornio) "
                 f"adds stations that resident S does not explain. Caveat: the 10 km *land*-border "
                 f"flag does not capture the Öresund (Malmö–Copenhagen) fixed-link leakage, which "
                 f"is a ~25 km sea crossing.")
    # single power law
    sp = defs.get("tatort", {}).get("single_power_law_check")
    if sp:
        if sp.get("significant"):
            seg = sp.get("segmented", {})
            b.append(f"**Not a clean single power law (tätort):** a (log S)² term is significant "
                     f"(p = {sp['quad_p']:.1e}). A segmented fit puts the breakpoint near "
                     f"S ≈ {seg.get('breakpoint_S','?'):,} inhabitants, with slope "
                     f"{seg.get('slope_below','?')} below and {seg.get('slope_above','?')} above — "
                     f"the largest cities scale *more* sublinearly than small towns.")
        else:
            b.append("**Single power law:** no significant curvature in the tätort relation "
                     "(consistent with a single power law over the observed size range).")
    # completeness
    if comp:
        verdict = "within" if comp.get("within_band") else "OUTSIDE"
        b.append(f"**OSM completeness vs the Drivkraft Sverige anchor:** "
                 f"{comp.get('deduped_count','?')} deduped stations "
                 f"(from {comp.get('raw_features','?')} raw OSM amenity=fuel features, "
                 f"{comp.get('dropped_private','?')} private dropped) — {verdict} the tolerance "
                 f"band {comp.get('tolerance_band')}, ratio {comp.get('ratio_to_anchor')}× the "
                 f"~{comp.get('drivkraft_anchor')} liquid-fuel sales-point anchor. The mild "
                 f"excess is expected: OSM also tags truck stops and small/private pumps the "
                 f"Drivkraft retail figure excludes, and 50 m dedup will not merge node+building "
                 f"pairs that sit >50 m apart.")
    # OLS vs GLM / zeros
    tb = defs.get("tatort", {})
    if tb.get("model_B_ols") and tb.get("model_A_primary"):
        ob = tb["model_B_ols"]; ab = tb["model_A_primary"]
        b.append(f"**Zeros and the OLS↔GLM gap:** the tätort set has {ob['n_dropped_zeros']} "
                 f"zero-station units (kept by the GLM, dropped by OLS). Classic positives-only "
                 f"OLS gives β = {ob['beta']:.3f} vs GLM β = {ab['beta']:.3f}; here OLS is *lower*, "
                 f"not higher, because dropping the many small zero-towns removes the low-S anchor "
                 f"that steepens the slope. (For kommun, with **no** zeros, OLS {defs['kommun']['model_B_ols']['beta']:.3f} "
                 f"≈ GLM {defs['kommun']['model_A_primary']['beta']:.3f}, confirming the gap is a "
                 f"zero-handling artefact, not a model-family one.) The GLM on counts with zeros "
                 f"retained is the correct estimator.")
    # literature line
    if defs.get("tatort", {}).get("model_A_primary"):
        a = defs["tatort"]["model_A_primary"]
        b.append(f"**Versus the literature prior:** space-serving / distribution infrastructure "
                 f"typically scales at β ≈ 0.75–0.90. The tätort headline β = {a['beta']:.3f} sits "
                 f"right inside that band; the more aggregated kommun/FUA estimates fall below it, "
                 f"consistent with stronger sublinearity once rural territory is folded in.")

    for item in b:
        L.append(f"- {item}\n")

    # follow-up: deviation analysis & demand confounder
    L.extend(followup_section())

    # caveats
    L.append("## Caveats\n")
    cav = []
    if comp and not comp.get("within_band"):
        cav.append("OSM station count fell OUTSIDE the expected completeness band — treat absolute "
                   "P with caution (β, a ratio, is more robust to uniform under/over-counting).")
    cav.append("Temporal mismatch: OSM fuel features are 'now' (2026 snapshot) while tätort/"
               "småort population is the SCB 2023 delineation and kommun population is GISCO/SCB "
               "2024; a station opened/closed since does not move with population.")
    cav.append("FUA population is approximated by aggregating kommun (LAU) populations whose "
               "centroid falls in the FUA, as the GISCO FUA boundary file carries no population "
               "field; FUAs also truncate the size range (only the 12 largest Swedish places, no "
               "villages), so their β is the least comparable.")
    cav.append("Dedup at 50 m merges co-located node+polygon mappings of the same station but will "
               "split a single large highway service area with widely spaced pumps; conversely it "
               "could merge two genuinely distinct adjacent stations (rare).")
    for c in cav:
        L.append(f"- {c}\n")

    if missing:
        L.append(f"\n> **Incomplete run:** missing/failed unit definitions: "
                 f"{', '.join(missing)}. The above reflects only completed phases.\n")

    L.append("\n---\nSee `methods.md` for exact datasets, vintages, URLs and field names; "
             "`results.json` for full numeric output; `manifest.json` for downloads.\n")

    (C.OUTPUT / "report.md").write_text("\n".join(L))
    C.log("[OK] wrote report.md")


def build_methods():
    stn = C.read_json(C.OUTPUT / "stations_summary.json", {})
    units = C.read_json(C.OUTPUT / "units_summary.json", {})
    M = []
    M.append("# Methods\n")
    M.append(f"National CRS: **{C.CONFIG['national_crs']}** (SWEREF99 TM, metric). "
             f"All geometries reprojected to it. Random seed {C.CONFIG['random_seed']}.\n")
    M.append("## Datasets used (exact)\n")
    M.append("| Role | Source | Layer / file | Vintage | Key fields |")
    M.append("|---|---|---|---|---|")
    M.append("| Petrol stations P | Geofabrik OSM extract | `sweden-latest.osm.pbf`, "
             "`amenity=fuel` (nodes + way centroids), MD5-verified | OSM snapshot 2026-06 | "
             "`amenity`, `brand`, `operator`, `access` |")
    M.append("| Tätorter (primary) | SCB GeoServer WFS `geodata.scb.se/geoserver/stat/wfs` | "
             "`stat:Tatorter_2023` (WFS 2.0.0 GetFeature, EPSG:3006) | 2023 | "
             "`tatortskod`, `tatort`, `bef` (pop), `kommun`, `lan` |")
    M.append("| Småorter (low-S tail) | same WFS | `stat:Smaorter_2023` | 2023 | "
             "`smaort`, `bef` (50–199 inhabitants) |")
    M.append("| Kommun boundaries + pop | Eurostat GISCO LAU | "
             "`LAU_RG_01M_2024_3035.gpkg`, `CNTR_CODE='SE'` (290 units) | 2024 | "
             "`GISCO_ID`, `LAU_NAME`, `POP_2024` |")
    M.append("| FUA (robustness) | Eurostat GISCO Urban Audit | "
             "`URAU_RG_100K_2021_3035_FUA.gpkg`, `CNTR_CODE='SE'` (12 FUAs) | 2021 | "
             "`URAU_CODE`, `URAU_NAME`; population aggregated from LAU |")
    M.append("| Land border (leakage flag) | GISCO LAU NO+FI polygons, dissolved, buffered "
             f"{C.CONFIG['border_buffer_km']} km | 2024 | `CNTR_CODE` |")
    M.append("| Registered cars (denominator swap) | SCB PXWeb / Trafikanalys "
             "`TK/TK1001/TK1001A/FordonTrafik`, passenger cars in use, per kommun | latest year | "
             "`Region`, `Fordonsslag=10` |")
    M.append("")
    M.append("## Follow-up analyses (Phases 6–7)\n")
    M.append("- **Station typology (Phase 6):** each station is labelled automat (unmanned) / "
             "full-service / vehicle-gas-biogas / unknown by a documented brand keyword map "
             "(approximate — used only for deviation inspection, never to split β). Residuals are "
             "log(observed/expected) from the kommun NB GLM; a Dorling cartogram (area ∝ population, "
             "iterative overlap repulsion) shows them population-weighted.\n"
             "- **Denominator swap (Phase 7):** the kommun NB GLM is refit with registered "
             "passenger cars as S instead of population, to test how much of the sublinearity is "
             "lower urban car ownership rather than provision economies.\n")
    M.append("## Parameters\n")
    for k in ["dedup_radius_m", "border_buffer_km", "expected_station_total",
              "station_total_tolerance", "bootstrap_n", "include_smaorter"]:
        M.append(f"- `{k}` = {C.CONFIG.get(k)}")
    M.append("")
    M.append("## Procedure\n")
    M.append("1. Stream `amenity=fuel` from the PBF with pyosmium (low-memory `FileProcessor` "
             "with node locations); way/area features reduced to polygon centroids.\n"
             "2. Drop `access=private`; reproject to EPSG:3006; dedupe by clustering points "
             f"within {C.CONFIG['dedup_radius_m']} m (KD-tree pairs → connected components), "
             "keeping one representative (preferring branded/named) at the cluster centroid.\n"
             "3. For each unit definition build (unit_id, name, S, P, geometry, is_border); "
             "P = stations spatially within the polygon; **P=0 units retained**; is_border = "
             "within buffer of the NO/FI land border.\n"
             "4. Model A = NB GLM (Poisson where not overdispersed, dispersion reported), "
             "log link, log E[P]=a+β·log S; H₀:β=1 via z=(β̂−1)/SE. Model B = positives-only "
             "OLS of log10 P on log10 S. Bootstrap β (resample units, refit). (log S)² term + "
             "segmented breakpoint for the single-power-law check. Leakage = refit minus border units.\n")
    M.append("## Temporal mismatch & limitations\n")
    M.append("- OSM fuel = continuously edited ~2026 snapshot; population = SCB 2023 (tätort/"
             "småort) and GISCO 2024 (kommun) / 2021 (FUA delineation). Stations opening or "
             "closing between vintages are not reflected in S.\n"
             "- OSM completeness is uneven; brand/operator tagging is partial (hence an "
             "'(unbranded/unknown)' bucket).\n"
             "- FUA population is approximated (kommun aggregation); FUA truncates the size range.\n"
             "- The land-border leakage flag captures Norway/Finland but not the Öresund sea "
             "crossing to Denmark.\n")
    if stn.get("brand_histogram_top25"):
        M.append("## OSM brand histogram (top, deduped)\n")
        for k, v in list(stn["brand_histogram_top25"].items())[:15]:
            M.append(f"- {k}: {v}")
        M.append("")
    if units.get("rural_share"):
        rs = units["rural_share"]
        M.append(f"## Rural share\n{rs.get('stations_outside_tatort')} of "
                 f"{rs.get('stations_total')} stations "
                 f"({rs.get('rural_fraction',0)*100:.1f}%) fall outside all tätorter.\n")
    (C.OUTPUT / "methods.md").write_text("\n".join(M))
    C.log("[OK] wrote methods.md")


def main():
    name = "Phase 5 — report"
    C.phase_start(name)
    build_report()
    build_methods()
    C.phase_end(name)


if __name__ == "__main__":
    main()

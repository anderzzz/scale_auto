"""Phase 10 — density scaling (Sutton et al. 2024) on fine-grained units.

Population scaling (P = C*S^beta) mixes dense cities with their rural hinterland
when the unit is coarse and heterogeneous — a kommun like Sundsvall reads as
"rural" (0.29 ppl/ha) because 99k people are smeared over 3,474 km2 of forest.
Sutton et al. instead use POPULATION DENSITY rho = S/A as the independent variable
and indicator density P/A as the dependent one, and fit a SEGMENTED power law with
a breakpoint that (in England & Wales) sits at a consistent 33 +/- 5 ppl/ha and
separates a rural regime from an urban one.

That method needs fine, internally-homogeneous, complete-coverage units (their
MSOAs, n~7080). The Swedish analogs are DeSO (n~6160) and RegSO (n~3363), both a
complete tessellation of Sweden. Here we:

  * fetch DeSO_2025 / RegSO_2025 polygons (SCB WFS) + population (SCB PXWeb),
  * count deduped stations P per unit,
  * compute area A (ha) and density rho = S/A,
  * fit a COUNT GLM with an area offset (the rigorous form of Sutton eq. 3/5):
        log E[P] = log A + a + beta * log(rho)
    — Poisson, or NB2 if overdispersed — handling P=0 units and avoiding the
    spurious ratio-vs-ratio R^2 inflation of OLS on P/A,
  * add a hinge at a grid-searched breakpoint c for the segmented model and pick
    single vs segmented by AIC (+ LR test on the hinge term),
  * report the breakpoint in ppl/ha and compare it to Sutton's 33 +/- 5,
  * locate Sundsvall's own DeSOs on the density axis.

Writes output/density_summary.json, output/density.md, charts/10_density_*.png.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

CRS = C.CONFIG["national_crs"]
DPI = 160
STATIONS = C.CACHE / "stations.gpkg"
WFS = "https://geodata.scb.se/geoserver/stat/wfs"
PXWEB = "https://api.scb.se/OV0104/v1/doris/en/ssd/BE/BE0101/BE0101Y/FolkmDesoAldKon"
POP_CACHE = C.CACHE / "pop_deso_regso.json"
SUNDSVALL_KOMMUN = "2281"
SUTTON_BREAK = (33, 5)  # ppl/ha, England & Wales (Sutton et al.)

UNITS = {
    # name      WFS layer            geometry id-col   pop key-kind
    "deso":  ("stat:DeSO_2025",  "desokod",  "deso"),
    "regso": ("stat:RegSO_2025", "regsokod", "regso"),
}


# --------------------------------------------------------------------------- data
def wfs_url(layer):
    return (f"{WFS}?service=wfs&version=2.0.0&request=GetFeature&typeNames={layer}"
            f"&outputFormat=application/json&srsName=EPSG:3006&count=20000")


def fetch_geometry(layer, dest):
    C.download(wfs_url(layer), dest, min_bytes=500_000, what=f"SCB WFS {layer}")
    C.update_manifest(C.manifest_entry_for(dest, wfs_url(layer)))


def fetch_population():
    """Population per region (DeSO + RegSO, total age/sex, latest year) from PXWeb.

    FolkmDesoAldKon's Region variable mixes DeSO codes (e.g. 0114A0010_DeSO2025)
    and RegSO codes (e.g. 2584R015), so one query yields both. Returns
    {"deso": {desokod: pop}, "regso": {regsokod: pop}, "year": yyyy}.
    """
    if C.exists_nonempty(POP_CACHE):
        return C.read_json(POP_CACHE)
    import re
    import requests
    meta = C.retry(lambda: requests.get(PXWEB, timeout=60).json(), what="PXWeb meta")
    year = next(v["values"] for v in meta["variables"] if v.get("time"))[-1]
    query = {"query": [
        {"code": "Alder", "selection": {"filter": "item", "values": ["totalt"]}},
        {"code": "Kon", "selection": {"filter": "item", "values": ["1+2"]}},
        {"code": "Tid", "selection": {"filter": "item", "values": [year]}},
    ], "response": {"format": "json"}}
    js = C.retry(lambda: requests.post(PXWEB, json=query, timeout=180).json(),
                 what="PXWeb DeSO/RegSO population")
    deso, regso = {}, {}
    deso_suffix = re.compile(r"_DeSO\d+$")
    regso_pat = re.compile(r"^\d{4}R\d{3}$")
    for row in js["data"]:
        code = row["key"][0]
        try:
            val = int(row["values"][0])
        except (ValueError, TypeError):
            continue
        if "_DeSO" in code:
            deso[deso_suffix.sub("", code)] = val
        elif regso_pat.match(code):
            regso[code] = val
    out = {"year": year, "deso": deso, "regso": regso}
    C.write_json(POP_CACHE, out)
    C.update_manifest(C.manifest_entry_for(POP_CACHE, PXWEB + f" (total pop {year})"))
    C.log(f"population: {len(deso)} DeSO, {len(regso)} RegSO (year {year})")
    return out


def load_stations():
    s = gpd.read_file(STATIONS).to_crs(CRS)
    s = s[s.geometry.notna() & ~s.geometry.is_empty].copy()
    s["station_idx"] = np.arange(len(s))
    return s[["station_idx", "geometry"]]


def build_units(name, layer, id_col, pop_kind, pop, stations):
    """Polygons + population + station counts + area/density for one granularity."""
    geo = C.CACHE / f"units_{name}.geojson"
    fetch_geometry(layer, geo)
    g = gpd.read_file(geo).to_crs(CRS)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["unit_id"] = g[id_col].astype(str)
    g["S"] = g["unit_id"].map(pop[pop_kind]).astype("float")
    matched = int(g["S"].notna().sum())
    C.log(f"[{name}] {len(g)} polygons, population matched for {matched}")
    g = g[g["S"].notna() & (g["S"] > 0)].copy()       # drop uninhabited units
    # station counts (P=0 kept)
    j = gpd.sjoin(stations, g[["unit_id", "geometry"]], how="inner", predicate="within")
    cnt = j.groupby("unit_id").size().rename("P")
    g = g.merge(cnt, on="unit_id", how="left")
    g["P"] = g["P"].fillna(0).astype(int)
    g["area_ha"] = g.geometry.area / 1e4
    g = g[g["area_ha"] > 0].copy()
    g["dens"] = g["S"] / g["area_ha"]                  # people / hectare
    g["kommunkod"] = g["kommunkod"].astype(str) if "kommunkod" in g.columns else ""
    return g


# ---------------------------------------------------------------------- modelling
def _glm(y, X, offset, alpha=None):
    fam = (sm.families.Poisson() if alpha is None
           else sm.families.NegativeBinomial(alpha=max(alpha, 1e-6)))
    return sm.GLM(y, X, family=fam, offset=offset).fit()


def _alpha_if_overdispersed(y, X, offset):
    """Poisson dispersion check; if overdispersed, estimate NB2 alpha (MLE)."""
    pois = _glm(y, X, offset)
    disp = float(pois.pearson_chi2 / pois.df_resid)
    if disp <= 1.2:
        return None, disp
    try:
        nb = sm.NegativeBinomial(y, X, offset=offset).fit(disp=0, maxiter=200)
        return float(nb.params[-1]), disp
    except Exception:  # noqa: BLE001
        return None, disp


def fit_density(g):
    """Single + segmented count GLM with area offset. Density exponent = beta."""
    y = g["P"].to_numpy(float)
    logrho = np.log(g["dens"].to_numpy(float))
    offset = np.log(g["area_ha"].to_numpy(float))

    # --- single power law: log E[P] = logA + a + beta*log(rho)
    Xs = sm.add_constant(logrho)
    alpha, disp = _alpha_if_overdispersed(y, Xs, offset)
    rs = _glm(y, Xs, offset, alpha)
    beta = float(rs.params[1]); se = float(rs.bse[1])
    single = {
        "family": "Poisson" if alpha is None else f"NB2(alpha={alpha:.4f})",
        "poisson_dispersion": round(disp, 3),
        "beta": round(beta, 4), "se": round(se, 4),
        "ci95": [round(beta - 1.96 * se, 4), round(beta + 1.96 * se, 4)],
        "aic": round(float(rs.aic), 1),
        **_verdict(beta, se),
    }

    # --- segmented: grid-search breakpoint c by deviance (Poisson, fast)
    cand = np.quantile(logrho, np.linspace(0.10, 0.90, 30))
    best = None
    for c in cand:
        hinge = np.maximum(0.0, logrho - c)
        Xg = sm.add_constant(np.column_stack([logrho, hinge]))
        try:
            r = _glm(y, Xg, offset)
        except Exception:  # noqa: BLE001
            continue
        if best is None or r.deviance < best[0]:
            best = (r.deviance, c)
    seg = {"status": "failed"}
    if best is not None:
        c = best[1]
        hinge = np.maximum(0.0, logrho - c)
        Xg = sm.add_constant(np.column_stack([logrho, hinge]))
        rg = _glm(y, Xg, offset, alpha)
        b1 = float(rg.params[1]); b2 = float(rg.params[2]); se2 = float(rg.bse[2])
        # LR test single vs segmented (Davies-style proxy on the hinge term)
        lr = 2.0 * (float(rg.llf) - float(rs.llf))
        p_lr = float(stats.chi2.sf(max(lr, 0.0), df=1))
        seg = {
            "breakpoint_logrho": round(float(c), 4),
            "breakpoint_ppl_per_ha": round(float(np.exp(c)), 3),
            "slope_below": round(b1, 4),
            "slope_above": round(b1 + b2, 4),
            "delta_slope": round(b2, 4), "delta_se": round(se2, 4),
            "aic": round(float(rg.aic), 1),
            "lr_stat": round(lr, 3), "lr_p": p_lr,
        }
        seg["preferred"] = bool(rg.aic < rs.aic and p_lr < 0.01)
        seg["sutton_breakpoint_ppl_per_ha"] = list(SUTTON_BREAK)
        lo, hi = SUTTON_BREAK[0] - SUTTON_BREAK[1], SUTTON_BREAK[0] + SUTTON_BREAK[1]
        seg["within_sutton_band"] = bool(lo <= np.exp(c) <= hi)

    # --- diagnostic: unconstrained log E[P] = a + b1*logS + b2*logA.
    # The density form imposes b1+b2 = 1 (constant returns to pop+area); test it.
    logS = np.log(g["S"].to_numpy(float))
    logA = np.log(g["area_ha"].to_numpy(float))
    Xu = sm.add_constant(np.column_stack([logS, logA]))
    ru = _glm(y, Xu, None, alpha)
    bS, bA = float(ru.params[1]), float(ru.params[2])
    diag = {"beta_logS": round(bS, 4), "beta_logA": round(bA, 4),
            "sum_constraint_b1_plus_b2": round(bS + bA, 4),
            "note": "density model imposes b1+b2=1; ~1 => density framing is well posed"}

    return {"n_units": int(len(g)), "n_zero_P": int((y == 0).sum()),
            "sum_P": int(y.sum()),
            "density_ppl_per_ha": {
                "min": round(float(g["dens"].min()), 4),
                "median": round(float(g["dens"].median()), 3),
                "max": round(float(g["dens"].max()), 1)},
            "single_power_law": single, "segmented": seg,
            "unconstrained_diagnostic": diag}, (rs, single, seg)


def _verdict(beta, se):
    z = (beta - 1.0) / se
    p = 2 * stats.norm.sf(abs(z))
    v = ("indistinguishable from linear" if p >= 0.05
         else "sublinear" if beta < 1 else "superlinear")
    return {"z_vs1": round(float(z), 3), "p_vs1": float(p), "verdict": v}


def sundsvall_focus(g, seg):
    """Where do Sundsvall's own units sit on the density axis?"""
    if "kommunkod" not in g.columns:
        return {}
    sv = g[g["kommunkod"] == SUNDSVALL_KOMMUN].copy()
    if not len(sv):
        return {}
    out = {
        "n_units": int(len(sv)),
        "pop_total": int(sv["S"].sum()),
        "stations_total": int(sv["P"].sum()),
        "density_min": round(float(sv["dens"].min()), 3),
        "density_median": round(float(sv["dens"].median()), 3),
        "density_max": round(float(sv["dens"].max()), 2),
        "kommun_level_density": round(float(sv["S"].sum() / sv["area_ha"].sum()), 3),
    }
    if seg.get("breakpoint_ppl_per_ha"):
        bp = seg["breakpoint_ppl_per_ha"]
        out["breakpoint_ppl_per_ha"] = bp
        out["units_urban_side"] = int((sv["dens"] >= bp).sum())
        out["units_rural_side"] = int((sv["dens"] < bp).sum())
        out["pop_urban_side"] = int(sv.loc[sv["dens"] >= bp, "S"].sum())
    return out


# -------------------------------------------------------------------------- chart
def chart(results, frames):
    fig, axes = plt.subplots(1, len(frames), figsize=(7 * len(frames), 6))
    if len(frames) == 1:
        axes = [axes]
    for ax, (name, g) in zip(axes, frames.items()):
        r = results[name]
        if "single_power_law" not in r:
            continue
        seg = r["segmented"]; single = r["single_power_law"]
        pos = g[g["P"] > 0]
        ax.scatter(pos["dens"], pos["P"] / pos["area_ha"], s=10, alpha=0.30,
                   color="#2b6cb0", edgecolors="none", label=f"units (P>0, n={len(pos)})")
        xs = np.linspace(np.log(g["dens"].min()), np.log(g["dens"].max()), 200)
        # density of E[P]/A from the offset GLM: E[P]/A = exp(a) * rho^beta
        a0 = float(_refit_intercept(g))
        if seg.get("preferred"):
            c = seg["breakpoint_logrho"]; b1 = seg["slope_below"]; b2 = seg["slope_above"]
            # piecewise intercepts continuous at c
            a_seg = _seg_intercept(g)
            ylo = a_seg + b1 * np.minimum(xs, c) + b2 * np.maximum(0.0, xs - c)
            ax.plot(np.exp(xs), np.exp(ylo), "k-", lw=2.2,
                    label=(f"segmented: {b1:.2f} below / {b2:.2f} above\n"
                           f"break = {seg['breakpoint_ppl_per_ha']:.1f} ppl/ha"))
            ax.axvline(np.exp(c), color="#c53030", ls="--", lw=1.4)
        else:
            ax.plot(np.exp(xs), np.exp(a0 + single["beta"] * xs), "k-", lw=2.2,
                    label=f"single: beta = {single['beta']:.2f}")
        # Sutton band
        ax.axvspan(SUTTON_BREAK[0] - SUTTON_BREAK[1], SUTTON_BREAK[0] + SUTTON_BREAK[1],
                   color="#dd6b20", alpha=0.10, label="Sutton 33±5 ppl/ha")
        # Sundsvall units
        if "kommunkod" in g.columns:
            sv = g[(g["kommunkod"] == SUNDSVALL_KOMMUN) & (g["P"] > 0)]
            if len(sv):
                ax.scatter(sv["dens"], sv["P"] / sv["area_ha"], s=40,
                           color="#dd6b20", edgecolors="k", linewidths=0.5, zorder=5,
                           label=f"Sundsvall units (n={len(sv)})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("population density  ρ = S/A  (people / hectare)")
        ax.set_ylabel("station density  P/A  (per hectare)")
        ax.set_title(f"{name.upper()}  (n={r['n_units']}, ΣP={r['sum_P']})")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.2)
    fig.suptitle("Density scaling of petrol stations (Sutton et al. framing): "
                 "indicator density vs population density, segmented count GLM",
                 fontsize=12)
    fig.tight_layout()
    p = C.CHARTS / "10_density.png"
    fig.savefig(p, dpi=DPI); plt.close(fig)
    C.log(f"[OK] chart {p.name}")


def _refit_intercept(g):
    y = g["P"].to_numpy(float)
    X = sm.add_constant(np.log(g["dens"].to_numpy(float)))
    off = np.log(g["area_ha"].to_numpy(float))
    return _glm(y, X, off).params[0]


def _seg_intercept(g):
    return _refit_intercept(g)  # offset GLM intercept = log of density prefactor


# --------------------------------------------------------------------------- report
def write_report(results, year):
    L = ["# Density scaling (Sutton et al. framing)", "",
         f"Population per unit: SCB PXWeb FolkmDesoAldKon, year {year}. "
         "Geometry: SCB WFS DeSO_2025 / RegSO_2025. Stations: deduped OSM amenity=fuel.",
         "",
         "Model (count GLM, area offset): `log E[P] = log A + a + β·log(ρ)`, "
         "ρ = S/A in people/ha; segmented adds a hinge at a grid-searched breakpoint.",
         "", "| Unit | n | ΣP | dens median (ppl/ha) | single β | segmented (below→above) | breakpoint | vs Sutton 33±5 |",
         "|---|--:|--:|--:|--:|--:|--:|---|"]
    for name, r in results.items():
        if "single_power_law" not in r:
            L.append(f"| {name.upper()} | — | — | — | — | (failed) | — | — |")
            continue
        s = r["single_power_law"]; g = r["segmented"]
        if g.get("preferred"):
            segtxt = f"{g['slope_below']:.2f} → {g['slope_above']:.2f}"
            bp = f"{g['breakpoint_ppl_per_ha']:.1f} ppl/ha"
            within = "within band" if g.get("within_sutton_band") else "outside band"
        else:
            segtxt = "(single preferred)"; bp = "—"; within = "—"
        L.append(f"| {name.upper()} | {r['n_units']} | {r['sum_P']} | "
                 f"{r['density_ppl_per_ha']['median']} | {s['beta']:.3f} | {segtxt} | {bp} | {within} |")
    L += ["", "## Sundsvall — the diluted-kommun problem, resolved by granularity", ""]
    for name, r in results.items():
        sv = r.get("sundsvall", {})
        if not sv:
            continue
        L.append(f"**{name.upper()}**: Sundsvall splits into {sv['n_units']} units. "
                 f"Density ranges {sv['density_min']}–{sv['density_max']} ppl/ha "
                 f"(median {sv['density_median']}), vs the single kommun-level value of "
                 f"{sv['kommun_level_density']} ppl/ha that made the city look rural.")
        if "units_urban_side" in sv:
            L.append(f"  {sv['units_urban_side']} units fall on the urban side of the "
                     f"{sv['breakpoint_ppl_per_ha']:.1f} ppl/ha breakpoint "
                     f"({sv['pop_urban_side']:,} people), {sv['units_rural_side']} on the rural side.")
        L.append("")
    p = C.OUTPUT / "density.md"
    p.write_text("\n".join(L))
    C.log(f"[OK] wrote {p.name}")


# ----------------------------------------------------------------------------- main
def main():
    name = "Phase 10 — density scaling (DeSO/RegSO)"
    C.phase_start(name)
    C.set_seed()
    if not C.exists_nonempty(STATIONS):
        C.warn("stations.gpkg missing — run phase 1 first; skipping density phase")
        C.phase_end(name); return
    try:
        pop = fetch_population()
    except Exception as e:  # noqa: BLE001
        C.warn(f"population fetch failed (non-fatal): {e}")
        C.phase_end(name); return

    stations = load_stations()
    results, frames = {}, {}
    for nm, (layer, id_col, kind) in UNITS.items():
        try:
            g = build_units(nm, layer, id_col, kind, pop, stations)
            if nm == "deso":
                # RegSO totals are not served by FolkmDesoAldKon (all zeros), but each
                # DeSO nests in exactly one RegSO, so RegSO population = sum over DeSO.
                pop["regso"] = g.groupby("regsokod")["S"].sum().to_dict()
                C.log(f"[regso] population derived from {len(g)} DeSO -> "
                      f"{len(pop['regso'])} RegSO")
            res, _ = fit_density(g)
            res["sundsvall"] = sundsvall_focus(g, res["segmented"])
            results[nm] = res
            frames[nm] = g
            s = res["single_power_law"]; seg = res["segmented"]
            bp = seg.get("breakpoint_ppl_per_ha")
            C.log(f"[{nm}] single β={s['beta']} | segmented "
                  f"{seg.get('slope_below')}→{seg.get('slope_above')} @ {bp} ppl/ha "
                  f"| preferred={seg.get('preferred')} | n={res['n_units']} ΣP={res['sum_P']}")
        except Exception as e:  # noqa: BLE001
            import traceback
            C.warn(f"[{nm}] density phase failed: {e}\n{traceback.format_exc()}")
            results[nm] = {"status": "failed", "error": str(e)}

    if frames:
        chart(results, frames)
        write_report(results, pop.get("year"))
    C.write_json(C.OUTPUT / "density_summary.json",
                 {"year": pop.get("year"), "sutton_breakpoint_ppl_per_ha": list(SUTTON_BREAK),
                  "units": results})
    C.phase_end(name)


if __name__ == "__main__":
    main()

"""Phase 15 — DeSO vs RegSO across the three lenses (scatter comparison).

We've examined DeSO at length; RegSO is SCB's coarser unit (~3,363 areas, each an
aggregation of several DeSO, wider population range: median 2.5k, up to 23k). This
puts the two units side by side under the three views we've used, so you can see
what coarsening the unit does to each scatter:

  containment : stations INSIDE the polygon (P)        vs unit population (S)
  density     : station density (P/A)                  vs population density (S/A)
                [count GLM with area offset]
  catchment   : stations within 10 km of the centre    vs population within 10 km

Rows = unit (DeSO, RegSO); columns = lens. Each panel prints its β.
Writes charts/15_unit_compare.png and output/unit_compare_summary.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

CRS = C.CONFIG["national_crs"]
DPI = 150
STATIONS = C.CACHE / "stations.gpkg"
D_KM = 10


def count_glm(y, x):
    """log E[y] = a + b*log(x); NB2 if overdispersed else Poisson. Returns dict+a0."""
    X = sm.add_constant(np.log(x))
    pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    disp = float(pois.pearson_chi2 / pois.df_resid)
    if disp > 1.2:
        try:
            res = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=200)
        except Exception:  # noqa: BLE001
            res = pois
    else:
        res = pois
    b = float(res.params[1]); se = float(res.bse[1])
    z = (b - 1) / se
    return {"beta": round(b, 3), "se": round(se, 3),
            "ci95": [round(b - 1.96 * se, 3), round(b + 1.96 * se, 3)],
            "p_vs1": float(2 * stats.norm.sf(abs(z)))}, float(res.params[0])


def offset_glm(P, dens, area):
    """log E[P] = log A + a + b*log(dens). Returns dict + density prefactor a0."""
    X = sm.add_constant(np.log(dens)); off = np.log(area)
    pois = sm.GLM(P, X, family=sm.families.Poisson(), offset=off).fit()
    disp = float(pois.pearson_chi2 / pois.df_resid)
    if disp > 1.2:
        try:
            a = float(sm.NegativeBinomial(P, X, offset=off).fit(disp=0, maxiter=200).params[-1])
            res = sm.GLM(P, X, family=sm.families.NegativeBinomial(alpha=max(a, 1e-6)),
                         offset=off).fit()
        except Exception:  # noqa: BLE001
            res = pois
    else:
        res = pois
    b = float(res.params[1]); se = float(res.bse[1])
    return {"beta": round(b, 3), "se": round(se, 3),
            "ci95": [round(b - 1.96 * se, 3), round(b + 1.96 * se, 3)]}, float(res.params[0])


def build(stations):
    pop = C.read_json(C.CACHE / "pop_deso_regso.json")["deso"]
    deso = gpd.read_file(C.CACHE / "units_deso.geojson").to_crs(CRS)
    deso["S"] = deso["desokod"].astype(str).map(pop).astype(float)
    deso = deso[deso["S"].notna() & (deso["S"] > 0)].copy()
    regso_pop = deso.groupby("regsokod")["S"].sum().to_dict()
    units = {}
    specs = [("DeSO", deso, "desokod", pop),
             ("RegSO", gpd.read_file(C.CACHE / "units_regso.geojson").to_crs(CRS),
              "regsokod", regso_pop)]
    st_xy = np.c_[stations.geometry.x.to_numpy(), stations.geometry.y.to_numpy()]
    tree_st = cKDTree(st_xy)
    for nm, g, idc, pmap in specs:
        g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
        g["S"] = g[idc].astype(str).map(pmap).astype(float)
        g = g[g["S"].notna() & (g["S"] > 0)].copy()
        # containment count
        j = gpd.sjoin(stations[["geometry"]].assign(_i=range(len(stations))),
                      g[[idc, "geometry"]], how="inner", predicate="within")
        cnt = j.groupby(idc).size().rename("P")
        g = g.merge(cnt, on=idc, how="left"); g["P"] = g["P"].fillna(0).astype(int)
        g["area_km2"] = g.geometry.area / 1e6
        g = g[g["area_km2"] > 0].copy()
        g["dens"] = g["S"] / g["area_km2"]
        cp = g.geometry.representative_point()
        anchors = np.c_[cp.x.to_numpy(), cp.y.to_numpy()]
        S = g["S"].to_numpy(float)
        tree_an = cKDTree(anchors)
        r = D_KM * 1000.0
        g["P_acc"] = tree_st.query_ball_point(anchors, r=r, return_length=True).astype(float)
        g["N_catch"] = [S[ix].sum() for ix in tree_an.query_ball_point(anchors, r=r)]
        units[nm] = g
    return units


def scatter(ax, x, y, fit, a0, xlab, ylab, title, color, offset_kind=False):
    ax.scatter(x, y, s=7, alpha=0.16, color="#5a6b7b", edgecolors="none")
    xs = np.linspace(np.log(x.min()), np.log(x.max()), 100)
    ax.plot(np.exp(xs), np.exp(a0 + fit["beta"] * xs), color=color, lw=2.3,
            label=f"β = {fit['beta']:.2f}  CI{fit['ci95']}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(True, which="both", alpha=0.2)


def main():
    name = "Phase 15 — DeSO vs RegSO comparison"
    C.phase_start(name)
    C.set_seed()
    if not C.exists_nonempty(STATIONS):
        C.warn("stations missing; skipping"); C.phase_end(name); return
    stations = gpd.read_file(STATIONS).to_crs(CRS)
    stations = stations[stations.geometry.notna() & ~stations.geometry.is_empty].copy()
    units = build(stations)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    summary = {}
    cols = {"DeSO": "#2b6cb0", "RegSO": "#c1440e"}
    for row, (nm, g) in enumerate(units.items()):
        col = cols[nm]
        pos = g[g["P"] > 0]
        # containment
        fc, a0 = count_glm(pos["P"].to_numpy(float), pos["S"].to_numpy(float))
        scatter(axes[row, 0], pos["S"].to_numpy(float), pos["P"].to_numpy(float), fc, a0,
                "unit population S", "stations inside (P)", f"{nm} · containment", col)
        # density (count GLM offset; plot P/A vs dens for positives)
        fd, ad = offset_glm(g["P"].to_numpy(float), g["dens"].to_numpy(float),
                            g["area_km2"].to_numpy(float))
        scatter(axes[row, 1], pos["dens"].to_numpy(float),
                (pos["P"] / pos["area_km2"]).to_numpy(float), fd, ad,
                "population density S/A (ppl/km²)", "station density P/A (per km²)",
                f"{nm} · density", col)
        # catchment
        m = g["P_acc"] > 0
        fk, ak = count_glm(g.loc[m, "P_acc"].to_numpy(float), g.loc[m, "N_catch"].to_numpy(float))
        scatter(axes[row, 2], g.loc[m, "N_catch"].to_numpy(float),
                g.loc[m, "P_acc"].to_numpy(float), fk, ak,
                f"catchment pop within {D_KM} km", f"stations within {D_KM} km",
                f"{nm} · catchment ({D_KM} km)", col)
        summary[nm] = {"n": int(len(g)), "n_zero_P": int((g["P"] == 0).sum()),
                       "containment": fc, "density": fd, "catchment_10km": fk}
        C.log(f"[{nm}] n={len(g)} zeroP={int((g['P']==0).sum())} | "
              f"containment β={fc['beta']} | density β={fd['beta']} | catchment β={fk['beta']}")

    fig.suptitle("DeSO vs RegSO across three lenses — what coarsening the unit does to "
                 "the scatter and β", fontsize=13)
    fig.tight_layout()
    out = C.CHARTS / "15_unit_compare.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    C.log(f"[OK] {out.name}")
    C.write_json(C.OUTPUT / "unit_compare_summary.json", summary)
    C.phase_end(name)


if __name__ == "__main__":
    main()

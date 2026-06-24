"""Phase 6 — deviation analysis.

The β exponent is the all-Sweden summary; the residuals are where the structure is.
This phase (1) classifies each station by brand into automat / full-service / gas-alt /
unknown (station TYPE is used ONLY here, for deviation inspection — never to split β),
(2) refits the kommun NB GLM and computes signed log-residuals, (3) tests whether
over-stationed units are automat-heavy, and renders:
  output/charts/05_residual_automat.png  — residual vs log S, coloured by automat-share
  output/charts/06_dorling_kommun.png    — Dorling cartogram (area∝pop, colour=residual)
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

CRS = C.CONFIG["national_crs"]
DPI = 160

# Brand-based station typology (approximate, documented). Unmanned automat chains vs
# manned full-service majors vs vehicle-gas/biogas ("doing other things").
AUTOMAT = ["ingo", "tanka", "qstar", "q star", "din-x", "dinx", "bilisten", "pump",
           "saifa", "såifa", "alltank", "paroy", "borjes", "börjes", "dalvik",
           "ps energi", "trb", "oljeshejkerna", "runes", "smaland", "småländ",
           "smaaländ", "independent"]
FULL = ["okq8", "preem", "circle k", "circlek", "st1", "st 1", "gulf", "statoil",
        "shell", "hydro", "jet", "uno-x", "unox"]
GAS = ["fordonsgas", "gasum", "e.on", "eon", "biogas", "energifabriken", "gasum",
       "energigas", "fordongas", "swedegas"]


def classify(label):
    s = re.sub(r"\s+", " ", str(label).strip().lower())
    if s in ("", "none", "(none)", "nan"):
        return "unknown"
    for kw in GAS:
        if kw in s:
            return "gas_alt"
    for kw in AUTOMAT:
        if kw in s:
            return "automat"
    for kw in FULL:
        if kw in s:
            return "full_service"
    return "unknown"


def load_typed_stations():
    s = gpd.read_file(C.CACHE / "stations.gpkg").to_crs(CRS)
    s = s[s.geometry.notna() & ~s.geometry.is_empty].copy()
    lab = s["brand"].fillna(s["operator"]).fillna(s["name"])
    s["stype"] = lab.map(classify)
    return s


def kommun_with_types(stations):
    kom = gpd.read_file(C.CACHE / "units_kommun.gpkg")
    kom = kom[kom["S"] > 0].copy()
    j = gpd.sjoin(stations[["stype", "geometry"]], kom[["unit_id", "geometry"]],
                  how="inner", predicate="within")
    piv = (j.groupby(["unit_id", "stype"]).size().unstack(fill_value=0)
           .reindex(columns=["automat", "full_service", "gas_alt", "unknown"],
                    fill_value=0))
    kom = kom.merge(piv, on="unit_id", how="left").fillna(
        {c: 0 for c in ["automat", "full_service", "gas_alt", "unknown"]})
    liquid = kom["automat"] + kom["full_service"]
    kom["automat_share"] = np.where(liquid > 0, kom["automat"] / liquid, np.nan)
    kom["gas_share"] = np.where(kom["P"] > 0, kom["gas_alt"] / kom["P"], np.nan)
    return kom


def fit_resid(kom):
    y = kom["P"].to_numpy(float)
    X = sm.add_constant(np.log(kom["S"].to_numpy(float)))
    res = sm.NegativeBinomial(y, X).fit(disp=0)
    glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=res.params[-1])).fit()
    kom = kom.copy()
    kom["expected"] = glm.predict(X)
    kom["logratio"] = np.log(np.maximum(kom["P"], 0.5) / kom["expected"])
    kom["beta"] = float(glm.params[1])
    return kom


def chart_residual_automat(kom, summary):
    d = kom[kom["automat_share"].notna()].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(d["S"], d["logratio"], c=d["automat_share"], cmap="RdYlBu_r",
                    s=22 + 4 * np.sqrt(d["P"]), vmin=0, vmax=1, alpha=0.85,
                    edgecolors="grey", linewidths=0.3)
    ax.axhline(0, color="black", lw=1)
    ax.set_xscale("log")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("automat (unmanned) share of liquid-fuel stations")
    # label most over- and under-stationed
    for _, r in pd.concat([d.nlargest(8, "logratio"), d.nsmallest(5, "logratio")]).iterrows():
        ax.annotate(r["name"], (r["S"], r["logratio"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    rho = d[["logratio", "automat_share"]].corr().iloc[0, 1]
    ax.set_xlabel("Population S (kommun)")
    ax.set_ylabel("log(observed P / expected P)   — over-stationed ↑ / under-stationed ↓")
    ax.set_title(f"Where kommuner deviate from the scaling law — and is it an automat effect?\n"
                 f"corr(residual, automat-share) = {rho:+.2f}  (point size ∝ √stations)")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout(); fig.savefig(C.CHARTS / "05_residual_automat.png", dpi=DPI)
    plt.close(fig); C.log("[OK] chart 05_residual_automat.png")
    summary["corr_residual_automatshare"] = round(float(rho), 3)


def dorling(kom, summary, iters=200, coverage=0.32):
    """Dorling cartogram: circles area∝population, repelled from overlap."""
    g = kom.copy()
    pts = g.geometry.representative_point()
    x = pts.x.to_numpy(float); y = pts.y.to_numpy(float)
    pop = g["S"].to_numpy(float)
    # radius scaling so total circle area ~ coverage * bbox area
    bbox = g.total_bounds
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    k = np.sqrt(coverage * bbox_area / (np.pi * pop.sum()))
    r = k * np.sqrt(pop)
    n = len(g)
    for _ in range(iters):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy) + 1e-9
        mind = r[:, None] + r[None, :]
        overlap = np.maximum(0.0, mind - dist)
        np.fill_diagonal(overlap, 0.0)
        # push proportional to overlap, half each
        fx = (overlap * dx / dist).sum(axis=1) * 0.5
        fy = (overlap * dy / dist).sum(axis=1) * 0.5
        x = x + fx; y = y + fy
    val = g["logratio"].to_numpy(float)
    vlim = np.nanpercentile(np.abs(val), 95)
    fig, ax = plt.subplots(figsize=(7, 10))
    # faint true outline for orientation
    try:
        bg = gpd.read_file(C.CACHE / "lau_2024_3035.gpkg", where="CNTR_CODE='SE'").to_crs(CRS)
        bg.dissolve().boundary.plot(ax=ax, color="#cccccc", lw=0.6)
    except Exception:  # noqa: BLE001
        pass
    order = np.argsort(-r)  # big circles first
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(-vlim, vlim)
    for i in order:
        edge = "black" if g["is_border"].iloc[i] else "#555555"
        lw = 1.6 if g["is_border"].iloc[i] else 0.4
        ax.add_patch(plt.Circle((x[i], y[i]), r[i], facecolor=cmap(norm(val[i])),
                                edgecolor=edge, lw=lw, alpha=0.92))
    ax.set_xlim(x.min() - r.max(), x.max() + r.max())
    ax.set_ylim(y.min() - r.max(), y.max() + r.max())
    ax.set_aspect("equal"); ax.set_axis_off()
    sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm_.set_array([])
    cb = fig.colorbar(sm_, ax=ax, shrink=0.4)
    cb.set_label("log(obs/exp) stations  (red = over-stationed)")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                              markeredgecolor="black", markeredgewidth=1.6, markersize=10,
                              label="within 10 km of NO/FI border")],
              loc="lower left", fontsize=8)
    ax.set_title("Dorling cartogram of Sweden's kommuner\n"
                 "(circle area ∝ population, colour = deviation from petrol-scaling law)")
    fig.tight_layout(); fig.savefig(C.CHARTS / "06_dorling_kommun.png", dpi=DPI)
    plt.close(fig); C.log("[OK] chart 06_dorling_kommun.png")


def main():
    name = "Phase 6 — deviation analysis"
    C.phase_start(name)
    C.set_seed()
    stations = load_typed_stations()
    tcounts = stations["stype"].value_counts().to_dict()
    C.log(f"station types: {tcounts}")
    kom = kommun_with_types(stations)
    kom = fit_resid(kom)

    summary = {
        "station_type_counts": {k: int(v) for k, v in tcounts.items()},
        "kommun_beta": round(float(kom["beta"].iloc[0]), 4),
        "top_over_stationed": kom.nlargest(10, "logratio")[
            ["name", "S", "P", "expected", "automat_share", "gas_alt", "is_border"]
        ].assign(expected=lambda d: d["expected"].round(1),
                 automat_share=lambda d: d["automat_share"].round(2)).to_dict("records"),
        "top_under_stationed": kom.nsmallest(8, "logratio")[
            ["name", "S", "P", "expected", "automat_share", "is_border"]
        ].assign(expected=lambda d: d["expected"].round(1),
                 automat_share=lambda d: d["automat_share"].round(2)).to_dict("records"),
    }
    chart_residual_automat(kom, summary)
    dorling(kom, summary)

    # automat-share by size tercile, and border vs not
    kom["size_class"] = pd.qcut(kom["S"], 3, labels=["small", "medium", "large"])
    summary["automat_share_by_size"] = {
        str(k): round(float(v), 3)
        for k, v in kom.groupby("size_class", observed=True)["automat_share"].mean().items()}
    summary["automat_share_border_vs_not"] = {
        "border": round(float(kom[kom.is_border]["automat_share"].mean()), 3),
        "non_border": round(float(kom[~kom.is_border]["automat_share"].mean()), 3)}
    C.write_json(C.OUTPUT / "deviation_summary.json", summary)
    kom.to_file(C.CACHE / "kommun_typed.gpkg", driver="GPKG")
    C.log(f"corr(resid, automat-share) = {summary['corr_residual_automatshare']}")
    C.log(f"automat-share by size: {summary['automat_share_by_size']}")
    C.phase_end(name)


if __name__ == "__main__":
    main()

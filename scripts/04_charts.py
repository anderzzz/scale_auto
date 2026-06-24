"""Phase 4 — charts -> output/charts/*.png (150 dpi)."""
import sys
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

DPI = 160
LAU = C.CACHE / "lau_2024_3035.gpkg"
CRS = C.CONFIG["national_crs"]


def headline_scatter(results):
    f = C.CACHE / "resid_tatort.gpkg"
    if not C.exists_nonempty(f):
        C.warn("headline: resid_tatort.gpkg missing; skip"); return
    g = gpd.read_file(f)
    a = results["definitions"]["tatort"]["model_A_primary"]
    beta, b0 = a["beta"], a["intercept"]
    pos = g[g["P"] > 0]
    zero = g[g["P"] == 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pos["S"], pos["P"], s=10, alpha=0.35, color="#2b6cb0",
               label=f"tätort with ≥1 station (n={len(pos)})", edgecolors="none")
    # rug for zeros along the bottom
    if len(zero):
        ax.plot(zero["S"], np.full(len(zero), 0.55), "|", color="#c05621",
                alpha=0.4, markersize=6,
                label=f"zero-station tätort (n={len(zero)}, off log axis)")
    xs = np.linspace(np.log(g["S"].min()), np.log(g["S"].max()), 100)
    yhat = np.exp(b0 + beta * xs)
    ax.plot(np.exp(xs), yhat, "-", color="black", lw=2,
            label=f"GLM fit  β={beta:.3f}  CI[{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}]")
    se = a["se"]
    lo = np.exp(b0 + (beta - 1.96 * se) * xs)
    hi = np.exp(b0 + (beta + 1.96 * se) * xs)
    ax.fill_between(np.exp(xs), lo, hi, color="black", alpha=0.12)
    # reference linear slope
    yref = np.exp(np.log(yhat[0]) + 1.0 * (xs - xs[0]))
    ax.plot(np.exp(xs), yref, "--", color="grey", lw=1, label="slope = 1 (linear)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Population S (tätort, SCB 2023)")
    ax.set_ylabel("Petrol stations P (OSM amenity=fuel)")
    ax.set_title(f"Petrol-station scaling in Swedish tätorter — {a['verdict']} (β={beta:.3f})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout(); fig.savefig(C.CHARTS / "01_headline_tatort.png", dpi=DPI)
    plt.close(fig); C.log("[OK] chart 01_headline_tatort.png")


def forest(results):
    order = ["tatort", "tatort_smaort", "fua", "kommun"]
    labels = {"tatort": "Tätort (settlement)", "tatort_smaort": "Tätort + småort",
              "fua": "FUA (functional)", "kommun": "Kommun (admin)"}
    rows = []
    for d in order:
        v = results["definitions"].get(d, {})
        a = v.get("model_A_primary")
        if not a:
            continue
        bs = v.get("bootstrap_modelA", {}).get("ci95")
        rows.append((labels[d], a["beta"], a["ci95"], bs, a["verdict"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, beta, ci, bsci, verd) in zip(ys, rows):
        ax.plot([ci[0], ci[1]], [y, y], color="#2b6cb0", lw=3, solid_capstyle="round")
        if bsci:
            ax.plot([bsci[0], bsci[1]], [y + 0.16, y + 0.16], color="#dd6b20", lw=1.5,
                    alpha=0.8)
        ax.plot(beta, y, "o", color="black", zorder=5)
        ax.text(ci[1] + 0.01, y, f"{beta:.3f}  ({verd})", va="center", fontsize=8)
    ax.axvline(1.0, color="red", ls="--", lw=1, label="β = 1 (linear)")
    ax.axvspan(0.75, 0.90, color="green", alpha=0.07,
               label="infrastructure prior ≈0.75–0.90")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Scaling exponent β  (blue = model 95% CI, orange = bootstrap CI)")
    ax.set_title("β across unit definitions — the MAUP / leakage payoff")
    ax.set_xlim(0.4, 1.08); ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(C.CHARTS / "02_forest_beta.png", dpi=DPI)
    plt.close(fig); C.log("[OK] chart 02_forest_beta.png")


def residuals_vs_s():
    f = C.CACHE / "resid_tatort.gpkg"
    if not C.exists_nonempty(f):
        return
    g = gpd.read_file(f)
    col = "resid_deviance" if "resid_deviance" in g.columns else "resid_pearson"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(g["S"], g[col], s=10, alpha=0.35, color="#2b6cb0", edgecolors="none")
    ax.axhline(0, color="black", lw=1)
    # lowess trend
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lo = lowess(g[col], np.log(g["S"]), frac=0.4)
        ax.plot(np.exp(lo[:, 0]), lo[:, 1], color="red", lw=2, label="LOWESS trend")
        ax.legend(fontsize=8)
    except Exception:  # noqa: BLE001
        pass
    ax.set_xscale("log")
    ax.set_xlabel("Population S (tätort)"); ax.set_ylabel(f"{col} (GLM)")
    ax.set_title("Tätort residuals vs size — systematic trend signals regime change")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout(); fig.savefig(C.CHARTS / "03_residuals_tatort.png", dpi=DPI)
    plt.close(fig); C.log("[OK] chart 03_residuals_tatort.png")


def residual_map():
    f = C.CACHE / "resid_tatort.gpkg"
    if not C.exists_nonempty(f):
        return
    g = gpd.read_file(f).to_crs(CRS)
    col = "resid_deviance" if "resid_deviance" in g.columns else "resid_pearson"
    pts = g.copy(); pts["geometry"] = g.geometry.representative_point()
    fig, ax = plt.subplots(figsize=(6.5, 9))
    try:
        bg = gpd.read_file(LAU, where="CNTR_CODE='SE'").to_crs(CRS)
        bg.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", lw=0.2)
    except Exception:  # noqa: BLE001
        pass
    v = np.nanpercentile(np.abs(pts[col]), 97)
    sc = ax.scatter(pts.geometry.x, pts.geometry.y, c=pts[col], cmap="RdBu_r",
                    vmin=-v, vmax=v, s=14, alpha=0.8, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.5)
    cb.set_label(f"{col}  (red = MORE stations than expected)")
    ax.set_title("Where Swedish tätorter are over-/under-stationed\n(GLM residual vs population expectation)")
    ax.set_axis_off()
    fig.tight_layout(); fig.savefig(C.CHARTS / "04_residual_map.png", dpi=DPI)
    plt.close(fig); C.log("[OK] chart 04_residual_map.png")


def main():
    name = "Phase 4 — charts"
    C.phase_start(name)
    results = C.read_json(C.OUTPUT / "results.json")
    if not results:
        C.warn("no results.json; cannot draw charts"); C.phase_end(name); return
    for fn in (lambda: headline_scatter(results), lambda: forest(results),
               residuals_vs_s, residual_map):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            C.warn(f"chart failed: {e}\n{traceback.format_exc()}")
    C.phase_end(name)


if __name__ == "__main__":
    C.set_seed()
    main()

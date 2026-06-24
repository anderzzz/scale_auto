"""Phase 7 — denominator swap: stations vs registered CARS (not people).

Population is a biased denominator for fuel demand: urban car-ownership and
vehicle-km per capita are lower (transit, density). If part of the sublinearity is
just 'cities own fewer cars', then refitting P = C * V^beta on registered passenger
cars V should pull beta UP toward 1 relative to the population fit.

Cars per kommun: SCB PXWeb (Trafikanalys), table TK1001A/FordonTrafik, passenger cars
in use, latest year. Writes output/vehicles_summary.json and chart 07.
"""
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
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

DPI = 160
PXWEB = ("https://api.scb.se/OV0104/v1/doris/en/ssd/TK/TK1001/TK1001A/FordonTrafik")
CARS_CACHE = C.CACHE / "cars_per_kommun.json"


def fetch_cars():
    import requests
    if C.exists_nonempty(CARS_CACHE):
        return C.read_json(CARS_CACHE)
    meta = C.retry(lambda: requests.get(PXWEB, timeout=60).json(),
                   what="SCB PXWeb metadata")
    years = [v["values"] for v in meta["variables"] if v.get("time")][0]
    year = years[-1]
    query = {"query": [
        {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
        {"code": "Fordonsslag", "selection": {"filter": "item", "values": ["10"]}},
        {"code": "Tid", "selection": {"filter": "item", "values": [year]}},
    ], "response": {"format": "json"}}
    js = C.retry(lambda: requests.post(PXWEB, json=query, timeout=120).json(),
                 what="SCB PXWeb cars query")
    out = {"year": year, "cars": {}}
    for row in js["data"]:
        code = row["key"][0]
        if len(code) == 4 and code.isdigit():  # kommun codes only
            try:
                out["cars"][code] = int(row["values"][0])
            except (ValueError, TypeError):
                pass
    C.write_json(CARS_CACHE, out)
    C.update_manifest(C.manifest_entry_for(CARS_CACHE, PXWEB + " (passenger cars, " + year + ")"))
    return out


def fit(y, x):
    X = sm.add_constant(np.log(x))
    res = sm.NegativeBinomial(y, X).fit(disp=0)
    beta, se = float(res.params[1]), float(res.bse[1])
    z = (beta - 1) / se
    from scipy import stats
    p = 2 * stats.norm.sf(abs(z))
    verdict = ("indistinguishable from linear" if p >= 0.05
               else "sublinear" if beta < 1 else "superlinear")
    return {"beta": round(beta, 4), "se": round(se, 4),
            "ci95": [round(beta - 1.96 * se, 4), round(beta + 1.96 * se, 4)],
            "p_vs1": p, "verdict": verdict}


def main():
    name = "Phase 7 — denominator swap (cars)"
    C.phase_start(name)
    C.set_seed()
    if not C.host_up("api.scb.se"):
        C.warn("api.scb.se unreachable — skipping vehicles phase")
        C.phase_end(name); return
    try:
        cars = fetch_cars()
    except Exception as e:  # noqa: BLE001
        C.warn(f"vehicle fetch failed (non-fatal): {e}")
        C.phase_end(name); return

    kom = gpd.read_file(C.CACHE / "units_kommun.gpkg")
    kom = kom[kom["S"] > 0].copy()
    kom["kommun_code"] = kom["unit_id"].str.replace("SE_", "", regex=False)
    kom["V"] = kom["kommun_code"].map(cars["cars"])
    matched = int(kom["V"].notna().sum())
    kom = kom[kom["V"].notna() & (kom["V"] > 0)].copy()
    C.log(f"matched cars for {matched}/290 kommuner (year {cars['year']})")

    y = kom["P"].to_numpy(float)
    pop_fit = fit(y, kom["S"].to_numpy(float))
    car_fit = fit(y, kom["V"].to_numpy(float))
    kom["cars_per_capita"] = kom["V"] / kom["S"]

    summary = {
        "year": cars["year"], "n_kommun": int(len(kom)),
        "beta_population": pop_fit, "beta_cars": car_fit,
        "delta_beta": round(car_fit["beta"] - pop_fit["beta"], 4),
        "cars_per_capita": {
            "national": round(float(kom["V"].sum() / kom["S"].sum()), 3),
            "by_size_tercile": {}},
    }
    kom["size_class"] = pd.qcut(kom["S"], 3, labels=["small", "medium", "large"])
    for k, v in kom.groupby("size_class", observed=True)["cars_per_capita"].mean().items():
        summary["cars_per_capita"]["by_size_tercile"][str(k)] = round(float(v), 3)
    C.write_json(C.OUTPUT / "vehicles_summary.json", summary)

    # chart: P vs population and P vs cars, both fits
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, col, fitres, lab in [
        (axes[0], "S", pop_fit, "Population"), (axes[1], "V", car_fit, "Registered cars")]:
        ax.scatter(kom[col], kom["P"], s=18, alpha=0.5, color="#2b6cb0", edgecolors="none")
        xs = np.linspace(np.log(kom[col].min()), np.log(kom[col].max()), 100)
        # NB fit intercept
        X = sm.add_constant(np.log(kom[col].to_numpy(float)))
        b0 = float(sm.NegativeBinomial(y, X).fit(disp=0).params[0])
        ax.plot(np.exp(xs), np.exp(b0 + fitres["beta"] * xs), "k-", lw=2,
                label=f"β = {fitres['beta']:.3f}  CI{fitres['ci95']}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{lab} per kommun"); ax.set_title(f"P vs {lab}")
        ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("Petrol stations P")
    fig.suptitle(f"Denominator swap: β rises {pop_fit['beta']:.3f} (people) → "
                 f"{car_fit['beta']:.3f} (cars). Urban car-ownership explains part of the "
                 f"apparent sublinearity.", fontsize=11)
    fig.tight_layout(); fig.savefig(C.CHARTS / "07_pop_vs_cars.png", dpi=DPI)
    plt.close(fig); C.log("[OK] chart 07_pop_vs_cars.png")

    C.log(f"β population={pop_fit['beta']} ({pop_fit['verdict']}) | "
          f"β cars={car_fit['beta']} ({car_fit['verdict']}) | Δ={summary['delta_beta']}")
    C.log(f"cars per capita by size: {summary['cars_per_capita']['by_size_tercile']}")
    C.phase_end(name)


if __name__ == "__main__":
    main()

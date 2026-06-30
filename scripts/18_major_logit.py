"""Phase 18 — quantify: does population density predict a station being a major brand?

Station-level logistic regression (the right tool for continuous predictor → binary
outcome):

    P(is_major) = logit⁻¹( a + b · log10(ρ) )          ρ = DeSO population density (ppl/ha)

is_major = tier == "major" (Circle K / OKQ8 / Preem / St1) among car-serving stations.
SEs are CLUSTER-ROBUST by DeSO, because many stations share one density (non-iid).

Reports: slope b, odds ratio per 10× density (e^b), p-value, McFadden pseudo-R², the
crossover density where P(major)=0.5, plus a non-parametric cross-check (Mann–Whitney
U on the density of major vs non-major stations) and a predicted-probability curve
against binned empirical fractions (Wilson CIs).

Writes charts/18_major_logit.png and output/major_logit.json.
"""
import sys
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
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

CRS = C.CONFIG["national_crs"]
DPI = 150
CLEAN = C.CACHE / "stations_clean.gpkg"
DESO = C.CACHE / "units_deso.geojson"
BP = 5.7   # ppl/ha breakpoint


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    name = "Phase 18 — major-brand logistic"
    C.phase_start(name)
    C.set_seed()
    for need in (CLEAN, DESO):
        if not C.exists_nonempty(need):
            C.warn(f"missing {Path(need).name}"); C.phase_end(name); return
    st = gpd.read_file(CLEAN)
    st = st[st["include_cars"]].copy()
    g = gpd.read_file(DESO).to_crs(CRS)
    pop = C.read_json(C.CACHE / "pop_deso_regso.json")["deso"]
    g["S"] = g["desokod"].astype(str).map(pop).astype(float)
    g = g[g["S"].notna() & (g["S"] > 0)].copy()
    g["dens"] = g["S"] / (g.geometry.area / 1e4)
    j = gpd.sjoin(st.to_crs(CRS), g[["desokod", "dens", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    j = j[j["dens"].notna() & (j["dens"] > 0)].copy()
    j["is_major"] = (j["tier"] == "major").astype(int)
    j["logrho"] = np.log10(j["dens"].to_numpy(float))

    y = j["is_major"].to_numpy(float)
    X = sm.add_constant(j["logrho"].to_numpy(float))
    groups = j["desokod"].astype(str).to_numpy()

    # cluster-robust logistic
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    b0, b1 = float(res.params[0]), float(res.params[1])
    se1, p1 = float(res.bse[1]), float(res.pvalues[1])
    OR_decade = float(np.exp(b1))
    ci_or = [float(np.exp(b1 - 1.96 * se1)), float(np.exp(b1 + 1.96 * se1))]
    # McFadden pseudo-R²
    llf = float(res.llf)
    ll0 = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial()).fit().llf)
    mcf = 1 - llf / ll0
    crossover = float(10 ** (-b0 / b1)) if b1 != 0 else np.nan

    # non-parametric cross-check
    dmaj = j.loc[j["is_major"] == 1, "dens"].to_numpy(float)
    dnon = j.loc[j["is_major"] == 0, "dens"].to_numpy(float)
    U, pU = stats.mannwhitneyu(dmaj, dnon, alternative="greater")
    rbc = 1 - 2 * U / (len(dmaj) * len(dnon))   # rank-biserial (negative => maj higher)
    rpb = float(stats.pointbiserialr(j["is_major"], j["logrho"]).statistic)

    n_clusters = int(pd.Series(groups).nunique())
    stats_out = {
        "n_stations": int(len(j)), "n_major": int(y.sum()),
        "n_deso_clusters": n_clusters,
        "logistic": {
            "slope_per_decade_logrho": round(b1, 4), "se": round(se1, 4),
            "p_value": p1, "odds_ratio_per_10x_density": round(OR_decade, 3),
            "or_ci95": [round(ci_or[0], 3), round(ci_or[1], 3)],
            "intercept": round(b0, 4), "mcfadden_r2": round(mcf, 4),
            "crossover_density_ppl_ha": round(crossover, 3),
            "cov": "cluster-robust by DeSO",
        },
        "mann_whitney": {
            "U": float(U), "p_value": float(pU),
            "median_density_major": round(float(np.median(dmaj)), 3),
            "median_density_nonmajor": round(float(np.median(dnon)), 3),
            "rank_biserial": round(float(rbc), 3),
        },
        "point_biserial_r_is_major_vs_logrho": round(rpb, 3),
    }
    C.write_json(C.OUTPUT / "major_logit.json", stats_out)

    # ---- chart: empirical binned fractions + logistic curve
    j["bin"] = pd.qcut(j["logrho"], 12, labels=False, duplicates="drop")
    binstat = j.groupby("bin").agg(d=("dens", "median"), k=("is_major", "sum"),
                                   n=("is_major", "size")).reset_index()
    binstat["frac"] = binstat["k"] / binstat["n"]
    lo, hi = zip(*[wilson(k, n) for k, n in zip(binstat["k"], binstat["n"])])
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.errorbar(binstat["d"], binstat["frac"],
                yerr=[binstat["frac"] - np.array(lo), np.array(hi) - binstat["frac"]],
                fmt="o", color="#2b3a67", ms=6, capsize=3, label="empirical (12 density bins, 95% Wilson)")
    xs = np.linspace(j["logrho"].min(), j["logrho"].max(), 200)
    ax.plot(10 ** xs, 1 / (1 + np.exp(-(b0 + b1 * xs))), "-", color="#c1440e", lw=2.4,
            label=f"logistic fit (OR={OR_decade:.2f}/10×, p={p1:.1e})")
    ax.axvline(crossover, color="green", ls=":", lw=1.5,
               label=f"P=0.5 at {crossover:.1f} ppl/ha")
    ax.axvline(BP, color="k", ls="--", lw=1.2, label=f"scaling breakpoint {BP} ppl/ha")
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_xlabel("DeSO population density ρ (ppl/ha)")
    ax.set_ylabel("P(station is a major brand)")
    ax.set_title(f"Major-brand probability rises with density\n"
                 f"OR = {OR_decade:.2f} per 10× density "
                 f"[{ci_or[0]:.2f}, {ci_or[1]:.2f}], McFadden R²={mcf:.2f}, "
                 f"n={len(j)} stations / {n_clusters} DeSO")
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    out = C.CHARTS / "18_major_logit.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    C.log(f"[OK] {out.name}")

    C.log(f"logistic: OR per 10× density = {OR_decade:.2f} [{ci_or[0]:.2f},{ci_or[1]:.2f}], "
          f"p={p1:.2e}, McFadden R²={mcf:.3f}, crossover={crossover:.2f} ppl/ha")
    C.log(f"Mann-Whitney: median ρ major={stats_out['mann_whitney']['median_density_major']} vs "
          f"non-major={stats_out['mann_whitney']['median_density_nonmajor']} ppl/ha, p={pU:.2e}")
    C.phase_end(name)


if __name__ == "__main__":
    main()

"""Phase 8 — per-unit "surprise" under the fitted inhomogeneous model, and formal
tests for whether the surprises CLUSTER (geographically, by brand, by border).

Model A is an inhomogeneous Poisson/NB: each kommun has its own fitted rate. One
kommun in the tail proves nothing (some always will); the signal is *structure in
the tail*. So we score each unit and then test the structure.

DENOMINATOR. Population is a biased exposure for fuel demand — urban car-ownership
per capita is lower — so the PRIMARY exposure here is registered CARS (Phase 7),
not people. We also keep the population fit and a cars+income fit, and report how
the urban "cold spot" moves across the three, to separate three explanations of an
urban deficit: (a) fewer cars per head, (b) high-income / restrictive land use,
(c) genuine residual under-provision.

  z_pop   : exposure = population
  z_cars  : exposure = registered cars                       [if cars available]
  z_ci    : exposure = cars, covariate = log(median income)  [PRIMARY if available]

For each unit: randomized-quantile residual z (~N(0,1) under the model; discreteness-
correct, so 0/1/2-station units are scored honestly) + two-sided mid-p tail p; then
BH-FDR over units. On the primary z we run:
  * GEOGRAPHIC clustering — global Moran's I (+perm) and local Moran (LISA) -> HH/LL.
  * BORDER split — distance to the Norway vs Finland land border, separately, and a
    permutation test of mean z in each border band vs the interior.
  * BRAND clustering — each station inherits host-kommun z; label-permutation test of
    whether a station type/brand sits in over-/under-stationed kommuner.

Outputs: output/surprise_kommun.csv, output/surprise_summary.json, and charts
08 (map), 09 (brand), 10 (what explains the deviations: income + border).
"""
import sys
import importlib.util
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
from matplotlib.lines import Line2D  # noqa: E402
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

CRS = C.CONFIG["national_crs"]
BUF_KM = C.CONFIG["border_buffer_km"]      # 10 km band
DPI = 160
K_NN = 6
N_PERM = 999
FDR_Q = 0.10
LISA_ALPHA = 0.05
LAU = C.CACHE / "lau_2024_3035.gpkg"

# reuse the Phase-6 brand classifier rather than duplicate the keyword lists
_spec = importlib.util.spec_from_file_location(
    "dev06", Path(__file__).resolve().parent / "06_deviation.py")
_dev06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dev06)
classify = _dev06.classify

INCOME_URL = ("https://api.scb.se/OV0104/v1/doris/en/ssd/HE/HE0110/HE0110A/SamForvInk2")
INCOME_CACHE = C.CACHE / "income_per_kommun.json"
CARS_CACHE = C.CACHE / "cars_per_kommun.json"
TOURISM_CACHE = C.CACHE / "tourism_per_kommun.json"  # from Phase 9


# ----------------------------------------------------------------- covariate load
def fetch_income():
    """Median earned income (SEK thousands, 20+), per kommun, latest year. Cached."""
    if C.exists_nonempty(INCOME_CACHE):
        return C.read_json(INCOME_CACHE)
    import requests
    meta = C.retry(lambda: requests.get(INCOME_URL, timeout=120).json(),
                   what="SCB income metadata")
    year = [v["values"] for v in meta["variables"] if v.get("time")][0][-1]
    q = {"query": [
        {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
        {"code": "Kon", "selection": {"filter": "item", "values": ["1+2"]}},
        {"code": "Alder", "selection": {"filter": "item", "values": ["tot20+"]}},
        {"code": "Inkomstklass", "selection": {"filter": "item", "values": ["TOT"]}},
        {"code": "ContentsCode", "selection": {"filter": "item", "values": ["HE0110K2"]}},
        {"code": "Tid", "selection": {"filter": "item", "values": [year]}},
    ], "response": {"format": "json"}}
    js = C.retry(lambda: requests.post(INCOME_URL, json=q, timeout=300).json(),
                 what="SCB income query")
    inc = {}
    for row in js["data"]:
        code = row["key"][0]
        if len(code) == 4 and code.isdigit():
            try:
                inc[code] = float(row["values"][0])
            except (ValueError, TypeError):
                pass
    out = {"year": year, "contents": "median_earned_income_SEKk_20plus", "income": inc}
    C.write_json(INCOME_CACHE, out)
    C.update_manifest(C.manifest_entry_for(INCOME_CACHE, INCOME_URL + f" (median income, {year})"))
    return out


def load_covariates(kom):
    """Attach V (cars) and INC (median income). Returns (kom, meta-dict)."""
    kom = kom.copy()
    kom["kommun_code"] = kom["unit_id"].str.replace("SE_", "", regex=False)
    meta = {"cars": None, "income": None}
    if C.exists_nonempty(CARS_CACHE):
        cars = C.read_json(CARS_CACHE)
        kom["V"] = kom["kommun_code"].map(cars["cars"]).astype(float)
        meta["cars"] = cars.get("year")
    else:
        kom["V"] = np.nan
    try:
        income = fetch_income()
        kom["INC"] = kom["kommun_code"].map(income["income"]).astype(float)
        meta["income"] = income.get("year")
    except Exception as e:  # noqa: BLE001
        C.warn(f"income unavailable (non-fatal): {e}")
        kom["INC"] = np.nan
    # tourism "effective customer base" proxy (Phase 9): accommodation + ski lifts.
    # TOUR enters the model as log1p, so it is a visitor-base ADD-ON to local cars.
    meta["tourism"] = False
    if C.exists_nonempty(TOURISM_CACHE):
        t = C.read_json(TOURISM_CACHE)["by_unit"]
        acc = kom["unit_id"].map(lambda u: t.get(u, {}).get("tour_accommodation", 0))
        ski = kom["unit_id"].map(lambda u: t.get(u, {}).get("tour_ski_lift", 0))
        kom["TOUR_acc"] = acc.astype(float)
        kom["TOUR_ski"] = ski.astype(float)
        kom["TOUR"] = kom["TOUR_acc"] + kom["TOUR_ski"]   # combined POI count
        meta["tourism"] = True
    return kom, meta


# ----------------------------------------------------------------------------- model
def fit_nb(y, X):
    """Poisson, escalate to NB2 if overdispersed. Returns (mu, alpha_or_None, res, disp)."""
    pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    disp = float(pois.pearson_chi2 / pois.df_resid)
    if disp > 1.2:
        nb = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=200)
        alpha = float(nb.params[-1])
        glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=max(alpha, 1e-6))).fit()
        return glm.predict(X), alpha, glm, disp
    return pois.predict(X), None, pois, disp


def surprise(y, mu, alpha, seed=C.SEED):
    """Randomized-quantile residual z (~N(0,1) under model) + two-sided mid-p."""
    y = y.astype(int)
    if alpha is None:
        dist = stats.poisson(mu)
    else:
        r = 1.0 / alpha
        dist = stats.nbinom(r, r / (r + mu))
    cdf_lo = dist.cdf(y - 1)
    pmf = dist.pmf(y)
    rng = np.random.default_rng(seed)
    u = np.clip(cdf_lo + rng.uniform(size=len(y)) * pmf, 1e-12, 1 - 1e-12)
    z = stats.norm.ppf(u)
    midp_lo = cdf_lo + 0.5 * pmf
    midp_hi = 1.0 - dist.cdf(y) + 0.5 * pmf
    p_two = np.clip(2.0 * np.minimum(midp_lo, midp_hi), 0.0, 1.0)
    return z, p_two


def fit_surprise(kom, exposure, covars=()):
    """Fit Model A with given exposure (log) + optional covariates. Each covariate
    may be a name (log transform) or a (name, 'log1p') tuple for count covariates
    with zeros (e.g. tourism). Returns (z, p, beta_exposure, coef_dict, alpha, disp)."""
    y = kom["P"].to_numpy(float)
    cols = [np.log(kom[exposure].to_numpy(float))]
    names = [f"log_{exposure}"]
    for cv in covars:
        cv_name, how = (cv if isinstance(cv, tuple) else (cv, "log"))
        v = kom[cv_name].to_numpy(float)
        cols.append(np.log1p(v) if how == "log1p" else np.log(v))
        names.append(f"{how}_{cv_name}")
    X = sm.add_constant(np.column_stack(cols))
    mu, alpha, res, disp = fit_nb(y, X)
    z, p = surprise(y, mu, alpha)
    coef = {names[i]: {"coef": round(float(res.params[i + 1]), 4),
                       "se": round(float(res.bse[i + 1]), 4),
                       "p": float(res.pvalues[i + 1])} for i in range(len(names))}
    return z, p, float(res.params[1]), coef, alpha, disp


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n); q[order] = np.clip(q_sorted, 0, 1)
    return q


# ----------------------------------------------------------------- spatial weights
def knn_weights(gdf, k=K_NN):
    pts = gdf.geometry.representative_point()
    xy = np.column_stack([pts.x.to_numpy(), pts.y.to_numpy()])
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nbr = np.argsort(d, axis=1)[:, :k]
    n = len(xy); W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), nbr.ravel()] = 1.0
    W /= W.sum(axis=1, keepdims=True)
    return W


def morans_i(x, W):
    z = x - x.mean()
    return (len(x) / W.sum()) * (z @ (W @ z)) / (z * z).sum()


def global_moran(x, W, n_perm=N_PERM, seed=C.SEED):
    I_obs = morans_i(x, W)
    rng = np.random.default_rng(seed)
    perm = np.array([morans_i(rng.permutation(x), W) for _ in range(n_perm)])
    return float(I_obs), float((1 + (perm >= I_obs).sum()) / (1 + n_perm)), \
        float(perm.mean()), float(perm.std())


def local_moran(x, W, n_perm=N_PERM, seed=C.SEED):
    n = len(x); z = x - x.mean(); m2 = (z * z).sum() / n; lag = W @ z
    Ii = (z / m2) * lag
    rng = np.random.default_rng(seed)
    p = np.empty(n); knn_idx = [np.nonzero(W[i] > 0)[0] for i in range(n)]
    for i in range(n):
        others = np.delete(z, i); wi = W[i, knn_idx[i]]
        sims = np.array([(z[i] / m2) * (wi @ rng.choice(others, size=len(wi), replace=False))
                         for _ in range(n_perm)])
        p[i] = (1 + (np.abs(sims) >= abs(Ii[i])).sum()) / (1 + n_perm)
    quad = np.where((z > 0) & (lag > 0), "HH",
           np.where((z < 0) & (lag < 0), "LL",
           np.where((z > 0) & (lag < 0), "HL", "LH")))
    return Ii, p, np.where(p < LISA_ALPHA, quad, "ns")


# --------------------------------------------------------------------- border split
def border_distances(kom):
    """Per-unit distance (km) to the Norway and Finland land borders, separately,
    and a band label (Norway / Finland / interior within BUF_KM)."""
    out = kom.copy()
    for cc, col in [("NO", "dist_no_km"), ("FI", "dist_fi_km")]:
        nb = gpd.read_file(LAU, where=f"CNTR_CODE='{cc}'").to_crs(CRS)
        union = nb.geometry.union_all()
        out[col] = out.geometry.distance(union) / 1000.0
    out["border_band"] = np.where(out["dist_no_km"] <= BUF_KM, "Norway",
                          np.where(out["dist_fi_km"] <= BUF_KM, "Finland", "interior"))
    return out


def band_test(kom, band, n_perm=N_PERM, seed=C.SEED):
    """Permutation test: mean z of a border band vs the whole-country mean."""
    mask = (kom["border_band"] == band).to_numpy()
    m = int(mask.sum())
    if m == 0:
        return {"band": band, "n": 0}
    z = kom["z"].to_numpy(float)
    obs = float(z[mask].mean())
    rng = np.random.default_rng(seed)
    sims = np.array([rng.choice(z, size=m, replace=False).mean() for _ in range(n_perm)])
    p = (1 + (np.abs(sims - z.mean()) >= abs(obs - z.mean())).sum()) / (1 + n_perm)
    return {"band": band, "n": m, "mean_z": round(obs, 3), "perm_p": round(float(p), 4)}


# ----------------------------------------------------------------- brand clustering
def brand_tests(kom_z, n_perm=N_PERM, seed=C.SEED):
    st = gpd.read_file(C.CACHE / "stations.gpkg").to_crs(CRS)
    st = st[st.geometry.notna() & ~st.geometry.is_empty].copy()
    lab = st["brand"].fillna(st["operator"]).fillna(st["name"])
    st["stype"] = lab.map(classify)
    st["brand_norm"] = lab.fillna("(unknown)").astype(str).str.strip()
    kz = kom_z[["unit_id", "geometry", "z"]]
    j = gpd.sjoin(st[["stype", "brand_norm", "geometry"]],
                  kz[["unit_id", "geometry"]], how="inner", predicate="within")
    j = j.merge(kz[["unit_id", "z"]], on="unit_id", how="left").dropna(subset=["z"])
    zvals = j["z"].to_numpy(float); pool_mean = float(zvals.mean())
    rng = np.random.default_rng(seed)

    def test_group(mask, name):
        m = int(mask.sum())
        if m < 5:
            return None
        obs = float(zvals[mask].mean()); effect = obs - pool_mean
        sims = np.array([rng.choice(zvals, size=m, replace=False).mean() for _ in range(n_perm)])
        p = (1 + (np.abs(sims - pool_mean) >= abs(effect)).sum()) / (1 + n_perm)
        return {"group": name, "n_stations": m, "mean_host_z": round(obs, 3),
                "effect_vs_pool": round(effect, 3), "perm_p": round(float(p), 4)}

    rows = []
    for t in ["automat", "full_service", "gas_alt", "unknown"]:
        r = test_group((j["stype"] == t).to_numpy(), f"type:{t}")
        if r:
            rows.append(r)
    for b in j["brand_norm"].value_counts().head(12).index:
        r = test_group((j["brand_norm"] == b).to_numpy(), f"brand:{b}")
        if r:
            rows.append(r)
    rows.sort(key=lambda d: d["effect_vs_pool"])
    return rows, pool_mean


# ------------------------------------------------------------------------- charts
def chart_map(kom):
    fig, ax = plt.subplots(figsize=(7.5, 10))
    vlim = float(np.nanpercentile(np.abs(kom["z"]), 98))
    kom.plot(ax=ax, column="z", cmap="RdBu_r", vmin=-vlim, vmax=vlim, linewidth=0.2,
             edgecolor="#888888", legend=True,
             legend_kwds={"label": "surprise z  (red = more stations than expected, per car)",
                          "shrink": 0.4})
    hit = kom[kom["q_fdr"] < FDR_Q]
    if len(hit):
        hit.boundary.plot(ax=ax, color="black", linewidth=1.1)
    for lab, mk, col in [("HH", "^", "darkred"), ("LL", "v", "darkblue")]:
        cl = kom[kom["lisa"] == lab]
        if len(cl):
            pts = cl.geometry.representative_point()
            ax.scatter(pts.x, pts.y, marker=mk, s=34, c=col, edgecolors="white",
                       linewidths=0.4, zorder=5)
    ax.set_axis_off()
    ax.set_title("Where the petrol-scaling surprises cluster (exposure = registered cars)\n"
                 f"black outline: FDR-significant (q<{FDR_Q});  ▲ HH over  ▼ LL under (LISA p<{LISA_ALPHA})")
    ax.legend(handles=[
        Line2D([0], [0], color="black", lw=1.1, label=f"FDR hit (q<{FDR_Q})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="darkred", markersize=9, label="HH cluster"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="darkblue", markersize=9, label="LL cluster")],
        loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(C.CHARTS / "08_surprise_map.png", dpi=DPI)
    plt.close(fig); C.log("[OK] chart 08_surprise_map.png")


def chart_brands(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    from matplotlib.colors import to_rgba
    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.42 * len(df))))
    y = np.arange(len(df))
    colors = [to_rgba("#b2182b" if e > 0 else "#2166ac", 0.95 if p < 0.05 else 0.4)
              for e, p in zip(df["effect_vs_pool"], df["perm_p"])]
    ax.barh(y, df["effect_vs_pool"], color=colors)
    for i, r in df.iterrows():
        star = " *" if r["perm_p"] < 0.05 else ""
        ax.text(r["effect_vs_pool"] + 0.006, i, f"n={r['n_stations']}, p={r['perm_p']:.3f}{star}",
                va="center", ha="left", fontsize=7)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(df["group"], fontsize=8)
    ax.set_xlabel("host-kommun surprise z RELATIVE TO THE AVERAGE STATION\n"
                  "← under-stationed (urban) kommuner   |   over-stationed (rural) kommuner →")
    ax.set_title("Do station types / brands concentrate in over- or under-stationed kommuner?\n"
                 "label-permutation test vs the average station; solid bars = p<0.05  (* significant)")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(C.CHARTS / "09_brand_surprise.png", dpi=DPI)
    plt.close(fig); C.log("[OK] chart 09_brand_surprise.png")


def chart_explain(coldset_means, band_compare, has_tourism):
    """Left: urban cold-spot mean z across denominator/covariate models.
    Right: border-band mean z BEFORE vs AFTER controlling for tourism — does the
    Norway over-stationing survive (fuel arbitrage) or shrink to 0 (ski tourism)?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    labs = list(coldset_means.keys()); vals = [coldset_means[k] for k in labs]
    ax1.bar(range(len(labs)), vals,
            color=["#777777", "#7fb0e0", "#2166ac", "#0a3d7a"][:len(labs)])
    ax1.set_xticks(range(len(labs))); ax1.set_xticklabels(labs, fontsize=9)
    ax1.axhline(0, color="black", lw=1)
    for i, v in enumerate(vals):
        ax1.text(i, v - 0.05, f"{v:+.2f}", ha="center", va="top", fontsize=10)
    ax1.set_ylabel("mean surprise z of the per-capita under-stationed set")
    ax1.set_title("Does the urban deficit survive better models?\n"
                  "(per-capita cold-spot kommuner, z closer to 0 = explained away)")
    ax1.grid(True, axis="y", alpha=0.2)

    bands = list(band_compare.keys())
    series = list(next(iter(band_compare.values())).keys())  # model columns present
    pretty = {"z_cars_income": "cars + income", "z_cars_income_tour": "cars + income + tourism"}
    x = np.arange(len(bands)); w = 0.8 / max(len(series), 1)
    shades = {"z_cars_income": "#9ecae1", "z_cars_income_tour": "#08519c"}
    for si, s in enumerate(series):
        vals = [band_compare[b][s] for b in bands]
        ax2.bar(x + si * w, vals, width=w, color=shades.get(s, "#888"),
                label=pretty.get(s, s))
        for xi, v in zip(x + si * w, vals):
            ax2.text(xi, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=7)
    ax2.set_xticks(x + w * (len(series) - 1) / 2); ax2.set_xticklabels(bands)
    ax2.axhline(0, color="black", lw=1); ax2.legend(fontsize=8)
    ax2.set_ylabel("mean surprise z by border band")
    ax2.set_title("Does border over-stationing survive a tourism control?\n"
                  "(Norway bar staying >0 = cross-border fuel, not only ski tourism)", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(C.CHARTS / "10_explain_deviations.png", dpi=DPI)
    plt.close(fig); C.log("[OK] chart 10_explain_deviations.png")


# --------------------------------------------------------------------------- main
def main():
    name = "Phase 8 — surprise & clustering"
    C.phase_start(name)
    C.set_seed()

    kom = gpd.read_file(C.CACHE / "units_kommun.gpkg")
    kom = kom[kom["S"].notna() & (kom["S"] > 0)].copy().reset_index(drop=True)
    kom, cov_meta = load_covariates(kom)

    has_cars = kom["V"].notna().any() and (kom["V"] > 0).any()
    has_income = kom["INC"].notna().sum() >= 0.9 * len(kom)
    # analysis frame: need positive exposure(s)
    keep = kom["S"] > 0
    if has_cars:
        keep &= kom["V"].notna() & (kom["V"] > 0)
    if has_income:
        keep &= kom["INC"].notna() & (kom["INC"] > 0)
    kom = kom[keep].copy().reset_index(drop=True)

    # ---- three surprise models for the denominator/income comparison
    models = {}
    z_pop, p_pop, b_pop, _, a_pop, d_pop = fit_surprise(kom, "S")
    kom["z_pop"] = z_pop
    models["population"] = {"exposure": "population", "beta": round(b_pop, 4),
                            "alpha": (round(a_pop, 4) if a_pop else None), "dispersion": round(d_pop, 3)}
    primary_z, primary_p = z_pop, p_pop
    primary_label = "population"
    income_coef = None
    if has_cars:
        z_c, p_c, b_c, _, a_c, d_c = fit_surprise(kom, "V")
        kom["z_cars"] = z_c
        models["cars"] = {"exposure": "cars", "beta": round(b_c, 4),
                          "alpha": (round(a_c, 4) if a_c else None), "dispersion": round(d_c, 3)}
        primary_z, primary_p, primary_label = z_c, p_c, "cars"
    if has_cars and has_income:
        z_ci, p_ci, b_ci, coef_ci, a_ci, d_ci = fit_surprise(kom, "V", covars=("INC",))
        kom["z_cars_income"] = z_ci
        income_coef = coef_ci.get("log_INC")
        models["cars_income"] = {"exposure": "cars", "covariate": "log(median_income)",
                                 "beta_cars": round(b_ci, 4), "income_coef": income_coef,
                                 "alpha": (round(a_ci, 4) if a_ci else None), "dispersion": round(d_ci, 3)}
        primary_z, primary_p, primary_label = z_ci, p_ci, "cars_income"
    has_tourism = bool(cov_meta.get("tourism"))
    tourism_coef = None
    if has_cars and has_income and has_tourism:
        z_cit, p_cit, b_cit, coef_cit, a_cit, d_cit = fit_surprise(
            kom, "V", covars=("INC", ("TOUR", "log1p")))
        kom["z_cars_income_tour"] = z_cit
        tourism_coef = coef_cit.get("log1p_TOUR")
        models["cars_income_tourism"] = {
            "exposure": "cars", "covariates": ["log(median_income)", "log1p(tourism_POIs)"],
            "beta_cars": round(b_cit, 4), "income_coef": coef_cit.get("log_INC"),
            "tourism_coef": tourism_coef,
            "alpha": (round(a_cit, 4) if a_cit else None), "dispersion": round(d_cit, 3)}
        primary_z, primary_p, primary_label = z_cit, p_cit, "cars_income_tourism"

    kom["z"] = primary_z
    kom["p_two"] = primary_p
    kom["q_fdr"] = bh_fdr(primary_p)

    # ---- "urban cold-spot" set defined by the population model, tracked across models
    cold = kom["z_pop"] < -1.0
    coldset_means = {"population": round(float(kom.loc[cold, "z_pop"].mean()), 3)}
    if "z_cars" in kom:
        coldset_means["cars"] = round(float(kom.loc[cold, "z_cars"].mean()), 3)
    if "z_cars_income" in kom:
        coldset_means["cars+income"] = round(float(kom.loc[cold, "z_cars_income"].mean()), 3)
    if "z_cars_income_tour" in kom:
        coldset_means["cars+income\n+tourism"] = round(float(kom.loc[cold, "z_cars_income_tour"].mean()), 3)

    # ---- geographic clustering on the primary z
    W = knn_weights(kom, K_NN)
    I_obs, I_p, I_mu, I_sd = global_moran(kom["z"].to_numpy(), W)
    Ii, lisa_p, lisa_lab = local_moran(kom["z"].to_numpy(), W)
    kom["lisa"] = lisa_lab; kom["lisa_p"] = lisa_p

    # ---- border split + tests
    kom = border_distances(kom)
    band_rows = [band_test(kom, b) for b in ["Norway", "Finland", "interior"]]
    # does the border over-stationing SURVIVE controlling for tourism? Compare band
    # mean z under cars+income vs cars+income+tourism. If the Norway bar shrinks to
    # ~0, it was ski tourism; if it persists, it is genuine cross-border fuel demand.
    survival_models = [c for c in ["z_cars_income", "z_cars_income_tour"] if c in kom]
    band_compare = {b: {c: round(float(kom.loc[kom["border_band"] == b, c].mean()), 3)
                        for c in survival_models}
                    for b in ["Norway", "Finland", "interior"]}

    # ---- brand clustering on the primary z
    brand_rows, pool_mean = brand_tests(kom[["unit_id", "geometry", "z"]].copy())

    # ---- table
    cols = ["unit_id", "name", "S", "P"]
    for c in ["V", "INC", "TOUR_acc", "TOUR_ski", "z_pop", "z_cars", "z_cars_income",
              "z_cars_income_tour"]:
        if c in kom:
            cols.append(c)
    cols += ["z", "p_two", "q_fdr", "lisa", "lisa_p", "border_band", "dist_no_km", "dist_fi_km"]
    tbl = kom[cols].copy()
    for c in ["z_pop", "z_cars", "z_cars_income", "z_cars_income_tour", "z",
              "dist_no_km", "dist_fi_km"]:
        if c in tbl:
            tbl[c] = tbl[c].round(3)
    tbl = tbl.sort_values("z")
    tbl.to_csv(C.OUTPUT / "surprise_kommun.csv", index=False)
    C.log(f"[OK] wrote surprise_kommun.csv (n={len(tbl)})")

    # ---- charts
    chart_map(kom)
    chart_brands(brand_rows)
    chart_explain(coldset_means, band_compare, has_tourism)

    # ---- summary
    hh = kom[kom["lisa"] == "HH"].sort_values("z", ascending=False)
    ll = kom[kom["lisa"] == "LL"].sort_values("z")
    fdr_hits = kom[kom["q_fdr"] < FDR_Q].sort_values("z")
    summary = {
        "primary_model": primary_label,
        "exposure_note": "PRIMARY exposure = registered cars (Phase 7); population & "
                         "cars+income kept for comparison. Cars year=%s, income year=%s." % (
                             cov_meta.get("cars"), cov_meta.get("income")),
        "models": models,
        "income_effect": (None if income_coef is None else {
            **income_coef,
            "interpretation": ("holding cars fixed, higher median income is associated with "
                               + ("FEWER" if income_coef["coef"] < 0 else "MORE")
                               + " petrol stations" + (" (significant)" if income_coef["p"] < 0.05
                                                       else " (not significant)"))}),
        "tourism_effect": (None if tourism_coef is None else {
            **tourism_coef,
            "proxy": "log1p(accommodation + ski-lift POIs from OSM)",
            "interpretation": ("holding cars & income fixed, more tourism POIs are associated with "
                               + ("MORE" if tourism_coef["coef"] > 0 else "FEWER")
                               + " petrol stations — a visitor 'effective customer base'"
                               + (" (significant)" if tourism_coef["p"] < 0.05 else " (not significant)"))}),
        "border_survives_tourism": {
            "band_mean_z": band_compare,
            "reading": "Norway band z staying high from 'cars+income' to '+tourism' => genuine "
                       "cross-border fuel demand; collapsing toward 0 => it was ski tourism"},
        "urban_coldspot_across_models": {
            "definition": "kommuner with population-model z < -1 (per-capita under-stationed)",
            "n": int(cold.sum()), "mean_z": coldset_means,
            "reading": "z moving toward 0 left->right means cars / income explain the urban deficit"},
        "n_units": int(len(kom)),
        "expected_extreme_if_null": {"|z|>1.96_expected": round(0.05 * len(kom), 1),
                                     "|z|>1.96_observed": int((kom["z"].abs() > 1.96).sum())},
        "fdr": {"q_threshold": FDR_Q, "n_significant": int(len(fdr_hits)),
                "units": fdr_hits[["name", "P", "z", "q_fdr", "border_band"]]
                .assign(z=lambda d: d["z"].round(2), q_fdr=lambda d: d["q_fdr"].round(3))
                .to_dict("records")},
        "global_moran": {"I": round(I_obs, 4), "perm_p": round(I_p, 4),
                         "null_mean": round(I_mu, 4), "null_sd": round(I_sd, 4),
                         "interpretation": ("surprises ARE geographically clustered"
                                            if I_p < 0.05 else "no significant geographic clustering")},
        "lisa_clusters": {
            "HH_over_stationed": hh[["name", "P", "z", "border_band"]]
            .assign(z=lambda d: d["z"].round(2)).to_dict("records"),
            "LL_under_stationed": ll[["name", "P", "z", "border_band"]]
            .assign(z=lambda d: d["z"].round(2)).to_dict("records")},
        "border_split": {"band_tests": band_rows,
                         "note": "distance to NO vs FI land border computed separately; "
                                 f"band = within {BUF_KM} km"},
        "brand_type_tests": {"pool_mean_z": round(pool_mean, 3), "groups": brand_rows},
    }
    C.write_json(C.OUTPUT / "surprise_summary.json", summary)
    kom.to_file(C.CACHE / "surprise_kommun.gpkg", driver="GPKG")
    C.log(f"primary={primary_label} | Moran I={I_obs:.3f} (p={I_p:.3f}) | "
          f"coldset z {coldset_means} | bands "
          f"{[ (r['band'], r.get('mean_z')) for r in band_rows ]}")
    C.phase_end(name)


if __name__ == "__main__":
    main()

"""Phase 3 — scaling regressions P = C * S^beta, per unit definition.

Model A (PRIMARY): Negative-Binomial GLM, log link  log E[P] = a + beta*log S
                   (Poisson if not overdispersed). Tests H0: beta = 1.
Model B (compare): OLS of log10 P on log10 S, positives only (zero-truncation biased).
Bootstrap CI for beta. Quadratic / single-power-law check + segmented breakpoint.
Leakage check: refit excluding is_border units.

P=0 units are KEPT for Model A. Writes output/results.json and
cache/resid_<def>.gpkg (deviance/Pearson residuals + fitted) for charts.
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

warnings.filterwarnings("ignore")

UNIT_FILES = {
    "tatort": C.CACHE / "units_tatort.gpkg",
    "tatort_smaort": C.CACHE / "units_tatort_smaort.gpkg",
    "kommun": C.CACHE / "units_kommun.gpkg",
    "fua": C.CACHE / "units_fua.gpkg",
}
BOOT_N = C.CONFIG["bootstrap_n"]


def verdict(beta, se):
    """H0: beta = 1. Two-sided z-test."""
    z = (beta - 1.0) / se
    p = 2 * stats.norm.sf(abs(z))
    if p >= 0.05:
        v = "indistinguishable from linear"
    elif beta < 1:
        v = "sublinear"
    else:
        v = "superlinear"
    return {"z_vs1": round(float(z), 3), "p_vs1": float(p), "verdict": v}


def fit_nb_mle(y, X):
    """NB2 MLE: returns (beta, se_beta, alpha, llf, llnull, prsquared, intercept)."""
    res = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=200)
    beta = float(res.params[1]); se = float(res.bse[1])
    alpha = float(res.params[-1])
    return res, beta, se, alpha


def fit_nb_glm(y, X, alpha):
    fam = sm.families.NegativeBinomial(alpha=max(alpha, 1e-6))
    return sm.GLM(y, X, family=fam).fit()


def model_a(df):
    y = df["P"].to_numpy(float)
    logS = np.log(df["S"].to_numpy(float))
    X = sm.add_constant(logS)

    # overdispersion diagnostic from Poisson
    pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    disp = float(pois.pearson_chi2 / pois.df_resid)
    overdispersed = disp > 1.2

    out = {"n": int(len(df)), "n_zero": int((y == 0).sum()),
           "poisson_dispersion": round(disp, 3), "overdispersed": bool(overdispersed)}

    if overdispersed:
        res, beta, se, alpha = fit_nb_mle(y, X)
        out["family"] = "NegativeBinomial(NB2, MLE)"
        out["nb_alpha"] = round(alpha, 4)
        out["intercept"] = float(res.params[0])
        out["pseudo_r2_mcfadden"] = round(float(res.prsquared), 4)
        alpha_for_boot = alpha
    else:
        res = pois; beta = float(pois.params[1]); se = float(pois.bse[1])
        out["family"] = "Poisson (not overdispersed)"
        out["intercept"] = float(pois.params[0])
        ll0 = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Poisson()).fit().llf
        out["pseudo_r2_mcfadden"] = round(1 - pois.llf / ll0, 4)
        alpha_for_boot = None

    ci = (beta - 1.96 * se, beta + 1.96 * se)
    out.update(beta=round(beta, 4), se=round(se, 4),
               ci95=[round(ci[0], 4), round(ci[1], 4)])
    out.update(verdict(beta, se))
    return out, res, alpha_for_boot


def model_b_ols(df):
    pos = df[df["P"] > 0].copy()
    n_drop = int((df["P"] == 0).sum())
    x = np.log10(pos["S"].to_numpy(float))
    yv = np.log10(pos["P"].to_numpy(float))
    X = sm.add_constant(x)
    res = sm.OLS(yv, X).fit()
    beta = float(res.params[1]); se = float(res.bse[1])
    return {
        "family": "OLS log10(P)~log10(S), positives only",
        "beta": round(beta, 4), "se": round(se, 4),
        "t": round(float(res.tvalues[1]), 3), "r2": round(float(res.rsquared), 4),
        "n_used": int(len(pos)), "n_dropped_zeros": n_drop,
        "note": "zero-truncation biased UPWARD; shown only to compare with classic US OLS literature",
        **{"verdict_vs1_" + k: v for k, v in verdict(beta, se).items()},
    }


def bootstrap_beta(df, alpha_for_boot, n=BOOT_N):
    rng = np.random.default_rng(C.SEED)
    y = df["P"].to_numpy(float)
    logS = np.log(df["S"].to_numpy(float))
    X = sm.add_constant(logS)
    idx = np.arange(len(df))
    betas = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        try:
            if alpha_for_boot is None:
                r = sm.GLM(y[s], X[s], family=sm.families.Poisson()).fit()
            else:
                r = fit_nb_glm(y[s], X[s], alpha_for_boot)
            betas.append(float(r.params[1]))
        except Exception:  # noqa: BLE001
            continue
    betas = np.array(betas)
    return {
        "n_success": int(len(betas)),
        "beta_mean": round(float(betas.mean()), 4),
        "ci95": [round(float(np.percentile(betas, 2.5)), 4),
                 round(float(np.percentile(betas, 97.5)), 4)],
    }


def quadratic_check(df):
    y = df["P"].to_numpy(float)
    logS = np.log(df["S"].to_numpy(float))
    X = sm.add_constant(np.column_stack([logS, logS ** 2]))
    pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    disp = pois.pearson_chi2 / pois.df_resid
    if disp > 1.2:
        try:
            res = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=200)
        except Exception:  # noqa: BLE001
            res = pois
    else:
        res = pois
    coef = float(res.params[2]); se = float(res.bse[2]); p = float(res.pvalues[2])
    out = {"quad_coef": round(coef, 5), "quad_se": round(se, 5), "quad_p": p,
           "significant": bool(p < 0.05),
           "interpretation": ("curvature present -> NOT a clean single power law"
                              if p < 0.05 else "consistent with single power law")}
    if p < 0.05:
        out["segmented"] = segmented(df)
    return out


def segmented(df):
    """Broken-stick NB/Poisson on log S; grid-search breakpoint by deviance."""
    y = df["P"].to_numpy(float)
    logS = np.log(df["S"].to_numpy(float))
    cands = np.quantile(logS, np.linspace(0.15, 0.85, 25))
    best = None
    for c in cands:
        hinge = np.maximum(0.0, logS - c)
        X = sm.add_constant(np.column_stack([logS, hinge]))
        try:
            r = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        except Exception:  # noqa: BLE001
            continue
        if best is None or r.deviance < best[0]:
            best = (r.deviance, c, r)
    if best is None:
        return {"status": "failed"}
    dev, c, r = best
    s1 = float(r.params[1]); s2 = float(r.params[1] + r.params[2])
    return {"breakpoint_logS": round(float(c), 3),
            "breakpoint_S": int(round(float(np.exp(c)))),
            "slope_below": round(s1, 4), "slope_above": round(s2, 4)}


def leakage_check(df, alpha_for_boot):
    sub = df[~df["is_border"]].copy()
    y = sub["P"].to_numpy(float)
    logS = np.log(sub["S"].to_numpy(float))
    X = sm.add_constant(logS)
    if alpha_for_boot is None:
        r = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    else:
        r = fit_nb_glm(y, X, alpha_for_boot)
    beta = float(r.params[1]); se = float(r.bse[1])
    return {"n_excluded_border": int(df["is_border"].sum()),
            "n_used": int(len(sub)),
            "beta_no_border": round(beta, 4), "se": round(se, 4),
            **verdict(beta, se)}


def save_residuals(df, res, defname):
    """Save deviance & Pearson residuals + fitted for charts (tätort especially)."""
    g = df.copy()
    logS = np.log(g["S"].to_numpy(float))
    X = sm.add_constant(logS)
    mu = res.predict(X) if hasattr(res, "predict") else None
    try:
        g["fitted"] = np.asarray(mu, float)
        g["resid_pearson"] = (g["P"].to_numpy(float) - g["fitted"]) / np.sqrt(
            np.maximum(g["fitted"].to_numpy(float), 1e-9))
        # deviance residual via statsmodels if GLM
        if hasattr(res, "resid_deviance"):
            g["resid_deviance"] = np.asarray(res.resid_deviance, float)
        else:
            g["resid_deviance"] = g["resid_pearson"]
    except Exception as e:  # noqa: BLE001
        C.warn(f"residual computation failed for {defname}: {e}")
        return
    out = C.CACHE / f"resid_{defname}.gpkg"
    g.to_file(out, driver="GPKG")
    C.log(f"[OK] residuals -> {out.name}")


def analyse(defname, path):
    df = gpd.read_file(path)
    df = df[df["S"].notna() & (df["S"] > 0)].copy()
    C.log(f"[{defname}] n={len(df)} zeros={int((df['P']==0).sum())} sumP={int(df['P'].sum())}")
    a, res, alpha_boot = model_a(df)
    result = {
        "definition": defname,
        "n_units": int(len(df)),
        "sum_P": int(df["P"].sum()),
        "model_A_primary": a,
        "model_B_ols": model_b_ols(df),
        "bootstrap_modelA": bootstrap_beta(df, alpha_boot),
        "single_power_law_check": quadratic_check(df),
        "leakage_check": leakage_check(df, alpha_boot),
    }
    # refit a GLM for residuals (use NB-fixed-alpha or Poisson) so resid_deviance exists
    y = df["P"].to_numpy(float); X = sm.add_constant(np.log(df["S"].to_numpy(float)))
    glm = (sm.GLM(y, X, family=sm.families.Poisson()).fit() if alpha_boot is None
           else fit_nb_glm(y, X, alpha_boot))
    save_residuals(df, glm, defname)
    return result


def main():
    name = "Phase 3 — scaling regressions"
    C.phase_start(name)
    C.set_seed()
    results = {"seed": C.SEED, "bootstrap_n": BOOT_N, "definitions": {}}
    for defname, path in UNIT_FILES.items():
        if not C.exists_nonempty(path):
            C.warn(f"[{defname}] units file missing ({path.name}); skipping")
            results["definitions"][defname] = {"status": "missing"}
            continue
        try:
            results["definitions"][defname] = analyse(defname, path)
            b = results["definitions"][defname]["model_A_primary"]
            C.log(f"[{defname}] beta={b['beta']} CI{b['ci95']} -> {b['verdict']}")
        except Exception as e:  # noqa: BLE001
            import traceback
            C.warn(f"[{defname}] regression failed: {e}\n{traceback.format_exc()}")
            results["definitions"][defname] = {"status": "failed", "error": str(e)}
    C.write_json(C.OUTPUT / "results.json", results)
    C.phase_end(name)


if __name__ == "__main__":
    main()

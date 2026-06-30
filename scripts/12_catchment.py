"""Phase 12 — catchment scaling with a travel-distance sweep.

Containment (stations inside polygon i) decouples from reality once the unit is
smaller than a fuel catchment: 91% of DeSO above 50 ppl/ha contain zero stations,
yet their nearest station sits ~0.7 km away in a neighbouring unit. So instead of
containment we ASSIGN stations to a population by a travel-distance cut-off d:
anchor each DeSO at its population point, and count the stations within d km.

For each d we fit a count GLM (NB2/Poisson, log link) two ways:

  beta_unit(d):   log E[P_acc] = a + beta * log(S_i)
      P_acc = stations within d of unit i, S_i = that unit's own population.
      The literal "fine population unit vs the stations reachable from it".

  beta_catch(d):  log E[P_acc] = a + beta * log(N_d)
      N_d = population within d of unit i (catchment population, DeSO anchors as
      point masses). Both sides are catchment aggregates — the cleaner
      economies-of-scale object: at travel scale d, is fuel provision sub-linear?

The shape of beta(d) is the result. If beta rises toward 1 as d grows, the
sub-linearity was largely a boundary-leakage artifact; if it stays < 1, there is a
real economy of scale in fuel-retail provision even after allowing cross-boundary
access. Reference lines: containment single-beta (Phase 10) and kommun-beta.

CAVEAT printed alongside: catchments overlap (shared stations) -> residual spatial
autocorrelation makes the reported SEs optimistic; and resident population still
ignores through-traffic, so corridor stations look like excess provision.

Writes output/catchment_summary.json, output/catchment.md, charts/12_*.png.
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
DESO = C.CACHE / "units_deso.geojson"
STATIONS = C.CACHE / "stations.gpkg"
DENS_SUMMARY = C.OUTPUT / "density_summary.json"
D_KM = [2, 5, 10, 15, 20, 30, 50]          # travel-distance sweep
EXAMPLE_D = 10                              # km, for the scatter panel


def load():
    g = gpd.read_file(DESO).to_crs(CRS)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    pop = C.read_json(C.CACHE / "pop_deso_regso.json")["deso"]
    g["S"] = g["desokod"].astype(str).map(pop).astype(float)
    g = g[g["S"].notna() & (g["S"] > 0)].copy()
    pt = g.geometry.representative_point()
    g["x"] = pt.x.to_numpy(); g["y"] = pt.y.to_numpy()
    st = gpd.read_file(STATIONS).to_crs(CRS)
    st = st[st.geometry.notna() & ~st.geometry.is_empty].copy()
    return g.reset_index(drop=True), st


def fit_loglog(y, x):
    """Count GLM log E[y] = a + beta*log(x); NB2 if overdispersed else Poisson."""
    X = sm.add_constant(np.log(x))
    pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    disp = float(pois.pearson_chi2 / pois.df_resid)
    if disp > 1.2:
        try:
            res = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=200)
            fam = "NB2"
        except Exception:  # noqa: BLE001
            res, fam = pois, "Poisson(NB-failed)"
    else:
        res, fam = pois, "Poisson"
    beta = float(res.params[1]); se = float(res.bse[1])
    z = (beta - 1.0) / se
    return {"family": fam, "dispersion": round(disp, 2),
            "beta": round(beta, 4), "se": round(se, 4),
            "ci95": [round(beta - 1.96 * se, 4), round(beta + 1.96 * se, 4)],
            "p_vs1": float(2 * stats.norm.sf(abs(z))),
            "verdict": ("~linear" if abs(z) < 1.96 else "sublinear" if beta < 1 else "superlinear")}


def sweep(g, st):
    anchors = np.c_[g["x"].to_numpy(), g["y"].to_numpy()]
    S = g["S"].to_numpy(float)
    st_xy = np.c_[st.geometry.x.to_numpy(), st.geometry.y.to_numpy()]
    tree_st = cKDTree(st_xy)
    tree_an = cKDTree(anchors)

    rows = []
    cache_pacc = {}
    cache_ncatch = {}
    for d in D_KM:
        r = d * 1000.0
        # stations within d of each anchor
        p_acc = tree_st.query_ball_point(anchors, r=r, return_length=True).astype(float)
        # catchment population within d (DeSO anchors as point masses, incl self)
        nbr = tree_an.query_ball_point(anchors, r=r)
        n_catch = np.array([S[ix].sum() for ix in nbr], float)
        cache_pacc[d] = p_acc; cache_ncatch[d] = n_catch
        bu = fit_loglog(p_acc, S)
        bc = fit_loglog(p_acc, n_catch)
        rows.append({
            "d_km": d,
            "median_P_acc": float(np.median(p_acc)),
            "frac_zero_P_acc": round(float((p_acc == 0).mean()), 3),
            "median_catchment_pop": int(np.median(n_catch)),
            "beta_unit": bu, "beta_catch": bc,
        })
        C.log(f"d={d:>3} km | P_acc med={np.median(p_acc):.0f} zero={int((p_acc==0).sum())} "
              f"| beta_unit={bu['beta']:.3f} {bu['ci95']} | beta_catch={bc['beta']:.3f} {bc['ci95']}")
    return rows, cache_pacc, cache_ncatch


def refs():
    out = {"containment_single_beta": None, "kommun_beta": None}
    s = C.read_json(DENS_SUMMARY)
    if s:
        out["containment_single_beta"] = s["units"]["deso"]["single_power_law"]["beta"]
    rj = C.read_json(C.OUTPUT / "results.json")
    if rj:
        out["kommun_beta"] = rj["definitions"]["kommun"]["model_A_primary"]["beta"]
    return out


def chart(rows, ref, g, pacc, ncatch):
    d = np.array([r["d_km"] for r in rows], float)
    bu = np.array([r["beta_unit"]["beta"] for r in rows])
    bul = np.array([r["beta_unit"]["ci95"][0] for r in rows])
    buh = np.array([r["beta_unit"]["ci95"][1] for r in rows])
    bc = np.array([r["beta_catch"]["beta"] for r in rows])
    bcl = np.array([r["beta_catch"]["ci95"][0] for r in rows])
    bch = np.array([r["beta_catch"]["ci95"][1] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.fill_between(d, bcl, bch, color="#2b6cb0", alpha=0.15)
    ax.plot(d, bc, "o-", color="#2b6cb0", lw=2, label="β_catch  (stations vs catchment pop within d)")
    ax.fill_between(d, bul, buh, color="#c1440e", alpha=0.12)
    ax.plot(d, bu, "s--", color="#c1440e", lw=2, label="β_unit  (stations within d vs DeSO own pop)")
    ax.axhline(1.0, color="k", ls=":", lw=1.2, label="linear (β=1)")
    if ref.get("containment_single_beta"):
        ax.axhline(ref["containment_single_beta"], color="#3a7d44", ls="-.", lw=1.2,
                   label=f"DeSO containment β={ref['containment_single_beta']:.2f}")
    if ref.get("kommun_beta"):
        ax.axhline(ref["kommun_beta"], color="#6b2d5c", ls="-.", lw=1.2,
                   label=f"kommun β={ref['kommun_beta']:.2f}")
    ax.set_xlabel("assumed fuel-travel distance  d  (km)")
    ax.set_ylabel("scaling exponent β")
    ax.set_title("β(d): does provision look linear once you allow travel?")
    ax.legend(fontsize=8.5, loc="best"); ax.grid(True, alpha=0.25)

    # example catchment-vs-catchment scatter at EXAMPLE_D
    ax = axes[1]
    N = ncatch[EXAMPLE_D]; Y = pacc[EXAMPLE_D]
    m = Y > 0
    ax.scatter(N[m], Y[m], s=8, alpha=0.18, color="#5a6b7b", edgecolors="none")
    bc_e = next(r["beta_catch"] for r in rows if r["d_km"] == EXAMPLE_D)
    xs = np.linspace(np.log(N[m].min()), np.log(N[m].max()), 100)
    X = sm.add_constant(np.log(N)); a0 = float(sm.GLM(Y, X, family=sm.families.Poisson()).fit().params[0])
    ax.plot(np.exp(xs), np.exp(a0 + bc_e["beta"] * xs), "k-", lw=2.2,
            label=f"β_catch = {bc_e['beta']:.2f}  CI{bc_e['ci95']}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"population within {EXAMPLE_D} km of unit")
    ax.set_ylabel(f"stations within {EXAMPLE_D} km of unit")
    ax.set_title(f"Catchment vs catchment at d = {EXAMPLE_D} km (DeSO anchors)")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2)

    fig.suptitle("Catchment scaling: assigning stations to populations by travel distance, "
                 "not administrative containment", fontsize=12.5)
    fig.tight_layout()
    out = C.CHARTS / "12_catchment_beta_sweep.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    C.log(f"[OK] {out.name}")


def write_md(rows, ref):
    L = ["# Catchment scaling — travel-distance sweep", "",
         "Stations assigned to each DeSO population anchor by a travel-distance cut-off "
         "d (not polygon containment). Count GLM, log link.", "",
         "| d (km) | median P_acc | β_unit [CI] | β_catch [CI] |",
         "|--:|--:|---|---|"]
    for r in rows:
        bu = r["beta_unit"]; bc = r["beta_catch"]
        L.append(f"| {r['d_km']} | {r['median_P_acc']:.0f} | "
                 f"{bu['beta']:.3f} {bu['ci95']} | {bc['beta']:.3f} {bc['ci95']} |")
    L += ["",
          f"Reference: DeSO containment single-β = {ref.get('containment_single_beta')}, "
          f"kommun β = {ref.get('kommun_beta')}.", "",
          "**Reading β_catch(d):** rising toward 1 ⇒ apparent sub-linearity was largely "
          "boundary leakage; staying < 1 ⇒ a real economy of scale in fuel provision at "
          "that travel scale.", "",
          "**Caveats.** Overlapping catchments share stations ⇒ residual spatial "
          "autocorrelation makes these SEs optimistic. Resident population ignores "
          "through-traffic, so corridor stations still read as excess provision. "
          "Catchment population uses DeSO anchors as point masses (Euclidean, not "
          "road-network distance). Border/coast anchors have truncated catchments."]
    p = C.OUTPUT / "catchment.md"
    p.write_text("\n".join(L))
    C.log(f"[OK] wrote {p.name}")


def main():
    name = "Phase 12 — catchment scaling (distance sweep)"
    C.phase_start(name)
    C.set_seed()
    for need in (DESO, STATIONS):
        if not C.exists_nonempty(need):
            C.warn(f"missing {Path(need).name} — run Phase 10 first; skipping")
            C.phase_end(name); return
    g, st = load()
    C.log(f"loaded {len(g)} DeSO anchors, {len(st)} stations")
    rows, pacc, ncatch = sweep(g, st)
    ref = refs()
    chart(rows, ref, g, pacc, ncatch)
    write_md(rows, ref)
    C.write_json(C.OUTPUT / "catchment_summary.json",
                 {"d_km_sweep": D_KM, "example_d_km": EXAMPLE_D,
                  "references": ref, "rows": rows,
                  "caveats": ["overlapping catchments -> optimistic SEs (spatial autocorr)",
                              "resident pop ignores through-traffic / corridor demand",
                              "Euclidean point-mass catchment population, not road-network",
                              "border/coast anchors have truncated catchments"]})
    C.phase_end(name)


if __name__ == "__main__":
    main()

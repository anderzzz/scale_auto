"""Phase 14 — radial profiles: is the urban-core station "donut" universal?

Two figures:

  charts/14_radial_cities.png   (DESCRIPTIVE, no aggregation)
      Four cities of distinct character — Stockholm, Malmö, Norrköping, Sundsvall.
      For each, vs distance from the city centre (tätort representative point):
        * population density (ppl/km², filled, left axis) — peaks at the core;
        * station density (stations/km², line, right axis) — if the donut is real,
          this dips at r=0 and peaks in a RING before decaying.
      Lets you see each city's own shape rather than a smoothed average.

  charts/14_radial_stacked.png  (AGGREGATE, size-normalised)
      Top N cores. To remove the absolute-radius problem (Stockholm's ring is far
      out, Sundsvall's is close in), each city's residents are split into equal-
      population shells from the centre outward, so the x-axis is CUMULATIVE
      POPULATION ENCLOSED (fraction), not kilometres. y = stations per 1,000
      residents in that shell. Mean + inter-quartile band across cities tests
      whether a core dip is universal or just Stockholm.

Reads cache/tatorter_2023.geojson (cores), cache/units_deso.geojson (population),
cache/stations.gpkg.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

CRS = C.CONFIG["national_crs"]
DPI = 150
TAT = C.CACHE / "tatorter_2023.geojson"
DESO = C.CACHE / "units_deso.geojson"
STATIONS = C.CACHE / "stations.gpkg"

NAMED = ["Stockholm", "Malmö", "Norrköping", "Sundsvall"]
CHARACTER = {"Stockholm": "capital metro", "Malmö": "flat coastal city",
             "Norrköping": "mid-size industrial town", "Sundsvall": "small city in a rural kommun"}
N_CORES = 15            # for the stacked aggregate
BIN_KM = 1.5            # individual-profile annulus width
R_MAX_KM = 21.0         # individual-profile extent
CAP_KM = 30.0           # aggregate: ignore DeSO beyond this from centre
N_SHELLS = 10           # aggregate: equal-population shells


def load():
    tat = gpd.read_file(TAT).to_crs(CRS)
    tat["bef"] = tat["bef"].astype(float)
    g = gpd.read_file(DESO).to_crs(CRS)
    pop = C.read_json(C.CACHE / "pop_deso_regso.json")["deso"]
    g["S"] = g["desokod"].astype(str).map(pop).astype(float)
    g = g[g["S"].notna() & (g["S"] > 0)].copy()
    cp = g.geometry.representative_point()
    g["x"] = cp.x.to_numpy(); g["y"] = cp.y.to_numpy()
    g["area_km2"] = g.geometry.area / 1e6
    st = gpd.read_file(STATIONS).to_crs(CRS)
    st = st[st.geometry.notna() & ~st.geometry.is_empty].copy()
    st_xy = np.c_[st.geometry.x.to_numpy(), st.geometry.y.to_numpy()]
    return tat, g, st_xy


def centre_of(tat, name):
    sub = tat[tat["tatort"].astype(str).str.fullmatch(name, case=False, na=False)]
    if not len(sub):
        sub = tat[tat["tatort"].astype(str).str.contains(name, case=False, na=False)]
    row = sub.sort_values("bef", ascending=False).iloc[0]
    p = row.geometry.representative_point()
    return float(p.x), float(p.y), float(row["bef"]), row["tatort"]


def dists(g, st_xy, cx, cy):
    dd = np.hypot(g["x"].to_numpy() - cx, g["y"].to_numpy() - cy) / 1000.0
    ds = np.hypot(st_xy[:, 0] - cx, st_xy[:, 1] - cy) / 1000.0
    return dd, ds


# ----------------------------------------------------------------- individual
def city_panel(ax, g, st_xy, name):
    cx, cy, bef, label = centre_of_cache[name]
    dd, ds = dists(g, st_xy, cx, cy)
    S = g["S"].to_numpy(float)
    edges = np.arange(0, R_MAX_KM + BIN_KM, BIN_KM)
    mid = 0.5 * (edges[:-1] + edges[1:])
    ann_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)   # km²
    pop_d, stn_d = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_d = (dd >= lo) & (dd < hi)
        in_s = (ds >= lo) & (ds < hi)
        pop_d.append(S[in_d].sum() / ann_area[i])
        stn_d.append(in_s.sum() / ann_area[i])
    pop_d = np.array(pop_d); stn_d = np.array(stn_d)

    ax.fill_between(mid, pop_d, color="#9bb8d3", alpha=0.55, label="population density")
    ax.set_ylabel("population density (ppl/km²)", color="#2b5d8a", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#2b5d8a")
    ax2 = ax.twinx()
    ax2.plot(mid, stn_d, "-o", color="#c1440e", lw=2, ms=4, label="station density")
    ax2.set_ylabel("station density (stations/km²)", color="#c1440e", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#c1440e")
    # mark the station-density peak (the "ring")
    if stn_d.max() > 0:
        rpk = mid[int(np.argmax(stn_d))]
        ax2.axvline(rpk, color="#c1440e", ls=":", lw=1.2)
        ax2.annotate(f"ring ≈ {rpk:.0f} km", (rpk, stn_d.max()), fontsize=8,
                     color="#7a2606", xytext=(4, -2), textcoords="offset points")
    ax.set_xlabel("distance from centre (km)", fontsize=9)
    ax.set_title(f"{name} — {CHARACTER[name]}\n(tätort pop {bef/1000:.0f}k)", fontsize=10.5)
    ax.set_xlim(0, R_MAX_KM)


def figure_cities(g, st_xy):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, name in zip(axes.ravel(), NAMED):
        city_panel(ax, g, st_xy, name)
    fig.suptitle("Radial profiles by city: where people live (blue) vs where stations sit "
                 "(orange) — the core dip & station ring", fontsize=13)
    fig.tight_layout()
    out = C.CHARTS / "14_radial_cities.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    C.log(f"[OK] {out.name}")
    return out.name


# ------------------------------------------------------------------- aggregate
def shell_profile(g, st_xy, cx, cy):
    """Equal-population shells from centre; stations per 1,000 residents per shell."""
    dd, ds = dists(g, st_xy, cx, cy)
    S = g["S"].to_numpy(float)
    keep = dd <= CAP_KM
    order = np.argsort(dd[keep])
    d_sorted = dd[keep][order]; s_sorted = S[keep][order]
    cum = np.cumsum(s_sorted); total = cum[-1]
    if total <= 0:
        return None
    # equal-population shell boundaries (radii)
    targets = np.linspace(0, total, N_SHELLS + 1)[1:]
    r_edges = [0.0]
    for t in targets:
        idx = np.searchsorted(cum, t)
        r_edges.append(float(d_sorted[min(idx, len(d_sorted) - 1)]))
    per1000 = []
    for i in range(N_SHELLS):
        lo, hi = r_edges[i], r_edges[i + 1]
        shell_pop = s_sorted[(d_sorted >= lo) & (d_sorted < hi)].sum()
        shell_stn = int(((ds >= lo) & (ds < hi)).sum())
        per1000.append(shell_stn / (shell_pop / 1000.0) if shell_pop > 0 else np.nan)
    return np.array(per1000)


def figure_stacked(tat, g, st_xy):
    cores = tat.sort_values("bef", ascending=False).head(N_CORES)
    xfrac = (np.arange(N_SHELLS) + 0.5) / N_SHELLS       # cumulative-pop fraction
    curves, names = [], []
    for _, row in cores.iterrows():
        p = row.geometry.representative_point()
        prof = shell_profile(g, st_xy, float(p.x), float(p.y))
        if prof is not None:
            curves.append(prof); names.append(str(row["tatort"]))
    M = np.vstack(curves)
    mean = np.nanmean(M, axis=0)
    q1 = np.nanpercentile(M, 25, axis=0); q3 = np.nanpercentile(M, 75, axis=0)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for c, nm in zip(curves, names):
        ax.plot(xfrac, c, color="#bcc6d0", lw=0.9, alpha=0.7, zorder=1)
    hl = {"Stockholm": "#c1440e", "Malmö": "#2b6cb0", "Norrköping": "#6b2d5c",
          "Sundsvall": "#3a7d44"}
    for c, nm in zip(curves, names):
        for key, col in hl.items():
            if nm.lower().startswith(key.lower()):
                ax.plot(xfrac, c, color=col, lw=1.8, alpha=0.95, zorder=3, label=nm)
    ax.fill_between(xfrac, q1, q3, color="#2b6cb0", alpha=0.15, zorder=2,
                    label="inter-quartile band")
    ax.plot(xfrac, mean, "k-", lw=3, zorder=4, label=f"mean of {len(curves)} cores")
    ax.set_xlabel("cumulative population enclosed  (core → edge, fraction of city)")
    ax.set_ylabel("stations per 1,000 residents in shell")
    ax.set_title(f"Stacked radial profile, {len(curves)} largest cores: is the core dip "
                 "universal?\n(equal-population shells normalise away city size)")
    ax.legend(fontsize=8.5, ncol=2); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = C.CHARTS / "14_radial_stacked.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    C.log(f"[OK] {out.name} | cores: {', '.join(names[:6])} …")
    return out.name


centre_of_cache = {}


def main():
    name = "Phase 14 — radial profiles"
    C.phase_start(name)
    C.set_seed()
    for need in (TAT, DESO, STATIONS):
        if not C.exists_nonempty(need):
            C.warn(f"missing {Path(need).name}; skipping")
            C.phase_end(name); return
    tat, g, st_xy = load()
    for nm in NAMED:
        centre_of_cache[nm] = centre_of(tat, nm)
        C.log(f"centre[{nm}] = {centre_of_cache[nm][3]} (pop {centre_of_cache[nm][2]/1000:.0f}k)")
    figure_cities(g, st_xy)
    figure_stacked(tat, g, st_xy)
    C.phase_end(name)


if __name__ == "__main__":
    main()

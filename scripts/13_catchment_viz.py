"""Phase 13 — visual edification for the catchment method (Phase 12).

Two exploratory figures, appended to the infographic appendix:

  charts/13_catchment_method.png
      HOW a unit's two numbers are formed, on a dense urban DeSO and a huge rural
      DeSO. Shows the population anchor (representative point), the d=10 km RADIUS
      used in Phase 12 (centre-out), AND the polygon+padding alternative (buffer the
      DeSO outward by d). Stations are coloured by which rule catches them. Makes
      the centre-vs-buffer divergence (negligible for compact urban units, large for
      sprawling rural ones) and the deliberate over-count visible.

  charts/13_catchment_residuals.png
      WHO the outliers are. Catchment scaling at d=10 km (stations vs catchment
      population), then the national map of residuals: red = far more reachable
      stations than the resident catchment predicts — the traveller/corridor/tourist
      signal the resident-population model cannot absorb. Top outliers labelled by
      kommun.

Reads cache/units_deso.geojson, cache/stations.gpkg, cache/units_kommun.gpkg.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

CRS = C.CONFIG["national_crs"]
DPI = 150
DESO = C.CACHE / "units_deso.geojson"
STATIONS = C.CACHE / "stations.gpkg"
KOMMUN = C.CACHE / "units_kommun.gpkg"
D_KM = 10
URBAN_EX = "2281C1100"   # dense Sundsvall DeSO (2,290 ppl, 0.30 km²)
RURAL_EX = "2281A0090"   # huge rural Sundsvall DeSO (1,093 ppl, 1,016 km²)


def load():
    g = gpd.read_file(DESO).to_crs(CRS)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    pop = C.read_json(C.CACHE / "pop_deso_regso.json")["deso"]
    g["S"] = g["desokod"].astype(str).map(pop).astype(float)
    g = g[g["S"].notna() & (g["S"] > 0)].copy()
    pt = g.geometry.representative_point()
    g["x"] = pt.x.to_numpy(); g["y"] = pt.y.to_numpy()
    g["area_km2"] = g.geometry.area / 1e6
    st = gpd.read_file(STATIONS).to_crs(CRS)
    st = st[st.geometry.notna() & ~st.geometry.is_empty].copy()
    return g.reset_index(drop=True), st


def kommun_names():
    k = gpd.read_file(KOMMUN)
    k["kk"] = k["unit_id"].astype(str).str.replace("SE_", "", regex=False)
    return dict(zip(k["kk"], k["name"]))


# --------------------------------------------------------------------- schematic
def method_panel(ax, g, st, code, title):
    row = g[g["desokod"].astype(str) == code].iloc[0]
    poly = row.geometry
    ax_pt = (row["x"], row["y"])
    r = D_KM * 1000.0
    buf = poly.buffer(r)
    # stations caught by each rule
    d2 = (st.geometry.x - ax_pt[0]) ** 2 + (st.geometry.y - ax_pt[1]) ** 2
    in_circle = d2 <= r ** 2
    in_buffer = st.within(buf)
    n_circle = int(in_circle.sum()); n_buffer = int(in_buffer.sum())

    gpd.GeoSeries([buf], crs=CRS).plot(ax=ax, facecolor="#2b6cb0", alpha=0.07,
                                       edgecolor="#2b6cb0", linewidth=1.4, linestyle="--")
    gpd.GeoSeries([poly], crs=CRS).plot(ax=ax, facecolor="#3a7d44", alpha=0.30,
                                        edgecolor="#1b3a22", linewidth=1.2)
    ax.add_patch(Circle(ax_pt, r, fill=False, edgecolor="#c1440e", lw=1.8, linestyle="-"))
    # stations: outside both (grey), buffer-only (blue), circle (orange on top)
    st.plot(ax=ax, color="#bbbbbb", markersize=6, zorder=3)
    st[in_buffer].plot(ax=ax, color="#2b6cb0", markersize=14, zorder=4)
    st[in_circle].plot(ax=ax, color="#c1440e", markersize=18, zorder=5, edgecolor="k", linewidth=0.3)
    ax.plot(*ax_pt, marker="*", ms=16, color="black", zorder=6)
    minx, miny, maxx, maxy = buf.bounds
    m = 0.06 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - m, maxx + m); ax.set_ylim(miny - m, maxy + m)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(f"{title}\nS={int(row['S']):,} ppl, area={row['area_km2']:.0f} km²\n"
                 f"★ anchor · ◯ centre+{D_KM}km = {n_circle} stn · "
                 f"▢ polygon+{D_KM}km = {n_buffer} stn", fontsize=9.5)


def schematic(g, st):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))
    method_panel(axes[0], g, st, URBAN_EX, "Dense urban DeSO (centre ≈ buffer)")
    method_panel(axes[1], g, st, RURAL_EX, "Huge rural DeSO (centre ≪ buffer)")
    # shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="*", color="k", ls="", ms=12, label="population anchor (rep. point)"),
        Line2D([], [], color="#c1440e", lw=1.8, label=f"centre + {D_KM} km radius (Phase 12)"),
        Line2D([], [], color="#2b6cb0", lw=1.4, ls="--", label=f"polygon + {D_KM} km padding (alt.)"),
        Line2D([], [], marker="o", color="#c1440e", ls="", label="station in radius"),
        Line2D([], [], marker="o", color="#2b6cb0", ls="", label="station in padding only"),
        Line2D([], [], marker="o", color="#bbbbbb", ls="", label="station outside"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("How each DeSO gets its two numbers: population (the unit) vs stations "
                 "reachable within travel distance d", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = C.CHARTS / "13_catchment_method.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    C.log(f"[OK] {out.name}")
    return out.name


# --------------------------------------------------------------------- residuals
def residual_map(g, st, knames):
    anchors = np.c_[g["x"].to_numpy(), g["y"].to_numpy()]
    S = g["S"].to_numpy(float)
    tree_st = cKDTree(np.c_[st.geometry.x.to_numpy(), st.geometry.y.to_numpy()])
    tree_an = cKDTree(anchors)
    r = D_KM * 1000.0
    p_acc = tree_st.query_ball_point(anchors, r=r, return_length=True).astype(float)
    n_catch = np.array([S[ix].sum() for ix in tree_an.query_ball_point(anchors, r=r)], float)
    g = g.assign(P_acc=p_acc, N_catch=n_catch)
    g = g[g["P_acc"] > 0].copy()

    X = sm.add_constant(np.log(g["N_catch"].to_numpy()))
    res = sm.GLM(g["P_acc"].to_numpy(float), X, family=sm.families.Poisson()).fit()
    g["resid"] = np.asarray(res.resid_deviance, float)
    g["kommun"] = g["kommunkod"].astype(str).map(knames).fillna(g["kommunkod"].astype(str))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.2),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    # (A) national residual map
    ax = axes[0]
    norm = TwoSlopeNorm(vmin=np.quantile(g["resid"], 0.02), vcenter=0,
                        vmax=np.quantile(g["resid"], 0.98))
    g.plot(ax=ax, column="resid", cmap="RdBu_r", norm=norm, markersize=0,
           edgecolor="none", legend=True,
           legend_kwds={"label": "deviance residual (red = more stations than catchment predicts)",
                        "shrink": 0.5})
    ax.set_axis_off(); ax.set_aspect("equal")
    ax.set_title(f"Catchment residuals, d={D_KM} km\nred = supply-rich vs resident demand", fontsize=10)
    # label top positive outliers (dedup by kommun)
    top = g.sort_values("resid", ascending=False)
    seen, picks = set(), []
    for _, rrow in top.iterrows():
        if rrow["kommun"] in seen:
            continue
        seen.add(rrow["kommun"]); picks.append(rrow)
        if len(picks) >= 8:
            break
    for rrow in picks:
        ax.annotate(rrow["kommun"], (rrow["x"], rrow["y"]), fontsize=7.5,
                    color="#7a1010", weight="bold")

    # (B) scatter with outliers highlighted
    ax = axes[1]
    ax.scatter(g["N_catch"], g["P_acc"], s=8, alpha=0.18, color="#5a6b7b", edgecolors="none")
    xs = np.linspace(np.log(g["N_catch"].min()), np.log(g["N_catch"].max()), 100)
    ax.plot(np.exp(xs), np.exp(res.params[0] + res.params[1] * xs), "k-", lw=2,
            label=f"β = {res.params[1]:.2f}")
    pk = g[g["kommun"].isin([p["kommun"] for p in picks])]
    ax.scatter(pk["N_catch"], pk["P_acc"], s=30, color="#c1440e", edgecolors="k",
               linewidths=0.4, zorder=5, label="top supply-rich outliers")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"catchment population  (Σ residents of all DeSO within {D_KM} km)")
    ax.set_ylabel(f"stations within {D_KM} km of centre")
    ax.set_title("Outliers sit ABOVE the line:\nmore stations than residents alone explain", fontsize=10)
    ax.legend(fontsize=8.5); ax.grid(True, which="both", alpha=0.2)

    fig.suptitle("Travellers don't live there: catchment outliers are corridor & "
                 "destination units", fontsize=12.5)
    fig.tight_layout()
    out = C.CHARTS / "13_catchment_residuals.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    C.log(f"[OK] {out.name} | top outliers: {', '.join(p['kommun'] for p in picks)}")
    return out.name, [p["kommun"] for p in picks]


# ------------------------------------------------------------ two-scatter clarifier
def two_scatter(g, st):
    """Same y (stations within d of centre), two different x's — to kill the axis
    confusion: DeSO OWN population (≤3.6k) vs CATCHMENT population within d (~1.2M)."""
    anchors = np.c_[g["x"].to_numpy(), g["y"].to_numpy()]
    S = g["S"].to_numpy(float)
    tree_st = cKDTree(np.c_[st.geometry.x.to_numpy(), st.geometry.y.to_numpy()])
    tree_an = cKDTree(anchors)
    r = D_KM * 1000.0
    Pacc = tree_st.query_ball_point(anchors, r=r, return_length=True).astype(float)
    Ncatch = np.array([S[ix].sum() for ix in tree_an.query_ball_point(anchors, r=r)], float)

    def fit(x, y):
        m = (x > 0) & (y > 0)
        X = sm.add_constant(np.log(x[m]))
        res = sm.GLM(y[m], X, family=sm.families.Poisson()).fit()
        return float(res.params[0]), float(res.params[1]), m

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    specs = [
        (axes[0], S, "#c1440e",
         "What you pictured:  β_unit",
         f"DeSO OWN population  (one block, max {int(S.max()):,})",
         "stations within 10 km serve far more than this block\n→ scale mismatch on x"),
        (axes[1], Ncatch, "#2b6cb0",
         "What the plot showed:  β_catch",
         f"CATCHMENT population  (Σ all DeSO ≤10 km, max {int(Ncatch.max()):,})",
         "central Stockholm: 671 DeSO inside 10 km\n→ overlapping metro catchments reach ~1.2M"),
    ]
    for ax, x, col, ttl, xlab, note in specs:
        a0, b1, m = fit(x, Pacc)
        ax.scatter(x[m], Pacc[m], s=8, alpha=0.16, color="#5a6b7b", edgecolors="none")
        xs = np.linspace(np.log(x[m].min()), np.log(x[m].max()), 100)
        ax.plot(np.exp(xs), np.exp(a0 + b1 * xs), color=col, lw=2.4, label=f"β = {b1:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlab); ax.set_title(ttl, fontsize=11)
        ax.legend(fontsize=10, loc="upper left"); ax.grid(True, which="both", alpha=0.2)
        ax.text(0.97, 0.04, note, transform=ax.transAxes, fontsize=8.5, color="#444",
                ha="right", va="bottom", style="italic")
    axes[0].set_ylabel("stations within 10 km of DeSO centre  (SAME on both panels)")
    fig.suptitle("Same dependent variable, two independent variables — why one x tops out "
                 "at 3.6k and the other at 1.2M", fontsize=12.5)
    fig.tight_layout()
    out = C.CHARTS / "13_unit_vs_catchment_x.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    C.log(f"[OK] {out.name}")
    return out.name


# ----------------------------------------------------------------- infographic patch
def append_infographic(images):
    import base64
    html = C.OUTPUT / "infographic.html"
    if not C.exists_nonempty(html):
        C.warn("infographic.html missing — skipping append")
        return
    txt = html.read_text()
    marker = "<!-- catchment-viz-appendix -->"
    if marker in txt:
        txt = txt[:txt.index(marker)].rstrip()

    def embed(name):
        b64 = base64.b64encode((C.CHARTS / name).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;border-radius:8px;margin:18px 0;"/>'

    blocks = "\n".join(
        f'<figure style="margin:0 0 30px 0;"><figcaption style="font:14px/1.5 -apple-system,sans-serif;'
        f'color:#bcd;margin-bottom:6px;">{cap}</figcaption>{embed(img)}</figure>'
        for img, cap in images)
    section = f"""{marker}
<section class="wrap" style="background:#0d1420;color:#e8eef6;padding:60px 24px;">
  <div style="max-width:1100px;margin:0 auto;">
    <h2 style="font:600 30px/1.2 -apple-system,sans-serif;color:#fff;">
      Appendix · Catchment method (how & who)</h2>
    <p style="font:17px/1.6 -apple-system,sans-serif;color:#aebfd0;max-width:780px;">
      Each DeSO is a population unit (people, area, density from SCB). We anchor it at
      its centre and count stations within a travel distance d — deliberately
      over-counting, because one station really does serve many neighbouring units.
      For compact urban units a centre-radius and a polygon-plus-padding agree; for
      sprawling rural units they diverge. And resident population can't see
      travellers, so destination and corridor units sit above the line.</p>
    {blocks}
  </div>
</section>
"""
    low = txt.lower()
    if "</body>" in low:
        i = low.rindex("</body>")
        txt = txt[:i] + section + "\n" + txt[i:]
    else:
        txt = txt + "\n" + section
    html.write_text(txt)
    C.log(f"[OK] appended catchment appendix to {html.name}")


def main():
    name = "Phase 13 — catchment method viz"
    C.phase_start(name)
    C.set_seed()
    for need in (DESO, STATIONS, KOMMUN):
        if not C.exists_nonempty(need):
            C.warn(f"missing {Path(need).name}; skipping")
            C.phase_end(name); return
    g, st = load()
    knames = kommun_names()
    img1 = schematic(g, st)
    img0 = two_scatter(g, st)
    img2, outliers = residual_map(g, st, knames)
    append_infographic([
        (img1, "How each unit's two numbers form: anchor + radius (used) vs polygon + "
               "padding (alternative). Centre≈buffer when compact, centre≪buffer when rural."),
        (img0, "Axis clarifier: the catchment scatter's x is the SUM of all DeSO residents "
               "within 10 km (max ~1.2M in central Stockholm), not a single DeSO's population "
               "(max 3.6k). Same y on both panels."),
        (img2, "Who breaks the model: supply-rich catchment outliers (red) — corridor and "
               "destination kommuner where stations serve travellers, not residents."),
    ])
    C.phase_end(name)


if __name__ == "__main__":
    main()

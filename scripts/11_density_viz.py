"""Phase 11 — visual intuition for density scaling (maps + breakpoint charts).

Companion to Phase 10. Produces, for grasping *how the breakpoint looks*:

  charts/11_granularity_<kommun>.png
      The same administrative area as ONE kommun polygon (a single density value)
      vs its DeSO mosaic — the Sundsvall/Örnsköldsvik "city hidden in a rural
      kommun" made visible. DeSO shaded by population density, then re-coloured
      rural/urban by the Phase-10 breakpoint; stations overlaid.

  charts/11_breakpoint_curves.png
      Three ways to see the linear→sublinear bend on DeSO:
        (A) log-log station-density vs population-density with the two-regime fit,
            example-kommun DeSOs highlighted;
        (B) the LOCAL slope (finite-difference of density-binned aggregates) falling
            from ≈1 (rural) to ≈0.26 (urban) across the breakpoint;
        (C) stations per 1,000 people vs density — flat, then saturating.

Reads cache/units_deso.geojson, cache/stations.gpkg, output/density_summary.json.
First-pass exploratory panels — not cleaned for publication.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

CRS = C.CONFIG["national_crs"]
DPI = 150
DESO = C.CACHE / "units_deso.geojson"
STATIONS = C.CACHE / "stations.gpkg"
SUMMARY = C.OUTPUT / "density_summary.json"

EXAMPLES = {"2281": "Sundsvall", "2284": "Örnsköldsvik"}
RURAL = "#3a7d44"   # below breakpoint
URBAN = "#c1440e"   # above breakpoint


def load():
    g = gpd.read_file(DESO).to_crs(CRS)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    # population + counts as in Phase 10
    pop = C.read_json(C.CACHE / "pop_deso_regso.json")["deso"]
    g["S"] = g["desokod"].astype(str).map(pop).astype("float")
    g = g[g["S"].notna() & (g["S"] > 0)].copy()
    st = gpd.read_file(STATIONS).to_crs(CRS)
    st = st[st.geometry.notna() & ~st.geometry.is_empty].copy()
    st["sidx"] = np.arange(len(st))
    j = gpd.sjoin(st[["sidx", "geometry"]], g[["desokod", "geometry"]],
                  how="inner", predicate="within")
    cnt = j.groupby("desokod").size().rename("P")
    g = g.merge(cnt, on="desokod", how="left")
    g["P"] = g["P"].fillna(0).astype(int)
    g["area_ha"] = g.geometry.area / 1e4
    g = g[g["area_ha"] > 0].copy()
    g["dens"] = g["S"] / g["area_ha"]
    g["kommunkod"] = g["kommunkod"].astype(str)
    return g, st


def breakpoint_params():
    s = C.read_json(SUMMARY)["units"]["deso"]
    seg = s["segmented"]
    return (seg["breakpoint_ppl_per_ha"], seg["slope_below"], seg["slope_above"],
            s["single_power_law"]["beta"])


# ------------------------------------------------------------------ granularity map
def granularity_map(g, st, code, name, bp):
    sub = g[g["kommunkod"] == code].copy()
    kommun = sub.geometry.union_all()
    kom_dens = float(sub["S"].sum() / sub["area_ha"].sum())
    sst = st[st.within(kommun)].copy()
    # diverging colour scale CENTRED on the breakpoint: blue = rural side, red = urban
    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=0.0, vcenter=bp, vmax=float(g["dens"].quantile(0.97)))

    def stations(ax):
        if len(sst):
            sst.plot(ax=ax, color="black", markersize=14, marker="o",
                     edgecolor="white", linewidth=0.6, zorder=6)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    # (1) one kommun polygon, single density value (same breakpoint scale)
    ax = axes[0]
    gpd.GeoSeries([kommun], crs=CRS).plot(ax=ax, color=cmap(norm(kom_dens)),
                                          edgecolor="k", linewidth=0.8)
    ax.set_title(f"As ONE kommun\nsingle density = {kom_dens:.2f} ppl/ha  → reads rural",
                 fontsize=10)
    # (2) DeSO mosaic, coloured by distance from the breakpoint
    ax = axes[1]
    sub.plot(ax=ax, column="dens", cmap=cmap, norm=norm, edgecolor="white",
             linewidth=0.25, legend=True,
             legend_kwds={"label": f"← rural   density (ppl/ha)   urban →   "
                                   f"[white = break {bp:.1f}]", "shrink": 0.6})
    stations(ax)
    ax.set_title(f"As {len(sub)} DeSO units\ndensity {sub['dens'].min():.2f}–"
                 f"{sub['dens'].max():.0f} ppl/ha  (• = station)", fontsize=10)
    # (3) DeSO coloured rural/urban by breakpoint
    ax = axes[2]
    sub["regime"] = np.where(sub["dens"] >= bp, "urban", "rural")
    n_urb = int((sub["regime"] == "urban").sum())
    pop_urb = int(sub.loc[sub["regime"] == "urban", "S"].sum())
    sub.plot(ax=ax, color=[URBAN if r == "urban" else RURAL for r in sub["regime"]],
             edgecolor="white", linewidth=0.25)
    stations(ax)
    ax.set_title(f"Split at {bp:.1f} ppl/ha breakpoint\n"
                 f"{n_urb} urban units = {pop_urb:,} people "
                 f"({pop_urb/sub['S'].sum():.0%} of kommun)", fontsize=10)
    for ax in axes:
        ax.set_axis_off(); ax.set_aspect("equal")
    fig.suptitle(f"{name} — one rural-looking kommun is really a dense core inside a "
                 f"rural shell", fontsize=13, y=0.98)
    fig.tight_layout()
    out = C.CHARTS / f"11_granularity_{name.lower().replace('ö','o')}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    C.log(f"[OK] {out.name}")
    return out.name


# --------------------------------------------------------------- breakpoint curves
def binned_aggregates(g, nbins=16):
    """Equal-count density bins; aggregate so zeros are handled by summation."""
    q = pd.qcut(np.log(g["dens"]), nbins, labels=False, duplicates="drop")
    d = g.assign(_b=q).groupby("_b").agg(
        S=("S", "sum"), P=("P", "sum"), A=("area_ha", "sum")).reset_index()
    d = d[d["P"] > 0].copy()
    d["rho"] = d["S"] / d["A"]
    d["PA"] = d["P"] / d["A"]
    d["per1000"] = d["P"] / d["S"] * 1000.0
    # local slope: finite difference of log(PA) wrt log(rho) between adjacent bins
    lr = np.log(d["rho"].to_numpy()); lp = np.log(d["PA"].to_numpy())
    slope = np.gradient(lp, lr)
    d["local_slope"] = slope
    return d


def breakpoint_curves(g, bp, b_lo, b_hi, beta, examples):
    d = binned_aggregates(g)
    pos = g[g["P"] > 0]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    # (A) log-log scatter + two-regime fit
    ax = axes[0]
    ax.scatter(pos["dens"], pos["P"] / pos["area_ha"], s=8, alpha=0.18,
               color="#5a6b7b", edgecolors="none", label="DeSO (P>0)")
    xs = np.linspace(np.log(g["dens"].min()), np.log(g["dens"].max()), 200)
    # continuous piecewise line anchored at the binned aggregates
    a_lo = np.log(d["PA"].iloc[len(d)//4]) - b_lo * np.log(d["rho"].iloc[len(d)//4])
    yb = np.log(bp); a_hi = a_lo + b_lo * yb - b_hi * yb  # continuity at breakpoint
    yfit = np.where(xs < yb, a_lo + b_lo * xs, a_hi + b_hi * xs)
    ax.plot(np.exp(xs), np.exp(yfit), "k-", lw=2.2,
            label=f"fit: {b_lo:.2f} → {b_hi:.2f}")
    ax.axvline(bp, color="#c1440e", ls="--", lw=1.5)
    ax.axvspan(g["dens"].min(), bp, color=RURAL, alpha=0.06)
    ax.axvspan(bp, g["dens"].max(), color=URBAN, alpha=0.06)
    pal = ["#dd6b20", "#2b6cb0"]
    for (code, name), col in zip(examples.items(), pal):
        sv = g[(g["kommunkod"] == code) & (g["P"] > 0)]
        ax.scatter(sv["dens"], sv["P"] / sv["area_ha"], s=34, color=col,
                   edgecolors="k", linewidths=0.4, zorder=5, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("population density ρ (ppl/ha)")
    ax.set_ylabel("station density P/A (per ha)")
    ax.set_title("(A) the bend, on log paper", fontsize=11)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.2)

    # (B) local slope falling from ~1 to ~0.26
    ax = axes[1]
    ax.plot(d["rho"], d["local_slope"], "o-", color="#2b3a67", lw=1.6, ms=5)
    ax.axhline(1.0, color=RURAL, ls=":", lw=1.4, label="linear (β=1)")
    ax.axhline(b_hi, color=URBAN, ls=":", lw=1.4, label=f"urban β≈{b_hi:.2f}")
    ax.axvline(bp, color="#c1440e", ls="--", lw=1.5, label=f"break {bp:.1f} ppl/ha")
    ax.set_xscale("log")
    ax.set_xlabel("population density ρ (ppl/ha)")
    ax.set_ylabel("local scaling slope  d log(P/A) / d log ρ")
    ax.set_title("(B) the slope itself, vs density", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.2)

    # (C) stations per 1000 people — flat then saturating
    ax = axes[2]
    ax.plot(d["rho"], d["per1000"], "o-", color="#6b2d5c", lw=1.6, ms=5)
    ax.axvline(bp, color="#c1440e", ls="--", lw=1.5, label=f"break {bp:.1f} ppl/ha")
    ax.axvspan(g["dens"].min(), bp, color=RURAL, alpha=0.06)
    ax.axvspan(bp, g["dens"].max(), color=URBAN, alpha=0.06)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("population density ρ (ppl/ha)")
    ax.set_ylabel("stations per 1,000 people")
    ax.set_title("(C) provision per person: flat in rural, falls in urban", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.2)

    fig.suptitle("Where rural (≈linear) turns into urban (saturating): the petrol-station "
                 f"density breakpoint at {bp:.1f} ppl/ha (DeSO, single-β={beta:.2f})",
                 fontsize=12.5)
    fig.tight_layout()
    out = C.CHARTS / "11_breakpoint_curves.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    C.log(f"[OK] {out.name}")
    return out.name


# ----------------------------------------------------------------- infographic patch
def append_infographic(images, bp, b_lo, b_hi):
    """Append a self-contained image section to output/infographic.html (additive)."""
    import base64
    html = C.OUTPUT / "infographic.html"
    if not C.exists_nonempty(html):
        C.warn("infographic.html missing — skipping append")
        return
    txt = html.read_text()
    marker = "<!-- density-viz-appendix -->"
    if marker in txt:  # idempotent: strip a previously appended block
        txt = txt[:txt.index(marker)]

    def embed(name):
        p = C.CHARTS / name
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;border-radius:8px;margin:18px 0;"/>'

    blocks = "\n".join(
        f'<figure style="margin:0 0 30px 0;"><figcaption style="font:14px/1.5 -apple-system,sans-serif;'
        f'color:#bcd;margin-bottom:6px;">{cap}</figcaption>{embed(img)}</figure>'
        for img, cap in images)
    section = f"""{marker}
<section class="wrap" style="background:#0f1726;color:#e8eef6;padding:60px 24px;">
  <div style="max-width:1100px;margin:0 auto;">
    <h2 style="font:600 30px/1.2 -apple-system,sans-serif;color:#fff;">
      Appendix · Density scaling (draft panels)</h2>
    <p style="font:17px/1.6 -apple-system,sans-serif;color:#aebfd0;max-width:760px;">
      Population scaling treats a kommun as one number. But a kommun like Sundsvall or
      Örnsköldsvik is a dense city wrapped in a rural shell. Switch the x-axis to
      <b>population density</b> on fine DeSO units and the line stops being straight:
      petrol stations track density <b>≈linearly in the countryside (slope {b_lo:.2f})</b>,
      then <b>saturate in the city (slope {b_hi:.2f})</b>, with the turn at
      <b>{bp:.1f} people per hectare</b>. These are exploratory panels, not final art.</p>
    {blocks}
  </div>
</section>
"""
    txt = txt.rstrip()
    low = txt.lower()
    if "</body>" in low:
        i = low.rindex("</body>")
        txt = txt[:i] + section + "\n" + txt[i:]
    else:
        txt = txt + "\n" + section
    html.write_text(txt)
    C.log(f"[OK] appended density appendix to {html.name}")


def main():
    name = "Phase 11 — density visualisation"
    C.phase_start(name)
    C.set_seed()
    for need in (DESO, STATIONS, SUMMARY):
        if not C.exists_nonempty(need):
            C.warn(f"missing {Path(need).name} — run Phase 10 first; skipping")
            C.phase_end(name); return
    g, st = load()
    bp, b_lo, b_hi, beta = breakpoint_params()
    images = []
    for code, nm in EXAMPLES.items():
        if (g["kommunkod"] == code).any():
            fn = granularity_map(g, st, code, nm, bp)
            images.append((fn, f"{nm}: the same area as one kommun, as a DeSO density "
                               f"mosaic, and split at the {bp:.1f} ppl/ha breakpoint."))
    curves = breakpoint_curves(g, bp, b_lo, b_hi, beta, EXAMPLES)
    images.append((curves, "Three views of the rural→urban breakpoint on DeSO units."))
    append_infographic(images, bp, b_lo, b_hi)
    C.phase_end(name)


if __name__ == "__main__":
    main()

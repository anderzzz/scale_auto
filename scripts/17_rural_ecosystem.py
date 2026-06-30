"""Phase 17 — does a separate small-operator ecosystem hold up the rural tail?

Hypothesis (rhymes with the scaling story): the big brands build out the high-volume
markets and exit thin places, so the countryside is served by a different ecosystem —
discount/unmanned chains, local independents, village macks. Test it on the CLEANED
layer (Phase 16, brands folded in, marine/gas scoped out): assign each car-serving
station to its DeSO population density, group by operator tier, and look at the
composition across the density gradient.

Tiers grouped: Major (Circle K/OKQ8/Preem/St1) · Discount chain (Qstar/Ingo/Tanka/
din-X/Bilisten/Pump/Gulf) · Independent (+community macks) · Haulier · Unknown.

Writes charts/17_rural_ecosystem.png and output/rural_ecosystem.json.
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
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

CRS = C.CONFIG["national_crs"]
DPI = 150
CLEAN = C.CACHE / "stations_clean.gpkg"
DESO = C.CACHE / "units_deso.geojson"
BP = 5.7   # ppl/ha breakpoint

GROUP = {"major": "Major", "discount_chain": "Discount chain",
         "independent": "Independent", "community_mack": "Independent",
         "haulier_diesel": "Haulier", "untagged_osm": "Untagged (OSM)"}
ORDER = ["Major", "Discount chain", "Independent", "Haulier", "Untagged (OSM)"]
COLORS = {"Major": "#2b3a67", "Discount chain": "#dd6b20", "Independent": "#3a7d44",
          "Haulier": "#6b2d5c", "Untagged (OSM)": "#9aa5b1"}


def main():
    name = "Phase 17 — rural operator ecosystem"
    C.phase_start(name)
    C.set_seed()
    for need in (CLEAN, DESO):
        if not C.exists_nonempty(need):
            C.warn(f"missing {Path(need).name}; run Phase 16/10"); C.phase_end(name); return
    st = gpd.read_file(CLEAN)
    st = st[st["include_cars"]].copy()
    st["grp"] = st["tier"].map(GROUP).fillna("Unknown")

    g = gpd.read_file(DESO).to_crs(CRS)
    pop = C.read_json(C.CACHE / "pop_deso_regso.json")["deso"]
    g["S"] = g["desokod"].astype(str).map(pop).astype(float)
    g = g[g["S"].notna() & (g["S"] > 0)].copy()
    g["dens"] = g["S"] / (g.geometry.area / 1e4)
    j = gpd.sjoin(st.to_crs(CRS), g[["dens", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    j = j[j["dens"].notna()].copy()
    C.log(f"car-serving stations placed in a DeSO: {len(j)} / {len(st)}")

    # density deciles
    j["bin"] = pd.qcut(np.log(j["dens"]), 10, labels=False, duplicates="drop")
    binmed = j.groupby("bin")["dens"].median()
    comp = (j.groupby(["bin", "grp"]).size().unstack(fill_value=0)
            .reindex(columns=ORDER, fill_value=0))
    share = comp.div(comp.sum(axis=1), axis=0)
    n_bin = comp.sum(axis=1)

    # rural vs urban split at breakpoint
    rural = j[j["dens"] < BP]; urban = j[j["dens"] >= BP]
    def mix(sub):
        v = sub["grp"].value_counts(normalize=True)
        return {k: round(float(v.get(k, 0)), 3) for k in ORDER}
    rural_mix, urban_mix = mix(rural), mix(urban)
    nonmajor_rural = round(1 - rural_mix["Major"], 3)
    nonmajor_urban = round(1 - urban_mix["Major"], 3)

    # ---- chart
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    x = binmed.values
    ax = axes[0]
    bottom = np.zeros(len(share))
    for grp in ORDER:
        ax.fill_between(x, bottom, bottom + share[grp].values, label=grp,
                        color=COLORS[grp], step="mid", alpha=0.9)
        bottom += share[grp].values
    ax.axvline(BP, color="k", ls="--", lw=1.3)
    ax.annotate("breakpoint\n5.7 ppl/ha", (BP, 0.02), fontsize=8, ha="center")
    ax.set_xscale("log"); ax.set_xlim(x.min(), x.max()); ax.set_ylim(0, 1)
    ax.set_xlabel("DeSO population density (ppl/ha)  — rural ← → urban")
    ax.set_ylabel("share of car-serving stations")
    ax.set_title("Operator mix across the density gradient")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    ax = axes[1]
    ax.plot(x, share["Major"].values, "-o", color=COLORS["Major"], lw=2, label="Major share")
    nonmajor = 1 - share["Major"].values
    ax.plot(x, nonmajor, "-s", color="#c1440e", lw=2, label="Non-major share")
    ax.axvline(BP, color="k", ls="--", lw=1.3)
    ax.set_xscale("log"); ax.set_xlim(x.min(), x.max()); ax.set_ylim(0, 1)
    ax.set_xlabel("DeSO population density (ppl/ha)")
    ax.set_ylabel("share")
    ax.set_title(f"Majors retreat from the thin tail\n"
                 f"rural non-major {nonmajor_rural:.0%}  vs  urban non-major {nonmajor_urban:.0%}")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2)

    fig.suptitle("A separate rural ecosystem? Operator composition vs population density "
                 "(cleaned, car-serving stations)", fontsize=12.5)
    fig.tight_layout()
    out = C.CHARTS / "17_rural_ecosystem.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    C.log(f"[OK] {out.name}")

    summary = {
        "n_stations_placed": int(len(j)),
        "breakpoint_ppl_per_ha": BP,
        "rural_mix": rural_mix, "urban_mix": urban_mix,
        "nonmajor_share_rural": nonmajor_rural, "nonmajor_share_urban": nonmajor_urban,
        "most_rural_decile_mix": {k: round(float(v), 3) for k, v in
                                  share.iloc[0].items()},
        "most_urban_decile_mix": {k: round(float(v), 3) for k, v in
                                  share.iloc[-1].items()},
        "n_per_bin": {int(k): int(v) for k, v in n_bin.items()},
    }
    C.write_json(C.OUTPUT / "rural_ecosystem.json", summary)
    C.log(f"rural non-major share={nonmajor_rural:.0%} | urban non-major={nonmajor_urban:.0%}")
    C.log(f"most-rural decile mix: { {k: round(float(v),2) for k,v in share.iloc[0].items()} }")
    C.log(f"most-urban decile mix: { {k: round(float(v),2) for k,v in share.iloc[-1].items()} }")
    C.phase_end(name)


if __name__ == "__main__":
    main()

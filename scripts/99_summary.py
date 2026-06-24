"""Final stdout summary (Definition of Done)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LABELS = {"tatort": "Tätort (settlement)", "tatort_smaort": "Tätort + småort",
          "kommun": "Kommun (admin)", "fua": "FUA (functional)"}


def main():
    res = C.read_json(C.OUTPUT / "results.json", {})
    stn = C.read_json(C.OUTPUT / "stations_summary.json", {})
    defs = res.get("definitions", {}) if res else {}
    print("\n" + "=" * 70)
    print("PETROL-STATION SCALING IN SWEDEN  —  P = C · S^beta")
    print("=" * 70)
    print(f"{'Unit definition':<24}{'beta':>8}{'95% CI':>18}   verdict (H0: beta=1)")
    print("-" * 70)
    for d in ["tatort", "tatort_smaort", "fua", "kommun"]:
        v = defs.get(d, {})
        a = v.get("model_A_primary")
        if not a:
            print(f"{LABELS.get(d,d):<24}{'—':>8}{'—':>18}   MISSING")
            continue
        ci = a["ci95"]
        print(f"{LABELS[d]:<24}{a['beta']:>8.3f}{f'[{ci[0]:.3f},{ci[1]:.3f}]':>18}"
              f"   {a['verdict']}  (p={a['p_vs1']:.1e})")
    print("-" * 70)
    comp = stn.get("completeness", {})
    if comp:
        ok = "WITHIN band" if comp.get("within_band") else "OUTSIDE band (caveat!)"
        print(f"OSM completeness: {comp.get('deduped_count')} stations vs Drivkraft anchor "
              f"~{comp.get('drivkraft_anchor')}  -> {ok} {comp.get('tolerance_band')}  "
              f"(ratio {comp.get('ratio_to_anchor')}x)")
    units = C.read_json(C.OUTPUT / "units_summary.json", {})
    rs = units.get("rural_share", {})
    if rs:
        print(f"Rural share (outside all tätorter): {rs.get('stations_outside_tatort')}/"
              f"{rs.get('stations_total')} = {rs.get('rural_fraction',0)*100:.1f}%")
    missing = [d for d in ["tatort", "tatort_smaort", "kommun", "fua"]
               if not defs.get(d, {}).get("model_A_primary")]
    if missing:
        print(f"INCOMPLETE: missing definitions -> {', '.join(missing)} (see report caveats)")
    print("=" * 70)
    print(f"Report: {(C.OUTPUT / 'report.md')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

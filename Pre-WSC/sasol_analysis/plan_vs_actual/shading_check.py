"""Canopy-shading check on the four MPPT strings (A rear-left, B rear-right, C front-left, D front-right).
Each logged packet gets a route position (integrated speed, as validate_motor.py), hence a heading and
coordinates from the route file; pvlib gives sun azimuth/elevation. Relative sun azimuth
rel = sun_az - heading  (0 = sun dead ahead, +90 = sun on the car's right, 180 = sun behind).
If the canopy shades the front strings, C/A and D/B should drop when the sun is ahead, C should drop when
the sun is on the right (shadow falls left) and D when the sun is on the left.
Writes shading_packets.csv, shading_by_relaz.csv, fig_shading.png
"""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd, pvlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import actuals, validate_motor as vm, stitch_days as sd

def per_channel(day):
    """re-read packets keeping per-string input power (load_day sums them)."""
    rows = []
    for path in sorted((sd.RAW_DIR / f"Day {day}").glob("*.jsonl")):
        recs, _ = sd.read_fragment(path)
        for r in recs:
            if sd.MAIN_MARKER in r:
                rows.append(tuple(r[f"Input_Voltage_{c}"] * r[f"Input_Current_{c}"] for c in "ABCD"))
    return rows

def main():
    out = []
    for (day, phase), pat in vm.STAGES.items():
        df = actuals.load_day(day)
        # attach per-string power by re-running the same load with string columns
        f = glob.glob(str(pat))[0]; prof = json.load(open(f, encoding="utf-8"))["profile"]
        dist = np.array(prof["Distance"], float); hd = np.r_[np.array(prof["Headings"], float), prof["Headings"][-1]]
        co = np.array(prof["Coordinates"], float)
        g = vm.stage_frame(df, phase)
        if (day, phase) in vm.FULL:
            g["x_km"] = g.x_km * (dist[-1] / g.x_km.iloc[-1])
        x = g.x_km.to_numpy()
        # heading: interpolate on unit circle
        hr = np.radians(hd)
        head = np.degrees(np.arctan2(np.interp(x, dist, np.sin(hr)), np.interp(x, dist, np.cos(hr)))) % 360
        lat = np.interp(x, dist, co[:, 0]); lon = np.interp(x, dist, co[:, 1])
        sp = pvlib.solarposition.get_solarposition(g.t.dt.tz_convert("UTC"), lat, lon)
        g["hdg"] = head; g["sun_az"] = sp.azimuth.to_numpy(); g["sun_el"] = sp.apparent_elevation.to_numpy()
        g["rel"] = ((g.sun_az - g.hdg + 180) % 360) - 180
        g["day"] = day; g["phase"] = phase
        out.append(g)
    P = pd.concat(out, ignore_index=True)
    return P

# String area shares from the tilted-at-sun stops (Day 1 10:20-10:30, Day 8 11:30-12:05 SAST, no shading):
# both days give the same split, and it matches the strings' MPP-voltage ratio (C has fewer cells).
SHARE = {"A": 0.298, "B": 0.266, "C": 0.185, "D": 0.251}


if __name__ == "__main__":
    P = main()
    D = P[(P.kmh > 20) & (P.sun_el > 25) & (P.dt_c > 0)].copy()
    D["bin"] = ((D.rel + 180) // 30) * 30 - 180 + 15
    # each string's output per m2 as a fraction of the best string in the same packet (irradiance cancels)
    pm = np.column_stack([D[f"p_{c}"] / SHARE[c] for c in "ABCD"])
    D = D[pm.max(1) > 100].copy(); pm = pm[pm.max(1) > 100]
    for i, c in enumerate("ABCD"):
        D[f"f_{c}"] = pm[:, i] / pm.max(1)
    D[["day", "phase", "t", "kmh", "hdg", "sun_az", "sun_el", "rel", "p_A", "p_B", "p_C", "p_D"] + [f"f_{c}" for c in "ABCD"]].round(3).to_csv(HERE / "shading_packets.csv", index=False)
    agg = {"n": ("f_A", "size"), "sun_el": ("sun_el", "mean")}
    for c in "ABCD":
        agg[f"{c}_mean"] = (f"f_{c}", "mean"); agg[f"{c}_std"] = (f"f_{c}", "std")
    g = D.groupby("bin").agg(**agg)
    g.round(3).to_csv(HERE / "shading_by_relaz.csv")
    print(g.round(2).to_string())
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sty = {"A": ("#7a9cc6", "o", -3), "B": ("#e3a07f", "o", -1), "C": ("#1f4e8c", "s", 1), "D": ("#a8361b", "s", 3)}
    for c, (col, mk, off) in sty.items():
        ax.errorbar(g.index + off, g[f"{c}_mean"], g[f"{c}_std"], fmt=mk + "-", capsize=2, color=col, label=f"string {c}", lw=1.2, ms=4)
    ax.set_xticks(range(-180, 181, 45)); ax.set_ylim(0, 1.15)
    ax.set_xlabel("sun azimuth relative to heading (deg): 0 = ahead, +90 = sun on car's right, +-180 = behind")
    ax.set_ylabel("output per m2 / best string")
    ax.legend(fontsize=8, ncol=4, loc="lower center")
    ax.set_title("String output while driving (>20 km/h, sun elevation >25 deg), mean +- 1 std per 30 deg bin", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "fig_shading.png", dpi=200)

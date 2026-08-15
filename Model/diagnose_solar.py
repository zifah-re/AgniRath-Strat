"""
diagnose_solar.py — Run from Model/ to check GHI values from the solar JSONs.

Usage:
    cd Model/
    python diagnose_solar.py

This will print hourly GHI from each day's weather file so we can see
if the solar data is realistic.
"""
import os
import glob
import json
import numpy as np

json_dir = os.path.join(os.path.dirname(__file__), "data", "solar")
if not os.path.isdir(json_dir):
    # Try relative to CWD
    json_dir = os.path.join("data", "solar")

print(f"Solar data directory: {json_dir}")
print()

for d in range(8):
    pattern = os.path.join(json_dir, f"*Day {d+1}*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Day {d+1}: NO JSON FILES FOUND")
        continue

    print(f"Day {d+1}: {len(files)} file(s)")

    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"  File: {fname}")
        print(f"  Nodes: {len(data)}")

        # Average GHI across all nodes for each hour
        all_ghi = []
        for node in data:
            ghi = node["historical_weather"]["hourly"]["shortwave_radiation"]
            all_ghi.append(ghi)

        all_ghi = np.array(all_ghi)  # shape: (n_nodes, 24)
        avg_ghi = all_ghi.mean(axis=0)

        print(f"  Avg GHI by hour (W/m²):")
        print(f"    ", end="")
        for h in range(24):
            print(f"h{h:02d}={avg_ghi[h]:6.1f}", end="  ")
            if h == 11:
                print()
                print(f"    ", end="")
        print()

        # Key metrics
        daytime = avg_ghi[8:17]  # 8 AM to 4 PM (race hours)
        print(f"  Daytime avg (8-17h): {daytime.mean():.1f} W/m²")
        print(f"  Peak: {avg_ghi.max():.1f} W/m² at hour {avg_ghi.argmax()}")
        print(f"  Dawn (6-8h): {avg_ghi[6:8].mean():.1f} W/m²")

        # Overnight gain check: GHI at 7:00 AM
        ghi_7am = avg_ghi[7]
        p_solar_7am = 5.95 * 0.18 * ghi_7am - 70.0
        overnight_wh = p_solar_7am * 2.0  # 2 hours (6-8 AM)
        overnight_pct = overnight_wh * 0.96 / 3528.0 * 100.0 if overnight_wh >= 0 else overnight_wh / 0.96 / 3528.0 * 100.0
        print(f"  Overnight gain estimate: {overnight_pct:+.1f}% "
              f"(GHI@7am={ghi_7am:.0f}, p_solar={p_solar_7am:+.0f}W)")

        # Quick energy balance at 40 km/h (typical Tier 1 speed)
        v_ms = 40.0 / 3.6  # 11.11 m/s
        p_rolling = 300 * 9.81 * 0.007 * v_ms  # W
        p_drag = 0.5 * 1.2 * 0.16 * v_ms**3  # W
        p_elec = (p_rolling + p_drag) / 0.95 + 70.0  # W (motor eff + idle)
        p_solar_avg = 5.95 * 0.18 * daytime.mean()  # W
        p_net = p_solar_avg - p_elec
        print(f"  Energy balance @40km/h: solar={p_solar_avg:.0f}W, "
              f"elec={p_elec:.0f}W, net={p_net:+.0f}W")
        print()

print("\n--- SUMMARY ---")
print("If daytime avg GHI < 400 W/m², the solar data may be unrealistic")
print("for September in South Africa (expect 500-700 W/m² avg during race hours).")
print("If overnight gain is negative, the car loses charge every night,")
print("making multi-day feasibility very tight.")
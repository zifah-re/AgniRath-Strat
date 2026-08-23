"""
diagnose_day1.py — Deep diagnostic of Day 1's energy drain.

Run from Model/:
    python diagnose_day1.py

Checks: route slopes, GHI values, regen cap losses, energy breakdown.
"""
import os, glob, json, sys
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.car_config import CarState
from configs import race_config as rc
from core.solar import HourlyJSONSolarProvider, GaussianProvider
from core.wind import HourlyJSONWindProvider, ConstantWindProvider
from core.route import Route
from core import physics
from core.battery import Battery

car = CarState()
current_dir = os.path.dirname(os.path.abspath(__file__))
json_dir  = os.path.abspath(os.path.join(current_dir, "data", "solar"))
save_dir  = os.path.abspath(os.path.join(current_dir, "data", "processed"))

# ---------- helpers (copied from trust_region __main__) ----------
def _route_sort_key(filepath):
    name = filepath.lower()
    if "stage 1" in name: return 1
    if "loop"    in name: return 2
    if "stage 2" in name: return 3
    return 4

def _load_route(route_files, day_num):
    route_files = sorted(route_files, key=_route_sort_key)
    day_dfs = []
    offset = 0.0
    for filepath in route_files:
        with open(filepath, "r", encoding="utf-8") as f:
            route_data = json.load(f)
        prof = route_data["profile"]
        dists    = [x * 1000.0 for x in prof["Distance"]]
        slopes   = prof["Gradient"]
        bearings = prof.get("Headings",   [0.0] * len(dists))
        alts     = prof.get("Altitude",   [0.0] * len(dists))
        lats     = [c[0] for c in prof["Coordinates"]]
        lons     = [c[1] for c in prof["Coordinates"]]
        v_maxs   = [v / 3.6 for v in prof["SpeedLimit"]]
        ml = min(len(dists), len(slopes), len(bearings),
                 len(alts), len(lats), len(lons), len(v_maxs))
        part_df = pd.DataFrame({
            "distance_m":      dists[:ml],
            "elevation_m":     alts[:ml],
            "slope_pct":       slopes[:ml],
            "bearing_deg":     bearings[:ml],
            "lat":             lats[:ml],
            "lon":             lons[:ml],
            "v_max_ms":        v_maxs[:ml],
            "curvature_1pm":   0.0,
            "circle_id":       0,
            "red_flag_trailer": False,
            "control_stop":    False,
            "day":             day_num,
            "seg_type":        "stage",
        })
        part_df["distance_m"] += offset
        offset = part_df["distance_m"].max()
        day_dfs.append(part_df)
    return Route(pd.concat(day_dfs, ignore_index=True))

# ---------- Load Day 1 ----------
route_files = sorted(glob.glob(os.path.join(save_dir, "*Day 1*.save")), key=_route_sort_key)
weather_files = glob.glob(os.path.join(json_dir, "*Day 1*.json"))

print(f"Route files: {[os.path.basename(f) for f in route_files]}")
print(f"Weather files: {[os.path.basename(f) for f in weather_files]}")

route = _load_route(route_files, 1)
solar = HourlyJSONSolarProvider(weather_files, route)
wind = ConstantWindProvider(0.0, 0.0)

# ---------- 1. Route slope analysis ----------
print("\n" + "="*70)
print("1. ROUTE SLOPE ANALYSIS")
print("="*70)
df = route.df
slopes = df["slope_pct"].values
print(f"Total points: {len(slopes)}")
print(f"Total distance: {df['distance_m'].max()/1000:.1f} km")
print(f"Slope stats: min={slopes.min():.1f}%, max={slopes.max():.1f}%, "
      f"mean={slopes.mean():.2f}%, std={slopes.std():.2f}%")

# Distribution
bins = [0, 1, 2, 3, 5, 8, 10, 15, 20, 50, 100]
for i in range(len(bins)-1):
    n = np.sum((np.abs(slopes) >= bins[i]) & (np.abs(slopes) < bins[i+1]))
    pct = n / len(slopes) * 100
    print(f"  |slope| {bins[i]:3d}-{bins[i+1]:3d}%: {n:5d} points ({pct:.1f}%)")
n_extreme = np.sum(np.abs(slopes) >= 20)
print(f"  |slope| >= 20%: {n_extreme} points ({n_extreme/len(slopes)*100:.1f}%)")

# Elevation profile
alts = df["elevation_m"].values
print(f"\nElevation: start={alts[0]:.0f}m, end={alts[-1]:.0f}m, "
      f"min={alts.min():.0f}m, max={alts.max():.0f}m")
print(f"Net elevation change: {alts[-1]-alts[0]:+.0f}m")

# Total absolute climb/descent
dists = df["distance_m"].values
d_alt = np.diff(alts)
total_climb = d_alt[d_alt > 0].sum()
total_descent = abs(d_alt[d_alt < 0].sum())
print(f"Total climb: {total_climb:.0f}m, total descent: {total_descent:.0f}m")

# ---------- 2. Simulate Day 1 like Tier 1 does ----------
print("\n" + "="*70)
print("2. TIER 1 ENERGY SIMULATION")
print("="*70)

# Get Day 1 plan
note = rc.DAY_ROUTE_NOTES[0]
stage1_km = note["stage1_km"] if note["stage1_km"] else 230.0
stage2_km = note["stage2_km"] if note["stage2_km"] else 0.0
loops = note["loops"] if note["loops"] else []
print(f"Plan: Stage1={stage1_km}km, Stage2={stage2_km}km, Loops={loops}")

base_km = stage1_km + stage2_km
t0_s = rc.day_start_time_s(0)  # Day 1 start
t_end = rc.day_finish_time_s(0)
t_window = t_end - t0_s
print(f"Time window: {t0_s/3600:.1f}h - {t_end/3600:.1f}h = {t_window/3600:.1f}h")

v_base_ms = (base_km * 1000.0) / max(t_window, 1e-6) if base_km > 0 else car.v_max_ms
v_base_ms = min(v_base_ms, car.v_max_ms)
print(f"Tier 1 base speed: {v_base_ms:.2f} m/s = {v_base_ms*3.6:.1f} km/h")

# Sample the route
TIER1_SAMPLE_M = 500.0
n_seg = max(1, int(round(base_km * 1000.0 / TIER1_SAMPLE_M)))
seg_len = base_km * 1000.0 / n_seg
edges = np.linspace(0, base_km * 1000.0, n_seg + 1)
mid = (edges[:-1] + edges[1:]) / 2.0
slope = np.asarray(route.slope_pct_at(mid), dtype=float)
bearing = np.asarray(route.bearing_deg_at(mid), dtype=float)

print(f"\nSampled {n_seg} segments @ {seg_len:.0f}m each")
print(f"Sampled slopes: min={slope.min():.1f}%, max={slope.max():.1f}%, "
      f"mean={slope.mean():.2f}%, std={slope.std():.2f}%")

# Time and position arrays
v_ms = np.full(n_seg, v_base_ms)
dt_s = np.full(n_seg, seg_len / v_base_ms)
t_pt = t0_s + np.cumsum(dt_s) - dt_s
x_m = np.cumsum(np.full(n_seg, seg_len)) - seg_len

# GHI
ghi = np.array([solar.ghi_wm2(float(t_pt[i]), float(x_m[i])) for i in range(n_seg)])
print(f"\nGHI: min={ghi.min():.0f}, max={ghi.max():.0f}, mean={ghi.mean():.0f} W/m²")
print(f"GHI at start (t={t_pt[0]/3600:.1f}h): {ghi[0]:.0f} W/m²")
print(f"GHI at midday: {ghi[n_seg//2]:.0f} W/m²")

# Forces
f = physics.forces(car, v_ms, slope, wind_along_ms=np.zeros(n_seg), yaw_deg=np.zeros(n_seg))
p_mech = (f["drag"] + f["rolling"] + f["gravity"]) * v_ms

regen_cap = car.p_max_continuous_w * car.p_max_derating  # 1530 W
regen_into_pack = np.where(p_mech < 0.0, np.minimum(-p_mech * car.regen_eff, regen_cap), 0.0)
p_electric = np.where(p_mech >= 0.0, p_mech / car.motor_eff, -regen_into_pack)

p_solar = car.array_area_m2 * car.array_efficiency * ghi  # no geom correction for simplicity
p_net = p_solar - p_electric - car.p_idle_w

print(f"\nRegen cap: {regen_cap:.0f} W")

# Energy breakdown
energy_wh = p_net * dt_s / 3600.0

# Categorize
uphill_mask = p_mech > 0
downhill_mask = p_mech < 0
flat_mask = p_mech == 0

# How much energy is wasted at the regen cap?
uncapped_regen = np.where(p_mech < 0, -p_mech * car.regen_eff, 0.0)
capped_regen = regen_into_pack
regen_waste = uncapped_regen - capped_regen
total_regen_waste_wh = (regen_waste * dt_s / 3600.0).sum()

print(f"\n--- Energy breakdown ---")
print(f"Solar energy:     {(p_solar * dt_s / 3600).sum():+.0f} Wh")
print(f"Idle loss:        {(-car.p_idle_w * dt_s / 3600).sum():+.0f} Wh")
print(f"Uphill elec cost: {(np.where(uphill_mask, -p_electric * dt_s / 3600, 0)).sum():+.0f} Wh")
print(f"Downhill regen:   {(np.where(downhill_mask, -p_electric * dt_s / 3600, 0)).sum():+.0f} Wh")
print(f"Regen wasted (cap): {-total_regen_waste_wh:+.0f} Wh ({total_regen_waste_wh/car.battery_nominal_wh*100:.1f}% of battery)")
print(f"Net energy:       {energy_wh.sum():+.0f} Wh ({energy_wh.sum()/car.battery_nominal_wh*100:+.1f}% SOC)")

# Worst segments
print(f"\n--- Worst 10 segments (most negative p_net) ---")
worst = np.argsort(p_net)[:10]
for i in worst:
    print(f"  seg {i}: x={x_m[i]/1000:.1f}km, slope={slope[i]:+.1f}%, "
          f"ghi={ghi[i]:.0f}, p_mech={p_mech[i]:+.0f}W, p_elec={p_electric[i]:+.0f}W, "
          f"p_solar={p_solar[i]:.0f}W, p_net={p_net[i]:+.0f}W")

# ---------- 3. Run battery simulation ----------
print(f"\n--- Battery simulation ---")
bat = Battery(car, 100.0)
min_soc = 100.0
for i, e in enumerate(energy_wh):
    bat.apply_energy_wh(float(e))
    if bat.soc_pct < min_soc:
        min_soc = bat.soc_pct

# Add control stop solar
t_cs = t0_s + (base_km / 2 * 1000 / v_base_ms)
p_cs = car.array_area_m2 * car.array_efficiency * solar.ghi_wm2(t_cs, base_km/2*1000) - car.p_idle_w
total_stop_s = rc.CONTROL_STOP_DURATION_S + rc.UNPLANNED_STOP_BUDGET_S
bat.apply_energy_wh(p_cs * total_stop_s / 3600.0)

print(f"Start SOC:  100.0%")
print(f"End SOC:    {bat.soc_pct:.1f}% (after driving + stops)")
print(f"Min SOC:    {min_soc:.1f}% (during driving)")
print(f"SOC change: {bat.soc_pct - 100.0:+.1f}%")
print(f"Stop solar: p_cs={p_cs:+.0f}W, duration={total_stop_s:.0f}s, energy={p_cs*total_stop_s/3600:+.0f}Wh")

# ---------- 4. Check solar_geom effect ----------
print(f"\n--- Solar geometry factor ---")
from core.solar import slope_geometry_factor
lat_deg = -26.2  # Approx Day 1 latitude
race_date = rc.RACE_DAY_DATES[0]
doy = race_date.timetuple().tm_yday
geom = np.array([
    slope_geometry_factor(lat_deg, doy, float(t_pt[i]), float(slope[i]), float(bearing[i]), car.panel_tilt_base_deg)
    for i in range(n_seg)
])
print(f"Solar geom factor: min={geom.min():.3f}, max={geom.max():.3f}, mean={geom.mean():.3f}")
print(f"Zeros (sun below horizon): {(geom == 0).sum()} segments")
print(f"Segments where geom < 0.5: {(geom < 0.5).sum()}")

# With geom correction
p_solar_geom = p_solar * geom
energy_geom = (p_solar_geom - p_electric - car.p_idle_w) * dt_s / 3600
print(f"\nWith solar geom correction:")
print(f"Solar energy:  {(p_solar_geom * dt_s / 3600).sum():+.0f} Wh (vs {(p_solar * dt_s / 3600).sum():+.0f} without)")
print(f"Net energy:    {energy_geom.sum():+.0f} Wh ({energy_geom.sum()/car.battery_nominal_wh*100:+.1f}% SOC)")

# Battery sim with geom
bat2 = Battery(car, 100.0)
for e in energy_geom:
    bat2.apply_energy_wh(float(e))
bat2.apply_energy_wh(p_cs * total_stop_s / 3600.0)
print(f"End SOC with geom: {bat2.soc_pct:.1f}%")

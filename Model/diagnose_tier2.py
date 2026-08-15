"""
diagnose_tier2.py — Pinpoint why singleday.solve returns infeasible.

Run from Model/:
    python diagnose_tier2.py

This bypasses the full optimizer and calls singleday.solve + tier2 internals
directly for Day 1, printing exactly where it fails.
"""
import os, sys, glob, json, logging, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("diagnose_tier2")

from configs.car_config import CarState
from configs import race_config as rc
from core.solar import HourlyJSONSolarProvider, GaussianProvider
from core.wind import HourlyJSONWindProvider, ConstantWindProvider
from core.route import Route
from optimizers import singleday
from optimizers import tier2

car = CarState()
current_dir = os.path.dirname(os.path.abspath(__file__))
json_dir  = os.path.join(current_dir, "data", "solar")
save_dir  = os.path.join(current_dir, "data", "processed")

# ---------- Load Day 1 ----------
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

route_files = sorted(glob.glob(os.path.join(save_dir, "*Day 1*.save")), key=_route_sort_key)
weather_files = glob.glob(os.path.join(json_dir, "*Day 1*.json"))

route = _load_route(route_files, 1)
solar = HourlyJSONSolarProvider(weather_files, route)
wind = HourlyJSONWindProvider(weather_files, route)

print(f"Route: {route.df.shape[0]} points, {route.df['distance_m'].max()/1000:.1f} km")
print(f"Weather: {len(weather_files)} files")

# ========== TEST 1: Call singleday.solve directly ==========
print("\n" + "=" * 70)
print("TEST 1: singleday.solve — Day 1, 0 loops, start=100%, alpha=20%")
print("=" * 70)

res1 = None
try:
    res1 = singleday.solve(
        route=route,
        car=car,
        solar_provider=solar,
        wind_provider=wind,
        day_index=0,
        start_soc_pct=100.0,
        alpha_next_day_pct=20.0,  # very lenient
        loops_committed=[],
    )
    allowed_time_s = (
        rc.day_finish_time_s(0) - rc.day_start_time_s(0) - rc.CONTROL_STOP_DURATION_S
    )
    time_ok = res1["total_time_s"] <= allowed_time_s
    soc_ok = res1["final_soc_pct"] >= car.soc_min_pct
    print(f"final_soc_pct: {res1.get('final_soc_pct')}")
    print(f"total_time_s: {res1.get('total_time_s'):.0f}  (allowed: {allowed_time_s:.0f})")
    print(f"REAL feasible (time & soc floor both satisfied): {time_ok and soc_ok}")
    for k, v in res1.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"  {k}: {v}")
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")
    traceback.print_exc()

# ========== TEST 2: singleday.solve with stricter alpha ==========
print("\n" + "=" * 70)
print("TEST 2: singleday.solve — Day 1, 0 loops, start=100%, alpha=26.8%")
print("=" * 70)

try:
    res = singleday.solve(
        route=route,
        car=car,
        solar_provider=solar,
        wind_provider=wind,
        day_index=0,
        start_soc_pct=100.0,
        alpha_next_day_pct=26.8,
        loops_committed=[],
    )
    allowed_time_s = (
        rc.day_finish_time_s(0) - rc.day_start_time_s(0) - rc.CONTROL_STOP_DURATION_S
    )
    time_ok = res["total_time_s"] <= allowed_time_s
    soc_ok = res["final_soc_pct"] >= car.soc_min_pct
    print(f"final_soc_pct: {res.get('final_soc_pct')}")
    print(f"total_time_s: {res.get('total_time_s'):.0f}  (allowed: {allowed_time_s:.0f})")
    print(f"REAL feasible (time & soc floor both satisfied): {time_ok and soc_ok}")
    for k, v in res.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"  {k}: {v}")
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")
    traceback.print_exc()

# ========== TEST 3: singleday.solve for Day 2 ==========
print("\n" + "=" * 70)
print("TEST 3: singleday.solve — Day 2, 0 loops, start=29.3%, alpha=20%")
print("=" * 70)

route2_files = sorted(glob.glob(os.path.join(save_dir, "*Day 2*.save")), key=_route_sort_key)
weather2_files = glob.glob(os.path.join(json_dir, "*Day 2*.json"))

if route2_files:
    route2 = _load_route(route2_files, 2)
    solar2 = HourlyJSONSolarProvider(weather2_files, route2)
    wind2 = HourlyJSONWindProvider(weather2_files, route2)

    try:
        res = singleday.solve(
            route=route2,
            car=car,
            solar_provider=solar2,
            wind_provider=wind2,
            day_index=1,
            start_soc_pct=29.3,
            alpha_next_day_pct=20.0,
            loops_committed=[],
        )
        allowed_time_s = (
            rc.day_finish_time_s(1) - rc.day_start_time_s(1) - rc.CONTROL_STOP_DURATION_S
        )
        time_ok = res["total_time_s"] <= allowed_time_s
        soc_ok = res["final_soc_pct"] >= car.soc_min_pct
        print(f"final_soc_pct: {res.get('final_soc_pct')}")
        print(f"total_time_s: {res.get('total_time_s'):.0f}  (allowed: {allowed_time_s:.0f})")
        print(f"REAL feasible (time & soc floor both satisfied): {time_ok and soc_ok}")
        for k, v in res.items():
            if isinstance(v, (int, float, str, bool)):
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()

# ========== TEST 4: Check tier2 internals ==========
print("\n" + "=" * 70)
print("TEST 4: tier2 internal — what does _ordered_combos / sample_day do?")
print("=" * 70)

for attr in ["SAMPLE_WINDOW_PCT", "SOC_OFFSETS_PCT", "MAX_COMBOS"]:
    val = getattr(tier2, attr, "NOT FOUND")
    print(f"  tier2.{attr} = {val}")

print(f"\n  tier2 public functions: {[x for x in dir(tier2) if not x.startswith('_')]}")

from optimizers.tier1 import _get_day_plan
plan = _get_day_plan(0)
print(f"\n  Day 1 plan: stage1={plan.stage1_km}km stage2={plan.stage2_km}km "
      f"loops={plan.loops}")

if hasattr(tier2, '_ordered_combos'):
    combos = tier2._ordered_combos(plan, 0, car, is_today=True, elapsed_s=0.0)
    print(f"  _ordered_combos returned {len(combos)} combos:")
    for c in combos:
        print(f"    {c}")
elif hasattr(tier2, '_sweep_one_offset'):
    print("  _sweep_one_offset exists but _ordered_combos not accessible")

if hasattr(tier2, 'sample_day'):
    import inspect
    sig = inspect.signature(tier2.sample_day)
    print(f"\n  sample_day signature: {sig}")

# ========== TEST 5: Per-segment breakdown — verify the turn-cap fix ==========
print("\n" + "=" * 70)
print("TEST 5: Per-segment v_chosen vs v_max (turn-cap fix verification)")
print("=" * 70)

if res1 is not None:
    from optimizers.singleday import apply_turn_speed_caps

    v_kmh = res1["v_kmh"]
    seg_start_m = res1["seg_start_m"]

    v_max_raw_kmh = route.v_max_ms_at(seg_start_m) * 3.6
    v_max_capped_kmh = apply_turn_speed_caps(route, v_max_raw_kmh, seg_start_m)
    slope_at_seg = np.asarray(route.slope_pct_at(seg_start_m), dtype=float)

    print(f"\n{'seg#':>5} {'km':>7} {'v_chosen':>9} {'v_max_raw':>10} {'v_max_capped':>13} {'slope%':>7} {'AT_CAP':>7}")
    n_at_cap = 0
    for i in range(len(v_kmh)):
        at_cap = (v_kmh[i] <= v_max_capped_kmh[i] + 0.5
                  and v_max_capped_kmh[i] < v_max_raw_kmh[i] - 0.5)
        n_at_cap += int(at_cap)
        marker = "  <<<" if at_cap else ""
        print(f"{i:5d} {seg_start_m[i]/1000:7.1f} {v_kmh[i]:9.1f} "
              f"{v_max_raw_kmh[i]:10.1f} {v_max_capped_kmh[i]:13.1f} "
              f"{slope_at_seg[i]:7.1f} {marker}")

    print(f"\nfinal_soc_pct: {res1['final_soc_pct']:.1f}")
    allowed_time_s = rc.day_finish_time_s(0) - rc.day_start_time_s(0) - rc.CONTROL_STOP_DURATION_S
    print(f"total_time_s: {res1['total_time_s']:.0f}  (allowed: {allowed_time_s:.0f})")
    print(f"segments where v_chosen sits at the turn cap: {n_at_cap} / {len(v_kmh)}")
else:
    print("  Skipped — Test 1 did not return a result.")

print("\n" + "=" * 70)
print("DONE — paste this output so we can see exactly where tier2 fails")
print("=" * 70)
"""
optimizers/hierarchical/trust_region.py — coarse-to-fine driver loop.
"""

from __future__ import annotations

import logging
import numpy as np
import dataclasses
import glob
import pandas as pd

from configs import race_config as rc
from configs.car_config import CarState
from optimizers import singleday
from .tier1 import _get_day_plan
from . import tier1, tier2, tier3
from .tier1 import _adjust_plan_for_today

logger = logging.getLogger(__name__)

MAX_ITERS = 4                          
CONVERGENCE_WINDOW_PCT = tier2.SAMPLE_WINDOW_PCT

def _alpha_floors_from_traj(s1_pct: np.ndarray, car: CarState, start_day: int) -> dict:
    n = len(s1_pct)
    floors = {}
    for d in range(start_day, n):
        val = float(s1_pct[d + 1]) if d + 1 < n else car.soc_min_pct
        # NaN from infeasible Tier 1 → fall back to minimum SOC floor.
        # This gives Tier 2 maximum freedom to find feasible solutions.
        floors[d] = val if np.isfinite(val) else car.soc_min_pct
    return floors

def replan(routes: list, base_car: CarState, solar_providers: dict, wind_providers: dict,
           cur_soc_pct: float, *, cur_day: int, dist_done_km: float = 0.0,
           elapsed_s: float = 0.0, cs_taken: bool = False,
           loops_done: dict | None = None, overrides: dict | None = None,
           kml_paths: dict | None = None, **kwargs) -> dict:
    
    car = dataclasses.replace(base_car, **(overrides or {}))
    return optimize(routes, car, solar_providers, wind_providers, cur_soc_pct,
                    start_day=cur_day, dist_done_km=dist_done_km,
                    elapsed_s=elapsed_s, cs_taken=cs_taken, 
                    loops_done=loops_done, kml_paths=kml_paths, **kwargs)

def optimize(routes: list, car: CarState, solar_providers: dict, wind_providers: dict,
             start_soc_pct: float = 100.0, *, start_day: int = 0,
             dist_done_km: float = 0.0, elapsed_s: float = 0.0,
             cs_taken: bool = False, loops_done: dict | None = None,
             kml_paths: dict | None = None, parallel: bool = True,
             n_workers: int | None = None, global_method: str = "ga",
             seed: int | None = None, max_iters: int = MAX_ITERS,
             window_pct: float = CONVERGENCE_WINDOW_PCT) -> dict:
             
    base = tier1.guess_baseline(routes, car, solar_providers, wind_providers,
                                start_soc_pct, start_day=start_day, 
                                dist_done_km=dist_done_km, elapsed_s=elapsed_s,
                                cs_taken=cs_taken, loops_done=loops_done,
                                kml_paths=kml_paths)
    plans = base["day_plans"]
    s_center = base["s0_pct"].copy()
    if not base["feasible"]:
        logger.warning("Tier 1 baseline infeasible from start_soc=%.1f%%", start_soc_pct)
        # Fill NaN entries so Tier 2 gets usable start-SOC guesses.
        # Strategy: linearly interpolate from last known SOC down to 50%.
        nan_mask = ~np.isfinite(s_center)
        if nan_mask.any():
            first_nan = int(np.argmax(nan_mask))
            last_known = float(s_center[first_nan - 1]) if first_nan > 0 else start_soc_pct
            n_nan = int(nan_mask.sum())
            # Spread from last_known toward 50% (mid-range guess)
            target = max(car.soc_min_pct + 10.0, 50.0)
            fill = np.linspace(last_known, target, n_nan + 2)[1:-1]
            s_center[nan_mask] = fill
            logger.info("Filled %d NaN SOC entries: %s",
                        n_nan, [round(float(x), 1) for x in s_center])

    history = []
    result = None
    
    for it in range(max_iters):
        alpha_floors = _alpha_floors_from_traj(
            np.append(s_center, car.soc_min_pct)[: len(plans) + 1], car, start_day) \
            if result is None else _alpha_floors_from_traj(result["s1_pct"], car, start_day)

        per_day = tier2.sample_all_days(
            routes, car, solar_providers, wind_providers, s_center, plans,
            alpha_floors, start_day=start_day, dist_done_km=dist_done_km, 
            elapsed_s=elapsed_s, cs_taken=cs_taken, loops_done=loops_done,
            parallel=parallel, n_workers=n_workers, global_method=global_method, seed=seed)

        result = tier3.allocate(car, solar_providers, per_day, plans, start_soc_pct, start_day=start_day)
        s_refined = result["s1_pct"]

        drift = np.nanmax(np.abs(s_refined[start_day:] - s_center[start_day:]))
        history.append(dict(iteration=it, drift_pct=float(drift),
                            total_km=result["total_distance_km"], feasible=result["feasible"]))
        logger.info("trust-region it=%d drift=%.2f%% total=%.1f km feasible=%s",
                    it, drift, result["total_distance_km"], result["feasible"])

        if not result["feasible"]:
            break
        if drift <= window_pct:
            return _package(result, plans, car, converged=True, iterations=it + 1, history=history, start_day=start_day)

        s_center = s_refined.copy()   

    return _package(result, plans, car, converged=False, iterations=len(history), history=history, start_day=start_day)

def _package(result: dict, plans: list, car: CarState, *, converged: bool,
             iterations: int, history: list, start_day: int) -> dict:
    if result is None:
        return dict(loop_plan={}, s_start_pct=None, total_distance_km=0.0,
                    converged=False, iterations=iterations, feasible=False,
                    alpha_day_pct={}, history=history, start_day_index=start_day)
    s1 = result["s1_pct"]
    alpha_day = {d: float(s1[d]) for d in range(start_day, len(plans)) if np.isfinite(s1[d])}
    return dict(
        loop_plan=result["loop_plan"],
        s_start_pct=s1,
        total_distance_km=result["total_distance_km"],
        converged=converged,
        iterations=iterations,
        feasible=result["feasible"],
        alpha_day_pct=alpha_day,
        history=history,
        start_day_index=start_day
    )

# ===========================================================================
# 1. Final Velocity Extraction (Post-Convergence)
# ===========================================================================

def extract_final_profiles(routes: list, base_car: CarState, solar_providers: dict, 
                           wind_providers: dict, optimize_result: dict, 
                           overrides: dict | None = None) -> dict:
                           
    if not optimize_result.get("feasible"):
        logger.error("Cannot extract profiles from an infeasible result.")
        return {}

    car = dataclasses.replace(base_car, **(overrides or {}))
    final_race_plan = {}
    
    start_socs = optimize_result["s_start_pct"]
    loop_plan = optimize_result["loop_plan"]
    alpha_floors = optimize_result["alpha_day_pct"]
    start_day = optimize_result.get("start_day_index", 0)
    
    for d in range(start_day, len(routes)):
        s_start = start_socs[d]
        d_loops = loop_plan.get(d, {})
        alpha_next = alpha_floors.get(d, car.soc_min_pct)
        
        solar_provider = solar_providers.get(d)
        wind_provider = wind_providers.get(d)
        
        nom_plan = _get_day_plan(d)
        loops_committed = []
        for name, km in nom_plan.loops:
            count = d_loops.get(name, 0)
            loops_committed.extend([(name, km)] * count)
            
        logger.info(f"Extracting exact profile for Day {d}...")
        
        res = singleday.solve(
            route=routes[d] if routes else None,
            car=car,
            solar_provider=solar_provider,
            wind_provider=wind_provider,
            day_index=d,
            start_soc_pct=s_start,
            alpha_next_day_pct=alpha_next,
            loops_committed=loops_committed
        )
        
        final_race_plan[d] = {
            "start_soc_pct": s_start,
            "loops_committed": loops_committed,
            "end_soc_pct": res.get("final_soc_pct"),
            "velocity_profile_kmh": res.get("v_kmh"),
            "time_array_s": res.get("t_s"),          
            "distance_array_m": res.get("x_m")       
        }
        
    return final_race_plan

# ===========================================================================
# 2. Fast Intra-Day Replan (Model Predictive Control / L2 Only)
# ===========================================================================

def fast_replan_today(route, base_car: CarState, solar_providers: dict, wind_providers: dict,
                      cur_soc_pct: float, target_end_soc_pct: float,
                      planned_loops: dict, cur_day: int, dist_done_km: float, 
                      elapsed_s: float, cs_taken: bool = False, 
                      loops_done: dict | None = None, 
                      overrides: dict | None = None, **kwargs) -> dict:
                      
    car = dataclasses.replace(base_car, **(overrides or {}))
    solar_provider = solar_providers.get(cur_day)
    wind_provider = wind_providers.get(cur_day)
    
    nom_plan = _get_day_plan(cur_day)
    adjusted_plan = _adjust_plan_for_today(nom_plan, dist_done_km, loops_done)
    
    loops_done_dict = loops_done or {}
    loops_committed = []
    
    for name, km in nom_plan.loops:
        total_planned = planned_loops.get(name, 0)
        already_done = loops_done_dict.get(name, 0)
        remaining = max(0, total_planned - already_done)
        loops_committed.extend([(name, km)] * remaining)
        
    logger.info(f"Fast Replan Day {cur_day}: {dist_done_km:.1f}km done. "
                f"Targeting end SOC {target_end_soc_pct:.1f}%")

    res = singleday.solve(
        route=route,
        car=car,
        solar_provider=solar_provider,
        wind_provider=wind_provider,
        day_index=cur_day,
        start_soc_pct=cur_soc_pct,
        alpha_next_day_pct=target_end_soc_pct,
        loops_committed=loops_committed,
        dist_done_km=dist_done_km,   
        elapsed_s=elapsed_s,         
        cs_taken=cs_taken,
        **kwargs
    )
    
    if res.get("final_soc_pct", -float('inf')) < car.soc_min_pct:
        logger.warning("Fast Replan Failed: Cannot hit target SOC. Macro-Replan required.")
        return {"feasible": False}
        
    return {
        "feasible": True,
        "velocity_profile_kmh": res.get("v_kmh"),
        "predicted_end_soc_pct": res.get("final_soc_pct")
    }

if __name__ == "__main__":
    import os
    import glob
    import logging
    import json
    from configs.car_config import CarState
    from core.solar import HourlyJSONSolarProvider, GaussianProvider
    from core.wind import HourlyJSONWindProvider, ConstantWindProvider
    from core.route import Route  # <-- Ensure this is imported!

    logging.basicConfig(level=logging.INFO)

    # 1. Initialize Car
    car = CarState() 

    # 2. Setup Directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.abspath(os.path.join(current_dir, "..", "data", "solar"))
    kml_dir = os.path.abspath(os.path.join(current_dir, "..", "data", "shaded"))
    save_dir = os.path.abspath(os.path.join(current_dir, "..", "data", "processed"))

    # 3. Load Routes, Weather, and KMLs
    routes = {}
    solar_providers = {}
    wind_providers = {}
    kml_paths = {}
    
    for d in range(8):
# --- ROUTE LOADING ---
        route_search = os.path.join(save_dir, f"*Day {d+1}*.save")
        day_route_files = glob.glob(route_search)
        
        # Specific filter for Day 3 (Index 2)
        if d == 2:
            day_route_files = [f for f in day_route_files if "prahlad" in f.lower()]
            
        if day_route_files:
            import pandas as pd
            
            # Smart-sort files so the car drives them in the right physical order
            def route_sort_key(filepath):
                name = filepath.lower()
                if "stage 1" in name: return 1
                if "loop" in name: return 2
                if "stage 2" in name: return 3
                return 4
            
            day_route_files = sorted(day_route_files, key=route_sort_key)
            
            day_dfs = []
            current_dist_offset = 0.0
            
            for filepath in day_route_files:
                logger.info(f"Day {d+1}: Loading part -> {os.path.basename(filepath)}")
                
                # 1. Open and parse the JSON .save file
                with open(filepath, 'r', encoding='utf-8') as f:
                    route_data = json.load(f)
                
                # 2. Extract arrays
                prof = route_data["profile"]
                dists = [x * 1000.0 for x in prof["Distance"]]
                slopes = prof["Gradient"]
                bearings = prof.get("Headings", [0.0] * len(dists))
                alts = prof.get("Altitude", [0.0] * len(dists))       
                lats = [c[0] for c in prof["Coordinates"]]
                lons = [c[1] for c in prof["Coordinates"]]
                v_maxs = [v / 3.6 for v in prof["SpeedLimit"]]
                
                # 3. Truncate off-by-one errors
                min_len = min(len(dists), len(slopes), len(bearings), len(alts), len(lats), len(lons), len(v_maxs))
                
                # 4. Build the chunk DataFrame
                part_df = pd.DataFrame({
                    "distance_m": dists[:min_len],
                    "elevation_m": alts[:min_len],          
                    "slope_pct": slopes[:min_len],
                    "bearing_deg": bearings[:min_len],
                    "lat": lats[:min_len],
                    "lon": lons[:min_len],
                    "v_max_ms": v_maxs[:min_len],
                    "curvature_1pm": 0.0,                   
                    "circle_id": 0,                         
                    "red_flag_trailer": False,              
                    "control_stop": False,                  
                    "day": d + 1,                           
                    "seg_type": "stage"
                })
                
                # 5. Shift distances so the segments connect seamlessly
                part_df["distance_m"] += current_dist_offset
                current_dist_offset = part_df["distance_m"].max()
                
                day_dfs.append(part_df)
            
            # Combine all chunks into one massive route for the day
            route_df = pd.concat(day_dfs, ignore_index=True)
            routes[d] = Route(route_df) 
        else:
            logger.warning(f"Route .save for Day {d+1} not found. Using flat fallback.")
            routes[d] = None

        # --- WEATHER LOADING ---
        weather_search = os.path.join(json_dir, f"*Day {d+1}*.json")
        day_weather_files = glob.glob(weather_search)
        
        if day_weather_files:
            logger.info(f"Day {d+1}: loading weather from {day_weather_files[0]}")
            solar_providers[d] = HourlyJSONSolarProvider(day_weather_files[0], routes.get(d))
            wind_providers[d] = HourlyJSONWindProvider(day_weather_files[0], routes.get(d))
        else:
            logger.warning(f"Day {d+1}: no weather JSON found, using GaussianProvider/zero-wind fallback")
            solar_providers[d] = GaussianProvider()
            wind_providers[d] = ConstantWindProvider(speed_ms=0.0, dir_deg_from=0.0)

        # --- KML TRAILERING LOADING ---
        kml_search = os.path.join(kml_dir, f"*Day {d+1}*.kml")
        day_kml_files = glob.glob(kml_search)
        
        if day_kml_files:
            kml_paths[d] = day_kml_files[0]
        else:
            kml_paths[d] = None

    # 4. Launch the Multi-Day Optimizer
    logger.info("Starting Trust-Region Convergence Loop (Sequential Mode)...")
    
    result = optimize(
        routes=routes,           
        car=car,
        solar_providers=solar_providers,
        wind_providers=wind_providers,
        kml_paths=kml_paths,       
        start_soc_pct=100.0,
        start_day=0,
        parallel=False,            
        max_iters=2                
    )

    print("\n" + "="*50)
    print(f"OPTIMIZATION COMPLETE")
    print(f"Converged: {result.get('converged')}")
    print(f"Feasible:  {result.get('feasible')}")
    print(f"Iterations: {result.get('iterations')}")
    print(f"Total Expected Distance: {result.get('total_distance_km', 0):.1f} km")
    print("="*50)
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
        floors[d] = float(s1_pct[d + 1]) if d + 1 < n else car.soc_min_pct
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
    from core.route import Route
    from analysis.results_io import (
        SegmentBoundary, record_segment_boundary, save_variant_results,
    )
 
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
 
    # ── 1. Initialize Car ─────────────────────────────────────────────────
    car = CarState()
 
    # ── 2. Setup Directories ──────────────────────────────────────────────
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(current_dir, ".."))
    json_dir = os.path.join(model_dir, "data", "solar")
    kml_dir = os.path.join(model_dir, "data", "shaded")
    save_dir = os.path.join(model_dir, "data", "processed")
    results_dir = os.path.join(model_dir, "data", "results")
 
    # ── 3. Discover Day 3 variants ────────────────────────────────────────
    all_day3_files = glob.glob(os.path.join(save_dir, "*Day 3*", "*.save")) + \
                     glob.glob(os.path.join(save_dir, "*Day 3*.save")) + \
                     glob.glob(os.path.join(save_dir, "Day 3 probables*", "*.save")) + \
                     glob.glob(os.path.join(save_dir, "*day 3*.save"))
    # Deduplicate
    all_day3_files = list(set(all_day3_files))
 
    # Group by variant name
    day3_variants = {}
    for f in all_day3_files:
        lower = f.lower()
        if "prahlad" in lower:
            day3_variants.setdefault("prahlad", []).append(f)
        elif "aryaman" in lower:
            day3_variants.setdefault("aryaman", []).append(f)
        elif "diyaansh" in lower:
            day3_variants.setdefault("diyaansh", []).append(f)
        else:
            day3_variants.setdefault("unknown", []).append(f)
 
    if not day3_variants:
        logger.warning("No Day 3 route files found — running without Day 3")
        day3_variants = {"default": []}
 
    logger.info(f"Day 3 variants discovered: {list(day3_variants.keys())}")
 
    # ── 4. Route sort helper ──────────────────────────────────────────────
    def route_sort_key(filepath):
        name = filepath.lower()
        if "stage 1" in name or "stage1" in name:
            return 1
        if "loop" in name:
            return 2
        if "stage 2" in name or "stage2" in name:
            return 3
        return 4
 
    # ── 5. Load a single day's route + record boundaries ──────────────────
    def load_day_route(day_index, route_files):
        """Load and concatenate .save files for one day.
 
        Returns (Route, list[SegmentBoundary]) or (None, []).
        """
        if not route_files:
            return None, []
 
        route_files = sorted(route_files, key=route_sort_key)
        day_dfs = []
        boundaries = []
        current_dist_offset = 0.0
 
        for filepath in route_files:
            logger.info(f"  Day {day_index + 1}: loading {os.path.basename(filepath)}")
 
            with open(filepath, 'r', encoding='utf-8') as f:
                route_data = json.load(f)
 
            prof = route_data["profile"]
            dists = [x * 1000.0 for x in prof["Distance"]]
            slopes = prof["Gradient"]
            bearings = prof.get("Headings", [0.0] * len(dists))
            alts = prof.get("Altitude", [0.0] * len(dists))
            lats = [c[0] for c in prof["Coordinates"]]
            lons = [c[1] for c in prof["Coordinates"]]
            v_maxs = [v / 3.6 for v in prof["SpeedLimit"]]
 
            min_len = min(len(dists), len(slopes), len(bearings),
                          len(alts), len(lats), len(lons), len(v_maxs))
 
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
                "day": day_index + 1,
                "seg_type": "stage",
            })
 
            seg_start_m = current_dist_offset
            part_df["distance_m"] += current_dist_offset
            current_dist_offset = float(part_df["distance_m"].max())
 
            # Record where this .save file lives in the concatenated route
            boundaries.append(record_segment_boundary(
                filepath, seg_start_m, current_dist_offset
            ))
 
            day_dfs.append(part_df)
 
        route_df = pd.concat(day_dfs, ignore_index=True)
        return Route(route_df), boundaries
 
    # ── 6. Load non-Day-3 routes (shared across all variants) ─────────────
    import pandas as pd
 
    shared_routes = {}
    shared_boundaries = {}
    shared_solar = {}
    shared_wind = {}
    kml_paths = {}
 
    for d in range(8):
        if d == 2:  # Day 3 — loaded per variant
            continue
 
        route_search = os.path.join(save_dir, f"*Day {d + 1}*.save")
        day_route_files = glob.glob(route_search)
 
        route, boundaries = load_day_route(d, day_route_files)
        shared_routes[d] = route
        shared_boundaries[d] = boundaries
 
        if route is None:
            logger.warning(f"Day {d + 1}: no route .save files found, flat fallback")
 
        # Weather: load ALL matching JSONs for the day
        weather_search = os.path.join(json_dir, f"*Day {d + 1}*.json")
        day_weather_files = sorted(glob.glob(weather_search))
 
        if day_weather_files:
            logger.info(f"Day {d + 1}: loading {len(day_weather_files)} weather files")
            shared_solar[d] = HourlyJSONSolarProvider(day_weather_files, route)
            shared_wind[d] = HourlyJSONWindProvider(day_weather_files, route)
        else:
            logger.warning(f"Day {d + 1}: no weather JSON, using fallback")
            shared_solar[d] = GaussianProvider()
            shared_wind[d] = ConstantWindProvider(speed_ms=0.0, dir_deg_from=0.0)
 
        # KML trailering
        kml_search = os.path.join(kml_dir, f"*Day {d + 1}*.kml")
        day_kml_files = glob.glob(kml_search)
        kml_paths[d] = day_kml_files[0] if day_kml_files else None
 
    # ── 7. Run optimizer per Day 3 variant ────────────────────────────────
    for variant_name, day3_files in day3_variants.items():
        print(f"\n{'=' * 60}")
        print(f"VARIANT: {variant_name}")
        print(f"{'=' * 60}")
 
        # Build the full routes dict for this variant
        routes = dict(shared_routes)        # copy shared days
        day_boundaries = dict(shared_boundaries)
        solar_providers = dict(shared_solar)
        wind_providers = dict(shared_wind)
 
        # Load Day 3 for this variant
        day3_files_sorted = sorted(day3_files, key=route_sort_key)
        route_d3, bounds_d3 = load_day_route(2, day3_files_sorted)
        routes[2] = route_d3
        day_boundaries[2] = bounds_d3
 
        # Day 3 weather
        day3_weather = []
        for f in day3_files_sorted:
            # Corresponding solar JSON: same basename but in solar/ with _historical_solar.json
            base = os.path.basename(f).replace('.save', '')
            solar_json = os.path.join(json_dir, base + "_historical_solar.json")
            if os.path.exists(solar_json):
                day3_weather.append(solar_json)
 
        if day3_weather:
            solar_providers[2] = HourlyJSONSolarProvider(day3_weather, route_d3)
            wind_providers[2] = HourlyJSONWindProvider(day3_weather, route_d3)
        else:
            logger.warning(f"Day 3 ({variant_name}): no weather JSON, using fallback")
            solar_providers[2] = GaussianProvider()
            wind_providers[2] = ConstantWindProvider(speed_ms=0.0, dir_deg_from=0.0)
 
        # ── 7a. Optimize ─────────────────────────────────────────────────
        logger.info(f"Starting Trust-Region for variant '{variant_name}'...")
 
        result = optimize(
            routes=routes,
            car=car,
            solar_providers=solar_providers,
            wind_providers=wind_providers,
            kml_paths=kml_paths,
            start_soc_pct=100.0,
            start_day=0,
            parallel=False,
            max_iters=2,
        )
 
        print(f"\nConverged: {result.get('converged')}")
        print(f"Feasible:  {result.get('feasible')}")
        print(f"Iterations: {result.get('iterations')}")
        print(f"Total Distance: {result.get('total_distance_km', 0):.1f} km")
 
        if result.get("s_start_pct") is not None:
            soc = result["s_start_pct"]
            print(f"SOC trajectory: {[round(float(s), 1) for s in soc]}")
 
        # ── 7b. Extract final profiles ────────────────────────────────────
        if not result.get("feasible"):
            logger.error(f"Variant '{variant_name}' infeasible — skipping CSV output")
            continue
 
        logger.info(f"Extracting final velocity profiles for '{variant_name}'...")
        solve_outputs = extract_final_profiles(
            routes=list(routes.values()) if isinstance(routes, dict) else routes,
            base_car=car,
            solar_providers=solar_providers,
            wind_providers=wind_providers,
            optimize_result=result,
        )
 
        # extract_final_profiles returns {day_index: {..., velocity_profile_kmh, ...}}
        # We need to convert to singleday.solve() format for build_run_csv
        formatted_outputs = {}
        for d, profile in solve_outputs.items():
            v_kmh = profile.get("velocity_profile_kmh")
            if v_kmh is None:
                continue
            # Reconstruct seg_start_m from the route and velocity count
            n_segs = len(v_kmh)
            from configs import solver_config as SCFG
            seg_start_m = np.arange(n_segs) * SCFG.CONTROL_SEGMENT_M
            formatted_outputs[d] = {
                "v_kmh": v_kmh,
                "seg_start_m": seg_start_m,
            }
 
        # ── 7c. Save per-stage CSVs + summary ─────────────────────────────
        logger.info(f"Saving per-stage CSVs for '{variant_name}'...")
 
        # Convert routes to the right format for save_variant_results
        variant_dir = save_variant_results(
            variant_name=variant_name,
            results_base_dir=results_dir,
            routes=routes,
            car=car,
            solar_providers=solar_providers,
            wind_providers=wind_providers,
            optimize_result=result,
            day_boundaries=day_boundaries,
            solve_outputs=formatted_outputs,
        )
 
        print(f"\nResults saved to: {variant_dir}")
        print(f"  Per-stage CSVs in day1/ through day8/ subfolders")
        print(f"  Summary at: {variant_dir}/summary.json")
 
    # ── 8. Final summary across all variants ──────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"ALL VARIANTS COMPLETE")
    print(f"Results root: {results_dir}")
    for vname in day3_variants:
        vdir = os.path.join(results_dir, vname)
        if os.path.exists(vdir):
            csv_count = sum(1 for _, _, files in os.walk(vdir)
                           for f in files if f.endswith('.csv'))
            print(f"  {vname}/: {csv_count} stage CSVs")
    print(f"{'=' * 60}")
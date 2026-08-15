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
        # NaN from infeasible Tier 1 → fall back to minimum SOC floor
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
        # Fill NaN entries so Tier 2 gets usable start-SOC guesses
        nan_mask = ~np.isfinite(s_center)
        if nan_mask.any():
            first_nan = int(np.argmax(nan_mask))
            last_known = float(s_center[first_nan - 1]) if first_nan > 0 else start_soc_pct
            n_nan = int(nan_mask.sum())
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
    import pandas as pd
    import numpy as np
    from configs.car_config import CarState
    from core.solar import HourlyJSONSolarProvider, GaussianProvider
    from core.wind import HourlyJSONWindProvider, ConstantWindProvider
    from core.route import Route

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger("main")

    # ------------------------------------------------------------------ #
    # 1.  Car + directory setup
    # ------------------------------------------------------------------ #
    car = CarState()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir  = os.path.abspath(os.path.join(current_dir, "..", "data", "solar"))
    kml_dir   = os.path.abspath(os.path.join(current_dir, "..", "data", "shaded"))
    save_dir  = os.path.abspath(os.path.join(current_dir, "..", "data", "processed"))

    # ------------------------------------------------------------------ #
    # 2.  Helpers
    # ------------------------------------------------------------------ #
    def _route_sort_key(filepath):
        name = filepath.lower()
        if "stage 1" in name: return 1
        if "loop"    in name: return 2
        if "stage 2" in name: return 3
        return 4

    def _load_route(route_files, day_num):
        """Parse .save files into a single Route for one day."""
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

    # Minimum acceptable daytime-avg GHI for September in South Africa.
    # Historical data below this is almost certainly an anomalous cloudy day
    # and should be blended with the clear-sky Gaussian fallback.
    MIN_DAYTIME_GHI_WM2 = 400.0
    GAUSSIAN_BLEND_FRAC  = 0.5   # blend weight for Gaussian when JSON is below floor

    class _FlooredSolarProvider:
        """Wraps a JSON solar provider with a GaussianProvider floor.

        If the JSON day has anomalously low solar (avg daytime GHI < threshold),
        ghi_wm2 returns  max(json_ghi, blend_frac * gaussian_ghi).
        This prevents a single bad-weather historical year from making the
        entire race infeasible while still respecting realistic cloudy-day data.
        """
        def __init__(self, json_prov, blend_frac: float = 0.5):
            self._json = json_prov
            self._gauss = GaussianProvider()
            self._blend = blend_frac

        def ghi_wm2(self, t_s: float, x_m: float = 0.0) -> float:
            j = self._json.ghi_wm2(t_s, x_m)
            g = self._gauss.ghi_wm2(t_s, x_m)
            return max(j, self._blend * g)

    def _check_daytime_avg_ghi(weather_files):
        """Quick scan: average GHI during race hours (8-17h) across all nodes."""
        all_ghi = []
        for fp in weather_files:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for node in data:
                ghi = node["historical_weather"]["hourly"]["shortwave_radiation"]
                all_ghi.append(np.mean(ghi[8:17]))  # hours 8-16 (race window)
        return float(np.mean(all_ghi)) if all_ghi else 0.0

    def _load_weather(weather_files, route, day_num=None):
        """Load ALL weather JSONs for a day into solar + wind providers.

        If daytime-avg GHI is anomalously low (< MIN_DAYTIME_GHI_WM2),
        the solar provider is wrapped with a GaussianProvider floor.
        """
        if not weather_files:
            return GaussianProvider(), ConstantWindProvider(0.0, 0.0)

        json_prov = HourlyJSONSolarProvider(weather_files, route)
        wind_prov = HourlyJSONWindProvider(weather_files, route)

        avg_ghi = _check_daytime_avg_ghi(weather_files)
        if avg_ghi < MIN_DAYTIME_GHI_WM2:
            logger.warning(
                "Day %s: daytime avg GHI = %.0f W/m² (below %.0f floor) — "
                "blending with Gaussian clear-sky fallback",
                day_num or "?", avg_ghi, MIN_DAYTIME_GHI_WM2)
            json_prov = _FlooredSolarProvider(json_prov, GAUSSIAN_BLEND_FRAC)

        return json_prov, wind_prov

    # ------------------------------------------------------------------ #
    # 3.  Load Days 1-8 (Day 3 handled separately below)
    # ------------------------------------------------------------------ #
    routes         = {}
    solar_providers = {}
    wind_providers  = {}
    kml_paths       = {}

    for d in range(8):
        day_num = d + 1

        # --- Route ---
        route_files = glob.glob(os.path.join(save_dir, f"*Day {day_num}*.save"))
        if d == 2:
            # Day 3 loaded separately as multi-variant below
            continue
        if route_files:
            routes[d] = _load_route(route_files, day_num)
            logger.info("Day %d: route loaded (%d files)", day_num, len(route_files))
        else:
            logger.warning("Day %d: no route .save files found — flat fallback", day_num)
            routes[d] = None

        # --- Weather (ALL files for this day) ---
        weather_files = glob.glob(os.path.join(json_dir, f"*Day {day_num}*.json"))
        solar_providers[d], wind_providers[d] = _load_weather(weather_files, routes.get(d), day_num)
        if weather_files:
            logger.info("Day %d: loaded %d weather JSONs", day_num, len(weather_files))

        # --- KML trailering ---
        kml_files = glob.glob(os.path.join(kml_dir, f"*Day {day_num}*.kml"))
        kml_paths[d] = kml_files[0] if kml_files else None

    # ------------------------------------------------------------------ #
    # 4.  Day 3 multi-variant discovery
    # ------------------------------------------------------------------ #
    day3_route_files = glob.glob(os.path.join(save_dir, "*Day 3*probables*.save")) or \
                       glob.glob(os.path.join(save_dir, "*Day 3*.save"))
    day3_weather_files = glob.glob(os.path.join(json_dir, "*Day 3*.json"))

    # Group by variant name (e.g. "Prahlad", "Aryaman")
    day3_variants = {}
    for fp in day3_route_files:
        bn = os.path.basename(fp).lower()
        if "prahlad" in bn:
            day3_variants.setdefault("prahlad", []).append(fp)
        elif "aryaman" in bn:
            day3_variants.setdefault("aryaman", []).append(fp)
        else:
            day3_variants.setdefault("unknown", []).append(fp)

    if not day3_variants:
        logger.warning("Day 3: no route files found — flat fallback for single variant")
        day3_variants = {"flat_fallback": []}

    logger.info("Day 3 variants discovered: %s", list(day3_variants.keys()))

    # Match weather files to each variant
    day3_variant_weather = {}
    for vname in day3_variants:
        matched = [f for f in day3_weather_files if vname.lower() in os.path.basename(f).lower()]
        if not matched:
            matched = day3_weather_files  # fallback: use all available
        day3_variant_weather[vname] = matched

    # ------------------------------------------------------------------ #
    # 5.  Run optimizer for each Day 3 variant
    # ------------------------------------------------------------------ #
    all_results = {}

    for variant_name, variant_route_files in day3_variants.items():
        logger.info("=" * 60)
        logger.info("RUNNING VARIANT: Day 3 = %s", variant_name)
        logger.info("=" * 60)

        # Build Day 3 route for this variant
        if variant_route_files:
            routes[2] = _load_route(variant_route_files, 3)
            logger.info("Day 3 [%s]: route loaded (%d files)", variant_name, len(variant_route_files))
        else:
            routes[2] = None

        # Build Day 3 weather for this variant
        weather_files = day3_variant_weather.get(variant_name, [])
        solar_providers[2], wind_providers[2] = _load_weather(weather_files, routes.get(2), 3)

        # KML for Day 3
        kml_files = glob.glob(os.path.join(kml_dir, "*Day 3*.kml"))
        kml_paths[2] = kml_files[0] if kml_files else None

        # --- Run trust-region optimizer ---
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

        # --- Extract final speed profiles if feasible ---
        profiles = {}
        if result.get("feasible"):
            profiles = extract_final_profiles(
                routes, car, solar_providers, wind_providers, result)

        all_results[variant_name] = {"result": result, "profiles": profiles}

    # ------------------------------------------------------------------ #
    # 6.  Print summary for all variants
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE — ALL DAY 3 VARIANTS")
    print("=" * 70)

    for variant_name, data in all_results.items():
        result   = data["result"]
        profiles = data["profiles"]

        print(f"\n{'─' * 70}")
        print(f"  Day 3 variant: {variant_name.upper()}")
        print(f"{'─' * 70}")
        print(f"  Converged:  {result.get('converged')}")
        print(f"  Feasible:   {result.get('feasible')}")
        print(f"  Iterations: {result.get('iterations')}")
        print(f"  Total km:   {result.get('total_distance_km', 0):.1f}")

        if result.get("alpha_day_pct"):
            print("  SOC trajectory:")
            for d_idx in sorted(result["alpha_day_pct"]):
                print(f"    Day {d_idx + 1} start: {result['alpha_day_pct'][d_idx]:.1f}%")

        if result.get("loop_plan"):
            print("  Loop plan:")
            for d_idx in sorted(result["loop_plan"]):
                lp = result["loop_plan"][d_idx]
                if lp:
                    print(f"    Day {d_idx + 1}: {lp}")

        if profiles:
            print("  Speed profiles (km/h per segment):")
            for d_idx in sorted(profiles):
                p = profiles[d_idx]
                v = p.get("velocity_profile_kmh")
                if v is not None:
                    v_arr = np.asarray(v)
                    print(f"    Day {d_idx + 1}: "
                          f"SOC {p['start_soc_pct']:.1f}% → {p['end_soc_pct']:.1f}% | "
                          f"v = [{', '.join(f'{x:.1f}' for x in v_arr)}]")

    print("\n" + "=" * 70)
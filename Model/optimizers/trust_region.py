"""
optimizers/hierarchical/trust_region.py — coarse-to-fine driver loop.
"""

from __future__ import annotations

# ── BLAS thread pinning (must run before numpy/scipy import) ──────────────
# scipy.optimize.minimize / differential_evolution call BLAS (LAPACK)
# routines that default to OMP's auto thread count. Under Tier 2's
# thread-per-day parallelism that oversubscribes cores and stalls every
# solve. Pin each process to one BLAS thread so worker threads actually
# parallelize instead of thrashing. Keep this BEFORE any numpy/scipy
# import in this module (and ideally in the entrypoint too).
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")
del _os, _v

import logging
import numpy as np
import dataclasses
import glob
import pandas as pd

from configs import race_config as rc
from configs.car_config import CarState
from optimizers import singleday
from .tier1 import _get_day_plan
from . import tier1, tier2, tier3, _threads
from .tier1 import _adjust_plan_for_today

logger = logging.getLogger(__name__)

MAX_ITERS = 4                          
CONVERGENCE_WINDOW_PCT = tier2.SAMPLE_WINDOW_PCT

def _alpha_floors_from_traj(s1_pct: np.ndarray, car: CarState, start_day: int,
                            overnight_gains: dict | None = None) -> dict:
    """Compute end-of-day SOC floors for Tier 2.

    Tier 2's job is to BUILD SURROGATES by sampling the full feasible space.
    The alpha passed to singleday.solve is only a minimum-SOC guard — it
    should be as lenient as possible so combos aren't rejected prematurely.

    Tier 1's uniform-speed DP produces end-of-day SOCs (e.g. 26.8%) that
    singleday.solve — with variable speeds, turn caps and tighter time
    budgets — often can't replicate (27977s vs 27000s allowed). Using those
    as alpha floors causes ALL combos to be marked infeasible, leaving
    surrogates empty for 7/8 days.

    Fix: use soc_min_pct for every day. Tier 3 handles inter-day SOC
    allocation using the populated surrogates.
    """
    n = len(s1_pct)
    floors = {}
    for d in range(start_day, n):
        floors[d] = car.soc_min_pct
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

def _trailered_km_by_day(routes):
    """Return trailered km from the exact mask consumed by forward_sim."""
    out = {}
    for d, route in enumerate(routes or []):
        if route is None or not hasattr(route, "df") or len(route.df) < 2:
            out[d] = 0.0
            continue
        dist = route.df["distance_m"].to_numpy(dtype=float)
        mask = route.df["red_flag_trailer"].to_numpy(dtype=bool)
        out[d] = float(np.sum(np.diff(dist)[mask[:-1]])) / 1000.0
    return out


def optimize(routes: list, car: CarState, solar_providers: dict, wind_providers: dict,
             start_soc_pct: float = 100.0, *, start_day: int = 0,
             dist_done_km: float = 0.0, elapsed_s: float = 0.0,
                          cs_taken: bool = False, loops_done: dict | None = None,
             kml_paths: dict | None = None,
             loop_geoms_by_day: dict | None = None,
             plans_override: list | None = None, parallel: bool = True,
             n_workers: int | None = None, global_method: str = "ga",
             seed: int | None = None, max_iters: int = MAX_ITERS,
             window_pct: float = CONVERGENCE_WINDOW_PCT,
             tier1_baseline: dict | None = None) -> dict:

        # Audit fix: normalize the worker count exactly once, in one place.
    # tier2.sample_all_days re-clamps through the same worker_cap, so an
    # explicit n_workers is honored end-to-end (previously it was passed to
    # sample_all_days and silently swallowed by its **kwargs, so every run
    # used a hardcoded min(n_days, os.cpu_count()) instead of the caller's
    # parallel= setting).
    n_workers = _threads.worker_cap(n_workers)

    if tier1_baseline is not None:
        base = tier1_baseline
        logger.info("Using pre-computed Tier 1 baseline (skipping recomputation)")
    else:
        base = tier1.guess_baseline(routes, car, solar_providers, wind_providers,
                                    start_soc_pct, start_day=start_day,
                                    dist_done_km=dist_done_km, elapsed_s=elapsed_s,
                                    cs_taken=cs_taken, loops_done=loops_done,
                                    kml_paths=kml_paths,
                                    plans_override=plans_override)
    plans = plans_override if plans_override is not None else base["day_plans"]
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
        # Alpha floors = soc_min_pct for all days.  Tier 2 samples the full
        # feasible space; Tier 3 handles multi-day SOC allocation.
        alpha_floors = _alpha_floors_from_traj(
            np.append(s_center, car.soc_min_pct)[: len(plans) + 1], car, start_day)

        logger.info("Tier2 it=%d alpha_floors (end-of-day targets): %s",
                    it, {d+1: f"{v:.1f}%" for d, v in alpha_floors.items()})
        logger.info("Tier2 it=%d s_center (start-of-day): %s",
                    it, [f"D{d+1}={v:.1f}%" for d, v in enumerate(s_center) if np.isfinite(v)])

        per_day = tier2.sample_all_days(
            routes, car, solar_providers, wind_providers, s_center, plans,
            alpha_floors, start_day=start_day, dist_done_km=dist_done_km,
            elapsed_s=elapsed_s, cs_taken=cs_taken, loops_done=loops_done,
            loop_geoms_by_day=loop_geoms_by_day,
            parallel=parallel, n_workers=n_workers, global_method=global_method, seed=seed)

        # Tier 2 surrogate diagnostic + Tier 1 fallback for empty days
        for d in range(start_day, len(plans)):
            dd = per_day.get(d)
            if dd is None:
                logger.warning("Tier2 Day %d: no data returned at all!", d + 1)
                dd = {}
            surr = dd.get("surrogates", {})
            n_keys = len(surr) if isinstance(surr, dict) else 0
            if n_keys > 0:
                logger.info("Tier2 Day %d: %d surrogate(s) populated", d + 1, n_keys)
            else:
                # ── Tier 1 fallback surrogate ──────────────────────────
                # singleday.solve failed for every combo at all offsets.
                # Build a synthetic linear surrogate from Tier 1's coarse
                # energy model so Tier 3 can still allocate this day.
                logger.warning("Tier2 Day %d: EMPTY — building Tier 1 fallback surrogate", d + 1)
                s0 = float(s_center[d])
                # Estimate end-of-day SOC from Tier 1 trajectory
                if d + 1 < len(s_center) and np.isfinite(s_center[d + 1]):
                    sp = solar_providers.get(d)
                    gain = tier1.overnight_soc_gain(car, sp, d) if sp else 0.0
                    end_est = float(s_center[d + 1]) - gain
                else:
                    end_est = car.soc_min_pct
                end_est = max(end_est, car.soc_min_pct)
                # Linear surrogate:  predict(s) = end_est + 1.0*(s - s0)
                reps = (0,) * len(plans[d].loops)
                fb = tier2.LinearSurrogate(
                    a=float(end_est), b=1.0, s0=s0, loop_km=0.0, reps=reps,
                    soc_lo=car.soc_min_pct, soc_hi=car.soc_max_pct)
                per_day[d] = {"surrogates": {reps: fb}, "s0_pct": s0,
                              "n_l2_solves": 0}
                logger.info("Tier2 Day %d: fallback a=%.1f b=%.2f s0=%.1f",
                            d + 1, end_est, 1.0, s0)

        trailered_km_by_day = _trailered_km_by_day(routes)
        if any(km > 0.0 for km in trailered_km_by_day.values()):
            logger.info("Tier3 credited trailered km by day: %s",
                        {d + 1: round(km, 1) for d, km in trailered_km_by_day.items() if km > 0.0})
        result = tier3.allocate(
            car, solar_providers, per_day, plans, start_soc_pct,
            start_day=start_day, trailered_km_by_day=trailered_km_by_day)
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
                           overrides: dict | None = None,
                           loop_geoms_by_day: dict | None = None,
                           plans_override: list | None = None) -> dict:
                           
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
        
        if plans_override is not None:
            if d >= len(plans_override):
                raise ValueError(f"plans_override has no plan for day index {d}")
            nom_plan = plans_override[d]
        else:
            nom_plan = _get_day_plan(d)
        loops_committed = []
        for name, km in nom_plan.loops:
            count = d_loops.get(name, 0)
            loops_committed.extend([(name, km)] * count)
            
        logger.info(f"Extracting exact profile for Day {d}...")
        
                        # Real per-loop geometry for this day so committed reps are actually
        # simulated (loaded once in __main__ via _load_loop_geometries).
        loop_geoms = (loop_geoms_by_day or {}).get(d)

        res = singleday.solve(
            route=routes[d] if routes else None,
            car=car,
            solar_provider=solar_provider,
            wind_provider=wind_provider,
            day_index=d,
            start_soc_pct=s_start,
            alpha_next_day_pct=alpha_next,
            loops_committed=loops_committed,
            loop_geoms=loop_geoms
        )
        
        final_race_plan[d] = {
            "start_soc_pct": s_start,
            "loops_committed": loops_committed,
            "end_soc_pct": res.get("final_soc_pct"),
            "velocity_profile_kmh": res.get("v_kmh"),
            "time_array_s": res.get("t_s"),
            "distance_array_m": res.get("x_m"),
            "total_time_s": res.get("total_time_s"),  # actual drive duration (not absolute)
            "trailered_km": res.get("trailered_km", 0.0),
            "trailered_substeps": res.get("trailered_substeps", 0),
            "driven_km": res.get("driven_km", 0.0),
            "motor_energy_wh": res.get("motor_energy_wh", 0.0),
            "solar_energy_wh": res.get("solar_energy_wh", 0.0),
        }
        
    return final_race_plan

# ===========================================================================
# 2. Fast Intra-Day Replan (Model Predictive Control / L2 Only)
# ===========================================================================

def fast_replan_today(route, base_car: CarState, solar_providers: dict, wind_providers: dict,
                      cur_soc_pct: float, target_end_soc_pct: float,
                      planned_loops: dict, cur_day: int, dist_done_km: float,
                      elapsed_s: float,              cs_taken: bool = False, loops_done: dict | None = None, 
                      loop_geoms: dict | None = None,
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
        loop_geoms=loop_geoms,
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
        if "stage 2" in name: return 3
        # Dedicated loop files (contain "loop" but NOT "stage") sort last
        if "loop" in name and "stage" not in name: return 2
        return 4

    def _is_loop_file(filepath):
        """True only for DEDICATED loop files, not stage files with 'loop' in a place name."""
        bn = os.path.basename(filepath).lower()
        return "loop" in bn and "stage" not in bn

    def _seg_type_for_file(filepath):
        """Real per-file stage tag matching core/route.py's documented schema
        ('stage1' | 'loop_<name>' | 'stage2'). Every row previously got the
        literal string "stage" regardless of source file, which collapsed
        the trailered-mask's "whole STAGE if any point flagged" rule into an
        unintended "whole DAY" rule, and made it impossible to target the
        Day 7 Stage 2 / Day 8 Stage 1 trailering override at just one stage."""
        name = filepath.lower()
        if "stage 1" in name:
            return "stage1"
        if "stage 2" in name:
            return "stage2"
        if "loop" in name:
            base = os.path.splitext(os.path.basename(filepath))[0]
            return f"loop_{base}"
        return "stage1"  # single-file days (no explicit stage split)

    def _parse_route_file(filepath, day_num, seg_type=None):
        """Parse ONE route .save file into a DataFrame using the same schema
        as _load_route's per-file block — factored out so dedicated loop
        geometry files reuse the exact same parsing. seg_type defaults to the
        real per-file tag (stage1/stage2/loop_<base>) via _seg_type_for_file;
        loop geometry's own seg_type is irrelevant because singleday's
        _splice_loops() re-tags every spliced leg as 'loop_<name>'."""
        with open(filepath, "r", encoding="utf-8") as f:
            route_data = json.load(f)
        prof = route_data["profile"]
        dists    = [x * 1000.0 for x in prof["Distance"]]
        slopes   = prof["Gradient"]
        bearings = prof.get("Headings",   [0.0] * len(dists))
        alts     = prof.get("Altitude",   [0.0] * len(dists))
        lats     = [c[0] for c in prof["Coordinates"]]
        lons     = [c[1] for c in prof["Coordinates"]]
        # Clamp speed limits: some route data has erroneous 5 km/h entries.
        # Floor at 30 km/h (highway minimum) to avoid artificial slowdowns.
        v_maxs   = [max(v, 30.0) / 3.6 for v in prof["SpeedLimit"]]
        ml = min(len(dists), len(slopes), len(bearings),
                 len(alts), len(lats), len(lons), len(v_maxs))
        return pd.DataFrame({
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
            "seg_type":        seg_type or _seg_type_for_file(filepath),
        })

    def _load_route(route_files, day_num):
        """Parse .save files into a single Route for one day.

        Dedicated loop .save files are EXCLUDED from the base route.
        A file is a loop file only if it contains 'loop' but NOT 'stage'
        in its name — this avoids false positives on South African place
        names like 'Loopspruit' that appear in stage file names.
        """
        route_files = sorted(route_files, key=_route_sort_key)
        # Remove dedicated loop files only.
        route_files = [f for f in route_files if not _is_loop_file(f)]
        day_dfs = []
        offset = 0.0
        for filepath in route_files:
            part_df = _parse_route_file(filepath, day_num)
            part_df["distance_m"] += offset
            offset = part_df["distance_m"].max()
            day_dfs.append(part_df)
        return Route(pd.concat(day_dfs, ignore_index=True))

    def _loop_name_tokens(loop_name):
        """Searchable city/place tokens for a plan loop name, generic words
        dropped. 'postmasburg_loop2' -> {'postmasburg'};
        'blind_loop_placeholder' -> set() (no usable token — falls through
        to closest-distance matching)."""
        drop = {"loop", "blind", "placeholder", "day", "stage"}
        tokens = set()
        for part in loop_name.replace("-", "_").split("_"):
            part = part.strip().lower()
            if part and part not in drop:
                tokens.add(part)
        return tokens

    def _norm_loop_filename(filepath):
        """Lowercase, punctuation-stripped basename used for token matching."""
        bn = os.path.splitext(os.path.basename(filepath))[0].lower()
        return "".join(ch for ch in bn if ch.isalnum())

    def _load_loop_geometries(route_files, plan, day_num):
        """Match each plan loop name to a DEDICATED loop .save file for this
        day and return {loop_name: DataFrame} of its real terrain/speed-limit
        geometry. Previously these loop files were discarded entirely and
        every committed rep got a flat synthetic leg — this is the piece that
        finally feeds real loop physics into Tier 2 and the final profiles.

        Matching is one-to-one (each loop file is consumed at most once):
          1. Preferred: token match — every non-generic word of the plan loop
             name must appear in the file's basename (e.g. 'springbok_loop'
             -> any file containing 'springbok'). Among token hits the file
             whose real length is closest to the plan's nominal km wins.
          2. Fallback: closest real driving distance to the nominal km among
             still-unused loop files (covers blind/placeholder names like
             Day 3's 'blind_loop_placeholder', whose real loop file is
             variant-specific).

        Plan loops left unmatched are logged — singleday's _splice_loops()
        falls back to a flat synthetic leg for them.
        """
        loop_files = [f for f in route_files if _is_loop_file(f)]
        if not loop_files or not (plan.loops or []):
            return {}

        parsed = []
        for fp in loop_files:
            try:
                df = _parse_route_file(fp, day_num)
            except Exception as exc:
                logger.warning("Day %d: skipping unparseable loop file %s (%s)",
                               day_num, os.path.basename(fp), exc)
                continue
            if df is None or len(df) == 0:
                continue
            parsed.append((fp, df, float(df["distance_m"].max()) / 1000.0))

        if not parsed:
            return {}

        geoms: dict = {}
        used: set = set()

        for name, nom_km in (plan.loops or []):
            cands = [(fp, df, km) for (fp, df, km) in parsed if fp not in used]
            if not cands:
                break
            match = None
            tokens = _loop_name_tokens(name)
            if tokens:
                hits = [(fp, df, km) for (fp, df, km) in cands
                        if all(tok in _norm_loop_filename(fp) for tok in tokens)]
                if hits:
                    match = min(hits, key=lambda t: abs(t[2] - nom_km))
            if match is None:
                # Closest real distance among all still-unused loop files.
                match = min(cands, key=lambda t: abs(t[2] - nom_km))
            fp, df, km = match
            used.add(fp)
            geoms[name] = df
            token_hit = tokens and all(tok in _norm_loop_filename(fp) for tok in tokens)
            logger.info("Day %d: loop '%s' (nom %.1f km) -> geometry %s (%.1f km%s)",
                        day_num, name, nom_km, os.path.basename(fp), km,
                        "" if token_hit else " [closest-distance fallback]")

        for name, nom_km in (plan.loops or []):
            if name not in geoms:
                logger.warning("Day %d: loop '%s' (%.1f km) has NO geometry file "
                               "— singleday will use a flat synthetic leg",
                               day_num, name, nom_km)
        return geoms

    def _apply_trailered_mask(route, kml_paths_for_day: dict | None, day_index: int):
        """Compute the real trailered mask (KML + hardcoded Day7/Day8 ground
        truth + whole-stage rounding) and write it onto route so the REAL
        optimizer (singleday.solve -> forward_sim -> route.red_flag_at)
        actually sees it. Previously this mask was only ever computed inside
        tier1.py's own coarse energy diagnostic and printed to the log --
        route.df["red_flag_trailer"] stayed False for every point on every
        day, so the report's TRAILER column and forward_sim's trailering-aware
        physics never fired, even on days confirmed fully trailered."""
        if route is None:
            return route
        mask = tier1.compute_trailered_mask_full(route, kml_paths_for_day, day_index)
        if mask.any():
            route.set_red_flag_mask(mask)
        return route

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

        # Audit hot-loop fix: forward_sim now resolves weather-node indices
        # per segment (node_index_array) and evaluates GHI per substep via
        # ghi_wm2_at_node. Re-expose both so this wrapper does NOT silently
        # degrade the vectorized path back to one KDTree query per substep.
        def node_index_array(self, x_m):
            return self._json.node_index_array(x_m)

        def ghi_wm2_at_node(self, t_s: float, node_index: int) -> float:
            j = self._json.ghi_wm2_at_node(t_s, node_index)
            # Gaussian floor is position-independent (x_m ignored upstream).
            g = self._gauss.ghi_wm2(t_s, 0.0)
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
    # Real per-day loop geometry {day_index: {loop_name: DataFrame}}, loaded
    # from each day's dedicated loop .save files and threaded down to
    # Tier 2 / singleday.solve so committed loop reps are actually simulated.
    loop_geoms_by_day = {}

    for d in range(8):
        day_num = d + 1

        # --- Route ---
        route_files = glob.glob(os.path.join(save_dir, f"*Day {day_num}*.save"))
        if d == 2:
            # Day 3 loaded separately as multi-variant below
            continue
        if route_files:
            # Log which files are stage vs loop (dedicated loops excluded by _load_route)
            stage_files = [f for f in route_files if not _is_loop_file(f)]
            loop_files  = [f for f in route_files if _is_loop_file(f)]
            routes[d] = _load_route(route_files, day_num)
            logger.info("Day %d: route loaded (%d stage files, %d loop files excluded)",
                        day_num, len(stage_files), len(loop_files))
        else:
            logger.warning("Day %d: no route .save files found — flat fallback", day_num)
            routes[d] = None

        # --- Real loop geometry (dedicated loop .save files) ---
        # Day 3 (d==2) is variant-specific and handled in section 5.
        if d != 2:
            loop_geoms_by_day[d] = _load_loop_geometries(
                route_files, _get_day_plan(d), day_num)

        # --- Weather (ALL files for this day) ---
        weather_files = glob.glob(os.path.join(json_dir, f"*Day {day_num}*.json"))
        solar_providers[d], wind_providers[d] = _load_weather(weather_files, routes.get(d), day_num)
        if weather_files:
            logger.info("Day %d: loaded %d weather JSONs", day_num, len(weather_files))

        # --- KML trailering ---
        kml_files = glob.glob(os.path.join(kml_dir, f"*Day {day_num}*.kml"))
        kml_paths[d] = kml_files[0] if kml_files else None
        _apply_trailered_mask(routes.get(d), kml_paths, d)

    # ------------------------------------------------------------------ #
    def _build_day3_variant_plan(variant_route_files, variant_name, day_num=3):
        """Build Day 3 strictly from the released variant route files.

        Aryaman: Stage 2 + Loop, with NO Stage 1.
        Prahlad: Stage 1 + Stage 2 + Loop.
        """
        stage1_files, stage2_files, loop_files = [], [], []

        for fp in variant_route_files:
            bn = os.path.basename(fp).lower()
            if _is_loop_file(fp):
                loop_files.append(fp)
            elif "stage 1" in bn:
                stage1_files.append(fp)
            elif "stage 2" in bn:
                stage2_files.append(fp)

        variant = variant_name.lower()

        if variant == "aryaman":
            if stage1_files:
                raise ValueError(
                    "Day 3 Aryaman unexpectedly contains a Stage 1 file: "
                    f"{stage1_files}"
                )
            if len(stage2_files) != 1:
                raise ValueError(
                    "Day 3 Aryaman must contain exactly one Stage 2 file; "
                    f"found {len(stage2_files)}: {stage2_files}"
                )
            if len(loop_files) != 1:
                raise ValueError(
                    "Day 3 Aryaman must contain exactly one loop file; "
                    f"found {len(loop_files)}: {loop_files}"
                )
            stage1_km = 0.0
        elif variant == "prahlad":
            if len(stage1_files) != 1:
                raise ValueError(
                    "Day 3 Prahlad must contain exactly one Stage 1 file; "
                    f"found {len(stage1_files)}: {stage1_files}"
                )
            if len(stage2_files) != 1:
                raise ValueError(
                    "Day 3 Prahlad must contain exactly one Stage 2 file; "
                    f"found {len(stage2_files)}: {stage2_files}"
                )
            if len(loop_files) != 1:
                raise ValueError(
                    "Day 3 Prahlad must contain exactly one loop file; "
                    f"found {len(loop_files)}: {loop_files}"
                )
            stage1_km = _parse_route_file(
                stage1_files[0], day_num
            )["distance_m"].max() / 1000.0
        else:
            raise ValueError(f"Unknown Day 3 variant '{variant_name}'")

        stage2_km = _parse_route_file(
            stage2_files[0], day_num
        )["distance_m"].max() / 1000.0
        loop_km = _parse_route_file(
            loop_files[0], day_num
        )["distance_m"].max() / 1000.0

        plan = tier1._DayPlan(
            stage1_km=float(stage1_km),
            stage2_km=float(stage2_km),
            loops=(("day3_loop", float(loop_km)),),
        )
        logger.info(
            "Day 3 [%s] plan: Stage1=%.1f km, Stage2=%.1f km, Loop=%.1f km",
            variant_name, plan.stage1_km, plan.stage2_km, loop_km
        )
        return plan

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
    # 5.  Run optimizer once per Day-3 variant, each with its OWN Tier 1
    #     baseline
    # ------------------------------------------------------------------ #
    # Tier 1's guess_baseline bakes in the full day-0..7 route + weather set
    # (the Day-3 variant is swapped into routes[2] before it runs), and the
    # resulting day_plans / s0_pct / feasible flag are all Day-3-dependent.
    # Reusing one shared baseline computed from the FIRST variant for every
    # other variant is wrong: different Day 3 route -> different energy
    # balance -> different s_center -> Tier 2 samples the wrong SOC window
    # and Tier 3 chains days around a Day-3-specific policy. Recompute per
    # variant so each result is self-consistent (cost: one ~10-min Tier 1
    # pass per variant).
    kml_files_d3 = glob.glob(os.path.join(kml_dir, "*Day 3*.kml"))

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

        # Day 3 is variant-specific: load ITS OWN dedicated loop geometry so
        # committed 'blind_loop_placeholder' reps simulate the real variant
        # loop terrain instead of a flat synthetic leg.
        day3_plan = _build_day3_variant_plan(
            variant_route_files, variant_name, day_num=3
        )
        loop_geoms_by_day[2] = _load_loop_geometries(
            variant_route_files, day3_plan, 3
        )

        # Same variant-specific plans are used by Tier 1, Tier 2/Tier 3,
        # final profile extraction and reporting.
        variant_plans = [_get_day_plan(d) for d in range(8) if d != 2]
        variant_plans.insert(2, day3_plan)
        if len(variant_plans) != rc.N_RACE_DAYS:
            raise RuntimeError(
                f"Day 3 variant plan has {len(variant_plans)} days; expected {rc.N_RACE_DAYS}"
            )

        # Build Day 3 weather for this variant
        weather_files = day3_variant_weather.get(variant_name, [])
        solar_providers[2], wind_providers[2] = _load_weather(weather_files, routes.get(2), 3)

                # KML for Day 3
        kml_paths[2] = kml_files_d3[0] if kml_files_d3 else None
        _apply_trailered_mask(routes.get(2), kml_paths, 2)

        # --- Tier 1 baseline for THIS variant (Day 3 already swapped in) ---
        logger.info("Computing Tier 1 baseline for variant '%s'...", variant_name)
        baseline = tier1.guess_baseline(
            routes, car, solar_providers, wind_providers, 100.0,
            start_day=0, kml_paths=kml_paths,
            plans_override=variant_plans)

        # --- Run trust-region optimizer (per-variant Tier 1 baseline) ---
        result = optimize(
            routes=routes,
            car=car,
            solar_providers=solar_providers,
            wind_providers=wind_providers,
            kml_paths=kml_paths,
            loop_geoms_by_day=loop_geoms_by_day,
            plans_override=variant_plans,
            start_soc_pct=100.0,
            start_day=0,
            parallel=True,
            max_iters=4,
            tier1_baseline=baseline,
        )

        # --- Extract final speed profiles if feasible ---
        profiles = {}
        if result.get("feasible"):
            profiles = extract_final_profiles(
                routes, car, solar_providers, wind_providers, result,
                loop_geoms_by_day=loop_geoms_by_day,
                plans_override=variant_plans)

        all_results[variant_name] = {"result": result, "profiles": profiles, "plans": variant_plans}

    # ------------------------------------------------------------------ #
    # 6.  Helpers for per-day strategy plan
    # ------------------------------------------------------------------ #
    # Day 3 is variant-dependent; reporting uses each variant's own plans.
    DAY_NAMES = {
        0: "Johannesburg → Rustenburg",
        1: "Rustenburg → Zeerust",
        2: "Zeerust → Vryburg",          # Day 3 (variant-dependent)
        3: "Vryburg → Upington",
        4: "Upington → Springbok",
        5: "Springbok → Van Rhynsdorp",
        6: "Van Rhynsdorp → Clanwilliam",
        7: "Clanwilliam → Cape Town",
    }

    def _hms(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h:02d}:{m:02d}"

    def _clock(abs_s: float) -> str:
        """Absolute seconds from midnight → HH:MM clock string."""
        return _hms(abs_s)

    # Car constants for output calculations.
    # FIX: these getattr() calls previously used attribute names that don't
    # exist on CarState ('solar_area', 'solar_eff', 'battery_capacity_wh'),
    # so every call silently fell through to the hardcoded default. That
    # made array_area_m2/battery figures coincidentally correct (they matched
    # their fallback defaults) but array_efficiency is really 0.22, not the
    # fallback 0.18 -- every "Solar input" figure was ~18% understated.
    _SOLAR_AREA  = car.array_area_m2
    _SOLAR_EFF   = car.array_efficiency
    _BATT_CAP_WH = car.battery_nominal_wh

    def _real_day_km(d_idx, plan, lp, profiles) -> float:
        """Distance that actually counts toward the race total: DRIVEN km
        only, excluding any trailered km (SR's asterisk rule ranks trailered
        teams below all non-trailered teams regardless of distance, so
        trailered km must never be added to the reported total). Prefers the
        real optimizer output (profiles[d_idx]["driven_km"]) when available;
        falls back to the static route-notes plan estimate otherwise (e.g.
        infeasible days with no profile).

        Note this also fixes a separate bug: _get_day_plan() returns a
        STATIC config-table estimate (rc.DAY_ROUTE_NOTES) that, for Day 3
        (full-blind day), is a fixed 230.0/0.0 fallback regardless of which
        Day-3 route variant was actually loaded and driven -- this made
        Day 3's printed Distance identical across variants even when
        genuinely different route files were used. profiles[d_idx] comes
        from a real singleday.solve()/forward_sim run on the actual loaded
        Route for that variant, so it doesn't have this problem.
        """
        if profiles and d_idx in profiles:
            driven_km = profiles[d_idx].get("driven_km")
            if driven_km is not None and driven_km > 0:
                return float(driven_km)
        loop_km = sum(cnt * km for (name, km) in plan.loops
                      for cnt in [lp.get(name, 0)]) if lp else 0.0
        return plan.stage1_km + plan.stage2_km + loop_km

    def _estimate_solar_input_wh(solar_prov, day_index: int) -> float:
        """Rough total solar energy over the race window (Wh)."""
        if solar_prov is None:
            return 0.0
        t0 = rc.day_start_time_s(day_index)
        t1 = rc.day_finish_time_s(day_index)
        dt = 300.0  # 5-min steps
        total = 0.0
        t = t0
        while t < t1:
            ghi = solar_prov.ghi_wm2(t, 0.0)
            total += ghi * _SOLAR_AREA * _SOLAR_EFF * dt / 3600.0  # Wh, not joules
            t += dt
        return total

    def _estimate_stall_points(start_soc, end_soc, distance_km, solar_wh,
                               battery_cap=_BATT_CAP_WH):
        """Flag if SOC likely dips below 25% mid-day (rough linear check)."""
        drain_wh = (start_soc - end_soc) / 100.0 * battery_cap
        motor_wh = drain_wh + solar_wh
        if motor_wh <= 0:
            return "None — net energy positive"
        # Approximate lowest SOC (assumes linear drain, solar ramps up mid-day)
        # Worst case: first 25% of distance has low solar but full motor drain
        early_drain_pct = 0.25 * drain_wh / battery_cap * 100
        early_solar_pct = 0.10 * solar_wh / battery_cap * 100  # ~10% of solar in first quarter
        min_soc_est = start_soc - early_drain_pct + early_solar_pct
        if min_soc_est < 25.0:
            return f"RISK — estimated min SOC ≈ {min_soc_est:.0f}% in first quarter"
        return "None"

    # ------------------------------------------------------------------ #
    # 7.  Print full strategy plan for all variants
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE — ALL DAY 3 VARIANTS")
    print("=" * 70)

    for variant_name, data in all_results.items():
        result   = data["result"]
        profiles = data["profiles"]
        plans    = data["plans"]

        print(f"\n{'━' * 70}")
        print(f"  Day 3 variant: {variant_name.upper()}")
        print(f"{'━' * 70}")
        print(f"  Converged:  {result.get('converged')}")
        print(f"  Feasible:   {result.get('feasible')}")
        print(f"  Iterations: {result.get('iterations')}")
        # NOTE: real total (driven km only, trailered days excluded) is
        # printed after the per-day table below, once _real_day_km has run
        # for each day — the DP-internal total_distance_km here is the
        # allocator's own planning estimate (static plan tables, doesn't
        # know about trailering) and is kept only as a cross-check.
        print(f"  Total Expected Distance (DP planning estimate): {result.get('total_distance_km', 0):.1f} km")

        if not result.get("feasible"):
            print("  ⚠ Infeasible — no per-day plan available")
            continue

        loop_plan = result.get("loop_plan", {})
        s_start = result.get("s_start_pct")

        print(f"\n  {'─' * 80}")
        print(f"  {'DAY':>5} │ {'ROUTE':<30} │ {'KM':>7} │ {'LOOPS':>5} │ {'SOC START→END':>14} │ {'ETA':>5} │ {'TRAILER':>8}")
        print(f"  {'─' * 80}")

        _real_total_km = 0.0
        _total_trailered_km = 0.0
        for d_idx in range(8):
            plan = plans[d_idx]
            lp = loop_plan.get(d_idx, {})
            n_loops = sum(lp.values()) if lp else 0
            loop_km = sum(cnt * km for (name, km) in plan.loops
                          for cnt in [lp.get(name, 0)])
            day_km = _real_day_km(d_idx, plan, lp, profiles)
            route_name = DAY_NAMES.get(d_idx, f"Day {d_idx + 1}")

            # SOC
            soc_start = float(s_start[d_idx]) if s_start is not None and np.isfinite(s_start[d_idx]) else 0.0

            # End SOC + ETA from profiles if available
            if profiles and d_idx in profiles:
                p = profiles[d_idx]
                soc_end = p.get("end_soc_pct", 0.0)
                # Use total_time_s (drive duration) if available; fall back to
                # deriving from t_s (but t_s is ABSOLUTE clock time, not duration).
                drive_time = p.get("total_time_s")
                if drive_time is None:
                    t_arr = p.get("time_array_s")
                    if t_arr is not None and hasattr(t_arr, '__iter__'):
                        # t_s is absolute → duration = max(t_s) - day_start
                        drive_time = float(max(t_arr)) - rc.day_start_time_s(d_idx)
                    else:
                        drive_time = 0.0
            else:
                soc_end = 0.0
                drive_time = 0.0

            t_start_abs = rc.day_start_time_s(d_idx)
            eta_abs = t_start_abs + drive_time
            eta_str = _clock(eta_abs) if drive_time > 0 else "—"

            trailer_km = 0.0
            if profiles and d_idx in profiles:
                trailer_km = profiles[d_idx].get("trailered_km", 0.0) or 0.0
            trailer_str = f"{trailer_km:.1f}km" if trailer_km > 0 else "—"
            _real_total_km += day_km
            _total_trailered_km += trailer_km
            print(f"  {d_idx+1:>5} │ {route_name:<30} │ {day_km:>6.1f} │ {n_loops:>5} │ "
                  f"{soc_start:>5.1f}% → {soc_end:>5.1f}% │ {eta_str:>5} │ {trailer_str:>8}")

        print(f"  {'─' * 80}")
        print(f"  Total Expected Distance (driven, trailered km excluded): {_real_total_km:.1f} km")
        if _total_trailered_km > 0:
            print(f"  Total trailered (NOT counted toward distance above): {_total_trailered_km:.1f} km")

        # ── Detailed per-day strategy ──
        print(f"\n  {'═' * 66}")
        print("  DETAILED DAILY STRATEGY")
        print(f"  {'═' * 66}")

        for d_idx in range(8):
            plan = plans[d_idx]
            lp = loop_plan.get(d_idx, {})
            n_loops = sum(lp.values()) if lp else 0
            loop_km = sum(cnt * km for (name, km) in plan.loops
                          for cnt in [lp.get(name, 0)])
            day_km = _real_day_km(d_idx, plan, lp, profiles)
            route_name = DAY_NAMES.get(d_idx, f"Day {d_idx + 1}")
            soc_start = float(s_start[d_idx]) if s_start is not None and np.isfinite(s_start[d_idx]) else 0.0

            print(f"\n  ── Day {d_idx + 1}: {route_name} ──")

            # 1. Km planned + loops
            loop_detail = ", ".join(f"{name}×{cnt}" for name, cnt in lp.items()) if lp else "none"
            print(f"  1) Distance: {day_km:.1f} km | Loops: {loop_detail}")

            # 2. End of day SOC
            if profiles and d_idx in profiles:
                p = profiles[d_idx]
                soc_end = p.get("end_soc_pct", 0.0) or 0.0
                # total_time_s is the actual drive duration (not absolute clock)
                total_time = p.get("total_time_s")
                if total_time is None:
                    t_arr = p.get("time_array_s")
                    if t_arr is not None and hasattr(t_arr, '__iter__'):
                        total_time = float(max(t_arr)) - rc.day_start_time_s(d_idx)
                    else:
                        total_time = 0.0
                v_arr = p.get("velocity_profile_kmh")
            else:
                soc_end = 0.0
                total_time = 0.0
                v_arr = None
            print(f"  2) End-of-day SOC: {soc_end:.1f}%")

            # 3. SOC curve (key checkpoints)
            soc_drain_pct = soc_start - soc_end
            solar_wh = float((profiles.get(d_idx, {}) or {}).get("solar_energy_wh", 0.0) or 0.0)
            drain_wh = soc_drain_pct / 100.0 * _BATT_CAP_WH
            # Prefer the REAL simulated motor energy (integrated directly
            # from physics every substep in forward_sim.py) over the old
            # circular back-out (motor_wh = drain_wh + solar_wh), which was
            # mathematically forced to equal solar_wh whenever drain_wh
            # happened to be 0 -- exactly what happens on every day that
            # clips at the SOC ceiling (100% -> 100%), making the printed
            # figure meaningless on precisely those days.
            _real_motor_wh = (profiles.get(d_idx, {}).get("motor_energy_wh")
                               if profiles else None)
            motor_wh = _real_motor_wh if _real_motor_wh else (drain_wh + solar_wh)
            # Approximate hourly SOC (linear interpolation)
            t_window = rc.day_finish_time_s(d_idx) - rc.day_start_time_s(d_idx)
            n_hours = max(1, int(t_window / 3600))
            hourly_soc = [soc_start - (soc_drain_pct * h / n_hours) for h in range(n_hours + 1)]
            hourly_labels = [_clock(rc.day_start_time_s(d_idx) + h * 3600) for h in range(n_hours + 1)]
            soc_str = " → ".join(f"{s:.0f}%" for s in hourly_soc[::max(1, len(hourly_soc)//5)])
            print(f"  3) SOC curve (approx): {soc_str}")

            # 4. Solar input
            print(f"  4) Solar input: {solar_wh:.0f} Wh total")

            # 5. Energy consumption
            print(f"  5) Motor energy: {motor_wh:.0f} Wh | Battery drain: {drain_wh:.0f} Wh ({soc_drain_pct:.1f}%)")

            # 6. Stall risk
            stall = _estimate_stall_points(soc_start, soc_end, day_km, solar_wh)
            print(f"  6) Stall risk: {stall}")

            # 7. Early start strategy (6-8 AM solar charging)
            if soc_start < 40.0:
                print(f"  7) Start strategy: EARLY START recommended — prop car roadside"
                      f" at 6 AM for 2h solar charge before driving (est. +{2*solar_wh/n_hours/_BATT_CAP_WH*100:.0f}% SOC)")
            else:
                print(f"  7) Start strategy: Normal start at {_clock(rc.day_start_time_s(d_idx))}")

            # 8. ETA
            t_start_abs = rc.day_start_time_s(d_idx)
            eta_abs = t_start_abs + total_time
            if total_time > 0:
                print(f"  8) ETA: {_clock(eta_abs)} (drive time {_hms(total_time)})")
            else:
                avg_speed_kmh = day_km / (t_window / 3600) if t_window > 0 else 40.0
                est_time = day_km / avg_speed_kmh * 3600
                print(f"  8) ETA: ~{_clock(t_start_abs + est_time)} (est. drive time {_hms(est_time)})")

            # 9. Trailering
            if profiles and d_idx in profiles:
                t_km = profiles[d_idx].get("trailered_km", 0.0) or 0.0
                t_sub = profiles[d_idx].get("trailered_substeps", 0) or 0
                if t_km > 0:
                    print(f"  9) Trailered: {t_km:.1f} km ({t_sub} segments) — no power in/out")
                else:
                    print(f"  9) Trailered: none")
            else:
                print(f"  9) Trailered: n/a")

            # Speed summary
            if v_arr is not None:
                v_np = np.asarray(v_arr)
                print(f"      Speed: avg {v_np.mean():.1f} km/h, min {v_np.min():.1f}, max {v_np.max():.1f}")

    print("\n" + "=" * 70)

    # ------------------------------------------------------------------ #
    # 8.  Save durable JSON output for dashboard
    # ------------------------------------------------------------------ #
    import datetime as _dt

    output_dir = os.path.abspath(os.path.join(current_dir, "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    for variant_name, data in all_results.items():
        result   = data["result"]
        profiles = data["profiles"]
        plans    = data["plans"]

        json_out = {
            "variant": variant_name,
            "timestamp": _dt.datetime.now().isoformat(),
            "converged": result.get("converged"),
            "feasible": result.get("feasible"),
            "iterations": result.get("iterations"),
            "total_distance_km": result.get("total_distance_km", 0),
            "history": result.get("history", []),
            "days": {},
        }

        if result.get("feasible"):
            loop_plan = result.get("loop_plan", {})
            s_start = result.get("s_start_pct")
            _json_real_total_km = 0.0
            _json_total_trailered_km = 0.0

            for d_idx in range(8):
                plan = plans[d_idx]
                lp = loop_plan.get(d_idx, {})
                n_loops = sum(lp.values()) if lp else 0
                loop_km = sum(cnt * km for (name, km) in plan.loops
                              for cnt in [lp.get(name, 0)])
                day_km = _real_day_km(d_idx, plan, lp, profiles)
                soc_start = float(s_start[d_idx]) if s_start is not None and np.isfinite(s_start[d_idx]) else 0.0

                day_data = {
                    "route": DAY_NAMES.get(d_idx, f"Day {d_idx + 1}"),
                    "distance_km": round(day_km, 1),
                    "stage1_km": round(plan.stage1_km, 1),
                    "stage2_km": round(plan.stage2_km, 1),
                    "loop_km": round(loop_km, 1),
                    "loops": {name: cnt for name, cnt in lp.items()} if lp else {},
                    "n_loops": n_loops,
                    "soc_start_pct": round(soc_start, 1),
                }

                solar_wh = float((profiles.get(d_idx, {}) or {}).get("solar_energy_wh", 0.0) or 0.0)
                day_data["solar_input_wh"] = round(solar_wh, 0)
                day_data["solar_underutil_wh"] = round(float((profiles.get(d_idx, {}) or {}).get("solar_underutil_wh", 0.0) or 0.0), 0)

                if profiles and d_idx in profiles:
                    p = profiles[d_idx]
                    soc_end = p.get("end_soc_pct", 0.0) or 0.0
                    drive_t = p.get("total_time_s")
                    if drive_t is None:
                        t_arr = p.get("time_array_s")
                        if t_arr is not None and hasattr(t_arr, '__iter__'):
                            drive_t = float(max(t_arr)) - rc.day_start_time_s(d_idx)
                        else:
                            drive_t = 0.0
                    v_arr = p.get("velocity_profile_kmh")
                    day_data["soc_end_pct"] = round(soc_end, 1)
                    day_data["drive_time_s"] = round(drive_t, 0)
                    day_data["eta"] = _clock(rc.day_start_time_s(d_idx) + drive_t)
                    if v_arr is not None:
                        v_np = np.asarray(v_arr)
                        day_data["speed_avg_kmh"] = round(float(v_np.mean()), 1)
                        day_data["speed_min_kmh"] = round(float(v_np.min()), 1)
                        day_data["speed_max_kmh"] = round(float(v_np.max()), 1)
                        day_data["velocity_profile_kmh"] = [int(v) for v in v_arr]

                if profiles and d_idx in profiles:
                    day_data["trailered_km"] = round(profiles[d_idx].get("trailered_km", 0.0) or 0.0, 1)
                    day_data["trailered_substeps"] = profiles[d_idx].get("trailered_substeps", 0) or 0

                soc_drain = soc_start - day_data.get("soc_end_pct", soc_start)
                drain_wh = soc_drain / 100.0 * _BATT_CAP_WH
                day_data["battery_drain_wh"] = round(drain_wh, 0)
                day_data["battery_drain_pct"] = round(soc_drain, 1)
                # Prefer real simulated motor energy over the circular
                # back-out (see the print-loop fix above for why).
                _real_motor_wh_json = (profiles.get(d_idx, {}).get("motor_energy_wh")
                                        if profiles else None)
                day_data["motor_energy_wh"] = round(
                    float(_real_motor_wh_json) if _real_motor_wh_json is not None else 0.0, 0)

                _json_real_total_km += day_km
                _json_total_trailered_km += day_data.get("trailered_km", 0.0)

                json_out["days"][str(d_idx + 1)] = day_data

            # Real total: driven km only, trailered km excluded (SR asterisk
            # rule). Overwrite the DP planning estimate that was set above.
            json_out["total_distance_km"] = round(_json_real_total_km, 1)
            json_out["total_trailered_km"] = round(_json_total_trailered_km, 1)
            json_out["total_distance_km_dp_estimate"] = result.get("total_distance_km", 0)

        out_path = os.path.join(output_dir, f"strategy_{variant_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_out, f, indent=2, default=str)
        logger.info("Saved strategy → %s", out_path)

    logger.info("All variant results saved to %s/", output_dir)
"""
optimizers/hierarchical/tier1.py — Tier 1 coarse baseline guesser.
"""

from __future__ import annotations

import dataclasses
import logging
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree

from configs import race_config as rc
from configs import solver_config as sc
from configs.car_config import CarState
from core import physics
from core import solar as solar_core
from core import wind as wind_core
from core.battery import Battery

logger = logging.getLogger(__name__)

TIER1_SAMPLE_M = 500.0
_DEFAULT_REGEN_CAP_W = None

# Energy diagnostic: fires once per day at high SOC to avoid log spam
_energy_diag_logged: set = set()

@dataclasses.dataclass(frozen=True)
class _DayPlan:
    """A day's route in driving order: Stage 1 -> named loop zone(s) -> Stage 2."""
    stage1_km: float
    stage2_km: float
    loops: tuple[tuple[str, float], ...]

def _get_day_plan(day_index: int) -> _DayPlan:
    """
    Return the nominal route plan for a known race day.

    Day 3 is intentionally handled by trust_region.py because it has
    variant-specific released route files:

        Aryaman:
            Stage 1: DOES NOT EXIST
            Stage 2: exists
            Loop: exists

        Prahlad:
            Stage 1: exists
            Stage 2: exists
            Loop: exists

    Therefore Day 3 must never fabricate a 230 km Stage 1 or a generic
    blind-loop placeholder here.
    """
    note = rc.DAY_ROUTE_NOTES[day_index]

    if day_index == rc.FULL_BLIND_DAY_INDEX:
        raise ValueError(
            "Day 3 requires a variant-specific route plan. "
            "Build it from the Aryaman/Prahlad route files before "
            "calling _get_day_plan(2)."
        )

    stage1_km = note["stage1_km"]
    stage2_km = note["stage2_km"]

    if stage1_km is None or stage2_km is None:
        raise ValueError(
            f"Day {day_index + 1} has incomplete route configuration: "
            f"stage1_km={stage1_km}, stage2_km={stage2_km}"
        )

    loops = tuple(note["loops"] or ())

    return _DayPlan(
        float(stage1_km),
        float(stage2_km),
        loops,
    )

def _regen_cap_w(car: CarState) -> float:
    # Single source of truth: core.physics.regen_cap_w. forward_sim
    # (Tier 2 / L2) uses the same helper, so Tier 1's coarse diagnostic and
    # the high-fidelity integrator agree on the regen clamp.
    return physics.regen_cap_w(car)

def _adjust_plan_for_today(plan: _DayPlan, distance_done_km_today: float,
                           loops_completed: dict | None = None) -> _DayPlan:
    if distance_done_km_today <= plan.stage1_km:
        return _DayPlan(plan.stage1_km - distance_done_km_today, plan.stage2_km, plan.loops)
    if loops_completed is not None:
        loop_drive_km = sum(loops_completed.get(name, 0) * km for name, km in plan.loops)
        stage2_done_km = max(0.0, distance_done_km_today - plan.stage1_km - loop_drive_km)
        return _DayPlan(0.0, max(0.0, plan.stage2_km - stage2_done_km), ())
    return _DayPlan(0.0, plan.stage2_km, ())

# Explicit ground-truth trailering override (0-indexed day -> seg_type).
# The KML-derived mask below depends on manually-drawn (False) zones in a
# routeshader KML that may be missing or incomplete for a given day (e.g.
# Day 7 previously showed trailered=0 because no KML zone existed for it,
# not because the day isn't actually trailered). These two are confirmed
# ground truth regardless of what the KML contains.
HARDCODED_TRAILER_STAGES: dict[int, str] = {
    6: "stage2",  # Day 7 (0-indexed 6): Stage 2 is trailered
    7: "stage1",  # Day 8 (0-indexed 7): Stage 1 is trailered
}


def compute_trailered_mask_full(route, kml_paths: dict | None, day_index: int) -> np.ndarray:
    """Parses routeshader KML and maps (False) blocks via KDTree onto EVERY
    point of route.df, at full route resolution (not Tier 1's coarse grid),
    OR'd with HARDCODED_TRAILER_STAGES ground truth for days with confirmed
    trailering that the KML source data may not capture.

    This is the single source of truth for trailering. Callers that only need
    Tier 1's coarser sample grid should use _get_trailered_mask() below, which
    re-indexes this result. Callers that need the REAL fine-grained optimizer
    (Tier 2 / singleday.solve / forward_sim.route.red_flag_at) to be
    trailering-aware must write this result onto route.df["red_flag_trailer"]
    directly — see trust_region.py's route-loading step.
    """
    if route is None:
        return np.zeros(0, dtype=bool)
    route_df = route.df
    n = len(route_df)
    is_trailered = np.zeros(n, dtype=bool)

    if kml_paths and day_index in kml_paths and kml_paths[day_index]:
        kml_path = kml_paths[day_index]
        tree = ET.parse(kml_path)
        root = tree.getroot()
        for el in root.iter():
            if "}" in el.tag: el.tag = el.tag.split("}", 1)[1]

        trailered_coords = []
        for pm in root.findall(".//Placemark"):
            name = pm.find("name")
            if name is not None and "(False)" in name.text:
                ls = pm.find("LineString")
                if ls is not None:
                    for c in ls.find("coordinates").text.strip().split():
                        lon, lat, _ = map(float, c.split(","))
                        trailered_coords.append([lat, lon])

        if trailered_coords:
            latlons = route_df[["lat", "lon"]].to_numpy()
            tree_kd = cKDTree(np.array(trailered_coords))
            dists, _ = tree_kd.query(latlons, distance_upper_bound=0.001)
            is_trailered = dists != float('inf')

    # Ground-truth override: force the whole named stage trailered regardless
    # of whether the KML flagged any point in it.
    forced_stage = HARDCODED_TRAILER_STAGES.get(day_index)
    if forced_stage is not None:
        is_trailered = is_trailered | (route_df["seg_type"].to_numpy() == forced_stage)

    # Enforce the whole stage rule: if any segment in stage is trailered, entire stage is trailered
    for seg in route_df["seg_type"].unique():
        mask = route_df["seg_type"] == seg
        if is_trailered[mask].any():
            is_trailered[mask] = True

    return is_trailered


def _get_trailered_mask(route, x_m_array: np.ndarray, kml_paths: dict | None, day_index: int) -> np.ndarray:
    """Trailered mask re-indexed onto Tier 1's coarser sample grid (x_m_array)."""
    if route is None:
        return np.zeros(len(x_m_array), dtype=bool)
    is_trailered = compute_trailered_mask_full(route, kml_paths, day_index)
    if not is_trailered.any():
        return np.zeros(len(x_m_array), dtype=bool)
    route_df = route.df
    idx = np.clip(np.searchsorted(route_df["distance_m"].to_numpy(), x_m_array), 0, len(route_df)-1)
    return is_trailered[idx]

def _sample_portion(route, start_km: float, dist_km: float):
    if dist_km <= 0: return np.zeros(0), np.zeros(0), np.zeros(0)
    n = max(1, int(round(dist_km * 1000.0 / TIER1_SAMPLE_M)))
    seg_len_m = np.full(n, dist_km * 1000.0 / n)
    if route is None: return np.zeros(n), np.zeros(n), seg_len_m
    edges = np.linspace(start_km * 1000.0, (start_km + dist_km) * 1000.0, n + 1)
    mid = (edges[:-1] + edges[1:]) / 2.0
    slope = np.asarray(route.slope_pct_at(mid), dtype=float)
    bearing = np.asarray(route.bearing_deg_at(mid), dtype=float)
    return slope, bearing, seg_len_m

def _wind_arrays(wind_provider, t_s: np.ndarray, x_m: np.ndarray, bearing_deg: np.ndarray, v_ms: np.ndarray):
    if wind_provider is None or len(x_m) == 0:
        return np.zeros(len(x_m)), np.zeros(len(x_m))
    along = np.empty(len(x_m))
    yaw = np.empty(len(x_m))
    for i in range(len(x_m)):
        spd, dir_from = wind_provider.wind(float(t_s[i]), float(x_m[i]))
        along[i] = wind_core.along_track_ms(spd, dir_from, float(bearing_deg[i]))
        _, yaw[i] = wind_core.relative_wind(float(v_ms[i]), spd, dir_from, float(bearing_deg[i]))
    return along, yaw


def _solar_geom(slope_pct: np.ndarray, bearing_deg: np.ndarray,
                lat_deg: float, day_of_year: int,
                t_solar_s: np.ndarray,
                panel_tilt_base_deg: float = 4.0) -> np.ndarray:
    """Vectorized slope geometry correction.

    Calls solar_core.slope_geometry_factor with all 6 required args:
      (lat_deg, day_of_year, t_solar_s, road_slope_pct, route_bearing_deg, panel_tilt_base_deg)

    Returns array of correction factors (1.0 fallback if the function is
    unavailable or any element errors).
    """
    fn = getattr(solar_core, "slope_geometry_factor", None)
    if fn is None:
        return np.ones_like(slope_pct)
    out = np.ones_like(slope_pct, dtype=float)
    for i in range(len(slope_pct)):
        try:
            out[i] = fn(
                lat_deg,
                day_of_year,
                float(t_solar_s[i]),
                float(slope_pct[i]),
                float(bearing_deg[i]),
                panel_tilt_base_deg,
            )
        except Exception:
            out[i] = 1.0
    return out


def _get_route_lat(route, x_m: np.ndarray) -> float:
    """Extract a representative latitude from the route for solar geometry.
    Falls back to Johannesburg latitude if route has no lat data."""
    if route is None:
        return -26.2  # Johannesburg, South Africa (race start area)
    try:
        # Use the midpoint latitude of the segment
        mid_x = float(x_m[len(x_m) // 2]) if len(x_m) > 0 else 0.0
        lat = route.latlon_at(mid_x)[0]
        return float(lat)
    except Exception:
        return -26.2


def _net_power_vectorised(car: CarState, v_ms: np.ndarray, slope_pct: np.ndarray,
                          ghi: np.ndarray, wind_along_ms: np.ndarray,
                          yaw_deg: np.ndarray, solar_geom: np.ndarray,
                          regen_cap_w: float) -> np.ndarray:
    f = physics.forces(car, v_ms, slope_pct, wind_along_ms=wind_along_ms, yaw_deg=yaw_deg)
    p_mech = (f["drag"] + f["rolling"] + f["gravity"]) * v_ms
    regen_into_pack = np.where(p_mech < 0.0, np.minimum(-p_mech * car.regen_eff, regen_cap_w), 0.0)
    p_electric = np.where(p_mech >= 0.0, p_mech / car.motor_eff, -regen_into_pack)
    p_solar = car.array_area_m2 * car.array_efficiency * ghi * solar_geom
    return p_solar - p_electric - car.p_idle_w

def _build_day_arrays(route, plan: _DayPlan, reps: tuple[int, ...],
                      route_offset_km: float, v_base_ms: float,
                      loop_speed_ms: float, pre_attempt_stop_s: float):
    slopes, bearings, seglens, vels, gaps = [], [], [], [], []

    def _add(slope, bearing, seglen, v_ms, lead_gap_s):
        if len(seglen) == 0: return
        g = np.zeros(len(seglen))
        g[0] = lead_gap_s
        slopes.append(slope); bearings.append(bearing); seglens.append(seglen)
        vels.append(np.full(len(seglen), v_ms)); gaps.append(g)

    if plan.stage1_km > 0:
        s, b, L = _sample_portion(route, route_offset_km, plan.stage1_km)
        _add(s, b, L, v_base_ms, 0.0)

    cursor = route_offset_km + plan.stage1_km
    for i, (_name, km_i) in enumerate(plan.loops):
        n_i = reps[i] if reps else 0
        for _ in range(n_i):
            s, b, L = _sample_portion(route, cursor, km_i)
            _add(s, b, L, loop_speed_ms, pre_attempt_stop_s)
        cursor += km_i

    if plan.stage2_km > 0:
        s, b, L = _sample_portion(route, cursor, plan.stage2_km)
        _add(s, b, L, v_base_ms, 0.0)

    if not seglens:
        z = np.zeros(0)
        return z, z, z, z, z
    return (np.concatenate(slopes), np.concatenate(bearings),
            np.concatenate(seglens), np.concatenate(vels), np.concatenate(gaps))

def evaluate_day(car: CarState, route, plan: _DayPlan, reps: tuple[int, ...],
                 route_offset_km: float, v_base_ms: float, loop_speed_ms: float,
                 pre_attempt_stop_s: float, solar_provider, wind_provider,
                 t0_s: float, start_soc_pct: float, day_index: int,
                 is_today: bool = False, cs_taken: bool = False, kml_paths: dict = None) -> float:

    slope, bearing, seglen_m, v_ms, gap_s = _build_day_arrays(
        route, plan, reps, route_offset_km, v_base_ms, loop_speed_ms, pre_attempt_stop_s)
    if len(seglen_m) == 0:
        return start_soc_pct

    dt_s = seglen_m / np.maximum(v_ms, 0.1)
    t_pt = t0_s + np.cumsum(gap_s) + np.cumsum(dt_s) - dt_s
    x_m = route_offset_km * 1000.0 + np.cumsum(seglen_m) - seglen_m

    ghi = np.array([solar_provider.ghi_wm2(float(t_pt[i]), float(x_m[i])) for i in range(len(x_m))])
    wind_along, yaw = _wind_arrays(wind_provider, t_pt, x_m, bearing, v_ms)

    # --- FIXED: pass all 6 required args to slope_geometry_factor ---
    lat_deg = _get_route_lat(route, x_m)
    race_date = rc.RACE_DAY_DATES[day_index]
    day_of_year = race_date.timetuple().tm_yday
    geom = _solar_geom(slope, bearing, lat_deg, day_of_year, t_pt, car.panel_tilt_base_deg)

    trailered_mask = _get_trailered_mask(route, x_m, kml_paths, day_index)

    regen_cap = _regen_cap_w(car)
    p_net = _net_power_vectorised(car, v_ms, slope, ghi, wind_along, yaw, geom, regen_cap)

    # Mask electrical drive drain if on trailer (keep solar gain)
    # Trailered: NO power flow at all — no motor consumption, no solar
    # charging, no idle draw. The car is inert cargo on a tow truck; SOC
    # must not move during trailered segments (explicit requirement).
    p_net = np.where(trailered_mask, 0.0, p_net)

    energy_wh = p_net * dt_s / 3600.0

    # ---- Diagnostic logging (once per day at ~100% SOC) ----
    if day_index not in _energy_diag_logged and start_soc_pct >= 95.0:
        _energy_diag_logged.add(day_index)
        f = physics.forces(car, v_ms, slope, wind_along_ms=wind_along, yaw_deg=yaw)
        p_mech = (f["drag"] + f["rolling"] + f["gravity"]) * v_ms
        p_solar_raw = car.array_area_m2 * car.array_efficiency * ghi * geom

        # Regen waste from cap
        uncapped_regen = np.where(p_mech < 0, -p_mech * car.regen_eff, 0.0)
        capped_regen = np.where(p_mech < 0, np.minimum(uncapped_regen, regen_cap), 0.0)
        regen_waste_wh = ((uncapped_regen - capped_regen) * dt_s / 3600.0).sum()

        logger.info(
            "Day %d ENERGY DIAG (v=%.1fkm/h, %d segs, %.1fkm): "
            "solar=+%.0fWh idle=-%.0fWh regen_waste=-%.0fWh net=%+.0fWh (%+.1f%% SOC) | "
            "slopes: |avg|=%.1f%% max=%.1f%% | GHI avg=%.0f | trailered=%d | loops=%s",
            day_index + 1, v_base_ms * 3.6, len(slope), seglen_m.sum() / 1000,
            (p_solar_raw * dt_s / 3600).sum(),
            car.p_idle_w * dt_s.sum() / 3600,
            regen_waste_wh,
            energy_wh.sum(), energy_wh.sum() / car.battery_nominal_wh * 100,
            np.abs(slope).mean(), np.abs(slope).max(),
            ghi.mean(), trailered_mask.sum(), reps)

    bat = Battery(car, start_soc_pct)
    for e in energy_wh:
        bat.apply_energy_wh(float(e))

    # Stationary Solar capture (Control Stop + Unplanned)
    base_km = plan.stage1_km + plan.stage2_km
    t_cs = t0_s + ((base_km / 2.0 * 1000.0) / max(v_base_ms, 1e-6) if base_km > 0 else 0.0)
    p_cs = (car.array_area_m2 * car.array_efficiency *
            solar_provider.ghi_wm2(t_cs, route_offset_km + base_km / 2) - car.p_idle_w)

    cs_duration = (0.0 if (is_today and cs_taken) else rc.CONTROL_STOP_DURATION_S)
    total_stop_s = cs_duration + rc.UNPLANNED_STOP_BUDGET_S
    bat.apply_energy_wh(p_cs * total_stop_s / 3600.0)

    # Stationary Solar capture (Loop Stops)
    cur_x = route_offset_km + plan.stage1_km
    t_loop = t0_s + ((plan.stage1_km * 1000.0) / max(v_base_ms, 1e-6) if plan.stage1_km > 0 else 0.0)
    for i, (_, k) in enumerate(plan.loops):
        if reps and reps[i] > 0:
            p_l = (car.array_area_m2 * car.array_efficiency *
                   solar_provider.ghi_wm2(t_loop, cur_x) - car.p_idle_w)
            bat.apply_energy_wh(p_l * reps[i] * pre_attempt_stop_s / 3600.0)
        cur_x += k

    return bat.soc_pct


def overnight_soc_gain(car: CarState, solar_provider, day_index: int) -> float:
    """Morning parc-fermé charging gain (06:30 capture start → next race start).

    SINGLE SOURCE OF TRUTH for the overnight/morning charge across all tiers:
    tier3.allocate and tier2's empty-day fallback both import THIS function, so
    the DP transition, the single-day start SOC (fed via the trajectory) and the
    report all see the same integrated morning energy — no per-tier drift.

    Window: [MORNING_CHARGE_START_S, day_start(day_index+1)] — i.e. 06:30→08:00
    on Days 2-7 (06:30→09:00 into Day 1). Packs are legally unsealed at 06:00
    (BATTERY_UNSEAL_TIME_S) but the strategist's directive (20/08) is to count
    capture only from 06:30, the realistic prop-up time after unseal/setup.

    During this window only the BMS + MPPT charge controller are active, so we
    use a reduced idle draw (PARC_FERME_IDLE_W, default 10 W) rather than the
    full driving electronics. Real GHI(t) is integrated in 15-minute steps for
    accuracy during the steep sunrise ramp, not a single midpoint sample.
    """
    if day_index >= rc.N_RACE_DAYS - 1:
        return 0.0

    # 06:30 capture start (>= 06:00 unseal), per strategist directive.
    t_start = getattr(rc, "MORNING_CHARGE_START_S", rc.BATTERY_UNSEAL_TIME_S)
    t_end   = rc.day_start_time_s(day_index + 1)    # 08:00 (or 09:00 Day 1)
    dur     = max(0.0, t_end - t_start)
    if dur == 0.0:
        return 0.0

    # Reduced idle: only BMS + charge controller, not full driving electronics
    parc_idle_w = getattr(rc, "PARC_FERME_IDLE_W", 10.0)

    # Integrate GHI in 15-min steps (sunrise ramp makes midpoint sampling lossy)
    n_steps = max(1, int(dur / 900.0))
    dt = dur / n_steps
    delta_wh = 0.0
    for i in range(n_steps):
        t_mid = t_start + (i + 0.5) * dt
        ghi = solar_provider.ghi_wm2(t_mid, 0.0)
        p_net = car.array_area_m2 * car.array_efficiency * ghi - parc_idle_w
        delta_wh += p_net * dt / 3600.0

    if getattr(rc, "CHARGING_MODE", "normal") == "extended_2h":
        extra_s = getattr(rc, "EXTENDED_EVENING_CHARGE_S", 2.0 * 3600.0)
        t_eve = rc.day_finish_time_s(day_index) - extra_s / 2.0
        ghi_eve = solar_provider.ghi_wm2(t_eve, 0.0)
        p_eve = car.array_area_m2 * car.array_efficiency * ghi_eve - parc_idle_w
        delta_wh += p_eve * extra_s / 3600.0

    stored = delta_wh * car.charge_eff if delta_wh >= 0 else delta_wh / car.discharge_eff
    return stored / car.battery_nominal_wh * 100.0


def relaxed_loop_combos(plan: _DayPlan, t_window_s: float, t_stops_base_s: float,
                        loop_speed_ms: float, pre_attempt_stop_s: float):
    loops = plan.loops
    if not loops:
        yield (), 0.0
        return
    budget = max(0.0, t_window_s - t_stops_base_s)
    t_per_attempt = [(km * 1000.0) / loop_speed_ms + pre_attempt_stop_s for _n, km in loops]
    caps = [int(budget // t) if t > 0 else 0 for t in t_per_attempt]
    import itertools
    for reps in itertools.product(*(range(c + 1) for c in caps)):
        if sum(n * t for n, t in zip(reps, t_per_attempt)) > budget:
            continue
        yield reps, sum(n * loops[i][1] for i, n in enumerate(reps))


def guess_baseline(routes: list, car: CarState, solar_providers: dict, wind_providers: dict,
                   start_soc_pct: float, start_day: int = 0, dist_done_km: float = 0.0,
                   elapsed_s: float = 0.0, cs_taken: bool = False, loops_done: dict | None = None,
                   kml_paths: dict | None = None,
                   plans_override: list | None = None) -> dict:

    _energy_diag_logged.clear()  # reset for fresh run

    n_days = rc.N_RACE_DAYS
    completion = (rc.RACE_MODE == "completion")
    soc_buckets = np.arange(car.soc_min_pct, car.soc_max_pct + 1e-9, sc.DP_SOC_BUCKET_PCT)
    nb = len(soc_buckets)

    loop_speed_ms = max(getattr(rc, "LOOP_CRUISE_SPEED_MS", car.v_max_ms), 1e-6)
    turnaround_s = getattr(rc, "LOOP_TURNAROUND_S", 0.0)
    pre_attempt_stop_s = rc.LOOP_STOP_DURATION_S + turnaround_s

    if plans_override is not None:
        if len(plans_override) != n_days:
            raise ValueError(
                f"plans_override must contain exactly {n_days} day plans; "
                f"got {len(plans_override)}"
            )
        plans = list(plans_override)
    else:
        plans = [_get_day_plan(d) for d in range(n_days)]

    V = np.full((n_days + 1, nb), -np.inf)
    V[n_days, :] = 0.0
    best_reps = [[() for _ in range(nb)] for _ in range(n_days)]
    best_end_soc = np.full((n_days, nb), np.nan)

    for d in range(n_days - 1, start_day - 1, -1):
        is_today = (d == start_day)
        route = routes[d] if routes and d < len(routes) else None

        # Extract today's weather
        solar_provider = solar_providers.get(d)
        wind_provider = wind_providers.get(d)

        nom_plan = plans[d]
        if is_today and dist_done_km > 0:
            plan = _adjust_plan_for_today(nom_plan, dist_done_km, loops_done)
        else:
            plan = nom_plan

        base_km = plan.stage1_km + plan.stage2_km

        t_window = max(0.0, rc.day_finish_time_s(d) - rc.day_start_time_s(d) - (elapsed_s if is_today else 0.0))
        t_stops_base = (0.0 if (is_today and cs_taken) else rc.CONTROL_STOP_DURATION_S) + rc.UNPLANNED_STOP_BUDGET_S
        t0_s = rc.day_start_time_s(d) + (elapsed_s if is_today else 0.0)
        gain = overnight_soc_gain(car, solar_provider, d)

        # Completion mode imposes a terminal SOC/finish requirement; it does
        # not disable race loops. Keep the candidate set identical to the
        # non-completion distance allocator so L1 and L2/L3 cannot disagree
        # about which loop decisions are even available.
        combos = list(relaxed_loop_combos(
            plan, t_window, t_stops_base, loop_speed_ms, pre_attempt_stop_s))

        for s_idx, s0 in enumerate(soc_buckets):
            best_val = -np.inf
            for reps, loop_km in combos:
                n_att = sum(reps) if reps else 0
                t_stops = t_stops_base + n_att * pre_attempt_stop_s
                t_loop_drive = (loop_km * 1000.0) / loop_speed_ms if loop_km else 0.0
                t_avail_base = t_window - t_stops - t_loop_drive

                if t_avail_base <= 0 and base_km > 0:
                    continue

                t_base_vmax = (base_km * 1000.0) / car.v_max_ms if base_km > 0 else 0.0

                # Late penalty tracking
                if t_avail_base < t_base_vmax:
                    late_s = t_base_vmax - t_avail_base
                    if rc.day_finish_time_s(d) + late_s > rc.FINISH_CUTOFF_ABS_S:
                        continue
                    v_base_ms = car.v_max_ms
                    pen_s = rc.late_finish_penalty_min(late_s / 60.0) * 60.0
                else:
                    v_base_ms = (base_km * 1000.0) / max(t_avail_base, 1e-6) if base_km > 0 else car.v_max_ms
                    v_base_ms = min(v_base_ms, car.v_max_ms)
                    pen_s = 0.0

                offset_km = dist_done_km if is_today else 0.0
                end_soc = evaluate_day(
                    car, route, plan, reps, offset_km, v_base_ms, loop_speed_ms,
                    pre_attempt_stop_s, solar_provider, wind_provider, t0_s, s0,
                    day_index=d,
                    is_today=is_today,
                    cs_taken=cs_taken,
                    kml_paths=kml_paths
                )

                floor = car.soc_min_pct + (_completion_margin() if completion else 0.0)
                if end_soc < floor:
                    continue

                next_soc = min(end_soc + gain, car.soc_max_pct)
                v_next = _interp_value(V[d + 1], soc_buckets, next_soc)
                if not np.isfinite(v_next):
                    continue

                dist_km = base_km + loop_km
                p_loss_km = (pen_s * v_base_ms / 1000.0) if pen_s > 0 else 0.0

                # Completion mode is a feasibility constraint, not a different
                # race objective. Optional loops still contribute race distance.
                val = dist_km + v_next - p_loss_km
                if val > best_val:
                    best_val = val
                    best_reps[d][s_idx] = reps
                    best_end_soc[d, s_idx] = end_soc
            V[d][s_idx] = best_val

    # Forward trace CONTINUOUS trajectory from live start SOC
    s0_traj = np.full(n_days, np.nan)
    feasible = True
    cur = float(np.clip(start_soc_pct, car.soc_min_pct, car.soc_max_pct))

    for d in range(start_day, n_days):
        s0_traj[d] = cur
        s_idx = int(np.clip(np.searchsorted(soc_buckets, cur) - 1, 0, nb - 1))
        reps = best_reps[d][s_idx]
        end = best_end_soc[d, s_idx]
        if not np.isfinite(end):
            feasible = False
            n_finite = int(np.isfinite(V[d]).sum())
            logger.error(
                "Tier1 forward trace broke at Day %d (soc_bucket=%.1f%%, idx=%d). "
                "V[d] has %d/%d finite entries.",
                d + 1, soc_buckets[s_idx], s_idx, n_finite, nb)
            feasible_buckets = soc_buckets[np.isfinite(V[d]) & (V[d] > -np.inf)]
            if len(feasible_buckets) > 0:
                logger.error("  Day %d IS feasible for SOC buckets: %.1f%% - %.1f%%",
                             d + 1, feasible_buckets[0], feasible_buckets[-1])
            else:
                logger.error("  Day %d has NO feasible SOC buckets at all!", d + 1)
            break
        gain = overnight_soc_gain(car, solar_providers.get(d), d)
        cur = min(end + gain, car.soc_max_pct)
        logger.info("Tier1 Day %d: start=%.1f%% end=%.1f%% +overnight=%.1f%% -> next=%.1f%%",
                    d + 1, s0_traj[d], end, gain, cur)

    return dict(s0_pct=s0_traj, day_plans=plans, feasible=feasible)

def _completion_margin() -> float:
    return getattr(rc, "DP_COMPLETION_MARGIN_PCT", 5.0)

def _interp_value(v_row: np.ndarray, buckets: np.ndarray, soc: float) -> float:
    finite = np.isfinite(v_row)
    if not finite.any():
        return -np.inf
    return float(np.interp(soc, buckets[finite], v_row[finite]))
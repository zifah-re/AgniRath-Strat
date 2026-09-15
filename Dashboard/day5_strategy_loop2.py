"""
day5_strategy.py -- AgniRath Sasol Solar Challenge, Day 5 strategy engine
==========================================================================

Battery: pack is now 24 of 28 series modules (was 22/28 on Day 4).
3200 Wh assumed nameplate over 28 modules -> 114.29 Wh/module ->
2742.86 Wh usable on 24 modules. Hard floor: never below 80 V pack
terminal voltage (Bus_Voltage-scale signal, same domain as
SOC_CURVE_V_PCT below).

Route: Stage 1 Olifantshoek->Upington (178.65 km) -> control stop at
Upington -> N x Upington Loop -> Stage 2 Upington->Augrabies (114.47 km).

*** LOOP2 VARIANT ***: instead of driving the full 60.76 km Upington
Loop each lap, this file truncates the loop profile to just its first
17 km and its last 17 km (stitched back-to-back), so each modeled lap
covers 34 km total instead of 60.76 km. See LOOP_TRUNCATE_KM_EACH and
truncate_route_first_last_km() below. Everything else is identical to
day5_strategy_loop1.py.

Two weather-driven strategy modes:

  MODE A (rain): morning rain means the car is TRAILERED (not driven)
  from Olifantshoek to the Upington control stop. Trailer speed is now
  a real input argument (default 65 km/h): arrival time at the control
  stop is CALCULATED from stage1 distance / trailer speed (departing
  08:02), rather than being supplied directly as a clock time.
      Arguments (3): n_loops, soc_at_control_stop_pct, trailer_speed_kmh

  MODE B (no rain): the car is driven normally the whole day. Speed is
  grid-searched over 50-70 km/h (uniform across Stage 1 + Loops + Stage 2)
  to find the "energy neutral" cruise speed -- the speed at which gross
  motor draw while moving is matched by gross solar collected while
  moving (a genuine solar-equilibrium cruise speed), for the specific
  number of loops given (more loops shifts driving into different
  daylight windows, so equilibrium speed depends on n_loops too).
      Arguments (2): n_loops, soc_at_departure_pct (SOC at 08:02 departure)

Both modes:
  * apply the same +30 min sun-tracking charge stop at the control
    stop before starting loops, and a 5-min charge stop after each lap
    (same "reach there +30 min, then loop" rule + per-loop stop pattern
    used on Day 3/4).
  * report SOC AND voltage on both the healthy 28-module curve and the
    actual degraded 24-module curve ("the full pack and this exact pack").
  * produce SOC-vs-distance and Solar-input(W)-vs-distance PLOTS (via
    matplotlib, saved as PNG + shown) across the whole day -- the solar
    trace is instantaneous/average W, not cumulative Wh. In Mode A, the
    Stage-1 (trailer) segment is shown as a FLAT SOC line at the given
    starting SOC with ZERO solar the whole 178.65 km (rain, trailered,
    nothing computed) -- exactly as instructed, not modeled physics.
  * n_loops is a genuine argument throughout (affects distance, time-of-
    day for every subsequent leg, and Mode B's equilibrium speed search).

Per-stage speeds (Stage 1 / Loop / Stage 2) are independent, tunable
constants below -- used directly in Mode A for the Loop/Stage2 legs
(Stage 1 is trailered in Mode A, so STAGE1_TARGET_SPEED_KMH is unused
there; it's kept for a manual what-if run). Mode B ignores these three
and instead grid-searches ONE uniform velocity for the whole day.
"""

from __future__ import annotations

import json
import math
import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 0. FILE LOCATIONS  (run this from inside your Dashboard/ folder)
# ---------------------------------------------------------------------
DASH = Path(".")
SAVES = DASH / "Saves"
SOLAR = DASH / "Solar"

ROUTE_FILES = {
    "stage1": SAVES / "2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 1 Olifantshoek to Upington.kml.save",
    "loop":   SAVES / "2026 Sasol Solar Challenge Route (Publish)_Day 5 _Upington Loop.kml.save",
    "stage2": SAVES / "2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 2 Upington to Augrabies.kml.save",
}

# Solar forecast files -- "_2" revision, as confirmed.
SOLAR_FILES = {
    "stage1": SOLAR / "mean_2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 1 Olifantshoek to Upington_2.jsonl",
    "loop":   SOLAR / "mean_2026 Sasol Solar Challenge Route (Publish)_Day 5 _Upington Loop_2.jsonl",
    "stage2": SOLAR / "mean_2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 2 Upington to Augrabies_2.jsonl",
}

LOCAL_TZ = timezone(timedelta(hours=2))   # SAST, UTC+2
RACE_DATE = "2026-09-14"


def local_to_utc(hh: int, mm: int) -> datetime:
    local = datetime.fromisoformat(f"{RACE_DATE}T{hh:02d}:{mm:02d}:00").replace(tzinfo=LOCAL_TZ)
    return local.astimezone(timezone.utc)


def parse_local_hhmm(hhmm: str) -> datetime:
    hh, mm = hhmm.split(":")
    return local_to_utc(int(hh), int(mm))


# =======================================================================
# 1. BATTERY MODEL -- degraded 24/28-module pack
# =======================================================================

SOC_CURVE_V_PCT_HEALTHY: list[tuple[float, float]] = [
    (115.36, 100.0), (114.38, 99.0), (113.40, 98.0), (112.70, 97.0),
    (112.00, 96.0), (111.44, 95.0), (110.88, 94.0), (110.60, 93.0),
    (110.32, 92.0), (110.04, 91.0), (109.76, 90.0), (109.48, 89.0),
    (109.20, 88.0), (108.92, 87.0), (108.64, 86.0), (108.36, 85.0),
    (108.08, 84.0), (107.80, 83.0), (107.52, 82.0), (107.24, 81.0),
    (106.96, 80.0), (106.75, 79.0), (106.54, 78.0), (106.33, 77.0),
    (106.12, 76.0), (105.91, 75.0), (105.70, 74.0), (105.49, 73.0),
    (105.28, 72.0), (105.07, 71.0), (104.86, 70.0), (104.65, 69.0),
    (104.44, 68.0), (104.16, 67.0), (103.88, 66.0), (103.60, 65.0),
    (103.32, 64.0), (103.04, 63.0), (102.76, 62.0), (102.48, 61.0),
    (102.20, 60.0), (101.92, 59.0), (101.64, 58.0), (101.36, 57.0),
    (101.08, 56.0), (100.80, 55.0), (100.52, 54.0), (100.24, 53.0),
    (99.96, 52.0), (99.61, 51.0), (99.26, 50.0), (98.91, 49.0),
    (98.56, 48.0), (98.21, 47.0), (97.86, 46.0), (97.51, 45.0),
    (97.16, 44.0), (96.81, 43.0), (96.46, 42.0), (96.11, 41.0),
    (95.76, 40.0), (95.41, 39.0), (95.06, 38.0), (94.71, 37.0),
    (94.36, 36.0), (93.94, 35.0), (93.52, 34.0), (93.10, 33.0),
    (92.68, 32.0), (92.26, 31.0), (91.84, 30.0), (91.42, 29.0),
    (91.00, 28.0), (90.51, 27.0), (90.02, 26.0), (89.53, 25.0),
    (89.04, 24.0), (88.55, 23.0), (88.06, 22.0), (87.57, 21.0),
    (87.08, 20.0), (86.52, 19.0), (85.96, 18.0), (85.40, 17.0),
    (84.84, 16.0), (84.14, 15.0), (83.44, 14.0), (82.74, 13.0),
    (82.04, 12.0), (81.20, 11.0), (80.36, 10.0), (79.52, 9.0),
    (78.68, 8.0), (77.56, 7.0), (76.44, 6.0), (74.90, 5.0),
    (73.36, 4.0), (71.40, 3.0), (69.44, 2.0), (69.02, 1.0),
    (68.60, 0.0),
]

N_MODULES_HEALTHY = 28
N_MODULES_NOW = 24          # today's estimate: 24 of 28 modules available
PACK_WH_ASSUMED = 3200.0
WH_PER_MODULE = PACK_WH_ASSUMED / N_MODULES_HEALTHY             # 114.29 Wh
USABLE_WH_NOW = WH_PER_MODULE * N_MODULES_NOW                   # 2742.86 Wh nameplate on 24 modules

VOLTAGE_FLOOR_V = 80.0

SCALE = N_MODULES_NOW / N_MODULES_HEALTHY                        # 0.8571...
_V_HEALTHY = np.array([v for v, _ in SOC_CURVE_V_PCT_HEALTHY])[::-1]
_PCT_HEALTHY = np.array([p for _, p in SOC_CURVE_V_PCT_HEALTHY])[::-1]
_V_DEG = _V_HEALTHY * SCALE
_PCT_DEG = _PCT_HEALTHY


def soc_from_voltage_degraded(v: float) -> float:
    return float(np.interp(v, _V_DEG, _PCT_DEG))


def voltage_from_soc_degraded(pct: float) -> float:
    return float(np.interp(pct, _PCT_DEG, _V_DEG))


def voltage_from_soc_healthy(pct: float) -> float:
    """What the same SOC% would read on a fully-healthy 28-module pack --
    for reference/comparison only, NOT the real reading of this car."""
    return float(np.interp(pct, _PCT_HEALTHY, _V_HEALTHY))


FLOOR_SOC_PCT = soc_from_voltage_degraded(VOLTAGE_FLOOR_V)
USABLE_WH_ABOVE_FLOOR = USABLE_WH_NOW * (100.0 - FLOOR_SOC_PCT) / 100.0


def wh_to_soc_delta(wh: float, charge_eff: float = 0.96, discharge_eff: float = 0.96) -> float:
    if wh >= 0:
        return (wh * charge_eff) / USABLE_WH_NOW * 100.0
    else:
        return (wh / discharge_eff) / USABLE_WH_NOW * 100.0


def print_battery_summary() -> None:
    print("=" * 78)
    print("BATTERY: degraded 24/28-module pack")
    print("=" * 78)
    print(f"Module count: healthy={N_MODULES_HEALTHY}, now={N_MODULES_NOW} "
          f"({N_MODULES_HEALTHY - N_MODULES_NOW} unavailable)")
    print(f"Assumed usable capacity: {PACK_WH_ASSUMED:.0f} Wh nameplate over {N_MODULES_HEALTHY} modules "
          f"-> {WH_PER_MODULE:.2f} Wh/module -> {USABLE_WH_NOW:.2f} Wh on {N_MODULES_NOW} modules")
    print(f"Voltage scale factor ({N_MODULES_NOW}/{N_MODULES_HEALTHY} series modules): {SCALE:.4f}")
    print()
    print(f"{'SOC%':>6}  {'V_healthy(28mod)':>17}  {'V_degraded(24mod)':>18}")
    for pct in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]:
        print(f"{pct:6d}  {voltage_from_soc_healthy(pct):17.2f}  {voltage_from_soc_degraded(pct):18.2f}")
    print()
    print(f"Mandated floor: {VOLTAGE_FLOOR_V:.1f} V  ->  {FLOOR_SOC_PCT:.2f} % SOC on the degraded curve")
    print(f"Usable energy ABOVE the 80V floor: {USABLE_WH_ABOVE_FLOOR:.1f} Wh "
          f"({100 - FLOOR_SOC_PCT:.2f} % of nameplate)")
    print()


# =======================================================================
# 2. SOLAR MODEL
# =======================================================================

ARRAY_AREA_M2 = 5.78
ARRAY_EFFICIENCY = 0.24
N_MPPT_CHANNELS = 4
AREA_PER_CHANNEL_M2 = ARRAY_AREA_M2 / N_MPPT_CHANNELS
MPPT_D_SHADE_FACTOR = 0.25   # carried over from Day 3/4 (human-set; derived telemetry value was ~0.086)


def load_solar_series(path: Path) -> list[tuple[datetime, float, float]]:
    with open(path) as f:
        payload = json.load(f)[0]
    out = []
    for row in payload["data"]:
        t = datetime.fromisoformat(row["period_end"])
        out.append((t, float(row["dni"]), float(row["ghi"])))
    out.sort(key=lambda r: r[0])
    return out


def irradiance_at(series, t_utc: datetime) -> tuple[float, float]:
    times = [r[0] for r in series]
    if t_utc <= times[0]:
        return series[0][1], series[0][2]
    if t_utc >= times[-1]:
        return series[-1][1], series[-1][2]
    for (t0, d0, g0), (t1, d1, g1) in zip(series, series[1:]):
        if t0 <= t_utc <= t1:
            span = (t1 - t0).total_seconds()
            f = 0.0 if span == 0 else (t_utc - t0).total_seconds() / span
            return d0 + f * (d1 - d0), g0 + f * (g1 - g0)
    return series[-1][1], series[-1][2]


def stationary_charge_power_w(dni_w_m2: float) -> float:
    """Sun-tracking, no shading (control stop / EOD charge)."""
    return dni_w_m2 * ARRAY_AREA_M2 * ARRAY_EFFICIENCY


def driving_solar_power_w(ghi_w_m2: float) -> float:
    """Fixed flat mount while moving -- MPPT-D shaded."""
    clean = 3 * AREA_PER_CHANNEL_M2
    shaded_d = AREA_PER_CHANNEL_M2 * MPPT_D_SHADE_FACTOR
    return ghi_w_m2 * (clean + shaded_d) * ARRAY_EFFICIENCY


def energy_over_window_wh(series, t0_utc: datetime, t1_utc: datetime, power_fn, step_min: float = 5.0) -> float:
    if t1_utc <= t0_utc:
        return 0.0
    n_steps = max(1, int((t1_utc - t0_utc).total_seconds() / 60.0 / step_min))
    ts = [t0_utc + timedelta(minutes=step_min * i) for i in range(n_steps + 1)]
    ts[-1] = t1_utc
    powers = []
    for t in ts:
        dni, ghi = irradiance_at(series, t)
        p = power_fn(dni) if power_fn is stationary_charge_power_w else power_fn(ghi)
        powers.append(p)
    wh = 0.0
    for (ta, pa), (tb, pb) in zip(zip(ts, powers), zip(ts[1:], powers[1:])):
        dt_h = (tb - ta).total_seconds() / 3600.0
        wh += 0.5 * (pa + pb) * dt_h
    return wh


# =======================================================================
# 3. ROUTE MODEL + constant-speed cruise/regen physics
# =======================================================================

G = 9.81
AIR_DENSITY = 1.20
MASS_KG = 336.5
CRR = 0.007
CDA_M2 = 0.16
MOTOR_EFF = 0.95
REGEN_EFF = 0.80
P_IDLE_W = 5.0

# ---- Tunable per-stage speeds (used directly by Mode A for Loop/Stage2;
# Stage 1 is trailered in Mode A so this constant only matters for a manual
# what-if run of Stage 1 under power). Mode B ignores all three and instead
# grid-searches ONE uniform velocity for the whole day. ----
STAGE1_TARGET_SPEED_KMH = 60.0
LOOP_TARGET_SPEED_KMH = 60.0
STAGE2_TARGET_SPEED_KMH = 60.0

TRAILER_SPEED_KMH = 65.0        # Mode A: reference-only, for an expected-transit-time printout
CONTROL_STOP_BUFFER_MIN = 30    # "reach there + 30 min, then loops" -- sun-tracking charge
LOOP_PENALTY_MIN = 5            # sun-tracking charge stop after each lap


@dataclasses.dataclass
class RouteProfile:
    name: str
    distance_km: np.ndarray
    gradient_pct: np.ndarray

    @property
    def total_km(self) -> float:
        return float(self.distance_km[-1])


def load_route(path: Path, name: str) -> RouteProfile:
    with open(path) as f:
        d = json.load(f)
    prof = d["profile"]
    return RouteProfile(
        name=name,
        distance_km=np.array(prof["Distance"], dtype=float),
        gradient_pct=np.array(prof["Gradient"], dtype=float),
    )


LOOP_TRUNCATE_KM_EACH = 17.0   # keep only the first/last N km of each loop lap


def truncate_route_first_last_km(route: RouteProfile, km_each: float = LOOP_TRUNCATE_KM_EACH) -> RouteProfile:
    """Collapse `route` down to just its first `km_each` km and its last
    `km_each` km, stitched back-to-back so the returned profile's total
    distance is 2*km_each. Used here for the Day-5 Upington Loop when we
    only want to model the start/finish sections of each lap instead of
    the full loop distance. Gradient values are carried over unchanged
    for the points kept; only the distance axis of the "last" section is
    re-baselined so it continues right where the "first" section ends."""
    total = route.total_km
    if 2 * km_each >= total:
        print(f"  ** {route.name}: 2 x {km_each:.1f} km >= total {total:.1f} km, "
              f"nothing to truncate -- using full route **")
        return route

    first_mask = route.distance_km <= km_each
    last_mask = route.distance_km >= (total - km_each)

    first_dist = route.distance_km[first_mask]
    first_grad = route.gradient_pct[first_mask]

    last_dist_raw = route.distance_km[last_mask]
    last_grad = route.gradient_pct[last_mask]
    # Re-baseline the "last" section's distance so it starts right where
    # the "first" section left off (at km_each) and runs to 2*km_each.
    last_dist = last_dist_raw - (total - km_each) + km_each

    # Avoid a zero-length duplicate point exactly at the km_each seam.
    if len(first_dist) and len(last_dist) and math.isclose(first_dist[-1], last_dist[0]):
        last_dist = last_dist[1:]
        last_grad = last_grad[1:]

    new_dist = np.concatenate([first_dist, last_dist])
    new_grad = np.concatenate([first_grad, last_grad])

    truncated = RouteProfile(
        name=f"{route.name} (first {km_each:.0f} km + last {km_each:.0f} km)",
        distance_km=new_dist,
        gradient_pct=new_grad,
    )
    print(f"  {route.name}: truncated {total:.2f} km -> {truncated.total_km:.2f} km "
          f"(first {km_each:.0f} km + last {km_each:.0f} km)")
    return truncated


def check_gradient_feasible(route: RouteProfile, speed_kmh: float, max_safe_grade_pct: float = 8.0) -> bool:
    bad = np.abs(route.gradient_pct) > max_safe_grade_pct
    if bad.any():
        worst = route.gradient_pct[np.argmax(np.abs(route.gradient_pct))]
        print(f"  ** {route.name}: {bad.sum()} point(s) exceed +/-{max_safe_grade_pct}% "
              f"(worst {worst:.2f}%) at {speed_kmh:.0f} km/h -- review before committing **")
        return False
    print(f"  {route.name}: gradients OK (max {route.gradient_pct.max():.2f}%, "
          f"min {route.gradient_pct.min():.2f}%) at {speed_kmh:.0f} km/h.")
    return True


def _segment_forces(route: RouteProfile, speed_ms: float):
    seg_km = np.diff(route.distance_km)
    seg_grad = 0.5 * (route.gradient_pct[:-1] + route.gradient_pct[1:])
    theta = np.arctan(seg_grad / 100.0)
    f_rr = CRR * MASS_KG * G * np.cos(theta)
    f_aero = 0.5 * AIR_DENSITY * CDA_M2 * speed_ms ** 2
    f_grav = MASS_KG * G * np.sin(theta)
    f_total = f_rr + f_aero + f_grav
    seg_time_s = (seg_km * 1000.0) / speed_ms
    return seg_km, f_total, seg_time_s


def drive_leg_trace(route: RouteProfile, start_utc: datetime, solar_series,
                     speed_kmh: float, n_laps: int, cum_km_start: float, soc_start: float):
    """Per-segment trace through `n_laps` laps of `route` at `speed_kmh`.
    Returns a dict with cumulative-distance / SOC / solar traces (for
    plotting) plus the aggregate totals (for the log)."""
    speed_ms = speed_kmh / 3.6
    seg_km, f_total, seg_time_s = _segment_forces(route, speed_ms)
    p_wheel = f_total * speed_ms

    cum_km_trace = [cum_km_start]
    soc_trace = [soc_start]
    cum_solar_wh_trace = [0.0]
    solar_power_w_trace = [0.0]   # instantaneous solar input (W) at each trace point

    t = start_utc
    soc = soc_start
    cum_km = cum_km_start
    cum_solar_wh = 0.0
    motor_wh_total = 0.0
    regen_wh_total = 0.0
    solar_wh_total = 0.0

    for lap in range(max(0, n_laps)):
        for p_w, t_s, km in zip(p_wheel, seg_time_s, seg_km):
            t_next = t + timedelta(seconds=t_s)
            dni, ghi = irradiance_at(solar_series, t + (t_next - t) / 2)
            solar_w = driving_solar_power_w(ghi)
            solar_wh = solar_w * (t_s / 3600.0)

            if p_w >= 0:
                elec_w = p_w / MOTOR_EFF + P_IDLE_W
                motor_wh_total += elec_w * (t_s / 3600.0)
            else:
                elec_w = p_w * REGEN_EFF + P_IDLE_W
                regen_wh_total += -elec_w * (t_s / 3600.0)
            drive_wh = elec_w * (t_s / 3600.0)

            net_wh = solar_wh - drive_wh
            soc += wh_to_soc_delta(net_wh)
            soc = min(100.0, max(0.0, soc))
            cum_km += km
            cum_solar_wh += solar_wh
            solar_wh_total += solar_wh
            t = t_next

            cum_km_trace.append(cum_km)
            soc_trace.append(soc)
            cum_solar_wh_trace.append(cum_solar_wh)
            solar_power_w_trace.append(solar_w)

    return {
        "cum_km_trace": cum_km_trace,
        "soc_trace": soc_trace,
        "cum_solar_wh_trace": cum_solar_wh_trace,
        "solar_power_w_trace": solar_power_w_trace,
        "end_utc": t,
        "end_soc": soc,
        "end_cum_km": cum_km,
        "end_cum_solar_wh": cum_solar_wh,
        "motor_wh": motor_wh_total,
        "regen_wh": regen_wh_total,
        "solar_wh": solar_wh_total,
        "net_wh": solar_wh_total - (motor_wh_total - regen_wh_total),
        "duration_min": (t - start_utc).total_seconds() / 60.0,
    }


def drive_leg_totals_only(route: RouteProfile, start_utc: datetime, solar_series,
                           speed_kmh: float, n_laps: int) -> dict:
    """Same physics as drive_leg_trace but skips building the point-by-point
    trace -- used inside the Mode B grid search where we only need totals,
    many times over, and don't want the overhead of full traces."""
    if n_laps <= 0:
        return {"motor_wh": 0.0, "regen_wh": 0.0, "solar_wh": 0.0, "net_wh": 0.0,
                "duration_min": 0.0, "end_utc": start_utc}
    speed_ms = speed_kmh / 3.6
    seg_km, f_total, seg_time_s = _segment_forces(route, speed_ms)
    p_wheel = f_total * speed_ms
    lap_time_s = float(seg_time_s.sum())

    motor_w_per_seg = np.where(p_wheel >= 0, p_wheel / MOTOR_EFF + P_IDLE_W, p_wheel * REGEN_EFF + P_IDLE_W)
    motor_wh = float(np.sum(np.where(p_wheel >= 0, motor_w_per_seg, 0.0) * seg_time_s / 3600.0))
    regen_wh = float(np.sum(np.where(p_wheel < 0, -motor_w_per_seg, 0.0) * seg_time_s / 3600.0))

    total_time_s = lap_time_s * n_laps
    end_utc = start_utc + timedelta(seconds=total_time_s)
    solar_wh = energy_over_window_wh(solar_series, start_utc, end_utc, driving_solar_power_w) if total_time_s > 0 else 0.0

    return {
        "motor_wh": motor_wh * n_laps,
        "regen_wh": regen_wh * n_laps,
        "solar_wh": solar_wh,
        "net_wh": solar_wh - (motor_wh - regen_wh) * n_laps,
        "duration_min": total_time_s / 60.0,
        "end_utc": end_utc,
    }


# =======================================================================
# 4. SHARED HELPERS: control stop + loops (both modes use these)
# =======================================================================

def run_control_stop_and_loops(n_loops: int, arrival_utc: datetime, soc: float, cum_km: float,
                                cum_solar_wh: float, log: list, cum_km_trace: list,
                                soc_trace: list, power_trace: list, loop_route: RouteProfile,
                                loop_solar_series, loop_speed_kmh: float) -> dict:
    """+30 min sun-tracking charge, then n_loops laps (each with a 5-min
    charge stop after it). Shared by Mode A and Mode B. `loop_speed_kmh` is
    explicit (not read from the global LOOP_TARGET_SPEED_KMH) so Mode B can
    drive loops at its grid-searched equilibrium speed, matching what the
    grid search itself assumed, while Mode A uses the fixed constant."""

    # ---- +30 min control-stop charge ----
    cs_start = arrival_utc
    cs_end = cs_start + timedelta(minutes=CONTROL_STOP_BUFFER_MIN)
    wh_cs = energy_over_window_wh(loop_solar_series, cs_start, cs_end, stationary_charge_power_w)
    d_soc = wh_to_soc_delta(wh_cs)
    soc = min(100.0, max(0.0, soc + d_soc))
    cum_solar_wh += wh_cs
    log.append({"event": f"{CONTROL_STOP_BUFFER_MIN}-min control-stop charge (sun-tracking)",
                "wh": wh_cs, "d_soc_pct": d_soc, "soc_pct_after": soc,
                "note": f"{cs_start.astimezone(LOCAL_TZ):%H:%M}->{cs_end.astimezone(LOCAL_TZ):%H:%M}"})
    avg_w_cs = wh_cs / (CONTROL_STOP_BUFFER_MIN / 60.0) if CONTROL_STOP_BUFFER_MIN > 0 else 0.0
    cum_km_trace.append(cum_km); soc_trace.append(soc); power_trace.append(avg_w_cs)

    t = cs_end
    for i in range(max(0, n_loops)):
        trace = drive_leg_trace(loop_route, t, loop_solar_series, loop_speed_kmh,
                                 n_laps=1, cum_km_start=cum_km, soc_start=soc)
        cum_km_trace.extend(trace["cum_km_trace"][1:])
        soc_trace.extend(trace["soc_trace"][1:])
        power_trace.extend(trace["solar_power_w_trace"][1:])

        cum_km = trace["end_cum_km"]
        soc = trace["end_soc"]
        cum_solar_wh += trace["end_cum_solar_wh"]
        t = trace["end_utc"]

        log.append({"event": f"Loop {i + 1}/{n_loops} ({loop_route.total_km:.1f} km @ {loop_speed_kmh:.1f} km/h)",
                     "wh": trace["net_wh"], "d_soc_pct": soc - soc_trace[-2 - len(trace['cum_km_trace'][1:]) + 1] if False else None,
                     "soc_pct_after": soc,
                     "note": f"motor={trace['motor_wh']:.1f} Wh, regen={trace['regen_wh']:.1f} Wh recovered, "
                             f"solar={trace['solar_wh']:.1f} Wh, {trace['duration_min']:.1f} min"})

        charge_end = t + timedelta(minutes=LOOP_PENALTY_MIN)
        wh_pen = energy_over_window_wh(loop_solar_series, t, charge_end, stationary_charge_power_w)
        d_soc = wh_to_soc_delta(wh_pen)
        soc = min(100.0, max(0.0, soc + d_soc))
        cum_solar_wh += wh_pen
        log.append({"event": f"{LOOP_PENALTY_MIN}-min charge stop after loop {i + 1}",
                     "wh": wh_pen, "d_soc_pct": d_soc, "soc_pct_after": soc,
                     "note": f"{t.astimezone(LOCAL_TZ):%H:%M}->{charge_end.astimezone(LOCAL_TZ):%H:%M}"})
        avg_w_pen = wh_pen / (LOOP_PENALTY_MIN / 60.0) if LOOP_PENALTY_MIN > 0 else 0.0
        cum_km_trace.append(cum_km); soc_trace.append(soc); power_trace.append(avg_w_pen)
        t = charge_end

    return {"end_utc": t, "soc": soc, "cum_km": cum_km, "cum_solar_wh": cum_solar_wh}


def run_eod_charge(arrival_utc: datetime, soc: float, cum_km: float, cum_solar_wh: float,
                    log: list, cum_km_trace: list, soc_trace: list, power_trace: list,
                    solar_series) -> float:
    four_pm = local_to_utc(16, 0)
    five_pm = local_to_utc(17, 0)
    avg_w = 0.0
    if arrival_utc <= four_pm:
        charge_start = arrival_utc + timedelta(hours=1)
        charge_end = five_pm
        if charge_start < charge_end:
            wh = energy_over_window_wh(solar_series, charge_start, charge_end, stationary_charge_power_w)
            d_soc = wh_to_soc_delta(wh)
            soc = min(100.0, max(0.0, soc + d_soc))
            cum_solar_wh += wh
            hrs = (charge_end - charge_start).total_seconds() / 3600.0
            avg_w = wh / hrs if hrs > 0 else 0.0
            log.append({"event": f"End-of-day charge ({charge_start.astimezone(LOCAL_TZ):%H:%M}"
                                  f"->{charge_end.astimezone(LOCAL_TZ):%H:%M})",
                         "wh": wh, "d_soc_pct": d_soc, "soc_pct_after": soc, "note": ""})
        else:
            log.append({"event": "End-of-day charge window", "wh": 0.0, "d_soc_pct": 0.0,
                         "soc_pct_after": soc, "note": "arrival+1h already past 17:00, skipped"})
    else:
        log.append({"event": "End-of-day charge window", "wh": 0.0, "d_soc_pct": 0.0,
                     "soc_pct_after": soc,
                     "note": f"arrived {arrival_utc.astimezone(LOCAL_TZ):%H:%M} (after 16:00) -> no EOD charge"})
    cum_km_trace.append(cum_km); soc_trace.append(soc); power_trace.append(avg_w)
    return soc


def print_log(log: list, title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'Event':58s} {'Wh':>9s} {'SOC%':>7s}")
    for row in log:
        print(f"{row['event'][:58]:58s} {row['wh']:9.1f} {row['soc_pct_after']:7.2f}")
        if row.get("note"):
            print(f"    -> {row['note']}")
    print()


# =======================================================================
# 5. MODE A -- rain, trailer Stage 1
# =======================================================================

def run_mode_a(n_loops: int, soc_at_control_stop_pct: float, trailer_speed_kmh: float,
               verbose: bool = True) -> dict:
    stage1 = load_route(ROUTE_FILES["stage1"], "Stage 1 (Olifantshoek->Upington)")
    loop = load_route(ROUTE_FILES["loop"], "Upington Loop")
    loop = truncate_route_first_last_km(loop, LOOP_TRUNCATE_KM_EACH)
    stage2 = load_route(ROUTE_FILES["stage2"], "Stage 2 (Upington->Augrabies)")

    solar_s1 = load_solar_series(SOLAR_FILES["stage1"])
    solar_loop = load_solar_series(SOLAR_FILES["loop"])
    solar_s2 = load_solar_series(SOLAR_FILES["stage2"])

    # Velocity is now the input -- transit time (and therefore arrival time
    # at the control stop) is CALCULATED from stage1 distance / trailer speed.
    depart_utc = local_to_utc(8, 2)
    transit_min = stage1.total_km / trailer_speed_kmh * 60.0
    arrival_utc = depart_utc + timedelta(minutes=transit_min)

    if verbose:
        print("=" * 78)
        print("MODE A: RAIN -- trailered to the control stop")
        print("=" * 78)
        print(f"Trailer speed (input): {trailer_speed_kmh:.1f} km/h over {stage1.total_km:.1f} km "
              f"-> transit time {transit_min:.1f} min -> departing 08:02, arrival at control stop "
              f"{arrival_utc.astimezone(LOCAL_TZ):%H:%M}")
        for r, spd in [(loop, LOOP_TARGET_SPEED_KMH), (stage2, STAGE2_TARGET_SPEED_KMH)]:
            check_gradient_feasible(r, spd)
        print()

    soc = soc_at_control_stop_pct
    log = []
    cum_km_trace = [0.0]
    soc_trace = [soc]
    power_trace = [0.0]

    log.append({"event": "Stage 1 (TRAILERED, rain -- 0 km driven under power)",
                "wh": 0.0, "d_soc_pct": 0.0, "soc_pct_after": soc,
                "note": f"{stage1.total_km:.1f} km trailered at {trailer_speed_kmh:.1f} km/h "
                        f"({transit_min:.1f} min), SOC held flat at input value, 0 W solar "
                        f"(per instruction: rain, panels not collecting)"})
    # Flat line across the whole Stage-1 distance, SOC constant, solar 0.
    cum_km_trace.append(stage1.total_km)
    soc_trace.append(soc)
    power_trace.append(0.0)
    cum_km = stage1.total_km
    cum_solar_wh = 0.0

    res = run_control_stop_and_loops(n_loops, arrival_utc, soc, cum_km, cum_solar_wh, log,
                                      cum_km_trace, soc_trace, power_trace, loop, solar_loop,
                                      loop_speed_kmh=LOOP_TARGET_SPEED_KMH)
    soc, cum_km, cum_solar_wh, t = res["soc"], res["cum_km"], res["cum_solar_wh"], res["end_utc"]

    trace2 = drive_leg_trace(stage2, t, solar_s2, STAGE2_TARGET_SPEED_KMH, n_laps=1,
                              cum_km_start=cum_km, soc_start=soc)
    cum_km_trace.extend(trace2["cum_km_trace"][1:])
    soc_trace.extend(trace2["soc_trace"][1:])
    power_trace.extend(trace2["solar_power_w_trace"][1:])
    log.append({"event": f"Drive Stage 2 ({stage2.total_km:.1f} km @ {STAGE2_TARGET_SPEED_KMH:.0f} km/h)",
                "wh": trace2["net_wh"], "d_soc_pct": None, "soc_pct_after": trace2["end_soc"],
                "note": f"motor={trace2['motor_wh']:.1f} Wh, regen={trace2['regen_wh']:.1f} Wh recovered, "
                        f"solar={trace2['solar_wh']:.1f} Wh, {trace2['duration_min']:.1f} min, "
                        f"arr {trace2['end_utc'].astimezone(LOCAL_TZ):%H:%M}"})
    soc = trace2["end_soc"]; cum_km = trace2["end_cum_km"]; cum_solar_wh += trace2["end_cum_solar_wh"]
    arrival_final_utc = trace2["end_utc"]

    soc = run_eod_charge(arrival_final_utc, soc, cum_km, cum_solar_wh, log,
                          cum_km_trace, soc_trace, power_trace, solar_s2)

    if verbose:
        print_log(log, f"MODE A LOG (loops={n_loops})")
        print(f"Arrival at Augrabies (local): {arrival_final_utc.astimezone(LOCAL_TZ):%H:%M}")
        print(f"Final SOC: {soc:.2f}%  ->  V(healthy 28mod)={voltage_from_soc_healthy(soc):.2f} V, "
              f"V(degraded 24mod, actual)={voltage_from_soc_degraded(soc):.2f} V")
        print()

    return {
        "mode": "A", "n_loops": n_loops, "trailer_speed_kmh": trailer_speed_kmh,
        "final_soc_pct": soc,
        "final_v_healthy": voltage_from_soc_healthy(soc), "final_v_degraded": voltage_from_soc_degraded(soc),
        "arrival_local": arrival_final_utc.astimezone(LOCAL_TZ),
        "cum_km_trace": cum_km_trace, "soc_trace": soc_trace, "solar_power_w_trace": power_trace,
        "log": log,
    }


# =======================================================================
# 6. MODE B -- no rain, grid-search the energy-neutral cruise speed
# =======================================================================

def mode_b_net_drive_wh(v_kmh: float, n_loops: int, depart_utc: datetime,
                         stage1, loop, stage2, solar_s1, solar_loop, solar_s2) -> float:
    """Net Wh of the DRIVING portion only (Stage1 + loops + Stage2, all at
    v_kmh), i.e. gross motor draw vs gross solar collected while moving --
    excludes the stationary control-stop/EOD charge bonuses. This is what
    gets driven to zero by the grid search ("energy neutral cruise speed")."""
    r1 = drive_leg_totals_only(stage1, depart_utc, solar_s1, v_kmh, n_laps=1)
    t_after_1 = r1["end_utc"]
    cs_end = t_after_1 + timedelta(minutes=CONTROL_STOP_BUFFER_MIN)
    r_loops = drive_leg_totals_only(loop, cs_end, solar_loop, v_kmh, n_laps=n_loops)
    # account for the 5-min stops' effect on *timing* (not energy, for this pass)
    t_after_loops = r_loops["end_utc"] + timedelta(minutes=LOOP_PENALTY_MIN * max(0, n_loops))
    r2 = drive_leg_totals_only(stage2, t_after_loops, solar_s2, v_kmh, n_laps=1)

    net = (r1["net_wh"]) + (r_loops["net_wh"]) + (r2["net_wh"])
    return net


def grid_search_equilibrium_velocity(n_loops: int, depart_utc: datetime, stage1, loop, stage2,
                                      solar_s1, solar_loop, solar_s2,
                                      v_min: float = 50.0, v_max: float = 70.0, step: float = 0.5) -> dict:
    vs = np.arange(v_min, v_max + step / 2, step)
    nets = np.array([mode_b_net_drive_wh(v, n_loops, depart_utc, stage1, loop, stage2,
                                          solar_s1, solar_loop, solar_s2) for v in vs])
    i_best = int(np.argmin(np.abs(nets)))
    v_best = float(vs[i_best])
    net_best = float(nets[i_best])

    # linear refine between the two closest bracketing points, if a sign change exists
    v_refined = v_best
    if 0 < i_best < len(vs) - 1:
        if nets[i_best - 1] * nets[i_best] < 0:
            v_lo, v_hi = vs[i_best - 1], vs[i_best]
            n_lo, n_hi = nets[i_best - 1], nets[i_best]
            v_refined = v_lo + (0 - n_lo) * (v_hi - v_lo) / (n_hi - n_lo)
        elif nets[i_best] * nets[i_best + 1] < 0:
            v_lo, v_hi = vs[i_best], vs[i_best + 1]
            n_lo, n_hi = nets[i_best], nets[i_best + 1]
            v_refined = v_lo + (0 - n_lo) * (v_hi - v_lo) / (n_hi - n_lo)

    return {"v_grid": vs, "net_grid": nets, "v_best": v_best, "net_best": net_best, "v_refined": v_refined}


def run_mode_b(n_loops: int, soc_at_departure_pct: float, verbose: bool = True,
               v_min: float = 50.0, v_max: float = 70.0, step: float = 0.5) -> dict:
    stage1 = load_route(ROUTE_FILES["stage1"], "Stage 1 (Olifantshoek->Upington)")
    loop = load_route(ROUTE_FILES["loop"], "Upington Loop")
    loop = truncate_route_first_last_km(loop, LOOP_TRUNCATE_KM_EACH)
    stage2 = load_route(ROUTE_FILES["stage2"], "Stage 2 (Upington->Augrabies)")

    solar_s1 = load_solar_series(SOLAR_FILES["stage1"])
    solar_loop = load_solar_series(SOLAR_FILES["loop"])
    solar_s2 = load_solar_series(SOLAR_FILES["stage2"])

    depart_utc = local_to_utc(8, 2)

    if verbose:
        print("=" * 78)
        print("MODE B: NO RAIN -- grid-searching the energy-neutral cruise speed")
        print("=" * 78)

    gs = grid_search_equilibrium_velocity(n_loops, depart_utc, stage1, loop, stage2,
                                           solar_s1, solar_loop, solar_s2, v_min, v_max, step)
    v_eq = round(gs["v_refined"], 1)

    if verbose:
        print(f"Grid search {v_min:.0f}-{v_max:.0f} km/h (step {step} km/h), n_loops={n_loops}:")
        print(f"  Best grid point: {gs['v_best']:.1f} km/h -> net drive Wh = {gs['net_best']:+.1f}")
        print(f"  Linearly refined equilibrium speed: {v_eq:.1f} km/h")
        for r, spd in [(stage1, v_eq), (loop, v_eq), (stage2, v_eq)]:
            check_gradient_feasible(r, spd)
        print()

    soc = soc_at_departure_pct
    log = [{"event": "08:02 departure (SOC as given -- no morning charge modeled, real value used)",
            "wh": 0.0, "d_soc_pct": 0.0, "soc_pct_after": soc, "note": f"equilibrium cruise speed = {v_eq:.1f} km/h"}]
    cum_km_trace = [0.0]
    soc_trace = [soc]
    power_trace = [0.0]

    trace1 = drive_leg_trace(stage1, depart_utc, solar_s1, v_eq, n_laps=1, cum_km_start=0.0, soc_start=soc)
    cum_km_trace.extend(trace1["cum_km_trace"][1:])
    soc_trace.extend(trace1["soc_trace"][1:])
    power_trace.extend(trace1["solar_power_w_trace"][1:])
    log.append({"event": f"Drive Stage 1 ({stage1.total_km:.1f} km @ {v_eq:.1f} km/h)",
                "wh": trace1["net_wh"], "d_soc_pct": None, "soc_pct_after": trace1["end_soc"],
                "note": f"motor={trace1['motor_wh']:.1f} Wh, regen={trace1['regen_wh']:.1f} Wh recovered, "
                        f"solar={trace1['solar_wh']:.1f} Wh, {trace1['duration_min']:.1f} min, "
                        f"arr {trace1['end_utc'].astimezone(LOCAL_TZ):%H:%M}"})
    soc = trace1["end_soc"]; cum_km = trace1["end_cum_km"]; cum_solar_wh = trace1["end_cum_solar_wh"]
    t = trace1["end_utc"]

    res = run_control_stop_and_loops(n_loops, t, soc, cum_km, cum_solar_wh, log,
                                      cum_km_trace, soc_trace, power_trace, loop, solar_loop,
                                      loop_speed_kmh=v_eq)
    soc, cum_km, cum_solar_wh, t = res["soc"], res["cum_km"], res["cum_solar_wh"], res["end_utc"]

    trace2 = drive_leg_trace(stage2, t, solar_s2, v_eq, n_laps=1, cum_km_start=cum_km, soc_start=soc)
    cum_km_trace.extend(trace2["cum_km_trace"][1:])
    soc_trace.extend(trace2["soc_trace"][1:])
    power_trace.extend(trace2["solar_power_w_trace"][1:])
    log.append({"event": f"Drive Stage 2 ({stage2.total_km:.1f} km @ {v_eq:.1f} km/h)",
                "wh": trace2["net_wh"], "d_soc_pct": None, "soc_pct_after": trace2["end_soc"],
                "note": f"motor={trace2['motor_wh']:.1f} Wh, regen={trace2['regen_wh']:.1f} Wh recovered, "
                        f"solar={trace2['solar_wh']:.1f} Wh, {trace2['duration_min']:.1f} min, "
                        f"arr {trace2['end_utc'].astimezone(LOCAL_TZ):%H:%M}"})
    soc = trace2["end_soc"]; cum_km = trace2["end_cum_km"]; cum_solar_wh += trace2["end_cum_solar_wh"]
    arrival_final_utc = trace2["end_utc"]

    soc = run_eod_charge(arrival_final_utc, soc, cum_km, cum_solar_wh, log,
                          cum_km_trace, soc_trace, power_trace, solar_s2)

    if verbose:
        print_log(log, f"MODE B LOG (loops={n_loops}, equilibrium speed={v_eq:.1f} km/h)")
        print(f"Arrival at Augrabies (local): {arrival_final_utc.astimezone(LOCAL_TZ):%H:%M}")
        print(f"Final SOC: {soc:.2f}%  ->  V(healthy 28mod)={voltage_from_soc_healthy(soc):.2f} V, "
              f"V(degraded 24mod, actual)={voltage_from_soc_degraded(soc):.2f} V")
        print()

    return {
        "mode": "B", "n_loops": n_loops, "v_equilibrium_kmh": v_eq, "final_soc_pct": soc,
        "final_v_healthy": voltage_from_soc_healthy(soc), "final_v_degraded": voltage_from_soc_degraded(soc),
        "arrival_local": arrival_final_utc.astimezone(LOCAL_TZ),
        "cum_km_trace": cum_km_trace, "soc_trace": soc_trace, "solar_power_w_trace": power_trace,
        "log": log, "grid_search": gs,
    }


# =======================================================================
# 7. CLI
# =======================================================================

if __name__ == "__main__":
    import sys

    print_battery_summary()

    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "A"

    if mode == "A":
        # 3 arguments: n_loops, soc_at_control_stop_pct, trailer_speed_kmh
        n_loops = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        soc_in = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
        trailer_speed_in = float(sys.argv[4]) if len(sys.argv) > 4 else TRAILER_SPEED_KMH
        result = run_mode_a(n_loops, soc_in, trailer_speed_in)
    elif mode == "B":
        # 2 arguments: n_loops, soc_at_departure_pct
        n_loops = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        soc_in = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
        result = run_mode_b(n_loops, soc_in)
    else:
        print(f"Unknown mode '{mode}'. Usage:")
        print("  python day5_strategy_loop2.py A <n_loops> <soc_at_control_stop_pct> <trailer_speed_kmh>")
        print("  python day5_strategy_loop2.py B <n_loops> <soc_at_departure_pct>")
        sys.exit(1)

    # ---- SOC-vs-distance and Solar-input(W)-vs-distance PLOTS ----
    km = result["cum_km_trace"]
    soc_t = result["soc_trace"]
    pw_t = result["solar_power_w_trace"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(km, soc_t, color="tab:blue", linewidth=1.5)
    axes[0].set_ylabel("SOC (%)")
    axes[0].set_title(f"Mode {result['mode']} (loops={result['n_loops']}) -- SOC vs Distance")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(km, pw_t, color="tab:orange", linewidth=1.0)
    axes[1].set_ylabel("Solar input (W)")
    axes[1].set_xlabel("Cumulative distance (km)")
    axes[1].set_title("Solar Input (W) vs Distance")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = DASH / f"day5_loop2_mode{result['mode']}_loops{result['n_loops']}_plots.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved plots to {out_png}")
    plt.show()
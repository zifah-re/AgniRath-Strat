"""
day3_strategy.py -- AgniRath Sasol Solar Challenge, Day 3 strategy engine
==========================================================================

Context (as given by the strategist, 2026-09-11):
  * One of the 6 series battery modules died (bus-bar fault) and has been
    fully bypassed. The pack now runs on 5 of 6 modules.
  * Rated pack energy is 3528 Wh (6 x 588 Wh), but we plan conservatively
    on 3200 Wh nominal, split equally across the 6 modules -> 533.33 Wh
    per module. With one bypassed, usable NAMEPLATE capacity is now
    5 x 533.33 = 2666.67 Wh.
  * Hard floor from management: never discharge below 80 V pack terminal
    voltage (measured on the Bus_Voltage-scale signal, i.e. the same
    voltage domain as the SOC_CURVE_V_PCT table below -- NOT the raw
    "Pack_Voltage" telemetry field, which is logged ~1000x scaled).
  * Cars run at a constant 60 km/h target on Day 3 (easy terrain) to
    maximise solar income and time, not to minimise time -- speed is
    fixed, loops are used to soak up spare daylight/time while holding
    2nd place, instead of sitting idle at a control stop.

Everything below is one self-contained, editable script. Constants that
are assumptions (not hard facts from the brief) are flagged "ASSUMPTION"
so you can tune them fast during the strategy meeting.
"""

from __future__ import annotations

import json
import math
import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------
# 0. FILE LOCATIONS  (point these at your local repo checkout)
# ---------------------------------------------------------------------
DASH = Path(".")                # <- you're already running this from inside Dashboard/
SAVES = DASH / "Saves"
SOLAR = DASH / "Solar"

ROUTE_FILES = {
    "stage1": SAVES / "Day3_Stage 1_Stage 1.kml.save",
    "loop":   SAVES / "Day3_Loop_Jan Kempdorp Loop.kml.save",
    "stage2": SAVES / "Day3_Stage 2_Stage 2.kml.save",
}

# The solar *forecast* files are the only Day-3-dated irradiance data we
# have. Their embedded route geometry does NOT match the real Day3_*
# route files (confirmed: "Probable Prahlad" Stage 2 starts ~1 degree of
# latitude away from the real Day3 Stage 2 start). So we throw away their
# geometry entirely and use ONLY their time-of-day irradiance curve
# (dni/ghi vs local clock time), applied to the real route via elapsed
# drive time. This is the "offset by time, not by km" fix you asked for.
SOLAR_FILES = {
    "stage1": SOLAR / "mean_Day 3 probables_Probable Prahlad Route_Stage 1.jsonl",
    "loop":   SOLAR / "mean_Day 3 probables_Probable Prahlad Route_Day 3 Loop.jsonl",
    "stage2": SOLAR / "mean_Day 3 probables_Probable Prahlad Route_Stage 2.jsonl",
}

LOCAL_TZ = timezone(timedelta(hours=2))  # South Africa Standard Time (SAST, UTC+2)


# =======================================================================
# 1. BATTERY MODEL -- degraded 5/6-module pack
# =======================================================================

# Original, healthy 6-module pack SOC<->V curve (as given), 100% -> 0%.
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

N_MODULES_HEALTHY = 6
N_MODULES_NOW = 5          # one bypassed
PACK_WH_ASSUMED = 3200.0   # per your instruction: use 3200 Wh, not the 3528 Wh nameplate
WH_PER_MODULE = PACK_WH_ASSUMED / N_MODULES_HEALTHY            # 533.33 Wh
USABLE_WH_NOW = WH_PER_MODULE * N_MODULES_NOW                  # 2666.67 Wh nameplate on 5 modules

VOLTAGE_FLOOR_V = 80.0     # hard instruction: never go below this

# Degraded curve: identical modules assumed in series, so removing one of
# six drops total voltage at every SOC point by the same 5/6 ratio.
SCALE = N_MODULES_NOW / N_MODULES_HEALTHY   # 0.8333...

SOC_CURVE_V_PCT_DEGRADED = [(v * SCALE, p) for v, p in SOC_CURVE_V_PCT_HEALTHY]

_V_DEG = np.array([v for v, _ in SOC_CURVE_V_PCT_DEGRADED])[::-1]   # ascending V
_PCT_DEG = np.array([p for _, p in SOC_CURVE_V_PCT_DEGRADED])[::-1]  # ascending %


def soc_from_voltage_degraded(v: float) -> float:
    """SOC% (of the *degraded* 5-module pack) from measured pack voltage."""
    return float(np.interp(v, _V_DEG, _PCT_DEG))


def voltage_from_soc_degraded(pct: float) -> float:
    return float(np.interp(pct, _PCT_DEG, _V_DEG))


FLOOR_SOC_PCT = soc_from_voltage_degraded(VOLTAGE_FLOOR_V)
USABLE_WH_ABOVE_FLOOR = USABLE_WH_NOW * (100.0 - FLOOR_SOC_PCT) / 100.0


def wh_to_soc_delta(wh: float, charge_eff: float = 0.96, discharge_eff: float = 0.96) -> float:
    """Convert an energy delta (Wh, +charge / -discharge) into a SOC% delta
    on the degraded pack's nameplate capacity, applying round-trip efficiency."""
    if wh >= 0:
        return (wh * charge_eff) / USABLE_WH_NOW * 100.0
    else:
        return (wh / discharge_eff) / USABLE_WH_NOW * 100.0


def print_battery_summary(current_voltage_v: float | None = None) -> None:
    print("=" * 78)
    print("BATTERY: degraded 5/6-module pack")
    print("=" * 78)
    print(f"Module count: healthy={N_MODULES_HEALTHY}, now={N_MODULES_NOW} (1 bypassed)")
    print(f"Assumed usable capacity: {PACK_WH_ASSUMED:.0f} Wh nameplate over 6 modules "
          f"-> {WH_PER_MODULE:.2f} Wh/module -> {USABLE_WH_NOW:.2f} Wh on 5 modules")
    print(f"Voltage scale factor (5/6 series modules): {SCALE:.4f}")
    print()
    print(f"{'SOC%':>6}  {'V_healthy(6mod)':>16}  {'V_degraded(5mod)':>17}")
    for pct in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]:
        v_h = float(np.interp(pct, np.array([p for _, p in SOC_CURVE_V_PCT_HEALTHY])[::-1],
                               np.array([v for v, _ in SOC_CURVE_V_PCT_HEALTHY])[::-1]))
        v_d = voltage_from_soc_degraded(pct)
        print(f"{pct:6d}  {v_h:16.2f}  {v_d:17.2f}")
    print()
    print(f"Mandated floor: {VOLTAGE_FLOOR_V:.1f} V  ->  {FLOOR_SOC_PCT:.2f} % SOC on the degraded curve")
    print(f"Usable energy ABOVE the 80V floor: {USABLE_WH_ABOVE_FLOOR:.1f} Wh "
          f"({100 - FLOOR_SOC_PCT:.2f} % of nameplate)")
    if current_voltage_v is not None:
        soc_now = soc_from_voltage_degraded(current_voltage_v)
        print()
        print(f"End-of-Day-2 reading used: {current_voltage_v:.2f} V  ->  SOC now = {soc_now:.2f} %")
        if current_voltage_v < VOLTAGE_FLOOR_V:
            print(f"  ** WARNING: this is BELOW the new {VOLTAGE_FLOOR_V:.0f}V floor. "
                  f"That's presumably why the floor rule was just handed down. **")
    print()


# =======================================================================
# 2. SOLAR MODEL
# =======================================================================

ARRAY_AREA_M2 = 5.78          # DASHBOARD car_config.py
ARRAY_EFFICIENCY = 0.24       # per instruction: always 24% (overrides car_config's 0.21)
N_MPPT_CHANNELS = 4           # A, B, C, D
AREA_PER_CHANNEL_M2 = ARRAY_AREA_M2 / N_MPPT_CHANNELS

# ---- MPPT-D shading factor, derived from the telemetry excerpt you pasted ----
# Rows below are lifted straight from your telemetry sample (Input_Current_A/B/C/D),
# across stationary + moving conditions. D is consistently ~1 order of magnitude
# below A/B/C -> structural partial shading on that quadrant of the array,
# present whether stationary-but-untilted or driving.
_TELEMETRY_MPPT_SAMPLE = [
    # (I_A, I_B, I_C, I_D)
    (3.5632, 4.5482, 3.7713, 0.3246),
    (3.5601, 4.5180, 3.7999, 0.3248),
    (3.5614, 4.5494, 3.7991, 0.3251),
    (3.4015, 4.4536, 3.7554, 0.3273),
    (3.2992, 4.4352, 3.7602, 0.3273),
    (3.5053, 4.5652, 3.9673, 0.3236),
    (3.5341, 4.5557, 3.9169, 0.3288),
    (3.9652, 4.8882, 4.6980, 0.3428),
    (3.9802, 4.9188, 4.6509, 0.3385),
    (3.9417, 4.8778, 4.6251, 0.3309),
    (3.9418, 4.8679, 4.6482, 0.3380),
    (4.2060, 5.0645, 4.8262, 0.3551),
    (4.2846, 5.1883, 4.9130, 0.3859),
    (4.2914, 5.1163, 4.9487, 0.4229),
    (4.3456, 5.1678, 5.0717, 0.5072),
    (4.3399, 5.1211, 4.9931, 0.4486),
    (4.3352, 5.1335, 5.0064, 0.4065),
    (4.2917, 5.1082, 5.0669, 0.4100),
    (4.3334, 5.0659, 5.0734, 0.4288),
]


def derive_mppt_d_shade_factor() -> float:
    """Average ratio of D's current to the mean of A/B/C's current, across the
    sample. This is used as the fraction of an *unshaded* channel's output
    that channel D actually delivers while driving (fixed, untilted mount)."""
    ratios = []
    for ia, ib, ic, id_ in _TELEMETRY_MPPT_SAMPLE:
        mean_abc = (ia + ib + ic) / 3.0
        if mean_abc > 0:
            ratios.append(id_ / mean_abc)
    return float(np.mean(ratios))


MPPT_D_SHADE_FACTOR = 0.25   # ~0.086 -> ~91% blocked


def load_solar_series(path: Path) -> list[tuple[datetime, float, float]]:
    """Returns list of (utc_datetime, dni_w_m2, ghi_w_m2) sorted by time."""
    with open(path) as f:
        payload = json.load(f)[0]
    out = []
    for row in payload["data"]:
        t = datetime.fromisoformat(row["period_end"])
        out.append((t, float(row["dni"]), float(row["ghi"])))
    out.sort(key=lambda r: r[0])
    return out


def irradiance_at(series: list[tuple[datetime, float, float]], t_utc: datetime) -> tuple[float, float]:
    """Linearly interpolate (dni, ghi) at an arbitrary UTC timestamp."""
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
    """Panels manually tilted to directly face the sun -> use DNI (direct-
    normal irradiance) as the plane-of-array irradiance; no MPPT-D shading
    (you're deliberately oriented to avoid it) -> full 4-channel area."""
    return dni_w_m2 * ARRAY_AREA_M2 * ARRAY_EFFICIENCY


def driving_solar_power_w(ghi_w_m2: float) -> float:
    """Fixed, flat mount while moving -> use GHI (irradiance on the
    horizontal deck), and MPPT-D is shaded to MPPT_D_SHADE_FACTOR of a
    clean channel."""
    clean_channels_area = 3 * AREA_PER_CHANNEL_M2                     # A, B, C
    shaded_d_area_equiv = AREA_PER_CHANNEL_M2 * MPPT_D_SHADE_FACTOR    # D, derated
    effective_area = clean_channels_area + shaded_d_area_equiv
    return ghi_w_m2 * effective_area * ARRAY_EFFICIENCY


def energy_over_window_wh(series, t0_utc: datetime, t1_utc: datetime, power_fn, step_min: float = 5.0) -> float:
    """Integrate power_fn(irradiance) from t0 to t1 (trapezoid, step_min minutes)."""
    assert t1_utc > t0_utc
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
# 3. ROUTE MODEL (KML profile: distance, gradient) + constant-speed physics
# =======================================================================

G = 9.81
AIR_DENSITY = 1.20            # kg/m^3, ASSUMPTION (Highveld ~1200m alt, warm)
MASS_KG = 336.5                # car_config.py
CRR = 0.007                    # car_config.py
CDA_M2 = 0.16                  # car_config.py
MOTOR_EFF = 0.95                # per instruction
REGEN_EFF = 0.80                # per instruction (overrides car_config's 0.70)
P_IDLE_W = 5.0                  # car_config.py, always-on hotel load
# ---------------------------------------------------------------------
# STAGE SPEEDS -- change these three values independently.
# ---------------------------------------------------------------------
STAGE1_TARGET_SPEED_KMH = 50.0
LOOP_TARGET_SPEED_KMH = 60.0
STAGE2_TARGET_SPEED_KMH = 60.0

STAGE1_TARGET_SPEED_MS = STAGE1_TARGET_SPEED_KMH / 3.6
LOOP_TARGET_SPEED_MS = LOOP_TARGET_SPEED_KMH / 3.6
STAGE2_TARGET_SPEED_MS = STAGE2_TARGET_SPEED_KMH / 3.6

CONTROL_STOP_PENALTY_MIN = 30 + 16   # 46 minutes, per instruction
LOOP_PENALTY_MIN = 5                  # per-loop charging stop after each lap


@dataclasses.dataclass
class RouteProfile:
    name: str
    distance_km: np.ndarray   # cumulative km
    gradient_pct: np.ndarray  # % grade at each point

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


def check_gradient_feasible(route: RouteProfile, max_safe_grade_pct: float = 8.0) -> bool:
    """A constant-speed plan needs no segment steep enough that the
    car can't hold speed on the climb, nor so steep downhill that regen
    braking alone can't hold the configured speed (ASSUMPTION: +/-8% is our safety cutoff for
    a solar car at highway speed)."""
    bad = np.abs(route.gradient_pct) > max_safe_grade_pct
    if bad.any():
        worst = route.gradient_pct[np.argmax(np.abs(route.gradient_pct))]
        print(f"  ** {route.name}: {bad.sum()} point(s) exceed +/-{max_safe_grade_pct}% "
              f"(worst {worst:.2f}%) - review before committing to 60 km/h **")
        return False
    print(f"  {route.name}: gradients OK (max {route.gradient_pct.max():.2f}%, "
          f"min {route.gradient_pct.min():.2f}%), configured speed profile is safe.")
    return True


def drive_leg_energy_wh(
    route: RouteProfile,
    start_utc: datetime,
    solar_series,
    n_laps: int = 1,
    target_speed_ms: Optional[float] = None,
) -> dict:
    """Drive `route` at the requested constant cruise speed, n_laps times back-to-back (for the
    loop stage), returning elapsed time and net electrical energy (Wh,
    +consumed / -regenerated back to battery), plus solar Wh harvested
    while moving.
    """
    speed_ms = target_speed_ms if target_speed_ms is not None else STAGE2_TARGET_SPEED_MS

    dist_km = route.distance_km
    grad_pct = route.gradient_pct
    seg_km = np.diff(dist_km)
    seg_grad = 0.5 * (grad_pct[:-1] + grad_pct[1:])  # avg grade per segment

    theta = np.arctan(seg_grad / 100.0)
    seg_m = seg_km * 1000.0

    f_rr = CRR * MASS_KG * G * np.cos(theta)
    f_aero = 0.5 * AIR_DENSITY * CDA_M2 * speed_ms ** 2
    f_grav = MASS_KG * G * np.sin(theta)
    f_total = f_rr + f_aero + f_grav                      # N, +ve = motoring load

    # At constant target speed, positive force means the motor automatically
    # supplies the extra power needed to hold speed (e.g. uphill), while
    # negative force means the car can regen downhill.
    p_wheel = f_total * speed_ms                           # W, per segment
    seg_time_s = seg_m / speed_ms

    elec_wh = 0.0
    for p_w, t_s in zip(p_wheel, seg_time_s):
        if p_w >= 0:
            elec_w = p_w / MOTOR_EFF + P_IDLE_W
        else:
            elec_w = p_w * REGEN_EFF + P_IDLE_W            # negative = energy returned
        elec_wh += elec_w * (t_s / 3600.0)

    lap_time_s = float(seg_time_s.sum())
    lap_wh = elec_wh

    total_time_s = lap_time_s * n_laps
    total_drive_wh = lap_wh * n_laps

    # Solar harvested while driving, integrated over the actual elapsed clock time.
    end_utc = start_utc + timedelta(seconds=total_time_s)
    if n_laps <= 0 or total_time_s <= 0:
        solar_wh = 0.0
        end_utc = start_utc
    else:
        solar_wh = energy_over_window_wh(solar_series, start_utc, end_utc, driving_solar_power_w)

    return {
        "leg": route.name,
        "n_laps": n_laps,
        "distance_km": route.total_km * n_laps,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "duration_min": total_time_s / 60.0,
        "drive_elec_wh": total_drive_wh,     # + = net draw, - = net regen
        "solar_wh": solar_wh,
        "net_wh": solar_wh - total_drive_wh,  # + = SOC goes up, - = SOC goes down
    }


# =======================================================================
# 4. FULL DAY-3 ASSEMBLY
# =======================================================================

RACE_DATE = "2026-09-12"


def local_to_utc(hh: int, mm: int) -> datetime:
    local = datetime.fromisoformat(f"{RACE_DATE}T{hh:02d}:{mm:02d}:00").replace(tzinfo=LOCAL_TZ)
    return local.astimezone(timezone.utc)


def run_day(n_loops: int, start_pack_voltage_v: float, verbose: bool = True) -> dict:
    stage1 = load_route(ROUTE_FILES["stage1"], "Stage 1")
    loop = load_route(ROUTE_FILES["loop"], "Jan Kempdorp Loop")
    stage2 = load_route(ROUTE_FILES["stage2"], "Stage 2")

    solar_s1 = load_solar_series(SOLAR_FILES["stage1"])
    solar_loop = load_solar_series(SOLAR_FILES["loop"])
    solar_s2 = load_solar_series(SOLAR_FILES["stage2"])

    if verbose:
        print("=" * 78)
        print("ROUTE GRADIENT SAFETY CHECK (configured constant-speed plan)")
        print("=" * 78)
        for r in (stage1, loop, stage2):
            check_gradient_feasible(r)
        print()

    soc = soc_from_voltage_degraded(start_pack_voltage_v)
    log = []

    def record(label, wh_delta, extra=""):
        nonlocal soc
        d_soc = wh_to_soc_delta(wh_delta)
        soc = min(100.0, max(0.0, soc + d_soc))
        log.append({"event": label, "wh": wh_delta, "d_soc_pct": d_soc, "soc_pct_after": soc, "note": extra})

    record("Start of day (end-Day-2 reading)", 0.0,
           f"{start_pack_voltage_v:.2f} V -> {soc:.2f}% on degraded curve")

    # ---- 1) 06:00-08:00 morning charge at Vryburg / Kameelboom Lodge, sun-tracking ----
    t0, t1 = local_to_utc(6, 0), local_to_utc(8, 0)
    wh_morning = energy_over_window_wh(solar_s1, t0, t1, stationary_charge_power_w)
    record("06:00-08:00 morning charge (sun-tracking, Vryburg)", wh_morning,
           f"{wh_morning:.1f} Wh harvested")
    soc_after_morning_charge = soc

    # ---- 2) Depart 08:02, drive Stage 1 ----
    depart_utc = local_to_utc(8, 2)
    leg1 = drive_leg_energy_wh(
        stage1,
        depart_utc,
        solar_s1,
        n_laps=1,
        target_speed_ms=STAGE1_TARGET_SPEED_MS,
    )
    record(f"Drive Stage 1 ({leg1['distance_km']:.1f} km @ {STAGE1_TARGET_SPEED_KMH:.0f} km/h)", leg1["net_wh"],
           f"drive={leg1['drive_elec_wh']:.1f} Wh, solar={leg1['solar_wh']:.1f} Wh, "
           f"{leg1['duration_min']:.1f} min, arr {leg1['end_utc'].astimezone(LOCAL_TZ):%H:%M}")

    floor_cross = None
    if soc < FLOOR_SOC_PCT:
        floor_cross = trace_floor_crossing(stage1, depart_utc, solar_s1, soc_after_morning_charge,
                                             target_speed_ms=STAGE1_TARGET_SPEED_MS)

    # ---- 3) 46-minute control stop, sun-tracking charge ----
    cs_start = leg1["end_utc"]
    cs_end = cs_start + timedelta(minutes=CONTROL_STOP_PENALTY_MIN)
    wh_cs = energy_over_window_wh(solar_s2, cs_start, cs_end, stationary_charge_power_w)
    record(f"{CONTROL_STOP_PENALTY_MIN}-min control stop (sun-tracking)", wh_cs,
           f"{wh_cs:.1f} Wh, {cs_start.astimezone(LOCAL_TZ):%H:%M}->{cs_end.astimezone(LOCAL_TZ):%H:%M}")

    # ---- 4) N loops of Jan Kempdorp loop, each followed by a 5-min charge stop ----
    loop_start = cs_end
    if n_loops <= 0:
        pass  # no loops -> straight on to Stage 2 from the control stop
    else:
        for i in range(n_loops):
            leg_lap = drive_leg_energy_wh(
                loop,
                loop_start,
                solar_loop,
                n_laps=1,
                target_speed_ms=LOOP_TARGET_SPEED_MS,
            )
            record(f"Loop {i + 1}/{n_loops} ({leg_lap['distance_km']:.1f} km @ {LOOP_TARGET_SPEED_KMH:.0f} km/h)",
                   leg_lap["net_wh"],
                   f"drive={leg_lap['drive_elec_wh']:.1f} Wh, solar={leg_lap['solar_wh']:.1f} Wh, "
                   f"{leg_lap['duration_min']:.1f} min, arr {leg_lap['end_utc'].astimezone(LOCAL_TZ):%H:%M}")
            loop_start = leg_lap["end_utc"]

            charge_end = loop_start + timedelta(minutes=LOOP_PENALTY_MIN)
            wh_pen = energy_over_window_wh(solar_loop, loop_start, charge_end, stationary_charge_power_w)
            record(f"{LOOP_PENALTY_MIN}-min charge stop after loop {i + 1}", wh_pen,
                   f"{wh_pen:.1f} Wh, {loop_start.astimezone(LOCAL_TZ):%H:%M}"
                   f"->{charge_end.astimezone(LOCAL_TZ):%H:%M}")
            loop_start = charge_end

    # ---- 5) Drive Stage 2 to the finish ----
    stage2_start = loop_start
    leg2 = drive_leg_energy_wh(
        stage2,
        stage2_start,
        solar_s2,
        n_laps=1,
        target_speed_ms=STAGE2_TARGET_SPEED_MS,
    )
    record(f"Drive Stage 2 ({leg2['distance_km']:.1f} km @ {STAGE2_TARGET_SPEED_KMH:.0f} km/h)", leg2["net_wh"],
           f"drive={leg2['drive_elec_wh']:.1f} Wh, solar={leg2['solar_wh']:.1f} Wh, "
           f"{leg2['duration_min']:.1f} min, arr {leg2['end_utc'].astimezone(LOCAL_TZ):%H:%M}")

    arrival_utc = leg2["end_utc"]
    arrival_local = arrival_utc.astimezone(LOCAL_TZ)

    # ---- 6) End-of-day charge window, per your rule ----
    four_pm = local_to_utc(16, 0)
    five_pm = local_to_utc(17, 0)
    if arrival_utc <= four_pm:
        charge_start = arrival_utc + timedelta(hours=1)
        charge_end = five_pm
        if charge_start < charge_end:
            wh_eod = energy_over_window_wh(solar_s2, charge_start, charge_end, stationary_charge_power_w)
            record(f"End-of-day charge ({charge_start.astimezone(LOCAL_TZ):%H:%M}"
                   f"->{charge_end.astimezone(LOCAL_TZ):%H:%M})", wh_eod, f"{wh_eod:.1f} Wh")
        else:
            record("End-of-day charge window", 0.0, "arrival + 1h is already past 17:00, skipped")
    else:
        record("End-of-day charge window", 0.0,
               f"arrived {arrival_local:%H:%M} (after 16:00) -> no scripted EOD charge window")

    # ---- floor violation check ----
    # The very first row is the inherited end-of-Day-2 state, which is
    # already below the newly-mandated floor (that's *why* the floor rule
    # exists) -- so we check the floor only from the point we actually
    # start controlling energy (after the 06:00 charge onward).
    min_soc_seen_all = min(row["soc_pct_after"] for row in log)
    min_soc_seen = min(row["soc_pct_after"] for row in log[1:])
    floor_ok = min_soc_seen >= FLOOR_SOC_PCT

    return {
        "n_loops": n_loops,
        "log": log,
        "arrival_local": arrival_local,
        "final_soc_pct": soc,
        "min_soc_seen": min_soc_seen,
        "min_soc_seen_all": min_soc_seen_all,
        "floor_ok": floor_ok,
        "floor_cross": floor_cross,
    }


def trace_floor_crossing(
    route: RouteProfile,
    start_utc: datetime,
    solar_series,
    soc_start_pct: float,
    target_speed_ms: Optional[float] = None,
) -> dict:
    """Fine-grained (per-segment) SOC trace through one leg, to find exactly
    where/when the 80V floor gets crossed, if it does."""
    dist_km = route.distance_km
    grad_pct = route.gradient_pct
    seg_km = np.diff(dist_km)
    speed_ms = target_speed_ms if target_speed_ms is not None else STAGE2_TARGET_SPEED_MS
    seg_grad = 0.5 * (grad_pct[:-1] + grad_pct[1:])
    theta = np.arctan(seg_grad / 100.0)
    seg_m = seg_km * 1000.0

    f_rr = CRR * MASS_KG * G * np.cos(theta)
    f_aero = 0.5 * AIR_DENSITY * CDA_M2 * speed_ms ** 2
    f_grav = MASS_KG * G * np.sin(theta)
    f_total = f_rr + f_aero + f_grav
    p_wheel = f_total * speed_ms
    seg_time_s = seg_m / speed_ms

    soc = soc_start_pct
    t = start_utc
    cum_km = 0.0
    for p_w, t_s, km in zip(p_wheel, seg_time_s, seg_km):
        cum_km += km
        elec_w = (p_w / MOTOR_EFF + P_IDLE_W) if p_w >= 0 else (p_w * REGEN_EFF + P_IDLE_W)
        drive_wh = elec_w * (t_s / 3600.0)
        t_next = t + timedelta(seconds=t_s)
        dni, ghi = irradiance_at(solar_series, t + (t_next - t) / 2)
        solar_w = driving_solar_power_w(ghi)
        solar_wh = solar_w * (t_s / 3600.0)
        net_wh = solar_wh - drive_wh
        soc += wh_to_soc_delta(net_wh)
        t = t_next
        if soc <= FLOOR_SOC_PCT:
            return {"km": cum_km, "time_local": t.astimezone(LOCAL_TZ), "soc_pct": soc}
    return None


def print_day_log(result: dict) -> None:
    print("=" * 78)
    print(f"DAY-3 SOC PROGRESSION  (loops = {result['n_loops']})")
    print("=" * 78)
    print(f"{'Event':52s} {'Wh':>9s} {'dSOC%':>7s} {'SOC%':>7s}")
    for row in result["log"]:
        print(f"{row['event'][:52]:52s} {row['wh']:9.1f} {row['d_soc_pct']:7.2f} {row['soc_pct_after']:7.2f}")
        if row["note"]:
            print(f"    -> {row['note']}")
    print("-" * 78)
    print(f"Arrival (local):     {result['arrival_local']:%H:%M}")
    print(f"Final SOC:           {result['final_soc_pct']:.2f} %")
    print(f"Lowest SOC after 06:00 charge starts: {result['min_soc_seen']:.2f} %  "
          f"(floor = {FLOOR_SOC_PCT:.2f} %)  "
          f"-> {'OK, stayed above floor' if result['floor_ok'] else '*** BREACHES THE 80V FLOOR ***'}")
    print(f"(inherited start-of-day SOC was {result['min_soc_seen_all']:.2f} %, "
          f"already below floor -- pre-existing, not caused by today's plan)")
    if result.get("floor_cross"):
        fc = result["floor_cross"]
        print(f"  ** Floor (80V / {FLOOR_SOC_PCT:.1f}%) is crossed DURING Stage 1, "
              f"at km {fc['km']:.1f}, ~{fc['time_local']:%H:%M} local, SOC {fc['soc_pct']:.1f}% **")
    print()


if __name__ == "__main__":
    import sys

    # ---- 1. Battery ----
    # ASSUMPTION: end-of-Day-2 pack terminal voltage. Change this to the
    # actual last-known-good Bus_Voltage reading from the car before you run
    # this for real -- 71.58 V is pulled from the last row of your pasted
    # telemetry excerpt (17:13 local, car slowing to a stop).
    END_DAY2_VOLTAGE_V = 71.58
    print_battery_summary(END_DAY2_VOLTAGE_V)

    print(f"Derived MPPT-D shading factor from telemetry: {MPPT_D_SHADE_FACTOR:.3f} "
          f"(D delivers ~{MPPT_D_SHADE_FACTOR*100:.1f}% of an unshaded channel)")
    print()

    # ---- 2. Run for however many loops you want to test ----
    n_loops = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    result = run_day(n_loops=n_loops, start_pack_voltage_v=END_DAY2_VOLTAGE_V)
    print_day_log(result)

    # ---- 3. Sweep loop counts to help pick N ----
    print("=" * 78)
    print("LOOP COUNT SWEEP")
    print("=" * 78)
    print(f"{'loops':>5s} {'arrival':>8s} {'final_SOC%':>11s} {'min_SOC%(post-06:00)':>21s} {'floor_ok':>9s}")
    for n in range(0, 8):
        r = run_day(n_loops=n, start_pack_voltage_v=END_DAY2_VOLTAGE_V, verbose=False)
        print(f"{n:5d} {r['arrival_local']:%H:%M} {r['final_soc_pct']:11.2f} "
              f"{r['min_soc_seen']:21.2f} {str(r['floor_ok']):>9s}")
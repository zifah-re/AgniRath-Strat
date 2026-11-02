"""2027 BWSC time-minimisation simulation with 5-Segment Global Optimization."""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import differential_evolution
from tqdm import tqdm

from solar_table import SolarIrradiance

# ============================ CONFIGURATION ================================
ACST = ZoneInfo("Australia/Darwin")
RACE_START_DATE = date(2027, 8, 22) # Reg 1.12.2: Official Start
MAX_RACE_DAYS = 6                   # Strict BWSC 6-day race limit
DRIVE_START = time(8, 0)            # Reg 3.21.1: 08:00 start time
DRIVE_END = time(17, 0)             # Reg 3.21.3: 17:00 finish of driving day

# Reg 3.28.8 & 3.17.2: Evening sealing and morning unsealing
CHARGE_START = time(6, 15)          
CHARGE_END = time(19, 15)           

N_SPEED_SPLITS = 5                  # 5 macro-segments (roughly 1 per day)
SPEED_MIN_KMH = 60.0                # Reg 3.29.1: Minimum open-road speed
SPEED_MAX_KMH = 110.0               # Open highway legal ceiling
BREAKDOWN_SECONDS_PER_DAY = 0       

INITIAL_SOC_PCT = 95.0
SOC_MIN_PCT = 20.0
SOC_MAX_PCT = 95.0
RACE_FINISH_KM = 3022.0             
ROUTE_STEP_KM = 10.0                # Fast macro-evaluation resolution

# Reg 3.27.8: Mandatory 30-min Control Stops
CONTROL_STOPS_KM = {
    "Katherine": 318.0, "Dunmarra": 635.0, "Tennant Creek": 988.0,
    "Barrow Creek": 1210.0, "Alice Springs": 1493.0, "Erldunda": 1693.0,
    "Coober Pedy": 2179.0, "Glendambo": 2432.0, "Port Augusta": 2719.0,
}
CONTROL_STOP_SECONDS = 30 * 60

# Reg 3.23.4: Compulsory 5-minute Driver Changes
DRIVER_CHANGES_KM = [160.0, 475.0, 810.0, 1100.0, 1350.0, 1850.0, 2020.0, 2300.0, 2580.0, 2870.0]
DRIVER_CHANGE_SECONDS = 5 * 60

# Vehicle & Array Physics Constants
MASS_KG, G_MS2, CRR, CDA_M2, AIR_DENSITY = 320.0, 9.81, 0.007, 0.16, 1.2
ARRAY_AREA_M2, ARRAY_EFFICIENCY, PANEL_TILT, ALBEDO = 5.78, 0.21, 4.0, 0.2
MOTOR_EFF, REGEN_EFF, P_LOSS_W, P_AUX = 0.95, 0.70, 70.0, 5.0
MAX_MOTOR_POWER_W = 5000.0
BATTERY_WH = 11_000_000.0 / 3600.0  # Reg 2.5.2: 11 MJ max (3055.56 Wh)
TEMP_COEFF_PER_C, NOCT_C, K_CONVECTIVE_COOLING = 0.004, 45.0, 0.03
T_AMBIENT_DEFAULT = 28.0            # Fallback for missing JSON weather data

# Transposition geometric constants 
sky_factor = (1.0 + np.cos(np.radians(PANEL_TILT))) / 2.0
ground_factor = (1.0 - np.cos(np.radians(PANEL_TILT))) / 2.0
b_constants = sky_factor + ALBEDO * ground_factor

# Original file & directory paths
HERE = Path(__file__).resolve().parent
MASTER_ROUTE = HERE / "WSC_WSC Route_WSC Route.kml.save"
SOLAR_DIR = HERE / "Solar"
ROUTE_SECTIONS = (
    (0.0, 618.95, "Day1_WSC_Day 1_solar.json"),
    (618.95, 1232.95, "Day2_WSC_Day 2_solar.json"),
    (1232.95, 1840.10, "Day3_WSC_Day 3_solar.json"),
    (1840.10, 2454.51, "Day4_WSC_Day 4_solar.json"),
    (2454.51, RACE_FINISH_KM, "Day5_WSC_Day 5_solar.json"),
)

class SolarCoverageError(RuntimeError):
    pass

@dataclass
class Route:
    distance_km: np.ndarray
    coords: np.ndarray
    gradient_pct: np.ndarray
    heading_deg: np.ndarray

@dataclass
class SimulationResult:
    distance_km: np.ndarray
    soc_pct: np.ndarray
    timestamps: list[datetime]
    day_index: np.ndarray
    finish_time: datetime | None
    min_soc_pct: float
    max_motor_power_w: float
    speed_kmh: np.ndarray


@functools.lru_cache(maxsize=65536)
def cached_solarpos(timestamp_rounded: int, lat_rounded: float, lon_rounded: float) -> tuple[float, float]:
    dt = datetime.fromtimestamp(timestamp_rounded, tz=ACST)
    solpos = pvlib.solarposition.ephemeris(pd.DatetimeIndex([dt]), lat_rounded, lon_rounded)
    return float(solpos['apparent_zenith'].iloc[0]), float(solpos['azimuth'].iloc[0])


def load_route() -> Route:
    if not MASTER_ROUTE.exists():
        raise FileNotFoundError(f"Could not find route file at: {MASTER_ROUTE}")
    print(f"[*] Reading route file: {MASTER_ROUTE.name}")
    profile = json.loads(MASTER_ROUTE.read_text(encoding="utf-8"))["profile"]
    source_d = np.asarray(profile["Distance"], dtype=float)
    
    grid = np.unique(np.r_[
        np.arange(0.0, RACE_FINISH_KM, ROUTE_STEP_KM),
        RACE_FINISH_KM,
        list(CONTROL_STOPS_KM.values()),
        DRIVER_CHANGES_KM
    ])
    
    if source_d[-1] < RACE_FINISH_KM:
        raise ValueError(f"Route ends at {source_d[-1]:.2f} km, before finish {RACE_FINISH_KM:.2f} km")
    coords = np.asarray(profile["Coordinates"], dtype=float)

    def aligned(values: object, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if len(values) == len(source_d):
            return values
        if len(values) == 0:
            raise ValueError(f"Route profile has no {name} values")
        if len(values) < len(source_d):
            return np.pad(values, (0, len(source_d) - len(values)), mode="edge")
        return values[:len(source_d)]

    gradient = aligned(profile["Gradient"], "Gradient")
    headings = aligned(profile["Headings"], "Headings")
    return Route(
        distance_km=grid,
        coords=np.column_stack((np.interp(grid, source_d, coords[:, 0]), np.interp(grid, source_d, coords[:, 1]))),
        gradient_pct=np.interp(grid, source_d, gradient),
        heading_deg=np.interp(grid, source_d, headings),
    )


def load_solar_fields() -> tuple[list[tuple[float, float, SolarIrradiance]], date]:
    print(f"[*] Loading solar data from {SOLAR_DIR.name}/...")
    missing = [name for _, _, name in ROUTE_SECTIONS if not (SOLAR_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing solar forecast files in {SOLAR_DIR}: " + ", ".join(missing))
    
    fields = []
    detected_start_date = RACE_START_DATE

    for start, end, name in ROUTE_SECTIONS:
        json_path = SOLAR_DIR / name
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        min_len = min(len(record.get("data", [])) for record in raw if "data" in record)
        for record in raw:
            record["data"] = record.get("data", [])[:min_len]
            for row in record["data"]:
                if isinstance(row.get("period_end"), str) and row["period_end"].endswith("Z"):
                    row["period_end"] = row["period_end"].replace("Z", "+00:00")

        si = SolarIrradiance(raw, time_key="period_end", interval="PT5M")
        fields.append((start, end, si))
        
        if detected_start_date == RACE_START_DATE and len(si._time_list) > 0:
            detected_start_date = datetime.fromtimestamp(si._time_list[0], tz=ACST).date()

    print(f"[*] Solar data successfully loaded. Race date synced to: {detected_start_date}")
    return fields, detected_start_date


def solar_field_covers(field: SolarIrradiance, timestamp: datetime) -> bool:
    t = timestamp.timestamp()
    return field._time_list[0] - 1.0 <= t <= field._time_list[-1] + 1.0


def field_for_distance(distance_km: float, coord: np.ndarray, timestamp: datetime,
                       fields: list[tuple[float, float, SolarIrradiance]], original_ts: datetime = None) -> tuple[SolarIrradiance, datetime]:
    """Return (field, weather_ts): the best-matching spatial field, and a timestamp
    guaranteed to fall inside that field's coverage. weather_ts preserves the exact
    time-of-day of the original query but is projected onto whichever calendar date
    the chosen field actually covers, so callers never index into the field with an
    out-of-range timestamp (which would silently clamp to the first/last entry)."""
    if original_ts is None:
        original_ts = timestamp
        
    ranked = sorted(
        fields,
        key=lambda item: 0.0 if item[0] <= distance_km <= item[1]
        else min(abs(distance_km - item[0]), abs(distance_km - item[1])),
    )
    for _, _, field in ranked:
        if solar_field_covers(field, timestamp):
            return field, timestamp

    previous_day = timestamp - timedelta(days=1)
    if (original_ts - previous_day).days > 7:
        fallback_field = ranked[0][2]
        fallback_date = datetime.fromtimestamp(fallback_field._time_list[0], tz=original_ts.tzinfo).date()
        fallback_ts = datetime.combine(fallback_date, original_ts.timetz())
        return fallback_field, fallback_ts
    return field_for_distance(distance_km, coord, previous_day, fields, original_ts)


def solar_power_w(field: SolarIrradiance, coord: np.ndarray, timestamp: datetime,
                  weather_timestamp: datetime | None = None,
                  speed_ms: float = 0.0, gradient_pct: float = 0.0, heading_deg: float = 0.0) -> float:
    # weather_timestamp (from field_for_distance) is guaranteed to fall inside `field`'s
    # coverage; `timestamp` (the true simulated time) still drives solar position below,
    # since the sun's angle depends on the real date, not on the forecast file's date.
    if weather_timestamp is None:
        weather_timestamp = timestamp
    res = field[(coord, weather_timestamp.timestamp())]
    ghi_raw, dni_raw, ambient_raw = res.data(["ghi", "dni", "air_temp"])

    ambient_arr = np.asarray(ambient_raw)
    if ambient_arr.size == 0 or ambient_arr.item() is None or np.isnan(ambient_arr.item()):
        ambient_c = T_AMBIENT_DEFAULT
    else:
        ambient_c = float(ambient_arr.item())

    ghi = float(np.asarray(ghi_raw).item())
    dni = float(np.asarray(dni_raw).item())

    t_rounded = int(round(timestamp.timestamp() / 300) * 300)
    lat_r = round(float(coord[0]), 2)
    lon_r = round(float(coord[1]), 2)
    zenith, azimuth = cached_solarpos(t_rounded, lat_r, lon_r)

    aoi = float(pvlib.irradiance.aoi(PANEL_TILT, heading_deg, zenith, azimuth))
    zenith_rad = np.radians(zenith)
    a_factor = np.cos(np.radians(aoi)) - (np.cos(zenith_rad) * sky_factor)

    plane_irradiance = max(0.0, a_factor * dni + b_constants * ghi)

    if gradient_pct != 0.0:
        pitch_rad = np.arctan(gradient_pct / 100.0)
        pitch_factor = float(np.clip(np.cos(pitch_rad + np.radians(PANEL_TILT)) / np.cos(np.radians(PANEL_TILT)), 0.75, 1.25))
        plane_irradiance *= pitch_factor

    cell_c = ambient_c + (plane_irradiance / 800.0) * (NOCT_C - 20.0) * np.exp(-K_CONVECTIVE_COOLING * speed_ms)
    thermal_factor = float(np.clip(1.0 - TEMP_COEFF_PER_C * (cell_c - 25.0), 0.70, 1.05))

    return plane_irradiance * ARRAY_AREA_M2 * ARRAY_EFFICIENCY * thermal_factor


def electrical_power(speed_ms: float, gradient_pct: float, solar_w: float) -> tuple[float, float]:
    grade = gradient_pct / 100.0
    f_roll = MASS_KG * G_MS2 * CRR * (1.0 - (grade ** 2) / 2.0)
    f_grav = MASS_KG * G_MS2 * grade
    f_drag = 0.5 * AIR_DENSITY * CDA_M2 * speed_ms**2
    
    force = f_drag + f_roll + f_grav
    mechanical_w = force * speed_ms
    
    motor_w = min(mechanical_w / MOTOR_EFF, MAX_MOTOR_POWER_W) if mechanical_w > 0 else 0.0
    drive_w = mechanical_w / MOTOR_EFF if mechanical_w >= 0 else mechanical_w * REGEN_EFF
    
    net_w = solar_w - drive_w - P_LOSS_W - P_AUX
    return net_w, motor_w


def speed_at(distance_km: float, speeds_kmh: np.ndarray) -> float:
    split_width = RACE_FINISH_KM / len(speeds_kmh)
    return float(speeds_kmh[min(int(distance_km / split_width), len(speeds_kmh) - 1)])


def simulate(speeds_kmh: Iterable[float], route: Route, fields: list[tuple[float, float, SolarIrradiance]], race_start_date: date, keep_trace: bool = True) -> SimulationResult:
    speeds = np.asarray(list(speeds_kmh), dtype=float)
    control_distances = set(CONTROL_STOPS_KM.values())
    driver_change_distances = set(DRIVER_CHANGES_KM)
    
    distance, soc = 0.0, INITIAL_SOC_PCT
    now = datetime.combine(race_start_date, DRIVE_START, tzinfo=ACST)
    max_motor, min_soc = 0.0, soc
    
    trace_d = [distance] if keep_trace else []
    trace_soc = [soc] if keep_trace else []
    trace_t = [now] if keep_trace else []
    trace_day = [0] if keep_trace else []

    def at_distance(d: float) -> tuple[np.ndarray, float, float]:
        return (np.array([np.interp(d, route.distance_km, route.coords[:, axis]) for axis in (0, 1)]),
                float(np.interp(d, route.distance_km, route.gradient_pct)),
                float(np.interp(d, route.distance_km, route.heading_deg)))

    def charge_stationary(seconds: float):
        nonlocal soc, now, min_soc
        remaining = seconds
        coord, _, heading = at_distance(distance)
        while remaining > 1e-6:
            dt = min(300.0, remaining)
            sample_time = now + timedelta(seconds=dt / 2)
            field, weather_ts = field_for_distance(distance, coord, sample_time, fields)
            solar_w = solar_power_w(field, coord, sample_time, weather_timestamp=weather_ts, speed_ms=0.0, gradient_pct=0.0, heading_deg=heading)
            soc = min(SOC_MAX_PCT, soc + max(0.0, solar_w - 10.0) * dt / (BATTERY_WH * 36.0))
            min_soc = min(min_soc, soc)
            now += timedelta(seconds=dt)
            remaining -= dt

    def process_overnight_stop():
        nonlocal now
        seal_time = datetime.combine(now.date(), CHARGE_END, tzinfo=ACST)
        if now < seal_time:
            charge_stationary((seal_time - now).total_seconds())
        
        next_date = now.date() + timedelta(days=1)
        now = datetime.combine(next_date, CHARGE_START, tzinfo=ACST)
        drive_start = datetime.combine(next_date, DRIVE_START, tzinfo=ACST)
        charge_stationary((drive_start - now).total_seconds())
        now = drive_start

    def serve_delay(seconds: float):
        nonlocal now
        remaining = seconds
        while remaining > 1e-6:
            day_end = datetime.combine(now.date(), DRIVE_END, tzinfo=ACST)
            # Match the main loop's 1ms boundary tolerance (line ~347 below). Without
            # this, a 'now' that's a hair under day_end due to accumulated float error
            # produces a sub-microsecond chunk here, which charge_stationary's own
            # 1e-6s threshold then silently no-ops -- remaining barely shrinks and
            # the loop effectively never terminates.
            if now >= day_end - timedelta(milliseconds=1):
                now = day_end
                process_overnight_stop()
                day_end = datetime.combine(now.date(), DRIVE_END, tzinfo=ACST)
            
            chunk = min(remaining, (day_end - now).total_seconds())
            if chunk > 0:
                charge_stationary(chunk)
                remaining -= chunk

    while distance < RACE_FINISH_KM - 1e-7:
        race_day = (now.date() - race_start_date).days
        if race_day >= MAX_RACE_DAYS:
            if not keep_trace:
                trace_d = [distance]
            return SimulationResult(np.array(trace_d), np.array(trace_soc), trace_t, np.array(trace_day), None, min_soc, max_motor, speeds)
        
        day_start = datetime.combine(now.date(), DRIVE_START, tzinfo=ACST)
        day_end = datetime.combine(now.date(), DRIVE_END, tzinfo=ACST)
        
        if now < day_start:
            charge_stationary((day_start - now).total_seconds())
            continue
        if now >= day_end - timedelta(milliseconds=1):
            now = day_end
            process_overnight_stop()
            continue

        speed_kmh = speed_at(distance, speeds)
        speed_ms = speed_kmh / 3.6
        next_grid = route.distance_km[np.searchsorted(route.distance_km, distance + 1e-7, side="right")]
        remaining_window = (day_end - now).total_seconds()
        travel_seconds = max(0.0, min((next_grid - distance) * 1000.0 / speed_ms, remaining_window))
        
        if travel_seconds <= 1e-5:
            if remaining_window <= 1e-3:
                now = day_end
                process_overnight_stop()
            else:
                raise RuntimeError("Simulation step stalled")
            continue
            
        end_distance = distance + speed_ms * travel_seconds / 1000.0
        mid_distance = (distance + end_distance) / 2.0
        coord, grade, heading = at_distance(mid_distance)
        sample_time = now + timedelta(seconds=travel_seconds / 2)
        field, weather_ts = field_for_distance(mid_distance, coord, sample_time, fields)
        
        solar_w = solar_power_w(field, coord, sample_time, weather_timestamp=weather_ts, speed_ms=speed_ms, gradient_pct=grade, heading_deg=heading)
        net_w, motor_w = electrical_power(speed_ms, grade, solar_w)
        
        soc += net_w * travel_seconds / (BATTERY_WH * 36.0)
        min_soc, max_motor = min(min_soc, soc), max(max_motor, motor_w)
        now += timedelta(seconds=travel_seconds)
        distance = end_distance

        if soc < SOC_MIN_PCT:
            soc = max(0.0, soc)
            if not keep_trace:
                trace_d = [distance]
            else:
                trace_d.append(distance); trace_soc.append(soc); trace_t.append(now); trace_day.append(race_day)
            return SimulationResult(np.array(trace_d), np.array(trace_soc), trace_t, np.array(trace_day), None, min_soc, max_motor, speeds)

        if keep_trace:
            trace_d.append(distance); trace_soc.append(soc); trace_t.append(now); trace_day.append(race_day)
            
        if any(abs(distance - stop) < 1e-5 for stop in control_distances):
            serve_delay(CONTROL_STOP_SECONDS)
            if keep_trace:
                trace_d.append(distance); trace_soc.append(soc); trace_t.append(now); trace_day.append(race_day)
                
        if any(abs(distance - swap) < 1e-5 for swap in driver_change_distances):
            serve_delay(DRIVER_CHANGE_SECONDS)
            if keep_trace:
                trace_d.append(distance); trace_soc.append(soc); trace_t.append(now); trace_day.append(race_day)

    return SimulationResult(np.array(trace_d), np.array(trace_soc), trace_t, np.array(trace_day), now, min_soc, max_motor, speeds)


def objective(x: np.ndarray, route: Route, fields: list[tuple[float, float, SolarIrradiance]], race_start_date: date) -> float:
    """Module-level (not nested) so it can be pickled and sent to worker processes.
    On Windows, multiprocessing uses 'spawn' rather than 'fork', which requires the
    target callable to be importable by name -- a function nested inside optimise()
    has no such name and crashes with AttributeError when workers != 1."""
    try:
        result = simulate(x, route, fields, race_start_date, keep_trace=False)
    except SolarCoverageError:
        return 1e9

    # If the car dies or times out, use the "Sloped Penalty" to guide the optimizer
    if result.finish_time is None:
        distance_remaining = max(0.0, RACE_FINISH_KM - result.distance_km[-1])
        soc_breach = max(0.0, SOC_MIN_PCT - result.min_soc_pct)

        # Downhill slope: closer to Adelaide = smaller penalty
        return 1e8 + (distance_remaining * 10000.0) + (soc_breach * 1e6)

    # If it survives, optimize for the fastest time + penalize 5kW motor spikes
    seconds = (result.finish_time - datetime.combine(race_start_date, DRIVE_START, tzinfo=ACST)).total_seconds()
    power_penalty = max(0.0, result.max_motor_power_w - MAX_MOTOR_POWER_W) * 1e4
    return seconds + power_penalty


def optimise(route: Route, fields: list[tuple[float, float, SolarIrradiance]], race_start_date: date) -> SimulationResult:
    print(f"\n[+] Running Global {N_SPEED_SPLITS}-Segment DE Optimization (Initial Guess: {x0_guess} km/h)...")
    pbar = tqdm(desc="Generations", unit="gen")

    # Bind route/fields/race_start_date here rather than via closure -- functools.partial
    # of a module-level function pickles fine (it just re-imports `objective` by name and
    # reattaches these bound args), unlike a nested def.
    bound_objective = functools.partial(objective, route=route, fields=fields, race_start_date=race_start_date)

    def callback(xk, convergence=None):
        pbar.update(1)
        mean_v = float(np.mean(xk))
        pbar.set_postfix({"Mean v": f"{mean_v:.1f} km/h", "Conv": f"{convergence:.3f}" if convergence else "running"})

    # Injecting the 75 km/h initial guess
    x0_guess = np.full(N_SPEED_SPLITS, 65.0)

    # Differential Evolution handles the cliffs natively
    result = differential_evolution(
        bound_objective,
        bounds=[(SPEED_MIN_KMH, SPEED_MAX_KMH)] * N_SPEED_SPLITS,
        x0=x0_guess,           
        maxiter=40,            
        popsize=25,            
        mutation=(0.5, 1.0),
        recombination=0.7,
        updating='deferred',   
        workers=-1,            
        callback=callback
    )
    pbar.close()
    
    print(f"\n[+] Optimization Finished! Target Objective: {result.fun:.0f}s")
    return simulate(result.x, route, fields, race_start_date, keep_trace=True)


def plot_result(result: SimulationResult) -> Path:
    output = HERE / "wsc_2027_optimised_soc.png"
    figure, axis = plt.subplots(figsize=(15, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, MAX_RACE_DAYS))
    for day in range(MAX_RACE_DAYS):
        mask = result.day_index == day
        if np.any(mask):
            axis.plot(result.distance_km[mask], result.soc_pct[mask], color=colors[day], linewidth=2.0, label=f"Race day {day + 1}")
    for name, stop in CONTROL_STOPS_KM.items():
        axis.axvline(stop, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        axis.text(stop, 2, name, rotation=90, va="bottom", ha="right", fontsize=7)
    axis.axvline(RACE_FINISH_KM, color="green", linestyle="--", linewidth=1.2, label="Finish")
    axis.axhline(SOC_MIN_PCT, color="red", linestyle="--", linewidth=1.0, label=f"Minimum SOC ({SOC_MIN_PCT:.0f}%)")
    axis.set(xlim=(0, RACE_FINISH_KM), ylim=(0, 100), xlabel="Cumulative distance (km)", ylabel="State of charge (%)",
             title="2027 BWSC Time-Minimised SOC Profile")
    axis.grid(True, linestyle=":", alpha=0.5)
    axis.legend(loc="lower left", ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    return output


def main() -> None:
    print("=" * 60)
    print("   2027 Bridgestone World Solar Challenge Optimizer")
    print("=" * 60)
    route = load_route()
    fields, race_start_date = load_solar_fields()
    
    result = optimise(route, fields, race_start_date)
    if result.finish_time is None:
        print("[!] No feasible finish achieved within allowable days.")
        return
        
    print(f"\nOfficial Finish: {result.finish_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Minimum Battery SoC: {result.min_soc_pct:.2f}% | Peak Motor Power: {result.max_motor_power_w:.0f} W")
    print(f"Optimal Cruise Speeds (km/h): {np.round(result.speed_kmh, 1).tolist()}")
    print(f"Trajectory plot generated at: {plot_result(result)}")


if __name__ == "__main__":
    main()
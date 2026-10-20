"""Fixed-speed 2027 BWSC simulation and detailed energy sheet.

This is adapted from optimizer.py.  It keeps the source model's route, solar,
vehicle, battery, timing, control-stop, driver-change, and overnight-charge
logic, but removes all speed optimization.

Speed arguments are constant within these five distance bands:
    0-600 km, 600-1200 km, 1200-1800 km, 1800-2400 km, 2400-3022 km.

Required alongside this file:
    - WSC_WSC Route_WSC Route.kml.save
    - Solar/Day1_WSC_Day 1_solar.json ... Day5_WSC_Day 5_solar.json
    - solar_table.py and the same Python dependencies as the original script
"""
from __future__ import annotations

import csv
import functools
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib

from solar_table import SolarIrradiance

# ============================ CONFIGURATION ================================
ACST = ZoneInfo("Australia/Darwin")
RACE_START_DATE = date(2027, 8, 22)
MAX_RACE_DAYS = 6
DRIVE_START = time(8, 0)
DRIVE_END = time(17, 0)
CHARGE_START = time(6, 15)
CHARGE_END = time(19, 15)

SPEED_MIN_KMH = 60.0
SPEED_MAX_KMH = 110.0
RACE_FINISH_KM = 3022.0
ROUTE_STEP_KM = 10.0

# Five fixed-speed bands. The final band is 622 km because the finish is 3022 km.
SPEED_BAND_KM = 600.0
N_SPEED_SPLITS = 5

CONTROL_STOPS_KM = {
    "Katherine": 318.0,
    "Dunmarra": 635.0,
    "Tennant Creek": 988.0,
    "Barrow Creek": 1210.0,
    "Alice Springs": 1493.0,
    "Erldunda": 1693.0,
    "Coober Pedy": 2179.0,
    "Glendambo": 2432.0,
    "Port Augusta": 2719.0,
}
CONTROL_STOP_SECONDS = 30 * 60
DRIVER_CHANGES_KM = [160.0, 475.0, 810.0, 1100.0, 1350.0, 1850.0, 2020.0, 2300.0, 2580.0, 2870.0]
DRIVER_CHANGE_SECONDS = 5 * 60

# Vehicle & array constants copied from optimizer.py.
MASS_KG, G_MS2, CRR, CDA_M2, AIR_DENSITY = 320.0, 9.81, 0.007, 0.16, 1.2
ARRAY_AREA_M2, ARRAY_EFFICIENCY, PANEL_TILT, ALBEDO = 5.78, 0.21, 4.0, 0.2
MOTOR_EFF, REGEN_EFF, P_LOSS_W, P_AUX = 0.95, 0.70, 70.0, 5.0
MAX_MOTOR_POWER_W = 5000.0
BATTERY_WH = 11_000_000.0 / 3600.0
BATTERY_J = BATTERY_WH * 3600.0
TEMP_COEFF_PER_C, NOCT_C, K_CONVECTIVE_COOLING = 0.004, 45.0, 0.03
T_AMBIENT_DEFAULT = 28.0

sky_factor = (1.0 + np.cos(np.radians(PANEL_TILT))) / 2.0
ground_factor = (1.0 - np.cos(np.radians(PANEL_TILT))) / 2.0
b_constants = sky_factor + ALBEDO * ground_factor

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

ENERGY_LABELS = {
    "solar_drive_j": "Solar-array electrical energy generated while driving",
    "solar_stationary_j": "Solar-array electrical energy generated while stationary",
    "wheel_rolling_j": "Wheel work against rolling resistance",
    "wheel_aero_drag_j": "Wheel work against aerodynamic drag",
    "wheel_gravity_signed_j": "Wheel work against gravity (positive uphill, negative downhill)",
    "wheel_mechanical_signed_j": "Net signed mechanical work at the wheels",
    "traction_electrical_j": "Electrical energy drawn for positive propulsion",
    "regen_electrical_j": "Electrical energy recovered by regenerative braking",
    "drivetrain_loss_j": "Motor/drivetrain conversion loss during propulsion",
    "regen_loss_j": "Mechanical energy not recovered during regenerative braking",
    "fixed_loss_j": "Fixed 70 W loss term while driving",
    "aux_energy_j": "5 W auxiliary load while driving",
    "stationary_offset_j": "Stationary 10 W charging offset in the source model",
    "stationary_charge_offered_j": "Stationary solar energy available after the 10 W offset",
    "stationary_charge_stored_j": "Energy actually added to the battery during stationary charging",
    "stationary_charge_curtailed_j": "Stationary charging energy not stored due to the SOC ceiling",
    "stationary_soc_cap_adjustment_j": "Negative SOC-ceiling adjustment made by the source model, if any",
    "driving_net_battery_j": "Net battery energy change from driving equations",
    "stationary_soc_delta_j": "Net battery energy change during stationary charging, including any clamp adjustment",
}

SEGMENT_FLOW_KEYS = (
    "solar_drive_j",
    "solar_stationary_j",
    "wheel_rolling_j",
    "wheel_aero_drag_j",
    "wheel_gravity_signed_j",
    "wheel_mechanical_signed_j",
    "traction_electrical_j",
    "regen_electrical_j",
    "drivetrain_loss_j",
    "regen_loss_j",
    "fixed_loss_j",
    "aux_energy_j",
    "stationary_offset_j",
    "stationary_charge_offered_j",
    "stationary_charge_stored_j",
    "stationary_charge_curtailed_j",
    "stationary_soc_cap_adjustment_j",
    "driving_net_battery_j",
    "stationary_soc_delta_j",
)


class SolarCoverageError(RuntimeError):
    """Kept for compatibility with the source simulator's error handling."""


@dataclass
class Route:
    distance_km: np.ndarray
    coords: np.ndarray
    gradient_pct: np.ndarray
    heading_deg: np.ndarray


@dataclass
class EnergyLedger:
    totals: dict[str, float] = field(default_factory=dict)
    by_segment: list[dict[str, float]] = field(
        default_factory=lambda: [dict() for _ in range(N_SPEED_SPLITS)]
    )
    stationary_modes: dict[str, dict[str, float]] = field(default_factory=dict)
    driving_seconds: float = 0.0

    def add(self, key: str, joules: float, distance_km: float) -> None:
        value = float(joules)
        self.totals[key] = self.totals.get(key, 0.0) + value
        segment = segment_index(distance_km)
        bucket = self.by_segment[segment]
        bucket[key] = bucket.get(key, 0.0) + value

    def add_stationary_mode(
        self,
        mode: str,
        *,
        seconds: float,
        solar_j: float,
        offset_j: float,
        offered_j: float,
        stored_j: float,
        curtailed_j: float,
        soc_delta_j: float,
        distance_km: float,
    ) -> None:
        info = self.stationary_modes.setdefault(
            mode,
            {
                "seconds": 0.0,
                "solar_j": 0.0,
                "offset_j": 0.0,
                "offered_j": 0.0,
                "stored_j": 0.0,
                "curtailed_j": 0.0,
                "soc_delta_j": 0.0,
            },
        )
        info["seconds"] += seconds
        info["solar_j"] += solar_j
        info["offset_j"] += offset_j
        info["offered_j"] += offered_j
        info["stored_j"] += stored_j
        info["curtailed_j"] += curtailed_j
        info["soc_delta_j"] += soc_delta_j
        self.add("solar_stationary_j", solar_j, distance_km)
        self.add("stationary_offset_j", offset_j, distance_km)
        self.add("stationary_charge_offered_j", offered_j, distance_km)
        self.add("stationary_charge_stored_j", stored_j, distance_km)
        self.add("stationary_charge_curtailed_j", curtailed_j, distance_km)
        self.add("stationary_soc_delta_j", soc_delta_j, distance_km)
        self.add("stationary_soc_cap_adjustment_j", min(0.0, soc_delta_j), distance_km)


@dataclass
class SimulationResult:
    speeds_kmh: np.ndarray
    start_time: datetime
    end_time: datetime
    finish_time: datetime | None
    finish_status: str
    distance_km: float
    initial_soc_pct: float
    final_soc_pct: float
    min_soc_pct: float
    max_motor_power_w: float
    ledger: EnergyLedger


def segment_index(distance_km: float) -> int:
    """Return the 0-based 600 km speed/energy band containing this distance."""
    if distance_km >= RACE_FINISH_KM:
        return N_SPEED_SPLITS - 1
    return max(0, min(int(max(distance_km, 0.0) // SPEED_BAND_KM), N_SPEED_SPLITS - 1))


def segment_bounds(index: int) -> tuple[float, float]:
    start = index * SPEED_BAND_KM
    end = min((index + 1) * SPEED_BAND_KM, RACE_FINISH_KM)
    return start, end


@functools.lru_cache(maxsize=65536)
def cached_solarpos(timestamp_rounded: int, lat_rounded: float, lon_rounded: float) -> tuple[float, float]:
    dt = datetime.fromtimestamp(timestamp_rounded, tz=ACST)
    solpos = pvlib.solarposition.ephemeris(pd.DatetimeIndex([dt]), lat_rounded, lon_rounded)
    return float(solpos["apparent_zenith"].iloc[0]), float(solpos["azimuth"].iloc[0])


def load_route() -> Route:
    if not MASTER_ROUTE.exists():
        raise FileNotFoundError(f"Could not find route file at: {MASTER_ROUTE}")
    print(f"[*] Reading route file: {MASTER_ROUTE.name}")
    profile = json.loads(MASTER_ROUTE.read_text(encoding="utf-8"))["profile"]
    source_d = np.asarray(profile["Distance"], dtype=float)

    # Include all speed-band boundaries as well as existing route and stop samples.
    band_boundaries = [600.0, 1200.0, 1800.0, 2400.0]
    grid = np.unique(np.r_[
        np.arange(0.0, RACE_FINISH_KM, ROUTE_STEP_KM),
        RACE_FINISH_KM,
        band_boundaries,
        list(CONTROL_STOPS_KM.values()),
        DRIVER_CHANGES_KM,
    ])

    if source_d[-1] < RACE_FINISH_KM:
        raise ValueError(
            f"Route ends at {source_d[-1]:.2f} km, before finish {RACE_FINISH_KM:.2f} km"
        )
    coords = np.asarray(profile["Coordinates"], dtype=float)

    def aligned(values: object, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if len(arr) == len(source_d):
            return arr
        if len(arr) == 0:
            raise ValueError(f"Route profile has no {name} values")
        if len(arr) < len(source_d):
            return np.pad(arr, (0, len(source_d) - len(arr)), mode="edge")
        return arr[: len(source_d)]

    gradient = aligned(profile["Gradient"], "Gradient")
    headings = aligned(profile["Headings"], "Headings")
    return Route(
        distance_km=grid,
        coords=np.column_stack((
            np.interp(grid, source_d, coords[:, 0]),
            np.interp(grid, source_d, coords[:, 1]),
        )),
        gradient_pct=np.interp(grid, source_d, gradient),
        heading_deg=np.interp(grid, source_d, headings),
    )


def load_solar_fields() -> tuple[list[tuple[float, float, SolarIrradiance]], date]:
    print(f"[*] Loading solar data from {SOLAR_DIR.name}/...")
    missing = [name for _, _, name in ROUTE_SECTIONS if not (SOLAR_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing solar forecast files in {SOLAR_DIR}: " + ", ".join(missing)
        )

    fields: list[tuple[float, float, SolarIrradiance]] = []
    detected_start_date = RACE_START_DATE
    for start, end, name in ROUTE_SECTIONS:
        with open(SOLAR_DIR / name, "r", encoding="utf-8") as f:
            raw = json.load(f)

        min_len = min(len(record.get("data", [])) for record in raw if "data" in record)
        for record in raw:
            record["data"] = record.get("data", [])[:min_len]
            for row in record["data"]:
                if isinstance(row.get("period_end"), str) and row["period_end"].endswith("Z"):
                    row["period_end"] = row["period_end"].replace("Z", "+00:00")

        irradiance = SolarIrradiance(raw, time_key="period_end", interval="PT5M")
        fields.append((start, end, irradiance))
        if detected_start_date == RACE_START_DATE and len(irradiance._time_list) > 0:
            detected_start_date = datetime.fromtimestamp(
                irradiance._time_list[0], tz=ACST
            ).date()

    print(f"[*] Solar data loaded. Race date synced to: {detected_start_date}")
    return fields, detected_start_date


def solar_field_covers(field: SolarIrradiance, timestamp: datetime) -> bool:
    t = timestamp.timestamp()
    return field._time_list[0] - 1.0 <= t <= field._time_list[-1] + 1.0


def field_for_distance(
    distance_km: float,
    coord: np.ndarray,
    timestamp: datetime,
    fields: list[tuple[float, float, SolarIrradiance]],
    original_ts: datetime | None = None,
) -> tuple[SolarIrradiance, datetime]:
    """Use the nearest spatial solar field and keep its lookup timestamp in range."""
    if original_ts is None:
        original_ts = timestamp
    ranked = sorted(
        fields,
        key=lambda item: 0.0
        if item[0] <= distance_km <= item[1]
        else min(abs(distance_km - item[0]), abs(distance_km - item[1])),
    )
    for _, _, solar_field in ranked:
        if solar_field_covers(solar_field, timestamp):
            return solar_field, timestamp

    previous_day = timestamp - timedelta(days=1)
    if (original_ts - previous_day).days > 7:
        fallback_field = ranked[0][2]
        fallback_date = datetime.fromtimestamp(
            fallback_field._time_list[0], tz=original_ts.tzinfo
        ).date()
        fallback_ts = datetime.combine(fallback_date, original_ts.timetz())
        return fallback_field, fallback_ts
    return field_for_distance(distance_km, coord, previous_day, fields, original_ts)


def solar_power_w(
    solar_field: SolarIrradiance,
    coord: np.ndarray,
    timestamp: datetime,
    weather_timestamp: datetime | None = None,
    speed_ms: float = 0.0,
    gradient_pct: float = 0.0,
    heading_deg: float = 0.0,
) -> float:
    # The forecast timestamp selects weather data; the true timestamp sets sun position.
    if weather_timestamp is None:
        weather_timestamp = timestamp
    res = solar_field[(coord, weather_timestamp.timestamp())]
    ghi_raw, dni_raw, ambient_raw = res.data(["ghi", "dni", "air_temp"])

    ambient_arr = np.asarray(ambient_raw)
    if ambient_arr.size == 0 or ambient_arr.item() is None or np.isnan(ambient_arr.item()):
        ambient_c = T_AMBIENT_DEFAULT
    else:
        ambient_c = float(ambient_arr.item())

    ghi = float(np.asarray(ghi_raw).item())
    dni = float(np.asarray(dni_raw).item())
    rounded_timestamp = int(round(timestamp.timestamp() / 300) * 300)
    lat_r = round(float(coord[0]), 2)
    lon_r = round(float(coord[1]), 2)
    zenith, azimuth = cached_solarpos(rounded_timestamp, lat_r, lon_r)

    aoi = float(pvlib.irradiance.aoi(PANEL_TILT, heading_deg, zenith, azimuth))
    zenith_rad = np.radians(zenith)
    a_factor = np.cos(np.radians(aoi)) - (np.cos(zenith_rad) * sky_factor)
    plane_irradiance = max(0.0, a_factor * dni + b_constants * ghi)

    if gradient_pct != 0.0:
        pitch_rad = np.arctan(gradient_pct / 100.0)
        pitch_factor = float(np.clip(
            np.cos(pitch_rad + np.radians(PANEL_TILT))
            / np.cos(np.radians(PANEL_TILT)),
            0.75,
            1.25,
        ))
        plane_irradiance *= pitch_factor

    cell_c = ambient_c + (plane_irradiance / 800.0) * (NOCT_C - 20.0) * np.exp(
        -K_CONVECTIVE_COOLING * speed_ms
    )
    thermal_factor = float(np.clip(
        1.0 - TEMP_COEFF_PER_C * (cell_c - 25.0), 0.70, 1.05
    ))
    return plane_irradiance * ARRAY_AREA_M2 * ARRAY_EFFICIENCY * thermal_factor


def electrical_power_components(
    speed_ms: float, gradient_pct: float, solar_w: float
) -> dict[str, float]:
    """Use the same power equations as optimizer.py and expose terms for accounting."""
    grade = gradient_pct / 100.0
    f_roll = MASS_KG * G_MS2 * CRR * (1.0 - (grade ** 2) / 2.0)
    f_grav = MASS_KG * G_MS2 * grade
    f_drag = 0.5 * AIR_DENSITY * CDA_M2 * speed_ms**2
    force = f_drag + f_roll + f_grav
    mechanical_w = force * speed_ms

    motor_w = min(mechanical_w / MOTOR_EFF, MAX_MOTOR_POWER_W) if mechanical_w > 0 else 0.0
    drive_w = mechanical_w / MOTOR_EFF if mechanical_w >= 0 else mechanical_w * REGEN_EFF
    net_w = solar_w - drive_w - P_LOSS_W - P_AUX
    return {
        "f_roll": f_roll,
        "f_grav": f_grav,
        "f_drag": f_drag,
        "force": force,
        "mechanical_w": mechanical_w,
        "motor_w": motor_w,
        "drive_w": drive_w,
        "net_w": net_w,
    }


def speed_at(distance_km: float, speeds_kmh: np.ndarray) -> float:
    # Unlike the original optimizer's equal fifths, these are exactly 600 km bands;
    # the last speed continues for the remaining 622 km to the 3022 km finish.
    return float(speeds_kmh[segment_index(distance_km)])


def simulate_fixed_speeds(
    speeds_kmh: Iterable[float],
    route: Route,
    fields: list[tuple[float, float, SolarIrradiance]],
    race_start_date: date,
) -> SimulationResult:
    speeds = np.asarray(list(speeds_kmh), dtype=float)
    if len(speeds) != N_SPEED_SPLITS:
        raise ValueError(f"Exactly {N_SPEED_SPLITS} speeds are required; received {len(speeds)}.")
    if not np.all(np.isfinite(speeds)):
        raise ValueError("All five speeds must be finite numbers.")
    if np.any(speeds < SPEED_MIN_KMH) or np.any(speeds > SPEED_MAX_KMH):
        raise ValueError(
            f"Speeds must each be between {SPEED_MIN_KMH:g} and {SPEED_MAX_KMH:g} km/h."
        )

    control_distances = set(CONTROL_STOPS_KM.values())
    driver_change_distances = set(DRIVER_CHANGES_KM)
    distance, soc = 0.0, 95.0
    initial_soc = soc
    start_time = datetime.combine(race_start_date, DRIVE_START, tzinfo=ACST)
    now = start_time
    max_motor, min_soc = 0.0, soc
    ledger = EnergyLedger()
    finish_status = "finished"

    def at_distance(d: float) -> tuple[np.ndarray, float, float]:
        coord = np.array([
            np.interp(d, route.distance_km, route.coords[:, axis]) for axis in (0, 1)
        ])
        grade = float(np.interp(d, route.distance_km, route.gradient_pct))
        heading = float(np.interp(d, route.distance_km, route.heading_deg))
        return coord, grade, heading

    def charge_stationary(seconds: float, mode: str) -> None:
        nonlocal soc, now, min_soc
        remaining = seconds
        coord, _, heading = at_distance(distance)
        while remaining > 1e-6:
            dt = min(300.0, remaining)
            sample_time = now + timedelta(seconds=dt / 2)
            solar_field, weather_ts = field_for_distance(
                distance, coord, sample_time, fields
            )
            solar_w = solar_power_w(
                solar_field,
                coord,
                sample_time,
                weather_timestamp=weather_ts,
                speed_ms=0.0,
                gradient_pct=0.0,
                heading_deg=heading,
            )

            # Preserve the source formula: max(0, solar_w - 10 W), then clamp SOC to 95%.
            offered_w = max(0.0, solar_w - 10.0)
            old_soc = soc
            soc = min(95.0, soc + offered_w * dt / (BATTERY_WH * 36.0))
            soc_delta_j = (soc - old_soc) * BATTERY_J / 100.0
            stored_j = max(0.0, soc_delta_j)
            solar_j = solar_w * dt
            offset_j = min(solar_w, 10.0) * dt
            offered_j = offered_w * dt
            curtailed_j = max(0.0, offered_j - stored_j)

            ledger.add_stationary_mode(
                mode,
                seconds=dt,
                solar_j=solar_j,
                offset_j=offset_j,
                offered_j=offered_j,
                stored_j=stored_j,
                curtailed_j=curtailed_j,
                soc_delta_j=soc_delta_j,
                distance_km=distance,
            )
            min_soc = min(min_soc, soc)
            now += timedelta(seconds=dt)
            remaining -= dt

    def process_overnight_stop() -> None:
        nonlocal now
        seal_time = datetime.combine(now.date(), CHARGE_END, tzinfo=ACST)
        if now < seal_time:
            charge_stationary(
                (seal_time - now).total_seconds(),
                "Evening stationary charging (until 19:15)",
            )
        next_date = now.date() + timedelta(days=1)
        now = datetime.combine(next_date, CHARGE_START, tzinfo=ACST)
        drive_start = datetime.combine(next_date, DRIVE_START, tzinfo=ACST)
        charge_stationary(
            (drive_start - now).total_seconds(),
            "Morning stationary charging (06:15-08:00)",
        )
        now = drive_start

    def serve_delay(seconds: float, mode: str) -> None:
        nonlocal now
        remaining = seconds
        while remaining > 1e-6:
            day_end = datetime.combine(now.date(), DRIVE_END, tzinfo=ACST)
            # Match the 1 ms boundary tolerance in the source simulator.
            if now >= day_end - timedelta(milliseconds=1):
                now = day_end
                process_overnight_stop()
                day_end = datetime.combine(now.date(), DRIVE_END, tzinfo=ACST)
            chunk = min(remaining, (day_end - now).total_seconds())
            if chunk > 0:
                charge_stationary(chunk, mode)
                remaining -= chunk

    while distance < RACE_FINISH_KM - 1e-7:
        race_day = (now.date() - race_start_date).days
        if race_day >= MAX_RACE_DAYS:
            finish_status = "six-day time limit reached"
            return SimulationResult(
                speeds, start_time, now, None, finish_status, distance,
                initial_soc, soc, min_soc, max_motor, ledger,
            )

        day_start = datetime.combine(now.date(), DRIVE_START, tzinfo=ACST)
        day_end = datetime.combine(now.date(), DRIVE_END, tzinfo=ACST)
        if now < day_start:
            charge_stationary(
                (day_start - now).total_seconds(),
                "Morning stationary charging (06:15-08:00)",
            )
            continue
        if now >= day_end - timedelta(milliseconds=1):
            now = day_end
            process_overnight_stop()
            continue

        speed_kmh = speed_at(distance, speeds)
        speed_ms = speed_kmh / 3.6
        next_index = np.searchsorted(route.distance_km, distance + 1e-7, side="right")
        if next_index >= len(route.distance_km):
            raise RuntimeError("Route grid ended before the finish distance.")
        next_grid = route.distance_km[next_index]
        remaining_window = (day_end - now).total_seconds()
        travel_seconds = max(
            0.0,
            min((next_grid - distance) * 1000.0 / speed_ms, remaining_window),
        )
        if travel_seconds <= 1e-5:
            if remaining_window <= 1e-3:
                now = day_end
                process_overnight_stop()
            else:
                raise RuntimeError("Simulation step stalled.")
            continue

        end_distance = distance + speed_ms * travel_seconds / 1000.0
        mid_distance = (distance + end_distance) / 2.0
        coord, grade, heading = at_distance(mid_distance)
        sample_time = now + timedelta(seconds=travel_seconds / 2)
        solar_field, weather_ts = field_for_distance(
            mid_distance, coord, sample_time, fields
        )
        solar_w = solar_power_w(
            solar_field,
            coord,
            sample_time,
            weather_timestamp=weather_ts,
            speed_ms=speed_ms,
            gradient_pct=grade,
            heading_deg=heading,
        )
        components = electrical_power_components(speed_ms, grade, solar_w)
        net_w = components["net_w"]
        motor_w = components["motor_w"]
        mechanical_w = components["mechanical_w"]
        drive_w = components["drive_w"]
        dt = travel_seconds

        # Record gross solar-array electricity generated while moving.
        ledger.add("solar_drive_j", solar_w * dt, mid_distance)
        # Breakdown of wheel-level mechanical work; these are components of the same
        # mechanical demand, not additional battery-side energy flows.
        ledger.add("wheel_rolling_j", components["f_roll"] * speed_ms * dt, mid_distance)
        ledger.add("wheel_aero_drag_j", components["f_drag"] * speed_ms * dt, mid_distance)
        ledger.add("wheel_gravity_signed_j", components["f_grav"] * speed_ms * dt, mid_distance)
        ledger.add("wheel_mechanical_signed_j", mechanical_w * dt, mid_distance)
        if mechanical_w >= 0.0:
            traction_j = drive_w * dt
            drivetrain_loss_j = max(0.0, drive_w - mechanical_w) * dt
            ledger.add("traction_electrical_j", traction_j, mid_distance)
            ledger.add("drivetrain_loss_j", drivetrain_loss_j, mid_distance)
        else:
            recovered_j = -drive_w * dt
            available_regen_j = -mechanical_w * dt
            ledger.add("regen_electrical_j", recovered_j, mid_distance)
            ledger.add("regen_loss_j", max(0.0, available_regen_j - recovered_j), mid_distance)

        ledger.add("fixed_loss_j", P_LOSS_W * dt, mid_distance)
        ledger.add("aux_energy_j", P_AUX * dt, mid_distance)
        ledger.add("driving_net_battery_j", net_w * dt, mid_distance)
        ledger.driving_seconds += dt

        soc += net_w * dt / (BATTERY_WH * 36.0)
        min_soc = min(min_soc, soc)
        max_motor = max(max_motor, motor_w)
        now += timedelta(seconds=dt)
        distance = end_distance

        if soc < 20.0:
            # Keep the source simulator's low-SOC termination behavior.
            soc = max(0.0, soc)
            finish_status = "battery SOC fell below the 20% minimum"
            return SimulationResult(
                speeds, start_time, now, None, finish_status, distance,
                initial_soc, soc, min_soc, max_motor, ledger,
            )

        control_name = next(
            (name for name, stop in CONTROL_STOPS_KM.items() if abs(distance - stop) < 1e-5),
            None,
        )
        if distance in control_distances or control_name is not None:
            stop_label = control_name or f"at {distance:.0f} km"
            serve_delay(CONTROL_STOP_SECONDS, f"Control stop: {stop_label}")

        if any(abs(distance - swap) < 1e-5 for swap in driver_change_distances):
            serve_delay(DRIVER_CHANGE_SECONDS, f"Driver change near {distance:.0f} km")

    return SimulationResult(
        speeds, start_time, now, now, "finished", distance,
        initial_soc, soc, min_soc, max_motor, ledger,
    )


def joules_to_kwh(joules: float) -> float:
    return joules / 3_600_000.0


def format_energy(joules: float) -> str:
    return f"{joules / 1_000_000.0:,.3f} MJ  |  {joules_to_kwh(joules):,.4f} kWh"


def net_battery_energy_from_flows(ledger: EnergyLedger) -> float:
    """Battery-side energy sum, using source-model terms without double-counting wheel work."""
    t = ledger.totals
    return (
        t.get("solar_drive_j", 0.0)
        + t.get("regen_electrical_j", 0.0)
        - t.get("traction_electrical_j", 0.0)
        - t.get("fixed_loss_j", 0.0)
        - t.get("aux_energy_j", 0.0)
        + t.get("stationary_soc_delta_j", 0.0)
    )


def render_energy_sheet(result: SimulationResult) -> str:
    ledger = result.ledger
    totals = ledger.totals
    actual_delta_j = (result.final_soc_pct - result.initial_soc_pct) * BATTERY_J / 100.0
    flow_delta_j = net_battery_energy_from_flows(ledger)
    residual_j = actual_delta_j - flow_delta_j
    start_energy_j = result.initial_soc_pct / 100.0 * BATTERY_J
    final_energy_j = result.final_soc_pct / 100.0 * BATTERY_J
    elapsed = result.end_time - result.start_time

    lines: list[str] = []
    lines.append("=" * 84)
    lines.append("2027 BRIDGESTONE WORLD SOLAR CHALLENGE — FIXED-SPEED ENERGY SHEET")
    lines.append("=" * 84)
    lines.append("Speed bands (constant within each band):")
    for i, speed in enumerate(result.speeds_kmh):
        lo, hi = segment_bounds(i)
        end_bracket = "]" if i == N_SPEED_SPLITS - 1 else ")"
        lines.append(f"  Segment {i + 1}: {lo:,.0f}-{hi:,.0f} km  |  {speed:.2f} km/h")
    lines.append("")
    lines.append(f"Outcome: {result.finish_status}")
    if result.finish_time is not None:
        lines.append(f"Finish timestamp: {result.finish_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        lines.append(f"Simulation stopped at: {result.end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"Distance reached: {result.distance_km:,.2f} / {RACE_FINISH_KM:,.2f} km")
    lines.append(f"Elapsed wall-clock time: {elapsed}")
    lines.append(f"Time spent driving: {timedelta(seconds=ledger.driving_seconds)}")
    lines.append(f"Minimum SOC observed: {result.min_soc_pct:.2f}%")
    lines.append(f"Final SOC: {result.final_soc_pct:.2f}%")
    lines.append(f"Peak model-reported motor power: {result.max_motor_power_w:,.1f} W")
    lines.append("")

    lines.append("BATTERY STATE")
    lines.append("-" * 84)
    lines.append(f"Battery capacity in source model: {BATTERY_J / 1_000_000:.3f} MJ ({BATTERY_WH:,.2f} Wh)")
    lines.append(f"Initial stored energy at {result.initial_soc_pct:.2f}% SOC: {format_energy(start_energy_j)}")
    lines.append(f"Final stored energy at {result.final_soc_pct:.2f}% SOC:   {format_energy(final_energy_j)}")
    lines.append(f"Observed battery energy change from SOC: {format_energy(actual_delta_j)}")
    lines.append("")

    lines.append("ENERGY GAINED / GENERATED")
    lines.append("-" * 84)
    lines.append(f"  Gross solar electricity generated while driving:   {format_energy(totals.get('solar_drive_j', 0.0))}")
    lines.append(f"  Gross solar electricity generated while stationary:{format_energy(totals.get('solar_stationary_j', 0.0))}")
    lines.append(f"  TOTAL gross solar-array electrical output:         {format_energy(totals.get('solar_drive_j', 0.0) + totals.get('solar_stationary_j', 0.0))}")
    lines.append(f"  Regenerative-braking energy returned electrically:{format_energy(totals.get('regen_electrical_j', 0.0))}")
    lines.append(f"  Stationary charging energy actually stored:        {format_energy(totals.get('stationary_charge_stored_j', 0.0))}")
    lines.append("")

    lines.append("ENERGY SPENT / CONVERSION LOSSES")
    lines.append("-" * 84)
    lines.append("Wheel-level mechanical work breakdown (these are upstream of electrical draw;")
    lines.append("do not add these values again to the battery-side energy flows below):")
    lines.append(f"  Rolling resistance work:                           {format_energy(totals.get('wheel_rolling_j', 0.0))}")
    lines.append(f"  Aerodynamic drag work:                             {format_energy(totals.get('wheel_aero_drag_j', 0.0))}")
    lines.append(f"  Gravity work (signed; negative means net downhill):{format_energy(totals.get('wheel_gravity_signed_j', 0.0))}")
    lines.append(f"  Net signed mechanical work at wheels:              {format_energy(totals.get('wheel_mechanical_signed_j', 0.0))}")
    lines.append("")
    lines.append("Battery-side terms:")
    lines.append(f"  Electrical propulsion draw:                        {format_energy(totals.get('traction_electrical_j', 0.0))}")
    lines.append(f"  Motor/drivetrain conversion loss (95% efficiency): {format_energy(totals.get('drivetrain_loss_j', 0.0))}")
    lines.append(f"  Mechanical energy not recovered in regeneration:   {format_energy(totals.get('regen_loss_j', 0.0))}")
    lines.append(f"  Fixed 70 W loss term while driving:                 {format_energy(totals.get('fixed_loss_j', 0.0))}")
    lines.append(f"  5 W auxiliary load while driving:                   {format_energy(totals.get('aux_energy_j', 0.0))}")
    lines.append("")

    lines.append("STATIONARY SOLAR CHARGING DETAIL")
    lines.append("-" * 84)
    lines.append(f"  Gross stationary solar-array output:               {format_energy(totals.get('solar_stationary_j', 0.0))}")
    lines.append(f"  10 W offset/threshold used by source formula:      {format_energy(totals.get('stationary_offset_j', 0.0))}")
    lines.append(f"  Energy offered to battery after offset:            {format_energy(totals.get('stationary_charge_offered_j', 0.0))}")
    lines.append(f"  Energy actually stored:                             {format_energy(totals.get('stationary_charge_stored_j', 0.0))}")
    lines.append(f"  Offered energy not stored at SOC ceiling:           {format_energy(totals.get('stationary_charge_curtailed_j', 0.0))}")
    lines.append(f"  Signed SOC-ceiling adjustment, if any:              {format_energy(totals.get('stationary_soc_cap_adjustment_j', 0.0))}")
    if ledger.stationary_modes:
        lines.append("  By stationary period/stop:")
        for mode, info in ledger.stationary_modes.items():
            lines.append(
                f"    {mode}: {timedelta(seconds=info['seconds'])}; "
                f"solar {format_energy(info['solar_j'])}; "
                f"stored {format_energy(info['stored_j'])}"
            )
    lines.append("")

    lines.append("BATTERY ENERGY BALANCE CHECK")
    lines.append("-" * 84)
    lines.append(f"  From driving and stationary model flows:           {format_energy(flow_delta_j)}")
    lines.append(f"  Observed change from initial to final SOC:          {format_energy(actual_delta_j)}")
    lines.append(f"  Difference (observed minus model-flow sum):         {format_energy(residual_j)}")
    lines.append("  A non-zero difference can occur if the source model applies its low-SOC floor/clamp.")
    lines.append("")

    lines.append("ENERGY BY 600 KM BAND")
    lines.append("-" * 84)
    lines.append(
        f"{'Band':<8}{'Distance':<19}{'Speed':>9}{'Solar drive':>15}{'Regen':>13}"
        f"{'Propulsion':>15}{'70W loss':>13}{'5W aux':>12}{'Station charge':>17}{'Net battery':>15}"
    )
    for i, bucket in enumerate(ledger.by_segment):
        lo, hi = segment_bounds(i)
        solar = bucket.get("solar_drive_j", 0.0)
        regen = bucket.get("regen_electrical_j", 0.0)
        traction = bucket.get("traction_electrical_j", 0.0)
        fixed = bucket.get("fixed_loss_j", 0.0)
        aux = bucket.get("aux_energy_j", 0.0)
        charge = bucket.get("stationary_soc_delta_j", 0.0)
        net = solar + regen - traction - fixed - aux + charge
        distance_text = f"{lo:.0f}-{hi:.0f} km"
        lines.append(
            f"{i + 1:<8}{distance_text:<19}{result.speeds_kmh[i]:>8.1f}  "
            f"{joules_to_kwh(solar):>12.3f}  {joules_to_kwh(regen):>10.3f}  "
            f"{joules_to_kwh(traction):>12.3f}  {joules_to_kwh(fixed):>10.3f}  "
            f"{joules_to_kwh(aux):>9.3f}  {joules_to_kwh(charge):>14.3f}  {joules_to_kwh(net):>12.3f}"
        )
    lines.append("  Segment table energy columns are kWh; 'Net battery' is calculated from battery-side flows.")
    lines.append("")
    lines.append("Model note: this report preserves the equations and rules in optimizer.py.")
    lines.append("It does not search for, adjust, or optimize the five speeds.")
    lines.append("=" * 84)
    return "\n".join(lines)


def save_segment_csv(result: SimulationResult, output_path: Path) -> None:
    """Write a machine-readable energy breakdown by the five speed bands."""
    fields = [
        "segment", "start_km", "end_km", "speed_kmh",
        "solar_driving_kWh", "solar_stationary_kWh", "regeneration_kWh",
        "propulsion_electrical_draw_kWh", "rolling_resistance_work_kWh",
        "aerodynamic_drag_work_kWh", "gravity_work_signed_kWh",
        "drivetrain_loss_kWh", "regeneration_loss_kWh",
        "fixed_70W_loss_kWh", "auxiliary_5W_kWh",
        "stationary_charge_stored_kWh", "stationary_charge_curtailed_kWh",
        "net_battery_energy_from_flows_kWh",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, bucket in enumerate(result.ledger.by_segment):
            lo, hi = segment_bounds(i)
            solar_drive = bucket.get("solar_drive_j", 0.0)
            solar_stationary = bucket.get("solar_stationary_j", 0.0)
            regen = bucket.get("regen_electrical_j", 0.0)
            traction = bucket.get("traction_electrical_j", 0.0)
            fixed = bucket.get("fixed_loss_j", 0.0)
            aux = bucket.get("aux_energy_j", 0.0)
            charge_delta = bucket.get("stationary_soc_delta_j", 0.0)
            net = solar_drive + regen - traction - fixed - aux + charge_delta
            writer.writerow({
                "segment": i + 1,
                "start_km": lo,
                "end_km": hi,
                "speed_kmh": result.speeds_kmh[i],
                "solar_driving_kWh": joules_to_kwh(solar_drive),
                "solar_stationary_kWh": joules_to_kwh(solar_stationary),
                "regeneration_kWh": joules_to_kwh(regen),
                "propulsion_electrical_draw_kWh": joules_to_kwh(traction),
                "rolling_resistance_work_kWh": joules_to_kwh(bucket.get("wheel_rolling_j", 0.0)),
                "aerodynamic_drag_work_kWh": joules_to_kwh(bucket.get("wheel_aero_drag_j", 0.0)),
                "gravity_work_signed_kWh": joules_to_kwh(bucket.get("wheel_gravity_signed_j", 0.0)),
                "drivetrain_loss_kWh": joules_to_kwh(bucket.get("drivetrain_loss_j", 0.0)),
                "regeneration_loss_kWh": joules_to_kwh(bucket.get("regen_loss_j", 0.0)),
                "fixed_70W_loss_kWh": joules_to_kwh(fixed),
                "auxiliary_5W_kWh": joules_to_kwh(aux),
                "stationary_charge_stored_kWh": joules_to_kwh(bucket.get("stationary_charge_stored_j", 0.0)),
                "stationary_charge_curtailed_kWh": joules_to_kwh(bucket.get("stationary_charge_curtailed_j", 0.0)),
                "net_battery_energy_from_flows_kWh": joules_to_kwh(net),
            })


def run_energy_sheet(
    v_0_600_kmh: float,
    v_600_1200_kmh: float,
    v_1200_1800_kmh: float,
    v_1800_2400_kmh: float,
    v_2400_finish_kmh: float,
) -> SimulationResult:
    """Run the existing simulation with five caller-supplied, fixed speeds.

    Example:
        run_energy_sheet(65, 70, 70, 65, 60)
    """
    speeds = (
        v_0_600_kmh,
        v_600_1200_kmh,
        v_1200_1800_kmh,
        v_1800_2400_kmh,
        v_2400_finish_kmh,
    )
    route = load_route()
    fields, race_start_date = load_solar_fields()
    result = simulate_fixed_speeds(speeds, route, fields, race_start_date)

    report = render_energy_sheet(result)
    report_path = HERE / "wsc_energy_sheet.txt"
    csv_path = HERE / "wsc_energy_by_600km.csv"
    report_path.write_text(report + "\n", encoding="utf-8")
    save_segment_csv(result, csv_path)
    print("\n" + report)
    print(f"\nFull text report saved to: {report_path}")
    print(f"Segment CSV saved to:      {csv_path}")
    return result


def main() -> None:
    # Change these five arguments for your fixed-speed run.
    run_energy_sheet(
        65.0,  # 0-600 km
        65.0,  # 600-1200 km
        65.0,  # 1200-1800 km
        65.0,  # 1800-2400 km
        65.0,  # 2400-3022 km
    )


if __name__ == "__main__":
    main()

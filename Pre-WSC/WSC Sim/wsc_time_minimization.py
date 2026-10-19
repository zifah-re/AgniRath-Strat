"""2025 BWSC time-minimisation simulation.

Edit the values in the CONFIGURATION block, then run:

    python wsc_time_minimization.py

It expects the 144-query solar files made by query_wsc_solar.py.  The model
uses a single cumulative WSC route, enforces ACST 08:00--17:00 driving, adds
the requested deterministic daily breakdowns and 30-minute control stops, and
optimises one speed per equally-sized distance split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution


# ============================ CONFIGURATION ================================
ACST = ZoneInfo("Australia/Darwin")
RACE_START_DATE = date(2026, 8, 24)
MAX_RACE_DAYS = 6
DRIVE_START = time(8, 0)
DRIVE_END = time(17, 0)

N_SPEED_SPLITS = 10                 # Any integer >= 5; equal-distance splits.
SPEED_MIN_KMH = 60.0
SPEED_MAX_KMH = 80.0
BREAKDOWN_SECONDS_PER_DAY = 0 # Set to 0 to remove the daily breakdown.
BREAKDOWN_RANDOM_SEED = 2025       # Makes the breakdown schedule repeatable.
OPTIMIZER_MAXITER = 35             # Increase for a more thorough search.
OPTIMIZER_POPSIZE = 8

INITIAL_SOC_PCT = 95.0
SOC_MIN_PCT = 20.0
SOC_MAX_PCT = 95.0
RACE_FINISH_KM = 3022.0            # Finish line, not the 2,998 km timing line.
ROUTE_STEP_KM = 1.0                # Integration resolution.

CONTROL_STOPS_KM = {
    "Katherine": 322.0, "Dunmarra": 633.0, "Tennant Creek": 988.0,
    "Barrow Creek": 1211.0, "Alice Springs": 1496.0, "Erldunda": 1696.0,
    "Coober Pedy": 2178.0, "Glendambo": 2432.0, "Port Augusta": 2719.0,
}
CONTROL_STOP_SECONDS = 30 * 60

# Carry these forward from Fallback Model/multiday.py.  They are deliberately
# here rather than hidden in the optimiser, so vehicle assumptions are easy to
# audit and tune.
MASS_KG, G_MS2, CRR, CDA_M2, AIR_DENSITY = 300.0, 9.81, 0.007, 0.16, 1.2
ARRAY_AREA_M2, ARRAY_EFFICIENCY = 5.78, 0.21
MOTOR_EFF, REGEN_EFF, P_LOSS_W, MAX_MOTOR_POWER_W = 0.95, 0.70, 70.0, 4000.0
BATTERY_WH = 3024.0
TEMP_COEFF_PER_C, NOCT_C = 0.004, 45.0

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
    """The candidate needs a calendar day that the agreed query plan omitted."""


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


class SolarField:
    """Nearest-route-location, time-interpolated historical Solcast field."""

    def __init__(self, json_path: Path):
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if not raw:
            raise ValueError(f"No solar records in {json_path}")
        self.coords = np.array([(record["lat"], record["lon"]) for record in raw], dtype=float)
        # Solcast may return a slightly different number of five-minute samples
        # per location (for example around a historical data boundary). Keep
        # independent time axes rather than requiring a rectangular array.
        self.times: list[np.ndarray] = []
        self.metrics: dict[str, list[np.ndarray]] = {name: [] for name in ("ghi", "dni", "air_temp")}
        for name in ("ghi", "dni", "air_temp"):
            for record_index, record in enumerate(raw):
                rows = sorted(record["data"], key=lambda row: row["period_end"])
                if not rows:
                    raise ValueError(f"Solar location {record_index} in {json_path.name} has no records")
                if name == "ghi":
                    self.times.append(np.array([datetime.fromisoformat(row["period_end"]).timestamp() for row in rows]))
                self.metrics[name].append(np.array([float(row.get(name, 0.0) or 0.0) for row in rows], dtype=float))
        self._coord_cache: dict[tuple[float, float], int] = {}

    def values(self, coord: np.ndarray, timestamp: datetime) -> tuple[float, float, float]:
        location_index = self.location_index(coord)
        t = timestamp.timestamp()
        location_times = self.times[location_index]
        if t < location_times[0] - 1 or t > location_times[-1] + 1:
            first = datetime.fromtimestamp(location_times[0], tz=ACST).date()
            last = datetime.fromtimestamp(location_times[-1], tz=ACST).date()
            raise SolarCoverageError(f"Solar file covers {first} to {last}, not {timestamp.date()}")
        return tuple(float(np.interp(t, location_times, self.metrics[name][location_index])) for name in ("ghi", "dni", "air_temp"))

    def location_index(self, coord: np.ndarray) -> int:
        key = (round(float(coord[0]), 5), round(float(coord[1]), 5))
        location_index = self._coord_cache.get(key)
        if location_index is None:
            # Route samples are closely spaced, so local lat/lon distance is adequate
            # for choosing one of the Solcast locations.
            location_index = int(np.argmin(np.sum((self.coords - coord) ** 2, axis=1)))
            self._coord_cache[key] = location_index
        return location_index

    def covers(self, coord: np.ndarray, timestamp: datetime) -> bool:
        times = self.times[self.location_index(coord)]
        return times[0] - 1 <= timestamp.timestamp() <= times[-1] + 1


def load_route() -> Route:
    profile = json.loads(MASTER_ROUTE.read_text(encoding="utf-8"))["profile"]
    source_d = np.asarray(profile["Distance"], dtype=float)
    grid = np.unique(np.r_[np.arange(0.0, RACE_FINISH_KM, ROUTE_STEP_KM), RACE_FINISH_KM,
                           list(CONTROL_STOPS_KM.values())])
    if source_d[-1] < RACE_FINISH_KM:
        raise ValueError(f"Route ends at {source_d[-1]:.2f} km, before finish {RACE_FINISH_KM:.2f} km")
    coords = np.asarray(profile["Coordinates"], dtype=float)

    def aligned(values: object, name: str) -> np.ndarray:
        """Match a route attribute to Distance, padding a missing final sample."""
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


def load_solar_fields() -> list[tuple[float, float, SolarField]]:
    missing = [name for _, _, name in ROUTE_SECTIONS if not (SOLAR_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("Solar data has not been queried. Run `python query_wsc_solar.py --execute` first. Missing: " + ", ".join(missing))
    return [(start, end, SolarField(SOLAR_DIR / name)) for start, end, name in ROUTE_SECTIONS]


def field_for_distance(distance_km: float, coord: np.ndarray, timestamp: datetime,
                       fields: list[tuple[float, float, SolarField]]) -> SolarField:
    """Choose the closest spatial section which has data for this instant.

    The requested query plan gives Day 1 only one date, while subsequent
    sections overlap dates as an optimisation buffer. If a vehicle carries
    over onto the next calendar day before crossing a route-file boundary,
    this safely uses the neighbouring section's buffered weather instead of
    extrapolating or failing.
    """
    ranked = sorted(
        fields,
        key=lambda item: 0.0 if item[0] <= distance_km <= item[1]
        else min(abs(distance_km - item[0]), abs(distance_km - item[1])),
    )
    for _, _, field in ranked:
        if field.covers(coord, timestamp):
            return field
    raise SolarCoverageError(f"No queried solar data covers {timestamp.date()} near {distance_km:.1f} km")


def solar_power_w(field: SolarField, coord: np.ndarray, timestamp: datetime, speed_ms: float) -> float:
    ghi, dni, ambient_c = field.values(coord, timestamp)
    # A near-horizontal 4° array is dominated by GHI. DNI is retained as a
    # modest orientation contribution; this is intentionally the same level of
    # solar detail as the fallback model's array-power calculation.
    plane_irradiance = max(0.0, 0.95 * ghi + 0.05 * dni)
    cell_c = ambient_c + (plane_irradiance / 800.0) * (NOCT_C - 20.0) * np.exp(-0.03 * speed_ms)
    thermal_factor = np.clip(1.0 - TEMP_COEFF_PER_C * (cell_c - 25.0), 0.70, 1.05)
    return float(plane_irradiance * ARRAY_AREA_M2 * ARRAY_EFFICIENCY * thermal_factor)


def electrical_power(speed_ms: float, gradient_pct: float, solar_w: float) -> tuple[float, float]:
    grade = gradient_pct / 100.0
    force = (0.5 * AIR_DENSITY * CDA_M2 * speed_ms**2 + MASS_KG * G_MS2 * CRR + MASS_KG * G_MS2 * grade)
    mechanical_w = force * speed_ms
    motor_w = min(mechanical_w / MOTOR_EFF, MAX_MOTOR_POWER_W) if mechanical_w > 0 else 0.0
    drive_w = mechanical_w / MOTOR_EFF if mechanical_w >= 0 else mechanical_w * REGEN_EFF
    return solar_w - drive_w - P_LOSS_W, motor_w


def breakdown_offsets() -> list[float]:
    """One reproducible in-window offset per race day, in driving seconds."""
    drive_seconds = (datetime.combine(date.min, DRIVE_END) - datetime.combine(date.min, DRIVE_START)).total_seconds()
    if BREAKDOWN_SECONDS_PER_DAY <= 0:
        return [float("inf")] * MAX_RACE_DAYS
    if BREAKDOWN_SECONDS_PER_DAY >= drive_seconds:
        raise ValueError("BREAKDOWN_SECONDS_PER_DAY must be shorter than the 08:00--17:00 drive window")
    rng = np.random.default_rng(BREAKDOWN_RANDOM_SEED)
    return list(rng.uniform(0.0, drive_seconds - BREAKDOWN_SECONDS_PER_DAY, size=MAX_RACE_DAYS))


def speed_at(distance_km: float, speeds_kmh: np.ndarray) -> float:
    split_width = RACE_FINISH_KM / len(speeds_kmh)
    return float(speeds_kmh[min(int(distance_km / split_width), len(speeds_kmh) - 1)])


def simulate(speeds_kmh: Iterable[float], route: Route, fields: list[tuple[float, float, SolarField]], keep_trace: bool = True) -> SimulationResult:
    speeds = np.asarray(list(speeds_kmh), dtype=float)
    if len(speeds) < 5:
        raise ValueError("N_SPEED_SPLITS must be at least 5")
    control_distances = set(CONTROL_STOPS_KM.values())
    offsets = breakdown_offsets()
    breakdown_done = [False] * MAX_RACE_DAYS
    distance, soc = 0.0, INITIAL_SOC_PCT
    now = datetime.combine(RACE_START_DATE, DRIVE_START, tzinfo=ACST)
    max_motor, min_soc = 0.0, soc
    trace_d, trace_soc, trace_t, trace_day = [distance], [soc], [now], [0]

    def at_distance(d: float) -> tuple[np.ndarray, float]:
        return (np.array([np.interp(d, route.distance_km, route.coords[:, axis]) for axis in (0, 1)]),
                float(np.interp(d, route.distance_km, route.gradient_pct)))

    def add_stationary(seconds: float) -> None:
        nonlocal soc, now, min_soc
        # 5-minute steps make the static solar-energy integration agree with data cadence.
        remaining = seconds
        coord, _ = at_distance(distance)
        while remaining > 1e-6:
            dt = min(300.0, remaining)
            sample_time = now + timedelta(seconds=dt / 2)
            field = field_for_distance(distance, coord, sample_time, fields)
            solar_w = solar_power_w(field, coord, sample_time, 0.0)
            soc = min(SOC_MAX_PCT, soc + max(0.0, solar_w - 10.0) * dt / (BATTERY_WH * 36.0))
            min_soc = min(min_soc, soc)
            now += timedelta(seconds=dt)
            remaining -= dt

    while distance < RACE_FINISH_KM - 1e-7:
        race_day = (now.date() - RACE_START_DATE).days
        if race_day >= MAX_RACE_DAYS:
            return SimulationResult(np.array(trace_d), np.array(trace_soc), trace_t, np.array(trace_day), None, min_soc, max_motor, speeds)
        day_start = datetime.combine(now.date(), DRIVE_START, tzinfo=ACST)
        day_end = datetime.combine(now.date(), DRIVE_END, tzinfo=ACST)
        if now < day_start:
            add_stationary((day_start - now).total_seconds())
            continue
        if now >= day_end - timedelta(milliseconds=1):
            now = day_end
            next_start = datetime.combine(now.date() + timedelta(days=1), DRIVE_START, tzinfo=ACST)
            add_stationary((next_start - now).total_seconds())
            continue

        elapsed_today = (now - day_start).total_seconds()
        if not breakdown_done[race_day] and elapsed_today >= offsets[race_day] - 1e-3:
            breakdown_done[race_day] = True
            add_stationary(BREAKDOWN_SECONDS_PER_DAY)
            continue

        speed_kmh = speed_at(distance, speeds)
        speed_ms = speed_kmh / 3.6
        next_grid = route.distance_km[np.searchsorted(route.distance_km, distance + 1e-7, side="right")]
        remaining_window = (day_end - now).total_seconds()
        remaining_to_breakdown = (offsets[race_day] - elapsed_today) if not breakdown_done[race_day] else np.inf
        travel_seconds = max(0.0, min((next_grid - distance) * 1000.0 / speed_ms, remaining_window, remaining_to_breakdown))
        if travel_seconds <= 1e-5:
            # Snap a floating-point boundary to its event; otherwise a tiny
            # positive remainder can repeatedly re-enter this loop forever.
            if remaining_window <= 1e-3:
                now = day_end
            elif remaining_to_breakdown <= 1e-3:
                breakdown_done[race_day] = True
                add_stationary(BREAKDOWN_SECONDS_PER_DAY)
            else:
                raise RuntimeError("Simulation could not advance to its next route or time event")
            continue
        end_distance = distance + speed_ms * travel_seconds / 1000.0
        mid_distance = (distance + end_distance) / 2.0
        coord, grade = at_distance(mid_distance)
        sample_time = now + timedelta(seconds=travel_seconds / 2)
        field = field_for_distance(mid_distance, coord, sample_time, fields)
        solar_w = solar_power_w(field, coord, sample_time, speed_ms)
        net_w, motor_w = electrical_power(speed_ms, grade, solar_w)
        soc += net_w * travel_seconds / (BATTERY_WH * 36.0)
        min_soc, max_motor = min(min_soc, soc), max(max_motor, motor_w)
        now += timedelta(seconds=travel_seconds)
        distance = end_distance
        if keep_trace:
            trace_d.append(distance); trace_soc.append(soc); trace_t.append(now); trace_day.append(race_day)
        if any(abs(distance - stop) < 1e-5 for stop in control_distances):
            add_stationary(CONTROL_STOP_SECONDS)
            if keep_trace:
                trace_d.append(distance); trace_soc.append(soc); trace_t.append(now); trace_day.append(race_day)

    return SimulationResult(np.array(trace_d), np.array(trace_soc), trace_t, np.array(trace_day), now, min_soc, max_motor, speeds)


def optimise(route: Route, fields: list[tuple[float, float, SolarField]]) -> SimulationResult:
    if N_SPEED_SPLITS < 5:
        raise ValueError("N_SPEED_SPLITS must be at least 5")
    def objective(x: np.ndarray) -> float:
        try:
            result = simulate(x, route, fields, keep_trace=False)
        except SolarCoverageError:
            # The approved dataset intentionally ends Aug 28. Candidates which
            # spill into Aug 29 are infeasible, rather than extrapolated.
            return 1e9
        if result.finish_time is None:
            return 1e8 + max(0.0, RACE_FINISH_KM - result.distance_km[-1]) * 1e4
        seconds = (result.finish_time - datetime.combine(RACE_START_DATE, DRIVE_START, tzinfo=ACST)).total_seconds()
        soc_penalty = max(0.0, 1e8+ (SOC_MIN_PCT - result.min_soc_pct)* 2e5)
        power_penalty = max(0.0, result.max_motor_power_w - MAX_MOTOR_POWER_W) * 1e4
        return seconds + soc_penalty + power_penalty

    result = differential_evolution(
        objective, [(SPEED_MIN_KMH, SPEED_MAX_KMH)] * N_SPEED_SPLITS,
        seed=BREAKDOWN_RANDOM_SEED, maxiter=OPTIMIZER_MAXITER, popsize=OPTIMIZER_POPSIZE,
        polish=True, updating="immediate", workers=1,
    )
    print(f"Optimiser objective: {result.fun:.0f}; speeds: {np.round(result.x, 2).tolist()}")
    return simulate(result.x, route, fields, keep_trace=True)


def plot_result(result: SimulationResult) -> Path:
    output = HERE / "wsc_2025_optimised_soc.png"
    figure, axis = plt.subplots(figsize=(15, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, MAX_RACE_DAYS))
    for day in range(MAX_RACE_DAYS):
        mask = result.day_index == day
        if np.any(mask):
            axis.plot(result.distance_km[mask], result.soc_pct[mask], color=colors[day], linewidth=2.0, label=f"Race day {day + 1}")
    for name, stop in CONTROL_STOPS_KM.items():
        axis.axvline(stop, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        axis.text(stop, 2, name, rotation=90, va="bottom", ha="right", fontsize=7)
    axis.axvline(2998, color="grey", linestyle=":", linewidth=1.0, label="Timing stop (no hold)")
    axis.axvline(RACE_FINISH_KM, color="green", linestyle="--", linewidth=1.2, label="Finish")
    axis.axhline(SOC_MIN_PCT, color="red", linestyle="--", linewidth=1.0, label=f"Minimum SOC ({SOC_MIN_PCT:.0f}%)")
    axis.set(xlim=(0, RACE_FINISH_KM), ylim=(0, 100), xlabel="Cumulative distance (km)", ylabel="State of charge (%)",
             title="2025 BWSC time-minimised SOC profile")
    axis.grid(True, linestyle=":", alpha=0.5)
    axis.legend(loc="lower left", ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    return output


def main() -> None:
    route, fields = load_route(), load_solar_fields()
    result = optimise(route, fields)
    if result.finish_time is None:
        print("No feasible finish within six days using the agreed Aug 24--28 solar coverage.")
        return
    print(f"Finish: {result.finish_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Minimum SOC: {result.min_soc_pct:.2f}% | Peak motor draw: {result.max_motor_power_w:.0f} W")
    print(f"Speed by {N_SPEED_SPLITS} equal-distance splits (km/h): {np.round(result.speed_kmh, 2).tolist()}")
    print(f"Saved plot: {plot_result(result)}")


if __name__ == "__main__":
    main()

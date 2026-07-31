#!/usr/bin/env python3
"""
solar_tilt_optimizer.py
========================

Given a stopping location (latitude/longitude), the local start time of a
charging stop, how long the stop will last, and the longitudinal road
gradient the car is parked on, this script finds the fixed panel tilt
angle and car bearing (compass heading) that maximise the total solar
energy collected by the array over the whole charging window.

The underlying solar-position / irradiance model is the hand-calculation
method described in:

    M. Elshafei, A. Al-Qutub, A-W. A. Saif, "Solar Car Optimization For
    the World Solar Challenge", 2016 13th Int'l Multi-Conference on
    Systems, Signals & Devices (SSD), pp. 751-756.


Because a fixed (non-tracking) solar-car array cannot re-aim itself
during the stop, the "ideal" tilt/bearing reported here is the single,
fixed orientation of the panel plane that maximises the *total* energy
collected across the whole [start, start + duration] window -- not the
instantaneous optimum at any single moment.

Usage (CLI)
-----------
    python solar_tilt_optimizer.py --lat -12.4667 --lon 130.8330 \\
        --start 2015-10-01T09:00:00 --tz 9.5 --duration 2.0 --gradient 0.02

Usage (as a library)
---------------------
    from solar_tilt_optimizer import optimize_orientation
    result = optimize_orientation(lat=-12.4667, lon=130.833, tz_offset=9.5,
                                   start=datetime(2015, 10, 1, 9, 0, 0),
                                   duration_hours=2.0, gradient=0.02)
    print(result)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants (from the reference paper / standard solar geometry)
# ---------------------------------------------------------------------------

SOLAR_CONSTANT_W_M2 = 1366.0      # G_sc in the paper
CLEAR_SKY_FACTOR = 0.7            # Ghat = 0.7 * G1 approximation used in the paper
DEFAULT_PANEL_AREA_M2 = 5.95       # As, matches the paper's worked example
DEFAULT_PANEL_EFFICIENCY = 0.18  # eta_s, matches the paper's worked example

COMPASS_POINTS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


# ---------------------------------------------------------------------------
# Core solar-geometry equations (paper Eqs. 2-9, standard Duffie & Beckman
# angle-of-incidence relation for Eq. 12)
# ---------------------------------------------------------------------------

def day_of_year(dt: datetime) -> int:
    """N: the day-of-year number used throughout the paper."""
    return dt.timetuple().tm_yday


def declination_deg(n: int) -> float:
    """Solar declination delta, in degrees. Paper Eq. 5."""
    return 23.45 * math.sin(math.radians(360.0 * (284 + n) / 365.0))


def equation_of_time_min(n: int) -> float:
    """Equation of time E, in minutes. Used inside paper Eq. 3."""
    b = math.radians((n - 1) * 360.0 / 365.0)
    return 229.2 * (
        0.000075
        + 0.001868 * math.cos(b)
        - 0.032077 * math.sin(b)
        - 0.014615 * math.cos(2 * b)
        - 0.04089 * math.sin(2 * b)
    )


def solar_time(clock_dt: datetime, lon_deg: float, tz_offset_hours: float) -> datetime:
    """
    Convert a local clock (standard) time to apparent solar time.
    Paper Eq. 3. Longitude and tz_offset use the convention "positive = East
    of Greenwich" (e.g. India = +80.27 deg, UTC+5.5 h).
    """
    n = day_of_year(clock_dt)
    eot = equation_of_time_min(n)
    standard_meridian_deg = tz_offset_hours * 15.0
    correction_min = 4.0 * (standard_meridian_deg - lon_deg) + eot
    return clock_dt + timedelta(minutes=correction_min)


def hour_angle_deg(solar_dt: datetime) -> float:
    """
    Hour angle omega, in degrees: 15 deg/hour from solar noon,
    negative in the morning, positive in the afternoon. Paper Eq. 4.
    """
    hours = solar_dt.hour + solar_dt.minute / 60.0 + solar_dt.second / 3600.0
    return 15.0 * (hours - 12.0)


def extraterrestrial_flux(n: int) -> float:
    """G1: extraterrestrial flux with eccentricity correction. Paper Eq. 6."""
    return SOLAR_CONSTANT_W_M2 * (1.0 + 0.033 * math.cos(math.radians(360.0 * n / 365.0)))


def clear_sky_beam_flux(n: int) -> float:
    """Ghat: clear-sky beam flux approximation used just before paper Eq. 7."""
    return CLEAR_SKY_FACTOR * extraterrestrial_flux(n)


def cos_zenith_angle(lat_deg: float, decl_deg: float, hour_ang_deg: float) -> float:
    """cos(theta_z): cosine of the solar zenith angle (sun above horizon if > 0)."""
    phi = math.radians(lat_deg)
    delta = math.radians(decl_deg)
    w = math.radians(hour_ang_deg)
    return math.cos(phi) * math.cos(delta) * math.cos(w) + math.sin(phi) * math.sin(delta)


def cos_incidence_angle(
    lat_deg: float, decl_deg: float, hour_ang_deg: float,
    tilt_deg: float, surface_azimuth_paper_deg: float,
) -> float:
    """
    cos(theta): cosine of the angle of incidence between the sun's rays and
    the panel-surface normal, for a panel tilted `tilt_deg` from horizontal
    and facing `surface_azimuth_paper_deg` (paper convention: measured from
    south, west positive, east negative). This is paper Eq. 12 (the
    standard Duffie & Beckman tilted-surface incidence-angle formula).
    """
    phi = math.radians(lat_deg)
    delta = math.radians(decl_deg)
    beta = math.radians(tilt_deg)
    gamma = math.radians(surface_azimuth_paper_deg)
    w = math.radians(hour_ang_deg)

    return (
        math.sin(phi) * math.sin(delta) * math.cos(beta)
        - math.cos(phi) * math.sin(delta) * math.sin(beta) * math.cos(gamma)
        + math.cos(phi) * math.cos(delta) * math.cos(beta) * math.cos(w)
        + math.sin(phi) * math.cos(delta) * math.sin(beta) * math.cos(gamma) * math.cos(w)
        + math.cos(delta) * math.sin(beta) * math.sin(gamma) * math.sin(w)
    )


def sunset_hour_angle_deg(lat_deg: float, decl_deg: float) -> float:
    """omega_s: the sunset hour angle, in degrees. Paper Eq. 9."""
    phi = math.radians(lat_deg)
    delta = math.radians(decl_deg)
    x = -math.tan(phi) * math.tan(delta)
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.acos(x))


def approx_sunrise_sunset_clock(
    date_dt: datetime, lat_deg: float, lon_deg: float, tz_offset_hours: float
) -> Tuple[datetime, datetime]:
    """
    Approximate local clock sunrise/sunset for the given date, by inverting
    the solar-time correction using a single (noon) estimate of the
    equation of time. Good enough for planning purposes.
    """
    n = day_of_year(date_dt)
    decl = declination_deg(n)
    ws = sunset_hour_angle_deg(lat_deg, decl)
    noon_solar = date_dt.replace(hour=12, minute=0, second=0, microsecond=0)
    eot = equation_of_time_min(n)
    standard_meridian_deg = tz_offset_hours * 15.0
    correction_min = 4.0 * (standard_meridian_deg - lon_deg) + eot
    solar_noon_clock = noon_solar - timedelta(minutes=correction_min)
    sunrise = solar_noon_clock - timedelta(hours=ws / 15.0)
    sunset = solar_noon_clock + timedelta(hours=ws / 15.0)
    return sunrise, sunset


def compass_to_paper_azimuth(bearing_deg: float) -> float:
    """
    Convert a compass bearing (0=N, 90=E, 180=S, 270=W, clockwise from
    North) into the paper's surface-azimuth convention (0=South, positive
    towards West, negative towards East).
    """
    gamma = bearing_deg - 180.0
    gamma = (gamma + 180.0) % 360.0 - 180.0
    return gamma


def bearing_to_compass_label(bearing_deg: float) -> str:
    idx = int((bearing_deg % 360.0) / 22.5 + 0.5) % 16
    return COMPASS_POINTS[idx]


# ---------------------------------------------------------------------------
# Energy integration for a fixed (tilt, bearing) panel over the charge window
# ---------------------------------------------------------------------------

def instantaneous_tilted_irradiance(
    dt_clock: datetime, lat_deg: float, lon_deg: float, tz_offset_hours: float,
    tilt_deg: float, bearing_deg: float,
) -> float:
    """
    Beam irradiance (W/m^2) striking a panel tilted `tilt_deg` and facing
    compass bearing `bearing_deg`, at local clock time `dt_clock`.
    Returns 0 if the sun is below the horizon or below the panel plane.
    """
    n = day_of_year(dt_clock)
    decl = declination_deg(n)
    s_dt = solar_time(dt_clock, lon_deg, tz_offset_hours)
    w = hour_angle_deg(s_dt)

    if cos_zenith_angle(lat_deg, decl, w) <= 0.0:
        return 0.0

    surf_az = compass_to_paper_azimuth(bearing_deg)
    cos_theta = cos_incidence_angle(lat_deg, decl, w, tilt_deg, surf_az)
    if cos_theta <= 0.0:
        return 0.0

    return clear_sky_beam_flux(n) * cos_theta


def collected_energy_wh_per_m2(
    lat_deg: float, lon_deg: float, tz_offset_hours: float,
    start: datetime, duration_hours: float,
    tilt_deg: float, bearing_deg: float, step_minutes: float = 2.0,
) -> float:
    """
    Numerically integrate `instantaneous_tilted_irradiance` over
    [start, start + duration_hours] using the trapezoidal rule, returning
    total energy in Wh/m^2 for a fixed panel orientation.
    """
    if duration_hours <= 0:
        return 0.0

    n_steps = max(1, int(round(duration_hours * 60.0 / step_minutes)))
    dt_step = timedelta(minutes=duration_hours * 60.0 / n_steps)

    t = start
    prev = instantaneous_tilted_irradiance(t, lat_deg, lon_deg, tz_offset_hours, tilt_deg, bearing_deg)
    total_wh = 0.0
    for _ in range(n_steps):
        t_next = t + dt_step
        cur = instantaneous_tilted_irradiance(
            t_next, lat_deg, lon_deg, tz_offset_hours, tilt_deg, bearing_deg
        )
        hours_step = dt_step.total_seconds() / 3600.0
        total_wh += 0.5 * (prev + cur) * hours_step
        prev = cur
        t = t_next
    return total_wh


# ---------------------------------------------------------------------------
# Optimisation: coarse grid search followed by a local refinement pass
# ---------------------------------------------------------------------------

def _grid_search(
    lat_deg: float, lon_deg: float, tz_offset_hours: float,
    start: datetime, duration_hours: float,
    tilt_range: Tuple[float, float], tilt_step: float,
    bearing_range: Tuple[float, float], bearing_step: float,
    step_minutes: float,
) -> Tuple[float, float, float]:
    best_tilt, best_bearing, best_energy = 0.0, 0.0, -1.0

    tilt = tilt_range[0]
    while tilt <= tilt_range[1] + 1e-9:
        bearing = bearing_range[0]
        while bearing < bearing_range[1] - 1e-9:
            energy = collected_energy_wh_per_m2(
                lat_deg, lon_deg, tz_offset_hours, start, duration_hours,
                tilt, bearing % 360.0, step_minutes=step_minutes,
            )
            if energy > best_energy:
                best_energy = energy
                best_tilt = tilt
                best_bearing = bearing % 360.0
            bearing += bearing_step
        tilt += tilt_step

    return best_tilt, best_bearing, best_energy


@dataclass
class OrientationResult:
    optimal_tilt_deg: float
    optimal_bearing_deg: float
    optimal_bearing_compass: str
    collected_energy_wh_per_m2: float
    collected_energy_wh: float
    average_power_w: float
    slope_angle_deg: float
    mechanical_tilt_adjustment_deg: float
    sunrise_local: Optional[datetime] = None
    sunset_local: Optional[datetime] = None
    warnings: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "Ideal panel orientation for this charging stop:",
            f"  Panel tilt from horizontal : {self.optimal_tilt_deg:6.1f} deg",
            f"  Car bearing (compass)      : {self.optimal_bearing_deg:6.1f} deg "
            f"({self.optimal_bearing_compass})",
            f"  Mechanical tilt adjustment : {self.mechanical_tilt_adjustment_deg:+6.1f} deg "
            f"(on top of the {self.slope_angle_deg:+.1f} deg the road slope already gives you)",
            "",
            f"  Estimated insolation       : {self.collected_energy_wh_per_m2:8.1f} Wh/m^2 over the stop",
            f"  Estimated array energy     : {self.collected_energy_wh:8.1f} Wh",
            f"  Estimated average power    : {self.average_power_w:8.1f} W",
        ]
        if self.sunrise_local and self.sunset_local:
            lines.append(
                f"  Approx. sunrise / sunset   : "
                f"{self.sunrise_local.strftime('%H:%M')} / {self.sunset_local.strftime('%H:%M')} local"
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def recommend_mechanical_adjustment(optimal_tilt_deg: float, gradient: float) -> float:
    """
    Estimate how much *extra* mechanical tilt (beyond what the road slope
    already provides for free) is needed to reach `optimal_tilt_deg`,
    assuming the car is parked nose-on to the optimal bearing and the road
    gradient is measured along that same direction (positive gradient =
    the ground rises in the direction the car's nose/panels are optimally
    facing). This is a simplifying, first-order approximation: it treats
    the road-slope-induced tilt and any adjustable panel tilt as acting
    about the same (fore-aft) axis, so they simply add/subtract.
    """
    slope_deg = math.degrees(math.atan(gradient))
    return optimal_tilt_deg - slope_deg


def optimize_orientation(
    lat: float,
    lon: float,
    start: datetime,
    duration_hours: float,
    gradient: float = 0.0,
    tz_offset: Optional[float] = None,
    panel_area_m2: float = DEFAULT_PANEL_AREA_M2,
    panel_efficiency: float = DEFAULT_PANEL_EFFICIENCY,
    max_tilt_deg: float = 90.0,
) -> OrientationResult:
    """
    Find the fixed panel tilt and car bearing that maximise total solar
    energy collected over [start, start + duration_hours] at the given
    location, using the hand-calculation solar-irradiance method from
    Elshafei et al. (2016), "Solar Car Optimization For the World Solar
    Challenge".

    Parameters
    ----------
    lat, lon : latitude/longitude in degrees (longitude positive East).
    start : local clock datetime the charging stop begins.
    duration_hours : how long the stop lasts, in hours.
    gradient : longitudinal road gradient (rise/run, e.g. 0.02 for 2%).
               Only used to report a suggested mechanical-tilt adjustment;
               it does not change the reported optimal tilt/bearing
               themselves (see `recommend_mechanical_adjustment`).
    tz_offset : UTC offset in hours (East positive). If omitted, it is
               approximated from longitude (lon / 15).
    panel_area_m2, panel_efficiency : used only to convert Wh/m^2 into an
               estimated total array energy and average power.
    max_tilt_deg : upper bound searched for panel tilt (default 90).
    """
    if tz_offset is None:
        tz_offset = lon / 15.0

    warnings = []
    if duration_hours <= 0:
        warnings.append("duration_hours <= 0; no energy can be collected.")

    # Stage 1: coarse grid search over the whole (tilt, bearing) space.
    tilt0, bearing0, _ = _grid_search(
        lat, lon, tz_offset, start, duration_hours,
        tilt_range=(0.0, max_tilt_deg), tilt_step=5.0,
        bearing_range=(0.0, 360.0), bearing_step=5.0,
        step_minutes=10.0,
    )

    # Stage 2: local refinement around the coarse optimum, finer time step.
    tilt_lo, tilt_hi = max(0.0, tilt0 - 6.0), min(max_tilt_deg, tilt0 + 6.0)
    bearing_lo, bearing_hi = bearing0 - 6.0, bearing0 + 6.0
    tilt1, bearing1, energy1 = _grid_search(
        lat, lon, tz_offset, start, duration_hours,
        tilt_range=(tilt_lo, tilt_hi), tilt_step=0.5,
        bearing_range=(bearing_lo, bearing_hi), bearing_step=0.5,
        step_minutes=2.0,
    )
    bearing1 %= 360.0

    if energy1 <= 0.0:
        warnings.append(
            "No usable direct sunlight was found in this window "
            "(check the stop time against local sunrise/sunset)."
        )

    slope_deg = math.degrees(math.atan(gradient))
    mech_adjust = recommend_mechanical_adjustment(tilt1, gradient)

    energy_wh = energy1 * panel_area_m2 * panel_efficiency
    avg_power_w = energy_wh / duration_hours if duration_hours > 0 else 0.0

    try:
        sunrise, sunset = approx_sunrise_sunset_clock(start, lat, lon, tz_offset)
    except Exception:
        sunrise, sunset = None, None

    return OrientationResult(
        optimal_tilt_deg=tilt1,
        optimal_bearing_deg=bearing1,
        optimal_bearing_compass=bearing_to_compass_label(bearing1),
        collected_energy_wh_per_m2=energy1,
        collected_energy_wh=energy_wh,
        average_power_w=avg_power_w,
        slope_angle_deg=slope_deg,
        mechanical_tilt_adjustment_deg=mech_adjust,
        sunrise_local=sunrise,
        sunset_local=sunset,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Find the ideal fixed solar-panel tilt and car bearing to "
            "maximise collected solar energy during a charging stop, using "
            "the hand-calculation solar-geometry method from Elshafei et "
            "al. (2016), 'Solar Car Optimization For the World Solar "
            "Challenge'."
        )
    )
    p.add_argument("--lat", type=float, required=True, help="Latitude in degrees (South negative).")
    p.add_argument("--lon", type=float, required=True, help="Longitude in degrees (West negative).")
    p.add_argument(
        "--start", type=str, required=True,
        help="Local clock start time of the charge, ISO format e.g. 2026-07-28T09:00:00",
    )
    p.add_argument(
        "--duration", type=float, required=True,
        help="Expected duration of the charging stop, in hours.",
    )
    p.add_argument(
        "--gradient", type=float, default=0.0,
        help="Longitudinal road gradient (rise/run), e.g. 0.02 for a 2%% uphill slope. Default 0.",
    )
    p.add_argument(
        "--tz", type=float, default=None,
        help="UTC offset in hours (East positive), e.g. 9.5 for Darwin. "
        "Defaults to an estimate from longitude if omitted.",
    )
    p.add_argument("--panel-area", type=float, default=DEFAULT_PANEL_AREA_M2, help="Panel area in m^2.")
    p.add_argument(
        "--panel-efficiency", type=float, default=DEFAULT_PANEL_EFFICIENCY,
        help="Panel conversion efficiency (0-1).",
    )
    p.add_argument("--max-tilt", type=float, default=90.0, help="Maximum searchable tilt angle, degrees.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    start_dt = datetime.fromisoformat(args.start)

    result = optimize_orientation(
        lat=args.lat,
        lon=args.lon,
        start=start_dt,
        duration_hours=args.duration,
        gradient=args.gradient,
        tz_offset=args.tz,
        panel_area_m2=args.panel_area,
        panel_efficiency=args.panel_efficiency,
        max_tilt_deg=args.max_tilt,
    )
    print(result)


if __name__ == "__main__":
    main()
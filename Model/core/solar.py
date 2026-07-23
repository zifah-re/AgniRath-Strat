"""
core/solar.py — solar pipeline behind ONE frozen signature (workplan 1.2;
Plan v3 §6.1).

FROZEN INTERFACE (contract, README §Interfaces):
    provider.ghi_wm2(t_s: float, x_m: float) -> float
        t_s : seconds since local midnight of the race day
        x_m : distance along the day's route (metres, from day start line)
    All layers (L1/L2/L3) consume solar ONLY through this call. Providers:

      GaussianProvider      seniors' fallback day-shape (race_completion/
                            solar.py verbatim: 1073.099*exp(-0.5*((t-43200)/
                            11600)^2)).
      PVLibClearSkyProvider pvlib clear-sky fallback (import-guarded).
      SolcastCurveFitProvider
                            Approach 1: pre-queried Solcast, per-circle
                            polynomial fits, zero API calls inside solves.

PMF correction (Approach 2, Paper 1 Table 5) is applied as a multiplicative
factor via pmf_correction_factor(tcc_pct, t_s).

Incidence geometry (Plan v3 §6.1 slope-dependent solar): Paper 4 (King Fahd)
equations 3-14 — declination, equation of time, hour angle, cos(theta_i),
R_b geometric factor for a panel tilted by (base tilt + road slope) with
azimuth ~= route bearing.
"""

from __future__ import annotations

import json
import math
import pathlib
import typing as _t

import numpy as np

# ---------------------------------------------------------------------------
# PMF correction — Paper 1 (TUT SSC2018) Table 5, verbatim expected values.
# Normalised 0.5-1.5; 1.0 = perfect prediction. Categories on daily-mean TCC.
# ---------------------------------------------------------------------------
PMF_TABLE = {
    # (tcc_lo_pct, tcc_hi_pct, window): expected correction factor
    ("Ia", 0.0, 2.0, "offpeak"): 0.9393,    # 08:00-10:00 & 14:00-17:00
    ("Ib", 0.0, 2.0, "midday"): 0.9532,     # 10:00-14:00
    ("IIa", 3.0, 10.0, "allday"): 1.1013,
    ("IIb", 11.0, 20.0, "allday"): 0.9965,
    ("IIc", 21.0, 40.0, "allday"): 0.9226,
    ("IId", 41.0, 100.0, "allday"): 1.1019,
}


def pmf_correction_factor(tcc_daily_mean_pct: float, t_s: float) -> float:
    """Expected forecast->actual GHI correction (Paper 1 Table 5).

    NOTE these factors were measured for Meteomatics forecasts over Pretoria
    (May-Aug 2018). They are the STARTING values; pipeline/update_pmf.py
    replaces them with our own Solcast-vs-actual factors as collection
    accumulates (Plan v3 §6.1 Approach 2).
    Interpretation: actual ~= forecast * factor... factors here are
    normalised expected values of actual/forecast ratio PMFs.
    """
    tcc = float(tcc_daily_mean_pct)
    hour = (t_s / 3600.0)
    if tcc <= 2.0:
        return PMF_TABLE[("Ib", 0.0, 2.0, "midday")] if 10.0 <= hour < 14.0 \
            else PMF_TABLE[("Ia", 0.0, 2.0, "offpeak")]
    if tcc <= 10.0:
        return PMF_TABLE[("IIa", 3.0, 10.0, "allday")]
    if tcc <= 20.0:
        return PMF_TABLE[("IIb", 11.0, 20.0, "allday")]
    if tcc <= 40.0:
        return PMF_TABLE[("IIc", 21.0, 40.0, "allday")]
    return PMF_TABLE[("IId", 41.0, 100.0, "allday")]


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class SolarProvider:
    """Frozen interface: ghi_wm2(t_s, x_m)."""

    def ghi_wm2(self, t_s: float, x_m: float) -> float:  # pragma: no cover
        raise NotImplementedError


class GaussianProvider(SolarProvider):
    """Seniors' Gaussian day-shape (race_completion/solar.py, verbatim).

    Peak 1073.099 W/m^2 at solar noon (43200 s), sigma 11600 s. Position-
    independent — this is the always-installed fallback (Plan v3 §6.1).
    """

    PEAK_WM2 = 1073.099
    NOON_S = 43_200.0
    SIGMA_S = 11_600.0

    def ghi_wm2(self, t_s: float, x_m: float = 0.0) -> float:
        return float(self.PEAK_WM2
                     * math.exp(-0.5 * ((t_s - self.NOON_S) / self.SIGMA_S) ** 2))


class PVLibClearSkyProvider(SolarProvider):
    """pvlib clear-sky GHI at a fixed reference location (fallback tier 2).

    Import-guarded: if pvlib is unavailable, constructing this raises and
    callers fall back to GaussianProvider. Position handling at circle
    granularity arrives with the route pipeline (block 2/4).
    """

    def __init__(self, lat: float, lon: float, date_iso: str,
                 tz: str = "Africa/Johannesburg"):
        import pandas as pd
        import pvlib  # noqa: F401  (raises ImportError if absent)
        self._pvlib = pvlib
        self._times = pd.date_range(f"{date_iso} 00:00", f"{date_iso} 23:59",
                                    freq="1min", tz=tz)
        loc = pvlib.location.Location(lat, lon, tz=tz)
        self._ghi = loc.get_clearsky(self._times, model="ineichen")["ghi"].values

    def ghi_wm2(self, t_s: float, x_m: float = 0.0) -> float:
        idx = int(np.clip(t_s // 60, 0, len(self._ghi) - 1))
        return float(self._ghi[idx])


class SolcastCurveFitProvider(SolarProvider):
    """Approach 1 (Plan v3 §6.1): per-circle polynomial GHI(t) fits.

    Loads a JSON produced by the pre-query + fit step:
      {"circles": [{"id": 0, "coeffs": [c0, c1, ...]}, ...]}
    where GHI(t) = polyval(coeffs, t_hours_since_midnight), and a mapping
    x_m -> circle id supplied by the route file (route.circle_id_at(x)).
    Zero API calls at solve time by construction.
    """

    def __init__(self, fits_json: str | pathlib.Path,
                 circle_id_at: _t.Callable[[float], int]):
        data = json.loads(pathlib.Path(fits_json).read_text())
        self._coeffs = {c["id"]: np.asarray(c["coeffs"], dtype=float)
                        for c in data["circles"]}
        self._circle_id_at = circle_id_at

    def ghi_wm2(self, t_s: float, x_m: float) -> float:
        cid = self._circle_id_at(x_m)
        c = self._coeffs[cid]
        return float(max(0.0, np.polyval(c, t_s / 3600.0)))


def best_available_provider(**pvlib_kwargs) -> SolarProvider:
    """Fallback chain: Solcast fits (if configured) -> pvlib -> Gaussian."""
    try:
        return PVLibClearSkyProvider(**pvlib_kwargs)
    except Exception:
        return GaussianProvider()


# ---------------------------------------------------------------------------
# Incidence geometry — Paper 4 (King Fahd) Eqs 3-14 + Plan v3 §6.1
# slope-dependent panel tilt. All angles degrees unless noted.
# ---------------------------------------------------------------------------

def solar_declination_deg(day_of_year: int) -> float:
    """delta = 23.45 sin(360 (284+N)/365)   (Paper 4 Eq. 5)."""
    return 23.45 * math.sin(math.radians(360.0 * (284 + day_of_year) / 365.0))


def equation_of_time_min(day_of_year: int) -> float:
    """E in minutes (Paper 4 Eq. 3)."""
    b = math.radians((day_of_year - 1) * 360.0 / 365.0)
    return 229.2 * (0.000075 + 0.001868 * math.cos(b) - 0.032077 * math.sin(b)
                    - 0.014615 * math.cos(2 * b) - 0.04089 * math.sin(2 * b))


def hour_angle_deg(t_solar_s: float) -> float:
    """omega: 15 deg/hour from solar noon, morning negative (Paper 4 Eq. 4).

    t_solar_s = SOLAR local time in seconds since midnight. Callers must
    apply the longitude + equation-of-time correction (Paper 4 Eq. ~2-3)
    before passing in; helper solar_time_s below does this.
    """
    return (t_solar_s / 3600.0 - 12.0) * 15.0


def solar_time_s(t_clock_s: float, lon_deg: float, day_of_year: int,
                 tz_hours: float = 2.0) -> float:
    """Clock time -> solar time (Paper 4: Ts = std + 4*(Lst - Lng) + E).

    Sign convention adopted: standard meridian Lst = 15 deg * tz_hours
    (=30 deg E for CAT); correction minutes = 4*(lon - Lst) + E so that a
    location EAST of the standard meridian reaches solar noon EARLIER.
    FLAGGED: Paper 4's printed form (Lst - Lng) is for west-longitude
    conventions; South Africa uses east-positive longitudes, hence the sign
    here. Verify once against pvlib solar position on a known case (the
    pipeline's cross-reference task, Plan v3 §6.1).
    """
    lst = 15.0 * tz_hours
    corr_min = 4.0 * (lon_deg - lst) + equation_of_time_min(day_of_year)
    return t_clock_s + corr_min * 60.0


def cos_incidence(
    lat_deg: float, decl_deg: float, omega_deg: float,
    tilt_deg: float, azimuth_deg: float,
) -> float:
    """cos(theta_i) between beam radiation and panel normal (Paper 4 Eq. 12).

    azimuth (gamma): panel surface azimuth, 0 = facing equator (NORTH in the
    southern hemisphere), positive toward west — Duffie-Beckman convention.
    """
    phi = math.radians(lat_deg)
    d = math.radians(decl_deg)
    w = math.radians(omega_deg)
    b = math.radians(tilt_deg)
    g = math.radians(azimuth_deg)
    return (
        math.sin(phi) * math.sin(d) * math.cos(b)
        - math.cos(phi) * math.sin(d) * math.sin(b) * math.cos(g)
        + math.cos(phi) * math.cos(d) * math.cos(w) * math.cos(b)
        + math.sin(phi) * math.cos(d) * math.cos(w) * math.sin(b) * math.cos(g)
        + math.cos(d) * math.sin(w) * math.sin(b) * math.sin(g)
    )


def cos_zenith(lat_deg: float, decl_deg: float, omega_deg: float) -> float:
    """cos(theta_z) (Paper 4, horizontal-surface insolation term)."""
    phi = math.radians(lat_deg)
    d = math.radians(decl_deg)
    w = math.radians(omega_deg)
    return math.cos(phi) * math.cos(d) * math.cos(w) + math.sin(phi) * math.sin(d)


def slope_geometry_factor(
    lat_deg: float, day_of_year: int, t_solar_s: float,
    road_slope_pct: float, route_bearing_deg: float,
    panel_tilt_base_deg: float = 4.0,
) -> float:
    """R_b = cos(theta_i)/cos(theta_z) for the gradient-tilted panel
    (Paper 4 Eqs 12-14; Plan v3 §6.1; Aryaman's notes suggestion).

    Effective panel tilt = base tilt + road pitch (atan of slope). Panel
    azimuth: the panel pitches about the car's lateral axis, so its normal
    leans along the direction of travel — uphill leans the normal backward
    (toward -bearing), downhill forward (+bearing). Convert compass bearing
    (0=N, 90=E) to surface azimuth (0=N equator-facing in SH, +ve toward W):
    gamma = -bearing (mod 360, mapped to [-180, 180]).
    FLAGGED: convention chain (bearing->gamma sign, SH equator-facing) is
    easy to get subtly wrong; validated in tests only for symmetric cases.
    Cross-check against pvlib GTI on a known segment before race use.
    Clamped to [0, ~2] and to 0 when the sun is below the horizon.
    """
    decl = solar_declination_deg(day_of_year)
    omega = hour_angle_deg(t_solar_s)
    cz = cos_zenith(lat_deg, decl, omega)
    if cz <= 0.0:
        return 0.0                       # sun below horizon
    road_pitch_deg = math.degrees(math.atan(road_slope_pct / 100.0))
    tilt = panel_tilt_base_deg + road_pitch_deg
    sign = 1.0 if tilt >= 0 else -1.0
    tilt_abs = abs(tilt)
    gamma = -route_bearing_deg * sign
    gamma = ((gamma + 180.0) % 360.0) - 180.0
    ci = cos_incidence(lat_deg, decl, omega, tilt_abs, gamma)
    return float(np.clip(ci / cz, 0.0, 2.0))

"""
core/wind.py — wind pipeline behind ONE frozen signature (workplan 1.2;
Plan v3 §6.2).

FROZEN INTERFACE (contract, README §Interfaces):
    provider.wind(t_s: float, x_m: float) -> (speed_ms, dir_deg_from)
        dir_deg_from : meteorological convention, direction the wind blows
                       FROM, degrees clockwise from north.
    Decomposition to along-track + yaw is done HERE (not in physics) so all
    providers stay dumb data sources.

Along-track component (Paper 2 simplification, Plan v3 §6.2):
    w_par > 0  == TAILWIND (pushes the car along its bearing).
Yaw angle psi: angle between the RELATIVE wind vector and the car
centreline, for CdA(psi) lookup.

Providers:
    ConstantWindProvider   trivial/fallback + scenario injection.
    TableWindProvider      CSV/collected data at circle centres (Open-Meteo /
                           ERA5-class source pulled by pipeline/
                           collect_weather.py; Solcast wind was never used —
                           senior conversation, 16 Jul 2026).
"""

from __future__ import annotations

import math
import pathlib
import typing as _t

import numpy as np


class WindProvider:
    def wind(self, t_s: float, x_m: float) -> tuple[float, float]:
        """Return (speed_ms, dir_deg_from)."""  # pragma: no cover
        raise NotImplementedError


class ConstantWindProvider(WindProvider):
    def __init__(self, speed_ms: float = 0.0, dir_deg_from: float = 0.0):
        self._s = float(speed_ms)
        self._d = float(dir_deg_from)

    def wind(self, t_s: float, x_m: float) -> tuple[float, float]:
        return self._s, self._d


class TableWindProvider(WindProvider):
    """Nearest-in-time lookup from a collected CSV.

    CSV schema (written by pipeline/collect_weather.py):
        t_s,circle_id,speed_ms,dir_deg_from
    plus a circle_id_at(x_m) mapping from the route file.
    """

    def __init__(self, csv_path: str | pathlib.Path,
                 circle_id_at: _t.Callable[[float], int]):
        rows = np.genfromtxt(str(csv_path), delimiter=",", names=True)
        self._t = np.atleast_1d(rows["t_s"]).astype(float)
        self._cid = np.atleast_1d(rows["circle_id"]).astype(int)
        self._spd = np.atleast_1d(rows["speed_ms"]).astype(float)
        self._dir = np.atleast_1d(rows["dir_deg_from"]).astype(float)
        self._circle_id_at = circle_id_at

    def wind(self, t_s: float, x_m: float) -> tuple[float, float]:
        cid = self._circle_id_at(x_m)
        mask = self._cid == cid
        if not mask.any():
            return 0.0, 0.0
        ts = self._t[mask]
        i = int(np.argmin(np.abs(ts - t_s)))
        return float(self._spd[mask][i]), float(self._dir[mask][i])


# ---------------------------------------------------------------------------
# Decomposition helpers
# ---------------------------------------------------------------------------

def along_track_ms(speed_ms: float, dir_deg_from: float,
                   bearing_deg: float) -> float:
    """Along-track wind component; POSITIVE = tailwind.

    Wind FROM dir_deg_from blows TOWARD (dir+180). A wind blowing toward the
    car's bearing is a tailwind:
        w_par = speed * cos(bearing - (dir_from + 180))
    """
    to_deg = (dir_deg_from + 180.0) % 360.0
    return speed_ms * math.cos(math.radians(bearing_deg - to_deg))


def relative_wind(v_car_ms: float, speed_ms: float, dir_deg_from: float,
                  bearing_deg: float) -> tuple[float, float]:
    """Relative airflow (magnitude, yaw psi) seen by the car.

    Air velocity relative to car = wind_vector(toward) - car_velocity_vector.
    Returns (|v_rel|, psi) where psi in [0, 180]: 0 = airflow from dead
    ahead, 180 = from dead astern — the CdA(psi) table index (Plan v3 §6.2).
    """
    to_rad = math.radians((dir_deg_from + 180.0) % 360.0)
    br = math.radians(bearing_deg)
    # world-frame components (x = east, y = north)
    wx, wy = speed_ms * math.sin(to_rad), speed_ms * math.cos(to_rad)
    cx, cy = v_car_ms * math.sin(br), v_car_ms * math.cos(br)
    rx, ry = wx - cx, wy - cy                    # air motion relative to car
    mag = math.hypot(rx, ry)
    if mag < 1e-9:
        return 0.0, 0.0
    # Angle between airflow ARRIVAL direction (-r) and car heading:
    ax, ay = -rx, -ry
    hx, hy = math.sin(br), math.cos(br)
    cospsi = max(-1.0, min(1.0, (ax * hx + ay * hy) / mag))
    return mag, math.degrees(math.acos(cospsi))

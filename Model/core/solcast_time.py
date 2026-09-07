"""
core/solcast_time.py — shared time-conversion helper for the ACTUAL race-week
Solcast forecast data (data/Solar_real/mean_*.jsonl), used by both
core.solar.RealSolcastSolarProvider and core.wind.RealSolcastWindProvider.

That data timestamps every sample with "period_end", an ISO8601 string in
UTC (e.g. "2026-09-10T07:00:00+00:00"). The rest of the simulator works in
LOCAL seconds-since-midnight of the race day (Africa/Johannesburg, UTC+2,
no daylight saving), so every sample has to be converted once at load time.
"""

from __future__ import annotations

import pandas as pd

RACE_TZ = "Africa/Johannesburg"


def period_end_to_local_s(period_end_iso: str) -> float:
    """UTC ISO8601 'period_end' -> local seconds since local midnight of the
    calendar day the sample falls on in Africa/Johannesburg (UTC+2 year-round,
    so this is just "+2 hours" with no DST edge cases to worry about)."""
    ts = pd.Timestamp(period_end_iso)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(RACE_TZ)
    return float(local.hour * 3600.0 + local.minute * 60.0 + local.second)


def period_end_to_local_date(period_end_iso: str) -> str:
    """UTC ISO8601 'period_end' -> local calendar date ('YYYY-MM-DD') in
    Africa/Johannesburg. Used for POA irradiance (core.solar.poa_wm2), which
    needs a real calendar date (not just seconds-since-midnight) to compute
    solar position via pvlib."""
    ts = pd.Timestamp(period_end_iso)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(RACE_TZ)
    return local.strftime("%Y-%m-%d")
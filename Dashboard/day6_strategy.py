"""
day6_strategy.py -- AgniRath Sasol Solar Challenge, Day 6 strategy engine
==========================================================================

Day 6 is a "hard drive" / unlimited-loop concept: the car drives OUT as
far as you choose (on the "One way max distance" route, net downhill --
912m start altitude down to 38m at the far end, ~104.65 km total, with
some rolling sections including several +/-8% grades), then gets
TRAILERED back to the start (no energy modeled on the way back).

Battery: still 24 of 28 series modules, carried over from Day 5 (flag me
if this changed). 3200 Wh assumed nameplate over 28 modules -> 114.29
Wh/module -> 2742.86 Wh usable on 24 modules. Hard floor: never below
80 V pack terminal voltage.

Flow (3 arguments):
  1. arrival_time_local ("HH:MM") -- when you reach the start point.
  2. soc_at_arrival_pct -- pack SOC at that arrival time.
  3. distance_km -- how far out to drive (truncates the route to its
     first `distance_km` km, since that's the downhill portion).

  -> arrival_time + 30 min sun-tracking charge (this becomes the actual
     starting SOC for the drive out, as instructed).
  -> drive out `distance_km` km at OUTBOUND_SPEED_KMH (tunable constant
     below), cruise control: regen downhill, accelerate/hold power
     uphill, exactly like every other day's model. There are genuine
     +/-8% grades in here -- the script flags every segment over the
     8% "safe cruise" threshold so you can see exactly where they are
     before committing to a speed.
  -> immediate trailer back at TRAILER_RETURN_SPEED_KMH (65 km/h given),
     zero energy change (no driving, no charging assumed while loading/
     transporting).

Reports SOC-vs-distance and Solar-vs-distance for the OUTBOUND leg (the
only leg with real energy dynamics), plus a full time/SOC/voltage log
for the whole exercise (charge -> drive out -> trailer back), voltage
given on BOTH the healthy-28-module curve and the actual degraded
24-module curve.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------
# 0. FILE LOCATIONS (run this from inside your Dashboard/ folder)
# ---------------------------------------------------------------------
DASH = Path(".")
SAVES = DASH / "Saves"
SOLAR = DASH / "Solar"

ROUTE_FILE = SAVES / "One way max distance_One way max distance_One way max distance.kml.save"
SOLAR_FILE = SOLAR / "mean_Day 6_One way max distance_One way max distance_One way max distance.jsonl"

LOCAL_TZ = timezone(timedelta(hours=2))   # SAST, UTC+2
RACE_DATE = "2026-09-15"                  # Day 6


def local_to_utc(hh: int, mm: int) -> datetime:
    local = datetime.fromisoformat(f"{RACE_DATE}T{hh:02d}:{mm:02d}:00").replace(tzinfo=LOCAL_TZ)
    return local.astimezone(timezone.utc)


def parse_local_hhmm(hhmm: str) -> datetime:
    hh, mm = hhmm.split(":")
    return local_to_utc(int(hh), int(mm))


# =======================================================================
# 1. BATTERY MODEL -- degraded 24/28-module pack (carried over from Day 5)
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
N_MODULES_NOW = 24
PACK_WH_ASSUMED = 3200.0
WH_PER_MODULE = PACK_WH_ASSUMED / N_MODULES_HEALTHY
USABLE_WH_NOW = WH_PER_MODULE * N_MODULES_NOW

VOLTAGE_FLOOR_V = 80.0

SCALE = N_MODULES_NOW / N_MODULES_HEALTHY
_V_HEALTHY = np.array([v for v, _ in SOC_CURVE_V_PCT_HEALTHY])[::-1]
_PCT_HEALTHY = np.array([p for _, p in SOC_CURVE_V_PCT_HEALTHY])[::-1]
_V_DEG = _V_HEALTHY * SCALE
_PCT_DEG = _PCT_HEALTHY


def soc_from_voltage_degraded(v: float) -> float:
    return float(np.interp(v, _V_DEG, _PCT_DEG))


def voltage_from_soc_degraded(pct: float) -> float:
    return float(np.interp(pct, _PCT_DEG, _V_DEG))


def voltage_from_soc_healthy(pct: float) -> float:
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
    print("BATTERY: degraded 24/28-module pack (carried over from Day 5)")
    print("=" * 78)
    print(f"Assumed usable capacity: {PACK_WH_ASSUMED:.0f} Wh nameplate over {N_MODULES_HEALTHY} modules "
          f"-> {WH_PER_MODULE:.2f} Wh/module -> {USABLE_WH_NOW:.2f} Wh on {N_MODULES_NOW} modules")
    print(f"Voltage scale factor ({N_MODULES_NOW}/{N_MODULES_HEALTHY}): {SCALE:.4f}")
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
MPPT_D_SHADE_FACTOR = 0.25   # carried over from Day 3/4/5


def load_solar_series(path: Path):
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
    return dni_w_m2 * ARRAY_AREA_M2 * ARRAY_EFFICIENCY


def driving_solar_power_w(ghi_w_m2: float) -> float:
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

# ---- Tunable constants (not CLI arguments -- edit here) ----
OUTBOUND_SPEED_KMH = 60.0        # cruise speed for the drive out (tune freely)
TRAILER_RETURN_SPEED_KMH = 65.0  # as instructed -- reference only, no energy modeled
CONTROL_STOP_BUFFER_MIN = 30     # "reach there + 30 min" sun-tracking charge before driving
GRADE_WARN_THRESHOLD_PCT = 8.0   # flag any segment steeper than this


def load_route_profile():
    with open(ROUTE_FILE) as f:
        d = json.load(f)
    prof = d["profile"]
    return (np.array(prof["Distance"], dtype=float),
            np.array(prof["Gradient"], dtype=float),
            np.array(prof["Altitude"], dtype=float),
            prof["Coordinates"])


def truncate_route(distance_km: np.ndarray, gradient_pct: np.ndarray, max_km: float):
    """Cut the route profile to its first `max_km` km, interpolating the
    exact cut point so the truncated route ends at precisely max_km."""
    if max_km >= distance_km[-1]:
        return distance_km.copy(), gradient_pct.copy()
    i_cut = int(np.searchsorted(distance_km, max_km))
    if i_cut == 0:
        return distance_km[:1], gradient_pct[:1]
    d0, d1 = distance_km[i_cut - 1], distance_km[i_cut]
    g0, g1 = gradient_pct[i_cut - 1], gradient_pct[i_cut]
    f = 0.0 if d1 == d0 else (max_km - d0) / (d1 - d0)
    g_cut = g0 + f * (g1 - g0)
    new_dist = np.concatenate([distance_km[:i_cut], [max_km]])
    new_grad = np.concatenate([gradient_pct[:i_cut], [g_cut]])
    return new_dist, new_grad


def check_gradients(distance_km: np.ndarray, gradient_pct: np.ndarray, speed_kmh: float) -> None:
    bad_mask = np.abs(gradient_pct) > GRADE_WARN_THRESHOLD_PCT
    n_bad = int(bad_mask.sum())
    print(f"Gradient check on the {distance_km[-1]:.2f} km outbound leg at {speed_kmh:.0f} km/h:")
    print(f"  max grade = {gradient_pct.max():+.2f}%, min grade = {gradient_pct.min():+.2f}%")
    if n_bad > 0:
        bad_idx = np.where(bad_mask)[0]
        # group into contiguous runs for a compact report
        runs = []
        run_start = bad_idx[0]
        prev = bad_idx[0]
        for i in bad_idx[1:]:
            if i - prev > 1:
                runs.append((run_start, prev))
                run_start = i
            prev = i
        runs.append((run_start, prev))
        print(f"  ** {n_bad} points exceed +/-{GRADE_WARN_THRESHOLD_PCT:.0f}% -- "
              f"{len(runs)} distinct steep section(s):")
        for a, b in runs:
            worst = gradient_pct[a:b + 1]
            worst_val = worst[np.argmax(np.abs(worst))]
            print(f"     km {distance_km[a]:.2f}-{distance_km[b]:.2f}: worst grade {worst_val:+.2f}%")
        print("  Cruise-control physics below still computes motor/regen through these "
              "sections (accelerate to hold speed uphill, regen-brake to hold speed "
              "downhill) -- but these are genuinely steep for a solar car, worth a visual "
              "gut-check against the real road before committing to this speed.")
    else:
        print(f"  All segments within +/-{GRADE_WARN_THRESHOLD_PCT:.0f}%, no flags.")
    print()


def drive_outbound_trace(distance_km: np.ndarray, gradient_pct: np.ndarray, start_utc: datetime,
                          solar_series, speed_kmh: float, soc_start: float):
    """Per-segment outbound trace: cumulative distance / SOC / cumulative
    solar Wh, driven once from 0 to distance_km[-1]."""
    speed_ms = speed_kmh / 3.6
    seg_km = np.diff(distance_km)
    seg_grad = 0.5 * (gradient_pct[:-1] + gradient_pct[1:])
    theta = np.arctan(seg_grad / 100.0)
    f_rr = CRR * MASS_KG * G * np.cos(theta)
    f_aero = 0.5 * AIR_DENSITY * CDA_M2 * speed_ms ** 2
    f_grav = MASS_KG * G * np.sin(theta)
    f_total = f_rr + f_aero + f_grav
    p_wheel = f_total * speed_ms
    seg_time_s = (seg_km * 1000.0) / speed_ms

    cum_km_trace = [0.0]
    soc_trace = [soc_start]
    cum_solar_wh_trace = [0.0]

    t = start_utc
    soc = soc_start
    cum_km = 0.0
    cum_solar_wh = 0.0
    motor_wh_total = 0.0
    regen_wh_total = 0.0

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
        t = t_next

        cum_km_trace.append(cum_km)
        soc_trace.append(soc)
        cum_solar_wh_trace.append(cum_solar_wh)

    return {
        "cum_km_trace": cum_km_trace, "soc_trace": soc_trace, "cum_solar_wh_trace": cum_solar_wh_trace,
        "end_utc": t, "end_soc": soc, "motor_wh": motor_wh_total, "regen_wh": regen_wh_total,
        "solar_wh": cum_solar_wh, "duration_min": (t - start_utc).total_seconds() / 60.0,
    }


# =======================================================================
# 4. MAIN: charge -> drive out -> trailer back
# =======================================================================

def run_day6(arrival_time_local: str, soc_at_arrival_pct: float, distance_km: float,
             verbose: bool = True) -> dict:
    dist_full, grad_full, alt_full, coords_full = load_route_profile()

    if distance_km > dist_full[-1]:
        raise ValueError(f"distance_km={distance_km} exceeds the route's total length "
                          f"({dist_full[-1]:.2f} km)")

    dist_cut, grad_cut = truncate_route(dist_full, grad_full, distance_km)
    solar_series = load_solar_series(SOLAR_FILE)

    arrival_utc = parse_local_hhmm(arrival_time_local)
    log = []

    if verbose:
        print("=" * 78)
        print(f"DAY 6: hard drive out {distance_km:.1f} km, trailer back")
        print("=" * 78)
        print(f"Route total length available: {dist_full[-1]:.2f} km (using first {distance_km:.1f} km)")
        check_gradients(dist_cut, grad_cut, OUTBOUND_SPEED_KMH)

    # ---- +30 min sun-tracking charge at arrival, before driving ----
    charge_end = arrival_utc + timedelta(minutes=CONTROL_STOP_BUFFER_MIN)
    wh_charge = energy_over_window_wh(solar_series, arrival_utc, charge_end, stationary_charge_power_w)
    d_soc = wh_to_soc_delta(wh_charge)
    soc_start_drive = min(100.0, max(0.0, soc_at_arrival_pct + d_soc))
    log.append({"event": f"Arrival + {CONTROL_STOP_BUFFER_MIN}-min sun-tracking charge",
                "time_local": arrival_utc.astimezone(LOCAL_TZ), "wh": wh_charge,
                "soc_pct": soc_at_arrival_pct, "note": "SOC at arrival (given)"})
    log.append({"event": f"Start of drive (after {CONTROL_STOP_BUFFER_MIN}-min charge)",
                "time_local": charge_end.astimezone(LOCAL_TZ), "wh": wh_charge,
                "soc_pct": soc_start_drive, "note": f"+{wh_charge:.1f} Wh -> +{d_soc:.2f} pts"})

    # ---- drive out ----
    trace = drive_outbound_trace(dist_cut, grad_cut, charge_end, solar_series,
                                  OUTBOUND_SPEED_KMH, soc_start_drive)
    turnaround_utc = trace["end_utc"]
    soc_turnaround = trace["end_soc"]
    log.append({"event": f"Arrive turnaround ({distance_km:.1f} km @ {OUTBOUND_SPEED_KMH:.0f} km/h)",
                "time_local": turnaround_utc.astimezone(LOCAL_TZ), "wh": trace["solar_wh"] - trace["motor_wh"] + trace["regen_wh"],
                "soc_pct": soc_turnaround,
                "note": f"motor={trace['motor_wh']:.1f} Wh, regen={trace['regen_wh']:.1f} Wh recovered, "
                        f"solar={trace['solar_wh']:.1f} Wh, {trace['duration_min']:.1f} min driving"})

    # ---- trailer back (no energy modeled) ----
    trailer_transit_min = distance_km / TRAILER_RETURN_SPEED_KMH * 60.0
    return_utc = turnaround_utc + timedelta(minutes=trailer_transit_min)
    log.append({"event": f"Trailer back ({distance_km:.1f} km @ {TRAILER_RETURN_SPEED_KMH:.0f} km/h, no energy modeled)",
                "time_local": return_utc.astimezone(LOCAL_TZ), "wh": 0.0, "soc_pct": soc_turnaround,
                "note": f"{trailer_transit_min:.1f} min transit, SOC unchanged"})

    if verbose:
        print("=" * 78)
        print("EVENT LOG")
        print("=" * 78)
        print(f"{'Event':60s} {'Local time':>10s} {'Wh':>9s} {'SOC%':>7s}")
        for row in log:
            print(f"{row['event'][:60]:60s} {row['time_local']:%H:%M} {row['wh']:9.1f} {row['soc_pct']:7.2f}")
            if row.get("note"):
                print(f"    -> {row['note']}")
        print()
        print(f"SOC at start of drive:  {soc_start_drive:.2f}%  ->  "
              f"V(healthy 28mod)={voltage_from_soc_healthy(soc_start_drive):.2f} V, "
              f"V(degraded 24mod, actual)={voltage_from_soc_degraded(soc_start_drive):.2f} V")
        print(f"SOC at turnaround:      {soc_turnaround:.2f}%  ->  "
              f"V(healthy 28mod)={voltage_from_soc_healthy(soc_turnaround):.2f} V, "
              f"V(degraded 24mod, actual)={voltage_from_soc_degraded(soc_turnaround):.2f} V")
        if soc_turnaround < FLOOR_SOC_PCT:
            print(f"  ** BELOW the {VOLTAGE_FLOOR_V:.0f}V floor ({FLOOR_SOC_PCT:.2f}%) at turnaround **")
        print(f"SOC back at start (post-trailer): {soc_turnaround:.2f}%  (unchanged)")
        print()

    return {
        "distance_km": distance_km,
        "soc_start_drive": soc_start_drive,
        "soc_turnaround": soc_turnaround,
        "turnaround_local": turnaround_utc.astimezone(LOCAL_TZ),
        "return_local": return_utc.astimezone(LOCAL_TZ),
        "log": log,
        "cum_km_trace": trace["cum_km_trace"],
        "soc_trace": trace["soc_trace"],
        "cum_solar_wh_trace": trace["cum_solar_wh_trace"],
    }


# =======================================================================
# 5. CLI
# =======================================================================

if __name__ == "__main__":
    import sys

    print_battery_summary()

    if len(sys.argv) < 4:
        print("Usage: python day6_strategy.py <arrival_time HH:MM> <soc_at_arrival_pct> <distance_km>")
        sys.exit(1)

    arrival_time_arg = sys.argv[1]
    soc_arg = float(sys.argv[2])
    distance_arg = float(sys.argv[3])

    result = run_day6(arrival_time_arg, soc_arg, distance_arg)

    print("=" * 78)
    print(f"SOC-vs-distance and Solar-vs-distance trace (outbound leg, {distance_arg:.1f} km)")
    print("=" * 78)
    print(f"{'dist_km':>10s} {'SOC%':>7s} {'cum_solar_Wh':>13s}")
    km = result["cum_km_trace"]
    soc_t = result["soc_trace"]
    sol_t = result["cum_solar_wh_trace"]
    n_pts = len(km)
    stride = max(1, n_pts // 60)
    for i in range(0, n_pts, stride):
        print(f"{km[i]:10.2f} {soc_t[i]:7.2f} {sol_t[i]:13.1f}")
    if (n_pts - 1) % stride != 0:
        print(f"{km[-1]:10.2f} {soc_t[-1]:7.2f} {sol_t[-1]:13.1f}")
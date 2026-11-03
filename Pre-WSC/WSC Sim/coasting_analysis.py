"""Coasting analysis for the 2027 BWSC route (Pre-WSC/WSC Sim).

QUESTION ANSWERED
    Where along the stitched route can the car travel with ZERO propulsion
    energy -- i.e. gain kinetic energy on a descent and spend it on the climb
    that follows -- while obeying every speed rule used by optimizer.py?

WHAT "COASTING" MEANS HERE
    Motor torque = 0 (no traction draw, no regen).  Only drag, rolling
    resistance and gravity act, using exactly the force model of
    optimizer.electrical_power():
        F_drag = 0.5 * rho * CdA * v^2
        F_roll = m g Crr (1 - grade^2 / 2)
        F_grav = m g grade
    With u = v^2 this is linear per route step, du/dx = -a u - c, so each step
    is integrated EXACTLY (no Euler error):
        u_next = u* + (u - u*) exp(-a dx),   a = 2k/m,  u* = -(F_roll+F_grav)/k

CONSTRAINTS ENFORCED AT EVERY STEP (taken from optimizer.py when importable)
    * 60 km/h <= v <= 110 km/h   (SPEED_MIN_KMH / SPEED_MAX_KMH, Reg 3.29.1)
    * control stops and driver changes are hard barriers (the car is stationary
      there, so a coast cannot run through them)
    * route is cut at RACE_FINISH_KM
    Motor power is zero while coasting, so MAX_MOTOR_POWER_W cannot be violated.

ZONE DEFINITIONS (for each cruise speed v_c from the 5 x 600 km speed bands)
    Energy-neutral coast (green):
        start at v_c at a point where the terrain accelerates the car, release
        the throttle, and coast until speed has fallen back to v_c.  Kinetic
        energy ends equal to where it started, so propulsion energy over the
        stretch is zero with no speed penalty.
    Extended coast (orange):
        keep coasting after speed returns to v_c, running the car down toward
        60 km/h.  Still zero propulsion, but this borrows kinetic energy that
        must be re-bought later (reported in the console summary).

ROUTE INPUT
    Each day's .kml.save is read individually and stitched in day order; the
    stitched route is then resampled on a fine uniform grid (GRID_STEP_M)
    instead of optimizer.py's 10 km grid.  Per-day files are auto-discovered
    (any "*Day <n>*.kml.save" next to this script, master file excluded); set
    DAY_ROUTE_FILES to override.  If none are found it falls back to the master
    route file and says so.

USAGE
    python coasting_analysis.py
    or:  fig, result = run_coasting_analysis(cruise_speeds_kmh=(65, 70, 70, 65, 60))
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ============================ CONFIGURATION ================================
HERE = Path(__file__).resolve().parent
ROUTE_DIR = HERE
MASTER_ROUTE_NAME = "WSC_WSC Route_WSC Route.kml.save"
DAY_ROUTE_FILES: list[str | Path] = []      # explicit per-day files, in day order

GRID_STEP_M = 25.0            # uniform resampling of the stitched route
GRADE_SMOOTH_M = 0.0          # optional moving-average smoothing of gradient (0 = off)
MAX_COAST_KM = 40.0           # safety cap on a single coast
CRUISE_SPEEDS_KMH = (65.0, 65.0, 65.0, 65.0, 65.0)   # per 600 km band (as energy_balance.py)
SPEED_BAND_KM = 600.0
OUTPUT_DIR = HERE
SHOW_PLOTS = True

# ---- constants: single source of truth is optimizer.py when it can be imported
try:
    from optimizer import (  # type: ignore
        MASS_KG, G_MS2, CRR, CDA_M2, AIR_DENSITY, SPEED_MIN_KMH, SPEED_MAX_KMH,
        RACE_FINISH_KM, CONTROL_STOPS_KM, DRIVER_CHANGES_KM, MOTOR_EFF, REGEN_EFF,
    )
    CONSTANTS_SOURCE = "optimizer.py"
except Exception:  # standalone fallback: verbatim copy of the optimizer values
    MASS_KG, G_MS2, CRR, CDA_M2, AIR_DENSITY = 320.0, 9.81, 0.007, 0.16, 1.2
    SPEED_MIN_KMH, SPEED_MAX_KMH = 60.0, 110.0
    RACE_FINISH_KM = 3022.0
    CONTROL_STOPS_KM = {
        "Katherine": 318.0, "Dunmarra": 635.0, "Tennant Creek": 988.0,
        "Barrow Creek": 1210.0, "Alice Springs": 1493.0, "Erldunda": 1693.0,
        "Coober Pedy": 2179.0, "Glendambo": 2432.0, "Port Augusta": 2719.0,
    }
    DRIVER_CHANGES_KM = [160.0, 475.0, 810.0, 1100.0, 1350.0, 1850.0, 2020.0,
                         2300.0, 2580.0, 2870.0]
    MOTOR_EFF, REGEN_EFF = 0.95, 0.70
    CONSTANTS_SOURCE = "built-in copy of optimizer.py values"

K_DRAG = 0.5 * AIR_DENSITY * CDA_M2

REASON_FLOOR, REASON_OVERSPEED, REASON_END, REASON_BARRIER, REASON_MAXDIST = 1, 2, 3, 4, 5


# ============================== DATA CLASSES ================================
@dataclass
class StitchedRoute:
    x_km: np.ndarray                 # uniform grid, km
    altitude_m: np.ndarray
    gradient_pct: np.ndarray
    step_m: float
    section_starts_km: list[float]   # where each day's file begins
    section_names: list[str]
    n_native_points: int


@dataclass
class CoastZone:
    start_km: float
    end_km: float                    # end of energy-neutral coast
    ext_end_km: float                # end of extended coast (to the 60 km/h floor)
    peak_kmh: float
    cruise_kmh: float
    trace_x_km: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    trace_v_kmh: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def length_km(self) -> float:
        return self.end_km - self.start_km


@dataclass
class CoastResult:
    route: StitchedRoute
    cruise_kmh: np.ndarray           # per grid point
    neutral: np.ndarray              # bool per segment (N-1)
    extended: np.ndarray             # bool per segment, EXCLUDING neutral
    zones: list[CoastZone]
    stats: dict
    v_min_kmh: float
    v_max_kmh: float


# ============================== ROUTE LOADING ===============================
def _aligned(values, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"Route profile has no {name} values")
    if arr.size < n:
        return np.pad(arr, (0, n - arr.size), mode="edge")
    return arr[:n]


def discover_day_files(folder: Path = ROUTE_DIR) -> list[Path]:
    found = []
    for p in folder.glob("*.kml.save"):
        if p.name == MASTER_ROUTE_NAME:
            continue
        m = re.search(r"Day\s*(\d+)", p.name, re.IGNORECASE)
        if m:
            found.append((int(m.group(1)), p.name, p))
    return [p for _, _, p in sorted(found)]


def _read_profile(path: Path):
    prof = json.loads(Path(path).read_text(encoding="utf-8"))["profile"]
    d = np.asarray(prof["Distance"], dtype=float)
    n = len(d)
    coords = np.asarray(prof["Coordinates"], dtype=float)[:n]
    n = min(n, len(coords))
    d, coords = d[:n], coords[:n]
    alt = _aligned(prof.get("Altitude", [0.0]), n, "Altitude")
    grad = _aligned(prof["Gradient"], n, "Gradient")
    return d, coords, alt, grad


def load_stitched_route(day_files=None, step_m: float = GRID_STEP_M,
                        smooth_m: float = GRADE_SMOOTH_M) -> StitchedRoute:
    files = [Path(f) for f in (day_files or DAY_ROUTE_FILES)] or discover_day_files()
    if not files:
        master = ROUTE_DIR / MASTER_ROUTE_NAME
        print(f"[!] No per-day *.kml.save files found in {ROUTE_DIR}; "
              f"falling back to master route {master.name}")
        files = [master]
    else:
        print(f"[*] Stitching {len(files)} per-day route file(s):")
        for f in files:
            print(f"      {f.name}")

    D, ALT, GRAD, starts, names = [], [], [], [], []
    offset, prev_last = 0.0, None
    for f in files:
        d, coords, alt, grad = _read_profile(f)
        d = d - d[0] + offset
        starts.append(float(offset))
        names.append(f.stem.replace(".kml", ""))
        if prev_last is not None:
            gap_deg = float(np.hypot(*(coords[0] - prev_last)))
            if gap_deg > 0.005:
                print(f"[!] {f.name}: starts {gap_deg:.4f} deg from the previous day's "
                      f"end -- check file order / continuity")
            if gap_deg < 0.002:            # duplicated junction point
                d, coords, alt, grad = d[1:], coords[1:], alt[1:], grad[1:]
        D.append(d); ALT.append(alt); GRAD.append(grad)
        prev_last, offset = coords[-1], float(d[-1])

    D, ALT, GRAD = np.concatenate(D), np.concatenate(ALT), np.concatenate(GRAD)
    keep = np.r_[True, np.diff(D) > 1e-9]
    D, ALT, GRAD = D[keep], ALT[keep], GRAD[keep]

    total = float(D[-1])
    if abs(total - RACE_FINISH_KM) > 1.0:
        print(f"[!] Stitched route is {total:.2f} km; optimizer.py expects {RACE_FINISH_KM:.2f} km")
    length = min(total, RACE_FINISH_KM)

    n_pts = int(np.floor(length * 1000.0 / step_m))
    x = np.arange(n_pts + 1) * step_m / 1000.0
    grad_g = np.interp(x, D, GRAD)
    if smooth_m > 0:
        w = max(1, int(round(smooth_m / step_m)))
        w += (w % 2 == 0)
        padded = np.pad(grad_g, (w // 2, w // 2), mode="edge")
        grad_g = np.convolve(padded, np.ones(w) / w, mode="valid")
    print(f"[*] Stitched {total:,.2f} km from {len(D):,} native points -> "
          f"{len(x):,} grid points @ {step_m:g} m")
    return StitchedRoute(x, np.interp(x, D, ALT), grad_g, step_m, starts, names, len(D))


# ================================ PHYSICS ===================================
def cruise_profile_kmh(x_km: np.ndarray, speeds_kmh) -> np.ndarray:
    speeds = np.atleast_1d(np.asarray(speeds_kmh, dtype=float))
    idx = np.minimum((x_km // SPEED_BAND_KM).astype(int), len(speeds) - 1)
    return speeds[idx]


def _segment_force_terms(gradient_pct: np.ndarray):
    """Per-segment (gradient taken as the mean of its endpoints) rolling+gravity force."""
    g = 0.5 * (gradient_pct[:-1] + gradient_pct[1:]) / 100.0
    f_rg = MASS_KG * G_MS2 * (CRR * (1.0 - g ** 2 / 2.0) + g)   # optimizer.electrical_power
    return g, f_rg


def _barrier_mask(route: StitchedRoute) -> np.ndarray:
    barrier = np.zeros(len(route.x_km), dtype=bool)
    stops = list(CONTROL_STOPS_KM.values()) + list(DRIVER_CHANGES_KM)
    for km in stops:
        i = int(round(km * 1000.0 / route.step_m))
        if 0 <= i < len(barrier):
            barrier[i] = True
    return barrier


def _runs(mask: np.ndarray):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# ================================ ANALYSIS ==================================
def analyse_coasting(route: StitchedRoute, cruise_speeds_kmh=CRUISE_SPEEDS_KMH,
                     v_min_kmh: float = SPEED_MIN_KMH, v_max_kmh: float = SPEED_MAX_KMH,
                     max_coast_km: float = MAX_COAST_KM) -> CoastResult:
    speeds = np.atleast_1d(np.asarray(cruise_speeds_kmh, dtype=float))
    if np.any(speeds < v_min_kmh) or np.any(speeds > v_max_kmh):
        raise ValueError(f"Cruise speeds must lie within {v_min_kmh:g}-{v_max_kmh:g} km/h")

    x, dx = route.x_km, route.step_m
    N = len(x)
    cruise_kmh = cruise_profile_kmh(x, speeds)
    vc = cruise_kmh / 3.6
    vmin, vmax = v_min_kmh / 3.6, v_max_kmh / 3.6

    g_seg, f_rg = _segment_force_terms(route.gradient_pct)
    a = 2.0 * K_DRAG / MASS_KG
    decay = np.exp(-a * dx)
    ustar = -f_rg / K_DRAG                       # equilibrium v^2 (>0 on steep descents)
    barrier = _barrier_mask(route)

    # candidate starts: at cruise speed, terrain accelerates the car on the first step
    u0_all = vc[:-1] ** 2
    u1_all = ustar + (u0_all - ustar) * decay
    accel = (u1_all > u0_all * (1.0 + 1e-9)) & ~barrier[:-1]
    over_first = accel & (np.sqrt(np.maximum(u1_all, 0.0)) > vmax)
    cand = accel & ~over_first
    s = np.flatnonzero(cand)
    n = s.size

    e1 = np.full(n, -1, dtype=np.int64)          # index where speed returns to cruise
    e2 = np.full(n, -1, dtype=np.int64)          # last valid index of the whole coast
    reason = np.zeros(n, dtype=np.int8)
    peak = np.full(n, 0.0)

    ids, cur, uu, vcs = np.arange(n), s.copy(), u0_all[s].copy(), vc[s]
    for _ in range(int(max_coast_km * 1000.0 / dx)):
        if ids.size == 0:
            break
        end = cur >= N - 1
        if end.any():
            e2[ids[end]], reason[ids[end]] = cur[end], REASON_END
            k = ~end
            ids, cur, uu, vcs = ids[k], cur[k], uu[k], vcs[k]
            if ids.size == 0:
                break
        un = ustar[cur] + (uu - ustar[cur]) * decay
        v = np.sqrt(np.maximum(un, 0.0))
        hi = v > vmax + 1e-9
        lo = (un <= 0.0) | (v < vmin - 1e-9)
        bad = hi | lo
        if bad.any():
            e2[ids[bad]] = cur[bad]
            reason[ids[bad]] = np.where(hi[bad], REASON_OVERSPEED, REASON_FLOOR)
            k = ~bad
            ids, cur, uu, vcs, un, v = ids[k], cur[k], uu[k], vcs[k], un[k], v[k]
            if ids.size == 0:
                break
        cur = cur + 1
        uu = un
        peak[ids] = np.maximum(peak[ids], v)
        ret = (v <= vcs + 1e-9) & (e1[ids] < 0)
        e1[ids[ret]] = cur[ret]
        hit = barrier[cur]
        if hit.any():
            e2[ids[hit]], reason[ids[hit]] = cur[hit], REASON_BARRIER
            k = ~hit
            ids, cur, uu, vcs = ids[k], cur[k], uu[k], vcs[k]
    if ids.size:
        e2[ids], reason[ids] = cur, REASON_MAXDIST

    valid = e1 >= 0
    sv, e1v, e2v = s[valid], e1[valid], e2[valid]

    d_neutral = np.zeros(N + 1)
    np.add.at(d_neutral, sv, 1.0)
    np.add.at(d_neutral, e1v, -1.0)
    neutral = np.cumsum(d_neutral)[: N - 1] > 0
    d_ext = np.zeros(N + 1)
    np.add.at(d_ext, e1v, 1.0)
    np.add.at(d_ext, e2v, -1.0)
    ext_all = np.cumsum(d_ext)[: N - 1] > 0
    extended = ext_all & ~neutral

    # ---- zones (merged neutral intervals) + representative speed traces -----
    peak_v = peak[valid]
    zones: list[CoastZone] = []

    def trace(i0: int, i1: int):
        vv = np.empty(i1 - i0 + 1)
        vv[0] = vc[i0]
        u = vc[i0] ** 2
        for k, i in enumerate(range(i0, i1)):
            u = ustar[i] + (u - ustar[i]) * decay
            vv[k + 1] = np.sqrt(max(u, 0.0))
        return x[i0:i1 + 1], vv * 3.6

    for a_i, b_i in _runs(neutral):
        sel = (sv >= a_i) & (sv < b_i)
        first = np.flatnonzero(sel)[0]
        i0 = int(sv[first])
        i1 = int(max(e2v[first], e1v[first]))
        tx, tv = trace(i0, i1)
        zones.append(CoastZone(
            start_km=float(x[a_i]), end_km=float(x[b_i]),
            ext_end_km=float(x[int(e2v[sel].max())]),
            peak_kmh=float(peak_v[sel].max() * 3.6), cruise_kmh=float(cruise_kmh[a_i]),
            trace_x_km=tx, trace_v_kmh=tv))

    # ---- energy bookkeeping (what cruising over these stretches would cost) --
    vseg = 0.5 * (vc[:-1] + vc[1:])
    p_mech = (K_DRAG * vseg ** 2 + f_rg) * vseg
    dt = dx / vseg
    draw_kwh = np.where(p_mech > 0, p_mech / MOTOR_EFF, 0.0) * dt / 3.6e6
    regen_kwh = np.where(p_mech < 0, -p_mech * REGEN_EFF, 0.0) * dt / 3.6e6
    both = neutral | extended
    seg_km = dx / 1000.0
    total_km = float(x[-1])

    # KE that the extended coast borrows (cruise -> floor) per zone, summed
    borrowed_kj = sum(
        0.5 * MASS_KG * ((z.cruise_kmh / 3.6) ** 2 - (float(z.trace_v_kmh.min()) / 3.6) ** 2)
        for z in zones if z.trace_v_kmh.size and z.trace_v_kmh.min() < z.cruise_kmh) / 1e3

    stats = dict(
        total_km=total_km,
        neutral_km=float(neutral.sum() * seg_km),
        extended_km=float(extended.sum() * seg_km),
        neutral_frac=float(neutral.sum() * seg_km / total_km),
        both_frac=float(both.sum() * seg_km / total_km),
        n_zones=len(zones),
        n_candidate_starts=int(n),
        n_overspeed_truncated=int((reason[~valid] == REASON_OVERSPEED).sum()
                                  + over_first.sum()),
        n_barrier_truncated=int((reason[~valid] == REASON_BARRIER).sum()),
        neutral_avoided_traction_kwh=float(draw_kwh[neutral].sum()),
        both_avoided_traction_kwh=float(draw_kwh[both].sum()),
        neutral_forgone_regen_kwh=float(regen_kwh[neutral].sum()),
        both_forgone_regen_kwh=float(regen_kwh[both].sum()),
        borrowed_ke_kj=float(borrowed_kj),
        max_descent_grade_pct=float(route.gradient_pct.min()),
    )
    return CoastResult(route, cruise_kmh, neutral, extended, zones, stats,
                       v_min_kmh, v_max_kmh)


# ================================ REPORTING =================================
def print_report(res: CoastResult, top_n: int = 15) -> None:
    s = res.stats
    print("\n" + "=" * 78)
    print("COASTING ANALYSIS  (constants from " + CONSTANTS_SOURCE + ")")
    print("=" * 78)
    print(f"Speed window          : {res.v_min_kmh:g} - {res.v_max_kmh:g} km/h")
    print(f"Cruise speeds (bands) : {sorted(set(np.round(res.cruise_kmh, 1).tolist()))} km/h")
    print(f"Route analysed        : {s['total_km']:,.1f} km")
    print(f"Steepest descent      : {s['max_descent_grade_pct']:.2f} %")
    print(f"Energy-neutral coast  : {s['neutral_km']:,.2f} km  ({100*s['neutral_frac']:.2f} % of route)")
    print(f"Extended coast (extra): {s['extended_km']:,.2f} km  "
          f"(neutral + extended = {100*s['both_frac']:.2f} %)")
    print(f"Distinct zones        : {s['n_zones']}")
    print(f"Traction energy avoided vs cruising: neutral {s['neutral_avoided_traction_kwh']:.3f} kWh, "
          f"incl. extended {s['both_avoided_traction_kwh']:.3f} kWh")
    print(f"Regen forgone vs cruising         : neutral {s['neutral_forgone_regen_kwh']:.3f} kWh, "
          f"incl. extended {s['both_forgone_regen_kwh']:.3f} kWh")
    print(f"Kinetic energy borrowed by extended coasts: {s['borrowed_ke_kj']:.1f} kJ")
    print(f"Starts rejected for exceeding {res.v_max_kmh:g} km/h: {s['n_overspeed_truncated']}, "
          f"cut by a stop barrier: {s['n_barrier_truncated']}")
    if res.zones:
        print(f"\nLongest {min(top_n, len(res.zones))} energy-neutral zones:")
        print(f"  {'start km':>9} {'end km':>9} {'len km':>7} {'ext end km':>11} {'peak km/h':>10} {'cruise':>7}")
        for z in sorted(res.zones, key=lambda z: -z.length_km)[:top_n]:
            print(f"  {z.start_km:9.2f} {z.end_km:9.2f} {z.length_km:7.2f} "
                  f"{z.ext_end_km:11.2f} {z.peak_kmh:10.1f} {z.cruise_kmh:7.0f}")
    else:
        print("\nNo downhill steep enough to accelerate the car at the chosen cruise speed.")
    print("=" * 78)


def sweep_cruise_speeds(route: StitchedRoute, speeds=range(60, 101, 5)) -> list[tuple]:
    print("\nSensitivity to (uniform) cruise speed:")
    print(f"  {'v_c km/h':>8} {'neutral km':>11} {'neutral %':>10} {'+ext %':>8} {'zones':>6}")
    rows = []
    for v in speeds:
        r = analyse_coasting(route, (float(v),) * 5)
        st = r.stats
        rows.append((v, st["neutral_km"], st["neutral_frac"], st["both_frac"], st["n_zones"]))
        print(f"  {v:8d} {st['neutral_km']:11.2f} {100*st['neutral_frac']:10.2f} "
              f"{100*st['both_frac']:8.2f} {st['n_zones']:6d}")
    return rows


# ================================ PLOTTING ==================================
GREEN, ORANGE = "tab:green", "tab:orange"


def _shade(ax, x: np.ndarray, mask: np.ndarray, color: str, alpha: float,
           x_lo: float | None = None, x_hi: float | None = None) -> None:
    """Shade runs of a per-segment mask.  A thin same-colour edge keeps zones that
    are far narrower than one pixel visible on the 3,000 km axis."""
    for a, b in _runs(mask):
        x0, x1 = x[a], x[b]
        if (x_hi is not None and x0 > x_hi) or (x_lo is not None and x1 < x_lo):
            continue
        ax.axvspan(x0, x1, color=color, alpha=alpha, lw=1.0, zorder=1)


def _annotate_route(ax, route: StitchedRoute, label_stops: bool, x_lo=0.0, x_hi=None) -> None:
    x_hi = x_hi if x_hi is not None else route.x_km[-1]
    for st, nm in zip(route.section_starts_km[1:], route.section_names[1:]):
        if x_lo <= st <= x_hi:
            ax.axvline(st, color="0.35", lw=1.0, ls="-", zorder=2)
    for name, km in CONTROL_STOPS_KM.items():
        if x_lo <= km <= x_hi:
            ax.axvline(km, color="k", lw=0.7, ls="--", alpha=0.55, zorder=2)
            if label_stops:
                ax.text(km, 0.02, name, rotation=90, va="bottom", ha="right", fontsize=7,
                        transform=ax.get_xaxis_transform())


def _legend_handles():
    return [Patch(facecolor=GREEN, alpha=0.5, label="Energy-neutral coast (speed returns to cruise)"),
            Patch(facecolor=ORANGE, alpha=0.4, label="Extended coast (borrows KE down to 60 km/h)")]


def plot_coasting(res: CoastResult, out_path: Path | None = None) -> plt.Figure:
    r, x = res.route, res.route.x_km
    fig, axes = plt.subplots(3, 1, figsize=(20, 11), sharex=True,
                             gridspec_kw=dict(height_ratios=[3.2, 1.2, 2.0]))
    ax_e, ax_g, ax_v = axes

    ax_e.plot(x, r.altitude_m, color="0.15", lw=0.9, zorder=3)
    ax_e.set_ylabel("Altitude (m)")
    ax_e.set_title(
        f"Feasible coasting regions on the stitched route  |  "
        f"neutral {res.stats['neutral_km']:.1f} km ({100*res.stats['neutral_frac']:.2f}%), "
        f"neutral+extended {100*res.stats['both_frac']:.2f}%  |  "
        f"v window {res.v_min_kmh:g}-{res.v_max_kmh:g} km/h", fontweight="bold")
    ax_g.plot(x[:-1] + 0.5 * r.step_m / 1000.0, 0.5 * (r.gradient_pct[:-1] + r.gradient_pct[1:]),
              color="0.25", lw=0.5, zorder=3)
    ax_g.axhline(0, color="k", lw=0.5)
    ax_g.set_ylabel("Gradient (%)")

    for ax in (ax_e, ax_g, ax_v):
        _shade(ax, x, res.extended, ORANGE, 0.35)
        _shade(ax, x, res.neutral, GREEN, 0.55)
        ax.grid(True, ls=":", alpha=0.4)
    _annotate_route(ax_e, r, label_stops=True)
    _annotate_route(ax_g, r, label_stops=False)
    _annotate_route(ax_v, r, label_stops=False)

    for z in res.zones:
        ax_v.plot(z.trace_x_km, z.trace_v_kmh, color="tab:blue", lw=1.3, zorder=3)
    ax_v.step(x, res.cruise_kmh, where="post", color="0.4", lw=1.0, label="Cruise speed")
    ax_v.axhline(res.v_min_kmh, color="red", ls="--", lw=1.0, label=f"Min {res.v_min_kmh:g} km/h")
    ax_v.axhline(res.v_max_kmh, color="red", ls="--", lw=1.0, label=f"Max {res.v_max_kmh:g} km/h")
    ax_v.set(ylabel="Coasting speed (km/h)", xlabel="Cumulative distance (km)",
             xlim=(0, x[-1]), ylim=(res.v_min_kmh - 5, res.v_max_kmh + 5))

    handles = _legend_handles() + [
        plt.Line2D([], [], color="tab:blue", lw=1.5, label="Speed while coasting (earliest start of each zone)"),
        plt.Line2D([], [], color="0.4", lw=1.0, label="Cruise speed"),
        plt.Line2D([], [], color="red", ls="--", label="Speed limits"),
        plt.Line2D([], [], color="k", ls="--", alpha=0.55, label="Control stops (barriers)"),
        plt.Line2D([], [], color="0.35", label="Per-day route file boundary")]
    ax_e.legend(handles=handles, loc="upper right", ncol=2, fontsize=8, framealpha=0.9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"[*] Saved {out_path}")
    return fig


def plot_band_zooms(res: CoastResult, out_path: Path | None = None) -> plt.Figure:
    """Same shading, one row per 600 km speed band, for legibility."""
    r, x = res.route, res.route.x_km
    n_bands = int(np.ceil(x[-1] / SPEED_BAND_KM))
    fig, axes = plt.subplots(n_bands, 1, figsize=(20, 3.0 * n_bands))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        lo, hi = i * SPEED_BAND_KM, min((i + 1) * SPEED_BAND_KM, x[-1])
        m = (x >= lo) & (x <= hi)
        ax.plot(x[m], r.altitude_m[m], color="0.15", lw=0.9, zorder=3)
        _shade(ax, x, res.extended, ORANGE, 0.35, lo, hi)
        _shade(ax, x, res.neutral, GREEN, 0.60, lo, hi)
        _annotate_route(ax, r, label_stops=True, x_lo=lo, x_hi=hi)
        ax.set(xlim=(lo, hi), ylabel="Altitude (m)")
        ax.set_title(f"Band {i + 1}: {lo:.0f}-{hi:.0f} km  (cruise {res.cruise_kmh[m][0]:.0f} km/h)",
                     fontsize=10, loc="left")
        ax.grid(True, ls=":", alpha=0.4)
    axes[-1].set_xlabel("Cumulative distance (km)")
    axes[0].legend(handles=_legend_handles(), loc="upper right", fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"[*] Saved {out_path}")
    return fig


# ================================= DRIVER ===================================
def run_coasting_analysis(cruise_speeds_kmh=CRUISE_SPEEDS_KMH, day_files=None,
                          step_m: float = GRID_STEP_M, output_dir: Path = OUTPUT_DIR,
                          make_zooms: bool = True, run_sweep: bool = False):
    """Load + stitch the per-day routes, run the analysis, save plots.

    Returns (fig, result): `fig` is the full stitched-route coasting plot.
    """
    route = load_stitched_route(day_files, step_m=step_m)
    res = analyse_coasting(route, cruise_speeds_kmh)
    print_report(res)
    output_dir = Path(output_dir)
    fig = plot_coasting(res, output_dir / "wsc_coasting_regions.png")
    if make_zooms:
        plot_band_zooms(res, output_dir / "wsc_coasting_regions_by_band.png")
    if run_sweep:
        sweep_cruise_speeds(route)
    return fig, res


def main() -> None:
    run_coasting_analysis(run_sweep=True)
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
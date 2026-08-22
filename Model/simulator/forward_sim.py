"""
simulator/forward_sim.py — the forward integrator (owner: Junior C, Plan v3 §8).
Every optimizer candidate is evaluated through this module.

This module owns the variable-speed integration loop used by the high-fidelity
Tier 2 solver. It models mandatory driver swaps, battery breakdown probabilities,
and integrates physics on the high-resolution grid.

BREAKDOWN MODEL — wired to core/options.py (12/08 fix). This used to have its
own inline simulate_breakdown() calling raw random.random() every substep,
which meant every DE/GA/SLSQP candidate evaluation saw a different fitness for
the *same* speed vector — noise the optimizer reads as "this direction is bad"
even when it isn't. That copy is deleted. We now use core.options.BreakdownModel,
called the same way core/options.py's own BreakdownModel docstring requires:
rng=None -> deterministic expected_stop_s() (smooth, no RNG, safe inside an
optimizer loop); rng=<seeded random.Random> -> one stochastic sample_stop_s()
draw per substep, for MC/scenario/robustness runs only.

SOLAR UNDERUTILIZATION TRACKING (13/08 addition):
  solar_underutil_j tracks wasted solar capacity — energy the panel could
  produce but the battery can't absorb (SOC ceiling, or motor draw < solar
  output). This feeds into singleday.py's enriched cost function.

PERF — HOT-LOOP VECTORIZATION (15/08): the integrator previously called
route.slope_pct_at / red_flag_at / control_stop_at / seg_type_at and
solar_provider.ghi_wm2 once PER SUBSTEP, each issuing its own searchsorted,
pandas row-boxing, or cKDTree query. At ENERGY_GRID_M=100 on 280-355 km
routes that is ~3,500 substeps per day per candidate, and SLSQP's
finite-difference gradient estimation multiplied that across every solve.
All route lookups are now resolved ONCE PER SEGMENT as vectorized arrays
(route.slope_pct_array etc. — core/route.py) and the weather node index is
resolved once per segment (solar.node_index_array), so the substep loop only
pays one scalar spline evaluation (ghi_wm2_at_node) — no per-substep
searchsorted / KDTree. Positions are built with the same float-addition
chain the old loop used, so results are bit-identical to the pre-vectorized
integrator.
"""

from __future__ import annotations

import dataclasses
import random
import typing as _t
import numpy as np

from configs import race_config
from configs import solver_config as _sc_forward
from configs.car_config import CarState
from core import physics
from core.battery import Battery
from core.route import Route
from core import wind as wind_core
from core.options import BreakdownModel

_DRIVER_SWAP_STANDALONE_DURATION_S = race_config.LOOP_STOP_DURATION_S
# Speed at which the trailer/tow truck moves through red-flag segments (km/h).
_TRAILER_SPEED_KMH = 80.0


@dataclasses.dataclass
class DayEvalResult:
    final_soc_pct: float
    total_time_s: float
    driver_swaps: list
    v_ms: np.ndarray
    t_s: np.ndarray
    x_m: np.ndarray
    # Total breakdown stall time folded into total_time_s above.
    # rng=None runs -> sum of expected_stop_s per substep (deterministic).
    # rng=<Random> runs -> sum of realized sample_stop_s draws (stochastic).
    breakdown_s: float = 0.0
    # Only populated on stochastic (rng given) runs — one entry per substep
    # that actually triggered a failure. Empty on deterministic runs since
    # there's no discrete "triggered" event in the expected-value path.
    breakdown_log: list = dataclasses.field(default_factory=list)
    # Solar energy the panel could produce but the system couldn't absorb (J).
    # Non-zero when: battery at SOC ceiling, or motor draw << solar output
    # and the excess can't be stored. Used by singleday.py's cost function.
    solar_underutil_j: float = 0.0
    # Real cumulative motor+idle electrical consumption (Wh), integrated
    # directly from physics every substep. NOT derived from start/end SOC --
    # unlike a SOC-delta-based estimate, this stays meaningful even on days
    # where the battery clips at the SOC ceiling (start_soc == end_soc would
    # make a SOC-delta estimate collapse to whatever the solar estimate says,
    # not what the motor actually drew).
    motor_energy_wh: float = 0.0
    solar_energy_wh: float = 0.0
    # Trailering: number of substeps and total distance on the trailer.
    trailered_substeps: int = 0
    trailered_km: float = 0.0
    # Distance actually DRIVEN (excludes trailered km). Report/strategy
    # "distance covered" figures must sum this, not trailered_km — SR's
    # asterisk rule ranks any trailered team below all non-trailered teams
    # regardless of distance, so trailered km must never be presented as
    # race distance covered.
    driven_km: float = 0.0
    # ── Per-substep dashboard traces (Plan v3 §7.2 Dashboard) ──────────────
    # One entry per integrated substep, aligned 1:1 with t_s / x_m above, so
    # the dashboard can plot SOC / velocity / solar / gradient vs DISTANCE
    # (x_m) or time (t_s). These are the continuous curves the coarse
    # per-control-segment velocity card cannot provide. Empty by default so
    # every existing caller/return path is unaffected.
    soc_pct_trace: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([]))
    v_kmh_trace: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([]))
    solar_w_trace: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([]))
    slope_pct_trace: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([]))
    # Battery-safety exposure: time-integrated SOC ABOVE the safe band, in
    # (SOC-fraction · seconds). The L2 objective penalizes this so the car
    # doesn't coast at ~100% for long (pack-cooking + wasted solar); the more
    # time near the ceiling, the bigger this grows.
    soc_over_safe_pct_s: float = 0.0


class DriverSwapScheduler:
    """Tracks on-seat elapsed time and decides when SR 2.24.4 swaps land."""

    def __init__(self, swap_interval_s: float = race_config.DRIVER_SWAP_INTERVAL_S,
                 standalone_duration_s: float = _DRIVER_SWAP_STANDALONE_DURATION_S):
        self.swap_interval_s = swap_interval_s
        self.standalone_duration_s = standalone_duration_s
        self._elapsed_since_last_swap_s = 0.0
        self.swap_log: list[dict] = []

    def advance(self, dt_s: float, t_now_s: float, x_m: float,
                coincides_with_stop: bool) -> float:
        self._elapsed_since_last_swap_s += dt_s
        if self._elapsed_since_last_swap_s < self.swap_interval_s:
            return 0.0
        self._elapsed_since_last_swap_s = 0.0
        added_s = 0.0 if coincides_with_stop else self.standalone_duration_s
        self.swap_log.append(dict(t_s=t_now_s, x_m=x_m,
                                   piggybacked=coincides_with_stop,
                                   added_s=added_s))
        return added_s


def _ghi_segment(solar_provider, t_nom: np.ndarray, x_pre: np.ndarray,
                 seg_nodes, ghi_at_node_fn) -> np.ndarray:
    """GHI for a whole segment's substeps in as few provider calls as possible.

    Fast path: providers exposing ghi_wm2_array (HourlyJSONSolarProvider and
    the Gaussian-floored wrapper) evaluate one cubic spline per weather node,
    vectorized over the substeps assigned to that node. Fallbacks preserve the
    old scalar behaviour for providers without the batch API. Always clipped
    to >= 0.
    """
    n = len(t_nom)
    arr_fn = getattr(solar_provider, "ghi_wm2_array", None)
    if arr_fn is not None:
        try:
            out = np.asarray(arr_fn(t_nom, x_pre), dtype=float)
            if out.shape == t_nom.shape:
                return np.clip(out, 0.0, None)
        except Exception:
            pass
    if ghi_at_node_fn is not None and seg_nodes is not None:
        out = np.array([ghi_at_node_fn(float(t_nom[k]), int(seg_nodes[k]))
                        for k in range(n)], dtype=float)
        return np.clip(out, 0.0, None)
    out = np.array([solar_provider.ghi_wm2(float(t_nom[k]), float(x_pre[k]))
                    for k in range(n)], dtype=float)
    return np.clip(out, 0.0, None)


def simulate_variable_speed(v_kmh: np.ndarray, route: Route, car: CarState,
                            solar_provider, wind_provider, t0_s: float,
                            start_soc_pct: float, seg_start_m: np.ndarray,
                            seg_len_m: float, energy_grid_m: float,
                            rng: _t.Optional[random.Random] = None, *,
                            regen_cap_w: float | None = None,
                            cs_taken: bool = False,
                            loop_stop_duration_s: float | None = None,
                            unplanned_stop_budget_s: float | None = None
                            ) -> DayEvalResult:
    """
    The centralized real integrator (reuses core.physics.net_power + core.battery.Battery).
    Velocity is held per control segment; physics integrates on the finer
    energy grid within each control segment.

        rng: None (default) -> breakdown stall time is the deterministic expected
         value (BreakdownModel.expected_stop_s), so calling this twice with the
         same v_kmh gives the exact same result — required for DE/GA/SLSQP,
         which needs a stable objective to converge on, and for DayEvaluator's
         cache to stay valid. Pass a seeded random.Random() only for scenario/
         Monte-Carlo runs (simulator/scenarios.py) where you explicitly want a
         sampled outcome, not the average.

    Keyword-only Tier 1 parity knobs (defaults replicate Tier 1's model):
      regen_cap_w             : hard regen charge-back clamp (W). None ->
                                core.physics.regen_cap_w(car), the SAME cap
                                Tier 1 has always applied (forward_sim used to
                                leave regen uncapped — Tier 2/L2 was optimistic
                                vs Tier 1 on downhill-heavy days).
      cs_taken                : control stop already done today -> the parked
                                control-stop solar credit covers only the
                                unplanned-stop budget, exactly like Tier 1.
      loop_stop_duration_s    : override parked time per loop turnaround
                                (default LOOP_STOP_DURATION_S + LOOP_TURNAROUND_S).
      unplanned_stop_budget_s : override parked time added at the control stop
                                (default race_config.UNPLANNED_STOP_BUDGET_S).
    """
    battery = Battery(car, start_soc_pct)
    swap_scheduler = DriverSwapScheduler()
    breakdown_model = BreakdownModel()
    breakdown_model.reset()  # each call is one independent day-sim, not a
                              # continuation of the previous candidate's risk state
    t_s = float(t0_s)
    x_m = float(seg_start_m[0]) if len(seg_start_m) > 0 else 0.0
    total_breakdown_s = 0.0
    breakdown_log: list[dict] = []
    solar_underutil_j = 0.0
    motor_energy_wh = 0.0
    solar_energy_wh = 0.0
    trailered_substeps = 0
    trailered_km_accum = 0.0
    driven_km_accum = 0.0
    soc_over_safe_accum = 0.0
    _soc_safe_max = getattr(_sc_forward, "SOC_SAFE_MAX_PCT", 100.0)

    n_seg = len(v_kmh)
    # Per-segment lengths, clamped so the final segment never integrates past
    # route.total_m. The route parquet is rarely an exact multiple of
    # CONTROL_SEGMENT_M; without this clamp the integrator would drive up to
    # one full phantom control segment (~10 km) past the end of the day's
    # route, inflating driven_km / trailered_km / total_time_s in the report.
    seg_lens_m = np.full(n_seg, float(seg_len_m))
    if route is not None and n_seg > 0:
        seg_lens_m[-1] = max(
            0.0,
            min(float(seg_len_m), float(route.total_m) - float(seg_start_m[-1])),
        )

    # Audit fix: t_array/x_array must be function-scope — the original had
    # them nested under the `if route is not None` block, so any route-less
    # call (route=None flat fallback) hit a NameError here.
    t_array = []
    x_array = []
    # Per-substep dashboard traces (aligned 1:1 with t_array / x_array).
    soc_array = []
    v_kmh_array = []
    solar_w_array = []
    slope_array = []

    # --- Model-parity knobs (keyword-only; defaults replicate Tier 1) ---
    # Regen clamp: None -> core.physics.regen_cap_w(car), the exact cap Tier 1
    # applies. forward_sim previously left regen charge-back uncapped, making
    # Tier 2 / L2 systematically more optimistic than Tier 1 on downhill-heavy
    # days (and letting Tier 3 reject plans Tier 1 proposed).
    if regen_cap_w is None:
        regen_cap_w = physics.regen_cap_w(car)

    # Stop-time solar charging (Tier 1 parity — Tier 1's coarse evaluate_day
    # credits parked solar at the control stop and at each loop turnaround;
    # forward_sim previously credited NONE of it).
    control_stop_dur_s = float(race_config.CONTROL_STOP_DURATION_S)
    loop_stop_dur_s = float(race_config.LOOP_STOP_DURATION_S
                            + getattr(race_config, "LOOP_TURNAROUND_S", 0.0))
    unplanned_budget_s = float(race_config.UNPLANNED_STOP_BUDGET_S)
    if loop_stop_duration_s is not None:
        loop_stop_dur_s = float(loop_stop_duration_s)
    if unplanned_stop_budget_s is not None:
        unplanned_budget_s = float(unplanned_stop_budget_s)

    # One-shot credits: the control-stop solar credit fires at the first
    # substep inside the control-stop zone (once per day); the loop-zone
    # credit fires once per contiguous loop zone (once per loop attempt).
    cs_credit_done = False
    in_loop_zone = False

    # Audit hot-loop fix (15/08): resolve weather-node indexing support ONCE.
    # Providers with the array API (HourlyJSONSolarProvider) expose
    # node_index_array + ghi_wm2_at_node; legacy/fallback providers only have
    # the scalar ghi_wm2 and fall back to a per-substep call here.
    use_route = route is not None
    node_index_fn = getattr(solar_provider, "node_index_array", None)
    ghi_at_node_fn = getattr(solar_provider, "ghi_wm2_at_node", None)

    for seg_i, v in enumerate(v_kmh):
        v_ms = float(v) / 3.6
        seg_len_i = float(seg_lens_m[seg_i])
        if seg_len_i <= 0.0:
            continue
        n_substeps = max(1, round(seg_len_i / energy_grid_m))
        substep_len_km = (seg_len_i / n_substeps) / 1000.0
        step_m_local = substep_len_km * 1000.0

        # ── Resolve every route + weather lookup for this whole segment ONCE,
        # as vectorized arrays, instead of per-substep Python-API calls
        # (route.slope_pct_at / red_flag_at / control_stop_at / seg_type_at /
        # provider.ghi_wm2 each did their own searchsorted, row-boxing, or
        # cKDTree query). GHI is still evaluated per substep via the cheap
        # ghi_wm2_at_node spline call because t_s is data-dependent inside
        # the segment (breakdown stalls shift subsequent substep times).
        if use_route:
            step_m = substep_len_km * 1000.0
            # Positions are built with the SAME repeated float-addition chain
            # the loop below performs (cumsum of [x_m, step, step, ...]), so
            # array lookups land on bit-identical grid points to the old
            # per-substep `x_m += step_m` walk.
            chain = np.cumsum(np.concatenate(
                (np.array([x_m]), np.full(n_substeps, step_m))))
            x_pre = chain[:-1]    # positions where slope/ghi/red are read
            x_post = chain[1:]    # positions where the stop-zone test runs
            seg_slopes = route.slope_pct_array(x_pre)
            seg_reds = route.red_flag_array(x_pre)
            seg_cs_stops = route.control_stop_array(x_post)
            seg_loop_stops = np.char.startswith(
                route.seg_type_array(x_post).astype(str), "loop_")
            seg_stops = seg_cs_stops | seg_loop_stops
            seg_nodes = (node_index_fn(x_pre)
                         if node_index_fn is not None else None)
        else:
            seg_slopes = seg_reds = seg_stops = None
            seg_nodes = None

        # ── VECTORIZED per-segment physics + solar (15/09 perf rewrite) ──
        # The old inner loop called physics.net_power (forces) and the GHI
        # spline ONCE PER SUBSTEP in pure Python — profiling showed ~88% of
        # forward_sim time was those two calls, ~2.4k times per candidate,
        # multiplied across every GA/SLSQP evaluation. Both are elementwise,
        # so they're now computed ONCE PER SEGMENT as numpy arrays; only the
        # genuinely-sequential state (SOC clip/underutil, breakdown risk,
        # driver-swap clock, traces) stays in the light loop below.
        #
        # GHI is evaluated on the within-segment NOMINAL time grid
        # (t_seg_start + cumulative dt, ignoring intra-segment breakdown/swap
        # drift). The segment-start t_s already carries ALL prior breakdown +
        # swap time, so the only error is sub-minute drift WITHIN one ~10 km
        # segment — negligible for GHI (which varies on an hourly scale) and
        # zero for energy (energy integrates dt, not clock time).
        if use_route:
            seg_reds_b = seg_reds.astype(bool)
            seg_cs_b = seg_cs_stops.astype(bool)
            seg_loop_b = seg_loop_stops.astype(bool)
            seg_stops_b = seg_stops.astype(bool)
            slopes_arr = np.asarray(seg_slopes, dtype=float)
            x_pre_arr = x_pre
        else:
            seg_reds_b = np.zeros(n_substeps, dtype=bool)
            seg_cs_b = np.zeros(n_substeps, dtype=bool)
            seg_loop_b = np.zeros(n_substeps, dtype=bool)
            seg_stops_b = np.zeros(n_substeps, dtype=bool)
            slopes_arr = np.zeros(n_substeps, dtype=float)
            x_pre_arr = x_m + np.arange(n_substeps, dtype=float) * step_m_local

        drive_dt = step_m_local / v_ms
        trailer_dt = step_m_local / (_TRAILER_SPEED_KMH / 3.6)
        dt_arr = np.where(seg_reds_b, trailer_dt, drive_dt)
        # Within-segment nominal clock for GHI lookups (start-of-substep).
        t_nom = t_s + np.concatenate(([0.0], np.cumsum(dt_arr)[:-1]))
        ghi_arr = _ghi_segment(solar_provider, t_nom, x_pre_arr,
                               seg_nodes, ghi_at_node_fn)
        # Physics for the whole segment at once (as if driving); trailered
        # substeps are zeroed right after (inert cargo — no energy flow).
        v_ms_arr = np.full(n_substeps, v_ms, dtype=float)
        p_net_arr, _dt_unused = physics.net_power(
            car, v_ms_arr, v_ms_arr, slopes_arr, ghi_arr, substep_len_km,
            regen_cap_w=regen_cap_w)
        p_net_arr = np.where(seg_reds_b, 0.0, p_net_arr)
        p_solar_arr = car.array_area_m2 * car.array_efficiency * ghi_arr

        for k in range(n_substeps):
            slope = float(slopes_arr[k])
            is_trailered = bool(seg_reds_b[k])
            stop_here = bool(seg_stops_b[k])
            cs_stop_here = bool(seg_cs_b[k])
            loop_stop_here = bool(seg_loop_b[k])
            ghi = float(ghi_arr[k])
            p_solar_w = float(p_solar_arr[k])

            # --- Check if this grid point is on a trailered segment ---
            if is_trailered:
                # Car is on the trailer: NO power input, NO power drain, NO
                # charging whatsoever (explicit requirement — trailered
                # segments must not accumulate any energy, solar or
                # otherwise; the car is inert cargo on a tow truck here).
                dt_s_step = float(trailer_dt)
                p_net = 0.0
                trailered_substeps += 1
                trailered_km_accum += substep_len_km
            else:
                dt_s_step = float(drive_dt)
                p_net = float(p_net_arr[k])
                driven_km_accum += substep_len_km

            # --- Stop-time solar charging (Tier 1 parity) ---
            # Tier 1's coarse evaluate_day credits parked solar at the control
            # stop (CONTROL_STOP_DURATION_S + UNPLANNED_STOP_BUDGET_S, the
            # control-stop term dropping to 0 when cs_taken) and at each loop
            # turnaround (LOOP_STOP_DURATION_S + LOOP_TURNAROUND_S).
            # forward_sim previously credited NO parked solar, so Tier 2 / L2
            # candidates were systematically more pessimistic than Tier 1 on
            # sunny days and Tier 3 could reject the very plan Tier 1 proposed.
            # Credit exactly once per day (control stop) and once per
            # contiguous loop zone (each loop attempt). Skipped on trailered
            # segments (car is inert cargo — no energy flow, same as Tier 1's
            # trailered-mask zeroing).
            if not is_trailered:
                if cs_stop_here and not cs_credit_done:
                    cs_credit_done = True
                    _parked_s = ((0.0 if cs_taken else control_stop_dur_s)
                                 + unplanned_budget_s)
                    _parked_w = (car.array_area_m2 * car.array_efficiency * ghi
                                 - car.p_idle_w)
                    battery.apply_energy_wh(_parked_w * _parked_s / 3600.0)
                if loop_stop_here:
                    if not in_loop_zone:
                        in_loop_zone = True
                        _parked_w = (car.array_area_m2 * car.array_efficiency * ghi
                                     - car.p_idle_w)
                        battery.apply_energy_wh(
                            _parked_w * loop_stop_dur_s / 3600.0)
                else:
                    in_loop_zone = False

            # --- Solar underutilization tracking ---
            # Skip entirely for trailered segments (no energy flow).
            if not is_trailered:
                # p_net = p_solar - p_electric - p_idle (core/physics.py convention)
                # Available solar power at this instant:
                p_solar_w = car.array_area_m2 * car.array_efficiency * ghi
                # Total consumption (motor + idle):
                p_consumed_w = p_solar_w - float(p_net)
                # Real, non-circular energy totals — integrated directly from
                # physics every substep, NOT backed out from start/end SOC.
                motor_energy_wh += p_consumed_w * float(dt_s_step) / 3600.0
                solar_energy_wh += p_solar_w * float(dt_s_step) / 3600.0
                # If solar exceeds consumption AND battery is near full, excess is wasted:
                solar_excess_w = max(0.0, p_solar_w - p_consumed_w)
                # Check if battery can absorb the excess (SOC headroom)
                soc_headroom_wh = (car.soc_max_pct - battery.soc_pct) / 100.0 * car.battery_nominal_wh
                absorbable_wh = soc_headroom_wh / car.charge_eff  # absorbable PANEL-Wh: battery stores delta*charge_eff (apply_energy_wh)
                excess_wh_this_step = solar_excess_w * float(dt_s_step) / 3600.0
                if excess_wh_this_step > absorbable_wh and absorbable_wh >= 0:
                    wasted_wh = excess_wh_this_step - absorbable_wh
                    solar_underutil_j += wasted_wh * 3600.0  # convert Wh -> J

            # p_net == 0 for trailered segments → no SOC change.
            battery.apply_energy_wh(float(p_net) * float(dt_s_step) / 3600.0)

            t_array.append(t_s)
            x_array.append(x_m)
            # Dashboard traces (aligned 1:1 with t_array/x_array). battery.soc_pct
            # already reflects this substep's energy (applied just above). Solar
            # is 0 on trailered substeps (car is inert cargo — no capture); its
            # driving speed is the tow speed, not the segment target.
            soc_array.append(battery.soc_pct)
            v_kmh_array.append(_TRAILER_SPEED_KMH if is_trailered else float(v))
            solar_w_array.append(0.0 if is_trailered else float(p_solar_w))
            slope_array.append(float(slope))
            # Battery-safety exposure: accumulate time spent above the safe band.
            if battery.soc_pct > _soc_safe_max:
                soc_over_safe_accum += (battery.soc_pct - _soc_safe_max) / 100.0 * float(dt_s_step)

            t_s += float(dt_s_step)
            x_m += step_m_local

            t_s += swap_scheduler.advance(
                float(dt_s_step), t_s, x_m, coincides_with_stop=stop_here)

            # BreakdownModel call — skip for trailered segments (motor off).
            if not is_trailered:
                # Input: motor ELECTRICAL DRAW, not physics.net_power's return.
                # DEFAULT_SCENARIOS' "p_net" threshold (2000-4100 W) is the
                # motor-power BMS-trip number. Back out from the solar term:
                # p_net = p_solar - p_electric - p_idle => p_electric = p_solar - p_net - p_idle
                motor_draw_w = max(0.0, p_solar_w - float(p_net) - car.p_idle_w)
                inputs = {"p_net": motor_draw_w}
                if rng is None:
                    breakdown_s = breakdown_model.expected_stop_s(inputs)
                    breakdown_model.step(float(dt_s_step), inputs)
                else:
                    breakdown_s, triggered_category = breakdown_model.sample_stop_s(inputs, rng)
                    breakdown_model.step(float(dt_s_step), inputs,
                                          triggered_category=triggered_category,
                                          triggered_duration_s=breakdown_s)
                    if triggered_category is not None:
                        breakdown_log.append(dict(t_s=t_s, x_m=x_m,
                                                   category=triggered_category,
                                                   duration_s=breakdown_s))
                total_breakdown_s += breakdown_s
                t_s += breakdown_s

    return DayEvalResult(
        final_soc_pct=battery.soc_pct,
        total_time_s=t_s - t0_s,
        driver_swaps=swap_scheduler.swap_log,
        v_ms=v_kmh / 3.6,
        t_s=np.array(t_array),
        x_m=np.array(x_array),
        breakdown_s=total_breakdown_s,
        breakdown_log=breakdown_log,
        solar_underutil_j=solar_underutil_j,
        motor_energy_wh=motor_energy_wh,
        solar_energy_wh=solar_energy_wh,
        trailered_substeps=trailered_substeps,
        trailered_km=trailered_km_accum,
        driven_km=driven_km_accum,
        soc_over_safe_pct_s=soc_over_safe_accum,
        soc_pct_trace=np.array(soc_array),
        v_kmh_trace=np.array(v_kmh_array),
        solar_w_trace=np.array(solar_w_array),
        slope_pct_trace=np.array(slope_array),
    )
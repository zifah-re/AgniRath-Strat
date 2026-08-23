"""
core/physics.py — THE single force/power/loss model (Plan v3 §8, workplan 1.1).

Consolidates the two inherited models and upgrades drag to relative airspeed
with optional CdA(psi):

  net_power(...)        2026 primary model. Force balance from the WSC'25
                        Dashboard (mpc.py:calculate_net_power) upgraded per
                        Plan v3 §6.2:
                          P_drag uses v_air = v_car - w_par  (along-track
                          relative airspeed) and CdA(psi) from CarState.
                        Sign convention: net electrical power INTO the battery
                        (positive charges, negative drains) — matches legacy
                        Dashboard convention.

  dashboard_power(...)  Exact port of Dashboard mpc.py:calculate_net_power
                        (WSC'25). Used by tests to prove we reproduce the
                        legacy model (workplan 1.1 "tests reproduce both
                        legacy codebases' outputs").

  motor_power_kr(...)   Exact port of Kevin/Ramana race_completion/car.py
                        calculate_power: motor-level torque model with
                        wind-direction drag, iterative winding-temperature
                        thermal loop, copper/eddy/windage losses. Used by
                        tests AND as the high-fidelity backend for the
                        trailering P_req red-flag (Plan v3 §5.1, notes doc
                        idea iv). Sign convention: positive = power DRAWN
                        (consumption), clipped at 0 — matches legacy.

All functions are numpy-vectorised. Units: SI (m/s, W, kg, deg).
"""

from __future__ import annotations

import numpy as np

from configs.car_config import CarState, LEGACY_KR

G = 9.81  # m/s^2 (both legacy codebases use 9.81)

# ===========================================================================
# 2026 PRIMARY MODEL
# ===========================================================================

def forces(
    car: CarState,
    v_ms: np.ndarray,
    slope_pct: np.ndarray,
    air_density: np.ndarray | float = 1.2,
    wind_along_ms: np.ndarray | float = 0.0,
    yaw_deg: np.ndarray | float = 0.0,
) -> dict:
    """Individual longitudinal force terms (N) at speed v on gradient slope.

    slope_pct : road gradient in percent (rise/run * 100), signed.
    wind_along_ms : wind component along track, positive = TAILWIND
                    (pushes the car). v_air = v - w_par.
    yaw_deg : relative-wind yaw angle for CdA(psi) (ignored unless the CdA
              table is enabled in car_config).

    Force convention: positive values RESIST motion.
    """
    v = np.asarray(v_ms, dtype=float)
    grad = np.asarray(slope_pct, dtype=float) / 100.0

    v_air = v - np.asarray(wind_along_ms, dtype=float)   # Plan v3 §6.2
    cda = np.vectorize(car.cda_at_yaw)(yaw_deg) if np.ndim(yaw_deg) else \
        car.cda_at_yaw(float(yaw_deg))

    # Drag resists relative airflow; sign(v_air) keeps a strong tailwind
    # (v_air < 0) as a pushing force. Legacy Dashboard used v^2 (no wind).
    f_drag = 0.5 * air_density * cda * v_air * np.abs(v_air)

    # Rolling resistance with small-angle correction, exactly as Dashboard:
    # cos(theta) ~ 1 - grad^2/2.
    f_roll = car.mass_kg * G * car.crr * (1.0 - (grad ** 2) / 2.0)

    # Gravity along slope, small-angle form as Dashboard: sin(theta) ~ grad.
    f_grav = car.mass_kg * G * grad

    return dict(drag=f_drag, rolling=f_roll, gravity=f_grav)


def net_power(
    car: CarState,
    v_current_ms: np.ndarray,
    v_next_ms: np.ndarray,
    slope_pct: np.ndarray,
    ghi_wm2: np.ndarray,
    seg_len_km: np.ndarray,
    air_density: np.ndarray | float = 1.2,
    wind_along_ms: np.ndarray | float = 0.0,
    yaw_deg: np.ndarray | float = 0.0,
    solar_geom_factor: np.ndarray | float = 1.0,
    regen_cap_w: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Net electrical power into the battery over a segment, and dt.

    Mirrors Dashboard calculate_net_power's structure (accel force from
    (v_next - v_current)/dt with dt from seg_len/v; motor eff on positive
    mech power, regen eff on negative; constant idle loss) with the v3
    upgrades: relative-airspeed drag, CdA(psi), and a solar geometry factor
    (slope/bearing incidence correction from core/solar.py, default 1).

    regen_cap_w (optional): clamps regenerative charge-back power to a hard
    limit, exactly like Tier 1's _regen_cap_w. Default None leaves the old
    behaviour (uncapped regen) untouched.

    Returns (net_power_w, dt_s). Positive net_power charges the battery.
    """
    v = np.asarray(v_current_ms, dtype=float)
    v_next = np.asarray(v_next_ms, dtype=float)

    f = forces(car, v, slope_pct, air_density, wind_along_ms, yaw_deg)

    v_safe = np.maximum(v, 0.1)                       # Dashboard guard
    dt_s = np.maximum((np.asarray(seg_len_km) / v_safe) * 1000.0, 0.01)

    f_acc = car.mass_kg * (v_next - v) / dt_s
    f_total = f["drag"] + f["rolling"] + f["gravity"] + f_acc

    p_mech = f_total * v
    regen_into_pack = np.where(p_mech < 0.0, -p_mech * car.regen_eff, 0.0)
    if regen_cap_w is not None:
        regen_into_pack = np.minimum(regen_into_pack, float(regen_cap_w))
    p_electric = np.where(p_mech >= 0.0, p_mech / car.motor_eff, -regen_into_pack)

    p_solar = car.array_area_m2 * car.array_efficiency * \
        np.asarray(ghi_wm2, dtype=float) * solar_geom_factor

    return p_solar - p_electric - car.p_idle_w, dt_s


def regen_cap_w(car: CarState) -> float:
    """Max regenerative charge-back power (W) the pack can accept.

    Single source of truth shared by Tier 1's coarse energy diagnostic
    (optimizers/tier1.py _regen_cap_w now delegates here) and forward_sim's
    high-fidelity integrator (Tier 2 / L2). Explicit car.p_regen_max_w wins;
    otherwise fall back to the thermal-derated continuous limit. This is the
    exact cap Tier 1 has always applied, so Tier 2 / L2 physics now match
    Tier 1's model instead of letting regen charge back uncapped.
    """
    explicit = getattr(car, "p_regen_max_w", None)
    if explicit is not None:
        return float(explicit)
    return float(car.p_max_continuous_w * car.p_max_derating)


def power_required_at_speed(
    car: CarState,
    v_ms: float,
    slope_pct: np.ndarray,
    air_density: float = 1.2,
    wind_along_ms: float = 0.0,
) -> np.ndarray:
    """Steady-state electrical power (W) to HOLD v on each gradient.

    This is P_req(v, slope) for the trailering red-flag map (Plan v3 §5.1):
    red-flag where power_required_at_speed(car, 60/3.6, slope) exceeds
    car.p_max_continuous_w * car.p_max_derating. Positive = consumption.
    """
    f = forces(car, np.full_like(np.asarray(slope_pct, float), v_ms),
               slope_pct, air_density, wind_along_ms)
    p_mech = (f["drag"] + f["rolling"] + f["gravity"]) * v_ms
    return np.where(p_mech >= 0.0, p_mech / car.motor_eff,
                    p_mech * car.regen_eff)


# ===========================================================================
# LEGACY PORT 1 — WSC'25 Dashboard (verbatim behaviour, for tests)
# ===========================================================================

def dashboard_power(
    v_current: float | np.ndarray,
    v_next: float | np.ndarray,
    slope: float | np.ndarray,
    solar_irradiance: float | np.ndarray,
    seg_len: float | np.ndarray,
    *,
    mass: float = 300.0, cda: float = 0.16, crr: float = 0.007,
    rho: float = 1.2, solar_area: float = 5.95, solar_eff: float = 0.18,
    motor_eff: float = 0.95, regen_eff: float = 0.70, power_loss: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact port of Dashboard/mpc.py:calculate_net_power (WSC'25).

    slope in PERCENT; seg_len in KM (converted *1000 inside, as legacy).
    Defaults are the Dashboard constants verbatim.
    """
    v = np.asarray(v_current, dtype=float)
    grad = np.asarray(slope, dtype=float) / 100.0
    f_drag = 0.5 * rho * cda * (v ** 2)
    f_rolling = mass * G * crr * (1.0 - (grad ** 2) / 2.0)
    f_gravity = mass * G * grad

    v_safe = np.maximum(v, 0.1)
    dt = np.maximum((np.asarray(seg_len, float) / v_safe) * 1000.0, 0.01)

    f_acceleration = mass * (np.asarray(v_next, float) - v) / dt
    f_total = f_drag + f_rolling + f_gravity + f_acceleration
    p_solar = solar_area * solar_eff * np.asarray(solar_irradiance, float)
    p_mech = f_total * v
    p_electric = np.where(p_mech >= 0.0, p_mech / motor_eff,
                          p_mech * regen_eff)
    return p_solar - p_electric - power_loss, dt


# ===========================================================================
# LEGACY PORT 2 — Kevin/Ramana race_completion/car.py (verbatim behaviour)
# ===========================================================================

# Constants derived exactly as race_completion/car.py from LEGACY_KR values.
_KR_R_OUT = LEGACY_KR["wheel_r_out_m"]
_KR_MASS = LEGACY_KR["mass_kg"]
_KR_CRR = LEGACY_KR["crr"]
_KR_CDA = LEGACY_KR["cda_m2"]
_KR_RHO = LEGACY_KR["air_density"]
_KR_TA = LEGACY_KR["ambient_temp_k"]

_FRICTIONAL_TORQUE_COEFF = _KR_R_OUT * _KR_MASS * G * _KR_CRR
_DRAG_COEFF_BASE = 0.5 * _KR_CDA * _KR_RHO * (_KR_R_OUT ** 3)
_DRAG_COEFF = _DRAG_COEFF_BASE / (_KR_R_OUT ** 2)
_SLOPE_COEFF = _KR_MASS * G
_WINDAGE_LOSS_COEFF = (170.4e-6) / (_KR_R_OUT ** 2)


def motor_power_kr(
    speed: np.ndarray,
    acceleration: np.ndarray,
    slope_deg: np.ndarray,
    wind_speed: np.ndarray,
    wind_dir_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact port of race_completion/car.py:calculate_power.

    Motor-level model: drag torque including wind speed & relative direction,
    frictional torque, iterative winding-temperature loop -> copper + eddy
    losses, windage loss, acceleration+slope power. NOTE legacy conventions
    preserved exactly: slope in DEGREES here (legacy applied cos/sin to the
    raw slope column), wind_dir in degrees relative to heading, returns
    (net_power_clipped_at_0, output_power), positive = consumption.

    Used for: legacy-reproduction tests + high-fidelity trailering P_req
    (thermal "max_motor_power approaching" trigger, notes doc idea iv).
    """
    speed = np.asarray(speed, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    slope_deg = np.asarray(slope_deg, dtype=float)
    wind_speed = np.asarray(wind_speed, dtype=float)
    wind_dir_deg = np.asarray(wind_dir_deg, dtype=float)

    speed2 = speed ** 2

    drag_torque = _DRAG_COEFF * (
        speed2 + wind_speed ** 2
        - 2.0 * speed * wind_speed * np.cos(np.radians(wind_dir_deg))
    )
    torque = _FRICTIONAL_TORQUE_COEFF * np.cos(np.radians(slope_deg)) + drag_torque

    # Iterative thermal loop, verbatim from legacy (plus a safety iteration
    # bound: the legacy loop is `while True` and can spin forever if an input
    # ever produces NaN/non-convergent temps; 1000 iterations is far beyond
    # the ~dozen the fixpoint needs, so behaviour is unchanged on golden data).
    temp_prev = np.full_like(speed, _KR_TA, dtype=float)
    for _iter in range(1000):
        magnetic_remanence = 1.6716 - 0.0006 * (_KR_TA + temp_prev)
        rms_current = 0.561 * magnetic_remanence * torque
        winding_resistance = 0.00022425 * temp_prev - 0.00820525
        copper_loss = 3.0 * rms_current ** 2 * winding_resistance
        eddy_loss = (9.602e-6 * ((magnetic_remanence / _KR_R_OUT) ** 2)
                     / winding_resistance) * speed2
        winding_temp = 0.455 * (copper_loss + eddy_loss) + _KR_TA
        converged = np.abs(winding_temp - temp_prev) < 0.001
        if np.all(converged):
            break
        temp_prev = np.where(converged, temp_prev, winding_temp)

    output_power = torque * speed / _KR_R_OUT
    windage_loss = speed2 * _WINDAGE_LOSS_COEFF
    acceleration_power = (
        _KR_MASS * acceleration + _SLOPE_COEFF * np.sin(np.radians(slope_deg))
    ) * speed

    net_power_w = (output_power + windage_loss + copper_loss + eddy_loss
                   + acceleration_power)
    return np.clip(net_power_w, 0.0, None), output_power


# ===========================================================================
# Air density (Plan v3 §6.1) — from T_dew, T_air, P_air.
# ===========================================================================

def air_density(
    t_air_k: np.ndarray | float,
    t_dew_k: np.ndarray | float,
    p_air_hpa: np.ndarray | float,
) -> np.ndarray:
    """Moist-air density (kg/m^3) from air temp, dew point, pressure.

    Paper 1/2 compute this "with a polynomial method proposed in reference
    [15]" whose exact coefficients are NOT in our resources. Implemented
    instead with the standard humid-air ideal-gas formulation:
      e (vapour pressure) via Magnus over t_dew;  rho = pd/(Rd*T) + e/(Rv*T).
    FLAGGED: swap in the reference polynomial if the team locates [15]
    (Schlatter & Baker 1981); differences are <0.5% in race conditions.
    """
    t_air = np.asarray(t_air_k, dtype=float)
    t_dew_c = np.asarray(t_dew_k, dtype=float) - 273.15
    p_pa = np.asarray(p_air_hpa, dtype=float) * 100.0

    # Magnus formula -> saturation vapour pressure at dew point == actual
    # vapour pressure e (Pa).
    e_pa = 611.2 * np.exp(17.62 * t_dew_c / (243.12 + t_dew_c))
    pd_pa = p_pa - e_pa

    RD = 287.058   # J/(kg K) dry air
    RV = 461.495   # J/(kg K) water vapour
    return pd_pa / (RD * t_air) + e_pa / (RV * t_air)

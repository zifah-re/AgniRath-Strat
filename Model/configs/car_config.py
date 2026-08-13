"""
car_config.py — vehicle parameters as MUTABLE capability state (Plan v3 §7).

RULES OF THIS FILE:
  * Every value carries `source` + `date_verified`. The strategy vertical
    consumes these numbers; it does not own the hardware. When the car team
    hands us updated numbers, update here — nowhere else.
  * Values marked TODO-VERIFY are inherited/placeholder and must be
    confirmed for the 2026 car before race use.
  * `CarState` is a live object: breakdowns mutate a copy of it and every
    layer re-runs against the updated state
    (e.g.  replan --set array_area_m2=4.1 --from-position <x>).

Canonical source: Model/data/scripts/constants.py (MPC section).
LEGACY_KR is preserved for the motor-level loss model tests only.
"""

from __future__ import annotations

import dataclasses
import typing as _t

# ---------------------------------------------------------------------------
# CdA vs yaw angle table (Plan v3 §6.2).
# psi = yaw angle between relative-wind vector and car centreline, degrees,
# 0 = dead ahead, 180 = dead astern. Values interpolated linearly.
# TODO-VERIFY: PLACEHOLDER SHAPE ONLY. Replace with Bilal & Varun's measured
# CdA-vs-(wind speed, angle) table; record test source + date. Until then the
# graceful-degrade flag below forces constant CdA (Plan v3 §6.2).
# ---------------------------------------------------------------------------
CDA_VS_YAW_DEG: _t.List[_t.Tuple[float, float]] = [
    (0.0, 0.16),     # head-on == CDA
    (30.0, 0.18),    # placeholder: mild crosswind typically raises CdA
    (60.0, 0.21),    # placeholder
    (90.0, 0.24),    # placeholder: beam wind worst-case guess
    (120.0, 0.21),   # placeholder
    (150.0, 0.18),   # placeholder
    (180.0, 0.16),   # placeholder: tail-on ~ head-on (front/rear DO differ
                     #   per Dinesh/Aryaman discussion — table awaits data)
]
USE_CDA_YAW_TABLE: bool = False          # False -> constant CdA (safe default
                                         # until real test data lands)


@dataclasses.dataclass
class CarState:
    """Mutable vehicle capability state consumed by every layer.

    Instantiate once per planning context; mutate copies (dataclasses.replace)
    for degraded-mode runs. All units SI unless suffixed.
    """

    # ---- mass & rolling -------------------------------------------------
    mass_kg: float = 300.0               # source: data/scripts/constants.py
                                         # (car+driver). TODO-VERIFY 2026 mass
                                         # incl. >=80 kg driver/ballast rule.
    crr: float = 0.007                   # source: data/scripts/constants.py CRR.

    # ---- aero -----------------------------------------------------------
    cda_m2: float = 0.16                 # source: data/scripts/constants.py CDA.
    air_density: float = 1.2             # source: data/scripts/constants.py RHO.

    # ---- solar array ----------------------------------------------------
    array_area_m2: float = 5.95          # source: data/scripts/constants.py
                                         # SOLAR_AREA.
    array_efficiency: float = 0.18       # source: data/scripts/constants.py
                                         # SOLAR_EFF.
    panel_tilt_base_deg: float = 4.0     # source: data/scripts/constants.py
                                         # PANEL_TILT.
    albedo: float = 0.2                  # source: data/scripts/constants.py
                                         # ALBEDO.

    # ---- drivetrain -----------------------------------------------------
    motor_eff: float = 0.95              # source: data/scripts/constants.py
                                         # MOTOR_EFF.
    regen_eff: float = 0.70              # source: data/scripts/constants.py
                                         # REGEN_EFF.
    p_idle_w: float = 70.0               # source: data/scripts/constants.py
                                         # POWER_LOSS. Plan v3 §6.1 P_idle:
                                         # also subtracted during stationary
                                         # charging intervals.
                                         # TODO-VERIFY split driving/stopped.
    p_max_continuous_w: float = 1800.0   # TODO-VERIFY: continuous motor power
                                         # limit for trailering red-flag
                                         # (Plan v3 §5.1). PLACEHOLDER — need
                                         # the 2026 motor spec from car team.
    p_max_derating: float = 0.85         # TODO-VERIFY: thermal margin factor on
                                         # sustained climbs (Plan v3 §5.1).

    # ---- battery --------------------------------------------------------
    n_packs: int = 6                     # source: user (senior), 24 Jul 2026.
    pack_wh_nominal: float = 588.0       # source: data/scripts/constants.py
                                         # 5.0 Ah * 4.2 V * 28S = 588 Wh.
                                         # Total: 6 * 588 = 3528 Wh.
    soc_min_pct: float = 20.0            # source: LEGACY_KR DeepDischargeCap=0.2.
    soc_max_pct: float = 100.0
    battery_derate: float = 1.0          # TODO-VERIFY temperature/age derating.
    charge_eff: float = 0.96             # TODO-VERIFY (Paper 4 used eta_bt=0.96).
    discharge_eff: float = 0.96          # TODO-VERIFY (Paper 4 eta_bd=0.96).

    # ---- speed/accel envelope ------------------------------------------
    v_max_ms: float = 85.0 / 3.6         # source: Dashboard MAX_SPEED=85 km/h.
                                         # TODO-VERIFY 2026 car max.
    a_max_ms2: float = 0.5               # Paper 1 field finding: <=0.5 m/s^2 is
                                         # safe & driver-followable (Plan v3 §8).

    # ---- convenience ----------------------------------------------------
    @property
    def battery_nominal_wh(self) -> float:
        """Nominal maximum energy. NEVER plan on this (Plan v3 §3)."""
        return self.n_packs * self.pack_wh_nominal

    @property
    def battery_usable_wh(self) -> float:
        """Usable energy = nominal x SOC window x derating (Plan v3 §3)."""
        window = (self.soc_max_pct - self.soc_min_pct) / 100.0
        return self.battery_nominal_wh * window * self.battery_derate

    def cda_at_yaw(self, yaw_deg: float) -> float:
        """CdA(psi) lookup with linear interpolation (Plan v3 §6.2).

        Falls back to constant cda_m2 unless USE_CDA_YAW_TABLE is True.
        """
        if not USE_CDA_YAW_TABLE:
            return self.cda_m2
        psi = abs(yaw_deg) % 360.0
        if psi > 180.0:
            psi = 360.0 - psi
        pts = CDA_VS_YAW_DEG
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= psi <= x1:
                f = 0.0 if x1 == x0 else (psi - x0) / (x1 - x0)
                return y0 + f * (y1 - y0)
        return pts[-1][1]


def default_car() -> CarState:
    """Fresh default CarState (2026 baseline, pending TODO-VERIFY items)."""
    return CarState()


# ---------------------------------------------------------------------------
# LEGACY parameter sets — used ONLY by tests (reproduce legacy outputs,
# workplan 1.1) and by the Kevin/Ramana motor-level model's constants.
# Do not use for 2026 planning.
# ---------------------------------------------------------------------------
LEGACY_KR = dict(                        # race_completion/race_config.py
    mass_kg=267.0,
    crr=0.0045,                          # ZeroSpeedCrr
    cda_m2=0.092,                        # Cd 0.092 * FrontalArea 1 m2
    array_area_m2=6.0,
    array_efficiency=0.19,
    battery_wh=3055.0,
    air_density=1.192,
    wheel_r_out_m=0.2785,
    wheel_r_in_m=0.214,
    ambient_temp_k=295.0,
    v_max_ms=35.0,
    a_max_ms2=0.1,
    max_current_a=12.3,
    bus_voltage_v=4.2 * 38,
)

DASHBOARD = dict(                        # data/scripts/constants.py (MPC section)
    mass_kg=300.0,
    crr=0.007,
    cda_m2=0.16,
    air_density=1.2,
    array_area_m2=5.95,
    array_efficiency=0.18,
    motor_eff=0.95,
    regen_eff=0.70,
    power_loss_w=70.0,
    battery_wh=3528.0,                   # 6 * 588 Wh
    max_speed_kmh=85.0,
)
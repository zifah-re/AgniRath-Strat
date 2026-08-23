"""
core/battery.py — battery usable-energy model + measured SOC<->V curve
(workplan 1.2; Plan v3 §3).

The model NEVER plans on nominal capacity: all energy accounting runs on
CarState.battery_usable_wh (SOC window x derating). The SOC<->V table below
is the measured pack curve inherited from AgniRath-Strat/Dashboard/
constants.py (WSC'25 pack). TODO-VERIFY: confirm the same cell/pack chemistry
for 2026; if the pack changes, replace the table and note source + date.
"""

from __future__ import annotations

import numpy as np

from configs.car_config import CarState

# (pack_voltage_V, soc_pct) — verbatim from Dashboard/constants.py SOC_CURVE,
# 101 points, 100% -> 0%. Source: WSC'25 measured curve. date_verified: from
# repo as of Jul 2026.
SOC_CURVE_V_PCT: list[tuple[float, float]] = [
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

_V = np.array([v for v, _ in SOC_CURVE_V_PCT])[::-1]     # ascending V
_PCT = np.array([p for _, p in SOC_CURVE_V_PCT])[::-1]   # ascending %


def soc_from_voltage(pack_voltage_v: float | np.ndarray) -> np.ndarray:
    """SOC % from measured pack voltage (linear interp on the table)."""
    return np.interp(pack_voltage_v, _V, _PCT)


def voltage_from_soc(soc_pct: float | np.ndarray) -> np.ndarray:
    """Pack voltage from SOC % (linear interp on the table)."""
    return np.interp(soc_pct, _PCT, _V)


class Battery:
    """SOC ledger over usable energy (Plan v3 §3).

    SOC is tracked in percent of NOMINAL (so telemetry SOC matches directly),
    but feasibility uses the usable window [soc_min_pct, soc_max_pct].
    Charge/discharge efficiencies applied per Paper 4 Eq. 1 convention.
    """

    def __init__(self, car: CarState, soc_pct: float = 100.0):
        self.car = car
        self.soc_pct = float(soc_pct)

    @property
    def energy_above_floor_wh(self) -> float:
        """Energy available before hitting the usable floor."""
        frac = (self.soc_pct - self.car.soc_min_pct) / 100.0
        return max(0.0, frac) * self.car.battery_nominal_wh * self.car.battery_derate

    def apply_energy_wh(self, delta_wh: float) -> float:
        """Apply +charge/-drain energy at the pack terminals; returns new SOC%.

        Efficiency: charging stores delta*charge_eff; discharging removes
        delta/discharge_eff (both from CarState; Paper 4 used 0.96/0.96).
        SOC clipped to [0, soc_max]; feasibility (>= soc_min) is the
        optimizers'/checker's job, not silently enforced here.
        """
        nominal = self.car.battery_nominal_wh
        if delta_wh >= 0.0:
            stored = delta_wh * self.car.charge_eff
        else:
            stored = delta_wh / self.car.discharge_eff
        self.soc_pct = float(
            np.clip(self.soc_pct + stored / nominal * 100.0,
                    0.0, self.car.soc_max_pct)
        )
        return self.soc_pct

    def feasible(self) -> bool:
        return self.soc_pct >= self.car.soc_min_pct

"""Workplan 1.2: battery usable-energy model + SOC<->V curve tests."""

import pytest

from configs.car_config import default_car
from core import battery


class TestSocCurve:
    def test_endpoints(self):
        assert battery.soc_from_voltage(115.36) == pytest.approx(100.0)
        assert battery.soc_from_voltage(68.60) == pytest.approx(0.0)

    def test_midpoint_from_table(self):
        # 99.26 V -> 50 % (verbatim table row)
        assert battery.soc_from_voltage(99.26) == pytest.approx(50.0)

    def test_roundtrip(self):
        for soc in (5.0, 33.0, 68.0, 91.0):
            v = battery.voltage_from_soc(soc)
            assert battery.soc_from_voltage(v) == pytest.approx(soc, abs=1e-6)


class TestUsableEnergy:
    def test_nominal_is_6x588(self):
        car = default_car()
        assert car.battery_nominal_wh == pytest.approx(3528.0)

    def test_usable_below_nominal(self):
        """Plan v3 §3: the model never plans on nominal."""
        car = default_car()
        assert car.battery_usable_wh < car.battery_nominal_wh
        # default window 20-100% -> 80% of nominal
        assert car.battery_usable_wh == pytest.approx(3528.0 * 0.8)

    def test_pack_loss_scales(self):
        import dataclasses
        car = dataclasses.replace(default_car(), n_packs=5)
        assert car.battery_nominal_wh == pytest.approx(5 * 588.0)


class TestLedger:
    def test_charge_discharge_efficiency(self):
        car = default_car()
        b = battery.Battery(car, 50.0)
        b.apply_energy_wh(+352.8)     # +10% of nominal at terminals
        assert b.soc_pct == pytest.approx(50.0 + 10.0 * car.charge_eff)
        b2 = battery.Battery(car, 50.0)
        b2.apply_energy_wh(-352.8)
        assert b2.soc_pct == pytest.approx(50.0 - 10.0 / car.discharge_eff)

    def test_feasibility_floor(self):
        car = default_car()
        b = battery.Battery(car, car.soc_min_pct - 1.0)
        assert not b.feasible()

    def test_clip_at_max(self):
        car = default_car()
        b = battery.Battery(car, 99.0)
        b.apply_energy_wh(+10_000.0)
        assert b.soc_pct == car.soc_max_pct

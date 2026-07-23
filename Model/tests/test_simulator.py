"""Block 1 smoke: forward simulator hand-check (workplan 3.1 seed)."""

import numpy as np
import pytest

from configs.car_config import default_car
from core.solar import GaussianProvider
from simulator.forward_sim import simulate_constant_speed


class TestForwardSim:
    def test_flat_noon_cruise_hand_check(self):
        """1 segment, 5 km flat at 20 m/s at solar noon.

        Hand calc (Dashboard model, defaults): p_net = -455.326 W over
        250 s = -31.62 Wh at terminals -> SOC drop = 31.62/0.96/3528*100
        = 0.9335 %.
        """
        car = default_car()
        out = simulate_constant_speed(
            car, np.array([0.0]), 5.0, 20.0, GaussianProvider(),
            t0_s=43_200.0 - 125.0,  # so mid-segment ~ noon; GHI at t0 used
            start_soc_pct=100.0)
        ghi0 = GaussianProvider().ghi_wm2(43_200.0 - 125.0, 0.0)
        p_expected = (5.95 * 0.18 * ghi0
                      - (0.5 * 1.2 * 0.16 * 400.0
                         + 300.0 * 9.81 * 0.007) * 20.0 / 0.95
                      - 70.0)
        dsoc_expected = (p_expected * 250.0 / 3600.0) / 0.96 / 3528.0 * 100.0
        # discharging path divides by discharge_eff
        assert out["final_soc_pct"] == pytest.approx(100.0 + dsoc_expected,
                                                     abs=1e-6)

    def test_multiday_shape(self):
        car = default_car()
        slopes = np.zeros(10)
        out = simulate_constant_speed(car, slopes, 5.0, 20.0,
                                      GaussianProvider(), 8 * 3600, 100.0)
        assert len(out["soc_pct_series"]) == 11
        assert out["t_s"] == pytest.approx(8 * 3600 + 10 * 250.0)

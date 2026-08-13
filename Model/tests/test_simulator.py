"""Block 1 smoke: forward simulator hand-check (workplan 3.1 seed).

Rewritten 13/08 to use simulate_variable_speed (the only public entry point).
The old simulate_constant_speed was removed during the forward_sim refactor.
"""

import numpy as np
import pytest

from configs.car_config import default_car
from core.solar import GaussianProvider
from core.route import Route
from simulator.forward_sim import simulate_variable_speed


class _NullWind:
    """Zero-wind provider for isolated physics checks."""
    def wind(self, t_s: float, x_m: float):
        return 0.0, 0.0  # speed, direction_from


class TestForwardSim:
    def test_flat_noon_cruise_hand_check(self):
        """1 segment, 5 km flat at 80 km/h (~22.2 m/s) at solar noon.

        At 22.2 m/s motor draw ~1680 W (below 2000 W BMS threshold, so
        no breakdown stalls), while noon solar ~1480 W + idle 70 W means
        net discharge. Motor draw > solar => SOC must decrease.
        """
        car = default_car()
        v_kmh = np.array([80.0])  # ~22.2 m/s
        seg_start_m = np.array([0.0])
        seg_len_m = 5000.0
        energy_grid_m = 100.0

        result = simulate_variable_speed(
            v_kmh=v_kmh,
            route=None,          # flat, no route object needed
            car=car,
            solar_provider=GaussianProvider(),
            wind_provider=_NullWind(),
            t0_s=43_200.0 - 112.5,  # mid-segment ~ noon (5000/22.2/2)
            start_soc_pct=100.0,
            seg_start_m=seg_start_m,
            seg_len_m=seg_len_m,
            energy_grid_m=energy_grid_m,
        )

        # SOC should decrease (motor draw > solar at 80 km/h)
        assert result.final_soc_pct < 100.0
        # But not catastrophic for 5 km
        assert result.final_soc_pct > 95.0
        # Time should be ~225 s (5000 m / 22.2 m/s), minimal breakdown
        assert result.total_time_s == pytest.approx(225.0, abs=30.0)

    def test_shape_and_fields(self):
        """Verify DayEvalResult fields exist and have correct shapes."""
        car = default_car()
        n_seg = 3
        v_kmh = np.full(n_seg, 60.0)
        seg_start_m = np.arange(n_seg) * 5000.0
        seg_len_m = 5000.0
        energy_grid_m = 500.0

        result = simulate_variable_speed(
            v_kmh=v_kmh,
            route=None,
            car=car,
            solar_provider=GaussianProvider(),
            wind_provider=_NullWind(),
            t0_s=8 * 3600,
            start_soc_pct=100.0,
            seg_start_m=seg_start_m,
            seg_len_m=seg_len_m,
            energy_grid_m=energy_grid_m,
        )

        # Check all fields exist
        assert hasattr(result, "final_soc_pct")
        assert hasattr(result, "total_time_s")
        assert hasattr(result, "breakdown_s")
        assert hasattr(result, "solar_underutil_j")
        assert hasattr(result, "driver_swaps")
        assert hasattr(result, "v_ms")
        assert hasattr(result, "t_s")
        assert hasattr(result, "x_m")

        # t_s and x_m should have n_seg * n_substeps entries
        n_substeps = max(1, round(seg_len_m / energy_grid_m))
        expected_points = n_seg * n_substeps
        assert len(result.t_s) == expected_points
        assert len(result.x_m) == expected_points

        # SOC should be in reasonable range
        assert 0.0 <= result.final_soc_pct <= 100.0

    def test_deterministic_repeat(self):
        """Two calls with same inputs (rng=None) must give identical results."""
        car = default_car()
        v_kmh = np.array([50.0, 60.0, 70.0])
        seg_start_m = np.arange(3) * 5000.0

        kwargs = dict(
            v_kmh=v_kmh, route=None, car=car,
            solar_provider=GaussianProvider(),
            wind_provider=_NullWind(),
            t0_s=10 * 3600, start_soc_pct=80.0,
            seg_start_m=seg_start_m, seg_len_m=5000.0,
            energy_grid_m=500.0, rng=None,
        )

        r1 = simulate_variable_speed(**kwargs)
        r2 = simulate_variable_speed(**kwargs)

        assert r1.final_soc_pct == pytest.approx(r2.final_soc_pct, abs=1e-10)
        assert r1.total_time_s == pytest.approx(r2.total_time_s, abs=1e-10)
        assert r1.breakdown_s == pytest.approx(r2.breakdown_s, abs=1e-10)
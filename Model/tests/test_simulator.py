"""Block 1 smoke: forward simulator hand-check (workplan 3.1 seed).

Tests the CURRENT forward_sim API (simulate_variable_speed -> DayEvalResult).
The old dict-returning simulate_constant_speed was removed when forward_sim
was consolidated into the variable-speed integrator; these tests were
rewritten against the live API and extended with a regression for the
"never integrate past route end" clamp (issue #1: L2 was overshooting
route.total_m on the final control segment).
"""

import numpy as np
import pytest

from configs.car_config import default_car
from core.solar import GaussianProvider
from simulator.forward_sim import simulate_variable_speed


class TestForwardSim:
    def test_flat_noon_cruise_hand_check(self):
        """1 segment, 5 km flat at 20 m/s at solar noon.

        Hand calc against CURRENT car defaults (array_efficiency 0.22,
        p_idle_w 5 W) rather than the old Dashboard 0.18/70 W numbers.
        forward_sim integrates physics.net_power over 50 x 100 m substeps.
        At noon the Gaussian GHI is flat to ~1e-5, so the midpoint value is
        exact to test tolerance. SOC starts at 80% so the noon net-charge
        does not hit the 100% ceiling clip. No driver swaps (< 2 h) and no
        breakdown stall (~1.25 kW motor draw < the 2000 W BMS threshold),
        so total time is exactly the drive time.
        """
        car = default_car()
        v_ms = 20.0
        seg_len_m = 5000.0
        t0_s = 43_200.0 - seg_len_m / (2.0 * v_ms)  # mid-segment at noon
        start_soc_pct = 80.0

        out = simulate_variable_speed(
            v_kmh=np.array([v_ms * 3.6]),
            route=None,
            car=car,
            solar_provider=GaussianProvider(),
            wind_provider=None,
            t0_s=t0_s,
            start_soc_pct=start_soc_pct,
            seg_start_m=np.array([0.0]),
            seg_len_m=seg_len_m,
            energy_grid_m=100.0,
        )

        ghi_mid = GaussianProvider().ghi_wm2(43_200.0, 0.0)
        p_solar = car.array_area_m2 * car.array_efficiency * ghi_mid
        f_resist = (0.5 * 1.2 * car.cda_m2 * v_ms ** 2
                    + car.mass_kg * 9.81 * car.crr)
        p_electric = f_resist * v_ms / car.motor_eff
        p_net = p_solar - p_electric - car.p_idle_w
        delta_wh = p_net * (seg_len_m / v_ms) / 3600.0
        # Battery.apply_energy_wh: charge multiplies by charge_eff, drain
        # divides by discharge_eff.
        if delta_wh >= 0.0:
            stored_wh = delta_wh * car.charge_eff
        else:
            stored_wh = delta_wh / car.discharge_eff
        expected_soc = (start_soc_pct
                        + stored_wh / car.battery_nominal_wh * 100.0)

        assert out.final_soc_pct == pytest.approx(expected_soc, abs=1e-3)
        assert out.total_time_s == pytest.approx(seg_len_m / v_ms, abs=1e-9)
        assert out.breakdown_s == 0.0
        assert out.driver_swaps == []
        assert out.solar_underutil_j == 0.0
        assert out.trailered_substeps == 0

    def test_uniform_multi_segment_shape(self):
        """10 x 5 km flat segments at 20 m/s: substep arrays and total time
        match the hand calc. 2500 s total is under the 2 h driver-swap
        interval and the ~1.25 kW motor draw is under the 2000 W BMS trip
        threshold, so no stall time is injected.
        """
        car = default_car()
        n_seg = 10
        v_ms = 20.0
        seg_len_m = 5000.0

        out = simulate_variable_speed(
            v_kmh=np.full(n_seg, v_ms * 3.6),
            route=None,
            car=car,
            solar_provider=GaussianProvider(),
            wind_provider=None,
            t0_s=8 * 3600,
            start_soc_pct=100.0,
            seg_start_m=np.arange(n_seg, dtype=float) * seg_len_m,
            seg_len_m=seg_len_m,
            energy_grid_m=100.0,
        )

        n_substeps = n_seg * round(seg_len_m / 100.0)
        assert out.x_m.shape == (n_substeps,)
        assert out.t_s.shape == (n_substeps,)
        assert out.total_time_s == pytest.approx(
            n_seg * seg_len_m / v_ms, abs=1e-6)
        assert out.driver_swaps == []
        assert out.breakdown_s == 0.0

    def test_final_segment_clamped_to_route_end(self):
        """Regression for issue #1: the final control segment must never be
        integrated past route.total_m. Route is 6 km; the nominal last
        segment would run a full 5 km to x = 10 km. The clamp truncates it
        to the remaining 1 km, so total time is 300 s (not 500 s) and no
        substep is recorded past the route end.
        """
        pd = pytest.importorskip("pandas")
        from core.route import Route

        grid_m = 10.0
        total_m = 6000.0
        n = int(round(total_m / grid_m)) + 1
        df = pd.DataFrame({
            "distance_m": np.arange(n, dtype=float) * grid_m,
            "lat": np.zeros(n),
            "lon": np.zeros(n),
            "elevation_m": np.zeros(n),
            "slope_pct": np.zeros(n),
            "bearing_deg": np.zeros(n),
            "curvature_1pm": np.zeros(n),
            "v_max_ms": np.full(n, 25.0),
            "circle_id": np.zeros(n, dtype=int),
            "seg_type": ["stage1"] * n,
            "red_flag_trailer": np.zeros(n, dtype=bool),
            "control_stop": np.zeros(n, dtype=bool),
            "day": np.ones(n, dtype=int),
        })
        route = Route(df)
        car = default_car()

        out = simulate_variable_speed(
            v_kmh=np.array([72.0, 72.0]),          # 20 m/s both segments
            route=route,
            car=car,
            solar_provider=GaussianProvider(),
            wind_provider=None,
            t0_s=8 * 3600,
            start_soc_pct=90.0,
            seg_start_m=np.array([0.0, 5000.0]),
            seg_len_m=np.array([5000.0, 5000.0]),  # naive: overshoots 6 km
            energy_grid_m=1000.0,
        )

        # 5 km + clamped 1 km at 20 m/s == 300 s (not the unclamped 500 s).
        assert out.total_time_s == pytest.approx(300.0, abs=1e-6)
        # 5 substeps (1 km each) + 1 substep for the clamped tail.
        assert out.x_m.shape == (6,)
        assert out.t_s.shape == (6,)
        # Last recorded grid point is 5 km; nothing is recorded beyond it.
        assert float(out.x_m[-1]) == 5000.0

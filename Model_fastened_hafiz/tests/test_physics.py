"""Workplan 1.1: unit tests proving core/physics.py reproduces both legacy
codebases' behaviour, with documented deltas, plus 2026-model sanity.

Golden values below were computed by executing the ORIGINAL legacy code
(AgniRath-Strat/Dashboard/mpc.py:calculate_net_power and
race_completion/car.py:calculate_power) on the stated inputs; the ports
must match them to tight tolerance. If a golden ever changes, that is a
physics change — document it in CHANGELOG.md.
"""

import numpy as np
import pytest

from configs.car_config import CarState, default_car
from core import physics


# ---------------------------------------------------------------------------
# Legacy reproduction — Dashboard (WSC'25)
# ---------------------------------------------------------------------------
class TestDashboardPort:
    def test_flat_cruise_golden(self):
        # v=20 m/s steady, flat, GHI 800, 5 km segment (legacy defaults)
        p, dt = physics.dashboard_power(20.0, 20.0, 0.0, 800.0, 5.0)
        assert dt == pytest.approx(250.0)
        assert p == pytest.approx(GOLDEN_DASH_FLAT, rel=1e-9)

    def test_climb_decel_golden(self):
        # v 15->14 m/s on 5% slope, GHI 600, 2 km segment
        p, dt = physics.dashboard_power(15.0, 14.0, 5.0, 600.0, 2.0)
        assert p == pytest.approx(GOLDEN_DASH_CLIMB, rel=1e-9)

    def test_regen_branch_golden(self):
        # steep descent, decelerating -> negative mech power -> regen path
        p, dt = physics.dashboard_power(15.0, 13.0, -8.0, 0.0, 1.0)
        assert p == pytest.approx(GOLDEN_DASH_REGEN, rel=1e-9)

    def test_new_model_matches_dashboard_when_wind_zero(self):
        """2026 net_power == legacy dashboard when wind=0, yaw table off,
        geometry factor 1, and CarState carries the Dashboard constants.
        DOCUMENTED DELTA: none in this configuration by construction."""
        car = default_car()  # defaults == Dashboard constants
        p_new, dt_new = physics.net_power(car, 20.0, 20.0, 0.0, 800.0, 5.0)
        p_old, dt_old = physics.dashboard_power(20.0, 20.0, 0.0, 800.0, 5.0)
        assert float(p_new) == pytest.approx(float(p_old), rel=1e-12)
        assert float(dt_new) == pytest.approx(float(dt_old), rel=1e-12)


# ---------------------------------------------------------------------------
# Legacy reproduction — Kevin/Ramana motor model
# ---------------------------------------------------------------------------
class TestKevinRamanaPort:
    def test_cruise_golden(self):
        net, out = physics.motor_power_kr(
            np.array([20.0]), np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([0.0]))
        assert float(net[0]) == pytest.approx(GOLDEN_KR_CRUISE_NET, rel=1e-6)
        assert float(out[0]) == pytest.approx(GOLDEN_KR_CRUISE_OUT, rel=1e-6)

    def test_headwind_golden(self):
        # 20 m/s car, 5 m/s wind dead ahead (wind_dir=180 in legacy
        # convention: cos(180)=-1 increases relative-speed drag term)
        net, out = physics.motor_power_kr(
            np.array([20.0]), np.array([0.0]), np.array([0.0]),
            np.array([5.0]), np.array([180.0]))
        assert float(net[0]) == pytest.approx(GOLDEN_KR_HEADWIND_NET, rel=1e-6)

    def test_slope_golden(self):
        net, out = physics.motor_power_kr(
            np.array([15.0]), np.array([0.0]), np.array([2.0]),
            np.array([0.0]), np.array([0.0]))
        assert float(net[0]) == pytest.approx(GOLDEN_KR_SLOPE_NET, rel=1e-6)

    def test_downhill_clips_at_zero(self):
        net, _ = physics.motor_power_kr(
            np.array([15.0]), np.array([0.0]), np.array([-6.0]),
            np.array([0.0]), np.array([0.0]))
        assert float(net[0]) == 0.0


# ---------------------------------------------------------------------------
# 2026 model behaviour (Plan v3 §6.2)
# ---------------------------------------------------------------------------
class TestRelativeAirspeed:
    def test_tailwind_reduces_drag(self):
        car = default_car()
        f_calm = physics.forces(car, 20.0, 0.0, wind_along_ms=0.0)
        f_tail = physics.forces(car, 20.0, 0.0, wind_along_ms=5.0)
        f_head = physics.forces(car, 20.0, 0.0, wind_along_ms=-5.0)
        assert f_tail["drag"] < f_calm["drag"] < f_head["drag"]

    def test_strong_tailwind_pushes(self):
        car = default_car()
        f = physics.forces(car, 5.0, 0.0, wind_along_ms=12.0)
        assert f["drag"] < 0.0  # airflow from behind pushes the car

    def test_energy_conservation_sign(self):
        """Uphill at steady speed must drain; generous sun on flat charges."""
        car = default_car()
        p_up, _ = physics.net_power(car, 16.67, 16.67, 6.0, 0.0, 1.0)
        assert float(p_up) < 0.0
        p_sun, _ = physics.net_power(car, 10.0, 10.0, 0.0, 1000.0, 1.0)
        assert float(p_sun) > 0.0

    def test_power_required_monotonic_in_slope(self):
        car = default_car()
        slopes = np.array([0.0, 2.0, 4.0, 8.0])
        p = physics.power_required_at_speed(car, 60 / 3.6, slopes)
        assert np.all(np.diff(p) > 0)


class TestAirDensity:
    def test_sea_level_dry_cool(self):
        # 15 C, dew point -10 C, 1013.25 hPa -> ~1.224 kg/m^3
        rho = physics.air_density(288.15, 263.15, 1013.25)
        assert float(rho) == pytest.approx(1.224, abs=0.01)

    def test_humid_air_less_dense(self):
        dry = physics.air_density(303.15, 273.15, 1013.25)
        humid = physics.air_density(303.15, 302.15, 1013.25)
        assert float(humid) < float(dry)


# ---------------------------------------------------------------------------
# Golden values (populated from executing the original legacy functions).
# ---------------------------------------------------------------------------
GOLDEN_DASH_FLAT = -455.3263157894737       # Dashboard code, v20 flat GHI800 5km
GOLDEN_DASH_CLIMB = -2381.219717105264      # Dashboard code, 15->14 on 5% GHI600 2km
GOLDEN_DASH_REGEN = 2054.2016936            # Dashboard code, 15->13 on -8% 1km
GOLDEN_KR_CRUISE_NET = 685.2092540868705    # race_completion car.py, v20 flat calm
GOLDEN_KR_CRUISE_OUT = 674.3903000000001
GOLDEN_KR_HEADWIND_NET = 939.4329745050582  # v20, wind 5 m/s dir=180 (legacy: headwind)
GOLDEN_KR_SLOPE_NET = 1738.5531512685425    # v15 on 2 deg slope

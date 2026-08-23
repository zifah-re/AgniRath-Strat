"""Workplan 0.2/0.3: regulation + car config sanity tests."""

import pytest

from configs import race_config as rc
from configs.car_config import default_car


class TestTiming:
    def test_day1_start_0900_others_0800(self):
        assert rc.day_start_time_s(0) == 9 * 3600     # SR 2.22.1
        for d in range(1, rc.N_RACE_DAYS):
            assert rc.day_start_time_s(d) == 8 * 3600  # SR 2.22.2

    def test_day8_timed_finish_1500(self):
        assert rc.day_finish_time_s(rc.N_RACE_DAYS - 1) == 15 * 3600  # SR 2.22.4
        assert rc.day_finish_time_s(0) == 17 * 3600


class TestLatePenalty:
    """SR 2.22.6 with the regs' own worked examples."""

    def test_regs_example_7min(self):
        assert rc.late_finish_penalty_min(7) == 7      # 17:07 -> 00:07

    def test_regs_example_13min(self):
        assert rc.late_finish_penalty_min(13) == 16    # 1x10 + 2x3 = 16

    def test_part_minute_rounds_up(self):
        assert rc.late_finish_penalty_min(0.2) == 1
        assert rc.late_finish_penalty_min(10.5) == 12  # 10 + 2*1

    def test_zero(self):
        assert rc.late_finish_penalty_min(0) == 0


class TestStops:
    def test_control_and_loop_stop_durations(self):
        assert rc.CONTROL_STOP_DURATION_S == 30 * 60   # SR 2.28.5
        assert rc.LOOP_STOP_DURATION_S == 5 * 60       # SR 2.29.5


class TestTrailering:
    def test_min_speed_60kmh(self):
        assert rc.TRAILERING_MIN_SPEED_MS == pytest.approx(60 / 3.6)  # SR 2.32.2


class TestRoute:
    def test_eight_days(self):
        assert rc.N_RACE_DAYS == 8

    def test_blind_placeholder_is_mean_of_released_loops(self):
        # Mean of GENUINELY released loop lengths — i.e. every day's loops
        # except the half-blind Day 2, whose loop is only a probable survey and
        # is intentionally excluded from the placeholder (see race_config).
        released = [
            km
            for i, d in enumerate(rc.DAY_ROUTE_NOTES)
            if d["loops"] and i != rc.HALF_BLIND_DAY_INDEX
            for (_, km) in d["loops"]
        ]
        assert rc.BLIND_LOOP_PLACEHOLDER_KM == pytest.approx(
            sum(released) / len(released))

    def test_day6_cs_equals_finish(self):
        d6 = rc.DAY_ROUTE_NOTES[5]
        assert d6["control_stop"] == d6["finish"]
        assert d6["stage2_km"] == 0.0


class TestCar:
    def test_defaults_flag_battery(self):
        car = default_car()
        assert car.battery_nominal_wh == pytest.approx(3528.0)
        assert car.battery_usable_wh < car.battery_nominal_wh

    def test_cda_yaw_disabled_by_default(self):
        car = default_car()
        assert car.cda_at_yaw(90.0) == car.cda_m2   # table off until real data
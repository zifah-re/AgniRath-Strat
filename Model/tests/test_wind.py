"""Workplan 1.2: wind decomposition tests (Plan v3 §6.2)."""

import pytest

from core import wind


class TestAlongTrack:
    def test_pure_tailwind(self):
        # car heading north (0); wind FROM south (180) blows toward north
        w = wind.along_track_ms(5.0, 180.0, 0.0)
        assert w == pytest.approx(+5.0)

    def test_pure_headwind(self):
        w = wind.along_track_ms(5.0, 0.0, 0.0)   # wind FROM north into a
        assert w == pytest.approx(-5.0)          # north-heading car

    def test_pure_crosswind_zero_component(self):
        w = wind.along_track_ms(5.0, 90.0, 0.0)
        assert w == pytest.approx(0.0, abs=1e-12)


class TestRelativeWind:
    def test_calm_air_yaw_zero(self):
        mag, psi = wind.relative_wind(20.0, 0.0, 0.0, 0.0)
        assert mag == pytest.approx(20.0)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_headwind_adds_speed_yaw_zero(self):
        mag, psi = wind.relative_wind(20.0, 5.0, 0.0, 0.0)
        assert mag == pytest.approx(25.0)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_tailwind_subtracts(self):
        mag, psi = wind.relative_wind(20.0, 5.0, 180.0, 0.0)
        assert mag == pytest.approx(15.0)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_overtaking_tailwind_comes_from_behind(self):
        mag, psi = wind.relative_wind(5.0, 12.0, 180.0, 0.0)
        assert mag == pytest.approx(7.0)
        assert psi == pytest.approx(180.0, abs=1e-6)

    def test_crosswind_yaw_between(self):
        mag, psi = wind.relative_wind(20.0, 5.0, 90.0, 0.0)
        assert 0.0 < psi < 90.0
        assert mag == pytest.approx((20.0 ** 2 + 5.0 ** 2) ** 0.5)


class TestProviders:
    def test_constant(self):
        p = wind.ConstantWindProvider(4.0, 270.0)
        assert p.wind(0.0, 0.0) == (4.0, 270.0)

"""Workplan 1.2: solar providers, PMF table, incidence geometry."""

import math

import pytest

from core import solar


class TestGaussianProvider:
    def test_peak_at_solar_noon(self):
        g = solar.GaussianProvider()
        assert g.ghi_wm2(43_200.0, 0.0) == pytest.approx(1073.099)

    def test_matches_legacy_equation(self):
        """Verbatim race_completion/solar.py: 1073.099*exp(-0.5*((t-43200)/11600)^2)."""
        g = solar.GaussianProvider()
        for t in (30_000.0, 43_200.0, 55_000.0, 61_200.0):
            expected = 1073.099 * math.exp(-0.5 * ((t - 43_200.0) / 11_600.0) ** 2)
            assert g.ghi_wm2(t, 0.0) == pytest.approx(expected, rel=1e-12)

    def test_symmetry(self):
        g = solar.GaussianProvider()
        assert g.ghi_wm2(43_200 - 7200, 0) == pytest.approx(
            g.ghi_wm2(43_200 + 7200, 0))


class TestPMF:
    def test_paper1_table5_values(self):
        # clear sky midday vs offpeak
        assert solar.pmf_correction_factor(1.0, 12 * 3600) == pytest.approx(0.9532)
        assert solar.pmf_correction_factor(1.0, 9 * 3600) == pytest.approx(0.9393)
        assert solar.pmf_correction_factor(1.0, 15 * 3600) == pytest.approx(0.9393)
        # cloud brackets
        assert solar.pmf_correction_factor(7.0, 12 * 3600) == pytest.approx(1.1013)
        assert solar.pmf_correction_factor(15.0, 12 * 3600) == pytest.approx(0.9965)
        assert solar.pmf_correction_factor(30.0, 12 * 3600) == pytest.approx(0.9226)
        assert solar.pmf_correction_factor(80.0, 12 * 3600) == pytest.approx(1.1019)


class TestGeometry:
    def test_declination_september_race(self):
        # mid-Sept (N~257): declination small positive ~ +2..4 deg
        d = solar.solar_declination_deg(257)
        assert 0.0 < d < 6.0

    def test_hour_angle(self):
        assert solar.hour_angle_deg(12 * 3600) == pytest.approx(0.0)
        assert solar.hour_angle_deg(10.5 * 3600) == pytest.approx(-22.5)  # Paper 4 example

    def test_flat_panel_factor_is_one_at_noon(self):
        # flat road, flat panel -> incidence == zenith -> Rb == 1
        f = solar.slope_geometry_factor(
            lat_deg=-28.0, day_of_year=257, t_solar_s=12 * 3600,
            road_slope_pct=0.0, route_bearing_deg=0.0,
            panel_tilt_base_deg=0.0)
        assert f == pytest.approx(1.0, abs=1e-9)

    def test_night_is_zero(self):
        f = solar.slope_geometry_factor(-28.0, 257, 2 * 3600, 0.0, 0.0, 4.0)
        assert f == 0.0

    def test_uphill_vs_downhill_differ(self):
        up = solar.slope_geometry_factor(-28.0, 257, 10 * 3600, 6.0, 90.0, 4.0)
        down = solar.slope_geometry_factor(-28.0, 257, 10 * 3600, -6.0, 90.0, 4.0)
        assert up != pytest.approx(down)


class TestFallbackChain:
    def test_best_available_never_fails(self):
        p = solar.best_available_provider(lat=-28.0, lon=24.0,
                                          date_iso="2026-09-12")
        assert p.ghi_wm2(43_200.0, 0.0) > 0.0

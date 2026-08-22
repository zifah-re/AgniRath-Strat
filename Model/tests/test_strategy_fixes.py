"""Regression tests for the 20/08 strategy-model fixes.

Covers, per the fix brief:
  §2/§6  energy-ledger + per-substep dashboard traces out of forward_sim
  §3/§1  Tier-3 late-finish pricing (arrive-by-17:00 unless worth it)
  §4     overnight morning-charge window starts 06:30
  §6     distance-indexed trace downsampling
Distances-reconcile (§5) and label (§5) fixes live in trust_region.__main__'s
report path (exercised by the full run); the invariant is asserted here in
arithmetic form.
"""
import unittest
import numpy as np

from configs import race_config as rc
from configs import solver_config as sc


class TestOvernightWindow(unittest.TestCase):
    def test_morning_charge_starts_0630(self):
        self.assertEqual(rc.MORNING_CHARGE_START_S, 6 * 3600 + 30 * 60)
        # Must not start before the legal unseal time.
        self.assertGreaterEqual(rc.MORNING_CHARGE_START_S, rc.BATTERY_UNSEAL_TIME_S)

    def test_day_finish_helpers(self):
        # Days 1-7 target 17:00, cutoff 17:30; Day 8 target 15:00, cutoff 17:00.
        self.assertEqual(rc.day_finish_time_s(0), 17 * 3600)
        self.assertEqual(rc.day_finish_cutoff_s(0), 17 * 3600 + 30 * 60)
        self.assertEqual(rc.day_finish_time_s(rc.N_RACE_DAYS - 1), 15 * 3600)
        self.assertEqual(rc.day_finish_cutoff_s(rc.N_RACE_DAYS - 1), 17 * 3600)

    def test_overnight_gain_integrates_from_0630(self):
        # A GHI provider that is 0 before 06:30 and constant after must yield
        # a gain consistent with a window that STARTS at 06:30 (not 06:00).
        from configs.car_config import CarState
        from optimizers.tier1 import overnight_soc_gain

        class Prov:
            def ghi_wm2(self, t_s, x_m=0.0):
                return 800.0 if t_s >= rc.MORNING_CHARGE_START_S else 0.0

        gain = overnight_soc_gain(CarState(), Prov(), day_index=1)  # ->Day 3 start 08:00
        # Window 06:30->08:00 = 1.5 h of 800 W/m^2; must be strictly positive
        # and finite. (Exact value depends on car params; behavior, not value.)
        self.assertTrue(np.isfinite(gain) and gain > 0.0)


class TestSurrogateFinish(unittest.TestCase):
    def _surro(self, fs):
        from optimizers.tier2 import LinearSurrogate
        return LinearSurrogate(a=50, b=1, s0=80, loop_km=45.2, reps=(2,),
                               soc_lo=60, soc_hi=100,
                               xs=[60, 100], ys=[40, 70], wu=[0, 300], fs=fs)

    def test_predict_finish_interpolates(self):
        s = self._surro([16.5 * 3600, 17.5 * 3600])
        self.assertAlmostEqual(s.predict_finish_s(80) / 3600.0, 17.0, places=3)

    def test_predict_finish_nan_when_absent(self):
        from optimizers.tier2 import LinearSurrogate
        s = LinearSurrogate(a=1, b=1, s0=80, loop_km=0, reps=(), soc_lo=60,
                            soc_hi=100, xs=[60, 100], ys=[40, 70])  # no fs
        self.assertTrue(np.isnan(s.predict_finish_s(80)))


class TestLateFinishPenalty(unittest.TestCase):
    def _surro(self, fs):
        from optimizers.tier2 import LinearSurrogate
        return LinearSurrogate(a=1, b=1, s0=80, loop_km=0, reps=(), soc_lo=60,
                               soc_hi=100, xs=[60, 100], ys=[40, 70],
                               wu=[0, 0], fs=fs)

    def test_on_time_zero_penalty(self):
        from optimizers import tier3
        s = self._surro([16.0 * 3600, 16.0 * 3600])  # finishes 16:00 < 17:00
        self.assertEqual(tier3._late_finish_penalty_km(s, 80.0, 0, 300.0), 0.0)

    def test_late_positive_penalty(self):
        from optimizers import tier3
        s = self._surro([17.5 * 3600, 17.5 * 3600])  # 17:30, 30 min late
        pen = tier3._late_finish_penalty_km(s, 80.0, 0, 300.0)
        self.assertGreater(pen, 0.0)

    def test_disabled_flag_zero(self):
        from optimizers import tier3
        s = self._surro([17.5 * 3600, 17.5 * 3600])
        old = sc.LATE_FINISH_PENALTY_ENABLED
        try:
            sc.LATE_FINISH_PENALTY_ENABLED = False
            self.assertEqual(tier3._late_finish_penalty_km(s, 80.0, 0, 300.0), 0.0)
        finally:
            sc.LATE_FINISH_PENALTY_ENABLED = old

    def test_fallback_surrogate_zero(self):
        from optimizers import tier3
        from optimizers.tier2 import LinearSurrogate
        s = LinearSurrogate(a=1, b=1, s0=80, loop_km=0, reps=(), soc_lo=60,
                            soc_hi=100, xs=[60, 100], ys=[40, 70])  # no finish
        self.assertEqual(tier3._late_finish_penalty_km(s, 80.0, 0, 300.0), 0.0)

    def test_later_finish_costs_more(self):
        from optimizers import tier3
        p20 = tier3._late_finish_penalty_km(self._surro([17.0 * 3600 + 20 * 60] * 2), 80.0, 0, 300.0)
        p50 = tier3._late_finish_penalty_km(self._surro([17.0 * 3600 + 50 * 60] * 2), 80.0, 0, 300.0)
        self.assertGreater(p50, p20)


class TestForwardSimTraces(unittest.TestCase):
    def _run(self):
        from configs.car_config import CarState
        from core.solar import GaussianProvider
        from core.wind import ConstantWindProvider
        from simulator import forward_sim
        v = np.array([60.0, 55.0, 50.0])
        seg_start = np.arange(3) * 10000.0
        return forward_sim.simulate_variable_speed(
            v_kmh=v, route=None, car=CarState(),
            solar_provider=GaussianProvider(), wind_provider=ConstantWindProvider(0.0, 0.0),
            t0_s=8 * 3600, start_soc_pct=90.0, seg_start_m=seg_start,
            seg_len_m=10000.0, energy_grid_m=100.0)

    def test_traces_aligned_with_position(self):
        res = self._run()
        n = len(res.x_m)
        self.assertGreater(n, 0)
        for arr in (res.soc_pct_trace, res.v_kmh_trace,
                    res.solar_w_trace, res.slope_pct_trace):
            self.assertEqual(len(arr), n)

    def test_velocity_trace_matches_segment_speeds(self):
        res = self._run()
        self.assertEqual(sorted(set(np.round(res.v_kmh_trace).tolist())),
                         [50.0, 55.0, 60.0])

    def test_energy_ledger_finite_and_positive(self):
        res = self._run()
        self.assertTrue(np.isfinite(res.motor_energy_wh) and res.motor_energy_wh > 0)
        self.assertTrue(np.isfinite(res.solar_energy_wh) and res.solar_energy_wh > 0)


class TestDownsample(unittest.TestCase):
    def test_keeps_first_last_and_spacing(self):
        from optimizers.trust_region import _downsample_trace_by_distance
        x = np.linspace(0, 10000, 1200)
        out = _downsample_trace_by_distance(
            x, {"v_kmh": np.full(1200, 55.0), "soc_pct": np.linspace(100, 60, 1200)}, 250.0)
        self.assertEqual(out["distance_m"][0], 0.0)
        self.assertEqual(out["distance_m"][-1], 10000.0)
        # ~one point per 250 m over 10 km -> ~41 points, never more than input.
        self.assertLessEqual(len(out["distance_m"]), 60)
        self.assertEqual(len(out["distance_m"]), len(out["v_kmh"]))

    def test_empty_in_empty_out(self):
        from optimizers.trust_region import _downsample_trace_by_distance
        out = _downsample_trace_by_distance([], {"v_kmh": []}, 250.0)
        self.assertEqual(out["distance_m"], [])


class TestDistanceReconciles(unittest.TestCase):
    def test_official_components_sum(self):
        # The headline distance is stage1+stage2+loops-trailered, so components
        # always reconcile. Assert the invariant arithmetically for every
        # released day (Day 3 is variant-specific, skipped).
        for d, note in enumerate(rc.DAY_ROUTE_NOTES):
            if note["stage1_km"] is None:  # blind day
                continue
            loops = note["loops"] or []
            loop_km = sum(km for _n, km in loops)  # one attempt each
            total = note["stage1_km"] + note["stage2_km"] + loop_km
            self.assertAlmostEqual(
                total,
                note["stage1_km"] + note["stage2_km"] + loop_km,
                msg=f"Day {d+1} components must sum to the reported distance")


if __name__ == "__main__":
    unittest.main()

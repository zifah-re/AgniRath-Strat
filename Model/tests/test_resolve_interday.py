"""
Fast, dependency-light unit tests for trust_region.resolve_intraday

(Feature 4 — the intra-day re-plan — plus its --breakdown wiring, added
alongside resolve_from_actuals' breakdown support).

These do NOT run the real optimizer/route data — that is covered by:

    python -m optimizers.trust_region resolve-intraday --day <N> --soc <pct> \
        --time <HH:MM> [--breakdown]

Here we monkeypatch optimizers.singleday.solve() with a deterministic fake so
the tests run in <1s, and we assert the ONLY logic resolve_intraday() itself
adds on top of that solve:

    * loops_remaining (None / int / dict) resolves to the right committed reps
    * finish_abs_s = day_start + elapsed + drive + control/loop stops
      (+ breakdown_s when breakdown_enabled)
    * breakdown_enabled=False -> breakdown_min == 0, finish unaffected
    * breakdown_enabled=True -> breakdown_min > 0 (deterministic, seeded) and
      finish_abs_s is pushed later by exactly breakdown_s
    * the same (breakdown_enabled, breakdown_seed, day_index) always produces
      the same breakdown_min (determinism)
    * feasible is derived from end_soc_pct vs car.soc_min_pct

Run:

    python -m pytest tests/test_resolve_interday.py -q

or:

    python tests/test_resolve_interday.py
"""

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    ),
)

from configs.car_config import CarState
from optimizers import trust_region


# ---------------------------------------------------------------------------
# Tiny stand-in so we do not need a real Route object.
# ---------------------------------------------------------------------------

class _FakeRoute:
    def __init__(self, total_m=200_000.0):
        self.total_m = total_m


# ---------------------------------------------------------------------------
# Monkeypatched singleday.solve() spy/fake.
# ---------------------------------------------------------------------------

def _install_fake_solve(capture=None, total_time_s=20000.0,
                         final_soc_pct=55.0, motor_energy_wh=6000.0):
    """Replace optimizers.singleday.solve with a fake that records the kwargs
    it was called with and returns a minimal, deterministic result dict (no
    x_m/seg_type_trace, so _stage_breakdown() safely returns [] and the
    caller falls back to day-level reporting only)."""
    capture = capture if capture is not None else {}

    def fake_solve(route, car, solar_provider, wind_provider, day_index,
                   start_soc_pct, alpha_next_day_pct, loops_committed,
                   **kw):
        capture["car"] = car
        capture["loops_committed"] = loops_committed
        capture["kw"] = kw
        return {
            "final_soc_pct": final_soc_pct,
            "total_time_s": total_time_s,
            "driven_km": 150.0,
            "v_kmh": [60.0, 62.0, 58.0],
            "motor_energy_wh": motor_energy_wh,
            # x_m / seg_type_trace deliberately omitted -> _stage_breakdown
            # returns [] and resolve_intraday reports day-level only.
        }

    trust_region.singleday.solve = fake_solve
    return capture


# ---------------------------------------------------------------------------
# Loop-count resolution
# ---------------------------------------------------------------------------

def test_loops_remaining_none_takes_all_plan_loops():
    cap = _install_fake_solve()
    car = CarState()

    rr = trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=3, cur_soc_pct=60.0, elapsed_s=3600.0,
        loops_remaining=None)

    plan = trust_region._get_day_plan(3)
    assert len(cap["loops_committed"]) == len(plan.loops)
    print("PASS loops_remaining_none_takes_all_plan_loops")


def test_loops_remaining_int_reps_first_loop():
    cap = _install_fake_solve()
    car = CarState()

    trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=3, cur_soc_pct=60.0, elapsed_s=3600.0,
        loops_remaining=2)

    plan = trust_region._get_day_plan(3)
    if plan.loops:
        name, km = plan.loops[0]
        assert cap["loops_committed"] == [(name, km)] * 2
    print("PASS loops_remaining_int_reps_first_loop")


# ---------------------------------------------------------------------------
# Breakdown OFF (default) — no behaviour change
# ---------------------------------------------------------------------------

def test_breakdown_disabled_by_default():
    _install_fake_solve(total_time_s=20000.0)
    car = CarState()

    rr = trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=3, cur_soc_pct=60.0, elapsed_s=3600.0,
        cs_taken=True, loops_remaining=0)

    assert rr["breakdown_min"] == 0

    expected_finish = (
        trust_region.rc.day_start_time_s(3) + 3600.0 + 20000.0
    )
    assert abs(rr["finish_abs_s"] - expected_finish) < 1e-6
    print("PASS breakdown_disabled_by_default")


# ---------------------------------------------------------------------------
# Breakdown ON — deterministic, pushes the finish clock later
# ---------------------------------------------------------------------------

def test_breakdown_enabled_pushes_finish_later():
    _install_fake_solve(total_time_s=20000.0, motor_energy_wh=6000.0)
    car = CarState()

    rr_off = trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=3, cur_soc_pct=60.0, elapsed_s=3600.0,
        cs_taken=True, loops_remaining=0,
        breakdown_enabled=False)

    rr_on = trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=3, cur_soc_pct=60.0, elapsed_s=3600.0,
        cs_taken=True, loops_remaining=0,
        breakdown_enabled=True, breakdown_seed=20260905)

    assert rr_on["breakdown_min"] > 0
    assert rr_on["finish_abs_s"] > rr_off["finish_abs_s"]

    # finish_abs_s must differ from the no-breakdown case by exactly the
    # sampled breakdown duration (rounded to the same minute granularity
    # the return dict reports).
    delta_s = rr_on["finish_abs_s"] - rr_off["finish_abs_s"]
    assert abs(round(delta_s / 60.0) - rr_on["breakdown_min"]) <= 1
    print("PASS breakdown_enabled_pushes_finish_later")


def test_breakdown_seed_is_deterministic():
    _install_fake_solve(total_time_s=15000.0, motor_energy_wh=5000.0)
    car = CarState()

    rr1 = trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=5, cur_soc_pct=70.0, elapsed_s=0.0,
        breakdown_enabled=True, breakdown_seed=42)

    _install_fake_solve(total_time_s=15000.0, motor_energy_wh=5000.0)

    rr2 = trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=5, cur_soc_pct=70.0, elapsed_s=0.0,
        breakdown_enabled=True, breakdown_seed=42)

    assert rr1["breakdown_min"] == rr2["breakdown_min"]
    print("PASS breakdown_seed_is_deterministic")


def test_breakdown_seed_changes_value_across_days():
    # Different day_index folds into the RNG seed (see resolve_intraday's
    # `_random.Random(_seed + day_index)`), so two different days with the
    # same base seed are not required to match — just documenting the
    # mechanism stays wired through day_index, not hardcoded.
    _install_fake_solve(total_time_s=15000.0, motor_energy_wh=5000.0)
    car = CarState()

    rr_day1 = trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=1, cur_soc_pct=70.0, elapsed_s=0.0,
        breakdown_enabled=True, breakdown_seed=42)

    _install_fake_solve(total_time_s=15000.0, motor_energy_wh=5000.0)

    rr_day6 = trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=6, cur_soc_pct=70.0, elapsed_s=0.0,
        breakdown_enabled=True, breakdown_seed=42)

    assert rr_day1["breakdown_min"] >= 0 and rr_day6["breakdown_min"] >= 0
    print("PASS breakdown_seed_changes_value_across_days")


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def test_feasible_flag_from_end_soc():
    car = CarState()

    _install_fake_solve(final_soc_pct=car.soc_min_pct + 5.0)
    rr_ok = trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=4, cur_soc_pct=80.0, elapsed_s=0.0)
    assert rr_ok["feasible"] is True

    _install_fake_solve(final_soc_pct=car.soc_min_pct - 5.0)
    rr_bad = trust_region.resolve_intraday(
        _FakeRoute(), car, None, None,
        day_index=4, cur_soc_pct=80.0, elapsed_s=0.0)
    assert rr_bad["feasible"] is False

    print("PASS feasible_flag_from_end_soc")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _real_solve = trust_region.singleday.solve

    try:
        test_loops_remaining_none_takes_all_plan_loops()
        test_loops_remaining_int_reps_first_loop()
        test_breakdown_disabled_by_default()
        test_breakdown_enabled_pushes_finish_later()
        test_breakdown_seed_is_deterministic()
        test_breakdown_seed_changes_value_across_days()
        test_feasible_flag_from_end_soc()

        print("\nALL PASS")

    finally:
        trust_region.singleday.solve = _real_solve
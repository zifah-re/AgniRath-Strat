"""
Fast, dependency-light unit tests for trust_region.resolve_from_actuals
(Feature C — the "re-solve from realized actuals" method).

These do NOT run the real 8-day optimizer — that is covered by the full
`python -m optimizers.trust_region` run. Here we monkeypatch optimize() and
extract_final_profiles() so the tests are deterministic and run in <1s, and
we assert the ONLY new logic the wrapper adds:

  * resume_day / start_soc / efficiency are threaded into optimize() correctly
  * solar_efficiency overrides the forward car's array_efficiency
  * extra car_overrides are merged (and array_efficiency from solar_efficiency
    wins over any array_efficiency in car_overrides)
  * feasible vs infeasible is branched correctly and profiles only extracted
    when feasible
  * input validation (bad resume_day, bad efficiency) raises
  * the returned audit block echoes the realized inputs

Run:  python -m pytest tests/test_resolve_from_actuals.py -q
  or:  python tests/test_resolve_from_actuals.py     (no pytest needed)
"""
import os
import sys
import dataclasses

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.car_config import CarState
from optimizers import trust_region


# --- a tiny stand-in so we don't need real Route objects ---------------------
class _FakeRoute:
    def __init__(self, total_m=200_000.0):
        self.total_m = total_m


def _make_routes(n=8):
    return {d: _FakeRoute() for d in range(n)}


def _install_spies(monkeypatch_feasible=True, capture=None):
    """Replace optimize() and extract_final_profiles() with spies that record
    the kwargs they were called with, so the wrapper's plumbing can be asserted
    without running the optimizer."""
    capture = capture if capture is not None else {}

    def fake_optimize(routes, car, solar_providers, wind_providers, **kw):
        capture["optimize_car"] = car
        capture["optimize_kw"] = kw
        return {"feasible": monkeypatch_feasible, "s_start_pct": [50.0] * 8,
                "loop_plan": {}, "alpha_day_pct": {}, "start_day_index": kw.get("start_day", 0)}

    def fake_extract(routes, base_car, solar_providers, wind_providers, result, **kw):
        capture["extract_car"] = base_car
        capture["extract_kw"] = kw
        return {7: {"start_soc_pct": 70.0, "end_soc_pct": 40.0, "driven_km": 300.0,
                    "total_time_s": 30000.0, "loops_committed": [],
                    "late_penalty_min": 0, "inherited_penalty_min": 0}}

    trust_region.optimize = fake_optimize
    trust_region.extract_final_profiles = fake_extract
    return capture


# -----------------------------------------------------------------------------
def test_solar_efficiency_overrides_array_efficiency():
    cap = _install_spies(monkeypatch_feasible=True)
    car = CarState()
    assert car.array_efficiency != 0.20  # nominal is 0.22
    out = trust_region.resolve_from_actuals(
        _make_routes(), car, {}, {},
        resume_day=6, start_soc_pct=70.0, solar_efficiency=0.20)

    assert out["feasible"] is True
    # forward car used for BOTH optimize and extract must carry the new eff
    assert abs(cap["optimize_car"].array_efficiency - 0.20) < 1e-9
    assert abs(cap["extract_car"].array_efficiency - 0.20) < 1e-9
    assert abs(out["forward_car"].array_efficiency - 0.20) < 1e-9
    # start day / start soc threaded through
    assert cap["optimize_kw"]["start_day"] == 6
    assert abs(cap["optimize_kw"]["start_soc_pct"] - 70.0) < 1e-9
    # audit block echoes the reality
    assert out["actuals"]["resume_day_index"] == 6
    assert abs(out["actuals"]["solar_efficiency_used"] - 0.20) < 1e-9
    print("PASS solar_efficiency_overrides_array_efficiency")


def test_default_efficiency_is_car_nominal():
    cap = _install_spies(monkeypatch_feasible=True)
    car = CarState()
    out = trust_region.resolve_from_actuals(
        _make_routes(), car, {}, {}, resume_day=2, start_soc_pct=58.0)
    # no solar_efficiency given -> forward car keeps nominal efficiency
    assert abs(out["forward_car"].array_efficiency - car.array_efficiency) < 1e-9
    print("PASS default_efficiency_is_car_nominal")


def test_car_overrides_merge_and_efficiency_wins():
    cap = _install_spies(monkeypatch_feasible=True)
    car = CarState()
    out = trust_region.resolve_from_actuals(
        _make_routes(), car, {}, {}, resume_day=3, start_soc_pct=60.0,
        solar_efficiency=0.19,
        car_overrides={"mass_kg": 999.0, "array_efficiency": 0.11})
    fc = out["forward_car"]
    assert abs(fc.mass_kg - 999.0) < 1e-9              # extra override applied
    assert abs(fc.array_efficiency - 0.19) < 1e-9      # solar_efficiency wins
    print("PASS car_overrides_merge_and_efficiency_wins")


def test_infeasible_returns_no_profiles():
    cap = _install_spies(monkeypatch_feasible=False)
    car = CarState()
    out = trust_region.resolve_from_actuals(
        _make_routes(), car, {}, {}, resume_day=5, start_soc_pct=25.0)
    assert out["feasible"] is False
    assert out["result"] is None
    assert out["profiles"] == {}
    assert "extract_car" not in cap  # extract never called when infeasible
    print("PASS infeasible_returns_no_profiles")


def test_validation_errors():
    _install_spies(monkeypatch_feasible=True)
    car = CarState()
    for bad_day in (-1, 99):
        try:
            trust_region.resolve_from_actuals(
                _make_routes(), car, {}, {}, resume_day=bad_day, start_soc_pct=60.0)
            raise AssertionError(f"resume_day={bad_day} should have raised")
        except ValueError:
            pass
    for bad_eff in (0.0, 1.5, -0.2):
        try:
            trust_region.resolve_from_actuals(
                _make_routes(), car, {}, {}, resume_day=2, start_soc_pct=60.0,
                solar_efficiency=bad_eff)
            raise AssertionError(f"solar_efficiency={bad_eff} should have raised")
        except ValueError:
            pass
    print("PASS validation_errors")


def test_midday_flag_detected():
    _install_spies(monkeypatch_feasible=True)
    car = CarState()
    out = trust_region.resolve_from_actuals(
        _make_routes(), car, {}, {}, resume_day=4, start_soc_pct=60.0,
        dist_done_km=42.0, elapsed_s=3600.0)
    assert out["actuals"]["mid_day"] is True
    out2 = trust_region.resolve_from_actuals(
        _make_routes(), car, {}, {}, resume_day=4, start_soc_pct=60.0)
    assert out2["actuals"]["mid_day"] is False
    print("PASS midday_flag_detected")


if __name__ == "__main__":
    # Snapshot the real callables so tests can freely monkeypatch module globals.
    _real_optimize = trust_region.optimize
    _real_extract = trust_region.extract_final_profiles
    try:
        test_solar_efficiency_overrides_array_efficiency()
        test_default_efficiency_is_car_nominal()
        test_car_overrides_merge_and_efficiency_wins()
        test_infeasible_returns_no_profiles()
        test_validation_errors()
        test_midday_flag_detected()
        print("\nALL PASS")
    finally:
        trust_region.optimize = _real_optimize
        trust_region.extract_final_profiles = _real_extract

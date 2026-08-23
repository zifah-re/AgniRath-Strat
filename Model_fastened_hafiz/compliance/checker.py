"""
compliance/checker.py — STUB with FROZEN API (block 3.2, owner: Junior C).

Plan v3 §4.3: the plan linter. Every velocity profile (L2, L3, human
override) passes through check() before a human sees it. Nothing ships
unchecked.

FROZEN API (README §Interfaces):
    check(plan, route, car, day_index) -> CheckResult
      plan: dict with keys
        v_ms          np.ndarray  target speed per control segment
        seg_start_m   np.ndarray  segment start distances
        stops         list[dict(kind, t_start_s, duration_s, x_m)]
                      kind in {control_stop, loop_stop, swap, charge, unplanned}
      CheckResult.ok: bool; CheckResult.violations: list[str] (reg-cited).

Checks (all constants from configs.race_config — cite clauses in messages):
  window timing incl. day-specific start/finish + 17:30 cutoff,
  v <= route.v_max_ms_at(x), SOC bounds vs car, swap cadence <= 2 h,
  30-min CS present + notify-able, 5-min loop stop before every loop entry,
  no driving outside competition hours, unplanned-stop budget present.
"""
import dataclasses

@dataclasses.dataclass
class CheckResult:
    ok: bool
    violations: list

def check(plan, route, car, day_index: int) -> CheckResult:
    raise NotImplementedError("block 3.2 — Junior C")

# Agnirath Strategy Model — Sasol Solar Challenge 2026

Race strategy system for the 2026 Sasol Solar Challenge (9–17 Sep).
Three layers on one shared physics core (Plan v3 §8):

| Layer | Module | Job |
|---|---|---|
| L1 | `optimizers/multiday_dp.py` | integer loops/day + start-of-day SOC floors |
| L2 | `optimizers/singleday.py` | velocity per 5 km segment (GA→SLSQP→integer) |
| L3 | `optimizers/mpc/` | live receding-horizon tracking + race logic |

Cross-cutting: `compliance/` (reg linter), `analysis/trailering.py`
(red-flag map), reliability/degraded-mode (`analysis/degraded_playbook.py`,
car-health gate in `optimizers/mpc/base.py`).

## Status (workplan blocks)

- **Block 0–1: DONE** — skeleton, frozen contracts (below), `race_config`
  (reg-cited), `car_config` (mutable state; TODO-VERIFY items flagged),
  weather collection cron, consolidated `core/physics.py` with both legacy
  models reproduced under test (57 tests green).
- Blocks 2–9: stubs in place with owners + frozen APIs in their docstrings.

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests/            # must be green before any merge
export SOLCAST_API_KEY=...         # then:
python -m pipeline.collect_weather test-keys
python -m pipeline.collect_weather pull --points data/raw/points.json
```

Run everything from the `Model/` directory (imports are rooted here).

## FROZEN INTERFACE CONTRACTS (workplan 0.1 — change only via senior-approved PR)

### 1. Route file — `data/processed/route_day{d}.parquet`
One row per ~10 m grid point. Columns (see `core/route.py:REQUIRED_COLUMNS`):
`distance_m, lat, lon, elevation_m, slope_pct, bearing_deg, curvature_1pm,
v_max_ms, circle_id, seg_type, red_flag_trailer, control_stop, day`.
Consumers use `core.route.Route`, never raw parquet parsing.

### 2. Solar — `provider.ghi_wm2(t_s, x_m) -> float` (W/m²)
`t_s` seconds since local midnight of the race day; `x_m` metres along the
day's route. Implementations: `GaussianProvider` (seniors' fallback,
verbatim), `PVLibClearSkyProvider`, `SolcastCurveFitProvider` (per-circle
fits, zero API calls in solves). PMF correction is a multiplier from
`core.solar.pmf_correction_factor(tcc, t_s)` (Paper 1 Table 5 seed values;
`pipeline/update_pmf.py` replaces with our own).
Slope-tilt geometry: `core.solar.slope_geometry_factor(...)` returns the
factor passed to `physics.net_power(solar_geom_factor=...)`.

### 3. Wind — `provider.wind(t_s, x_m) -> (speed_ms, dir_deg_from)`
Meteorological FROM convention. Decomposition lives in `core.wind`:
`along_track_ms(...)` (positive = tailwind) and `relative_wind(...) ->
(magnitude, yaw_psi_deg)` for CdA(ψ).

### 4. Physics — `core.physics`
`net_power(car, v, v_next, slope_pct, ghi, seg_len_km, ...) -> (P_w, dt_s)`
— positive charges the battery. `power_required_at_speed(...)` for the
trailering red flag. Legacy ports `dashboard_power` / `motor_power_kr` are
test/reference only.

### 5. Compliance — `compliance.checker.check(plan, route, car, day_index)`
Plan dict: `v_ms`, `seg_start_m`, `stops[{kind,t_start_s,duration_s,x_m}]`.
Returns `CheckResult(ok, violations)` with reg-cited messages. Nothing is
shown to a human unchecked.

### 6. Car state — `configs.car_config.CarState`
Mutable capability state. Degraded runs use
`dataclasses.replace(car, array_area_m2=4.1, ...)` — never edit module
constants at runtime. Every value in the file carries source + date;
`TODO-VERIFY` marks values awaiting the car team.

### 7. Config discipline
Regulation constants ONLY in `configs/race_config.py` (clause-cited).
Solver knobs ONLY in `configs/solver_config.py`. No magic numbers anywhere
else; units suffixed (`_ms`, `_wh`, `_pct`, `_s`, `_kmh`).

## Engineering standards (Plan v3 §9)
Protected `main`; PR + one reviewer minimum (senior for `core/`,
`optimizers/`, `compliance/`); regression suite green = merge gate; member
folders are sandboxes — shipping means merged here; docstrings state
purpose/inputs/units; comments explain *why*.

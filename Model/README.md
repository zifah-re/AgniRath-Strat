# Agnirath Strategy Model — Sasol Solar Challenge 2026

Offline race-strategy optimiser for the solar car. Given the route, weather and
car parameters it produces, for each of the 8 race days, the number of loops, a
speed profile and the start/end SOC, and writes them to a strategy JSON that the
offline dashboard renders.

## How it works

| Stage | File | Job |
|---|---|---|
| Tier 1 | `optimizers/tier1.py` | coarse 500 m DP over all 8 days (baseline SOC trajectory) |
| Tier 2 | `optimizers/tier2.py` | per-day, per-loop-count single-day solves at 3 start SOCs, giving surrogates (end SOC, finish time, wasted solar) |
| Tier 3 | `optimizers/tier3.py` | backward multi-day DP over the surrogates (overnight charge, late-finish and high-SOC penalties) |
| Outer loop | `optimizers/trust_region.py` | repeats Tier 2/3 inside a trust region, then extracts the final profiles |
| Single-day solver | `optimizers/singleday.py` | speed per 10 km segment: GA, then SLSQP, then integer km/h |
| Physics | `core/physics.py`, `core/battery.py`, `simulator/forward_sim.py` | force balance, motor/regen, SOC–voltage battery, forward simulation |
| Route / solar / wind | `core/route.py`, `core/solar.py`, `core/wind.py` | route grid, POA irradiance (Solcast), relative wind |
| Rules | `configs/race_config.py`, `compliance/` | regulation constants (clause-cited) and checker |

`optimizers/mpc/` holds the live receding-horizon code; it is not part of the
offline pipeline.

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q          # 87 pass, 1 known failure (see below)
```

Run everything from `Model/` (imports are rooted here).

## Running

```bash
# Full 8-day cold start (Day 3 route variant: aryaman | prahlad | both)
python -m optimizers.trust_region --variant aryaman

# Re-solve Day N onward from a given start SOC (between days)
python -m optimizers.trust_region resolve --resume-day 4 --start-soc 62 --variant aryaman

# Re-plan the rest of one day (intra-day)
python -m optimizers.trust_region resolve-intraday --day 4 --soc 62 --time 11:30 --variant aryaman

# Live demo: print the saved plan, then re-solve one day (~10-20 s, single process)
python run_demo.py --day 4 --soc 60 --time 09:30
```

Flags: `--breakdown` (one random breakdown per day), `--solar-eff 0.2x`
(measured array efficiency for a re-solve), `--dist-done`, `--loops-left`, `--cs-taken`
(intra-day state).

Output goes to `output/` as `strategy_<variant>*.json`. View it with
`python -m analysis.offline_dashboard --all output/` (offline Plotly HTML).

## Parallelism

Tier 2 samples the days in parallel. Set `TIER2_USE_PROCESS_POOL` in
`configs/solver_config.py` for the default, or override per run without editing
anything:

```bash
AGNIRATH_TIER2=serial     python -m optimizers.trust_region ...   # one process, safest
AGNIRATH_TIER2=threads    python -m optimizers.trust_region ...   # no child processes
AGNIRATH_TIER2=processes  python -m optimizers.trust_region ...   # fastest, many cores
```

(Windows PowerShell: `$env:AGNIRATH_TIER2="serial"`; cmd: `set AGNIRATH_TIER2=serial`.)
If the process pool fails for any reason Tier 2 logs a warning and falls back to
threads. `AGNIRATH_FORCE_SPAWN=1` forces the Windows/macOS 'spawn' start method on
Linux for testing. Other speed knobs (`SKIP_BREAKDOWN_WHEN_UNUSED`,
`ENERGY_GRID_M`, `GA_GENERATIONS`) are described in `docs/README_PERF_old.md`.

## Car parameters (`configs/car_config.py`)

Mass 336.5 kg, array area 5.78 m², array efficiency 24.5 % (measured), system
derate 1.0, CdA 0.16 m², Crr 0.007, motor efficiency 0.95, regen 0.70, 6 packs
x 588 Wh, v_max 85 km/h. Update values here only.

## Deployed plan

`output/strategy_aryaman_idk.json` is the plan that was actually run at Sasol
(3201.9 km planned over 8 days, Day 3 = aryaman variant). Older plans are in
`output/` and `output/archive_aug_plans/`.

## Known issues

- `tests/test_physics.py::test_new_model_matches_dashboard_when_wind_zero`
  compares against the stale legacy Dashboard defaults and fails; it predates
  the current car parameters.
- `core/physics.py` keeps a fallback `SYSTEM_DERATE = 0.90`; the model uses
  `CarState.system_derate` (1.0).

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

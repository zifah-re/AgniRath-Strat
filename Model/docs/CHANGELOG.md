# Changelog

## [0.1.0] — 2026-07-24 — Blocks 0–1 (foundations + core consolidation)
- Repo skeleton per Plan v3 §9; frozen interface contracts in README.
- configs/race_config.py: 2026 regs extracted, every constant clause-cited;
  RACE_MODE switch; late-finish penalty function matches the regs' worked
  examples; released route notes + blind-loop placeholder.
- configs/car_config.py: mutable CarState; defaults from WSC'25 Dashboard
  constants with TODO-VERIFY flags; both legacy parameter sets recorded;
  CdA(psi) placeholder table (disabled until Bilal & Varun's data lands).
- core/physics.py: consolidated model — Dashboard force balance upgraded to
  relative-airspeed drag + CdA(psi) + solar geometry hook; exact ports of
  Dashboard calculate_net_power and Kevin/Ramana calculate_power (goldens
  captured from executing the original code); humid-air density (flagged
  substitute for Paper 1 ref [15] polynomial).
- core/battery.py: measured SOC<->V curve (101 pts, WSC'25); usable-energy
  window model; charge/discharge efficiency ledger (Paper 4 convention).
- core/solar.py: frozen GHI(t,x) interface; Gaussian (verbatim seniors'),
  pvlib clear-sky, Solcast per-circle curve-fit providers; Paper 1 Table 5
  PMF seed; Paper 4 incidence geometry + slope-tilt factor.
- core/wind.py: frozen wind(t,x) interface; along-track (+=tailwind) and
  relative-wind/yaw decomposition; constant + table providers.
- core/route.py: route parquet schema contract + typed loader.
- pipeline/collect_weather.py: Solcast key test, forecast+actuals+wind
  collection cron (workplan 0.4).
- analysis/trailering.py: working red-flag map (SR 2.32 60 km/h criterion).
- simulator/forward_sim.py: minimal consolidated integrator (block-3 seed).
- Stubs with frozen APIs + owners: build_route, blind_stage, update_pmf,
  checker, turn_audit, multiday_dp, singleday, mpc/base, scenarios,
  degraded_playbook, wind_sensitivity, debrief.
- tests/: 57 tests green — legacy reproduction goldens, config/regs
  sanity (incl. regs' own penalty examples), battery, solar, wind,
  simulator hand-check.

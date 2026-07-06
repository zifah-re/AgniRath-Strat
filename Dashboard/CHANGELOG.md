# Changelog

All notable changes to the Agnirath Strategy Dashboard will be documented in this file.

---

## [3.2.3] - 2026-07-05
### Added
- Integrated solar data support directly into Model Predictive Control (`mpc.py`) calculations.
- Added a dynamic scrolling interface for enhanced readability of MPC prediction vectors.

### Fixed
- Resolved dependency version mismatches (specifically addressing `pandas` environment issues).
- Fixed log parsing bugs related to `Bala25_T5.jsonl` data streams.

### Removed
- Removed `numba` from `requirements.txt` to streamline environment setup.

---

## [3.2.2] - 2026-07-04
### Added
- Created an offline simulation model inside a new `complete_sim.py` framework to include solar and target velocity profiles.
- Integrated file saving and loading functionality natively across simulation runs.

### Changed
- Completed functional MPC model integration into the strategy pipeline.

### Fixed
- Fixed critical bugs regarding the file save/load feature and handled edge cases where files failed to write.
- Removed verbose, performance-heavy logging routines from `mpc.py`.

---

## [3.2.1] - 2026-07-02
### Added
- Implemented core Model Predictive Control optimization engine (`mpc.py`).
- Added a new API endpoint/call to serve vehicle profile data alongside helper scripts.
- Introduced explicit support for `TargetProfile` constraints.

### Changed
- Reworked track alignment math: Renamed `bearing` to `heading` and switched to a precalculated heading matrix to optimize runtime performance.
- Modified `traffic.py` and `google_earth.py` return structures to pass dictionaries instead of tuples, preventing breaking changes when new fields are appended.

### Fixed
- Resolved track data bugs causing duplicate latitude/longitude coordinate pairs.
- Fixed complex coordinate geometry bugs relating to concurrent latitude, longitude, and heading calculations.

---

## [3.2.0] - 2026-06-30
### Added
- Introduced initial MPC task scaffolding and baseline team documentation.

### Changed
- Optimized the frontend user experience by removing the redundant "Inspect" button requirement for dashboard updates.
- Centralized configuration by moving environment constants from `main.py` into a unified `constants.py` file.

### Fixed
- Resolved API authentication errors by restructuring token handling.
- Cleaned up node dependencies by removing unused `br` and `zstd` compressors from frontend packages.

---

## [3.1.0] - 2026-06-28
### Added
- Calculated Estimated Time of Arrival (ETA) metrics directly inside the strategy modules.
- Integrated TomTom road snapping and speed limit fetching into `traffic.py`.
- Added toggles to overlay live traffic layers and custom map layers onto `map.html`.
- Implemented map styling elements (color states, icons, and info markers) matching Google Earth schemas.

### Changed
- Improved the gradient profile algorithm by factoring track altitude directly into the rolling distance calculations.
- Elevated map tracking UX with smooth car icon animations and real-time bearing updates.

### Fixed
- Enhanced the reliability of the road-snapping algorithm by introducing route exclusion logic.
- Patched motor power equations to properly account for road slope variations.
- Fixed directory reference bugs in `real_sim.py` allowing it to execute flawlessly from any working directory.
- Scrubbed hardcoded server and client access tokens from public git history.

---

## [3.0.0] - 2026-06-23
### Added
- Initial deployment of the Agnirath Strategy Dashboard web asset suite.
- Added live telemetry telemetry tracking maps capturing Latitude, Longitude, Altitude, and Gradient.

### Fixed
- Fixed telemetry plots, including an Altitude vs. Time graph in the Strategy tab.

---

## [2.1.0] - 2026-06-23
### Added
- Added base codebase structure including `main.py`, `downlink.py`, `simulator.py`, requirements configurations, and documentation.

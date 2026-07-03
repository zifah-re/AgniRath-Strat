# MPC Simulation 

**Route:** Sasolburg → Zeerust | Base route without control stop loops

---

## Folder Structure

```
Diyaansh/
└── mpc_simulation_task/
    ├── physics.py
    ├── simulation_engine.py
    ├── mpc_controller.py
    ├── final_race_telemetry.csv
    ├── v_optimal_base_ipopt.npy
    ├── soc_hist_base_ipopt.npy
    ├── t_hist_base_ipopt.npy
    ├── race_summary_ipopt.npy
    └── results/
        ├── mpc_log_ideal.csv
        ├── mpc_log_realistic.csv
        ├── mpc_log_headwind.csv
        ├── mpc_log_cloudy.csv
        ├── mpc_log_worst.csv
        ├── mpc_simulation_results_ideal.png
        ├── mpc_simulation_results_realistic.png
        ├── mpc_simulation_results_headwind.png
        ├── mpc_simulation_results_cloudy.png
        └── mpc_simulation_results_worst.png
```

---

## How to Run

```bash
cd Diyaansh/mpc_simulation_task

python mpc_controller.py --scenario realistic --seed 42
```

Available scenarios: `ideal`, `realistic`, `headwind`, `cloudy`, `worst`

Optional flags: `--seed <int>`, `--lag <float>` (motor lag time constant in seconds), `--quiet` (suppress per-tick output)

The script loads the telemetry CSV and the three `.npy` offline plan files, runs the full MPC loop, saves the log to `mpc_log.csv`, and saves the plot to `mpc_simulation_results.png`.

**Dependencies:** `numpy`, `pandas`, `casadi`, `scipy`, `matplotlib`

---

## File Descriptions

### `physics.py`

Defines all vehicle constants and shared physics functions used across the project.

Vehicle parameters: 330 kg total mass, Cd = 0.0845, frontal area 1.155 m², rolling resistance coefficient 0.0048, air density 1.05 kg/m³. Motor efficiency 90%, regen braking efficiency 70%, 20 W auxiliary load. Solar panel: 6 m², 20% efficiency, 3 kWh battery pack. Minimum SOC floor: 20%.

Contains `calculate_irradiance(t)` which returns solar irradiance in W/m² using a Gaussian model centred at solar noon (t = 14400 s from 08:00), peaking at 1073 W/m². Contains `calculate_power(v, t, slope)` which computes net electrical power draw from the battery at a given speed, time, and terrain slope. Contains `calculate_soc_update(SOC, P, dt)` which steps the battery state forward by one time interval.

---

### `final_race_telemetry.csv`

GPS-derived route data for the Sasolburg–Zeerust base route. One row per waypoint at 50 m spacing. Columns include `distance_m` and `gradient_deg`. This file is the terrain input to both the offline optimiser and the real-time MPC.

---

### `v_optimal_base_ipopt.npy` · `soc_hist_base_ipopt.npy` · `t_hist_base_ipopt.npy`

Output arrays from the offline IPOPT optimiser (run separately, not included here). Each array has one entry per 50 m waypoint of the base route.

- `v_optimal_base_ipopt.npy` — optimal speed in m/s at every waypoint
- `soc_hist_base_ipopt.npy` — corresponding battery SOC (fraction) at every waypoint
- `t_hist_base_ipopt.npy` — elapsed time in seconds at every waypoint

The MPC controller uses these three arrays as its reference plan. At each tick it looks up the waypoint closest to the car's current position and computes how far the actual state has deviated from what was planned.

---

### `race_summary_ipopt.npy`

A pickled dictionary saved by the offline optimiser. Contains scalar summary values: predicted arrival time at Zeerust, SOC on arrival, SOC after the 30-minute control stop, optimal loop count, loop target speed, and total race distance. Used for pre-race briefing, not loaded by the MPC controller at runtime.

---

### `simulation_engine.py`

Physics-accurate simulator that stands in for real car hardware during development. The MPC controller calls `read_sensors()` and `step()` on it exactly as it would call real telemetry and motor interfaces.

Internally advances at 1-second sub-steps. At each sub-step it applies a first-order actuation lag (default 3 s time constant) between the MPC speed target and actual wheel speed, evolves wind via a correlated AR(1) process, perturbs rolling resistance with Gaussian noise, and applies a slow sinusoidal cloud-cover envelope on top of the baseline irradiance. The MPC only ever sees the noisy sensor output, not the true hidden state.

Five disturbance scenarios are available:

| Scenario | Mean wind | Cloud amplitude | Description |
|---|---|---|---|
| `ideal` | 0 m/s | 0% | No disturbances — baseline validation |
| `realistic` | 0 m/s | 8% | Light wind and partial cloud |
| `headwind` | 4 m/s | 6% | Sustained headwind throughout |
| `cloudy` | 1 m/s | 30% | Significant cloud cover |
| `worst` | 5 m/s | 40% | Heaviest wind and cloud combined |

---

### `mpc_controller.py`

The real-time MPC loop. Runs against the simulator and produces logs and plots.

**Solver setup.** The NLP is built once at startup using CasADi symbolic variables and compiled into an IPOPT solver object (linear solver: MUMPS). At each tick only the parameter vector is updated — slopes, current time, current SOC, reference SOC trajectory, last speed, measured wind — so the symbolic graph is never rebuilt. Horizon: 150 nodes at 200 m spacing (every 4th telemetry waypoint), giving a 30 km lookahead. Decision variables per node: speed, SOC, normalised time (t / 9 h), and a slack variable for the soft speed floor.

**Cost function weights:**

| Term | Weight | Purpose |
|---|---|---|
| SOC tracking vs offline plan | 5.0 | Follow the energy budget |
| Speed tracking vs offline plan | 10.0 | Stay close to planned velocity |
| Acceleration² | 5.0 | Penalise rapid speed changes |
| Speed continuity across ticks | 50.0 | No abrupt jump between MPC calls |
| Slack variable (soft speed floor) | 250.0 | Enforce minimum speed softly |
| Terminal SOC error | 10.0 | Correct end-of-horizon SOC |
| SOC warning zone (< 25%) | 100.0 | Ramp up caution near floor |
| SOC critical (< 20%) | 100 000.0 | Hard barrier at floor |

Speed bounds: 60–120 km/h (soft lower bound via slack variable). Acceleration bound: ±2 m/s². SOC hard lower bound: 21% at every node. If the solver returns a non-converged status, the controller holds the last accepted speed target and logs a warning to `diagnostics.log`.

**Tick loop:**
1. Read noisy sensor snapshot from simulator
2. Look up current position in offline plan, compute speed/SOC/time deviations
3. Call IPOPT with updated parameters and warm-start from offline speed profile
4. Send first element of optimal speed sequence to simulator as target
5. Log everything, advance simulator by 60 seconds, repeat

---

### `mpc_log.csv`

One row per 60-second MPC tick. Columns: elapsed time (s and hrs), distance (km), actual speed (km/h), MPC target speed (km/h), actual SOC (%), planned speed and SOC from the offline reference, speed and SOC tracking errors, solar irradiance (W/m²), wind speed (m/s), solver time (ms), solver success flag.

---

### `mpc_simulation_results.png`

Seven-panel plot saved at the end of each run:

1. Speed — offline plan (dashed), actual simulator speed, MPC target
2. Battery SOC — offline plan vs actual, with 20% floor line
3. Tracking error — speed error and SOC error vs time
4. Solar irradiance — planned Gaussian vs actual (with cloud disturbance), deficit shaded in red
5. Wind speed — headwind/tailwind shaded separately
6. Longitudinal acceleration — computed from 1-second sub-step trace, with ±2 m/s² limit lines
7. Route gradient — uphill/downhill shaded, mapped from distance to time axis

## Ideas/Concepts 
 
**Receding Horizon Control.** At each tick the controller solves an optimisation problem over a fixed future window, applies only the first action, then re-solves at the next tick with updated measurements. The plan never goes stale for more than one interval, and real-world deviations are absorbed automatically through feedback.
 
**Nonlinear programming with IPOPT.** The energy dynamics are nonlinear (drag scales with v³, solar input is time-varying), so the optimisation problem cannot be reduced to a linear or quadratic programme. IPOPT uses a primal-dual interior-point method to find a local optimum, with CasADi providing exact first and second derivatives via automatic differentiation.
 
**Multiple shooting discretisation.** Speed, SOC, and time are independent decision variables at every node. Physics defect constraints enforce that adjacent nodes are consistent with the equations of motion. This avoids error accumulation across the horizon that would occur with forward simulation.
 
**Soft constraints for guaranteed feasibility.** The minimum speed bound is enforced via a slack variable rather than a hard inequality. This ensures IPOPT always finds a feasible solution even when simultaneous headwind and solar deficit make the hard-constrained problem infeasible.
 
**Static reference tracking.** The MPC tracks the offline SOC plan rather than a dynamically adjusted achievable projection. This prevents the controller from perceiving zero error while the battery drains — any weather-induced deficit shows up immediately as a tracking error, causing the controller to reduce speed proactively.
 
**Offline / online separation.** The CasADi computation graph is compiled once at startup. At each tick only the parameter vector (current state, terrain, reference, wind) is updated. This avoids the cost of re-parsing and re-compiling the symbolic problem at runtime, keeping solve times within a 60-second range.
 
**Physics-based simulation with disturbance modelling.** The simulator advances at 1-second sub-steps and models actuation lag (first-order filter on speed), correlated wind via an AR(1) process, cloud cover as a slow sinusoidal envelope, and road surface variation as Gaussian noise on the rolling resistance coefficient. The MPC only sees noisy sensor output, not the true hidden state, matching real conditions.
 

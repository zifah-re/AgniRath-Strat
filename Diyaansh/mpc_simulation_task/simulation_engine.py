"""
simulation_engine.py
====================
Realistic race simulation engine that acts as the "hardware" telemetry source
for the MPC loop. Replaces read_sensors() with physics-accurate state propagation.

The car follows whatever speed target the MPC gives it, subject to:
  - Actuation lag   : driver/motor can't change speed instantly
  - Road disturbance: wind gusts, rolling resistance variation
  - Solar noise     : cloud cover causing irradiance fluctuations
  - Sensor noise    : GPS, BMS, pyranometer measurement error

Usage
-----
    from simulation_engine import RaceSimulator
    sim = RaceSimulator(scenario='realistic')   # or 'cloudy', 'headwind', 'worst'
    state = sim.read_sensors()                # current telemetry dict
    sim.step(target_speed_kmh=85.0)          # advance 1 MPC timestep (60 s)

The simulator advances in small 1-second sub-steps internally for accuracy,
then surfaces the state at each 60-second MPC tick.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from physics import (
    MASS, CRR, CD, A_FRONTAL, RHO,
    ETA_MOTOR, ETA_REGEN, P_AUX,
    PANEL_AREA, PANEL_EFF, E_MAX, SOC_MIN,
    calculate_irradiance,
)

# ── simulation constants ────────────────────────────────────────────────────
MPC_INTERVAL_S   = 60          # how often MPC fires (seconds)
SIM_DT_S         = 1.0         # internal physics sub-step (seconds)
SAMPLING_DIST_M  = 50          # telemetry waypoint spacing (must match optimizer)
RACE_START_S     = 0
RACE_END_S       = 32400       # 9 hours
V_MIN_MS         = 60  / 3.6
V_MAX_MS         = 120 / 3.6
SOC_INITIAL      = 0.80
CONTROL_STOP_S   = 1800        # 30-min mandatory stop at Zeerust


# ── disturbance scenarios ───────────────────────────────────────────────────
SCENARIOS = {
    # (wind_mean_ms, wind_std_ms, cloud_amplitude, cloud_freq_hz, crr_noise_std)
    'ideal':    dict(wind_mean=0.0,  wind_std=0.0,  cloud_amp=0.00, cloud_freq=1/600, crr_noise=0.0000),
    'realistic':  dict(wind_mean=0.0,  wind_std=1.5,  cloud_amp=0.08, cloud_freq=1/600, crr_noise=0.0002),
    'headwind': dict(wind_mean=4.0,  wind_std=2.5,  cloud_amp=0.06, cloud_freq=1/400, crr_noise=0.0003),
    'cloudy':   dict(wind_mean=1.0,  wind_std=1.5,  cloud_amp=0.30, cloud_freq=1/300, crr_noise=0.0002),
    'worst':  dict(wind_mean=5.0,  wind_std=3.0,  cloud_amp=0.40, cloud_freq=1/200, crr_noise=0.0005),
}

# ── sensor noise levels (1-sigma) ──────────────────────────────────────────
SENSOR_NOISE = dict(
    speed_kmh   = 0.3,    # GPS/wheel encoder ±0.3 km/h
    soc_pct     = 0.4,    # BMS ±0.4%
    irradiance  = 15.0,   # pyranometer ±15 W/m²
    distance_m  = 2.0,    # GPS odometry ±2 m
    wind_ms     = 0.5,    # anemometer ±0.5 m/s
)


@dataclass
class CarState:
    """True (hidden) physics state of the car."""
    elapsed_t_s:   float = RACE_START_S     # seconds since 08:00
    distance_m:    float = 0.0              # metres travelled on base route
    speed_ms:      float = 0.0      # true speed in m/s
    soc:           float = SOC_INITIAL      # true SOC [0, 1]
    at_zeerust:    bool  = False            # currently in control stop
    race_finished: bool  = False


@dataclass
class TelemetryReading:
    """What the MPC actually sees — noisy sensor values."""
    elapsed_t_s:  float
    distance_m:   float
    speed_kmh:    float
    soc_pct:      float
    irradiance:   float
    wind_ms:      float
    waypoint_idx: int
    at_zeerust:   bool
    race_finished: bool


class RaceSimulator:
    """
    Physics-accurate race simulator.

    Internally steps at 1-second resolution. At each MPC tick (60 s),
    read_sensors() returns noisy measurements of the true state.

    Parameters
    ----------
    df_telemetry : pd.DataFrame
        The race telemetry CSV (must have 'distance_m' and 'gradient_deg').
    scenario : str
        One of 'realistic', 'headwind', 'cloudy', 'worst'.
    seed : int
        RNG seed for reproducibility.
    actuation_lag_s : float
        Time constant (seconds) of first-order speed tracking.
        0 = perfect tracking, 3 = sluggish motor response.
    verbose : bool
        Print a summary line at each MPC tick.
    """

    def __init__(
        self,
        df_telemetry: pd.DataFrame,
        scenario:        str   = 'realistic',
        seed:            int   = 42,
        actuation_lag_s: float = 3.0,
        verbose:         bool  = True,
    ):
        self.df       = df_telemetry.reset_index(drop=True)
        self.slopes   = df_telemetry['gradient_deg'].values
        self.dist_arr = df_telemetry['distance_m'].values
        self.route_end_m = float(self.dist_arr[-1])

        cfg = SCENARIOS[scenario]
        self.wind_mean    = cfg['wind_mean']
        self.wind_std     = cfg['wind_std']
        self.cloud_amp    = cfg['cloud_amp']
        self.cloud_freq   = cfg['cloud_freq']
        self.crr_noise    = cfg['crr_noise']
        self.lag_tau      = actuation_lag_s
        self.verbose      = verbose

        self.rng    = np.random.default_rng(seed)
        self.state  = CarState()

        # persistent wind state (correlated over time via AR(1) process)
        self._wind_ms = self.wind_mean
        self._phi     = 0.98    # AR coefficient — wind changes slowly

        # MPC tick counter
        self.tick = 0

        # history for post-analysis
        self.history = []
        self._substep_trace = []

    # ── public interface ────────────────────────────────────────────────────

    def read_sensors(self) -> TelemetryReading:
        """Return noisy sensor snapshot of current true state."""
        s = self.state
        rng = self.rng

        noisy_speed = max(0.0, s.speed_ms * 3.6
                         + rng.normal(0, SENSOR_NOISE['speed_kmh']))
        noisy_soc   = float(np.clip(
                         s.soc * 100 + rng.normal(0, SENSOR_NOISE['soc_pct']),
                         0, 100))
        noisy_irr   = float(np.clip(
                         self._true_irradiance(s.elapsed_t_s)
                         + rng.normal(0, SENSOR_NOISE['irradiance']),
                         0, 1200))
        noisy_dist  = max(0.0, s.distance_m
                         + rng.normal(0, SENSOR_NOISE['distance_m']))
        wp_idx      = int(np.clip(
                         round(noisy_dist / SAMPLING_DIST_M),
                         0, len(self.df) - 1))

        noisy_wind  = float(self._wind_ms + rng.normal(0, SENSOR_NOISE['wind_ms']))

        return TelemetryReading(
            elapsed_t_s  = s.elapsed_t_s,
            distance_m   = noisy_dist,
            speed_kmh    = noisy_speed,
            soc_pct      = noisy_soc,
            irradiance   = noisy_irr,
            wind_ms      = noisy_wind,
            waypoint_idx = wp_idx,
            at_zeerust   = s.at_zeerust,
            race_finished= s.race_finished,
        )

    def step(self, target_speed_kmh: float) -> TelemetryReading:
        """
        Advance simulation by one MPC interval (MPC_INTERVAL_S seconds).
        The car tries to track target_speed_kmh subject to lag and disturbances.
        Returns noisy sensor reading at the end of the interval.
        """
        if self.state.race_finished:
            return self.read_sensors()

        target_ms = float(np.clip(target_speed_kmh / 3.6, V_MIN_MS, V_MAX_MS))

        for _ in range(int(MPC_INTERVAL_S / SIM_DT_S)):
            self._physics_step(target_ms)
            self._substep_trace.append((self.state.elapsed_t_s, self.state.speed_ms))
            if self.state.race_finished or self.state.at_zeerust:
                break

        self.tick += 1
        reading = self.read_sensors()

        if self.verbose:
            self._print_tick(reading, target_speed_kmh)

        self.history.append({
            'tick':         self.tick,
            'elapsed_t_s':  self.state.elapsed_t_s,
            'distance_m':   self.state.distance_m,
            'true_speed_kmh': self.state.speed_ms * 3.6,
            'true_soc_pct': self.state.soc * 100,
            'target_kmh':   target_speed_kmh,
            'wind_ms':      self._wind_ms,
        })

        return reading

    def force_zeerust_stop(self):
        """
        Call this when the MPC decides the car has arrived at Zeerust.
        Simulates the mandatory 30-min control stop — solar still charges.
        """
        print(f"\n{'─'*52}")
        print(f"  ZEERUST CONTROL STOP — {CONTROL_STOP_S//60} min")
        print(f"  SOC on arrival : {self.state.soc*100:.2f}%")

        for _ in range(CONTROL_STOP_S):
            irr   = self._true_irradiance(self.state.elapsed_t_s)
            p_sol = PANEL_EFF * PANEL_AREA * irr
            dsoc  = ((p_sol - P_AUX) * SIM_DT_S) / E_MAX   # net charging
            self.state.soc = float(np.clip(self.state.soc + dsoc, SOC_MIN, 1.0))
            self.state.elapsed_t_s += SIM_DT_S

        self.state.at_zeerust = False
        print(f"  SOC after stop : {self.state.soc*100:.2f}%")
        print(f"{'─'*52}\n")

    def summary_dataframe(self) -> pd.DataFrame:
        """Return tick-level history as a DataFrame for post-analysis."""
        return pd.DataFrame(self.history)

    # ── internal physics ────────────────────────────────────────────────────

    def _physics_step(self, target_ms: float):
        """Advance true car state by SIM_DT_S seconds."""
        s = self.state

        # 1. Actuation lag — first-order tracking of target speed
        alpha    = SIM_DT_S / (self.lag_tau + SIM_DT_S)
        s.speed_ms = float(np.clip(
            (1 - alpha) * s.speed_ms + alpha * target_ms,
            V_MIN_MS, V_MAX_MS
        ))

        # 2. Wind disturbance — AR(1) correlated headwind
        self._wind_ms = (
            self._phi * self._wind_ms
            + (1 - self._phi) * self.wind_mean
            + self.rng.normal(0, self.wind_std * np.sqrt(1 - self._phi**2))
        )
        # effective speed through air (headwind increases drag)
        v_air = s.speed_ms + self._wind_ms

        # 3. Terrain — slope at current position
        wp_idx = int(np.clip(
            round(s.distance_m / SAMPLING_DIST_M), 0, len(self.slopes) - 1
        ))
        slope_deg = self.slopes[wp_idx]

        # 4. CRR perturbation (road surface variation)
        crr_eff = CRR + self.rng.normal(0, self.crr_noise)

        # 5. Power calculation with disturbed parameters
        irr   = self._true_irradiance(s.elapsed_t_s)
        p_sol = PANEL_EFF * PANEL_AREA * irr

        # use v_air for drag (headwind), s.speed_ms for grade/rolling (ground speed)
        p_drag    = 0.5 * RHO * CD * A_FRONTAL * v_air**2 * s.speed_ms
        p_roll    = crr_eff * MASS * 9.81 * s.speed_ms * np.cos(np.radians(slope_deg))
        p_grade   = MASS * 9.81 * s.speed_ms * np.sin(np.radians(slope_deg))
        p_mech    = p_drag + p_roll + p_grade

        if p_mech > 0:
            p_elec = p_mech / ETA_MOTOR + P_AUX - p_sol
        else:
            p_elec = p_mech * ETA_REGEN + P_AUX - p_sol

        # 6. SOC update
        dsoc = (p_elec * SIM_DT_S) / E_MAX
        s.soc = float(np.clip(s.soc - dsoc, SOC_MIN, 1.0))

        # 7. Advance position and time
        s.distance_m  += s.speed_ms * SIM_DT_S
        s.elapsed_t_s += SIM_DT_S

        # 8. Check end conditions
        if s.distance_m >= self.route_end_m:
            s.distance_m  = self.route_end_m
            s.at_zeerust  = True
            s.race_finished = False   # loops still to go
        if s.elapsed_t_s >= RACE_END_S:
            s.race_finished = True

    def _true_irradiance(self, t_s: float) -> float:
        """
        True irradiance with cloud-cover modulation.
        Cloud cover is a slow sinusoidal perturbation + high-freq flicker.
        """
        base = calculate_irradiance(t_s)
        # slow cloud envelope
        cloud_envelope = 1.0 - self.cloud_amp * (
            0.5 + 0.5 * np.sin(2 * np.pi * self.cloud_freq * t_s + 1.3)
        )
        # fast flicker (gusts of shadow)
        flicker = 1.0 + 0.03 * self.rng.standard_normal()
        return float(np.clip(base * cloud_envelope * flicker, 0, 1400))

    def _print_tick(self, r: TelemetryReading, target_kmh: float):
        t_hr = 8 + r.elapsed_t_s / 3600
        print(
            f"  t={t_hr:.3f}h  d={r.distance_m/1000:5.1f}km  "
            f"spd={r.speed_kmh:5.1f}→{target_kmh:.1f}km/h  "
            f"SOC={r.soc_pct:5.1f}%  "
            f"irr={r.irradiance:4.0f}W/m²  "
            f"wind={self._wind_ms:+.1f}m/s"
        )

    def substep_trace_arrays(self):
        """Return (t_s, speed_ms) numpy arrays at 1 Hz sub-step resolution."""
        if not self._substep_trace:
            return np.array([]), np.array([])
        t_s, v_ms = zip(*self._substep_trace)
        return np.asarray(t_s, dtype=float), np.asarray(v_ms, dtype=float)

"""
mpc_controller.py
=================
Real-time MPC controller for the solar car race.
Uses RaceSimulator as the telemetry source in place of real hardware.

Run standalone:
    python mpc_controller.py --scenario cloudy --seed 7

Or import and drive programmatically:
    from mpc_controller import MPCController
    ctrl = MPCController(df, slopes, scenario='headwind')
    ctrl.run()
"""

import argparse
import logging
import os
import time
import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Log file sits next to this script regardless of where Python is launched from
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagnostics.log')
logging.basicConfig(
    filename=_LOG_PATH,
    filemode='w',           # overwrite each run
    level=logging.DEBUG,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    force=True,             # reconfigure if logging was already set up by CasADi/scipy
)
_log = logging.getLogger('mpc')

from physics import (
    MASS, CRR, CD, A_FRONTAL, RHO,
    ETA_MOTOR, ETA_REGEN, P_AUX,
    PANEL_AREA, PANEL_EFF, E_MAX, SOC_MIN,
    calculate_irradiance,
)
from simulation_engine import RaceSimulator, MPC_INTERVAL_S, SAMPLING_DIST_M

# ── constants (must match optimizer) ───────────────────────────────────────
V_MIN_MS         = 60  / 3.6
V_MAX_MS         = 120 / 3.6
ACC_MAX          = 2.0
OPT_SPACING_M    = 200          # optimizer downsampling (every 4th waypoint)
RACE_END_S       = 32400
SOC_FLOOR_ALERT  = 0.25         # warn if SOC drops below 25%

BG, PANEL, ORANGE, WHITE, DIM = '#0D0D0D', '#1A1A1A', '#E87722', '#F0F0F0', '#555555'


def casadi_irradiance(t):
    """CasADi symbolic irradiance — same Gaussian as physics.py."""
    return 1073 * ca.exp(-0.5 * ((t - 14400) / 11600) ** 2)


class MPCController:
    """
    Drives the MPC loop against a RaceSimulator.

    At each 60-second tick:
      1. Read sensors from simulator
      2. Compare actual vs offline-plan state
      3. Re-solve IPOPT from current state over remaining route
      4. Return first action (target speed) to simulator
      5. Log everything for post-analysis
    """

    def __init__(
        self,
        df_telemetry:   pd.DataFrame,
        offline_v:      np.ndarray,     # v_optimal_base_ipopt.npy
        offline_soc:    np.ndarray,     # soc_hist_base_ipopt.npy
        offline_t:      np.ndarray,     # t_hist_base_ipopt.npy
        scenario:       str   = 'realistic',
        seed:           int   = 42,
        actuation_lag:  float = 3.0,
        verbose:        bool  = True,
    ):
        self.df      = df_telemetry
        self.slopes  = df_telemetry['gradient_deg'].values
        self.off_v   = offline_v
        self.off_soc = offline_soc
        self.off_t   = offline_t

        self.sim = RaceSimulator(
            df_telemetry    = df_telemetry,
            scenario        = scenario,
            seed            = seed,
            actuation_lag_s = actuation_lag,
            verbose         = verbose,
        )

        self.sim.state.speed_ms = float(offline_v[0])
        _log.info("Run started — scenario=%s seed=%s", scenario, seed)
        # MPC log — one row per tick
        self.log = []

        # last known good speed (fallback if solver fails)
        self._last_target_kmh = float(offline_v[0]*3.6)

        # build MPC solver once — reused every tick via CasADi parameters
        self._build_mpc_solver()

    # ── solver build (called once in __init__) ──────────────────────────────

    def _build_mpc_solver(self):
        N = 150  # fixed horizon size — 150 × 200 m = 30 km lookahead
        # Parameters packed as: [slopes_rad(N), t0, soc0, soc_ref(N), last_v]
        slopes_p  = ca.MX.sym('slopes_p',  N)
        t0_p      = ca.MX.sym('t0_p')
        soc0_p    = ca.MX.sym('soc0_p')
        soc_ref_p = ca.MX.sym('soc_ref_p', N)
        last_v_p  = ca.MX.sym('last_v_p')
        v_ref_p   = ca.MX.sym('v_ref_p', N)
        wind_p    = ca.MX.sym('wind_p')    # measured headwind [m/s], positive = headwind
        mask_p    = ca.MX.sym('mask_p', N)          # 1 = real waypoint, 0 = phantom
        P = ca.vertcat(slopes_p, t0_p, soc0_p, soc_ref_p, last_v_p, v_ref_p, wind_p,mask_p)

        # Decision variables — tau = time / RACE_END_S (normalised to O(1))
        v       = ca.MX.sym('v',       N)
        soc     = ca.MX.sym('soc',     N)
        tau     = ca.MX.sym('tau',     N)
        slack_v = ca.MX.sym('slack_v', N)
        X = ca.vertcat(v, soc, tau, slack_v)

        dt         = OPT_SPACING_M / v
        dtau       = dt / RACE_END_S
        irradiance = casadi_irradiance(tau * RACE_END_S)
        P_solar    = PANEL_EFF * PANEL_AREA * irradiance
        v_air      = v + wind_p                              # effective airspeed [m/s]
        P_drag     = 0.5 * RHO * CD * A_FRONTAL * v_air**2 * v  # F_drag × v_ground
        P_rolling  = CRR * MASS * 9.81 * v * ca.cos(slopes_p)  # slopes_p in radians
        P_grade    = MASS * 9.81 * v * ca.sin(slopes_p)
        P_mech     = P_drag + P_rolling + P_grade

        P_elec = ca.if_else(
            P_mech > 0,
            P_mech / ETA_MOTOR + P_AUX - P_solar,
            P_mech * ETA_REGEN + P_AUX - P_solar,
        )
        soc_drop = (P_elec * dt) / E_MAX

        tau_defect   = tau[1:] - (tau[:-1] + dtau[:-1])
        soc_defect   = soc[1:] - (soc[:-1] - soc_drop[:-1])
        accel        = (v[1:] - v[:-1]) / dt[:-1]
        jerk         = (accel[1:] - accel[:-1]) / dt[:-2]
        soft_v_limit = V_MIN_MS - v - slack_v

        g_vec = ca.vertcat(
            tau[0]  - t0_p / RACE_END_S,
            soc[0]  - soc0_p,
            tau_defect,
            soc_defect,
            accel,
            soft_v_limit,
        )

        soc_warning = ca.fmax(SOC_MIN + 0.05 - soc, 0)
        soc_critical = ca.fmax(SOC_MIN - soc, 0)

        cost = (5.0      * ca.sumsqr((soc - soc_ref_p) * mask_p)
              + 10.0     * ca.sumsqr((v   - v_ref_p)   * mask_p)
              + 5.0      * ca.sumsqr(accel * mask_p[:-1])
              + 50.0     * ca.sumsqr(v[0] - last_v_p)
              + 250.0    * ca.sumsqr(slack_v * mask_p)
              + 10.0     * ca.sumsqr((soc[-1] - soc_ref_p[-1]) * mask_p[-1])
              + 100.0    * ca.sumsqr(soc_warning  * mask_p)
              + 100000.0 * ca.sumsqr(soc_critical * mask_p))

        nlp  = {'x': X, 'f': cost, 'g': g_vec, 'p': P}
        opts = {
            'ipopt.linear_solver':   'mumps',
            'ipopt.max_iter':        500,
            'ipopt.tol':             1e-3,
            'ipopt.acceptable_tol':  1e-2,
            'ipopt.acceptable_iter': 5,
            'ipopt.print_level':     0,
            'ipopt.sb':              'yes',
        }
        self._solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)
        self._H      = N

        n_eq = 2 + 2 * (N - 1)
        self._lbx = [1.0] * N + [0.05] * N + [0.0] * N + [0.0] * N
        self._ubx = [V_MAX_MS] * N + [1.0] * N + [1.0] * N + [100.0] * N
        self._lbg = [0.0] * n_eq + [-ACC_MAX] * (N - 1) + [-1000.0] * N
        self._ubg = [0.0] * n_eq + [ ACC_MAX] * (N - 1) + [    0.0] * N

    # ── main loop ───────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Run the full MPC loop until race end or route completion.
        Returns a DataFrame of per-tick decisions and deviations.
        """
        print(f"\n{'='*56}")
        print(f"  MPC LOOP — scenario: {self.sim.__class__.__name__}")
        print(f"  Tick interval : {MPC_INTERVAL_S}s  |  Sub-step : 1s")
        print(f"{'='*56}\n")

        # initial reading before any movement
        reading = self.sim.read_sensors()

        while not reading.race_finished and not reading.at_zeerust:

            # ② Compare
            deviations = self._compare_to_plan(reading)

            # ③ Re-solve
            target_kmh, solve_ms, solver_ok = self._resolve(reading)

            # fallback if solver fails
            if not solver_ok:
                target_kmh = self._last_target_kmh
                print(f"  ⚠  Solver failed — holding {target_kmh:.1f} km/h")
                _log.warning("Solver non-OK at wp=%d  t=%.1fs  SOC=%.1f%%  holding=%.1f km/h",
                             reading.waypoint_idx, reading.elapsed_t_s,
                             reading.soc_pct, target_kmh)
            else:
                self._last_target_kmh = target_kmh

            # alerts
            if reading.soc_pct < SOC_FLOOR_ALERT * 100:
                print(f"  ⚡ LOW SOC ALERT: {reading.soc_pct:.1f}%")

            # log
            self.log.append({
                'tick':           len(self.log),
                'elapsed_t_s':    reading.elapsed_t_s,
                'elapsed_t_hr':   8 + reading.elapsed_t_s / 3600,
                'distance_km':    reading.distance_m / 1000,
                'actual_spd_kmh': reading.speed_kmh,
                'target_kmh':     target_kmh,
                'actual_soc_pct': reading.soc_pct,
                'plan_spd_kmh':   deviations['plan_spd_kmh'],
                'plan_soc_pct':   deviations['plan_soc_pct'],
                'spd_err_kmh':    deviations['spd_err_kmh'],
                'soc_err_pct':    deviations['soc_err_pct'],
                'time_err_s':     deviations['time_err_s'],
                'irradiance':     reading.irradiance,
                'wind_ms':        self.sim._wind_ms,
                'solve_ms':       solve_ms,
                'solver_ok':      solver_ok,
            })

            # ④ Step simulator with new target
            reading = self.sim.step(target_kmh)

        # ── Zeerust stop ────────────────────────────────────────────────────
        if reading.at_zeerust:
            self.sim.force_zeerust_stop()
            reading = self.sim.read_sensors()

        print(f"\n{'='*56}")
        print(f"  Base route complete")
        print(f"  Arrival   : {8 + reading.elapsed_t_s/3600:.3f} hrs")
        print(f"  Final SOC : {reading.soc_pct:.2f}%")
        print(f"{'='*56}\n")

        _log.info("Run complete — ticks=%d", len(self.log))
        n_failures = sum(1 for row in self.log if not row['solver_ok'])
        if n_failures:
            print(f"\n  ⚠ {n_failures} solver failure(s) — see {_LOG_PATH}")
        else:
            print(f"\n  ✔ No solver errors detected.")

        return pd.DataFrame(self.log)

    # ── step ② — compare to plan ────────────────────────────────────────────

    def _compare_to_plan(self, reading) -> dict:
        wp = np.clip(reading.waypoint_idx, 0, len(self.off_v) - 1)

        plan_spd = self.off_v[wp] * 3.6
        plan_soc = self.off_soc[wp] * 100
        plan_t   = self.off_t[wp]

        return {
            'plan_spd_kmh': plan_spd,
            'plan_soc_pct': plan_soc,
            'spd_err_kmh':  reading.speed_kmh - plan_spd,
            'soc_err_pct':  reading.soc_pct   - plan_soc,
            'time_err_s':   reading.elapsed_t_s - plan_t,
        }

    # ── step ③ — MPC re-solve ───────────────────────────────────────────────

    def _resolve(self, reading) -> tuple[float, float, bool]:
        """
        Call the pre-built IPOPT solver with updated parameter values.
        Returns (target_speed_kmh, solve_time_ms, success_bool).
        """
        wp_idx = reading.waypoint_idx
        N      = self._H

        # slopes at 200 m spacing from current position — pad at end of route
        slopes_raw = self.slopes[wp_idx::4][:N]
        if len(slopes_raw) < 3:
            return self._last_target_kmh, 0.0, True
        n_valid = len(slopes_raw)                    # real waypoints, pre-padding
        if len(slopes_raw) < N:
            slopes_raw = np.pad(slopes_raw, (0, N - len(slopes_raw)), mode='edge')

        valid_mask = np.zeros(N)
        valid_mask[:n_valid] = 1.0
        v_ref = self.off_v[wp_idx::4][:N]
        if len(v_ref) < N:
            v_ref = np.pad(v_ref, (0, N - len(v_ref)), mode='edge')

        actual_soc_frac = np.clip(reading.soc_pct / 100, SOC_MIN + 0.01, 1.0)
        actual_t_s      = reading.elapsed_t_s
        last_v_ms       = self._last_target_kmh / 3.6
        wind_ms         = float(reading.wind_ms)   # measured headwind [m/s]

        # Offline plan SOC at the same 200 m-spaced future waypoints.
        off_soc_future = self.off_soc[wp_idx::4][:N]
        if len(off_soc_future) < N:
            off_soc_future = np.pad(off_soc_future, (0, N - len(off_soc_future)), mode='edge')

        soc_ref = off_soc_future

        p_val = np.concatenate([
            np.radians(slopes_raw),          # slopes_p (radians)
            [actual_t_s, actual_soc_frac],   # t0_p, soc0_p
            soc_ref,                         # soc_ref_p
            [last_v_ms],                     # last_v_p
            v_ref,
            [wind_ms],                       # wind_p
            valid_mask,
        ])

        # warm start — v aligned to 200 m grid (Bug 5 fix: was off_v[ds_idx::4])
        v_warm = list(self.off_v[wp_idx::4][:N])
        if len(v_warm) < N:
            v_warm += [v_warm[-1] if v_warm else 80/3.6] * (N - len(v_warm))

        soc_warm, tau_warm = [], []
        soc_c = actual_soc_frac
        t_c   = actual_t_s
        for i in range(N):
            soc_warm.append(soc_c)
            tau_warm.append(t_c / RACE_END_S)
            dt_i  = OPT_SPACING_M / max(v_warm[i], V_MIN_MS)
            soc_c = max(soc_c - 0.001, SOC_MIN + 0.01)
            t_c  += dt_i

        x0 = v_warm + soc_warm + tau_warm + [0.0] * N

        t0 = time.perf_counter()
        try:
            sol    = self._solver(x0=x0, lbx=self._lbx, ubx=self._ubx,
                                  lbg=self._lbg, ubg=self._ubg, p=p_val)
            status = self._solver.stats()['return_status']
            ok     = status in ('Solve_Succeeded', 'Solved_To_Acceptable_Level')

            v_sol      = sol['x'].full().flatten()[:N]
            target_kmh = float(np.clip(v_sol[0] * 3.6, 60.0, 120.0))
            if ok:
                self._last_target_kmh = target_kmh
            else:
                target_kmh = self._last_target_kmh
            solve_ms   = (time.perf_counter() - t0) * 1000
            return target_kmh, solve_ms, ok

        except Exception as e:
            solve_ms = (time.perf_counter() - t0) * 1000
            _log.error("SOLVER EXCEPTION at wp=%d  t=%.1fs  SOC=%.1f%%: %s",
                       wp_idx, actual_t_s, reading.soc_pct, e, exc_info=True)
            return self._last_target_kmh, solve_ms, False


# ── plotting ────────────────────────────────────────────────────────────────

def apply_theme(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(DIM)
    ax.grid(True, color=DIM, linewidth=0.4, linestyle='--', alpha=0.6)


def plot_mpc_results(log: pd.DataFrame, offline_v, offline_soc, offline_t, df,sim):
    """Five-panel comparison: MPC actual vs offline plan."""
    t_off_hr      = offline_t / 3600 + 8
    t_off_hr_base = t_off_hr[:len(offline_v)]
    v_off_kmh     = offline_v * 3.6
    v_off_smooth  = uniform_filter1d(v_off_kmh, size=20)

    # planned (realistic) irradiance — same Gaussian the offline optimizer assumed
    t_s_log    = (log['elapsed_t_hr'] - 8) * 3600
    planned_irr = 1073 * np.exp(-0.5 * ((t_s_log - 14400) / 11600) ** 2)

    fig, axes = plt.subplots(7, 1, figsize=(14, 26), sharex=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle('MPC vs Offline Plan — Race Simulation', color=WHITE,
                 fontsize=13, y=0.99)

    # 1 — speed
    ax = axes[0]
    apply_theme(ax)
    ax.plot(t_off_hr_base, v_off_smooth, color=DIM, linewidth=1.2,
            linestyle='--', label='Offline plan', zorder=2)
    ax.plot(log['elapsed_t_hr'], log['actual_spd_kmh'], color=ORANGE,
            linewidth=1.4, label='Actual (sim)', zorder=3)
    ax.step(log['elapsed_t_hr'], log['target_kmh'], color='#76DE7F',
            linewidth=1.0, where='post', label='MPC target', alpha=0.8, zorder=3)
    ax.set_ylabel('Speed (km/h)', color=WHITE)
    ax.set_title('Speed profile', color=WHITE, pad=6)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE,
              edgecolor=DIM, facecolor=PANEL)

    # 2 — SOC
    ax = axes[1]
    apply_theme(ax)
    t_end_hr = log['elapsed_t_hr'].iloc[-1]
    off_mask = t_off_hr <= t_end_hr
    ax.plot(t_off_hr[off_mask], offline_soc[off_mask] * 100, color=DIM,
            linewidth=1.2, linestyle='--', label='Offline plan', zorder=2)
    ax.plot(log['elapsed_t_hr'], log['actual_soc_pct'], color=ORANGE,
            linewidth=1.4, label='Actual (sim)', zorder=3)
    ax.axhline(SOC_MIN * 100, color='#E24B4A', linewidth=0.8,
               linestyle='--', alpha=0.7, label='SOC floor')
    ax.set_ylabel('SOC (%)', color=WHITE)
    ax.set_title('Battery state of charge', color=WHITE, pad=6)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE,
              edgecolor=DIM, facecolor=PANEL)

    # 3 — deviations from plan
    ax = axes[2]
    apply_theme(ax)
    ax.axhline(0, color=DIM, linewidth=0.5)
    ax.plot(log['elapsed_t_hr'], log['soc_err_pct'], color=ORANGE,
            linewidth=1.2, label='SOC error (%)')
    ax.plot(log['elapsed_t_hr'], log['spd_err_kmh'], color='#76DE7F',
            linewidth=1.2, label='Speed error (km/h)', alpha=0.8)
    ax.set_ylabel('Deviation from plan', color=WHITE)
    ax.set_title('MPC tracking error vs offline reference', color=WHITE, pad=6)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE,
              edgecolor=DIM, facecolor=PANEL)

    # 4 — irradiance: actual vs planned, with deficit shading
    ax = axes[3]
    apply_theme(ax)
    ax.plot(log['elapsed_t_hr'], planned_irr, color='#F5C518', linewidth=1.2,
            linestyle='--', label='Planned (realistic)', zorder=2)
    ax.plot(log['elapsed_t_hr'], log['irradiance'], color=ORANGE, linewidth=1.4,
            label='Actual (sim)', zorder=3)
    ax.fill_between(log['elapsed_t_hr'], log['irradiance'], planned_irr,
                    where=(planned_irr > log['irradiance']),
                    alpha=0.25, color='red', label='Solar deficit')
    ax.set_ylabel('Irradiance (W/m²)', color=WHITE)
    ax.set_title('Solar irradiance: planned vs actual', color=WHITE, pad=6)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE,
              edgecolor=DIM, facecolor=PANEL)

    # 5 — wind speed
    ax = axes[4]
    apply_theme(ax)
    ax.axhline(0, color=DIM, linewidth=0.5)
    ax.plot(log['elapsed_t_hr'], log['wind_ms'], color='#76B5E8', linewidth=1.2,
            label='Wind speed (m/s)  [+ve = headwind]')
    ax.fill_between(log['elapsed_t_hr'], log['wind_ms'], 0,
                    where=(log['wind_ms'] > 0), alpha=0.2, color='red',  label='Headwind')
    ax.fill_between(log['elapsed_t_hr'], log['wind_ms'], 0,
                    where=(log['wind_ms'] < 0), alpha=0.2, color='green', label='Tailwind')
    ax.set_ylabel('Wind (m/s)', color=WHITE)
    ax.set_xlabel('Time of day (hrs)', color=WHITE)
    ax.set_title('Wind speed', color=WHITE, pad=6)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE,
              edgecolor=DIM, facecolor=PANEL)

   # 6 — acceleration vs time (1 Hz sub-step trace)
    ax = axes[5]
    apply_theme(ax)
    t_fine_s, v_fine_ms = sim.substep_trace_arrays()
    ax.axhline(0,        color=DIM,      linewidth=0.5)
    ax.axhline( ACC_MAX, color='#E24B4A', linewidth=0.7, linestyle='--', alpha=0.7)
    ax.axhline(-ACC_MAX, color='#E24B4A', linewidth=0.7, linestyle='--', alpha=0.7,
               label=f'±{ACC_MAX} m/s² limit')
    if len(t_fine_s) > 2:
        t_fine_hr = 8 + t_fine_s / 3600
        a_fine    = np.gradient(v_fine_ms, t_fine_s)   # dv/dt using true 1 s spacing
        ax.plot(t_fine_hr, a_fine, color=ORANGE, linewidth=0.8,
                label='Acceleration (m/s²)')
    ax.set_ylabel('Acceleration (m/s²)', color=WHITE)
    ax.set_title('Longitudinal acceleration', color=WHITE, pad=6)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE, edgecolor=DIM, facecolor=PANEL)

    # 7 — gradient vs distance (mapped through distance_km log column)
    ax = axes[6]
    apply_theme(ax)
    dist_m_df  = df['distance_m'].values if 'distance_m' in df.columns else np.arange(len(df)) * 50
    grad_deg   = df['gradient_deg'].values
    log_dist_m = log['distance_km'] * 1000
    grad_interp = np.interp(log_dist_m, dist_m_df, grad_deg)
    ax.axhline(0, color=DIM, linewidth=0.5)
    ax.fill_between(log['elapsed_t_hr'], grad_interp, 0,
                    where=(grad_interp > 0), alpha=0.3, color='red',   label='Uphill')
    ax.fill_between(log['elapsed_t_hr'], grad_interp, 0,
                    where=(grad_interp < 0), alpha=0.3, color='green', label='Downhill')
    ax.plot(log['elapsed_t_hr'], grad_interp, color=WHITE, linewidth=0.8, alpha=0.6)
    ax.set_ylabel('Gradient (°)', color=WHITE)
    ax.set_xlabel('Time of day (hrs)', color=WHITE)
    ax.set_title('Route gradient', color=WHITE, pad=6)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE, edgecolor=DIM, facecolor=PANEL)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = 'mpc_simulation_results.png'
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved → {out}")


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solar car MPC simulation')
    parser.add_argument('--scenario', default='realistic',
                        choices=['ideal', 'realistic', 'headwind', 'cloudy', 'worst'])
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--lag',     type=float, default=3.0,
                        help='Motor actuation lag time constant (s)')
    parser.add_argument('--quiet',   action='store_true',
                        help='Suppress per-tick console output')
    args = parser.parse_args()

    print(f"Loading telemetry and offline plan...")
    df          = pd.read_csv(r"sasol\mpc_solver\data\final_race_telemetry.csv")
    offline_v   = np.load(r'sasol\mpc_solver\v_optimal_base_ipopt.npy')
    offline_soc = np.load(r'sasol\mpc_solver\soc_hist_base_ipopt.npy')
    offline_t   = np.load(r'sasol\mpc_solver\t_hist_base_ipopt.npy')

    ctrl = MPCController(
        df_telemetry  = df,
        offline_v     = offline_v,
        offline_soc   = offline_soc,
        offline_t     = offline_t,
        scenario      = args.scenario,
        seed          = args.seed,
        actuation_lag = args.lag,
        verbose       = not args.quiet,
    )

    log = ctrl.run()

    print("\nMPC Log summary:")
    print(log[['elapsed_t_hr', 'distance_km', 'actual_spd_kmh',
               'target_kmh', 'actual_soc_pct',
               'soc_err_pct', 'solve_ms']].to_string(index=False))

    import datetime
    log_path = "mpc_log.csv"
    log.to_csv(log_path, index=False)
    print(f"\nSaved → {log_path}")

    plot_mpc_results(log, offline_v, offline_soc, offline_t, df, ctrl.sim)

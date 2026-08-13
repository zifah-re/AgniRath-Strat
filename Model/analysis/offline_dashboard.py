"""
analysis/offline_dashboard.py — offline strategy dashboard.

Purpose (11 Aug 2026 conversation): the single-day optimizer produces a
velocity profile but Prahlad has no way to *look* at the output. Junior C
still hasn't built the dashboard (block 3.3). This file is a self-contained
Dash+Plotly port of Kevin's race_completion/dashboard.py, adapted so it
consumes the L2 solver's output directly.

Two entry points:

  build_run_csv(...)      Take a solved velocity profile + route + car +
                          weather providers, integrate through core.physics,
                          and write the same 7-column CSV Kevin's dashboard
                          expected: CumulativeDistance, Velocity,
                          Acceleration, Battery, EnergyConsumption, Solar,
                          Time. This is the L2 -> plot bridge.

  run_dashboard(...)      Load that CSV (or take the DataFrame directly),
                          spin up the Dash app on localhost:8050 with all
                          the panels Kevin had, plus a few upgrades:
                            * wind speed & along-track component
                            * slope %
                            * marker overlays for stops (control_stop,
                              loop_stop, long_charge, swap)
                            * config parameters panel populated from our
                              CarState + race_config (not Kevin's constants)

Usage
-----
    # After solving:
    from optimizers.singleday import solve
    out = solve(route, car, solar_provider, wind_provider,
                day_index=0, start_soc_pct=100.0,
                alpha_next_day_pct=40.0, loops_committed=[])

    from analysis.offline_dashboard import build_run_csv, run_dashboard
    df = build_run_csv(out, route, car, solar_provider, wind_provider,
                       day_index=0, start_soc_pct=100.0,
                       out_path="run_dat.csv")
    run_dashboard(df)             # or run_dashboard(csv_path="run_dat.csv")

CLI:
    python -m analysis.offline_dashboard --csv run_dat.csv
    python -m analysis.offline_dashboard --csv run_dat.csv --port 8051

Dependencies: dash, plotly (add to requirements.txt: dash>=2.15, plotly>=5.20).

NOTE this module does NOT depend on simulator/forward_sim.py — that file is
still the block-1 stub in the repo, and the L2 solver needs to see its plots
today. The integration loop here is deliberately small and self-contained,
using core.physics.net_power + core.battery.Battery directly, so the moment
forward_sim.py's real implementation lands (block 3.1), build_run_csv can be
swapped to call it instead. That swap is the only expected change.
"""

from __future__ import annotations

import argparse
import pathlib
import typing as _t

import numpy as np
import pandas as pd

from configs import race_config as RC
from configs.car_config import CarState
from core import physics
from core import solar as solar_mod
from core import wind as wind_mod
from core.battery import Battery
from core.route import Route


# ===========================================================================
# 1. L2 output -> Kevin-format CSV
# ===========================================================================

# Column names match Kevin's dashboard.py exactly, so the loader on that end
# doesn't have to translate.
CSV_COLUMNS = (
    "CumulativeDistance",   # metres, one point per segment boundary
    "Velocity",             # m/s, target speed the driver holds
    "Acceleration",         # m/s^2, computed from v_i -> v_{i+1}
    "Battery",              # percent SOC
    "EnergyConsumption",    # Wh, per-segment battery draw (positive = drain)
    "Solar",                # Wh, per-segment solar gain (positive = charge)
    "Time",                 # seconds since day start (RC.day_start_time_s)
)


def build_run_csv(
    solve_output: dict,
    route: Route,
    car: CarState,
    solar_provider,
    wind_provider=None,
    *,
    day_index: int = 0,
    start_soc_pct: float = 100.0,
    out_path: str | pathlib.Path | None = "run_dat.csv",
    step_m: float = 100.0,
    apply_solar_geometry: bool = True,
) -> pd.DataFrame:
    """Integrate the solved velocity profile and produce the dashboard CSV.

    solve_output : dict from optimizers/singleday.py:solve(). Uses only
                   `v_kmh` and `seg_start_m`. Other fields (final_soc_pct,
                   total_time_s, diagnostics) are ignored here — this
                   function reproduces them from scratch by running the
                   integration itself, which is deliberate: the plot then
                   shows the ACTUAL physics-integrated trajectory, not the
                   optimizer's internal (possibly cached / approximated)
                   estimate.

    step_m       : integration substep inside each control segment. 100 m
                   matches solver_config.ENERGY_GRID_M and captures the
                   route's slope variation without exploding runtime.

    apply_solar_geometry : pass the slope/bearing incidence factor from
                   core.solar.slope_geometry_factor into net_power. Turn
                   off if you want a Kevin-style horizontal-array baseline.
    """
    if wind_provider is None:
        wind_provider = wind_mod.ConstantWindProvider(0.0, 0.0)

    v_kmh = np.asarray(solve_output["v_kmh"], dtype=float)
    seg_start_m = np.asarray(solve_output["seg_start_m"], dtype=float)
    if len(v_kmh) != len(seg_start_m):
        raise ValueError(
            f"v_kmh and seg_start_m must be same length "
            f"({len(v_kmh)} vs {len(seg_start_m)})")

    # Segment ends: start of next segment, last segment ends at route total.
    seg_end_m = np.concatenate([seg_start_m[1:], [route.total_m]])

    bat = Battery(car, start_soc_pct)
    t_s = float(RC.day_start_time_s(day_index))
    day_of_year = 253 + day_index      # 10 Sep 2026 == DOY 253

    # Buffers: one point per segment BOUNDARY -> len(v_kmh) + 1 points.
    n_pts = len(v_kmh) + 1
    dist_pts = np.zeros(n_pts)
    vel_pts = np.zeros(n_pts)
    acc_pts = np.full(n_pts, np.nan)
    batt_pts = np.zeros(n_pts)
    energy_pts = np.full(n_pts, np.nan)
    solar_pts = np.full(n_pts, np.nan)
    time_pts = np.zeros(n_pts)

    dist_pts[0] = float(seg_start_m[0])
    vel_pts[0] = float(v_kmh[0]) / 3.6
    batt_pts[0] = bat.soc_pct
    time_pts[0] = t_s

    for i in range(len(v_kmh)):
        v_ms = max(float(v_kmh[i]) / 3.6, 0.1)
        v_next_ms = (max(float(v_kmh[i + 1]) / 3.6, 0.1)
                     if i + 1 < len(v_kmh) else v_ms)
        seg_len_m = float(seg_end_m[i] - seg_start_m[i])
        if seg_len_m <= 0.0:
            # zero-length segment — advance the buffers, skip physics
            dist_pts[i + 1] = dist_pts[i]
            vel_pts[i + 1] = v_next_ms
            batt_pts[i + 1] = bat.soc_pct
            time_pts[i + 1] = t_s
            energy_pts[i + 1] = 0.0
            solar_pts[i + 1] = 0.0
            acc_pts[i + 1] = 0.0
            continue

        # Sub-integrate inside this control segment on step_m to pick up
        # slope variation (Kevin's model was already at native ~50 m grid;
        # ours is 100 m by default).
        n_sub = max(1, int(round(seg_len_m / step_m)))
        sub_len_km = (seg_len_m / n_sub) / 1000.0

        energy_wh = 0.0
        solar_wh = 0.0
        t_seg_start = t_s
        x_local = seg_start_m[i]

        for _ in range(n_sub):
            slope = float(route.slope_pct_at(x_local))
            bearing = float(route.bearing_deg_at(x_local))
            lat, lon = route.latlon_at(x_local)
            ghi = float(solar_provider.ghi_wm2(t_s, x_local))

            w_speed, w_dir = wind_provider.wind(t_s, x_local)
            w_along = wind_mod.along_track_ms(w_speed, w_dir, bearing)
            _, yaw = wind_mod.relative_wind(v_ms, w_speed, w_dir, bearing)

            geom = 1.0
            if apply_solar_geometry:
                t_solar = solar_mod.solar_time_s(t_s, float(lon), day_of_year)
                geom = solar_mod.slope_geometry_factor(
                    float(lat), day_of_year, t_solar, slope, bearing,
                    car.panel_tilt_base_deg)

            # Linear interpolation of v across the substeps -> smooth accel
            # inside the control segment.
            frac = (x_local - seg_start_m[i]) / max(seg_len_m, 1e-9)
            v_here = v_ms + frac * (v_next_ms - v_ms)
            v_here_next = v_ms + (frac + 1.0 / n_sub) * (v_next_ms - v_ms)

            p_net, dt_s = physics.net_power(
                car, v_here, v_here_next, slope, ghi, sub_len_km,
                wind_along_ms=w_along, yaw_deg=yaw,
                solar_geom_factor=geom)
            p_net = float(p_net)
            dt_s = float(dt_s)

            # Decompose into solar gain and electrical draw for the CSV
            # (Kevin logged them separately). physics.net_power returns
            # p_solar - p_electric - p_idle; recover p_solar directly.
            p_solar = car.array_area_m2 * car.array_efficiency * ghi * geom
            wh_seg = p_net * dt_s / 3600.0     # net into pack (Wh)
            wh_solar = p_solar * dt_s / 3600.0

            # Kevin's "EnergyConsumption" is what LEAVES the pack — battery
            # draw before solar is added back. Solar column is standalone.
            wh_draw = wh_solar - wh_seg        # >0 when net draining
            energy_wh += max(wh_draw, 0.0)     # keep sign convention Kevin
            solar_wh += wh_solar               #   used (positive = gain)

            bat.apply_energy_wh(wh_seg)
            t_s += dt_s
            x_local += sub_len_km * 1000.0

        dt_total = t_s - t_seg_start
        dist_pts[i + 1] = float(seg_end_m[i])
        vel_pts[i + 1] = v_next_ms
        batt_pts[i + 1] = bat.soc_pct
        time_pts[i + 1] = t_s
        energy_pts[i + 1] = energy_wh
        solar_pts[i + 1] = solar_wh
        acc_pts[i + 1] = ((v_next_ms - v_ms) / dt_total
                          if dt_total > 1e-9 else 0.0)

    df = pd.DataFrame({
        "CumulativeDistance": dist_pts,
        "Velocity": vel_pts,
        "Acceleration": acc_pts,
        "Battery": batt_pts,
        "EnergyConsumption": energy_pts,
        "Solar": solar_pts,
        "Time": time_pts,
    })

    if out_path is not None:
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df


# ===========================================================================
# 2. Dash app — Kevin's layout, adapted to our config
# ===========================================================================

_CUSTOM_STYLES = {
    "font-family": '"Quicksand", sans-serif',
    "background-color": "#f0f0f0",
    "text-align": "center",
    "margin": "10px",
    "padding": "10px",
    "border": "1px solid #ccc",
    "border-radius": "5px",
}
_EXTERNAL_STYLESHEETS = [
    "https://fonts.googleapis.com/css2?family=Quicksand:wght@300..700"
    "&family=Roboto+Slab:wght@100..900"
    "&family=Space+Grotesk:wght@300..700&display=swap",
]


def _config_panel(car: CarState) -> _t.Any:
    """Config parameters panel populated from CarState (not Kevin's constants).

    Split into three columns like Kevin's original for consistency.
    """
    from dash import html
    return html.Div([
        html.H2("Configuration Parameters",
                style={"text-align": "center",
                       "font-family": '"Space Grotesk", sans-serif'}),
        html.Div([
            html.Div([
                html.P(f"Battery Nominal: {car.battery_nominal_wh:.0f} Wh"),
                html.P(f"Battery Usable:  {car.battery_usable_wh:.0f} Wh"),
                html.P(f"SOC floor: {car.soc_min_pct:.0f}%"),
                html.P(f"Mass: {car.mass_kg:.0f} kg"),
                html.P(f"Packs: {car.n_packs}"),
            ], style={"width": "33%"}),
            html.Div([
                html.P(f"CdA: {car.cda_m2}"),
                html.P(f"Crr: {car.crr}"),
                html.P(f"Solar Area: {car.array_area_m2} m^2"),
                html.P(f"Solar Efficiency: {car.array_efficiency}"),
                html.P(f"Panel Tilt Base: {car.panel_tilt_base_deg} deg"),
            ], style={"width": "33%"}),
            html.Div([
                html.P(f"Motor Eff: {car.motor_eff}"),
                html.P(f"Regen Eff: {car.regen_eff}"),
                html.P(f"Idle Draw: {car.p_idle_w} W"),
                html.P(f"Max Speed: {car.v_max_ms * 3.6:.1f} km/h"),
                html.P(f"Max Accel: {car.a_max_ms2} m/s^2"),
            ], style={"width": "33%"}),
        ], style={"display": "flex", "justify-content": "center",
                  "text-align": "left"}),
    ], style={"width": "80%", "margin": "auto", "padding": "20px",
              "border": "1px solid #ccc", "border-radius": "5px"})


def _time_str(t_s: float) -> str:
    """Kevin's time-of-day format."""
    h = int(t_s // 3600)
    m = int((t_s % 3600) // 60)
    s = (t_s % 3600) % 60
    return f"{h}hrs {m}mins {s:.3f}secs"


def create_app(df: pd.DataFrame, car: CarState,
               day_index: int = 0,
               stops: list | None = None) -> _t.Any:
    """Build the Dash app from a run_dat DataFrame + CarState.

    stops : optional list of dicts [{"kind","x_m","t_s","duration_s"}] for
            annotating control_stop / loop_stop / long_charge markers on
            distance-axis plots. Pass the events list from a SimResult, or
            construct it manually from the L2 output. Ignored if None.
    """
    import dash
    from dash import dcc, html
    import plotly.graph_objs as go

    dist = df["CumulativeDistance"].to_numpy()
    vel = df["Velocity"].to_numpy()
    acc = df["Acceleration"].to_numpy()
    batt = df["Battery"].to_numpy()
    energy = df["EnergyConsumption"].to_numpy()
    solar = df["Solar"].to_numpy()
    time_arr = df["Time"].to_numpy()

    day_start = RC.day_start_time_s(day_index)
    day_finish = RC.day_finish_time_s(day_index)
    elapsed_s = time_arr[-1] - day_start if len(time_arr) else 0.0
    total_dist_m = dist[-1] - dist[0] if len(dist) else 0.0
    avg_vel = total_dist_m / elapsed_s if elapsed_s > 0 else 0.0

    # Stop markers on distance axis
    stop_shapes: list = []
    stop_annos: list = []
    if stops:
        colors = {
            "control_stop": "rgba(200, 0, 0, 0.15)",
            "loop_stop": "rgba(0, 100, 200, 0.15)",
            "long_charge": "rgba(0, 150, 0, 0.15)",
            "swap": "rgba(150, 100, 0, 0.15)",
        }
        for s in stops:
            k = s.get("kind") or s.get("stop_kind")
            x = s.get("x_m")
            if k is None or x is None:
                continue
            stop_shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=x - 500, x1=x + 500, y0=0, y1=1,
                fillcolor=colors.get(k, "rgba(120,120,120,0.15)"),
                line=dict(width=0), layer="below"))
            stop_annos.append(dict(x=x, y=1.02, xref="x", yref="paper",
                                    text=k, showarrow=False,
                                    font=dict(size=9)))

    app = dash.Dash(__name__, external_stylesheets=_EXTERNAL_STYLESHEETS)

    def _fig(traces, title, xtitle="Distance (m)", ytitle=""):
        return dict(data=traces,
                    layout=go.Layout(title=title,
                                     xaxis=dict(title=xtitle),
                                     yaxis=dict(title=ytitle),
                                     shapes=stop_shapes,
                                     annotations=stop_annos))

    app.layout = html.Div([
        # Header
        html.Div([
            html.H1(f"Strategy Analysis Dashboard — Day {day_index + 1}",
                    style={"text-align": "center",
                           "font-family": '"Roboto Slab", serif'}),
        ], style={"display": "flex", "justify-content": "center",
                  "align-items": "center"}),

        _config_panel(car),

        # Summary + analysis
        html.Div([
            html.Div([
                html.H2("Summary",
                        style={"text-align": "center",
                               "font-family": '"Space Grotesk", sans-serif'}),
                html.P(f"Total Distance: {total_dist_m / 1000:.3f} km"),
                html.P(f"Time Taken: {_time_str(elapsed_s)}"),
                html.P(f"Finish Time: {_time_str(time_arr[-1])} "
                       f"(official close {day_finish / 3600:.0f}:00)"),
                html.P(f"No. of points: {len(dist)}"),
            ], style={"width": "30%", "display": "inline-block",
                      "vertical-align": "top", **_CUSTOM_STYLES}),
            html.Div([
                html.H2("Data Analysis",
                        style={"text-align": "center",
                               "font-family": '"Space Grotesk", sans-serif'}),
                html.P(f"Max Velocity: {max(vel):.3f} m/s "
                       f"({max(vel) * 3.6:.2f} km/h)"),
                html.P(f"Avg Velocity: {avg_vel:.3f} m/s "
                       f"({avg_vel * 3.6:.2f} km/h)"),
                html.P(f"Start SOC: {batt[0]:.2f}%"),
                html.P(f"Final SOC: {batt[-1]:.2f}%"),
                html.P(f"Min SOC:   {np.nanmin(batt):.2f}%"),
                html.P(f"Total Solar Gain: "
                       f"{np.nansum(solar):.1f} Wh"),
                html.P(f"Total Energy Drawn: "
                       f"{np.nansum(energy):.1f} Wh"),
            ], style={"width": "60%", "display": "inline-block",
                      "vertical-align": "top", **_CUSTOM_STYLES}),
        ], style={"width": "93%", "display": "flex",
                  "justify-content": "center"}),

        # Graphs
        html.Div([
            dcc.Graph(id="velocity-profile",
                      figure=_fig([
                          go.Scatter(x=dist, y=vel, mode="lines+markers",
                                     name="Velocity (m/s)"),
                          go.Scatter(x=[dist[0], dist[-1]],
                                     y=[car.v_max_ms] * 2, mode="lines",
                                     name="Max Velocity",
                                     line=dict(color="red", dash="dot")),
                          go.Scatter(x=[dist[0], dist[-1]],
                                     y=[avg_vel] * 2, mode="lines",
                                     name="Avg Velocity",
                                     line=dict(color="green", dash="dot")),
                      ], "Velocity Profile (m/s)", ytitle="m/s"),
                      style={"width": "93%", "display": "inline-block",
                             **_CUSTOM_STYLES}),

            dcc.Graph(id="velocity-kmh",
                      figure=_fig([
                          go.Scatter(x=dist, y=vel * 3.6,
                                     mode="lines+markers",
                                     name="Velocity (km/h)"),
                          go.Scatter(x=[dist[0], dist[-1]],
                                     y=[car.v_max_ms * 3.6] * 2,
                                     mode="lines", name="Max Velocity",
                                     line=dict(color="red", dash="dot")),
                          go.Scatter(x=[dist[0], dist[-1]],
                                     y=[avg_vel * 3.6] * 2, mode="lines",
                                     name="Avg Velocity",
                                     line=dict(color="green", dash="dot")),
                      ], "Velocity Profile (km/h)", ytitle="km/h"),
                      style={"width": "93%", "display": "inline-block",
                             **_CUSTOM_STYLES}),

            dcc.Graph(id="acceleration-profile",
                      figure=_fig([
                          go.Scatter(x=dist[1:], y=acc[1:],
                                     mode="lines+markers",
                                     name="Acceleration"),
                          go.Scatter(x=[dist[0], dist[-1]],
                                     y=[car.a_max_ms2] * 2, mode="lines",
                                     name="Max |a|",
                                     line=dict(color="red", dash="dot")),
                          go.Scatter(x=[dist[0], dist[-1]],
                                     y=[-car.a_max_ms2] * 2, mode="lines",
                                     name="-Max |a|", showlegend=False,
                                     line=dict(color="red", dash="dot")),
                      ], "Acceleration Profile", ytitle="m/s^2"),
                      style={"width": "45%", "display": "inline-block",
                             **_CUSTOM_STYLES}),

            dcc.Graph(id="battery-profile",
                      figure=_fig([
                          go.Scatter(x=dist, y=batt, mode="lines+markers",
                                     name="Battery %"),
                          go.Scatter(x=[dist[0], dist[-1]],
                                     y=[car.soc_max_pct] * 2, mode="lines",
                                     name="Max",
                                     line=dict(color="red", dash="dot")),
                          go.Scatter(x=[dist[0], dist[-1]],
                                     y=[car.soc_min_pct] * 2, mode="lines",
                                     name="Floor",
                                     line=dict(color="orange", dash="dot")),
                      ], "Battery Level Profile", ytitle="Charge (%)"),
                      style={"width": "45%", "display": "inline-block",
                             **_CUSTOM_STYLES}),

            dcc.Graph(id="energy-consumption",
                      figure=_fig([
                          go.Scatter(x=dist[1:], y=energy[1:],
                                     mode="lines+markers",
                                     name="Energy (Wh)"),
                      ], "Segment Energy Consumption", ytitle="Wh"),
                      style={"width": "45%", "display": "inline-block",
                             **_CUSTOM_STYLES}),

            dcc.Graph(id="net-energy",
                      figure=_fig([
                          go.Scatter(x=dist[1:],
                                     y=np.nancumsum(energy[1:]),
                                     mode="lines+markers",
                                     name="Net Energy (Wh)"),
                      ], "Cumulative Energy Consumption", ytitle="Wh"),
                      style={"width": "45%", "display": "inline-block",
                             **_CUSTOM_STYLES}),

            dcc.Graph(id="solar-profile",
                      figure=_fig([
                          go.Scatter(x=dist[1:], y=solar[1:],
                                     mode="lines+markers",
                                     name="Solar (Wh)"),
                      ], "Segment Solar Gain", ytitle="Wh"),
                      style={"width": "45%", "display": "inline-block",
                             **_CUSTOM_STYLES}),

            dcc.Graph(id="net-solar",
                      figure=_fig([
                          go.Scatter(x=dist[1:],
                                     y=np.nancumsum(solar[1:]),
                                     mode="lines+markers",
                                     name="Net Solar (Wh)"),
                      ], "Cumulative Solar Gain", ytitle="Wh"),
                      style={"width": "45%", "display": "inline-block",
                             **_CUSTOM_STYLES}),

            # Time vs distance with official-window overlay
            dcc.Graph(id="time-profile",
                      figure=dict(
                          data=[
                              go.Scatter(x=dist, y=time_arr / 3600,
                                         mode="lines+markers",
                                         name="Time (hrs)"),
                              go.Scatter(x=[dist[0], dist[-1]],
                                         y=[day_finish / 3600] * 2,
                                         mode="lines", name="Official close",
                                         line=dict(color="red", dash="dot")),
                              go.Scatter(
                                  x=[dist[0], dist[-1]],
                                  y=[RC.FINISH_CUTOFF_ABS_S / 3600] * 2,
                                  mode="lines", name="Absolute cutoff",
                                  line=dict(color="darkred", dash="dash")),
                          ],
                          layout=go.Layout(
                              title="Time vs Distance",
                              xaxis=dict(title="Distance (m)"),
                              yaxis=dict(title="Total Time (hrs)"),
                              shapes=stop_shapes,
                              annotations=stop_annos)),
                      style={"width": "93%", "display": "inline-block",
                             **_CUSTOM_STYLES}),
        ], style={"display": "flex", "flex-wrap": "wrap",
                  "justify-content": "center"}),
    ], style={"background-color": "#ffffff", "padding": "20px"})

    return app


def run_dashboard(df: pd.DataFrame | None = None,
                  csv_path: str | pathlib.Path | None = None,
                  car: CarState | None = None,
                  day_index: int = 0,
                  stops: list | None = None,
                  host: str = "127.0.0.1",
                  port: int = 8050,
                  debug: bool = True) -> None:
    """Launch the Dash app.

    Provide either `df` or `csv_path`. `car` defaults to configs.car_config.
    default_car() so a bare CLI call still works — pass the real degraded
    CarState if you want the config panel to reflect it.
    """
    if df is None:
        if csv_path is None:
            raise ValueError("provide df or csv_path")
        df = pd.read_csv(csv_path).fillna(0)
    if car is None:
        from configs.car_config import default_car
        car = default_car()
    app = create_app(df, car, day_index=day_index, stops=stops)
    app.run(host=host, port=port, debug=debug)

def load_day_dir(day_dir: str | pathlib.Path) -> pd.DataFrame:
    """Load all per-stage CSVs in a day folder, concatenating them in order.
 
    Files are sorted by the stage ordering convention:
      stage1_* < loop_* < stage2_*
    Distance values are re-accumulated so the combined DataFrame is
    continuous (each stage picks up where the last one ended).
    """
    day_dir = pathlib.Path(day_dir)
    csv_files = sorted(day_dir.glob("*.csv"), key=_stage_sort_key)
 
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {day_dir}")
 
    frames = []
    dist_offset = 0.0
 
    for csv_path in csv_files:
        df = pd.read_csv(csv_path).fillna(0)
        df["CumulativeDistance"] += dist_offset
        df["_stage_label"] = csv_path.stem   # for transition markers
        dist_offset = float(df["CumulativeDistance"].iloc[-1])
        frames.append(df)
 
    combined = pd.concat(frames, ignore_index=True)
    return combined
 
 
def load_tree(tree_dir: str | pathlib.Path) -> dict:
    """Load an entire variant tree: summary.json + all day CSVs.
 
    Returns {
        "summary": dict (from summary.json),
        "days": {1: DataFrame, 2: DataFrame, ...},
        "variant": str,
    }
    """
    tree_dir = pathlib.Path(tree_dir)
    summary_path = tree_dir / "summary.json"
 
    summary = {}
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
 
    days = {}
    for day_subdir in sorted(tree_dir.glob("day*")):
        if not day_subdir.is_dir():
            continue
        # Extract day number from "day1", "day2", etc.
        day_num = int(day_subdir.name.replace("day", ""))
        try:
            days[day_num] = load_day_dir(day_subdir)
        except FileNotFoundError:
            pass
 
    return {
        "summary": summary,
        "days": days,
        "variant": tree_dir.name,
    }
 
 
def _stage_sort_key(path: pathlib.Path) -> int:
    """Sort CSV files in physical driving order."""
    name = path.stem.lower()
    if name.startswith("stage1") or "stage_1" in name:
        return 1
    if "loop" in name:
        return 2
    if name.startswith("stage2") or "stage_2" in name:
        return 3
    return 4
 
 
def create_race_app(tree_data: dict, car: CarState) -> "dash.Dash":
    """Full-race dashboard with per-day tabs + SOC arc overview.
 
    tree_data: output of load_tree()
    """
    import dash
    from dash import dcc, html
    import plotly.graph_objs as go
 
    summary = tree_data["summary"]
    days = tree_data["days"]
    variant = tree_data["variant"]
 
    app = dash.Dash(__name__, external_stylesheets=_EXTERNAL_STYLESHEETS)
 
    # ── SOC trajectory overview ───────────────────────────────────────────
    soc_traj = summary.get("soc_trajectory_pct", [])
    day_labels = [f"Day {i + 1}" for i in range(len(soc_traj))]
 
    soc_fig = go.Figure()
    soc_fig.add_trace(go.Scatter(
        x=day_labels, y=soc_traj,
        mode="lines+markers+text",
        text=[f"{s:.0f}%" for s in soc_traj],
        textposition="top center",
        name="Start SOC",
        line=dict(width=3),
    ))
    soc_fig.update_layout(
        title=f"SOC Trajectory — {variant}",
        yaxis=dict(title="SOC (%)", range=[0, 105]),
        xaxis=dict(title="Race Day"),
    )
 
    # ── Per-day tabs ──────────────────────────────────────────────────────
    tabs = []
    for day_num in sorted(days.keys()):
        df = days[day_num]
        dist = df["CumulativeDistance"].to_numpy()
        vel = df["Velocity"].to_numpy()
        batt = df["Battery"].to_numpy()
 
        tab_content = html.Div([
            dcc.Graph(figure=dict(
                data=[go.Scatter(x=dist / 1000, y=vel * 3.6,
                                 mode="lines", name="Speed (km/h)")],
                layout=go.Layout(title=f"Day {day_num} Velocity",
                                  xaxis=dict(title="Distance (km)"),
                                  yaxis=dict(title="km/h"))
            )),
            dcc.Graph(figure=dict(
                data=[go.Scatter(x=dist / 1000, y=batt,
                                 mode="lines", name="Battery %")],
                layout=go.Layout(title=f"Day {day_num} Battery",
                                  xaxis=dict(title="Distance (km)"),
                                  yaxis=dict(title="SOC (%)"))
            )),
        ])
 
        tabs.append(dcc.Tab(label=f"Day {day_num}", children=tab_content))
 
    # ── Layout ────────────────────────────────────────────────────────────
    app.layout = html.Div([
        html.H1(f"Race Overview — {variant}",
                style={"text-align": "center",
                       "font-family": '"Roboto Slab", serif'}),
        _config_panel(car),
        html.Div([
            html.H3("Race Summary"),
            html.P(f"Converged: {summary.get('converged')}"),
            html.P(f"Total Distance: {summary.get('total_distance_km', 0):.1f} km"),
            html.P(f"Iterations: {summary.get('iterations')}"),
        ], style={"text-align": "center", "padding": "20px"}),
        dcc.Graph(figure=soc_fig,
                  style={"width": "93%", "margin": "auto"}),
        dcc.Tabs(tabs),
    ], style={"background-color": "#ffffff", "padding": "20px"})
 
    return app
 
 
# ── Replace _main() with this ────────────────────────────────────────────
 
def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Strategy Dashboard — single stage, full day, or full race")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="Single stage CSV (Kevin-style view)")
    group.add_argument("--day", help="Day folder with per-stage CSVs")
    group.add_argument("--tree", help="Variant folder (e.g. data/results/prahlad/)")
    ap.add_argument("--day-index", type=int, default=1,
                    help="1-based race day for time-window overlays (--csv mode)")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-debug", action="store_true")
    args = ap.parse_args()
 
    debug = not args.no_debug
 
    if args.csv:
        # Existing single-file mode
        run_dashboard(csv_path=args.csv, day_index=args.day_index - 1,
                      host=args.host, port=args.port, debug=debug)
 
    elif args.day:
        # Full-day mode: concat per-stage CSVs
        df = load_day_dir(args.day)
        run_dashboard(df=df, day_index=args.day_index - 1,
                      host=args.host, port=args.port, debug=debug)
 
    elif args.tree:
        # Full-race mode: summary + per-day tabs
        tree_data = load_tree(args.tree)
        from configs.car_config import default_car
        car = default_car()
        app = create_race_app(tree_data, car)
        app.run(host=args.host, port=args.port, debug=debug)
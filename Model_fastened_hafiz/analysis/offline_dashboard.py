"""
analysis/offline_dashboard.py — offline strategy dashboard (v2, 23 Aug 2026).

REWRITTEN to consume the optimizer's own JSON output
(``output/strategy_<variant>.json`` produced by
``python -m optimizers.trust_region``) instead of re-integrating a solved
velocity profile. That JSON now carries a per-STAGE breakdown for every day —
stage1 / loop / stage2, each with its own summary stats AND its own plotting
trace (distance / velocity / solar / SOC / slope) — so the dashboard shows the
race the way the strategist thinks about it: one summary per day, then the day
split into its stages (with the loop stage annotated by its rep count, and
absent stages simply skipped).

Two ways to use it:

  build_dashboard_html(json_path, out_html)
      Read a strategy JSON and write ONE self-contained interactive HTML file
      (Plotly inlined — no server, no internet needed to view). This is the
      recommended path: generate the file, open it in any browser, share it.

  run_dashboard(json_path, ...)
      Spin up a live Dash server (optional; needs `dash`). Same content, served
      on localhost with a day dropdown.

CLI:
    # self-contained HTML (default):
    python -m analysis.offline_dashboard --json output/strategy_prahlad.json
    python -m analysis.offline_dashboard --json output/strategy_prahlad.json \
                                         --out prahlad_dashboard.html
    # both variants at once -> two HTML files:
    python -m analysis.offline_dashboard --all output/
    # live Dash server instead:
    python -m analysis.offline_dashboard --json output/strategy_prahlad.json --serve

Dependencies: plotly>=5 (for HTML). dash optional (only for --serve).
The JSON schema this expects is documented in _EXPECTED_SCHEMA below.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import typing as _t

# --- palette (stage colours, brand-neutral, colour-blind-safe-ish) ----------
_STAGE_COLOR = {
    "stage1": "#2563eb",   # blue
    "loop":   "#f59e0b",   # amber
    "stage2": "#10b981",   # green
}
_STAGE_LABEL = {"stage1": "Stage 1", "loop": "Loop(s)", "stage2": "Stage 2"}
_TRAILER_COLOR = "#9ca3af"  # grey — trailered stretches (no power)

_EXPECTED_SCHEMA = """
strategy_<variant>.json
  variant, converged, feasible, iterations
  total_distance_km, total_trailered_km, total_distance_km_dp_estimate
  days: { "1": DAY, ... "8": DAY }

DAY
  route, distance_km, trailered_km, n_loops
  soc_start_pct, soc_end_pct, next_start_soc_pct, morning_charge_pct
  late_penalty_min, inherited_penalty_min
  solar_input_wh, solar_underutil_wh, solar_stored_wh
  drive_time_s, stop_time_s, control_stop_s, loop_stop_s, eta, eta_drive_only
  speed_avg_kmh, speed_min_kmh, speed_max_kmh
  motor_energy_wh, battery_drain_pct
  stages: [ STAGE, ... ]            # ordered as the route runs
  stage1 / loop / stage2: STAGE or null   # direct access, null when absent

STAGE
  stage ('stage1'|'loop'|'stage2'), distance_km, trailered_km, n_loops
  soc_start_pct, soc_end_pct, speed_avg_kmh, speed_min_kmh, speed_max_kmh
  solar_wh, stop_min, elapsed_s, eta
  trace: { distance_km[], velocity_kmh[], solar_w[], soc_pct[], slope_pct[] }
         # distance_km resets to 0 at the stage start
"""

_ORDER = ("stage1", "loop", "stage2")


# ===========================================================================
# 1. Load + small helpers
# ===========================================================================

def load_strategy(json_path: str | pathlib.Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "days" not in data:
        raise ValueError(
            f"{json_path} doesn't look like a strategy JSON (no 'days' key). "
            f"Expected the output of `python -m optimizers.trust_region`.")
    return data


def _num(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _fmt(x, unit="", nd=1) -> str:
    if x is None:
        return "—"
    return f"{_num(x):.{nd}f}{unit}"


def _day_items(data: dict):
    """Yield (day_index0, day_dict) in day order (keys are '1'..'8')."""
    for k in sorted(data.get("days", {}), key=lambda s: int(s)):
        yield int(k) - 1, data["days"][k]


# ===========================================================================
# 2. Per-day figure (per-stage curves on a cumulative-distance axis)
# ===========================================================================

def _stage_ordered(day: dict) -> list:
    """Stages in route order, tolerant of either the list or the keyed form."""
    if isinstance(day.get("stages"), list) and day["stages"]:
        return day["stages"]
    out = []
    for k in _ORDER:
        s = day.get(k)
        if s:
            out.append(s)
    return out


def _day_figure(day_idx0: int, day: dict):
    """A 2x2 Plotly figure (velocity / SOC / solar / slope vs distance) with the
    day's stages drawn as separate coloured segments on a shared cumulative
    distance axis. Missing stages are simply not drawn. Returns a go.Figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    stages = _stage_ordered(day)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Velocity (km/h)", "State of Charge (%)",
                        "Solar input (W)", "Road gradient (%)"),
        vertical_spacing=0.13, horizontal_spacing=0.08)

    # metric -> (trace key, row, col)
    metrics = [
        ("velocity_kmh", 1, 1),
        ("soc_pct",      1, 2),
        ("solar_w",      2, 1),
        ("slope_pct",    2, 2),
    ]

    offset_km = 0.0
    boundaries = []   # cumulative km where each stage starts (for shading)
    for s in stages:
        name = s.get("stage", "stage1")
        color = _STAGE_COLOR.get(name, "#6b7280")
        tr = s.get("trace") or {}
        d = tr.get("distance_km") or []
        if not d:
            # No trace (e.g. a synthetic day) — still advance the offset by the
            # stage's summary distance so later stages line up.
            offset_km += _num(s.get("distance_km"))
            continue
        xd = [offset_km + _num(v) for v in d]
        boundaries.append((offset_km, name, s))
        legend_shown = False
        for key, r, c in metrics:
            y = tr.get(key) or []
            if not y:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xd, y=y, mode="lines",
                    line=dict(color=color, width=2),
                    name=_STAGE_LABEL.get(name, name),
                    legendgroup=name, showlegend=(not legend_shown),
                    hovertemplate=(f"<b>{_STAGE_LABEL.get(name, name)}</b><br>"
                                   "%{x:.1f} km<br>%{y:.1f}<extra></extra>")),
                row=r, col=c)
            legend_shown = True
        offset_km += _num(s.get("distance_km"))

    # Comfort/target reference lines on the velocity plot.
    total_km = offset_km
    for yref, dash, txt in ((70, "dot", "comfort ~70"), (85, "dash", "hard max 85")):
        fig.add_trace(go.Scatter(
            x=[0, total_km], y=[yref, yref], mode="lines",
            line=dict(color="#ef4444", dash=dash, width=1),
            showlegend=False, hoverinfo="skip"), row=1, col=1)

    # Light shading + label per stage across all panels.
    for (start_km, name, s) in boundaries:
        width_km = _num(s.get("distance_km"))
        for r, c in ((1, 1), (1, 2), (2, 1), (2, 2)):
            fig.add_vrect(x0=start_km, x1=start_km + width_km,
                          fillcolor=_STAGE_COLOR.get(name, "#6b7280"),
                          opacity=0.05, line_width=0, row=r, col=c)

    fig.update_xaxes(title_text="Cumulative distance (km)", row=2, col=1)
    fig.update_xaxes(title_text="Cumulative distance (km)", row=2, col=2)
    fig.update_layout(
        height=680, margin=dict(t=60, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.06,
                    xanchor="center", x=0.5),
        template="plotly_white",
        title=dict(text=f"Day {day_idx0 + 1} — {day.get('route', '')}",
                   x=0.5, xanchor="center", font=dict(size=16)))
    return fig


# ===========================================================================
# 3. Per-day summary (the important facts) as an HTML block
# ===========================================================================

def _stat(label: str, value: str, hint: str = "") -> str:
    h = f'<div class="hint">{hint}</div>' if hint else ""
    return (f'<div class="stat"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div>{h}</div>')


def _day_summary_html(day_idx0: int, day: dict) -> str:
    driven = _num(day.get("distance_km"))
    trailered = _num(day.get("trailered_km"))
    eta = day.get("eta", "—")
    eta_drive = day.get("eta_drive_only", "—")
    drive_s = _num(day.get("drive_time_s"))
    stop_s = _num(day.get("stop_time_s"))
    ctrl_m = int(_num(day.get("control_stop_s")) // 60)
    loop_m = int(_num(day.get("loop_stop_s")) // 60)
    late = int(_num(day.get("late_penalty_min")))
    inh = int(_num(day.get("inherited_penalty_min")))
    solar_in = _num(day.get("solar_input_wh"))
    solar_waste = _num(day.get("solar_underutil_wh"))
    solar_stored = _num(day.get("solar_stored_wh"))

    def _hms(s):
        s = int(s); return f"{s // 3600}h{(s % 3600) // 60:02d}m"

    pen_txt = "none"
    if late > 0:
        pen_txt = f'<span class="warn">{late} min → next day</span>'
    inh_txt = f"{inh} min carried in" if inh > 0 else "none"

    stages = _stage_ordered(day)
    stage_pills = ""
    for k in _ORDER:
        s = day.get(k) or next((x for x in stages if x.get("stage") == k), None)
        if not s:
            stage_pills += (f'<span class="pill off">{_STAGE_LABEL[k]}: —</span>')
        else:
            extra = (f" ×{int(_num(s.get('n_loops')))}"
                     if k == "loop" and _num(s.get("n_loops")) else "")
            tr = f" · {_fmt(s.get('trailered_km'),' km trailered')}" if _num(s.get("trailered_km")) else ""
            stage_pills += (
                f'<span class="pill" style="border-color:{_STAGE_COLOR[k]}">'
                f'<b style="color:{_STAGE_COLOR[k]}">{_STAGE_LABEL[k]}{extra}</b> '
                f'{_fmt(s.get("distance_km"), " km")} · '
                f'{_fmt(s.get("speed_avg_kmh"), "")}<span class="u">km/h avg</span> · '
                f'SOC {_fmt(s.get("soc_start_pct"),"%",0)}→{_fmt(s.get("soc_end_pct"),"%",0)} · '
                f'{_fmt(s.get("solar_wh")," Wh",0)}{tr}</span>')

    stats = "".join([
        _stat("Counted distance", _fmt(driven, " km"),
              (f"+ {trailered:.0f} km trailered" if trailered else "no trailering")),
        _stat("Finish (ETA)", eta,
              f"drive {_hms(drive_s)} + stops {_hms(stop_s)} [ctrl {ctrl_m}m + loops {loop_m}m]"),
        _stat("SOC start → end",
              f'{_fmt(day.get("soc_start_pct"),"%",0)} → {_fmt(day.get("soc_end_pct"),"%",0)}',
              f'next day starts ~{_fmt(day.get("next_start_soc_pct"),"%",0)}'),
        _stat("Avg / max speed",
              f'{_fmt(day.get("speed_avg_kmh"),"")}/{_fmt(day.get("speed_max_kmh"),"")} km/h',
              f'min {_fmt(day.get("speed_min_kmh"),"")} km/h'),
        _stat("Loops", str(int(_num(day.get("n_loops")))),
              "5-min stop each"),
        _stat("Solar (in / wasted / stored)",
              f'{solar_in:.0f} / {solar_waste:.0f} / {solar_stored:.0f} Wh',
              "wasted = clipped at SOC ceiling"),
        _stat("Late penalty", pen_txt, f"inherited: {inh_txt}"),
        _stat("Motor energy", _fmt(day.get("motor_energy_wh"), " Wh", 0),
              f'battery {_fmt(day.get("battery_drain_pct"),"%")} '),
    ])

    return (
        f'<div class="daycard" id="day{day_idx0 + 1}">'
        f'<h2>Day {day_idx0 + 1} <span class="route">{day.get("route","")}</span></h2>'
        f'<div class="stats">{stats}</div>'
        f'<div class="pills">{stage_pills}</div>'
        f'<div class="plot" id="plot{day_idx0 + 1}"></div>'
        f'</div>')


# ===========================================================================
# 4. Build the self-contained HTML
# ===========================================================================

_CSS = """
:root{--bg:#f8fafc;--card:#ffffff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;}
*{box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
header{background:#0f172a;color:#fff;padding:18px 26px}
header h1{margin:0;font-size:20px}
header .sub{color:#94a3b8;font-size:13px;margin-top:4px}
.wrap{max-width:1180px;margin:0 auto;padding:18px}
.toolbar{position:sticky;top:0;background:var(--bg);padding:12px 0;z-index:5;border-bottom:1px solid var(--line);margin-bottom:8px}
select{font-size:15px;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:#fff}
.daycard{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.daycard h2{margin:0 0 12px;font-size:18px}
.daycard h2 .route{color:var(--muted);font-weight:400;font-size:14px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
@media(max-width:900px){.stats{grid-template-columns:repeat(2,1fr)}}
.stat{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.stat .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.stat .val{font-size:16px;font-weight:700;margin-top:2px}
.stat .hint{font-size:11px;color:var(--muted);margin-top:2px}
.warn{color:#b45309}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 14px}
.pill{font-size:12px;border:1px solid var(--line);border-left-width:4px;border-radius:8px;padding:6px 10px;background:#fff}
.pill.off{color:#94a3b8;border-left-color:#cbd5e1}
.pill .u{font-size:10px;color:var(--muted);margin-left:2px}
.summary-tot{display:flex;gap:22px;flex-wrap:wrap;font-size:14px}
.summary-tot b{font-size:18px}
"""


def build_dashboard_html(json_path: str | pathlib.Path,
                         out_html: str | pathlib.Path | None = None) -> str:
    """Read a strategy JSON and write a single self-contained HTML dashboard.

    Returns the output path. Plotly.js is inlined so the file opens offline.
    """
    import plotly.io as pio

    data = load_strategy(json_path)
    variant = data.get("variant", "?")
    if out_html is None:
        stem = pathlib.Path(json_path).stem  # strategy_prahlad
        out_html = pathlib.Path(json_path).with_name(f"{stem}_dashboard.html")

    days = list(_day_items(data))

    # Header / totals
    tot = (f'<div class="summary-tot">'
           f'<div>Counted distance <b>{_num(data.get("total_distance_km")):.0f} km</b></div>'
           f'<div>Trailered <b>{_num(data.get("total_trailered_km")):.0f} km</b></div>'
           f'<div>Ground covered <b>{_num(data.get("total_distance_km")) + _num(data.get("total_trailered_km")):.0f} km</b></div>'
           f'<div>Converged <b>{data.get("converged")}</b> · Feasible <b>{data.get("feasible")}</b></div>'
           f'</div>')

    # Day selector
    opts = "".join(f'<option value="{i+1}">Day {i+1} — {d.get("route","")}</option>'
                   for i, d in days)
    toolbar = (f'<div class="toolbar">Show day: '
               f'<select id="daysel" onchange="showDay(this.value)">'
               f'<option value="all">All days</option>{opts}</select></div>')

    # Per-day sections + figures
    body_cards = []
    plot_divs_js = []
    first = True
    for i, d in days:
        body_cards.append(_day_summary_html(i, d))
        fig = _day_figure(i, d)
        # First figure includes plotly.js inline; the rest reuse it.
        html_fig = pio.to_html(
            fig, include_plotlyjs=(True if first else False),
            full_html=False, div_id=f"plotdiv{i+1}", config={"displayModeBar": True})
        plot_divs_js.append((i + 1, html_fig))
        first = False

    # Assemble
    cards_html = "\n".join(body_cards)
    figs_html = "\n".join(h for _, h in plot_divs_js)

    js = """
    function showDay(v){
      document.querySelectorAll('.daycard').forEach(function(c){
        c.style.display = (v==='all' || c.id==='day'+v) ? 'block' : 'none';
      });
      window.dispatchEvent(new Event('resize'));
    }
    // Move each Plotly figure into its day card's .plot slot.
    window.addEventListener('DOMContentLoaded', function(){
      %MOVES%
    });
    """
    moves = "\n".join(
        f"var f{n}=document.getElementById('plotdiv{n}');"
        f"if(f{n}){{document.getElementById('plot{n}').appendChild(f{n});}}"
        for n, _ in plot_divs_js)
    js = js.replace("%MOVES%", moves)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>AgniRath Strategy — {variant}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{_CSS}</style></head><body>
<header><h1>AgniRath — Race Strategy Dashboard <span style="color:#38bdf8">({variant})</span></h1>
<div class="sub">Per-day summary + stage-by-stage curves. Source: {os.path.basename(str(json_path))}</div></header>
<div class="wrap">
{tot}
{toolbar}
<div id="hidden-figs" style="display:none">{figs_html}</div>
{cards_html}
</div>
<script>{js}</script>
</body></html>"""

    pathlib.Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return str(out_html)


# ===========================================================================
# 5. Optional live Dash server
# ===========================================================================

def run_dashboard(json_path: str | pathlib.Path,
                  host: str = "127.0.0.1", port: int = 8050,
                  debug: bool = False) -> None:
    """Serve the same content live via Dash (needs `pip install dash`)."""
    import dash
    from dash import dcc, html, Input, Output
    data = load_strategy(json_path)
    days = list(_day_items(data))
    day_map = {i + 1: d for i, d in days}

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2(f"AgniRath Strategy — {data.get('variant','?')}"),
        dcc.Dropdown(id="day",
                     options=[{"label": f"Day {i+1} — {d.get('route','')}", "value": i + 1}
                              for i, d in days],
                     value=(days[0][0] + 1 if days else 1), clearable=False),
        dcc.Graph(id="fig", style={"height": "700px"}),
        html.Pre(id="facts", style={"whiteSpace": "pre-wrap",
                                    "fontFamily": "monospace"}),
    ], style={"maxWidth": "1180px", "margin": "auto", "padding": "16px"})

    @app.callback(Output("fig", "figure"), Output("facts", "children"),
                  Input("day", "value"))
    def _update(day_no):
        d = day_map.get(int(day_no), {})
        fig = _day_figure(int(day_no) - 1, d)
        facts = (f"Distance {d.get('distance_km')} km | ETA {d.get('eta')} "
                 f"(drive-only {d.get('eta_drive_only')}) | "
                 f"SOC {d.get('soc_start_pct')}→{d.get('soc_end_pct')}% | "
                 f"loops {d.get('n_loops')} | late penalty {d.get('late_penalty_min')} min "
                 f"| inherited {d.get('inherited_penalty_min')} min")
        return fig, facts

    app.run(host=host, port=port, debug=debug)


# ===========================================================================
# CLI
# ===========================================================================

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Offline strategy dashboard — consumes strategy_<variant>.json")
    ap.add_argument("--json", help="path to a strategy_<variant>.json")
    ap.add_argument("--all", metavar="DIR",
                    help="build HTML for every strategy_*.json in DIR")
    ap.add_argument("--out", help="output HTML path (single --json only)")
    ap.add_argument("--serve", action="store_true",
                    help="run a live Dash server instead of writing HTML")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()

    if args.all:
        files = sorted(glob.glob(os.path.join(args.all, "strategy_*.json")))
        files = [f for f in files if "_old" not in os.path.basename(f)]
        if not files:
            raise SystemExit(f"no strategy_*.json found in {args.all}")
        for f in files:
            out = build_dashboard_html(f)
            print(f"wrote {out}")
        return

    if not args.json:
        raise SystemExit("provide --json <strategy.json> or --all <dir>")

    if args.serve:
        run_dashboard(args.json, host=args.host, port=args.port)
    else:
        out = build_dashboard_html(args.json, args.out)
        print(f"wrote {out}")


if __name__ == "__main__":
    _main()
"""
Per-day / per-stage actuals from the Sasol telemetry, for the plan-vs-actual comparison.

Rebuilds the fragments exactly as stitch_days.py does (same clock/overlap fixes) but keeps
the source file name on every record, so each stretch of telemetry is labelled stage1 /
loop / stage2 / charging from the logger file it came from.

Writes  actuals_by_day.csv, actuals_by_phase.csv, stops.csv, timeline_runs.csv  (next to this file)
Units: Vehicle_Velocity is logged in m/s.  Distance = sum(v * dt) over packets that are
<= 10 s apart; silent stretches (logger drop-outs) are NOT integrated, so telemetry
distance is a lower bound of the real distance (compare with the debrief totals).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stitch_days as sd  # noqa: E402

MOVE_KMH = 5.0       # moving above this
STOP_KMH = 1.0       # stationary below this
MIN_STOP_S = 60.0    # stationary stretches shorter than this are not "stops"
MAX_DT_S = 10.0      # packets further apart than this are a telemetry gap
DRIVE_PHASES = ["stage1", "loop", "stage2"]


def phase_of(fname):
    f = fname.lower()
    if "charging" in f:
        return "charging"
    if "beforerunning" in f:
        return "pre-start"
    if "loop" in f:
        return "loop"
    if "stage1" in f:
        return "stage1"
    if "stage2" in f:
        return "stage2"
    return "other"


def load_day(day):
    """DataFrame of MAIN packets with source file + phase, time-sorted (SAST)."""
    frags = []
    for path in sorted((sd.RAW_DIR / f"Day {day}").glob("*.jsonl")):
        recs, _ = sd.read_fragment(path)
        frags.append((path.name, recs))
    main_keys, temp_keys = sd.canonical_schemas(r for _, rs in frags for r in rs)
    for fname, recs in frags:
        recs[:] = [r for r in recs if frozenset(r) in (main_keys, temp_keys)]
        sd.to_sast(recs, fname)
        sd.fix_backward_steps(recs, fname)
    sd.shift_overlaps(frags)
    rows = []
    for fname, recs in frags:
        ph = phase_of(fname)
        for r in recs:
            if sd.MAIN_MARKER not in r:
                continue
            rows.append(dict(
                t=r["__t"], file=fname, phase=ph,
                v_ms=r["Vehicle_Velocity"], vbus=r["Bus_Voltage"], ibus=r["Bus_Current"],
                p_sol_w=sum(r[f"Input_Voltage_{c}"] * r[f"Input_Current_{c}"] for c in "ABCD"),
                **{f"p_{c}": r[f"Input_Voltage_{c}"] * r[f"Input_Current_{c}"] for c in "ABCD"},
            ))
    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    ts = (df.t - df.t.iloc[0]).dt.total_seconds().to_numpy()
    dt = np.diff(ts, prepend=ts[0])
    dt[0] = 0.0
    df["ts"] = ts
    df["dt"] = np.where(dt <= MAX_DT_S, dt, 0.0)       # integrated step
    df["gap_dt"] = np.where(dt > MAX_DT_S, dt, 0.0)    # silent time before this packet
    df["kmh"] = df.v_ms * 3.6
    df["p_bus_w"] = df.vbus * df.ibus
    return df


def stops_in(df):
    """Stationary stretches >= MIN_STOP_S between packets <= MAX_DT_S apart."""
    out, i, n = [], 0, len(df)
    kmh, ts, gap = df.kmh.to_numpy(), df.ts.to_numpy(), df.gap_dt.to_numpy()
    while i < n:
        if kmh[i] < STOP_KMH:
            j = i
            while j + 1 < n and kmh[j + 1] < STOP_KMH and gap[j + 1] == 0:
                j += 1
            dur = ts[j] - ts[i]
            if dur >= MIN_STOP_S:
                out.append((df.t.iloc[i], df.t.iloc[j], dur, df.phase.iloc[i]))
            i = j + 1
        else:
            i += 1
    return out


def timeline_runs(df, step_s=60):
    """Run-length per-minute states (driving / stopped / no data), hours since midnight SAST."""
    t0 = df.t.iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
    sec = (df.t - t0).dt.total_seconds().to_numpy()
    g = pd.DataFrame({"b": (sec // step_s).astype(int), "kmh": df.kmh.to_numpy()}).groupby("b").kmh.mean()
    lo, hi = int(g.index.min()), int(g.index.max())
    runs, cur, start = [], None, lo
    for k in range(lo, hi + 2):
        st = None if k > hi else ("no data" if k not in g.index else ("driving" if g[k] > MOVE_KMH else "stopped"))
        if st != cur:
            if cur is not None:
                runs.append((start * step_s / 3600.0, k * step_s / 3600.0, cur))
            cur, start = st, k
    return runs


def summarise(df):
    """Metrics for any slice of MAIN packets."""
    mv = df.kmh > MOVE_KMH
    cov = df.dt.sum()
    km = (df.v_ms * df.dt).sum() / 1000.0
    w = df.dt[mv]
    avg_mv = float((df.kmh[mv] * w).sum() / w.sum()) if w.sum() else np.nan
    motor_wh = float((df.p_bus_w * df.dt).sum() / 3600.0)
    sol_wh = float((df.p_sol_w * df.dt).sum() / 3600.0)
    return dict(
        first=df.t.iloc[0], last=df.t.iloc[-1], covered_min=cov / 60.0, km_telemetry=km,
        avg_moving_kmh=avg_mv, p95_kmh=float(np.percentile(df.kmh[mv], 95)) if mv.any() else np.nan,
        max_kmh=float(df.kmh.max()), motor_wh=motor_wh, solar_wh=sol_wh,
        motor_wh_per_km=motor_wh / km if km > 0.5 else np.nan,
        avg_solar_w=sol_wh * 3600.0 / cov if cov else np.nan,
        vbus_start=float(df.vbus.iloc[:20].median()), vbus_end=float(df.vbus.iloc[-20:].median()),
    )


def main():
    day_rows, phase_rows, stop_rows, tl_rows = [], [], [], []
    for day in sd.RACE_DAYS:
        df = load_day(day)
        mv = df[(df.kmh > MOVE_KMH) & df.phase.isin(DRIVE_PHASES)]
        s = summarise(df)
        if len(mv):
            span = df[(df.t >= mv.t.iloc[0]) & (df.t <= mv.t.iloc[-1])]
            s["drive_start"], s["drive_end"] = mv.t.iloc[0], mv.t.iloc[-1]
            s["span_min"] = (mv.t.iloc[-1] - mv.t.iloc[0]).total_seconds() / 60.0
            st = stops_in(span)
            s["stopped_min_logged"] = sum(x[2] for x in st) / 60.0
            s["n_stops"] = len(st)
            s["silent_min_in_span"] = span.gap_dt.sum() / 60.0
            for a, b, d, ph in st:
                stop_rows.append(dict(day=day, start=a, end=b, minutes=d / 60.0, phase=ph))
        s["day"] = day
        day_rows.append(s)
        for a, b, st_ in timeline_runs(df):
            tl_rows.append(dict(day=day, start_h=a, end_h=b, state=st_))
        for ph, g in df.groupby("phase", sort=False):
            if len(g) < 5:
                continue
            m = summarise(g)
            m.update(day=day, phase=ph, files=", ".join(sorted(g.file.unique())))
            phase_rows.append(m)
    days = pd.DataFrame(day_rows).set_index("day")
    days.to_csv(HERE / "actuals_by_day.csv")
    pd.DataFrame(phase_rows).to_csv(HERE / "actuals_by_phase.csv", index=False)
    pd.DataFrame(stop_rows).to_csv(HERE / "stops.csv", index=False)
    pd.DataFrame(tl_rows).to_csv(HERE / "timeline_runs.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cols = ["km_telemetry", "avg_moving_kmh", "p95_kmh", "stopped_min_logged", "n_stops",
            "silent_min_in_span", "motor_wh_per_km", "avg_solar_w"]
    print(days[cols].round(1).to_string())


if __name__ == "__main__":
    main()

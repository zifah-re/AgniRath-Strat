"""
Model validation (motor side): drive the model's own physics with the speed the car actually
drove and compare predicted vs measured electrical energy.

For each stage with telemetry:
  1. Rebuild distance along the stage by integrating logged speed from the stage start
     (drop-outs up to 5 min are filled at the neighbouring speed, longer ones add 0 km; for
     stages that reach the finish the result is scaled to the official stage length).
  2. Read the gradient from the route file's existing smoothed profile (route 'Gradient').
  3. core.physics.net_power with car_config values (solar set to 0) gives the predicted
     battery-side power; measured power is Bus_Voltage * Bus_Current.
Compared over logged packets only, moving samples (> 2 m/s); stationary power is reported
separately. Wind is not modelled here (no wind logged).

Writes validation_motor_by_stage.csv, validation_motor_segments.csv, fig_validation_motor.png
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL = REPO / "Model"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODEL))
import actuals  # noqa: E402
from configs.car_config import CarState  # noqa: E402
from core.physics import net_power  # noqa: E402

PROC = MODEL / "data" / "processed"
ASG = MODEL / "data" / "as_given"
MOVING_MS = 2.0
SEG_KM = 10.0
MAX_FILL_S = 300.0   # logger gaps up to 5 min are filled at the neighbouring speed; longer ones add 0 km

# (day, phase) -> route file glob.  Day 3 uses the routes actually issued.
STAGES = {
    (1, "stage1"): PROC / "*Day 1 _10 Sept Stage 1*.save",
    (2, "stage1"): PROC / "*Day 2 Half Blind_11 Sept Stage 1*.save",
    (2, "stage2"): PROC / "*Day 2 Half Blind_11 Sept Stage 2*.save",
    (3, "stage1"): ASG / "Day 3 as-given_Stage 1.kml.save",
    (3, "stage2"): ASG / "Day 3 as-given_Stage 2.kml.save",
    (5, "stage2"): PROC / "*Day 5 _14 Sept Stage 2*.save",
    (6, "stage1"): PROC / "*Day 6 _15 Sept Stage 1*.save",
    (7, "stage1"): PROC / "*Day 7_16 Sept Stage 1*.save",
    (8, "stage2"): PROC / "*Day 8_17 Sept Stage 2*.save",
}
FULL = {(1, "stage1"), (2, "stage1"), (2, "stage2"), (3, "stage1"), (3, "stage2"), (5, "stage2")}


def load_route(pattern):
    f = glob.glob(str(pattern))[0]
    p = json.load(open(f, encoding="utf-8"))["profile"]
    return np.array(p["Distance"], float), np.array(p["Gradient"], float)


def stage_frame(df, phase):
    g = df[df.phase == phase].sort_values("t").reset_index(drop=True)
    # drop leading standstill so distance starts at the start line
    mv = np.flatnonzero(g.v_ms.to_numpy() > 1.0)
    g = g.iloc[mv[0]:].reset_index(drop=True)
    ts = (g.t - g.t.iloc[0]).dt.total_seconds().to_numpy()
    dt = np.diff(ts, prepend=ts[0])
    dt[0] = 0.0
    covered = dt <= actuals.MAX_DT_S
    v = g.v_ms.to_numpy()
    v_prev = np.r_[v[0], v[:-1]]
    short_gap = (~covered) & (dt <= MAX_FILL_S)
    step = np.where(covered, v * dt, np.where(short_gap, 0.5 * (v + v_prev) * dt, 0.0))   # long gaps: unknown, 0 km
    g["x_km"] = np.cumsum(step) / 1000.0
    g["dt_c"] = np.where(covered, dt, 0.0)
    return g


def predict(g, dist_km, grad_pct, car):
    v = g.v_ms.to_numpy()
    dt = g.dt_c.to_numpy()
    nxt = np.r_[v[1:], v[-1]]
    ok = (dt > 0) & (v > MOVING_MS)
    slope = np.interp(g.x_km.to_numpy(), dist_km, grad_pct)
    seg = np.maximum(v * dt / 1000.0, 1e-9)
    net, _ = net_power(car, v, nxt, slope, np.zeros_like(v), seg)
    p_el = -(net + car.p_idle_w)                                      # battery-side electrical draw (W)
    return ok, slope, p_el


def main():
    car = CarState()
    rows, segs = [], []
    days = {}
    for (day, phase), pat in STAGES.items():
        if day not in days:
            days[day] = actuals.load_day(day)
        dist_km, grad = load_route(pat)
        g = stage_frame(days[day], phase)
        L = float(dist_km[-1])
        x_end = float(g.x_km.iloc[-1])
        if (day, phase) in FULL:
            g["x_km"] = g.x_km * (L / x_end)
        ok, slope, p_el = predict(g, dist_km, grad, car)
        dt = g.dt_c.to_numpy()
        meas_p = (g.vbus * g.ibus).to_numpy()
        e_meas = float((meas_p * dt)[ok].sum() / 3600.0)
        e_pred = float((p_el * dt)[ok].sum() / 3600.0)
        km = float((g.v_ms.to_numpy() * dt)[ok].sum() / 1000.0)
        stop = (dt > 0) & (g.v_ms.to_numpy() < 0.5)
        rows.append(dict(day=day, phase=phase, stage_km=round(L, 1), x_end_raw_km=round(x_end, 1),
                         scale=round(L / x_end, 3) if (day, phase) in FULL else np.nan,
                         moving_km=km, avg_kmh=float((g.v_ms.to_numpy() * dt)[ok].sum() / dt[ok].sum() * 3.6),
                         measured_wh=e_meas, predicted_wh=e_pred, ratio_pred_over_meas=e_pred / e_meas,
                         measured_wh_per_km=e_meas / km, predicted_wh_per_km=e_pred / km,
                         error_wh_per_km=(e_pred - e_meas) / km,
                         stationary_w_measured=float(np.average(meas_p[stop], weights=dt[stop])) if stop.any() else np.nan,
                         stationary_w_model=car.p_idle_w))
        # 10 km segments
        sid = (g.x_km.to_numpy() // SEG_KM).astype(int)
        for s in np.unique(sid[ok]):
            m = ok & (sid == s)
            if dt[m].sum() < 120:
                continue
            kms = float((g.v_ms.to_numpy() * dt)[m].sum() / 1000.0)
            segs.append(dict(day=day, phase=phase, seg=int(s), km=kms,
                             meas_wh_km=float((meas_p * dt)[m].sum() / 3600.0 / kms),
                             pred_wh_km=float((p_el * dt)[m].sum() / 3600.0 / kms),
                             mean_slope=float(slope[m].mean()), mean_kmh=float((g.v_ms.to_numpy()[m]).mean() * 3.6)))
    st = pd.DataFrame(rows)
    sg = pd.DataFrame(segs)
    st.to_csv(HERE / "validation_motor_by_stage.csv", index=False)
    sg.to_csv(HERE / "validation_motor_segments.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(st.round(2).to_string(index=False))
    tot_m, tot_p = st.measured_wh.sum(), st.predicted_wh.sum()
    print(f"\nALL STAGES: predicted/measured = {tot_p / tot_m:.3f}   ({tot_p:.0f} vs {tot_m:.0f} Wh)")
    r = np.corrcoef(sg.meas_wh_km, sg.pred_wh_km)[0, 1]
    mae = (sg.pred_wh_km - sg.meas_wh_km).abs().mean()
    print(f"10-km segments (n={len(sg)}): r = {r:.3f}, mean abs error = {mae:.1f} Wh/km, "
          f"mean bias = {(sg.pred_wh_km - sg.meas_wh_km).mean():+.1f} Wh/km")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axs[0]
    lab = [f"D{d} {p[-1] if p.startswith('stage') else p}" for d, p in zip(st.day, st.phase)]
    lab = [f"D{d} S{p[-1]}" for d, p in zip(st.day, st.phase)]
    x = np.arange(len(st))
    ax.bar(x - 0.2, st.measured_wh_per_km, 0.4, color="#c2542d", label="measured")
    ax.bar(x + 0.2, st.predicted_wh_per_km, 0.4, color="#8a8a8a", label="model (actual speed in)")
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("Wh/km at the battery (moving)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Energy per km by stage")
    ax = axs[1]
    for (d, p), c in zip(sg.groupby("day"), plt.cm.tab10(np.arange(10))):
        ax.scatter(p.meas_wh_km, p.pred_wh_km, s=22, color=c, label=f"Day {d}")
    lim = [min(sg.meas_wh_km.min(), sg.pred_wh_km.min()) - 3, max(sg.meas_wh_km.max(), sg.pred_wh_km.max()) + 3]
    ax.plot(lim, lim, color="black", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("measured Wh/km (10 km segments)")
    ax.set_ylabel("model Wh/km")
    ax.set_title(f"Segments: r = {r:.2f}")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "fig_validation_motor.png")


if __name__ == "__main__":
    main()

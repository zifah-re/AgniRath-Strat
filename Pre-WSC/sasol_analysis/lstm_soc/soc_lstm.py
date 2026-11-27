"""
LSTM state-of-charge estimation as a replacement for coulomb counting (Sasol 2026 telemetry).

1. Reference SOC ("ground truth").  The logger's SOC_Ah is stuck and Pack_Current is 0, so a reference is
   built from what IS logged: cell voltages (CMU) and currents (bus + array).  Model per 10 s bin:
       SOC = SOC0_seg + 100 * cumAh / Q                     (coulomb counting inside a continuous segment)
       v_cell = OCV(SOC) + R * i_net                        (Rint equivalent circuit, + i = charging)
   Q (Ah), R (ohm, per series cell referred to pack current) and one SOC0 per continuous log segment are
   fitted by least squares over every bin of every day. OCV(SOC): typical NMC Li-ion cell curve
   (table below). Segments break at logger gaps > 5 min, because current during a gap is not recorded.
2. Estimators compared on each held-out day (leave-one-day-out):
   a) LSTM: input = last WINDOW bins (10 min) of [v_cell, v_min, v_max, i_net, i_bus, t_cell]; output =
      SOC now. No initial SOC, no integration. Implemented in JAX (lax.scan LSTM cell + Adam).
   b) Coulomb counting as a car would run it live: SOC0 from the first loaded cell voltage through the
      OCV curve (no rest), integrate the current, nominal capacity. Plus a realistic sensor case:
      0.5 A offset + 2 % gain error on current.
   c) Voltage lookup: OCV^-1(v_cell - R i_net) with the fitted R (instantaneous, no memory).
   The LSTM is also run with the same corrupted current input, to test robustness.
Writes results/*.csv, fig_*.png in this folder.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA, RES = HERE / "data", HERE / "results"
DT_H = 10.0 / 3600.0
GAP_BREAK_S = 300
WINDOW = 60                       # 60 x 10 s = 10 min of history
FEATS = ["v_cell", "v_ir", "i_net", "i_bus", "t_cell"]   # v_ir = v_cell - R i_net (physics feature)
DAYS = [1, 2, 3, 5, 6, 7, 8]      # Day 4: trailered (no discharge), used for training only
SEED = 0

# typical NMC 18650-class open-circuit-voltage curve (SOC %, V)
OCV_SOC = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], float)
OCV_V = np.array([3.00, 3.30, 3.45, 3.55, 3.62, 3.68, 3.75, 3.84, 3.93, 4.02, 4.10, 4.18])


def ocv(soc):
    return np.interp(soc, OCV_SOC, OCV_V)


def ocv_inv(v):
    return np.interp(v, OCV_V, OCV_SOC)


# ---------------------------------------------------------------- data + reference SOC
def load_days():
    out = {}
    for f in sorted(DATA.glob("day*.csv")):
        d = pd.read_csv(f, parse_dates=["t"]).sort_values("t").reset_index(drop=True)
        d["t_cell"] = d.t_cell.fillna(d.t_cell.median() if d.t_cell.notna().any() else 30.0)
        gap = d.t.diff().dt.total_seconds().fillna(0) > GAP_BREAK_S
        d["seg"] = gap.cumsum()
        d["cumAh"] = d.groupby("seg").i_net.cumsum() * DT_H
        out[int(d.day.iloc[0])] = d
    return out


def fit_reference(days):
    segs = [(dn, s) for dn, d in days.items() for s in sorted(d.seg.unique())]
    idx = {k: i for i, k in enumerate(segs)}

    def resid(p):
        Q, R = p[0], p[1]
        r = []
        for dn, d in days.items():
            s0 = np.array([p[2 + idx[(dn, s)]] for s in d.seg])
            soc = s0 + 100 * d.cumAh.to_numpy() / Q
            r.append(d.v_cell.to_numpy() - ocv(soc) - R * d.i_net.to_numpy())
            r.append(0.02 * np.clip(soc - 100, 0, None) + 0.02 * np.clip(-soc, 0, None))   # keep SOC in 0-100
        return np.concatenate(r)

    p0 = np.r_[40.0, 0.005, [ocv_inv(days[dn].v_cell[days[dn].seg == s].iloc[0]) for dn, s in segs]]
    lo = np.r_[10.0, 0.0, np.zeros(len(segs))]
    hi = np.r_[150.0, 0.05, np.full(len(segs), 100.0)]
    fit = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=0.02)
    Q, R = fit.x[:2]
    for dn, d in days.items():
        s0 = np.array([fit.x[2 + idx[(dn, s)]] for s in d.seg])
        d["soc_ref"] = s0 + 100 * d.cumAh.to_numpy() / Q
        d["v_fit"] = ocv(d.soc_ref) + R * d.i_net
    rms_mv = 1000 * np.sqrt(np.mean(np.concatenate([(d.v_cell - d.v_fit) ** 2 for d in days.values()])))
    return Q, R, rms_mv, len(segs)


# ---------------------------------------------------------------- LSTM (JAX)
def init_params(key, n_in, n_h):
    k1, k2 = jax.random.split(key)
    s = 1.0 / np.sqrt(n_h)
    return dict(W=jax.random.uniform(k1, (n_in + n_h, 4 * n_h), minval=-s, maxval=s),
                b=jnp.zeros(4 * n_h).at[n_h:2 * n_h].set(1.0),          # forget-gate bias 1
                Wo=jax.random.uniform(k2, (n_h, 1), minval=-s, maxval=s), bo=jnp.zeros(1))


def lstm_forward(p, x):
    """x: (batch, time, feat) -> (batch,) SOC/100 at the last step."""
    n_h = p["Wo"].shape[0]
    h = jnp.zeros((x.shape[0], n_h))
    c = jnp.zeros_like(h)

    def step(carry, xt):
        h, c = carry
        z = jnp.concatenate([xt, h], axis=1) @ p["W"] + p["b"]
        i, f, g, o = jnp.split(z, 4, axis=1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return (h, c), None

    (h, c), _ = jax.lax.scan(step, (h, c), jnp.swapaxes(x, 0, 1))
    return (h @ p["Wo"] + p["bo"])[:, 0]


def loss_fn(p, x, y):
    return jnp.mean((lstm_forward(p, x) - y) ** 2)


@jax.jit
def adam_step(p, m, v, t, x, y, lr):
    g = jax.grad(loss_fn)(p, x, y)
    m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
    v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
    mh = jax.tree_util.tree_map(lambda a: a / (1 - 0.9 ** t), m)
    vh = jax.tree_util.tree_map(lambda a: a / (1 - 0.999 ** t), v)
    p = jax.tree_util.tree_map(lambda a, b, c: a - lr * b / (jnp.sqrt(c) + 1e-8), p, mh, vh)
    return p, m, v


predict = jax.jit(lstm_forward)


def windows(d, mu, sd):
    X = ((d[FEATS].to_numpy() - mu) / sd).astype(np.float32)
    n = len(d)
    pad = np.repeat(X[:1], WINDOW - 1, axis=0)          # first minutes: repeat the first sample
    Xp = np.vstack([pad, X])
    idx = np.arange(n)[:, None] + np.arange(WINDOW)[None, :]
    return Xp[idx]


def base_lookup(d):
    return ocv_inv(d.v_ir.to_numpy())


def train(train_days, epochs=40, n_h=32, batch=256, lr=3e-3, hybrid=False):
    D = pd.concat(train_days)
    mu, sd = D[FEATS].mean().to_numpy(), D[FEATS].std().to_numpy() + 1e-6
    X = np.concatenate([windows(d, mu, sd) for d in train_days])
    y = np.concatenate([(d.soc_ref.to_numpy() - (base_lookup(d) if hybrid else 0.0)) / 100.0
                        for d in train_days]).astype(np.float32)
    key = jax.random.PRNGKey(SEED)
    p = init_params(key, len(FEATS), n_h)
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    rng = np.random.default_rng(SEED)
    t = 0
    for ep in range(epochs):
        order = rng.permutation(len(X))
        lr_ep = lr * (0.5 if ep >= epochs * 0.6 else 1.0) * (0.5 if ep >= epochs * 0.85 else 1.0)
        for k in range(0, len(X) - batch + 1, batch):
            b = order[k:k + batch]
            t += 1
            p, m, v = adam_step(p, m, v, t, jnp.asarray(X[b]), jnp.asarray(y[b]), lr_ep)
    return p, mu, sd


def run_lstm(p, mu, sd, d, hybrid=False):
    X = windows(d, mu, sd)
    out = [np.asarray(predict(p, jnp.asarray(X[k:k + 2048]))) for k in range(0, len(X), 2048)]
    return 100.0 * np.concatenate(out) + (base_lookup(d) if hybrid else 0.0)


# ---------------------------------------------------------------- baselines
def coulomb_live(d, Q_nom, offset=0.0, gain=1.0):
    """what the car could run live: start from the first loaded voltage, integrate measured current."""
    i_meas = gain * d.i_net.to_numpy() + offset
    soc = np.empty(len(d))
    for s in d.seg.unique():
        m = (d.seg == s).to_numpy()
        s0 = ocv_inv(d.v_cell.to_numpy()[m][0])
        soc[m] = s0 + 100 * np.cumsum(i_meas[m]) * DT_H / Q_nom
    return soc


def corrupt(d, offset=0.5, gain=1.02):
    e = d.copy()
    e["i_net"] = gain * e.i_net + offset
    e["i_bus"] = gain * e.i_bus - offset
    return e


def metrics(y, yhat):
    e = yhat - y
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), max_abs=float(np.max(np.abs(e))))


def main():
    RES.mkdir(exist_ok=True)
    days = load_days()
    Q, R, rms_mv, nseg = fit_reference(days)
    for d in days.values():
        d["v_ir"] = d.v_cell - R * d.i_net
    print(f"reference fit: Q = {Q:.1f} Ah, R = {1000 * R:.2f} mOhm (per series cell, pack current), "
          f"voltage RMS {rms_mv:.1f} mV over {nseg} segments")
    pd.DataFrame([dict(Q_ah=Q, R_ohm=R, v_rms_mv=rms_mv, segments=nseg)]).to_csv(RES / "reference_fit.csv", index=False)

    rows, traces = [], {}
    t0 = time.time()
    for test in DAYS:
        tr = [days[k] for k in days if k != test]
        p, mu, sd = train(tr)
        ph, muh, sdh = train(tr, hybrid=True)
        d = days[test]
        y = d.soc_ref.to_numpy()
        dc = corrupt(d)
        dc["v_ir"] = dc.v_cell - R * dc.i_net
        est = {"Hybrid: voltage lookup + LSTM correction": run_lstm(ph, muh, sdh, d, hybrid=True),
               "Hybrid, corrupted current": run_lstm(ph, muh, sdh, dc, hybrid=True),
               "LSTM": run_lstm(p, mu, sd, d),
               "LSTM, corrupted current": run_lstm(p, mu, sd, dc),
               "Coulomb counting (live start)": coulomb_live(d, Q),
               "Coulomb counting, corrupted current": coulomb_live(d, Q, offset=0.5, gain=1.02),
               "Voltage lookup (OCV + IR)": ocv_inv(d.v_cell.to_numpy() - R * d.i_net.to_numpy())}
        for name, yh in est.items():
            rows.append(dict(test_day=test, method=name, hours=len(d) / 360, **metrics(y, yh)))
        traces[test] = (d.t, y, est)
        print(f"Day {test} done ({time.time() - t0:.0f} s): " + ", ".join(
            f"{r['method']} {r['rmse']:.1f}" for r in rows if r["test_day"] == test))
    M = pd.DataFrame(rows)
    M.round(3).to_csv(RES / "leave_one_day_out.csv", index=False)
    S = M.groupby("method")[["rmse", "mae", "max_abs"]].mean().sort_values("rmse")
    S.round(2).to_csv(RES / "summary.csv")
    print("\nmean over held-out days (SOC percentage points):")
    print(S.round(2).to_string())

    # figures
    fig, axs = plt.subplots(len(DAYS), 1, figsize=(10, 2.3 * len(DAYS)), sharey=True)
    for ax, test in zip(axs, DAYS):
        t, y, est = traces[test]
        tl = pd.to_datetime(t)
        ax.plot(tl, y, "k", lw=1.6, label="reference SOC")
        ax.plot(tl, est["LSTM"], color="#3b6ea0", lw=1.1, label="LSTM (held-out day)")
        ax.plot(tl, est["Coulomb counting, corrupted current"], color="#c2542d", lw=1, ls="--",
                label="coulomb counting, 0.5 A / 2 % sensor error")
        ax.plot(tl, est["Voltage lookup (OCV + IR)"], color="#999", lw=.7, alpha=.8, label="voltage lookup")
        ax.set_ylabel(f"Day {test}\nSOC %", fontsize=8)
        ax.set_ylim(-5, 105)
    axs[0].legend(fontsize=7, ncol=4, loc="lower left")
    plt.tight_layout(); plt.savefig(HERE / "fig_soc_traces.png", dpi=150)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    order = list(S.index)
    piv = M.pivot(index="test_day", columns="method", values="rmse")[order]
    x = np.arange(len(DAYS)); w = 0.8 / len(order)
    cols = {"Hybrid: voltage lookup + LSTM correction": "#1f4e8c", "Hybrid, corrupted current": "#7a9cc6", "LSTM": "#6f8fb3", "LSTM, corrupted current": "#9bb7d4", "Coulomb counting (live start)": "#e3a07f",
            "Coulomb counting, corrupted current": "#c2542d", "Voltage lookup (OCV + IR)": "#999"}
    for i, mth in enumerate(order):
        ax.bar(x + (i - len(order) / 2 + .5) * w, piv[mth], w, label=mth, color=cols[mth])
    ax.set_xticks(x); ax.set_xticklabels([f"Day {k}" for k in DAYS])
    ax.set_ylabel("SOC RMSE (percentage points)"); ax.legend(fontsize=7)
    ax.set_title("Held-out day error, leave-one-day-out", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "fig_soc_rmse.png", dpi=170)


def drift_analysis():
    """Where coulomb counting fails on the car: start error + error that grows with sensor offset.
    Start error is measured (first loaded voltage of every log segment through the OCV curve, vs the
    reference); drift is offset x time / Q. The LSTM's error does not grow with time (it re-estimates
    from the last 10 min every step), so its held-out RMSE is drawn flat."""
    days = load_days()
    Q, R, _, _ = fit_reference(days)
    start_err = []
    for d in days.values():
        for s in d.seg.unique():
            m = d[d.seg == s]
            start_err.append(abs(ocv_inv(m.v_cell.iloc[0]) - m.soc_ref.iloc[0]))
    start_err = float(np.mean(start_err))
    S = pd.read_csv(RES / "summary.csv", index_col=0)
    hours = np.linspace(0, 9, 91)
    rows = []
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for off, col in ((0.2, "#e3a07f"), (0.5, "#c2542d"), (1.0, "#8a2f12")):
        e = start_err + 100 * off * hours / Q
        ax.plot(hours, e, color=col, label=f"coulomb counting, {off} A sensor offset")
        rows += [dict(offset_a=off, hours=h, cc_error_pts=v) for h, v in zip(hours, e)]
    for name, col, ls in (("LSTM", "#3b6ea0", "-"), ("Voltage lookup (OCV + IR)", "#999", ":")):
        ax.axhline(S.loc[name, "mae"], color=col, ls=ls, label=f"{name} (held-out mean abs error)")
    ax.set_xlabel("hours since the SOC was last known"); ax.set_ylabel("SOC error (percentage points)")
    ax.set_title(f"Coulomb counting drifts; voltage-based estimators do not (Q = {Q:.1f} Ah, start error {start_err:.1f} pts)", fontsize=8)
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(HERE / "fig_soc_drift.png", dpi=170)
    pd.DataFrame(rows).round(2).to_csv(RES / "coulomb_drift.csv", index=False)
    print(f"coulomb counting start error (loaded voltage -> OCV): {start_err:.1f} pts; "
          f"drift {100 * 0.5 / Q:.2f} pts/h per 0.5 A offset -> {start_err + 100 * 0.5 * 9 / Q:.1f} pts after a 9 h day")


if __name__ == "__main__":
    main()
    drift_analysis()

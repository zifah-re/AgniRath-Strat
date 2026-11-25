"""Solar side: measured MPPT array power vs Solcast actual irradiance, with position-matched weather.
Model/ untouched. Telemetry has no GPS (lat/lon logged as 0), so each packet's route position is
distance-travelled / stage distance, mapped onto the stage file's weather nodes (~9 km apart, in route order)
and interpolated in space and time.  Array flat  =>  POA = DNI*cos(zenith) + DHI = GHI.
Tilted at the sun (stationary stops) => POA = DNI + DHI (+ no ground term, array faces the sky).
eta = P_array / (area * POA).  Writes solar_check_5min.csv, solar_check_by_day.csv, solar_tilted_peaks.csv,
fig_solar_check.png (scatter) and fig_solar_eff_by_day.png (eta vs time, +-1 std, every day).
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import pvlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import actuals as ac
from shading_check import SHARE   # string area shares from the tilted stops
MODEL = HERE.parents[2]/"Model"/"data"
AREA = 5.78
GHI_MIN = 300.0
FILES = {}      # day -> {phase: [files]}
for d in range(1, 9):
    base = sorted(glob.glob(str(MODEL/"Solar_real"/f"*Day {d}[ _]*.jsonl")))
    if d == 3:
        base = sorted(glob.glob(str(MODEL/"as_given"/"mean_Day 3 as-given_*.jsonl")))
    FILES[d] = {"stage1": [f for f in base if "Stage 1" in f or "Stage1" in f],
                "stage2": [f for f in base if "Stage 2" in f or "Stage2" in f],
                "loop": [f for f in base if "Loop" in f]}

def load_nodes(path):
    nodes = json.load(open(path))
    t = pd.to_datetime([r["period_end"] for r in nodes[0]["data"]]) - pd.Timedelta("5min")   # interval start
    ghi = np.array([[r["ghi"] for r in n["data"]] for n in nodes]); dni = np.array([[r["dni"] for r in n["data"]] for n in nodes])
    return t, ghi, dni, np.mean([n["lat"] for n in nodes]), np.mean([n["lon"] for n in nodes])

def attach_weather(df):
    """add ghi, dni, lat0, lon0 columns using position-matched interpolation."""
    df = df.copy(); df["ghi"] = np.nan; df["dni"] = np.nan; df["lat0"] = np.nan; df["lon0"] = np.nan
    day = df.attrs["day"]
    for ph in ("stage1", "stage2", "loop"):
        sel = (df.phase == ph).to_numpy()
        if not sel.any() or not FILES[day][ph]: continue
        t, G, D, la, lo = load_nodes(FILES[day][ph][0])
        x = np.cumsum((df.v_ms * df.dt).to_numpy()[sel]); frac = x / max(x[-1], 1e-9)
        pos = frac * (G.shape[0] - 1)
        tt = (df.t[sel].dt.tz_convert("UTC").dt.tz_localize(None) - t[0].tz_localize(None) if t[0].tzinfo else df.t[sel].dt.tz_convert("UTC").dt.tz_localize(None) - t[0]).dt.total_seconds().to_numpy() / 300.0
        tt = np.clip(tt, 0, len(t) - 1)
        def interp2(M):
            i0 = np.clip(np.floor(pos).astype(int), 0, M.shape[0] - 2) if M.shape[0] > 1 else np.zeros(len(pos), int)
            j0 = np.clip(np.floor(tt).astype(int), 0, M.shape[1] - 2); fi = pos - i0; fj = tt - j0
            if M.shape[0] == 1:
                return M[0, j0] * (1 - fj) + M[0, j0 + 1] * fj
            a = M[i0, j0] * (1 - fj) + M[i0, j0 + 1] * fj; b = M[i0 + 1, j0] * (1 - fj) + M[i0 + 1, j0 + 1] * fj
            return a * (1 - fi) + b * fi
        df.loc[sel, "ghi"] = interp2(G); df.loc[sel, "dni"] = interp2(D)
        df.loc[sel, "lat0"] = la; df.loc[sel, "lon0"] = lo
    return df

def main():
    bins = []
    for day in range(1, 9):
        df = ac.load_day(day); df.attrs["day"] = day
        df = df[df.phase.isin(["stage1", "stage2", "loop"])]
        df = attach_weather(df).dropna(subset=["ghi"])
        if df.empty: continue
        sp = pvlib.solarposition.get_solarposition(df.t.dt.tz_convert("UTC"), df.lat0.to_numpy().mean(), df.lon0.to_numpy().mean())
        cz = np.cos(np.radians(sp.apparent_zenith.to_numpy()))
        dhi = np.clip(df.ghi.to_numpy() - df.dni.to_numpy() * cz, 0, None)
        df["poa_flat"] = df.dni.to_numpy() * np.clip(cz, 0, None) + dhi          # == GHI for a horizontal plane
        df["poa_sun"] = df.dni.to_numpy() + dhi                                   # plane facing the sun
        df["cz"] = cz
        b = df.set_index(df.t.dt.tz_convert("UTC")).resample("5min").agg(
            {"p_sol_w": "mean", "kmh": "mean", "poa_flat": "mean", "poa_sun": "mean", "ghi": "mean", "cz": "mean",
             "p_A": "mean", "p_B": "mean", "p_C": "mean", "p_D": "mean"}).dropna()
        b["day"] = day
        bins.append(b)
    B = pd.concat(bins); B = B[B.ghi > GHI_MIN].copy()
    B["moving"] = B.kmh > 5; B["still"] = B.kmh < 1
    B["eta_flat"] = B.p_sol_w / (AREA * B.poa_flat)
    B["eta_sun"] = B.p_sol_w / (AREA * B.poa_sun)
    for c in "ABCD":
        B[f"eta_{c}"] = B[f"p_{c}"] / (SHARE[c] * AREA * B.poa_flat)
    B["eta_best_string"] = B[[f"eta_{c}" for c in "ABCD"]].max(axis=1)
    B.round(4).to_csv(HERE/"solar_check_5min.csv")
    rows = []
    for day, g in B.groupby("day"):
        mv = g[g.moving]; st = g[g.still & (g.p_sol_w > 1100)]
        rows.append(dict(day=day, bins=len(g), eta_flat_moving_mean=mv.eta_flat.mean(), eta_flat_moving_std=mv.eta_flat.std(),
                         eta_best_string_moving_mean=mv.eta_best_string.mean(), eta_best_string_moving_std=mv.eta_best_string.std(),
                         **{f"eta_{c}_moving_mean": mv[f"eta_{c}"].mean() for c in "ABCD"},
                         eta_flat_moving_median=mv.eta_flat.median(), eta_tilted_mean=st.eta_sun.mean(), eta_tilted_std=st.eta_sun.std(),
                         n_tilted=len(st), peak_array_w=g.p_sol_w.max()))
    S = pd.DataFrame(rows).round(3); S.to_csv(HERE/"solar_check_by_day.csv", index=False)
    B[B.still & (B.p_sol_w > 1100)][["day", "p_sol_w", "poa_sun", "eta_sun"]].round(3).to_csv(HERE/"solar_tilted_peaks.csv")
    print(S.to_string(index=False))
    T = B[B.still & (B.p_sol_w > 1100)]
    print("\ntilted-at-sun eta: mean %.3f std %.3f n=%d" % (T.eta_sun.mean(), T.eta_sun.std(), len(T)))
    M = B[B.moving]
    print("driving flat eta: mean %.3f std %.3f median %.3f n=%d" % (M.eta_flat.mean(), M.eta_flat.std(), M.eta_flat.median(), len(M)))
    C = M[M.day.isin([1, 2, 3, 8])]
    print("clear days 1,2,3,8 driving: whole array %.3f +- %.3f   best string %.3f +- %.3f  (n=%d)" % (
        C.eta_flat.mean(), C.eta_flat.std(), C.eta_best_string.mean(), C.eta_best_string.std(), len(C)))
    # --- figures
    days = sorted(B.day.unique()); fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for ax, day in zip(axs.ravel(), days):
        g = B[B.day == day]; tl = g.index.tz_convert("Africa/Johannesburg"); h = tl.hour + tl.minute / 60
        mv = g.moving.to_numpy(); st = g.still.to_numpy()
        ax.scatter(h[mv], g.eta_best_string[mv], s=10, c="#9bb7d4", label="driving, best string")
        ax.scatter(h[mv], g.eta_flat[mv], s=12, c="#3b6ea0", label="driving, whole array")
        ax.scatter(h[st], g.eta_sun[st], s=12, c="#c2542d", label="stopped (tilted at sun)")
        if mv.sum() > 1:
            m, s = g.eta_flat[mv].mean(), g.eta_flat[mv].std()
            ax.axhspan(m - s, m + s, color="#3b6ea0", alpha=.15); ax.axhline(m, color="#3b6ea0", lw=1)
        ax.axhline(.245, color="k", ls="--", lw=.8)
        ax.set_title(f"Day {day}", fontsize=9); ax.set_ylim(0, .4); ax.set_xlabel("local hour", fontsize=8)
    axs[0, 0].set_ylabel("array efficiency"); axs[1, 0].set_ylabel("array efficiency"); axs[0, 0].legend(fontsize=6)
    for ax in axs.ravel()[len(days):]: ax.axis("off")
    fig.suptitle("Array efficiency = P_array / (5.78 m2 x POA); band = mean +- 1 std of whole-array driving 5-min bins; light dots = best single string (unshaded); dashed = 24.5 % model", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE/"fig_solar_eff_by_day.png", dpi=180)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(AREA*M.poa_flat, M.p_sol_w, s=9, c="#3b6ea0", label="driving"); ax.scatter(AREA*T.poa_sun, T.p_sol_w, s=14, c="#c2542d", label="tilted at sun")
    xx = np.array([0, 6500]); ax.plot(xx, .245*xx, "k--", lw=1, label="24.5 %"); ax.plot(xx, M.eta_flat.mean()*xx, "k:", lw=1, label=f"{M.eta_flat.mean():.1%} (driving mean)")
    ax.set_xlabel("area x POA (W)"); ax.set_ylabel("measured array power (W)"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(HERE/"fig_solar_check.png", dpi=200)

if __name__ == "__main__":
    main()

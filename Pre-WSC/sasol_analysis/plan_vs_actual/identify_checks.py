"""
How far can the identified parameters (identify.py) be trusted?

1. Robustness: re-run the identification with different settings (block length, cruising-speed cut,
   which days are fitted, alignment penalty, no alignment) and see how much each parameter moves.
2. Independent check on coasting data (throttle 0, brake off, bus power ~0 W). The fit never uses these
   blocks. While coasting, the only forces are the road load and gravity, so
   -(m a + m g sin th) must equal m g Crr + 1/2 rho CdA v^2. Compare that with car_config and the
   identified values, per speed bin.
3. Bootstrap correlation between Crr and CdA (do the two trade off against each other?).

Writes identify_checks_robustness.csv, identify_checks_coast.csv, fig_identify_checks.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import align as al          # noqa: E402
import identify as idf      # noqa: E402

M, G, RHO, CAR = idf.M, idf.G, idf.RHO, idf.CAR
V_REF = 62.5 / 3.6


def p0_of(S, days):
    st = S[(S.v < 0.3) & (S.dt > 0) & S.day.isin(days)]
    return float(np.average(st.p, weights=st.dt))


def flat_w(p, v=V_REF):
    return (M * G * p["crr"] * v + 0.5 * RHO * p["cda"] * v ** 3) / p["eta"] + p["p0"]


def heldout_ratio(S, p, fit_days):
    ok = S.ok & ~S.day.isin(fit_days)
    g = S[ok]
    pred = idf.predict_power(g, p, mode_aware=True)
    return float((pred * g.dt).sum() / (g.p * g.dt).sum())


def run(S, label, days=idf.FIT_DAYS, block=20.0, vfit=55.0):
    idf.BLOCK_S, idf.V_FIT = block, vfit / 3.6
    p = idf.identify(S, days, p0_of(S, days))
    idf.BLOCK_S, idf.V_FIT = 20.0, 55 / 3.6
    held = heldout_ratio(S, p, days) if set(days) != {1, 2, 3, 5, 6, 7, 8} else np.nan
    return dict(variant=label, eta=p["eta"], crr=p["crr"], cda=p["cda"], eta_regen=p["eta_regen"],
                flat_w_62kmh=flat_w(p), heldout_energy_ratio=held)


def main():
    S = idf.build(aligned=True)
    rows = [run(S, "baseline (Days 1-3, 20 s blocks, >=55 km/h, pen 0.05)"),
            run(S, "10 s blocks", block=10.0), run(S, "40 s blocks", block=40.0),
            run(S, "road-load fit >= 50 km/h", vfit=50.0), run(S, "road-load fit >= 60 km/h", vfit=60.0),
            run(S, "fit Days 2-3 only", days={2, 3}), run(S, "fit Days 1-2 only", days={1, 2}),
            run(S, "fit Days 1,3 only", days={1, 3}), run(S, "fit all days (no hold-out)", days={1, 2, 3, 5, 6, 7, 8})]
    for pen in (0.02, 0.10):
        al.JUMP_PEN = pen
        rows.append(run(idf.build(aligned=True), f"alignment penalty {pen}"))
    al.JUMP_PEN = 0.05
    rows.append(run(idf.build(aligned=False), "no route alignment"))
    R = pd.DataFrame(rows)
    cfg = dict(variant="car_config (track testing)", eta=CAR.motor_eff, crr=CAR.crr, cda=CAR.cda_m2,
               eta_regen=CAR.regen_eff, flat_w_62kmh=flat_w(dict(crr=CAR.crr, cda=CAR.cda_m2, eta=CAR.motor_eff, p0=CAR.p_idle_w)),
               heldout_energy_ratio=heldout_ratio(S, dict(crr=CAR.crr, cda=CAR.cda_m2, eta=CAR.motor_eff, p0=CAR.p_idle_w,
                                                          eta_regen=CAR.regen_eff), idf.FIT_DAYS))
    R = pd.concat([pd.DataFrame([cfg]), R], ignore_index=True)
    R.round(4).to_csv(HERE / "identify_checks_robustness.csv", index=False)
    print(R.round(4).to_string(index=False))

    # bootstrap Crr-CdA correlation
    p0 = p0_of(S, idf.FIT_DAYS)
    bs = pd.DataFrame([idf.identify(S, idf.FIT_DAYS, p0, rng=np.random.RandomState(i)) for i in range(200)])
    bs["flat"] = [flat_w(dict(crr=a, cda=b, eta=c, p0=p0)) for a, b, c in zip(bs.crr, bs.cda, bs.eta)]
    print(f"\nbootstrap corr(Crr, CdA) = {np.corrcoef(bs.crr, bs.cda)[0, 1]:.2f};  flat-road power at 62.5 km/h "
          f"{bs.flat.mean():.0f} +- {bs.flat.std():.0f} W ({100 * bs.flat.std() / bs.flat.mean():.1f} %)")

    # independent coast check (all days, coast blocks never used by the fit)
    base = R.iloc[1]
    pid = dict(crr=base.crr, cda=base.cda)
    Bc = idf.blocks(S, S.ok & (S["mode"] == "coast") & (S.p.abs() < 20))
    y = -(M * Bc.a + M * G * Bc.s)
    Bc = Bc.assign(F_obs=y, F_cfg=M * G * CAR.crr * Bc.c + 0.5 * RHO * CAR.cda_m2 * Bc.v2,
                   F_id=M * G * pid["crr"] * Bc.c + 0.5 * RHO * pid["cda"] * Bc.v2,
                   kmh=Bc.v * 3.6)
    Bc["bin"] = pd.cut(Bc.kmh, [20, 40, 50, 60, 70, 90])
    C = Bc.groupby("bin", observed=True).agg(n=("F_obs", "size"), F_obs_mean=("F_obs", "mean"),
                                             F_obs_se=("F_obs", lambda s: s.std() / np.sqrt(len(s))),
                                             F_car_config=("F_cfg", "mean"), F_identified=("F_id", "mean"))
    C.round(1).to_csv(HERE / "identify_checks_coast.csv")
    print("\ncoasting road load (N), blocks not used in the fit:")
    print(C.round(1).to_string())
    print(f"all coast blocks: observed {Bc.F_obs.mean():.1f} +- {Bc.F_obs.std() / np.sqrt(len(Bc)):.1f} N | "
          f"car_config {Bc.F_cfg.mean():.1f} N | identified {Bc.F_id.mean():.1f} N  (n={len(Bc)})")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(bs.crr, bs.cda, s=8, alpha=.5, label="bootstrap fits")
    ax[0].scatter([CAR.crr], [CAR.cda_m2], c="k", marker="x", s=60, label="car_config")
    ax[0].set_xlabel("Crr"); ax[0].set_ylabel("CdA (m2)"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Crr vs CdA trade-off (corr {np.corrcoef(bs.crr, bs.cda)[0, 1]:.2f})", fontsize=9)
    x = np.arange(len(C))
    ax[1].errorbar(x, C.F_obs_mean, 1.96 * C.F_obs_se, fmt="o", color="k", capsize=3, label="observed while coasting (95 % CI)")
    ax[1].plot(x, C.F_car_config, "s--", color="#888", label="car_config")
    ax[1].plot(x, C.F_identified, "d-", color="#3b6ea0", label="identified")
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(b) for b in C.index]); ax[1].set_xlabel("speed bin (km/h)")
    ax[1].set_ylabel("road-load force (N)"); ax[1].legend(fontsize=8)
    ax[1].set_title("Independent check: coasting blocks (not used in the fit)", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "fig_identify_checks.png", dpi=180)


if __name__ == "__main__":
    main()

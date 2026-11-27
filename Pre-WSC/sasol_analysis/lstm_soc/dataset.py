"""
Build the per-day battery time series for SOC estimation from the cleaned Sasol logs.

Signals (10 s bins):
  v_cell   mean of the valid cell voltages (CMU packets; 2.5-4.35 V kept, corrupt channels dropped) [V]
  v_min / v_max / t_cell   min / max cell voltage, mean CMU temperature
  n_mod    modules connected (CMUs with >= 4 valid cells); modules are in series
  i_bus    motor-controller bus current [A] (+ = discharge)
  i_sol    array charge current = sum MPPT power / bus voltage [A] (MPPT output power = input power in the logs)
  i_net    i_sol - i_bus [A]  (+ = charging). Auxiliary electronics are not logged, so not included.
Writes series.parquet-free CSV: data/day{N}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "plan_vs_actual"))
import stitch_days as sd          # noqa: E402

BIN = "10s"
VMIN, VMAX = 2.5, 4.35
OUT = HERE / "data"


def load(day):
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
    main, cmu = [], []
    for fname, recs in frags:
        for r in recs:
            if sd.MAIN_MARKER in r:
                vb = r["Bus_Voltage"]
                main.append(dict(t=r["__t"], vbus=vb, i_bus=r["Bus_Current"],
                                 p_sol=sum(r[f"Input_Voltage_{c}"] * r[f"Input_Current_{c}"] for c in "ABCD"),
                                 kmh=r["Vehicle_Velocity"] * 3.6))
            elif "CMU1_Cell0_Voltage" in r:
                cells, temps, nmod = [], [], 0
                for m in range(1, 6):
                    v = [r.get(f"CMU{m}_Cell{i}_Voltage", 0) / 1000.0 for i in range(8)]
                    ok = [x for x in v if VMIN <= x <= VMAX]
                    if len(ok) >= 4:
                        nmod += 1
                        cells += ok
                        tt = r.get(f"CMU{m}_Temp", 0)
                        if 5 <= tt <= 80:
                            temps.append(tt)
                if cells:
                    cmu.append(dict(t=r["__t"], v_cell=np.mean(cells), v_min=np.min(cells), v_max=np.max(cells),
                                    t_cell=np.mean(temps) if temps else np.nan, n_mod=nmod))
    M = pd.DataFrame(main).set_index("t").sort_index()
    C = pd.DataFrame(cmu).set_index("t").sort_index()
    M["i_sol"] = M.p_sol / M.vbus.where(M.vbus > 30)
    m = M[["vbus", "i_bus", "i_sol", "kmh"]].resample(BIN).mean()
    c = C.resample(BIN).mean()
    df = m.join(c, how="outer")
    # gaps: keep bins with main data; carry cell readings at most 60 s (CMU packets are ~8x sparser)
    for k in ("v_cell", "v_min", "v_max", "t_cell", "n_mod"):
        df[k] = df[k].ffill(limit=6)
    df = df.dropna(subset=["i_bus", "v_cell"])
    df["i_net"] = df.i_sol.fillna(0) - df.i_bus
    df["day"] = day
    return df


def main():
    OUT.mkdir(exist_ok=True)
    for day in range(1, 9):
        try:
            df = load(day)
        except Exception as e:      # noqa: BLE001
            print(f"Day {day}: skipped ({e})")
            continue
        df.to_csv(OUT / f"day{day}.csv")
        h = len(df) * 10 / 3600
        print(f"Day {day}: {len(df)} bins ({h:.1f} h)  v_cell {df.v_cell.min():.3f}-{df.v_cell.max():.3f} V  "
              f"modules {df.n_mod.round().mode().iat[0]:.0f}  net Ah {df.i_net.sum() * 10 / 3600:+.1f}  "
              f"discharge Ah {df.i_bus.clip(lower=0).sum() * 10 / 3600:.1f}  charge Ah {df.i_sol.sum() * 10 / 3600:.1f}")


if __name__ == "__main__":
    main()

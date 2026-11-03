"""
run_demo.py - live-demo entry point (runs fine on a laptop, ~20-30 s).

1. Loads a SAVED strategy (default: output/strategy_aryaman_idk.json, the plan
   deployed at Sasol) and prints the per-day plan. No solving.
2. Re-solves ONE day from a given state with the single-day optimiser
   (the same code path as `trust_region resolve-intraday`), single process.

    python run_demo.py                          # plan table + re-solve Day 8
    python run_demo.py --day 4 --soc 60 --time 09:30
    python run_demo.py --plan-only              # just the saved plan
    python run_demo.py --file output/strategy_aryaman.json --day 2 --soc 85 --time 08:00

Nothing here uses multiprocessing. The re-solve is saved to output/ like any
other intraday run (strategy_<variant>_intraday_day<N>_*.json).
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def show_plan(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    print(f"\nSaved plan: {os.path.relpath(path, HERE)}")
    print(f"  variant {d.get('variant')} | {d.get('timestamp')} | feasible {d.get('feasible')} "
          f"| total {d.get('total_distance_km')} km | trailered {d.get('total_trailered_km')} km\n")
    print(f"  {'Day':<4}{'km':>8}{'loops':>7}{'ETA':>8}{'SOC start':>11}{'SOC end':>9}{'avg km/h':>10}")
    for k in sorted(d["days"], key=int):
        r = d["days"][k]
        print(f"  {k:<4}{r['distance_km']:>8.1f}{r.get('n_loops', 0):>7}{r.get('eta', '-'):>8}"
              f"{r['soc_start_pct']:>11.1f}{r['soc_end_pct']:>9.1f}{r.get('speed_avg_kmh', 0):>10.1f}")
    print()
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=os.path.join(HERE, "output", "strategy_aryaman_idk.json"))
    ap.add_argument("--day", type=int, default=8)
    ap.add_argument("--soc", type=float, default=None, help="current SOC %% (default: planned start SOC of that day)")
    ap.add_argument("--time", default="09:00", help="HH:MM clock time of the re-solve")
    ap.add_argument("--dist-done", type=float, default=0.0)
    ap.add_argument("--solar-eff", type=float, default=None, help="measured array efficiency 0..1")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    plan = show_plan(a.file)
    if a.plan_only:
        return 0

    row = plan["days"][str(a.day)]
    soc = a.soc if a.soc is not None else row["soc_start_pct"]
    variant = plan.get("variant", "aryaman")
    print(f"Re-solving Day {a.day} from {a.time}, SOC {soc:.1f}% "
          f"(plan said {row['distance_km']:.1f} km, finish {row.get('eta')}, end SOC {row['soc_end_pct']:.1f}%)\n")

    cmd = [sys.executable, "-m", "optimizers.trust_region", "resolve-intraday",
           "--day", str(a.day), "--soc", str(soc), "--time", a.time,
           "--dist-done", str(a.dist_done), "--variant", variant]
    if a.solar_eff is not None:
        cmd += ["--solar-eff", str(a.solar_eff)]
    env = dict(os.environ, AGNIRATH_TIER2="serial", PYTHONPATH=HERE)
    return subprocess.call(cmd, cwd=HERE, env=env)


if __name__ == "__main__":
    sys.exit(main())

"""
Run the strategy-model scenarios for the final report and slides, one after another, and save each result.

The model code is not changed. Scenarios that use the race-identified parameters set them on CarState at run
time, exactly like run_refined.py. Everything else is the normal trust_region CLI.

Scenarios (in order):
  0  demo_timing      run_demo.py --day 4 --soc 60 --time 09:30, timed (single-day live re-plan)
  1  pre_race         what the model gives before the race: original parameters, probable routes
                      (Day 2 surveyed loop, Day 3 'aryaman'), solar 0.189 as deployed, full 8 days from 100 %
  2  refined_actual   race-identified parameters on the routes actually issued (as_given), solar 0.17,
                      full 8 days from 100 %: the plan we should have followed
  3  old_day2_actual  original parameters, as_given routes, re-solve from Day 2 at the actual Day 2 start SOC,
                      solar 0.189: what the old model would have said on the morning of Day 2
  4  refined_day2_actual   same as 3 with the race-identified parameters and solar 0.17

Day 2 start SOC: 87 % by default. From the BMS cell voltages at 07:57 on 11 Sept (4.09 V per cell at rest):
voltage lookup 88.6 %, fitted reference SOC 86.5 % (lstm_soc/soc_lstm.py). Override with --day2-soc.

usage (from anywhere, after git pull):
  python Pre-WSC/sasol_analysis/plan_vs_actual/run_scenarios.py              # all, skips ones already saved
  python Pre-WSC/sasol_analysis/plan_vs_actual/run_scenarios.py --only 1,3   # by number or name
  python Pre-WSC/sasol_analysis/plan_vs_actual/run_scenarios.py --force      # re-run even if saved

Output: Pre-WSC/sasol_analysis/plan_vs_actual/scenarios/
  <name>.json        the strategy JSON written by the model
  <name>.log         everything the model printed
  summary.csv / .md  converged, feasible, km per day, total km, run time; NOT CONVERGED is flagged
"""
import csv
import glob
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[2] / "Model"
OUT = HERE / "scenarios"

IDENTIFIED = dict(motor_eff=0.773, crr=0.0103, cda_m2=0.126, p_idle_w=9.2)   # same as run_refined.py


def scenarios(day2_soc):
    return [
        ("demo_timing", "demo", ["run_demo.py", "--day", "4", "--soc", "60", "--time", "09:30"]),
        ("pre_race", "original",
         ["resolve", "--resume-day", "1", "--start-soc", "100", "--variant", "aryaman", "--solar-eff", "0.189"]),
        ("refined_actual", "refined",
         ["resolve", "--resume-day", "1", "--start-soc", "100", "--variant", "as_given", "--solar-eff", "0.17"]),
        ("old_day2_actual", "original",
         ["resolve", "--resume-day", "2", "--start-soc", str(day2_soc), "--variant", "as_given",
          "--solar-eff", "0.189"]),
        ("refined_day2_actual", "refined",
         ["resolve", "--resume-day", "2", "--start-soc", str(day2_soc), "--variant", "as_given",
          "--solar-eff", "0.17"]),
    ]


def child(mode, args):
    """Runs inside the subprocess: the model's own CLI, with the identified parameters if asked."""
    os.chdir(MODEL)
    sys.path.insert(0, str(MODEL))
    if mode == "refined":
        import configs.car_config as cc
        orig_init = cc.CarState.__init__

        def init(self, *a, **k):
            orig_init(self, *a, **k)
            for key, val in IDENTIFIED.items():
                object.__setattr__(self, key, val)

        cc.CarState.__init__ = init
    sys.argv = ["optimizers.trust_region"] + args
    runpy.run_module("optimizers.trust_region", run_name="__main__", alter_sys=True)


def run_one(name, mode, args):
    log = OUT / f"{name}.log"
    before = set(glob.glob(str(MODEL / "output" / "strategy_*.json")))
    if mode == "demo":
        cmd = [sys.executable] + args
    else:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", mode] + args
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    print(f"\n=== {name}: {' '.join(args)}  ({mode} parameters)" if mode != "demo" else f"\n=== {name}")
    print(f"    log: {log}")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=MODEL, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    dt = time.time() - t0
    new = sorted(set(glob.glob(str(MODEL / "output" / "strategy_*.json"))) - before, key=os.path.getmtime)
    row = dict(name=name, params="identified" if mode == "refined" else "original", args=" ".join(args),
               exit_code=rc, runtime_min=round(dt / 60, 2), converged="", feasible="", total_km="",
               **{f"day{d}_km": "" for d in range(1, 9)}, file="")
    if new:
        dst = OUT / f"{name}.json"
        shutil.move(new[-1], dst)
        for extra in new[:-1]:
            shutil.move(extra, OUT / f"{name}__{Path(extra).name}")
        row["file"] = dst.name
        d = json.load(open(dst, encoding="utf-8"))
        if mode != "demo":
            row.update(converged=d.get("converged"), feasible=d.get("feasible"),
                       total_km=d.get("total_distance_km"))
            for k, day in d.get("days", {}).items():
                row[f"day{k}_km"] = day.get("distance_km")
        if mode == "refined":
            json.dump(dict(identified=IDENTIFIED, args=args), open(OUT / f"{name}.params.json", "w"), indent=2)
    flag = ""
    text = open(log, encoding="utf-8", errors="replace").read()
    if mode != "demo" and not new and re.search(r"feasible\s*:\s*False", text):
        # The model saves no JSON when no legal plan exists; that is a result, not a crash.
        row.update(converged="n/a", feasible=False)
        m = re.findall(r"total=([\d.]+) km feasible=False", text)
        row["total_km"] = f"infeasible (last try {m[-1]} km)" if m else "infeasible"
        flag = "  <-- INFEASIBLE: no legal plan from this state, nothing saved"
    elif mode != "demo":
        if rc != 0 or not new:
            flag = "  <-- FAILED, see log"
        elif row["converged"] is not True:
            flag = "  <-- NOT CONVERGED"
        elif row["feasible"] is not True:
            flag = "  <-- converged but INFEASIBLE"
    print(f"    done in {row['runtime_min']} min | exit {rc} | converged {row['converged']} | "
          f"feasible {row['feasible']} | total {row['total_km']} km{flag}")
    return row


def write_summary(rows):
    keys = list(rows[0].keys())
    old = {}
    path = OUT / "summary.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            old = {r["name"]: r for r in csv.DictReader(f)}
    for r in rows:
        old[r["name"]] = r
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(old.values())
    with open(OUT / "summary.md", "w", encoding="utf-8") as f:
        f.write("| scenario | params | converged | feasible | total km | " +
                " | ".join(f"D{d}" for d in range(1, 9)) + " | min |\n")
        f.write("|" + "---|" * 14 + "\n")
        for r in old.values():
            conv = r["converged"]
            conv = "**NO**" if str(conv) == "False" else conv
            f.write(f"| {r['name']} | {r['params']} | {conv} | {r['feasible']} | {r['total_km']} | " +
                    " | ".join(str(r[f'day{d}_km']) for d in range(1, 9)) + f" | {r['runtime_min']} |\n")


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["--child"]:
        return child(argv[1], argv[2:])
    day2_soc = float(argv[argv.index("--day2-soc") + 1]) if "--day2-soc" in argv else 87.0
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    force = "--force" in argv
    OUT.mkdir(exist_ok=True)
    rows = []
    for i, (name, mode, args) in enumerate(scenarios(day2_soc)):
        if only and str(i) not in only and name not in only:
            continue
        if (OUT / f"{name}.json").exists() and not force:
            print(f"=== {name}: already saved, skipping (use --force to re-run)")
            continue
        rows.append(run_one(name, mode, args))
        write_summary(rows)
    if rows:
        print(f"\nSummary: {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()

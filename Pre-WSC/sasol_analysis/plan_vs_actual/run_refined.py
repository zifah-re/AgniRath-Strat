"""
Run the unchanged strategy model with the parameters identified from the Sasol telemetry (separate.py).

No Model code is changed: CarState instances get the identified values right after construction, then the
normal CLI entry point runs. Car objects travel to the Tier 2 workers already carrying these values
(pickled instance state; dataclasses.replace copies them).

Effective planner parameters = separate.py fit under the model's own air assumptions (rho 1.2, 10 m wind):
    motor_eff 0.773, crr 0.0103, cda_m2 0.126, p_idle_w 9.2 (measured stationary bus draw)
Solar: --solar-eff passed through (0.17 = whole array while driving with canopy shading, solar_check.py;
0.245 = unshaded cells, the bracket).

usage (from anywhere):
  python run_refined.py --solar-eff 0.17 [any other trust_region resolve args]
  default args: resolve --resume-day 2 --start-soc 90 --variant as_given
Output JSON is moved to Pre-WSC/sasol_analysis/plan_vs_actual/refined/ with a sidecar of the parameters.
"""
import glob
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[2] / "Model"
OUT = HERE / "refined"

IDENTIFIED = dict(motor_eff=0.773, crr=0.0103, cda_m2=0.126, p_idle_w=9.2)


def main():
    args = sys.argv[1:]
    if not any(a in ("resolve", "resolve-intraday") for a in args):
        args = ["resolve", "--resume-day", "2", "--start-soc", "90", "--variant", "as_given"] + args
    os.chdir(MODEL)
    sys.path.insert(0, str(MODEL))
    import configs.car_config as cc

    orig_init = cc.CarState.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        for key, val in IDENTIFIED.items():
            object.__setattr__(self, key, val)

    cc.CarState.__init__ = init
    t0 = time.time()
    before = set(glob.glob(str(MODEL / "output" / "strategy_*.json")))
    sys.argv = ["optimizers.trust_region"] + args
    runpy.run_module("optimizers.trust_region", run_name="__main__", alter_sys=True)
    new = sorted(set(glob.glob(str(MODEL / "output" / "strategy_*.json"))) - before)
    OUT.mkdir(exist_ok=True)
    for f in new:
        dst = OUT / Path(f).name
        shutil.move(f, dst)
        json.dump(dict(identified=IDENTIFIED, args=args, runtime_s=round(time.time() - t0)),
                  open(dst.with_suffix(".params.json"), "w"), indent=2)
        print("saved", dst)


if __name__ == "__main__":
    main()

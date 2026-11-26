"""
Run one resolve-intraday solve of the UNCHANGED model with chosen car values (helper for task1_day2.py and
sensitivity.py). Model code is not modified: CarState instances get the values right after construction.
The solver's GA is seeded (numpy default_rng(None) -> 12345) so runs are reproducible.

run_intraday(params, day, soc, time_hhmm, variant, loops_left=None, solar_eff=None) -> result dict
"""
import glob
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[2] / "Model"


def _child(params_json, argv_json):
    params, argv = json.loads(params_json), json.loads(argv_json)
    os.chdir(MODEL)
    sys.path.insert(0, str(MODEL))
    import numpy as np
    orig_rng = np.random.default_rng
    np.random.default_rng = lambda seed=None: orig_rng(12345 if seed is None else seed)
    import configs.car_config as cc
    orig_init = cc.CarState.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        for key, val in params.items():
            object.__setattr__(self, key, val)

    cc.CarState.__init__ = init
    os.environ["AGNIRATH_TIER2"] = "serial"
    before = set(glob.glob("output/strategy_*intraday*.json"))
    sys.argv = ["optimizers.trust_region"] + argv
    try:
        runpy.run_module("optimizers.trust_region", run_name="__main__", alter_sys=True)
    except SystemExit:
        pass
    new = sorted(set(glob.glob("output/strategy_*intraday*.json")) - before)
    if new:
        print("RESULT_FILE", os.path.abspath(new[-1]))


def run_intraday(params, day, soc, time_hhmm, variant="aryaman", loops_left=None, solar_eff=None):
    argv = ["resolve-intraday", "--day", str(day), "--soc", str(soc), "--time", time_hhmm,
            "--dist-done", "0", "--variant", variant]
    if loops_left is not None:
        argv += ["--loops-left", str(loops_left)]
    if solar_eff is not None:
        argv += ["--solar-eff", str(solar_eff)]
    out = subprocess.run([sys.executable, __file__, json.dumps(params), json.dumps(argv)],
                         capture_output=True, text=True)
    path = [l.split(" ", 1)[1] for l in out.stdout.splitlines() if l.startswith("RESULT_FILE")]
    if not path:
        return dict(feasible=False, error=(out.stdout + out.stderr)[-1500:])
    r = json.load(open(path[0]))["result"]
    os.remove(path[0])
    return r


if __name__ == "__main__":
    _child(sys.argv[1], sys.argv[2])

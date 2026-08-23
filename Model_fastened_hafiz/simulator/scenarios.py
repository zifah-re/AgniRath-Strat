"""
simulator/scenarios.py — STUB (block 3.2, owner: Junior C).

Scenario/regression suite: ideal / cloudy / headwind / worst (inherited
from Diyaansh's mpc_simulation_task + Prahlad's benchmarks) + log-replay
(WSC'25 Chennai logs) + degraded (capability drops, Plan v3 §7). Regression
baselines recorded here become the merge gate (Plan v3 §9 standards).
"""

SCENARIOS = ("ideal", "cloudy", "headwind", "worst", "log_replay", "degraded")

def run(name: str):
    raise NotImplementedError("block 3.2 — Junior C")

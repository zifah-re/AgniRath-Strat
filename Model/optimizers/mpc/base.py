"""
optimizers/mpc/base.py — backend-agnostic MPC interface (block 7, Junior C).

Baseline backend: SLSQP port of Dashboard/mpc.py (N=10). Challenger: IPOPT
(Dashboard/mpc_ipopt.py), promoted only if it wins the automated benchmark
on solve time AND violation rate (Plan v3 §8).

Race logic wrapped around any backend (Plan v3 §7.1, §8):
  * routine periodic L2 re-solve every solver_config.
    PERIODIC_RESOLVE_INTERVAL_S,
  * threshold re-plan triggers (SOC deviation / forecast revision /
    capability change),
  * BOTH suspended by the car-health gate while vehicle_state != DRIVING,
  * loop commit/abort ahead of turnaround, trailering advisory, swap
    countdown. All outputs are advisories to the strategist.
"""
import enum

class VehicleState(enum.Enum):
    DRIVING = "driving"
    STOPPED = "stopped"
    REPAIRING = "repairing"
    TRAILERED = "trailered"

class MPCBackend:
    def step(self, telemetry, reference, horizon):  # -> target speeds
        raise NotImplementedError

def make_backend(name: str) -> MPCBackend:
    raise NotImplementedError("block 7 — Junior C (slsqp first, ipopt challenger)")

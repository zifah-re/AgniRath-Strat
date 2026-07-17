import time
import numpy as np
from helper import get_profile, get_current_state
from mpc import main as run_slsqp
from mpc_ipopt import main as run_ipopt

def calculate_objective_cost(speeds_ms, target_speeds_ms):
    """
    Calculates the combined mathematical cost based on your specific cost function:
    Tracking Penalty + Delta-V (Control Effort) Penalty.
    (Note: SoC penalty is omitted here to keep the benchmark lightweight, 
    as it requires re-simulating the full PVLib solar irradiance graph).
    """
    cost = 0.0
    v_prev = speeds_ms[0]
    
    for i in range(1, len(speeds_ms)):
        v_next = speeds_ms[i]
        
        # Safe array index fallback for targets
        target = target_speeds_ms[i-1] if (i-1) < len(target_speeds_ms) else target_speeds_ms[-1]
        
        # 1. Target Tracking Cost
        cost += 1.0 * (v_next - target) ** 2
        # 2. Control Effort / Smoothness Cost
        cost += 0.5 * (v_next - v_prev) ** 2
        
        v_prev = v_next
        
    return cost

def run_benchmark():
    print("Starting Comprehensive Benchmark...")
    print("-" * 50)
    
    # 1. Fetch exact state and targets for accurate cost calculation
    try:
        state = get_current_state()
        profiles = get_profile(["TargetProfile"])
        current_speed_ms = state['Speed'] * (5 / 18)
        
        target_profile = profiles.get("TargetProfile", [])
        if len(target_profile) > 0 and isinstance(target_profile[0], (tuple, list)):
            target_profile = [i for _, i in target_profile]
        elif len(target_profile) == 0:
            target_profile = [current_speed_ms] * 50 # Fallback
            
        target_profile_ms = np.array(target_profile) * (5 / 18)
    except Exception as e:
        print(f"Warning: Could not fetch exact target profiles for cost analysis ({e}).")
        target_profile_ms = np.array([0.0] * 50)

    # 2. Run Solvers
    try:
        print("Running SciPy (SLSQP)...")
        start_slsqp = time.perf_counter()
        slsqp_output = run_slsqp()
        slsqp_duration = time.perf_counter() - start_slsqp
        
        print("Running CasADi (IPOPT)...")
        start_ipopt = time.perf_counter()
        ipopt_output = run_ipopt()
        ipopt_duration = time.perf_counter() - start_ipopt
    except Exception as e:
        print(f"Error during execution: {e}")
        return

    # 3. Process Outputs (Convert to m/s)
    slsqp_speeds_ms = np.array([v for t, v in slsqp_output]) * (5 / 18)
    ipopt_speeds_ms = np.array([v for t, v in ipopt_output]) * (5 / 18)

    # 4. Performance Metrics
    print("-" * 50)
    print("PERFORMANCE RESULTS")
    print("-" * 50)
    print(f"SLSQP Execution Time: {slsqp_duration:.4f} seconds")
    print(f"IPOPT Execution Time: {ipopt_duration:.4f} seconds")
    
    speedup = slsqp_duration / ipopt_duration if ipopt_duration < slsqp_duration else ipopt_duration / slsqp_duration
    winner = "IPOPT" if ipopt_duration < slsqp_duration else "SLSQP"
    print(f"Result: {winner} is {speedup:.2f}x faster.")

    # 5. Trajectory Optimality & Bounds
    print("-" * 50)
    print("TRAJECTORY BOUNDS & CONTROL EFFORT")
    print("-" * 50)
    
    def evaluate_bounds(speeds_ms, name):
        if len(speeds_ms) == 0:
            print(f"{name} Analysis: Failed (Empty Array)")
            return
            
        control_effort = np.sum(np.diff(speeds_ms)**2)
        min_v = np.min(speeds_ms)
        max_v = np.max(speeds_ms)
        bounds_respected = "Yes" if (min_v >= 0.099 and max_v <= 25.001) else "No (Violated Constraints)"
        
        print(f"{name} Analysis:")
        print(f"  Control Effort (Delta-V Penalty): {control_effort:.4f}")
        print(f"  Min Speed: {min_v:.2f} m/s | Max Speed: {max_v:.2f} m/s")
        print(f"  Bounds Respected (0.1 - 25.0 m/s): {bounds_respected}\n")

    evaluate_bounds(slsqp_speeds_ms, "SLSQP")
    evaluate_bounds(ipopt_speeds_ms, "IPOPT")

    # 6. Similarity Metrics
    print("-" * 50)
    print("SIMILARITY METRICS (IPOPT vs SLSQP)")
    print("-" * 50)
    
    if len(slsqp_speeds_ms) == len(ipopt_speeds_ms) and len(slsqp_speeds_ms) > 0:
        diff = np.abs(ipopt_speeds_ms - slsqp_speeds_ms)
        mae = np.mean(diff)
        rmse = np.sqrt(np.mean(diff**2))
        max_diff = np.max(diff)
        
        print(f"Mean Absolute Error (MAE): {mae:.4f} m/s")
        print(f"Root Mean Square Error (RMSE): {rmse:.4f} m/s")
        print(f"Max Absolute Difference: {max_diff:.4f} m/s")
    else:
        print("Error: Output lengths do not match or are empty.")

    # 7. Total Cost Analysis
    print("-" * 50)
    print("TOTAL OBJECTIVE COST ANALYSIS")
    print("-" * 50)
    
    cost_slsqp = calculate_objective_cost(slsqp_speeds_ms, target_profile_ms)
    cost_ipopt = calculate_objective_cost(ipopt_speeds_ms, target_profile_ms)
    
    print(f"SLSQP Total Objective Cost: {cost_slsqp:.4f}")
    print(f"IPOPT Total Objective Cost: {cost_ipopt:.4f}")
    
    print("\nVerdict:")
    if cost_ipopt < cost_slsqp:
        print("-> IPOPT found a mathematically superior (lower total cost) solution.")
        print("-> The speed discrepancy (MAE/RMSE) indicates SLSQP failed to converge to the global minimum.")
    elif cost_slsqp < cost_ipopt:
        print("-> SLSQP found a mathematically superior (lower total cost) solution.")
    else:
        print("-> Both solvers converged to identical cost minimums.")

if __name__ == "__main__":
    run_benchmark()
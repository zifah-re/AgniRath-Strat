import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# =====================================================================
# 1. SIMULATED ENVIRONMENT & PROBLEM SETUP (Based on reference file)
# =====================================================================
np.random.seed(42)
NUM_POINTS = 200  # Reduced profile dimensions for benchmark efficiency
ds = np.random.uniform(80, 120, NUM_POINTS)       # Segment distances (m)
theta = np.random.uniform(-0.05, 0.05, NUM_POINTS) # Segment slope angles (rad)

# Reduced dimensionality for optimizer nodes to keep problem smooth and tractable
NUM_NODES = 20  
node_indices = np.linspace(0, len(ds) - 1, NUM_NODES, dtype=int)
node_distances = np.cumsum(ds)[node_indices]
full_distances = np.cumsum(ds)

def interpolate_velocity(v_coarse):
    interp_fn = interp1d(node_distances, v_coarse, kind='linear', fill_value='extrapolate')
    return interp_fn(full_distances)

def P_solar(t):
    # Simplified solar model based on time of day
    mean = 12 * 3600
    sigma = 11600
    irr_max = 1073
    return irr_max * 0.24 * 6 * np.exp(-0.5 * ((t - mean) / sigma)**2)

def simulate_car(v_profile, start_SoC=1.0):
    capacity_Wh = 3100
    SoC = start_SoC
    t_elapsed = 0.0
    min_SoC = start_SoC
    
    # Physics constants
    k, m, g, mu, eta_motor, eta_regen = 0.09, 300, 9.81, 0.007, 0.9, 0.7
    
    dt = ds / v_profile
    t_cumulative = 8 * 3600 + np.concatenate([[0], np.cumsum(dt)[:-1]])
    
    # Vectorized / looped battery evaluation
    for i in range(len(v_profile)):
        v = v_profile[i]
        P_mech = k*(v**3) + m*g*v*np.sin(theta[i]) + m*g*v*np.cos(theta[i])*mu
        P_elec = P_mech / eta_motor if P_mech >= 0 else P_mech * eta_regen
        P_batt = P_elec + 100 - P_solar(t_cumulative[i])
        
        SoC -= (P_batt * dt[i] / 3600) / capacity_Wh
        if SoC < min_SoC:
            min_SoC = SoC
            
    return SoC, np.sum(dt), min_SoC

def objective_function(v_coarse):
    v_full = interpolate_velocity(v_coarse)
    SoC_final, t_final, min_SoC = simulate_car(v_full)
    
    # Loss components
    cost_time = 10.0 * ((t_final / (9 * 3600))**2)
    cost_SoC = 4.0 * ((1.0 - SoC_final)**2)
    cost_control = 1e-4 * np.mean(np.diff(v_coarse)**2)
    
    # Handle hard constraint via penalization for the global metaheuristic meta-step
    penalty = 0.0
    if min_SoC < 0.20:
        penalty = 1000.0 * (0.20 - min_SoC)**2
        
    return cost_time + cost_SoC + cost_control + penalty

def soc_constraint(v_coarse):
    v_full = interpolate_velocity(v_coarse)
    _, _, min_SoC = simulate_car(v_full)
    return min_SoC - 0.20

# Optimization bounds (60 km/h to 100 km/h in m/s)
V_MIN, V_MAX = 60.0 / 3.6, 100.0 / 3.6
bounds = [(V_MIN, V_MAX) for _ in range(NUM_NODES)]
v_guess = np.full(NUM_NODES, 20.0)

# =====================================================================
# 2. HYBRID GLOBAL METAHURISTIC PHASE (PSO + DE Global Exploration)
# =====================================================================
def hybrid_pso_de_explore(obj_func, num_nodes, pop_size=15, max_iter=25):
    """
    Hybrid exploration: Uses Particle Swarm Optimization mechanics combined with 
    Differential Evolution mutation components to avoid premature local stagnation.
    """
    # Initialize Population
    X = np.random.uniform(V_MIN, V_MAX, (pop_size, num_nodes))
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_fit = np.array([obj_func(ind) for ind in X])
    
    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    w, c1, c2, F = 0.5, 1.5, 1.5, 0.6  # Metaheuristic coefficients
    
    for _ in range(max_iter):
        for i in range(pop_size):
            # DE Mutation Component: Choose 3 distinct random individuals
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            de_mutation = pbest[r1] + F * (pbest[r2] - pbest[r3])
            
            # PSO Velocity Update utilizing the DE mutation element as cognitive guidance
            r_1, r_2 = np.random.rand(), np.random.rand()
            V[i] = w * V[i] + c1 * r_1 * (de_mutation - X[i]) + c2 * r_2 * (gbest - X[i])
            X[i] = np.clip(X[i] + V[i], V_MIN, V_MAX)
            
            # Evaluate Fitness
            current_fit = obj_func(X[i])
            if current_fit < pbest_fit[i]:
                pbest[i] = X[i].copy()
                pbest_fit[i] = current_fit
                if current_fit < gbest_fit:
                    gbest = X[i].copy()
                    gbest_fit = current_fit
                    
    return gbest

# =====================================================================
# 3. BENCHMARK EXECUTION INTERFACE
# =====================================================================
print("Executing Pure SLSQP...")
start_pure = time.time()
pure_res = minimize(
    objective_function, v_guess, method='SLSQP', bounds=bounds,
    constraints={'type': 'ineq', 'fun': soc_constraint},
    options={'maxiter': 500, 'ftol': 1e-4}
)
pure_time = time.time() - start_pure

print("Executing Hybrid Global + SLSQP Local Search Pipeline...")
start_hybrid = time.time()
# Step A: Broad Global Exploration Phase to clear bad local minima
global_valley_guess = hybrid_pso_de_explore(objective_function, NUM_NODES)
# Step B: Strict Local Mathematical Refinement
hybrid_res = minimize(
    objective_function, global_valley_guess, method='SLSQP', bounds=bounds,
    constraints={'type': 'ineq', 'fun': soc_constraint},
    options={'maxiter': 500, 'ftol': 1e-4}
)
hybrid_time = time.time() - start_hybrid

# =====================================================================
# 4. DATA VISUALIZATION AND PERFORMANCE ANALYSIS
# =====================================================================
print(f"\n--- RESULTS ---\nPure SLSQP Loss: {pure_res.fun:.4f} | Time: {pure_time:.2f}s")
print(f"Hybrid + SLSQP Loss: {hybrid_res.fun:.4f} | Time: {hybrid_time:.2f}s\n")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Convergence Minimum Loss Comparison
bars_loss = ax[0].bar(['Pure SLSQP', 'Hybrid + SLSQP'], [pure_res.fun, hybrid_res.fun], color=['#e74c3c', '#2ecc71'], width=0.5)
ax[0].set_ylabel('Minimum Objective Loss Value', fontsize=12)
ax[0].set_title('Global Optima Performance Comparison\n(Lower Loss is Better)', fontsize=13, fontweight='bold')
ax[0].grid(axis='y', linestyle='--', alpha=0.7)
ax[0].bar_label(bars_loss, fmt='%.4f', padding=3)

# Subplot 2: Computational Performance Execution Speed
bars_time = ax[1].bar(['Pure SLSQP', 'Hybrid + SLSQP'], [pure_time, hybrid_time], color=['#c0392b', '#27ae60'], width=0.5)
ax[1].set_ylabel('Execution Time (Seconds)', fontsize=12)
ax[1].set_title('Computational Speed Comparison', fontsize=13, fontweight='bold')
ax[1].grid(axis='y', linestyle='--', alpha=0.7)
ax[1].bar_label(bars_time, fmt='%.2f s', padding=3)

plt.tight_layout()
plt.show()
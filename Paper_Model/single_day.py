import numpy as np
from scipy.optimize import minimize
import constants as const
from tqdm import tqdm

class SingleDayOptimizer:
    def __init__(self, route_distances, route_gradients, speed_caps, wind_speeds, expected_ghi, allowed_time,desc="SLSQP Iterations"):
        self.N = len(route_distances)
        self.dx = np.array(route_distances)       
        self.theta = np.arctan(np.array(route_gradients) / 100.0)   
        self.v_cap = np.array(speed_caps)         
        self.v_wind = np.array(wind_speeds)       
        self.ghi = np.array(expected_ghi)         
        
        self.initial_soc = 1.0 
        self.allowed_time = allowed_time
        self.desc = desc
        
        # Pre-calculate time-independent physical constants across the array
        self.p_sun = self.ghi * const.ARRAY_EFFICIENCY * const.ARRAY_AREA_M2
        self.f_roll = const.CRR * const.MASS_KG * const.G_MS2 * np.cos(self.theta)
        self.f_grav = const.MASS_KG * const.G_MS2 * np.sin(self.theta)

    def _calculate_arrays(self, v):
        """
        Core vectorized computation with explicit zero-division protection.
        """
        v_next = np.append(v[1:], v[-1])
        
        # Safe time vector calculation
        dt = np.zeros_like(v)
        valid_dt = v > 0
        dt[valid_dt] = self.dx[valid_dt] / v[valid_dt]
        
        # Mechanical forces
        f_drag = 0.5 * const.AIR_DENSITY * const.CDA_M2 * (v + self.v_wind)*abs((v + self.v_wind))
        f_total = f_drag + self.f_roll + self.f_grav
        
        # Acceleration power mapping without runtime warnings
        e_acc = 0.5 * const.MASS_KG * (v_next**2 - v**2)
        p_acc = np.zeros_like(v)
        safe_acc_mask = (dt > 0) & (self.dx > 0)
        p_acc[safe_acc_mask] = e_acc[safe_acc_mask] / dt[safe_acc_mask]
        
        p_wheels = (f_total * v) + p_acc
        
        # Vectorized drivetrain and regen logic
        p_motor = np.where(p_wheels >= 0, p_wheels / const.MOTOR_EFF, 0.0)
        p_regen = np.where(p_wheels < 0, np.abs(p_wheels) * const.REGEN_EFF, 0.0)
        
        p_loss = p_motor - p_regen + const.P_AUX
        p_net = self.p_sun - p_loss
        
        return p_net, p_loss, dt

    def objective_function(self, v):
        p_net, _, dt = self._calculate_arrays(v)
        total_energy_j = self.initial_soc * const.BATTERY_JOULES + np.sum(p_net * dt)
        final_soc = total_energy_j / const.BATTERY_JOULES
        return -final_soc 

    def calculate_total_energy_used(self, v):
        p_net, _, dt = self._calculate_arrays(v)
        return -np.sum(p_net * dt)

    def constraint_time(self, v):
        dt = np.zeros_like(v)
        valid_mask = (self.dx > 0) & (v > 0)
        dt[valid_mask] = self.dx[valid_mask] / v[valid_mask]
        return self.allowed_time - np.sum(dt)

    def constraint_energy_min(self, v):
        p_net, _, dt = self._calculate_arrays(v)
        cum_energy = self.initial_soc * const.BATTERY_JOULES + np.cumsum(p_net * dt)
        min_energy_j = (const.SOC_MIN_PCT / 100.0) * const.BATTERY_JOULES
        return np.min(cum_energy) - min_energy_j

    def constraint_motor_power(self, v):
        _, p_loss, _ = self._calculate_arrays(v)
        return const.P_MAX - np.max(p_loss)

    def constraint_energy_max(self, v):
        p_net, _, dt = self._calculate_arrays(v)
        cum_energy = self.initial_soc * const.BATTERY_JOULES + np.cumsum(p_net * dt)
        max_energy_j = (const.SOC_MAX_PCT / 100.0) * const.BATTERY_JOULES
        return max_energy_j - np.max(cum_energy)

    def run_optimization(self):
        v0 = np.full(self.N, const.V_MAX_MS * 0.7)
        bounds = []
        for i in range(self.N):
            upper_bound = min(self.v_cap[i], const.V_MAX_MS)
            lower_bound = min(const.V_MIN_MS, upper_bound)
            bounds.append((lower_bound, upper_bound))

        constraints = [
            {'type': 'ineq', 'fun': self.constraint_time},
            {'type': 'ineq', 'fun': self.constraint_energy_min},
            {'type': 'ineq', 'fun': self.constraint_energy_max},
            {'type': 'ineq', 'fun': self.constraint_motor_power}
        ]

        options = {'ftol': const.SLSQP_FTOL, 'eps': const.SLSQP_EPS, 'maxiter': const.MAX_ITER}

        # 1. Initialize the progress bar
        pbar = tqdm(total=const.MAX_ITER, desc=self.desc, leave=True)

        # 2. Define the callback function
        def iteration_callback(xk):
            pbar.update(1)

        # 3. Attach the callback to the minimize function
        result = minimize(
            self.objective_function, 
            v0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints, 
            options=options, 
            callback=iteration_callback
        )
        
        # 4. Close the progress bar cleanly
        pbar.close()
        
        opt_v = result.x if result.success else v0
        energy_used = self.calculate_total_energy_used(opt_v)
        
        return opt_v, energy_used
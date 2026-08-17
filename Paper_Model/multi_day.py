import numpy as np
import constants as const
from single_day import SingleDayOptimizer

class MultiDayOptimizer:
    def __init__(self, day_route_notes, route_data_provider):
        """
        route_data_provider is a callback function: provider(day_index, num_loops) 
        that returns (distances, gradients, speed_caps, winds, ghis, allowed_time_s)
        so this class can run SingleDayOptimizer without parsing files directly.
        """
        self.num_days = 8
        self.max_loops_per_day = 4
        self.routes = day_route_notes
        self.provider = route_data_provider
        self.memo = {}
        
        # Precomputed arrays tracking metrics for all 8 days
        self.delta_day = np.zeros(self.num_days)
        self.epsilon_day = np.zeros(self.num_days)
        self.arr0_day = np.zeros(self.num_days)
        
        self._precompute_energy_deltas()

    def _precompute_energy_deltas(self):
        """
        Calculates Arr0, Arr1, and epsilon_day(x) for all days by 
        running the high-resolution SLSQP optimizer twice per day.
        """
        for day_idx in range(self.num_days):
            loops = self.routes[day_idx].get('loops')
            
            if not loops:
                self.delta_day[day_idx] = 0.0
                self.epsilon_day[day_idx] = 0.0
                self.arr0_day[day_idx] = 0.0
                continue
                
            self.delta_day[day_idx] = loops[0][1] * 1000.0 # Extract distance in meters
            
            # Run 0 loops (Arr0)
            data_0 = self.provider(day_idx, 0)
            opt_0 = SingleDayOptimizer(*data_0, desc=f"[Day {day_idx + 1} | 0 Loops]")
            _, arr0 = opt_0.run_optimization()
            self.arr0_day[day_idx] = arr0
            
            # Run 1 loop (Arr1)
            data_1 = self.provider(day_idx, 1)
            opt_1 = SingleDayOptimizer(*data_1, desc=f"[Day {day_idx + 1} | 1 Loop]")
            _, arr1 = opt_1.run_optimization()
            
            # Energy Delta
            self.epsilon_day[day_idx] = arr1 - arr0
            
        self.sum_delta = np.sum(self.delta_day)
        self.sum_epsilon = np.sum(self.epsilon_day)

    def l_day(self, day_index):
        """ Distance Ratio: Sum of all loop distances / Specific loop distance """
        if self.delta_day[day_index] == 0:
            return 0.0
        return self.sum_delta / self.delta_day[day_index]

    def e_day(self, day_index):
        """ Energy Ratio: Expected energy delta / Sum of all energy deltas """
        if self.sum_epsilon == 0:
            return 0.0
        return self.epsilon_day[day_index] / self.sum_epsilon

    def calculate_fx(self, day_index, num_loops):
        """ Objective Function (fx) = (e_day * l_day) / phi_day """
        if num_loops == 0:
            return 0.0 
        
        return (self.e_day(day_index) * self.l_day(day_index)) / num_loops

    def requires_trailering(self, day_index):
        return day_index in [6, 7] # Days 7 and 8

    def dp_backward_induction(self, current_day, accumulated_dist):
        if current_day == self.num_days:
            return 0, []

        state_key = (current_day, accumulated_dist)
        if state_key in self.memo:
            return self.memo[state_key]

        min_cost = float('inf')
        best_tail_path = []

        max_possible_loops = self.max_loops_per_day
        if self.requires_trailering(current_day):
            max_possible_loops = min(1, max_possible_loops)
            
        if self.delta_day[current_day] == 0.0:
            max_possible_loops = 0 

        for loops in range(max_possible_loops + 1):
            added_dist = loops * self.delta_day[current_day]
            new_total_dist = accumulated_dist + added_dist
            
            cost = self.calculate_fx(current_day, loops)
            tail_cost, tail_path = self.dp_backward_induction(current_day + 1, new_total_dist)
            
            if cost + tail_cost < min_cost:
                min_cost = cost + tail_cost
                best_tail_path = [(current_day, loops)] + tail_path

        self.memo[state_key] = (min_cost, best_tail_path)
        return min_cost, best_tail_path

    def optimize_schedule(self):
        """ Executes the DP and calculates alpha_day (minimum starting SoC). """
        _, optimal_policy = self.dp_backward_induction(0, 0.0)
        
        phi_day = [0] * self.num_days
        alpha_day = [0.0] * self.num_days
        
        # alpha_day(1) is guaranteed to be 100%
        alpha_day[0] = 100.0 
        
        for day, loops in optimal_policy:
            phi_day[day] = loops
            
            # Calculate SoC drop for the day to establish alpha_day(x+1)
            if day < self.num_days - 1:
                energy_used_j = self.arr0_day[day] + (loops * self.epsilon_day[day])
                soc_drop_pct = (energy_used_j / const.BATTERY_JOULES) * 100.0
                alpha_day[day + 1] = alpha_day[day] - soc_drop_pct
                
        return phi_day, alpha_day
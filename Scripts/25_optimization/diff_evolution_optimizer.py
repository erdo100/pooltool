import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import copy
import pickle
import time
from tkinter import filedialog
from scipy.optimize import differential_evolution
from loss_funcs import evaluate_loss
from billiardenv import BilliardEnv
from helper_funcs import get_ball_positions

class DEOptimizer:
    def __init__(self, SA, params, maxiter=1000, popsize=(200, 200),
                 mutation=(1.0, 0.5), recombination=(0.9, 0.5), strategy='rand2bin',
                 polish=False, workers=24):
        self.SA = SA
        self.sim_env = BilliardEnv()
        self.base_params = copy.deepcopy(params)
        self.bounds = [params.limits[key] for key in params.limits.keys()]
        self.maxiter = maxiter
        self.pop_start, self.pop_end = (popsize, popsize) if isinstance(popsize, int) else popsize
        self.mut_start, self.mut_end = mutation
        self.rec_start, self.rec_end = recombination
        self.strategy = strategy
        self.polish = polish
        self.workers = workers or mp.cpu_count()
        self.parameter_history = []
        self.loss_history = []

    def run_optimization(self):
        def scale_params(params_np, bounds):
            scaled = np.zeros_like(params_np)
            for i, (low, high) in enumerate(bounds):
                scaled[i] = (params_np[i] - low) / (high - low)
            return scaled

        def unscale_params(scaled_params, bounds):
            original = np.zeros_like(scaled_params)
            for i, (low, high) in enumerate(bounds):
                original[i] = scaled_params[i] * (high - low) + low
            return original

        def vector_to_params(params, x_vec):
            for key, x in zip(params.limits.keys(), x_vec):
                params.value[key] = x
            return params

        def loss_wrapper(scaled_vec):
            x = unscale_params(scaled_vec, self.bounds)
            current_params = copy.deepcopy(self.base_params)
            current_params = vector_to_params(current_params, x)
            # self.sim_env.ball_xy_ini, self.sim_env.ball_cols, cueball_phi = get_ball_positions(shot_actual, b1b2b3)
            self.sim_env.prepare_new_shot(current_params)
            self.sim_env.simulate_shot()
            loss = evaluate_loss(self.sim_env, self.SA['Shot'])
            loss_val = sum(np.sum(loss['ball'][i]['total']) for i in range(len(loss['ball'])))
            return loss_val

        def callback_fn(xk, convergence):
            loss = loss_wrapper(xk)
            unscaled = unscale_params(xk, self.bounds)
            self.parameter_history.append(unscaled.copy())
            self.loss_history.append(loss)
            fmt = ", ".join(f"{v:.5f}" for v in unscaled)
            print(f"Gen {len(self.loss_history)} | Best: [{fmt}]  Loss: {loss:.5f}  Conv: {convergence:.5f}")

        init = 'random'
        file_path = filedialog.askopenfilename(title="Load initial population", filetypes=[("Pickle", "*.pkl")])
        if file_path:
            with open(file_path, 'rb') as f:
                init = pickle.load(f)
            print("Initial population loaded from file.")
        else:
            print("No file selected. Keeping random initial population.")

        result = None
        final_pop = init
        for gen in range(self.maxiter):
            curr_pop = int(self.pop_start + (self.pop_end - self.pop_start) * gen / self.maxiter)
            mut = self.mut_start + (self.mut_end - self.mut_start) * (gen / self.maxiter)
            rec = self.rec_start + (self.rec_end - self.rec_start) * (gen / self.maxiter)
            print(f"Generation {gen + 1}/{self.maxiter} | pop={curr_pop} mut={mut:.3f} rec={rec:.3f}")
            bounds_scaled = [(0, 1)] * len(self.bounds)
            b1b2b3 = self.SA["Data"]["B1B2B3"][0]
            shot_actual = self.SA["Shot"][0]


            result = differential_evolution(
                func=loss_wrapper,
                bounds=bounds_scaled,
                strategy='rand2bin',
                maxiter=1,  # Adjust as needed; 1 for per-generation iteration
                mutation=mut,
                recombination=rec,
                popsize=curr_pop,
                updating='deferred',
                callback=callback_fn,
                tol=0.001,
                polish=False,
                disp=True,
                init=final_pop,
            )
            
            final_pop = result.population
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            with open(f"population_{timestamp}.pkl", 'wb') as f:
                pickle.dump(final_pop, f)

        best_unscaled = unscale_params(result.x, self.bounds)
        best_params = copy.deepcopy(self.base_params)
        best_params = vector_to_params(best_params, best_unscaled)
        plt.ioff()
        plt.show()
        return result, best_params
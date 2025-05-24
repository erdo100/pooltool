import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import time
from tkinter import filedialog
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from loss_funcs import evaluate_loss
from billiardenv import BilliardEnv
from helper_funcs import get_ball_positions
import pooltool as pt

class DEOptimizer:
    def __init__(self, shot_actual, params, balls_xy_ini, ball_cols, maxiter, popsize=(50, 50),
                 mutation=(1.0, 0.5), recombination=(0.9, 0.5), strategy='rand2bin',
                 polish=False):

        self.sim_env = BilliardEnv()
        self.shot_actual = shot_actual
        
        self.sim_env.balls_xy_ini = balls_xy_ini
        self.sim_env.ball_cols = ball_cols

        self.base_params = copy.deepcopy(params)
        self.bounds = [self.base_params.limits[key] for key in self.base_params.limits.keys()]
        self.maxiter = maxiter
        self.pop_start, self.pop_end = (popsize, popsize) if isinstance(popsize, int) else popsize
        self.mut_start, self.mut_end = mutation
        self.rec_start, self.rec_end = recombination
        self.strategy = strategy
        self.polish = polish
        self.parameter_history = []
        self.loss_history = []
        self.all_x = []
        self.all_y = []
        

    @staticmethod
    def scale_params(params_np, bounds):
        scaled = np.zeros_like(params_np)
        for i, (low, high) in enumerate(bounds):
            scaled[i] = (params_np[i] - low) / (high - low)
        return scaled

    @staticmethod
    def unscale_params(scaled_params, bounds):
        original = np.zeros_like(scaled_params)
        for i, (low, high) in enumerate(bounds):
            original[i] = scaled_params[i] * (high - low) + low
        return original

    @staticmethod
    def vector_to_params(params, x_vec):
        for key, x in zip(params.limits.keys(), x_vec):
            params.value[key] = x
        return params
    

    def _loss_wrapper(self, scaled_vec):
        x = self.unscale_params(scaled_vec, self.bounds)

        current_params = copy.deepcopy(self.base_params)
        current_params = self.vector_to_params(current_params, x)
        
        self.sim_env.prepare_new_shot(current_params)
        
        self.sim_env.simulate_shot()

        loss = evaluate_loss(self.sim_env, self.shot_actual)

        self.all_x.append(x)
        self.all_y.append(loss["total"])

        return loss["total"]

    def plot_convergence(self, convergence):

        # Format each parameter in best_params to 5 decimal places
        formatted_params = ", ".join(f"{param:.5f}" for param in self.parameter_history[-1])

        # Print the iteration, formatted parameters, and convergence
        print(f"Iteration {len(self.parameter_history)} - Loss: {self.loss_history[-1]:.10f} - Best Params: [{formatted_params}], Convergence: {convergence:.5f}")

        # Plot the loss history and parameter history in a single figure
        plt.figure(1)
        plt.ion()  # Enable interactive mode for real-time updates
        plt.clf()
        
        # Number of parameters + 1 (for the loss plot)
        num_params = len(self.parameter_history[-1])
        param_names = list(self.base_params.limits.keys())
        rows = int(np.ceil(np.sqrt(num_params + 1)))
        cols = int(np.ceil((num_params + 1) / rows))
        
        # Plot loss history
        plt.subplot(rows, cols, 1)
        plt.plot(self.loss_history, 'b-')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title('Loss Function History')
        plt.grid(True)

        for i, (name, (low, high)) in enumerate(zip(self.base_params.limits.keys(), self.base_params.limits.values())):
            plt.subplot(rows, cols, i + 2)
            param_values = [p[i] for p in self.parameter_history]
            plt.plot(param_values, 'r-')
            plt.title(param_names[i])
            plt.ylim(low, high)  # Set y-axis limits to parameter bounds
            plt.grid(True)
        
        plt.tight_layout()
        plt.draw()  # Update the figure
        plt.pause(0.01)  # Pause to update the plot



    def callback_fn(self, xk, convergence):
        loss = self._loss_wrapper(xk)
        unscaled = self.unscale_params(xk, self.bounds)
        self.parameter_history.append(unscaled.copy())
        self.loss_history.append(loss)
        self.plot_convergence(convergence)

    def run_optimization(self):

        init_population = None

        file_path = filedialog.askopenfilename(title="Load initial population", filetypes=[("Pickle", "*.pkl")])
        if file_path:
            with open(file_path, 'rb') as f:
                init_population = pickle.load(f)
            print("Initial population loaded from file.")
        else:
            print("Creating new initial population.")
            # Your custom candidate
            my_candidate = np.array([self.base_params.value[key] for key in self.base_params.limits.keys()])
            my_candidate_scaled = self.scale_params(my_candidate, self.bounds)
            # my_candidate_scaled = my_candidate_scaled.reshape((1, len(self.bounds)))

            # random population
            sampler = qmc.LatinHypercube(d=len(self.bounds))
            random_samples = sampler.random(n=self.pop_start*len(self.bounds) - 1)  # reserve 1 spot for your candidate

            # Insert your candidate
            init_population = np.vstack([my_candidate_scaled, random_samples])

        result = None

        for gen in range(self.maxiter):
            curr_pop = int(self.pop_start + (self.pop_end - self.pop_start) * gen / self.maxiter)
            mut = self.mut_start + (self.mut_end - self.mut_start) * (gen / self.maxiter)
            rec = self.rec_start + (self.rec_end - self.rec_start) * (gen / self.maxiter)
            print(f"Generation {gen + 1}/{self.maxiter} | pop={curr_pop} mut={mut:.3f} rec={rec:.3f}")
            bounds_scaled = [(0, 1)] * len(self.bounds)

            result = differential_evolution(
                func=self._loss_wrapper,
                bounds=bounds_scaled,
                strategy='rand2bin',
                maxiter=1,  # Adjust as needed; 1 for per-generation iteration
                mutation=mut,
                recombination=rec,
                popsize=curr_pop,
                updating='deferred',
                callback=self.callback_fn,
                tol=0.001,
                polish=False,
                disp=False,
                init=init_population,
                workers=-1
            )
            
            init_population = result.population
            # timestamp = time.strftime("%Y%m%d_%H%M%S")
            # with open(f"population_{timestamp}.pkl", 'wb') as f:
            #     pickle.dump(final_pop, f)

        best_unscaled = self.unscale_params(result.x, self.bounds)
        best_params = copy.deepcopy(self.base_params)
        best_params = self.vector_to_params(best_params, best_unscaled)
        plt.ioff()
        plt.show()
        return result, best_params
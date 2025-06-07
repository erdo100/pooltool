import matplotlib
matplotlib.use('TkAgg')  # Ensure consistent backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode by default
import numpy as np
import copy
import pickle
import time
from tkinter import filedialog
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from loss_funcs import evaluate_loss
from billiardenv import BilliardEnv
from helper_funcs import get_ball_positions

class DEOptimizer:
    def __init__(self, shot_actual, params, balls_xy_ini, ball_cols, maxiter, selected_params=None,
                 popsize=50, mutation=(0.5, 1.5), recombination=0.9, 
                 strategy='best1bin', polish=False):

        self.sim_env = BilliardEnv()
        self.shot_actual = shot_actual
        
        self.sim_env.balls_xy_ini = balls_xy_ini
        self.sim_env.ball_cols = ball_cols

        self.base_params = copy.deepcopy(params)
        
        # If selected_params is provided, only optimize those parameters
        if selected_params is None:
            self.selected_params = list(self.base_params.limits.keys())
        else:
            self.selected_params = selected_params
              # Create bounds only for selected parameters
        
        self.bounds = []
        for key in self.selected_params:
            if key == "shot_phi":
                # Modify the limit for shot_phi as needed, e.g.:
                self.bounds.append((self.base_params.value[key]-6, self.base_params.value[key]+6))  # Example: set to [-pi, pi]
            else:
                self.bounds.append(self.base_params.limits[key])

        self.maxiter = maxiter
        self.pop_size = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.strategy = strategy
        self.polish = polish
        self.parameter_history = []
        self.loss_history = []
        self.opt_fig = None  # Initialize to None to prevent accidental figure creation


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
    def vector_to_params(params, x_vec, selected_params):
        for key, x in zip(selected_params, x_vec):
            params.value[key] = x
        return params
        
    def _loss_wrapper(self, scaled_vec):
        x = self.unscale_params(scaled_vec, self.bounds)

        current_params = copy.deepcopy(self.base_params)
        current_params = self.vector_to_params(current_params, x, self.selected_params)
        
        self.sim_env.prepare_new_shot(current_params)
        
        self.sim_env.simulate_shot()

        loss = evaluate_loss(self.sim_env, self.shot_actual, method="distance")

        return loss["total"]

    def plot_convergence(self, convergence):
        # Format each parameter in best_params to 5 decimal places
        formatted_params = ", ".join(f"{param:.5f}" for param in self.parameter_history[-1])

        it = len(self.loss_history)
        if it == 1:
        # Print the iteration, formatted parameters, and convergence
            print(f"Iteration {it} - Loss: {self.loss_history[-1]:.10f} - Best Params: [{formatted_params}], Convergence: {convergence:.5f}")
        elif self.loss_history[-1] < self.loss_history[-2]:
            print(f"Iteration {it} - Loss: {self.loss_history[-1]:.10f} - Best Params: [{formatted_params}], Convergence: {convergence:.5f}")

        # Create a separate optimization figure if it doesn't exist
        # if len(self.parameter_history) == 1: #not hasattr(self, 'opt_fig') or self.opt_fig is None:
        #     print("Creating new optimization figure...")
        #     # Store current interactive state and figure
        #     was_interactive = plt.isinteractive()
        #     current_fig = plt.gcf()
            
        #     # Temporarily enable interactive mode for figure creation
        #     plt.ion()
        #     self.opt_fig = plt.figure(figsize=(12, 8))
        #     self.opt_fig_created = True  # Flag to track figure creation
        #     self.opt_fig.canvas.manager.set_window_title('Optimization Progress')
            
        #     # Position the window to avoid overlap with main GUI
        #     mngr = self.opt_fig.canvas.manager
        #     if hasattr(mngr, 'window'):
        #         if hasattr(mngr.window, 'wm_geometry'):
        #             mngr.window.wm_geometry("+100+100")  # Position window
        #         # Prevent window from stealing focus
        #         if hasattr(mngr.window, 'attributes'):
        #             mngr.window.attributes('-topmost', False)
        #             mngr.window.focus_set = lambda: None  # Disable focus stealing
            
        #     # Restore previous interactive state and figure
        #     if not was_interactive:
        #         plt.ioff()
        #     plt.figure(current_fig.number)
        
        # # Update the optimization figure without changing focus
        # current_fig = plt.gcf()
        
        # # Switch to optimization figure temporarily
        # plt.figure(self.opt_fig.number)
        # plt.clf()
        
        # # Number of parameters + 1 (for the loss plot)
        # num_params = len(self.parameter_history[-1])
        # param_names = self.selected_params
        # rows = int(np.ceil(np.sqrt(num_params + 1)))
        # cols = int(np.ceil((num_params + 1) / rows))
        
        # # Plot loss history
        # plt.subplot(rows, cols, 1)
        # plt.plot(self.loss_history, 'b-')
        # plt.xlabel('Generation')
        # plt.ylabel('Loss')
        # plt.title('Loss Function History')
        # plt.grid(True)
        # print(f"Current Loss: {self.loss_history[-1]:.10f}")

        # for i, name in enumerate(self.selected_params):
        #     plt.subplot(rows, cols, i + 2)
        #     param_values = [p[i] for p in self.parameter_history]
        #     plt.plot(param_values, 'r-')
        #     plt.title(name)
        #     low, high = self.bounds[i]
        #     plt.ylim(low, high)  # Set y-axis limits to parameter bounds
        #     plt.grid(True)
        
        # plt.tight_layout()
        
        # # Update the figure without stealing focus
        # self.opt_fig.canvas.draw_idle()
        # self.opt_fig.canvas.flush_events()
        
        # # Return to the original figure
        # if current_fig != self.opt_fig and plt.fignum_exists(current_fig.number):
        #     plt.figure(current_fig.number)

    def callback_fn(self, xk, convergence):
        loss = self._loss_wrapper(xk)
        unscaled = self.unscale_params(xk, self.bounds)
        self.parameter_history.append(unscaled.copy())
        self.loss_history.append(loss)
        self.plot_convergence(convergence)

    def run_optimization(self):

        init_population = None

        # file_path = False # filedialog.askopenfilename(title="Load initial population", filetypes=[("Pickle", "*.pkl")])
        file_path = None
        if file_path:
            with open(file_path, 'rb') as f:
                init_population = pickle.load(f)
            print("Initial population loaded from file.")
        else:
            print("Creating new initial population.")
            # Your custom candidate - only for selected parameters
            my_candidate = np.array([self.base_params.value[key] for key in self.selected_params])
            my_candidate_scaled = self.scale_params(my_candidate, self.bounds)
            # my_candidate_scaled = my_candidate_scaled.reshape((1, len(self.bounds)))

            # random population
            sampler = qmc.LatinHypercube(d=len(self.bounds))
            random_samples = sampler.random(n=self.pop_size*len(self.bounds) - 1)  # reserve 1 spot for your candidate

            # Insert your candidate
            init_population = np.vstack([my_candidate_scaled, random_samples])

        result = None

        bounds_scaled = [(0, 1)] * len(self.bounds)
        result = differential_evolution(
            func=self._loss_wrapper,
            bounds=bounds_scaled,
            strategy=self.strategy,
            maxiter=self.maxiter,  # Adjust as needed; 1 for per-generation iteration
            mutation=self.mutation,
            recombination=self.recombination,
            popsize=self.pop_size,  # Use first element for population size
            updating='deferred',
            callback=self.callback_fn,
            tol=0.0001,
            polish=self.polish,
            disp=False,
            init=init_population,
            workers=-1
        )

        # for gen in range(self.maxiter):
        #     curr_pop = int(self.pop_start + (self.pop_end - self.pop_start) * gen / self.maxiter)
        #     mut = self.mut_start + (self.mut_end - self.mut_start) * (gen / self.maxiter)
        #     rec = self.rec_start + (self.rec_end - self.rec_start) * (gen / self.maxiter)
        #     print(f"Generation {gen + 1}/{self.maxiter} | pop={curr_pop} mut={mut:.3f} rec={rec:.3f}")
        #     bounds_scaled = [(0, 1)] * len(self.bounds)

        #     result = differential_evolution(
        #         func=self._loss_wrapper,
        #         bounds=bounds_scaled,
        #         strategy='rand2bin',
        #         maxiter=1000,  # Adjust as needed; 1 for per-generation iteration
        #         mutation=mut,
        #         recombination=rec,
        #         popsize=curr_pop,
        #         updating='deferred',
        #         callback=self.callback_fn,
        #         tol=0.01,
        #         polish=False,
        #         disp=False,
        #         init=init_population,
        #         workers=-1
        #     )        #     init_population = result.population
            # timestamp = time.strftime("%Y%m%d_%H%M%S")
            # with open(f"population_{timestamp}.pkl", 'wb') as f:
            #     pickle.dump(final_pop, f)
        
        best_unscaled = self.unscale_params(result.x, self.bounds)
        # Use shallow copy for final result to reduce memory usage
        best_params = copy.copy(self.base_params)
        best_params.value = copy.copy(self.base_params.value)
        best_params = self.vector_to_params(best_params, best_unscaled, self.selected_params)
        
        # Update the optimization figure with final results and clean up memory leak
        if hasattr(self, 'opt_fig') and self.opt_fig is not None:
            # Update the title to show optimization is complete
            plt.figure(self.opt_fig.number)
            plt.suptitle('Optimization Completed!', fontsize=14, fontweight='bold')
            self.opt_fig.canvas.draw_idle()
            # Properly close the specific figure that was causing memory leaks
            plt.close(self.opt_fig)
            self.opt_fig = None  # Set to None to prevent reuse
            plt.ioff()  # Ensure interactive mode is off
        
        print("Optimization completed!")
        return result, best_params

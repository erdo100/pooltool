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
        """
        Differential Evolution optimizer with advanced control and history plotting.

        Args:
            SA: Shot analysis data containing 'Shot' and 'Data'.
            params: base Parameters instance with value and limits.
            maxiter: total number of generations.
            popsize: tuple (start, end) population sizes or single int for static.
            mutation: tuple (start, end) of mutation factor.
            recombination: tuple (start, end) of recombination rate.
            strategy: DE strategy.
            polish: whether to polish final result.
            workers: number of parallel workers; defaults to cpu_count().
        """
        self.SA = SA
        # prepare simulation environment
        self.sim_env = BilliardEnv()

        self.base_params = copy.deepcopy(params)
        # dynamically retrieve bounds from params.limits in value order
        self.bounds = [params.limits[key] for key in params.limits.keys()]
        self.maxiter = maxiter
        self.pop_start, self.pop_end = (popsize, popsize) if isinstance(popsize, int) else popsize
        self.mut_start, self.mut_end = mutation
        self.rec_start, self.rec_end = recombination
        self.strategy = strategy
        self.polish = polish
        self.workers = workers or mp.cpu_count()

        # history tracking
        self.parameter_history = []
        self.loss_history = []

    def scale_params(self, params_np):
        scaled = np.zeros_like(params_np)
        for i, (low, high) in enumerate(self.bounds):
            scaled[i] = (params_np[i] - low) / (high - low)
        return scaled

    def unscale_params(self, scaled_params):
        original = np.zeros_like(scaled_params)
        for i, (low, high) in enumerate(self.bounds):
            original[i] = scaled_params[i] * (high - low) + low
        return original

    def _vector_to_params(self, x_vec):
        for key, x in zip(self.base_params.limits.keys(), x_vec):
            self.base_params.value[key] = x
        return self.base_params


    def _loss_wrapper(self, scaled_vec):
        # start timing
        # unscale and evaluate loss
        x = self.unscale_params(scaled_vec)
        params = self._vector_to_params(x)
        shot_id = params.value['shot_id']
        shot_actual = self.SA['Shot'][shot_id]
        b1b2b3_col = self.SA['Data']['B1B2B3'][shot_id]
        ball_xy_ini, ball_cols, _ = get_ball_positions(shot_actual, b1b2b3_col)
        self.sim_env.balls_xy_ini = ball_xy_ini
        self.sim_env.ball_cols = ball_cols
        self.sim_env.prepare_new_shot(params)

        # start_time = time.time()
        self.sim_env.simulate_shot()
        # # elapsed time
        # elapsed_time = time.time() - start_time
        # print(f"    Elapsed time: {elapsed_time:.8f} seconds")
        loss = evaluate_loss(self.sim_env, shot_actual)
        loss_val = sum(np.sum(loss['ball'][i]['total']) for i in range(len(loss['ball'])))

        return loss_val

    def callback(self, xk, convergence):
        loss = self._loss_wrapper(xk)
    # if not self.loss_history or loss < min(self.loss_history):
        unscaled = self.unscale_params(xk)
        self.parameter_history.append(unscaled.copy())
        self.loss_history.append(loss)
        fmt = ", ".join(f"{v:.5f}" for v in unscaled)
        print(f"Gen {len(self.loss_history)} | Best: [{fmt}]  Loss: {loss:.5f}  Conv: {convergence:.5f}")
            # n = len(self.bounds)
            # total = n + 1
            # rows = int(np.ceil(np.sqrt(total)))
            # cols = int(np.ceil(total / rows))
            # plt.figure(1)
            # plt.clf()
            # # loss history
            # plt.subplot(rows, cols, 1)
            # plt.plot(self.loss_history)
            # plt.title('Loss History')
            # plt.xlabel('Generation')
            # plt.grid(True)
            # # parameter histories
            # for i in range(n):
            #     plt.subplot(rows, cols, i + 2)
            #     vals = [p[i] for p in self.parameter_history]
            #     low, high = self.bounds[i]
            #     plt.plot(vals)
            #     plt.title(list(self.base_params.limits.keys())[i])
            #     plt.ylim(low, high)
            #     plt.grid(True)
            # plt.tight_layout()
            # plt.pause(0.01)

    def run_optimization(self):
        # default random population
        init = 'random'
        print("Defaulting to random initial population.")
        # optional load
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
            # scaled bounds always [0,1]
            bounds_scaled = [(0, 1)] * len(self.bounds)

            result = differential_evolution(
                func=self._loss_wrapper,
                bounds=bounds_scaled,
                strategy='rand2bin',
                maxiter=1000,
                mutation=mut,
                recombination=rec,
                popsize=curr_pop,
                updating='deferred',
                callback=self.callback,
                tol=0.001,
                polish=False,
                workers=self.workers,
                disp=True,
                init=final_pop,
            )
            
            print(f"Result: {result}")
            final_pop = result.population
            # save population
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            with open(f"population_{timestamp}.pkl", 'wb') as f:
                pickle.dump(final_pop, f)

        # finalize results
        best_unscaled = self.unscale_params(result.x)
        best_params = self._vector_to_params(best_unscaled)
        plt.ioff()
        plt.show()
        return result, best_params

# Usage example:
# optimizer = DEOptimizer(SA, params,
#                         maxiter=1000,
#                         popsize=(200,50),
#                         mutation=(1.0,0.5),
#                         recombination=(0.9,0.5))
# result, best = optimizer.run_optimization()

import optuna
import numpy as np
from optuna.samplers import RandomSampler, CmaEsSampler, NSGAIISampler
from optuna.storages import InMemoryStorage
import multiprocessing as mp
from functools import partial
import copy
from loss_funcs import evaluate_loss
from billiardenv import BilliardEnv
from helper_funcs import get_ball_positions
from parameters import Parameters
from joblib import Parallel, delayed
import time


class Optimizer:
    def __init__(self, SA, params):
        self.SA = SA
        self.base_params = params
        self.storage = InMemoryStorage()
        
    def _create_objective(self, method):
        """Create objective function with method-specific parameter space"""
        def objective(trial):
            # Create independent copy of parameters for this trial
            params = copy.deepcopy(self.base_params)
            shot_id = params.value['shot_id']
            shot_actual = self.SA["Shot"][shot_id]
            b1b2b3_col = self.SA["Data"]["B1B2B3"][shot_id]
            ball_xy_ini, ball_cols, _ = get_ball_positions(shot_actual, b1b2b3_col)

            # Suggest parameters based on optimization method
            if method == "Random":
                params.value['shot_a'] = trial.suggest_float('shot_a', -0.5, 0.5)
                params.value['shot_b'] = trial.suggest_float('shot_b', -0.5, 0.5)
                params.value['shot_phi'] = trial.suggest_float('shot_phi', -180, 180)
                params.value['shot_v'] = trial.suggest_float('shot_v', 1.0, 7.0)
                params.value['shot_theta'] = trial.suggest_float('shot_theta', 0.0, 20.0)

                params.value['physics_ballball_a'] = trial.suggest_float('physics_ballball_a', 0.0, 0.3)
                params.value['physics_u_slide'] = trial.suggest_float('physics_u_slide', 0.0, 0.3)
                params.value['physics_u_roll'] = trial.suggest_float('physics_u_roll', 0.0, 0.015)
                params.value['physics_u_sp_prop'] = trial.suggest_float('physics_u_sp_prop', 0.0, 1.0)
                params.value['physics_e_ballball'] = trial.suggest_float('physics_e_ballball', 0.5, 1.0)
                params.value['physics_e_cushion'] = trial.suggest_float('physics_e_cushion', 0.5, 1.0)
                params.value['physics_f_cushion'] = trial.suggest_float('physics_f_cushion', 0.1, 0.4)

            elif method == "Evolutionary":
                params.value['shot_phi'] = trial.suggest_float('phi', -180, 180)
            elif method == "Genetic":
                params.value['shot_phi'] = trial.suggest_float('phi', -180, 180)

            # Run simulation
            sim_env = BilliardEnv()
            sim_env.balls_xy_ini = ball_xy_ini
            sim_env.ball_cols = ball_cols
            sim_env.prepare_new_shot(params)
            sim_env.simulate_shot()

            # Calculate loss
            loss = evaluate_loss(sim_env, shot_actual)
            total_loss = sum(np.sum(loss["ball"][balli]["total"]) for balli in range(3))
            
            return total_loss
        return objective

    def run_optimization(self, method, totalruns):
        """Run optimization with specified method"""
        # Configure sampler
        if method == "Random":
            sampler = RandomSampler()
        elif method == "Evolutionary":
            sampler = CmaEsSampler()
        elif method == "Genetic":
            sampler = NSGAIISampler()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create study with in-memory storage
        study = optuna.create_study(
            direction="minimize",
            storage=self.storage,
            sampler=sampler
        )
        print(f"Running {method} optimization with {totalruns} trials.")
        print(f"Study created with storage: {self.storage}")
        
        # start timing
        start_time = time.time()

        # Simple sequential execution
        for _ in range(totalruns):
            trial = study.ask()
            value = self._create_objective(method)(trial)
            study.tell(trial, value)
            
            # Update GUI progress
            if hasattr(self, 'gui'):
                self.gui.progress_bar.step(1)
                self.gui.root.update_idletasks()
    

        print(f"Study finished")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time} seconds")
        
        return study



import pickle
import time
import numpy as np
from scipy.interpolate import interp1d
from tkinter import filedialog
from billiardenv import BilliardEnv
import pooltool as pt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tkinter import Button, DISABLED, NORMAL
import os
import numpy as np
from scipy.interpolate import interp1d
import pickle
from tkinter import filedialog

import optuna
import copy
import threading
import multiprocessing as mp

from optuna.storages import RDBStorage
import sqlite3
from pathlib import Path
from sqlalchemy.pool import NullPool
import logging
from logging_config import configure_logging
configure_logging()  # First line after imports

def open_shotfile(file_name=None):
    if file_name is None:
        file_name = filedialog.askopenfilename()

    with open(file_name, "rb") as f:
        SA = pickle.load(f)
    
    return SA

# Function to save parameters
def save_parameters(params):
    filename = "Shot_" + str(params.value["shot_id"]) + "_parameters_" + time.strftime("%Y%m%d_%H%M%S") + ".pkl"
    file_path = filedialog.asksaveasfilename(initialfile=filename, defaultextension='.pkl', filetypes=[('Pickle files', '*.pkl')])
    if file_path:
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)
        print(f"Parameters saved to {file_path}")


def load_parameters(slider_frame, update_plot, **sliders):
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if file_path:
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
            print(f"Parameters loaded from {file_path}:")
        for key, slider in sliders.items():
            print(f"{key}: {params.value[key]}")
            if key in params.value:
                print(f"Setting {key} to {params.value[key]}")
                slider.set(params.value[key])

        update_plot()   

# Function to save parameters
def save_system(update_plot):
    system = update_plot()
    
    filename = "system_" + time.strftime("%Y%m%d_%H%M%S") + ".msgpack"
    
    file_path = filedialog.asksaveasfilename(initialfile=filename ,defaultextension='.pkl', filetypes=[('Pickle files', '*.pkl')])
    if file_path:
        with open(file_path, 'wb') as f:
            pickle.dump(system, f)
        print(f"System saved to {file_path}")
        system.save("Ball_dislocation_bug.msgpack")

def initial_shot_direction(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    phi = np.arctan2(dy, dx)
    phi = np.degrees(phi)

    return phi[0]

def get_ball_ids(b1b2b3):
    # in b1b2b3 a string with "wyr" is defined. the location where the first is the cue ball.
    # initiate empty lists for ball_ids and ball_cols
    ball_ids = [None] * 3
    ball_cols = [None] * 3
    
    for i in range(3):
        if b1b2b3[i] == 'W':
            ball_ids[i]= 0
            ball_cols[i]= 'white'
        elif b1b2b3[i] == 'Y':
            ball_ids[i]= 1
            ball_cols[i]= 'yellow'
        elif b1b2b3[i] == 'R':
            ball_ids[i]= 2
            ball_cols[i]= 'red'
    
    return ball_ids, ball_cols



def get_ball_positions(shot_actual, b1b2b3):

    ball_ids, ball_cols = get_ball_ids(b1b2b3)

    balls_xy_ini = {}
    balls_xy_ini[0] = (shot_actual['Ball'][0]["x"][0], shot_actual['Ball'][0]["y"][0])
    balls_xy_ini[1] = (shot_actual['Ball'][1]["x"][0], shot_actual['Ball'][1]["y"][0])
    balls_xy_ini[2] = (shot_actual['Ball'][2]["x"][0], shot_actual['Ball'][2]["y"][0])

    # Calculate initial shot direction
    cueball_x = shot_actual['Ball'][ball_ids[0]]["x"]
    cueball_y = shot_actual['Ball'][ball_ids[0]]["y"]
    cueball_phi = initial_shot_direction(cueball_x, cueball_y)

    return balls_xy_ini, ball_cols, cueball_phi

def custom_tanh(t, t0, L, U, S):
    """
    Custom tanh function with parametrized lower and upper values, transition point, and slope.

    Parameters:
        t (float or array): Input values.
        t0 (float): Transition point.
        L (float): Lower value.
        U (float): Upper value.
        S (float): Slope at the transition point.

    Returns:
        float or array: Output values of the custom tanh function.
    """
    # Step 1: Calculate the midpoint between L and U
    midpoint = (U + L) / 2
    # Step 2: Calculate the range between U and L
    range_value = (U - L) / 2
    # Step 3: Scale and flip and Shift the input t by the transition point t0
    t = np.array(t)
    scaled_t = -S*(t - t0)
    # Step 6: Apply the tanh function to the scaled input
    tanh_value = np.tanh(scaled_t)
    # Step 7: Scale the tanh output by the range and shift by the midpoint
    result = range_value * tanh_value + midpoint

    return result



def calculate_distance_loss(act_t, act_x, act_y, sim_t, sim_x, sim_y):

    # calculate loss based on which duration is longer
    tmax = max(act_t[-1], sim_t[-1])
    tmin = min(act_t[0], sim_t[0])

    tloss = np.linspace(tmin, tmax, 200)

    sim_x_interp = interpolate_shot(sim_x, sim_t, tloss)
    sim_y_interp = interpolate_shot(sim_y, sim_t, tloss)
    act_x_interp = interpolate_shot(act_x, act_t, tloss)
    act_y_interp = interpolate_shot(act_y, act_t, tloss)

    # calculate loss
    loss_dist = np.sqrt((act_x_interp - sim_x_interp) ** 2 + (act_y_interp - sim_y_interp) ** 2)/2.84/len(tloss)
    loss = np.sum(loss_dist)

    return loss



def calculate_angle_loss(xsim, ysim, xact, yact):
    # Calculate the angle between the two vectors
    dx_sim = xsim[-1]- xsim[0]
    dy_sim = ysim[-1]- ysim[0]
    dx_act = xact[-1] - xact[0]
    dy_act = yact[-1] - yact[0]

    # Calculate the angle using the dot product formula
    dot_product = dx_sim * dx_act + dy_sim * dy_act
    norm_sim = np.sqrt(dx_sim**2 + dy_sim**2)
    norm_actual = np.sqrt(dx_act**2 + dy_act**2)

    cos_angle = dot_product / (norm_sim * norm_actual + 1e-10)  # Avoid division by zero

    # Ensure the angle is in the range [0, 1]
    angle = np.abs(np.arccos(np.clip(cos_angle, -1.0, 1.0))) / np.pi

    return angle

def evaluate_loss(sim_env, shot_actual, method="combined_straighten_route"):
    
    sim_t, white_rvw, yellow_rvw, red_rvw = sim_env.get_ball_routes()
    sim_hit_all = sim_env.get_events(sim_env)
    
    balls_rvw = [white_rvw, yellow_rvw, red_rvw]
    
    losses = {}
    losses["ball"] = [{} for _ in range(3)]

    # loop over all balls
    for balli in range(3):
        losses["ball"][balli]["time"] = []
        losses["ball"][balli]["hit"] = []
        losses["ball"][balli]["angle"] = []
        losses["ball"][balli]["distance"] = []
        losses["ball"][balli]["total"] = []
        
        sim_hit = sim_hit_all[balli]
        act_hit = shot_actual["hit"][balli]
    
        sim_x = balls_rvw[balli][:, 0, 0]
        sim_y = balls_rvw[balli][:, 0, 1]

        act_t = shot_actual['Ball'][balli]["t"]
        act_x = shot_actual['Ball'][balli]["x"]
        act_y = shot_actual['Ball'][balli]["y"]

        # arrayify
        act_t = np.array(act_t)
        act_x = np.array(act_x)
        act_y = np.array(act_y)
        sim_t = np.array(sim_t)
        sim_x = np.array(sim_x)
        sim_y = np.array(sim_y) 

        correct_hit = True

        # run through events based on actual data and simulation data
        for ei in range(max(len(act_hit["with"]), len(sim_hit["with"]))):
            if ei < len(act_hit["with"]) and act_hit["with"][ei] == "-":
                # assign the loss
                losses["ball"][balli]["time"].append(0)
                losses["ball"][balli]["hit"].append(0)
                losses["ball"][balli]["angle"].append(0)
                losses["ball"][balli]["distance"].append(0)
                losses["ball"][balli]["total"].append(0)
                continue

            # find the time index of the current event in actual data
            if ei <= len(act_hit["with"])-1 and ei <= len(sim_hit["with"])-1:            
                # check if hit partner is same
                if act_hit["with"][ei] == sim_hit["with"][ei] and correct_hit == True:                    
                    correct_hit = True
                    loss_hit = 0.0
                

                    act_event_time = act_hit["t"][ei]
                    sim_event_time = sim_hit["t"][ei]
                    current_act_event_time_index = np.where(act_t >= act_event_time)[0][0]
                    current_sim_event_time_index = np.where(sim_t >= sim_event_time)[0][0]

                    current_act_time = act_t[current_act_event_time_index]
                    current_sim_time = sim_t[current_sim_event_time_index]

                    current_time = np.max([current_act_time, current_sim_time])
                    
                    # find the time of the next event in actual data
                    # either as next event or last time step
                    if ei < len(act_hit["with"])-1:
                        event_time = act_hit["t"][ei + 1]
                    else:
                        event_time = act_t[-1]
                        correct_hit = False
                    next_act_event_time_index = np.where(act_t >= event_time)[0][0]

                    # find the time of the next event in simulation data
                    if ei < len(sim_hit["with"])-1:
                        event_time = sim_hit["t"][ei + 1]
                    else:
                        event_time = sim_t[-1]
                        correct_hit = False
                    next_sim_event_time_index = np.where(sim_t >= event_time)[0][0]


                    if len(sim_x) >= 2 and len(act_x) >= 2 and current_sim_event_time_index < next_sim_event_time_index and current_act_event_time_index < next_act_event_time_index:
                        # calculate the angle between act and sim data
                        loss_angle = calculate_angle_loss(sim_x[current_sim_event_time_index:next_sim_event_time_index], 
                                                        sim_y[current_sim_event_time_index:next_sim_event_time_index], 
                                                        act_x[current_act_event_time_index:next_act_event_time_index], 
                                                        act_y[current_act_event_time_index:next_act_event_time_index])
                        
                        loss_distance = calculate_distance_loss(sim_t[current_sim_event_time_index:next_sim_event_time_index], 
                                                        sim_x[current_sim_event_time_index:next_sim_event_time_index], 
                                                        sim_y[current_sim_event_time_index:next_sim_event_time_index], 
                                                        act_t[current_act_event_time_index:next_act_event_time_index], 
                                                        act_x[current_act_event_time_index:next_act_event_time_index], 
                                                        act_y[current_act_event_time_index:next_act_event_time_index])
                    else:
                        loss_angle = 1.0
                        loss_distance = 1.0
                    
                    # print(f"Ball {balli} - Event {ei}: Time: {current_time:.6f}, Loss Hit: {loss_hit:.6f}, Loss Angle: {loss_angle:.6f}, Loss Distance: {loss_distance:.6f}")
                else:
                    correct_hit = False

            if correct_hit == False:
                loss_hit = 1.0
                loss_angle = 1.0
                loss_distance = 1.0

                # find the current time of the event in actual data
                if ei <= len(act_hit["with"])-1:
                    event_time = act_hit["t"][ei]
                    event_time_index = np.where(act_t >= act_event_time)[0][0]
                    current_time = act_t[event_time_index]
                elif ei <= len(sim_hit["with"])-1:
                    event_time = sim_hit["t"][ei]
                    event_time_index = np.where(sim_t >= event_time)[0][0]
                    current_time = sim_t[event_time_index]



            # assign the loss
            losses["ball"][balli]["time"].append(current_time)
            losses["ball"][balli]["hit"].append(loss_hit)
            losses["ball"][balli]["angle"].append(loss_angle)
            losses["ball"][balli]["distance"].append(loss_distance)
            losses["ball"][balli]["total"].append(loss_hit + loss_angle + loss_distance)
    
    
    return losses



def interpolate_shot(simulated, tsim, actual_times):
    interp_func = interp1d(
        tsim,
        simulated,
        kind="linear",
        bounds_error=False,
        fill_value=(simulated[-1]),
    )
    interpolated = interp_func(actual_times)
    return interpolated



# calculate absolute velocity
def abs_velocity(t, x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    v = np.sqrt(dx ** 2 + dy ** 2) / dt
    # add a zero to the beginning of the array to make it the same length as x and y
    v = np.insert(v, 0, 0)
    return v


def run_study(SA, params):
    print('Running study with parameters:')
    # Retrieve current shot and slider values
    shot_id = params.value['shot_id']
    shot_actual = SA['Shot'][shot_id]
    b1b2b3_col = SA["Data"]["B1B2B3"][shot_id]
    ball_xy_ini, ball_cols, cue_phi = get_ball_positions(shot_actual, b1b2b3_col)

    # set new parameters and simulate shot
    sim_env = BilliardEnv()
    sim_env.balls_xy_ini = ball_xy_ini
    sim_env.ball_cols = ball_cols

    steps = 5000
    total_loss = np.zeros(steps)
    phi_range = np.linspace(-180, 180, steps)
    
    
    for i, phi in enumerate(phi_range):
        print(f"Running simulation {i+1} of {len(phi_range)}")
        params.value['shot_phi'] = phi
        sim_env.prepare_new_shot(params)
        sim_env.simulate_shot()

        loss = evaluate_loss(sim_env, shot_actual)

        for balli in range(3):
            total_loss[i] += np.sum(loss["ball"][balli]["total"]) 
    
   
    # create a figure and axis
    fig, ax = plt.subplots()
    # ax.plot(phi_range, hitpointx_detrended)
    ax.plot(phi_range, total_loss)
    ax.set_xlabel('phi')
    ax.set_ylabel('Loss')
    plt.title('Loss function calculated distance from straightened shots')
    plt.grid()
    plt.show()
    # save hitpointx to a file



def run_optuna(self, SA, params):
    print('Starting optimization...')
    root = self.root  # Get Tkinter root reference

    shot_id = params.value['shot_id']
    shot_actual = SA['Shot'][shot_id]
    b1b2b3_col = SA["Data"]["B1B2B3"][shot_id]
    ball_xy_ini, ball_cols, cue_phi = get_ball_positions(shot_actual, b1b2b3_col)

    # Store references needed for GUI updates
    sliders = self.sliders
    sim_env = self.sim_env

    def objective(trial):
        trial_params = copy.deepcopy(params)
        for param_name in trial_params.limits:
            low, high = trial_params.limits[param_name]
            trial_params.value[param_name] = trial.suggest_float(param_name, low, high)
        
        # Reuse existing simulation environment
        sim_env.balls_xy_ini = ball_xy_ini
        sim_env.ball_cols = ball_cols
        sim_env.prepare_new_shot(trial_params)
        sim_env.simulate_shot()
        loss = evaluate_loss(sim_env, shot_actual)
        return sum(np.sum(loss["ball"][i]["total"]) for i in range(3))

    def optimize_thread():
        study = optuna.create_study(direction='minimize')
        
        try:
            study.optimize(objective, n_trials=20000, callbacks=[update_progress])
        except Exception as e:
            root.after(0, lambda: print(f"Optimization failed: {e}"))
            return
        
        # Update GUI with best parameters
        root.after(0, lambda: finalize_optimization(study))

    def update_progress(study, trial):
        if trial.number % 5 == 0:
            root.after(0, lambda: [
                print(f"Trial {trial.number}: Loss {trial.value:.3f}")
            ])

    def finalize_optimization(study):
        # Update parameters and sliders
        for param_name, value in study.best_params.items():
            # Update parameter value
            params.value[param_name] = value
            
            # Update slider if it exists
            if param_name in sliders:
                slider = sliders[param_name]
                
                # Disable callback temporarily
                original_command = slider.cget('command')
                slider.config(command=None)
                
                # Update slider value
                slider.set(value)
                
                # Restore callback
                slider.config(command=original_command)
        
        # Force plot refresh
        self.update_plot(is_optimization_update=True)
        
        # Print results
        print("\n=== Optimization Complete ===")
        print(f"Best loss: {study.best_value:.4f}")
        print("Best parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v:.4f}")

        # Show optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()


    # Start thread
    threading.Thread(target=optimize_thread, daemon=True).start()




class Optimizer:

    def __init__(self, params, shot_data, initial_state):
        self.db_path = Path(__file__).parent / "optimization.db"
        self.storage_url = f"sqlite:///{self.db_path}"
        self._init_storage()
        print(f"Database location: {self.db_path.absolute()}")

    def _init_storage(self):
        """Initialize database with explicit path"""
        try:
            if self.db_path.exists():
                self.db_path.unlink()
                print("Removed existing database")
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

        storage = RDBStorage(
            url=self.storage_url,
            engine_kwargs={
                "poolclass": NullPool,
                "connect_args": {
                    "timeout": 60,
                    "check_same_thread": False
                }
            }
        )

    def _objective(self, trial):
        try:
            # Create fresh environment for each trial
            local_env = BilliardEnv()
            local_env.set_initial_state(self.initial_state)
            
            # Create parameter copy
            trial_params = copy.deepcopy(self.params)
            
            # Suggest parameters
            for param_name in trial_params.limits:
                low, high = trial_params.limits[param_name]
                trial_params.value[param_name] = trial.suggest_float(param_name, low, high)

            # Get shot data
            shot_id = trial_params.value['shot_id']
            shot_actual = self.shot_data['Shot'][shot_id]
            b1b2b3_col = self.shot_data["Data"]["B1B2B3"][shot_id]
            ball_xy_ini, ball_cols, _ = get_ball_positions(shot_actual, b1b2b3_col)

            # Configure environment
            local_env.balls_xy_ini = ball_xy_ini
            local_env.ball_cols = ball_cols
            local_env.prepare_new_shot(trial_params)
            local_env.simulate_shot()

            # Calculate loss
            loss = evaluate_loss(local_env, shot_actual)
            return sum(np.sum(loss["ball"][i]["total"]) for i in range(3))
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            raise


class OptimizationManager:
    def __init__(self, params, shot_data, initial_state):
        import os
        from pathlib import Path
        
        # Process management
        self.processes = []
        self.n_workers = max(1, mp.cpu_count() - 1)
        
        # Database configuration
        self.db_path = Path(os.getenv('APPDATA')) / 'PoolTool' / 'optimization.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_url = f"sqlite:///{self.db_path}"
        
        # Study data
        self.params = params
        self.shot_data = shot_data
        self.initial_state = initial_state
        
        self._init_storage()

    def _init_storage(self):
        """Initialize database connection and study"""
        from optuna.storages import RDBStorage
        
        self.storage = RDBStorage(
            url=self.storage_url,
            engine_kwargs={
                "poolclass": NullPool,
                "connect_args": {"timeout": 30, "check_same_thread": False}
            }
        )
        
        # Clean up existing study
        study_name = "main_study"
        try:
            optuna.delete_study(study_name=study_name, storage=self.storage)
        except KeyError:
            pass
            
        # Create new study
        self.study = optuna.create_study(
            direction="minimize",
            storage=self.storage,
            study_name=study_name,
            load_if_exists=False
        )
        print(f"✅ Storage initialized at {self.db_path}")

    def start(self, total_trials=100):
        """Start parallel optimization"""
        trials_per_worker = total_trials // self.n_workers
        remainder = total_trials % self.n_workers

        for i in range(self.n_workers):
            worker_trials = trials_per_worker + (1 if i < remainder else 0)
            if worker_trials <= 0: continue
            
            p = mp.Process(
                target=self._worker_task,
                args=(worker_trials,)
            )
            p.start()
            self.processes.append(p)  # Now works with initialized list

    def _worker_task(self, n_trials):
        import os
        print(f"\n🚀 Worker process started (PID: {os.getpid()})")
        print(f"📂 Database path: {self.db_path}")
        
        try:
            # Test database connection
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1")
            print("✅ Database connection verified")
            
            # Rest of worker code...
            
        except Exception as e:
            print(f"❌ Worker failed: {str(e)}")
            raise
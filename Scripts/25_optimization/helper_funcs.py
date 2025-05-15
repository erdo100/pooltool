import pickle
import time
import numpy as np
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

from pathlib import Path
from sqlalchemy.pool import NullPool

from loss_funcs import evaluate_loss

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
    angle_loss = np.zeros(steps)
    distance_loss = np.zeros(steps)
    phi_range = np.linspace(-180, 180, steps)
        
    for i, phi in enumerate(phi_range):
        print(f"Running simulation {i+1} of {len(phi_range)}")
        params.value['shot_phi'] = phi
        sim_env.prepare_new_shot(params)
        sim_env.simulate_shot()

        loss = evaluate_loss(sim_env, shot_actual)

        for balli in range(3):
            total_loss[i] += np.sum(loss["ball"][balli]["total"])
            angle_loss[i] += np.sum(loss["ball"][balli]["angle"])
            distance_loss[i] += np.sum(loss["ball"][balli]["distance"])

   # create a figure and axis
    fig, ax = plt.subplots()
    # ax.plot(phi_range, hitpointx_detrended)
    ax.plot(phi_range, total_loss)
    ax.plot(phi_range, angle_loss)
    ax.plot(phi_range, distance_loss)
    ax.set_xlabel('phi')
    ax.set_ylabel('Loss')
    plt.title('Loss function')
    plt.grid()
    plt.show()
    # save hitpointx to a file



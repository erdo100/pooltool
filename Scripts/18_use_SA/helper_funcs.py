import pickle
import time
import numpy as np
from scipy.interpolate import interp1d
from tkinter import filedialog
from billiardenv import BilliardEnv
import pooltool as pt
import matplotlib.pyplot as plt


import numpy as np
from scipy.interpolate import interp1d
import pickle
from tkinter import filedialog

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


def loss_func(actual_t, actual_x, actual_y, simulated_t, simulated_x, simulated_y):
    simulated_x_interp = interpolate_simulated_to_actual(simulated_x, simulated_t, actual_t)
    simulated_y_interp = interpolate_simulated_to_actual(simulated_y, simulated_t, actual_t)


    # calculate loss
    loss_dist = np.sqrt((actual_x - simulated_x_interp) ** 2 + (actual_y - simulated_y_interp) ** 2)

    # calculate angle of velocities for actual and simulated for each time step
    # additional data point to maintain length
    actual_vx = np.diff(actual_x) / np.diff(actual_t)
    actual_vx = np.insert(actual_vx, 0, 0)
    actual_vy = np.diff(actual_y) / np.diff(actual_t)
    actual_vy = np.insert(actual_vy, 0, 0)
    simulated_vx = np.diff(simulated_x_interp) / np.diff(actual_t)
    simulated_vx = np.insert(simulated_vx, 0, 0)
    simulated_vy = np.diff(simulated_y_interp) / np.diff(actual_t)
    simulated_vy = np.insert(simulated_vy, 0, 0)
    # use cosine law to calculate angle between actual and simulated velocity
    actual_v = np.sqrt(actual_vx ** 2 + actual_vy ** 2)
    simulated_v = np.sqrt(simulated_vx ** 2 + simulated_vy ** 2)
    
    # avoid division by zero
    eps = 1e-10
    # store index with velocity smaller eps in v0index
    v0actual_index = np.where(actual_v < eps)[0]
    v0simulated_index = np.where(simulated_v < eps)[0]
    v0both = np.intersect1d(v0actual_index, v0simulated_index)

    actual_v[v0actual_index] = eps
    simulated_v[v0simulated_index] = eps

    # calculate cosine of angle between actual and simulated velocity
    cos_angle = (actual_vx * simulated_vx + actual_vy * simulated_vy) / (actual_v * simulated_v)
    # if both balls are moving use the angle between the two velocities
    # else use 180 degrees
    # therefore replace the cosangle values with 180 for v0actual_index and v0simulated_index

    angle = np.arccos(cos_angle)
        
        
    # replace the values for v0actual_index and v0simulated_index with 180 degrees
    angle[v0actual_index] = np.pi
    angle[v0simulated_index] = np.pi
    # replace the values for v0both with 0 degrees
    angle[v0both] = 0

    loss_angle = angle

    # reverse order of time
    weight = custom_tanh(actual_t, 0.5, 1, 100, 1)
    
    loss = (loss_dist + loss_angle*10 + loss_dist*loss_angle)*weight
    
    dt = np.diff(actual_t)
    dt = np.append(dt, dt[-1])

    loss = np.cumsum(loss)

    return loss


def evaluate_loss(sim_env, shot_actual):
    tsim, white_rvw, yellow_rvw, red_rvw = sim_env.get_ball_routes()

    total_loss = 0
    for i, rvw in enumerate([white_rvw, yellow_rvw, red_rvw]):
        simulated_x = rvw[:, 0, 0]
        simulated_y = rvw[:, 0, 1]
        measured_t = shot_actual['Ball'][i]["t"]
        dt = np.diff(measured_t)
        dt = np.append(dt, dt[-1])
        
        measured_x = shot_actual['Ball'][i]["x"]
        measured_y = shot_actual['Ball'][i]["y"]
        loss_trans = loss_func(measured_t, measured_x, measured_y, tsim, simulated_x, simulated_y)
        loss = loss_trans[-1]
        total_loss += loss

    return total_loss


def loss_value(shot_actual, sim_env):

    tsim, white_rvw, yellow_rvw, red_rvw = sim_env.get_ball_routes()

    loss = 0
    for i, ball_rvw in enumerate([white_rvw, yellow_rvw, red_rvw]):
        
        measured_x = shot_actual['Ball'][i]["x"]
        measured_y = shot_actual['Ball'][i]["y"]
        measured_t = shot_actual['Ball'][i]["t"]
        
        # Interpolate simulated trajectory to match measured timestamps
        simx = interpolate_simulated_to_actual(ball_rvw[:, 0, 0], tsim, measured_t)
        simy = interpolate_simulated_to_actual(ball_rvw[:, 0, 1], tsim, measured_t)
        
        # Calculate sum of distances
        loss += np.sum(np.sqrt((measured_x - simx)**2 + (measured_y - simy)**2))
    
    return loss

def interpolate_simulated_to_actual(simulated, tsim, actual_times):
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

    steps = 1440
    loss = np.zeros(steps)
    phi_range = np.linspace(-180, 180, steps)
    
    
    for i, phi in enumerate(phi_range):
        print(f"Running simulation {i+1} of {len(phi_range)}")
        params.value['shot_phi'] = phi
        sim_env.prepare_new_shot(params)
        sim_env.simulate_shot()

        loss[i] = evaluate_loss(sim_env, shot_actual)
            
    
   
    # create a figure and axis
    fig, ax = plt.subplots()
    # ax.plot(phi_range, hitpointx_detrended)
    ax.plot(phi_range, loss)
    ax.set_xlabel('phi')
    ax.set_ylabel('Loss')
    plt.grid()
    plt.show()
    # save hitpointx to a file

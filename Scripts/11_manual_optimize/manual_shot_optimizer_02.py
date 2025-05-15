import os
import pickle
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from threecushion_shot import BilliardEnv
import pooltool as pt
import optuna
from optuna.samplers import TPESampler
import matplotlib.animation as animation

def read_shotfile(file_path):
    # Construct the file path
    print(f"The current folder is: {file_path}")

    # Load shots from the pickle file
    with open(file_path, "rb") as f:
        shots_actual = pickle.load(f)

    # Swap x and y axes
    for shot_actual in shots_actual:
        for ball in shot_actual["balls"].values():
            ball["x"], ball["y"] = np.array(ball["y"]), np.array(ball["x"])

    return shots_actual

def loss_func(actual_x, actual_y, simulated_x, simulated_y):
    distances = np.sqrt((actual_x - simulated_x) ** 2 + (actual_y - simulated_y) ** 2)
    rms = np.sqrt(np.mean(distances ** 2))
    return rms

def interpolate_simulated_to_actual(simulated, tsim, actual_times):
    interp_func_x = interp1d(
        tsim,
        simulated[:, 0],
        kind="linear",
        bounds_error=False,
        fill_value=(simulated[0, 0], simulated[-1, 0]),
    )
    interp_func_y = interp1d(
        tsim,
        simulated[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=(simulated[0, 1], simulated[-1, 1]),
    )
    interpolated_x = interp_func_x(actual_times)
    interpolated_y = interp_func_y(actual_times)
    return interpolated_x, interpolated_y

def plot_settings():
    plt.clf()  # Clear the current figure
    plt.ion()  # Enable interactive mode
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.xlim(0, 1.42)
    plt.ylim(0, 2.84)
    plt.gca().set_aspect("equal", adjustable="box")  # Set the aspect ratio to 1:2
    plt.gca().set_facecolor("lightgray")  # Set the background color to light gray
    grid_size = 2.84 / 8
    plt.xticks(np.arange(0, 1.42 + grid_size, grid_size))
    plt.yticks(np.arange(0, 2.84 + grid_size, grid_size))
    plt.grid(True)

def plot_current_shot(col, actual_x, actual_y, simulated_x, simulated_y):
    colortable = ["white", "yellow", "red"]
    colname = colortable[col - 1]
    plt.plot(actual_x, actual_y, "-", color=colname)
    if col==1:
        line, = plt.plot(simulated_x, simulated_y, "--", color=colname)

    if col==2:
        line, = plt.plot(simulated_x, simulated_y, "--", color=colname)

    if col==3:
        line, = plt.plot(simulated_x, simulated_y, "--", color=colname)

    return 

def set_ball_positions(shot_actual):
    ball_num = 0
    for ball_col, ball_data in shot_actual["balls"].items():
        ball_num += 1

        if ball_num == 1:
            ball1xy = (ball_data["x"][0], ball_data["y"][0])

        elif ball_num == 2:
            ball2xy = (ball_data["x"][0], ball_data["y"][0])

        elif ball_num == 3:
            ball3xy = (ball_data["x"][0], ball_data["y"][0])
    return ball1xy,ball2xy,ball3xy

file_path = r"E:\PYTHON_PROJECTS\POOLTOOL\3cushiontool\Scripts\20221225_2_Match_Ersin_Cemal.pkl"
shots_actual = read_shotfile(file_path)

# Alciatori Ball-Ball hit model parameters
# Friction fit curve u_b = a + b * exp(-c * v_rel) used in David Alciatore's TP A-14
ballball_hit_params = {
    'a': 0.009951,
    'b': 0.108,
    'c': 1.088
}

# Physics parameters
physics_params = {
    'u_slide': 0.2,
    'u_roll': 0.005,
    'u_sp_prop': 10 * 2 / 5 / 9,
    'e_ballball': 0.95,
    'e_cushion': 0.9,
    'f_cushion': 0.15,
    'u_ballball': 0.05 # not relevant for Alciatori Ball-Ball hit model
}

# Shot Parameter
shot_param = {
    'a': 0.0,
    'b': 0.0,
    'phi': -81.0,
    'v': 3.0,
    'theta': 0.0
}

# Set up shot
num_shots = 1  # Change this value to use a different number of shots


# Initialize figure dictionary
fig = {}

# Select the current shot
shot_actual = shots_actual[0]

fig = plt.figure(figsize=(7.1, 14.2))  # Set the figure size to maintain a 1:2 aspect ratio


print("optimizing for Shot: ", shot_actual["shotID"])
ball1xy, ball2xy, ball3xy = set_ball_positions(shot_actual)    

# set the physics parameters
a_ballball, b_ballball, c_ballball = ballball_hit_params['a'], ballball_hit_params['b'], ballball_hit_params['c']
u_slide, u_roll, u_sp_prop, u_ballball, e_ballball, e_cushion, f_cushion = physics_params['u_slide'], physics_params['u_roll'], physics_params['u_sp_prop'], physics_params['u_ballball'], physics_params['e_ballball'], physics_params['e_cushion'], physics_params['f_cushion']
a, b, phi, v, theta = shot_param['a'], shot_param['b'], shot_param['phi'], shot_param['v'], shot_param['theta']

# Create billiard environment
shot_env = BilliardEnv(u_slide, u_roll, u_sp_prop, u_ballball, e_ballball, e_cushion, f_cushion)

# Prepare and simulate shot with best parameters
shot_env.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, phi, v, theta)
point, result, tsim, system = shot_env.simulate_shot(a_ballball, b_ballball, c_ballball)

plot_settings()
total_loss = 0.0
for ball_col, ball_data in shot_actual["balls"].items():
    actual_times = ball_data["t"]
    actual_x = ball_data["x"]
    actual_y = ball_data["y"]
    simulated_x, simulated_y = interpolate_simulated_to_actual(result[ball_col - 1], tsim, actual_times)
    total_loss += loss_func(actual_x, actual_y, simulated_x, simulated_y)
    
    # Plot the actual and simulated shots
    plot_current_shot(ball_col, actual_x, actual_y, simulated_x, simulated_y)

plt.gca().set_title(f"ShotID: {shot_actual['shotID']}, Loss: {total_loss:.2f}\n"
                f"a: {round(a, 2)}\n"
                f"b: {round(b, 2)}\n"
                f"phi: {round(phi, 2)}\n"
                f"v: {round(v, 2)}\n"
                f"theta: {round(theta, 2)}")
plt.draw()  # Update the plot
plt.pause(0.001)  # Pause to allow the plot to be updated

output_dir = r"E:\PYTHON_PROJECTS\pooltool\11_manual_optimizer"
os.makedirs(output_dir, exist_ok=True)
# Save the plot as an image with date-time stamp
plt.savefig(os.path.join(output_dir, f"Shot_{shot_actual['shotID']}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"))
# system.save(r"E:\PYTHON_PROJECTS\pooltool\3cushiontool\NOK_Ball_off.msgpack")
# pt.show(system)
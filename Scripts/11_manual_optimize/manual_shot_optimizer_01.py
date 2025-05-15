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

def initial_shot_direction(ball_trajectory):
    dx = ball_trajectory["x"][3] - ball_trajectory["x"][0]
    dy = ball_trajectory["y"][3] - ball_trajectory["y"][0]
    phi = float(np.degrees(np.arctan2(dy, dx)))

    return phi

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

def find_tlim_index(t, tlim):
    # find the index of t which is smaller than tlim
    ind = np.where(t < tlim)[0][-1] + 1
    if ind == 0:
        ind = 1

    return ind

def objective(trial):
    # Suggest values for the shot parameters using the defined boundaries
    a = trial.suggest_uniform('a', *shot_param_limits['a'])
    b = trial.suggest_uniform('b', *shot_param_limits['b'])
    phi = trial.suggest_uniform('phi', *shot_param_limits['phi'])
    v = trial.suggest_uniform('v', *shot_param_limits['v'])
    theta = trial.suggest_uniform('theta', *shot_param_limits['theta'])

    # Prepare and simulate shot
    shot_env.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, phi, v, theta)
    point, result, tsim, system = shot_env.simulate_shot(a_ballball, b_ballball, c_ballball)

    # Calculate the loss
    total_loss = 0.0
    
    tlim = frac * np.max([shot_actual["balls"][1]["t"][-1], shot_actual["balls"][2]["t"][-1], shot_actual["balls"][3]["t"][-1]])

    plot_settings()
    
    for ball_col, ball_data in shot_actual["balls"].items():
        # only use frac amount of the actual data
        ind = find_tlim_index(ball_data["t"], tlim)        
        actual_times = ball_data["t"][:ind]
        actual_x = ball_data["x"][:ind]
        actual_y = ball_data["y"][:ind]

        simulated_x, simulated_y = interpolate_simulated_to_actual(result[ball_col - 1], tsim, actual_times)
        total_loss += loss_func(actual_x, actual_y, simulated_x, simulated_y)
    
        # Plot the actual and simulated shots
        plot_current_shot(ball_col, actual_x, actual_y, simulated_x, simulated_y)

    lenstudy = len(study.trials)
    if  lenstudy> 1:
        # Get the best parameters
        best_params = study.best_params
        best_a, best_b, best_phi, best_v, best_theta = best_params['a'], best_params['b'], best_params['phi'], best_params['v'], best_params['theta']
        best_loss = study.best_value
    else:
        best_a, best_b, best_phi, best_v, best_theta = a, b, phi, v, theta
        best_loss = total_loss
    

    # Plot the actual and simulated shots
    plt.gca().set_title(f"ShotID: {shot_actual['shotID']}\n"
                    f"trial: {lenstudy}, Loss: {total_loss:.2f}, Best Loss: {best_loss:.4f}\n"
                    f"a: {round(a, 2)}, limits: ({round(shot_param_limits['a'][0], 2)}, {round(shot_param_limits['a'][1], 2)}), Best: {round(best_a, 4)}\n"
                    f"b: {round(b, 2)}, limits: ({round(shot_param_limits['b'][0], 2)}, {round(shot_param_limits['b'][1], 2)}), Best: {round(best_b, 4)}\n"
                    f"phi: {round(phi, 2)}, limits: ({round(shot_param_limits['phi'][0], 2)}, {round(shot_param_limits['phi'][1], 2)}), Best: {round(best_phi, 4)}\n"
                    f"v: {round(v, 2)}, limits: ({round(shot_param_limits['v'][0], 2)}, {round(shot_param_limits['v'][1], 2)}), Best: {round(best_v, 4)}\n"
                    f"theta: {round(theta, 2)}, limits: ({round(shot_param_limits['theta'][0], 2)}, {round(shot_param_limits['theta'][1], 2)}), Best: {round(best_theta, 4)}")
    plt.draw()  # Update the plot

    plt.pause(0.001)  # Pause to allow the plot to be updated
    return total_loss

def set_param_limits(ind, shot_param, shot_param_tol, shot_param_maxlimits):
    # Calculate for each keys in shot_param_tol the lower and upper limits based on the shot_param and shot_param_tol
    shot_param_limits = {}
    if ind >= len(shot_param_tol['a']) - 1:
        ind = len(shot_param_tol['a']) - 1
    print(f"len a = {len(shot_param_tol['a'])}")
    print(f"ind = {ind}")

    for key, value in shot_param.items():
        # assign the current key min max limits in to shot_param_limits
        shot_param_limits[key] = (value - shot_param_tol[key][ind], value + shot_param_tol[key][ind])
        # check if the limits are within the max limits and adjust if necessary
        if shot_param_limits[key][0] < shot_param_maxlimits[key][0]:
            shot_param_limits[key] = (shot_param_maxlimits[key][0], shot_param_limits[key][1])
        if shot_param_limits[key][1] > shot_param_maxlimits[key][1]:
            shot_param_limits[key] = (shot_param_limits[key][0], shot_param_maxlimits[key][1])

    return shot_param_limits

def set_ball_positions(initial_shot_direction, shot_param, shot_actual):
    ball_num = 0
    for ball_col, ball_data in shot_actual["balls"].items():
        ball_num += 1

        if ball_num == 1:
            ball1xy = (ball_data["x"][0], ball_data["y"][0])
            shot_param["phi"] = initial_shot_direction(ball_data)

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

ballball_hit_param_ranges = {
    'a': (0.005, 0.02), 
    'b': (0.05, 0.2),
    'c': (0.5, 2.0)
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

# pyhsics parameter ranges
physics_param_ranges = {
    'u_slide': (0.05, 0.3),
    'u_roll': (0.001, 0.05),
    'u_sp_prop': (0.01, 1.0),
    'e_ballball': (0.75, 0.98),
    'e_cushion': (0.6, 0.95),
    'f_cushion': (0.05, 0.3),
    'u_ballball': (0.01, 0.1)
}

# Shot Parameter
shot_param = {
    'a': 0.0,
    'b': 0.0,
    'phi': 0.0,
    'v': 3.0,
    'theta': 5.0
}

# define tolerance steps for each shot parameter
# Define the fraction of the actual data to use in for loop
# fraction_of_shot = [0.05, 0.15, 0.3, 1, 1]
# shot_param_tol = {
#     'a': (0.5, 0.3, 0.2, 0.1),
#     'b': (0.5, 0.3, 0.2, 0.1),
#     'phi': (1, 0.4, 0.2, 0.1),
#     'v': (3, 1, 0.5, 0.25),
#     'theta': (45, 5, 2, 1)
# }
fraction_of_shot = [1]
shot_param_tol = {
    'a': (0.5,),
    'b': (0.5,),
    'phi': (0.5,),
    'v': (5.0,),
    'theta': (5.0,)
}

shot_param_maxlimits = {
    'a': (-0.5, 0.5),
    'b': (-0.5, 0.5),
    'phi': (-360, 360),
    'v': (1, 8),
    'theta': (0, 89)
}




# Set up shot
num_shots = 1  # Change this value to use a different number of shots


# Initialize figure dictionary
fig = {}

# Optimize shot parameters
for i, shot_actual in enumerate(shots_actual):
    if i >= num_shots:
        break

    fig[i] = plt.figure(figsize=(7.1, 14.2))  # Set the figure size to maintain a 1:2 aspect ratio


    print("optimizing for Shot: ", shot_actual["shotID"])
    ball1xy, ball2xy, ball3xy = set_ball_positions(initial_shot_direction, shot_param, shot_actual)    

    # set the physics parameters
    a_ballball, b_ballball, c_ballball = ballball_hit_params['a'], ballball_hit_params['b'], ballball_hit_params['c']
    u_slide, u_roll, u_sp_prop, u_ballball, e_ballball, e_cushion, f_cushion = physics_params['u_slide'], physics_params['u_roll'], physics_params['u_sp_prop'], physics_params['u_ballball'], physics_params['e_ballball'], physics_params['e_cushion'], physics_params['f_cushion']
    
    # Create billiard environment
    shot_env = BilliardEnv(u_slide, u_roll, u_sp_prop, u_ballball, e_ballball, e_cushion, f_cushion)

    # Use Optuna to find the best shot parameters
    sampler = optuna.samplers.TPESampler(
        consider_prior=True,   # Use prior information from past trials
        multivariate=True,     # Model dependencies between parameters
        group=True,            # Use a single sampler for all parameters
        # n_startup_trials=50,   # Number of initial random trials
        # n_ei_candidates=25,    # Number of candidates for the expected improvement acquisition function
    )
    
    for fraci, frac in enumerate(fraction_of_shot): #np.arange(0.1, 1.1, 0.1):
        
        shot_param_limits = set_param_limits(fraci, shot_param, shot_param_tol, shot_param_maxlimits)
        
        study = optuna.create_study(direction='minimize', sampler=sampler)

        if fraci > 0:
            if fraction_of_shot[fraci-1] == 1:
                # Transfer completed trials from old study to new study
                for trial in oldstudy.trials:
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        study.add_trial(trial)

        study.optimize(objective, n_trials=100)

        # Replace old study with the new study
        oldstudy = study

        # Get the best parameters
        best_params = study.best_params
        a, b, phi, v, theta = best_params['a'], best_params['b'], best_params['phi'], best_params['v'], best_params['theta']

        shot_param = best_params

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
                    f"a: {round(a, 2)}, limits: ({round(shot_param_limits['a'][0], 2)}, {round(shot_param_limits['a'][1], 2)})\n"
                    f"b: {round(b, 2)}, limits: ({round(shot_param_limits['b'][0], 2)}, {round(shot_param_limits['b'][1], 2)})\n"
                    f"phi: {round(phi, 2)}, limits: ({round(shot_param_limits['phi'][0], 2)}, {round(shot_param_limits['phi'][1], 2)})\n"
                    f"v: {round(v, 2)}, limits: ({round(shot_param_limits['v'][0], 2)}, {round(shot_param_limits['v'][1], 2)})\n"
                    f"theta: {round(theta, 2)}, limits: ({round(shot_param_limits['theta'][0], 2)}, {round(shot_param_limits['theta'][1], 2)})")
    plt.draw()  # Update the plot
    plt.pause(0.001)  # Pause to allow the plot to be updated

    output_dir = r"E:\PYTHON_PROJECTS\pooltool\11_manual_optimizer"
    os.makedirs(output_dir, exist_ok=True)
    # Save the plot as an image with date-time stamp
    plt.savefig(os.path.join(output_dir, f"Shot_{shot_actual['shotID']}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"))
    # system.save(r"E:\PYTHON_PROJECTS\pooltool\3cushiontool\NOK_Ball_off.msgpack")
    # pt.show(system)
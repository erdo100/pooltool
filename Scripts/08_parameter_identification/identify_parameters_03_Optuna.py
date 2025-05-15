import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.interpolate import interp1d
from threecushion_shot import BilliardEnv

print(sys.executable)

# Get the current working directory
current_folder = os.getcwd()
print(f"The current folder is: {current_folder}")

# Construct the file path
file_path = r"E:\PYTHON_PROJECTS\POOLTOOL\3cushiontool\Scripts\20221225_2_Match_Ersin_Cemal.pkl"
print(f"The current folder is: {file_path}")

# Load shots from the pickle file
with open(file_path, "rb") as f:
    shots_actual = pickle.load(f)

# Swap x and y axes
for shot_actual in shots_actual:
    for ball in shot_actual["balls"].values():
        ball["x"], ball["y"] = np.array(ball["y"]), np.array(ball["x"])

# Define physics parameters and their limits
physics_params = {
    "u_slide": 0.15,
    "u_roll": 0.005,
    "u_sp_prop": 10 * 2 / 5 / 9,
    "u_ballball": 0.05,
    "e_ballball": 0.95,
    "e_cushion": 0.9,
    "f_cushion": 0.15,
}

# Create billiard environment
shot_env = BilliardEnv(**physics_params)


def initial_shot_direction(ball_trajectory):
    dx = ball_trajectory["x"][3] - ball_trajectory["x"][0]
    dy = ball_trajectory["y"][3] - ball_trajectory["y"][0]
    phi = float(np.degrees(np.arctan2(dy, dx)))

    return phi


def initial_cut_angle(ball1_trajectory, ball2_trajectory):
    dx = ball2_trajectory["x"][0] - ball2_trajectory["x"][0]
    dy = ball2_trajectory["y"][0] - ball2_trajectory["y"][0]
    phi = np.degrees(np.arctan2(dy, dx))

    return phi


def calculate_weights(varlength):
    # Calculate the quadratic weights
    weights = np.arange(varlength) ** 2

    # Normalize the weights to start from 10 and end at 1
    weights = 10 - 9 * (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    return weights


def rms_difference(actual_x, actual_y, simulated_x, simulated_y, weights):
    distances = np.sqrt((actual_x - simulated_x) ** 2 + (actual_y - simulated_y) ** 2)

    rms = distances
    rms = np.sum(rms)

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


def plot_shot_ini():
    plt.figure(
        figsize=(7.1, 14.2)
    )  # Set the figure size to maintain a 1:2 aspect ratio


def plot_shot_update():
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


def optimize_shot_parameters(trial, shot, ball1xy, ball2xy, ball3xy, initial_phi):
    a = trial.suggest_uniform("a", -0.5, 0.)
    b = trial.suggest_uniform("b", -0., 0.5)
    phi = trial.suggest_uniform("phi", initial_phi - 1, initial_phi + 1)
    v = trial.suggest_uniform("v", 2, 8)
    theta = trial.suggest_uniform("theta", 0, 20)

    shot_env.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, phi, v, theta)
    point, result, tsim = shot_env.simulate_shot()
    plot_shot_update()
    rms_total = 0
    ball_colors = {1: "white", 2: "yellow", 3: "red"}
    for ball_col, ball_data in shot["balls"].items():
        actual_times = ball_data["t"]
        actual_x = ball_data["x"]
        actual_y = ball_data["y"]
        simulated_x, simulated_y = interpolate_simulated_to_actual(
            result[ball_col - 1], tsim, actual_times
        )

        weights = calculate_weights(len(actual_x))
        rms_total += rms_difference(
            actual_x, actual_y, simulated_x, simulated_y, weights
        )

        plt.plot(actual_x, actual_y, "-", color=ball_colors[ball_col])
        plt.plot(
            result[ball_col - 1][:, 0],
            result[ball_col - 1][:, 1],
            "--",
            color=ball_colors[ball_col],
        )
        # plt.plot(simulated_x, simulated_y, 'o', color=ball_colors[ball_col])

    plt.gca().set_title(f"RMS: {rms_total:.2f}")
    plt.draw()  # Update the plot
    plt.pause(0.001)  # Pause to allow the plot to be updated
    return rms_total


def simulate_and_plot_best_shot(shot_actual, ball1xy, ball2xy, ball3xy, best_params):
    a = best_params["a"]
    b = best_params["b"]
    phi = best_params["phi"]
    v = best_params["v"]
    theta = best_params["theta"]

    shot_env.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, phi, v, theta)
    point, result, tsim = shot_env.simulate_shot()
    plot_shot_ini()
    plot_shot_update()
    ball_colors = {1: "white", 2: "yellow", 3: "red"}
    for ball_col, ball_data in shot_actual["balls"].items():
        actual_times = ball_data["t"]
        actual_x = ball_data["x"]
        actual_y = ball_data["y"]
        simulated_x, simulated_y = interpolate_simulated_to_actual(
            result[ball_col - 1], tsim, actual_times
        )

        plt.plot(actual_x, actual_y, "-", color=ball_colors[ball_col])
        plt.plot(
            result[ball_col - 1][:, 0],
            result[ball_col - 1][:, 1],
            "--",
            color=ball_colors[ball_col],
        )
        # plt.plot(simulated_x, simulated_y, 'o', color=ball_colors[ball_col])

    plt.gca().set_title(f"Best Shot Simulation for Shot {shot_actual['shotID']}")
    plt.draw()  # Update the plot
    plt.pause(0.001)  # Pause to allow the plot to be updated
    output_dir = r"E:\PYTHON_PROJECTS\pooltool\3cushiontool\Scripts\08_parameter_identification"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"Shot_{shot_actual['shotID']}.jpg"))


# Number of shots to be used in the optimization
num_shots = 1  # Change this value to use a different number of shots

# # Create a TPESampler with consider_endpoints set to True
# sampler = optuna.samplers.TPESampler(
#     consider_prior=True,
#     prior_weight=1.0,
#     consider_magic_clip=True,
#     consider_endpoints=True,  # Consider the endpoints of the parameter range
#     n_startup_trials=10,
#     n_ei_candidates=24,
#     gamma=lambda n: min(25, int(np.sqrt(n))),  # Ensure gamma returns an integer
#     weights=lambda i: np.ones(i)
# )

# Configure TPESampler
sampler = optuna.samplers.TPESampler(
    consider_prior=True,
    consider_magic_clip=False,
    n_startup_trials=15,  # Initial random trials before using TPE
    n_ei_candidates=50,    # Number of candidates per iteration
    multivariate=True,     # Consider correlations between hyperparameters
    group=True,            # Group sampling for related hyperparameters
    prior_weight=0.3,      # Adjusts balance between exploration and exploitation
    constant_liar=True,    # Helps parallel processing
    seed=42                # Ensures reproducibility
)

# Optimize shot parameters using Optuna
for i, shot_actual in enumerate(shots_actual):
    if i >= num_shots:
        break
    plot_shot_ini()
    plot_shot_update()
    print("optimizing for Shot: ", shot_actual["shotID"])
    ball_num = 0
    for ball_col, ball_data in shot_actual["balls"].items():
        ball_num += 1

        if ball_num == 1:
            ball1xy = (ball_data["x"][0], ball_data["y"][0])
            initial_phi = initial_shot_direction(ball_data)

        elif ball_num == 2:
            ball2xy = (ball_data["x"][0], ball_data["y"][0])

        elif ball_num == 3:
            ball3xy = (ball_data["x"][0], ball_data["y"][0])

    # Optimize shot parameters using Optuna with the custom TPESampler
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: optimize_shot_parameters(trial, shot_actual, ball1xy, ball2xy, ball3xy, initial_phi), n_trials=5000)

    optimized_shot_params = study.best_params
    total_rms = study.best_value

    print(f"Optimized shot parameters for shot {shot_actual['shotID']}: {optimized_shot_params}")
    print(f"Total RMS after optimizing shot parameters for shot {shot_actual['shotID']}: {total_rms}")

    # Simulate and plot the best shot
    simulate_and_plot_best_shot(shot_actual, ball1xy, ball2xy, ball3xy, optimized_shot_params)

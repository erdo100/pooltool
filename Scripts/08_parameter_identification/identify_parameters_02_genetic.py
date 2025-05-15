import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from threecushion_shot import BilliardEnv

print(sys.executable)

# Get the current working directory
current_folder = os.getcwd()
print(f"The current folder is: {current_folder}")

# Construct the file path
file_path = os.path.join(current_folder, "20221225_2_Match_Ersin_Cemal.pkl")
print(f"The current folder is: {file_path}")

# Load shots from the pickle file
with open(file_path, "rb") as f:
    shots_actual = pickle.load(f)


# make data quality check of digitized data
# determine cue ball for current shot
# determine ball 2 for current shot
# Calculation of phi

# make data quality check of digitized data

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
physics_limits = {
    "u_slide": (0.01, 0.2),
    "u_roll": (0.001, 0.02),
    "u_sp_prop": (0.1, 0.9),
    "u_ballball": (0.01, 0.2),
    "e_ballball": (0.8, 0.98),
    "e_cushion": (0.5, 0.95),
    "f_cushion": (0.01, 0.2),
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

    rms = weights * distances
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


def optimize_shot_parameters(shot, initial_params, ball1xy, ball2xy, ball3xy):
    def objective(params):
        a, b, phi, v, theta = params
        shot_env.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, phi, v, theta)
        point, result, tsim = shot_env.simulate_shot()
        # shot_env.plot_shot()
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

    bounds = [
        (-0.5, 0.5),  # a
        (-0.5, 0.5),  # b
        (0, 360),  # phi
        (1, 8),  # v
        (0, 60),  # theta
    ]

    result = differential_evolution(
        objective,
        bounds,
        strategy="best1bin",
        maxiter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        callback=None,
        disp=True,
        polish=True,
        init="latinhypercube",
        atol=0,
        updating="deferred",
        workers=1,
        constraints=(),
    )

    return result.x, result.fun


def optimize_physics_parameters():
    def objective(params):
        for i, key in enumerate(physics_params.keys()):
            physics_params[key] = params[i]

        # Create billiard environment
        shot_env = BilliardEnv(**physics_params)

        total_rms = 0
        shot_i = 0
        # make for loop over shots, provide shots and number of shot
        for shot_actual in shots_actual:
            plot_shot_ini()
            plot_shot_update()
            print("optimizing for Shot: ", shot_actual["shotID"])
            ball_num = 0
            for ball_col, ball_data in shot_actual["balls"].items():
                ball_num += 1

                if ball_num == 1:
                    ball1xy = (ball_data["x"][0], ball_data["y"][0])
                    phi = initial_shot_direction(ball_data)

                elif ball_num == 2:
                    ball2xy = (ball_data["x"][0], ball_data["y"][0])

                elif ball_num == 3:
                    ball3xy = (ball_data["x"][0], ball_data["y"][0])

            initial_params = [0.4, 0.0, phi, 3.0, 0]

            _, rms = optimize_shot_parameters(
                shot_actual, initial_params, ball1xy, ball2xy, ball3xy
            )
            total_rms += rms
        return total_rms

    bounds = [physics_limits[key] for key in physics_params.keys()]
    initial_physics_params = list(physics_params.values())
    result = differential_evolution(objective, bounds)
    return result.x, result.fun


# Optimize physics parameters
optimized_physics_params, total_rms = optimize_physics_parameters()
print(f"Optimized physics parameters: {optimized_physics_params}")
print(f"Total RMS after optimizing physics parameters: {total_rms}")

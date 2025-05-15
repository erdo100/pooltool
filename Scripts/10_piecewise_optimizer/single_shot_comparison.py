import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from threecushion_shot import BilliardEnv
import pooltool as pt

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

def plot_shot_ini():
    plt.figure(figsize=(7.1, 14.2))  # Set the figure size to maintain a 1:2 aspect ratio

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


# Set up shot
num_shots = 1  # Change this value to use a different number of shots

# Optimize shot parameters
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

        elif ball_num == 2:
            ball2xy = (ball_data["x"][0], ball_data["y"][0])

        elif ball_num == 3:
            ball3xy = (ball_data["x"][0], ball_data["y"][0])


    # Shot Parameter
    a0 = -0.0
    b0 = 0.6
    phi0 = -81.3
    v0 = 3.6
    theta0 = 0.0
    
    # Alciatori Ball-Ball hit model parameters
    a_ballball0 = 0.009951
    b_ballball0 = 0.108
    c_ballball0 = 1.088

    # Physics parameters
    u_slide0 = 0.2
    u_roll0 = 0.005
    u_sp_prop0 = 10 * 2 / 5 / 9
    e_ballball0 = 0.95
    e_cushion0 = 0.9
    f_cushion0 = 0.15
    u_ballball0 = 0.05 # not relevant for Alciatori Ball-Ball hit model
    

    # set the physics parameters
    a_ballball, b_ballball, c_ballball = a_ballball0, b_ballball0, c_ballball0
    u_slide, u_roll, u_sp_prop, u_ballball, e_ballball, e_cushion, f_cushion = (
        u_slide0,
        u_roll0,
        u_sp_prop0,
        u_ballball0,
        e_ballball0,
        e_cushion0,
        f_cushion0,
    )
    
    # Create billiard environment
    shot_env = BilliardEnv(u_slide, u_roll, u_sp_prop, u_ballball, e_ballball, e_cushion, f_cushion)

    # optimizer loop for shot parameters
    # use the optimizer to find the best shot parameters
    
    a, b, phi, v, theta = a0, b0, phi0, v0, theta0
    # Prepare and simulate shot
    shot_env.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, phi, v, theta)
    point, result, tsim, system = shot_env.simulate_shot(a_ballball, b_ballball, c_ballball)

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

    plt.gca().set_title(f"Best Shot Simulation for Shot {shot_actual['shotID']}")
    plt.draw()  # Update the plot
    plt.pause(1000.001)  # Pause to allow the plot to be updated
    output_dir = r"E:\PYTHON_PROJECTS\pooltool\3cushiontool\Scripts\09_analyze_shots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"Shot_{shot_actual['shotID']}.jpg"))
    system.save(r"E:\PYTHON_PROJECTS\pooltool\3cushiontool\NOK_Ball_off.msgpack")
    pt.show(system)
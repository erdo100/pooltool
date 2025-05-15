import pickle
import time
import numpy as np
from scipy.interpolate import interp1d
from tkinter import filedialog
  
def read_shotfile():
    # pick file woth UI
    # file_path = r"E:\PYTHON_PROJECTS\POOLTOOL\3cushiontool\Scripts\20221225_2_Match_Ersin_Cemal.pkl"
    file_path = filedialog.askopenfilename()

    # Load shots from the pickle file
    with open(file_path, "rb") as f:
        shots_actual = pickle.load(f)

    # Swap x and y axes
    for shot_actual in shots_actual:
        for ball in shot_actual["balls"].values():
            ball["x"], ball["y"] = np.array(ball["y"]), np.array(ball["x"])

    return shots_actual

# Function to save parameters
def save_parameters(params):
    filename = "Shot_" + str(params.value["shot_id"]) + "_parameters_" + time.strftime("%Y%m%d_%H%M%S") + ".pkl"
    file_path = filedialog.asksaveasfilename(initialfile=filename, defaultextension='.pkl', filetypes=[('Pickle files', '*.pkl')])
    if file_path:
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)
        print(f"Parameters saved to {file_path}")
        print(params)

def load_parameters(slider_frame, update_plot, **sliders):
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if file_path:
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
            print(params)
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

    return phi[3]

def get_ball_ids(shot_actual):
    color_mapping = {1: "white", 2: "yellow", 3: "red"}

    ball_ids = {}
    ball_cols = {}
    # Get the second entry (based on insertion order)
    ball_ids[0] = list(shot_actual["balls"].keys())[0] # This is the cue ball
    ball_cols[0] = color_mapping.get(ball_ids[0])  # Assign color
    ball_ids[1] = list(shot_actual["balls"].keys())[1]
    ball_cols[1] = color_mapping.get(ball_ids[1])  # Assign color
    ball_ids[2] = list(shot_actual["balls"].keys())[2]
    ball_cols[2] = color_mapping.get(ball_ids[2])  # Assign color
    
    return ball_ids, ball_cols

def get_ball_positions(shot_actual):

    ball_ids, ball_cols = get_ball_ids(shot_actual)

    balls_xy_ini = {}
    balls_xy_ini[0] = (shot_actual["balls"][1]["x"][0], shot_actual["balls"][1]["y"][0])
    balls_xy_ini[1] = (shot_actual["balls"][2]["x"][0], shot_actual["balls"][2]["y"][0])
    balls_xy_ini[2] = (shot_actual["balls"][3]["x"][0], shot_actual["balls"][3]["y"][0])

    # Calculate initial shot direction
    cueball_x = shot_actual["balls"][ball_ids[0]]["x"]
    cueball_y = shot_actual["balls"][ball_ids[0]]["y"]
    cueball_phi = initial_shot_direction(cueball_x, cueball_y)

    return balls_xy_ini, ball_ids, ball_cols, cueball_phi


def loss_func(actual_x, actual_y, simulated_x, simulated_y):
    distances = np.sqrt((actual_x - simulated_x) ** 2 + (actual_y - simulated_y) ** 2)
    rms = np.sqrt(np.mean(distances ** 2))
    return rms

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
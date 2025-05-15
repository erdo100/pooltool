import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pooltool as pt
from tkinter import Tk, Scale, HORIZONTAL, Label, Button, Frame, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pooltool.ruleset.three_cushion import is_point
from billiardenv import BilliardEnv
from helper_funcs import read_shotfile, interpolate_simulated_to_actual, loss_func, save_parameters, load_parameters, save_system
from slider_definitions import create_ballball_sliders, create_physics_sliders, create_shot_sliders


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

    return balls_xy_ini, ball_ids, ball_cols

def plot_initial_positions(ball_xy_ini):
    for ball_col, ball_xy in ball_xy_ini.items():
        circle = plt.Circle(ball_xy, 0.0615 / 2, color=ball_col, fill=True)
        plt.gca().add_patch(circle)

def plot_current_shot(colind, actual_x, actual_y, simulated_x, simulated_y):
    colortable = ["white", "yellow", "red"]
    colname = colortable[colind]
    circle = plt.Circle((simulated_x[0], simulated_y[0]), 0.0615 / 2, color=colname, fill=True)
    plt.gca().add_patch(circle)
    plt.plot(actual_x, actual_y, "--", color=colname, linewidth=1)
    plt.plot(simulated_x, simulated_y, "-", color=colname, linewidth=1)

def update_plot(event=None):
    global cueball_id, a, b, phi, v, theta, a_ballball, b_ballball, c_ballball, u_slide, u_roll
    global u_sp_prop, e_ballball, e_cushion, f_cushion, shot_actual, ball1xy, ball2xy, ball3xy

    shot_index = shot_slider.get()
    shot_actual = shots_actual[shot_index]
    ball_xy_ini, ball_ids, ball_cols = get_ball_positions(shot_actual)
   
    a = shot_param['a'] = shot_a_slider.get()
    b = shot_param['b'] = shot_b_slider.get()
    phi = shot_param['phi'] = shot_phi_slider.get()
    v = shot_param['v'] = shot_v_slider.get()
    theta = shot_param['theta'] = shot_theta_slider.get()

    a_ballball = ballball_hit_params['a'] = ballball_a_slider.get()
    b_ballball = ballball_hit_params['b'] = ballball_b_slider.get()
    c_ballball = ballball_hit_params['c'] = ballball_c_slider.get()

    u_slide = physics_params['u_slide'] = physics_u_slide_slider.get()
    u_roll = physics_params['u_roll'] = physics_u_roll_slider.get()
    u_sp_prop = physics_params['u_sp_prop'] = physics_u_sp_prop_slider.get()
    e_ballball = physics_params['e_ballball'] = physics_e_ballball_slider.get()
    e_cushion = physics_params['e_cushion'] = physics_e_cushion_slider.get()
    f_cushion = physics_params['f_cushion'] = physics_f_cushion_slider.get()

    # Create billiard environment
    shot_env = BilliardEnv(u_slide, u_roll, u_sp_prop, e_ballball, e_cushion, f_cushion)

    # Prepare and simulate shot with updated parameters
    shot_env.prepare_new_shot(ball_cols, ball_xy_ini, a, b, phi, v, theta)
    point, result, tsim, system = shot_env.simulate_shot(a_ballball, b_ballball, c_ballball)

    plot_settings()
    # plot_initial_positions(ball_xy_ini)
    total_loss = 0.0
    for ball_col, ball_data in shot_actual["balls"].items():
        actual_times = ball_data["t"]
        actual_x = ball_data["x"]
        actual_y = ball_data["y"]
        simulated_x, simulated_y = interpolate_simulated_to_actual(result[ball_col - 1], tsim, actual_times)
        total_loss += loss_func(actual_x, actual_y, simulated_x, simulated_y)
        
        # Plot the actual and simulated shots
        # plot_current_shot(ball_col, actual_x, actual_y, simulated_x, simulated_y)
        
        colind = ball_col - 1
        plot_current_shot(colind, actual_x, actual_y, result[colind][:,0], result[colind][:,1])
        
    plt.gca().set_title(f"ShotID: {shot_actual['shotID']}, Loss: {total_loss:.2f}\n"
                    f"a: {round(a, 2)}\n"
                    f"b: {round(b, 2)}\n"
                    f"phi: {round(phi, 2)}\n"
                    f"v: {round(v, 2)}\n"
                    f"theta: {round(theta, 2)}")
    canvas.draw()  # Update the plot

    return system

def on_closing():
    root.destroy()
    sys.exit()

def show_system():
    system = update_plot()
    pt.show(system)


shots_actual = read_shotfile()

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

# Select the current shot
shot_actual = shots_actual[0]

fig = plt.figure(figsize=(7.1, 14.2))  # Set the figure size to maintain a 1:2 aspect ratio

# GUI setup
root = Tk()
root.title("3-Cushion Shot Optimizer")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Create a frame for the plot
plot_frame = Frame(root)
plot_frame.pack(side="left", fill="both", expand=True)

# Create a canvas for the plot
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

# Create a frame for the sliders
slider_frame = Frame(root)
slider_frame.pack(side="right", fill="both", expand=True)

# Shot parameter sliders
slider_length = 400  # Increase the length of the sliders
slider_height = 30  # Increase the height of the sliders by a factor of 1.2

# Shot selector slider
shot_slider = Scale(slider_frame, from_=0, to=len(shots_actual) - 1, orient=HORIZONTAL, label="Shot", length=slider_length, command=update_plot)
shot_slider.set(0)
shot_slider.pack()

# Shot Shot Parameters slider
shot_a_slider, shot_b_slider, shot_phi_slider, shot_v_slider, shot_theta_slider = create_shot_sliders(slider_frame, shot_param, update_plot)

ballball_a_slider, ballball_b_slider, ballball_c_slider = create_ballball_sliders(slider_frame, ballball_hit_params, update_plot)

physics_u_slide_slider, physics_u_roll_slider, physics_u_sp_prop_slider, physics_e_ballball_slider, physics_e_cushion_slider, physics_f_cushion_slider = create_physics_sliders(slider_frame, physics_params, update_plot)

# Add a button to show the system
show_button = Button(slider_frame, text="Show System", command=show_system)
show_button.pack()

# Add a button to save the parameters
save_button = Button(slider_frame, text="Save Parameters", command=lambda: save_parameters(ballball_hit_params, physics_params))
save_button.pack()

# Add a button to load the parameters
load_button = Button(slider_frame, text="Load Parameters", command=lambda: load_parameters(slider_frame, update_plot, ballball_a_slider, ballball_b_slider, ballball_c_slider, physics_u_slide_slider, physics_u_roll_slider, physics_u_sp_prop_slider, physics_e_ballball_slider, physics_e_cushion_slider, physics_f_cushion_slider))
load_button.pack()

# Add a button to save the system
save_system_button = Button(slider_frame, text="Save System", command=lambda: save_system(update_plot))
save_system_button.pack()

root.mainloop()
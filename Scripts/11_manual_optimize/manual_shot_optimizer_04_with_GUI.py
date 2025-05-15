import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pooltool as pt
from tkinter import Tk, Scale, HORIZONTAL, Label, Button, Frame, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pooltool.ruleset.three_cushion import is_point


class BilliardEnv:
    def __init__(
        self, u_slide, u_roll, u_sp_prop, u_ballball, e_ballball, e_cushion, f_cushion
    ):
        self.table_width = 2.84  # Table dimensions (meters)
        self.table_height = 1.42
        self.series_length = 0
        self.current_step = 0
        self.episode_rewards = []

        # Ball Positions
        self.ball1_ini = (0.1, 0, 1)  # White
        self.ball2_ini = (0.5, 0.5)  # Yellow
        self.ball3_ini = (1.0, 1.0)  # Red

        # define the properties
        self.u_slide = u_slide
        self.u_roll = u_roll
        self.u_sp_prop = u_sp_prop
        self.u_ballball = u_ballball
        self.e_ballball = e_ballball
        self.e_cushion = e_cushion
        self.f_cushion = f_cushion

        self.grav = 9.81

        self.mball = 0.210
        self.Rball = 61.5 / 1000 / 2

        cue_mass = 0.576
        cue_len = 1.47
        cue_tip_R = 21.21 / 2000  # radius nickel=21.21 mm, dime=17.91 mm
        cue_tip_mass = 0.008

        # Build a table with default BILLIARD specs
        self.table = pt.Table.default(pt.TableType.BILLIARD)

        # create the cue
        cue_specs = pt.objects.CueSpecs(
            M=cue_mass,
            length=cue_len,
            tip_radius=cue_tip_R,
            end_mass=cue_tip_mass,
        )

        self.cue = pt.Cue(cue_ball_id="white", specs=cue_specs)

    def prepare_new_shot(self, ball_cols, ball_xy_ini, a, b, phi, v, theta):
        
        for ball_col, ball_xy in ball_xy_ini.items():
            if ball_col == "white":
                ball1xy = ball_xy
            elif ball_col == "yellow":
                ball2xy = ball_xy
            elif ball_col == "red":
                ball3xy = ball_xy

        # Create balls in new positions
        wball = pt.Ball.create(
            "white",
            xy=ball1xy,
            m=self.mball,
            R=self.Rball,
            u_s=self.u_slide,
            u_r=self.u_roll,
            u_sp_proportionality=self.u_sp_prop,
            u_b=self.u_ballball,
            e_b=self.e_ballball,
            e_c=self.e_cushion,
            f_c=self.f_cushion,
            g=self.grav,
        )

        yball = pt.Ball.create(
            "yellow",
            xy=ball3xy,
            m=self.mball,
            R=self.Rball,
            u_s=self.u_slide,
            u_r=self.u_roll,
            u_sp_proportionality=self.u_sp_prop,
            u_b=self.u_ballball,
            e_b=self.e_ballball,
            e_c=self.e_cushion,
            f_c=self.f_cushion,
            g=self.grav,
        )
        
        rball = pt.Ball.create(
            "red",
            xy=ball2xy,
            m=self.mball,
            R=self.Rball,
            u_s=self.u_slide,
            u_r=self.u_roll,
            u_sp_proportionality=self.u_sp_prop,
            u_b=self.u_ballball,
            e_b=self.e_ballball,
            e_c=self.e_cushion,
            f_c=self.f_cushion,
            g=self.grav,
        )

        # modify the cue ball in self.cue
        self.cue.cue_ball_id = ball_cols[0]

        # phi = pt.aim.at_ball(self.system, "red", cut=cut)
        # set the cue
        self.cue.set_state(a=a, b=b, V0=v, phi=phi, theta=theta)

        # Wrap it up as a System
        self.system = pt.System(
            table=self.table, balls=(wball, yball, rball), cue=self.cue
        )

    def get_ball_routes(self):
        shot = self.system
        shotcont = pt.continuize(shot, dt=0.01, inplace=False)
        white = shotcont.balls["white"]
        white_history = white.history_cts
        white_rvw, s_cue, tsim = white_history.vectorize()
        yellow = shotcont.balls["yellow"]
        yellow_history = yellow.history_cts
        yellow_rvw, s_cue, tsim = yellow_history.vectorize()
        red = shotcont.balls["red"]
        red_history = red.history_cts
        red_rvw, s_cue, tsim = red_history.vectorize()

        # We can grab the xy-coordinates for each ball from the `rvw` array by with the following.
        results = {}
        results[0] = white_rvw[:, 0, :2]
        results[1] = yellow_rvw[:, 0, :2]
        results[2] = red_rvw[:, 0, :2]

        return results, tsim

    def simulate_shot(self, a, b, c):
        # run the physics model
        point = 0

        engine = pt.physics.PhysicsEngine()  # start with default
        engine.resolver.stick_ball.squirt_throttle = 0.0
        engine.resolver.ball_linear_cushion = pt.physics.ball_lcushion_models[
            pt.physics.BallLCushionModel.MATHAVAN_2010
        ]() # HAN_2005 and MATHAVAN_2010
        # Friction fit curve u_b = a + b * exp(-c * v_rel) used in David Alciatore's TP A-14
        engine.resolver.ball_ball.friction.a = a
        engine.resolver.ball_ball.friction.b = b
        engine.resolver.ball_ball.friction.c = c
    
        # Pass the engine to your simulate call.
        pt.simulate(self.system, engine=engine, inplace=True)

        results, tsim = self.get_ball_routes()
        if is_point(self.system):
            point = 1

        return point, results, tsim, self.system
    
def read_shotfile():

    # pick file woth UI
    file_path = r"E:\PYTHON_PROJECTS\POOLTOOL\3cushiontool\Scripts\20221225_2_Match_Ersin_Cemal.pkl"
    file_path = filedialog.askopenfilename()

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
    plt.plot(actual_x, actual_y, "--", color=colname, linewidth=1)
    plt.plot(simulated_x, simulated_y, "-", color=colname, linewidth=3)

def get_ball_ids(shot_actual):
    color_mapping = {1: "white", 2: "yellow", 3: "red"}

    ball_ids = {}
    ball_cols = {}
    # Get the second entry (based on insertion order)
    ball_ids[0] = list(shot_actual["balls"].keys())[0]
    ball_cols[0] = color_mapping.get(ball_ids[0])  # Assign color
    ball_ids[1] = list(shot_actual["balls"].keys())[1]
    ball_cols[1] = color_mapping.get(ball_ids[1])  # Assign color
    ball_ids[2] = list(shot_actual["balls"].keys())[2]
    ball_cols[2] = color_mapping.get(ball_ids[2])  # Assign color
    
    return ball_ids, ball_cols

def get_ball_positions(shot_actual):

    ball_ids, ball_cols = get_ball_ids(shot_actual)
    balls_xy_ini = {}
    for ball_id, ball_data in shot_actual["balls"].items():
        balls_xy_ini[ball_cols[ball_id-1]] = (ball_data["x"][0], ball_data["y"][0])

    return balls_xy_ini, ball_ids, ball_cols

def plot_initial_positions(ball_xy_ini):
    for ball_col, ball_xy in ball_xy_ini.items():
        circle = plt.Circle(ball_xy, 0.0615 / 2, color=ball_col, fill=False)
        plt.gca().add_patch(circle)

def update_plot(event=None):
    global cueball_id, a, b, phi, v, theta, a_ballball, b_ballball, c_ballball, u_slide, u_roll, u_sp_prop, e_ballball, e_cushion, f_cushion, u_ballball, shot_actual, ball1xy, ball2xy, ball3xy

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
    u_ballball = physics_params['u_ballball'] = physics_u_ballball_slider.get()

    # Create billiard environment
    shot_env = BilliardEnv(u_slide, u_roll, u_sp_prop, u_ballball, e_ballball, e_cushion, f_cushion)

    # Prepare and simulate shot with updated parameters
    shot_env.prepare_new_shot(ball_cols, ball_xy_ini, a, b, phi, v, theta)
    point, result, tsim, system = shot_env.simulate_shot(a_ballball, b_ballball, c_ballball)

    plot_settings()
    plot_initial_positions(ball_xy_ini)
    total_loss = 0.0
    for ball_col, ball_data in shot_actual["balls"].items():
        actual_times = ball_data["t"]
        actual_x = ball_data["x"]
        actual_y = ball_data["y"]
        simulated_x, simulated_y = interpolate_simulated_to_actual(result[ball_col - 1], tsim, actual_times)
        total_loss += loss_func(actual_x, actual_y, simulated_x, simulated_y)
        
        # Plot the actual and simulated shots
        # plot_current_shot(ball_col, actual_x, actual_y, simulated_x, simulated_y)
        plot_current_shot(ball_col, actual_x, actual_y, result[ball_col - 1][:,0], result[ball_col - 1][:,1])

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

# Set up shot
num_shots = 1  # Change this value to use a different number of shots

# Initialize figure dictionary
fig = {}

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

# Shot selection slider
shot_slider = Scale(slider_frame, from_=0, to=len(shots_actual) - 1, orient=HORIZONTAL, label="Shot", length=slider_length, command=update_plot)
shot_slider.set(0)
shot_slider.pack()

shot_a_slider = Scale(slider_frame, from_=-0.6, to=0.6, resolution=0.001, orient=HORIZONTAL, label="Shot a", length=slider_length, command=update_plot)
shot_a_slider.set(shot_param['a'])
shot_a_slider.pack()

shot_b_slider = Scale(slider_frame, from_=-0.6, to=0.6, resolution=0.001, orient=HORIZONTAL, label="Shot b", length=slider_length, command=update_plot)
shot_b_slider.set(shot_param['b'])
shot_b_slider.pack()

shot_phi_slider = Scale(slider_frame, from_=-180, to=180, resolution=0.01, orient=HORIZONTAL, label="Shot phi", length=slider_length, command=update_plot)
shot_phi_slider.set(shot_param['phi'])
shot_phi_slider.pack()

shot_v_slider = Scale(slider_frame, from_=0, to=10, resolution=0.01, orient=HORIZONTAL, label="Shot v", length=slider_length, command=update_plot)
shot_v_slider.set(shot_param['v'])
shot_v_slider.pack()

shot_theta_slider = Scale(slider_frame, from_=0, to=90, resolution=0.01, orient=HORIZONTAL, label="Shot theta", length=slider_length, command=update_plot)
shot_theta_slider.set(shot_param['theta'])
shot_theta_slider.pack()

# Ball-ball parameter sliders
ballball_a_slider = Scale(slider_frame, from_=0, to=0.02, resolution=0.0001, orient=HORIZONTAL, label="Ball-Ball a", length=slider_length, command=update_plot)
ballball_a_slider.set(ballball_hit_params['a'])
ballball_a_slider.pack()

ballball_b_slider = Scale(slider_frame, from_=0, to=0.2, resolution=0.001, orient=HORIZONTAL, label="Ball-Ball b", length=slider_length, command=update_plot)
ballball_b_slider.set(ballball_hit_params['b'])
ballball_b_slider.pack()

ballball_c_slider = Scale(slider_frame, from_=0, to=2, resolution=0.01, orient=HORIZONTAL, label="Ball-Ball c", length=slider_length, command=update_plot)
ballball_c_slider.set(ballball_hit_params['c'])
ballball_c_slider.pack()

# Physics parameter sliders
physics_u_slide_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics u_slide", length=slider_length, command=update_plot)
physics_u_slide_slider.set(physics_params['u_slide'])
physics_u_slide_slider.pack()

physics_u_roll_slider = Scale(slider_frame, from_=0, to=0.1, resolution=0.001, orient=HORIZONTAL, label="Physics u_roll", length=slider_length, command=update_plot)
physics_u_roll_slider.set(physics_params['u_roll'])
physics_u_roll_slider.pack()

physics_u_sp_prop_slider = Scale(slider_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, label="Physics u_sp_prop", length=slider_length, command=update_plot)
physics_u_sp_prop_slider.set(physics_params['u_sp_prop'])
physics_u_sp_prop_slider.pack()

physics_e_ballball_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics e_ballball", length=slider_length, command=update_plot)
physics_e_ballball_slider.set(physics_params['e_ballball'])
physics_e_ballball_slider.pack()

physics_e_cushion_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics e_cushion", length=slider_length, command=update_plot)
physics_e_cushion_slider.set(physics_params['e_cushion'])
physics_e_cushion_slider.pack()

physics_f_cushion_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics f_cushion", length=slider_length, command=update_plot)
physics_f_cushion_slider.set(physics_params['f_cushion'])
physics_f_cushion_slider.pack()

# Add a button to show the system
show_button = Button(slider_frame, text="Show System", command=show_system)
show_button.pack()

root.mainloop()
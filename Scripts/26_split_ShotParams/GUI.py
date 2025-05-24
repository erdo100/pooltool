import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from tkinter import *

import pooltool as pt
from helper_funcs import run_study, get_ball_positions, open_shotfile, evaluate_loss, save_parameters, load_parameters, save_system, abs_velocity
import multiprocessing as mp
import copy
from pathlib import Path
from diff_evolution_optimizer import DEOptimizer

from pathlib import Path
class plot_3cushion():
    def __init__(self, sim_env, params):
        self.sim_env = sim_env
        self.params = params

        self.root = Tk()
        self.root.title("3-Cushion Shot Optimizer")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        res_dpi = 100
        fig_width = int(screen_width / res_dpi * 0.75)
        fig_height = int(screen_height / res_dpi * 0.8)
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=res_dpi)

        # Data initialization
        total_loss = 0
        actual_times = (0, 1)
        actual_x = {
            0: np.array([-0.5, -0.5]),
            1: np.array([-0.7, -0.7]),
            2: np.array([-0.6, -0.6])
        }
        actual_y = {
            0: np.array([0.5, 0.5]),
            1: np.array([0.5, 0.5]),
            2: np.array([0.6, 0.6])
        }
        tsim = (0, 1)
        simulated_x = {k: np.copy(v) for k, v in actual_x.items()}
        simulated_y = {k: np.copy(v) for k, v in actual_y.items()}
        actual_v = (0, 0)
        simulated_v = (0, 0)

        # Main layout frames
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True)

        left_frame = Frame(main_frame)
        left_frame.pack(side=LEFT, fill=Y)

        right_frame = Frame(main_frame)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)        # Button grid frame on the left
        button_grid_frame = Frame(left_frame)
        button_grid_frame.pack(side=TOP, fill=X, pady=10)

        # Configure grid columns to be uniform
        for i in range(3):
            button_grid_frame.columnconfigure(i, weight=1, uniform="button")        # Create buttons in a 3x2 grid
        Button(button_grid_frame, text="Load Shots", command=self.start_read_shotfile).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        Button(button_grid_frame, text="Show System", command=self.show_system).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        Button(button_grid_frame, text="Save Parameters", command=lambda: save_parameters(self.params)).grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        Button(button_grid_frame, text="Load Parameters", command=lambda: load_parameters(button_grid_frame, self.update_plot, **self.sliders)).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        Button(button_grid_frame, text="Save System", command=lambda: save_system(self.update_plot)).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        Button(button_grid_frame, text="Run Phi", command=lambda: run_study(self.SA, self.params)).grid(row=1, column=2, padx=2, pady=2, sticky="ew")        # Slider frame
        slider_frame = Frame(left_frame)
        slider_frame.pack(side=TOP, fill=Y)

        # Optimization controls frame (directly below sliders)
        optimization_frame = Frame(left_frame)
        optimization_frame.pack(side=TOP, fill=X, pady=10)

        # Trial configuration
        trial_frame = Frame(optimization_frame)
        trial_frame.pack(fill=X, pady=2)
        Label(trial_frame, text="Total Trials:").pack(side=LEFT)
        self.trial_entry = Entry(trial_frame, width=8)
        self.trial_entry.insert(0, "100")
        self.trial_entry.pack(side=LEFT, padx=5)

        # Start optimization button
        self.start_button = Button(trial_frame, text="Start Optimization", command=self.start_optimization, state=NORMAL)
        self.start_button.pack(side=RIGHT, padx=5)

        # Plot canvas in right frame
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, right_frame)
        toolbar.update()

        # Axes layout
        left_margin_px = 5
        bottom_margin_px = 20
        right_margin_px = 5
        top_margin_px = 5
        gap_px = 50

        left_margin = left_margin_px / (fig.get_figwidth() * fig.dpi)
        bottom_margin = bottom_margin_px / (fig.get_figheight() * fig.dpi)
        right_margin = right_margin_px / (fig.get_figwidth() * fig.dpi)
        top_margin = top_margin_px / (fig.get_figheight() * fig.dpi)
        gapx = gap_px / (fig.get_figwidth() * fig.dpi)
        gapy = gap_px / (fig.get_figheight() * fig.dpi)

        first_width = 1 / 3
        first_height = 0.95

        ax = [None] * 5
        handles = {
            "title": {},
            "circle_actual": {},
            "shotline_actual": {},
            "shotline_simulated": {},
            "varline_actual": {},
            "varline_simulated": {},
            "loss": {}
        }

        ax[0] = fig.add_axes([left_margin, bottom_margin, first_width, first_height])
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        handles["title"] = ax[0].set_title("")
        ax[0].set_xlim(0.0, 1.42)
        ax[0].set_ylim(0.0, 2.84)
        ax[0].set_aspect("equal", adjustable="box")
        ax[0].set_facecolor("lightgray")
        grid_size = 2.84 / 8
        ax[0].set_xticks(np.arange(0, 1.42 + grid_size, grid_size))
        ax[0].set_yticks(np.arange(0, 2.84 + grid_size, grid_size))
        ax[0].grid(True)

        available_height = 1 - top_margin - bottom_margin - 3 * gapy
        right_height = available_height / 4
        right_width = 0.6
        right_x = left_margin + first_width + gapx

        for i in range(4):
            bottom_y = bottom_margin + (3 - i) * (right_height + gapy)
            ax[i + 1] = fig.add_axes([right_x, bottom_y, right_width, right_height])

        colortable = ["white", "yellow", "red"]
        for i in range(3):
            handles["circle_actual"][i] = ax[0].add_patch(Circle((actual_x[i][0], actual_y[i][0]), 0.0615 / 2, color=colortable[i], fill=True))
            handles["shotline_actual"][i] = ax[0].plot(actual_x[i], actual_y[i], color=colortable[i], linestyle='--', linewidth=2)
            handles["shotline_simulated"][i] = ax[0].plot(simulated_x[i], simulated_y[i], color=colortable[i], linestyle='-', linewidth=2)
            handles["varline_actual"][i] = ax[i + 1].plot(actual_times, actual_v, label=f"{colortable[i]} actual v in m/s", linestyle='--', linewidth=2)
            handles["varline_simulated"][i] = ax[i + 1].plot(tsim, simulated_v, label=f"{colortable[i]} simulated v in m/s", linestyle='-', linewidth=2)
            handles["loss"][i] = ax[4].plot(actual_times, actual_v, label=f"{colortable[i]} loss in m", linestyle='-', linewidth=2)

        for axi in [ax[1], ax[2], ax[3], ax[4]]:
            axi.legend(loc="best")
            axi.grid(True)

        # Initialize sliders dictionary
        self.sliders = {}
        self.shot_id_changed = False
        
        # Create sliders
        self.create_sliders(slider_frame)        # Store references
        canvas.draw()
        self.root.canvas = canvas
        self.root.ax = ax
        self.root.handles = handles
        self.root.debug = {"ax": None}  # Add debug reference

    def create_sliders(self, slider_frame):
        """Create all parameter sliders"""
        # Shot selector slider
        self.add_slider(slider_frame, 'shot_id', 0, 9, 1, "Shot", self.update_shot_id)
        
        # Shot parameter sliders
        self.add_slider(slider_frame, 'shot_a', -0.6, 0.6, 0.001, "Shot a")
        self.add_slider(slider_frame, 'shot_b', -0.6, 0.6, 0.001, "Shot b")
        self.add_slider(slider_frame, 'shot_phi', -180, 180, 0.01, "Shot phi")
        self.add_slider(slider_frame, 'shot_v', 0, 10, 0.01, "Shot v")
        self.add_slider(slider_frame, 'shot_theta', 0, 90, 0.1, "Shot theta")
        
        # Ball-ball parameter sliders
        self.add_slider(slider_frame, 'physics_ballball_a', 0, 0.3, 0.001, "Ball-Ball a")
        self.add_slider(slider_frame, 'physics_ballball_b', 0, 1.0, 0.01, "Ball-Ball b")
        self.add_slider(slider_frame, 'physics_ballball_c', 0, 5, 0.1, "Ball-Ball c")
        
        # Physics parameter sliders
        self.add_slider(slider_frame, 'physics_u_slide', 0, 0.3, 0.001, "Physics u_slide")
        self.add_slider(slider_frame, 'physics_u_roll', 0, 0.015, 0.0001, "Physics u_roll")
        self.add_slider(slider_frame, 'physics_u_sp_prop', 0, 1, 0.01, "Physics u_sp_prop")
        self.add_slider(slider_frame, 'physics_e_ballball', 0.5, 1, 0.001, "Physics e_ballball")
        self.add_slider(slider_frame, 'physics_e_cushion', 0.5, 1, 0.001, "Physics e_cushion")
        self.add_slider(slider_frame, 'physics_f_cushion', 0, 0.5, 0.001, "Physics f_cushion")
        self.add_slider(slider_frame, 'physics_h_cushion', 0.035, 0.039, 0.0001, "Physics h_cushion")

    def add_slider(self, master, key, frm, to, res, label, cmd=None):
        """Add a slider to the frame"""
        s = Scale(master, from_=frm, to=to, resolution=res, orient=HORIZONTAL, label=label, length=300, command=cmd or self.update_plot)
        s.set(self.params.value[key])
        s.pack()
        self.sliders[key] = s


    def start_read_shotfile(self):
        self.SA = open_shotfile()
        self.shot_id_changed = True
        self.update_plot()

    def on_closing(self):
        self.root.destroy()
        sys.exit()

    def show_system(self):
        system = self.update_plot()
        pt.show(system)


    def update_shot_id(self, event=None):
        self.shot_id_changed = True
        self.update_plot()

    def get_slider_values(self, sliders, params):
        # update the actual parameters

        params.value['shot_id'] = sliders['shot_id'].get()
        params.value['shot_a'] = sliders['shot_a'].get()
        params.value['shot_b'] = sliders['shot_b'].get()
        params.value['shot_phi'] = sliders['shot_phi'].get()
        params.value['shot_v'] = sliders['shot_v'].get()
        params.value['shot_theta'] = sliders['shot_theta'].get()

        params.value['physics_ballball_a'] = sliders['physics_ballball_a'].get()
        params.value['physics_ballball_b'] = sliders['physics_ballball_b'].get()
        params.value['physics_ballball_c'] = sliders['physics_ballball_c'].get()
        params.value['physics_u_slide'] = sliders['physics_u_slide'].get()
        params.value['physics_u_roll'] = sliders['physics_u_roll'].get()
        params.value['physics_u_sp_prop'] = sliders['physics_u_sp_prop'].get()
        params.value['physics_e_ballball'] = sliders['physics_e_ballball'].get()
        params.value['physics_e_cushion'] = sliders['physics_e_cushion'].get()
        params.value['physics_f_cushion'] = sliders['physics_f_cushion'].get()
        params.value['physics_h_cushion'] = sliders['physics_h_cushion'].get()
        
        return params
    def start_optimization(self):
        # Get settings from GUI
        print("Starting optimization...")

        b1b2b3 = self.SA['Data']["B1B2B3"][0]
        shot_actual = self.SA["Shot"][self.params.value['shot_id']]
        balls_xy_ini, ball_cols, cueball_phi = get_ball_positions(shot_actual, b1b2b3)

        totalruns = int(self.trial_entry.get())
        
        optimizer = DEOptimizer(shot_actual, self.params, balls_xy_ini, ball_cols, maxiter=totalruns)
        result, best_params = optimizer.run_optimization()

        # Get best parameters and update sliders
        print("Best parameters:")
        for key, value in best_params.limits.items():
            if key in self.sliders:
                print(f"Updating slider {key} to {best_params.value[key]}")
                self.sliders[key].set(best_params.value[key])

        print("Optimization completed.")


    def update_plot(self, event=None, is_optimization_update=False):
        
        if not hasattr(self, 'SA'):
            return
        

        # check if the number of shots has changed
        newmax = len(self.SA["Shot"])
        currentmax = self.sliders["shot_id"].config()['to'][2]
        if newmax != currentmax:
            # update the length of the slider
            self.sliders["shot_id"].config(to=newmax-1)
            currentpos=self.sliders["shot_id"].get()

            # check whether current position of the slider is valid, otherwise reset it
            if currentpos >= newmax:
                self.sliders["shot_id"].set(newmax-1)

        
        SA = self.SA
        sliders = self.sliders
        h = self.root.handles
        ax = self.root.ax
        debug_ax = self.root.debug["ax"]

        # Retrieve current shot and slider values
        shot_id = self.params.value['shot_id'] = sliders['shot_id'].get()
        shot_actual = SA["Shot"][shot_id]
        b1b2b3 = self.SA['Data']["B1B2B3"][shot_id]
        ball_xy_ini, ball_cols, cue_phi = get_ball_positions(shot_actual, b1b2b3)

        self.params = self.get_slider_values(sliders, self.params)


        if self.shot_id_changed:
            # params.value['shot_phi'] = cue_phi
            # sliders['shot_phi'].set(cue_phi)
            self.shot_id_changed = False
        else:
            self.shot_id_changed = False

        # set new parameters and simulate shot
        self.sim_env.ball_cols = ball_cols
        self.sim_env.balls_xy_ini = ball_xy_ini
        self.sim_env.prepare_new_shot(self.params)
        self.sim_env.simulate_shot()

        # calculate the reward
        loss = evaluate_loss(self.sim_env, shot_actual)

        tsim, white_rvw, yellow_rvw, red_rvw = self.sim_env.get_ball_routes()

        # update the plot with the new data
        loss_max = 0
        for i, rvw in enumerate([white_rvw, yellow_rvw, red_rvw]):
            (tmax, vmax) = (0,0)
            # Simulated shot
            xs = rvw[:, 0, 0]
            ys = rvw[:, 0, 1]

            # Actual Shot
            ta = np.array(shot_actual['Ball'][i]["t"])
            xa = np.array(shot_actual['Ball'][i]["x"])
            ya = np.array(shot_actual['Ball'][i]["y"])

            # shot plot axes
            h["circle_actual"][i].center = (rvw[0, 0, 0], rvw[0, 0, 1])
            
            h["shotline_actual"][i][0].set_data(xa, ya)            
            h["shotline_simulated"][i][0].set_data(xs ,ys)

            # velocity plot axes
            va = abs_velocity(ta, xa, ya)
            tmax = max(tmax, max(ta))
            vmax = max(vmax, max(va))
            h["varline_actual"][i][0].set_data(ta, va)

            vs = abs_velocity(tsim, xs, ys)
            tmax = max(tmax, max(tsim))
            vmax = max(vmax, max(vs))
            h["varline_simulated"][i][0].set_data(tsim, vs)


            h["loss"][i][0].set_data(loss["ball"][i]["time"], loss["ball"][i]["total"])
            loss_max = max(loss_max, np.max(loss["ball"][i]["total"]))
            OM = 10**(np.floor(np.log10(tmax))-1)
            tlim = np.ceil(tmax/OM*1.1)*OM
            
            if vmax < 0.1:
                vlim = 0.1
            else:
                OM = 10**(np.floor(np.log10(vmax))-1)
                vlim = np.ceil(vmax/OM*1.1)*OM

            self.root.ax[i+1].set_xlim((0, tlim))
            self.root.ax[i+1].set_ylim((0, vlim ))

        if loss_max < 0.001:
            loss_lim = 0.001
        else:
            OM = 10**(np.floor(np.log10(loss_max))-1)
            loss_lim = np.ceil(loss_max/OM*1.1)*OM

        self.root.ax[4].set_xlim((0, tlim ))
        self.root.ax[4].set_ylim((0, loss_lim ))


        if is_optimization_update:
            h["title"].set_text(f"OPTIMIZED Shot {shot_id} - loss = {loss["total"]:.3f}")
        else:
            h["title"].set_text(f"Shot {shot_id} - loss = {loss["total"]:.3f}")

        # plt.draw()
        self.root.canvas.draw()  # Update the figure display
        # Only refresh existing canvas
        self.root.canvas.draw_idle()  # Ensure the canvas is refreshed
        return self.sim_env.system




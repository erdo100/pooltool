import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pooltool as pt
from helper_funcs import run_study, get_ball_positions, open_shotfile, evaluate_loss, save_parameters, load_parameters, save_system, abs_velocity
import multiprocessing as mp
import copy
from pathlib import Path
from tkinter import Tk, Scale, HORIZONTAL, Label, Button, Frame, filedialog, Menu, Spinbox, Entry
from tkinter import LEFT, RIGHT, TOP, BOTTOM, X, StringVar, OptionMenu, NORMAL, DISABLED
from diff_evolution_optimizer import DEOptimizer

from pathlib import Path

class plot_3cushion():
    def __init__(self, sim_env, params):

        root = Tk()
        # initial data to plot and initiate the elements
        total_loss = 0
        actual_times = (0,1)
        actual_x = {}
        actual_y = {}
        actual_x[0] = np.array((-0.5,-0.5))
        actual_y[0] = np.array((0.5,0.5))
        actual_x[1] = np.array((-0.7,-0.7))
        actual_y[1] = np.array((0.5,0.5))
        actual_x[2] = np.array((-0.6,-0.6))
        actual_y[2] = np.array((0.6,0.6))
        
        tsim = (0,1)
        simulated_x = {key: np.copy(value) for key, value in actual_x.items()}
        simulated_y = {key: np.copy(value) for key, value in actual_y.items()}
            
        actual_v = (0,0)
        simulated_v = (0,0)
        
        
        debug = {}
        debug['plot_flag'] = False
        debug['ax'] = None
        if debug['plot_flag']:
            fig, debug['ax'] = plt.subplots()

            
        # Create the main window
        root.title("3-Cushion Shot Optimizer")
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        screen_width = root.winfo_screenwidth()  # Screen width in pixels
        screen_height = root.winfo_screenheight()  # Screen height in pixels

        res_dpi = 100
        # Figure size
        fig_width = int(screen_width /res_dpi * 0.75)
        fig_height = int(screen_height/res_dpi * 0.8)
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=res_dpi)


        # Create a frame for the plot
        plot_frame = Frame(root)
        plot_frame.pack(side="left", fill="both", expand=True)

        # Create a canvas for the plot
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Add the toolbar for zooming and panning
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()

        # Create a frame for the sliders
        slider_frame = Frame(root)
        slider_frame.pack(side="right", fill="both", expand=False)


        # Add a button to show the system
        load_shots_button = Button(slider_frame, text="Load Shots", command=self.start_read_shotfile)
        load_shots_button.pack()

        # Add a button to show the system
        show_button = Button(slider_frame, text="Show System", command=self.show_system)
        show_button.pack()

        # Add a button to save the parameters
        save_button = Button(slider_frame, text="Save Parameters", command=lambda: save_parameters(self.params))
        save_button.pack()

        # Add a button to load the parameters
        load_button = Button(slider_frame, text="Load Parameters", command=lambda: load_parameters(slider_frame, self.update_plot, **self.sliders))
        load_button.pack()

        # Add a button to save the system
        save_system_button = Button(slider_frame, text="Save System", command=lambda: save_system(self.update_plot))
        save_system_button.pack()

        # Add a button to run phi variation the system
        run_study_button = Button(slider_frame, text="Run Phi", command=lambda: run_study(self.SA, self.params))
        run_study_button.pack()
                

        ### AXES LAYOUT ###
        # Left side (main plot)
        left_margin_px = 5
        bottom_margin_px = 20
        right_margin_px = 5
        top_margin_px = 5
        gap_px = 50  # Gap between the three right-side diagrams
        
        # Convert margins to figure-relative coordinates
        left_margin = left_margin_px / (fig.get_figwidth() * fig.dpi)
        bottom_margin = bottom_margin_px / (fig.get_figheight() * fig.dpi)
        right_margin = right_margin_px / (fig.get_figwidth() * fig.dpi)
        top_margin = top_margin_px / (fig.get_figheight() * fig.dpi)
        gapx = gap_px / (fig.get_figwidth() * fig.dpi)
        gapy = gap_px / (fig.get_figheight() * fig.dpi)

        # Compute the width of the first diagram (normalized)
        first_width = 1/3
        first_height = 0.95  # Normalized height

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
        ax[0].set_xticklabels([])  # Remove x-axis tick labels
        ax[0].set_yticklabels([])  # Remove y-axis tick labels
        handles["title"] = ax[0].set_title("")
        ax[0].set_xlim(0.0, 1.42)
        ax[0].set_ylim(0.0, 2.84)
        ax[0].set_aspect("equal", adjustable="box")  # Set the aspect ratio to 1:2
        ax[0].set_facecolor("lightgray")  # Set the background color to light gray
        grid_size = 2.84 / 8
        ax[0].set_xticks(np.arange(0, 1.42 + grid_size, grid_size))
        ax[0].set_yticks(np.arange(0, 2.84 + grid_size, grid_size))
        ax[0].grid(True)

        # Right side: Create three vertically arranged subplots with double width.
        # Compute available height for the 3 stacked diagrams on the right
        available_height = 1 - top_margin - bottom_margin - 3 * gapy  # Space after top & bottom margins + gaps
        right_height = available_height / 4  # Equal height for 3 diagrams
        right_width = 0.6  # Given width
        right_x = left_margin + first_width + gapx  # Offset by 50px

        # Create 3 right-side diagrams stacked vertically
        bottom_y = bottom_margin + 3 * (right_height + gapy)
        ax[1] = fig.add_axes([right_x, bottom_y, right_width, right_height])
        bottom_y = bottom_margin + 2 * (right_height + gapy)
        ax[2] = fig.add_axes([right_x, bottom_y, right_width, right_height])
        bottom_y = bottom_margin + 1 * (right_height + gapy)
        ax[3] = fig.add_axes([right_x, bottom_y, right_width, right_height])
        bottom_y = bottom_margin + 0 * (right_height + gapy)
        ax[4] = fig.add_axes([right_x, bottom_y, right_width, right_height])
        

        # Adjust the spacing between subplots
        fig.subplots_adjust(hspace=0.07, wspace=0.07, left=0.05)  # Adjust these values as needed
        
        colortable = ["white", "yellow", "red"]
        # plot the initial positions and the current shot routes
        for i in range(3):
            
            handles["circle_actual"][i] = ax[0].add_patch(plt.Circle((actual_x[i][0], actual_y[i][0]), 0.0615 / 2, color=colortable[i], fill=True))
            handles["shotline_actual"][i] = ax[0].plot(actual_x[i], actual_y[i], color=colortable[i], linestyle='--', linewidth=2)
            handles["shotline_simulated"][i] = ax[0].plot(simulated_x[i], simulated_y[i], color=colortable[i], linestyle='-',linewidth=2)

            # plot the velocities for each axes
            handles["varline_actual"][i] = ax[i+1].plot(actual_times, actual_v, label=f"{colortable[i]} actual v in m/s", linestyle='--', linewidth=2)#, marker='o')
            handles["varline_simulated"][i] = ax[i+1].plot(tsim, simulated_v, label=f"{colortable[i]} simulated v in m/s", linestyle='-', linewidth=2)#, marker='o')

            # plot the loss for each ball
            handles["loss"][i] = ax[4].plot(actual_times, actual_v, label=f"{colortable[i]} loss in m", linestyle='-', linewidth=2)#, marker='o')


        for axi in [ax[1], ax[2], ax[3], ax[4]]:
            axi.legend(loc="best")
            axi.grid(True)

        
        root.handles = handles
        root.ax = ax
        root.debug = debug
        root.canvas = canvas
        self.root = root

        self.sim_env = sim_env
        self.params = params
        
        ### Create a frame for the calculation controls
        self.calculation_controls_frame = Frame(self.root)
        self.calculation_controls_frame.pack(side=TOP, fill=X, expand=False, padx=5, pady=5)

        # Configure grid layout for better organization
        self.calculation_controls_frame.columnconfigure(0, weight=1)
        self.calculation_controls_frame.columnconfigure(1, weight=1)

        # Loss function selection
        loss_frame = Frame(self.calculation_controls_frame)
        loss_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=2)

        Label(loss_frame, text="Loss Function:").pack(side=LEFT)
        self.loss_var = StringVar(value="Position MSE")
        OptionMenu(
            loss_frame, self.loss_var,
            "Position MSE", 
            "Velocity MAE",
            "Combined Loss"
        ).pack(side=LEFT, padx=5)

        # Optimization algorithm selection
        optimizer_frame = Frame(self.calculation_controls_frame)
        optimizer_frame.grid(row=1, column=0, sticky="w", pady=2)

        Label(optimizer_frame, text="Optimizer:").pack(side=LEFT)
        self.optimizer_var = StringVar(value="Random")
        OptionMenu(
            optimizer_frame, self.optimizer_var,
            "Random", 
            "Evolutionary",
            "Genetic",
        ).pack(side=LEFT, padx=5)

        # Trial configuration with validation
        trial_frame = Frame(self.calculation_controls_frame)
        trial_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)

        Label(trial_frame, text="Total Trials:").pack(side=LEFT)
        self.trial_entry = Entry(trial_frame, width=8, validate='key')#,
                            # validatecommand=(self.root.register(self._validate_int), '%P'))
        self.trial_entry.insert(0, "100")
        self.trial_entry.pack(side=LEFT, padx=5)

        # Start button with safety checks
        self.start_button = Button(
            trial_frame, 
            text="Start Optimization", 
            command=self.start_optimization,
            state=NORMAL
        )
        self.start_button.pack(side=RIGHT, padx=5)
        

        # Initialize sliders dictionary
        self.sliders = self.create_sliders(slider_frame, params, 10, self.update_shot_id, self.update_plot)

        canvas.draw()  # Update the figure display




    # Shot parameter sliders
    def create_sliders(self, slider_frame, params, total_shots, update_shot_id, update_plot):
        # Shot parameter sliders
        slider_length = 400
        slider_height = 30  

        sliders = {}

        # Shot selector slider
        sliders['shot_id'] = shot_id_slider = Scale(slider_frame, from_=0, to=total_shots - 1, orient=HORIZONTAL, label="Shot", length=slider_length, command=update_shot_id)
        shot_id_slider.set(0)
        shot_id_slider.pack()

        sliders['shot_a'] = shot_a_slider = Scale(slider_frame, from_=-0.6, to=0.6, resolution=0.001, orient=HORIZONTAL, label="Shot a", length=slider_length, command=update_plot)
        shot_a_slider.set(params.value['shot_a'])
        shot_a_slider.pack()

        sliders['shot_b'] = shot_b_slider = Scale(slider_frame, from_=-0.6, to=0.6, resolution=0.001, orient=HORIZONTAL, label="Shot b", length=slider_length, command=update_plot)
        shot_b_slider.set(params.value['shot_b'])
        shot_b_slider.pack()

        sliders['shot_phi'] = shot_phi_slider = Scale(slider_frame, from_=-180, to=180, resolution=0.01, orient=HORIZONTAL, label="Shot phi", length=slider_length, command=update_plot)
        shot_phi_slider.set(params.value['shot_phi'])
        shot_phi_slider.pack()

        sliders['shot_v'] = shot_v_slider = Scale(slider_frame, from_=0, to=10, resolution=0.01, orient=HORIZONTAL, label="Shot v", length=slider_length, command=update_plot)
        shot_v_slider.set(params.value['shot_v'])
        shot_v_slider.pack()

        sliders['shot_theta'] = shot_theta_slider = Scale(slider_frame, from_=0, to=90, resolution=0.1, orient=HORIZONTAL, label="Shot theta", length=slider_length, command=update_plot)
        shot_theta_slider.set(params.value['shot_theta'])
        shot_theta_slider.pack()

        # Ball-ball parameter sliders
        sliders['physics_ballball_a'] = ballball_a_slider = Scale(slider_frame, from_=0, to=0.3, resolution=0.001, orient=HORIZONTAL, label="Ball-Ball a", length=slider_length, command=update_plot)
        ballball_a_slider.set(params.value['physics_ballball_a'])
        ballball_a_slider.pack()

        sliders['physics_ballball_b'] = ballball_b_slider = Scale(slider_frame, from_=0, to=1.0, resolution=0.01, orient=HORIZONTAL, label="Ball-Ball b", length=slider_length, command=update_plot)
        ballball_b_slider.set(params.value['physics_ballball_b'])
        ballball_b_slider.pack()

        sliders['physics_ballball_c'] = ballball_c_slider = Scale(slider_frame, from_=0, to=5, resolution=0.1, orient=HORIZONTAL, label="Ball-Ball c", length=slider_length, command=update_plot)
        ballball_c_slider.set(params.value['physics_ballball_c'])
        ballball_c_slider.pack()

        # Physics parameter sliders
        sliders['physics_u_slide'] = physics_u_slide_slider = Scale(slider_frame, from_=0, to=0.3, resolution=0.001, orient=HORIZONTAL, label="Physics u_slide", length=slider_length, command=update_plot)
        physics_u_slide_slider.set(params.value['physics_u_slide'])
        physics_u_slide_slider.pack()

        sliders['physics_u_roll'] = physics_u_roll_slider = Scale(slider_frame, from_=0, to=0.015, resolution=0.0001, orient=HORIZONTAL, label="Physics u_roll", length=slider_length, command=update_plot)
        physics_u_roll_slider.set(params.value['physics_u_roll'])
        physics_u_roll_slider.pack()

        sliders['physics_u_sp_prop'] = physics_u_sp_prop_slider = Scale(slider_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, label="Physics u_sp_prop", length=slider_length, command=update_plot)
        physics_u_sp_prop_slider.set(params.value['physics_u_sp_prop'])
        physics_u_sp_prop_slider.pack()

        sliders['physics_e_ballball'] = physics_e_ballball_slider = Scale(slider_frame, from_=0.6, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics e_ballball", length=slider_length, command=update_plot)
        physics_e_ballball_slider.set(params.value['physics_e_ballball'])
        physics_e_ballball_slider.pack()

        sliders['physics_e_cushion'] = physics_e_cushion_slider = Scale(slider_frame, from_=0.5, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics e_cushion", length=slider_length, command=update_plot)
        physics_e_cushion_slider.set(params.value['physics_e_cushion'])
        physics_e_cushion_slider.pack()

        sliders['physics_f_cushion'] = physics_f_cushion_slider = Scale(slider_frame, from_=0, to=0.5, resolution=0.001, orient=HORIZONTAL, label="Physics f_cushion", length=slider_length, command=update_plot)
        physics_f_cushion_slider.set(params.value['physics_f_cushion'])
        physics_f_cushion_slider.pack()

        sliders['physics_h_cushion'] = physics_cushion_height_slider = Scale(slider_frame, from_=0.035, to=0.039, resolution=0.0001, orient=HORIZONTAL, label="Physics h_cushion", length=slider_length, command=update_plot)
        physics_cushion_height_slider.set(params.value['physics_h_cushion'])
        physics_cushion_height_slider.pack()

        return sliders


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

        method = self.optimizer_var.get()
        totalruns = int(self.trial_entry.get())
        
        optimizer = DEOptimizer(shot_actual, self.params, balls_xy_ini, ball_cols, workers=mp.cpu_count(), maxiter=totalruns)
        result, best_params = optimizer.run_optimization()

        print("Optimization completed.")
        # Get best parameters and update sliders
        # for loop over keys in best_params
        print("Best parameters:")
        for key, value in best_params.limits.items():
            if key in self.sliders:
                print(f"Updating slider {key} to {best_params.value[key]}")
                self.sliders[key].set(best_params.value[key])



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
        params = self.params
        h = self.root.handles
        ax = self.root.ax
        debug_ax = self.root.debug["ax"]

        # Retrieve current shot and slider values
        shot_id = params.value['shot_id'] = sliders['shot_id'].get()
        shot_actual = SA["Shot"][shot_id]
        b1b2b3 = self.SA['Data']["B1B2B3"][shot_id]
        ball_xy_ini, ball_cols, cue_phi = get_ball_positions(shot_actual, b1b2b3)

        params = self.get_slider_values(sliders, params)


        if self.shot_id_changed:
            # params.value['shot_phi'] = cue_phi
            # sliders['shot_phi'].set(cue_phi)
            self.shot_id_changed = False
        else:
            self.shot_id_changed = False

        # set new parameters and simulate shot
        self.sim_env.ball_cols = ball_cols
        self.sim_env.balls_xy_ini = ball_xy_ini
        self.sim_env.prepare_new_shot(params)
        self.sim_env.simulate_shot()

        # calculate the reward
        loss = evaluate_loss(self.sim_env, shot_actual)
        total_loss = 0
        for balli in range(3):
            total_loss += np.sum(loss["ball"][balli]["total"]) 

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


            h["loss"][i][0].set_data(loss["ball"][i]["time"], np.cumsum(loss["ball"][i]["total"]))
            loss_max = max(loss_max, np.max(np.cumsum(loss["ball"][i]["total"])))
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
            h["title"].set_text(f"OPTIMIZED Shot {shot_id} - loss = {total_loss:.3f}")
        else:
            h["title"].set_text(f"Shot {shot_id} - loss = {total_loss:.3f}")

        # plt.draw()
        self.root.canvas.draw()  # Update the figure display
        # Only refresh existing canvas
        self.root.canvas.draw_idle()  # Ensure the canvas is refreshed
        return self.sim_env.system


        

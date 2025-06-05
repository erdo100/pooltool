import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from tkinter import *
from tkinter import Menu

import pooltool as pt
from helper_funcs import get_ball_spins, run_study, get_ball_positions, open_shotfile, save_parameters, load_parameters, save_system, abs_velocity
from loss_funcs import evaluate_loss
import multiprocessing as mp
import copy
from pathlib import Path
from diff_evolution_optimizer import DEOptimizer

class plot_3cushion():
    def __init__(self, sim_env, params):
        self.sim_env = sim_env
        self.params = params

        self.root = Tk()
        self.root.title("3-Cushion Shot Optimizer")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)        # Create menu bar
        self.create_menu_bar()

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        res_dpi = 100
        fig_width = int(screen_width / res_dpi * 0.75)
        fig_height = int(screen_height / res_dpi * 0.8)
        fig = Figure(figsize=(fig_width, fig_height), dpi=res_dpi)

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
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)
        
        
        # Slider frame
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
        
        # Population size configuration
        Label(trial_frame, text="Population Size:").pack(side=LEFT, padx=(10, 0))
        self.popsize_entry = Entry(trial_frame, width=8)
        self.popsize_entry.insert(0, "50")
        self.popsize_entry.pack(side=LEFT, padx=5)

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

        first_width = 1 / 4  # Reduced to make room for more plots
        first_height = 0.95

        ax = [None] * 8  # Now we need 8 axes: 1 table + 3 velocity + 1 loss + 3 spin
        handles = {
            "title": {},
            "circle_actual": {},
            "shotline_actual": {},
            "shotline_simulated": {},
            "varline_actual": {},
            "varline_simulated": {},
            "loss": {},
            "spin_roll": {},
            "spin_top": {},
            "spin_side": {}
        }

        # Create table plot (billiard table view)
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

        # Calculate layout for velocity and loss plots (left column)
        available_height = 1 - top_margin - bottom_margin - 3 * gapy
        right_height = available_height / 4
        right_width_left = 0.35  # Width for left column of plots
        right_x_left = left_margin + first_width + gapx

        # Create velocity plots (white, yellow, red) and loss plot
        for i in range(4):
            bottom_y = bottom_margin + (3 - i) * (right_height + gapy)
            ax[i + 1] = fig.add_axes([right_x_left, bottom_y, right_width_left, right_height])

        # Calculate layout for spin plots (right column)
        right_width_right = 0.35  # Width for right column of plots
        right_x_right = right_x_left + right_width_left + gapx

        # Create spin plots (white, yellow, red) - only 3 plots needed
        for i in range(3):
            bottom_y = bottom_margin + (2 - i) * (right_height + gapy) + right_height + gapy  # Align with velocity plots
            ax[i + 5] = fig.add_axes([right_x_right, bottom_y, right_width_right, right_height])

        colortable = ["white", "yellow", "red"]
        for i in range(3):
            handles["circle_actual"][i] = ax[0].add_patch(Circle((actual_x[i][0], actual_y[i][0]), 0.0615 / 2, color=colortable[i], fill=True))
            handles["shotline_actual"][i] = ax[0].plot(actual_x[i], actual_y[i], color=colortable[i], linestyle='--', linewidth=2)
            handles["shotline_simulated"][i] = ax[0].plot(simulated_x[i], simulated_y[i], color=colortable[i], linestyle='-', linewidth=2)
            handles["varline_actual"][i] = ax[i + 1].plot(actual_times, actual_v, label=f"{colortable[i]} actual v in m/s", linestyle='--', linewidth=2)
            handles["varline_simulated"][i] = ax[i + 1].plot(tsim, simulated_v, label=f"{colortable[i]} simulated v in m/s", linestyle='-', linewidth=2)
            handles["loss"][i] = ax[4].plot(actual_times, actual_v, label=f"{colortable[i]} loss in m", linestyle='-', linewidth=2, marker='o', markersize=5)
            
            # Add spin plots for each ball
            handles["spin_roll"][i] = ax[i + 5].plot(tsim, simulated_v, label=f"{colortable[i].capitalize()} Roll spin", linestyle='-', linewidth=2, color='blue')
            handles["spin_top"][i] = ax[i + 5].plot(tsim, simulated_v, label=f"{colortable[i].capitalize()} Top spin", linestyle='-', linewidth=2, color='black')
            handles["spin_side"][i] = ax[i + 5].plot(tsim, simulated_v, label=f"{colortable[i].capitalize()} Side spin", linestyle='-', linewidth=2, color='red')
            ax[i + 5].set_xlabel("Time (s)")
            ax[i + 5].set_ylabel("Spin (rad/s)")

        for axi in [ax[1], ax[2], ax[3], ax[4]]:
            axi.legend(loc="best")
            axi.grid(True)
            
        # Set up spin plot legends and grids
        for i in range(3):
            ax[i + 5].legend(loc="best")
            ax[i + 5].grid(True)

        # Initialize sliders and checkboxes dictionaries
        self.sliders = {}
        self.checkboxes = {}
        self.checkbox_vars = {}
        self.shot_id_changed = False
        
        # Create sliders
        self.create_sliders(slider_frame)
        
        # Store references
        canvas.draw()
        self.root.canvas = canvas
        self.root.ax = ax
        self.root.handles = handles
        self.root.debug = {"ax": None}  # Add debug reference

    def create_menu_bar(self):
        """Create the menu bar with figure options"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Shots", command=self.start_read_shotfile)
        file_menu.add_command(label="Save Parameters", command=lambda: save_parameters(self.params))
        file_menu.add_command(label="Load Parameters", command=lambda: load_parameters(None, self.update_plot, **self.sliders))
        file_menu.add_command(label="Save System", command=lambda: save_system(self.update_plot))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Figure menu
        figure_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Figure", menu=figure_menu)
        figure_menu.add_command(label="Save Figure as PNG", command=self.save_figure_png)
        figure_menu.add_command(label="Save Figure as PDF", command=self.save_figure_pdf)
        figure_menu.add_command(label="Save Figure as SVG", command=self.save_figure_svg)
        figure_menu.add_separator()
        figure_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        figure_menu.add_command(label="Tight Layout", command=self.apply_tight_layout)
        figure_menu.add_separator()
        figure_menu.add_command(label="Grid On/Off", command=self.toggle_grid)
        figure_menu.add_command(label="Refresh Plot", command=self.update_plot)
        
        # View menu
        view_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show System", command=self.show_system)
        view_menu.add_command(label="Zoom to Fit", command=self.zoom_to_fit)
        
        # Tools menu
        tools_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Run Phi Study", command=lambda: run_study(self.SA, self.params))
        tools_menu.add_command(label="Start Optimization", command=self.start_optimization)
          # Plot menu
        plot_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Plot", menu=plot_menu)
        
        # Initialize marker visibility states
        self.marker_visible = {"white": False, "yellow": False, "red": False}
        
        # Add marker toggle options
        plot_menu.add_command(label="Toggle White Ball Markers", command=lambda: self.toggle_markers("white"))
        plot_menu.add_command(label="Toggle Yellow Ball Markers", command=lambda: self.toggle_markers("yellow"))
        plot_menu.add_command(label="Toggle Red Ball Markers", command=lambda: self.toggle_markers("red"))
        plot_menu.add_separator()
        plot_menu.add_command(label="Toggle All Markers", command=self.toggle_all_markers)

        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def save_figure_png(self):
        """Save the current figure as PNG"""
        from tkinter.filedialog import asksaveasfilename
        filename = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Figure as PNG"
        )
        if filename:
            try:
                self.root.canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {filename}")
            except Exception as e:
                print(f"Error saving figure: {e}")

    def save_figure_pdf(self):
        """Save the current figure as PDF"""
        from tkinter.filedialog import asksaveasfilename
        filename = asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save Figure as PDF"
        )
        if filename:
            try:
                self.root.canvas.figure.savefig(filename, bbox_inches='tight')
                print(f"Figure saved as {filename}")
            except Exception as e:
                print(f"Error saving figure: {e}")

    def save_figure_svg(self):
        """Save the current figure as SVG"""
        from tkinter.filedialog import asksaveasfilename
        filename = asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
            title="Save Figure as SVG"
        )
        if filename:
            try:
                self.root.canvas.figure.savefig(filename, bbox_inches='tight')
                print(f"Figure saved as {filename}")
            except Exception as e:
                print(f"Error saving figure: {e}")

    def reset_zoom(self):
        """Reset zoom on all axes to default view"""
        try:
            for ax in self.root.ax:
                if ax is not None:
                    ax.relim()
                    ax.autoscale()
            self.root.canvas.draw()
            print("Zoom reset to default view")
        except Exception as e:
            print(f"Error resetting zoom: {e}")

    def apply_tight_layout(self):
        """Apply tight layout to the figure"""
        try:
            self.root.canvas.figure.tight_layout()
            self.root.canvas.draw()
            print("Tight layout applied")
        except Exception as e:
            print(f"Error applying tight layout: {e}")

    def toggle_grid(self):
        """Toggle grid on/off for all axes"""
        try:
            for ax in self.root.ax:
                if ax is not None:
                    ax.grid(not ax.grid.get_visible() if hasattr(ax.grid, 'get_visible') else True)
            self.root.canvas.draw()
            print("Grid toggled")
        except Exception as e:
            print(f"Error toggling grid: {e}")

    def zoom_to_fit(self):
        """Zoom to fit all data in the plots"""
        try:
            for ax in self.root.ax:
                if ax is not None:
                    ax.relim()
                    ax.autoscale_view()
            self.root.canvas.draw()
            print("Zoomed to fit data")
        except Exception as e:
            print(f"Error zooming to fit: {e}")

    def show_about(self):
        """Show about dialog"""
        from tkinter.messagebox import showinfo
        showinfo("About", 
                "3-Cushion Shot Optimizer\n\n"
                "A tool for analyzing and optimizing billiard shots\n"
                "with physics simulation and parameter optimization.\n\n"
                "Features:\n"
                "• Shot visualization and analysis\n"
                "• Parameter optimization\n"
                "• Physics simulation\n"
                "• Multiple export formats")

    def create_sliders(self, slider_frame):
        """Create all parameter sliders with checkboxes"""
        # Shot selector slider (no checkbox for this one)
        self.add_slider(slider_frame, 'shot_id', 0, 9, 1, "Shot", self.update_shot_id, include_checkbox=False)
        
        # Shot parameter sliders - use limits from parameters.py
        for key in ['shot_a', 'shot_b', 'shot_phi', 'shot_v', 'shot_theta']:
            if key in self.params.limits:
                min_val, max_val = self.params.limits[key]
                resolution = self.get_resolution(min_val, max_val)
                self.add_slider(slider_frame, key, min_val, max_val, resolution, key.replace('_', ' ').title())
        
        # Ball-ball parameter sliders - use limits from parameters.py
        for key in ['physics_ballball_a', 'physics_ballball_b', 'physics_ballball_c']:
            if key in self.params.limits:
                min_val, max_val = self.params.limits[key]
                resolution = self.get_resolution(min_val, max_val)
                self.add_slider(slider_frame, key, min_val, max_val, resolution, key.replace('_', ' ').title())
                
        # Physics parameter sliders - use limits from parameters.py
        physics_keys = ['physics_u_slide', 'physics_u_roll', 'physics_u_sp_prop', 
                       'physics_e_ballball', 'physics_e_cushion', 'physics_f_cushion', 'physics_h_cushion']
        for key in physics_keys:
            if key in self.params.limits:
                min_val, max_val = self.params.limits[key]
                resolution = self.get_resolution(min_val, max_val)
                self.add_slider(slider_frame, key, min_val, max_val, resolution, key.replace('_', ' ').title())

    def get_resolution(self, min_val, max_val):
        """Calculate appropriate resolution based on parameter range"""
        range_val = max_val - min_val

        # if range_val > 100:
        #     return 0.1
        # elif range_val > 10:
        #     return 0.1
        # elif range_val > 1:
        #     return 0.01
        # elif range_val > 0.1:
        #     return 0.001
        # elif range_val > 0.01:
        #     return 0.0001
        # else:
        #     return 0.00001

        return 1e-6

    def add_slider(self, master, key, frm, to, res, label, cmd=None, include_checkbox=True):
        """Add a slider to the frame with optional checkbox and enhanced interactions"""
        # Create a frame to hold checkbox and slider
        slider_container = Frame(master)
        slider_container.pack(fill=X, padx=5, pady=2)
        
        if include_checkbox:
            # Create checkbox for optimization selection
            checkbox_var = BooleanVar()
            checkbox = Checkbutton(slider_container, variable=checkbox_var, width=2)
            checkbox.pack(side=LEFT)
            
            # Store checkbox reference
            self.checkbox_vars[key] = checkbox_var
            self.checkboxes[key] = checkbox
        
        # Create slider
        s = Scale(slider_container, from_=frm, to=to, resolution=res, orient=HORIZONTAL, 
                 label=label, length=280, command=cmd or self.update_plot)
        s.set(self.params.value[key])
        s.pack(side=LEFT, fill=X, expand=True)
        
        # Add enhanced slider interactions
        self.setup_slider_interactions(s, key, frm, to, res)
        
        self.sliders[key] = s

    def setup_slider_interactions(self, slider, key, min_val, max_val, resolution):
        """Setup enhanced interactions for sliders"""
        # Store slider properties for calculations
        slider.min_val = min_val
        slider.max_val = max_val
        slider.resolution = resolution
        slider.param_key = key
        
        # Calculate step sizes
        normal_step = resolution
        big_step = resolution * 10
        range_val = max_val - min_val
        
        # Ensure big_step doesn't exceed reasonable bounds
        if big_step > range_val / 20:
            big_step = range_val / 20
        if big_step < normal_step:
            big_step = normal_step * 2
            
        slider.normal_step = normal_step
        slider.big_step = big_step
        
        # Mouse wheel support
        def on_mousewheel(event):
            current_val = slider.get()
            step = slider.big_step if (event.state & 0x1) else slider.normal_step  # Shift key for big steps
            
            if event.delta > 0:  # Scroll up
                new_val = min(current_val + step, slider.max_val)
            else:  # Scroll down
                new_val = max(current_val - step, slider.min_val)
            
            slider.set(new_val)
            self.update_plot()
        
        # Right-click context menu for big steps
        def on_right_click(event):
            # Prevent the event from propagating to avoid slider jump
            self.show_slider_context_menu(event, slider)
            return "break"  # This prevents further event handling
        
        # Keyboard shortcuts when slider has focus
        def on_key_press(event):
            current_val = slider.get()
            
            if event.keysym == 'Left':
                step = slider.big_step if (event.state & 0x1) else slider.normal_step  # Shift for big steps
                new_val = max(current_val - step, slider.min_val)
                slider.set(new_val)
                self.update_plot()
            elif event.keysym == 'Right':
                step = slider.big_step if (event.state & 0x1) else slider.normal_step  # Shift for big steps
                new_val = min(current_val + step, slider.max_val)
                slider.set(new_val)
                self.update_plot()
            elif event.keysym == 'Home':
                slider.set(slider.min_val)
                self.update_plot()
            elif event.keysym == 'End':
                slider.set(slider.max_val)
                self.update_plot()
        
        # Double-click to reset to default value
        def on_double_click(event):
            if key in self.params.value:
                default_val = self.params.value[key]
                # Ensure default is within bounds
                default_val = max(min_val, min(default_val, max_val))
                slider.set(default_val)
                self.update_plot()
        
        # Bind events
        slider.bind("<MouseWheel>", on_mousewheel)
        slider.bind("<Button-3>", on_right_click)  # Right click
        slider.bind("<KeyPress>", on_key_press)
        slider.bind("<Double-Button-1>", on_double_click)
        # Make slider focusable
        slider.config(takefocus=True)

    def show_slider_context_menu(self, event, slider):
        """Show context menu for slider with step size options"""
        import tkinter.messagebox as messagebox
        
        menu = Menu(self.root, tearoff=0)
        
        # Add menu items for different step operations
        current_val = slider.get()
        
        def set_custom_value():
            result = self.ask_custom_value(slider)
            if result is not None:
                slider.set(result)
                self.update_plot()
        
        def set_custom_step_size():
            result = self.ask_custom_step_size(slider)
            if result is not None:
                slider.normal_step = result
                slider.big_step = result * 10  # Big step is 10x normal step
                # Update the slider's resolution to match the new step size
                slider.config(resolution=result)
        
        # Add menu items
        menu.add_command(label="Enter Custom Value...", command=set_custom_value)
        menu.add_command(label=f"Set Custom Step Size... (current: {slider.normal_step:.6f})", command=set_custom_step_size)
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def ask_custom_value(self, slider):
        """Ask user for custom slider value"""
        from tkinter.simpledialog import askfloat
        
        current_val = slider.get()
        result = askfloat("Set Custom Value", 
                         f"Enter value for {slider.param_key} (Range: {slider.min_val} to {slider.max_val}):",
                         initialvalue=current_val,
                         minvalue=slider.min_val,
                         maxvalue=slider.max_val)
        return result

    def ask_custom_step_size(self, slider):
        """Ask user for custom step size for slider"""
        from tkinter.simpledialog import askfloat
        
        # Calculate a reasonable default step size based on the range
        range_size = slider.max_val - slider.min_val
        default_step = range_size / 100  # 1% of range as default
        
        current_step = slider.normal_step
        result = askfloat("Set Custom Step Size", 
                         f"Enter step size for {slider.param_key}:\n"
                         f"Current: {current_step:.6f}\n"
                         f"Range: {slider.min_val} to {slider.max_val}\n"
                         f"Suggested: {default_step:.6f}",
                         initialvalue=current_step,
                         minvalue=0.000001,  # Very small minimum step
                         maxvalue=range_size)  # Maximum step is the full range
        return result

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
        
        # Disable the optimization button during optimization
        self.start_button.config(state=DISABLED, text="Optimizing...")
        
        # Get selected parameters for optimization
        selected_params = []
        for key, checkbox_var in self.checkbox_vars.items():
            if checkbox_var.get():
                selected_params.append(key)
        
        if not selected_params:
            print("No parameters selected for optimization!")
            # Re-enable the optimization button since we're returning early
            self.start_button.config(state=NORMAL, text="Start Optimization")
            return

        print(f"Optimizing parameters: {selected_params}")

        b1b2b3 = self.SA['Data']["B1B2B3"][0]
        shot_actual = self.SA["Shot"][self.params.value['shot_id']]
        balls_xy_ini, ball_cols, cueball_phi = get_ball_positions(shot_actual, b1b2b3)

        totalruns = int(self.trial_entry.get())
        
        # Get and validate population size
        try:
            popsize = int(self.popsize_entry.get())
            if popsize <= 0:
                print("Population size must be a positive integer. Using default value of 50.")
                popsize = 50
        except ValueError:
            print("Invalid population size entered. Using default value of 50.")
            popsize = 50
        
        print(f"Using population size: {popsize}")
        
        optimizer = DEOptimizer(shot_actual, self.params, balls_xy_ini, ball_cols, 
                               maxiter=totalruns, selected_params=selected_params, popsize=popsize)
        # try:
        result, best_params = optimizer.run_optimization()
        # Get best parameters and update sliders
        print("Best parameters:")
        for key in selected_params:
            if key in self.sliders:
                print(f"Updating slider {key} to {best_params.value[key]}")
                self.sliders[key].set(best_params.value[key])

        # Update the main GUI plot after optimization
        self.update_plot(is_optimization_update=True)
        
        print("Optimization completed.")
        # except Exception as e:
        #     print(f"Optimization failed with error: {e}")
        #     # Re-enable the optimization button in case of error
        #     self.start_button.config(state=NORMAL, text="Start Optimization")
        #     return

        # Re-enable the optimization button at the end
        self.start_button.config(state=NORMAL, text="Start Optimization")

    def update_plot(self, event=None, is_optimization_update=False):
        if not hasattr(self, 'SA'):
            return        # check if the number of shots has changed
        newmax = len(self.SA["Shot"])
        currentmax = self.sliders["shot_id"].config()['to'][2]
        if newmax != currentmax:
            # update the length of the slider
            self.sliders["shot_id"].config(to=newmax-1)
            currentpos = self.sliders["shot_id"].get()

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
        distance_dev = evaluate_loss(self.sim_env, shot_actual, method="distance")
        loss = evaluate_loss(self.sim_env, shot_actual, method="eventbased")

        tsim, white_rvw, yellow_rvw, red_rvw = self.sim_env.get_ball_routes()        # update the plot with the new data
        
        loss_max = 0
        colortable = ["white", "yellow", "red"]  # Define color table for ball colors
        for i, rvw in enumerate([white_rvw, yellow_rvw, red_rvw]):
    
            roll_spin, top_spin, side_spin = get_ball_spins(rvw)
            
            (tmax, vmax) = (0,0)
            # Simulated shot
            xs = rvw[:, 0, 0]
            ys = rvw[:, 0, 1]

            # Actual Shot
            ta = np.array(shot_actual['Ball'][i]["t"])
            xa = np.array(shot_actual['Ball'][i]["x"])
            ya = np.array(shot_actual['Ball'][i]["y"])            # shot plot axes
            h["circle_actual"][i].center = (rvw[0, 0, 0], rvw[0, 0, 1])
            
            h["shotline_actual"][i][0].set_data(xa, ya)            
            h["shotline_simulated"][i][0].set_data(xs ,ys)
              # Add markers if enabled for this ball color
            ball_color = colortable[i]
            if hasattr(self, 'marker_visible') and self.marker_visible.get(ball_color, False):
                # Add colored markers only to actual trajectories
                # Remove existing markers first if they exist
                if f"markers_actual_{i}" in h:
                    h[f"markers_actual_{i}"].remove()
                
                # Add new markers with ball color (only for actual shots)
                h[f"markers_actual_{i}"] = ax[0].scatter(xa, ya, c='black', s=5, marker='o', zorder=5, edgecolors='none')
            else:
                # Remove markers if they exist but are disabled
                if f"markers_actual_{i}" in h:
                    h[f"markers_actual_{i}"].remove()
                    del h[f"markers_actual_{i}"]

            # velocity plot axes
            va = abs_velocity(ta, xa, ya)
            tmax = max(tmax, max(ta))
            vmax = max(vmax, max(va))
            h["varline_actual"][i][0].set_data(ta, va)

            vs = abs_velocity(tsim, xs, ys)
            tmax = max(tmax, max(tsim))
            vmax = max(vmax, max(vs))
            h["varline_simulated"][i][0].set_data(tsim, vs)

            # Update spin plots
            h["spin_roll"][i][0].set_data(tsim, roll_spin)
            h["spin_top"][i][0].set_data(tsim, top_spin)
            h["spin_side"][i][0].set_data(tsim, side_spin)

            h["loss"][i][0].set_data(distance_dev["ball"][i]["time"], distance_dev["ball"][i]["total"])
            loss_max = max(loss_max, np.max(distance_dev["ball"][i]["total"]))
            OM = 10**(np.floor(np.log10(tmax))-1)
            tlim = np.ceil(tmax/OM*1.1)*OM
            
            if vmax < 0.1:
                vlim = 0.1
            else:
                OM = 10**(np.floor(np.log10(vmax))-1)
                vlim = np.ceil(vmax/OM*1.1)*OM

            self.root.ax[i+1].set_xlim((0, tlim))
            self.root.ax[i+1].set_ylim((0, vlim ))
            
            # Set spin plot limits
            spin_values = np.concatenate([roll_spin, top_spin, side_spin])
            if len(spin_values) > 0:
                spin_max = max(abs(np.min(spin_values)), abs(np.max(spin_values)))
                if spin_max < 0.1:
                    spin_lim = 0.1
                else:
                    OM_spin = 10**(np.floor(np.log10(spin_max))-1)
                    spin_lim = np.ceil(spin_max/OM_spin*1.1)*OM_spin
                self.root.ax[i+5].set_xlim((0, tlim))
                self.root.ax[i+5].set_ylim((-spin_lim, spin_lim))

        if loss_max < 0.001:
            loss_lim = 0.001
        else:
            OM = 10**(np.floor(np.log10(loss_max))-1)
            loss_lim = np.ceil(loss_max/OM*1.1)*OM

        self.root.ax[4].set_xlim((0, tlim ))
        self.root.ax[4].set_ylim((0, loss_lim ))

        if is_optimization_update:
            h["title"].set_text(f"OPTIMIZED Shot {shot_id} - loss = {distance_dev['total']:.3f}")
        else:
            h["title"].set_text(f"Shot {shot_id} - loss = {distance_dev['total']:.3f}")

        # plt.draw()
        self.root.canvas.draw()  # Update the figure display
        # Only refresh existing canvas
        self.root.canvas.draw_idle()  # Ensure the canvas is refreshed
        return self.sim_env.system

    def toggle_markers(self, ball_color):
        """Toggle markers for a specific ball color."""
        self.marker_visible[ball_color] = not self.marker_visible[ball_color]
        self.update_plot()
    
    def toggle_all_markers(self):
        """Toggle all ball markers on/off."""
        # Check if any markers are currently visible
        any_visible = any(self.marker_visible.values())
        
        # If any are visible, turn all off; otherwise turn all on
        new_state = not any_visible
        
        for ball_color in self.marker_visible:
            self.marker_visible[ball_color] = new_state
        
        self.update_plot()

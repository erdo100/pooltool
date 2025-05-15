import json
import pickle
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import matplotlib.pyplot as plt
import mplcursors  # For interactive cursor functionality
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants
BALL_COLORS = {1: "white", 2: "yellow", 3: "red"}  # Ball 1 is now white
BALL_DIAMETER = 0.0615  # in mm


class BilliardDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Billiard Shot Viewer")

        # Create a frame to hold both buttons side by side
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        # File Selection Button
        self.load_button = tk.Button(
            button_frame, text="Load Data File", command=self.load_data
        )
        self.load_button.pack(side=tk.LEFT)

        # Save Data Button
        self.save_button = tk.Button(
            button_frame, text="Save Data", command=self.save_data
        )
        self.save_button.pack(side=tk.LEFT)

        # Frame for Listbox and Scrollbar
        self.listbox_frame = tk.Frame(root)
        self.listbox_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Shot List Table with Scrollbar
        self.shot_listbox = tk.Listbox(
            self.listbox_frame, selectmode=tk.BROWSE, height=20, width=30
        )
        self.shot_listbox.pack(side=tk.LEFT, fill=tk.Y)

        # Add a vertical scrollbar
        self.scrollbar = tk.Scrollbar(self.listbox_frame, orient=tk.VERTICAL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Link the scrollbar to the listbox
        self.shot_listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.shot_listbox.yview)

        # Bind selection event to update plot
        self.shot_listbox.bind("<<ListboxSelect>>", self.update_plot)

        # Initialize table width before setting up axes
        self.table_width = 2.84  # Default table width

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(10, 5))  # Ensure 2:1 aspect ratio
        self._setup_axes()  # Set up axes, grid, and background
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Enable cursor interaction
        self.cursor = mplcursors.cursor(hover=True)

        # Bind click event to deselect old selections
        self.fig.canvas.mpl_connect("button_press_event", self.on_plot_click)

        # Bind 'd' key to delete shot
        self.root.bind("d", self.delete_shot)

        self.all_shots = []

    def _setup_axes(self):
        """Configure axis limits, ticks, grid, and background"""
        self.ax.set_xlim(0, self.table_width)
        self.ax.set_ylim(0, self.table_width / 2)  # Ensure 2:1 aspect ratio
        self.ax.set_xticks(np.linspace(0, self.table_width, 9))  # 9 ticks for x-axis
        self.ax.set_yticks(
            np.linspace(0, self.table_width / 2, 5)
        )  # 5 ticks for y-axis
        self.ax.set_xticklabels([])  # Remove x-axis tick labels
        self.ax.set_yticklabels([])  # Remove y-axis tick labels
        self.ax.grid(True, linestyle="--", alpha=0.6)  # Add a grid with dashed lines
        self.ax.set_facecolor("lightblue")  # Set background color to light blue

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if not file_path:
            return

        # Load the file
        if file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                self.all_shots = pickle.load(f)
                print(f"Loaded {len(self.all_shots)} shots from {file_path}.")

        # Sort shots by shotID
        self.all_shots.sort(key=lambda x: x["shotID"])

        # Populate the listbox with shots
        self.shot_listbox.delete(0, tk.END)
        for i, shot in enumerate(self.all_shots):
            self.shot_listbox.insert(tk.END, f"{shot['shotID']}: {shot['filename']}")

        self.shot_listbox.focus_set()  # Give the listbox keyboard focus so arrow keys work
        # Select the first item in the listbox and update the plot
        if self.all_shots:
            self.shot_listbox.selection_set(0)
            self.update_plot()

    def update_plot(self, event=None):
        selected_indices = self.shot_listbox.curselection()
        if not selected_indices:
            return

        self.ax.clear()
        self._setup_axes()  # Reapply axis and grid configuration after clear

        # Only show the last selected item
        idx = selected_indices[-1]  # Get the last selected index
        shot = self.all_shots[idx]
        self.ax.set_title(f"Shot ID: {shot['shotID']}")  # Set plot title to Shot ID
        for ball_num, ball_data in shot["balls"].items():
            color = BALL_COLORS[ball_num]
            self.ax.plot(
                ball_data["x"],
                ball_data["y"],
                color=color,
                label=f"Ball {ball_num} (Shot {shot['shotID']})",
                marker="o",
                markersize=2,
            )

            # Plot initial position as a circle with black border
            initial_x, initial_y = ball_data["x"][0], ball_data["y"][0]
            circle = plt.Circle(
                (initial_x, initial_y),
                BALL_DIAMETER / 2,
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
                fill=True,
            )
            self.ax.add_patch(circle)

        self.canvas.draw()

    def on_plot_click(self, event):
        """Deselect old selections in the listbox when clicking on the plot"""
        if event.inaxes == self.ax:  # Check if the click is inside the plot
            self.shot_listbox.selection_clear(0, tk.END)
            self.update_plot()  # Update the plot to reflect the deselection

    def delete_shot(self, event):
        """Delete the selected shot and update the listbox and all_shots variable"""
        selected_indices = self.shot_listbox.curselection()
        if not selected_indices:
            return

        idx = selected_indices[-1]  # Get the last selected index
        del self.all_shots[idx]
        self.shot_listbox.delete(idx)

        # Determine how many items remain in the listbox
        remaining = self.shot_listbox.size()
        if remaining > 0:
            # If idx is beyond the last item, adjust it
            if idx >= remaining:
                idx = remaining - 1
            # Select the next valid item
            self.shot_listbox.selection_set(idx)
            # Trigger the <<ListboxSelect>> event so update_plot is called
            self.shot_listbox.event_generate("<<ListboxSelect>>")
        else:
            # If no shots remain, just clear the plot
            self.ax.clear()
            self.canvas.draw()

    def save_data(self):
        """Save all shots to a new file with the current date"""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("JSON files", "*.json")],
            initialfile=f"billiard_shots_{date_str}",
        )
        if not file_path:
            return

        # Save the file
        if file_path.endswith(".pkl"):
            with open(file_path, "wb") as f:
                pickle.dump(self.all_shots, f)
        elif file_path.endswith(".json"):
            with open(file_path, "w") as f:
                json.dump(self.all_shots, f, indent=4, default=self._convert_to_list)

    def _convert_to_list(self, obj):
        """Helper function to convert NumPy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError("Type not serializable")


if __name__ == "__main__":
    root = tk.Tk()
    viewer = BilliardDataViewer(root)
    root.mainloop()

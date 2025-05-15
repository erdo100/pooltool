import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set backend (Important!)
matplotlib.use("Qt5Agg")  # Or 'TkAgg' if Qt is not available

# Load data (with dummy data for testing)
try:
    samples_df = pd.read_parquet("mein_dataframe.parquet")
except FileNotFoundError:
    print("Error: mein_dataframe.parquet not found. Creating dummy data.")
    samples_df = pd.DataFrame(
        {
            "point": [1, 1, 1, 1, 1, 1],
            "a": [1, 2, 3, 1, 2, 3],
            "b": [4, 5, 6, 7, 8, 9],
            "cut": [10, 11, 12, 13, 14, 15],
        }
    )

# Widgets erstellen (moved up)
xi_dropdown = widgets.Dropdown(options=["cut", "a", "b"], description="Schnittachse:")
cut_slider = widgets.FloatSlider(
    min=samples_df["cut"].min(),
    max=samples_df["cut"].max(),
    step=0.01,
    value=samples_df["cut"].mean(),
    description="Schnittwert:",
)
a_slider = widgets.FloatSlider(
    min=samples_df["a"].min(),
    max=samples_df["a"].max(),
    step=0.01,
    value=samples_df["a"].mean(),
    description="Schnittwert:",
)
b_slider = widgets.FloatSlider(
    min=samples_df["b"].min(),
    max=samples_df["b"].max(),
    step=0.01,
    value=samples_df["b"].mean(),
    description="Schnittwert:",
)


def plot_3d(xi_name, cut_value, ax):  # Pass the axes object
    samples_df_1 = samples_df[samples_df["point"] == 1]

    if not samples_df_1.empty:
        ax.cla()  # Clear the axes
        ax.scatter(
            samples_df_1["a"],
            samples_df_1["b"],
            samples_df_1["cut"],
            c="red",
            marker="x",
            label="Point 1",
            alpha=0.5,
        )

        if xi_name == "cut":
            x_range = np.linspace(samples_df_1["a"].min(), samples_df_1["a"].max(), 100)
            y_range = np.linspace(samples_df_1["b"].min(), samples_df_1["b"].max(), 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.full_like(X, cut_value)
            ax.plot_surface(X, Y, Z, alpha=0.5, color="green")
            ax.set_zlabel("cut")
        elif xi_name == "a":
            y_range = np.linspace(samples_df_1["b"].min(), samples_df_1["b"].max(), 100)
            z_range = np.linspace(
                samples_df_1["cut"].min(), samples_df_1["cut"].max(), 100
            )
            Y, Z = np.meshgrid(y_range, z_range)
            X = np.full_like(Y, cut_value)
            ax.plot_surface(X, Y, Z, alpha=0.5, color="green")
            ax.set_xlabel("a")
        elif xi_name == "b":
            x_range = np.linspace(samples_df_1["a"].min(), samples_df_1["a"].max(), 100)
            z_range = np.linspace(
                samples_df_1["cut"].min(), samples_df_1["cut"].max(), 100
            )
            X, Z = np.meshgrid(x_range, z_range)
            Y = np.full_like(X, cut_value)
            ax.plot_surface(X, Y, Z, alpha=0.5, color="green")
            ax.set_ylabel("b")

        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_title("3D Scatterplot mit Schnittfläche")
        ax.legend()
    else:
        print("Keine Punkte mit point == 1 gefunden.")


# Create figure and axes ONCE
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Initial plot
plot_3d(xi_dropdown.value, cut_slider.value, ax)
plt.show(block=False)  # Show the plot non-blocking


def update_plot(change):
    plot_3d(xi_dropdown.value, globals()[xi_dropdown.value + "_slider"].value, ax)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# Connect widgets and display
xi_dropdown.observe(update_plot, names="value")
cut_slider.observe(update_plot, names="value")
a_slider.observe(update_plot, names="value")
b_slider.observe(update_plot, names="value")

# display(xi_dropdown)
# display(cut_slider)
# display(a_slider)
# display(b_slider)

plt.show()  # Keep the window open

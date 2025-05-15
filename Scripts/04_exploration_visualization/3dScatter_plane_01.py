import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import Dropdown, FloatSlider, VBox

# Load data
shots_df = pd.read_parquet("2_18_shots.parquet")
scatter_df = shots_df[shots_df["point"] == 1]

# Initialize 3D Figure
fig = go.FigureWidget(
    layout=go.Layout(width=1000, height=1000, scene=dict(aspectmode="cube"))
)

fig.add_trace(
    go.Scatter3d(
        x=scatter_df["a"],
        y=scatter_df["b"],
        z=scatter_df["cut"],
        mode="markers",
        marker=dict(size=2, opacity=0.1, color="blue"),
    )
)

# Add invisible surface for initial setup
fig.add_trace(
    go.Surface(visible=True, colorscale="Viridis", opacity=0.5, showscale=True)
)

# Calculate axis ranges
var_ranges = {
    "a": (shots_df["a"].min(), shots_df["a"].max()),
    "b": (shots_df["b"].min(), shots_df["b"].max()),
    "cut": (shots_df["cut"].min(), shots_df["cut"].max()),
}

# Lock axis ranges to data limits
fig.update_layout(
    scene=dict(
        xaxis=dict(range=var_ranges["a"], autorange=False, title="horizontal spin"),
        yaxis=dict(range=var_ranges["b"], autorange=False),
        title="vertical spin",
        zaxis=dict(range=var_ranges["cut"], autorange=False, title="cut angle"),
    )
)

# Widget setup
var_dropdown = Dropdown(
    options=["a", "b", "cut"], value="cut", description="Fixed Variable"
)
value_slider = FloatSlider(
    min=var_ranges["cut"][0],
    max=var_ranges["cut"][1],
    value=np.median(shots_df["cut"]),
    description="Value",
)

# Grid configuration
num_bins = 40  # Reduced for better performance


def update_plane(fixed_var, fixed_value):
    # Determine plane variables
    vars_plane = [v for v in ["a", "b", "cut"] if v != fixed_var]
    x_var, y_var = vars_plane

    # Create bin edges and centers
    x_edges = np.linspace(*var_ranges[x_var], num_bins + 1)
    y_edges = np.linspace(*var_ranges[y_var], num_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers)

    # Calculate density data
    epsilon = 0.025 * (var_ranges[fixed_var][1] - var_ranges[fixed_var][0])
    sliced = shots_df[np.abs(shots_df[fixed_var] - fixed_value) <= epsilon]

    print(epsilon)
    print(sliced)

    H, _, _ = np.histogram2d(
        sliced[x_var], sliced[y_var], bins=(x_edges, y_edges), weights=sliced["point"]
    )
    counts, _, _ = np.histogram2d(sliced[x_var], sliced[y_var], bins=(x_edges, y_edges))

    with np.errstate(divide="ignore", invalid="ignore"):
        contour = np.nan_to_num((H / counts).T)

    # Set 3D coordinates based on fixed variable
    if fixed_var == "a":
        coordinates = {"x": np.full_like(xx, fixed_value), "y": xx, "z": yy}
    elif fixed_var == "b":
        coordinates = {"x": xx, "y": np.full_like(yy, fixed_value), "z": yy}
    else:
        coordinates = {"x": xx, "y": yy, "z": np.full_like(xx, fixed_value)}

    # Update surface plot
    with fig.batch_update():
        fig.data[1].x = coordinates["x"]
        fig.data[1].y = coordinates["y"]
        fig.data[1].z = coordinates["z"]
        fig.data[1].surfacecolor = contour
        fig.data[1].colorscale = "Viridis"


# Widget handlers
def on_dropdown_change(change):
    new_var = change["new"]
    current_min, current_max = var_ranges[new_var]
    value_slider.min = current_min
    value_slider.max = current_max
    value_slider.step = (current_max - current_min) / 100
    value_slider.value = np.median(shots_df[new_var])


def handle_slider_change(change):
    update_plane(var_dropdown.value, value_slider.value)


# Attach the dropdown handler to update slider properties
var_dropdown.observe(on_dropdown_change, names="value")
# Attach the slider handler to update the plane
value_slider.observe(handle_slider_change, names="value")

# Initial update to set the correct slider limits and plane
on_dropdown_change({"new": var_dropdown.value})  # Trigger initial slider setup

# Display UI
display(VBox([var_dropdown, value_slider, fig]))

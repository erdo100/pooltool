# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.2
# ---

# %%
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

# Add scatter plot
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
    go.Surface(visible=True, colorscale="Viridis", opacity=0.8, showscale=True)
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
        xaxis=dict(
            range=var_ranges["a"], autorange=False, title=dict(text="Horizontal Spin")
        ),
        yaxis=dict(
            range=var_ranges["b"], autorange=False, title=dict(text="Vertical Spin")
        ),
        zaxis=dict(
            range=var_ranges["cut"], autorange=False, title=dict(text="Cut Angle")
        ),
    )
)

# Compute 3D density grid for iso-surface
num_bins_3d = 50
x_edges_3d = np.linspace(*var_ranges["a"], num_bins_3d + 1)
y_edges_3d = np.linspace(*var_ranges["b"], num_bins_3d + 1)
z_edges_3d = np.linspace(*var_ranges["cut"], num_bins_3d + 1)

x_centers_3d = (x_edges_3d[:-1] + x_edges_3d[1:]) / 2
y_centers_3d = (y_edges_3d[:-1] + y_edges_3d[1:]) / 2
z_centers_3d = (z_edges_3d[:-1] + z_edges_3d[1:]) / 2

# Compute 3D histograms
H_3d, _ = np.histogramdd(
    shots_df[["a", "b", "cut"]].values,
    bins=(x_edges_3d, y_edges_3d, z_edges_3d),
    weights=shots_df["point"],
)
counts_3d, _ = np.histogramdd(
    shots_df[["a", "b", "cut"]].values, bins=(x_edges_3d, y_edges_3d, z_edges_3d)
)

density_3d = np.zeros_like(H_3d)
np.divide(H_3d, counts_3d, out=density_3d, where=counts_3d != 0)
density_3d = np.nan_to_num(density_3d)

# Create meshgrid and flatten coordinates
xx, yy, zz = np.meshgrid(x_centers_3d, y_centers_3d, z_centers_3d, indexing="ij")
x_flat, y_flat, z_flat = xx.flatten(), yy.flatten(), zz.flatten()
density_flat = density_3d.flatten()

# Add iso-surface trace
density_min, density_max = density_flat.min(), density_flat.max()
initial_density = (density_min + density_max) / 2

fig.add_trace(
    go.Isosurface(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        value=density_flat,
        isomin=initial_density,
        isomax=initial_density,
        surface_count=1,
        colorscale="Viridis",
        opacity=0.5,
        showscale=True,
        caps=dict(x_show=False, y_show=False, z_show=False),
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
density_slider = FloatSlider(
    min=density_min,
    max=density_max,
    value=initial_density,
    step=(density_max - density_min) / 100,
    description="Density Threshold",
)


# Update functions
def update_plane(fixed_var, fixed_value):
    vars_plane = [v for v in ["a", "b", "cut"] if v != fixed_var]
    x_var, y_var = vars_plane

    x_edges = np.linspace(*var_ranges[x_var], num_bins + 1)
    y_edges = np.linspace(*var_ranges[y_var], num_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers)

    epsilon = 0.025 * (var_ranges[fixed_var][1] - var_ranges[fixed_var][0])
    sliced = shots_df[np.abs(shots_df[fixed_var] - fixed_value) <= epsilon]

    H, _, _ = np.histogram2d(
        sliced[x_var], sliced[y_var], bins=(x_edges, y_edges), weights=sliced["point"]
    )
    counts, _, _ = np.histogram2d(sliced[x_var], sliced[y_var], bins=(x_edges, y_edges))

    with np.errstate(divide="ignore", invalid="ignore"):
        contour = np.nan_to_num((H / counts).T)

    if fixed_var == "a":
        coordinates = {"x": np.full_like(xx, fixed_value), "y": xx, "z": yy}
    elif fixed_var == "b":
        coordinates = {"x": xx, "y": np.full_like(yy, fixed_value), "z": yy}
    else:
        coordinates = {"x": xx, "y": yy, "z": np.full_like(xx, fixed_value)}

    with fig.batch_update():
        fig.data[1].x = coordinates["x"]
        fig.data[1].y = coordinates["y"]
        fig.data[1].z = coordinates["z"]
        fig.data[1].surfacecolor = contour


def handle_density_change(change):
    new_value = change["new"]
    with fig.batch_update():
        fig.data[2].isomin = new_value
        fig.data[2].isomax = new_value


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


# Attach observers
var_dropdown.observe(on_dropdown_change, names="value")
value_slider.observe(handle_slider_change, names="value")
density_slider.observe(handle_density_change, names="value")

# Initial update
on_dropdown_change({"new": var_dropdown.value})

# Display UI
display(VBox([var_dropdown, value_slider, density_slider, fig]))

# %%

# %%

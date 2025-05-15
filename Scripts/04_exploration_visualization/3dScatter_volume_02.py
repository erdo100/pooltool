import pandas as pd
import plotly.graph_objects as go
from ipywidgets import Dropdown, FloatSlider, VBox

# Load data
shots_df = pd.read_parquet("shots_dataframe.parquet")
scatter_df = shots_df[shots_df["point"] == 1]

# Create a larger figure
fig = go.FigureWidget(
    layout=go.Layout(width=1200, height=800, scene=dict(aspectmode="cube"))
)

# Add main scatter plot
fig.add_trace(
    go.Scatter3d(
        x=scatter_df["a"],
        y=scatter_df["b"],
        z=scatter_df["cut"],
        mode="markers",
        marker=dict(size=3, opacity=0.1, color="blue"),
        name="All Points",
    )
)

# Add highlighted points trace
fig.add_trace(
    go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="markers",
        marker=dict(size=4, opacity=0.8, color="red"),
        name="Highlighted",
    )
)

# Calculate axis ranges
var_ranges = {
    "a": (shots_df["a"].min(), shots_df["a"].max()),
    "b": (shots_df["b"].min(), shots_df["b"].max()),
    "cut": (shots_df["cut"].min(), shots_df["cut"].max()),
}

# Lock axis ranges
fig.update_layout(
    scene=dict(
        xaxis=dict(range=var_ranges["a"]),
        yaxis=dict(range=var_ranges["b"]),
        zaxis=dict(range=var_ranges["cut"]),
    )
)

# Widget setup
var_dropdown = Dropdown(options=["a", "b", "cut"], value="cut", description="Axis")
range_slider = FloatSlider(description="Center", continuous_update=False)


def update_slider_params(var):
    v_min, v_max = var_ranges[var]
    v_range = v_max - v_min
    window_size = 0.1 * v_range

    range_slider.min = v_min + window_size / 2
    range_slider.max = v_max - window_size / 2
    range_slider.value = (v_min + v_max) / 2
    range_slider.step = window_size / 20


def update_highlight(var, center):
    v_min, v_max = var_ranges[var]
    v_range = v_max - v_min
    window_size = 0.1 * v_range

    lower = center - window_size / 2
    upper = center + window_size / 2

    mask = (shots_df[var] >= lower) & (shots_df[var] <= upper)
    highlighted = shots_df[mask & (shots_df["point"] == 1)]

    with fig.batch_update():
        fig.data[1].x = highlighted["a"]
        fig.data[1].y = highlighted["b"]
        fig.data[1].z = highlighted["cut"]


def on_dropdown_change(change):
    var = change["new"]
    update_slider_params(var)
    update_highlight(var, range_slider.value)


def on_slider_change(change):
    update_highlight(var_dropdown.value, range_slider.value)


var_dropdown.observe(on_dropdown_change, "value")
range_slider.observe(on_slider_change, "value")

# Initial setup
update_slider_params(var_dropdown.value)
update_highlight(var_dropdown.value, range_slider.value)

# Display UI
display(VBox([var_dropdown, range_slider, fig]))

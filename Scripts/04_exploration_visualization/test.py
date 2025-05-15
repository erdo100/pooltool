import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.widgets import Slider

# Load data
shots_df = pd.read_parquet("shots_dataframe.parquet")

data = shots_df[["a", "b", "cut"]].to_numpy()
values = shots_df["point"].to_numpy()


def compute_density(d1_range, d2_range, fixed_dims):
    d3_fixed, d4_fixed = fixed_dims
    density = np.zeros((len(d1_range), len(d2_range)))

    cov_matrix = np.diag([0.05**2, 0.05**2, 5**2, 5**2])

    for i, d1 in enumerate(d1_range):
        for j, d2 in enumerate(d2_range):
            # Compute Gaussian weights
            weights = stats.multivariate_normal.pdf(
                data, mean=[d1, d2, d3_fixed, d4_fixed], cov=cov_matrix
            )

            # Weighted average of binary values
            density[i, j] = np.sum(weights * values) / np.sum(weights)

    return density


# Define grid resolution
num_steps = 21
linspace = np.linspace(0, 1, num_steps)
d1_fixed, d2_fixed = 0.5, 0.5  # Initial fixed values
d1_range, d2_range = np.meshgrid(linspace, linspace)


def plot_density(d3_fixed, d4_fixed):
    density = compute_density(linspace, linspace, (d3_fixed, d4_fixed))
    plt.clf()
    plt.contourf(d1_range, d2_range, density, levels=20, cmap="viridis")
    plt.colorbar(label="Density")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"Density Plot for Fixed d3={d3_fixed}, d4={d4_fixed}")
    plt.draw()


# Interactive plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
plot_density(d1_fixed, d2_fixed)

ax_d3 = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_d4 = plt.axes([0.25, 0.05, 0.65, 0.03])
slider_d3 = Slider(ax_d3, "d3", 0, 1, valinit=d1_fixed)
slider_d4 = Slider(ax_d4, "d4", 0, 1, valinit=d2_fixed)


def update(val):
    plot_density(slider_d3.val, slider_d4.val)


slider_d3.on_changed(update)
slider_d4.on_changed(update)

plt.show()

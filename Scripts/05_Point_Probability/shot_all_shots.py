import multiprocessing
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.widgets import RadioButtons, Slider
from numba import prange
from SALib.sample import sobol
from scipy.stats import norm
from sobol_seq import i4_sobol_generate
from threecushion_shot import BilliardEnv


# Define these at the top level for pickling
def init_worker():
    global worker_env
    from threecushion_shot import BilliardEnv  # Import inside to avoid pickling issues

    worker_env = BilliardEnv()


def process_sample(sample):
    a, b, cut = sample
    try:
        # Fixed parameters directly in the function
        worker_env.prepare_new_shot(
            (0.5275, 0.71), (0.71, 0.71), (0.71, 2.13), a, b, 3.5, cut, 0
        )
        return worker_env.simulate_shot()
    except Exception as e:
        print(f"Error processing sample {sample}: {e}")
        return -1


def run_optimized_simulation(resolution):
    problem = {
        "num_vars": 3,
        "names": ["side", "vert", "cut"],
        "bounds": [[-0.5, 0.5], [-0.5, 0.5], [-89, 89]],
    }

    runs = resolution * (2 * 3 + 2)
    method = "Sobol"

    if method == "Sobol":
        samples = i4_sobol_generate(problem["num_vars"], resolution)
        samples = samples * (
            np.array([0.5, 0.5, 89]) - np.array([-0.5, -0.5, -89])
        ) + np.array([-0.5, -0.5, -89])
    elif method == "UniformRandom":
        samples = np.random.uniform(
            low=[-0.5, -0.5, -89], high=[0.5, 0.5, 89], size=(runs, 3)
        )

    shots_df = pd.DataFrame(samples, columns=problem["names"])
    samples_list = [tuple(row) for row in samples]

    with Pool(processes=multiprocessing.cpu_count(), initializer=init_worker) as pool:
        results = pool.map(process_sample, samples_list, chunksize=1000)

    shots_df["point"] = results

    shots_df.to_parquet("2_17_shots_optimized.parquet")
    return shots_df


def calculate_probability_map_old(
    df: pd.DataFrame,
    variables: list,
    result_col: str,
    stddev_dict: dict,
    steps: float,
) -> tuple:
    """
    Calculate the probability density for each point in the n-dimensional space.
    Returns:
    tuple: (grid_axes, density_array)
    """
    # Validate inputs
    missing_vars = [var for var in variables if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing variables: {missing_vars}")
    if result_col not in df.columns:
        raise ValueError(f"Result column '{result_col}' not found.")
    missing_stddev = [var for var in variables if var not in stddev_dict]
    if missing_stddev:
        raise ValueError(f"Missing std deviations for: {missing_stddev}")

    # Precompute variable arrays for faster access
    var_arrays = {var: df[var].values for var in variables}
    results = df[result_col].values

    # Prepare grid axes
    grid_axes = []
    for var in variables:
        min_val = df[var].min()
        max_val = df[var].max()
        range_var = max_val - min_val

        if range_var == 0:
            grid = np.array([min_val])
        else:
            step = range_var / steps
            grid = np.arange(min_val, max_val + step, step)

        grid_axes.append(grid)

    # Create n-dimensional grid
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    total_probability = np.zeros_like(mesh[0], dtype=np.float64)

    # Loop over all mesh grid points
    for idx in np.ndindex(mesh[0].shape):
        # Compute combined mask for 3 sigma ranges
        combined_mask = np.ones(len(df), dtype=bool)
        for vari, var in enumerate(variables):
            m = mesh[vari][idx]
            stddev = stddev_dict[var]
            var_data = var_arrays[var]
            var_mask = (var_data >= (m - 3 * stddev)) & (var_data <= (m + 3 * stddev))
            combined_mask &= var_mask

        masked_indices = np.where(combined_mask)[0]

        if len(masked_indices) == 0:
            P = np.zeros_like(results)
        else:
            P = np.zeros_like(results)
            product = np.ones(len(masked_indices), dtype=np.float64)
            for vari, var in enumerate(variables):
                m = mesh[vari][idx]
                stddev = stddev_dict[var]
                var_data = var_arrays[var]
                p = norm.pdf(var_data[masked_indices], loc=m, scale=stddev)
                product *= p
            P[masked_indices] = product

        # Normalize the probability
        sum_P = np.sum(P)
        if sum_P != 0:
            P /= sum_P

        total_probability[idx] = np.sum(P * results)

    # Save to Parquet
    grid_coords = {f"{var}_grid": grid.flatten() for var, grid in zip(variables, mesh)}
    grid_coords["total_probability"] = total_probability.flatten()
    pd.DataFrame(grid_coords).to_parquet("total_probability.parquet")

    return grid_axes, total_probability


def calculate_probability_map(
    df: pd.DataFrame,
    variables: list,
    result_col: str,
    stddev_dict: dict,
    steps: float,
    filename: str,
) -> tuple:
    # Input validation (same as before)
    missing_vars = [var for var in variables if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing variables: {missing_vars}")
    if result_col not in df.columns:
        raise ValueError(f"Result column '{result_col}' not found.")
    missing_stddev = [var for var in variables if var not in stddev_dict]
    if missing_stddev:
        raise ValueError(f"Missing std deviations for: {missing_stddev}")

    # Convert DataFrame columns to numpy arrays for Numba compatibility
    var_arrays = {var: df[var].values.astype(np.float64) for var in variables}
    results = df[result_col].values.astype(np.float64)
    stddevs = np.array([stddev_dict[var] for var in variables], dtype=np.float64)

    # Prepare grid axes (same logic but store as numpy arrays)
    grid_axes = []
    grid_shapes = []
    for var in variables:
        min_val = df[var].min()
        max_val = df[var].max()
        range_var = max_val - min_val

        if range_var == 0:
            grid = np.array([min_val], dtype=np.float64)
        else:
            step = range_var / steps
            grid = np.arange(min_val, max_val + step, step, dtype=np.float64)

        grid_axes.append(grid)
        grid_shapes.append(len(grid))

    # Create meshgrid and flatten for parallel processing
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid_points = np.stack([m.flatten() for m in mesh], axis=1)
    total_probability = np.zeros(grid_points.shape[0], dtype=np.float64)

    # Convert variables to 2D array (n_vars x n_samples)
    data_matrix = np.stack([var_arrays[var] for var in variables], axis=0)

    # Precompute constants for PDF calculation
    sqrt_2pi = np.sqrt(2 * np.pi)
    stddevs_squared = stddevs**2
    coefficients = 1 / (stddevs * sqrt_2pi)

    # Call Numba-optimized function
    _compute_probabilities(
        data_matrix,
        results,
        grid_points,
        total_probability,
        stddevs,
        coefficients,
        stddevs_squared,
        *grid_shapes,
    )

    # Reshape back to original grid shape
    total_probability = total_probability.reshape(*grid_shapes)

    # Save to Parquet (same as before)
    grid_coords = {f"{var}_grid": grid.flatten() for var, grid in zip(variables, mesh)}
    grid_coords["total_probability"] = total_probability.flatten()
    pd.DataFrame(grid_coords).to_parquet(filename)

    return grid_axes, total_probability


@nb.njit(nogil=True, fastmath=True, parallel=True)
def _compute_probabilities(
    data_matrix: np.ndarray,
    results: np.ndarray,
    grid_points: np.ndarray,
    total_probability: np.ndarray,
    stddevs: np.ndarray,
    coefficients: np.ndarray,
    stddevs_squared: np.ndarray,
    *grid_shapes: tuple,
):
    n_vars, n_samples = data_matrix.shape
    n_points = grid_points.shape[0]

    for idx in prange(n_points):
        # Get current grid point coordinates
        current_point = grid_points[idx]

        # Compute 3σ ranges mask
        mask = np.ones(n_samples, dtype=np.bool_)
        for var_idx in range(n_vars):
            stddev = stddevs[var_idx]
            lower = current_point[var_idx] - 3 * stddev
            upper = current_point[var_idx] + 3 * stddev
            var_data = data_matrix[var_idx]

            # Vectorized mask computation
            mask &= (var_data >= lower) & (var_data <= upper)

        # Get indices of points within 3σ range
        valid_indices = np.where(mask)[0]
        n_valid = valid_indices.size

        if n_valid == 0:
            total_probability[idx] = 0.0
            continue

        # Compute probabilities using explicit PDF formula
        log_probs = np.zeros(n_valid, dtype=np.float64)

        for var_idx in range(n_vars):
            m = current_point[var_idx]
            data = data_matrix[var_idx, valid_indices]
            coeff = coefficients[var_idx]
            inv_2sigma2 = 1 / (2 * stddevs_squared[var_idx])

            # Vectorized PDF calculation
            diff = data - m
            log_probs += np.log(coeff) - (diff * diff) * inv_2sigma2

        # Convert from log space to linear
        probs = np.exp(log_probs)
        total_prob = np.sum(probs)

        if total_prob > 0:
            probs /= total_prob
            total_probability[idx] = np.sum(probs * results[valid_indices])
        else:
            total_probability[idx] = 0.0


def load_density_from_parquet(parquet_path: str) -> tuple:
    """
    Load the saved Parquet file and reconstruct grid axes and density array.

    Args:
        parquet_path (str): Path to the saved Parquet file.

    Returns:
        tuple: (grid_axes, density_array)
            - grid_axes: List of 1D arrays for each variable's grid.
            - density_array: N-dimensional numpy array of densities.
    """
    # Read the Parquet file
    df = pd.read_parquet(parquet_path)

    # Extract variable names (assumes grid columns end with "_grid")
    grid_cols = [col for col in df.columns if col.endswith("_grid")]
    variables = [col.replace("_grid", "") for col in grid_cols]

    # Reconstruct grid axes from unique values in each grid column
    grid_axes = []
    for var_grid in grid_cols:
        # Extract unique values and sort them to reconstruct the original grid
        grid = np.sort(df[var_grid].unique())
        grid_axes.append(grid)

    # Reshape the density column into the original n-dimensional array
    shape = [len(axis) for axis in grid_axes]
    total_probability = df["total_probability"].values.reshape(shape)

    return grid_axes, total_probability


# 2. Interactive 3D Visualization with Plotly
def plot_3d_probability_interactive(grid_axes, total_probability, variable_labels=None):
    """
    Visualize the 3D density using Plotly Volume.

    Args:
        grid_axes (list): List of 1D arrays (grid coordinates for each variable).
        density (np.ndarray): 3D density array.
        variable_labels (list): Optional labels for axes (e.g., ["v1", "v2", "v3"]).
    """
    # Create 3D coordinate grids
    X, Y, Z = np.meshgrid(*grid_axes, indexing="ij")

    # Flatten coordinates and total_probability for Plotly
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    P_flat = total_probability.flatten()

    # Default axis labels
    if variable_labels is None:
        variable_labels = [f"Variable {i + 1}" for i in range(len(grid_axes))]

    # Create Volume plot
    fig = go.Figure(
        data=go.Volume(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            value=P_flat,
            isomin=0.1 * P_flat.max(),
            isomax=P_flat.max(),
            opacity=0.1,
            surface_count=20,
            colorscale="viridis",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title=variable_labels[0],
            yaxis_title=variable_labels[1],
            zaxis_title=variable_labels[2],
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.show()


def standalone_slice_viewer(grid_axes, total_probability, variables):
    """Interactive 2D slice viewer with dynamic slider labels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=0.3)

    # Initial setup
    var_idx = 0
    slice_idx = len(grid_axes[var_idx]) // 2
    x, y = grid_axes[1], grid_axes[2]
    slice_data = total_probability[slice_idx, :, :]
    contour = ax.contourf(x, y, slice_data.T, cmap="viridis", levels=20)
    cbar = fig.colorbar(contour, ax=ax)
    ax.set_xlabel(variables[1])
    ax.set_ylabel(variables[2])

    # Slider axis
    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label=f"{variables[var_idx]}",  # Initialize with current variable
        valmin=grid_axes[var_idx].min(),
        valmax=grid_axes[var_idx].max(),
        valinit=grid_axes[var_idx][slice_idx],
        valstep=np.diff(grid_axes[var_idx])[0],
    )

    # Radio buttons
    ax_radio = plt.axes([0.8, 0.1, 0.15, 0.15])
    radio = RadioButtons(ax_radio, labels=variables, active=0)

    def update_slice(val):
        """Update the plot with new slice position."""
        nonlocal var_idx
        idx = np.abs(grid_axes[var_idx] - val).argmin()

        if var_idx == 0:
            data = total_probability[idx, :, :]
            x, y = grid_axes[1], grid_axes[2]
            xl, yl = variables[1], variables[2]
        elif var_idx == 1:
            data = total_probability[:, idx, :]
            x, y = grid_axes[0], grid_axes[2]
            xl, yl = variables[0], variables[2]
        else:
            data = total_probability[:, :, idx]
            x, y = grid_axes[0], grid_axes[1]
            xl, yl = variables[0], variables[1]

        ax.clear()
        ax.contourf(x, y, data.T, cmap="viridis", levels=20)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"Slice at {variables[var_idx]} = {grid_axes[var_idx][idx]:.2f}")
        fig.canvas.draw_idle()

    def select_variable(label):
        """Update slider and label when variable changes."""
        nonlocal var_idx
        var_idx = variables.index(label)

        # Update slider properties
        slider.valmin = grid_axes[var_idx].min()
        slider.valmax = grid_axes[var_idx].max()
        slider.valstep = np.diff(grid_axes[var_idx])[0]
        slider.set_val(grid_axes[var_idx][len(grid_axes[var_idx]) // 2])

        # Update slider label to show current variable
        slider.label.set_text(variables[var_idx])  # Fix: Update label text
        slider.ax.set_xlim(slider.valmin, slider.valmax)
        fig.canvas.draw_idle()

        # Update plot
        update_slice(slider.val)

    # Connect events
    slider.on_changed(update_slice)
    radio.on_clicked(select_variable)

    plt.show()


if __name__ == "__main__":
    runsims = True
    calculate_density = False
    resolution = 2**14
    filebase = "2_18_shots"
    filename_results = filebase + "_results.parquet"
    filename_probability = filebase + "_probability.parquet"

    if runsims == True:
        # start timing
        start = time.time()

        if False:
            shots_df = run_optimized_simulation(resolution)

            shots_df.to_parquet(filename_results)

        else:
            # Define the problem: 4 variables, each with 3 levels
            problem = {
                "num_vars": 3,
                "names": ["side", "vert", "cut"],  # Names for the variables
                # Bounds for each variable
                "bounds": [
                    [-0.5, 0.5],  # a
                    [-0.5, 0.5],  # b
                    # [2, 7],      # velocity
                    [-89, 89],  # cut angle
                ],
            }

            runs = resolution * (2 * 3 + 2)
            method = "Sobol"
            # method = 'UniformRandom'

            if method == "Sobol":
                # Generate the Sobol design using Sobol sampling
                samples = sobol.sample(problem, resolution)

            elif method == "UniformRandom":
                # Generate random uniform samples within the bounds for each variable
                samples = np.random.uniform(
                    low=[-0.5, -0.5, -89],  # Lower bounds for each variable
                    high=[0.5, 0.5, 89],  # Upper bounds for each variable
                    size=(runs, problem["num_vars"]),  # Number of samples and variables
                )

            # Convert to a pandas DataFrame for easier inspection
            shots_df = pd.DataFrame(samples, columns=problem["names"])

            # Print the size (number of samples and number of variables)
            print("Size of the samples array:", samples.shape)

            env = BilliardEnv()

            for i in range(runs):
                a = shots_df.loc[i, "side"]
                b = shots_df.loc[i, "vert"]
                v = 3.5
                cut = shots_df.loc[i, "cut"]
                theta = 0

                ball1xy = (0.5275, 0.71)
                ball2xy = (0.71, 0.71)
                ball3xy = (0.71, 2.13)

                env.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, v, cut, theta)

                point = env.simulate_shot()
                shots_df.at[i, "point"] = point

                if (i + 1) % 200000 == 0:
                    print((time.time() - start) / 3600, "h: ", i, " runs")
                    filtered_df = shots_df[shots_df["point"] == 1]
                    # Create an interactive 3D scatter plot
                    fig = px.scatter_3d(
                        filtered_df,
                        x="side",
                        y="cut",
                        z="vert",
                        title="Total runs=" + str(i),
                    )
                    fig.show()

            # Speichern im Parquet-Format
            shots_df.to_parquet("2_12_shots.parquet")
            print("Dataframe saved to parquet file")

        # print runtime
        print("Runtime of run_optimized_simulation is", time.time() - start)

    if calculate_density == True:
        # Laden aus der Parquet-Datei
        shots_df = pd.read_parquet(filename_results)

        # start timing
        start = time.time()

        # Compute density
        grid_axes, total_probability = calculate_probability_map(
            df=shots_df,
            variables=["side", "vert", "cut"],
            result_col="point",
            stddev_dict={"side": 0.02, "vert": 0.02, "cut": 3},
            steps=100,
            filename=filename_probability,
        )

        # print runtime
        print("Runtime of calculate_probability_map is", time.time() - start)

    else:
        # Load density data from file:
        grid_axes, total_probability = load_density_from_parquet(filename_probability)

    variables = ["side spin", "vertical spin", "cut angle"]

    # Visualize interactively
    plot_3d_probability_interactive(
        grid_axes=grid_axes,
        total_probability=total_probability,
        variable_labels=variables,  # Optional: ["Temperature", "Pressure", "Velocity"]
    )

    standalone_slice_viewer(grid_axes, total_probability, variables)

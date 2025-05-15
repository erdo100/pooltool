import multiprocessing
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from matplotlib.widgets import RadioButtons, Slider
from numba import prange
from sobol_seq import i4_sobol_generate
from threecushion_shot import BilliardEnv


# Define these at the top level for pickling
def init_worker():
    global worker_env
    worker_env = BilliardEnv()


def process_sample(sample):
    a, b, cut = sample
    try:
        # Fixed parameters directly in the function
        worker_env.prepare_new_shot(
            (0.5, 1.0), (1.1, 1.0), (0.08, 0.5), a, b, cut, 3.0, 0
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

    # Extract bounds for scaling
    lower_bounds = np.array([param[0] for param in problem["bounds"]])
    upper_bounds = np.array([param[1] for param in problem["bounds"]])

    if method == "Sobol":
        samples = i4_sobol_generate(problem["num_vars"], resolution)
        # Dynamic scaling using problem bounds
        samples = samples * (upper_bounds - lower_bounds) + lower_bounds
    elif method == "UniformRandom":
        # Dynamic parameter ranges for uniform sampling
        samples = np.random.uniform(
            low=lower_bounds, high=upper_bounds, size=(runs, problem["num_vars"])
        )
    print("resolution:", resolution)
    print("Shape:", samples.shape)
    shots_df = pd.DataFrame(samples, columns=problem["names"])
    samples_list = [tuple(row) for row in samples]

    with Pool(processes=multiprocessing.cpu_count(), initializer=init_worker) as pool:
        results = pool.map(process_sample, samples_list, chunksize=1000)

    shots_df["point"] = results

    return shots_df


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


def standalone_slice_viewer(grid_axes, total_probability, variables):
    """Interactive 2D slice viewer with keyboard controls."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=0.3)

    # Initial setup
    var_idx = 0
    slice_idx = len(grid_axes[var_idx]) // 2
    x, y = grid_axes[1], grid_axes[2]
    slice_data = total_probability[slice_idx, :, :]

    # Initial plot elements
    contour = ax.contourf(x, y, slice_data.T, cmap="viridis", levels=20)
    cbar = fig.colorbar(contour, ax=ax)
    max_val_initial = slice_data.max()
    j_max_initial, k_max_initial = np.unravel_index(
        slice_data.argmax(), slice_data.shape
    )
    scatter = ax.scatter(
        x[j_max_initial], y[k_max_initial], color="red", marker="*", s=100
    )
    text = ax.text(
        x[j_max_initial] + 0.02 * (x.ptp()),
        y[k_max_initial],
        f"Max: {max_val_initial:.2f}",
        color="red",
        va="center",
        ha="left",
    )

    ax.set_xlabel(variables[1])
    ax.set_ylabel(variables[2])

    # Slider setup
    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label=variables[var_idx],
        valmin=grid_axes[var_idx].min(),
        valmax=grid_axes[var_idx].max(),
        valinit=grid_axes[var_idx][slice_idx],
        valstep=np.diff(grid_axes[var_idx])[0],
    )

    # Radio buttons setup
    ax_radio = plt.axes([0.8, 0.1, 0.15, 0.15])
    radio = RadioButtons(ax_radio, labels=variables, active=0)

    def update_slice(val):
        """Update plot elements with proper axis scaling."""
        nonlocal var_idx, contour, scatter, text
        idx = np.abs(grid_axes[var_idx] - val).argmin()

        # Get new data slice
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

        # Clear previous elements
        for coll in contour.collections:
            coll.remove()
        scatter.remove()
        text.remove()

        # Create new plot elements with updated axis limits
        contour = ax.contourf(
            x, y, data.T if var_idx != 2 else data, cmap="viridis", levels=20
        )
        cbar.mappable = contour
        cbar.update_normal(contour)

        # Set axis limits to match current grid axes
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        # Update max value annotation
        max_val = data.max()
        j_max, k_max = np.unravel_index(data.argmax(), data.shape)
        x_coord = x[j_max] if var_idx != 2 else x[k_max]
        y_coord = y[k_max] if var_idx != 2 else y[j_max]

        scatter = ax.scatter(x_coord, y_coord, color="red", marker="*", s=100)
        text = ax.text(
            x_coord + 0.02 * (x.ptp()),
            y_coord,
            f"Max: {max_val:.2f}",
            color="red",
            va="center",
            ha="left",
        )

        # Update labels and title
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"Slice at {variables[var_idx]} = {grid_axes[var_idx][idx]:.2f}")
        fig.canvas.draw_idle()

    def select_variable(label):
        """Handle variable selection changes with full slider reset."""
        nonlocal var_idx
        var_idx = variables.index(label)

        # Update slider configuration
        current_var_grid = grid_axes[var_idx]
        slider.valmin = current_var_grid.min()
        slider.valmax = current_var_grid.max()
        slider.valstep = np.diff(current_var_grid)[0]
        slider.set_val(current_var_grid[len(current_var_grid) // 2])

        # Update slider track visualization
        slider.ax.set_xlim(slider.valmin, slider.valmax)
        slider.label.set_text(variables[var_idx])

        # Force UI refresh
        fig.canvas.draw_idle()

        # Trigger plot update
        update_slice(slider.val)

    def on_key_press(event):
        """Handle keyboard events for slider control."""
        if event.key in ["left", "right"]:
            current_val = slider.val
            step = slider.valstep

            if event.key == "left":
                new_val = current_val - step
            else:
                new_val = current_val + step

            # Keep within bounds
            new_val = np.clip(new_val, slider.valmin, slider.valmax)
            slider.set_val(new_val)

    # Connect UI elements
    slider.on_changed(update_slice)
    radio.on_clicked(select_variable)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    plt.show()


if __name__ == "__main__":
    runsims = True
    calculate_density = True
    resolution = 2**10
    filebase = "short_angle_01"
    filename_results = filebase + "_results.parquet"
    filename_probability = filebase + "_probability.parquet"

    if runsims == True:
        # start timing
        start = time.time()

        shots_df = run_optimized_simulation(resolution)

        shots_df.to_parquet(filename_results)

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
            stddev_dict={"side": 0.025, "vert": 0.025, "cut": 3},
            steps=150,
            filename=filename_probability,
        )

        # print runtime
        print("Runtime of calculate_probability_map is", time.time() - start)

    else:
        # Load density data from file:
        grid_axes, total_probability = load_density_from_parquet(filename_probability)

    variables = ["side spin", "vertical spin", "cut angle"]

    standalone_slice_viewer(grid_axes, total_probability, variables)

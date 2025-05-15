import numpy as np
import matplotlib.pyplot as plt
import pooltool as pt
from scipy.optimize import differential_evolution
from billiardenv import BilliardEnv
from helper_funcs import get_ball_positions, open_shotfile, interpolate_simulated_to_actual, evaluate_loss
import time
import pickle
from tkinter import filedialog


class Parameters:
    def __init__(self):
        # Initialize with default values and limits (same as before)
        self.value = {
            'shot_a': 0.0,
            'shot_b': 0.0,
            'shot_phi': 0.0,
            'shot_v': 3.0,
            'shot_theta': 0.0,
            'physics_ballball_a': 0.01,
            'physics_ballball_b': 0.11,
            'physics_ballball_c': 1.1,
            'physics_u_slide': 0.2,
            'physics_u_roll': 0.005,
            'physics_u_sp_prop': 0.5,
            'physics_e_ballball': 0.95,
            'physics_e_cushion': 0.9,
            'physics_f_cushion': 0.15,
            'physics_h_cushion': 0.037
        }
        self.limits = {
            'shot_a': (-0.2, 0.2),
            'shot_b': (-0., 0.5),
            'shot_phi': (100.0, 112.0),
            'shot_v': (2.5, 4.5),
            'shot_theta': (0.0, 15.0),

            # Alciatori Ball-Ball hit model parameters
            # Friction fit curve u_b = a + b * exp(-c * v_rel) used in David Alciatore's TP A-14
            'physics_ballball_a': (0.0, 0.2),
            'physics_ballball_b': (0, 0.001),
            'physics_ballball_c': (0.0, 5.0),

            # Physics parameters
            'physics_u_slide': (0.1, 0.3),
            'physics_u_roll': (0.004, 0.006),
            'physics_u_sp_prop': (0.2, 1.0),
            'physics_e_ballball': (0.5, 1.0),
            'physics_e_cushion': (0.5, 1.0),
            'physics_f_cushion': (0.1, 0.3),
            'physics_h_cushion': (0.035, 0.039)
        }

def loss_function(scaled_params, shot_actual, sim_env, param_limits):
    
    # Unscale parameters
    params_np = unscale_params(scaled_params, param_limits)
    
    params = Parameters()
    for i, name in enumerate(params.value.keys()):
        params.value[name] = params_np[i]
    sim_env.prepare_new_shot(params)

    sim_env.simulate_shot()

    loss = evaluate_loss(sim_env, shot_actual)
    
    return loss

def scale_params(params_np, param_limits):
    """Scale parameters to [0, 1] based on bounds"""
    scaled = np.zeros_like(params_np)
    for i, (low, high) in enumerate(param_limits):
        scaled[i] = (params_np[i] - low) / (high - low)
    return scaled

def unscale_params(scaled_params, param_limits):
    """Convert scaled parameters back to original ranges"""
    original = np.zeros_like(scaled_params)
    for i, (low, high) in enumerate(param_limits):
        original[i] = scaled_params[i] * (high - low) + low
    return original

def callback_fn(xk, convergence):
    """Updates subplots for each parameter at each iteration."""
    global parameter_history, loss_history
    
    best_params = unscale_params(xk, param_limits)
    loss = loss_function(xk, shot_actual, sim_env, param_limits)
    
    if len(loss_history) == 0 or loss < min(loss_history):

        parameter_history.append(best_params)
        loss_history.append(loss)
        # Format each parameter in best_params to 5 decimal places
        formatted_params = ", ".join(f"{param:.5f}" for param in best_params)

        # Print the iteration, formatted parameters, and convergence
        print(f"Iteration {len(parameter_history)} - Best Params: [{formatted_params}], Loss: {loss:.5f}, Convergence: {convergence:.5f}")

        # Plot the loss history and parameter history in a single figure
        plt.figure(1)
        plt.clf()
        
        # Number of parameters + 1 (for the loss plot)
        num_params = len(param_limits)
        rows = int(np.ceil(np.sqrt(num_params + 1)))
        cols = int(np.ceil((num_params + 1) / rows))
        
        # Plot loss history
        plt.subplot(rows, cols, 1)
        plt.plot(loss_history, 'b-')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title('Loss Function History')
        plt.grid(True)
        
        
        # Plot parameter history
        for i, (name, (low, high)) in enumerate(zip(params.value.keys(), param_limits)):
            plt.subplot(rows, cols, i + 2)
            param_values = [p[i] for p in parameter_history]
            plt.plot(param_values, 'r-')
            plt.title(name)

            plt.grid(True)
            plt.ylim(low, high)  # Set y-axis limits to parameter bounds
        
        plt.tight_layout()
        plt.draw()  # Update the figure
        plt.pause(0.01)  # Pause to update the plot

if __name__ == "__main__":
    # Initialize
    params = Parameters()
    sim_env = BilliardEnv()
    file_path = "E:/PYTHON_PROJECTS/pooltool/Parameter_identification/20221225_2_Match_Ersin_Cemal.pkl"
    shots_actual = open_shotfile()

    
    # Global list to store parameter values and loss at each step
    parameter_history = []
    loss_history = []

    # Setup for first shot
    shot_actual = shots_actual[50]
    sim_env.balls_xy_ini, ball_ids, sim_env.ball_cols, cueball_phi = get_ball_positions(shot_actual)
    
    # Optimization setup
    initial_params = np.array(list(params.value.values()), dtype=np.float32)
    param_limits = [params.limits[name] for name in params.value.keys()]
    initial_params_scaled = scale_params(initial_params, param_limits)
    bounds = [(0.0, 1.0) for _ in param_limits]  # Scaled bounds [0, 1]

    # Parameters
    max_gen = 1000  # Total generations
    popsize_start = 200  # Start with a high popsize
    popsize_end = 200  # Reduce to a smaller popsize
    mutation_start = 1.0  # Start with a high mutation
    mutation_end = 0.5  # Reduce to a smaller mutation
    recombination_start = 0.9  # Start with a high explotation
    recombination_end = 0.5  # Decrease to a more exploration

    # Predefine figure size
    num_params = len(param_limits)
    rows = int(np.ceil(np.sqrt(num_params + 1)))
    cols = int(np.ceil((num_params + 1) / rows))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Adjust size as needed
    plt.ion()  # Enable interactive mode for real-time updates

    # Initial population
    final_population = None
    # load population from disk
    # choose file from disk
    file_path = filedialog.askopenfilename(title="Select a population file", filetypes=[("Pickle files", "*.pkl")])
    if file_path:
        with open(file_path, 'rb') as f:
            # Load the population from the file
            final_population = pickle.load(f)
            print("Initial population loaded.")
    else:
        # If no file is selected, use the initial population
        print("No file selected. Generating initial population.")
        # Use the initial population as a starting point for the optimization
        # initial_population = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(popsize_start, len(bounds)))

    
    result = None
    for gen in range(max_gen):
        current_popsize = max(1, int(popsize_start - (popsize_start - popsize_end) * (gen / max_gen)))
        mutation_val = mutation_end + (mutation_start - mutation_end) * (1 - gen / max_gen)  # Decreasing mutation
        recombination_val = recombination_start + (recombination_end - recombination_start) * (gen / max_gen)  # Increasing recombination

        print(f"Generation {gen + 1}/{max_gen} - Popsize: {current_popsize}, Mutation: {mutation_val}, Recombination: {recombination_val}")
        # Create initial population
        if final_population is None:
            initial_population = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(current_popsize, len(bounds)))
        else:
            # Adjust the population size
            if current_popsize > len(final_population):
                # Add new random individuals
                additional_individuals = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(current_popsize - len(final_population), len(bounds)))
                initial_population = np.vstack((final_population, additional_individuals))
            else:
                # Remove some individuals (e.g., keep the best ones)
                initial_population = final_population[:current_popsize]


        result = differential_evolution(
            func=loss_function,
            bounds=bounds,
            args=(shot_actual, sim_env, param_limits),
            strategy='rand2bin',
            maxiter=1000,  # Run step-by-step
            mutation=mutation_val,
            recombination=recombination_val,
            popsize=current_popsize,
            updating='deferred',
            callback=callback_fn,
            tol=0.001,
            polish=False,
            workers=24,
            disp=False,
            init=initial_population
        )

        # Save the final population for the next generation
        final_population = result.population
        # save the population as pickle to disk with date and time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"population_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(final_population, f)
        

    # Show the final plot
    plt.ioff() 
    plt.show()
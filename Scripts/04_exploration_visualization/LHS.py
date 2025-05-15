import numpy as np
import pandas as pd
import plotly.express as px
from SALib.sample import sobol
from threecushion_shot import BilliardEnv

# Define the problem: 4 variables, each with 3 levels
problem = {
    "num_vars": 3,
    "names": ["a", "b", "cut"],  # Names for the variables
    # Bounds for each variable
    "bounds": [
        [-0.5, 0.5],  # a
        [-0.5, 0.5],  # b
        # [2, 7],      # velocity
        [-89, 89],  # cut angle
    ],
}


resolution = 2**17
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

# start measure runtime
import time

start = time.time()

for i in range(runs):
    a = shots_df.loc[i, "a"]
    b = shots_df.loc[i, "b"]
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
            filtered_df, x="a", y="cut", z="b", title="Total runs=" + str(i)
        )
        fig.show()


# print runtime
print("Runtime of the program is", time.time() - start)
print("Runtime of the program is", time.time() - start)
# Speichern im Parquet-Format
shots_df.to_parquet("2_18_shots.parquet")
print("Dataframe saved to parquet file")

# Laden aus der Parquet-Datei
shots_df = pd.read_parquet("1e6_shots.parquet")
print(shots_df)

# Filtern der Daten für point == 1
shots_df_1 = shots_df[shots_df["point"] == 1]

# Create an interactive 3D scatter plot
fig = px.scatter_3d(
    shots_df_1, x="a", y="cut", z="b", title="3D Scatter Plot of a, b, cut"
)
fig.show()

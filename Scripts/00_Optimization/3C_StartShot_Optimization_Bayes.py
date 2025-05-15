#! /usr/bin/env python
import time

import GPyOpt
import numpy as np

import pooltool as pt

start_time = time.time()

# We need a table, some balls, and a cue stick
# table = pt.Table.default("billiard")

# Ball Positions
wpos = (0.5275, 0.71)  # White
ypos = (0.71, 0.71)  # Yellow
rpos = (0.71, 2.13)  # Red

# define the properties
u_slide = 0.15
u_roll = 0.005
u_sp_prop = 10 * 2 / 5 / 9
u_ballball = 0.05
e_ballball = 0.95
e_cushion = 0.9
f_cushion = 0.15
grav = 9.81

mball = 0.210
Rball = 61.5 / 1000 / 2

cue_mass = 0.576
cue_len = 1.47
cue_tip_R = 0.022
cue_tip_mass = 0.0000001

# Build a table with default BILLIARD specs
table = pt.Table.default(pt.TableType.BILLIARD)

# create the cue
cue_specs = pt.objects.CueSpecs(
    brand="Predator",
    M=cue_mass,
    length=cue_len,
    tip_radius=cue_tip_R,
    butt_radius=0.02,
    end_mass=cue_tip_mass,
)
cue = pt.Cue(cue_ball_id="white", specs=cue_specs)

# Generate the ball layout from the THREECUSHION GameType using the BILLIARD table
# balls = pt.get_rack(pt.GameType.THREECUSHION, table=table)

# Create balls
wball = pt.Ball.create(
    "white",
    xy=wpos,
    m=mball,
    R=Rball,
    u_s=u_slide,
    u_r=u_roll,
    u_sp_proportionality=u_sp_prop,
    u_b=u_ballball,
    e_b=e_ballball,
    e_c=e_cushion,
    f_c=f_cushion,
    g=grav,
)

yball = pt.Ball.create(
    "yellow",
    xy=ypos,
    m=mball,
    R=Rball,
    u_s=u_slide,
    u_r=u_roll,
    u_sp_proportionality=u_sp_prop,
    u_b=u_ballball,
    e_b=e_ballball,
    e_c=e_cushion,
    f_c=f_cushion,
    g=grav,
)

rball = pt.Ball.create(
    "red",
    xy=rpos,
    m=mball,
    R=Rball,
    u_s=u_slide,
    u_r=u_roll,
    u_sp_proportionality=u_sp_prop,
    u_b=u_ballball,
    e_b=e_ballball,
    e_c=e_cushion,
    f_c=f_cushion,
    g=grav,
)

# Wrap it up as a System
system_template = pt.System(
    table=table,
    balls=(wball, yball, rball),
    cue=cue,
)

# Creates a deep copy of the template
system = system_template.copy()

# shot props
sidespin_stddev = 0.05
vertspin_stddev = 0.05
cuespeed_stddev = 0.2
phi_delta_stddev = 0.1
shotnums = 250

# generate shot props from new mean values
sidespin_delta = np.random.normal(loc=0, scale=sidespin_stddev, size=shotnums)
vertspin_delta = np.random.normal(loc=0, scale=vertspin_stddev, size=shotnums)
cuespeed_delta = np.random.normal(loc=0, scale=cuespeed_stddev, size=shotnums)
phi_delta = np.random.normal(loc=0, scale=phi_delta_stddev, size=shotnums)

phi_base = pt.aim.at_ball(system, "red", cut=0)


def my_function(vars):
    sidespin0, vertspin0, cuespeed0, phi0 = (
        vars[:, 0],
        vars[:, 1],
        vars[:, 2],
        vars[:, 3],
    )

    # Initialize an empty list to store angles
    points = np.zeros(shotnums)
    sidespin = sidespin0 + sidespin_delta
    vertspin = vertspin0 + vertspin_delta
    cuespeed = cuespeed0 + cuespeed_delta
    phi = phi0 + phi_delta

    for i in range(shotnums):
        # Creates a deep copy of the template
        system = system_template.copy()
        points[i] = 0
        # check if shot is outside of squirt limit. If so, no point
        if 0.5**2 >= (
            sidespin.item(i) ** 2 + vertspin.item(i) ** 2
        ):  # This will ensure R^2 - a^2 - b^2 >= 0
            system.cue.set_state(
                a=sidespin.item(i),
                b=vertspin.item(i),
                V0=cuespeed.item(i),
                phi=phi.item(i),
            )

            # Evolve the shot.
            pt.simulate(system, inplace=True)

            # Open up the shot in the GUI
            points[i] = 1 if pt.ruleset.three_cushion.is_point(system) else 0
            pt.show(system)

    print(
        "ss=",
        sidespin0,
        ", vs=",
        vertspin0,
        ", speed=",
        cuespeed0,
        ", phi=",
        phi0,
        ", SUC=",
        np.sum(points) / shotnums * 100,
    )
    return 1 - np.sum(points) / shotnums


# Define the bounds for the optimization variables a and b
bounds = [
    {"name": "sidespin0", "type": "continuous", "domain": (0.0, 0.000000000025)},
    {"name": "vertspin0", "type": "continuous", "domain": (0.2, 0.2000005)},
    {"name": "cuespeed0", "type": "continuous", "domain": (3.0, 3.0000010)},
    {"name": "phi0", "type": "continuous", "domain": (phi_base, phi_base + 0.00000001)},
]

# Create the Bayesian Optimization problem using GPyOpt
my_problem = GPyOpt.methods.BayesianOptimization(
    f=my_function, domain=bounds, acquisition_type="EI", normalize_Y=True
)

# Run the optimization for 50 iterations
my_problem.run_optimization(max_iter=1000)

# Get the optimal values
optimal_values = my_problem.x_opt
optimal_value = my_problem.Y_best

print("Optimal values (a, b):", optimal_values)
print("Minimum objective function value:", optimal_value)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")

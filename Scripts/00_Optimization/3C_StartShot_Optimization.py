#! /usr/bin/env python
import time

import numpy as np
from scipy.optimize import minimize

import pooltool as pt
from pooltool.ruleset.three_cushion import is_point

start_time = time.time()

# We need a table, some balls, and a cue stick
# table = pt.Table.default("billiard")


def my_function(
    vars,
    system,
    sidespin_stddev,
    vertspin_stddev,
    cuespeed_stddev,
    phi_delta_stddev,
    shotnums,
    Rball,
):
    sidespin_avg, vertspin_avg, cuespeed_avg, phi_delta_avg = vars

    # Initialize an empty list to store angles
    points = np.zeros(shotnums)

    # generate shot props from new mean values
    sidespin = np.random.normal(loc=sidespin_avg, scale=sidespin_stddev, size=shotnums)
    vertspin = np.random.normal(loc=vertspin_avg, scale=vertspin_stddev, size=shotnums)
    cuespeed = np.random.normal(loc=cuespeed_avg, scale=cuespeed_stddev, size=shotnums)
    phi = np.random.normal(loc=phi_delta_avg, scale=phi_delta_stddev, size=shotnums)

    for i in range(shotnums):
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

            points[i] = 1 if is_point(system) else 0

    success = np.sum(points) / shotnums
    print(
        "ss=",
        sidespin_avg,
        ", vs=",
        vertspin_avg,
        ", speed=",
        cuespeed_avg,
        ", phi=",
        phi_delta_avg,
        ", success=",
        success,
    )
    return 1 - success


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

phi = pt.aim.at_ball(system, "red", cut=37)
initial_guess = [0.25, 0.2, 3.0, phi]

# shot props
sidespin_stddev = 0.1
vertspin_stddev = 0.1
cuespeed_stddev = 0.3
phi_delta_stddev = 0.15
shotnums = 500

# Use scipy.optimize.minimize to optimize only a, b, c
result = minimize(
    my_function,
    initial_guess,
    args=(
        system,
        sidespin_stddev,
        vertspin_stddev,
        cuespeed_stddev,
        phi_delta_stddev,
        shotnums,
        Rball,
    ),
)
# Minimum found at sidespin = 0.2500075650240656, topspin = 0.20001886155275453, speed = 3.00002262680956, phi = 81.19511213669074
# Minimum found at sidespin = 0.19999959771846243, topspin = 0.24999959771846242, speed = 2.4999987931553873, phi = 81.33573971212816
# Minimum found at sidespin = 0.20000683526706306, topspin = 0.24999652311780152, speed = 2.5000271988714515, phi = 81.33577086126175

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")

# Print the result
print(
    f"Minimum found at sidespin = {result.x[0]}, topspin = {result.x[1]}, speed = {result.x[2]}, phi = {result.x[3]}"
)
print(f"Minimum value of the function: {result.fun}")

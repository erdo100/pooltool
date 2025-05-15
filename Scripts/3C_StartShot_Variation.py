#! /usr/bin/env python
import time

import matplotlib.pyplot as plt
import numpy as np

import pooltool as pt
from pooltool.ruleset.three_cushion import is_point

start_time = time.time()

# We need a table, some balls, and a cue stick
# table = pt.Table.default("billiard")

# Ball Positions
wpos = (0.5275, 0.71)  # White
ypos = (0.71, 0.71)  # Yellow
rpos = (0.71, 2.13)  # Red

# shot props
sidespin_avg = 0.0002582888074151872
sidespin_stddev = 0.05

vertspin_avg = 0.25191966690315837
vertspin_stddev = 0.05

cuespeed_avg = 6.848007004576553
cuespeed_stddev = 0.2

phi_avg = 80.87620300936305
phi_stddev = 0.1

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

shotnums = 250
# Initialize an empty list to store angles
points = np.zeros(shotnums)

# shot props
sidespin = np.random.normal(loc=sidespin_avg, scale=sidespin_stddev, size=shotnums)
vertspin = np.random.normal(loc=vertspin_avg, scale=vertspin_stddev, size=shotnums)
cuespeed = np.random.normal(loc=cuespeed_avg, scale=cuespeed_stddev, size=shotnums)
phi = np.random.normal(loc=phi_avg, scale=phi_stddev, size=shotnums)
# sidespin = np.random.uniform(sidespin_avg-sidespin_stddev, sidespin_avg+sidespin_stddev, shotnums)
# vertspin = np.random.uniform(vertspin_avg-vertspin_stddev, vertspin_avg+vertspin_stddev, shotnums)
# cuespeed = np.random.uniform(cuespeed_avg-cuespeed_stddev, cuespeed_avg+cuespeed_stddev, shotnums)
# phi_delta = np.random.uniform(phi_delta_avg-phi_delta_stddev, phi_delta_avg+phi_delta_stddev, shotnums)

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
# phi = pt.aim.at_ball(system_template, "red", cut=37) + phi_delta
# phi = phi_avg + phi_delta

for i in range(shotnums):
    # Creates a deep copy of the template
    system = system_template.copy()

    system.cue.set_state(
        V0=cuespeed.item(i), phi=phi.item(i), a=sidespin.item(i), b=vertspin.item(i)
    )

    # Evolve the shot.
    pt.simulate(system, inplace=True)
    # pt.show(system)
    points[i] = 1 if is_point(system) else 0


successrate = round(np.sum(points) / shotnums * 100, 2)
print(f"Success rate: {successrate} %")


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")

fig1, ax1 = plt.subplots(4, 4, figsize=(12, 12))
# ax1[0,0].scatter(sidespin[points==0], sidespin[points==0], color='red', label='0', s=5)
ax1[0, 0].scatter(
    sidespin[points == 1], sidespin[points == 1], color="black", label="1", s=5
)
# ax1[1,0].scatter(sidespin[points==0], vertspin[points==0], color='red', label='0', s=5)
ax1[1, 0].scatter(
    sidespin[points == 1], vertspin[points == 1], color="black", label="1", s=5
)
# ax1[2,0].scatter(sidespin[points==0], cuespeed[points==0], color='red', label='0', s=5)
ax1[2, 0].scatter(
    sidespin[points == 1], cuespeed[points == 1], color="black", label="1", s=5
)
# ax1[3,0].scatter(sidespin[points==0], phi_delta[points==0], color='red', label='0', s=5)
ax1[3, 0].scatter(
    sidespin[points == 1], phi[points == 1], color="black", label="1", s=5
)


# ax1[0,1].scatter(vertspin[points==0], sidespin[points==0], color='red', label='0', s=5)
ax1[0, 1].scatter(
    vertspin[points == 1], sidespin[points == 1], color="black", label="1", s=5
)
# ax1[1,1].scatter(vertspin[points==0], vertspin[points==0], color='red', label='0', s=5)
ax1[1, 1].scatter(
    vertspin[points == 1], vertspin[points == 1], color="black", label="1", s=5
)
# ax1[2,1].scatter(vertspin[points==0], cuespeed[points==0], color='red', label='0', s=5)
ax1[2, 1].scatter(
    vertspin[points == 1], cuespeed[points == 1], color="black", label="1", s=5
)
# ax1[3,1].scatter(vertspin[points==0], phi_delta[points==0], color='red', label='0', s=5)
ax1[3, 1].scatter(
    vertspin[points == 1], phi[points == 1], color="black", label="1", s=5
)

# ax1[0,2].scatter(cuespeed[points==0], sidespin[points==0], color='red', label='0', s=5)
ax1[0, 2].scatter(
    cuespeed[points == 1], sidespin[points == 1], color="black", label="1", s=5
)
# ax1[1,2].scatter(cuespeed[points==0], vertspin[points==0], color='red', label='0', s=5)
ax1[1, 2].scatter(
    cuespeed[points == 1], vertspin[points == 1], color="black", label="1", s=5
)
# ax1[2,2].scatter(cuespeed[points==0], cuespeed[points==0], color='red', label='0', s=5)
ax1[2, 2].scatter(
    cuespeed[points == 1], cuespeed[points == 1], color="black", label="1", s=5
)
# ax1[3,2].scatter(cuespeed[points==0], phi_delta[points==0], color='red', label='0', s=5)
ax1[3, 2].scatter(
    cuespeed[points == 1], phi[points == 1], color="black", label="1", s=5
)

# ax1[0,3].scatter(phi_delta[points==0], sidespin[points==0], color='red', label='0', s=5)
ax1[0, 3].scatter(
    phi[points == 1], sidespin[points == 1], color="black", label="1", s=5
)
# ax1[1,3].scatter(phi_delta[points==0], vertspin[points==0], color='red', label='0', s=5)
ax1[1, 3].scatter(
    phi[points == 1], vertspin[points == 1], color="black", label="1", s=5
)
# ax1[2,3].scatter(phi_delta[points==0], cuespeed[points==0], color='red', label='0', s=5)
ax1[2, 3].scatter(
    phi[points == 1], cuespeed[points == 1], color="black", label="1", s=5
)
# ax1[3,3].scatter(phi_delta[points==0], phi_delta[points==0], color='red', label='0', s=5)
ax1[3, 3].scatter(phi[points == 1], phi[points == 1], color="black", label="1", s=5)

ax1[0, 0].set_ylabel("side spin")
ax1[1, 0].set_ylabel("top spin")
ax1[2, 0].set_ylabel("speed")
ax1[3, 0].set_ylabel("direction")
ax1[3, 0].set_xlabel("side spin")
ax1[3, 1].set_xlabel("top spin")
ax1[3, 2].set_xlabel("speed")
ax1[3, 3].set_xlabel("direction")
plt.tight_layout()
plt.show()

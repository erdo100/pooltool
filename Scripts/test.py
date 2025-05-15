#! /usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

import pooltool as pt


def angle_between_points(P1, P2, P3):
    # Define the vectors P1P2 and P2P3
    vector_P1P2 = np.array([P2[0] - P1[0], P2[1] - P1[1]])
    vector_P2P3 = np.array([P3[0] - P2[0], P3[1] - P2[1]])

    # Compute the dot product of the two vectors
    dot_product = np.dot(vector_P1P2, vector_P2P3)

    # Compute the magnitudes of the two vectors
    magnitude_P1P2 = np.linalg.norm(vector_P1P2)
    magnitude_P2P3 = np.linalg.norm(vector_P2P3)

    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_P1P2 * magnitude_P2P3)

    # Ensure the value is within the valid range for arccos due to floating point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate the angle in radians
    angle_radians = np.arccos(cos_theta)

    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


# Ball Positions
wpos = (0.71, 0.71)  # White
ypos = (-0.71, 0.71)  # Yellow
rpos = (-0.71, 2.13)  # Red

# Shot definition
side = np.arange(0.5, 1.1, 0.1)
top = 0.0
vel = 4
direction = 90

# define the properties
u_slide = 0.2
u_roll = 0.008
u_sp_prop = 10 * 2 / 5 / 9
u_ballball = 0.05
e_ballball = 0.95
e_cushion = 0.8
f_cushion = 0.2
grav = 9.81

mball = 0.220
Rball = 61.5 / 1000 / 2

cue_mass = 0.576
cue_len = 1.47
cue_tip_R = 0.022
cue_tip_mass = 0.00001

# We need a table, some balls, and a cue stick
table = pt.Table.default("billiard")

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

# Wrap it up as a System
system_template = pt.System(table=table, balls=(wball, yball, rball), cue=cue)

# Initialize an empty list to store angles
angles = np.zeros(len(side))
avels0 = np.zeros(len(side))
avels1 = np.zeros(len(side))
avels2 = np.zeros(len(side))
avels0r = np.zeros(len(side))
avels1r = np.zeros(len(side))
avels2r = np.zeros(len(side))
avel_v = np.zeros(len(side))
fig1, ax1 = plt.subplots()


i = 0
for side_i in side:
    # Creates a deep copy of the template
    system = system_template.copy()

    system.cue.set_state(V0=vel, phi=direction, a=side_i, b=top)

    # Evolve the shot.
    pt.simulate(system, inplace=True)

    # Open up the shot in the GUI
    # pt.show(system)

    # =========================

    pt.continuize(system, dt=0.01, inplace=True)

    print(f"System simulated: {system.simulated}")

    cue_ball = system.balls["white"]
    cue_history = cue_ball.history_cts
    rvw_cue, s_cue, t_cue = cue_history.vectorize()
    # We can grab the xy-coordinates from the `rvw` array by with the following.
    coords_cue = rvw_cue[:, 0, :2]

    # extract contact angles
    filtered_events = pt.events.filter_type(
        system.events, pt.EventType.BALL_LINEAR_CUSHION
    )
    # angular velocity of velocity (theoretical)
    avel_v[i] = rvw_cue[1, 1, 1] / (0.0615 / 2)
    ax1.plot(coords_cue[:, 1], coords_cue[:, 0])

    if filtered_events:  # This checks if the list is not empty
        hit_c1 = filtered_events[0]
        x0 = hit_c1.agents[0].initial.xyz[0]
        y0 = hit_c1.agents[0].initial.xyz[1]
        pos1 = (x0, y0)
        deflection = 0.71 - x0

        if len(filtered_events) >= 2:
            hit_c2 = filtered_events[1]
            x1 = hit_c2.agents[0].initial.xyz[0]
            y1 = hit_c2.agents[0].initial.xyz[1]
        else:
            x1 = coords_cue[-1, 0]
            y1 = coords_cue[-1, 1]

        pos2 = (x1, y1)

        # Calculate the angle
        angle = 180 - angle_between_points(wpos, pos1, pos2)
        angles[i] = angle
        avels0[i] = hit_c1.agents[0].initial.avel[0]
        avels1[i] = hit_c1.agents[0].initial.avel[1]
        avels2[i] = hit_c1.agents[0].initial.avel[2]
        avels0r[i] = rvw_cue[1, 2, 0]
        avels1r[i] = rvw_cue[1, 2, 1]
        avels2r[i] = rvw_cue[1, 2, 2]

    if i == 20:
        ax1.plot(coords_cue[:, 1], coords_cue[:, 0])
        fig11, ax11 = plt.subplots()
        ax11.plot(t_cue, rvw_cue[:, 1, 0], label="rvw_cue[:, 1, 0]")
        ax11.plot(t_cue, rvw_cue[:, 1, 1], label="rvw_cue[:, 1, 1]")
        ax11.plot(t_cue, rvw_cue[:, 1, 2], label="rvw_cue[:, 1, 2]")
        ax11.grid(True)  # Show grid lines on both axes
        ax11.set_xlabel("time in s")
        ax11.legend()
        ax11.set_title(f"Velocities, vel= {vel_i} m/s")

        fig12, ax12 = plt.subplots()
        ax12.plot(t_cue, rvw_cue[:, 2, 0], label="rvw_cue[:, 2, 0]")
        ax12.plot(t_cue, rvw_cue[:, 2, 1], label="rvw_cue[:, 2, 1]")
        ax12.plot(t_cue, rvw_cue[:, 2, 2], label="rvw_cue[:, 2, 2]")
        ax12.grid(True)  # Show grid lines on both axes
        ax12.set_xlabel("time in s")
        ax12.legend()
        ax12.set_title(f"Angular Velocities, vel= {vel_i} m/s")

    i = i + 1

ax1.set_aspect("equal")
ax1.set_xlim([0, 2.840])
ax1.set_ylim([0, 2.840 / 2])
ax1.set_xticklabels([])  # Remove x-axis labels
ax1.set_yticklabels([])  # Remove y-axis labels
ax1.xaxis.set_major_locator(
    MultipleLocator(2.840 / 8)
)  # Set x-axis ticks every 2 units
ax1.yaxis.set_major_locator(
    MultipleLocator(2.840 / 8)
)  # Set y-axis ticks every 0.5 units
ax1.grid(True)  # Show grid lines on both axes
# Adjust the layout to fit the figure tightly
fig1.set_figwidth(10)
plt.tight_layout()
ax1.set_title("Table")

# Convert the angles list to a numpy array
# angles = np.array(angles)

# Plot the angles over input using matplotlib
fig2, ax2 = plt.subplots()
ax2.plot(side, angles, marker="o", linestyle="-", color="b")
ax2.set_xlabel("Side in a/R")
ax2.set_ylabel("angle in deg")
ax2.set_title("Rebound angle vs Speed (a=0.5, b=0.0)")
ax2.grid(True)

# Plot the angles over input using matplotlib
fig3, ax3 = plt.subplots()
ax3.plot(side, avels0, marker="d", linestyle="-", color="k", label="avel[0]")
ax3.plot(side, avels1, marker="o", linestyle="-", color="b", label="avel[1]")
ax3.plot(side, avels2, marker="o", linestyle="-", color="r", label="avel[2]")
ax3.set_xlabel("Side a")
ax3.grid(True)
ax3.legend()
ax3.set_title("Angular Velocity at cushion hit")

# Plot the angles over input using matplotlib
fig4, ax4 = plt.subplots()
ax4.plot(side, avels0r, marker="d", linestyle="-", color="k", label="rvw_cue[1, 2, 0]")
ax4.plot(side, avels1r, marker="o", linestyle="-", color="b", label="rvw_cue[1, 2, 1]")
ax4.plot(side, avels2r, marker="o", linestyle="-", color="r", label="rvw_cue[1, 2, 2]")
ax4.set_xlabel("Side a")
ax4.grid(True)
ax4.legend()
ax4.set_title("Angular Velocity at start")

# Plot the angles over input using matplotlib
fig5, ax5 = plt.subplots()
ax5.plot(
    side,
    avels0r / avel_v,
    marker="d",
    linestyle="-",
    color="k",
    label="rvw_cue[1, 2, 0]",
)
ax5.plot(
    side,
    avels1r / avel_v,
    marker="o",
    linestyle="-",
    color="b",
    label="rvw_cue[1, 2, 1]",
)
ax5.plot(
    side,
    avels2r / avel_v,
    marker="o",
    linestyle="-",
    color="r",
    label="rvw_cue[1, 2, 2]",
)
ax5.set_xlabel("Side a")
ax5.grid(True)
ax5.legend()
ax5.set_title("SpinRate Factor at start")

plt.show()

print("End of Program")


def angle_between_points(P1, P2, P3):
    # Define the vectors P1P2 and P2P3
    vector_P1P2 = np.array([P2[0] - P1[0], P2[1] - P1[1]])
    vector_P2P3 = np.array([P3[0] - P2[0], P3[1] - P2[1]])

    # Compute the dot product of the two vectors
    dot_product = np.dot(vector_P1P2, vector_P2P3)

    # Compute the magnitudes of the two vectors
    magnitude_P1P2 = np.linalg.norm(vector_P1P2)
    magnitude_P2P3 = np.linalg.norm(vector_P2P3)

    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_P1P2 * magnitude_P2P3)

    # Ensure the value is within the valid range for arccos due to floating point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate the angle in radians
    angle_radians = np.arccos(cos_theta)

    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


"""
# %% [markdown]
# ## Calculating the carom angle
#
# One could calculate the carom angle for the above trajectory by manually splicing the trajectory coordinates of the cue ball and determining ball direction by comparing temporally adjacent coordinates. However, pooltool has much more precise methods for dissecting shot dynamics.
#
# As mentioned before, the carom angle is the angle between the cue ball velocity right before collision, and the cue ball velocity post-collision, once the ball has stopped sliding on the cloth. Hidden somewhere in the system **event list** one can find the events corresponding to these precise moments in time:

# %% trusted=true
system.events[:6]

# %% [markdown]
# Programatically, we can pick out these two events of interest with event selection syntax.
#
# Since there is only one ball-ball collision, it's easy to select with [filter_type](../autoapi/pooltool/events/index.rst#pooltool.events.filter_type):

# %% trusted=true
collision = pt.events.filter_type(system.events, pt.EventType.BALL_BALL)[0]
collision

# %% [markdown]
# To get the event when the cue ball stops sliding, we can similarly try filtering by the sliding to rolling transition event:

# %% trusted=true
pt.events.filter_type(system.events, pt.EventType.SLIDING_ROLLING)

# %% [markdown]
# But there are many sliding to rolling transition events, and to make matters worse, they are shared by both the cue ball and the object ball. What we need is the **first** **sliding to rolling** transition that the **cue ball** undergoes **after** the **ball-ball** collision. We can achieve this multi-criteria query with [filter_events](../autoapi/pooltool/events/index.rst#pooltool.events.filter_events):

# %% trusted=true
transition = pt.events.filter_events(
    system.events,
    pt.events.by_time(t=collision.time, after=True),
    pt.events.by_ball("cue"),
    pt.events.by_type(pt.EventType.SLIDING_ROLLING),
)[0]
transition

# %% [markdown]
# Now, we can dive into these two events and pull out the cue ball velocities we need to calculate the carom angle.

# %% trusted=true
# Velocity prior to impact
for agent in collision.agents:
    if agent.id == "cue":
        # agent.initial is a copy of the Ball before resolving the collision
        velocity_initial = agent.initial.state.rvw[1, :2]

# Velocity post sliding
# We choose `final` here for posterity, but the velocity is the same both before and after resolving the transition.
velocity_final = transition.agents[0].final.state.rvw[1, :2]

carom_angle = pt.ptmath.utils.angle_between_vectors(velocity_final, velocity_initial)

print(f"The carom angle is {carom_angle:.1f} degrees")


# %% [markdown]
# ## Carom angle as a function of cut angle

# %% [markdown]
# We calculated the carom angle for a single cut angle, 30 degrees. Let's write a function called `get_carom_angle` so we can do that repeatedly for different cut angles.


# %% trusted=true
def get_carom_angle(system: pt.System) -> float:
    assert system.simulated

    collision = pt.events.filter_type(system.events, pt.EventType.BALL_BALL)[0]
    transition = pt.events.filter_events(
        system.events,
        pt.events.by_time(t=collision.time, after=True),
        pt.events.by_ball("cue"),
        pt.events.by_type(pt.EventType.SLIDING_ROLLING),
    )[0]

    velocity_final = transition.agents[0].final.state.rvw[1, :2]
    for agent in collision.agents:
        if agent.id == "cue":
            velocity_initial = agent.initial.state.rvw[1, :2]

    return pt.ptmath.utils.angle_between_vectors(velocity_final, velocity_initial)


# %% [markdown]
# `get_carom_angle` assumes the passed system has already been simulated, so we'll need another function to take care of that. We'll cue stick speed and cut angle as parameters.


# %% trusted=true
def simulate_experiment(V0: float, cut_angle: float) -> pt.System:
    system = system_template.copy()
    phi = pt.aim.at_ball(system, "obj", cut=cut_angle)
    system.cue.set_state(V0=V0, phi=phi, b=0.8)
    pt.simulate(system, inplace=True)
    return system


# %% [markdown]
# We'll also want the ball hit fraction:


# %% trusted=true
import numpy as np

def get_ball_hit_fraction(cut_angle: float) -> float:
    return 1 - np.sin(cut_angle * np.pi / 180)


# %% [markdown]
# With these functions, we are ready to simulate how carom angle varies as a function of cut angle.

# %% trusted=true
import pandas as pd

data = {
    "phi": [],
    "f": [],
    "theta": [],
}

V0 = 2.5

for cut_angle in np.linspace(0, 88, 50):
    system = simulate_experiment(V0, cut_angle)
    data["theta"].append(get_carom_angle(system))
    data["f"].append(get_ball_hit_fraction(cut_angle))
    data["phi"].append(cut_angle)

frame = pd.DataFrame(data)
frame.head(10)

# %% [markdown]
# From this dataframe we can make some plots. On top of the ball-hit fraction, plot, I'll create a box between a $1/4$ ball hit and a $3/4$ ball hit, since this is the carom angle range that the 30-degree rule is defined with respect to.

# %% trusted=true editable=true slideshow={"slide_type": ""} tags=["nbsphinx-thumbnail"]
import matplotlib.pyplot as plt

x_min = 0.25
x_max = 0.75
y_min = frame.loc[(frame.f >= x_min) & (frame.f <= x_max), "theta"].min()
y_max = frame.loc[(frame.f >= x_min) & (frame.f <= x_max), "theta"].max()
box_data_x = [x_min, x_min, x_max, x_max, x_min]
box_data_y = [y_min, y_max, y_max, y_min, y_min]

fig, ax = plt.subplots()
ax.plot(box_data_x, box_data_y, linestyle='--', color='gray', label='30-degree range')
ax.scatter(frame['f'], frame['theta'], color='#1f77b4', label='Simulation')
ax.set_title('Carom Angle vs Ball Hit Fraction', fontsize=20)
ax.set_xlabel('Ball Hit Fraction (f)', fontsize=16)
ax.set_ylabel('Carom Angle (theta, degrees)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(fontsize=14)
plt.show()

# %% [markdown]
# Between a $1/4$ and $3/4$ ball hit, there is a relative invariance in the carom angle, with a range of around 6 degrees.
#
# For your reference, here is the same plot but with cut angle $\phi$ as the x-axis:

# %% trusted=true
fig, ax = plt.subplots()
ax.scatter(frame['phi'], frame['theta'], color='#1f77b4')
ax.set_title('Carom Angle vs Cut Angle', fontsize=20)
ax.set_xlabel('Cut Angle (phi)', fontsize=16)
ax.set_ylabel('Carom Angle (theta, degrees)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.show()


# %% [markdown]
# ## Comparison to theory
#
# Under the assumption of a perfectly elastic and frictionless ball-ball collision, Dr. Dave has calculated the theoretical carom angle $\theta$ to be
#
# $$
# \theta_{\text{ideal}}(\phi) = \arctan{\frac{\sin\phi \times \cos\phi}{\sin^2\phi + \frac{2}{5}}}
# $$
#
# *(source: [https://billiards.colostate.edu/technical_proofs/new/TP_B-13.pdf](https://billiards.colostate.edu/technical_proofs/new/TP_B-13.pdf))*
#
# Since pooltool's baseline physics engine makes the same assumptions, we should expect the angles to be the same. Let's directly compare:


# %% trusted=true
def get_theoretical_carom_angle(phi) -> float:
    return np.atan2(np.sin(phi) * np.cos(phi), (np.sin(phi) ** 2 + 2 / 5))

phi_theory = np.linspace(0, np.pi / 2, 500)
theta_theory = get_theoretical_carom_angle(phi_theory)

phi_theory *= 180 / np.pi
theta_theory *= 180 / np.pi
f_theory = get_ball_hit_fraction(phi_theory)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=box_data_x,
        y=box_data_y,
        mode="lines",
        name="30-degree range",
        line=dict(dash="dash", color="gray"),
    )
)
fig.add_trace(
    go.Scatter(
        x=frame["f"],
        y=frame["theta"],
        mode="markers",
        name="Simulation",
        marker=dict(color="#1f77b4"),
    )
)
fig.add_trace(go.Scatter(x=f_theory, y=theta_theory, mode="lines", name="Theory"))
fig.update_layout(
    title="Carom Angle vs Ball Hit Fraction",
    xaxis_title="Ball Hit Fraction (f)",
    yaxis_title="Carom Angle (theta, degrees)",
    template="presentation",
)
fig.show()

# %% [markdown]
# A perfect match.

# %% [markdown]
# ## Impact speed independence
#
# Interestingly, the carom angle is independent of the speed:

# %% trusted=true
for V0 in np.linspace(1, 4, 20):
    system = simulate_experiment(V0, 30)
    carom_angle = get_carom_angle(system)
    print(f"Carom angle for V0={V0:2f} is {carom_angle:4f}")

# %% [markdown]
# This doesn't mean that the trajectories are the same though. Here are the trajectories:

# %% trusted=true
import numpy as np
import plotly.graph_objects as go


def get_coordinates(system: pt.System):
    rvw, s, t = system.balls["cue"].history_cts.vectorize()
    xy = rvw[:, 0, :2]

    return xy, s, t

print("test")

fig = go.Figure()

for V0 in np.linspace(1, 3, 6):
    system = simulate_experiment(V0, 30)
    pt.continuize(system, dt=0.03, inplace=True)
    rvw, s, t = system.balls["cue"].history_cts.vectorize()
    xy = rvw[:, 0, :2]

    fig.add_trace(
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="lines",
            name=f"Speed {V0}",
            showlegend=True,
        )
    )

fig.update_layout(
    title="Ball trajectories",
    xaxis_title="X [m]",
    yaxis_title="Y [m]",
    yaxis_scaleanchor="x",
    yaxis_scaleratio=1,
    width=600,
    xaxis=dict(range=[1.5, 4.5]),
    yaxis=dict(range=[1.5, 4.5]),
    template="presentation",
)
fig.show()

# %% [markdown]
# Harder shots follow the *tangent line* (aka the line perpendicular to the line connected the balls' centers during contact) for longer, but they all converge to the same outgoing angle.
"""

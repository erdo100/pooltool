import numpy as np

import pooltool as pt

# Build a table with default BILLIARD specs
table = pt.Table.default(pt.TableType.BILLIARD)

# Generate the ball layout from the THREECUSHION GameType using the BILLIARD table
balls = pt.get_rack(pt.GameType.THREECUSHION, table=table)

# Build a cue stick
cue = pt.Cue.default()

# Compile everything into a system
system = pt.System(
    cue=cue,
    table=table,
    balls=balls,
)

# phi = pt.aim.at_ball(system, "red", cut=30)
# system.cue.set_state(V0=3.0, phi=phi, a=0.4, b=0.3)
system.cue.set_state(V0=3.0, phi=90, a=0.4, b=0.3)

# Now simulate it
pt.simulate(system, inplace=True)


# Get ball-ball event
ball_ball = pt.events.filter_type(system.events, pt.events.EventType.BALL_BALL)[0]

# Use ball_ball.ids to see which ball IDs are involved in the event
cb_initial = ball_ball.get_ball("white", initial=True)
ob_initial = ball_ball.get_ball("red", initial=True)

center_to_center = pt.ptmath.unit_vector(ob_initial.xyz - cb_initial.xyz)
direction = pt.ptmath.unit_vector(cb_initial.vel)

cut_angle_radians = np.arccos(np.dot(direction, center_to_center))
cut_angle_degrees = cut_angle_radians * 180 / np.pi
hit_fraction = 1 - np.sin(cut_angle_radians)

print(f"{cut_angle_degrees=}", f"{hit_fraction=}")

# Now visualize it (no motion yet)
pt.show(system)

import numpy as np

import pooltool as pt

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

cutangle = np.linspace(1, 89, 20)

for i in range(len(cutangle)):
    phi = pt.aim.at_ball(system, "red", cut=cutangle.item(i))
    print(cutangle.item(i), phi)

#! /usr/bin/env python

import sys
import pooltool as pt


cbX = 0.5
cbY = 0.5

obX = 0.5
obY = 1.0

cA = 90
cS = 0.5


# Found these used in the 8 ball script. Should be useful 
from pooltool.ruleset.utils import (
    is_ball_hit,
    is_ball_pocketed,
)

# We need a table, some balls, and a cue stick
table = pt.Table.default()

cue = pt.Cue(cue_ball_id="cue")

# x = short rail, y = long
# x: 1 = full width
# y: 2 = full length

cue_ball = pt.Ball.create("cue", xy=(cbX, cbY))
obj_ball = pt.Ball.create("1", xy=(obX, obY))


balls=(cue_ball, obj_ball)

# Wrap it up as a System
shot = pt.System(table=table, balls=balls, cue=cue)

shot.cue.set_state(V0=cS, phi=cA)

# Evolve the shot.
pt.simulate(shot, inplace=True)
pt.continuize(shot, dt=0.0001, inplace=False)

# Open up the shot in the GUI
#pt.show(shot)

cue_history = cue_ball.history_cts
print (f"history {cue_ball.history_cts}")
print (f"vector {cue_history.vectorize()}")

rvw_cue, s_cue, t_cue = cue_history.vectorize()
print(rvw_cue.shape)
print(s_cue.shape)
print(t_cue.shape)

if is_ball_pocketed(shot, "cue"):
    print ("scratch")
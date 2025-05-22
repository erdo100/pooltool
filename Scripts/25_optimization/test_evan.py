#! /usr/bin/env python

import pooltool as pt

# We need a table, some balls, and a cue stick
table = pt.Table.default()
balls = pt.get_rack(pt.GameType.NINEBALL, table)
cue = pt.Cue(cue_ball_id="cue")

# Wrap it up as a System
shot = pt.System(table=table, balls=balls, cue=cue)

# Aim at the head ball with a strong impact
shot.cue.set_state(V0=8, phi=pt.aim.at_ball(shot, "1"))

# Evolve the shot.
pt.simulate(shot, inplace=True)

print("Opening up the interface. The call to pt.show is blocking, meaning code will not execute until the interface is exited.")

# Open up the shot in the GUI
pt.show(shot)

print("The program doesn't end, it continues, as proven by this print statement.")

print("We can even open up the interface a second time, but first, sleeping for 4 seconds")
import time
time.sleep(4)

pt.show(shot)

print("...And the program execution continues")
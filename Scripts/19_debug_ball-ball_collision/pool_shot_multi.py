import numpy as np

import pooltool as pt

table = pt.Table.default()
balls = {
    "cue": pt.Ball.create("cue", xy=(0.5 * table.w, 0.25 * table.l)),
    "1": pt.Ball.create("1", xy=(0.5 * table.w, 0.5 * table.l)),
}
cue = pt.Cue.default()
cue.set_state(a=0.4, b=0.2, theta=15, V0=3.3)

template = pt.System(table=table, balls=balls, cue=cue)
template.set_ballset(pt.objects.ball.sets.BallSet("pooltool_pocket"))

systems = []
for phi_delta in np.linspace(6.9, 6.9, 1):
    system = template.copy()
    system.cue.set_state(phi=(pt.aim.at_ball(system, "1") + phi_delta))
    pt.simulate(system, inplace=True, continuous=True, dt=0.01)
    systems.append(system)

pt.show(pt.MultiSystem(systems))
#! /usr/bin/env python
import pooltool as pt


def main():
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

    # Now visualize it (no motion yet)
    pt.show(system)


if __name__ == "__main__":
    main()

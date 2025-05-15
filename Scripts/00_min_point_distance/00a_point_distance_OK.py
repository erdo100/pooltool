#! /usr/bin/env python
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

import pooltool as pt
from pooltool.events.datatypes import EventType
from pooltool.events.filter import (
    filter_ball,
    filter_type,
)
from pooltool.ruleset.three_cushion import is_point

# from pooltool.physics.resolve.resolver import RESOLVER_CONFIG_PATH
# print(RESOLVER_CONFIG_PATH)

# We need a table, some balls, and a cue stick
# table = pt.Table.default("billiard")

# Ball Positions
wpos = (0.5275, 0.71)  # White
ypos = (0.71, 0.71)  # Yellow
rpos = (0.71, 2.13)  # Red

cutangle0 = 30

# shot props
sidespin = 0.24
vertspin = 0.2
cuespeed = 3.2

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

phi = pt.aim.at_ball(system, "red", cut=cutangle0)
system.cue.set_state(V0=cuespeed, phi=phi, a=sidespin, b=vertspin)


# Shot analysis
def eval_shot(shot: system):
    # Shot analysis and evaluation of results
    # Identify the balls b1=cueball, b2=objectball, b3=targetball
    #   - if no ball-ball event, b2 is the ball which closest to b1 after 3 cushions
    # Get the ball-ball events for each ball
    # Calculate point distance
    # Calculate hit fractions for
    #   - b1b2 hit to get first contact
    #   - b1b3 hit to get how safe was point
    #   - b2b3 hit to get how bad was the kiss
    #   - b2b3 hit to get how bad was the kiss
    #   - b1b3 hit before 3 cushions to get how bad was the shot
    #

    # Player Ability  model:
    # Shot quality is defined by the distribution (mean value and standard deviation) of shot parameters
    #     - hit direction horizontal, elevation,
    #     - hitpoint on cue ball
    #     - velocity

    # These should be depending on player profile
    # - Depending on player size the reach to cue ball influences the shot quality:
    #     - standard, no restriction, both feet on ground,
    #     - one foot on ground
    #     - use of extension
    #     - use of bridge
    # - shot quality depending shot speed
    # - left / right hand
    # - dominant eye dependency:
    #     - playing left / right spin
    #     - playing to b2 left / right
    # - optical illusions by playing parallel or angled to cushions

    # for evaluations and further actions (e.g. machine learning), we need to generate/store the following information
    # INITIAL DATA:
    #   - ball-ball distances
    #   - angles between balls
    #   - distances of the balls to the cushions
    #   - evaluate reachable positions and shot quality from player ability model
    #       - by body size, max reach, shot quality depending on distance
    #       - by left / right hand, shot quality depending on hand
    # IN-SHOT DATA:
    #   - for each ball-ball event
    #       - hit fractions
    #       - ball-ball events for each ball string positions, velocities, angular velocities
    #   - for each cushion hit
    #       - cushion hit positions, velocities, angular velocities,
    #       - direction in
    #       - direction initial out
    #       - direction final out with offset from cushion hit point
    # POST-SHOT:
    #   - ispoint and point distance
    #   - iskiss and kisses distances
    #   - isfluke and fluke distance
    #   - point distance
    #   - classification of route (name the shot)
    #   - rate the following position
    #   - transform to standard orientation to reduce effort by using symmetries
    #       - with hand (left/right) restriction 21 symmetries are available ==> reduction of total position by factor 2
    #       - without restriction 2 symmetries are available ==> reduction of total position by factor 4

    # NEXT STEPS:
    # - plot table with routes in figure
    # - Correct identification the balls b1, b2, b3 considering the also the case when no ball was hit

    def get_ball_order(shot):
        # identify the balls b1=cueball, b2=objectball, b3=targetball

        b1 = shot.cue.cue_ball_id

        # identy b2 and b3.
        # if b1 hits only one ball, b2 is the ball which is hit by b1, b3 is the remaining ball
        # if b1 has no ball-ball event, b2 is the closest ball to b1 after 3 cushions, b3 is the remaining ball

        # get ball events of cue ball b1
        b1events = filter_ball(shot.events, b1)
        b1ballhits = filter_type(b1events, EventType.BALL_BALL)

        if b1ballhits != []:
            # b2 is the first ball which is touched by b1
            b2 = [color for color in b1ballhits[0].ids if color != b1][0]
            # remaining ball is b3
            b3 = [
                color for color in ("white", "yellow", "red") if color not in (b1, b2)
            ][0]
        else:
            # no ball contact, so we define b1 and b2
            # in future change it to the closest ball to the cueball after 3 cushions
            b2 = [color for color in ("white", "yellow") if color != b1][0]
            b3 = "red"

        return [b1, b2, b3]

    def get_ball_events(shot):
        # collect all hits for each ball
        b1events = filter_ball(shot.events, b1)
        b1hit = filter_type(
            b1events, [EventType.BALL_BALL, EventType.BALL_LINEAR_CUSHION]
        )
        b2events = filter_ball(shot.events, b2)
        b2hit = filter_type(
            b2events, [EventType.BALL_BALL, EventType.BALL_LINEAR_CUSHION]
        )
        b3events = filter_ball(shot.events, b3)
        b3hit = filter_type(
            b3events, [EventType.BALL_BALL, EventType.BALL_LINEAR_CUSHION]
        )

        return [b1hit, b2hit, b3hit]

    def add_events_to_coords(shot):
        # Add events to the vectorized time series to have accurate positions for each event

        shotcont = pt.continuize(shot, dt=0.01, inplace=False)

        # Add events to the vectorized time series
        b1_obj = shotcont.balls[b1]
        b1_hist = b1_obj.history_cts
        rvw_b1, s_b1, t_b1 = b1_hist.vectorize()
        b1_coords = rvw_b1[:, 0, :2]

        b2_obj = shotcont.balls[b2]
        b2_hist = b2_obj.history_cts
        rvw_b2, s_b2, t_b2 = b2_hist.vectorize()
        b2_coords = rvw_b2[:, 0, :2]

        b3_obj = shotcont.balls[b3]
        b3_hist = b3_obj.history_cts
        rvw_b3, s_b3, t_b3 = b3_hist.vectorize()
        b3_coords = rvw_b3[:, 0, :2]

        all_ball_events = filter_type(
            shot.events, [EventType.BALL_BALL, EventType.BALL_LINEAR_CUSHION]
        )

        for event in all_ball_events:
            event_time = event.time
            # Find the index to insert the event time
            index = np.searchsorted(t_b1, event_time)

            # Insert event time into t_b1
            t_b1 = np.insert(t_b1, index, event_time)

            # find ball which was not involved in the event
            otherballs = tuple(set((b1, b2, b3)) - set(event.ids))

            # add positions to involved balls
            for id in event.ids:
                if id == b1:
                    event_xy = event.get_ball(id, initial=True).xyz[0:2]
                    b1_coords = np.insert(b1_coords, index, event_xy, axis=0)
                elif id == b2:
                    event_xy = event.get_ball(id, initial=True).xyz[0:2]
                    b2_coords = np.insert(b2_coords, index, event_xy, axis=0)
                elif id == b3:
                    event_xy = event.get_ball(id, initial=True).xyz[0:2]
                    b3_coords = np.insert(b3_coords, index, event_xy, axis=0)

            for id in otherballs:
                # linear interpolate the position of the ball which was not involved in the event
                if id == b1:
                    interp = interp1d(
                        [t_b1[index - 1], t_b1[index]],
                        [b1_coords[index - 1], b1_coords[index]],
                        axis=0,
                        kind="linear",
                    )
                    xy = interp(event_time)
                    b1_coords = np.insert(b1_coords, index, xy, axis=0)
                elif id == b2:
                    interp = interp1d(
                        [t_b1[index - 1], t_b1[index]],
                        [b2_coords[index - 1], b2_coords[index]],
                        axis=0,
                        kind="linear",
                    )
                    xy = interp(event_time)
                    b2_coords = np.insert(b2_coords, index, xy, axis=0)
                elif id == b3:
                    interp = interp1d(
                        [t_b1[index - 1], t_b1[index]],
                        [b3_coords[index - 1], b3_coords[index]],
                        axis=0,
                        kind="linear",
                    )
                    xy = interp(event_time)
                    b3_coords = np.insert(b3_coords, index, xy, axis=0)

        return [t_b1, b1_coords, b2_coords, b3_coords]

    def ball_ball_distances():
        # Calculate ball to ball distance
        b1b2dist = np.sqrt(np.sum((b1_coords - b2_coords) ** 2, axis=1))
        b1b3dist = np.sqrt(np.sum((b1_coords - b3_coords) ** 2, axis=1))
        b2b3dist = np.sqrt(np.sum((b2_coords - b3_coords) ** 2, axis=1))

        return [b1b2dist, b1b3dist, b2b3dist]

    def eval_hit_fraction(shot, event):
        # calculate hit_fraction of given event
        # check if the event is a ball-ball event
        if event.event_type != EventType.BALL_BALL:
            print("Event is not a ball-ball event.")
            return None

        # Use ball_ball.ids to see which ball IDs are involved in the event
        ball1 = event.get_ball(event.ids[0], initial=True)
        ball2 = event.get_ball(event.ids[1], initial=True)

        center_to_center = pt.ptmath.unit_vector(ball2.xyz - ball1.xyz)
        direction = pt.ptmath.unit_vector(ball1.vel - ball2.vel)

        cut_angle_radians = np.arccos(np.dot(direction, center_to_center))
        # cut_angle_degrees = cut_angle_radians * 180 / np.pi
        hit_fraction = 1 - np.sin(cut_angle_radians)

        # print(f"{cut_angle_degrees=}", f"{hit_fraction=}")

        return hit_fraction

    def cushion_count(shot):
        # count the cushion hits before b1 hits b3
        #

        return cushion_count

    def kisses(shot):
        return kisses

    def eval_point_distance(shot):
        # calculate point distance

        # calculate 3 closest distances to make a point
        # if the shot is a point, calculate the distance at the point of contact
        # if b1 hit b2 and b2 before hitting 3 cushions, set point_distance = 3000

        # Initialize variables
        point_distance = (3000.0, 3000.0, 3000.0)
        cushion_hit_count = 0  # Counter for ball_linear_cushion events
        check_time = None  # Variable to store the time when conditions are met
        b2_found = 0  # Flag for `b2` in agents
        hit_fraction = 0
        point_distance0 = 3000.0
        point_time = -1.0

        # Iterate through events
        for event in b1hit:
            # Condition 1: Check if type is ball_linear_cushion
            if event.event_type == "ball_linear_cushion":
                cushion_hit_count += 1  # Increment cushion hit counter
                # Store the time of the last `ball_linear_cushion` event

            # Condition 2: Check if b2 exists in agents (excluding b1)
            if b2 in event.ids and b2_found is False:
                b2_found = b2_found + 1

            # Check if the conditions are met
            if cushion_hit_count >= 3 and b2_found == 1 and check_time is None:
                # We have met the requirements, store the time
                check_time = event.time

            if cushion_hit_count >= 3 and b2_found >= 1 and b3 in event.ids:
                point_time = event.time
                hit_fraction = eval_hit_fraction(shot, event)
                point_distance0 = hit_fraction * Rball
                print(point_distance0)
                break

        if cushion_hit_count >= 3 and b2_found >= 1:
            point_distance = eval_point_distance_3c_nopoint(
                check_time, point_distance0, point_time
            )

        # elif b2hit != [] and b3hit == []:
        #     # one ball was hit
        #     # print('One ball was hit')
        #     tmp = 0

        # elif b2hit == [] and b3hit == []:
        #     # no ball was hit
        #     # print('No ball was hit')
        #     tmp = 0

        return point_distance

    def eval_point_distance_3c_nopoint(check_time, point_distance0, point_time):
        # Calculate the point distance for a 3-cushion shot that is not a point
        # find 3 different minima (if available) of b1b3dist[tsel]
        point_distance = (3000.0, 3000.0, 3000.0)
        tsel = t_b1 > check_time
        y = b1b3dist[tsel]
        t = t_b1[tsel]

        # Find local minima (relative minima) in the data
        minima_indices = argrelextrema(y, np.less)[0]

        # When there is a point, then distance is limited by ball diameter
        # Therefore replace the minima of the point
        # Use the point time if it is given
        if point_time > 0 and np.min(np.abs(t - point_time)) < 1.0e-6:
            point_index = np.argmin(np.abs(t - point_time))

            # now replace in y(minima_indices) the value y(point_index) = point_distance0
            y[point_index] = point_distance0

        # check if the distance was getting closer at the end, but not yet minimum
        if y[-2] >= y[-1]:
            # Add the last index to the minima_indices if we have negative slope at the end
            minima_indices = np.append(minima_indices, len(y) - 1)

        # Ensure the array has 3 elements
        while len(minima_indices) < 3:
            minima_indices = np.append(minima_indices, len(y) - 1)

        # Sort the minima by their values (y values) and get the 3 smallest
        sorted_minima_indices = minima_indices[np.argsort(y[minima_indices])][:3]
        point_distance = y[sorted_minima_indices]

        # Plot distances
        plt.figure(figsize=(8, 5))

        plt.plot(t_b1[tsel], b1b2dist[tsel], linestyle="-", color="r", label="Distance")
        plt.plot(t_b1[tsel], b1b3dist[tsel], linestyle="-", color="b", label="Distance")
        plt.plot(t_b1[tsel], b2b3dist[tsel], linestyle="-", color="k", label="Distance")

        plt.title("Distance Between Corresponding Points")
        plt.xlabel("time in s")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.legend()
        plt.show()

        return point_distance

    # START of evaluation
    # identify the balls cueball, objectball, targetball and store in ballorder
    (b1, b2, b3) = get_ball_order(shot)
    print(f"{b1=}, {b2=}, {b3=}")

    (b1hit, b2hit, b3hit) = get_ball_events(shot)
    (t_b1, b1_coords, b2_coords, b3_coords) = add_events_to_coords(shot)

    (b1b2dist, b1b3dist, b2b3dist) = ball_ball_distances()
    # calculate point distance
    point_distance = eval_point_distance(shot)
    print(f"{point_distance=}")


start_time = time.time()

for i in range(1):
    # Evolve the shot.
    pt.simulate(system, inplace=True)
    print(f"is_point: {is_point(system)}")
    eval_shot(system)

elapsed_time = time.time() - start_time
print(f"Execution time: {elapsed_time} seconds")
print(f"time: {time.time() - start_time} s")

# Open up the shot in the GUI
pt.show(system)

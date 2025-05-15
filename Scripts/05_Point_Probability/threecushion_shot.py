#! /usr/bin/env python

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


class Shots:
    def __init__(self, b1pos, b2pos, b3pos, a, b, v, cut, incline, point):
        self.b1pos = b1pos
        self.b2pos = b2pos
        self.b3pos = b3pos
        self.a = a
        self.b = b
        self.v = v
        self.cut = cut
        self.incline = incline
        self.point = point


class BilliardEnv:
    def __init__(self):
        self.table_width = 2.84  # Table dimensions (meters)
        self.table_height = 1.42
        self.series_length = 0
        self.current_step = 0
        self.episode_rewards = []

        # Ball Positions
        self.ball1_ini = (0.5275, 0.71)  # White
        self.ball2_ini = (0.71, 0.71)  # Yellow
        self.ball3_ini = (0.71, 2.13)  # Red

        # define the properties
        self.u_slide = 0.15
        self.u_roll = 0.005
        self.u_sp_prop = 10 * 2 / 5 / 9
        self.u_ballball = 0.05
        self.e_ballball = 0.95
        self.e_cushion = 0.9
        self.f_cushion = 0.15
        self.grav = 9.81

        self.mball = 0.210
        self.Rball = 61.5 / 1000 / 2

        cue_mass = 0.576
        cue_len = 1.47
        cue_tip_R = 0.022
        cue_tip_mass = 0.0000001

        # Build a table with default BILLIARD specs
        self.table = pt.Table.default(pt.TableType.BILLIARD)

        # create the cue
        cue_specs = pt.objects.CueSpecs(
            brand="Predator",
            M=cue_mass,
            length=cue_len,
            tip_radius=cue_tip_R,
            #=0.02,
            end_mass=cue_tip_mass,
        )
        self.cue = pt.Cue(cue_ball_id="white", specs=cue_specs)

    def prepare_new_shot(self, ball1xy, ball2xy, ball3xy, a, b, v, cut, theta):
        # Create balls in new positions
        wball = pt.Ball.create(
            "white",
            xy=ball1xy,
            m=self.mball,
            R=self.Rball,
            u_s=self.u_slide,
            u_r=self.u_roll,
            u_sp_proportionality=self.u_sp_prop,
            u_b=self.u_ballball,
            e_b=self.e_ballball,
            e_c=self.e_cushion,
            f_c=self.f_cushion,
            g=self.grav,
        )

        yball = pt.Ball.create(
            "yellow",
            xy=ball3xy,
            m=self.mball,
            R=self.Rball,
            u_s=self.u_slide,
            u_r=self.u_roll,
            u_sp_proportionality=self.u_sp_prop,
            u_b=self.u_ballball,
            e_b=self.e_ballball,
            e_c=self.e_cushion,
            f_c=self.f_cushion,
            g=self.grav,
        )

        rball = pt.Ball.create(
            "red",
            xy=ball2xy,
            m=self.mball,
            R=self.Rball,
            u_s=self.u_slide,
            u_r=self.u_roll,
            u_sp_proportionality=self.u_sp_prop,
            u_b=self.u_ballball,
            e_b=self.e_ballball,
            e_c=self.e_cushion,
            f_c=self.f_cushion,
            g=self.grav,
        )

        # Wrap it up as a System
        self.system = pt.System(
            table=self.table, balls=(wball, yball, rball), cue=self.cue
        )

        phi = pt.aim.at_ball(self.system, "red", cut=cut)
        # set the cue
        self.cue.set_state(a=a, b=b, V0=v, phi=phi, theta=theta)

    def step(self, action):
        self.prepare_new_shot(a, b, v, cut, theta)

        self.simulate_shot()

        reward = self.calculate_reward()

    def simulate_shot(self):
        # run the physics model
        point = 0
        pt.simulate(self.system, inplace=True)

        if is_point(self.system):
            point = 1

        return point

    def calculate_reward(self):
        point_distance = self.eval_shot()
        reward = self.point_distance_reward(point_distance)

        return reward

    def shot_all(self, shots):
        points = np.zeros(len(shots.a))
        for i in range(len(shots.a)):
            self.prepare_new_shot(self, shots, i)

            # check if shot is inside of squirt limit. If so, simulate the shot
            # This will ensure R^2 - a^2 - b^2 >= 0
            if (
                0.6**2 >= (shots.a[i] ** 2 + shots.b[i] ** 2)
                and -89 <= shots.cut[i] <= 89
                and 0 <= shots.inc[i] <= 90
            ):
                phi = pt.aim.at_ball(self.system, "red", cut=shots.cut[i])

                self.cue.set_state(
                    a=shots.a[i],
                    b=shots.b[i],
                    V0=shots.v[i],
                    phi=phi,
                    theta=shots.incline[i],
                )

                # Evolve the shot.
                pt.simulate(self.system, inplace=True)

                points[i] = is_point(self.system)

        return points

    # Shot analysis
    def eval_shot(self):
        shot = self.system

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
                    color
                    for color in ("white", "yellow", "red")
                    if color not in (b1, b2)
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
            cut_angle_degrees = cut_angle_radians * 180 / np.pi
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
            # if b1 hit b2 and b2 before hitting 3 cushions, set point_distance = 3.0

            # Initialize variables
            point_distance = (3.0, 3.0, 3.0)
            cushion_hit_count = 0  # Counter for ball_linear_cushion events
            check_time = None  # Variable to store the time when conditions are met
            b2_found = 0  # Flag for `b2` in agents
            hit_fraction = 0
            point_distance0 = 3.0
            point_time = -1.0
            Rball = shot.balls["white"].params.R

            # Iterate through events
            for event in b1hit:
                # Condition 1: Check if type is ball_linear_cushion
                if event.event_type == "ball_linear_cushion":
                    cushion_hit_count += 1  # Increment cushion hit counter
                    # Store the time of the last `ball_linear_cushion` event

                # Condition 2: Check if b2 exists in agents (excluding b1)
                if b2 in event.ids and b2_found == False:
                    b2_found = b2_found + 1

                # Check if the conditions are met
                if cushion_hit_count >= 3 and b2_found == 1 and check_time == None:
                    # We have met the requirements, store the time
                    check_time = event.time

                if cushion_hit_count >= 3 and b2_found >= 1 and b3 in event.ids:
                    point_time = event.time
                    hit_fraction = eval_hit_fraction(shot, event)
                    point_distance0 = hit_fraction * Rball
                    # print(point_distance0)
                    break

            if cushion_hit_count >= 3 and b2_found >= 1:
                point_distance = eval_point_distance_3c_nopoint(
                    check_time, point_distance0, point_time
                )

            elif b2hit != [] and b3hit == []:
                # one ball was hit
                # print('One ball was hit')
                tmp = 0

            elif b2hit == [] and b3hit == []:
                # no ball was hit
                # print('No ball was hit')
                tmp = 0

            return point_distance

        def eval_point_distance_3c_nopoint(check_time, point_distance0, point_time):
            # Calculate the point distance for a 3-cushion shot that is not a point
            # find 3 different minima (if available) of b1b3dist[tsel]
            point_distance = (3.0, 3.0, 3.0)
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

            # # Plot distances
            # plt.figure(figsize=(8, 5))

            # plt.plot(t_b1[tsel], b1b2dist[tsel], linestyle='-', color='r', label='Distance')
            # plt.plot(t_b1[tsel], b1b3dist[tsel], linestyle='-', color='b', label='Distance')
            # plt.plot(t_b1[tsel], b2b3dist[tsel], linestyle='-', color='k', label='Distance')

            # plt.title('Distance Between Corresponding Points')
            # plt.xlabel('time in s')
            # plt.ylabel('Distance')
            # plt.grid(True)
            # plt.legend()
            # plt.show()

            return point_distance

        # START of evaluation
        # identify the balls cueball, objectball, targetball and store in ballorder
        (b1, b2, b3) = get_ball_order(shot)
        # print(f"{b1=}, {b2=}, {b3=}")

        (b1hit, b2hit, b3hit) = get_ball_events(shot)
        (t_b1, b1_coords, b2_coords, b3_coords) = add_events_to_coords(shot)

        (b1b2dist, b1b3dist, b2b3dist) = ball_ball_distances()
        # calculate point distances
        point_distance = eval_point_distance(shot)
        # print(f"{point_distance[0]=}")

        return point_distance

    def point_distance_reward(self, point_distance):
        point_distance_0 = (
            0,
            2 * self.Rball * 0.7,
            2 * self.Rball,
            2 * self.Rball * 0.3,
            0.2,
            3,
        )
        reward_0 = (10, 10, 5, -5, -10, -50)

        # Interpolation function
        interpolator = interp1d(
            point_distance_0, reward_0, kind="linear", fill_value="extrapolate"
        )

        # New points for interpolation
        reward = interpolator(point_distance[0])

        return reward

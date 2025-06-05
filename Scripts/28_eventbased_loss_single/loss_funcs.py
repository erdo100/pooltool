import numpy as np
from scipy.interpolate import interp1d
import time


def interpolate_ball(act_t, act_x, act_y, sim_t, sim_x, sim_y):

    # calculate loss based on which duration is longer
    tmax = max(act_t[-1], sim_t[-1])
    tmin = min(act_t[0], sim_t[0])

    t_interp = np.linspace(tmin, tmax, 200)

    sim_x_interp = interpolate_coordinate(sim_x, sim_t, t_interp)
    sim_y_interp = interpolate_coordinate(sim_y, sim_t, t_interp)
    act_x_interp = interpolate_coordinate(act_x, act_t, t_interp)
    act_y_interp = interpolate_coordinate(act_y, act_t, t_interp)

    return t_interp, sim_x_interp, sim_y_interp, act_x_interp, act_y_interp

def calculate_distance_loss(t, act_x, act_y, sim_x, sim_y):
    # calculate loss
    dist = np.sqrt((act_x - sim_x) ** 2 + (act_y - sim_y) ** 2)
    # # total length sqrt(deltax**2+dy**2)
    # act_length = np.sqrt(act_x**2 + act_y**2)
    # sim_length = np.sqrt(sim_x**2 + sim_y**2)

    # normalize by the length of the actual and simulated path
    # dist = dist / sim_length

    return dist


def evaluate_loss(sim_env, shot_actual, method="distance"):
    starttime = time.time()

    sim_t, white_rvw, yellow_rvw, red_rvw = sim_env.get_ball_routes()
    balls_rvw = [white_rvw, yellow_rvw, red_rvw]
    sim_hit = sim_env.get_events(sim_env)
    act_hit = shot_actual["hit"]

    # Initialize losses dictionary with zeros
    losses = {
        "total": 0,
        "ball": [
            {"time": [0], "total": [0]},
            {"time": [0], "total": [0]},
            {"time": [0], "total": [0]},
        ],
    }


    if method == "eventbased":
        # Step 1: Convert hit data to chronologically ordered tables
        act_events = convert_hits_to_chronological_table(act_hit)
        sim_events = convert_hits_to_chronological_table(sim_hit)

        # Step 2: Compare the chronological event sequences
        comparison_result = compare_chronological_events(act_events, sim_events)
 
        losses =  loss_fun_eventbased(losses, sim_t, balls_rvw, shot_actual, sim_hit, act_hit, act_events, sim_events, comparison_result)


    elif method == "distance":
        # loop over all balls
        for balli in range(3):
            
            sim_x = balls_rvw[balli][:, 0, 0]
            sim_y = balls_rvw[balli][:, 0, 1]

            act_t = shot_actual['Ball'][balli]["t"]
            act_x = shot_actual['Ball'][balli]["x"]
            act_y = shot_actual['Ball'][balli]["y"]

            # arrayify
            act_t = np.array(act_t)
            act_x = np.array(act_x)
            act_y = np.array(act_y)
            sim_t = np.array(sim_t)
            sim_x = np.array(sim_x)
            sim_y = np.array(sim_y)
            t_interp, sim_x_interp, sim_y_interp, act_x_interp, act_y_interp = interpolate_ball(act_t, act_x, act_y, sim_t, sim_x, sim_y)

            losses =  loss_func_distance(balli, losses, t_interp, sim_x_interp, sim_y_interp, act_x_interp, act_y_interp)
            

    # calculate total loss for the ball
    for balli in range(3):
        losses["total"] += np.sum(losses["ball"][balli]["total"]) 

    endtime = time.time()
    #print("Loss calculation took", endtime - starttime, "seconds")
    
    return losses
        # print("ball", balli, ", loss=", losses["ball"][balli]["total"])

    return losses

def loss_func_distance(balli, losses, t_interp, sim_x_interp, sim_y_interp, act_x_interp, act_y_interp):

    loss_distance = np.sqrt((act_x_interp - sim_x_interp) ** 2 + (act_y_interp - sim_y_interp) ** 2)

    losses["ball"][balli]["time"] = t_interp
    losses["ball"][balli]["total"] = loss_distance
    losses["total"] += np.sum(losses["ball"][balli]["total"]) 

    return losses

def loss_fun_eventbased(losses, sim_t, balls_rvw, shot_actual, sim_hit, act_hit, act_events, sim_events, comp_result):
        
    col = ["W", "Y", "R"]
    allbi = [0, 1, 2]
    correct_hit = True
    nevents = len(act_events['events'])
    # run through events based on actual data and simulation data
    for ei in range(nevents):
        allbi = [0, 1, 2]
        loss_hit_b1 = 10
        loss_hit_b2 = 10
        loss_distance_b1 = 0
        loss_distance_b2 = 0
        current_time = act_events['times'][ei]

        # identify ball index based on "event"
        ball1i = col.index(act_events['events'][ei][0])
        # check whther secong letter is in col
        if act_events['events'][ei][1] in col:
            ball2i = col.index(act_events['events'][ei][1])
            hittype = "ball-ball"
            ball3i = 3 - ball1i - ball2i  # the third ball index
        else:
            hittype = "ball-cushion"
            # remove from allbi = [0, 1, 2] ball1i
            allbi.remove(ball1i)
            ball2i = allbi[0]
            ball3i = allbi[1]

        if correct_hit == True:

            if ei < len(sim_events['events']):
                loss_distance_b1, current_time = distance_loss_event(ball1i, ei, act_events, sim_events, act_hit, sim_hit, shot_actual, balls_rvw, sim_t)
                loss_distance_b2 = 0

                if hittype == "ball-ball":
                    loss_distance_b2, current_time = distance_loss_event(ball2i, ei, act_events, sim_events, act_hit, sim_hit, shot_actual, balls_rvw, sim_t)

                # check whether event is in sim_events and act_events
                if act_events['events'][ei] == sim_events['events'][ei]:
                    loss_hit_b1 = 0
                    if hittype == "ball-ball":
                        loss_hit_b2 = 0
                else:
                    correct_hit = False

            else:
                correct_hit = False

        # loss weight based on event index. spread is defining factor between first ei and last ei
        spread = 2
        loss_weight = (spread -1)/(spread-1)*ei + spread

        # assign the loss
        losses["ball"][ball1i]["time"].append(current_time)
        losses["ball"][ball1i]["total"].append(loss_hit_b1 + loss_distance_b1*loss_weight)
        if hittype == "ball-ball":
            losses["ball"][ball2i]["time"].append(current_time)
            losses["ball"][ball2i]["total"].append(loss_hit_b2 + loss_distance_b2*loss_weight)
        else:
            losses["ball"][ball2i]["time"].append(current_time)
            losses["ball"][ball2i]["total"].append(0)

        losses["ball"][ball3i]["time"].append(current_time)
        losses["ball"][ball3i]["total"].append(0)

    return losses

def distance_loss_event(balli, ei, act_events, sim_events, act_hit, sim_hit, shot_actual, balls_rvw, sim_t):


    act_t = np.array(shot_actual["Ball"][balli]["t"])

    act_ei_ind = np.where(act_t >= act_events['times'][ei])[0][0]
    sim_ei_ind = np.where(sim_t >= sim_events['times'][ei])[0][0]
    # find the time index of the of ei-1 and ei in act_hit and sim_hit
    act_e1_ind = act_ei_ind - 1
    sim_e1_ind = sim_ei_ind - 1

    # calculate distance loss for ball1i
    t0 = sim_t[sim_e1_ind:sim_ei_ind]
    x0 = balls_rvw[balli][sim_e1_ind:sim_ei_ind, 0, 0]
    y0 = balls_rvw[balli][sim_e1_ind:sim_ei_ind, 0, 1]

    t1 = act_t[act_e1_ind:act_ei_ind]
    x1 = shot_actual["Ball"][balli]["x"][act_e1_ind:act_ei_ind]
    y1 = shot_actual["Ball"][balli]["y"][act_e1_ind:act_ei_ind]

    tmin = min(t0[0], t1[0])
    tmax = max(t0[-1], t1[-1])

    tloss = np.linspace(tmin, tmax, 200)

    sim_x_interp = interpolate_coordinate(x0, t0, tloss)
    sim_y_interp = interpolate_coordinate(y0, t0, tloss)
    act_x_interp = interpolate_coordinate(x1, t1, tloss)
    act_y_interp = interpolate_coordinate(y1, t1, tloss)

    # calculate loss
    distance = np.sqrt((act_x_interp - sim_x_interp) ** 2 + (act_y_interp - sim_y_interp) ** 2)
    loss_distance = np.sum(distance)/2.84/len(tloss)


    # if ei is the last event of that ball, then calculate loss diatnce from the last event until end of time
    # check in sim_hit for the current ball are more events after the current event
    # Find the index in sim_hit[balli]['t'] that matches sim_events['times'][ei] within a tolerance of +/-0.01s
    sim_event_time = sim_events['times'][ei]
    sim_ei_times = np.array(sim_hit[balli]['t'])
    time_diffs = np.abs(sim_ei_times - sim_event_time)
    matching_indices = np.where(time_diffs <= 0.01)[0]
    sim_is_end = False
    if len(matching_indices) > 0:
        ball_ei = matching_indices[0]

        if ball_ei == len(sim_hit[balli]['t']) - 1:
            # identify the index of last event in sim_hit for the current ball
            sim_e1_ind = np.where(sim_t >= sim_hit[balli]['t'][ball_ei])[0][0]
            # calculate distance loss from the last event until end of time
            t0 = sim_t[sim_e1_ind:]
            x0 = balls_rvw[balli][sim_e1_ind:, 0, 0]
            y0 = balls_rvw[balli][sim_e1_ind:, 0, 1]
            sim_is_end = True
    


    act_event_time = act_events['times'][ei]
    act_ei_times = np.array(act_hit[balli]['t'])
    time_diffs = np.abs(act_ei_times - act_event_time)
    matching_indices = np.where(time_diffs <= 0.01)[0]
    act_is_end = False
    if len(matching_indices) > 0:
        ball_ei = matching_indices[0]

        if ball_ei == len(act_hit[balli]['t']) - 1:
            act_e1_ind = np.where(act_t >= act_hit[balli]['t'][ball_ei])[0][0]
            # calculate distance loss from the last event until end of time
            t1 = act_t[act_e1_ind:]
            x1 = shot_actual["Ball"][balli]["x"][act_e1_ind:]
            y1 = shot_actual["Ball"][balli]["y"][act_e1_ind:]

            act_is_end = True



    if sim_is_end and act_is_end:
        tmin = min(t0[0], t1[0])
        tmax = max(t0[-1], t1[-1])

        tloss = np.linspace(tmin, tmax, 200)

        sim_x_interp = interpolate_coordinate(x0, t0, tloss)
        sim_y_interp = interpolate_coordinate(y0, t0, tloss)
        act_x_interp = interpolate_coordinate(x1, t1, tloss)
        act_y_interp = interpolate_coordinate(y1, t1, tloss)

        # calculate loss
        distance = np.sqrt((act_x_interp - sim_x_interp) ** 2 + (act_y_interp - sim_y_interp) ** 2)
        loss_distance += np.sum(distance)/2.84/len(tloss)

    return loss_distance, tmax


def interpolate_coordinate(simulated, tsim, actual_times):
    interp_func = interp1d(
        tsim,
        simulated,
        kind="linear",
        bounds_error=False,
        fill_value=(simulated[-1]),
    )
    interpolated = interp_func(actual_times)
    return interpolated

def convert_hits_to_chronological_table(hit_data):
    """
    Convert hit data from all balls into a single chronologically ordered table.
    
    Args:
        hit_data: List of hit data for each ball [ball0, ball1, ball2]
                 Each ball has {"with": [...], "t": [...]} format
                 where "with" contains strings like '-R123Y4'
    
    Returns:
        dict: Chronologically ordered events with keys:
            - 'events': List of event identifiers (WY, WR, YR, W1, W2, etc.)
            - 'times': List of corresponding times
            - 'ball': List of ball indices that experienced the collision
    """
    all_events = []
    
    # Ball color mapping
    ball_colors = ['W', 'Y', 'R']
    cushion_numbers = ['0', '1', '2', '3']
    ball_order = {'W': 0, 'Y': 1, 'R': 2}
    
    # Process each ball's collision data
    for ball_index in range(3):
        current_ball_color = ball_colors[ball_index]
        
        # Process collision string for each event
        for i, char in enumerate(hit_data[ball_index]["with"]):
                
            collision_time = hit_data[ball_index]["t"][i]
            
            collision_identifier = None
            
            if char in ball_colors:  # Ball-ball collision
                # Create consistent ball pair identifier (alphabetical order)
                ball_pair = ''.join(sorted([current_ball_color, char], 
                                            key=lambda x: ball_order[x]))
                collision_identifier = ball_pair
                
            elif char in cushion_numbers:  # Ball-cushion collision
                # Create ball-cushion identifier
                collision_identifier = current_ball_color + char
            
            # Add collision if it's valid
            if collision_identifier:
                all_events.append({
                    'event': collision_identifier,
                    'time': collision_time,
                    'ball': ball_index
                })

    # Remove duplicates for ball-ball collisions (same event from different balls)
    unique_events = []
    seen_ball_collisions = set()
    
    for event_data in all_events:
        event_id = event_data['event']
        event_time = event_data['time']
        
        # For ball-ball collisions, avoid duplicates
        if len(event_id) == 2 and event_id[0] in ball_colors and event_id[1] in ball_colors:
            # Ball-ball collision
            collision_key = (event_id, round(event_time, 6))  # Round to avoid floating point issues
            if collision_key not in seen_ball_collisions:
                unique_events.append(event_data)
                seen_ball_collisions.add(collision_key)
        else:
            # Ball-cushion collision (no deduplication needed)
            unique_events.append(event_data)
    
    # Sort chronologically by time
    unique_events.sort(key=lambda x: x['time'])
    
    # Convert to the required format
    result = {
        'events': [event['event'] for event in unique_events],
        'times': [event['time'] for event in unique_events],
        'ball': [event['ball'] for event in unique_events]
    }
    
    return result


def compare_chronological_events(actual_events, sim_events):

    actual_sequence = actual_events['events']
    sim_sequence = sim_events['events']
    
    # Compare sequences up to the length of the shorter one
    max_length = max(len(actual_sequence), len(sim_sequence))
    min_length = min(len(actual_sequence), len(sim_sequence))
    
    event_matches = []
    all_OK = True
    
    # Compare event by event in chronological order
    for i in range(max_length):
        if i < len(actual_sequence) and i < len(sim_sequence):
            # Both sequences have an event at this position
            actual_event = actual_sequence[i]
            sim_event = sim_sequence[i]
            
            if actual_event == sim_event and all_OK:
                event_matches.append(True)
            else:
                event_matches.append(False)
                all_OK = False
        else:
            # One sequence is longer than the other
            event_matches.append(False)
    
    return event_matches

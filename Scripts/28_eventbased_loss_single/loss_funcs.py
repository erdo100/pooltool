import numpy as np
from scipy.interpolate import interp1d



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

    sim_t, white_rvw, yellow_rvw, red_rvw = sim_env.get_ball_routes()
    balls_rvw = [white_rvw, yellow_rvw, red_rvw]
    sim_hit_all = sim_env.get_events(sim_env)

    if method == "eventbased":
        # Step 1: Convert hit data to chronologically ordered tables
        actual_events_table = convert_hits_to_chronological_table(shot_actual["hit"])
        sim_events_table = convert_hits_to_chronological_table(sim_hit_all)
        
        # Step 2: Compare the chronological event sequences
        comparison_result = compare_chronological_events(actual_events_table, sim_events_table)
        
        # Calculate loss

        # Create loss structure based on comparison
        losses = {}
        losses["comparison"] = comparison_result
        losses["total"] = 1.0 - comparison_result['match_score']  # Loss is 1 - match_score
        
        # Optional: Add detailed event-by-event loss information
        losses["event_details"] = {
            "actual_events": actual_events_table['events'],
            "sim_events": sim_events_table['events'],
            "actual_times": actual_events_table['times'],
            "sim_times": sim_events_table['times'],
            "event_matches": comparison_result['event_matches']
        }
        
        return losses

    # Original distance-based method for backwards compatibility
    losses = {}
    losses["ball"] = [{} for _ in range(3)]
    losses["total"] = 0
    
    # loop over all balls
    for balli in range(3):
        losses["ball"][balli]["time"] = []
        losses["ball"][balli]["distance"] = []
        losses["ball"][balli]["total"] = []
        
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

        if method == "distance":
            losses =  loss_func_distance(balli, losses, t_interp, sim_x_interp, sim_y_interp, act_x_interp, act_y_interp)
        
        # calculate total loss for the ball
        losses["total"] += np.sum(losses["ball"][balli]["total"]) 

    return losses
        # print("ball", balli, ", loss=", losses["ball"][balli]["total"])

    return losses

def loss_func_distance(balli, losses, t_interp, sim_x_interp, sim_y_interp, act_x_interp, act_y_interp):

    loss_distance = np.sqrt((act_x_interp - sim_x_interp) ** 2 + (act_y_interp - sim_y_interp) ** 2)

    losses["ball"][balli]["time"] = t_interp
    losses["ball"][balli]["total"] = loss_distance
    losses["total"] += np.sum(losses["ball"][balli]["total"]) 

    return losses

def loss_fun_eventbased(balli, losses, sim_t, sim_x, sim_y, act_t, act_x, act_y, sim_hit, act_hit, act_events, sim_events):
        
    col = ["W", "Y", "R"]
    current_ball_color = col[balli]
    correct_hit = True
    loss_hit = 0
    loss_distance = 0
    current_time = 0
    loss_hit_std = 1
    losses["ball"][balli]["time"].append(0)
    losses["ball"][balli]["total"].append(0)

    # run through events based on actual data and simulation data
    # Starts with Start and look to the next event
    # total_events = range(max(len(act_hit["with"]), len(sim_hit["with"]))-1)
    total_events = range(len(act_hit["with"])-1)
    for ei in total_events:

        t_act1 = 0
        t_sim1 = 0

        if correct_hit == True:
            # len() ==  1 does mean ==> no contact, 1 element is 'S', or '-'

            if len(act_hit["with"]) == 1 and len(sim_hit["with"]) > 1:
                # ball has no contact, the in sim or actual data
                correct_hit = False
                loss_hit = loss_hit_std

            if len(act_hit["with"]) > 1 and len(sim_hit["with"]) == 1:
                # ball has no contact, the in sim or actual data
                correct_hit = False
                loss_hit = loss_hit_std

            if len(act_hit["with"]) == 1 and len(sim_hit["with"]) == 1:
                # both sim and act have no contact
                loss_hit = 0.0

            # case when more than 1 events left in both
            if ei < len(act_hit["with"])-1 and ei < len(sim_hit["with"])-1:                # calculate the distance loss
                loss_distance, current_time = distance_loss_event(ei, sim_t, sim_x, sim_y, sim_hit, act_t, act_x, act_y, act_hit)

                if ei < len(act_hit["with"])-2 and ei < len(sim_hit["with"])-2:
                    # Check if the upcoming event is same and exists in both act_events and sim_events
                    upcoming_act_partner = act_hit["with"][ei+1]
                    upcoming_sim_partner = sim_hit["with"][ei+1]
                    upcoming_act_time = act_hit["t"][ei+1]
                    upcoming_sim_time = sim_hit["t"][ei+1]
                    
                    if upcoming_act_partner != upcoming_sim_partner:
                        correct_hit = False
                        loss_hit = loss_hit_std
                    else:
                        # Additional check: verify this collision exists in both event lists with matching time
                        if upcoming_act_partner in col:  # Only check ball-ball collisions
                            # Create ball pair identifier for verification
                            ball_order = {'W': 0, 'Y': 1, 'R': 2}
                            ball_pair = ''.join(sorted([current_ball_color, upcoming_act_partner], 
                                               key=lambda x: ball_order[x]))
                            
                            # Check if this collision exists in both act_events and sim_events
                            collision_found_in_both = False
                            time_tolerance = 0.1  # Allow small time differences due to numerical precision
                              # Find matching collision in actual events
                            act_indices = [i for i, hit in enumerate(act_events["hits"]) if hit == ball_pair]
                            sim_indices = [i for i, hit in enumerate(sim_events["hits"]) if hit == ball_pair]
                            
                            # Check if there's a time match within tolerance
                            for act_idx in act_indices:
                                act_event_time = act_events["t"][act_idx]
                                if abs(act_event_time - upcoming_act_time) < time_tolerance:
                                    # Found matching actual event, now check simulation
                                    for sim_idx in sim_indices:
                                        sim_event_time = sim_events["t"][sim_idx]
                                        if (abs(sim_event_time - upcoming_sim_time) < time_tolerance and
                                            abs(act_event_time - sim_event_time) < time_tolerance):
                                            collision_found_in_both = True
                                            break
                                    if collision_found_in_both:
                                        break
                            
                            if not collision_found_in_both:
                                correct_hit = False
                                loss_hit = loss_distance
            else:
                correct_hit = False
                loss_hit = loss_hit_std
            
        else:
            # wrong hit happened, soo following events must be wrong
            loss_hit = loss_hit_std
            loss_distance = 0
        
            current_time = np.max([t_act1, t_sim1])

        # assign the loss
        losses["ball"][balli]["time"].append(ei)
        losses["ball"][balli]["total"].append(loss_hit + loss_distance)

    return losses

def distance_loss_event(ei, sim_t, sim_x, sim_y, sim_hit, act_t, act_x, act_y, act_hit):

    if ei <= len(act_hit["with"])-2:
        event_time = act_hit["t"][ei]
        t_act0_ind = np.where(act_t >= event_time)[0][0]
        event_time = act_hit["t"][ei+1]
        t_act1_ind = np.where(act_t >= event_time)[0][0]
    elif ei <= len(act_hit["with"])-1: 
        event_time = act_hit["t"][ei]
        t_act0_ind = np.where(act_t >= event_time)[0][0]
        event_time = act_t[-1]
        t_act1_ind = np.where(act_t >= event_time)[0][0]


    if ei <= len(sim_hit["with"])-2:
        event_time = sim_hit["t"][ei]
        t_sim0_ind = np.where(sim_t >= event_time)[0][0]
        event_time = sim_hit["t"][ei+1]
        t_sim1_ind = np.where(sim_t >= event_time)[0][0]

    elif ei <= len(sim_hit["with"])-1:
        event_time = sim_hit["t"][ei]
        t_sim0_ind = np.where(sim_t >= event_time)[0][0]
        event_time = sim_t[-1]
        t_sim1_ind = np.where(sim_t >= event_time)[0][0]


    t0 = sim_t[t_sim0_ind:t_sim1_ind]
    x0 = sim_x[t_sim0_ind:t_sim1_ind]
    y0 = sim_y[t_sim0_ind:t_sim1_ind]

    t1 = act_t[t_act0_ind:t_act1_ind]
    x1 = act_x[t_act0_ind:t_act1_ind]
    y1 = act_y[t_act0_ind:t_act1_ind]

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
    cushion_numbers = ['1', '2', '3', '4']
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

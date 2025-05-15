import numpy as np
from scipy.interpolate import interp1d


def calculate_distance_loss(act_t, act_x, act_y, sim_t, sim_x, sim_y):

    # calculate loss based on which duration is longer
    tmax = max(act_t[-1], sim_t[-1])
    tmin = min(act_t[0], sim_t[0])

    tloss = np.linspace(tmin, tmax, 200)

    sim_x_interp = interpolate_shot(sim_x, sim_t, tloss)
    sim_y_interp = interpolate_shot(sim_y, sim_t, tloss)
    act_x_interp = interpolate_shot(act_x, act_t, tloss)
    act_y_interp = interpolate_shot(act_y, act_t, tloss)

    # calculate loss
    loss_dist = np.sqrt((act_x_interp - sim_x_interp) ** 2 + (act_y_interp - sim_y_interp) ** 2)/2.84/len(tloss)
    loss = np.sum(loss_dist)

    return loss



def calculate_angle_loss(xsim, ysim, xact, yact):
    # Calculate the angle between the two vectors
    dx_sim = xsim[-1]- xsim[0]
    dy_sim = ysim[-1]- ysim[0]
    dx_act = xact[-1] - xact[0]
    dy_act = yact[-1] - yact[0]

    # Calculate the angle using the dot product formula
    dot_product = dx_sim * dx_act + dy_sim * dy_act
    norm_sim = np.sqrt(dx_sim**2 + dy_sim**2)
    norm_actual = np.sqrt(dx_act**2 + dy_act**2)

    cos_angle = dot_product / (norm_sim * norm_actual + 1e-10)  # Avoid division by zero

    # Ensure the angle is in the range [0, 1]
    angle = np.abs(np.arccos(np.clip(cos_angle, -1.0, 1.0))) / np.pi

    return angle

def evaluate_loss(sim_env, shot_actual, method="combined_straighten_route"):
    
    sim_t, white_rvw, yellow_rvw, red_rvw = sim_env.get_ball_routes()
    sim_hit_all = sim_env.get_events(sim_env)
    
    balls_rvw = [white_rvw, yellow_rvw, red_rvw]
    
    losses = {}
    losses["ball"] = [{} for _ in range(3)]

    # loop over all balls
    for balli in range(3):
        losses["ball"][balli]["time"] = []
        losses["ball"][balli]["hit"] = []
        losses["ball"][balli]["angle"] = []
        losses["ball"][balli]["distance"] = []
        losses["ball"][balli]["total"] = []
        
        sim_hit = sim_hit_all[balli]
        act_hit = shot_actual["hit"][balli]
    
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

        correct_hit = True

        # run through events based on actual data and simulation data
        for ei in range(max(len(act_hit["with"]), len(sim_hit["with"]))):
            if ei < len(act_hit["with"]) and act_hit["with"][ei] == "-":
                # assign the loss
                losses["ball"][balli]["time"].append(0)
                losses["ball"][balli]["hit"].append(0)
                losses["ball"][balli]["angle"].append(0)
                losses["ball"][balli]["distance"].append(0)
                losses["ball"][balli]["total"].append(0)
                continue

            # find the time index of the current event in actual data
            if ei <= len(act_hit["with"])-1 and ei <= len(sim_hit["with"])-1:            
                # check if hit partner is same
                if act_hit["with"][ei] == sim_hit["with"][ei] and correct_hit == True:                    
                    correct_hit = True
                    loss_hit = 0.0
                

                    act_event_time = act_hit["t"][ei]
                    sim_event_time = sim_hit["t"][ei]
                    current_act_event_time_index = np.where(act_t >= act_event_time)[0][0]
                    current_sim_event_time_index = np.where(sim_t >= sim_event_time)[0][0]

                    current_act_time = act_t[current_act_event_time_index]
                    current_sim_time = sim_t[current_sim_event_time_index]

                    current_time = np.max([current_act_time, current_sim_time])
                    
                    # find the time of the next event in actual data
                    # either as next event or last time step
                    if ei < len(act_hit["with"])-1:
                        event_time = act_hit["t"][ei + 1]
                    else:
                        event_time = act_t[-1]
                        correct_hit = False
                    next_act_event_time_index = np.where(act_t >= event_time)[0][0]

                    # find the time of the next event in simulation data
                    if ei < len(sim_hit["with"])-1:
                        event_time = sim_hit["t"][ei + 1]
                    else:
                        event_time = sim_t[-1]
                        correct_hit = False
                    next_sim_event_time_index = np.where(sim_t >= event_time)[0][0]


                    if len(sim_x) >= 2 and len(act_x) >= 2 and current_sim_event_time_index < next_sim_event_time_index and current_act_event_time_index < next_act_event_time_index:
                        # calculate the angle between act and sim data
                        loss_angle = calculate_angle_loss(sim_x[current_sim_event_time_index:next_sim_event_time_index], 
                                                        sim_y[current_sim_event_time_index:next_sim_event_time_index], 
                                                        act_x[current_act_event_time_index:next_act_event_time_index], 
                                                        act_y[current_act_event_time_index:next_act_event_time_index])
                        
                        loss_distance = calculate_distance_loss(sim_t[current_sim_event_time_index:next_sim_event_time_index], 
                                                        sim_x[current_sim_event_time_index:next_sim_event_time_index], 
                                                        sim_y[current_sim_event_time_index:next_sim_event_time_index], 
                                                        act_t[current_act_event_time_index:next_act_event_time_index], 
                                                        act_x[current_act_event_time_index:next_act_event_time_index], 
                                                        act_y[current_act_event_time_index:next_act_event_time_index])
                    else:
                        loss_angle = 1.0
                        loss_distance = 1.0
                
                else:
                    correct_hit = False

            if correct_hit == False:
                loss_hit = 1.0
                loss_angle = 1.0
                loss_distance = 1.0

                # find the current time of the event in actual data
                if ei <= len(act_hit["with"])-1:
                    event_time = act_hit["t"][ei]
                    event_time_index = np.where(act_t >= act_event_time)[0][0]
                    current_time = act_t[event_time_index]
                elif ei <= len(sim_hit["with"])-1:
                    event_time = sim_hit["t"][ei]
                    event_time_index = np.where(sim_t >= event_time)[0][0]
                    current_time = sim_t[event_time_index]



            # assign the loss
            losses["ball"][balli]["time"].append(current_time)
            losses["ball"][balli]["hit"].append(loss_hit)
            losses["ball"][balli]["angle"].append(loss_angle)
            losses["ball"][balli]["distance"].append(loss_distance)
            losses["ball"][balli]["total"].append(loss_hit + loss_angle + loss_distance)
    
    
    return losses


def interpolate_shot(simulated, tsim, actual_times):
    interp_func = interp1d(
        tsim,
        simulated,
        kind="linear",
        bounds_error=False,
        fill_value=(simulated[-1]),
    )
    interpolated = interp_func(actual_times)
    return interpolated



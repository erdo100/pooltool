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
        
        elif method == "eventbased":
            
            losses =  loss_fun_eventbased(balli, losses, sim_t, sim_x, sim_y, 
                                       act_t, act_x, act_y, sim_hit_all[balli], shot_actual["hit"][balli])
        
        # calculate total loss for the ball
        losses["total"] += np.sum(losses["ball"][balli]["total"]) 
        # print("ball", balli, ", loss=", losses["ball"][balli]["total"])

    return losses

def loss_func_distance(balli, losses, t_interp, sim_x_interp, sim_y_interp, act_x_interp, act_y_interp):

    loss_distance = np.sqrt((act_x_interp - sim_x_interp) ** 2 + (act_y_interp - sim_y_interp) ** 2)

    losses["ball"][balli]["time"] = t_interp
    losses["ball"][balli]["total"] = loss_distance
    losses["total"] += np.sum(losses["ball"][balli]["total"]) 

    return losses

def loss_fun_eventbased(balli, losses, sim_t, sim_x, sim_y, act_t, act_x, act_y, sim_hit, act_hit):
        
    correct_hit = True
    loss_hit = 0
    loss_distance = 0
    current_time = 0

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

            if len(act_hit["with"]) == 1 and len(sim_hit["with"]) > 1:
                correct_hit = False
                loss_hit = 10.0

            if len(act_hit["with"]) > 1 and len(sim_hit["with"]) == 1:
                correct_hit = False
                loss_hit = 10.0

            if len(act_hit["with"]) == 1 and len(sim_hit["with"]) == 1:
                loss_hit = 0.0

            # case when more than 1 events left in both
            if ei < len(act_hit["with"])-1 and ei < len(sim_hit["with"])-1:

                # calculate the distance loss
                loss_distance, current_time = distance_loss_event(ei, sim_t, sim_x, sim_y, sim_hit, act_t, act_x, act_y, act_hit)

                if ei < len(act_hit["with"])-2 and ei < len(sim_hit["with"])-2:
                    # Check if the upcoming event is same?
                        if act_hit["with"][ei+1] != sim_hit["with"][ei+1]:
                            correct_hit = False
                            loss_hit = 10.0
            else:
                correct_hit = False
                loss_hit = 10.0
            
        else:
            # wrong hit happened, soo following events must be wrong
            loss_hit = 10.0
            loss_distance = 0
        
            current_time = np.max([t_act1, t_sim1])

        # assign the loss
        losses["ball"][balli]["time"].append(current_time)
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



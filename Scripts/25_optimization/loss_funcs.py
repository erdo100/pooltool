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
    # total length sqrt(deltax**2+dy**2)
    act_length = np.sqrt(act_x**2 + act_y**2)
    sim_length = np.sqrt(sim_x**2 + sim_y**2)

    # normalize by the length of the actual and simulated path
    dist = dist / sim_length

    return dist


def calculate_angle_loss(t, xsim, ysim, xact, yact):

    # calculate angle of velocities for actual and simulated for each time step
    # additional data point to maintain length
    actual_vx = np.diff(xact) / np.diff(t)
    actual_vx = np.insert(actual_vx, 0, 0)
    actual_vy = np.diff(yact) / np.diff(t)
    actual_vy = np.insert(actual_vy, 0, 0)
    simulated_vx = np.diff(xsim) / np.diff(t)
    simulated_vx = np.insert(simulated_vx, 0, 0)
    simulated_vy = np.diff(ysim) / np.diff(t)
    simulated_vy = np.insert(simulated_vy, 0, 0)
    # use cosine law to calculate angle between actual and simulated velocity
    actual_v = np.sqrt(actual_vx ** 2 + actual_vy ** 2)
    simulated_v = np.sqrt(simulated_vx ** 2 + simulated_vy ** 2)
    
    # avoid division by zero
    eps = 1e-10
    # store index with velocity smaller eps in v0index
    v0actual_index = np.where(actual_v < eps)[0]
    v0simulated_index = np.where(simulated_v < eps)[0]
    v0both = np.intersect1d(v0actual_index, v0simulated_index)

    actual_v[v0actual_index] = eps
    simulated_v[v0simulated_index] = eps

    # calculate cosine of angle between actual and simulated velocity
    cos_angle = (actual_vx * simulated_vx + actual_vy * simulated_vy) / (actual_v * simulated_v)
    # if both balls are moving use the angle between the two velocities
    # else use 180 degrees
    # therefore replace the cosangle values with 180 for v0actual_index and v0simulated_index

    angle = np.arccos(cos_angle) / np.pi
        
        
    # replace the values for v0actual_index and v0simulated_index with 180 degrees
    angle[v0actual_index] = 1
    angle[v0simulated_index] = 1
    # replace the values for v0both with 0 degrees
    angle[v0both] = 0

    return (angle)**2

def evaluate_loss(sim_env, shot_actual, method="combined_distance_angle"):
    
    sim_t, white_rvw, yellow_rvw, red_rvw = sim_env.get_ball_routes()
    
    balls_rvw = [white_rvw, yellow_rvw, red_rvw]
    
    losses = {}
    losses["ball"] = [{} for _ in range(3)]
    losses["total"] = 0

    # loop over all balls
    for balli in range(3):
        losses["ball"][balli]["time"] = []
        losses["ball"][balli]["angle"] = []
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

        dt_interp = np.diff(t_interp)
        # add last dt to the end of the array to make it the same length as x and y
        dt_interp = np.insert(dt_interp, len(dt_interp), dt_interp[-1])

        # run through events based on actual data and simulation data
        # calculate the angle between act and sim data
        loss_angle = calculate_angle_loss(t_interp, sim_x_interp, sim_y_interp, act_x_interp, act_y_interp)

        loss_distance = calculate_distance_loss(t_interp, sim_x_interp, sim_y_interp, act_x_interp, act_y_interp)


        losses["ball"][balli]["time"] = t_interp
        losses["ball"][balli]["angle"] = loss_angle
        losses["ball"][balli]["distance"] = loss_distance#/dt_interp
        #losses["ball"][balli]["total"] = loss_angle/(2+t_interp) + loss_distance*3 + loss_angle*loss_distance
        losses["ball"][balli]["total"] = losses["ball"][balli]["distance"]

        losses["total"] += np.sum(losses["ball"][balli]["total"]) 

    
    return losses


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



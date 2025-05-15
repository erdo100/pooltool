import numpy as np
import matplotlib.pyplot as plt
import pooltool as pt
from billiardenv import BilliardEnv
from helper_funcs import get_ball_positions, read_shotfile, interpolate_simulated_to_actual, loss_func, save_parameters, load_parameters, save_system, abs_velocity, initial_shot_direction
import time

def run_study(shots_actual, params):
    print('Running study with parameters:')
    # Retrieve current shot and slider values
    shot_id = params.value['shot_id']
    shot_actual = shots_actual[shot_id]
    ball_xy_ini, ball_ids, ball_cols, cue_phi = get_ball_positions(shot_actual)

    # update the actual parameters
    a = params.value['shot_a']
    b = params.value['shot_b']
    phi = params.value['shot_phi']
    v = params.value['shot_v']
    theta = params.value['shot_theta']

    a_ballball = params.value['ballball_a']
    b_ballball = params.value['ballball_b']
    c_ballball = params.value['ballball_c']

    u_slide = params.value['physics_u_slide']
    u_roll = params.value['physics_u_roll']
    u_sp_prop = params.value['physics_u_sp_prop']
    e_ballball = params.value['physics_e_ballball']
    e_cushion = params.value['physics_e_cushion']
    f_cushion = params.value['physics_f_cushion']
    h_cushion = params.value['physics_h_cushion']

    ball_xy_ini[1] = (-0.5, 0.5)
    ball_xy_ini[2] = (-0.5, 0.5)

    # Create billiard environment and simulate shot
    shot_env = BilliardEnv(u_slide, u_roll, u_sp_prop, e_ballball, e_cushion, f_cushion, h_cushion)
    hitpointx = np.zeros(500)
    hitpointy = np.zeros(500)
    phi_range = np.linspace(57.0,73.0, 500)
    
    import time
    start_time = time.time()
    
    for i, phi in enumerate(phi_range):
        print(f"Running simulation {i+1} of {len(phi_range)}")
        shot_env.prepare_new_shot(ball_cols, ball_xy_ini, a, b, phi, v, theta)
        system = shot_env.simulate_shot(a_ballball, b_ballball, c_ballball)

        # identify last cushion collision
        collision = pt.events.filter_type(system.events, pt.EventType.BALL_LINEAR_CUSHION)[4]
        hitpointx[i] = collision.get_ball("white",initial=True).state.rvw[0,0]
        hitpointy[i] = collision.get_ball("white",initial=True).state.rvw[0,1]
    
    total_runtime = time.time() - start_time
    
    trend = np.polyfit(phi_range, hitpointx, 1)
    hitpointx_detrended = hitpointx - np.polyval(trend, phi_range)
    std_hitpointx = np.std(hitpointx_detrended)
    print("Standard Deviation of detrended hitpointx:", std_hitpointx)
   
    # create a figure and axis
    fig, ax = plt.subplots()
    # ax.plot(phi_range, hitpointx_detrended)
    ax.plot(phi_range, hitpointx)
    ax.set_xlabel('phi')
    ax.set_ylabel('hitpoint')
    # ax.set_title(f"HAN2005, runtime: {total_runtime:.4f}s, STD: {std_hitpointx:.4f}")
    ax.set_title(f"Mathavan, N={1000}, dPmin={0.001}, runtime: {total_runtime:.4f}s, STD: {std_hitpointx:.4f}")
    plt.grid()
    plt.show()
    # save hitpointx to a file
    np.savetxt('hitpointx_C4.csv', hitpointx, delimiter=',')


def run_simulation(ball_cols, ball_xy_ini, params):
    u_slide = params.value['physics_u_slide']
    u_roll = params.value['physics_u_roll']
    u_sp_prop = params.value['physics_u_sp_prop']
    e_ballball = params.value['physics_e_ballball']
    e_cushion = params.value['physics_e_cushion']
    f_cushion = params.value['physics_f_cushion']
    h_cushion = params.value['physics_h_cushion']
    a_ballball = params.value['ballball_a']
    b_ballball = params.value['ballball_b']
    c_ballball = params.value['ballball_c']
    a = params.value['shot_a']
    b = params.value['shot_b']
    phi = params.value['shot_phi']
    v = params.value['shot_v']
    theta = params.value['shot_theta']

    # Create billiard environment and simulate shot
    shot_env = BilliardEnv(u_slide, u_roll, u_sp_prop, e_ballball, e_cushion, f_cushion, h_cushion)
    shot_env.prepare_new_shot(ball_cols, ball_xy_ini, a, b, phi, v, theta)
    system = shot_env.simulate_shot(a_ballball, b_ballball, c_ballball)

    # update the plots
    tsim, white_rvw, yellow_rvw, red_rvw = shot_env.get_ball_routes()

    return system, tsim, white_rvw, yellow_rvw, red_rvw
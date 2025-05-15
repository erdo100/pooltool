import matplotlib.pyplot as plt
import numpy as np
from billiardenv import BilliardEnv

class Parameters:
    def __init__(self):
        # Shot Parameter
        self.value = {}
        self.value = {
            'shot_id': 0,

            # Shot parameters
            'shot_a': 0.0,
            'shot_b': 0.0,
            'shot_phi': 89.0,
            'shot_v': 3.0,
            'shot_theta': 0.0,

            # Alciatori Ball-Ball hit model parameters
            # Friction fit curve u_b = a + b * exp(-c * v_rel) used in David Alciatore's TP A-14
            'physics_ballball_a': 0.1,
            'physics_ballball_b': 0.0,
            'physics_ballball_c': 1.1,

            # Physics parameters
            'physics_u_slide': 0.2,
            'physics_u_roll': 0.005,
            'physics_u_sp_prop': 0.5,
            'physics_e_ballball': 0.95,
            'physics_e_cushion': 0.9,
            'physics_f_cushion': 0.15,
            'physics_h_cushion': 0.037,

            "size": [1.420, 2.840],
            "ballR": 61.5 / 2000
        }

        self.limits = {}
        self.limits = {
            # Shot parameters
            'shot_a': (-0.5, 0.5),
            'shot_b': (-0.5, 0.5),
            'shot_phi': (-180.0, 180.0),
            'shot_v': (1.0, 7.0),
            'shot_theta': (0.0, 90.0),

            # Alciatori Ball-Ball hit model parameters
            # Friction fit curve u_b = a + b * exp(-c * v_rel) used in David Alciatore's TP A-14
            'physics_ballball_a': (0.0, 0.1),
            'physics_ballball_b': (0.0, 1),
            'physics_ballball_c': (0.0, 5.0),

            # Physics parameters
            'physics_u_slide': (0.0, 0.3),
            'physics_u_roll': (0.0, 0.015),
            'physics_u_sp_prop': (0.0, 1.0),
            'physics_e_ballball': (0.0, 1.0),
            'physics_e_cushion': (0.0, 1.0),
            'physics_f_cushion': (0.0, 0.5),
            'physics_h_cushion': (0.033, 0.0341)
        }


def setup_axes(params):
    
    # initialize figure
    fig, ax = plt.subplots(1, 2, figsize=(16, 14))

    ax[0].set_xlim(0, params['size'][0])
    ax[0].set_ylim(0, params['size'][1])
    ax[0].set_xlabel('X-axis (m)')
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_xticks(np.linspace(0, params['size'][0], 5))
    ax[0].set_yticks(np.linspace(0, params['size'][1], 9))
    ax[0].grid(True, linestyle='--', linewidth=0.8, color='gray')
    ax[0].set_facecolor((0.4, 0.4, 1.0))
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].tick_params(axis='both', which='both', length=0)

    return fig, ax


# main function to run the simulation
def main():
    params = Parameters()
    sim_env = BilliardEnv()

    fig, ax = setup_axes(params.value)

    # set new parameters and simulate shot
    ball_cols = {
        0: "white",
        1: "yellow",
        2: "red"
}

    ball_xy_ini = {
        0: (2/4*params.value['size'][0], 2/8*params.value['size'][1]),  # White ball
        1: (2/4*params.value['size'][0], 3.3/8*params.value['size'][1]),  # Yellow ball
        2: (2/4*params.value['size'][0], 5.3/8*params.value['size'][1])   # Red ball
    }


    sim_env.ball_cols = ball_cols
    sim_env.balls_xy_ini = ball_xy_ini

    # plot the balls initial positions with patch and circles
    for i, ball in enumerate(ball_xy_ini):
        print(f"Ball {i}: {sim_env.ball_cols[i]}")
        ball_xy = ball_xy_ini[ball]
        circle = plt.Circle(ball_xy, params.value['ballR'], color=sim_env.ball_cols[i])
        ax[0].add_patch(circle)


    tmax = 0.5
    # params.value['shot_v'] = 3.3
    # params.value['shot_theta'] = 15

    # params.value['shot_a'] = 0.4
    params.value['shot_b'] = -0.3
    phi0 = 90.0

    # add title to ax[0]
    # addd shot_v, shot;thetha, shot_a, shot_b to title
    # add in new line the text of engine.ballball collision modell
    
    ax[0].set_title(f"a={params.value['shot_a']}, b={params.value['shot_b']}, v={params.value['shot_v']}, theta={params.value['shot_theta']}, tmax={tmax}\nBall-Ball Collision Model: MATHAVAN", fontsize=14, color='black')

    # loop phi from 85 to 89 degrees
    dphi = 10
    for phi in np.linspace(phi0-dphi, phi0+dphi, 50):
        params.value['shot_phi'] = phi
        sim_env.prepare_new_shot(params)
        sim_env.simulate_shot()

        tsim, white_rvw, yellow_rvw, red_rvw = sim_env.get_ball_routes()

        for i, rvw in enumerate([white_rvw, yellow_rvw, red_rvw]):
            # Simulated shot
            xs = rvw[:, 0, 0]
            ys = rvw[:, 0, 1]

            vx = rvw[:, 1, 0]
            vy = rvw[:, 1, 1]
            v = np.sqrt(vx**2 + vy**2)

            # plot until t = tmax
            t = tsim[0:rvw.shape[0]]

            xs = xs[0:rvw.shape[0]][t < tmax]
            ys = ys[0:rvw.shape[0]][t < tmax]
            v = v[0:rvw.shape[0]][t < tmax]
            t = t[t < tmax]        
            ax[0].plot(xs, ys, label=sim_env.ball_cols[i], color=sim_env.ball_cols[i], linewidth=1.0)


    # draw plot
    plt.draw()
    plt.pause(0.001)

    # Keep the figure window open
    plt.show()

if __name__ == "__main__":
    main()

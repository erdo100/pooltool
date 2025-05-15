import matplotlib.pyplot as plt
import numpy as np
import pooltool as pt

def setup_axes(template):    
    # initialize figure
    fig, ax = plt.subplots(1, 2, figsize=(16, 14))

    ax[0].set_xlim(0, template.table.w)
    ax[0].set_ylim(0, template.table.l)
    ax[0].set_xlabel('X-axis (m)')
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_xticks(np.linspace(0, template.table.w, 5))
    ax[0].set_yticks(np.linspace(0, template.table.l, 9))
    ax[0].grid(True, linestyle='--', linewidth=0.8, color='gray')
    ax[0].set_facecolor((0.4, 0.4, 1.0))
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].tick_params(axis='both', which='both', length=0)

    return fig, ax


# main function to run the simulation
def main():

    template = pt.System.example()
    template.balls["cue"].state.rvw[0, 0] = 0.3
    template.cue.set_state(a=0.4, b=0.126, phi=pt.aim.at_ball(template, "1") + 2.5, V0=3.3, theta=15.0)

    fig, ax = setup_axes(template)
    tmax = 1.0
    col = ['white', 'yellow']
    systems = []
    for phi_delta in np.linspace(5.8, 6.8, 4):
        system = template.copy()
        system.cue.set_state(phi=(pt.aim.at_ball(system, "1") + phi_delta))
        shot = pt.simulate(system, inplace=True, continuous=True, dt=0.01)

        shotcont = pt.continuize(shot, dt=0.01, inplace=False)
        white = shotcont.balls["cue"]
        white_history = white.history_cts
        white_rvw, s_cue, tsim = white_history.vectorize()

        yellow = shotcont.balls["1"]
        yellow_history = yellow.history_cts
        yellow_rvw, s_cue, tsim = yellow_history.vectorize()


        for i, rvw in enumerate([white_rvw, yellow_rvw]):
            # Simulated shot
            xs = rvw[:, 0, 0]
            ys = rvw[:, 0, 1]

            vx = rvw[:, 1, 0]
            vy = rvw[:, 1, 1]
            v = np.sqrt(vx**2 + vy**2)

            # plot until t = 0.5s
            t = tsim[0:rvw.shape[0]]

            xs = xs[0:rvw.shape[0]][t < tmax]
            ys = ys[0:rvw.shape[0]][t < tmax]
            v = v[0:rvw.shape[0]][t < tmax]
            t = t[t < tmax]        
            ax[0].plot(xs, ys, linewidth=1.0, color=col[i])

        circle = plt.Circle((xs[0], ys[0]), 0.05/2, color=col[i], alpha=0.5)
        ax[0].add_patch(circle)

    # draw plot
    plt.draw()
    plt.pause(0.001)

    # Keep the figure window open
    plt.show()

if __name__ == "__main__":
    main()

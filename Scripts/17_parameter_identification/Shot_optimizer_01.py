from tkinter import Tk
from GUI import plot_3cushion
from helper_funcs import open_shotfile
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
            'shot_phi': 0.0,
            'shot_v': 3.0,
            'shot_theta': 0.0,

            # Alciatori Ball-Ball hit model parameters
            # Friction fit curve u_b = a + b * exp(-c * v_rel) used in David Alciatore's TP A-14
            'physics_ballball_a': 0.01,
            'physics_ballball_b': 0.11,
            'physics_ballball_c': 1.1,

            # Physics parameters
            'physics_u_slide': 0.2,
            'physics_u_roll': 0.005,
            'physics_u_sp_prop': 0.5,
            'physics_e_ballball': 0.95,
            'physics_e_cushion': 0.9,
            'physics_f_cushion': 0.15,
            'physics_h_cushion': 0.037
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

params = Parameters()
sim_env = BilliardEnv()

# load data
# shots_actual = read_shotfile()

# # Select the current shot
# shot_actual = shots_actual[0]
# shot_id_changed = True

# GUI setup
plt3c = plot_3cushion(sim_env, params)
plt3c.root.mainloop()

'''
Shot 50:
[-0.07212, 0.22138, 106.29921, 2.80145, 6.44663, 0.01343, 0.00086, 1.90918, 0.19283, 0.00403, 0.20854, 0.91879, 0.91685, 0.27756, 0.03777], Loss: 2580.52202, Convergence: 0.01466
'''
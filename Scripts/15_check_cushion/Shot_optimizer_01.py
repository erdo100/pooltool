from tkinter import Tk
from GUI import plot_3cushion
from helper_funcs import read_shotfile

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
            'ballball_a': 0.009951,
            'ballball_b': 0.108,
            'ballball_c': 1.088,

            # Physics parameters
            'physics_u_slide': 0.2,
            'physics_u_roll': 0.005,
            'physics_u_sp_prop': 0.5,
            'physics_e_ballball': 0.95,
            'physics_e_cushion': 0.9,
            'physics_f_cushion': 0.15,
            'physics_u_ballball': 0.05, # not relevant for Alciatori Ball-Ball hit model
            'physics_h_cushion': 0.037
        }


params = Parameters()

# load data
# shots_actual = read_shotfile()

# # Select the current shot
# shot_actual = shots_actual[0]
# shot_id_changed = True

# GUI setup
plt3c = plot_3cushion(params)
plt3c.root.mainloop()
class Parameters:
    def __init__(self):
        # Shot Parameter
        self.value = {}
        self.value = {
            'shot_id': 0,

            # Shot parameters
            'shot_a': 0.0,
            'shot_b': 0.0,
            'shot_phi': 60.0,
            'shot_v': 3.0,
            'shot_theta': 0.0,

            # Alciatori Ball-Ball hit model parameters
            # Friction fit curve u_b = a + b * exp(-c * v_rel) used in David Alciatore's TP A-14
            'physics_ballball_a': 0.01,
            'physics_ballball_b': 0.0,
            'physics_ballball_c': 1.1,

            # Physics parameters
            'physics_u_slide': 0.22,
            'physics_u_roll': 0.005,
            'physics_u_sp_prop': 0.5,
            'physics_e_ballball': 0.95,
            'physics_e_cushion': 0.9,
            'physics_f_cushion': 0.15,
            'physics_h_cushion': 0.037,

            # others
            'squirt_throttle': 0.0
        }

        self.limits = {}
        self.limits = {
            # Shot parameters
            'shot_a': (-0.5, 0.5),
            'shot_b': (-0.5, 0.5),
            'shot_phi': (-185, 185),
            'shot_v': (2.0, 7.0),
            'shot_theta': (0.0, 10.0),

            'physics_ballball_a': (0.001, 0.3),
            'physics_ballball_b': (0.0, 1),
            'physics_ballball_c': (0.0, 5.0),

            # Physics parameters
            'physics_u_slide': (0., 0.25),
            'physics_u_roll': (0.004, 0.007),
            'physics_u_sp_prop': (0.1, 0.5),
            'physics_e_ballball': (0.5, 1.0),
            'physics_e_cushion': (0.5, 1.0),
            'physics_f_cushion': (0.1, 0.4),
            'physics_h_cushion': (0.03, 0.04)
        }

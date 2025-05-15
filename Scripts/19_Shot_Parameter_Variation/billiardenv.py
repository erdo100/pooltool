import numpy as np
import pooltool as pt
from pooltool.ruleset.three_cushion import is_point

class BilliardEnv:
    def __init__(self):
        self.table_width = 2.84  # Table dimensions (meters)
        self.table_height = 1.42

        # Ball Positions
        self.balls_xy_ini = {}
        self.balls_xy_ini[0] = (0.1, 0, 1)  # White
        self.balls_xy_ini[1] = (0.5, 0.5)  # Yellow
        self.balls_xy_ini[2] = (1.0, 1.0)  # Red

        self.grav = 9.81

        self.mball = 0.210
        self.Rball = 61.5 / 1000 / 2
        
        cue_mass = 0.576
        cue_len = 1.47
        cue_tip_R = 21.21 / 2000  # radius nickel=21.21 mm, dime=17.91 mm
        cue_tip_mass = 0.008


        # Build a table with default BILLIARD specs
        self.table = pt.Table.default(pt.TableType.BILLIARD)

        # create the cue
        cue_specs = pt.objects.CueSpecs(
            M=cue_mass,
            length=cue_len,
            tip_radius=cue_tip_R,
            end_mass=cue_tip_mass,
        )

        self.cue = pt.Cue(cue_ball_id="white", specs=cue_specs)


    def prepare_new_shot(self, params):
        
        # ball color and initial position
        for ball_col, ball_xy in self.balls_xy_ini.items():
            if ball_col == 0:#"white":
                ball_white_xy = ball_xy # White Ball
            elif ball_col == 1:#"yellow":
                ball_yellow_xy = ball_xy
            elif ball_col == 2:#"red":
                ball_red_xy = ball_xy

        # Create balls in new positions
        wball = pt.Ball.create(
            "white",
            xy=ball_white_xy,
            m=self.mball,
            R=self.Rball,
            u_s=params.value['physics_u_slide'],
            u_r=params.value['physics_u_roll'],
            u_sp_proportionality=params.value['physics_u_sp_prop'],
            u_b=0.1,
            e_b=params.value['physics_e_ballball'],
            e_c=params.value['physics_e_cushion'],
            f_c=params.value['physics_f_cushion'],
            g=self.grav,
        )

        yball = pt.Ball.create(
            "yellow",
            xy=ball_yellow_xy,
            m=self.mball,
            R=self.Rball,
            u_s=params.value['physics_u_slide'],
            u_r=params.value['physics_u_roll'],
            u_sp_proportionality=params.value['physics_u_sp_prop'],
            u_b=0.1,
            e_b=params.value['physics_e_ballball'],
            e_c=params.value['physics_e_cushion'],
            f_c=params.value['physics_f_cushion'],
            g=self.grav,
        )        
        
        rball = pt.Ball.create(
            "red",
            xy=ball_red_xy,
            m=self.mball,
            R=self.Rball,
            u_s=params.value['physics_u_slide'],
            u_r=params.value['physics_u_roll'],
            u_sp_proportionality=params.value['physics_u_sp_prop'],
            u_b=0.1,
            e_b=params.value['physics_e_ballball'],
            e_c=params.value['physics_e_cushion'],
            f_c=params.value['physics_f_cushion'],
            g=self.grav,
        )

        # modify the cue ball in self.cue
        self.cue.cue_ball_id = self.ball_cols[0]

        # phi = pt.aim.at_ball(self.system, "red", cut=cut)
        # set the cue
        self.cue.set_state(a=params.value['shot_a'],
                           b=params.value['shot_b'],
                           V0=params.value['shot_v'],
                           phi=params.value['shot_phi'],
                           theta=params.value['shot_theta']
                           )
        # modify cushion height
        specs = pt.objects.BilliardTableSpecs(cushion_height=params.value['physics_h_cushion'])
        self.table = pt.Table.from_table_specs(specs)


        # Wrap it up as a System
        self.system = pt.System(
            table=self.table, balls=(wball, yball, rball), cue=self.cue
        )

        self.engine = pt.physics.PhysicsEngine()  # start with default
        # Friction fit curve u_b = a + b * exp(-c * v_rel) used in David Alciatore's TP A-14
        self.engine.resolver.ball_ball = pt.physics.ball_ball_models[pt.physics.BallBallModel.FRICTIONAL_INELASTIC]()

        self.engine.resolver.ball_ball.friction.a = params.value['physics_ballball_a']
        self.engine.resolver.ball_ball.friction.b = params.value['physics_ballball_b']
        self.engine.resolver.ball_ball.friction.c = params.value['physics_ballball_c']


        self.engine.resolver.stick_ball.squirt_throttle = 0


        # engine.resolver.ball_linear_cushion = pt.physics.ball_lcushion_models[pt.physics.BallLCushionModel.MATHAVAN_2010]()

    def simulate_shot(self):
        # run the physics model
    
        # pt.serialize.conversion.unstructure_to(engine, "engine.yaml")
        # print("engine saved to engine.yaml")

        # Pass the engine to your simulate call.
        pt.simulate(self.system, engine=self.engine, inplace=True)

        return self.system
    

    def get_ball_routes(self):
        shot = self.system
        shotcont = pt.continuize(shot, dt=0.01, inplace=False)
        white = shotcont.balls["white"]
        white_history = white.history_cts
        white_rvw, s_cue, tsim = white_history.vectorize()
        yellow = shotcont.balls["yellow"]
        yellow_history = yellow.history_cts
        yellow_rvw, s_cue, tsim = yellow_history.vectorize()
        red = shotcont.balls["red"]
        red_history = red.history_cts
        red_rvw, s_cue, tsim = red_history.vectorize()

        return tsim, white_rvw, yellow_rvw, red_rvw


    def run_simulation(self, ball_cols, ball_xy_ini, params):

        # set new parameters and simulate shot
        self.balls_xy_ini = ball_xy_ini
        self.prepare_new_shot(params)
        system = self.simulate_shot()

        # update the plots
        tsim, white_rvw, yellow_rvw, red_rvw = self.get_ball_routes()

        return system, tsim, white_rvw, yellow_rvw, red_rvw
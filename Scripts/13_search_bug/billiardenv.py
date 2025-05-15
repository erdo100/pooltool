import numpy as np
import pooltool as pt
from pooltool.ruleset.three_cushion import is_point

class BilliardEnv:
    def __init__(
        self, u_slide, u_roll, u_sp_prop, e_ballball, e_cushion, f_cushion
    ):
        self.table_width = 2.84  # Table dimensions (meters)
        self.table_height = 1.42
        self.series_length = 0
        self.current_step = 0
        self.episode_rewards = []

        # Ball Positions
        self.ball1_ini = (0.1, 0, 1)  # White
        self.ball2_ini = (0.5, 0.5)  # Yellow
        self.ball3_ini = (1.0, 1.0)  # Red

        # define the properties
        self.u_slide = u_slide
        self.u_roll = u_roll
        self.u_sp_prop = u_sp_prop
        self.u_ballball = 0.1
        self.e_ballball = e_ballball
        self.e_cushion = e_cushion
        self.f_cushion = f_cushion

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

    def prepare_new_shot(self, ball_cols, ball_xy_ini, a, b, phi, v, theta):
        
        for ball_col, ball_xy in ball_xy_ini.items():
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
            u_s=self.u_slide,
            u_r=self.u_roll,
            u_sp_proportionality=self.u_sp_prop,
            u_b=self.u_ballball,
            e_b=self.e_ballball,
            e_c=self.e_cushion,
            f_c=self.f_cushion,
            g=self.grav,
        )

        yball = pt.Ball.create(
            "yellow",
            xy=ball_yellow_xy,
            m=self.mball,
            R=self.Rball,
            u_s=self.u_slide,
            u_r=self.u_roll,
            u_sp_proportionality=self.u_sp_prop,
            u_b=self.u_ballball,
            e_b=self.e_ballball,
            e_c=self.e_cushion,
            f_c=self.f_cushion,
            g=self.grav,
        )
        
        rball = pt.Ball.create(
            "red",
            xy=ball_red_xy,
            m=self.mball,
            R=self.Rball,
            u_s=self.u_slide,
            u_r=self.u_roll,
            u_sp_proportionality=self.u_sp_prop,
            u_b=self.u_ballball,
            e_b=self.e_ballball,
            e_c=self.e_cushion,
            f_c=self.f_cushion,
            g=self.grav,
        )

        # modify the cue ball in self.cue
        self.cue.cue_ball_id = ball_cols[0]

        # phi = pt.aim.at_ball(self.system, "red", cut=cut)
        # set the cue
        self.cue.set_state(a=a, b=b, V0=v, phi=phi, theta=theta)

        # Wrap it up as a System
        self.system = pt.System(
            table=self.table, balls=(wball, yball, rball), cue=self.cue
        )

    def get_ball_routes(self):
        shot = self.system
        shotcont = pt.continuize(shot, dt=0.0025, inplace=False)
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

    def simulate_shot(self, a, b, c):
        # run the physics model
        point = 0

        engine = pt.physics.PhysicsEngine()  # start with default
        engine.resolver.stick_ball.squirt_throttle = 0.0
        engine.resolver.ball_linear_cushion = pt.physics.ball_lcushion_models[
            pt.physics.BallLCushionModel.MATHAVAN_2010
        ]() # HAN_2005 and MATHAVAN_2010
        # Friction fit curve u_b = a + b * exp(-c * v_rel) used in David Alciatore's TP A-14
        engine.resolver.ball_ball.friction.a = a
        engine.resolver.ball_ball.friction.b = b
        engine.resolver.ball_ball.friction.c = c
    
        # pt.serialize.conversion.unstructure_to(engine, "engine.yaml")
        # print("engine saved to engine.yaml")

        # Pass the engine to your simulate call.
        pt.simulate(self.system, engine=engine, inplace=True)

        return self.system
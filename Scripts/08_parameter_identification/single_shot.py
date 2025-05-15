from threecushion_shot import BilliardEnv

# physics parameters
u_slide = 0.15
u_roll = 0.005
u_sp_prop = 10 * 2 / 5 / 9
u_ballball = 0.05
e_ballball = 0.95
e_cushion = 0.9
f_cushion = 0.15

# create billard environment
shot = BilliardEnv(
    u_slide, u_roll, u_sp_prop, u_ballball, e_ballball, e_cushion, f_cushion
)

# set up shot
diamond = 2.84 / 8
ball1xy = (0.5 * diamond, 0.5 * diamond)
ball2xy = (2 * diamond, 10 * diamond)
ball3xy = (6 * diamond, 10 * diamond)

a = 0.5
b = 0.0
phi = 35.0
v = 3.0
theta = 0.0

# for theta in np.linspace(0, 8, 17):
# for b in np.linspace(-.5, 0.5, 11):
# prepare and simulate shot
shot.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, phi, v, theta)

result = shot.simulate_shot()

# print(result)

shot.plot_shot()

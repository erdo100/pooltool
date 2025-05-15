from threecushion_shot import BilliardEnv

shot = BilliardEnv()

ball1xy = (0.1, 2.2)
ball2xy = (0.75, 2.6)
ball3xy = (2.9, 2.9)
a = -0.4
b = 0.0
cut = 89.0
v = 2.5
theta = 0.0

shot.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, cut, v, theta)

result = shot.simulate_shot()

print(result)

shot.plot_shot()

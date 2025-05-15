#! /usr/bin/env python
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

import pooltool as pt
from pooltool.events.datatypes import EventType
from pooltool.events.filter import (
    filter_ball,
    filter_type,
)
from pooltool.ruleset.three_cushion import is_point

# Hyperparameters
STATE_DIM = 24  # Input size (positions, distances, directions, etc.)
ACTION_DIM = 5  # Output size (v, phi, theta, a, b)
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.005  # For soft updates
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
torch.autograd.set_detect_anomaly(True)


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)  # Outputs continuous action values

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.out(x))  # Actions in [-1, 1]


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)  # Outputs Q-value

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # def sample(self, batch_size):
    #     batch = random.sample(self.buffer, batch_size)
    #     states, actions, rewards, next_states, dones = zip(*batch)

    #     return (torch.tensor(states, dtype=torch.float32),
    #             torch.tensor(actions, dtype=torch.float32),
    #             torch.tensor(rewards, dtype=torch.float32),
    #             torch.tensor(next_states, dtype=torch.float32),
    #             torch.tensor(dones, dtype=torch.float32))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert each list of arrays into a numpy array, then into a tensor
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # Initialize target networks with same weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, state, noise=0.1):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action = action + noise * np.random.normal(
            size=action.shape
        )  # Add exploration noise
        return np.clip(action, -1, 1)

    def train_step(self, state, action, reward, next_state, done):
        """
        Stepwise update: Train the agent immediately after collecting a transition.
        """
        # Store transition in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Ensure enough samples in buffer for training
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            BATCH_SIZE
        )

        # # Convert to tensors
        # states = torch.tensor(states.clone().detach(), dtype=torch.float32)
        # actions = torch.tensor(actions.clone().detach(), dtype=torch.float32)
        # rewards = torch.tensor(rewards.clone().detach(), dtype=torch.float32).unsqueeze(1)
        # next_states = torch.tensor(next_states.clone().detach(), dtype=torch.float32)
        # dones = torch.tensor(dones.clone().detach(), dtype=torch.float32).unsqueeze(1)

        # Critic loss
        next_actions = self.actor_target(next_states)
        target_q = rewards.unsqueeze(1) + GAMMA * self.critic_target(
            next_states, next_actions
        ).detach() * (1 - dones.unsqueeze(1))
        # target_q = rewards + GAMMA * self.critic_target(next_states, next_actions) * (1 - dones)
        current_q = self.critic(states, actions)

        # Debugging shapes
        print(f"states shape: {states.shape}")  # Expected: [BATCH_SIZE, STATE_DIM]
        print(f"actions shape: {actions.shape}")  # Expected: [BATCH_SIZE, ACTION_DIM]
        print(f"rewards shape: {rewards.shape}")  # Expected: [BATCH_SIZE, 1]
        print(
            f"next_states shape: {next_states.shape}"
        )  # Expected: [BATCH_SIZE, STATE_DIM]
        print(f"dones shape: {dones.shape}")  # Expected: [BATCH_SIZE, 1]

        print(f"current_q shape: {current_q.shape}")  # Expected: [BATCH_SIZE, 1]
        print(f"target_q shape: {target_q.shape}")  # Expected: [BATCH_SIZE, 1]

        critic_loss = nn.MSELoss()(current_q, target_q.detach())

        # Actor loss
        # actor_loss = -self.critic(states, self.actor(states).detach()).mean()
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def train_episode(self, episode_trajectory, episode_reward):
        """
        Episode-based update: Train the agent at the end of an episode.
        """
        # Store the entire trajectory in the replay buffer
        for transition in episode_trajectory:
            state, action, reward, next_state, done = transition
            self.replay_buffer.add(state, action, reward, next_state, done)

        # Perform multiple training steps (often equal to the episode length)
        for _ in range(len(episode_trajectory)):
            if self.replay_buffer.size() < BATCH_SIZE:
                return

            # Sample a batch from the replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                BATCH_SIZE
            )

            # Critic loss
            next_actions = self.actor_target(next_states)
            target_q = rewards + GAMMA * self.critic_target(
                next_states, next_actions
            ) * (1 - dones)
            current_q = self.critic(states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q.detach())

            # Actor loss
            actor_loss = -self.critic(states, self.actor(states).detach()).mean()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

    def soft_update(self, net, target_net):
        """
        Soft update of the target network using TAU.
        """
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


class BilliardEnv:
    def __init__(self):
        self.table_width = 2.84  # Table dimensions (meters)
        self.table_height = 1.42
        self.series_length = 0
        self.current_step = 0
        self.episode_rewards = []

        # Ball Positions
        self.ball1_ini = (0.5275, 0.71)  # White
        self.ball2_ini = (0.71, 0.71)  # Yellow
        self.ball3_ini = (0.71, 2.13)  # Red

        # define the properties
        self.u_slide = 0.15
        self.u_roll = 0.005
        self.u_sp_prop = 10 * 2 / 5 / 9
        self.u_ballball = 0.05
        self.e_ballball = 0.95
        self.e_cushion = 0.9
        self.f_cushion = 0.15
        self.grav = 9.81

        self.mball = 0.210
        self.Rball = 61.5 / 1000 / 2

        cue_mass = 0.576
        cue_len = 1.47
        cue_tip_R = 0.022
        cue_tip_mass = 0.0000001

        # Build a table with default BILLIARD specs
        self.table = pt.Table.default(pt.TableType.BILLIARD)

        # create the cue
        cue_specs = pt.objects.CueSpecs(
            brand="Predator",
            M=cue_mass,
            length=cue_len,
            tip_radius=cue_tip_R,
            butt_radius=0.02,
            end_mass=cue_tip_mass,
        )
        self.cue = pt.Cue(cue_ball_id="white", specs=cue_specs)

        self.interp_v = interp1d(
            (-1.0, 1.0), (1.0, 7.0), kind="linear", fill_value="extrapolate"
        )
        self.interp_phi = interp1d(
            (-1.0, 1.0), (0, 360), kind="linear", fill_value="extrapolate"
        )
        self.interp_theta = interp1d(
            (-1.0, 1.0), (0.0, 90), kind="linear", fill_value="extrapolate"
        )
        self.interp_a = interp1d(
            (-1.0, 1.0), (-0.6, 0.6), kind="linear", fill_value="extrapolate"
        )
        self.interp_b = interp1d(
            (-1.0, 1.0), (-0.6, 0.6), kind="linear", fill_value="extrapolate"
        )

        # rewards
        point_distance_0 = (
            0,
            2 * self.Rball * 0.7,
            2 * self.Rball,
            2 * self.Rball * 0.3,
            0.2,
            3,
        )
        reward_0 = (10, 10, 5, -5, -10, -50)

        # Interpolation function
        self.reward_point_distance = interp1d(
            point_distance_0, reward_0, kind="linear", fill_value="extrapolate"
        )

        self.reset()

    def reset(self):
        # Randomize ball positions within valid regions
        # self.ball1 = np.random.uniform(
        #     [self.Rball, self.Rball], [self.table_width - self.Rball, self.table_height - self.Rball])
        # self.ball2 = np.random.uniform(
        #     [self.Rball, self.Rball], [self.table_width - self.Rball, self.table_height - self.Rball])
        # self.ball3 = np.random.uniform(
        #     [self.Rball, self.Rball], [self.table_width - self.Rball, self.table_height - self.Rball])

        # start from Position from init
        self.ball1 = self.ball1_ini
        self.ball2 = self.ball2_ini
        self.ball3 = self.ball3_ini

        self.series_length = 0
        self.current_step = 0
        self.episode_rewards = []

        state = self.prepare_new_shot(3.0, 50, 0.0, 0.0, 0.0)

        return state

    def prepare_new_shot(self, v, phi, theta, a, b):
        # Create balls in new positions
        wball = pt.Ball.create(
            "white",
            xy=self.ball1,
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
            xy=self.ball2,
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
            xy=self.ball3,
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

        # Wrap it up as a System
        self.cue.set_state(V0=v, phi=phi, theta=theta, a=a, b=b)
        self.system = pt.System(
            table=self.table, balls=(wball, yball, rball), cue=self.cue
        )

        state = self.get_state()
        return state

    def get_state(self):
        # Calculate distances and directions
        # all values must be normalized
        # ball positions relative to table center
        # distances scaled by table length
        # angles by 360deg
        #
        d_cue_to_2 = (
            np.linalg.norm(np.array(self.ball1) - np.array(self.ball2))
            / self.table_width
        )
        d_cue_to_3 = (
            np.linalg.norm(np.array(self.ball1) - np.array(self.ball3))
            / self.table_width
        )
        d_2_to_3 = (
            np.linalg.norm(np.array(self.ball2) - np.array(self.ball3))
            / self.table_width
        )

        # print('Ersin-Line 233: Check the angle whether it is in table coordinate system')
        phi_cue_to_2 = pt.aim.at_ball(self.system, "yellow", cut=0.0) / 360
        phi_cue_to_3 = pt.aim.at_ball(self.system, "red", cut=0.0) / 360
        phi_2_to_3 = (
            np.arctan2(self.ball3[1] - self.ball2[1], self.ball3[0] - self.ball2[0])
            * 180
            / np.pi
            / 360
        )

        d_cushions_to_1 = [
            (self.ball1[0] - self.Rball) / self.table_width,
            (self.table_width - self.ball1[1] - self.Rball) / self.table_width,
            (self.ball1[0] - self.Rball) / self.table_width,
            (self.table_height - self.ball1[0] - self.Rball) / self.table_width,
        ]
        d_cushions_to_2 = [
            (self.ball2[0] - self.Rball) / self.table_width,
            (self.table_width - self.ball2[1] - self.Rball) / self.table_width,
            (self.ball2[0] - self.Rball) / self.table_width,
            (self.table_height - self.ball2[0] - self.Rball) / self.table_width,
        ]
        d_cushions_to_3 = [
            (self.ball3[0] - self.Rball) / self.table_width,
            (self.table_width - self.ball3[1] - self.Rball),
            (self.ball3[0] - self.Rball) / self.table_width,
            (self.table_height - self.ball3[0] - self.Rball) / self.table_width,
        ]

        return np.array(
            [
                *self.ball1,
                *self.ball2,
                *self.ball3,
                d_cue_to_2,
                phi_cue_to_2,
                d_cue_to_3,
                phi_cue_to_3,
                d_2_to_3,
                phi_2_to_3,
                *d_cushions_to_1,
                *d_cushions_to_2,
                *d_cushions_to_3,
            ]
        )

    def denormalize_action(self, action):
        v_n, phi_n, theta_n, a_n, b_n = action

        # Denormalize actions
        v = self.interp_v(v_n)
        phi = self.interp_phi(phi_n)
        theta = self.interp_theta(theta_n)
        a = self.interp_a(a_n)
        b = self.interp_b(b_n)

        return v, phi, theta, a, b

    def step(self, action):
        v, phi, theta, a, b = self.denormalize_action(action)
        print(
            f"  v=:{np.round(v,2)}, phi={np.round(phi,2)}, theta={np.round(theta,1)}, a={np.round(a,2)}, b={np.round(b,2)}"
        )

        # Simulate shot using physics engine or model (implement this)

        self.prepare_new_shot(v, phi, theta, a, b)

        self.simulate_shot()

        reward = self.calculate_shot_reward()

        # Track rewards for episode-based updates
        self.episode_rewards.append(reward)

        # Update state and check if the episode is done
        next_state = self.get_state()

        if self.is_point:
            self.series_length += 1
        else:
            self.series_length = 0

        return next_state, reward, self.is_point

    def simulate_shot(self):
        system = self.system
        # run the phsics model
        pt.simulate(system, inplace=True)

        self.ball1 = system.balls["white"].state.rvw[0, :2]
        self.ball2 = system.balls["yellow"].state.rvw[0, :2]
        self.ball3 = system.balls["red"].state.rvw[0, :2]

        self.is_point = is_point(system)

        pass

    def calculate_shot_reward(self):
        # definition of the rewards
        # point distance: +10 .. -30
        # safe hit between (1.0 ball .. 0.25 ball) : +10
        # thin hit/miss between (0.25 ball .. -0.5 ball): +10 ... (-10)
        # big miss  (-0.5 ball .. 3m) : -10 .. -30
        # b1 cushion count +1 per cushion, max 3 points

        # Series    +5 per additional point

        point_distance, b1_cushion_count, isb2hit = self.eval_shot()

        # New points for interpolation
        reward_pd = self.reward_point_distance(point_distance[0])

        reward_cushion = max(b1_cushion_count, 3)
        reward_b2hit = isb2hit * 1

        reward = reward_pd + reward_cushion + reward_b2hit

        return reward

    # Shot analysis
    def eval_shot(self):
        # outputs:
        # is_point
        # is_hitb1b2
        # b1_cushions
        # point_distance

        shot = self.system

        def get_ball_order(shot):
            # identify the balls b1=cueball, b2=objectball, b3=targetball

            b1 = shot.cue.cue_ball_id

            # identy b2 and b3.
            # if b1 hits only one ball, b2 is the ball which is hit by b1, b3 is the remaining ball
            # Future task:
            # if b1 has no ball-ball event, b2 is the closest ball to b1 after 3 cushions, b3 is the remaining ball

            # get ball events of cue ball b1
            b1events = filter_ball(shot.events, b1)
            b1ballhits = filter_type(b1events, EventType.BALL_BALL)

            if b1ballhits != []:
                # b2 is the first ball which is touched by b1
                b2 = [color for color in b1ballhits[0].ids if color != b1][0]
                # remaining ball is b3
                b3 = [
                    color
                    for color in ("white", "yellow", "red")
                    if color not in (b1, b2)
                ][0]
                isb2hit = True
            else:
                # no ball contact, so we define b1 and b2
                # in future change it to the closest ball to the cueball after 3 cushions
                b2 = [color for color in ("white", "yellow") if color != b1][0]
                b3 = "red"
                isb2hit = False

            return [b1, b2, b3, isb2hit]

        def get_ball_events(shot):
            # collect all hits for each ball
            b1events = filter_ball(shot.events, b1)
            b1hit = filter_type(
                b1events, [EventType.BALL_BALL, EventType.BALL_LINEAR_CUSHION]
            )
            b2events = filter_ball(shot.events, b2)
            b2hit = filter_type(
                b2events, [EventType.BALL_BALL, EventType.BALL_LINEAR_CUSHION]
            )
            b3events = filter_ball(shot.events, b3)
            b3hit = filter_type(
                b3events, [EventType.BALL_BALL, EventType.BALL_LINEAR_CUSHION]
            )

            return [b1hit, b2hit, b3hit]

        def add_events_to_coords(shot):
            # Add events to the vectorized time series to have accurate positions for each event

            shotcont = pt.continuize(shot, dt=0.01, inplace=False)

            # Add events to the vectorized time series
            b1_obj = shotcont.balls[b1]
            b1_hist = b1_obj.history_cts
            rvw_b1, s_b1, t_b1 = b1_hist.vectorize()
            b1_coords = rvw_b1[:, 0, :2]

            b2_obj = shotcont.balls[b2]
            b2_hist = b2_obj.history_cts
            rvw_b2, s_b2, t_b2 = b2_hist.vectorize()
            b2_coords = rvw_b2[:, 0, :2]

            b3_obj = shotcont.balls[b3]
            b3_hist = b3_obj.history_cts
            rvw_b3, s_b3, t_b3 = b3_hist.vectorize()
            b3_coords = rvw_b3[:, 0, :2]

            all_ball_events = filter_type(
                shot.events, [EventType.BALL_BALL, EventType.BALL_LINEAR_CUSHION]
            )

            for event in all_ball_events:
                event_time = event.time
                # Find the index to insert the event time
                index = np.searchsorted(t_b1, event_time)

                # Insert event time into t_b1
                t_b1 = np.insert(t_b1, index, event_time)

                # find ball which was not involved in the event
                otherballs = tuple(set((b1, b2, b3)) - set(event.ids))

                # add positions to involved balls
                for id in event.ids:
                    if id == b1:
                        event_xy = event.get_ball(id, initial=True).xyz[0:2]
                        b1_coords = np.insert(b1_coords, index, event_xy, axis=0)
                    elif id == b2:
                        event_xy = event.get_ball(id, initial=True).xyz[0:2]
                        b2_coords = np.insert(b2_coords, index, event_xy, axis=0)
                    elif id == b3:
                        event_xy = event.get_ball(id, initial=True).xyz[0:2]
                        b3_coords = np.insert(b3_coords, index, event_xy, axis=0)

                for id in otherballs:
                    # linear interpolate the position of the ball which was not involved in the event
                    if id == b1:
                        interp = interp1d(
                            [t_b1[index - 1], t_b1[index]],
                            [b1_coords[index - 1], b1_coords[index]],
                            axis=0,
                            kind="linear",
                        )
                        xy = interp(event_time)
                        b1_coords = np.insert(b1_coords, index, xy, axis=0)
                    elif id == b2:
                        interp = interp1d(
                            [t_b1[index - 1], t_b1[index]],
                            [b2_coords[index - 1], b2_coords[index]],
                            axis=0,
                            kind="linear",
                        )
                        xy = interp(event_time)
                        b2_coords = np.insert(b2_coords, index, xy, axis=0)
                    elif id == b3:
                        interp = interp1d(
                            [t_b1[index - 1], t_b1[index]],
                            [b3_coords[index - 1], b3_coords[index]],
                            axis=0,
                            kind="linear",
                        )
                        xy = interp(event_time)
                        b3_coords = np.insert(b3_coords, index, xy, axis=0)

            return [t_b1, b1_coords, b2_coords, b3_coords]

        def ball_ball_distances():
            # Calculate ball to ball distance
            b1b2dist = np.sqrt(np.sum((b1_coords - b2_coords) ** 2, axis=1))
            b1b3dist = np.sqrt(np.sum((b1_coords - b3_coords) ** 2, axis=1))
            b2b3dist = np.sqrt(np.sum((b2_coords - b3_coords) ** 2, axis=1))

            return [b1b2dist, b1b3dist, b2b3dist]

        def eval_hit_fraction(shot, event):
            # calculate hit_fraction of given event
            # check if the event is a ball-ball event
            if event.event_type != EventType.BALL_BALL:
                print("Event is not a ball-ball event.")
                return None

            # Use ball_ball.ids to see which ball IDs are involved in the event
            ball1 = event.get_ball(event.ids[0], initial=True)
            ball2 = event.get_ball(event.ids[1], initial=True)

            center_to_center = pt.ptmath.unit_vector(ball2.xyz - ball1.xyz)
            direction = pt.ptmath.unit_vector(ball1.vel - ball2.vel)

            cut_angle_radians = np.arccos(np.dot(direction, center_to_center))
            # cut_angle_degrees = cut_angle_radians * 180 / np.pi
            hit_fraction = 1 - np.sin(cut_angle_radians)

            # print(f"{cut_angle_degrees=}", f"{hit_fraction=}")

            return hit_fraction

        def cushion_count(shot, bx):
            # count the cushion hits before b1 hits b3
            # get ball events of cue ball b1
            events = filter_ball(shot.events, bx)
            cushion_hits = filter_type(events, EventType.BALL_LINEAR_CUSHION)
            n = len(cushion_hits)

            return n

        def kisses(shot):
            return kisses

        def eval_point_distance(shot):
            # calculate point distance

            # calculate 3 closest distances to make a point
            # if the shot is a point, calculate the distance at the point of contact
            # if b1 hit b2 and b2 before hitting 3 cushions, set point_distance = 3.0

            # Initialize variables
            point_distance = (3.0, 3.0, 3.0)
            cushion_hit_count = 0  # Counter for ball_linear_cushion events
            check_time = None  # Variable to store the time when conditions are met
            b2_found = 0  # Flag for `b2` in agents
            hit_fraction = 0
            point_distance0 = 3.0
            point_time = -1.0
            Rball = shot.balls["white"].params.R

            # Iterate through events
            for event in b1hit:
                # Condition 1: Check if type is ball_linear_cushion
                if event.event_type == "ball_linear_cushion":
                    cushion_hit_count += 1  # Increment cushion hit counter
                    # Store the time of the last `ball_linear_cushion` event

                # Condition 2: Check if b2 exists in agents (excluding b1)
                if b2 in event.ids and b2_found is False:
                    b2_found = b2_found + 1

                # Check if the conditions are met
                if cushion_hit_count >= 3 and b2_found == 1 and check_time is None:
                    # We have met the requirements, store the time
                    check_time = event.time

                if cushion_hit_count >= 3 and b2_found >= 1 and b3 in event.ids:
                    point_time = event.time
                    hit_fraction = eval_hit_fraction(shot, event)
                    point_distance0 = hit_fraction * Rball
                    # print(point_distance0)
                    break

            if cushion_hit_count >= 3 and b2_found >= 1:
                point_distance = eval_point_distance_3c_nopoint(
                    check_time, point_distance0, point_time
                )

            # elif b2hit != [] and b3hit == []:
            #     # one ball was hit
            #     # print('One ball was hit')
            #     tmp = 0

            # elif b2hit == [] and b3hit == []:
            #     # no ball was hit
            #     # print('No ball was hit')
            #     tmp = 0

            return point_distance

        def eval_point_distance_3c_nopoint(check_time, point_distance0, point_time):
            # Calculate the point distance for a 3-cushion shot that is not a point
            # find 3 different minima (if available) of b1b3dist[tsel]
            point_distance = (3.0, 3.0, 3.0)
            tsel = t_b1 > check_time
            y = b1b3dist[tsel]
            t = t_b1[tsel]

            # Find local minima (relative minima) in the data
            minima_indices = argrelextrema(y, np.less)[0]

            # When there is a point, then distance is limited by ball diameter
            # Therefore replace the minima of the point
            # Use the point time if it is given
            if point_time > 0 and np.min(np.abs(t - point_time)) < 1.0e-6:
                point_index = np.argmin(np.abs(t - point_time))

                # now replace in y(minima_indices) the value y(point_index) = point_distance0
                y[point_index] = point_distance0

            # check if the distance was getting closer at the end, but not yet minimum
            if y[-2] >= y[-1]:
                # Add the last index to the minima_indices if we have negative slope at the end
                minima_indices = np.append(minima_indices, len(y) - 1)

            # Ensure the array has 3 elements
            while len(minima_indices) < 3:
                minima_indices = np.append(minima_indices, len(y) - 1)

            # Sort the minima by their values (y values) and get the 3 smallest
            sorted_minima_indices = minima_indices[np.argsort(y[minima_indices])][:3]
            point_distance = y[sorted_minima_indices]

            # # Plot distances
            # plt.figure(figsize=(8, 5))

            # plt.plot(t_b1[tsel], b1b2dist[tsel], linestyle='-', color='r', label='Distance')
            # plt.plot(t_b1[tsel], b1b3dist[tsel], linestyle='-', color='b', label='Distance')
            # plt.plot(t_b1[tsel], b2b3dist[tsel], linestyle='-', color='k', label='Distance')

            # plt.title('Distance Between Corresponding Points')
            # plt.xlabel('time in s')
            # plt.ylabel('Distance')
            # plt.grid(True)
            # plt.legend()
            # plt.show()

            return point_distance

        # START of evaluation
        # identify the balls cueball, objectball, targetball and store in ballorder
        (b1, b2, b3, isb2hit) = get_ball_order(shot)
        # print(f"{b1=}, {b2=}, {b3=}")

        (b1hit, b2hit, b3hit) = get_ball_events(shot)
        (t_b1, b1_coords, b2_coords, b3_coords) = add_events_to_coords(shot)

        (b1b2dist, b1b3dist, b2b3dist) = ball_ball_distances()
        # calculate point distances
        b1_cushion_count = cushion_count(shot, b1)
        point_distance = eval_point_distance(shot)
        print(f"  {point_distance[0]=}")

        return point_distance, b1_cushion_count, isb2hit


env = BilliardEnv()
agent = DDPGAgent(STATE_DIM, ACTION_DIM)

num_episodes = 1000
max_steps = 200  # Max steps per episode
highscore = 0
highscore_reward = 10
rewardrecord = 0
rewardrecord_reward = 10

for episode in range(num_episodes):
    print(f"Episode {episode}, ....")
    state = env.reset()
    episode_trajectory = []  # Store transitions for episode-based updates

    total_reward = 0
    score = 0

    for step in range(max_steps):
        # Select an action
        action = agent.select_action(state)

        # Step in the environment
        next_state, reward, ispoint = env.step(action)
        done = not ispoint

        print(f"  Step {step}, Step Reward: {reward}, Total reward:{total_reward}")

        # Open up the shot in the GUI
        # pt.show(env.system)

        # Store the transition for episode-based updates
        episode_trajectory.append((state, action, reward, next_state, done))

        # Stepwise Update (immediate)
        agent.train_step(state, action, reward, next_state, done)

        # Update state and accumulate episode reward
        state = next_state

        if ispoint:
            # scored, episode to continue
            score += 1
            total_reward += reward

        else:
            # missed, episode to end
            if score > highscore:
                highscore = score
                total_reward += highscore_reward
            if total_reward > rewardrecord:
                rewardrecord_reward = total_reward
                total_reward += rewardrecord_reward

            break

    # Episode-Based Update (end of episode)
    if score > 0:
        print(f"Episode Reward: {total_reward}")
        agent.train_episode(episode_trajectory, total_reward)
        episode_trajectory = []

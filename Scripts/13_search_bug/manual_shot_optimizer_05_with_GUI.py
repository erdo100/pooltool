import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.interpolate import interp1d
import pooltool as pt
from tkinter import Tk, Scale, HORIZONTAL, Label, Button, Frame, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pooltool.ruleset.three_cushion import is_point
from billiardenv import BilliardEnv
from helper_funcs import read_shotfile, interpolate_simulated_to_actual, loss_func, save_parameters, load_parameters, save_system
from slider_definitions import create_ballball_sliders, create_physics_sliders, create_shot_sliders


def plot_settings():
    plt.clf()  # Clear the current figure
    plt.ion()  # Enable interactive mode

    # Create the axes
    global ax1, ax2, ax3, ax4, ax5, ax6, ax7
    ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])  # Left axes for trajectories
    ax2 = fig.add_axes([0.4, 0.6, 0.3, 0.3])   # Middle top axes for white ball velocities
    ax3 = fig.add_axes([0.4, 0.35, 0.3, 0.3])  # Middle middle axes for yellow ball velocities
    ax4 = fig.add_axes([0.4, 0.1, 0.3, 0.3])   # Middle bottom axes for red ball velocities
    ax5 = fig.add_axes([0.75, 0.6, 0.2, 0.3])  # Right top axes for white ball angular velocities
    ax6 = fig.add_axes([0.75, 0.35, 0.2, 0.3]) # Right middle axes for yellow ball angular velocities
    ax7 = fig.add_axes([0.75, 0.1, 0.2, 0.3])  # Right bottom axes for red ball angular velocities

    # Set background color and grid for ax1
    ax1.set_xlim(0, 1.42)
    ax1.set_ylim(0, 2.84)
    grid_size = 2.84 / 8
    ax1.set_xticks(np.arange(0, 1.42 + grid_size, grid_size))
    ax1.set_yticks(np.arange(0, 2.84 + grid_size, grid_size))
    ax1.grid(True)

    # Set background color and grid for other axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_facecolor("lightgray")
        ax.grid(True)


def get_ball_ids(shot_actual):
    color_mapping = {1: "white", 2: "yellow", 3: "red"}

    ball_ids = {}
    ball_cols = {}
    # Get the second entry (based on insertion order)
    ball_ids[0] = list(shot_actual["balls"].keys())[0] # This is the cue ball
    ball_cols[0] = color_mapping.get(ball_ids[0])  # Assign color
    ball_ids[1] = list(shot_actual["balls"].keys())[1]
    ball_cols[1] = color_mapping.get(ball_ids[1])  # Assign color
    ball_ids[2] = list(shot_actual["balls"].keys())[2]
    ball_cols[2] = color_mapping.get(ball_ids[2])  # Assign color
    
    return ball_ids, ball_cols

def get_ball_positions(shot_actual):

    ball_ids, ball_cols = get_ball_ids(shot_actual)

    balls_xy_ini = {}
    balls_xy_ini[0] = (shot_actual["balls"][1]["x"][0], shot_actual["balls"][1]["y"][0])
    balls_xy_ini[1] = (shot_actual["balls"][2]["x"][0], shot_actual["balls"][2]["y"][0])
    balls_xy_ini[2] = (shot_actual["balls"][3]["x"][0], shot_actual["balls"][3]["y"][0])

    return balls_xy_ini, ball_ids, ball_cols

def plot_initial_positions(ball_xy_ini):
    for ball_col, ball_xy in ball_xy_ini.items():
        circle = plt.Circle(ball_xy, 0.0615 / 2, color=ball_col, fill=True)
        plt.gca().add_patch(circle)

def plot_current_shot(tsim, white_rwv, yellow_rwv, red_rwv):
    colortable = ["white", "yellow", "red"]

    # Plot initial positions and trajectories in the left axes
    for i, rwv in enumerate([white_rwv, yellow_rwv, red_rwv]):
        ax1.add_patch(plt.Circle((rwv[0, 0, 0], rwv[0, 0, 1]), 0.0615 / 2, color=colortable[i], fill=True))
        ax1.plot(rwv[:, 0, 0], rwv[:, 0, 1], color=colortable[i], linewidth=1)

    # Plot x velocities in the top middle axes
    for i, rwv in enumerate([white_rwv, yellow_rwv, red_rwv]):
        ax2.plot(tsim, rwv[:, 1, 0], color=colortable[i], linewidth=1, label=f'{colortable[i]} Vx')
    ax2.legend()
    ax2.relim()
    ax2.autoscale_view()

    # Plot y velocities in the middle middle axes
    for i, rwv in enumerate([white_rwv, yellow_rwv, red_rwv]):
        ax3.plot(tsim, rwv[:, 1, 1], color=colortable[i], linewidth=1, label=f'{colortable[i]} Vy')
    ax3.legend()
    ax3.relim()
    ax3.autoscale_view()

    # Plot abs velocities in the bottom middle axes
    for i, rwv in enumerate([white_rwv, yellow_rwv, red_rwv]):
        vabs = np.sqrt(np.diff(rwv[:, 0, 0])**2 + np.diff(rwv[:, 0, 1])**2) / np.diff(tsim)
        vabs = np.insert(vabs, 0, vabs[0])
        ax4.plot(tsim, vabs, color=colortable[i], linewidth=1, label=f'{colortable[i]} Vz')
    ax4.legend()
    ax4.relim()
    ax4.autoscale_view()

    # Plot x avelocities in the top middle axes
    for i, rwv in enumerate([white_rwv, yellow_rwv, red_rwv]):
        ax5.plot(tsim, rwv[:, 2, 0], color=colortable[i], linewidth=1, label=f'{colortable[i]} Vx')
    ax5.legend()
    ax5.relim()
    ax5.autoscale_view()

    # Plot y avelocities in the middle middle axes
    for i, rwv in enumerate([white_rwv, yellow_rwv, red_rwv]):
        ax6.plot(tsim, rwv[:, 2, 1], color=colortable[i], linewidth=1, label=f'{colortable[i]} Vy')
    ax6.legend()
    ax6.relim()
    ax6.autoscale_view()

    # Plot z avelocities in the bottom middle axes
    for i, rwv in enumerate([white_rwv, yellow_rwv, red_rwv]):
        ax7.plot(tsim, rwv[:, 2, 2], color=colortable[i], linewidth=1, label=f'{colortable[i]} Vz')
    ax7.legend()
    ax7.relim()
    ax7.autoscale_view()

    plt.draw()  # Update the plot
    plt.pause(10.001)


# Read the shots file
shots_actual = read_shotfile()

var0={}
var0[0] = shot_index = 0
# var0[1] = a0=(-0.181,-0.179)
# var0[2] = b0=(0.149, 0.151)
# var0[3] = phi0=(128.1,128.3)
# var0[4] = v0=(2.9,3.1)
# var0[5] = theta0=(5.0, 5,2)
# var0[6] = a_ballball0=(0.019,0.021)
# var0[7] = b_ballball0=(0.139, 0.141)
# var0[8] = c_ballball0=(4.09, 4.11)
# var0[9] = u_slide0=(0.2, 0.202)
# var0[10] = u_roll0=(0.004, 0.006)
# var0[11] = u_sp_prop0=(0.589, 0.591)
# var0[12] = e_ballball0=(0.955, 0.956)
# var0[13] = e_cushion0=(0.689, 0.691)
# var0[14] = f_cushion0=(0.162, 0.164) 
var0[1] = a0=(-0.5, 0.5)
var0[2] = b0=(-0.5, 0.5)
var0[3] = phi0=(-130,-70)
var0[4] = v0=(5,10)
var0[5] = theta0=(0,50)
var0[6] = a_ballball0=(0,0.1)
var0[7] = b_ballball0=(0,1)
var0[8] = c_ballball0=(1,3)
var0[9] = u_slide0=(0.1,0.5)
var0[10] = u_roll0=(0.001,0.1)
var0[11] = u_sp_prop0=(0.001,5)
var0[12] = e_ballball0=(0.7,1)
var0[13] = e_cushion0=(0.5,0.95)
var0[14] = f_cushion0=(0.05,0.3) 

# Select the current shot
shot_actual = shots_actual[shot_index]

var={}
counter_ball_outside = 0
counter_ball_jumped = 0

def is_ball_ouside(ball_rwv):
    tablexlim = [0.0, 1.42]
    tableylim = [0.0, 2.84]
    ballr = 0.0615 / 2
    check = False
    # Check if the ball is outside the table
    # find all points are smaller than x is smaller than tablexlim[0]+ballr or bigger than tablexlim[1]-ballr
    # or smaller than y is smaller than tableylim[0]+ballr or bigger than tableylim[1]-ballr
    if np.any(ball_rwv[:, 0, 0] < tablexlim[0] + ballr):
        check = True
    if np.any(ball_rwv[:, 0, 0] > tablexlim[1] - ballr):
        check = True
    if np.any(ball_rwv[:, 0, 1] < tableylim[0] + ballr):
        check = True
    if np.any(ball_rwv[:, 0, 1] > tableylim[1] - ballr):
        check = True
    if check:
        return True
    return check

def is_ball_jumped(rwv):
    # Check if the ball has jumped
    # find all ball speed above 10 m/s
    check = False
    vabs = np.sqrt(np.diff(rwv[:, 0, 0])**2 + np.diff(rwv[:, 0, 1])**2) / np.diff(tsim)
    if np.max(vabs) > 15:
        print(f"Ball jumped with speed {np.max(vabs)} m/s")
        check = True
    if check:
        return True
    return check

    
# Create the figure, width 2, height 1
fig = plt.figure(figsize=(14.2, 7.1))  # Set the figure size to maintain a 1:2 aspect ratio

for runi in range(1,10000):
    for i in range(1,15):
        var[i] = random.uniform(var0[i][0],var0[i][1])

    # position of balls
    ball_xy_ini, ball_ids, ball_cols = get_ball_positions(shot_actual)
    
    # Create billiard environment
    # shot_env = BilliardEnv(u_slide, u_roll, u_sp_prop, e_ballball, e_cushion, f_cushion)
    shot_env = BilliardEnv(var[9], var[10], var[11], var[12], var[13], var[14])

    # Prepare and simulate shot with updated parameters
    # shot_env.prepare_new_shot(ball_cols, ball_xy_ini, a, b, phi, v, theta)
    shot_env.prepare_new_shot(ball_cols, ball_xy_ini, var[1], var[2], var[3], var[4], var[5])

    # point, result, tsim, system = shot_env.simulate_shot(a_ballball, b_ballball, c_ballball)
    system = shot_env.simulate_shot(var[6], var[7], var[8])

    tsim, white_rwv, yellow_rwv, red_rwv = shot_env.get_ball_routes()

    # check ball is not outside table
    if is_ball_ouside(white_rwv) or is_ball_ouside(yellow_rwv) or is_ball_ouside(red_rwv):
        counter_ball_outside += 1
        # Plot the shots
        # plot_settings()
        # plot_current_shot(tsim, white_rwv, yellow_rwv, red_rwv)
        print(f"run: {runi} total outside the table: {counter_ball_outside}, times a ball jumped: {counter_ball_jumped}")
        
    # check whether a ball has jumped
    if is_ball_jumped(white_rwv) or is_ball_jumped(yellow_rwv) or is_ball_jumped(red_rwv):
        counter_ball_jumped += 1
        # Plot the shots
        # plot_settings()
        # plot_current_shot(tsim, white_rwv, yellow_rwv, red_rwv)
        print(f"run: {runi} total outside the table: {counter_ball_outside}, times a ball jumped: {counter_ball_jumped}")
    # plot_settings()
    # plot_current_shot(tsim, white_rwv, yellow_rwv, red_rwv)
    print(f"run: {runi} total outside the table: {counter_ball_outside}, times a ball jumped: {counter_ball_jumped}")

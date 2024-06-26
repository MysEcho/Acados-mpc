import numpy as np
from dataclasses import dataclass


# Problem setup
@dataclass
class Problem:
    # workspace
    ws_x = [-10.0, 10.0]  # m
    ws_y = [-6.0, 6.0]  # m
    # robot control bound
    robot_maxVel = 1.0  # m/s
    robot_maxOmega = np.deg2rad(15.0)  # rad/s
    # robot size, start and goal positions
    robot_size = [0.3, 0.3]  # ellipse, m
    robot_pos_start = [-8.0, 0.0]  # m
    robot_theta_start = [np.deg2rad(0.0)]  # rad
    robot_pos_goal = [8.0, 0.0]  # m
    # obstacle size, position,
    obs_size = [1.0, 1.0]  # m
    obs_pos = [-1.0, 0.2]  # m
    # MPC settings
    dt = 0.1  # sampling time, s
    N = 40  # horizon length
    T = N * dt  # horizon time
    nx = 3  # state dimension
    nu = 2  # control dimension
    nparam = 13  # parameter dimension
    # MPC cost terms weights
    w_pos = 5.0
    w_input = 0.1
    w_coll = 1e-3


# Vector index
@dataclass
class Index:
    # state vector, x = [px, py, theta]
    x_pos = np.s_[0:2]  # 0, 1
    x_theta = np.s_[2:3]  # 2
    # control vector, u = [vel, omega]
    u_vel = np.s_[0:1]  # 0
    u_omega = np.s_[1:2]  # 1
    # param vector
    p_robot_pos_start = np.s_[0:2]  # 0, 1
    p_robot_pos_goal = np.s_[2:4]  # 2, 3
    p_robot_size = np.s_[4:6]  # 4, 5
    p_obs_pos = np.s_[6:8]  # 6, 7
    p_obs_size = np.s_[8:10]  # 8, 9
    p_mpc_weights = np.s_[10:13]  # 10, 11, 12
    p_mpc_weights_w_pos = np.s_[0]  # 0 in mpc_weights
    p_mpc_weights_w_input = np.s_[1]  # 1 in mpc_weights
    p_mpc_weights_w_coll = np.s_[2]  # 2 in mpc_weights

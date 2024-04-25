import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from time import perf_counter
from acados_template import AcadosOcp, AcadosOcpSolver
from setup_mpc import Problem, Index
from model import mpc_model
from utils import RK2
from dynamics import continuous_dynamics


def acados_mpc():
    # Setting up MPC and indexing
    pr = Problem()
    index = Index()

    # Acados MPC model
    model_ac = mpc_model(pr, index)

    # AcadosOcp object
    ocp = AcadosOcp()
    ocp.model = model_ac

    # Set ocp dimensions
    ocp.dims.N = pr.N

    # cost type
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # initial constraint
    ocp.constraints.x0 = np.zeros(pr.nx)

    # Set state bound
    ocp.constraints.lbx = np.array([pr.ws_x[0], pr.ws_y[0], -np.pi])
    ocp.constraints.ubx = np.array([pr.ws_x[1], pr.ws_y[1], np.pi])
    ocp.constraints.idxbx = np.array(range(pr.nx))

    # Set control input bound
    ocp.constraints.lbu = np.array([-pr.robot_maxVel, -pr.robot_maxOmega])
    ocp.constraints.ubu = np.array([pr.robot_maxVel, pr.robot_maxOmega])
    ocp.constraints.idxbu = np.array(range(pr.nu))

    # Path constraint bounds
    ocp.constraints.lh = np.array([0.0])
    ocp.constraints.uh = np.array([100.0])
    # ocp.constraints.lh_e = np.array([0.0])
    # ocp.constraints.uh_e = np.array([100.0])

    # Parameters
    ocp.parameter_values = np.zeros((pr.nparam,))

    # Solver options

    # horizon
    ocp.solver_options.tf = pr.T

    # integrator option
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 1

    # nlp solver options
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.nlp_solver_tol_eq = 1e-3
    ocp.solver_options.nlp_solver_tol_ineq = 1e-3
    ocp.solver_options.nlp_solver_tol_comp = 1e-3
    ocp.solver_options.nlp_solver_tol_stat = 1e-3

    # hession approximation
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    # qp solver options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.qp_solver_cond_N = 5
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.qp_solver_warm_start = 1

    # code generation options
    ocp.code_export_directory = "./mpc_c_generated_code"
    # print options
    # ocp.solver_options.print_level = 0

    # Solver
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    # Plot
    plt.ion()
    fig_main, ax_main = plt.subplots()
    ax_main.grid(visible=True, ls="-.")
    ax_main.set_aspect("equal")
    ax_main.set_xlim(pr.ws_x)
    ax_main.set_ylim(pr.ws_y)
    ax_main.set_xlabel("x [m]")
    ax_main.set_ylabel("y [m]")

    # plot obejects

    # obstalce
    obs_list = []

    # Elliptical Obstacle
    obs_pos_ellipse = mpatches.Ellipse(
        pr.obs_pos,
        2 * pr.obs_size[0],
        2 * pr.obs_size[1],
        angle=0,
        fc=(0, 1, 0, 0.7),
        ec=(0, 1, 0, 0.2),
    )
    obs_list.append(obs_pos_ellipse)
    # Square Obstacle
    rect_pos = [3.0, -1.4]
    obs_pos_rectangle = mpatches.Rectangle(
        rect_pos,
        2 * pr.obs_size[1],
        2 * pr.obs_size[0],
        fc=(0, 1, 0, 0.7),
        ec=(0, 1, 0, 0.2),
    )
    # obs_list.append(obs_pos_rectangle)

    for obs in obs_list:
        ax_main.add_artist(obs)
    # robot current pos
    robot_pos_ellipse = mpatches.Ellipse(
        pr.robot_pos_start,
        2 * pr.robot_size[0],
        2 * pr.robot_size[1],
        angle=np.rad2deg(pr.robot_theta_start[0]),
        fc=(1, 0, 0, 0.8),
        ec=(1, 0, 0, 0.8),
    )
    fig_robot_pos = ax_main.add_artist(robot_pos_ellipse)
    # robot goal location
    fig_robot_goal = ax_main.plot(
        pr.robot_pos_goal[0], pr.robot_pos_goal[1], marker="d", mec="r", mfc="r", ms=10
    )
    # robot planner path
    fig_robot_mpc_path = ax_main.plot(
        pr.robot_pos_start[0], pr.robot_pos_start[1], c="c", ls="-", lw=2.0
    )
    plt.draw()

    # Main loop

    # flags
    mpc_feasible = False
    n_loop = 0  # no. of loops performed
    max_n_loop = 1000  # max no. of loops
    robot_state_current = (
        pr.robot_pos_start + pr.robot_theta_start
    )  # list, [px, py, theta]
    robot_control_current = list(np.zeros(pr.nu))
    mpc_x_plan = np.tile(np.array(robot_state_current).reshape((-1, 1)), (1, pr.N))
    mpc_u_plan = np.zeros((pr.nu, pr.N))

    # Rollout Time
    time_list = []

    # loop
    # Taking around 260 iterations to reach goal for now
    while n_loop <= 260:
        n_loop += 1
        start_time = perf_counter()
        # Set initial condition
        acados_ocp_solver.constraints_set(0, "lbx", np.array(robot_state_current))
        acados_ocp_solver.constraints_set(0, "ubx", np.array(robot_state_current))
        # Set real time paraterms
        for iStage in range(0, pr.N):
            param_this_stage = np.zeros((pr.nparam,))
            param_this_stage[index.p_robot_pos_start] = np.array(
                robot_state_current[index.x_pos]
            )
            param_this_stage[index.p_robot_pos_goal] = np.array(pr.robot_pos_goal)
            param_this_stage[index.p_robot_size] = np.array(pr.robot_size)
            param_this_stage[index.p_obs_pos] = np.array(pr.obs_pos)
            param_this_stage[index.p_obs_size] = np.array(pr.obs_size)
            param_this_stage[index.p_mpc_weights] = np.array(
                [0.2 * pr.w_pos, pr.w_input, pr.w_coll]
            )
            if iStage == pr.N - 1:
                param_this_stage[index.p_mpc_weights] = np.array(
                    [pr.w_pos, pr.w_input, pr.w_coll]
                )
            acados_ocp_solver.set(iStage, "p", param_this_stage)
        # Set initial guess
        if mpc_feasible:  # If MPC feasible
            x_traj_init = np.concatenate(
                (mpc_x_plan[:, 1:], mpc_x_plan[:, -1:]), axis=1
            )
            u_traj_init = np.concatenate(
                (mpc_u_plan[:, 1:], mpc_u_plan[:, -1:]), axis=1
            )
        else:  # If MPC infeasible
            x_traj_init = np.tile(
                np.array(robot_state_current).reshape((-1, 1)), (1, pr.N)
            )
            u_traj_init = np.zeros((pr.nu, pr.N))
        for iStage in range(0, pr.N):
            acados_ocp_solver.set(iStage, "x", x_traj_init[:, iStage])
            acados_ocp_solver.set(iStage, "u", u_traj_init[:, iStage])

        # Call solver
        status = acados_ocp_solver.solve()
        stop_time = perf_counter()
        time_list.append(stop_time - start_time)
        print("Time Elapsed:", stop_time - start_time)

        # Obtain solution
        for iStage in range(0, pr.N):
            mpc_x_plan[:, iStage] = acados_ocp_solver.get(iStage, "x")
            mpc_u_plan[:, iStage] = acados_ocp_solver.get(iStage, "u")
        if status != 0:  # infeasible
            print(
                "acados returned status {} in closed loop iteration {}.".format(
                    status, n_loop
                )
            )
            mpc_feasible = False
            robot_control_current = list(0.1 * mpc_u_plan[:, 0])
        else:  # feasible
            mpc_feasible = True
            robot_control_current = list(mpc_u_plan[:, 0])

        # Executing the control input
        # in simulation via RungeKutta
        robot_state_next = RK2(
            robot_state_current,
            robot_control_current,
            continuous_dynamics,
            pr.dt,
            [],
        )
        # Update system
        robot_state_current = robot_state_next

        # Update visualization
        fig_robot_pos.set_center(robot_state_current[index.x_pos])
        fig_robot_pos.set_angle(np.rad2deg(robot_state_current[index.x_theta]))
        fig_robot_mpc_path[0].set_data(mpc_x_plan[index.x_pos, 1:])
        fig_main.canvas.draw()
        fig_main.canvas.flush_events()
        time.sleep(0.01)

    time_array = np.array(time_list)
    print(
        f"Average rollout time: {time_array[1:].mean() * 1000:.2f} ms"
    )  # Single static obstacle gives avg rollout time of 21.15 ms
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    acados_mpc()

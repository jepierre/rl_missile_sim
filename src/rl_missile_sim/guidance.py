import numpy as np
import scipy.linalg


def lqr_gain(A, B, Q, R):
    """
    Compute optimal LQR gain matrix by solving continuous-time ARE.
    """
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


def lqr_guidance(missile_pos, missile_vel, target_pos, target_vel, Q=None, R=None):
    """
    Linear Quadratic Regulator (LQR) terminal guidance law.
    Models relative kinematics as a 2D double-integrator and projects
    commanded acceleration onto the normal to the missile velocity vector.
    """
    rel_pos = missile_pos - target_pos
    rel_vel = missile_vel - target_vel

    x = np.array([rel_pos[0], rel_vel[0], rel_pos[1], rel_vel[1]], dtype=np.float64)

    # 2D double-integrator system matrices
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ], dtype=np.float64)
    B = np.array([
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 1],
    ], dtype=np.float64)

    if Q is None:
        Q = np.diag([10.0, 1.0, 10.0, 1.0])
    if R is None:
        R = np.diag([1.0, 1.0])

    K = lqr_gain(A, B, Q, R)
    u_vec = -K @ x

    # Project acceleration onto direction normal to missile velocity
    missile_speed = np.linalg.norm(missile_vel)
    if missile_speed > 1e-6:
        vel_unit = missile_vel / missile_speed
        normal_unit = np.array([-vel_unit[1], vel_unit[0]])
        normal_acc = np.dot(u_vec, normal_unit)
    else:
        normal_acc = 0.0

    return float(normal_acc)


def pronav_policy(obs, nav_constant=4.0):
    """
    Pure Proportional Navigation guidance law.
    Returns scalar normal acceleration command a_n = N' * V_M * lambda_dot.

    Parameters:
        obs: Observation vector from MissileSim environment.
             obs[12]: pursuer speed V_M
             obs[16]: line-of-sight angular rate lambda_dot
        nav_constant: Effective navigation ratio N' (typically 3 to 5).
    """
    return float(nav_constant * obs[12] * obs[16])


def optimal_guidance_policy(obs, min_tgo=0.1):
    """
    Optimal guidance law (Zero Effort Miss / Zero Effort Velocity).
    Computes normal acceleration command to null predicted miss distance.

    Parameters:
        obs: Observation vector from MissileSim environment.
             obs[0:2]: missile position [N, E]
             obs[3:5]: missile velocity [Nv, Ev]
             obs[6:8]: target position [N, E]
             obs[9:11]: target velocity [Nv, Ev]
        min_tgo: Minimum time-to-go lower bound to prevent division by zero.
    """
    missile_pos = obs[0:2]
    missile_vel = obs[3:5]
    target_pos = obs[6:8]
    target_vel = obs[9:11]

    pos_error = target_pos - missile_pos
    vel_error = target_vel - missile_vel
    pos_error_norm = np.linalg.norm(pos_error)

    if pos_error_norm < 1e-6:
        return 0.0

    closing_speed = np.dot(-pos_error, vel_error) / pos_error_norm
    tgo = pos_error_norm / max(closing_speed, 1e-2)
    tgo = max(tgo, min_tgo)

    acc = (6.0 / (tgo**2)) * pos_error + (4.0 / tgo) * vel_error

    missile_speed = np.linalg.norm(missile_vel)
    if missile_speed > 1e-6:
        vel_unit = missile_vel / missile_speed
        normal_unit = np.array([-vel_unit[1], vel_unit[0]])
        normal_acc = np.dot(acc, normal_unit)
    else:
        normal_acc = 0.0

    return float(normal_acc)
